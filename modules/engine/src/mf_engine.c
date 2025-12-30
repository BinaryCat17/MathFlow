#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_log.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Internal State Management ---

static void mf_state_shutdown(mf_state* state) {
    if (!state->registers || !state->allocator) return;
    
    for (u32 i = 0; i < state->register_count; ++i) {
        mf_tensor* t = &state->registers[i];
        if (t->data && (t->flags & MF_TENSOR_OWNS_DATA)) {
            state->allocator->free(state->allocator, t->data);
            t->data = NULL;
            t->flags &= ~MF_TENSOR_OWNS_DATA;
        }
    }
    state->registers = NULL;
    state->register_count = 0;
}

static void mf_state_reset(mf_state* state, const mf_program* prog, mf_arena* arena) {
    if (!prog) return;
    
    state->register_count = prog->meta.tensor_count;
    state->registers = MF_ARENA_PUSH(arena, mf_tensor, state->register_count);

    for (u32 i = 0; i < state->register_count; ++i) {
        mf_tensor* dst = &state->registers[i];
        const mf_tensor* src = &prog->tensors[i];
        
        *dst = *src;
        dst->flags = 0; 
        
        if (src->data) {
            if (state->allocator) {
                size_t bytes = src->capacity_bytes;
                dst->data = state->allocator->alloc(state->allocator, bytes);
                if (dst->data) {
                    memcpy(dst->data, src->data, bytes);
                } else {
                    MF_LOG_ERROR("Failed to allocate %zu bytes for constant tensor %u.", bytes, i);
                }
                dst->capacity_bytes = bytes;
                dst->flags |= MF_TENSOR_OWNS_DATA | MF_TENSOR_DYNAMIC;
            } else {
                 dst->data = src->data; 
            }
        } else {
            if (state->allocator) {
                 size_t type_size = mf_dtype_size(src->dtype);
                 if (type_size == 0) type_size = 4;
                 size_t bytes = (src->size > 0) ? src->size * type_size : type_size;
                 
                 dst->data = state->allocator->alloc(state->allocator, bytes);
                 if (dst->data) memset(dst->data, 0, bytes);
                 else {
                     MF_LOG_ERROR("Failed to allocate %zu bytes for register %u.", bytes, i);
                 }
                 
                 dst->capacity_bytes = bytes;
                 dst->flags |= MF_TENSOR_OWNS_DATA | MF_TENSOR_DYNAMIC;
            } else {
                 dst->data = NULL;
                 dst->capacity_bytes = 0;
                 dst->flags |= MF_TENSOR_DYNAMIC;
            }
        }
    }
}

// --- Engine API ---

mf_engine* mf_engine_create(const mf_engine_desc* desc) {
    MF_LOG_INFO("Creating Engine...");
    mf_engine* engine = calloc(1, sizeof(mf_engine));
    if (!engine) {
        MF_LOG_FATAL("Failed to allocate memory for engine structure.");
        return NULL;
    }

    size_t arena_size = (desc && desc->arena_size > 0) ? desc->arena_size : MF_MB(8);
    size_t heap_size = (desc && desc->heap_size > 0) ? desc->heap_size : MF_MB(64);

    engine->arena_buffer = malloc(arena_size);
    if (!engine->arena_buffer) {
        MF_LOG_FATAL("Failed to allocate memory for engine arena (%zu bytes).", arena_size);
        free(engine);
        return NULL;
    }
    mf_arena_init(&engine->arena, engine->arena_buffer, arena_size);

    engine->heap_buffer = malloc(heap_size);
    if (!engine->heap_buffer) {
        MF_LOG_FATAL("Failed to allocate memory for engine heap (%zu bytes).", heap_size);
        free(engine->arena_buffer);
        free(engine);
        return NULL;
    }
    mf_heap_init(&engine->heap, engine->heap_buffer, heap_size);

    if (desc) {
        engine->backend = desc->backend;
    }

    return engine;
}

void mf_engine_destroy(mf_engine* engine) {
    if (!engine) return;
    mf_engine_reset(engine);
    if (engine->backend.shutdown) engine->backend.shutdown(engine->backend.state);
    if (engine->heap_buffer) free(engine->heap_buffer);
    if (engine->arena_buffer) free(engine->arena_buffer);
    free(engine);
}

void mf_engine_reset(mf_engine* engine) {
    if (!engine) return;

    for (u32 i = 0; i < engine->kernel_count; ++i) {
        mf_state_shutdown(&engine->kernels[i].state);
        free(engine->kernels[i].id);
    }
    for (u32 i = 0; i < engine->resource_count; ++i) {
        free(engine->resources[i].name);
    }

    mf_arena_reset(&engine->arena);
    
    if (engine->heap_buffer) {
        mf_heap_init(&engine->heap, engine->heap_buffer, engine->heap.size);
    }
    
    engine->kernel_count = 0;
    engine->resource_count = 0;
}

mf_arena* mf_engine_get_arena(mf_engine* engine) {
    if (!engine) return NULL;
    return &engine->arena;
}

void mf_engine_bind_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe, mf_program** programs) {
    if (!engine || !pipe) return;
    
    MF_LOG_INFO("Binding Pipeline: %u resources, %u kernels", pipe->resource_count, pipe->kernel_count);

    // 1. Allocate Global Resources
    engine->resource_count = pipe->resource_count;
    engine->resources = MF_ARENA_PUSH(&engine->arena, mf_resource_inst, engine->resource_count);
    
    mf_allocator* allocator = (mf_allocator*)&engine->heap;

    for (u32 i = 0; i < pipe->resource_count; ++i) {
        mf_pipeline_resource* res_desc = &pipe->resources[i];
        mf_resource_inst* res = &engine->resources[i];
        
        MF_LOG_INFO("  Resource[%u]: %s (%s, persistent=%d)", i, res_desc->name, res_desc->dtype == MF_DTYPE_F32 ? "F32" : "Other", res_desc->persistent);

        res->name = strdup(res_desc->name);
        res->persistent = res_desc->persistent;
        
        // Descriptor Prototype
        memset(&res->desc, 0, sizeof(mf_tensor));
        res->desc.dtype = res_desc->dtype;
        res->desc.ndim = res_desc->ndim;
        memcpy(res->desc.shape, res_desc->shape, sizeof(int32_t) * res_desc->ndim);
        res->desc.size = 1;
        for(int d=0; d<res->desc.ndim; ++d) res->desc.size *= res->desc.shape[d];
        
        size_t bytes = mf_tensor_size_bytes(&res->desc);
        res->size_bytes = bytes;
        
        res->buffer_a = allocator->alloc(allocator, bytes);
        if (res->buffer_a) memset(res->buffer_a, 0, bytes);
        
        if (res->persistent) {
            res->buffer_b = allocator->alloc(allocator, bytes);
            if (res->buffer_b) memset(res->buffer_b, 0, bytes);
        } else {
            res->buffer_b = NULL;
        }
    }

    // 2. Load Kernels
    engine->kernel_count = pipe->kernel_count;
    engine->kernels = MF_ARENA_PUSH(&engine->arena, mf_kernel_inst, engine->kernel_count);
    
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        mf_pipeline_kernel* ker_desc = &pipe->kernels[i];
        mf_kernel_inst* ker = &engine->kernels[i];
        
        MF_LOG_INFO("  Kernel[%u]: %s (freq=%u)", i, ker_desc->id, ker_desc->frequency);

        ker->id = strdup(ker_desc->id);
        ker->program = programs[i];
        ker->frequency = ker_desc->frequency;
        
        // Initialize Local State (Registers)
        ker->state.allocator = allocator;
        mf_state_reset(&ker->state, ker->program, &engine->arena);
        
        // 3. Setup Bindings
        ker->binding_count = ker_desc->binding_count;
        ker->bindings = MF_ARENA_PUSH(&engine->arena, mf_kernel_binding, ker->binding_count);
        
        for (u32 b = 0; b < ker_desc->binding_count; ++b) {
            mf_pipeline_binding* bind = &ker_desc->bindings[b];
            
            // Find Local Reg Index
            int32_t local_idx = -1;
            for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
                if (strcmp(ker->program->symbols[s].name, bind->kernel_port) == 0) {
                    local_idx = ker->program->symbols[s].register_idx;
                    break;
                }
            }
            
            // Find Global Resource Index
            int32_t global_idx = -1;
            for (u32 r = 0; r < engine->resource_count; ++r) {
                if (strcmp(engine->resources[r].name, bind->global_resource) == 0) {
                    global_idx = r;
                    break;
                }
            }
            
            if (local_idx != -1 && global_idx != -1) {
                ker->bindings[b].local_reg = (u16)local_idx;
                ker->bindings[b].global_res = (u16)global_idx;
            } else {
                MF_LOG_ERROR("Failed to bind %s -> %s in kernel %s", bind->kernel_port, bind->global_resource, ker->id);
            }
        }
    }
}

void mf_engine_dispatch(mf_engine* engine, const mf_tensor* domain) {
    if (!engine) return;

    MF_LOG_TRACE("Dispatching Pipeline frame %llu", (unsigned long long)engine->frame_index);
    
    bool is_even = (engine->frame_index % 2 == 0);

    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        mf_kernel_inst* ker = &engine->kernels[k_idx];
        MF_LOG_TRACE("  Executing Kernel: %s", ker->id);
        
        // Zero-Copy Bindings: Map Global Resources to Local Registers
        for (u32 b = 0; b < ker->binding_count; ++b) {
            u16 local_reg = ker->bindings[b].local_reg;
            u16 global_res = ker->bindings[b].global_res;
            
            if (global_res >= engine->resource_count) continue;
            
            mf_resource_inst* res = &engine->resources[global_res];
            mf_tensor* t = &ker->state.registers[local_reg];

            void* data_ptr = is_even ? res->buffer_a : res->buffer_b;
            if (!res->persistent) data_ptr = res->buffer_a;

            const char* port_name = NULL;
            if (ker->program->symbols) {
                for(u32 s=0; s<ker->program->meta.symbol_count; ++s) {
                    if (ker->program->symbols[s].register_idx == local_reg) {
                        port_name = ker->program->symbols[s].name;
                        break;
                    }
                }
            }

            if (port_name && strncmp(port_name, "out_", 4) == 0 && res->persistent) {
                // It is an output! Write to the "Next" buffer.
                data_ptr = is_even ? res->buffer_b : res->buffer_a;
            } else {
                // Input or non-persistent (shared scratchpad)
                data_ptr = is_even ? res->buffer_a : res->buffer_b;
                if (!res->persistent) data_ptr = res->buffer_a;
            }

            t->data = data_ptr;
            t->capacity_bytes = res->size_bytes;
            
            t->ndim = res->desc.ndim;
            t->size = res->desc.size;
            memcpy(t->shape, res->desc.shape, sizeof(int32_t) * MF_MAX_DIMS);
        }

        // Determine Execution Domain
        // Phase 20: Use explicit domain tensor if provided, or fallback to first bound output
        const mf_tensor* kernel_domain = domain;
        if (!kernel_domain && ker->binding_count > 0) {
             // Fallback: Use the first bound register as domain (heuristic)
             // Ideally we should use the explicit 'implicit domain' logic from roadmap
             kernel_domain = &ker->state.registers[ker->bindings[0].local_reg];
        }

        // Execute Kernel
        for (u32 f = 0; f < ker->frequency; ++f) {
            if (engine->backend.dispatch && kernel_domain) {
                engine->backend.dispatch(engine->backend.state, ker->program, &ker->state, kernel_domain);
            }
        }
    }
    
    engine->frame_index++;
}

mf_tensor* mf_engine_map_resource(mf_engine* engine, const char* name) {
    if (!engine) return NULL;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (strcmp(engine->resources[i].name, name) == 0) {
            mf_resource_inst* res = &engine->resources[i];
            
            // Host always reads from the "Result of Previous Frame"?
            // Or "Result of Current Frame" (which just finished)?
            // frame_index incremented at end of dispatch.
            // So frame_index is now Odd (if start 0). 0 was Even.
            // Frame 0 executed: Read A, Write B.
            // Now frame_index=1.
            // We want to see what was written. That is B.
            // Logic:
            // Prev Frame (Index-1) was Even (0). Writes went to B.
            // Current Frame (Index) is Odd (1). Reads come from B.
            // So Host should read B?
            
            // Let's look at dispatch logic again:
            // bool is_even = (engine->frame_index % 2 == 0);
            // If frame=0 (even): Write to B.
            // End of dispatch: frame=1.
            // Host wants to read result of frame 0. It is in B.
            // is_even (for frame 1) is False.
            // data_ptr = is_even ? res->buffer_a : res->buffer_b => buffer_b.
            
            // Checks out. Host reads from the "Input for next frame", which is "Output of last frame".
            
            bool is_even = (engine->frame_index % 2 == 0);
            res->desc.data = is_even ? res->buffer_a : res->buffer_b;
            if (!res->persistent) res->desc.data = res->buffer_a;
            
            return &res->desc;
        }
    }
    return NULL;
}

bool mf_engine_resize_resource(mf_engine* engine, const char* name, const int32_t* new_shape, uint8_t new_ndim) {
    if (!engine || !name) return false;
    
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (strcmp(engine->resources[i].name, name) == 0) {
            mf_resource_inst* res = &engine->resources[i];
            
            // Calculate new size
            size_t new_size = 1;
            for(int d=0; d<new_ndim; ++d) new_size *= new_shape[d];
            size_t new_bytes = new_size * mf_dtype_size(res->desc.dtype);
            
            // Reallocate Buffer A
            if (res->buffer_a) {
                // We use the engine's allocator (heap)
                res->buffer_a = ((mf_allocator*)&engine->heap)->realloc((mf_allocator*)&engine->heap, res->buffer_a, res->size_bytes, new_bytes);
            } else {
                res->buffer_a = ((mf_allocator*)&engine->heap)->alloc((mf_allocator*)&engine->heap, new_bytes);
            }
            
            // Reallocate Buffer B if persistent
            if (res->persistent) {
                if (res->buffer_b) {
                     res->buffer_b = ((mf_allocator*)&engine->heap)->realloc((mf_allocator*)&engine->heap, res->buffer_b, res->size_bytes, new_bytes);
                } else {
                     res->buffer_b = ((mf_allocator*)&engine->heap)->alloc((mf_allocator*)&engine->heap, new_bytes);
                }
            }
            
            // Update Descriptor
            res->size_bytes = new_bytes;
            res->desc.ndim = new_ndim;
            res->desc.size = new_size;
            res->desc.capacity_bytes = new_bytes;
            memcpy(res->desc.shape, new_shape, sizeof(int32_t) * MF_MAX_DIMS);
            
            // Update data pointer in descriptor immediately to avoid stale access
            bool is_even = (engine->frame_index % 2 == 0);
            res->desc.data = is_even ? res->buffer_a : res->buffer_b;
            if (!res->persistent) res->desc.data = res->buffer_a;

            return true;
        }
    }
    return false;
}

mf_engine_error mf_engine_get_error(mf_engine* engine) {
    if (!engine) return MF_ENGINE_ERR_NONE;
    // TODO: Aggregate errors from all kernels?
    // For now, return NONE.
    return MF_ENGINE_ERR_NONE;
}

void mf_engine_iterate_resources(mf_engine* engine, mf_engine_resource_cb cb, void* user_data) {
    if (!engine || !cb) return;

    for (u32 i = 0; i < engine->resource_count; ++i) {
        mf_resource_inst* res = &engine->resources[i];
        
        bool is_even = (engine->frame_index % 2 == 0);
        res->desc.data = is_even ? res->buffer_a : res->buffer_b;
        if (!res->persistent) res->desc.data = res->buffer_a;

        cb(res->name, &res->desc, user_data);
    }
}