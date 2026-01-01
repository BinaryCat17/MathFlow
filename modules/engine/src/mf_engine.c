#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Internal State Management ---

void mf_state_reset(mf_state* state, const mf_program* prog, mf_arena* arena) {
    if (!prog) return;
    
    state->register_count = prog->meta.tensor_count;
    state->registers = MF_ARENA_PUSH(arena, mf_tensor, state->register_count);
    state->ownership_flags = MF_ARENA_PUSH(arena, uint8_t, state->register_count);
    memset(state->ownership_flags, 0, state->register_count);

    for (u32 i = 0; i < state->register_count; ++i) {
        mf_tensor* t_prog = &prog->tensors[i];
        mf_tensor* t_reg = &state->registers[i];
        
        if (t_prog->buffer) {
            // Constant -> View into Program
            mf_tensor_view(t_reg, t_prog);
        } else {
            // Temps & Bindings
            t_reg->info = t_prog->info;
            
            // Check if it's an internal temporary that needs allocation
            bool is_external = false;
            if (prog->symbols) {
                for (u32 s = 0; s < prog->meta.symbol_count; ++s) {
                    if (prog->symbols[s].register_idx == i) {
                        is_external = true;
                        break;
                    }
                }
            }

            if (!is_external) {
                // If the shape is fully defined (not dynamic), allocate now
                bool is_static = true;
                for (int d = 0; d < t_reg->info.ndim; ++d) {
                    if (t_reg->info.shape[d] < 0) {
                        is_static = false;
                        break;
                    }
                }
                
                if (is_static) {
                    if (mf_tensor_alloc(t_reg, state->allocator, &t_reg->info)) {
                        state->ownership_flags[i] = 1;
                    }
                }
            }
        }
    }
}

static void mf_state_shutdown(mf_state* state) {
    if (!state->registers || !state->allocator) return;
    
    for (u32 i = 0; i < state->register_count; ++i) {
        if (state->ownership_flags && state->ownership_flags[i]) {
            mf_tensor* t = &state->registers[i];
            if (t->buffer) {
                mf_buffer_free(t->buffer); 
                state->allocator->free(state->allocator, t->buffer);
                t->buffer = NULL;
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

    engine->front_idx = 0;
    engine->back_idx = 1;

    return engine;
}

void mf_engine_destroy(mf_engine* engine) {
    if (!engine) return;
    MF_LOG_DEBUG("Destroying Engine...");
    mf_engine_reset(engine);
    if (engine->backend.shutdown) engine->backend.shutdown(engine->backend.state);
    
    if (engine->heap_buffer) {
        MF_LOG_DEBUG("Freeing Heap Buffer...");
        free(engine->heap_buffer);
    }
    if (engine->arena_buffer) {
        MF_LOG_DEBUG("Freeing Arena Buffer...");
        free(engine->arena_buffer);
    }
    MF_LOG_DEBUG("Freeing Engine Structure...");
    free(engine);
}

void mf_engine_reset(mf_engine* engine) {
    if (!engine) return;

    for (u32 i = 0; i < engine->kernel_count; ++i) {
        mf_state_shutdown(&engine->kernels[i].state);
    }
    for (u32 i = 0; i < engine->resource_count; ++i) {
        // Free resource buffers (from Heap)
        if (engine->resources[i].buffers[0]) mf_buffer_free(engine->resources[i].buffers[0]);
        if (engine->resources[i].buffers[1]) mf_buffer_free(engine->resources[i].buffers[1]);
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

void mf_engine_dispatch(mf_engine* engine) {
    if (!engine) return;

    MF_LOG_TRACE("Dispatching Pipeline frame %llu", (unsigned long long)engine->frame_index);
    
    u8 front = engine->front_idx;
    u8 back  = engine->back_idx;

    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        mf_kernel_inst* ker = &engine->kernels[k_idx];
        
        // 1. Pre-Execution Tasks: Auto-Resize
        for (u32 i = 0; i < ker->resize_task_count; ++i) {
            mf_resource_inst* res_in = &engine->resources[ker->resize_tasks[i].src_res_idx];
            mf_resource_inst* res_out = &engine->resources[ker->resize_tasks[i].dst_res_idx];
            
            if (!mf_tensor_same_shape(&res_in->desc, &res_out->desc)) {
                if (res_in->desc.buffer && res_in->desc.buffer->data) {
                    MF_LOG_TRACE("Auto-Resizing '%s' to match '%s'", res_out->name, res_in->name);
                    mf_engine_resize_resource(engine, res_out->name, res_in->desc.info.shape, res_in->desc.info.ndim);
                }
            }
        }

        const mf_tensor* kernel_domain = NULL;
        
        // 2. Linear Binding: Map Global Resources to Local Registers
        for (u32 b = 0; b < ker->binding_count; ++b) {
            mf_kernel_binding* bind = &ker->bindings[b];
            mf_resource_inst* res = &engine->resources[bind->global_res];
            mf_tensor* t = &ker->state.registers[bind->local_reg];

            // Zero-Copy buffer assignment
            t->buffer = (bind->flags & MF_SYMBOL_FLAG_OUTPUT) ? res->buffers[back] : res->buffers[front];
            t->byte_offset = 0;
            t->info = res->desc.info; // Sync metadata
            
            if (bind->flags & MF_SYMBOL_FLAG_OUTPUT && !kernel_domain) {
                kernel_domain = t;
            }
        }
        
        if (!kernel_domain) {
            MF_LOG_TRACE("  Skipping Kernel '%s': No output binding found to define domain.", ker->id);
            continue;
        }

        // 3. Execution (Frequency Loop)
        MF_LOG_TRACE("  Executing Kernel: %s", ker->id);

        for (u32 f = 0; f < ker->frequency; ++f) {
            if (engine->backend.dispatch) {
                engine->backend.dispatch(engine->backend.state, ker->program, &ker->state, kernel_domain);
                
                if (ker->state.error_code != 0) {
                    MF_LOG_ERROR("Kernel '%s' failed with error code %d.", ker->id, ker->state.error_code);
                    goto end_dispatch;
                }
            }
        }
    }
    
end_dispatch:
    engine->frame_index++;
    engine->front_idx = 1 - engine->front_idx;
    engine->back_idx  = 1 - engine->back_idx;
}

mf_tensor* mf_engine_map_resource(mf_engine* engine, const char* name) {
    if (!engine) return NULL;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (strcmp(engine->resources[i].name, name) == 0) {
            mf_resource_inst* res = &engine->resources[i];
            res->desc.buffer = res->buffers[engine->front_idx];
            res->desc.byte_offset = 0;
            return &res->desc;
        }
    }
    return NULL;
}

bool mf_engine_resize_resource(mf_engine* engine, const char* name, const int32_t* new_shape, uint8_t new_ndim) {
    if (!engine || !name) return false;
    
    u32 hash = mf_fnv1a_hash(name);
    int32_t res_idx = find_resource_idx(engine, hash);
    if (res_idx == -1) return false;

    mf_resource_inst* res = &engine->resources[res_idx];
    mf_allocator* alloc = (mf_allocator*)&engine->heap;
    
    mf_type_info new_info = res->desc.info;
    new_info.ndim = new_ndim;
    memcpy(new_info.shape, new_shape, sizeof(int32_t) * MF_MAX_DIMS);
    
    // Recalculate strides
    int32_t stride = 1;
    for (int k = new_ndim - 1; k >= 0; --k) {
        new_info.strides[k] = stride;
        stride *= (new_shape[k] > 0 ? new_shape[k] : 1);
    }
    
    size_t new_bytes = 1;
    for(int d=0; d<new_ndim; ++d) new_bytes *= (new_shape[d] > 0 ? new_shape[d] : 1);
    new_bytes *= mf_dtype_size(new_info.dtype);
    
    if (res->size_bytes != new_bytes) {
        mf_buffer_free(res->buffers[0]);
        mf_buffer_alloc(res->buffers[0], alloc, new_bytes);
        
        mf_buffer_free(res->buffers[1]);
        mf_buffer_alloc(res->buffers[1], alloc, new_bytes);
        
        res->size_bytes = new_bytes;
    }
    
    res->desc.info = new_info;
    return true;
}

mf_engine_error mf_engine_get_error(mf_engine* engine) {
    if (!engine) return MF_ENGINE_ERR_NONE;
    
    // Check for internal engine errors (if we add them later to engine struct)
    
    // Check all kernels for runtime errors
    for (u32 i = 0; i < engine->kernel_count; ++i) {
        if (engine->kernels[i].state.error_code != 0) {
            return MF_ENGINE_ERR_RUNTIME;
        }
    }
    
    return MF_ENGINE_ERR_NONE;
}

void mf_engine_iterate_resources(mf_engine* engine, mf_engine_resource_cb cb, void* user_data) {
    if (!engine || !cb) return;

    for (u32 i = 0; i < engine->resource_count; ++i) {
        mf_resource_inst* res = &engine->resources[i];
        res->desc.buffer = res->buffers[engine->front_idx];
        res->desc.byte_offset = 0;

        cb(res->name, &res->desc, user_data);
    }
}
