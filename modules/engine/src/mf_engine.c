#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/isa/mf_exec_ctx.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Internal State Management ---

static void mf_state_shutdown(mf_state* state) {
    if (!state->registers || !state->allocator) return;
    
    for (u32 i = 0; i < state->register_count; ++i) {
        mf_tensor* t = &state->registers[i];
        
        // Only free buffers that are explicitly owned by this state
        if (state->ownership_flags && state->ownership_flags[i]) {
            if (t->buffer) {
                mf_buffer_free(t->buffer); 
                state->allocator->free(state->allocator, t->buffer);
                t->buffer = NULL;
            }
        }
    }
    state->registers = NULL;
    state->ownership_flags = NULL;
    state->register_count = 0;
}

static void mf_state_reset(mf_state* state, const mf_program* prog, mf_arena* arena) {
    if (!prog) return;
    
    state->register_count = prog->meta.tensor_count;
    state->registers = MF_ARENA_PUSH(arena, mf_tensor, state->register_count);
    state->ownership_flags = MF_ARENA_PUSH(arena, uint8_t, state->register_count);
    memset(state->ownership_flags, 0, state->register_count);

    for (u32 i = 0; i < state->register_count; ++i) {
        // Init temp registers
        // For constants, we view the program data.
        // For temps, we alloc.
        
        mf_tensor* t_prog = &prog->tensors[i];
        mf_tensor* t_reg = &state->registers[i];
        
        if (t_prog->buffer) {
            // Constant -> View
            mf_tensor_view(t_reg, t_prog);
        } else {
            // Check if this register is an external binding (Input/Output)
            bool is_external = false;
            if (prog->symbols) {
                for (u32 s = 0; s < prog->meta.symbol_count; ++s) {
                    if (prog->symbols[s].register_idx == i) {
                        is_external = true;
                        break;
                    }
                }
            }

            if (is_external) {
                // External registers are bound at dispatch time. No allocation needed.
                memset(t_reg, 0, sizeof(mf_tensor));
                t_reg->info = t_prog->info;
            } else {
                // Internal Temporary -> Alloc
                mf_type_info alloc_info = t_prog->info;
                bool is_dynamic = false;
                for(int k=0; k<alloc_info.ndim; ++k) {
                    if (alloc_info.shape[k] < 0) {
                        is_dynamic = true;
                        break;
                    }
                }

                if (is_dynamic) {
                    // Stay unallocated for now
                    memset(t_reg, 0, sizeof(mf_tensor));
                    t_reg->info = alloc_info;
                } else {
                    if (!mf_tensor_alloc(t_reg, state->allocator, &alloc_info)) {
                        MF_LOG_ERROR("Failed to allocate register %u during reset.", i);
                    } else {
                        state->ownership_flags[i] = 1; // Mark as owned
                    }
                }
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
        // Free resource buffers
        if (engine->resources[i].buffer_a) {
            mf_buffer_free(engine->resources[i].buffer_a);
            // struct is on arena, no need to free pointer
        }
        if (engine->resources[i].buffer_b) {
            mf_buffer_free(engine->resources[i].buffer_b);
        }
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
        
        MF_LOG_TRACE("  Resource[%u]: %s (%s)", i, res_desc->name, res_desc->dtype == MF_DTYPE_F32 ? "F32" : "Other");

        res->name = strdup(res_desc->name);
        res->name_hash = mf_fnv1a_hash(res->name);
        
        // Descriptor Prototype
        memset(&res->desc, 0, sizeof(mf_tensor));
        res->desc.info.dtype = res_desc->dtype;
        res->desc.info.ndim = res_desc->ndim;
        memcpy(res->desc.info.shape, res_desc->shape, sizeof(int32_t) * res_desc->ndim);
        
        // Handle Dynamic Shapes
        bool is_dynamic = false;
        for (int k = 0; k < res->desc.info.ndim; ++k) {
            if (res->desc.info.shape[k] < 0) {
                is_dynamic = true;
                break;
            }
        }

        // Init strides
        int32_t stride = 1;
        for (int k = res->desc.info.ndim - 1; k >= 0; --k) {
            res->desc.info.strides[k] = stride;
            if (res->desc.info.shape[k] > 0) stride *= res->desc.info.shape[k];
        }
        
        size_t bytes = is_dynamic ? 0 : mf_tensor_size_bytes(&res->desc);
        res->size_bytes = bytes;
        
        // Allocate Buffers
        res->buffer_a = MF_ARENA_PUSH(&engine->arena, mf_buffer, 1);
        res->buffer_b = MF_ARENA_PUSH(&engine->arena, mf_buffer, 1);
        
        if (bytes > 0) {
            mf_buffer_alloc(res->buffer_a, allocator, bytes);
            mf_buffer_alloc(res->buffer_b, allocator, bytes);
        } else {
            memset(res->buffer_a, 0, sizeof(mf_buffer));
            memset(res->buffer_b, 0, sizeof(mf_buffer));
            MF_LOG_TRACE("Resource '%s' has dynamic shape. Staying unallocated.", res->name);
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
        
        // --- NEW: Initialize Global Resources from Program Constants ---
        for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
            mf_bin_symbol* sym = &ker->program->symbols[s];
            mf_tensor* t_prog = &ker->program->tensors[sym->register_idx];
            
            if (mf_tensor_is_valid(t_prog)) {
                // Find matching resource
                for (u32 r = 0; r < engine->resource_count; ++r) {
                    // Optimized: Compare hashes first
                    if (engine->resources[r].name_hash == sym->name_hash) {
                        mf_resource_inst* res = &engine->resources[r];
                        void* prog_data = mf_tensor_data(t_prog);
                        size_t bytes = mf_tensor_size_bytes(t_prog);
                        
                        if (res->size_bytes == bytes) {
                            memcpy(res->buffer_a->data, prog_data, bytes);
                            memcpy(res->buffer_b->data, prog_data, bytes);
                            MF_LOG_TRACE("    Initialized resource '%s' from kernel constant.", res->name);
                        }
                        break;
                    }
                }
            }
        }
        
        // 3. Setup Bindings
        ker->binding_count = ker_desc->binding_count;
        ker->bindings = MF_ARENA_PUSH(&engine->arena, mf_kernel_binding, ker->binding_count);
        
        mf_resource_inst* reference_output = NULL;

        for (u32 b = 0; b < ker_desc->binding_count; ++b) {
            mf_pipeline_binding* bind = &ker_desc->bindings[b];
            u32 port_hash = mf_fnv1a_hash(bind->kernel_port);
            u32 res_hash = mf_fnv1a_hash(bind->global_resource);
            
            // Find Local Reg Index
            int32_t local_idx = -1;
            u8 symbol_flags = 0;
            
            for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
                if (ker->program->symbols[s].name_hash == port_hash) {
                    local_idx = ker->program->symbols[s].register_idx;
                    symbol_flags = ker->program->symbols[s].flags;
                    break;
                }
            }
            
            // Find Global Resource Index
            int32_t global_idx = -1;
            for (u32 r = 0; r < engine->resource_count; ++r) {
                if (engine->resources[r].name_hash == res_hash) {
                    global_idx = r;
                    break;
                }
            }
            
            if (local_idx != -1 && global_idx != -1) {
                ker->bindings[b].local_reg = (u16)local_idx;
                ker->bindings[b].global_res = (u16)global_idx;
                ker->bindings[b].flags = symbol_flags; // Cache flags

                if (symbol_flags & MF_SYMBOL_FLAG_OUTPUT) {
                    mf_resource_inst* curr_res = &engine->resources[global_idx];
                    
                    if (!reference_output) {
                        reference_output = curr_res;
                        MF_LOG_TRACE("  Kernel %s domain set by output '%s' (%zu bytes)", 
                            ker->id, bind->kernel_port, curr_res->size_bytes);
                    } else {
                        // Validate Uniform Grid Rule: All outputs must have same shape (size in bytes is a good proxy for now)
                        if (reference_output->size_bytes != curr_res->size_bytes) {
                            MF_LOG_ERROR("Kernel %s: Output shape mismatch! '%s' (%zu bytes) vs reference '%s' (%zu bytes). All outputs must match.",
                                ker->id, bind->kernel_port, curr_res->size_bytes, 
                                reference_output->name, reference_output->size_bytes);
                        }
                    }
                }
            } else {
                MF_LOG_ERROR("Failed to bind %s -> %s in kernel %s", bind->kernel_port, bind->global_resource, ker->id);
            }
        }
        
        if (!reference_output) {
            MF_LOG_WARN("Kernel %s has no bound OUTPUT symbols. It will not execute!", ker->id);
        }
    }
}

void mf_engine_dispatch(mf_engine* engine) {
    if (!engine) return;

    MF_LOG_TRACE("Dispatching Pipeline frame %llu", (unsigned long long)engine->frame_index);
    
    bool is_even = (engine->frame_index % 2 == 0);

    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        mf_kernel_inst* ker = &engine->kernels[k_idx];
        
        // --- Step 1: Auto-Resize based on Reference Inputs ---
        if (ker->program->symbols) {
             for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
                 mf_bin_symbol* sym = &ker->program->symbols[s];
                 // Check if it's an OUTPUT with a declared dependency
                 if ((sym->flags & MF_SYMBOL_FLAG_OUTPUT) && sym->related_name_hash != 0) {
                     
                     // 1. Find Binding for Output
                     u32 out_res_idx = 0xFFFFFFFF;
                     for(u32 kb=0; kb < ker->binding_count; ++kb) {
                         if (ker->bindings[kb].local_reg == sym->register_idx) {
                             out_res_idx = ker->bindings[kb].global_res;
                             break;
                         }
                     }
                     if (out_res_idx >= engine->resource_count) continue;
                     
                     // 2. Find Input Symbol (by Hash)
                     u32 related_reg_idx = 0xFFFFFFFF;
                     for (u32 s2 = 0; s2 < ker->program->meta.symbol_count; ++s2) {
                         if (ker->program->symbols[s2].name_hash == sym->related_name_hash) {
                             related_reg_idx = ker->program->symbols[s2].register_idx;
                             break;
                         }
                     }
                     if (related_reg_idx == 0xFFFFFFFF) continue;
                     
                     // 3. Find Binding for Input
                     u32 in_res_idx = 0xFFFFFFFF;
                     for(u32 kb=0; kb < ker->binding_count; ++kb) {
                         if (ker->bindings[kb].local_reg == related_reg_idx) {
                             in_res_idx = ker->bindings[kb].global_res;
                             break;
                         }
                     }
                     if (in_res_idx >= engine->resource_count) continue;
                     
                     // 4. Compare & Resize
                     mf_resource_inst* res_in = &engine->resources[in_res_idx];
                     mf_resource_inst* res_out = &engine->resources[out_res_idx];
                     
                     if (!mf_tensor_same_shape(&res_in->desc, &res_out->desc)) {
                         // Only resize if input is valid (allocated)
                         if (res_in->desc.buffer && res_in->desc.buffer->data) {
                             MF_LOG_TRACE("Auto-Resizing '%s' to match '%s'", res_out->name, res_in->name);
                             mf_engine_resize_resource(engine, res_out->name, res_in->desc.info.shape, res_in->desc.info.ndim);
                         }
                     }
                 }
             }
        }

        const mf_tensor* kernel_domain = NULL;
        
        // Zero-Copy Bindings: Map Global Resources to Local Registers
        for (u32 b = 0; b < ker->binding_count; ++b) {
            u16 local_reg = ker->bindings[b].local_reg;
            u16 global_res = ker->bindings[b].global_res;
            
            if (global_res >= engine->resource_count) continue;
            
            mf_resource_inst* res = &engine->resources[global_res];
            mf_tensor* t = &ker->state.registers[local_reg];

            // Determine if Output or Input based on flags
            u8 flags = ker->bindings[b].flags;

            mf_buffer* active_buf;
            if (flags & MF_SYMBOL_FLAG_OUTPUT) {
                // Write to Next Frame (Back Buffer)
                active_buf = is_even ? res->buffer_b : res->buffer_a;
                
                // Use the first output found as the execution domain
                if (!kernel_domain) {
                    kernel_domain = t;
                }
            } else {
                // Read from Previous Frame (Front Buffer)
                active_buf = is_even ? res->buffer_a : res->buffer_b;
            }

            // Bind buffer to tensor view
            t->buffer = active_buf;
            t->byte_offset = 0;
            
            // Sync metadata from resource
            t->info = res->desc.info;
        }
        
        if (!kernel_domain) continue;

        // --- Step 2: Pre-Dispatch Validation ---
        bool can_execute = true;
        for (u32 b = 0; b < ker->binding_count; ++b) {
            if (ker->bindings[b].flags & MF_SYMBOL_FLAG_OUTPUT) {
                u16 local_reg = ker->bindings[b].local_reg;
                mf_tensor* t = &ker->state.registers[local_reg];
                if (!mf_tensor_is_valid(t)) {
                    const char* port_name = "unknown";
                    if (ker->program->symbols) {
                        for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
                            if (ker->program->symbols[s].register_idx == local_reg) {
                                port_name = ker->program->symbols[s].name;
                                break;
                            }
                        }
                    }
                    MF_LOG_ERROR("Kernel %s: Output port '%s' is unallocated! Resize the resource before dispatch.", 
                        ker->id, port_name);
                    can_execute = false;
                    break;
                }
            }
        }
        if (!can_execute) continue;

        MF_LOG_TRACE("  Executing Kernel: %s", ker->id);

        // Execute Kernel
        // Execute Kernel
        for (u32 f = 0; f < ker->frequency; ++f) {
            if (engine->backend.dispatch && kernel_domain) {
                engine->backend.dispatch(engine->backend.state, ker->program, &ker->state, kernel_domain);
                
                // Stop dispatching if a kernel encountered a runtime error
                if (ker->state.error_code != 0) {
                    MF_LOG_ERROR("Kernel '%s' failed with error code %d. Aborting pipeline dispatch.", ker->id, ker->state.error_code);
                    goto end_dispatch;
                }
            }
        }
    }
    
end_dispatch:
    engine->frame_index++;
}

mf_tensor* mf_engine_map_resource(mf_engine* engine, const char* name) {
    if (!engine) return NULL;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (strcmp(engine->resources[i].name, name) == 0) {
            mf_resource_inst* res = &engine->resources[i];
            
            bool is_even = (engine->frame_index % 2 == 0);
            res->desc.buffer = is_even ? res->buffer_a : res->buffer_b;
            res->desc.byte_offset = 0;
            
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
            
            mf_allocator* alloc = (mf_allocator*)&engine->heap;
            
            // Update descriptor logic first to get new size
            // This is slightly inefficient (temp copy), but safe
            mf_type_info new_info = res->desc.info;
            new_info.ndim = new_ndim;
            memcpy(new_info.shape, new_shape, sizeof(int32_t) * MF_MAX_DIMS);
            // Recalc strides
            int32_t stride = 1;
            for (int k = new_ndim - 1; k >= 0; --k) {
                new_info.strides[k] = stride;
                stride *= new_info.shape[k];
            }
            
            size_t new_bytes = 1;
            for(int d=0; d<new_ndim; ++d) new_bytes *= new_shape[d];
            new_bytes *= mf_dtype_size(new_info.dtype);
            
            // Reallocate Buffers (Using helper to simplify)
            // Note: mf_buffer_alloc doesn't resize, it allocs new.
            // We need to free old and alloc new? Or add resize support to mf_buffer?
            // Let's just free and alloc for now (lossy resize). 
            // Ideally we should preserve data if possible, but for double-buffered pipelines,
            // resizing usually implies a complete reset of that resource.
            
            mf_buffer_free(res->buffer_a);
            mf_buffer_alloc(res->buffer_a, alloc, new_bytes);
            
            mf_buffer_free(res->buffer_b);
            mf_buffer_alloc(res->buffer_b, alloc, new_bytes);
            
            res->size_bytes = new_bytes;
            res->desc.info = new_info;
            
            return true;
        }
    }
    return false;
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
        
        bool is_even = (engine->frame_index % 2 == 0);
        res->desc.buffer = is_even ? res->buffer_a : res->buffer_b;
        res->desc.byte_offset = 0;

        cb(res->name, &res->desc, user_data);
    }
}
