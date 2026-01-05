#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_shape.h>
#include <string.h>

// --- Helpers ---

int32_t find_resource_idx(mf_engine* engine, u32 name_hash) {
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (engine->resources[i].name_hash == name_hash) return (int32_t)i;
    }
    return -1;
}

int32_t find_symbol_idx(const mf_program* prog, u32 name_hash) {
    if (!prog->symbols) return -1;
    for (u32 i = 0; i < prog->meta.symbol_count; ++i) {
        if (prog->symbols[i].name_hash == name_hash) return (int32_t)i;
    }
    return -1;
}

static bool is_shape_static(const mf_type_info* info) {
    for (int i = 0; i < info->ndim; ++i) {
        if (info->shape[i] < 0) return false;
    }
    return true;
}

// --- Pipeline Setup Stages ---


static void init_resources(mf_engine* engine, const mf_pipeline_desc* pipe) {
    engine->resource_count = pipe->resource_count;
    engine->resources = MF_ARENA_PUSH(&engine->arena, mf_resource_inst, engine->resource_count);
    
    for (u32 i = 0; i < pipe->resource_count; ++i) {
        mf_pipeline_resource* desc = &pipe->resources[i];
        mf_resource_inst* res = &engine->resources[i];
        
        res->name = mf_arena_strdup(&engine->arena, desc->name);
        res->name_hash = mf_fnv1a_hash(res->name);
        res->provider = desc->provider ? mf_arena_strdup(&engine->arena, desc->provider) : NULL;
        res->flags = desc->flags;
        
        memset(&res->desc, 0, sizeof(mf_tensor));
        mf_type_info_init_contiguous(&res->desc.info, desc->dtype, desc->shape, desc->ndim);
        
        bool is_dynamic = false;
        for (int k = 0; k < desc->ndim; ++k) if (desc->shape[k] <= 0) is_dynamic = true;
        res->size_bytes = is_dynamic ? 0 : mf_tensor_size_bytes(&res->desc);
        
        res->buffers[0] = NULL;
        res->buffers[1] = NULL;
    }
}

static void analyze_transience(mf_engine* engine) {
    for (u32 r_idx = 0; r_idx < engine->resource_count; ++r_idx) {
        mf_resource_inst* res = &engine->resources[r_idx];
        
        if (res->flags & MF_RESOURCE_FLAG_PERSISTENT) continue;
        if (res->flags & MF_RESOURCE_FLAG_TRANSIENT) continue;

        bool read_before_write = false;
        bool write_happened = false;

        for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
            mf_kernel_inst* ker = &engine->kernels[k_idx];
            bool k_reads = false;
            bool k_writes = false;

            for (u32 b = 0; b < ker->binding_count; ++b) {
                if (ker->bindings[b].global_res == r_idx) {
                    if (ker->bindings[b].flags & MF_SYMBOL_FLAG_INPUT) k_reads = true;
                    if (ker->bindings[b].flags & MF_SYMBOL_FLAG_OUTPUT) k_writes = true;
                }
            }

            if (k_reads && !write_happened) { read_before_write = true; break; }
            if (k_writes) write_happened = true;
        }

        if (!read_before_write && write_happened) {
            res->flags |= MF_RESOURCE_FLAG_TRANSIENT;
        }
    }
}

static void allocate_resources(mf_engine* engine) {
    mf_allocator* allocator = (mf_allocator*)&engine->heap;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        mf_resource_inst* res = &engine->resources[i];
        bool is_transient = (res->flags & MF_RESOURCE_FLAG_TRANSIENT) != 0;
        
        res->buffers[0] = MF_ARENA_PUSH(&engine->arena, mf_buffer, 1);
        if (res->size_bytes > 0) mf_buffer_alloc(res->buffers[0], allocator, res->size_bytes);
        else memset(res->buffers[0], 0, sizeof(mf_buffer));

        if (is_transient) {
            res->buffers[1] = res->buffers[0]; // Point to same buffer
        } else {
            res->buffers[1] = MF_ARENA_PUSH(&engine->arena, mf_buffer, 1);
            if (res->size_bytes > 0) mf_buffer_alloc(res->buffers[1], allocator, res->size_bytes);
            else memset(res->buffers[1], 0, sizeof(mf_buffer));
        }
    }
}

static void init_kernels(mf_engine* engine, const mf_pipeline_desc* pipe, mf_program** programs) {
    engine->kernel_count = pipe->kernel_count;
    engine->kernels = MF_ARENA_PUSH(&engine->arena, mf_kernel_inst, engine->kernel_count);
    
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        mf_pipeline_kernel* desc = &pipe->kernels[i];
        mf_kernel_inst* ker = &engine->kernels[i];
        
        ker->id = mf_arena_strdup(&engine->arena, desc->id);
        ker->id_hash = mf_fnv1a_hash(ker->id);
        ker->program = programs[i];
        ker->frequency = desc->frequency;
        ker->state.allocator = (mf_allocator*)&engine->heap;
        
        mf_state_reset(&ker->state, ker->program, &engine->arena, &engine->backend);
    }
}

static void resolve_bindings(mf_engine* engine, const mf_pipeline_desc* pipe) {
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        mf_pipeline_kernel* desc = &pipe->kernels[i];
        mf_kernel_inst* ker = &engine->kernels[i];
        
        // Count total potential bindings: explicit in manifest + all symbols in program
        u32 max_bindings = desc->binding_count + ker->program->meta.symbol_count;
        ker->bindings = MF_ARENA_PUSH(&engine->arena, mf_kernel_binding, max_bindings);
        u32 actual_bindings = 0;
        
        u32 max_resize = 0;
        for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
            if (ker->program->symbols[s].related_name_hash != 0) max_resize++;
        }
        ker->resize_tasks = MF_ARENA_PUSH(&engine->arena, mf_auto_resize_task, max_resize);
        ker->resize_task_count = 0;

        // 1. Process Explicit Bindings from Manifest
        for (u32 b = 0; b < desc->binding_count; ++b) {
            mf_pipeline_binding* pb = &desc->bindings[b];
            u32 port_hash = mf_fnv1a_hash(pb->kernel_port);
            u32 res_hash = mf_fnv1a_hash(pb->global_resource);
            
            int32_t sym_idx = find_symbol_idx(ker->program, port_hash);
            int32_t res_idx = find_resource_idx(engine, res_hash);
            
            if (sym_idx == -1) {
                MF_LOG_WARN("Kernel '%s': Port '%s' not found in program symbols.", ker->id, pb->kernel_port);
                continue;
            }
            if (res_idx == -1) {
                MF_LOG_ERROR("Kernel '%s': Global resource '%s' not found.", ker->id, pb->global_resource);
                continue;
            }

            mf_bin_symbol* sym = &ker->program->symbols[sym_idx];
            mf_resource_inst* res = &engine->resources[res_idx];

            // Validate and Bind
            bool is_out = (sym->flags & MF_SYMBOL_FLAG_OUTPUT) != 0;
            mf_type_info res_info;
            mf_type_info_init_contiguous(&res_info, res->desc.info.dtype, res->desc.info.shape, res->desc.info.ndim);

            if (!mf_shape_is_compatible(&ker->program->tensor_infos[sym->register_idx], &res_info, is_out)) {
                MF_LOG_ERROR("Kernel '%s': Binding failure for port '%s'. Incompatible shape/type.", ker->id, pb->kernel_port);
                continue;
            }

            mf_kernel_binding* kb = &ker->bindings[actual_bindings++];
            kb->local_reg = (u16)sym->register_idx;
            kb->global_res = (u16)res_idx;
            kb->flags = sym->flags;
        }

        // 2. Process Implicit Bindings (by Name) for remaining I/O symbols
        for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
            mf_bin_symbol* sym = &ker->program->symbols[s];
            if (!(sym->flags & (MF_SYMBOL_FLAG_INPUT | MF_SYMBOL_FLAG_OUTPUT))) continue;

            // Check if already bound explicitly
            bool already_bound = false;
            for (u32 b = 0; b < actual_bindings; ++b) {
                if (ker->bindings[b].local_reg == sym->register_idx) { already_bound = true; break; }
            }
            if (already_bound) continue;

            // Try to find by name
            int32_t res_idx = find_resource_idx(engine, sym->name_hash);
            if (res_idx != -1) {
                mf_resource_inst* res = &engine->resources[res_idx];
                bool is_out = (sym->flags & MF_SYMBOL_FLAG_OUTPUT) != 0;
                
                mf_type_info res_info;
                mf_type_info_init_contiguous(&res_info, res->desc.info.dtype, res->desc.info.shape, res->desc.info.ndim);

                if (mf_shape_is_compatible(&ker->program->tensor_infos[sym->register_idx], &res_info, is_out)) {
                    MF_LOG_DEBUG("Kernel '%s': Implicitly binding port '%s' to resource '%s'", ker->id, sym->name, res->name);
                    mf_kernel_binding* kb = &ker->bindings[actual_bindings++];
                    kb->local_reg = (u16)sym->register_idx;
                    kb->global_res = (u16)res_idx;
                    kb->flags = sym->flags;
                }
            }
        }

        // 3. Setup Auto-Resize Tasks and Apply Initial Data
        for (u32 b = 0; b < actual_bindings; ++b) {
            mf_kernel_binding* kb = &ker->bindings[b];
            mf_bin_symbol* sym = NULL;
            for(u32 s=0; s < ker->program->meta.symbol_count; ++s) {
                if(ker->program->symbols[s].register_idx == kb->local_reg) { sym = &ker->program->symbols[s]; break; }
            }
            if (!sym) continue;

            // Apply Initial Data if present
            mf_type_info* info_const = &ker->program->tensor_infos[sym->register_idx];
            void* data_const = ker->program->tensor_data[sym->register_idx];
            if (data_const) {
                mf_resource_inst* res = &engine->resources[kb->global_res];
                size_t bytes = mf_shape_calc_bytes(info_const->dtype, info_const->shape, info_const->ndim);
                // We'll defer actual memcpy until resources are allocated in bind_pipeline
                // But wait, apply_initial_data happens AFTER allocate_resources.
                // So I should keep the separation but make it O(Bindings) instead of O(Symbols * Bindings).
            }

            if (sym->related_name_hash == 0) continue;

            for (u32 b2 = 0; b2 < actual_bindings; ++b2) {
                mf_bin_symbol* sym2 = NULL;
                for(u32 s2=0; s2 < ker->program->meta.symbol_count; ++s2) {
                    if(ker->program->symbols[s2].register_idx == ker->bindings[b2].local_reg) { sym2 = &ker->program->symbols[s2]; break; }
                }
                if (sym2 && sym2->name_hash == sym->related_name_hash) {
                    ker->resize_tasks[ker->resize_task_count++] = (mf_auto_resize_task){(u16)ker->bindings[b2].global_res, (u16)kb->global_res};
                    break;
                }
            }
        }
        ker->binding_count = actual_bindings;
    }
}

static void apply_initial_data(mf_engine* engine) {
    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        mf_kernel_inst* ker = &engine->kernels[k_idx];
        for (u32 b = 0; b < ker->binding_count; ++b) {
            mf_kernel_binding* bind = &ker->bindings[b];
            u16 reg_idx = bind->local_reg;
            void* data_const = ker->program->tensor_data[reg_idx];
            
            if (data_const) {
                mf_resource_inst* res = &engine->resources[bind->global_res];
                mf_type_info* info_const = &ker->program->tensor_infos[reg_idx];
                size_t bytes = mf_shape_calc_bytes(info_const->dtype, info_const->shape, info_const->ndim);
                
                if (res->size_bytes == bytes && res->buffers[0]->data) {
                    MF_LOG_DEBUG("Engine: Initializing resource '%s' from kernel '%s' register %u (%zu bytes)", 
                        res->name, ker->id, reg_idx, bytes);
                    memcpy(res->buffers[0]->data, data_const, bytes);
                    if (res->buffers[1] != res->buffers[0] && res->buffers[1]->data) {
                        memcpy(res->buffers[1]->data, data_const, bytes);
                    }
                }
            }
        }
    }
}

// --- Public API ---

void mf_engine_bind_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe, mf_program** programs) {
    if (!engine || !pipe) return;
    MF_LOG_INFO("Binding Pipeline: %u resources, %u kernels", pipe->resource_count, pipe->kernel_count);

    init_resources(engine, pipe);
    init_kernels(engine, pipe, programs);
    resolve_bindings(engine, pipe);
    
    analyze_transience(engine);
    allocate_resources(engine);
    
    apply_initial_data(engine);

    size_t total_mem = 0;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        mf_resource_inst* res = &engine->resources[i];
        char s_shape[64];
        mf_shape_format(&res->desc.info, s_shape, sizeof(s_shape));
        bool is_transient = (res->flags & MF_RESOURCE_FLAG_TRANSIENT) != 0;
        
        MF_LOG_TRACE("Engine: Resource '%s' ready. Shape: %s, Size: %zu bytes %s", 
            res->name, s_shape, res->size_bytes, 
            is_transient ? "[TRANSIENT]" : "");
            
        total_mem += res->size_bytes * (is_transient ? 1 : 2);
    }
    MF_LOG_INFO("Engine: Pipeline ready. Total GPU/Shared Memory: %.2f MB", (double)total_mem / (1024.0 * 1024.0));
}
