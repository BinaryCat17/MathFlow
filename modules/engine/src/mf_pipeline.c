#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>
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
    mf_allocator* allocator = (mf_allocator*)&engine->heap;
    engine->resource_count = pipe->resource_count;
    engine->resources = MF_ARENA_PUSH(&engine->arena, mf_resource_inst, engine->resource_count);
    
    for (u32 i = 0; i < pipe->resource_count; ++i) {
        mf_pipeline_resource* desc = &pipe->resources[i];
        mf_resource_inst* res = &engine->resources[i];
        
        res->name = mf_arena_strdup(&engine->arena, desc->name);
        res->name_hash = mf_fnv1a_hash(res->name);
        
        memset(&res->desc, 0, sizeof(mf_tensor));
        res->desc.info.dtype = desc->dtype;
        res->desc.info.ndim = desc->ndim;
        memcpy(res->desc.info.shape, desc->shape, sizeof(int32_t) * desc->ndim);
        
        int32_t stride = 1;
        bool is_dynamic = false;
        for (int k = desc->ndim - 1; k >= 0; --k) {
            res->desc.info.strides[k] = stride;
            if (desc->shape[k] > 0) stride *= desc->shape[k];
            else is_dynamic = true;
        }
        
        res->size_bytes = is_dynamic ? 0 : mf_tensor_size_bytes(&res->desc);
        res->buffers[0] = MF_ARENA_PUSH(&engine->arena, mf_buffer, 1);
        res->buffers[1] = MF_ARENA_PUSH(&engine->arena, mf_buffer, 1);
        
        if (res->size_bytes > 0) {
            mf_buffer_alloc(res->buffers[0], allocator, res->size_bytes);
            mf_buffer_alloc(res->buffers[1], allocator, res->size_bytes);
        } else {
            memset(res->buffers[0], 0, sizeof(mf_buffer));
            memset(res->buffers[1], 0, sizeof(mf_buffer));
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
        
        mf_state_reset(&ker->state, ker->program, &engine->arena);
    }
}

static void resolve_bindings(mf_engine* engine, const mf_pipeline_desc* pipe) {
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        mf_pipeline_kernel* desc = &pipe->kernels[i];
        mf_kernel_inst* ker = &engine->kernels[i];
        
        ker->bindings = MF_ARENA_PUSH(&engine->arena, mf_kernel_binding, desc->binding_count);
        u32 actual_bindings = 0;
        
        u32 max_resize = 0;
        for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
            if (ker->program->symbols[s].related_name_hash != 0) max_resize++;
        }
        ker->resize_tasks = MF_ARENA_PUSH(&engine->arena, mf_auto_resize_task, max_resize);
        ker->resize_task_count = 0;

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
            mf_kernel_binding* kb = &ker->bindings[actual_bindings++];
            kb->local_reg = (u16)sym->register_idx;
            kb->global_res = (u16)res_idx;
            kb->flags = sym->flags;

            if (sym->related_name_hash != 0) {
                for (u32 b2 = 0; b2 < desc->binding_count; ++b2) {
                    if (mf_fnv1a_hash(desc->bindings[b2].kernel_port) == sym->related_name_hash) {
                        int32_t rel_res_idx = find_resource_idx(engine, mf_fnv1a_hash(desc->bindings[b2].global_resource));
                        if (rel_res_idx != -1) {
                            ker->resize_tasks[ker->resize_task_count++] = (mf_auto_resize_task){(u16)rel_res_idx, (u16)res_idx};
                        }
                        break;
                    }
                }
            }
        }
        ker->binding_count = actual_bindings;
    }
}

static void apply_initial_data(mf_engine* engine) {
    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        mf_kernel_inst* ker = &engine->kernels[k_idx];
        for (u32 s = 0; s < ker->program->meta.symbol_count; ++s) {
            mf_bin_symbol* sym = &ker->program->symbols[s];
            mf_tensor* t_const = &ker->program->tensors[sym->register_idx];
            
            if (mf_tensor_is_valid(t_const)) {
                for (u32 b = 0; b < ker->binding_count; ++b) {
                    if (ker->bindings[b].local_reg == sym->register_idx) {
                        mf_resource_inst* res = &engine->resources[ker->bindings[b].global_res];
                        size_t bytes = mf_tensor_size_bytes(t_const);
                        if (res->size_bytes == bytes) {
                            MF_LOG_DEBUG("Engine: Initializing resource '%s' from kernel '%s' symbol '%s' (%zu bytes)", 
                                res->name, ker->id, sym->name, bytes);
                            memcpy(res->buffers[0]->data, mf_tensor_data(t_const), bytes);
                            memcpy(res->buffers[1]->data, mf_tensor_data(t_const), bytes);
                        } else {
                            MF_LOG_WARN("Engine: Resource '%s' size mismatch for initial data from '%s' (expected %zu, got %zu)",
                                res->name, ker->id, res->size_bytes, bytes);
                        }
                        break;
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
    apply_initial_data(engine);
}
