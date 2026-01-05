#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_shape.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/isa/mf_exec_ctx.h>
#include <string.h>
#include <stdio.h>

// --- Internal State Management ---

void mf_state_reset(mf_state* state, const mf_program* prog, mf_arena* arena, mf_backend* backend) {
    if (!prog) return;
    
    state->register_count = prog->meta.tensor_count;
    state->registers = MF_ARENA_PUSH(arena, mf_tensor, state->register_count);
    state->ownership_flags = MF_ARENA_PUSH(arena, uint8_t, state->register_count);
    
    if (!state->registers || !state->ownership_flags) {
        MF_LOG_ERROR("Engine: Failed to allocate state registers for program.");
        return;
    }
    
    memset(state->ownership_flags, 0, state->register_count);

    for (u32 i = 0; i < state->register_count; ++i) {
        mf_type_info* info_prog = &prog->tensor_infos[i];
        void* data_prog = prog->tensor_data[i];
        mf_tensor* t_reg = &state->registers[i];
        uint8_t flags = prog->tensor_flags[i];
        
        t_reg->info = *info_prog;
        t_reg->byte_offset = 0;
        t_reg->buffer = NULL;

        if (data_prog) {
            t_reg->buffer = state->allocator->alloc(state->allocator, sizeof(mf_buffer));
            mf_buffer_init_view(t_reg->buffer, data_prog, mf_shape_calc_bytes(info_prog->dtype, info_prog->shape, info_prog->ndim));
            state->ownership_flags[i] = 1; // Mark for cleanup
        } else {
            // Pre-allocate only non-alias, non-generator static tensors
            if (!(flags & MF_TENSOR_FLAG_ALIAS) && !(flags & MF_TENSOR_FLAG_GENERATOR)) {
                bool is_static = true;
                for (int d = 0; d < t_reg->info.ndim; ++d) if (t_reg->info.shape[d] < 0) { is_static = false; break; }
                if (is_static) {
                    t_reg->buffer = state->allocator->alloc(state->allocator, sizeof(mf_buffer));
                    if (mf_tensor_alloc(t_reg, state->allocator, &t_reg->info)) {
                        state->ownership_flags[i] = 1;
                    } else {
                        state->allocator->free(state->allocator, t_reg->buffer);
                        t_reg->buffer = NULL;
                    }
                }
            }
        }
    }

    // --- BAKING PHASE ---
    if (backend && backend->bake) {
        state->baked_data = backend->bake(backend->state, prog);
    }
}

static void mf_state_shutdown(mf_state* state, mf_backend* backend) {
    if (!state->registers || !state->allocator) return;
    
    if (backend && backend->free_baked && state->baked_data) {
        backend->free_baked(backend->state, state->baked_data);
        state->baked_data = NULL;
    }

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
    if (!engine) return NULL;

    size_t arena_size = (desc && desc->arena_size > 0) ? desc->arena_size : MF_MB(8);
    size_t heap_size = (desc && desc->heap_size > 0) ? desc->heap_size : MF_MB(64);

    engine->arena_buffer = malloc(arena_size);
    if (!engine->arena_buffer) { free(engine); return NULL; }
    mf_arena_init(&engine->arena, engine->arena_buffer, arena_size);

    engine->heap_buffer = malloc(heap_size);
    if (!engine->heap_buffer) { free(engine->arena_buffer); free(engine); return NULL; }
    mf_heap_init(&engine->heap, engine->heap_buffer, heap_size);

    if (desc) engine->backend = desc->backend;

    engine->front_idx = 0;
    engine->back_idx = 1;
    mf_atomic_store(&engine->error_code, 0);

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
        mf_state_shutdown(&engine->kernels[i].state, &engine->backend);
    }
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (engine->resources[i].buffers[0]) mf_buffer_free(engine->resources[i].buffers[0]);
        if (engine->resources[i].buffers[1] && engine->resources[i].buffers[1] != engine->resources[i].buffers[0]) {
            mf_buffer_free(engine->resources[i].buffers[1]);
        }
    }
    mf_arena_reset(&engine->arena);
    if (engine->heap_buffer) mf_heap_init(&engine->heap, engine->heap_buffer, engine->heap.size);
    engine->kernel_count = 0;
    engine->resource_count = 0;
    mf_atomic_store(&engine->error_code, 0);
}

mf_arena* mf_engine_get_arena(mf_engine* engine) {
    return engine ? &engine->arena : NULL;
}

void mf_engine_dispatch(mf_engine* engine) {
    if (!engine || mf_atomic_load(&engine->error_code) != 0) return;

    u8 front = engine->front_idx;
    u8 back  = engine->back_idx;

    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        mf_kernel_inst* ker = &engine->kernels[k_idx];
        if (mf_atomic_load(&engine->error_code) != 0) break;
        
        // 1. Resource Binding
        for (u32 b = 0; b < ker->binding_count; ++b) {
            mf_kernel_binding* bind = &ker->bindings[b];
            mf_resource_inst* res = &engine->resources[bind->global_res];
            mf_tensor* t = &ker->state.registers[bind->local_reg];
            *t = res->desc;
            t->buffer = (bind->flags & MF_SYMBOL_FLAG_OUTPUT) ? res->buffers[back] : res->buffers[front];
            t->byte_offset = 0;
        }
        
        // 3. Execution
        for (u32 f = 0; f < ker->frequency; ++f) {
            if (engine->backend.dispatch) {
                ker->state.global_error_ptr = &engine->error_code;
                for (u32 t = 0; t < ker->program->meta.task_count; ++t) {
                    mf_task* task = &ker->program->tasks[t];
                    const mf_tensor* task_domain = &ker->state.registers[task->domain_reg];
                    engine->backend.dispatch(engine->backend.state, ker->program, &ker->state, task_domain, task->start_inst, task->inst_count);
                    if (mf_atomic_load(&engine->error_code) != 0) goto end_dispatch;
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
    
    mf_type_info new_info;
    mf_type_info_init_contiguous(&new_info, res->desc.info.dtype, new_shape, new_ndim);
    size_t new_bytes = mf_shape_calc_count(new_shape, new_ndim) * mf_dtype_size(new_info.dtype);
    
    if (res->size_bytes != new_bytes) {
        bool is_transient = (res->buffers[0] == res->buffers[1]);
        if (res->buffers[0] && res->buffers[0]->data) mf_buffer_free(res->buffers[0]);
        if (!mf_buffer_alloc(res->buffers[0], alloc, new_bytes)) return false;
        
        if (is_transient) res->buffers[1] = res->buffers[0];
        else {
            if (res->buffers[1] && res->buffers[1]->data) mf_buffer_free(res->buffers[1]);
            if (!mf_buffer_alloc(res->buffers[1], alloc, new_bytes)) return false;
        }
        res->size_bytes = new_bytes;
    }
    res->desc.info = new_info;
    return true;
}

void mf_engine_sync_resource(mf_engine* engine, const char* name) {
    if (!engine || !name) return;
    u32 hash = mf_fnv1a_hash(name);
    int32_t idx = find_resource_idx(engine, hash);
    if (idx == -1) return;
    mf_resource_inst* res = &engine->resources[idx];
    if (res->buffers[0] && res->buffers[1] && res->buffers[0] != res->buffers[1]) {
        if (res->buffers[0]->data && res->buffers[1]->data) {
            memcpy(res->buffers[1 - engine->front_idx]->data, res->buffers[engine->front_idx]->data, res->size_bytes);
        }
    }
}

mf_engine_error mf_engine_get_error(mf_engine* engine) {
    if (!engine) return MF_ENGINE_ERR_NONE;
    int32_t err = mf_atomic_load(&engine->error_code);
    if (err == 0) return MF_ENGINE_ERR_NONE;
    if (err == MF_ERROR_OOM) return MF_ENGINE_ERR_OOM;
    if (err == MF_ERROR_SHAPE_MISMATCH) return MF_ENGINE_ERR_SHAPE;
    if (err == MF_ERROR_INVALID_OP) return MF_ENGINE_ERR_INVALID_OP;
    return MF_ENGINE_ERR_RUNTIME;
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