#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"
#include <mathflow/isa/mf_exec_ctx.h>

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
                if (dst->data) memcpy(dst->data, src->data, bytes);
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

    if (desc) {
        engine->backend = desc->backend;
    }

    engine->state.allocator = (mf_allocator*)&engine->heap;

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

    if (engine->state_buffers && engine->program) {
        for (u32 i = 0; i < engine->program->meta.state_count; ++i) {
            free(engine->state_buffers[i].buffer_a);
            free(engine->state_buffers[i].buffer_b);
        }
        engine->state_buffers = NULL;
    }

    mf_state_shutdown(&engine->state);
    mf_arena_reset(&engine->arena);
    
    if (engine->heap_buffer) {
        mf_heap_init(&engine->heap, engine->heap_buffer, engine->heap.size);
    }
    
    engine->program = NULL;
    engine->state.allocator = (mf_allocator*)&engine->heap;
}

mf_arena* mf_engine_get_arena(mf_engine* engine) {
    if (!engine) return NULL;
    return &engine->arena;
}

void mf_engine_bind_program(mf_engine* engine, mf_program* prog) {
    if (!engine || !prog) return;
    engine->program = prog;
    
    mf_state_reset(&engine->state, engine->program, &engine->arena);
    
    u32 state_count = prog->meta.state_count;
    engine->frame_index = 0;
    
    if (state_count > 0 && prog->state_table) {
        engine->state_buffers = MF_ARENA_PUSH(&engine->arena, mf_state_buffer, state_count);
        
        for (u32 i = 0; i < state_count; ++i) {
            mf_bin_state_link* link = &prog->state_table[i];
            mf_tensor* t_proto = &prog->tensors[link->read_reg];
            size_t bytes = mf_tensor_size_bytes(t_proto);
            if (bytes == 0) bytes = 4;

            engine->state_buffers[i].size = bytes;
            engine->state_buffers[i].buffer_a = malloc(bytes);
            engine->state_buffers[i].buffer_b = malloc(bytes);
            
            if (t_proto->data) {
                memcpy(engine->state_buffers[i].buffer_a, t_proto->data, bytes);
                memset(engine->state_buffers[i].buffer_b, 0, bytes);
            } else {
                memset(engine->state_buffers[i].buffer_a, 0, bytes);
                memset(engine->state_buffers[i].buffer_b, 0, bytes);
            }
        }
    }
}

void mf_engine_dispatch(mf_engine* engine, u32 count_x, u32 count_y) {
    if (!engine || !engine->program) return;
    
    // Reset error state
    engine->state.error_code = 0;
    
    engine->global_size[0] = count_y;
    engine->global_size[1] = count_x;
    engine->global_size[2] = 1;

    if (engine->state_buffers && engine->program->meta.state_count > 0) {
        bool is_even = (engine->frame_index % 2 == 0);
        for (u32 i = 0; i < engine->program->meta.state_count; ++i) {
            mf_bin_state_link* link = &engine->program->state_table[i];
            mf_state_buffer* buf = &engine->state_buffers[i];
            void* prev = is_even ? buf->buffer_a : buf->buffer_b;
            void* next = is_even ? buf->buffer_b : buf->buffer_a;
            
            mf_tensor* t_read = &engine->state.registers[link->read_reg];
            mf_tensor* t_write = &engine->state.registers[link->write_reg];
            
            if (t_read->flags & MF_TENSOR_OWNS_DATA) {
                engine->state.allocator->free(engine->state.allocator, t_read->data);
                t_read->flags &= ~MF_TENSOR_OWNS_DATA;
            }
            t_read->data = prev;
            t_read->capacity_bytes = buf->size;
             
            if (t_write->flags & MF_TENSOR_OWNS_DATA) {
                engine->state.allocator->free(engine->state.allocator, t_write->data);
                t_write->flags &= ~MF_TENSOR_OWNS_DATA;
            }
            t_write->data = next;
            t_write->capacity_bytes = buf->size;
        }
    }

    if (engine->backend.dispatch) {
        engine->backend.dispatch(engine->backend.state, engine->program, &engine->state, count_x, count_y);
    }
    engine->frame_index++;
}

int32_t mf_engine_find_register(mf_engine* engine, const char* name) {
    if (!engine || !engine->program) return -1;
    const mf_program* prog = engine->program;
    if (!prog->symbols) return -1;
    for (size_t i = 0; i < prog->meta.symbol_count; ++i) {
        if (strcmp(prog->symbols[i].name, name) == 0) return (int32_t)prog->symbols[i].register_idx;
    }
    return -1;
}

mf_tensor* mf_engine_map_tensor(mf_engine* engine, u16 reg_idx, mf_access_mode mode) {
    (void)mode;
    if (!engine || reg_idx >= engine->state.register_count) return NULL;
    return &engine->state.registers[reg_idx];
}

bool mf_engine_resize_tensor(mf_engine* engine, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim) {
    if (!engine) return false;
    // We don't have a VM here, but we can use a temporary context or just the allocator
    mf_exec_ctx tmp_ctx;
    mf_exec_ctx_init(&tmp_ctx, engine->state.registers, engine->state.register_count, engine->state.allocator);
    return mf_exec_ctx_resize_tensor(&tmp_ctx, tensor, new_shape, new_ndim);
}

mf_engine_error mf_engine_get_error(mf_engine* engine) {
    if (!engine) return MF_ENGINE_ERR_NONE;
    return (mf_engine_error)engine->state.error_code;
}

void mf_engine_iterate_registers(mf_engine* engine, mf_engine_register_cb cb, void* user_data) {
    if (!engine || !cb) return;
    for (size_t i = 0; i < engine->state.register_count; ++i) {
        mf_tensor* t = &engine->state.registers[i];
        const char* name = NULL;
        if (engine->program && engine->program->symbols) {
            for (size_t s = 0; s < engine->program->meta.symbol_count; ++s) {
                if (engine->program->symbols[s].register_idx == i) {
                    name = engine->program->symbols[s].name;
                    break;
                }
            }
        }
        cb((u16)i, name, t, user_data);
    }
}
