#include <mathflow/engine/mf_engine.h>
#include "mf_engine_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

mf_engine* mf_engine_create(const mf_engine_desc* desc) {
    mf_engine* engine = calloc(1, sizeof(mf_engine));
    if (!engine) return NULL;

    size_t arena_size = (desc && desc->arena_size > 0) ? desc->arena_size : MF_MB(8);
    size_t heap_size = (desc && desc->heap_size > 0) ? desc->heap_size : MF_MB(64);

    // 1. Arena
    engine->arena_buffer = malloc(arena_size);
    if (!engine->arena_buffer) {
        free(engine);
        return NULL;
    }
    mf_arena_init(&engine->arena, engine->arena_buffer, arena_size);

    // 2. Heap
    engine->heap_buffer = malloc(heap_size);
    if (!engine->heap_buffer) {
        free(engine->arena_buffer);
        free(engine);
        return NULL;
    }
    mf_heap_init(&engine->heap, engine->heap_buffer, heap_size);

    // 3. Backend (Injection)
    if (desc) {
        engine->backend = desc->backend;
    }

    // 4. VM (Initialize without context)
    mf_vm_init(&engine->vm, (mf_allocator*)&engine->heap);

    return engine;
}

void mf_engine_destroy(mf_engine* engine) {
    if (!engine) return;

    mf_engine_reset(engine); // Clean up state buffers first
    
    // Shutdown Backend
    if (engine->backend.shutdown) {
        engine->backend.shutdown(engine->backend.state);
    }

    if (engine->heap_buffer) free(engine->heap_buffer);
    if (engine->arena_buffer) free(engine->arena_buffer);
    
    free(engine);
}

void mf_engine_reset(mf_engine* engine) {
    if (!engine) return;

    // 1. Clean State Buffers
    if (engine->state_buffers && engine->program) {
        for (u32 i = 0; i < engine->program->meta.state_count; ++i) {
            free(engine->state_buffers[i].buffer_a);
            free(engine->state_buffers[i].buffer_b);
        }
        // Array itself is in Arena, so it will be freed below
        engine->state_buffers = NULL;
    }

    // 2. Shutdown VM (clears registers pointer, etc)
    mf_vm_shutdown(&engine->vm);
    
    // 3. Reset Allocators
    // Arena reset is fast (pos = 0)
    mf_arena_reset(&engine->arena);
    
    // Heap reset: just re-initialize the free list
    // This is valid because we own the buffer
    if (engine->heap_buffer) {
        mf_heap_init(&engine->heap, engine->heap_buffer, engine->heap.size);
    }
    
    // 4. Clear Program State
    engine->program = NULL;
    
    // 5. Re-init VM (restore allocator pointers)
    mf_vm_init(&engine->vm, (mf_allocator*)&engine->heap);
}

mf_arena* mf_engine_get_arena(mf_engine* engine) {
    if (!engine) return NULL;
    return &engine->arena;
}

void mf_engine_bind_program(mf_engine* engine, mf_program* prog) {
    if (!engine || !prog) return;

    engine->program = prog;
    
    // Setup Context - Removed
    // mf_context_init(&engine->ctx, engine->program, &engine->backend);
    
    // Reset VM (Allocates registers in the Heap using the Arena for descriptors)
    // Pass program so VM can allocate registers
    mf_vm_reset(&engine->vm, engine->program, &engine->arena);
    
    // Alloc State Buffers (Double Buffering)
    u32 state_count = prog->meta.state_count;
    engine->frame_index = 0;
    
    if (state_count > 0 && prog->state_table) {
        engine->state_buffers = MF_ARENA_PUSH(&engine->arena, mf_state_buffer, state_count);
        
        for (u32 i = 0; i < state_count; ++i) {
            mf_bin_state_link* link = &prog->state_table[i];
            
            if (link->read_reg >= prog->meta.tensor_count) continue;
            mf_tensor* t_proto = &prog->tensors[link->read_reg];
            
            size_t bytes = mf_tensor_size_bytes(t_proto);
            if (bytes == 0) bytes = 4; // Safety

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

// --- Dispatch Bridge ---

void mf_engine_dispatch(
    mf_engine* engine, 
    u32 count_x, u32 count_y
) {
    if (!engine) return;
    
    // Update Global Size (for Resolution Ops)
    engine->vm.global_size[0] = count_y; // H
    engine->vm.global_size[1] = count_x; // W
    engine->vm.global_size[2] = 1;

    // Apply Double Buffering (Ping-Pong)
    if (engine->state_buffers && engine->program && engine->program->meta.state_count > 0) {
        bool is_even = (engine->frame_index % 2 == 0);
        
        for (u32 i = 0; i < engine->program->meta.state_count; ++i) {
            mf_bin_state_link* link = &engine->program->state_table[i];
            mf_state_buffer* buf = &engine->state_buffers[i];
            
            // Swap Buffers
            void* prev = is_even ? buf->buffer_a : buf->buffer_b;
            void* next = is_even ? buf->buffer_b : buf->buffer_a;
            
            mf_tensor* t_read = &engine->vm.registers[link->read_reg];
            mf_tensor* t_write = &engine->vm.registers[link->write_reg];
            
            // Update Read Register
            if (t_read->flags & MF_TENSOR_OWNS_DATA) {
                if (engine->vm.allocator) engine->vm.allocator->free(engine->vm.allocator, t_read->data);
                t_read->flags &= ~MF_TENSOR_OWNS_DATA;
            }
            t_read->data = prev;
            t_read->capacity_bytes = buf->size;
             
            // Update Write Register
            if (t_write->flags & MF_TENSOR_OWNS_DATA) {
                if (engine->vm.allocator) engine->vm.allocator->free(engine->vm.allocator, t_write->data);
                t_write->flags &= ~MF_TENSOR_OWNS_DATA;
            }
            t_write->data = next;
            t_write->capacity_bytes = buf->size;
        }
    }

    if (engine->backend.dispatch) {
        engine->backend.dispatch(
            engine->backend.state,
            engine->program,
            &engine->vm,
            count_x, count_y
        );
    }
    
    engine->frame_index++;
}

// --- State Access ---

int32_t mf_engine_find_register(mf_engine* engine, const char* name) {
    if (!engine || !engine->program) return -1;
    
    const mf_program* prog = engine->program;
    if (!prog->symbols) return -1;

    for (size_t i = 0; i < prog->meta.symbol_count; ++i) {
        if (strcmp(prog->symbols[i].name, name) == 0) {
            return (int32_t)prog->symbols[i].register_idx;
        }
    }
    return -1;
}

mf_tensor* mf_engine_map_tensor(mf_engine* engine, u16 reg_idx, mf_access_mode mode) {
    if (!engine) return NULL;
    return mf_vm_map_tensor(&engine->vm, reg_idx, mode);
}

bool mf_engine_resize_tensor(mf_engine* engine, mf_tensor* tensor, const int32_t* new_shape, uint8_t new_ndim) {
    if (!engine) return false;
    return mf_vm_resize_tensor(&engine->vm, tensor, new_shape, new_ndim);
}

mf_engine_error mf_engine_get_error(mf_engine* engine) {
    if (!engine) return MF_ENGINE_ERR_NONE;
    switch (engine->vm.error) {
        case MF_ERROR_NONE: return MF_ENGINE_ERR_NONE;
        case MF_ERROR_OOM: return MF_ENGINE_ERR_OOM;
        case MF_ERROR_SHAPE_MISMATCH: return MF_ENGINE_ERR_SHAPE;
        case MF_ERROR_INVALID_OP: return MF_ENGINE_ERR_INVALID_OP;
        default: return MF_ENGINE_ERR_INVALID_OP;
    }
}

void mf_engine_iterate_registers(mf_engine* engine, mf_engine_register_cb cb, void* user_data) {
    if (!engine || !cb) return;

    for (size_t i = 0; i < engine->vm.register_count; ++i) {
        mf_tensor* t = &engine->vm.registers[i];
        const char* name = NULL;
        
        if (engine->program && engine->program->symbols) {
             const mf_program* prog = engine->program;
            for (size_t s = 0; s < prog->meta.symbol_count; ++s) {
                if (prog->symbols[s].register_idx == i) {
                    name = prog->symbols[s].name;
                    break;
                }
            }
        }
        cb((u16)i, name, t, user_data);
    }
}