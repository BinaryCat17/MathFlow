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

    // 4. VM (Initialize but don't reset until bind)
    // Note: VM needs context, which is set in bind.
    // However, vm_init stores the pointers.
    mf_vm_init(&engine->vm, &engine->ctx, (mf_allocator*)&engine->heap);

    return engine;
}

void mf_engine_destroy(mf_engine* engine) {
    if (!engine) return;

    mf_vm_shutdown(&engine->vm);
    
    // Shutdown Backend
    if (engine->backend.shutdown) {
        engine->backend.shutdown(engine->backend.state);
    }

    if (engine->heap_buffer) free(engine->heap_buffer);
    if (engine->arena_buffer) free(engine->arena_buffer);
    
    free(engine);
}

mf_arena* mf_engine_get_arena(mf_engine* engine) {
    if (!engine) return NULL;
    return &engine->arena;
}

void mf_engine_bind_program(mf_engine* engine, mf_program* prog) {
    if (!engine || !prog) return;

    engine->program = prog;
    
    // Setup Context
    mf_context_init(&engine->ctx, engine->program, &engine->backend);
    
    // Reset VM (Allocates registers in the Heap using the Arena for descriptors)
    mf_vm_reset(&engine->vm, &engine->arena);
}

// --- Dispatch Bridge ---

void mf_engine_dispatch(
    mf_engine* engine, 
    u32 count_x, u32 count_y
) {
    if (!engine) return;
    
    // Smart Mode Selection
    // 1x1 -> Stateful Run on Main VM
    // NxM -> Stateless/Parallel Run on Backend
    if (count_x == 1 && count_y == 1) {
        // Stateful execution (Script Mode)
        // No setup/finish needed, the Host writes directly to engine registers via map_tensor
        mf_vm_exec(&engine->vm);
    } else {
        // Parallel execution (Shader Mode)
        // Delegates to Backend which will propagate state from the Main VM
        if (!engine->backend.dispatch) return;
        
        engine->backend.dispatch(
            engine->backend.state,
            &engine->ctx,
            &engine->vm,
            count_x, count_y
        );
    }
}

// --- State Access ---

int32_t mf_engine_find_register(mf_engine* engine, const char* name) {
    if (!engine || !engine->program) return -1;
    
    const mf_context* ctx = &engine->ctx;
    if (!ctx->symbols) return -1;

    for (size_t i = 0; i < ctx->symbol_count; ++i) {
        if (strcmp(ctx->symbols[i].name, name) == 0) {
            return (int32_t)ctx->symbols[i].register_idx;
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
        
        if (engine->ctx.symbols) {
            for (size_t s = 0; s < engine->ctx.symbol_count; ++s) {
                if (engine->ctx.symbols[s].register_idx == i) {
                    name = engine->ctx.symbols[s].name;
                    break;
                }
            }
        }
        cb((u16)i, name, t, user_data);
    }
}