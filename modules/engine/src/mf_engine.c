#include <mathflow/engine/mf_engine.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void mf_engine_init(mf_engine* engine, const mf_engine_desc* desc) {
    memset(engine, 0, sizeof(mf_engine));
    
    size_t size = (desc && desc->arena_size > 0) ? desc->arena_size : MF_MB(8);
    
    engine->arena_buffer = malloc(size);
    mf_arena_init(&engine->arena, engine->arena_buffer, size);
    
    // Initialize default CPU backend
    // In the future, this could be configurable (GPU/Metal/Vulkan)
    mf_backend_cpu_init(&engine->backend);

    // Initialize Thread Pool
    mf_thread_pool_desc pool_desc = {
        .num_threads = desc ? desc->num_threads : 0,
        .init_fn = mf_vm_worker_init,
        .cleanup_fn = mf_vm_worker_cleanup,
        .user_data = NULL
    };
    engine->pool = mf_thread_pool_create(&pool_desc);
}

void mf_engine_bind_program(mf_engine* engine, mf_program* prog) {
    if (!engine || !prog) return;

    engine->program = prog;
    
    // Setup Context
    mf_context_init(&engine->ctx, engine->program, &engine->backend);
}

void mf_engine_shutdown(mf_engine* engine) {
    if (engine->pool) {
        mf_thread_pool_destroy(engine->pool);
        engine->pool = NULL;
    }

    if (engine->arena_buffer) {
        free(engine->arena_buffer);
        engine->arena_buffer = NULL;
    }
    memset(engine, 0, sizeof(mf_engine));
}

bool mf_engine_create_instance(mf_engine* engine, mf_instance* out_inst, size_t heap_size) {
    if (!engine || !engine->program || !out_inst) return false;
    
    if (heap_size == 0) heap_size = MF_MB(64);
    
    out_inst->heap_buffer = malloc(heap_size);
    if (!out_inst->heap_buffer) return false;
    
    mf_heap_init(&out_inst->heap, out_inst->heap_buffer, heap_size);
    
    mf_vm_init(&out_inst->vm, &engine->ctx, (mf_allocator*)&out_inst->heap);
    
    // Initial Reset to allocate registers
    // Note: We use the engine's arena for register metadata (it's small and static per instance usually)
    // Actually, mf_vm_reset allocates mf_tensor structs. If we do this multiple times, we might fill the arena?
    // The current mf_vm_reset design pushes to arena.
    // Ideally, registers should be in the heap or a separate small arena per instance if we want many instances.
    // For now, let's assume we use the main arena. If we run out, we need to increase arena size.
    // Optimization: Use a temp arena or heap for registers.
    // Let's use heap for registers? No, mf_vm_reset takes arena.
    // Workaround: We pass the engine arena. It must be big enough.
    mf_vm_reset(&out_inst->vm, &engine->arena);
    
    return true;
}

void mf_instance_destroy(mf_instance* inst) {
    if (!inst) return;
    
    mf_vm_shutdown(&inst->vm);
    
    if (inst->heap_buffer) {
        free(inst->heap_buffer);
        inst->heap_buffer = NULL;
    }
    memset(inst, 0, sizeof(mf_instance));
}