#ifndef MF_ENGINE_INTERNAL_H
#define MF_ENGINE_INTERNAL_H

#include <mathflow/engine/mf_engine.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_dispatch_table.h>

// The concrete implementation of the Engine.
// Combines Static Resources (Code) and Execution State (Data).
struct mf_engine {
    // Static Resources
    mf_arena arena;
    void* arena_buffer;
    mf_program* program;
    mf_backend_dispatch_table backend;
    mf_context ctx;

    // Execution State (Single Source of Truth)
    mf_vm vm;
    mf_heap heap;
    void* heap_buffer;
};

#endif // MF_ENGINE_INTERNAL_H