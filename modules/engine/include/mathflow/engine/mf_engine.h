#ifndef MF_ENGINE_H
#define MF_ENGINE_H

#include <mathflow/isa/mf_tensor.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/vm/mf_vm.h>

// Configuration for initializing the engine
typedef struct mf_engine_desc {
    // Size of the static arena for Program Code & Metadata.
    // Default: 8MB if set to 0.
    size_t arena_size;
} mf_engine_desc;

// The MathFlow Engine
// Represents the "Static" part of the runtime:
// - Code (Program)
// - Constants (Arena)
// - Backend Operations (Dispatch Table)
// - Context (Symbols)
//
// It does NOT contain the execution state (Heap, Registers, Memory Nodes).
// Those are managed by mf_vm (single-threaded) or mf_scheduler (multi-threaded).
typedef struct mf_engine {
    // Memory
    mf_arena arena;
    void* arena_buffer; // Owned

    // Code
    mf_program* program;

    // Operations
    mf_backend_dispatch_table backend;

    // Shared Context (for VM/Scheduler)
    mf_context ctx;
} mf_engine;

// Initialize the engine. Allocates the Arena.
// Does NOT load a graph (use mf_engine_load_graph).
void mf_engine_init(mf_engine* engine, const mf_engine_desc* desc);

// Loads a graph from a file.
// Supports:
// - .json: Compiles source graph (requires mf_compiler)
// - .bin:  Loads binary program
// Returns true on success.
bool mf_engine_load_graph(mf_engine* engine, const char* path);

// Shuts down the engine and frees the Arena.
void mf_engine_shutdown(mf_engine* engine);

// --- Instance Management (Optional Helper) ---

// Represents a running instance of a program (VM + Memory)
typedef struct mf_instance {
    mf_vm vm;
    mf_heap heap;
    void* heap_buffer;
} mf_instance;

// Creates a new execution instance with its own Heap.
// This simplifies VM creation, handling memory allocation internally.
// @param heap_size Size of dynamic memory (Variables). Default: 64MB if 0.
bool mf_engine_create_instance(mf_engine* engine, mf_instance* out_inst, size_t heap_size);

// Destroys the instance and frees its memory.
void mf_instance_destroy(mf_instance* inst);

#endif // MF_ENGINE_H
