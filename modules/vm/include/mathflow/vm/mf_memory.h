#ifndef MF_MEMORY_H
#define MF_MEMORY_H

#include <mathflow/isa/mf_base.h>

// --- Allocator Interface ---

typedef struct mf_allocator mf_allocator;

struct mf_allocator {
    // Function pointers for polymorphism
    void* (*alloc)(mf_allocator* self, size_t size);
    void* (*realloc)(mf_allocator* self, void* ptr, size_t size);
    void  (*free)(mf_allocator* self, void* ptr);
};

// --- Arena Allocator (Linear / Frame Memory) ---
// Fast, no free(), reset() only.

typedef struct {
    mf_allocator base; // Inheritance
    u8* memory;
    size_t size;
    size_t pos;
} mf_arena;

void mf_arena_init(mf_arena* arena, void* backing_buffer, size_t size);
void* mf_arena_alloc(mf_allocator* self, size_t size); // Implements interface
void  mf_arena_reset(mf_arena* arena);

#define MF_ARENA_PUSH(arena, type, count) (type*)mf_arena_alloc((mf_allocator*)arena, sizeof(type) * (count))

// --- Heap Allocator (General Purpose) ---
// Supports alloc/free/realloc. Uses a Free List strategy.

typedef struct mf_heap_block mf_heap_block;

typedef struct {
    mf_allocator base;
    u8* memory;
    size_t size;
    mf_heap_block* free_list; // Head of the free blocks list
    
    // Stats
    size_t used_memory;       
    size_t peak_memory;
    size_t allocation_count;
} mf_heap;

void mf_heap_init(mf_heap* heap, void* backing_buffer, size_t size);
void* mf_heap_alloc(mf_allocator* self, size_t size);
void* mf_heap_realloc(mf_allocator* self, void* ptr, size_t size);
void  mf_heap_free(mf_allocator* self, void* ptr);

#endif // MF_MEMORY_H
