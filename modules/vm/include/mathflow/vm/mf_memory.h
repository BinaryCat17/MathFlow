#ifndef MF_MEMORY_H
#define MF_MEMORY_H

#include <mathflow/isa/mf_base.h>

// --- Arena Allocator ---

typedef struct {
    u8* memory;
    size_t size;
    size_t pos;
} mf_arena;

void mf_arena_init(mf_arena* arena, void* backing_buffer, size_t size);
void* mf_arena_alloc(mf_arena* arena, size_t size);
void mf_arena_reset(mf_arena* arena);

#define MF_ARENA_PUSH(arena, type, count) (type*)mf_arena_alloc(arena, sizeof(type) * (count))

// --- Column (Dynamic Array) ---

typedef struct {
    u8* data;
    size_t count;
    size_t capacity;
    size_t stride; // Size of one element in bytes
} mf_column;

void mf_column_init(mf_column* col, size_t stride, size_t initial_cap, mf_arena* arena);
void* mf_column_push(mf_column* col, void* item, mf_arena* arena);
void* mf_column_get(mf_column* col, size_t index);

#endif // MF_MEMORY_H
