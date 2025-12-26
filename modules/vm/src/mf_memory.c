#include <mathflow/vm/mf_memory.h>
#include <string.h>

// --- Arena Allocator ---

void mf_arena_init(mf_arena* arena, void* backing_buffer, size_t size) {
    arena->memory = (u8*)backing_buffer;
    arena->size = size;
    arena->pos = 0;
}

void* mf_arena_alloc(mf_arena* arena, size_t size) {
    // Align to 8 bytes
    size_t aligned_size = (size + 7) & ~7;
    
    if (arena->pos + aligned_size > arena->size) {
        return NULL; // Out of memory
    }

    void* ptr = arena->memory + arena->pos;
    arena->pos += aligned_size;
    
    // Optional: Zero memory? No, for performance we usually don't.
    return ptr;
}

void mf_arena_reset(mf_arena* arena) {
    arena->pos = 0;
}

// --- Column ---

void mf_column_init(mf_column* col, size_t stride, size_t initial_cap, mf_arena* arena) {
    col->stride = stride;
    col->count = 0;
    col->capacity = initial_cap > 0 ? initial_cap : 8;
    col->data = mf_arena_alloc(arena, col->stride * col->capacity);
}

void* mf_column_push(mf_column* col, void* item, mf_arena* arena) {
    if (col->count >= col->capacity) {
        size_t new_cap = col->capacity * 2;
        u8* new_data = mf_arena_alloc(arena, col->stride * new_cap);
        
        if (!new_data) return NULL;

        memcpy(new_data, col->data, col->count * col->stride);
        col->data = new_data;
        col->capacity = new_cap;
    }

    void* dest = col->data + (col->count * col->stride);
    if (item) {
        memcpy(dest, item, col->stride);
    } else {
        memset(dest, 0, col->stride);
    }
    col->count++;
    return dest;
}

void* mf_column_get(mf_column* col, size_t index) {
    if (index >= col->count) return NULL;
    return col->data + (index * col->stride);
}
