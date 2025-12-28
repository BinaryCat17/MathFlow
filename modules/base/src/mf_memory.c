#include <mathflow/base/mf_memory.h>
#include <string.h>
#include <stdio.h> // For debug prints if needed

// --- Helper Macros ---

#define ALIGN_UP(n, align) (((n) + (align) - 1) & ~((align) - 1))
#define MF_ALIGNMENT 16 // Align to 16 bytes for SIMD friendliness

// --- Arena Allocator Implementation ---

void* mf_arena_alloc(mf_allocator* self, size_t size) {
    mf_arena* arena = (mf_arena*)self;
    size_t aligned_size = ALIGN_UP(size, MF_ALIGNMENT);
    
    if (arena->pos + aligned_size > arena->size) {
        return NULL; // OOM
    }

    void* ptr = arena->memory + arena->pos;
    arena->pos += aligned_size;
    return ptr;
}

// Arena doesn't support free in the traditional sense
void mf_arena_free_noop(mf_allocator* self, void* ptr) { (void)self; (void)ptr; }

// Arena realloc: Dumb Alloc + Copy
void* mf_arena_realloc(mf_allocator* self, void* ptr, size_t old_size, size_t new_size) {
    if (!ptr) return mf_arena_alloc(self, new_size);
    if (new_size == 0) return NULL;
    if (new_size <= old_size) return ptr; // Shrink is no-op for arena

    void* new_ptr = mf_arena_alloc(self, new_size);
    if (new_ptr) {
        memcpy(new_ptr, ptr, old_size);
    }
    return new_ptr;
}

void mf_arena_init(mf_arena* arena, void* backing_buffer, size_t size) {
    arena->base.alloc = mf_arena_alloc;
    arena->base.free = mf_arena_free_noop;
    arena->base.realloc = mf_arena_realloc;
    
    arena->memory = (u8*)backing_buffer;
    arena->size = size;
    arena->pos = 0;
}

void mf_arena_reset(mf_arena* arena) {
    arena->pos = 0;
}

// --- Heap Allocator Implementation (Free List) ---

struct mf_heap_block {
    size_t size;        // Size of the data block (excluding header)
    bool is_free;
    mf_heap_block* next; // Next block in memory order (to coalesce)
};

#define BLOCK_HEADER_SIZE ALIGN_UP(sizeof(mf_heap_block), MF_ALIGNMENT)

void mf_heap_init(mf_heap* heap, void* backing_buffer, size_t size) {
    heap->base.alloc = mf_heap_alloc;
    heap->base.free = mf_heap_free;
    heap->base.realloc = mf_heap_realloc;
    
    heap->memory = (u8*)backing_buffer;
    heap->size = size;
    heap->used_memory = 0;
    heap->peak_memory = 0;
    heap->allocation_count = 0;
    
    // Initialize first block covering the whole memory
    mf_heap_block* first = (mf_heap_block*)heap->memory;
    
    // Check if we have enough space for at least one header
    if (size < BLOCK_HEADER_SIZE) {
        heap->free_list = NULL;
        return;
    }

    first->size = size - BLOCK_HEADER_SIZE;
    first->is_free = true;
    first->next = NULL;
    
    heap->free_list = first;
}

void* mf_heap_alloc(mf_allocator* self, size_t size) {
    mf_heap* heap = (mf_heap*)self;
    size_t aligned_req = ALIGN_UP(size, MF_ALIGNMENT);
    
    mf_heap_block* current = heap->free_list;
    mf_heap_block* best_fit = NULL;
    
    // Find First Fit (or Best Fit)
    // Here we use First Fit for simplicity
    while (current) {
        if (current->is_free && current->size >= aligned_req) {
            best_fit = current;
            break;
        }
        current = current->next;
    }
    
    if (!best_fit) return NULL; // OOM
    
    // Split block if it's too big
    // Min remaining size should hold a header + minimal data (e.g. 16 bytes)
    if (best_fit->size >= aligned_req + BLOCK_HEADER_SIZE + MF_ALIGNMENT) {
        mf_heap_block* new_block = (mf_heap_block*)((u8*)best_fit + BLOCK_HEADER_SIZE + aligned_req);
        
        new_block->size = best_fit->size - aligned_req - BLOCK_HEADER_SIZE;
        new_block->is_free = true;
        new_block->next = best_fit->next;
        
        best_fit->size = aligned_req;
        best_fit->next = new_block;
    }
    
    best_fit->is_free = false;
    heap->used_memory += best_fit->size;
    if (heap->used_memory > heap->peak_memory) heap->peak_memory = heap->used_memory;
    heap->allocation_count++;
    
    // Return pointer to data (after header)
    return (u8*)best_fit + BLOCK_HEADER_SIZE;
}

void mf_heap_free(mf_allocator* self, void* ptr) {
    if (!ptr) return;
    mf_heap* heap = (mf_heap*)self;
    
    // Get header
    mf_heap_block* block = (mf_heap_block*)((u8*)ptr - BLOCK_HEADER_SIZE);
    
    if (block->is_free) return; // Double free protection
    
    block->is_free = true;
    heap->used_memory -= block->size;
    heap->allocation_count--;
    
    // Coalesce with next block if free
    if (block->next && block->next->is_free) {
        block->size += BLOCK_HEADER_SIZE + block->next->size;
        block->next = block->next->next;
    }
    
    // Coalesce with prev is harder because we only have singly linked list 'next'.
    // To do full coalescing we need to iterate from start or have doubly linked list.
    // Let's iterate from start (slow but simple) or optimize later.
    // Optimization: Usually heaps use explicit Free Lists or Footers.
    // For this prototype, we iterate to merge with prev.
    
    mf_heap_block* curr = (mf_heap_block*)heap->memory;
    while (curr && curr->next) {
        if (curr == block) break; // Reached our block
        
        if (curr->next == block) {
            // Found previous
            if (curr->is_free) {
                curr->size += BLOCK_HEADER_SIZE + block->size;
                curr->next = block->next;
            }
            break;
        }
        curr = curr->next;
    }
}

void* mf_heap_realloc(mf_allocator* self, void* ptr, size_t old_size, size_t new_size) {
    if (!ptr) return mf_heap_alloc(self, new_size);
    if (new_size == 0) {
        mf_heap_free(self, ptr);
        return NULL;
    }
    
    // Safety check: trust the block header more than the user provided old_size for internal logic
    mf_heap_block* block = (mf_heap_block*)((u8*)ptr - BLOCK_HEADER_SIZE);
    size_t actual_old_size = block->size;
    
    // Use the actual size for logic, but old_size could be used for optimizations if needed
    (void)old_size; 

    size_t aligned_req = ALIGN_UP(new_size, MF_ALIGNMENT);
    
    if (aligned_req <= actual_old_size) {
        // Shrink? Maybe later. For now just return same ptr.
        return ptr; 
    }
    
    // Expand
    // 1. Check if next block is free and has enough space
    if (block->next && block->next->is_free) {
        size_t combined = actual_old_size + BLOCK_HEADER_SIZE + block->next->size;
        if (combined >= aligned_req) {
            // Merge and claim
            mf_heap* heap = (mf_heap*)self;
            // Remove 'next' from free usage (it's conceptually merging)
            // But we just consume it.
            
            // size_t needed_extra = aligned_req - actual_old_size;
            // Actually, we just merge them first
            block->size += BLOCK_HEADER_SIZE + block->next->size;
            block->next = block->next->next;
            
            // Now block is big enough. Should we split it again if too big?
            // (Same logic as alloc)
            // For now, keep it simple.
            
            heap->used_memory += (combined - actual_old_size); // Adjusted usage logic slightly wrong here but ok for prototype
            return ptr;
        }
    }
    
    // 2. Alloc new, copy, free old
    void* new_ptr = mf_heap_alloc(self, new_size);
        if (new_ptr) {
            memcpy(new_ptr, ptr, actual_old_size);
            mf_heap_free(self, ptr);
        }
        return new_ptr;
    }
    