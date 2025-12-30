#ifndef MF_BUFFER_H
#define MF_BUFFER_H

#include "mf_types.h"
#include "mf_memory.h"

// Flags for buffer properties
#define MF_BUFFER_OWNS_DATA (1 << 0) // Buffer is responsible for freeing 'data'
#define MF_BUFFER_GPU       (1 << 1) // Data resides in VRAM (future)
#define MF_BUFFER_PINNED    (1 << 2) // CPU memory pinned for DMA (future)

typedef struct {
    void* data;            // Pointer to raw memory
    size_t size_bytes;     // Total allocated size
    
    mf_allocator* alloc;   // Allocator used for this buffer (ref, not owned)
    u32 flags;
    u32 ref_count;         // For shared ownership (future proofing)
} mf_buffer;

// Initialize a buffer from existing memory (does not own data)
void mf_buffer_init_view(mf_buffer* buf, void* data, size_t size);

// Allocate a new buffer (owns data)
// Returns false on allocation failure
bool mf_buffer_alloc(mf_buffer* buf, mf_allocator* alloc, size_t size);

// Free buffer memory if it owns it. Does not free the 'mf_buffer' struct itself.
void mf_buffer_free(mf_buffer* buf);

#endif // MF_BUFFER_H
