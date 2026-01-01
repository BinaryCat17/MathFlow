#include <mathflow/base/mf_buffer.h>
#include <mathflow/base/mf_log.h>
#include <string.h>

void mf_buffer_init_view(mf_buffer* buf, void* data, size_t size) {
    if (!buf) return;
    buf->data = data;
    buf->size_bytes = size;
    buf->alloc = NULL;
    buf->flags = 0;
    buf->ref_count = 1;
}

bool mf_buffer_alloc(mf_buffer* buf, mf_allocator* alloc, size_t size) {
    if (!buf || !alloc) return false;
    
    void* mem = alloc->alloc(alloc, size);
    if (!mem) {
        MF_LOG_ERROR("Buffer allocation failed for size %zu", size);
        return false;
    }
    
    // Zero init for safety
    memset(mem, 0, size);
    
    buf->data = mem;
    buf->size_bytes = size;
    buf->alloc = alloc;
    buf->flags = MF_BUFFER_OWNS_DATA;
    buf->ref_count = 1;
    
    return true;
}

void mf_buffer_free(mf_buffer* buf) {
    if (!buf) return;
    
    if ((buf->flags & MF_BUFFER_OWNS_DATA) && buf->alloc && buf->data) {
        buf->alloc->free(buf->alloc, buf->data);
    }
    
    buf->data = NULL;
    buf->size_bytes = 0;
    buf->alloc = NULL;
    buf->flags = 0;
    buf->ref_count = 0;
}
