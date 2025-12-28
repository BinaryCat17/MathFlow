#include <mathflow/host/mf_asset_loader.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/engine/mf_engine.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

static char* read_file_bin(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(length);
    if (!buffer) return NULL;
    fread(buffer, 1, length, f);
    fclose(f);
    if (out_len) *out_len = length;
    return buffer;
}

static mf_program* _load_binary(const char* path, mf_arena* arena) {
    size_t len = 0;
    char* data = read_file_bin(path, &len);
    if (!data) return NULL;

    if (len < sizeof(mf_bin_header)) { free(data); return NULL; }
    
    mf_bin_header* head = (mf_bin_header*)data;
    if (head->magic != MF_BINARY_MAGIC || head->version != MF_BINARY_VERSION) {
        printf("[AssetLoader] Invalid binary version or magic.\n");
        free(data);
        return NULL;
    }

    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta = *head;

    // Calc offsets
    size_t offset = sizeof(mf_bin_header);
    
    // Code
    prog->code = MF_ARENA_PUSH(arena, mf_instruction, head->instruction_count);
    memcpy(prog->code, data + offset, sizeof(mf_instruction) * head->instruction_count);
    offset += sizeof(mf_instruction) * head->instruction_count;

    // Symbol Table
    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, head->symbol_count);
    memcpy(prog->symbols, data + offset, sizeof(mf_bin_symbol) * head->symbol_count);
    offset += sizeof(mf_bin_symbol) * head->symbol_count;

    // Tensor Descriptors
    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, head->tensor_count);
    
    // Read descriptors (First pass to set up structs)
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + offset);
        offset += sizeof(mf_bin_tensor_desc);
        
        mf_tensor* t = &prog->tensors[i];
        t->dtype = (mf_dtype)desc->dtype;
        t->ndim = desc->ndim;
        memcpy(t->shape, desc->shape, sizeof(int32_t) * MF_MAX_DIMS);
        t->flags = 0; 
        
        t->size = 1;
        for(int k=0; k<t->ndim; ++k) t->size *= t->shape[k];
        if (t->ndim == 0) t->size = 1; // Scalar
        
        t->capacity_bytes = 0; // Program descriptors don't own memory yet
    }

    // Read Data Blob
    // Reset offset to iterate again to find constants data
    size_t desc_start_offset = sizeof(mf_bin_header) + 
                               sizeof(mf_instruction) * head->instruction_count +
                               sizeof(mf_bin_symbol) * head->symbol_count;
    
    // Data starts after all descriptors
    size_t data_start_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * head->tensor_count;
    offset = data_start_offset;

    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        
        // Re-read descriptor to check is_constant
        size_t this_desc_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * i;
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + this_desc_offset);
        
        if (desc->is_constant) {
            size_t bytes = mf_dtype_size(t->dtype) * t->size;
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            t->data = mem;
            t->capacity_bytes = bytes; 
            offset += bytes;
        } else {
            t->data = NULL;
            t->capacity_bytes = 0;
        }
    }

    free(data);
    return prog;
}

bool mf_asset_loader_load(mf_engine* engine, const char* path) {
    mf_arena* arena = mf_engine_get_arena(engine);
    if (!arena) return false;
    
    const char* ext = get_filename_ext(path);
    mf_program* prog = NULL;

    if (strcmp(ext, "json") == 0) {
        // Compile from source
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, arena)) {
            printf("[AssetLoader] Failed to load/parse JSON: %s\n", path);
            return false;
        }
        
        prog = mf_compile(&ir, arena);
    } else if (strcmp(ext, "bin") == 0) {
        // Load binary
        prog = _load_binary(path, arena);
    } else {
        printf("[AssetLoader] Unknown file extension: %s\n", ext);
        return false;
    }

    if (!prog) {
        printf("[AssetLoader] Failed to generate program.\n");
        return false;
    }

    // Bind program to engine
    // Since we don't have a public API for this yet in the engine refactor step,
    // we access the struct directly for now, OR we add the API to engine first.
    // Ideally, we should add mf_engine_bind_program in the next step.
    // For now, I'll assume direct access or use the function I'm about to add.
    
    // Use direct access for this intermediate step, will be cleaned up when engine is updated.
    // Actually, I'll update engine first in the next tool call, but I'm writing this file now.
    // I will use a hypothetical mf_engine_bind_program function and implement it in engine next.
    mf_engine_bind_program(engine, prog);
    
    return true;
}
