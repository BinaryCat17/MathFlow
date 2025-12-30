#include <mathflow/loader/mf_loader.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void mf_loader_init_backend(mf_backend* backend, int num_threads) {
    if (!backend) return;
    // Hardcoded to CPU for now, but this is the Injection Point for future backends.
    mf_backend_cpu_init(backend, num_threads);
}

static const char* get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}

static char* read_file_bin(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        MF_LOG_ERROR("Could not open file for reading: %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(length);
    if (!buffer) {
        MF_LOG_ERROR("Memory allocation failed for file buffer (%ld bytes): %s", length, path);
        fclose(f);
        return NULL;
    }
    if (fread(buffer, 1, length, f) != (size_t)length) {
        MF_LOG_ERROR("Failed to read file content: %s", path);
        free(buffer);
        fclose(f);
        return NULL;
    }
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
        MF_LOG_ERROR("Invalid binary version or magic.");
        free(data);
        return NULL;
    }

    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta = *head;

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
    
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + offset);
        offset += sizeof(mf_bin_tensor_desc);
        
        mf_tensor* t = &prog->tensors[i];
        t->info.dtype = (mf_dtype)desc->dtype;
        t->info.ndim = desc->ndim;
        memcpy(t->info.shape, desc->shape, sizeof(int32_t) * MF_MAX_DIMS);
        
        // Initialize strides? Or leave 0? Ops usually calculate strides on fly if 0, or we should init them.
        // Let's init them for safety.
        // TODO: mf_tensor_calc_strides(t); 
        // For now, simple standard layout:
        int32_t stride = 1;
        for (int k = t->info.ndim - 1; k >= 0; --k) {
            t->info.strides[k] = stride;
            stride *= t->info.shape[k];
        }
    }

    // Data Blob
    size_t desc_start_offset = sizeof(mf_bin_header) + 
                               sizeof(mf_instruction) * head->instruction_count +
                               sizeof(mf_bin_symbol) * head->symbol_count;

    size_t data_start_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * head->tensor_count;
    offset = data_start_offset;

    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_tensor* t = &prog->tensors[i];
        size_t this_desc_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * i;
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + this_desc_offset);
        
        if (desc->is_constant) {
            size_t bytes = mf_tensor_size_bytes(t);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            
            // Create buffer wrapper on arena
            mf_buffer* buf = MF_ARENA_PUSH(arena, mf_buffer, 1);
            mf_buffer_init_view(buf, mem, bytes);
            
            t->buffer = buf;
            t->byte_offset = 0;
            
            offset += bytes;
        } else {
            t->buffer = NULL;
            t->byte_offset = 0;
        }
    }

    free(data);
    return prog;
}

static mf_program* load_prog_from_file(mf_arena* arena, const char* path) {
    const char* ext = get_filename_ext(path);
    if (strcmp(ext, "json") == 0) {
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, arena)) {
            MF_LOG_ERROR("Failed to load/parse JSON: %s", path);
            return NULL;
        }
        return mf_compile(&ir, arena);
    } else if (strcmp(ext, "bin") == 0) {
        return _load_binary(path, arena);
    }
    return NULL;
}

static mf_pipeline_desc* mf_loader_synthesize_pipeline(const mf_program* prog, const char* kernel_id, mf_arena* arena) {
    u32 res_count = prog->meta.symbol_count;
    
    mf_pipeline_desc* pipe = MF_ARENA_PUSH(arena, mf_pipeline_desc, 1);
    pipe->resource_count = res_count;
    pipe->resources = MF_ARENA_PUSH(arena, mf_pipeline_resource, res_count);
    
    pipe->kernel_count = 1;
    pipe->kernels = MF_ARENA_PUSH(arena, mf_pipeline_kernel, 1);
    
    mf_pipeline_kernel* kernel = &pipe->kernels[0];
    kernel->id = kernel_id;
    kernel->frequency = 1;
    kernel->binding_count = res_count;
    kernel->bindings = MF_ARENA_PUSH(arena, mf_pipeline_binding, res_count);

    for (u32 i = 0; i < res_count; ++i) {
        mf_bin_symbol* sym = &prog->symbols[i];
        mf_tensor* t = &prog->tensors[sym->register_idx];
        
        // Resource
        pipe->resources[i].name = sym->name;
        pipe->resources[i].dtype = t->info.dtype;
        pipe->resources[i].ndim = t->info.ndim;
        memcpy(pipe->resources[i].shape, t->info.shape, sizeof(int32_t) * MF_MAX_DIMS);
        
        // Binding
        kernel->bindings[i].kernel_port = sym->name;
        kernel->bindings[i].global_resource = sym->name;
    }
    
    return pipe;
}

bool mf_loader_load_graph(mf_engine* engine, const char* path) {
    if (!engine || !path) return false;

    // 1. Reset & Load Program
    mf_engine_reset(engine);
    mf_arena* arena = mf_engine_get_arena(engine);
    if (!arena) return false;

    mf_program* prog = load_prog_from_file(arena, path);
    if (!prog) return false;

    MF_LOG_INFO("Loader: Synthesizing Implicit Pipeline for %s", path);

    // 2. Synthesize & Bind
    mf_pipeline_desc* pipe = mf_loader_synthesize_pipeline(prog, "main", arena);
    
    mf_program* programs[] = { prog };
    mf_engine_bind_pipeline(engine, pipe, programs);

    return true;
}

bool mf_loader_load_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe) {
    if (!engine || !pipe) return false;
    
    // 1. Reset engine BEFORE loading anything into the arena
    mf_engine_reset(engine);

    mf_arena* arena = mf_engine_get_arena(engine);
    if (!arena) return false;

    MF_LOG_INFO("Loader: Loading Pipeline with %u kernels", pipe->kernel_count);

    mf_program** programs = malloc(sizeof(mf_program*) * pipe->kernel_count);
    
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        programs[i] = load_prog_from_file(arena, pipe->kernels[i].graph_path);
        if (!programs[i]) {
            MF_LOG_ERROR("Loader: Failed to load kernel program %s", pipe->kernels[i].graph_path);
            free(programs);
            return false;
        }
    }

    mf_engine_bind_pipeline(engine, pipe, programs);
    free(programs);

    return true;
}