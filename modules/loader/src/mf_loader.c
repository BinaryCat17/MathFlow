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
        t->dtype = (mf_dtype)desc->dtype;
        t->ndim = desc->ndim;
        memcpy(t->shape, desc->shape, sizeof(int32_t) * MF_MAX_DIMS);
        t->flags = 0; 
        
        t->size = 1;
        for(int k=0; k<t->ndim; ++k) t->size *= t->shape[k];
        if (t->ndim == 0) t->size = 1;
        
        t->capacity_bytes = 0;
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

bool mf_loader_load_graph(mf_engine* engine, const char* path) {
    if (!engine || !path) return false;

    // 1. Reset & Load Program
    mf_engine_reset(engine);
    mf_arena* arena = mf_engine_get_arena(engine);
    if (!arena) return false;

    mf_program* prog = load_prog_from_file(arena, path);
    if (!prog) return false;

    MF_LOG_INFO("Loader: Synthesizing Implicit Pipeline for %s", path);

    // 2. Inspect Symbols to create Resources
    // We need to count distinct resources. For a single graph, every symbol is a resource.
    u32 res_count = prog->meta.symbol_count;
    
    // Allocate temporary descriptors (on stack if small, else heap)
    mf_pipeline_resource* resources = calloc(res_count, sizeof(mf_pipeline_resource));
    mf_pipeline_binding* bindings = calloc(res_count, sizeof(mf_pipeline_binding));

    for (u32 i = 0; i < res_count; ++i) {
        mf_bin_symbol* sym = &prog->symbols[i];
        mf_tensor* t = &prog->tensors[sym->register_idx];
        
        // Resource Desc
        resources[i].name = sym->name; // Borrow pointer, bind_pipeline will strdup
        resources[i].dtype = t->dtype;
        resources[i].ndim = t->ndim;
        memcpy(resources[i].shape, t->shape, sizeof(int32_t) * MF_MAX_DIMS);
        
        // Binding Desc
        bindings[i].kernel_port = sym->name;
        bindings[i].global_resource = sym->name;
    }

    // 3. Create Kernel Desc
    mf_pipeline_kernel kernel = {0};
    kernel.id = "main";
    kernel.graph_path = path;
    kernel.frequency = 1;

    // Heuristic: If graph uses Spatial Ops (Index/Resolution), assume Spatial Domain.
    // Otherwise, assume Scalar (Script) Domain.
    // Phase 20: Removed domain field.
    // kernel.domain = MF_DOMAIN_SCALAR;
    // ... loop removed ...

    kernel.bindings = bindings;
    kernel.binding_count = res_count;

    // 4. Create Pipeline Desc
    mf_pipeline_desc pipe = {0};
    pipe.resources = resources;
    pipe.resource_count = res_count;
    pipe.kernels = &kernel;
    pipe.kernel_count = 1;

    // 5. Bind
    mf_program* programs[] = { prog };
    mf_engine_bind_pipeline(engine, &pipe, programs);

    // Cleanup temporary arrays
    free(resources);
    free(bindings);

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