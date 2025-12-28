#include <mathflow/host/mf_host_headless.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/host/mf_asset_loader.h>
#include <stdio.h>
#include <string.h>

static void print_tensor(u32 idx, const char* name, mf_tensor* t) {
    if (!t || !t->data) {
        printf("  [%u]%s: (Empty)\n", idx, name ? name : "");
        return;
    }
    
    printf("  [%u] ", idx);
    if (name) printf("'%s' ", name); 
    
    printf("Shape: [");
    for(int i=0; i<t->ndim; ++i) printf("%d%s", t->shape[i], i < t->ndim-1 ? "," : "");
    printf("] ");
    
    if (t->dtype == MF_DTYPE_F32) {
        printf("F32: {");
        f32* data = (f32*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) printf("%.2f%s", data[i], i < limit-1 ? ", " : "");
        if (t->size > limit) printf("... (+%zu)", t->size - limit);
        printf("}\n");
    } else if (t->dtype == MF_DTYPE_I32) {
        printf("I32: {");
        int32_t* data = (int32_t*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) printf("%d%s", data[i], i < limit-1 ? ", " : "");
        if (t->size > limit) printf("... (+%zu)", t->size - limit);
        printf("}\n");
    } else if (t->dtype == MF_DTYPE_U8) {
        printf("Bool: {");
        u8* data = (u8*)t->data;
        size_t limit = t->size > 16 ? 16 : t->size;
        for(size_t i=0; i<limit; ++i) printf("%s%s", data[i] ? "true" : "false", i < limit-1 ? ", " : "");
        printf("}\n");
    }
}

static void debug_print_callback(u16 idx, const char* name, mf_tensor* t, void* user_data) {
    // Simple filter: print first 20 registers or any named output
    if (idx < 20 || (name && strncmp(name, "out_", 4) == 0)) {
        print_tensor(idx, name, t);
    }
}

static void final_print_callback(u16 idx, const char* name, mf_tensor* t, void* user_data) {
    // Always print
    print_tensor(idx, name, t);
}

int mf_host_run_headless(const mf_host_desc* desc, int frames) {
    if (!desc || !desc->graph_path) return 1;

    mf_engine* engine = mf_engine_create(NULL);
    if (!engine) return 1;

    if (!mf_asset_loader_load(engine, desc->graph_path)) {
        mf_engine_destroy(engine);
        return 1;
    }

    // No explicit instance creation needed. Engine owns the state.

    printf("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        // Dispatch 1x1 = Script Mode (Stateful)
        mf_engine_dispatch(engine, 1, 1);
        
        mf_engine_error err = mf_engine_get_error(engine);
        if (err != MF_ENGINE_ERR_NONE) {
            printf("Error: Runtime error %d\n", err);
            break;
        }
        
        if (f < 3) {
             printf("--- Frame %d ---\n", f);
             mf_engine_iterate_registers(engine, debug_print_callback, NULL);
        }
    }
    
    printf("\n--- Final State ---\n");
    mf_engine_iterate_registers(engine, final_print_callback, NULL);

    mf_engine_destroy(engine);
    return 0;
}
