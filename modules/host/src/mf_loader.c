#include "mf_loader.h"
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/host/mf_host_desc.h>
#include <mathflow/base/mf_json.h>
#include <mathflow/base/mf_shape.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <stb_image.h>
#include <stb_truetype.h>

int mf_app_load_manifest(const char* path, mf_host_desc* out_desc);

void mf_loader_init_backend(mf_backend* backend, int num_threads) {
    if (!backend) return;
    mf_backend_cpu_init(backend, num_threads);
}

static mf_program* _load_binary(const char* path, mf_arena* arena) {
    size_t len = 0;
    u8* data = (u8*)mf_file_read_bin(path, &len);
    if (!data) return NULL;

    if (len < sizeof(mf_bin_header)) { free(data); return NULL; }
    mf_bin_header* head = (mf_bin_header*)data;
    if (head->magic != MF_BINARY_MAGIC || head->version != MF_BINARY_VERSION) {
        MF_LOG_ERROR("Invalid binary version or magic in %s.", path);
        free(data); return NULL;
    }

    mf_program* prog = MF_ARENA_PUSH(arena, mf_program, 1);
    prog->meta = *head;
    size_t offset = sizeof(mf_bin_header);
    
    // 1. Code
    prog->code = MF_ARENA_PUSH(arena, mf_instruction, head->instruction_count);
    memcpy(prog->code, data + offset, sizeof(mf_instruction) * head->instruction_count);
    offset += sizeof(mf_instruction) * head->instruction_count;

    // 2. Symbols
    if (head->symbol_count > 0) {
        prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, head->symbol_count);
        memcpy(prog->symbols, data + offset, sizeof(mf_bin_symbol) * head->symbol_count);
        offset += sizeof(mf_bin_symbol) * head->symbol_count;
    } else prog->symbols = NULL;

    // 3. Tasks
    if (head->task_count > 0) {
        prog->tasks = MF_ARENA_PUSH(arena, mf_task, head->task_count);
        memcpy(prog->tasks, data + offset, sizeof(mf_task) * head->task_count);
        offset += sizeof(mf_task) * head->task_count;
    } else prog->tasks = NULL;

    // 4. Task Bindings
    if (head->binding_count > 0) {
        prog->bindings = MF_ARENA_PUSH(arena, mf_bin_task_binding, head->binding_count);
        memcpy(prog->bindings, data + offset, sizeof(mf_bin_task_binding) * head->binding_count);
        offset += sizeof(mf_bin_task_binding) * head->binding_count;
    } else prog->bindings = NULL;

    // 5. Tensor Descriptors (Temporary array to find constant data later)
    mf_bin_tensor_desc* descs = (mf_bin_tensor_desc*)(data + offset);
    offset += sizeof(mf_bin_tensor_desc) * head->tensor_count;

    prog->tensor_infos = MF_ARENA_PUSH(arena, mf_type_info, head->tensor_count);
    prog->tensor_data = MF_ARENA_PUSH(arena, void*, head->tensor_count);
    prog->builtin_ids = MF_ARENA_PUSH(arena, uint8_t, head->tensor_count);
    prog->builtin_axes = MF_ARENA_PUSH(arena, uint8_t, head->tensor_count);
    prog->tensor_flags = MF_ARENA_PUSH(arena, uint8_t, head->tensor_count);
    
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* d = &descs[i];
        mf_type_info_init_contiguous(&prog->tensor_infos[i], (mf_dtype)d->dtype, d->shape, d->ndim);
        prog->builtin_ids[i] = d->builtin_id;
        prog->builtin_axes[i] = d->builtin_axis;
        prog->tensor_flags[i] = d->flags;
    }

    // 6. Constant Data
    for (u32 i = 0; i < head->tensor_count; ++i) {
        if (descs[i].is_constant) {
            size_t bytes = mf_shape_calc_bytes(prog->tensor_infos[i].dtype, prog->tensor_infos[i].shape, prog->tensor_infos[i].ndim);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            prog->tensor_data[i] = mem;
            offset += bytes;
        } else prog->tensor_data[i] = NULL;
    }

    free(data); return prog;
}

int mf_app_load_config(const char* path, mf_host_desc* out_desc) {
    if (!path || !out_desc) return -1;
    const char* ext = mf_path_get_ext(path);
    if (strcmp(ext, "bin") == 0 || strcmp(ext, "mfc") == 0) {
        FILE* f = fopen(path, "rb");
        if (!f) return -1;
        mf_bin_header head;
        if (fread(&head, sizeof(head), 1, f) != 1) { fclose(f); return -1; }
        fclose(f);
        if (head.magic != MF_BINARY_MAGIC || head.version != MF_BINARY_VERSION) return -1;
        out_desc->window_title = strdup(head.app_title[0] ? head.app_title : "MathFlow App");
        out_desc->width = head.window_width ? (int)head.window_width : 800;
        out_desc->height = head.window_height ? (int)head.window_height : 600;
        out_desc->resizable = head.resizable;
        out_desc->vsync = head.vsync;
        out_desc->fullscreen = head.fullscreen;
        out_desc->num_threads = (int)head.num_threads;
        out_desc->has_pipeline = true;
        out_desc->pipeline.kernel_count = 1;
        out_desc->pipeline.kernels = calloc(1, sizeof(mf_pipeline_kernel));
        out_desc->pipeline.kernels[0].id = strdup("main");
        out_desc->pipeline.kernels[0].graph_path = strdup(path);
        out_desc->pipeline.kernels[0].frequency = 1;
        return 0;
    }
    if (strcmp(ext, "json") == 0 || strcmp(ext, "mfapp") == 0) return mf_app_load_manifest(path, out_desc);
    return -3;
}

bool mf_loader_load_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe) {
    if (!engine || !pipe) return false;
    mf_engine_reset(engine);
    mf_arena* arena = mf_engine_get_arena(engine);
    mf_program** programs = malloc(sizeof(mf_program*) * pipe->kernel_count);
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        const char* path = pipe->kernels[i].graph_path;
        const char* ext = mf_path_get_ext(path);
        if (strcmp(ext, "json") == 0) {
            mf_compiler_diag diag; mf_compiler_diag_init(&diag, arena);
            mf_graph_ir ir = {0};
            if (!mf_compile_load_json(path, &ir, arena, &diag)) { free(programs); return false; }
            programs[i] = mf_compile(&ir, arena, &diag);
        } else programs[i] = _load_binary(path, arena);
        if (!programs[i]) { free(programs); return false; }
    }
    if (pipe->resource_count == 0) mf_engine_bind_cartridge(engine, programs, pipe->kernel_count);
    else mf_engine_bind_pipeline(engine, pipe, programs);
    free(programs); return true;
}

bool mf_loader_load_image(mf_engine* engine, const char* name, const char* path) {
    mf_tensor* t = mf_engine_map_resource(engine, name);
    if (!t) return false;
    int w, h, c, d = 0;
    if (t->info.ndim >= 3) d = t->info.shape[t->info.ndim - 1];
    unsigned char* data = stbi_load(path, &w, &h, &c, d);
    if (!data) return false;
    if (d == 0) d = c;
    int32_t sh[3]; uint8_t n = 0;
    if (d > 1) { sh[0] = h; sh[1] = w; sh[2] = d; n = 3; } else { sh[0] = h; sh[1] = w; n = 2; }
    if (!mf_engine_resize_resource(engine, name, sh, n)) { stbi_image_free(data); return false; }
    t = mf_engine_map_resource(engine, name);
    size_t p = (size_t)w * h * d;
    if (t->info.dtype == MF_DTYPE_F32) { f32* dst = (f32*)t->buffer->data; for (size_t i = 0; i < p; ++i) dst[i] = (f32)data[i] / 255.0f; }
    else if (t->info.dtype == MF_DTYPE_U8) memcpy(t->buffer->data, data, p);
    stbi_image_free(data); mf_engine_sync_resource(engine, name); return true;
}

static bool _bake_sdf(stbtt_fontinfo* f, int s, int e, u8* a, int aw, int ah, int* cx, int* cy, int l, f32* inf, int* c, float sc, int p, u8 edge, float dist) {
    for (int cp = s; cp < e; ++cp) {
        int g = stbtt_FindGlyphIndex(f, cp); if (g == 0) continue;
        int adv, lsb, gw, gh, xo, yo; stbtt_GetGlyphHMetrics(f, g, &adv, &lsb);
        u8* sdf = stbtt_GetGlyphSDF(f, sc, g, p, edge, dist, &gw, &gh, &xo, &yo); if (!sdf) continue;
        if (*cx + gw >= aw) { *cx = 0; *cy += l; } if (*cy + gh >= ah) { stbtt_FreeSDF(sdf, NULL); return false; }
        for (int y = 0; y < gh; ++y) memcpy(a + (*cy + y) * aw + *cx, sdf + y * gw, gw); stbtt_FreeSDF(sdf, NULL);
        int i = cp * 8; inf[i+0]=(f32)cp; inf[i+1]=(f32)*cx/aw; inf[i+2]=(f32)*cy/ah; inf[i+3]=(f32)(*cx+gw)/aw; inf[i+4]=(f32)(*cy+gh)/ah; inf[i+5]=(f32)adv*sc; inf[i+6]=(f32)xo; inf[i+7]=(f32)yo;
        (*c)++; *cx += gw + 1;
    }
    return true;
}

bool mf_loader_load_font(mf_engine* engine, const char* name, const char* path, float size) {
    size_t len; char* ttf = mf_file_read_bin(path, &len); if (!ttf) return false;
    stbtt_fontinfo f; if (!stbtt_InitFont(&f, (unsigned char*)ttf, 0)) { free(ttf); return false; }
    float sc = stbtt_ScaleForPixelHeight(&f, size);
    int aw = 512, ah = 512, pad = 2; u8* a = calloc(1, aw * ah);
    int mcp = 1200; f32* inf = calloc(mcp * 8, sizeof(f32));
    int ct = 0, cx = 0, cy = 0, cell = (int)(size * 1.5f);
    _bake_sdf(&f, 32, 127, a, aw, ah, &cx, &cy, cell, inf, &ct, sc, pad, 128, 32.0f);
    _bake_sdf(&f, 1024, 1104, a, aw, ah, &cx, &cy, cell, inf, &ct, sc, pad, 128, 32.0f);
    int32_t sh[] = { ah * aw }; if (mf_engine_resize_resource(engine, name, sh, 1)) {
        mf_tensor* t = mf_engine_map_resource(engine, name); for(size_t i=0; i<(size_t)aw*ah; ++i) ((f32*)t->buffer->data)[i] = (f32)a[i] / 255.0f;
        mf_engine_sync_resource(engine, name);
    }
    char in[128]; snprintf(in, 128, "%s_Info", name);
    int32_t ish[] = { mcp * 8 }; if (mf_engine_resize_resource(engine, in, ish, 1)) {
        mf_tensor* ti = mf_engine_map_resource(engine, in); memcpy(ti->buffer->data, inf, mcp * 8 * sizeof(f32));
        mf_engine_sync_resource(engine, in);
    }
    free(a); free(inf); free(ttf); return true;
}