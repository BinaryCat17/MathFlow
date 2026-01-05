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

static char g_current_cartridge_path[512] = {0};
static mf_cartridge_header g_current_cart = {0};

void mf_loader_init_backend(mf_backend* backend, int num_threads) {
    if (!backend) return;
    mf_backend_cpu_init(backend, num_threads);
}

static mf_program* _load_program_from_mem(const u8* data, size_t len, mf_arena* arena) {
    if (len < sizeof(mf_bin_header)) return NULL;
    
    mf_bin_header* head = (mf_bin_header*)data;
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

    // 5. Tensor Descriptors
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

    return prog;
}

static void* _find_section_in_cartridge(const char* name, mf_section_type type, size_t* out_size) {
    if (!g_current_cartridge_path[0]) return NULL;
    
    // Check if we need to load/refresh the header cache
    if (g_current_cart.magic != MF_BINARY_MAGIC) {
        FILE* f = fopen(g_current_cartridge_path, "rb");
        if (!f) return NULL;
        if (fread(&g_current_cart, sizeof(mf_cartridge_header), 1, f) != 1) { 
            fclose(f); memset(&g_current_cart, 0, sizeof(g_current_cart)); return NULL; 
        }
        fclose(f);
    }

    void* result = NULL;
    for (u32 i = 0; i < g_current_cart.section_count; ++i) {
        if (g_current_cart.sections[i].type == (u32)type && strcmp(g_current_cart.sections[i].name, name) == 0) {
            FILE* f = fopen(g_current_cartridge_path, "rb");
            if (!f) return NULL;
            fseek(f, g_current_cart.sections[i].offset, SEEK_SET);
            result = malloc(g_current_cart.sections[i].size);
            if (fread(result, 1, g_current_cart.sections[i].size, f) != g_current_cart.sections[i].size) {
                free(result); result = NULL;
            } else {
                *out_size = g_current_cart.sections[i].size;
            }
            fclose(f);
            break;
        }
    }

    return result;
}

int mf_app_load_config(const char* path, mf_host_desc* out_desc) {
    if (!path || !out_desc) return -1;
    const char* ext = mf_path_get_ext(path);
    if (strcmp(ext, "bin") == 0 || strcmp(ext, "mfc") == 0) {
        strncpy(g_current_cartridge_path, path, 511);
        memset(&g_current_cart, 0, sizeof(g_current_cart)); // Clear cache

        FILE* f = fopen(path, "rb");
        if (!f) return -1;
        if (fread(&g_current_cart, sizeof(g_current_cart), 1, f) != 1) { fclose(f); return -1; }
        fclose(f);

        if (g_current_cart.magic != MF_BINARY_MAGIC || g_current_cart.version != MF_BINARY_VERSION) return -1;
        out_desc->window_title = strdup(g_current_cart.app_title[0] ? g_current_cart.app_title : "MathFlow App");
        out_desc->width = g_current_cart.window_width ? (int)g_current_cart.window_width : 800;
        out_desc->height = g_current_cart.window_height ? (int)g_current_cart.window_height : 600;
        out_desc->resizable = g_current_cart.resizable;
        out_desc->vsync = g_current_cart.vsync;
        out_desc->fullscreen = g_current_cart.fullscreen;
        out_desc->num_threads = (int)g_current_cart.num_threads;
        out_desc->has_pipeline = true;
        
        u32 prog_count = 0;
        for (u32 i = 0; i < g_current_cart.section_count; ++i) if (g_current_cart.sections[i].type == MF_SECTION_PROGRAM) prog_count++;
        out_desc->pipeline.kernel_count = prog_count;
        out_desc->pipeline.kernels = calloc(prog_count, sizeof(mf_pipeline_kernel));
        u32 current_prog = 0;
        for (u32 i = 0; i < g_current_cart.section_count; ++i) {
            if (g_current_cart.sections[i].type == MF_SECTION_PROGRAM) {
                out_desc->pipeline.kernels[current_prog].id = strdup(g_current_cart.sections[i].name);
                out_desc->pipeline.kernels[current_prog].graph_path = strdup(path); 
                out_desc->pipeline.kernels[current_prog].frequency = 1;
                current_prog++;
            }
        }
        return 0;
    }
    
    // Fallback: load as manifest or raw graph via compiler
    g_current_cartridge_path[0] = '\0';
    memset(&g_current_cart, 0, sizeof(g_current_cart));

    u8 temp_backing[1024 * 64]; 
    mf_arena temp_arena;
    mf_arena_init(&temp_arena, temp_backing, sizeof(temp_backing));

    mf_compiler_manifest manifest;
    if (mf_compiler_load_manifest(path, &manifest, &temp_arena)) {
        out_desc->window_title = strdup(manifest.app_ir.app_title[0] ? manifest.app_ir.app_title : "MathFlow App");
        out_desc->width = manifest.app_ir.window_width ? manifest.app_ir.window_width : 800;
        out_desc->height = manifest.app_ir.window_height ? manifest.app_ir.window_height : 600;
        out_desc->resizable = manifest.app_ir.resizable;
        out_desc->vsync = manifest.app_ir.vsync;
        out_desc->fullscreen = manifest.app_ir.fullscreen;
        out_desc->num_threads = manifest.app_ir.num_threads;
        out_desc->has_pipeline = true;
        out_desc->pipeline.kernel_count = manifest.kernel_count;
        out_desc->pipeline.kernels = calloc(manifest.kernel_count, sizeof(mf_pipeline_kernel));
        for (u32 i = 0; i < manifest.kernel_count; ++i) {
            out_desc->pipeline.kernels[i].id = strdup(manifest.kernels[i].id);
            out_desc->pipeline.kernels[i].graph_path = strdup(manifest.kernels[i].path);
            out_desc->pipeline.kernels[i].frequency = 1;
        }
        // Assets
        out_desc->asset_count = manifest.asset_count;
        out_desc->assets = calloc(manifest.asset_count, sizeof(mf_host_asset));
        for (u32 i = 0; i < manifest.asset_count; ++i) {
            out_desc->assets[i].resource_name = strdup(manifest.assets[i].name);
            out_desc->assets[i].path = strdup(manifest.assets[i].path);
            out_desc->assets[i].type = (manifest.assets[i].type == MF_SECTION_IMAGE) ? MF_ASSET_IMAGE : MF_ASSET_FONT;
            out_desc->assets[i].font_size = 32.0f;
        }

        return 0;
    }

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
        } else {
            // Load specific section from cartridge
            size_t len = 0;
            u8* data = (u8*)mf_file_read_bin(path, &len);
            if (!data) { free(programs); return false; }
            mf_cartridge_header* cart = (mf_cartridge_header*)data;
            programs[i] = NULL;
            for (u32 s = 0; s < cart->section_count; ++s) {
                if (cart->sections[s].type == MF_SECTION_PROGRAM && strcmp(cart->sections[s].name, pipe->kernels[i].id) == 0) {
                    programs[i] = _load_program_from_mem(data + cart->sections[s].offset, cart->sections[s].size, arena);
                    break;
                }
            }
            if (!programs[i]) {
                 for (u32 s = 0; s < cart->section_count; ++s) {
                    if (cart->sections[s].type == MF_SECTION_PROGRAM) {
                        programs[i] = _load_program_from_mem(data + cart->sections[s].offset, cart->sections[s].size, arena);
                        break;
                    }
                }
            }
            free(data);
        }
        if (!programs[i]) { free(programs); return false; }
    }
    if (pipe->resource_count == 0) {
        const char** names = malloc(sizeof(char*) * pipe->kernel_count);
        for (u32 i = 0; i < pipe->kernel_count; ++i) names[i] = pipe->kernels[i].id;
        mf_engine_bind_cartridge(engine, programs, names, pipe->kernel_count);
        free(names);
    } else {
        mf_engine_bind_pipeline(engine, pipe, programs);
    }
    free(programs); return true;
}

bool mf_loader_load_image(mf_engine* engine, const char* name, const char* path) {
    mf_tensor* t = mf_engine_map_resource(engine, name);
    if (!t) return false;

    unsigned char* data = NULL;
    int w, h, c, d = 0;
    if (t->info.ndim >= 3) d = t->info.shape[t->info.ndim - 1];

    // Try loading from cartridge first
    size_t section_size = 0;
    void* section_data = _find_section_in_cartridge(name, MF_SECTION_IMAGE, &section_size);
    if (section_data) {
        data = stbi_load_from_memory(section_data, (int)section_size, &w, &h, &c, d);
        free(section_data);
        if (data) MF_LOG_INFO("Loaded embedded image '%s' from cartridge.", name);
    }

    // Fallback to filesystem
    if (!data) {
        data = stbi_load(path, &w, &h, &c, d);
    }

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
    size_t len; 
    unsigned char* ttf = NULL;
    
    // Try cartridge first
    size_t section_size = 0;
    ttf = _find_section_in_cartridge(name, MF_SECTION_FONT, &section_size);
    if (ttf) {
        MF_LOG_INFO("Loaded embedded font '%s' from cartridge.", name);
    } else {
        ttf = (unsigned char*)mf_file_read_bin(path, &len);
    }

    if (!ttf) return false;
    stbtt_fontinfo f; if (!stbtt_InitFont(&f, ttf, 0)) { free(ttf); return false; }
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
