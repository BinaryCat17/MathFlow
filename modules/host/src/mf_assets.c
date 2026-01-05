#include <mathflow/host/mf_host_desc.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>
#include <mathflow/base/mf_shape.h>
#include "mf_host_internal.h"
#include "mf_loader.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

bool mf_loader_load_image(mf_engine* engine, const char* name, const char* path) {
    mf_tensor* t = mf_engine_map_resource(engine, name);
    if (!t) return false;

    unsigned char* data = NULL;
    int w, h, c, d = 0;
    if (t->info.ndim >= 3) d = t->info.shape[t->info.ndim - 1];

    // Try loading from cartridge first
    size_t section_size = 0;
    void* section_data = mf_loader_find_section(name, MF_SECTION_IMAGE, &section_size);
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
    ttf = mf_loader_find_section(name, MF_SECTION_FONT, &section_size);
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