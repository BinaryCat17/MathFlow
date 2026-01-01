#include "mf_loader.h"
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/base/mf_log.h>
#include <mathflow/base/mf_utils.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

void mf_loader_init_backend(mf_backend* backend, int num_threads) {
    if (!backend) return;
    mf_backend_cpu_init(backend, num_threads);
}

static mf_program* _load_binary(const char* path, mf_arena* arena) {
    size_t len = 0;
    char* data = mf_file_read_bin(path, &len);
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
    
    prog->code = MF_ARENA_PUSH(arena, mf_instruction, head->instruction_count);
    memcpy(prog->code, data + offset, sizeof(mf_instruction) * head->instruction_count);
    offset += sizeof(mf_instruction) * head->instruction_count;

    prog->symbols = MF_ARENA_PUSH(arena, mf_bin_symbol, head->symbol_count);
    memcpy(prog->symbols, data + offset, sizeof(mf_bin_symbol) * head->symbol_count);
    offset += sizeof(mf_bin_symbol) * head->symbol_count;

    prog->tasks = MF_ARENA_PUSH(arena, mf_task, head->task_count);
    if (head->task_count > 0) {
        memcpy(prog->tasks, data + offset, sizeof(mf_task) * head->task_count);
        offset += sizeof(mf_task) * head->task_count;
    }

    prog->tensors = MF_ARENA_PUSH(arena, mf_tensor, head->tensor_count);
    
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + offset);
        offset += sizeof(mf_bin_tensor_desc);
        
        mf_tensor* t = &prog->tensors[i];
        mf_type_info_init_contiguous(&t->info, (mf_dtype)desc->dtype, desc->shape, desc->ndim);
    }

    size_t desc_start_offset = sizeof(mf_bin_header) + 
                               sizeof(mf_instruction) * head->instruction_count +
                               sizeof(mf_bin_symbol) * head->symbol_count +
                               sizeof(mf_task) * head->task_count;

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
    const char* ext = mf_path_get_ext(path);
    if (strcmp(ext, "json") == 0) {
        mf_compiler_diag diag;
        mf_compiler_diag_init(&diag, arena);
        
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, arena, &diag)) return NULL;
        
        mf_program* prog = mf_compile(&ir, arena, &diag);
        return prog;
    } else if (strcmp(ext, "bin") == 0) {
        return _load_binary(path, arena);
    }
    return NULL;
}

static void _synthesize_raw_pipeline(mf_arena* arena, mf_pipeline_desc* out_pipe, mf_program** programs) {
    u32 total_ext = 0;
    for (u32 k = 0; k < out_pipe->kernel_count; ++k) {
        for (u32 s = 0; s < programs[k]->meta.symbol_count; ++s) {
            if (programs[k]->symbols[s].flags & (MF_SYMBOL_FLAG_INPUT | MF_SYMBOL_FLAG_OUTPUT)) {
                total_ext++;
            }
        }
    }

    mf_pipeline_resource* res = MF_ARENA_PUSH(arena, mf_pipeline_resource, total_ext);
    mf_pipeline_kernel* kernels_copy = MF_ARENA_PUSH(arena, mf_pipeline_kernel, out_pipe->kernel_count);
    memcpy(kernels_copy, out_pipe->kernels, sizeof(mf_pipeline_kernel) * out_pipe->kernel_count);

    u32 res_idx = 0;
    for (u32 k = 0; k < out_pipe->kernel_count; ++k) {
        mf_pipeline_kernel* pk = &kernels_copy[k];
        
        u32 k_ext = 0;
        for (u32 s = 0; s < programs[k]->meta.symbol_count; ++s) {
            if (programs[k]->symbols[s].flags & (MF_SYMBOL_FLAG_INPUT | MF_SYMBOL_FLAG_OUTPUT)) k_ext++;
        }
        
        pk->binding_count = 0;
        pk->bindings = MF_ARENA_PUSH(arena, mf_pipeline_binding, k_ext);

        for (u32 s = 0; s < programs[k]->meta.symbol_count; ++s) {
            mf_bin_symbol* sym = &programs[k]->symbols[s];
            if (!(sym->flags & (MF_SYMBOL_FLAG_INPUT | MF_SYMBOL_FLAG_OUTPUT))) continue;

            mf_tensor* t = &programs[k]->tensors[sym->register_idx];
            res[res_idx].name = sym->name;
            res[res_idx].dtype = t->info.dtype;
            res[res_idx].ndim = t->info.ndim;
            memcpy(res[res_idx].shape, t->info.shape, sizeof(int32_t) * MF_MAX_DIMS);

            pk->bindings[pk->binding_count].kernel_port = sym->name;
            pk->bindings[pk->binding_count].global_resource = sym->name;
            pk->binding_count++;
            res_idx++;
        }
    }

    out_pipe->resource_count = total_ext;
    out_pipe->resources = res;
    out_pipe->kernels = kernels_copy;
}

bool mf_loader_load_graph(mf_engine* engine, const char* path) {
    if (!engine || !path) return false;

    mf_pipeline_kernel ker = {0};
    ker.id = "main";
    ker.graph_path = path;
    ker.frequency = 1;

    mf_pipeline_desc pipe = {0};
    pipe.kernel_count = 1;
    pipe.kernels = &ker;
    pipe.resource_count = 0; // Trigger synthesis

    return mf_loader_load_pipeline(engine, &pipe);
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

    // Synthesize resources if none provided (Raw Graph mode)
    mf_pipeline_desc final_pipe = *pipe;
    if (pipe->resource_count == 0 && pipe->kernel_count > 0) {
        MF_LOG_DEBUG("Loader: Synthesizing resources for raw pipeline...");
        _synthesize_raw_pipeline(arena, &final_pipe, programs);
    }

    mf_engine_bind_pipeline(engine, &final_pipe, programs);
    free(programs);

    return true;
}

bool mf_loader_load_image(mf_engine* engine, const char* resource_name, const char* path) {
    if (!engine || !resource_name || !path) return false;

    mf_tensor* t = mf_engine_map_resource(engine, resource_name);
    if (!t) {
        MF_LOG_ERROR("Loader: Resource '%s' not found for image loading", resource_name);
        return false;
    }

    int w, h, c;
    // Force 4 channels (RGBA) for consistency if tensor implies it, but stbi is flexible.
    // Check tensor shape to decide desired channels
    int desired_channels = 0;
    if (t->info.ndim >= 3) {
        desired_channels = t->info.shape[t->info.ndim - 1]; // Last dim is usually channels
        if (desired_channels < 1 || desired_channels > 4) desired_channels = 0; // Fallback
    }

    unsigned char* data = stbi_load(path, &w, &h, &c, desired_channels);
    if (!data) {
        MF_LOG_ERROR("Loader: Failed to load image %s", path);
        return false;
    }

    if (desired_channels == 0) desired_channels = c;

    // Resize resource if needed (and if supported)
    // Assume Shape [H, W, C] or [H, W]
    // If resource is smaller/different, we should try to resize it.
    // Construct new shape
    int32_t new_shape[MF_MAX_DIMS];
    uint8_t new_ndim = 0;
    
    if (desired_channels > 1) {
        new_shape[0] = h;
        new_shape[1] = w;
        new_shape[2] = desired_channels;
        new_ndim = 3;
    } else {
        new_shape[0] = h;
        new_shape[1] = w;
        new_ndim = 2;
    }
    
    // Check compatibility
    // For now, we assume the resource MUST be resizable or match.
    // Try resize
    if (!mf_engine_resize_resource(engine, resource_name, new_shape, new_ndim)) {
        // If resize failed, check if current shape matches
        bool match = (t->info.ndim == new_ndim);
        for(int k=0; k<new_ndim; ++k) if(t->info.shape[k] != new_shape[k]) match = false;
        
        if (!match) {
            MF_LOG_ERROR("Loader: Image shape [%d,%d,%d] does not match resource '%s' and resize failed", h, w, desired_channels, resource_name);
            stbi_image_free(data);
            return false;
        }
    }
    
    // Re-map after resize (pointer might change if reallocated, though resize keeps tensor handle usually valid, 
    // but the buffer pointer inside might change)
    t = mf_engine_map_resource(engine, resource_name);

    // Copy and Convert
    size_t pixel_count = (size_t)w * h * desired_channels;
    
    if (t->info.dtype == MF_DTYPE_F32) {
        f32* dst = (f32*)t->buffer->data; // Offset is usually 0 for resources
        for (size_t i = 0; i < pixel_count; ++i) {
            dst[i] = (f32)data[i] / 255.0f;
        }
    } else if (t->info.dtype == MF_DTYPE_U8) {
        u8* dst = (u8*)t->buffer->data;
        memcpy(dst, data, pixel_count);
    } else {
        MF_LOG_ERROR("Loader: Unsupported dtype for image '%s'", resource_name);
        stbi_image_free(data);
        return false;
    }

    stbi_image_free(data);
    MF_LOG_INFO("Loader: Loaded image %s into '%s' [%dx%d]", path, resource_name, w, h);
    return true;
}

// Simple packing: Grid
static bool bake_range_sdf(stbtt_fontinfo* font, int start_char, int end_char, 
                          u8* atlas, int atlas_w, int atlas_h, 
                          int* current_x, int* current_y, int cell_size, 
                          f32* glyph_info_buffer, int* glyph_count, 
                          float scale, int padding, u8 onedge_value, float pixel_dist_scale) 
{
    for (int codepoint = start_char; codepoint < end_char; ++codepoint) {
        int g = stbtt_FindGlyphIndex(font, codepoint);
        if (g == 0) continue; // Missing glyph

        int advance, lsb;
        stbtt_GetGlyphHMetrics(font, g, &advance, &lsb);

        int gw, gh, xoff, yoff;
        u8* sdf = stbtt_GetGlyphSDF(font, scale, g, padding, onedge_value, pixel_dist_scale, &gw, &gh, &xoff, &yoff);
        
        if (!sdf) continue;

        // Wrap to next line
        if (*current_x + gw >= atlas_w) {
            *current_x = 0;
            *current_y += cell_size; // Move down by cell height (approx)
        }
        
        if (*current_y + gh >= atlas_h) {
            stbtt_FreeSDF(sdf, NULL);
            MF_LOG_ERROR("Loader: Font Atlas full!");
            return false;
        }

        // Blit
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                int dst_idx = (*current_y + y) * atlas_w + (*current_x + x);
                atlas[dst_idx] = sdf[y * gw + x];
            }
        }
        
        stbtt_FreeSDF(sdf, NULL);

        // Store Info
        // Format: [CodePoint, U0, V0, U1, V1, AdvanceX, OffsetX, OffsetY]
        // 8 floats per char.
        // Direct indexing: idx = codepoint * 8
        int idx = codepoint * 8;
        
        // UVs
        f32 u0 = (f32)(*current_x) / (f32)atlas_w;
        f32 v0 = (f32)(*current_y) / (f32)atlas_h;
        f32 u1 = (f32)(*current_x + gw) / (f32)atlas_w;
        f32 v1 = (f32)(*current_y + gh) / (f32)atlas_h;

        glyph_info_buffer[idx + 0] = (f32)codepoint;
        glyph_info_buffer[idx + 1] = u0;
        glyph_info_buffer[idx + 2] = v0;
        glyph_info_buffer[idx + 3] = u1;
        glyph_info_buffer[idx + 4] = v1;
        glyph_info_buffer[idx + 5] = (f32)advance * scale;
        glyph_info_buffer[idx + 6] = (f32)xoff;
        glyph_info_buffer[idx + 7] = (f32)yoff;

        (*glyph_count)++;
        *current_x += gw + 1; // 1px spacing
    }
    return true;
}

bool mf_loader_load_font(mf_engine* engine, const char* resource_name, const char* path, float font_size) {
    size_t len;
    char* ttf_buffer = mf_file_read_bin(path, &len);
    if (!ttf_buffer) return false;

    stbtt_fontinfo font;
    if (!stbtt_InitFont(&font, (unsigned char*)ttf_buffer, stbtt_GetFontOffsetForIndex((unsigned char*)ttf_buffer,0))) {
        MF_LOG_ERROR("Loader: Failed to init font %s", path);
        free(ttf_buffer);
        return false;
    }

    float scale = stbtt_ScaleForPixelHeight(&font, font_size);
    
    // Atlas Config
    int atlas_w = 512;
    int atlas_h = 512;
    int padding = 2; // SDF padding
    u8 onedge_value = 128;
    float pixel_dist_scale = 32.0f; // Softness

    u8* atlas_data = calloc(1, atlas_w * atlas_h);
    
    // Info Buffer (Large enough to hold up to Cyrillic range)
    int max_codepoint = 1200; 
    f32* glyph_info = calloc(max_codepoint * 8, sizeof(f32));
    int glyph_count = 0;
    int cur_x = 0;
    int cur_y = 0;
    int cell_h = (int)(font_size * 1.5f); // Rough line height

    // Bake ranges
    // ASCII
    bake_range_sdf(&font, 32, 127, atlas_data, atlas_w, atlas_h, &cur_x, &cur_y, cell_h, glyph_info, &glyph_count, scale, padding, onedge_value, pixel_dist_scale);
    // Cyrillic
    bake_range_sdf(&font, 1024, 1104, atlas_data, atlas_w, atlas_h, &cur_x, &cur_y, cell_h, glyph_info, &glyph_count, scale, padding, onedge_value, pixel_dist_scale);

    // Upload Atlas
    {
        mf_tensor* t = mf_engine_map_resource(engine, resource_name);
        if (!t) { 
            MF_LOG_ERROR("Loader: Resource '%s' not found for font atlas", resource_name);
            goto cleanup;
        }
        
        // Resize atlas to 1D
        int32_t shape[] = { atlas_h * atlas_w }; 
        if (!mf_engine_resize_resource(engine, resource_name, shape, 1)) {
             MF_LOG_ERROR("Loader: Failed to resize font atlas buffer");
             goto cleanup;
        }
        
        // Re-map
        t = mf_engine_map_resource(engine, resource_name);
        
        // Convert U8 -> F32 (0..1)
        size_t pixels = (size_t)atlas_w * atlas_h;
        f32* dst = (f32*)t->buffer->data;
        for(size_t i=0; i<pixels; ++i) dst[i] = (f32)atlas_data[i] / 255.0f;
    }

    // Upload Info
    {
        char info_name[128];
        snprintf(info_name, 128, "%s_Info", resource_name);
        
        mf_tensor* t_info = mf_engine_map_resource(engine, info_name);
        if (t_info) {
             // Resize info to 1D: [MaxCodepoint * 8]
             int32_t shape[] = { max_codepoint * 8 };
             if (mf_engine_resize_resource(engine, info_name, shape, 1)) {
                 t_info = mf_engine_map_resource(engine, info_name);
                 memcpy(t_info->buffer->data, glyph_info, max_codepoint * 8 * sizeof(f32));
             }
        } else {
            MF_LOG_WARN("Loader: Info resource '%s' not found. Font loaded but metadata lost.", info_name);
        }
    }

    MF_LOG_INFO("Loader: Loaded Font %s. Atlas: %dx%d, Glyphs: %d", path, atlas_w, atlas_h, glyph_count);

cleanup:
    free(atlas_data);
    free(glyph_info);
    free(ttf_buffer);
    return true;
}