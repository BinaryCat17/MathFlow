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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

// --- Backend Initialization ---

void mf_loader_init_backend(mf_backend* backend, int num_threads) {
    if (!backend) return;
    mf_backend_cpu_init(backend, num_threads);
}

// --- Binary Loader ---

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

    prog->bindings = MF_ARENA_PUSH(arena, mf_bin_task_binding, head->binding_count);
    if (head->binding_count > 0) {
        memcpy(prog->bindings, data + offset, sizeof(mf_bin_task_binding) * head->binding_count);
        offset += sizeof(mf_bin_task_binding) * head->binding_count;
    }

    prog->tensor_infos = MF_ARENA_PUSH(arena, mf_type_info, head->tensor_count);
    prog->tensor_data = MF_ARENA_PUSH(arena, void*, head->tensor_count);
    prog->builtin_ids = MF_ARENA_PUSH(arena, uint8_t, head->tensor_count);
    prog->builtin_axes = MF_ARENA_PUSH(arena, uint8_t, head->tensor_count);
    prog->tensor_flags = MF_ARENA_PUSH(arena, uint8_t, head->tensor_count);
    
    for (u32 i = 0; i < head->tensor_count; ++i) {
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + offset);
        offset += sizeof(mf_bin_tensor_desc);
        
        mf_type_info_init_contiguous(&prog->tensor_infos[i], (mf_dtype)desc->dtype, desc->shape, desc->ndim);
        prog->builtin_ids[i] = desc->builtin_id;
        prog->builtin_axes[i] = desc->builtin_axis;
        prog->tensor_flags[i] = desc->flags;
    }

    size_t desc_start_offset = sizeof(mf_bin_header) + 
                               sizeof(mf_instruction) * head->instruction_count +
                               sizeof(mf_bin_symbol) * head->symbol_count +
                               sizeof(mf_task) * head->task_count +
                               sizeof(mf_bin_task_binding) * head->binding_count;

    size_t data_start_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * head->tensor_count;
    offset = data_start_offset;

    for (u32 i = 0; i < head->tensor_count; ++i) {
        size_t this_desc_offset = desc_start_offset + sizeof(mf_bin_tensor_desc) * i;
        mf_bin_tensor_desc* desc = (mf_bin_tensor_desc*)(data + this_desc_offset);
        
        if (desc->is_constant) {
            size_t bytes = mf_shape_calc_bytes(prog->tensor_infos[i].dtype, prog->tensor_infos[i].shape, prog->tensor_infos[i].ndim);
            void* mem = MF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            prog->tensor_data[i] = mem;
            offset += bytes;
        } else {
            prog->tensor_data[i] = NULL;
        }
    }

    free(data);
    return prog;
}

// --- Graph/Program Loading ---

static mf_program* load_prog_from_file(mf_arena* arena, const char* path) {
    const char* ext = mf_path_get_ext(path);
    if (strcmp(ext, "json") == 0) {
        mf_compiler_diag diag;
        mf_compiler_diag_init(&diag, arena);
        
        mf_graph_ir ir = {0};
        if (!mf_compile_load_json(path, &ir, arena, &diag)) return NULL;
        
        // Autonomous Compilation: Compiler uses metadata from JSON (or subgraphs)
        mf_program* prog = mf_compile(&ir, arena, &diag);
        
        if (!prog && diag.has_error) {
            for (u32 e = 0; e < diag.error_count; ++e) {
                MF_LOG_ERROR("Compiler: %s:%u:%u: %s", 
                    diag.errors[e].loc.file ? diag.errors[e].loc.file : "unknown",
                    diag.errors[e].loc.line, diag.errors[e].loc.column,
                    diag.errors[e].message);
            }
        }
        
        return prog;
    } else if (strcmp(ext, "bin") == 0) {
        return _load_binary(path, arena);
    }
    return NULL;
}

// --- Manifest Parsing (.mfapp) ---

int mf_app_load_config(const char* mfapp_path, mf_host_desc* out_desc) {
    if (!mfapp_path || !out_desc) return -1;

    const char* ext = mf_path_get_ext(mfapp_path);
    if (strcmp(ext, "json") == 0 || strcmp(ext, "bin") == 0) {
        MF_LOG_INFO("Host: Loading raw graph directly: %s", mfapp_path);
        
        out_desc->window_title = strdup("MathFlow Visualizer");
        out_desc->width = 800;
        out_desc->height = 600;
        out_desc->resizable = true;
        out_desc->vsync = true;
        out_desc->has_pipeline = true;
        
        out_desc->pipeline.kernel_count = 1;
        out_desc->pipeline.kernels = calloc(1, sizeof(mf_pipeline_kernel));
        out_desc->pipeline.kernels[0].id = strdup("main");
        out_desc->pipeline.kernels[0].graph_path = strdup(mfapp_path);
        out_desc->pipeline.kernels[0].frequency = 1;
        
        return 0;
    }

    size_t arena_size = 1 * 1024 * 1024;
    void* arena_mem = malloc(arena_size);
    mf_arena arena;
    mf_arena_init(&arena, arena_mem, arena_size);

    char* json_str = mf_file_read(mfapp_path, &arena);
    if (!json_str) {
        MF_LOG_ERROR("Could not read manifest %s", mfapp_path);
        free(arena_mem);
        return -1;
    }

    mf_json_value* root = mf_json_parse(json_str, &arena);
    if (!root || root->type != MF_JSON_VAL_OBJECT) {
        MF_LOG_ERROR("Failed to parse manifest JSON");
        free(arena_mem);
        return -2;
    }

    out_desc->num_threads = 0;
    out_desc->fullscreen = false;
    out_desc->resizable = true;
    out_desc->vsync = true;
    out_desc->width = 800;
    out_desc->height = 600;
    out_desc->window_title = "MathFlow App";
    out_desc->has_pipeline = false;
    memset(&out_desc->pipeline, 0, sizeof(out_desc->pipeline));

    char* base_dir = mf_path_get_dir(mfapp_path, &arena);

    const mf_json_value* runtime = mf_json_get_field(root, "runtime");
    if (runtime && runtime->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* threads = mf_json_get_field(runtime, "threads");
        if (threads && threads->type == MF_JSON_VAL_NUMBER) out_desc->num_threads = (u32)threads->as.n;
        
        const mf_json_value* entry = mf_json_get_field(runtime, "entry");
        if (entry && entry->type == MF_JSON_VAL_STRING && !mf_json_get_field(root, "pipeline")) {
            out_desc->has_pipeline = true;
            out_desc->pipeline.kernel_count = 1;
            out_desc->pipeline.kernels = calloc(1, sizeof(mf_pipeline_kernel));
            out_desc->pipeline.kernels[0].id = strdup("main");
            char* path = mf_path_join(base_dir, entry->as.s, &arena);
            out_desc->pipeline.kernels[0].graph_path = strdup(path);
            out_desc->pipeline.kernels[0].frequency = 1;
        }
    }

    const mf_json_value* window = mf_json_get_field(root, "window");
    if (window && window->type == MF_JSON_VAL_OBJECT) {
        const mf_json_value* title = mf_json_get_field(window, "title");
        if (title && title->type == MF_JSON_VAL_STRING) out_desc->window_title = strdup(title->as.s);

        const mf_json_value* w = mf_json_get_field(window, "width");
        if (w && w->type == MF_JSON_VAL_NUMBER) out_desc->width = (u32)w->as.n;

        const mf_json_value* h = mf_json_get_field(window, "height");
        if (h && h->type == MF_JSON_VAL_NUMBER) out_desc->height = (u32)h->as.n;

        const mf_json_value* resizable = mf_json_get_field(window, "resizable");
        if (resizable && resizable->type == MF_JSON_VAL_BOOL) out_desc->resizable = resizable->as.b;

        const mf_json_value* vsync = mf_json_get_field(window, "vsync");
        if (vsync && vsync->type == MF_JSON_VAL_BOOL) out_desc->vsync = vsync->as.b;

        const mf_json_value* fullscreen = mf_json_get_field(window, "fullscreen");
        if (fullscreen && fullscreen->type == MF_JSON_VAL_BOOL) out_desc->fullscreen = fullscreen->as.b;
    }

    const mf_json_value* pipeline = mf_json_get_field(root, "pipeline");
    if (pipeline && pipeline->type == MF_JSON_VAL_OBJECT) {
        out_desc->has_pipeline = true;
        
        const mf_json_value* resources = mf_json_get_field(pipeline, "resources");
        if (resources && resources->type == MF_JSON_VAL_ARRAY) {
            out_desc->pipeline.resource_count = (u32)resources->as.array.count;
            out_desc->pipeline.resources = calloc(out_desc->pipeline.resource_count, sizeof(mf_pipeline_resource));
            
            for (size_t i = 0; i < resources->as.array.count; ++i) {
                mf_pipeline_resource* pr = &out_desc->pipeline.resources[i];
                const mf_json_value* res = &resources->as.array.items[i];
                
                const mf_json_value* name = mf_json_get_field(res, "name");
                if (name && name->type == MF_JSON_VAL_STRING) pr->name = strdup(name->as.s);
                
                const mf_json_value* provider = mf_json_get_field(res, "provider");
                if (provider && provider->type == MF_JSON_VAL_STRING) pr->provider = strdup(provider->as.s);
                else pr->provider = NULL;

                const mf_json_value* dtype = mf_json_get_field(res, "dtype");
                if (dtype && dtype->type == MF_JSON_VAL_STRING) pr->dtype = mf_dtype_from_str(dtype->as.s);
                
                const mf_json_value* readonly = mf_json_get_field(res, "readonly");
                if (readonly && readonly->type == MF_JSON_VAL_BOOL && readonly->as.b) pr->flags |= MF_RESOURCE_FLAG_READONLY;

                const mf_json_value* shape = mf_json_get_field(res, "shape");
                if (shape && shape->type == MF_JSON_VAL_ARRAY) {
                    pr->ndim = (uint8_t)shape->as.array.count;
                    if (pr->ndim > MF_MAX_DIMS) pr->ndim = MF_MAX_DIMS;
                    for(int d=0; d < pr->ndim; ++d) {
                        const mf_json_value* dim = &shape->as.array.items[d];
                        if (dim->type == MF_JSON_VAL_NUMBER) pr->shape[d] = (int)dim->as.n;
                    }
                }
            }
        }

        const mf_json_value* kernels = mf_json_get_field(pipeline, "kernels");
        if (kernels && kernels->type == MF_JSON_VAL_ARRAY) {
            out_desc->pipeline.kernel_count = (u32)kernels->as.array.count;
            out_desc->pipeline.kernels = calloc(out_desc->pipeline.kernel_count, sizeof(mf_pipeline_kernel));

            for (size_t i = 0; i < kernels->as.array.count; ++i) {
                mf_pipeline_kernel* pk = &out_desc->pipeline.kernels[i];
                const mf_json_value* ker = &kernels->as.array.items[i];
                
                const mf_json_value* id = mf_json_get_field(ker, "id");
                if (id && id->type == MF_JSON_VAL_STRING) pk->id = strdup(id->as.s);

                const mf_json_value* entry = mf_json_get_field(ker, "entry");
                if (entry && entry->type == MF_JSON_VAL_STRING) {
                    char* path = mf_path_join(base_dir, entry->as.s, &arena);
                    pk->graph_path = strdup(path);
                }

                const mf_json_value* freq = mf_json_get_field(ker, "frequency");
                if (freq && freq->type == MF_JSON_VAL_NUMBER) pk->frequency = (u32)freq->as.n;
                else pk->frequency = 1;

                const mf_json_value* bindings = mf_json_get_field(ker, "bindings");
                if (bindings && bindings->type == MF_JSON_VAL_ARRAY) {
                    pk->binding_count = (u32)bindings->as.array.count;
                    pk->bindings = calloc(pk->binding_count, sizeof(mf_pipeline_binding));
                    
                    for (size_t b = 0; b < bindings->as.array.count; ++b) {
                        mf_pipeline_binding* pb = &pk->bindings[b];
                        const mf_json_value* bind = &bindings->as.array.items[b];
                        
                        const mf_json_value* port = mf_json_get_field(bind, "port");
                        const mf_json_value* res = mf_json_get_field(bind, "resource");
                        if (port && port->type == MF_JSON_VAL_STRING) pb->kernel_port = strdup(port->as.s);
                        if (res && res->type == MF_JSON_VAL_STRING) pb->global_resource = strdup(res->as.s);
                    }
                }
            }
        }
    }

    const mf_json_value* assets = mf_json_get_field(root, "assets");
    if (assets && assets->type == MF_JSON_VAL_ARRAY) {
        out_desc->asset_count = (int)assets->as.array.count;
        out_desc->assets = calloc(out_desc->asset_count, sizeof(mf_host_asset));
        
        for (size_t i = 0; i < assets->as.array.count; ++i) {
            mf_host_asset* pa = &out_desc->assets[i];
            const mf_json_value* asset = &assets->as.array.items[i];
            
            const mf_json_value* type = mf_json_get_field(asset, "type");
            if (type && type->type == MF_JSON_VAL_STRING) {
                if (strcmp(type->as.s, "image") == 0) pa->type = MF_ASSET_IMAGE;
                else if (strcmp(type->as.s, "font") == 0) pa->type = MF_ASSET_FONT;
            }
            
            const mf_json_value* res = mf_json_get_field(asset, "resource");
            if (res && res->type == MF_JSON_VAL_STRING) pa->resource_name = strdup(res->as.s);
            
            const mf_json_value* path = mf_json_get_field(asset, "path");
            if (path && path->type == MF_JSON_VAL_STRING) {
                char* full_path = mf_path_join(base_dir, path->as.s, &arena);
                pa->path = strdup(full_path);
            }
            
            const mf_json_value* size = mf_json_get_field(asset, "size");
            if (size && size->type == MF_JSON_VAL_NUMBER) pa->font_size = (float)size->as.n;
            else pa->font_size = 32.0f;
        }
    }

    free(arena_mem);
    return 0;
}

// --- Pipeline Loading ---

bool mf_loader_load_graph(mf_engine* engine, const char* path) {
    if (!engine || !path) return false;

    mf_pipeline_kernel ker = {0};
    ker.id = "main";
    ker.graph_path = path;
    ker.frequency = 1;

    mf_pipeline_desc pipe = {0};
    pipe.kernel_count = 1;
    pipe.kernels = &ker;
    pipe.resource_count = 0;

    return mf_loader_load_pipeline(engine, &pipe);
}

bool mf_loader_load_pipeline(mf_engine* engine, const mf_pipeline_desc* pipe) {
    if (!engine || !pipe) return false;
    
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

    // --- Automatic Resource Discovery (For Raw Graphs or Unspecified Resources) ---
    mf_pipeline_desc final_pipe = *pipe;
    if (pipe->resource_count == 0) {
        u32 discovery_count = 0;
        for (u32 k = 0; k < pipe->kernel_count; ++k) {
            discovery_count += programs[k]->meta.symbol_count;
        }
        
        final_pipe.resources = calloc(discovery_count, sizeof(mf_pipeline_resource));
        u32 actual_resources = 0;

        for (u32 k = 0; k < pipe->kernel_count; ++k) {
            mf_program* prog = programs[k];
            for (u32 s = 0; s < prog->meta.symbol_count; ++s) {
                mf_bin_symbol* sym = &prog->symbols[s];
                if (!(sym->flags & (MF_SYMBOL_FLAG_INPUT | MF_SYMBOL_FLAG_OUTPUT))) continue;

                // Check if already added
                bool exists = false;
                for (u32 r = 0; r < actual_resources; ++r) {
                    if (strcmp(final_pipe.resources[r].name, sym->name) == 0) { exists = true; break; }
                }
                if (exists) continue;

                mf_pipeline_resource* pr = &final_pipe.resources[actual_resources++];
                pr->name = sym->name;
                pr->dtype = prog->tensor_infos[sym->register_idx].dtype;
                pr->ndim = prog->tensor_infos[sym->register_idx].ndim;
                memcpy(pr->shape, prog->tensor_infos[sym->register_idx].shape, sizeof(int32_t) * MF_MAX_DIMS);
                if (sym->provider[0]) pr->provider = sym->provider;
            }
        }
        final_pipe.resource_count = actual_resources;
    }

    mf_engine_bind_pipeline(engine, &final_pipe, programs);
    
    if (pipe->resource_count == 0 && final_pipe.resources) {
        free(final_pipe.resources);
    }

    free(programs);
    return true;
}

// --- Asset Loading (Images/Fonts) ---

bool mf_loader_load_image(mf_engine* engine, const char* resource_name, const char* path) {
    if (!engine || !resource_name || !path) return false;

    mf_tensor* t = mf_engine_map_resource(engine, resource_name);
    if (!t) {
        MF_LOG_ERROR("Loader: Resource '%s' not found for image loading", resource_name);
        return false;
    }

    int w, h, c;
    int desired_channels = 0;
    if (t->info.ndim >= 3) {
        desired_channels = t->info.shape[t->info.ndim - 1];
        if (desired_channels < 1 || desired_channels > 4) desired_channels = 0;
    }

    unsigned char* data = stbi_load(path, &w, &h, &c, desired_channels);
    if (!data) {
        MF_LOG_ERROR("Loader: Failed to load image %s", path);
        return false;
    }

    if (desired_channels == 0) desired_channels = c;

    int32_t new_shape[MF_MAX_DIMS];
    uint8_t new_ndim = 0;
    
    if (desired_channels > 1) {
        new_shape[0] = h; new_shape[1] = w; new_shape[2] = desired_channels;
        new_ndim = 3;
    } else {
        new_shape[0] = h; new_shape[1] = w;
        new_ndim = 2;
    }
    
    if (!mf_engine_resize_resource(engine, resource_name, new_shape, new_ndim)) {
        bool match = (t->info.ndim == new_ndim);
        for(int k=0; k<new_ndim; ++k) if(t->info.shape[k] != new_shape[k]) match = false;
        if (!match) {
            MF_LOG_ERROR("Loader: Image shape [%dx%d] does not match resource '%s'", w, h, resource_name);
            stbi_image_free(data);
            return false;
        }
    }
    
    t = mf_engine_map_resource(engine, resource_name);
    size_t pixel_count = (size_t)w * h * desired_channels;
    
    if (t->info.dtype == MF_DTYPE_F32) {
        f32* dst = (f32*)t->buffer->data;
        for (size_t i = 0; i < pixel_count; ++i) dst[i] = (f32)data[i] / 255.0f;
    } else if (t->info.dtype == MF_DTYPE_U8) {
        u8* dst = (u8*)t->buffer->data;
        memcpy(dst, data, pixel_count);
    }

    stbi_image_free(data);
    mf_engine_sync_resource(engine, resource_name);
    MF_LOG_INFO("Loader: Loaded image %s into '%s' [%dx%d]", path, resource_name, w, h);
    return true;
}

static bool bake_range_sdf(stbtt_fontinfo* font, int start_char, int end_char, 
                          u8* atlas, int atlas_w, int atlas_h, 
                          int* current_x, int* current_y, int cell_size, 
                          f32* glyph_info_buffer, int* glyph_count, 
                          float scale, int padding, u8 onedge_value, float pixel_dist_scale) 
{
    for (int codepoint = start_char; codepoint < end_char; ++codepoint) {
        int g = stbtt_FindGlyphIndex(font, codepoint);
        if (g == 0) continue;
        int advance, lsb;
        stbtt_GetGlyphHMetrics(font, g, &advance, &lsb);
        int gw, gh, xoff, yoff;
        u8* sdf = stbtt_GetGlyphSDF(font, scale, g, padding, onedge_value, pixel_dist_scale, &gw, &gh, &xoff, &yoff);
        if (!sdf) continue;
        if (*current_x + gw >= atlas_w) { *current_x = 0; *current_y += cell_size; }
        if (*current_y + gh >= atlas_h) { stbtt_FreeSDF(sdf, NULL); return false; }
        for (int y = 0; y < gh; ++y) memcpy(atlas + (*current_y + y) * atlas_w + *current_x, sdf + y * gw, gw);
        stbtt_FreeSDF(sdf, NULL);
        int idx = codepoint * 8;
        glyph_info_buffer[idx + 0] = (f32)codepoint;
        glyph_info_buffer[idx + 1] = (f32)(*current_x) / (f32)atlas_w;
        glyph_info_buffer[idx + 2] = (f32)(*current_y) / (f32)atlas_h;
        glyph_info_buffer[idx + 3] = (f32)(*current_x + gw) / (f32)atlas_w;
        glyph_info_buffer[idx + 4] = (f32)(*current_y + gh) / (f32)atlas_h;
        glyph_info_buffer[idx + 5] = (f32)advance * scale;
        glyph_info_buffer[idx + 6] = (f32)xoff;
        glyph_info_buffer[idx + 7] = (f32)yoff;
        (*glyph_count)++; *current_x += gw + 1;
    }
    return true;
}

bool mf_loader_load_font(mf_engine* engine, const char* resource_name, const char* path, float font_size) {
    size_t len;
    char* ttf_buffer = mf_file_read_bin(path, &len);
    if (!ttf_buffer) return false;
    stbtt_fontinfo font;
    if (!stbtt_InitFont(&font, (unsigned char*)ttf_buffer, 0)) { free(ttf_buffer); return false; }
    float scale = stbtt_ScaleForPixelHeight(&font, font_size);
    int atlas_w = 512, atlas_h = 512, padding = 2;
    u8* atlas_data = calloc(1, atlas_w * atlas_h);
    int max_cp = 1200; f32* glyph_info = calloc(max_cp * 8, sizeof(f32));
    int glyph_count = 0, cur_x = 0, cur_y = 0, cell_h = (int)(font_size * 1.5f);
    bake_range_sdf(&font, 32, 127, atlas_data, atlas_w, atlas_h, &cur_x, &cur_y, cell_h, glyph_info, &glyph_count, scale, padding, 128, 32.0f);
    bake_range_sdf(&font, 1024, 1104, atlas_data, atlas_w, atlas_h, &cur_x, &cur_y, cell_h, glyph_info, &glyph_count, scale, padding, 128, 32.0f);
    mf_tensor* t = mf_engine_map_resource(engine, resource_name);
    if (t) {
        int32_t shape[] = { atlas_h * atlas_w };
        if (mf_engine_resize_resource(engine, resource_name, shape, 1)) {
            t = mf_engine_map_resource(engine, resource_name);
            for(size_t i=0; i<(size_t)atlas_w*atlas_h; ++i) ((f32*)t->buffer->data)[i] = (f32)atlas_data[i] / 255.0f;
            mf_engine_sync_resource(engine, resource_name);
        }
    }
    char info_name[128]; snprintf(info_name, 128, "%s_Info", resource_name);
    mf_tensor* t_info = mf_engine_map_resource(engine, info_name);
    if (t_info) {
        int32_t shape[] = { max_cp * 8 };
        if (mf_engine_resize_resource(engine, info_name, shape, 1)) {
            t_info = mf_engine_map_resource(engine, info_name);
            memcpy(t_info->buffer->data, glyph_info, max_cp * 8 * sizeof(f32));
            mf_engine_sync_resource(engine, info_name);
        }
    }
    free(atlas_data); free(glyph_info); free(ttf_buffer);
    return true;
}
