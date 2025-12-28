#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/host/mf_asset_loader.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/base/mf_platform.h>

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Pixel Conversion Utility ---
static void convert_to_pixels(mf_tensor* tensor, void* pixels, int pitch, int tex_w, int tex_h) {
    if (!tensor || tensor->dtype != MF_DTYPE_F32) return;
    
    f32* src = (f32*)tensor->data;
    if (!src) return; 

    u8* dst = (u8*)pixels;
    
    int t_w = (tensor->ndim > 1) ? tensor->shape[1] : 1;
    int t_h = (tensor->ndim > 0) ? tensor->shape[0] : 1;
    int channels = (tensor->ndim > 2) ? tensor->shape[2] : ((tensor->ndim == 1) ? tensor->shape[0] : 1);
    
    if (t_w == 0 || t_h == 0) return;

    bool is_scalar = (tensor->size <= 4);
    
    if (is_scalar) {
         float r, g, b, a;
         if (channels >= 3) {
             r = src[0]; g = src[1]; b = src[2];
             a = (channels > 3) ? src[3] : 1.0f;
         } else {
             r = g = b = src[0];
             a = 1.0f;
         }
         
         u8 ur = (u8)(r * 255.0f);
         u8 ug = (u8)(g * 255.0f);
         u8 ub = (u8)(b * 255.0f);
         u8 ua = (u8)(a * 255.0f);

         for (int y = 0; y < tex_h; ++y) {
             u8* row = dst + y * pitch;
             for (int x = 0; x < tex_w; ++x) {
                row[x * 4 + 0] = ur;
                row[x * 4 + 1] = ug;
                row[x * 4 + 2] = ub;
                row[x * 4 + 3] = ua;
             }
         }
    } else {
        int w = (t_w < tex_w) ? t_w : tex_w;
        int h = (t_h < tex_h) ? t_h : tex_h;

        for (int y = 0; y < h; ++y) {
            u8* row = dst + y * pitch;
            for (int x = 0; x < w; ++x) {
                int idx = (y * w + x) * channels;
                
                float r, g, b, a;
                if (channels == 1) {
                    float val = src[idx];
                    r = g = b = val;
                    a = 1.0f;
                } else if (channels >= 3) {
                    r = src[idx + 0];
                    g = src[idx + 1];
                    b = src[idx + 2];
                    a = (channels > 3) ? src[idx + 3] : 1.0f;
                } else {
                    continue; 
                }
                
                if (r < 0) r = 0; if (r > 1) r = 1;
                if (g < 0) g = 0; if (g > 1) g = 1;
                if (b < 0) b = 0; if (b > 1) b = 1;
                if (a < 0) a = 0; if (a > 1) a = 1;

                row[x * 4 + 0] = (u8)(r * 255.0f);
                row[x * 4 + 1] = (u8)(g * 255.0f);
                row[x * 4 + 2] = (u8)(b * 255.0f);
                row[x * 4 + 3] = (u8)(a * 255.0f);
            }
        }
    }
}

typedef struct {
    float time;
    float mouse[4];
    int width;
    int height;
    void* pixels;
    int tile_count;
    int tile_height;
    u16 r_time;
    u16 r_res, r_resx, r_resy, r_aspect;
    u16 r_mouse, r_mousex, r_mousey;
    u16 r_fragx, r_fragy;
    u16 r_out;
} mf_host_job_ctx;

static void host_job_setup(mf_job_handle job, u32 job_idx, void* user_data) {
    mf_host_job_ctx* rc = (mf_host_job_ctx*)user_data;
    
    int y_start = job_idx * rc->tile_height;
    int y_end = y_start + rc->tile_height;
    if (y_end > rc->height) y_end = rc->height;
    int local_h = y_end - y_start;
    if (local_h <= 0) return;

    mf_tensor* t;

    if (rc->r_time != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_time, MF_ACCESS_WRITE))) {
        if (t->data) *((f32*)t->data) = rc->time;
    }
    
    if (rc->r_res != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_res, MF_ACCESS_WRITE))) {
        if (t->data) {
             f32* d = (f32*)t->data; d[0] = (f32)rc->width; d[1] = (f32)rc->height;
        }
    }

    if (rc->r_resx != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_resx, MF_ACCESS_WRITE))) {
        if (t->data) *((f32*)t->data) = (f32)rc->width;
    }

    if (rc->r_resy != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_resy, MF_ACCESS_WRITE))) {
        if (t->data) *((f32*)t->data) = (f32)rc->height;
    }
    
    if (rc->r_aspect != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_aspect, MF_ACCESS_WRITE))) {
        if (t->data) *((f32*)t->data) = (f32)rc->width / (f32)rc->height;
    }

    if (rc->r_mouse != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_mouse, MF_ACCESS_WRITE))) {
        if (t->data) memcpy(t->data, rc->mouse, sizeof(float)*4);
    }
    
    if (rc->r_mousex != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_mousex, MF_ACCESS_WRITE))) {
        if (t->data) *((f32*)t->data) = rc->mouse[0];
    }

    if (rc->r_mousey != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_mousey, MF_ACCESS_WRITE))) {
        if (t->data) *((f32*)t->data) = rc->mouse[1];
    }
    
    int dims[2] = { local_h, rc->width };
    
    if (rc->r_fragx != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_fragx, MF_ACCESS_WRITE))) {
        if (mf_job_resize_tensor(job, t, dims, 2)) {
            f32* d = (f32*)t->data;
            for (int y = 0; y < local_h; ++y) {
                for (int x = 0; x < rc->width; ++x) {
                    d[y * rc->width + x] = (f32)x + 0.5f;
                }
            }
        }
    }

    if (rc->r_fragy != 0xFFFF && (t = mf_job_map_tensor(job, rc->r_fragy, MF_ACCESS_WRITE))) {
        if (mf_job_resize_tensor(job, t, dims, 2)) {
            f32* d = (f32*)t->data;
            for (int y = 0; y < local_h; ++y) {
                float val = (f32)(y_start + y) + 0.5f;
                for (int x = 0; x < rc->width; ++x) {
                    d[y * rc->width + x] = val;
                }
            }
        }
    }
}

static void host_job_finish(mf_job_handle job, u32 job_idx, void* user_data) {
    mf_host_job_ctx* rc = (mf_host_job_ctx*)user_data;
    if (rc->r_out == 0xFFFF) return; 
    
    mf_tensor* out = mf_job_map_tensor(job, rc->r_out, MF_ACCESS_READ);
    if (!out || !out->data) return;
    
    int y_start = job_idx * rc->tile_height;
    int pitch = rc->width * 4; 
    u8* pixels_base = (u8*)rc->pixels;
    
    int local_h = rc->tile_height;
    if (y_start + local_h > rc->height) local_h = rc->height - y_start;

    convert_to_pixels(out, pixels_base + (y_start * pitch), pitch, rc->width, local_h);
}

int mf_host_run(const mf_host_desc* desc) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("[Host] SDL Init Error: %s\n", SDL_GetError());
        return 1;
    }

    u32 flags = SDL_WINDOW_SHOWN;
    if (desc->resizable) flags |= SDL_WINDOW_RESIZABLE;
    if (desc->fullscreen) flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;

    SDL_Window* window = SDL_CreateWindow(
        desc->window_title ? desc->window_title : "MathFlow App",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        desc->width, desc->height, flags
    );

    if (!window) {
        printf("[Host] Window Creation Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    u32 render_flags = SDL_RENDERER_ACCELERATED;
    if (desc->vsync) render_flags |= SDL_RENDERER_PRESENTVSYNC;

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, render_flags);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, desc->width, desc->height);

    mf_engine_desc engine_desc = {0};
    engine_desc.arena_size = 16 * 1024 * 1024;
    engine_desc.num_threads = desc->num_threads;

    mf_engine* engine = mf_engine_create(&engine_desc);
    if (!engine) {
        printf("[Host] Failed to create engine\n");
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }

    if (!mf_asset_loader_load(engine, desc->graph_path)) {
        printf("[Host] Failed to load graph: %s\n", desc->graph_path);
        mf_engine_destroy(engine);
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }
    
    // Cache Registers
    int32_t ir_out    = mf_engine_find_register(engine, "out_Color");
    int32_t ir_fragx  = mf_engine_find_register(engine, "u_FragX");
    int32_t ir_fragy  = mf_engine_find_register(engine, "u_FragY");
    
    u16 r_out   = (ir_out >= 0) ? (u16)ir_out : 0xFFFF;
    u16 r_fragx = (ir_fragx >= 0) ? (u16)ir_fragx : 0xFFFF;
    u16 r_fragy = (ir_fragy >= 0) ? (u16)ir_fragy : 0xFFFF;

    mf_host_job_ctx job_ctx = {0};
    job_ctx.width = desc->width;
    job_ctx.height = desc->height;
    
    int32_t idx;
    idx = mf_engine_find_register(engine, "u_Time"); job_ctx.r_time = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_Resolution"); job_ctx.r_res = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_ResX"); job_ctx.r_resx = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_ResY"); job_ctx.r_resy = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_Aspect"); job_ctx.r_aspect = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_Mouse"); job_ctx.r_mouse = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_MouseX"); job_ctx.r_mousex = (idx >= 0) ? (u16)idx : 0xFFFF;
    idx = mf_engine_find_register(engine, "u_MouseY"); job_ctx.r_mousey = (idx >= 0) ? (u16)idx : 0xFFFF;
    
    job_ctx.r_fragx  = r_fragx;
    job_ctx.r_fragy  = r_fragy;
    job_ctx.r_out    = r_out;

    u32* frame_buffer = malloc(desc->width * desc->height * 4);
    job_ctx.pixels = frame_buffer;

    bool running = true;
    SDL_Event event;
    u32 start_time = SDL_GetTicks();
    
    int tile_count = (desc->num_threads > 0) ? desc->num_threads : 4; 

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
        }
        
        int mx, my;
        u32 buttons = SDL_GetMouseState(&mx, &my);
        job_ctx.time = (SDL_GetTicks() - start_time) / 1000.0f;
        job_ctx.mouse[0] = (float)mx;
        job_ctx.mouse[1] = (float)my;
        job_ctx.mouse[2] = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) ? 1.0f : 0.0f;
        job_ctx.mouse[3] = (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) ? 1.0f : 0.0f;
        
        job_ctx.tile_height = (job_ctx.height + tile_count - 1) / tile_count;
        mf_engine_dispatch(engine, tile_count, 1, host_job_setup, host_job_finish, &job_ctx);
        
        SDL_UpdateTexture(texture, NULL, frame_buffer, job_ctx.width * 4);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
    
    free(frame_buffer);
    mf_engine_destroy(engine);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
