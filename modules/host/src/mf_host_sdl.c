#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/loader/mf_loader.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/base/mf_platform.h>
#include <mathflow/base/mf_log.h>

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- Pixel Conversion Utility ---

static void convert_to_pixels(mf_tensor* tensor, void* pixels, int pitch, int tex_w, int tex_h) {
    (void)pitch;
    void* data_ptr = mf_tensor_data(tensor);
    if (!tensor || !data_ptr) return;
    
    f32* src = (f32*)data_ptr;
    u8* dst = (u8*)pixels;
    
    int total_pixels = tex_w * tex_h;
    int channels = tensor->info.ndim >= 3 ? tensor->info.shape[tensor->info.ndim - 1] : 1;

    for (int i = 0; i < total_pixels; ++i) {
        float r, g, b, a;
        if (channels >= 4) {
            r = src[i*4 + 0];
            g = src[i*4 + 1];
            b = src[i*4 + 2];
            a = src[i*4 + 3];
        } else if (channels == 3) {
            r = src[i*3 + 0];
            g = src[i*3 + 1];
            b = src[i*3 + 2];
            a = 1.0f;
        } else {
            // Grayscale / Single channel
            r = g = b = src[i];
            a = 1.0f;
        }

        if (r < 0) r = 0; if (r > 1) r = 1;
        if (g < 0) g = 0; if (g > 1) g = 1;
        if (b < 0) b = 0; if (b > 1) b = 1;
        if (a < 0) a = 0; if (a > 1) a = 1;

        dst[i*4 + 0] = (u8)(r * 255.0f);
        dst[i*4 + 1] = (u8)(g * 255.0f);
        dst[i*4 + 2] = (u8)(b * 255.0f);
        dst[i*4 + 3] = (u8)(a * 255.0f);
    }
}

int mf_host_run(const mf_host_desc* desc) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        MF_LOG_ERROR("SDL Init Error: %s", SDL_GetError());
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
        MF_LOG_ERROR("Window Creation Error: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    u32 render_flags = SDL_RENDERER_ACCELERATED;
    if (desc->vsync) render_flags |= SDL_RENDERER_PRESENTVSYNC;

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, render_flags);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, desc->width, desc->height);

    mf_host_app app;
    if (mf_host_app_init(&app, desc) != 0) {
        MF_LOG_ERROR("Failed to initialize Host App");
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }

    u32* frame_buffer = malloc((size_t)desc->width * (size_t)desc->height * 4);
    bool running = true;
    u32 start_ticks = SDL_GetTicks();
    f32 last_log_time = -desc->log_interval - 1.0f; 

    int win_w = desc->width;
    int win_h = desc->height;
    SDL_Event event;

    while (running) {
        u32 current_ticks = SDL_GetTicks() - start_ticks;
        f32 current_time = current_ticks / 1000.0f;
        
        bool do_log = (desc->log_interval > 0) && (current_time - last_log_time) >= desc->log_interval;
        if (do_log) {
            mf_log_set_global_level(MF_LOG_LEVEL_TRACE);
            last_log_time = current_time;
            MF_LOG_INFO("--- Frame Log @ %.2fs ---", current_time);
        } else {
            mf_log_set_global_level(MF_LOG_LEVEL_WARN);
        }

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
                win_w = event.window.data1;
                win_h = event.window.data2;
                
                SDL_DestroyTexture(texture);
                texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, win_w, win_h);
                
                free(frame_buffer);
                frame_buffer = malloc((size_t)win_w * (size_t)win_h * 4);
                
                mf_host_app_handle_resize(&app, win_w, win_h);
            }
        }
        
        int mx, my;
        u32 buttons = SDL_GetMouseState(&mx, &my);
        mf_host_app_update_system_resources(&app, current_time, (f32)mx, (f32)my, 
                                            (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0, 
                                            (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0);

        mf_engine_dispatch(app.engine);
        
        mf_tensor* t_out = mf_engine_map_resource(app.engine, "out_Color");
        if (t_out && frame_buffer) {
            convert_to_pixels(t_out, frame_buffer, win_w * 4, win_w, win_h);
            SDL_UpdateTexture(texture, NULL, frame_buffer, win_w * 4);
        }
        
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        if (do_log && frame_buffer) {
            char shot_path[256];
            time_t now = time(NULL);
            struct tm* t_struct = localtime(&now);
            strftime(shot_path, sizeof(shot_path), "logs/screenshot_%Y-%m-%d_%H-%M-%S.bmp", t_struct);
            
            SDL_Surface* ss = SDL_CreateRGBSurfaceFrom(
                frame_buffer, win_w, win_h, 32, win_w * 4,
                0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000
            );
            
            if (ss) {
                if (SDL_SaveBMP(ss, shot_path) == 0) {
                    MF_LOG_INFO("Screenshot saved: %s", shot_path);
                }
                SDL_FreeSurface(ss);
            }
        }
    }
    
    free(frame_buffer);
    mf_host_app_cleanup(&app);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}