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
    
    size_t count = mf_tensor_count(tensor);
    size_t required_floats = (size_t)total_pixels * 4; 
    if (count < required_floats) return; 

    for (int i = 0; i < total_pixels; ++i) {
        float r = src[i*4 + 0];
        float g = src[i*4 + 1];
        float b = src[i*4 + 2];
        float a = src[i*4 + 3];

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

    mf_engine_desc engine_desc = {0};
    engine_desc.arena_size = 32 * 1024 * 1024; 
    engine_desc.heap_size = 128 * 1024 * 1024; 
    
    mf_loader_init_backend(&engine_desc.backend, desc->num_threads);

    mf_engine* engine = mf_engine_create(&engine_desc);
    if (!engine) {
        MF_LOG_ERROR("Failed to create engine");
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }

    if (desc->has_pipeline) {
        if (!mf_loader_load_pipeline(engine, &desc->pipeline)) {
            MF_LOG_ERROR("Failed to load pipeline");
            mf_engine_destroy(engine);
            SDL_DestroyWindow(window); SDL_Quit();
            return 1;
        }
        
        // --- Init Hack: Inventory ---
        mf_tensor* t_inv = mf_engine_map_resource(engine, "Inventory");
        if (t_inv) {
            MF_LOG_INFO("Initializing Inventory Resource...");
            f32* d = (f32*)mf_tensor_data(t_inv);
            if (d && mf_tensor_count(t_inv) >= 4) {
                d[0] = 1.0f; // Item 1 (Red)
                d[1] = 2.0f; // Item 2 (Green)
                d[2] = 3.0f; // Item 3 (Blue)
                d[3] = 0.0f; // Empty
            }
        }
    } else {
        if (!mf_loader_load_graph(engine, desc->graph_path)) {
            MF_LOG_ERROR("Failed to load graph: %s", desc->graph_path);
            mf_engine_destroy(engine);
            SDL_DestroyWindow(window); SDL_Quit();
            return 1;
        }
    }
    
    int32_t screen_shape[] = { desc->height, desc->width, 4 };
    mf_engine_resize_resource(engine, "out_Color", screen_shape, 3);
    
    mf_tensor *t_time, *t_res, *t_mouse, *t_out;
    
    t_res   = mf_engine_map_resource(engine, "u_Resolution");
    if (t_res) {
        void* res_data = mf_tensor_data(t_res);
        if (res_data) {
            size_t count = mf_tensor_count(t_res);
            if (count >= 2) {
                f32* d = (f32*)res_data;
                d[0] = (f32)desc->width;
                d[1] = (f32)desc->height;
            }
        }
    }

    u32* frame_buffer = malloc((size_t)desc->width * (size_t)desc->height * 4);

    bool running = true;
    
    u32 start_time = SDL_GetTicks();
    // Initialize so that the first frame (time ~0) triggers a log.
    // using a negative value ensures (current - last) >= interval immediately.
    f32 last_log_time = -desc->log_interval - 1.0f; 

    int win_w = desc->width;
    int win_h = desc->height;
    SDL_Event event;

    while (running) {
        // --- 1. Periodic Logging Control ---
        f32 current_time = (SDL_GetTicks() - start_time) / 1000.0f;
        bool do_log = (current_time - last_log_time) >= desc->log_interval;
        
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
                
                int32_t shape[] = { win_h, win_w, 4 };
                mf_engine_resize_resource(engine, "out_Color", shape, 3);
                
                t_res = mf_engine_map_resource(engine, "u_Resolution");
                if (t_res) {
                    void* res_data = mf_tensor_data(t_res);
                    if (res_data && mf_tensor_count(t_res) >= 2) {
                        f32* d = (f32*)res_data;
                        d[0] = (f32)win_w;
                        d[1] = (f32)win_h;
                    }
                }
            }
        }
        
        // --- Standard Uniforms ---
        t_time  = mf_engine_map_resource(engine, "u_Time");
        t_mouse = mf_engine_map_resource(engine, "u_Mouse");
        
        mf_tensor* t_rx = mf_engine_map_resource(engine, "u_ResX");
        mf_tensor* t_ry = mf_engine_map_resource(engine, "u_ResY");
        mf_tensor* t_aspect = mf_engine_map_resource(engine, "u_Aspect");
        mf_tensor* t_mx = mf_engine_map_resource(engine, "u_MouseX");
        mf_tensor* t_my = mf_engine_map_resource(engine, "u_MouseY");

        if (t_time) {
            void* time_data = mf_tensor_data(t_time);
            if (time_data) *((f32*)time_data) = current_time;
        }

        f32 mx_val = 0, my_val = 0;
        
        if (t_mouse || t_mx || t_my) {
            int mx, my;
            u32 buttons = SDL_GetMouseState(&mx, &my);
            mx_val = (f32)mx;
            my_val = (f32)my;

            if (t_mouse) {
                void* mouse_data = mf_tensor_data(t_mouse);
                if (mouse_data) {
                    f32* d = (f32*)mouse_data;
                    d[0] = mx_val;
                    d[1] = my_val;
                    d[2] = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) ? 1.0f : 0.0f;
                    d[3] = (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) ? 1.0f : 0.0f;
                }
            }
            if (t_mx) {
                 void* d = mf_tensor_data(t_mx);
                 if (d) *((f32*)d) = mx_val;
            }
            if (t_my) {
                 void* d = mf_tensor_data(t_my);
                 if (d) *((f32*)d) = my_val;
            }
        }
        
        if (t_rx) {
            void* d = mf_tensor_data(t_rx);
            if (d) {
                *((f32*)d) = (f32)win_w;
            }
        } else {
            MF_LOG_ERROR("u_ResX not found!");
        }
        if (t_ry) {
            void* d = mf_tensor_data(t_ry);
            if (d) *((f32*)d) = (f32)win_h;
        }
        if (t_aspect) {
            void* d = mf_tensor_data(t_aspect);
            if (d) *((f32*)d) = (f32)win_w / (f32)win_h;
        }

        mf_engine_dispatch(engine);
        
        t_out = mf_engine_map_resource(engine, "out_Color");

        if (t_out && frame_buffer) {
            void* out_data = mf_tensor_data(t_out);
            if (out_data) {
                convert_to_pixels(t_out, frame_buffer, win_w * 4, win_w, win_h);
            }
        }
        
        if (frame_buffer) {
            SDL_UpdateTexture(texture, NULL, frame_buffer, win_w * 4);
        }
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        // --- 2. Screenshot Logic ---
        if (do_log && frame_buffer) {
            time_t now = time(NULL);
            struct tm* t = localtime(&now);
            char shot_path[256];
            // Use time_buf from log? Or just raw timestamp.
            // Using system time for filename to avoid overwrites
            strftime(shot_path, sizeof(shot_path), "logs/screenshot_%Y-%m-%d_%H-%M-%S.bmp", t);
            
            SDL_Surface* ss = SDL_CreateRGBSurfaceFrom(
                frame_buffer, win_w, win_h, 32, win_w * 4,
                0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000
            );
            
            if (ss) {
                if (SDL_SaveBMP(ss, shot_path) == 0) {
                    MF_LOG_INFO("Screenshot saved: %s", shot_path);
                } else {
                    MF_LOG_ERROR("Failed to save screenshot: %s", SDL_GetError());
                }
                SDL_FreeSurface(ss);
            }
        }
    }
    
    free(frame_buffer);
    mf_engine_destroy(engine);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}