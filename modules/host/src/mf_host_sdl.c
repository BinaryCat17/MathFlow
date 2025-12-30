#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/loader/mf_loader.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/base/mf_platform.h>
#include <mathflow/base/mf_log.h>

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Pixel Conversion Utility ---
static void convert_to_pixels(mf_tensor* tensor, void* pixels, int pitch, int tex_w, int tex_h) {
    if (!tensor || !tensor->data) return;
    
    // Flattened source: [Tiles, Pixels, 4] -> treat as linear stream of floats
    f32* src = (f32*)tensor->data;
    u8* dst = (u8*)pixels;
    
    // We assume the tensor data is exactly what we need for the screen size
    // Just iterating 0..tex_h*tex_w
    int total_pixels = tex_w * tex_h;
    
    // Check tensor size
    size_t required_floats = total_pixels * 4; // RGBA
    if (tensor->size < required_floats) return; // Safety

    // Basic F32 [0..1] -> U8 [0..255] conversion
    // Optimized loop
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
    engine_desc.arena_size = 32 * 1024 * 1024; // Increase arena for complex graphs
    engine_desc.heap_size = 128 * 1024 * 1024; // Increase heap for full resolution tensors
    
    // Init Backend (Injection)
    mf_loader_init_backend(&engine_desc.backend, desc->num_threads);

    mf_engine* engine = mf_engine_create(&engine_desc);
    if (!engine) {
        MF_LOG_ERROR("Failed to create engine");
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }

    // Unified Pipeline Loading
    if (desc->has_pipeline) {
        if (!mf_loader_load_pipeline(engine, &desc->pipeline)) {
            MF_LOG_ERROR("Failed to load pipeline");
            mf_engine_destroy(engine);
            SDL_DestroyWindow(window); SDL_Quit();
            return 1;
        }
    } else {
        // Automatically synthesizes a pipeline from the graph
        if (!mf_loader_load_graph(engine, desc->graph_path)) {
            MF_LOG_ERROR("Failed to load graph: %s", desc->graph_path);
            mf_engine_destroy(engine);
            SDL_DestroyWindow(window); SDL_Quit();
            return 1;
        }
    }
    
    // Resize Output to match Screen Size initially
    int32_t screen_shape[] = { desc->height, desc->width, 4 };
    mf_engine_resize_resource(engine, "out_Color", screen_shape, 3);
    mf_engine_resize_resource(engine, "u_Resolution", screen_shape, 3); // Often resolution is vec3(w,h,0)

    // Lookup Resources
    mf_tensor *t_time, *t_res, *t_mouse, *t_out;
    t_time  = mf_engine_map_resource(engine, "u_Time");
    t_res   = mf_engine_map_resource(engine, "u_Resolution");
    t_mouse = mf_engine_map_resource(engine, "u_Mouse");
    t_out   = mf_engine_map_resource(engine, "out_Color");
    
    // Set Resolution Uniform (if using vec2/3 uniform style)
    if (t_res && t_res->data) {
        f32* d = (f32*)t_res->data;
        // Check size? Assume at least 2 floats
        if (t_res->size >= 2) {
            d[0] = (f32)desc->width;
            d[1] = (f32)desc->height;
        }
    }

    u32* frame_buffer = malloc(desc->width * desc->height * 4);

    bool running = true;
    SDL_Event event;
    u32 start_time = SDL_GetTicks();
    int win_w = desc->width;
    int win_h = desc->height;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
                win_w = event.window.data1;
                win_h = event.window.data2;
                
                // Recreate Texture
                SDL_DestroyTexture(texture);
                texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, win_w, win_h);
                
                // Resize Framebuffer
                free(frame_buffer);
                frame_buffer = malloc(win_w * win_h * 4);
                
                // Resize Engine Resources
                int32_t shape[] = { win_h, win_w, 4 };
                mf_engine_resize_resource(engine, "out_Color", shape, 3);
                
                // Update Resolution Uniform
                // Note: We need to re-map because resize might have changed pointers? 
                // Wait, map_resource always returns the current descriptor.
                // We should re-map every frame or check if pointer changed?
                // The Engine promises that map_resource returns a descriptor that points to the correct buffer for the current frame.
                // But resize invalidates the buffer.
                // So we definitely need to update our cached tensor pointers or re-map.
                
                // For simplicity, let's re-map everything inside the loop (it's cheap string lookup).
                // Or just re-map here.
                t_out = mf_engine_map_resource(engine, "out_Color");
                t_res = mf_engine_map_resource(engine, "u_Resolution");
                
                 if (t_res && t_res->data && t_res->size >= 2) {
                    f32* d = (f32*)t_res->data;
                    d[0] = (f32)win_w;
                    d[1] = (f32)win_h;
                }
            }
        }
        
        // Per-Frame Re-Mapping (Ping-Pong Safety)
        // Because of Double Buffering, the 'data' pointer flips every frame.
        // We MUST call map_resource every frame to get the correct pointer.
        t_time  = mf_engine_map_resource(engine, "u_Time");
        t_mouse = mf_engine_map_resource(engine, "u_Mouse");
        t_out   = mf_engine_map_resource(engine, "out_Color"); // Read from result of LAST frame? Or write to NEXT? 
        // Host reads OUTPUT. Output is produced by kernel.
        // Kernel writes to B. Host should read B?
        // Map returns "Current Data".
        
        // Update Inputs
        if (t_time && t_time->data) {
            *((f32*)t_time->data) = (SDL_GetTicks() - start_time) / 1000.0f;
        }
        
        if (t_mouse && t_mouse->data) {
            int mx, my;
            u32 buttons = SDL_GetMouseState(&mx, &my);
            f32* d = (f32*)t_mouse->data;
            d[0] = (f32)mx;
            d[1] = (f32)my;
            d[2] = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) ? 1.0f : 0.0f;
            d[3] = (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) ? 1.0f : 0.0f;
        }

        // Dispatch!
        mf_engine_dispatch(engine, win_w, win_h);
        
        // Read Back
        // NOTE: We need to re-map t_out AFTER dispatch if we want to read what was just written?
        // Or BEFORE?
        // Engine dispatch logic: "Executes kernel, writes to B".
        // Frame index increments.
        // Next map_resource call returns B (because is_even flipped).
        // So we should call map_resource NOW.
        t_out = mf_engine_map_resource(engine, "out_Color");

        if (t_out && t_out->data && frame_buffer) {
            convert_to_pixels(t_out, frame_buffer, win_w * 4, win_w, win_h);
        }
        
        if (frame_buffer) {
            SDL_UpdateTexture(texture, NULL, frame_buffer, win_w * 4);
        }
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
