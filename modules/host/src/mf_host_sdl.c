#include <mathflow/host/mf_host_sdl.h>
#include <mathflow/loader/mf_loader.h>
#include <mathflow/engine/mf_engine.h>
#include <mathflow/base/mf_platform.h>

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
    engine_desc.arena_size = 32 * 1024 * 1024; // Increase arena for complex graphs
    engine_desc.heap_size = 128 * 1024 * 1024; // Increase heap for full resolution tensors
    
    // Init Backend (Injection)
    mf_loader_init_backend(&engine_desc.backend, desc->num_threads);

    mf_engine* engine = mf_engine_create(&engine_desc);
    if (!engine) {
        printf("[Host] Failed to create engine\n");
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }

    if (!mf_loader_load_graph(engine, desc->graph_path)) {
        printf("[Host] Failed to load graph: %s\n", desc->graph_path);
        mf_engine_destroy(engine);
        SDL_DestroyWindow(window); SDL_Quit();
        return 1;
    }
    
    // --- Register Lookup ---
    mf_tensor* t_time = mf_engine_map_tensor(engine, mf_engine_find_register(engine, "u_Time"), MF_ACCESS_WRITE);
    mf_tensor* t_res  = mf_engine_map_tensor(engine, mf_engine_find_register(engine, "u_Resolution"), MF_ACCESS_WRITE);
    mf_tensor* t_mouse = mf_engine_map_tensor(engine, mf_engine_find_register(engine, "u_Mouse"), MF_ACCESS_WRITE);
    mf_tensor* t_out   = mf_engine_map_tensor(engine, mf_engine_find_register(engine, "out_Color"), MF_ACCESS_READ);
    
    // Set Resolution Uniform (Legacy support for graphs using u_Resolution)
    if (t_res && t_res->data) {
        // Resize if needed or just update data
        if (t_res->size >= 2) {
             f32* d = (f32*)t_res->data;
             d[0] = (f32)desc->width;
             d[1] = (f32)desc->height;
        }
    }
    
    // Resize Output to match Screen Size
    if (t_out) {
        // Shape: [Height, Width, 4] (Row-Major for SDL Texture)
        int32_t shape[] = { desc->height, desc->width, 4 };
        mf_engine_resize_tensor(engine, t_out, shape, 3);
    }

    u32* frame_buffer = malloc(desc->width * desc->height * 4);

    bool running = true;
    SDL_Event event;
    u32 start_time = SDL_GetTicks();

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
        }
        
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
        // New API: Run on the domain of the screen [Width, Height]
        // The engine/backend handles tiling automatically.
        mf_engine_dispatch(engine, desc->width, desc->height);
        
        // Read Back
        if (t_out) {
            convert_to_pixels(t_out, frame_buffer, desc->width * 4, desc->width, desc->height);
        }
        
        SDL_UpdateTexture(texture, NULL, frame_buffer, desc->width * 4);
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