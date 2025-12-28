#include <SDL2/SDL.h>
#include <mathflow/compiler/mf_compiler.h>
#include <mathflow/vm/mf_vm.h>
#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

// --- Helper: Read File ---
char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    return buf;
}

// --- Pixel Conversion (F32 -> U8 Color) ---
void convert_to_pixels(mf_tensor* tensor, void* pixels, int pitch, int tex_w, int tex_h) {
    if (tensor->dtype != MF_DTYPE_F32) return;
    
    // Assume Shape: [H, W, 4] or [H, W, 3] or [1, 1, 4] (Broadcast)
    
    f32* src = (f32*)tensor->data;
    u8* dst = (u8*)pixels;
    
    int t_w = (tensor->ndim > 1) ? tensor->shape[1] : 1;
    int t_h = (tensor->ndim > 0) ? tensor->shape[0] : 1;
    int channels = (tensor->ndim > 2) ? tensor->shape[2] : ((tensor->ndim == 1) ? tensor->shape[0] : 1);
    
    // Case 1: Single Color (Broadcasting)
    if (tensor->size <= 4) {
         float r, g, b, a;
         if (channels >= 3) {
             r = src[0]; g = src[1]; b = src[2];
             a = (channels > 3) ? src[3] : 1.0f;
         } else {
             // Grayscale / Scalar
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
                row[x * 4 + 0] = ur; // R (SDL PIXELFORMAT ABGR8888 actually expects R in byte 0 on little endian usually?)
                // Wait, ABGR8888 on Little Endian is:
                // Byte 0: R
                // Byte 1: G
                // Byte 2: B
                // Byte 3: A
                // So this order is correct for ABGR8888.
                row[x * 4 + 1] = ug;
                row[x * 4 + 2] = ub;
                row[x * 4 + 3] = ua;
             }
         }
         return;
    }
    
    // Case 2: Full Buffer
    // Determine min bounds to avoid overflow
    int w = (t_w < tex_w) ? t_w : tex_w;
    int h = (t_h < tex_h) ? t_h : tex_h;

    for (int y = 0; y < h; ++y) {
        u8* row = dst + y * pitch;
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * channels;
            
            float r, g, b, a;
            
            if (channels == 1) {
                // Grayscale
                float val = src[idx];
                r = g = b = val;
                a = 1.0f;
            } else if (channels >= 3) {
                r = src[idx + 0];
                g = src[idx + 1];
                b = src[idx + 2];
                a = (channels > 3) ? src[idx + 3] : 1.0f;
            } else {
                continue; // Skip weird 2-channel for now
            }
            
            // Clamp
            if (r < 0) r = 0; if (r > 1) r = 1;
            if (g < 0) g = 0; if (g > 1) g = 1;
            if (b < 0) b = 0; if (b > 1) b = 1;
            if (a < 0) a = 0; if (a > 1) a = 1;

            row[x * 4 + 0] = (u8)(r * 255.0f); // R
            row[x * 4 + 1] = (u8)(g * 255.0f); // G
            row[x * 4 + 2] = (u8)(b * 255.0f); // B
            row[x * 4 + 3] = (u8)(a * 255.0f); // A
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <graph.json>\n", argv[0]);
        return 1;
    }

    // 1. Init SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("MathFlow Visualizer", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    
    // Texture for pixel buffer
    SDL_Texture* texture = SDL_CreateTexture(renderer, 
        SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, 
        WINDOW_WIDTH, WINDOW_HEIGHT);

    // 2. Setup MathFlow VM
    // Allocators
    void* arena_mem = malloc(16 * 1024 * 1024); // 16MB
    mf_arena arena;
    mf_arena_init(&arena, arena_mem, 16 * 1024 * 1024);

    void* heap_mem = malloc(64 * 1024 * 1024); // 64MB
    mf_heap heap;
    mf_heap_init(&heap, heap_mem, 64 * 1024 * 1024);

    // Load Backend
    mf_backend_dispatch_table dispatch;
    mf_backend_cpu_init(&dispatch);
    
    // Compile Graph
    char* json_src = read_file(argv[1]);
    if (!json_src) {
        printf("Error reading file: %s\n", argv[1]);
        return 1;
    }

    mf_graph_ir ir = {0};
    if (!mf_compile_load_json(json_src, &ir, &arena)) return 1;
    
    mf_program* prog = mf_compile(&ir, &arena);
    if (!prog) { printf("Compilation failed\n"); return 1; }
    
    // Setup Context (Stateless)
    mf_context ctx;
    mf_context_init(&ctx, prog, &dispatch);

    // Setup VM (Stateful)
    mf_vm vm_instance;
    mf_vm* vm = &vm_instance;
    mf_vm_init(vm, &ctx, (mf_allocator*)&heap);
    
    // Allocate State
    mf_vm_reset(vm, &arena);

    // Find Inputs
    u16 reg_time = mf_vm_find_register(vm, "u_Time");
    u16 reg_res = mf_vm_find_register(vm, "u_Resolution");
    u16 reg_resx = mf_vm_find_register(vm, "u_ResX");
    u16 reg_resy = mf_vm_find_register(vm, "u_ResY");
    u16 reg_aspect = mf_vm_find_register(vm, "u_Aspect"); // Scalar F32
    u16 reg_mouse = mf_vm_find_register(vm, "u_Mouse");
    u16 reg_mousex = mf_vm_find_register(vm, "u_MouseX");
    u16 reg_mousey = mf_vm_find_register(vm, "u_MouseY");
    u16 reg_fragx = mf_vm_find_register(vm, "u_FragX"); // [H, W]
    u16 reg_fragy = mf_vm_find_register(vm, "u_FragY"); // [H, W]
    u16 reg_color = mf_vm_find_register(vm, "out_Color");

    if (reg_color == 0xFFFF) {
        printf("Error: Graph must have a node named 'out_Color'\n");
        return 1;
    }

    // Set Resolution Once
    mf_tensor* t_res = mf_vm_map_tensor(vm, reg_res, MF_ACCESS_WRITE);
    if (t_res) {
        f32* d = (f32*)t_res->data;
        if (d && t_res->size >= 2) {
            d[0] = (f32)WINDOW_WIDTH;
            d[1] = (f32)WINDOW_HEIGHT;
        }
    }

    mf_tensor* t_rx = mf_vm_map_tensor(vm, reg_resx, MF_ACCESS_WRITE);
    if (t_rx) *((f32*)t_rx->data) = (f32)WINDOW_WIDTH;

    mf_tensor* t_ry = mf_vm_map_tensor(vm, reg_resy, MF_ACCESS_WRITE);
    if (t_ry) *((f32*)t_ry->data) = (f32)WINDOW_HEIGHT;
    
    // Set Aspect Ratio
    mf_tensor* t_aspect = mf_vm_map_tensor(vm, reg_aspect, MF_ACCESS_WRITE);
    if (t_aspect) {
        f32* d = (f32*)t_aspect->data;
        if (d) *d = (f32)WINDOW_WIDTH / (f32)WINDOW_HEIGHT;
    }
    
    // Set u_FragX / u_FragY Once (Static Grids)
    int dims[2] = { WINDOW_HEIGHT, WINDOW_WIDTH };
    
    mf_tensor* t_x = mf_vm_map_tensor(vm, reg_fragx, MF_ACCESS_WRITE);
    if (t_x && mf_vm_resize_tensor(vm, t_x, dims, 2)) {
         printf("Resized u_FragX to [600, 800]. Filling...\n");
         f32* d = (f32*)t_x->data;
         for (int y = 0; y < WINDOW_HEIGHT; ++y) {
             for (int x = 0; x < WINDOW_WIDTH; ++x) {
                 d[y * WINDOW_WIDTH + x] = (f32)x + 0.5f;
             }
         }
         printf("u_FragX[0] = %f, u_FragX[end] = %f\n", d[0], d[WINDOW_HEIGHT*WINDOW_WIDTH - 1]);
    } else {
        printf("Failed to resize u_FragX!\n");
    }

    mf_tensor* t_y = mf_vm_map_tensor(vm, reg_fragy, MF_ACCESS_WRITE);
    if (t_y && mf_vm_resize_tensor(vm, t_y, dims, 2)) {
         f32* d = (f32*)t_y->data;
         for (int y = 0; y < WINDOW_HEIGHT; ++y) {
             for (int x = 0; x < WINDOW_WIDTH; ++x) {
                 d[y * WINDOW_WIDTH + x] = (f32)y + 0.5f;
             }
         }
    }

    // 3. Main Loop
    bool running = true;
    SDL_Event event;
    u32 start_time = SDL_GetTicks();
    int frame_count = 0;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
        }

        int mx, my;
        u32 buttons = SDL_GetMouseState(&mx, &my);

        // Update Inputs
        
        // Time
        mf_tensor* t_time = mf_vm_map_tensor(vm, reg_time, MF_ACCESS_WRITE);
        if (t_time) {
            f32* d = (f32*)t_time->data;
            *d = (SDL_GetTicks() - start_time) / 1000.0f;
        }

        // Mouse
        mf_tensor* t_mouse = mf_vm_map_tensor(vm, reg_mouse, MF_ACCESS_WRITE);
        if (t_mouse && t_mouse->size >= 2) {
             f32* d = (f32*)t_mouse->data;
             d[0] = (f32)mx;
             d[1] = (f32)my;
        }

        mf_tensor* t_mx = mf_vm_map_tensor(vm, reg_mousex, MF_ACCESS_WRITE);
        if (t_mx) *((f32*)t_mx->data) = (f32)mx;

        mf_tensor* t_my = mf_vm_map_tensor(vm, reg_mousey, MF_ACCESS_WRITE);
        if (t_my) *((f32*)t_my->data) = (f32)my;

        // Execute
        mf_vm_exec(vm);
        if (vm->error != MF_ERROR_NONE) {
            printf("VM Error: %d\n", vm->error);
            running = false;
        }

        // Render
        void* pixels;
        int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        
        mf_tensor* out = mf_vm_map_tensor(vm, reg_color, MF_ACCESS_READ);
        if (out) {
            convert_to_pixels(out, pixels, pitch, WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        
        SDL_UnlockTexture(texture);
        
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        
        // Auto-Screenshot (Frame 30)
        frame_count++;
        static bool screenshot_taken = false;
        if (!screenshot_taken && frame_count > 30 && out) {
            SDL_Surface* ss = SDL_CreateRGBSurfaceWithFormat(0, WINDOW_WIDTH, WINDOW_HEIGHT, 32, SDL_PIXELFORMAT_RGBA32);
            if (ss) {
                if (SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_RGBA32, ss->pixels, ss->pitch) == 0) {
                    SDL_SaveBMP(ss, "logs/debug_frame.bmp");
                    printf("Screenshot saved to logs/debug_frame.bmp (Frame %d)\n", frame_count);
                }
                SDL_FreeSurface(ss);
            }
            screenshot_taken = true;
        }

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    mf_vm_shutdown(vm);
    free(arena_mem);
    free(heap_mem);

    return 0;
}
