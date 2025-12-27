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
    
    if (channels < 3) return; // Need at least RGB for full buffer copy

    // Case 2: Full Buffer
    // Determine min bounds to avoid overflow
    int w = (t_w < tex_w) ? t_w : tex_w;
    int h = (t_h < tex_h) ? t_h : tex_h;

    for (int y = 0; y < h; ++y) {
        u8* row = dst + y * pitch;
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * channels;
            
            // F32 [0..1] -> U8 [0..255]
            // SDL Texture is typically ABGR or ARGB on little endian
            // Let's assume ABGR8888 for now
            
            float r = src[idx + 0];
            float g = src[idx + 1];
            float b = src[idx + 2];
            float a = (channels > 3) ? src[idx + 3] : 1.0f;
            
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
        SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 
        WINDOW_WIDTH, WINDOW_HEIGHT);

    // 2. Setup MathFlow VM
    // Allocators
    void* arena_mem = malloc(16 * 1024 * 1024); // 16MB
    mf_arena arena;
    mf_arena_init(&arena, arena_mem, 16 * 1024 * 1024);

    void* heap_mem = malloc(64 * 1024 * 1024); // 64MB
    mf_heap heap;
    mf_heap_init(&heap, heap_mem, 64 * 1024 * 1024);

    // VM
    mf_vm vm_instance;
    mf_vm* vm = &vm_instance;
    mf_vm_init(vm, (mf_allocator*)&heap);

    // Load Backend
    mf_backend_dispatch_table dispatch;
    mf_backend_cpu_init(&dispatch);
    vm->backend = &dispatch;

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
    
    // Save program for debug (optional)
    // mf_compile_save_program(prog, "out/debug_out.bin");

    // Load Program
    mf_vm_load_program(vm, prog, &arena);

    // Find Inputs
    u16 reg_time = mf_vm_find_register(vm, "u_Time");
    u16 reg_res = mf_vm_find_register(vm, "u_Resolution");
    u16 reg_mouse = mf_vm_find_register(vm, "u_Mouse");
    u16 reg_color = mf_vm_find_register(vm, "out_Color");

    if (reg_color == 0xFFFF) {
        printf("Error: Graph must have a node named 'out_Color'\n");
        return 1;
    }

    // Set Resolution Once
    mf_tensor* t_res = mf_vm_map_tensor(vm, reg_res, MF_ACCESS_WRITE);
    if (t_res) {
        // Assume graph expects [2] F32
        // We might need to resize it if it's not initialized
        // Ideally, graph has "Input" with data, so shape is already [2]
        f32* d = (f32*)t_res->data;
        if (d && t_res->size >= 2) {
            d[0] = (f32)WINDOW_WIDTH;
            d[1] = (f32)WINDOW_HEIGHT;
        }
    }

    // 3. Main Loop
    bool running = true;
    SDL_Event event;
    u32 start_time = SDL_GetTicks();

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
        if (t_mouse && t_mouse->size >= 4) {
             f32* d = (f32*)t_mouse->data;
             d[0] = (f32)mx;
             d[1] = (f32)my;
             d[2] = (buttons & SDL_BUTTON(1)) ? 1.0f : 0.0f;
             d[3] = (buttons & SDL_BUTTON(3)) ? 1.0f : 0.0f;
        }

        // Execute
        mf_vm_exec(vm);

        // Render
        void* pixels;
        int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        
        mf_tensor* out = mf_vm_map_tensor(vm, reg_color, MF_ACCESS_READ);
        if (out) {
            convert_to_pixels(out, pixels, pitch, WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        
        SDL_UnlockTexture(texture);
        
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    // mf_arena_destroy(arena); // (If implemented)
    return 0;
}
