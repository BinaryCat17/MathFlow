cmake_minimum_required(VERSION 3.25)

project(MathFlow LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(cJSON CONFIG REQUIRED)

# --- Modules ---

# 1. ISA (Header-only)
add_library(mf_isa INTERFACE)
target_include_directories(mf_isa INTERFACE
    modules/isa/include
)

# 2. VM (Runtime)
add_library(mf_vm STATIC
    modules/vm/src/mf_memory.c
    modules/vm/src/mf_vm.c
)
target_include_directories(mf_vm PUBLIC
    modules/vm/include
)
target_link_libraries(mf_vm PUBLIC mf_isa)

# 3. Compiler (Host Tool)
add_library(mf_compiler STATIC
    modules/compiler/src/mf_compiler.c
)
target_include_directories(mf_compiler PUBLIC
    modules/compiler/include
)
target_link_libraries(mf_compiler PUBLIC mf_vm cjson) # Needs VM for column types, cjson for parsing

# 4. Backend: CPU
add_library(mf_backend_cpu STATIC
    modules/backend_cpu/src/mf_backend_cpu.c
)
target_include_directories(mf_backend_cpu PUBLIC
    modules/backend_cpu/include
)
target_link_libraries(mf_backend_cpu PUBLIC mf_vm)

# --- Apps ---

# CLI Runner
add_executable(mf-runner apps/mf-runner/src/main.c)
target_link_libraries(mf-runner PRIVATE 
    mf_vm 
    mf_compiler 
    mf_backend_cpu
)

# --- Installation (TODO: Update for new structure if needed) ---
# For now, we just build.