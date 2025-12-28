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

# 2b. Platform (Threading)
add_library(mf_platform STATIC
    modules/platform/src/mf_platform.c
)
target_include_directories(mf_platform PUBLIC
    modules/platform/include
)
if(UNIX)
    target_link_libraries(mf_platform PUBLIC pthread)
endif()

# 2c. Scheduler
add_library(mf_scheduler STATIC
    modules/scheduler/src/mf_scheduler.c
)
target_include_directories(mf_scheduler PUBLIC
    modules/scheduler/include
)
target_link_libraries(mf_scheduler PUBLIC mf_vm mf_platform)

# 3. Compiler (Host Tool)
add_library(mf_compiler STATIC
    modules/compiler/src/mf_compiler.c
    modules/compiler/src/mf_json_parser.c
    modules/compiler/src/mf_semantics.c
    modules/compiler/src/mf_codegen.c
)
target_include_directories(mf_compiler PUBLIC
    modules/compiler/include
)
target_link_libraries(mf_compiler PUBLIC mf_vm cjson) # Needs VM for column types, cjson for parsing

# 4. Ops: Core
add_library(mf_ops_core STATIC
    modules/ops_core/src/mf_ops_core.c
)
target_include_directories(mf_ops_core PUBLIC
    modules/ops_core/include
)
target_link_libraries(mf_ops_core PUBLIC mf_vm m)

# 5. Ops: Array
add_library(mf_ops_array STATIC
    modules/ops_array/src/mf_ops_array.c
)
target_include_directories(mf_ops_array PUBLIC
    modules/ops_array/include
)
target_link_libraries(mf_ops_array PUBLIC mf_vm m)

# 6. Backend: CPU
add_library(mf_backend_cpu STATIC
    modules/backend_cpu/src/mf_backend_cpu.c
)
target_include_directories(mf_backend_cpu PUBLIC
    modules/backend_cpu/include
)
target_link_libraries(mf_backend_cpu PUBLIC mf_vm mf_ops_core mf_ops_array)

# 7. Engine (Unified Runtime & Loader)
add_library(mf_engine STATIC
    modules/engine/src/mf_engine.c
)
target_include_directories(mf_engine PUBLIC
    modules/engine/include
)
target_link_libraries(mf_engine PUBLIC 
    mf_vm 
    mf_compiler 
    mf_backend_cpu
)

# 8. Host (SDL2 Application Framework)
# Only build if SDL2 is found
find_package(SDL2 CONFIG QUIET)
if(TARGET SDL2::SDL2)
    add_library(mf_host STATIC
        modules/host/src/mf_host.c
    )
    target_include_directories(mf_host PUBLIC
        modules/host/include
    )
    target_link_libraries(mf_host PUBLIC 
        mf_engine
        mf_scheduler 
        SDL2::SDL2 
        SDL2::SDL2main
    )
    message(STATUS "Building mf_host (SDL2 found)")
else()
    message(WARNING "SDL2 not found. mf_host will not be built.")
endif()

# --- Apps ---

# CLI Runner
add_executable(mf-runner apps/mf-runner/src/main.c)
target_link_libraries(mf-runner PRIVATE 
    mf_engine
)

# Window Visualizer
add_subdirectory(apps/mf-window)

# --- Installation (TODO: Update for new structure if needed) ---
# For now, we just build.