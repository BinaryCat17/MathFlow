cmake_minimum_required(VERSION 3.25)

project(MathFlow LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# --- Dependencies ---
# Global dependencies can still be found here if they are common, 
# but modules should ideally find what they need.
find_package(SDL2 CONFIG QUIET)
find_package(Stb REQUIRED)

# --- Modules ---
add_subdirectory(modules/base)
add_subdirectory(modules/isa)
add_subdirectory(modules/compiler)
add_subdirectory(modules/ops)
add_subdirectory(modules/backend_cpu)
add_subdirectory(modules/engine)
add_subdirectory(modules/loader)
add_subdirectory(modules/host)

# --- Apps ---
add_subdirectory(apps/mf-runner)
add_subdirectory(apps/mf-window) # This one already had its own CMakeLists.txt

# --- Installation ---
# TODO: Setup install targets
