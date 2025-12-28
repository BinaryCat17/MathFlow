cmake_minimum_required(VERSION 3.25)

project(mf-window)

find_package(SDL2 CONFIG REQUIRED)

add_executable(mf-window src/main.c)

target_link_libraries(mf-window 
    PRIVATE 
        SDL2::SDL2
        SDL2::SDL2main
        mf_vm
        mf_compiler
        mf_backend_cpu
        mf_scheduler
        mf_platform
)

target_include_directories(mf-window PRIVATE 
    ${CMAKE_SOURCE_DIR}/modules/vm/include
    ${CMAKE_SOURCE_DIR}/modules/compiler/include
    ${CMAKE_SOURCE_DIR}/modules/backend_cpu/include
    ${CMAKE_SOURCE_DIR}/modules/isa/include
    ${CMAKE_SOURCE_DIR}/modules/scheduler/include
    ${CMAKE_SOURCE_DIR}/modules/platform/include
)
