cmake_minimum_required(VERSION 3.25)

project(mf-window)

if(TARGET MathFlow::host_sdl)
    add_executable(mf-window src/main.c)

    target_link_libraries(mf-window 
        PRIVATE 
            MathFlow::host_sdl
            MathFlow::base
    )
else()
    message(STATUS "Skipping mf-window (MathFlow::host_sdl not available)")
endif()