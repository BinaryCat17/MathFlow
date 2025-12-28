find_package(SDL2 CONFIG QUIET)

# --- Host Core (Manifest, Config, Loader, Headless) ---
add_library(mf_host_core STATIC
    src/mf_manifest_loader.c
    src/mf_host_headless.c
)
add_library(MathFlow::host_core ALIAS mf_host_core)

target_include_directories(mf_host_core 
    PUBLIC include
    PRIVATE src
)
target_link_libraries(mf_host_core 
    PUBLIC 
        cjson
        MathFlow::engine
        MathFlow::loader
    PRIVATE
        MathFlow::compiler
)

# --- Host SDL (Window & Input) ---
if(TARGET SDL2::SDL2)
    add_library(mf_host_sdl STATIC
        src/mf_host_sdl.c
    )
    add_library(MathFlow::host_sdl ALIAS mf_host_sdl)

    target_include_directories(mf_host_sdl 
        PUBLIC include
        PRIVATE src
    )

    target_link_libraries(mf_host_sdl 
        PUBLIC 
            MathFlow::host_core
        PRIVATE
            MathFlow::engine
            SDL2::SDL2 
            SDL2::SDL2main
            cjson
    )
    message(STATUS "Building mf_host_sdl (SDL2 found)")
else()
    message(WARNING "SDL2 not found. mf_host_sdl (SDL backend) will not be built. mf_host_core is available.")
endif()
