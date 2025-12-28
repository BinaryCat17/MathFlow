find_package(SDL2 CONFIG QUIET)

# --- Host Core (Manifest & Config) - No SDL dependency ---
add_library(mf_host_core STATIC
    src/mf_manifest_loader.c
)
add_library(MathFlow::host_core ALIAS mf_host_core)

target_include_directories(mf_host_core 
    PUBLIC include
    PRIVATE src
)
target_link_libraries(mf_host_core PUBLIC cjson)

# --- Host SDL (Window & Input) ---
if(TARGET SDL2::SDL2)
    add_library(mf_host STATIC
        src/mf_host_sdl.c
    )
    add_library(MathFlow::host_sdl ALIAS mf_host)

    target_include_directories(mf_host 
        PUBLIC include
        PRIVATE src
    )

    target_link_libraries(mf_host 
        PUBLIC 
            MathFlow::engine
            MathFlow::host_core
            SDL2::SDL2 
            SDL2::SDL2main
            cjson
    )
    message(STATUS "Building mf_host (SDL2 found)")
else()
    message(WARNING "SDL2 not found. mf_host (SDL backend) will not be built. mf_host_core is available.")
endif()
