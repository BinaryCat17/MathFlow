add_library(mf_loader STATIC
    src/mf_loader.c
)
add_library(MathFlow::loader ALIAS mf_loader)

target_include_directories(mf_loader 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(mf_loader 
    PRIVATE
        MathFlow::engine
        MathFlow::compiler
        MathFlow::backend_cpu
)
