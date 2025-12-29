add_library(mf_backend_cpu STATIC
    src/mf_backend_cpu.c
)
add_library(MathFlow::backend_cpu ALIAS mf_backend_cpu)

target_include_directories(mf_backend_cpu 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(mf_backend_cpu 
    PUBLIC 
    MathFlow::base
    MathFlow::isa 
        MathFlow::ops
)
