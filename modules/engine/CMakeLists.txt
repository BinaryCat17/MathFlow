add_library(mf_engine STATIC
    src/mf_engine.c
)
add_library(MathFlow::engine ALIAS mf_engine)

target_include_directories(mf_engine 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(mf_engine 
    PUBLIC 
        MathFlow::isa
        MathFlow::vm 
        MathFlow::backend_cpu
)
