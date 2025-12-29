add_library(mf_ops STATIC
    src/mf_ops_array.c
    src/mf_ops_core.c
    src/mf_ops_math.c
    src/mf_ops_logic.c
    src/mf_ops_matrix.c
    src/mf_ops_state.c
)
add_library(MathFlow::ops ALIAS mf_ops)

target_include_directories(mf_ops 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(mf_ops 
    PUBLIC 
        MathFlow::isa
        m
)
