add_library(mf_ops STATIC
    src/mf_ops_core.c
    src/mf_ops_array.c
)
add_library(MathFlow::ops ALIAS mf_ops)

target_include_directories(mf_ops 
    PUBLIC include
    PRIVATE src
)

# TODO: Break dependency on VM. Ops should be pure math kernels.
target_link_libraries(mf_ops 
    PUBLIC 
        MathFlow::vm 
        m
)
