add_library(mf_isa STATIC
    src/mf_tensor.c
    src/mf_opcodes.c
)
add_library(MathFlow::isa ALIAS mf_isa)

target_include_directories(mf_isa PUBLIC
    include
)

target_link_libraries(mf_isa PUBLIC
    MathFlow::base
)