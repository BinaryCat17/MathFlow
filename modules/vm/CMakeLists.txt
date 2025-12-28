add_library(mf_vm STATIC
    src/mf_vm.c
)
add_library(MathFlow::vm ALIAS mf_vm)

target_include_directories(mf_vm 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(mf_vm 
    PUBLIC 
        MathFlow::isa 
        MathFlow::base
)
