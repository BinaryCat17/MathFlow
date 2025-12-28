add_library(mf_isa INTERFACE)
add_library(MathFlow::isa ALIAS mf_isa)

target_include_directories(mf_isa INTERFACE
    include
)

target_link_libraries(mf_isa INTERFACE
    MathFlow::base
)
