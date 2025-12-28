add_library(mf_isa INTERFACE)
add_library(MathFlow::isa ALIAS mf_isa)

target_include_directories(mf_isa INTERFACE
    include
)
