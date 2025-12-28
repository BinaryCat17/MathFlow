add_library(mf_compiler STATIC
    src/mf_compiler.c
    src/mf_json_parser.c
    src/mf_semantics.c
    src/mf_codegen.c
)
add_library(MathFlow::compiler ALIAS mf_compiler)

target_include_directories(mf_compiler 
    PUBLIC include
    PRIVATE src
)

# FIXED: Removed dependency on mf_vm. 
# Compiler only needs ISA definitions and cJSON for parsing.
target_link_libraries(mf_compiler 
    PUBLIC 
        MathFlow::isa
        MathFlow::base
    PRIVATE
        cjson
)
