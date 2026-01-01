add_library(mf_compiler STATIC
    src/mf_compiler.c
    src/mf_op_metadata.c
    src/passes/mf_pass_lower.c
    src/passes/mf_pass_inline.c
    src/passes/mf_pass_analyze.c
    src/passes/mf_pass_domain_split.c
    src/passes/mf_pass_fuse.c
    src/passes/mf_pass_liveness.c
    src/mf_json_parser.c
    src/mf_codegen.c
    src/mf_graph_utils.c
)
add_library(MathFlow::compiler ALIAS mf_compiler)

target_include_directories(mf_compiler 
    PUBLIC include
    PRIVATE src
)

# Compiler only needs ISA definitions and cJSON for parsing.
target_link_libraries(mf_compiler 
    PUBLIC 
        MathFlow::isa
        MathFlow::base
)
