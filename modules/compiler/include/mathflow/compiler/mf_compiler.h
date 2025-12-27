#ifndef MF_COMPILER_H
#define MF_COMPILER_H

#include <mathflow/isa/mf_base.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/isa/mf_tensor.h>
#include <mathflow/vm/mf_memory.h> 

// --- Graph IR (Intermediate Representation) ---

typedef enum {
    MF_NODE_UNKNOWN = 0,
    
    // Inputs (Constants)
    MF_NODE_INPUT,  // Any constant tensor input
    
    // Arithmetic
    MF_NODE_ADD,
    MF_NODE_SUB,
    MF_NODE_MUL,
    MF_NODE_DIV,
    
    // Math
    MF_NODE_MIN,
    MF_NODE_MAX,
    MF_NODE_ABS,
    MF_NODE_CLAMP,
    MF_NODE_FLOOR,
    MF_NODE_CEIL,
    MF_NODE_SIN,
    MF_NODE_COS,
    MF_NODE_ATAN2,
    MF_NODE_SQRT,
    MF_NODE_POW,
    MF_NODE_MIX,
    MF_NODE_STEP,
    MF_NODE_SMOOTHSTEP,

    // Matrix
    MF_NODE_MATMUL,
    MF_NODE_TRANSPOSE,
    MF_NODE_INVERSE,
    MF_NODE_NORMALIZE,
    MF_NODE_DOT,
    MF_NODE_LENGTH,
    MF_NODE_JOIN, // Join/Pack

    // Comparison
    MF_NODE_LESS,
    MF_NODE_GREATER,
    MF_NODE_EQUAL,
    MF_NODE_NEQUAL,
    MF_NODE_LEQUAL,
    MF_NODE_GEQUAL,

    // Logic
    MF_NODE_AND,
    MF_NODE_OR,
    MF_NODE_XOR,
    MF_NODE_NOT,
    
    // Selection
    MF_NODE_SELECT, // Where/Select

    // Array Ops
    MF_NODE_RANGE,
    MF_NODE_CUMSUM,
    MF_NODE_COMPRESS,

    // State
    MF_NODE_MEMORY,

    MF_NODE_COUNT
} mf_node_type;

typedef struct {
    const char* id; 
    mf_node_type type;
    
    // Constant Data (valid if type == MF_NODE_INPUT)
    mf_tensor constant; 

    // Compiler Generated info
    u16 out_reg_idx;    // Index in the global Tensor Pool
    mf_tensor out_shape; // Predicted output shape and dtype
} mf_ir_node;

typedef struct {
    u32 src_node_idx; 
    u32 src_port;
    u32 dst_node_idx; 
    u32 dst_port;
} mf_ir_link;

typedef struct {
    mf_ir_node* nodes;
    size_t node_count;
    size_t node_cap;
    
    mf_ir_link* links;
    size_t link_count;
    size_t link_cap;
} mf_graph_ir;

// --- Compiler Interface ---

// 1. Parse JSON -> IR
bool mf_compile_load_json(const char* json_str, mf_graph_ir* out_ir, mf_arena* arena);

// 2. IR -> Program
mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena);

// 3. Save Program
bool mf_compile_save_program(const mf_program* prog, const char* path);

#endif // MF_COMPILER_H