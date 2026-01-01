#ifndef MF_COMPILER_H
#define MF_COMPILER_H

#include <mathflow/base/mf_types.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/base/mf_memory.h>

#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_op_defs.h>

// --- Diagnostics ---
typedef struct {
    mf_source_loc loc;
    char message[256];
} mf_compiler_error;

typedef struct {
    mf_compiler_error* errors;
    uint32_t error_count;
    uint32_t error_capacity;
    bool has_error;
} mf_compiler_diag;

void mf_compiler_diag_init(mf_compiler_diag* diag, mf_arena* arena);
void mf_compiler_diag_report(mf_compiler_diag* diag, mf_source_loc loc, const char* fmt, ...);

// --- IR Definitions ---

typedef enum {
    MF_NODE_UNKNOWN = 0,
    
#define MF_OP(suffix, name, opcode, cat, mask, out_rule, shape_rule, p1, p2, p3) MF_NODE_##suffix,
    MF_OP_LIST
#undef MF_OP

    MF_NODE_COUNT
} mf_node_type;

typedef struct {
    const char* name;
    u16 opcode;
    mf_op_category category;
    u32 type_mask;
    mf_out_rule out_rule;
    mf_shape_rule shape_rule;
    const char* ports[3];
} mf_op_metadata;

extern const mf_op_metadata MF_OP_METADATA[MF_NODE_COUNT];

typedef struct {
    const char* id; 
    mf_node_type type;
    
    // Constant Data (valid if type == MF_NODE_INPUT)
    mf_tensor constant; 

    // Sub-Graph Data
    const char* sub_graph_path; // For MF_NODE_CALL

    // Debug Info
    mf_source_loc loc;

    // Compiler Generated info
    u16 out_reg_idx;    // Index in the global Tensor Pool
    mf_tensor out_shape; // Predicted output shape and dtype
} mf_ir_node;

typedef struct {
    u32 src_node_idx; 
    u32 src_port;
    const char* src_port_name;

    u32 dst_node_idx; 
    u32 dst_port;
    const char* dst_port_name;
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
bool mf_compile_load_json(const char* json_path, mf_graph_ir* out_ir, mf_arena* arena, mf_compiler_diag* diag);

// 2. IR -> Program
mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena, mf_compiler_diag* diag);

// 3. Save Program
bool mf_compile_save_program(const mf_program* prog, const char* path);

#endif // MF_COMPILER_H
