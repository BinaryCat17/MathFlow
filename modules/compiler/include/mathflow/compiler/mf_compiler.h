#ifndef MF_COMPILER_H
#define MF_COMPILER_H

#include <mathflow/isa/mf_base.h>
#include <mathflow/isa/mf_program.h>
#include <mathflow/vm/mf_memory.h> // For mf_arena

// --- Graph IR (Intermediate Representation) ---

typedef enum {
    MF_NODE_UNKNOWN = 0,
    
    // Inputs
    MF_NODE_INPUT_F32,
    MF_NODE_INPUT_VEC2,
    MF_NODE_INPUT_VEC3,
    MF_NODE_INPUT_VEC4,
    MF_NODE_INPUT_BOOL,
    
    // Math
    MF_NODE_ADD_F32,
    MF_NODE_ADD_VEC3,
    MF_NODE_SCALE_VEC3, // vec3 * f32
    
    // Comparison (f32 -> bool)
    MF_NODE_GREATER_F32,
    MF_NODE_LESS_F32,
    MF_NODE_EQUAL_F32,
    
    // Logic (bool)
    MF_NODE_AND,
    MF_NODE_OR,
    MF_NODE_NOT,
    
    // Selection
    MF_NODE_SELECT_F32,  // bool ? f32 : f32
    MF_NODE_SELECT_VEC3, // bool ? vec3 : vec3
    MF_NODE_SELECT_VEC4, // bool ? vec4 : vec4

    MF_NODE_COUNT
} mf_node_type;

typedef struct {
    u32 id;             // JSON ID
    mf_node_type type;
    
    // Initial data (if Input node)
    union {
        f32 val_f32;
        mf_vec2 val_vec2;
        mf_vec3 val_vec3;
        mf_vec4 val_vec4;
        u8 val_bool;
    };

    // Compiler Generated info
    u16 out_reg_idx;    // Index in the Data Column
} mf_ir_node;

typedef struct {
    u32 src_id;
    u32 src_port;
    u32 dst_id;
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

// 2. IR -> Program (Allocation & Bytecode)
// Generates a complete program structure allocated in the arena.
mf_program* mf_compile(mf_graph_ir* ir, mf_arena* arena);

// 3. Save Program to Binary File
bool mf_compile_save_program(const mf_program* prog, const char* path);

#endif // MF_COMPILER_H
