#include <mathflow/isa/mf_opcodes.h>
#include <mathflow/isa/mf_op_defs.h>
#include <stddef.h>
#include <stdbool.h>

static mf_runtime_op_metadata OP_METADATA[MF_OP_LIMIT] = {0};
static bool op_metadata_initialized = false;

static void init_op_metadata() {
    if (op_metadata_initialized) return;

#define MF_OP(suffix, op_name, op_suffix, cat, in_mask, out_mask, type_rule, shape_rule, access_rule, p1, p2, p3, p4, ktype, kernel, karity) \
    if ((int)MF_OP_##op_suffix < MF_OP_LIMIT) { \
        OP_METADATA[(int)MF_OP_##op_suffix].name = op_name; \
        OP_METADATA[(int)MF_OP_##op_suffix].ports[0] = p1; \
        OP_METADATA[(int)MF_OP_##op_suffix].ports[1] = p2; \
        OP_METADATA[(int)MF_OP_##op_suffix].ports[2] = p3; \
        OP_METADATA[(int)MF_OP_##op_suffix].ports[3] = p4; \
    }
    MF_OP_LIST
#undef MF_OP

    op_metadata_initialized = true;
}

const char* mf_opcode_to_str(u16 opcode) {
    init_op_metadata();
    if (opcode >= MF_OP_LIMIT || !OP_METADATA[opcode].name) return "UNKNOWN";
    return OP_METADATA[opcode].name;
}

const mf_runtime_op_metadata* mf_get_op_metadata(u16 opcode) {
    init_op_metadata();
    if (opcode >= MF_OP_LIMIT) return NULL;
    return &OP_METADATA[opcode];
}
