#include <mathflow/compiler/mf_compiler.h>

const mf_op_metadata MF_OP_METADATA[MF_NODE_COUNT] = {
    [MF_NODE_UNKNOWN] = { "Unknown", 0, MF_OP_CAT_SPECIAL, 0, 0, 0, MF_ACCESS_SPECIAL, {NULL, NULL, NULL} },

#define MF_OP(suffix, json_name, opcode, cat, mask, out_r, shape_r, access_r, p1, p2, p3) \
    [MF_NODE_##suffix] = { \
        json_name, \
        opcode, \
        cat, \
        mask, \
        out_r, \
        shape_r, \
        access_r, \
        {p1, p2, p3} \
    },

    MF_OP_LIST
#undef MF_OP
};
