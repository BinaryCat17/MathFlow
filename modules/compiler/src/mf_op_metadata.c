#include <mathflow/compiler/mf_compiler.h>

const mf_op_metadata MF_OP_METADATA[MF_NODE_COUNT] = {
    [MF_NODE_UNKNOWN] = { "Unknown", 0, MF_OP_CAT_SPECIAL, MF_STRATEGY_DEFAULT, 0, 0, 0, 0, MF_ACCESS_SPECIAL, {NULL, NULL, NULL, NULL}, 0 },

#define MF_OP(_s, _n, _op, _cat, _strat, _in, _out, _t_rule, _s_rule, _a_rule, _p1, _p2, _p3, _p4, _kt, _ke, _ar) \
    [MF_NODE_##_s] = { \
        _n, \
        MF_OP_##_op, \
        _cat, \
        _strat, \
        _in, \
        _out, \
        _t_rule, \
        _s_rule, \
        _a_rule, \
        { _p1, _p2, _p3, _p4 }, \
        _ar \
    },

    MF_OP_LIST
#undef MF_OP
};