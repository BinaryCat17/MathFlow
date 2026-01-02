#include <mathflow/isa/mf_opcodes.h>

const char* mf_opcode_to_str(u16 opcode) {
    switch (opcode) {
        case MF_OP_NOOP:      return "NOOP";
        case MF_OP_ADD:       return "ADD";
        case MF_OP_SUB:       return "SUB";
        case MF_OP_MUL:       return "MUL";
        case MF_OP_DIV:       return "DIV";
        case MF_OP_MIN:       return "MIN";
        case MF_OP_MAX:       return "MAX";
        case MF_OP_ABS:       return "ABS";
        case MF_OP_CLAMP:     return "CLAMP";
        case MF_OP_STEP:      return "STEP";
        case MF_OP_FLOOR:     return "FLOOR";
        case MF_OP_CEIL:      return "CEIL";
        case MF_OP_SIN:       return "SIN";
        case MF_OP_COS:       return "COS";
        case MF_OP_ATAN2:     return "ATAN2";
        case MF_OP_SQRT:      return "SQRT";
        case MF_OP_POW:       return "POW";
        case MF_OP_SUM:       return "SUM";
        case MF_OP_FMA:       return "FMA";
        case MF_OP_MATMUL:    return "MATMUL";
        case MF_OP_TRANSPOSE: return "TRANSPOSE";
        case MF_OP_INVERSE:   return "INVERSE";
        case MF_OP_DOT:       return "DOT";
        case MF_OP_LENGTH:    return "LENGTH";
        case MF_OP_NORMALIZE: return "NORMALIZE";
        case MF_OP_MIX:       return "MIX";
        case MF_OP_SMOOTHSTEP: return "SMOOTHSTEP";
        case MF_OP_JOIN:      return "JOIN";
        case MF_OP_LESS:      return "LESS";
        case MF_OP_GREATER:   return "GREATER";
        case MF_OP_EQUAL:     return "EQUAL";
        case MF_OP_NEQUAL:    return "NEQUAL";
        case MF_OP_LEQUAL:    return "LEQUAL";
        case MF_OP_GEQUAL:    return "GEQUAL";
        case MF_OP_AND:       return "AND";
        case MF_OP_OR:        return "OR";
        case MF_OP_XOR:       return "XOR";
        case MF_OP_NOT:       return "NOT";
        case MF_OP_SELECT:    return "SELECT";
        case MF_OP_RANGE:     return "RANGE";
        case MF_OP_INDEX:     return "INDEX";
        case MF_OP_GATHER:    return "GATHER";
        case MF_OP_CUMSUM:    return "CUMSUM";
        case MF_OP_COMPRESS:  return "FILTER";
        case MF_OP_COPY:      return "COPY";
        case MF_OP_SLICE:     return "SLICE";
        case MF_OP_RESHAPE:   return "RESHAPE";
        default:              return "UNKNOWN";
    }
}
