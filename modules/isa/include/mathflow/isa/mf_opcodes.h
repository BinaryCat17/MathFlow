#ifndef MF_OPCODES_H
#define MF_OPCODES_H

#include <mathflow/base/mf_types.h>
#include <mathflow/isa/mf_op_defs.h>

// MathFlow Instruction Set Architecture (Opcodes)

#define MF_OP_LIMIT 1024

typedef enum {
#define MF_OPCODE(suffix, value) MF_OP_##suffix = value,
    MF_OPCODE_LIST
#undef MF_OPCODE

    // Special Markers (Optional, for validation)
    MF_OP_CORE_BEGIN  = 0,
    MF_OP_CORE_END    = 255,
    MF_OP_ARRAY_BEGIN = 256,
    MF_OP_ARRAY_END   = 511,
    MF_OP_STATE_BEGIN = 512,
    MF_OP_STATE_END   = 767,

} mf_opcode;

/**
 * @brief Runtime metadata for an operation.
 */
typedef struct {
    const char* name;
    const char* ports[4]; // Names of input ports (src1, src2, src3, src4)
} mf_runtime_op_metadata;

/**
 * @brief Returns a human-readable name for a given opcode.
 */
const char* mf_opcode_to_str(u16 opcode);

/**
 * @brief Returns runtime metadata for a given opcode.
 */
const mf_runtime_op_metadata* mf_get_op_metadata(u16 opcode);

#endif // MF_OPCODES_H