#ifndef MF_INSTRUCTION_H
#define MF_INSTRUCTION_H

#include <mathflow/base/mf_types.h>
#include "mf_opcodes.h"

// Standard 3-address code instruction
// Updated to support 3 source operands (e.g. Select, Clamp, Mix)
// Layout: [ Opcode (16) | Dest (16) | Src1 (16) | Src2 (16) | Src3 (16) | Reserved (16) ]
// Total Size: 96 bits (12 bytes).
// Note: We keep it packed/minimal. 
typedef struct {
    u16 opcode;
    u16 dest_idx;
    u16 src1_idx;
    u16 src2_idx;
    u16 src3_idx;
    u16 padding;

    // Element-strides for [Dest, Src1, Src2, Src3]
    // Determines pointer advancement per domain element.
    // 0 = Constant/Broadcast, 1 = Sequential, C = Channels
    i32 strides[4]; 
} mf_instruction;

#endif // MF_INSTRUCTION_H