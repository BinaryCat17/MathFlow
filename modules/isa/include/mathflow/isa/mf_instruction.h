#ifndef MF_INSTRUCTION_H
#define MF_INSTRUCTION_H

#include <mathflow/base/mf_types.h>
#include "mf_opcodes.h"

/**
 * @brief Standard 4-address code instruction.
 * 
 * Layout: [ Opcode (16) | Dest (16) | Src1 (16) | Src2 (16) | Src3 (16) | Src4 (16) ]
 * Total Size: 12 bytes.
 * 
 * Strides are no longer stored per-instruction. Instead, they are inferred 
 * at runtime from the tensor's identity (Spatial vs Uniform).
 */
typedef struct mf_instruction {
    u16 opcode;
    u16 dest_idx;
    u16 src1_idx;
    u16 src2_idx;
    u16 src3_idx;
    u16 src4_idx;
    
    // Linear strides for [Dest, Src1, Src2, Src3]
    // 0 = Constant/Uniform, 1 = Spatial/Linear, -1 = Reduction
    i32 strides[4]; 
} mf_instruction;

#endif // MF_INSTRUCTION_H
