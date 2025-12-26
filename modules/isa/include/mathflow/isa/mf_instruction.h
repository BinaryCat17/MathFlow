#ifndef MF_INSTRUCTION_H
#define MF_INSTRUCTION_H

#include "mf_base.h"
#include "mf_opcodes.h"

// Standard 3-address code instruction (64 bits total)
// [ Opcode (16) | Dest (16) | Src1 (16) | Src2 (16) ]
typedef struct {
    u16 opcode;
    u16 dest_idx;
    u16 src1_idx;
    u16 src2_idx;
} mf_instruction;

#endif // MF_INSTRUCTION_H
