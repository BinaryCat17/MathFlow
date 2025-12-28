#include <mathflow/backend_cpu/mf_backend_cpu.h>
#include <mathflow/ops/mf_ops_core.h>
#include <mathflow/ops/mf_ops_array.h>
#include <mathflow/isa/mf_opcodes.h>
#include <string.h>

void mf_backend_cpu_init(mf_backend_dispatch_table* table) {
    memset(table, 0, sizeof(mf_backend_dispatch_table));
    
    // Register Core Operations (0-255)
    mf_ops_core_register(table);
    
    // Register Array Operations (256-511)
    mf_ops_array_register(table);
}