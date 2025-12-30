#ifndef MF_PIPELINE_H
#define MF_PIPELINE_H

#include <mathflow/base/mf_types.h>
#include <mathflow/isa/mf_tensor.h>

// Description of a Global Resource (Blackboard Buffer)
typedef struct {
    const char* name;
    mf_dtype dtype;
    int32_t shape[MF_MAX_DIMS];
    uint8_t ndim;
    bool persistent; // If true, Engine manages Ping-Pong buffers for this resource
} mf_pipeline_resource;

// Mapping between a Kernel's internal Symbol and a Global Resource
typedef struct {
    const char* kernel_port;     // Symbol name in the .json/.bin
    const char* global_resource; // Resource name defined in mf_pipeline_desc
} mf_pipeline_binding;

// Description of a single execution unit (Shader/Kernel)
typedef struct {
    const char* id;
    const char* graph_path; // Path to .json or .bin
    uint32_t frequency;     // 1 = every frame, N = N times per frame
    
    mf_pipeline_binding* bindings;
    uint32_t binding_count;
} mf_pipeline_kernel;

// Complete Pipeline Configuration (from .mfapp)
typedef struct {
    mf_pipeline_resource* resources;
    uint32_t resource_count;
    
    mf_pipeline_kernel* kernels;
    uint32_t kernel_count;
} mf_pipeline_desc;

#endif // MF_PIPELINE_H
