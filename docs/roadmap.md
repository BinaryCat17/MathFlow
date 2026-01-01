# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Milestone

### Milestone 7: System Hardening & Reliability (Phases 31-35)
*Goal: Eliminate silent failures, fix critical compiler bugs, and ensure memory safety across all modules.*

*   **Phase 31: Compiler Integrity**
    *   [x] **Fix Topological Sort:** Implement proper cycle detection and reporting in `mf_topo_sort`.
    *   [x] **Strict Shape Analysis:** Remove risky fallbacks in `MatMul` and other ops; ensure all type mismatches are logged.
*   **Phase 32: Memory Safety & OOM Handling**
    *   [x] **Explicit OOM Checks:** Add return value validation for all `MF_ARENA_PUSH` and `mf_buffer_alloc` calls.
    *   [x] **Allocator Diagnostics:** Ensure `mf_arena_alloc` and `mf_heap_alloc` log errors on failure.
    *   [x] **Alignment Audit:** Ensure all internal buffers (especially for workers) are properly aligned for SIMD.
*   **Phase 33: Runtime Observability**
    *   [x] **Tensor Op Logs:** Add `MF_LOG_ERROR` to `Slice`, `Reshape`, and `Transpose` when bounds or shapes are invalid.
    *   [x] **Backend Robustness:** Validate program limits (e.g., Max Registers) and check worker initialization.
*   **Phase 34: Refactoring for Safety**
    *   [x] **Atomic Error Reporting:** Standardize how runtime errors are propagated from background threads to the main engine.
*   **Phase 35: Integration Testing**
    *   [x] **Negative Tests:** Create a test suite for "invalid" graphs (cycles, type mismatches, OOM) to verify diagnostic output.

---

## Upcoming Milestone

### Milestone 8: Architecture Purity & Performance (Phases 36-40)
*Goal: Transform the compiler into a meta-data driven system and unlock hardware-level optimizations.*

*   **Phase 36: Unified Operation Definitions (X-Macros)**
    *   [ ] **Single Source of Truth:** Create `mf_op_defs.h` with metadata: Opcode, Name, Category (UNARY, BINARY, etc.), and **Type Masks**.
    *   [ ] **Signature-Based Inference:** Add shape/type propagation rules directly into the macro to automate `mf_pass_analyze.c`.
    *   [ ] **Auto-Mapping:** Replace manual `NODE_MAP` and `PORT_MAP` in `mf_pass_lower.c` with macro-generated tables.
*   **Phase 37: Compiler Consolidation**
    *   [ ] **Generic Analyze Pass:** Use operation categories to handle shape inference logic.
    *   [ ] **Macro-Driven CodeGen:** Eliminate giant switch-cases in `mf_codegen.c` by expanding opcodes directly from definitions.
*   **Phase 38: Advanced Memory Access (Strides)**
    *   [ ] **Stride-Aware Kernels:** Refactor `modules/ops` to respect tensor strides.
    *   [ ] **Iterator API:** Implement a lightweight N-dimensional iterator.
    *   [ ] **Contiguous Fast-Path:** Optimize the iterator to collapse into simple pointer increments for dense data (inline/SIMD friendly).
*   **Phase 39: SIMD Acceleration**
    *   [ ] **SIMD Abstraction Layer:** Create `mf_simd.h` wrapping SSE/AVX/NEON intrinsics.
    *   [ ] **Vectorized Kernels:** Use X-Macros to automatically generate vectorized paths for `ELEMENTWISE` operations.
*   **Phase 40: Binary Hardening & Tooling**
    *   [ ] **Program Checksums:** Add CRC32 to `mf_program` to detect corruption.
    *   [ ] **Reflection API:** Expose kernel metadata to the Host for automated UI generation.

---

## Future Milestone

### Milestone 9: Intelligence & Developer Experience (Phases 41-43)
*Goal: Move from "Correct execution" to "Optimal execution" and seamless developer workflow.*

*   **Phase 41: Compiler Optimizations**
    *   [ ] **Constant Folding:** Pre-calculate operations on constants during compilation.
    *   [ ] **Dead Code Elimination (DCE):** Remove nodes that don't contribute to any `Output`.
*   **Phase 42: Resource Management & Aliasing**
    *   [ ] **Memory Aliasing:** Analyze resource life-cycles to reuse memory buffers for temporary data.
    *   [ ] **Dependency Graph:** Build an explicit graph of kernel dependencies to optimize execution order.
*   **Phase 43: Live Systems**
    *   [ ] **Hot Reloading:** Implement file-watchers in the Host to recompile and swap `mf_program` without restarting the engine.
    *   [ ] **Telemetry:** Real-time performance counters (ns per kernel, memory bandwidth).

---

## Completed Phases (Archive)

### Milestone 6: Pixel Engine & Framework Maturity (Phases 19 - 30)
- **Architecture:** Transitioned to an explicit Pipeline model with modular Kernels and global resource bindings.
- **Compute:** Unified "Script" and "Shader" modes under a single Data-Driven Dispatch system.
- **Data Purity:** Refactored `mf_tensor` into Metadata, Storage, and View, enabling O(1) slicing and reshaping.
- **State Safety:** Enforced full Double-Buffering for all global resources and implemented explicit memory ownership.
- **Compiler:** Modularized into a pass-based pipeline with source tracking, static analysis, and strong typing.
- **Features:** Added SDF text rendering, UTF-32 string support, ternary operations, and random-access memory (`Gather`).
- **Framework:** Refactored `host` module to manage unified application lifecycles, asset loading, and platform-specific drivers (SDL/Headless).

### Milestone 5: Architecture Purity (Phase 17.6 - 18)- **Ops Isolation:** Moved dispatch logic from ISA to Ops. `mf_backend` is now a pure interface.
- **State Separation:** `mf_state` (Data) is fully decoupled from `mf_exec_ctx` (Execution).
- **Error Handling:** Implemented robust error propagation from parallel workers to the Engine.
- **Module Hierarchy:** Established clear layering: Foundation (ISA/Base) -> Implementation (Ops/Backend) -> Core (Engine/Compiler) -> Host.

### Milestone 4: Architecture Cleanup (Phase 16)
- **Modularization:** Decomposed monolithic build into `base`, `isa`, `vm`, `compiler`, `engine`, `host`, `backend`.
- **Decoupling:** Removed circular dependencies (`Ops->VM`, `Engine->Threads`, `VM->IO`).
- **Purity:** VM is now IO-agnostic. Asset loading moved to Host.
- **Single Source of Truth:** Removed `mf_instance`, unified execution under `mf_engine_dispatch`.
- **State Propagation:** Automatic data transfer from Main VM to Worker VMs.

### Milestone 3: Engine Unification & Apps (Phases 11-15)
- **Execution Unification:** Merged Scheduler into VM. VM is now the single execution entity supporting both serial and parallel (tiled) modes.
- **Architecture:** Unified `mf_engine` API, stateless VM, and consolidated `base`/`ops` modules.
- **Performance:** Multi-threaded Thread Pool moved to `base` for generic use.
- **Application Layer:** Manifest-driven runtime (`.mfapp`) with automated input handling and window management.

### Milestone 2: Visuals & Modularity (Phases 8-10)
- **Rendering:** SDF-based Pixel Math engine with anti-aliasing (`SmoothStep`, `Circle`).
- **Composition:** Sub-graph system with recursive inlining and interface mapping.
- **Visualizer:** SDL2-based host (`mf-window`) for real-time math-to-pixel execution.

### Milestone 1: Core Foundation (Phases 1-7)
- **Virtual Machine:** Bytecode compiler and SoA-based execution engine in pure C11.
- **Memory:** Dual-allocator system (Static Arena + Dynamic Heap).
- **State:** Support for inter-frame persistence via `Memory` nodes and cycle-breaking.
