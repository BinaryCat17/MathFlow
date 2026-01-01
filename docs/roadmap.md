# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Milestone

### Milestone 8: Architecture Purity & Performance (Phases 36-40)
*Goal: Transform the compiler into a meta-data driven system and unlock hardware-level optimizations.*

*   **Phase 36: Unified Operation Definitions (X-Macros)**
    *   [x] **Single Source of Truth:** Create `mf_op_defs.h` with metadata: Opcode, Name, Category (UNARY, BINARY, etc.), and **Type Masks**.
    *   [x] **Signature-Based Inference:** Add shape/type propagation rules directly into the macro to automate `mf_pass_analyze.c`.
    *   [x] **Auto-Mapping:** Replace manual `NODE_MAP` and `PORT_MAP` in `mf_pass_lower.c` with macro-generated tables.
*   **Phase 37: Compiler Consolidation**
    *   [x] **Generic Analyze Pass:** Use operation categories to handle shape inference logic.
    *   [x] **Macro-Driven CodeGen:** Eliminate giant switch-cases in `mf_codegen.c` by expanding opcodes directly from definitions.
*   **Phase 38: Advanced Memory Access (Strides)**
    *   [x] **Stride-Aware Kernels:** Refactor `modules/ops` to respect tensor strides.
    *   [x] **Iterator API:** Implement a lightweight N-dimensional iterator.
    *   [x] **Contiguous Fast-Path:** Optimize the iterator to collapse into simple pointer increments for dense data (inline/SIMD friendly).
*   **Phase 39: Code Quality & Refactoring (Cleanup)**
    *   [x] **Matrix Indexing Cleanup:** Standardize indexing in `mf_ops_matrix.c`, removing manual offset calculations where possible.
    *   [x] **Kernel Macro Overhaul:** Simplify `mf_kernel_utils.h` to reduce boilerplate and improve readability of generated kernels.
    *   [x] **Compiler Simplification:** Unify JSON parsing in `Lower` pass and move broadcast logic to `Base` module.

---

## Upcoming Milestone

### Milestone 10: Hardware Acceleration & JIT (Phases 47-50)
*Goal: Move from interpreted bytecode to machine code or GPU kernels.*

*   **Phase 47: LLVM/Cranelift Backend (Prototype)**
    *   [ ] **IR Translation:** Map MathFlow instructions to JIT IR.
*   **Phase 48: SIMD Auto-Vectorization**
    *   [ ] **AVX/NEON Kernels:** Use JIT to emit vectorized element-wise loops.

---

## Completed Phases (Archive)

### Milestone 9: The Task-Based Evolution (The Multi-Domain Engine)
- **Multi-Domain Support:** Implemented domain splitting and task-based execution (Phases 40-44).
- **Advanced Lowering:** Implemented `MEAN` decomposition and `FMA` fusion (Phase 45).
- **Register Allocation:** Implemented Liveness Analysis and Register Reuse (Phase 46).

### Milestone 7: System Hardening & Reliability (Phases 31-35)
- **Compiler:** Fixed Topological Sort and enforced strict Shape Analysis (Phase 31).
- **Safety:** Implemented explicit OOM checks, alignment audits, and atomic error reporting (Phases 32, 34).
- **Observability:** Added detailed tensor op logs and backend diagnostics (Phase 33).
- **Testing:** Created negative test suite for invalid graphs (Phase 35).

### Milestone 6: Pixel Engine & Framework Maturity (Phases 19 - 30)
- **Architecture:** Transitioned to an explicit Pipeline model with modular Kernels and global resource bindings.
- **Compute:** Unified "Script" and "Shader" modes under a single Data-Driven Dispatch system.
- **Data Purity:** Refactored `mf_tensor` into Metadata, Storage, and View, enabling O(1) slicing and reshaping.
- **State Safety:** Enforced full Double-Buffering for all global resources and implemented explicit memory ownership.
- **Compiler:** Modularized into a pass-based pipeline with source tracking, static analysis, and strong typing.
- **Features:** Added SDF text rendering, UTF-32 string support, ternary operations, and random-access memory (`Gather`).

### Milestone 5: Architecture Purity (Phase 17.6 - 18)
- **Ops Isolation:** Moved dispatch logic from ISA to Ops. `mf_backend` is now a pure interface.
- **State Separation:** `mf_state` (Data) is fully decoupled from `mf_exec_ctx` (Execution).

### Milestone 4: Architecture Cleanup (Phase 16)
- **Modularization:** Decomposed monolithic build into `base`, `isa`, `vm`, `compiler`, `engine`, `host`, `backend`.
- **Decoupling:** Removed circular dependencies.

### Milestone 3: Engine Unification & Apps (Phases 11-15)
- **Execution Unification:** Merged Scheduler into VM. VM is now the single execution entity.
- **Application Layer:** Manifest-driven runtime (`.mfapp`) with automated input handling.

### Milestone 2: Visuals & Modularity (Phases 8-10)
- **Rendering:** SDF-based Pixel Math engine.
- **Composition:** Sub-graph system with recursive inlining.

### Milestone 1: Core Foundation (Phases 1-7)
- **Virtual Machine:** Bytecode compiler and SoA-based execution engine in pure C11.
- **Memory:** Dual-allocator system (Static Arena + Dynamic Heap).