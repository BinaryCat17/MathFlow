# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Phase 19: Pipeline Architecture (Explicit Kernels) (Completed)
**Objective:** Transition from a "Monolithic Script" to a "System of Interacting Programs". We abandon automatic graph partitioning in favor of explicit, modular Kernels (Programs) connected via a Global State. This mirrors GPU Compute Shader architectures.

> **Philosophy:** "1 JSON = 1 Kernel". A graph defines a stateless function $Y = F(X)$. State persistence and communication between Kernels happen strictly via external Buffers managed by the Engine.

- [x] **Step 1: Compiler & ISA Cleanup:**
    - **Remove `Memory` Nodes:** Delete internal state logic (`MF_NODE_MEMORY`). All state must be passed via Input/Output.
    - **Multi-Output Support:** Ensure Compiler correctly exports metadata for graphs with multiple `Output` nodes (MRT - Multiple Render Targets).
    - **ISA Update:** Update `mf_program` to list explicit Input/Output symbol requirements.

- [x] **Step 2: The Pipeline Manifest (.mfapp Extension):**
    - Extend the existing `.mfapp` JSON format to describe the System:
        - **Resources:** Global Buffers (e.g., `Physics.Pos`, `Render.Color`).
        - **Kernels:** List of programs to load (`physics.bin`, `render.bin`).
        - **Bindings:** Map Kernel I/O to Global Resources (e.g., `Physics.Out -> State.Pos`, `Render.In -> State.Pos`).
        - **Scheduling:** Execution order and frequency (e.g., Physics x10, Render x1).

- [x] **Step 3: Engine Orchestrator:**
    - Implement `mf_pipeline` loader.
    - **Resource Manager:** Allocates global buffers (including Ping-Pong pairs for time-dependent data).
    - **Scheduler:** Replaces the single `dispatch` call with a loop over active Kernels, binding the correct buffers (Zero-Copy) before execution.

- [x] **Step 4: Dual-Use Graphs:**
    - Verify that the same JSON graph can be used as:
        1.  **Inline Subgraph:** Embedded into another graph (Compiled to one binary, max performance).
        2.  **Standalone Kernel:** Running independently in the pipeline (Modular, easy testing).

## Phase 20: Compute Unification (Data-Driven Dispatch)
**Objective:** Remove artificial distinctions between "Script" and "Shader". The execution domain is strictly defined by the data (Output Tensors).

- [x] **Step 1: API Update (Domain Tensor):**
    - Change `mf_engine_dispatch` and `mf_backend_dispatch` to accept `mf_tensor* domain` instead of `x, y`.
    - This tensor acts as the iteration space. It can be a real resource (e.g. `out_Color`) or a dummy layout.

- [x] **Step 2: N-Dimensional Context:**
    - Update `mf_exec_ctx` to replace hardcoded 2D/3D fields with `u32 tile_offset[MF_MAX_DIMS]` and `u32 domain_shape[MF_MAX_DIMS]`.

- [x] **Step 3: Backend "Flat Scheduling":**
    - Implement **Linearization**: The Thread Pool works with a flat job index (0 to TotalElements).
    - Implement **Unflattening**: Workers convert the flat job ID back to N-dim coordinates (`ctx.tile_offset`) at the start of execution.
    - Implement **Fast Path**: If `total_elements < THRESHOLD`, bypass the thread pool and execute inline (replaces the old "Script Mode").

- [x] **Step 4: Opcode Update:**
    - Update `MF_OP_INDEX` to read from the new N-dim `ctx.tile_offset`.
    - **Decision:** No new `OP_INVOCATION_ID` instruction needed.

## Phase 21: The Pure State Machine (Explicit Symbols & Auto-Buffering) (Completed)
**Objective:** Make the engine fully deterministic and stateless by enforcing Double Buffering for ALL resources. Eliminate heuristics and the `persistent` flag.

- [x] **Step 1: ISA Update (Symbol Flags):**
    - Update `mf_bin_symbol` to include `u8 flags`.
    - Define `MF_SYMBOL_INPUT` (Read-Only) and `MF_SYMBOL_OUTPUT` (Write-Only).
    - Bump `MF_BINARY_VERSION` to 8.

- [x] **Step 2: Compiler Update:**
    - Update `mf_codegen.c` to identify and tag symbol types based on Node Type (`MF_NODE_INPUT` vs `MF_NODE_OUTPUT`).

- [x] **Step 3: Engine Architecture Update:**
    - **Remove `persistent` flag:** Every global resource is now double-buffered (Front/Back) by default.
    - Logic:
        - `INPUT` symbols bind to **Front Buffer** (Previous State).
        - `OUTPUT` symbols bind to **Back Buffer** (Next State).
    - **Auto-Swap:** Engine swaps Front/Back pointers for all resources at the end of the frame.
    - Remove string-based heuristics ("out_").

## Phase 22: Data Purity (Tensor Refactor) (Completed)
**Objective:** Resolve the ambiguity of the `mf_tensor` structure by separating Metadata, Storage, and View. This enables Zero-Copy Slicing, safer memory management, and simplified GPU interoperability.

- [x] **Step 1: Structural Split:**
    - `mf_type_info`: Pure metadata (DType, Shape, Strides). Lightweight "value semantics".
    - `mf_buffer`: Raw memory handle (`void*`, `size_t`, `allocator`). Owns the memory.
    - `mf_tensor`: A **View** combining `type_info`, `buffer`, and a `byte_offset`.
- [x] **Step 2: ISA Update:**
    - Update `mf_program` to store a table of `mf_type_info` separately.
    - Constant Data becomes a single monolithic `mf_buffer`.
- [x] **Step 3: Zero-Copy Mechanics:**
    - Implement `mf_tensor_slice()`: Create a new tensor view pointing to a subset of data (modifying `offset` and `shape`) without allocation.
    - Update `mf_engine` to manage `mf_buffer` (A/B) swapping while keeping `mf_tensor` views stable.
- [x] **Step 4: O(1) Tensor Ops:**
    - Reimplement `Slice`, `Reshape`, and `Transpose` to modify Metadata/Offset only (no allocation/copy).
    - Note: Matrix ops use a fallback "ensure_contiguous" copy for strided inputs until generic iterators are implemented.
- [x] **Step 5: Windowed Execution (Backend Optimization):**
    - Update `mf_backend_cpu` to create "Window Views" for worker threads.
    - Workers see a local tensor (0..width) that maps to the global buffer via strides/offsets.

## Phase 22.6: ISA Extension & Ternary Ops (Completed)
**Objective:** Expand the Instruction Set Architecture to support 3-operand instructions (ternary operations), bringing the engine closer to GPU standards (Mix, Clamp, FMA) and simplifying control flow.

- [x] **Step 1: ISA Update:**
    - Expanded `mf_instruction` struct to support 3 source operands (`src3_idx`).
    - Added `MF_OP_SELECT` (ternary) and `MF_OP_CLAMP` (ternary).
    - Removed legacy `WHERE_TRUE` / `WHERE_FALSE` hacks.
- [x] **Step 2: Core Refactor:**
    - Updated `mf_op_func` signature to accept `const mf_instruction*` instead of individual indices, ensuring future extensibility.
    - Refactored all math kernels to use the new signature.
- [x] **Step 3: Compiler Update:**
    - Updated CodeGen to emit native single instructions for `Select` and `Clamp` instead of decomposed chains.
    - Enforced string-based port names in JSON (removed numeric port support).

## Phase 22.5: SubGraph Interface & Named Ports (Completed)
**Objective:** Simplify the SubGraph system by removing dedicated `ExportInput`/`ExportOutput` nodes and numerical indices in JSON. Transition to purely named ports.

- [x] **Step 1: Named Links:**
    - Update JSON Parser to accept strings for `src_port` and `dst_port` in links.
    - Compiler resolves these names to internal indices during the build process.
- [x] **Step 2: Interface Definition:**
    - `Input` nodes inside a SubGraph automatically become named input ports.
    - `Output` nodes automatically become named output ports.
    - Remove `ExportInput`/`ExportOutput`.
- [x] **Step 3: Call Node Update:**
    - The `Call` node dynamically exposes ports matching the `Input`/`Output` names of the referenced graph.

## Phase 23: Compiler Modularization (Completed)
**Objective:** Decompose the monolithic `mf_json_parser.c` into a pipeline of independent passes. This prepares the ground for advanced features like Generics and Optimizations.

- [x] **Step 1: AST Separation & Source Tracking:**
    - Create a distinct `mf_ast` (Abstract Syntax Tree) representing the raw JSON structure.
    - **Improvement:** Embed source location (line/column) in AST nodes for precise error reporting.
    - Implement `JSON -> AST` parser.
- [x] **Step 2: Pass Architecture:**
    - Organize passes in `modules/compiler/src/passes/`.
    - `Pass_Desugar`: Converts legacy nodes to standard `Input`/`Const` nodes.
    - `Pass_Inline`: Recursively expands Subgraphs (`MF_NODE_CALL`).
    - `Pass_Lower`: Converts AST to IR (Index allocation, basic validation).
- [x] **Step 3: Clean Implementation:**
    - Remove logic duplication between Loader and Compiler.
    - Ensure strict separation: Parser only parses, Compiler only compiles.
- [x] **Step 4: Remove cJSON Dependency:**
    - Implemented custom JSON parser (`modules/base/src/mf_json.c`) with source tracking.
    - Removed `cJSON` from all CMake lists and vcpkg.

## Phase 24: Strong Typing (Static Analysis) (Completed)
**Objective:** Prevent runtime errors and undefined behavior by enforcing type safety at compile time, providing rich error messages with source locations.

- [x] **Step 1: Source Tracking:**
    - Add `mf_source_loc` to `mf_ir_node` to retain line/column info from AST.
    - Update `mf_pass_lower` to propagate this metadata.
- [x] **Step 2: Analysis Pass (`mf_pass_analyze`):**
    - Implement a dedicated pass that runs *before* CodeGen.
    - Move shape inference logic from `mf_semantics.c` to this pass.
    - Implement rigorous Type Checking (DType compatibility).
    - Validate Shape Broadcasting rules.
- [x] **Step 3: CodeGen Simplification:**
    - Strip inference logic from `mf_codegen_emit`.
    - CodeGen should solely rely on the pre-computed shapes/types from the Analysis pass.
- [x] **Step 4: Error Reporting:**
    - Use `mf_source_loc` to print GCC-style error messages (e.g., `graph.json:15:4: error: shape mismatch`).

## Phase 25: Zero-Overhead Linking (Hash-Based)
**Objective:** Optimize the binding process and reduce memory overhead by replacing string comparisons with hash lookups.

- [ ] **Step 1: ISA Hashing:**
    - Update `mf_bin_symbol` to store `u32 name_hash` (FNV-1a).
    - Keep string names in a separate "Debug Table" (stripped in Release builds).
- [ ] **Step 2: Engine Update:**
    - `mf_engine_bind_pipeline` matches resources using hashes (`O(1)`).
- [ ] **Step 3: Loader Update:**
    - Compute hashes during JSON parsing/Binary loading.
    - Ensure collision detection (optional but recommended for robustness).

---

## Completed Phases (Archive)

### Milestone 5: Architecture Purity (Phase 17.6 - 18)
- **Ops Isolation:** Moved dispatch logic from ISA to Ops. `mf_backend` is now a pure interface.
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
