# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Phase 16.10: Data Propagation & API Cleanup (Completed)
**Objective:** Remove low-level execution callbacks (`setup_cb`, `finish_cb`, `mf_job_handle`) from the public API. Implement automatic state propagation so the Host only interacts with the Main Engine State.

- [x] **Step 1: Backend Context Access:** Update `mf_backend_dispatch_func` to accept the `Main VM` state (or its memory pointers) as a read-only source.
- [x] **Step 2: Worker Initialization:** Modify CPU Backend workers to automatically bind to the Main VM's constants/uniforms (Zero-Copy or Fast-Copy).
- [x] **Step 3: Output Consolidation:** Implement a mechanism where parallel workers write directly to the Main VM's output tensor buffers (slicing/tiling handled internally).
- [x] **Step 4: API Purification:** Remove job-related types and callbacks from `mf_engine.h`. `mf_engine_dispatch` becomes a simple call: `dispatch(engine, x, y)`.

## Phase 16.11: Runtime Optimization & Lifecycle
**Objective:** Optimize CPU worker memory usage and improve engine lifecycle management. Currently, workers use `mf_heap` (slow) for temporary tensors, and `mf_engine` lacks a reset mechanism.

- [ ] **Step 1: Arena Realloc:** Implement "dumb realloc" (Alloc + Copy) in `mf_arena` to support tensor resizing on linear memory.
- [ ] **Step 2: Worker Memory Optimization:** Switch Backend CPU workers from `mf_heap` (Free List) to `mf_arena` (Linear) for temporary frame data. This removes allocation overhead per frame.
- [ ] **Step 3: Engine Reset:** Implement `mf_engine_reset()` to allow clearing the Heap and Arena pointers without destroying OS threads/buffers, enabling efficient graph hot-reloading.

## Phase 17: Heterogeneous Compute Architecture (The Map Op)
**Objective:** Prepare the architecture for hybrid CPU/GPU execution by moving from "Flat Inlined Graphs" to a "Kernel + Dispatch" model. Instead of imperative function calls, we introduce `OP_MAP` — a functional primitive ideal for SIMD and Compute Shaders.

- [ ] **Step 1: Kernel Definition:** Extend `mf_program` to support multiple "Kernels" (independent bytecode blobs). The Main Graph becomes just a coordinator that dispatches data to Kernels.
- [ ] **Step 2: ISA Update (OP_MAP):** Introduce `MF_OP_MAP`.
    - **Semantics:** "Apply Kernel K to every element of Tensor T".
    - **Usage:** VM sees `OP_MAP`, delegates to Backend. Backend decides *where* to run it (CPU Threads or GPU).
- [ ] **Step 3: Compiler Evolution (Kernel Extraction):** Modify `mf_compiler` to stop inlining "Pure" sub-graphs. Instead, compile them into separate Kernels and emit `OP_MAP` instructions in the main flow.
- [ ] **Step 4: Backend Implementation:**
    - Update `backend_cpu` to handle `map` by spinning up the Thread Pool (replacing the current global dispatch logic).
    - *Future Proofing:* This structure maps 1:1 to `vkCmdDispatch` in Vulkan.

## Phase 18: Advanced State Management (Double Buffering)
**Objective:** Enable parallel execution for graphs with state (Memory Nodes) by implementing a double-buffering mechanism. This allows "Stateful Shaders" (e.g. Game of Life, fluid sim) to run efficiently on the CPU/GPU without race conditions.

- [ ] **Step 1: Memory Model Update:** Update `mf_vm` to handle two sets of buffers for Memory Nodes (Read-Previous / Write-Current).
- [ ] **Step 2: Buffer Swap:** Implement `mf_engine_swap_buffers()` to be called at the end of a frame.
- [ ] **Step 3: Engine Logic:** Update `mf_engine_run` (or `dispatch`) to include automatic strategy selection:
    - **Logic:** If `workload_size > threshold` AND (`Graph is Pure` OR `Double Buffering Active`) -> **Auto-Parallelize**.
    - **Goal:** The Host simply requests "Run on this domain", and the Engine utilizes available cores efficiently without manual flags.


---

## Completed Phases (Archive)

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