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

- [x] **Step 1: Arena Realloc:** Implement "dumb realloc" (Alloc + Copy) in `mf_arena` to support tensor resizing on linear memory.
- [x] **Step 2: Worker Memory Optimization:** Switch Backend CPU workers from `mf_heap` (Free List) to `mf_arena` (Linear) for temporary frame data. This removes allocation overhead per frame.
- [x] **Step 3: Engine Reset:** Implement `mf_engine_reset()` to allow clearing the Heap and Arena pointers without destroying OS threads/buffers, enabling efficient graph hot-reloading.

## Phase 17: Implicit Parallelism (Smart Tiling)
**Objective:** Achieve high-performance execution of complex graphs (e.g., N objects interactions on Full HD screen) without introducing explicit Kernels or `OP_MAP`. We utilize **Tiled Execution** (Chunking) to keep intermediate tensors small (L1/L2 Cache friendly) while preserving the linear simplicity of the Graph ISA.

- [x] **Step 1: Virtual Batching:** Update `mf_vm` to support "Active Batch Size". Allow the VM to execute operations on a logical subset (`N` elements) of the allocated tensors, decoupling calculation size from buffer capacity.
- [x] **Step 2: Tiled Dispatcher:** Rewrite `mf_backend_cpu` dispatch logic. Instead of a naive loop or one huge job, implement 2D Tiling (e.g., 64x64 blocks).
    - The Backend calculates tiles.
    - Assigns a Tile to a Worker.
    - Worker sets VM Batch Size = Tile Size (e.g., 4096).
    - Runs the linear graph.
- [x] **Step 3: Intrinsic Coordinates (The Index Op):** Implement a generic `MF_OP_INDEX(axis)` instruction.
    - **Concept:** Enables the Graph to generate its own spatial coordinates (like `gl_FragCoord` or `arange`), decoupling logic from Host inputs.
    - **Architecture:** Update `mf_kernel_ctx` to carry Tiling Context (`global_offset`, `tile_size`).
    - **Backend:** CPU Backend calculates offsets per tile and populates the context.
    - **Result:** The graph becomes resolution-independent and fully parallelizable without external "magic" buffers.

## Phase 17.5: Host Cleanup & Modernization
**Objective:** Clean up the Host code (`mf_host_sdl.c`) to fully utilize the new Phase 17 architecture. Remove legacy "Coordinate Buffer" logic and switch to the new Tiled Dispatch API.

- [ ] **Step 1: Remove Legacy Host Logic:** Delete the code in `mf_host_sdl.c` that manually allocates and fills `u_FragX` / `u_FragY` buffers. The Graph now handles this internally via `Index` ops.
- [ ] **Step 2: Switch to Tiled Dispatch:** Update `mf_host_run` to call `mf_engine_dispatch(engine, width, height)` instead of the old hacky `dispatch(1, num_tiles)`.
- [ ] **Step 3: Direct Texture Output:** Optimize the write-back path. Instead of `float->byte` conversion loop, can we render directly to `u8` tensors? Or make `convert_to_pixels` part of the engine via a `Color` node?
- [ ] **Step 4: Intrinsic Resolution:** Introduce `MF_OP_RESOLUTION` to allow graphs to query canvas size without Host `u_Resolution` uniforms.

## Phase 18: Advanced State Management (Double Buffering)
**Objective:** Enable parallel execution for graphs with state (Memory Nodes) by implementing a double-buffering mechanism. This allows "Stateful Shaders" (e.g. Game of Life, fluid sim) to run efficiently on the CPU/GPU without race conditions.

- [ ] **Step 1: Memory Model Update:** Update `mf_vm` to handle two sets of buffers for Memory Nodes (Read-Previous / Write-Current).
- [ ] **Step 2: Buffer Swap:** Implement `mf_engine_swap_buffers()` to be called at the end of a frame.
- [ ] **Step 3: Unified Dispatch:** Remove the hardcoded `if (1x1)` check in `mf_engine_dispatch`. Move the execution strategy logic into the Backend. The Backend should intelligently handle single-threaded stateful execution vs multi-threaded stateless execution, unifying the architecture.
- [ ] **Step 4: Auto-Parallelize:** Update Engine logic to select strategy:
    - **Logic:** If `workload_size > threshold` AND (`Graph is Pure` OR `Double Buffering Active`) -> **Parallel**.
        - **Goal:** The Host simply requests "Run on this domain", and the Engine utilizes available cores efficiently without manual flags.
    
    ## Phase 19: Smart VM Optimization (Dependency Masking)
    **Objective:** Enable a single graph to handle heterogeneous outputs (Image, Audio, Physics) with different execution domains (2D vs 1D) without introducing `OP_KERNEL`. We use **Dependency Masking** to selectively execute only the necessary parts of the bytecode for a specific output.
    
    - [ ] **Step 1: Dependency Analysis Pass:** Implement a backward-pass analyzer in the Compiler/Loader. It tags every instruction with a bitmask indicating which Outputs it contributes to.
    - [ ] **Step 2: Selective Execution (Masked VM):** Update `mf_vm_exec` to accept an `execution_mask`. The VM loop skips instructions that don't match the current mask.
    - [ ] **Step 3: Multi-Domain Dispatch API:** Refactor Engine API to allow requesting specific outputs on specific domains: `mf_engine_eval(output_name, const mf_domain_desc* domain)`. This supports 1D (Audio), 2D (Image), or N-D computation seamlessly.
    - [ ] **Step 4: Sub-Graph Sharing:** Ensure that common logic (e.g., a shared Noise function used by both Audio and Video) is correctly tagged and reusable, avoiding redundant definitions in the IR.
    
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