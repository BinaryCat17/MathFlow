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

## Phase 17.5: Host Cleanup & Modernization (Completed)
**Objective:** Clean up the Host code (`mf_host_sdl.c`) to fully utilize the new Phase 17 architecture. Remove legacy "Coordinate Buffer" logic and switch to the new Tiled Dispatch API.

- [x] **Step 1: Remove Legacy Host Logic:** Delete the code in `mf_host_sdl.c` that manually allocates and fills `u_FragX` / `u_FragY` buffers. The Graph now handles this internally via `Index` ops.
- [x] **Step 2: Switch to Tiled Dispatch:** Update `mf_host_run` to call `mf_engine_dispatch(engine, width, height)` instead of the old hacky `dispatch(1, num_tiles)`.
- [x] **Step 3: Direct Texture Output:** Optimize the write-back path. Instead of `float->byte` conversion loop, can we render directly to `u8` tensors? Or make `convert_to_pixels` part of the engine via a `Color` node?
- [x] **Step 4: Intrinsic Resolution:** Introduce `MF_OP_RESOLUTION` to allow graphs to query canvas size without Host `u_Resolution` uniforms.

## Phase 17.6: System Logging (Advanced)
**Objective:** Replace raw `printf` calls with a structured, thread-safe logging system in `base`. The system must support multiple output sinks (Console, File) with independent verbosity levels and minimal overhead when disabled.

- [x] **Step 1: Logging Core (Base):**
    - **Interface:** Implement `mf_log.h` using macros (`MF_LOG_INFO(...)`) to capture call-site context (`__FILE__`, `__LINE__`).
    - **Zero-Cost Overhead:** Ensure logging macros check the global log level *before* formatting strings or acquiring locks.
    - **Sink API:** Define sinks as `void (*mf_log_sink_fn)(void* user_data, int level, const char* file, int line, const char* msg)`.
    - **Thread Safety:** Use `mf_mutex` to prevent interleaved output from concurrent workers.
    - **Default Sink:** Implement a Console Sink with ANSI color support.
- [x] **Step 2: Integration:**
    - Replace `printf` / `fprintf` in `Compiler` (Parsing errors) and `Loader` (Asset loading events).
    - Add `MF_LOG_TRACE` calls in `Engine` dispatch loop for high-frequency debugging (guarded by zero-cost checks).

## Phase 17.7: VM-Backend Separation (Pure Data VM) (Completed)
**Objective:** Transform `modules/vm` into a pure state container (Data) without execution logic. Move the reference interpreter loop (`mf_vm_exec`) out of `modules/vm` and into `modules/backend_cpu` as a private implementation detail. This ensures the VM module is a passive data structure, allowing different backends (CPU, JIT, GPU) to handle execution logic independently.

- [x] **Step 1: Move Interpreter Logic:** Move the execution loop code from `mf_vm.c` to a new internal helper in `modules/backend_cpu` (e.g., `mf_cpu_interpreter`).
- [x] **Step 2: Purify VM Struct:** Remove `mf_context` and `mf_backend_dispatch_table` dependencies from `mf_vm`. The VM should only manage Registers, Memory, and Allocators.
- [x] **Step 3: Update Backend Interface:** Update `mf_backend_cpu` to accept `mf_program` (Code) and `mf_vm` (State) explicitly during dispatch, rather than relying on the VM to know about the program code.

## Phase 17.8: State-Execution Separation (Architecture Polish) (Completed)
**Objective:** Finalize the decoupling by splitting the overloaded `mf_vm` concept into two distinct entities: `mf_state` (Persistent Data) and `mf_exec_ctx` (Ephemeral Execution Context). The Engine will own `mf_state`, while the Backend will create temporary `mf_exec_ctx` instances on-demand.

- [x] **Step 1: Define `mf_state`:** Create a new structure in `modules/engine` (or `base`) that strictly holds Tensor Data and Heap Memory. This replaces `engine->vm` as the Source of Truth.
- [x] **Step 2: Rename VM to `mf_exec_ctx`:** Rename the `mf_vm` structure and its related functions to `mf_exec_ctx`. This clarifies that it is a light-weight "view" of the state, not a heavy machine.
- [x] **Step 3: Ephemeral Contexts:** Modify `mf_backend_cpu` to construct `mf_exec_ctx` instances on the stack (or thread-local) for both Serial and Parallel execution, linking them to `mf_state` tensors or tiled buffers.
- [x] **Step 4: Cleanup Module Structure:** Refactor `modules/vm` into a library that provides the execution context API, ensuring it doesn't "own" any persistent state.

## Phase 18: Advanced State Management (Ping-Pong) (Completed)
**Objective:** Enable robust state persistence without race conditions. The Compiler transforms high-level `Memory` nodes into explicit Read/Write register pairs linked via a `StateTable`. The Engine manages Double Buffering (Ping-Pong) for these registers, keeping the VM stateless.

- [x] **Step 1: Compiler State Logic:**
    - Restore `MF_NODE_MEMORY` in Parser.
    - Update Compiler to split `Memory` nodes into two registers:
        - **Read Register:** Acts as an Input for the current frame.
        - **Write Register:** Acts as an Output for the next frame.
    - Generate a `StateTable` in the program binary mapping `ReadReg <-> WriteReg`.
- [x] **Step 2: Engine State Manager:**
    - Update `mf_engine` to parse the `StateTable`.
    - Implement Double Buffering: Allocate two buffers per state variable.
    - In `mf_engine_dispatch`, bind the correct buffers (Ping/Pong) to the VM registers before execution.
- [x] **Step 3: Cleanup:**
    - Ensure no "magic" updates happen inside the VM execution loop.
    - Verify thread safety (Parallel workers read from 'Read' and write to 'Write').

## Phase 19: Pipeline Architecture (Explicit Kernels)
**Objective:** Transition from a "Monolithic Script" to a "System of Interacting Programs". We abandon automatic graph partitioning in favor of explicit, modular Kernels (Programs) connected via a Global State. This mirrors GPU Compute Shader architectures.

> **Philosophy:** "1 JSON = 1 Kernel". A graph defines a stateless function $Y = F(X)$. State persistence and communication between Kernels happen strictly via external Buffers managed by the Engine.

- [ ] **Step 1: Compiler & ISA Cleanup:**
    - **Remove `Memory` Nodes:** Delete internal state logic (`MF_NODE_MEMORY`). All state must be passed via Input/Output.
    - **Multi-Output Support:** Ensure Compiler correctly exports metadata for graphs with multiple `Output` nodes (MRT - Multiple Render Targets).
    - **ISA Update:** Update `mf_program` to list explicit Input/Output symbol requirements.

- [ ] **Step 2: The Pipeline Manifest (.mfapp Extension):**
    - Extend the existing `.mfapp` JSON format to describe the System:
        - **Resources:** Global Buffers (e.g., `Physics.Pos`, `Render.Color`).
        - **Kernels:** List of programs to load (`physics.bin`, `render.bin`).
        - **Bindings:** Map Kernel I/O to Global Resources (e.g., `Physics.Out -> State.Pos`, `Render.In -> State.Pos`).
        - **Scheduling:** Execution order and frequency (e.g., Physics x10, Render x1).

- [ ] **Step 3: Engine Orchestrator:**
    - Implement `mf_pipeline` loader.
    - **Resource Manager:** Allocates global buffers (including Ping-Pong pairs for time-dependent data).
    - **Scheduler:** Replaces the single `dispatch` call with a loop over active Kernels, binding the correct buffers (Zero-Copy) before execution.

- [ ] **Step 4: Dual-Use Graphs:**
    - Verify that the same JSON graph can be used as:
        1.  **Inline Subgraph:** Embedded into another graph (Compiled to one binary, max performance).
        2.  **Standalone Kernel:** Running independently in the pipeline (Modular, easy testing).

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