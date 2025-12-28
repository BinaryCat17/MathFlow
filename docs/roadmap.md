# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Phase 16: Architecture Enforcement (CMake Modularization)
**Objective:** Decompose the monolithic `CMakeLists.txt` into strict per-module build files using Modern CMake practices (Namespacing, Alias Targets). This enforces visibility rules and breaks architectural spaghetti.

> **Rule:** All targets must be exported as `MathFlow::module_name`.
> **Rule:** `target_include_directories` must strictly separate `PUBLIC` (interface) and `PRIVATE` (impl) paths.

- [x] **Step 1: Foundation (The Base):**
    - Create `modules/base/CMakeLists.txt` (`MathFlow::base`).
    - Create `modules/isa/CMakeLists.txt` (`MathFlow::isa`).
    - *Constraint:* These must depend ONLY on system libs (libc, pthreads).
- [x] **Step 2: Core Decoupling (The Compiler):**
    - Create `modules/compiler/CMakeLists.txt`.
    - **Fix:** Remove the legacy `target_link_libraries(..., mf_vm)` dependency. The compiler produces ISA binaries and should not depend on the Runtime.
- [x] **Step 3: The Engine Room (VM & Ops):**
    - Create `modules/vm/CMakeLists.txt`.
    - Create `modules/ops/CMakeLists.txt`.
    - *Note:* Currently `ops` depends on `vm` (for `mf_vm_map_tensor`). This circular dependency is technical debt but permissible for this phase.
- [x] **Step 4: Integration (Host & Apps):**
    - Create `modules/host/CMakeLists.txt` (manages SDL2 dependency).
    - Create `apps/mf-runner/CMakeLists.txt`.
    - Refactor root `CMakeLists.txt` to simply `add_subdirectory()` these targets.
- [x] **Step 5: Validation:**
    - Verify that `apps` cannot accidentally include internal headers from `compiler` or `vm`.

## Phase 16.5: Kernel API Refactoring (Strict Layering)
**Objective:** Eliminate the circular dependency `Ops -> VM`. The Math Kernels (`ops`) should be pure functions or depend only on a lightweight context, not the full Virtual Machine state.

- [x] **Step 1: Kernel Context:** Define a generic `mf_kernel_ctx` interface in `isa` (or `base`) for memory allocation and error reporting.
- [x] **Step 2: Refactor Ops:** Rewrite core math kernels to accept `mf_tensor*` directly or `mf_kernel_ctx`, removing all `#include <mathflow/vm/...>`.
- [x] **Step 3: Update Backend:** Adapt the `mf_backend_cpu` dispatch table to bridge the VM state to the new Kernel API.
- [x] **Step 4: Cleanup:** Remove `target_link_libraries(mf_ops ... mf_vm)` from CMake.

## Phase 16.6: Runtime Purity (Detach Compiler)
**Objective:** The `mf_engine` should be a pure Runtime execution environment. Compilation logic moves to the Host layer via a new `mf_asset_loader`.

- [x] **Step 1: Asset Loader:** Implement `mf_asset_loader` in `modules/host`. It handles file detection:
    - If `.json`: Invokes `mf_compiler`.
    - If `.bin`: Loads directly.
- [x] **Step 2: Refactor Engine:** Remove `mf_engine_load_graph_from_json` from `mf_engine`. The Engine now only accepts `mf_program*`.
- [x] **Step 3: Headless Runtime:** Implement `mf_host_run_headless` in `host_core` for CLI tools.
- [x] **Step 4: Build Cleanup:** Remove `target_link_libraries(mf_engine ... mf_compiler)` from CMake.

## Phase 16.7: Host Core Unification
**Objective:** Make `mf-runner` independent of `mf_engine`, matching the high-level abstraction of `mf-window`.

- [x] **Step 1: Upgrade Headless Host:** Refactor `mf_host_run_headless` to accept `mf_host_desc*` instead of `mf_engine*`. It should handle engine initialization internally.
- [x] **Step 2: Clean Runner:** Update `apps/mf-runner/src/main.c` to remove engine initialization code.
- [x] **Step 3: Build Cleanup:** Remove `MathFlow::engine` dependency from `apps/mf-runner`.

## Phase 16.8: Execution Unification (Compute Dispatch)
**Objective:** Replace the rigid `Script` vs `Shader` separation with a unified "Compute Dispatch" model. Shift the responsibility of parallelism from the VM to the **Backend**, allowing future backends (Vulkan/Metal) to handle execution natively.

- [x] **Step 1: ISA Update:** Add a `dispatch` function pointer to the `mf_backend_dispatch_table`.
- [x] **Step 2: Backend CPU Migration:** Move the thread pool and tiling logic (`mf_vm_parallel.c`) into `modules/backend_cpu`. The CPU backend will now manage the creation of worker VMs.
- [x] **Step 3: VM Purification:** Remove parallel execution logic from `modules/vm`, making it a pure single-threaded bytecode interpreter.
- [x] **Step 4: Engine Dispatch:** Implement `mf_engine_dispatch(engine, dim_x, dim_y, ...)` which delegates to the active backend.
- [x] **Step 5: Host Cleanup:** Update hosts to use the unified `dispatch` API, removing `MF_HOST_RUNTIME_*` flags.

## Phase 16.9: Engine Encapsulation & Single State (Completed)
**Objective:** Fully encapsulate the Virtual Machine within the Engine. Remove `mf_instance` to enforce a "Single Source of Truth" architecture. Unify execution under `mf_engine_dispatch`.

- [x] **Step 1: Opaque Architecture:** Redefined `mf_engine` as an opaque handle. Removed `mf_instance` entirely from the public API.
- [x] **Step 2: Engine Proxy API:** Implemented direct data access (`mf_engine_map_tensor`, `mf_engine_find_register`) mapping to the internal VM state.
- [x] **Step 3: Unified Dispatch:** Implemented smart `mf_engine_dispatch`:
    - **1x1 Dispatch:** Runs on the Main VM (Stateful, Script Mode).
    - **NxM Dispatch:** Delegates to Backend (Stateless, Parallel/Shader Mode).
- [x] **Step 4: Host Isolation:** Rewrote `mf_host_sdl.c` and `mf_host_headless.c` to use ONLY `mf_engine.h` and opaque `mf_job_handle`.
- [x] **Step 5: CMake Decoupling:** Removed `MathFlow::vm` from `modules/host/CMakeLists.txt` public interface.

## Phase 16.10: Data Propagation & API Cleanup
**Objective:** Remove low-level execution callbacks (`setup_cb`, `finish_cb`, `mf_job_handle`) from the public API. Implement automatic state propagation so the Host only interacts with the Main Engine State.

- [ ] **Step 1: Backend Context Access:** Update `mf_backend_dispatch_func` to accept the `Main VM` state (or its memory pointers) as a read-only source.
- [ ] **Step 2: Worker Initialization:** Modify CPU Backend workers to automatically bind to the Main VM's constants/uniforms (Zero-Copy or Fast-Copy).
- [ ] **Step 3: Output Consolidation:** Implement a mechanism where parallel workers write directly to the Main VM's output tensor buffers (slicing/tiling handled internally).
- [ ] **Step 4: API Purification:** Remove job-related types and callbacks from `mf_engine.h`. `mf_engine_dispatch` becomes a simple call: `dispatch(engine, x, y)`.

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
