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
**Objective:** The `mf_engine` should be a pure Runtime execution environment, capable of running on constrained devices without the overhead of a JSON parser or Compiler. Compilation logic must move up to the Host layer.

- [ ] **Step 1: Refactor Loading:** Move `mf_engine_load_graph_from_json` logic out of `mf_engine` and into `mf_host` (or a helper utility).
- [ ] **Step 2: Binary Only:** Ensure `mf_engine` API only accepts `mf_program*` (binary data).
- [ ] **Step 3: Headless Runtime:** Implement `mf_host_run_headless` (or similar) in `host_core` to encapsulate the execution loop for CLI apps, matching the symmetry of `mf_host_run`.
- [ ] **Step 4: Build Cleanup:** Remove `target_link_libraries(mf_engine ... mf_compiler)` from CMake.

## Phase 17: UI Widget System
**Objective:** Implement a basic Widget Library (Button, Slider, Text) using the new Sub-Graph system.

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
