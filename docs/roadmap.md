# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Phase 7: State Management (The Memory) [COMPLETED]
**Objective:** Enable components to remember state (e.g., "Is Pressed", "Counter").
- [x] **Opcode: Memory (Delay):** A node that outputs its value from the *previous* frame (`MF_NODE_MEMORY`).
- [x] **Opcode: Copy:** `MF_OP_COPY` for updating state at the end of the frame.
- [x] **Compiler Cycle Breaking:** Treated Memory nodes as roots for the current frame.
- [x] **Validation:** `counter.json` test graph passed (Count = Count + 1).

---

## Phase 7.5: Architecture Hardening (Refactoring) [COMPLETED]
**Objective:** Strengthen the codebase before building the Visualizer. Ensure strict interfaces and clean code structure.

- [x] **IO Symbol Table:**
    - [x] Implemented `mf_bin_symbol` in ISA.
    - [x] Compiler populates Symbol Table.
    - [x] VM loads Symbol Table and provides `mf_vm_find_register(vm, name)`.
- [x] **Compiler Modularization:**
    - [x] Split `mf_compiler.c` into `mf_json_parser.c`, `mf_semantics.c`, `mf_codegen.c`.

---

потом можно будет это перевести на GPU шейдеры?

## Phase 7.8: GLSL Math Extension (Shader Ops) [COMPLETED]
**Objective:** Add high-level math operations essential for graphics and SDF, preventing graph explosion.

- [x] **Opcodes:** `Mix` (Decomposed), `Step`, `Dot`, `Length`.
- [x] **Implementation:** Kernels added to `ops_core`.
- [x] **Compiler:** Decomposed `Mix` into `Add(a, Mul(Sub(b, a), t))` to keep ISA minimal.
- [x] **Verification:** `glsl_math_test.json` passed.

---

## Phase 8: The Visualizer (SDL2 Host) [COMPLETED]
**Objective:** Create a runtime environment capable of displaying the MathFlow output buffer as an image.

- [x] **Dependencies:** Add `SDL2` to `vcpkg.json`.
- [x] **New App:** `apps/mf-window`.
- [x] **Input Protocol:**
    - Host injects Tensors via Name (using `mf_vm_find_register`):
        - `u_Time` (Scalar F32)
        - `u_Resolution` (Vec2 F32)
        - `u_Mouse` (Vec4 F32: x, y, click_left, click_right)
- [x] **Output Protocol:**
    - Host expects a single Output Tensor:
        - `out_Color` (Shape: `[Height, Width, 4]`, Type: `U8` or `F32`).
- [x] **Render Loop:**
    - Lock Texture -> `mf_vm_exec` -> Copy Tensor Data to Texture -> Unlock -> Present.

## Phase 9: Pixel Math (SDF UI)
**Objective:** Implement "Rendering" using only math nodes. Prove we can draw a button with Anti-Aliasing without a "DrawRect" function.

- [x] **Missing Kernels (Math):**
    - [x] Implement `MF_OP_CLAMP` (via Compiler Decomposition).
    - [x] Add and Implement `MF_OP_SMOOTHSTEP` (Critical for Anti-Aliasing).
- [x] **Structural Opcodes:**
    - [x] Add `MF_OP_JOIN` (Pack): To combine `X` (Shape: `[W]`) and `Y` (Shape: `[H]`) broadcasts into a `Vec2` (Shape: `[H, W, 2]`).
- [x] **Coordinate Generation:**
    - [x] Implement **node sequence** (manual node chain) that generates UVs.
    - [x] Handle **Aspect Ratio** correction (Host injects `u_Aspect`).
- [x] **SDF Primitives:**
    - [x] `Circle`: `length(uv) - r`.
- [x] **Rendering:**
    - [x] `Mix`: Use `SmoothStep` for soft edges (AA).
- [x] **Demo:** `sdf_button.json`. A circle that behaves like a button (hover state, AA edges).

## Phase 10: Scalability (Sub-Graphs) [COMPLETED]
**Objective:** Enable reuse of graph logic (Prefabs/Macros) to construct complex UIs from primitives.

- [x] **Sub-Graph Node:** Implement `Call("path/to/circle.json")` logic in Compiler.
- [x] **Compiler Inlining:** Recursively load and inline sub-graphs, prefixing IDs to avoid collisions.
- [x] **Interface Definition:** Schema for defining Inputs/Outputs of a sub-graph module (`ExportInput`, `ExportOutput`).

## Phase 11: Parallel Architecture (Multithreading) [COMPLETED]
**Objective:** Move from single-threaded execution to a Job System. This requires refactoring the VM to be "Stateless" (separating Code from Data) to run multiple instances on different threads safely.

- [x] **Step 1: VM Refactoring (Context Segregation):**
    - Split `mf_vm` into:
        - `mf_context` (Read-Only): Program, Symbol Table, Constants.
        - `mf_execution_state` (Mutable): Heap, Registers, Temp Buffers.
    - Update API to `mf_vm_exec(const mf_context* ctx, mf_execution_state* state)`.
- [x] **Step 2: Platform Abstraction Layer:**
    - Create `modules/platform`:
        - Unified wrappers for Threads/Mutexes (Win32 `CreateThread` vs POSIX `pthread`).
        - Atomic primitives.
- [x] **Step 3: Domain Decomposition (Slicing):**
    - Implement "View" tensors (accessing memory ranges without copying).
    - Allow the Host to split large Input Tensors (e.g. `u_FragX` of 480k pixels) into N chunks.
- [x] **Step 4: The Job System:**
    - Create a Worker Thread Pool.
    - Implement `mf_scheduler_dispatch(ctx, inputs, job_count)`.
    - **Sync:** Main thread aggregates results and handles `MF_OP_COPY` (State Logic is single-threaded).

## Phase 12: Architecture Hardening (Prep for UI)
**Objective:** Optimize the system for complex applications (Inventory UI) by eliminating performance bottlenecks and improving API ergonomics.

- [x] **Step 1: Platform Upgrade (CondVars):**
    - [x] Add `mf_cond_t` (Condition Variables) to `modules/platform`.
    - [x] Implement for Win32 (`CONDITION_VARIABLE`) and POSIX (`pthread_cond_t`).
- [x] **Step 2: Scheduler Optimization (Thread Pool):**
    - [x] Replace "Spawn-and-Join" model with a persistent Thread Pool.
    - [x] Worker threads sleep on CondVar waiting for jobs.
- [x] **Step 3: Memory Management (Thread Scratch):**
    - [x] Replace hardcoded stack buffers in workers with proper Thread-Local Arenas.
    - [x] Ensure robust handling of large graphs without stack overflow (Fixed VM Memory reset bug).
- [x] **Step 4: Engine Abstraction (The Host Layer):**
    - [x] Create `modules/host` to encapsulate SDL2, Input, and Rendering.
    - [x] **Goal:** Allow creating a new app in < 20 lines of C code.
    - [x] **Standard Uniforms:** Auto-inject `u_Time`, `u_Mouse`, `u_Resolution`.
    - [x] **Managed Loop:** Hide the main loop, event polling, and scheduler dispatch.
    - [ ] **Hot-Reload (Bonus):** Recompile graph on file change.
- [x] **Refactor:** Rewrite `apps/mf-window/main.c` to use `mf_host`.

## Phase 13: Engine Unification (Refactoring) [COMPLETED]
**Objective:** Eliminate code duplication between CLI runner and GUI host by introducing a unified `mf_engine` layer. This creates a standard way to initialize MathFlow without depending on a windowing system.

- [x] **Step 1: The Engine Module (`modules/engine`):**
    - Create `mf_engine` struct that owns: `mf_context`, `mf_program`, `mf_arena`.
    - Implement `mf_engine_init(mf_engine* engine, const mf_engine_desc* desc)`.
    - Implement `mf_engine_load_graph(mf_engine* engine, const char* path)` (handles both .json and .bin).
    - **Constraint:** Must be Pure C (no SDL/Graphics dependencies).
- [x] **Step 2: Refactor `mf-runner`:**
    - Rewrite CLI tool to use `mf_engine` for loading.
    - Initialize `mf_vm` from `engine->context` for single-threaded execution.
- [x] **Step 3: Refactor `mf_host`:**
    - Update `mf_host` to wrap `mf_engine` internal instance.
    - Initialize `mf_scheduler` from `engine->context` for multi-threaded execution.

## Phase 14: Application Layer (Manifest System)
**Objective:** Decouple "Application Configuration" from "Logic Definition". The Host will strictly run Applications defined by a `.mfapp` manifest, not raw Graphs. This prevents the Host from guessing context and allows different configurations (Window size, Runtime mode) for the same logic.

- [ ] **Step 1: Manifest Definition (`.mfapp`):**
    - Define JSON schema:
        - `runtime`: `{ "type": "shader"|"script", "entry": "path/to/graph.json" }`
        - `window`: `{ "width": 800, "height": 600, "title": "App Name", "resizable": true }`
- [ ] **Step 2: App Loader:**
    - Implement a simple parser in `modules/host` to read `.mfapp` files.
- [ ] **Step 3: Strict Host Refactoring:**
    - Modify `mf_host` to **only** accept `.mfapp` files.
    - Remove hardcoded window defaults.
    - Initialize `mf_engine` with the graph specified in the manifest's `entry`.
    - Select Execution Strategy dynamically:
        - `runtime.type == "shader"` -> Init ThreadPool & Scheduler.
        - `runtime.type == "script"` -> Init simple VM Loop (single thread).
- [ ] **Step 4: Migration:**
    - Create `.mfapp` files for existing assets (`demo_inventory`, `sdf_button`).
    - Update CMake presets and launch configurations.

## Phase 15: UI Widget System
**Objective:** Implement a basic Widget Library (Button, Slider, Text) using the new Sub-Graph system.

---

## Completed Phases (Archive)

### Phase 1-6: Core Foundation (Completed)
- **Architecture:** Compiler/VM separation, C11 implementation.
- **Memory:** Arena + Heap dual-allocator.
- **Math:** Basic arithmetic, matrix ops, broadcasting.
- **Arrays:** Range, CumSum, Filter ops.
- **Tooling:** `mf-runner` CLI.
