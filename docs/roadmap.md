# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state. We treat the Screen as a Tensor, and the UI as a Pixel Shader running on the CPU (for now).

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Phase 16: Architecture Enforcement (CMake Modularization)
**Objective:** Make the codebase physically match the Architecture Diagram by splitting the monolithic `CMakeLists.txt` into strict per-module files. This enforces visibility rules (Private/Public) and prevents architectural drift (e.g., lower layers accidentally depending on upper layers).

- [ ] **Step 1: Leaf Modules:** Create `CMakeLists.txt` for `isa` (Interface), `base`, and `ops`.
- [ ] **Step 2: Core Modules:** Create build files for `vm` and `backend_cpu`.
- [ ] **Step 3: Dependency Break (Compiler):** Investigate and remove `mf_compiler`'s dependency on `mf_vm`. Move shared logic (e.g. tensor size calc) to `isa` or `base`.
- [ ] **Step 4: Integration:** Create build files for `engine`, `host`, and `apps`.
- [ ] **Step 5: Root Cleanup:** Replace root build logic with simple `add_subdirectory` calls.

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
