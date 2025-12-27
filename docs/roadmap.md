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

## Phase 7.8: GLSL Math Extension (Shader Ops)
**Objective:** Add high-level math operations essential for graphics and SDF, preventing graph explosion.

- [ ] **Opcodes:** Add `Mix` (Lerp), `Step`, `SmoothStep`, `Dot`, `Length`.
- [ ] **Implementation:** Add kernels to `ops_core`.
- [ ] **Compiler:** Add support for these nodes.
- [ ] **Why:** Without `Length` and `Mix`, calculating a simple circle requires 5-10 low-level nodes, making graphs unreadable and slow.

---

## Phase 8: The Visualizer (SDL2 Host)
**Objective:** Create a runtime environment capable of displaying the MathFlow output buffer as an image.

- [ ] **Dependencies:** Add `SDL2` to `vcpkg.json`.
- [ ] **New App:** `apps/mf-window`.
- [ ] **Input Protocol:**
    - Host injects Tensors via Name (using `mf_vm_find_register`):
        - `u_Time` (Scalar F32)
        - `u_Resolution` (Vec2 F32)
        - `u_Mouse` (Vec4 F32: x, y, click_left, click_right)
- [ ] **Output Protocol:**
    - Host expects a single Output Tensor:
        - `out_Color` (Shape: `[Height, Width, 4]`, Type: `U8` or `F32`).
- [ ] **Render Loop:**
    - Lock Texture -> `mf_vm_exec` -> Copy Tensor Data to Texture -> Unlock -> Present.

## Phase 9: Pixel Math (SDF UI)
**Objective:** Implement "Rendering" using only math nodes. Prove we can draw a button without a "DrawRect" function.

- [ ] **Coordinate Generation:**
    - Create a sub-graph/macro that generates normalized UV coordinates for the grid using `Range` + Broadcasting.
    - `X = Range(0, W) / W`, `Y = Range(0, H) / H`.
- [ ] **SDF Primitives:**
    - Implement `Circle(uv, center, radius)` -> Distance.
    - Implement `Box(uv, center, size)` -> Distance.
- [ ] **Composition (Mixing):**
    - Implement `Mix(colorA, colorB, mask)` using `Select` or `Lerp` logic.
    - Combine multiple shapes using `Min` (Union) and `Max` (Intersection) operations on distance fields.
- [ ] **Demo:** `sdf_button.json`. A circle that changes color when `length(uv - mouse) < radius`.

## Phase 10: Scalability (Sub-Graphs)
**Objective:** Enable reuse of graph logic (Prefabs/Macros) to construct complex UIs from primitives.

- [ ] **Sub-Graph Node:** Implement `Call("path/to/circle.json")` logic in Compiler.
- [ ] **Compiler Inlining:** Recursively load and inline sub-graphs, prefixing IDs to avoid collisions.
- [ ] **Interface Definition:** Schema for defining Inputs/Outputs of a sub-graph module.

## Phase 11: Performance (Parallel CPU)
**Objective:** Software rendering 800x600 pixels on a single thread might be slow (480k pixels * N nodes). We need parallelism.

- [ ] **Tile-Based Execution:** Split the execution domain (Shape) into tiles.
- [ ] **Task System:** Simple thread pool.
- [ ] **Stateless VM:** Ensure `mf_vm_exec` is thread-safe (read-only graphs, separate memory for working buffers per thread).

---

## Completed Phases (Archive)

### Phase 1-6: Core Foundation (Completed)
- **Architecture:** Compiler/VM separation, C11 implementation.
- **Memory:** Arena + Heap dual-allocator.
- **Math:** Basic arithmetic, matrix ops, broadcasting.
- **Arrays:** Range, CumSum, Filter ops.
- **Tooling:** `mf-runner` CLI.
