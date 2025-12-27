# MathFlow Roadmap v2: The Tensor UI Engine

**Vision:** To build a modular, high-performance data processing engine where UI is just another mathematical function of the state. The Core remains pure and portable, while capabilities are extended via Modules and Host Applications.

> **Philosophy:** Core is Math. UI is Data. The Graph defines *What* to draw, the Host defines *How* to draw it.

---

## Phase 7: State Management (The Memory)
**Objective:** Enable components to remember state (e.g., "Is Pressed", "Scroll Position", "Velocity"). Without this, the UI is purely reactive and cannot implement logic like Toggles or Drag-and-Drop.

- [ ] **Opcode: Memory (Delay):** A node that outputs its value from the *previous* frame.
- [ ] **Compiler Cycle Breaking:** Update Topological Sort to handle cycles by identifying `Memory` nodes as loop breakers.
- [ ] **Persistent Heap:** Ensure `Memory` nodes allocate in a persistent heap area that survives `mf_vm_exec` resets.
- [ ] **Test:** Create a `counter.json` graph (Count = Count + 1) to verify state retention.

## Phase 8: The UI Host (Raylib Integration)
**Objective:** Create a visual runtime environment capable of rendering rich interfaces (Graphs, Node Editors).

- [ ] **New App:** `apps/mf-ui-host` (linked with Raylib).
- [ ] **Input Protocol:**
    - Host injects: `ScreenSize`, `MousePos`, `MouseDown` (Left/Right), `Time`, `ScrollDelta`.
- [ ] **Extended Draw Protocol:**
    - Host reads Output Tensor: `DrawCommands` `[N, 12]` (Expanded size to fit Bezier params).
    - **Primitives:**
        - `Rect`: `{Type=1, X, Y, W, H, Color, ...}`
        - `Circle`: `{Type=2, CX, CY, Radius, Color, ...}`
        - `Text`: `{Type=3, X, Y, Size, Color, ...}`
    - **Vector / Structural:**
        - `Bezier Cubic`: `{Type=4, X1, Y1, C1X, C1Y, C2X, C2Y, X2, Y2, Thick, Color}`. Essential for drawing node connections.
        - `Scissor (Clip):` `{Type=5, X, Y, W, H}`. Essential for Scroll Areas / Windows.

## Phase 9: Resource Management (Assets)
**Objective:** Allow the graph to reference external assets (Fonts, Icons) purely by ID.

- [ ] **Resource Table:** Host maintains a map `ID -> Texture/Font`.
- [ ] **Text Measurement:** Host provides a "TextSize" tensor or helper, so the graph can calculate layout for text labels.
- [ ] **Texture Atlas:** Support UV coordinates in Draw List.

## Phase 10: Scalability (Graph Composition)
**Objective:** Enable reuse of graph logic (Prefabs/Macros) to avoid spaghetti code.

- [ ] **Sub-Graph Node:** Implement `Call("path/to/button.json")` node type.
- [ ] **Compiler Inlining:** Compiler recursively loads sub-graphs, prefixes their node IDs (to ensure uniqueness), and merges them into the main execution list.
- [ ] **Interface Definition:** JSON schema to define "Public Inputs" and "Public Outputs" of a sub-graph.

## Phase 11: Advanced Strings (Formatting)
**Objective:** Display dynamic data (e.g., "Health: 100") without string manipulation in the Core.

- [ ] **Format Protocol:** Graph outputs a struct `{FormatStringID, Value}`.
- [ ] **Host Formatting:** Host resolves the ID to a format string (e.g., "Score: %d") and applies the value before rendering.

---

## Completed Phases (Archive)

### Phase 1-3: Tensor Foundation & Basic Types (Completed)
- Unified `mf_tensor` struct.
- String hashing and storage.
- Basic math operations.

### Phase 4: Dynamic Execution (Completed)
- **Memory:** Dual-Allocator system (Arena + Heap).
- **Dynamics:** `mf_vm_resize_tensor` and `capacity` tracking.
- **Runtime:** Broadcasting and Shape Inference in CPU backend.
- **Validation:** Inventory Demo (Broadcasting + Masking).

### Phase 5: Modular Architecture (Completed)
- [x] **Opcode Ranges:** Refactor `isa/mf_opcodes.h` to use fixed ranges (e.g., Core: 0-255, Array: 256-511).
- [x] **Dispatch Table Refactor:** Replace the rigid `struct` in `mf_backend_dispatch_table` with a flat array `mf_op_func operations[MF_OP_LIMIT]`.
- [x] **Backend Utils:** Extract generic helper functions (Broadcasting logic, Shape resolution, Iterator macros) from `backend_cpu` into a shared header (e.g., `mf_backend_utils.h`) so new Ops modules can reuse them easily.
- [x] **Ops Registration:** Implement registration functions for modules (e.g., `mf_ops_core_register(table)`).
- [x] **Module Split:** Move math implementations from `backend_cpu` to `modules/ops_core`.

### Phase 6: Array Operations (Completed)
- [x] **New Module:** `modules/ops_array` created.
- [x] **Range (Iota):** Implemented.
- [x] **CumSum:** Implemented.
- [x] **Filter/Compress:** Implemented dynamic filtering.