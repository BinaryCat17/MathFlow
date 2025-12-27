# MathFlow Roadmap v2: The Tensor UI Engine

**Vision:** To build a modular, high-performance data processing engine where UI is just another mathematical function of the state. The Core remains pure and portable, while capabilities are extended via Modules and Host Applications.

> **Philosophy:** Core is Math. UI is Data. The Graph defines *What* to draw, the Host defines *How* to draw it.

---

## Phase 6: Array Operations Module (The Layout Engine)
**Objective:** Implement mathematical primitives required for layout calculations (Stacking, Grids) without introducing UI concepts into Core.

- [x] **New Module:** `modules/ops_array` (Instruction Set Extension).
    - *Note:* Renamed from `backend_array`. Backends define *execution* (CPU/GPU), Ops modules define *functionality*.
- [x] **Opcode: Range (Iota):** Generates sequences `[0, 1, 2, ... N]`. Essential for iterating over items.
- [x] **Opcode: CumSum (Prefix Scan):** Calculates cumulative sum.
    - Input: `Heights [10, 20, 10]`
    - Output: `Y_Positions [0, 10, 30]`
    - Critical for stacking UI elements dynamically.
- [ ] **Opcode: Filter/Compress:** (Optional) Selects elements based on a mask, changing the tensor size.

## Phase 7: The UI Host (Raylib Integration)
**Objective:** Create a visual runtime environment that drives the graph.

- [ ] **New App:** `apps/mf-ui-host` (linked with Raylib).
- [ ] **Input Protocol:**
    - Host injects: `ScreenSize`, `MousePos`, `MouseDown`, `Time`.
- [ ] **Draw List Protocol:**
    - Host reads specific Output Tensor: `DrawCommands` `[N, 8]`.
    - Format: `{Type, X, Y, W, H, R, G, B}`.
- [ ] **Renderer Implementation:**
    - Map `Type=1` to `DrawRectangle`.
    - Map `Type=2` to `DrawCircle`.
    - Map `Type=3` to `DrawText`.

## Phase 8: Resource Management (Assets)
**Objective:** Allow the graph to reference external assets (Fonts, Icons) purely by ID.

- [ ] **Resource Table:** Host maintains a map `ID -> Texture/Font`.
- [ ] **Text Measurement:** Host provides a "TextSize" tensor or helper, so the graph can calculate layout for text labels.
- [ ] **Texture Atlas:** Support UV coordinates in Draw List.

## Phase 9: Scalability (Graph Composition)
**Objective:** Enable reuse of graph logic (Prefabs/Macros) to avoid spaghetti code.

- [ ] **Sub-Graph Node:** Implement `Call("path/to/button.json")` node type.
- [ ] **Compiler Inlining:** Compiler recursively loads sub-graphs, prefixes their node IDs (to ensure uniqueness), and merges them into the main execution list.
- [ ] **Interface Definition:** JSON schema to define "Public Inputs" and "Public Outputs" of a sub-graph.

## Phase 10: State Management (Feedback Loops)
**Objective:** Enable UI components to remember state (e.g., "Is Pressed", "Scroll Position").

- [ ] **Memory Node:** A node that outputs its value from the *previous* frame.
- [ ] **Cycle Detection:** Update Compiler's Topological Sort to handle cycles by breaking them at `Memory` nodes.
- [ ] **Persistent Memory:** Ensure `Memory` nodes are allocated in the Heap (not Frame Arena) and preserved between `mf_vm_exec` calls.

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
