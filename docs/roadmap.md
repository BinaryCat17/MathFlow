# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Milestone

### Milestone 10: The Standard Library & ISA Purity
*Goal: Transform MathFlow into a minimal-core engine where compound logic lives in a versioned JSON library.*

*   **Phase 47: Explicit Import System & Search Paths**
    *   [ ] **Import Field:** Support `"imports": ["path/to/lib", ...]` in the root of JSON graphs.
    *   [ ] **Type Resolution:** If a node `type` is not built-in, search for `<type>.json` in the specified import paths and treat it as a `Call` node.
    *   [ ] **Global Prelude:** Automatically include `assets/lib/` in the import paths for all graphs unless disabled.
*   **Phase 48: The Great Decomposition**
    *   [ ] **Compound Op Migration:** Move `Dot`, `Length`, `Normalize`, `Mix`, and `SmoothStep` into the JSON library.
    *   [ ] **Mean Removal:** Fully remove `Mean` from the compiler C-code and define it as a library alias.
*   **Phase 49: ISA Consolidation & Heavy Kernels**
    *   [ ] **ISA Categorization:** Group remaining instructions into strict categories (Atomic Math, Logic, Reductions, Memory Access, Accelerators).
    *   [ ] **Accelerator Preservation:** Formally define "Heavy Kernels" (`MatMul`, `Inverse`, `Sum`, `Gather`) that remain in C/C++ for performance.
    *   [ ] **Dead Code Removal:** Clean up `mf_backend_cpu.c` and `mf_ops_*.c` from opcodes that were successfully decomposed in Phase 48.
    *   [ ] **Runtime Sanitization:** Ensure the engine doesn't allocate resources for decomposed nodes before inlining.

---

## Completed Phases (Archive)

### Milestone 9: Advanced Compilation (Jan 2026)
- **Instruction Fusion:** Implemented `FMA` detection and fusion pass.
- **Advanced Lowering:** Automated `MEAN` decomposition into `SUM/DIV`.
- **Register Allocation:** Implemented **Liveness Analysis** and register reuse (Buffer Aliasing).
- **Task System:** Full support for multi-domain execution and automated task-splitting.

### Milestone 8: Compiler Consolidation (Dec 2025)
- **Metadata-Driven:** Unified operation definitions using X-Macros.
- **Type Inference:** Automated shape/type propagation rules.

### Milestones 1-7: Foundation & Pixel Engine (2024-2025)
- **Core VM:** High-performance interpreted VM with SoA memory model.
- **Pipeline:** Explicit Manifest-driven execution (`.mfapp`).
- **Pixel Math:** SDF-based rendering engine with anti-aliasing.
- **Compiler:** Modular pass-based architecture.
- **Host:** Cross-platform SDL2 and Headless drivers.