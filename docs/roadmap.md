# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Active Milestone

### Milestone 10: The Standard Library & ISA Purity
*Goal: Transform MathFlow into a minimal-core engine where compound logic lives in a versioned JSON library.*

*   **Phase 47: Global Registry & Search Paths**
    *   [ ] **Implicit Call System:** Allow the compiler to resolve unknown node types (e.g., `Normalize`) by searching in `assets/lib/` without explicit `path`.
    *   [ ] **Global Prelude:** Automatically "import" a set of standard operations during the Inline pass.
*   **Phase 48: The Great Decomposition**
    *   [ ] **Compound Op Migration:** Move `Dot`, `Length`, `Normalize`, `Mix`, and `SmoothStep` into the JSON library.
    *   [ ] **Mean Removal:** Fully remove `Mean` from the compiler C-code and define it as a library alias.
*   **Phase 49: ISA Purity & Backend Shrinking**
    *   [ ] **Kernel Cleanup:** Remove `mf_ops_matrix.c` and other complex kernels from the runtime.
    *   [ ] **Minimal ISA:** Refine the ISA to only include hardware-aligned primitives (Arithmetic, Logic, FMA, Sum, Gather/Scatter).

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