# MathFlow Roadmap v3: The Pixel Engine

**Vision:** To prove that MathFlow is a true Data-Oriented Engine by implementing a UI/Renderer purely as a mathematical function of state.

> **Philosophy:** No "Draw Commands". No "Widgets". Only Math.
> The Host provides the Canvas (Window) and the Input (Mouse/Time). The Graph calculates the Color of every pixel.

---

## Completed Phases (Archive)

### Milestone 10: Standard Library & ISA Consolidation (Jan 2026)
- **Explicit Import System:** Added `"imports"` field and search paths (Prelude).
- **Default Ports:** Automated port mapping for single-input/single-output subgraphs.
- **Decomposition:** Moved `Mean`, `Dot`, `Length`, `Normalize`, `Mix`, and `SmoothStep` to JSON.
- **ISA Cleanup:** Grouped instructions into Atomic, Reduction, Accel, and Memory categories.
- **Core Expansion:** Added `XOR` and `SIZE` primitives.

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