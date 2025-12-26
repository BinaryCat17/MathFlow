# MathFlow Roadmap

## Phase 1: The Modular Core (Completed)
Goal: Establish a decoupled, Data-Oriented architecture.

- [x] **Monorepo Structure:** Separation into `modules/` (isa, vm, compiler, backend).
- [x] **ISA Definition:** Abstract definition of Opcodes and Types.
- [x] **Abstract VM:** Implementation of a dispatch-based Virtual Machine.
- [x] **CPU Backend:** Pure C implementation of math kernels.
- [x] **Compiler:** JSON -> Bytecode translator.
- [x] **CLI Runner:** `mf-runner` tool for testing and headless execution.

## Phase 2: Standard Library & Stability
Goal: Expand the capability of the CPU backend to support UI logic and robust serialization.

### 2.1. Binary Asset Format (Decoupling)
- [x] **Program Structure:** Define `mf_program` struct containing Header, Metadata (counts), Code Section, and Data Section (initial constants).
- [x] **Compiler Update:** Refactor compiler to emit `mf_program` instead of modifying VM directly.
- [x] **VM Loader:** Implement `mf_vm_load_program(vm, program)` to allocate memory and copy initial state.
- [x] **File I/O:** Add functions to write/read `mf_program` to/from `.bin` files.

### 2.2. Type System Expansion (UI Foundation)
- [x] **New Types:** Implement `vec2` (positions), `vec4` (colors), `bool` (logic).
- [x] **Comparison Ops:** `GREATER`, `LESS`, `EQUAL`.
- [x] **Logic Ops:** `AND`, `OR`, `NOT`.
- [x] **Control Ops:** `SELECT` (Ternary `? :`) for conditional data flow.

### 2.3. Math Expansion
- [x] **Layout Math:** `MIN`, `MAX`, `CLAMP`, `FLOOR`, `CEIL`.
- [x] **Matrix Math:** `MAT3` (2D Affine), `Inverse`, `Transpose`.
- [x] **Trigonometry:** `SIN`, `COS`, `ATAN2`.

### 2.4. Memory Abstraction (Safety & Portability)
- [x] **Accessor API:** Replace direct column access in backends with an abstraction layer (e.g., `mf_vm_map_f32(vm, reg_idx)`).
- [x] **View Structs:** Define unified views (e.g., `mf_span` for CPU, `mf_handle` for GPU) to prepare for non-RAM memory types.
- [x] **Bounds Checking:** Optional debug-mode verification of memory access.

## Phase 3: Headless Applications & Testing
Goal: Verify the engine with complex scenarios before adding graphics.

- [ ] **Unit Tests:** Comprehensive coverage for all opcodes.
- [ ] **Layout Engine Demo:** Create a graph that calculates positions for a list of items (Flexbox-like logic) and prints coordinates.
- [ ] **Spring Simulation:** Physics demo using `DeltaTime` input.

## Phase 4: High-Performance & Graphics (Future)
- [ ] **Vulkan Backend:** Declarative rendering of Output Columns.
- [ ] **SIMD Optimization:** AVX/NEON for CPU backend.
- [ ] **GPGPU:** Compute Shader execution strategy.