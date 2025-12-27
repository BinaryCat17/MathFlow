# MathFlow Roadmap: The Tensor Era

**Goal:** Transform MathFlow from a fixed-register VM into a **Generalized Tensor Compute Engine**.
This shift unifies scalar, vector, matrix, and array operations under a single mathematical abstraction, enabling support for arbitrary data dimensions (ML-style), dynamic tables (Dataframes), and batch processing.

> **Philosophy:** Functionality first, optimization later. We prioritize code simplicity and maintainability.

---

## Phase 1: The Tensor Foundation (Completed)
**Objective:** Define the universal data structure and memory model.

- [x] **Unified Type System:** Implemented `mf_tensor` struct and `mf_dtype`.
- [x] **Universal ISA:** Defined generic opcodes (`OP_ADD`, `OP_MUL`, `OP_MATMUL`).
- [x] **Compiler Update:** Rewrote compiler to parse JSON into `mf_tensor` programs.
- [x] **Generic Backend:** Implemented C-based kernel with broadcasting support for Add/Mul.
- [x] **Tensor Runner:** Updated `mf-runner` to execute and visualize tensor graphs.

---

## Phase 2: Compiler & Toolchain
**Objective:** Update the toolchain to understand tensors and clean up legacy debt.

### 2.1. Shape Inference Pass
The compiler must predict tensor shapes to validate the graph before execution.
- [ ] **Static Analysis:** Propagate shapes from Inputs through the graph.
- [ ] **Validation:** Error out on shape mismatches (e.g., trying to dot-product incompatible dimensions).

### 2.2. JSON Format Cleanup (Refactoring) (Completed)
- [x] **Deprecation:** Remove support for legacy types like `InputVec3`, `AddFloat`.
- [x] **Standardization:** Enforce generic names (`Input`, `Add`, `Mul`) in the compiler.
- [x] **Enforce String IDs:** Transition all IDs to strings (e.g., `"id": "1"`) to unify the format.
- [x] **Asset Update:** Convert all existing `.json` tests to the new format.

---

## Phase 3: High-Level Data & Strings
**Objective:** Support non-numerical business logic (Strings, IDs, Categories) using mathematical primitives.

### 3.1. Dictionary Encoding (String Interop)
Treat strings as mathematical entities for performance.
- [ ] **Global String Pool:** A central storage for unique string characters (`Blob`).
- [ ] **Tokenization:** Strings entering the system (Inputs) are hashed and converted to `I32` (Indices).
- [ ] **String Tensor:** A tensor of type `I32` that represents a list of strings.
- [ ] **Debug View:** The Runner/Debugger resolves `I32` -> `String` only for display purposes.

### 3.2. Dataframe Operations
Enable database-like logic.
- [ ] **Structs as Tensors:** Represent a "Product" `{Price, Name}` not as a C-struct, but as a collection of synchronized tensors (Columnar Store / SoA).
- [ ] **Selection/Masking:** Implement `OP_WHERE` (Tensor masking) to filter data (e.g., `Select * From Products Where Price > 100`).

---

## Phase 4: Dynamic Execution
**Objective:** Handle variable workloads.

### 4.1. Dynamic Batching
- [ ] **Runtime Resizing:** Allow tensors to change size (Dimension `N`) between frames (e.g., adding an item to a shopping cart).
- [ ] **Memory Reallocation:** Smart handling of growing buffers.

---

## Summary of Changes
| Feature | Old MathFlow | **Tensor MathFlow** |
| :--- | :--- | :--- |
| **Data Type** | `f32`, `vec3`, `mat4` | `Tensor<DType, Shape>` |
| **Opcodes** | `ADD_VEC3`, `MUL_MAT4` | `ADD`, `MATMUL` (Generic) |
| **Logic** | Single Item (Scalar) | Batch / Array Processing |
| **Strings** | Not Supported | Dictionary Encoded (I32) |
| **Memory** | Fixed Columns | Dynamic Buffer Pool |