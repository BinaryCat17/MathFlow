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

### 2.1. Shape Inference Pass (Completed)
The compiler must predict tensor shapes to validate the graph before execution.
- [x] **Static Analysis:** Propagate shapes from Inputs through the graph.
- [x] **Validation:** Error out on shape mismatches (e.g., trying to dot-product incompatible dimensions).

### 2.2. JSON Format Cleanup (Refactoring) (Completed)
- [x] **Deprecation:** Remove support for legacy types like `InputVec3`, `AddFloat`.
- [x] **Standardization:** Enforce generic names (`Input`, `Add`, `Mul`) in the compiler.
- [x] **Enforce String IDs:** Transition all IDs to strings (e.g., `"id": "1"`) to unify the format.
- [x] **Asset Update:** Convert all existing `.json` tests to the new format.

---

## Phase 3: High-Level Data & Strings (Completed)
**Objective:** Support non-numerical business logic (Strings, IDs, Categories) using mathematical primitives.

### 3.1. Dictionary Encoding (String Interop)
Treat strings as mathematical entities for performance.
- [x] **Global String Pool:** A central storage for unique string characters (`Blob`). (Implemented as FNV1a Hash)
- [x] **Tokenization:** Strings entering the system (Inputs) are hashed and converted to `I32` (Indices).
- [x] **String Tensor:** A tensor of type `I32` that represents a list of strings.
- [x] **Debug View:** The Runner/Debugger resolves `I32` -> `String` only for display purposes. (Implemented I32 view)

### 3.2. Dataframe Operations
Enable database-like logic.
- [x] **Structs as Tensors:** Represent a "Product" `{Price, Name}` not as a C-struct, but as a collection of synchronized tensors (Columnar Store / SoA).
- [x] **Selection/Masking:** Implement `OP_WHERE` (Tensor masking) to filter data (e.g., `Select * From Products Where Price > 100`).

---

## Phase 4: Dynamic Execution & Memory Management
**Objective:** Handle variable workloads (dynamic batch sizes) and manage memory efficiently without system `malloc` in the hot loop.

### 4.1. Memory Allocator System
Implement a robust memory subsystem to replace raw `malloc`/`free` and simple Arenas.
- [x] **Allocator Interface:** Define `mf_allocator` interface (Alloc, Free, Realloc).
- [x] **Linear Allocator:** Enhance the existing Arena for "Frame Temporary" memory (reset every frame).
- [x] **Heap Allocator:** Implement a General Purpose Allocator (e.g., Free List or Buddy System) capable of managing a fixed memory block for persistent but dynamic tensors.
- [ ] **Memory Stats:** Tracking usage, peak memory, and fragmentation.

### 4.2. Dynamic Tensors
Allow tensors to change shape during runtime.
- [x] **Capacity Tracking:** Add `capacity` field to `mf_tensor` to distinguish between allocated size and used size.
- [x] **Resize API:** Implement `mf_tensor_resize(ctx, tensor, new_shape)` which handles re-allocation and data preservation.
- [x] **Growth Strategy:** Implement geometric growth (e.g., 1.5x) to minimize allocation frequency.

### 4.3. Runtime Shape Resolution
Operations must adapt to input data changes.
- [x] **Runtime Inference:** Update Opcodes (Backend) to calculate output shapes based on actual input shapes, not just compile-time constants.
- [x] **Broadcasting:** Update kernels to handle dynamic broadcasting (e.g., `Vec3` + `BatchVec3`).

### 4.4. Validation
- [x] **Resize Test:** A graph that accumulates data, forcing buffer growth. (Verified via mf-runner dynamic test)
- [ ] **OOM Handling:** Graceful error handling when the pre-allocated memory block is exhausted.

---

## Summary of Changes
| Feature | Old MathFlow | **Tensor MathFlow** |
| :--- | :--- | :--- |
| **Data Type** | `f32`, `vec3`, `mat4` | `Tensor<DType, Shape>` |
| **Opcodes** | `ADD_VEC3`, `MUL_MAT4` | `ADD`, `MATMUL` (Generic) |
| **Logic** | Single Item (Scalar) | Batch / Array Processing |
| **Strings** | Not Supported | Dictionary Encoded (I32) |
| **Memory** | Fixed Columns | Dynamic Buffer Pool |