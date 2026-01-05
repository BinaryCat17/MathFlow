# MathFlow Architecture

MathFlow is a **Data-Oriented Computation Engine**. It is designed to process data streams (Tensors) efficiently by strictly separating the **definition** of logic from its **execution**.

## System Overview

The system is built as a layered hierarchy of modules. Lower layers (Base, ISA) have no dependencies on upper layers.

```mermaid
graph TD
    subgraph App ["Application Layer"]
        Runner["mf-runner (CLI)"]
        Window["mf-window (GUI)"]
        Host["Host (Window/Input/Assets)"]
    end

    subgraph Core ["Core Orchestration"]
        Engine["Engine (State/Resources)"]
        Compiler["Compiler (JSON -> Bin)"]
    end

    subgraph Exec ["Execution Layer"]
        Backend["Backend (Thread Pool)"]
        Ops["Ops (Math Kernels)"]
    end

    subgraph Foundation ["Foundation (Shared)"]
        ISA["ISA (Bytecode/Tensors)"]
        Base["Base (Memory/Math/IO)"]
    end

    %% Active Workflow Flow
    Runner --> Host
    Window --> Host
    
    Host --> Compiler
    Host --> Engine
    
    Engine -.-> Backend
    Backend --> Ops

    %% Style
    style Foundation fill:#eee,stroke:#999,stroke-dasharray: 5 5
```

---

## Module Responsibilities

### 1. Foundation Layers

#### **Base** (`modules/base`)
*   **Role:** The bedrock. Zero external dependencies.
*   **Contents:**
    *   `mf_types.h`: Core typedefs (`f32`, `u8`, `mf_type_info`) and access modes.
    *   `mf_memory`: Dual-allocator system (Stack Arena + Heap).
    *   `mf_buffer`: Raw memory container (owns `void* data`).
    *   `mf_shape`: Shape inference and **Linear Stride Calculation**.
    *   `mf_math`: Basic scalar math functions.
    *   `mf_log`: Thread-safe logging subsystem.
    *   `mf_platform`: OS-independent abstractions (Atomics, Threads, Mutexes).
    *   `mf_utils`: Common utilities (hashing, path manipulation, file IO).

#### **ISA** (`modules/isa`)
*   **Role:** The "Contract" or Interface. Defines the data structures used to communicate between modules. Pure data, no logic.
*   **Contents:**
    *   `mf_program`: The compiled bytecode format.
    *   `mf_instruction`: **STEP_N Layout**. Contains explicit **Linear Strides** for every operand.
    *   `mf_tensor`: **The View**. A lightweight struct (`info`, `buffer*`, `offset`) pointing to data.
    *   `mf_state`: Holds registers (tensors) and the **Global Error Pointer**.

#### **Ops** (`modules/ops`)
*   **Role:** The "Standard Library" of math functions. Stateless kernels implementing instructions.

### 2. Compilation & Orchestration

#### **Compiler** (`modules/compiler`)
*   **Role:** Translates human-readable Graphs (JSON) into machine-efficient Bytecode (`mf_program`).
*   **Architecture:** Pipeline of passes:
    *   **Lowering:** JSON AST -> Flat IR.
    *   **Inlining:** Recursive expansion of sub-graphs.
    *   **Optimization (Fusion):** Combines operations (e.g., `Mul + Add -> FMA`).
    *   **Analysis:** Shape and Type inference/propagation.
    *   **Register Allocation:** Liveness analysis to minimize memory by reusing registers (**Buffer Aliasing**).
    *   **Domain Splitting:** Groups instructions into tasks based on output shapes.
    *   **CodeGen:** Emits binary bytecode and constant data.

#### **Engine** (`modules/engine`)
*   **Role:** The "Brain" / Orchestrator.
*   **Responsibilities:**
    *   **Resource Management:** Allocates and manages Global Buffers.
    *   **Pipeline Management:** Coordinates multiple Kernels and execution order.
    *   **Double Buffering:** Manages Ping-Pong (Front/Back) state.

### 3. Application Layer

#### **Host** (`modules/host`)
*   **Role:** The Interface between Engine and the Outside World (OS, Files, Window).
*   **Responsibilities:**
    *   **Application Lifecycle:** Manages `mf_engine` creation, initialization, and shutdown.
    *   **Manifest Loading:** Parses `.mfapp` files and configures the system. Supports "Raw Graph" loading by synthesizing implicit pipelines.
    *   **Asset Loading:** Loads external data (Images, Fonts) into Engine resources.
    *   **Platform Support:**
        *   `mf_host_headless`: For CLI execution and testing.
        *   `mf_host_sdl`: For interactive GUI applications.
    *   **System Resources:** Automated updates for `u_Time`, `u_Resolution`, and `u_Mouse`.

#### **Backend** (`modules/backend_cpu`)
*   **Role:** The execution engine. Distributes work across CPU threads using a windowed approach.

---

## Memory Safety & Error Handling

MathFlow priorities visibility and fault isolation through two defensive layers:

1.  **Atomic Kill Switch:** The `mf_engine` maintains an atomic error code. If any thread fails, it sets the global flag, stopping all other threads and kernels immediately.
2.  **Kernel Crash Reports:** Detailed reports on failure including **Opcode Names**, register IDs, domain coordinates, and memory ranges.

---

## The Pipeline Model
MathFlow orchestrates execution via a **Pipeline**.

1.  **Kernel:** A compiled Graph (Program). Stateless function $Y = F(X)$.
2.  **Resource:** A named Global Buffer managed by the Engine.
3.  **Binding:** Link between a Kernel Port and a Global Resource.
4.  **Scheduler:** The Host/Engine executes Kernels sequentially, swapping Front/Back buffers at the end of the frame.

## Data Flow

1.  **Load:** Host loads configuration (Manifest or raw JSON).
2.  **Initialize:** Host creates Engine, compiles programs, and allocates resources.
3.  **Loop:**
    *   Host updates system inputs (Time, Mouse).
    *   Engine determines active buffers.
    *   Backend executes kernels in parallel.
    *   Host presents output (e.g., rendering `out_Color` via SDL).

---

## Memory Model

MathFlow distinguishes between **Storage** and **View**.

1.  **mf_buffer (Storage):**
    *   A raw allocation of bytes.
    *   Owned by `mf_engine` (for global resources) or `mf_state` (for temp data).
    *   Heavyweight (allocation/free).

2.  **mf_tensor (View):**
    *   Metadata (`shape`, `dtype`, `strides`) + Pointer to Buffer + Offset.
    *   Lightweight (created on stack/arena).
    *   **Zero-Copy Ops:** `Slice`, `Reshape`, and `Transpose` creates a new *View* without touching the *Buffer*.

3.  **Register Allocation (Buffer Aliasing):**
    *   The compiler performs **Liveness Analysis** to detect when a tensor is no longer needed.
    *   Registers are reused for non-overlapping lifetimes.
    *   **In-place Operations:** Element-wise ops (Add, Sub, etc.) can reuse their input register for the output if the input "dies" at that instruction.
    *   **Persistent Registers:** Inputs, Constants, and Outputs are protected from reuse to maintain interface integrity.

4.  **Execution:**
    *   The Engine allocates **A** and **B** buffers for every global resource.
    *   On Frame N: Inputs read from **A**, Outputs write to **B**.
    *   On Frame N+1: Swap A/B.
    *   The Backend creates temporary Views into these buffers for each worker thread.

---

## Pipeline Manifest (.mfapp)

The `.mfapp` file is the entry point for applications. It defines the window settings and the computation pipeline.

**Schema (Strict Arrays):**

```json
{
    "window": {
        "title": "My App",
        "width": 800,
        "height": 600
    },
    "pipeline": {
        "resources": [
            { "name": "State", "dtype": "F32", "shape": [1024] },
            { "name": "Screen", "dtype": "F32", "shape": [800, 600, 4] }
        ],
        "kernels": [
            {
                "id": "logic",
                "entry": "logic.json",
                "bindings": [
                    { "port": "State", "resource": "State" }
                ]
            },
            {
                "id": "render",
                "entry": "render.json",
                "bindings": [
                    { "port": "Data", "resource": "State" },
                    { "port": "Out",  "resource": "Screen" }
                ]
            }
        ]
    }
}
```

## The STEP_N Execution Model

MathFlow uses a generalized **Stride Model** for data processing.

### Linear Strides
The compiler calculates how pointers advance per domain element:
*   **Stride 0:** Scalar/Broadcast. Pointer stays static.
*   **Stride 1:** Flat linear array.
*   **Stride N:** Multi-channel data (e.g. RGBA).

Kernels perform unconditional pointer arithmetic (`ptr += stride`), eliminating branching in the hot loop.

### Intrinsic Coordinates (Index)
To know "where" the current thread is running (e.g. pixel coordinate), the graph must use `Input` nodes with specialized **Providers**.
*   **Mechanism:** These nodes (e.g., `provider: "host.index.0"`) read the current multi-dimensional index from the execution context (`tile_offset`) and output it as a spatial stream.
*   **Builtin Mapping:** The compiler recognizes `host.index.N` and maps it to `MF_BUILTIN_INDEX` with a specific axis.

### Random Access (Gather)
Standard operations are linear. For non-linear logic, MathFlow uses `MF_OP_GATHER`.
*   **Safety:** Explicit bounds checking against the source size. Invalid access triggers the **Kill Switch**.