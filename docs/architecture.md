# MathFlow Architecture

MathFlow is a high-performance, **Data-Oriented** computation engine. It treats everything — from physics to UI layout — as mathematical operations on arrays (tensors).

> **Core Philosophy:** "The Graph is a Shader."
> Whether running on CPU (Interpreter) or GPU (Transpiled), the logic is pure math. The Host Application provides the Canvas and Inputs; the Graph calculates the State and Pixels.

## 1. System Overview

The project is structured as a **Monorepo** with strict decoupling between Definition (ISA), Resource Management (Engine), Translation (Compiler), and Execution (VM/Scheduler).

```mermaid
flowchart TD
    %% Assets
    JSON("Graph file .json")
    BIN("Program Binary .bin")

    %% Module: Engine
    subgraph "Module: Engine (Resource Manager)"
        Engine[Engine Core]
        Arena[Arena Allocator]
        Program[Program Data]
        Context[Execution Context]
    end

    %% Module: Compiler
    subgraph "Module: Compiler"
        Parser[JSON Parser]
        Semantics[Semantics & Inference]
        CodeGen[CodeGen & Optimizer]
    end

    %% Execution Strategies
    subgraph "Execution Strategies"
        VM[VM (Single-Threaded)]
        Scheduler[Scheduler (Multi-Threaded)]
    end

    %% Backends
    subgraph "Module: Backends & Ops"
        Interface[Dispatch Interface]
        CPU[CPU Backend]
        OpsCore[Ops: Core Math]
        OpsArray[Ops: Array Utils]
    end

    %% Flow
    JSON --> Parser --> Semantics --> CodeGen --> BIN
    
    BIN -.-> Engine
    Engine --> Arena
    Engine --> Program
    Engine --> Context
    
    Context --> VM
    Context --> Scheduler
    
    VM <--> Interface
    Scheduler <--> Interface
    
    Interface -.-> CPU
    CPU -.-> OpsCore
    CPU -.-> OpsArray
```

---

## 2. Modules

### 2.0. Engine (`modules/engine`)
*   **Role:** The "Owner". Unified Resource Manager.
*   **Responsibility:**
    *   Manages the **Static Lifecycle**: Loads graphs/binaries, allocates the Arena (Program memory), and sets up the Execution Context.
    *   Acts as the central API for initializing the library, independent of the execution strategy (Single-threaded vs Multi-threaded).

### 2.1. ISA (`modules/isa`)
*   **Role:** The "Contract". Defines the Instruction Set Architecture.
*   **Content:** Header-only definitions of Opcodes (`MF_OP_ADD`, `MF_OP_COPY`), Instruction Formats, and Binary Header structures.
*   **Versioning:** Includes a versioned binary format to ensure backward compatibility.

### 2.2. Virtual Machine (`modules/vm`)
*   **Role:** The "Runner" (Single-Threaded).
*   **Key Responsibilities:**
    *   **Execution State:** Manages the Heap (Variables) and Registers for a single execution context.
    *   **Stateful:** Good for simple scripts or sequential logic where state persists in the VM instance.
    *   **Symbol Table Access:** Uses the Engine's context to map names to registers.

### 2.3. Compiler (`modules/compiler`)
*   **Role:** The Translator.
*   **Structure:**
    *   `mf_json_parser`: Converts JSON to Intermediate Representation (IR). Performs **Recursive Expansion** (Inlining) of Sub-Graphs.
    *   `mf_semantics`: Performs Shape Inference and Type Checking.
    *   `mf_codegen`: Topological Sort (Cycle Breaking for State nodes), Register Allocation, and Bytecode generation.
*   **Expansion Logic:** Sub-graphs are completely flattened during the parsing phase. The VM only sees a single linear list of instructions.

### 2.4. Scheduler (`modules/scheduler`)
*   **Role:** The "Job System" (Multi-Threaded).
*   **Key Responsibilities:**
    *   **Parallel Execution:** Splits the domain (e.g., screen pixels) into tiles and distributes them across a thread pool.
    *   **Stateless:** Uses the Engine's immutable Context, but allocates temporary per-thread scratch memory for execution.

### 2.5. Platform (`modules/platform`)
*   **Role:** OS Abstraction Layer.
*   **Content:** Unified API for Threads, Mutexes, Condition Variables, and Atomics. Supports Windows (Win32) and Linux (pthreads).

### 2.6. Host (`modules/host`)
*   **Role:** Application Framework.
*   **Responsibility:** Provides a high-level wrapper around SDL2 and `mf_engine`. Handles window creation, input polling (Time, Mouse), and the render loop.

### 2.7. Backend: CPU (`modules/backend_cpu`)
*   **Role:** Reference Implementation (Software Renderer).
*   **Responsibility:** Initializes the Dispatch Table with CPU implementations.
*   **Capabilities:** Supports dynamic broadcasting and reshaping.

### 2.8. Operations Libraries (`modules/ops_*`)
These modules contain the actual mathematical kernels.
*   **`modules/ops_core`:** Basic arithmetic, Trigonometry, Logic, Matrix ops, and State Relay.
*   **`modules/ops_array`:** Array manipulation kernels.

---

## 3. Sub-Graphs (Modularity)

MathFlow supports modularity through a "Call-by-Inlining" mechanism. This allows creating complex logic from simple, reusable primitives.

### 3.1. The "Call" Node
A `Call` node references another `.json` file. During compilation, the parser:
1.  Loads the target graph.
2.  **Prefixing:** Adds the caller node's ID as a prefix to all internal nodes (e.g., `button_1::circle::sdf`) to ensure unique names.
3.  **Interface Mapping:**
    *   `ExportInput`: Maps parent `links` (by port index) to internal sub-graph entry points.
    *   `ExportOutput`: Maps internal results back to parent ports.
4.  **Flattening:** Merges the expanded node list into the main graph IR.

### 3.2. Relative Path Resolution
The compiler supports relative paths within sub-graphs. If `A.json` calls `B.json` and `B.json` calls `C.json`, the path to `C` is resolved relative to the location of `B`.

---

## 4. The "Pixel Engine" Concept

MathFlow is evolving into a system capable of rendering UI purely through mathematics (SDFs, Pixel Math), similar to a Fragment Shader.

### 4.1. I/O Protocol (Symbol Table)
The Host Application interacts with the VM using Named Registers:
1.  **Host Initialization:** `time_reg = mf_vm_find_register(vm, "u_Time")`.
2.  **Per-Frame:** Write value to `time_reg`.
3.  **Execution:** `mf_vm_exec(vm)` (or `mf_scheduler_run`).
4.  **Readback:** Read from `mf_vm_find_register(vm, "out_Color")`.

### 4.2. State Management
To support interactive UI (toggles, animations) without external logic:
*   **`MF_NODE_MEMORY`:** Acts as a "delay" line. Outputs the value from the *previous* frame.
*   **Cycle Breaking:** The compiler treats Memory nodes as inputs (Roots) for the current frame to resolve dependency cycles.
*   **`MF_OP_COPY`:** At the end of the frame, the VM executes hidden copy instructions to update Memory nodes.

---

## 5. Memory Model

### 5.1. Dual-Allocator Strategy
1.  **Arena (Static - Engine Owned):** Stores the Program Code, Constants, Symbol Table, and Tensor Metadata. Allocated once at startup.
2.  **Heap (Dynamic - VM/Scheduler Owned):** Stores Tensor Data (Variables). Supports `realloc` for dynamic resizing (e.g., resolution change).

### 5.2. Tensor Ownership
*   **Constants:** Stored in the Program Binary (Arena).
*   **Variables:** Allocated in the VM Heap.
*   **View (Planned):** Support for referencing external memory (e.g., direct write to SDL Surface or GPU Buffer).
