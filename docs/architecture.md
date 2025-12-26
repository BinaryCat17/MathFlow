# MathFlow Architecture

MathFlow is a high-performance, modular data processing engine designed around Data-Oriented Design principles. It separates the definition of computation (Graph/Bytecode) from its execution (Backend).

## 1. System Overview

The project is structured as a **Monorepo** containing several decoupled modules.

```mermaid
flowchart TD
    %% Assets
    JSON("Graph file .json")
    BIN("Program Binary .bin")

    %% Module: Compiler
    subgraph "Module: Compiler"
        Compiler[mf_compiler]
    end

    %% Module: Core VM
    subgraph "Module: Core VM"
        Loader[Program Loader]
        VM[Virtual Machine]
        Memory[("Data Columns / Memory")]
    end

    %% Backends
    subgraph "Module: Backends"
        Interface[Dispatch Interface]
        CPU[CPU Implementation]
        GPU[GPU Implementation]
        
        Interface -.-> CPU
        Interface -.-> GPU
    end

    %% Flow
    JSON --> Compiler
    
    Compiler --> BIN
    
    BIN -.-> Loader
    Loader --> Memory
    Loader --> VM
    
    VM --> Memory
    VM --> Interface
    
    CPU --> Memory
    GPU --> Memory
```

---

## 2. Modules

### 2.1. ISA (`modules/isa`)
*   **Role:** The "Contract". Defines the Instruction Set Architecture.
*   **Content:** Header-only definitions of Opcodes (`MF_OP_ADD`), Instruction Formats, and Basic Types.
*   **Dependencies:** None.

### 2.2. Virtual Machine (`modules/vm`)
*   **Role:** The Orchestrator.
*   **Responsibility:**
    *   **Loader:** Reads `mf_program` asset.
    *   **Memory:** Allocates arenas and columns based on program metadata.
    *   **Init:** Copies initial constants (data section) into memory columns.
    *   **Execution Strategy:** Delegates execution to a Backend.
*   **Dependencies:** `isa`.

### 2.3. Compiler (`modules/compiler`)
*   **Role:** The Translator (Offline Tool).
*   **Responsibility:**
    *   Parses human-readable JSON graphs.
    *   Performs Topological Sorting (Dependency Resolution).
    *   Allocates Registers (Indices).
    *   **Output:** Generates a self-contained `mf_program` (Bytecode + Data Section). Does NOT interact with VM memory directly.
*   **Dependencies:** `isa`, `cJSON`.

### 2.4. Backend: CPU (`modules/backend_cpu`)
*   **Role:** Reference Implementation.
*   **Responsibility:** Provides C11 implementations for all mathematical operations defined in the ISA.
*   **Performance:** Uses pointer-based access to `mf_column` arrays (Zero-Copy).
*   **Mode:** Immediate Execution (Interpreter).

---

## 3. Data Flow & I/O

MathFlow uses a **Declarative I/O Model**. The VM does not perform side effects (drawing, audio, networking) during execution. Instead, it transforms input data into output data.

### 3.1. Input
External systems (Physics Engine, UI, Network) write raw data directly into the **Input Columns** of the VM before execution begins.

### 3.2. Execution
The VM runs the graph (via CPU interpretation or GPU compute). It reads Input Columns and populates Intermediate/Output Columns.

### 3.3. Output
After execution finishes, the external system reads the **Output Columns**.
*   **Visualizer:** Reads `Pos` and `Color` columns to render instances.
*   **Game Logic:** Reads `Health` or `Velocity` columns to update game state.

This "Batch Processing" approach ensures maximum cache locality on CPU and efficient bus transfer for GPU.