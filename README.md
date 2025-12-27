# MathFlow

**MathFlow** is a high-performance, data-oriented computation engine designed for real-time applications, games, and creative tools. It separates the **definition** of logic (Node Graphs) from its **execution** (Virtual Machine), using a Columnar Memory Model (SoA) for maximum cache efficiency and future GPU scalability.

> **Status:** Active Development (Core VM & Compiler Phase).

## Key Features

*   **Data-Oriented Architecture:** Data is stored in Structure-of-Arrays (Columns), optimized for SIMD and GPU transfer.
*   **Modular Design:** Decoupled modules for Compiler, VM, and Backends.
*   **Portable C11:** Written in strict C11, no heavy dependencies (only `cJSON` for the compiler).
*   **Safety & Abstraction:** Accessor API with bounds checking protects memory integrity while keeping the backend agnostic to storage location (RAM/VRAM).
*   **Cross-Platform:** Built with CMake, supports Linux and Windows via presets.

## Getting Started

### Prerequisites
*   CMake (3.25+)
*   C Compiler (GCC/Clang/MSVC)
*   [vcpkg](https://github.com/microsoft/vcpkg) (for dependency management)

### Build Instructions

1.  **Configure vcpkg:**
    Ensure `VCPKG_ROOT` environment variable is set.

2.  **Build (Linux):**
    ```bash
    cmake --preset x64-debug-linux
    cmake --build out/build/x64-debug-linux
    ```

3.  **Build (Windows):**
    Open the folder in Visual Studio or use:
    ```cmd
    cmake --preset x64-debug-win
    cmake --build out/build/x64-debug-win
    ```

## Usage

MathFlow comes with a CLI tool `mf-runner` that can compile JSON graphs and execute them using the CPU backend.

**Run a test graph:**
```bash
./out/build/x64-debug-linux/mf-runner assets/graphs/simple_math.json
```

**Output:**
```
MathFlow Tensor Runner. Loading: assets/graphs/simple_math.json
Program: 5 tensors, 2 insts
...
  [4] Shape: [3] F32: {10.00, 14.00, 18.00}
```

## Documentation

For deep dives into the system design:

*   [Architecture](docs/architecture.md) - System overview, modules, and memory model.
*   [Roadmap](docs/roadmap.md) - Current progress and future plans.

## Project Structure

*   `modules/`
    *   `isa/` - Instruction Set Architecture definitions (The Contract).
    *   `vm/` - Virtual Machine core, Memory management, Accessor API.
    *   `compiler/` - Translates JSON Graphs to Bytecode (`mf_program`).
    *   `backend_cpu/` - Reference C11 implementation of math operations.
*   `apps/`
    *   `mf-runner/` - CLI tool for testing and execution.
*   `assets/` - Test graphs and data.
