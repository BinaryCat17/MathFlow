# Guide: Building Applications with MathFlow

This guide explains how to create interactive, graphical applications using the MathFlow engine.

## 1. Concept

A MathFlow application consists of three parts:
1.  **Manifest (`.mfapp`):** Describes the Window, Global Resources (Memory), and the Pipeline (Kernels).
2.  **Kernels (`.json`):** Graph files that define the logic. Usually split into **Logic** (State Update) and **Render** (Pixel Drawing).
3.  **Assets:** Project files are typically stored in `assets/projects/<your_project>/`.

## 2. Project Structure

Create a folder for your project:
```
assets/projects/my_game/
  ├── my_game.mfapp      # The Entry Point
  ├── logic.json         # Updates game state
  └── render.json        # Draws the game state
```

## 3. The Manifest (.mfapp)

The manifest links everything together. It allocates memory (Resources) and schedules kernels.

**Example `my_game.mfapp`:**
```json
{
    "window": {
        "title": "My Game",
        "width": 800,
        "height": 600,
        "vsync": true
    },
    "pipeline": {
        "resources": [
            { "name": "GameState", "dtype": "F32", "shape": [10] },     // Shared Memory
            { "name": "u_Mouse",   "dtype": "F32", "shape": [4] },      // Mouse Input
            { "name": "Screen",    "dtype": "F32", "shape": [600, 800, 4] } // Output Image
        ],
        "kernels": [
            {
                "id": "update",
                "entry": "logic.json",
                "bindings": [
                    { "port": "StateIn",  "resource": "GameState" },
                    { "port": "StateOut", "resource": "GameState" },
                    { "port": "Mouse",    "resource": "u_Mouse" }
                ]
            },
            {
                "id": "draw",
                "entry": "render.json",
                "bindings": [
                    { "port": "State", "resource": "GameState" },
                    { "port": "Color", "resource": "Screen" }
                ]
            }
        ]
    }
}
```

## 4. The Logic Kernel (logic.json)

This graph reads the **Previous State** (`StateIn`) and computes the **Next State** (`StateOut`).

**Key Concept:** Double Buffering.
*   `StateIn` is the Front Buffer (Read-Only).
*   `StateOut` is the Back Buffer (Write-Only).
*   The engine automatically swaps them at the end of the frame.

**Example `logic.json`:**
```json
{
    "nodes": [
        { "id": "StateIn",  "type": "Input", "data": {"shape": [10], "dtype": "f32"} },
        { "id": "Mouse",    "type": "Input", "data": {"shape": [4], "dtype": "f32"} },
        
        { "id": "One",      "type": "Const", "data": {"value": 1.0} },
        
        { "id": "NewState", "type": "Add" },
        
        { "id": "StateOut", "type": "Output" }
    ],
    "links": [
        // NewState = StateIn + 1.0
        { "src": "StateIn", "dst": "NewState", "dst_port": "a" },
        { "src": "One",     "dst": "NewState", "dst_port": "b" },
        
        // Output result
        { "src": "NewState", "dst": "StateOut", "dst_port": "in" }
    ]
}
```

## 5. The Render Kernel (render.json)

This graph computes the color for **every pixel**. The backend executes this graph $Width \times Height$ times.

**Key Concept:** Domain Iteration.
*   The Output (`Color`) has shape `[600, 800, 4]`.
*   To know "which pixel" you are drawing, use `Index` nodes.

**Example `render.json`:**
```json
{
    "nodes": [
        { "id": "State", "type": "Input", "data": {"shape": [10], "dtype": "f32"} },
        
        // Get Pixel Coordinates
        { "id": "X", "type": "Index", "data": {"axis": 1} },
        { "id": "Y", "type": "Index", "data": {"axis": 0} },

        // Create Red Color
        { "id": "Red", "type": "Const", "data": {"value": [1, 0, 0, 1]} },
        
        { "id": "Color", "type": "Output" }
    ],
    "links": [
        { "src": "Red", "dst": "Color", "dst_port": "in" }
    ]
}
```

**Tip: UV Coordinates**
To get standard `[0, 1]` UV coordinates:
1.  Get `X` (Index 1) and `Y` (Index 0).
2.  Get Resolution inputs (`u_ResX`, `u_ResY`).
3.  `UV.x = Div(X, u_ResX)`
4.  `UV.y = Div(Y, u_ResY)`

## 6. Accessing State (Gather)

The Render Kernel processes pixels linearly. To read a specific value from `GameState` (e.g., "Player Position"), use the `Gather` node.

```json
{ "id": "PlayerPos", "type": "Gather" }
```
*   **Data:** The `State` array.
*   **Indices:** A Constant (e.g., `0`) representing the index to read.

## 7. Reusability (Subgraphs)

You can call other graphs using the `Call` node. This is useful for shared logic (e.g., `sdf_circle.json`).

```json
{ 
    "id": "DrawCircle", 
    "type": "Call", 
    "data": { "path": "lib/sdf_circle.json" } 
}
```
*   **Inputs/Outputs:** The `Call` node dynamically exposes ports matching the `Input` and `Output` nodes inside the referenced graph.

## 8. Running

Build the project and run `mf-window`:

```bash
# Linux
cmake --preset x64-debug-linux
cmake --build out/build/x64-debug-linux
./out/build/x64-debug-linux/apps/mf-window/mf-window assets/projects/my_game/my_game.mfapp

# Windows
cmake --preset x64-debug-win
cmake --build out/build/x64-debug-win
.\out\build\x64-debug-win\apps\mf-window\mf-window.exe assets\projects\my_game\my_game.mfapp
```
