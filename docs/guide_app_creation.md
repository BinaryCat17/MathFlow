# Guide: Building Applications with MathFlow

This guide explains how to create interactive, graphical applications using the MathFlow engine.

## 1. Concept

A MathFlow application consists of three parts:
1.  **Manifest (`.mfapp`):** Describes the Window, Assets, and the Computation Pipeline.
2.  **Kernels (`.json`):** Graph files that define the logic (Logic vs Render).
3.  **Assets:** External data like SDF fonts and images.

## 2. Project Structure

Create a folder for your project:
```
assets/projects/my_game/
  ├── my_game.mfapp      # The Entry Point
  ├── logic.json         # Updates game state
  └── render.json        # Draws the game state
  └── font.ttf           # Optional font asset
```

## 3. The Manifest (.mfapp)

The manifest is the orchestrator. It configures the window, allocates global memory, and binds kernels.

**Example `my_game.mfapp`:**
```json
{
    "window": {
        "title": "My Game",
        "width": 800,
        "height": 600,
        "vsync": true
    },
    "runtime": {
        "threads": 0 // 0 = Auto-detect CPU cores
    },
    "assets": [
        { "type": "font", "resource": "u_Font", "path": "font.ttf", "size": 32 }
    ],
    "pipeline": {
        "resources": [
            { "name": "GameState", "dtype": "F32", "shape": [10] }
        ],
        "kernels": [
            {
                "id": "update",
                "entry": "logic.json",
                "bindings": [
                    { "port": "StateIn",  "resource": "GameState" },
                    { "port": "StateOut", "resource": "GameState" }
                ]
            },
            {
                "id": "draw",
                "entry": "render.json",
                "bindings": [
                    { "port": "State", "resource": "GameState" },
                    { "port": "Color", "resource": "out_Color" }
                ]
            }
        ]
    }
}
```

### Automated Resources
The Host automatically provides and updates these resources if they are used in your graphs:
*   **`out_Color`**: The screen buffer (RGBA). Automatically resized on window resize.
*   **`u_Time`**: Current time in seconds (F32).
*   **`u_Resolution`**: [Width, Height] (F32[2]).
*   **`u_ResX` / `u_ResY`**: Individual dimensions (F32).
*   **`u_Aspect`**: Window aspect ratio (F32).
*   **`u_Mouse`**: [X, Y, LeftClick, RightClick] (F32[4]).

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
*   The Output (`Color`) has shape `[height, width, 4]`.
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
2.  `UV.x = Div(X, u_ResX)`
3.  `UV.y = Div(Y, u_ResY)`

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
*   **Inputs/Outputs:** The `Call` node dynamically exposes ports matching the `Input` and `Output` names inside the referenced graph.

## 8. Running

### Run a Manifest
```bash
./out/build/x64-debug-linux/apps/mf-window/mf-window assets/projects/my_game/my_game.mfapp
```

### Quick Look (Raw Graph)
You can also run a single `.json` graph directly. The host will synthesize a default manifest for you:
```bash
./out/build/x64-debug-linux/apps/mf-window/mf-window my_shader.json
```