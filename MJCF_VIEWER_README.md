# Simple MJCF Viewer with Vuer

A minimal, clean implementation for visualizing and controlling MuJoCo MJCF models in the browser using Vuer.

## 🎯 Design Philosophy

- **Simplicity**: Single class, minimal dependencies, clear API
- **No Complexity**: Removed all robot-specific code, SDK integrations, and unnecessary features
- **Real-time Control**: Bidirectional communication for control input and state output
- **Clean Abstraction**: Works with ANY MJCF file without modification

## 📦 Core Implementation

### `MJCFVuerViewer` Class

The entire viewer is implemented in a single, simple class (`mjcf_viewer.py`):

```python
viewer = MJCFVuerViewer(
    mjcf_path="path/to/model.xml",
    port=8012,
    fps=50.0,
    sim_dt=0.002
)
```

### Key Features

- **Loads MJCF files** directly with MuJoCo
- **Starts Vuer server** for browser-based visualization  
- **Real-time control** via WebSocket RPC (MjStep)
- **State synchronization** between browser and Python
- **Simple API** with `step()`, `reset()`, `get_state()`, `set_control()`

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements_simple.txt
```

### Simplest Example

```python
from vuer_sim_example.mjcf_viewer import MJCFVuerViewer

# Create viewer
viewer = MJCFVuerViewer("car.mjcf.xml", port=8012)

# Start visualization (blocking)
viewer.start()
```

Open browser at `http://localhost:8012` to see the model.

### Control Example

```python
import numpy as np
from threading import Thread

def control_loop(viewer):
    while True:
        # Random control
        ctrl = np.random.uniform(-1, 1, size=2)
        viewer.set_control(ctrl)
        
        # Step simulation
        obs, info = viewer.step()
        time.sleep(0.02)  # 50Hz

# Start control in background
Thread(target=control_loop, args=(viewer,), daemon=True).start()

# Run viewer
viewer.start()
```

## 📁 Example Scripts

1. **`simple_car_test.py`** - Minimal visualization
2. **`car_control_example.py`** - Pattern-based control demo
3. **`test_car_viewer.py`** - Random control with detailed logging

## 🏗️ Architecture

### What We Kept (Essential)
- MuJoCo model loading
- Vuer server with MuJoCo schema
- Real-time control via MjStep RPC
- State synchronization (qpos, qvel)
- Simple step/reset interface

### What We Removed (Unnecessary)
- Robot SDK integrations (Unitree, etc.)
- Anchor actuators and force controls
- Key loggers and complex event handlers
- Domain randomization wrappers
- Camera wrappers and multiview setups
- Complex configuration systems
- Task-specific abstractions

## 🔧 How It Works

1. **Initialize**: Load MJCF with MuJoCo, create Vuer server
2. **Serve Assets**: Static file serving for MJCF and meshes
3. **Connect**: Browser connects via WebSocket
4. **Render**: MuJoCo component renders in browser
5. **Control Loop**: 
   - Python sends control signals
   - Vuer steps simulation
   - Returns state (qpos, qvel)
   - Python syncs state locally

## 📊 Comparison with Original Codebases

| Feature | sim2real | tasks/wrappers | **MJCFVuerViewer** |
|---------|----------|----------------|-------------------|
| Lines of Code | ~400 | ~200 | **~150** |
| Dependencies | Many | Moderate | **Minimal** |
| Robot-specific | Yes | No | **No** |
| Complexity | High | Medium | **Low** |
| Vuer Control | Advanced | Basic | **Simple & Complete** |
| Use Case | Specific robots | Gym envs | **Any MJCF** |

## 🎮 API Reference

### Constructor
```python
MJCFVuerViewer(mjcf_path, port=8012, fps=50.0, sim_dt=0.002)
```

### Methods
- `step(ctrl)` - Step with control, return observation
- `reset()` - Reset to initial state
- `get_state()` - Get current qpos, qvel, ctrl
- `set_control(ctrl)` - Set control for next step
- `start()` - Start server (blocking)
- `stop()` - Stop simulation

## 🚦 Testing

Run any example:
```bash
python simple_car_test.py
# or
python car_control_example.py
```

## ✨ Benefits

- **Simplicity**: Understand entire codebase in minutes
- **Flexibility**: Works with any MJCF without modification  
- **Maintainability**: Single file, clear structure
- **Performance**: Minimal overhead, direct MuJoCo integration
- **Extensibility**: Easy to add features as needed

## 📝 Notes

- Requires MuJoCo 2.3.0+ and Vuer 0.0.30+
- Tested with car.mjcf.xml but works with any valid MJCF
- Browser must support WebGL for rendering
- Control runs at specified FPS (default 50Hz)