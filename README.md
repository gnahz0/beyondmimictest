# Simple MJCF Vuer Viewer

A minimal, clean implementation for visualizing and controlling MuJoCo MJCF models in the browser using Vuer.

## 🎯 Features

- **Simple**: Single class implementation (~150 lines)
- **Universal**: Works with any MJCF file
- **Real-time**: Bidirectional control and state sync
- **Clean**: No unnecessary dependencies or complexity

## 📁 Project Structure

```
vuer-sim-example/
├── vuer_sim_example/
│   ├── __init__.py
│   └── mjcf_viewer.py       # Core viewer class
├── examples/
│   ├── mjcf_models/
│   │   └── car.mjcf.xml     # Example MJCF model
│   ├── simple_car_test.py   # Minimal example
│   ├── car_control_example.py # Control patterns demo
│   ├── test_car_viewer.py   # Random control test
│   └── view_mjcf.py         # Generic MJCF viewer
├── requirements_simple.txt   # Dependencies
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements_simple.txt
```

### Run Examples

```bash
# Simple visualization
python examples/simple_car_test.py

# Control patterns demo
python examples/car_control_example.py

# View any MJCF file
python examples/view_mjcf.py path/to/model.mjcf.xml
```

Open browser at `http://localhost:8012` to see the visualization.

## 💡 Usage

```python
from vuer_sim_example.mjcf_viewer import MJCFVuerViewer

# Create viewer
viewer = MJCFVuerViewer("path/to/model.mjcf.xml", port=8012)

# Optional: Control in background
import numpy as np
viewer.set_control(np.array([1.0, 0.0]))  # Set control
obs, info = viewer.step()                  # Step simulation

# Start visualization
viewer.start()  # Blocking - opens browser interface
```

## 📖 API

### MJCFVuerViewer

```python
MJCFVuerViewer(mjcf_path, port=8012, fps=50.0, sim_dt=0.002)
```

**Methods:**
- `step(ctrl)` - Step simulation with control input
- `reset()` - Reset to initial state  
- `get_state()` - Get current qpos, qvel, ctrl
- `set_control(ctrl)` - Set control for next step
- `start()` - Start Vuer server (blocking)
- `stop()` - Stop simulation

## 🔧 How It Works

1. Loads MJCF model with MuJoCo
2. Starts Vuer server with static file serving
3. Browser connects via WebSocket
4. Real-time control via MjStep RPC
5. Bidirectional state synchronization

## 📝 License

MIT