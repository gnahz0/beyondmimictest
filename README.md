# VuerSim

Abstraction layer for MuJoCo physics simulation with Vuer visualization. Provides Environment, Simulation, and Task components for rapid deployment.

## Setup

```bash
pip install -r requirements_simple.txt
pip install -e .
```

## Examples

### Car Control
```bash
python examples/car_control_gamepad.py
# Open http://localhost:8012, use gamepad to drive
```

### Humanoid Policy (G1 Robot)
```bash
# Place ONNX model at: examples/policies/g1_29dof.onnx
python examples/falcon_policy_example.py
# Open http://localhost:8012
```

## Architecture

- `vuer_sim_example/envs/` - Environment wrappers (gym-like interface)
- `vuer_sim_example/sim/` - Physics simulation (async→sync bridge)
- `vuer_sim_example/tasks/` - Task definitions (observations, rewards)
- `vuer_sim_example/policies/` - Policy implementations (ONNX + PD control)

## Features

- Synchronous control interface
- ONNX policy deployment with ParamsProto configuration
- Automatic position-to-torque PD control
- Web-based visualization
- Modular design for different robots and tasks