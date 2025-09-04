# Car Control with MuJoCo Vuer

Simple car control simulation with gamepad support.

## Setup

```bash
pip install -r requirements_simple.txt
pip install -e .
```

## Run

### Gamepad Control
1. Connect a gamepad/controller to your computer
2. Run:
```bash
python examples/car_control.py
```
3. Open browser at `http://localhost:8012`
4. Use gamepad sticks to control the car

### Programmatic Control
```bash
python examples/car_control_threaded.py
```

## Controls

- **Gamepad**: Left stick = steering, Right stick = forward/backward
- **Threaded**: Automatically drives forward

Open `http://localhost:8012` to view the simulation.