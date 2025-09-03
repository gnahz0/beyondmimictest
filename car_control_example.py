#!/usr/bin/env python3
"""Example showing how to control the car step by step."""

import time
import numpy as np
from pathlib import Path
from threading import Thread

from vuer_sim_example.mjcf_viewer import MJCFVuerViewer


def control_loop(viewer: MJCFVuerViewer):
    """Simple control loop that runs in background."""
    time.sleep(3)  # Wait for viewer to initialize
    
    print("\n[Control] Starting control loop...")
    print("[Control] Car actuators: [forward/back, left/right turn]")
    
    # Different control patterns to demo
    patterns = [
        ("Forward", [1.0, 0.0], 100),
        ("Turn right", [0.5, 0.5], 50),  
        ("Backward", [-0.5, 0.0], 100),
        ("Turn left", [0.5, -0.5], 50),
        ("Spin", [0.0, 1.0], 100),
    ]
    
    step = 0
    while True:
        # Cycle through patterns
        for name, ctrl, duration in patterns:
            print(f"\n[Control] {name}: ctrl={ctrl}")
            
            for _ in range(duration):
                # Apply control
                viewer.set_control(np.array(ctrl))
                
                # Step simulation
                obs, info = viewer.step()
                
                # Print position occasionally
                if step % 50 == 0:
                    pos = info['qpos'][:2] if len(info['qpos']) >= 2 else [0, 0]
                    print(f"  Step {step}: position={pos}")
                
                step += 1
                time.sleep(1.0 / 50)  # 50 Hz control
        
        # Random exploration
        print("\n[Control] Random exploration...")
        for _ in range(200):
            ctrl = np.random.uniform(-1, 1, size=2)
            viewer.set_control(ctrl)
            viewer.step()
            step += 1
            time.sleep(1.0 / 50)


def main():
    # Setup path
    mjcf_path = Path(__file__).parent / "vuer_sim_example" / "tasks" / "car.mjcf.xml"
    
    if not mjcf_path.exists():
        print(f"Error: MJCF file not found at {mjcf_path}")
        return
    
    # Create viewer
    viewer = MJCFVuerViewer(
        mjcf_path=str(mjcf_path),
        port=8012,
        fps=50.0,
    )
    
    print("="*60)
    print("🚗 Car Control Example")
    print("="*60)
    print("📍 Open browser at: http://localhost:8012")
    print("🎮 The car will follow predefined patterns then explore")
    print("⏹️  Press Ctrl+C to stop")
    print("="*60)
    
    # Start control in background thread
    control_thread = Thread(target=control_loop, args=(viewer,), daemon=True)
    control_thread.start()
    
    # Run viewer (blocking)
    try:
        viewer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()