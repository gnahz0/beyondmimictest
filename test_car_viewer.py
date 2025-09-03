#!/usr/bin/env python3
"""Simple test script to visualize car with random controls."""

import asyncio
import time
import numpy as np
from pathlib import Path

from vuer_sim_example.mjcf_viewer import MJCFVuerViewer


async def random_control_loop(viewer: MJCFVuerViewer, duration: int = 60):
    """Send random control signals to the car."""
    print("\n[Controller] Starting random control loop...")
    print("[Controller] Car has 2 actuators: forward [-1, 1] and turn [-1, 1]")
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < duration:
        # Generate random control
        # ctrl[0] = forward/backward motor
        # ctrl[1] = turning motor
        ctrl = np.random.uniform(-1, 1, size=2)
        
        # Occasionally do specific maneuvers
        if np.random.rand() < 0.1:  # 10% chance
            if np.random.rand() < 0.5:
                print("[Controller] Full forward!")
                ctrl = np.array([1.0, 0.0])
            else:
                print("[Controller] Spinning!")
                ctrl = np.array([0.0, 1.0])
        
        # Set control in viewer
        viewer.set_control(ctrl)
        
        # Step and get observation
        obs, info = viewer.step(ctrl)
        
        step_count += 1
        if step_count % 50 == 0:  # Print status every 50 steps
            car_pos = info['qpos'][:3] if len(info['qpos']) >= 3 else [0, 0, 0]
            print(f"[Controller] Step {step_count}: pos={car_pos[:2]}, ctrl={ctrl}")
        
        # Control loop rate (20 Hz)
        await asyncio.sleep(0.05)
    
    print(f"[Controller] Completed {step_count} steps in {duration} seconds")


def main():
    """Main function to run the car viewer with random controls."""
    # Path to car MJCF
    mjcf_path = Path(__file__).parent / "vuer_sim_example" / "tasks" / "car.mjcf.xml"
    
    if not mjcf_path.exists():
        print(f"Error: MJCF file not found at {mjcf_path}")
        return
    
    print(f"Loading MJCF from: {mjcf_path}")
    
    # Create viewer
    viewer = MJCFVuerViewer(
        mjcf_path=str(mjcf_path),
        port=8012,
        fps=50.0,
        sim_dt=0.002,
    )
    
    print("\n" + "="*60)
    print("MJCF Car Viewer Test")
    print("="*60)
    print(f"Open browser at: http://localhost:8012")
    print("The car will be controlled with random signals")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Start random control in background
    async def run_control():
        await asyncio.sleep(3)  # Wait for viewer to initialize
        await random_control_loop(viewer, duration=300)  # Run for 5 minutes
    
    # Run control loop in background
    control_task = asyncio.create_task(run_control())
    
    try:
        # Start viewer (blocking)
        viewer.start()
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        control_task.cancel()
        viewer.stop()


if __name__ == "__main__":
    # Handle event loop for async control
    import threading
    
    def run_async_control(viewer):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(random_control_loop(viewer, duration=300))
    
    # Path to car MJCF
    mjcf_path = Path(__file__).parent / "vuer_sim_example" / "tasks" / "car.mjcf.xml"
    
    if not mjcf_path.exists():
        print(f"Error: MJCF file not found at {mjcf_path}")
        exit(1)
    
    print(f"Loading MJCF from: {mjcf_path}")
    
    # Create viewer
    viewer = MJCFVuerViewer(
        mjcf_path=str(mjcf_path),
        port=8012,
        fps=50.0,
        sim_dt=0.002,
    )
    
    print("\n" + "="*60)
    print("🚗 MJCF Car Viewer Test")
    print("="*60)
    print(f"📍 Open browser at: http://localhost:8012")
    print("🎮 The car will be controlled with random signals")
    print("⏹️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Start control in separate thread
    control_thread = threading.Thread(
        target=run_async_control,
        args=(viewer,),
        daemon=True
    )
    control_thread.start()
    
    try:
        # Start viewer (blocking)
        viewer.start()
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        viewer.stop()