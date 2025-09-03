#!/usr/bin/env python3
"""Generic MJCF viewer - pass any MJCF file as argument."""

import sys
import numpy as np
from pathlib import Path
from threading import Thread
import time

from vuer_sim_example.mjcf_viewer import MJCFVuerViewer


def auto_control(viewer):
    """Automatic control based on number of actuators."""
    time.sleep(3)
    
    n_ctrl = viewer.model.nu
    print(f"\n[Auto Control] Model has {n_ctrl} actuators")
    
    if n_ctrl == 0:
        print("[Auto Control] No actuators - passive viewing only")
        return
    
    step = 0
    while True:
        # Generate smooth sinusoidal control
        t = step * 0.02  # Time in seconds
        ctrl = np.sin(t + np.arange(n_ctrl) * np.pi / n_ctrl)
        
        viewer.set_control(ctrl)
        obs, info = viewer.step()
        
        if step % 100 == 0:
            print(f"[Auto Control] Step {step}, ctrl range: [{ctrl.min():.2f}, {ctrl.max():.2f}]")
        
        step += 1
        time.sleep(0.02)


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_mjcf.py <path_to_mjcf_file>")
        print("\nExample:")
        print("  python view_mjcf.py vuer_sim_example/tasks/car.mjcf.xml")
        sys.exit(1)
    
    mjcf_path = Path(sys.argv[1])
    
    if not mjcf_path.exists():
        print(f"Error: File not found: {mjcf_path}")
        sys.exit(1)
    
    # Optional port argument
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8012
    
    print(f"\n{'='*60}")
    print(f"🔧 Generic MJCF Viewer")
    print(f"{'='*60}")
    print(f"📁 File: {mjcf_path.name}")
    print(f"📂 Path: {mjcf_path.parent}")
    print(f"📍 URL: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    # Create viewer
    viewer = MJCFVuerViewer(
        mjcf_path=str(mjcf_path),
        port=port,
        fps=50.0,
    )
    
    # Start auto control if model has actuators
    if viewer.model.nu > 0:
        control_thread = Thread(target=auto_control, args=(viewer,), daemon=True)
        control_thread.start()
        print(f"✅ Auto-control enabled for {viewer.model.nu} actuators")
    else:
        print("ℹ️  No actuators detected - passive viewing mode")
    
    print("⏹️  Press Ctrl+C to stop\n")
    
    try:
        viewer.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        viewer.stop()


if __name__ == "__main__":
    main()