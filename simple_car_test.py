#!/usr/bin/env python3
"""Simplest possible test - just visualize the car MJCF in Vuer."""

from pathlib import Path
from vuer_sim_example.mjcf_viewer import MJCFVuerViewer

# Get car MJCF path
mjcf_path = Path(__file__).parent / "vuer_sim_example" / "tasks" / "car.mjcf.xml"

# Create and start viewer
viewer = MJCFVuerViewer(mjcf_path=str(mjcf_path), port=8012)

print("="*50)
print("🚗 Simple Car Viewer")
print("="*50)
print("📍 Open: http://localhost:8012")
print("⏹️  Press Ctrl+C to stop")
print("="*50)

viewer.start()