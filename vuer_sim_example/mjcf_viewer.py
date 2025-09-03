"""Simple MJCF viewer with Vuer integration for real-time visualization."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import mujoco
import numpy as np
from vuer import Vuer
from vuer.events import MjStep
from vuer.schemas import MuJoCo


class MJCFVuerViewer:
    """Minimal MJCF viewer that displays models in Vuer with real-time control."""
    
    def __init__(
        self,
        mjcf_path: str,
        port: int = 8012,
        fps: float = 50.0,
        sim_dt: float = 0.002,
    ):
        """
        Initialize the MJCF viewer.
        
        Args:
            mjcf_path: Path to MJCF XML file
            port: Port for Vuer server
            fps: Frames per second for visualization
            sim_dt: Simulation timestep
        """
        self.mjcf_path = Path(mjcf_path).absolute()
        self.port = port
        self.fps = fps
        self.sim_dt = sim_dt
        
        # Initialize MuJoCo
        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        
        # Initialize state
        self.running = False
        self.is_loaded = False
        
        # Setup Vuer
        self._setup_vuer()
    
    def _setup_vuer(self):
        """Setup Vuer server with static asset serving."""
        # Use parent directory as static root for asset serving
        asset_root = self.mjcf_path.parent
        self.app = Vuer(static_root=str(asset_root), port=self.port)
        
        @self.app.add_handler("ON_MUJOCO_LOAD")
        async def on_load(event, session):
            self.is_loaded = True
            print(f"[Vuer] MuJoCo loaded successfully")
        
        @self.app.spawn(start=True)
        async def main_session(session):
            await self._vuer_session(session)
    
    async def _vuer_session(self, session):
        """Main Vuer session that handles visualization and control."""
        await asyncio.sleep(1)  # Give time for connection
        
        print(f"[Vuer] Setting up scene...")
        
        # Get scene URL relative to static root
        scene_url = f"http://localhost:{self.port}/static/{self.mjcf_path.name}"
        
        # Insert MuJoCo component
        session.upsert @ MuJoCo(
            key="mjcf_model",
            src=scene_url,
            assets=[],  # Simple case: assets in same directory
            frameKeys="qpos qvel ctrl",
            pause=False,
            useLights=True,
            fps=self.fps,
        )
        
        await asyncio.sleep(2)  # Wait for load
        
        if not self.is_loaded:
            print("[Vuer] Warning: MuJoCo may not have loaded properly")
        
        # Start simulation loop
        self.running = True
        await self._simulation_loop(session)
    
    async def _simulation_loop(self, session):
        """Main simulation loop."""
        print(f"[Vuer] Starting simulation loop at {self.fps} FPS...")
        
        while self.running:
            loop_start = time.perf_counter()
            
            try:
                # Get current control (can be set externally)
                ctrl = self.data.ctrl.tolist()
                
                # Step simulation in Vuer and get state back
                frame = await session.rpc(
                    MjStep(key="mjcf_model", sim_steps=1, ctrl=ctrl),
                    ttl=5,
                )
                
                # Sync state from Vuer
                if frame and frame.value:
                    keyframe = frame.value.get("keyFrame", {})
                    if "qpos" in keyframe:
                        self.data.qpos[:] = keyframe["qpos"]
                    if "qvel" in keyframe:
                        self.data.qvel[:] = keyframe["qvel"]
                
            except Exception as e:
                print(f"[Vuer] Step error: {e}")
            
            # Maintain framerate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 1.0/self.fps - elapsed)
            await asyncio.sleep(sleep_time)
    
    def step(self, ctrl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Step simulation with control input.
        
        Args:
            ctrl: Control input array. If None, uses zero control.
        
        Returns:
            observation: Current state (qpos, qvel concatenated)
            info: Additional information
        """
        # Set control
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[:] = 0
        
        # Step local simulation (for state estimation)
        mujoco.mj_step(self.model, self.data)
        
        # Return observation
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        info = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }
        
        return obs, info
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        info = {"time": self.data.time}
        
        return obs, info
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current simulation state."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": self.data.time,
        }
    
    def set_control(self, ctrl: np.ndarray):
        """Set control input for next step."""
        self.data.ctrl[:] = ctrl
    
    def start(self):
        """Start the Vuer server (blocking)."""
        print(f"[Vuer] Starting server on http://localhost:{self.port}")
        try:
            self.app.run()
        except KeyboardInterrupt:
            print("\n[Vuer] Shutting down...")
            self.running = False
    
    def stop(self):
        """Stop the simulation."""
        self.running = False