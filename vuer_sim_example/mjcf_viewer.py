import asyncio
import os
import time
from glob import glob
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import mujoco
import numpy as np
from vuer import Vuer
from vuer.events import MjStep
from vuer.schemas import MuJoCo, Gamepad


class MJCFVuer:
    def __init__(
            self,
            mjcf_path: str,
            port: int = 8012,
            fps: float = 50.0,
            sim_dt: float = 0.002,
    ):
        self.mjcf_path = Path(mjcf_path).absolute()
        self.port = port
        self.fps = fps
        self.sim_dt = sim_dt

        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt

        # Initialize control to zero
        self.data.ctrl[:] = 0

        self.running = False
        self.is_loaded = False

        self._setup_vuer()

    def _setup_vuer(self):
        asset_root = self.mjcf_path.parent
        self.app = Vuer(static_root=str(asset_root), port=self.port)

        @self.app.add_handler("GAMEPAD")
        async def on_load(event, session):
            self.is_loaded = True
            print("gamepads:", event.value['axes'])
            self.data.ctrl[0] = -event.value['axes'][3]
            self.data.ctrl[1] = -event.value['axes'][2]

        @self.app.add_handler("ON_MUJOCO_LOAD")
        async def on_load(event, session):
            self.is_loaded = True
            print("MuJoCo loaded")

        @self.app.spawn(start=False)
        async def main_session(session):
            await self._vuer_session(session)

    async def _vuer_session(self, session):
        await asyncio.sleep(1)

        print("Setting up scene...")

        asset_root = str(self.mjcf_path.parent)
        original_dir = os.getcwd()
        try:
            os.chdir(asset_root)
            all_files = glob("**/*.*", recursive=True)
        finally:
            os.chdir(original_dir)

        scene_url = f"http://localhost:{self.port}/static/{self.mjcf_path.name}"
        asset_urls = [f"http://localhost:{self.port}/static/{f}" for f in all_files]

        session.upsert(Gamepad(key="gamepads"), to="bgChildren", )
        session.upsert @ MuJoCo(
            key="mjcf_model",
            src=scene_url,
            assets=asset_urls,
            frameKeys="qpos qvel ctrl",
            pause=False,
            useLights=True,
            fps=self.fps,
        )

        await asyncio.sleep(2)

        self.running = True
        await self._simulation_loop(session)

    async def _simulation_loop(self, session):
        print(f"Starting simulation at {self.fps} FPS...")

        while self.running:
            loop_start = time.perf_counter()

            try:
                ctrl = self.data.ctrl.tolist()

                frame = await session.rpc(
                    MjStep(key="mjcf_model", sim_steps=1, ctrl=ctrl),
                    ttl=5,
                )

                if frame and frame.value:
                    key_frame = frame.value["keyFrame"]
                    if "qpos" in key_frame:
                        self.data.qpos[:] = key_frame.get("qpos")
                    if "qvel" in key_frame:
                        self.data.qvel[:] = key_frame.get("qvel")

            except Exception as e:
                print(f"Step error: {e}")

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 1.0 / self.fps - elapsed)
            await asyncio.sleep(sleep_time)

    def step(self, ctrl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[:] = 0

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        info = {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

        return obs, info

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        info = {"time": self.data.time}

        return obs, info

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": self.data.time,
        }

    def set_control(self, ctrl: np.ndarray):
        self.data.ctrl[:] = ctrl

    def start(self):
        print(f"Starting server on http://localhost:{self.port}")
        try:
            self.app.run()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False

    def stop(self):
        self.running = False
