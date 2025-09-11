import asyncio
import os
import queue
from glob import glob
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from vuer import Vuer
from vuer.events import MjStep
from vuer.schemas import MuJoCo, Gamepad


class VuerSim:
    def __init__(
            self,
            mjcf_path: str,
            port: int = 8012,
            fps: float = 50.0,
    ):
        self.mjcf_path = Path(mjcf_path).absolute()
        self.port = port
        self.fps = fps

        self.running = False
        self.is_loaded = False
        self.session = None
        self.current_keyframe = None
        self.initial_keyframe = None
        
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self._setup_vuer()

    def _setup_vuer(self):
        asset_root = self.mjcf_path.parent
        self.app = Vuer(static_root=str(asset_root), port=self.port)

        @self.app.add_handler("GAMEPAD")
        async def on_gamepad(event, session):
            print("gamepads:", event.value['axes'])

        @self.app.add_handler("ON_MUJOCO_LOAD")
        async def on_load(event, session):
            self.is_loaded = True
            if event.value and "keyFrame" in event.value:
                self.initial_keyframe = event.value["keyFrame"]
                self.current_keyframe = self.initial_keyframe.copy()
                print(f"MuJoCo loaded - Initial state received with {len(self.initial_keyframe.get('qpos', []))} DOFs")

        @self.app.spawn(start=False)
        async def main_session(session):
            await self._vuer_session(session)

    async def _vuer_session(self, session):
        self.session = session
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

        session.upsert(Gamepad(key="gamepads"))
        session.upsert @ MuJoCo(
            key="mjcf_model",
            src=scene_url,
            assets=asset_urls,
            frameKeys="qpos qvel ctrl",
            pause=True,
            useLights=True,
            fps=self.fps,
        )

        await asyncio.sleep(2)
        self.running = True
        print("VuerSim session ready")

        # process commands from external thread
        while self.running:
            try:
                command = self.command_queue.get_nowait()
                await self._process_command(command, session)
            except queue.Empty:
                pass

            # for fps
            await asyncio.sleep(1.0 / self.fps)

    async def _process_command(self, command: Dict[str, Any], session) -> None:
        action = command.get('action')
        
        if self.current_keyframe and "ctrl" in self.current_keyframe:
            ctrl = self.current_keyframe["ctrl"].copy()
        else:
            ctrl = [0.0] * 100
        
        if action is not None:
            ctrl[:len(action)] = action.tolist()

        try:
            frame = await session.rpc(
                MjStep(key="mjcf_model", sim_steps=1, ctrl=ctrl),
                ttl=5,
            )
            
            if frame and frame.value:
                self.current_keyframe = frame.value["keyFrame"]
                
            self.result_queue.put({
                'success': True,
                'keyframe': self.current_keyframe
            })
            
        except Exception as e:
            error_msg = f"RPC error: {type(e).__name__}: {str(e)}"
            print(error_msg, flush=True)
            self.result_queue.put({
                'success': False,
                'error': error_msg
            })

    async def step(self, action: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not self.running:
            raise RuntimeError("VuerSim not running. Call start() first.")

        self.command_queue.put({'action': action})
        
        while True:
            try:
                result = self.result_queue.get_nowait()
                if result['success']:
                    self.current_keyframe = result['keyframe']
                    return self.get_state()
                else:
                    raise RuntimeError(f"Step failed: {result.get('error', 'Unknown error')}")
            except queue.Empty:
                await asyncio.sleep(0.001)

    async def reset(self) -> Dict[str, Any]:
        if not self.session or not self.running:
            raise RuntimeError("VuerSim session not ready. Call start() first.")

        asset_root = str(self.mjcf_path.parent)
        scene_url = f"http://localhost:{self.port}/static/{self.mjcf_path.name}"

        original_dir = os.getcwd()
        try:
            os.chdir(asset_root)
            all_files = glob("**/*.*", recursive=True)
        finally:
            os.chdir(original_dir)

        asset_urls = [f"http://localhost:{self.port}/static/{f}" for f in all_files]

        self.session.upsert @ MuJoCo(
            key="mjcf_model",
            src=scene_url,
            assets=asset_urls,
            frameKeys="qpos qvel ctrl time",
            pause=True,
            useLights=True,
            fps=self.fps,
        )

        await asyncio.sleep(1)

        self.current_keyframe = self.initial_keyframe.copy() if self.initial_keyframe else None

        return self.get_state()

    def observe(self) -> np.ndarray:
        if self.current_keyframe is None:
            return np.array([])
        
        qpos = self.current_keyframe.get("qpos", [])
        qvel = self.current_keyframe.get("qvel", [])
        return np.concatenate([qpos, qvel])

    def get_state(self) -> Dict[str, Any]:
        if self.current_keyframe is None:
            return {"qpos": [], "qvel": [], "ctrl": [], "time": 0.0}

        return {
            "qpos": self.current_keyframe.get("qpos", []),
            "qvel": self.current_keyframe.get("qvel", []),
            "ctrl": self.current_keyframe.get("ctrl", []),
            "time": self.current_keyframe.get("time", 0.0),
        }

    def start(self):
        print(f"Starting VuerSim server on http://localhost:{self.port}")
        try:
            self.app.run()
        except KeyboardInterrupt:
            print("\nShutting down VuerSim...")
            self.running = False

    def stop(self):
        self.running = False

    def is_ready(self) -> bool:
        return self.session is not None and self.running