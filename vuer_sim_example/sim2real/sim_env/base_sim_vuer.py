import argparse
import sys
import time
from asyncio import sleep
from pathlib import Path
import os

try:
    from pynput import keyboard

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

import mujoco
import numpy as np
import yaml
from loguru import logger
from loop_rate_limiters import RateLimiter

# Vuer imports
from vuer import Vuer
from vuer.events import MjStep
from vuer.schemas import MuJoCo
from vuer.schemas.scene_components import SceneElement

sys.path.append("../")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from sim2real.utils.robot import Robot
from sim2real.utils.sdk2py_bridge import create_sdk2py_bridge

PORT = 8016

class AnchorActuator(SceneElement):
    tag = "AnchorActuator"

    bodyId: int = -1
    enabled: bool = True
    scale: float = 20
    forceDestination: tuple = (1, 1, 1)
    showForceVector: bool = False
    showForceText: bool = False
    showPivotControls: bool = True

class BaseSimulatorVuer:
    def __init__(self, config):
        self.config = config

        self.is_mujoco_loaded = False
        self.current_ctrl = None
        self.sim_running = True
        self.j = 0
        self.key_listener = None
        self.force_y = 3.0
        self.update_anchors_flag = False
        self.fps = 50.0
        self.scale_values = [75, 0]
        self.scale_idx = 0

        self.init_config()
        print('initialized scene')
        self.init_scene()
        self.init_factory()
        self.init_robot_bridge()
        self.init_key_logger()
        self.init_vuer_app()

    def init_config(self):
        self.robot = Robot(self.config)
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.logger = logger
        self.rate = RateLimiter(1 / self.config["SIMULATE_DT"])

    def init_factory(self):
        if self.sdk_type == "unitree":
            if self.config.get("INTERFACE", None):
                if sys.platform == "linux":
                    self.config["INTERFACE"] = "lo"
                elif sys.platform == "darwin":
                    self.config["INTERFACE"] = "lo0"
                else:
                    raise NotImplementedError("Only support Linux and MacOS.")
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
        self.logger.info(f"SDK TYPE: {self.sdk_type}")

    def init_scene(self):
        """Initialize MuJoCo scene."""
        scene_path = self.config["ROBOT_SCENE"]

        if not os.path.isabs(scene_path):
            sim2real_path = Path(__file__).parent.parent
            scene_path = sim2real_path / scene_path
            scene_path = str(scene_path.resolve())

        self.logger.info(f"Loading scene from: {scene_path}")
        self.scene_path = scene_path

        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        self.base_id = self.mj_model.body(base_body_name).id

    def init_robot_bridge(self):
        self.robot_bridge = create_sdk2py_bridge(self.mj_model, self.mj_data, self.config)

    def init_key_logger(self):
        """Initialize Python-side key logger using pynput."""
        if not PYNPUT_AVAILABLE:
            self.logger.warning("pynput not available - key logging disabled")
            return

        def on_key_press(key):
            try:
                if key.char == '7':
                    print("Python Key Logger: 7 pressed")
                    self.force_y -= 0.1
                    print(f"Force Y decreased to: {self.force_y}")
                    self.update_anchors_flag = True
                elif key.char == '8':
                    print("Python Key Logger: 8 pressed")
                    self.force_y += 0.1
                    print(f"Force Y increased to: {self.force_y}")
                    self.update_anchors_flag = True
                elif key.char == '9':
                    print("Python Key Logger: 9 pressed")
                    self.scale_idx = 1 - self.scale_idx
                    print(f"Scale toggled to: {self.scale_values[self.scale_idx]}")
                    self.update_anchors_flag = True
            except AttributeError:
                pass

        self.key_listener = keyboard.Listener(on_press=on_key_press)
        self.key_listener.start()
        self.logger.info("Python key logger started")

    def init_vuer_app(self):
        asset_root = self.config.get("ASSET_ROOT", "../scenes/mujoco_scenes/g1")
        if not os.path.isabs(asset_root):
            sim2real_path = Path(__file__).parent.parent
            asset_root = sim2real_path / asset_root
            asset_root = str(asset_root.resolve())

        self.asset_root = asset_root
        self.app = Vuer(static_root=asset_root, port=PORT)

        @self.app.add_handler("WEBSOCKET_CONNECT")
        async def on_connect(event, session):
            print("WebSocket connected!")

        @self.app.add_handler("*")
        async def debug_all_events(event, session):
            print(f"DEBUG: Event received: {event.type}")

        @self.app.add_handler("ON_MUJOCO_LOAD")
        async def mujoco_load_handler(event, session):
            self.is_mujoco_loaded = True
            self.logger.info("[Vuer] MuJoCo component loaded successfully.")
            print("ON_MUJOCO_LOAD triggered!")

        @self.app.spawn(start=True)
        async def main_session(session):
            await self.setup_scene_and_run(session)

    async def setup_scene_and_run(self, session):
        """Setup Vuer scene with MuJoCo model and run simulation."""
        await sleep(2)

        self.logger.info("[Vuer] Setting up scene...")

        from glob import glob
        original_dir = os.getcwd()
        try:
            os.chdir(self.asset_root)
            all_files = glob("**/*.*", recursive=True)
        finally:
            os.chdir(original_dir)

        scene_file = None
        for f in all_files:
            if f.endswith("scene_g1_29dof_freebase.xml"):
                scene_file = f
                break

        if not scene_file:
            self.logger.error(f"[Vuer] Scene file not found in assets!")
            return

        scene_url = f"http://localhost:{PORT}/static/{scene_file}"

        # create asset URLs
        asset_urls = [f"http://localhost:{PORT}/static/{f}" for f in all_files]
        self.logger.info(f"[Vuer] Found {len(asset_urls)} assets")
        self.logger.info(f"[Vuer] Scene URL: {scene_url}")

        print(f"Assets count: {len(asset_urls)}")
        print(f"Scene URL: {scene_url}")

        self.logger.info("[Vuer] Inserting MuJoCo component...")
        session.upsert @ MuJoCo(
            AnchorActuator(
                key="right",
                bodyId=34,
                forceDestination=[0, self.force_y, 0],
                anchor=[0, 0.0, 0],
                enabled=True,
                showForceVector=True,
                showForceText=True,
                showPivotControls=False,
                maxForce=500.0,
                scale=self.scale_values[self.scale_idx],
                damping=0.5,
            ),
            AnchorActuator(
                key="left",
                bodyId=26,
                forceDestination=[0, self.force_y, 0],
                anchor=[0, 0.0, 0],
                enabled=True,
                showForceVector=True,
                showForceText=True,
                showPivotControls=False,
                maxForce=500.0,
                scale=self.scale_values[self.scale_idx],
                damping=0.5,
            ),
            key="robot",
            src=scene_url,
            assets=asset_urls,
            frameKeys="qpos qvel ctrl mocap_pos mocap_quat",
            pause=False,
            useLights=True,
            fps=self.fps,
        )

        self.logger.info("[Vuer] MuJoCo component inserted, waiting for load...")
        await sleep(5)

        if not self.is_mujoco_loaded:
            self.logger.error("[Vuer] MuJoCo failed to load after 5 seconds!")

        await self.vuer_simulation_loop(session)

    async def vuer_simulation_loop(self, session):
        """Main simulation loop controlled by Vuer - now includes control computation."""
        self.logger.info("[Vuer] Starting simulation loop...")

        await sleep(2)

        while self.sim_running:
            loop_start = time.perf_counter()
            try:
                if self.update_anchors_flag:
                    print('Updating anchor actuators via MuJoCo upsert')
                    session.upsert @ MuJoCo(
                        AnchorActuator(
                            key="right",
                            bodyId=34,
                            forceDestination=[0, self.force_y, 0],
                            anchor=[0, 0.0, 0],
                            enabled=True,
                            showForceVector=True,
                            showForceText=True,
                            showPivotControls=False,
                            maxForce=500.0,
                            scale=self.scale_values[self.scale_idx],
                            damping=0.5,
                        ),
                        AnchorActuator(
                            key="left",
                            bodyId=26,
                            forceDestination=[0, self.force_y, 0],
                            anchor=[0, 0.0, 0],
                            enabled=True,
                            showForceVector=True,
                            showForceText=True,
                            showPivotControls=False,
                            maxForce=500.0,
                            scale=self.scale_values[self.scale_idx],
                            damping=0.5,
                        ),
                        key="robot"
                    )
                    self.update_anchors_flag = False

                ctrl_array = self.compute_control_step()

                frame = await session.rpc(
                    MjStep(key="robot", sim_steps=1, ctrl=ctrl_array),
                    ttl=5,
                )

                key_frame = frame.value["keyFrame"]
                self.mj_data.qpos = key_frame.get("qpos")
                self.mj_data.qvel = key_frame.get("qvel")

            except Exception as e:
                self.logger.error(f"[Vuer] Simulation step failed: {e}")

            time_elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 1.0 / self.fps - time_elapsed)
            await sleep(sleep_time)

    def compute_control_step(self):
        """Compute control signals for a single simulation step."""
        try:
            self.robot_bridge.PublishLowState()

            self.compute_torques()

            if self.robot_bridge.free_base:
                ctrl_array = np.concatenate((np.zeros(6), self.torques))
            else:
                ctrl_array = self.torques

            ctrl_array = np.clip(ctrl_array, -35.0, 30.0)

            return ctrl_array.tolist()

        except Exception as e:
            self.logger.error(f"[Control] Control computation error: {e}")
            return []

    def compute_torques(self):
        """Compute motor torques from SDK commands."""
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)

            # Safety check: if qpos or qvel are None, return zero torques until we get valid frame data
            if self.mj_data.qpos is None or self.mj_data.qvel is None:
                self.logger.debug("No valid frame data yet, returning zero torques")
                return

            try:
                for i in range(self.robot_bridge.num_motor):
                    self.torques[i] = (
                            motor_cmd[i].tau
                            + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[7 + i])
                            + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[6 + i])
                    )

                self.j += 1

                if self.j % 50 == 0:
                    motor_cmd_list_tau = [motor_cmd[i].tau for i in range(self.robot_bridge.num_motor)]
                    motor_cmd_list_kp = [motor_cmd[i].kp for i in range(self.robot_bridge.num_motor)]
                    motor_cmd_list_kd = [motor_cmd[i].kd for i in range(self.robot_bridge.num_motor)]
                    motor_cmd_list_q = [motor_cmd[i].q for i in range(self.robot_bridge.num_motor)]
                    motor_cmd_list_dq = [motor_cmd[i].dq for i in range(self.robot_bridge.num_motor)]

                    print(f'[VuerSim] tau (length {len(motor_cmd_list_tau)}): {motor_cmd_list_tau[:3]}')
                    print(f'[VuerSim] Joint kp (length {len(motor_cmd_list_kp)}): {motor_cmd_list_kp[:3]}')
                    print(f'[VuerSim] Joint motor cmd q (length {len(motor_cmd_list_q)}): {motor_cmd_list_q[:3]}')
                    print(f'[VuerSim] Joint mj data q (length {len(self.mj_data.qpos)}): {self.mj_data.qpos[7:10]}')
                    print(f'[VuerSim] Joint kd (length {len(motor_cmd_list_kd)}): {motor_cmd_list_kd[:3]}')
                    print(f'[VuerSim] Joint motor cmd dq (length {len(motor_cmd_list_dq)}): {motor_cmd_list_dq[:3]}')
                    print(f'[VuerSim] Joint mj data qvel (length {len(self.mj_data.qvel)}): {self.mj_data.qvel[6:9]}')
                    print(f'[VuerSim] Final Torques (length {len(self.torques)}): {self.torques[:3]}')
                    print('-------------------------------------------------------------')

            except Exception as e:
                self.logger.error(f"Joint {i} not found in motor_cmd: {e}")

        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    def cleanup(self):
        """Clean shutdown of all threads and connections."""
        self.logger.info("Cleaning up resources...")
        self.sim_running = False

        if self.key_listener:
            self.key_listener.stop()
            self.logger.info("Key listener stopped")

    def start(self):
        """Start the Vuer server (blocking)."""
        self.logger.info("Starting Vuer server...")
        try:
            self.app.run()
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self.sim_running = False
        finally:
            self.cleanup()

if __name__ == "__main__":
    import subprocess

    try:
        subprocess.run(["pkill", "-f", f"port.*{PORT}"], check=False)
        subprocess.run(["lsof", f"-ti:{PORT}", "|", "xargs", "kill", "-9"], shell=True, check=False)
    except:
        pass

    parser = argparse.ArgumentParser(description="Robot Simulator with Vuer")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = BaseSimulatorVuer(config)
    simulation.start()
