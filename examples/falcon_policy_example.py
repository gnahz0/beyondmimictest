import time
import threading
from pathlib import Path
import numpy as np

from vuer_sim_example.sim.vuer_sim import VuerSim
from vuer_sim_example.envs.base_env import BaseEnv
from vuer_sim_example.tasks.falcon_task import FalconTask
from vuer_sim_example.policies import LocoManipPolicy

def control_loop(env, policy):
    while not env.physics.is_ready():
        time.sleep(0.1)
    
    policy.reset()
    o = env.reset()
    
    while True:
        state = env.physics.get_state()
        qpos = np.array(state["qpos"])
        qvel = np.array(state["qvel"])
        gamepad = state["gamepad"]

        action = policy.predict(qpos, qvel, gamepad)
        o, r, d, info = env.step(action)

        if d:
            policy.reset()
            env.reset()

def start_server_thread(env):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    env.start()

def main():
    port_num = 8012
    mjcf_path = Path(__file__).parent / "mjcf_models" / "scene_g1_29dof_freebase.mjcf.xml"
    onnx_path = Path(__file__).parent / "policies" / "g1_29dof.onnx"
    
    policy = LocoManipPolicy(str(onnx_path), policy_action_scale=0.25)
    env = BaseEnv(physics=VuerSim(mjcf_path=str(mjcf_path), port=port_num), task=FalconTask())
    
    threading.Thread(target=start_server_thread, args=(env,), daemon=True).start()
    
    print(f"Open: http://localhost:{port_num}")
    control_loop(env, policy)

if __name__ == "__main__":
    main()