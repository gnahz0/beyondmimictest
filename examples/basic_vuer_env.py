import asyncio
import threading
import time
from pathlib import Path

from vuer_sim_example.sim.vuer_sim import VuerSim
from vuer_sim_example.envs.base_env import BaseEnv
from vuer_sim_example.tasks.base_task import Task


async def control_loop(env):
    """Simple control loop showing API usage."""
    print("Control loop started", flush=True)
    
    while not env.physics.is_ready():
        print("Physics is not ready")
        await asyncio.sleep(0.1)
    
    obs = await env.reset()
    rand_act = env.action_space.sample()
    o, r, d, info = await env.step(rand_act)

    while True:
        rand_act = env.action_space.sample()
        o, r, d, info = await env.step(rand_act)

def control_thread(env):
    """Control thread that sets up async loop."""
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(control_loop(env))

def main():
    mjcf_path = Path(__file__).parent / "mjcf_models" / "scene_g1_29dof_freebase.mjcf.xml"
    
    env = BaseEnv(physics=VuerSim(mjcf_path=str(mjcf_path), port=8012), task=Task())
    threading.Thread(target=control_thread, args=(env,), daemon=True).start()
    
    print("Open: http://localhost:8012")
    env.start()

if __name__ == "__main__":
    main()