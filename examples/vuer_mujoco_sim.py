import time
import threading
import argparse
from pathlib import Path
import numpy as np
import csv

from vuer_sim_example.sim.vuer_sim import VuerSim
from vuer_sim_example.envs.base_env import BaseEnv
from vuer_sim_example.tasks.falcon_task import FalconTask
from vuer_sim_example.policies import LocoManipPolicy

def control_loop(env, policy, output_file, save_npz=False, steps=5000):
    while not env.physics.is_ready():
        time.sleep(0.1)
    
    policy.reset()
    o = env.reset()
    
    # Data storage - either just qpos for CSV or all states for NPZ
    if save_npz:
        data_arrays = {
            'qpos': [],
            'qvel': [],
            'qacc': [],
            'ctrl': [],
            'qfrc_applied': [],
            'qfrc_constraint': [],
            'qfrc_bias': []
        }
    else:
        qpos_data = []
    
    step_count = 0
    
    while True:
        state = env.physics.get_state()
        qpos = np.array(state["qpos"])
        qvel = np.array(state["qvel"])

        if save_npz:
            # Record all keyframe data
            data_arrays['qpos'].append(np.array(state["qpos"], dtype=np.float64))
            data_arrays['qvel'].append(np.array(state["qvel"], dtype=np.float64))
            data_arrays['qacc'].append(np.array(state["qacc"], dtype=np.float64))
            data_arrays['ctrl'].append(np.array(state["ctrl"], dtype=np.float64))
            data_arrays['qfrc_applied'].append(np.array(state["qfrc_applied"], dtype=np.float64))
            data_arrays['qfrc_constraint'].append(np.array(state["qfrc_constraint"], dtype=np.float64))
            data_arrays['qfrc_bias'].append(np.array(state["qfrc_bias"], dtype=np.float64))
        else:
            # Record only qpos for CSV
            qpos_data.append(qpos.copy())
        
        step_count += 1
        
        o, r, d, info = env.step()
        
        # Save and exit after specified steps
        if step_count >= steps:
            if save_npz:
                # Convert lists to numpy arrays
                final_arrays = {}
                print("NPZ Data Dimensions:")
                for key, data_list in data_arrays.items():
                    final_arrays[key] = np.array(data_list)
                    print(f"  {key}: {final_arrays[key].shape}")

                print(f"Saving NPZ data to {output_file}")
                np.savez(output_file, **final_arrays)
            else:
                print(f"Saving {step_count} qpos values to {output_file}")
                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    for qpos_step in qpos_data:
                        writer.writerow(qpos_step)
            print("Done! Exiting.")
            break

        if d:
            policy.reset()
            env.reset()

def start_server_thread(env):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    env.start()

def main():
    parser = argparse.ArgumentParser(description="Vuer MuJoCo simulation with FALCON policy")
    parser.add_argument("--xml", required=True, help="Path to MJCF XML file")
    parser.add_argument("--file", required=True, help="Output file path")
    parser.add_argument("--npz", action="store_true", help="Save all keyframe data to NPZ format instead of CSV qpos only")
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps to record (default: 5000)")
    args = parser.parse_args()
    
    # Resolve paths
    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = Path(__file__).parent / xml_path
    
    output_file = Path(args.file)
    if not output_file.is_absolute():
        output_file = Path(__file__).parent / output_file
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the G1 ONNX policy (assumes humanoid robot structure)
    onnx_path = Path(__file__).parent / "policies" / "g1_29dof.onnx"
    
    policy = LocoManipPolicy(str(onnx_path), policy_action_scale=0.25)
    env = BaseEnv(physics=VuerSim(mjcf_path=str(xml_path), port=8012), task=FalconTask())
    
    threading.Thread(target=start_server_thread, args=(env,), daemon=True).start()
    
    print(f"Using XML: {xml_path}")
    print(f"Output file: {output_file}")
    print(f"Save format: {'NPZ (all keyframe data)' if args.npz else 'CSV (qpos only)'}")
    print("Open: http://localhost:8012")
    control_loop(env, policy, str(output_file), save_npz=args.npz, steps=args.steps)

if __name__ == "__main__":
    main()