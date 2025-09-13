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

def control_loop(env, policy, output_file, save_npz=False):
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
            # Record all keyframe data - ensure consistent numpy arrays
            try:
                qpos_arr = np.array(state["qpos"], dtype=np.float64)
                qvel_arr = np.array(state["qvel"], dtype=np.float64)
                qacc_arr = np.array(state["qacc"], dtype=np.float64)
                ctrl_arr = np.array(state["ctrl"], dtype=np.float64)
                qfrc_applied_arr = np.array(state["qfrc_applied"], dtype=np.float64)
                qfrc_constraint_arr = np.array(state["qfrc_constraint"], dtype=np.float64)
                qfrc_bias_arr = np.array(state["qfrc_bias"], dtype=np.float64)
                
                # Debug print for first few steps to check shapes
                if step_count <= 3:
                    print(f"Step {step_count} shapes:")
                    print(f"  qpos: {qpos_arr.shape}, qvel: {qvel_arr.shape}")
                    print(f"  qacc: {qacc_arr.shape}, ctrl: {ctrl_arr.shape}")
                    print(f"  qfrc_applied: {qfrc_applied_arr.shape}")
                    print(f"  qfrc_constraint: {qfrc_constraint_arr.shape}")
                    print(f"  qfrc_bias: {qfrc_bias_arr.shape}")
                
                data_arrays['qpos'].append(qpos_arr)
                data_arrays['qvel'].append(qvel_arr)
                data_arrays['qacc'].append(qacc_arr)
                data_arrays['ctrl'].append(ctrl_arr)
                data_arrays['qfrc_applied'].append(qfrc_applied_arr)
                data_arrays['qfrc_constraint'].append(qfrc_constraint_arr)
                data_arrays['qfrc_bias'].append(qfrc_bias_arr)
                
            except Exception as e:
                print(f"Error processing keyframe data at step {step_count}: {e}")
                print(f"State keys: {state.keys()}")
                for key in ['qpos', 'qvel', 'qacc', 'ctrl', 'qfrc_applied', 'qfrc_constraint', 'qfrc_bias']:
                    if key in state:
                        print(f"  {key}: type={type(state[key])}, len={len(state[key]) if hasattr(state[key], '__len__') else 'N/A'}")
                    else:
                        print(f"  {key}: MISSING")
                raise
        else:
            # Record only qpos for CSV
            qpos_data.append(qpos.copy())
        
        step_count += 1
        
        o, r, d, info = env.step()
        
        # Save and exit after 5000 steps
        if step_count >= 5000:
            if save_npz:
                # Convert lists to numpy arrays with better error handling
                final_arrays = {}
                print("NPZ Data Dimensions:")
                for key, data_list in data_arrays.items():
                    try:
                        # Check if all arrays in the list have the same shape
                        if len(data_list) > 0:
                            first_shape = data_list[0].shape if hasattr(data_list[0], 'shape') else len(data_list[0])
                            print(f"  {key}: checking {len(data_list)} arrays, first shape: {first_shape}")
                            
                            # Check for shape consistency
                            inconsistent_shapes = []
                            for i, arr in enumerate(data_list):
                                current_shape = arr.shape if hasattr(arr, 'shape') else len(arr)
                                if current_shape != first_shape:
                                    inconsistent_shapes.append((i, current_shape))
                            
                            if inconsistent_shapes:
                                print(f"  WARNING: {key} has inconsistent shapes:")
                                for i, shape in inconsistent_shapes[:5]:  # Show first 5
                                    print(f"    Step {i}: {shape} (expected: {first_shape})")
                                if len(inconsistent_shapes) > 5:
                                    print(f"    ... and {len(inconsistent_shapes)-5} more")
                                
                                # Try to pad or truncate to make consistent
                                print(f"  Attempting to fix {key} by using consistent length...")
                                # Find the most common length
                                lengths = [len(arr) if hasattr(arr, '__len__') else 0 for arr in data_list]
                                from collections import Counter
                                most_common_length = Counter(lengths).most_common(1)[0][0]
                                print(f"  Using length: {most_common_length}")
                                
                                # Fix inconsistent arrays
                                fixed_data = []
                                for arr in data_list:
                                    if len(arr) == most_common_length:
                                        fixed_data.append(arr)
                                    elif len(arr) > most_common_length:
                                        fixed_data.append(arr[:most_common_length])  # Truncate
                                    else:
                                        # Pad with zeros
                                        padded = np.zeros(most_common_length, dtype=np.float64)
                                        padded[:len(arr)] = arr
                                        fixed_data.append(padded)
                                final_arrays[key] = np.array(fixed_data)
                            else:
                                final_arrays[key] = np.array(data_list)
                                
                            print(f"  {key}: {final_arrays[key].shape}")
                        else:
                            print(f"  {key}: empty data")
                            final_arrays[key] = np.array([])
                            
                    except Exception as e:
                        print(f"  ERROR with {key}: {e}")
                        print(f"    Data types: {[type(x) for x in data_list[:3]]}")
                        raise
                
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
    control_loop(env, policy, str(output_file), save_npz=args.npz)

if __name__ == "__main__":
    main()