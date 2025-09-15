import sys
import time
import csv
import argparse
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

def collect_data_step(data, data_arrays, save_npz):
    """Collect data for one simulation step"""
    if save_npz:
        data_arrays['qpos'].append(data.qpos.copy())
        data_arrays['qvel'].append(data.qvel.copy())
        data_arrays['qacc'].append(data.qacc.copy())
        data_arrays['ctrl'].append(data.ctrl.copy())
        data_arrays['qfrc_applied'].append(data.qfrc_applied.copy())
        data_arrays['qfrc_constraint'].append(data.qfrc_constraint.copy())
        data_arrays['qfrc_bias'].append(data.qfrc_bias.copy())
    else:
        data_arrays['qpos'].append(data.qpos.copy())

def save_data(data_arrays, output_file, save_npz, step_count):
    """Save collected data to file"""
    if save_npz:
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
            for qpos_step in data_arrays['qpos']:
                writer.writerow(qpos_step)
    print("Done! Exiting.")

def main():
    parser = argparse.ArgumentParser(description="MuJoCo simulation viewer")
    parser.add_argument("--xml", required=True, help="Path to MJCF XML file")
    parser.add_argument("--file", required=True, help="Output file path")
    parser.add_argument("--npz", action="store_true", help="Save all keyframe data to NPZ format instead of CSV qpos only")
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps to record (default: 5000)")
    args = parser.parse_args()
    
    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = Path(__file__).parent / xml_path
    
    if not xml_path.exists():
        print(f"[ERROR] Could not find XML file at: {xml_path}", file=sys.stderr)
        sys.exit(1)
        
    output_file = Path(args.file)
    if not output_file.is_absolute():
        output_file = Path(__file__).parent / output_file
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if model.nu > 0:
        data.ctrl[:] = 0.0

    if args.npz:
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
        data_arrays = {'qpos': []}

    step_count = 0

    print(f"Using XML: {xml_path}")
    print(f"Output file: {output_file}")
    print(f"Save format: {'NPZ (all keyframe data)' if args.npz else 'CSV (qpos only)'}")
    
    try:
        # Passive viewer: you step the physics; viewer renders & handles UI.
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                step_start = time.time()
                
                # Record data
                collect_data_step(data, data_arrays, args.npz)
                step_count += 1

                # Save and exit at specified steps
                if step_count >= args.steps:
                    save_data(data_arrays, str(output_file), args.npz, step_count)
                    break
                
                mujoco.mj_step(model, data)  # one physics step
                v.sync()                      # render the latest state

                # Keep real-time pacing (sleep the remainder of dt if we're ahead)
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("Viewer not available (requires mjpython on macOS). Running headless simulation...")
            for step_count in range(args.steps):
                step_start = time.time()
                
                collect_data_step(data, data_arrays, args.npz)
                
                mujoco.mj_step(model, data)

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)
            
            save_data(data_arrays, str(output_file), args.npz, len(data_arrays['qpos']))
        else:
            raise

if __name__ == "__main__":
    main()