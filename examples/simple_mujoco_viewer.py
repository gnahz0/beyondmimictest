# run_g1_viewer.py
import sys
import time
import csv
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

# Resolve the XML path relative to this file; fall back to CWD if needed.
def resolve_xml() -> str:
    here = Path(__file__).resolve().parent
    cand1 = here / "mjcf_models" / "scene_g1_29dof_freebase.mjcf.xml"
    if cand1.exists():
        return str(cand1)
    cand2 = Path.cwd() / "mjcf_models" / "scene_g1_29dof_freebase.mjcf.xml"
    if cand2.exists():
        return str(cand2)
    print(f"[ERROR] Could not find model XML at:\n  {cand1}\n  {cand2}", file=sys.stderr)
    sys.exit(1)

def main():
    xml_path = resolve_xml()
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Optional: ensure zero control (safe default)
    if model.nu > 0:
        data.ctrl[:] = 0.0

    qpos_data = []
    step_count = 0

    # Passive viewer: you step the physics; viewer renders & handles UI.
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            step_start = time.time()
            
            # Record qpos
            qpos_data.append(data.qpos.copy())
            step_count += 1
            
            # Save and exit at 5000 steps
            if step_count >= 5000:
                print(f"Saving {step_count} qpos values to simple_mujoco_qpos.csv")
                with open("simple_mujoco_qpos.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    for qpos_step in qpos_data:
                        writer.writerow(qpos_step)
                print("Done! Exiting.")
                break
            
            mujoco.mj_step(model, data)  # one physics step
            v.sync()                      # render the latest state

            # Keep real-time pacing (sleep the remainder of dt if we're ahead)
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

if __name__ == "__main__":
    main()