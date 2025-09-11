import threading
import time
import math
from pathlib import Path
from vuer_sim_example.mjcf_viewer import MJCFVuer

def control_thread(viewer):
    print("Control thread started", flush=True)
    time.sleep(3)
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        ctrl_array = [0.0] * len(viewer.data.ctrl)
        ctrl_array[6] = 80 * math.sin(2 * math.pi * 0.5 * current_time)
        viewer.set_control(ctrl_array)
        time.sleep(0.01)

def main():
    mjcf_path = Path(__file__).parent / "mjcf_models" / "scene_g1_29dof_freebase.mjcf.xml"
    viewer = MJCFVuer(mjcf_path=str(mjcf_path), port=8012)
    
    threading.Thread(target=control_thread, args=(viewer,), daemon=True).start()
    
    print("Open: http://localhost:8012")
    viewer.start()

if __name__ == "__main__":
    main()