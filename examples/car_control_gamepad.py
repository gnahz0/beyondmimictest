import threading
import time
from pathlib import Path
from vuer_sim_example.mjcf_viewer import MJCFVuer

def control_thread(viewer):
    print("Control thread started", flush=True)
    time.sleep(3)
    while True:
        time.sleep(0.1)

def main():
    mjcf_path = Path(__file__).parent / "mjcf_models" / "car.mjcf.xml"
    viewer = MJCFVuer(mjcf_path=str(mjcf_path), port=8012)
    
    threading.Thread(target=control_thread, args=(viewer,), daemon=True).start()
    
    print("Open: http://localhost:8012")
    viewer.start()

if __name__ == "__main__":
    main()