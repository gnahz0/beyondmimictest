import sys
import argparse
import yaml

import numpy as np

sys.path.append("../")
sys.path.append("./sim2real")

from sim2real.sim_env.base_sim_vuer import BaseSimulatorVuer

class LocoManipSimulatorVuer(BaseSimulatorVuer):
    def __init__(self, config):
        super().__init__(config)
        self.EE_xfrc = 0
        self.t = 0
        self.left_hand_link_name = self.config.get("left_hand_link_name", "left_hand_link")
        self.right_hand_link_name = self.config.get("right_hand_link_name", "right_hand_link")

    def init_scene(self):
        """Initialize scene with LocoManip specific setup."""
        super().init_scene()
        NUM_FEET_SENSORS = 8
        self.ffss_idx = len(self.mj_data.sensordata) - NUM_FEET_SENSORS * 3

    async def setup_vuer_scene(self, session):
        """Setup Vuer scene with LocoManip specific camera positioning."""
        await super().setup_vuer_scene(session)
        self.logger.info("[LocoManipVuer] LocoManip scene setup completed.")

    def control_computation_thread(self):
        """LocoManip specific control computation thread."""
        self.logger.info("[LocoManipControl] Starting LocoManip control computation thread...")
        
        while self.sim_running:
            try:
                self.robot_bridge.PublishLowState()
                if self.robot_bridge.joystick:
                    self.robot_bridge.PublishWirelessController()
                
                self.compute_torques()
                
                if self.robot_bridge.free_base:
                    ctrl_array = np.concatenate((np.zeros(6), self.torques))
                else:
                    ctrl_array = self.torques
                    
                self.current_ctrl = ctrl_array.tolist()
                
            except Exception as e:
                self.logger.error(f"[LocoManipControl] Control computation error: {e}")
                
            self.rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LocoManip Robot Simulator with Vuer v2")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = LocoManipSimulatorVuer(config)
    simulation.start()