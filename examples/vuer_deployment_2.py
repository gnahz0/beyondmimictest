#!/usr/bin/env python3
"""
Vuer Deployment Test with BeyondMimic Policy

This script tests the deployment of BeyondMimic ONNX models in MuJoCo using Vuer.
It uses the BeyondMimicPolicy class created for the exported whole_body_tracking models.
"""

import argparse
import time
import threading
from pathlib import Path
import numpy as np

# Import the BeyondMimic policy we created
from beyond_mimic_policy import create_beyond_mimic_policy


def control_loop(env, policy, steps=10000):
    """
    Simple control loop that runs the policy - matches vuer_mujoco_sim.py style
    """
    # Wait for simulation to be ready
    while not env.physics.is_ready():
        time.sleep(0.1)
    
    print("Simulation ready! Starting policy test...")
    
    # Reset policy and environment
    policy.reset()
    o = env.reset()
    
    step_count = 0
    
    # Main control loop - simple like vuer_mujoco_sim.py
    while True:
        state = env.physics.get_state()
        qpos = np.array(state["qpos"])
        qvel = np.array(state["qvel"])

        # Policy predicts action (should return torques or be handled internally)
        action = policy.predict(qpos, qvel)
        
        # Debug: Check action and state occasionally
        if step_count % 100 == 0:
            print(f"🔍 Step {step_count:3d} Debug:")
            print(f"   - qpos shape: {qpos.shape}, qvel shape: {qvel.shape}")
            print(f"   - Base pos: {qpos[4:7] if len(qpos) >= 7 else 'N/A'}")
            print(f"   - Joint pos (first 5): {qpos[7:12] if len(qpos) >= 12 else 'N/A'}")
            print(f"   - Action shape: {action.shape}")
            print(f"   - Action range: [{action.min():.3f}, {action.max():.3f}]")
            print(f"   - Motion time: {getattr(policy, 'time_step', 'N/A')}")
            
            # Check if robot is falling or spinning
            if len(qpos) >= 7:
                base_height = qpos[6]  # Z position
                if base_height < 0.3:
                    print(f"   ⚠️ Robot falling! Base height: {base_height:.3f}")
        
        o, r, d, info = env.step(action)
        step_count += 1
        
        # Exit after specified steps
        if step_count >= steps:
            break

        # Reset if done
        if d:
            policy.reset()
            env.reset()

    print("Control loop test completed!")

def start_server_thread(env):
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    env.start()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test BeyondMimic policy deployment in Vuer")
    parser.add_argument("--xml", required=True,
                       help="MuJoCo XML file path")
    parser.add_argument("--steps", type=int, default=1000, help="Number of test steps")
    args = parser.parse_args()

    # Resolve paths
    xml_path = Path(__file__).parent / args.xml
    if not xml_path.is_absolute():
        xml_path = Path(__file__).parent / xml_path

    # Set model paths
    onnx_path = Path(__file__).parent / "policies" / "2025-09-28_03-05-40_g1_lafan1_dance1_s1.onnx"
    motion_npz_path = Path(__file__).parent / "motion_data" / "motions_lafan1_dance1_s1" / "motion.npz"


    print("=== BeyondMimic Vuer Deployment Test ===")
    print(f"XML file: {xml_path}")
    print(f"ONNX model: {onnx_path}")
    print(f"Motion NPZ: {motion_npz_path}")
    print(f"Test steps: {args.steps}")

    # Create the BeyondMimic policy
    print("\n=== Creating BeyondMimic Policy ===")
    
    # Debug: Check what joints are found in XML vs ONNX
    from beyond_mimic_policy import get_mj_actuator_joint_names_recursive
    all_actuator_joints = get_mj_actuator_joint_names_recursive(str(xml_path))
    joint_only = [j for j in all_actuator_joints if j != "floating_base_joint"]
    
    print(f"🔍 JOINT MAPPING DEBUG:")
    print(f"   - MuJoCo joints (first 10): {joint_only[:10]}")
    
    # Create the BeyondMimic policy with joint mapping
    policy = create_beyond_mimic_policy(
        str(onnx_path),
        str(motion_npz_path),
        str(xml_path)
    )
    
    print(f"   - ONNX joints: {len(policy.joint_names)} total")
    if hasattr(policy, 'onnx_to_mj') and policy.onnx_to_mj is not None:
        print(f"   - Joint mapping applied!")
        print(f"\n📋 COMPLETE JOINT MAPPING (ONNX -> MuJoCo):")
        for i in range(len(policy.joint_names)):
            mj_idx = policy.onnx_to_mj[i]
            print(f"   [{i:2d}] {policy.joint_names[i]:30s} -> MJ[{mj_idx:2d}]")
        print()
    else:
        print(f"   - No joint mapping applied")

    print("✅ Policy created successfully!")
    info = policy.get_info()
    print(f"   - Model: {info['n_joints']} joints, {info['obs_dim']} obs_dim")
    print(f"   - Motion: Embedded in ONNX model")
    print(f"   - Time step: {info['time_step']}")

    from vuer_sim_example.sim.vuer_sim import VuerSim
    from vuer_sim_example.envs.base_env import BaseEnv
    from vuer_sim_example.tasks.falcon_task import FalconTask

    print("\n=== Setting up Vuer Environment ===")

    # Create environment
    env = BaseEnv(
        physics=VuerSim(mjcf_path=str(xml_path), port=8012),
        task=FalconTask()
    )

    threading.Thread(target=start_server_thread, args=(env,), daemon=True).start()

    print("✅ Environment created")
    print("🌐 Open: http://localhost:8012")

    # Run control loop test
    control_loop(env, policy, args.steps)

    print("\n✅ Deployment test completed successfully!")

if __name__ == "__main__":
    exit(main())
