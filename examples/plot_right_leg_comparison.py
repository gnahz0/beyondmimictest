import csv
import numpy as np
import matplotlib.pyplot as plt

# Robot joint configuration from params.py
DOF_NAMES = [
    'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
    'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

def get_right_leg_indices():
    """Get indices for right leg joints in qpos array"""
    # qpos structure: [base_pos(3), base_quat(4), joints(29)]
    # Joint indices start at index 7 in qpos
    base_offset = 7
    
    right_leg_joints = [
        'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
    ]
    
    indices = []
    joint_names = []
    
    for joint_name in right_leg_joints:
        if joint_name in DOF_NAMES:
            joint_idx = DOF_NAMES.index(joint_name)
            print(joint_idx)
            qpos_idx = base_offset + joint_idx
            indices.append(qpos_idx)
            joint_names.append(joint_name)
    
    return indices, joint_names

def load_qpos_data(filename):
    """Load qpos data from CSV file"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append([float(x) for x in row])
        return np.array(data)
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None

def plot_right_leg_comparison():
    # Load data
    falcon_data = load_qpos_data("falcon_policy_qpos.csv")
    mujoco_data = load_qpos_data("simple_mujoco_qpos.csv")
    
    if falcon_data is None or mujoco_data is None:
        print("Could not load data files. Make sure both CSV files exist.")
        return
    
    # Get right leg joint indices and names
    right_leg_indices, right_leg_names = get_right_leg_indices()
    
    print(f"Right leg joint indices in qpos: {right_leg_indices}")
    print(f"Right leg joint names: {right_leg_names}")
    print(f"Falcon policy data shape: {falcon_data.shape}")
    print(f"MuJoCo physics data shape: {mujoco_data.shape}")
    
    # Create time arrays
    falcon_time = np.arange(len(falcon_data))
    mujoco_time = np.arange(len(mujoco_data))
    
    # Plot right leg joints
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Clean joint names for display
    display_names = []
    for name in right_leg_names:
        clean_name = name.replace('right_', '').replace('_joint', '').replace('_', ' ').title()
        display_names.append(clean_name)
    
    for i, (idx, display_name) in enumerate(zip(right_leg_indices, display_names)):
        axes[i].plot(falcon_time, falcon_data[:, idx], label='Vuer MuJoCo Sim', color='blue', linewidth=1.5)
        axes[i].plot(mujoco_time, mujoco_data[:, idx], label='MuJoCo Sim', color='red', linewidth=1.5, alpha=0.8)
        
        # Add horizontal line at initial value
        initial_value = falcon_data[0, idx]  # Use falcon data initial value as reference
        axes[i].axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Find first occurrence where difference > 0.05
        min_length = min(len(falcon_data), len(mujoco_data))
        differences = np.abs(falcon_data[:min_length, idx] - mujoco_data[:min_length, idx])
        first_divergence = np.where(differences > 0.025)[0]
        
        if len(first_divergence) > 0:
            divergence_step = first_divergence[0]
            axes[i].axvline(x=divergence_step, color='orange', linestyle='-', alpha=0.8, linewidth=2)
            axes[i].text(divergence_step + 50, axes[i].get_ylim()[1] * 0.9, 
                        f'Diverge: {divergence_step}', rotation=90, 
                        verticalalignment='top', color='orange', fontweight='bold')
        
        axes[i].set_title(f'Right {display_name}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Angle (rad)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Right Leg Joints: Vuer MuJoCo Sim vs MuJoCo Sim', fontsize=16, y=1.02)
    plt.show()
    

if __name__ == "__main__":
    plot_right_leg_comparison()