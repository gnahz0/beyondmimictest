import csv
import numpy as np
import matplotlib.pyplot as plt

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

def plot_comparison():
    # Load data
    falcon_data = load_qpos_data("falcon_policy_qpos.csv")
    mujoco_data = load_qpos_data("simple_mujoco_qpos.csv")
    
    if falcon_data is None or mujoco_data is None:
        print("Could not load data files. Make sure both CSV files exist.")
        return
    
    print(f"Falcon policy data shape: {falcon_data.shape}")
    print(f"MuJoCo physics data shape: {mujoco_data.shape}")
    
    # Create time arrays
    falcon_time = np.arange(len(falcon_data))
    mujoco_time = np.arange(len(mujoco_data))
    
    # Plot first 3 joints
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    joint_names = ['Joint 0 (X position)', 'Joint 1 (Y position)', 'Joint 2 (Z position)']
    
    for i in range(3):
        axes[i].plot(falcon_time, falcon_data[:, i], label='Vuer MuJoCo Sim', color='blue', linewidth=1.5)
        axes[i].plot(mujoco_time, mujoco_data[:, i], label='MuJoCo Sim', color='red', linewidth=1.5, alpha=0.8)
        
        # Add horizontal line at initial value
        initial_value = falcon_data[0, i]  # Use falcon data initial value as reference
        axes[i].axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Find first occurrence where difference > 0.05
        min_length = min(len(falcon_data), len(mujoco_data))
        differences = np.abs(falcon_data[:min_length, i] - mujoco_data[:min_length, i])
        first_divergence = np.where(differences > 0.025)[0]
        
        if len(first_divergence) > 0:
            divergence_step = first_divergence[0]
            axes[i].axvline(x=divergence_step, color='orange', linestyle='-', alpha=0.8, linewidth=2)
            axes[i].text(divergence_step + 50, axes[i].get_ylim()[1] * 0.9, 
                        f'Diverge: {divergence_step}', rotation=90, 
                        verticalalignment='top', color='orange', fontweight='bold')
        
        axes[i].set_title(f'{joint_names[i]}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Position (rad/m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('First 3 Joints: Vuer MuJoCo Sim vs MuJoCo Sim', fontsize=16, y=1.02)
    plt.show()
    

if __name__ == "__main__":
    plot_comparison()