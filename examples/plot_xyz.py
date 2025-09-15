import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_qpos_data(filename):
    """Load qpos data from CSV file"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    data.append([float(x) for x in row])
        return np.array(data)
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def plot_xyz_comparison(vuer_mujoco_file, mujoco_file, output_name):
    """Plot XYZ comparison between Vuer MuJoCo and MuJoCo simulations"""
    # Load data from both simulators
    vuer_data = load_qpos_data(vuer_mujoco_file)
    mujoco_data = load_qpos_data(mujoco_file)
    
    if vuer_data is None or mujoco_data is None:
        print("Could not load one or both data files")
        return
    
    print(f"Vuer MuJoCo data shape: {vuer_data.shape}")
    print(f"MuJoCo data shape: {mujoco_data.shape}")
    
    # Create time arrays
    vuer_time = np.arange(len(vuer_data))
    mujoco_time = np.arange(len(mujoco_data))
    
    # Plot first 3 positions (X, Y, Z)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    coord_names = ['X Position', 'Y Position', 'Z Position']
    
    for i in range(3):
        axes[i].plot(vuer_time, vuer_data[:, i], label='Vuer MuJoCo Sim', color='red', linewidth=1.5)
        axes[i].plot(mujoco_time, mujoco_data[:, i], label='MuJoCo Sim', color='blue', linewidth=1.5, alpha=0.8)
        
        # Add horizontal line at initial value (use vuer_data as reference)
        initial_value = vuer_data[0, i]
        axes[i].axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Find first occurrence where difference > 0.025
        min_length = min(len(vuer_data), len(mujoco_data))
        differences = np.abs(vuer_data[:min_length, i] - mujoco_data[:min_length, i])
        first_divergence = np.where(differences > 0.025)[0]
        
        if len(first_divergence) > 0:
            divergence_step = first_divergence[0]
            axes[i].axvline(x=divergence_step, color='orange', linestyle='-', alpha=0.8, linewidth=2)
            axes[i].text(divergence_step + 50, axes[i].get_ylim()[1] * 0.9, 
                        f'Diverge: {divergence_step}', rotation=90, 
                        verticalalignment='top', color='orange', fontweight='bold')
        
        axes[i].set_title(f'{coord_names[i]}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Position (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'XYZ Trajectory Comparison: Vuer MuJoCo vs MuJoCo', fontsize=16, y=1.02)
    
    # Save the plot
    output_path = Path("plots") / f"{output_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Close the plot to avoid display issues in headless environments
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare XYZ trajectories between Vuer MuJoCo and MuJoCo simulations")
    parser.add_argument("--data_vuer_mujoco", required=True, help="Path to Vuer MuJoCo CSV data file")
    parser.add_argument("--data_mujoco", required=True, help="Path to MuJoCo CSV data file")
    parser.add_argument("--output", required=True, help="Output filename (without .png extension)")
    args = parser.parse_args()
    
    # Resolve file paths
    vuer_file = Path(args.data_vuer_mujoco)
    if not vuer_file.is_absolute():
        vuer_file = Path(__file__).parent / vuer_file
    
    mujoco_file = Path(args.data_mujoco)
    if not mujoco_file.is_absolute():
        mujoco_file = Path(__file__).parent / mujoco_file
    
    if not vuer_file.exists():
        print(f"Vuer MuJoCo data file not found: {vuer_file}")
        return
    
    if not mujoco_file.exists():
        print(f"MuJoCo data file not found: {mujoco_file}")
        return
    
    plot_xyz_comparison(str(vuer_file), str(mujoco_file), args.output)

if __name__ == "__main__":
    main()