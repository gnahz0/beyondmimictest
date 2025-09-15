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

def plot_xyz_trails(csv_files, labels, output_name):
    """Plot XYZ trails from multiple CSV files with different colors"""
    # Define a set of distinct colors for different trails
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Load all datasets
    datasets = []
    for i, csv_file in enumerate(csv_files):
        data = load_qpos_data(csv_file)
        if data is None:
            print(f"Skipping {csv_file} due to loading error")
            continue
        datasets.append((data, labels[i] if i < len(labels) else f"Dataset {i+1}", colors[i % len(colors)]))
        print(f"Loaded {labels[i] if i < len(labels) else f'Dataset {i+1}'} shape: {data.shape}")
    
    if not datasets:
        print("No valid datasets loaded!")
        return
    
    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    coord_names = ['X Position', 'Y Position', 'Z Position']
    
    for i in range(3):  # X, Y, Z coordinates
        for data, label, color in datasets:
            time_steps = np.arange(len(data))
            axes[i].plot(time_steps, data[:, i], label=label, color=color, linewidth=1.5, alpha=0.8)
        
        # Add horizontal line at first dataset's initial value as reference
        if datasets:
            initial_value = datasets[0][0][0, i]  # First dataset, first timestep, coordinate i
            axes[i].axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[i].set_title(f'{coord_names[i]}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Position (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'XYZ Trajectory Trails Comparison', fontsize=16, y=1.02)
    
    # Save the plot
    output_path = Path("plots") / f"{output_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Close the plot to avoid display issues in headless environments
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot XYZ trajectories from multiple CSV files with different colors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 3 different simulations
  python plot_xyz_trails.py --csvs data/sim1.csv data/sim2.csv data/sim3.csv --labels "Sim 1" "Sim 2" "Sim 3" --output multi_sim_comparison
  
  # Compare with automatic labels
  python plot_xyz_trails.py --csvs data/ball_mujoco.csv data/ball_vuer.csv --output ball_comparison
        """)
    
    parser.add_argument("--csvs", nargs='+', required=True, help="Paths to CSV data files")
    parser.add_argument("--labels", nargs='*', help="Labels for each CSV file (optional)")
    parser.add_argument("--output", required=True, help="Output filename (without .png extension)")
    args = parser.parse_args()
    
    # Resolve CSV file paths
    csv_files = []
    for csv_file in args.csvs:
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = Path(__file__).parent / csv_path
        
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}")
            continue
        csv_files.append(str(csv_path))
    
    if not csv_files:
        print("No valid CSV files found!")
        return
    
    # Generate labels if not provided
    labels = args.labels if args.labels else []
    if len(labels) < len(csv_files):
        # Generate labels for missing ones based on filename
        for i in range(len(labels), len(csv_files)):
            filename = Path(csv_files[i]).stem
            labels.append(filename)
    
    print(f"Plotting {len(csv_files)} datasets:")
    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        print(f"  {i+1}. {label}: {csv_file}")
    
    plot_xyz_trails(csv_files, labels, args.output)

if __name__ == "__main__":
    main()