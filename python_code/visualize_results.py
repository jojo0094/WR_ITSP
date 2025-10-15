"""
Visualization utilities for sensor selection results
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


def plot_sensor_selection_results(results_file, output_file=None, k_values=None):
    """
    Plot sensor selection results from saved pickle file.
    
    Parameters
    ----------
    results_file : str
        Path to sensor_selection.pkl file
    output_file : str, optional
        Path to save the plot (if None, displays plot)
    k_values : list, optional
        Specific k values to highlight (default: [25, 50, 100, 150, 200, 250])
    """
    
    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    k_max = results.shape[0]
    
    if k_values is None:
        k_values = [25, 50, 100, 150, 200, 250]
        k_values = [k for k in k_values if k <= k_max]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MI vs number of sensors
    x = np.arange(1, k_max + 1)
    mi_values = results[:, -1]
    
    ax1.plot(x, mi_values, 'b-', linewidth=2, label='Algorithm 1')
    
    # Mark specific k values
    for k in k_values:
        if k <= k_max:
            ax1.plot(k, mi_values[k-1], 'ro', markersize=8)
            ax1.annotate(f'k={k}', xy=(k, mi_values[k-1]), 
                        xytext=(10, -15), textcoords='offset points',
                        fontsize=9, ha='left')
    
    ax1.set_xlabel('Number of Sensors (k)', fontsize=13)
    ax1.set_ylabel('Mutual Information', fontsize=13)
    ax1.set_title('Optimal Sensor Selection - MI Values', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: MI gain rate (derivative)
    if k_max > 1:
        mi_gain = np.diff(mi_values)
        x_gain = x[:-1]
        
        ax2.plot(x_gain, mi_gain, 'g-', linewidth=2, label='MI Gain Rate')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Mark diminishing returns
        if len(mi_gain) > 10:
            avg_gain = np.mean(mi_gain[:10])
            half_gain_idx = np.where(mi_gain < avg_gain * 0.5)[0]
            if len(half_gain_idx) > 0:
                first_half = half_gain_idx[0]
                ax2.axvline(x=first_half, color='r', linestyle='--', alpha=0.5,
                           label=f'Diminishing returns (~k={first_half})')
        
        ax2.set_xlabel('Number of Sensors (k)', fontsize=13)
        ax2.set_ylabel('MI Gain (Δ MI)', fontsize=13)
        ax2.set_title('Marginal Information Gain', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Sensor Selection Results Summary")
    print("="*70)
    print(f"Total nodes in network: {results.shape[1] - 5}")
    print(f"Maximum sensors evaluated: {k_max}")
    print(f"\nMutual Information Values:")
    for k in k_values:
        if k <= k_max:
            sensors = results[k-1, 4:4+k].astype(int)
            mi = results[k-1, -1]
            print(f"  k={k:3d}: MI={mi:8.4f}, Sensors (first 5)={sensors[:5]}")
    
    if k_max > 1:
        print(f"\nTotal MI gain from k=1 to k={k_max}: {mi_values[-1] - mi_values[0]:.4f}")
        print(f"Average MI gain per sensor: {(mi_values[-1] - mi_values[0]) / k_max:.4f}")
    print("="*70)


def plot_mi_comparison(results_path, output_file=None):
    """
    Plot comparison of different sensor selection methods.
    
    Parameters
    ----------
    results_path : str
        Path to results folder
    output_file : str, optional
        Path to save the plot
    """
    
    # This would load the MI comparison results
    # For now, just check if the figure exists
    figure_path = Path(results_path) / "Figure5_MI_comparison.png"
    
    if figure_path.exists():
        print(f"MI comparison plot already exists at: {figure_path}")
    else:
        print(f"Run the main pipeline to generate MI comparison plot")


def export_sensor_locations(results_file, k, output_file="selected_sensors.txt"):
    """
    Export selected sensor locations for a given k.
    
    Parameters
    ----------
    results_file : str
        Path to sensor_selection.pkl file
    k : int
        Number of sensors
    output_file : str
        Output file path
    """
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    if k > results.shape[0]:
        raise ValueError(f"k={k} exceeds maximum evaluated sensors ({results.shape[0]})")
    
    sensors = results[k-1, 4:4+k].astype(int)
    mi_value = results[k-1, -1]
    
    with open(output_file, 'w') as f:
        f.write(f"# Optimal Sensor Placement Results\n")
        f.write(f"# Number of sensors: {k}\n")
        f.write(f"# Mutual Information: {mi_value:.6f}\n")
        f.write(f"#\n")
        f.write(f"# Sensor locations (0-based indexing):\n")
        for i, sensor in enumerate(sensors):
            f.write(f"{sensor}\n")
    
    print(f"Sensor locations exported to: {output_file}")
    print(f"Number of sensors: {k}")
    print(f"MI value: {mi_value:.6f}")
    print(f"Sensors: {sensors}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize sensor selection results')
    parser.add_argument('--results-file', default='Results_folder_python/Sensor_selection/sensor_selection.pkl',
                       help='Path to sensor selection results file')
    parser.add_argument('--output', default='sensor_selection_plot.png',
                       help='Output plot file')
    parser.add_argument('--export-sensors', type=int, metavar='K',
                       help='Export sensor locations for k sensors')
    
    args = parser.parse_args()
    
    if args.export_sensors:
        export_sensor_locations(args.results_file, args.export_sensors)
    else:
        plot_sensor_selection_results(args.results_file, args.output)
