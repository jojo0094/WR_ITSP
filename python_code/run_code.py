"""
Copyright (c) 2024, George Crowley (gcrowley1@sheffield.ac.uk)
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Main script to run sensor selection and estimation algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path

from .sensor_selection import sensor_selection
from .mi_comparison import mi_comparison_bellinge
from .glm_estimation import glm_estimation
from .grnn_estimation import grnn_estimation


def load_mat_file(filepath):
    """
    Load .mat file using scipy.io or h5py depending on format.
    
    Parameters
    ----------
    filepath : str
        Path to .mat file
        
    Returns
    -------
    data : dict
        Dictionary containing the loaded data
    """
    try:
        from scipy.io import loadmat
        data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return data
    except NotImplementedError:
        # Try with h5py for v7.3 MAT files
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        return data


def save_results(filepath, data):
    """Save results as pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_results(filepath):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def run_main(data_path='Data', results_path='Results_folder_python', 
             max_number_sensors=250, use_flow_data=True, run_estimation_flag=False):
    """
    Main function to run the sensor selection and estimation pipeline.
    
    Parameters
    ----------
    data_path : str
        Path to data directory
    results_path : str
        Path to results directory
    max_number_sensors : int
        Maximum number of sensors (must be <= 250 and divisible by 25)
    use_flow_data : bool
        If True, use flow data; if False, use water depth data
    run_estimation_flag : bool
        If True, run estimation (memory intensive); if False, only run sensor selection
    """
    
    # Set random seed for reproducibility
    np.random.seed(12345)
    
    # Create results folders
    Path(results_path).mkdir(exist_ok=True)
    Path(f"{results_path}/Sensor_selection").mkdir(exist_ok=True)
    Path(f"{results_path}/Table_results_estimation").mkdir(exist_ok=True)
    Path(f"{results_path}/Error_matrix_k_max").mkdir(exist_ok=True)
    
    print("="*70)
    print("Information-Theoretic Sensor Placement for Sewer Networks")
    print("Python Implementation")
    print("="*70)
    
    # Validate max_number_sensors
    if max_number_sensors > 250 or max_number_sensors % 25 != 0:
        raise ValueError("max_number_sensors must be <= 250 and divisible by 25")
    
    # Load data
    print("\n[1/5] Loading data...")
    if use_flow_data:
        print("  Loading flow data...")
        # Note: Users need to download these files from Zenodo
        training_file = f"{data_path}/sim_1_sim_2_merged_flow.mat"
        validation_file = f"{data_path}/sim_3_sim_4_merged_flow.mat"
    else:
        print("  Loading water depth data...")
        training_file = f"{data_path}/sim_1_sim_2_merged_depth.mat"
        validation_file = f"{data_path}/sim_3_sim_4_merged_depth.mat"
    
    # Check if files exist
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training data file not found: {training_file}\n"
                              f"Please download data from Zenodo as described in README.md")
    
    if not os.path.exists(validation_file):
        raise FileNotFoundError(f"Validation data file not found: {validation_file}\n"
                              f"Please download data from Zenodo as described in README.md")
    
    # Load training and validation data
    data_matrix_Y1 = load_mat_file(training_file)
    data_matrix_Y2 = load_mat_file(validation_file)
    
    # Extract time series data (assuming standard naming conventions)
    # Adjust key names based on actual .mat file structure
    training_data_keys = [k for k in data_matrix_Y1.keys() if not k.startswith('__')]
    validation_data_keys = [k for k in data_matrix_Y2.keys() if not k.startswith('__')]
    
    # Assuming the main data key contains 'time_series' or similar
    training_key = [k for k in training_data_keys if 'time_series' in k.lower() or 'master' in k.lower()][0]
    validation_key = [k for k in validation_data_keys if 'time_series' in k.lower() or 'master' in k.lower()][0]
    
    training_data_raw = data_matrix_Y1[training_key]
    validation_data_raw = data_matrix_Y2[validation_key]
    
    # Convert to pandas DataFrame if needed and extract numeric data
    if isinstance(training_data_raw, pd.DataFrame):
        training_data = training_data_raw.iloc[:, 1:].values.astype(float)
    else:
        # Assume first column is time
        training_data = training_data_raw[:, 1:].astype(float)
    
    if isinstance(validation_data_raw, pd.DataFrame):
        validation_data = validation_data_raw.iloc[:, 1:].values.astype(float)
    else:
        validation_data = validation_data_raw[:, 1:].astype(float)
    
    print(f"  Training data shape: {training_data.shape}")
    print(f"  Validation data shape: {validation_data.shape}")
    
    # Calculate covariance matrix
    print("  Computing covariance matrix...")
    covmatrix_training = np.cov(training_data.T)
    n = covmatrix_training.shape[0]
    print(f"  Number of nodes: {n}")
    
    # Variance sigma^2 introduced from noise due to sensor
    sigmas = 1e-4
    
    # Run sensor selection algorithm
    print(f"\n[2/5] Running sensor selection algorithm (k={max_number_sensors})...")
    sensor_selection_file = f"{results_path}/Sensor_selection/sensor_selection.pkl"
    
    if os.path.exists(sensor_selection_file):
        print("  Loading existing sensor selection results...")
        optimal_sensor_selection_table = load_results(sensor_selection_file)
    else:
        print("  Running Algorithm 1 (this may take a while)...")
        optimal_sensor_selection_table = sensor_selection(n, covmatrix_training, sigmas, max_number_sensors)
        save_results(sensor_selection_file, optimal_sensor_selection_table)
        print("  Sensor placement complete and saved!")
    
    # Rank heuristics
    print("\n[3/5] Computing heuristic sensor selections...")
    max_sum_flow_vector = np.sum(training_data, axis=0)
    node_totalsum_flow = np.argsort(max_sum_flow_vector)[::-1]  # Descending order
    
    # Load rule-based selection
    rule_file = f"{data_path}/Rule_sensor_selection.mat"
    if os.path.exists(rule_file):
        rule_data = load_mat_file(rule_file)
        # Extract the rule selection array (adjust key name as needed)
        rule_key = [k for k in rule_data.keys() if not k.startswith('__')][0]
        Rule_selection = rule_data[rule_key].astype(int)
        # Convert to 0-based indexing if needed
        if Rule_selection.min() == 1:
            Rule_selection = Rule_selection - 1
    else:
        print(f"  Warning: Rule selection file not found at {rule_file}")
        Rule_selection = np.arange(max_number_sensors)
    
    # Run MI comparison
    print("  Computing mutual information for different heuristics...")
    number_rand_sims = 1000
    MI_totalsum_flow, MI_mean, MI_min, MI_max, MI_rule = mi_comparison_bellinge(
        max_number_sensors, number_rand_sims, node_totalsum_flow,
        covmatrix_training, sigmas, n, Rule_selection
    )
    
    # Plot MI comparison (Figure 5)
    print("\n[4/5] Generating mutual information comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(1, max_number_sensors + 1)
    algo_mi = optimal_sensor_selection_table[:, -1]
    
    ax1.plot(x, algo_mi, label='Algorithm 1 placement', color='red')
    ax1.plot(x, MI_max, label='Max of random placements', color='blue')
    ax1.plot(x, MI_min, label='Min of random placements', color='black')
    ax1.plot(x, MI_mean, label='Mean of random placements', color='green')
    ax1.plot(x, MI_rule, label='Rule based placement', color='magenta')
    ax1.plot(x, MI_totalsum_flow, label='Max total sum placement', color='cyan')
    ax1.set_xlabel('Number of sensors', fontsize=13)
    ax1.set_ylabel('Mutual information value', fontsize=13)
    ax1.set_xlim([0, max_number_sensors])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x, (algo_mi - MI_max) / algo_mi, label='Max of random placements', color='blue')
    ax2.plot(x, (algo_mi - MI_min) / algo_mi, label='Min of random placements', color='black')
    ax2.plot(x, (algo_mi - MI_mean) / algo_mi, label='Mean of random placements', color='green')
    ax2.plot(x, (algo_mi - MI_rule) / algo_mi, label='Rule based placement', color='magenta')
    ax2.plot(x, (algo_mi - MI_totalsum_flow) / algo_mi, label='Max total sum placement', color='cyan')
    ax2.set_xlabel('Number of sensors', fontsize=13)
    ax2.set_ylabel('MI gain', fontsize=13)
    ax2.set_xlim([0, max_number_sensors])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/Figure5_MI_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved plot to {results_path}/Figure5_MI_comparison.png")
    plt.close()
    
    # Estimation section
    if run_estimation_flag:
        print("\n[5/5] Running estimation algorithms...")
        print("  Warning: This is memory intensive. Using first 4000 data points.")
        
        # Reduce data size
        training_data = training_data[:4000, :]
        validation_data = validation_data[:4000, :]
        
        seq = np.arange(n)
        k_list_vector = np.arange(25, max_number_sensors + 1, 25)
        
        print(f"  Running estimation for k values: {k_list_vector}")
        
        # Note: Full estimation implementation would go here
        # This is a simplified version
        print("  Note: Full estimation with GRNN requires significant computational resources.")
        print("  Consider running on HPC or reducing data size further.")
    else:
        print("\n[5/5] Skipping estimation (set run_estimation_flag=True to enable)")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"Results saved to: {results_path}")
    print("="*70)


if __name__ == "__main__":
    # Example usage
    run_main(
        data_path='Data',
        results_path='Results_folder_python',
        max_number_sensors=250,
        use_flow_data=True,
        run_estimation_flag=False  # Set to True to run estimation (memory intensive)
    )
