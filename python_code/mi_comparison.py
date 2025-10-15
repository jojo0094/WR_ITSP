"""
Copyright (c) 2024, George Crowley (gcrowley1@sheffield.ac.uk)
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Mutual Information comparison functions
"""

import numpy as np
from typing import Tuple


def mi_comparison_bellinge(max_number_sensors: int, number_rand_sims: int,
                           node_totalsum_flow: np.ndarray, covmatrix_training: np.ndarray,
                           sigmas: float, n: int, rule_selection: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare mutual information between different sensor placement heuristics.
    
    Parameters
    ----------
    max_number_sensors : int
        Maximum number of sensors to evaluate
    number_rand_sims : int
        Number of random simulations to run
    node_totalsum_flow : np.ndarray
        Nodes ranked by total sum of flow
    covmatrix_training : np.ndarray
        Covariance matrix of training data
    sigmas : float
        Sensor noise variance
    n : int
        Total number of nodes
    rule_selection : np.ndarray
        Rule-based sensor selection
        
    Returns
    -------
    MI_totalsum_flow : np.ndarray
        MI values for total sum heuristic
    MI_mean : np.ndarray
        Mean MI values for random placements
    MI_min : np.ndarray
        Minimum MI values for random placements
    MI_max : np.ndarray
        Maximum MI values for random placements
    MI_rule : np.ndarray
        MI values for rule-based placement
    """
    
    # Pre-define vectors of mutual information for each heuristic
    MI_mean = np.zeros(max_number_sensors)
    MI_min = np.zeros(max_number_sensors)
    MI_max = np.zeros(max_number_sensors)
    MI_totalsum_flow = np.zeros(max_number_sensors)
    MI_rule = np.zeros(max_number_sensors)
    
    # Add noise to covariance matrix
    covmatrix_training = covmatrix_training.copy()
    for i in range(n):
        covmatrix_training[i, i] += sigmas
    
    # Matrix normalization for determinant calculations
    minval = np.mean(covmatrix_training[covmatrix_training > 0])
    covmatrix_training = covmatrix_training / minval
    
    print("Computing mutual information for heuristics...")
    
    # Run first simulation for the Rule based and total sum mutual Information
    for i in range(max_number_sensors):
        # Adjust indices to 0-based (Python) from 1-based (MATLAB)
        det_matrix_totalsum_flow = np.linalg.det(
            covmatrix_training[np.ix_(node_totalsum_flow[:i+1], node_totalsum_flow[:i+1])]
        )
        det_matrix_rule = np.linalg.det(
            covmatrix_training[np.ix_(rule_selection[:i+1], rule_selection[:i+1])]
        )
        
        MI_totalsum_flow[i] = -(i+1) * 0.5 * np.log(sigmas) + 0.5 * (np.log(det_matrix_totalsum_flow) - (i+1) * np.log(1/minval))
        MI_rule[i] = -(i+1) * 0.5 * np.log(sigmas) + 0.5 * (np.log(det_matrix_rule) - (i+1) * np.log(1/minval))
        
        if (i + 1) % 50 == 0:
            print(f"Heuristic MI calculation: {i + 1}/{max_number_sensors}")
    
    print(f"Running {number_rand_sims} random simulations...")
    
    # Run random simulations for sensor selection and evaluate Mutual information
    for j in range(max_number_sensors):
        number_sensors = j + 1
        perm_matrix_rand = np.zeros((number_rand_sims, number_sensors + 1))
        
        for i in range(number_rand_sims):
            # Generate random permutation and sort
            random_selection = np.sort(np.random.choice(n, number_sensors, replace=False))
            perm_matrix_rand[i, :number_sensors] = random_selection
            
            # Calculate determinant of chosen selection
            det_rand_matrix = np.linalg.det(
                covmatrix_training[np.ix_(random_selection, random_selection)]
            )
            
            # Calculate MI of selection chosen randomly
            perm_matrix_rand[i, number_sensors] = -number_sensors * 0.5 * np.log(sigmas) + 0.5 * (np.log(det_rand_matrix) - number_sensors * np.log(1/minval))
        
        MI_mean[j] = np.mean(perm_matrix_rand[:, number_sensors])
        MI_min[j] = np.min(perm_matrix_rand[:, number_sensors])
        MI_max[j] = np.max(perm_matrix_rand[:, number_sensors])
        
        if (j + 1) % 50 == 0:
            print(f"Random MI calculation: {j + 1}/{max_number_sensors}")
    
    print("MI comparison complete!")
    
    return MI_totalsum_flow, MI_mean, MI_min, MI_max, MI_rule
