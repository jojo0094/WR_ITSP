"""
Copyright (c) 2024, George Crowley (gcrowley1@sheffield.ac.uk)
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

General Regression Neural Network (GRNN) estimation functions
"""

import numpy as np
from typing import Tuple
from .error_calculations import error_calculations


def grnn_estimation(n: int, max_chosen_selection: np.ndarray, seq: np.ndarray,
                   training_un_observed: np.ndarray, training_observed: np.ndarray,
                   validation_un_observed: np.ndarray, validation_observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    General Regression Neural Network estimation for sensor placement validation.
    
    Parameters
    ----------
    n : int
        Total number of nodes
    max_chosen_selection : np.ndarray
        Selected sensor locations
    seq : np.ndarray
        Sequence of all node indices
    training_un_observed : np.ndarray
        Training data (to be modified for unobserved nodes)
    training_observed : np.ndarray
        Training data (to be modified for observed nodes)
    validation_un_observed : np.ndarray
        Validation data (to be modified for unobserved nodes)
    validation_observed : np.ndarray
        Validation data (to be modified for observed nodes)
        
    Returns
    -------
    prediction_matrix : np.ndarray
        Predicted values for unobserved nodes
    validation_un_observed : np.ndarray
        True values for unobserved nodes
    error_matrix : np.ndarray
        Error matrix
    NMSE : float
        Normalized mean square error
    """
    
    # Make copies to avoid modifying original arrays
    training_un_observed = training_un_observed.copy()
    training_observed = training_observed.copy()
    validation_un_observed = validation_un_observed.copy()
    validation_observed = validation_observed.copy()
    
    # Convert selection to 0-based indices
    max_chosen_selection = max_chosen_selection.astype(int)
    
    # Set chosen sensors columns to be deleted from unobserved
    training_un_observed = np.delete(training_un_observed, max_chosen_selection, axis=1)
    validation_un_observed = np.delete(validation_un_observed, max_chosen_selection, axis=1)
    
    # Set un-selected sensors columns to be deleted from observed
    unselected = np.setdiff1d(seq, max_chosen_selection)
    training_observed = np.delete(training_observed, unselected, axis=1)
    validation_observed = np.delete(validation_observed, unselected, axis=1)
    
    # Normalize the observed data
    training_observed_normalized = training_observed.copy()
    validation_observed_normalized = validation_observed.copy()
    
    for i in range(training_observed_normalized.shape[1]):
        stdev = np.std(training_observed[:, i])
        
        # Check because some nodes can have 0's for all time realizations in the simulated SWMM data
        if stdev == 0:
            stdev = 1
        
        mean_val = np.mean(training_observed[:, i])
        training_observed_normalized[:, i] = (training_observed_normalized[:, i] - mean_val) / stdev
        validation_observed_normalized[:, i] = (validation_observed_normalized[:, i] - mean_val) / stdev
    
    # GRNN parameters
    spread = 0.5
    
    # Transpose for network format (features x samples)
    p = training_observed_normalized.T
    t = training_un_observed.T
    
    # Get dimensions
    R, Q = p.shape  # R = number of inputs, Q = number of samples
    S = t.shape[0]  # S = number of outputs
    
    # Implement GRNN algorithm
    # For each validation point, compute distances to all training points
    validation_observed_normalized_T = validation_observed_normalized.T
    
    # Calculate squared distances (using broadcasting)
    # distances shape: (Q_train, Q_val) where Q_train = training samples, Q_val = validation samples
    prediction_matrix = np.zeros((validation_observed.shape[0], training_un_observed.shape[1]))
    
    for val_idx in range(validation_observed.shape[0]):
        # Calculate Euclidean distances from this validation point to all training points
        val_point = validation_observed_normalized[val_idx, :]
        distances = np.sum((p.T - val_point)**2, axis=1)  # shape: (Q,)
        
        # Calculate Gaussian kernel weights
        # Using radial basis with spread parameter
        weights = np.exp(-distances / (2 * spread**2))
        
        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            # If all weights are zero, use uniform weights
            weights = np.ones(Q) / Q
        
        # Calculate weighted output
        prediction_matrix[val_idx, :] = t @ weights
    
    # Call error function
    error_matrix, NMSE = error_calculations(prediction_matrix, validation_un_observed)
    
    return prediction_matrix, validation_un_observed, error_matrix, NMSE
