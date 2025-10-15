"""
Copyright (c) 2024, George Crowley (gcrowley1@sheffield.ac.uk)
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

General Linear Model (GLM) estimation functions
"""

import numpy as np
from typing import Tuple
from .error_calculations import error_calculations


def glm_estimation(n: int, max_chosen_selection: np.ndarray, seq: np.ndarray,
                  training_un_observed: np.ndarray, training_observed: np.ndarray,
                  validation_un_observed: np.ndarray, validation_observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    General Linear Model estimation for sensor placement validation.
    
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
    
    # Create data matrix with intercept (beta_0)
    X_data_matrix = np.ones((training_observed.shape[0], training_observed.shape[1] + 1))
    X_data_matrix[:, 1:] = training_observed
    
    X_data_matrix_validation = np.ones((validation_observed.shape[0], validation_observed.shape[1] + 1))
    X_data_matrix_validation[:, 1:] = validation_observed
    
    # Calculate beta coefficients from training data and make predictions
    prediction_matrix = np.zeros((validation_un_observed.shape[0], validation_un_observed.shape[1]))
    
    for i in range(validation_un_observed.shape[1]):
        # Use least squares regression to find coefficients
        beta_coefficients, _, _, _ = np.linalg.lstsq(X_data_matrix, training_un_observed[:, i], rcond=None)
        prediction_matrix[:, i] = X_data_matrix_validation @ beta_coefficients
    
    # Call error function
    error_matrix, NMSE = error_calculations(prediction_matrix, validation_un_observed)
    
    return prediction_matrix, validation_un_observed, error_matrix, NMSE
