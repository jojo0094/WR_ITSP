"""
Copyright (c) 2024, George Crowley (gcrowley1@sheffield.ac.uk)
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Error calculation functions
"""

import numpy as np
from typing import Tuple


def error_calculations(estimated_matrix_validation_un_observed: np.ndarray,
                      validation_un_observed: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate error matrix and normalized mean square error (NMSE).
    
    Parameters
    ----------
    estimated_matrix_validation_un_observed : np.ndarray
        Estimated values for unobserved nodes
    validation_un_observed : np.ndarray
        True values for unobserved nodes
        
    Returns
    -------
    error_matrix : np.ndarray
        Error matrix (estimated - true)
    NMSE : float
        Normalized mean square error
    """
    
    # Calculate error matrix
    error_matrix = estimated_matrix_validation_un_observed - validation_un_observed
    
    # Calculate square error for each entry
    error_matrix_sq = error_matrix ** 2
    
    # Calculate energy of true un_observed data from the validation set
    energy_matrix = validation_un_observed ** 2
    energy = np.sum(energy_matrix)
    
    # Calculate normalized total square error of all estimated nodes and readings
    NMSE = np.sum(error_matrix_sq) / energy
    
    return error_matrix, NMSE
