"""
Copyright (c) 2024, George Crowley (gcrowley1@sheffield.ac.uk)
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Sensor selection algorithm implementation
"""

import numpy as np
from typing import Tuple


def sensor_selection(n: int, covmatrix: np.ndarray, sigmas: float, 
                     choice_number_sensors: int) -> np.ndarray:
    """
    Modified one-step greedy algorithm for sensor placement.
    
    Parameters
    ----------
    n : int
        Number of total nodes in the network
    covmatrix : np.ndarray
        Covariance matrix of the training data
    sigmas : float
        Variance of additive white Gaussian noise from sensors
    choice_number_sensors : int
        Maximum number of sensors to place
        
    Returns
    -------
    optimal_sensor_selection_table : np.ndarray
        Table with optimal sensor selections and mutual information values
    """
    
    # Add noise to covariance matrix diagonal
    covmatrix = covmatrix.copy()
    for i in range(n):
        covmatrix[i, i] += sigmas
    
    vectorsequence = np.arange(n)
    
    # Simplifying notation and defining relevant variables
    k = choice_number_sensors
    sensorselection = np.zeros((k, k))
    dataset_sensorselection_mutualinformation = np.zeros((k**2, k+1))
    placeholder = 0  # Placeholder for table positioning
    
    # Matrix normalization for determinant calculations
    # Divide each element of the matrix by the mean of positive values
    chosenval = np.mean(covmatrix[covmatrix > 0])
    covmatrix_adjusted = covmatrix / chosenval
    
    # Store mutual information for all starting nodes
    mutualinformationtable = np.zeros((k+1, n+4))
    
    print("Starting sensor selection algorithm...")
    
    # Main loop over all possible initial nodes
    for node in range(n):
        initialnode = node
        nextiterationplacement = np.array([initialnode])
        counter = 2  # Counter represents number of sensors
        
        if (node + 1) % 50 == 0:
            print(f"Progress: {node + 1}/{n} nodes processed")
        
        # Build up sensor placement until we reach k sensors
        while len(nextiterationplacement) < k:
            choicenodes = np.setdiff1d(vectorsequence, nextiterationplacement)
            
            collection_nodes = []
            mutualinformationvalues = []
            
            for choice_idx, choice in enumerate(choicenodes):
                seqchoicenodes = np.append(nextiterationplacement, choice)
                seqchoicenodes = np.sort(seqchoicenodes)
                collection_nodes.append(seqchoicenodes)
                
                # Extract submatrix for determinant calculation
                detmatrix = covmatrix_adjusted[np.ix_(seqchoicenodes, seqchoicenodes)]
                
                # Calculate mutual information
                detval = np.linalg.det(detmatrix)
                mi = -counter * 0.5 * np.log(sigmas) + 0.5 * (np.log(detval) - counter * np.log(1/chosenval))
                mutualinformationvalues.append(mi)
            
            # Find max MI selection
            mutualinformationvalues = np.array(mutualinformationvalues)
            idxmax = np.argmax(mutualinformationvalues)
            valmax = mutualinformationvalues[idxmax]
            nextiterationplacement = collection_nodes[idxmax]
            
            # Store results
            sensorselection[counter-1, :len(nextiterationplacement)] = nextiterationplacement
            if counter == 2:
                sensorselection_mutual_information = [valmax]
            else:
                sensorselection_mutual_information.append(valmax)
            
            counter += 1
        
        # Store first node in sensor selection matrix
        sensorselection[0, 0] = initialnode
        
        # Save placements and MI scores in general matrix
        dataset_sensorselection_mutualinformation[placeholder:placeholder+k, :k] = sensorselection
        dataset_sensorselection_mutualinformation[placeholder:placeholder+k, k] = sensorselection_mutual_information
        
        # Update table placement
        placeholder += k
        
        # Update mutual information table
        mutualinformationtable[1:len(sensorselection_mutual_information)+1, node+1] = sensorselection_mutual_information
        
        # Reset sensor selection for next iteration
        sensorselection = np.zeros((k, k))
    
    print("Sensor selection algorithm complete!")
    
    # Manually input values for first node MI
    for i in range(n):
        mutualinformationtable[1, i+1] = 0.5 * np.log(1 / sigmas) + 0.5 * (np.log(np.linalg.det(covmatrix_adjusted[i:i+1, i:i+1])) - np.log(1/chosenval))
    
    # Create table structure
    mutualinformationtable[0, 1:n+1] = np.arange(n)
    mutualinformationtable[1:k+1, 0] = np.arange(1, k+1)
    
    # Find best starting node for each k
    for i in range(1, k+1):
        valmax_MI_table = np.max(mutualinformationtable[i, 1:n+1])
        idxmax_MI_table = np.argmax(mutualinformationtable[i, 1:n+1])
        mutualinformationtable[i, n+2] = valmax_MI_table
        mutualinformationtable[i, n+3] = idxmax_MI_table
    
    # Create optimal sensor selection table
    optimal_sensor_selection_table = np.zeros((k, k+5))
    optimal_sensor_selection_table[:, 0] = np.arange(1, k+1)  # Label number of sensors
    
    # Optimal starting sensor - place in table
    optimal_sensor_selection_table[:, 2] = mutualinformationtable[1:k+1, n+3]
    
    for i in range(k):
        start_idx = int(optimal_sensor_selection_table[i, 2]) * k + i
        optimal_sensor_selection_table[i, 4:k+4] = dataset_sensorselection_mutualinformation[start_idx, :k]
    
    optimal_sensor_selection_table[:, k+4] = mutualinformationtable[1:k+1, n+2]
    
    return optimal_sensor_selection_table
