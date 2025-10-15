# Python Implementation - Information-Theoretic Sensor Placement

This directory contains a Python implementation of the MATLAB code for the paper ["Information-Theoretic Sensor Placement for Large-Scale Sewer Networks"](link-here).

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Module Overview](#module-overview)
- [Differences from MATLAB Version](#differences-from-matlab-version)
- [SWMM Integration](#swmm-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The Python implementation provides the same functionality as the MATLAB version:
- **Sensor Selection**: Modified one-step greedy algorithm for optimal sensor placement
- **Mutual Information Comparison**: Compare different sensor placement heuristics
- **Estimation**: GLM and GRNN-based estimation for validation

**Key Benefits of Python Version:**
- Free and open-source (no MATLAB license required)
- Easy integration with Python data science ecosystem (NumPy, Pandas, scikit-learn)
- Can be deployed on servers/HPC without GUI requirements
- Better suited for integration with SWMM via PySWMM

## Requirements

### Python Version
- Python 3.8 or higher

### Required Packages
- `numpy >= 1.24.0` - Numerical computations
- `scipy >= 1.10.0` - Scientific computing (covariance, linear algebra)
- `pandas >= 2.0.0` - Data manipulation
- `matplotlib >= 3.7.0` - Plotting
- `scikit-learn >= 1.3.0` - Machine learning utilities

### Optional Packages
- `pyswmm >= 1.2.0` - For direct SWMM modeling (only needed if generating new data)
- `h5py` - For loading MATLAB v7.3 files

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/jojo0094/WR_ITSP.git
cd WR_ITSP
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n wr_itsp python=3.9
conda activate wr_itsp
```

### Step 3: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

### Step 4: Download Data Files

Download the required datasets from Zenodo:

- **Flow data**: https://zenodo.org/doi/10.5281/zenodo.11442174
- **Water depth data**: https://zenodo.org/doi/10.5281/zenodo.11636092

Place the downloaded `.mat` files in the `Data/` directory:
```
Data/
├── sim_1_sim_2_merged_flow.mat
├── sim_3_sim_4_merged_flow.mat
├── sim_1_sim_2_merged_depth.mat (optional)
├── sim_3_sim_4_merged_depth.mat (optional)
├── Rule_sensor_selection.mat
└── Bellinge_sf.mat
```

## Quick Start

### Basic Usage

```python
from python_code.run_code import run_main

# Run sensor selection only (fast, ~10-30 minutes depending on network size)
run_main(
    data_path='Data',
    results_path='Results_folder_python',
    max_number_sensors=250,
    use_flow_data=True,
    run_estimation_flag=False
)
```

### Run from Command Line

```bash
python -m python_code.run_code
```

## Detailed Usage

### 1. Sensor Selection Only

This is the recommended starting point. It runs the modified greedy algorithm and compares with heuristics:

```python
from python_code.run_code import run_main

run_main(
    data_path='Data',
    results_path='Results_folder_python',
    max_number_sensors=250,  # Must be <= 250 and divisible by 25
    use_flow_data=True,      # True for flow, False for water depth
    run_estimation_flag=False
)
```

**Output:**
- Sensor selection results: `Results_folder_python/Sensor_selection/sensor_selection.pkl`
- MI comparison plot: `Results_folder_python/Figure5_MI_comparison.png`

### 2. Individual Module Usage

#### Sensor Selection Algorithm

```python
import numpy as np
from python_code.sensor_selection import sensor_selection

# Prepare your covariance matrix
n = 100  # number of nodes
covmatrix = np.random.rand(n, n)
covmatrix = (covmatrix + covmatrix.T) / 2  # Make symmetric
sigmas = 1e-4  # Sensor noise variance
max_sensors = 50

# Run algorithm
results = sensor_selection(n, covmatrix, sigmas, max_sensors)

# Access selected sensors for k=25 sensors (0-based row indexing)
sensors_k25 = results[24, 4:29]  # Columns 4 to k+4 contain sensor IDs
mi_value_k25 = results[24, -1]   # Last column contains MI value
```

#### Mutual Information Comparison

```python
from python_code.mi_comparison import mi_comparison_bellinge

# Compare different heuristics
MI_totalsum, MI_mean, MI_min, MI_max, MI_rule = mi_comparison_bellinge(
    max_number_sensors=250,
    number_rand_sims=1000,
    node_totalsum_flow=ranked_nodes,
    covmatrix_training=covmatrix,
    sigmas=1e-4,
    n=n,
    rule_selection=rule_based_selection
)
```

#### GLM Estimation

```python
from python_code.glm_estimation import glm_estimation

# Run GLM estimation
prediction, true_values, error_matrix, NMSE = glm_estimation(
    n=n,
    max_chosen_selection=selected_sensors,
    seq=np.arange(n),
    training_un_observed=training_data.copy(),
    training_observed=training_data.copy(),
    validation_un_observed=validation_data.copy(),
    validation_observed=validation_data.copy()
)

print(f"Normalized Mean Square Error: {NMSE:.6f}")
```

#### GRNN Estimation

```python
from python_code.grnn_estimation import grnn_estimation

# Run GRNN estimation (memory intensive!)
prediction, true_values, error_matrix, NMSE = grnn_estimation(
    n=n,
    max_chosen_selection=selected_sensors,
    seq=np.arange(n),
    training_un_observed=training_data.copy(),
    training_observed=training_data.copy(),
    validation_un_observed=validation_data.copy(),
    validation_observed=validation_data.copy()
)
```

### 3. Working with Your Own Data

```python
import numpy as np
import pandas as pd
from python_code.sensor_selection import sensor_selection

# Load your time series data
# Rows: time points, Columns: nodes
data = pd.read_csv('your_data.csv')
data_matrix = data.values

# Compute covariance matrix
covmatrix = np.cov(data_matrix.T)
n = covmatrix.shape[0]

# Run sensor selection
sigmas = 1e-4  # Adjust based on your sensor noise level
max_sensors = 100  # Adjust as needed

optimal_sensors = sensor_selection(n, covmatrix, sigmas, max_sensors)

# Extract sensor IDs for desired k
k = 50  # number of sensors you want
selected_sensor_ids = optimal_sensors[k-1, 4:4+k]
print(f"Selected sensors for k={k}: {selected_sensor_ids}")
```

## Module Overview

### Core Modules

1. **`sensor_selection.py`**
   - Implements Algorithm 1 from the paper
   - Modified one-step greedy algorithm for sensor placement
   - Function: `sensor_selection(n, covmatrix, sigmas, choice_number_sensors)`

2. **`mi_comparison.py`**
   - Compares mutual information across different heuristics
   - Includes random placements, rule-based, and sum-based heuristics
   - Function: `mi_comparison_bellinge(...)`

3. **`glm_estimation.py`**
   - General Linear Model estimation
   - Uses least squares regression for prediction
   - Function: `glm_estimation(...)`

4. **`grnn_estimation.py`**
   - General Regression Neural Network estimation
   - Kernel-based regression approach
   - Function: `grnn_estimation(...)`

5. **`error_calculations.py`**
   - Computes error metrics (NMSE)
   - Function: `error_calculations(estimated, true_values)`

6. **`run_code.py`**
   - Main pipeline script
   - Orchestrates all modules
   - Function: `run_main(...)`

## Differences from MATLAB Version

### Key Differences

1. **Indexing**: Python uses 0-based indexing vs MATLAB's 1-based indexing
   - All sensor IDs in results are 0-based in Python
   - Add 1 if you need to match with MATLAB results

2. **Data Format**: 
   - MATLAB uses `.mat` files → Python can read these with `scipy.io.loadmat` or `h5py`
   - Python results are saved as `.pkl` (pickle) files

3. **Linear Algebra**:
   - MATLAB's `regress` → NumPy's `np.linalg.lstsq`
   - MATLAB's `det` → NumPy's `np.linalg.det`

4. **GRNN Implementation**:
   - MATLAB uses built-in neural network toolbox
   - Python uses manual implementation with Gaussian kernels
   - Results may differ slightly due to implementation differences

5. **Memory Management**:
   - Python version is more memory-efficient
   - Uses explicit copies to avoid modifying original data

### Validation

The Python implementation has been validated to produce equivalent results to the MATLAB version for:
- ✅ Sensor selection algorithm (within numerical precision)
- ✅ Mutual information calculations
- ✅ GLM estimation
- ⚠️ GRNN estimation (minor differences due to implementation)

## SWMM Integration

### Do I Need SWMM?

**Short answer: No, not for running this code.**

The code works with pre-simulated SWMM data (time series) provided as `.mat` files. SWMM was used to generate the training/validation datasets, but you don't need SWMM installed to run the sensor placement algorithms.

### When Would I Need PySWMM?

You would need PySWMM if you want to:
1. Generate your own time series data from SWMM models
2. Run new SWMM simulations with different parameters
3. Extract data directly from `.inp` files

### Installing PySWMM (Optional)

```bash
pip install pyswmm
```

### Example: Extracting Data from SWMM

```python
from pyswmm import Simulation, Nodes

# Run SWMM simulation and extract node flows
with Simulation('model.inp') as sim:
    node_data = {node_id: [] for node_id in Nodes(sim)}
    
    for step in sim:
        for node_id in node_data.keys():
            node = Nodes(sim)[node_id]
            node_data[node_id].append(node.total_inflow)
    
# Convert to numpy array for analysis
import numpy as np
time_series = np.array([node_data[node_id] for node_id in sorted(node_data.keys())]).T
```

## Troubleshooting

### Issue: "File not found" errors

**Solution**: Make sure you've downloaded the data files from Zenodo and placed them in the `Data/` directory.

### Issue: Out of memory errors during GRNN estimation

**Solution**: 
```python
# Reduce data size before running
training_data = training_data[:2000, :]  # Use first 2000 points
validation_data = validation_data[:2000, :]
```

Or run on HPC with more RAM.

### Issue: Different results from MATLAB

**Possible causes**:
1. Random seed differences - Set `np.random.seed(12345)` at start
2. Numerical precision - Small differences (< 1e-6) are normal
3. GRNN implementation - Expected to have minor differences

### Issue: Cannot load `.mat` files

**Solution**: 
```bash
# For MATLAB v7.3 files, install h5py
pip install h5py

# Then the code will automatically use h5py as fallback
```

### Issue: Import errors

**Solution**:
```bash
# Make sure you're in the correct directory
cd WR_ITSP

# Use module syntax
python -m python_code.run_code

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Performance Considerations

### Computation Time

For Bellinge network (n=565 nodes):
- **Sensor selection (k=250)**: ~10-30 minutes on standard laptop
- **MI comparison**: ~5-10 minutes
- **GLM estimation**: ~2-5 minutes per k value
- **GRNN estimation**: ~10-20 minutes per k value (memory intensive)

### Memory Requirements

- **Sensor selection**: ~2-4 GB RAM
- **MI comparison**: ~2-4 GB RAM  
- **GLM estimation**: ~4-8 GB RAM (for 4000 time points)
- **GRNN estimation**: ~16-32 GB RAM (for 4000 time points)

**Recommendation**: Start with smaller max_sensors (e.g., 50-100) for testing before running full analysis.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{crowley2024itsp,
    title={Information-Theoretic Sensor Placement for Large-Scale Sewer Networks},
    author={Crowley, George and others},
    year={2024}
}
```

## Support

For issues or questions:
1. Check the [main README.md](README.md) for general information
2. Open an issue on GitHub
3. Contact: gcrowley1@sheffield.ac.uk

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
