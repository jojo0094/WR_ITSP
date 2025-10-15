# Quick Start Guide - Python Implementation

Get up and running with the Python implementation in 5 minutes!

## Option 1: Try with Synthetic Data (No Downloads Required)

This is the fastest way to test the implementation:

```bash
# 1. Install dependencies
pip install numpy scipy pandas matplotlib scikit-learn

# 2. Run the example
cd WR_ITSP
python -m python_code --example
```

This will:
- Generate synthetic sensor network data
- Run the sensor selection algorithm
- Create comparison plots
- Show you the results in `example_results.png`

**Time:** ~2-5 minutes depending on your machine

## Option 2: Run with Real Data

To replicate the paper results with the Bellinge network:

### Step 1: Download Data

Download the datasets from Zenodo (required, ~2GB total):
- Flow data: https://zenodo.org/doi/10.5281/zenodo.11442174
- Water depth (optional): https://zenodo.org/doi/10.5281/zenodo.11636092

Place the `.mat` files in the `Data/` folder:
```
Data/
├── sim_1_sim_2_merged_flow.mat
├── sim_3_sim_4_merged_flow.mat
└── Rule_sensor_selection.mat
```

### Step 2: Install and Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run sensor selection (no estimation, fast)
python -m python_code --max-sensors 250
```

**Time:** ~10-30 minutes for sensor selection

**Output:**
- `Results_folder_python/Figure5_MI_comparison.png` - Comparison plot
- `Results_folder_python/Sensor_selection/sensor_selection.pkl` - Results

### Step 3 (Optional): Run Estimation

⚠️ Warning: This requires significant RAM (16-32 GB)

```bash
python -m python_code --max-sensors 250 --run-estimation
```

## Testing Your Installation

Run the basic test suite:

```bash
python test_basic.py
```

This will test all modules and should complete in ~1 minute.

## Command-Line Options

```bash
# Show help
python -m python_code --help

# Use water depth data instead of flow
python -m python_code --use-depth --max-sensors 100

# Change output directory
python -m python_code --results-path my_results --max-sensors 250

# Run with custom data location
python -m python_code --data-path /path/to/data --max-sensors 250
```

## Using in Python Scripts

```python
from python_code.sensor_selection import sensor_selection
import numpy as np

# Your time series data (rows=time, cols=nodes)
data = np.loadtxt('my_network_data.csv', delimiter=',')

# Compute covariance
covmatrix = np.cov(data.T)
n = covmatrix.shape[0]

# Run algorithm
results = sensor_selection(
    n=n,
    covmatrix=covmatrix,
    sigmas=1e-4,  # sensor noise
    choice_number_sensors=50
)

# Get sensors for k=25
sensors = results[24, 4:29].astype(int)
print(f"Selected sensors: {sensors}")
```

## Common Issues

### "File not found" error
**Solution:** Download the data files from Zenodo (see Step 1 above)

### Out of memory
**Solution:** Run without estimation first:
```bash
python -m python_code --max-sensors 100  # Start with fewer sensors
```

### Import errors
**Solution:** Make sure you're in the correct directory:
```bash
cd WR_ITSP
python -m python_code --example
```

## Next Steps

- Read the full [PYTHON_README.md](PYTHON_README.md) for detailed documentation
- Check the [README.md](README.md) for paper background and theory
- Explore `python_code/example_simple.py` for code examples
- Run `python test_basic.py` to verify your installation

## Performance Expectations

For the Bellinge network (n=565 nodes):

| Task | Time | Memory |
|------|------|--------|
| Sensor selection (k=250) | 10-30 min | 2-4 GB |
| MI comparison | 5-10 min | 2-4 GB |
| GLM estimation (4000 pts) | 2-5 min | 4-8 GB |
| GRNN estimation (4000 pts) | 10-20 min | 16-32 GB |

For testing, start with smaller networks (n~50-100) or fewer sensors (k~50-100).

## Need Help?

1. Check the [PYTHON_README.md](PYTHON_README.md) troubleshooting section
2. Open an issue on GitHub
3. Contact: gcrowley1@sheffield.ac.uk
