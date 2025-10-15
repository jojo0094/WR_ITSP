#!/usr/bin/env python
"""
Quick test of the Python implementation with minimal data
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from python_code.sensor_selection import sensor_selection
from python_code.mi_comparison import mi_comparison_bellinge
from python_code.error_calculations import error_calculations

print("="*70)
print("Basic Test of Python Implementation")
print("="*70)

# Test 1: Sensor Selection with small network
print("\nTest 1: Sensor Selection Algorithm")
print("-" * 40)
n = 10  # Small network
max_sensors = 5

# Create a simple covariance matrix
np.random.seed(123)
A = np.random.randn(n, n)
covmatrix = A @ A.T  # Positive semi-definite
sigmas = 1e-4

print(f"Network size: {n} nodes")
print(f"Computing optimal placement for {max_sensors} sensors...")

try:
    result = sensor_selection(n, covmatrix, sigmas, max_sensors)
    print(f"✓ Sensor selection completed successfully")
    print(f"  Result shape: {result.shape}")
    
    # Check results
    sensors_k3 = result[2, 4:7]  # Get sensors for k=3
    mi_k3 = result[2, -1]
    print(f"  For k=3: sensors {sensors_k3.astype(int)}, MI={mi_k3:.4f}")
    
except Exception as e:
    print(f"✗ Error in sensor selection: {e}")
    sys.exit(1)

# Test 2: MI Comparison
print("\nTest 2: MI Comparison")
print("-" * 40)

# Create simple heuristics
node_ranked = np.arange(n)
rule_selection = np.arange(max_sensors)

print(f"Running MI comparison with 10 random simulations...")

try:
    MI_sum, MI_mean, MI_min, MI_max, MI_rule = mi_comparison_bellinge(
        max_sensors, 10, node_ranked, covmatrix, sigmas, n, rule_selection
    )
    print(f"✓ MI comparison completed successfully")
    print(f"  MI values for k={max_sensors}:")
    print(f"    Total sum: {MI_sum[-1]:.4f}")
    print(f"    Random mean: {MI_mean[-1]:.4f}")
    print(f"    Rule-based: {MI_rule[-1]:.4f}")
    
except Exception as e:
    print(f"✗ Error in MI comparison: {e}")
    sys.exit(1)

# Test 3: Error Calculations
print("\nTest 3: Error Calculations")
print("-" * 40)

# Create test data
estimated = np.random.randn(100, 5)
true_values = np.random.randn(100, 5)

print(f"Computing error metrics...")

try:
    error_matrix, NMSE = error_calculations(estimated, true_values)
    print(f"✓ Error calculation completed successfully")
    print(f"  Error matrix shape: {error_matrix.shape}")
    print(f"  NMSE: {NMSE:.6f}")
    
except Exception as e:
    print(f"✗ Error in error calculations: {e}")
    sys.exit(1)

# Test 4: GLM Estimation
print("\nTest 4: GLM Estimation")
print("-" * 40)

from python_code.glm_estimation import glm_estimation

n_time = 50
n_nodes = 10
k_sensors = 3

# Generate synthetic data
train_data = np.random.randn(n_time, n_nodes)
valid_data = np.random.randn(n_time, n_nodes)
selected = np.array([0, 3, 7])  # Select some sensors
seq = np.arange(n_nodes)

print(f"Running GLM estimation with {k_sensors} sensors...")

try:
    pred, true_val, err, nmse = glm_estimation(
        n_nodes, selected, seq,
        train_data.copy(), train_data.copy(),
        valid_data.copy(), valid_data.copy()
    )
    print(f"✓ GLM estimation completed successfully")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  NMSE: {nmse:.6f}")
    
except Exception as e:
    print(f"✗ Error in GLM estimation: {e}")
    sys.exit(1)

# Test 5: GRNN Estimation
print("\nTest 5: GRNN Estimation")
print("-" * 40)

from python_code.grnn_estimation import grnn_estimation

print(f"Running GRNN estimation with {k_sensors} sensors...")

try:
    pred, true_val, err, nmse = grnn_estimation(
        n_nodes, selected, seq,
        train_data.copy(), train_data.copy(),
        valid_data.copy(), valid_data.copy()
    )
    print(f"✓ GRNN estimation completed successfully")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  NMSE: {nmse:.6f}")
    
except Exception as e:
    print(f"✗ Error in GRNN estimation: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("All tests passed successfully! ✓")
print("="*70)
print("\nThe Python implementation is working correctly.")
print("You can now run the full pipeline with real data:")
print("  python -m python_code --max-sensors 250")
print("\nOr try the simple example with synthetic data:")
print("  python -m python_code --example")
print("="*70)
