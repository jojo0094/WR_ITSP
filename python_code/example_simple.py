"""
Simple example demonstrating the sensor selection algorithm
with synthetic data (no need to download large files)
"""

import numpy as np
import matplotlib.pyplot as plt
from sensor_selection import sensor_selection
from mi_comparison import mi_comparison_bellinge

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data
print("="*70)
print("Simple Example: Sensor Selection with Synthetic Data")
print("="*70)

# Parameters
n_nodes = 50  # Number of nodes (smaller for demonstration)
n_timepoints = 1000  # Number of time points
max_sensors = 25  # Maximum sensors to place

print(f"\nNetwork parameters:")
print(f"  Number of nodes: {n_nodes}")
print(f"  Time points: {n_timepoints}")
print(f"  Max sensors: {max_sensors}")

# Generate synthetic correlated time series
print("\n[1/4] Generating synthetic time series data...")
# Create a base signal
time = np.linspace(0, 10, n_timepoints)
base_signal = np.sin(2 * np.pi * 0.5 * time) + 0.5 * np.sin(2 * np.pi * 1.5 * time)

# Create correlated node data
data = np.zeros((n_timepoints, n_nodes))
for i in range(n_nodes):
    # Each node has the base signal plus some independent variation
    phase_shift = np.random.rand() * 2 * np.pi
    amplitude = 0.5 + np.random.rand() * 1.5
    noise = np.random.randn(n_timepoints) * 0.2
    data[:, i] = amplitude * np.sin(2 * np.pi * 0.5 * time + phase_shift) + noise

# Compute covariance matrix
print("[2/4] Computing covariance matrix...")
covmatrix = np.cov(data.T)
print(f"  Covariance matrix shape: {covmatrix.shape}")

# Sensor noise variance
sigmas = 1e-4

# Run sensor selection algorithm
print(f"\n[3/4] Running sensor selection algorithm (k={max_sensors})...")
print("  This may take a few minutes...")
optimal_sensors = sensor_selection(n_nodes, covmatrix, sigmas, max_sensors)
print("  Sensor selection complete!")

# Display results for different k values
print("\n[4/4] Results:")
print("\nOptimal sensor placements for different k values:")
for k in [5, 10, 15, 20, 25]:
    sensors = optimal_sensors[k-1, 4:4+k].astype(int)
    mi_value = optimal_sensors[k-1, -1]
    print(f"  k={k:2d}: Sensors = {sensors[:5]}... (first 5), MI = {mi_value:.4f}")

# Compare with heuristics
print("\n[5/4] Comparing with heuristic methods...")
# Rank nodes by total sum
node_totalsum = np.argsort(np.sum(data, axis=0))[::-1]
# Create a simple rule-based selection (e.g., evenly spaced)
rule_selection = np.linspace(0, n_nodes-1, max_sensors, dtype=int)

# Run MI comparison (with fewer random simulations for speed)
MI_totalsum, MI_mean, MI_min, MI_max, MI_rule = mi_comparison_bellinge(
    max_sensors, 100, node_totalsum, covmatrix, sigmas, n_nodes, rule_selection
)

# Plot results
print("\n[6/4] Generating comparison plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(1, max_sensors + 1)
algo_mi = optimal_sensors[:, -1]

ax1.plot(x, algo_mi, 'o-', label='Algorithm 1 (Optimal)', color='red', linewidth=2)
ax1.plot(x, MI_max, 's-', label='Best Random', color='blue', alpha=0.7)
ax1.plot(x, MI_mean, '^-', label='Avg Random', color='green', alpha=0.7)
ax1.plot(x, MI_rule, 'v-', label='Rule-based', color='magenta', alpha=0.7)
ax1.plot(x, MI_totalsum, 'd-', label='Total Sum', color='cyan', alpha=0.7)
ax1.set_xlabel('Number of sensors', fontsize=12)
ax1.set_ylabel('Mutual Information', fontsize=12)
ax1.set_title('MI Comparison', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# MI Gain plot
ax2.plot(x, (algo_mi - MI_max) / algo_mi * 100, label='vs Best Random', color='blue')
ax2.plot(x, (algo_mi - MI_mean) / algo_mi * 100, label='vs Avg Random', color='green')
ax2.plot(x, (algo_mi - MI_rule) / algo_mi * 100, label='vs Rule-based', color='magenta')
ax2.plot(x, (algo_mi - MI_totalsum) / algo_mi * 100, label='vs Total Sum', color='cyan')
ax2.set_xlabel('Number of sensors', fontsize=12)
ax2.set_ylabel('MI Gain (%)', fontsize=12)
ax2.set_title('Algorithm 1 Improvement', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example_results.png', dpi=150, bbox_inches='tight')
print("  Plot saved as 'example_results.png'")

# Summary statistics
print("\n" + "="*70)
print("Summary Statistics:")
print("="*70)
avg_improvement_random = np.mean((algo_mi - MI_mean) / algo_mi * 100)
avg_improvement_rule = np.mean((algo_mi - MI_rule) / algo_mi * 100)
print(f"Average MI improvement over random placement: {avg_improvement_random:.2f}%")
print(f"Average MI improvement over rule-based: {avg_improvement_rule:.2f}%")
print("\nExample complete! Check 'example_results.png' for visualization.")
print("="*70)
