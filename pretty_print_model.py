#!/usr/bin/env python3
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LearnedThermalModel:
    """Persistent thermal model parameters learned over time"""

    H_matrix: np.ndarray  # Heat transfer coefficients [W/K]
    C_vector: Optional[np.ndarray] = None  # Thermal capacitances [J/K]
    training_samples: int = 0  # Number of samples used for training
    last_updated: float = 0.0  # Timestamp of last update
    kalman_state: Optional[np.ndarray] = None  # Augmented state vector
    kalman_covariance: Optional[np.ndarray] = None  # Covariance matrix
    prediction_errors: List[float] = field(default_factory=list)  # Recent MSE
    learning_rate_history: List[float] = field(default_factory=list)
    H_variance: Optional[np.ndarray] = None  # Uncertainty in H estimates
    model_version: int = 1  # For future compatibility


# Load the pickle
with open("thermal_model_server.pkl", "rb") as f:
    model = pickle.load(f)

# Component and fan names for reference
node_names = [
    "SOC",
    "CORE_VRD",
    "DIMM_CH0",
    "DIMM_CH1",
    "DIMM_CH4",
    "DIMM_CH5",
    "PERIPHERAL",
    "SOC_VRD",
    "DIMM_VRD",
    "SYSTEM",
    "NVME",
]
fan_names = ["FAN1", "FAN2", "FAN3", "FAN6"]

print("=" * 80)
print("LEARNED THERMAL MODEL ANALYSIS")
print("=" * 80)
print(f"\nTraining samples: {model.training_samples}")
print(f"Model version: {model.model_version}")
print()

print("H MATRIX - Heat Transfer Coefficients [W/°C]")
print("-" * 80)
print("Component      | FAN1     FAN2     FAN3     FAN6     | Total    | Analysis")
print("-" * 80)

for i, node in enumerate(node_names):
    values = model.H_matrix[i]
    total = np.sum(values)

    # Analyze the cooling pattern
    if total > 15:
        analysis = "HIGH - Large thermal mass/ambient sensor"
    elif total > 10:
        analysis = "MEDIUM-HIGH - Distributed component"
    elif total > 5:
        analysis = "MEDIUM - Moderate cooling"
    elif total > 2:
        analysis = "LOW - Direct heatsink cooling"
    else:
        analysis = "VERY LOW - Minimal direct cooling"

    print(
        f"{node:12s}  | {values[0]:7.3f}  {values[1]:7.3f}  {values[2]:7.3f}  {values[3]:7.3f}  | {total:7.3f}  | {analysis}"
    )

print("-" * 80)
print(f"Overall mean H: {np.mean(model.H_matrix):7.3f} W/°C")
print(f"Overall max H:  {np.max(model.H_matrix):7.3f} W/°C")
print(f"Overall min H:  {np.min(model.H_matrix):7.3f} W/°C")

print("\n\nCOOLING PATTERN ANALYSIS")
print("-" * 80)

# Analyze which fans cool which components most
print("\nPrimary cooling relationships (H > 1.0 W/°C):")
for i, node in enumerate(node_names):
    strong_fans = []
    for j, fan in enumerate(fan_names):
        if model.H_matrix[i, j] > 1.0:
            strong_fans.append(f"{fan} ({model.H_matrix[i, j]:.1f})")
    if strong_fans:
        print(f"  {node}: {', '.join(strong_fans)}")

# Additional statistics
if model.prediction_errors:
    print(
        f"\nRecent prediction errors (MSE): {np.mean(model.prediction_errors[-10:]):.3f}"
    )

print(f"\nLearning progress: {model.training_samples} samples collected")
