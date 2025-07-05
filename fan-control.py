#!/usr/bin/env python3
"""
Thermal Fan Control System for Supermicro ARS-210M-NR Server
===========================================================

This implementation provides a physics-based adaptive fan control system using:
- First-principles thermodynamics for heat transfer modeling
- Model Predictive Control (MPC) for optimal fan speed selection
- Online parameter estimation via Extended Kalman Filter
- Multi-objective optimization balancing temperature, power, and acoustics

Motivation:
-----------

Homelab not sounding like a jet engine

Mathematical Foundation:
-----------------------
The system is modeled as a network of thermal nodes following the heat equation:

    C_i * dT_i/dt = Q_i(t) - Σ_j H_ij(u_j) * (T_i - T_inlet)

Where:
    C_i: Thermal capacitance of node i [J/K]
    T_i: Temperature of node i [K]
    Q_i: Heat generation at node i [W]
    H_ij: Heat transfer coefficient from node i due to fan j [W/K]
    u_j: PWM duty cycle of fan j [0-1]
    T_inlet: Ambient inlet temperature [K]

Author: 0xAlcibiades (Thermal Control Plumber)
Date: July 5 2025
"""

import subprocess
import time
import logging
import signal
import sys
import pickle
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
from enum import Enum

import numpy as np
import scipy.optimize as opt


# Physical Constants
AIR_DENSITY = 1.18  # kg/m³ at 35°C
AIR_SPECIFIC_HEAT = 1005  # J/(kg·K)
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# Fan Specifications (80mm server fans)
FAN_DIAMETER = 0.08  # m
FAN_AREA = np.pi * (FAN_DIAMETER / 2) ** 2  # m²
MAX_FLOW_RATE = 0.038  # m³/s at 100% PWM (80 CFM)

# Default power dissipation values for missing sensors [W]
# Based on typical component TDPs and idle power consumption
POWER_DEFAULTS = {
    "SOC": 15.0,  # ARM SOC typical idle
    "CORE_VRD": 50.0,  # CPU at light load
    "DIMM_CH0": 3.0,  # DDR4 per channel
    "DIMM_CH1": 3.0,
    "DIMM_CH4": 3.0,
    "DIMM_CH5": 3.0,
    "PERIPHERAL": 20.0,  # PCIe devices
    "SOC_VRD": 1.5,  # VRM losses
    "DIMM_VRD": 1.0,  # VRM losses
    "SYSTEM": 10.0,  # Chipset, etc
    "NVME": 5.0,  # Per NVMe drive idle
}


class SensorType(Enum):
    """Enumeration of IPMI sensor types for parsing"""

    TEMPERATURE = "degrees C"
    POWER = "Watts"
    VOLTAGE = "Volts"
    FAN_SPEED = "RPM"
    PWM = "unspecified"  # PWM values show as unspecified
    DISCRETE = "discrete"


@dataclass
class ThermalNode:
    """Represents a thermal component in the system"""

    name: str
    sensor_name: str
    warning_temp: float  # °C
    critical_temp: float  # °C
    thermal_capacitance: float = 100.0  # J/K (default, component-specific)

    def __post_init__(self):
        """Convert temperatures to Kelvin for calculations"""
        self.warning_temp_K = self.warning_temp + 273.15
        self.critical_temp_K = self.critical_temp + 273.15


@dataclass
class LearnedThermalModel:
    """Persistent thermal model parameters learned over time"""

    H_matrix: np.ndarray  # Heat transfer coefficients [W/K]
    C_vector: Optional[np.ndarray] = None  # Thermal capacitances [J/K]
    training_samples: int = 0  # Number of samples used for training
    last_updated: float = 0.0  # Timestamp of last update

    # Kalman filter state
    kalman_state: Optional[np.ndarray] = None  # Augmented state vector
    kalman_covariance: Optional[np.ndarray] = None  # Covariance matrix

    # Performance metrics
    prediction_errors: List[float] = field(default_factory=list)  # Recent MSE
    learning_rate_history: List[float] = field(default_factory=list)

    # Model confidence/quality metrics
    H_variance: Optional[np.ndarray] = None  # Uncertainty in H estimates
    model_version: int = 1  # For future compatibility

    def save(self, path: str):
        """Save model to disk with error handling"""
        try:
            # Create backup if file exists
            if os.path.exists(path):
                backup_path = f"{path}.backup"
                os.rename(path, backup_path)

            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Remove backup on success
            if os.path.exists(f"{path}.backup"):
                os.remove(f"{path}.backup")

        except Exception as e:
            # Restore backup on failure
            if os.path.exists(f"{path}.backup"):
                os.rename(f"{path}.backup", path)
            raise e

    @staticmethod
    def load(path: str) -> "LearnedThermalModel":
        """Load model from disk with validation"""
        with open(path, "rb") as f:
            model = pickle.load(f)

        # Validate loaded model
        if not isinstance(model, LearnedThermalModel):
            raise ValueError("Invalid model file format")

        if model.H_matrix.ndim != 2:
            raise ValueError("Invalid H matrix dimensions")

        return model

    def update_metrics(self, prediction_error: float):
        """Update performance metrics with sliding window"""
        self.prediction_errors.append(prediction_error)
        # Keep last 100 errors
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)

        self.training_samples += 1
        self.last_updated = time.time()


@dataclass
class FanSpec:
    """Fan specifications and characteristics"""

    name: str
    zone_id: str  # IPMI zone hex code
    pwm_sensor: str  # PWM sensor name
    rpm_sensor: str  # RPM sensor name
    max_rpm: float = 14000.0  # Maximum RPM
    rpm_per_pwm: float = 140.0  # RPM per PWM percent (estimated)

    def pwm_to_rpm(self, pwm: float) -> float:
        """Convert PWM duty cycle (0-1) to RPM"""
        return pwm * 100 * self.rpm_per_pwm

    def rpm_to_flow_rate(self, rpm: float) -> float:
        """Convert RPM to volumetric flow rate [m³/s]

        Assumes linear relationship: Q = k * RPM
        where k is calibrated from max specs
        """
        return (rpm / self.max_rpm) * MAX_FLOW_RATE


@dataclass
class SystemState:
    """Complete system state at a point in time"""

    timestamp: float
    temperatures: Dict[str, float]  # °C
    powers: Dict[str, float]  # Watts
    fan_pwms: Dict[str, float]  # 0-100
    fan_rpms: Dict[str, float]  # RPM
    inlet_temp: float  # °C

    def to_arrays(
        self,
        node_order: List[str],
        fan_order: List[str],
        power_defaults: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to numpy arrays for numerical computation

        Args:
            node_order: List of thermal node names in order
            fan_order: List of fan names in order
            power_defaults: Default power values for missing sensors

        Returns:
            temps: Temperature array [°C]
            powers: Power dissipation array [W]
            pwms: Fan PWM array [0-1]

        Raises:
            ValueError: If critical sensors are missing
        """
        # Validate temperature sensors - all must be present
        missing_temps = [node for node in node_order if node not in self.temperatures]
        if missing_temps:
            raise ValueError(f"Missing critical temperature sensors: {missing_temps}")

        # Validate fan sensors - all must be present
        missing_fans = [fan for fan in fan_order if fan not in self.fan_pwms]
        if missing_fans:
            raise ValueError(f"Missing fan PWM sensors: {missing_fans}")

        # Build arrays
        temps = np.array([self.temperatures[node] for node in node_order])

        # For power, use defaults if provided, otherwise raise error
        if power_defaults:
            powers = np.array(
                [
                    self.powers.get(node, power_defaults.get(node, 0.0))
                    for node in node_order
                ]
            )
        else:
            missing_powers = [node for node in node_order if node not in self.powers]
            if missing_powers:
                raise ValueError(
                    f"Missing power sensors and no defaults provided: {missing_powers}"
                )
            powers = np.array([self.powers[node] for node in node_order])

        # Convert PWM from 0-100 to 0-1 range
        pwms = np.array([self.fan_pwms[fan] / 100.0 for fan in fan_order])

        return temps, powers, pwms


class IPMIInterface:
    """Interface to IPMI for sensor reading and fan control"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def read_sensors(self) -> Dict[str, Tuple[float, str]]:
        """Read all IPMI sensors

        Returns:
            Dict mapping sensor name to (value, unit) tuples
        """
        try:
            result = subprocess.run(
                ["ipmitool", "sensor"],
                capture_output=True,
                text=True,
                timeout=10,  # IPMI tool is slow
            )

            if result.returncode != 0:
                self.logger.error("IPMI sensor read failed: %s", result.stderr)
                return {}

            sensors = {}
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    name = parts[0]
                    value_str = parts[1]
                    unit = parts[2]

                    # Skip discrete sensors and parse numeric values
                    if unit != SensorType.DISCRETE.value and value_str != "na":
                        try:
                            value = float(value_str)
                            sensors[name] = (value, unit)
                        except ValueError:
                            continue

            return sensors

        except Exception as e:
            self.logger.error("Error reading sensors: %s", e)
            return {}

    def set_fan_mode(self, mode: str = "FULL") -> bool:
        """Set IPMI fan control mode

        Args:
            mode: One of 'STANDARD', 'FULL', 'OPTIMAL', 'PUE2'
        """
        mode_codes = {
            "STANDARD": "0x00",
            "FULL": "0x01",
            "OPTIMAL": "0x02",
            "PUE2": "0x04",
        }

        if mode not in mode_codes:
            return False

        try:
            result = subprocess.run(
                ["ipmitool", "raw", "0x30", "0x45", "0x01", mode_codes[mode]],
                capture_output=True,
                timeout=10,  # IPMITool is slow
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error("Error setting fan mode: %s", e)
            return False

    def set_fan_duty(self, zone_id: str, duty_percent: int) -> bool:
        """Set fan PWM duty cycle

        Args:
            zone_id: Hex zone ID (e.g., '0x00')
            duty_percent: Duty cycle 0-100
        """
        duty_percent = max(0, min(100, duty_percent))
        duty_hex = f"0x{duty_percent:02x}"

        try:
            result = subprocess.run(
                ["ipmitool", "raw", "0x30", "0x70", "0x66", "0x01", zone_id, duty_hex],
                capture_output=True,
                timeout=10,  # IPMITool is slow
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error("Error setting fan duty: %s", e)
            return False


class ThermalModel:
    """Physics-based thermal model of the server

    Implements the discrete-time state-space model:
        x[k+1] = A*x[k] + B*u[k] + E*w[k]

    Where:
        x: Temperature state vector (relative to inlet)
        u: Fan PWM control vector
        w: Power disturbance vector
    """

    def __init__(self, thermal_nodes: List[ThermalNode], fans: List[FanSpec]):
        self.nodes = {node.name: node for node in thermal_nodes}
        self.node_order = [node.name for node in thermal_nodes]
        self.fans = {fan.name: fan for fan in fans}
        self.fan_order = [fan.name for fan in fans]

        self.n_states = len(thermal_nodes)
        self.n_controls = len(fans)

        # Initialize thermal parameters
        self.C = np.diag(
            [node.thermal_capacitance for node in thermal_nodes]
        )  # Capacitance matrix
        self.H = self._initialize_heat_transfer_matrix()  # Heat transfer coefficients

        # Initialize state-space matrices (will be updated with dt)
        self.dt = 15.0  # Default timestep - matches control loop interval
        self._update_discrete_matrices()

        # Cache frequently used matrices
        self._C_inv = np.linalg.inv(self.C)

        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_heat_transfer_matrix(self) -> np.ndarray:
        """Initialize heat transfer coefficient matrix H[i,j]

        H[i,j] represents heat transfer from node i due to fan j
        Units: W/K at 100% PWM

        Initial estimates based on physical layout:
        - Front fans (1,2) primarily cool CPU/SOC
        - Middle fan (3) cools peripherals
        - Rear fan (6) cools DIMMs
        """
        h_matrix = np.zeros((self.n_states, self.n_controls))

        # Map node indices
        node_map = {name: i for i, name in enumerate(self.node_order)}

        # Initial estimates based on airflow patterns
        # These will be refined through online learning

        # Front fans strongly cool CPU components
        if "CORE_VRD" in node_map:
            h_matrix[node_map["CORE_VRD"], 0] = 3.0  # W/K - strong coupling
            h_matrix[node_map["CORE_VRD"], 1] = 3.0

        if "SOC" in node_map:
            h_matrix[node_map["SOC"], 0] = 3.0  # Increased for better initial response
            h_matrix[node_map["SOC"], 1] = 3.0

        # Middle fan cools peripherals
        if "PERIPHERAL" in node_map:
            h_matrix[node_map["PERIPHERAL"], 2] = 3.0

        # Rear fan cools memory
        for dimm in ["DIMM_CH0", "DIMM_CH1", "DIMM_CH4", "DIMM_CH5"]:
            if dimm in node_map:
                h_matrix[node_map[dimm], 3] = 0.8

        # All fans contribute somewhat to general cooling
        h_matrix += 0.2  # Background coupling

        return h_matrix

    def _update_discrete_matrices(self):
        """Update discrete-time state-space matrices for current timestep"""
        # Continuous-time system: dx/dt = -C^(-1) * H * u * (x - x_inlet) + C^(-1) * w
        # Discretize using forward Euler: x[k+1] = x[k] + dt * f(x[k], u[k])

        # For now, linearize around nominal operating point
        # More sophisticated: use exact discretization of time-varying system

        self.A = np.eye(self.n_states)  # Will be updated based on operating point
        self.B = np.zeros((self.n_states, self.n_controls))
        self._C_inv = np.linalg.inv(self.C)
        self.E = self.dt * self._C_inv  # Power input matrix

    def heat_transfer_function(self, pwm: float, alpha: float = 0.8) -> float:
        """Nonlinear heat transfer as function of PWM

        Models the relationship: h(u) = h_max * u^alpha
        where alpha ~ 0.8 for turbulent flow
        """
        return pwm**alpha

    def predict_step(
        self,
        temps: np.ndarray,
        controls: np.ndarray,
        powers: np.ndarray,
        inlet_temp: float,
    ) -> np.ndarray:
        """Predict temperature evolution one timestep forward

        Args:
            temps: Current temperatures [°C]
            controls: Fan PWM values [0-1]
            powers: Heat generation [W]
            inlet_temp: Inlet temperature [°C]

        Returns:
            Predicted temperatures at next timestep [°C]
        """
        # Convert to temperature rise above inlet
        temp_rise = temps - inlet_temp

        # Compute heat transfer coefficients for current fan speeds
        h_effective = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_controls):
                h_effective[i, i] += self.H[i, j] * self.heat_transfer_function(
                    controls[j]
                )

        # Temperature derivative: C * dT/dt = Q - H * (T - T_inlet)
        dT_dt = self._C_inv @ (powers - h_effective @ temp_rise)

        # Forward Euler integration
        new_temp_rise = temp_rise + self.dt * dT_dt

        # Sanity check: temperatures can't drop below inlet or rise above boiling
        new_temps = new_temp_rise + inlet_temp
        new_temps = np.clip(new_temps, inlet_temp - 5.0, 150.0)

        return new_temps

    def linearize_dynamics(
        self,
        temps: np.ndarray,
        controls: np.ndarray,
        powers: np.ndarray,
        inlet_temp: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize system dynamics around operating point

        Returns:
            A, B matrices for linearized discrete-time system
        """
        temp_rise = temps - inlet_temp

        # A matrix: ∂f/∂x
        a_matrix = np.eye(self.n_states)
        h_effective = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_controls):
                h_effective[i, i] += self.H[i, j] * self.heat_transfer_function(
                    controls[j]
                )

        a_matrix -= self.dt * self._C_inv @ h_effective

        # B matrix: ∂f/∂u
        b_matrix = np.zeros((self.n_states, self.n_controls))
        alpha = 0.8
        for i in range(self.n_states):
            for j in range(self.n_controls):
                if controls[j] > 0:
                    b_matrix[i, j] = (
                        -self.dt
                        * self.H[i, j]
                        * alpha
                        * controls[j] ** (alpha - 1)
                        * temp_rise[i]
                        / self.C[i, i]
                    )

        return a_matrix, b_matrix


class KalmanFilter:
    """Extended Kalman Filter for joint state and parameter estimation

    Estimates both temperatures and thermal model parameters online
    """

    def __init__(
        self,
        model: ThermalModel,
        process_noise: float = 1.0,
        measurement_noise: float = 0.5,
        param_noise: float = 0.001,
        learned_model: Optional[LearnedThermalModel] = None,
    ):
        self.model = model

        # Augmented state: [temperatures; vec(H)]
        self.n_temps = model.n_states
        self.n_params = model.n_states * model.n_controls
        self.n_augmented = self.n_temps + self.n_params

        # Initialize or restore state
        if learned_model and learned_model.kalman_state is not None:
            # Restore from saved state
            self.x = learned_model.kalman_state.copy()
            self.P = learned_model.kalman_covariance.copy()
            # Update model H matrix from saved state
            model.H = self.x[self.n_temps :].reshape((self.n_temps, model.n_controls))
        else:
            # Initialize state and covariance
            self.x = np.zeros(self.n_augmented)
            self.x[: self.n_temps] = 40.0  # Initial temp estimate
            self.x[self.n_temps :] = model.H.flatten()  # Initial H matrix

            self.P = np.eye(self.n_augmented)
            self.P[: self.n_temps, : self.n_temps] *= process_noise
            self.P[self.n_temps :, self.n_temps :] *= param_noise

        # Noise covariances
        self.Q = np.eye(self.n_augmented)
        self.Q[: self.n_temps, : self.n_temps] *= process_noise
        self.Q[self.n_temps :, self.n_temps :] *= param_noise

        self.R = measurement_noise * np.eye(self.n_temps)

        # Track prediction errors and updates
        self.last_prediction_error = 0.0
        self.iter_count = 0
        self.initial_H = model.H.copy()

    def get_state_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state for saving"""
        return self.x.copy(), self.P.copy()

    def predict(self, controls: np.ndarray, powers: np.ndarray, inlet_temp: float):
        """Prediction step of EKF"""
        # Extract current estimates
        temps = self.x[: self.n_temps]
        h_flat = self.x[self.n_temps :]
        self.model.H = h_flat.reshape((self.n_temps, self.model.n_controls))

        # Predict temperature evolution
        new_temps = self.model.predict_step(temps, controls, powers, inlet_temp)

        # Parameters don't change in prediction
        self.x[: self.n_temps] = new_temps

        # Linearize dynamics for covariance update
        a_temp, b_temp = self.model.linearize_dynamics(
            temps, controls, powers, inlet_temp
        )

        # Compute Jacobian of temperature dynamics w.r.t. H parameters
        # ∂f/∂H[i,j] = -dt/C[i] * u[j]^alpha * (T[i] - T_inlet)
        temp_rise = temps - inlet_temp
        F_H = np.zeros((self.n_temps, self.n_params))
        alpha = 0.8

        for i in range(self.n_temps):
            for j in range(self.model.n_controls):
                param_idx = i * self.model.n_controls + j
                # Derivative of temperature i w.r.t. H[i,j]
                F_H[i, param_idx] = (
                    -self.model.dt
                    * (controls[j] ** alpha)
                    * temp_rise[i]
                    / self.model.C[i, i]
                )

        # Augmented A matrix with parameter coupling
        a_aug = np.eye(self.n_augmented)
        a_aug[: self.n_temps, : self.n_temps] = a_temp
        a_aug[: self.n_temps, self.n_temps :] = (
            F_H  # Temperature depends on H parameters
        )

        # Covariance prediction
        self.P = a_aug @ self.P @ a_aug.T + self.Q

    def update(self, measured_temps: np.ndarray):
        """Update step of EKF with temperature measurements"""
        # For parameter learning, we need the predicted temperatures to depend on H
        # Since we just observe temps directly, the measurement Jacobian is simple
        h_meas = np.zeros((self.n_temps, self.n_augmented))
        h_meas[: self.n_temps, : self.n_temps] = np.eye(self.n_temps)

        # Innovation
        y_pred = self.x[: self.n_temps]
        innovation = measured_temps - y_pred

        # Track prediction error
        self.last_prediction_error = np.mean(innovation**2)  # MSE
        self.iter_count += 1

        # Kalman gain
        s_innovation = h_meas @ self.P @ h_meas.T + self.R
        k_gain = self.P @ h_meas.T @ np.linalg.inv(s_innovation)

        # State update - this now updates both temps AND H parameters
        self.x += k_gain @ innovation

        # Covariance update
        self.P = (np.eye(self.n_augmented) - k_gain @ h_meas) @ self.P

        # Extract updated H matrix
        h_flat = self.x[self.n_temps :]
        self.model.H = h_flat.reshape((self.n_temps, self.model.n_controls))

        # Ensure H remains in reasonable bounds based on physics
        # Lower bound: some minimal cooling effect
        # Upper bound: allow high values for high-power components like 64-core CPUs
        self.model.H = np.clip(self.model.H, 0.1, 100.0)

        # Log significant H changes
        if self.iter_count % 10 == 0:  # Every 10 updates
            max_h_change = np.max(np.abs(self.model.H - self.initial_H))
            if max_h_change > 0.1:
                logging.getLogger(__class__.__name__).info(
                    "H matrix learning: max change = %.3f", max_h_change
                )


class AcousticModel:
    """Model for fan noise generation

    Based on fan acoustics theory:
    - Rotational noise ~ omega^5
    - Turbulent noise ~ omega^3 * delta_p^2
    """

    def __init__(self, reference_noise: float = 45.0, reference_rpm: float = 5000.0):
        self.L_ref = reference_noise  # dB at reference speed
        self.rpm_ref = reference_rpm

    def fan_noise(self, rpm: float) -> float:
        """Calculate noise level for single fan

        Uses 5th power law for rotational noise (dominant in server fans)
        """
        if rpm <= 0:
            return 0.0

        return self.L_ref + 50 * np.log10(rpm / self.rpm_ref)

    def total_noise(self, rpms: List[float]) -> float:
        """Calculate total noise from all fans

        Combines individual fan noise levels using logarithmic addition
        """
        # Convert to sound pressure
        pressures = [10 ** (self.fan_noise(rpm) / 10) for rpm in rpms if rpm > 0]

        if not pressures:
            return 0.0

        # Total sound pressure
        total_pressure = sum(pressures)

        # Convert back to dB
        return 10 * np.log10(total_pressure)


class MPCController:
    """Model Predictive Controller for optimal fan control

    Solves the finite-horizon optimal control problem:
        min Σ L(x[k], u[k]) for k=0 to N-1

    Subject to:
        - System dynamics
        - Temperature constraints
        - Control constraints
    """

    def __init__(
        self,
        model: ThermalModel,
        acoustic_model: AcousticModel,
        horizon: int = 6,
        dt: float = None,
    ):
        self.model = model
        self.acoustic = acoustic_model
        self.horizon = horizon
        # Use model's dt to ensure consistency
        self.dt = dt if dt is not None else model.dt
        self.last_solution = None  # Store for warm-starting

        # Cost function weights - balanced for homelab use
        # Temperature is primary concern but not overwhelming
        self.w_temp = 1.0  # Temperature deviation from target
        self.w_power = 0.0001  # Fan power (cubic in speed) - electricity is cheap
        self.w_noise = 0.1  # Acoustic comfort - 10% as important as temperature
        self.w_change = 0.05  # Smooth transitions - 5% as important as temperature

        # Temperature targets and limits
        # Set targets based on component thermal limits
        self.temp_targets = np.zeros(model.n_states)
        for i, node_name in enumerate(model.node_order):
            node = model.nodes[node_name]
            # Target = warning_temp - 20°C
            self.temp_targets[i] = node.warning_temp - 20.0

        self.temp_comfort = 5.0  # Comfort band around target (±5°C)

        # Control limits
        self.u_min = 0.25  # Minimum PWM (25%) - prevents stalling
        self.u_max = 1.0  # Maximum PWM (100%)
        self.du_max = (
            1.0  # Maximum change per step (100% - allows full range in one step)
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def temperature_cost(
        self, temp: float, target: float, warning: float, critical: float
    ) -> float:
        """Smooth temperature cost function

        Provides continuous cost that increases smoothly as temperature
        deviates from target, with rapid growth near critical.
        """
        error = temp - target

        # Within comfort band - no penalty
        if abs(error) <= self.temp_comfort:
            return 0.0

        # Undercooling - mild penalty
        elif error < 0:
            # Distance beyond comfort zone
            undercool = abs(error) - self.temp_comfort
            return 0.1 * undercool**2

        # Overheating - increasing penalty
        else:
            # Distance above comfort zone
            overheat = error - self.temp_comfort

            # Fraction of the way from comfort edge to critical
            # E.g., for SOC: comfort ends at 75°C, critical at 80°C
            # So we have 5°C of margin
            margin = critical - (target + self.temp_comfort)
            if margin <= 0:
                margin = 1.0  # Prevent division by zero

            progress_to_critical = min(overheat / margin, 1.0)

            # Cost increases as cubic of progress (smooth acceleration)
            # overheat^2 provides base quadratic growth
            # progress^3 provides urgency multiplier near critical
            urgency = 1.0 + 9.0 * progress_to_critical**3

            return urgency * overheat**2

    def control_cost(self, controls: np.ndarray) -> float:
        """Cost of running fans (power and noise)

        Implements continuous penalty based on psychoacoustic perception.
        Uses Stevens' power law: perceived loudness doubles every 10 dB.
        """
        # Power consumption ~ speed^3 (fluid dynamics)
        power_cost = self.w_power * np.sum(controls**3)

        # Calculate actual noise level
        rpms = [
            self.model.fans[fan].pwm_to_rpm(u)
            for fan, u in zip(self.model.fan_order, controls)
        ]
        noise_db = self.acoustic.total_noise(rpms)

        # Continuous noise penalty based on psychoacoustic perception
        # Reference: 40 dB is typical home background noise
        # Every 10 dB increase doubles perceived loudness (Stevens' law)
        # Convert dB to perceived loudness units (sones)
        # 40 dB ≈ 1 sone, 50 dB ≈ 2 sones, 60 dB ≈ 4 sones, etc.

        if noise_db <= 40:
            # Below background noise - minimal penalty
            perceived_loudness = 1.0
        else:
            # Stevens' power law: L = 2^((dB - 40) / 10)
            perceived_loudness = 2 ** ((noise_db - 40) / 10)

        # Cost grows with perceived loudness
        # Subtract 1 so quiet operation (40 dB) has near-zero cost
        noise_cost = self.w_noise * (perceived_loudness - 1) ** 2

        return power_cost + noise_cost

    def objective_gradient(
        self,
        u_flat: np.ndarray,
        temps: np.ndarray,
        current_controls: np.ndarray,
        predicted_powers: np.ndarray,
        inlet_temp: float,
        horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Analytical gradient of objective function for faster optimization

        Returns gradient of objective w.r.t. control variables
        """
        if horizon is None:
            horizon = self.horizon

        u = u_flat.reshape((horizon, self.model.n_controls))
        grad = np.zeros_like(u_flat)

        x = temps.copy()
        alpha = 0.8  # Heat transfer exponent

        # Forward pass to compute states
        states = [x]
        for k in range(horizon):
            x = self.model.predict_step(x, u[k], predicted_powers, inlet_temp)
            states.append(x)

        # Backward pass to compute gradients
        for k in range(horizon):
            x = states[k]
            x_next = states[k + 1]
            temp_rise = x - inlet_temp

            # Temperature cost gradients
            for i in range(self.model.n_states):
                node = self.model.nodes[self.model.node_order[i]]

                # Gradient of temperature cost
                error = x_next[i] - self.temp_targets[i]
                if abs(error) <= self.temp_comfort:
                    dc_dT = 0.0
                elif error < 0:
                    undercool = abs(error) - self.temp_comfort
                    dc_dT = -0.2 * undercool * self.w_temp
                else:
                    overheat = error - self.temp_comfort
                    margin = node.critical_temp - (
                        self.temp_targets[i] + self.temp_comfort
                    )
                    if margin <= 0:
                        margin = 1.0
                    progress = min(overheat / margin, 1.0)
                    urgency = 1.0 + 9.0 * progress**3
                    dc_dT = 2.0 * urgency * overheat * self.w_temp

                    # Additional gradient from urgency change
                    if progress < 1.0:
                        d_urgency = 27.0 * progress**2 / margin
                        dc_dT += d_urgency * overheat**2 * self.w_temp

                # Temperature gradient w.r.t. each control
                for j in range(self.model.n_controls):
                    if u[k, j] > 0:
                        # ∂T_next/∂u = -dt * H[i,j] * alpha * u^(alpha-1) * (T - T_inlet) / C[i]
                        dT_du = (
                            -self.dt
                            * self.model.H[i, j]
                            * alpha
                            * (u[k, j] ** (alpha - 1))
                            * temp_rise[i]
                            * self.model._C_inv[i, i]
                        )
                        grad[k * self.model.n_controls + j] += dc_dT * dT_du

            # Control cost gradients (power ~ u³, noise is more complex)
            for j in range(self.model.n_controls):
                # Power gradient: ∂(u³)/∂u = 3u²
                grad[k * self.model.n_controls + j] += 3 * self.w_power * u[k, j] ** 2

                # Noise gradient (simplified - assumes linear in PWM for gradient)
                # More accurate would compute via acoustic model derivative
                grad[k * self.model.n_controls + j] += (
                    self.w_noise * 0.01
                )  # Approximate

            # Rate of change gradient with soft constraints
            if k > 0:
                for j in range(self.model.n_controls):
                    du = u[k, j] - u[k - 1, j]
                    # Quadratic penalty gradient
                    grad[k * self.model.n_controls + j] += 2 * self.w_change * du
                    grad[(k - 1) * self.model.n_controls + j] -= 2 * self.w_change * du

                    # Soft constraint gradient (exponential penalty)
                    if abs(du) > self.du_max:
                        sign_du = np.sign(du)
                        violation = abs(du) - self.du_max
                        grad_penalty = 100.0 * np.exp(violation) * sign_du
                        grad[k * self.model.n_controls + j] += grad_penalty
                        grad[(k - 1) * self.model.n_controls + j] -= grad_penalty
            else:
                for j in range(self.model.n_controls):
                    du = u[k, j] - current_controls[j]
                    # Quadratic penalty gradient
                    grad[k * self.model.n_controls + j] += 2 * self.w_change * du

                    # Soft constraint gradient
                    if abs(du) > self.du_max:
                        sign_du = np.sign(du)
                        violation = abs(du) - self.du_max
                        grad[k * self.model.n_controls + j] += (
                            100.0 * np.exp(violation) * sign_du
                        )

        return grad

    def _get_temperature_state(self, temps: np.ndarray) -> str:
        """Determine overall temperature state of the system

        Returns:
            'safe': All temps below warning thresholds
            'warning': Some temps near warning but none critical
            'critical': Any temp near or above critical threshold
        """
        max_severity = 0.0

        for i, (temp, node_name) in enumerate(zip(temps, self.model.node_order)):
            node = self.model.nodes[node_name]

            # How close to critical (0-1 scale)
            criticality = (temp - node.warning_temp) / (
                node.critical_temp - node.warning_temp
            )
            criticality = np.clip(criticality, 0.0, 1.0)

            max_severity = max(max_severity, criticality)

        if max_severity >= 0.8:
            return "critical"
        elif max_severity >= 0.3:
            return "warning"
        else:
            return "safe"

    def solve(
        self, current_state: SystemState, predicted_powers: np.ndarray
    ) -> np.ndarray:
        """Solve MPC optimization problem

        Args:
            current_state: Current system state
            predicted_powers: Predicted power over horizon [W]

        Returns:
            Optimal control sequence (only first action is applied)
        """
        # Extract current state
        temps, _, current_controls = current_state.to_arrays(
            self.model.node_order, self.model.fan_order, power_defaults=POWER_DEFAULTS
        )
        inlet_temp = current_state.inlet_temp

        # Adaptive problem formulation based on temperature state
        temp_state = self._get_temperature_state(temps)

        if temp_state == "critical":
            # Emergency mode: Use short horizon, aggressive optimization
            effective_horizon = min(2, self.horizon)
            max_iterations = 50
            tolerance = 1e-2  # Lower tolerance for faster solving
            self.logger.warning(
                "Critical temperatures detected, using emergency MPC mode"
            )
        elif temp_state == "warning":
            # Normal mode with standard settings
            effective_horizon = self.horizon
            max_iterations = 50
            tolerance = 1e-3
        else:
            # Safe mode: Can use longer horizon for better efficiency
            effective_horizon = self.horizon
            max_iterations = 100  # Allow more iterations for complex optimization
            tolerance = 5e-3  # Slightly relaxed tolerance

        # Decision variables: controls over horizon
        n_vars = self.model.n_controls * effective_horizon

        # Initial guess: always use current controls as starting point
        u0 = np.tile(current_controls, effective_horizon)

        # Bounds
        bounds = [(self.u_min, self.u_max)] * n_vars

        # Constraint function
        def constraints(u_flat):
            u = u_flat.reshape((effective_horizon, self.model.n_controls))

            # Rate of change constraints
            cons = []
            u_prev = current_controls
            for k in range(effective_horizon):
                du = u[k] - u_prev
                cons.extend(du - self.du_max)  # du <= du_max
                cons.extend(-self.du_max - du)  # -du_max <= du
                u_prev = u[k]

            return np.array(cons)

        # Objective function
        def objective(u_flat):
            u = u_flat.reshape((effective_horizon, self.model.n_controls))

            cost = 0.0
            x = temps.copy()

            # Simulate over horizon
            for k in range(effective_horizon):
                # Predict next state
                x = self.model.predict_step(x, u[k], predicted_powers, inlet_temp)

                # Temperature costs
                for i, (temp, node_name) in enumerate(zip(x, self.model.node_order)):
                    node = self.model.nodes[node_name]
                    cost += self.temperature_cost(
                        temp,
                        self.temp_targets[i],
                        node.warning_temp,
                        node.critical_temp,
                    )

                # Control costs
                cost += self.control_cost(u[k])

                # Rate of change penalty with soft constraints
                if k > 0:
                    du = u[k] - u[k - 1]
                else:
                    du = u[k] - current_controls

                # Quadratic penalty for small changes
                cost += self.w_change * np.sum(du**2)

                # Exponential penalty for exceeding rate limits (soft constraint)
                # This allows violation but at high cost
                violation = np.maximum(0, np.abs(du) - self.du_max)
                if np.any(violation > 0):
                    # Cap violation to prevent numerical overflow
                    violation = np.minimum(violation, 10.0)  # exp(10) ≈ 22000
                    cost += 100.0 * np.sum(np.exp(violation) - 1)

            return cost

        # Add iteration counter and early exit logic
        self.iter_count = 0
        self.best_cost = float("inf")
        self.best_solution = u0.copy()
        self.no_improvement_count = 0

        def objective_with_early_exit(u):
            self.iter_count += 1
            cost = objective(u)

            # Debug extreme costs
            if cost > 1e9 and self.iter_count == 1:
                u_test = u.reshape((effective_horizon, self.model.n_controls))
                x = temps.copy()
                self.logger.warning(f"Extreme cost detected: {cost:.2e}")
                self.logger.warning(f"Initial temps: {x}")
                self.logger.warning(f"Controls: {u_test[0]}")
                # Check first prediction
                x_next = self.model.predict_step(
                    x, u_test[0], predicted_powers, inlet_temp
                )
                self.logger.warning(f"Predicted temps: {x_next}")

            # Track improvement
            if cost < self.best_cost * 0.99:  # 1% improvement threshold
                self.best_cost = cost
                self.best_solution = u.copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.iter_count % 5 == 0:
                self.logger.info(f"MPC iteration {self.iter_count}, cost: {cost:.3f}")

            return cost

        self.logger.info(
            f"Starting SLSQP optimization ({temp_state} mode) with initial guess: %s",
            u0[:4],
        )

        # Use L-BFGS-B with analytical gradients for faster optimization
        # L-BFGS-B is better for bound-constrained problems with many variables
        try:
            result = opt.minimize(
                objective_with_early_exit,
                u0,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": lambda u: -constraints(u)},
                jac=None,  # Use numerical gradients for stability
                options={
                    "maxiter": max_iterations,
                    "ftol": tolerance,
                    "iprint": 2,  # Print iterations
                    "disp": True,  # Display convergence messages
                    "eps": 1e-7,  # Step size for numerical gradients
                },
            )
        except StopIteration:
            # Early exit was triggered - create a successful result
            result = opt.OptimizeResult()
            result.success = True
            result.x = self.best_solution  # Use best solution found
            result.message = "Early exit: Converged"
            self.logger.info("Optimization stopped early due to convergence")

        # Check for early termination opportunity
        if (
            temp_state == "safe"
            and self.no_improvement_count >= 5
            and self.iter_count < max_iterations
            and result.success
        ):
            self.logger.info("Optimization converged early in safe mode")

        if not result.success:
            self.logger.warning("MPC optimization failed: %s", result.message)
            # Fall back to maintaining current controls
            return current_controls

        # Extract optimal control sequence
        if result.success:
            # Store for warm-starting next iteration
            self.last_solution = result.x.copy()
            u_opt = result.x.reshape((effective_horizon, self.model.n_controls))
        else:
            self.logger.warning("MPC failed with message: %s", result.message)
            # Fall back to simple heuristic
            max_temp_error = np.max(
                [temps[i] - self.temp_targets[i] for i in range(len(temps))]
            )
            if max_temp_error > 10:
                # Way over target
                fallback_control = 0.7
            elif max_temp_error > 5:
                # Over target
                fallback_control = 0.5
            elif max_temp_error > 0:
                # Slightly over
                fallback_control = 0.35
            else:
                # Under target
                fallback_control = 0.25

            u_opt = np.full(
                (effective_horizon, self.model.n_controls), fallback_control
            )
            self.last_solution = u_opt.flatten()

        # Return first control action
        return u_opt[0]


class ThermalFanController:
    """Main controller class that orchestrates the thermal management system"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the thermal fan controller

        Args:
            config_file: Path to configuration file (optional)
        """
        # Setup logging
        self._setup_logging()

        # Initialize hardware interface
        self.ipmi = IPMIInterface()

        # Define thermal nodes based on Supermicro ARS-210M-NR layout
        # Thermal capacitance based on component type and mass
        self.thermal_nodes = [
            ThermalNode(
                "SOC", "S0 SOC Temp", 95, 105, 1000.0
            ),  # Ampere Altra - throttles at 105°C
            ThermalNode(
                "CORE_VRD", "S0 CORE VRD Temp", 85, 95, 200.0
            ),  # VRM with heatsink
            ThermalNode("DIMM_CH0", "S0 DIMM CH0 Temp", 75, 85, 50.0),  # Memory module
            ThermalNode("DIMM_CH1", "S0 DIMM CH1 Temp", 75, 85, 50.0),
            ThermalNode("DIMM_CH4", "S0 DIMM CH4 Temp", 75, 85, 50.0),
            ThermalNode("DIMM_CH5", "S0 DIMM CH5 Temp", 75, 85, 50.0),
            ThermalNode("PERIPHERAL", "Peripheral Temp", 60, 70, 1000.0),  # PCB area
            ThermalNode("SOC_VRD", "S0 SOC VRD Temp", 85, 95, 150.0),  # Smaller VRM
            ThermalNode("DIMM_VRD", "S0 DIMM VRD Temp", 85, 95, 150.0),
            ThermalNode("SYSTEM", "System Temp", 60, 70, 5000.0),  # Case/ambient
            ThermalNode("NVME", "NVMe SSDA Temp", 70, 80, 30.0),  # SSD controller
        ]

        # Define fans
        self.fans = [
            FanSpec("FAN1", "0x00", "Pwm 1", "FAN1"),
            FanSpec("FAN2", "0x01", "Pwm 2", "FAN2"),
            FanSpec("FAN3", "0x02", "Pwm 3", "FAN3"),
            FanSpec("FAN6", "0x05", "Pwm 6", "FAN6"),
        ]

        # Power sensor mapping
        self.power_sensors = {
            "CORE_VRD": "S0 CORE VRD Pwr",
            "SOC": "S0 SOC VRD Pwr",
            "DIMM": "S0 DIMM VR1 Pwr",  # Will sum VR1 + VR2
            "SYSTEM": "PW Consumption",
        }

        # Initialize models
        self.thermal_model = ThermalModel(self.thermal_nodes, self.fans)
        self.acoustic_model = AcousticModel()

        # Load saved model if available
        self.learned_model = self._load_model()

        # Initialize Kalman filter with saved state if available
        self.kalman = KalmanFilter(
            self.thermal_model, learned_model=self.learned_model, param_noise=0.001
        )  # Reduced for stability with better C values
        self.mpc = MPCController(
            self.thermal_model, self.acoustic_model, horizon=3
        )  # Reduced for faster solving

        # State tracking
        self.state_history = deque(maxlen=1000)  # Keep last ~1.5 hours
        self.running = True
        self.emergency_mode = False
        self.emergency_start_time = 0.0
        self.emergency_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("/var/log/thermal_fan_control.log"),
            ],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received signal %s, initiating graceful shutdown...", signum)
        self.running = False

    def _load_model(self) -> Optional[LearnedThermalModel]:
        """Load previously learned thermal model parameters"""
        model_path = "/var/lib/thermal_model.pkl"
        if os.path.exists(model_path):
            try:
                # Try to load new format first
                model = LearnedThermalModel.load(model_path)

                # Validate dimensions
                if model.H_matrix.shape == self.thermal_model.H.shape:
                    self.thermal_model.H = model.H_matrix
                    self.logger.info(
                        "Loaded thermal model with %d training samples",
                        model.training_samples,
                    )
                    return model
                else:
                    self.logger.warning(
                        "Saved model dimensions don't match, using defaults"
                    )

            except (ValueError, AttributeError):
                # Try legacy format
                try:
                    with open(model_path, "rb") as f:
                        saved_data = pickle.load(f)

                    if isinstance(saved_data, dict) and "H_matrix" in saved_data:
                        # Convert legacy format to new format
                        h_legacy = saved_data["H_matrix"]
                        if h_legacy.shape == self.thermal_model.H.shape:
                            model = LearnedThermalModel(
                                H_matrix=h_legacy,
                                training_samples=saved_data.get("history_length", 0),
                                last_updated=saved_data.get("timestamp", time.time()),
                            )
                            self.thermal_model.H = h_legacy
                            self.logger.info("Converted legacy model to new format")
                            return model

                except Exception as e:
                    self.logger.error("Failed to load legacy model: %s", e)

            except Exception as e:
                self.logger.error("Failed to load saved model: %s", e)

        return None

    def _save_model(self):
        """Save learned thermal model parameters"""
        model_path = "/var/lib/thermal_model.pkl"
        try:
            # Get current Kalman filter state
            kalman_state, kalman_cov = self.kalman.get_state_snapshot()

            # Create or update learned model
            if self.learned_model is None:
                self.learned_model = LearnedThermalModel(
                    H_matrix=self.thermal_model.H.copy(),
                    training_samples=len(self.state_history),
                )
            else:
                self.learned_model.H_matrix = self.thermal_model.H.copy()

            # Update Kalman state
            self.learned_model.kalman_state = kalman_state
            self.learned_model.kalman_covariance = kalman_cov

            # Update metrics
            if hasattr(self.kalman, "last_prediction_error"):
                self.learned_model.update_metrics(self.kalman.last_prediction_error)

            # Save to disk
            self.learned_model.save(model_path)

            self.logger.info(
                "Saved thermal model (%d samples)", self.learned_model.training_samples
            )

        except Exception as e:
            self.logger.error("Failed to save model: %s", e)

    def parse_sensor_data(self, sensors: Dict[str, Tuple[float, str]]) -> SystemState:
        """Parse IPMI sensor data into system state"""
        # Extract temperatures
        temperatures = {}
        for node in self.thermal_nodes:
            if node.sensor_name in sensors:
                temperatures[node.name] = sensors[node.sensor_name][0]

        # Get inlet temperature
        inlet_temp = sensors.get("Inlet Temp", (35.0, "degrees C"))[0]

        # Extract power readings
        powers = {}

        # Direct power measurements
        if self.power_sensors["CORE_VRD"] in sensors:
            powers["CORE_VRD"] = sensors[self.power_sensors["CORE_VRD"]][0]

        if self.power_sensors["SOC"] in sensors:
            powers["SOC"] = sensors[self.power_sensors["SOC"]][0]

        # Sum DIMM powers
        dimm_power = 0.0
        if "S0 DIMM VR1 Pwr" in sensors:
            dimm_power += sensors["S0 DIMM VR1 Pwr"][0]
        if "S0 DIMM VR2 Pwr" in sensors:
            dimm_power += sensors["S0 DIMM VR2 Pwr"][0]

        # Distribute DIMM power across channels
        if dimm_power > 0:
            powers["DIMM_CH0"] = dimm_power / 4
            powers["DIMM_CH1"] = dimm_power / 4
            powers["DIMM_CH4"] = dimm_power / 4
            powers["DIMM_CH5"] = dimm_power / 4

        # Estimate other component powers from total
        total_power = sensors.get(self.power_sensors["SYSTEM"], (200.0, "Watts"))[0]

        # Calculate accounted power (excluding SOC which we'll recalculate)
        soc_sensor_power = powers.get("SOC", 0.0)
        accounted_power = sum(p for k, p in powers.items() if k != "SOC")
        unaccounted = max(0, total_power - accounted_power)

        # For a 64-core ARM system, attribute ALL unaccounted power to SOC
        # The SOC sensor only measures a small auxiliary rail
        powers["SOC"] = soc_sensor_power + unaccounted

        # Ensure other components have minimal power if not measured
        if "PERIPHERAL" not in powers:
            powers["PERIPHERAL"] = 5.0  # Minimal baseline
        if "NVME" not in powers:
            powers["NVME"] = 5.0  # Minimal baseline
        if "SYSTEM" not in powers:
            powers["SYSTEM"] = 5.0  # Minimal baseline

        # Add SOC_VRD and DIMM_VRD power estimates based on actual sensors
        if "SOC" in powers:
            powers["SOC_VRD"] = (
                soc_sensor_power * 0.08
            )  # VRM losses based on aux power only

        if dimm_power > 0:
            powers["DIMM_VRD"] = dimm_power * 0.08  # VRM losses ~8% of DIMM power

        # VRD losses (inefficiency creates local heating)
        for component in ["CORE_VRD", "SOC"]:
            if component in powers:
                powers[component] *= 1.08  # 8% VRM inefficiency

        # Extract fan states
        fan_pwms = {}
        fan_rpms = {}

        for fan in self.fans:
            if fan.pwm_sensor in sensors:
                fan_pwms[fan.name] = sensors[fan.pwm_sensor][0]
            if fan.rpm_sensor in sensors:
                fan_rpms[fan.name] = sensors[fan.rpm_sensor][0]

        return SystemState(
            timestamp=time.time(),
            temperatures=temperatures,
            powers=powers,
            fan_pwms=fan_pwms,
            fan_rpms=fan_rpms,
            inlet_temp=inlet_temp,
        )

    def update_fan_speeds(self, controls: np.ndarray) -> bool:
        """Apply fan control commands via IPMI

        Args:
            controls: PWM values [0-1] for each fan

        Returns:
            Success status
        """
        success = True

        for fan, pwm in zip(self.fans, controls):
            duty_percent = int(pwm * 100)
            duty_percent = max(25, min(100, duty_percent))  # Enforce limits

            if not self.ipmi.set_fan_duty(fan.zone_id, duty_percent):
                self.logger.error("Failed to set %s to %d%%", fan.name, duty_percent)
                success = False
            else:
                self.logger.info("Set %s to %d%%", fan.name, duty_percent)

        return success

    def emergency_response(self, state: SystemState):
        """Handle emergency thermal conditions

        Sets all fans to maximum temporarily to cool down the system.
        The adaptive controller will learn from this event.
        """
        self.logger.critical("THERMAL EMERGENCY - Setting all fans to maximum")

        # Max all fans
        emergency_controls = np.ones(len(self.fans))
        self.update_fan_speeds(emergency_controls)

        # Log critical temperatures
        critical_nodes = []
        for node in self.thermal_nodes:
            if node.name in state.temperatures:
                temp = state.temperatures[node.name]
                if temp >= node.critical_temp:
                    self.logger.critical(
                        "%s: %.1f°C (CRITICAL: %.1f°C)",
                        node.name,
                        temp,
                        node.critical_temp,
                    )
                    critical_nodes.append(node.name)

        # Enter emergency mode
        if not self.emergency_mode:
            self.emergency_mode = True
            self.emergency_start_time = time.time()
            self.emergency_count += 1

        # Only exit if we've had too many emergencies in a short time
        if self.emergency_count > 10:
            self.logger.critical(
                "Too many thermal emergencies, shutting down for safety"
            )
            self.ipmi.set_fan_mode("OPTIMAL")
            self.running = False
        else:
            self.logger.info(
                "Emergency cooling active - system will learn from this event"
            )

    def run(self):
        """Main control loop"""
        self.logger.info("Thermal Fan Controller starting...")

        # Set IPMI to manual control mode
        if not self.ipmi.set_fan_mode("FULL"):
            self.logger.error("Failed to set IPMI to FULL mode")
            return

        # Initial fan spinup for safety check
        self.logger.info("Performing initial fan test...")
        self.update_fan_speeds(np.ones(len(self.fans)))
        time.sleep(3)

        # Set moderate initial speed
        initial_pwm = 0.35  # 35%
        self.update_fan_speeds(np.full(len(self.fans), initial_pwm))

        # Control loop timing
        control_interval = 15.0  # Read sensors AND update controls every 15 seconds
        model_save_interval = 60.0  # Save model every 60 seconds

        last_control_time = time.time()
        last_save_time = time.time()

        self.logger.info("Entering main control loop...")

        while self.running:
            try:
                loop_start = time.time()

                # Read current state
                sensors = self.ipmi.read_sensors()
                if not sensors:
                    self.logger.error("Failed to read sensors")
                    time.sleep(control_interval)
                    continue

                state = self.parse_sensor_data(sensors)
                self.state_history.append(state)

                # Check for emergency conditions
                emergency = False
                for node in self.thermal_nodes:
                    if node.name in state.temperatures:
                        if state.temperatures[node.name] >= node.critical_temp:
                            emergency = True
                            break

                if emergency:
                    self.emergency_response(state)
                    # Don't skip control update - let the system learn and adapt

                # Check if we can exit emergency mode
                if self.emergency_mode:
                    # Check if all temps are below warning levels
                    all_safe = True
                    for node in self.thermal_nodes:
                        if node.name in state.temperatures:
                            if state.temperatures[node.name] >= node.warning_temp - 5:
                                all_safe = False
                                break

                    # Require minimum emergency duration (30 seconds) to prevent oscillation
                    emergency_duration = time.time() - self.emergency_start_time
                    if all_safe and emergency_duration > 30:
                        self.emergency_mode = False
                        self.logger.info(
                            "Exiting emergency mode - temperatures are safe"
                        )
                    else:
                        self.logger.info(
                            "Staying in emergency mode (duration: %.0fs, max temp: %.1f°C)",
                            emergency_duration,
                            (
                                max(state.temperatures.values())
                                if state.temperatures
                                else 0
                            ),
                        )

                # Get current state arrays
                temps, powers, current_controls = state.to_arrays(
                    self.thermal_model.node_order,
                    self.thermal_model.fan_order,
                    power_defaults=POWER_DEFAULTS,
                )

                # Kalman filter update
                self.kalman.predict(current_controls, powers, state.inlet_temp)
                self.kalman.update(temps)

                # Control update - happens every iteration now (every 15 seconds)
                current_time = time.time()

                # Log all current temperatures
                temp_status = ", ".join(
                    [
                        f"{node}: {temp:.1f}°C"
                        for node, temp in zip(self.thermal_model.node_order, temps)
                    ]
                )
                self.logger.info("Current temps: %s", temp_status)

                # Log current fan RPMs
                rpm_status = ", ".join(
                    [
                        f"{fan}: {state.fan_rpms.get(fan, 0):.0f}rpm"
                        for fan in state.fan_rpms
                    ]
                )
                self.logger.info("Current RPMs: %s", rpm_status)

                # Log power readings
                power_status = ", ".join(
                    [
                        f"{node}: {power:.1f}W"
                        for node, power in zip(self.thermal_model.node_order, powers)
                    ]
                )
                self.logger.info("Power readings: %s", power_status)

                # Solve MPC for optimal controls
                # For now, assume constant power over horizon
                predicted_powers = np.tile(powers, (self.mpc.horizon, 1)).T

                # Check if any temperature is within 5% of critical
                critical_override = False
                for i, (temp, node) in enumerate(zip(temps, self.thermal_nodes)):
                    critical_margin = node.critical_temp * 0.05  # 5% of critical temp
                    if temp >= node.critical_temp - critical_margin:
                        self.logger.warning(
                            "CRITICAL OVERRIDE: %s at %.1f°C (critical-5%%: %.1f°C)",
                            node.name,
                            temp,
                            node.critical_temp - critical_margin,
                        )
                        critical_override = True
                        break

                if critical_override:
                    # Max all fans if any temp within 5% of critical
                    optimal_controls = np.ones(len(self.fans))  # 100%
                else:
                    # Normal MPC operation
                    self.logger.info("Solving MPC optimization...")
                    try:
                        optimal_controls = self.mpc.solve(state, powers)
                    except Exception as e:
                        self.logger.error("MPC solve failed: %s", e)
                        # Fall back to simple temperature-based control
                        max_temp = np.max(temps)
                        if max_temp > 70:
                            optimal_controls = np.full(
                                len(self.fans), 0.50
                            )  # 50% for high temp
                        elif max_temp > 65:
                            optimal_controls = np.full(
                                len(self.fans), 0.40
                            )  # 40% for warning
                        else:
                            optimal_controls = current_controls  # Keep current

                # Log what MPC decided
                self.logger.info(
                    "MPC decision: Current controls: %s -> Optimal: %s",
                    [f"{c*100:.0f}%" for c in current_controls],
                    [f"{c*100:.0f}%" for c in optimal_controls],
                )

                # Apply controls
                # Check if we need to change fan speeds
                controls_changed = not np.allclose(
                    current_controls, optimal_controls, atol=0.01
                )
                if controls_changed:
                    self.logger.info("Applying new fan speeds...")
                    if self.update_fan_speeds(optimal_controls):
                        last_control_time = current_time
                else:
                    self.logger.info(
                        "Fan speeds unchanged - maintaining current settings"
                    )

                # Log status
                max_temp_idx = np.argmax(temps)
                max_temp = temps[max_temp_idx]
                max_node = self.thermal_nodes[max_temp_idx].name

                fan_status = ", ".join(
                    [
                        f"{fan.name}: {int(u*100)}%"
                        for fan, u in zip(self.fans, optimal_controls)
                    ]
                )

                self.logger.info(
                    "Control update - Max: %s=%.1f°C | %s",
                    max_node,
                    max_temp,
                    fan_status,
                )

                # Periodic model save
                if current_time - last_save_time >= model_save_interval:
                    self._save_model()
                    last_save_time = current_time

                    # Log learned parameters
                    self.logger.info("Thermal model parameters (H matrix):")
                    for i, node in enumerate(self.thermal_nodes):
                        h_values = ", ".join(
                            [
                                f"{fan.name}: {self.thermal_model.H[i,j]:.3f}"
                                for j, fan in enumerate(self.fans)
                            ]
                        )
                        self.logger.info("  %s: %s", node.name, h_values)

                # Sleep until next iteration
                elapsed = time.time() - loop_start
                sleep_time = max(0, control_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error("Error in control loop: %s", e, exc_info=True)
                time.sleep(control_interval)

        # Cleanup on exit
        self.logger.info("Shutting down...")
        self.ipmi.set_fan_mode("OPTIMAL")
        self._save_model()


def main():
    """Entry point"""
    controller = ThermalFanController()
    controller.run()


if __name__ == "__main__":
    main()
