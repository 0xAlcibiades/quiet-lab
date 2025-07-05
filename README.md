# quiet-lab 🔇

*Physics-based adaptive fan control for the Supermicro ARS-210M-NR server*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hardware: ARS-210M-NR](https://img.shields.io/badge/hardware-ARS--210M--NR-green.svg)](https://www.supermicro.com/en/products/system/datasheet/ars-210m-nr)

## Why quiet-lab?

Your Supermicro ARS-210M-NR came with fan curves designed for worst-case datacenter scenarios. quiet-lab learns YOUR specific setup and runs as quiet as physics allows.

### The Problem
- Datacenter fan curves assume 40°C ambient and maximum load 24/7
- Your homelab is probably 20°C ambient with <10% average load
- Result: Unnecessary noise, wasted power, and unhappy family members

### The Solution
quiet-lab uses **Model Predictive Control (MPC)** and **Extended Kalman Filtering** to:
1. Learn your server's actual thermal dynamics
2. Predict temperature changes before they happen
3. Optimize fan speeds to minimize noise while maintaining safety
4. Adapt to seasonal changes and hardware aging

## Results

From real-world deployment on a 64-core Ampere Altra server:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Idle Fan Speed | 80-100% | 25-35% | **-65%** |
| Noise Level | 65 dB | 42 dB | **-23 dB** |
| Power (Fans) | 120W | 15W | **-87%** |
| Temperature Prediction | N/A | ±0.96°C | **Learns your system** |

## Features

- 🧮 **Model Predictive Control** - Anticipates thermal changes before they happen
- 📈 **Self-Learning** - Adapts to your specific hardware via Kalman filtering
- 🔇 **Multi-Objective Optimization** - Balances temperature, power, and acoustics
- 🛡️ **Multiple Safety Layers** - Hardware limits, software limits, and failsafe modes
- 📊 **Production Mode** - Stops learning once converged for minimal overhead
- ⚡ **Efficient** - <1% of one core usage after convergence

## Quick Start

### Requirements
- **Supermicro ARS-210M-NR server** (currently the only supported model)
- Linux OS with systemd
- Python 3.8+
- `ipmitool`, `python3-numpy`, `python3-scipy` installed
- Root access (for IPMI commands)

### Installation

```bash
# Clone the repository
git clone https://github.com/0xalcibiades/quiet-lab.git
cd quiet-lab

# Run the installation script (installs dependencies automatically)
sudo ./install-fan-control.sh

# Start the service
sudo systemctl start fan-control

# Check status
sudo systemctl status fan-control
sudo journalctl -u fan-control -f
```

The installation script will:
- Install required packages (ipmitool, python3-numpy, python3-scipy)
- Copy the fan control script to `/usr/local/bin/`
- Install and enable the systemd service
- Create necessary directories for logs and saved models

**Note**: The learned thermal model is saved at `/var/lib/thermal_model.pkl`

## How It Works

quiet-lab implements a sophisticated control system based on first-principles thermodynamics, online parameter learning, and predictive optimization. Here's the detailed mathematical foundation:

### 1. Physics-Based Thermal Model

The server is modeled as a network of thermal nodes, each representing a component (CPU, DIMM, VRM, etc.). The heat transfer dynamics follow Newton's law of cooling with forced convection:

$$
C \frac{dT}{dt} = Q(t) - H(u) \cdot (T - T_{\text{inlet}})
$$

Where:
- $C = \text{diag}(C_1, ..., C_n) \in \mathbb{R}^{n \times n}$ — Thermal capacitance matrix [J/K]
- $T = [T_1, ..., T_n]^{\top} \in \mathbb{R}^n$ — Temperature state vector [K]
- $Q(t) = [Q_1(t), ..., Q_n(t)]^{\top} \in \mathbb{R}^n$ — Heat generation vector [W]
- $H(u) \in \mathbb{R}^{n \times n}$ — Heat transfer matrix (diagonal) [W/K]
- $u = [u_1, ..., u_m]^{\top} \in [0,1]^m$ — Fan PWM control vector
- $T_{\text{inlet}} \in \mathbb{R}$ — Ambient inlet temperature [K]

The heat transfer matrix has diagonal structure:

$$H_{ii}(u) = \sum_{j=1}^{m} H_{ij}^{\max} \cdot u_j^{\alpha}$$

where $\alpha \approx 0.8$ captures the turbulent flow regime typical in server cooling.

### 2. Extended Kalman Filter for Online Learning

The EKF jointly estimates both system states (temperatures) and parameters (heat transfer coefficients). The augmented state vector is:

$$\xi = \begin{bmatrix} T \\ \text{vec}(H^{\max}) \end{bmatrix} \in \mathbb{R}^{n + nm}$$

**Prediction Step:**

$$\xi_{k|k-1} = f(\xi_{k-1|k-1}, u_k, Q_k)$$

$$P_{k \mid k-1} = F_k P_{k-1 \mid k-1} F_k^{\top} + G_k W G_k^{\top}$$

where $F_k = \frac{\partial f}{\partial \xi}\Big|_{\xi_{k-1|k-1}}$ is the Jacobian of the state transition.

**Update Step:**

$$K_k = P_{k \mid k-1} C^{\top} (C P_{k \mid k-1} C^{\top} + R)^{-1}$$

$$\xi_{k|k} = \xi_{k|k-1} + K_k(z_k - C\xi_{k|k-1})$$

$$P_{k \mid k} = (I - K_kC)P_{k \mid k-1}$$

The innovation $\nu_k = z_k - C\xi_{k|k-1}$ drives both state correction and parameter learning.

### 3. Model Predictive Control

MPC solves a finite-horizon optimal control problem every 15 seconds:

$$
\min_{U} \sum_{k=0}^{N-1} \ell(x_k, u_k) + \ell_N(x_N)
$$

where $U = [u_0^{\top}, ..., u_{N-1}^{\top}]^{\top} \in \mathbb{R}^{Nm}$

Subject to:
- System dynamics: $x_{k+1} = f_d(x_k, u_k, w_k)$
- Box constraints: $u_{\min} \preceq u_k \preceq u_{\max}$
- Rate constraints: $\lVert u_k - u_{k-1} \rVert_{\infty} \leq \Delta u_{\max}$

The stage cost $\ell(x, u)$ balances multiple objectives:

#### Temperature Cost

$$
\ell_{\text{temp}}(T) = \sum_{i=1}^{n} \phi_i(T_i - T_{i}^{\text{target}})
$$

where the penalty function is:

$$
\phi_i(\Delta T) = \begin{cases}
0 & |\Delta T| \leq \delta \\
w_{\text{under}} \cdot (\Delta T + \delta)^2 & \Delta T < -\delta \\
w_{\text{over}}(i) \cdot (\Delta T - \delta)^2 & \Delta T > \delta
\end{cases}
$$

with urgency scaling: $w_{\text{over}}(i) = w_0 \cdot (1 + 9\sigma_i^3)$ where $\sigma_i = \frac{T_i - T_i^{\text{warn}}}{T_i^{\text{crit}} - T_i^{\text{warn}}}$

#### Acoustic Cost

The psychoacoustic model captures human perception of fan noise:

$$L_{fan,j} = L_{ref} + 50\log_{10}\left(\frac{\omega_j}{\omega_{ref}}\right) \quad \text{[dB SPL]}$$

$$L_{total} = 10\log_{10}\left(\sum_{j=1}^{m} 10^{L_{fan,j}/10}\right) \quad \text{[dB SPL]}$$

$$\Lambda = 2^{(L_{total} - 40)/10} \quad \text{[sones]}$$

$$
\ell_{\text{acoustic}}(u) = w_{\text{noise}} \cdot (\Lambda - 1)^2
$$

#### Power Cost

$$
\ell_{\text{power}}(u) = w_{\text{power}} \sum_{j=1}^{m} u_j^3
$$

### 4. Control System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    IPMI Hardware Interface                 │
│                 ┌─────────────┬─────────────┐              │
│                 │   Sensors   │  Actuators  │              │
│                 └──────┬──────┴──────┬──────┘              │
└────────────────────────┼─────────────┼─────────────────────┘
                         │ ΔT = 15s    │
┌────────────────────────▼─────────────▼─────────────────────┐
│                  Extended Kalman Filter                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ State Prediction:  x̂ₖ₊₁ = f(x̂ₖ, uₖ, wₖ)             │    │
│  │ Covariance Update: Pₖ₊₁ = FₖPₖFₖᵀ + GₖWGₖᵀ           │    │
│  │ Innovation:        νₖ = zₖ - Cx̂ₖ                    │    │
│  │ Kalman Gain:       Kₖ = PₖCᵀ(CPₖCᵀ + R)⁻¹           │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────┬───────────────────────────────────┘
                         │ (T̂, Ĥ)
┌────────────────────────▼───────────────────────────────────┐
│              Model Predictive Controller                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Solve: min Σ ℓ(xₖ, uₖ) s.t. dynamics & constraints  │    │
│  │        U                                           │    │
│  │ Horizon: N = 3 steps (45 seconds lookahead)        │    │
│  │ Solver: Sequential Quadratic Programming (SLSQP)   │    │
│  └────────────────────────────────────────────────────┘    │
└────────────────────────┬───────────────────────────────────┘
                         │ u*₀
┌────────────────────────▼───────────────────────────────────┐
│                   Safety Governor                          │
│  • Emergency override: T > Tᶜʳⁱᵗ - ε → u = 1               │
│  • Rate limiting: clip(Δu, -Δuₘₐₓ, Δuₘₐₓ)                  │
│  • Saturation: project(u, [0.25, 1.0])                     │
└────────────────────────────────────────────────────────────┘
```

### 5. Learning Dynamics and Convergence

The system exhibits distinct learning phases characterized by the Frobenius norm of the heat transfer matrix updates:

$$\lVert \Delta H \rVert_F = \sqrt{\sum_{i,j} \left( H^k_{ij} - H^{k-1}_{ij} \right)^2}$$

**Phase 1: Exploration** (t ∈ [0, 30 min])
- $\lVert \Delta H \rVert_F > 1.0$ — Rapid initial learning
- Conservative control: $u \approx 0.5 \cdot 1$

**Phase 2: Refinement** (t ∈ [30 min, 2 hr])
- $0.1 < \lVert \Delta H \rVert_F < 1.0$ — Convergence to true parameters
- Aggressive optimization begins

**Phase 3: Adaptation** (t > 2 hr)
- $\lVert \Delta H \rVert_F < 0.1$ — Fine-tuning for load variations
- Optimal steady-state operation

The learned heat transfer matrix $H^{\max} \in \mathbb{R}^{n \times m}$ encodes:
- Primary cooling paths (large $H_{ij}$)
- Cross-component thermal coupling
- Non-intuitive airflow patterns unique to the chassis

## Safety Features

1. **Pre-start**: Fans set to 100% before service starts
2. **Graceful shutdown**: Returns to BIOS control on stop
3. **Crash protection**: Automatic failover to safe mode
4. **Temperature limits**: Hard-coded critical thresholds
5. **Watchdog**: Detects hung processes
6. **Minimum fan speeds**: Prevents stalling (25% minimum)

## Hardware Support

### Currently Implemented For
- **Supermicro ARS-210M-NR** (64-core Ampere Altra)
  - All thermal nodes and fan zones are hardcoded for this specific model
  - Sensor names match the ARS-210M-NR IPMI interface
  - Thermal capacitance values tuned for this hardware

### Generalizing to Other Servers
While the current implementation is specific to the ARS-210M-NR, the underlying physics-based approach and control algorithms can be adapted to any server with:
- IPMI 2.0+ for sensor reading and fan control
- Temperature sensors for critical components
- PWM-controllable fans with zones
- Power consumption measurements (optional but helpful)

To adapt for a different server, you would need to:
1. Map your server's IPMI sensor names in the `thermal_nodes` configuration
2. Identify your fan zones and PWM control addresses
3. Adjust thermal capacitance values for your components
4. Update the initial H matrix estimates based on your server's airflow patterns

The core algorithms (Extended Kalman Filter, Model Predictive Control, thermal physics model) remain the same.

## Contributing

Contributions are welcome

## FAQ

**Q: How long until it learns my system?**  
A: 50% learning in ~30 minutes, 95% in 2-3 hours, fully converged in 4-6 hours.

**Q: What if it fails?**  
A: Fans automatically return to BIOS control. Your hardware's built-in thermal protection remains active.

**Q: Does it work with other servers?**  
A: Currently hardcoded for the Supermicro ARS-210M-NR, but the physics-based approach generalizes. See the Hardware Support section for adaptation requirements.

**Q: Power savings?**  
A: Typical 10W per fan reduction from fan power alone, plus better overall cooling efficiency.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by higher WAF
- Built with NumPy, SciPy, and 🔇
- Special thanks to everyone who said "that's sounds like a giant mosquito in your garage"

---

*The best fan controller is the one you forget exists*