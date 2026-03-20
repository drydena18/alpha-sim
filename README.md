# EEG Alpha-Wave Simulator

Two coupled van der Pol oscillators modelling cortical alpha rhythms (8–13 Hz),
visualised in real time. The C engine runs RK4 integration; Python/PyQtGraph
handles all rendering and controls.

## Quick start

### 1. Install Python dependencies
```
pip install pyqtgraph PyQt5 numpy
```

### 2. Compile the C engine
**Linux / macOS**
```
gcc -O2 -march=native -shared -fPIC -o vdp_core.so vdp_core.c -lm
```

**macOS (Apple Silicon)**
```
gcc -O2 -arch arm64 -shared -fPIC -o vdp_core.so vdp_core.c -lm
```

**Windows (MinGW)**
```
gcc -O2 -shared -o vdp_core.dll vdp_core.c -lm
```
Then change `vdp_core.so` → `vdp_core.dll` in `eeg_sim.py` line 28.

### 3. Run
```
python eeg_sim.py
```

---

## What you're looking at

| Panel | Description |
|---|---|
| **Oscillator A / B** | Raw waveform of each alpha generator (arbitrary μV units) |
| **Phase diff A − B** | Instantaneous phase difference. Flat near 0 = in-phase sync; flat near ±π = anti-phase lock; wandering = desynchronised |
| **Sync R** | Kuramoto order parameter (0–1). R > 0.9 = strong synchrony |

## Parameters

| Parameter | Physiology |
|---|---|
| **Freq (Hz)** | Natural alpha frequency; ~10 Hz typical, slows with drowsiness |
| **Damping μ** | Van der Pol nonlinearity — controls how tightly the limit cycle is enforced |
| **Amplitude** | Limit-cycle radius, proxying cortical excitability |
| **Alertness** | Shifts frequency upward (up to +1.5 Hz); models arousal-driven alpha acceleration |
| **Coupling K** | Diffusive inter-oscillator coupling; mimics long-range cortico-cortical synchronisation |
| **Noise σ** | Additive Gaussian noise; represents stochastic synaptic input |

## Model equations

Each oscillator follows the van der Pol equation with diffusive coupling:

```
ẍ_A = μ_A(a_A² − x_A²)ẋ_A  −  ω_A²·x_A  +  K(x_B − x_A)  +  σ·η
ẍ_B = μ_B(a_B² − x_B²)ẋ_B  −  ω_B²·x_B  +  K(x_A − x_B)  +  σ·η

ω = 2π(f₀ + alertness × 1.5)
```

Integration: 4th-order Runge-Kutta, dt = 1 ms.

## Extending the model

- **Power spectrum**: FFT `_arr_a` and plot with `np.fft.rfft`
- **Third oscillator**: add a thalamic pacemaker driving both A and B
- **Real EEG input**: replace `sim_advance` with a live MNE-Python stream; use the
  C engine only for the coupling/sync computation
- **More channels**: duplicate the OscState struct; the RK4 generalises to N oscillators