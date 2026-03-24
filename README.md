# Alpha Interaction Sandbox

An interactive visualization and replay framework for exploring **alpha-band dynamics** and **slow-fast alpha interactions** in EEG data.

This project combines:
- A **dynamical systems simulator** (C backend)
- A **real-time visualization GUI** (Python + PyQtGraph)
- A modular pipeline for **EEG preprocessing, spectral extraction, and time-resolved replay** (in development)

---

## Purpose
This tool was built to:
- Explore how **slow (8–10 Hz)** and **fast (10–12 Hz)** alpha rhythms interact
- Visualize **time-resolved alpha dynamics** in an intuitive way
- Provide a framework for replaying **empirically-derived EEG features**
- Support development and communication of **novel alpha interaction metrics**

> WARNING: This is **not the basis of scientific conclusions**.
> It is a **visualization and exploratory framework** designed to compliment empirical analysis.

---

## Features

### Current (Demo Mode)
- Two coupled oscillators simulating alpha rhythms
- Adjustable parameters:
  - Frequency
  - Amplitude
  - Damping (μ)
  - Coupling strength
  - Noise
  - Alertness modulation
- Real-time visualization of:
  - Signal A / B
  - Phase difference
  - Interaction metrics (e.g., Kuramoto R)

### In Progress
- Load EEGLAB `.set` files
- Configurable preprocessing pipeline:
  - Bandpass / notch filtering
  - Re-referencing
  - ICA (optional)
  - Epoching / baseline correction
- Channel / ROI selection
- Spectral feature extraction:
  - Slow vs Fast alpha power
  - Peak Alpha Frequency (PAF)
  - Center of Gravity (CoG)
- Time-resolved replay of alpha metrics
- Modular interaction metric framework

---

## Project Structure
```text
alpha_interaction_app/
├── eeg_sim.py              # Main GUI application
├── vdp_core.c             # C simulation backend
├── app/                   # GUI + controller logic
├── eeg/                   # EEG processing pipeline (in development)
├── configs/               # YAML configs for preprocessing / metrics
├── data/                  # Local data (ignored by git)
└── docs/                  # Notes, architecture, plans
```

---

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/drydena18/alpha-sim.git
cd alpha-sim
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Minimal dependencies:
- numpy
- pyqtgraph
- PyQt5

Future EEG support:
- mne
- scipy

### 3. Build the simulation backend
```bash
gcc -O2 -march=native -shared -fPIC -o vdp_core.so cdp_core.c -lm
```
macOS (Apple Silicon):
```bash
gcc -O2 0arch arm64 -shared -fPIC -o vdp_core.so vdp_core.c -lm
```
Windows(MinGW):
```bash
gcc -O2 -shared -o vdp_core.dll vdp_core.c -lm
```

### 4. Run the App
```bash
python eeg_sim.py
```

---

## How to Use
### Demo Mode
- Adjust parameters
- Observe:
  - synchronization
  - phase dynamics
  - amplitude interactions
- Switch between available interaction metrics

### EEG Replay Mode (coming soon)
- Load .set file
- Apply preprocessing
- Select ROIs
- Replay alpha dynamics over time

---

## Scientific Context
This project is motivated by research investigating whether:
> The interaction between slow and fast alpha oscillations predicts individual differences in perceptual or behavioural outcomes (e.g., pain sensitivity).

The simulator provides a controlled environment to:
- Test candidate interaction metrics
- Visualize dynamic relationships
- Communicate complex oscillatory behaviour

---

## Roadmap
- EEG Loader (.set via MNE)
- Config-driven preprocessing pipeline
- ROI/channel selection
- Spectral feature extraction module
- Alpha interaction metric framework
- Replay pipeline (time-resolved packets)
- Export tools (figures, CSVs)
- Documentation + examples

---

## Disclaimer
This tool is intended for:
- Visualization
- Exploration
- Communication

**It does not replace formal statistical analysis and should not be used to draw scientific conclusions without proper validation.**

---

## License

MIT License

---

## Author
Developed by Dryden Arseneau  
M.Sc. Neuroscience Candidate – Cognitive Neuroscience of Pain  
University of Western Ontario, London, ON

---

## Future Directions
- Real-time EEG streaming (BCI-adjacent)
- Source-space integration (e.g., sLORETA / DeepSIF)
- Multi-channel network dynamics
- Advanced coupling metrics (nonlinear, subject-specific)

---

## If you find this useful
Feel free to:
- Star the repo
- Open issues
- Suggest features
- Contribute