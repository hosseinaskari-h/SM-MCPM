# SM-MCPM
State-Modulated Monte Carlo Physarum Machine
# SM-MCPM: State-Modulated Monte Carlo Physarum Machine

This repository contains the reference implementation for **SM-MCPM**, a computational framework extending classical *Physarum polycephalum* models with behavioral state machines and metabolic energy constraints.

The associated paper, *"SM-MCPM: State-Modulated Monte Carlo Physarum Machine for Energy-Constrained Collective Navigation,"* details the methodology, mathematical formulation, and experimental results reproduced by this codebase.
Available preprint link : https://doi.org/10.13140/RG.2.2.13933.24803

## Overview

SM-MCPM introduces internal agent states (e.g., *Scouting*, *LowEnergy*, *Returning*) to traditional memoryless physarum models. These states dynamically modulate sensing weights and movement priorities based on environmental context and energy levels. The framework simulates emergent infrastructure formation, where collective movement patterns consolidate into persistent "vein" networks that offer metabolic efficiency advantages.

The simulation engine is implemented in Python and utilizes [Taichi Lang](https://github.com/taichi-dev/taichi) for GPU-accelerated parallel processing.

## Requirements

* Python 3.8 or higher
* Taichi
* NumPy
* Matplotlib
* PyYAML

To install the necessary dependencies:

```bash
pip install taichi numpy matplotlib pyyaml
```

## Usage

### Reproducing Experiments

The `experiments.py` script is configured to replicate the specific scenarios described in the paper (Tables 4–5 and Figures 2–3). It performs the following:

1.  **Full System Analysis:** Runs $N=10$ trials of the complete SM-MCPM model.
2.  **Ablation Study:** Runs $N=10$ trials of a baseline model without state modulation.
3.  **Infrastructure Analysis:** Compares performance between environments with and without recharge nodes.

To execute the full suite of experiments:

```bash
python experiments.py

```
Results, including high-resolution figures and statistical summaries, will be output to the outputs/ directory.
