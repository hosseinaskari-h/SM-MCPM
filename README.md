# SM-MCPM
State-Modulated Monte Carlo Physarum Machine
# SM-MCPM: State-Modulated Monte Carlo Physarum Machine

This repository contains the reference implementation for **SM-MCPM**, a computational framework extending classical *Physarum polycephalum* models with behavioral state machines and metabolic energy constraints.

The associated paper, *"SM-MCPM: State-Modulated Monte Carlo Physarum Machine for Energy-Constrained Collective Navigation,"* details the methodology, mathematical formulation, and experimental results reproduced by this codebase.

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
