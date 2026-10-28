# Two Faces of Symmetry Match in Gear-Like Coupled Rotary Motors

Simulation and analysis code accompanying the manuscript:

**Two faces of symmetry match in gear-like coupled rotary motors**

## Overview

This repository contains numerical simulations and analysis scripts for a model of two coupled rotary motors with gear-like interactions.

The model investigates how the symmetry of the two rotors affects power transmission under different driving protocols.

## Models

Both models use gear-like coupling.

### Model 3: Constant Driving

- `model3`
- Constant upstream driving force as upstream symmetry varies.
- Constant downstream driving force.

### Model 4: Scaling Driving

- `model4`
- Upstream driving force scales with upstream symmetry.
- Downstream driving force scales with downstream symmetry.

## Repository Structure

- `simulations/main_model3.py`: Main simulation script for constant driving.
- `simulations/fpe_model3.pyx`: Cython implementation for constant driving.
- `simulations/calculate_model3.py`: Flux, power, and efficiency calculations.
- `simulations/move_model3.py`: Simulation output organization.

- `simulations/main_model4.py`: Main simulation script for scaling driving.
- `simulations/fpe_model4.pyx`: Cython implementation for scaling driving.
- `simulations/calculate_model4.py`: Flux, power, and efficiency calculations.
- `simulations/move_model4.py`: Simulation output organization.

- `simulations/utilities.pyx`: Shared numerical utilities.
- `Processing_plots.ipynb`: Data analysis and plotting notebook.

## Requirements

The simulations use Python, NumPy, SciPy, pandas, Cython, and Matplotlib.

Some scripts may require configuration of input/output directories before running.

## Manuscript

Manuscript source files and figures will be added separately.
