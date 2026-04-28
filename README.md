# Ion Channel Degeneracy Underlies Pacemaker Identity

**Large-scale computational evidence from a biophysically constrained Hodgkin-Huxley framework**

[![DOI](https://img.shields.io/badge/bioRxiv-DOI_will_be_added -brightgreen)](https://doi.org/10.1101/will_be_added)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Preprint

Kar, B.S. (2026). *Ion Channel Degeneracy Underlies Pacemaker Identity: Large-Scale Computational Evidence from a Biophysically Constrained Hodgkin-Huxley Framework*. bioRxiv. DOI: will be added.

## Overview

This repository contains the simulation code and validated dataset for a large-scale computational study demonstrating that pacemaker neuronal identity is an emergent property of ion channel composition space rather than the presence of specific ion channels.

### Key Numbers

| Metric | Value |
|--------|-------|
| Simulated cells | 1,500,000 |
| Validated pacemakers | 390,022 |
| Ion channel types | 21 |
| Firing frequency range | 0.2–264.6 Hz |
| Ca channel inter-correlation | mean \|r\| = 0.004 |
| Pacemakers without Kv1 | 29.4% (n = 114,776) |
| Pacemakers without HCN | 24.0% (n = 93,648) |
| Double knockout (HCN⁻/Kv1⁻) | 26,375 cells at 4.10 Hz median |

## Repository Contents

### Code
- `UniversalCell_v5.cpp` — Full 21-channel Hodgkin-Huxley simulator with:
  - Dynamic Nernst potentials (Na⁺, Ca²⁺)
  - Full Ca²⁺/Na⁺ homeostasis (PMCA, SERCA, NCX, Na⁺/K⁺-ATPase)
  - Compartment-specific weights (soma/dendrite)
  - Firing mode classification (spontaneous/evokable/silent)
  - Phase-based parameter configurations (A: pacemaker, B: burst, C: inhibitory, D: general)

### Data
- `validated_cells.csv` — 390,022 validated pacemaker cells with all 21 conductance values, firing frequency, ISI CV, V_max, V_min, phase ID, and firing mode.

## Build & Run

### Requirements
- C++17 compiler (GCC 9+ or MinGW-w64)
- OpenMP

### Compilation
```bash
g++ -std=c++17 -O3 -fopenmp UniversalCell_v5.cpp -o uc5

./uc5 1500000 ## Simulates 1.5 million cells across four phases and outputs results to CSV.

Data Columns
validated_cells.csv contains the following columns:

Column	Description
g_NaF – g_NCX	Maximal conductances (mS/cm²) for 21 channels
Freq_Hz	Firing frequency
ISI_CV	Inter-spike interval coefficient of variation
V_max	Peak voltage (mV)
V_min	Trough voltage (mV)
Na_i_final	Final intracellular Na⁺ concentration (mM)
phase_id	A_pacemaker or B_burst
compartment_mode	single
firing_mode	spontaneous, evokable, or silent
Note
The simulator supports four phases. Only Phase A (pacemaker-type, n=335,014) and a subset of Phase B (burst-type, n=54,994) were executed for the preprint analysis, totaling 390,022 validated pacemaker cells.

Cross-Validation Data
Human SNc snRNA-seq data from Kamath et al. (2022) is publicly available at:
https://cellxgene.cziscience.com/collections/2856d06c-0ff9-4e01-bfc9-202b74d0b60f

Citation
If you use this code or data, please cite:
Kar, B.S. (2026). Ion Channel Degeneracy Underlies Pacemaker Identity. bioRxiv. DOI: [will be added]

License
This project is licensed under the MIT License — see the LICENSE file for details.
