# Integrating Neuroplasticity into Whole-brain Modeling: from Synapses to Neural Populations 

This repository contains the code necessary to generate figures ilustrating the implementation of various neuroplasticity mechanisms within Neural Field Models (NFMs).

The code is organized into two directories, one with MATLAB scripts and one with Python scripts. The Python folder has an additional script to generate the Schirner et al. E/I balance tuning algorithm using numba to greatly speed up the processing time and therefor requires numba to be installed to run.

## Implemented Plasticity Rules
The simulations cover a broad range of biological plasticity mechanisms:

**Short-Term Plasticity (STP)**
- Kilpatrick-Bressloff (Neural Field Heuristic STD): Spatially continuous depression where available resources Q(x) modulate synaptic weight.
- Taher et al. (Exact QIF Mean-Field STP): Dynamic gain control via neurotransmitter depletion and calcium accumulation.
- Gast et al. (Exact QIF Mean-Field STP): Resource depletion driven by individual presynaptic spike trains to account for heterogeneity

**Long-Term Plasticity**
- Milstein et al. (Bidirectional Behavioral Timescale Plasticity): Weight-dependent LTP and LTD via eligibility traces and instructive signals.
- Fung & Robinson (BCM Calcium-dependent metaplasticity): Calcium concentration [Ca2+] dynamics driving synaptic weight changes.
- Fennelly et al. / Duchet et al. (Exact Mean-Field PDDP): Coupling strength updates based on the Kuramoto order parameter *z*


**Homeostatic Plasticity**
- Abeysuriya et al. (Inhibitory Plasticity): Local inhibitory strength changes to regulate excitatory firing rates.
- Stasinski et al. (Homeodynamic Feedback): Two-step homeostatic tuning in Jansen-Rit NMM networks.
- Deco et al. (FIC): Adjustment of local inhibition Ji​ to clamp excitatory firing rates at specific targets.
- Schirner et al. (E/I Balance): Tuning of excitation-inhibition balance by adjusting the ratio of long-range excitation (w^LRE) to feedforward inhibition .

**Structural Plasticity**
- Stam et al. (SDP and GDP): Phase-driven LTP/LTD combined with distance-based structural growth.
- Diaz-Pier et al. (Structural Plasticity): Growth or retraction of "synaptic elements" to maintain a calcium homeostatic target.
