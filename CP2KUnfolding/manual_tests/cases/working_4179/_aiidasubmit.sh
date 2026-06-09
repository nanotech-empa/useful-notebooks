#!/bin/bash

# Lightweight metadata-only example copied from an AiiDA unfolding calculation.
# Populate parent_calc_folder/ with the real CP2K files before running.

cp2k-unfold-wfn-sparse \
  parent_calc_folder/aiida-RESTART.wfn \
  parent_calc_folder/aiida-overlap_matrix.out-1_0.Log \
  unfolding_bands.npz \
  --xyz parent_calc_folder/aiida.coords.xyz \
  --cp2k-input parent_calc_folder/aiida.inp \
  --primitive-vectors "2.51 0.0 0.0; 1.26 2.17 0.0" \
  --path G-K-M-G \
  --lattice-type auto \
  --overlap-format log \
  --overlap-threshold 1e-10
