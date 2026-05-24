# CP2K band unfolding

This folder contains a prototype notebook for unfolding band structures from CP2K supercell calculations with localized Gaussian basis functions.

The notebook expects the following files in the working directory:

- `aiida-RESTART.wfn`
- `overlap_matrix.out`
- `aiida.coords.xyz`
- `aiida.inp`
- optionally `aiida.out`

The reusable implementation lives in `useful_notebooks_cp2k_unfolding/`.

Current scope:

- Gamma-only CP2K supercells
- real and complex-compatible coefficient handling
- sparse overlap-matrix parsing and sparse projector assembly
- automatic 1D/2D primitive-cell guess from geometry
- editable primitive/supercell widgets
- standard 1D/2D k-path projection
- unfolded band plot with marker size proportional to spectral weight

The overlap matrix is kept sparse. Dense arrays are formed only for the small primitive-cell Bloch overlap metric `S(k)` and for optional diagnostics.
