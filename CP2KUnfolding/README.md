# CP2K band unfolding

This folder contains a prototype notebook for unfolding band structures from CP2K supercell calculations with localized Gaussian basis functions.

The original prototype notebook expects the following files in the working directory:

- `aiida-RESTART.wfn`
- `overlap_matrix.out`
- `aiida.coords.xyz`
- `aiida.inp`
- optionally `aiida.out`

The reusable implementation and command-line tools live in `cp2k-spm-tools`, under `cp2k_spm_tools.cp2k_unfolding`. This folder contains only the notebook frontend and its local regression harness in `manual_tests/`.

Current scope:

- Gamma-only CP2K supercells
- real and complex-compatible coefficient handling
- sparse overlap-matrix parsing and sparse projector assembly
- MPI unfolding over molecular-orbital blocks
- explicit user-selected primitive basis atoms
- atom-to-primitive-site displacement diagnostics
- automatic 1D/2D primitive-cell guess from geometry as a fallback
- editable primitive/supercell widgets
- standard 1D/2D k-path projection
- unfolded band plot with marker size proportional to spectral weight

The overlap matrix is kept sparse. Dense arrays are formed only for the small primitive-cell Bloch overlap metric `S(k)` and for optional diagnostics.
