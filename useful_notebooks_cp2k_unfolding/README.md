# `useful_notebooks_cp2k_unfolding`

Reusable helpers for CP2K localized-basis band unfolding notebooks.

Main modules:

- `io.py`: CP2K WFN loading through `cp2k-spm-tools`, overlap-matrix parsing, XYZ parsing, and CP2K cell parsing.
- `geometry.py`: primitive/supercell vector inference and AO mapping.
- `unfolding.py`: sparse non-orthogonal unfolding implementation.
- `kpath.py`: folded k-point generation and standard 1D/2D k-path helpers.
- `widgets.py`: notebook widgets for primitive-cell review and correction.
- `plotting.py`: unfolded band plotting.

`cp2k-spm-tools` is required for reading CP2K `.wfn` files.
