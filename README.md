# useful-notebooks

A collection of utility Jupyter notebooks for common post-processing and analysis tasks in atomistic simulations. The notebooks are organized by topic and are intended as practical starting points rather than fully polished workflows.

This `README` is a first draft based on the current repository structure and notebook contents. The descriptions below are intentionally conservative so they can be adapted easily.

## Repository organization

The repository is currently organized into the following folders:

- `AiidaPostProcess/` – notebooks related to extracting or collecting results from AiiDA workflows.
- `ChargeAnalysis/` – notebooks for charge-related analysis, including integrated charge profiles and population analysis.
- `CubeFiles/` – notebooks for reading, integrating, and plotting data from Gaussian cube files.

## Requirements

Depending on the notebook, you may need some or all of the following:

- Python 3
- Jupyter Notebook or JupyterLab
- `numpy`
- `matplotlib`
- `scipy`
- `aiida-core` and an active AiiDA profile for notebooks that load AiiDA nodes
- SSH / SCP access to remote machines for notebooks that retrieve files from HPC systems

## AiidaPostProcess

Draft description: notebooks in this folder are meant to help retrieve, inspect, or post-process files associated with AiiDA workflows, especially when the relevant outputs remain on remote HPC storage.

### `GetSpinDensityFromGeoOpt.ipynb`

Draft description: retrieves volumetric cube files associated with a geometry-optimization workflow from the remote working directory. In its current form, the notebook loads an AiiDA node, finds the corresponding remote folder, and copies selected cube files (for example spin-density and electron-density files) to a local downloads directory via `scp`.

In use by:
-Deborah

Potential future refinements:
- generalize the workflow / node selection
- make the list of file patterns configurable
- add error handling when no matching files are found

### `NscfEigenvalues.ipynb`
Draft description: starting form the PK of a QE app workchain, idenfifyes the NSCF calculation, reads the eigenvalues and plot a DOS based solely on the eigenvalues. HOMO and LUMO values are provided according to teh number of electrons.

In use by:
- Suyash
- Saketh

## ChargeAnalysis

Draft description: notebooks in this folder focus on charge partitioning, cumulative charge analysis, and related post-processing derived either from CP2K outputs or from cube files.

### `ChargeTransfer.ipynb`

Draft description: reads charge-density cube files and computes charge quantities integrated along the `z` direction. The notebook appears to support comparison between different systems or fragments, interpolation-based queries such as finding `z` for a target cumulative charge (and vice versa), and generation of charge-density-difference cube files by subtracting component densities.

Potential future refinements:
- clarify the physical system and naming conventions used in the example inputs
- factor common cube I/O utilities into a shared helper module
- document assumptions on orthogonality and alignment of the simulation cell

### `PopulationAnalysis.ipynb`

Draft description: extracts and sums atomic charges from CP2K population-analysis output stored in `aiida.out` within an AiiDA retrieved folder. The notebook includes utilities to parse flexible atom-index specifications and to compute total net charge over selected atoms for different analysis schemes such as Mulliken and Hirshfeld.

Potential future refinements:
- add examples for mixed atom ranges and individual atom selections
- support additional charge-analysis sections if needed
- package the parsing and summation logic into reusable functions or a small module

## CubeFiles

Draft description: notebooks in this folder provide small utilities for inspecting scalar fields stored in Gaussian cube files, with emphasis on one-dimensional profiles and integral consistency checks.

### `ChargeAverage.ipynb`

Draft description: reads a charge-density cube file and computes the in-plane integrated charge profile as a function of `z`. The notebook then plots the resulting line density, checks that its integral reproduces the total charge, and compares the one-dimensional integral against the full three-dimensional volume integral.

In use by:
- Deborah

Potential future refinements:
- make input-file selection more user friendly
- support non-orthogonal cells if needed
- add optional normalization or planar-average modes

## Suggested next improvements for this repository

A few natural improvements for the repository documentation could be:

1. add a short top-level paragraph describing the scientific context of the notebooks
2. specify whether notebooks are examples, production utilities, or exploratory analyses
3. add a small usage note at the top of each notebook describing inputs, assumptions, and outputs
4. move shared helper functions into a Python module if code duplication starts growing

## License

This repository includes a `LICENSE` file. If desired, a short sentence summarizing the license type can be added here.
