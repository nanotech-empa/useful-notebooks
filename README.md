# useful-notebooks

A collection of practical Jupyter notebooks for post-processing atomistic-simulation data, together with a small shared Python package used by the cube-related notebooks.

The repository is organized so that notebooks remain thin, task-oriented frontends, while reusable logic is moved into `useful_notebooks_cube/`.

## Repository layout

- `AiidaPostProcess/`  
  Notebooks that retrieve or post-process data associated with AiiDA workflows.

- `ChargeAnalysis/`  
  Notebooks focused on charge-related analysis, including cumulative charge along `z`, charge-transfer workflows, and population analysis.

- `CubeFiles/`  
  Notebooks for reading, integrating, and visualizing Gaussian cube files.

- `useful_notebooks_cube/`  
  Shared helper package for cube-file I/O, line and plane analysis, plotting, and multi-cube workflows.

## General requirements

Depending on the notebook, you may need some or all of the following:

- Python 3
- Jupyter Notebook or JupyterLab
- `numpy`
- `matplotlib`
- `scipy`
- `aiida-core` and an active AiiDA profile for notebooks that read AiiDA nodes
- SSH / SCP access for notebooks that retrieve files from remote HPC systems

## Design principle

The repository is being cleaned up so that notebook-specific code stays in the notebooks, while reusable functionality is kept in `useful_notebooks_cube/`.

In practice this means:

- cube-file I/O is implemented once and reused
- cumulative-charge analysis is implemented once and reused
- directional profile / perpendicular-plane analysis is implemented once and reused
- multi-cube algebra and comparison workflows are implemented once and reused

This should make notebooks easier to maintain and less likely to diverge over time.

## The `useful_notebooks_cube` package

`useful_notebooks_cube` provides the shared cube-related functionality used by the notebooks.

### Main capabilities

#### 1. Cube I/O

- `read_cube_full(...)`  
  Read a Gaussian cube file and return header lines, atom lines, volumetric data, and grid shape.

- `write_cube(...)`  
  Write cube data back to disk using a supplied header and atom list.

- `read_cube_full_cached(...)`  
  Small in-memory cached reader for repeated interactive use in notebooks.

#### 2. Single-cube charge analysis

- `z_charge_density_profile(...)`  
  Compute the in-plane integrated charge profile `λ(z)` such that integrating over `z` gives the total charge.

- `cumulative_charge_z(...)`  
  Compute the cumulative integrated charge `Q(z)`.

- `z_at_charge(...)`  
  Find the `z` value at which a chosen cumulative charge is reached.

- `charge_at_z(...)`  
  Evaluate the cumulative charge at a chosen `z` value.

#### 3. Directional cube analysis

- `cube_plane_average_profile(...)`  
  Compute a 1D profile along an arbitrary direction `P1 → P2` by averaging the field over rectangles perpendicular to that direction.

- `cube_perpendicular_plane_map(...)`  
  Compute a 2D field map in the plane perpendicular to `P1 → P2` at a chosen position.

- `plot_line_profile(...)` and `plot_plane_map(...)`  
  Lightweight plotting helpers for the corresponding analysis results.

#### 4. Multi-cube workflows

- `read_cubes_same_grid(...)`  
  Read several cube files and assert that they share the same grid definition.

- `evaluate_cube_expression(...)`  
  Evaluate algebraic expressions such as:
  - `cube1 - cube2 - cube3`
  - `2*cube1 - cube2**2 + 3*cube3`

- `write_cube_expression(...)`  
  Evaluate an algebraic cube expression and write the result to disk, keeping the header and atoms of a chosen reference cube.

- `plot_cumulative_charge_multi(...)`  
  Plot cumulative charge for a selected subset of cubes, with optional custom display labels and optional vertical shifts.

- `z_at_charge_multi(...)`  
  For each selected cube, find the `z` value where a target cumulative charge is reached.

- `charge_at_z_multi(...)`  
  For each selected cube, evaluate the cumulative charge at a chosen `z` value.

### Notes

- The package uses the shared Bohr-to-Å conversion factor from its own constants module.
- The cube-analysis helpers are written to be notebook-friendly, but they are regular Python functions and can also be reused in scripts.
- The current design assumes that `z`-resolved cumulative-charge analysis is applied to cubes whose third cube axis corresponds to the physical Cartesian `z` direction.

## Notebook overview

## `AiidaPostProcess/`

Notebooks in this folder help retrieve, inspect, or post-process outputs associated with AiiDA workflows.

### `RetrieveSpinGuess.ipynb`

Given the PK of a CP2K calculation, retrieve the input spin guess and multiplicity.

In use by:
- Nicolò

Potential future refinements:
- use `list2range`-style formatting for output
- check portability across CP2K apps
- add a small GUI

### `GetSpinDensityFromGeoOpt.ipynb`

Retrieve volumetric cube files associated with a geometry-optimization workflow from the remote working directory. In its current form, the notebook loads an AiiDA node, identifies the corresponding remote folder, and copies selected cube files to a local directory via `scp`.

In use by:
- Deborah

Potential future refinements:
- generalize workflow / node selection
- make file-pattern selection configurable
- add clearer error handling when no matching files are found

### `NscfEigenvalues.ipynb`

Starting from the PK of a QE app workchain, identify the NSCF calculation, read the eigenvalues, and plot a DOS based solely on those eigenvalues. HOMO and LUMO values are inferred from the number of electrons.

In use by:
- Suyash
- Saketh

## `ChargeAnalysis/`

Notebooks in this folder focus on charge partitioning and related post-processing.

### `ChargeTransfer.ipynb`

Thin frontend for cumulative-charge comparison and cube-algebra workflows based on several compatible cube files. The notebook is intended to:

- define a set of cube files on a common grid
- load them once through `read_cubes_same_grid(...)`
- plot cumulative charge for a selected subset of cubes
- assign custom display labels in plots
- optionally apply vertical shifts to selected curves
- query `z` at target cumulative charge values
- query cumulative charge at chosen `z` values
- write algebraic combinations of cube files such as charge-density differences

The reusable logic now lives in `useful_notebooks_cube`, rather than being redefined inside the notebook.

### `PopulationAnalysis.ipynb`

Extract and sum atomic charges from CP2K population-analysis output stored in `aiida.out` within an AiiDA retrieved folder. The notebook includes utilities to parse flexible atom-index specifications and to compute total net charge over selected atoms for different analysis schemes such as Mulliken and Hirshfeld.

Potential future refinements:
- add clearer examples for mixed atom ranges and individual atom selections
- support additional population-analysis sections if needed
- move any repeated parsing helpers into a small shared module if they start being reused elsewhere

## `CubeFiles/`

Notebooks in this folder provide utilities for inspecting scalar fields stored in Gaussian cube files.

### `ChargeAverage.ipynb`

Read a charge-density cube file, compute the in-plane integrated charge profile as a function of `z`, plot it, and verify consistency between:

- the integral of the 1D profile, and
- the full 3D integral of the cube data.

This notebook now uses the shared package functions instead of redefining local cube readers and charge-profile logic.

In use by:
- Deborah

### `CubeDirectionalAnalysis.ipynb`

Read a cube file and analyze either a charge density or a Hartree/Rydberg potential along an arbitrary direction defined by two points `P1` and `P2` in Å.

The notebook supports two complementary modes:

- **1D directional profile**  
  Average the field over a rectangle perpendicular to `P1 → P2` and plot the result as a function of distance along that direction.

- **2D perpendicular-plane map**  
  Plot the field in a plane perpendicular to `P1 → P2` at a chosen position, optionally with Gaussian broadening along the normal direction.

The rectangle dimensions can be set manually or initialized automatically from the cell geometry. Periodic boundary conditions are applied consistently when sampling outside the original cell.

Potential future refinements:
- expose additional interpolation controls more explicitly
- add export of computed 1D/2D data
- continue refining the widget layout

## Suggested workflow for cube notebooks

For new cube-related notebooks, the preferred pattern is:

1. keep user-facing inputs and plotting in the notebook
2. import reusable logic from `useful_notebooks_cube`
3. avoid redefining cube readers, writers, or interpolation utilities locally
4. add new shared functionality to the package only if it is genuinely reusable

This should keep the repository easier to extend without accumulating slightly different versions of the same helper code.
