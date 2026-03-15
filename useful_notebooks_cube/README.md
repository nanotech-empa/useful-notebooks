# `useful_notebooks_cube`

Small utility package for reading, writing, analyzing, and plotting Gaussian cube files from Jupyter notebooks.

The package is meant to keep notebook cells short and reusable by moving common cube logic into a shared library. It currently covers:

- cube file I/O
- charge-density analysis along `z`
- directional profiles and perpendicular maps
- multi-cube workflows on a shared grid
- a few small notebook-oriented utilities

## Typical use cases

- read a cube file once and reuse the data in several cells
- compute the integrated charge profile `λ(z)` and cumulative charge `Q(z)`
- compare cumulative charge from several cube files on the same plot
- find the `z` coordinate at which a cube reaches a target cumulative charge
- evaluate and write cube algebra such as `cube1 - cube2 - cube3`
- sample a directional profile between two user-defined points
- plot a perpendicular plane map through a chosen position

## Public API

### Constants

#### `bohr_to_ang`
Shared Bohr-to-Å conversion factor used throughout the package.

---

### Cube I/O

#### `read_cube_full(filename)`
Read a Gaussian cube file and return:

- `header_lines`
- `atom_lines`
- `rho`
- `grid_shape`

`rho` is returned with shape `(nx, ny, nz)`.

#### `read_cube_full_cached(filename, verbose=False)`
Same as `read_cube_full`, but with a small in-memory LRU cache. Useful when the same one or two cube files are replotted many times in a notebook.

#### `write_cube(filename, header_lines, atom_lines, rho)`
Write a cube file from a header, atom list, and 3D array.

---

### Single-cube charge analysis

#### `z_charge_density_profile(header_lines, rho, bohr_to_ang, ...)`
Compute the in-plane integrated charge profile `λ(z)` such that:

\[
\int \lambda(z)\,dz = Q_{\mathrm{tot}}
\]

Returns:

- `z_ang`
- `lambda_z_ang`

This assumes the third cube axis corresponds to Cartesian `z`.

#### `cumulative_charge_z(header_lines, rho, bohr_to_ang, zmin=None, zmax=None, ...)`
Compute the cumulative charge profile `Q(z)` in electrons.

Returns:

- `z_ang`
- `Qz`

#### `z_at_charge(header_lines, rho, target_charge, bohr_to_ang, zmin=None, zmax=None, ...)`
Return the `z` coordinate in Å at which the cumulative charge reaches a target value.

#### `charge_at_z(header_lines, rho, z_value, bohr_to_ang, zmin=None, zmax=None, ...)`
Return the cumulative charge in electrons at a chosen `z` value.

---

### Directional analysis

#### `cube_plane_average_profile(header_lines, cube_values, bohr_to_ang, P1, P2, ...)`
Compute a 1D profile along the line `P1 -> P2` by averaging the cube field over rectangles perpendicular to that line.

Useful for charge-density or potential profiles along arbitrary directions.

Returns a dictionary containing:

- sampled coordinates
- profile values
- axis labels
- units
- geometric metadata
- sampling metadata

#### `cube_perpendicular_plane_map(header_lines, cube_values, bohr_to_ang, P1, P2, position=0.0, ...)`
Compute a 2D map in the plane perpendicular to `P1 -> P2` at a chosen position.

Optional Gaussian broadening along the normal direction is supported.

Returns a dictionary containing:

- `u/v` grids
- `map_2d`
- axis labels
- colorbar label
- geometric metadata
- sampling metadata

---

### Plotting helpers

#### `plot_line_profile(result, title=None, ...)`
Plot the dictionary returned by `cube_plane_average_profile`.

#### `plot_plane_map(result, title=None, ...)`
Plot the dictionary returned by `cube_perpendicular_plane_map`.

---

### Multi-cube workflows

These helpers assume several cube files share the same volumetric grid. Atom lists may differ.

#### `read_cubes_same_grid(cube_files, use_cache=True, ...)`
Read several cubes and assert that they have the same:

- grid shape
- origin
- voxel step vectors

Input can be either:

- a mapping such as `{label: path}`
- a sequence of paths, in which case labels `cube1`, `cube2`, ... are assigned automatically

Returns a `cube_set` dictionary used by the functions below.

#### `evaluate_cube_expression(cube_set, expression)`
Evaluate algebraic expressions such as:

- `cube1 - cube2 - cube3`
- `2*cube1 - cube2**2 + 3*cube3`
- `(cube1 - cube2) / 2`

The expression language accepts only simple arithmetic on cube labels and scalar constants.

#### `write_cube_expression(cube_set, expression, output_filename, reference_label=None, ...)`
Evaluate a cube expression and write the result to a cube file.

The output uses the header and atom list of the reference cube. By default, this is the first cube in the `cube_set`.

#### `plot_cumulative_charge_multi(cube_set, labels=None, shifts=None, ...)`
Plot cumulative charge curves `Q(z)` for a selected set of cubes.

`labels` may be:

- `None` → plot all cubes with internal labels
- a list of internal labels
- a mapping `{internal_label: display_label}`

`shifts` may be:

- `None` → all shifts are zero
- a scalar → same vertical shift for all curves
- a list of shifts in the selected-label order
- a mapping `{internal_label: shift}`

The shift is applied on the plotted `Q(z)` value, i.e. `Q_plot(z) = Q(z) + shift`.

#### `z_at_charge_multi(cube_set, target_charge, labels=None, ...)`
For each selected cube, return the `z` value at which the cumulative charge reaches a target value.

#### `charge_at_z_multi(cube_set, z_value, labels=None, ...)`
For each selected cube, return the cumulative charge at a chosen `z` value.

---

### Small utilities

#### `parse_point(text)`
Parse a 3D point from flexible text input such as:

- `"0 1 2"`
- `"(0, 1, 2)"`
- `"0, 1, 2"`

#### `float_or_none(text)`
Convert optional text input to float. Empty input returns `None`.

#### `collect_cube_files(download_dir=..., recursive=False)`
Collect `.cube` files from a directory for notebook dropdowns or file selection.

## Minimal examples

### Read one cube and compute `Q(z)`

```python
from useful_notebooks_cube import bohr_to_ang, read_cube_full, cumulative_charge_z

header_lines, atom_lines, rho, shape = read_cube_full("charge.cube")
z_ang, Qz = cumulative_charge_z(header_lines, rho, bohr_to_ang)
```

### Write a simple cube difference

```python
from useful_notebooks_cube import read_cubes_same_grid, write_cube_expression

cube_set = read_cubes_same_grid({
    "cube1": "system.cube",
    "cube2": "slab.cube",
    "cube3": "molecule.cube",
})

write_cube_expression(
    cube_set,
    expression="cube1 - cube2 - cube3",
    output_filename="difference.cube",
    reference_label="cube1",
)
```

### Plot cumulative charge of selected cubes with custom labels

```python
from useful_notebooks_cube import read_cubes_same_grid, plot_cumulative_charge_multi

cube_set = read_cubes_same_grid({
    "cube4": "T4/charge.cube",
    "cube2": "T2/charge.cube",
    "slab4": "T4/slab.cube",
    "slab2": "T2/slab.cube",
})

cubes_to_plot = {
    "cube4": "T4Au",
    "cube2": "T2Au",
    "slab4": "slab T4",
    "slab2": "slab T2",
}

shifts = {
    "slab2": 1000.0,
}

fig, ax, curves = plot_cumulative_charge_multi(
    cube_set,
    labels=cubes_to_plot,
    shifts=shifts,
    title="Cumulative charge",
)
```

### Find where a target cumulative charge is reached

```python
from useful_notebooks_cube import z_at_charge_multi

z_targets = z_at_charge_multi(
    cube_set,
    target_charge=62.0,
    labels=["cube4", "cube2"],
)
```

## Design principles

The package is intentionally notebook-oriented:

- public functions should be easy to call from a single cell
- low-level cube logic should not be redefined inside notebooks
- plotting helpers should stay lightweight
- multi-cube workflows should reuse single-cube primitives internally

If new notebook patterns appear more than once, they should generally be moved here rather than copied into another notebook.
