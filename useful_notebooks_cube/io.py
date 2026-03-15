from functools import lru_cache

import numpy as np

def write_cube(filename, header_lines, atom_lines, rho):
    """
    Write a Gaussian cube file.

    Parameters
    ----------
    filename : str or path-like
        Output cube filename.
    header_lines : list[str]
        Header lines exactly as returned by ``read_cube_full``:
        2 comment lines + origin/natoms line + 3 grid-definition lines.
    atom_lines : list[str]
        Atom specification lines exactly as returned by ``read_cube_full``.
    rho : numpy.ndarray
        Volumetric data with shape (nx, ny, nz).

    Notes
    -----
    The function writes volumetric data using the standard cube convention
    of 6 values per line in scientific notation.

    The array is flattened in C order, consistent with the repository
    convention used in ``read_cube_full(...).reshape((nx, ny, nz))``.
    """
    filename = str(filename)
    rho = np.asarray(rho, dtype=float)

    if rho.ndim != 3:
        raise ValueError(
            f"`rho` must be a 3D array, got shape {rho.shape}."
        )

    if len(header_lines) != 6:
        raise ValueError(
            f"`header_lines` must contain exactly 6 lines, got {len(header_lines)}."
        )

    # Validate consistency between header and data shape
    try:
        nx_h = abs(int(header_lines[3].split()[0]))
        ny_h = abs(int(header_lines[4].split()[0]))
        nz_h = abs(int(header_lines[5].split()[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "Could not parse grid dimensions from `header_lines`."
        ) from exc

    if rho.shape != (nx_h, ny_h, nz_h):
        raise ValueError(
            f"Shape mismatch between header and data: header expects "
            f"({nx_h}, {ny_h}, {nz_h}), but rho has shape {rho.shape}."
        )

    # Basic atom-count consistency check
    try:
        natoms_h = abs(int(header_lines[2].split()[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "Could not parse number of atoms from `header_lines`."
        ) from exc

    if len(atom_lines) != natoms_h:
        raise ValueError(
            f"Atom-count mismatch: header expects {natoms_h} atom lines, "
            f"but received {len(atom_lines)}."
        )

    with open(filename, "w", encoding="utf-8") as handle:
        # Preserve header and atom lines exactly as provided
        for line in header_lines:
            handle.write(line if line.endswith("\n") else line + "\n")

        for line in atom_lines:
            handle.write(line if line.endswith("\n") else line + "\n")

        flat = rho.ravel(order="C")
        for i in range(0, flat.size, 6):
            chunk = flat[i:i + 6]
            handle.write(" ".join(f"{val: .6e}" for val in chunk) + "\n")
            
def read_cube_full(filename):
    """
    Read a Gaussian cube file.

    Parameters
    ----------
    filename : str or path-like
        Path to the cube file.

    Returns
    -------
    header_lines : list[str]
        Original header lines up to and including the grid-definition lines:
        2 comment lines + origin/natoms line + 3 grid lines.
        These lines are preserved exactly, including their trailing newlines,
        so they can be reused when writing a cube back to disk.
    atom_lines : list[str]
        Atomic specification lines, preserved exactly as in the file.
    rho : numpy.ndarray
        Volumetric data with shape (nx, ny, nz), in the units stored in the
        cube file.
    grid_shape : tuple[int, int, int]
        Grid shape (nx, ny, nz).

    Notes
    -----
    This function keeps the current repository convention that the cube data
    are reshaped as (nx, ny, nz), consistent with how the existing notebooks
    use the array.

    It also handles the common cube convention where the number of atoms or
    grid counts may appear with a negative sign.
    """
    filename = str(filename)

    with open(filename, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if len(lines) < 6:
        raise ValueError(
            f"Cube file {filename!r} is too short: expected at least 6 header lines, "
            f"found {len(lines)}."
        )

    # First two comment lines
    header_lines = lines[:2]

    # Third line: natoms origin_x origin_y origin_z
    try:
        natoms = abs(int(lines[2].split()[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Could not parse the number of atoms from line 3 of cube file {filename!r}."
        ) from exc

    # Lines 3-6 of the cube logical header: origin + 3 grid lines
    grid_lines = lines[2:6]
    header_lines += grid_lines

    try:
        nx = abs(int(grid_lines[1].split()[0]))
        ny = abs(int(grid_lines[2].split()[0]))
        nz = abs(int(grid_lines[3].split()[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Could not parse grid dimensions from header of cube file {filename!r}."
        ) from exc

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(
            f"Invalid grid dimensions in cube file {filename!r}: ({nx}, {ny}, {nz})."
        )

    atom_start = 6
    atom_stop = atom_start + natoms
    if len(lines) < atom_stop:
        raise ValueError(
            f"Cube file {filename!r} ended before the full atom list was read: "
            f"expected {natoms} atom lines."
        )

    atom_lines = lines[atom_start:atom_stop]
    data_lines = lines[atom_stop:]

    raw_values = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw_values.extend(float(x) for x in stripped.split())
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse volumetric data in cube file {filename!r}. "
                f"Offending line: {line!r}"
            ) from exc

    expected = nx * ny * nz
    found = len(raw_values)
    if found != expected:
        raise ValueError(
            f"Cube data size mismatch in {filename!r}: expected {expected} values, "
            f"found {found}."
        )

    rho = np.asarray(raw_values, dtype=float).reshape((nx, ny, nz))

    return header_lines, atom_lines, rho, (nx, ny, nz)

@lru_cache(maxsize=2)
def _read_cube_full_cached(filename):
    """
    Internal LRU-cached wrapper around ``read_cube_full``.

    Parameters
    ----------
    filename : str
        Cube filename. It must already be normalized to string form before
        reaching this function, so equivalent ``Path`` objects map to the same
        cache entry.

    Returns
    -------
    tuple
        Exactly the output of ``read_cube_full(filename)``.
    """
    return read_cube_full(filename)


def read_cube_full_cached(filename, verbose=False):
    """
    Read a Gaussian cube file with a small in-memory LRU cache.

    Parameters
    ----------
    filename : str or path-like
        Path to the cube file.
    verbose : bool, default False
        If True, print whether the file was read from disk or returned from cache.

    Returns
    -------
    header_lines : list[str]
        Original header lines up to and including the grid-definition lines.
    atom_lines : list[str]
        Atomic specification lines.
    rho : numpy.ndarray
        Volumetric data with shape (nx, ny, nz).
    grid_shape : tuple[int, int, int]
        Grid shape (nx, ny, nz).

    Notes
    -----
    The cache is intentionally small because notebook workflows often alternate
    between one or two cube files repeatedly. This keeps repeated plotting or
    re-analysis responsive without holding many large cube arrays in memory.
    """
    filename = str(filename)

    hits_before = _read_cube_full_cached.cache_info().hits
    result = _read_cube_full_cached(filename)
    hits_after = _read_cube_full_cached.cache_info().hits

    if verbose:
        if hits_after > hits_before:
            print(f"Using cached cube file: {filename}")
        else:
            print(f"Reading cube file from disk: {filename}")

    return result