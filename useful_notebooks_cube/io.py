import numpy as np
from functools import lru_cache
def read_cube_full(filename):
    """
    Read a Gaussian cube file.

    Returns
    -------
    header_lines : list[str]
        Original header lines up to atom list (used for writing)
    atom_lines : list[str]
        Atom specification lines
    rho : ndarray (nx, ny, nz)
        Volumetric data as stored in the cube file
    grid_shape : tuple
        (nx, ny, nz)
    """
    filename = str(filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    header_lines = lines[:2]

    # Cube line 3: natoms origin_x origin_y origin_z
    natoms = abs(int(lines[2].split()[0]))
    grid_lines = lines[2:6]
    header_lines += grid_lines

    nx = abs(int(grid_lines[1].split()[0]))
    ny = abs(int(grid_lines[2].split()[0]))
    nz = abs(int(grid_lines[3].split()[0]))

    atom_lines = lines[6:6 + natoms]
    data_lines = lines[6 + natoms:]

    raw = []
    for line in data_lines:
        raw.extend(map(float, line.split()))

    expected = nx * ny * nz
    if len(raw) != expected:
        raise ValueError(
            f"Cube data size mismatch in {filename}: "
            f"expected {expected} values, found {len(raw)}."
        )

    rho = np.array(raw, dtype=float).reshape((nx, ny, nz))

    return header_lines, atom_lines, rho, (nx, ny, nz)

@lru_cache(maxsize=2)
def _read_cube_full_cached(filename):
    return read_cube_full(filename)


def read_cube_full_cached(filename, verbose=False):
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