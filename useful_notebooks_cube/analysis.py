import numpy as np
from scipy.ndimage import map_coordinates

def _normalize_field_type(field_type):
    """
    Normalize user-provided field type aliases.
    Accepted families:
      - charge
      - hartree
      - rydberg
    """
    key = str(field_type).strip().lower()

    aliases = {
        "charge": "charge",
        "rho": "charge",
        "density": "charge",
        "charge_density": "charge",

        "hartree": "hartree",
        "ha": "hartree",
        "h": "hartree",
        "potential_hartree": "hartree",

        "rydberg": "rydberg",
        "ry": "rydberg",
        "potential_rydberg": "rydberg",
    }

    if key not in aliases:
        raise ValueError(
            f"Unknown field_type={field_type!r}. "
            "Use one of: 'charge', 'hartree', 'rydberg'."
        )
    return aliases[key]

def _cube_geometry(header_lines, bohr_to_ang):
    """
    Parse geometric information from a Gaussian cube header.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines, expected in the standard format returned by
        ``read_cube_full``:
          - 2 comment lines
          - 1 line with natoms and origin
          - 3 grid-definition lines
    bohr_to_ang : float
        Bohr-to-Å conversion factor.

    Returns
    -------
    dict
        Dictionary with:
        - origin_ang : ndarray, shape (3,)
            Grid origin in Å.
        - step_matrix_ang : ndarray, shape (3, 3)
            Matrix whose columns are the three voxel step vectors in Å.
        - cell_matrix_ang : ndarray, shape (3, 3)
            Matrix whose columns are the full cell vectors in Å.
        - inv_step_matrix_ang : ndarray, shape (3, 3)
            Inverse of ``step_matrix_ang``.
        - nxyz : ndarray, shape (3,)
            Grid shape [nx, ny, nz].

    Notes
    -----
    The cube convention is:
    - line 3: natoms origin_x origin_y origin_z
    - next 3 lines: n_i v_i_x v_i_y v_i_z

    The grid counts may appear with a negative sign in some cube variants;
    only their absolute values are physically relevant here.
    """
    if len(header_lines) < 6:
        raise ValueError(
            f"Expected at least 6 header lines, got {len(header_lines)}."
        )

    try:
        origin_tokens = header_lines[2].split()
        xgrid_tokens = header_lines[3].split()
        ygrid_tokens = header_lines[4].split()
        zgrid_tokens = header_lines[5].split()
    except IndexError as exc:
        raise ValueError("Cube header does not contain the required geometry lines.") from exc

    try:
        origin_bohr = np.array(list(map(float, origin_tokens[1:4])), dtype=float)

        nx = abs(int(xgrid_tokens[0]))
        ny = abs(int(ygrid_tokens[0]))
        nz = abs(int(zgrid_tokens[0]))

        vx_bohr = np.array(list(map(float, xgrid_tokens[1:4])), dtype=float)
        vy_bohr = np.array(list(map(float, ygrid_tokens[1:4])), dtype=float)
        vz_bohr = np.array(list(map(float, zgrid_tokens[1:4])), dtype=float)
    except (IndexError, ValueError) as exc:
        raise ValueError("Failed to parse origin and grid vectors from cube header.") from exc

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(
            f"Invalid cube grid dimensions: ({nx}, {ny}, {nz})."
        )

    step_matrix_bohr = np.column_stack([vx_bohr, vy_bohr, vz_bohr])

    det = float(np.linalg.det(step_matrix_bohr))
    if np.isclose(det, 0.0, atol=1e-15):
        raise ValueError(
            "Cube step matrix is singular; the three grid vectors are not linearly independent."
        )

    origin_ang = origin_bohr * bohr_to_ang
    step_matrix_ang = step_matrix_bohr * bohr_to_ang

    nxyz = np.array([nx, ny, nz], dtype=int)
    cell_matrix_ang = step_matrix_ang @ np.diag(nxyz)

    return {
        "origin_ang": origin_ang,
        "step_matrix_ang": step_matrix_ang,
        "cell_matrix_ang": cell_matrix_ang,
        "inv_step_matrix_ang": np.linalg.inv(step_matrix_ang),
        "nxyz": nxyz,
    }

def _convert_cube_values(values, field_type, bohr_to_ang):
    """
    Convert cube values to requested physical units.

    charge:
        e / bohr^3  ->  e / Å^3
    hartree:
        unchanged
    rydberg:
        unchanged
    """
    field_type = _normalize_field_type(field_type)
    values = np.asarray(values, dtype=float)

    if field_type == "charge":
        out = values / (bohr_to_ang ** 3)
        unit = "e/Å³"
        quantity_name = "Charge density"
    elif field_type == "hartree":
        out = values.copy()
        unit = "Hartree"
        quantity_name = "Potential"
    elif field_type == "rydberg":
        out = values.copy()
        unit = "Rydberg"
        quantity_name = "Potential"
    else:
        raise RuntimeError("Unexpected field type after normalization.")

    return out, unit, quantity_name, field_type

def _orthonormal_plane_basis(direction):
    """
    Given a nonzero direction vector, return:
      - uhat: unit vector along direction
      - e1, e2: orthonormal basis spanning the plane perpendicular to uhat
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("P1 and P2 must be different points.")

    uhat = direction / norm

    # choose a Cartesian axis least parallel to uhat
    cart_axes = np.eye(3)
    ref = cart_axes[np.argmin(np.abs(cart_axes @ uhat))]

    e1 = ref - np.dot(ref, uhat) * uhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(uhat, e1)
    e2 /= np.linalg.norm(e2)

    return uhat, e1, e2
def _default_perp_rectangle(cell_matrix_ang, e1, e2):
    """
    Default rectangle sizes in the plane perpendicular to the analysis direction.

    We use the bounding rectangle of the projection of the full periodic cell
    parallelepiped onto the perpendicular plane basis (e1, e2).
    """
    frac_corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    )

    # columns of cell_matrix_ang are the three cell vectors
    corners = frac_corners @ cell_matrix_ang.T

    u_proj = corners @ e1
    v_proj = corners @ e2

    L_default = u_proj.max() - u_proj.min()
    W_default = v_proj.max() - v_proj.min()

    return float(L_default), float(W_default)

def _centered_axis(length, target_step):
    """
    Build 1D centered sampling coordinates across an interval of size `length`.

    Returns:
      coords  : cell-centered coordinates from -length/2 to +length/2
      step    : exact step actually used
      npts    : number of points
    """
    length = float(length)
    target_step = float(target_step)

    if length <= 0:
        raise ValueError("A rectangle length/width must be > 0.")
    if target_step <= 0:
        raise ValueError("Sampling step must be > 0.")

    npts = max(1, int(np.ceil(length / target_step)))
    step = length / npts
    coords = -0.5 * length + (np.arange(npts) + 0.5) * step

    return coords, step, npts

def _line_centers(total_length, target_dl):
    """
    Build line-centered coordinates from P1 to P2.

    Returns:
      l_centers : line bin centers
      l_edges   : line bin edges
      dl        : exact step actually used
      npts      : number of line bins
    """
    total_length = float(total_length)
    target_dl = float(target_dl)

    if total_length <= 0:
        raise ValueError("The line length |P2-P1| must be > 0.")
    if target_dl <= 0:
        raise ValueError("dl must be > 0.")

    npts = max(1, int(np.ceil(total_length / target_dl)))
    dl = total_length / npts
    l_edges = np.linspace(0.0, total_length, npts + 1)
    l_centers = 0.5 * (l_edges[:-1] + l_edges[1:])

    return l_centers, l_edges, dl, npts


def _cartesian_to_grid_indices(points_ang, geom):
    """
    Convert Cartesian points in Å to cube grid coordinates (i, j, k),
    where the field is sampled as rho[i, j, k].

    `points_ang` shape can be (..., 3).
    """
    pts = np.asarray(points_ang, dtype=float)
    pts_2d = pts.reshape(-1, 3)

    rel = pts_2d - geom["origin_ang"]
    ijk = rel @ geom["inv_step_matrix_ang"].T

    return ijk.reshape(pts.shape)


def _sample_cube_periodic(points_ang, values_phys, geom, order=1):
    """
    Periodic interpolation of the cube field at arbitrary Cartesian points.

    Uses scipy.ndimage.map_coordinates with mode='wrap'.
    """
    if order not in (0, 1, 2, 3, 4, 5):
        raise ValueError("Interpolation order must be an integer between 0 and 5.")

    ijk = _cartesian_to_grid_indices(points_ang, geom).reshape(-1, 3)
    coords = np.vstack([ijk[:, 0], ijk[:, 1], ijk[:, 2]])

    sampled = map_coordinates(
        values_phys,
        coords,
        order=order,
        mode="wrap",
        prefilter=(order > 1),
    )

    return sampled.reshape(np.asarray(points_ang).shape[:-1])


# ----------------------------
# public functions
# ----------------------------

def cube_plane_average_profile(
    header_lines,
    cube_values,
    bohr_to_ang,
    P1,
    P2,
    field_type="charge",
    dl=0.1,
    L=None,
    W=None,
    du=0.1,
    dv=None,
    order=1,
):
    """
    Compute a 1D profile along the line P1 -> P2 by averaging the cube field
    over rectangles perpendicular to that line.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines as returned by ``read_cube_full``.
    cube_values : numpy.ndarray
        Cube scalar field with shape (nx, ny, nz).
    bohr_to_ang : float
        Bohr-to-Å conversion factor.
    P1, P2 : array-like of length 3
        Endpoints of the analysis line in Å.
    field_type : {'charge', 'hartree', 'rydberg'}, default 'charge'
        Physical interpretation of the cube values.
    dl : float, default 0.1
        Target sampling step along the line P1 -> P2, in Å.
    L, W : float or None, optional
        Size of the perpendicular averaging rectangle in Å along the two
        in-plane basis vectors. If omitted, use the projected bounding
        rectangle of the full cell.
    du, dv : float or None, optional
        Target in-plane sampling steps in Å. If ``dv`` is None, use ``du``.
    order : int, default 1
        Interpolation order passed to ``scipy.ndimage.map_coordinates``.
        Allowed values are 0 through 5.

    Returns
    -------
    dict
        Dictionary containing the sampled profile, geometric metadata, labels,
        units, and sampling information.

    Notes
    -----
    The returned ``profile`` is an average over each perpendicular rectangle,
    not an integral over it. Therefore:
    - for ``field_type='charge'`` the profile is in e / Å³
    - for ``field_type='hartree'`` it is in Hartree
    - for ``field_type='rydberg'`` it is in Rydberg

    Periodic wrapping is used when sampling outside the primitive cube cell.
    """
    cube_values = np.asarray(cube_values, dtype=float)
    if cube_values.ndim != 3:
        raise ValueError(
            f"`cube_values` must be a 3D array, got shape {cube_values.shape}."
        )

    geom = _cube_geometry(header_lines, bohr_to_ang)
    expected_shape = tuple(int(x) for x in geom["nxyz"])
    if cube_values.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: cube header expects {expected_shape}, "
            f"but `cube_values` has shape {cube_values.shape}."
        )

    values_phys, unit, quantity_name, field_type = _convert_cube_values(
        cube_values, field_type, bohr_to_ang
    )

    P1 = np.asarray(P1, dtype=float).reshape(-1)
    P2 = np.asarray(P2, dtype=float).reshape(-1)
    if P1.shape != (3,) or P2.shape != (3,):
        raise ValueError("`P1` and `P2` must each contain exactly 3 coordinates.")

    direction = P2 - P1
    axis_length = float(np.linalg.norm(direction))
    if axis_length <= 0:
        raise ValueError("`P1` and `P2` must be different points.")

    uhat, e1, e2 = _orthonormal_plane_basis(direction)
    L_default, W_default = _default_perp_rectangle(geom["cell_matrix_ang"], e1, e2)

    if L is None:
        L = L_default
    if W is None:
        W = W_default
    if dv is None:
        dv = du

    L = float(L)
    W = float(W)
    du = float(du)
    dv = float(dv)

    l_ang, l_edges_ang, dl_eff, nl = _line_centers(axis_length, dl)
    u_ang, du_eff, nu = _centered_axis(L, du)
    v_ang, dv_eff, nv = _centered_axis(W, dv)

    U, V = np.meshgrid(u_ang, v_ang, indexing="ij")
    plane_offsets = U[..., None] * e1 + V[..., None] * e2
    plane_offsets_flat = plane_offsets.reshape(-1, 3)

    profile = np.empty(nl, dtype=float)

    for i, s in enumerate(l_ang):
        center = P1 + s * uhat
        points = center[None, :] + plane_offsets_flat
        vals = _sample_cube_periodic(points, values_phys, geom, order=order)
        profile[i] = float(np.mean(vals))

    rectangle_area = float(L * W)
    dA = float(du_eff * dv_eff)
    dV = float(dA * dl_eff)

    direction_str = f"({direction[0]:.6g}, {direction[1]:.6g}, {direction[2]:.6g})"
    xlabel = f"L (Å) along {direction_str}"

    if field_type == "charge":
        ylabel = f"Average charge density ({unit})"
    else:
        ylabel = f"Average potential ({unit})"

    return {
        # main data
        "l_ang": l_ang,
        "l_edges_ang": l_edges_ang,
        "profile": profile,

        # labels / units
        "xlabel": xlabel,
        "ylabel": ylabel,
        "field_unit": unit,
        "field_type": field_type,
        "quantity_name": quantity_name,

        # geometry
        "P1_ang": P1,
        "P2_ang": P2,
        "direction_ang": direction,
        "axis_length_ang": axis_length,
        "axis_hat": uhat,
        "plane_e1": e1,
        "plane_e2": e2,

        # rectangle
        "L_ang": float(L),
        "W_ang": float(W),
        "L_default_ang": float(L_default),
        "W_default_ang": float(W_default),
        "rectangle_area_ang2": rectangle_area,

        # plane sampling
        "u_ang": u_ang,
        "v_ang": v_ang,
        "du_ang": float(du_eff),
        "dv_ang": float(dv_eff),
        "dA_ang2": dA,
        "nu": int(nu),
        "nv": int(nv),

        # line sampling
        "dl_ang": float(dl_eff),
        "dV_ang3": dV,
        "nl": int(nl),

        # useful references
        "origin_ang": geom["origin_ang"],
        "cell_matrix_ang": geom["cell_matrix_ang"],
        "step_matrix_ang": geom["step_matrix_ang"],
        "grid_shape": geom["nxyz"],
    }


def cube_perpendicular_plane_map(
    header_lines,
    cube_values,
    bohr_to_ang,
    P1,
    P2,
    position=0.0,
    field_type="charge",
    L=None,
    W=None,
    du=0.1,
    dv=None,
    sigma=0.0,
    dn=None,
    truncate=4.0,
    order=1,
):
    """
    Compute a 2D map in the plane perpendicular to P1 -> P2 at a given
    position, with optional Gaussian broadening along the plane normal.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines as returned by ``read_cube_full``.
    cube_values : numpy.ndarray
        Cube scalar field with shape (nx, ny, nz).
    bohr_to_ang : float
        Bohr-to-Å conversion factor.
    P1, P2 : array-like of length 3
        Endpoints of the reference line in Å.
    position : float, default 0.0
        Position along the P1 -> P2 direction in Å, measured from P1.
        Thus:
        - ``position = 0`` gives the plane through P1
        - ``position = |P2 - P1|`` gives the plane through P2
    field_type : {'charge', 'hartree', 'rydberg'}, default 'charge'
        Physical interpretation of the cube values.
    L, W : float or None, optional
        Size of the perpendicular map rectangle in Å along the two in-plane
        basis vectors. If omitted, use the projected bounding rectangle of the
        full cell.
    du, dv : float or None, optional
        Target in-plane sampling steps in Å. If ``dv`` is None, use ``du``.
    sigma : float, default 0.0
        Gaussian broadening width in Å along the plane normal.
        ``sigma = 0`` means a direct slice with no broadening.
    dn : float or None, optional
        Sampling step in Å along the normal direction used for the Gaussian
        quadrature when ``sigma > 0``. If omitted, choose an automatic value.
    truncate : float, default 4.0
        Gaussian cutoff in units of sigma.
    order : int, default 1
        Interpolation order passed to ``scipy.ndimage.map_coordinates``.
        Allowed values are 0 through 5.

    Returns
    -------
    dict
        Dictionary containing the sampled 2D map, geometric metadata, labels,
        units, and sampling information.

    Notes
    -----
    The returned map is a field value on the perpendicular plane:
    - for ``field_type='charge'`` it is in e / Å³
    - for ``field_type='hartree'`` it is in Hartree
    - for ``field_type='rydberg'`` it is in Rydberg

    When ``sigma > 0``, the returned map is Gaussian-averaged along the plane
    normal, i.e. along the P1 -> P2 direction.
    """
    cube_values = np.asarray(cube_values, dtype=float)
    if cube_values.ndim != 3:
        raise ValueError(
            f"`cube_values` must be a 3D array, got shape {cube_values.shape}."
        )

    geom = _cube_geometry(header_lines, bohr_to_ang)
    expected_shape = tuple(int(x) for x in geom["nxyz"])
    if cube_values.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch: cube header expects {expected_shape}, "
            f"but `cube_values` has shape {cube_values.shape}."
        )

    values_phys, unit, quantity_name, field_type = _convert_cube_values(
        cube_values, field_type, bohr_to_ang
    )

    P1 = np.asarray(P1, dtype=float).reshape(-1)
    P2 = np.asarray(P2, dtype=float).reshape(-1)
    if P1.shape != (3,) or P2.shape != (3,):
        raise ValueError("`P1` and `P2` must each contain exactly 3 coordinates.")

    direction = P2 - P1
    axis_length = float(np.linalg.norm(direction))
    if axis_length <= 0:
        raise ValueError("`P1` and `P2` must be different points.")

    position = float(position)
    sigma = float(sigma)

    uhat, e1, e2 = _orthonormal_plane_basis(direction)
    L_default, W_default = _default_perp_rectangle(geom["cell_matrix_ang"], e1, e2)

    if L is None:
        L = L_default
    if W is None:
        W = W_default
    if dv is None:
        dv = du

    L = float(L)
    W = float(W)
    du = float(du)
    dv = float(dv)
    truncate = float(truncate)

    u_ang, du_eff, nu = _centered_axis(L, du)
    v_ang, dv_eff, nv = _centered_axis(W, dv)

    U, V = np.meshgrid(u_ang, v_ang, indexing="ij")
    center = P1 + position * uhat

    plane_points = center + U[..., None] * e1 + V[..., None] * e2
    plane_points_flat = plane_points.reshape(-1, 3)

    if sigma < 0:
        raise ValueError("`sigma` must be >= 0.")

    if sigma == 0.0:
        field_2d = _sample_cube_periodic(
            plane_points_flat,
            values_phys,
            geom,
            order=order,
        ).reshape(U.shape)
        normal_offsets = np.array([0.0], dtype=float)
        normal_weights = np.array([1.0], dtype=float)
        dn_eff = 0.0
    else:
        if dn is None:
            dn = min(du_eff, dv_eff, max(sigma / 3.0, 0.05))
        dn = float(dn)

        if dn <= 0:
            raise ValueError("`dn` must be > 0 when `sigma > 0`.")
        if truncate <= 0:
            raise ValueError("`truncate` must be > 0.")

        n_side = max(1, int(np.ceil(truncate * sigma / dn)))
        normal_offsets = np.arange(-n_side, n_side + 1, dtype=float) * dn
        normal_weights = np.exp(-0.5 * (normal_offsets / sigma) ** 2)
        normal_weights /= normal_weights.sum()
        dn_eff = dn

        field_2d = np.zeros(U.shape, dtype=float)
        for t, w in zip(normal_offsets, normal_weights):
            vals = _sample_cube_periodic(
                plane_points_flat + t * uhat,
                values_phys,
                geom,
                order=order,
            ).reshape(U.shape)
            field_2d += w * vals

    if field_type == "charge":
        colorbar_label = f"Charge density ({unit})"
    else:
        colorbar_label = f"Potential ({unit})"

    return {
        # main data
        "u_ang_grid": U,
        "v_ang_grid": V,
        "map_2d": field_2d,

        # labels / units
        "xlabel": "u (Å)",
        "ylabel": "v (Å)",
        "colorbar_label": colorbar_label,
        "field_unit": unit,
        "field_type": field_type,
        "quantity_name": quantity_name,

        # geometry
        "P1_ang": P1,
        "P2_ang": P2,
        "direction_ang": direction,
        "axis_length_ang": axis_length,
        "axis_hat": uhat,
        "plane_e1": e1,
        "plane_e2": e2,
        "plane_center_ang": center,
        "position_ang": position,

        # rectangle
        "L_ang": float(L),
        "W_ang": float(W),
        "L_default_ang": float(L_default),
        "W_default_ang": float(W_default),

        # plane sampling
        "u_ang": u_ang,
        "v_ang": v_ang,
        "du_ang": float(du_eff),
        "dv_ang": float(dv_eff),
        "dA_ang2": float(du_eff * dv_eff),
        "nu": int(nu),
        "nv": int(nv),

        # Gaussian broadening
        "sigma_ang": sigma,
        "dn_ang": float(dn_eff),
        "normal_offsets_ang": normal_offsets,
        "normal_weights": normal_weights,
        "truncate": truncate,

        # useful references
        "origin_ang": geom["origin_ang"],
        "cell_matrix_ang": geom["cell_matrix_ang"],
        "step_matrix_ang": geom["step_matrix_ang"],
        "grid_shape": geom["nxyz"],
    }

def z_charge_density_profile(
    header_lines,
    rho,
    bohr_to_ang,
    check_axis_alignment=True,
    alignment_tol=1e-8,
):
    """
    Compute the in-plane integrated charge profile lambda(z) such that

        integral lambda(z) dz = total charge.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines as returned by ``read_cube_full``.
    rho : numpy.ndarray
        Charge density array with shape (nx, ny, nz), in e / bohr^3.
    bohr_to_ang : float
        Bohr-to-Å conversion factor.
    check_axis_alignment : bool, default True
        If True, validate that the third cube axis corresponds to the
        Cartesian z direction, i.e. that the notebook assumption used for
        charge-vs-z analysis is satisfied.
    alignment_tol : float, default 1e-8
        Tolerance used for the axis-alignment check in bohr.

    Returns
    -------
    z_ang : numpy.ndarray
        z coordinates in Å.
    lambda_z_ang : numpy.ndarray
        In-plane integrated charge density in e / Å.

    Notes
    -----
    This function is intended for the slab-style use case in which the third
    cube axis is the physical z direction. It therefore computes the charge
    profile by integrating over the first two grid directions for each z slice.

    If the cube is written with decreasing z, the returned arrays are flipped
    so that ``z_ang`` is always increasing.
    """
    rho = np.asarray(rho, dtype=float)

    if rho.ndim != 3:
        raise ValueError(f"`rho` must be a 3D array, got shape {rho.shape}.")

    try:
        origin_tokens = header_lines[2].split()
        xgrid_tokens = header_lines[3].split()
        ygrid_tokens = header_lines[4].split()
        zgrid_tokens = header_lines[5].split()
    except IndexError as exc:
        raise ValueError("`header_lines` does not contain the expected 6 cube header lines.") from exc

    try:
        nx = abs(int(xgrid_tokens[0]))
        ny = abs(int(ygrid_tokens[0]))
        nz = abs(int(zgrid_tokens[0]))
    except (IndexError, ValueError) as exc:
        raise ValueError("Could not parse grid dimensions from cube header.") from exc

    if rho.shape != (nx, ny, nz):
        raise ValueError(
            f"Shape mismatch: header expects ({nx}, {ny}, {nz}), "
            f"but rho has shape {rho.shape}."
        )

    try:
        z0_bohr = float(origin_tokens[3])

        vx_bohr = np.array(list(map(float, xgrid_tokens[1:4])), dtype=float)
        vy_bohr = np.array(list(map(float, ygrid_tokens[1:4])), dtype=float)
        vz_bohr = np.array(list(map(float, zgrid_tokens[1:4])), dtype=float)
    except (IndexError, ValueError) as exc:
        raise ValueError("Could not parse origin/grid vectors from cube header.") from exc

    if check_axis_alignment:
        if (
            abs(vx_bohr[2]) > alignment_tol
            or abs(vy_bohr[2]) > alignment_tol
            or np.linalg.norm(vz_bohr[:2]) > alignment_tol
        ):
            raise ValueError(
                "z_charge_density_profile assumes that the third cube axis is the "
                "Cartesian z direction. This cube does not satisfy that assumption."
            )

    dz_bohr = float(vz_bohr[2])
    if abs(dz_bohr) <= alignment_tol:
        raise ValueError(
            "The third cube axis has zero Cartesian z component, so a z-profile "
            "cannot be constructed with this function."
        )

    # Area element of one (x, y) grid cell in bohr^2.
    # This is more robust than dx * dy and also works for skewed x/y planes.
    dA_xy_bohr2 = np.linalg.norm(np.cross(vx_bohr, vy_bohr))

    # Integrate rho over x and y for each z slice -> e / bohr
    lambda_z_bohr = rho.sum(axis=(0, 1)) * dA_xy_bohr2

    # z coordinates in bohr, then Å
    z_bohr = z0_bohr + np.arange(nz, dtype=float) * dz_bohr

    # Keep output ordered with increasing z
    if z_bohr[0] > z_bohr[-1]:
        z_bohr = z_bohr[::-1]
        lambda_z_bohr = lambda_z_bohr[::-1]

    z_ang = z_bohr * bohr_to_ang
    lambda_z_ang = lambda_z_bohr / bohr_to_ang

    return z_ang, lambda_z_ang

def cumulative_charge_z(
    header_lines,
    rho,
    bohr_to_ang,
    zmin=None,
    zmax=None,
    check_axis_alignment=True,
    alignment_tol=1e-8,
):
    """
    Compute the cumulative in-plane integrated charge along z.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines as returned by ``read_cube_full``.
    rho : numpy.ndarray
        Charge density array with shape (nx, ny, nz), in e / bohr^3.
    bohr_to_ang : float
        Bohr-to-Å conversion factor.
    zmin, zmax : float or None, optional
        Optional lower/upper bounds in Å for selecting the z interval over
        which the cumulative charge is computed. If omitted, use the full z range.
    check_axis_alignment : bool, default True
        If True, validate that the third cube axis corresponds to the
        Cartesian z direction.
    alignment_tol : float, default 1e-8
        Tolerance used for the axis-alignment check in bohr.

    Returns
    -------
    z_sel_ang : numpy.ndarray
        Selected z coordinates in Å, ordered increasingly.
    Qz : numpy.ndarray
        Cumulative charge in electrons, obtained by integrating the
        in-plane charge profile from the beginning of the selected interval
        up to each z point.

    Notes
    -----
    This function is built on top of ``z_charge_density_profile``. Since
    ``lambda(z)`` is returned in e / Å, the cumulative integral over z is
    naturally in electrons.

    The integral is evaluated on the cube grid as a left-Riemann sum,
    consistent with the discrete cube representation:
        Q(z_k) = sum_{i <= k} lambda(z_i) * dz
    """
    z_ang, lambda_z_ang = z_charge_density_profile(
        header_lines=header_lines,
        rho=rho,
        bohr_to_ang=bohr_to_ang,
        check_axis_alignment=check_axis_alignment,
        alignment_tol=alignment_tol,
    )

    if z_ang.ndim != 1 or lambda_z_ang.ndim != 1 or z_ang.size != lambda_z_ang.size:
        raise ValueError("Internal error: z profile arrays must be 1D and have the same length.")

    if z_ang.size == 0:
        raise ValueError("Empty z profile.")

    mask = np.ones_like(z_ang, dtype=bool)
    if zmin is not None:
        mask &= z_ang >= float(zmin)
    if zmax is not None:
        mask &= z_ang <= float(zmax)

    if not np.any(mask):
        raise ValueError(
            "The requested z interval does not contain any grid points."
        )

    z_sel_ang = z_ang[mask]
    lambda_sel_ang = lambda_z_ang[mask]

    if z_sel_ang.size == 1:
        # Degenerate interval with a single grid point
        Qz = np.array([0.0], dtype=float)
        return z_sel_ang, Qz

    dz = np.diff(z_sel_ang)
    if np.any(dz <= 0):
        raise ValueError("z grid is not strictly increasing.")

    # For a regular cube grid, dz should be constant; allow tiny numerical noise.
    dz0 = float(np.mean(dz))
    if not np.allclose(dz, dz0, rtol=0.0, atol=1e-12):
        raise ValueError("Selected z grid is not uniform, which is unexpected for a cube file.")

    Qz = np.cumsum(lambda_sel_ang) * dz0

    return z_sel_ang, Qz

def z_at_charge(
    header_lines,
    rho,
    target_charge,
    bohr_to_ang,
    zmin=None,
    zmax=None,
    check_axis_alignment=True,
    alignment_tol=1e-8,
):
    """
    Find the z coordinate at which the cumulative charge reaches a target value.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines as returned by ``read_cube_full``.
    rho : numpy.ndarray
        Charge density array with shape (nx, ny, nz), in e / bohr^3.
    target_charge : float
        Target cumulative charge in electrons.
    bohr_to_ang : float
        Bohr-to-Å conversion factor.
    zmin, zmax : float or None, optional
        Optional lower/upper bounds in Å defining the z interval in which the
        cumulative charge is constructed.
    check_axis_alignment : bool, default True
        If True, validate that the third cube axis corresponds to the
        Cartesian z direction.
    alignment_tol : float, default 1e-8
        Tolerance used for the axis-alignment check in bohr.

    Returns
    -------
    z_target_ang : float
        Interpolated z coordinate in Å at which the cumulative charge reaches
        ``target_charge``.

    Notes
    -----
    The cumulative charge is computed on the discrete z grid using
    ``cumulative_charge_z`` and then linearly interpolated between the two
    neighboring grid points that bracket the target value.

    If the target charge coincides with a grid value (within floating-point
    precision), the corresponding grid z is returned directly.
    """
    target_charge = float(target_charge)

    z_ang, Qz = cumulative_charge_z(
        header_lines=header_lines,
        rho=rho,
        bohr_to_ang=bohr_to_ang,
        zmin=zmin,
        zmax=zmax,
        check_axis_alignment=check_axis_alignment,
        alignment_tol=alignment_tol,
    )

    if z_ang.size == 0:
        raise ValueError("Empty z interval.")

    if z_ang.size == 1:
        if np.isclose(Qz[0], target_charge):
            return float(z_ang[0])
        raise ValueError(
            f"Target charge {target_charge} e is outside the cumulative-charge "
            f"range available in the selected interval."
        )

    qmin = float(np.min(Qz))
    qmax = float(np.max(Qz))
    if target_charge < qmin or target_charge > qmax:
        raise ValueError(
            f"Target charge {target_charge} e is outside the cumulative-charge "
            f"range [{qmin}, {qmax}] e of the selected interval."
        )

    idx_exact = np.where(np.isclose(Qz, target_charge, rtol=0.0, atol=1e-14))[0]
    if idx_exact.size:
        return float(z_ang[idx_exact[0]])

    idx_hi = int(np.searchsorted(Qz, target_charge, side="left"))
    if idx_hi == 0 or idx_hi >= Qz.size:
        raise ValueError(
            "Could not bracket the target charge for interpolation."
        )

    idx_lo = idx_hi - 1
    q1, q2 = float(Qz[idx_lo]), float(Qz[idx_hi])
    z1, z2 = float(z_ang[idx_lo]), float(z_ang[idx_hi])

    if np.isclose(q2, q1):
        raise ValueError(
            "Cannot interpolate z at target charge because consecutive cumulative "
            "charge values are equal."
        )

    frac = (target_charge - q1) / (q2 - q1)
    z_target_ang = z1 + frac * (z2 - z1)

    return float(z_target_ang)

def charge_at_z(
    header_lines,
    rho,
    z_value,
    bohr_to_ang,
    zmin=None,
    zmax=None,
    check_axis_alignment=True,
    alignment_tol=1e-8,
):
    """
    Return the cumulative charge at a given z coordinate.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines as returned by ``read_cube_full``.
    rho : numpy.ndarray
        Charge density array with shape (nx, ny, nz), in e / bohr^3.
    z_value : float
        z coordinate in Å at which the cumulative charge should be evaluated.
    bohr_to_ang : float
        Bohr-to-Å conversion factor.
    zmin, zmax : float or None, optional
        Optional lower/upper bounds in Å defining the z interval in which the
        cumulative charge is constructed.
    check_axis_alignment : bool, default True
        If True, validate that the third cube axis corresponds to the
        Cartesian z direction.
    alignment_tol : float, default 1e-8
        Tolerance used for the axis-alignment check in bohr.

    Returns
    -------
    charge : float
        Interpolated cumulative charge in electrons at ``z_value``.

    Notes
    -----
    The cumulative charge is first computed on the discrete z grid using
    ``cumulative_charge_z`` and then linearly interpolated to ``z_value``.

    If ``z_value`` coincides with a grid point (within floating-point
    precision), the corresponding cumulative charge is returned directly.
    """
    z_value = float(z_value)

    z_ang, Qz = cumulative_charge_z(
        header_lines=header_lines,
        rho=rho,
        bohr_to_ang=bohr_to_ang,
        zmin=zmin,
        zmax=zmax,
        check_axis_alignment=check_axis_alignment,
        alignment_tol=alignment_tol,
    )

    if z_ang.size == 0:
        raise ValueError("Empty z interval.")

    if z_value < float(z_ang[0]) or z_value > float(z_ang[-1]):
        raise ValueError(
            f"Requested z = {z_value} Å is outside the available range "
            f"[{float(z_ang[0])}, {float(z_ang[-1])}] Å."
        )

    idx_exact = np.where(np.isclose(z_ang, z_value, rtol=0.0, atol=1e-14))[0]
    if idx_exact.size:
        return float(Qz[idx_exact[0]])

    idx_hi = int(np.searchsorted(z_ang, z_value, side="left"))
    if idx_hi == 0 or idx_hi >= z_ang.size:
        raise ValueError(
            "Could not bracket the requested z value for interpolation."
        )

    idx_lo = idx_hi - 1
    z1, z2 = float(z_ang[idx_lo]), float(z_ang[idx_hi])
    q1, q2 = float(Qz[idx_lo]), float(Qz[idx_hi])

    if np.isclose(z2, z1):
        raise ValueError(
            "Cannot interpolate cumulative charge because consecutive z values are equal."
        )

    frac = (z_value - z1) / (z2 - z1)
    charge = q1 + frac * (q2 - q1)

    return float(charge)