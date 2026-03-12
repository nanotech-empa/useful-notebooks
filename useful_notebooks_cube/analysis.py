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
    Parse cube header geometry.

    Returns a dictionary with:
      - origin_ang: origin in Å
      - step_matrix_ang: 3x3 matrix, columns are voxel step vectors in Å
      - cell_matrix_ang: 3x3 matrix, columns are full cell vectors in Å
      - nxyz: grid shape [nx, ny, nz]
      - inv_step_matrix_ang: inverse of step_matrix_ang
    """
    origin_tokens = header_lines[2].split()
    xgrid_tokens = header_lines[3].split()
    ygrid_tokens = header_lines[4].split()
    zgrid_tokens = header_lines[5].split()

    # cube convention: line 3 is [natoms, ox, oy, oz]
    origin_bohr = np.array(list(map(float, origin_tokens[1:4])), dtype=float)

    nx = abs(int(xgrid_tokens[0]))
    ny = abs(int(ygrid_tokens[0]))
    nz = abs(int(zgrid_tokens[0]))

    vx_bohr = np.array(list(map(float, xgrid_tokens[1:4])), dtype=float)
    vy_bohr = np.array(list(map(float, ygrid_tokens[1:4])), dtype=float)
    vz_bohr = np.array(list(map(float, zgrid_tokens[1:4])), dtype=float)

    origin_ang = origin_bohr * bohr_to_ang
    step_matrix_ang = bohr_to_ang * np.column_stack([vx_bohr, vy_bohr, vz_bohr])

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
    1D profile along the line P1 -> P2 from averages over perpendicular rectangles.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines.
    cube_values : ndarray, shape (nx, ny, nz)
        Scalar field read from cube.
    bohr_to_ang : float
        Conversion factor Bohr -> Å.
    P1, P2 : array-like, shape (3,)
        Endpoints in Å.
    field_type : {'charge', 'hartree', 'rydberg'}
        Type of scalar field.
    dl : float, default 0.1
        Target step along P1 -> P2 in Å.
    L, W : float or None
        Rectangle size in the plane perpendicular to P2-P1, in Å.
        If None, use the default largest projected cross-section of the cell.
    du, dv : float
        Target in-plane sampling steps in Å.
    order : int
        Interpolation order for scipy.ndimage.map_coordinates.

    Returns
    -------
    result : dict
        Contains profile, geometry, sampling metadata, labels, and units.
    """
    geom = _cube_geometry(header_lines, bohr_to_ang)
    values_phys, unit, quantity_name, field_type = _convert_cube_values(
        cube_values, field_type, bohr_to_ang
    )

    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    direction = P2 - P1
    axis_length = np.linalg.norm(direction)

    uhat, e1, e2 = _orthonormal_plane_basis(direction)
    L_default, W_default = _default_perp_rectangle(geom["cell_matrix_ang"], e1, e2)

    if L is None:
        L = L_default
    if W is None:
        W = W_default
    if dv is None:
        dv = du

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
        profile[i] = vals.mean()

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
    2D map in the plane perpendicular to P1 -> P2 at a given position, with
    optional Gaussian broadening along the normal direction.

    Parameters
    ----------
    header_lines : list[str]
        Cube header lines.
    cube_values : ndarray, shape (nx, ny, nz)
        Scalar field read from cube.
    bohr_to_ang : float
        Conversion factor Bohr -> Å.
    P1, P2 : array-like, shape (3,)
        Endpoints in Å defining the axis.
    position : float, default 0.0
        Position along P1 -> P2 in Å. position=0 means plane through P1.
    field_type : {'charge', 'hartree', 'rydberg'}
        Type of scalar field.
    L, W : float or None
        Rectangle size in the perpendicular plane, in Å.
        If None, use default largest projected cross-section.
    du, dv : float
        Target in-plane sampling steps in Å.
    sigma : float, default 0.0
        Gaussian broadening sigma in Å along the plane normal.
        sigma=0 means direct slice.
    dn : float or None
        Sampling step along the normal direction for the Gaussian quadrature.
        If None, choose an automatic value.
    truncate : float
        Gaussian cutoff in units of sigma.
    order : int
        Interpolation order for scipy.ndimage.map_coordinates.

    Returns
    -------
    result : dict
        Contains 2D map, geometry, sampling metadata, labels, and units.
    """
    geom = _cube_geometry(header_lines, bohr_to_ang)
    values_phys, unit, quantity_name, field_type = _convert_cube_values(
        cube_values, field_type, bohr_to_ang
    )

    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    direction = P2 - P1
    axis_length = np.linalg.norm(direction)

    uhat, e1, e2 = _orthonormal_plane_basis(direction)
    L_default, W_default = _default_perp_rectangle(geom["cell_matrix_ang"], e1, e2)

    if L is None:
        L = L_default
    if W is None:
        W = W_default
    if dv is None:
        dv = du

    u_ang, du_eff, nu = _centered_axis(L, du)
    v_ang, dv_eff, nv = _centered_axis(W, dv)

    U, V = np.meshgrid(u_ang, v_ang, indexing="ij")
    center = P1 + float(position) * uhat

    plane_points = center + U[..., None] * e1 + V[..., None] * e2
    plane_points_flat = plane_points.reshape(-1, 3)

    if sigma < 0:
        raise ValueError("sigma must be >= 0.")

    if sigma == 0:
        field_2d = _sample_cube_periodic(plane_points_flat, values_phys, geom, order=order)
        field_2d = field_2d.reshape(U.shape)
        normal_offsets = np.array([0.0])
        normal_weights = np.array([1.0])
        dn_eff = 0.0
    else:
        if dn is None:
            # reasonable automatic normal step
            dn = min(du_eff, dv_eff, max(sigma / 3.0, 0.05))
        if dn <= 0:
            raise ValueError("dn must be > 0 when sigma > 0.")
        if truncate <= 0:
            raise ValueError("truncate must be > 0.")

        n_side = max(1, int(np.ceil(truncate * sigma / dn)))
        normal_offsets = np.arange(-n_side, n_side + 1, dtype=float) * dn
        normal_weights = np.exp(-0.5 * (normal_offsets / sigma) ** 2)
        normal_weights /= normal_weights.sum()
        dn_eff = float(dn)

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
        "position_ang": float(position),

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
        "sigma_ang": float(sigma),
        "dn_ang": float(dn_eff),
        "normal_offsets_ang": normal_offsets,
        "normal_weights": normal_weights,
        "truncate": float(truncate),

        # useful references
        "origin_ang": geom["origin_ang"],
        "cell_matrix_ang": geom["cell_matrix_ang"],
        "step_matrix_ang": geom["step_matrix_ang"],
        "grid_shape": geom["nxyz"],
    }
