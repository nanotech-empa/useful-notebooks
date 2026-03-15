from pathlib import Path
from collections.abc import Mapping, Sequence
import ast
import numpy as np

from .io import read_cube_full, read_cube_full_cached, write_cube
from .constants import bohr_to_ang as BOHR_TO_ANG
from .analysis import cumulative_charge_z, z_at_charge, charge_at_z


def read_cubes_same_grid(
    cube_files, use_cache=True, rtol=0.0, atol=1e-12, verbose=False
):
    """
    Read several cube files and assert that they share the same volumetric grid.

    Parameters
    ----------
    cube_files : mapping or sequence
        Either:
        - a mapping ``{label: path}``, which is the recommended form, or
        - a sequence of paths, in which case labels ``cube1``, ``cube2``, ...
          are assigned automatically.

        The labels are later used for cube algebra expressions such as
        ``cube1 - cube2 - cube3`` or ``2*cube1 - cube2**2 + 3*cube3``.
    use_cache : bool, default True
        If True, use ``read_cube_full_cached``.
    rtol : float, default 0.0
        Relative tolerance used when comparing origins and grid vectors.
    atol : float, default 1e-12
        Absolute tolerance used when comparing origins and grid vectors.
    verbose : bool, default False
        If True, print which files are being read.

    Returns
    -------
    cube_set : dict
        Dictionary with the following keys:

        - ``labels`` : list[str]
            Cube labels in the preserved input order.
        - ``reference_label`` : str
            Label of the first cube, used as the default reference cube.
        - ``grid_shape`` : tuple[int, int, int]
            Shared grid shape.
        - ``origin_bohr`` : ndarray, shape (3,)
            Shared grid origin in bohr.
        - ``step_matrix_bohr`` : ndarray, shape (3, 3)
            Shared voxel-step matrix in bohr, with the three cube step vectors
            as columns.
        - ``cubes`` : dict
            Mapping ``label -> cube_record`` where each record contains:
              - ``label``
              - ``path``
              - ``header_lines``
              - ``atom_lines``
              - ``rho``
              - ``grid_shape``

    Notes
    -----
    Cubes are considered compatible if they have the same:
    - grid shape
    - origin
    - three voxel step vectors

    The atom list is *not* required to match, because in workflows such as
    slab / molecule / combined-system charge decomposition the atoms may differ.

    The first cube is the reference cube. Later algebra-writing helpers should
    keep its header and atom list unless explicitly requested otherwise.
    """
    if isinstance(cube_files, Mapping):
        items = list(cube_files.items())
    elif isinstance(cube_files, Sequence) and not isinstance(
        cube_files, (str, bytes, Path)
    ):
        items = [(f"cube{i+1}", path) for i, path in enumerate(cube_files)]
    else:
        raise TypeError(
            "`cube_files` must be either a mapping {label: path} or a sequence of paths."
        )

    if not items:
        raise ValueError("No cube files were provided.")

    read_fn = read_cube_full_cached if use_cache else read_cube_full

    def _parse_grid_signature(header_lines, grid_shape):
        """
        Extract only the grid-defining information, ignoring comments and atoms.
        """
        try:
            origin_tokens = header_lines[2].split()
            xgrid_tokens = header_lines[3].split()
            ygrid_tokens = header_lines[4].split()
            zgrid_tokens = header_lines[5].split()
        except IndexError as exc:
            raise ValueError(
                "Cube header does not contain the required geometry lines."
            ) from exc

        try:
            origin_bohr = np.array(list(map(float, origin_tokens[1:4])), dtype=float)

            nx_h = abs(int(xgrid_tokens[0]))
            ny_h = abs(int(ygrid_tokens[0]))
            nz_h = abs(int(zgrid_tokens[0]))

            vx_bohr = np.array(list(map(float, xgrid_tokens[1:4])), dtype=float)
            vy_bohr = np.array(list(map(float, ygrid_tokens[1:4])), dtype=float)
            vz_bohr = np.array(list(map(float, zgrid_tokens[1:4])), dtype=float)
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "Failed to parse cube grid information from header."
            ) from exc

        if tuple(grid_shape) != (nx_h, ny_h, nz_h):
            raise ValueError(
                f"Inconsistent cube header/data shape: header has ({nx_h}, {ny_h}, {nz_h}) "
                f"but reader returned {tuple(grid_shape)}."
            )

        step_matrix_bohr = np.column_stack([vx_bohr, vy_bohr, vz_bohr])

        return origin_bohr, step_matrix_bohr

    cubes = {}
    labels = []

    ref_label = None
    ref_shape = None
    ref_origin_bohr = None
    ref_step_matrix_bohr = None

    for label, path in items:
        label = str(label)
        path = str(Path(path).expanduser())

        if label in cubes:
            raise ValueError(f"Duplicate cube label: {label!r}")

        if verbose:
            print(f"Reading {label}: {path}")

        header_lines, atom_lines, rho, grid_shape = read_fn(path)
        origin_bohr, step_matrix_bohr = _parse_grid_signature(header_lines, grid_shape)

        if ref_label is None:
            ref_label = label
            ref_shape = tuple(grid_shape)
            ref_origin_bohr = origin_bohr
            ref_step_matrix_bohr = step_matrix_bohr
        else:
            if tuple(grid_shape) != ref_shape:
                raise ValueError(
                    f"Grid-shape mismatch for {label!r}: got {tuple(grid_shape)}, "
                    f"expected {ref_shape} from {ref_label!r}."
                )

            if not np.allclose(origin_bohr, ref_origin_bohr, rtol=rtol, atol=atol):
                raise ValueError(
                    f"Origin mismatch for {label!r} relative to {ref_label!r}."
                )

            if not np.allclose(
                step_matrix_bohr, ref_step_matrix_bohr, rtol=rtol, atol=atol
            ):
                raise ValueError(
                    f"Grid-vector mismatch for {label!r} relative to {ref_label!r}."
                )

        cubes[label] = {
            "label": label,
            "path": path,
            "header_lines": header_lines,
            "atom_lines": atom_lines,
            "rho": rho,
            "grid_shape": tuple(grid_shape),
        }
        labels.append(label)

    return {
        "labels": labels,
        "reference_label": ref_label,
        "grid_shape": ref_shape,
        "origin_bohr": ref_origin_bohr,
        "step_matrix_bohr": ref_step_matrix_bohr,
        "cubes": cubes,
    }


def evaluate_cube_expression(cube_set, expression):
    """
    Evaluate an algebraic expression on a compatible set of cubes.

    Parameters
    ----------
    cube_set : dict
        Output of ``read_cubes_same_grid(...)``.
    expression : str
        Algebraic expression involving cube labels and scalar constants, e.g.
        - ``"cube1 - cube2 - cube3"``
        - ``"2*cube1 - cube2**2 + 3*cube3"``
        - ``"(cube1 - cube2) / 2"``
        - ``"-cube1 + 0.5*cube2"``

    Returns
    -------
    rho : numpy.ndarray
        Evaluated cube array with the same shape as the input cubes.

    Notes
    -----
    Allowed syntax is intentionally restricted to simple arithmetic:
    - binary operators: ``+``, ``-``, ``*``, ``/``, ``**``
    - unary operators: ``+``, ``-``
    - parentheses
    - scalar numeric constants
    - cube labels that are valid Python identifiers

    Labels such as ``cube1``, ``slab4`` or ``mol_A`` work. Labels containing
    spaces, hyphens, or other non-identifier characters cannot be used in the
    expression language.

    No function calls are allowed here. This is deliberate, both for safety
    and to keep the algebra transparent and reproducible.
    """
    if not isinstance(expression, str):
        raise TypeError("`expression` must be a string.")

    expr = expression.strip()
    if not expr:
        raise ValueError("`expression` is empty.")

    if "cubes" not in cube_set or "labels" not in cube_set:
        raise ValueError(
            "`cube_set` does not look like the output of `read_cubes_same_grid(...)`."
        )

    cubes = cube_set["cubes"]
    labels = list(cube_set["labels"])

    invalid_labels = [label for label in labels if not str(label).isidentifier()]
    if invalid_labels:
        raise ValueError(
            "The following cube labels are not valid Python identifiers and therefore "
            f"cannot be used in expressions: {invalid_labels}. "
            "Use labels such as 'cube1', 'slab4', 'mol_A'."
        )

    allowed_names = set(labels)
    name_to_array = {
        label: np.asarray(cubes[label]["rho"], dtype=float) for label in labels
    }

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid cube expression syntax: {expression!r}") from exc

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    allowed_unaryops = (ast.UAdd, ast.USub)
    allowed_constants = (int, float)

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_binops):
                raise ValueError(
                    f"Unsupported operator in expression: {ast.dump(node.op)}"
                )
            left = _eval_node(node.left)
            right = _eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unaryops):
                raise ValueError(
                    f"Unsupported unary operator in expression: {ast.dump(node.op)}"
                )
            value = _eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(
                    f"Unknown cube label {node.id!r} in expression. "
                    f"Available labels: {labels}"
                )
            return name_to_array[node.id]

        # Python >=3.8
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, allowed_constants):
                raise ValueError(f"Unsupported constant in expression: {node.value!r}")
            return float(node.value)

        # Python <3.8 compatibility style nodes, harmless to keep
        if isinstance(node, ast.Num):
            return float(node.n)

        raise ValueError(
            f"Unsupported syntax in cube expression: {ast.dump(node, include_attributes=False)}"
        )

    rho = _eval_node(tree)

    rho = np.asarray(rho, dtype=float)

    ref_shape = tuple(cube_set["grid_shape"])
    if rho.shape != ref_shape:
        raise ValueError(
            f"Evaluated expression has shape {rho.shape}, expected {ref_shape}."
        )

    return rho


def write_cube_expression(
    cube_set,
    expression,
    output_filename,
    reference_label=None,
    verbose=True,
):
    """
    Evaluate a cube algebra expression and write the result as a cube file.

    Parameters
    ----------
    cube_set : dict
        Output of ``read_cubes_same_grid(...)``.
    expression : str
        Algebraic expression involving cube labels, e.g.
        - ``"cube1 - cube2 - cube3"``
        - ``"2*cube1 - cube2**2 + 3*cube3"``
    output_filename : str or path-like
        Output cube filename.
    reference_label : str or None, optional
        Label of the cube whose header and atom list should be reused for the
        output cube. If None, use ``cube_set["reference_label"]``.
    verbose : bool, default True
        If True, print a short confirmation message.

    Returns
    -------
    rho_out : numpy.ndarray
        Evaluated cube data written to disk.
    output_filename : str
        Output filename as a string.

    Notes
    -----
    The output cube keeps:
    - the header lines of the reference cube
    - the atom list of the reference cube

    This matches the intended workflow where cube algebra is performed on
    compatible grids, while the first cube provides the structural reference.
    """
    if "cubes" not in cube_set or "reference_label" not in cube_set:
        raise ValueError(
            "`cube_set` does not look like the output of `read_cubes_same_grid(...)`."
        )

    if reference_label is None:
        reference_label = cube_set["reference_label"]

    if reference_label not in cube_set["cubes"]:
        raise ValueError(
            f"Unknown reference label {reference_label!r}. "
            f"Available labels: {cube_set['labels']}"
        )

    rho_out = evaluate_cube_expression(cube_set, expression)

    ref_cube = cube_set["cubes"][reference_label]
    header_lines = ref_cube["header_lines"]
    atom_lines = ref_cube["atom_lines"]

    output_filename = str(Path(output_filename).expanduser())

    write_cube(
        output_filename,
        header_lines=header_lines,
        atom_lines=atom_lines,
        rho=rho_out,
    )

    if verbose:
        print(
            f"Wrote cube expression to {output_filename}: "
            f"{expression}  [reference atoms/header: {reference_label}]"
        )

    return rho_out, output_filename


def plot_cumulative_charge_multi(
    cube_set,
    labels=None,
    shifts=None,
    bohr_to_ang=None,
    zmin=None,
    zmax=None,
    qmin=None,
    qmax=None,
    ax=None,
    figsize=(7, 4.5),
    show=True,
    title=None,
    legend=True,
    **plot_kwargs,
):
    """
    Plot cumulative charge Q(z) for a selected subset of cubes.

    Parameters
    ----------
    cube_set : dict
        Output of ``read_cubes_same_grid(...)``.
    labels : sequence[str] or mapping[str, str] or None, optional
        Cubes to plot.

        Accepted forms:
        - None:
            plot all cubes in input order, with their internal labels.
        - sequence of strings:
            plot only those cube labels, using the same labels in the legend.
        - mapping {internal_label: display_label}:
            plot only those cube labels, and use ``display_label`` in the legend.
    shifts : None, sequence[float], mapping[str, float], or float, optional
        Vertical shifts applied to the plotted cumulative charge, i.e. the
        plotted quantity is ``Q(z) + shift``.

        Accepted forms:
        - None:
            all shifts are zero.
        - scalar:
            apply the same shift to all selected curves.
        - sequence:
            one shift per selected label, in the same order as the selected labels.
        - mapping {internal_label: shift}:
            only listed curves are shifted; missing labels default to 0.
    bohr_to_ang : float or None, optional
        Bohr-to-Å conversion factor. If None, use the shared library constant.
    zmin, zmax : float or None, optional
        Optional x-axis limits in Å.
    qmin, qmax : float or None, optional
        Optional y-axis limits in electrons.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on. If None, create a new figure and axes.
    figsize : tuple, default (7, 4.5)
        Figure size used only when ``ax`` is None.
    show : bool, default True
        If True, call ``plt.show()`` when a new figure is created.
    title : str or None, optional
        Plot title.
    legend : bool, default True
        If True, show a legend.
    **plot_kwargs
        Extra keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    curves : dict
        Mapping ``internal_label -> (z_ang, Qz_shifted)`` for the plotted curves.

    Notes
    -----
    This helper computes Q(z) from the full cube and applies axis limits only
    at the plotting level. If shifts are provided, they are applied only for
    plotting and in the returned ``curves`` data.
    """
    import matplotlib.pyplot as plt
    from collections.abc import Mapping, Sequence

    if bohr_to_ang is None:
        bohr_to_ang = BOHR_TO_ANG

    if "cubes" not in cube_set or "labels" not in cube_set:
        raise ValueError(
            "`cube_set` does not look like the output of `read_cubes_same_grid(...)`."
        )

    available = list(cube_set["labels"])

    if labels is None:
        selected_labels = available
        display_labels = {label: label for label in selected_labels}
    elif isinstance(labels, Mapping):
        selected_labels = list(labels.keys())
        display_labels = {str(k): str(v) for k, v in labels.items()}
    else:
        selected_labels = [str(label) for label in labels]
        display_labels = {label: label for label in selected_labels}

    unknown = [label for label in selected_labels if label not in cube_set["cubes"]]
    if unknown:
        raise ValueError(
            f"Unknown cube labels requested for plotting: {unknown}. "
            f"Available labels: {available}"
        )

    if shifts is None:
        shift_map = {label: 0.0 for label in selected_labels}
    elif np.isscalar(shifts):
        shift_map = {label: float(shifts) for label in selected_labels}
    elif isinstance(shifts, Mapping):
        shift_map = {label: float(shifts.get(label, 0.0)) for label in selected_labels}
    elif isinstance(shifts, Sequence) and not isinstance(shifts, (str, bytes)):
        shifts = list(shifts)
        if len(shifts) != len(selected_labels):
            raise ValueError(
                f"When `shifts` is a sequence, it must have the same length as the "
                f"selected labels: got {len(shifts)} shifts for {len(selected_labels)} labels."
            )
        shift_map = {
            label: float(shift) for label, shift in zip(selected_labels, shifts)
        }
    else:
        raise TypeError("`shifts` must be None, a scalar, a sequence, or a mapping.")

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if "lw" not in plot_kwargs and "linewidth" not in plot_kwargs:
        plot_kwargs["lw"] = 1.8

    curves = {}

    for label in selected_labels:
        record = cube_set["cubes"][label]
        z_ang, Qz = cumulative_charge_z(
            record["header_lines"],
            record["rho"],
            bohr_to_ang,
        )

        Qz_shifted = Qz + shift_map[label]
        curves[label] = (z_ang, Qz_shifted)

        ax.plot(
            z_ang,
            Qz_shifted,
            label=display_labels[label],
            **plot_kwargs,
        )

    ax.set_xlabel("Z (Å)")
    ax.set_ylabel("Q(z) (e)")

    if zmin is not None or zmax is not None:
        ax.set_xlim(
            left=None if zmin is None else float(zmin),
            right=None if zmax is None else float(zmax),
        )

    if qmin is not None or qmax is not None:
        ax.set_ylim(
            bottom=None if qmin is None else float(qmin),
            top=None if qmax is None else float(qmax),
        )

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend()

    if created_fig:
        fig.tight_layout()
        if show:
            plt.show()

    return fig, ax, curves


def z_at_charge_multi(
    cube_set,
    target_charge,
    labels=None,
    bohr_to_ang=None,
    zmin=None,
    zmax=None,
):
    """
    For each selected cube, return the z coordinate at which the cumulative
    charge reaches a target value.

    Parameters
    ----------
    cube_set : dict
        Output of ``read_cubes_same_grid(...)``.
    target_charge : float
        Target cumulative charge in electrons.
    labels : sequence[str] or None, optional
        Cube labels to evaluate. If None, use all cubes in input order.
    bohr_to_ang : float or None, optional
        Bohr-to-Å conversion factor. If None, use the shared library constant.
    zmin, zmax : float or None, optional
        Optional bounds in Å defining the z interval over which the cumulative
        charge is constructed before locating the target charge.

    Returns
    -------
    results : dict
        Mapping ``label -> z_target_ang`` in Å.

    Notes
    -----
    This is a thin multi-cube wrapper around ``z_at_charge(...)``.
    """
    if bohr_to_ang is None:
        bohr_to_ang = BOHR_TO_ANG

    if "cubes" not in cube_set or "labels" not in cube_set:
        raise ValueError(
            "`cube_set` does not look like the output of `read_cubes_same_grid(...)`."
        )

    available = list(cube_set["labels"])
    if labels is None:
        labels = available
    else:
        labels = list(labels)

    unknown = [label for label in labels if label not in cube_set["cubes"]]
    if unknown:
        raise ValueError(
            f"Unknown cube labels requested: {unknown}. "
            f"Available labels: {available}"
        )

    target_charge = float(target_charge)

    results = {}
    for label in labels:
        record = cube_set["cubes"][label]
        results[label] = z_at_charge(
            header_lines=record["header_lines"],
            rho=record["rho"],
            target_charge=target_charge,
            bohr_to_ang=bohr_to_ang,
            zmin=zmin,
            zmax=zmax,
        )

    return results


def charge_at_z_multi(
    cube_set,
    z_value,
    labels=None,
    bohr_to_ang=None,
    zmin=None,
    zmax=None,
):
    """
    For each selected cube, return the cumulative charge at a given z coordinate.

    Parameters
    ----------
    cube_set : dict
        Output of ``read_cubes_same_grid(...)``.
    z_value : float
        z coordinate in Å at which the cumulative charge should be evaluated.
    labels : sequence[str] or None, optional
        Cube labels to evaluate. If None, use all cubes in input order.
    bohr_to_ang : float or None, optional
        Bohr-to-Å conversion factor. If None, use the shared library constant.
    zmin, zmax : float or None, optional
        Optional bounds in Å defining the z interval over which the cumulative
        charge is constructed before evaluating Q(z).

    Returns
    -------
    results : dict
        Mapping ``label -> charge`` in electrons.

    Notes
    -----
    This is a thin multi-cube wrapper around ``charge_at_z(...)``.
    """
    if bohr_to_ang is None:
        bohr_to_ang = BOHR_TO_ANG

    if "cubes" not in cube_set or "labels" not in cube_set:
        raise ValueError(
            "`cube_set` does not look like the output of `read_cubes_same_grid(...)`."
        )

    available = list(cube_set["labels"])
    if labels is None:
        labels = available
    else:
        labels = list(labels)

    unknown = [label for label in labels if label not in cube_set["cubes"]]
    if unknown:
        raise ValueError(
            f"Unknown cube labels requested: {unknown}. "
            f"Available labels: {available}"
        )

    z_value = float(z_value)

    results = {}
    for label in labels:
        record = cube_set["cubes"][label]
        results[label] = charge_at_z(
            header_lines=record["header_lines"],
            rho=record["rho"],
            z_value=z_value,
            bohr_to_ang=bohr_to_ang,
            zmin=zmin,
            zmax=zmax,
        )

    return results
