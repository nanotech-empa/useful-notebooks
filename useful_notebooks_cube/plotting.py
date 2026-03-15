import matplotlib.pyplot as plt


def plot_line_profile(
    result,
    title=None,
    figsize=(7, 4.5),
    ax=None,
    show=True,
    **plot_kwargs,
):
    """
    Plot a 1D line profile returned by ``cube_plane_average_profile``.

    Parameters
    ----------
    result : dict
        Result dictionary returned by ``cube_plane_average_profile``.
        It must contain at least:
        - ``l_ang``
        - ``profile``
        - ``xlabel``
        - ``ylabel``
    title : str or None, optional
        Plot title.
    figsize : tuple, default (7, 4.5)
        Figure size used only when ``ax`` is None.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on. If None, create a new figure and axes.
    show : bool, default True
        If True, call ``plt.show()`` when a new figure is created.
    **plot_kwargs
        Additional keyword arguments passed to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Notes
    -----
    This function is intentionally lightweight so it can be used both in
    notebooks and in scripts. If you want to customize line color, style,
    markers, etc., pass the corresponding matplotlib keyword arguments.
    """
    required_keys = ("l_ang", "profile", "xlabel", "ylabel")
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise KeyError(
            f"Missing keys in `result`: {missing}. "
            "Expected output from `cube_plane_average_profile`."
        )

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if "lw" not in plot_kwargs and "linewidth" not in plot_kwargs:
        plot_kwargs["lw"] = 1.8

    ax.plot(result["l_ang"], result["profile"], **plot_kwargs)
    ax.set_xlabel(result["xlabel"])
    ax.set_ylabel(result["ylabel"])

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        if show:
            plt.show()

    return fig, ax


def plot_plane_map(
    result,
    title=None,
    figsize=(6.2, 5.2),
    ax=None,
    show=True,
    cmap="viridis",
    shading="auto",
    colorbar=True,
    colorbar_kwargs=None,
    **pcolormesh_kwargs,
):
    """
    Plot a 2D perpendicular-plane map returned by
    ``cube_perpendicular_plane_map``.

    Parameters
    ----------
    result : dict
        Result dictionary returned by ``cube_perpendicular_plane_map``.
        It must contain at least:
        - ``u_ang_grid``
        - ``v_ang_grid``
        - ``map_2d``
        - ``xlabel``
        - ``ylabel``
        - ``colorbar_label``
    title : str or None, optional
        Plot title.
    figsize : tuple, default (6.2, 5.2)
        Figure size used only when ``ax`` is None.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on. If None, create a new figure and axes.
    show : bool, default True
        If True, call ``plt.show()`` when a new figure is created.
    cmap : str, default "viridis"
        Matplotlib colormap name.
    shading : str, default "auto"
        Value passed to ``ax.pcolormesh``.
    colorbar : bool, default True
        If True, add a colorbar.
    colorbar_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``fig.colorbar``.
    **pcolormesh_kwargs
        Additional keyword arguments passed to ``ax.pcolormesh``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    mesh : matplotlib.collections.QuadMesh
        The pcolormesh artist.

    Notes
    -----
    The function expects a regular result dictionary produced by
    ``cube_perpendicular_plane_map``. It uses the supplied physical labels
    directly, so charge-density and potential maps are labeled correctly.
    """
    required_keys = (
        "u_ang_grid",
        "v_ang_grid",
        "map_2d",
        "xlabel",
        "ylabel",
        "colorbar_label",
    )
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise KeyError(
            f"Missing keys in `result`: {missing}. "
            "Expected output from `cube_perpendicular_plane_map`."
        )

    U = np.asarray(result["u_ang_grid"], dtype=float)
    V = np.asarray(result["v_ang_grid"], dtype=float)
    M = np.asarray(result["map_2d"], dtype=float)

    if U.shape != V.shape or U.shape != M.shape:
        raise ValueError(
            f"Shape mismatch: got U {U.shape}, V {V.shape}, map {M.shape}. "
            "All must have the same shape."
        )

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        U,
        V,
        M,
        cmap=cmap,
        shading=shading,
        **pcolormesh_kwargs,
    )

    ax.set_xlabel(result["xlabel"])
    ax.set_ylabel(result["ylabel"])

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal")
    ax.grid(False)

    if colorbar:
        if colorbar_kwargs is None:
            colorbar_kwargs = {}
        cbar = fig.colorbar(mesh, ax=ax, **colorbar_kwargs)
        cbar.set_label(result["colorbar_label"])

    if created_fig:
        fig.tight_layout()
        if show:
            plt.show()

    return fig, ax, mesh