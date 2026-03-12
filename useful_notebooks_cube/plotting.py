import matplotlib.pyplot as plt


def plot_line_profile(result, title=None, figsize=(7, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result["l_ang"], result["profile"])
    ax.set_xlabel(result["xlabel"])
    ax.set_ylabel(result["ylabel"])
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_plane_map(result, title=None, figsize=(6.5, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        result["u_ang_grid"],
        result["v_ang_grid"],
        result["map_2d"],
        shading="auto",
    )
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(result["colorbar_label"])

    ax.set_xlabel(result["xlabel"])
    ax.set_ylabel(result["ylabel"])
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
