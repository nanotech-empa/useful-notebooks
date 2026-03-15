import numpy as np
import re
from pathlib import Path

def parse_point(text):
    """
    Parse a 3D point from flexible text input.

    Accepted examples
    -----------------
    - "0.1 1 3"
    - "0.1, 1, 3"
    - "0.1 ; 1 ; 3"
    - "(0.1, 1, 3)"
    - "[0.1 1 3]"
    - "1e-3, -2.5, 7"

    Parameters
    ----------
    text : str or object convertible to str
        Input text containing exactly three numeric coordinates.

    Returns
    -------
    numpy.ndarray
        Array of shape (3,) with dtype float.

    Raises
    ------
    ValueError
        If the input is empty or does not contain exactly three numbers.
    """
    if text is None:
        raise ValueError("Point text is empty.")

    s = str(text).strip()
    if not s:
        raise ValueError("Point text is empty.")

    # Extract numbers robustly, including scientific notation.
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    nums = re.findall(number_pattern, s)

    if len(nums) != 3:
        raise ValueError(
            f"Could not parse exactly 3 coordinates from {text!r}. "
            "Examples: '0.1 1 3', '0.1, 1, 3', '(0.1, 1, 3)'."
        )

    point = np.array([float(x) for x in nums], dtype=float)

    if point.shape != (3,):
        raise ValueError(
            f"Parsed point has unexpected shape {point.shape}; expected (3,)."
        )

    return point

def float_or_none(text):
    """
    Convert optional text input to float.

    Parameters
    ----------
    text : str or object convertible to str
        Input text. Empty or whitespace-only input returns ``None``.

    Returns
    -------
    float or None
        ``None`` if the input is empty, otherwise the parsed float.

    Accepted examples
    -----------------
    - ""
    - "   "
    - "1.5"
    - "-3"
    - "2e-4"

    Raises
    ------
    ValueError
        If the input is non-empty but cannot be parsed as a float.
    """
    if text is None:
        return None

    s = str(text).strip()
    if s == "":
        return None

    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse a float from {text!r}. "
            "Use an empty value for None, or a valid number such as '1.5' or '2e-4'."
        ) from exc

def collect_cube_files(download_dir=Path.home() / "Downloads", recursive=False):
    """
    Collect cube files from a directory.

    Parameters
    ----------
    download_dir : str or path-like, default Path.home() / "Downloads"
        Directory to search.
    recursive : bool, default False
        If True, search recursively under ``download_dir``.
        If False, search only the top level.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of existing ``.cube`` files.

    Notes
    -----
    Sorting is case-insensitive on the filename, then by full path, so the
    order is stable and predictable in notebook dropdowns.
    """
    base = Path(download_dir).expanduser()

    if not base.exists():
        return []

    if not base.is_dir():
        raise ValueError(f"{base} exists but is not a directory.")

    pattern = "**/*.cube" if recursive else "*.cube"
    files = [p for p in base.glob(pattern) if p.is_file()]

    return sorted(files, key=lambda p: (p.name.lower(), str(p).lower()))
