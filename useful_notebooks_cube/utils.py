import numpy as np
import re
from pathlib import Path

def _parse_point(text):
    """
    Parse a 3D point from flexible text formats such as:
      '0.1 1 3'
      '0.1, 1, 3'
      '0.1 1 ; 3'
      '0.1 ; 1 ; 3'
    """
    if text is None:
        raise ValueError("Point text is empty.")

    s = str(text).strip()
    if not s:
        raise ValueError("Point text is empty.")

    # Extract numbers robustly, including scientific notation
    nums = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', s)

    if len(nums) != 3:
        raise ValueError(
            f"Could not parse 3 coordinates from {text!r}. "
            "Examples: '0.1 1 3' or '0.1, 1, 3'."
        )

    return np.array([float(x) for x in nums], dtype=float)

def _float_or_none(text):
    s = str(text).strip()
    return None if s == "" else float(s)

def _collect_cube_files(search_dir):
    search_dir = Path(search_dir)
    patterns = ["*.cube", "*.CUBE", "*.cub", "*.CUB"]
    files = []
    for patt in patterns:
        files.extend(search_dir.glob(patt))
    files = sorted(set(files), key=lambda p: p.name.lower())
    return files

parse_point = _parse_point
float_or_none = _float_or_none
collect_cube_files = _collect_cube_files