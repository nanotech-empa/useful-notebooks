#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array


def inspect(path: Path) -> None:
    with np.load(path) as data:
        print(path)
        print(f"files: {', '.join(data.files)}")
        for key in (
            "dim",
            "lattice_type",
            "primitive_vectors",
            "supercell_vectors",
            "supercell_matrix",
            "correction_norm",
            "ref_energy_ev",
        ):
            if key in data:
                print(f"{key}: {_scalar(data[key])}")
        if "supercell_matrix" in data:
            matrix = np.asarray(data["supercell_matrix"], dtype=float)
            print(f"det(supercell_matrix): {round(abs(np.linalg.det(matrix)))}")
        if "k_frac_folded" in data:
            print(f"folded k-points: {len(data['k_frac_folded'])}")
        if "path_k_indices" in data:
            print(f"k-points on path: {len(data['path_k_indices'])}")

        if "primitive_basis_atom_indices" in data:
            print(f"primitive_basis_atom_indices: {data['primitive_basis_atom_indices']}")
        for spin in range(8):
            dkey = f"atom_mapping_displacements_spin_{spin}"
            if dkey not in data:
                continue
            disp = np.linalg.norm(data[dkey], axis=1)
            print(f"atom mapping spin {spin} max/mean [A]: {disp.max():.8g} {disp.mean():.8g}")
            print(f"atom mapping spin {spin} worst atom [1-based]: {int(np.argmax(disp)) + 1}")
        for spin in range(8):
            wkey = f"weights_spin_{spin}"
            ekey = f"evals_ev_spin_{spin}"
            if wkey not in data:
                continue
            weights = data[wkey]
            evals = data[ekey] if ekey in data else None
            print()
            print(f"spin {spin}")
            print(f"  weights shape: {weights.shape}")
            print(f"  weights min/max/std: {weights.min():.8g} {weights.max():.8g} {weights.std():.8g}")
            if evals is not None:
                print(f"  evals shape: {evals.shape}")
                print(f"  evals min/max: {evals.min():.8g} {evals.max():.8g}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print compact diagnostics for an unfolding NPZ.")
    parser.add_argument("npz", nargs="+")
    args = parser.parse_args()
    for item in args.npz:
        inspect(Path(item))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
