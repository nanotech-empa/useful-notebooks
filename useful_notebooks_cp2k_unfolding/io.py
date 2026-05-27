from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import scipy.sparse as sp
from scipy import constants

hartree_to_ev = constants.physical_constants["Hartree energy in eV"][0]


@dataclass
class SupercellWavefunctions:
    evals_ev: list[np.ndarray]
    occs: list[np.ndarray]
    coeffs: list[np.ndarray]  # coeffs[ispin][imo, iao]
    ref_energy_ev: float


def read_cp2k_wfn(
    wfn_path: str | Path,
    *,
    emin: float | None = None,
    emax: float | None = None,
    n_occ: int | None = None,
    n_virt: int | None = None,
) -> SupercellWavefunctions:
    try:
        from cp2k_spm_tools.cp2k_wfn_file import Cp2kWfnFile
    except ImportError as exc:
        raise ImportError(
            "cp2k-spm-tools is required. Install it or add the repository to PYTHONPATH."
        ) from exc

    cwf = Cp2kWfnFile(mpi_rank=0, mpi_size=1, mpi_comm=None)
    try:
        cwf.load_restart_wfn_file(
            str(wfn_path),
            emin=emin,
            emax=emax,
            n_occ=n_occ,
            n_virt=n_virt,
        )
    except IndexError as exc:
        raise RuntimeError(
            "Failed while reading the CP2K WFN. Most likely the .wfn contains "
            "occupied orbitals only, so cp2k-spm-tools cannot access the LUMO "
            "to define the HOMO-LUMO reference energy. Re-run CP2K with ADDED_MOS > 0."
        ) from exc

    return SupercellWavefunctions(
        evals_ev=[np.asarray(x, dtype=float) for x in cwf.evals_sel],
        occs=[np.asarray(x, dtype=float) for x in cwf.occs_sel],
        coeffs=[np.asarray(x, dtype=float) for x in cwf.coef_array],
        ref_energy_ev=float(cwf.ref_energy),
    )


def print_eigenvalue_summary(wfn: SupercellWavefunctions, spin: int = 0, n: int = 10) -> None:
    ev = wfn.evals_ev[spin]
    eh = ev / hartree_to_ev

    print(f"Hartree to eV: {hartree_to_ev:.12f}")
    print(f"Number of selected eigenvalues: {len(ev)}")
    print(f"Reference energy: {wfn.ref_energy_ev:.10f} eV")
    print(f"Reference energy: {wfn.ref_energy_ev / hartree_to_ev:.10f} Ha")
    print()

    print(f"First {min(n, len(ev))} selected eigenvalues:")
    for i, (e_ev, e_ha) in enumerate(zip(ev[:n], eh[:n])):
        print(f"{i:4d}  {e_ev:16.10f} eV   {e_ha:16.10f} Ha")

    print()

    print(f"Last {min(n, len(ev))} selected eigenvalues:")
    offset = max(0, len(ev) - n)
    for i, (e_ev, e_ha) in enumerate(zip(ev[-n:], eh[-n:]), start=offset):
        print(f"{i:4d}  {e_ev:16.10f} eV   {e_ha:16.10f} Ha")


@dataclass
class Cp2kOverlapMatrixLog:
    matrix: sp.csr_matrix
    basis_index: np.ndarray
    atom_index: np.ndarray
    element: np.ndarray
    orbital: np.ndarray


def parse_cp2k_overlap_matrix_log_data(
    path: str | Path,
    nao: int | None = None,
    *,
    threshold: float | None = None,
) -> Cp2kOverlapMatrixLog:
    """Parse CP2K human-readable OVERLAP MATRIX blocks and AO row metadata."""
    path = Path(path)
    float_re = re.compile(r"^[+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+)(?:[EeDd][+-]?[0-9]+)?$")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    basis: dict[int, tuple[int, str, str]] = {}
    current_cols: list[int] | None = None
    inside = False
    max_index = 0
    threshold_value = 0.0 if threshold is None else float(threshold)

    def is_int_token(tok: str) -> bool:
        return tok.isdigit()

    def is_float_token(tok: str) -> bool:
        return bool(float_re.match(tok))

    with path.open("r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if "OVERLAP MATRIX" in line:
                inside = True
                current_cols = None
                continue

            if not inside:
                continue

            parts = line.split()

            if parts and all(is_int_token(tok) for tok in parts):
                current_cols = [int(tok) - 1 for tok in parts]
                max_index = max(max_index, max(current_cols) + 1)
                continue

            if current_cols is None or len(parts) < 5:
                continue
            if not (is_int_token(parts[0]) and is_int_token(parts[1])):
                continue

            value_tokens = parts[4:]
            if len(value_tokens) != len(current_cols):
                continue
            if not all(is_float_token(tok) for tok in value_tokens):
                continue

            irow = int(parts[0]) - 1
            atom = int(parts[1])
            element = parts[2]
            orbital = parts[3]
            basis[irow] = (atom, element, orbital)
            max_index = max(max_index, irow + 1)
            for jcol, tok in zip(current_cols, value_tokens):
                value = float(tok.replace("D", "E").replace("d", "e"))
                if abs(value) > threshold_value:
                    rows.append(irow)
                    cols.append(jcol)
                    vals.append(value)

    if not basis:
        raise ValueError(f"No overlap-matrix entries found in {path}")

    n = int(nao) if nao is not None else max_index
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    basis_index = np.arange(1, n + 1, dtype=np.int64)
    atom_index = np.zeros(n, dtype=np.int64)
    elements = np.full(n, "", dtype="U8")
    orbitals = np.full(n, "", dtype="U16")
    for irow, (atom, element, orbital) in basis.items():
        if irow < n:
            atom_index[irow] = atom
            elements[irow] = element
            orbitals[irow] = orbital

    return Cp2kOverlapMatrixLog(
        matrix=matrix,
        basis_index=basis_index,
        atom_index=atom_index,
        element=elements,
        orbital=orbitals,
    )


def parse_cp2k_overlap_matrix_log(path: str | Path, nao: int | None = None) -> sp.csr_matrix:
    """Parse CP2K human-readable OVERLAP MATRIX blocks as a sparse CSR matrix."""
    return parse_cp2k_overlap_matrix_log_data(path, nao).matrix


def read_sparse_overlap_npz(path_or_file) -> Cp2kOverlapMatrixLog:
    """Read sparse CP2K overlap data written by ``write_sparse_overlap_npz``."""
    with np.load(path_or_file) as data:
        arrays = {key: data[key] for key in data.files}

    matrix = sp.coo_matrix(
        (arrays["data"], (arrays["row"], arrays["col"])),
        shape=tuple(arrays["shape"]),
    ).tocsr()
    return Cp2kOverlapMatrixLog(
        matrix=matrix,
        basis_index=arrays["basis_index"],
        atom_index=arrays["atom_index"],
        element=arrays["element"],
        orbital=arrays["orbital"],
    )


def write_sparse_overlap_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float = 0.0,
    nao: int | None = None,
) -> None:
    """Write CP2K overlap data as compressed COO arrays plus AO metadata."""
    parsed = parse_cp2k_overlap_matrix_log_data(
        input_path, nao=nao, threshold=threshold
    )
    matrix = parsed.matrix.tocoo()
    np.savez_compressed(
        output_path,
        row=matrix.row.astype(np.int64),
        col=matrix.col.astype(np.int64),
        data=matrix.data.astype(np.float64),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        basis_index=parsed.basis_index,
        atom_index=parsed.atom_index,
        element=parsed.element,
        orbital=parsed.orbital,
        threshold=np.asarray(threshold, dtype=np.float64),
    )


def read_xyz_coordinates(path: str | Path) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    lines = path.read_text().splitlines()
    natom = int(lines[0].split()[0])

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natom]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.asarray(coords, dtype=float)


def parse_cp2k_cell_vectors(cp2k_input_file: str | Path, dim: int | None = None) -> np.ndarray:
    """Parse CP2K &CELL vectors A/B/C or ABC from an input file."""
    path = Path(cp2k_input_file)
    if not path.exists():
        raise FileNotFoundError(f"CP2K input file not found: {path}")

    lines = path.read_text(errors="replace").splitlines()
    in_cell = False
    vectors: dict[str, np.ndarray] = {}
    abc = None

    def strip_unit(tokens: list[str]) -> tuple[str | None, list[str]]:
        if tokens and tokens[0].startswith("[") and tokens[0].endswith("]"):
            return tokens[0].strip("[]").lower(), tokens[1:]
        return None, tokens

    for raw in lines:
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()

        if key == "&CELL":
            in_cell = True
            continue
        if in_cell and key.startswith("&END"):
            in_cell = False
            continue
        if not in_cell:
            continue

        if key in {"A", "B", "C"}:
            _, values = strip_unit(parts[1:])
            if len(values) >= 3:
                vectors[key] = np.array([float(x) for x in values[:3]], dtype=float)
        elif key == "ABC":
            _, values = strip_unit(parts[1:])
            if len(values) >= 3:
                abc = np.array([float(x) for x in values[:3]], dtype=float)

    if all(k in vectors for k in ("A", "B", "C")):
        cell = np.vstack([vectors["A"], vectors["B"], vectors["C"]])
    elif abc is not None:
        cell = np.diag(abc)
    else:
        raise ValueError("Could not parse CP2K cell vectors from input file")

    if dim is None:
        dim = 3
    return cell[:dim]
