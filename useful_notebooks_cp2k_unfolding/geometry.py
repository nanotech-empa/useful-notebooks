from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AOMapping:
    ao_to_replica: np.ndarray
    ao_to_local: np.ndarray
    replica_vectors_cart: np.ndarray
    nao_prim: int
    basis_frac_coords: np.ndarray
    atom_to_basis: np.ndarray
    atom_to_replica: np.ndarray
    supercell_integer_matrix: np.ndarray

    @property
    def nrep(self) -> int:
        return int(self.replica_vectors_cart.shape[0])

    @property
    def nao_super(self) -> int:
        return int(self.ao_to_replica.size)


def lattice_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return a 2D/3D column-vector lattice matrix from row-vector input."""
    vectors = np.asarray(vectors, dtype=float)
    dim = vectors.shape[0]
    return vectors[:dim, :dim].T


def fractional_coordinates(coords_cart: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Fractional coordinates in the lattice spanned by vectors."""
    L = lattice_matrix(vectors)
    dim = L.shape[0]
    return np.linalg.solve(L, coords_cart[:, :dim].T).T


def modulo_one(frac: np.ndarray) -> np.ndarray:
    return frac - np.floor(frac)


def periodic_frac_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    delta = f1 - f2
    delta -= np.round(delta)
    return float(np.linalg.norm(delta))


def cluster_basis_fractional_coords(
    symbols: Sequence[str],
    coords_cart: np.ndarray,
    primitive_vectors: np.ndarray,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find unique atomic basis positions modulo primitive translations."""
    frac = modulo_one(fractional_coordinates(coords_cart, primitive_vectors))

    basis_symbols: list[str] = []
    basis_frac: list[np.ndarray] = []
    atom_to_basis = np.full(len(symbols), -1, dtype=int)

    for i, (sym, f) in enumerate(zip(symbols, frac)):
        match = None
        for ib, (bsym, bf) in enumerate(zip(basis_symbols, basis_frac)):
            if sym == bsym and periodic_frac_distance(f, bf) < tol:
                match = ib
                break

        if match is None:
            match = len(basis_frac)
            basis_symbols.append(sym)
            basis_frac.append(f)

        atom_to_basis[i] = match

    return np.asarray(basis_frac, dtype=float), atom_to_basis


def integer_supercell_matrix(primitive_vectors: np.ndarray, supercell_vectors: np.ndarray) -> np.ndarray:
    """Return integer M such that S = A @ M, with column-vector convention."""
    A = lattice_matrix(primitive_vectors)
    S = lattice_matrix(supercell_vectors)
    M_float = np.linalg.solve(A, S)
    M_int = np.rint(M_float).astype(int)

    if not np.allclose(M_float, M_int, atol=1e-5):
        raise ValueError(f"Supercell is not an integer combination of primitive vectors:\n{M_float}")

    return M_int




def primitive_vectors_from_supercell_matrix(
    supercell_vectors: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Return primitive row vectors for ``S = A @ M``."""
    matrix = np.asarray(matrix, dtype=float)
    dim = matrix.shape[0]
    supercell_lattice = lattice_matrix(supercell_vectors)
    primitive_lattice = supercell_lattice @ np.linalg.inv(matrix)
    vectors = np.zeros((dim, 3), dtype=float)
    vectors[:, :dim] = primitive_lattice.T
    return vectors


def snap_primitive_vectors_to_supercell(
    approx_vectors: np.ndarray,
    supercell_vectors: np.ndarray,
    *,
    search_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Snap approximate primitive vectors to the nearest exact supercell tiling.

    Returns ``snapped_vectors, integer_matrix, floating_matrix, correction_norm``.
    """
    approx_vectors = np.asarray(approx_vectors, dtype=float)
    supercell_vectors = np.asarray(supercell_vectors, dtype=float)
    dim = approx_vectors.shape[0]
    approx_lattice = lattice_matrix(approx_vectors)
    supercell_lattice = lattice_matrix(supercell_vectors)
    matrix_float = np.linalg.solve(approx_lattice, supercell_lattice)
    matrix_center = np.rint(matrix_float).astype(int)

    best = None
    width = 2 * int(search_radius) + 1
    for delta in np.ndindex(*([width] * (dim * dim))):
        delta_matrix = np.asarray(delta, dtype=int).reshape(dim, dim) - search_radius
        matrix = matrix_center + delta_matrix
        det = np.linalg.det(matrix)
        if abs(det) < 0.5:
            continue
        snapped = primitive_vectors_from_supercell_matrix(supercell_vectors, matrix)
        try:
            integer_supercell_matrix(snapped, supercell_vectors)
        except ValueError:
            continue
        correction_norm = float(np.linalg.norm(snapped - approx_vectors))
        det_penalty = abs(int(round(abs(det)))) * 1.0e-8
        item = (correction_norm + det_penalty, correction_norm, matrix, snapped)
        if best is None or item[0] < best[0]:
            best = item

    if best is None:
        raise ValueError(
            "Could not snap primitive vectors to an exact integer tiling of the supercell."
        )

    _, correction_norm, matrix, snapped = best
    return snapped, matrix, matrix_float, correction_norm


def same_replica_class(n1: np.ndarray, n2: np.ndarray, M: np.ndarray, tol: float = 1e-8) -> bool:
    z = np.linalg.solve(M, n1 - n2)
    return np.allclose(z, np.rint(z), atol=tol)


def build_modulo_lattice_ao_mapping(
    *,
    symbols: Sequence[str],
    coords_cart: np.ndarray,
    primitive_vectors: np.ndarray,
    supercell_vectors: np.ndarray,
    aos_per_symbol: dict[str, int],
    tol: float = 1e-5,
) -> AOMapping:
    dim = primitive_vectors.shape[0]
    frac = fractional_coordinates(coords_cart, primitive_vectors)
    basis_frac, atom_to_basis = cluster_basis_fractional_coords(
        symbols, coords_cart, primitive_vectors, tol=tol
    )

    order = np.lexsort(tuple(basis_frac[:, i] for i in reversed(range(dim))))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    basis_frac = basis_frac[order]
    atom_to_basis = inverse[atom_to_basis]

    M = integer_supercell_matrix(primitive_vectors, supercell_vectors)
    det = int(round(abs(np.linalg.det(M))))

    integer_translations = np.rint(frac - basis_frac[atom_to_basis]).astype(int)

    rep_representatives: list[np.ndarray] = []
    atom_to_replica = np.full(len(symbols), -1, dtype=int)

    for i, n in enumerate(integer_translations):
        found = None
        for irep, rep in enumerate(rep_representatives):
            if same_replica_class(n, rep, M):
                found = irep
                break

        if found is None:
            found = len(rep_representatives)
            rep_representatives.append(n)

        atom_to_replica[i] = found

    if len(rep_representatives) != det:
        print("WARNING: number of replica classes found differs from det(M).")
        print("found:", len(rep_representatives), "det(M):", det)

    A = lattice_matrix(primitive_vectors)
    replica_vectors_dim = np.asarray([A @ rep for rep in rep_representatives])
    replica_vectors_cart = np.zeros((len(rep_representatives), 3))
    replica_vectors_cart[:, :dim] = replica_vectors_dim

    basis_ao_counts = np.array(
        [aos_per_symbol[symbols[np.where(atom_to_basis == ib)[0][0]]] for ib in range(len(basis_frac))]
    )
    basis_offsets = np.zeros(len(basis_frac), dtype=int)
    basis_offsets[1:] = np.cumsum(basis_ao_counts[:-1])
    nao_prim = int(np.sum(basis_ao_counts))

    ao_to_replica: list[int] = []
    ao_to_local: list[int] = []

    for iatom, sym in enumerate(symbols):
        nbasis = aos_per_symbol[sym]
        ibasis = atom_to_basis[iatom]
        irep = atom_to_replica[iatom]
        offset = basis_offsets[ibasis]

        for iao_atom in range(nbasis):
            ao_to_replica.append(irep)
            ao_to_local.append(offset + iao_atom)

    return AOMapping(
        ao_to_replica=np.asarray(ao_to_replica, dtype=int),
        ao_to_local=np.asarray(ao_to_local, dtype=int),
        replica_vectors_cart=replica_vectors_cart,
        nao_prim=nao_prim,
        basis_frac_coords=basis_frac,
        atom_to_basis=atom_to_basis,
        atom_to_replica=atom_to_replica,
        supercell_integer_matrix=M,
    )


def guess_dimensionality_from_cell_and_coords(cell_vectors: np.ndarray, coords_cart: np.ndarray) -> int:
    """Very simple dimensionality guess: if z spread is tiny and c is large, use 2D."""
    z_span = float(np.ptp(coords_cart[:, 2])) if coords_cart.shape[1] >= 3 else 0.0
    if cell_vectors.shape[0] >= 3 and z_span < 1e-3:
        return 2
    return min(3, cell_vectors.shape[0])


def wrap_fractional_difference(delta: np.ndarray) -> np.ndarray:
    return delta - np.round(delta)


def translation_quality(
    vector_cart: np.ndarray,
    symbols: Sequence[str],
    coords_cart: np.ndarray,
    supercell_vectors: np.ndarray,
    tol: float = 1e-3,
) -> float:
    """Fraction of atoms mapped onto same-symbol atoms by a trial translation."""
    dim = supercell_vectors.shape[0]
    S = lattice_matrix(supercell_vectors)
    frac = np.linalg.solve(S, coords_cart[:, :dim].T).T
    trial_shift = np.linalg.solve(S, vector_cart[:dim])

    count = 0
    for sym, f in zip(symbols, frac):
        target = modulo_one(f + trial_shift)
        ok = False
        for sym2, f2 in zip(symbols, frac):
            if sym2 != sym:
                continue
            df = wrap_fractional_difference(target - f2)
            dcart = S @ df
            if np.linalg.norm(dcart) < tol:
                ok = True
                break
        count += int(ok)
    return count / len(symbols)


def integer_cell_matrix_if_valid(
    primitive_vectors: np.ndarray, supercell_vectors: np.ndarray, tol: float = 1e-4
):
    try:
        M = integer_supercell_matrix(primitive_vectors, supercell_vectors)
    except Exception:
        return None
    det = int(round(abs(np.linalg.det(M))))
    if det <= 0:
        return None
    return M


def guess_primitive_vectors_from_geometry(
    symbols: Sequence[str],
    coords_cart: np.ndarray,
    supercell_vectors: np.ndarray,
    *,
    dim: int = 2,
    tol: float = 2e-3,
    min_quality: float = 0.80,
    max_candidates: int = 80,
) -> np.ndarray:
    """Guess primitive vectors from atomic translations for 1D/2D systems."""
    dim = int(dim)
    if dim not in (1, 2):
        raise NotImplementedError("Automatic primitive-vector guess is implemented for 1D/2D only.")

    S = lattice_matrix(supercell_vectors)
    frac = np.linalg.solve(S, coords_cart[:, :dim].T).T

    raw: list[np.ndarray] = []
    shifts = list(np.ndindex(*([3] * dim)))
    shifts = [np.asarray(s, dtype=float) - 1.0 for s in shifts]

    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if i == j or symbols[i] != symbols[j]:
                continue
            for sh in shifts:
                df = frac[j] + sh - frac[i]
                v_dim = S @ df
                v = np.zeros(3)
                v[:dim] = v_dim
                if np.linalg.norm(v[:dim]) >= 1e-5:
                    raw.append(v)

    candidates: list[np.ndarray] = []
    for v in sorted(raw, key=lambda x: np.linalg.norm(x[:dim])):
        duplicate = any(
            np.linalg.norm(v[:dim] - w[:dim]) < tol or np.linalg.norm(v[:dim] + w[:dim]) < tol
            for w in candidates
        )
        if duplicate:
            continue
        q = translation_quality(v, symbols, coords_cart, supercell_vectors, tol=5 * tol)
        if q >= min_quality:
            candidates.append(v)
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        raise ValueError("Could not find any plausible primitive translation candidate.")

    if dim == 1:
        for v in candidates:
            prim = v.reshape(1, 3)
            if integer_cell_matrix_if_valid(prim, supercell_vectors) is not None:
                return prim
        return candidates[0].reshape(1, 3)

    best = None
    for ia, a in enumerate(candidates):
        for b in candidates[ia + 1 :]:
            area = abs(np.cross(a[:2], b[:2]))
            if area < 1e-6:
                continue
            prim = np.vstack([a, b])
            M = integer_cell_matrix_if_valid(prim, supercell_vectors)
            if M is None:
                continue
            det = int(round(abs(np.linalg.det(M))))
            try:
                basis_frac, _ = cluster_basis_fractional_coords(symbols, coords_cart, prim, tol=5 * tol)
            except Exception:
                continue
            if det * len(basis_frac) != len(symbols):
                continue
            if best is None or area < best[0]:
                best = (area, prim, M, len(basis_frac))

    if best is not None:
        return best[1]

    for ia, a in enumerate(candidates):
        for b in candidates[ia + 1 :]:
            if abs(np.cross(a[:2], b[:2])) > 1e-6:
                return np.vstack([a, b])

    raise ValueError("Could not find two non-collinear primitive vectors.")


def matrix_to_text(mat: np.ndarray) -> str:
    mat = np.asarray(mat, dtype=float)
    return "\n".join(" ".join(f"{x:.10f}" for x in row) for row in mat)


def parse_matrix_text(text: str, *, expected_dim: int | None = None) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", " ").split()])
    mat = np.asarray(rows, dtype=float)
    if mat.ndim != 2 or mat.shape[1] not in (2, 3):
        raise ValueError("Matrix text must have one vector per line and 2 or 3 columns.")
    if mat.shape[1] == 2:
        mat3 = np.zeros((mat.shape[0], 3))
        mat3[:, :2] = mat
        mat = mat3
    if expected_dim is not None and mat.shape[0] != expected_dim:
        raise ValueError(f"Expected {expected_dim} vectors, got {mat.shape[0]}.")
    return mat


def infer_aos_per_symbol_from_wfn(symbols: Sequence[str], nao: int) -> dict[str, int]:
    """Infer AO count per atom for mono-element systems."""
    unique = sorted(set(symbols))
    if len(unique) != 1:
        raise NotImplementedError(
            "Automatic AO-count inference is currently only implemented for mono-element systems. "
            "For multi-kind systems, parse CP2K ATOMIC KIND INFORMATION or provide aos_per_symbol."
        )
    natom = len(symbols)
    if nao % natom != 0:
        raise ValueError(f"Cannot infer AO count: nao={nao} is not divisible by natom={natom}.")
    return {unique[0]: nao // natom}


def infer_aos_per_symbol_from_overlap_metadata(
    symbols: Sequence[str],
    atom_index: np.ndarray,
) -> dict[str, int]:
    """Infer AO counts per chemical symbol from CP2K AO-matrix row metadata.

    CP2K prints one AO-matrix row per AO and includes the 1-based atom index on
    each row. Counting rows per atom is more robust than dividing the total AO
    count by the number of atoms and supports multi-element primitive cells such
    as BN. All atoms with the same symbol are required to have the same AO count;
    if this is not true the structure uses multiple basis/kind definitions for
    one symbol and the current mapping needs explicit per-atom/kind AO counts.
    """
    atom_index = np.asarray(atom_index, dtype=int)
    natom = len(symbols)
    if atom_index.size == 0:
        raise ValueError("Cannot infer AO counts: overlap metadata has no atom indices.")

    counts_by_atom = np.bincount(atom_index, minlength=natom + 1)[1 : natom + 1]
    missing = np.where(counts_by_atom == 0)[0] + 1
    if len(missing):
        raise ValueError(
            "Cannot infer AO counts: overlap metadata is missing rows for atom indices "
            + ", ".join(str(int(i)) for i in missing)
        )

    counts_by_symbol: dict[str, set[int]] = {}
    for symbol, count in zip(symbols, counts_by_atom):
        counts_by_symbol.setdefault(symbol, set()).add(int(count))

    ambiguous = {
        symbol: sorted(counts)
        for symbol, counts in counts_by_symbol.items()
        if len(counts) > 1
    }
    if ambiguous:
        raise ValueError(
            "Cannot infer one AO count per symbol because some symbols have multiple AO counts: "
            + "; ".join(f"{symbol}: {counts}" for symbol, counts in ambiguous.items())
        )

    return {symbol: counts.pop() for symbol, counts in counts_by_symbol.items()}
