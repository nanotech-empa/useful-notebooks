#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _case_dir(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path.resolve()
    return (DEFAULT_ROOT / "cases" / name_or_path).resolve()


def _parent_folder(case_dir: Path) -> Path:
    for name in ("parent_calc_folder", "paret_calc_folder"):
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No parent_calc_folder found in {case_dir}")


def _primitive_vectors_from_submit(case_dir: Path) -> str | None:
    submit = case_dir / "_aiidasubmit.sh"
    if not submit.exists():
        return None
    tokens = shlex.split(submit.read_text())
    for i, token in enumerate(tokens[:-1]):
        if token == "--primitive-vectors":
            return tokens[i + 1]
    return None


def _path_from_submit(case_dir: Path) -> str:
    submit = case_dir / "_aiidasubmit.sh"
    if not submit.exists():
        return "G-K-M-G"
    tokens = shlex.split(submit.read_text())
    for i, token in enumerate(tokens[:-1]):
        if token == "--path":
            return tokens[i + 1]
    return "G-K-M-G"


def build_command(args: argparse.Namespace) -> list[str]:
    case_dir = _case_dir(args.case)
    parent = _parent_folder(case_dir)
    primitive_vectors = args.primitive_vectors or _primitive_vectors_from_submit(case_dir)
    if not primitive_vectors:
        raise ValueError("Primitive vectors were not provided and could not be read from _aiidasubmit.sh")

    output = case_dir / args.output
    pdos_output = case_dir / args.pdos_output if args.pdos else None

    runner = [
        sys.executable,
        "-m",
        "cp2k_spm_tools.cli.unfold_wfn_sparse",
    ]
    if args.mpi_ranks and args.mpi_ranks > 1:
        runner = [
            "mpiexec",
            "-n",
            str(args.mpi_ranks),
            sys.executable,
            "-m",
            "mpi4py",
            "-m",
            "cp2k_spm_tools.cli.unfold_wfn_sparse_mpi",
        ]

    command = [
        *runner,
        str(parent / "aiida-RESTART.wfn"),
        str(parent / "aiida-overlap_matrix.out-1_0.Log"),
        str(output),
        "--xyz",
        str(parent / "aiida.coords.xyz"),
        "--cp2k-input",
        str(parent / "aiida.inp"),
        "--primitive-vectors",
        primitive_vectors,
        "--path",
        args.path or _path_from_submit(case_dir),
        "--lattice-type",
        args.lattice_type,
        "--overlap-format",
        "log",
        "--overlap-threshold",
        str(args.overlap_threshold),
        "--basis-cluster-tol",
        str(args.basis_cluster_tol),
    ]
    if args.primitive_basis_atoms:
        command += ["--primitive-basis-atoms", args.primitive_basis_atoms]
    if args.emin is not None:
        command += ["--emin", str(args.emin)]
    if args.emax is not None:
        command += ["--emax", str(args.emax)]
    if args.pdos:
        command += [
            "--pdos-glob",
            str(parent / "aiida-*list*-1.pdos"),
            "--pdos-output",
            str(pdos_output),
            "--pdos-threshold",
            str(args.pdos_threshold),
        ]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local CP2K unfolding test case.")
    parser.add_argument("case", help="Case name below cases/ or an explicit case directory.")
    parser.add_argument("--output", default="unfolding_bands.local.npz")
    parser.add_argument("--pdos-output", default="unfolding_projections.local.npz")
    parser.add_argument("--pdos", action="store_true", help="Also parse atom-resolved PDOS files.")
    parser.add_argument("--mpi-ranks", type=int, default=1, help="Use the MPI runner with this many ranks.")
    parser.add_argument("--primitive-vectors", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--lattice-type", default="auto")
    parser.add_argument("--basis-cluster-tol", type=float, default=5.0e-2)
    parser.add_argument("--primitive-basis-atoms", default=None, help="1-based atom indices/ranges defining one primitive basis, e.g. '1 2' or '1..2'.")
    parser.add_argument("--overlap-threshold", type=float, default=1.0e-10)
    parser.add_argument("--pdos-threshold", type=float, default=1.0e-4)
    parser.add_argument("--emin", type=float, default=None)
    parser.add_argument("--emax", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_command(args)
    print(" ".join(shlex.quote(x) for x in command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
