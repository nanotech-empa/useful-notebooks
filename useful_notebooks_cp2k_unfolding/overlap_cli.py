from __future__ import annotations

import argparse

from .io import write_sparse_overlap_npz


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert CP2K AO overlap matrix logs to sparse NPZ files."
    )
    parser.add_argument("input", help="CP2K overlap matrix log file")
    parser.add_argument("output", help="Output compressed NPZ file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keep entries with absolute value larger than this threshold.",
    )
    parser.add_argument("--nao", type=int, default=None, help="Expected AO count")
    args = parser.parse_args(argv)

    write_sparse_overlap_npz(
        args.input,
        args.output,
        threshold=args.threshold,
        nao=args.nao,
    )


if __name__ == "__main__":
    main()
