# Local unfolding regression tests

This folder contains a lightweight harness to reproduce CP2K band-unfolding
calculations locally from files produced by a completed AiiDA workchain.

The heavy calculation files are intentionally not committed.  For each case,
populate `parent_calc_folder/` with:

- `aiida-RESTART.wfn`
- `aiida-overlap_matrix.out-1_0.Log`
- `aiida.coords.xyz`
- `aiida.inp`
- optional `aiida-*list*-1.pdos` files for atom/orbital projections

The case directory can also contain the original `_aiidasubmit.sh`.  If present,
`scripts/run_case.py` reads the primitive vectors and k-path from it unless they
are overridden explicitly.

## Serial run

```bash
python scripts/run_case.py working_4179 \
  --primitive-basis-atoms "1 2" \
  --basis-cluster-tol 5e-2 \
  --pdos
```

## MPI run

The MPI implementation splits selected molecular orbitals over ranks.  In this
first version each rank still reads the full WFN file before slicing its local
MO block.

```bash
python scripts/run_case.py problematic_4808 \
  --primitive-basis-atoms "1 2" \
  --basis-cluster-tol 5e-2 \
  --mpi-ranks 8 \
  --pdos
```

## Inspect output

```bash
python scripts/inspect_unfolding_npz.py cases/problematic_4808/unfolding_bands.local.npz
```

Useful diagnostics:

- `primitive_basis_atom_indices` records the 1-based atom indices used as the
  reference primitive basis.
- `atom_mapping_displacements_spin_*` stores the displacement of each atom from
  its assigned ideal translated primitive-basis site.
- `nao_prim` printed during the run must be the primitive AO count, not the full
  supercell AO count.
- `weights std` should not collapse to zero for a meaningful unfolding.
