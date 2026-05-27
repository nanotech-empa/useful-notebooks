"""Reusable helpers for CP2K localized-basis band unfolding notebooks."""

from .geometry import (
    AOMapping,
    build_modulo_lattice_ao_mapping,
    cluster_basis_fractional_coords,
    fractional_coordinates,
    guess_dimensionality_from_cell_and_coords,
    guess_primitive_vectors_from_geometry,
    infer_aos_per_symbol_from_wfn,
    integer_supercell_matrix,
    lattice_matrix,
    matrix_to_text,
    parse_matrix_text,
)
from .io import (
    SupercellWavefunctions,
    hartree_to_ev,
    parse_cp2k_cell_vectors,
    parse_cp2k_overlap_matrix_log,
    read_sparse_overlap_npz,
    print_eigenvalue_summary,
    read_cp2k_wfn,
    read_xyz_coordinates,
)
from .kpath import (
    folded_kpoints_from_supercell_matrix,
    guess_2d_lattice_type,
    kfrac_to_cart,
    kpath_axis_from_fractional_path,
    project_kpoints_to_kpath,
    reciprocal_vectors,
    standard_kpath,
)
from .plotting import plot_unfolded_kpath
from .unfolding import (
    SparseUnfoldingCache,
    fourier_project_coefficients,
    mo_norms_sparse,
    prepare_sparse_unfolding_cache,
    primitive_overlap_metric,
    sparse_bloch_overlap_metric_from_cache,
    sparse_bloch_rhs_from_cache,
    spectral_weight_full,
    spectral_weight_simple,
    unfold_band_weights,
    unfold_band_weights_full,
    unfold_band_weights_sparse_full,
)
from .widgets import PrimitiveCellWidgets, create_primitive_cell_widgets, read_primitive_cell_widgets

