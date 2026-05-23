"""Shared feature and target configuration for the polymer ML project."""

from __future__ import annotations

TARGET_REGRESSION = "cross_presentation_pct"

LEAKAGE_COLUMNS = [
    "lymph_node_mass_mg",
    "cross_presentation_pct",
    "cd86_pct",
    "mhc_ii_pct",
    "ifng_cd8_pct",
    "tnfa_cd8_pct",
    "ifng_cd4_pct",
]

ID_COLUMNS = ["group_id", "smiles", "R_group"]

DESCRIPTOR_FEATURES = [
    "polymerization_degree_X",
    "Mn",
    "pdi",
    "particle_size_nm",
    "pka",
    "X_dp",
    "R_B",
    "R_C1",
    "R_C10",
    "R_C12",
    "R_C18",
    "R_C3",
    "R_C6",
    "R_C8",
    "carbon_length",
    "LogP",
    "endosomal_alignment",
    "hydrophobic_activation",
    "hydrophobic_density",
    "dp_per_carbon",
    "chain_dp_interaction",
    "endosomal_hydro_balance",
    "hydrophobic_activation_sq",
    "pka_sq",
]

DEFAULT_HIGH_EFFICIENCY_THRESHOLD = 6.219

