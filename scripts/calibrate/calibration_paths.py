"""
Default locations under ``results/calibration`` and ``results/plots/calibration``,
plus calibration **input** CSVs under ``config/calibration/`` (catalog, seeds, limits, loss settings)
and **summary** artifacts under ``summary_statistics/`` (parameter tables, per-``set_id`` eval rollups).

Import these instead of duplicating path strings across scripts.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Catalog, seeds, limits, and loss weights (not under ``data/raw``).
CALIBRATION_CONFIG_DIR = _PROJECT_ROOT / "config" / "calibration"
BRB_SPECIMENS_CSV = CALIBRATION_CONFIG_DIR / "BRB-Specimens.csv"
STEEL_SEED_SETS_CSV = CALIBRATION_CONFIG_DIR / "steel_seed_sets.csv"
PARAM_LIMITS_CSV = CALIBRATION_CONFIG_DIR / "params_limits.csv"
CALIBRATION_LOSS_SETTINGS_CSV = CALIBRATION_CONFIG_DIR / "calibration_loss_settings.csv"

# Human-readable / tabular summaries (not raw inputs or per-run metrics blobs).
SUMMARY_STATISTICS_DIR = _PROJECT_ROOT / "summary_statistics"
CALIBRATION_PARAMETER_SUMMARY_MD = SUMMARY_STATISTICS_DIR / "calibration_parameter_summary.md"
CALIBRATION_PARAMETER_SUMMARY_GENERALIZED_CSV = (
    SUMMARY_STATISTICS_DIR / "calibration_parameter_summary_generalized.csv"
)
CALIBRATION_PARAMETER_SUMMARY_INDIVIDUAL_CSV = (
    SUMMARY_STATISTICS_DIR / "calibration_parameter_summary_individual.csv"
)
GENERALIZED_SET_ID_EVAL_SUMMARY_CSV = SUMMARY_STATISTICS_DIR / "generalized_set_id_eval_summary.csv"
GENERALIZED_UNORDERED_J_SUMMARY_TRAIN_CSV = (
    SUMMARY_STATISTICS_DIR / "generalized_unordered_J_binenv_summary_train.csv"
)
GENERALIZED_UNORDERED_J_SUMMARY_VALIDATION_CSV = (
    SUMMARY_STATISTICS_DIR / "generalized_unordered_J_binenv_summary_validation.csv"
)

RESULTS_CALIBRATION = _PROJECT_ROOT / "results" / "calibration"
INDIVIDUAL_OPTIMIZE_DIR = RESULTS_CALIBRATION / "individual_optimize"
AVERAGED_OPTIMIZE_DIR = RESULTS_CALIBRATION / "averaged_optimize"
GENERALIZED_OPTIMIZE_DIR = RESULTS_CALIBRATION / "generalized_optimize"

INITIAL_BRB_PARAMETERS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "initial_brb_parameters.csv"
OPTIMIZED_BRB_PARAMETERS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "optimized_brb_parameters.csv"
OPTIMIZED_BRB_PARAMETERS_METRICS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "optimized_brb_parameters_metrics.csv"

SPECIMEN_APPARENT_BN_BP_PATH = RESULTS_CALIBRATION / "specimen_apparent_bn_bp.csv"

AVERAGED_BRB_PARAMETERS_PATH = AVERAGED_OPTIMIZE_DIR / "averaged_brb_parameters.csv"
AVERAGED_PARAMS_EVAL_METRICS_PATH = AVERAGED_OPTIMIZE_DIR / "averaged_params_eval_metrics.csv"

GENERALIZED_BRB_PARAMETERS_PATH = GENERALIZED_OPTIMIZE_DIR / "generalized_brb_parameters.csv"
GENERALIZED_PARAMS_EVAL_METRICS_PATH = GENERALIZED_OPTIMIZE_DIR / "generalized_params_eval_metrics.csv"

# Per-specimen hysteresis artifacts: ``{Name}_set{k}_simulated.csv`` (``Deformation[in],Force[kip],Force_sim[kip]``).
# Stem matches the metrics/parameters CSV stem used when the eval/optimize run wrote histories.
INDIVIDUAL_SIMULATED_FORCE_DIR = INDIVIDUAL_OPTIMIZE_DIR / "optimized_brb_parameters_simulated_force"
# Preset / initial-params overlay pipeline (``plot_preset_overlays.py``): parameters snapshot + numerical CSVs.
INITIAL_PARAMS_OVERLAY_PARAMETERS_PATH = INDIVIDUAL_OPTIMIZE_DIR / "initial_params_overlay_parameters.csv"
INITIAL_PARAMS_SIMULATED_FORCE_DIR = INDIVIDUAL_OPTIMIZE_DIR / "initial_params_simulated_force"
GENERALIZED_SIMULATED_FORCE_DIR = GENERALIZED_OPTIMIZE_DIR / "generalized_params_eval_metrics_simulated_force"
# ``plot_generalized_train_mean_bn_bp_overlays.py``: same generalized steel per set_id, b_p/b_n = train-only weighted mean.
GENERALIZED_TRAIN_MEAN_BN_BP_SIMULATED_FORCE_DIR = (
    GENERALIZED_OPTIMIZE_DIR / "generalized_train_mean_bn_bp_simulated_force"
)
AVERAGED_SIMULATED_FORCE_DIR = AVERAGED_OPTIMIZE_DIR / "averaged_params_eval_metrics_simulated_force"

PLOTS_CALIBRATION = _PROJECT_ROOT / "results" / "plots" / "calibration"
PLOTS_INDIVIDUAL_OPTIMIZE = PLOTS_CALIBRATION / "individual_optimize"
PLOTS_INITIAL_PARAMS_OVERLAYS = PLOTS_INDIVIDUAL_OPTIMIZE / "overlays_initial_params"
PLOTS_AVERAGED_OPTIMIZE = PLOTS_CALIBRATION / "averaged_optimize"
PLOTS_GENERALIZED_OPTIMIZE = PLOTS_CALIBRATION / "generalized_optimize"
PLOTS_GENERALIZED_TRAIN_MEAN_BN_BP_OVERLAYS = PLOTS_GENERALIZED_OPTIMIZE / "overlays_train_mean_bn_bp"
