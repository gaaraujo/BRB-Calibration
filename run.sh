#!/usr/bin/env bash
set -euo pipefail

# J_feat: amplitude-based cycle weights w_c (--amplitude-weights on calibration Python steps).
# false or 0 = uniform w_c=1 (default). true or 1 = amplitude weighting for the whole pipeline run.
USE_AMPLITUDE_WEIGHTS=false
AMP_W_ARGS=()
if [[ "${USE_AMPLITUDE_WEIGHTS}" == "true" || "${USE_AMPLITUDE_WEIGHTS}" == "1" ]]; then
  AMP_W_ARGS=(--amplitude-weights)
fi

# Run the full pipeline. By default all output goes to the terminal only.
#
# To mirror the same output to a log file *and* keep it on screen, set PIPELINE_LOG:
#   PIPELINE_LOG=pipeline_log.txt ./run.sh
#
# Do not use `./run.sh > file` if you want live progress -- that hides the terminal.

_pipeline_print_footer() {
  local _start_s=$1 _pipe_end_s _elapsed_s _h _m _s
  _pipe_end_s=$(date +%s)
  _elapsed_s=$((_pipe_end_s - _start_s))
  _h=$((_elapsed_s / 3600))
  _m=$(((_elapsed_s % 3600) / 60))
  _s=$((_elapsed_s % 60))
  echo ""
  echo "========================================================================"
  echo "  BRB-Calibration pipeline finished"
  echo "  $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  if ((_h >= 1)); then
    echo "  Elapsed:  ${_h}h ${_m}m ${_s}s"
  elif ((_m >= 1)); then
    echo "  Elapsed:  ${_m}m ${_s}s"
  else
    echo "  Elapsed:  ${_s}s"
  fi
  echo "========================================================================"
  echo ""
}

run_pipeline_steps() {
  # Echo ``config/calibration/*.csv`` heads (same paths as ``calibration_paths.py``).
  python scripts/calibrate/print_calibration_config_heads.py

  # Full reset (optional): uncomment to wipe regenerated data/results before rerun.
  # Preserves data/raw/ and config/calibration/*.csv (removes regenerated data/* pipeline dirs and all results/calibration/, including individual_optimize/initial_brb_parameters.csv)
  # ./clean_outputs.sh

  # --- Postprocess: cycle_points_original -> filtered -> resampled + cycle_points_resampled ---
  python scripts/postprocess/cycle_points.py --overwrite
  python scripts/postprocess/filter_force.py
  python scripts/postprocess/resample_filtered.py
  # Batch specimen plots (force-def, time histories)
  python scripts/postprocess/plot_specimens.py

  # --- Apparent b_n / b_p and geometry (exploratory) ---
  # Apparent b_n, b_p from resampled experiments (incl. segment stats for steel_seed_sets.csv keywords)
  python scripts/calibrate/extract_bn_bp.py
  # Requires repo-root steel_seed_sets.csv (one row per set_id: steel overrides + b_p/b_n as number or stat keyword)
  python scripts/calibrate/build_initial_brb_parameters.py
  # Hysteresis with fitted hardening slopes
  python scripts/calibrate/plot_b_slopes.py
  # Histograms and geometry scatter for b_n / b_p
  python scripts/calibrate/plot_b_histograms_and_scatter.py

  # --- Preset sim vs exp overlays (fixed b_p, b_n before L-BFGS; steel from steel_seed_sets.csv) ---
  python scripts/calibrate/plot_preset_overlays.py --params results/calibration/individual_optimize/initial_brb_parameters.csv
  # Alternative: --seeds steel_seed_sets.csv if you want plot_preset_overlays to rebuild from catalog + seeds instead of this CSV :)

  # --- SteelMPF calibration and sim vs exp overlays ---
  # SteelMPF calibration: J_feat (+ optional J_E); cycle-weight PNGs under plots/calibration/individual_optimize/cycle_weights/; *_metrics.csv and *_simulated_force/ next to --output.
  python scripts/calibrate/optimize_brb_mse.py --initial-params results/calibration/individual_optimize/initial_brb_parameters.csv --output results/calibration/individual_optimize/optimized_brb_parameters.csv
  # Sim vs resampled exp overlays (physical + normalized)
  python scripts/calibrate/plot_params_vs_filtered.py --params results/calibration/individual_optimize/optimized_brb_parameters.csv --output-dir overlays

  # --- Averaged-parameter evaluation (metrics CSV + NPZs + hysteresis PNGs) ---
  # Averaged/generalized weights: BRB-Specimens.csv averaged_weight / generalized_weight (path-ordered only); individual_optimize for per-specimen L-BFGS (see README).
  python scripts/calibrate/eval_averaged_params.py "${AMP_W_ARGS[@]}" \
      --params results/calibration/individual_optimize/optimized_brb_parameters.csv \
      --output-params results/calibration/averaged_optimize/averaged_brb_parameters.csv \
      --output-metrics results/calibration/averaged_optimize/averaged_params_eval_metrics.csv \
      --output-plots-dir results/plots/calibration/averaged_optimize/overlays

  # --- Generalized optimization (shared PARAMS_TO_OPTIMIZE) + specimen-set eval --- (same config/calibration/params_limits.csv unless --param-limits)
  python scripts/calibrate/optimize_generalized_brb_mse.py "${AMP_W_ARGS[@]}" \
      --params results/calibration/individual_optimize/optimized_brb_parameters.csv \
      --output-params results/calibration/generalized_optimize/generalized_brb_parameters.csv \
      --output-metrics results/calibration/generalized_optimize/generalized_params_eval_metrics.csv \
      --output-plots-dir results/plots/calibration/generalized_optimize/overlays

  # --- Combined normalized overlays: one set{k}_combined_force_def_norm.png per set_id per method ---
  # Reads numerical-model *_simulated.csv under each method's *simulated_force/ (from optimize_brb_mse / eval above).
  python scripts/calibrate/plot_compare_calibration_overlays.py

  # --- Generalized overlays with train-weighted mean b_p/b_n (same steel as generalized per set_id) ---
  python scripts/calibrate/plot_generalized_train_mean_bn_bp_overlays.py

  # --- Optimized-parameter summary (summary_statistics/) ---
  # Writes calibration_parameter_summary.md and rollup + per-set_id CSVs (see report_calibration_param_tables.py --help).
  python scripts/calibrate/report_calibration_param_tables.py --write summary_statistics/calibration_parameter_summary.md

  # --- Debug figures (optional inspection) ---
  # Per-cycle panels, shaded int F du and E/S_E
  python scripts/calibrate/plot_cycle_energy_debug.py "${AMP_W_ARGS[@]}" --params results/calibration/individual_optimize/optimized_brb_parameters.csv
  # J_feat landmark markers on exp vs sim hysteresis
  python scripts/calibrate/plot_cycle_landmarks_debug.py "${AMP_W_ARGS[@]}" --params results/calibration/individual_optimize/optimized_brb_parameters.csv

  # --- Averaged vs generalized narrative report (default: results/calibration/averaged_vs_generalized_metrics_report.md) ---
  python scripts/calibrate/report_averaged_vs_generalized_metrics.py
}

run_pipeline() {
  local _pipe_start_s _rc=0
  _pipe_start_s=$(date +%s)
  echo ""
  echo "========================================================================"
  echo "  BRB-Calibration pipeline"
  echo "  $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  echo "========================================================================"
  echo ""

  run_pipeline_steps || _rc=$?
  _pipeline_print_footer "$_pipe_start_s"
  return "$_rc"
}

if [[ -n "${PIPELINE_LOG:-}" ]]; then
  # stderr + stdout -> terminal and file (overwrite each run; use tee -a only if you want to stack runs)
  run_pipeline 2>&1 | tee "$PIPELINE_LOG"
else
  run_pipeline
fi
