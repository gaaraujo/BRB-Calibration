#!/usr/bin/env bash
# Zip everything clean_outputs.sh would wipe, plus full results/plots and results/calibration.
# Run from repo root: ./archive_pipeline_outputs.sh [--label myrun] [--out-dir run_snapshots]
# Output: run_snapshots/<yyyyMMdd_HHmmss>_<label>.zip - unzip at repo root to restore paths.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LABEL="full"
OUT_DIR="$ROOT/run_snapshots"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--label NAME] [--out-dir DIR]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

SANITIZED="$(echo "$LABEL" | sed 's/[^A-Za-z0-9._-]/_/g; s/^_//; s/_$//')"
[[ -z "$SANITIZED" ]] && SANITIZED="full"

STAMP="$(date -u +"%Y%m%d_%H%M%S")"
ZIP_NAME="${STAMP}_${SANITIZED}.zip"
ZIP_PATH="$OUT_DIR/$ZIP_NAME"
STAGING="$OUT_DIR/.staging_${STAMP}_$$"
mkdir -p "$OUT_DIR" "$STAGING"

cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

DATA_DIRS=(data/filtered data/resampled data/cycle_points_original data/cycle_points_resampled)
ROOT_FILES=(
  summary_statistics/calibration_parameter_summary.md
  summary_statistics/calibration_parameter_summary_generalized.csv
  summary_statistics/calibration_parameter_summary_individual.csv
  summary_statistics/calibration_parameter_summary_generalized_by_set.csv
  summary_statistics/calibration_parameter_summary_individual_by_set.csv
  summary_statistics/generalized_set_id_eval_summary.csv
  summary_statistics/generalized_unordered_J_binenv_summary_train.csv
  summary_statistics/generalized_unordered_J_binenv_summary_validation.csv
)
INCLUDED=()

copy_if_exists() {
  local rel="$1"
  local src="$ROOT/$rel"
  if [[ -e "$src" ]]; then
    local parent dest_parent leaf
    parent="$(dirname "$rel")"
    leaf="$(basename "$rel")"
    dest_parent="$STAGING/$parent"
    mkdir -p "$dest_parent"
    cp -a "$src" "$dest_parent/$leaf"
    INCLUDED+=("$rel")
  fi
}

for rel in "${DATA_DIRS[@]}"; do
  copy_if_exists "$rel"
done
for rel in results/plots results/calibration; do
  copy_if_exists "$rel"
done
for f in "${ROOT_FILES[@]}"; do
  if [[ -f "$ROOT/$f" ]]; then
    mkdir -p "$(dirname "$STAGING/$f")"
    cp -a "$ROOT/$f" "$STAGING/$f"
    INCLUDED+=("$f")
  fi
done

if [[ ${#INCLUDED[@]} -eq 0 ]]; then
  echo "Nothing to archive (no matching paths under repo root)." >&2
  exit 1
fi

GIT_HEAD=""
if command -v git >/dev/null 2>&1; then
  GIT_HEAD="$(cd "$ROOT" && git rev-parse HEAD 2>/dev/null || true)"
fi

MANIFEST="$STAGING/MANIFEST.txt"
{
  echo "created_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "label=$SANITIZED"
  echo "git_head=$GIT_HEAD"
  echo "repo_relative_paths_included:"
  for p in "${INCLUDED[@]}"; do echo "  $p"; done
  echo "  MANIFEST.txt"
} > "$MANIFEST"

if ! command -v zip >/dev/null 2>&1; then
  echo "zip(1) is required (e.g. apt install zip, or use Git Bash on Windows)." >&2
  exit 1
fi

rm -f "$ZIP_PATH"
(cd "$STAGING" && zip -rq "$ZIP_PATH" .)
trap - EXIT
rm -rf "$STAGING"

echo "Wrote $ZIP_PATH"
