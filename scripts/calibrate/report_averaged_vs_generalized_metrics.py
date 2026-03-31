"""

Calibration report: narrative (specimens, seed sets, objective, three analyses) plus tables of best
**set_id** per specimen, specimen-set–wide best sets for averaged/generalized stages, and parameter extracts.

- **Individual optimize**: for each specimen, which ``set_id`` minimizes ``final_J_total``

  (among successful rows with finite J), plus a **single wide parameters table** (one row per specimen, ``set_id`` column + BRB fields from ``optimized_brb_parameters.csv``).

- **Averaged / generalized**: same per specimen, plus **best overall** ``set_id`` that minimizes the

  specimen-weighted mean ``final_J_total`` over rows with **contributes_to_aggregate**; the report also

  identifies best overall ``set_id`` by contributor-weighted **final_unordered_J_binenv** (when present),

  with parameter extracts.

- Writes **shared-backbone** columns of the parameters CSV for the best overall set (values

  identical for every specimen at that ``set_id``, typically the averaged backbone

  ``R0,cR1,cR2,a1–a4``) and a **full** parameters table for all specimens at that ``set_id``.



Example::

    python scripts/calibrate/report_averaged_vs_generalized_metrics.py
    python scripts/calibrate/report_averaged_vs_generalized_metrics.py --per-specimen-delta
"""

from __future__ import annotations



import argparse

import sys

from datetime import datetime, timezone

from pathlib import Path



import numpy as np

import pandas as pd



_SCRIPT_DIR = Path(__file__).resolve().parent

_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

_SCRIPTS = _PROJECT_ROOT / "scripts"

sys.path.insert(0, str(_SCRIPTS))



from calibrate.calibration_paths import (  # noqa: E402
    AVERAGED_BRB_PARAMETERS_PATH,
    AVERAGED_PARAMS_EVAL_METRICS_PATH,
    BRB_SPECIMENS_CSV,
    GENERALIZED_BRB_PARAMETERS_PATH,
    GENERALIZED_PARAMS_EVAL_METRICS_PATH,
    OPTIMIZED_BRB_PARAMETERS_METRICS_PATH,
    OPTIMIZED_BRB_PARAMETERS_PATH,
    RESULTS_CALIBRATION,
    SET_ID_SETTINGS_CSV,
)

from calibrate.calibration_loss_settings import DEFAULT_CALIBRATION_LOSS_SETTINGS  # noqa: E402





def _weighted_mean_finite(series: pd.Series, weights: pd.Series) -> float:
    """Weighted average over finite ``series`` / ``weights`` pairs; NaNs in ``series`` are ignored."""
    w = np.asarray(weights, dtype=float)
    x = np.asarray(series, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    x = x[m]
    w = w[m]
    sw = float(np.sum(w))
    if sw <= 0.0:
        return float("nan")
    return float(np.sum(x * w) / sw)





def _load_metrics(path: Path) -> pd.DataFrame:

    """Read metrics CSV with existence check."""
    if not path.is_file():

        raise FileNotFoundError(f"metrics CSV not found: {path}")

    return pd.read_csv(path)





def _load_params(path: Path) -> pd.DataFrame:

    """Read parameters CSV with existence check."""
    if not path.is_file():

        raise FileNotFoundError(f"parameters CSV not found: {path}")

    return pd.read_csv(path)





def _as_bool_series(s: pd.Series) -> pd.Series:

    """Coerce a Series to bool (accepts true/1/yes)."""
    if s.dtype == object:

        return s.map(lambda v: str(v).lower() in ("true", "1", "yes")).astype(bool)

    return s.astype(bool)





def _eval_mask(df: pd.DataFrame) -> pd.Series:

    """Boolean mask for evaluable individual metrics rows."""
    jt = pd.to_numeric(df["final_J_total"], errors="coerce")

    return _as_bool_series(df["success"]) & jt.notna() & np.isfinite(jt)





def _contributing_mask(df: pd.DataFrame) -> pd.Series:

    """Rows with contributes_to_aggregate, success, and finite final_J_total."""
    return (

        _as_bool_series(df["contributes_to_aggregate"])

        & _as_bool_series(df["success"])

        & pd.to_numeric(df["final_J_total"], errors="coerce").notna()

        & np.isfinite(pd.to_numeric(df["final_J_total"], errors="coerce"))

    )





def _aggregate_by_set(

    df: pd.DataFrame, mask: pd.Series, metrics: list[str]

) -> pd.DataFrame:

    """Weighted mean of metric columns per set_id (contributing rows only)."""
    sub = df.loc[mask, ["set_id", "specimen_weight"] + metrics].copy()

    rows = []

    for sid, g in sub.groupby("set_id", sort=True):

        w = pd.to_numeric(g["specimen_weight"], errors="coerce").fillna(0.0)

        rec: dict[str, object] = {"set_id": sid, "n_rows": int(len(g))}

        for m in metrics:

            rec[m] = _weighted_mean_finite(g[m], w)

        rows.append(rec)

    return pd.DataFrame(rows)


# Detail columns for “best set per specimen” tables and weighted-mean aggregation (metrics CSV schema).
REPORT_METRIC_COLS: tuple[str, ...] = (
    "final_J_feat_raw",
    "final_J_feat_l1_raw",
    "final_J_E_raw",
    "final_J_E_l1_raw",
    "final_unordered_J_binenv",
    "final_unordered_J_binenv_l1",
)
REPORT_METRIC_HEADERS: tuple[str, ...] = (
    "J_feat",
    "J_feat_L1",
    "J_E",
    "J_E_L1",
    "J_binenv",
    "J_binenv_L1",
)


def _fmt_metric_table_cell(r: pd.Series, col: str) -> str:
    if col not in r.index:
        return ""
    v = r.get(col)
    if pd.isna(v):
        return ""
    fv = float(v)
    return _fmt_sci(fv) if np.isfinite(fv) else ""


def _best_set_per_specimen(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """One row per Name: set_id with minimum final_J_total (tie-break: smallest set_id)."""
    cols = ["Name", "set_id", "final_J_total", *REPORT_METRIC_COLS]
    sub = df.loc[mask, cols].copy()
    sub["final_J_total"] = pd.to_numeric(sub["final_J_total"], errors="coerce")
    for c in cols:
        if c in (
            "Name",
            "set_id",
        ):
            continue
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub[np.isfinite(sub["final_J_total"])]
    empty_cols = ["Name", "best_set_id", "final_J_total", *REPORT_METRIC_COLS]
    if sub.empty:
        return pd.DataFrame(columns=empty_cols)
    sub = sub.sort_values(["Name", "final_J_total", "set_id"])
    out = sub.groupby("Name", sort=True, as_index=False).first()
    return out.rename(columns={"set_id": "best_set_id"})





def _best_overall_set_id(agg: pd.DataFrame) -> tuple[int, float] | tuple[None, float]:

    """set_id with minimum weighted mean final_J_total on aggregated frame (tie-break: smallest set_id)."""
    if agg.empty or "final_J_total" not in agg.columns:

        return None, float("nan")

    sub = agg[["set_id", "final_J_total"]].copy()
    sub["final_J_total"] = pd.to_numeric(sub["final_J_total"], errors="coerce")
    sub = sub[np.isfinite(sub["final_J_total"])]
    if sub.empty:
        return None, float("nan")
    sub = sub.sort_values(["final_J_total", "set_id"], ascending=[True, True])
    sid = int(sub.iloc[0]["set_id"])
    j = float(sub.iloc[0]["final_J_total"])
    return sid, j


def _best_overall_set_id_for_metric(
    agg: pd.DataFrame, metric_col: str
) -> tuple[int | None, float]:
    """set_id with minimum weighted mean ``metric_col`` (tie-break: smallest set_id)."""
    if agg.empty or metric_col not in agg.columns:
        return None, float("nan")
    sub = agg[["set_id", metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[np.isfinite(sub[metric_col])]
    if sub.empty:
        return None, float("nan")
    sub = sub.sort_values([metric_col, "set_id"], ascending=[True, True])
    sid = int(sub.iloc[0]["set_id"])
    val = float(sub.iloc[0][metric_col])
    return sid, val


def _md_lines_best_set_by_metric(
    params: pd.DataFrame,
    set_id: int | None,
    metric_label: str,
    value: float,
) -> list[str]:
    """Markdown lines: best set bullet, shared backbone, full parameter table."""
    out: list[str] = []
    if set_id is None or not np.isfinite(value):
        out.append(
            f"*(could not determine — no finite contributing weighted mean **{metric_label}**)*"
        )
        out.append("")
        return out
    out.append(
        f"- **set_id = {set_id}** (minimum weighted mean **{metric_label}** = {_fmt_sci(value)})."
    )
    out.append("")
    const = _shared_backbone_constant_columns(params, set_id)
    if const:
        out.append("Shared backbone (identical across specimens at this `set_id`):")
        out.append("")
        for k, v in const.items():
            out.append(f"- `{k}` = {v}")
        out.append("")
    else:
        out.append("*(no shared backbone constants or missing parameters slice)*")
        out.append("")
    out.append("All specimen rows at this set:")
    out.append("")
    out.append(_params_slice_to_md(params, set_id))
    out.append("")
    return out





def _shared_backbone_constant_columns(params: pd.DataFrame, set_id: int) -> dict[str, str]:

    """Parameter names equal across all rows at a set_id."""
    blk = params.loc[params["set_id"] == set_id]

    if blk.empty:

        return {}

    const: dict[str, str] = {}

    for col in blk.columns:

        if col in ("ID", "Name", "set_id"):

            continue

        u = blk[col].dropna().unique()

        if len(u) == 1:

            v = u[0]

            if isinstance(v, (float, np.floating)):

                const[col] = f"{float(v):.12g}"

            else:

                const[col] = str(v)

    return dict(sorted(const.items()))





def _fmt_sci(x: float, nd: int = 4) -> str:

    """Format float in scientific notation for report tables."""
    if x is None or not np.isfinite(x):

        return ""

    return f"{x:.{nd}e}"





def _md_table(headers: list[str], rows: list[list[str]]) -> str:

    """Markdown pipe table from headers and string rows."""
    w = [len(h) for h in headers]

    for r in rows:

        for i, c in enumerate(r):

            w[i] = max(w[i], len(c))



    def fmt_row(cells: list[str]) -> str:
        """Pad cells to column width for plain-text table."""
        return "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"



    lines = [fmt_row(headers), fmt_row(["-" * n for n in w])]

    lines.extend(fmt_row(r) for r in rows)

    return "\n".join(lines)


def _dataframe_to_md_table(df: pd.DataFrame) -> str:
    """Render full DataFrame as a GitHub-style markdown table."""
    cols = [str(c) for c in df.columns]
    rows: list[list[str]] = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append("" if pd.isna(v) else str(v).strip())
        rows.append(cells)
    return _md_table(cols, rows)


def _report_narrative_intro(specimens_path: Path, steel_seed_path: Path) -> list[str]:
    """Markdown blocks: catalog table, seeds, objective, analyses."""
    parts: list[str] = []
    parts.append("## 1. Specimen metadata")
    parts.append("")
    parts.append(
        f"The table below lists specimens from the project catalog (`{specimens_path.name}`). "
        "Column **path_ordered** marks tests for which a time-ordered force–deformation history "
        "is available for calibration; where it is false, only digitized unordered F–u data are used in the "
        "pipeline. **averaged_weight** and **generalized_weight** control each specimen’s influence on "
        "averaged parameter aggregation and on generalized optimization (often zero for unordered-only tests). "
        "**individual_optimize** flags specimens included in the per-specimen optimization sweep."
    )
    parts.append("")
    if specimens_path.is_file():
        cat = pd.read_csv(specimens_path)
        prefer = [
            "Name",
            "path_ordered",
            "individual_optimize",
            "averaged_weight",
            "generalized_weight",
            "f_yc_ksi",
            "A_c_in2",
            "L_T_in",
            "L_y_in",
            "experimental_layout",
        ]
        use = [c for c in prefer if c in cat.columns]
        if not use:
            use = list(cat.columns)
        parts.append(_dataframe_to_md_table(cat[use]))
    else:
        parts.append(f"*(Catalog not found: `{specimens_path}`)*")
    parts.append("")
    parts.append("## 2. Parameter sets we try")
    parts.append("")
    parts.append(
        "Each **set_id** (below) defines a candidate starting **steel / backbone** configuration "
        "used when building initial BRB parameters: modulus **E**, backbone anchors **R0**, **cR1**, "
        "**cR2**, **a1**–**a4**, and rules for segment slopes **b_p** and **b_n**. All calibration "
        "stages compare results across these "
        f"alternatives. The table below reproduces **`{steel_seed_path.name}`**."
    )
    parts.append(
        "When **b_p** or **b_n** is a word (not a number), it names a **summary statistic** of that "
        "specimen’s **apparent** branch slopes from **`specimen_apparent_bn_bp.csv`** "
        "(built by **`extract_bn_bp.py`** and consumed when assembling initial parameters). "
        "**median** picks the **median** of those apparent values; **q1** picks the **first quartile** "
        "(25th percentile—typically a lower, more conservative slope than the median). Other tokens "
        "in the same family include **mean**, **q3**, **min**, and **max**; numeric literals are copied "
        "unchanged."
    )
    parts.append("")
    if steel_seed_path.is_file():
        seeds = pd.read_csv(steel_seed_path, comment="#")
        parts.append(_dataframe_to_md_table(seeds))
    else:
        parts.append(f"*(Steel seed file not found: `{steel_seed_path}`)*")
    parts.append("")
    wf2, we2 = (
        DEFAULT_CALIBRATION_LOSS_SETTINGS.w_feat_l2,
        DEFAULT_CALIBRATION_LOSS_SETTINGS.w_energy_l2,
    )
    parts.append("## 3. Objective function")
    parts.append("")
    parts.append(
        f"The optimization target matches **`optimize_brb_mse`**: a weighted sum of **raw** metrics from "
        f"``config/calibration/set_id_settings.csv`` — per-``set_id`` objective weights for "
        f"cycle **landmark** error (**J_feat**, L2/L1), "
        f"per-cycle **energy** mismatch (**J_E**, L2/L1), and **binned cloud** terms (**J_binenv**, L2/L1). "
        f"Cycle weights **w_c** for **J_feat** default to uniform 1 (**`--amplitude-weights`** uses amplitude-based **w_c**). "
        f"Default L2 weights: **w_feat_l2** = {wf2:g}, **w_energy_l2** = {we2:g}. "
        f"**final_J_total** is the weighted objective; metrics CSVs store the raw L2/L1 terms and **J_binenv**."
    )
    parts.append("")
    parts.append("## 4. Analyses and what this report contains")
    parts.append("")
    parts.append(
        "Three analyses reuse the same simulator and loss. Each subsequent section of this document "
        "corresponds to one stage."
    )
    parts.append("")
    parts.append("### 4.1 Individual optimization")
    parts.append("")
    parts.append(
        "For specimens that undergo individual calibration, we optimize **separately** for each "
        "**set_id** seed, minimizing **J** over the optimizer degrees of freedom while holding geometry "
        "and catalog material fixed as prescribed. The **best** outcome per specimen is the seed whose "
        "run achieves the lowest **final_J_total** among **successful** fits."
    )
    parts.append("")
    parts.append(
        "**Reported here:** the table *Individual optimize — best set per specimen* (best **set_id**, "
        "**J**, **J_feat**, **J_E**, **J_binenv** columns), counts of **evaluable rows** (successful metrics with finite "
        "**final_J_total**), and *Parameters at each specimen's best set* (one wide row per specimen of "
        "the optimized BRB parameters at that best **set_id**)."
    )
    parts.append("")
    parts.append("### 4.2 Averaged parameters (mean per set) and evaluation")
    parts.append("")
    parts.append(
        "After individual optimization, for each **set_id** we compute a **weighted average** of the "
        "optimized parameters across specimens (catalog **averaged_weight**), producing one **averaged** "
        "parameter vector per seed family. That vector is merged into **all** specimen rows and the "
        "model is **re-simulated without further optimization**. This quantifies the error if the "
        "specimen set adopted the mean of individually tuned parameters for each steel-set choice."
    )
    parts.append("")
    parts.append(
        "**Reported here:** *Averaged optimize — best set per specimen* (which **set_id** minimizes "
        "**J** for each specimen under averaged steel), **Best overall set** for contributors "
        "(specimen-weighted mean **J** minimized over **set_id** among rows with **contributes_to_aggregate**), "
        "plus **best overall set** by contributor-weighted **final_unordered_J_binenv** "
        "(when present in the metrics CSV), with shared backbone and full parameter tables, "
        "and *Parameters for best overall sets* (shared backbone fields plus the full "
        "parameter table at the winning **set_id** for **J**)."
    )
    parts.append("")
    parts.append("### 4.3 Generalized (combined) optimization")
    parts.append("")
    parts.append(
        "Using the **individually optimized parameters CSV** as input, we run **generalized** optimization "
        "(`optimize_generalized_brb_mse`). The starting iterate is a catalog-weighted **averaged** vector derived "
        "from that table (the same construction idea as in **4.2 Averaged parameters**), then a shared subset of parameters "
        "(`PARAMS_TO_OPTIMIZE`) is refined **simultaneously** across specimens to lower a **generalized_weight**-weighted "
        "average of per-specimen **final_J_total**. The same landmark and energy objective apply. "
        "The result is one coupled parameter set per **set_id** for the specimen set, written out and evaluated "
        "like the averaged evaluation pass."
    )
    parts.append("")
    parts.append(
        "**Reported here:** the generalized counterparts of the averaged sections (*Generalized optimize*, **Best overall set**, "
        "best overall by **final_unordered_J_binenv** when present, "
        "and generalized *Parameters for best overall sets*). The appendix *mean J by set (contributors only)* "
        "compares contributor-averaged **J** between averaged evaluation and generalized evaluation at each **set_id**."
    )
    parts.append("")
    parts.append("## 5. Definitions used in the tables below")
    parts.append("")
    parts.append(
        "- **Best set_id per specimen** (individual / averaged / generalized blocks): the **set_id** with lowest "
        "**final_J_total** among **evaluable** metrics rows for that specimen (**success** and finite **J**)."
    )
    parts.append(
        "- **Best overall set_id** (averaged / generalized): the **set_id** that minimizes the **specimen_weight**-weighted "
        "mean **final_J_total** over rows with **contributes_to_aggregate**."
    )
    parts.append(
        "- **Best overall set_id by `final_unordered_J_binenv`**: the **set_id** that minimizes "
        "the same contributor **specimen_weight**-weighted mean of that column (finite values only per row; lower is better). "
        "This is a cloud diagnostic, not the calibration loss **J**."
    )
    parts.append("")
    return parts





def _params_cell_str(val: object, float_fmt: str) -> str:
    """String for one parameter cell in wide markdown tables."""
    if pd.isna(val):
        return ""
    if isinstance(val, (bool, np.bool_)):
        return str(val)
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return float_fmt.format(float(val))
    return str(val)


def _params_slice_to_md(params: pd.DataFrame, set_id: int, float_fmt: str = "{:.8g}") -> str:
    """Markdown table for all rows at one set_id."""
    blk = params.loc[params["set_id"] == set_id].sort_values("Name")
    if blk.empty:
        return "*(no rows)*\n"
    cols = list(blk.columns)
    rows: list[list[str]] = []
    for _, r in blk.iterrows():
        rows.append([_params_cell_str(r[c], float_fmt) for c in cols])
    return _md_table([str(c) for c in cols], rows)


def _individual_best_params_wide_md(
    best_ind: pd.DataFrame,
    individual_params: pd.DataFrame,
    float_fmt: str = "{:.8g}",
) -> str:
    """One markdown table: each row = one specimen; columns Name, set_id, then other param fields."""
    if best_ind.empty:
        return "*(no specimens)*\n"
    base_cols = list(individual_params.columns)
    if "Name" not in base_cols or "set_id" not in base_cols:
        col_order = base_cols
    else:
        col_order = ["Name", "set_id"] + [c for c in base_cols if c not in ("Name", "set_id")]
    ind_name = individual_params["Name"].astype(str)
    ind_sid = pd.to_numeric(individual_params["set_id"], errors="coerce")
    rows_out: list[list[str]] = []
    for _, br in best_ind.sort_values("Name").iterrows():
        nm = str(br["Name"])
        sid = int(br["best_set_id"])
        match = individual_params.loc[(ind_name == nm) & (ind_sid == float(sid))]
        if match.empty:
            row_cells = []
            for c in col_order:
                if c == "Name":
                    row_cells.append(nm)
                elif c == "set_id":
                    row_cells.append(str(sid))
                else:
                    row_cells.append("")
            rows_out.append(row_cells)
        else:
            row0 = match.iloc[0]
            rows_out.append([_params_cell_str(row0[c], float_fmt) for c in col_order])
    return _md_table([str(c) for c in col_order], rows_out)


def main() -> None:

    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    ap.add_argument("--individual-metrics", type=Path, default=OPTIMIZED_BRB_PARAMETERS_METRICS_PATH)

    ap.add_argument("--individual-params", type=Path, default=OPTIMIZED_BRB_PARAMETERS_PATH)

    ap.add_argument("--averaged-metrics", type=Path, default=AVERAGED_PARAMS_EVAL_METRICS_PATH)

    ap.add_argument("--generalized-metrics", type=Path, default=GENERALIZED_PARAMS_EVAL_METRICS_PATH)

    ap.add_argument("--averaged-params", type=Path, default=AVERAGED_BRB_PARAMETERS_PATH)

    ap.add_argument("--generalized-params", type=Path, default=GENERALIZED_BRB_PARAMETERS_PATH)

    ap.add_argument(
        "--specimens-csv",
        type=Path,
        default=BRB_SPECIMENS_CSV,
        help="Specimen catalog for the metadata table in the report narrative",
    )
    ap.add_argument(
        "--set-id-settings",
        type=Path,
        default=SET_ID_SETTINGS_CSV,
        help="Unified set_id settings table for the report narrative",
    )

    ap.add_argument(

        "--output",

        type=Path,

        default=RESULTS_CALIBRATION / "averaged_vs_generalized_metrics_report.md",

    )

    ap.add_argument(

        "--per-specimen-delta",

        action="store_true",

        help="Append averaged vs generalized contributing-row J comparison table",

    )

    ap.add_argument(

        "--per-specimen",

        action="store_true",

        help=argparse.SUPPRESS,

    )

    args = ap.parse_args()

    if args.per_specimen:

        args.per_specimen_delta = True



    ind_df = _load_metrics(args.individual_metrics)

    individual_params = _load_params(args.individual_params)

    averaged_df = _load_metrics(args.averaged_metrics)

    generalized_df = _load_metrics(args.generalized_metrics)

    averaged_params = _load_params(args.averaged_params)

    generalized_params = _load_params(args.generalized_params)



    metrics_cols = ["final_J_total", *REPORT_METRIC_COLS]

    for d in (ind_df, averaged_df, generalized_df):

        for c in metrics_cols:

            if c not in d.columns:

                d[c] = np.nan

            d[c] = pd.to_numeric(d[c], errors="coerce")



    ind_mask = _eval_mask(ind_df)

    p_mask_eval = _eval_mask(averaged_df)

    j_mask_eval = _eval_mask(generalized_df)

    p_mask_c = _contributing_mask(averaged_df)

    j_mask_c = _contributing_mask(generalized_df)



    best_ind = _best_set_per_specimen(ind_df, ind_mask)

    best_p = _best_set_per_specimen(averaged_df, p_mask_eval)

    best_j = _best_set_per_specimen(generalized_df, j_mask_eval)



    p_agg = _aggregate_by_set(averaged_df, p_mask_c, metrics_cols)

    j_agg = _aggregate_by_set(generalized_df, j_mask_c, metrics_cols)

    best_set_averaged, mean_j_p = _best_overall_set_id(p_agg)

    best_set_generalized, mean_j_j = _best_overall_set_id(j_agg)

    best_set_binenv_p, mean_binenv_p = _best_overall_set_id_for_metric(
        p_agg, "final_unordered_J_binenv"
    )
    best_set_binenv_j, mean_binenv_j = _best_overall_set_id_for_metric(
        j_agg, "final_unordered_J_binenv"
    )

    const_p = _shared_backbone_constant_columns(averaged_params, best_set_averaged) if best_set_averaged is not None else {}

    const_j = _shared_backbone_constant_columns(generalized_params, best_set_generalized) if best_set_generalized is not None else {}



    lines: list[str] = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# BRB calibration report: individual, averaged, and generalized analyses")

    lines.append("")

    lines.append(f"Generated: **{now}**")

    lines.append("")

    lines.extend(_report_narrative_intro(args.specimens_csv, args.set_id_settings))

    lines.append("## Inputs")

    lines.append("")

    lines.append(f"- Specimen catalog (narrative): `{args.specimens_csv.resolve()}`")

    lines.append(f"- set_id settings (narrative): `{args.set_id_settings.resolve()}`")

    lines.append(f"- Individual metrics: `{args.individual_metrics.resolve()}`")

    lines.append(f"- Individual parameters: `{args.individual_params.resolve()}`")

    lines.append(f"- Averaged metrics: `{args.averaged_metrics.resolve()}`")

    lines.append(f"- Generalized metrics: `{args.generalized_metrics.resolve()}`")

    lines.append(f"- Averaged parameters: `{args.averaged_params.resolve()}`")

    lines.append(f"- Generalized parameters: `{args.generalized_params.resolve()}`")

    lines.append("")

    lines.append("## Executive summary")

    lines.append("")

    if best_set_averaged is not None:

        lines.append(

            f"- **Averaged** (contributors): best overall **set_id = {best_set_averaged}** "

            f"(weighted mean **final_J_total** = {_fmt_sci(mean_j_p)}). Parameters: shared-backbone list + full table below."

        )

    else:

        lines.append("- **Averaged** (contributors): could not determine a best overall set.")

    if best_set_generalized is not None:

        lines.append(

            f"- **Generalized** (contributors): best overall **set_id = {best_set_generalized}** "

            f"(weighted mean **final_J_total** = {_fmt_sci(mean_j_j)}). Parameters: shared-backbone list + full table below."

        )

    else:

        lines.append("- **Generalized** (contributors): could not determine a best overall set.")

    if best_set_binenv_p is not None and np.isfinite(mean_binenv_p):

        lines.append(

            f"- **Averaged** (contributors): best overall **set_id** by weighted mean **final_unordered_J_binenv** = **{best_set_binenv_p}** "

            f"(mean = {_fmt_sci(mean_binenv_p)}); parameters in *Best overall set by `final_unordered_J_binenv`* under **Averaged optimize**."

        )

    if best_set_binenv_j is not None and np.isfinite(mean_binenv_j):

        lines.append(

            f"- **Generalized** (contributors): best overall **set_id** by weighted mean **final_unordered_J_binenv** = **{best_set_binenv_j}** "

            f"(mean = {_fmt_sci(mean_binenv_j)}); parameters in *Best overall set by `final_unordered_J_binenv`* under **Generalized optimize**."

        )

    lines.append("")

    lines.append("## Individual optimize — best set per specimen")

    lines.append("")

    lines.append(f"Evaluable rows: **{int(ind_mask.sum())}** / {len(ind_df)}.")

    lines.append("")

    ir: list[list[str]] = []

    for _, r in best_ind.sort_values("Name").iterrows():

        row_cells = [
            str(r["Name"]),
            str(int(r["best_set_id"])),
            _fmt_sci(float(r["final_J_total"])),
        ]
        for col in REPORT_METRIC_COLS:
            row_cells.append(_fmt_metric_table_cell(r, col))
        ir.append(row_cells)

    ind_headers = ["Name", "best_set_id", "J", *REPORT_METRIC_HEADERS]
    lines.append(_md_table(ind_headers, ir))

    lines.append("")

    lines.append("### Parameters at each specimen's best set")

    lines.append("")

    lines.append(

        "Single wide table — **one row per specimen**; **`Name`**, **`set_id`** (best set), then "

        "the remaining columns from individual parameters "

        f"(`{args.individual_params.resolve()}`). Missing rows show blanks."

    )

    lines.append("")

    lines.append(_individual_best_params_wide_md(best_ind, individual_params))

    lines.append("")

    lines.append("## Averaged optimize — best set per specimen")

    lines.append("")

    lines.append(f"Evaluable rows: **{int(p_mask_eval.sum())}** / {len(averaged_df)}.")

    lines.append("")

    pr = []

    for _, r in best_p.sort_values("Name").iterrows():

        row_cells = [
            str(r["Name"]),
            str(int(r["best_set_id"])),
            _fmt_sci(float(r["final_J_total"])),
        ]
        for col in REPORT_METRIC_COLS:
            row_cells.append(_fmt_metric_table_cell(r, col))
        pr.append(row_cells)

    avg_headers = ["Name", "best_set_id", "J", *REPORT_METRIC_HEADERS]
    lines.append(_md_table(avg_headers, pr))

    lines.append("")

    lines.append("### Best overall set (contributing specimens)")

    lines.append("")

    lines.append(

        "**Primary objective:** minimum weighted mean **final_J_total** over rows with **contributes_to_aggregate**."

    )

    lines.append("")

    if best_set_averaged is not None:

        lines.append(

            f"- **set_id = {best_set_averaged}** (minimum weighted mean **final_J_total** = {_fmt_sci(mean_j_p)})."

        )

        lines.append(f"- Contributing rows used: **{int(p_mask_c.sum())}**.")

    else:

        lines.append("*(could not determine — no contributing evaluable rows)*")

    lines.append("")

    lines.append("#### Best overall set by `final_unordered_J_binenv` (contributors)")

    lines.append("")

    lines.append(

        "Minimum weighted mean **final_unordered_J_binenv** over contributing rows (same weights; lower is better)."

    )

    lines.append("")

    lines.extend(

        _md_lines_best_set_by_metric(

            averaged_params,

            best_set_binenv_p,

            "final_unordered_J_binenv",

            mean_binenv_p,

        )

    )

    lines.append("## Generalized optimize — best set per specimen")

    lines.append("")

    lines.append(f"Evaluable rows: **{int(j_mask_eval.sum())}** / {len(generalized_df)}.")

    lines.append("")

    jr = []

    for _, r in best_j.sort_values("Name").iterrows():

        row_cells = [
            str(r["Name"]),
            str(int(r["best_set_id"])),
            _fmt_sci(float(r["final_J_total"])),
        ]
        for col in REPORT_METRIC_COLS:
            row_cells.append(_fmt_metric_table_cell(r, col))
        jr.append(row_cells)

    gen_headers = ["Name", "best_set_id", "J", *REPORT_METRIC_HEADERS]
    lines.append(_md_table(gen_headers, jr))

    lines.append("")

    lines.append("### Best overall set (contributing specimens)")

    lines.append("")

    lines.append(

        "**Primary objective:** minimum weighted mean **final_J_total** over rows with **contributes_to_aggregate**."

    )

    lines.append("")

    if best_set_generalized is not None:

        lines.append(

            f"- **set_id = {best_set_generalized}** (minimum weighted mean **final_J_total** = {_fmt_sci(mean_j_j)})."

        )

        lines.append(f"- Contributing rows used: **{int(j_mask_c.sum())}**.")

    else:

        lines.append("*(could not determine — no contributing evaluable rows)*")

    lines.append("")

    lines.append("#### Best overall set by `final_unordered_J_binenv` (contributors)")

    lines.append("")

    lines.append(

        "Minimum weighted mean **final_unordered_J_binenv** over contributing rows (same weights; lower is better)."

    )

    lines.append("")

    lines.extend(

        _md_lines_best_set_by_metric(

            generalized_params,

            best_set_binenv_j,

            "final_unordered_J_binenv",

            mean_binenv_j,

        )

    )

    lines.append("## Parameters for best overall sets")

    lines.append("")

    lines.append(

        "Columns that are **identical for every specimen** at the chosen ``set_id`` "

        "(shared across the specimen set at that ``set_id``; usually the averaged/generalized backbone ``R0``, ``cR1``, ``cR2``, ``a1``–``a4``)."

    )

    lines.append("")

    lines.append(f"### Averaged — set_id {best_set_averaged if best_set_averaged is not None else '—'}")

    lines.append("")

    if const_p:

        for k, v in const_p.items():

            lines.append(f"- `{k}` = {v}")

    else:

        lines.append("*(none or missing parameters CSV slice)*")

    lines.append("")

    lines.append("All specimen rows at this set (full parameter vector):")

    lines.append("")

    if best_set_averaged is not None:

        lines.append(_params_slice_to_md(averaged_params, best_set_averaged))

    lines.append("")



    lines.append(f"### Generalized — set_id {best_set_generalized if best_set_generalized is not None else '—'}")

    lines.append("")

    if const_j:

        for k, v in const_j.items():

            lines.append(f"- `{k}` = {v}")

    else:

        lines.append("*(none or missing parameters CSV slice)*")

    lines.append("")

    lines.append("All specimen rows at this set (full parameter vector):")

    lines.append("")

    if best_set_generalized is not None:

        lines.append(_params_slice_to_md(generalized_params, best_set_generalized))

    lines.append("")



    merged = j_agg.merge(

        p_agg,

        on="set_id",

        how="outer",

        suffixes=("_generalized", "_averaged"),

    ).sort_values("set_id")

    lines.append("## Appendix: mean J by set (contributors only) — generalized vs averaged")

    lines.append("")

    ar = []

    for _, row in merged.iterrows():

        jj = float(row["final_J_total_generalized"]) if pd.notna(row.get("final_J_total_generalized")) else float("nan")

        jp = float(row["final_J_total_averaged"]) if pd.notna(row.get("final_J_total_averaged")) else float("nan")

        win = ""

        if np.isfinite(jj) and np.isfinite(jp):

            if jj < jp - 1e-18:

                win = "generalized"

            elif jp < jj - 1e-18:

                win = "averaged"

            else:

                win = "tie"

        ar.append(

            [

                str(int(row["set_id"])),

                _fmt_sci(jj),

                _fmt_sci(jp),

                _fmt_sci(jp - jj),

                win,

            ]

        )

    lines.append(_md_table(["set_id", "mean_J_generalized", "mean_J_averaged", "Δ(avg−gen)", "lower_J"], ar))

    lines.append("")



    if args.per_specimen_delta:

        cols = ["Name", "set_id", "specimen_weight"] + metrics_cols

        js = generalized_df.loc[j_mask_c, cols].copy()

        ps = averaged_df.loc[j_mask_c, cols].copy()

        comp = js.merge(

            ps,

            on=["Name", "set_id"],

            how="outer",

            suffixes=("_generalized", "_averaged"),

            indicator=True,

        ).sort_values(["Name", "set_id"])

        lines.append("## Appendix: per-(Name, set_id) J — contributing rows only")

        lines.append("")

        ph = [

            "Name",

            "set_id",

            "J_generalized",

            "J_averaged",

            "Δ(A−G)",

            "_merge",

        ]

        pr2 = []

        for _, r in comp.iterrows():

            jtj = r.get("final_J_total_generalized")

            jtp = r.get("final_J_total_averaged")

            d = (

                float(jtp) - float(jtj)

                if pd.notna(jtj) and pd.notna(jtp)

                else float("nan")

            )

            pr2.append(

                [

                    str(r["Name"]),

                    str(int(r["set_id"])) if pd.notna(r["set_id"]) else "",

                    _fmt_sci(float(jtj)) if pd.notna(jtj) else "",

                    _fmt_sci(float(jtp)) if pd.notna(jtp) else "",

                    _fmt_sci(d),

                    str(r["_merge"]),

                ]

            )

        lines.append(_md_table(ph, pr2))

        lines.append("")



    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output.resolve()}")





if __name__ == "__main__":

    main()


