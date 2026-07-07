"""Symmetric overlay axis limits/ticks shared by per-set and all-set F–u figures."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from postprocess.plot_specimens import apply_normalized_fu_axes

_LIMIT_MARGIN = 1.05
_MAX_TICKS_TOTAL = 5  # including zero
_MAX_STEPS_FROM_ZERO = (_MAX_TICKS_TOTAL - 1) // 2  # -2s,-s,0,s,2s


@dataclass(frozen=True)
class SymmetricAxisSpec:
    half: float
    step: float


@dataclass(frozen=True)
class ForceDefOverlayAxisSpecs:
    physical_x: SymmetricAxisSpec
    physical_y: SymmetricAxisSpec
    normalized_x: SymmetricAxisSpec
    normalized_y: SymmetricAxisSpec


def _margined_half_data(half_data: float) -> float:
    return max(float(half_data), 1e-12) * _LIMIT_MARGIN


def _nice_step_at_least(min_step: float) -> float:
    if min_step <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(min_step))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * mag
        if step + 1e-15 >= min_step:
            return step
    return 10.0 * mag


def _nice_log_step(half_data: float) -> float:
    min_step = _margined_half_data(half_data) / _MAX_STEPS_FROM_ZERO
    return _nice_step_at_least(min_step)


def _norm_strain_tick_step(half_data: float) -> float:
    min_step = _margined_half_data(half_data) / _MAX_STEPS_FROM_ZERO
    for pct in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0):
        step = pct / 100.0
        if step + 1e-15 >= min_step:
            return step
    return _nice_step_at_least(min_step)


def _symmetric_half_limit(half_data: float, step: float) -> float:
    raw = _margined_half_data(half_data)
    n_steps = int(np.ceil(raw / step))
    n_steps = max(1, min(n_steps, _MAX_STEPS_FROM_ZERO))
    return float(n_steps * step)


def _axis_spec(half_data: float, *, normalized_strain: bool) -> SymmetricAxisSpec:
    step = _norm_strain_tick_step(half_data) if normalized_strain else _nice_log_step(half_data)
    return SymmetricAxisSpec(half=_symmetric_half_limit(half_data, step), step=step)


def compute_force_def_overlay_axis_specs(
    displacement: np.ndarray,
    F_exp: np.ndarray,
    sim_runs: list[tuple[np.ndarray, float, float, float]],
) -> ForceDefOverlayAxisSpecs:
    """
    Global symmetric limits for one specimen across all parameter sets.

    ``sim_runs``: ``(F_sim, fyp, A_sc, L_y)`` per successful set.
    """
    D = np.asarray(displacement, dtype=float)
    F_exp = np.asarray(F_exp, dtype=float)
    phys_y = [F_exp, *(F_sim for F_sim, *_ in sim_runs)]
    norm_y_parts: list[np.ndarray] = []
    norm_x_parts: list[np.ndarray] = []
    for F_sim, fy, A_sc, L_y in sim_runs:
        fyA = fy * A_sc
        if fyA > 0:
            norm_y_parts.append(F_exp / fyA)
            norm_y_parts.append(np.asarray(F_sim, dtype=float) / fyA)
        d_norm = D / L_y if L_y > 0 else D
        norm_x_parts.append(d_norm)

    phys_x_data = float(np.nanmax(np.abs(D))) or 1.0
    phys_y_data = float(np.nanmax(np.abs(np.concatenate(phys_y)))) or 1.0
    norm_x_data = (
        float(np.nanmax(np.abs(np.concatenate(norm_x_parts)))) if norm_x_parts else 1.0
    )
    norm_y_data = (
        float(np.nanmax(np.abs(np.concatenate(norm_y_parts)))) if norm_y_parts else 1.0
    )

    return ForceDefOverlayAxisSpecs(
        physical_x=_axis_spec(phys_x_data, normalized_strain=False),
        physical_y=_axis_spec(phys_y_data, normalized_strain=False),
        normalized_x=_axis_spec(norm_x_data, normalized_strain=True),
        normalized_y=_axis_spec(norm_y_data, normalized_strain=False),
    )


def apply_symmetric_axis_specs(
    ax: Axes,
    *,
    x: SymmetricAxisSpec,
    y: SymmetricAxisSpec,
    normalized_strain_x: bool = False,
) -> None:
    ax.set_xlim(-x.half, x.half)
    ax.set_ylim(-y.half, y.half)
    ax.xaxis.set_major_locator(MultipleLocator(x.step))
    ax.yaxis.set_major_locator(MultipleLocator(y.step))
    if normalized_strain_x:
        apply_normalized_fu_axes(ax)
