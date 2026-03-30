"""BRB calibration models."""

from .brace_geometry import compute_Q

__all__ = ["run_simulation", "compute_Q"]


def run_simulation(*args, **kwargs):
    """Lazy import so ``import model`` / ``model.brace_geometry`` does not load OpenSees."""
    from .corotruss import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)
