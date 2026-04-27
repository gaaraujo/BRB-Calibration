"""OpenSees BRB: uniaxial SteelMPF or Steel4 for the corotational truss.

SteelMPF: with **native** ``opensees``, after ``a4`` pass ``-ult`` then ``fup``, ``fun``, ``Ru0``
(same units as ``fyp``/``fyn``). With **openseespy** only, the tail is omitted (stock ``SteelMPF`` API).
"""

from __future__ import annotations

try:
    import opensees as ops

    OPENSEES_IS_NATIVE = True
except ImportError:
    import openseespy.opensees as ops

    OPENSEES_IS_NATIVE = False

BRB_MAT_TAG = 1000


def ops_BRB_material(
    *,
    fyp: float = 40.0,
    fyn: float = 40.0,
    E: float = 29000.0,
    b_p: float = 0.02,
    b_n: float = 0.02,
    R0: float = 20.0,
    cR1: float = 0.925,
    cR2: float = 0.15,
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
    # Ultimate tail (``-ult``, ``fup``, ``fun``, ``Ru0``): native OpenSees only; ignored for openseespy.
    fup: float | None = None,
    fun: float | None = None,
    Ru0: float = 5.0,
) -> int:
    """Define uniaxial SteelMPF for the BRB truss; returns the material tag."""
    base = (
        "SteelMPF",
        BRB_MAT_TAG,
        fyp,
        fyn,
        E,
        b_p,
        b_n,
        R0,
        cR1,
        cR2,
        a1,
        a2,
        a3,
        a4,
    )
    if OPENSEES_IS_NATIVE:
        fup_v = float(fyp) * 4.0 if fup is None else float(fup)
        fun_v = float(fyn) * 4.0 if fun is None else float(fun)
        ops.uniaxialMaterial(*base, "-ult", fup_v, fun_v, float(Ru0))
    else:
        ops.uniaxialMaterial(*base)
    return BRB_MAT_TAG


def ops_BRB_material_steel4(
    *,
    Fy: float,
    E0: float,
    b_p: float,
    R0: float,
    cR1: float,
    cR2: float,
    b_n: float,
    b_ip: float,
    rho_ip: float,
    b_lp: float,
    R_i: float,
    l_yp: float,
    b_ic: float,
    rho_ic: float,
    b_lc: float,
) -> int:
    """
    Steel4 with ``-asym``, ``-kin`` (shared R0/cR1/cR2 on both sides), ``-iso``, and ``-mem 0``.

    Uses one calibrated ``R_i`` for both tension and compression isotropic transition slots.
    """
    ops.uniaxialMaterial(
        "Steel4", BRB_MAT_TAG,
        float(Fy), float(E0),
        "-asym", "-kin",
        float(b_p), float(R0), float(cR1), float(cR2),
        float(b_n), float(R0), float(cR1), float(cR2),
        "-iso",
        float(b_ip), float(rho_ip), float(b_lp), float(R_i), float(l_yp),
        float(b_ic), float(rho_ic), float(b_lc), float(R_i),
        "-mem", 50,
    )
    return BRB_MAT_TAG


# Back-compat alias used in older references
STEEL_MPF_TAG = BRB_MAT_TAG


if __name__ == "__main__":
    ops_BRB_material()
