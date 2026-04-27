"""
Corotational truss (BRB) model using OpenSees.

Material is uniaxial SteelMPF or Steel4 (``material.ops_BRB_material`` / ``ops_BRB_material_steel4``).
Uses E_hat = Q*E with Q from BRB geometry.
"""

from __future__ import annotations

import numpy as np

try:
    import opensees as ops
except ImportError:
    import openseespy.opensees as ops

from calibrate.steel_model import (
    STEEL_MODEL_STEEL4,
    STEEL_MODEL_STEELMPF,
    clamp_steel4_isotropic_slopes,
    normalize_steel_model,
)

from .brace_geometry import compute_Q
from .material import ops_BRB_material, ops_BRB_material_steel4


def run_simulation(
    displacement: np.ndarray,
    *,
    steel_model: str = STEEL_MODEL_STEELMPF,
    # Geometry (e.g. inches, in^2)
    L_T: float,
    L_y: float,
    A_sc: float,
    A_t: float,
    # Strengths (e.g. ksi). Steel4 uses ``fyp`` as ``Fy``; SteelMPF uses both.
    fyp: float,
    fyn: float,
    # Modulus and kinematic ratios (E in same units as stress, e.g. ksi)
    E: float,
    b_p: float,
    b_n: float,
    R0: float = 20.0,
    cR1: float = 0.925,
    cR2: float = 0.15,
    # SteelMPF isotropic (ignored for Steel4)
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
    # SteelMPF post-a4: ratios × fyp/fyn → ``fup``/``fun``; ``-ult`` then ``fup``, ``fun``, ``Ru0`` (ignored for Steel4).
    fup_ratio: float = 4.0,
    fun_ratio: float = 4.0,
    Ru0: float = 5.0,
    # Steel4 ``-iso`` branch (ignored for SteelMPF)
    b_ip: float = 0.01,
    rho_ip: float = 2.0,
    b_lp: float = 0.001,
    R_i: float = 20.0,
    l_yp: float = 0.01,
    b_ic: float = 0.01,
    rho_ic: float = 2.0,
    b_lc: float = 0.001,
) -> np.ndarray:
    """
    Run the BRB corotruss simulation and return force history for the given displacement history.

    Young's modulus is adjusted with E_hat = Q*E, where
    Q = 1 / ( (2(L_T - L_y)/L_T)/(A_t/A_sc) + L_y/L_T ).

    ``steel_model`` selects SteelMPF or Steel4.
    """
    displacement = np.asarray(displacement, dtype=float)
    n = len(displacement)

    Q = compute_Q(L_T, L_y, A_sc, A_t)
    E_hat = Q * E

    sm = normalize_steel_model(steel_model)

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    ops.node(1, 0.0, 0.0)
    ops.fix(1, 1, 1)
    ops.node(2, L_T, 0.0)
    ops.fix(2, 0, 1)  # fix y only; x prescribed by pattern

    if sm == STEEL_MODEL_STEEL4:
        sl = clamp_steel4_isotropic_slopes(
            {
                "b_p": float(b_p),
                "b_n": float(b_n),
                "b_ip": float(b_ip),
                "b_lp": float(b_lp),
                "b_ic": float(b_ic),
                "b_lc": float(b_lc),
            }
        )
        brb_mat_tag = ops_BRB_material_steel4(
            Fy=float(fyp),
            E0=float(E_hat),
            b_p=float(b_p),
            R0=float(R0),
            cR1=float(cR1),
            cR2=float(cR2),
            b_n=float(b_n),
            b_ip=float(sl["b_ip"]),
            rho_ip=float(rho_ip),
            b_lp=float(sl["b_lp"]),
            R_i=float(R_i),
            l_yp=float(l_yp),
            b_ic=float(sl["b_ic"]),
            rho_ic=float(rho_ic),
            b_lc=float(sl["b_lc"]),
        )
    elif sm == STEEL_MODEL_STEELMPF:
        fup = float(fup_ratio) * float(fyp)
        fun = float(fun_ratio) * float(fyn)
        brb_mat_tag = ops_BRB_material(
            fyp=fyp,
            fyn=fyn,
            E=E_hat,
            b_p=b_p,
            b_n=b_n,
            R0=R0,
            cR1=cR1,
            cR2=cR2,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            fup=fup,
            fun=fun,
            Ru0=Ru0,
        )
    else:
        raise ValueError(
            f"Unknown steel_model {steel_model!r}; expected "
            f"{STEEL_MODEL_STEELMPF!r} or {STEEL_MODEL_STEEL4!r}"
        )

    ops.element("corotTruss", 1, 1, 2, A_sc, brb_mat_tag)

    # Path series: uniform dt=1 => times 0..n; length n+1 with zero initial disp.
    # -useLast avoids load factor 0 when pseudo-time rounds past the final sample.
    dt = 1.0
    path_values = np.empty(n + 1, dtype=np.float64)
    path_values[0] = 0.0
    path_values[1:] = displacement
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *path_values, "-useLast")
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, 1.0)

    ops.integrator("LoadControl", dt)
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.analysis("Static", "-noWarnings")

    force = np.zeros(n)
    for i in range(n):
        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(
                f"OpenSees analyze failed at step {i + 1}/{n}. "
                "Try smaller displacement increments or check material/geometry."
            )
        axial_force = ops.eleResponse(1, "axialForce")
        force[i] = axial_force[0] if isinstance(axial_force, (list, tuple)) else axial_force

    return force
