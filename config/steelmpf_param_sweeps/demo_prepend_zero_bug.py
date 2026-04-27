"""SteelMPF zeroLength demo: Path series with vs without ``-prependZero``. Writes ``demo_prepend_zero_bug.png``.

Uses the same post-``a4`` tail as ``model.material.ops_BRB_material`` (``-ult``, then ``fup``, ``fun``, ``Ru0``).
"""

import matplotlib.pyplot as plt
import numpy as np

try:
    import opensees as ops
except ImportError:
    import openseespy.opensees as ops

E = 29000.0
FY = 50.0
B_P = 0.02
B_N = 0.02
R0 = 20.0
C_R1 = 0.925
C_R2 = 0.15
A1 = 0.04
A2 = 1.0
A3 = 0.04
A4 = 1.0
fup = 4.0 * FY
fun = 4.0 * FY
Ru0 = 5.0
EPS_Y = FY / E

EPS_PEAK = 2.0 * EPS_Y
N = 10
D = np.linspace(0.0, EPS_PEAK, N, endpoint=False)
DISP_HISTORY = np.r_[D, EPS_PEAK - D, -D, -(EPS_PEAK - D)]


def run_simulation(disp_history, prepend_zero=False):
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.fix(1, 1)
    ops.node(2, 0.0)
    ops.uniaxialMaterial(
        "SteelMPF",
        1,
        FY,
        FY,
        E,
        B_P,
        B_N,
        R0,
        C_R1,
        C_R2,
        A1,
        A2,
        A3,
        A4,
        "-ult",
        fup,
        fun,
        Ru0,
    )
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    dt = 1.0
    extra = ["-useLast"] + (["-prependZero"] if prepend_zero else [])
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *disp_history, *extra)
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, 1.0)
    ops.integrator("LoadControl", dt)
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("UmfPack")
    ops.analysis("Static", "-noWarnings")

    n = len(disp_history)
    stress = np.zeros(n)
    strain = np.zeros(n)
    time = np.zeros(n)
    for i in range(n):
        if ops.analyze(1) != 0:
            raise RuntimeError("analyze failed at step %d/%d" % (i + 1, n))
        stress[i] = ops.eleResponse(1, "material", 1, "stress")[0]
        strain[i] = ops.eleResponse(1, "material", 1, "strain")[0]
        time[i] = ops.getTime()
    return stress, strain, time


def main():
    runs = (
        (run_simulation(DISP_HISTORY, False), "prepend_zero=False", "-x"),
        (run_simulation(DISP_HISTORY, True), "prepend_zero=True", "--o"),
    )

    fig, (ax_t, ax_ss) = plt.subplots(2, 1, figsize=(5.5, 6.0))
    for (s, e, t), lab, sty in runs:
        en = e / EPS_Y
        ax_t.plot(t, en, sty, label=lab, lw=1.2, markersize=4)
        ax_ss.plot(en, s / FY, sty, label=lab, lw=1.2, markersize=4)

    ax_t.set_ylabel(r"$\varepsilon / \varepsilon_y$")
    ax_t.set_xlabel("pseudo-time")
    ax_t.axhline(0.0, color="0.8", lw=0.8)
    ax_t.grid(True, alpha=0.35)

    ax_ss.set_xlabel(r"$\varepsilon / \varepsilon_y$")
    ax_ss.set_ylabel(r"$\sigma / f_y$")
    ax_ss.axhline(0.0, color="0.8", lw=0.8)
    ax_ss.axvline(0.0, color="0.8", lw=0.8)
    ax_ss.grid(True, alpha=0.35)
    ax_ss.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig("demo_prepend_zero_bug.png", dpi=150)
    plt.close(fig)
    print("Wrote demo_prepend_zero_bug.png")


if __name__ == "__main__":
    main()
