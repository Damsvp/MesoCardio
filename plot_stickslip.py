"""
Reproduce the 3x3 stick-slip figure from osc_*.npz files.

Run in the directory containing the osc_*.npz files and clamp_sweep.npy:
    python plot_stickslip.py

Or pass the three oscillation files explicitly:
    python plot_stickslip.py osc_file1.npz osc_file2.npz osc_file3.npz

Normalization:
    tau0 is computed with the same convention as plot_S.py:
    tau0 = mean(tau(v=+0.1), tau(v=-0.1)).
    This avoids using the v=0 clamp point, which is noted in plot_S.py as
    a special, quenched finite-N case.

Soft-device load per head:
    p(t) = p0 + kL * (V*t - xc(t))

Shortening velocity:
    u = -dxc/dt, smoothed over 0.25 ms.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FILES = [
    "osc_N100_k0.04_V2.npz",
    "osc_N100_k0.10_V2.npz",
    "osc_N300_k0.04_V2.npz",
]
WINDOW_MS = 0.25


def get_tau0(sweep_file="clamp_sweep.npy"):
    """Use exactly the reference-force convention from plot_S.py."""
    path = Path(sweep_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {sweep_file}. Run sweep_clamp.py first, "
            "or place the resulting clamp_sweep.npy beside this script."
        )

    sweep = np.asarray(np.load(path), dtype=float)
    if sweep.ndim != 2 or sweep.shape[1] < 2:
        raise ValueError(
            f"{sweep_file} must contain at least two columns: velocity, tau_mean."
        )

    v, tau = sweep[:, 0], sweep[:, 1]

    def value_at(target):
        # Tolerant exact match first; otherwise use the nearest recorded velocity.
        idx = np.flatnonzero(np.isclose(v, target, atol=1e-8, rtol=0.0))
        if len(idx):
            return float(tau[idx[0]])
        nearest = int(np.argmin(np.abs(v - target)))
        if not np.isclose(v[nearest], target, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"{sweep_file} has no velocity v={target:g}; "
                "the tau0 convention requires v=+0.1 and v=-0.1."
            )
        return float(tau[nearest])

    tau_plus = value_at(+0.1)
    tau_minus = value_at(-0.1)
    tau0 = 0.5 * (tau_plus + tau_minus)

    if not np.isfinite(tau0) or np.isclose(tau0, 0.0):
        raise ValueError(f"Computed invalid tau0={tau0!r} from {sweep_file}.")

    print(f"tau(v=+0.1) = {tau_plus:.6g}")
    print(f"tau(v=-0.1) = {tau_minus:.6g}")
    print(f"tau0 = {tau0:.6g} (mean of the v=+/-0.1 clamp forces)")
    return tau0


def centered_moving_average(values, n):
    """Moving average with output length equal to input length."""
    n = max(1, int(n))
    return np.convolve(values, np.ones(n) / n, mode="same")


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else FILES
    if len(files) != 3:
        raise ValueError(
            f"Expected exactly 3 osc_*.npz files for the 3x3 plot; got {len(files)}."
        )

    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Could not find these oscillation files:\n  "
            + "\n  ".join(missing)
            + "\nRun osc_run.py to create them, or pass their correct filenames."
        )

    tau0 = get_tau0()
    fig, axes = plt.subplots(3, 3, figsize=(15, 7.5))

    for row, filename in enumerate(files):
        with np.load(filename) as data:
            t = np.asarray(data["t"], dtype=float)       # ms
            xc = np.asarray(data["xc"], dtype=float)     # nm
            vel = np.asarray(data["vel"], dtype=float)   # nm/ms
            p0 = float(data["p0"])                       # pN
            kL = float(data["kL"])                       # pN/nm
            V = float(data["V"])                         # nm/ms

        if len(t) < 2 or len(xc) != len(t) or len(vel) != len(t):
            raise ValueError(f"{filename}: t, xc and vel must have equal length >= 2.")

        # Applied opposing load per head from osc_run.py's soft-device force law.
        load = p0 + kL * (V * t - xc)

        # H-model convention: x_c-dot < 0 is contraction, so u = -x_c-dot.
        u = -vel

        # osc_run.py records every 25 integration steps (dt=0.002 ms => 0.05 ms).
        # Calculate the window from the actual saved time array, rather than assuming it.
        dt_record = float(np.median(np.diff(t)))
        n_window = max(1, int(round(WINDOW_MS / dt_record)))
        u_mean = centered_moving_average(u, n_window)

        normalized_load = load / tau0

        # Column 1: normalized load versus time.
        ax = axes[row, 0]
        ax.plot(t, normalized_load, linewidth=0.7)
        ax.set_title(Path(filename).name)
        ax.set_ylabel(r"load per head / $\tau_0$")
        ax.set_xlabel("t (ms)")

        # Column 2: shortening velocity versus time.
        ax = axes[row, 1]
        ax.plot(t, u_mean, linewidth=0.7)
        ax.set_ylabel(r"$u$ (0.25 ms mean)")
        ax.set_xlabel("t (ms)")

        # Column 3: phase portrait, velocity versus normalized load.
        ax = axes[row, 2]
        ax.plot(normalized_load, u_mean, linewidth=0.6)
        ax.set_xlabel(r"load/$\tau_0$")
        ax.set_ylabel("u")

    fig.tight_layout()
    output = "stickslip.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output}")
    plt.show()


if __name__ == "__main__":
    main()
