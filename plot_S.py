"""Force-speed picture of the h-model: velocity clamp (paper) vs soft device (imposed force).
Needs clamp_sweep.npy (sweep_clamp.py) and ramp_*.npz (soft_ramp.py)."""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = np.load("clamp_sweep.npy")
S = S[np.argsort(S[:, 0])]
v, tau = S[:, 0], S[:, 1]
# v = 0 is special: with x_c frozen the ensemble is quenched in h (finite-N noise). Use the mean of v = +-0.1.
i0 = np.where(v == 0)[0][0]
tau[i0] = 0.5 * (tau[np.where(v == 0.1)[0][0]] + tau[np.where(v == -0.1)[0][0]])
u = -v                                   # shortening speed (nm/ms = um/s)
order = np.argsort(u)
u, tau = u[order], tau[order]
tau0 = tau[np.argmin(np.abs(u))]
ug = np.linspace(u.min(), u.max(), 400)
tg = np.interp(ug, u, tau)

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

# (a) velocity clamp
ax[0].plot(tau / tau0, u, "o-", color="C1")
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xlabel(r"$\tau_c/\tau_{c,0}$")
ax[0].set_ylabel(r"$u=-\dot x_c$  ($\mu$m/s)")
ax[0].set_title("velocity clamp (Fig. 5.2 axes), u<0: lengthening")

# (b) prediction for soft device: P(u) = tau_ss(u) - nu_t u
for nu_t, c in [(0.0, "k"), (0.05, "C0"), (0.15, "C2"), (0.30, "C3")]:
    ax[1].plot((tg - nu_t * ug) / tau0, ug, color=c, label=r"$\tilde\nu=%.2f$" % nu_t)
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlim(0, 1.5); ax[1].set_ylim(-7, 16)
ax[1].set_xlabel(r"imposed load per head $\bar p/\tau_{c,0}$")
ax[1].set_ylabel(r"$u$  ($\mu$m/s)")
ax[1].set_title(r"predicted steady states $\bar p=\tau_{ss}(u)-\tilde\nu u$")
ax[1].legend()

# (c) soft-device ramps
cols = {30: "C4", 100: "C0", 300: "C2"}
for f in sorted(glob.glob("ramp_*.npz")):
    r = np.load(f)
    N, nu_t = int(r["N"]), float(r["nu_t"])
    t, xc, p = r["t"], r["xc"], r["p"]
    w = 100                                  # window of 100 records = 5 ms
    uu = -(xc[w:] - xc[:-w]) / (t[w:] - t[:-w])
    pm = 0.5 * (p[w:] + p[:-w])
    up = (t[:-w] > 60.0) & (t[:-w] < 360.0)
    dn = t[:-w] >= 360.0
    ls = "-" if nu_t < 0.2 else ":"
    ax[2].plot(pm[up] / tau0, uu[up], ls, lw=0.8, color=cols[N], alpha=0.9, label="N=%d  nu=%.2f  up" % (N, nu_t))
    ax[2].plot(pm[dn] / tau0, uu[dn], ls, lw=0.8, color=cols[N], alpha=0.45)
ax[2].plot((tg - 0.05 * ug) / tau0, ug, "k--", lw=1.2, label=r"clamp prediction, $\tilde\nu=0.05$")
ax[2].set_xlim(0.2, 1.35)
ax[2].set_xlabel(r"imposed load per head $\bar p/\tau_{c,0}$")
ax[2].set_ylabel(r"$u$ (5 ms means)")
ax[2].set_title("soft-device ramps (dark: up, light: down)")
ax[2].legend(fontsize=7)
ax[2].axhline(0, color="k", lw=0.5)
plt.tight_layout()
plt.savefig("S_curve.png", dpi=130)
print("tau0 =", tau0)
