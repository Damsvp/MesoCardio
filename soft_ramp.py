"""Soft-device force ramp (up then down) of the h-model.   usage: python soft_ramp.py N nu_tilde [tag]"""
import sys, time
import numpy as np
import hmodel_soft as H

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
nu_t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05      # pN ms / nm per head
tag = sys.argv[3] if len(sys.argv) > 3 else f"N{N}_nu{nu_t}"
tau0 = 0.88                                                  # isometric force per head (pN), from velocity clamp
p_lo, p_hi = 0.25 * tau0, 1.30 * tau0
T_hold, T_ramp = 60.0, 300.0
dt = 0.002

def pbar(t):                                                 # load per head, opposing contraction
    if t < T_hold:
        return p_lo
    s = t - T_hold
    if s < T_ramp:
        return p_lo + (p_hi - p_lo) * s / T_ramp
    s -= T_ramp
    if s < T_ramp:
        return p_hi - (p_hi - p_lo) * s / T_ramp
    return p_lo

m = H.HModel()
rng = np.random.default_rng(7)
t0 = time.time()
o = H.run(m, N=N, T=T_hold + 2 * T_ramp, dt=dt, mode="soft", nu=N * nu_t,
          Fext_fn=lambda t: -N * pbar(t), rng=rng, record_every=25)
t = o["t"]
np.savez(f"ramp_{tag}.npz", t=t, xc=o["xc"], vel=o["vel"], tau=o["tau"], na=o["na"],
         p=np.array([pbar(x) for x in t]), N=N, nu_t=nu_t, tau0=tau0)
print("done", tag, "time %.0fs" % (time.time() - t0))
