"""Soft device with a spring in series (mixed device), far end fixed or driven at speed V.

Load per head:  pbar = pbar0 + kL*V*t - kL*x_c      (kL = spring stiffness per head, pN/nm)

usage:  python osc_run.py [N] [nu_t] [pbar0] [kL] [T] [tag] [V]
Every argument is optional. Defaults reproduce the strongest stick-slip run of the report:
        N=100  nu_t=0.05  pbar0=0.50  kL=0.10  T=200  tag=N100_k0.10_V2  V=2.0
Units: nu_t in pN ms/nm, pbar0 in pN, kL in pN/nm, T in ms, V in nm/ms (= um/s).
Output: osc_<tag>.npz
"""
import sys, time
import numpy as np
import hmodel_soft as H

def arg(i, default, cast=float):
    return cast(sys.argv[i]) if len(sys.argv) > i else default

N    = arg(1, 100, int)
nu_t = arg(2, 0.05)
p0   = arg(3, 0.50)
kL   = arg(4, 0.10)
T    = arg(5, 200.0)
tag  = arg(6, "N100_k0.10_V2", str)
V    = arg(7, 2.0)

m = H.HModel()
rng = np.random.default_rng(11)
t0 = time.time()
o = H.run(m, N=N, T=T, dt=0.002, mode="soft", nu=N * nu_t,
          Fext_fn=lambda t, xc: -N * (p0 + kL * V * t - kL * xc),
          rng=rng, record_every=25)
np.savez(f"osc_{tag}.npz", t=o["t"], xc=o["xc"], vel=o["vel"], tau=o["tau"], na=o["na"],
         N=N, nu_t=nu_t, p0=p0, kL=kL, V=V)
print("done", tag, "%.0fs" % (time.time() - t0), flush=True)
