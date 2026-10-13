import numpy as np, os, time
import hmodel_soft as H
m = H.HModel()
vs = np.array([+6,+3,+1.5,+0.7,+0.3,+0.1, 0.0, -0.1,-0.2,-0.35,-0.5,-0.75,-1,-1.5,-2,-3,-4,-5,-7,-10,-15], float)
res = list(np.load("clamp_sweep.npy")) if os.path.exists("clamp_sweep.npy") else []
done = {round(r[0],6) for r in res}
for i,v in enumerate(vs):
    if round(v,6) in done: continue
    t0=time.time()
    o = H.run(m, N=400, T=60.0, dt=0.002, mode="clamp", v=v, rng=np.random.default_rng(100+i), burn=15.0, record_every=1000)
    res.append((v,o["tau_mean"],o["na_mean"]))
    print("v=%6.2f  tau=%.4f  na=%.4f  (%.0fs)"%(v,o["tau_mean"],o["na_mean"],time.time()-t0), flush=True)
    np.save("clamp_sweep.npy", np.array(res))
print("DONE", flush=True)
