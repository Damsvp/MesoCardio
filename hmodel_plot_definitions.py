# %% Graphs of the potentials w0, w1, of the forces, and of the rates used in the simulations (self-contained, same parameters
#    as hmodel_standalone.py / hmodel_threshold_macro.py)
import numpy as np, matplotlib.pyplot as plt
D, KBT, K0, K1 = 40.0, 4.1, 0.2, 0.5                 # d [nm], kBT [zJ], k0, k1 [pN/nm]
H_THR, K_ATT, LAM_DET = 5.0, 0.10, 0.10              # threshold [nm], attachment rate above it [ms^-1] (np.inf: threshold alone), detachment rate [ms^-1]

x = np.linspace(-30, 30, 601); h = np.linspace(-D/2, D/2, 1001)
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
ax[0].plot(x, K0*x**2/2, label=r"$w_0=k_0x^2/2$ (detached)"); ax[0].plot(x, K1*x**2/2, label=r"$w_1=k_1x^2/2$ (attached)")
ax[0].axhline(KBT, color="gray", lw=0.5, ls=":"); ax[0].set(xlabel="x (nm)", ylabel="energy (zJ)", ylim=(0, 10*KBT)); ax[0].legend()
ax[1].plot(x, -K0*x, label=r"$-w_0'$"); ax[1].plot(x, -K1*x, label=r"$-w_1'$ (force of an attached head)"); ax[1].axhline(0, color="k", lw=0.4)
ax[1].set(xlabel="x (nm)", ylabel="force (pN)"); ax[1].legend()
ax[2].plot(h, K_ATT*(h > H_THR) if np.isfinite(K_ATT) else 1.0*(h > H_THR)); ax[2].axvline(H_THR, color="r", lw=0.7)
ax[2].set(xlabel="h (nm)", ylabel=r"$K_{0\to1}(h)$ (ms$^{-1}$)" if np.isfinite(K_ATT) else r"$K_{0\to1}(h)/K_{\rm att}\ (K_{\rm att}\to\infty)$", title="attachment: step at $h_*$")
ax[3].plot(h, (LAM_DET/(D/2))*(h < 0)); ax[3].set(xlabel="landing mark h (nm)", ylabel=r"$K_{1\to0}(h)$ (ms$^{-1}$nm$^{-1}$)", title="detachment: uniform on $(-d/2,0)$")
fig.tight_layout(); plt.show()
