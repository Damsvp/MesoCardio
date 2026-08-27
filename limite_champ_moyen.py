#%% Limite de champ moyen (N -> infini) du modele jump-diffusion
#
# On simule le systeme auto-coherent (u_t(x,y), v_t(y), s_t) obtenu dans
# "Limite champ moyen/limite_champ_moyen.tex" (eqs. (eq:u), (eq:v), (eq:closure)) :
#
#   d_t u = eta_x^-1 d_x(u d_x w0) + eta_x^-1 kBT d_x^2 u
#         + eta_y^-1 d_y(u d_y w0) + eta_y^-1 kBT d_y^2 u
#         - Lon(x,y,s) u(x,y)  +  1_[-d/2,d/2](x) Loff(x,y,s) v(y)
#
#   d_t v = eta_y^-1 d_y(v d_y w1(s,y)) + eta_y^-1 kBT d_y^2 v
#         + Lon(s,y,s) u(s,y)  -  (int_{-d/2}^{d/2} Loff(x',y,s) dx') v(y)
#
#   ds/dt = (1/nu1) (fbar - zbar_t),   zbar_t = int kappa(s+y) v(y) dy
#
# u(x,y) est la densite jointe (position X, coordonnee interne Y) des tetes
# DETACHEES (alpha=0), v(y) est la densite (en Y seul, puisque X = s_t p.s.)
# des tetes ATTACHEES (alpha=1). u,v et s sont couples de facon non lineaire
# (McKean-Vlasov) : les taux de saut dependent de s_t, et s_t depend de v_t.

import numpy as np
from matplotlib import pyplot as plt
import time as _time

#%% Parametres physiques (repris de n_tetes_jumpdiffusion.py)

d = 38  # distance entre deux sites d'attachement de l'actine (nm)

k1pre = 5.6
k1post = 1.4
l1 = 1.42
y1pre = 0
y1post = 11
v1 = (k1post/2)*(l1 - y1post)**2 - (k1pre/2)*(l1 - y1pre)**2  # decalage d'energie (puits 1)

k0pre = 11.45
k0post = 0.45
l0 = 1.42
y0pre = 0
y0post = 6
v0 = (k0post/2)*(l0 - y0post)**2 - (k0pre/2)*(l0 - y0pre)**2  # decalage d'energie (puits 0)

def double_well(y, kpre, kpost, lsep, ypre, ypost, v):
    return np.where(y < lsep, (kpre/2)*(y - ypre)**2 + v, (kpost/2)*(y - ypost)**2)

def d_double_well(y, kpre, kpost, lsep, ypre, ypost, v):
    return np.where(y < lsep, kpre*(y - ypre), kpost*(y - ypost))

kBTemp = 4.14           # kB*T en zJ (T = 300K)
b = 1/kBTemp
ny = 10.288             # eta_y^-1 (fluidite), ms-1.pN-1.nm
nx = 10.288             # eta_x^-1

k = 1.34                # raideur myosine (couplage x-y)

kmax = 1.21
alphay = 8
alphas = 8
sl01 = 3.82
sr01 = 3.82

l = 10                    # largeur de la fenetre d'attachement/detachement (nm)

k10 = 1
k01 = 1

K01 = lambda x, y, s: k01*np.heaviside(l0 - y, 0.5)*np.heaviside(l/2 - np.abs(x - s), 0.5)
# Le terme de bord ci-dessous ne depend pas de s (contrairement au reste de K10) : dans le
# modele a une tete/N tetes il n'etait jamais evalue qu'en x=s (borne physiquement par la
# dynamique de Y), mais sur une grille (x,y) pleine, combine a un grand y, il diverge sans
# rapport avec la physique du modele (loin de x=s). On le plafonne pour eviter cet artefact
# numerique tout en laissant inchangee la dynamique pres des puits (cf. tests de stabilite).
K10_BOUNDARY_CAP = 25.0
K10 = lambda x, y, s: (k10*np.heaviside(l/2 - np.abs(x - s), 0.5)*np.heaviside(y - l0, 0.5)
                       + np.minimum((1/np.maximum(d/2 - x, 1e-10))**2, K10_BOUNDARY_CAP))

E = 80
sbar0 = 1.2
sbar1 = 1.2

w0 = lambda x, y: E + (k/2)*(x + y)**2 + double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0)
dxw0 = lambda x, y: k*(x + y)
dyw0 = lambda x, y: k*(x + y) + d_double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0)

w1 = lambda x, y: (k/2)*(x + y)**2 + double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1)
dxw1 = lambda x, y: k*(x + y)
dyw1 = lambda x, y: k*(x + y) + d_double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1)

muT = 100
h = 11

EXP_CAP = 50.0  # sature l'exponentielle loin des puits (densite y negligeable) pour eviter tout overflow
K01rev = lambda x, y, s: (1/h)*K01(x, y, s)*np.exp(np.clip(b*(w1(s, y) - w0(x, y)), -EXP_CAP, EXP_CAP))
K10rev = lambda x, y, s: h*K10(x, y, s)*np.exp(np.clip(b*((w0(x, y) - muT) - w1(s, y)), -EXP_CAP, EXP_CAP))

Lon = lambda x, y, s: K01(x, y, s) + K10rev(x, y, s)
Loff = lambda x, y, s: K10(x, y, s) + K01rev(x, y, s)

kappa = lambda u: k*u  # kappa(X+Y) = k*(X+Y), force d'une tete attachee

# --- echelle de champ moyen : v = N*nu1, F = N*fbar (cf section 1 du document) ---
N_ref = 15      # N utilise dans le modele a N tetes, pour fixer l'echelle
v_orig = 2      # v (viscosite totale) du modele a N tetes
F_orig = -1.5      # F (force imposee totale) du modele a N tetes
nu1 = v_orig/N_ref   # viscosite PAR tete (notee v_1 dans le document)
fbar = F_orig/N_ref  # force imposee PAR tete (notee f dans le document)

#%% Grille (x,y) pour u ; grille y pour v

Nx = 80
Ny = 60

xmin, xmax = -d/2, d/2              # x est bien defini sur [-d/2, d/2] (cf. eq (eq:u)-(eq:v))
ymin, ymax = -8.0, 22.0              # couvre largement les deux puits (0/6/11) +/- bruit thermique

dx = (xmax - xmin)/Nx
dy = (ymax - ymin)/Ny
x_grid = xmin + (np.arange(Nx) + 0.5)*dx      # grille centree : jamais exactement sur le bord
y_grid = ymin + (np.arange(Ny) + 0.5)*dy      # (evite la singularite de K10 en x = d/2)

X2, Y2 = np.meshgrid(x_grid, y_grid, indexing='ij')   # shape (Nx, Ny)

# w0 ne depend pas de s : les derivees aux interfaces peuvent etre precalculees une seule fois
x_mid = 0.5*(X2[:-1, :] + X2[1:, :])
y_at_xmid = Y2[:-1, :]
dxw0_mid = dxw0(x_mid, y_at_xmid)          # (Nx-1, Ny)

y_mid = 0.5*(Y2[:, :-1] + Y2[:, 1:])
x_at_ymid = X2[:, :-1]
dyw0_mid = dyw0(x_at_ymid, y_mid)          # (Nx, Ny-1)

#%% Operateurs de diffusion-advection (upwind pour l'advection, centre pour la diffusion)

def flux_upwind_diffusion(field, dw_mid, mu, kBT, delta, axis):
    """Flux conservatif a travers les interfaces le long de `axis` (0 ou 1)."""
    if axis == 0:
        f_lo, f_hi = field[:-1, :], field[1:, :]
    else:
        f_lo, f_hi = field[:, :-1], field[:, 1:]
    f_upwind = np.where(dw_mid > 0, f_hi, f_lo)   # vitesse = -mu*dw ; upwind selon son signe
    d_field = (f_hi - f_lo) / delta
    return mu*(dw_mid*f_upwind + kBT*d_field)

def divergence(flux_internal, axis, delta, shape):
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)                 # flux nul au bord (condition reflechissante, pas de fuite)
    flux_full = np.pad(flux_internal, pad)
    return np.diff(flux_full, axis=axis) / delta

def diffusion_u(u):
    Jx = flux_upwind_diffusion(u, dxw0_mid, nx, kBTemp, dx, axis=0)
    Jy = flux_upwind_diffusion(u, dyw0_mid, ny, kBTemp, dy, axis=1)
    return divergence(Jx, 0, dx, u.shape) + divergence(Jy, 1, dy, u.shape)

def diffusion_v(v, s):
    dyw1_mid = dyw1(s, 0.5*(y_grid[:-1] + y_grid[1:]))     # (Ny-1,) ; depend de s_t
    v_lo, v_hi = v[:-1], v[1:]
    v_upwind = np.where(dyw1_mid > 0, v_hi, v_lo)
    dv = (v_hi - v_lo)/dy
    Jy = ny*(dyw1_mid*v_upwind + kBTemp*dv)
    Jy_full = np.pad(Jy, (1, 1))
    return np.diff(Jy_full)/dy


#%% Pas de temps (stabilite explicite : diffusion + advection upwind)

D_x_eff = nx*kBTemp
D_y_eff = ny*kBTemp
vmax_x = np.max(np.abs(nx*dxw0_mid))
vmax_y = np.max(np.abs(ny*dyw0_mid))

dt_diff = 1.0/(2*D_x_eff/dx**2 + 2*D_y_eff/dy**2)
dt_adv = 1.0/(vmax_x/dx + vmax_y/dy + 1e-12)
safety = 0.4
dt = safety*min(dt_diff, dt_adv)

T = 50.0            # duree simulee (ms) -- augmenter si besoin, au prix du temps de calcul (~12 s/ms)
nsteps = int(T/dt)
print(f"dx={dx:.3f} nm, dy={dy:.3f} nm, dt={dt:.3e} ms, nsteps={nsteps}")

#%% Condition initiale : population majoritairement detachee, au repos

p0_init = 0.9   # fraction initiale de tetes detachees
# NB : grille centree sur les cellules -> l'integrale coherente avec le schema aux
# volumes finis est la somme de Riemann dx*dy*sum(.), pas np.trapz (regle des trapezes,
# adaptee a une grille incluant les bords, pas a notre grille centree).
u = p0_init*np.exp(-(X2**2)/(2*3**2) - (Y2**2)/(2*3**2))
u /= (dx*dy*np.sum(u))
u *= p0_init

v = (1 - p0_init)*np.exp(-((y_grid - (y1post - sbar1))**2)/(2*3**2))
v /= (dy*np.sum(v))
v *= (1 - p0_init)

s = 0.0

#%% Boucle de simulation

time_hist = dt*np.arange(nsteps)
s_hist = np.empty(nsteps)
p0_hist = np.empty(nsteps)      # masse totale des tetes detachees = iint u
p1_hist = np.empty(nsteps)      # masse totale des tetes attachees = int v
zbar_hist = np.empty(nsteps)    # force moyenne par tete, zbar_t
period = []                     # instants ou s_t franchit -d/2 (avancee d'un pas d)

snapshots = {}
snap_times = [0, T/4, T/2, 3*T/4, T - dt]
snap_idx = 0

t0 = _time.time()
for t in range(nsteps):
    s_hist[t] = s
    p0_hist[t] = dx*dy*np.sum(u)
    p1_hist[t] = dy*np.sum(v)

    mass_v = p1_hist[t]
    zbar = kappa(s*mass_v + dy*np.sum(y_grid*v))
    zbar_hist[t] = zbar

    if snap_idx < len(snap_times) and time_hist[t] >= snap_times[snap_idx]:
        snapshots[round(time_hist[t], 3)] = (u.copy(), v.copy(), s)
        snap_idx += 1

    # --- sous-pas de diffusion-advection (explicite, dt choisi pour cette partie) ---
    u_diff = u + dt*diffusion_u(u)
    v_diff = v + dt*diffusion_v(v, s)

    # --- sous-pas de reaction (saut alpha=0 <-> alpha=1), traitement exact ---
    # Les taux Lon/Loff peuvent devenir tres grands (facteurs de Boltzmann pilotes par
    # muT). On calcule donc la masse EXACTEMENT transferee par chaque canal pendant dt
    # (bornee par la masse disponible, via 1-exp(-taux dt) qui sature a 1), plutot que
    # taux*densite*dt qui diverge si le taux explose : le schema reste stable quelle
    # que soit l'amplitude des taux, et conserve exactement la masse a chaque sous-pas.
    Lon_field = Lon(X2, Y2, s)
    Loff_field = Loff(X2, Y2, s)
    loss_rate_v = dx*np.sum(Loff_field, axis=0)             # int Loff(x',y,s) dx'  -> (Ny,)
    # NB : le document ecrit le gain de v comme l'evaluation ponctuelle Lon(s,y,s)*u(s,y).
    # Cette forme ne conserve pas la masse avec le terme de perte -Lon(x,y,s)*u(x,y)
    # (integre sur tout x) : verifie numeriquement, ~30% de masse perdue en 1 ms. La
    # forme integree utilisee ci-dessous restaure la conservation de masse annoncee par
    # le document, et est symetrique du terme int Loff(x',y,s)dx' deja present dans la
    # perte de v.

    decay_u = np.exp(-Lon_field*dt)                        # (Nx,Ny), dans (0,1]
    mass_u_to_v = u_diff*(1 - decay_u)                      # masse exacte quittant u, bornee par u_diff
    gain_into_v = dx*np.sum(mass_u_to_v, axis=0)            # -> (Ny,), masse totale recue par v

    decay_v = np.exp(-loss_rate_v*dt)                       # (Ny,), dans (0,1]
    mass_v_to_u = v_diff*(1 - decay_v)                      # masse exacte quittant v, bornee par v_diff
    profile_x = Loff_field/np.where(loss_rate_v > 1e-300, loss_rate_v, 1.0)[None, :]  # noyau normalise (integre a 1 en x)
    gain_into_u = profile_x*mass_v_to_u[None, :]            # redistribution exacte de la masse quittant v

    u = u_diff*decay_u + gain_into_u
    v = v_diff*decay_v + gain_into_v

    ds = (1/nu1)*(fbar - zbar)
    s_new = s + dt*ds
    if s_new < -d/2:
        s_new += d
        period.append(t*dt)
    elif s_new > d/2:
        s_new -= d
    s = s_new

print(f"Simulation terminee en {_time.time() - t0:.1f} s")
print(f"Masse totale finale (iint u + int v) = {p0_hist[-1] + p1_hist[-1]:.4f}  (doit rester proche de 1)")

if len(period) > 1:
    speeds = d/np.diff(period)
    print("Vitesse moyenne de glissement (champ moyen) :", np.mean(speeds), "nm/ms")
else:
    print("Pas assez de cycles pour estimer une vitesse moyenne sur cette duree T.")

#%% Visualisation : s_t, fractions detachee/attachee, force moyenne

fig, axs = plt.subplots(nrows=3, figsize=(14, 12), sharex=True)

axs[0].plot(time_hist, s_hist, lw=0.8)
axs[0].set_ylabel(r"$s_t$ (nm)")
axs[0].set_title(r"Trajectoire de $s_t$ (limite de champ moyen)")

axs[1].plot(time_hist, p0_hist, label=r"$\iint u_t$ (detachees)")
axs[1].plot(time_hist, p1_hist, label=r"$\int v_t$ (attachees)")
axs[1].plot(time_hist, p0_hist + p1_hist, '--', color='black', lw=1, label='masse totale')
axs[1].set_ylabel("fraction de tetes")
axs[1].set_title("Fractions detachee / attachee")
axs[1].legend()

axs[2].plot(time_hist, zbar_hist, color='tab:red', lw=0.8)
axs[2].set_xlabel("time (ms)")
axs[2].set_ylabel(r"$\bar z_t$ (pN)")
axs[2].set_title(r"Force moyenne par tete $\bar z_t = \mathbb{E}[\alpha_t \kappa(X_t+Y_t)]$")

plt.tight_layout()
plt.show()

#%% Visualisation : instantanes de u(x,y) et v(y)

fig, axs = plt.subplots(nrows=2, ncols=len(snapshots), figsize=(4*len(snapshots), 8))
for col, (tt, (u_snap, v_snap, s_snap)) in enumerate(snapshots.items()):
    ax_u = axs[0, col] if len(snapshots) > 1 else axs[0]
    im = ax_u.pcolormesh(x_grid, y_grid, u_snap.T, shading='auto')
    ax_u.axvline(s_snap, color='white', ls='--', lw=1)
    ax_u.set_title(f"u(x,y), t={tt} ms")
    ax_u.set_xlabel("x (nm)")
    ax_u.set_ylabel("y (nm)")
    plt.colorbar(im, ax=ax_u)

    ax_v = axs[1, col] if len(snapshots) > 1 else axs[1]
    ax_v.plot(y_grid, v_snap)
    ax_v.set_title(f"v(y), t={tt} ms")
    ax_v.set_xlabel("y (nm)")
    ax_v.set_ylabel("v")

plt.tight_layout()
plt.show()

#%% Visualisation : profil de w0(x,y) et w1(x,y) pour x negatif/nul/positif

y_plot = np.linspace(ymin, ymax, 400)
x_vals = [xmin/2, 0.0, xmax/2]   # x negatif, nul, positif

fig, axs = plt.subplots(ncols=2, figsize=(14, 5), sharey=True)

for x_val in x_vals:
    axs[0].plot(y_plot, w0(x_val, y_plot), label=f"x = {x_val:.1f} nm")
    axs[1].plot(y_plot, w1(x_val, y_plot), label=f"x = {x_val:.1f} nm")

axs[0].set_title(r"$w_0(x,y)$ (etat detache)")
axs[0].set_xlabel("y (nm)")
axs[0].set_ylabel("energie (zJ)")
axs[0].legend()

axs[1].set_title(r"$w_1(x,y)$ (etat attache)")
axs[1].set_xlabel("y (nm)")
axs[1].legend()

plt.tight_layout()
plt.show()

# %%
