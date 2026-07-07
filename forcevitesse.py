

import numpy as np
from matplotlib import pyplot as plt
import random

d = 38 #distance between two actin attachment sites
N = 16
v = 2

k1pre = 5.6 #stiffness pre power stroke (pN/nm)
k1post = 1.4 #stiffness post power stroke
l1 = 1.42 #separation between the two wells (nm)
y1pre = 0 #position of the minimum of the pre power stroke well
y1post = 11 #position of the minimum of the post power stroke well
v1 = (k1post/2)*(l1 - y1post)**2 - (k1pre/2)*(l1 - y1pre)**2 #energy difference between the two states at the position of the minimum of the post power stroke well

k0pre = 11.45 #stiffness pre power stroke
k0post = 0.45 #stiffness post power stroke
l0 = 1.42 #separation between the two wells (nm)
y0pre = 0 #position of the minimum of the pre power stroke well
y0post = 6 #position of the minimum of the post power stroke well
v0 = (k0post/2)*(l0 - y0post)**2 - (k0pre/2)*(l0 - y0pre)**2 #energy difference between the two states at the position of the minimum of the post power stroke well

l = 5 #length of the interval in which the myosin head can detach
dx = 0.01 #length discretization
npos = int(l/dx) #number of possible positions
positions = [k*dx - l/2 for k in range(npos)]   #all possible positions


kBTemp = 4.14 #kB*temperature in zJ (T = 300K)
b = 1/kBTemp #inverse temperature
ny = 10.288 #inverse viscosity coefficients (fluidity) (ms-1.pN-1.nm)
nx = 10.288

T = 3000  #max time of the simulation, in ms
dt = 0.01    #time step (ms)
nsteps = int(T/dt)    #number of steps in the simulation

kmax = 1.21 
alphay = 8
alphas = 8
sl01 = 3.82
sr01 = 3.82

k = 1.34 #stiffness of the myosin

k10 = 1 #transition rates
k01 = 1

K01 = lambda x, y, s : k01*np.heaviside(l0 - y,0.5)*np.heaviside(l/2 - np.abs(x - s),0.5) #0.1 + kmax*(1 - np.tanh(alphay*(y - l0)))*(0.5*(1 - heaviside(s))*(1 + np.tanh(alphas*(s + sl01))) + 0.5*heaviside(s)*(1 - np.tanh(alphas*(s - sr01))))   #direct transition rates
K10 = lambda x, y, s : k10*np.heaviside(l/2 - np.abs(x - s),0.5)*np.heaviside(y - l0,0.5)+(1/np.maximum(d/2 - x, 1e-10))**2#
#note that K10 is supposed to vanish outside of [-l/2, l/2]

E = 80 #energy shift in zJ
sbar0 = 1.2 #shift of the potential, in nm
sbar1 = 1.2

muT = 100   #shift due to ATP consumption, in zJ

h = 11       #caracteristic length scale (nm) used to make the expression of reverse transition rates homogeneous
#Modifier à terme pour que l'intégrale fasse 1 je crois NONONONONON
#c'est un taux de transition, pas une probabilité

K01rev = lambda x, y, s : (1/h)*K01(x, y, s)*np.exp(b*(w1(s, y) - w0(x, y)))     #reverse transition rates
K10rev = lambda x, y, s : h*K10(x, y, s)*np.exp(b*((w0(x, y) - muT) - w1(s, y)))

# --------- Vectorised function --------------------
def double_well(y, kpre, kpost, lsep, ypre, ypost, v):
    return np.where(
        y < lsep,
        (kpre/2) * (y - ypre)**2 + v,
        (kpost/2) * (y - ypost)**2
        )

def d_double_well(y, kpre, kpost, lsep, ypre, ypost, v):
    return np.where(
        y < lsep,
        kpre * (y - ypre),
        kpost * (y - ypost)
        )

w0 = lambda x, y : E + ((k/2)*(x + y)**2 + double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0))     #energy landscape and derivatives for detached head
dxw0 = lambda x, y : k*(x + y)
dyw0 = lambda x, y : k*(x + y) + d_double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0)

w1 = lambda x, y : ((k/2)*(x + y)**2 + double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1))     #energy landscape and derivatives for attached head
dxw1 = lambda x, y : k*(x + y)
dyw1 = lambda x, y : k*(x + y) + d_double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1)

vitesses = []
s0    = d * (np.random.random(size=N) - 0.5)

nforces = 30

for f in range(nforces):

    F = 0.05*f #force exerted on the actin filament

    # ── Pré-calculs hors boucle ──────────────────────────────────────────
    pos = np.arange(npos) * dx                        # (npos,)  — sorti de la boucle

    alpha = np.zeros((N, nsteps), dtype=np.int8)
    s     = np.empty((N, nsteps))
    s[:, 0] = s0
    X     = np.empty((N, nsteps))
    X[:, 0] = s0
    Y     = np.zeros((N, nsteps))

    # Horloges persistantes entre timesteps
    c01    = np.zeros(N);  e01    = -np.log(np.random.random(size=N))
    c10    = np.zeros(N);  e10    = -np.log(np.random.random(size=N))
    c01rev = np.zeros(N);  e01rev = -np.log(np.random.random(size=N))
    c10rev = np.zeros(N);  e10rev = -np.log(np.random.random(size=N))

    speeds = []

    for t in range(nsteps - 1):
        at  = alpha[:, t]          # (N,)
        Xt  = X[:, t]
        Yt  = Y[:, t]
        st  = s[:, t]

        # ── Dynamique de s (vectorisée) ───────────────────────────────────
        # dxw1 doit accepter des tableaux → vérifier/vectoriser la fonction
        delta_s = F*dt/v - (dt/v) * np.sum(at * dxw1(st, Yt))
        # (si dxw1 est scalaire, utiliser np.vectorize ou la réécrire)

        st1 = st + delta_s
        wrap_minus = st1 < -d / 2
        wrap_plus  = st1 >  d / 2
        st1[wrap_minus] += d
        st1[wrap_plus]  -= d
        s[:, t + 1] = st1
        if not(wrap_minus[0] or wrap_plus[0]) and t > 0:                 # équivalent au if i==0 d'origine
            speeds.append(-delta_s/dt)
        

        # ── Bruits browniens vectorisés ───────────────────────────────────
        Bx = np.random.randn(N)
        By = np.random.randn(N)

        # ── Calcul vectorisé des taux (doit accepter des arrays) ──────────
        K01t    = K01(Xt, Yt, st)       # (N,)
        K10t    = K10(Xt, Yt, st)
        K01revt = K01rev(Xt, Yt, st)
        K10revt = K10rev(Xt, Yt, st)

        # ── Masques α=0 / α=1 ────────────────────────────────────────────
        m0 = (at == 0)    # (N,) bool
        m1 = (at == 1)

        # ─── Bloc α = 0 ───────────────────────────────────────────────────
        sig0 = np.sqrt(2 * ny * dt / b)
        Y[m0, t + 1] = (Yt[m0]
                        - dt * ny * dyw0(Xt[m0], Yt[m0])
                        + sig0 * By[m0])

        c01[m0]    += K01t[m0]    * dt
        c10rev[m0] += K10revt[m0] * dt

        jump0 = m0 & ((c01 > e01) | (c10rev > e10rev))
        stay0 = m0 & ~jump0

        # Particules qui sautent → α = 1
        alpha[jump0, t + 1] = 1
        X[jump0, t + 1]     = st1[jump0]
        # reset horloges
        n_j0 = jump0.sum()
        c01[jump0]    = 0.0;  e01[jump0]    = -np.log(np.random.random(size=n_j0))
        c10rev[jump0] = 0.0;  e10rev[jump0] = -np.log(np.random.random(size=n_j0))

        # Particules qui restent → α = 0
        alpha[stay0, t + 1] = 0
        X[stay0, t + 1] = (Xt[stay0]
                        - dt * nx * dxw0(Xt[stay0], Yt[stay0])
                        + np.sqrt(2 * nx * dt / b) * Bx[stay0])

        # ─── Bloc α = 1 ───────────────────────────────────────────────────
        sig1 = np.sqrt(2 * ny * dt / b)
        Y[m1, t + 1] = (Yt[m1]
                        - dt * ny * dyw1(Xt[m1], Yt[m1])
                        + sig1 * By[m1])

        c01rev[m1] += K01revt[m1] * dt
        c10[m1]    += K10t[m1]    * dt

        jump1 = m1 & ((c01rev > e01rev) | (c10 > e10))
        stay1 = m1 & ~jump1

        # Particules qui détachent → α = 0, position tirée selon prob
        # np.random.choice avec p= ne se vectorise pas directement :
        # on calcule les proba pour toutes les particules m1 et on échantillonne
        if jump1.any():
            idx1  = np.where(jump1)[0]
            # prob shape: (|jump1|, npos)
            prob_mat = np.array([
                [( K10(np.array([pos[j]]), Yt[i:i+1], st[i:i+1])
                + K01rev(np.array([pos[j]]), Yt[i:i+1], st[i:i+1]))[0] * dx
                for j in range(npos)]
                for i in idx1
            ])
            row_sums = prob_mat.sum(axis=1, keepdims=True)
            prob_mat /= row_sums
            chosen = np.array([
                np.random.choice(pos, p=prob_mat[k])
                for k in range(len(idx1))
            ])
            alpha[jump1, t + 1] = 0
            X[jump1, t + 1]     = st1[jump1] + chosen
            n_j1 = jump1.sum()
            c01rev[jump1] = 0.0; e01rev[jump1] = -np.log(np.random.random(size=n_j1))
            c10[jump1]    = 0.0; e10[jump1]    = -np.log(np.random.random(size=n_j1))

        # Particules qui restent attachées → α = 1
        alpha[stay1, t + 1] = 1
        X[stay1, t + 1]     = st1[stay1]

    # %%  Visualization of the results
    # fig, axs = plt.subplots(nrows=3, figsize=(18,18))

    # axs[0].plot([dt*t for t in range(nsteps)], X[0, :],label='X')
    # axs[0].plot([dt*t for t in range(nsteps)], s[0,:],label='s')
    # axs[0].set_xlabel("time")
    # axs[0].set_ylabel("X_t")
    # axs[0].legend()
    # axs[0].set_title("X over time")

    # axs[1].plot([dt*t for t in range(nsteps)], Y[0, :],label='Y')
    # axs[1].set_xlabel("time")
    # axs[1].set_ylabel("Y_t")
    # axs[1].set_title("Y over time")

    # axs[2].plot([dt*t for t in range(nsteps)], alpha[0, :],label='alpha')
    # axs[2].set_xlabel("time")
    # axs[2].set_ylabel(r"$\alpha$")
    # axs[2].set_title(r"$\alpha$ over time")

    # plt.show()

    mean_speed=np.mean(speeds)
    print(mean_speed)
    vitesses.append(mean_speed)

print(vitesses)

plt.plot([0.05*f for f in range(nforces)], vitesses)
plt.show()

