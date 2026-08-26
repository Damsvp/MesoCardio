#%%
import numpy as np
from matplotlib import pyplot as plt
import random



N = 15 #number of heads we are going to simulate
delta = [0 for i in range(N)] #random position shifts for the heads
v = 2 #viscosity of the sarcomere's surroundings
F = 0 #force exerted on the actin filament
d = 38 #distance between two actin attachment sites

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


def double_well(y, kpre, kpost, lsep, ypre, ypost, v) :
  if y < lsep :
    return (kpre/2)*(y - ypre)**2 + v
  else :
    return (kpost/2)*(y - ypost)**2

def d_double_well(y, kpre, kpost, lsep, ypre, ypost, v) :
  if y < lsep :
    return kpre*(y - ypre)
  else :
    return kpost*(y - ypost)


l = 5 #length of the interval in which the myosin head can detach
dx = 0.01 #length discretization
npos = int(l/dx) #number of possible positions
positions = [k*dx - l/2 for k in range(npos)]   #all possible positions


kBTemp = 4.14 #kB*temperature in zJ (T = 300K)
b = 1/kBTemp #inverse temperature
ny = 10.288 #inverse viscosity coefficients (fluidity) (ms-1.pN-1.nm)
nx = 10.288

T = 400  #max time of the simulation, in ms
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

w0 = lambda x, y : E + ((k/2)*(x + y)**2 + double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0))     #energy landscape and derivatives for detached head
dxw0 = lambda x, y : k*(x + y)
dyw0 = lambda x, y : k*(x + y) + d_double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0)

w1 = lambda x, y : ((k/2)*(x + y)**2 + double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1))     #energy landscape and derivatives for attached head
dxw1 = lambda x, y : k*(x + y)
dyw1 = lambda x, y : k*(x + y) + d_double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1)

muT = 100   #shift due to ATP consumption, in zJ

# plot energy profil

# plt.plot([dx*t - d/4 for t in range(4*npos)], [w0(0, dx*t - d/4) for t in range(4*npos)], label = 'detached')
# plt.plot([dx*t - d/4 for t in range(4*npos)], [w1(0, dx*t - d/4) for t in range(4*npos)], label = 'attached')
# plt.plot([dx*t - d/4 for t in range(4*npos)], [w0(0, dx*t - d/4) - muT for t in range(4*npos)], label = 'detached')
# plt.legend()

# plt.show()




h = 11       #caracteristic length scale (nm) used to make the expression of reverse transition rates homogeneous
#Modifier à terme pour que l'intégrale fasse 1 je crois NONONONONON
#c'est un taux de transition, pas une probabilité

K01rev = lambda x, y, s : (1/h)*K01(x, y, s)*np.exp(b*(w1(s, y) - w0(x, y)))     #reverse transition rates
K10rev = lambda x, y, s : h*K10(x, y, s)*np.exp(b*((w0(x, y) - muT) - w1(s, y)))

#%% Simulation

alpha = np.array([[0 for t in range(nsteps)] for i in range(N)])      #actual description of the stochastic process
s0 = d * (np.random.random(size=N) - 0.5) #Initial Condition
s = np.array([[s0[i] for t in range(nsteps)] for i in range(N)], dtype = float)
X = np.array([[s[i,0] for t in range(nsteps)] for i in range(N)], dtype = float)
Y = np.array([[0 for t in range(nsteps)] for i in range(N)], dtype = float)
period=[]

for t in range(nsteps - 1):
  #possible positions
  pos=np.array([j*dx for j in range(npos)])

  #Clock
  c01 = [0.0 for i in range(N)]
  c10 = [0.0 for i in range(N)]
  e01 = -np.log(np.random.random(size=N))   # realisation of an exponential of parameter 1
  e10 = -np.log(np.random.random(size=N))

  #Clock_rev
  c01rev = [0.0 for i in range(N)]
  c10rev = [0.0 for i in range(N)]
  e01rev = -np.log(np.random.random(size=N))   # realisation of an exponential of parameter 1
  e10rev = -np.log(np.random.random(size=N))

  
  #precompute for s
  delta_s = F*dt/v - (dt/v)*sum([alpha[i, t]*dxw1(s[i,t], Y[i, t]) for i in range(N)])
  for i in range(N) :
    #s dynamics
    s[i,t + 1] = s[i,t] + delta_s
     #torus condition on s
    if s[i,t + 1] < -d/2 :
      s[i,t + 1] += d
      if i==0:  
        period.append(t*dt)
    elif s[i,t + 1] > d/2 :
      s[i,t + 1] -= d
    #generating random numbers to decide if a jump takes place between t and t+dt
    x = random.random()
    y = random.random()
    #X and Y brownian motions
    Bx = random.gauss(0, 1)
    By = random.gauss(0, 1)

      #Compute the rates K and K_rev
    K01t= K01(X[i, t], Y[i, t], s[i,t])
    K10revt= K10rev(X[i, t], Y[i, t], s[i,t])

    K10t= K10(X[i, t], Y[i, t], s[i,t])
    K01revt=K01rev(X[i, t], Y[i, t], s[i,t])
    
    
    
    if alpha[i, t] == 0 :
      #Y dynamics
      Y[i, t + 1] = Y[i, t] - dt*ny*dyw0(X[i, t], Y[i, t]) + np.sqrt(2*ny*dt/b)*By

      #clocks increases:
      c01[i]+= K01t*dt
      c10rev[i]+= K10revt*dt

      #alpha and X dynamics
      if c01[i] > e01[i] or c10rev[i] > e10rev[i]:
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[i,t + 1]
        c01[i]=0.0
        e01[i]=-np.log(np.random.random())
        c10rev[i]=0.0
        e10rev[i]=-np.log(np.random.random())
      else :
        alpha[i, t + 1] = 0
        X[i, t + 1] = X[i, t] - dt*nx*dxw0(X[i, t], Y[i, t]) + np.sqrt(2*nx*dt/b)*Bx

    if alpha[i, t] == 1 :
      #Y dynamics
      #print(t, Y[i, t], X[i, t])
      Y[i, int(t + 1)] = Y[i, t] - dt*ny*dyw1(X[i, t], Y[i, t]) + np.sqrt(2*ny*dt/b)*By

      #clocks increases:
      c01rev[i]+= K01revt*dt
      c10[i]+= K10t*dt

      #alpha and X dynamics
      prob = [(K10(j*dx, Y[i, t], s[i,t]) + K01rev(j*dx, Y[i, t], s[i,t]))*dx for j in range(npos)] #space discretized probabilities of detachment
      detach_rate = sum(prob)
     #calculate the overall detachment rate
      if c01rev[i] > e01rev[i] or c10[i] > e10[i]:   # guard: no detachment possible, skip the jump
        alpha[i, t + 1] = 0
        X[i, t + 1] = s[i,t] +np.random.choice(pos,p= 1/detach_rate*np.array(prob))
        c01rev[i]=0.0
        e01rev[i]=-np.log(np.random.random())
        c10[i]=0.0
        e10[i]=-np.log(np.random.random())
      else :
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[i,t + 1]

#%%  Visualization of the results
fig, axs = plt.subplots(nrows=3, figsize=(18,18))

axs[0].plot([dt*t for t in range(nsteps)], X[0, :],label='X')
axs[0].plot([dt*t for t in range(nsteps)], s[0,:],label='s')
axs[0].set_xlabel("time")
axs[0].set_ylabel("X_t")
axs[0].legend()
axs[0].set_title("X over time")

axs[1].plot([dt*t for t in range(nsteps)], Y[0, :],label='Y')
axs[1].set_xlabel("time")
axs[1].set_ylabel("Y_t")
axs[1].set_title("Y over time")

axs[2].plot([dt*t for t in range(nsteps)], alpha[0, :],label='alpha')
axs[2].set_xlabel("time")
axs[2].set_ylabel(r"$\alpha$")
axs[2].set_title(r"$\alpha$ over time")


plt.show()

#%% Force/Speed

speed=[]
for i in range(len(period)-1):
  speed.append(d/(period[i+1]-period[i]))

mean_speed=np.mean(speed)
print(mean_speed)

#%% Simulation accelerated Claude

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


# ── Pré-calculs hors boucle ──────────────────────────────────────────
pos = np.arange(npos) * dx                        # (npos,)  — sorti de la boucle

alpha = np.zeros((N, nsteps), dtype=np.int8)
s0    = d * (np.random.random(size=N) - 0.5)
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

period = []

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
    if wrap_minus[0]:                 # équivalent au if i==0 d'origine
        period.append(t * dt)
    s[:, t + 1] = st1

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

speed=[]
for i in range(len(period)-1):
  speed.append(d/(period[i+1]-period[i]))

mean_speed=np.mean(speed)
print(mean_speed)

# %% VErsion finale corrigée

# ── Initialisation des horloges hors boucle ──────────────────────────
c01    = np.zeros(N)
c10    = np.zeros(N)
c01rev = np.zeros(N)
c10rev = np.zeros(N)
e01    = -np.log(np.random.random(size=N))
e10    = -np.log(np.random.random(size=N))
e01rev = -np.log(np.random.random(size=N))
e10rev = -np.log(np.random.random(size=N))

for t in range(nsteps - 1):
    at = alpha[:, t]
    Xt = X[:, t]
    Yt = Y[:, t]
    st = s[:, t]

    # ── Dynamique de s ────────────────────────────────────────────────
    delta_s = F*dt/v - (dt/v) * np.sum(at * dxw1(st, Yt))
    st1 = st + delta_s
    wrap_minus = st1 < -d/2
    wrap_plus  = st1 >  d/2
    st1[wrap_minus] += d
    st1[wrap_plus]  -= d
    if wrap_minus[0]:
        period.append(t * dt)
    s[:, t+1] = st1

    # ── Bruits et taux ────────────────────────────────────────────────
    Bx = np.random.randn(N)
    By = np.random.randn(N)

    K01t    = K01(Xt, Yt, st)
    K10t    = K10(Xt, Yt, st)
    K01revt = K01rev(Xt, Yt, st)
    K10revt = K10rev(Xt, Yt, st)

    m0 = (at == 0)
    m1 = (at == 1)

    # ── Accumulation des compteurs (selon l'état actuel) ─────────────
    c01[m0]    += K01t[m0]    * dt
    c10rev[m0] += K10revt[m0] * dt
    c01rev[m1] += K01revt[m1] * dt
    c10[m1]    += K10t[m1]    * dt

    # ─── Bloc α = 0 ───────────────────────────────────────────────────
    Y[m0, t+1] = (Yt[m0]
                  - dt*ny * dyw0(Xt[m0], Yt[m0])
                  + np.sqrt(2*ny*dt/b) * By[m0])

    jump0 = m0 & ((c01 > e01) | (c10rev > e10rev))
    stay0 = m0 & ~jump0

    alpha[jump0, t+1] = 1
    X[jump0, t+1]     = st1[jump0]
    # Reset des horloges ayant sonné
    n_j0 = jump0.sum()
    c01[jump0]    = 0.0;  e01[jump0]    = -np.log(np.random.random(size=n_j0))
    c10rev[jump0] = 0.0;  e10rev[jump0] = -np.log(np.random.random(size=n_j0))

    alpha[stay0, t+1] = 0
    X[stay0, t+1] = (Xt[stay0]
                     - dt*nx * dxw0(Xt[stay0], Yt[stay0])
                     + np.sqrt(2*nx*dt/b) * Bx[stay0])

    # ─── Bloc α = 1 ───────────────────────────────────────────────────
    Y[m1, t+1] = (Yt[m1]
                  - dt*ny * dyw1(Xt[m1], Yt[m1])
                  + np.sqrt(2*ny*dt/b) * By[m1])

    jump1 = m1 & ((c01rev > e01rev) | (c10 > e10))
    stay1 = m1 & ~jump1

    if jump1.any():
        idx1 = np.where(jump1)[0]
        prob_mat = (
            K10(pos[None, :], Yt[idx1, None], st[idx1, None])
          + K01rev(pos[None, :], Yt[idx1, None], st[idx1, None])
        ) * dx
        row_sums = prob_mat.sum(axis=1, keepdims=True)
        prob_mat /= row_sums
        chosen = np.array([
            np.random.choice(pos, p=prob_mat[k])
            for k in range(len(idx1))
        ])
        alpha[jump1, t+1] = 0
        X[jump1, t+1]     = st1[jump1] + chosen
        n_j1 = jump1.sum()
        c01rev[jump1] = 0.0; e01rev[jump1] = -np.log(np.random.random(size=n_j1))
        c10[jump1]    = 0.0; e10[jump1]    = -np.log(np.random.random(size=n_j1))

    alpha[stay1, t+1] = 1
    X[stay1, t+1]     = st1[stay1]

# %% Trajectoire de Z_t = sum_i alpha_i (X_i + Y_i)

time = dt * np.arange(nsteps)

Z = np.sum(alpha * (X + Y), axis=0)          # (nsteps,)  somme sur les têtes attachées
n_attached = alpha.sum(axis=0)               # nombre de têtes attachées à chaque instant

fig, axs = plt.subplots(nrows=3, figsize=(18, 12), sharex=True)

axs[0].plot(time, Z, lw=0.8)
axs[0].set_ylabel(r"$\sum_i \alpha_i (X_i + Y_i)$  (nm)")
axs[0].set_title(r"Trajectoire de $Z_t = \sum_i \alpha_i (X_i + Y_i)$")

axs[1].plot(time, k * Z, lw=0.8, color='tab:red')
axs[1].set_ylabel(r"$k\,Z_t$  (pN)")
axs[1].set_title("Force totale exercée par les têtes attachées")

axs[2].step(time, n_attached, where='post', lw=0.8, color='tab:green', label='têtes attachées')
axs[2].axhline(N, color='black', ls='--', lw=1, label=f'N = {N} (total)')
axs[2].set_xlabel("time (ms)")
axs[2].set_ylabel(r"$\sum_i \alpha_i$")
axs[2].set_ylim(0, N + 1)
axs[2].set_title(f"Nombre de têtes attachées (max atteint : {int(n_attached.max())} / {N})")
axs[2].legend()

plt.tight_layout()
plt.show()

print("Z moyen  :", Z.mean(), "nm")
print("Force moyenne k*Z :", k * Z.mean(), "pN")
print(f"Nombre max de têtes attachées : {int(n_attached.max())} / {N}")

# %%
