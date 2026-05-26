#%%
import numpy as np
import matplotlib.pyplot as plt

#%%  simulation du modèle 
# ── Paramètres ──────────────────────────────────────────────
N       = 25 #nombre de particules
T       = 1000.0  #Horizon de temps
dt      = 0.01 #pas de temps
steps   = int(T/dt)
eps     = 0.3 #température 
lam_b   = 0.0 # KESACO?
z       = 1.0 #à adapter
# X0 = np.random.randn(N) #états initiaux des N particules
X0 = np.zeros(N)
p=1 #nombres de trajectoires que je veux voir

def u_prime(x):
    return x**3-x #potentiel à deux puits. 

# ── Initialisation ───────────────────────────────────────────


def simulation(N,T,dt,eps,lam_b,z,p,X0):
    steps   = int(T/dt)
    X=X0
    m_history    = np.zeros(steps)
    traj_history = np.zeros((steps,p))   # ← trajectoire de la particule 0
    
    # ── Simulation Euler-Maruyama ────────────────────────────────
    for k in range(steps):
        m_t = X.mean()
        m_history[k]    = m_t
        traj_history[k] = X[:p]        # ← on enregistre X_t^0 avant la mise à jour
    
        drift = (- u_prime(X)
                 + (lam_b * z + m_t) / (1 + lam_b)
                 - X)
        noise = np.sqrt(2 * eps * dt) * np.random.randn(N)
        X    += drift * dt + noise
    return traj_history, m_history
# ── Visualisation ────────────────────────────────────────────

traj_history , m_history  = simulation(N,T,dt,eps,lam_b,z,p,X0)
steps   = int(T/dt)
t = np.linspace(0, T, steps)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for l in range(p):
    axes[0].plot(t, traj_history[:,l], lw=0.8, label=r"$X_t^$")
axes[0].set_ylabel(r"$X_t^0$")
axes[0].set_title("Trajectoire de " +str(p) + " particules")
axes[0].grid(True)


axes[1].plot(t, m_history, color="crimson", lw=1.5, label=r"$m_t = \bar{X}_t^N$")
axes[1].set_ylabel(r"$m_t$")
axes[1].set_xlabel("t")
axes[1].set_title("Champ moyen empirique")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
print(np.shape(traj_history[0]))

#%% Observation de plusieurs réalisations
# p=1 

# ── Paramètres ──────────────────────────────────────────────
N       = 50 #nombre de particules
T       = 10000.0  #Horizon de temps
dt      = 0.01 #pas de temps
steps   = int(T/dt)
eps     = 0.3 #température 
lam_b   = 0.0 # KESACO?
z       = 1.0 #à adapter
# X0 = np.random.randn(N) #états initiaux des N particules
X0 = np.zeros(N)
p=1 #nombres de trajectoires que je veux voir

def u_prime(x):
    return x**3-x #potentiel à deux puits. 

# ------------------------------------------------------------
n=5
traj=np.zeros((n,steps))
X0=np.random.randn(N)
traj_history , m_history  = simulation(N,T,dt,eps,lam_b,z,n,X0)
traj[n,:]= traj_history.T
t = np.linspace(0, T, steps)

moy_ind=np.zeros(steps)
for k in range(steps):
    moy_ind[k]=traj[:,k].mean()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for l in range(n):
    axes[0].plot(t, traj[l,:], lw=0.8, label=r"$X_t^$")
axes[0].set_ylabel(r"$X_t^0$")
axes[0].set_title("Trajectoire de " +str(n) + " particules")
axes[0].grid(True)


axes[1].plot(t, moy_ind, color="crimson", lw=1.5, label=r"moy = $\bar{X}_t^N$")
axes[1].set_ylabel(r"$\moy_ind$")
axes[1].set_xlabel("t")
axes[1].set_title("moyenne des n processus indépendants")
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(-1,1)

plt.tight_layout()
plt.show()



# %%
