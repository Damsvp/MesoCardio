#%% 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from numba import njit, prange
from scipy.integrate import quad

#%% Parameters
T     = 20.0     # horizon
N     = 10000    # nombre de pas
M=10 #Nombre de réalisation

kappa = 1 # coefficient d'élasticité
sigma = 1 #Coefficient de diffusiont de X_t et Y_t. 
d=5       #Longueur typique 

dt = T / N #pas de temps 
t  = np.linspace(0, T, N + 1)

x0, y0, alpha0, s0= 0, 0, 0, 0

#%% Fonctions paramétriques

### Potentiels:
@njit
def u(y,alpha):
    if alpha==0:return (y**2-1)**2
    else: return (y**2-1)**2
        
@njit
def du(y,alpha):
    if alpha==0:return 4*y*(y**2-1)
    else: return 4*y*(y**2-1)
    
### Fonction seuils:
@njit
def K01(x,y,s,d):
    if -d/2 < x < d/2: return 1/d     #le saut en X_n sera tiré uniformément sur (-d/2,d/2)
    else: return 0.0
        
@njit
def K10(x,y,s,d):
    if -d/2 < x < d/2: return 1/d
    else: return 0.0

@njit
def k10(y,s,d):
    return 1.0  #A remplacer par l'intégrale de K10

### vitesse de l'actin
@njit
def dxc(t):
    return -0.1

#%% Fonction de simulation
@njit
def simulate(N, T, kappa, sigma, d, x0, y0, alpha0, s0):
    dt = T / N
    
    # Allocation des tableaux
    X     = np.zeros(N + 1)
    Y     = np.zeros(N + 1)
    alpha = np.zeros(N + 1)
    s     = np.zeros(N + 1)
    
    # Conditions initiales
    X[0], Y[0], alpha[0], s[0] = x0, y0, alpha0, s0
    
    # Compteurs et seuils initiaux
    c01 = 0.0
    c10 = 0.0
    e01 = -np.log(np.random.random())   # tirage exponentiel(1)
    e10 = -np.log(np.random.random())
    
    for k in range(N):
        K01k = K01(X[k], Y[k], s[k], d)
        K10k = K10(X[k], Y[k], s[k], d)
        
        if alpha[k] == 0:
            Zx  = np.random.randn()
            c01 += K01k * dt
            
            if c01 < e01:
                alpha[k+1] = 0
                X[k+1] = X[k] - kappa * (X[k] + Y[k]) * dt + sigma * np.sqrt(dt) * Zx
            else:
                c01 = 0.0
                e01 = -np.log(np.random.random())
                alpha[k+1] = 1
                X[k+1] = s[k] - kappa * (X[k] + Y[k]) * dt + sigma * np.sqrt(dt) * Zx
        else:
            c10 += K10k * dt
            
            if c10 < e10:
                alpha[k+1] = 1
                X[k+1] = X[k] + dt * dxc(t[k] if False else k * dt)  # ← t[k] = k*dt
            else:
                c10 = 0.0
                e10 = -np.log(np.random.random())
                alpha[k+1] = 0
                deltaXk = np.random.uniform(-d/2, d/2)
                X[k+1]  = deltaXk + dt * dxc(k * dt)
        
        # Dynamique de Y
        Zy    = np.random.randn()
        Y[k+1] = Y[k] - (du(Y[k], alpha[k]) + kappa * (X[k] + Y[k])) * dt \
                       + sigma * np.sqrt(dt) * Zy
        
        s[k+1] = s[k] + dt * dxc(k * dt)
    
    return X, Y, alpha, s

#%% Simulation 

X, Y, alpha, s = simulate(N,T,kappa,sigma,d, x0, y0, alpha0, s0)

#%% Visualisation de alpha au cours du temps

fig,axs=plt.subplots()
axs.set_title("évolution de alpha au cours du temps")
axs.set_xlabel("temps")
axs.set_ylabel("alpha")
axs.plot(t,alpha,label='alpha')
axs.legend()
axs.grid(True)
plt.show()

#%% Visualisation de X et Y au cours du temps
fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
axs[0].set_title("évolution de X au cours du temps")
axs[0].set_xlabel("temps")
axs[0].set_ylabel("X_t")
axs[0].plot(t,X,label='X_t')
axs[0].plot(t,s,label='s_t')
axs[0].legend()
axs[0].grid(True)

axs[1].set_title("évolution de Y au cours du temps")
axs[1].set_xlabel("temps")
axs[1].set_ylabel("Y_t")
axs[1].plot(t,Y,label='Y_t')
axs[1].legend()
axs[1].grid(True)

plt.show()

# %% Observation de plusieurs réalisations
@njit(parallel=True)
def simulate_all(M, N, T, kappa, sigma, d, x0, y0, alpha0, s0):
    X_all     = np.zeros((M, N+1))
    Y_all     = np.zeros((M, N+1))
    alpha_all = np.zeros((M, N+1))
    s_all     = np.zeros((M, N+1))

    for m in prange(M):   # prange = parallel range
        X_all[m], Y_all[m], alpha_all[m], s_all[m] = simulate(N, T, kappa, sigma, d, x0, y0, alpha0, s0)

    return X_all, Y_all, alpha_all, s_all

#%% simulation

X_all, Y_all, alpha_all, s_all = simulate_all(M, N, T, kappa, sigma, d, x0, y0, alpha0, s0)

#%% visualisation de plusieurs réalisations de alpha au cours du temps
for m in range(M):
    fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(10,5))
    axs[0].set_title("évolution de alpha au cours du temps")
    axs[0].set_xlabel("temps")
    axs[0].set_ylabel("alpha")
    axs[0].plot(t,alpha_all[m],label='alpha')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].set_title("évolution de X au cours du temps")
    axs[1].set_xlabel("temps")
    axs[1].set_ylabel("X_t")
    axs[1].plot(t,X_all[m],label='X_t')
    axs[1].plot(t,s_all[m],label='s_t')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_title("évolution de Y au cours du temps")
    axs[2].set_xlabel("temps")
    axs[2].set_ylabel("Y_t")
    axs[2].plot(t,Y,label='Y_t')
    axs[2].legend()
    axs[2].grid(True)

    plt.show()

# %%
