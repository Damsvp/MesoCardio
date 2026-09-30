# %%
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
lambda_b = 0.5            # λ_b
lambda_b_tilde = lambda_b/(1+lambda_b)   # \tilde λ_b
v0 = 0.5
z0 = v0/lambda_b_tilde -1/2
beta = 1.5* 4 * (1 + lambda_b)   # β > 4(1 + λ_b) : pente max de la sigmoïde > 1


assert beta > 4 * (1 + lambda_b)

delta= np.sqrt(1 - 4*(1 + lambda_b)/beta)
delta_z = delta/(2*lambda_b) - (2/(beta*lambda_b_tilde)*np.arctanh(delta))        # demi-largeur de la plage de z autour de z0
n_z = 2                  # nombre de valeurs de part et d'autre de z0
z_values = z0 + np.linspace(-delta_z, delta_z, 2 * n_z + 1)   # centrées en z0



def rhs(p, z):
    """Membre de droite de l'équation auto-cohérente p = F(p; z)."""
    arg = v0 - 0.5 - lambda_b_tilde * z + p / (1 + lambda_b)
    return 1.0 / (1.0 + np.exp(-beta * arg))


p = np.linspace(1e-4, 1 - 1e-4, 2000)
colors = plt.cm.viridis(np.linspace(0, 1, len(z_values)))

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(p, p, "k--", lw=1.5, label=r"$p$")

for z, c in zip(z_values, colors):
    F = rhs(p, z)
    ax.plot(p, F, color=c, lw=2, label=rf"$F(p)$, $z={z:.2f}$")
    # Points fixes : changements de signe de F(p) - p
    g = F - p
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    p_star = p[idx] - g[idx] * (p[idx + 1] - p[idx]) / (g[idx + 1] - g[idx])
    ax.plot(p_star, p_star, "o", color=c, mec="k", ms=6)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$F(p)$")
ax.set_title(
    rf"$p = \left[1+e^{{-\beta(v_0 - 1/2 - \tilde\lambda_b z + p/(1+\lambda_b))}}\right]^{{-1}}$"
    "\n"
    rf"$\beta={beta:.2f} > 4(1+\lambda_b)={4 * (1 + lambda_b):.2f}$, "
    rf"$\lambda_b={lambda_b}$, $\tilde\lambda_b={lambda_b_tilde:.3f}$, $z_0={z0:.2f}$, $v_0={v0}$"
)
ax.legend(fontsize=8, loc="upper left")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %%
# Diagramme de bifurcation : solutions p* de p = F(p; z) en fonction de z - z0.
# L'équation s'inverse exactement en z(p) :
#   z = [ v0 - 1/2 + p/(1+λ_b) - logit(p)/β ] / \tilde λ_b
# ce qui donne la courbe complète (y compris la branche instable et les points critiques).
p_branch = np.linspace(1e-6, 1 - 1e-6, 20000)
z_branch = (v0 - 0.5 + p_branch / (1 + lambda_b) - np.log(p_branch / (1 - p_branch)) / beta) / lambda_b_tilde

# Stabilité du point fixe : |F'(p*)| = β p(1-p)/(1+λ_b) < 1
stable = beta * p_branch * (1 - p_branch) / (1 + lambda_b) < 1

# Points critiques (dz/dp = 0) : p± = (1 ± δ)/2, atteints en z = z0 ∓ delta_z
p_crit = 0.5 * (1 + np.array([delta, -delta]))
dz_crit = np.array([delta_z, -delta_z])

x_max = 3 * delta_z   # fenêtre d'affichage autour de z0

fig, ax = plt.subplots(figsize=(8, 6))
ax.axvspan(-delta_z, delta_z, color="0.9", zorder=0,
           label=rf"zone bistable $|z-z_0|<\delta_z={delta_z:.3f}$")
ax.plot(np.where(stable, z_branch - z0, np.nan), p_branch, "C0-", lw=2, label="branche stable")
ax.plot(np.where(~stable, z_branch - z0, np.nan), p_branch, "C3--", lw=2, label="branche instable")
ax.plot(dz_crit, p_crit, "ks", ms=7, label="points critiques (2 solutions)")

# Racines trouvées numériquement pour une grille de z (même méthode que la cellule précédente)
z_scan = z0 + np.linspace(-x_max, x_max, 41)
for z in z_scan:
    g = rhs(p, z) - p
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    p_star = p[idx] - g[idx] * (p[idx + 1] - p[idx]) / (g[idx + 1] - g[idx])
    ax.plot(np.full_like(p_star, z - z0), p_star, "o", color="k", ms=3)

ax.set_xlim(-x_max, x_max)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$z - z_0$")
ax.set_ylabel(r"$p^*$")
ax.set_title(
    rf"Solutions de $p = F(p;z)$ — $\beta={beta:.2f}$, $\lambda_b={lambda_b}$, "
    rf"$\tilde\lambda_b={lambda_b_tilde:.3f}$, $z_0={z0:.2f}$"
)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.show()

# %%
