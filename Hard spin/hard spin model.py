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
# Paysage d'énergie v(p, y) à z fixé :
#   v(p,y) = (1-p) v0 + (1-p)/2 y² + p/2 (y+1)² + λ_b/2 (z-y)²
# v est affine en p et quadratique en y. À p fixé, le minimum en y est atteint en
#   y*(p) = (λ_b z - p) / (1 + λ_b)
# On trace v en niveaux de couleur + courbes de niveau, pour z = z0 - δz, z0, z0 + δz.
def energy(p, y, z):
    return (1 - p) * v0 + 0.5 * (1 - p) * y**2 + 0.5 * p * (y + 1)**2 + 0.5 * lambda_b * (z - y)**2


y_grid = np.linspace(-1.2, 1.2, 400)
p_grid = np.linspace(0, 1, 300)
Y, P = np.meshgrid(y_grid, p_grid)
z_plot = [z0 - delta_z, z0, z0 + delta_z]
z_labels = [r"z_0-\delta_z", r"z_0", r"z_0+\delta_z"]

V_all = [energy(P, Y, z) for z in z_plot]
v_min, v_max = min(V.min() for V in V_all), max(V.max() for V in V_all)
levels = v_min + (v_max - v_min) * np.linspace(0, 1, 40)**2   # niveaux resserrés près du fond de la vallée

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True, constrained_layout=True)
for ax, z, zl, V in zip(axes, z_plot, z_labels, V_all):
    cf = ax.contourf(Y, P, V, levels=levels, cmap="magma_r")
    cs = ax.contour(Y, P, V, levels=levels[::2], colors="k", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, cs.levels[::2], fontsize=7, fmt="%.2f")

    # Vallée : minimum en y à p fixé
    y_star = (lambda_b * z - p_grid) / (1 + lambda_b)
    ax.plot(y_star, p_grid, "w--", lw=2, label=r"$y^*(p)=\frac{\lambda_b z - p}{1+\lambda_b}$")

    # Le long de la vallée, v est concave en p -> les minima sont aux bords p=0 et p=1
    v_ends = energy(np.array([0.0, 1.0]), y_star[[0, -1]], z)
    k = np.argmin(v_ends)
    ax.plot(y_star[[0, -1]][k], [0, 1][k], "*", color="cyan", mec="k", ms=18, clip_on=False,
            label=rf"min. global $v={v_ends[k]:.3f}$")
    ax.plot(y_star[[0, -1]][1 - k], [0, 1][1 - k], "o", color="w", mec="k", ms=9, clip_on=False,
            label=rf"min. local $v={v_ends[1 - k]:.3f}$")

    ax.set_title(rf"$z = {zl} = {z:.3f}$", pad=12)
    ax.set_xlabel(r"$y$")
    ax.legend(fontsize=8, loc="center left", framealpha=0.85)

axes[0].set_ylabel(r"$p$")
fig.colorbar(cf, ax=axes, label=r"$v(p,y)$", shrink=0.9)
fig.suptitle(
    r"$v(p,y) = (1-p)v_0 + \frac{1-p}{2}y^2 + \frac{p}{2}(y+1)^2 + \frac{\lambda_b}{2}(z-y)^2$"
    rf"  —  $\lambda_b={lambda_b}$, $v_0={v0}$"
)
plt.show()

# %%
# Coupes de v(p, y) en fonction de p, à y et z fixés.
# v est affine en p :  v(p,y) = v0 + y²/2 + λ_b/2 (z-y)² + p (y - y0),  avec y0 = v0 - 1/2
# -> pente négative pour y < y0 (p=1 favorisé), nulle pour y = y0, positive pour y > y0 (p=0 favorisé).
# z ne fait que translater verticalement les droites.
y0 = v0 - 0.5
dy = 0.3
y_cut = [y0 - dy, y0, y0 + dy]
y_cut_labels = [rf"y = y_0 - {dy}", r"y = y_0", rf"y = y_0 + {dy}"]
z_colors = plt.cm.viridis(np.linspace(0, 0.85, len(z_plot)))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
for ax, y, yl in zip(axes, y_cut, y_cut_labels):
    for z, zl, c in zip(z_plot, z_labels, z_colors):
        ax.plot(p_grid, energy(p_grid, y, z), color=c, lw=2.5, label=rf"$z = {zl}$")
    ax.set_title(rf"${yl} = {y:.2f}$   (pente $y - y_0 = {y - y0:+.2f}$)")
    ax.set_xlabel(r"$p$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel(r"$v(p,y)$")
fig.suptitle(
    r"$v(p,y) = v_0 + \frac{y^2}{2} + \frac{\lambda_b}{2}(z-y)^2 + p\,(y - y_0)$, $y_0 = v_0 - \frac{1}{2}$"
    rf"  —  $\lambda_b={lambda_b}$, $v_0={v0}$, $y_0={y0}$"
)
plt.show()
