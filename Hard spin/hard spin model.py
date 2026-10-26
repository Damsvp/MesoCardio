# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

# Paramètres
lambda_b = 0.5            # λ_b
lambda_b_tilde = lambda_b/(1+lambda_b)   # \tilde λ_b
v0 = 0.5
y0 = v0 - 1/2             # pente de ν en p nulle en y = y0
z0 = v0/lambda_b_tilde -1/2
beta = 1.5* 4 * (1 + lambda_b)   # β > 4(1 + λ_b) : pente max de la sigmoïde > 1
N = 20                    # nombre de spins, p ∈ Λ^N = {0, 1/N, ..., 1}
l = -1/2                  # position de l'état barrière entre x = 0 et x = -1
v_star = 2*v0             # v(l) = v_⋆ : hauteur de la barrière
tau = np.exp(-beta * v_star)   # τ = e^{-β v_⋆} : unité de temps de la dynamique (préfacteur commun des taux k_±)
epsilon = 1               # rapport d'échelles de temps entre Y (rapide si ε petit) et p, en unités de τ


assert beta > 4 * (1 + lambda_b)

delta= np.sqrt(1 - 4*(1 + lambda_b)/beta)
delta_z = delta/(2*lambda_b) - (2/(beta*lambda_b_tilde)*np.arctanh(delta))        # demi-largeur de la plage de z autour de z0
n_z = 2                  # nombre de valeurs de part et d'autre de z0
z_values = z0 + np.linspace(-delta_z, delta_z, 2 * n_z + 1)   # centrées en z0

# Valeurs de z utilisées pour les paysages d'énergie : bords et centre de la zone bistable
z_plot = [z0 - delta_z/2, z0, z0 + delta_z]


def z_label(z):
    """Étiquette LaTeX de z sous la forme z_0 + c δ_z (c = (z - z0)/δ_z)."""
    c = (z - z0) / delta_z
    if np.isclose(c, 0):
        return r"z_0"
    coef = "" if np.isclose(abs(c), 1) else f"{abs(c):.3g}"
    return rf"z_0{'+' if c > 0 else '-'}{coef}\delta_z"


z_labels = [z_label(z) for z in z_plot]   # suit automatiquement z_plot
p_N = np.arange(N + 1) / N   # Λ^N


def nu(p, y, z):
    """Énergie ν(p,y;z) = (1-p) v0 + (1-p)/2 y² + p/2 (y+1)² + λ_b/2 (z-y)²."""
    return (1 - p) * v0 + 0.5 * (1 - p) * y**2 + 0.5 * p * (y + 1)**2 + 0.5 * lambda_b * (z - y)**2


def y_hat(p, z):
    """Valeur d'équilibre de y à p fixé : ŷ(p;z) = (λ_b z - p)/(1+λ_b)."""
    return (lambda_b * z - p) / (1 + lambda_b)


def nu_hat(p, z):
    """ν̂(p;z) = ν(p, ŷ(p)) = (1-p) v0 + p(1-p)/2 + λ̃_b/2 (z+p)²."""
    return (1 - p) * v0 + 0.5 * p * (1 - p) + 0.5 * lambda_b_tilde * (z + p)**2


def s_N(p):
    """Entropie s_N(p) = 1/N log C(N, Np), pour p ∈ Λ^N."""
    k = np.rint(N * np.asarray(p))
    return (gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)) / N


def f_N(p, y, z):
    """Énergie avec entropie f_N(p,y;z) = ν(p,y;z) - s_N(p)/β."""
    return nu(p, y, z) - s_N(p) / beta


def f_N_hat(p, z):
    """f̂_N(p;z) = f_N(p, ŷ(p);z) = ν̂(p;z) - s_N(p)/β."""
    return nu_hat(p, z) - s_N(p) / beta


def rhs(p, z):
    """Membre de droite de l'équation auto-cohérente p = F(p; z) (point critique de ν̂ - s/β, N → ∞)."""
    arg = y0 - lambda_b_tilde * z + p / (1 + lambda_b)
    return 1.0 / (1.0 + np.exp(-beta * arg))


def z_of_p(p):
    """Inverse exact de p = F(p; z) : z = [y0 + p/(1+λ_b) - logit(p)/β] / λ̃_b."""
    return (y0 + p / (1 + lambda_b) - np.log(p / (1 - p)) / beta) / lambda_b_tilde


def fixed_points(z, p=np.linspace(1e-4, 1 - 1e-4, 2000)):
    """Solutions de p = F(p; z) : changements de signe de F(p) - p, affinés par interpolation linéaire."""
    g = rhs(p, z) - p
    idx = np.where(np.sign(g[:-1]) != np.sign(g[1:]))[0]
    return p[idx] - g[idx] * (p[idx + 1] - p[idx]) / (g[idx + 1] - g[idx])


def dy_nu(p, y, z):
    """Dérivée partielle de ν par rapport à y : ∂ν/∂y = (1-p) y + p (y+1) - λ_b (z-y)."""
    return (1 - p) * y + p * (y + 1) - lambda_b * (z - y)


def k_plus(y):
    """Taux par spin 0 -> -1 (via l) : exp(-β δu_+), δu_+ = u(l,y) - u(0,y) = v_⋆ - l y + l²/2 - v0."""
    return np.exp(-beta * (v_star - l * y + 0.5 * l**2 - v0))


def k_minus(y):
    """Taux par spin -1 -> 0 (via l) : exp(-β δu_-), δu_- = u(l,y) - u(-1,y) = v_⋆ - (l+1) y + l²/2 - 1/2."""
    return np.exp(-beta * (v_star - (l + 1) * y + 0.5 * l**2 - 0.5))


p = np.linspace(1e-4, 1 - 1e-4, 2000)
colors = plt.cm.viridis(np.linspace(0, 1, len(z_values)))

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(p, p, "k--", lw=1.5, label=r"$p$")

for z, c in zip(z_values, colors):
    ax.plot(p, rhs(p, z), color=c, lw=2, label=rf"$F(p)$, $z={z:.2f}$")
    p_star = fixed_points(z)
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
# L'équation s'inverse exactement en z(p) (fonction z_of_p),
# ce qui donne la courbe complète (y compris la branche instable et les points critiques).
p_branch = np.linspace(1e-6, 1 - 1e-6, 20000)
z_branch = z_of_p(p_branch)

# Stabilité du point fixe : |F'(p*)| = β p(1-p)/(1+λ_b) < 1
stable = beta * p_branch * (1 - p_branch) / (1 + lambda_b) < 1

# Points critiques (dz/dp = 0) : p± = (1 ± δ)/2, atteints en z = z0 ± delta_z
p_crit = 0.5 * (1 + np.array([delta, -delta]))
dz_crit = z_of_p(p_crit) - z0   # = [+delta_z, -delta_z]

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
    p_star = fixed_points(z)
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
# Paysage d'énergie ν(p, y) à z fixé, pour z = z0 - δz, z0, z0 + δz.
# ν est affine en p et quadratique en y. À p fixé, le minimum en y est atteint en ŷ(p),
# et le long de cette vallée ν̂(p) est concave en p -> les minima sont aux bords p=0 et p=1.
y_grid = np.linspace(-1.2, 1.2, 400)
p_grid = np.linspace(0, 1, 300)
Y, P = np.meshgrid(y_grid, p_grid)

V_all = [nu(P, Y, z) for z in z_plot]
v_min, v_max = min(V.min() for V in V_all), max(V.max() for V in V_all)
levels = v_min + (v_max - v_min) * np.linspace(0, 1, 40)**2   # niveaux resserrés près du fond de la vallée

fig, axes = plt.subplots(1, len(z_plot), figsize=(5.3 * len(z_plot), 5.5), squeeze=False, sharey=True, constrained_layout=True)
for ax, z, zl, V in zip(axes[0], z_plot, z_labels, V_all):
    cf = ax.contourf(Y, P, V, levels=levels, cmap="magma_r")
    cs = ax.contour(Y, P, V, levels=levels[::2], colors="k", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, cs.levels[::2], fontsize=7, fmt="%.2f")

    ax.plot(y_hat(p_grid, z), p_grid, "w--", lw=2, label=r"$\hat y(p)=\frac{\lambda_b z - p}{1+\lambda_b}$")

    p_ends = np.array([0.0, 1.0])
    v_ends = nu_hat(p_ends, z)
    i_glob = np.argmin(v_ends)
    for i in (i_glob, 1 - i_glob):
        is_glob = i == i_glob
        ax.plot(y_hat(p_ends[i], z), p_ends[i], "*" if is_glob else "o", color="cyan" if is_glob else "w",
                mec="k", ms=18 if is_glob else 9, clip_on=False,
                label=rf"min. {'global' if is_glob else 'local'} $\nu={v_ends[i]:.3f}$")

    ax.set_title(rf"$z = {zl} = {z:.3f}$", pad=12)
    ax.set_xlabel(r"$y$")
    ax.legend(fontsize=8, loc="center left", framealpha=0.85)

axes[0, 0].set_ylabel(r"$p$")
fig.colorbar(cf, ax=axes, label=r"$\nu(p,y)$", shrink=0.9)
fig.suptitle(
    r"$\nu(p,y) = (1-p)v_0 + \frac{1-p}{2}y^2 + \frac{p}{2}(y+1)^2 + \frac{\lambda_b}{2}(z-y)^2$"
    rf"  —  $\lambda_b={lambda_b}$, $v_0={v0}$"
)
plt.show()

# %%
# Coupes de ν(p, y) en fonction de p, à y et z fixés.
# ν est affine en p :  ν(p,y) = v0 + y²/2 + λ_b/2 (z-y)² + p (y - y0),  avec y0 = v0 - 1/2
# -> pente négative pour y < y0 (p=1 favorisé), nulle pour y = y0, positive pour y > y0 (p=0 favorisé).
# z ne fait que translater verticalement les droites.
dy = 0.3
y_cut = [y0 - dy, y0, y0 + dy]
y_cut_labels = [rf"y = y_0 - {dy}", r"y = y_0", rf"y = y_0 + {dy}"]
z_colors = plt.cm.viridis(np.linspace(0, 0.85, len(z_plot)))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
for ax, y, yl in zip(axes, y_cut, y_cut_labels):
    for z, zl, c in zip(z_plot, z_labels, z_colors):
        ax.plot(p_grid, nu(p_grid, y, z), color=c, lw=2.5, label=rf"$z = {zl}$")
    ax.set_title(rf"${yl} = {y:.2f}$   (pente $y - y_0 = {y - y0:+.2f}$)")
    ax.set_xlabel(r"$p$")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel(r"$\nu(p,y)$")
fig.suptitle(
    r"$\nu(p,y) = v_0 + \frac{y^2}{2} + \frac{\lambda_b}{2}(z-y)^2 + p\,(y - y_0)$, $y_0 = v_0 - \frac{1}{2}$"
    rf"  —  $\lambda_b={lambda_b}$, $v_0={v0}$, $y_0={y0}$"
)
plt.show()

# %%
# Mêmes coupes, pour l'énergie à N fini :
#   f_N(p,y) = ν(p,y) - s_N(p)/β,  s_N(p) = 1/N log C(N, Np),   p ∈ Λ^N = {0, 1/N, ..., 1}
# L'entropie est nulle en p=0 et p=1 et maximale en p=1/2 : elle creuse les droites ν(p,y).
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
for ax, y, yl in zip(axes, y_cut, y_cut_labels):
    for z, zl, c in zip(z_plot, z_labels, z_colors):
        ax.plot(p_grid, nu(p_grid, y, z), color=c, lw=1, ls="--", alpha=0.5)   # ν(p,y) de référence
        f = f_N(p_N, y, z)
        ax.plot(p_N, f, "-o", color=c, lw=1.2, ms=5, label=rf"$z = {zl}$")
        i_min = np.argmin(f)
        ax.plot(p_N[i_min], f[i_min], "*", color=c, mec="k", ms=15, zorder=3)
    ax.set_title(rf"${yl} = {y:.2f}$")
    ax.set_xlabel(r"$p \in \Lambda^N$")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel(r"$f_N(p,y)$")
fig.suptitle(
    r"$f_N(p,y) = \nu(p,y) - \frac{1}{\beta}s_N(p)$, $s_N(p) = \frac{1}{N}\log\binom{N}{Np}$"
    rf"  —  $N={N}$, $\beta={beta:.2f}$, $\lambda_b={lambda_b}$, $v_0={v0}$"
    "\n(points : $f_N$ ;  pointillés : $\\nu$ ;  étoiles : minimum sur $\\Lambda^N$)"
)
plt.show()

# %%
# Paysage d'énergie de f_N(p,y) = ν(p,y) - s_N(p)/β sur Λ^N × R, à z fixé.
# Chaque ligne horizontale correspond à un p = k/N de Λ^N.
# À p fixé, le minimum en y est ŷ(p), où f_N vaut f̂_N(p) = ν̂(p) - s_N(p)/β.
# On marque les minima locaux de f̂_N sur Λ^N (plus bas que leurs voisins p ± 1/N),
# et le point selle entre deux puits : maximum de f̂_N le long de la vallée entre deux minima
# (minimum en y, maximum en p). Sa hauteur au-dessus de chaque puits donne la barrière d'énergie.
F_all = [f_N(p_N[:, None], y_grid[None, :], z) for z in z_plot]   # forme (N+1, len(y_grid))
f_min, f_max = min(F.min() for F in F_all), max(F.max() for F in F_all)
levels_N = f_min + (f_max - f_min) * np.linspace(0, 1, 40)**2
p_edges = np.append(p_N - 0.5 / N, 1 + 0.5 / N)   # bords des cases de Λ^N
y_edges = np.concatenate(([y_grid[0]], 0.5 * (y_grid[1:] + y_grid[:-1]), [y_grid[-1]]))

fig, axes = plt.subplots(1, len(z_plot), figsize=(5.3 * len(z_plot), 5.5), squeeze=False, sharey=True, constrained_layout=True)
for ax, z, zl, F in zip(axes[0], z_plot, z_labels, F_all):
    pm = ax.pcolormesh(y_edges, p_edges, F, cmap="magma_r", vmin=f_min, vmax=f_max)
    cs = ax.contour(y_grid, p_N, F, levels=levels_N[::2], colors="k", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, cs.levels[::2], fontsize=7, fmt="%.2f")

    y_eq = y_hat(p_N, z)
    f_hat = f_N_hat(p_N, z)
    ax.plot(y_eq, p_N, "w--o", lw=1.5, ms=3, label=r"$\hat y(p)=\frac{\lambda_b z - p}{1+\lambda_b}$")

    padded = np.concatenate(([np.inf], f_hat, [np.inf]))
    loc = np.where((f_hat < padded[:-2]) & (f_hat < padded[2:]))[0]
    i_glob = loc[np.argmin(f_hat[loc])]
    for i in loc:
        if i == i_glob:
            ax.plot(y_eq[i], p_N[i], "*", color="cyan", mec="k", ms=18, clip_on=False,
                    label=rf"min. global $p={p_N[i]:.2f}$, $f_N={f_hat[i]:.3f}$")
        else:
            ax.plot(y_eq[i], p_N[i], "o", color="w", mec="k", ms=9, clip_on=False,
                    label=rf"min. local $p={p_N[i]:.2f}$, $f_N={f_hat[i]:.3f}$")

    for i_a, i_b in zip(loc[:-1], loc[1:]):   # un point selle entre chaque paire de puits voisins
        i_s = i_a + np.argmax(f_hat[i_a:i_b + 1])
        ax.plot(y_eq[i_s], p_N[i_s], "X", color="lime", mec="k", ms=13, zorder=4,
                label=rf"point selle $p={p_N[i_s]:.2f}$, $f_N={f_hat[i_s]:.3f}$")
        ax.annotate("barrière depuis\n"
                    + "\n".join(rf"$p={p_N[i]:.2f}$ : $\Delta f_N={f_hat[i_s] - f_hat[i]:.3f}$" for i in (i_b, i_a)),
                    (y_eq[i_s], p_N[i_s]), xytext=(12, 0), textcoords="offset points",
                    fontsize=8, va="center", bbox=dict(boxstyle="round", fc="w", alpha=0.85))

    ax.set_title(rf"$z = {zl} = {z:.3f}$", pad=12)
    ax.set_xlabel(r"$y$")
    ax.set_ylim(-0.5 / N, 1 + 0.5 / N)
    ax.legend(fontsize=8, loc="center left", framealpha=0.85)

axes[0, 0].set_ylabel(r"$p \in \Lambda^N$")
fig.colorbar(pm, ax=axes, label=r"$f_N(p,y)$", shrink=0.9)
fig.suptitle(
    r"$f_N(p,y) = \nu(p,y) - \frac{1}{\beta}s_N(p)$, $s_N(p) = \frac{1}{N}\log\binom{N}{Np}$"
    rf"  —  $N={N}$, $\beta={beta:.2f}$, $\lambda_b={lambda_b}$, $v_0={v0}$"
)
plt.show()

# %%
# Simulation de la dynamique (p_t, Y_t) pour N fini, à z fixé, avec bruit thermique.
# Unité de temps : τ = e^{-β v_⋆}. Les taux par spin k_± = e^{-β δu_±} = τ · e^{-β(δu_± - v_⋆)},
# donc en unités de τ ils deviennent k̃_± = k_± / τ = e^{-β(δu_± - v_⋆)} (indépendants de la hauteur v_⋆).
#   - Y : ε dY = -∂_y ν(p,Y) dt + sqrt(2ε/(βN)) dW        (schéma d'Euler, pas dt)
#   - p ∈ Λ^N saute de ±1/N : p -> p + 1/N au taux N(1-p) k̃_+(Y),  p -> p - 1/N au taux N p k̃_-(Y)
#     (N(1-p) spins en 0 peuvent passer en -1, N p spins en -1 peuvent passer en 0).
#     Les k̃_± sont grands (~e^{β v0}) : on simule les sauts exactement (Gillespie) pendant chaque pas dt, Y fixé.
# Ces taux vérifient le bilan détaillé pour la mesure ∝ exp(-βN f_N(p,y)) : la dynamique visite les puits de f_N.

#paramètres de la simulation    
lambda_b = 0.2            # λ_b
lambda_b_tilde = lambda_b/(1+lambda_b)   # \tilde λ_b
v0 = 0.5
y0 = v0 - 1/2             # pente de ν en p nulle en y = y0
z0 = v0/lambda_b_tilde -1/2
beta = 1.1* 4 * (1 + lambda_b)   # β > 4(1 + λ_b) : pente max de la sigmoïde > 1
N = 100                    # nombre de spins, p ∈ Λ^N = {0, 1/N, ..., 1}
l = -1/2                  # position de l'état barrière entre x = 0 et x = -1
v_star = 1.2*v0             # v(l) = v_⋆ : hauteur de la barrière
tau = np.exp(-beta * v_star)   # τ = e^{-β v_⋆} : unité de temps de la dynamique (préfacteur commun des taux k_±)
epsilon = 1e-2              # rapport d'échelles de temps entre Y (rapide si ε petit) et p, en unités de τ

rng = np.random.default_rng(0)
z = z0
T = 200                    # durée simulée, en unités de τ
dt = 1e-3                 # pas de temps pour Y (doit rester petit devant ε/(1+λ_b))
n = int(T / dt)           # nombre de pas de temps



k_t = np.zeros(n, dtype=int)    # nombre de spins en -1 : p_t = k_t / N
Y_t = np.zeros(n)
k_t[0] = rng.integers(0, N + 1)   # p_0 aléatoire
Y_t[0] = y_hat(k_t[0] / N, z)     # Y_0 à l'équilibre pour p_0

dW = rng.normal(0, np.sqrt(dt), n)   # incréments browniens ~ N(0, dt)
diffusion = np.sqrt(2 / (beta * epsilon * N))

for t in range(1, n):
    Y_t[t] = Y_t[t - 1] - dy_nu(k_t[t - 1] / N, Y_t[t - 1], z) / epsilon * dt + diffusion * dW[t]

    # Sauts de p sur [t-1, t] à Y = Y_t[t] fixé (algorithme de Gillespie)
    kp, km = k_plus(Y_t[t]) / tau, k_minus(Y_t[t]) / tau   # taux par spin en unités de τ
    k = k_t[t - 1]
    s_left = dt
    while True:
        rate_up, rate_down = (N - k) * kp, k * km
        rate_tot = rate_up + rate_down
        wait = rng.exponential(1 / rate_tot)
        if wait > s_left:
            break
        s_left -= wait
        k += 1 if rng.random() * rate_tot < rate_up else -1
    k_t[t] = k

p_t = k_t / N

# Puits de f̂_N à ce z, pour repère
f_hat = f_N_hat(p_N, z)
padded = np.concatenate(([np.inf], f_hat, [np.inf]))
wells = p_N[(f_hat < padded[:-2]) & (f_hat < padded[2:])]

Time = np.arange(n) * dt
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].plot(Time, Y_t, lw=0.6)
for pw in wells:
    axs[0].axhline(y_hat(pw, z), color="k", ls="--", lw=1)
axs[0].set_ylabel(r"$Y_t$")
axs[0].set_title(rf"Évolution de $Y_t$ (pointillés : $\hat y(p)$ aux puits de $\hat f_N$)")
axs[1].plot(Time, p_t, lw=0.8, drawstyle="steps-post")
for pw in wells:
    axs[1].axhline(pw, color="k", ls="--", lw=1)
axs[1].set_xlabel(r"Temps (unité $\tau = e^{-\beta v_*}$)")
axs[1].set_ylabel(r"$p_t$")
axs[1].set_ylim(-0.02, 1.02)
axs[1].set_title(r"Évolution de $p_t$ (pointillés : puits de $\hat f_N$)")
fig.suptitle(rf"$z = {z_label(z)} = {z:.3f}$, $N={N}$, $\beta={beta:.2f}$, $\varepsilon={epsilon}$, "
             rf"$l={l}$, $v_*={v_star}$, $\tau={tau:.2e}$")
plt.tight_layout()
plt.show()

# %%
