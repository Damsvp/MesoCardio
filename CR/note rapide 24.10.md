**Discussion du 24/10/2026**

1) <u>Présentation de Damien</u>: Modèle Hard Spin présenté sous forme de $N\beta z$ et $N \beta \sigma$. Discussion sur les dynamiques de $(p_t,Y_t)$ et l'approximation $N$ grand de la dynamique de $p_t$ comme diffusion. Discussion autour du Soft device et de la réponse de la variable $z$ à un petit step de $\sigma$. 

$\Longrightarrow$ Simuler la dynamique complète $(p_t,Y_t,Z_t)$ pour un $\sigma(t)$ faisant un petit step. 

$\Longrightarrow$ Regarder la dynamique de $\langle p \rangle$. Cette dynamique est-elle représentative ?

$\Longrightarrow$ Comparer la limite de diffusion avec les sauts notamment quand $N$ grand.

$\Longrightarrow$ Les dynamqiues des $x_i$ peuvent-elles être traiter indépendamment quand $N$ grand ? 

$\Longrightarrow$ Equivalence des ensembles $ N \beta z$ et $N \beta \sigma$. 

2) <u>Présentation de Titouan</u>: Framezork du modèle $h$ avec force imposée et non plus $\dot{y}$ imposé pour un nombre N fini de moteurs. Comparaison avec le $h$-model Chaintron 2023 avec N têtes indépendantes. Compréhension de l'EDS $d h_t = \dot{y} d t - \pi(X_t)$. 

$\Longrightarrow$ Introduire un modèle simple de sauts avec variantes (type constantes ; attachement pour $h>0$ et détachement pour $h<0$)

$\Longrightarrow$ Faire des simulations pour comparer les différentes distributions quand $N$ grand. (Phénomène de propagation de chaos)

$\Longrightarrow$ Mieux comprendre l'EDS sur la varaible $h$. (cf ci-dessous)



**Explication de $\pi$ et de l'EDS $d h_t = \dot{y} d t - \pi(X_t)$**

1) <u>Définition du Tore et de $\pi$.</u> 

On définit l'action de groupe $\rho_d: (k,z) \in \mathbb{Z} \times \mathbb{R} \mapsto x + kd \in \mathbb{R} $. On peut écrire que $\mathbb{R}= \bigcup_{x \in [ -d/2,d/2)} \Omega_x$ où $\Omega_x = \{y \in \mathbb{R} ; \exists k \in \mathbb{Z}, y=x+kd\}$ sont les orbites pour cette action. Autrement dit, en français, les $\Omega_x$ sont les ensembles des réels espacés d'une distance un multiple de $d$ de $x$. Ainsi, chaque $x \in \R$ a un unique représentant $x^\star$ dans $[-d/2,d/2)$ i.e. $x \equiv x^\star [d]$.  En posant l'application
$$ \psi : x \in \R \mapsto x^\star \in [-d/2,d/2) $$
je définis un morphisme de groupe (\psi( x + y) = \psi (x) + \psi(y)) surjectif dont le noyau (l'ensemble des éléments $x \in \R$ tels que $\psi(x)=0$) est $d\Z$. Par propriété universelle du quotient, je peux définir une nouvelle application
$$ \Psi : x \in \R / d\Z \mapsto x^\star \in [-d/2,d/2)$$
qui est bijective par construction et reste toujours un morphisme de groupe. Le tore $\mathbb{T}_d$ est par définition $\R / d\Z$. Ainsi, l'application inverse est bien défini et correspond dans l'article à $\pi$
$$ \Psi^{-1} = \pi : x^\star \in [-d/2,d/2) \mapsto x \in \mathbb{T}_d $$
Je peux étendre cette fonction sur tout $\R$ en périodisant ma fonction. Je la renote $\pi: \tilde{x} \in \R \mapsto x \in \mathbb{T}_d$.

2) <u>Compréhension de l'EDS.</u>

Pour l'intuition: Les petits incréments resntent les mêmes dans $\R$ et dans le tore. En effet, par exemple pour $d=1$, dans $\R$, si je passe de $0,45$ à $0,55$ j'effectue un saut de $0,1$. Dans $[-d/2,d/2)$, le même saut correspond au saut de $0,45$ à $-0,45$, il y a donc un incrément de $0,9$. Enfin, dans $\mathbb{T}_d$, le même saut fait passer de $\pi(0,45)$ à $\pi(0,55) = \pi(0,55 -1) = \pi(-0,45)$ ($\pi(1)= 0$) L'incrément est toujours de $0,1$. Conclusion: dans le tore, le déplacement est bien le même que dans $\R$. 

On dit que $\pi$ est une isométrie locale c'est à dire que sur de petits intervalles l'application $\pi$ est une bijection qui conserve les distances. Comme on le ressent intuitivement, un incrément petit $dt \to 0$ se lit de la même façon dans $\R$ que dans $\mathbb{T}_d$:
$$ d \pi(X_t)  ~ "=" ~ \pi(X_{t + dt}) -  \pi(X_{t}) = \pi (X_{t + dt} - X_t) "=" X_{t + dt} - X_t. $$

Aussi, $(\R,\pi)$ est un *revêtement* du tore $\mathbb{T}_d$. Outre la définition mathématique formelle d'un *revêtement* ceci implique que pour tout chemin $h_t$ dans le tore (fonction du temps à valeur dans le tore), toute position initiale $\tilde{h}_0 \in \R$ tel que $h_0 = \pi(\tilde{h}_0)$, il existe un unique chemin $\tilde{h}$ dans $\R$ (fonction du temps à valeurs réelles) telle que à tout instant $h_t = \pi(\tilde{h_t})$. 

Ainsi, on peut définir l'EDS sur le tore $d h_t = \dot{y} d t - d \pi(X_t)$ dans $\mathbb{T}_d$ comme le projeté sur le Tore de l'EDS sur $\R$ vérfié par $\tilde{h}_t$:
$$ \left \{ \begin{array}{rcl}
d \tilde{h}_t & = & \dot{y}dt - d X_t \quad \text{dans} \quad \R \\
h_t & = & \pi(\tilde{h}_t)
\end{array} \right. $$

Pour bien distinguer les différentes notations, un petit récap:
* Ici, $\tilde{h}_t \in \R$ est continu et peut sortir de $[-d/2,d/2)$. Notamment, il compte combien de fois on a fait le tour. 
* $h^\star_t$ correspond à $\tilde{h}_t$ replié dans $[-d/2,d/2)$, il saut de $\pm d$ à chaque changement de site le plus proche
* Enfin, $h_t = \pi(\tilde{h}_t)$ correspond juste au projeté du $\tilde{h}_t$ sur le tore.

On a donc plus qu'une EDS ordinaire sur $\R$. Si je définis $y_t$ comme le déplacement de l'actine par rapport à la myosine à partir d'un point de référence, typiquement un site d'ancrage. Les sites sont aux positions $y_t + kd$ avec $k \in \Z$. Alors la précédente EDS s'intègre directement en $\tilde{h}_t - \tilde{h}_0 = ( y_t - y_0 ) - (X_t -X_0)$ ce qui redonne $h_t = \pi (y_t - X_t) = \pi (s_t - X_t)$ comme $\tilde{h}_0 = h_0^\star \equiv y_0 - X_0 [d]$. Notons la différence de définition entre $y_t = X_t + \tilde{h}_t$ position d'un site d'ancrage de référence et $s_t = X_t + h_t^\star$ position du site d'ancrage le plus proche de $X_t$. 

