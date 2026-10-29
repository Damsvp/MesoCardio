**Avancement au 27/10/2026**

<u>Comparaison des h-models à $N$ fini et $N$ indépendants</u>

Cadre du soft device : Force externe imposée et non plus la vitesse $\dot{y}$

On considère une tête definie par sa position $X_t$. On définit une fenêtre de taille $d$ centrée sur cette tête, $X_t$ peut parcourir tout $\R$ et la distance de la tête au site d'actine est appelée $h_t$. Le filament contient $P$ sites d'actine, répartis uniformément. 

La distance totale du point d'ancrage de la tête jusau'au site d'actine le plus proche est appelée $s_t$ telle que  $s_t = X_t + h_t$

On modélise l'attachement-détachement par un paramètre $\alpha_t$ tel que pour $\alpha_t = 1$ la tête est attachée avec l'énergie $w_1$ et $\alpha_t = 0$ la tête est détachée avec l'énergie $w_0$

$\Longrightarrow$ Pour le cas $N$ indépendants on se focalise sur un seul moteur qui subit une force $\sigma$ 

- $ \sigma - \tau_c - \nu \dot{y} = 0 \qquad \tau_c (t) = \int \partial_x w_1 \,p(x,1,t)\, dx$

$\Longrightarrow$ Pour le cas $N$ fini, les $N$ moteurs subissent une force totale $\Sigma_{ext} = N\sigma$

En adimensionnant :
-   $\ \sigma - \frac{1}{N}\sum_i\alpha_t^iX_t^i - \tilde\gamma\dot{y} = 0  \qquad \tilde\gamma = \frac{N\nu}{\eta}$

$\textcolor{green}{\text{Même problématique que Damien, ce } } $ $\textcolor{green}{\tilde\gamma}$ $\textcolor{green}{\text{a-t-il du sens, est-ce le bon scaling ? }}$

La dynamique des têtes détachées $(\alpha =0)$ s'écrit 
- $d{X}_t^i = - \left. \frac{dw_0}{dx} \right|_{{x} =X_t^i} dt + \sqrt{2\beta^{-1}} dB_t^i + \text{jumps} \qquad \beta^{-1} = \frac{k_B T}{k d^2} $ 

La dynamique des têtes attachées $(\alpha =1)$ s'écrit 

- $dX_t^i = \dot{y} dt + \text{jumps}$ 

Même dynamique dans le cas $N$ têtes indépendantes, sans indice $i$

$\Longrightarrow$ On souhaite comparer les distributions de probabilités de ces deux modèles en faisant tendre $N$ vers $+\infty$ dans le cas $N$ fini et ainsi vérifier la propagation du chaos tel que 

$ \sigma - \frac{1}{N} \sum_i\alpha_t^iX_t^i - \tilde\gamma\dot{y} \longrightarrow \sigma - \mathbb{E}[\alpha_t\bar X_t] - \tilde\gamma\dot{\bar y} $

$\Longrightarrow$ <u>1ère implémentation pour le modèle Chaintron 2023 </u> : On considère que l'attachement n'a lieu que lorsque $h$ est au-dessus d'un certain seuil (ex $h \geq 0$), sinon la tête est détachée ($h < 0$)

- La tête $X_t$ avance selon la dynamique définie, à chaque pas de temps on définit la fenêtre de taille $d$ autour de la tête et contenant un seul site d'actine qui sont répartis uniformément sur le filament (espacés de $d$)

- On calcule la distance $h_t = s_t - X_t $ à ce site et on adapte l'état $\alpha$

$\textcolor{green}{\text{On a la position de chaque site car ils sont disposés de manière régulière sur le filament, ou bien prendre le}}$ $\textcolor{green}{h_t}$  $\textcolor{green}{\text{de manière random ?}}$

- Le filament effectue un sliding à vitesse constante $ \dot{y} = V$

- On peut calculer $\tau_c$ avec l'équation macro

$\Longrightarrow$ On envisage un modèle on/off, si $h \geq h_{thr}$ la tête s'attache, si $ h < h_{thr}$ il y a détachement 
 et $X_t$ suit la dynamique associée à l'état $\alpha$ 

- Plot de $X_t$, de $h_t$ et de $\alpha_t$ en fonction du temps

- On veut obtenir les distributions de probabilité $p(x,t,0)$ et $p(x,t,1)$  samplées sur les $R$ réalisations 

$\textcolor{green}{\text{Utilisation de l'équation de Fokker Planck ?}}$


$\Longrightarrow$ On peut envisager des probabilités d'attachement et de détachement telles que 

- Si $h \geq h_{thr}$ le binding de la tête détachée se fait selon la proba $1 - e^{-K_{att}dt}$ et saute $X\to X+h$. Elle se détache avec une proba $1-e^{-\lambda_{\rm det}dt}$

- Donc concrètement une tête détachée doit attendre de dépasser le seuil puis attendre un temps supplémentaire $1/K_{att}$ pour se lier, puis est tirée sur une longueur $1/\lambda_{\rm det}$ avant de se détacher