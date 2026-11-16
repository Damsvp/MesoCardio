**Avancement au 27/10/2026**

<u>Comparaison des h-models à $N$ fini et $N$ indépendants</u>

Cadre du soft device : Force externe imposée et non plus la vitesse $\dot{y}$

On considère une tête definie par sa position $X_t$. On définit une fenêtre de taille $d$ centrée sur cette tête, $X_t$ peut parcourir tout $\R$ et la distance de la tête au site d'actine est appelée $h_t$. Le filament contient $P$ sites d'actine, répartis uniformément. 

La distance totale du point d'ancrage de la tête jusau'au site d'actine le plus proche est appelée $s_t$ telle que  $s_t = X_t + h_t$

On modélise l'attachement-détachement par un paramètre $\alpha_t$ tel que pour $\alpha_t = 1$ la tête est attachée avec l'énergie $w_1$ et $\alpha_t = 0$ la tête est détachée avec l'énergie $w_0$

$\Longrightarrow$ Pour le cas $N$ indépendants on se focalise sur un seul moteur qui subit une force $\sigma$ 

- $\sigma - \tau_c - \nu \dot{y} = 0 \qquad \tau_c (t) = \int \partial_x w_1 \,p(x,1,t)\, dx$

$\Longrightarrow$ Pour le cas $N$ fini, les $N$ moteurs subissent une force totale $\Sigma_{ext} = N\sigma$

En adimensionnant :
-   $\sigma - \frac{1}{N}\sum_i\alpha_t^iX_t^i - \gamma\dot{y} = 0  \qquad \gamma = \frac{N\nu}{\eta}$

$\textcolor{green}{\text{Même problématique que Damien, ce } }$ $\textcolor{green}{\tilde\gamma}$ $\textcolor{green}{\text{a-t-il du sens, est-ce le bon scaling ? }}$

La dynamique des têtes détachées $(\alpha =0)$ s'écrit 
- $d{X}_t^i = - \left. \frac{dw_0}{dx} \right|_{{x} =X_t^i} dt + \sqrt{2\beta^{-1}} dB_t^i + \text{jumps} \qquad \beta^{-1} = \frac{k_B T}{k d^2}$ 

La dynamique des têtes attachées $(\alpha =1)$ s'écrit 

- $dX_t^i = \dot{y} dt + \text{jumps}$ 

Même dynamique dans le cas $N$ têtes indépendantes, sans indice $i$

$\Longrightarrow$ On souhaite comparer les distributions de probabilités de ces deux modèles en faisant tendre $N$ vers $+\infty$ dans le cas $N$ fini et ainsi vérifier la propagation du chaos tel que 

$\sigma - \frac{1}{N} \sum_i\alpha_t^iX_t^i - \tilde\gamma\dot{y} \longrightarrow \sigma - \mathbb{E}[\alpha_t\bar X_t] - \tilde\gamma\dot{\bar y}$

$\Longrightarrow$ <u>1ère implémentation pour le modèle Chaintron 2023 </u> : On considère que l'attachement n'a lieu que lorsque $h$ est au-dessus d'un certain seuil (ex $h \geq 0$), sinon la tête est détachée ($h < 0$)

- La tête $X_t$ avance selon la dynamique définie, à chaque pas de temps on définit la fenêtre de taille $d$ autour de la tête et contenant un seul site d'actine qui sont répartis uniformément sur le filament (espacés de $d$)

- On calcule la distance $h_t = s_t - X_t$ à ce site et on adapte l'état $\alpha$

$\textcolor{green}{\text{On a la position de chaque site car ils sont disposés de manière régulière sur le filament, ou bien prendre le}}$ $\textcolor{green}{h_t}$  $\textcolor{green}{\text{de manière random ?}}$

- Le filament effectue un sliding à vitesse constante $\dot{y} = V$

- On peut calculer $\tau_c$ avec l'équation macro

$\Longrightarrow$ On envisage un modèle on/off, si $h\geq h_{thr}$ la tête s'attache, si $h < h_{thr}$ il y a détachement 
 et $X_t$ suit la dynamique associée à l'état $\alpha$ 

- Plot de $X_t$, de $h_t$ et de $\alpha_t$ en fonction du temps

- On veut obtenir les distributions de probabilité $p(x,t,0)$ et $p(x,t,1)$  samplées sur les $R$ réalisations 

$\textcolor{green}{\text{Utilisation de l'équation de Fokker Planck ?}}$


$\Longrightarrow$ On peut envisager des probabilités d'attachement et de détachement telles que 

- Si $h \geq h_{thr}$ le binding de la tête détachée se fait selon la proba $1 - e^{-K_{att}dt}$ et saute $X\to X+h$. Elle se détache avec une proba $1-e^{-\lambda_{\rm det}dt}$

- Donc concrètement une tête détachée doit attendre de dépasser le seuil puis attendre un temps supplémentaire $1/K_{att}$ pour se lier, puis est tirée sur une longueur $1/\lambda_{\rm det}$ avant de se détacher

$\Longrightarrow$ On définit trois modèles sur la base du h-model :

- <u> Modèle 1  (N = 1) </u> 
    
     $ \sigma - k\alpha_t X_t - \gamma \dot{y} = 0$

- <u> Modèle 2  (N = 1) + Meanfield </u> 

     $ \sigma - k\left<\alpha_t X_t\right>- \gamma \dot{y} = 0$
- <u> Modèle 3  (N fini) </u> 

     $\sigma - \frac{k}{N}\sum_i^N\alpha_t X_t - \gamma \dot{y} = 0$


## 1. Fokker-Planck equations

We want to write the Fokker Planck's equations for the case $N$ finite so that we now consider a probability $p(\underline{\alpha}_t,\underline{X}_t,\underline{h}_t )$ where the underlined variables are such that $\underline{\alpha}_t = (\alpha_t^1, \alpha_t^2,\,... \,,\alpha_t^N)$

We start by re deriving the expression in the case $N=1$ 



The goal is to find the operator $\mathcal{L}$ so that we can write for any test function $f$

$$\partial_t\mathbb{E}[f(\alpha_t, X_t, h_t)] = \mathbb{E}[\mathcal{L}f(\alpha_t, X_t, h_t)]$$

Separating the two cases of attachement and detachment the dynamics in the detached case $(\alpha = 0)$ read in a dimensionless form 

$$dX_t = - w_0'(X_t) dt + \sqrt{2\beta^{-1}}dB_t + jumps$$

$$dh_t = \left(\dot{y} + w_0'(X_t) \right)dt - \sqrt{2\beta^{-1}}dB_t$$

And the attached case $(\alpha = 1)$ is therefore 

$$dX_t = \dot{y} dt + jumps \qquad dh_t = 0$$

Writing the differential for any function $f$ we have 

$$df (\alpha_t, X_t, h_t) = \frac{\partial f}{\partial x} dX_t + \frac{\partial f}{\partial h} dh_t + \frac{1}{2}\frac{\partial^2 f }{\partial x^2} dX_t^2 + \frac{1}{2}\frac{\partial^2 f }{\partial h^2} dh_t^2 +\frac{\partial^2 f }{\partial x \partial h} dX_t dh_t + \mathcal{D}_tf \,d\alpha_t$$

Separating the two cases we then have for ($\alpha =0$)

$$df(\alpha_t \lvert _{\alpha_t =0}, X_t, h_t) = \left(- \frac{\partial f}{\partial x} w_0 '(X_t) + \beta^{-1} \frac{\partial^2 f }{\partial x^2}\right)dt \,+ \sqrt{2\beta^{-1}}\frac{\partial f}{\partial x}dB_t$$
$$+ \,(f(1,X_t +h^*,0)-f(0,X_t,h_t))d\alpha_t + \frac{\partial f}{\partial h}\left(\left(\dot{y} + w_0'(X_t)\right)dt - \sqrt{2\beta^{-1}}dB_t\right)$$
$$+\,\beta^{-1}\frac{\partial^2 f }{\partial h^2}dt -2\beta^{-1}\frac{\partial^2 f }{\partial x \partial h}dt$$

The case ($\alpha =1$) is 

$$df(\alpha_t \lvert _{\alpha_t =1}, X_t, h_t) = \dot{y}\frac{\partial f}{\partial x}dt \,+ \left(\int_{\mathbb{T_d}} (f(0,X_t-h^*,h_t) - f(1,X_t,0) ) dh_t \right) d\alpha_t$$

We then want to take the expectation value so that 

$$d\mathbb{E}[f] = \mathbb{E}[df ] = \mathbb{E}[f(\alpha_{t +\delta t,},X_{t +\delta t,},h_{t +\delta t,})-f(\alpha_t, X_t,h_t)]$$

Also,

$$\mathbb{E}[f] = \int f(0,X_t,h_t)p(0,X_t,h_t) + f(1,X_t,h_t)p(1,X_t,h_t)dt$$

So that

$$\mathbb{E}[df] = \int f(\alpha_{t+\delta_t} \lvert _{\alpha_{t+\delta_t} =0},X_{t+\delta_t},h_t)p(0,X_{t+\delta_t},h_{t+\delta_t}) - f(\alpha_t\lvert _{\alpha_t =1},X_t,h_t)p(1,X_t,h_t)$$
$$+\, f(\alpha_{t+\delta_t} \lvert _{\alpha_{t+\delta_t} =1},X_{t+\delta_t},h_t)p(1,X_{t+\delta_t},h_{t+\delta_t}) - f(\alpha_t\lvert _{\alpha_t =0},X_t,h_t)p(0,X_t,h_t) dt$$


From the two expressions for $df$, for the attached and detached states we take the expectation, separating the continuous part and the jumps. 

Let's write the jumping part for each of the state, if the partcile

$$\mathbb{E}[df(\alpha_t \lvert _{\alpha_t =0}, X_t, h_t)]_{jumps} = \mathbb{E}[(f(1,X_t +h_t^*,0)-f(0,X_t,h_t))[d\alpha_t \lvert {\alpha_t =0} ]]$$

$$\mathbb{E}[df(\alpha_t \lvert _{\alpha_t =0}, X_t, h_t)]_{jumps} =\mathbb E\Big[\mathbf 1_{\alpha_t=0}\,\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(X_t,h_t)\,\big(f(1,X_t+h_t^\star,0)-f(0,X_t,h_t)\big)\Big]\,\mathrm dt$$
And

$$\mathbb{E}[df(\alpha_t \lvert _{\alpha_t =1}, X_t, h_t)]_{jumps} = \mathbb{E}[(f(0,X_t -h_t^*,h_t)-f(1,X_t,0))[d\alpha_t \lvert {\alpha_t =0} ]]$$

$$\mathbb{E}[df(\alpha_t \lvert _{\alpha_t =1}, X_t, h_t)]_{jumps}  =\mathbb E\Big[\mathbf 1_{\alpha_t=1}\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(X_t,h)\,\big(f(0,X_t-h^\star,h)-f(1,X_t,0)\big)\,\mathrm dh\Big]\mathrm dt$$

Writing explicitely the expectations give

$$\mathbb{E}[df(\alpha_t, X_t, h_t)]_{jumps} = \int \mathbf 1_{\alpha_t=0}\,\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x,h)\,\big(f(1,x+h^\star,0)-f(0,x,h)\big)p(0,x,h,t)\,\mathrm dxdt$$
$$+\int\mathbf 1_{\alpha_t=1}\int_{\mathbb T_d}\Big[\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x,h)\,\big(f(0,x-h^\star,h)-f(1,x,0)\big)\,\mathrm dh\Big]p(1,x,t)\mathrm dxdt$$

We can thus identify the operator for the jump part $\mathcal{L}_{jump}$, considering $dt \rightarrow 0$ so that 

$$\mathcal{L}_{jump}[f(\alpha_t,X_t,h_t)] = \mathbf 1_{\alpha_t=0}\,\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(X_t,h_t)\,\big[f(1,X_t+h_t^\star,0)-f(0,X_t,h_t)\big]$$
$$- \mathbf 1_{\alpha_t=1}\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(X_t,h)\,\big[f(1,X_t,0)-f(0,X_t-h^\star,h)\big]\,\mathrm dh$$

And finally, the total operator taking into account the continuous part is 

$$\mathcal{L}_{tot}[f(\alpha_t,X_t,h_t)]= \left(\mathcal{L}_{cont} +\mathcal{L}_{jump}\right)[f(\alpha_t,X_t,h_t)]$$

So that we finally get the full operator 

$$\mathcal{L}_{tot}[f(\alpha_t,X_t,h_t)] = \mathbf 1_{\alpha_t=0}\,\Big[\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(X_t,h_t)\,\big[f(1,X_t+h_t^\star,0)-f(0,X_t,h_t)\big] $$
$$- w_0 '(X_t){\partial_x}f
+ \big(\dot{y} + w_0'(X_t) \big){\partial_h} f
+\,\beta^{-1}(\partial_{hh} f  -2\partial_{xh} f  + \partial_{xx} f \Big]$$
$$- \mathbf 1_{\alpha_t=1}\Big[\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(X_t,h)\,\big[f(1,X_t,0)-f(0,X_t-h^\star,h)\big]\,\mathrm dh- \dot{y}\frac{\partial f}{\partial x}\Big]$$

For all test function $f$ we have 

$$\left< \mathcal{L}_t f,p \right > = \left<  f,\mathcal{L}_t^\star p \right >$$

We know want to derive the ajoint operator $\mathcal L_t^\star$ for one motor with the integrations by parts

**Notation.** $f_0(x,h):=f(0,x,h)$ and $f_1(x):=f(1,x,0)$ (an attached motor has $h=0$). With $p_0(t,x,h)=p(t,0,x,h)$ and $p_1(t,x)=p(t,1,x)$,
$$
\langle f,p\rangle=\iint_{\mathbb R\times\mathbb T_d}f_0\,p_0\,\mathrm dx\,\mathrm dh+\int_{\mathbb R}f_1\,p_1\,\mathrm dx ,
$$
and the operator reads
$$
\begin{aligned}
(\mathcal L_tf)_0&=-w_0'(x)\,\partial_xf_0+\big(\dot y+w_0'(x)\big)\partial_hf_0+\beta^{-1}\big(\partial_{xx}-2\partial_{xh}+\partial_{hh}\big)f_0\\
&\quad+\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x,h)\,\big[f_1(x+h^\star)-f_0(x,h)\big],\\[4pt]
(\mathcal L_tf)_1&=\dot y\,\partial_xf_1+\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x,h)\,\big[f_0(x-h^\star,h)-f_1(x)\big]\,\mathrm dh .
\end{aligned}
$$

**Assumptions.** $f$ is bounded with bounded derivatives, $w_0'$ is bounded, $p_0\in C^2$ and $p_1\in C^1$, and $p_0,p_1$ and their derivatives vanish as $|x|\to\infty$. The torus $\mathbb T_d$ has no boundary, so all boundary terms in $h$ cancel by periodicity.

**Computation of $\langle\mathcal L_tf,p\rangle$.**

(i) Drift in $x$, $\alpha=0$:
$$
-\iint w_0'\,\partial_xf_0\,p_0=-\int_{\mathbb T_d}\big[w_0'f_0p_0\big]_{x=-\infty}^{x=+\infty}\mathrm dh+\iint f_0\,\partial_x(w_0'p_0)=\iint f_0\,\partial_x(w_0'p_0).
$$

(ii) Drift in $h$, $\alpha=0$ ($\dot y+w_0'(x)$ does not depend on $h$):
$$
\iint(\dot y+w_0')\,\partial_hf_0\,p_0=\underbrace{\int_{\mathbb R}\big[(\dot y+w_0')f_0p_0\big]_{h=-d/2}^{h=d/2}\mathrm dx}_{=0}-\iint f_0\,(\dot y+w_0')\,\partial_hp_0 .
$$

(iii) Second-order terms, $\alpha=0$. Two integrations by parts, with boundary terms $[\partial_xf_0\,p_0-f_0\,\partial_xp_0]_{x=\pm\infty}=0$ and the periodicity in $h$:
$$
\iint\partial_{xx}f_0\,p_0=\iint f_0\,\partial_{xx}p_0,\qquad
\iint\partial_{hh}f_0\,p_0=\iint f_0\,\partial_{hh}p_0,\qquad
\iint\partial_{xh}f_0\,p_0=-\iint\partial_hf_0\,\partial_xp_0=\iint f_0\,\partial_{xh}p_0 .
$$
Hence $\beta^{-1}\iint(\partial_{xx}-2\partial_{xh}+\partial_{hh})f_0\,p_0=\beta^{-1}\iint f_0\,(\partial_{xx}-2\partial_{xh}+\partial_{hh})p_0$.

(iv) Transport, $\alpha=1$:
$$
\int\dot y\,\partial_xf_1\,p_1\,\mathrm dx=-\int f_1\,\dot y\,\partial_xp_1\,\mathrm dx .
$$

(v) Attachment, loss: $-\iint\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x,h)\,f_0\,p_0$ is already of the form $\iint f_0\,[\cdots]$.
Attachment, gain: at fixed $h$ set $u=x+h^\star$ (a translation, $\mathrm du=\mathrm dx$) and use Fubini:
$$
\iint\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x,h)\,f_1(x+h^\star)\,p_0(x,h)\,\mathrm dx\,\mathrm dh=\int f_1(u)\Big[\int_{\mathbb T_d}\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(u-h^\star,h)\,p_0(u-h^\star,h)\,\mathrm dh\Big]\mathrm du .
$$

(vi) Detachment, loss: $-\int f_1\,p_1(x)\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x,h)\,\mathrm dh\,\mathrm dx$ is already of the form $\int f_1[\cdots]$.
Detachment, gain: at fixed $h$ set $u=x-h^\star$, so $x=u+h^\star$:
$$
\int p_1(x)\!\int_{\mathbb T_d}\!\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x,h)\,f_0(x-h^\star,h)\,\mathrm dh\,\mathrm dx=\iint f_0(u,h)\,\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(u+h^\star,h)\,p_1(u+h^\star)\,\mathrm du\,\mathrm dh .
$$

**Result.** Collecting the coefficients of $f_0(x,h)$ and $f_1(x)$, $\langle\mathcal L_tf,p\rangle=\langle f,\mathcal L_t^\star p\rangle$ with
$$
\begin{aligned}
(\mathcal L_t^\star p)_0&=\partial_x(w_0'p_0)-(\dot y+w_0')\,\partial_hp_0+\beta^{-1}(\partial_{xx}-2\partial_{xh}+\partial_{hh})p_0\\
&\quad-\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x,h)\,p_0(x,h)+\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x+h^\star,h)\,p_1(x+h^\star),\\[4pt]
(\mathcal L_t^\star p)_1&=-\dot x_c\,\partial_xp_1-p_1(x)\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x,h)\,\mathrm dh+\int_{\mathbb T_d}\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x-h^\star,h)\,p_0(x-h^\star,h)\,\mathrm dh .
\end{aligned}
$$
Since $w_0'$ depends on $x$ only, $-w_0'\partial_hp_0=-\partial_h(w_0'p_0)$, and the first line of $(\mathcal L_t^\star p)_0$ equals $-\dot y\,\partial_hp_0-2\beta^{-1}\partial_x\partial_hp_0-\partial_h\big[w_0'p_0-\beta^{-1}\partial_hp_0\big]+\partial_x\big[w_0'p_0+\beta^{-1}\partial_xp_0\big]$. The system $\partial_tp_0=(\mathcal L_t^\star p)_0$, $\partial_tp_1=(\mathcal L_t^\star p)_1$ is therefore (3.6a)--(3.6b), with $\beta^{-1}=\eta^{-1}k_BT$ and $w_0'=\eta^{-1}\partial_xw_0$.

## 2. The case $N$ motors

**Model.** For $i=1,\dots,N$, $(\alpha^i_t,X^i_t,h^i_t)$ solves (3.2) driven by its own Brownian motion $B^i$ and its own Poisson measures $N^i_{0\to1},N^{i,\rm rev}_{1\to0},N^i_{1\to0},N^{i,\rm rev}_{0\to1}$. All of these are mutually independent, and $\dot y(t)$ is common. The rates of motor $i$ depend on its own state. Consequently:

- $\mathrm d\langle X^i,X^j\rangle=\mathrm d\langle X^i,h^j\rangle=\mathrm d\langle h^i,h^j\rangle=0$ for $i\neq j$, so there is no cross term between different motors. The cross term $-2\beta^{-1}\partial_{x_i}\partial_{h_i}$ only couples $X^i$ and $h^i$.
- Almost surely, two motors never jump at the same time. At a jump of motor $i$, the other $N-1$ motors keep their $(\alpha^j,X^j,h^j)$.

Hence the generator is a sum over $i$ of the single-motor generator acting on the variables of motor $i$.

**Notation.** For $\boldsymbol\alpha\in\{0,1\}^N$, let $D=D(\boldsymbol\alpha)=\{j:\alpha_j=0\}$ be the detached motors and $A=A(\boldsymbol\alpha)=\{j:\alpha_j=1\}$ the attached ones. The state is $(\boldsymbol\alpha,\mathbf x,\mathbf h_D)$ with $\mathbf x\in\mathbb R^N$ and $\mathbf h_D=(h_j)_{j\in D}\in\mathbb T_d^{|D|}$ (attached motors have $h_j=0$). For a vector $\mathbf v$, $\mathbf v^{[i\leftarrow u]}$ is $\mathbf v$ with its $i$-th entry replaced by $u$. We write $f_{\boldsymbol\alpha}(\mathbf x,\mathbf h_D)$ for the test function, $p_{\boldsymbol\alpha}(t,\mathbf x,\mathbf h_D)$ for the law, and
$$
\langle f,p\rangle=\sum_{\boldsymbol\alpha\in\{0,1\}^N}\int_{\mathbb R^N\times\mathbb T_d^{|D(\boldsymbol\alpha)|}}f_{\boldsymbol\alpha}\,p_{\boldsymbol\alpha}\,\mathrm d\mathbf x\,\mathrm d\mathbf h_D ,\qquad \mathrm d\mathbf h_D=\prod_{j\in D}\mathrm dh_j .
$$

**Generator.** The jump of motor $i$ from $\alpha_i=0$ goes to $(\boldsymbol\alpha^{[i\leftarrow1]},\mathbf x^{[i\leftarrow x_i+h_i^\star]},h_i\text{ removed})$, and from $\alpha_i=1$ with mark $h$ goes to $(\boldsymbol\alpha^{[i\leftarrow0]},\mathbf x^{[i\leftarrow x_i-h^\star]},h_i{=}h)$. Therefore
$$
\begin{aligned}
(\mathcal L^N_tf)_{\boldsymbol\alpha}&=\sum_{i\in D}\Big[-w_0'(x_i)\,\partial_{x_i}f_{\boldsymbol\alpha}+\big(\dot y+w_0'(x_i)\big)\partial_{h_i}f_{\boldsymbol\alpha}+\beta^{-1}(\partial_{x_i}-\partial_{h_i})^2f_{\boldsymbol\alpha}\Big]+\sum_{i\in A}\dot y\,\partial_{x_i}f_{\boldsymbol\alpha}\\
&\quad+\sum_{i\in D}\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x_i,h_i)\Big[f_{\boldsymbol\alpha^{[i\leftarrow1]}}\big(\mathbf x^{[i\leftarrow x_i+h_i^\star]},\mathbf h_{D\setminus\{i\}}\big)-f_{\boldsymbol\alpha}(\mathbf x,\mathbf h_D)\Big]\\
&\quad+\sum_{i\in A}\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x_i,h)\Big[f_{\boldsymbol\alpha^{[i\leftarrow0]}}\big(\mathbf x^{[i\leftarrow x_i-h^\star]},(\mathbf h_D,h_i{=}h)\big)-f_{\boldsymbol\alpha}(\mathbf x,\mathbf h_D)\Big]\mathrm dh .
\end{aligned}
$$
In the last two sums only the $i$-th entries of $(\alpha,x,h)$ change inside $f$, and the other $N-1$ motors are in the same state before and after the jump. The first line is a sum of one-motor operators, with no coupling between motors.

**Adjoint.** We compute $\langle\mathcal L^N_tf,p\rangle$ term by term.

*Continuous part.* For each $\boldsymbol\alpha$ and each $i$, steps (i)--(iv) of Section 1 are applied in the variables $(x_i,h_i)$ (for $i\in D$) or $x_i$ (for $i\in A$). The other variables are spectators, so there is nothing to integrate by parts in them. Boundary terms vanish as before.

*Attachment of motor $i$.* The loss needs no change. For the gain, the sum over $\boldsymbol\alpha$ with $\alpha_i=0$ and the integral over $(\mathbf x,\mathbf h_D)$ read
$$
\sum_{\boldsymbol\alpha:\,\alpha_i=0}\int\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x_i,h_i)\,f_{\boldsymbol\alpha^{[i\leftarrow1]}}\big(\mathbf x^{[i\leftarrow x_i+h_i^\star]},\mathbf h_{D\setminus\{i\}}\big)\,p_{\boldsymbol\alpha}(\mathbf x,\mathbf h_D)\,\mathrm d\mathbf x\,\mathrm d\mathbf h_D .
$$
Set $\boldsymbol\beta:=\boldsymbol\alpha^{[i\leftarrow1]}$, a bijection from $\{\alpha_i=0\}$ onto $\{\beta_i=1\}$ with $D(\boldsymbol\beta)=D(\boldsymbol\alpha)\setminus\ \{i\}$. At fixed $h_i$, substitute $u_i=x_i+h_i^\star$ (the other $x_j$ unchanged) and rename $h_i\to h$:
$$
\sum_{\boldsymbol\beta:\,\beta_i=1}\int f_{\boldsymbol\beta}(\mathbf u,\mathbf h_{D(\boldsymbol\beta)})\Big[\int_{\mathbb T_d}\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(u_i-h^\star,h)\,p_{\boldsymbol\beta^{[i\leftarrow0]}}\big(\mathbf u^{[i\leftarrow u_i-h^\star]},(\mathbf h_{D(\boldsymbol\beta)},h_i{=}h)\big)\mathrm dh\Big]\mathrm d\mathbf u\,\mathrm d\mathbf h_{D(\boldsymbol\beta)} .
$$

*Detachment of motor $i$.* The loss needs no change. For the gain, with $\boldsymbol\beta:=\boldsymbol\alpha^{[i\leftarrow0]}$ (a bijection from $\{\alpha_i=1\}$ onto $\{\beta_i=0\}$, $D(\boldsymbol\alpha)=D(\boldsymbol\beta)\setminus\{i\}$), at fixed $h$ substitute $u_i=x_i-h^\star$, so $x_i=u_i+h^\star$, and identify $h_i=h$:
$$
\sum_{\boldsymbol\beta:\,\beta_i=0}\int f_{\boldsymbol\beta}(\mathbf u,\mathbf h_{D(\boldsymbol\beta)})\,\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(u_i+h_i^\star,h_i)\,p_{\boldsymbol\beta^{[i\leftarrow1]}}\big(\mathbf u^{[i\leftarrow u_i+h_i^\star]},\mathbf h_{D(\boldsymbol\beta)\setminus\{i\}}\big)\,\mathrm d\mathbf u\,\mathrm d\mathbf h_{D(\boldsymbol\beta)} .
$$

**Result.** Since $f_{\boldsymbol\alpha}$ is arbitrary for each $\boldsymbol\alpha$ (renaming $\boldsymbol\beta\to\boldsymbol\alpha$, $\mathbf u\to\mathbf x$), $\langle\mathcal L^N_tf,p\rangle=\langle f,\mathcal L^{N\star}_tp\rangle$ with
$$
\begin{aligned}
(\mathcal L^{N\star}_tp)_{\boldsymbol\alpha}&=\sum_{i\in D}\Big[\partial_{x_i}\big(w_0'(x_i)\,p_{\boldsymbol\alpha}\big)-\partial_{h_i}\big(\big(\dot y+w_0'(x_i)\big)p_{\boldsymbol\alpha}\big)+\beta^{-1}(\partial_{x_i}-\partial_{h_i})^2p_{\boldsymbol\alpha}\Big]-\sum_{i\in A}\dot y\,\partial_{x_i}p_{\boldsymbol\alpha}\\
&\quad-\sum_{i\in D}\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x_i,h_i)\,p_{\boldsymbol\alpha}-\sum_{i\in A}\Big[\int_{\mathbb T_d}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x_i,h)\,\mathrm dh\Big]p_{\boldsymbol\alpha}\\
&\quad+\sum_{i\in D}\big(K_{1\to0}+K^{\rm rev}_{0\to1}\big)(x_i+h_i^\star,h_i)\,p_{\boldsymbol\alpha^{[i\leftarrow1]}}\big(\mathbf x^{[i\leftarrow x_i+h_i^\star]},\mathbf h_{D\setminus\{i\}}\big)\\
&\quad+\sum_{i\in A}\int_{\mathbb T_d}\big(K_{0\to1}+K^{\rm rev}_{1\to0}\big)(x_i-h^\star,h)\,p_{\boldsymbol\alpha^{[i\leftarrow0]}}\big(\mathbf x^{[i\leftarrow x_i-h^\star]},(\mathbf h_D,h_i{=}h)\big)\,\mathrm dh ,
\end{aligned}
$$
and the Fokker--Planck system for the $N$ motors is $\partial_tp_{\boldsymbol\alpha}=(\mathcal L^{N\star}_tp)_{\boldsymbol\alpha}$ for each $\boldsymbol\alpha\in\{0,1\}^N$ ($2^N$ coupled equations).



**Reduction to one motor :** If $p_{\boldsymbol\alpha}=\prod_{i=1}^Np_{\alpha_i}(t,x_i,h_i)$ (no $h_i$ factor when $\alpha_i=1$), then $\mathcal L^{N\star}_tp=\sum_i(\mathcal L^\star_t p)_{\alpha_i}\prod_{j\neq i}p_{\alpha_j}$. A product of solutions of (3.6) therefore solves the $N$-motor system. This is the expected consequence of independent noises, own-state rates and a common imposed $\dot y(t)$.