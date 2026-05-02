# Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Progressively applying Gaussian noise transforms complex data distributions to approximately Gaussian. Reversing this dynamic defines a generative model. When the forward noising process is given by a Stochastic Differential Equation (SDE), Song et al. (2021) demonstrate how the time inhomogeneous drift of the associated reverse-time SDE may be estimated using score-matching. A limitation of this approach is that the forward-time SDE must be run for a sufficiently long time for the final distribution to be approximately Gaussian. In contrast, solving the Schrödinger Bridge (SB) problem, i.e. an entropy-regularized optimal transport problem on path spaces, yields diffusions which generate samples from the data distribution in finite time. We present Diffusion SB (DSB), an original approximation of the Iterative Proportional Fitting (IPF) procedure to solve the SB problem, and provide theoretical analysis along with generative modeling experiments. The first DSB iteration recovers the methodology proposed by Song et al. (2021), with the flexibility of using shorter time intervals, as subsequent DSB iterations reduce the discrepancy between the final-time marginal of the forward (resp. backward) SDE with respect to the prior (resp. data) distribution. Beyond generative modeling, DSB offers a widely applicable computational optimal transport tool as the continuous state-space analogue of the popular Sinkhorn algorithm (Cuturi, 2013).

# 1 Introduction

Score-Based Generative Modeling (SGM) is a recently developed approach to probabilistic generative modeling that exhibits state-of-the-art performance on several audio and image synthesis tasks; see e.g. Song and Ermon (2019); Cai et al. (2020); Chen et al. (2021a); Kong et al. (2021); Gao et al. (2020); Jolicoeur-Martineau et al. (2021); Ho et al. (2020); Song and Ermon (2020); Song et al. (2020, 2021); Niu et al. (2020); Durkan and Song (2021); Hoogeboom et al. (2021); Sahara et al. (2021); Luhman and Luhman (2021, 2020); Nichol and Dhariwal (2021); Popov et al. (2021); Dhariwal and Nichol (2021). Existing SGMs generally consist of two parts. Firstly, noise is incrementally added to the data in order to obtain a perturbed data distribution approximating an easy-to-sample prior density e.g. Gaussian. Secondly, a neural network is used to learn the reverse-time denoising dynamics, which when initialized at this prior distribution, defines a generative model (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song and Ermon, 2019; Song et al., 2021). Song et al. (2021) have shown that one could fruitfully think of the noisng process as a Stochastic Differential Equation (SDE) progressively perturbing the initial data distribution into an approximately Gaussian one. The corresponding reverse-time SDE is an inhomogeneous diffusion whose drift depends on the logarithmic gradients of the perturbed data distributions, i.e. the scores. In practice, these scores are approximated using neural networks and score-matching techniques (Hyvarinen and Dayan, 2005; Vincent, 2011) while numerical SDE integrators are used for the sampling procedure.

![](images/a2588c63081498a067d3ede8533538da9cf949a0e2e325009d7486a14f3dd7ff.jpg)  
Figure 1: The reference forward diffusion initialized from the 2-dimensional data distribution fails to converge to the Gaussian prior in  $T = 0.2$  diffusion-time ( $N = 20$  discrete time steps), and the reverse diffusion initialized from the Gaussian prior does not converge to the data distribution. However, convergence does occur after 5 DSB iterations.

Although SGM provides state-of-the-art results (Dhariwal and Nichol, 2021), sample generation is computationally expensive. In order to learn the reverse-time SDE from the prior, i.e. the generative model, the forward noising SDE must be run for sufficiently long to converge to the prior and the step size must be sufficiently small for a good SDE approximation. By reformulating generative modeling as a Schrödinger bridge (SB) problem we alleviate this issue and propose a novel algorithm to solve SB problems. Our detailed contributions are as follows.

Generative modeling as a Schrödinger bridge problem. The SB problem is a famous entropy-regularized Optimal Transport (OT) problem introduced by Schrödinger (1932); see e.g. (Léonard, 2014b; Chen et al., 2021b) for reviews. Given a reference diffusion with finite time horizon  $T$ , a data distribution and a prior distribution, solving the SB amounts to finding the closest diffusion to the reference (in terms of Kullback-Leibler divergence on path spaces) which admits the data distribution as marginal at time  $t = 0$  and the prior at time  $t = T$ . The reverse-time diffusion solving this Schrödinger bridge problem provides a new SGM algorithm which enables approximate sample generation from the data distribution using shorter time intervals compared to the original SGM methods. This approach differs from the entropy-regularized OT formulation proposed in (Genevay et al., 2018) which deals with discrete distributions and relies on a static formulation of SB, as opposed to our dynamical approach for continuous distributions which operates on path spaces.

Solving the Schrödinger bridge problem using score-based diffusions. The Schrödinger bridge problem can be solved using Iterative Proportional Fitting (IPF) (Fortet, 1940; Kullback, 1968; Chen et al., 2021b). We propose Diffusion SB (DSB), a novel implementation of IPF using score-based diffusion techniques and an original mean-matching loss. DSB does not require discretizing the state-space (Chen et al., 2016; Reich, 2019), approximating potential functions using regression (Bernton et al., 2019; Dessein et al., 2017; Pavon et al., 2021), nor performing kernel density estimation (Pavon et al., 2021). The first DSB iteration recovers the method proposed by Song et al. (2021), with the flexibility of using shorter time intervals, as additional DSB iterations reduce the discrepancy between the final-time marginal of the forward (resp. backward) SDE w.r.t. the prior (resp. data) distribution; see Figure 1 for an illustration.

Theoretical analysis. We provide quantitative convergence results for the methodology of Song et al. (2021), which elucidates the need for long diffusion times in existing SGM methods. Additionally, we derive novel quantitative convergence results for IPF in continuous state-space which do not rely on classical compactness assumptions (Chen et al., 2016; Ruschendorf et al., 1995) and improve on the recent results of Leger (2020). Finally, we show that our methodology may be viewed as the time discretization of a dynamic version of IPF on path spaces.

Experiments. We validate our methodology by generating image datasets such as MNIST and CelebA. In particular, we show that using multiple steps of DSB always improve the generative model. We also show how DSM can be used to interpolate between two data distributions.

Notation. In the continuous-time setting, we set  $\mathcal{C} = \mathrm{C}([0,T],\mathbb{R}^d)$  the space of continuous functions from  $[0,T]$  to  $\mathbb{R}^d$  and  $\mathcal{B}(\mathcal{C})$  the Borel sets on  $\mathcal{C}$ . For any measurable space  $(\mathsf{E},\mathcal{E})$ , we denote by  $\mathcal{P}(\mathsf{E})$  the space of probability measures on  $(\mathsf{E},\mathcal{E})$ . For any  $\ell \in \mathbb{N}$ , let  $\mathcal{P}_{\ell} = \mathcal{P}((\mathbb{R}^{d})^{\ell})$ . When it is defined, we denote  $\mathrm{H}(p) = -\int_{\mathbb{R}^d}p(x)\log p(x)\mathrm{d}x$  as the entropy of  $p$  and  $\mathrm{KL}(p|q)$  as the Kullback-Leibler divergence between  $p$  and  $q$ . When there is no ambiguity we use the same notation for the distributions and their densities. All proofs are postponed to the supplementary.

# 2 Denoising Diffusion, Score-Matching and Reverse-Time SDEs

# 2.1 Discrete-Time: Markov Chains and Time Reversal

Consider a data distribution with positive density  $p_{\mathrm{data}}^1$ , a positive prior density  $p_{\mathrm{prior}}$  w.r.t. Lebesgue measure both with support on  $\mathbb{R}^d$  and a Markov chain with initial density  $p_0 = p_{\mathrm{data}}$  on  $\mathbb{R}^d$  evolving according to positive transition densities  $p_{k + 1|k}$  for  $k \in \{0, \dots, N - 1\}$ . Hence for any  $x_{0:N} = \{x_k\}_{k=0}^N \in \mathcal{X} = (\mathbb{R}^d)^{N + 1}$ , the joint density may be expressed

$$
p \left(x _ {0: N}\right) = p _ {0} \left(x _ {0}\right) \prod_ {k = 0} ^ {N - 1} p _ {k + 1 \mid k} \left(x _ {k + 1} \mid x _ {k}\right). \tag {1}
$$

This joint density also admits the backward decomposition

$$
p \left(x _ {0: N}\right) = p _ {N} \left(x _ {N}\right) \prod_ {k = 0} ^ {N - 1} p _ {k \mid k + 1} \left(x _ {k} \mid x _ {k + 1}\right), \text {w i t h} p _ {k \mid k + 1} \left(x _ {k} \mid x _ {k + 1}\right) = \frac {p _ {k} \left(x _ {k}\right) p _ {k + 1 \mid k} \left(x _ {k + 1} \mid x _ {k}\right)}{p _ {k + 1} \left(x _ {k + 1}\right)}, \tag {2}
$$

where  $p_k(x_k) = \int p_{k|k-1}(x_k|x_{k-1})p_{k-1}(x_{k-1})\mathrm{d}x_{k-1}$  is the marginal density at step  $k \geq 1$ . For the purpose of generative modeling, we will choose transition densities such that  $p_N(x_N) \approx p_{\text{prior}}(x_N)$  for large  $N$ , where  $p_{\text{prior}}$  is an easy-to-sample prior density. One may sample approximately from  $p_{\text{data}}$  using ancestral sampling with the reverse-time decomposition (2), i.e., first sample  $X_N \sim p_{\text{prior}}$  followed by  $X_k \sim p_{k|k+1}(\cdot | X_{k+1})$  for  $k \in \{N-1, \ldots, 0\}$ . This idea is at the core of all recent SGM. The reverse-time transitions in (2) cannot be simulated exactly but may be approximated if we consider a forward transition density of the form

$$
p _ {k + 1 \mid k} \left(x _ {k + 1} \mid x _ {k}\right) = \mathcal {N} \left(x _ {k + 1}; x _ {k} + \gamma_ {k + 1} f \left(x _ {k}\right), 2 \gamma_ {k + 1} \mathbf {I}\right), \tag {3}
$$

with drift  $f: \mathbb{R}^d \to \mathbb{R}^d$  and stepsize  $\gamma_{k+1} > 0$ . We first make the following approximation from (2)

$$
\begin{array}{l} p _ {k \mid k + 1} \left(x _ {k} \mid x _ {k + 1}\right) = p _ {k + 1 \mid k} \left(x _ {k + 1} \mid x _ {k}\right) \exp \left[ \log p _ {k} \left(x _ {k}\right) - \log p _ {k + 1} \left(x _ {k + 1}\right) \right] \\ \approx \mathcal {N} \left(x _ {k}; x _ {k + 1} - \gamma_ {k + 1} f \left(x _ {k + 1}\right) + 2 \gamma_ {k + 1} \nabla \log p _ {k + 1} \left(x _ {k + 1}\right), 2 \gamma_ {k + 1} \mathbf {I}\right), \tag {4} \\ \end{array}
$$

using that  $p_k \approx p_{k+1}$ , a Taylor expansion of  $\log p_{k+1}$  at  $x_{k+1}$  and  $f(x_k) \approx f(x_{k+1})$ . In practice, the approximation holds if  $\| x_{k+1} - x_k \|$  is small which is ensured by choosing  $\gamma_{k+1}$  small enough. Although  $\nabla \log p_{k+1}$  is not available, one may obtain an approximation using denoising score-matching methods (Hyvärinen and Dayan, 2005; Vincent, 2011; Song et al., 2021). Assume that the conditional density  $p_{k+1|0}(x_{k+1}|x_0)$  is available analytically as in (Ho et al., 2020; Song et al., 2021). We have  $p_{k+1}(x_{k+1}) = \int p_0(x_0)p_{k+1|0}(x_{k+1}|x_0)\mathrm{d}x_0$  and elementary calculations show that  $\nabla \log p_{k+1}(x_{k+1}) = \mathbb{E}_{p_{0|k+1}}[\nabla_{x_{k+1}}\log p_{k+1|0}(x_{k+1}|X_0)]$ . We can therefore formulate score estimation as a regression problem and use a flexible class of functions, e.g. neural networks, to parametrize an approximation  $s_{\theta^*}(k,x_k) \approx \nabla \log p_k(x_k)$  such that

$$
\theta^ {\star} = \arg \min _ {\theta} \sum_ {k = 1} ^ {N} \mathbb {E} _ {p _ {0, k}} [ | | s _ {\theta} (k, X _ {k}) - \nabla_ {x _ {k}} \log p _ {k | 0} (X _ {k} | X _ {0}) | | ^ {2} ],
$$

where  $p_{0,k}(x_0,x_k) = p_0(x_0)p_{k|0}(x_k|x_0)$  is the joint density at steps 0 and  $k$ . If  $p_{k|0}$  is not available, we use  $\theta^{\star} = \arg \min_{\theta}\sum_{k = 1}^{N}\mathbb{E}_{p_{k - 1,k}}[||s_{\theta}(k,X_k) - \nabla_{x_k}\log p_{k|k - 1}(X_k|X_{k - 1})||^2 ]$ . In summary, SGM involves first estimating the score function  $s_{\theta^{\star}}$  from noisy data, and then sampling  $X_0$  using  $X_N\sim p_{\mathrm{prior}}$  and the approximation (4), i.e.

$$
X _ {k} = X _ {k + 1} - \gamma_ {k + 1} f \left(X _ {k + 1}\right) + 2 \gamma_ {k + 1} s _ {\theta^ {\star}} (k + 1, X _ {k + 1}) + \sqrt {2 \gamma_ {k + 1}} Z _ {k + 1}, Z _ {k} \stackrel {{\text {i . i . d .}}} {{\sim}} \mathcal {N} (0, \mathbf {I}). \tag {5}
$$

The random variable  $X_0$  is approximately  $p_0 = p_{\mathrm{data}}$  distributed if  $p_N(x_N) \approx p_{\mathrm{prior}}(x_N)$ . In what follows, we let  $\{Y_k\}_{k=0}^N = \{X_{N-k}\}_{k=0}^N$  and remark that  $\{Y_k\}_{k=0}^N$  satisfies a forward recursion.

# 2.2 Continuous-Time: SDEs, Reverse-Time SDEs and Theoretical results

For appropriate transition densities, Song et al. (2021) showed that the forward and reverse-time Markov chains may be viewed as discretized diffusions. We derive the continuous-time limit of the procedure presented in Section 2.1 and establish convergence results. The Markov chain with kernel (3) corresponds to an Euler-Maruyama discretization of  $(\mathbf{X}_t)_{t\in [0,T]}$ , solving the following SDE

$$
\mathrm {d} \mathbf {X} _ {t} = f \left(\mathbf {X} _ {t}\right) \mathrm {d} t + \sqrt {2} \mathrm {d} \mathbf {B} _ {t}, \quad \mathbf {X} _ {0} \sim p _ {0} = p _ {\text {d a t a}}, \tag {6}
$$

where  $(\mathbf{B}_t)_{t\in [0,T]}$  is a Brownian motion and  $f:\mathbb{R}^d\to \mathbb{R}^d$  is regular enough so that (strong) solutions exist. Under conditions on  $f$ , it is well-known (see Haussmann and Pardoux (1986); Föllmer (1985); Cattiaux et al. (2021) for instance) that the reverse-time process  $(\mathbf{Y}_t)_{t\in [0,T]} = (\mathbf{X}_{T - t})_{t\in [0,T]}$  satisfies

$$
\mathrm {d} \mathbf {Y} _ {t} = \left\{- f \left(\mathbf {Y} _ {t}\right) + 2 \nabla \log p _ {T - t} \left(\mathbf {Y} _ {t}\right) \right\} \mathrm {d} t + \sqrt {2} \mathrm {d} \mathbf {B} _ {t}, \tag {7}
$$

with initialization  $\mathbf{Y}_0\sim p_T$ , where  $p_t$  denotes the marginal density of  $\mathbf{X}_t$ . The reverse-time Markov chain  $\{Y_k\}_{k = 0}^N$  associated with (5) corresponds to an Euler-Maruyama discretization of (7), where the score functions  $\nabla \log p_{t}(x)$  are approximated by  $s_{\theta^{*}}(t,x)$ .

In what follows, we consider  $f(x) = -\alpha x$  for  $\alpha \geq 0$ . This framework includes the one of Song and Ermon (2019) ( $\alpha = 0$ ,  $p_{\mathrm{prior}}(x) = \mathcal{N}(x;0,2T)$ ) for which  $(\mathbf{X}_t)_{t\in [0,T]}$  is simply a Brownian motion and Ho et al. (2020) ( $\alpha > 0$ ,  $p_{\mathrm{prior}}(x) = \mathcal{N}(x;0,1 / \alpha)$ ) for which it is an Ornstein-Uhlenbeck process, see Section S3.3 for more details. Contrary to Song et al. (2021) we consider time homogeneous diffusions. Both approaches approximate (5) using distinct discretizations but our setting leverages the ergodic properties of the Ornstein-Uhlenbeck process to establish Theorem 1.

Theorem 1. Assume that there exists  $\mathbf{M} \geq 0$  such that for any  $t \in [0, T]$  and  $x \in \mathbb{R}^d$

$$
\left\| s _ {\theta^ {*}} (t, x) - \nabla \log p _ {t} (x) \right\| \leq \mathbb {M}, \tag {8}
$$

with  $s_{\theta^{\star}} \in \mathrm{C}([0,T] \times \mathbb{R}^{d}, \mathbb{R}^{d})$ . Assume that  $p_{\mathrm{data}} \in \mathrm{C}^{3}(\mathbb{R}^{d}, (0, + \infty))$  is bounded and that there exist  $d_1, A_1, A_2, A_3 \geq 0, \beta_1, \beta_2, \beta_3 \in \mathbb{N}$  and  $\mathfrak{m}_1 > 0$  such that for any  $x \in \mathbb{R}^d$  and  $i \in \{1,2,3\}$

$$
\left\| \nabla^ {i} \log p _ {\text {d a t a}} (x) \right\| \leq A _ {i} \left(1 + \| x \| ^ {\beta_ {i}}\right), \quad \left\langle \nabla \log p _ {\text {d a t a}} (x), x \right\rangle \leq - \mathfrak {m} _ {1} \| x \| ^ {2} + d _ {1} \| x \|,
$$

with  $\beta_{1} = 1$ . Then for any  $\alpha \geq 0$ , there exist  $B_{\alpha}, C_{\alpha}, D_{\alpha} \geq 0$  such that for any  $N \in \mathbb{N}$  and  $\{\gamma_k\}_{k=1}^N$  with  $\gamma_k > 0$  for any  $k \in \{1, \ldots, N\}$ , the following hold:

(a) if  $\alpha > 0$ , we have  $\| \mathcal{L}(X_0) - p_{\mathrm{data}} \|_{\mathrm{TV}} \leq B_\alpha \exp[-\alpha^{1/2} T] + C_\alpha (\mathsf{M} + \bar{\gamma}^{1/2}) \exp[D_\alpha T]$ ;  
(b) if  $\alpha = 0$ , we have  $\| \mathcal{L}(X_0) - p_{\mathrm{data}} \|_{\mathrm{TV}} \leq B_0 (T^{-1} + T^{-1/2}) + C_0 (\mathsf{M} + \bar{\gamma}^{1/2}) \exp [D_0 T]$ ;

where  $T = \sum_{k=1}^{N} \gamma_k$ ,  $\bar{\gamma} = \sup_{k \in \{1, \dots, N\}} \gamma_k$  and  $\mathcal{L}(X_0)$  is the distribution of  $X_0$  given in (5).

Condition (8) ensures that the neural network approximates the score with a given precision  $\mathbf{M} \geq 0$ . Under this assumption and conditions on  $p_{\mathrm{data}}$ , Theorem 1 states how the Markov chain defined by (5) approximates  $p_{\mathrm{data}}$  in the total variation norm  $\| \cdot \|_{\mathrm{TV}}$ . In both cases,  $\alpha = 0$  and  $\alpha > 0$ , the error consists of two terms. The first term decreases with  $T \geq 0$  and corresponds to the error between  $p_T$  and  $p_{\mathrm{prior}}$ . The second term stems from the error between the continuous-time process (7) with initialization  $\mathbf{Y}_0 \sim p_{\mathrm{prior}}$  and its discrete-time approximation (5). Those bounds show that there is a trade-off between the mixing properties of the Markov chain which increases with  $\alpha$ , and the quality of the discrete-time approximation which deteriorates as  $\alpha$  and  $T$  increase. To the best of our knowledge Theorem 1 is the first result assessing the convergence of SGM methods.

# 3 Diffusion Schrödinger Bridge and Generative Modeling

# 3.1 Schrödinger Bridges

The Schrödinger Bridge (SB) problem is a classical problem appearing in applied mathematics, optimal control and probability; see e.g. Föllmer (1988); Léonard (2014b); Chen et al. (2021b). In the discrete-time setting, it takes the following (dynamic) form. Consider as reference density  $p(x_{0:N})$  given by (1), describing the process adding noise to the data. We aim to find  $\pi^{\star} \in \mathcal{P}_{N+1}$  such that

$$
\pi^ {\star} = \arg \min  \left\{\mathrm {K L} (\pi | p): \pi \in \mathcal {P} _ {N + 1}, \pi_ {0} = p _ {\text {d a t a}}, \pi_ {N} = p _ {\text {p r i o r}} \right\}. \tag {9}
$$

Assuming  $\pi^{\star}$  is available, a generative model can be obtained by sampling  $X_{N} \sim p_{\mathrm{prior}}$ , followed by the reverse-time dynamics  $X_{k} \sim \pi_{k|k+1}^{\star}(\cdot | X_{k+1})$  for  $k \in \{N-1, \ldots, 0\}$ . Before deriving a method to approximate  $\pi^{\star}$  in Section 3.2, we highlight some desirable features of Schrödinger bridges.

Static Schrödinger bridge problem. First, we recall that the dynamic formulation (9) admits a static analogue. Using e.g. Léonard (2014a, Theorem 2.4), the following decomposition holds for any  $\pi \in \mathcal{P}_{N + 1}$ ,  $\mathrm{KL}(\pi |p) = \mathrm{KL}(\pi_{0,N}|p_{0,N}) + \mathbb{E}_{\pi_{0,N}}[\mathrm{KL}(\pi_{|0,N}|p_{|0,N})]$ , where for any  $\mu \in \mathcal{P}_{N + 1}$  we have  $\mu = \mu_{0,N}\mu_{|0,N}$  with  $\mu_{|0,N}$  the conditional distribution of  $X_{1:N - 1}$  given  $X_0,X_N^2$ . Hence we have  $\pi^{\star}(x_{0:N}) = \pi^{s,\star}(x_0,x_N)p_{|0,N}(x_{1:N - 1}|x_0,x_N)$  where  $\pi^{s,\star}\in \mathcal{P}_2$  with marginals  $\pi_0^{s,\star}$  and  $\pi_N^{s,\star}$  is the solution of the static SB problem

$$
\pi^ {\mathrm {s}, \star} = \arg \min  \left\{\mathrm {K L} \left(\pi^ {\mathrm {s}} \mid p _ {0, N}\right): \pi^ {\mathrm {s}} \in \mathcal {P} _ {2}, \pi_ {0} ^ {\mathrm {s}} = p _ {\text {d a t a}}, \pi_ {N} ^ {\mathrm {s}} = p _ {\text {p r i o r}} \right\}. \tag {10}
$$

Link with optimal transport. Under mild assumptions, the static SB problem can be seen as an entropy-regularized optimal transport problem since (10) is equivalent to

$$
\pi^ {\mathrm {s}, \star} = \arg \min  \left\{- \mathbb {E} _ {\pi^ {\mathrm {s}}} [ \log p _ {N | 0} (X _ {N} | X _ {0}) ] - \mathrm {H} (\pi^ {\mathrm {s}}): \pi^ {\mathrm {s}} \in \mathcal {P} _ {2}, \pi_ {0} ^ {\mathrm {s}} = p _ {\text {d a t a}}, \pi_ {N} ^ {\mathrm {s}} = p _ {\text {p r i o r}} \right\}.
$$

If  $p_{k+1|k}(x_{k+1}|x_k) = \mathcal{N}(x_{k+1}; x_k, \sigma_{k+1}^2)$  as in Song and Ermon (2019), then  $p_{N|0}(x_N|x_0) = \mathcal{N}(x_N; x_0, \sigma^2)$  with  $\sigma^2 = \sum_{k=1}^{N} \sigma_k^2$  which induces a quadratic cost and

$$
\pi^ {\mathrm {s}, \star} = \arg \min  \left\{\mathbb {E} _ {\pi^ {\mathrm {s}}} [ | | X _ {0} - X _ {N} | | ^ {2} ] - 2 \sigma^ {2} \mathrm {H} (\pi^ {\mathrm {s}}): \pi^ {\mathrm {s}} \in \mathcal {P} _ {2}, \pi_ {0} ^ {\mathrm {s}} = p _ {\text {d a t a}}, \pi_ {N} ^ {\mathrm {s}} = p _ {\text {p r i o r}} \right\}.
$$

Mikami (2004) showed that  $\pi^{\mathrm{s},\star}\rightarrow \pi_{\mathcal{W}}^{\star}$  weakly and  $2\sigma^{2}\mathrm{KL}(\pi^{\mathrm{s},\star}|p_{0,N})\to \mathcal{W}_{2}^{2}(p_{\mathrm{data}},p_{\mathrm{prior}})$  as  $\sigma \rightarrow 0$  where  $\pi_{\mathcal{W}}^{\star}$  is the optimal transport plan between  $p_{\mathrm{data}}$  and  $p_{\mathrm{prior}}$  and  $\mathcal{W}_2$  is the 2-Wasserstein distance. Note that the transport cost  $c(x,x^{\prime}) = -\log p_{N|0}(x^{\prime}|x)$  is not necessarily symmetric.

# 3.2 Iterative Proportional Fitting and Time Reversal

In all but trivial cases, the SB problem does not admit a closed-form solution. However, it can be solved using Iterative Proportional Fitting (IPF) (Fortet, 1940; Kullback, 1968; Ruschendorf et al., 1995) which is defined by the following recursion for  $n \in \mathbb{N}$  with initialization  $\pi^0 = p$  given in (1):

$$
\pi^ {2 n + 1} = \arg \min  \left\{\mathrm {K L} \left(\pi \mid \pi^ {2 n}\right): \pi \in \mathscr {P} _ {N + 1}, \pi_ {N} = p _ {\text {p r i o r}} \right\}, \tag {11}
$$

$$
\pi^ {2 n + 2} = \arg \min  \left\{\mathrm {K L} \left(\pi \mid \pi^ {2 n + 1}\right): \pi \in \mathscr {P} _ {N + 1}, \pi_ {0} = p _ {\mathrm {d a t a}} \right\}.
$$

This sequence is well-defined if there exists  $\tilde{\pi} \in \mathcal{P}_{N+1}$  such that  $\tilde{\pi}_0 = p_{\mathrm{data}}$ ,  $\tilde{\pi}_N = p_{\mathrm{prior}}$  and  $\mathrm{KL}(\tilde{\pi}|p) < +\infty$ . A standard representation of  $\pi^n$  is obtained by updating the joint density  $p$  using potential functions, see Section S4.2 for details. However, this representation of the IPF iterates is difficult to numerically approximate as it not only requires approximating the potentials but also requires evaluating  $\pi_N^n$  pointwise. Our methodology builds upon an alternative representation that is better suited to numerical approximations in the context of generative modeling where one has access to samples of  $p_{\mathrm{data}}$  and  $p_{\mathrm{prior}}$ .

Proposition 2. Assume that  $\mathrm{KL}(p_{\mathrm{data}}\otimes p_{\mathrm{prior}}|p_{0,N}) < + \infty$  . Then for any  $n\in \mathbb{N}$ $\pi^{2n}$  and  $\pi^{2n + 1}$  admit positive densities w.r.t. the Lebesgue measure denoted as  $p^n$  resp.  $q^n$  and for any  $x_{0:N}\in \mathcal{X}$  we have  $p^0 (x_{0:N}) = p(x_{0:N})$  and

$$
q ^ {n} (x _ {0: N}) = p _ {\text {p r i o r}} (x _ {N}) \prod_ {k = 0} ^ {N - 1} p _ {k | k + 1} ^ {n} (x _ {k} | x _ {k + 1}), p ^ {n + 1} (x _ {0: N}) = p _ {\text {d a t a}} (x _ {0}) \prod_ {k = 0} ^ {N - 1} q _ {k + 1 | k} ^ {n} (x _ {k + 1} | x _ {k}).
$$

In practice we have access to  $p_{k+1|k}^n$  and  $q_{k|k+1}^n$ . Hence, to compute  $p_{k|k+1}^n$  and  $q_{k+1|k}^n$  we use

$$
p _ {k | k + 1} ^ {n} (x _ {k} | x _ {k + 1}) = \frac {p _ {k + 1 | k} ^ {n} (x _ {k + 1} | x _ {k}) p _ {k} ^ {n} (x _ {k})}{p _ {k + 1} ^ {n} (x _ {k + 1})}, q _ {k + 1 | k} ^ {n} (x _ {k + 1} | x _ {k}) = \frac {q _ {k | k + 1} ^ {n} (x _ {k} | x _ {k + 1}) q _ {k + 1} ^ {n} (x _ {k + 1})}{q _ {k} ^ {n} (x _ {k})}.
$$

To the best of our knowledge, this representation of the IPF iterates has surprisingly neither been presented nor explored in the literature. One may interpret these formulas as follows. At iteration  $2n$ , we have  $\pi^{2n} = p^n$  with  $p^0 = p$  given by the noising process (1). This forward process initialized with  $p_0^n = p_{\mathrm{data}}$  defines reverse-time transitions  $p_{k|k+1}^n$ , which, when combined with an initialization  $p_{\mathrm{prior}}$  at step  $N$  defines the reverse-time process  $\pi^{2n+1} = q^n$ . The forward transitions  $q_{k+1|k}^n$  associated to  $q^n$  are then used to obtain  $\pi^{2n+2} = p^{n+1}$ . IPF then iterates this procedure.

# 3.3 Diffusion Schrödinger Bridge as Iterative Mean-Matching Proportional Fitting

To approximate the IPF recursion defined in Proposition 2, we use similar approximations to Section 2.1. If at step  $n \in \mathbb{N}$  we have  $p_{k+1|k}^n(x_{k+1}|x_k) = \mathcal{N}(x_{k+1}; x_k + \gamma_{k+1}f_k^n(x_k), 2\gamma_{k+1}\mathbf{I})$  where  $p^0 = p$  and  $f_k^0 = f$ , then we can approximate the reverse-time transitions in Proposition 2 by

$$
\begin{array}{l} q _ {k | k + 1} ^ {n} \left(x _ {k} \mid x _ {k + 1}\right) = p _ {k + 1 \mid k} ^ {n} \left(x _ {k + 1} \mid x _ {k}\right) \exp \left[ \log p _ {k} ^ {n} \left(x _ {k}\right) - \log p _ {k + 1} ^ {n} \left(x _ {k + 1}\right) \right] \\ \approx \mathcal {N} (x _ {k}; x _ {k + 1} + \gamma_ {k + 1} b _ {k + 1} ^ {n} (x _ {k + 1}), 2 \gamma_ {k + 1} \mathbf {I}), \\ \end{array}
$$

with  $b_{k+1}^n(x_{k+1}) = -f_k^n(x_{k+1}) + 2\nabla \log p_{k+1}^n(x_{k+1})$ . Similarly, we can approximate the forward transitions in Proposition 2 by  $p_{k+1|k}^{n+1}(x_{k+1}|x_k) \approx \mathcal{N}(x_{k+1}; x_k + \gamma_{k+1} f_k^{n+1}(x_k), 2\gamma_{k+1}\mathbf{I})$  with  $f_k^{n+1}(x_k) = -b_{k+1}^n(x_k) + 2\nabla \log q_k^n(x_k)$ . Hence we have  $f_k^{n+1}(x_k) = f_k^n(x_k) - 2\nabla \log p_{k+1}^n(x_k) + 2\nabla \log q_k^n(x_k)$  (and similarly for  $b_k^{n+1}$ ). It follows that, one could estimate  $f_k^{n+1}, b_k^{n+1}$  using score-matching to approximate  $\{\nabla \log p_{k+1}^i(x)\}_{i=0}^n$ ,  $\{\nabla \log q_k^i(x)\}_{i=0}^n$  individually. However, this approach is prohibitively costly in terms of memory and compute, see Section S5 for details. Each of the reverse-time dynamics may also be learnt sequentially through a variational approach as in Ho et al. (2020), see Section S4.4. We follow an alternate approach coined mean-matching which avoids the aforementioned difficulties. This approach yields convenient loss terms from Proposition 3.

Proposition 3. Assume that for any  $n\in \mathbb{N}$  and  $k\in \{0,\dots ,N - 1\}$

$$
q _ {k | k + 1} ^ {n} (x _ {k} | x _ {k + 1}) = \mathcal {N} (x _ {k}; B _ {k + 1} ^ {n} (x _ {k + 1}), 2 \gamma_ {k + 1} \mathbf {I}), p _ {k + 1 | k} ^ {n} (x _ {k + 1} | x _ {k}) = \mathcal {N} (x _ {k + 1}; F _ {k} ^ {n} (x _ {k}), 2 \gamma_ {k + 1} \mathbf {I}),
$$

with  $B_{k+1}^n(x) = x + \gamma_{k+1}b_{k+1}^n(x), F_k^n(x) = x + \gamma_{k+1}f_k^n(x)$  for any  $x \in \mathbb{R}^d$ . Then we have for any  $n \in \mathbb{N}$  and  $k \in \{0, \dots, N-1\}$

$$
B _ {k + 1} ^ {n} = \arg \min  _ {\mathrm {B} \in \mathrm {L} ^ {2} \left(\mathbb {R} ^ {d}, \mathbb {R} ^ {d}\right)} \mathbb {E} p _ {k, k + 1} ^ {n} [ \| \mathrm {B} (X _ {k + 1}) - (X _ {k + 1} + F _ {k} ^ {n} (X _ {k}) - F _ {k} ^ {n} (X _ {k + 1})) \| ^ {2} ], \tag {12}
$$

$$
F _ {k} ^ {n + 1} = \arg \min  _ {\mathrm {F} \in \mathrm {L} ^ {2} \left(\mathbb {R} ^ {d}, \mathbb {R} ^ {d}\right)} \mathbb {E} _ {q _ {k, k + 1} ^ {n}} [ \| \mathrm {F} (X _ {k}) - (X _ {k} + B _ {k + 1} ^ {n} (X _ {k + 1}) - B _ {k + 1} ^ {n} (X _ {k})) \| ^ {2} ]. \tag {13}
$$

Proposition 3 shows how one can recursively approximate  $B_{k+1}^n$  and  $F_k^{n+1}$ . In practice, we use neural networks  $B_{\beta^n}(k,x) \approx B_k^n(x)$  and  $F_{\alpha^n}(k,x) \approx F_k^n(x)$ .

# Algorithm 1 Diffusion Schrödinger Bridge

1: for  $n \in \{0, \dots, L\}$  do  
2: while not converged do  
3: Sample  $\{X_k^j\}_{k,j=0}^{N,M}$ , where  $X_0^j \sim p_{\mathrm{data}}$ , and

$$
X _ {k + 1} ^ {j} = F _ {\alpha^ {n}} (k, X _ {k} ^ {j}) + \sqrt {2 \gamma_ {k + 1}} Z _ {k + 1} ^ {j}
$$

4: Compute  $\hat{\ell}_n^b (\beta^n)$  approximating (12)

5:  $\beta^n\gets \operatorname {Gradient}\operatorname {Step}(\hat{\ell}_n^b (\beta^n))$  
6: end while  
7: while not converged do  
8: Sample  $\{X_k^j\}_{k,j=0}^{N,M}$ , where  $X_N^j \sim p_{\text{prior}}$ , and

$$
X _ {k - 1} ^ {j} = B _ {\beta^ {n}} (k, X _ {k} ^ {j}) + \sqrt {2 \gamma_ {k}} \tilde {Z} _ {k} ^ {j}
$$

9: Compute  $\hat{\ell}_{n + 1}^{f}(\alpha^{n + 1})$  approximating (13)

10:  $\alpha^{n + 1}\gets \mathrm{Gradient~Step}(\hat{\ell}_{n + 1}^f (\alpha^{n + 1}))$  
11: end while  
12: end for  
13: Output:  $(\alpha^{L + 1},\beta^L)$

Network parameters  $\alpha^n,\beta^n$  are learnt through gradient descent to minimize empirical versions of the sum over  $k$  of the loss functions given by (12) and (13) computed using  $M$  samples and denoted  $\hat{\ell}_n^b (\beta)$  and  $\hat{\ell}_{n + 1}^{f}(\alpha)$ . The resulting algorithm approximating  $L$  IPF iterations is called Diffusion Schrodinger Bridge (DSB) and is summarized in Algorithm 1 with  $Z_{k}^{j},\tilde{Z}_{k}^{j}\stackrel {\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,\mathbf{I})$ , see Figure 1 for an illustration. This algorithm is initialized using the reference dynamics  $f_{\alpha^0}(k,x) = f(x)$ . Once  $\beta^L$  is learnt we can easily approximately sample from  $p_{\mathrm{data}}$  by sampling  $X_{N}\sim p_{\mathrm{prior}}$  and then using  $X_{k - 1} = B_{\beta^L}(k,X_k) + \sqrt{2\gamma_k} Z_k$  with  $Z_{k}\stackrel {\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,\mathbf{I})$ . The resulting samples  $X_0$  will be approximately distributed from  $p_{\mathrm{data}}$ .

Although the DSB requires learning a sequence of network parameters,  $\alpha^n,\overline{\beta^n}$ , fewer diffusion steps are needed compared to standard SGM and hence the networks are quicker to train. In addition, as detailed in Section S9, the initial  $\beta^0$  may be trained efficiently similar to previous SGM methods and then  $\alpha^{n + 1},\beta^{n + 1}$  are refinements of  $\alpha^n,\beta^n$  hence may be fine-tuned from previous iterations.

# 3.4 Convergence of Iterative Proportional Fitting

In this section, we investigate the theoretical properties of IPF. When the state-space is discrete and finite (Franklin and Lorenz, 1989; Peyré and Cuturi, 2019) or in the case where  $p_{\mathrm{data}}$  and  $p_{\mathrm{prior}}$

are compactly supported (Chen et al., 2016), IPF converges at a geometric rate w.r.t. the Hilbert-Birkhoff metric, see Lemmens and Nussbaum (2014) for a definition. Other than recent work by Leger (2020), only qualitative results exist in the general case where  $p_{\mathrm{data}}$  or  $p_{\mathrm{prior}}$  is not compactly supported (Ruschendorf et al., 1995; Ruschendorf and Thomsen, 1993). We establish here quantitative convergence of IPF in this non-compact setting as well as novel monotonicity results. We require only the following mild assumption.

A1.  $p_N, p_{\text{prior}} > 0, \mathrm{H}(p_{\text{prior}}) < +\infty, \int_{\mathbb{R}^d} |\log(p_{N|0}(x_N|x_0))| p_{\text{data}}(x_0)p_{\text{prior}}(x_N) \, \mathrm{d}x_0 \, \mathrm{d}x_N < +\infty.$

Assumption A1 is satisfied in all of our experimental settings. We recall that for  $\mu, \nu \in \mathcal{P}(\mathsf{E})$  with  $(\mathsf{E}, \mathcal{E})$  a measurable space, the Jeffreys divergence is given by  $\mathrm{J}(\mu, \nu) = \mathrm{KL}(\mu|\nu) + \mathrm{KL}(\nu|\mu)$ .

Proposition 4. Assume A1. Then  $(\pi^n)_{n\in \mathbb{N}}$  is well-defined and for any  $n\geq 1$  we have

$$
\mathrm {K L} (\pi^ {n + 1} | \pi^ {n}) \leq \mathrm {K L} (\pi^ {n - 1} | \pi^ {n}), \quad \mathrm {K L} (\pi^ {n} | \pi^ {n + 1}) \leq \mathrm {K L} (\pi^ {n} | \pi^ {n - 1}).
$$

In addition,  $(\| \pi^{n + 1} - \pi^n \|_{\mathrm{TV}})_{n\in \mathbb{N}}$  and  $(\mathrm{J}(\pi^{n + 1},\pi^n))_{n\in \mathbb{N}}$  are non-increasing. Finally, we have  $\lim_{n\to +\infty}n\left\{\mathrm{KL}(\pi_0^n |p_{\mathrm{data}}) + \mathrm{KL}(\pi_N^n |p_{\mathrm{prior}})\right\} = 0.$

A more general result with additional monotonicity properties is given in Section S6. Under similar assumptions, Léger (2020, Corollary 1) derives that  $\mathrm{KL}(\pi_0^n | p_0) \leq C / n$  with  $C \geq 0$  using a Bregman divergence gradient descent perspective. In contrast, our proof relies only on tools from information geometry. In addition, we improve the convergence rate and show that  $(\pi^n)_{n \in \mathbb{N}}$  converges in total variation towards  $\pi^{\infty}$ , i.e. we not only obtain the convergence of the marginals but the convergence of the joint distribution. Under restrictive conditions on  $p_{\mathrm{data}}$  and  $p_{\mathrm{prior}}$ , Ruschendorf et al. (1995) show that  $\pi^{\infty}$  is the Schrödinger bridge. In the following proposition we avoid this assumption using results on automorphisms of measures (Beurling, 1960).

Proposition 5. Assume A1. Then there exists a solution  $\pi^{\star} \in \mathcal{P}_{N+1}$  to the SB problem and we have  $\lim_{n \to +\infty} \| \pi^n - \pi^\infty \|_{\mathrm{TV}} = 0$  with  $\pi^\infty \in \mathcal{P}_{N+1}$ . Let  $h = p_{0,N} / (p_0 \otimes p_N)$  and assume that  $h \in \mathrm{C}((\mathbb{R}^d)^2, (0, +\infty))$  and that there exist  $\Phi_0, \Phi_N \in \mathrm{C}(\mathbb{R}^d, (0, +\infty))$  such that

$$
\int_ {\mathbb {R} ^ {d} \times \mathbb {R} ^ {d}} \left(\left| \log h \left(x _ {0}, x _ {N}\right) \right| + \left| \log \Phi_ {0} \left(x _ {0}\right) \right| + \left| \log \Phi_ {N} \left(x _ {N}\right) \right|\right) p _ {\text {d a t a}} \left(x _ {0}\right) p _ {\text {p r i o r}} \left(x _ {N}\right) d x _ {0} d x _ {N} <   + \infty ,
$$

with  $h(x_0,x_N)\leq \Phi_0(x_0)\Phi_N(x_N)$  . If  $p$  is absolutely continuous w.r.t.  $\pi^{\infty}$  then  $\pi^{\infty} = \pi^{\star}$

Proposition 5 extends previous IPF convergence results without the assumption that the mapping  $h$  is lower bounded, see Ruschendorf et al. (1995); Chen et al. (2016). Our assumption on  $h$  can be relaxed and replaced by a tighter condition on  $\pi^{\infty}$ , see Section S6.2. Proposition 4 suggests a convergence of order  $o(n)$  for the IPF in the non compact setting. However, in some situations, we recover geometric convergence rates with explicit dependency w.r.t. the problem constants, see Section S7.

# 3.5 Continuous-time IPF

We describe an IPF algorithm for solving SB problems in continuous-time. We show that DSB proposed in Algorithm 1 can be seen as a discretization of this IPF. Given a reference measure  $\mathbb{P} \in \mathcal{P}(\mathcal{C})$ , the continuous formulation of the SB involves solving the following problem

$$
\Pi^ {\star} = \arg \min  \left\{\mathrm {K L} (\Pi | \mathbb {P}): \Pi \in \mathcal {P} (\mathcal {C}), \Pi_ {0} = p _ {\text {d a t a}}, \Pi_ {T} = p _ {\text {p r i o r}} \right\}, \quad T = \sum_ {k = 0} ^ {N - 1} \gamma_ {k + 1}.
$$

Similarly to (11), we define the IPF  $(\Pi^n)_{n\in \mathbb{N}}$  with  $\Pi^0 = \mathbb{P}$  associated with (6) and for any  $n\in \mathbb{N}$

$$
\Pi^ {2 n + 1} = \arg \min  \left\{\mathrm {K L} (\Pi | \Pi^ {2 n}): \Pi \in \mathcal {P} (\mathcal {C}), \Pi_ {T} = p _ {\text {p r i o r}} \right\},
$$

$$
\Pi^ {2 n + 2} = \arg \min  \left\{\mathrm {K L} (\Pi | \Pi^ {2 n + 1}): \Pi \in \mathcal {P} (\mathcal {C}), \Pi_ {0} = p _ {\text {d a t a}} \right\}.
$$

One can show that for any  $n \in \mathbb{N}$ ,  $\Pi^n = \pi^{s,n} \mathbb{P}_{|0,T}$ , with  $(\pi^{s,n})_{n \in \mathbb{N}}$  the IPF for the static SB problem. In particular, Proposition 4 and Proposition 5 extend to the continuous IPF framework. In what follows, for any  $\mathbb{P} \in \mathcal{P}(\mathcal{C})$ , we define  $\mathbb{P}^R$  as the reverse-time measure, i.e. for any  $\mathsf{A} \in \mathcal{B}(\mathcal{C})$  we have  $\mathbb{P}^R(\mathsf{A}) = \mathbb{P}(\mathsf{A}^R)$  where  $\mathsf{A}^R = \{t \mapsto \omega(T - t) : \omega \in \mathsf{A}\}$ . The following result is the continuous counterpart of Proposition 2 and states that each IPF iteration is associated with a diffusion, showing that DSB can be seen as a discretization of the continuous IPF.

Proposition 6. Assume A1 and that there exist  $\mathbb{M} \in \mathcal{P}(\mathcal{C})$ ,  $U \in \mathrm{C}^1(\mathbb{R}^d, \mathbb{R})$ ,  $C \geq 0$  such that for any  $n \in \mathbb{N}$ ,  $x \in \mathbb{R}^d$ ,  $\mathrm{KL}(\Pi^n|\mathbb{M}) < +\infty$ ,  $\langle x, \nabla U(x) \rangle \geq -C(1 + \|x\|^2)$  and  $\mathbb{M}$  is associated with

$$
\mathrm {d} \mathbf {X} _ {t} = - \nabla U (\mathbf {X} _ {t}) \mathrm {d} t + \sqrt {2} \mathrm {d} \mathbf {B} _ {t}, \tag {14}
$$

with  $\mathbf{X}_0$  distributed according to the invariant distribution of (14). Then, for any  $n\in \mathbb{N}$  we have:

(a)  $(\Pi^{2n + 1})^R$  is associated with  $\mathrm{d}\mathbf{Y}_t^{2n + 1} = b_{T - t}^n (\mathbf{Y}_t^{2n + 1})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_t$  with  $\mathbf{X}_0^{2n + 1}\sim p_{\mathrm{prior}}$ ;  
(b)  $\Pi^{2n + 2}$  is associated with  $\mathrm{d}\mathbf{X}_t^{2n + 2} = f_t^{n + 1}(\mathbf{X}_t^{2n + 2})\mathrm{d}t + \sqrt{2}\mathrm{d}\mathbf{B}_t$  with  $\mathbf{X}_0^{2n + 2}\sim p_{\mathrm{data}}$

where for any  $n \in \mathbb{N}$ ,  $t \in [0, T]$  and  $x \in \mathbb{R}^d$ ,  $b_t^n(x) = -f_t^n(x) + 2\nabla \log p_t^n(x)$ ,  $f_t^{n+1}(x) = -b_t^n(x) + 2\nabla \log q_t^n(x)$ , with  $f_t^0(x) = f(x)$ , see (6), and  $p_t^n$ ,  $q_t^n$  the densities of  $\Pi_t^{2n}$  and  $\Pi_t^{2n+1}$ .

# 4 Experiments

Two dimensional toy experiments. We evaluate the validity of our approach on toy two dimensional examples. Contrary to existing SGM approaches we do not require that the number of steps is large enough for  $p_N \approx p_{\mathrm{prior}}$  to hold. We use a fully connected network with positional encoding (Vaswani et al., 2017) to approximate  $B_k^n$  and  $F_k^n$ , see Section S10.1 for details about our implementation. Animated plots of the DSB iterations may be found online on our project webpage<sup>3</sup>. In Figure 2, we illustrate the benefits of DSB over classical SGM. We fix  $f(x) = \alpha x$  and choose  $p_{\mathrm{prior}} = \mathcal{N}(0, \sigma_{\mathrm{data}}^2 \mathbf{I})$ , hence  $\alpha = 1 / \sigma_{\mathrm{data}}^2$  where  $\sigma_{\mathrm{data}}^2$  is the variance of the dataset. We let  $N = 20$  and  $\gamma_k = 0.01$ , i.e.  $T = 0.2$ . Since  $T$  is low, the noisy process

![](images/ad69c35dd11fbd6f43bf4980742f3ec51fdae09544a73499fa77f01001019625.jpg)  
Data distribution  
DSB Iteration 1

![](images/5d63be11d7627411a7670d55dba839f8bef8deb4ffb42018cd4bd5b2960faa25.jpg)  
DSB Iteration 20

![](images/301bb06f7fad09092dc23f3af7408ca396017022deb079cdd24b05925e6d1fa9.jpg)  
Figure 2: Data distributions  $p_{\mathrm{data}}$  vs distribution at final time  $T = 0.2$  after 1 and 20 DSB iterations.

does not satisfy  $p_N \approx p_{\mathrm{prior}}$  and the reverse-time process obtained after the first DSB iteration (corresponding to original SGM methods) does not yield a satisfactory generative model. However, multiple iterations of DSB improve the quality of the synthesis. We emphasize that even though in theory Schrödinger bridges allow for  $T$  to be arbitrary small, we observe that decreasing values of  $T$  require an increasing number of DSB iterations to obtain valid generative models. We discuss this trade-off and present additional experiments in Section S10.1.

Generative modeling. DSB is the first practical algorithm for approximating the solution to the SB problem in high dimension. Whilst our implementation does not yet compete with state-of-the-art methods, we show promising results with fewer diffusion steps compared to initial SGMs (Song and Ermon, 2019) and demonstrate its performance on MNIST (LeCun and Cortes, 2010) and CelebA (Liu et al., 2015). A reduced U-net architecture based on Nichol and Dhariwal (2021) is used to approximate  $B_{k}^{n}$  and  $F_{k}^{n}$ . Further details are given in Section S10.2. Our method is validated on downscaled CelebA in Figure 3. Figure 4 illustrates qualitative improvement over 8 DSB iterations with as few as  $N = 12$  diffusion steps. Note, as shown in the Section S10.2, we obtain better results with higher  $N$  yet still significantly fewer steps than in the original SGM procedures (Song and Ermon, 2020, 2019) which use  $N = 100$ . Figure 4 also shows good diversity of generated samples (red) and coverage of the original dataset (blue) as shown by the two dimensional representation in the latent space of a pre-trained Variational Auto-Encoder (VAE). Figure 5 illustrates how the sample quality, measured quantitatively in terms of Fréchet Inception Distance (FID) (Heusel et al., 2017), improves with the number of DSB iterations for various numbers of steps  $N$ .

![](images/856490ba33352b68973e9a65fe2d8a3fbe6d483f39dc2a0a37ebd0a1090c7fbf.jpg)  
$t = 0$  
Figure 3: Generative model for CelebA  $32 \times 32$  after 10 DSB iterations with  $N = 50$  ( $T = 0.63$ )

![](images/65562759366387ee812160ebef26d8b5206eb7ed743796816824a4151ef6fd69.jpg)  
$t = 0.31$

![](images/a9102bf0df4834bd1ff8ab367fe47fb2b29d9a0f7d14012cc02f572f086a7454.jpg)  
$t = 0.60$

![](images/33f3824b1b0187bcd774ba26abb27e8b88aec5da3d6318bedef9a84b0ba881bc.jpg)  
$t = 0.63$

![](images/f0c2f312dacd28f52c2551f92d72045fe913349391c1ca0ce7bf63483e05ef27.jpg)  
DSB 1

![](images/21b9fe1694b4d73af78b891702ef6416074cc266e562594bbf230294359ed669.jpg)

![](images/65fcb4c38d906372000f067a2d130d379322013ba63552b45cdabcd058cac9eb.jpg)  
DSB 8

![](images/eb9feb12351f8f68be2090926f803d0e2d54c4b5929f5b2e8b428b2bb3ffe046.jpg)  
Figure 4: Generated samples  $(N = 12)$  and two-dimensional visualization of samples (red) compared to original MNIST data (blue) using pre-trained VAE

![](images/e28566b42bd32b7c0647ab90c1ee6a1af5c35bae679a9f7c8ad25e6ac1db25cf.jpg)  
Figure 5: FID vs DSB Iterations. Green dashed line: baseline FID obtained with 1 DSB iteration and  $N = 40$

Datasets interpolation. Schrödinger bridges not only allow us to reduce the number of iterations in SGM methods but also enable flexibility in the choice of the prior density  $p_{\mathrm{prior}}$  which is not necessarily Gaussian contrary to previous works on SGM. In particular, our approach is still valid if  $p_{\mathrm{prior}}$  is any other data distribution  $p_{\mathrm{data}}^{\prime}$ . In this case DSB converges towards a bridge between  $p_{\mathrm{data}}$  and  $p_{\mathrm{data}}^{\prime}$ , see Figure 6. These experiments pave the way towards high-dimensional optimal transport between arbitrary data distributions.

![](images/a2a548639a37322714de4c68ad2f0facab9ec7744ca5e146cc32157b1c493660.jpg)  
Figure 6: First row: Swiss-roll to S-curve (2D). Iteration 9 of DSB with  $T = 1$  ( $N = 50$ ). From left to right:  $t = 0, 0.4, 0.6, 1$ . Second row: EMNIST (Cohen et al., 2017) to MNIST. Iteration 10 of DSB with  $T = 1.5$  ( $N = 30$ ). From left to right:  $t = 0, 0.4, 1.25, 1.5$ .

# 5 Discussion

Score-based generative modeling (SGM) may be viewed as the first stage in approximating a solution to the Schrödinger bridge problem. Through this interpretation we have developed a novel methodology, the Diffusion Schrödinger Bridge (DSB), that extends initial SGM approaches and allows one to perform generative modeling with fewer diffusion steps. DSB complements recent techniques used to speed up existing SGM algorithms which rely on either different noise schedules (Nichol and Dhariwal, 2021; San-Roman et al., 2021) or knowledge distillation (Luhman and Luhman, 2021). Additionally, as the solution of the Schrödinger problem is a diffusion, it is possible as in Song et al. (2021, Section 4.3) to obtain an equivalent neural ordinary differential equation that admits the same marginals as the diffusion but enables exact likelihood computation. From a theoretical point of view, we have provided the first convergence result for SGM methods and derived new state-of-the-art convergence bounds for IPF as well as novel monotonicity results. We have demonstrated DSB on generative modeling and interpolation tasks. Finally, while motivated by generative modeling, DSB is much more widely applicable as it can be thought of as the continuous state-space counterpart of the celebrated Sinkhorn algorithm (Cuturi, 2013; Peyré and Cuturi, 2019). For example, DSB could be used to solve multi-marginal Schrödinger bridges problems (Di Marino and Gerolin, 2020), compute Wasserstein barycenters, find the minimizers of entropy-regularized Gromov-Wasserstein problems (Mémoli, 2011) or perform domain adaptation in continuous state-spaces.

# References

Bernton, E., Heng, J., Doucet, A., and Jacob, P. E. (2019). Schrödinger bridge samplers. arXiv preprint arXiv:1912.13170.  
Beurling, A. (1960). An automorphism of product measures. Annals of Mathematics, 72(1):189-200.  
Cai, R., Yang, G., Averbuch-Elor, H., Hao, Z., Belongie, S., Snavely, N., and Hariharan, B. (2020). Learning gradient fields for shape generation. European Conference on Computer Vision.  
Cattiaux, P., Conforti, G., Gentil, I., and Léonard, C. (2021). Time reversal of diffusion processes under a finite entropy condition. arXiv preprint arXiv:2104.07708.  
Chen, N., Zhang, Y., Zen, H., Weiss, R. J., Norouzi, M., and Chan, W. (2021a). Wavegrad: Estimating gradients for waveform generation. International Conference on Learning Representations.  
Chen, Y., Georgiou, T., and Pavon, M. (2016). Entropic and displacement interpolation: a computational approach using the Hilbert metric. SIAM Journal on Applied Mathematics, 76(6):2375-2396.  
Chen, Y., Georgiou, T. T., and Pavon, M. (2021b). Optimal transport in systems and control. Annual Review of Control, Robotics, and Autonomous Systems, 4.  
Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. arXiv preprint arXiv:1702.05373.  
Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems.  
Dessein, A., Papadakis, N., and Deledalle, C.-A. (2017). Parameter estimation in finite mixture models by regularized optimal transport: A unified framework for hard and soft clustering. arXiv preprint arXiv:1711.04366.  
Dhariwal, P. and Nichol, A. (2021). Diffusion models beat GAN on image synthesis. arXiv preprint arXiv:2105.05233.  
Di Marino, S. and Gerolin, A. (2020). An optimal transport approach for the Schrödinger bridge problem and convergence of Sinkhorn algorithm. Journal of Scientific Computing, 85(2):1-28.  
Durkan, C. and Song, Y. (2021). On maximum likelihood training of score-based generative models. arXiv preprint arXiv:2101.09258.  
Föllmer, H. (1985). An entropy approach to the time reversal of diffusion processes. In Stochastic Differential Systems: Filtering and Control, pages 156-163. Springer.  
Föllmer, H. (1988). Random fields and diffusion processes. In École d'Été de Probabilités de Saint-Flour XV-XVII, 1985-87, pages 101-203. Springer.  
Fortet, R. (1940). Résolution d'un système d'équations de M. Schrödinger. Journal de Mathématiques Pures et Appliqués, 1:83-105.  
Franklin, J. and Lorenz, J. (1989). On the scaling of multidimensional matrices. Linear Algebra and Its Applications, 114:717-735.  
Gao, R., Song, Y., Poole, B., Wu, Y. N., and Kingma, D. P. (2020). Learning energy-based models by diffusion recovery likelihood. arXiv preprint arXiv:2012.08125.  
Genevay, A., Peyre, G., and Cuturi, M. (2018). Learning generative models with Sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics.  
Haussmann, U. G. and Pardoux, E. (1986). Time reversal of diffusions. The Annals of Probability, 14(4):1188-1205.  
Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. arXiv preprint arXiv:1706.08500.

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems.  
Hoogeboom, E., Nielsen, D., Jaini, P., Forre, P., and Welling, M. (2021). Argmax flows and multinomial diffusion: Towards non-autoregressive language models. arXiv preprint arXiv:2102.05379.  
Hyvarinen, A. and Dayan, P. (2005). Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(4).  
Jolicoeur-Martineau, A., Piché-Taillefer, R., Tachet des Combes, R., and Mitliagkas, I. (2021). Adversarial score matching and improved sampling for image generation. International Conference on Learning Representations.  
Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. (2021). Diffwave: A versatile diffusion model for audio synthesis. International Conference on Learning Representations.  
Kullback, S. (1968). Probability densities with given marginals. The Annals of Mathematical Statistics, 39(4):1236-1243.  
LeCun, Y. and Cortes, C. (2010). MNIST handwritten digit database.  
Léger, F. (2020). A gradient descent perspective on Sinkhorn. Applied Mathematics & Optimization, pages 1-13.  
Lemmens, B. and Nussbaum, R. D. (2014). Birkhoff's version of Hilbert's metric and its applications in analysis. Handbook of Hilbert Geometry, pages 275-303.  
Léonard, C. (2014a). Some properties of path measures. In Séminaire de Probabilités XLVI, pages 207-230. Springer.  
Léonard, C. (2014b). A survey of the Schrödinger problem and some of its connections with optimal transport. Discrete & Continuous Dynamical Systems-A, 34(4):1533-1574.  
Liu, Z., Luo, P., Wang, X., and Tang, X. (2015). Deep learning face attributes in the wild. In International Conference on Computer Vision.  
Luhman, E. and Luhman, T. (2021). Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388.  
Luhman, T. and Luhman, E. (2020). Diffusion models for handwriting generation. arXiv preprint arXiv:2011.06704.  
Mémoli, F. (2011). Gromov-Wasserstein distances and the metric approach to object matching. Foundations of Computational Mathematics, 11(4):417-487.  
Mikami, T. (2004). Monge's problem with a quadratic cost by the zero-noise limit of  $h$ -path processes. Probability Theory and Related Fields, 129(2):245-260.  
Nichol, A. and Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. arXiv preprint arXiv:2102.09672.  
Niu, C., Song, Y., Song, J., Zhao, S., Grover, A., and Ermon, S. (2020). Permutation invariant graph generation via score-based generative modeling. In International Conference on Artificial Intelligence and Statistics.  
Pavon, M., Trigila, G., and Tabak, E. G. (2021). The data-driven Schrödinger bridge. Communications on Pure and Applied Mathematics, 74:1545-1573.  
Peyre, G. and Cuturi, M. (2019). Computational optimal transport. Foundations and Trends® in Machine Learning, 11(5-6):355-607.  
Popov, V., Vovk, I., Gogoryan, V., Sadekova, T., and Kudinov, M. (2021). Grad-tts: A diffusion probabilistic model for text-to-speech. arXiv preprint arXiv:2105.06337.  
Reich, S. (2019). Data assimilation: the Schrödinger perspective. Acta Numerica, 28:635-711.

Ruschendorf, L. et al. (1995). Convergence of the iterative proportional fitting procedure. The Annals of Statistics, 23(4):1160-1174.  
Ruschendorf, L. and Thomsen, W. (1993). Note on the Schrödinger equation and i-projections. Statistics & Probability letters, 17(5):369-375.  
Saharia, C., Ho, J., Chan, W., Salimans, T., Fleet, D. J., and Norouzi, M. (2021). Image superresolution via iterative refinement. arXiv preprint arXiv:2104.07636.  
San-Roman, R., Nachmani, E., and Wolf, L. (2021). Noise estimation for generative diffusion models. arXiv preprint arXiv:2104.02600.  
Schrodinger, E. (1932). Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique. Annales de l'Institut Henri Poincaré, 2(4):269-310.  
Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning.  
Song, J., Meng, C., and Ermon, S. (2020). Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.  
Song, Y. and Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems.  
Song, Y. and Ermon, S. (2020). Improved techniques for training score-based generative models. Advances in Neural Information Processing Systems.  
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations.  
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.  
Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661-1674.
