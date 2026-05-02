# An analysis of Ermakov-Zolotukhin quadrature using kernels

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We study a quadrature, proposed by Ermakov and Zolotukhin in the sixties, through the lens of kernel methods. The nodes of this quadrature rule follow the distribution of a determinantal point process, while the weights are defined through a linear system, similarly to the optimal kernel quadrature. In this work, we show how these two classes of quadrature are related, and we prove a tractable formula of the expected value of the squared worst-case integration error on the unit ball of an RKHS of the former quadrature. In particular, this formula involves the eigenvalues of the corresponding kernel and leads to improving on the existing theoretical guarantees of the optimal kernel quadrature with determinantal point processes.

# 1 Introduction

Integrals appear in many scientific fields as quantities of interest interesting per se. For example, in statistics, they represent expectations [25], while in mathematical finance, they represent the prices of financial products [16]. Unfortunately, integrals that can be written in closed form are exceptional. In general, their values are only known through approximations. For this reason, numerical integration is at the heart of many tasks in applied mathematics and statistics. Among all the possible approximation schemes, quadratures are the most practical since they approximate the integral of a function by a finite mixture of its evaluations. In this work, we focus on quadrature rules that take the form

$$
\int_ {\mathcal {X}} f (x) g (x) \mathrm {d} \omega (x) \approx \sum_ {i \in [ N ]} w _ {i} f \left(x _ {i}\right), \tag {1}
$$

where the nodes  $x_{i}$  are independent of  $f$  and  $g$ , while the weights  $w_{i}$  depend only on  $g$ . The nodes and the weights of a quadrature may be seen as degrees of freedom that the practitioner may tune in order to achieve a given level of approximation error. The design of quadratures gave birth to a rich literature from Gaussian quadrature [14] to Monte Carlo methods [23] to quadratures based on determinantal point processes (DPPs) [1]. These latter form a large class of probabilistic models of repulsive random subsets that make numerical integration possible in a variety of domains with strong theoretical guarantees. In particular, CLTs with asymptotic convergence rates that scale better than the typical Monte Carlo rate  $\mathcal{O}(N^{-1/2})$  were proven for several DPP based quadratures: when the integrand is a  $\mathcal{C}^1$  function [1] or even when the integrand is non-differentiable [10]. Moreover, it is possible to design quadrature rules based on DPPs with non-asymptotic guarantees and with rates of convergence that adapt to the smoothness of the integrand. This is the case of the quadrature proposed by Ermakov and Zolotukhin in [13], and the optimal kernel quadrature [3].

In this work, we study the quadrature rule proposed by Ermakov and Zolotukhin (EZQ) through the lens of kernel methods. We start by comparing the weights of EZQ to the weights of the optimal kernel quadrature (OKQ), and we prove that they both belong to a broader class of quadrature rules

that we call kernel based interpolation quadrature. Then, we study the approximation quality of EZQ in reproducing kernel Hilbert spaces. This is done by proving a general tractable formula of the expected value of the squared worst-case integration error for functions that belong to the unit ball of an RKHS when the nodes follow the distribution of a determinantal point process. This formula involves principally the eigenvalues of the integral operator, and converges to 0 at a slightly slower rate than the optimal rate. Interestingly, this analysis yields a better upper bound for the optimal kernel quadrature with DPPs proposed initially in [3]. Comparably to the theoretical guarantees given in [13], our theoretical guarantees are independent of the choice of the test function. This facilitates the comparison of EZQ with other quadratures such as OKQ.

The rest of the article is organized as follows. Section 2 reviews the work of [13] and recall key concepts on kernel based quadrature. In Section 3, we present the main results of this work and their consequences. A sketch of the proof of the main theorem is given in Section 4. We illustrate the theoretical results by numerical experiments in Section 5. Finally, we give a conclusion in Section 6.

Notation and assumptions. We use the notation  $\mathbb{N}^* = \mathbb{N} \setminus \{0\}$ . We denote by  $\omega$  a Borel measure supported on  $\mathcal{X}$ , and we denote by  $\mathcal{L}_2(\omega)$  the Hilbert space of square integrable real-valued functions on  $\mathcal{X}$  with respect to  $\omega$ , equipped with the inner product  $\langle \cdot, \cdot \rangle_{\omega}$ , and the associated norm  $\| \cdot \|_{\omega}$ . For  $N \in \mathbb{N}^*$ , we denote by  $\omega^{\otimes N}$  the tensor product of  $\omega$  defined on  $\mathcal{X}^N$ . Moreover, we denote by  $\mathcal{F}$  the RKHS associated to the kernel  $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$  that we assume to be continuous and satisfying the condition  $\int_{\mathcal{X}} k(x, x) \mathrm{d}\omega(x) < +\infty$ . In particular, we assume the Mercer decomposition

$$
k (x, y) = \sum_ {m \in \mathbb {N} ^ {*}} \sigma_ {m} \phi_ {m} (x) \phi_ {m} (y), \tag {2}
$$

to hold, where the convergence is pointwise, and  $\sigma_{m}$  and  $\phi_{m}$  are the corresponding eigenvalues and eigenfunctions of the integral operator  $\pmb{\Sigma}$  defined for  $f\in \mathcal{L}_2(\omega)$  by

$$
\boldsymbol {\Sigma} f (\cdot) = \int_ {\mathcal {X}} k (\cdot , y) f (y) \mathrm {d} \omega (y). \tag {3}
$$

We assume that the sequence  $\sigma = (\sigma_{m})_{m\in \mathbb{N}^{*}}$  is non-increasing and its elements are non-vanishing so that the corresponding eigenfunctions  $\phi_m$  can be taken to be continuous. We precise that the  $\phi_{m}$  are normalized:  $\| \phi_m\|_\omega = 1$  for  $m\in \mathbb{N}^*$ . In particular,  $(\phi_{m})_{m\in \mathbb{N}^{*}}$  is an o.n.b. of  $\mathcal{L}_2(\omega)$ , and every element  $f\in \mathcal{F}$  satisfies

$$
\sum_ {m \in \mathbb {N} ^ {*}} \frac {\langle f , \phi_ {m} \rangle_ {\omega} ^ {2}}{\sigma_ {m}} <   + \infty . \tag {4}
$$

Moreover, for every  $N \in \mathbb{N}^*$ , we denote by  $\mathcal{E}_N$  the eigen-subspace of  $\mathcal{L}(\omega)$  spanned by  $\phi_1, \ldots, \phi_N$ . For any kernel  $\kappa: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ , and for  $\pmb{x} \in \mathcal{X}^N$ , we define the kernel matrix  $\pmb{\kappa}(\pmb{x}) := (\kappa(x_i, x_j))_{i,j \in [N]} \in \mathbb{R}^{N \times N}$ . Finally, we denote in bold fonts the corresponding kernel matrices:  $\pmb{K}(\pmb{x})$  for the kernel  $k$ ,  $\pmb{K}_N(\pmb{x})$  for the kernel  $k_N$ ,  $\pmb{K}_N^\perp(\pmb{x})$  for the kernel  $k_N^\perp$ ,  $\pmb{\kappa}(\pmb{x})$  for the kernel  $\kappa$ . Similarly, for any function  $\mu: \mathcal{X} \to \mathbb{R}$  and for  $\pmb{x} \in \mathcal{X}^N$ , we define the vector of evaluations  $\mu(\pmb{x}) := (\mu(x_i))_{i \in [N]} \in \mathbb{R}^N$ .

# 2 Related work

In the section, we review some results that are relevant to our contribution.

# 2.1 Ermakov-Zolotukhin quadrature

The quadrature rule proposed by Ermakov and Zolotukhin in [13] deals with integrals that write as

$$
\int_ {\mathcal {X}} f (x) \phi_ {m} (x) \mathrm {d} \omega (x), \tag {5}
$$

where  $f \in \mathcal{L}_2(\omega)$ , and  $(\phi_m)_{m \in \mathbb{N}^*}$  is an orthonormal family with respect to the measure  $\omega$ . Its construction goes as follows. For  $N \in \mathbb{N}^*$ , let  $\pmb{x} \in \mathcal{X}^N$  such that the matrix  $\Phi_N(\pmb{x}) := (\phi_n(x_i))_{(n,i) \in [N] \times [N]}$  is non-singular. For  $n \in [N]$ , define

$$
I ^ {\mathrm {E Z}, n} (f) = \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, n} f \left(x _ {i}\right), \tag {6}
$$

where  $\hat{\pmb{w}}^{\mathrm{EZ},n} \coloneqq (\hat{w}_i^{\mathrm{EZ},n})_{i\in [N]} \in \mathbb{R}^N$  is given by  $\hat{\pmb{w}}^{\mathrm{EZ},n} = \Phi_N(\pmb{x})^{-1}\pmb{e}_n$ , with  $\pmb{e}_n$  is the  $n$ -th element of the canonical basis of  $\mathbb{R}^{N-1}$ . We can prove easily that for every  $f \in \operatorname{Span}(\phi_n)_{n\in [N]}$ , this quadrature is exact

$$
\sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, n} f (x _ {i}) = \int_ {\mathcal {X}} f (x) \phi_ {n} (x) \mathrm {d} \omega (x). \tag {7}
$$

Now, if  $f \notin \operatorname{Span}(\phi_n)_{n \in [N]}$ , the authors studied the expected value and the variance of  $I^{\mathrm{EZ}, n}(f)$  when  $\boldsymbol{x} = (x_1, \ldots, x_N)$  is taken to be a random variable in  $\mathcal{X}^N$  that follows the distribution of density

$$
p _ {\mathrm {D P P}} \left(x _ {1}, \dots , x _ {N}\right) := \frac {1}{N !} \operatorname {D e t} ^ {2} \Phi_ {N} (\boldsymbol {x}), \tag {8}
$$

with respect to the product measure  $\omega^{\otimes N}$  defined on  $\mathcal{X}^N$ . As it was observed in [15], the nodes of the quadrature follow the distribution of the determinantal point process of reference measure  $\omega$  and marginal kernel  $\kappa_N$  defined by  $\kappa_N(x,y) = \sum_{n\in [N]}\phi_n(x)\phi_n(y)$ . We refer to [19] for further details on determinantal point processes. Now, we recall the main result of [13].

Theorem 1. Let  $\mathbf{x}$  be a random subset of  $\mathcal{X}$  that follows the distribution of DPP of kernel  $\kappa_N$  and reference measure  $\omega$ . Let  $f \in \mathcal{L}_2(\omega)$ , and  $n \in [N]$ . Then

$$
\mathbb {E} _ {\mathrm {D P P}} I ^ {\mathrm {E Z}, n} (f) = \int_ {\mathcal {X}} f (x) \phi_ {n} (x) \mathrm {d} \omega (x), \tag {9}
$$

and

$$
\mathbb {V} _ {\mathrm {D P P}} I ^ {\mathrm {E Z}, n} (f) = \sum_ {m \geq N + 1} \langle f, \phi_ {m} \rangle_ {\omega} ^ {2}. \tag {10}
$$

Theorem 1 shows that the  $I^{\mathrm{EZ},n}(f)$  is an unbiased estimator of  $\int_{\mathcal{X}} f(x) \phi_n(x) \, \mathrm{d}\omega(x)$ , and its variance depends on the coefficients  $\langle f, \phi_m \rangle_\omega$  for  $m \geq N + 1$ . Consequently, the expected squared error of the quadrature is equal to the variance of  $I^{\mathrm{EZ},n}(f)$  and it is given by

$$
\mathbb {E} _ {\mathrm {D P P}} \left| \int_ {\mathcal {X}} f (x) \phi_ {n} (x) \mathrm {d} \omega (x) - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, n} f \left(x _ {i}\right) \right| ^ {2} = \sum_ {m \geq N + 1} \langle f, \phi_ {m} \rangle_ {\omega} ^ {2}. \tag {11}
$$

This suggests that the expected squared error depends strongly on the function  $f$ . This makes the comparison between EZQ and other quadratures, based on some test function  $f$ , tricky: the choice of  $f$  may favor (or disfavor) EZQ. In order to circumvent this difficulty, we suggest to study a figure of merit that is independent of the choice of the function  $f$ . This is possible using kernels through the study of the worst-case integration error on the unit ball of an RKHS. The definition of this quantity will be recalled in the following section.

# 2.2 The worst integration error in kernel quadrature

The use of the kernel framework in the context of numerical integration can be tracked back to the work of Hickernell [17, 18], who introduced the use of kernels to the quasi Monte Carlo community. Their use was popularized in the machine learning community by [26, 9]. In this framework, the quality of a quadrature is assessed by the worst-case integration error on the unit ball of an RKHS  $\mathcal{F}$  associated to some kernel  $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_+$ . This quantity is defined as follows

$$
\sup  _ {f \in \mathcal {F}, \| f \| _ {\mathcal {F}} \leq 1} \left| \int_ {\mathcal {X}} f (x) g (x) d \omega (x) - \sum_ {i \in [ N ]} w _ {i} f \left(x _ {i}\right) \right|. \tag {12}
$$

This quantity reflects how good is the quadrature uniformly on the unit ball of  $\mathcal{F}$ . Interestingly, this quantity have a closed formula

$$
\left\| \mu_ {g} - \sum_ {i \in [ N ]} w _ {i} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}}, \tag {13}
$$

where  $\mu_g = \Sigma g$  is the so-called embedding of  $g$  in the RKHS  $\mathcal{F}$ . We shall use in Section 3.2 the equivalent expression (13) of the worst-case integration error, to derive a closed formula of

$$
\mathbb {E} _ {\mathrm {D P P}} \sup  _ {f \in \mathcal {F}, \| f \| _ {\mathcal {F}} \leq 1} \left| \int_ {\mathcal {X}} f (x) g (x) \mathrm {d} \omega (x) - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, n} f \left(x _ {i}\right) \right| ^ {2}. \tag {14}
$$

By now, we observe that the weights  $\hat{w}_i^{\mathrm{EZ},n}$  of EZQ are non-optimal in the sense that they do not minimize (13). By definition, the optimal kernel quadrature for a given configuration  $x$ , such that the kernel matrix  $K(x)$  is non-singular, is the quadrature with nodes the  $x_i$  and weights the  $\hat{w}_i$  that minimize (13). We precise in Section 3.1 the subtle difference between the quadrature of Ermakov and Zolotukhin and the optimal kernel quadrature. Before that, we review the existing constructions of the optimal kernel quadrature in the following section.

# 2.3 The design of the optimal kernel quadrature

The optimal kernel quadrature may be calculated numerically under the assumption that the matrix  $\pmb{K}(\pmb{x})$  is non-singular. Indeed, for a given configuration of nodes  $\pmb{x} \in \mathcal{X}^N$ , the square of (13) is quadratic on  $\pmb{w}$  and have a unique solution given by  $\hat{w}^{\mathrm{OKQ},g} = K(\pmb{x})^{-1}\mu_g(\pmb{x})^2$ . In particular, the optimal mixture  $\sum_{i\in [N]}\hat{w}_i^{\mathrm{OKQ},g}k(x_i,.)$  takes the same values as  $\mu_g$  on the nodes  $x_i$ : the optimal mixture interpolates the function  $\mu_g$  on the configuration of nodes  $\pmb{x}$ . At this level,  $\pmb{x}$  is still a degree of freedom and need to be designed. This task was tackled by different approaches. One approach consists on using adhoc designs for which a theoretical analysis of the convergence rate is possible. This is the case of, inter alia, the uniform grid in the periodic Sobolev space [6, 24], higher-order digital nets sequences in tensor products of Sobolev spaces [8], or tensor product of scaled Hermite roots in the RKHS defined by the Gaussian kernel [21]. Another approach consists on using a sequential algorithm to build up the configuration  $\pmb{x}$  [11, 12, 20, 7]. In general, each step of these greedy algorithms requires to solve a non-convex problem and costly approximations must be employed. Alternatively, random designs, based on determinantal point processes and their mixtures [3, 4], were shown to have strong theoretical guarantees and competitive empirical performances. More precisely, it was shown that if  $\pmb{x}$  follows the distribution of DPP of reference measure  $\omega$  and marginal kernel  $\kappa_N$ , and if  $g \in \mathcal{L}(\omega)$  such that  $\| g\|_{\omega} \leq 1$ , then

$$
\mathbb {E} _ {\mathrm {D P P}} \left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {O K Q}, g} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}} ^ {2} \leq 2 \sigma_ {N + 1} + 2 \left(\sum_ {n \in [ N ]} | \langle g, \phi_ {n} \rangle_ {\omega} |\right) ^ {2} N r _ {N}, \tag {15}
$$

where  $r_N = \sum_{m\geq N + 1}\sigma_m$  [2] (Theorem 4.8). However, numerical simulations suggest that the l.h.s. of (15) converges to 0 at the faster rate  $\mathcal{O}(\sigma_{N + 1})$ , which corresponds to the best achievable rate according to [4]. This optimal rate was proved to be achieved, under some mild conditions on the eigenvalues  $\sigma_{n}$ , using the distribution of continuous volume sampling (CVS) [4]. This distribution is a mixture of determinantal point processes and is closely tied to the projection DPP used in [3] and comes with the following guarantee

$$
\forall g \in \mathcal {L} _ {2} (\omega), \mathbb {E} _ {\mathrm {C V S}} \left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {O K Q}, g} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}} ^ {2} = \sum_ {m \in \mathbb {N} ^ {*}} \langle g, \phi_ {n} \rangle_ {\omega} ^ {2} \epsilon_ {m} (N), \tag {16}
$$

where  $\epsilon_{m}(N) = \mathcal{O}(\sigma_{N + 1})$  for every  $m\in \mathbb{N}^*$ , so that the expected squared worst-integration error of OKQ under the continuous volume sampling distribution scales as  $\mathcal{O}(\sigma_{N + 1})$  for every  $g\in \mathcal{L}(\omega)$ .

# 3 Main results

This section gathers the main contributions of this article. In Section 3.1, we prove that both EZQ and OKQ belong to a larger class of quadrature rules called kernel-based interpolation quadrature (KBIQ). In Section 3.2, we prove a close formula of the expected squared worst-case integration error of EZQ. In Section 3.3, we use Theorem 3 to improve on the existing theoretical guarantees of OKQ with DPPs.

# 3.1 Kernel-based interpolation quadrature

In this section, we define a new class of quadrature rules that extends both Ermakov-Zolotukhin quadrature and the optimal kernel quadrature. We start by the following observation: the weights  $\hat{w}_i^{\mathrm{EZ},n}(\pmb{x})$  of EZQ, defined in (6), writes as

$$
\hat {\boldsymbol {w}} ^ {\mathrm {E Z}, n} (\boldsymbol {x}) = \Phi_ {N} (\boldsymbol {x}) ^ {- 1} \boldsymbol {e} _ {n}. \tag {17}
$$

By observing that  $\phi_n(\pmb{x}) = \Phi_N(\pmb{x})^\top \pmb{e}_n$ , and  $\kappa_{N}(\pmb{x}) = \Phi_{N}(\pmb{x})^{\top}\Phi_{N}(\pmb{x})$ , we prove that

$$
\hat {\boldsymbol {w}} ^ {\mathrm {E Z}, n} (\boldsymbol {x}) = \kappa_ {N} (\boldsymbol {x}) ^ {- 1} \phi_ {n} (\boldsymbol {x}), \tag {18}
$$

Equivalently, we have  $\phi_n(\pmb{x}) = \kappa_N(\pmb{x})\hat{\pmb{w}}^{\mathrm{EZ},n}(\pmb{x})$ . In other words,  $\sum_{i\in [N]}\hat{w}_i^{\mathrm{EZ},n}(\pmb{x})\kappa_N(x_i,\cdot)$  takes the same values as  $\phi_n$  on the nodes  $x_i$ :  $\hat{\pmb{w}}^{\mathrm{EZ},n}(\pmb{x})$  is the vector resulting of the interpolation of  $\phi_n$  by the kernel  $\kappa_N$ . From this observation, we define kernel-based interpolation quadrature as an extension of EZQ as follows: let  $\gamma \coloneqq (\gamma_m)_{m\in \mathbb{N}^*}$  be a sequence of positive real numbers, and let  $M\in \mathbb{N}^{*}\cup \{+\infty \}$ . Define the kernel  $\kappa^{\gamma ,M}$  on  $\mathcal{X}\times \mathcal{X}$  by

$$
\forall x, y \in \mathcal {X}, \kappa^ {\gamma , M} (x, y) = \sum_ {m = 1} ^ {M} \gamma_ {m} \phi_ {m} (x) \phi_ {m} (y). \tag {19}
$$

Now, starting from a configuration  $\pmb{x} \in \mathcal{X}^N$  such that  $\operatorname{Det} \kappa_N(\pmb{x}) > 0$ , we have  $\operatorname{Det} \kappa^{\gamma, M}(\pmb{x}) > 0^3$  and for a given  $g \in \mathcal{L}_2(\omega)$ , we define the vector of weights  $\hat{\pmb{w}}^{\gamma, M, g}(\pmb{x}) \in \mathbb{R}^N$  by

$$
\hat {\boldsymbol {w}} ^ {\gamma , M, g} (\boldsymbol {x}) = \kappa^ {\gamma , M} (\boldsymbol {x}) ^ {- 1} \mu_ {g} ^ {\gamma , M} (\boldsymbol {x}), \tag {20}
$$

where

$$
\mu_ {g} ^ {\gamma , M} (x) = \sum_ {m = 1} ^ {M} \gamma_ {m} \langle g, \phi_ {m} \rangle_ {\omega} \phi_ {m} (x). \tag {21}
$$

We check again that  $\sum_{i\in [N]}\hat{w}_i^{\gamma ,M,g}\kappa^{\gamma ,M}(x_i,.)$  takes the same values as  $\mu_g^{\gamma ,M}$  on the nodes  $x_{i}$ : the mixture interpolates  $\mu_g^{\gamma ,M}$  on the nodes  $x_{i}$ . Now, for a given  $g\in \mathcal{L}_2(\omega)$ , the vector of weights  $\hat{w}^{\gamma ,M,g}(\pmb {x})$  have two degrees of freedom: the sequence  $\gamma$  and the rank of the kernel  $M$ . These degrees of freedom may be mixed in a variety of ways to cover a large class of quadrature rules. In particular, we show in Section 3.1.1 that, for any sequence  $\gamma$ , KBIQ is equivalent to EZQ when  $M = N$ , and we show in Section 3.1.2 that KBIQ is equivalent to OKQ when  $M = +\infty$  and  $\gamma = \sigma$ ; as summarized in Table 1. We may also consider  $M$  to be finite but strictly larger than  $N$ . Yet, the theoretical analysis of these intermediate quadrature rules is beyond the scope of this work.

Table 1: An overview of some examples of KBIQ with the corresponding couples  $(\kappa^{\gamma ,M},\mu_g^{\gamma ,M})$  

<table><tr><td>Quadrature</td><td>M</td><td>γ</td><td>μγ,M</td><td>κγ,M</td></tr><tr><td>EZQ</td><td>N</td><td>Any</td><td>∑n∈[N] γn(g,φn)ωφn</td><td>κγ,N</td></tr><tr><td>EZQ</td><td>N</td><td>γm=1</td><td>gn := ∑n∈[N] (g,φn)ωφn</td><td>κN</td></tr><tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr><tr><td>OKQ</td><td>+∞</td><td>σ</td><td>μg</td><td>k</td></tr></table>

# 3.1.1 EZQ is a special case of KBIQ

We recover EZQ, as defined in [13], by taking  $M = N$ , and  $\gamma$  is defined by  $\gamma_{m} = 1$  for every  $m \in \mathbb{N}^*$ , and  $g \equiv \phi_{n}$  for some  $n \in [N]$ . The equivalent definition (20) extends EZQ to the situation when  $g \notin \mathcal{E}_N$ . Even better, we show in the following that  $\hat{\pmb{w}}^{\gamma ,N,g}(\pmb {x})$  is independent of  $\gamma$  when  $M = N$ . In particular, for any sequence of positive numbers  $\gamma$  we have

$$
\forall n \in [ N ], \quad \hat {\boldsymbol {w}} ^ {\gamma , N, \phi_ {n}} (\boldsymbol {x}) = \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, n} (\boldsymbol {x}). \tag {22}
$$

Proposition 2. Let  $g \in \mathcal{E}_N$ , and let  $\pmb{x} \in \mathcal{X}^N$  such that  $\operatorname{Det} \kappa_N(\pmb{x}) > 0$ . Let  $\gamma = (\gamma_m)_{m \in \mathbb{N}^*}$  and  $\tilde{\gamma} = (\tilde{\gamma}_m)_{m \in \mathbb{N}^*}$  be two sequences of positive numbers. We have

$$
\hat {\boldsymbol {w}} ^ {\gamma , N, g} (\boldsymbol {x}) = \hat {\boldsymbol {w}} ^ {\tilde {\gamma}, N, g} (\boldsymbol {x}). \tag {23}
$$

Thanks to the invariance of  $\hat{w}^{\gamma, N, g}$  with respect to  $\gamma$ , we simplify the notation and we write  $\hat{w}^{\mathrm{EZ}, g}$  instead. Moreover, using this invariance, EZQ may be seen as an approximation of OKQ when  $g \in \mathcal{E}_N$ . Indeed, by approximating the kernel matrix  $\pmb{K}(\pmb{x}) \approx \pmb{K}_N(\pmb{x})$  where  $k_N(x, y) = \sum_{n \in [N]} \sigma_n \phi_n(x) \phi_n(y)$ , we have

$$
\boldsymbol {K} (\boldsymbol {x}) ^ {- 1} \mu_ {g} (\boldsymbol {x}) \approx \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g}, \tag {24}
$$

since  $K_N(\pmb{x})^{-1}\mu_g(\pmb{x}) = \hat{\pmb{w}}^{\mathrm{EZ},g}$  by Proposition 2. Interestingly, this approximation is reminiscent to the one used in [22] in the case of the Gaussian kernel.

# 3.1.2 OKQ is a special case of KBIQ

The optimal kernel quadrature is a special case of KBIQ when  $M = +\infty$  and  $\gamma = \sigma$ . Indeed, in this case, we have  $\kappa^{\gamma, M} = k$ , and  $\mu_g^{\gamma, M} = \mu_g$ , so that

$$
\hat {\boldsymbol {w}} ^ {\sigma , M, g} (\boldsymbol {x}) = \boldsymbol {K} (\boldsymbol {x}) ^ {- 1} \mu_ {g} (\boldsymbol {x}) = \hat {\boldsymbol {w}} ^ {\mathrm {O K Q}, g}. \tag {25}
$$

In other words, Ermakov-Zolotukhin quadrature and the optimal kernel quadrature are extreme instances of interpolation based kernel quadrature that correspond to the regimes  $M = N$  and  $M = +\infty$ . As it was shown in Proposition 2, the weights of EZQ depend only on the eigenfunctions  $\phi_{m}$  and do not depend on the eigenvalues  $\sigma_{m}$ . This is to be compared to the weights of OKQ that depend simultaneously on the eigenvalues and the eigenfunctions.

# 3.2 Main theorem

We give in this section the theoretical analysis of the worst case integration error of EZQ under the distribution of the projection DPP.

Theorem 3. Let  $N\in \mathbb{N}^*$ . We have

$$
\forall g \in \mathcal {E} _ {N}, \mathbb {E} _ {\mathrm {D P P}} \| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k (x _ {i},.) \| _ {\mathcal {F}} ^ {2} = \sum_ {n \in [ N ]} \left\langle g, \phi_ {n} \right\rangle_ {\omega} ^ {2} r _ {N}, \tag {26}
$$

where  $r_N = \sum_{m\geq N + 1}\sigma_m$  . In particular,

$$
\forall g \in \mathcal {L} _ {2} (\omega), \mathbb {E} _ {\mathrm {D P P}} \| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k (x _ {i},.) \| _ {\mathcal {F}} ^ {2} \leq 4 \| g \| _ {\omega} ^ {2} r _ {N}. \tag {27}
$$

As an immediate consequence of Theorem 3, we have

$$
\forall g \in \mathcal {L} _ {2} (\omega), \mathbb {E} _ {\mathrm {D P P}} \sup  _ {\substack {f \in \mathcal {F} \\ \| f \| _ {\mathcal {F}} = 1}} \left| \int_ {\mathcal {X}} f (x) g (x) \mathrm {d} \omega (x) - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} f (x _ {i}) \right| ^ {2} = \mathcal {O} (r _ {N}). \tag{28}
$$

In other words, the squared worst-case integration error of Ermakov-Zolotukhin quadrature with DPP nodes converges to 0 at the rate  $\mathcal{O}(r_{N + 1})$ . This rate is slower than the rate of convergence of  $\mathbb{E}_{\mathrm{DPP}}I^{\mathrm{EZ},n}(f)^2$  given by Theorem 1. Indeed, if  $f\in \mathcal{F}$  then  $\| f\|_{\mathcal{F}}^{2} = \sum_{m\in \mathbb{N}^{*}}\langle f,\phi_{m}\rangle_{\omega}^{2} / \sigma_{m} < + \infty$  and Theorem 1 yields

$$
\mathbb {V} _ {\mathrm {D P P}} I ^ {\mathrm {E Z}, n} (f) ^ {2} \leq \sigma_ {N + 1} \| f \| _ {\mathcal {F}} ^ {2}, \tag {29}
$$

so that

$$
\mathbb {E} _ {\mathrm {D P P}} \left| \int_ {\mathcal {X}} f (x) \phi_ {n} (x) \mathrm {d} \omega (x) - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, n} f \left(x _ {i}\right) \right| ^ {2} = \mathbb {V} _ {\mathrm {D P P}} I ^ {\mathrm {E Z}, n} (f) ^ {2} = \mathcal {O} (\sigma_ {N + 1}). \tag {30}
$$

Now, observe that for some sequences we have  $\sigma_{N + 1} = o(r_{N + 1})$ . For instance, if  $\sigma_{m} = m^{-2s}$  for some  $s > 1/2$ , then  $r_{N + 1} = \mathcal{O}(N^{1 - 2s})$ . We conclude that the convergence of EZQ under DPP is slower than the optimal rate  $\mathcal{O}(\sigma_{N + 1})$ , that was observed empirically for OKQ under DPP in [3] and proved theoretically for OKQ under CVS in [4], if we consider the worst-case integration error as a figure of merit. This is to be compared with the theoretical result of [13] that can not predict the difference in the rate of convergence between EZQ and OKQ: our analysis highlights the interest of using kernels when comparing quadratures.

# 3.3 Improved theoretical guarantees for the optimal kernel quadrature with DPPs

Theorem 3 improves on the existing theoretical guarantees of the optimal kernel quadrature with determinantal point processes initially proposed in [3]. This is the purpose of the following result.

Theorem 4. Let  $N\in \mathbb{N}^*$ . We have

$$
\forall g \in \mathcal {L} _ {2} (\omega), \mathbb {E} _ {\mathrm {D P P}} \| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {O K Q}, g} k (x _ {i},.) \| _ {\mathcal {F}} ^ {2} \leq 4 \| g \| _ {\omega} ^ {2} r _ {N}. \tag {31}
$$

Compared to the analysis conducted in [3], Theorem 4 offers a sharper upper bound of

$$
\mathbb {E} _ {\mathrm {D P P}} \| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {O K Q}, g} k \left(x _ {i}, .\right) \| _ {\mathcal {F}} ^ {2}. \tag {32}
$$

Indeed, the upper bound (31) is dominated by  $\sum_{n=1}^{N} \langle g, \phi_n \rangle_\omega^2 r_N$  comparably to the upper bound (15), proved in [3], dominated by  $\left( \sum_{n=1}^{N} |\langle g, \phi_n \rangle_\omega| \right)^2 N r_N$ : our bound improves upon (15) by a factor of  $N^2$ , since

$$
\left(\sum_ {n = 1} ^ {N} \left| \langle g, \phi_ {n} \rangle_ {\omega} \right|\right) ^ {2} \leq N \sum_ {n = 1} ^ {N} \langle g, \phi_ {n} \rangle_ {\omega} ^ {2} \leq N \| g \| _ {\omega} ^ {2}. \tag {33}
$$

Theorem 4 follows immediately from Theorem 3 by observing that

$$
\left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {O K Q}, g} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}} ^ {2} \leq \left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}} ^ {2}. \tag {34}
$$

Table 2 summarizes the theoretical contributions of this work compared to the existing literature.

Table 2: A comparison of the rates given by Theorem 3 and Theorem 4 compared to the existing guarantees in the literature.  

<table><tr><td>Quadrature</td><td>Distribution</td><td>Theoretical rate</td><td>Empirical rate</td><td>Reference</td></tr><tr><td>EZQ</td><td>DPP</td><td>O(rN+1)</td><td>O(rN+1)</td><td>Theorem 3</td></tr><tr><td rowspan="2">OKQ</td><td rowspan="2">DPP</td><td>N2O(rN+1)</td><td>O(σN+1)</td><td>[3]</td></tr><tr><td>O(rN+1)</td><td>O(σN+1)</td><td>Theorem 4</td></tr><tr><td>OKQ</td><td>CVS</td><td>O(σN+1)</td><td>O(σN+1)</td><td>[4]</td></tr></table>

We give in the following section, a sketch of the main ideas behind the proof of Theorem 3.

# 4 Sketch of the proof

The proof of Theorem 3 decomposes into two steps. First, in Section 4.1, we give a decomposition of the squared approximation error  $\| \mu_g - \sum_{i\in [N]}\hat{w}_i^{\mathrm{EZ},g}k(x_i,.)\|_{\mathcal{F}}^2$ , then, in Section 4.2, we use this decomposition to prove a closed formula of  $\mathbb{E}_{\mathrm{DPP}}\| \mu_g - \sum_{i\in [N]}\hat{w}_i^{\mathrm{EZ},g}k(x_i,.)\|_{\mathcal{F}}^2$ .

# 4.1 A decomposition of the approximation error

Let  $g\in \mathcal{E}_N$  and let  $\pmb {x}\in \mathcal{X}^N$  such that  $\operatorname *{Det}\kappa_N(\pmb {x}) > 0$  , we have

$$
\left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}} ^ {2} = \left\| \mu_ {g} \right\| _ {\mathcal {F}} ^ {2} - 2 \mu_ {g} (\boldsymbol {x}) ^ {\intercal} \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g} + \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g ^ {\intercal}} \boldsymbol {K} (\boldsymbol {x}) \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g}. \tag {35}
$$

The last two terms of the r.h.s of (35) decompose as follows.

Proposition 5. Let  $g \in \mathcal{E}_N$  and let  $\pmb{x} \in \mathcal{X}^N$  such that  $\operatorname{Det} \kappa_N(\pmb{x}) > 0$ . We have

$$
\mu_ {g} (\boldsymbol {x}) ^ {\intercal} \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g} = \| \mu_ {g} \| _ {\mathcal {F}} ^ {2}, \tag {36}
$$

and

$$
\hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g ^ {\intercal}} \boldsymbol {K} (\boldsymbol {x}) \hat {\boldsymbol {w}} ^ {\mathrm {E Z}, g} = \| \mu_ {g} \| _ {\mathcal {F}} ^ {2} + \epsilon^ {\intercal} \Phi_ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} \boldsymbol {K} _ {N} ^ {\perp} (\boldsymbol {x}) \Phi_ {N} (\boldsymbol {x}) ^ {- 1} \epsilon , \tag {37}
$$

where  $\epsilon = \sum_{n\in [N]}\langle g,\phi_n\rangle_{\omega}e_n$  and  $k_N^{\perp}$  is the kernel defined by

$$
k _ {N} ^ {\perp} (x, y) = \sum_ {m \geq N + 1} \sigma_ {m} \phi_ {m} (x) \phi_ {m} (y). \tag {38}
$$

The proof of Proposition 5 is detailed in Appendix A.3. Now, by combining (35), (36) and (37), we get

$$
\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k (x _ {i},.) \| _ {\mathcal {F}} ^ {2} = \| \mu_ {\mathcal {G}} \| _ {\mathcal {F}} ^ {2} - 2 \| \mu_ {\mathcal {G}} \| _ {\mathcal {F}} ^ {2} + \| \mu_ {\mathcal {G}} \| _ {\mathcal {F}} ^ {2} + \epsilon^ {\intercal} \Phi_ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} K _ {N} ^ {\perp} (\boldsymbol {x}) \Phi_ {N} (\boldsymbol {x}) ^ {- 1} \epsilon .
$$

This proves the following result.

Theorem 6. Let  $g \in \mathcal{E}_N$  and let  $\pmb{x} \in \mathcal{X}^N$  such that  $\operatorname{Det} \kappa_N(\pmb{x}) > 0$ . We have

$$
\left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k \left(x _ {i}, .\right) \right\| _ {\mathcal {F}} ^ {2} = \boldsymbol {\epsilon} ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} \boldsymbol {K} _ {N} ^ {\perp} (\boldsymbol {x}) \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1} \boldsymbol {\epsilon}, \tag {39}
$$

where  $\epsilon = \sum_{n\in [N]}\langle g,\phi_n\rangle_{\omega}e_n$

# 4.2 A tractable formula of the expected approximation error

In the following, we prove a closed formula for  $\mathbb{E}_{\mathrm{DPP}}\| \mu_g - \sum_{i\in [N]}\hat{w}_i^{\mathrm{EZ},g}k(x_i,.)\|_{\mathcal{F}}^2$ . By Theorem 6, it is enough to calculate

$$
\mathbb {E} _ {\mathrm {D P P}} \boldsymbol {\epsilon} ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} \boldsymbol {K} _ {N} ^ {\perp} (\boldsymbol {x}) \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1} \boldsymbol {\epsilon}, \tag {40}
$$

for  $\pmb {\epsilon}\in \mathbb{R}^{N}$  . For this purpose, observe that  $K_N^\perp (\pmb {x}) = \sum_{m\geq N + 1}\sigma_m\phi_m(\pmb {x})\phi_m(\pmb {x})^\top$  , so that

$$
\boldsymbol {\epsilon} ^ {\mathsf {T}} \Phi_ {N} (\boldsymbol {x}) ^ {- 1 ^ {\mathsf {T}}} \boldsymbol {K} _ {N} ^ {\perp} (\boldsymbol {x}) \Phi_ {N} (\boldsymbol {x}) ^ {- 1} \boldsymbol {\epsilon} = \sum_ {m \geq N + 1} \sigma_ {m} \boldsymbol {\epsilon} ^ {\mathsf {T}} \Phi_ {N} (\boldsymbol {x}) ^ {- 1 ^ {\mathsf {T}}} \phi_ {m} (\boldsymbol {x}) \phi_ {m} (\boldsymbol {x}) ^ {\mathsf {T}} \Phi_ {N} (\boldsymbol {x}) ^ {- 1} \boldsymbol {\epsilon}. \tag {41}
$$

Therefore, the calculation of 40 boils down to the calculation of

$$
\mathbb {E} _ {\mathrm {D P P}} \boldsymbol {\epsilon} ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} \phi_ {m} (\boldsymbol {x}) \phi_ {m} (\boldsymbol {x}) ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1} \boldsymbol {\epsilon}, \tag {42}
$$

for  $m \geq N + 1$ . This is the purpose of the following result.

Theorem 7. Let  $\epsilon = \sum_{n\in [N]}\epsilon_n e_n,\tilde{\epsilon} = \sum_{n\in [N]}\tilde{\epsilon}_n e_n\in \mathbb{R}^N$  , and  $m\geq N + 1$  . Then

$$
\mathbb {E} _ {\mathrm {D P P}} \boldsymbol {\epsilon} ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} \phi_ {m} (\boldsymbol {x}) \phi_ {m} (\boldsymbol {x}) ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1} \tilde {\boldsymbol {\epsilon}} = \sum_ {n \in [ N ]} \epsilon_ {n} \tilde {\epsilon} _ {n}. \tag {43}
$$

In particular,

$$
\mathbb {E} _ {\mathrm {D P P}} \boldsymbol {\epsilon} ^ {\intercal} \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1 ^ {\intercal}} \boldsymbol {K} _ {N} ^ {\perp} (\boldsymbol {x}) \boldsymbol {\Phi} _ {N} (\boldsymbol {x}) ^ {- 1} \tilde {\boldsymbol {\epsilon}} = \sum_ {m \geq N + 1} \sigma_ {m} \sum_ {n \in [ N ]} \epsilon_ {n} \tilde {\epsilon} _ {n}. \tag {44}
$$

We give the proof of Theorem 7 in Appendix A.4. By taking  $\epsilon = \tilde{\epsilon} = \sum_{n\in [N]}\langle g,\phi_n\rangle_{\omega}e_n$  in Theorem 7, we obtain (26). As for (27), it is sufficient to observe that

$$
\left\| \mu_ {g} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k (x _ {i},.) \right\| _ {\mathcal {F}} ^ {2} \leq 2 \Big (\left\| \mu_ {g} - \mu_ {g _ {N}} \right\| _ {\mathcal {F}} ^ {2} + \left\| \mu_ {g _ {N}} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k (x _ {i},.) \right\| _ {\mathcal {F}} ^ {2} \Big), \tag {45}
$$

where  $g_{N} = \sum_{n\in [N]}\langle g,\phi_{n}\rangle_{\omega}\phi_{n}\in \mathcal{E}_{N}$ , so that we can apply (26) to  $g_{N}$  and we obtain

$$
\mathbb {E} _ {\mathrm {D P P}} \| \mu_ {g _ {N}} - \sum_ {i \in [ N ]} \hat {w} _ {i} ^ {\mathrm {E Z}, g} k (x _ {i},.) \| _ {\mathcal {F}} ^ {2} = \sum_ {n \in [ N ]} \left\langle g, \phi_ {n} \right\rangle_ {\omega} ^ {2} \sum_ {m \geq N + 1} \sigma_ {m} \leq \| g \| _ {\omega} ^ {2} \sum_ {m \geq N + 1} \sigma_ {m}. \tag {46}
$$

The term  $\| \mu_g - \mu_{g_N}\| _\mathcal{F}^2$  is upper bounded by  $\sigma_{N + 1}\| g\|_{\omega}^{2}$ . We give the details in Appendix A. This concludes the proof of Theorem 3. In the following section, we give numerical experiments illustrating this result.

# 5 Numerical experiments

In this section, we illustrate the theoretical results presented in Section 3 in the case of the RKHS associated to the kernel

$$
k _ {s} (x, y) = 1 + \sum_ {m \in \mathbb {N} ^ {*}} \frac {1}{m ^ {2 s}} \cos (2 \pi m (x - y)), \tag {47}
$$

![](images/92b2582f646fb7faf39f904add048df64990bd43f193867a443193ba67355f21.jpg)  
(a) EZQ  $(s = 2)$

![](images/ae9974c946eaaa47696c526ac3bc1d67aaa4e11642258b246fe6837a6e2a4d6d.jpg)  
(b) KBIQ  $(s = 2)$

![](images/4826f6a2674f1208bac0fa7966fcf1746a2962c7459c1e8f05328fdfe76655af.jpg)  
(c) OKQ  $(s = 2)$

![](images/a1c99c00bfeeb74cd65df1b5216e11bddaa7e7aa5eba43ec3391836cb1c87d77.jpg)  
(d) EZQ  $(s = 3)$

![](images/129c153f8d885c1603c687b878c7e96cae9533b00f6b47ad15804cf1ea47617a.jpg)  
(e) KBIQ  $(s = 3)$

![](images/0f565c0bca545405f49054c7311901e3228be9e609568356b19888f63f0b5325.jpg)  
Figure 1: Squared worst-case integration error vs. number of nodes  $N$  for EZQ, KBIQ and OKQ in the Sobolev space of periodic functions of order  $s \in \{2,3\}$ .  
(f) OKQ  $(s = 3)$

that corresponds to the periodic Sobolev space of order  $s$  on  $[0,1]$  [5], and we take  $\omega$  to be the uniform measure on  $\mathcal{X} = [0,1]$ . We compare the squared worst-case integration error of EZQ and OKQ and KBIQ, with  $M = 2N$  and  $\gamma = \sigma$ , for  $\pmb{x}$  that follows the distribution of the projection DPP and for  $g \in \{e_1, e_{10}, e_{20}\}$ . We take  $N \in [5,100]$ . Figure 1 shows log-log plots of the squared error w.r.t.  $N$ , averaged over 1000 samples for each point, for  $s \in \{2,3\}$ . We observe that the squared error of EZQ converges to 0 at the exact rate  $\mathcal{O}(r_{N+1})$  predicted by Theorem 3, while the squared error of OKQ converges to 0 at the rate  $\mathcal{O}(\sigma_{N+1})$  as it was already observed in [3], which is still better than the rate  $\mathcal{O}(r_{N+1})$  proved in Theorem 4. Finally, KBIQ ( $M = 2N$  and  $\gamma = \sigma$ ) converges to 0 at the rate  $\mathcal{O}(\sigma_{N+1})$ . We conclude that, by taking  $M = \alpha N$  with  $\alpha > 1$ , KBIQ have practically the same averaged error as OKQ ( $M = +\infty$ ). As we have mentioned before, the theoretical analysis of KBIQ in the regime when  $M$  is finite and strictly larger than  $N$  is beyond the scope of this work, and we defer it for future work.

# 6 Conclusion

We studied the quadrature rule proposed by Ermakov and Zolotukhin through the lens of kernel methods. We proved that EZQ and OKQ belong to a larger class of quadrature rules that may be defined through kernel based interpolation. From this new perspective, EZQ may be seen as an approximation of OKQ. Moreover, we studied the expected value of the squared worst-case integration error of EZQ when the nodes follow the distribution of a DPP. In particular, we proved that EZQ converges to 0 at the rate  $\mathcal{O}(r_{N + 1})$  which is slower than the optimal rate  $\mathcal{O}(\sigma_{N + 1})$  typically observed for OKQ with DPPs. This work shows the importance of the worst-case integration error as a figure of merit when comparing quadrature rules. Interestingly, we use our analysis of EZQ to improve upon the existing theoretical guarantees of OKQ under DPPs. Finally, we illustrated the theoretical results by some numerical experiments that hint that KBIQ in the regime  $M > N$  may have similar performances as OKQ. It would be interesting to study this broader class of quadratures in the future.

# Broader impact

This article makes contributions to the fundamentals of numerical integration and due to its theoretical nature, we see no ethical or immediate societal consequence of our work.

# References

[1] R. Bardenet and A. Hardy. Monte carlo with determinantal point processes. The Annals of Applied Probability, 30(1):368-417, 2020.  
[2] A. Belhadji. Subspace sampling using determinantal point processes. PhD thesis, Ecole Centrale de Lille, 2020.  
[3] A. Belhadji, R. Bardenet, and P. Chainais. Kernel quadrature with DPPs. In Advances in Neural Information Processing Systems 32, pages 12907-12917. 2019.  
[4] A. Belhadji, R. Bardenet, and P. Chainais. Kernel interpolation with continuous volume sampling. Proceedings of the 37th International Coference on International Conference on Machine Learning, 2020.  
[5] A. Berlinet and C. Thomas-Agnan. Reproducing kernel Hilbert spaces in probability and statistics. Springer Science & Business Media, 2011.  
[6] B. Bojanov. Uniqueness of the optimal nodes of quadrature formulae. Mathematics of computation, 36(154):525-546, 1981.  
[7] F. Briol, C. Oates, M. Girolami, and M. Osborne. Frank-Wolfe Bayesian quadrature: Probabilistic integration with theoretical guarantees. In Advances in Neural Information Processing Systems, pages 1162-1170, 2015.  
[8] F. X. Briol, C. J. Oates, M. Girolami, M. A. Osborne, D. Sejdinovic, et al. Probabilistic integration: A role in statistical computation? Statistical Science, 34(1):1-22, 2019.  
[9] Y. Chen, M. Welling, and A. Smola. Super-samples from kernel herding. In Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial Intelligence, UAI'10, pages 109-116, Arlington, Virginia, United States, 2010. AUAI Press.  
[10] J. F. Coeurjolly, A. Mazoyer, and P. O. Amblard. Monte carlo integration of non-differentiable functions on  $[0, 1]^{\iota}$ ,  $\iota = 1, \ldots, d$ , using a single determinantal point pattern defined on  $[0, 1]^d$ . arXiv preprint arXiv:2003.10323, 2020.  
[11] S. De Marchi. On optimal center locations for radial basis function interpolation: computational aspects. Rend. Splines Radial Basis Functions and Applications, 61(3):343-358, 2003.  
[12] S. De Marchi, R. Schaback, and H. Wendland. Near-optimal data-independent point locations for radial basis function interpolation. Advances in Computational Mathematics, 23(3):317-330, 2005.  
[13] S. M. Ermakov and V. Zolotukhin. Polynomial approximations and the monte-carlo method. Theory of Probability & Its Applications, 5(4):428-431, 1960.  
[14] C. F. Gauss. Methodus nova integralium valores per approximationem inventiendi. apvd Henricvm Dieterich, 1815.  
[15] G. Gautier, R. Bardenet, and M. Valko. On two ways to use determinantal point processes for monte carlo integration. In Advances in Neural Information Processing Systems, volume 32, 2019.  
[16] P. Glasserman. Monte Carlo methods in financial engineering, volume 53. Springer Science & Business Media, 2013.  
[17] F. J. Hickernell. Quadrature error bounds with applications to lattice rules. SIAM Journal on Numerical Analysis, 33(5):1995-2016, 1996.  
[18] F. J. Hickernell. A generalized discrepancy and quadrature error bound. Mathematics of computation, 67(221):299-322, 1998.  
[19] J. B. Hough, M. Krishnapur, Y. Peres, and B. Virág. Determinantal processes and independence. Probability surveys, 3:206-229, 2006.

[20] F. Huszar and D. Duvenaud. Optimally-weighted herding is Bayesian quadrature. In Proceedings of the Twenty-Eighth Conference on Uncertainty in Artificial Intelligence, UAI'12, pages 377-386. AUAI Press, 2012.  
[21] T. Karvonen, C. J. Oates, and M. Girolami. Integration in reproducing kernel hilbert spaces of gaussian kernels. arXiv preprint arXiv:2004.12654, 2020.  
[22] T. Karvonen and S. Särkkä. Gaussian kernel quadrature at scaled Gauss-Hermite nodes. BIT Numerical Mathematics, pages 1-26, 2019.  
[23] N. Metropolis and S. Ulam. The Monte Carlo method. Journal of the American statistical association, 44(247):335-341, 1949.  
[24] E. Novak, M. Ullrich, and H. Wozniakowski. Complexity of oscillatory integration for univariate sobolev spaces. Journal of Complexity, 31(1):15-41, 2015.  
[25] C. P. Robert and G. Casella. Monte Carlo statistical methods. Springer, 2004.  
[26] A. Smola, A. Gretton, L. Song, and B. Schölkopf. A Hilbert space embedding for distributions. In International Conference on Algorithmic Learning Theory, pages 13-31. Springer, 2007.
