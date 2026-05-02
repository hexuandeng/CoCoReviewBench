# ON GENERALIZATION ERROR BOUNDS OF NOISY GRADIENT METHODS FOR NON-CONVEX LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Generalization error (also known as the out-of-sample error) measures how well the hypothesis learned from training data generalizes to previously unseen data. Proving tight generalization error bounds is a central question in statistical learning theory. In this paper, we obtain generalization error bounds for learning general non-convex objectives, which has attracted significant attention in recent years. We develop a new framework, termed Bayes-Stability, for proving algorithm-dependent generalization error bounds. The new framework combines ideas from both the PAC-Bayesian theory and the notion of algorithmic stability. Applying the Bayes-Stability method, we obtain new data-dependent generalization bounds for stochastic gradient Langevin dynamics (SGLD) and several other noisy gradient methods (e.g., with momentum, mini-batch and acceleration, Entropy-SGD). Our result recovers (and is typically tighter than) a recent result in Mou et al. (2018) and improves upon the results in Pensia et al. (2018). Our experiments demonstrate that our data-dependent bounds can distinguish randomly labelled data from normal data, which provides an explanation to the intriguing phenomena observed in Zhang et al. (2017a). We also study the setting where the total loss is the sum of a bounded loss and an additional  $\ell_2$  regularization term. We obtain new generalization bounds for the continuous Langevin dynamic in this setting by developing a new Log-Sobolev inequality for the parameter distribution at any time. Our new bounds are more desirable when the noise level of the process is not very small, and do not become vacuous even when  $T$  tends to infinity.

# 1 INTRODUCTION

Non-convex stochastic optimization is the major workhorse of modern machine learning. For instance, the standard supervised learning on a model class parametrized by  $\mathbb{R}^d$  can be formulated as the following optimization problem:

$$
\min  _ {w \in \mathbb {R} ^ {d}} \mathbb {E} _ {z \sim \mathcal {D}} \left[ F (w, z) \right],
$$

where  $w$  denotes the model parameter,  $\mathcal{D}$  is an unknown data distribution over the instance space  $\mathcal{Z}$ , and  $F: \mathbb{R}^d \times \mathcal{Z} \to \mathbb{R}$  is a given objective function which may be non-convex. A learning algorithm takes as input a sequence  $S = (z_1, z_2, \ldots, z_n)$  of  $n$  data points sampled i.i.d. from  $\mathcal{D}$ , and outputs a (possibly randomized) parameter configuration  $\hat{w} \in \mathbb{R}^d$ .

A fundamental problem in learning theory is to understand the generalization performance of learning algorithms--is the algorithm guaranteed to output a model that generalizes well to the data distribution  $\mathcal{D}$ ? Specifically, we aim to prove upper bounds on the generalization error  $\mathrm{err}_{\mathrm{gen}}(S) = \mathcal{L}(\hat{w},\mathcal{D}) - \mathcal{L}(\hat{w},S)$ , where  $\mathcal{L}(\hat{w},\mathcal{D}) = \mathbb{E}_{z\sim \mathcal{D}}[\mathcal{L}(\hat{w},z)]$  and  $\mathcal{L}(\hat{w},S) = \frac{1}{n}\sum_{i = 1}^{n}\mathcal{L}(\hat{w},z_i)$  are the population and empirical losses, respectively. We note that the loss function  $\mathcal{L}$  (e.g., the  $0/1$  loss) could be different from the objective function  $F$  (e.g., the cross-entropy loss) used in the training process (which serves as a surrogate for the loss  $\mathcal{L}$ ).

Classical learning theory relates the generalization error to various complexity measures (e.g., the VC-dimension and Rademacher complexity) of the model class. Directly applying these classical complexity measures, however, often fails to explain the recent success of over-parametrized neural networks, where the model complexity significantly exceeds the amount of available training data

(see e.g., Zhang et al. (2017a)). By incorporating certain data-dependent quantities such as margin and compressibility into the classical framework, some recent work (e.g., Bartlett et al. (2017); Arora et al. (2018); Wei & Ma (2019)) obtains more meaningful generalization bounds in the deep learning context.

An alternative approach to generalization is to prove algorithm-dependent bounds. One celebrated example along this line is the algorithmic stability framework initiated by Bousquet & Elisseeff (2002). Roughly speaking, the generalization error can be bounded by the stability of the algorithm (see Section 2 for the details). Using this framework, Hardt et al. (2016) study the stability (hence the generalization) of stochastic gradient descent (SGD) for both convex and non-convex functions. Their work motivates recent study of the generalization performance of several other gradient-based optimization methods: Kuzborskij & Lampert (2018); London (2016); Chaudhari et al. (2017); Raginsky et al. (2017); Mou et al. (2018); Pensia et al. (2018); Chen et al. (2018).

In this paper, we study the algorithmic stability and generalization performance of various iterative gradient-based method, with certain continuous noise injected in each iteration, in a non-convex setting. As a concrete example, we consider the stochastic gradient Langevin dynamics (SGLD) (see Raginsky et al. (2017); Mou et al. (2018); Pensia et al. (2018)). Viewed as a variant of SGD, SGLD adds an isotropic Gaussian noise at every update step:

$$
W _ {t} \leftarrow W _ {t - 1} - \gamma_ {t} g _ {t} \left(W _ {t - 1}\right) + \frac {\sigma_ {t}}{\sqrt {2}} \mathcal {N} \left(0, I _ {d}\right), \tag {1}
$$

where  $g_{t}(W_{t - 1})$  denotes either the full gradient or the gradient over a mini-batch sampled from training dataset. We also study a continuous version of (1), which is the dynamic defined by the following stochastic differential equation (SDE):

$$
\mathrm {d} W _ {t} = - \nabla F (W _ {t}) \mathrm {d} t + \sqrt {2 \beta^ {- 1}} \mathrm {d} B _ {t}, \tag {2}
$$

where  $B_{t}$  is the standard Brownian motion.

# 1.1 RELATED WORK

Most related to our work is the study of algorithm-dependent generalization bounds of stochastic gradient methods. Hardt et al. (2016) first study the generalization performance of SGD via algorithmic stability. They prove a generalization bound that scales linearly with  $T$ , the number of iterations, when the loss function is convex, but their results for general non-convex optimization are more restricted. London (2017) presents a generalization bound that also combines PAC-Bayesian analysis with stability. However, their prior and posterior are probability distributions on the hyperparameter space, while ours are distributions on the hypothesis space. Our work is a follow-up of the recent work by Mou et al. (2018), in which they provide generalization bounds for SGLD from both stability and PAC-Bayesian perspectives. Another closely related work by Pensia et al. (2018) derives similar bounds for noisy stochastic gradient methods, based on the information theoretic framework of Xu & Raginsky (2017). However, their bounds scale as  $O(\sqrt{T / n})$  ( $n$  is the size of the training dataset) and are sub-optimal even for SGLD.

We acknowledge that besides the algorithm-dependent approach that we follow, recent advances in learning theory aim to explain the generalization performance of neural networks from many other perspectives. Some of the most prominent ideas include bounding the network capacity by the norms of weight matrices Neyshabur et al. (2015); Liang et al. (2019), margin theory Bartlett et al. (2017); Wei et al. (2018), PAC-Bayesian theory Dziugaite & Roy (2017); Neyshabur et al. (2018); Dziugaite & Roy (2018), network compressibility Arora et al. (2018), and over-parametrization Du et al. (2019); Allen-Zhu et al. (2018); Zou et al. (2018); Chizat & Bach (2018). Most of these results are stated in the context of neural networks (some are tailored to networks with specific architecture), whereas our work addresses generalization in non-convex stochastic optimization in general. We also note that some recent work provides explanations for the phenomenon reported in Zhang et al. (2017a) from a variety of different perspectives (e.g., Bartlett et al. (2017); Arora et al. (2018; 2019)).

Welling & Teh (2011) first consider stochastic gradient Langevin dynamics (SGLD) as a sampling algorithm in the Bayesian inference context. Raginsky et al. (2017) give a non-asymptotic analysis and establish the finite-time convergence guarantee of SGLD to an approximate global minimum.

Zhang et al. (2017b) analyze the hitting time of SGLD and prove that SGLD converges to an approximate local minimum. These results are further improved and generalized to a family of Langevin dynamics based algorithms by the subsequent work of Xu et al. (2018).

# 1.2 OVERVIEW OF OUR RESULTS

In this paper, we provide generalization guarantees for the noisy variants of several popular stochastic gradient methods.

The Bayes-Stability method and data-dependent generalization bounds. We develop a new method for proving generalization bounds, termed as Bayes-Stability, by incorporating ideas from the PAC-Bayesian theory into the stability framework. In particular, assuming the loss takes value in  $[0,C]$ , our method shows that the generalization error is bounded by both  $2C\mathbb{E}_z[\sqrt{2\mathrm{KL}(P,Q_z)} ]$  and  $2C\mathbb{E}_z[\sqrt{2\mathrm{KL}(Q_z,P)} ]$ , where  $P$  is a prior distribution independent of the training set  $S$ , and  $Q_{z}$  is the expected posterior distribution conditioned on  $z_{n} = z$  (i.e., the last training data is  $z$ ). The formal definition and the results can be found in Definition 5 and Theorem 7.

Inspired by Lever et al. (2013), instead of using a fixed prior distribution, we bound the KL-divergence from the posterior to a distribution-dependent prior. This enables us to derive the following generalization error bound that depends on the expected norm of the gradient along the optimization path:

$$
\operatorname {e r r} _ {\mathrm {g e n}} = O \left(\frac {C}{n} \sqrt {\mathbb {E} _ {S} \left[ \sum_ {t = 1} ^ {T} \frac {\gamma_ {t} ^ {2}}{\sigma_ {t} ^ {2}} \mathbf {g} _ {\mathrm {e}} (t) \right]}\right). \tag {3}
$$

Here  $S$  is the dataset and  $\mathbf{g}_{\mathrm{e}}(t) = \mathbb{E}_{W_{t - 1}}[\frac{1}{n}\sum_{i = 1}^{n}\| \nabla F(W_{t - 1},z_i)\|^2 ]$  is the expected empirical squared gradient norm at step  $t$ ; see Theorem 11 for the details.

Compared with the previous  $O\left(\frac{LC}{n}\sqrt{\sum_{t}\frac{\gamma_{t}^{2}}{\sigma_{t}^{2}}}\right)$  bound in (Mou et al., 2018, Theorem 1), where  $L$  is the global Lipschitz constant of the loss, our new bound (3) depends on the data distribution and is typically tighter (as the gradient norm is at most  $L$ ). In modern deep neural networks, the worst-case Lipschitz constant  $L$  can be quite large, and typically much larger than the expected empirical gradient norm along the optimization trajectory. Specifically, in the later stage of the training, the expected empirical gradient is small (see Figure 1(d) for the details). Hence, our generalization bound does not grow much even if we train longer at this stage.

Our new bound also offers an explanation to the difference between training on correct and random labels observed by Zhang et al. (2017a). In particular, we show empirically that the sum of expected squared gradient norm (along the optimization path) is significantly higher when the training labels are replaced with random labels (Section 3, Remark 13, Figure 1, Appendix C.2).

We would also like to mention the PAC-Bayesian bound (for SGLD with  $\ell_2$ -regularization) proposed by Mou et al. (2018). (This bound is different from what we mentioned before; see Theorem 2 in their paper.) Their bound scales as  $O(1 / \sqrt{n})$  and the numerator of their bound has a similar sum of gradient norms (with a decaying weight if the regularization coefficient  $\lambda > 0$ ). Their bound is based on the PAC-Bayesian approach and holds with high probability, while our bound only holds in expectation.

Extensions. We remark that our technique allows for an arguably simpler proof of (Mou et al., 2018, Theorem 1); the original proof is based on SDE and Fokker-Planck equation. More importantly, our technique can be easily extended to handle mini-batches and a variety of general settings as follows.

1. Extension to other gradient-based methods. Our results naturally extend to other noisy stochastic gradient methods including momentum due to Polyak (1964) (Theorem 26), Nesterov's accelerated gradient method in Nesterov (1983) (Theorem 26), and Entropy-SGD proposed by Chaudhari et al. (2017) (Theorem 27).

2. Extension to general noises. The proof of the generalization bound in Mou et al. (2018) relies heavily on that the noise is Gaussian<sup>1</sup>, which makes it difficult to generalize to other noise distributions such as the Laplace distribution. In contrast, our analysis easily carries over to the class of log-Lipschitz noises (i.e., noises drawn from distributions with Lipschitz log densities).  
3. Pathwise stability. In practice, it is also natural to output a certain function of the entire optimization path, e.g., the one with the smallest empirical risk or a weighted average. We show that the same generalization bound holds for all such variants (Remark 12). We note that the analysis in an independent work of Pensia et al. (2018) also satisfies this property, yet their bound is  $O\left(\sqrt{C^2L^2n^{-1}\sum_{t=1}^{T}\eta_t^2 / \sigma_t^2}\right)$  (see Corollary 1 in their work), which scales at a slower  $O(1 / \sqrt{n})$  rate (instead of  $O(1 / n)$ ) when dealing with  $C$ -bounded loss. $^2$

Generalization bounds with  $\ell_2$  regularization via Log-Sobolev inequalities. We also study the setting where the total objective function  $F$  is the sum of a  $C$ -bounded differentiable objective  $F_0$  and an additional  $\ell_2$  regularization term  $\frac{\lambda}{2} \| w \|_2^2$ . In this case,  $F$  can be treated as a perturbation of a quadratic function, and the continuous Langevin dynamics (CLD) is well understood for quadratic functions. We obtain two generalization bounds for CLD, both via the technique of Log-Sobolev inequalities, a powerful tool for proving the convergence rate of CLD. One of our bounds is as follows (Theorem 15):

$$
\operatorname {e r r} _ {\mathrm {g e n}} \leq \frac {2 e ^ {4 \beta C} C L}{n} \sqrt {\frac {\beta}{\lambda} \left(1 - \exp \left(- \frac {\lambda T}{e ^ {8 \beta C}}\right)\right)}. \tag {4}
$$

The above bound has the following advantages:

1. Applying  $e^{-x} \geq 1 - x$ , one can see that our bound is at most  $O(\sqrt{T}/n)$ , which matches the previous bound in (Mou et al., 2018, Proposition 8) $^3$ .  
2. As time  $T$  grows, the bound is upper bounded by and approaches to  $2e^{4\beta C}CLn^{-1}\sqrt{\beta / \lambda}$  (unlike the previous  $O(\sqrt{T} /n)$  bound that goes to infinity as  $T\to +\infty$ ).  
3. If the noise level is not so small (i.e.,  $\beta$  is not very large), the generalization bound is quite desirable.

Our analysis is based on a Log-Sobolev inequality (LSI) for the parameter distribution at time  $t$ , whereas most known LSIs only hold for the stationary distribution of the Markov process. We prove the new LSI by exploiting the variational formulation of the entropy formula.

# 2 PRELIMINARIES

Notations. We use  $\mathcal{D}$  to denote the data distribution. The training dataset  $S = (z_{1},\ldots ,z_{n})$  is a sequence of  $n$  independent samples drawn from  $\mathcal{D}$ .  $S,S^{\prime}\in \mathcal{Z}^{n}$  are called neighboring datasets if and only if they differ at exactly one data point (we could assume without loss of generality that  $z_{n}\neq z_{n}^{\prime}$ ). Let  $F(w,z)$  and  $\mathcal{L}(w,z)$  be the objective and the loss functions, respectively, where  $w\in \mathbb{R}^d$  denotes a model parameter and  $z\in \mathcal{Z}$  is a data point. Define  $F(w,S) = \frac{1}{n}\sum_{i = 1}^{n}F(w,z_i)$  and  $F(w,\mathcal{D}) = \mathbb{E}_{z\sim \mathcal{D}}[F(w,z)]; \mathcal{L}(w,S)$  and  $\mathcal{L}(w,\mathcal{D})$  are defined similarly. A learning algorithm  $\mathcal{A}$  takes as input a dataset  $S$ , and outputs a parameter  $w\in \mathbb{R}^d$  randomly. Let  $G$  be the set of all possible mini-batches.  $G_{n} = \{B\in G:n\in B\}$  denotes the collection of mini-batches that contain the  $n$ -th data point, while  $\overline{G_n} = G\setminus G_n$ . Let  $\mathrm{diam}(A) = \sup_{x,y\in A}\| x - y\| _2$  denote the diameter of a set  $A$ .

Definition 1 (L-lipschitz). A function  $F: \mathbb{R}^d \times \mathcal{Z} \to \mathbb{R}$  is L-lipschitz if and only if  $|F(w_1, z) - F(w_2, z)| \leq L \| w_1 - w_2 \|_2$  holds for any  $w_1, w_2 \in \mathbb{R}^d$  and  $z \in \mathcal{Z}$ .

Definition 2 (Expected generalization error). The expected generalization error of a learning algorithm  $\mathcal{A}$  is defined as

$$
\operatorname {e r r} _ {g e n} := \underset {S \sim \mathcal {D} ^ {n}} {\mathbb {E}} [ \operatorname {e r r} _ {g e n} (S) ] = \underset {S \sim \mathcal {D} ^ {n}, \mathcal {A}} {\mathbb {E}} [ \mathcal {L} (\mathcal {A} (S), \mathcal {D}) - \mathcal {L} (\mathcal {A} (S), S) ].
$$

Algorithmic Stability. Intuitively, a learning algorithm that is stable (i.e., a small perturbation of the training data does not affect its output too much) can generalize well. In the seminal work of Bousquet & Elisseeff (2002) (see also Hardt et al. (2016)), the authors formally defined algorithmic stability and established a close connection between the stability of a learning algorithm and its generalization performance.

Definition 3 (Uniform stability). (Bousquet & Elisseeff (2002); Elisseeff et al. (2005)) A randomized algorithm  $\mathcal{A}$  is  $\epsilon_{n}$ -uniformly stable w.r.t. loss  $\mathcal{L}$ , if for all neighboring sets  $S, S' \in \mathcal{Z}^n$ , it holds that

$$
\sup _ {z \in \mathcal {Z}} | \mathbb {E} _ {\mathcal {A}} [ \mathcal {L} (w _ {S}, z) ] - \mathbb {E} _ {\mathcal {A}} [ \mathcal {L} (w _ {S ^ {\prime}}, z) ] | \leq \epsilon_ {n},
$$

where  $w_{S}$  and  $w_{S'}$  denote the outputs of  $\mathcal{A}$  on  $S$  and  $S'$  respectively.

Lemma 4 (Generalization in expectation). (Hardt et al. (2016)) Suppose a randomized algorithm  $\mathcal{A}$  is  $\epsilon_{n}$ -uniformly stable. Then,  $|\mathrm{err}_{gen}| \leq \epsilon_{n}$ .

# 3 BAYES-STABILITY METHOD

In this section, we incorporate ideas from the PAC-Bayesian theory (see e.g., Lever et al. (2013)) into the algorithmic stability framework. Combined with the technical tools introduced in previous sections, the new framework enables us to prove tighter data-dependent generalization bounds.

First, we define the posterior of a dataset and the posterior of a single data point.

Definition 5 (Single-point posterior). Let  $Q_S$  be the posterior distribution of the parameter for a given training dataset  $S = (z_1, \ldots, z_n)$ . In other words, it is the probability distribution of the output of the learning algorithm on dataset  $S$  (e.g., for  $T$  iterations of SGLD in (1),  $Q_S$  is the pdf of  $W_T$ ). The single-point posterior  $Q_{(i,z)}$  is defined as

$$
Q _ {(i, z)} = \underset {(z _ {1}, \ldots , z _ {i - 1}, z _ {i + 1}, \ldots , z _ {n})} {\mathbb {E}} \left[ Q _ {(z _ {1}, \ldots , z _ {i - 1}, z, z _ {i + 1}, \ldots , z _ {n})} \right].
$$

For convenience, we make the following natural assumption on the learning algorithm:

Assumption 6 (Order-independent). For any fixed dataset  $S = (z_{1},\ldots ,z_{n})$  and any permutation  $p$ ,  $Q_{S}$  is the same as  $Q_{Sp}$ , where  $S^p = (z_{p_1},\dots,z_{p_n})$ .

Assumption 6 implies  $Q_{(1,z)} = \dots = Q_{(n,z)}$ , so we use  $Q_{z}$  as a shorthand for  $Q_{(i,z)}$  in the following. Note that this assumption can be easily satisfied by letting the learning algorithm randomly permute the training data at the beginning. It is also easy to verify that both SGD and SGLD satisfy the order-independent assumption.

Now, we state our new Bayes-stability framework, which holds for any prior distribution  $P$  over the parameter space that is independent of the training dataset  $S$ .

Theorem 7 (Bayes-Stability). Suppose the loss function  $\mathcal{L}(w,z)$  is  $C$ -bounded and the learning algorithm is order-independent (Assumption 6). Then for any prior distribution  $P$  not depending on  $S$ , the generalization error is bounded by both  $2C\mathbb{E}_z\left[\sqrt{2\mathrm{KL}(P,Q_z)}\right]$  and  $2C\mathbb{E}_z\left[\sqrt{2\mathrm{KL}(Q_z,P)}\right]$ .

Remark 8. Our Bayes-Stability framework originates from the algorithmic stability framework, and hence is similar to the notions of uniform stability and leave-one-out error (see Elisseeff et al. (2003)). However, there are important differences. Uniform stability is a distribution-independent property, while Bayes-Stability can incorporate the information of the data distribution (through the prior  $P$ ). Leave-one-out error measures the loss of a learned model on an unseen data point, yet Bayes-Stability focuses on the extent to which a single data point affects the outcome of the learning algorithm (compared to the prior).

To establish an intuition, we first apply this framework to obtain an expectation generalization bound for (full) gradient Langevin dynamics (GLD), which is a special case of SGLD in (1) (i.e., GLD uses the full gradient  $\nabla_w F(W_{t-1}, S)$  as  $g_t(W_{t-1})$ ).

Theorem 9. Suppose that the loss function  $\mathcal{L}$  is  $C$ -bounded. Then we have the following expected generalization bound for  $T$  iterations of GLD:

$$
\operatorname {e r r} _ {g e n} \leq \frac {2 \sqrt {2} C}{n} \sqrt {\underset {S \sim \mathcal {D} ^ {n}} {\mathbb {E}} \left[ \sum_ {t = 1} ^ {T} \frac {\gamma_ {t} ^ {2}}{\sigma_ {t} ^ {2}} \mathbf {g} _ {\mathrm {e}} (t) \right]},
$$

where  $\mathbf{g}_{\mathrm{e}}(t) = \mathbb{E}_{w\sim W_{t - 1}}[\frac{1}{n}\sum_{i = 1}^{n}\| \nabla F(w,z_i)\| _2^2 ]$  is the empirical squared gradient norm, and  $W_{t}$  is the parameter at step  $t$  of GLD.

Proof The proof builds upon the following technical lemma, which we prove in Appendix A.2.

Lemma 10. Let  $(W_0, \ldots, W_T)$  and  $(W_0', \ldots, W_T')$  be two independent sequences of random variables such that for each  $t \in \{0, \ldots, T\}$ ,  $W_t$  and  $W_t'$  have the same support. Suppose  $W_0$  and  $W_0'$  follow the same distribution. Then,

$$
\operatorname {K L} \left(W _ {\leq T}, W _ {\leq T} ^ {\prime}\right) = \sum_ {t = 1} ^ {T} \underset {w _ {<   t} \sim W _ {<   t}} {\mathbb {E}} \left[ \operatorname {K L} \left(W _ {t} | W _ {<   t} = w _ {<   t}, W _ {t} ^ {\prime} | W _ {<   t} ^ {\prime} = w _ {<   t}\right) \right],
$$

where  $W_{\leq t}$  denotes  $(W_0,\dots ,W_t)$  and  $W_{<  t}$  denotes  $W_{\leq t - 1}$

Define  $P = \mathbb{E}_{\overline{S}\sim \mathcal{D}^{n - 1}}[Q_{(\overline{S},\mathbf{0})}]$ , where  $\mathbf{0}$  denotes the zero data point (i.e.,  $f(w,\mathbf{0}) = 0$  for any  $w$ ). Theorem 7 shows that

$$
\operatorname {e r r} _ {\text {g e n}} \leq 2 C _ {z} \sqrt {2 \mathrm {K L} (Q _ {z} , P)}. \tag {5}
$$

By the convexity of KL-divergence, for a fixed  $z \in \mathcal{Z}$ , we have

$$
\operatorname {K L} \left(Q _ {z}, P\right) = \operatorname {K L} \left(\frac {\mathbb {E}}{S} \left[ Q _ {\left(\bar {S}, z\right)} \right], \frac {\mathbb {E}}{S} \left[ Q _ {\left(\bar {S}, \mathbf {0}\right)} \right]\right) \leq \frac {\mathbb {E}}{S} \left[ \operatorname {K L} \left(Q _ {\left(\bar {S}, z\right)}, Q _ {\left(\bar {S}, \mathbf {0}\right)}\right) \right]. \tag {6}
$$

Let  $(W_{t})_{t\geq 0}$  and  $(W_{t}^{\prime})_{t\geq 0}$  be the training process of GLD for  $S = (\overline{S},z)$  and  $S^{\prime} = (\overline{S},0)$ , respectively. Note that for a fixed  $w_{< t}$ , both  $W_{t}|W_{< t} = w_{< t}$  and  $W_{t}^{\prime}|W_{< t}^{\prime} = w_{< t}$  are Gaussian distributions. Since  $\mathrm{KL}(\mathcal{N}(\mu_1,\sigma^2 I),\mathcal{N}(\mu_2,\sigma^2 I)) = \frac{\|\mu_1 - \mu_2\|_2^2}{2\sigma^2}$  (see Lemma 18 in Appendix A.2).

$$
\mathrm {K L} (W _ {t} | W _ {<   t} = w _ {<   t}, W _ {t} ^ {\prime} | W _ {<   t} ^ {\prime} = w _ {<   t}) = \frac {\gamma_ {t} ^ {2} \| \nabla F (w _ {t - 1} , z) \| _ {2} ^ {2}}{\sigma_ {t} ^ {2} n ^ {2}}.
$$

Applying Lemma 10 and  $\mathrm{KL}(W_T, W_T') \leq \mathrm{KL}(W_{\leq T}, W_{\leq T}')$  gives

$$
\mathrm {K L} (Q _ {S}, Q _ {S ^ {\prime}}) \leq \frac {1}{n ^ {2}} \sum_ {t = 1} ^ {T} \frac {\gamma_ {t} ^ {2}}{\sigma_ {t} ^ {2}} \underset {w \sim W _ {t - 1}} {\mathbb {E}} \| \nabla F (w, z) \| _ {2} ^ {2}.
$$

Recall that  $W_{t-1}$  is the parameter at step  $t-1$  using  $S = (\overline{S}, z)$  as a dataset. In this case, we can rewrite  $z$  as  $z_n$  since it is the  $n$ -th data point of  $S$ . Note that SGLD satisfies the order-independent assumption, we can rewrite  $z$  as  $z_i$  for all  $i \in [n]$ . Together with (5), (6), and using  $\frac{1}{n} \sum_{i=1}^{n} \sqrt{x_i} \leq \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i}$ , we can prove this theorem.

More generally, we give the following bound for SGLD. The proof is similar to that of Theorem 9; the difference is that we need to bound the KL-divergence between two Gaussian mixtures instead of two Gaussians. This proof is more technical and deferred to Appendix A.3.

Theorem 11. Suppose that the loss function  $\mathcal{L}$  is  $C$ -bounded and the objective function  $f$  is  $L$ -lipschitz. Assume that the following conditions hold:

1. Batch size  $b \leq n / 2$ .

2. Learning rate  $\gamma_{t} \leq \sigma_{t} / (20L)$ .

Then, the following expected generalization error bound holds for  $T$  iterations of SGLD (1):

$$
\operatorname {e r r} _ {g e n} \leq \frac {8 . 1 2 C}{n} \sqrt {\underset {S \sim \mathcal {D} ^ {n}} {\mathbb {E}} \left[ \sum_ {t = 1} ^ {T} \frac {\gamma_ {t} ^ {2}}{\sigma_ {t} ^ {2}} \mathbf {g} _ {\mathrm {e}} (t) \right]}, \quad (\text {e m p i r i c a l n o r m})
$$

where  $\mathbf{g}_{\mathrm{e}}(t) = \mathbb{E}_{w\sim W_{t - 1}}[\frac{1}{n}\sum_{i = 1}^{n}\| \nabla F(w,z_i)\| _2^2 ]$  is the empirical squared gradient norm, and  $W_{t}$  is the parameter at step  $t$  of SGLD.

Furthermore, based on essentially the same proof, we can obtain the following bound that depends on the population gradient norm:

$$
\operatorname {e r r} _ {\text {g e n}} \leq \frac {8 . 1 2 C}{n} \sqrt {\mathbb {E} _ {S ^ {\prime}} \left[ \sum_ {t = 1} ^ {T} \frac {\gamma_ {t} ^ {2}}{\sigma_ {t} ^ {2}} \underset {w \sim W ^ {\prime} _ {t - 1}} {\mathbb {E}} \left[ \underset {z \sim D} {\mathbb {E}} \| \nabla F (w , z) \| _ {2} ^ {2} \right] \right]}.
$$

The full proofs of the above results are postponed to Appendix A, and we provide some remarks about the new bounds.

![](images/bcbbd84ea901f47b29ce03f7133bce02af5cb4c8125781be6586885cc269b821.jpg)  
(a)

![](images/baa55fff70da5645cd4a01e4322e27344e5c40c699607d9fd2d92cd864c3554b.jpg)  
(b)  
Figure 1: Running GLD (with  $\sigma_t = 0.2\sqrt{2}\gamma_t$ ) on a smaller version of MNIST with different random label portion  $p$ . (a) shows the training accuracy. (b) shows the generalization error, i.e., the gap between the  $0/1$  loss  $\mathcal{L}^{01}$  on the training data and on the test data. (c) plots our bound in Theorem 9. (d) shows that for  $p = 0$ , the gradient norms become much smaller at later stages of training.

![](images/50256466a64f6ded220e32d7817dd11f3b649dfa1ab4b2a1d3f08493c54a0bc1.jpg)  
(c)

![](images/9cb3f835577535d9a6a8df23767a9b76fba0b81c1fb21d514d99814245cd75f0.jpg)  
(d)

Remark 12. In fact, our proof establishes that the above upper bound holds for the two sequences  $W_{\leq T}$  and  $W_{\leq T}'$ :  $\mathrm{KL}(W_{\leq T}, W_{\leq T}') \leq \frac{8.23}{n^2} \sum_{t=1}^{T} \frac{\gamma_t^2}{\sigma_t^2} \mathbf{g}_{\mathrm{e}}(t)$ . Hence, our bound holds for any sufficiently regular function over the parameter sequences:  $\mathrm{KL}(f(W_{\leq T}), f(W_{\leq T}') \leq \frac{8.23}{n^2} \sum_{t=1}^{T} \frac{\gamma_t^2}{\sigma_t^2} \mathbf{g}_{\mathrm{e}}(t)$ . In particular, our generalization error bound automatically extends to several variants of SGLD, such as outputting the average of the trajectory, the average of the suffix of certain length, or the exponential moving average.

Remark 13. Inspired by Zhang et al. (2017a), we run both GLD (Figure 1) and SGLD (Appendix C.2) to fit both normal data and randomly labelled data (see Appendix C for more experiment details). As shown in Figure 1 and Figure 2 in Appendix C.2, larger random label portion p leads to both much higher generalization error and much larger generalization error bound. Moreover, the shapes of the curves our bounds look quite similar to that of the generalization error curves.

# 4 GENERALIZATION OF CLD AND GLD WITH  $\ell_2$  REGULARIZATION

In this section, we study the generalization error of Continuous Langevin Dynamics (CLD) with  $\ell_2$  regularization. Throughout this section, we assume that the objective function over training set  $S$  is defined as  $F(w,S) = F_{0}(w,S) + \frac{\lambda}{2}\| w\|_{2}^{2}$ , and moreover, the following assumption holds.

Assumption 14. The loss function  $\mathcal{L}$  and the original objective  $F_{0}$  are  $C$ -bounded. Moreover,  $F_{0}$  is differentiable and  $L$ -lipschitz.

The Continuous Langevin Dynamics is defined by the following SDE:

$$
\mathrm {d} W _ {t} = - \nabla F (W _ {t}, S) \mathrm {d} t + \sqrt {2 \beta^ {- 1}} \mathrm {d} B _ {t}, \quad W _ {0} \sim \mu_ {0}, \tag {CLD}
$$

where  $(B_{t})_{t\geq 0}$  is the standard Brownian motion on  $\mathbb{R}^d$  and the initial distribution  $\mu_0$  is the centered Gaussian distribution in  $\mathbb{R}^d$  with covariance  $\frac{1}{\lambda\beta} I_d$ . We show that the generalization error of CLD is upper bounded by  $O\left(e^{4\beta C}n^{-1}\sqrt{\beta / \lambda}\right)$ , which is independent of the training time  $T$  (Theorem 15). Furthermore, as  $T$  goes to infinity, we have a tighter generalization error bound  $O\left(\beta C^2 n^{-1}\right)$  (Theorem 39 in Appendix B). We also study the generalization of Gradient Langevin Dynamics (GLD), which is the discretization of CLD:

$$
W _ {k + 1} = W _ {k} - \eta \nabla F (W _ {k}, S) + \sqrt {2 \eta \beta^ {- 1}} \xi_ {k}, \quad (\text {G L D})
$$

where  $\xi_{k}$  is the standard Gaussian random vector in  $\mathbb{R}^d$ . By leveraging a result developed in Raginsky et al. (2017), we show that, as  $K\eta^{2}$  tends to zero, GLD has the same generalization as CLD (see Theorems 15 and 39). We first formally state our first main result in this section.

Theorem 15. Under Assumption 14, CLD (with initial probability measure  $\mathrm{d}\mu_0 = \frac{1}{Z} e^{\frac{-\lambda\beta\|w\|^2}{2}}\mathrm{d}w$ ) has the following expected generalization error bound:

$$
\operatorname {e r r} _ {\text {g e n}} \leq \frac {2 e ^ {4 \beta C} C L}{n} \sqrt {\frac {\beta}{\lambda} \left(1 - \exp \left(- \frac {\lambda T}{e ^ {8 \beta C}}\right)\right)}. \tag {7}
$$

In addition, if  $\mathcal{L}$  is  $M$ -smooth and non-negative, by setting  $\lambda \beta > 2$ ,  $\lambda > \frac{1}{2}$  and  $\eta \in [0,1 \wedge \frac{2\lambda - 1}{8M^2})$ , GLD (running  $K$  iterations with the same  $\mu_0$  as CLD) has the expected generalization error bound:

$$
\operatorname {e r r} _ {\text {g e n}} \leq 2 C \sqrt {2 K C _ {1} \eta^ {2}} + \frac {2 C L e ^ {4 \beta C}}{n} \sqrt {\frac {\beta}{\lambda} \left(1 - \exp \left(- \frac {\lambda \eta K}{e ^ {8 \beta C}}\right)\right)}, \tag {8}
$$

where  $C_1$  is a constant that only depends on  $M$ ,  $\lambda$ ,  $\beta$ ,  $b$ ,  $L$  and  $d$ .

The following lemma is crucial for establishing the above generalization bound for CLD. In particular, we need to establish a Log-Sobolev inequality for  $\mu_t$ , the parameter distribution at time  $t$ , for every time step  $t > 0$ . In contrast, most known LSIs only characterize the stationary distribution of the Markov process. The proof of the lemma can be found in Appendix B.

Lemma 16. Under Assumption 14, let  $\mu_t$  be the probability measure of  $W_t$  in CLD (with  $\mathrm{d}\mu_0 = \frac{1}{Z} e^{\frac{-\lambda\beta\|w\|^2}{2}}\mathrm{d}w$ ). Let  $\nu$  be a probability measure that is absolutely continuous with respect to  $\mu_t$ . Suppose  $\mathrm{d}\mu_t = \pi_t(w)\mathrm{d}w$  and  $\mathrm{d}\nu = \gamma(w)\mathrm{d}w$ . Then, it holds that

$$
\mathrm {K L} (\gamma , \pi_ {t}) \leq \frac {\exp (8 \beta C)}{2 \lambda \beta} \int_ {\mathbb {R} ^ {d}} \left\| \nabla \log \frac {\gamma (w)}{\pi_ {t} (w)} \right\| _ {2} ^ {2} \gamma (w) \mathrm {d} w.
$$

We sketch the proof of Theorem 15, and the complete proof is relegated to Appendix B.

Proof Sketch of Theorem 15 Suppose  $S$  and  $S'$  are two neighboring datasets. Let  $(W_t)_{t \geq 0}$  and  $(W_t')_{t \geq 0}$  be the process of CLD running on  $S$  and  $S'$ , respectively. Let  $\gamma_t$  and  $\pi_t$  be the pdf of  $W_t'$  and  $W_t$ . Let  $F_S(w)$  denote  $F(w, S)$ . We have

$$
\begin{array}{l} \frac {\mathrm {d}}{\mathrm {d} t} \mathrm {K L} (\gamma_ {t}, \pi_ {t}) = \frac {- 1}{\beta} \int_ {\mathbb {R} ^ {d}} \gamma_ {t} \left\| \nabla \log \frac {\gamma_ {t}}{\pi_ {t}} \right\| _ {2} ^ {2} \mathrm {d} w + \int_ {\mathbb {R} ^ {d}} \gamma_ {t} \langle \nabla \log \frac {\gamma_ {t}}{\pi_ {t}}, \nabla F _ {S} - \nabla F _ {S ^ {\prime}} \rangle \mathrm {d} w \\ \leq \frac {- 1}{2 \beta} \int_ {\mathbb {R} ^ {d}} \gamma_ {t} \left\| \nabla \log \frac {\gamma_ {t}}{\pi_ {t}} \right\| _ {2} ^ {2} \mathrm {d} w + \frac {\beta}{2} \int_ {\mathbb {R} ^ {d}} \gamma_ {t} \left\| \nabla F _ {S} - \nabla F _ {S ^ {\prime}} \right\| _ {2} ^ {2} \mathrm {d} w. \\ \leq \frac {- \lambda}{e ^ {8 \beta C}} \mathrm {K L} \left(\gamma_ {t}, \pi_ {t}\right) + \frac {2 \beta L ^ {2}}{n ^ {2}} \tag {Lemma 16} \\ \end{array}
$$

Solving this inequality gives  $\mathrm{KL}(\gamma_t,\pi_t)\leq \frac{1}{n^2\lambda} 2\beta L^2 e^{8\beta C}(1 - e^{-\lambda t / e^{8\beta C}})$ . Hence the generalization error of CLD can be bounded by  $2C\sqrt{\frac{1}{2}\mathrm{KL}(\gamma_T,\pi_T)}$ , which proves the first part. The second part of the theorem follows from Lemma 36 in Appendix B.

Our second generalization bound for CLD (Theorem 39 in Appendix B) is

$$
\mathrm {e r r} _ {\mathrm {g e n}} \leq \frac {8 \beta C ^ {2}}{n} + 4 C \exp \left(\frac {- \lambda T}{e ^ {4 \beta C}}\right) \sqrt {\beta C}.
$$

The high level idea to prove this bound is very similar to that in Raginsky et al. (2017). We first observe that the (stationary) Gibbs distribution  $\mu$  has a small generalization error. Then, we bound the distance from  $\mu_t$  to  $\mu$ . In our setting, we can use the Holley-Stroock perturbation lemma which allows us to bound the Logarithmic Sobolev constant, and we can thus bound the above distance easily.

# 5 FUTURE DIRECTIONS

In this paper, we prove new generalization bounds for a variety of noisy gradient-based methods. Our current techniques can only handle continuous noises for which we can bound the KL-divergence. One future direction is to study the discrete noise introduced in SGD (in this case the KL-divergence may not be well defined). For either SGLD or CLD, if the noise level is small (i.e.,  $\beta$  is large), it may take a long time for the diffusion process to reach the stable distribution. Hence, another interesting future direction is to consider the local behavior and generalization of the diffusion process in finite time through the techniques developed in the studies of metastability (see e.g., Bovier et al. (2005); Bovier & den Hollander (2006); Tzen et al. (2018)). In particular, the technique may be helpful for further improving the bounds in Theorems 15 and 39 (when  $T$  is not very large).

# REFERENCES

Zeyuan Allen-Zhu, Yuanzhi Li, and Yingyu Liang. Learning and generalization in overparameterized neural networks, going beyond two layers. arXiv preprint arXiv:1811.04918, 2018.  
Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. In International Conference on Machine Learning (ICML), pp. 254-263, 2018.  
Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. arXiv preprint arXiv:1904.11955, 2019.  
Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and geometry of Markov diffusion operators, volume 348. Springer Science &amp; Business Media, 2013.  
Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. In Advances in Neural Information Processing Systems (NeurIPS), pp. 6240-6249, 2017.  
Olivier Bousquet and André Elisseeff. Stability and generalization. Journal of machine learning research, 2(Mar):499-526, 2002.  
Anton Bovier and Frank den Hollander. Metastability: a potential theoretic approach. In International Congress of Mathematicians, volume 3, pp. 499-518. Eur. Math. Soc. Zürich, 2006.  
Anton Bovier, Véronique Gayrard, and Markus Klein. Metastability in reversible diffusion processes ii: Precise asymptotics for small eigenvalues. Journal of the European Mathematical Society, 7 (1):69-99, 2005.  
Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-SGD: Biasing gradient descent into wide valleys. In International Conference on Learning Representations (ICLR), 2017.  
Yuansi Chen, Chi Jin, and Bin Yu. Stability and convergence trade-off of iterative optimization algorithms. arXiv preprint arXiv:1804.01619, 2018.  
Lenaic Chizat and Francis Bach. A note on lazy training in supervised differentiable programming. arXiv preprint arXiv:1812.07956, 2018.

Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International Conference on Machine Learning (ICML), pp. 1675-1685, 2019.  
John Duchi. Derivations for linear algebra and optimization. Berkeley, California, 3, 2007.  
Gintare Karolina Dziugaite and Daniel Roy. Entropy-SGD optimizes the prior of a PAC-Bayes bound: Generalization properties of entropy-SGD and data-dependent priors. In International Conference on Machine Learning (ICML), pp. 1377-1386, 2018.  
Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. In Uncertainty in Artificial Intelligence (UAI), 2017.  
André Elisseeff, Massimiliano Pontil, et al. Leave-one-out error and stability of learning algorithms with applications. NATO science series sub series iii computer and systems sciences, 190:111-130, 2003.  
Andre Elisseeff, Theodoros Evgeniou, and Massimiliano Pontil. Stability of randomized learning algorithms. Journal of Machine Learning Research, 6(Jan):55-79, 2005.  
Moritz Hardt, Benjamin Recht, and Yoram Singer. Train faster, generalize better: stability of stochastic gradient descent. In International Conference on Machine Learning (ICML), pp. 1225-1234, 2016.  
Richard Holley and Daniel Stroock. Logarithmic sobolev inequalities and stochastic ising models. Journal of statistical physics, 46(5):1159-1194, 1987.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NeurIPS), pp. 1097-1105, 2012.  
I. Kuzborskij and C. H. Lampert. Data-Dependent Stability of Stochastic Gradient Descent. In International Conference on Machine Learning (ICML), 2018.  
Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Guy Lever, François Laviolette, and John Shawe-Taylor. Tighter pac-bayes bounds through distribution-dependent priors. Theoretical Computer Science, 473:4-28, 2013.  
Tengyuan Liang, Tomaso Poggio, Alexander Rakhlin, and James Stokes. Fisher-rao metric, geometry, and complexity of neural networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 888-896, 2019.  
Ben London. Generalization bounds for randomized learning with application to stochastic gradient descent. In NIPS Workshop on Optimizing the Optimizers, 2016.  
Ben London. A pac-bayesian analysis of randomized learning with application to stochastic gradient descent. In Advances in Neural Information Processing Systems (NeurIPS), pp. 2931-2940, 2017.  
Georg Menz, André Schlichting, et al. Poincaré and logarithmic sobolev inequalities by decomposition of the energy landscape. The Annals of Probability, 42(5):1809-1884, 2014.  
Wenlong Mou, Liwei Wang, Xiyu Zhai, and Kai Zheng. Generalization bounds of SGLD for nonconvex learning: Two theoretical viewpoints. In Conference on Learning Theory (COLT), pp. 605-638, 2018.  
Yuri E Nesterov. A method for solving the convex programming problem with convergence rate  $\mathrm{O}(1 / k^2)$ . In Dokl. Akad. Nauk SSSR, volume 269, pp. 543-547, 1983.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Conference on Learning Theory (COLT), pp. 1376-1401, 2015.  
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nathan Srebro. A PAC-Bayesian approach to spectrally-normalized margin bounds for neural networks. In International Conference on Learning Representations (ICLR), 2018.  
Grigorios A Pavliotis. Stochastic processes and applications: diffusion processes, the Fokker-Planck and Langevin equations, volume 60. Springer, 2014.  
Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algorithms. In International Symposium on Information Theory (ISIT), pp. 546-550, 2018.  
Boris T Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.  
Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via stochastic gradient Langevin dynamics: a nonasymptotic analysis. In Conference on Learning Theory (COLT), pp. 1674–1703, 2017.  
Hannes Risken. Fokker-planck equation. In The Fokker-Planck Equation, pp. 63-95. Springer, 1996.  
Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International Conference on Machine Learning (ICML), pp. 1139-1147, 2013.  
Flemming Topsoe. Some inequalities for information divergence and related measures of discrimination. IEEE Transactions on Information Theory, 46(4):1602-1609, 2000.  
Belinda Tzen, Tengyuan Liang, and Maxim Raginsky. Local optimality and generalization guarantees for the Langevin algorithm via empirical metastability. Proceedings of the 2018 Conference on Learning Theory (COLT), 2018.  
Colin Wei and Tengyu Ma. Data-dependent sample complexity of deep neural networks via lipschitz augmentation. arXiv preprint arXiv:1905.03684, 2019.  
Colin Wei, Jason D Lee, Qiang Liu, and Tengyu Ma. On the margin theory of feedforward neural networks. arXiv preprint arXiv:1810.05369, 2018.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In International Conference on Machine Learning (ICML), pp. 681-688, 2011.  
Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. In Advances in Neural Information Processing Systems (NeurIPS), pp. 2524-2533, 2017.  
Pan Xu, Jinghui Chen, Difan Zou, and Quanquan Gu. Global convergence of Langevin dynamics based algorithms for nonconvex optimization. In Advances in Neural Information Processing Systems (NeurIPS), pp. 3126-3137, 2018.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations (ICLR), 2017a.  
Yuchen Zhang, Percy Liang, and Moses Charikar. A hitting time analysis of stochastic gradient Langevin dynamics. In Conference on Learning Theory (COLT), pp. 1980-2022, 2017b.  
Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes over-parameterized deep relu networks. arXiv preprint arXiv:1811.08888, 2018.
