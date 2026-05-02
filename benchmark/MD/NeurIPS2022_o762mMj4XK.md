# Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Modern approaches for simulation-based inference rely upon deep learning surrogates to enable approximate inference with computer simulators. In practice, the estimated posteriors' computational faithfulness is, however, rarely guaranteed. For example, Hermans et al. [1] show that current simulation-based inference algorithms can produce posteriors that are overconfident, hence risking false inferences. In this work, we introduce Balanced Neural Ratio Estimation (BNRE), a variation of the NRE algorithm [2] designed to produce posterior approximations that tend to be more conservative, hence improving their reliability, while sharing the same Bayes optimal solution. We achieve this by enforcing a balancing condition that increases the quantified uncertainty in small simulation budget regimes while still converging to the exact posterior as the budget increases. We provide theoretical arguments showing that BNRE tends to produce posterior surrogates that are more conservative than NRE's. We evaluate BNRE on a wide variety of tasks and show that it produces conservative posterior surrogates on all tested benchmarks and simulation budgets. Finally, we emphasize that BNRE is straightforward to implement over NRE and does not introduce any computational overhead.

# 1 Introduction

Many areas of science and engineering use parametric computer simulations to describe complex stochastic generative processes. In this setting, Bayesian inference provides a principled framework to identify parameters matching empirical observations. Computer simulations, however, define the necessary likelihood function only implicitly, which prevents its evaluation and the use of classical inference algorithms. To overcome this obstacle, recent simulation-based inference (SBI) algorithms [3] build upon deep learning surrogates to approximate parts of the Bayes rule and enable approximate inference. For example, [4, 5] build a surrogate of the likelihood function while [6, 7, 2, 8, 9] approximate the likelihood-to-evidence ratio. The posterior can also be targeted directly with variational inference, as proposed by [10, 11, 5]. These algorithms can each be made amortized or be run sequentially to drive the training towards a target observation and improve the simulation efficiency of the procedure [10, 12, 2, 11, 4, 8, 5]. However, sequential methods have the drawback of being computationally expensive to diagnose as the surrogates are only valid for the target observation [1]. Truncated marginal neural ratio estimation [9] alleviates this issue by introducing a sequential algorithm that builds a surrogate valid in a local region around the target.

Since modern simulation-based inference algorithms rely on deep learning surrogates, concerns naturally arise regarding their computational faithfulness and whether they are sufficiently adequate for the inference task of interest. In Bayesian inference, these concerns can be at least partially addressed with diagnostics designed to probe the correct behaviour of the inference method, such as  $\hat{R}$  diagnostics for MCMC [13], or to assess the quality of posterior approximations directly.

The latter include diagnostics such as simulation-based calibration (SBC) [14] or coverage-based diagnostics [15, 1]. As discussed by Hermans et al. [1], posterior approximations must be conservative to guarantee reliable and meaningful inferences, even when approximations are not faithful. For example, in the physical sciences, where the goal is often to constrain parameters of interest, wrongly excluding plausible values could drive the scientific inquiry in the wrong direction, whereas failing to exclude implausible values because of (too) conservative estimations is much less detrimental. Unfortunately, the same authors also demonstrate that current simulation-based inference algorithms can lead to overconfident surrogates and therefore false inferences. In this work, we develop a novel algorithm that not only converges to exact inference as the simulation budget increases, but which is also more likely to produce conservative surrogates in small simulation budget regimes. Towards this objective, we propose a variant of the NRE algorithm called Balanced Neural Ratio Estimation (BNRE), which enforces a balancing condition on the binary neural classifier to increase the reliability of its posterior approximations.

The structure of the manuscript is outlined as follows. Section 2 describes the formalism and the necessary background. Section 3 describes BNRE and provides theoretical arguments towards its conservativeness and reliability. Section 4 illustrates our main results and provides insights regarding the behaviour of the method. Finally, Section 5 discusses related work while Section 6 summarizes our contributions and hints at future work. Code is available at github.com/anonymous/anonymous.

# 2 Background

# 2.1 Statistical formalism

This work is concerned with simulation-based inference algorithms that produce posterior approximations  $\hat{p}(\boldsymbol{\vartheta} \mid \boldsymbol{x})$  under the following semantics. Target parameters  $\boldsymbol{\vartheta}$  denote the parameters of the model and we make the reasonable assumption that the prior  $p(\boldsymbol{\vartheta})$  is tractable. The model is generically expressed as a computer program, a simulator, that describes the forward dynamics of interest based on the input parameters  $\boldsymbol{\vartheta}$ . The simulator implicitly defines the likelihood function  $p(\boldsymbol{x} \mid \boldsymbol{\vartheta})$ . While we cannot directly evaluate the density  $p(\boldsymbol{x} \mid \boldsymbol{\vartheta})$ , we can execute the computer program to generate synthetic observables  $\boldsymbol{x} \sim p(\boldsymbol{x} \mid \boldsymbol{\vartheta})$ . Every observable  $\boldsymbol{x}_o$  is tied to ground truth parameters  $\boldsymbol{\vartheta}^*$  whose forward evaluation within the simulator produced  $\boldsymbol{x}^*$ .

Of special importance to Bayesians is the notion of a credible region, which is a domain  $\Theta$  within the target parameter space that satisfies  $\int_{\Theta} p(\vartheta \mid x = x^{*}) \mathrm{d}\vartheta = 1 - \alpha$  for some observable  $x^{*}$  and confidence level  $1 - \alpha$ . Because many such regions exist, we target the credible region with the smallest volume, also known as the highest posterior density region [16, 17].

# 2.2 Neural ratio estimation

Neural Ratio Estimation (NRE) is an established approach in the simulation-based inference literature both from frequentist [6] and Bayesian [7, 2, 8, 9] perspectives. In essence, all protocols rely on the density-ratio trick [18, 19, 6] to construct a surrogate of a likelihood ratio. In this work, we consider an amortized estimator  $\hat{r}(\boldsymbol{x}|\boldsymbol{\vartheta})$  of the intractable likelihood-to-evidence ratio  $r(\boldsymbol{x}|\boldsymbol{\vartheta}) = p(\boldsymbol{\vartheta},\boldsymbol{x}) / p(\boldsymbol{\vartheta})p(\boldsymbol{x}) = p(\boldsymbol{x}|\boldsymbol{\vartheta}) / p(\boldsymbol{x})$  that can be learned by training a binary classifier to distinguish between samples of the joint  $p(\boldsymbol{\vartheta},\boldsymbol{x})$  with class label 1 and samples of the product of marginals  $p(\boldsymbol{\vartheta})p(\boldsymbol{x})$  with class label 0. For the binary cross-entropy loss, the Bayes optimal classifier is

$$
d (\boldsymbol {\vartheta}, \boldsymbol {x}) = \frac {p (\boldsymbol {\vartheta} , \boldsymbol {x})}{p (\boldsymbol {\vartheta} , \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})} = \sigma \left(\log \frac {p (\boldsymbol {\vartheta} , \boldsymbol {x})}{p (\boldsymbol {\vartheta}) p (\boldsymbol {x})}\right), \tag {1}
$$

where  $\sigma (\cdot)$  is the sigmoid function. Given target parameters  $\pmb{\vartheta}$  and an observable  $\pmb{x}$  supported by  $p(\pmb{\vartheta})$  and  $p(\pmb{x})$  respectively, the learned classifier  $\hat{d}$  provides an approximation for the log likelihood-to-evidence ratio  $\log r(\pmb{x}|\pmb{\vartheta})$  because  $\log r(\pmb{x}|\pmb{\vartheta}) = \mathrm{logit}(d(\pmb{\vartheta},\pmb{x}))\approx \mathrm{logit}(\hat{d} (\pmb{\vartheta},\pmb{x})) = \log \hat{r} (\pmb{x}|\pmb{\vartheta})$ . The log posterior density function is approximated as  $\log \hat{p} (\pmb {\vartheta}|\pmb {x}) = \log p(\pmb {\vartheta}) + \log \hat{r} (\pmb {x}|\pmb {\vartheta})$ .

# 3 Balanced binary classification for neural ratio estimation

Following Hermans et al. [1], let us first define the expected coverage probability of the  $1 - \alpha$  highest posterior density regions derived from the posterior estimator  $\hat{p}(\boldsymbol{\vartheta} \mid \boldsymbol{x})$  as

$$
\mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} \left[ \mathbb {1} \left[ \boldsymbol {\vartheta} \in \Theta_ {\hat {p} (\boldsymbol {\vartheta} \mid \boldsymbol {x})} (1 - \alpha) \right] \right], \tag {2}
$$

where the function  $\Theta_{\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})}(1 - \alpha)$  yields the  $1 - \alpha$  highest posterior density region of  $\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})$ . This diagnostic probes the conservativeness of the posterior estimator (or the lack thereof) and can be interpreted as the expected frequentist coverage  $\mathbb{E}_{p(\boldsymbol {\vartheta})}\mathbb{E}_{p(\boldsymbol {x}\mid \boldsymbol {\vartheta})}\left[\mathbb{1}\left[\boldsymbol {\vartheta}\in \Theta_{\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})}(1 - \alpha)\right]\right]$ .

In this work, a posterior estimator has coverage at the confidence level  $1 - \alpha$  whenever the expected coverage probability is larger or equal to the nominal coverage probability,  $1 - \alpha$ . We say that a posterior estimator is conservative when it has coverage for all confidence levels. The expected coverage probability can be plotted for various levels  $\alpha$ , which allows to visually identify conservative posterior estimators. The expected coverage can also be shown to be a special case of the SBC diagnostic [14] (see Appendix A), further motivating the usage of expected coverage.

Our main objective is to restrict the hypothesis space of the approximate classifiers  $\hat{d}$  to those leading to conservative posterior estimators, hence solving the reliability concerns of NRE. Towards this goal, we construct a hypothesis space of balanced classifiers and show both theoretically and empirically that they lead to posterior estimators that tend to be more conservative.

# 3.1 Balanced binary classification

Definition 1. A classifier  $\hat{d}$  is balanced if  $\mathbb{E}_{p(\boldsymbol{\vartheta},\boldsymbol{x})}\left[\hat{d}(\boldsymbol{\vartheta},\boldsymbol{x})\right] = \mathbb{E}_{p(\boldsymbol{\vartheta})p(\boldsymbol{x})}\left[1 - \hat{d}(\boldsymbol{\vartheta},\boldsymbol{x})\right]$ , or

$$
\mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} \left[ \hat {d} (\boldsymbol {\vartheta}, \boldsymbol {x}) \right] + \mathbb {E} _ {p (\boldsymbol {\vartheta}) p (\boldsymbol {x})} \left[ \hat {d} (\boldsymbol {\vartheta}, \boldsymbol {x}) \right] = 1. \tag {3}
$$

Theorem 1. Any balanced classifier  $\hat{d}$  satisfies  $\mathbb{E}_{p(\vartheta, \boldsymbol{x})} \left[ \frac{d(\vartheta, \boldsymbol{x})}{\hat{d}(\vartheta, \boldsymbol{x})} \right] \geq 1$ .

Proof. The integral form of the balancing condition

$$
\iint \left(p (\boldsymbol {\vartheta}, \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})\right) \hat {d} (\boldsymbol {\vartheta}, \boldsymbol {x}) \mathrm {d} \boldsymbol {\vartheta} \mathrm {d} \boldsymbol {x} = 1 \tag {4}
$$

implies that  $\left(p(\pmb{x},\pmb{\vartheta}) + p(\pmb{\vartheta})p(\pmb{x})\right)\hat{d}(\pmb{\vartheta},\pmb{x})$  is a valid density, both integrating to 1 and positive everywhere. Therefore, its Kullback-Leibler (KL) divergence with  $p(\pmb{\vartheta},\pmb{x})$  is positive. Through Jensen's inequality, we obtain

$$
\begin{array}{l} 0 \leq \operatorname {K L} \left(p (\boldsymbol {\vartheta}, \boldsymbol {x}) \mid \mid (p (\boldsymbol {\vartheta}, \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})) \hat {d} (\boldsymbol {\vartheta}, \boldsymbol {x})\right) \\ \leq \mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} \left[ \log \frac {p (\boldsymbol {\vartheta} , \boldsymbol {x})}{(p (\boldsymbol {\vartheta} , \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})) \hat {d} (\boldsymbol {\vartheta} , \boldsymbol {x})} \right] \\ \leq \mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} \left[ \log \frac {d (\boldsymbol {\vartheta} , \boldsymbol {x})}{\hat {d} (\boldsymbol {\vartheta} , \boldsymbol {x})} \right] \\ \Rightarrow \quad 1 \leq \mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} \left[ \exp \left(\log \frac {d (\boldsymbol {\vartheta} , \boldsymbol {x})}{\hat {d} (\boldsymbol {\vartheta} , \boldsymbol {x})}\right) \right] = \mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} \left[ \frac {d (\boldsymbol {\vartheta} , \boldsymbol {x})}{\hat {d} (\boldsymbol {\vartheta} , \boldsymbol {x})} \right]. \\ \end{array}
$$

Theorem 2. Any balanced classifier  $\hat{d}$  satisfies  $\mathbb{E}_{p(\boldsymbol{\vartheta})p(\boldsymbol{x})}\left[\frac{1 - d(\boldsymbol{\vartheta},\boldsymbol{x})}{1 - \hat{d}(\boldsymbol{\vartheta},\boldsymbol{x})}\right] \geq 1$ .

Proof. Similar to Theorem 1, see Appendix B.

Theorem 1 shows that, in expectation over the joint distribution  $p(\vartheta, \boldsymbol{x})$ , a balanced classifier  $\hat{d}$  tends to make predictions whose probability values  $\hat{d}(\vartheta, \boldsymbol{x})$  are smaller than the exact probability values  $d(\vartheta, \boldsymbol{x})$ . In other words, a balanced classifier  $\hat{d}$  tends to be less confident than the Bayes optimal classifier  $d$ . Similarly, Theorem 2 shows that, in expectation over the product of the marginals  $p(\vartheta)p(\boldsymbol{x})$ , a balanced classifier tends to make predictions whose probability values  $1 - \hat{d}(\vartheta, \boldsymbol{x})$  are smaller than the exact probability values  $1 - d(\vartheta, \boldsymbol{x})$ , hence showing that a balanced classifier  $\hat{d}$  tends to also be less confident than the Bayes optimal classifier  $d$ . We note however that these two theorems hold only in expectation, which implies that neither  $\hat{d}(\vartheta, \boldsymbol{x}) \leq d(\vartheta, \boldsymbol{x})$  for all  $\vartheta, \boldsymbol{x}$  nor  $1 - \hat{d}(\vartheta, \boldsymbol{x}) \leq 1 - d(\vartheta, \boldsymbol{x})$  for all  $\vartheta, \boldsymbol{x}$  can generally be guaranteed.

Theorem 3. The Bayes optimal classifier  $d(\vartheta, \mathbf{x})$  is balanced.

Proof. Replacing the Bayes optimal classifier

$$
d (\boldsymbol {\vartheta}, \boldsymbol {x}) \triangleq \frac {p (\boldsymbol {\vartheta} , \boldsymbol {x})}{p (\boldsymbol {\vartheta} , \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})} \tag {5}
$$

in the integral form of the balancing condition, we have

$$
\begin{array}{l} \iint (p (\boldsymbol {\vartheta}, \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})) d (\boldsymbol {\vartheta}, \boldsymbol {x}) \mathrm {d} \boldsymbol {\vartheta} \mathrm {d} \boldsymbol {x} \\ = \iint \frac {\left(p (\boldsymbol {\vartheta} , \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})\right) p (\boldsymbol {\vartheta} , \boldsymbol {x})}{p (\boldsymbol {\vartheta} , \boldsymbol {x}) + p (\boldsymbol {\vartheta}) p (\boldsymbol {x})} d \boldsymbol {\vartheta} d \boldsymbol {x} \\ = \iint p (\boldsymbol {\vartheta}, \boldsymbol {x}) \mathrm {d} \boldsymbol {\vartheta} \mathrm {d} \boldsymbol {x} = 1. \\ \end{array}
$$

Theorem 3 states that the Bayes optimal classifier is balanced. Therefore, restricting the model hypothesis space to balanced classifiers does not modify the global optimum.

# 3.2 Balanced neural ratio estimation

We now extend the NRE algorithm to enforce the balancing condition. On the one hand, the previous results show that enforcing the condition should result in increasingly conservative classifiers  $\hat{d}$  and therefore to dispersed posterior approximations. Ideally, whenever  $\hat{d}(\vartheta, \mathbf{x}) \leq d(\vartheta, \mathbf{x})$  then

$$
\frac {\hat {d} (\boldsymbol {\vartheta} , \boldsymbol {x})}{1 - \hat {d} (\boldsymbol {\vartheta} , \boldsymbol {x})} \leq \frac {d (\boldsymbol {\vartheta} , \boldsymbol {x})}{1 - d (\boldsymbol {\vartheta} , \boldsymbol {x})}, \text {w h i c h i s e q u i v a l e n t t o} \hat {r} (\boldsymbol {x} \mid \boldsymbol {\vartheta}) \leq r (\boldsymbol {x} \mid \boldsymbol {\vartheta}), \tag {6}
$$

and  $\hat{p}(\boldsymbol{\vartheta} \mid \boldsymbol{x}) \leq p(\boldsymbol{\vartheta} \mid \boldsymbol{x})$  since  $\hat{p}(\boldsymbol{\vartheta} \mid \boldsymbol{x}) = p(\boldsymbol{\vartheta})\hat{r}(\boldsymbol{x} \mid \boldsymbol{\vartheta})$ . On the other hand, the classifier  $\hat{d}$  may not be better than a random classifier, which results in probability values  $\hat{d}(\boldsymbol{\vartheta}, \boldsymbol{x}) = 0.5$  for all  $\boldsymbol{\vartheta}, \boldsymbol{x}$ . In that case,  $\hat{r}(\boldsymbol{x} \mid \boldsymbol{\vartheta}) = 1$  and the approximate posterior  $\hat{p}(\boldsymbol{\vartheta} \mid \boldsymbol{x})$  degenerates to the prior  $p(\boldsymbol{\vartheta})$ . Overall, imposing the balancing condition should therefore result in approximate posteriors that lie between the prior and the exact posterior, without being more confident than they should.

Practically, the balancing condition can be targeted through a regularization penalty. For the binary cross-entropy  $\mathcal{L}\left[\hat{d}\right] \triangleq -\mathbb{E}_{p(\boldsymbol{\vartheta}, \boldsymbol{x})}\left[\log \hat{d}(\boldsymbol{\vartheta}, \boldsymbol{x})\right] - \mathbb{E}_{p(\boldsymbol{\vartheta})p(\boldsymbol{x})}\left[\log(1 - \hat{d}(\boldsymbol{\vartheta}, \boldsymbol{x}))\right]$  and given that the balancing condition only depends on samples from  $p(\boldsymbol{x})p(\boldsymbol{\vartheta})$  and  $p(\boldsymbol{x}, \boldsymbol{\vartheta})$ , the full loss functional including the balancing condition can be expressed as

$$
\mathcal {L} _ {b} [ \hat {d} ] \triangleq \mathcal {L} [ \hat {d} ] + \lambda (\mathbb {E} _ {p (\boldsymbol {\vartheta}) p (\boldsymbol {x})} [ \hat {d} (\boldsymbol {\vartheta}, \boldsymbol {x}) ] + \mathbb {E} _ {p (\boldsymbol {\vartheta}, \boldsymbol {x})} [ \hat {d} (\boldsymbol {\vartheta}, \boldsymbol {x}) ] - 1) ^ {2}, \tag {7}
$$

where  $\lambda$  is a (scalar) hyper-parameter controlling the strength of the balancing condition's contribution. The training procedure is summarized in Algorithm 1. Since the balancing condition needs to be 0 for a classifier to be balanced, the hyper-parameter controlling the strength of the balancing condition could, in principle, be set arbitrarily large. However, as the balancing condition is estimated via Monte Carlo sampling, setting  $\lambda$  to a large value could impair the classifier's learning ability. We found that  $\lambda = 100$  works well across many problem domains with varying simulation budgets.

Algorithm 1 Training algorithm for Balanced Neural Ratio Estimation (BNRE). Inputs: Implicit generative model  $p(\boldsymbol {x}\mid \boldsymbol {\vartheta})$  (simulator) and prior  $p(\vartheta)$  Outputs: Approximate classifier  $\hat{d}_{\psi}(\vartheta ,\boldsymbol {x})$  parameterized by  $\psi$  hyper-parameters: Balancing condition strength  $\lambda$  (default  $= 100$  ) and batch-size  $n$  repeat Sample data from the joint  $\{\vartheta_{i},\boldsymbol{x}_{i}\sim p(\boldsymbol {\vartheta},\boldsymbol {x}),y_{i} = 1\}_{i = 1}^{n / 2}$  Sample data from the marginals  $\{\vartheta_i,\boldsymbol {x}_i\sim p(\boldsymbol {\vartheta})p(\boldsymbol {x}),y_i = 0\}_{i = n / 2 + 1}^n$ $\mathcal{L}[\hat{d}_{\psi}] = -\frac{1}{n}\sum_{i = 1}^{n}y_{i}\log \hat{d}_{\psi}(\boldsymbol{\vartheta}_{i},\boldsymbol{x}_{i}) + (1 - y_{i})\log (1 - \hat{d}_{\psi}(\boldsymbol{\vartheta}_{i},\boldsymbol{x}_{i}))$ $\mathcal{B}[\hat{d}_{\psi}] = \frac{2}{n}\sum_{i = 1}^{n / 2}\hat{d}_{\psi}(\boldsymbol{\vartheta}_{i},\boldsymbol{x}_{i}) + \frac{2}{n}\sum_{i = n / 2 + 1}^{n}\hat{d}_{\psi}(\boldsymbol{\vartheta}_{i},\boldsymbol{x}_{i})$ $\psi =$  minimizer_step.params  $\coloneqq$ $\psi$  loss  $\coloneqq$  L[hat]  $+\lambda (\mathcal{B}[\hat{d}_{\psi}] - 1)^{2})$  until convergence return  $\hat{d}_{\psi}(\vartheta ,\boldsymbol {x})$

# 4 Experiments

We start by providing an extensive validation of BNRE on a broad range of benchmarks demonstrating that the proposed method alleviates the problem. Section 4.2 follows up with an illustrative demonstration on the behaviour of BNRE and its hyper-parameters.

# 4.1 Extensive validation

Setup We evaluate the expected coverage of posterior estimators produced by both NRE and BNRE on various problems, whose descriptions can be found in Appendix C. The architectures and hyperparameters used for each problem are defined in Appendix D. Our evaluation considers simulation budgets of increasing size, ranging from  $2^{10} = 1024$  to  $2^{17} = 131$ , 072 samples, and credibility levels from 0.05 to 0.95. For every simulation budget, we train 5 posterior estimators for 500 epochs and determine the credible region by evaluating the approximated posterior density function in a discretized and empirically normalized grid of the parameter space with sufficient resolution. The subsequent credible region is the set of parameters whose estimated (and normalized) posterior density is higher or equal to an inclusion threshold fitted to obtain the desired credibility level  $1 - \alpha$ . Details on this procedure are described in Appendix E. The expected coverage probability is estimated on 10000 unseen samples from the joint  $p(\vartheta, x)$ , for each considered credibility level.

Expected coverage The expected coverage curves and their interpretation are detailed in Figure 1. We observe that NRE often produces posterior estimators that are overconfident, especially for small simulation budgets. However, NRE's reliability increases with the availability of training data. By contrast, BNRE produces posterior estimators that are conservative on all benchmarks for all simulation budgets. Figure 2 explores the same phenomena, highlighting the effect of the simulation budget, by computing the integrated signed area between the expected coverage curve and the diagonal of a particular simulation which we call Coverage AUC. From this quantity it is evident there is a clear distinction between NRE and BNRE with respect to the available simulation budget. Both methods have the tendency to converge towards 0, indicating both methods are moving closer to the Bayes optimal classifier. However, the difference between these methods lies with how this solution is approached. While NRE can approach this limit from both sides, BNRE consistently produces coverage AUC's above 0, corresponding to conservative posterior approximations, and therefore exhibits the desired behaviour (in expectation).

Statistical performance In addition to the reliability of the posteriors, we evaluate and compare the statistical performance of the posterior approximations produced by NRE and BNRE. We estimate the expected approximate log posterior density  $\mathbb{E}_{p(\boldsymbol{\vartheta},\boldsymbol{x})}\big[\log \hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})\big]$  over a large number of pairs  $\boldsymbol{\vartheta}$ ,  $\boldsymbol{x}$ , which captures how well the posterior surrogates  $\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})$  approximate the true posteriors  $p(\boldsymbol {\vartheta}\mid \boldsymbol {x})$  since  $\mathbb{E}_{p(\boldsymbol {\vartheta},\boldsymbol{x})}[\log \hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})] = -\mathbb{E}_{p(\boldsymbol{x})}\mathrm{KL}\left[p(\boldsymbol {\vartheta}\mid \boldsymbol {x})||\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})\right] + \mathbb{E}_{p(x)}\mathbb{E}_{p(\boldsymbol {\vartheta}\mid \boldsymbol {x})}[\log p(\boldsymbol {\vartheta}\mid \boldsymbol {x})][20]$ .

Figure 3 shows our results. We observe that enforcing the balancing condition for  $\lambda = 100$  is associated with a loss in statistical performance. However, the loss in statistical performance is

![](images/8f1b0b553e255fcfb09dd0eb80dbd71fa9552bc4b8b4a4cbed17f9d02e14fa76.jpg)  
Figure 1: Expected coverage for increasing simulation budgets. A perfectly calibrated posterior has an expected coverage probability equal to the nominal coverage probability and hence produces a diagonal line. A conservative estimator has an expected coverage curve at or above the diagonal line, while an overconfident estimator produces curves below the diagonal line. The diagnostic therefore provides an immediate visual interpretation. We observe that NRE can produce overconfident estimators, while BNRE always produces coverage curves above the diagonal line and therefore the desired behaviour: conservative posterior approximations. The means over 5 runs are reported.

![](images/b1d5d0d0332be3b388bb8ecb35a7ae8e7e0cbbf33691ab90270e25e712c4acfc.jpg)  
Figure 2: Coverage AUC measures the integrated signed area between the expected coverage curve and the diagonal. A perfectly calibrated posterior has an expected coverage probability equal to the nominal coverage probability, producing a diagonal line and has a coverage AUC of 0, as shown on the left subplot. A conservative estimator on the other hand has a coverage AUC larger than 0 and an overconfident estimator smaller than 0. We observe that while NRE can produce coverage AUC both below or above 0, BNRE always produces a coverage AUC larger than 0, implying that its posterior approximations are conservative on average. The means over 5 runs are reported. A complete overview, including standard deviations, are provided in Appendix F.

eventually recovered by increasing the simulation budget. In fact, practitioners might be inclined to favor reliability over statistical performance [1] and would therefore be willing to cover this cost. Nevertheless, it is possible to improve the statistical performance by tuning the surrogate, or by increasing the available simulation budget as we have demonstrated.

# 4.2 In-depth analysis

In this section, we consider the Weinberg benchmark as described in Appendix C. The quality of the posterior approximations produced by BNRE is initially discussed with respect to the simulation budget. Afterwards, the effects of the hyper-parameter  $\lambda$  are studied.

![](images/f6cc41ff5b1cba1cc408bfa896ab462be86b66d13a04a9acff2254dbda195f14.jpg)  
Figure 3: Expected value  $\mathbb{E}_{p(\boldsymbol{\vartheta},\boldsymbol{x})}\left[\log \hat{p} (\boldsymbol {\vartheta}|\boldsymbol {x})\right]$  of the approximate log posterior density of the nominal parameters with respect to the simulation budget. We observe that BNRE produces log posterior densities lower than NRE. This shows that enforcing the balancing condition to have more reliable posterior approximates comes at the price of a small loss in information gain. However, BNRE improves over the prior and eventually converges towards NRE as the simulation budget increases. Solid lines represent the mean over 5 runs and shaded areas represent the standard deviation.

![](images/2e90b0b3f449d210742b2ea5711ae0a76313911b7ac3322d873ee9b2a78d1531.jpg)  
Figure 4: Comparison between NRE and BNRE in terms of expected coverage, bias and variance on the Weinberg benchmark. On the left side, the coverage is shown with respect to the simulation budget represented by the colormap. The bias and variance are represented on the right side of the plot. BNRE is run with  $\lambda = 100$ . Consistent with our previous observations in Figure 3, we observe that the gap in both bias and variance reduces as the simulation budget increases. Furthermore, in contrast with NRE, the posterior approximations of BNRE are tending towards being increasingly calibrated while at the same time being conservative. Solid lines represent the mean over 5 runs and shaded areas represent the standard deviation.

![](images/468019c60a57ba24332c68f3a8bd945f8f5aa8da8316b827693cf5b67890a73c.jpg)

Quality assessment Because the expected coverage does not capture the quality of an approximation in terms of information gain, we complement our assessment with a bias and variance analysis of the posterior approximations. Let us consider the expected squared error over the approximate posterior  $\mathbb{E}_{\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})}\left[(\boldsymbol {\vartheta} - \boldsymbol{\vartheta}^{*})^{2}\right]$ . With  $\bar{\vartheta} (\boldsymbol {x}) = \mathbb{E}_{\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})}[\boldsymbol {\vartheta}]$ , we decompose  $\mathbb{E}_{\hat{p} (\boldsymbol {\vartheta}\mid \boldsymbol {x})}\left[(\boldsymbol {\vartheta} - \boldsymbol{\vartheta}^{*})^{2}\right]$  as

$$
\begin{array}{l} \mathbb {E} _ {\hat {p} (\boldsymbol {\vartheta} \mid \boldsymbol {x})} \left[ \left(\boldsymbol {\vartheta} - \bar {\boldsymbol {\vartheta}} (\boldsymbol {x})\right) ^ {2} \right] + 2 \left(\bar {\boldsymbol {\vartheta}} (\boldsymbol {x}) - \boldsymbol {\vartheta} ^ {*}\right) \underbrace {\mathbb {E} _ {\hat {p} (\boldsymbol {\vartheta} \mid \boldsymbol {x})} \left[ \left(\boldsymbol {\vartheta} - \bar {\boldsymbol {\vartheta}} (\boldsymbol {x})\right) \right]} _ {= 0} + \mathbb {E} _ {\hat {p} (\boldsymbol {\vartheta} \mid \boldsymbol {x})} \left[ \left(\bar {\boldsymbol {\vartheta}} (\boldsymbol {x}) - \boldsymbol {\vartheta} ^ {*}\right) ^ {2} \right] \\ = \mathbb {E} _ {\hat {\rho} (\boldsymbol {\vartheta} \mid \boldsymbol {x})} \left[ \left(\boldsymbol {\vartheta} - \bar {\boldsymbol {\vartheta}} (\boldsymbol {x})\right) ^ {2} \right] + \left(\bar {\boldsymbol {\vartheta}} (\boldsymbol {x}) - \boldsymbol {\vartheta} ^ {*}\right) ^ {2}. \\ \end{array}
$$

The expectation over the joint distribution  $p(\boldsymbol{\vartheta}^*, \boldsymbol{x})$  of the expected squared error can hence be decomposed in a bias term defined as

$$
\operatorname {b i a s} \left(\hat {p} (\boldsymbol {\vartheta} \mid \boldsymbol {x})\right) \triangleq \mathbb {E} _ {p \left(\boldsymbol {\vartheta} ^ {*}, \boldsymbol {x}\right)} \left[ \left(\bar {\boldsymbol {\vartheta}} (\boldsymbol {x}) - \boldsymbol {\vartheta} ^ {*}\right) ^ {2} \right], \tag {8}
$$

which can be interpreted as the expected discrepancy between the nominal value  $\vartheta^{*}$  and the expected posterior value  $\bar{\vartheta}$ . The variance term is

$$
\operatorname {v a r i a n c e} \left(\hat {p} (\boldsymbol {\vartheta} \mid \boldsymbol {x})\right) \triangleq \mathbb {E} _ {p \left(\boldsymbol {\vartheta} ^ {*}, \boldsymbol {x}\right)} \left[ \mathbb {E} _ {\hat {p} \left(\boldsymbol {\vartheta} \mid \boldsymbol {x}\right)} \left[ \left(\boldsymbol {\vartheta} - \bar {\boldsymbol {\vartheta}} (\boldsymbol {x})\right) ^ {2} \right] \right] \tag {9}
$$

![](images/b10d6f90bf931641cf1c97381d0ef3b22edee057b789164ad05455870722341a.jpg)

![](images/a2a2ec93eb5f5c7018d0f7fee55a66a44957e46163649078cb6cf099b49d3347.jpg)

![](images/e6c825b2a1174d437d81670acacd3b6ff5603d42737c9868d4eff4eed63c3762.jpg)

![](images/b4847a327f59b57b86302cd0b7a9c70b8c1eec1fc801c97ed49437e28f4ff0a9.jpg)

![](images/30ab1c99fcc7ffb39864e34cf10b9dd6a47060ce851d82e375382e962342ff33.jpg)  
Figure 5: Effect of the hyper-parameter  $\lambda$  for a fixed simulation budget of 1024. The first plot shows the evolution of the approximate posterior for a given observation at a fixed  $\vartheta^{*}$ , indicated by the red vertical line. The second plot illustrates the empirical expected coverage. The third plot provides a summarized view of the second plot using the coverage AUC as summary statistic. The fourth plot shows that classifiers are becoming increasingly more balanced as  $\lambda$  increases. In addition, the plots show that  $\lambda$  is directly tied to the statistical performance and reliability of the posterior approximations. In general, we observe that classifiers trained with small  $\lambda$ 's are associated with (relatively) tight posteriors and overconfident approximations, while classifiers trained with larger values of  $\lambda$  are increasingly more dispersed and conservative until the posterior approximations reduce to the prior due to inflated statistical noise of the Monte Carlo estimation of the balancing condition.

and measures the dispersion of the posterior approximations. Note that these terms differ from the typical statistical bias and variance of point estimators since we are considering full posterior estimators. In particular, in our case, the bias of the Bayes optimal model does not necessarily reduce to 0.

Figure 4 shows the evolution of expected coverage, bias and variance with respect to the available simulation budget. By taking all plots into consideration with respect to the simulation budget, we can validate that – as suggested by theorems 1 and 2 – the increase in expected coverage is tied to an increase in variance. However, this increase comes at the price of a slight increase in bias. Consistent with our previous observations in Figure 3, we observe that the gap in both bias and variance reduces as the simulation budget increases. A bias and variance analysis for all remaining benchmarks is discussed in Appendix G.

Effects of  $\lambda$  Finally, Figure 5 shows the effect the hyper-parameter  $\lambda$  on the posterior approximations, their expected coverage and the balancing condition. BNRE is run 5 times for  $\lambda$  ranging from 1 to  $2^{15}$  and for a fixed simulation budget of 1024. Initially, the effect on the posterior approximations is limited for small values of  $\lambda$ . However, once  $\lambda$  increases, the balancing condition forces the posterior approximations to become increasingly dispersed and conservative. Eventually, at least for this specific simulation budget, the posterior approximation reduces to the prior as the balancing condition becomes dominant over the cross-entropy term. Although the global optimum remains unchanged as stated by Theorem 3, large  $\lambda$  values are likely to impair the training procedure. In particular, a large  $\lambda$  can inflate the statistical noise of the Monte Carlo estimation of the balancing condition and make the classifier  $\hat{d}$  degenerate to a classifier that is trivially balanced such as the random classifier  $\hat{d}(\vartheta, x) = 0.5$  for all  $\vartheta, x$ . In this case,  $\hat{r}(x|\vartheta) = 1$  for all  $\vartheta, x$  and the approximate posterior degenerates to the prior. This effect is directly evident from Figure 5, starting from  $\lambda \simeq 1000$ . In practice,  $\lambda$  should be sufficiently large such that the approximate classifier is balanced, while maximizing the statistical performance of the posterior estimator. We found  $\lambda = 100$  to perform well across all benchmarks, which again, is supported by Figure 5.

# 5 Related work

In the Bayesian setting, BNRE improves the reliability of NRE by constraining the classifier hypothesis space to balanced classifiers, which results in more conservative posteriors. Towards the same objective of conservative and reliable approximate posteriors, Hermans et al. [1] have shown empirically

that ensembling posterior estimators increases their expected coverage. Since the two solutions are complementary, we suggest that ensembling BNRE is a safe practice to follow. To the best of our knowledge, no other related work exists to make Bayesian simulation-based inference algorithms more conservative and reliable.

In the frequentist setting, Cranmer et al. [6] make use of neural ratio estimation to learn likelihood ratio test statistics. They show that the classifier  $\hat{d}$  does not need to be exact for the statistic to remain the most powerful, provided that the approximate likelihood ratio is monotonic with exact likelihood ratio. When this is not the case, robust inference remains possible by calibrating the classifier's output, at the price of a loss in statistical power. Similarly, for frequentist likelihood-free inference, Dalmasso et al. [21] use classifiers to estimate likelihood ratio statistics and propose a procedure for guaranteeing valid hypothesis tests and confidence sets. Finally, Dalmasso et al. [22] propose a practical procedure for the Neyman construction of confidence sets with finite-sample guarantees of nominal coverage as well as diagnostics that estimate conditional coverage over the entire parameter space.

In this work, we make the assumption that the simulator is well-specified, in the sense that it accurately models the real data generation process. However, this assumption is often violated. To overcome this issue, Generalized Bayesian inference (GBI) extends Bayesian inference by replacing the likelihood term by an arbitrary loss function. In particular, those loss functions can be designed to mitigate specific types of misspecifications and enable robust inference, even with intractable likelihoods [23-25]. Recently, Dellaporta et al. [26] further improved GBI by combining it with Bayesian non-parametric learning, making inference with misspecified simulator models both robust and computationally efficient.

# 6 Conclusions and future work

In this work, we introduced Balanced Neural Ratio Estimation (BNRE), a variation of neural ratio estimation designed to produce more conservative posterior estimators, even when the likelihood-to-evidence ratio estimator is not faithful. We provide theoretical arguments that suggest that enforcing the balancing condition should lead to more conservative posteriors without sacrificing exactness in the large simulation budget regime. Our theoretical results are experimentally validated on benchmarks of varying complexity.

Nevertheless, our inference algorithm comes with limitations that practitioners should keep in mind. First, we emphasize that theorems 1 and 2 hold only in expectation, which means that we cannot provide any guarantee at the level of single inferences. Second, the balancing condition is enforced through a regularization penalty that is not estimated exactly. This implies that the classifier  $\hat{d}$  is rarely strictly balanced, although close to be, in which case theorems 1 and 2 do not hold. In conclusion, BNRE should not be viewed as a way to obtain conservative posterior estimators with  $100\%$  reliability, but rather as a way to increase the reliability of the posterior estimators with minimal effort and no computational overhead.

Looking forward, the balancing condition could potentially be applied to other simulation-based inference algorithms. Future works could include a generalization to neural posterior estimation (NPE). In fact, the likelihood-to-evidence ratio can be extracted from an approximate posterior by removing its dependence on the prior,  $\log \hat{r} (\pmb {x}\mid \pmb {\vartheta}) = \log \hat{p} (\pmb {\vartheta}\mid \pmb {x}) - \log p(\pmb {\vartheta})$ , which in turn can be expressed as a classifier  $\hat{d} (\pmb {\vartheta},\pmb {x}) = \sigma (\log \hat{r} (\pmb {x}\mid \pmb {\vartheta}))$  on which the balancing condition can be evaluated and enforced. Although our work focuses on amortized approximate inference, the balancing condition could also be applied to sequential inference algorithms to increase their reliability.

Finally, although our initial motivation is framed within the field of simulation-based inference, our theoretical results are directly applicable to any binary classification task by replacing the joint and marginal distributions in the balancing condition with the distributions of the two considered classes. Therefore, it provides an easy-to-implement modification for high-risk classification problems.

# References

[1] Joeri Hermans, Arnaud Delaunoy, François Rozet, Antoine Wehenkel, and Gilles Louppe. Averting A Crisis In Simulation-Based Inference. arXiv e-prints, art. arXiv:2110.06581, October

2021.  
[2] Joeri Hermans, Volodimir Begy, and Gilles Louppe. Likelihood-free MCMC with amortized approximate ratio estimators. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 4239-4248. PMLR, 13-18 Jul 2020.  
[3] Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 2020.  
[4] George Papamakarios, David Sterratt, and Iain Murray. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 837-848. PMLR, 2019.  
[5] Manuel Glickler, Michael Deistler, and Jakob H Macke. Variational methods for simulation-based inference. In International Conference on Learning Representations, 2021.  
[6] Kyle Cranmer, Juan Pavez, and Gilles Loupe. Approximating likelihood ratios with calibrated discriminative classifiers. arXiv preprint arXiv:1506.02169, 2015.  
[7] Owen Thomas, Ritabrata Dutta, Jukka Corander, Samuel Kaski, Michael U Gutmann, et al. Likelihood-free inference by ratio estimation. Bayesian Analysis, 2016.  
[8] Conor Durkan, Iain Murray, and George Papamakarios. On contrastive learning for likelihood-free inference. In International Conference on Machine Learning, pages 2771-2781. PMLR, 2020.  
[9] Benjamin K Miller, Alex Cole, Patrick Forre, Gilles Louppe, and Christoph Weniger. Truncated marginal neural ratio estimation. Advances in Neural Information Processing Systems, 34: 129-143, 2021.  
[10] George Papamakarios and Iain Murray. Fast  $\varepsilon$ -free inference of simulation models with bayesian conditional density estimation. In Advances in neural information processing systems, pages 1028-1036, 2016.  
[11] David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation for likelihood-free inference. In International Conference on Machine Learning, pages 2404-2414. PMLR, 2019.  
[12] Jan-Matthis Lueckmann, Pedro J Goncalves, Giacomo Bassetto, Kaan Öcal, Marcel Nonnen-macher, and Jakob H Macke. Flexible statistical inference for mechanistic models of neural dynamics. Advances in Neural Information Processing Systems, 30, 2017.  
[13] Andrew Gelman and Donald B Rubin. Inference from iterative simulation using multiple sequences. Statistical science, 7(4):457-472, 1992.  
[14] Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating bayesian inference algorithms with simulation-based calibration. arXiv preprint arXiv:1804.06788, 2018.  
[15] David Zhao, Niccolò Dalmasso, Rafael Izbicki, and Ann B Lee. Diagnostics for conditional density models and bayesian inference algorithms. In Uncertainty in Artificial Intelligence, pages 1830–1840. PMLR, 2021.  
[16] Rob J Hyndman. Computing and graphing highest density regions. The American Statistician, 50(2):120-126, 1996.  
[17] George EP Box and George C Tiao. Bayesian inference in statistical analysis, volume 40. John Wiley & Sons, 1973.  
[18] Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density-ratio matching under the bregman divergence: a unified framework of density-ratio estimation. Annals of the Institute of Statistical Mathematics, 64(5):1009-1044, 2012.

[19] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.  
[20] Jan-Matthis Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke. Benchmarking simulation-based inference. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 343-351. PMLR, 13-15 Apr 2021.  
[21] Niccolò Dalmasso, Rafael Izbicki, and Ann Lee. Confidence sets and hypothesis testing in a likelihood-free inference setting. In International Conference on Machine Learning, pages 2323-2334. PMLR, 2020.  
[22] Niccolo Dalmasso, David Zhao, Rafael Izbicki, and Ann B Lee. Likelihood-free frequentist inference: Bridging classical statistics and machine learning in simulation and uncertainty quantification. arXiv preprint arXiv:2107.03920, 2021.  
[23] Sebastian M Schmon, Patrick W Cannon, and Jeremias Knoblauch. Generalized posteriors in approximate bayesian computation. arXiv preprint arXiv:2011.08644, 2020.  
[24] Takuo Matsubara, Jeremias Knoblauch, François-Xavier Briol, Chris Oates, et al. Robust generalised bayesian inference for intractable likelihoods. arXiv preprint arXiv:2104.07359, 2021.  
[25] Lorenzo Pacchiardi and Ritabrata Dutta. Score matched neural exponential families for likelihood-free inference. Journal of Machine Learning Research, 23(38):1-71, 2022.  
[26] Charita Dellaporta, Jeremias Knoblauch, Theodoros Damoulas, and François-Xavier Briol. Robust bayesian inference for simulator-based models via the mmd posterior bootstrap. In International Conference on Artificial Intelligence and Statistics, pages 943-970. PMLR, 2022.  
[27] Kyle Cranmer, Lukas Heinrich, Tim Head, and Gilles Louppe. "Active Scienencing" with Reusable Workflows. https://github.com/cranmer/active_sciencing, 2017.  
[28] Alfred J Lotka. Analytical note on certain rhythmic relations in organic systems. Proceedings of the National Academy of Sciences, 6(7):410-415, 1920.  
[29] Vito Volterra. Fluctuations in the abundance of a species considered mathematically. Nature, 118(2972):558-560, 1926.  
[30] LIGO Scientific Collaboration. LIGO Algorithm Library - LALSuite. free software (GPL), 2018.  
[31] C. M. Biwer, Collin D. Capano, Soumi De, Miriam Cabero, Duncan A. Brown, Alexander H. Nitz, and V. Raymond. PyCBC Inference: A Python-based parameter estimation toolkit for compact binary coalescence signals. Publ. Astron. Soc. Pac., 131(996):024503, 2019. doi: 10.1088/1538-3873/aaef0b.
