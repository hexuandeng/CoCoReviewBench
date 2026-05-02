# AUTO-ENCODINGSEQUENTIAL MONTE CARLO

Anonymous authors

Paper under double-blind review

# ABSTRACT

We build on auto-encoding sequential Monte Carlo (AESMC): a method for model and proposal learning based on maximizing the lower bound to the log marginal likelihood in a broad family of structured probabilistic models. Our approach relies on the efficiency of sequential Monte Carlo (SMC) for performing inference in structured probabilistic models and the flexibility of deep neural networks to model complex conditional probability distributions. We develop additional theoretical insights and introduce a new training procedure which improves both model and proposal learning. We demonstrate that our approach provides a fast, easy-to-implement and scalable means for simultaneous model learning and proposal adaptation in deep generative models.

# 1 INTRODUCTION

We build upon AESMC (Anon, 2017), a method for model learning that itself builds on variational auto-encoders (VAEs) (Kingma & Welling, 2014; Rezende et al., 2014) and importance weighted auto-encoders (IWAEs) (Burda et al., 2016). AESMC is similarly based on maximizing a lower bound to the log marginal likelihood, but uses SMC (Doucet & Johansen, 2009) as the underlying marginal likelihood estimator instead of importance sampling. For a very wide array of models, particularly those with sequential structure, SMC forms a substantially more powerful inference method than importance sampling, typically returning lower variance estimates for the marginal likelihood. Consequently, by using SMC for its marginal likelihood estimation, AESMC often leads to improvements in model learning compared with VAEs and IWAES. We provide experiments on structured time-series data that show AESMC based learning was able to learn useful representations of the latent space for both reconstruction and prediction more effectively than the IwAE counterpart.

AESMC was introduced in an earlier preprint (Anon, 2017) concurrently with the closely related methods of Maddison et al. (2017); Naesseth et al. (2017). In this work we take these ideas further by providing new theoretical insights for the resulting evidence lower bounds (ELBOs), extending these to explore the relative efficiency of different approaches to proposal learning, and using our results to develop a new and improved training procedure. In particular, we introduce a method for expressing the gap between an ELBO and the log marginal likelihood as a Kullback-Leibler (KL) divergence between two distributions on an extended sampling space. Doing so allows us to investigate the behavior of this family of algorithms when the objective is maximized perfectly, which occurs only if the KL divergence becomes zero. In the IWAE case, this implies that the proposal distributions are equal to the posterior distributions under the learned model. In the AESMC case, it has implications for both the proposal distributions and the intermediate set of targets that are learned. We demonstrate that, somewhat counter-intuitively, using lower variance estimates for the marginal likelihood can actually be harmful to proposal learning. In other words, we show that tighter is not necessarily better, and often substantially worse, in the context of variational bounds. Using these insights, we propose an adaptation to the AESMC algorithm, which we call alternating ELBOs, that uses different lower bounds for updating the model parameters and proposal parameters. We further show that this adaptation empirically leads to both improved model learning and proposal adaptation performance.

# 2 BACKGROUND

# 2.1 STATE-SPACE MODELS

State-space models (SSMs) are probabilistic models over a set of latent variables  $x_{1:T}$  and observed variables  $y_{1:T}$ . Given parameters  $\theta$ , a SSM is characterized by an initial density  $\mu_{\theta}(x_1)$ , a series of transition densities  $f_{t,\theta}(x_t|x_{1:t - 1})$ , and a series of emission densities  $g_{t,\theta}(y_t|x_{1:t})$  with the joint density being  $p_{\theta}(x_{1:T},y_{1:T}) = \mu_{\theta}(x_1)\prod_{t = 2}^{T}f_{t,\theta}(x_t|x_{1:t - 1})\prod_{t = 1}^{T}g_{t,\theta}(y_t|x_{1:t})$ .

We are usually interested in approximating the posterior  $p_{\theta}(x_{1:T}|y_{1:T})$  or the expectation of some test function  $\varphi$  under this posterior  $I(\varphi) \coloneqq \int \varphi(x_{1:T})p_{\theta}(x_{1:T}|y_{1:T})\mathrm{d}x_{1:T}$ . We refer to these two tasks as inference. Inference in models which are non-linear, non-discrete, and non-Gaussian is difficult and one must resort to approximate methods, for which SMC has been shown to be one of the most powerful approaches (Doucet & Johansen, 2009).

We will consider model learning as a problem of maximizing the marginal likelihood  $p_{\theta}(y_{1:T}) = \int p_{\theta}(x_{1:T}, y_{1:T}) \, \mathrm{d}x_{1:T}$  in the family of models parameterized by  $\theta$ .

# 2.2 SEQUENTIAL MONTE CARLO

SMC performs approximate inference on a sequence of target distributions  $(\pi_t(x_{1:t}))_{t=1}^T$ . In the context of ssms, the target distributions are often taken to be  $(p_\theta(x_{1:t}|y_{1:t}))_{t=1}^T$ . Given a parameter  $\phi$  and proposal distributions  $q_{1,\phi}(x_1|y_1)$  and  $(q_{t,\phi}(x_t|y_{1:t},x_{1:t-1}))_{t=2}^T$  from which we can sample and whose densities we can evaluate, smc is described in Algorithm 1.

Using the set of weighted particles  $(\tilde{x}_{1:T}^{k}, w_{T}^{k})_{k=1}^{K}$  at the last time step, we can approximate the posterior as  $\sum_{k=1}^{K} \bar{w}_{T}^{k} \delta_{\tilde{x}_{1:T}^{k}}(x_{1:T})$  and the integral  $I_{\varphi}$  as  $\sum_{k=1}^{K} \bar{w}_{T}^{k} \varphi(\tilde{x}_{1:T}^{k})$ , where  $\bar{w}_{T}^{k} := w_{T}^{k} / \sum_{j} w_{T}^{j}$  is the normalized weight and  $\delta_{z}$  is a Dirac measure centred on  $z$ . Furthermore, one can obtain an unbiased estimator of the marginal likelihood  $p_{\theta}(y_{1:T})$  using the intermediate particle weights:

$$
\hat {Z} _ {\mathrm {S M C}} := \prod_ {t = 1} ^ {T} \left[ \frac {1}{K} \sum_ {k = 1} ^ {K} w _ {t} ^ {k} \right]. \tag {1}
$$

# Algorithm 1: Sequential Monte Carlo

Data: observed values  $y_{1:T}$ , model parameters  $\theta$ , proposal parameters  $\phi$

Result: particles  $(\tilde{x}_{1:T}^{k})_{k=1}^{K}$ , weights  $(w_{T}^{k})_{k=1}^{K}$ , marginal likelihood estimate  $\hat{Z}_{\mathrm{SMC}}$  begin

Sample initial particle values  $x_{1}^{k}\sim q_{1,\phi}(\cdot |y_{1})$

Compute and normalize weights:

$$
w _ {1} ^ {k} = \frac {\mu_ {\theta} (x _ {1} ^ {k}) g _ {1 , \theta} (y _ {1} | x _ {1} ^ {k})}{q _ {1 , \phi} (x _ {1} ^ {k} | y _ {1})}, \quad \bar {w} _ {1} ^ {k} = \frac {w _ {1} ^ {k}}{\sum_ {\ell = 1} ^ {K} w _ {1} ^ {\ell}}.
$$

Initialize particle set:  $\tilde{x}_1^k\gets x_1^k$

for  $t = 2,3,\ldots ,T$  do

Sample ancestor index  $a_{t - 1}^{k}\sim \mathrm{Discrete}(\cdot |\bar{w}_{t - 1}^{1},\dots ,\bar{w}_{t - 1}^{K})$

Sample particle value  $x_{t}^{k}\sim q_{t,\phi}(\cdot |y_{1:t},\tilde{x}_{1:t - 1}^{a_{t - 1}^{k}})$

Update particle set  $\tilde{x}_{1:t}^{k}\gets (\tilde{x}_{1:t - 1}^{a_{t - 1}^{k}},x_{t}^{k})$

Compute and normalize weights:

$$
w _ {t} ^ {k} = \frac {f _ {t , \theta} (x _ {t} ^ {k} | \tilde {x} _ {1 : t - 1} ^ {a _ {t - 1} ^ {k}}) g _ {t , \theta} (y _ {t} | \tilde {x} _ {1 : t} ^ {k})}{q _ {t , \phi} (x _ {t} ^ {k} | y _ {1 : t} , \tilde {x} _ {1 : t - 1} ^ {a _ {t - 1} ^ {k}})}, \qquad \qquad \bar {w} _ {t} ^ {k} = \frac {w _ {t} ^ {k}}{\sum_ {\ell = 1} ^ {K} w _ {t} ^ {\ell}}.
$$

Compute marginal likelihood:  $\hat{Z}_{\mathrm{SMC}} = \prod_{t=1}^{T} \frac{1}{K} \sum_{k=1}^{K} w_t^k$ .

return particles  $(\tilde{x}_{1:T}^{k})_{k=1}^{K}$ , weights  $(w_{T}^{k})_{k=1}^{K}$ , marginal likelihood estimate  $\hat{Z}_{SMC}$

The sequential nature of SMC and the resampling step are crucial in making SMC scalable to large  $T$ . The former makes it easier to design efficient proposal distributions as each step need only target the next set of variables  $x_{t}$ . The resampling step allows the algorithm to focus on promising particles in light of new observations, avoiding the exponential divergence between the weights of different samples that occurs for importance sampling as  $T$  increases. This can be demonstrated both empirically and theoretically (Del Moral, 2004, Chapter 9). We refer the reader to (Doucet & Johansen, 2009) for an in-depth treatment of SMC.

# 2.3 IMPORTANCE WEIGHTED AUTO-ENCODERS

Given a dataset of observations  $(y^{(n)})_{n = 1}^{N}$ , a generative model  $p_{\theta}(x,y)$  and an inference network  $q_{\phi}(x|y)$ , IWAES (Burda et al., 2016) maximize  $\frac{1}{N}\sum_{n = 1}^{N}\mathrm{ELBO}_{\mathrm{IS}}(\theta ,\phi ,y^{(n)})$  where, for a given observation  $y$ , the ELBOIS (with  $K$  particles) is a lower bound on  $\log p_{\theta}(y)$  by Jensen's inequality:

$$
\operatorname {E L B O} _ {\mathrm {I S}} (\theta , \phi , y) = \int Q _ {\mathrm {I S}} \left(x ^ {1: K}\right) \log \hat {Z} _ {\mathrm {I S}} \left(x ^ {1: K}\right) \mathrm {d} x ^ {1: K} \leq \log p _ {\theta} (y), \text {w h e r e} \tag {2}
$$

$$
Q _ {\mathrm {I S}} \left(x ^ {1: K}\right) = \prod_ {k = 1} ^ {K} q _ {\phi} \left(x ^ {k} | y\right), \quad \hat {Z} _ {\mathrm {I S}} \left(x ^ {1: K}\right) = \sum_ {k = 1} ^ {K} \frac {p _ {\theta} \left(x ^ {k} , y\right)}{q _ {\phi} \left(x ^ {k} | y\right)}. \tag {3}
$$

Note that for  $K = 1$  particle, this objective reduces to a vAE (Kingma & Welling, 2014; Rezende et al., 2014) objective to which we will refer to as

$$
\operatorname {E L B O} _ {\mathrm {V A E}} (\theta , \phi , y) = \int q _ {\phi} (x | y) \left(\log p _ {\theta} (x, y) - \log q _ {\phi} (x | y)\right) \mathrm {d} x. \tag {4}
$$

The IWAE optimization is performed using stochastic gradient ascent (SGA) where a sample from  $\left(\prod_{k=1}^{K} q_{\phi}(x^k | y^{(n)})\right)$  is obtained using the reparameterization trick (Kingma & Welling, 2014) and the gradient  $\frac{1}{N} \sum_{n=1}^{N} \nabla_{\theta, \phi} \log \left(\sum_{k=1}^{K} \frac{p_{\theta}(x^k, y^{(n)})}{q_{\phi}(x^k | y^{(n)})}\right)$  is used to perform an optimization step.

# 3 AUTO-ENCODINGSEQUENTIAL MONTECARLO

AESMC implements model learning, proposal adaptation, and inference amortization in a similar manner to the VAE and the IWAE: it uses SGA on an empirical average of the ELBO over observations. However, it varies in the form of this ELBO. In this section, we will introduce the AESMC ELBO, explain how gradients of it can be estimated, and discuss the implications of these changes.

# 3.1 OBJECTIVE FUNCTION

Consider a family of sSMs  $\{p_{\theta}(x_{1:T},y_{1:T}):\theta \in \Theta \}$  and a family of proposal distributions  $\{q_{\phi}(x_{1:T}|y_{1:T}) = q_{1,\phi}(x_1|y_1)\prod_{t = 2}^T q_{t,\phi}(x_t|x_{1:t - 1},y_{1:t}):\phi \in \Phi \}$ . AESMC uses an ELBO objective based on the SMC marginal likelihood estimator (1). In particular, for a given  $y_{1:T}$ , the objective is defined as

$$
\operatorname {E L B O} _ {\mathrm {S M C}} (\theta , \phi , y _ {1: T}) := \int Q _ {\mathrm {S M C}} (x _ {1: T} ^ {1: K}, a _ {1: T - 1} ^ {1: K}) \log \hat {Z} _ {\mathrm {S M C}} (x _ {1: T} ^ {1: K}, a _ {1: T - 1} ^ {1: K}) \mathrm {d} x _ {1: T} ^ {1: K} \mathrm {d} a _ {1: T - 1} ^ {1: K}, \quad (5)
$$

where  $\hat{Z}_{\mathrm{SMC}}(x_{1:T}^{1:K},a_{1:T-1}^{1:K})$  is defined in (1) and  $Q_{\mathrm{SMC}}$  is the sampling distribution of sMC,

$$
Q _ {\mathrm {S M C}} \left(x _ {1: T} ^ {1: K}, a _ {1: T - 1} ^ {1: K}\right) = \left(\prod_ {k = 1} ^ {K} q _ {1, \phi} \left(x _ {1} ^ {k}\right)\right) \left(\prod_ {t = 2} ^ {T} \prod_ {k = 1} ^ {K} q _ {t, \phi} \left(x _ {t} ^ {k} \mid \tilde {x} _ {1: t - 1} ^ {a _ {t - 1} ^ {k}}\right) \cdot \operatorname {D i s c r e t e} \left(a _ {t - 1} ^ {k} \mid w _ {t - 1} ^ {1: K}\right)\right). \tag {6}
$$

ELBO $_{\text{SMC}}$  forms a lower bound to the log marginal likelihood  $\log p_{\theta}(y_{1:T})$  due to Jensen's inequality. Hence, given a dataset  $(y_{1:T}^{(n)})_{n=1}^{N}$ , we can perform model parameter learning based on maximizing the lower bound of  $\frac{1}{N} \sum_{n=1}^{N} \log p_{\theta}(y_{1:T}^{(n)})$ :

$$
\mathcal {J} (\theta , \phi) := \frac {1}{N} \sum_ {n = 1} ^ {N} \operatorname {E L B O S M C} (\theta , \phi , y _ {1: T} ^ {(n)}). \tag {7}
$$

For notational convenience, we will talk about optimizing ELBOs in the rest of this section. However, we note that the main intended use of AESMC is to amortize over datasets, for which we replace the ELBO gradient with an empirical sum as per (7).

# 3.2 GRADIENT ESTIMATION

We describe a gradient estimator used for optimizing ELBOsmC  $(\theta ,\phi ,y_{1:T})$  using SGA. The smc sampler in Algorithm 1 proceeds by sampling  $x_{1}^{1:K},a_{1}^{1:K},x_{2}^{1:K},\ldots$  sequentially from their respective distributions  $\prod_{k = 1}^{K}q_{1}(x_{1}^{k}),\prod_{k = 1}^{K}\mathrm{Discrete}(a_{1}^{k}|w_{1}^{1:K}),\prod_{k = 1}^{K}q_{2}(x_{2}^{k}|x_{1}^{a_{1}^{k}}),\ldots$  until the whole particle-weight trajectory  $(x_{1:T}^{1:T},a_{1:T - 1}^{1:K})$  is sampled using which the marginal likelihood estimator in (1) is formed.

Assuming that the sampling of latent variables  $x_{1:T}^{1:K}$  is reparameterizable, we can make their sampling independent of  $(\theta, \phi)$ . In particular, assume that there exists a set of auxiliary random variables  $\epsilon_{1:T}^{1:K}$  where  $\epsilon_t^k \sim s_t$  and a set of reparameterization functions  $r_t$  using which we can simulate the SMC sampler as follows: sample  $\epsilon_1^{1:K} \sim \prod_{k=1}^K s_1$  and set  $x_1^k = r_1(\epsilon_1^k)$ , then sample  $a_1^{1:K}$  from  $\prod_{k=1}^K \mathrm{Discrete}(a_1^k | w_1^{1:K})$ , then sample  $\epsilon_2^{1:K} \sim \prod_{k=1}^K s_2$  and set  $x_2^k = r_2(\epsilon_2^k, x_1^{a_1^k})$ , until we obtain  $(x_{1:T}^{1:T}, a_{1:T-1}^{1:K})$ . We use this reparameterized sample of  $(x_{1:T}^{1:T}, a_{1:T-1}^{1:K})$  to evaluate the gradient estimator  $\nabla_{\theta, \phi} \log \hat{Z}_{\mathrm{SMC}}(x_{1:T}^{1:K}, a_{1:T-1}^{1:K})$ .

To account for the discrete choices of ancestor indices  $a_{t}^{k}$  we can additionally use the REINFORCE (Williams, 1992) trick, however in practice, we find that the additional term in the estimator has problematically high variance. We explore various other possible gradient estimators and empirical assessments of their variances in Appendix A.

# 3.3 BIAS & IMPLICATIONS ON THE PROPOSALS

In this section, we express the gap between ELBOs and the log marginal likelihood as a KL divergence and study implications on the proposal distributions. We present a set of claims and propositions whose full proofs are in Appendix B. These give insight into the behavior of AESMC, compared with alternatives and show the advantages, and disadvantages, of using our different ELBO. This insight motivates Section 4 which proposes an algorithm for improving proposal learning.

Definition 1. Given an unnormalized target density  $\tilde{P}:\mathcal{X}\to [0,\infty)$  with normalizing constant  $Z_{P} > 0$ ,  $P\coloneqq \tilde{P} /Z_P$ , and a proposal density  $Q:\mathcal{X}\rightarrow [0,\infty)$ , the ELBO

$$
\mathrm {E L B O} = \int Q (x) \log \frac {\tilde {P} (x)}{Q (x)} \mathrm {d} x, \tag {8}
$$

is a lower bound on  $\log Z_P$  and satisfies

$$
\operatorname {E L B O} = \log Z _ {P} - \operatorname {K L} (Q \| P). \tag {9}
$$

This is a standard identity used in variational inference (Wainwright et al., 2008) and VAEs. In the case of VAEs, applying Definition 1 with  $P$  being  $p_{\theta}(x|y)$ ,  $\tilde{P}$  being  $p_{\theta}(x,y)$ ,  $Z_P$  being  $p_{\theta}(y)$ , and  $Q$  being  $q_{\phi}(x|y)$ , we directly can rewrite (4) as  $\mathrm{ELBOVAE}(\theta ,\phi ,y) = \log p_{\theta}(y) - \mathrm{KL}\left(q_{\phi}(x|y)||p_{\theta}(x|y)\right)$ .

The key observation for expressing such a bound for general ELBOs such as ELBO $_{\mathrm{IS}}$  and ELBO $_{\mathrm{SMC}}$  is that the target density  $P$  and the proposal density  $Q$  need not directly correspond to  $p_{\theta}(x|y)$  and  $q_{\phi}(x|y)$ . This allows us to view the underlying sampling distributions of the marginal likelihood Monte Carlo estimators such as  $Q_{\mathrm{IS}}$  in (3) and  $Q_{\mathrm{SMC}}$  in (6) as proposal distributions on an extended space  $\mathcal{X}$ . The following claim uses this observation to express the bound between a general ELBO and the log marginal likelihood as KL divergence from the extended space sampling distribution to a corresponding target distribution.

Claim 1. Given a non-negative unbiased estimator  $\hat{Z}_P(x) \geq 0$  of the normalizing constant  $Z_P$  where  $x$  is distributed from the proposal distribution  $Q(x)$ , the following holds:

$$
\operatorname {E L B O} = \int Q (x) \log \hat {Z} _ {P} (x) \mathrm {d} x = \log Z _ {P} - \operatorname {K L} (Q | | P), \tag {10}
$$

$$
w h e r e \quad P (x) = \frac {Q (x) \hat {Z} _ {P} (x)}{Z _ {P}} \tag {11}
$$

is a normalized target density.

In the case of IWAEs, we can apply Claim 1 with  $Q$  and  $\hat{Z}_P$  being  $Q_{\mathrm{IS}}$  and  $\hat{Z}_{\mathrm{IS}}$  defined in (3) and  $Z_P$  being  $p_\theta(y)$ . This yields

$$
\operatorname {E L B O} _ {\mathrm {I S}} (\theta , \phi , y) = \log p _ {\theta} (y) - \mathrm {K L} \left(Q _ {\mathrm {I S}} | | P _ {\mathrm {I S}}\right), \text {w h e r e} \tag {12}
$$

$$
P _ {\mathrm {I S}} \left(x ^ {1: K}\right) = \frac {1}{K} \sum_ {k = 1} ^ {K} \left(q \left(x ^ {1} | y\right) \dots q \left(x ^ {k - 1} | y\right) p \left(x ^ {k} | y\right) q \left(x ^ {k + 1} | y\right) \dots q \left(x ^ {K} | y\right)\right). \tag {13}
$$

Similarly, in the case of AESMC, we obtain

$$
\operatorname {E L B O} _ {\mathrm {S M C}} (\theta , \phi , y _ {1: T}) = \log p _ {\theta} (y _ {1: T}) - \mathrm {K L} \left(Q _ {\mathrm {S M C}} | | P _ {\mathrm {S M C}}\right), \text {w h e r e} \tag {14}
$$

$$
P _ {\mathrm {S M C}} (x _ {1: T} ^ {1: K}, a _ {1: T - 1} ^ {1: K}) = Q _ {\mathrm {S M C}} (x _ {1: T} ^ {1: K}, a _ {1: T - 1} ^ {1: K}) \hat {Z} _ {\mathrm {S M C}} (x _ {1: T} ^ {1: K}, a _ {1: T - 1} ^ {1: K}) / p _ {\theta} (y _ {1: T}). \tag {15}
$$

Having expressions for the target distribution  $P$  and the sampling distribution  $Q$  for a given ELBO allows us to investigate what happens when we maximize that ELBO, remembering that the KL term is strictly non-negative and zero if and only if  $P = Q$ . For the VAE and IwAE cases then, provided the proposal is sufficiently flexible, one can always perfectly maximize the ELBO by setting  $p_{\theta}(x|y) = q_{\phi}(x|y)$  for all  $x$ . The reverse implication also holds: if  $\mathrm{ELBO}_{\mathrm{VAE}} = \log Z_P$  then it must be the case that  $p_{\theta}(x|y) = q_{\phi}(x|y)$ . However, for AESMC, achieving  $\mathrm{ELBO} = \log Z_P$  is only possible when one also has sufficient flexibility to learn a particular series intermediate target distributions, namely the marginals of the final target distribution. In other words, it is necessary to learn a particular factorization of the generative model, not just the correct individual proposals, to achieve  $P = Q$ . These observations are formalized in Propositions 1 and 2 below.

Proposition 1.  $Q_{IS}(x^{1:K}) = P_{IS}(x^{1:K})$  for all  $x^{1:K}$  if and only if  $q(x|y) = p(x|y)$  for all  $x$ . Proposition 2. If  $K > 1$ , then  $P_{SMC}(x_{1:T}^{1:K}, a_{1:T-1}^{1:K}) = Q_{SMC}(x_{1:T}^{1:K}, a_{1:T-1}^{1:K})$  for all  $(x_{1:T}^{1:K}, a_{1:T-1}^{1:K})$  if and only if

1.  $\pi_t(x_{1:t}) = \int p(x_{1:T}|y_{1:T})\mathrm{d}x_{t + 1:T} = p(x_{1:t}|y_{1:T})$  for all  $x_{1:t}$  and  $t = 1,\ldots ,T$  , and  
2.  $q_{1}(x_{1}|y_{1}) = p(x_{1}|y_{1:T})$  for all  $x_{1}$  and  $q_{t}(x_{t}|x_{1:t - 1},y_{1:t}) = p(x_{1:t}|y_{1:T}) / p(x_{1:t - 1}|y_{1:T})$  for  $t = 2,\ldots ,T$  for all  $x_{1:t}$

where  $\pi_t(x_{1:t})$  are the intermediate targets used by SMC.

Proposition 2 has the consequence that if the family of generative models is such that condition 1 does not hold, we will not be able to make the bound tight  $\mathrm{ELBO}_{\mathrm{SMC}} = Z_P$ . This means that, except for a very small class of models, then, for most convenient parameterizations, it will be impossible to learn a perfect proposal that gives a tight bound, i.e. there will be no  $\theta$  and  $\phi$  such that the above conditions can be satisfied. However, it also means that  $\mathrm{ELBO}_{\mathrm{SMC}}$  encodes important additional information about the implications the factorization of the generative model has on the inference—the model depends only on the final target  $\pi_T(x_{1:T}) = p_\theta(x_{1:T}|y_{1:T})$ , but some choices of the intermediate targets  $\pi_t(x_{1:t})$  will lead to much more efficient inference than others. Perhaps more importantly, SMC is usually a far more powerful inference algorithm than importance sampling and so the AESMC setup allows for more ambitious model learning problems to be effectively tackled than the VAE or IWAE. After all, even though it is well known in the SMC literature that, unlike for importance sampling, most problems have no perfect set of SMC proposals which will generate exact samples from the posterior (Doucet & Johansen, 2009), SMC is still gives superior performance on most problems with more than a few dimensions. These intuitions are backed up by our experiments that show that using  $\mathrm{ELBO}_{\mathrm{SMC}}$  regularly learns better models than using  $\mathrm{ELBO}_{\mathrm{IS}}$ .

# 4 IMPROVING PROPOSAL LEARNING

Given the implications from the previous section, we now ask whether optimizing ELBOIS and ELBOSMC actually improves the proposal distribution? In other words, does the optimization procedure make  $q_{\phi}(x_{1:T}|y_{1:T})$  closer to  $p_{\theta}(x_{1:T}|y_{1:T})$  and how does the number of particles  $K$  affect this? In the VAE case, we are directly optimizing KL ( $q_{\phi}(x|y)||p_{\theta}(x|y)$ ) for fixed model parameters and so it is straightforward to see that we will induce proposal learning. In the IWAE and AESMC cases, such optimization only minimizes KL ( $Q_{\mathrm{IS}}||P_{\mathrm{IS}}$ ) and KL ( $Q_{\mathrm{SMC}}||P_{\mathrm{SMC}}$ ) respectively, which does not directly imply that KL ( $q_{\phi}(x|y)||p_{\theta}(x|y)$ ) is small.

Counter-intuitively, it transpires that the tighter bounds implied by using a larger  $K$  is often harmful to proposal learning for both IWAE and AESMC. At a high-level, this is because an accurate estimate for  $\hat{Z}_P$  can be achieved for a wide range of proposal parameters  $\phi$  and so the magnitude of  $\nabla_{\phi}$  ELBO reduces as  $K$  increases. Typically, this shrinkage happens faster than increasing  $K$  reduces the standard deviation of the estimate and so the signal-to-noise ratio (SNR) for the gradient estimate actually decreases, even though it is a lower variance estimate. This effect is demonstrated in Figure 1 which shows a kernel density estimator for the distribution of the ELBO gradient estimate for different  $K$  and the model given in Section 5.2. Here we see that as we increase  $K$ , both the expected gradient estimate (i.e. true gradient) and standard deviation of the estimate decrease. However, the former decreases faster and so the SNR

decreases. This is perhaps easiest to appreciate by noting that for  $K \geq 10$ , there is a roughly equal probability of the estimate being positive or negative, such that we are equally likely to increase or decrease the parameter value at the next SGA iteration, inevitably leading to poor performance. On the other hand, when  $K = 1$ , it is far more likely that the gradient estimate is positive than negative, and so there is clear drift to the gradient steps. We add to the empirical evidence for this behavior is Section 5. Note the critical difference for model learning is that  $\nabla_{\theta}$  ELBO does not, in general, decrease in magnitude as  $K$  increases. Note also that using a larger  $K$  should always give better performance at test time—the implication of our result is that it may be better to learn  $\phi$  using a smaller  $K$ .

![](images/f70313e71629c3cfa659ad05966cee4681cf0fb9225f92336c6addb37308447c.jpg)  
Figure 1: Density estimate of  $\nabla_{\phi}$  ELBO for different  $K$

We can further demonstrate this result using an informal theoretical argument for the case of the IWAE. Our gradient estimate for the  $K$  particle IWAE is

$$
I _ {K} = \nabla_ {\phi} \log \left(\frac {1}{K} \sum_ {k = 1} ^ {K} \frac {p _ {\theta} \left(x ^ {k} , y\right)}{q _ {\phi} \left(x ^ {k} \mid y\right)}\right), \tag {16}
$$

where  $I = \lim_{K\to \infty}I_K = 0$  because with infinite samples, the estimate is exact and thus independent of the proposal parameters. Now adapting the IWAE result of Rainforth et al. (2017) shows that

$$
\mathbb {E} \left[ I _ {K} ^ {2} \right] = \mathbb {E} \left[ (I _ {K} - I) ^ {2} \right] \leq \frac {C _ {0} ^ {2} \varsigma_ {1} ^ {4}}{4 K ^ {2}} + \frac {C _ {0} ^ {2} \varsigma_ {1} ^ {4}}{4 K ^ {2}} + \frac {\kappa_ {0} ^ {2} \varsigma_ {1} ^ {2}}{K} + \frac {C _ {0} \kappa_ {0} \varsigma_ {1} ^ {3}}{K ^ {3 / 2}} + O \left(\frac {1}{K ^ {3}}\right), \tag {17}
$$

where  $C_0$ ,  $\kappa_0$  ( $K_0$  in Rainforth et al. (2017)), and  $\varsigma_1$  are constants and we have set  $N = 1$  in their formulation. Here the first term,  $\frac{C_0^2 \varsigma_1^4}{4K^2}$ , is a "bias term", in our context  $(\mathbb{E}[I_K])^2 = (\nabla_\phi \mathrm{ELBO})^2$ , and the rest are variance terms. We can now define our SNR as follows

$$
\mathrm {S N R} = \frac {\nabla_ {\phi} \operatorname {E L B O}}{\sqrt {\operatorname {V a r} [ I _ {K} ]}} \approx \sqrt {\frac {\frac {C _ {0} ^ {2} \varsigma_ {1} ^ {4}}{4 K ^ {2}}}{\frac {C _ {0} ^ {2} \varsigma_ {1} ^ {4}}{4 K ^ {2}} + \frac {\kappa_ {0} ^ {2} \varsigma_ {1} ^ {2}}{K} + \frac {C _ {0} \kappa_ {0} \varsigma_ {1} ^ {3}}{K ^ {3 / 2}} + O \left(\frac {1}{K ^ {3}}\right)}} \approx \sqrt {\frac {C _ {0} ^ {2} \varsigma_ {1} ^ {2}}{4 \kappa_ {0} ^ {2} K}} = O \left(\sqrt {\frac {1}{K}}\right), \tag {18}
$$

where we have substituted in the bounds for the bias and variance from (17) and the approximations will, in general, become increasingly exact as  $K$  increases. We thus see that increasing  $K$  reduces the SNR and so a lower  $K$  is preferable for proposal learning.

# 4.1 ALTERNATING ELBOS

To address these issues, we propose the alternating ELBOs (ALT) algorithm which updates  $(\theta ,\phi)$  in a coordinate descent fashion using different ELBOs, and thus gradient esti

mates, for each. We pick a  $\theta$ -optimizing pair and a  $\phi$ -optimizing pair  $(A_{\theta},K_{\theta}),(A_{\phi},K_{\phi})\in$  {importantsampling(Is),SMC}  $\times$  {1,2,...}, corresponding to a inference type number of particles. In an optimization step, we obtain an estimator for  $\nabla_{\theta}$  ELBO  $A_{\theta}$  with  $K_{\theta}$  particles and an estimator for  $\nabla_{\phi}$  ELBO  $A_{\phi}$  with  $K_{\phi}$  particles which we call  $g_{\theta}$  and  $g_{\phi}$  respectively. We use  $g_{\theta}$  to update the current  $\theta$  and  $g_{\phi}$  to update the current  $\phi$ . The results from the previous sections suggest that using  $A_{\theta} = \mathrm{SMC}$  and  $A_{\phi} = \mathrm{IS}$  with a large  $K_{\theta}$  and a small  $K_{\phi}$  should perform better model and proposal learning than just fixing  $(A_{\theta},K_{\theta}) = (A_{\phi},K_{\phi})$  to (SMC, large) since using  $A_{\phi} = \mathrm{IS}$  with small  $K_{\phi}$  helps learning  $\phi$  and using  $A_{\theta} = \mathrm{SMC}$  with large  $K_{\theta}$  helps learning  $\theta$ . We experimentally find that this procedure in fact improves both model and proposal learning, leading to a new algorithm that improves the AESMC approach even further.

# 5 EXPERIMENTS

We now present a series of experiments designed to answer the following questions: 1) Does reducing the gap by using more particles or a better inference procedure lead to an adverse effect on proposal learning? 2) Can AESMC, despite this effect, outperform IWAE? 3) Can we further improve the learned model and proposal by using our newly proposed algorithm ALT?

First we investigate a linear Gaussian state space model (LGSSM) for model learning and a latent variable model for proposal adaptation. This allows us to compare the learned parameters to the optimal ones. Doing so, we confirm our conclusions for this simple problem.

We then extend those results to more complex, high dimensional observation spaces that require models and proposals parameterized by neural networks. We do so by investigating the Moving Agents dataset, a set of partially occluded video sequences.

# 5.1 LINEAR GAUSSIAN STATE SPACE MODEL

Given the following LGSSM

$$
p \left(x _ {1}\right) = \text {N o r m a l} \left(x _ {1}; 0, 1 ^ {2}\right), \tag {19}
$$

$$
p \left(x _ {t} \mid x _ {t - 1}\right) = \text {N o r m a l} \left(x _ {t}; \theta_ {1} x _ {t - 1}, 1 ^ {2}\right), \quad t = 2, \dots T, \tag {20}
$$

$$
p \left(y _ {t} \mid x _ {t}\right) = \text {N o r m a l} \left(y _ {t}; \theta_ {2} x _ {t}, \sqrt {0 . 1} ^ {2}\right), \quad t = 1, \dots , T, \tag {21}
$$

we find that optimizing  $\mathrm{ELBO}_{\mathrm{SMC}}(\theta, \phi, y_{1:T})$  w.r.t  $\theta$  leads to better generative models than optimizing  $\mathrm{ELBO}_{\mathrm{IS}}(\theta, \phi, y_{1:T})$ . The same is true for using more particles.

To do so, we generate on sequence  $y_{1:T}$  for  $T = 200$  by sampling from the model with  $\theta = (\theta_1, \theta_2) = (0.9, 1.0)$ . We then optimize the different ELBOs w.r.t  $\theta$  using the bootstrap proposal  $q_1(x_1 | y_1) = \mu_\theta(x_1)$  and  $q_t(x_t | x_{1:t-1}, y_{1:t}) = f_{t,\theta}(x_t | x_{1:t-1})$ . Any  $\theta$  appearing in  $q$  terms is detached from the computational graph to not influence the gradient updates through  $q$ .

We use a fixed learning rate of 0.01 and optimize for 500 steps. Figure 2 shows that the convergence of both  $\log p_{\theta}(y_{1:T})$  to  $\max_{\theta} \log p_{\theta}(y_{1:T})$  and  $\theta$  to  $\arg \max_{\theta} \log p_{\theta}(y_{1:T})$  is faster when ELBOSMC and more particles are used.

# 5.2 PROPOSAL LEARNING

We now investigate how learning  $\phi$ , i.e. the proposal, is affected by the choice of ELBO and the number of particles.

Consider a simple, fixed generative model  $p(\mu)p(x|\mu) = \mathrm{Normal}(\mu;0,1^2)\mathrm{Normal}(x;\mu,1^2)$  where  $\mu$  and  $x$  are the latent and observed variables respectively and a family of proposal distributions  $q_{\phi}(\mu) = \mathrm{Normal}(\mu;\mu_q,\sigma_q^2)$  parameterized by  $\phi = (\mu_q,\log \sigma_q^2)$ . For a fixed observation  $x = 2.3$ , we initialize  $\phi = (0.01,0.01)$  and optimize ELBOIS with respect to  $\phi$ . We investigate the quality of the learned parameter  $\phi$  as we increase the number of particles  $K$  during training. Note that for  $K = 1$ , this reduces to stochastic variational inference (Hoffman et al., 2013). Figure 3 (left) clearly demonstrates that the quality of  $\phi$  compared to the analytic posterior decreases as we increase  $K$ .

Similar behavior is observed in Figure 3 (middle, right) where we optimize ELBO $_{\text{SMC}}$  with respect to both  $\theta$  and  $\phi$  for the LGSSM described in Section 5.1. We see that using more

![](images/d983eef05a01c677dcd162a76d14923f4aa2690260bccc77de11eb1a31acd9cf.jpg)  
Figure 2: (Left) Log marginal likelihood analytically evaluated at every  $\theta$  during optimization; the black line indicates  $\max_{\theta} \log p_{\theta}(y_{1:T})$  obtained by the expectation maximization (EM) algorithm. (Right) learning of model parameters; the black line indicates  $\arg \max_{\theta} \log p_{\theta}(y_{1:T})$  obtained by the EM algorithm.

![](images/3328b620a10ae02db261e28c68fec199096767e963ae58fddfa7a5addcb8e31f.jpg)

particles helps model learning but makes proposal learning worse. Using our ALT algorithm alleviates this problem and at the same time makes model learning faster as it profits from a more accurate proposal distribution.

![](images/98b9becd757b95b3330fb10ef4222404ea3762bee8e7c1e71a5527730ff53327.jpg)  
Figure 3: (Left) Optimizing ELBOIS for the Gaussian unknown mean model with respect to  $\phi$  results in worse  $\phi$  as we increase number of particles  $K$ . (Middle, right) Optimizing ELBOSMC with respect to  $(\theta, \phi)$  for LGSSM and using the ALT algorithm for updating  $(\theta, \phi)$  with  $(A_{\theta}, K_{\theta}) = (\mathrm{SMC}, 1000)$  and  $(A_{\phi}, K_{\phi}) = (\mathrm{Is}, 10)$ . Right measures the quality of  $\phi$  by showing  $\sqrt{\sum_{t=1}^{T} (\mu_t^{\mathrm{kalman}} - \mu_t^{\mathrm{approx}})^2}$  where  $\mu_t^{\mathrm{kalman}}$  is the marginal mean obtained from the Kalman smoothing algorithm under the model with EM-optimized parameters and  $\mu_t^{\mathrm{approx}}$  is an marginal mean obtained from the set of 10 SMC particles with learned/bootstrap proposal.

![](images/8cf3cd0ae7f9cfa4d1a77a7862d3a4a75faf31a914ba07a4d9dbfc19e417b184.jpg)

![](images/75f8985c88958b2e1eed663bd327d7ea1e8d5cc6ff5ceb19aaf48d7df1a9ed59.jpg)

# 5.3 MOVING AGENTS

To show that our results are applicable to complex, high dimensional data we compare ALT, AESMC and IWAE on stochastic, partially observable video sequences. Figure 7 in the appendix shows an example of such a sequence.

The dataset consists of 5000 sequences of images  $(y_{0:39}^{(i)})_1^{5000}$  of which 1000 are randomly held out as test set. Each sequence contains 40 images represented as a 3 dimensional array of size  $1\times 32\times 32$ . In each sequence there is one agent, represented as circle, whose starting position is sampled randomly along the top and bottom of the image. The dataset is inspired by (Ondruska & Posner, 2016), however with the crucial difference that the movement of the agent is stochastic. The agent performs a directed random walk through the image. At

each timestep, it moves according to

$$
y _ {t + 1} \sim \mathcal {N} \left(y _ {t + 1} \mid y _ {t} + 0. 1 5, 0. 0 2 ^ {2}\right) \tag {22}
$$

$$
x _ {t + 1} \sim \mathcal {N} (x _ {t + 1} | 0, 0. 0 2 ^ {2})
$$

where  $(x_{t},y_{t})$  are the coordinates in frame  $t$  in a unit square that is then projected onto  $32\times 32$  pixels. In addition to the stochasticity of the movement, half of the image is occluded, preventing the agent to be observed.

As generative model and proposal distribution we use a Variational Recurrent Neural Network (VRNN) (Chung et al., 2015). It extends recurrent neural networks (RNNs) by introducing a stochastic latent state  $x_{t}$  at each timestep  $t$ . Together with the observation  $y_{t}$ , this state conditions the deterministic transition of the RNN. By introducing this unobserved stochastic state, the VRNN is able to better model complex long range variability in stochastic sequences. Architecture and hyperparameter details are given in the appendix.

One can see in Figure 4 that models trained using the alternating training schedule ALT outperform AESMC which in turn outperform IWAE. Using more particles improves the ELBO, but more so for AESMC and ALT as the differences in estimators and training methods become more pronounced with higher particle numbers. In the appendix we inspect different learned generative models by using them for prediction, confirming the results presented here.

![](images/3a54c64a56b4cf1ecd1c882077e1e06d293de2609f82a7cbcbbcde30eb2dbb79.jpg)  
Figure 4: (Left) Rolling mean over 5 epochs of  $\max(\text{ELBO}_{\text{SMC}}, \text{ELBO}_{\text{IS}})$  on the test set. The color indicates the number of particles, the line style the used algorithm. (Right) The table shows the final  $\max(\text{ELBO}_{\text{SMC}}, \text{ELBO}_{\text{IS}})$  for each learned model.

<table><tr><td>Particles</td><td>Method</td><td>Moving Agents</td></tr><tr><td rowspan="3">5</td><td>IWAE</td><td>-365.1</td></tr><tr><td>AESMC</td><td>-364.0</td></tr><tr><td>ALT</td><td>-363.9</td></tr><tr><td rowspan="3">10</td><td>IWAE</td><td>-364.6</td></tr><tr><td>AESMC</td><td>-363.3</td></tr><tr><td>ALT</td><td>-363.1</td></tr><tr><td rowspan="3">20</td><td>IWAE</td><td>-364.4</td></tr><tr><td>AESMC</td><td>-363.06</td></tr><tr><td>ALT</td><td>-362.6</td></tr></table>

# 6 CONCLUSIONS

We have developed AESMC—a method for performing model learning using a new ELBO objective which is based on the SMC marginal likelihood estimator. This ELBO objective is optimized using SGA and the reparameterization trick. Our approach utilizes the efficiency of SMC in models with intermediate observations and hence is suitable for a highly structured models. We experimentally demonstrated that this objective has a tighter gap to the log marginal likelihood than the IWAE objective and that it works well on structured problems such as learning the generative model of moving balls which can be used for tasks such as reconstruction or prediction.

Additionally, in Claim 1, we provide a simple way to express the bias of objectives induced by log of marginal likelihood estimators as a KL divergence. In Propositions 1 and 2, we investigate the implications of these KLs being zero in the case of IWAE and AESMC. In the latter case, we find that we can achieve zero KL only if we are able to learn SMC intermediate target distributions corresponding to marginals of the target distribution. We then built on these results to show that tighter in not necessarily better when it comes to variational bounds and used this insight to develop a new method, alternating ELBOs, that address some of the issues of AESMC to improve proposal learning.

# REFERENCES

Anon. Auto-encoding sequential Monte Carlo. arXiv preprint arXiv:1705.****v1, 2017.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. In ICLR, 2016.  
Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. In Advances in neural information processing systems, pp. 2980-2988, 2015.  
P Del Moral. Feynman-Kac formulae: genealogical and interacting particle systems with applications. Probability and its applications, 2004.  
Arnaud Doucet and Adam M Johansen. A tutorial on particle filtering and smoothing: Fifteen years later. Handbook of nonlinear filtering, 12(656-704):3, 2009.  
Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic variational inference. The Journal of Machine Learning Research, 14(1):1303-1347, 2013.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In ICLR, 2014.  
Chris J Maddison, Dieterich Lawson, George Tucker, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. Filtering variational objectives. arXiv preprint arXiv:1705.09279, 2017.  
Christian A Naesseth, Scott W Linderman, Rajesh Ranganath, and David M Blei. Variational sequential Monte Carlo. arXiv preprint arXiv:1705.11140, 2017.  
Peter Ondruska and Ingmar Posner. Deep tracking: Seeing beyond seeing using recurrent neural networks. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.  
Tom Rainforth, Robert Cornish, Hongseok Yang, Andrew Warrington, and Frank Wood. On the opportunities and pitfalls of nesting monte carlo estimators. arXiv preprint arXiv:1709.06181, 2017.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In ICML, 2014.  
Martin J Wainwright, Michael I Jordan, et al. Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1(1-2):1-305, 2008.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.
