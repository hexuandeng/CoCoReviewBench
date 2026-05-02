# ERGODIC MEASURE PRESERVING FLOWS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Training probabilistic models with neural network components is intractable in most cases and requires to use approximations such as Markov chain Monte Carlo (MCMC), which is not scalable and requires significant hyper-parameter tuning, or mean-field variational inference (VI), which is biased. While there has been attempts at combining both approaches, the resulting methods have some important limitations in theory and in practice. As an alternative, we propose a novel method which is scalable, like mean-field VI, and, due to its theoretical foundation in ergodic theory, is also asymptotically accurate, like MCMC. We test our method on popular benchmark problems with deep generative models and Bayesian neural networks. Our results show that we can outperform existing approximate inference methods.

# 1 INTRODUCTION

Approximate statistical inference with unnormalised density functions is fundamentally important problem both Bayesian and frequentist inference. In particular, the successes of many sophisticated generative models in machine learning rely on power inference algorithms. Markov chain Monte Carlo (MCMC) methods and variational inference (VI), originally developed in statistics and physics, are two most important approximate inference methods in machine learning, which has been widely used in all kinds of probabilistic models, like latent topic models (Blei et al., 2003), Boltzmann machines (Hinton, 2002; Salakhutdinov & Larochelle, 2010), Bayesian non-parametric models (Neal, 2000; Kurihara et al., 2007). However, they are facing great challenges in the recent research on probabilistic modelling with deep neural networks (NNs). In particular, Bayesian deep neural networks become popular in recent works, because it exploits the Bayesian framework to overcome the overfitting and data demanding problems in deep learning (Neal, 2012). Another interesting research direction is to use deep neural networks in latent variable models to transform the simple latent random variables into complex distribution, which is also known as to Deep Generative Models (DGM). DGMs have been proved to be very powerful generative models.

Inspired by DGMs, many recent works on variational inference adopted NN to construct flexible approximate distributions. In particular, variational autoencoders (Kingma & Welling, 2014) and normalising flows (Rezende & Mohamed, 2015) are two most influential works in this direction. However, due to lack of understanding of the convergence of specific NNs, the research of NN-based inference is focused on engineering the architecture of inference NNs based on experiments and heuristics.

In this work, we propose a novel approximate inference method based on the classic inference theory of MCMC. Our method is inspired by the idea of parallel simulations of MCMC and the recent advances in variational inference with NNs. Like these variational methods, it is straightforward to accelerate the computation of our method using parallelised simulations on Graphical Processing Units. More importantly, with solid theoretical foundations in the theory of MCMC, the proposed method guarantees asymptotic convergence to arbitrary distributions of interest. It is a great advantage over variational inference, because of the approximation bias in variational methods. Our method is also attractive to a wide range of probabilistic models without NNs and Bayesian NNs.

# 2 BACKGROUND

# 2.1 BAYESIAN NEURAL NETWORKS

Given data  $\mathcal{D} = \{\mathbf{x}_n, y_n\}_{n=1}^N$  formed by feature vectors  $\mathbf{x}_n$  and corresponding scalar targets  $y_n$ , we can assume that each  $y_n$  is obtained as  $y_n = f(\mathbf{x}_n; \boldsymbol{\theta}) + \epsilon_n$ , where  $f(\cdot; \boldsymbol{\theta})$  is the output of a deep neural network with weights  $\boldsymbol{\theta}$  and the  $\epsilon_n$  are independent noise variables with  $\epsilon_n \sim \mathcal{N}(0, \sigma^2)$ . This model specifies a likelihood function  $p(y_1, \ldots, y_n | \mathbf{x}_1, \ldots, \mathbf{x}_n, \boldsymbol{\theta})$  which can be combined with a Gaussian prior on  $\boldsymbol{\theta}$  to obtain a posterior distribution  $p(\boldsymbol{\theta} | \mathcal{D})$ . Predictions for the  $y_\star$  corresponding to a new feature vector  $\mathbf{x}_\star$  are then obtained by using the predictive distribution  $p(y_\star | \mathbf{x}_\star, \mathcal{D}) = \int p(y_\star | \mathbf{x}_\star, \boldsymbol{\theta}) p(\boldsymbol{\theta} | \mathcal{D}) d\boldsymbol{\theta}$ . However, integrating with respect to the posterior distribution  $p(\boldsymbol{\theta} | \mathcal{D})$  is intractable and approximations have to be performed in practice, with the most popular methods for this being VI and MCMC.

# 2.2 DEEP GENERATIVE MODELS

Generative models extract intrinsic structure from data by making use of latent variables. Let  $\mathcal{D}$  be a dataset with  $n$  data points  $\{\mathbf{x}_n\}_{n=1}^N$ . Given a latent representation  $\mathbf{z}$ , the data point  $\mathbf{x}$  is assumed to be sampled from the conditional distribution  $p_{\theta}(\mathbf{x}|\mathbf{z})$ , which is specified in terms of some parameters  $\theta$ . This conditional distribution is often referred to as the decoder. Given a prior distribution  $p(\mathbf{z})$  over the latent variables, the joint distribution of data and latent variables is  $p_{\theta}(\mathbf{x},\mathbf{z}) = p(\mathbf{z})p_{\theta}(\mathbf{x}|\mathbf{z})$ . The marginal probability of data  $\mathbf{x}$  under the model is then  $p_{\theta}(\mathbf{x}) = \int p(\mathbf{z})p_{\theta}(\mathbf{x}|\mathbf{z})d\mathbf{z}$ .

Until recently,  $p_{\theta}(\mathbf{x}|\mathbf{z})$  was typically specified using a simple distributional family, e.g. generalized linear models (Murphy, 2012; Bishop, 2006). However, more recently, deep generative models (DGMs) use deep neural networks with weights  $\theta$  to specify the decoder (Kingma & Welling, 2014; Goodfellow et al., 2014).

Maximum likelihood is a straightforward way to train probabilistic models with latent variables. However, the marginal likelihood  $p_{\theta}(\mathbf{x})$  is intractable to compute in DGMs and approximations are needed in practice. As before, the most popular methods for this are based on VI and MCMC. We briefly describe these methods in the following sections.

# 2.3 VARIATIONAL INFERENCE AND NORMALIZING FLOWS

VI approximates a complex posterior distribution with another simpler parametric distribution which is found by optimizing a lower bound on the marginal likelihood. Let the complex posterior be  $p_{\theta}(\mathbf{z}|\mathbf{x})$ , with associated marginal likelihood  $p(\mathbf{x})$ , and let  $q_{\phi}(\mathbf{z})$ , parameterized by  $\phi$ , be a simpler tractable distribution. The lower bound of the marginal likelihood is then defined as

$$
\log p _ {\boldsymbol {\theta}} (\mathbf {x}) \geq \mathbb {E} _ {q _ {\phi}} \left[ \log p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z}) - \log q _ {\phi} (\mathbf {z}) \right], \tag {1}
$$

which is often known as the evidence lower bound (ELBO). The more flexible the parametric family  $q_{\phi}$ , the better the approximation quality to the true posterior and the tighter the value of the ELBO.

Mean-field VI uses a form for  $q_{\phi}(\mathbf{z})$  which assumes independence between random variables. This reduces the computational cost of the optimization problem but often leads to poor performance with complex posterior distributions, such as the ones arising in DGMs or in Bayesian neural networks.

Amortization can be used to accelerate convergence and reduce computational cost when multiple inference problems have to be solved simultaneously. The optimization of (1) can be amortized by making  $q_{\phi}$  depend explicitly on the data  $\mathbf{x}$ . In this case,  $q_{\phi}(\mathbf{z})$  is replaced with  $q_{\phi}(\mathbf{z}|\mathbf{x})$ , where  $\phi$  are now the weights of a neural network which computes the parameters of a tractable parametric distribution on  $\mathbf{z}$  from  $\mathbf{x}$ . In this manner, for any new value of  $\mathbf{x}$ , we can readily obtain a corresponding variational approximation given by  $q_{\phi}(\mathbf{z}|\mathbf{x})$ .

Variational auto-encoders (VAEs) (Kingma & Welling, 2014) are DGMs trained by using mean-field VI with a Gaussian parametric distribution and amortization. Rezende & Mohamed (2015) improve over this method by using a more flexible variational family called normalizing flows (NFs). The NF family is obtained by applying  $L$  invertible non-linear transformations  $f_{1},\ldots ,f_{L}$  to a random variable  $\mathbf{z}_0$  with tractable density  $q_{0}(\mathbf{z}_{0})$  and exact simulation. The resulting output is a random

variable  $\mathbf{z}_L = f_L\circ \dots \circ f_1(\mathbf{z}_0)$  with density  $q_{\phi}(\mathbf{z}_{L}|\mathbf{x}) = q_{0}(\mathbf{z}_{0}|\mathbf{x})\prod_{l = 1}^{L}|\partial f_{l}(\mathbf{z}_{l - 1}) / \partial \mathbf{z}_{l - 1}|^{-1}$  and with  $\phi$  being the parameters of  $f_{1},\ldots ,f_{L}$

Stochastic gradient descent (SGD), in combination with the reparameterisation trick (Kingma & Welling, 2014), can be used for the scalable optimization of the ELBO in VAEs and NFs. However, the main limitation of VAEs and NFs is the bias present in their variational approximations. This bias can be quite high, even in the case of NFs, since the transformations  $f_{1}, \ldots, f_{L}$  have to be rather simple to ensure invertibility and to reduce computational costs.

# 2.4 MARKOV CHAIN MONTE CARLO

Markov chain Monte Carlo (MCMC) is an approximate inference method which does not have the aforementioned bias problem. MCMC works by simulating a stationary Markov chain that generates asymptotically unbiased samples from the distributions of interest. Formally, a Markov chain is a sequence of random variables  $\mathbf{z}_0, \mathbf{z}_1, \ldots$  in which the transition from state  $\mathbf{z}_l$  to  $\mathbf{z}_{l+1}$  is defined by the conditional probability distribution of  $\mathbf{z}_{l+1}$  given  $\mathbf{z}_l$ , denoted by  $K(\mathbf{z}_l, \mathbf{z}_{l+1})$ . Markov chains in MCMC methods have the following strong stationary property: if one state  $\mathbf{z}_l$  of the chain follows the stationary distribution  $\pi$ , so does the next state  $\mathbf{z}_{l+1}$ . In particular,

$$
\pi \left(\mathbf {z} _ {l + 1}\right) = \int \pi (\mathbf {z} _ {l}) K \left(\mathbf {z} _ {l}, \mathbf {z} _ {l + 1}\right) d \mathbf {z} _ {l}. \tag {2}
$$

If  $\mathbf{z}_l$  follows a distribution  $\pi_l$  that is different from the stationary one, then the distribution of  $\mathbf{z}_{l + 1}$

$$
\pi_ {l + 1} \left(\mathbf {z} _ {l + 1}\right) = \int \pi_ {l} \left(\mathbf {z} _ {l}\right) K \left(\mathbf {z} _ {l}, \mathbf {z} _ {l + 1}\right) d \mathbf {z} _ {l}, \tag {3}
$$

is guaranteed to be closer to  $\pi$  than  $\pi_{l}$ . This property implies that, with sufficiently many transitions, the distribution  $\pi_{l}$  of  $\mathbf{z}_{l}$  converges to  $\pi$  irrespectively of the distribution the initial state  $\mathbf{z}_0$ .

Despite beign asymptotically unbiased, MCMC methods are less popular than VI for two reasons. First, they are computationally more expensive and second, they typically include hyper-parameters in the Markov kernel  $K$  which are highly problem dependent and are hard to tune in practice.

# 3 ERGODIC MEASURE PRESERVING FLOWS

In this section, we describe an inference method that combines the strengths of MCMC and VI and avoids their drawbacks. The idea is to use the output distribution of a MCMC chain, given by (3), as the variational distribution and optimize a simple to evaluate objective function for tuning MCMC parameters. Since MCMC converges to the target asymptotically, our variational approximation can be arbitrarily accurate. The result is a computationally efficient method which avoids the bias of parametric approximations and which can do automatic tuning of hyper-parameters.

# 3.1 DEFINITIONS

Given the target distribution  $\pi$  with unnormalised density function  $\pi^{*}$ , we define an approximate distribution  $q$  by a mixture of sequential deterministic transformations that preserves the measure  $\pi^{*}$ . We call such approximate distributions measure preserving flows (MPFs). The transformations preserving a given measure  $\pi^{*}$  are formally defined as follows (Billingsley, 1986).

Definition 3.1. Measure Preserving Transformation (MPT). Let  $(\Omega, \mathcal{F}, P)$  be a probability space and  $\mu$  be a consistent measure with  $P$ . A mapping  $T: \Omega \to \Omega$  is a measure-preserving transformation if  $T$  is measurable in both the input field  $\mathcal{F}$  and the output field  $\mathcal{F}$  and  $\mu(A) = \mu(T^{-1}(A))$  for all  $A \in \mathcal{F}$ . If  $T$  is a one-to-one mapping onto  $\Omega$ , then  $\mathrm{T}$  preserves  $\mu: \mu(A) = \mu(T^{-1}TA) = \mu(TA)$ .

An example of MPT is any transformation with Jacobian determinant equal to 1, which preserves the Lebesgue measure. In practice, it is straightforward to verify if a transformation  $T$  preserves the measure with density function  $\pi^*$  with the following conditions:

(i)Bijection:  $T$  is invertible,  
(ii) Preservation of density function:  $\pi^{*}(\mathbf{z}) = \pi^{*}(T(\mathbf{z}))$  for all  $\mathbf{z}$ .

(iii) Preservation of base measure: the Jacobian determinant is one if the Lebesgue measure is the base measure.

In probability theory, MPTs are often used within the area of ergodic stochastic processes since many of these processes can be reformulated as a composition of MPTs. In particular, MCMC kernels are MPTs and stationary Markov chains in MCMC are ergodic processes (Robert & Casella, 2005).

The joint probability of states in a MCMC chain is  $q(\mathbf{z}_0, \mathbf{z}_1, \dots, \mathbf{z}_L) = q_0(\mathbf{z}_0) \prod_{l=1}^{L} K(\mathbf{z}_{l-1}, \mathbf{z}_l)$ , where  $q_0$  is the distribution of the initial state  $\mathbf{z}_0$ . The density of the last state  $\mathbf{z}_L$  is then obtained by integrating out all the previous states:

$$
q _ {L} \left(\mathbf {z} _ {L}\right) = \int q _ {0} \left(\mathbf {z} _ {0}\right) \prod_ {l = 1} ^ {L} K \left(\mathbf {z} _ {l - 1}, \mathbf {z} _ {l}\right) d \mathbf {z} _ {0} d \mathbf {z} _ {1} \dots d \mathbf {z} _ {L - 1}. \tag {4}
$$

If the Markov chain is ergodic,  $q_{L}$  converges to the stationary distribution  $\pi$  in total variation distance as the length  $L$  of the chain increases (Robert & Casella, 2005).

We define a measure preserving flow (MPF) as a representation of (4) in which the kernel  $K$  becomes a deterministic transformation  $T_{\mathbf{r}}:Z\to Z$  with stochastic auxiliary input  $\mathbf{r}$  following distribution  $\mu$ . We can then define  $\mathbf{z}_L$  as the result of applying these deterministic transformations to  $\mathbf{z}_0$ , that is,  $\mathbf{z}_L = T_{\mathbf{r}_L}\circ \dots \circ T_{\mathbf{r}_1}(\mathbf{z}_0)$ . By following the rule of changing variables, it is then straightforward to derive the density of  $\mathbf{z}_L$  as

$$
q _ {L} (\mathbf {z} _ {L}) = \int q \left(\mathbf {z} _ {L}, \mathbf {r} _ {1: L}\right) d \mathbf {r} _ {1: L} = \int q \left(\mathbf {z} _ {0}\right) \mu \left(\mathbf {r} _ {1: L}\right) \delta \left(\mathbf {z} _ {L} - T _ {\mathbf {r} _ {L}} \circ \dots \circ T _ {\mathbf {r} _ {1}} \left(\mathbf {z} _ {0}\right)\right) d \mathbf {z} _ {0} d \mathbf {r} _ {1: L}, \tag {5}
$$

where  $\delta$  denotes the Dirac delta function. Note that there is no Jacobian term in (5) because of the preservation of the Lebesgue measure. Because MPFs are equivalent to ergodic Markov chains, the density obtained at the output of an MPF, that is,  $q_{L}$ , will converge to the stationary distribution  $\pi$  as  $L$  increases.

Hamiltonian Monte Carlo (HMC) is one of the most successful MCMC methods, which also can be interpreted as an MPF. Given the target random variable  $\mathbf{z} \in \mathbb{R}^d$  with unnormalised density function  $\pi^*$ , the HMC kernel is essentially applying a deterministic transformation  $\mathcal{H}$  to the previous state  $\mathbf{z}_i$  with an auxiliary random variable  $\mathbf{r} \in \mathbb{R}^n$  with the density function  $\mu(\cdot)$ . The transformation  $\mathcal{H}$  is given by the solution of Hamiltonian dynamics

$$
\dot {\mathbf {z}} (t) = \partial_ {\mathbf {r}} K (\mathbf {r}), \quad \dot {\mathbf {r}} (t) = - \partial_ {\mathbf {z}} U (\mathbf {z}), \tag {6}
$$

where  $\dot{\mathbf{z}}$  denotes the derivative of  $\mathbf{z}$  w.r.t. the time of the dynamics  $t$ ,  $U(\mathbf{z}) = -\log \pi^{*}(\mathbf{z})$  and  $K(\mathbf{r}) = -\log \mu (\mathbf{r})$ . The preservation of total Hamiltonian energy,  $H(\mathbf{z},\mathbf{r}) = U(\mathbf{z}) + K(\mathbf{r})$ , is the characteristic property of Hamiltonian dynamics, which can be easily verified by  $\dot{H} (\mathbf{z},\mathbf{r}) = 0$  using (6). It is straightforward to see that  $\mathcal{H}$  preserves the joint measure  $\pi (\mathbf{z})\mu (\mathbf{r})$ . In particular,  $\mathcal{H}$  is a bijective transformation because the dynamics are deterministic and time reversible (Neal, 2010) and the preservation of the Hamiltonian energy implies the preservation of density. Finally, the volume preservation of  $\mathcal{H}$  in the space of  $(\mathbf{z},\mathbf{r})$  is a well known property of Hamiltonian dynamics, which can be proved by Liouville's theorem (Leimkuhler & Reich, 2004; Neal, 2010).

We can write the marginal distribution of the last sample generated by HMC as an MPF:

$$
q \left(\mathbf {z} _ {L}\right) = \int q \left(\mathbf {z} _ {L}, \mathbf {r} _ {1: L}\right) d \mathbf {r} _ {1: L} = \int q \left(\mathbf {z} _ {0}\right) \mu (\mathbf {r}) \delta \left(\mathbf {z} _ {L} - \mathcal {H} _ {\mathbf {r} _ {L}} \circ \dots \circ \mathcal {H} _ {\mathbf {r} _ {1}} \left(\mathbf {z} _ {0}\right)\right) d \mathbf {z} _ {0} d \mathbf {r} _ {1: L}. \tag {7}
$$

We call the MPF generated by Hamiltonian dynamics a Hamiltonian MPF (HMPF).

# 3.2 UNDERSTANDING MEASURE PRESERVING CONDITIONS

We would like to address a common misunderstanding on the preservation of volume condition stated by (iii) in Section 3.1. Note that we are interested in sampling the random variable  $\mathbf{z}$ , but the Hamiltonian dynamics preserve the joint measure  $\pi (\mathbf{z},\mathbf{r})$  rather than  $\pi (\mathbf{z})$ . Following the conditions of MPTs in Section 3.1, it seems necessary to show that, for any specific value of  $\mathbf{r}$ , any  $T_{\mathbf{r}}$  used within an MPF should preserve volume in  $\mathbf{z}$  space. However, this is not the case since the measure preservation conditions in the augmented space  $(\mathbf{z},\mathbf{r})$  are enough to guarantee the preservation of the marginal distribution in  $\mathbf{z}$  space. Formally, we have the following proposition:

Proposition 1. Let  $T: Z \times \mathcal{E} \to Z \times \mathcal{E}$  preserve the distribution  $\pi(\mathbf{z}, \mathbf{r})$ . Then, if  $\mathbf{r}$  is sampled from  $\pi(\mathbf{r}) = \int \pi(\mathbf{z}, \mathbf{r}) dr$ , the marginal distribution

$$
\pi (\mathbf {z}) = \int \pi (\mathbf {z}, \mathbf {r}) d \mathbf {r}
$$

is also preserved by the projection of  $T$  in the space of  $\mathbf{z}$ , that is, by  $T_{\mathbf{r}}: Z \to Z$ .

Proposition 1 gives us some insights on the difference between MPFs and normalising flows (NFs). As mentioned earlier, NFs also use a sequence of transformations  $T_{\mathbf{r}}: Z \to Z$ . However, these do not preserve the distribution of  $\mathbf{z}$  and, consequently, they require the computation of Jacobian determinants by the rule of changing variables. By contrast, Proposition 1 implies that, in MPFs,  $T_{\mathbf{r}}$  preserves the marginal  $\pi(\mathbf{z})$  if  $T$  preserves the joint, which means that there is no need to include any Jacobian computations. For this reason, the transformations used in MPFs can be much more complicated than those used in NFs. For example, in Hamiltonian MPFs, for a given  $\mathbf{r}$ , the Jacobian of  $\mathcal{H}_{\mathbf{r}}$  can be very complicated.

# 3.3 VARIATIONAL INFERENCE WITH MPFS

Given an unnormalized posterior distribution  $p_{\theta}(\mathbf{x}, \mathbf{z})$  for  $\mathbf{z}$ , we can construct an MPF that preserves  $p_{\theta}(\mathbf{x}, \mathbf{z}) \mu(\mathbf{r})$ , where  $\mu(\mathbf{r})$  is a simple distribution with tractable density and sampling algorithm. Let  $T^{\phi_l}$  be the  $l$ -th transformation in the flow, where  $\phi_l$  are hyper-parameters. This transformation maps the state of the flow from  $(\mathbf{z}_{l-1}, \mathbf{r}_l)$  to  $(\mathbf{z}_l, \mathbf{r}_l') = T^{\phi_l}(\mathbf{z}_{l-1}, \mathbf{r}_l)$ . Similarly, the composition  $T^{\phi_L} \circ \dots \circ T^{\phi_1}(\mathbf{z}_0, \mathbf{r}_{1:L})$ , which we denote by  $T^\phi$ , transforms  $(\mathbf{z}_0, \mathbf{r}_{1:L})$  to  $(\mathbf{z}_L, \mathbf{r}_{1:L}')$ . By the preservation of density and the preservation of Lebesgue measure of  $T^\phi$ , as given by conditions (ii) and (iii) in Section 3.1), we have the following equalities

$$
p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z} _ {0}) \prod_ {l = 1} ^ {L} \mu (\mathbf {r} _ {l}) = p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z} _ {1}) \mu \left(\mathbf {r} _ {1} ^ {\prime}\right) \prod_ {l = 2} ^ {L} \mu (\mathbf {r} _ {l}) = \dots = p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z} _ {L}) \prod_ {l = 1} ^ {L} \mu \left(\mathbf {r} _ {l} ^ {\prime}\right), \tag {8}
$$

$$
q _ {0} \left(\mathbf {z} _ {0}\right) \prod_ {l = 1} ^ {L} \mu \left(\mathbf {r} _ {l}\right) = q _ {L} \left(\mathbf {z} _ {1}, \mathbf {r} _ {1} ^ {\prime}; \phi_ {1}\right) \prod_ {l = 2} ^ {L} \mu \left(\mathbf {r} _ {l}\right) = \dots = q _ {L} \left(\mathbf {z} _ {L}, \mathbf {r} _ {1} ^ {\prime}, \mathbf {r} _ {2} ^ {\prime} \dots , \mathbf {r} _ {L} ^ {\prime}; \phi\right), \tag {9}
$$

where  $q_{0}(\mathbf{z}_{0})$  is an initial proposal distribution and  $\phi = (\phi_{1},\dots,\phi_{L})$  are the transformation hyperparameters. It is important to clarify that, according to (9), the joint density of  $\mathbf{z}_L,\mathbf{r}_1',\mathbf{r}_2',\ldots ,\mathbf{r}_L'$  is known, but the marginal density for these variables is intractable to compute in general.

Following (1), we can obtain the ELBO for the initial proposal distribution  $q_{0}(\mathbf{z}_{0})$  as

$$
\mathcal {L} (\mathbf {x}; \boldsymbol {\theta}) = \int \log \frac {p _ {\boldsymbol {\theta}} (\mathbf {x} , \mathbf {z} _ {0})}{q _ {0} (\mathbf {z} _ {0})} q _ {0} (\mathbf {z} _ {0}) d \mathbf {z} _ {0}. \tag {10}
$$

We call this expression the simple ELBO. We can then multiplying by the density of the auxiliary variables  $\mu (\mathbf{r}_{1:L}) = \prod_{l = 1}^{L}\mu (\mathbf{r}_l)$  to obtain

$$
\mathcal {L} (\mathbf {x}; \boldsymbol {\theta}) = \int \log \frac {p _ {\boldsymbol {\theta}} (\mathbf {x} , \mathbf {z} _ {0}) \prod_ {l = 1} ^ {L} \mu (\mathbf {r} _ {l})}{q _ {0} (\mathbf {z} _ {0}) \prod_ {l = 1} ^ {L} \mu (\mathbf {r} _ {l})} q _ {0} (\mathbf {z} _ {0}) \prod_ {l = 1} ^ {L} \mu (\mathbf {r} _ {l}) d \mathbf {z} _ {0} d \mathbf {r} _ {1: L}. \tag {11}
$$

We can then replace  $(\mathbf{z}_0,\mathbf{r}_{1:L})$  with  $(\mathbf{z}_L,\mathbf{r}_{1:L}^{\prime})$  in (11) by making use of using transformation  $T^{\phi}$ , (8) and (9). The result is

$$
\mathcal {L} (\mathbf {x}; \boldsymbol {\theta}, \phi) = \int \log \frac {p _ {\boldsymbol {\theta}} (\mathbf {x} , \mathbf {z} _ {L}) \prod_ {l = 1} ^ {L} \mu \left(\mathbf {r} _ {l} ^ {\prime}\right)}{q _ {L} \left(\mathbf {z} _ {L} , \mathbf {r} _ {1 : L} ^ {\prime} ; \phi\right)} q _ {L} \left(\mathbf {z} _ {L}, \mathbf {r} _ {1: L} ^ {\prime}; \phi\right) d \mathbf {z} _ {L} d \mathbf {r} _ {1: L} ^ {\prime}, \tag {12}
$$

where we have omitted the dependence of  $(\mathbf{z}_L, \mathbf{r}_{1:L}^{\prime})$  on  $\phi$ , since these variables are determined by the hyper-parameters of the MPTs. We call (12) the reparameterised ELBO.

# 3.4 ERGODIC LOWER BOUND AND ERGODIC INFERENCE

The reparameterised ELBO is of limited use, because it can only be as tight as the ELBO with initial proposal distribution  $q_{0}$ . This seems to erase the benefits of using an ergodic MPF, which we know

will converge to the target posterior distribution given a sufficiently long flow. To overcome the drawback of the reparameterised ELBO, we propose another ELBO tailored to the MPF framework, which becomes arbitrarily tight as the length of the flow grows. We call such an ELBO ergodic lower bound (ERLBO).

To derive ERLBO, we first rewrite (12) as

$$
\begin{array}{l} \mathcal {L} (\mathbf {x}; \pmb {\theta}, \phi) = \int \log \frac {p _ {\pmb {\theta}} (\mathbf {x} , \mathbf {z} _ {L})}{q _ {L} (\mathbf {z} _ {L} ; \phi)} q _ {L} (\mathbf {z} _ {L}; \phi) d \mathbf {z} _ {L} + \int \log \frac {\prod_ {l = 1} ^ {L} \mu (\mathbf {r} _ {l} ^ {\prime})}{q _ {L} (\mathbf {r} _ {1 : L} ^ {\prime} | \mathbf {z} _ {L} ; \phi)} q _ {L} (\mathbf {z} _ {L}, \mathbf {r} _ {1: L} ^ {\prime}; \phi) d \mathbf {z} _ {L} d \mathbf {r} _ {1: L} ^ {\prime} \\ = \int \log \frac {p _ {\boldsymbol {\theta}} (\mathbf {x} , \mathbf {z} _ {L})}{q _ {L} \left(\mathbf {z} _ {L} ; \phi\right)} q _ {L} \left(\mathbf {z} _ {L}; \phi\right) d \mathbf {z} _ {L} - D _ {\mathrm {K L}} ^ {L}. \tag {13} \\ \end{array}
$$

where  $D_{\mathrm{KL}}^{L}$  is the Kullback-Liebler divergence between  $q_{L}(\mathbf{z}_{L},\mathbf{r}_{1:L}^{\prime};\phi)$  and  $q_{L}(\mathbf{z}_{L};\phi)\prod_{l = 1}^{L}\mu (\mathbf{r}_{l}^{\prime})$  It is straightforward to show that the first term on the RHS in (13) is a lower bound of the marginal likelihood by Jensen's inequality. This leads to the ERLBO given by

$$
\tilde {\mathcal {L}} (\mathbf {x}; \boldsymbol {\theta}, \phi) = \int \log \frac {p _ {\boldsymbol {\theta}} (\mathbf {x} , \mathbf {z} _ {L})}{q _ {L} (\mathbf {z} _ {L} ; \phi)} q _ {L} (\mathbf {z} _ {L}; \phi) d \mathbf {z} _ {L} = \mathcal {L} (\mathbf {x}; \boldsymbol {\theta}, \phi) + D _ {\mathrm {K L}} ^ {L}. \tag {14}
$$

This is a tighter lower bound than the simple ELBO because the difference between  $\tilde{\mathcal{L}} (\mathbf{x};\pmb {\theta},\phi)$  and  $\mathcal{L}(\mathbf{x};\pmb {\theta})$  in (10) is  $D_{\mathrm{KL}}^L\geq 0$  . Moreover, the ERLBO can be shown to monotonically increase w.r.t.  $L$

Proposition 2. The lower bound  $\tilde{\mathcal{L}} (\mathbf{x};\pmb {\theta},\phi)$  in (14) becomes tighter and tighter as  $L$  increases, that is,  $\tilde{\mathcal{L}} (\mathbf{x};\pmb {\theta},\phi_{1:L})\geq \tilde{\mathcal{L}} (\mathbf{x};\pmb {\theta},\phi_{1:L - 1})$  and the equality holds if and only if  $D_{KL}^{L} = 0$

The complete proof is included in appendix.

Recall that  $q_{L}(\mathbf{z}_{L};\phi)$  is obtained by a sequence of transformations that preserve the probability measure  $p(\mathbf{z}|\mathbf{x})$ . It is well-known that MCMC chains have a unique invariant distribution and so do the MCMC-equivalent MPFs. Therefore, we know that if  $\tilde{\mathcal{L}} (\mathbf{x};\boldsymbol {\theta},\phi_{1:L})$  stops growing,  $q_{L}(\mathbf{z}_{L};\phi)$  must converge to  $p(\mathbf{z}|\mathbf{x})$ . This is formally described by the following theorem.

Theorem 1. Given an ergodic measure preserving flow with invariant measure  $\pi$ , the ergodic lower bound  $\tilde{\mathcal{L}}(\mathbf{x}; \boldsymbol{\theta}, \boldsymbol{\phi})$  increases in the length of the flow  $L$  and becomes an unbiased estimator of the marginal log  $p(\mathbf{x})$  as  $L$  increases to infinity.

We could tune  $\phi$  by optimizing the ERLBO. To better understand the values of  $\phi$  that would be favored by this optimisation process, we can rewrite the ERLBO by making explicit its dependence on the entropy of  $q_{L}(\mathbf{z}_{L};\phi)$ , which we denote by  $\mathrm{H}[q_L(\mathbf{z}_L;\phi)] = -\int \log q_L(\mathbf{z}_L;\phi)q_L(\mathbf{z}_L;\phi)d\mathbf{z}_L$ . In particular,

$$
\tilde {\mathcal {L}} (\mathbf {x}; \boldsymbol {\theta}, \phi) = \mathbf {E} _ {q _ {L} (\mathbf {z} _ {L}; \phi)} \left[ \log p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z} _ {L}) \right] + \mathrm {H} \left[ q _ {L} (\mathbf {z} _ {L}; \phi) \right]. \tag {15}
$$

When optimizing this quantity w.r.t.  $\phi$ , the first term in the RHS will encourage  $q_{L}(\mathbf{z}_{L};\phi)$  to have high density in regions where  $p_{\theta}(\mathbf{x},\mathbf{z}_L)$  is high, while the second term will favor high entropy solutions and will prevent  $q_{L}(\mathbf{z}_{L};\phi)$  from converging to a Dirac delta centered at the maximizer of  $\log p_{\theta}(\mathbf{x},\mathbf{z}_L)$ . Note that the first term in the RHS of (15) can be easily approximated by Monte Carlo, while the second term is intractable because  $q_{L}(\mathbf{z}_{L};\phi)$  is not available. However, since  $q_{L}(\mathbf{z}_{L};\phi)$  converges to  $p_{\theta}(\mathbf{z}|\mathbf{x})$  as  $L$  increases, we expect the effect of  $\mathrm{H}[q_L(\mathbf{z}_L;\phi)]$  on  $\phi$  to be small and that most of the similarity of  $q_{L}(\mathbf{z}_{L};\phi)$  to  $p_{\theta}(\mathbf{z}|\mathbf{x})$  will be captured by the first term in the RHS of (15). Therefore, we propose to tune  $\phi$  by optimizing the tractable objective given by the first term, that is,

$$
\tilde {\mathcal {F}} (\mathbf {x}; \boldsymbol {\theta}, \phi) = \mathbf {E} _ {q _ {L} \left(\mathbf {z} _ {L}; \phi\right)} \left[ \log p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z} _ {L}) \right]. \tag {16}
$$

If with the initial flow parameter  $\phi_0$ , the objective  $\tilde{\mathcal{F}} < \mathbf{E}_{p_{\theta}(\mathbf{x},\mathbf{z})}\left[\log p_{\theta}(\mathbf{x},\mathbf{z})\right]$ , we expect optimising  $\tilde{\mathcal{F}}$  produces faster convergence of  $q_{L}(\mathbf{z}_{L};\boldsymbol {\phi})$  towards  $p_{\theta}(\mathbf{x},\mathbf{z})$ . Importantly, the model parameters  $\pmb{\theta}$  can also be adjusted by optimizing  $\tilde{\mathcal{F}}$  because the omitted  $\mathrm{H}[q_L(\mathbf{z}_L;\boldsymbol {\phi})]$  does not depend on  $\pmb{\theta}$  and, consequently, optimizing  $\tilde{\mathcal{F}} (\mathbf{x};\pmb {\theta},\boldsymbol {\phi})$  and  $\tilde{\mathcal{L}} (\mathbf{x};\pmb {\theta},\boldsymbol {\phi})$  w.r.t.  $\pmb{\theta}$  are equivalent operations.

# 3.5 IMPLEMENTATION OF HMPFS

The hyper-parameters to be tuned in a HMPF include the parameters of  $q_{0}$  and the transformation parameters for the Hamiltonian simulation, that is,  $\phi = (\phi_{1},\dots,\phi_{L})$ . A natural choice for  $q_{0}$

is multivariate Gaussian with mean  $\mu = (\mu_1, \dots, \mu_d)$  and diagonal covariance matrix with entries  $\sigma^2 = (\sigma_1^2, \dots, \sigma_d^2)$ , where  $d$  is the dimensionality of the sample space. The most popular algorithm for simulating Hamiltonian dynamics is the vanilla Leapfrog integrator. We refer to the tutorial of Neal (2010) for more detailed description of the implementation of this algorithm. Leapfrog is a numeric integrator that approximates the Hamiltonian dynamics (6) by an iterative procedure with discretized time  $\Delta t$ , that is,

$$
\mathbf {x} (t + \Delta t) = \mathbf {x} (t) + \Delta t \partial_ {\mathbf {r}} K (\mathbf {r} (t)), \quad \mathbf {r} (t + \Delta t) = \mathbf {r} (t) - \Delta t \partial_ {\mathbf {x}} U (\mathbf {x} (t)). \tag {17}
$$

For the flow parameters, we consider the total simulation time  $T$ . Given a fixed number of Leapfrog iterations  $m$ , the simulation time  $T$  can be reparameterized as the time step size  $\Delta t = T / m$ . Neal (2010) shows that it is possible to use different  $\Delta t$  for each dimension of the sample space to improve the quality of Leapfrog. Therefore, we consider the parameters of the  $l$ -th Hamiltonian simulation in the flow to be  $\phi_{l} = (\Delta t_{l,1},\dots,\Delta t_{l,d})$ . The pseudo code for ergodic inference with HMPFs is shown in Algorithm 1.

Algorithm 1: Ergodic Inference on Hamiltonian Measure Preserving Flow.  
input: potential function  $U(\mathbf{z};\mathbf{x},\pmb {\theta})$  dataset  $\mathcal{D}$  and large  $L$    
output: optimal decoder and flow parameters  $\pmb{\theta}^{*}$  and  $\phi^*$    
initialize  $\pmb{\theta}$  and  $\phi$    
while not converged do.   
 $\mathbf{x}\gets$  sample one data point from  $\mathcal{D}$  .   
 $\mathbf{z}_0\sim \mathcal{N}(\boldsymbol {\mu},\mathrm{diag}(\sigma^2));$    
/\* Start of simulation of HMPF /\*/   
for  $l = 1,\dots ,L$  do  $\begin{array}{rlr}{\bf r}\sim \mathcal{N}(0,1); & & \\ {\bf z}_l\leftarrow \mathcal{H}({\bf z}_{l - 1},{\bf r};U({\bf z};{\bf x},\pmb {\theta}),\phi_l); & /*\mathrm{Leapfrog~simulation} & \ast /\\ \end{array}$    
end   
/\* End of simulation of HMPF /\*/   
obj  $\longleftarrow$  U(zL;x,0); /\* one sample Monte Carlo Approx. of  $\tilde{\mathcal{F}} (\mathbf{x};\pmb {\theta},\pmb {\phi})$  /\*   
 $\pmb{\theta}\gets$  AdamUpdate(0,  $\partial_{\theta}$  obj);   
 $\phi \gets$  AdamUpdate(  $\phi ,\partial_{\phi}$  obj);   
end

We do not include any Metropolis-Hastings (MH) correction steps in our method since this is not necessary. The MH steps are included in MCMC methods to ensure asymptotic convergence to the correct target with an unlimited number of transitions. By contrast, MPFs are in the finite-length regime and the main concern is to accelerate the convergence of MPFs to be as close as possible to the target measure. In this setting, it is not immediately clear that MH steps would be helpful. In particular, in Section 4.1 we provide empirical evidence of how HMPFs can converge to the correct target without MH steps.

Due to the composition of the MPTs, computing the gradient can be expensive when the flow is long. To speed up training, we stop the gradient computations when evaluating  $-\partial_{\mathbf{z}}U(\mathbf{z})$  in the Leapfrog steps. This trick leads to incorrect gradients. However, we noticed that the optimization was not significantly affected by this and still worked very well in practice. Finally, note that working with incorrect gradients does not affect the convergence of the flow to the correct target distribution because that is guaranteed by the convergence of the ERLBO as we mentioned in previous section.

# 4 EXPERIMENTS

We provide empirical evidence of HMPFs in three inference tasks. Our goal is to show that HMPFs can provide better approximations than other approximate inference methods.

# 4.1 DEMONSTRATION OF CONVERGENCE

To verify the theoretical results on the convergence of MPFs in Section 3.4, we test HMPFs on 8 bivariate distributions. The full list of benchmark distributions and results are included in the

appendix. Here, we focus on two multimodal benchmarks. The first testing target distribution is a bimodal moon shaped distribution as shown in Figure 2a. We call this target dual moon. Dual moon is one of the benchmarks in normalising flows (Rezende et al., 2014). The second testing target is a mixture of 6 Gaussian distributions placed in a circle. We use 15 Hamiltonian transformations with 5 Leapfrog steps each. The architecture detail of HMPFs<sup>1</sup> can be found in Section D.1. The initial state of MPFs is sampled from a standard Gaussian distribution. The gradient of the objective function is estimated using 1000 samples from HMPFs.

To illustrate the convergence of HMPFs to the target distribution, figures 1c and 1d show histograms of samples as a function of the flow length and the training iterations. To confirm the convergence numerically, we compute the ERLBO using a numeric method for the estimation of the entropy. Plots for the ERLBO and the ground truth log normalization constants are show in figures 1a and 1b

# 4.2 DEEP GENERATIVE MODELS

MNIST is a standard benchmark for testing approximate inference algorithms for training deep generative models. This dataset contains 60,000 grey level  $28 \times 28$  images of handwritten digits. For fair comparison with previous work, we use the 10,000 prebinarised MNIST test images from (Burda et al., 2015) $^{2}$ . Our benchmark deep generative model is based on the deconvolutional network used by Salimans et al. (2015) for testing Hamiltonian variational inference (HVI). In particular, the decoder  $p_{\theta}(\mathbf{x},\mathbf{z})$  consists of 32 dimensional latent variables  $\mathbf{z}$  with isotropic Gaussian prior  $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$  and a deconvolutional network with the architecture from top to bottom including a single fully-connected layer with 500 RELU hidden units, then three deconvolutional layers with  $5 \times 5$  filters, [16, 32, 32] feature maps and RELU activations and the final output layer is simply element-wise logistic activation functions. In the convolutional VAE, the encoder network mirrors the architecture of the decoder.

The code for HVI (Salimans et al., 2015) is not publicly available. Nevertheless, we implemented their convolutional VAE and were able to reproduce the marginal likelihood reported by Salimans et al. (2015), as shown in Table 1. This verifies that our implementation of the convolutional VAE is correct and that our results are comparable to the ones reported originally by Salimans et al. (2015). We also implemented HVI following Salimans et al. (2015). We used a single hidden layer network with 640 hidden units and RELU activations as the reverse model for the HMC transitions. We also implemented another VI method similar to HVI and called the Hamiltonian variational encoder (HVAE) (Caterini et al., 2018). Unlike HVI, HVAE does not use a reverse model. This method optimizes instead a bound derived from the stationary distribution of reverse momenta. Furthermore, HVAE uses tempering Hamiltonian dynamics that requires additional Jacobian corrections. In our implementation of HVAE, we simply ignore the temperature for computational efficiency.

<table><tr><td>Encoders</td><td>Training hours</td><td>Training Epochs</td><td>Test log(x)</td><td>ESS</td></tr><tr><td>Conv VAE(nh=300) (Salimans et al., 2015)</td><td>-</td><td>-</td><td>-83.20</td><td>-</td></tr><tr><td>HVI(1HMPF-16LF, nh=800) (Salimans et al., 2015)</td><td>-</td><td>-</td><td>-81.94</td><td>-</td></tr><tr><td>HVAE(1HMPF-20LF, nh=300)(Caterini et al., 2018)</td><td>-</td><td>-</td><td>-84.78</td><td>-</td></tr><tr><td>Conv VAE(nh=500) (Baseline)</td><td>6.00</td><td>3000</td><td>-83.57</td><td>50</td></tr><tr><td>HVI(1HMPF-16LF, nh=800)</td><td>6.00</td><td>360</td><td>-83.68</td><td>48</td></tr><tr><td>HVAE(1HMPF-16LF, nh=500)</td><td>6.00</td><td>360</td><td>-84.22</td><td>48</td></tr><tr><td>HMPF(30HMPT-5LF, nh=500, no encoder network)</td><td>1.65</td><td>54</td><td>-83.17</td><td>48</td></tr><tr><td>HMPF(30HMPT-5LF, nh=500, no encoder network)</td><td>3.00</td><td>100</td><td>-82.76</td><td>46</td></tr><tr><td>HMPF(30HMPT-5LF, nh=500, no encoder network)</td><td>6.00</td><td>200</td><td>-82.65</td><td>45</td></tr><tr><td>HMPF(30HMPT-5LF, nh=500, no encoder network)</td><td>12.00</td><td>400</td><td>-81.43</td><td>38</td></tr></table>

Table 1: Comparison in terms of computational efficiency and approximate test log-likelihood. For fair comparison, we implemented the deconvolutional decoder network in (Salimans et al., 2015) to test HVI. In (Salimans et al., 2015), the test likelihood is estimated using importance-weighted samples from the encoder network. In our experiment, we use a more reliable estimation method based on Hamiltonian annealed importance sampling and report the effective sample size (ESS).

![](images/278e9d0291d3d376b8feb91c85781d2680b2eb3050d62b1d7bd195b273a6b02f.jpg)  
(a) Dual Moon

![](images/dff0c5f52a386d0a12c16dc2567af53832b7cba97be05c3e082cc1ec26654e74.jpg)

![](images/9455c0c9862059119411f38c392d73ebd0190229a4a6b314b685d93701a56309.jpg)  
(b) Circular Gaussian Mixture

![](images/f7f11b1740a669f7bc90d499d18f21fcd731f182419e134a1d03fd9b75f40d36.jpg)  
(c) Dual moon Mixture  
(d) Circular Gaussian Mixture  
Figure 1: The demonstration of the convergence of measure preserving flows. Figure (a) and (b) show ergodic lower bounds to the true log normalising constant of ergodic measure preserving flows with 14 transformations. The lower bound is estimated after each transformation as indicated by the axis 'flow length'. The legend 'ERLBO-0' indicates the ERLBO of the flow with the initial randomized flow parameters and the legend 'ERLBO-80' indicates the ERLBO of the flow after 80 training iterations of the flow parameters. Figure (c) and (d) show how the histograms of the 50000 samples from the flow evolve after each transformation (flow direction axis) and every 10 training iterations (training iterations axis).

For HMPF encoder, we use 30 HMPTs with 5 Leapfrog steps per HMPT. The initial distribution  $q_{0}$  is 32 dimensional independent Gaussian. More detailed description of the architecture of HMPFs is in the appendix Section D.1. We optimise the HMPF encoder and the decoder jointly using Adam. The initial state of the flow is sampled from independent Gaussian. The mean and variance of the initial Gaussian is also optimised jointly with flow and model parameters and their gradients are computed by back propagation given the value of momenta. However, we noticed that with sufficient number of transformations, the effect of optimising initial Gaussian distribution is not significant. Table 1 shows the marginal likelihood of HMPFs and other methods estimated using 100 Hamiltonian annealled importance samples (HAIS) (Wu et al., 2017). We also report the effective sample size (ESS). Overall, HMPFs produce better results and are faster than the baselines.

We also tested the same decoder with HMPFs and convolutional encoder on dynamically binarised Fashion-MNIST (Xiao et al., 2017). The results of test marginal likelihood can be found Table 2.

<table><tr><td>Encoders</td><td>Training hours</td><td>Training Epochs</td><td>Test log(x)</td><td>ESS</td></tr><tr><td>Conv VAE(nh=500)</td><td>6.00</td><td>3000</td><td>-104.90</td><td>26.3</td></tr><tr><td>HMPF(30HMPT-5LF, nh=500, no encoder)</td><td>6.00</td><td>200</td><td>-103.087</td><td>16.2</td></tr></table>

# 4.3 BAYESIAN NEURAL NETWORKS

In our final experiment we approximate the posterior distribution of Bayesian neural networks. We use four UCI datasets and compare HMPFs with relevant stochastic gradient Hamilton Monte Carlo (SGHMC) methods from (Springenberg et al., 2016). The networks used in these experiments have 50 hidden layers and 1 real valued output unit, as stated in (Springenberg et al., 2016). The HMPFs contain 50 HMC transformation with 3 Leapfrog steps each. The distribution the initial state of the flow is independent Gaussian with mean and variance parameters obtained by fitting a variational Gaussian proposal  $q_{0}$  with Adam optimiser for 200 iterations. To reduce the cost of the Leapfrog iterations, we split training data into 19 mini-batches and only use one random sampled mini-batch for computing the gradient of the potential energy. We train our HMPFs for 10 epochs and the stationary distribution of the flow is chosen as approximate posterior on a random sampled minibatch. The resulting test log-likelihoods are shown in Table 3. Overall, HMPFs produce significantly better results than SGHMC.

Table 2: The comparison of log marginal likelihood on fashion MNIST between convolutional VAE and HMPFs. We also evaluate HMPFs with different settings of HAIS that gives higher effective sample size (ESS), but the result of test log likelihood is roughly the same.  

<table><tr><td>Method/Dataset</td><td>Boston</td><td>Yacht</td><td>Concrete</td><td>Wine</td></tr><tr><td>SGHMC (best average) (Springenberg et al., 2016)</td><td>-3.47±0.51</td><td>-13.58±0.98</td><td>-4.87±0.05</td><td>-1.82±0.75</td></tr><tr><td>SGHMC (tuned per dataset) (Springenberg et al., 2016)</td><td>-2.49±0.15</td><td>-1.75±0.19</td><td>-4.16±0.72</td><td>-1.29±0.28</td></tr><tr><td>SGHMC (scale-adapted) (Springenberg et al., 2016)</td><td>-2.54±0.04</td><td>-1.11±0.08</td><td>-3.38±0.24</td><td>-1.04±0.17</td></tr><tr><td>HMPFs</td><td>-2.17±0.07</td><td>-0.47±0.06</td><td>-2.71±0.03</td><td>-0.71±0.03</td></tr></table>

Table 3: The test log-likelihood of Bayesian neural networks on UCI datasets averaged over 20 splits with 100 sampled network parameters from HMPFs.

# 5 SUMMARY

We have proposed a novel method for approximate inference that combines advantages of variational inference and MCMC methods. We call this method ergodic measure preserving flows (EMPFs). Different from most previous works combining HMC and variational inference, EMPFs enjoy the same asymptotic convergence as HMC and can tune sampling hyper-parameters by optimizing a tractable objective function at a low computational cost. We have shown that EMPFs achieve better results than existing baselines on standard benchmarks. For future work, it will be interesting to study the convergence rate of EMPFs to the target distribution with increasing flow length. Finally, the proposed method can be easily extended to consider recent Riemannian-manifold HMC methods (Zhang & Sutton, 2014; Girolami & Calderhead, 2011) for the construction of the flow.

# REFERENCES

Patrick Billingsley. Probability and Measure. John Wiley and Sons, third edition, 1986.  
Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg, 2006. ISBN 0387310738.  
David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan):993-1022, 2003.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
A. L. Caterini, A. Doucet, and D. Sejdinovic. Hamiltonian Variational Auto-Encoder. ArXiv e-prints, May 2018.  
Mark Girolami and Ben Calderhead. Riemann manifold langlevin and hamiltonian monte carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2):123-214, 2011. ISSN 1467-9868. doi: 10.1111/j.1467-9868.2010.00765.x. URL http://dx.doi.org/10.1111/j. 1467-9868.2010.00765.x.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation, 14(8):1771-1800, 2002.  
Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In The International Conference on Learning Representations (ICLR), 2014.  
Kenichi Kurihara, Max Welling, and Yee Whye Teh. Collapsed variational dirichlet process mixture models. In IJCAI, volume 7, pp. 2796-2801, 2007.  
Benedict Leimkuhler and Sebastian Reich. Simulating hamiltonian dynamics, volume 14. Cambridge university press, 2004.  
Kevin P. Murphy. Machine Learning: A Probabilistic Perspective. The MIT Press, 2012. ISBN 0262018020, 9780262018029.  
Radford M. Neal. Markov chain sampling methods for dirichlet process mixture models. Journal of Computational and Graphical Statistics, 9(2):249-265, 2000.  
Radford M. Neal. MCMC using Hamiltonian Dynamics. 2010.  
Radford M Neal. Bayesian learning for neural networks, volume 118. Springer Science & Business Media, 2012.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational Inference with Normalizing Flows. In Proceedings of the 32Nd International Conference on International Conference on Machine Learning - Volume 37, ICML'15, pp. 1530-1538, 2015.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082, 2014.  
Christian P. Robert and George Casella. Monte Carlo Statistical Methods (Springer Texts in Statistics). Springer-Verlag New York, Inc., Secaucus, NJ, USA, 2005. ISBN 0387212396.  
Ruslan Salakhutdinov and Hugo Larochelle. Efficient learning of deep boltzmann machines. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 693-700, 2010.  
Tim Salimans, Diederik Kingma, and Max Welling. Markov Chain Monte Carlo and Variational Inference: Bridging the Gap. In International Conference on Machine Learning, pp. 1218-1226, 2015.  
Jost Tobias Springenberg, Aaron Klein, Stefan Falkner, and Frank Hutter. Bayesian Optimization with Robust Bayesian Neural Networks. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems 29, pp. 4134-4142. Curran Associates, Inc., 2016. URL http://papers.nips.cc/paper/6117-bayesian-optimization-with-robust-bayesian-neural-networks.pdf.

Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, and Grosse Roger. On the Quantitative Analysis of Deep Belief Networks analysis of Decoder-based Generative Models. In ICLR. 2017.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.  
Yichuan Zhang and Charles Sutton. Semi-separable hamiltonian monte carlo for inference in bayesian hierarchical models. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 27. Curran Associates, Inc., 2014.
