# ARM: AUGMENT-REINFORCE-MERGE GRADIENT FOR STOCHASTIC BINARY NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

To backpropagate the gradients through stochastic binary layers, we propose the augment-REINFORCE-merge (ARM) estimator that is unbiased and has low variance. Exploiting data augmentation, REINFORCE, and reparameterization, the ARM estimator achieves adaptive variance reduction for Monte Carlo integration by merging two expectations via common random numbers. The variance-reduction mechanism of the ARM estimator can also be attributed to antithetic sampling in an augmented space. Experimental results show the ARM estimator provides state-of-the-art performance in auto-encoding variational Bayes and maximum likelihood inference, for discrete latent variable models with one or multiple stochastic binary layers. Python code is available at https://github.com/ABC-anonymous-1.

# 1 INTRODUCTION

Given a function  $f(z)$  of a random variable  $z = (z_{1},\ldots ,z_{V})^{T}$ , which follows a distribution  $q_{\phi}(z)$  parameterized by  $\phi$ , there has been significant recent interest in estimating  $\phi$  to maximize (or minimize) the expectation of  $f(z)$  with respect to  $z\sim q_{\phi}(z)$ , expressed as

$$
\mathcal {E} (\phi) = \int f (z) q _ {\phi} (z) d z = \mathbb {E} _ {z \sim q _ {\phi} (z)} [ f (z) ]. \tag {1}
$$

In particular, maximizing the marginal likelihood of a hierarchical Bayesian model (Bishop, 1995) and maximizing the evidence lower bound (ELBO) for variational inference (Jordan et al., 1999; Blei et al., 2017), two fundamental problems in statistical inference, both boil down to maximizing an expectation as in (1). To maximize (1), if  $\nabla_{z}f(z)$  is tractable to compute and  $z\sim q_{\phi}(z)$  can be generated via reparameterization as  $z = \mathcal{T}_{\phi}(\epsilon)$ ,  $\epsilon \sim p(\epsilon)$ , where  $\epsilon$  are random noises and  $\mathcal{T}_{\phi}(\cdot)$  denotes a deterministic transform parameterized by  $\phi$ , then one may apply the reparameterization trick (Kingma & Welling, 2013; Rezende et al., 2014) to compute the gradient as

$$
\nabla_ {\phi} \mathcal {E} (\phi) = \nabla_ {\phi} \mathbb {E} _ {\epsilon \sim p (\epsilon)} [ f (\mathcal {T} _ {\phi} (\epsilon)) ] = \mathbb {E} _ {\epsilon \sim p (\epsilon)} [ \nabla_ {\phi} f (\mathcal {T} _ {\phi} (\epsilon)) ]. \tag {2}
$$

Unfortunately, this trick is often not applicable to discrete random variables, which are widely used to construct discrete latent variable models such as the sigmoid belief net (Neal, 1992; Saul et al., 1996).

To maximize (1) for discrete  $z$ , using the score function  $\nabla_{\phi}\log q_{\phi}(z) = \nabla_{\phi}q_{\phi}(z) / q_{\phi}(z)$ , one may compute  $\nabla_{\phi}\mathcal{E}(\phi)$  via REINFORCE (Williams, 1992) as

$$
\nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {\boldsymbol {z} \sim q _ {\phi} (\boldsymbol {z})} [ f (\boldsymbol {z}) \nabla_ {\phi} \log q _ {\phi} (\boldsymbol {z}) ] \approx \frac {1}{K} \sum_ {k = 1} ^ {K} f (\boldsymbol {z} ^ {(k)}) \nabla_ {\phi} \log q _ {\phi} (\boldsymbol {z} ^ {(k)}),
$$

where  $z^{(k)} \stackrel{iid}{\sim} q_{\phi}(z)$  are independent, and identically distributed (iid). This unbiased estimator is also known as (a.k.a.) the score-function (Fu, 2006) or likelihood-ratio estimator (Glynn, 1990). While it is unbiased and only requires drawing iid random samples from  $q_{\phi}(z)$  and computing  $\nabla_{\phi} \log q_{\phi}(z^{(k)})$ , its high Monte-Carlo-integration variance often limits its use in practice. Note that if  $f(z)$  depends on  $\phi$ , then we assume it is always true that  $\mathbb{E}_{z \sim q_{\phi}(z)}[\nabla_{\phi} f(z)] = 0$ . For example, in variational inference, we need to maximize the ELBO as  $\mathbb{E}_{z \sim q_{\phi}(z)}[f(z)]$ , where  $f(z) = \log[p(x|z)p(z)/q_{\phi}(z)]$ . In this case, although  $f(z)$  depends on  $\phi$ , since  $\mathbb{E}_{z \sim q_{\phi}(z)}[\nabla_{\phi} \log q_{\phi}(z)] = \int \nabla_{\phi} q_{\phi}(z) dz = \nabla_{\phi} \int q_{\phi}(z) dz = 0$ , we have  $\mathbb{E}_{z \sim q_{\phi}(z)}[\nabla_{\phi} f(z)] = 0$ .

To address the high-variance issue, one may introduce appropriate control variates (a.k.a. baselines) to reduce the variance of REINFORCE (Paisley et al., 2012; Ranganath et al., 2014; Mnih & Gregor,

2014; Gu et al., 2016; Mnih & Rezende, 2016; Ruiz et al., 2016; Kucukelbir et al., 2017; Naesseth et al., 2017). Alternatively, one may first relax the discrete random variables with continuous ones and then apply the reparameterization trick to estimate the gradients, which reduces the variance of Monte Carlo integration at the expense of introducing bias (Maddison et al., 2017; Jang et al., 2017). Combining both REINFORCE and the continuous relaxation of discrete random variables, Tucker et al. (2017) and Grathwohl et al. (2018) both aim to produce a low-variance and unbiased gradient estimator by introducing a continuous relaxation based control variate, whose parameter, however, needs to be estimated at each mini-batch by minimizing the sample variance of the estimator with stochastic gradient descent (SGD), increasing not only the computational complexity, but also the risk of overfitting the training data. Another interesting variance-control idea applicable to discrete latent variables is using local expectation gradients, which estimates the gradients based on REINFORCE, by performing Monte Carlo integration using a single global sample together with exact integration of the local variable for each latent dimension (Titsias & Lázaro-Gredilla, 2015).

Distinct from the usual idea of introducing control variates to reduce the estimation variance of REINFORCE, we propose the augment-REINFORCE-merge (ARM) estimator, a novel unbiased and low-variance gradient estimator for binary latent variables. We show by rewriting the expectation with respect to Bernoulli random variables as one with respect to augmented exponential random variables and then expressing the gradient as an expectation via REINFORCE, with the assistance of appropriate reparameterization, one can derive the ARM estimator in the augmented space by using either the strategy of sharing common random numbers between two expectations or the strategy of applying antithetic sampling. Both strategies, as detailedly discussed in Owen (2013), can both be used to explain why the ARM estimator is unbiased and could lead to significant variance reduction.

Our experimental results on both auto-encoding variational Bayes and maximum likelihood inference for discrete latent variable models, with one or multiple discrete stochastic layers, show that the ARM estimator converges fast, has low computational complexity, and provides state-of-the-art out-of-sample prediction performance, suggesting the effectiveness of using the ARM estimator for gradient backpropagation through stochastic binary layers.

# 2 ARM: AUGMENT-REINFORCE-MERGE ESTIMATOR

In this section, we first present the key theorem of the paper, and then provide its derivation. With this theorem, we summarize ARM gradient ascent for multivariate binary latent variables in Algorithm 1, as shown in the Appendix. Let us denote  $\sigma (\phi) = e^{\phi} / (1 + e^{\phi})$  as the sigmoid function and  $\mathbf{1}_{[\cdot ]}$  as an indicator function that equals to one if the argument is true and zero otherwise.

Theorem 1 (ARM). For a vector of  $V$  binary random variables  $\mathbf{z} = (z_{1},\dots ,z_{V})^{T}$ , the gradient of

$$
\mathcal {E} (\phi) = \mathbb {E} _ {\boldsymbol {z} \sim \prod_ {v = 1} ^ {V} \operatorname {B e r n o u l l i} \left(z _ {v}; \sigma \left(\phi_ {v}\right)\right)} [ f (\boldsymbol {z}) ] \tag {3}
$$

with respect to  $\phi = (\phi_1, \ldots, \phi_V)^T$ , the logits of the Bernoulli probability parameters, can be expressed as

$$
\nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {\boldsymbol {u} \sim \prod_ {v = 1} ^ {V} \operatorname {U n i f o r m} \left(u _ {v}; 0, 1\right)} \left[ \left(f \left(\mathbf {1} _ {\left[ \boldsymbol {u} > \sigma (- \phi) \right]}\right) - f \left(\mathbf {1} _ {\left[ \boldsymbol {u} <   \sigma (\phi) \right]}\right)\right) (\boldsymbol {u} - 1 / 2) \right], \tag {4}
$$

where  $\mathbf{1}_{[u > \sigma (-\phi)]} \coloneqq \left(\mathbf{1}_{[u_1 > \sigma (-\phi_1)]}, \ldots, \mathbf{1}_{[u_V > \sigma (-\phi_V)]}\right)^T$ .

For simplicity, we will first present the ARM estimator for a univariate binary latent variable (i.e.,  $V = 1$ ), and then generalize it to a multivariate one (i.e.,  $V > 1$ ). In the univariate case, we need to evaluate the gradient of  $\mathcal{E}(\phi) = \mathbb{E}_{z\sim \mathrm{Bernoulli}(\sigma (\phi))}[f(z)]$  with respect to  $\phi$ . Let us denote  $t\sim \mathrm{Exp}(\lambda)$  as an exponential distribution, whose probability density function is defined as  $p(t|\lambda) = \lambda e^{-\lambda t}$ , where  $\lambda >0$  and  $t > 0$ . The mean and variance are  $\mathbb{E}[t] = \lambda^{-1}$  and  $\mathrm{var}[t] = \lambda^{-2}$ , respectively. The exponential random variable  $t\sim \mathrm{Exp}(\lambda)$  can be reparameterized as  $t = \epsilon /\lambda$ ,  $\epsilon \sim \mathrm{Exp}(1)$ . It is well known, e.g., in Ross (2006), that if  $t_1\sim \mathrm{Exp}(\lambda_1)$  and  $t_2\sim \mathrm{Exp}(\lambda_2)$  are two independent exponential random variables, then the probability that  $t_1$  is smaller than  $t_2$  can be expressed as  $P(t_{1} < t_{2}) = \lambda_{1} / (\lambda_{1} + \lambda_{2})$ ; moreover, since  $t_1\stackrel {d}{=}\epsilon_1 / \lambda_1$  and  $t_2\stackrel {d}{=}\epsilon_2 / \lambda_2$ , where  $\epsilon_1,\epsilon_2\stackrel {iid}{\sim}\mathrm{Exp}(1)$  and the symbol "  $d$  ", denotes "equal in distribution," we have

$$
P \left(t _ {1} <   t _ {2}\right) = P \left(\epsilon_ {1} / \lambda_ {1} <   \epsilon_ {2} / \lambda_ {2}\right) = P \left(\epsilon_ {1} <   \epsilon_ {2} \lambda_ {1} / \lambda_ {2}\right) = \lambda_ {1} / \left(\lambda_ {1} + \lambda_ {2}\right). \tag {5}
$$

# 2.1 AUGMENTATION OF A BERNOULLI RANDOM VARIABLE AND REPARAMETERIZATION

From (5) it becomes clear that the Bernoulli random variable  $z \sim \mathrm{Bernoulli}(\sigma(\phi))$  can be reparameterized by racing two augmented exponential random variables as

$$
z = \mathbf {1} _ {\left[ \epsilon_ {1} <   \epsilon_ {2} e ^ {\phi} \right]}, \epsilon_ {1} \sim \operatorname {E x p} (1), \epsilon_ {2} \sim \operatorname {E x p} (1). \tag {6}
$$

Consequently, the expectation with respect to the Bernoulli random variable can be reparameterized as one with respect to two augmented exponential random variables as

$$
\mathcal {E} (\phi) = \mathbb {E} _ {z \sim \operatorname {B e r n o u l l i} (\sigma (\phi))} [ f (z) ] = \mathbb {E} _ {\epsilon_ {1}, \epsilon_ {2} \stackrel {i i d} {\sim} \operatorname {E x p} (1)} [ f \left(\mathbf {1} _ {[ \epsilon_ {1} e ^ {- \phi} <   \epsilon_ {2} ]}\right) ]. \tag {7}
$$

# 2.2 REINFORCE ESTIMATOR IN THE AUGMENTED SPACE

Since the indicator function  $\mathbf{1}_{[\epsilon_1e^{-\phi} < \epsilon_2]}$  is not differentiable, the reparameterization trick in (2) is not directly applicable to computing the gradient of (7). Fortunately, as  $t_1 = \epsilon_1e^{-\phi}$ ,  $\epsilon_1 \sim \mathrm{Exp}(1)$  is equal in distribution to  $t_1 \sim \mathrm{Exp}(e^{\phi})$ , the expectation in (7) can be further reparameterized as

$$
\mathcal {E} (\phi) = \mathbb {E} _ {\epsilon_ {1}, \epsilon_ {2} \sim \operatorname {E x p} (1)} \left[ f \left(\mathbf {1} _ {\left[ \epsilon_ {1} e ^ {- \phi} <   \epsilon_ {2} \right]}\right) \right] = \mathbb {E} _ {t _ {1} \sim \operatorname {E x p} \left(e ^ {\phi}\right), \epsilon_ {2} \sim \operatorname {E x p} (1)} \left[ f \left(\mathbf {1} _ {\left[ t _ {1} <   \epsilon_ {2} \right]}\right) \right], \tag {8}
$$

and hence, via REINFORCE and then another reparameterization, we can express the gradient as

$$
\begin{array}{l} \nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {t _ {1} \sim \mathrm {E x p} (e ^ {\phi}), \epsilon_ {2} \sim \mathrm {E x p} (1)} [ f (\mathbf {1} _ {[ t _ {1} <   \epsilon_ {2} ]}) \nabla_ {\phi} \log \mathrm {E x p} (t _ {1}; e ^ {\phi}) ] \\ = \mathbb {E} _ {t _ {1} \sim \mathrm {E x p} (e ^ {\phi}), \epsilon_ {2} \sim \mathrm {E x p} (1)} [ f (\mathbf {1} _ {[ t _ {1} <   \epsilon_ {2} ]}) (1 - t _ {1} e ^ {\phi}) ] \\ = \mathbb {E} _ {\epsilon_ {1}, \epsilon_ {2} \sim \operatorname {E x p} (1)} \left[ f \left(\mathbf {1} _ {\left[ \epsilon_ {1} e ^ {- \phi} <   \epsilon_ {2} \right]}\right) \left(1 - \epsilon_ {1}\right) \right]. \tag {9} \\ \end{array}
$$

Similarly, we have  $\mathcal{E}(\phi) = \mathbb{E}_{\epsilon_1,\epsilon_2\sim \mathrm{Exp}(1)}[f(\mathbf{1}_{[\epsilon_1 <   \epsilon_2e^{\phi}]})] = \mathbb{E}_{\epsilon_1\sim \mathrm{Exp}(1)},t_2\sim \mathrm{Exp}(-e^{\phi})[f(\mathbf{1}_{[\epsilon_1 <   t_2]})]$ , and hence can also express the gradient as

$$
\begin{array}{l} \nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {\epsilon_ {1} \sim \mathrm {E x p} (1), t _ {2} \sim \mathrm {E x p} (e ^ {- \phi})} [ f (\mathbf {1} _ {[ \epsilon_ {1} <   t _ {2} ]}) \nabla_ {\phi} \log \mathrm {E x p} (t _ {2}; e ^ {- \phi}) ] \\ = - \mathbb {E} _ {\epsilon_ {1} \sim \mathrm {E x p} (1), t _ {2} \sim \mathrm {E x p} (e ^ {- \phi})} [ f (\mathbf {1} _ {[ \epsilon_ {1} <   t _ {2} ]}) (1 - t _ {2} e ^ {- \phi}) ] \\ = - \mathbb {E} _ {\epsilon_ {1}, \epsilon_ {2} \sim \operatorname {E x p} (1)} \left[ f \left(\mathbf {1} _ {[ \epsilon_ {1} e ^ {- \phi} <   \epsilon_ {2} ]}\right) \left(1 - \epsilon_ {2}\right) \right]. \tag {10} \\ \end{array}
$$

# 2.3 MERGE OF REINFORCE GRADIENTS

A key observation of the paper is that by swapping the indices of the two  $iid$  standard exponential random variables in (10), the gradient  $\nabla_{\phi}\mathcal{E}(\phi)$  can be equivalently expressed as

$$
\nabla_ {\phi} \mathcal {E} (\phi) = - \mathbb {E} _ {\epsilon_ {1}, \epsilon_ {2} \sim \operatorname {E x p} (1)} \left[ f \left(\mathbf {1} _ {\left[ \epsilon_ {2} e ^ {- \phi} <   \epsilon_ {1} \right]}\right) \left(1 - \epsilon_ {1}\right) \right]. \tag {11}
$$

As the term inside the expectation in (9) and that in (11) could be highly positively correlated, we are motivated to merge (9) and (11) by sharing the same set of standard exponential random variables for Monte Carlo integration, which provides a new opportunity to well control the estimation variance (Owen, 2013). More specifically, simply taking the average of (9) and (11) leads to

$$
\nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {\epsilon_ {1}, \epsilon_ {2} \stackrel {i i d} {\sim} \operatorname {E x p} (1)} \left[ \left(f \left(\mathbf {1} _ {[ \epsilon_ {1} e ^ {- \phi} <   \epsilon_ {2} ]}\right) - f \left(\mathbf {1} _ {[ \epsilon_ {2} e ^ {- \phi} <   \epsilon_ {1} ]}\right)\right) \left(1 / 2 - \epsilon_ {1} / 2\right) \right]. \tag {12}
$$

Note one may also take a weighted average of (9) and (11), and optimize the combination weight to potentially further reduce the variance of the estimator. We leave that for future study.

Note that letting  $\epsilon_1, \epsilon_2 \stackrel{iid}{\sim} \mathrm{Exp}(1)$  is the same in distribution as letting

$$
\epsilon_ {1} = \epsilon u, \epsilon_ {2} = \epsilon (1 - u), \text {w h e r e} u \sim \operatorname {U n i f o r m} (0, 1), \epsilon \sim \operatorname {G a m m a} (2, 1), \tag {13}
$$

which can be proved using  $\mathrm{Exp}(1) \stackrel{d}{=} \mathrm{Gamma}(1, 1)$ ,  $(u, 1 - u)^T \stackrel{d}{=} \mathrm{Dirichlet}(\mathbf{1}_2)$ , where  $u \sim \mathrm{Uniform}(0, 1)$ , and Lemma IV.3 of Zhou & Carin (2012). Thus, (12) can be reparameterized as

$$
\nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {u \sim \operatorname {U n i f o r m} (0, 1), \epsilon \sim \operatorname {G a m m a} (2, 1)} \left[ \big (f (\mathbf {1} _ {[ u > \sigma (- \phi) ]}) - f (\mathbf {1} _ {[ u <   \sigma (\phi) ]}) \big) (\epsilon u / 2 - 1 / 2) \right],
$$

Applying Rao Blackwellization (Casella & Robert, 1996), we can further express the gradient as

$$
\nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {u \sim \text {U n i f o r m} (0, 1)} \left[ \left(f \left(\mathbf {1} _ {[ u > \sigma (- \phi) ]}\right) - f \left(\mathbf {1} _ {[ u <   \sigma (\phi) ]}\right)\right) (u - 1 / 2) \right], \tag {14}
$$

which will be referred to as the Augment-REINFORCE-merge (ARM) estimator.

# 2.4 RELATIONSHIP TO ANTITHETIC SAMPLING

While we use augmentation, REINFORCE, and merge steps to obtain (12) and another reparameterization and marginalizing step to obtain the ARM gradient in (14), we find it can also be obtained by performing augmentation, REINFORCE, reparameterization, and antithetic sampling steps. More specifically, using the equivalence between  $\epsilon_1, \epsilon_2 \stackrel{iid}{\sim} \mathrm{Exp}(1)$  and (13), we can reparameterize (9) as

$$
\begin{array}{l} \nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {u \sim \text {U n i f o r m} (0, 1), \epsilon \sim \text {G a m m a} (2, 1)} [ f (\mathbf {1} _ {[ u <   \sigma (\phi) ]}) (1 - \epsilon u) ] \\ = \mathbb {E} _ {u \sim \operatorname {U n i f o r m} (0, 1)} [ f (\mathbf {1} _ {[ u <   \sigma (\phi) ]}) (1 - 2 u) ]. \tag {15} \\ \end{array}
$$

Further using antithetic sampling (Owen, 2013) with  $\tilde{u} = 1 - u$ , we have

$$
\begin{array}{l} \nabla_ {\phi} \mathcal {E} (\phi) = \mathbb {E} _ {u \sim \text {U n i f o r m} (0, 1)} [ f (\mathbf {1} _ {[ u <   \sigma (\phi) ]}) (1 / 2 - u) ] + \mathbb {E} _ {\tilde {u} \sim \text {U n i f o r m} (0, 1)} [ f (\mathbf {1} _ {[ \tilde {u} <   \sigma (\phi) ]}) (1 / 2 - \tilde {u}) ] (16) \\ = \mathbb {E} _ {u \sim \operatorname {U n i f o r m} (0, 1)} \left[ f \left(\mathbf {1} _ {[ u <   \sigma (\phi) ]}\right) (1 / 2 - u) + f \left(\mathbf {1} _ {[ \tilde {u} <   \sigma (\phi) ]}\right) (1 / 2 - \tilde {u}) \right] (17) \\ \end{array}
$$

which becomes the same as the ARM estimator in (14).

# 2.5 MULTIVARIATE GENERALIZATION

The ARM estimator for univariate binary, however, is of little use in practice, as one may first analytically solve the expectation and then compute its gradient. Below we show how to generalize the univariate ARM estimator to a multivariate one. Let us denote  $(\cdot)_{\backslash v}$  as a vector whose  $v$ th element is removed. For the expectation in (3), applying the univariate ARM estimator in (14), we have

$$
\begin{array}{l} \nabla_ {\phi_ {v}} \mathcal {E} (\phi) = \mathbb {E} _ {\boldsymbol {z} \backslash v \sim \prod_ {\nu \neq v} \operatorname {B e r n o u l l i} \left(z _ {\nu}; \sigma \left(\phi_ {\nu}\right)\right)} \left\{\nabla_ {\phi_ {v}} \mathbb {E} _ {z _ {v} \sim \operatorname {B e r n o u l l i} \left(\sigma \left(\phi_ {v}\right) \right.} [ f (z) ] \right\} \\ = \mathbb {E} _ {\boldsymbol {z} \backslash v} \sim \prod_ {\nu \neq v} \operatorname {B e r n o u l l i} \left(z _ {\nu}; \sigma \left(\phi_ {\nu}\right)\right) \left\{\mathbb {E} _ {u _ {v} \sim \text {U n i f o r m} (0, 1)} \left[ \left(u _ {v} - 1 / 2\right) \right. \right. \\ \left. \times \left(f \left(\boldsymbol {z} _ {\backslash v}, z _ {v} = \mathbf {1} _ {[ u _ {v} > \sigma (- \phi_ {v}) ]}\right) - f \left(\boldsymbol {z} _ {\backslash v}, z _ {v} = \mathbf {1} _ {[ u _ {v} <   \sigma (\phi_ {v}) ]}\right)\right) \right] \}. \tag {18} \\ \end{array}
$$

Since  $z_{\backslash v} \sim \prod_{\nu \neq v} \operatorname{Bernoulli}(z_{\nu}; \sigma(\phi_{\nu}))$  can be equivalently generated as  $z_{\backslash v} = \mathbf{1}_{[u_{\backslash v} < \sigma(\phi_{\backslash v})]}$  or as  $z_{\backslash v} = \mathbf{1}_{[u_{\backslash v} > \sigma(-\phi_{\backslash v})]}$ , where  $u_{\backslash v} \sim \prod_{\nu \neq v} \operatorname{Uniform}(u_{\nu}; 0, 1)$ , exchanging the order of the two expectations in (18) and applying reparameterization, we have

$$
\begin{array}{l} \nabla_ {\phi_ {v}} \mathcal {E} (\phi) = \mathbb {E} _ {u _ {v} \sim \text {U n i f o r m} (0, 1)} \left\{\left(u _ {v} - 1 / 2\right) \mathbb {E} _ {\mathbf {z} _ {\backslash v} \sim \prod_ {\nu \neq v} \operatorname {B e r n o u l l i} \left(z _ {\nu}; \sigma \left(\phi_ {\nu}\right)\right)} \right. \Big [ \\ \left. f \left(\boldsymbol {z} _ {\backslash v}, z _ {v} = \mathbf {1} _ {[ u _ {v} > \sigma (- \phi_ {v}) ]}\right) - f \left(\boldsymbol {z} _ {\backslash v}, z _ {v} = \mathbf {1} _ {[ u _ {v} <   \sigma (\phi_ {v}) ]}\right) \right] \rbrace \\ = \mathbb {E} _ {\boldsymbol {u} \sim \prod_ {v = 1} ^ {V} \text {U n i f o r m} (u _ {v}; 0, 1)} \left[ (u _ {v} - 1 / 2) \left(f \left(\mathbf {1} _ {[ \boldsymbol {u} > \sigma (- \phi) ]}\right) - f (\mathbf {1} _ {[ \boldsymbol {u} <   \sigma (\phi) ]})\right) \right], \tag {19} \\ \end{array}
$$

which concludes the proof for (4) shown in Theorem 1.

Alternatively, instead of generalizing the univariate ARM gradient as in (18) and (19), we can first do a multivariate generalization of the univariate Augment-REINFORCE gradient in (15) as

$$
\begin{array}{l} \nabla_ {\phi_ {v}} \mathcal {E} (\phi) = \mathbb {E} _ {\boldsymbol {z} \setminus v \sim \prod_ {\nu \neq v} \operatorname {B e r n o u l l i} \left(z _ {\nu}; \sigma \left(\phi_ {\nu}\right)\right)} \left\{\nabla_ {\phi_ {v}} \mathbb {E} _ {z _ {v} \sim \operatorname {B e r n o u l l i} \left(\sigma \left(\phi_ {v}\right)\right)} [ f (\boldsymbol {z}) ] \right\} \\ = \mathbb {E} _ {\boldsymbol {z} _ {\backslash v} \sim \prod_ {\nu \neq v} \operatorname {B e r n o u l l i} \left(z _ {\nu}; \sigma \left(\phi_ {\nu}\right)\right)} \left\{\mathbb {E} _ {u _ {v} \sim \operatorname {U n i f o r m} (0, 1)} \left[ \left(1 - 2 u _ {v}\right) f \left(\boldsymbol {z} _ {\backslash v}, z _ {v} = \mathbf {1} _ {\left[ u _ {v} <   \sigma \left(\phi_ {v}\right) \right]}\right) \right] \right\} \\ = \mathbb {E} _ {\boldsymbol {u} \sim \prod_ {v = 1} ^ {V} \operatorname {U n i f o r m} (u _ {v}; 0, 1)} \left[ (1 - 2 u _ {v}) f \left(\mathbf {1} _ {[ \boldsymbol {u} <   \sigma (\phi) ]}\right) \right], \tag {20} \\ \end{array}
$$

and then add an antithetic sampling step to arrive at (4).

# 2.6 EFFECTIVENESS OF ARM IN VARIANCE REDUCTION

Let us denote  $g_v(\pmb{u}) = f(\mathbf{1}_{[\pmb{u} < \sigma(\phi)]})(1 - 2u_v)$  and  $\tilde{\pmb{u}} = 1 - \pmb{u}$ . With  $\pmb{u}^{(k)} \stackrel{i\,id}{\sim} \prod_{v=1}^{V} \mathrm{Uniform}(0,1)$ , we define the ARM estimate of  $\nabla_{\phi_v}\mathcal{E}(\phi)$  with  $K$  Monte Carlo samples, denoted as  $g_{\mathrm{ARM},v}$ , and the augment-REINFORCE (AR) estimate with  $2K$  Monte Carlo samples, denoted as  $g_{\mathrm{AR},v}$ , using

$$
g _ {\mathrm {A R M}, v} = \frac {1}{2 K} \sum_ {k = 1} ^ {K} \left(g _ {v} \left(\boldsymbol {u} ^ {(k)}\right) + g _ {v} \left(\tilde {\boldsymbol {u}} ^ {(k)}\right)\right), \quad g _ {\mathrm {A R}, v} = \frac {1}{2 K} \sum_ {k = 1} ^ {2 K} g _ {v} \left(\boldsymbol {u} ^ {(k)}\right).
$$

Similar to the analysis in Owen (2013), the amount of variance reduction brought by the ARM estimator can be reflected by the ratio of the variance of  $g_{\mathrm{ARM},v}$  to that of  $g_{\mathrm{AR},v}$  as

$$
\frac {\operatorname {v a r} \left[ g _ {\mathrm {A R M} , v} \right]}{\operatorname {v a r} \left[ g _ {\mathrm {A R} , v} \right]} = \frac {\operatorname {v a r} \left[ g _ {v} (\boldsymbol {u}) \right] - \operatorname {c o v} \left(- g _ {v} (\boldsymbol {u}) , g _ {v} (\tilde {\boldsymbol {u}})\right)}{\operatorname {v a r} \left[ g _ {v} (\boldsymbol {u}) \right]} = 1 - \rho_ {v}, \rho_ {v} = \operatorname {C o r r} \left(- g _ {v} (\boldsymbol {u}), g _ {v} (1 - \boldsymbol {u})\right).
$$

Note  $-g_{v}(\pmb{u}) = f(\mathbf{1}_{[\pmb{u} < \sigma(\phi)]})(2u_{v} - 1)$ ,  $g_{v}(1 - \pmb{u}) = f(\mathbf{1}_{[\pmb{u} > \sigma(-\phi)]})(2u_{v} - 1)$ , and  $P(\mathbf{1}_{[u_v < \sigma(\phi_v)]} = \mathbf{1}_{[u_v > \sigma(-\phi_v)]}) = \sigma(|\phi_v|) - \sigma(-|\phi_v|)$ , thus a strong positive correlation (i.e.,  $\rho_v \to 1$ ) and hence noticeable variance reduction is likely especially if  $\phi_v$  moves far away from zero during training.

# 3 BACKPROPAGATION THROUGH DISCRETE STOCHASTIC LAYERS

A latent variable model with multiple stochastic hidden layers can be constructed as

$$
\boldsymbol {x} \sim p _ {\boldsymbol {\theta} _ {0}} (\boldsymbol {x} \mid \boldsymbol {b} _ {1}), \boldsymbol {b} _ {1} \sim p _ {\boldsymbol {\theta} _ {1}} (\boldsymbol {b} _ {1} \mid \boldsymbol {b} _ {2}), \dots , \boldsymbol {b} _ {t} \sim p _ {\boldsymbol {\theta} _ {t}} (\boldsymbol {b} _ {t} \mid \boldsymbol {b} _ {t + 1}), \dots , \boldsymbol {b} _ {T} \sim p _ {\boldsymbol {\theta} _ {T}} (\boldsymbol {b} _ {T}), \tag {21}
$$

whose joint likelihood given the distribution parameters  $\pmb{\theta}_{0:T} = \{\pmb{\theta}_0, \dots, \pmb{\theta}_T\}$  is expressed as

$$
p \left(\boldsymbol {x}, \boldsymbol {b} _ {1: T} \mid \boldsymbol {\theta} _ {0: T}\right) = p _ {\boldsymbol {\theta} _ {0}} \left(\boldsymbol {x} \mid \boldsymbol {b} _ {1}\right) \left[ \prod_ {t = 1} ^ {T - 1} p _ {\boldsymbol {\theta} _ {t}} \left(\boldsymbol {b} _ {t} \mid \boldsymbol {b} _ {t + 1}\right) \right] p _ {\boldsymbol {\theta} _ {T}} \left(\boldsymbol {b} _ {T}\right). \tag {22}
$$

In comparison to deterministic feedforward neural networks, stochastic ones can represent complex distributions and show natural resistance to overfitting (Neal, 1992; Saul et al., 1996; Tang & Salakhutdinov, 2013; Raiko et al., 2014; Gu et al., 2016; Tang & Salakhutdinov, 2013). However, the training, especially if there are stochastic discrete layers, is often much more challenging. Below we show for both auto-encoding variational Bayes and maximum likelihood inference, how to apply the ARM estimator for gradient backpropagation in stochastic binary networks.

# 3.1 ARM VARIATIONAL AUTO-ENCODER

For auto-encoding variational Bayes inference (Kingma & Welling, 2013; Rezende et al., 2014), we construct a variational distribution as

$$
\left. q _ {\boldsymbol {w} _ {1: T}} \left(\boldsymbol {b} _ {1: T} \mid \boldsymbol {x}\right) = q _ {\boldsymbol {w} _ {1}} \left(\boldsymbol {b} _ {1} \mid \boldsymbol {x}\right) \left[ \prod_ {t = 1} ^ {T - 1} q _ {\boldsymbol {w} _ {t + 1}} \left(\boldsymbol {b} _ {t + 1} \mid \boldsymbol {b} _ {t}\right) \right], \right. \tag {23}
$$

with which the ELBO can be expressed as

$$
\mathcal {E} (\boldsymbol {w} _ {1: T}) = \mathbb {E} _ {\boldsymbol {b} _ {1: T} \sim q _ {\boldsymbol {w} _ {1: T}} (\boldsymbol {b} _ {1: T} \mid \boldsymbol {x})} [ f (\boldsymbol {b} _ {1: T}) ], \text {w h e r e}
$$

$$
f \left(\boldsymbol {b} _ {1: T}\right) = \log p _ {\boldsymbol {\theta} _ {0}} (\boldsymbol {x} \mid \boldsymbol {b} _ {1}) + \log p _ {\boldsymbol {\theta} _ {1: T}} \left(\boldsymbol {b} _ {1: T}\right) - \log q _ {\boldsymbol {w} _ {1: T}} \left(\boldsymbol {b} _ {1: T} \mid \boldsymbol {x}\right). \tag {24}
$$

Proposition 2 (ARM backpropagation). For a stochastic binary network with  $T$  binary stochastic hidden layers, constructing a variational auto-encoder (VAE) defined with  $\mathbf{b}_0 = \mathbf{x}$  and

$$
\left. q _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t} \mid \boldsymbol {b} _ {t - 1}\right) = \operatorname {B e r n o u l l i} \left(\boldsymbol {b} _ {t}; \sigma \left(\mathcal {T} _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right)\right)\right) \right. \tag {25}
$$

for  $t = 1,\dots ,T$  , the gradient of the ELBO with respect to  $\pmb{w}_t$  can be expressed as

$$
\begin{array}{l} \nabla_ {\boldsymbol {w} _ {t}} \mathcal {E} (\boldsymbol {w} _ {1: T}) = \mathbb {E} _ {q \left(\boldsymbol {b} _ {1: t - 1}\right)} \left[ \mathbb {E} _ {\boldsymbol {u} _ {t} \sim \text {U n i f o r m} (0, 1)} \left[ f _ {\Delta} \left(\boldsymbol {u} _ {t}, \mathcal {T} _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right), \boldsymbol {b} _ {1: t - 1}\right) \left(\boldsymbol {u} _ {t} - 1 / 2\right) \right] \nabla_ {\boldsymbol {w} _ {t}} \mathcal {T} _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right) \right], \\ w h e r e f _ {\Delta} \left(\boldsymbol {u} _ {t}, \mathcal {T} _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right), \boldsymbol {b} _ {1: t - 1}\right) = \mathbb {E} _ \boldsymbol {b} _ {t + 1: T} \sim q \left(\boldsymbol {b} _ {t + 1: T} \mid \boldsymbol {b} _ {t}\right), \boldsymbol {b} _ {t} = \mathbf {1} _ {\left\{\boldsymbol {u} _ {t} > \sigma \left(- \mathcal {T} _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right)\right)\right\}} \left. \right] [ f (\boldsymbol {b} _ {1: T}) ] \\ - \mathbb {E} _ \boldsymbol {b} _ {t + 1: T} \sim q \left(\boldsymbol {b} _ {t + 1: T} \mid \boldsymbol {b} _ {t}\right), \boldsymbol {b} _ {t} = \mathbf {1} _ {\left[ \boldsymbol {u} _ {t} <   \sigma \left(\tau_ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right)\right) \right]} [ f (\boldsymbol {b} _ {1: T}) ] \tag {26} \\ \end{array}
$$

The gradient presented in (26) can be estimated with a single Monte Carlo sample as

$$
\hat {f} _ {\Delta} \left(\boldsymbol {u} _ {t}, \mathcal {T} _ {\boldsymbol {w} _ {t}} \left(\boldsymbol {b} _ {t - 1}\right), \boldsymbol {b} _ {1: t - 1}\right) = \left\{ \begin{array}{l l} 0, & \text {i f} \boldsymbol {b} _ {t} ^ {(1)} = \boldsymbol {b} _ {t} ^ {(2)} \\ f \left(\boldsymbol {b} _ {1: t - 1}, \boldsymbol {b} _ {t: T} ^ {(1)}\right) - f \left(\boldsymbol {b} _ {1: t - 1}, \boldsymbol {b} _ {t: T} ^ {(2)}\right), & \text {o t h e r w i s e} \end{array} , \right. \tag {27}
$$

where  $\pmb{b}_{t}^{(1)} = \mathbf{1}_{[\pmb{u}_{t} > \sigma(-\mathcal{T}_{\pmb{w}_{t}}(\pmb{b}_{t-1}))]}, \pmb{b}_{t+1:T}^{(1)} \sim q(\pmb{b}_{t+1:T} | \pmb{b}_{t}^{(1)})$ ,  $\pmb{b}_{t}^{(2)} = \mathbf{1}_{[\pmb{u}_{t} < \sigma(\mathcal{T}_{\pmb{w}_{t}}(\pmb{b}_{t-1}))]}$ , and  $\pmb{b}_{t+1:T}^{(2)} \sim q(\pmb{b}_{t+1:T} | \pmb{b}_{t}^{(2)})$ . The proof of Proposition 2 is provided in the Appendix.

# 3.2 ARM MAXIMUM LIKELIHOOD INFERENCE

For maximum likelihood inference, the log marginal likelihood can be expressed as

$$
\begin{array}{l} \log p _ {\boldsymbol {\theta} _ {0: T}} (\boldsymbol {x}) = \log \mathbb {E} _ {\boldsymbol {b} _ {1: T} \sim p _ {\boldsymbol {\theta} _ {1: T}} (\boldsymbol {b} _ {1: T})} [ p _ {\boldsymbol {\theta} _ {0}} (\boldsymbol {x} \mid \boldsymbol {b} _ {1}) ] \\ \geq \mathcal {E} \left(\boldsymbol {\theta} _ {1: T}\right) = \mathbb {E} _ {\boldsymbol {b} _ {1: T} \sim p _ {\boldsymbol {\theta} _ {1: T}} \left(\boldsymbol {b} _ {1: T}\right)} \left[ \log p _ {\boldsymbol {\theta} _ {0}} (\boldsymbol {x} \mid \boldsymbol {b} _ {1}) \right]. \tag {28} \\ \end{array}
$$

Generalizing Proposition 2 leads to the following proposition.

Proposition 3. For a stochastic binary network defined as

$$
p _ {\boldsymbol {\theta} _ {t}} \left(\boldsymbol {b} _ {t} \mid \boldsymbol {b} _ {t + 1}\right) = \operatorname {B e r n o u l l i} \left(\boldsymbol {b} _ {t}; \sigma \left(\mathcal {T} _ {\boldsymbol {\theta} _ {t}} \left(\boldsymbol {b} _ {t + 1}\right)\right)\right), \tag {29}
$$

the gradient of the lower bound in (28) with respect to  $\theta_{t}$  can be expressed as

$$
\nabla_ {\boldsymbol {\theta} _ {t}} \mathcal {E} (\boldsymbol {\theta} _ {1: T}) = \mathbb {E} _ {p (\boldsymbol {b} _ {t + 1: T})} \left[ \mathbb {E} _ {\boldsymbol {u} _ {t} \sim \operatorname {U n i f o r m} (0, 1)} \left[ f _ {\Delta} (\boldsymbol {u} _ {t}, \mathcal {T} _ {\boldsymbol {\theta} _ {t}} (\boldsymbol {b} _ {t + 1}), \boldsymbol {b} _ {t + 1: T}) (\boldsymbol {u} _ {t} - 1 / 2) \right] \nabla_ {\boldsymbol {\theta} _ {t}} \mathcal {T} _ {\boldsymbol {\theta} _ {t}} (\boldsymbol {b} _ {t + 1}) \right],
$$

$$
\begin{array}{l} w h e r e f _ {\Delta} \left(\boldsymbol {u} _ {t}, \mathcal {T} _ {\boldsymbol {\theta} _ {t}} \left(\boldsymbol {b} _ {t + 1}\right), \boldsymbol {b} _ {t + 1: T}\right) = \mathbb {E} _ \boldsymbol {b} _ {1: t - 1} \sim p \left(\boldsymbol {b} _ {1: t - 1} \mid \boldsymbol {b} _ {t}\right), \boldsymbol {b} _ {t} = \mathbf {1} _ {\left[ \boldsymbol {u} _ {t} > \sigma \left(- \mathcal {T} _ {\boldsymbol {\theta} _ {t}} \left(\boldsymbol {b} _ {t + 1}\right)\right)\right]}\left. \right) [ \log p _ {\boldsymbol {\theta} _ {0}} (\boldsymbol {x} \mid \boldsymbol {b} _ {1}) ] \\ - \mathbb {E} _ \boldsymbol {b} _ {1: t - 1} \sim p (\boldsymbol {b} _ {1: t - 1} \mid \boldsymbol {b} _ {t}), \boldsymbol {b} _ {t} = \mathbf {1} _ {[ \boldsymbol {u} _ {t} <   \sigma (\mathcal {T} _ {\boldsymbol {\theta} _ {t}} (\boldsymbol {b} _ {t + 1})) ]}) [ \log p _ {\boldsymbol {\theta} _ {0}} (\boldsymbol {x} \mid \boldsymbol {b} _ {1}) ]. \\ \end{array}
$$

![](images/d1c9759dfaf6f3358e82850268db5f562b339af1463dbb388018333711eb040e.jpg)  
Figure 1: Left: Trace plots of the true/estimated gradients and estimated Bernoulli probability parameters for  $p_0 \in \{0.49, 0.499, 0.501, 0.51\}$ ; Right: Trace plots of the loss functions for  $p_0 = 0.499$ .

![](images/b1d9d5b99e4074cf5ba1653652dba8763c73a3616cf297b67b8863ad499b1341.jpg)

# 4 EXPERIMENTAL RESULTS

To illustrate the working mechanism of the ARM estimator, related to Tucker et al. (2017) and Grathwohl et al. (2018), we consider learning  $\phi$  to maximize  $\mathcal{E}(\phi) = \mathbb{E}_{z\sim \mathrm{Bernoulli}(\sigma (\phi))}[(z - p_0)^2 ]$ , where  $p_0\in \{0.49,0.499,0.501,0.51\}$ , or equivalently, minimize the loss as  $-\mathcal{E}(\phi)$ . The optimal solution is  $\sigma (\phi) = 1(p_0 < 0.5)$ . The closer  $p_0$  is to 0.5, the more challenging the optimization becomes. We compare the ARM estimator to the true gradient as  $g_{\phi} = (1 - 2p_{0})\sigma (\phi)(1 - \sigma (\phi))$  and three previously proposed unbiased estimators, including REINFORCE, REBAR (Tucker et al., 2017), and RELAX (Grathwohl et al., 2018). With a single random sample  $u\sim \mathrm{Uniform}(0,1)$  for Monte Carlo integration, the ARM gradient can be expressed as

$$
g _ {\phi , \mathrm {A R M}} = \left[ \left(\mathbf {1} _ {[ u > \sigma (- \phi) ]} - p _ {0}\right) ^ {2} - \left(\mathbf {1} _ {[ u <   \sigma (\phi) ]} - p _ {0}\right) ^ {2} \right] (u - 1 / 2),
$$

while the REINFORCE gradient can be expressed as

$$
g _ {\phi , \text {R E I N F O R C E}} = \left(\mathbf {1} _ {[ u <   \sigma (\phi) ]} - p _ {0}\right) ^ {2} \left(\mathbf {1} _ {[ u <   \sigma (\phi) ]} - \sigma (\phi)\right).
$$

See Tucker et al. (2017) and Grathwohl et al. (2018) for the details about REBAR and RELAX, respectively, which both introduce stochastically estimated control variates to improve REINFORCE.

As shown in Figure 1 (a), the REINFORCE gradients have large variances. Consequently, a REINFORCE based gradient ascent algorithm may diverge. For example, when  $p_0 = 0.501$ , the optimal value for the Bernoulli probability  $\sigma(\phi)$  is 0, but the algorithm infers it to be close to 1 at the end of 3000 iterations of a random trial. By contrast, the univariate ARM estimator well approximates the time-varying true gradients by adjusting the frequencies, amplitudes, and signs of its gradient estimates, with larger and more frequent spikes for larger true gradients. As shown in Figure 1 (b), using the ARM estimator is indistinguishable from using the true gradient for updating  $\phi$  to minimize the loss  $-\mathbb{E}_{z \sim \text{Bernoulli}(\sigma(\phi))}[(z - 0.499)^2]$ , significantly outperforming not only REINFORCE, which has a large variance, but also both REBAR and RELAX, which improve on REINFORCE by introducing carefully constructed control variates that are stochastically updated for variance reduction. We further plot in Figure 3 of the Appendix the gradient estimated with multiple Monte Carlo samples against the true gradient at each iteration, showing the ARM estimator has significant lower variance than REINFORCE given the same number of Monte Carlo samples.

# 4.1 DISCRETE VARIATIONAL AUTO-ENCODERS

To optimize a variational auto-encoder (VAE) for a discrete latent variable model, existing solutions often rely on biased but low-variance stochastic gradient estimators (Bengio et al., 2013; Jang et al., 2017), unbiased but high-variance ones (Mnih & Gregor, 2014), or unbiased REINFORCE combined with computationally expensive control variates, whose parameters are estimated by minimizing the sample variance of the estimator with SGD (Tucker et al., 2017; Grathwohl et al., 2018). Comparing to previously proposed methods for discrete latent variables, the ARM estimator exhibits low variance and is unbiased, computationally efficient, and simple to implement.

For discrete VAEs, we compare ARM with a variety of representative stochastic gradient estimators for discrete latent variables, including Wake-Sleep (Hinton et al., 1995), NVIL (Mnih & Gregor,

Table 1: The constructions of three differently structured discrete variational auto-encoders. The following symbols “ $\rightarrow$ ”, “ $\leftarrow$ ”, “ $\right)$ , and “ $\rightsquigarrow$ ” represent deterministic linear transform, leaky rectified linear units (LeakyReLU) (Maas et al., 2013) nonlinear activation, sigmoid nonlinear activation, and random sampling respectively, in the encoder (a.k.a. recognition network); their reversed versions are used in the decoder (a.k.a. generator).  

<table><tr><td></td><td>Nonlinear</td><td>Linear</td><td>Linear two layers</td></tr><tr><td>Encoder</td><td>784→200]→200]→200)~~200</td><td>784→200)~~200</td><td>784→200)~~200→200)~~200</td></tr><tr><td>Decoder</td><td>784~(784←[200←[200←200</td><td>784~(784←[200←[200←200</td><td>784~(784←[200←[200←200</td></tr></table>

![](images/0eee9941c30830d46a132a9f726a80630b00c3291eca6b0e925b9bade4e6a5ad.jpg)  
(a) Nonlinear

![](images/8f4d2df6c2302151e218708b95d0b2daaa232ae4d174faf4e591eac467c02204.jpg)  
(b) Linear

![](images/5372f072e218392f97c0b1d618f94e5c7607e269a4b697eab645889a872e6f19.jpg)  
(c) Linear two layers

![](images/ae6af4f011b58cdcbd05264cd1224144cd7cd4368bd476af5f51fae80606bb6a.jpg)  
(d) Nonlinear

![](images/c324e3754816cc77a8d16b54e5f3752d8301425b4575dbc5ac330f82877123aa.jpg)  
(e) Linear

![](images/d3d4804f70f8d696e2b4026df7128064b80096ca154ec27c07f4f6e9214c4c63.jpg)  
(f) Linear two layers  
Figure 2: Test negative ELBOs on MNIST-static with respect to training iterations, shown in the top row, and wall clock times on Tesla-K40 GPU, shown in the bottom row, for three differently structured Bernoulli VAEs.

2014), LeGrad (Titsias & Lázaro-Gredilla, 2015), MuProp (Gu et al., 2016), Concrete (Gumbel-Softmax) (Jang et al., 2017; Maddison et al., 2017), REBAR (Grathwohl et al., 2018), and RELAX (Tucker et al., 2017). Following the settings in Tucker et al. (2017) and Grathwohl et al. (2018), for the encoder defined in (22) and decoder defined in (23), we consider three different network architectures, as summarized in Table 1, including "Nonlinear" that has one stochastic but two Leaky-ReLU (Maas et al., 2013) deterministic hidden layers, "Linear" that has one stochastic hidden layer, and "Linear two layers" that has two stochastic hidden layers. We consider a widely used binarization (Salakhutdinov & Murray, 2008; Larochelle & Murray, 2011), referred to as MNIST-static and available at http://www.dmi.usherb.ca/\~larocheh/mlpython/_modules/datasets/binarized_mnist.html, making our numerical results directly comparable to those reported in the literature. In addition to MNIST-static, we also consider MNIST-threshold (van den Oord et al., 2017), which binarizes MNIST by thresholding each pixel value at 0.5, and the binarized OMNIGLOT dataset.

We train discrete VAEs with 200 conditionally  $iid$  Bernoulli random variables as the hidden units of each stochastic binary layer. We maximize a single-Monte-Carlo-sample ELBO using Adam (Kingma & Ba, 2014), with the learning rate selected from  $\{5,1,0.5\} \times 10^{-4}$  by the validation set. We set the batch size as 50 for MNIST and 25 for OMNIGLOT. For each dataset, using its default training/validation/testing partition, we train all methods on the training set, calculate the validation log-likelihood for every epoch, and report the test negative log-likelihood when the validation negative log-likelihood reaches its minimum within a predefined maximum number of iterations.

We summarize the test negative log-likelihoods in Table 2 for MNIST-static. We also summarize the test negative ELBOs in Table 4 of the Appendix, and provide related trace plots of the training and validation negative ELBOs on MNIST-static in Figure 2, and these on MNIST-threshold and OMNIGLOT in Figures 5 and 6 of the Appendix, respectively. For these trace plots, for a fair comparison of convergence speed between different algorithms, we use publicly available code from

Table 2: Test negative log-likelihoods of discrete VAEs trained with a variety of stochastic gradient estimators on MNIST-static and OMNIGLOT, where *,  $\star$ , †, ‡ represent the results reported in Mnih & Gregor (2014), Tucker et al. (2017), Gu et al. (2016), and Grathwohl et al. (2018), respectively. The results for LeGrad (Titsias & Lázaro-Gredilla, 2015) are obtained by running the code provided by the authors. We report the results of ARM using the sample mean and standard deviation over five independent trials with random initializations.  
(a) MNIST  

<table><tr><td colspan="2">Linear</td><td colspan="2">Nonlinear</td><td colspan="2">Two layers</td></tr><tr><td>Algorithm</td><td>- log p(x)</td><td>Algorithm</td><td>- log p(x)</td><td>Algorithm</td><td>- log p(x)</td></tr><tr><td>REINFORCE</td><td>= 164.0</td><td>REINFORCE</td><td>= 114.6</td><td>REINFORCE</td><td>= 159.2</td></tr><tr><td>Wake-Sleep*</td><td>= 120.8</td><td>Wake-Sleep*</td><td>-</td><td>Wake-Sleep*</td><td>= 107.7</td></tr><tr><td>NVIL*</td><td>= 113.1</td><td>NVIL*</td><td>= 102.2</td><td>NVIL*</td><td>= 99.8</td></tr><tr><td>LeGrad</td><td>≤ 117.5</td><td>LeGrad</td><td>-</td><td>LeGrad</td><td>-</td></tr><tr><td>MuProp†</td><td>≤ 113.0</td><td>MuProp*</td><td>= 99.1</td><td>MuProp†</td><td>≤ 100.4</td></tr><tr><td>Concrete*</td><td>= 107.3</td><td>Concrete*</td><td>= 99.6</td><td>Concrete*</td><td>= 95.6</td></tr><tr><td>REBAR*</td><td>= 107.7</td><td>REBAR*</td><td>= 101.4</td><td>REBAR*</td><td>= 95.4</td></tr><tr><td>RELAX‡</td><td>≤ 113.6</td><td>RELAX‡</td><td>≤ 119.2</td><td>RELAX‡</td><td>≤ 100.9</td></tr><tr><td>ARM</td><td>= 107.2 ± 0.1</td><td>ARM</td><td>= 98.4 ± 0.3</td><td>ARM</td><td>= 96.7 ± 0.3</td></tr></table>

(b) OMNIGLOT  

<table><tr><td colspan="2">Linear</td><td colspan="2">Nonlinear</td><td colspan="2">Two layers</td></tr><tr><td>Algorithm</td><td>- log p(x)</td><td>Algorithm</td><td>- log p(x)</td><td>Algorithm</td><td>- log p(x)</td></tr><tr><td>NVIL*</td><td>= 117.6</td><td>NVIL*</td><td>= 116.6</td><td>NVIL*</td><td>= 111.4</td></tr><tr><td>MuProp*</td><td>= 117.6</td><td>MuProp*</td><td>= 117.5</td><td>MuProp*</td><td>= 111.2</td></tr><tr><td>Concrete*</td><td>= 117.7</td><td>Concrete*</td><td>= 116.7</td><td>Concrete*</td><td>= 111.3</td></tr><tr><td>REBAR*</td><td>= 117.7</td><td>REBAR*</td><td>= 118.0</td><td>REBAR*</td><td>= 110.8</td></tr><tr><td>RELAX‡</td><td>≤ 122.1</td><td>RELAX‡</td><td>≤ 128.2</td><td>RELAX‡</td><td>≤ 115.4</td></tr><tr><td>ARM</td><td>= 115.8 ± 0.2</td><td>ARM</td><td>= 117.6 ± 0.4</td><td>ARM</td><td>= 109.8 ± 0.3</td></tr></table>

the authors and setting the learning rate of ARM the same as that selected by REBAR/RELAX in Grathwohl et al. (2018).

These results show that ARM provides state-of-the-art performance in delivering not only fast convergence, but also low negative log-likelihoods and negative ELBOs on both the validation and test sets, with low computational cost, for all three different network architectures. In comparison to the vanilla REINFORCE on MNIST-static, as shown in Table 2 (a), ARM achieves significantly lower test log-likelihoods, which can be explained by having much lower variance in its gradient estimation, while only costing  $20\%$  to  $30\%$  more computation time to finish the same number of iterations.

The trace plots in Figures 2, 5, and 6 show that ARM achieves its objective better or on a par with the state-of-the-art methods in all three different network architectures. In particular, the performance of ARM on MNIST-threshold is significantly better, suggesting ARM is more robust, better resists overfitting, and has better generalization ability. On OMNIGLOT, with "Nonlinear" network architecture, both REBAR and RELAX exhibit severe overfitting, which could be caused by their training procedure that updates the parameters of the control variates, which are designed to minimize the true variance of the gradient estimator, by minimizing the sample variance of the gradient estimator using SGD. For less overfitting linear and two-stochastic-layer networks, ARM overall performs better than both REBAR and RELAX and converges significantly faster (about 6-8 times faster) in terms of computation time.

# 4.2 MAXIMUM LIKELIHOOD INFERENCE FOR A STOCHASTIC BINARY NETWORK

Denoting  $\boldsymbol{x}_l, \boldsymbol{x}_u \in \mathbb{R}^{394}$  as the lower and upper halves of an MNIST digit, respectively, we consider a standard benchmark task of estimating the conditional distribution  $p_{\theta_{0:2}}(\boldsymbol{x}_l \mid \boldsymbol{x}_u)$  (Raiko et al., 2014; Bengio et al., 2013; Gu et al., 2016; Jang et al., 2017; Tucker et al., 2017), using a stochastic binary network with two stochastic binary hidden layers, expressed as

$$
\boldsymbol {x} _ {l} \sim \operatorname {B e r n o u l l i} \left(\sigma \left(\mathcal {T} _ {\boldsymbol {\theta} _ {0}} \left(\boldsymbol {b} _ {1}\right)\right)\right), \boldsymbol {b} _ {1} \sim \operatorname {B e r n o u l l i} \left(\sigma \left(\mathcal {T} _ {\boldsymbol {\theta} _ {1}} \left(\boldsymbol {b} _ {2}\right)\right)\right), \boldsymbol {b} _ {2} \sim \operatorname {B e r n o u l l i} \left(\sigma \left(\mathcal {T} _ {\boldsymbol {\theta} _ {2}} \left(\boldsymbol {x} _ {u}\right)\right)\right). \tag {30}
$$

Table 3: For the MNIST conditional distribution estimation benchmark task, comparison of the test negative log-likelihood between ARM and various gradient estimators in Jang et al. (2017) is reported here.  

<table><tr><td>Gradient estimator</td><td>ARM</td><td>ST</td><td>DARN</td><td>Annealed ST</td><td>ST Gumbel-S.</td><td>SF</td><td>MuProp</td></tr><tr><td>- log p(xl | xu)</td><td>57.9 ± 0.1</td><td>58.9</td><td>59.7</td><td>58.7</td><td>59.3</td><td>72.0</td><td>58.9</td></tr></table>

We set the network structure as 392-200-200-392 which means both  $\pmb{b}_{1}$  and  $\pmb{b}_{2}$  are 200 dimensional binary vectors and the transformation  $\mathcal{T}_{\theta}$  are linear so the results are directly comparable with those in Jang et al. (2017). We approximate  $\log p_{\theta_{0:2}}(\pmb{x}_l \mid \pmb{x}_u)$  with  $\log \frac{1}{K} \sum_{k=1}^{K} \text{Bernoulli}(\pmb{x}_l; \sigma(\mathcal{T}_{\theta_0}(\pmb{b}_1^{(k)})))$ , where  $\pmb{b}_1^{(k)} \sim \text{Bernoulli}(\sigma(\mathcal{T}_{\theta_1}(\pmb{b}_2^{(k)})))$ ,  $\pmb{b}_2^{(k)} \sim \text{Bernoulli}(\sigma(\mathcal{T}_{\theta_2}(\pmb{x}_u)))$ . We perform training with  $K = 1$ , which can also be considered as optimizing on a single-Monte-Carlo-sample estimate of the lower bound of the log likelihood shown in (28). We use Adam (Kingma & Ba, 2014), with the learning rate set as  $10^{-4}$ , mini-batch size as 100, and number of epochs for training as 2000. Given the inferred point estimate of  $\theta_{0:2}$  after training, we evaluate the accuracy of conditional density estimation by estimating the negative log-likelihood as  $-\log p_{\theta_{0:2}}(\pmb{x}_l \mid \pmb{x}_u)$ , averaging over the test set using  $K = 1000$ . We show example results of predicting the activation probabilities of the pixels of  $\pmb{x}_l$  given  $\pmb{x}_u$  in Figure 4 of the Appendix.

As shown in Table 3, optimizing a stochastic binary network with the ARM estimator, which is unbiased and computationally efficient, achieves the lowest test negative log-likelihood, outperforming previously proposed biased stochastic gradient estimators on similarly structured stochastic networks, including DARN (Gregor et al., 2013), straight through (ST) (Bengio et al., 2013), slope-annealed ST (Chung et al., 2016), and ST Gumbel-softmax (Jang et al., 2017), and unbiased ones, including score-function (SF) and MuProp (Gu et al., 2016).

# 5 CONCLUSIONS

To train a discrete latent variable model with one or multiple stochastic binary layers, we propose the augment-REINFORCE-merge (ARM) estimator to provide unbiased and low-variance gradient estimates of the parameters of Bernoulli distributions. With a single Monte Carlo sample, the estimated gradient is the product of uniform random noises and the difference of a function of two vectors of correlated binary latent variables. Without relying on learning control variates for variance reduction, it maintains efficient computation and avoids increasing the risk of overfitting. Applying the ARM gradient leads to not only fast convergence, but also low test negative log-likelihoods (and low test negative evidence lower bounds for variational inference), on both auto-encoding variational Bayes and maximum likelihood inference for stochastic binary feedforward neural networks. Some natural extensions of the proposed ARM estimator include generalizing it to multivariate categorical latent variables, combining it with a control-variate or local-expectation based variance reduction method, and applying it to reinforcement learning whose action space is discrete.

# REFERENCES

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Christopher M Bishop. Neural Networks for Pattern Recognition. Oxford university press, 1995.  
David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518):859-877, 2017.  
George Casella and Christian P Robert. Rao-blackwellisation of sampling schemes. Biometrika, 83 (1):81-94, 1996.  
Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.  
Michael C Fu. Gradient estimation. *Handbooks in operations research and management science*, 13: 575-616, 2006.

Peter W Glynn. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM, 33(10):75-84, 1990.  
Will Grathwohl, Dami Choi, Yuhuai Wu, Geoff Roeder, and David Duvenaud. Backpropagation through the Void: Optimizing control variates for black-box gradient estimation. In ICLR, 2018.  
Karol Gregor, Ivo Danihelka, Andriy Mnih, Charles Blundell, and Daan Wierstra. Deep autoregressive networks. arXiv preprint arXiv:1310.8499, 2013.  
Shixiang Gu, Sergey Levine, Ilya Sutskever, and Andriy Mnih. MuProp: Unbiased backpropagation for stochastic neural networks. In *ICLR*, 2016.  
Geoffrey E Hinton, Peter Dayan, Brendan J Frey, and Radford M Neal. "The" wake-sleep algorithm for unsupervised neural networks. Science, 268(5214):1158-1161, 1995.  
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-softmax. In ICLR, 2017.  
Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. Machine learning, 37(2):183-233, 1999.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.  
Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M Blei. Automatic differentiation variational inference. Journal of Machine Learning Research, 18(14):1-45, 2017.  
Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 29-37, 2011.  
Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In ICML, 2013.  
Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In ICLR, 2017.  
Andriy Mnih and Karol Gregor. Neural variational inference and learning in belief networks. In ICML, pp. 1791-1799, 2014.  
Andriy Mnih and Danilo J Rezende. Variational inference for monte carlo objectives. arXiv preprint arXiv:1602.06725, 2016.  
Christian Naesseth, Francisco Ruiz, Scott Linderman, and David Blei. Reparameterization gradients through acceptance-rejection sampling algorithms. In AISTATS, pp. 489-498, 2017.  
R. M. Neal. Connectionist learning of belief networks. Artificial Intelligence, pp. 71-113, 1992.  
Art B. Owen. Monte Carlo Theory, Methods and Examples, chapter 8 Variance Reduction. 2013.  
John Paisley, David M Blei, and Michael I Jordan. Variational Bayesian inference with stochastic search. In ICML, pp. 1363-1370, 2012.  
Tapani Raiko, Mathias Berglund, Guillaume Alain, and Laurent Dinh. Techniques for learning binary stochastic feedforward neural networks. arXiv preprint arXiv:1406.2989, 2014.  
Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational inference. In AISTATS, pp. 814-822, 2014.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In ICML, pp. 1278-1286, 2014.  
Sheldon M. Ross. Introduction to Probability Models. Academic Press, 10th edition, 2006.

Francisco J. R. Ruiz, Michalis K. Titsias, and David M. Blei. The generalized reparameterization gradient. In NIPS, pp. 460-468, 2016.  
Ruslan Salakhutdinov and Iain Murray. On the quantitative analysis of deep belief networks. In ICML, pp. 872-879, 2008.  
Lawrence K Saul, Tommi Jaakkola, and Michael I Jordan. Mean field theory for sigmoid belief networks. Journal of Artificial Intelligence Research, 4:61-76, 1996.  
Yichuan Tang and Ruslan R Salakhutdinov. Learning stochastic feedforward neural networks. In NIPS, pp. 530-538, 2013.  
Michalis K Titsias and Miguel Lázaro-Gredilla. Local expectation gradients for black box variational inference. In NIPS, pp. 2638-2646. MIT Press, 2015.  
George Tucker, Andriy Mnih, Chris J Maddison, John Lawson, and Jascha Sohl-Dickstein. Rebar: Low-variance, unbiased gradient estimates for discrete latent variable models. In NIPS, pp. 2624-2633, 2017.  
Aaron van den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In Advances in Neural Information Processing Systems, pp. 6306-6315, 2017.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. In Reinforcement Learning, pp. 5-32. Springer, 1992.  
Mingyuan Zhou and Lawrence Carin. Negative binomial process count and mixture modeling. arXiv preprint arXiv:1209.3442v1, 2012.
