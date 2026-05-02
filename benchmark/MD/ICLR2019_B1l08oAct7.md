# FIXING VARIATIONAL BAYES: DETERMINISTIC VARIATIONAL INFERENCE FOR BAYESIAN NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Bayesian neural networks (BNNs) hold great promise as a flexible and principled solution to deal with uncertainty when learning from finite data. Among approaches to realize probabilistic inference in deep neural networks, variational Bayes (VB) is theoretically grounded, generally applicable, and computationally efficient. With wide recognition of potential advantages, why is it that variational Bayes has seen very limited practical use for BNNs in real applications? We argue that variational inference in neural networks is fragile: successful implementations require careful initialization and tuning of prior variances, as well as controlling the variance of Monte Carlo gradient estimates. We fix VB and turn it into a robust inference tool for Bayesian neural networks. We achieve this with two innovations: first, we introduce a novel deterministic method to approximate moments in neural networks, eliminating gradient variance; second, we introduce a hierarchical prior for parameters and a novel Empirical Bayes procedure for automatically selecting prior variances. Combining these two innovations, the resulting method is highly efficient and robust. On the application of heteroscedastic regression we demonstrate strong predictive performance over alternative approaches.

# 1 INTRODUCTION

Bayesian approaches to neural network training marry the representational flexibility of deep neural networks with principled parameter estimation in probabilistic models. Compared to "standard" parameter estimation by maximum likelihood, the Bayesian framework promises to bring key advantages such as better uncertainty estimates on predictions and automatic model regularization (MacKay, 1992; Graves, 2011). These features are often crucial for informing downstream decision tasks and reducing overfitting, particularly on small datasets. However, despite potential advantages, such Bayesian neural networks (BNNs) are often overlooked due to two limitations: First, posterior inference in deep neural networks is analytically intractable and approximate inference with Monte Carlo (MC) techniques can suffer from crippling variance given only a reasonable computation budget (Kingma et al., 2015; Molchanov et al., 2017; Miller et al., 2017; Zhu et al., 2018). Second, performance of the Bayesian approach is sensitive to the choice of prior (Neal, 1993), and although we may have a priori knowledge concerning the function represented by a neural network, it is generally difficult to translate this into a meaningful prior on neural network weights. Sensitivity to priors and initialization makes BNNs non-robust and thus often irrelevant in practice.

In this paper, we describe a novel approach for inference in feed-forward BNNs that is simple to implement and aims to solve these two limitations. We adopt the paradigm of variational Bayes (VB) for BNNs (Hinton & van Camp, 1993; MacKay, 1995c) which is normally deployed using Monte Carlo variational inference (MCVI) (Graves, 2011; Blundell et al., 2015). Within this paradigm we address the two shortcomings of current practice outlined above: First, we address the issue of high variance in MCVI, by reducing this variance to zero through novel deterministic approximations to variational inference in neural networks. Second, we derive a general and robust Empirical Bayes (EB) approach to prior choice using hierarchical priors. By exploiting conjugacy we derive data-adaptive closed-form variance priors for neural network weights, which we experimentally demonstrate to be remarkably effective.

Combining these two novel ingredients gives us a performant and robust BNN inference scheme that we refer to as "deterministic variational inference" (DVI). We demonstrate robustness and superior predictive performance in the context of non-linear regression models, deriving novel closed-form results for expected log-likelihoods in homoscedastic and heteroscedastic regression (similar derivations for classification can be found in the appendix).

Experiments on standard regression datasets from the UCI repository, (Dheeru & Karra Taniskidou, 2017), show that for identical models DVI converges to local optima with better predictive log-likelihoods than existing methods based on MCVI. In direct comparisons, we show that our Empirical Bayes formulation automatically provides better or comparable test performance than manual tuning of the prior and that heteroscedastic models consistently outperform the homoscedastic models.

Concretely, our contributions are:

- Development of a deterministic procedure for propagating uncertain activations through neural networks with uncertain weights and ReLU or Heaviside activation functions.  
- Development of an EB method for principled tuning of weight priors during BNN training.  
- Experimental results showing the accuracy and efficiency of our method and applicability to heteroscedastic and homoscedastic regression on real datasets.

# 2 VARIATIONAL INFERENCE IN BAYESIAN NEURAL NETWORKS

We start by describing the inference task that our method must solve to successfully train a BNN. Given a model  $\mathcal{M}$  parameterized by weights  $\boldsymbol{w}$  and a dataset  $\mathcal{D} = (\boldsymbol{x},\boldsymbol{y})$ , the inference task is to discover the posterior distribution  $p(\boldsymbol{w}|\boldsymbol{x},\boldsymbol{y})$ . A variational approach acknowledges that this posterior generally does not have an analytic form, and introduces a variational distribution  $q(\boldsymbol{w};\boldsymbol{\theta})$  parameterized by  $\boldsymbol{\theta}$  to approximate  $p(\boldsymbol{w}|\boldsymbol{x},\boldsymbol{y})$ . The approximation is considered optimal within the variational family for  $\boldsymbol{\theta}^*$  that minimizes the Kullback-Leibler (KL) divergence between  $q$  and the true posterior.

$$
\boldsymbol {\theta} ^ {*} = \underset {\boldsymbol {\theta}} {\operatorname {a r g m i n}} D _ {\mathrm {K L}} \left[ q (\boldsymbol {w}; \boldsymbol {\theta}) | | p (\boldsymbol {w} | \boldsymbol {x}, \boldsymbol {y}) \right].
$$

Introducing a prior  $p(\pmb{w})$  and applying Bayes rule allows us to rewrite this as optimization of the quantity known as the evidence lower bound (ELBO):

$$
\boldsymbol {\theta} ^ {*} = \underset {\boldsymbol {\theta}} {\operatorname {a r g m a x}} \left\{\mathbb {E} _ {\boldsymbol {w} \sim q} [ \log p (\boldsymbol {y} | \boldsymbol {w}, \boldsymbol {x}) ] - D _ {\mathrm {K L}} [ q (\boldsymbol {w}; \boldsymbol {\theta}) | | p (\boldsymbol {w}) ] \right\}. \tag {1}
$$

Analytic results exist for the KL term in the ELBO for careful choice of prior and variational distributions (e.g. Gaussian families). However, when  $\mathcal{M}$  is a non-linear neural network, the first term in equation 1 (referred to as the reconstruction term) cannot be computed exactly: this is where MC approximations with finite sample size  $S$  are typically employed:

$$
\mathbb {E} _ {\boldsymbol {w} \sim q} [ \log p (\boldsymbol {y} | \boldsymbol {w}, \boldsymbol {x}) ] \approx \frac {1}{S} \sum_ {s = 1} ^ {S} \log p (\boldsymbol {y} | \boldsymbol {w} ^ {(s)}, \boldsymbol {x}), \quad \boldsymbol {w} ^ {(s)} \sim q (\boldsymbol {w}; \boldsymbol {\theta}). \tag {2}
$$

Our goal in the next section is to develop an explicit and accurate approximation for this expectation, which provides a deterministic, closed-form expectation calculation, stabilizing BNN training by removing all stochasticity due to Monte Carlo sampling.

# 3 DETERMINISTIC VARIATIONAL APPROXIMATION

Figure 1 shows the architecture of the computation of  $\mathbb{E}_{\boldsymbol{w} \sim q}[\log p(\mathcal{D}|\boldsymbol{w})]$  for a feed-forward neural network. The computation can be divided into two parts: first, propagation of activations through parameterized layers and second, evaluation of an unparameterized log-likelihood function  $(\mathcal{L})$ . In this section, we describe how each of these stages is handled in our deterministic framework.

# 3.1 MOMENT PROPAGATION

We begin by considering activation propagation (figure 1(a)), with the aim of deriving the form of an approximation  $\tilde{q} (\pmb{a}^L)$  to the final layer activation distribution  $q(\pmb{a}^{L})$  that will be passed to

![](images/126a404df9df959e9aa7eb436ba11e752e30e7cdb7b12fe7a37bbf6b854768f5.jpg)  
Figure 1: Architecture of a Bayesian neural network. Computation is divided into (a) propagation of activations  $(\pmb{a})$  from an input  $x$  and (b) computation of a log-likelihood function  $\mathcal{L}$  for outputs  $y$ . Weights are represented as high dimensional variational distributions (blue) that induce distributions over activations (yellow). MCVI computes using samples (dots); our method propagates a full distribution.

the likelihood computation. We compute  $\pmb{a}^{L}$  by sequentially computing the distributions for the activations in the preceding layers. Concretely, we define the action of the  $l^{\mathrm{th}}$  layer that maps  $\pmb{a}^{(l-1)}$  to  $\pmb{a}^{l}$  as follows:

$$
\boldsymbol {h} ^ {l} = f (\boldsymbol {a} ^ {(l - 1)}),
$$

$$
\boldsymbol {a} ^ {l} = \boldsymbol {h} ^ {l} \boldsymbol {W} ^ {l} + \boldsymbol {b} ^ {l},
$$

where  $f$  is a non-linearity and  $\{W^l, b^l\} \subset \mathbf{w}$  are random variables representing the weights and biases of the  $l^{\mathrm{th}}$  layer that are assumed independent from weights in other layers. For notational clarity, in the following we will suppress the explicit layer index  $l$ , and use primed symbols to denote variables from the  $(l - 1)^{\mathrm{th}}$  layer, e.g.  $\pmb{a}' = \pmb{a}^{(l - 1)}$ . Note that we have made the non-conventional choice to draw the boundaries of the layers such that the linear transform is applied after the nonlinearity. This is to emphasize that  $\pmb{a}^l$  is constructed by linear combination of many distinct elements of  $h'$ , and in the limit of vanishing correlation between terms in this combination, we can appeal to the central limit theorem (CLT). Under the CLT, for a large enough hidden dimension, elements  $a_i$  will be normally distributed regardless of the potentially complicated distribution for  $h_j$  induced by  $f^1$ . We empirically observe that this claim is approximately valid even when (weak) correlations appear between the elements of  $h$  during training (see section 3.1.1).

Having argued that  $\pmb{a}$  adopts a Gaussian form, it remains to compute the first and second moments. In general, these cannot be computed exactly, so we develop an approximate expression. An overview of this derivation is presented here with more details in appendix A. First, we model  $W$ ,  $b$  and  $h$  as independent random variables, allowing us to write:

$$
\langle a _ {i} \rangle = \langle h _ {j} \rangle \langle W _ {j i} \rangle + \langle b _ {i} \rangle ,
$$

$$
\operatorname {C o v} \left(a _ {i}, a _ {k}\right) = \left\langle h _ {j} h _ {l} \right\rangle \operatorname {C o v} \left(W _ {j i}, W _ {l k}\right) + \left\langle W _ {j i} \right\rangle \operatorname {C o v} \left(h _ {j}, h _ {l}\right) \left\langle W _ {l k} \right\rangle + \operatorname {C o v} \left(b _ {i}, b _ {k}\right), \tag {3}
$$

where we have employed the Einstein summation convention and used angle brackets to indicate expectation over  $q$ . If we choose a variational family with analytic forms for weight means and covariances (e.g. Gaussian with variational parameters  $\langle W_{ji} \rangle$  and  $\mathrm{Cov}(W_{ji}, W_{lk})$ ), then the only difficult terms are the moments of  $h$ :

$$
\left\langle h _ {j} \right\rangle \propto \int f \left(\alpha_ {j}\right) \exp \left[ - \frac {\left(\alpha_ {j} - \left\langle a _ {j} ^ {\prime} \right\rangle\right) ^ {2}}{2 \Sigma_ {j j} ^ {\prime}} \right] d \alpha_ {j}, \tag {4}
$$

$$
\langle h _ {j} h _ {l} \rangle \propto \int f (\alpha_ {j}) f (\alpha_ {l}) \exp \left[ - \frac {1}{2} \binom {\alpha_ {j} - \left\langle a _ {j} ^ {\prime} \right\rangle} {\alpha_ {l} - \left\langle a _ {l} ^ {\prime} \right\rangle} ^ {\top} \binom {\Sigma_ {j j} ^ {\prime}} {\Sigma_ {l j} ^ {\prime}} \binom {\Sigma_ {j l} ^ {\prime}} {\Sigma_ {l l} ^ {\prime}} ^ {- 1} \binom {\alpha_ {j} - \left\langle a _ {j} ^ {\prime} \right\rangle} {\alpha_ {l} - \left\langle a _ {l} ^ {\prime} \right\rangle} \right] d \alpha_ {j} d \alpha_ {l}, \tag {5}
$$

where we have used the Gaussian form of  $\pmb{a}^{\prime}$  parameterized by mean  $\langle a^{\prime}\rangle$  and covariance  $\Sigma^{\prime}$ , and for brevity we have omitted the normalizing constants. Closed form solutions for the integral in equation 4 exist for Heaviside or ReLU choices of non-linearity  $f$  (see appendix A). Furthermore, for these non-linearities, the  $\langle a_j^\prime \rangle \to \pm \infty$  and  $\langle a_l^\prime \rangle \rightarrow \pm \infty$  asymptotes of the integral in equation 5 have closed form. Figure 2 shows schematically how these asymptotes can be used as a first approximation for equation 5. This approximation is improved by considering that (by definition) the residual decays to zero far from the origin in the  $(\langle a_j^\prime \rangle ,\langle a_l^\prime \rangle)$  plane, and so is well modelled by a decaying function

<table><tr><td></td><td colspan="2">A(μ1, μ2, ρ)</td><td colspan="2">Q(μ1, μ2, ρ)</td></tr><tr><td>Heaviside</td><td colspan="2">Φ(μ1)Φ(μ2)</td><td colspan="2">- log(gh/2π) + ρ/2ghρ[μ12 + μ22 - 2ρ/1+ρμ1μ2] + O(μ4)</td></tr><tr><td>ReLU</td><td colspan="2">SR(μ1)SR(μ2) + ρΦ(μ1)Φ(μ2)</td><td colspan="2">- log(gr/2π) + [ρ/2gr(1+ρ) (μ12 + μ22) - arcsin ρ-ρ/grρμ1μ2] + O(μ4)</td></tr></table>

Table 1: Forms for the components of the approximation in equation 6 for Heaviside and ReLU non-linearities.  $\Phi$  is the CDF of a standard Gaussian, SR is a "soft ReLU" that we define as  $\mathrm{SR}(x) = \phi(x) + x\Phi(x)$  where  $\phi$  is a standard Gaussian,  $\bar{\rho} = \sqrt{1 - \rho^2}$ ,  $g_h = \arcsin \rho$  and  $g_r = g_h + \frac{\rho}{1 + \bar{\rho}}$

$\exp[-Q(\langle a_j' \rangle, \langle a_l' \rangle, \Sigma')]$ , where  $Q$  is a polynomial in  $\langle a' \rangle$  with a dominant positive even term. In practice we truncate  $Q$  at the quadratic term, and calculate the polynomial coefficients by matching the moments of the resulting Gaussian with the analytic moments of the residual. Specifically, using dimensionless variables  $\mu_i' = \langle a_i' \rangle / \sqrt{\Sigma_{ii}'}$  and  $\rho_{ij}' = \Sigma_{ij}' / \sqrt{\Sigma_{ii}'\Sigma_{jj}'}$ , this improved approximation takes the form

$$
\left\langle h _ {j} h _ {l} \right\rangle = \frac {\Sigma_ {j l} ^ {\prime}}{\rho_ {j l} ^ {\prime}} \left\{A \left(\mu_ {j} ^ {\prime}, \mu_ {l} ^ {\prime}, \rho_ {j l} ^ {\prime}\right) + \exp \left[ - Q \left(\mu_ {j} ^ {\prime}, \mu_ {l} ^ {\prime}, \rho_ {j l} ^ {\prime}\right) \right] \right\}, \tag {6}
$$

where the expressions for the asymptote  $A$  and quadratic  $Q$  are given in table table 1 and derived in appendix A.2.1 and A.2.2. Using equation 6 in equation 3 gives a closed form approximation for the moments of  $\pmb{a}$  as a function of moments of  $\pmb{a}^{\prime}$ . Since  $\pmb{a}$  is approximately normally distributed by the CLT, this is sufficient information to sequentially propagate moments all the way through the network to compute the mean and covariances of  $\tilde{q} (\pmb{a}^{L})$ , our explicit multivariate Gaussian approximation to  $q(\pmb{a}^{L})$ . Any deep learning framework supporting special functions  $\arcsin$  and  $\Phi$  will immediately support backpropagation through the deterministic expressions we have presented. Below we briefly empirically verify the presented approximation, and in section 3.2 we will show how it is used to compute an approximate log-likelihood and posterior predictive distribution for regression and classification tasks.

# 3.1.1 EMPIRICAL VERIFICATION

Approximation accuracy The approximation derived above relies on three assumptions. First, that some form of CLT holds for the hidden units during training where the iid assumption of the classic CLT is not strictly enforced; second, that a quadratic truncation of  $Q$  is sufficient<sup>2</sup>; and third that there are only weak correlation between layers so that they can be represented using independent variables in the variational distribution. To provide evidence that these assumptions hold in practice, we train a small ReLU network with two hidden layers each of 128 units to perform 1D heteroscedastic regression on a toy dataset

of 500 points drawn from the distribution shown in figure 3(b). The training objective is taken from section 4, and the only detail required here is that  $\pmb{a}^{L}$  is a 2-element vector where the elements are labelled as  $(m,\ell)$ . We use a diagonal Gaussian variational family to represent the weights, but we preserve the full covariance of  $\pmb{a}$  during propagation. Using an input  $x = 0.25$  (see arrow, Figure 3(b)) we compute the distributions for  $m$  and  $\ell$  both at the start of training (where we expect the iid assumption to hold) and at convergence (where iid does not necessarily hold). Figure 3(c) shows the comparison between  $\pmb{a}^{L}$  distributions reported by our deterministic approximation and MC evaluation using 20k samples from  $q(\pmb{w};\pmb{\theta})$ . This comparison is qualitatively excellent for all cases considered.

![](images/d24c07bb978c5c16a18c27e9902d67d6bd37467113a0fd375abde50a125f4b9d.jpg)

![](images/249ea470653749c0387abe844ea312b8093baad3d03e4a441d553bc2f3b5e8af.jpg)  
Figure 2: Approximation of  $\langle h_jh_l\rangle$  using an asymptote and Gaussian correction for (a) Heaviside and (b) ReLU non-linearities. Yellow functions have closed-forms, and blue indicates residuals. The examples are plotted for  $-6 < \mu^{\prime} < 6$  and  $\rho_{jl}^{\prime} = 0.5$ , and the relative magnitude of each correction term is indicated on the vertical axis.

![](images/26dd3c87d28b9e7a7a6b1c99402999399647a54ae15672900563fb6bb47ab192.jpg)  
(a)

![](images/1950059d1d89b7cc719653d57564b4956c0b297d022e8971831a06a0523ac7e4.jpg)  
(b)

![](images/c3f94a41fa5d5e75119129798b64c3f19e32b6ccabcf7db0e3946b3581a6c029.jpg)  
(c)

![](images/c90cfd73116fd03e1b043c54fd4df3969eadee25d945583cc17a69df4d8985c8.jpg)

![](images/08db0b5234e9fde02e24c15ab6bba6bc8ff261019e42cdc4062963075203d893.jpg)  
Figure 3: Empirical accuracy of our approximation on toy 1-dimensional data. (a) We train a 2 layer ReLU network to perform heteroscedastic regression on the dataset shown in (b) and obtain the fit shown in blue. (c) The output distributions for the activation units  $m$  and  $\ell$  evaluated at  $x = 0.25$  are in excellent agreement with Monte Carlo (MC) integration with a large number (20k) of samples both before and after training.

![](images/6fa0f16ffe01ac59ecd89b94e69b9bb7cf9b2d0efed955641b6e87a0f165867d.jpg)

![](images/83341059ced98c7711000954f3eeaaa6eef8fd18ff82733ba252c197990dabe7.jpg)  
Figure 4: Runtime performance of VI methods. We show the time to propagate a batch of 10 activation vectors through a single  $d \times d$  layer. For MCVI we label curves with the number of samples used, and we show quadratic and cubic scaling guides-to-the-eye (black). Black dots indicate where our implementation runs out of memory (16GB).

Computational efficiency In traditional MCVI, propagation of  $S$  samples of  $d$ -dimensional activations through a layer containing a  $d \times d$ -dimensional transformation requires  $\mathcal{O}(Sd^2)$  compute and  $\mathcal{O}(Sd)$  memory. Our DVI method approximates the  $S \to \infty$  limit, while only demanding  $\mathcal{O}(d^3)$  compute and  $\mathcal{O}(d^2)$  memory (the additional factor of  $d$  arises from manipulation of the quadratically

large covariance matrix  $\mathrm{Cov}[h_j, h_l]$ . Whereas MCVI can always trade compute and memory for accuracy by choosing a small value for  $S$ , the inherent scaling of DVI with  $d$  could potentially limit its practical use for networks with large hidden size. To avoid this limitation, we also consider the case where only the diagonal entries  $\mathrm{Cov}(h_j, h_j)$  are computed and stored at each layer. We refer to this method as "diagonal-DVI" (dDVI), and in section 6 we show the surprising result that the strong test performance of DVI is largely retained by dDVI across a range of datasets. Figure 4 shows the time required to propagate activations through a single layer using the MCVI, DVI and dDVI methods on a Tesla V100 GPU. As a rough rule of thumb (on this hardware), for layer sizes of practical relevance, we see that absolute DVI runtimes roughly equate to MCVI with  $S = 300$  and dDVI runtime equates to  $S = 1$ .

# 3.2 LOG-LIKELIHOOD EVALUATION

To use the moment propagation procedure derived above for training BNNs, we need to build a function  $\mathcal{L}$  that maps final layer activations  $\pmb{a}^{L}$  to the expected log-likelihood term in equation 1 (see figure 1(b)). In appendix B.1 we show the intuitive result that this expected log-likelihood over  $q(\pmb{w})$  can be rewritten as an expectation over  $\tilde{q}(\pmb{a}^{L})$ .

$$
\mathbb {E} _ {\boldsymbol {w} \sim q} [ \log p (y | \boldsymbol {x}, \boldsymbol {w}) ] = \mathbb {E} _ {\boldsymbol {a} ^ {L} \sim q (\boldsymbol {a} ^ {L})} [ \log p (y | \boldsymbol {a} ^ {L}) ]. \tag {7}
$$

With this form we can derive closed forms for specific tasks; for brevity we focus on the regression case and refer the reader to appendices B.4 and B.5 for the classification case.

Regression Case For simplicity we consider scalar  $y$  and a Gaussian noise model parameterized by mean  $m(\pmb{x};\pmb{w})$  and heteroscedastic log-variance  $\log \sigma_y^2 (\pmb {x}) = \ell (\pmb {x};\pmb {w})$ . The parameters of this Gaussian are read off as the elements of a 2-dimensional output layer  $\pmb{a}^{L} = (m,\ell)$  so that  $p(y|\pmb{a}^{L}) = \mathcal{N}\left[y|m,e^{\ell}\right]$ . Recall that these parameters themselves are uncertain and the statistics  $\langle \pmb{a}^{L}\rangle$  and  $\pmb{\Sigma}^{L}$  can be computed following section 3.1. Inserting the Gaussian forms for  $p(y|\pmb{a}^{L})$  and  $q(\pmb{a}^{L})$  into equation 7 and performing the integral (see appendix B.2) gives a closed form expression

for the ELBO reconstruction term:

$$
\mathbb {E} _ {\boldsymbol {a} ^ {L} \sim \tilde {q} (\boldsymbol {a} ^ {L})} [ \log p (y | \boldsymbol {a} ^ {L}) ] = - \frac {1}{2} \left[ \log 2 \pi + \langle \ell \rangle + \frac {\Sigma_ {m m} + \left(\langle m \rangle - \Sigma_ {m \ell} - y\right) ^ {2}}{e ^ {\langle \ell \rangle} - \Sigma_ {\ell \ell / 2}} \right]. \tag {8}
$$

This heteroscedastic model can be made homoscedastic by setting  $\langle \ell \rangle = \Sigma_{\ell \ell} = \Sigma_{m\ell} = 0$ . The expression in equation 8 completes the derivations required to implement the closed form approximation to the ELBO reconstruction term for training a network. In addition, we can also compute a closed form approximation to the predictive distribution that is used at test-time to produce predictions that incorporate all parameter uncertainties. By approximating the moments of the posterior predictive and assuming normality (see appendix B.3), we find:

$$
p (y) \approx \int p (y | \boldsymbol {a} ^ {L}) \tilde {q} (\boldsymbol {a} ^ {L}) d \boldsymbol {a} ^ {L} \approx \mathcal {N} \left(y \mid \langle m \rangle , \Sigma_ {m m} + e ^ {\langle \ell \rangle + \Sigma_ {\ell \ell} / 2}\right). \tag {9}
$$

# 4 EMPIRICAL BAYES FOR VARIATIONAL BNNS

So far, we have described methods for deterministic approximation of the reconstruction term in the ELBO. We now turn to the KL term. For a  $d$ -dimensional Gaussian prior  $p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{\mu}_{\mathrm{p}}, \boldsymbol{\Sigma}_{\mathrm{p}})$ , the KL divergence with the Gaussian variational distribution  $q = \mathcal{N}(\boldsymbol{\mu}_{\mathrm{q}}, \boldsymbol{\Sigma}_{\mathrm{q}})$  has closed form:

$$
D _ {\mathrm {K L}} [ q | | p ] = \frac {1}{2} \left[ \log \frac {| \boldsymbol {\Sigma} _ {\mathrm {p}} |}{| \boldsymbol {\Sigma} _ {\mathrm {q}} |} - d + \operatorname {T r} \left(\boldsymbol {\Sigma} _ {\mathrm {p}} ^ {- 1} \boldsymbol {\Sigma} _ {\mathrm {q}}\right) + \left(\boldsymbol {\mu} _ {\mathrm {p}} - \boldsymbol {\mu} _ {\mathrm {q}}\right) ^ {\top} \boldsymbol {\Sigma} _ {\mathrm {p}} ^ {- 1} \left(\boldsymbol {\mu} _ {\mathrm {p}} - \boldsymbol {\mu} _ {\mathrm {q}}\right) \right]. \tag {10}
$$

However, this requires selection of  $(\mu_{\mathrm{p}}, \Sigma_{\mathrm{p}})$  for which there is usually little intuition beyond arguing  $\boldsymbol{\mu}_{\mathrm{p}} = \mathbf{0}$  by symmetry and choosing  $\Sigma_{\mathrm{p}}$  to preserve the expected magnitude of the propagated activations (Glorot & Bengio, 2010; He et al., 2015). In practice, variational Bayes for neural network parameters is sensitive to the choice of prior variance parameters, and we will demonstrate this problem empirically in section 6 (figure 5).

To make variational Bayes robust we parameterize the prior hierarchically, retaining a conditional diagonal Gaussian prior and variational distribution on the weights. The hierarchical prior takes the form  $\mathbf{s} \sim p(\mathbf{s})$ ;  $\boldsymbol{w} \sim p(\boldsymbol{w}|\mathbf{s})$ , using an inverse gamma distribution on  $\mathbf{s}$  as the conjugate prior to the elements of the diagonal Gaussian variance. We partition the weights into sets  $\{\lambda\}$  that typically coincide with the layer partitioning<sup>3</sup>, and assign a single element in  $\mathbf{s}$  to each set:

$$
s _ {\lambda} \sim \operatorname {I n v - G a m m a} (\alpha , \beta), \quad w _ {i} ^ {\lambda} \sim \mathcal {N} (0, s _ {\lambda}), \tag {11}
$$

for shape  $\alpha$  and scale  $\beta$ , and where  $w_{i}^{\lambda}$  is the  $i^{\mathrm{th}}$  weight in set  $\lambda$ .

Rather than taking the fully Bayesian approach, we adopt an empirical Bayes approach (Type-2 MAP), optimizing  $s^{\lambda}$ , assuming that the integral is dominated by a contribution from this optimal value  $s^{\lambda} = s_{*}^{\lambda}$ . We use the data to inform the optimal setting of  $s_{*}^{\lambda}$  to produce the tightest ELBO:

$$
\begin{array}{l} \operatorname {E L B O} = \mathbb {E} _ {\boldsymbol {w} \sim q} \left[ \log p \left(y \mid \boldsymbol {h} ^ {L} (\boldsymbol {w})\right) \right] - \left\{D _ {\mathrm {K L}} \left[ q (\boldsymbol {w}; \boldsymbol {\theta}) | | p (\boldsymbol {w} | \mathbf {s} _ {*}) p (\mathbf {s} _ {*}) \right] \right\} \\ \Rightarrow s _ {*} ^ {\lambda} = \underset {s ^ {\lambda}} {\operatorname {a r g m i n}} \left\{D _ {\mathrm {K L}} \left[ q (\boldsymbol {w}; \boldsymbol {\theta}) | | p \left(\boldsymbol {w} ^ {\lambda} \mid s ^ {\lambda}\right) \right] - \log p \left(s ^ {\lambda}\right) \right\} \tag {12} \\ \end{array}
$$

Writing out the integral for the KL in equation 12, substituting in the forms of the distributions in equation 11 and differentiating to find the optimum gives

$$
s _ {*} ^ {\lambda} = \frac {\operatorname {T r} \left[ \boldsymbol {\Sigma} _ {\mathrm {q}} ^ {\lambda} + \boldsymbol {\mu} _ {\mathrm {q}} ^ {\lambda} \left(\boldsymbol {\mu} _ {\mathrm {q}} ^ {\lambda}\right) ^ {\top} \right] + 2 \beta}{\Omega^ {\lambda} + 2 \alpha + 2}, \tag {13}
$$

where  $\Omega^{\lambda}$  is the number of weights in the set  $\lambda$ . The influence of the data on the choice of  $s_{*}^{\lambda}$  is made explicit here through dependence on the learned variational parameters  $\Sigma_{\mathrm{q}}$  and  $\mu_{\mathrm{q}}$ . Using  $s_{*}^{\lambda}$  to populate the elements of the diagonal prior variance  $\Sigma_{\mathrm{p}}$ , we can evaluate the KL in equation 10 under the empirical Bayes prior. Optimization of the resulting ELBO then simultaneously tunes the variational distribution and prior.

In the experiments we will demonstrate that the proposed empirical Bayes approach works well; however, it only approximates the full Bayesian solution, and it could fail if we were to allow too many degrees of freedom. To see this, assume we were to use one prior per weight element, and we would also define a hyperprior for each prior mean. Then, adjusting both the prior variance and prior mean using empirical Bayes would always lead to a KL-divergence of zero and the ELBO objective would degenerate into maximum likelihood.

# 5 RELATED WORK

Bayesian neural networks have a rich history. In a 1992 landmark paper David MacKay demonstrated the many potential benefits of a Bayesian approach to neural network learning (MacKay, 1992); in particular, this work contained a convincing demonstration of naturally accounting for model flexibility in the form of the Bayesian Occam's razor, facilitating comparison between different models, accurate calibration of predictive uncertainty, and to perform learning robust to overfitting. However, at the time Bayesian inference was achieved only for small and shallow neural networks using a comparatively crude Laplace approximation. Another early review article summarizing advantages and challenges in Bayesian neural network learning is (MacKay, 1995c).

This initial excitement around Bayesian neural networks led to two main methods being developed; First, Hinton & van Camp (1993) and MacKay (1995b) developed the variational Bayes (VB) approach for posterior inference. Whereas Hinton & van Camp (1993) were motivated from a minimum description length (MDL) compression perspective, MacKay (1995b) motivated his equivalent ensemble learning method from a statistical physics perspective of variational free energy minimization. Barber & Bishop (1998) extended the methodology for two-layer neural networks to use general multivariate Normal variational distributions. Second, Neal (1993) developed efficient gradient-based Monte Carlo methods in the form of "hybrid Monte Carlo", now known as Hamiltonian Monte Carlo, and also raised the question of prior design and limiting behaviour of Bayesian neural networks.

Rebirth of Bayesian neural networks. After more than a decade of no further work on Bayesian neural networks Graves (2011) revived the field by using Monte Carlo variational inference (MCVI) to make VB practical and scalable, demonstrating gains in predictive performance on real world tasks.

Since 2015 the VB approach to Bayesian neural networks is mainstream (Blundell et al., 2015); key research drivers since then are the problems of high variance in MCVI and the search for useful variational families. One approach to reduce variance in feedforward networks is the local reparameterization trick (Kingma et al., 2015) (see appendix D). To enhance the variational families more complicated distributions such as Matrix Gaussian posteriors (Louizos & Welling, 2016), multiplicative posteriors (Kingma et al., 2015), and hierarchical posteriors (Louizos & Welling, 2017) are used. Both our methods, the deterministic moment approximation and the empirical Bayes estimation, can potentially be extended to these richer families.

Prior choice. Choosing priors in Bayesian neural networks remains an open issue. The hierarchical priors for feedforward neural networks that we use have been investigated before by Neal (1993) and MacKay (1995a), the latter proposing a "cheap and cheerful" heuristic, alternating optimization of weights and inverse variance parameters. Barber & Bishop (1998) also used a hierarchical prior and an efficient closed-form factored VB approximation; our approach can be seen as a point estimate to their approach in order to enable use of our closed-form moment approximation. Graves (2011) also used hierarchical Gaussian priors with flat hyperpriors, deriving a closed-form update for the prior mean and variance. Compared to these prior works our approach is rigorous and with sufficient data accurately approximates the Bayesian approach of integrating over the prior parameters.

Alternative inference procedures. As an alternative to variational Bayes, probabilistic backpropagation (PBP) (Hernández-Lobato & Adams, 2015) applies approximate inference in the form of assumed density filtering (ADF) to refine a Gaussian posterior approximation. Like in our work, each update to the approximate posterior requires propagating means and variances of activations through the network. (Hernández-Lobato & Adams, 2015) only consider the diagonal propagation case and regression. Since the original work, PBP has been generalized to classification (Ghosh et al., 2016) and richer posterior families such as the matrix variate Normal posteriors (Sun et al., 2017). Our moment approximation could be used to improve the inference accuracy of PBP.

Gaussianity in neural networks. Our demonstration of Gaussianity of ReLU network activations is also directly relevant to recent work on Gaussian process interpretations of deep neural networks (Matthews et al., 2018; Lee et al., 2017), validating the insight that activations in deep neural networks are closely approximated by Gaussian processes. Two recent works derived deterministic moment approximations for deep neural networks: Bibi et al. (2018), using Price's theorem, derived exact first and second moment expressions for ReLU activations but limit themselves to the case of zero-mean Gaussian activations. Kandemir et al. (2018) also derive closed-form solutions to the

ELBO for the case of diagonal Gaussian variational families. However, their approach is limited to linear layers without bias.

Markov chain Monte Carlo approaches. Another rich class of approximate inference methods for Bayesian neural networks are stochastic gradient Markov chain Monte Carlo (SG-MCMC) methods. These methods allow for approximate posterior parameter inference using unbiased log-likelihood estimates. Stochastic gradient Langevin dynamics (SGLD) was the first method in this class (Welling & Teh, 2011). SGLD is particularly simple and efficient to implement, but recent methods increase efficiency in the case of correlated posteriors by estimating the Fisher information matrix (Ahn et al., 2012) and extend Hamiltonian Monte Carlo to the stochastic gradient case (Chen et al., 2014). A complete characterization of SG-MCMC methods is given by (Ma et al., 2015; Gong et al., 2018). However, despite this progress, important theoretical questions regarding approximation guarantees for practical computational budgets remain (Nagapetyan et al., 2017). Moreover, while SG-MCMC methods work robustly in practice, they remain computationally inefficient, especially because evaluation of the posterior predictive requires evaluating an ensemble of models.

Wild approximations. The above methods are principled but often require sophisticated implementations; recently, a few methods aim to provide "cheap" approximations to the Bayes posterior. Dropout has been interpreted by Gal & Ghahramani (2016) to approximately correspond to variational inference. Likewise, Bootstrap posteriors (Lakshminarayanan et al., 2017; Fushiki et al., 2005; Harris, 1989) have been proposed as a general, robust, and accurate method for posterior inference. However, obtaining a bootstrap posterior ensemble of size  $k$  is computationally intense at  $k$  times the computation of training a single model.

# 6 EXPERIMENTS

We implement<sup>4</sup> deterministic variational inference (DVI) as described above to train small ReLU networks on UCI regression datasets (Dheeru & Karra Taniskidou, 2017). The experiments address the claims that our methods for eliminating gradient variance and automatic tuning of the prior improve the performance of the final trained model. In Appendix C we present extended results to demonstrate that our method is competitive against a variety of models and inference schemes.

<table><tr><td>Dataset</td><td>|D|</td><td>dx</td><td>DVI</td><td>dDVI</td><td>MCVI</td><td>hoDVI</td></tr><tr><td>bost</td><td>506</td><td>13</td><td>-2.41 ± 0.02</td><td>-2.42 ± 0.02</td><td>-2.46 ± 0.02</td><td>-2.58 ± 0.04</td></tr><tr><td>conc</td><td>1030</td><td>8</td><td>-3.06 ± 0.01</td><td>-3.07 ± 0.02</td><td>-3.07 ± 0.01</td><td>-3.23 ± 0.01</td></tr><tr><td>ener</td><td>768</td><td>8</td><td>-1.01 ± 0.06</td><td>-1.06 ± 0.06</td><td>-1.03 ± 0.04</td><td>-2.09 ± 0.06</td></tr><tr><td>kin8</td><td>8192</td><td>8</td><td>1.13 ± 0.00</td><td>1.13 ± 0.00</td><td>1.14 ± 0.00</td><td>1.01 ± 0.01</td></tr><tr><td>nava</td><td>11934</td><td>16</td><td>6.29 ± 0.04</td><td>6.22 ± 0.06</td><td>5.94 ± 0.05</td><td>5.84 ± 0.06</td></tr><tr><td>powe</td><td>9568</td><td>4</td><td>-2.80 ± 0.00</td><td>-2.80 ± 0.00</td><td>-2.80 ± 0.00</td><td>-2.82 ± 0.00</td></tr><tr><td>prot</td><td>45730</td><td>9</td><td>-2.85 ± 0.01</td><td>-2.84 ± 0.01</td><td>-2.87 ± 0.01</td><td>-2.94 ± 0.00</td></tr><tr><td>wine</td><td>1588</td><td>11</td><td>-0.90 ± 0.01</td><td>-0.91 ± 0.02</td><td>-0.92 ± 0.01</td><td>-0.96 ± 0.01</td></tr><tr><td>yach</td><td>308</td><td>6</td><td>-0.47 ± 0.03</td><td>-0.47 ± 0.03</td><td>-0.68 ± 0.03</td><td>-1.41 ± 0.03</td></tr></table>

Table 2: Average test log-likelihood on UCI datasets.  $|\mathcal{D}|$  is the dataset size, and  $d_{x}$  is the input dimension.

Deterministic vs. Stochastic We compare DVI with MCVI from equation 2 with  $S = 10$  samples. The same model is used for each inference method: a single hidden layer of 50 units for each dataset considered, extending this to 100 units in the special case of the larger protein structure dataset, prot. Additionally, both methods use the same EB prior from equation 13 with a broad inverse Gamma hyperprior ( $\alpha = 1$ ,  $\beta = 10$ ) and an independent  $s_{\lambda}$  for each linear transformation. Each dataset is split into random training and test sets with  $90\%$  and  $10\%$  of the data respectively. This splitting process is repeated 20 times and the average test performance of each method at convergence is reported in table 2 (see also learning curves in appendix E). We see that DVI consistently outperforms MCVI, by up to 0.35 nats per data point on some datasets. The computationally efficient diagonal-DVI (dDVI) surprisingly retains much of this performance. By default we use the heteroscedastic

model, and we observe that this uniformly delivers better results than a homoscedastic model (hoDVI; rightmost column in table 2) on these datasets with no overfitting issues<sup>5</sup>.

Empirical Bayes In Figure 5 we compare the performance of networks trained with manual tuning of a fixed Gaussian prior to networks trained with the automatic EB tuning. We find that the EB method consistently finds priors that produce models with competitive or significantly improved test log-likelihood relative to the best manual setting. Since this observation holds across all datasets considered, we say that our method is "robust". Note that the EB method can outperform manual tuning because it automatically finds different prior variances for each weight matrix, whereas in the manual tuning case we search over a single hyperparameter controlling all prior variances.

![](images/7718d0194d6032d1ec2cf354786ee79ea384d9dc8500de1ce2979af302ef5878.jpg)  
Figure 5: Comparison of converged test log-likelihood with a manually tuned prior variance (orange) or empirical Bayes (blue).

# 7 CONCLUSION

We introduced two innovations to make variational inference for neural networks more robust: 1. an effective deterministic approximation to the moments of activations of a neural networks; and 2. a simple empirical Bayes hyperparameter update. We demonstrate that together these innovations make variational Bayes a competitive method for Bayesian inference in neural heteroscedastic regression models.

Beside the challenge of efficient posterior inference, for Bayesian neural networks two major issues remain open. First, how to design suitable priors for functions represented by neural network parameters? And second, what structure do the posterior distributions in neural network models have and how can this be used to improve approximate inference (Watanabe, 2009)?

# REFERENCES

Sungjin Ahn, Anoop Korattikara, and Max Welling. Bayesian posterior sampling via stochastic gradient Fisher scoring. arXiv preprint arXiv:1206.6380, 2012.  
David Barber and Christopher M Bishop. Ensemble learning in Bayesian neural networks. NATO ASI SERIES F COMPUTER AND SYSTEMS SCIENCES, 168:215-238, 1998.  
Adel Bibi, Modar Alfadly, and Bernard Ghanem. Analytic expressions for probabilistic moments of PL-DNN with Gaussian input. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.  
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424, 2015.  
Thang Bui, Daniel Hernández-Lobato, Jose Hernandez-Lobato, Yingzhen Li, and Richard Turner. Deep Gaussian processes for regression using approximate expectation propagation. In International Conference on Machine Learning, pp. 1472-1481, 2016.  
Tianqi Chen, Emily Fox, and Carlos Guestrin. Stochastic gradient Hamiltonian Monte Carlo. In International Conference on Machine Learning, pp. 1683-1691, 2014.

Dua Dheeru and Efi Karra Taniskidou. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.  
Tadayoshi Fushiki, Fumiyasu Komaki, Kazuyuki Aihara, et al. Nonparametric bootstrap prediction. Bernoulli, 11(2):293-307, 2005.  
Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050-1059, 2016.  
Soumya Ghosh, Francesco Maria Delle Fave, and Jonathan S Yedidia. Assumed density filtering methods for learning Bayesian neural networks. In AAAI, pp. 1589-1595, 2016.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-256, 2010.  
Wenbo Gong, Yingzhen Li, and José Miguel Hernández-Lobato. Meta-learning for stochastic gradient MCMC. arXiv preprint arXiv:1806.04522, 2018.  
Alex Graves. Practical variational inference for neural networks. In Advances in neural information processing systems, pp. 2348-2356, 2011.  
Ian R Harris. Predictive fit for natural exponential families. Biometrika, 76(4):675-684, 1989.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026-1034, 2015.  
Jose Miguel Hernandez-Lobato and Ryan Adams. Probabilistic backpropagation for scalable learning of Bayesian neural networks. In International Conference on Machine Learning, pp. 1861-1869, 2015.  
GE Hinton and Drew van Camp. Keeping neural networks simple by minimising the description length of weights. In Proceedings of COLT-93, pp. 5-13, 1993.  
Melih Kandemir, Manuel Haussmann, and Fred A Hamprecht. Sampling-free variational inference of Bayesian neural nets. arXiv preprint arXiv:1805.07654, 2018.  
Diederik P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems, pp. 2575-2583, 2015.  
Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, pp. 6402-6413, 2017.  
Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. Deep neural networks as gaussian processes. arXiv preprint arXiv:1711.00165, 2017.  
Christos Louizos and Max Welling. Structured and efficient variational deep learning with matrix Gaussian posteriors. In International Conference on Machine Learning, pp. 1708-1716, 2016.  
Christos Louizos and Max Welling. Multiplicative normalizing flows for variational Bayesian neural networks. arXiv preprint arXiv:1703.01961, 2017.  
Yi-An Ma, Tianqi Chen, and Emily Fox. A complete recipe for stochastic gradient MCMC. In Advances in Neural Information Processing Systems, pp. 2917-2925, 2015.  
David JC MacKay. A practical Bayesian framework for backpropagation networks. Neural computation, 4(3):448-472, 1992.  
David JC MacKay. Bayesian neural networks and density networks. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*, 354(1):73-80, 1995a.

David JC MacKay. Developments in probabilistic modelling with neural networks—ensemble learning. In *Neural Networks: Artificial Intelligence and Industrial Applications*, pp. 191–198. Springer, 1995b.  
David JC MacKay. Probable networks and plausible predictionsa review of practical Bayesian methods for supervised neural networks. Network: Computation in Neural Systems, 6(3):469-505, 1995c.  
Alexander G de G Matthews, Mark Rowland, Jiri Hron, Richard E Turner, and Zoubin Ghahramani. Gaussian process behaviour in wide deep neural networks. arXiv preprint arXiv:1804.11271, 2018.  
Andrew Miller, Nick Foti, Alexander D'Amour, and Ryan P Adams. Reducing reparameterization gradient variance. In Advances in Neural Information Processing Systems, pp. 3708-3718, 2017.  
Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. arXiv preprint arXiv:1701.05369, 2017.  
Tigran Nagapetyan, Andrew B Duncan, Leonard Hasenclever, Sebastian J Vollmer, Lukasz Szpruch, and Konstantinos Zygalakis. The true cost of stochastic gradient Langevin dynamics. arXiv preprint arXiv:1706.02692, 2017.  
Radford M Neal. Bayesian learning via stochastic dynamics. In Advances in neural information processing systems, pp. 475-482, 1993.  
Christopher G. Small. *Expansions and Asymptotics for Statistics*. CRC Press, 2010.  
Shengyang Sun, Changyou Chen, and Lawrence Carin. Learning structured weight uncertainty in Bayesian neural networks. In Artificial Intelligence and Statistics, pp. 1283-1292, 2017.  
Jarno Vanhatalo and Aki Vehtari. Mcmc methods for MLP-network and Gaussian process and stuff—a documentation for Matlab toolbox MCMCstuff. *Laboratory of computational engineering*, Helsinki university of technology, 2006.  
Sumio Watanabe. Algebraic geometry and statistical learning theory, volume 25. Cambridge University Press, 2009.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pp. 681-688, 2011.  
Zhanxing Zhu, Ruosi Wan, and Mingjun Zhong. Neural control variates for variance reduction. arXiv preprint arXiv:1806.00159, 2018.
