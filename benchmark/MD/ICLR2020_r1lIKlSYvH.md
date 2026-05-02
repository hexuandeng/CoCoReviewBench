# THE USUAL SUSPECTS? REASSESSING BLAME FOR VAE POSTERIOR COLLAPSE

Anonymous authors

Paper under double-blind review

# ABSTRACT

In narrow asymptotic settings Gaussian VAE models of continuous data have been shown to possess global optima aligned with ground-truth distributions. Even so, it is well known that poor solutions whereby the latent posterior collapses to an uninformative prior are sometimes obtained in practice. However, contrary to conventional wisdom that largely assigns blame for this phenomena on the undue influence of KL-divergence regularization, we will argue that posterior collapse is, at least in part, a direct consequence of bad local minima inherent to the loss surface of deep autoencoder networks. In particular, we prove that even small nonlinear perturbations of affine VAE decoder models can produce such minima, and in deeper models, analogous minima can force the VAE to behave like an aggressive truncation operator, provably discarding information along all latent dimensions in certain circumstances. Regardless, the underlying message here is not meant to undercut valuable existing explanations of posterior collapse, but rather, to refine the discussion and elucidate alternative risk factors that may have been previously underappreciated.

# 1 INTRODUCTION

The variational autoencoder (VAE) (Kingma & Welling, 2014; Rezende et al., 2014) represents a powerful generative model of data points that are assumed to possess some complex yet unknown latent structure. This assumption is instantiated via the marginalized distribution

$$
p _ {\theta} (\boldsymbol {x}) = \int p _ {\theta} (\boldsymbol {x} | \boldsymbol {z}) p (\boldsymbol {z}) d \boldsymbol {z}, \tag {1}
$$

which forms the basis of prevailing VAE models. Here  $\pmb{z} \in \mathbb{R}^{\kappa}$  is a collection of unobservable latent factors of variation that, when drawn from the prior  $p(\pmb{z})$ , are colloquially said to generate an observed data point  $\pmb{x} \in \mathbb{R}^{d}$  through the conditional distribution  $p_{\theta}(\pmb{x}|\pmb{z})$ . The latter is controlled by parameters  $\theta$  that can, at least conceptually speaking, be optimized by maximum likelihood over  $p_{\theta}(\pmb{x})$  given available training examples.

In particular, assuming  $n$  training points  $\mathbf{X} = [\pmb{x}^{(1)},\dots ,\pmb{x}^{(n)}]$ , maximum likelihood estimation is tantamount to minimizing the negative log-likelihood expression  $\frac{1}{n}\sum_{i} - \log \left[p_{\theta}\left(\pmb{x}^{(i)}\right)\right]$ . Proceeding further, because the marginalization over  $z$  in (1) is often intractable, the VAE instead minimizes a convenient variational upper bound given by  $\mathcal{L}(\theta ,\phi)\triangleq$

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} \left\{- \mathbb {E} _ {q _ {\phi} (\boldsymbol {z} | \boldsymbol {x} ^ {(i)})} \left[ \log p _ {\theta} (\boldsymbol {x} ^ {(i)} | \boldsymbol {z}) \right] + \mathbb {K L} \left[ q _ {\phi} (\boldsymbol {z} | \boldsymbol {x} ^ {(i)} | | p (\boldsymbol {z}) \right] \right\} \geq \frac {1}{n} \sum_ {i = 1} ^ {n} - \log \left[ p _ {\theta} (\boldsymbol {x} ^ {(i)}) \right], \tag {2}
$$

with equality iff  $q_{\phi}(z|\pmb{x}^{(i)}) = p_{\theta}(z|\pmb{x}^{(i)})$  for all  $i$ . The additional parameters  $\phi$  govern the shape of the variational distribution  $q_{\phi}(z|\pmb{x})$  that is designed to approximate the true but often intractable latent posterior  $p_{\theta}(z|\pmb{x})$ .

The VAE energy from (2) is composed of two terms, a data-fitting loss that borrows the basic structure of an autoencoder (AE), and a KL-divergence-based regularization factor. The former incentivizes assigning high probability to latent codes  $\mathbf{z}$  that facilitate accurate reconstructions of each  $\mathbf{x}^{(i)}$ . In fact, if  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is a Dirac delta function, this term is exactly equivalent to a deterministic AE with data reconstruction loss defined by  $-\log p_{\theta}(\mathbf{x}|\mathbf{z})$ . Overall, it is because of this association that  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is generally referred to as the encoder distribution, while  $p_{\theta}(\mathbf{x}|\mathbf{z})$  denotes the decoder

distribution. Additionally, the KL regularizer  $\mathbb{KL}[q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p(\boldsymbol{z})]$  pushes the encoder distribution towards the prior without violating the variational bound.

For continuous data, which will be our primary focus herein, it is typical to assume that

$$
p (\boldsymbol {z}) = \mathcal {N} (\boldsymbol {z} | \boldsymbol {0}, \boldsymbol {I}), p _ {\theta} (\boldsymbol {x} | \boldsymbol {z}) = \mathcal {N} (\boldsymbol {x} | \boldsymbol {\mu} _ {x}, \gamma \boldsymbol {I}), \text {a n d} q _ {\phi} (\boldsymbol {z} | \boldsymbol {x}) = \mathcal {N} (\boldsymbol {z} | \boldsymbol {\mu} _ {z}, \boldsymbol {\Sigma} _ {z}), \tag {3}
$$

where  $\gamma > 0$  is a scalar variance parameter, while the Gaussian moments  $\pmb{\mu}_x \equiv \pmb{\mu}_x(z; \theta)$ ,  $\pmb{\mu}_z \equiv \pmb{\mu}_z(x; \phi)$ , and  $\pmb{\Sigma}_z \equiv \mathrm{diag}[\pmb{\sigma}_z(x; \phi)]^2$  are computed via feedforward neural network layers. The encoder network parameterized by  $\phi$  takes  $\pmb{x}$  as an input and outputs  $\pmb{\mu}_z$  and  $\pmb{\Sigma}_z$ . Similarly the decoder network parameterized by  $\theta$  converts a latent code  $z$  into  $\pmb{\mu}_x$ . Given these assumptions, the generic VAE objective from (2) can be refined to

$$
\begin{array}{l} \mathcal {L} (\theta , \phi) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\{\mathbb {E} _ {q _ {\phi}} \left(\boldsymbol {z} \mid \boldsymbol {x} ^ {(i)}\right) \left[ \frac {1}{\gamma} \| \boldsymbol {x} ^ {(i)} - \boldsymbol {\mu} _ {x} (\boldsymbol {z}; \theta) \| _ {2} ^ {2} \right] \right. \tag {4} \\ \left. + d \log \gamma + \left\| \boldsymbol {\sigma} _ {z} \left(\boldsymbol {x} ^ {(i)}; \phi\right) \right\| _ {2} ^ {2} - \log \left| \operatorname {d i a g} \left[ \boldsymbol {\sigma} _ {z} \left(\boldsymbol {x} ^ {(i)}; \phi\right) \right] ^ {2} \right| + \left\| \boldsymbol {\mu} _ {z} \left(\boldsymbol {x} ^ {(i)}; \phi\right) \right\| _ {2} ^ {2} \right\}. \\ \end{array}
$$

This expression can be optimized over using SGD and a simple reparameterization strategy (Kingma & Welling, 2014; Rezende et al., 2014) to produce parameter estimates  $\{\theta^{*},\phi^{*}\}$ . Among other things, new samples approximating the training data can then be generated via the ancestral process  $z^{new} \sim \mathcal{N}(z|\mathbf{0},I)$  and  $x^{new} \sim p_{\theta^{*}}(x|z^{new})$ .

Although it has been argued that global minima of (4) may correspond with the optimal recovery of ground truth distributions in certain asymptotic settings (Dai & Wipf, 2019), it is well known that in practice, VAE models are at risk of converging to degenerate solutions where, for example, it may be that  $q_{\phi}(z|x) = p(z)$ . This phenomena, commonly referred to as VAE posterior collapse (He et al., 2019; Razavi et al., 2019), has been acknowledged and analyzed from a variety of different perspectives as we detail in Section 2. That being said, we would argue that there remains lingering ambiguity regarding the different types and respective causes of posterior collapse. Consequently, Section 3 provides a useful taxonomy that will serve to contextualize our main technical contributions. These include the following:

- Building upon existing analysis of affine VAE decoder models, in Section 4 we prove that even arbitrarily small nonlinear activations can introduce suboptimal local minima exhibiting posterior collapse.  
- We demonstrate in Section 5 that if the encoder/decoder networks are incapable of sufficiently reducing the VAE reconstruction errors, even in a deterministic setting with no KL-divergence regularizer, there will exist an implicit lower bound on the optimal value of  $\gamma$ . Moreover, we prove that if this  $\gamma$  is sufficiently large, the VAE will behave like an aggressive thresholding operator, enforcing exact posterior collapse, i.e.,  $q_{\phi}(z|x) = p(z)$ .  
- Based on these observations, we present experiments in Section 6 establishing that as network depth/capacity is increased, even for deterministic AE models with no regularization, reconstruction errors become worse. This bounds the effective VAE trade-off parameter  $\gamma$  such that posterior collapse is essentially inevitable. Collectively then, we provide convincing evidence that posterior collapse is, at least in certain settings, the fault of deep AE local minima, and need not be exclusively a consequence of usual suspects such as the KL-divergence term.

We conclude in Section 7 with practical take-home messages, and motivate the search for improved AE architectures and training regimes that might be leveraged by analogous VAE models.

# 2 RECENT WORK AND THE USUAL SUSPECTS FOR INSTIGATING COLLAPSE

Posterior collapse under various guises is one of the most frequently addressed topics related to VAE performance. Depending on the context, arguably the most common and seemingly transparent suspect for causing collapse is the KL regularization factor that is obviously minimized by  $q_{\phi}(\pmb{z}|\pmb{x}) = p(\pmb{z})$ . This perception has inspired various countermeasures, including heuristic annealing of the KL penalty or KL warm-start (Bowman et al., 2015; Huang et al., 2018; Sønderby et al., 2016), tighter bounds on the log-likelihood (Burda et al., 2015; Rezende & Mohamed, 2015), more

complex priors (Bauer & Mnih, 2018; Tomczak & Welling, 2018), modified decoder architectures (Cai et al., 2017; Dieng et al., 2018; Yeung et al., 2017), or efforts to explicitly disallow the prior from ever equaling the variational distribution (Razavi et al., 2019). Thus far though, most published results do not indicate success generating high-resolution images, and in the majority of cases, evaluations are limited to small images and/or relatively shallow networks. This suggests that there may be more nuance involved in pinpointing the causes and potential remedies of posterior collapse. One notable exception though is the BIVA model from (Maaløe et al., 2019), which employs a bidirectional hierarchy of latent variables, in part to combat posterior collapse. While improvements in NLL scores have been demonstrated with BIVA using relatively deep encoder/decoders, this model is significantly more complex and difficult to analyze.

On the analysis side, there have been various efforts to explicitly characterize posterior collapse in restricted settings. For example, Lucas et al. (2019) demonstrate that if  $\gamma$  is fixed to a sufficiently large value, then a VAE energy function with an affine decoder mean will have minima that overprune latent dimensions. A related linearized approximation to the VAE objective is analyzed in (Rolinek et al., 2019); however, collapsed latent dimensions are excluded and it remains somewhat unclear how the surrogate objective relates to the original. Posterior collapse has also been associated with data-dependent decoder covariance networks  $\Sigma_x(z;\theta)\neq \gamma I$  (Mattei & Frellsen, 2018), which allows for degenerate solutions assigning infinite density to a single data point and a diffuse, collapsed density everywhere else. Finally, from the perspective of training dynamics, (He et al., 2019) argue that a lagging inference network can also lead to posterior collapse.

# 3 TAXONOMY OF POSTERIOR COLLAPSE

Although there is now a vast literature on the various potential causes of posterior collapse, there remains ambiguity as to exactly what this phenomena is referring to. In this regard, we believe that it is critical to differentiate five subtle yet quite distinct scenarios that could reasonably fall under the generic rubric of posterior collapse:

(i) Latent dimensions of  $\pmb{z}$  that are not needed for providing good reconstructions of the training data are set to the prior, meaning  $q_{\phi}(z_j|\pmb{x}) \approx p(z_j) = \mathcal{N}(0,1)$  at any superfluous dimension  $j$ . Along other dimensions  $\sigma_z^2$  will be near zero and  $\mu_z$  will provide a usable predictive signal leading to accurate reconstructions of the training data. This case can actually be viewed as a desirable form of selective posterior collapse that, as argued in (Dai & Wipf, 2019), is a necessary (albeit not sufficient) condition for generating good samples.  
(ii) The decoder variance  $\gamma$  is set too large such that the KL term from (2) is overly dominant, forcing most or all dimensions of  $z$  to follow the prior  $\mathcal{N}(0,1)$ . In this scenario, the actual global optimum of the VAE energy (assume I adding  $\gamma$  is fixed) will lead to deleterious posterior collapse and the model reconstructions of the training data will be poor. In fact, even the original marginal like-likelihood can potentially default to a trivial/useless solution if  $\gamma$  is fixed too large, assigning a small marginal likelihood to the training data, provably so in the affine case (Lucas et al., 2019).  
(iii) As mentioned previously, if the Gaussian decoder covariance is learned as a separate network structure (instead of simply  $\pmb{\Sigma}_x(\pmb{z};\theta) = \gamma \pmb{I}$ ), there can exist degenerate solutions that assign infinite density to a single data point and a diffuse, isotropic Gaussian elsewhere (Mattei & Frellsen, 2018). This implies that (4) can be unbounded from below at what amounts to a posterior collapsed solution and bad reconstructions almost everywhere.  
(iv) When powerful non-Gaussian decoders are used, and in particular those that can parameterize complex distributions regardless of the value of  $z$  (e.g., PixelCNN-based (Van den Oord et al., 2016)), it is possible for the VAE to assign high-probability to the training data even if  $q_{\phi}(z|x) = p(z)$  (Alemi et al., 2017; Bowman et al., 2015; Chen et al., 2016). This category of posterior collapse is quite distinct from categories (ii) and (iii) above in that, although the reconstructions are similarly poor, the associated NLL scores can still be good.  
(v) The previous fifth categories of posterior collapse can all be directly associated with emergent properties of the VAE global minimum under various modeling conditions. In contrast, a forth type of collapse exists that is the explicit progeny of bad VAE local minima. More

specifically, as we will argue shortly, when deeper encoder/decoder networks are used, the risk of converging to bad, overregularized solutions increases.

The remainder of this paper will primarily focus on category (v), with brief mention of the other types for comparison purposes where appropriate. Our rationale for this selection bias is that, unlike the others, category (i) collapse is actually advantageous and hence need not be mitigated. In contrast, while category (ii) is undesirable, it can be avoided by learning  $\gamma$ . As for category (iii), this represents an unavoidable consequence of models with flexible decoder covariances capable of detecting outliers (Dai et al., 2018). In fact, even simpler inlier/outlier decomposition models such as robust PCA are inevitably at risk for this phenomena (Candès et al., 2011). Regardless, when  $\pmb{\Sigma}_{z}(\pmb{x};\theta) = \gamma \pmb{I}$  this problem goes away. And finally, we do not address category (iv) in depth simply because it is unrelated to the canonical Gaussian VAE models of continuous data that we have chosen to examine herein. Regardless, it is still worthwhile to explicitly differentiate these five types and bare them in mind when considering attempts to both explain and improve VAE models.

# 4 INSIGHTS FROM SIMPLIFIED CASES

Because different categories of posterior collapse can be impacted by different global/local minima structures, a useful starting point is a restricted setting whereby we can comprehensively characterize all such minima. For this purpose, we first consider a VAE model with the decoder network set to an affine function. As is often assumed in practice, we choose  $\pmb{\Sigma}_{x} = \gamma \pmb{I}$ , where  $\gamma > 0$  is a scalar parameter within the parameter set  $\theta$ . In contrast, for the mean function we choose  $\pmb{\mu}_{x} = \pmb{W}_{x}\pmb{z} + \pmb{b}_{x}$  for some weight matrix  $\pmb{W}_{x}$  and bias vector  $\pmb{b}_{x}$ . The encoder can be arbitrarily complex (although the optimal structure can be shown to be affine as well).

Given these simplifications, and assuming the data has  $r \geq \kappa$  nonzero singular values, it has been demonstrated that at any global optima, the columns of  $W_{x}$  will correspond with the first  $\kappa$  principal components of the training data matrix  $X$  provided that we simultaneously learn  $\gamma$  or set it to the optimal value (which is available in closed form) (Dai et al., 2018; Lucas et al., 2019; Tipping & Bishop, 1999). Additionally, it has also been shown that no spurious, suboptimal local minima will exist. Note also that if  $r < \kappa$  the same basic conclusions still apply; however,  $W_{x}$  will only have  $r$  nonzero columns, each corresponding with a different principal component of the data. The unused latent dimensions will satisfy  $q_{\phi}(z|x) = \mathcal{N}(0,I)$ , which represents the canonical form of the benign category (i) posterior collapse. Collectively, these results imply that provided we converge to any local minima of the VAE energy, we will obtain the best possible linear approximation to the data using a minimal number of latent dimensions, and malignant posterior collapse is not an issue, i.e., categories (ii)-(v) will not arise.

Even so, if instead of learning  $\gamma$ , we choose a fixed value that is larger than any of the significant singular values of  $X X^{\top}$ , then category (ii) posterior collapse can be inadvertently introduced. More specifically, let  $\tilde{r}_{\gamma}$  denote the number of such singular values that are smaller than some fixed  $\gamma$  value. Then along  $\kappa - \tilde{r}_{\gamma}$  latent dimensions  $q_{\phi}(z|x) = \mathcal{N}(0,I)$ , and the corresponding columns of  $W_{x}$  will be set to zero, regardless of whether or not these dimensions are necessary for accurately reconstructing the data. The risk of this type of posterior collapse will likely be inherited by deeper models as well as argued in (Lucas et al., 2019).

Of course when we move to more complex architectures, the risk of bad local minima or other suboptimal stationary points becomes a new potential concern, and it is not clear that the affine case described above contributes to reliable, predictive intuitions. To illustrate this point, we will now demonstrate that the introduction of an arbitrarily small nonlinearity can nonetheless produce a pernicious local minimum that exhibits category (v) posterior collapse. In particular, we now assume the decoder mean function

$$
\boldsymbol {\mu} _ {x} = \pi_ {\alpha} \left(\boldsymbol {W} _ {x} \boldsymbol {z}\right) + \boldsymbol {b} _ {x}, \text {w i t h} \pi_ {\alpha} (u) \triangleq \operatorname {s i g n} (u) (| u | - \alpha) _ {+}, \alpha \geq 0. \tag {5}
$$

The function  $\pi_{\alpha}$  is nothing more than a soft-threshold operator as is commonly used in neural network architectures designed to reflect unfolded iterative algorithms for representation learning (Gregor & LeCun, 2010; Sprechmann et al., 2015). In the present context though, we choose this nonlinearity largely because it allows (5) to reflect arbitrarily small perturbations away from a strictly affine model, and indeed if  $\alpha = 0$  the exact affine model is recovered. Collectively, these specifications lead to the parameterization  $\theta = \{\pmb{W}_x, \pmb{b}_x, \gamma\}$  and  $\phi = \{\pmb{\mu}_z^{(i)}, \pmb{\sigma}_z^{(i)}\}_{i=1}^n$  and energy given

by

$$
\begin{array}{l} \mathcal {L} (\theta , \phi) = \sum_ {i = 1} ^ {n} \left\{\mathbb {E} _ {q _ {\phi}} \left(\boldsymbol {z} \mid \boldsymbol {x} ^ {(i)}\right) \left[ \frac {1}{\gamma} \left\| \boldsymbol {x} ^ {(i)} - \pi_ {\alpha} \left(\boldsymbol {W} _ {x} \boldsymbol {z}\right) - \boldsymbol {b} _ {x} \right\| _ {2} ^ {2} \right] \right. \tag {6} \\ + d \log \gamma + \left\| \boldsymbol {\sigma} _ {z} ^ {(i)} \right\| _ {2} ^ {2} - \log \left| \operatorname {d i a g} \left[ \boldsymbol {\sigma} _ {z} ^ {(i)} \right] ^ {2} \right| + \left\| \boldsymbol {\mu} _ {z} ^ {(i)} \right\| _ {2} ^ {2} \}, \\ \end{array}
$$

where  $\pmb{\mu}_{z}^{(i)}$  and  $\sigma_{z}^{(i)}$  denote arbitrary encoder moments for data point  $i$  (this is consistent with the assumption of an arbitrarily complex encoder as used in previous analysis of affine decoder models). Now define  $\bar{\gamma}\triangleq \frac{1}{nd}\sum_{i}\| \pmb{x}^{(i)} - \bar{\pmb{x}}\| _2^2$ , with  $\bar{\pmb{x}}\triangleq \frac{1}{n}\sum_{i}\pmb{x}^{(i)}$ . We then have the following result:

Proposition 4.1 For any  $\alpha >0$ , there will always exist data sets  $X$  such (6) has a global minimum that perfectly reconstructs the training data, but also a bad local minimum characterized by

$$
q _ {\phi} (\boldsymbol {z} | \boldsymbol {x}) = \mathcal {N} (\boldsymbol {z} | \boldsymbol {0}, \boldsymbol {I}) \quad a n d \quad p _ {\theta} (\boldsymbol {x}) = \mathcal {N} (\boldsymbol {x} | \bar {\boldsymbol {x}}, \bar {\gamma} \boldsymbol {I}). \tag {7}
$$

Hence the moment we allow for nonlinear (or more precisely, non-affine) decoders there can exist a poor local minimum that exhibits category (v) posterior collapse. In other words, no predictive information about  $\pmb{x}$  passes through the latent space, and a useless/non-informative distribution  $p_{\theta}(\pmb{x})$  emerges that is incapable of assigning high probability to the data (except obviously in the trivial degenerate case where all the data points are equal to the empirical mean  $\bar{\pmb{x}}$ ). We will next investigate the degree to which such concerns can influence behavior in arbitrarily deep architectures.

# 5 EXTRAPOLATING TO PRACTICAL DEEP ARCHITECTURES

Previously we have demonstrated the possibility of local minima aligned with category (v) posterior collapse the moment we allow for decoders that deviate ever so slightly from an affine model. But nuanced counterexamples designed for proving technical results notwithstanding, it is reasonable to examine what realistic factors are largely responsible for leading optimization trajectories towards such potential bad local solutions. For example, is it merely the strength of the KL regularization term, and if so, why can we not just use KL warm-start to navigate around such points? In this section we will elucidate a deceptively simple, alternative risk factor that will be corroborated empirically in Section 6.

From the outset, we should mention that with deep encoder/decoder architectures commonly used in practice, a stationary point can more-or-less always exist at solutions exhibiting posterior collapse. As a representative and ubiquitous example, please see Appendix A.4. But of course without further details, this type of stationary point could conceivably manifest as a saddle point (stable or unstable), a local maximum, or a local minimum. For the strictly affine decoder model, there will only be a harmless unstable saddle point at any collapsed solution (the Hessian has negative eigenvalues), while for the special case described in Section 4 we can instead have a bad local minima if too many latent dimensions satisfy the stated conditions. We will now argue that as the depth of common feedforward architectures increases, the risk of converging to solutions with most or all latent dimensions stuck at bad stationary points analogous to those described can also increase.

Somewhat orthogonal to existing explanations of posterior collapse, our basis for this argument is not directly related to the VAE KL-divergence term. Instead, we consider a deceptively simple yet potentially influential alternative: Unregularized, deterministic deep AE models can have bad local solutions with high reconstruction errors that directly translate to category (v) posterior collapse when training a corresponding VAE model with a matching architecture. Moreover, to the extent that this is true, KL warm-start or related countermeasures will be helpless to prevent it.

To make this point more concrete, consider the deterministic AE model formed by concatenating the encoder mean  $\pmb{\mu}_x \equiv \pmb{\mu}_x(\cdot; \theta)$  and decoder mean  $\pmb{\mu}_z \equiv \pmb{\mu}_z(\cdot; \phi)$  networks from a VAE model, i.e., reconstructions  $\hat{\pmb{x}}$  are computed via  $\hat{\pmb{x}} = \pmb{\mu}_x[\pmb{\mu}_z(\pmb{x}; \phi); \theta]$ . We then train this AE to minimize

the squared-error loss  $\frac{1}{nd}\sum_{i=1}^{n}\left\|\pmb{x}^{(i)} - \hat{\pmb{x}}^{(i)}\right\|_2^2$ , producing parameters  $\{\theta_{ae}, \phi_{ae}\}$ . Analogously, the corresponding VAE trained to minimize (4) arrives at a parameter set denoted  $\{\theta_{vae}, \phi_{vae}\}$ . In this scenario, it will naturally follow that

$$
\frac {1}{n d} \sum_ {i = 1} ^ {n} \left\| \boldsymbol {x} ^ {(i)} - \boldsymbol {\mu} _ {x} \left[ \boldsymbol {\mu} _ {z} \left(\boldsymbol {x} ^ {(i)}; \phi_ {a e}\right); \theta_ {a e} \right] \right\| _ {2} ^ {2} \leq \frac {1}{n d} \sum_ {i = 1} ^ {n} \mathbb {E} _ {q _ {\phi_ {v a e}} (\boldsymbol {z} | \boldsymbol {x} ^ {(i)})} \left[ \| \boldsymbol {x} ^ {(i)} - \boldsymbol {\mu} _ {x} (\boldsymbol {z}; \theta_ {v a e}) \| _ {2} ^ {2} \right], \tag {8}
$$

meaning that the deterministic AE reconstruction error will generally be smaller than the stochastic VAE version. Note that if  $\sigma_z^2\to 0$ , the VAE defaults to the same deterministic encoder as the AE and hence will have identical representational capacity; however, the KL regularization prevents this from happening, and any  $\sigma_z^2 >0$  can only make the reconstructions worse.3 Likewise, the KL penalty factor  $\| \mu_z^2\| _2^2$  can further restrict the effective capacity and increase the reconstruction error of the training data.

We next define the set

$$
\mathcal {S} _ {\varepsilon} \triangleq \left\{\theta , \phi : \frac {1}{n d} \sum_ {i = 1} ^ {n} \left\| \boldsymbol {x} ^ {(i)} - \hat {\boldsymbol {x}} ^ {(i)} \right\| _ {2} ^ {2} \leq \varepsilon \right\} \tag {9}
$$

for any  $\epsilon > 0$ . Now suppose that the chosen encoder/decoder architecture is such that with high probability, achievable optimization trajectories (e.g., via SGD or related) lead to parameters  $\{\theta_{AE}, \phi_{AE}\} \notin S_{\varepsilon}$ , i.e.,  $\mathrm{Prob}(\{\theta_{AE}, \phi_{AE}\} \in S_{\varepsilon}) \approx 0$ . It then follows from (8) that the optimal VAE  $\gamma$  parameter, when conditioned on practically-achievable values for other network parameters, will have a lower bound given by

$$
\gamma^ {*} \triangleq \arg \min  _ {\gamma} \mathcal {L} \left(\theta_ {v a e} \backslash \gamma , \phi_ {v a e}\right) = \frac {1}{n d} \sum_ {i = 1} ^ {n} \mathbb {E} _ {q _ {\phi_ {v a e}}} \left(\boldsymbol {z} | \boldsymbol {x} ^ {(i)}\right) \left[ \| \boldsymbol {x} ^ {(i)} - \boldsymbol {\mu} _ {x} (\boldsymbol {z}; \theta_ {v a e}) \| _ {2} ^ {2} \right] \geq \varepsilon . \tag {10}
$$

The equality in (10) can be confirmed by simply differentiating the VAE cost and equating to zero, noting that  $\theta_{vae} \backslash \gamma$  is referencing all decoder parameters excluding  $\gamma$ .

From these observations, it becomes clear that the potential for category (v) posterior collapse arises when  $\varepsilon$  is large, which implies that the optimal  $\gamma^{*}$  must also be large per (10). To make this notion more explicit, it is helpful to introduce a slightly narrower but nonetheless representative class of VAE models.

Specifically, let  $f(\boldsymbol{\mu}_z, \boldsymbol{\sigma}_z, \theta, \boldsymbol{x}^{(i)}) \triangleq \mathbb{E}_{q_\phi(\boldsymbol{z} | \boldsymbol{x}^{(i)})} \left[ \| \boldsymbol{x}^{(i)} - \boldsymbol{\mu}_x(\boldsymbol{z}; \theta) \|_2^2 \right]$ , i.e., the VAE data term evaluated at a single data point without the  $1/\gamma$  scale factor. We then define a well-behaved VAE as a model with energy function (4) designed such that  $\nabla_{\mu_z} f(\boldsymbol{\mu}_z, \boldsymbol{\sigma}_z, \theta, \boldsymbol{x}^{(i)})$  and  $\nabla_{\sigma_z} f(\boldsymbol{\mu}_z, \boldsymbol{\sigma}_z, \theta, \boldsymbol{x}^{(i)})$  are Lipschitz continuous gradients for all  $i$ . Furthermore, we specify a nondegenerate decoder as any  $\boldsymbol{\mu}_x(\boldsymbol{z}; \theta = \tilde{\theta})$  with  $\theta$  set to a  $\tilde{\theta}$  value such that  $\nabla_{\sigma_z} f(\boldsymbol{\mu}_z, \boldsymbol{\sigma}_z, \tilde{\theta}, \boldsymbol{x}^{(i)}) \geq c$  for some constant  $c > 0$  that can be arbitrarily small. This ensures that  $f$  is an increasing function of  $\boldsymbol{\sigma}_z$ , a quite natural stipulation given that increasing the encoder variance will generally only serve to corrupt the reconstruction, unless of course the decoder is completely blocking the signal from the encoder. In the latter degenerate situation, it would follow that  $\nabla_{\mu_z} f(\boldsymbol{\mu}_z, \boldsymbol{\sigma}_z, \theta, \boldsymbol{x}^{(i)}) = \nabla_{\sigma_z} f(\boldsymbol{\mu}_z, \boldsymbol{\sigma}_z, \theta, \boldsymbol{x}^{(i)}) = 0$ , which is more-or-less tantamount to category (v) posterior collapse.

Based on these definitions, we can now present the following:

Proposition 5.1 For any well-behaved VAE with arbitrary, non-degenerate decoder  $\pmb{\mu}_x(z;\theta = \tilde{\theta})$ , there will always exist a  $\gamma' < \infty$  such that the trivial solution  $\pmb{\mu}_x(z;\theta \neq \tilde{\theta}) = \bar{\pmb{x}}$  and  $q_{\phi}(z|\pmb{x}) = p(z)$  will have lower cost.

Around any evaluation point, the sufficient condition we applied to demonstrate posterior collapse (see proof details) can also be achieved with some  $\gamma'' < \gamma'$  if we allow for partial collapse, i.e.,  $q_{\phi^*}(z_j|\pmb{x}) = p(z_j)$  along some but not all latent dimensions  $j \in \{1,\dots,\kappa\}$ . Overall, the analysis loosely suggests that the number of dimensions vulnerable to collapse will increase monotonically with  $\gamma$ .

Proposition 5.1 also provides evidence that the VAE behaves like a strict thresholding operator, completely shutting off latent dimensions using a finite value for  $\gamma$ . This is exactly analogous to the distinction between using the  $\ell_1$  versus  $\ell_2$  norm for solving regularized regression problems of the standard form  $\min_{\boldsymbol{u}} \| \boldsymbol{x} - \boldsymbol{A}\boldsymbol{u} \|_2^2 + \gamma \eta(\boldsymbol{u})$ , where  $\boldsymbol{A}$  is a design matrix and  $\eta$  is a penalty function. When  $\eta$  is the  $\ell_1$  norm, some or all elements of  $\boldsymbol{u}$  can be pruned to exactly zero with a sufficiently large but finite  $\gamma$  Zhao & Yu (2006). In contrast, when the  $\ell_2$  norm is applied, the coefficients will be shrunk to smaller values but never pushed all the way to zero unless  $\gamma \rightarrow \infty$ .

In aggregate then, if the AE base model displays unavoidably high reconstruction errors, this implicitly constrains the corresponding VAE model to have a large optimal  $\gamma$  value, which can potentially lead to undesirable posterior collapse per Proposition 5.1. In Section 6 we will demonstrate empirically that training unregularized AE models can become increasingly difficult and prone to bad local minima (or at least bad stable stationary points) as the depth increases; and this difficulty can persist even with counter-measures such as skip connections. Therefore, from this vantage point we would argue that it is actually the AE base architecture that is effectively the guilty party when it comes to posterior collapse.

The perspective described above also helps to explain why heuristics like KL warm-start are not always useful for improving VAE performance. With the standard Gaussian model (4) considered herein, KL warm-start amounts to adopting a pre-defined schedule for incrementally increasing  $\gamma$  starting from a small initial value, the motivation being that a small  $\gamma$  will steer optimization trajectories away from overregularized solutions and posterior collapse.

However, regardless of how arbitrarily small  $\gamma$  may be fixed at any point during this process, the VAE reconstructions are not likely to be better than the analogous deterministic AE (which is roughly equivalent to forcing  $\gamma = 0$  within the present context). This implies that there can exist an implicit  $\gamma^{*}$  as computed by (10) that can be significantly larger such that, even if KL warm-start is used, the optimization trajectory may well lead to a collapsed posterior stationary point that has this  $\gamma^{*}$  as the optimal value in terms of minimizing the VAE cost with other parameters fixed. Note that if posterior collapse does occur, the gradient from the KL term will equal zero and hence, to be at a stationary point it must be that the data term gradient is also zero, and therefore varying  $\gamma$  manually will not impact the gradient balance.

# 6 EMPIRICAL ASSESSMENTS

In this section we empirically demonstrate the association between bad AE local minima with high reconstruction errors and imminent VAE posterior collapse. For this purpose, we first train fully connected AE and VAE models with 1, 2, 4, 6, 8 and 10 hidden layers on the Fashion-MNIST dataset (Xiao et al., 2017). Each hidden layer is 512-dimensional and followed by ReLU activations (see Appendix A.1 for further details). The reconstruction error is shown in Figure 1(left). As the depth of the network increases, the reconstruction error of the AE model first decreases because of the increased capacity. However, when the network becomes too deep, the error starts to increase, indicating convergence to a bad local minima (or at least stable stationary point/plateau) that is unrelated to KL-divergence regularization. The reconstruction error of a VAE model is always worse than that of the corresponding AE model as expected.

We next train AE and VAE models using a more complex convolutional network on Cifar100 data (Krizhevsky & Hinton, 2009). At each spatial scale, we use 1 to 5 convolution layers followed by ReLU activations. We also apply  $2 \times 2$  max pooling to downsample the feature maps to a smaller spatial scale in the encoder and use a transposed convolution layer to upscale the feature map in the decoder. The reconstruction errors are shown in Figure 1(middle). Again, the trend is similar to the fully-connected network results. See Appendix A.1 for an additional ImageNet example.

It has been argued in the past that skip connections can increase the mutual information between observations  $\pmb{x}^{(i)}$  and the inferred latent variables  $z$  (Dieng et al., 2018). And it is well-known that ResNet architectures based on skip connections can improve performance on numerous recognition tasks (He et al., 2016). To this end, we train a number of AE models using ResNet-inspired encoder/decoder architectures on multiple datasets including Cifar10, Cifar100, SVHN and CelebA. Similar to the convolution network structure from above, we use 1, 2, and 4 residual blocks within each spatial scale. Inside each block, we apply 2 to 5 convolution layers. For aggregate comparison purposes, we normalize the reconstruction error obtained on each dataset by dividing it with the

![](images/0c0c04f091a61b6d862995cbc1b23d1d6207621df12ea6d195f964b80a1e6be3.jpg)  
Figure 1: Reconstruction errors for various encoder/decoder models of varying complexity. Left: Fully connected networks with different depths trained on Fashion-MNIST. Middle: Convolution networks with increasing depth/# of spatial scales trained on Cifar100. Right: Averaged AE results from residual networks with varying number of residual blocks and block depth trained on SVHN, Cifar10, Cifar100 and CelebA. In all plots, once the encoder/decoder complexity is sufficiently high, the reconstruction errors begin to increase.

![](images/e33e11cc7dadff429c4acf40697ebd9715cdd36f2a293db1edd44537af6a3089.jpg)

![](images/b2ee5a643ac9bb5f3725eb6510de02d7f3bbccc3ba662035f2f18ad1976a0e5b.jpg)

![](images/a03d072052aad0242df13a03731daa1fac7206cc549b6d490f569732452a88c2.jpg)  
Figure 2: Histogram of  $\sigma_z$  values as VAE encoder/decoder network depth is varied. There are 2, 4 and 5 convolution layers in each spatial scale from left to right. As depth increases, the reconstruction error grows and more  $\sigma_z$  values are near 1, indicative of impending posterior collapse.

![](images/d4787ee96caf5951965e78885f2a80de13cd45db590fb9480194317efaa37c79.jpg)

![](images/7254d6109b9319bbe2069fe95be01dee28f6aaaf56b256e44bdaeebf3e8bd51e.jpg)

corresponding error produced by the most shallow network structure (1 residual block with 2 convolution layers). We then average the normalized reconstruction errors over all four datasets. The average normalized errors are shown in Figure 1(right), where we observe that adding more convolution layers inside each residual block can increase the reconstruction error when the network is too deep. Moreover, adding more residual blocks can also lead to higher reconstruction errors.

We emphasize that in all these models, as the network complexity/depth increases, the simpler models are always contained within the capacity of the larger ones. Therefore, because the reconstruction error on the training data is becoming worse, it must be the case that the AE is becoming stuck at bad local minima or plateaus. Again since the AE reconstruction error serves as a lower bound for that of the VAE model, a deeper VAE model will likely suffer the same problem, only exacerbated by the KL-divergence term in the form of posterior collapse. This implies that there will be more  $\sigma_z$  values moving closer to 1 as the VAE model becomes deeper; similarly  $\mu_z$  values will push towards 0. The corresponding dimensions will encode no information and become completely useless.

To help corroborate this association between bad AE local minima and VAE posterior collapse, we plot histograms of VAE  $\sigma_z$  values as network depth is varied in Figure 2. The models are trained on CelebA and the number of convolution layers in each spatial scale is 2, 4 and 5 from left to right. As the depth increases, the reconstruction error becomes larger and there are more  $\sigma_z$  near 1.

# 7 CONCLUSIONS

In this work we have emphasized the previously-underappreciated role of bad local minima (particularly those shared by deterministic AEs and unrelated to KL-divergence regularization) in trapping VAE models at posterior collapsed solutions. While we believe that this message is interesting in and of itself, there are nonetheless several practically-relevant implications. For example, complex hierarchical VAEs like BIVA notwithstanding, skip connections and KL warm-start have modest ability to steer optimization trajectories towards good solutions; however, this limitation may not manifest until networks are sufficiently deep as we have considered. Fortunately, any advances or insights gleaned from developing deeper unregularized AEs, e.g., better initializations (Li & Nguyen, 2019), could likely be adapted to reduce the risk of posterior collapse in corresponding VAE models.

# REFERENCES

A. Alemi, B. Poole, I. Fischer, J. Dillon, R. Saurous, and K. Murphy. Fixing a broken ELBO. arXiv preprint arXiv:1711.00464, 2017.  
M. Bauer and A. Mnih. Resampled priors for variational autoencoders. arXiv preprint arXiv:1810.11428, 2018.  
S. Bowman, L. Vilnis, O. Vinyals, A. Dai, R. Jozefowicz, and S. Bengio. Generating sentences from a continuous space. arXiv preprint arXiv:1511.06349, 2015.  
Y. Burda, R. Grosse, and R. Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
L. Cai, H. Gao, and S. Ji. Multi-stage variational auto-encoders for coarse-to-fine image generation. arXiv preprint arXiv:1705.07202, 2017.  
E. Candès, X. Li, Y. Ma, and J. Wright. Robust principal component analysis? J. ACM, 58(2), 2011.  
X. Chen, D. Kingma, T. Salimans, Y. Duan, P. Dhariwal, J. Schulman, I. Sutskever, and P. Abbeel. Variational lossy autoencoder. arXiv preprint arXiv:1611.02731, 2016.  
B. Dai and D. Wipf. Diagnosing and enhancing VAE models. International Conference on Learning Representations, 2019.  
B. Dai, Y. Wang, J. Aston, G. Hua, and D. Wipf. Hidden talents of the variational autoencoder. arXiv preprint arXiv:1706.05148, 2018.  
A. Dieng, Y. Kim, A. Rush, and D. Blei. Avoiding latent variable collapse with generative skip models. arXiv preprint arXiv:1807.04863, 2018.  
K. Gregor and Y. LeCun. Learning fast approximations of sparse coding. International Conference on Machine Learning, 2010.  
J. He, D. Spokoyny, G. Neubig, and T. Berg-Kirkpatrick. Lagging inference networks and posterior collapse in variational autoencoders. International Conference on Learning Representations, 2019.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CVPR, 2016.  
I. Higgins, L. Matthew, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, , and A. Lerchner.  $\beta$ -vae: Learning basic visual concepts with a constrained variational framework. International Conference on Learning Representations, 2017.  
C. Huang, S. Tan, A. Lacoste, and A. Courville. Improving explorability in variational inference with annealed variational objectives. Advances in Neural Information Processing Systems, 2018.  
K. Kawaguchi. Deep learning without poor local minima. Advances in Neural Information Processing Systems, 2016.  
D. Kingma and M. Welling. Auto-encoding variational Bayes. International Conference on Learning Representations, 2014.  
A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
P. Li and P.M. Nguyen. On random deep weight-tied autoencoders: Exact asymptotic analysis, phase transitions, and implications to training. International Conference on Learning Representations, 2019.  
J. Lucas, G. Tucker, R. Grosse, and M. Norouzi. Understanding posterior collapse in generative latent variable models. International Conference on Learning Representations, Workshop Paper, 2019.  
Lars Maalège, Marco Fraccaro, Valentin Lievin, and Ole Winther. BIVA: A very deep hierarchy of latent variables for generative modeling. arXiv preprint arXiv:1902.02102, 2019.

P.A. Mattei and J. Frellsen. Leveraging the exact likelihood of deep latent variable models. In Advances in Neural Information Processing Systems, 2018.  
E. Orjebin. A recursive formula for the moments of a truncated univariate normal distribution. 2014. URL https://people.smp.uq.edu.au/YoniNazarathy/teachingProjects/studentWork/EricOrjebin-TruncatedNormalMoments.pdf.  
A. Razavi, A. Oord, B. Poole, and O. Vinyals. Preventing posterior collapse with  $\delta$ -VAEs. International Conference on Learning Representations, 2019.  
D. Rezende and S. Mohamed. Variational inference with normalizing flows. arXiv preprint arXiv:1505.05770, 2015.  
D. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. International Conference on Machine Learning, 2014.  
M. Rolinek, D. Zietlow, and G. Martius. Variational autoencoders pursue PCA directions (by accident). 2019.  
C. Sønderby, T. Raiko, L. Maaløe, S. Sønderby, and O. Winther. How to train deep variational autoencoders and probabilistic ladder networks. arXiv preprint arXiv:1602.02282, 2016.  
P. Sprechmann, A.M. Bronstein, and G. Sapiro. Learning efficient sparse and low rank models. IEEE Trans. Pattern Analysis and Machine Intelligence, 37(9), 2015.  
M. Tipping and C. Bishop. Probabilistic principal component analysis. J. Royal Statistical Society, Series B, 61(3):611-622, 1999.  
J. Tomczak and M. Welling. VAE with a VampPrior. International Conference on Artificial Intelligence and Statistics, 2018.  
A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves, and K. Kavukcuoglu. Conditional image generation with PixelCNN decoders. Advances in Neural Information Processing Systems, 2016.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
S. Yeung, A. Kannan, Y. Dauphin, and L. Fei-Fei. Tackling over-pruning in variational autoencoders. arXiv preprint arXiv:1706.03643, 2017.  
C. Yun, S. Sra, and A. Jabbabaie. Small nonlinearities in activation functions create bad local minima in neural networks. International Conference on Learning Representations, 2019.  
P. Zhao and B. Yu. On model selection consistency of Lasso. Journal of Machine learning research, 7:2541-2563, 2006.
