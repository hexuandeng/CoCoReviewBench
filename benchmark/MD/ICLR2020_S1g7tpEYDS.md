# FROM VARIATIONAL TO DETERMINISTIC AUTOENCODERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Variational Autoencoders (VAEs) provide a theoretically-backed and popular framework for deep generative models. However, learning a VAE from data poses still unanswered theoretical questions and considerable practical challenges. In this work, we propose an alternative framework for generative modeling that is simpler, easier to train, and deterministic, yet has many of the advantages of the VAE. We observe that sampling a stochastic encoder in a Gaussian VAE can be interpreted as simply injecting noise into the input of a deterministic decoder. We investigate how substituting this kind of stochasticity, with other explicit and implicit regularization schemes, can lead to an equally smooth and meaningful latent space without having to force it to conform to an arbitrarily chosen prior. To retrieve a generative mechanism to sample new data points, we introduce an ex-post density estimation step that can be readily applied to the proposed framework as well as existing VAEs, improving their sample quality. We show, in a rigorous empirical study, that the proposed regularized deterministic autoencoders are able to generate samples that are comparable to, or better than, those of VAEs and more powerful alternatives when applied to images as well as to structured data such as molecules.

# 1 INTRODUCTION

Generative models lie at the core of machine learning. By capturing the mechanisms behind the data generation process, one can reason about data probabilistically, access and traverse the low-dimensional manifold the data is assumed to live on, and ultimately generate new data. It is therefore not surprising that generative models have gained momentum in applications such as computer vision (Sohn et al., 2015; Brock et al., 2019), NLP (Bowman et al., 2016; Severyn et al., 2017), and chemistry (Kusner et al., 2017; Jin et al., 2018; Gómez-Bombarelli et al., 2018).

Variational Autoencoders (VAEs) (Kingma & Welling, 2014; Rezende et al., 2014) cast learning representations for high-dimensional distributions as a variational inference problem. Learning a VAE amounts to the optimization of an objective balancing the quality of samples that are autoencoded through a stochastic encoder-decoder pair while encouraging the latent space to follow a fixed prior distribution. Since their introduction, VAEs have become one of the frameworks of choice among the different generative models. VAEs promise theoretically well-founded and more stable training than Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) and more efficient sampling mechanisms than autoregressive models (Larochelle & Murray, 2011; Germain et al., 2015).

However, the VAE framework is still far from delivering the promised generative mechanism, as there are several practical and theoretical challenges yet to be solved. A major weakness of VAEs is the tendency to strike an unsatisfying compromise between sample quality and reconstruction quality. In practice, this has been attributed to overly simplistic prior distributions (Tomczak & Welling, 2018; Dai & Wipf, 2019) or alternatively, to the inherent over-regularization induced by the KL divergence term in the VAE objective (Tolstikhin et al., 2017). Most importantly, the VAE objective itself poses several challenges as it admits trivial solutions that decouple the latent space from the input (Chen et al., 2017; Zhao et al., 2017), leading to the posterior collapse phenomenon in conjunction with powerful decoders (van den Oord et al., 2017). Furthermore, due to its variational formulation, training a VAE requires approximating expectations through sampling at the cost of increased variance in gradients (Burda et al., 2015; Tucker et al., 2017), making initialization, validation, and

annealing of hyperparameters essential in practice (Bowman et al., 2016; Higgins et al., 2017; Bauer & Mnih, 2019). Lastly, even after a satisfactory convergence of the objective, the learned aggregated posterior distribution rarely matches the assumed latent prior in practice (Kingma et al., 2016; Bauer & Mnih, 2019; Dai & Wipf, 2019), ultimately hurting the quality of generated samples. All in all, much of the attention around VAEs is still directed towards "fixing" the aforementioned drawbacks associated with them.

In this work, we take a different route: we question whether the variational framework adopted by VAEs is necessary for generative modeling and, in particular, to obtain a smooth latent space. We propose to adopt a simpler, deterministic version of VAEs that scales better, is simpler to optimize, and, most importantly, still produces a meaningful latent space and equivalently good or better samples than VAEs or stronger alternatives, e.g. Wasserstein Autoencoders (WAEs) (Tolstikhin et al., 2017). We do so by observing that, under commonly used distributional assumptions, training a stochastic encoder-decoder pair in VAEs does not differ from training a deterministic architecture where noise is added to the decoder's input. We investigate how to substitute this noise injection mechanism with other regularization schemes in the proposed deterministic Regularized Autoencoders (RAEs), and we thoroughly analyze how this affects performance. Finally, we equip RAEs with a generative mechanism via a simple ex-post density estimation step on the learned latent space.

In summary, our contributions are as follows: i) we introduce the RAE framework for generative modeling as a drop-in replacement for many common VAE architectures; ii) we propose an ex-post density estimation scheme which greatly improves sample quality for VAEs, WAEs and RAEs without the need to retrain the models; iii) we conduct a rigorous empirical evaluation to compare RAEs with VAEs and several baselines on standard image datasets and on more challenging structured domains such as molecule generation (Kusner et al., 2017; Gómez-Bombarelli et al., 2018).

# 2 VARIATIONAL AUTOENCODERS

For a general discussion, we consider a collection of high-dimensional i.i.d. samples  $\mathcal{X} = \{\mathbf{x}_i\}_{i=1}^N$  drawn from the true data distribution  $p_{\mathrm{data}}(\mathbf{x})$  over a random variable  $\mathbf{X}$  taking values in the input space. The aim of generative modeling is to learn from  $\mathcal{X}$  a mechanism to draw new samples  $\mathbf{x}_{\mathrm{new}} \sim p_{\mathrm{data}}$ . Variational Autoencoders provide a powerful latent variable framework to infer such a mechanism. The generative process of the VAE is defined as

$$
\mathbf {z} _ {\text {n e w}} \sim p (\mathbf {Z}), \quad \mathbf {x} _ {\text {n e w}} \sim p _ {\theta} (\mathbf {X} \mid \mathbf {Z} = \mathbf {z} _ {\text {n e w}}) \tag {1}
$$

where  $p(\mathbf{Z})$  is a fixed prior distribution over a low-dimensional latent space  $\mathbf{Z}$ . A stochastic decoder

$$
D _ {\theta} (\mathbf {z}) = \mathbf {x} \sim p _ {\theta} (\mathbf {x} \mid \mathbf {z}) = p (\mathbf {X} \mid g _ {\theta} (\mathbf {z})) \tag {2}
$$

links the latent space to the input space through the likelihood distribution  $p_{\theta}$ , where  $g_{\theta}$  is an expressive non-linear function parameterized by  $\theta$ . As a result, a VAE estimates  $p_{\mathrm{data}}(\mathbf{x})$  as the infinite mixture model  $p_{\theta}(\mathbf{x}) = \int p_{\theta}(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$ . At the same time, the input space is mapped to the latent space via a stochastic encoder

$$
E _ {\phi} (\mathbf {x}) = \mathbf {z} \sim q _ {\phi} (\mathbf {z} \mid \mathbf {x}) = q (\mathbf {Z} \mid f _ {\phi} (\mathbf {x})) \tag {3}
$$

where  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is the posterior distribution given by a second function  $f_{\phi}$  parameterized by  $\phi$ . Computing the marginal log-likelihood  $\log p_{\theta}(\mathbf{x})$  is generally intractable. One therefore follows a variational approach, maximizing the evidence lower bound (ELBO) for a sample  $\mathbf{x}$ :

$$
\log p _ {\theta} (\mathbf {x}) \geq \operatorname {E L B O} (\phi , \theta , \mathbf {x}) = \mathbb {E} _ {\mathbf {z} \sim q _ {\phi} (\mathbf {z} \mid \mathbf {x})} \log p _ {\theta} (\mathbf {x} \mid \mathbf {z}) - \mathbb {K L} (q _ {\phi} (\mathbf {z} \mid \mathbf {x}) | | p (\mathbf {z})) \tag {4}
$$

Maximizing Eq. 4 over data  $\mathcal{X}$  w.r.t. model parameters  $\phi, \theta$  corresponds to minimizing the loss

$$
\underset {\phi , \theta} {\arg \min } \mathbb {E} _ {\mathbf {x} \sim p _ {\text {d a t a}}} \mathcal {L} _ {\text {EL B O}} = \mathbb {E} _ {\mathbf {x} \sim p _ {\text {d a t a}}} \mathcal {L} _ {\text {R E C}} + \mathcal {L} _ {\text {K L}} \tag {5}
$$

where  $\mathcal{L}_{\mathrm{REC}}$  and  $\mathcal{L}_{\mathrm{KL}}$  are defined for a sample  $\mathbf{x}$  as follows:

$$
\mathcal {L} _ {\mathrm {R E C}} = - \mathbb {E} _ {\mathbf {z} \sim q _ {\phi} (\mathbf {z} \mid \mathbf {x})} \log p _ {\theta} (\mathbf {x} \mid \mathbf {z}) \quad \mathcal {L} _ {\mathrm {K L}} = \mathbb {K L} \left(q _ {\phi} (\mathbf {z} \mid \mathbf {x}) \| p (\mathbf {z})\right) \tag {6}
$$

Intuitively, the reconstruction loss  $\mathcal{L}_{\mathrm{REC}}$  takes into account the quality of autoencoded samples  $\mathbf{x}$  through  $D_{\theta}(E_{\phi}(\mathbf{x}))$ , while the KL-divergence term  $\mathcal{L}_{\mathrm{KL}}$  encourages  $q_{\phi}(\mathbf{z}|\mathbf{x})$  to match the prior  $p(\mathbf{z})$  for each  $\mathbf{z}$  which acts as a regularizer during training (Hoffman & Johnson, 2016).

# 2.1 PRACTICE AND SHORTCOMINGS OF VAES

To fit a VAE to data through Eq. 5 one has to specify the parametric forms for  $p(\mathbf{z})$ ,  $q_{\phi}(\mathbf{z}|\mathbf{x})$ ,  $p_{\theta}(\mathbf{x}|\mathbf{z})$ , and hence the deterministic mappings  $f_{\phi}$  and  $g_{\theta}$ . In practice, the choice for the above distributions is guided by trading off computational complexity with model expressiveness. In the most commonly adopted formulation of the VAE,  $q_{\phi}(\mathbf{z}|\mathbf{x})$  and  $p_{\theta}(\mathbf{x}|\mathbf{z})$  are assumed to be Gaussian:

$$
E _ {\phi} (\mathbf {x}) \sim \mathcal {N} (\mathbf {Z} | \boldsymbol {\mu} _ {\phi} (\mathbf {x}), \operatorname {d i a g} \left(\boldsymbol {\sigma} _ {\phi} (\mathbf {x}))\right) \quad D _ {\theta} \left(E _ {\phi} (\mathbf {x})\right) \sim \mathcal {N} (\mathbf {X} | \boldsymbol {\mu} _ {\theta} (\mathbf {z}), \operatorname {d i a g} \left(\boldsymbol {\sigma} _ {\theta} (\mathbf {z}))\right) \tag {7}
$$

with means  $\mu_{\phi}, \mu_{\theta}$  and covariance parameters  $\sigma_{\phi}, \sigma_{\theta}$  given by  $f_{\phi}$  and  $g_{\theta}$ . In practice, the covariance of the decoder is set to the identity matrix for all  $\mathbf{z}$ , i.e.  $\sigma_{\theta}(\mathbf{z}) = 1$  (Dai & Wipf, 2019). The expectation of  $\mathcal{L}_{\mathrm{REC}}$  in Eq. 6 must be approximated via  $k$  Monte Carlo point estimates. It is expected that the quality of the Monte Carlo estimate, and hence convergence during learning and sample quality increases for larger  $k$  (Burda et al., 2015). However, only a 1-sample approximation is generally carried out (Kingma & Welling, 2014) since memory and time requirements are prohibitive for large  $k$ . With the 1-sample approximation,  $\mathcal{L}_{\mathrm{REC}}$  can be computed as the mean squared error between input samples and their mean reconstructions  $\mu_{\theta}$  by a decoder that is deterministic in practice:

$$
\mathcal {L} _ {\mathrm {R E C}} = \left| \left| \mathbf {x} - \boldsymbol {\mu} _ {\theta} \left(E _ {\phi} (\mathbf {x})\right) \right| \right| _ {2} ^ {2} \tag {8}
$$

Gradients w.r.t. the encoder parameters  $\phi$  are computed through the expectation of  $\mathcal{L}_{\mathrm{REC}}$  in Eq. 6 via the reparametrization trick (Kingma & Welling, 2014) where the stochasticity of  $E_{\phi}$  is relegated to an auxiliary random variable  $\epsilon$  which does not depend on  $\phi$ :

$$
E _ {\phi} (\mathbf {x}) = \boldsymbol {\mu} _ {\phi} (\mathbf {x}) + \boldsymbol {\sigma} _ {\phi} (\mathbf {x}) \odot \boldsymbol {\epsilon}, \quad \boldsymbol {\epsilon} \sim \mathcal {N} (\mathbf {0}, \mathbf {I}) \tag {9}
$$

where  $\odot$  denotes the Hadamard product. An additional simplifying assumption involves fixing the prior  $p(\mathbf{z})$  to be a  $d$ -dimensional isotropic Gaussian  $\mathcal{N}(\mathbf{Z} \mid \mathbf{0}, \mathbf{I})$ . For this choice, the KL-divergence for a sample  $\mathbf{x}$  is given in closed form:  $2\mathcal{L}_{\mathrm{KL}} = ||\boldsymbol{\mu}_{\phi}(\mathbf{x})||_2^2 + d + \sum_i^d \sigma_{\phi}(\mathbf{x})_i - \log \sigma_{\phi}(\mathbf{x})_i$ .

While the above assumptions make VAEs easy to implement, the stochasticity in the encoder and decoder are still problematic in practice (Makhzani et al., 2016; Tolstikhin et al., 2017; Dai & Wipf, 2019). In particular, one has to carefully balance the trade-off between the  $\mathcal{L}_{\mathrm{KL}}$  term and  $\mathcal{L}_{\mathrm{REC}}$  during optimization (Dai & Wipf, 2019; Bauer & Mnih, 2019). A too-large weight on the  $\mathcal{L}_{\mathrm{KL}}$  term can dominate  $\mathcal{L}_{\mathrm{ELBO}}$ , having the effect of over-regularization. As this would smooth the latent space, it can directly affect sample quality in a negative way. Heuristics to avoid this include manually finetuning or gradually annealing the importance of  $\mathcal{L}_{\mathrm{KL}}$  during training (Bowman et al., 2016; Bauer & Mnih, 2019). We also observe this trade-off in a practical experiment in Appendix A.

Even after employing the full array of approximations and "tricks" to reach convergence of Eq. 5 for a satisfactory set of parameters, there is no guarantee that the learned latent space is distributed according to the assumed prior distribution. In other words, the aggregated posterior distribution  $q_{\phi}(\mathbf{z}) = \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}q(\mathbf{z}|\mathbf{x})$  has been shown not to conform well to  $p(\mathbf{z})$  after training (Tolstikhin et al., 2017; Bauer & Mnih, 2019; Dai & Wipf, 2019). This critical issue severely hinders the generative mechanism of VAEs (Eq. 1) since latent codes sampled from  $p(\mathbf{z})$  (instead of  $q(\mathbf{z})$ ) might lead to regions of the latent space that are previously unseen to  $D_{\theta}$  during training. The result is out-of-distribution samples. We analyze solutions to this problem in Section 4.

# 2.2 CONSTANT-VARIANCE ENCODERS

Before introducing our fully-deterministic take on VAEs, it is worth investigating intermediate flavors of VAEs with reduced stochasticity. Analogous to what is commonly done for decoders as discussed in the previous section, one can fix the variance of  $q_{\phi}(\mathbf{z} \mid \mathbf{x})$  to be constant for all  $\mathbf{x}$ . This simplifies the computation of  $E_{\phi}$  from Eq. 9 to

$$
E _ {\phi} ^ {C V} (\mathbf {x}) = \boldsymbol {\mu} _ {\phi} (\mathbf {x}) + \boldsymbol {\epsilon}, \quad \epsilon \sim \mathcal {N} (\mathbf {0}, \sigma \mathbf {I}) \tag {10}
$$

where  $\sigma$  is a fixed scalar. Then, the KL loss term in a Gaussian VAE simplifies (up to a constant) to  $\mathcal{L}_{\mathrm{KL}}^{\mathrm{CV}} = ||\pmb{\mu}_{\phi}(\mathbf{x})||_2^2$ . We name this variant Constant-Variance VAEs (CV-VAEs). While CV-VAEs have been adopted in some applications such as variational image compression (Balle et al., 2017) and adversarial robustness (Ghosh et al., 2019), to the best of our knowledge, there is no systematic study of them in the literature. We will fill this gap in our experiments in Section 6. Lastly, treating  $\sigma_{\phi}$  as a constant impairs the assumption of  $p(\mathbf{z})$  to be an isotropic Gaussian. We address this distributional mismatch in a general way in Section 4 by providing a more complex and flexible prior structure over  $\mathbf{Z}$  via ex-post density estimation.

# 3 DETERMINISTIC REGULARIZED AUTOENCODERS

Autoencoding in VAEs is defined in a probabilistic fashion:  $E_{\phi}$  and  $D_{\theta}$  map data points not to a single point, but rather to parameterized distributions (cf. Eq. 7). However, common implementations of VAEs as discussed in Section 2 admit a simpler, deterministic view for this probabilistic mechanism. A glance at the autoencoding mechanism of the VAE is revealing.

The encoder maps a data point  $\mathbf{x}$  to a mean  $\mu_{\phi}(\mathbf{x})$  and variance  $\sigma_{\phi}(\mathbf{x})$  in the latent space via the reparametrization trick (cf. Eq. 9). The input to the decoder is then simply the mean  $\mu_{\phi}(\mathbf{x})$  augmented with random Gaussian noise scaled by  $\sigma_{\phi}(\mathbf{x})$ . In the CV-VAE, this relationship is even more obvious, as the magnitude of the noise is fixed for all data points (cf. Eq. 10). In this light, a VAE can be seen as a deterministic autoencoder where (Gaussian) noise is added to the decoder's input.

We argue that this noise injection mechanism is a key factor in having a regularized decoder. Using random noise injection to regularize neural networks is a well-known technique that dates back several decades (Sietsma & Dow, 1991; An, 1996). It implicitly helps to smooth the function learned by the network at the price of increased variance in the gradients during training. In turn, decoder regularization is a key component in generalization for VAEs, as it improves random sample quality and achieves a smoother latent space. Indeed, from a generative perspective, regularization is motivated by the goal to learn a smooth latent space where similar data points  $\mathbf{x}$  are mapped to similar latent codes  $\mathbf{z}$ , and small variations in  $\mathbf{Z}$  lead to reconstructions by  $D_{\theta}$  that vary only slightly.

We propose to substitute noise injection with an explicit regularization scheme for the decoder. This entails the substitution of the variational framework in VAEs, which enforces regularization on the encoder posterior through  $\mathcal{L}_{\mathrm{KL}}$ , with a deterministic framework that applies other flavors of decoder regularization. By removing noise injection from a CV-VAE, we are effectively left with a deterministic autoencoder (AE). Coupled with explicit regularization for the decoder, we obtain a Regularized Autoencoder (RAE). Training a RAE thus involves minimizing the simplified loss

$$
\mathcal {L} _ {\mathrm {R A E}} = \mathcal {L} _ {\mathrm {R E C}} + \beta \mathcal {L} _ {\mathbf {Z}} ^ {\mathrm {R A E}} + \lambda \mathcal {L} _ {\mathrm {R E G}} \tag {11}
$$

where  $\mathcal{L}_{\mathrm{REG}}$  represents the explicit regularizer for  $D_{\theta}$  (discussed in Section 3.1) and  $\mathcal{L}_{\mathbf{Z}}^{\mathrm{RAE}} = 1/2||\mathbf{z}||_2^2$  (resulting from simplifying  $\mathcal{L}_{\mathrm{KL}}^{\mathrm{CV}}$ ) is equivalent to constraining the size of the learned latent space, which is still needed to prevent unbounded optimization. Finally,  $\beta$  and  $\lambda$  are two hyper parameters that balance the different loss terms.

Note that for RAEs, no Monte Carlo approximation is required to compute  $\mathcal{L}_{\mathrm{REC}}$ . This relieves the need for more samples from  $q_{\phi}(\mathbf{z}|\mathbf{x})$  to achieve better image quality (cf. Appendix A). Moreover, by abandoning the variational framework and the  $\mathcal{L}_{\mathrm{KL}}$  term, there is no need in RAEs for a fixed prior distribution over  $\mathbf{Z}$ . Doing so however loses a clear generative mechanism for RAEs to sample from  $\mathbf{Z}$ . We propose a method to regain random sampling ability in Section 4 by performing density estimation on  $\mathbf{Z}$  ex-post, a step that is otherwise still needed for VAEs to alleviate the posterior mismatch issue.

# 3.1 REGULARIZATION SCHEMES FOR RAES

Among possible choices for a mechanism to use for  $\mathcal{L}_{\mathrm{REG}}$ , a first obvious candidate is Tikhonov regularization (Tikhonov & Arsenin, 1977) since it is known to be related to the addition of low-magnitude input noise (Bishop, 2006). Training a RAE within this framework thus amounts to adopting  $\mathcal{L}_{\mathrm{REG}} = \mathcal{L}_{\mathrm{L}_2} = ||\theta||_2^2$  which effectively applies weight decay on the decoder parameters  $\theta$ . Another option comes from the recent GAN literature where regularization is a hot topic (Kurach et al., 2018) and where injecting noise to the input of the adversarial discriminator has led to improved performance in a technique called instance noise (Sonderby et al., 2017). To enforce Lipschitz continuity on adversarial discriminators, weight clipping has been proposed (Arjovsky et al., 2017), which is however known to significantly slow down training. More successfully, a gradient penalty on the discriminator can be used similar to Gulrajani et al. (2017); Mescheder et al. (2018), yielding the objective  $\mathcal{L}_{\mathrm{REG}} = \mathcal{L}_{\mathrm{GP}} = ||\nabla D_{\theta}(E_{\phi}(\mathbf{x}))||_2^2$  which bounds the gradient norm of the decoder w.r.t. its input. Additionally, spectral normalization (SN) has been successfully proposed as an alternative way to bound the Lipschitz norm of an adversarial discriminator (Miyato et al., 2018). SN normalizes each weight matrix  $\theta_{\ell}$  in the decoder by an estimate of its largest singular value:  $\theta_{\ell}^{\mathrm{SN}} = \theta_{\ell} / s(\theta_{\ell})$  where  $s(\theta_{\ell})$  is the current estimate obtained through the power method.

In light of the recent successes of deep networks without explicit regularization (Zagoruyko & Komodakis, 2016; Zhang et al., 2017), it is intriguing to question the need for explicit regularization of the decoder in order to obtain a meaningful latent space. The assumption here is that techniques such as dropout (Srivastava et al., 2014), batch normalization (Ioffe & Szegedy, 2015), adding noise during training (An, 1996) implicitly regularize the networks enough. Therefore, as a natural baseline to the  $\mathcal{L}_{\mathrm{RAE}}$  objectives introduced above, we also consider the RAE framework without  $\mathcal{L}_{\mathrm{REG}}$  and  $\mathcal{L}_{\mathbf{Z}}^{\mathrm{RAE}}$ , i.e. a standard deterministic autoencoder optimizing  $\mathcal{L}_{\mathrm{REC}}$  only.

Lastly, it is worth questioning if it is possible to formally derive our RAE framework. We answer this affirmatively, and show how to augment the ELBO optimization problem of a VAE with an explicit constraint, while not fixing a parametric form for  $q_{\phi}(\mathbf{z} \mid \mathbf{x})$ . This indeed leads to a special case of the RAE loss in Eq. 11. Specifically, we derive a regularizer like  $\mathcal{L}_{\mathrm{GP}}$  for a deterministic version of the CV-VAE. We accommodate the full proof in Appendix B.

# 4 EX-POST DENSITY ESTIMATION

By removing stochasticity and ultimately, the KL divergence term  $\mathcal{L}_{\mathrm{KL}}$  from RAEs, we have simplified the original VAE objective at the cost of detaching the encoder from the prior  $p(\mathbf{z})$  over the latent space. This implies that i) we cannot ensure that the latent space  $\mathbf{Z}$  is distributed according to a simple distribution (e.g. isotropic Gaussian) anymore and consequently, ii) we lose the simple mechanism provided by  $p(\mathbf{z})$  to sample from  $\mathbf{Z}$  as in Eq. 1.

As discussed in Section 2.1, issue i) is compromising the VAE framework in any case, as reported in several works (Hoffman & Johnson, 2016; Rosca et al., 2018; Dai & Wipf, 2019). To fix this, some works extend the VAE objective by encouraging the aggregated posterior to match  $p(\mathbf{z})$  (Tolstikhin et al., 2017) or by utilizing more complex priors (Kingma et al., 2016; Tomczak & Welling, 2018; Bauer & Mnih, 2019).

To overcome both i) and ii), we instead propose to employ ex-post density estimation over  $\mathbf{Z}$ . We fit a density estimator denoted as  $q_{\delta}(\mathbf{z})$  to  $\{\mathbf{z} = E_{\phi}(\mathbf{x})|\mathbf{x}\in \mathcal{X}\}$ . This simple approach not only fits our RAE framework well, but it can also be readily adopted for any VAE or variants thereof such as the WAE as a practical remedy to the aggregated posterior mismatch without adding any computational overhead to the costly training phase.

The choice of  $q_{\delta}(\mathbf{z})$  needs to trade-off expressiveness – to provide a good fit of an arbitrary space for  $\mathbf{Z}$  – with simplicity, to improve generalization. For example, placing a Dirac distribution on each latent point  $\mathbf{z}$  would allow the decoder to output only training sample reconstructions which have a high quality, but do not generalize. Striving for simplicity, we employ and compare a full covariance multivariate Gaussian with a 10-component Gaussian mixture model (GMM) in our experiments.

# 5 RELATED WORKS

Many works have focused on diagnosing the VAE framework, the terms in its objective (Hoffman & Johnson, 2016; Zhao et al., 2017; Alemi et al., 2018), and ultimately augmenting it to solve optimization issues (Rezende & Viola, 2018; Dai & Wipf, 2019). With RAE, we argue that a simpler deterministic framework can be competitive for generative modeling.

Deterministic denoising (Vincent et al., 2008) and contractive autoencoders (CAEs) (Rifai et al., 2011) have received attention in the past for their ability to capture a smooth data manifold. Heuristic attempts to equip them with a generative mechanism include MCMC schemes (Rifai et al., 2012; Bengio et al., 2013). However, they are hard to diagnose for convergence, require a considerable effort in tuning (Cowles & Carlin, 1996), and have not scaled beyond MNIST, leading to them being superseded by VAEs. While computing the Jacobian for CAEs (Rifai et al., 2011) is close in spirit to  $\mathcal{L}_{\mathrm{GP}}$  for RAEs, the latter is much more computationally efficient.

Approaches to cope with the aggregated posterior mismatch involve fixing a more expressive form for  $p(\mathbf{z})$  (Kingma et al., 2016; Bauer & Mnih, 2019) therefore altering the VAE objective and requiring considerable additional computational efforts. Estimating the latent space of a VAE with a second VAE (Dai & Wipf, 2019) reintroduces many of the optimization shortcomings discussed for VAEs and is much more expensive in practice compared to fitting a simple  $q_{\delta}(\mathbf{z})$  after training.

![](images/2d5b3ed47bc89d7d21847fb5655d1e5b7dc2c3a01730a7d64bf43d0e3b9bdfac.jpg)  
Figure 1: Qualitative evaluation of sample quality for VAEs, WAEs, 2sVAEs, and RAEs on CelebA. RAE provides slightly sharper samples and reconstructions while interpolating smoothly in the latent space. Corresponding qualitative overviews for MNIST and CIFAR-10 are provided in Appendix F.

Adversarial Autoencoders (AAE) (Makhzani et al., 2016) add a discriminator to a deterministic encoder-decoder pair, leading to sharper samples at the expense of higher computational overhead and the introduction of instabilities caused by the adversarial nature of the training process. Wasserstein Autoencoders (WAE) (Tolstikhin et al., 2017) have been introduced as a generalization of AAEs by casting autoencoding as an optimal transport (OT) problem. Both stochastic and deterministic models can be trained by minimizing a relaxed OT cost function employing either an adversarial loss term or the maximum mean discrepancy score between  $p(\mathbf{z})$  and  $q_{\phi}(\mathbf{z})$  as a regularizer in place of  $\mathcal{L}_{\mathrm{KL}}$ . Within the RAE framework, we look at this problem from a different perspective: instead of explicitly imposing a simple structure on  $\mathbf{Z}$  that might impair the ability to fit high-dimensional data during training, we propose to model the latent space by an ex-post density estimation step.

The most successful VAE architectures for images and audio so far are variations of the VQ-VAE (van den Oord et al., 2017; Razavi et al., 2019). Despite the name, VQ-VAEs are neither stochastic, nor variational, but they are deterministic autoencoders. VQ-VAEs are similar to RAEs in that they adopt ex-post density estimation. However, VQ-VAEs necessitates complex discrete autoregressive density estimators and a training loss that is non-differentiable due to quantizing  $\mathbf{Z}$ .

# 6 EXPERIMENTS

Our experiments are designed to answer the following questions: Q1: Are sample quality and latent space structure in RAEs comparable to VAEs? Q2: How do different regularizations impact RAE performance? Q3: What is the effect of ex-post density estimation on VAEs and its variants?

# 6.1 RAES FOR IMAGE MODELING

We evaluate all regularization schemes from Section 3.1: RAE-GP, RAE-L2, and RAE-SN. For a thorough ablation study, we also consider only adding the latent code regularizer  $\mathcal{L}_{\mathbf{Z}}^{\mathrm{RAE}}$  to  $\mathcal{L}_{\mathrm{REC}}$  (RAE), and an autoencoder without any explicit regularization (AE). As baselines, we employ the regular VAE, constant-variance VAE (CV-VAE), Wasserstein Autoencoder (WAE) with the MMD loss as a state-of-the-art method, and the recent 2-stage VAE (2sVAE) (Dai & Wipf, 2019) which performs a form of ex-post density estimation via another VAE. For a fair comparison, we use the same network architecture for all models. Further details about the architecture and training are given in Appendix C.

<table><tr><td rowspan="3"></td><td colspan="4">MNIST</td><td colspan="4">CIFAR</td><td colspan="4">CELEBA</td></tr><tr><td rowspan="2">REC.</td><td colspan="3">SAMPLES</td><td rowspan="2">REC.</td><td colspan="3">SAMPLES</td><td rowspan="2">REC.</td><td colspan="3">SAMPLES</td></tr><tr><td>N</td><td>GMM</td><td>Interp.</td><td>N</td><td>GMM</td><td>Interp.</td><td>N</td><td>GMM</td><td>Interp.</td></tr><tr><td>VAE</td><td>18.26</td><td>19.21</td><td>17.66</td><td>18.21</td><td>57.94</td><td>106.37</td><td>103.78</td><td>88.62</td><td>39.12</td><td>48.12</td><td>45.52</td><td>44.49</td></tr><tr><td>CV-VAE</td><td>15.15</td><td>33.79</td><td>17.87</td><td>25.12</td><td>37.74</td><td>94.75</td><td>86.64</td><td>69.71</td><td>40.41</td><td>48.87</td><td>49.30</td><td>44.96</td></tr><tr><td>WAE</td><td>10.03</td><td>20.42</td><td>9.39</td><td>14.34</td><td>35.97</td><td>117.44</td><td>93.53</td><td>76.89</td><td>34.81</td><td>53.67</td><td>42.73</td><td>40.93</td></tr><tr><td>2sVAE</td><td>20.31</td><td>18.81</td><td>-</td><td>18.35</td><td>62.54</td><td>109.77</td><td>-</td><td>89.06</td><td>42.04</td><td>49.70</td><td>-</td><td>47.54</td></tr><tr><td>RAE-GP</td><td>14.04</td><td>22.21</td><td>11.54</td><td>15.32</td><td>32.17</td><td>83.05</td><td>76.33</td><td>64.08</td><td>39.71</td><td>116.30</td><td>45.63</td><td>47.00</td></tr><tr><td>RAE-L2</td><td>10.53</td><td>22.22</td><td>8.69</td><td>14.54</td><td>32.24</td><td>80.80</td><td>74.16</td><td>62.54</td><td>43.52</td><td>51.13</td><td>47.97</td><td>45.98</td></tr><tr><td>RAE-SN</td><td>15.65</td><td>19.67</td><td>11.74</td><td>15.15</td><td>27.61</td><td>84.25</td><td>75.30</td><td>63.62</td><td>36.01</td><td>44.74</td><td>40.95</td><td>39.53</td></tr><tr><td>RAE</td><td>11.67</td><td>23.92</td><td>9.81</td><td>14.67</td><td>29.05</td><td>83.87</td><td>76.28</td><td>63.27</td><td>40.18</td><td>48.20</td><td>44.68</td><td>43.67</td></tr><tr><td>AE</td><td>12.95</td><td>58.73</td><td>10.66</td><td>17.12</td><td>30.52</td><td>84.74</td><td>76.47</td><td>61.57</td><td>40.79</td><td>127.85</td><td>45.10</td><td>50.94</td></tr></table>

Table 1: Evaluation of all models by FID (lower is better, best models in bold). We evaluate each model by REC.: test sample reconstruction;  $\mathcal{N}$ : random samples generated according to the prior distribution  $p(\mathbf{z})$  (isotropic Gaussian for VAE / WAE, another VAE for 2SVAE) or by fitting a Gaussian to  $q_{\delta}(\mathbf{z})$  (for the remaining models); GMM: random samples generated by fitting a mixture of 10 Gaussians in the latent space; Interp.: mid-point interpolation between random pairs of test reconstructions. The RAE models are competitive with or outperform previous models throughout the evaluation. Interestingly, interpolations do not suffer from the lack of explicit priors on the latent space in our models.

We measure the following quantities: held-out sample reconstruction quality, random sample quality, and interpolation quality. While reconstructions give us a lower bound on the best quality achievable by the generative model, random sample quality indicates how well the model generalizes. Finally, interpolation quality sheds light on the structure of the learned latent space. The evaluation of generative models is a nontrivial research question (Theis et al., 2016; Sajjadi et al., 2017; Lucic et al., 2018). We report here the ubiquitous Fréchet Inception Distance (FID) (Heusel et al., 2017) and we provide precision and recall scores (PRD) (Sajjadi et al., 2018) in Appendix E.

Table 1 summarizes our main results. All of the proposed RAE variants are competitive with the VAE, WAE and 2sVAE w.r.t. generated image quality in all settings. Sampling RAEs achieve the best FIDs across all datasets when a modest 10-component GMM is employed for ex-post density estimation. Furthermore, even when  $\mathcal{N}$  is considered as  $q_{\delta}(\mathbf{z})$ , RAEs rank first with the exception of MNIST, where it competes for the second position with a VAE. Our best RAE FIDs are lower than the best results reported for VAEs in the large scale comparison of (Lucic et al., 2018), challenging even the best scores reported for GANs. While we are employing a slightly different architecture than theirs, our models underwent only modest finetuning instead of an extensive hyperparameter search. A comparison of the different regularization schemes for RAEs (Q2) yields no clear winner across all settings as all perform equally well. Striving for a simpler implementation, one may prefer RAE-L2 over the GP and SN variants.

Surprisingly, the implicitly regularized RAE and AE models are shown to be able to score impressive FIDs when  $q_{\delta}(\mathbf{z})$  is fit through GMMs. FIDs for AEs decrease from 58.73 to 10.66 on MNIST and from 127.85 to 45.10 on CelebA – a value close to the state of the art. This is a remarkable result that follows a long series of recent confirmations that neural networks are surprisingly smooth by design (Neyshabur et al., 2017). It is also surprising that the lack of an explicitly fixed structure on the latent space of the RAE does not impede interpolation quality. This is further confirmed by the qualitative evaluation on CelebA as reported in Fig. 1 and for the other datasets in Appendix F, where RAE interpolated samples seem sharper than competitors and transitions smoother.

Our results further confirm and quantify the effect of the aggregated posterior mismatch. In Table 1, ex-post density estimation consistently improves sample quality across all settings and models. A 10-component GMM halves FID scores from  $\sim 20$  to  $\sim 10$  for WAE and RAE models on MNIST and from 116 to 46 on CelebA. This is especially striking since this additional step is much cheaper and simpler than training a second-stage VAE as in 2sVAE (Q3). In summary, the results strongly support the conjecture that the simple deterministic RAE framework can challenge VAEs and stronger alternatives (Q1).

<table><tr><td>PROBLEM</td><td>MODEL</td><td>% VALID</td><td>AVG. SCORE</td></tr><tr><td rowspan="3">EXPRESSIONS</td><td>GRAE</td><td>1.00 ± 0.00</td><td>3.22 ± 0.03</td></tr><tr><td>GVAE</td><td>0.99 ± 0.01</td><td>3.26 ± 0.20</td></tr><tr><td>CVAE</td><td>0.82 ± 0.07</td><td>4.74 ± 0.25</td></tr><tr><td rowspan="3">MOLECULES</td><td>GRAE</td><td>0.72 ± 0.09</td><td>-5.62 ± 0.71</td></tr><tr><td>GVAE</td><td>0.28 ± 0.04</td><td>-7.89 ± 1.90</td></tr><tr><td>CVAE</td><td>0.16 ± 0.04</td><td>-25.64 ± 6.35</td></tr></table>

<table><tr><td>MODEL</td><td>#</td><td>EXPRESSION</td><td>SCORE</td></tr><tr><td rowspan="3">GRAE</td><td>1</td><td>sin(3) + x</td><td>0.39</td></tr><tr><td>2</td><td>x + 1/ exp(1)</td><td>0.39</td></tr><tr><td>3</td><td>x + 1 + 2 * sin(3 + 1 + 2)</td><td>0.43</td></tr><tr><td rowspan="3">GVAE</td><td>1</td><td>x/1 + sin(x) + sin(x * x)</td><td>0.10</td></tr><tr><td>2</td><td>1/2 + (x) + sin(x * x)</td><td>0.46</td></tr><tr><td>3</td><td>x/2 + sin(1) + (x/2)</td><td>0.52</td></tr><tr><td rowspan="3">CVAE</td><td>1</td><td>x * 1 + sin(x) + sin(3 + x)</td><td>0.45</td></tr><tr><td>2</td><td>x/1 + sin(1) + sin(2 * 2)</td><td>0.48</td></tr><tr><td>3</td><td>1/1 + (x) + sin(1/2)</td><td>0.61</td></tr></table>

<table><tr><td>MODEL</td><td>1ST</td><td>2ND</td><td>3RD</td></tr><tr><td>GRAE</td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>3.74</td><td>3.52</td><td>3.14</td></tr><tr><td>GVAE</td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>3.13</td><td>3.10</td><td>2.37</td></tr><tr><td>CVAE</td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>2.75</td><td>0.82</td><td>0.63</td></tr></table>

Figure 2: Generating structured objects by GVAE, CVAE and GRAE. (Upper left) Percentage of valid samples and their average mean score (see text, Section 6.2). The three best expressions (lower left) and molecules (upper right) and their scores are reported for all models.

# 6.2 GRAMMARRAE: MODELING STRUCTURED INPUTS

We now evaluate RAEs for generating complex structured objects such as molecules and arithmetic expressions. We do this with a twofold aim: i) to investigate the latent space learned by RAE for more challenging input spaces that abide to some structural constraints, and ii) to quantify the gain of replacing the VAE in a state-of-the-art generative model with a RAE.

To this end, we adopt the exact architectures and experimental settings of the GrammarVAE (GVAE) (Kusner et al., 2017), which has been shown to outperform other generative alternatives such as the CharacterVAE (CVAE) (Gómez-Bombarelli et al., 2018). As in Kusner et al. (2017), we are interested in traversing the latent space learned by our models to generate samples (molecules or expressions) that best fit some downstream metric. This is done by Bayesian optimization (BO) by considering the  $\log(1 + MSE)$  (lower is better) for the generated expressions w.r.t. some ground truth points, and the water-octanol partition coefficient  $(\log P)$  (Pyzer-Knapp et al., 2015) (higher is better) in the case of molecules. A well-behaved latent space will not only generate molecules or expressions with better scores during the BO step, but it will also contain syntactically valid ones, i.e., samples abide to a grammar of rules describing the problem.

Figure 2 summarizes our results over 5 trials of BO. Our GRAEs (Grammar RAE) achieve better average scores than CVAEs and GVAEs in generating expressions and molecules. This is visible also for the three best samples and their scores for all models, with the exception of the first best expression of GVAE. More interestingly, while GRAEs are almost equivalent to GVAEs for the easier task of generating expressions, the proportion of syntactically valid molecules for GRAEs greatly improves over GVAEs (from  $28\%$  to  $72\%$ ).

# 7 CONCLUSION

While the theoretical derivation of the VAE has helped popularize the framework for generative modeling, recent works have started to expose some discrepancies between theory and practice. We have shown that viewing sampling in VAEs as noise injection to enforce smoothness can enable one to distill a deterministic autoencoding framework that is compatible with several regularization techniques to learn a meaningful latent space. We have demonstrated that such an autoencoding framework can generate comparable or better samples than VAEs while getting around the practical drawbacks tied to a stochastic framework. Furthermore, we have shown that our solution of fitting a simple density estimator on the learned latent space consistently improves sample quality both for the proposed RAE framework as well as for VAEs, WAEs, and 2sVAEs which solves the mismatch between the prior and the aggregated posterior in VAEs.

# REFERENCES

Alexander Alemi, Ben Poole, Ian Fischer, Joshua Dillon, Rif A Saurous, and Kevin Murphy. Fixing a Broken ELBO. In ICML, 2018.  
Guozhong An. The effects of adding noise during backpropagation training on a generalization performance. In Neural computation, 1996.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In ICML, 2017.  
Johannes Balle, Valero Laparra, and Eero P Simoncelli. End-to-end optimized image compression. In ICLR, 2017.  
M. Bauer and A. Mnih. Resampled Priors for Variational Autoencoders. In AISTATS, 2019.  
Yoshua Bengio, Li Yao, Guillaume Alain, and Pascal Vincent. Generalized denoising auto-encoders as generative models. In NeurIPS, 2013.  
Christopher M Bishop. Pattern recognition and machine learning. Springer, 2006.  
Samuel R Bowman, Luke Vilnis, Oriol Vinyals, Andrew M Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. In CoNLL, 2016.  
Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In ICLR, 2019.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
Xi Chen, Diederik P Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, and Pieter Abbeel. Variational lossy autoencoder. In *ICLR*, 2017.  
Mary Kathryn Cowles and Bradley P Carlin. Markov chain Monte Carlo convergence diagnostics: a comparative review. In Journal of the American Statistical Association, 1996.  
Bin Dai and David Wipf. Diagnosing and Enhancing VAE Models. In ICLR, 2019.  
Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. Made: Masked autoencoder for distribution estimation. In International Conference on Machine Learning, pp. 881-889, 2015.  
Partha Ghosh, Arpan Losalka, and Michael J Black. Resisting Adversarial Attacks using Gaussian Mixture Variational Autoencoders. In AAAI, 2019.  
Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamin Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán Aspiru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. In ACS central science, 2018.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In NeurIPS, 2017.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium. In NeurIPS, 2017.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. Beta-VAE: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017.  
Matthew D Hoffman and Matthew J Johnson. Elbo surgery: yet another way to carve up the variational evidence lower bound. In Workshop in Advances in Approximate Bayesian Inference, NeurIPS, 2016.

Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In ICML, 2015.  
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. arXiv preprint arXiv:1802.04364, 2018.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In ICLR, 2014.  
Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improving variational inference with inverse autoregressive flow. In NeurIPS, 2016.  
Alex Krizhevsky and Geoffrey Hinton. Learning Multiple Layers of Features from Tiny Images, 2009.  
Karol Kurach, Mario Lucic, Xiaohua Zhai, Marcin Michalski, and Sylvain Gelly. The GAN landscape: Losses, architectures, regularization, and normalization. arXiv preprint arXiv:1807.04720, 2018.  
Matt J Kusner, Brooks Paige, and José Miguel Hernández-Lobato. Grammar Variational Autoencoder. In ICML, 2017.  
Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In AISTATS, 2011.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. In IEEE, 1998.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep Learning Face Attributes in the Wild. In ICCV, 2015.  
Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, and Olivier Bousquet. Are GANs Created Equal? A Large-Scale Study. In NeurIPS, 2018.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. In ICLR, 2016.  
Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which Training Methods for GANs do actually Converge? In ICML, 2018.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In ICLR, 2018.  
Behnam Neyshabur, Ryota Tomioka, Ruslan Salakhutdinov, and Nathan Srebro. Geometry of Optimization and Implicit Regularization in Deep Learning. arXiv preprint arXiv:1705.03071, 2017.  
Edward O Pyzer-Knapp, Changwon Suh, Rafael Gomez-Bombarelli, Jorge Aguilera-Iparraguirre, and Alán Aspuru-Guzik. What is high-throughput virtual screening? a perspective from organic materials discovery. Annual Review of Materials Research, 2015.  
Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. arXiv preprint arXiv:1906.00446, 2019.  
Danilo Jimenez Rezende and Fabio Viola. Taming VAEs. arXiv preprint arXiv:1810.00597, 2018.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In ICML, 2014.  
Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, and Yoshua Bengio. Contractive auto-encoders: Explicit invariance during feature extraction. In ICML, 2011.  
Salah Rifai, Yoshua Bengio, Yann Dauphin, and Pascal Vincent. A generative process for sampling contractive auto-encoders. In ICML, 2012.  
Mihaela Rosca, Balaji Lakshminarayanan, and Shakir Mohamed. Distribution Matching in Variational Inference. arXiv preprint arXiv:1802.06847, 2018.

Mehdi S. M. Sajjadi, Bernhard Scholkopf, and Michael Hirsch. EnhanceNet: Single Image SuperResolution Through Automated Texture Synthesis. In ICCV, 2017.  
Mehdi S. M. Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, and Sylvain Gelly. Assessing Generative Models via Precision and Recall. In NeurIPS, 2018.  
Aliaksei Severyn, Erhardt Barth, and Stanislau Semeniuta. A Hybrid Convolutional Variational Autoencoder for Text Generation. In Empirical Methods in Natural Language Processing, 2017.  
Jocelyn Sietsma and Robert JF Dow. Creating artificial neural networks that generalize. In Neural networks. Elsevier, 1991.  
Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. In NeurIPS, 2015.  
Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Huszár. Amortised MAP Inference for Image Super-resolution. In ICLR, 2017.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. In JMLR, 2014.  
Lucas Theis, Aaron van den Oord, and Matthias Bethge. A note on the evaluation of generative models. In ICLR, 2016.  
Andrey N Tikhonov and Vasilii Iakkovlevich Arsenin. *Solutions of Ill Posed Problems*. Vh Winston, 1977.  
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schölkopf. Wasserstein autoencoders. In ICLR, 2017.  
Jakub Tomczak and Max Welling. VAE with a VampPrior. In AISTATS, 2018.  
George Tucker, Andriy Mnih, Chris J Maddison, John Lawson, and Jascha Sohl-Dickstein. REBAR: low-variance, unbiased gradient estimates for discrete latent variable models. In NeurIPS, 2017.  
Aaron van den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In NeurIPS, 2017.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.  
Sergey Zagoruyko and Nikos Komodakis. Wide Residual Networks. In BMVC, 2016.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In ICLR, 2017.  
Shengjia Zhao, Jiaming Song, and Stefano Ermon. Towards deeper understanding of variational autoencoding models. arXiv preprint arXiv:1702.08658, 2017.
