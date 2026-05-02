# SEMANTICS PRESERVING ADVERSARIAL ATTACKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

While progress has been made in crafting visually imperceptible adversarial examples, constructing semantically meaningful ones remains a challenge. In this paper, we propose a framework to generate semantics preserving adversarial examples. First, we present a manifold learning method to capture the semantics of the inputs. The motivating principle is to learn the low-dimensional geometric summaries of the inputs via statistical inference. Then, we perturb the elements of the learned manifold using the Gram-Schmidt process to induce the perturbed elements to remain in the manifold. To produce adversarial examples, we propose an efficient algorithm whereby we leverage the semantics of the inputs as a source of knowledge upon which we impose adversarial constraints. We apply our approach on toy data, images and text, and show its effectiveness in producing semantics preserving adversarial examples which evade existing defenses against adversarial attacks.

# 1 INTRODUCTION

In response to the susceptibility of deep neural networks to small adversarial perturbations (Szegedy et al., 2014), several defenses have been proposed (Liu et al., 2019; Sinha et al., 2018; Raghunathan et al., 2018; Madry et al., 2017; Kolter & Wong, 2017). Recent attacks have, however, cast serious doubts on the robustness of these defenses (Athalye et al., 2018; Carlini & Wagner, 2016). A standard way to increase robustness is to inject adversarial examples into the training inputs (Goodfellow et al., 2014a). This method, known as adversarial training, is however sensitive to distributional shifts between the inputs and their adversarial examples (Ilyas et al., 2019). Indeed, distortions, occlusions or changes of illumination in an image, to name a few, do not always preserve the nature of the image. In text, slight changes to a sentence often alter its readability or lead to substantial differences in meaning. Constructing semantics preserving adversarial examples would provide reliable adversarial training signals to robustify deep learning models, and make them generalize better. However, several approaches in adversarial attacks fail to enforce the semantic relatedness that ought to exist between the inputs and their adversarial counterparts. This is due to inadequate characterizations of the semantics of the inputs and the adversarial examples — Song et al. (2018) and Zhao et al. (2018b) confine the distribution of the latents of the adversarial examples to a Gaussian. Moreover, the search for adversarial examples is customarily restricted to uniformly-bounded regions or conducted along suboptimal gradient directions (Szegedy et al., 2014; Kurakin et al., 2016; Goodfellow et al., 2014b).

In this study, we introduce a method to address the limitations of previous approaches by constructing adversarial examples that explicitly preserve the semantics of the inputs. We achieve this by characterizing and aligning the low dimensional geometric summaries of the inputs and the adversarial examples. The summaries capture the semantics of the inputs and the adversarial examples. The alignment ensures that the adversarial examples reflect the unbiased semantics of the inputs. We decompose our attack mechanism into: (i.) manifold learning, (ii.) perturbation invariance, and (iii.) adversarial attack. The motivating principle behind step (i.) is to learn the low dimensional geometric summaries of the inputs via statistical inference. Thus, we present a variational inference technique that relaxes the rigid Gaussian prior assumption typically placed on VAEs encoder networks (Kingma & Welling, 2014) to capture faithfully such summaries. In step (ii.), we develop an approach around the manifold invariance concept of (Roussel, 2019) to perturb the elements of the learned manifold while ensuring the perturbed elements remain within the manifold. Finally, in step (iii.), we propose a learning algorithm whereby we leverage the rich semantics of the inputs and the perturbations as a source of knowledge upon which we impose adversarial constraints to produce adversarial examples. Unlike (Song et al., 2018; Carlini & Wagner, 2016; Zhao et al., 2018b; Goodfellow et al., 2014b) that resort to a costly search of adversarial examples, our algorithm is efficient and end-to-end.

The main contributions of our work are thus: (i.) a variational inference method for manifold learning in the presence of continuous latent variables with minimal assumptions about their distribution, (ii.) an intuitive perturbation strategy that encourages perturbed elements of a manifold to remain within the manifold, (iii.) an end-to-end and computationally efficient algorithm that combines (i.) and (ii.) to generate adversarial examples in a black-box setting, and (iv.) illustration on toy data, images and text, as well as empirical validation against strong certified and non-certified adversarial defenses.

# 2 PRELIMINARIES & ARCHITECTURE

Notations. Let  $x$  be a sample from the input space  $\mathcal{X}$ , with label  $y$  from a set of possible labels  $\mathcal{Y}$ , and  $\mathcal{D} = \{x_n\}_{n=1}^N$  a set of  $N$  such samples  $x$ . Also, let  $d$  be a distance measure on  $\mathcal{X}$  capturing closeness in input space, or on  $\mathcal{Z}$ , the embedding space of  $\mathcal{X}$ , capturing semantics similarity.

Adversarial Examples. Given a classifier  $g$ , and a loss function  $\ell$ , an adversarial example of  $x$  is produced by maximizing the objective below over an  $\epsilon$ -radius ball around  $x$  (Athalye et al., 2017).

$$
x^{\prime} = \operatorname *{arg  max}_{x^{\prime}\in \mathcal{X}}\ell (g(x^{\prime}),y)\text{such that} x^{\prime}\in \mathcal{B}(x;\epsilon)
$$

Above, the search region for adversarial examples is confined to a uniformly-bounded ball  $\mathcal{B}(x;\epsilon)$ . In reality, however, the shape imposed on  $\mathcal{B}$  is quite restrictive as the optimal search region may have a different topology. It is also common practice to produce adversarial examples in the input space  $\mathcal{X}$  — via an exhaustive and costly search procedure (Shaham et al., 2018; Song et al., 2018; Zhao et al., 2018b; Athalye et al., 2017; Carlini & Wagner, 2016; Goodfellow et al., 2014b). Unlike these approaches, however, we wish to operate in  $\mathcal{Z}$ , the lower dimensional embedding space of  $\mathcal{X}$ , with minimal computational overhead. Our primary intuition is that  $\mathcal{Z}$  captures well the semantics of  $\mathcal{D}$ . Thus, to construct semantics preserving adversarial examples, we propose the following attack model.

![](images/56593c4a329ae85ddbe6f9dac58a0fcdb3001c7115a1fa168cc13842ade3e31c.jpg)  
Figure 1: Architecture. The set of model parameters  $\Theta = \{\theta_{m}\}_{m = 1}^{M}$  and  $\Theta^{\prime} = \{\theta_{m}^{\prime}\}_{m = 1}^{M}$  are sampled from the recognition networks  $f_{\eta}$  and  $f_{\eta '}$ . Given an input  $x\in$ $\mathcal{D}$ , we use  $E$  to sample the latent codes  $z_{1},\ldots ,z_{M}$  via  $\Theta$ . These codes are passed to  $E^{\prime}$  to learn their perturbed versions  $z_1^\prime ,\dots,z_M^\prime$  using  $\Theta^{\prime}$ . The output  $x^{\prime}\sim p_{\phi}(x^{\prime}|z^{\prime})$  is generated via posterior sampling of a  $z^{\prime}$  (in red).

Attack Model. Given a sample  $x \in \mathcal{D}$  and its class  $y \in \mathcal{V}$ , we want to construct an adversarial example  $x'$  that shares the same semantics as  $x$ . We assume the semantics of  $x$  (resp.  $x'$ ) is modeled by a learned latent variable model  $p(z)$  (resp.  $p'(z'))$  where  $z, z' \in \mathcal{Z}$ . In this setting, observing  $x$  (resp.  $x'$ ) is conditioned on the observation model  $p(x|z)$  (resp.  $p(x'|z'))$  with  $z \sim p(z)$  (resp.  $z' \sim p'(z')$ ). We learn this model in a way that  $d(x, x')$  is small, with  $x \sim p(x|z)$  and  $x' \sim p(x'|z')$ . We ensure also that  $d(z, z')$  is small and  $g(x) = y \wedge g(x') \neq y$ .

Intuitively, we get the latent  $z \sim p(z)$  which encodes the semantics of  $x$ . Then, we perturb  $z$  in a way that its perturbed version  $z' \sim p'(z')$  lies in the manifold that supports  $p(z)$  while ensuring  $d(z, z')$  is small. We define a manifold as a set of points in  $\mathcal{Z}$  where every point is locally Euclidean (Roussel, 2019). We devise our perturbation procedure by generalizing the

manifold invariance concept of (Roussel, 2019) to  $\mathcal{Z}$ . For that, we consider two embedding maps  $h\colon \mathcal{X}\to \mathcal{Z}$  and  $h^\prime \colon \mathcal{X}\to \mathcal{Z}$ , parameterized by  $\theta$  and  $\theta^{\prime}$ , and a map  $dec_{\phi}\colon \mathcal{Z}\rightarrow \mathcal{X}$ . We assume  $\theta$  and  $\theta^{\prime}$  follow the implicit distributions  $p(\theta)$  and  $p(\theta^{\prime})$ .<sup>1</sup> We use  $h^\prime$  to find points in the vicinity of  $h(x)$  that we map onto  $\mathcal{X}$  using  $dec_{\phi}$ . The mappings distant to  $x$  by  $\epsilon$  that fool  $g$  are said to be adversarial.

Model Architecture. To implement our attack model, we propose as a framework the architecture illustrated in Figure 1. Our framework is essentially a variational auto-encoder with two encoders  $E$  and  $E'$  that learn the geometric summaries of  $\mathcal{D}$  via statistical inference. We present two inference mechanisms — implicit manifold learning via Stein variational gradient descent (Liu & Wang, 2016) and Gram-Schmidt basis sign method (Dukes, 2014) — to draw instances of model parameters from the implicit distributions  $p(\theta)$  and  $p(\theta')$  that we parameterize  $E$  and  $E'$  with. Both encoders optimize

the uncertainty inherent to embedding  $\mathcal{D}$  in  $\mathcal{Z}$  while guaranteeing easy sampling via Bayesian ensembling. Finally, the decoder  $p_{\phi}$  acts as a generative model for constructing adversarial examples.

Threat Model. We consider in this paper a black-box scenario where we, as an attacker, have only access to the predictions of a classifier  $g$ . As the attacker, we want to construct adversarial examples not knowing the intricacies of  $g$  such as its loss function, nor having access to its gradient. We focus on this scenario because it is challenging and more plausible in real-life than the white-box case. This threat model serves to evaluate both certified defenses and non-certified ones under our attack model.

# 3 IMPLICIT MANIFOLD LEARNING

Manifold learning is based on the assumption that high dimensional data lies on or near lower dimensional manifolds in a data embedding space. In the variational auto-encoder (VAE) (Kingma & Welling, 2014) setting, the datapoints  $x_{n} \in \mathcal{D}$  are modeled via a decoder  $x_{n} \sim p(x_{n} | z_{n}; \phi)$ . To learn the parameters  $\phi$ , one typically maximizes a variational approximation to the empirical expected log-likelihood  $1 / N \sum_{n=1}^{N} \log p(x_{n}; \phi)$ , called evidence lower bound (ELBO), defined as:

$$
\mathcal {L} _ {\mathrm {e}} (\phi , \psi ; x) = \mathbb {E} _ {z | x; \psi} \log \left[ \frac {p (x | z ; \phi) p (z)}{q (z | x ; \psi)} \right] = - \mathbb {K L} (q (z | x; \psi) \| p (z | x; \phi)) + \log p (x; \phi). \tag {1}
$$

The expectation  $\mathbb{E}_{z|x;\psi}$  can be re-expressed as a sum of a reconstruction loss, or expected negative log-likelihood of  $x$ , and a  $\mathbb{KL}(q(z|x;\psi)\| p(z))$  term. The KL term acts as a regularizer and forces the encoder  $q(z|x;\psi)$  to follow a distribution similar to  $p(z)$ . In VAEs,  $p(z)$  is defined as a spherical Gaussian distribution. The Gaussian form imposed on  $p(z)$  is, however, quite restrictive (Jimenez Rezende & Mohamed, 2015) and may lead to learning poorly the semantics of  $\mathcal{D}$  (Zhao et al., 2017). To sidestep this issue, we minimize the divergence term  $\mathbb{KL}(q(z|x;\psi)\| p(z|x;\phi))$  using Stein Variational Gradient Descent (Liu & Wang, 2016) instead of explicitly optimizing the ELBO.

Stein Variational Gradient Descent (SVGD) is a nonparametric variational inference method that combines the advantages of MCMC sampling and variational inference. Unlike ELBO (Kingma & Welling, 2014), SVGD does not confine a target distribution  $p(z)$  it approximates to simple or tractable parametric distributions. It remains yet an efficient algorithm. To approximate  $p(z)$ , SVGD maintains  $M$  particles  $\mathbf{z} = \{z_m\}_{m=1}^M$ , initially sampled from a simple distribution, it iteratively transports via functional gradient descent. At iteration  $t$ , each particle  $z_t \in \mathbf{z}_t$  is updated as follows:

$$
z _ {t + 1} \leftarrow z _ {t} + \alpha_ {t} \tau (z _ {t}) \text {w h e r e} \tau (z _ {t}) = \frac {1}{M} \sum_ {m = 1} ^ {M} \left[ k (z _ {t} ^ {m}, z _ {t}) \nabla_ {z _ {t} ^ {m}} \log p (z _ {t} ^ {m}) + \nabla_ {z _ {t} ^ {m}} k (z _ {t} ^ {m}, z _ {t}) \right],
$$

where  $\alpha_{t}$  is a step-size and  $k(.,.)$  is a positive-definite kernel. In the equation above, each particle determines its update direction by consulting with other particles and asking their gradients. The importance of the latter particles is weighted according to the distance measure  $k(.,.)$ . Closer particles are given higher consideration than those lying further away. The term  $\nabla_{z^m}k(z^m,z)$  is a regularizer that acts as a repulsive force between the particles to prevent them from collapsing into one particle. Upon convergence, the particles  $z_{m}$  will be unbiased samples of the true implicit distribution  $p(z)$ .

Manifold Learning via SVGD. To faithfully characterize the manifold of  $\mathcal{D}$ , which we denote  $\mathcal{M}$ , we optimize the divergence  $\mathbb{KL}(q(z|x;\psi)\| p(z|x;\phi))$  using SVGD, similar to Pu et al. (2017). Learning  $\mathcal{M}$ , however, induces inherent uncertainty we ought to capture in order to learn  $\mathcal{M}$  efficiently. Pu et al. (2017) use dropout in their manifold learning to capture potentially such uncertainty. However, according to Hron et al. (2017), dropout is not principled. Bayesian methods, on the contrary, provide a principled way to model uncertainty through the posterior distribution over model parameters. In this regard, we introduce  $M$  instances of model parameters  $\Theta = \{\theta_m\}_{m=1}^M$ , where every  $\theta_m \in \Theta$  is a particle that defines the weights and biases of a Bayesian neural network, to which we apply SVGD.

SVGD always maintains  $M$  particles. For large  $M$ , however, maintaining  $\Theta$  can be computationally prohibitive because of the memory footprint. Furthermore, the need to generate the particles during inference for each test case is undesirable. To sidestep these issues, we maintain only one (recognition) network  $f_{\eta}$  that takes as input  $\xi_{m} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  and outputs a particle  $\theta_{m}$ . The recognition network  $f_{\eta}$  learns the trajectories of the particles as they get updated via SVGD.  $f_{\eta}$  serves as a proxy to SVGD

![](images/59d56de287de636c10c211f059a95580fa2ca090ed646e2b1a30cd9c77f8723e.jpg)  
Figure 2: Inversion. Process for computing the likelihood  $p(\mathcal{D}|\theta)$ . As the decoder  $p_{\phi}$  gets accurate, the error  $\| x - \tilde{x}\| _2$  becomes small (see Algorithm 2), and we get closer to sampling the optimal  $\tilde{z}$ .

# Algorithm 1 Inversion with one particle  $\theta$

Require: Input  $x\in \mathcal{D}$

Require: Model parameters  $\pmb{\eta}$

1: Sample  $\xi \sim \mathcal{N}(\mathbf{0},\mathbf{I})$  
2: Sample  $\pmb{\theta} \sim f_{\pmb{\eta}}(\pmb{\xi})$  
3: Given  $x$ , sample  $z \sim p(z|x;\pmb{\theta})$  
4: Sample  $\tilde{x} \sim p(x|z,\phi)$  
5: Sample  $\tilde{z} \sim p(z|\tilde{x},\pmb{\theta})$  
6: Use  $x$  and  $\tilde{z}$  to compute  $p(\tilde{z} | x; \boldsymbol{\theta})$

sampling strategy, and is refined through a small number of gradient steps to get good generalization.

$$
\eta^ {t + 1} \leftarrow \arg \min  _ {\eta} \sum_ {m = 1} ^ {M} \left\| \underbrace {f \left(\xi_ {m} ; \eta^ {t}\right)} _ {\theta_ {m} ^ {t}} - \theta_ {m} ^ {t + 1} \right\| _ {2} \quad \text {w i t h} \theta_ {m} ^ {t + 1} \leftarrow \theta_ {m} ^ {t} + \alpha_ {t} \tau \left(\theta_ {m} ^ {t}\right), \tag {2}
$$

$$
\text {w h e r e} \quad \tau (\theta^ {t}) = \frac {1}{M} \sum_ {j = 1} ^ {M} \left[ k (\theta_ {j} ^ {t}, \theta^ {t}) \nabla_ {\theta_ {t} ^ {j}} \log p (\theta_ {j} ^ {t}) + \nabla_ {\theta_ {t} ^ {j}} k (\theta_ {j} ^ {t}, \theta^ {t}) \right].
$$

We use the notation  $\mathrm{SVGD}_{\tau}(\Theta)$  to denote an SVGD update of  $\Theta$  using the operator  $\tau(.)$ . As the particles  $\theta$  are Bayesian, upon observing  $\mathcal{D}$ , we update the prior  $p(\theta_j^t)$  to obtain the posterior  $p(\theta_j^t|\mathcal{D}) \propto p(\mathcal{D}|\theta_j^t)p(\theta_j^t)$  which captures the uncertainty. We refer the reader to Appendix A for a formulation of  $p(\theta_j^t|\mathcal{D})$  and  $p(\mathcal{D}|\theta_j^t)$ . The data likelihood  $p(\mathcal{D}|\theta_j^t)$  is evaluated over all pairs  $(x,\tilde{z})$  where  $x \in \mathcal{D}$  and  $\tilde{z}$  is a dependent variable. However,  $\tilde{z}$  is not given. Thus, we introduce the inversion process described in Figure 2 to generate such  $\tilde{z}$  using Algorithm 1. For any input  $x \in \mathcal{D}$ , we sample its latent code  $z$  from  $p(z|x;\mathcal{D})$ , which we approximate by Monte Carlo over  $\Theta$ ; that is:

$$
p (z | x; \mathcal {D}) = \int p (z | x; \theta) p (\theta | \mathcal {D}) \mathrm {d} z \approx \frac {1}{M} \sum_ {m = 1} ^ {M} p (z | x; \theta_ {m}) \text {w h e r e} \theta_ {m} \sim p (\theta | \mathcal {D}). \tag {3}
$$

# 4 PERTURBATION INVARIANCE

Here, we focus on perturbing the elements of  $\mathcal{M}$ . We want the perturbed elements to reside in  $\mathcal{M}$  and exhibit the semantics of  $\mathcal{D}$  that  $\mathcal{M}$  captures. Formally, we seek a linear mapping  $h' \colon \mathcal{M} \to \mathcal{M}$  such that for any point  $z \in \mathcal{M}$ , a neighborhood  $\mathcal{U}$  of  $z$  is invariant under  $h'$ ; that is:  $z' \in \mathcal{U} \Rightarrow h'(z') \in \mathcal{U}$ . In this case, we say that  $\mathcal{M}$  is preserved under  $h'$ . Trivial examples of such mappings are linear combinations of the basis vectors of subspaces  $\mathcal{S}$  of  $\mathcal{M}$  called linear spans of  $\mathcal{S}$ .

Rather than finding a linear span  $h'$  directly, we introduce a new set of instances of model parameters  $\Theta' = \{\theta_m'\}_{m=1}^M$ . Each  $\theta_m'$  denotes the weights and biases of a Bayesian neural network. Then, for any input  $x \in \mathcal{D}$  and its latent code  $z \sim p(z|x;\mathcal{D})$ , a point in  $\mathcal{M}$ , we set  $h'(z) = z'$  where  $z' \sim p(z'|x;\mathcal{D})$ . We approximate  $p(z'|x;\mathcal{D})$  by Monte Carlo using  $\Theta'$ , as in Equation 3. We leverage the local smoothness of  $\mathcal{M}$  to learn each  $\theta_m'$  in a way to encourage  $z'$  to reside in  $\mathcal{M}$  in a close neighborhood of  $z$  using a technique called Gram-Schmidt Basis Sign Method.

Gram-Schmidt Basis Sign Method (GBSM). Let  $\mathbf{X}$  be a batch of samples of  $\mathcal{D}$ ,  $\mathbf{Z}_m$  a set of latent codes  $z_m \sim p(z|x;\theta_m)$  where  $x \in \mathbf{X}$ , and  $\theta_m \in \Theta$ . For any  $m \in \{1..,M\}$ , we learn  $\theta_m'$  to generate perturbed versions of  $z_m \in \mathbf{Z}_m$  along the directions of an orthonormal basis  $\mathbf{U}_m$ . As  $\mathcal{M}$  is locally Euclidean, we compute the dimensions of the subspace  $\mathbf{Z}_m$  by applying Gram-Schmidt (Dukes, 2014) to orthogonalize the span of representative local points. We formalize GBSM as follows:

$$
\underset {\delta_ {m}, \theta_ {m} ^ {\prime}} {\arg \min } \varrho (\delta_ {m}, \theta_ {m} ^ {\prime}) := \sum_ {z _ {m}} \left\| z _ {m} ^ {\prime} - \left[ z _ {m} + \delta_ {m} \odot \operatorname {s i g n} (u _ {i m}) \right] \right\| _ {2} \quad \text {w h e r e} z _ {m} ^ {\prime} \sim p (z ^ {\prime} | x _ {i}; \theta_ {m} ^ {\prime}).
$$

The intuition behind GBSM is to utilize the fact that topological spaces are closed under their basis vectors to render  $\mathcal{M}$  invariant to the perturbations  $\delta_{m}$ . To elaborate more on GBSM, we first sample

a model instance  $\theta_{m}^{\prime}$ . Then, we generate  $z_{m}^{\prime} \sim p(z^{\prime}|x; \theta_{m}^{\prime})$  for all  $x \in \mathbf{X}$ . We orthogonalize  $\mathbf{Z}_{m}$  and find the perturbations  $\delta_{m}$  that minimizes  $\varrho$  along the directions of the basis vectors  $u_{im} \in \mathbf{U}_{m}$ . We want the perturbations  $\delta_{m}$  to be small. With  $\delta_{m}$  fixed, we update  $\theta_{m}^{\prime}$  by minimizing  $\varrho$  again. We use the notation  $\mathrm{GBSM}(\Theta^{\prime}, \Delta)$  where  $\Delta = \{\delta_{m}\}_{m=1}^{M}$  to denote one update of  $\Theta^{\prime}$  via GBSM.

Manifold Alignment. Although GBSM confers us latent noise imperceptibility and sampling speed,  $\Theta^{\prime}$  may deviate from  $\Theta$ ; in which case the manifolds they learn will mis-align. To mitigate this issue, we regularize each  $\theta_{m}^{\prime} \in \Theta^{\prime}$  after every GBSM update. In essence, we apply one SVGD update on  $\Theta^{\prime}$  to ensure that  $\Theta^{\prime}$  follows the transform maps constructed by the particles  $\Theta$  (Han & Liu, 2017).

$$
\theta_ {t + 1} ^ {\prime} \leftarrow \theta_ {t} ^ {\prime} + \alpha_ {t} \pi \left(\theta_ {t} ^ {\prime}\right) \text {w h e r e} \pi \left(\theta_ {t} ^ {\prime}\right) = \frac {1}{M} \sum_ {m = 1} ^ {M} \left[ k \left(\theta_ {t} ^ {\prime}, \theta_ {t} ^ {m}\right) \nabla_ {\theta_ {t} ^ {m}} \log p \left(\theta_ {t} ^ {m}\right) + \nabla_ {\theta_ {t} ^ {m}} k \left(\theta_ {t} ^ {\prime}, \theta_ {t} ^ {m}\right) \right]. \tag {4}
$$

We use the notation  $\mathrm{SVGD}_{\pi}(\Theta')$  to refer to the gradient update rule in Equation 4. In this rule, the model instances  $\Theta'$  determine their own update direction by consulting only the particles  $\Theta$  instead of consulting each other. Maintaining  $\Theta' = \{\theta_m'\}_{m=1}^M$  for large  $M$  is, however, computationally prohibitive. Thus, as in Section 3, we keep only one (recognition) network  $f_{\eta'}$  that takes as input  $\xi_m' \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  and outputs  $\theta_m' \sim f(\xi_m'; \eta')$ . Here too we refine  $\eta'$  through a small number of gradient steps to learn the trajectories that  $\Theta'$  follows as it gets updated via GBSM and  $\mathrm{SVGD}_{\pi}$ .

$$
\eta^ {\prime t + 1} \leftarrow \arg \min  _ {\eta^ {\prime}} \sum_ {m = 1} ^ {M} \left\| \underbrace {f \left(\xi_ {m} ^ {\prime} ; \eta^ {\prime t}\right)} _ {\theta_ {m} ^ {\prime t}} - \theta_ {m} ^ {\prime t + 1} \right\| _ {2} \quad \text {w h e r e} \theta_ {m} ^ {\prime t + 1} \leftarrow \theta_ {m} ^ {\prime t} + \alpha_ {t} \pi \left(\theta_ {m} ^ {\prime t}\right). \tag {5}
$$

# 5 GENERATING ADVERSARIAL EXAMPLES

In this paper, a black-box scenario is considered. In this scenario, we have only access to the predictions of the classifier  $g$ . We produce adversarial examples by optimizing the loss below. The first term is the reconstruction loss. This loss accounts for the dissimilarity between any input  $x \in \mathcal{D}$  and its adversarial counterpart  $x'$ , and is constrained to be smaller than  $\epsilon_{\text{attack}}$  so that  $x'$  resides within an  $\epsilon_{\text{attack}}$ -radius ball of  $x$ . The second term is an auxiliary log-likelihood loss (for  $g$ ) of a target class  $y' \in \mathcal{Y} \setminus \{y\}$  where  $y$  is the class of  $x$ . This loss defines the cost incurred for failing to fool  $g$ .

$$
\mathcal {L} _ {x ^ {\prime}} = \| x - x ^ {\prime} \| _ {2} + \min  _ {y ^ {\prime} \in \mathcal {V}} \left[ \mathbb {1} _ {y = y ^ {\prime}} \cdot \log \left(1 - P \left(y ^ {\prime} \mid x ^ {\prime}\right)\right) \right] \text {s u c h t h a t} \| x - x ^ {\prime} \| _ {2} \leq \epsilon_ {\text {a t t a c k}}. \tag {6}
$$

In Algorithm 2 (see also next page), we show how we unify our manifold learning and perturbation invariance techniques into one learning procedure to generate adversarial examples without resorting to an exhaustive search as in (Song et al., 2018; Zhao et al., 2018b; Goodfellow et al., 2014b).

Algorithm 2 Generating Adversarial Examples. Lines 2 and 4 compute distances between sets keeping a one-to-one mapping between them.  $x^{\prime}$  is adversarial to  $x$  when  $\mathcal{L}_{x^{\prime}}\leq \epsilon_{\mathrm{attack}}$  and  $y\neq y^{\prime}$    
1: function INNERTRAINING(Θ, Θ', η, η', Δ, x̂) ▷ local gradient updates of f, f, △  
Require: Learning rates β,β'  
2: η← η - β∇η||Θ - SVGDτ(Θ)||2 ▷ apply inversion on x̂ and update η  
3: Δ, Θ' ← GBSM(Θ', Δ) ▷ update △ and Θ' using GBSM  
4: η' ← η' - β'∇η' ||Θ' - SVGDπ(Θ')||2 ▷ align Θ' with Θ and update η'  
5: return η, η', △  
Require: Training samples (x, y) ∈ D × Y  
Require: Number of model instances M  
Require: Number of inner updates T  
Require: Initialize weights η, η', φ ▷ recognition nets f, f, decoder pφ  
Require: Initialize perturbations △ := {δm}M m=1 ▷ latent (adversarial) perturbations  
Require: Learning rates ε, α, α', and noise margin εattack  
6: Sampleξ1,...,ξM from N(0,I) ▷ inputs to recognition nets f, f,  
7: for t = 1 to T do  
8: Sample Θ = {θm}M m=1 where θm ∼ f(ξm)

9: Sample  $\Theta^{\prime} = \{\theta_{m}^{\prime}\}_{m = 1}^{M}$  where  $\theta_{m}^{\prime}\sim f_{\eta^{\prime}}(\xi_{m})$    
10: Use  $\Theta$  and  $\Theta^{\prime}$  in Equation 3 to sample  $z$  and  $z^{\prime}$    
11: Sample  $\tilde{x}\sim p(x|z,\phi)$  and  $x^{\prime}\sim p(x^{\prime}|z^{\prime},\phi)$  clean and perturbed reconstructions   
12:  $\pmb {\eta},\pmb {\eta}',\Delta \leftarrow$  InnerTraining(  $\Theta ,\Theta ',\eta ,\eta ',\Delta ,\tilde{x})$    
13:  $\mathcal{L}_{\tilde{x}}\coloneqq \| x - \tilde{x}\| _2;\quad \mathcal{L}_{x'}\coloneqq \| x - x'\| _2$  reconstruction losses on  $\tilde{x}$  and  $x^{\prime}$    
14:  $\mathcal{L}_{x'}\coloneqq \left\{ \begin{array}{ll}\mathcal{L}_{x'},\\ \mathcal{L}_{x'} + \min_{y'\in \mathcal{Y}}\left[\mathbb{1}_{y = y'}\cdot \log \left(1 - P(y'|x')\right)\right],\\ \end{array} \right.$  if  $\mathcal{L}_{x^{\prime}} > \epsilon_{attack}$  otherwise   
15:  $\eta \gets \eta -\alpha \nabla_{\eta}\mathcal{L}_{\tilde{x}};\quad \eta^{\prime}\gets \eta^{\prime} - \alpha^{\prime}\nabla_{\eta^{\prime}}\mathcal{L}_{x^{\prime}}$  SGD update using Adam optimizer   
16:  $\phi \gets \phi -\epsilon \nabla_{\phi}(\mathcal{L}_{\tilde{x}} + \mathcal{L}_{x'})$  SGD update using Adam optimizer

# 6 RELATED WORK

Manifold Learning. VAEs are generally used to learn manifolds (Yu et al., 2018; Falorsi et al., 2018; Higgins et al., 2016) by maximizing the ELBO of the data log-likelihood (Alemi et al., 2017; Chen et al., 2017). Optimizing the ELBO entails reparameterizing the encoder to a Gaussian distribution (Kingma & Welling, 2014). This reparameterization is, however, restrictive (Jimenez Rezende & Mohamed, 2015) as it may lead to learning poorly the manifold of the data (Zhao et al., 2017). To alleviate this issue, we use SVGD, similar to Pu et al. (2017). While our approach and that of Pu et al. (2017) may look similar, ours is more principled. As discussed in (Hron et al., 2017), dropout which Pu et al. (2017) use is not Bayesian. Since our model instances are Bayesian, we are better equipped to capture the uncertainty. Capturing the uncertainty requires, however, evaluating the data likelihood. As we are operating in latent space, this raises the interesting challenge of assigning target dependent variables to the inputs. We overcome this challenge using our inversion process.

Adversarial Examples. Studies in adversarial deep learning (Athalye et al., 2018; Kurakin et al., 2016; Goodfellow et al., 2014b; Athalye et al., 2017) can be categorized into two groups. The first group (Carlini & Wagner, 2016; Athalye et al., 2017; Moosavi-Dezfooli et al., 2016) proposes to generate adversarial examples directly in the input space of the original dat by distorting, occluding or changing illumination in images to cause changes in classification. The second group (Song et al., 2018; Zhao et al., 2018b), where our work belongs, uses generative models to search for adversarial examples in the dense and continuous representations of the data rather than in its input space.

Adversarial Images. Song et al. (2018) propose to construct unrestricted adversarial examples in the image domain by training a conditional GAN that constrains the search region for a latent code  $z'$  in the neighborhood of a target  $z$ . Zhao et al. (2018b) use also a GAN to map input images to a latent space where they conduct their search for adversarial examples. These studies are the closest to ours. Unlike in (Song et al., 2018) and (Zhao et al., 2018b), however, our adversarial perturbations are learned and we do not constrain the search for adversarial examples to uniformly-bounded regions. In stark contrast to Song et al. (2018) and Zhao et al. (2018b) approaches also, where the search for adversarial examples is exhaustive and decoupled from the training of the GANs, our approach is efficient and end-to-end. Lastly, by capturing the uncertainty induced by embedding the data, we characterize the semantics of the data better, allowing us thus to generate sound adversarial examples.

Adversarial Text. Previous studies on adversarial text generation (Zhao et al., 2018a; Jia & Liang, 2017; Alvarez-Melis & Jaakkola, 2017; Li et al., 2016) perform word erasures and replacements directly in the input space using domain-specific rules or heuristics, or they require manual curation. Similar to us, Zhao et al. (2018b) propose to search for textual adversarial examples in the latent representation of the data. However, in addition to the differences aforementioned for images, the search for adversarial examples is handled more gracefully in our case thanks to an efficient gradient-based optimization method in lieu of a computationally expensive search in the latent space.

# 7 EXPERIMENTS & RESULTS

Before, we presented an attack model whereby we align the semantics of the inputs with their adversarial counterparts. As a reminder, our attack model is black-box and non-targeted. Our adversarial examples reside within an  $\epsilon_{\mathrm{attack}}$  -radius ball of the inputs as our reconstruction loss, which measures the amount of changes in the inputs, is bounded by  $\epsilon_{\mathrm{attack}}$  (see Equation 6). We

validate the adversarial examples we produce based on three evaluation criteria: (i.) manifold preservation, (ii.) adversarial strength, and (iii.) soundness via manual evaluation. We provide in Appendix A examples of the adversarial images and sentences that we construct.

# 7.1 MANIFOLD PRESERVATION

We experiment with a 3D non-linear Swiss Roll dataset which comprises 1600 datapoints grouped in 4 classes. We show in Figure 3, on the left, the 2D plots of the manifold we learn. In the middle, we plot the manifold and its elements that we perturbed and whose reconstructions are adversarial. On the right, we show the manifold overlaid with the latent codes of the adversarial examples produced by PGD (Madry et al., 2017) with  $\epsilon_{\mathrm{attack}} \leq 0.3$ . Observe in Figure 3, in the middle plot, how the latent codes of our adversarial examples espouse the Swiss Roll manifold, unlike the plot on the right.

![](images/d6a8c3d5f87b45641ad1ebf5a2398ef98cf6a49c0ae150690cad94e47a5e32a1.jpg)  
Figure 3: Invariance. Swiss Roll manifold learned with our encoder  $E$  (left), and after perturbing its elements with our encoder  $E'$  (middle) vs. that of PGD adversarial examples (right) learned using  $E$ .

# 7.2 ADVERSARIAL STRENGTH

In this section, we evaluate the strength of the adversarial images and sentences we construct.

Setup. As argued in (Athalye et al., 2018), the strongest non-certified defense against adversarial attacks is adversarial training with Projected Gradient Descent (PGD) (Madry et al., 2017). Thus, we evaluate the strength of our MNIST, CelebA and SVHN adversarial examples against adversially trained ResNets (He et al., 2015) with a 40-step PGD and noise margin  $\epsilon_{\mathrm{attack}} \leq 0.3$ . The ResNet models follow the architecture design of (Song et al., 2018). For MNIST, we also target the certified defenses of (Raghunathan et al., 2018; Kolter & Wong, 2017) with  $\epsilon_{\mathrm{attack}}$  set to 0.1, similar to Song et al. (2018) whose attack model resembles ours. These defenses defend against  $L_p$ -norm attacks like ours. For all the datasets, the accuracies of the models we target are higher than  $96.3\%$ . Next, we present our adversarial success rates and give examples of our adversarial images in Figure 4.

Adversarial Success Rate (ASR) is the percentage of examples that are misclassified by the adversarially trained Resnet models. For  $\epsilon_{\mathrm{attack}} = 0.3$ , the publicly known ASR of PGD attacks on MNIST is  $88.79\%$ . However, our ASR for MNIST is  $97.2\%$ , higher than PGD. Also, with  $\epsilon_{\mathrm{attack}} = 0.3$ , we achieve an ASR of  $96.8\%$  against (Kolter & Wong, 2017). Against the remaining adversarially trained Resnet models, we achieve an ASR of  $87.6\%$  for SVHN, and  $84.4\%$  for CelebA.

# 7.2.1 ADVERSARIAL TEXT

Datasets. For text, we consider the SNLI (Bowman et al., 2015) dataset. SNLI consists of sentence pairs where each pair contains a premise and a hypothesis, and a label indicating the relationship (entailment, neutral, contradiction) between the premise and hypothesis. For instance, the following pair is assigned the label entailment to indicate that the premise entails the hypothesis.

Premise: A soccer game with multiple males playing. Hypothesis: Some men are playing a sport.

Table 1: Test samples and adversarial hypotheses:  $(P)$  for premise,  $(H)$  for Hypothesis.  

<table><tr><td>True Input 1</td><td>P: A biker races. H: A person is riding a bike. Label: Entailment</td></tr><tr><td>Adversary 1</td><td>H: A man races. Label: Contradiction</td></tr><tr><td>True Input 2</td><td>P: The girls walk down the street. H: Girls walk down the street. Label: Entailment</td></tr><tr><td>Adversary 2</td><td>H: A choir walks down the street. Label: Neutral</td></tr><tr><td>True Input 3</td><td>P: Two dogs playing fetch. H: Two puppies play with a red ball. Label: Neutral</td></tr><tr><td>Adversary 3</td><td>H: Two people play in the snow. Label: Contradiction</td></tr></table>

Setup. We perturb the hypotheses sentences to attack our SNLI classifier while keeping the premise sentences unchanged. Similar to Zhao et al. (2018b), we use ARAE (Zhao et al., 2018a) for word embedding, and a CNN for sentence embedding. To generate adversarial sentences from the perturbed latent codes, we experiment with three decoders: (i.)  $p_{\phi}$  is a transpose CNN, (ii.)  $p_{\phi}$  is a language model, and (iii.) we use the decoder of a pre-trained ARAE (Zhao et al., 2018a) model. In all three cases, we condition the generation of the adversarial hypotheses on the sentence pairs premises. We detail the configuration design of each decoder in Appendix B. We generate adversarial text at word level using a vocabulary of 11,000 words only, similar to (Zhao et al., 2018b).

Adversarial Success Rate (ASR). With the transpose CNN, we achieve an ASR of  $77.77\%$  against the SNLI classifier that has an accuracy of  $89.42\%$ . We generate more legible hypotheses with the transpose CNN than with the language model and the pre-trained ARAE model. Table 1 shows samples of the generated adversarial hypotheses. Also, the hypotheses are more informative, and convey better the meaning of the perturbed sentences. Sometimes, however, we notice some changes in the meaning of the original hypotheses. We discuss these limitations in Appendix A and provide more examples of our adversarial hypotheses.

# 7.3 MANUAL EVALUATION

To validate our adversarial examples and assess their soundness vs. Song et al. (2018), Zhao et al. (2018b) and PGD (Madry et al., 2017) adversarial examples, we carry out a pilot study whereby we

ask three yes-or-no questions: (Q1) are the adversarial examples semantically sound?, (Q2) are the true inputs similar perceptually or in meaning to their adversarial counterparts? and (Q3) are there any interpretable visual cues in the adversarial images that support their misclassification?

Pilot Study. For MNIST, we pick 50 images (5 for each digit), generate their clean reconstructions, and their adversarial examples against a 40-step PGD ResNet with  $\epsilon_{\mathrm{attack}} \leq 0.3$ . We target also the certified defenses of Raghunathan et al. (2018) and Kolter & Wong (2017) with  $\epsilon_{\mathrm{attack}} = 0.1$ . We hand the images and the questionnaire to 10 human subjects. We report the results in Table 2.

We carry out a similar pilot study for SVHN, CelebA, and SNLI. For SVHN, we attack a 40-step PGD ResNet. For CelebA, we pick 50 images (25 for each gender) and generate adversarial examples against a 40-step PGD ResNet. Finally, for SNLI, we select 20 pairs of sentences (premise and

![](images/d4f290ae8005ca7af237c6ae3ff179f8ecc1fbb4b41f3c77ec5b6aa12aa87902.jpg)  
(a)

![](images/07d9c65b6320c672c28ce5386a35d3453f188c73c6b4de316880bb0a9183aac8.jpg)  
(b)

![](images/bb2bd47adc9fe7e7e310132a0765f9c4009ca6156053316496bd5f929432e373.jpg)  
(c)

![](images/4f63b72905b1d141876c9f2fba02c80752f2dbd689da5ad53386b9096b981cc1.jpg)  
(d)

![](images/7976e12db0afe0e8d29af8541aa24721f5995ce55a0ed539aad07da635aa5619.jpg)  
(e)  
Figure 4: Inputs (left) - Adversarial examples (right, inside red boxes). MNIST: (a)-(b), CelebA: (c)-(d), SVHN: (e)-(f). See Appendix A for more samples with higher resolution.

![](images/c684fad9d2d6090278978b2d7a81f1088d1be02b2519e284eef8e23d7c719b35.jpg)  
(f)

hypothesis). Using the transpose CNN as decoder  $p_{\phi}$ , we generate adversarial hypotheses for each pair with the premise sentence kept unchanged. We also pick 20 pairs of sentences and adversarial hypotheses generated using Zhao et al. (2018b)'s treeLSTM. We choose their treeLSTM as its accuracy (89.04%) is close to that of our SNLI classifier (89.42%). We report the results in Table 3.

Table 2: Pilot Study (MNIST). Note that against the certified defenses of Raghunathan et al. (2018) and Kolter & Wong (2017), Song et al. (2018) achieved (manual) success rates of  $86.6\%$  and  $88.6\%$ .  

<table><tr><td rowspan="2">QUESTIONNAIRE</td><td colspan="3">MNIST</td></tr><tr><td>40-STEP PGD</td><td>RAGHUNATHAN ET AL. (2018)</td><td>KOLTER &amp; WONG (2017)</td></tr><tr><td>QUESTION Q1: YES</td><td>100 %</td><td>100 %</td><td>100 %</td></tr><tr><td>QUESTION Q2: YES</td><td>100 %</td><td>100 %</td><td>100 %</td></tr><tr><td>QUESTION Q3: No</td><td>100 %</td><td>100 %</td><td>100 %</td></tr></table>

Table 3: Pilot Study. † Some adversarial images and original ones were found blurry to evaluate.  

<table><tr><td rowspan="2">QUESTIONNAIRE</td><td rowspan="2">CELEBA</td><td rowspan="2">SVHN</td><td colspan="2">SNLI</td></tr><tr><td>OUR METHOD</td><td>ZHAO ET AL. (2018B)</td></tr><tr><td>QUESTION Q1: YES</td><td>100 %</td><td>95† %</td><td>82 %</td><td>76%</td></tr><tr><td>QUESTION Q2: YES</td><td>100 %</td><td>97 %</td><td>61 %</td><td>57%</td></tr><tr><td>QUESTION Q3: No</td><td>100 %</td><td>100 %</td><td>--</td><td>--</td></tr></table>

We hand the same questionnaire to the subjects with 50 MNIST images, their clean reconstructions, and the adversarial examples we craft with our method. We also handed the adversarial examples generated using Song et al. (2018), Zhao et al. (2018b) and PGD methods. We ask the subjects to assess the soundness of the adversarial examples based on the semantic features (e.g., shape, distortion, contours, class) of the real MNIST images. We report the evaluation results in Table 4.

Table 4: Pilot Study. The adversarial examples are generated against the adversarially trained Resnets.  

<table><tr><td>QUESTIONNAIRE</td><td>OUR METHOD</td><td>SONG ET AL. (2018)</td><td>ZHAO ET AL. (2018B)</td><td>PGD</td></tr><tr><td>QUESTION Q1: YES</td><td>100 %</td><td>85.9 %</td><td>97.8 %</td><td>76.7 %</td></tr><tr><td>QUESTION Q2: YES</td><td>100 %</td><td>79.3 %</td><td>89.7 %</td><td>66.8 %</td></tr><tr><td>QUESTION Q3: No</td><td>100 %</td><td>71.8 %</td><td>94.6 %</td><td>42.7 %</td></tr></table>

Takeaways. As reflected in the pilot study, and in the adversarial success rates, we achieve good results in the image and text classification tasks. In the image classification tasks, our results are better than PGD and Song et al. (2018)'s results both against the certified and non-certified defenses. The other key learning with our results is the following. Although the targeted defenses are resilient to adversarial examples crafted in the input space, they remain to be as effective against adversarial examples constructed in the latent space — Song et al. (2018) also reached the same conclusion, or when the search region of adversarial examples is unrestricted. In text classification, we achieve comparable with Zhao et al. (2018b)'s treeLSTM and LSTM results (see their paper for the LSTM).

# 8 CONCLUSION

Many approaches in adversarial attacks fail to enforce the semantic relatedness that ought to exist between original inputs and their adversarial counterparts. Motivated by this fact, we developed a method tailored to ensuring that the original inputs and their adversarial examples exhibit similar semantics by conducting the search for adversarial examples in the manifold of the inputs. Our success rates against certified and non-certified defenses known to be resilient to traditional adversarial attacks illustrate the effectiveness of our method in generating sound and strong adversarial examples.

Although in the text classification task we achieved good results and generated informative adversarial sentences, each of the three sentence generators we introduced has some limitations. First, they

are small in size. Second, the language model and the pre-trained ARAE model performed poorly, compared to the transpose CNN that generates legible sentences. Our intuition is that the compounding effect of the perturbations affected the performance of the language model, and that ARAE suffered a distributional shift. Also, as the transpose CNN gets more accurate — recall that it is partly trained to minimize a reconstruction error, generating adversarial sentences that are different from the input sentences and yet preserve their semantic meaning becomes more challenging. In the future, we intend to build upon the recent advances in text understanding to improve our text generation process.

# REFERENCES

Alexander A. Alemi, Ben Poole, Ian Fischer, Joshua V. Dillon, Rif A. Saurous, and Kevin Murphy. An information-theoretic analysis of deep latent-variable models. CoRR, abs/1711.00464, 2017. URL http://arxiv.org/abs/1711.00464.  
David Alvarez-Melis and Tommi S. Jaakkola. A causal framework for explaining the predictions of black-box sequence-to-sequence models. CoRR, abs/1707.01943, 2017. URL http://arxiv.org/abs/1707.01943.  
Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. CoRR, abs/1707.07397, 2017. URL http://arxiv.org/abs/1707.07397.  
Anish Athalye, Nicholas Carlini, and David A. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. CoRR, abs/1802.00420, 2018. URL http://arxiv.org/abs/1802.00420.  
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. CoRR, abs/1508.05326, 2015. URL http://arxiv.org/abs/1508.05326.  
Nicholas Carlini and David A. Wagner. Towards evaluating the robustness of neural networks. CoRR, abs/1608.04644, 2016. URL http://arxiv.org/abs/1608.04644.  
Liqun Chen, Shuyang Dai, Yunchen Pu, Chunyuan Li, Qinliang Su, and Lawrence Carin. Symmetric Variational Autoencoder and Connections to Adversarial Learning. arXiv e-prints, art. arXiv:1709.01846, Sep 2017.  
Kimberly A. Dukes. GramSchmidt Process. American Cancer Society, 2014. ISBN 9781118445112. doi: 10.1002/9781118445112.stat05633. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118445112.stat05633.  
Luca Falorsi, Pim de Haan, Tim R. Davidson, Nicola De Cao, Maurice Weiler, Patrick Forre, and Taco S. Cohen. Explorations in Homeomorphic Variational Auto-Encoding. arXiv e-prints, art. arXiv:1807.04689, Jul 2018.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014a.  
Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harvesting adversarial examples. CoRR abs/1412.6572, 2014b.  
Jun Han and Qiang Liu. Stein variational adaptive importance sampling. Conference on Uncertainty in Artificial Intelligence (UAI), 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015. URL http://arxiv.org/abs/1512.03385.  
Irina Higgins, Loic Matthew, Xavier Glorot, Arka Pal, Benigno Uria, Charles Blundell, Shakir Mohamed, and Alexander Lerchner. Early Visual Concept Learning with Unsupervised Deep Learning. arXiv e-prints, art. arXiv:1606.05579, Jun 2016.  
Jiri Hron, Alexander G. de G. Matthews, and Zoubin Ghahramani. Variational Gaussian Dropout is not Bayesian. arXiv e-prints, art. arXiv:1711.02989, Nov 2017.

Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander Madry. Adversarial Examples Are Not Bugs, They Are Features. arXiv e-prints, art. arXiv:1905.02175, May 2019.  
Robin Jia and Percy Liang. Adversarial examples for evaluating reading comprehension systems. CoRR, abs/1707.07328, 2017. URL http://arxiv.org/abs/1707.07328.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational Inference with Normalizing Flows. arXiv e-prints, art. arXiv:1505.05770, May 2015.  
Taesup Kim, Jaesik Yoon, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian model-agnostic meta-learning. CoRR, abs/1806.03836, 2018. URL http://arxiv.org/abs/1806.03836.  
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. International Conference on Learning Representations (ICLR), 2014.  
J. Zico Kolter and Eric Wong. Provable defenses against adversarial examples via the convex outer adversarial polytope. CoRR, abs/1711.00851, 2017. URL http://arxiv.org/abs/1711.00851.  
Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial examples in the physical world. CoRR, abs/1607.02533, 2016. URL http://arxiv.org/abs/1607.02533.  
Jiwei Li, Will Monroe, and Dan Jurafsky. Understanding neural networks through representation erasure. CoRR, abs/1612.08220, 2016. URL http://arxiv.org/abs/1612.08220.  
Chen Liu, Ryota Tomioka, and Volkan Cevher. On certifying non-uniform bound against adversarial attacks. CoRR, abs/1903.06603, 2019. URL http://arxiv.org/abs/1903.06603.  
Qiang Liu and Dilin Wang. Stein variational gradient descent: A general purpose Bayesian inference algorithm. Neural Information Processing Systems (NIPS), 2016.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. CoRR, abs/1706.06083, 2017. URL http://arxiv.org/abs/1706.06083.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. CoRR, abs/1610.08401, 2016. URL http://arxiv.org/abs/1610.08401.  
Yunchen Pu, Zhe Gan, Ricardo Henao, Chunyuan Li, Shaobo Han, and Lawrence Carin. VAE learning via Stein variational gradient descent. Neural Information Processing Systems (NIPS), 2017.  
Alec Radford. Improving language understanding by generative pre-training. In arXiv, 2018.  
Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. CoRR, abs/1801.09344, 2018. URL http://arxiv.org/abs/1801.09344.  
Marc R Roussel. Invariant manifolds. In Nonlinear Dynamics, 2053-2571, pp. 6-1 to 6-20. Morgan & Claypool Publishers, 2019. ISBN 978-1-64327-464-5. doi: 10.1088/2053-2571/ab0281ch6. URL http://dx.doi.org/10.1088/2053-2571/ab0281ch6.  
Uri Shaham, Yutaro Yamada, and Sahand Negahban. Understanding adversarial training: Increasing local stability of supervised models through robust optimization. Neurocomputing, 307:195-204, 2018.  
Aman Sinha, Hongseok Namkoong, and John Duchi. Certifiable distributional robustness with principled adversarial training. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=Hk6kPgZA-.

Yang Song, Rui Shu, Nate Kushman, and Stefano Ermon. Constructing unrestricted adversarial examples with generative models. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, 3-8 December 2018, Montréal, Canada., pp. 8322-8333, 2018. URL http://papers.nips.cc/paper/8052-constructing-unrestricted-adversarial-examples-with-generative-models.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014. URL http://arxiv.org/abs/1312.6199.  
Bing Yu, Jingfeng Wu, and Zhanxing Zhu. Tangent-normal adversarial regularization for semi-supervised learning. CoRR, abs/1808.06088, 2018. URL http://arxiv.org/abs/1808.06088.  
Junbo Zhao, Yoon Kim, Kelly Zhang, Alexander Rush, and Yann LeCun. Adversarily regularized autoencoders. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 5902-5911, Stockholm, Sweden, 10-15 Jul 2018a. PMLR. URL http://proceedings.mlr.press/v80/zhao18b.html.  
Shengjia Zhao, Jiaming Song, and Stefano Ermon. InfoVAE: Information maximizing variational autoencoders. CoRR, abs/1706.02262, 2017. URL http://arxiv.org/abs/1706.02262.  
Zhengli Zhao, Dheeru Dua, and Sameer Singh. Generating natural adversarial examples. In International Conference on Learning Representations, 2018b. URL https://openreview.net/forum?id=H1BLjgZCb.
