# GM-VAE: REPRESENTATION LEARNING WITH VAE ON GAUSSIAN MANIFOLD

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a Gaussian manifold variational auto-encoder (GM-VAE) whose latent space consists of a set of diagonal Gaussian distributions. It is known that the set of the diagonal Gaussian distributions with the Fisher information metric forms a product hyperbolic space, which we call a Gaussian manifold. To learn the VAE endowed with the Gaussian manifold, we first propose a pseudo Gaussian manifold normal distribution based on the Kullback-Leibler divergence, a local approximation of the squared Fisher-Rao distance, to define a density over the latent space. With the newly proposed distribution, we introduce geometric transformations at the last and the first of the encoder and the decoder of VAE, respectively to help the transition between the Euclidean and Gaussian manifolds. Through the empirical experiments, we show competitive generalization performance of GM-VAE against other variants of hyperbolic- and Euclidean-VAEs. Our model achieves strong numerical stability, which is a common limitation reported with previous hyperbolic-VAEs.

# 1 INTRODUCTION

The geometry of latent space in generative models, such as the variational auto-encoders (VAE) (Kingma & Welling, 2013) and generative adversarial networks (GAN) (Goodfellow et al., 2020), reflects the structure of the representation of the data. Mathieu et al. (2019); Nagano et al. (2019); Cho et al. (2022) show that employing a hyperbolic space as the latent space improves in preserving the hierarchical structure of the data in the latent space. The expanded geometry is not just limited to the hyperbolic space, as the space can be other types of Riemannian manifolds, such as spherical manifolds (Xu & Durrett, 2018; Davidson et al., 2018) and the product of Riemannian manifolds with mixed curvatures (Skopek et al., 2019).

Meanwhile, it is known that univariate Gaussian distributions equipped with Fisher information metric (FIM) form a Riemannian manifold, sharing the manifold with Poincaré half-plane which is one of the four isometric hyperbolic models. This statistical manifold is known to have a metric tensor akin to that of the Poincaré half-plane (Costa et al., 2015), providing a possibility of viewing it as a hyperbolic space. Furthermore, the diagonal Gaussian distributions form a product of Riemannian manifolds showing the presence of an extended statistical manifold.

Based on the connection between hyperbolic spaces and statistical manifolds, in this work, we add an alternative perspective on hyperbolic VAEs with a viewpoint from the information geometry. Previously proposed hyperbolic VAEs rely on the distributions defined over the hyperbolic space. Riemannian normal and wrapped normal are commonly used as prior and variational distributions over the hyperbolic space. Unlike the Gaussian distribution in Euclidean space, these distributions suffer from numerical instability (Mathieu et al., 2019; Skopek et al., 2019). In addition, the Riemannian normal requires performing rejection sampling, which often generates too many unwanted samples.

From the information geometric perspective of the hyperbolic space, we introduce a new distribution, named a pseudo Gaussian manifold normal distribution (PGM normal). The Gaussian manifold, here, refers to the statistical manifold with univariate Gaussian distributions. The newly proposed distribution uses the KL divergence as a statistical distance between two distributions in the Gaussian manifold. Since the KL divergence approximates the squared Riemannian distance of the statistical manifold, derived from FIM, the proposed distribution follows the geometric property of the Gaussian

distributions. We show that the PGM normal is easy to sample, and the KL divergence between two PGM normals can be computed analytically.

With the PGM normal as prior and variational distributions, we define a Gaussian manifold VAE (GM-VAE), whose latent space is defined over the Gaussian manifold. Nevertheless, the data points are still assumed to be defined over the Euclidean space. To correct the mismatch between the data space and the latent space, we introduce a transformation from Euclidean to hyperbolic space at the last and the first layers of the encoder and decoder, respectively.

Empirical experiments with multiple datasets show that GM-VAE can achieve a competitive generalization performance against existing hyperbolic VAEs. During the experiments, we observe that the PGM normal is robust in terms of sampling and computation of the KL divergence, compared to the commonly-used hyperbolic distributions; we briefly explain the reason why others are numerically unstable. Analysis of the latent space exhibits that the geometrical structures and probabilistic semantics of the dataset can be captured in the representations learned with GM-VAE.

We summarize our contributions as follows:

- We propose a variant of VAE whose latent space is defined on a statistical manifold formed by diagonal Gaussian distributions.  
- We propose a new distribution called pseudo Gaussian manifold normal distribution, which is easy to sample and has closed form KL-divergence, to train the VAE on the manifold.  
- We propose new encoder and decoder structures to support the proper transition between Euclidean (data) space and the statistical manifold.  
- We empirically verify that the newly proposed model performs similarly to existing hyperbolic VAEs while achieving stable training without numerical issues.

# 2 PRELIMINARIES

In this section, we first review the fundamental concepts of the Riemannian manifold. We then explain the commonly-used distributions over the Riemannian manifolds and visit the concepts of Riemannian geometry between statistical objects.

# 2.1 REVIEW OF RIEMANNIAN MANIFOLD

A  $n$ -dimensional Riemannian manifold consists of a manifold  $\mathcal{M}$  and a metric tensor  $g: \mathcal{M} \to \mathbb{R}^{n \times n}$ , which is a smooth map from each point  $\mathbf{x} \in \mathcal{M}$  to a symmetric positive definite matrix. The metric tensor  $g(\mathbf{x})$  defines the inner product of two tangent vectors for each point of the manifold  $\langle \cdot, \cdot \rangle_{\mathbf{x}}: \mathcal{T}_{\mathbf{x}}\mathcal{M} \times \mathcal{T}_{\mathbf{x}}\mathcal{M} \to \mathbb{R}$ , where  $\mathcal{T}_{\mathbf{x}}\mathcal{M}$  is the tangent space of  $\mathbf{x}$ .

A Riemannian manifold can be characterized by the curvature of the curves defined on it. The curvature of a Riemannian manifold can be computed at each point of the curves, while some manifolds have curvature of a constant value. For example, the unit sphere  $S$  has constant positive curvature of  $+1$ , and the Poincaré half-plane  $\mathcal{U}$  has constant negative curvature of  $-1$ . The hyperbolic models Among the hyperbolic space, the Klein model, the Poincaré disk model, the Lorentz (Hyperboloid) model, and Poincaré half-plane model are known to be isometric and have the same value of curvature  $-1$  (Nickel & Kiela, 2018; Gulcehre et al., 2018; Tifrea et al., 2018).

The metric tensor induces basic operations of the Riemannian manifold such as a geodesic, exponential map, log map, and parallel transport. Given two points  $\mathbf{x},\mathbf{y}\in \mathcal{M}$ , geodesic  $\gamma_{\mathbf{x}}:[0,1]\to \mathcal{M}$  is a unit speed curve on  $\mathcal{M}$  being the shortest path between  $\gamma (0) = \mathbf{x}$  and  $\gamma (1) = \mathbf{y}$ . This can be interpreted as the generalized curve of a straight line in the Euclidean space. The exponential map  $\exp_{\mathbf{x}}:\mathcal{T}_{\mathbf{x}}\mathcal{M}\rightarrow \mathcal{M}$  is defined as  $\gamma (1)$ , where  $\gamma$  is a geodesic starting from  $\mathbf{x}$  and  $\gamma^{\prime}(0) = \mathbf{v}$ , where a tangent vector  $\mathbf{v}\in \mathcal{T}_{\mathbf{x}}\mathcal{M}$ . The log map  $\log_{\mathbf{x}}:\mathcal{M}\to \mathcal{T}_{\mathbf{x}}\mathcal{M}$  is the inverse of the exponential map, i.e.,  $\log_{\mathbf{x}}(\exp_{\mathbf{x}}(\mathbf{v})) = \mathbf{v}$ . The parallel transport  $\mathrm{PT}_{\mathbf{x}\to \mathbf{y}}:\mathcal{T}_{\mathbf{x}}\mathcal{M}\to \mathcal{T}_{\mathbf{y}}\mathcal{M}$  moves the tangent vector  $\mathbf{v}$  along the geodesic between  $\mathbf{x}$  and  $\mathbf{y}$ . The distance function  $d_{\mathcal{M}}(\mathbf{x},\mathbf{y})$  can be induced from the metric tensor as follows:

$$
d _ {\mathcal {M}} (\mathbf {x}, \mathbf {y}) = \int_ {0} ^ {1} \sqrt {\left\langle \dot {\gamma} (t) , \dot {\gamma} (t) \right\rangle_ {\gamma (t)}} d t. \tag {1}
$$

# 2.2 DISTRIBUTIONS OVER RIEMANNIAN MANIFOLD

Given a squared distance function  $d_{\mathcal{M}}^2: \mathcal{M} \times \mathcal{M} \to \mathbb{R}_{>0}$  of a Riemannian manifold  $\mathcal{M}$ , the probability density function of the Riemannian normal distribution can be computed by:

$$
p _ {\boldsymbol {\mu}, \sigma} (\mathbf {z}) = \frac {1}{Z ^ {\mathcal {M}}} \exp \left(- \frac {d _ {\mathcal {M}} ^ {2} (\mathbf {z} , \boldsymbol {\mu})}{2 \sigma^ {2}}\right), \tag {2}
$$

where  $\pmb{\mu} \in \mathcal{M}$  is the Fréchet mean of the distribution, and  $\sigma \in \mathbb{R}_{>0}$  is the dispersion parameter and  $Z^{\mathcal{M}}$  is the normalizing factor. This is known to be preserving the maximum entropy property of the Gaussian distribution (Pennec, 2006). Note that the distribution requires computing the integral shown in Equation 1, which often does not have an analytic solution. In some special cases, one can compute the distance analytically but the computation is intractable in general. Mathieu et al. (2019) propose a rejection sampling method of the Riemannian normal defined on the Poincaré disk model, which we call a Poincaré normal distribution.

An alternative to Riemannian normal is the wrapped normal distribution. The wrapped normal distribution is constructed by transforming a sample from Gaussian distribution via parallel transportation and an exponential map:

$$
\mathbf {z} = \exp_ {\boldsymbol {\mu}} \left(\mathrm {P T} _ {\mathbf {0} _ {\mathcal {M}} \rightarrow \boldsymbol {\mu}} (f (\mathbf {v}))\right), \mathbf {v} \sim \mathcal {N} (\mathbf {0}, \Sigma), \tag {3}
$$

where  $\pmb{\mu} \in \mathcal{M}$  is the mean vector of the distribution,  $\mathbf{0}_{\mathcal{M}}$  is the origin of  $\mathcal{M}$ ,  $f(\cdot)$  maps a Euclidean vector to a tangent vector of  $\mathbf{0}_{\mathcal{M}}$ , and  $\mathbf{v}$  is a sample obtained from Euclidean normal with the zero mean and covariance  $\Sigma$ . The probability density of the sample can be computed by using the change of variable technique. Note that  $f(\cdot)$  is well-defined in hyperbolic spaces. For example, in the Lorentz model, we concatenate zero at the first dimension of the vector, and in the Poincaré disk model, it is an identity function. (Nagano et al., 2019) propose wrapped normal distribution on hyperbolic space, and we call it hyperbolic wrapped normal distribution.

# 2.3 STATISTICAL MANIFOLD

The parameter manifold  $\mathcal{M}$  of the probability distributions  $p_{\theta}:\mathcal{X}\to \mathbb{R}$ , where  $\theta \in \mathcal{M}$ , equipped with the Fisher information metric (FIM) forms a Riemannian manifold (Rao, 1992). The FIM is defined as:

$$
g _ {i j} (\pmb {\theta}) = \int_ {\mathcal {X}} \frac {\partial \log p _ {\theta} (x)}{\partial \theta_ {j}} \frac {\partial \log p _ {\theta} (x)}{\partial \theta_ {j}} p _ {\theta} (x) d x.
$$

In the parameter space of univariate Gaussian distributions  $\{(\mu, \sigma) \mid \mu \in \mathbb{R}, \sigma \in \mathbb{R}_{>0}\}$ , the FIM can be simplified as two-dimensional diagonal matrix  $\sigma^{-2} \mathrm{diag}(1, 0.5)$  (Costa et al., 2015). The diagonal form of the FIM implies that the Riemannian manifold with  $\{(\mu, \sigma)\}$  has the same set of points as the manifold of the Poincaré half-plane, but with different curvature of value  $-0.5$ .

The parameter space of the  $n$ -dimensional diagonal Gaussian distributions becomes the product of  $n$  manifolds of the parameter space of univariate Gaussian distributions. The operations on the product of the Riemannian manifolds  $\bigotimes_{i=1}^{n}\mathcal{M}_{i}$  are defined manifold-wise. For example, an exponential map applied on a point  $(p_i)_{i=1}^n \in \bigotimes_{i=1}^n\mathcal{M}_i$ , with tangent vector  $v_i \in \mathcal{T}_{p_i}\mathcal{M}_i$  for each  $i \in \{1, \dots, n\}$ , can be represented as  $(\exp_{p_i}(v_i))_{i=1}^n$ .

# 2.4 STATISTICAL DISTANCE

The statistical distance is the distance, but may not be a metric, between two statistical objects such as random variables and probability density function. The statistical distance can provide similarities between two probability density functions.

On a statistical manifold equipped with FIM, a statistical distance called the Fisher-Rao distance can be well-derived. The Fisher-Rao distance of the statistical manifold is the Riemannian distance induced from the Fisher information metric using Equation 1. For example, the Fisher-Rao distance in the statistical manifold with the univariate Gaussian distribution can be easily induced using the Riemannian distance of the Poincaré half-plane model, where the Riemannian metric is similar (Costa et al., 2015).

Kullback-Leibler (KL) divergence is another widely-used statistical distance, which is defined as  $D_{\mathrm{KL}}(p(x)\parallel q(x))\coloneqq \int_{x}p(x)\log \frac{p(x)}{q(x)} dx$  for two distributions  $p(x),q(x)$  in the same statistical manifold. For example, the KL divergence for two univariate Gaussian distributions,  $\mathcal{N}(\mu_1,\sigma_1)$  and  $\mathcal{N}(\mu_2,\sigma_2)$ , can be computed as:

$$
D _ {\mathrm {K L}} \left(\mathcal {N} (\mu_ {1}, \sigma_ {1}) \parallel \mathcal {N} (\mu_ {2}, \sigma_ {2})\right) = \log \frac {\sigma_ {2}}{\sigma_ {1}} + \frac {\sigma_ {1} ^ {2} + \left(\mu_ {1} - \mu_ {2}\right) ^ {2}}{2 \sigma_ {2} ^ {2}} - \frac {1}{2}.
$$

For the  $n$ -dimensional diagonal Gaussians, the KL divergence is calculated by summing the KL divergence of the univariate Gaussians for each dimension. One notable property of KL divergence is that it can locally approximate the squared Fisher-Rao distance.

# 3 METHOD

In this section, we first derive a reparameterization of the Gaussian distribution to form a statistical manifold with an arbitrary curvature. We then propose a Pseudo Gaussian manifold (PGM) normal distribution. Finally, we suggest a new variant of the variational auto-encoder, whose latent space is defined over the statistical manifold.

# 3.1 MANIFOLD WITH ARBITRARY CURVATURE

As shown in Section 2.3, the univariate Gaussian distributions form a statistical manifold with a negative half curvature, whose manifold is the same as the manifold of Poincaré half-plane. Previous studies on hyperbolic spaces emphasize the importance of having an arbitrary curvature (Skopek et al., 2019; Mathieu et al., 2019). These works empirically show that the generalization performances of hyperbolic VAEs can be improved with varying curvatures.

We show that the statistical manifold of univariate Gaussian can have an arbitrary curvature by reparameterizing the Gaussian distribution properly. Let  $\mathcal{N}(\sqrt{2} c\mu, \sigma)$  be the reparameterized Gaussian distribution with additional parameter  $c > 0$ . The reparameterization leads to the FIM of  $\sigma^{-2}\mathrm{diag}(1, c)$  showing that the curvature of the statistical manifold is  $-c$ .

With the arbitrary curvature, we also verify that the KL divergence still approximates the Riemannian distance as:

$$
\frac {D _ {\mathrm {K L}} \left(\mathcal {N} (\sqrt {2 c} (\mu + d \mu) , \sigma + d \sigma) \| \mathcal {N} (\sqrt {2 c} \mu , \sigma)\right)}{2 c} = \frac {1}{2} \binom {d \mu} {d \sigma} ^ {T} \left( \begin{array}{c c} \frac {1}{\sigma^ {2}} & 0 \\ 0 & \frac {1}{c \sigma^ {2}} \end{array} \right) \binom {d \mu} {d \sigma} + \mathcal {O} \left((d \sigma) ^ {3}\right), \tag {4}
$$

where the first term is the squared Riemannian norm of the vector  $(d\mu, d\sigma)$  in the manifold, which approximates the squared Fisher-Rao distance between  $(\mu, \sigma)$  and  $(\mu + d\mu, \sigma + d\sigma)$ . The derivation of FIM and KL divergence with the reparameterized normal is described in Appendix A.1 and A.2. We call the statistical manifold with Gaussian distributions having a curvature of  $-c$  as the Gaussian manifold  $\mathcal{G}_c$ .

# 3.2 PSEUDO GAUSSIAN MANIFOLD NORMAL DISTRIBUTION

We propose a pseudo Gaussian manifold normal distribution (PGM normal) defining a distribution over the Gaussian manifold. Let  $(\mu, \sigma) \in \mathcal{G}$  be a point in the Gaussian manifold. Inspired by the Riemannian normal, we define the probability density function of PGM normal distribution with KL-divergence as:

$$
\mathcal {K} _ {c} (\mu , \sigma ; \alpha , \beta , \gamma^ {2}) = \frac {(\sigma / \beta) ^ {3}}{Z (c , \gamma^ {2})} \exp \left(- \frac {D _ {\mathrm {K L}} (\mathcal {N} (\sqrt {2 c} \cdot \mu , \sigma) \| \mathcal {N} (\sqrt {2 c} \cdot \alpha , \beta))}{(\sqrt {2 c} \cdot \gamma) ^ {2}}\right), \tag {5}
$$

where  $\alpha, \beta$ , and  $\gamma^2$  are the parameters of the distribution, and  $-c$  is the curvature. As shown in the previous section, the KL divergence approximate the Fisher-Rao distance from Gaussian distribution  $\mathcal{N}(\sqrt{2c} \cdot \alpha, \beta)$  to  $\mathcal{N}(\sqrt{2c} \cdot \mu, \sigma)$ . Therefore, the PGM normal accounts for the geometric structure of the Gaussian distributions.

![](images/7e565689c743a59d868de2e6afa881c29cdc9bf98e95f4fc5a2f11df050f92c6.jpg)  
Figure 1: An exemplar architecture of GM-VAE. The illustration shows the architecture of GM-VAE which sets the latent space to a three-dimensional diagonal Gaussian manifold. The encoder outputs parameters of the PGM normal, which are the points of the Gaussian manifold. The gray line refers to the sampling process. The decoder reconstructs data from the samples.

The factorization of the probability density function in Equation 5 multiplied with the square root of the determinant of the metric tensor shows the advantages of the PGM normal, which can be written as:

$$
\mathcal {K} _ {c} (\mu , \sigma ; \alpha , \beta , \gamma^ {2}) \cdot \sqrt {\det  (g)} = \mathcal {N} (\mu ; \alpha , \beta^ {2} \gamma^ {2}) \cdot \operatorname {G a m m a} \left(\sigma^ {2}; \frac {1}{4 c \gamma^ {2}} + 1, \frac {1}{4 c \beta^ {2} \gamma^ {2}}\right), \tag {6}
$$

where  $\mathrm{Gamma}(z;a,b) = \frac{b^a}{\Gamma(a)} z^{a - 1}\exp (-bz)$  and  $g$  is the Fisher information metric of the Gaussian manifold. Note that the factorization has the same form as the well-known conjugate prior to the Gaussian distribution. In that sense, the PGM normal incorporates the geometric structure into the prior distribution explicitly. Thanks to the properties of Gaussian and Gamma distribution, the PGM normal is easy to sample and has a closed-form KL divergence. The detailed derivation is available in Appendix B.

# 3.3 GAUSSIAN MANIFOLD VAE

We propose a Gaussian manifold VAE (GM-VAE) whose latent space is defined over the Gaussian manifold. To be specific, we place a PGM normal prior over the latent space of the VAE and add a proper geometric transformation at the last layer of the encoder and the first layer of the decoder for the conversion between the Euclidean space and Gaussian manifold.

The evidence lower bound (ELBO) of the GM-VAE can be formalized with the Gaussian-manifold  $\{(\pmb {\mu},\Sigma)\mid \pmb {\mu}\in \mathbb{R}^n,\Sigma \in \mathbb{R}_{>0}^n\}$  as:

$$
\mathbb {E} _ {q _ {\phi} (\boldsymbol {\mu}, \Sigma | \mathbf {x}) \cdot \sqrt {\det  (g)}} \left[ \log p _ {\theta} (\mathbf {x} \mid \boldsymbol {\mu}, \Sigma) \right] - D _ {\mathrm {K L}} \left(q _ {\phi} (\boldsymbol {\mu}, \Sigma \mid \mathbf {x}) \cdot \sqrt {\det  (g)} \| p (\boldsymbol {\mu}, \Sigma) \cdot \sqrt {\det  (g)}\right), \tag {7}
$$

where  $p_{\theta}(\mathbf{x}\mid \boldsymbol {\mu},\Sigma)$  is the decoder network,  $q_{\phi}(\boldsymbol {\mu},\Sigma \mid \mathbf{x})$  is the encoder network and  $p(\boldsymbol {\mu},\Sigma)$  is the prior. The variational distribution is set to  $q_{\phi}(\boldsymbol {\mu},\Sigma \mid \mathbf{x}) = \mathcal{K}(\alpha_{\phi}(\mathbf{x}),\beta_{\phi}(\mathbf{x}),\gamma_{\phi}^{2}(\mathbf{x}))$ , where  $\alpha_{\theta}(\mathbf{x})\in \mathbb{R}^n$  and  $\beta_{\phi}(\mathbf{x}),\gamma_{\phi}^{2}(\mathbf{x})\in \mathbb{R}_{>0}^{n}$ , and the prior is set to  $p(\boldsymbol {\mu},\Sigma) = \mathcal{K}(\mathbf{0},I,I)$  in our experiments. The training for the parameters of GM-VAE ( $\theta$  and  $\phi$ ) is to maximize the ELBO.

<table><tr><td colspan="2">Algorithm 1 Encoder</td><td colspan="2">Algorithm 2 Decoder</td></tr><tr><td colspan="2">Input Input data x, Encoding layers Enc(·)</td><td colspan="2">Input Sample (μ, σ) ~ K(·), Decoding layers Dec(·)</td></tr><tr><td colspan="2">Output Parameter (α, β) ∈ Gc, γ2 ∈ R&gt;0</td><td colspan="2">Output Reconstruction x&#x27;</td></tr><tr><td>1: v, γ2 = Enc(x)</td><td>▷ v ∈ E</td><td>1: v = Tc-1(μ, σ)</td><td>▷ (μ, σ) ∈ Gc, v ∈ L2c</td></tr><tr><td>2: v = exp0c(f(v))</td><td>▷ v ∈ L2c</td><td>2: v = log0c(v)</td><td>▷ v ∈ E</td></tr><tr><td>3: (α, β) = Tc(v)</td><td>▷ (α, β) ∈ Gc</td><td>3: x&#x27; = Dec(v)</td><td></td></tr><tr><td>4: return (α, β), γ2</td><td></td><td>4: return x&#x27;</td><td></td></tr></table>

Geometric transformations on GM-VAE Mathieu et al. (2019) propose a transformation from a Euclidean space to the Poincaré disk to define a latent space over the Poincaré disk. We propose a novel transformation in VAE from a Euclidean space to the Gaussian manifold and vice versa.

For a numerically stable transformation between two spaces, we adopt operations defined on the Lorentz model, which is isometric to the half-plane manifold (Nickel & Kiela, 2018). The isometry  $T_{c}:\mathcal{L}_{c}^{2}\to \mathcal{G}_{c}$  between the two-dimensional Lorentz model with curvature  $-c$  and the Gaussian manifold with curvature  $-c$  can be defined as:

$$
T _ {c} ((t, x, y)) = \left(\frac {- y}{\sqrt {c} (t - x)}, \frac {1}{\sqrt {c} (t - x)}\right),
$$

and the inverse is as:

$$
T _ {c} ^ {- 1} ((x, y)) = \left(\frac {1 + c x ^ {2} + y ^ {2}}{2 \sqrt {c} y}, \frac {- 1 + c x ^ {2} + y ^ {2}}{2 \sqrt {c} y}, - \frac {x}{y}\right).
$$

In the encoder, we convert the output of the last layer, which is in the Euclidean space, to the Lorentz model using the exponential map at the origin and then convert it to the Gaussian manifold using  $T_{c}$ . In the decoder, we convert the input of the first layer, which is in the Gaussian manifold, to the Lorentz model using the inverse of the transformation  $T_{c}^{-1}$  and then convert it to the Euclidean space using the log map at the origin of the Lorentz model. Figure 1 illustrates the architecture of GM-VAE and the pseudo code for the encoder and decoder are shown in Algorithm 1 and Algorithm 2.

Remark Unlike a typical VAE, where the latent space consists of samples from a Gaussian distribution, the latent space of GM-VAE consists of a set of Gaussian distributions. With this aspect, GM-VAE can be considered as a hierarchical VAE with an additional prior over the Gaussian prior. However, instead of sampling another latent variable from the latent distribution, we directly transform the latent distribution itself to  $\mathbf{x}$  in the decoder network via transformation from hyperbolic to Euclidean space. From this perspective, GM-VAE can also be considered as a variant of Poincaré VAE (Mathieu et al., 2019), whose latent space can be interpreted via the Gaussian manifold.

# 4 RELATED WORK

Information geometry on VAE Focusing on the virtue of bridging probability theory and differential geometry, the adaptation of information geometry to the deep learning framework has been investigated in various aspects (Karakida et al., 2019; Bay & Sengupta, 2017; Gomes et al., 2022). Having said that, Han et al. (2020) show that the training process of VAE can be seen as minimizing the distance between the two statistical manifolds: manifolds with the parameters of the decoder and the encoder. Not only can the parameters but the outputs from the VAE decoder be modeled as probability distributions. Arvanitidis et al. (2021) suggest a method of using the pull-back metric defined with arbitrary decoders on latent space. Our work focuses more on the statistical manifolds lying on the outputs of the encoder with the benefits from the information geometry.

VAE with Riemannian manifold latent space The latent space of VAE reflects the geometrical property of the representations of the data. The efficacy of setting the latent space to be hyperbolic space (Mathieu et al., 2019; Nagano et al., 2019; Cho et al., 2022) or spherical space (Xu & Durrett, 2018; Davidson et al., 2018) has been verified for various datasets. Skopek et al. (2019) further extends the approach to enable the latent space to be the product of Riemannian manifolds with

Table 1: A comparison of the PGM normal (ours) with the commonly-used distributions on the hyperbolic space: Hyperbolic wrapped normal and Poincaré normal. Our method enables the easy sampling and computation of closed-form KL, with the utilization of the information geometry.  

<table><tr><td></td><td>Easy sampling</td><td>Information geometry</td><td>Closed-form KL</td></tr><tr><td>Hyperbolic wrapped normal</td><td>○</td><td>×</td><td>×</td></tr><tr><td>Poincaré normal</td><td>△</td><td>○</td><td>×</td></tr><tr><td>PGM normal (ours)</td><td>○</td><td>○</td><td>○</td></tr></table>

different learnable curvatures. On top of these arts, we explore the method of setting the latent space to be a diagonal Gaussian manifold, which is isometric to the product of the hyperbolic space, providing a novel viewpoint on prior work with information geometry.

Distributions on the hyperbolic space Defining a distribution in the hyperbolic space with easy sampling is challenging. Nagano et al. (2019) suggests hyperbolic wrapped normal distribution from the observation that the tangent space is Euclidean space. Leveraging operations defined on the tangent spaces, e.g., parallel transport, enables an easy sampling algorithm. Mathieu et al. (2019) propose a sampling method for the Riemannian normal defined on the Poincaré disk model using rejection sampling. This method rejects the pathological samples and enables accurate sampling from the distribution, but this demands a high amount of time complexity. These distributions are applied in many cases (Cho et al., 2022; Skopek et al., 2019; Mathieu & Nickel, 2020) but suffer from stability issues because of the absence of closed-form KL divergence. Our proposed distribution, however, not only share the common merits but also has overcome the stability problem with closed-form KL divergence. Table 1 summarizes the properties of each distribution.

# 5 EXPERIMENTS

In this section, we compare the performance of GM-VAE with the three baselines: Euclidean VAE, hyperbolic wrapped normal VAE (HWN VAE), and Poincaré VAE. The Euclidean VAE is the standard VAE with Euclidean latent space. The HWN VAE uses the product of two-dimensional Lorentz models as a latent space and uses the hyperbolic wrapped normal to model the prior and variational distributions. The Poincaré VAE uses the product of two-dimensional Poincaré disk models as a latent space and uses the Poincaré normal to model the prior and variational distributions. The Euclidean VAE, HWN VAE, and Poincaré VAE are denoted as  $\mathcal{E}$ -VAE,  $\mathcal{L}$ -VAE,  $\mathcal{P}$ -VAE, respectively in the following results.

# 5.1 DENSITY ESTIMATION

We first conduct a density estimation task to check the generalization ability of different models. We use three datasets: binarized-MNIST (Deng, 2012), binarized-Omniglot (Lake et al., 2015), and the images from Atari 2600 Breakout with binarization (binarized-Breakout) (Nagano et al., 2019). The binarized-Breakout are collected from plays with a pre-trained Deep Q-Network (Mnih et al., 2015). The size of images are  $28 \times 28$ ,  $28 \times 28$ , and  $80 \times 80$  for binarized-MNIST, binarized-Omniglot, and binarized-Breakout, respectively. The value of the threshold for binarization is set to 0.5, 0.5, and 0.1 for binarized-MNIST, binarized-Omniglot, and binarized-Breakout, respectively; the threshold

Table 2: Density estimation on real-world datasets.  $d$  denotes the latent dimension. We report the negative test log-likelihoods of average 10 runs for binarized-MNIST and binarized-Omniglot, and an average 5 runs for binarized-Breakout with the  $95\%$  confidence interval. N/A in the log-likelihood indicates that the results are not available due to the failure of all runs, and N/A in the standard deviation indicates the results are not available due to failures of some runs. The best results are bolded.  

<table><tr><td></td><td>d</td><td>E-VAE</td><td>L-VAE</td><td>P-VAE</td><td>GM-VAE (c=1)</td><td>GM-VAE (c=1/2)</td><td>GM-VAE (c=3/2)</td></tr><tr><td rowspan="3">MNIST</td><td>10</td><td>79.60±.13</td><td>79.95±.19</td><td>80.52±.20</td><td>80.34±.30</td><td>80.10±.20</td><td>80.38±.18</td></tr><tr><td>20</td><td>74.48±.46</td><td>73.67±.32</td><td>72.95±.11</td><td>73.27±.22</td><td>73.31±.29</td><td>73.28±.19</td></tr><tr><td>30</td><td>73.80±.07</td><td>73.46±.23</td><td>72.94±.10</td><td>73.49±.13</td><td>73.35±.16</td><td>73.55±.22</td></tr><tr><td rowspan="3">Omniglot</td><td>10</td><td>136.53±.30</td><td>136.25±.36</td><td>134.95±.47</td><td>134.01±.28</td><td>135.20±.21</td><td>133.79±.30</td></tr><tr><td>20</td><td>121.18±.33</td><td>119.95±.40</td><td>117.79±.13</td><td>118.79±.53</td><td>118.73±.39</td><td>119.03±.57</td></tr><tr><td>30</td><td>118.67±.67</td><td>117.16±.48</td><td>115.09±.56</td><td>117.97±.35</td><td>117.70±.47</td><td>117.95±.37</td></tr><tr><td rowspan="3">Breakout</td><td>24</td><td>50.37±.46</td><td>50.82N/A</td><td>N/A</td><td>49.35±.67</td><td>50.82±.92</td><td>49.88±.51</td></tr><tr><td>28</td><td>48.07±.20</td><td>48.74N/A</td><td>N/A</td><td>47.01±.31</td><td>48.31±.47</td><td>46.82±.77</td></tr><tr><td>32</td><td>48.06±.18</td><td>48.64N/A</td><td>N/A</td><td>47.04±.23</td><td>48.13±.30</td><td>46.88±.25</td></tr></table>

Table 3: Ablation study on the geometric transformations of GM-VAE. Vanilla denotes the models without the geometric transformations, and Geo denotes the models with the geometric transformations. The geometric transformations enhance the generalization performance in most cases.  

<table><tr><td rowspan="2"></td><td rowspan="2">d</td><td colspan="2">c = 1</td><td colspan="2">c = 1/2</td><td colspan="2">c = 3/2</td></tr><tr><td>Vanilla</td><td>Geo</td><td>Vanilla</td><td>Geo</td><td>Vanilla</td><td>Geo</td></tr><tr><td rowspan="3">MNIST</td><td>10</td><td>80.28±.17</td><td>80.34±.30</td><td>80.28±.19</td><td>80.10±.20</td><td>80.38±.20</td><td>80.38±.18</td></tr><tr><td>20</td><td>75.33±.75</td><td>73.27±.22</td><td>74.94±.58</td><td>73.31±.29</td><td>75.74±.70</td><td>73.28±.19</td></tr><tr><td>30</td><td>74.63±.08</td><td>73.49±.13</td><td>74.48±.50</td><td>73.35±.16</td><td>75.05±.43</td><td>73.55±.22</td></tr><tr><td rowspan="3">Omniglot</td><td>10</td><td>135.12±.35</td><td>134.01±.28</td><td>135.47±.24</td><td>135.20±.21</td><td>134.69±.37</td><td>133.79±.30</td></tr><tr><td>20</td><td>122.24±.67</td><td>118.79±.53</td><td>121.10±.59</td><td>118.73±.39</td><td>122.44±.61</td><td>119.03±.57</td></tr><tr><td>30</td><td>120.85±.33</td><td>117.97±.35</td><td>119.48±.56</td><td>117.70±.47</td><td>121.23±.68</td><td>117.95±.37</td></tr><tr><td rowspan="3">Breakout</td><td>24</td><td>51.69±.20</td><td>49.35±.67</td><td>51.04±.72</td><td>50.82±.92</td><td>51.57±.50</td><td>49.88±.51</td></tr><tr><td>28</td><td>49.56±.69</td><td>47.01±.31</td><td>49.51±.51</td><td>48.31±.47</td><td>49.45±.63</td><td>46.82±.77</td></tr><tr><td>32</td><td>49.45±.29</td><td>47.04±.23</td><td>49.03±.38</td><td>48.13±.30</td><td>49.07±.54</td><td>46.88±.25</td></tr></table>

for binarized-Breakout is determined to visualize the components clear. The other details on the implementation and experimental setups are described in Appendix D.

The results are reported at Table 2. In binarized-MNIST and binarized-Omniglot, the models learned on the product hyperbolic space and the Gaussian manifold mostly outperform the Euclidean VAE. In binarized-Breakout, the GM-VAE with curvature values 1 and  $3/2$  outperform the baselines while the Poincaré VAE fails to run in all the settings and the HWN VAE fails to run in some of the settings due to numerical issues, which we further investigate in details.

Numerical stability We conduct an analysis of the numerical stability of the PGM normal distribution compared to the HWN and Poincaré normal. During the density estimation experiment, the HWN VAE and Poincaré VAE are often shown to be numerically unstable and fail to run in binarized-Breakout. Similar observations have been reported in several previous works (Mathieu et al., 2019; Chen et al., 2021; Skopek et al., 2019).

The hyperbolic wrapped normal uses the exponential map when transforming the output of the encoder to the Lorentz model and during the sampling, as described in Equation 3. The overlapped Lorentz model exponential map often causes an overflow. In the training of Poincaré VAE, the KL divergence between the variational distribution and the prior distribution needs to be approximated by the log-probability of the samples due to the absence of closed-form KL divergence in Poincaré normal. To compute the log probability of a given sample, the distance between two Poincaré disk model points, which are the sample and the Fréchet mean of the distribution, needs to be calculated, where the denominator term is numerically unstable. The PGM normal, on the other hand, is free

![](images/24441cc26af91879fcfa9990e5e35ed0667f07d84f323834d0093288e59353b7.jpg)  
(a) t-SNE visualization.

![](images/a8a3fbcc98bcc319766d39309cf75c7a53504f9ed3c879182100af765e9dc7b1.jpg)  
Figure 2: Analysis of the learned latent space of GM-VAE with binarized-MNIST. (a) t-SNE visualization of the representation with respect to the class labels. (b) Increasing the value of  $\beta$ , along the gray line, results in an increasing degree of uncertainty in the reconstructed images.  
(b) Latent traversal of representations.

![](images/6e89091a7506a459dc74707d4a5ed149f0517ee209c39311747bafbb73fa5291.jpg)  
(a) Latent traversal of representations.

![](images/a0a00724c01886ea487514e7a9f4adfab993c6efacd182bf5a1022b91fb8b38c.jpg)  
Figure 3: Analysis of the latent space learned from GM-VAE with the binarized-Breakout. (a) Reconstructing the representations, along the gray line, shows a similar hierarchical structure, where (b) the hierarchy between the images is expressed as the dotted line.  
(b) Hierarhcy in the data.

from instability with the help of stability when using log-covariance. Please check the detailed arguments with equations in Appendix E.

**Geometric transformations** We conduct an ablation study on the geometric transformations of GM-VAE. We compare the setting of GM-VAEs incorporating the geometric transformations to the setting of GM-VAEs using only exponential function to send the output of the encoder to the Gaussian manifold but no additional geometric transformation at the first layer of the decoder. The results are in Table 3. We can see that the geometric transformations enhance the performance of the GM-VAE, except for two results but with similar performance.

# 5.2 LATENT SPACE ANALYSIS

To check whether the latent representation coincides with the known labels, we first plot the latent spaces of binarized-MNIST via t-SNE (van der Maaten & Hinton, 2008) visualization, with representations from all dimensions. The visualization shown in Figure 2a presents that the label semantics are well clustered in the learned latent space. We also analyze the changes in the reconstructed images along the geodesic of the latent space. Figure 2b shows the reconstructed images from a geodesic interpolation between two latent representations, with a fixed value of  $\alpha$ . The interpolation of the latent space is performed within one dimension while fixing the value of representations in other dimensions. As  $\beta$  increases, the reconstructed images become ambiguous, matching our intuition on the role of variance. Reconstruction images with a fixed value of  $\beta$  is available at Appendix F.

Figure 3 shows the analysis with binarized-Breakout. The images in the binarized-Breakout possess a hierarchy as the cumulative rewards and the amount of the breakout bricks, or the portion of blank space in the image, are highly correlated (Nagano et al., 2019). We observe that there is a high correlation between  $\beta$  and the hierarchy. For example, as shown in Figure 3a, increasing  $\beta$  reconstructs a more general image in the hierarchy. The highest Pearson correlation between the  $\beta$  values and the negative cumulative reward is 0.655. Again, as  $\beta$  represents the variance, we conjecture the increasing variance induces a more general image in the dataset.

# 6 CONCLUSION

In this work, we propose a novel method of representation learning with GM-VAE, utilizing the Gaussian manifold for the latent space. With the newly-proposed PGM normal distribution defined over Gaussian manifold, which shows better stability and ease of sampling compared to the commonly-used ones, we verify the efficacy of our method on several real-world datasets. Our analysis of latent space and representations exhibits that GM-VAE is beneficial for capturing both the geometrical structures and probabilistic semantics. We believe that the connection between the statistical manifold and hyperbolic spaces provides a new insight to the research community and hope to see more interesting connections and analyses in the future.

# REFERENCES

Georgios Arvanitidis, Miguel González-Duque, Alison Pouplin, Dimitris Kalatzis, and Søren Hauberg. Pulling back information geometry. arXiv preprint arXiv:2106.05367, 2021.  
Alessandro Bay and Biswa Sengupta. Geoseq2seq: Information geometric sequence-to-sequence networks. arXiv preprint arXiv:1710.09363, 2017.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
Weize Chen, Xu Han, Yankai Lin, Hexu Zhao, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Fully hyperbolic neural networks. arXiv preprint arXiv:2105.14686, 2021.  
Seunghyuk Cho, Juyong Lee, Jaesik Park, and Dongwoo Kim. A rotated hyperbolic wrapped normal distribution for hierarchical representation learning. arXiv preprint arXiv:2205.13371, 2022.  
Sueli IR Costa, Sandra A Santos, and Joao E Strapasson. Fisher information distance: A geometrical reading. Discrete Applied Mathematics, 197:59-69, 2015.  
Tim R Davidson, Luca Falorsi, Nicola De Cao, Thomas Kipf, and Jakub M Tomczak. Hyperspherical variational auto-encoders. arXiv preprint arXiv:1804.00891, 2018.  
Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141-142, 2012.  
Eduardo Dadalto Camara Gomes, Florence Alberge, Pierre Duhamel, and Pablo Piantanida. Igeood: An information geometry approach to out-of-distribution detection. arXiv preprint arXiv:2203.07798, 2022.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139-144, 2020.  
Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, et al. Hyperbolic attention networks. arXiv preprint arXiv:1805.09786, 2018.  
Tian Han, Jun Zhang, and Ying Nian Wu. From em-projections to variational auto-encoder. 2020.  
Ryo Karakida, Shotaro Akaho, and Shun-ichi Amari. Universal statistics of fisher information in deep neural networks: Mean field approach. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1032–1041. PMLR, 2019.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332-1338, 2015. doi: 10.1126/science.aab3050. URL https://www.science.org/doi/abs/10.1126/science.aab3050.  
Emile Mathieu and Maximilian Nickel. Riemannian continuous normalizing flows. Advances in Neural Information Processing Systems, 33:2503-2515, 2020.  
Emile Mathieu, Charline Le Lan, Chris J Maddison, Ryota Tomioka, and Yee Whye Teh. Continuous hierarchical representations with poincaré variational auto-encoders. Advances in neural information processing systems, 32, 2019.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529-533, 2015.

Yoshihiro Nagano, Shoichiro Yamaguchi, Yasuhiro Fujita, and Masanori Koyama. A wrapped normal distribution on hyperbolic space for gradient-based learning. In International Conference on Machine Learning, pp. 4693-4702. PMLR, 2019.  
Maximillian Nickel and Douwe Kiela. Learning continuous hierarchies in the lorentz model of hyperbolic geometry. In International Conference on Machine Learning, pp. 3779-3788. PMLR, 2018.  
Xavier Pennec. Intrinsic statistics on riemannian manifolds: Basic tools for geometric measurements. Journal of Mathematical Imaging and Vision, 25(1):127-154, 2006.  
C. Radhakrishna Rao. Information and the Accuracy Attainable in the Estimation of Statistical Parameters, pp. 235-247. Springer New York, New York, NY, 1992. ISBN 978-1-4612-0919-5. doi: 10.1007/978-1-4612-0919-5_16. URL https://doi.org/10.1007/978-1-4612-0919-5_16.  
Ondrej Skopek, Octavian-Eugen Ganea, and Gary Bécigneul. Mixed-curvature variational autoencoders. arXiv preprint arXiv:1911.08411, 2019.  
Alexandru Tifrea, Gary Bécigneul, and Octavian-Eugen Ganea. Poincar\`e glove: Hyperbolic word embeddings. arXiv preprint arXiv:1810.06546, 2018.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9:2579-2605, 2008. URL http://www.jmlr.org/papers/v9/vandermaaten08a.html.  
Jiacheng Xu and Greg Durrett. Spherical latent spaces for stable variational autoencoders. arXiv preprint arXiv:1808.10805, 2018.
