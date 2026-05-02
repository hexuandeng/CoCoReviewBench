# DISCRETE VARIATIONAL AUTOENCODERS

Jason Tyler Rolfe

D-Wave Systems

Burnaby, BC V5G-4M9, Canada

jrolfe@dwavesys.com

# ABSTRACT

Probabilistic models with discrete latent variables naturally capture datasets composed of discrete classes. However, they are difficult to train efficiently, since backpropagation through discrete variables is generally not possible. We present a novel method to train a class of probabilistic models with discrete latent variables using the variational autoencoder framework, including backpropagation through the discrete latent variables. The associated class of probabilistic models comprises an undirected discrete component and a directed hierarchical continuous component. The discrete component captures the distribution over the disconnected smooth manifolds induced by the continuous component. As a result, this class of models efficiently learns both the class of objects in an image, and their specific realization in pixels, from unsupervised data; and outperforms state-of-the-art methods on the permutation-invariant MNIST, Omniglot, and Caltech-101 Silhouettes datasets.

# 1 INTRODUCTION

Unsupervised learning of probabilistic models is a powerful technique, facilitating tasks such as denoising and inpainting, and regularizing supervised tasks such as classification (Hinton et al., 2006; Salakhutdinov & Hinton, 2009; Rasmus et al., 2015). Many datasets of practical interest are projections of underlying distributions over real-world objects into an observation space; the pixels of an image, for example. When the real-world objects are of discrete types subject to continuous transformations, these datasets comprise multiple disconnected smooth manifolds. For instance, natural images change smoothly with respect to the position and pose of objects, as well as scene lighting. At the same time, it is extremely difficult to directly transform the image of a person to one of a car while remaining on the manifold of natural images.

It would be natural to represent the space within each disconnected component with continuous variables, and the selection amongst these components with discrete variables. In contrast, most state-of-the-art probabilistic models use exclusively discrete variables — as do DBMs (Salakhutdinov & Hinton, 2009), NADEs (Larochelle & Murray, 2011), sigmoid belief networks (Spiegelhalter & Lauritzen, 1990; Bornschein et al., 2015), and DARNs (Gregor et al., 2014) — or exclusively continuous variables — as do VAEs (Kingma & Welling, 2014; Rezende et al., 2014) and GANs (Goodfellow et al., 2014). Moreover, it would be desirable to apply the efficient variational autoencoder framework to models with discrete values, but this has proven difficult, since backpropagation through discrete variables is generally not possible (Bengio et al., 2013; Raiko et al., 2015).

We introduce a novel class of probabilistic models, comprising an undirected graphical model defined over binary latent variables, followed by multiple directed layers of continuous latent variables. This class of models captures both the discrete class of the object in an image, and its specific continuously deformable realization. Moreover, we show how these models can be trained efficiently using the variational autoencoder framework, including backpropagation through the binary latent variables. We ensure that the evidence lower bound remains tight by incorporating a hierarchical approximation to the posterior distribution of the latent variables, which can model strong correlations. Since these models efficiently marry the variational autoencoder framework with discrete latent variables, we call them discrete variational autoencoders (discrete VAEs).

# 1.1 VARIATIONAL AUTOENCODERS ARE INCOMPATIBLE WITH DISCRETE DISTRIBUTIONS

Conventionally, unsupervised learning algorithms maximize the log-likelihood of an observed dataset under a probabilistic model. Even stochastic approximations to the gradient of the log-likelihood generally require samples from the posterior and prior of the model. However, sampling from undirected graphical models is generally intractable (Long & Servedio, 2010), as is sampling from the posterior of a directed graphical model conditioned on its leaf variables (Dagum & Luby, 1993).

In contrast to the exact log-likelihood, it can be computationally efficient to optimize a lower bound on the log-likelihood (Jordan et al., 1999), such as the evidence lower bound (ELBO;  $\mathcal{L}(x,\theta ,\phi)$ ) (Hinton & Zemel, 1994):

$$
\mathcal {L} (x, \theta , \phi) = \log p (x | \theta) - \mathrm {K L} [ q (z | x, \phi) | | p (z | x, \theta) ], \tag {1}
$$

where  $q(z|x,\phi)$  is a computationally tractable approximation to the posterior distribution  $p(z|x,\theta)$ . We denote the observed random variables by  $x$ , the latent random variables by  $z$ , the parameters of the generative model by  $\theta$ , and the parameters of the approximating posterior by  $\phi$ . The variational autoencoder (VAE) (Kingma & Welling, 2014; Rezende et al., 2014; Kingma et al., 2014) regroups the evidence lower bound of Equation 1 as:

$$
\mathcal {L} (x, \theta , \phi) = - \underbrace {\operatorname {K L} [ q (z | x , \phi) | | p (z | \theta) ]} _ {\text {K L t e r m}} + \underbrace {\mathbb {E} _ {q} [ \log p (x | z , \theta) ]} _ {\text {a u t o e n c o d i n g t e r m}}. \tag {2}
$$

In many cases of practical interest, such as Gaussian  $q(z|x)$  and  $p(z)$ , the KL term of Equation 2 can be computed analytically. Moreover, a low-variance stochastic approximation to the gradient of the autoencoding term can be obtained using backpropagation and the reparameterization trick, so long as samples from the approximating posterior  $q(z|x)$  can be drawn using a differentiable, deterministic function  $f(x,\phi,\rho)$  of the combination of the inputs, the parameters, and a set of input-and parameter-independent random variables  $\rho \sim D$ . For instance, samples can be drawn from a Gaussian distribution with mean and variance determined by the input,  $\mathcal{N}(m(x,\phi),v(x,\phi))$ , using  $f(x,\phi,\rho) = m(x,\phi) + \sqrt{v(x,\phi)}\cdot \rho$ , where  $\rho \sim \mathcal{N}(0,1)$ . When such an  $f(x,\phi,\rho)$  exists,

$$
\frac {\partial}{\partial \phi} \mathbb {E} _ {q (z | x, \phi)} [ \log p (x | z, \theta) ] \approx \frac {1}{N} \sum_ {\rho \sim \mathcal {D}} \frac {\partial}{\partial \phi} \log p (x | f (x, \rho , \phi), \theta). \tag {3}
$$

The reparameterization trick can be generalized to a large set of distributions, including nonfactorial approximating posteriors. We address this issue carefully in Appendix A, where we find that an analog of Equation 3 holds. Specifically,  $\mathcal{D}_i$  is the uniform distribution between 0 and 1, and

$$
f (x) = \mathbf {F} ^ {- 1} (x), \tag {4}
$$

where  $\mathbf{F}$  is the conditional-marginal cumulative distribution function (CDF) defined by:

$$
F _ {i} (\mathbf {x}) = \int_ {x _ {i} ^ {\prime} = - \infty} ^ {x} p \left(x _ {i} ^ {\prime} \mid x _ {1}, \dots , x _ {i - 1}\right). \tag {5}
$$

However, this generalization is only possible if the inverse of the conditional-marginal CDF exists and is differentiable.

A formulation comparable to Equation 3 is not possible for discrete distributions, such as restricted Boltzmann machines (RBMs) (Smolensky, 1986):

$$
p (z) = \frac {1}{\mathcal {Z} _ {p}} e ^ {- E _ {p} (z)} = \frac {1}{\mathcal {Z} _ {p}} \cdot e ^ {\left(z ^ {\top} W z + b ^ {\top} z\right)}, \tag {6}
$$

where  $z \in \{0,1\}^n$ ,  $\mathcal{Z}_p$  is the partition function of  $p(z)$ , and the lateral connection matrix  $W$  is triangular. Any approximating posterior that only assigns nonzero probability to a discrete domain corresponds to a CDF that is piecewise-contant. That is, the range of the CDF is a proper subset of the interval [0, 1]. The domain of the inverse CDF is thus also a proper subset of [0, 1], and its derivative is not defined, as required in Equations 3 and 4.2

In the following sections, we present the discrete variational autoencoder (discrete VAE), a hierarchical probabilistic model consisting of an RBM, followed by multiple directed layers of continuous latent variables. This model is efficiently trainable using the variational autoencoder formalism, as in Equation 3, including backpropagation through its discrete latent variables.

# 1.2 RELATED WORK

Recently, there have been many efforts to develop effective unsupervised learning techniques by building upon variational autoencoders. Importance weighted autoencoders (Burda et al., 2015a), Hamiltonian variational inference (Salimans et al., 2015), normalizing flows (Rezende & Mohamed, 2015), and variational Gaussian processes (Tran et al., 2015) improve the approximation to the posterior distribution. Ladder variational autoencoders (Sonderby et al., 2016) increase the power of the architecture of both approximating posterior and prior. Neural adaptive importance sampling (Du et al., 2015) and reweighted wake-sleep (Bornshein & Bengio, 2015) use sophisticated approximations to the gradient of the log-likelihood that do not admit direct backpropagation. Structured variational autoencoders use conjugate priors to construct powerful approximating posterior distributions (Johnson et al., 2016).

It is easy to construct a stochastic approximation to the gradient of the ELBO that admits both discrete and continuous latent variables, and only requires computationally tractable samples. Unfortunately, this naive estimate is impractically high-variance, leading to slow training and poor performance (Paisley et al., 2012). The variance of the gradient can be reduced somewhat using the baseline technique, originally called REINFORCE in the reinforcement learning literature (Mnih & Gregor, 2014; Williams, 1992; Mnih & Rezende, 2016), which we discuss in greater detail in Appendix B.

Prior efforts by Makhzani et al. (2015) to use multimodal priors with implicit discrete variables governing the modes did not successfully align the modes of the prior with the intrinsic clusters of the dataset. Rectified Gaussian units allow spike-and-slab sparsity in a VAE, but the discrete variables are also implicit, and their prior factorial and thus unimodal (Salimans, 2016). Graves (2016) computes VAE-like gradient approximations for mixture models, but the component models are assumed to be simple factorial distributions. In contrast, discrete VAEs generalize to powerful multimodal priors on the discrete variables, and a wider set of mappings to the continuous units.

# 2 BACK PROPAGATING THROUGH DISCRETE LATENT VARIABLES BY ADDING CONTINUOUS LATENT VARIABLES

When working with an approximating posterior over discrete latent variables, we can effectively smooth the conditional-marginal CDF (defined by Equation 5 and Appendix A) by augmenting the latent representation with a set of continuous random variables. The conditional-marginal CDF over the new continuous variables is invertible and its inverse is differentiable, as required in Equations 3 and 4. We redefine the generative model so that the conditional distribution of the observed variables given the latent variables only depends on the new continuous latent space. This does not alter the fundamental form of the model, or the KL term of Equation 2; rather, it can be interpreted as adding a noisy nonlinearity, like dropout (Srivastava et al., 2014) or batch normalization with a small minibatch (Ioffe & Szegedy, 2015), to each latent variable in the approximating posterior and the prior. The conceptual motivation for this approach is discussed in Appendix C.

Specifically, as shown in Figure 1a, we augment the latent representation in the approximating posterior with continuous random variables  $\zeta$ ,<sup>3</sup> conditioned on the discrete latent variables  $z$  of the RBM:

$$
q (\zeta , z | x, \phi) = r (\zeta | z) \cdot q (z | x, \phi), \quad \text {w h e r e}
$$

$$
r (\zeta | z) = \prod_ {i} r (\zeta_ {i} | z _ {i}).
$$

The support of  $r(\zeta | z)$  for all values of  $z$  must be connected, so the marginal distribution  $q(\zeta | x, \phi) = \sum_{z} r(\zeta | z) \cdot q(z | x, \phi)$  has a constant, connected support so long as  $0 < q(z | x, \phi) < 1$ .

![](images/35c27c186e0d2efb187306767b37de8921bfc08ec26f681fc6e97dd1a5c3b87a.jpg)  
(a) Approximating posterior  $q(\zeta, z|x)$

![](images/acfa73e657faf36e8809c52afc3fe1fed0e0078e1e6f188f5fdaec3c16f27dc7.jpg)  
(b) Prior  $p(x,\zeta ,z)$

![](images/46f21b1331205622fcd413f464ee45dfe43980879621b007a70caecdc9f5374f.jpg)  
(c) Autoencoding term  
Figure 1: Graphical models of the smoothed approximating posterior (a) and prior (b), and the network realizing the autoencoding term of the ELBO from Equation 2 (c). Continuous latent variables  $\zeta_{i}$  are smoothed analogs of discrete latent variables  $z_{i}$ , and insulate  $z$  from the observed variables  $x$  in the prior (b). This facilitates the marginalization of the discrete  $z$  in the autoencoding term of the ELBO, resulting in a network (c) in which all operations are deterministic and differentiable given independent stochastic input  $\rho \sim U[0,1]$ .

We further require that  $r(\zeta | z)$  is continuous and differentiable except at the endpoints of its support, so the inverse conditional-marginal CDF of  $q(\zeta | x, \phi)$  is differentiable in Equations 3 and 4, as we discuss in Appendix A.

As shown in Figure 1b, we correspondingly augment the prior with  $\zeta$ :

$$
p (\zeta , z | \theta) = r (\zeta | z) \cdot p (z | \theta),
$$

where  $r(\zeta |z)$  is the same as for the approximating posterior. Finally, we require that the conditional distribution over the observed variables only depends on  $\zeta$ :

$$
p (x | \zeta , z, \theta) = p (x | \zeta , \theta). \tag {7}
$$

The smoothing distribution  $r(\zeta |z)$  transforms the model into a continuous function of the distribution over  $z$ , and allows us to use Equations 2 and 3 directly to obtain low-variance stochastic approximations to the gradient.

Given this expansion, we can simplify Equations 3 and 4 by dropping the dependence on  $z$  and applying Equation 16 of Appendix A, which generalizes Equation 3:

$$
\frac {\partial}{\partial \phi} \mathbb {E} _ {q (\zeta , z | x, \phi)} [ \log p (x | \zeta , z, \theta) ] \approx \frac {1}{N} \sum_ {\rho \sim U (0, 1) ^ {n}} \frac {\partial}{\partial \phi} \log p (x | \mathbf {F} _ {q (\zeta | x, \phi)} ^ {- 1} (\rho), \theta). \tag {8}
$$

If the approximating posterior is factorial, then each  $F_{i}$  is an independent CDF, without conditioning or marginalization.

As we shall demonstrate in Section 2.1,  $\mathbf{F}_{q(\zeta |x,\phi)}^{-1}(\rho)$  is a function of  $q(z = 1|x,\phi)$ , where  $q(z = 1|x,\phi)$  is a deterministic probability value calculated by a parameterized function, such as a neural network. The autoencoder implicit in Equation 8 is shown in Figure 1c. Initially, input  $x$  is passed into a deterministic feedforward network  $q(z = 1|x,\phi)$ , for which the final nonlinearity is the logistic function. Its output  $q$ , along with an independent random variable  $\rho \sim U[0,1]$ , is passed into the deterministic function  $\mathbf{F}_{q(\zeta |x,\phi)}^{-1}(\rho)$  to produce a sample of  $\zeta$ . This  $\zeta$ , along with the original input  $x$ , is finally passed to  $\log p(x|\zeta,\theta)$ . The expectation of this log probability with respect to  $\rho$  is the autoencoding term of the VAE formalism, as in Equation 2. Moreover, conditioned on the input and the independent  $\rho$ , this autoencoder is deterministic and differentiable, so backpropagation can be used to produce a low-variance, computationally-efficient approximation to the gradient.

# 2.1 SPIKE-AND-EXPONENTIAL SMOOTHING TRANSFORMATION

As a concrete example consistent with sparse coding, consider the spike-and-exponential transformation from binary  $z$  to continuous  $\zeta$ :

$$
r (\zeta_ {i} | z _ {i} = 0) = \left\{ \begin{array}{l l} \infty , & \text {i f} \zeta_ {i} = 0 \\ 0, & \text {o t h e r w i s e} \end{array} \right. \quad F _ {r (\zeta_ {i} | z _ {i} = 0)} (\zeta^ {\prime}) = 1
$$

$$
r(\zeta_{i}|z_{i} = 1) = \left\{ \begin{array}{ll}\frac{\beta e^{\beta\zeta}}{e^{\beta} - 1}, & \text{if} 0\leq \zeta_{i}\leq 1\\ 0, & \text{otherwise} \end{array} \right.\qquad F_{r(\zeta_{i}|z_{i} = 1)}(\zeta^{\prime}) = \left.\frac{e^{\beta\zeta}}{e^{\beta} - 1}\right|_{0}^{\zeta^{\prime}} = \frac{e^{\beta\zeta^{\prime}} - 1}{e^{\beta} - 1}
$$

where  $F_{p}(\zeta^{\prime}) = \int_{-\infty}^{\zeta^{\prime}} p(\zeta) \cdot d\zeta$  is the CDF of probability distribution  $p$  in the domain [0,1]. This transformation from  $z_{i}$  to  $\zeta_{i}$  is invertible:  $\zeta_{i} = 0 \Leftrightarrow z_{i} = 0$ , and  $\zeta_{i} > 0 \Leftrightarrow z_{i} = 1$  almost surely.

We can now find the CDF for  $q(\zeta | x, \phi)$  as a function of  $q(z = 1|x, \phi)$  in the domain [0, 1], marginalizing out the discrete  $z$ :

$$
\begin{array}{l} F _ {q (\zeta | x, \phi)} \left(\zeta^ {\prime}\right) = \left(1 - q (z = 1 | x, \phi)\right) \cdot F _ {r \left(\zeta_ {i} \mid z _ {i} = 0\right)} \left(\zeta^ {\prime}\right) + q (z = 1 | x, \phi) \cdot F _ {r \left(\zeta_ {i} \mid z _ {i} = 1\right)} \left(\zeta^ {\prime}\right) \\ = q (z = 1 | x, \phi) \cdot \left(\frac {e ^ {\beta \zeta^ {\prime}} - 1}{e ^ {\beta} - 1} - 1\right) + 1. \\ \end{array}
$$

To evaluate the autoencoder of Figure 1c, and through it the gradient approximation of Equation 8, we must invert the conditional-marginal CDF  $F_{q(\zeta |x,\phi)}$ :

$$
F _ {q (\zeta | x, \phi)} ^ {- 1} (\rho) = \left\{ \begin{array}{l l} \frac {1}{\beta} \cdot \log \left[ \left(\frac {\rho + q - 1}{q}\right) \cdot (e ^ {\beta} - 1) + 1 \right], & \text {i f} \rho \geq 1 - q \\ 0, & \text {o t h e r w i s e} \end{array} \right. \tag {9}
$$

where we use the substitution  $q(z = 1|x,\phi) \to q$  to simplify notation. For all values of the independent random variable  $\rho \sim U[0,1]$ , the function  $F_{q(\zeta |x,\phi)}^{-1}(\rho)$  rectifies the input  $q(z = 1|x,\phi)$  if  $q \leq 1 - \rho$  in a manner analogous to a rectified linear unit (ReLU), as shown in Figure 2a. It is also quasi-sigmoidal, in that  $F^{-1}$  is increasing but concave-down if  $q > 1 - \rho$ . The effect of  $\rho$  on  $F^{-1}$  is qualitatively similar to that of dropout (Srivastava et al., 2014), depicted in Figure 2b, or the noise injected by batch normalization (Ioffe & Szegedy, 2015) using small minibatches, shown in Figure 2c.

![](images/26b1df0e108fb0ed28af1060bd9562abff755cb76cb454fc0b09ace875c922a6.jpg)  
(a) Spike-and-exp,  $\beta \in \{1,3,5\}$

![](images/4fd47b2e8e9df784dbaebd336ee55027bfcdcba796c846944ae4733e7683f8e8.jpg)  
(b) ReLU with dropout  
Figure 2: Inverse CDF of the spike-and-exponential smoothing transformation for  $\rho \in \{0.2,0.5,0.8\}$ ;  $\beta = 1$  (dotted),  $\beta = 3$  (solid), and  $\beta = 5$  (dashed) (a). Rectified linear unit with dropout rate 0.5 (b). Shift (red) and scale (green) noise from batch normalization; with magnitude 0.3 (dashed),  $-0.3$  (dotted), or 0 (solid blue); before a rectified linear unit (c). In all cases, the abcissa is the input and the ordinate is the output of the effective transfer function. The novel stochastic nonlinearity  $F_{q(\zeta |x,\phi)}^{-1}(\rho)$  from Figure 1c, of which (a) is an example, is qualitatively similar to the familiar stochastic nonlinearities induced by dropout (b) or batch normalization (c).

![](images/57c75b112967bbcb90fc08c34512fd3b17e02b73d6bb5397a39495eaef0194f7.jpg)  
(c) ReLU with batch norm

Other expansions to the continuous space are possible. In Appendix D.1, we consider the case where both  $r(\zeta_i|z_i = 0)$  and  $r(\zeta_i|z_i = 1)$  are linear functions of  $\zeta$ ; in Appendix D.2, we develop a spike-

and-slab transformation; and in Appendix E, we explore a spike-and-Gaussian transformation where the continuous  $\zeta$  is directly dependent on the input  $x$  in addition to the discrete  $z$ .

# 3 ACCOMMODATING EXPLAINING-AWAY WITH A HIERARCHICAL APPROXIMATING POSTERIOR

When a probabilistic model is defined in terms of a prior distribution  $p(z)$  and a conditional distribution  $p(x|z)$ , the observation of  $x$  often induces strong correlations in the posterior  $p(z|x)$  due to phenomena such as explaining-away (Pearl, 1988). Moreover, we wish to use an RBM as the prior distribution (Equation 6), which itself may have strong correlations. In contrast, to maintain tractability, many variational approximations use a product of independent approximating posterior distributions (e.g., mean-field methods, but also Kingma & Welling (2014); Rezende et al. (2014)).

To accommodate strong correlations in the posterior distribution while maintaining tractability, we introduce a hierarchy into the approximating posterior  $q(z|x)$  over the discrete latent variables. Specifically, we divide the latent variables  $z$  of the RBM into disjoint groups,  $z_{1},\ldots ,z_{k}$ , and define the approximating posterior via a directed acyclic graphical model over these groups:

$$
q \left(z _ {1}, \zeta_ {1}, \dots , z _ {k}, \zeta_ {k} \mid x, \phi\right) = \prod_ {1 \leq j \leq k} r \left(\zeta_ {j} \mid z _ {j}\right) \cdot q \left(z _ {j} \mid \zeta_ {i <   j}, x, \phi\right) \quad \text {w h e r e}
$$

$$
q \left(z _ {j} \mid \zeta_ {i <   j}, x, \phi\right) = \frac {e ^ {g _ {j} \left(\zeta_ {i <   j} , x , \phi\right) ^ {\top} \cdot z _ {j}}}{\prod_ {z _ {\iota} \in z _ {j}} \left(1 + e ^ {g _ {z _ {\iota}} \left(\zeta_ {i <   j} , x , \phi\right)}\right)}, \tag {10}
$$

$z_{j} \in \{0,1\}^{n}$ , and  $g_{j}(\zeta_{i < j}, x, \phi)$  is a parameterized function of the inputs and preceding  $\zeta_{i}$ , such as a neural network. The corresponding graphical model is depicted in Figure 3a. If each group  $z_{j}$  contains a single variable, this dependence structure is analogous to that of a deep autoregressive network (DARN) (Gregor et al., 2014), and can represent any distribution. However, the dependence of  $z_{j}$  on the preceding discrete variables  $z_{i < j}$  is always mediated by the continuous variables  $\zeta_{i < j}$ .

![](images/c5dcbdeeb3812a4e83d5e3798dc5ff9906100b02489cdc0b6583e0bf351a8b77.jpg)

![](images/0efff77ab8825418fc48c3adfcd9016714a59cc072e401575a7f384545dffab3.jpg)  
(a) Hierarch approx post  $q(\zeta ,z|x)$  
(b) Hierarchical ELBO autoencoding term  
Figure 3: Graphical model of the hierarchical approximating posterior (a) and the network realizing the autoencoding term of the ELBO (b) from Equation 2. Discrete latent variables  $z_{j}$  only depend on the previous  $z_{i < j}$  through their smoothed analogs  $\zeta_{i < j}$ . The autoregressive hierarchy allows the approximating posterior to capture correlations and multiple modes. Again, all operations in (b) are deterministic and differentiable given the stochastic input  $\rho$ .

This hierarchical approximating posterior does not affect the form of the autoencoding term in Equation 8, except to increase the depth of the autoencoder, as shown in Figure 3b. The deterministic probability value  $q(z_{j} = 1|\zeta_{i < j},x,\phi)$  of Equation 10 is parameterized, generally by a neural network, in a manner analogous to Section 2. However, the final logistic function is made explicit in

Equation 10 to simplify Equation 12. For each successive layer  $j$  of the autoencoder, input  $x$  and all previous  $\zeta_{i < j}$  are passed into the network computing  $q(z = 1|\zeta_{i < j},x,\phi)$ . Its output  $q_{j}$ , along with an independent random variable  $\rho \sim U[0,1]$ , is passed to the deterministic function  $\mathbf{F}_{q(\zeta_j|\zeta_{i < j},x,\phi)}^{-1}(\rho)$  to produce a sample of  $\zeta_{j}$ . Once all  $\zeta_{j}$  have been recursively computed, the full  $\zeta$  along with the original input  $x$  is finally passed to  $\log p(x|\zeta ,\theta)$ . The expectation of this log probability with respect to  $\rho$  is again the autoencoding term of the VAE formalism, as in Equation 2.

In Appendix F, we show that the gradients of the remaining KL term of the ELBO (Equation 2) can be estimated stochastically using:

$$
\frac {\partial}{\partial \theta} \mathrm {K L} [ q | | p ] = \mathbb {E} _ {q \left(z _ {1} \mid x, \phi\right)} \left[ \dots \left[ \mathbb {E} _ {q \left(z _ {k} \mid \zeta_ {i <   k}, x, \phi\right)} \left[ \frac {\partial E _ {p} (z , \theta)}{\partial \theta} \right] \right] \right] - \mathbb {E} _ {p (z | \theta)} \left[ \frac {\partial E _ {p} (z , \theta)}{\partial \theta} \right] \quad \text {a n d} \tag {11}
$$

$$
\frac {\partial}{\partial \phi} \mathrm {K L} [ q | | p ] = \mathbb {E} _ {\rho} \left[ (g (x, \zeta) - b) ^ {\top} \cdot \frac {\partial q}{\partial \phi} - z ^ {\top} \cdot W \cdot \left(\frac {1 - z}{1 - q} \odot \frac {\partial q}{\partial \phi}\right) \right]. \tag {12}
$$

In particular, Equation 12 is substantially lower variance than the naive approach to calculate  $\frac{\partial}{\partial\phi}\mathrm{KL}[q||p]$ , based upon REINFORCE.

# 4 MODELLING CONTINUOUS DEFORMATIONS WITH A HIERARCHY OF CONTINUOUS LATENT VARIABLES

We can make both the generative model and the approximating posterior more powerful by adding additional layers of latent variables below the RBM. While these layers can be discrete, we focus on continuous variables, which have proven to be powerful in generative adversarial networks (Goodfellow et al., 2014) and traditional variational autoencoders (Kingma & Welling, 2014; Rezende et al., 2014). When positioned below and conditioned on a layer of discrete variables, continuous variables can build continuous manifolds, from which the discrete variables can choose. This complements the structure of the natural world, where a percept is determined first by a discrete selection of the types of objects present in the scene, and then by the position, pose, and other continuous attributes of these objects.

Specifically, we augment the latent representation with continuous random variables  $\mathfrak{z},^5$  and define both the approximating posterior and the prior to be fully autoregressive directed graphical models. We use the same autoregressive variable order for the approximating posterior as for the prior, as in DRAW (Gregor et al., 2015), variational recurrent neural networks (Chung et al., 2015), the deep VAE of Salimans (2016), and ladder networks (Rasmus et al., 2015; Sønderby et al., 2016). We discuss the motivation for this ordering in Appendix G.

The directed graphical model of the approximating posterior and prior are defined by:

$$
q \left(\mathfrak {z} _ {0}, \dots , \mathfrak {z} _ {n} | x, \phi\right) = \prod_ {0 \leq m \leq n} q \left(\mathfrak {z} _ {m} | \mathfrak {z} _ {l <   m}, x, \phi\right) \quad \text {a n d}
$$

$$
p \left(\mathfrak {z} _ {0}, \dots , \mathfrak {z} _ {n} | \theta\right) = \prod_ {0 \leq m \leq n} p \left(\mathfrak {z} _ {m} | \mathfrak {z} _ {l <   m}, \theta\right). \tag {13}
$$

The full set of latent variables associated with the RBM is now denoted by  $\mathfrak{z}_0 = \{z_1, \zeta_1, \dots, z_k, \zeta_k\}$ . However, the conditional distributions in Equation 13 only depend on the continuous  $\zeta_j$ . Each  $\mathfrak{z}_{m \geq 1}$  denotes a layer of continuous latent variables, and Figure 4 shows the resulting graphical model.

The ELBO decomposes as:

$$
\mathcal {L} (x, \theta , \phi) = \mathbb {E} _ {q (\mathfrak {z} | x, \phi)} [ \log p (x | \mathfrak {z}, \theta) ] - \sum_ {m} \mathbb {E} _ {q (\mathfrak {z} l <   m | x, \phi)} [ \mathrm {K L} [ q (\mathfrak {z} m | \mathfrak {z} l <   m, x, \phi) | | p (\mathfrak {z} m | \mathfrak {z} l <   m, \theta) ] ]. \tag {14}
$$

If both  $q(\mathfrak{z}_m|\mathfrak{z}_{l < m},x,\phi)$  and  $p(\mathfrak{z}_m|\mathfrak{z}_{l < m},\theta)$  are Gaussian, then their KL divergence has a simple closed form, which is computationally efficient if the covariance matrices are diagonal. Gradients can be passed through the  $q(\mathfrak{z}_{l < m}|x,\phi)$  using the traditional reparameterization trick, described in Section 1.1.

![](images/8fd8cbc4b5f71bfa293e2b779d6ce0b2f3ec84fa3605947b7e4f70b1a667e3b5.jpg)  
(a) Approx post w/ cont latent vars  $q(\mathfrak{z},\zeta ,z|x)$

![](images/3160d2edee910d09b90506bf41f9f481b761b6a63da6f62abbf61659a5883f06.jpg)  
(b) Prior w/ cont latent vars  $p(x,\mathfrak{z},\zeta ,z)$  
Figure 4: Graphical models of the approximating posterior (a) and prior (b) with a hierarchy of continuous latent variables. The shaded regions in parts (a) and (b) expand to Figures 3a and 1b respectively. The continuous latent variables  $\mathfrak{z}$  build continuous manifolds, capturing properties like position and pose, conditioned on the discrete latent variables  $z$ , which can represent the discrete types of objects in the image.

# 5 RESULTS

Discrete variational autoencoders comprise a smoothed RBM (Section 2) with a hierarchical approximating posterior (Section 3), followed by a hierarchy of continuous latent variables (Section 4). We parameterize all distributions with neural networks, except the smoothing distribution  $r(\zeta | z)$  discussed in Section 2. Like NVIL (Mnih & Gregor, 2014) and VAEs (Kingma & Welling, 2014; Rezende et al., 2014), we define all approximating posteriors  $q$  to be explicit functions of  $x$ , with parameters  $\phi$  shared between all inputs  $x$ . For distributions over discrete variables, the neural networks output the parameters of a factorial Bernoulli distribution using a logistic final layer, as in Equation 10; for the continuous  $\mathfrak{z}$ , the neural networks output the mean and log-standard deviation of a diagonal-covariance Gaussian distribution using a linear final layer. Each layer of the neural networks parameterizing the distributions over  $z$ ,  $\mathfrak{z}$ , and  $x$  consists of a linear transformation, batch normalization (Ioffe & Szegedy, 2015) (but see Appendix H.2), and a rectified-linear pointwise nonlinearity (ReLU). We stochastically approximate the expectation with respect to the RBM prior  $p(z|\theta)$  in Equation 11 using block Gibbs sampling on persistent Markov chains, analogous to persistent contrastive divergence (Tieleman, 2008). We minimize the ELBO using ADAM (Kingma & Ba, 2015) with a decaying step size.

The hierarchical structure of Section 4 is very powerful, and overfits without strong regularization of the prior, as shown in Appendix H. In contrast, powerful approximating posteriors do not induce significant overfitting. To address this problem, we use conditional distributions over the input  $p(x|\zeta, \theta)$  without any deterministic hidden layers, except on Omniglot. Moreover, all other neural networks in the prior have only one hidden layer, the size of which is carefully controlled. On statically binarized MNIST, Omniglot, and Caltech-101, we share parameters between the layers of the hierarchy over  $\mathfrak{z}$ . We present the details of the architecture in Appendix H.

We train the resulting discrete VAEs on the permutation-invariant MNIST (LeCun et al., 1998), Omniglot<sup>6</sup> (Lake et al., 2013), and Caltech-101 Silhouettes datasets (Marlin et al., 2010). For MNIST, we use both the static binarization of Salakhutdinov & Murray (2008) and dynamic binarization. Estimates of the log-likelihood<sup>7</sup> of these models, computed using the method of (Burda et al., 2015a) with  $10^{4}$  importance-weighted samples, are listed in Table 1. The reported log-likelihoods for discrete VAEs are the average of 16 runs; the standard deviation of these log-likelihoods are 0.08, 0.04,

0.05, and 0.11 for dynamically and statically binarized MNIST, Omniglot, and Caltech-101 Silhouettes, respectively. Removing the RBM reduces the test set log-likelihood by 0.09, 0.37, 0.69, and 0.66.

<table><tr><td colspan="2">MNIST (dynamic binarization)</td><td colspan="3">MNIST (static binarization)</td></tr><tr><td></td><td>LL</td><td></td><td>ELBO</td><td>LL</td></tr><tr><td>DBN</td><td>-84.55</td><td>HVI</td><td>-88.30</td><td>-85.51</td></tr><tr><td>IWAE</td><td>-82.90</td><td>DRAW</td><td>-87.40</td><td></td></tr><tr><td>Ladder VAE</td><td>-81.74</td><td>NAIS NADE</td><td></td><td>-83.67</td></tr><tr><td>Discrete VAE</td><td>-80.15</td><td>Normalizing flows</td><td>-85.10</td><td></td></tr><tr><td></td><td></td><td>Variational Gaussian process</td><td></td><td>-81.32</td></tr><tr><td></td><td></td><td>Discrete VAE</td><td>-84.58</td><td>-81.01</td></tr><tr><td colspan="2">Omniglot</td><td colspan="3">Caltech-101 Silhouettes</td></tr><tr><td></td><td>LL</td><td></td><td></td><td>LL</td></tr><tr><td>IWAE</td><td>-103.38</td><td>IWAE</td><td></td><td>-117.2</td></tr><tr><td>Ladder VAE</td><td>-102.11</td><td>RWS SBN</td><td></td><td>-113.3</td></tr><tr><td>RBM</td><td>-100.46</td><td>RBM</td><td></td><td>-107.8</td></tr><tr><td>DBN</td><td>-100.45</td><td>NAIS NADE</td><td></td><td>-100.0</td></tr><tr><td>Discrete VAE</td><td>-97.43</td><td>Discrete VAE</td><td></td><td>-97.6</td></tr></table>

Table 1: Test set log-likelihood of various models on the permutation-invariant MNIST, Omniglot, and Caltech-101 Silhouettes datasets. For the discrete VAE, the reported log-likelihood is estimated with  $10^{4}$  importance-weighted samples (Burda et al., 2015a). For comparison, we also report performance of some recent state-of-the-art techniques. Full names and references are listed in Appendix I.

We further analyze the performance of discrete VAEs on dynamically binarized MNIST: the largest of the datasets, requiring the least regularization. Figure 5 shows the generative output of a discrete VAE as the Markov chain over the RBM evolves via block Gibbs sampling. The RBM is held constant across each sub-row of five samples, and variation amongst these samples is due to the layers of continuous latent variables. Given a multimodal distribution with well-separated modes, Gibbs sampling passes through the large, low-probability space between the modes only infrequently. As a result, consistency of the digit class over many successive rows in Figure 5 indicates that the RBM prior has well-separated modes. The RBM learns distinct, separated modes corresponding to the different digit types, except for  $3/5$  and  $4/9$ , which are either nearby or overlapping; at least tens of thousands of iterations of single-temperature block Gibbs sampling is required to mix between the modes. We present corresponding figures for the other datasets in Appendix J.

The large mixing time of block Gibbs sampling on the RBM suggests that training may be constrained by sample quality. Figure 6a shows that performance<sup>8</sup> improves as we increase the number of iterations of block Gibbs sampling performed per minibatch on the RBM prior:  $p(z|\theta)$  in Equation 11. This suggests that a further improvement may be achieved by using a more effective sampling algorithm, such as parallel tempering (Swendsen & Wang, 1986).

Commensurate with the small number of intrinsic classes, a moderately sized RBM yields the best performance on MNIST. As shown in Figure 6b, the log-likelihood plateaus once the number of units in the RBM reaches at least 64. Presumably, we would need a much larger RBM to model a dataset likeImagenet, which has many classes and complicated relationships between the elements of various classes.

The benefit of the hierarchical approximating posterior over the RBM, introduced in Section 3, is apparent from Figure 6c. The reduction in performance when moving from 4 to 8 layers in the approximating posterior may be due to the fact that each additional hierarchical layer over the approximating posterior adds three layers to the encoder neural network: there are two deterministic hidden layers for each stochastic latent layer. As a result, expanding the number of RBM approximating posterior layers significantly increases the number of parameters that must be trained, and increases the risk of overfitting.

![](images/5ee1ecf4fc86606f724bb3071f4ae68027852e47dd94e1a290fe6b7c36e84b48.jpg)  
Figure 5: Evolution of samples from a discrete VAE trained on dynamically binarized MNIST, using persistent RBM Markov chains. We perform 100 iterations of block-Gibbs sampling on the RBM between successive rows. Each horizontal group of 5 uses a single, shared sample from the RBM, but independent continuous latent variables, and shows the variation induced by the continuous layers as opposed to the RBM. The long vertical sequences in which the digit ID remains constant demonstrate that the RBM has well-separated modes, each of which corresponds to a single (or occasionally two) digit IDs, despite being trained in a wholly unsupervised manner.

![](images/917009acaca29d80a25b64fa280ab749eac4921361f868f13fd2a53dee3f901f.jpg)  
(a) Block Gibbs iterations

![](images/996de247547e9f98ca69545addcb472d4e840439acf207351f9a59de7d94c7c0.jpg)  
(b) Num RBM units  
Figure 6: Log likelihood versus the number of iterations of block Gibbs sampling per minibatch (a), the number of units in the RBM (b), and the number of layers in the approximating posterior over the RBM (c). Better sampling (a) and hierarchical approximating posteriors (c) support better performance, but the network is robust to the size of the RBM (b).

![](images/f551c6929a728aaba87e23ea45083091697d1d44eb737ff6da3d7cf106010a6d.jpg)  
(c) RBM approx post layers

# 6 CONCLUSION

Datasets consisting of a discrete set of classes are naturally modeled using discrete latent variables. However, it is difficult to train probabilistic models over discrete latent variables using efficient gradient approximations based upon backpropagation, such as variational autoencoders, since it is generally not possible to backpropagate through a discrete variable (Bengio et al., 2013).

We avoid this problem by symmetrically projecting the approximating posterior and the prior into a continuous space. We then evaluate the autoencoding term of the evidence lower bound exclusively in the continuous space, marginalizing out the original discrete latent representation. At the same time, we evaluate the KL divergence between the approximating posterior and the true prior in the original discrete space; due to the symmetry of the projection into the continuous space, it does not

contribute to the KL term. To increase representational power, we make the approximating posterior over the discrete latent variables hierarchical, and add a hierarchy of continuous latent variables below them. The resulting discrete variational autoencoder achieves state-of-the-art performance on the permutation-invariant MNIST, Omniglot, and Caltech-101 Silhouettes datasets.

# ACKNOWLEDGEMENTS

Zhengbing Bian, Fabian Chudak, Arash Vahdat helped run experiments. Jack Raymond provided the library used to estimate the log partition function of RBMs. Mani Ranjbar wrote the cluster management system, and a custom GPU acceleration library used for an earlier version of the code. We thank Evgeny Andriyash, William Macready, and Aaron Courville for helpful discussions.

# REFERENCES

Jimmy Ba and Brendan Frey. Adaptive dropout for training deep neural networks. In Advances in Neural Information Processing Systems, pp. 3084-3092, 2013.  
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Charles H. Bennett. Efficient estimation of free energy differences from Monte Carlo data. Journal of Computational Physics, 22(2):245-268, 1976.  
Jörg Bornschein and Yoshua Bengio. Reweighted wake-sleep. In *ICLR* 2015, arXiv:1406.2751, 2015.  
Jorg Bornschein, Samira Shabanian, Asja Fischer, and Yoshua Bengio. Bidirectional Helmholtz machines. arXiv preprint arXiv:1506.03877, 2015.  
Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. arXiv preprint arXiv:1511.06349, 2015.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015a.  
Yuri Burda, Roger B. Grosse, and Ruslan Salakhutdinov. Accurate and conservative estimates of MRF log-likelihood using reverse annealing. In AISTATS, 2015b.  
Steve Cheng. Differentiation under the integral sign with weak derivatives. Technical report, Working paper, 2006.  
KyungHyun Cho, Tapani Raiko, and Alexander Ilin. Enhanced gradient for training restricted Boltzmann machines. Neural Computation, 25(3):805-831, 2013.  
Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron C. Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. In Advances in Neural Information Processing Systems, pp. 2980-2988, 2015.  
Aaron C. Courville, James S. Bergstra, and Yoshua Bengio. Unsupervised models of images by spike-and-slab rbms. In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pp. 1145-1152, 2011.  
Paul Dagum and Michael Luby. Approximating probabilistic inference in Bayesian belief networks is NP-hard. Artificial Intelligence, 60(1):141-153, 1993.  
Chao Du, Jun Zhu, and Bo Zhang. Learning deep generative models with doubly stochastic MCMC. arXiv preprint arXiv:1506.04557, 2015.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Alex Graves. Stochastic backpropagation through mixture density distributions. arXiv preprint arXiv:1607.05690, 2016.

Karol Gregor, Ivo Danihelka, Andriy Mnih, Charles Blundell, and Daan Wierstra. Deep autoregressive networks. In Proceedings of the 31st International Conference on Machine Learning, pp. 1242-1250, 2014.  
Karol Gregor, Ivo Danihelka, Alex Graves, and Daan Wierstra. DRAW: A recurrent neural network for image generation. arXiv preprint arXiv:1502.04623, 2015.  
Geoffrey Hinton, Simon Osindero, and Yee-Whye Teh. A fast learning algorithm for deep belief nets. Neural Computation, 18(7):1527-1554, 2006.  
Geoffrey E. Hinton and R. S. Zemel. Autoencoders, minimum description length, and Helmholtz free energy. In J. D. Cowan, G. Tesauro, and J. Alspector (eds.), Advances in Neural Information Processing Systems 6 (NIPS'93), pp. 3-10. Morgan Kaufmann Publishers, Inc., 1994.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, pp. 448-456, 2015.  
Matthew James Johnson, David Duvenaud, Alexander B. Wiltschko, Sandeep R. Datta, and Ryan P. Adams. Composing graphical models with neural networks for structured representations and fast inference. arXiv preprint arXiv:1603.06277, 2016.  
Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. An introduction to variational methods for graphical models. Machine learning, 37(2):183-233, 1999.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR 2015, arXiv preprint arXiv:1412.6980, 2015.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, pp. 3581-3589, 2014.  
Durk P. Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR), 2014.  
Brenden M. Lake, Ruslan R. Salakhutdinov, and Josh Tenenbaum. One-shot learning by inverting a compositional causal process. In Advances in Neural Information Processing Systems, pp. 2526-2534, 2013.  
Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Yingzhen Li and Richard E. Turner. Variational inference with Rényi divergence. arXiv preprint arXiv:1602.02311, 2016.  
Philip M. Long and Rocco Servedio. Restricted Boltzmann machines are hard to approximately evaluate or simulate. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pp. 703-710, 2010.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian Goodfellow. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.  
Benjamin M Marlin, Kevin Swersky, Bo Chen, and Nando de Freitas. Inductive principles for restricted Boltzmann machine learning. In AISTATS, pp. 509-516, 2010.  
Andriy Mnih and Karol Gregor. Neural variational inference and learning in belief networks. Proceedings of the 31st International Conference on Machine Learning, pp. 1791-1799, 2014.  
Andriy Mnih and Danilo J. Rezende. Variational inference for Monte Carlo objectives. arXiv preprint arXiv:1602.06725, 2016.

Iain Murray and Ruslan R. Salakhutdinov. Evaluating probabilities under high-dimensional latent variable models. In Advances in Neural Information Processing Systems, pp. 1137-1144, 2009.  
Bruno A. Olshausen and David J. Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583):607-609, 1996.  
John Paisley, David M. Blei, and Michael I. Jordan. Variational Baysian inference with stochastic search. In Proceedings of the 29th International Conference on Machine Learning, 2012.  
Judea Pearl. *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann, 1988.  
Tapani Raiko, Harri Valpola, Markus Harva, and Juha Karhunen. Building blocks for variational Bayesian learning of latent variable models. Journal of Machine Learning Research, 8(Jan): 155-201, 2007.  
Tapani Raiko, Mathias Berglund, Guillaume Alain, and Laurent Dinh. Techniques for learning binary stochastic feedforward neural networks. In ICLR 2015, arXiv:1406.2989, 2015.  
Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, and Tapani Raiko. Semi-supervised learning with ladder networks. In Advances in Neural Information Processing Systems, pp. 3546-3554, 2015.  
Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of the 32nd International Conference on Machine Learning, pp. 1530-1538, 2015.  
Danilo J. Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of The 31st International Conference on Machine Learning, pp. 1278-1286, 2014.  
Ruslan Salakhutdinov and Geoffrey E. Hinton. Deep Boltzmann machines. In International Conference on Artificial Intelligence and Statistics, pp. 448-455, 2009.  
Ruslan Salakhutdinov and Iain Murray. On the quantitative analysis of deep belief networks. In Proceedings of the 25th International Conference on Machine Learning, pp. 872-879. ACM, 2008.  
Tim Salimans. A structured variational auto-encoder for learning deep hierarchies of sparse features. arXiv preprint arXiv:1602.08734, 2016.  
Tim Salimans, Diederik P. Kingma, Max Welling, et al. Markov chain Monte Carlo and variational inference: Bridging the gap. In International Conference on Machine Learning, pp. 1218-1226, 2015.  
Michael R. Shirts and John D. Chodera. Statistically optimal analysis of samples from multiple equilibrium states. The Journal of Chemical Physics, 129(12), 2008.  
Paul Smolensky. Information processing in dynamical systems: Foundations of harmony theory. In D. E. Rumelhart and J. L. McClelland (eds.), Parallel Distributed Processing, volume 1, chapter 6, pp. 194-281. MIT Press, Cambridge, 1986.  
Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. Ladder variational autoencoders. arXiv preprint arXiv:1602.02282, 2016.  
David J. Spiegelhalter and Steffen L. Lauritzen. Sequential updating of conditional probabilities on directed graphical structures. Networks, 20(5):579-605, 1990.  
Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Robert H. Swendsen and Jian-Sheng Wang. Replica Monte Carlo simulation of spin-glasses. *Physical Review Letters*, 57(21):2607, 1986.

Tijmen Tieleman. Training restricted Boltzmann machines using approximations to the likelihood gradient. In Proceedings of the 25th International Conference on Machine Learning, pp. 1064-1071. ACM, 2008.  
Dustin Tran, Rajesh Ranganath, and David M. Blei. Variational Gaussian process. arXiv preprint arXiv:1511.06499, 2015.  
Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.
