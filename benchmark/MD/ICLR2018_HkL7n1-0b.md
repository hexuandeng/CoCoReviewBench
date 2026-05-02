# WASSERSTEIN AUTO-ENCODERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose the Wasserstein Auto-Encoder (WAE)—a new algorithm for building a generative model of the data distribution. WAE minimizes a penalized form of the Wasserstein distance between the model distribution and the target distribution, which leads to a different regularizer than the one used by the Variational Auto-Encoder (VAE) (Kingma & Welling, 2014). This regularizer encourages the encoded training distribution to match the prior. We compare our algorithm with several other techniques and show that it is a generalization of adversarial auto-encoders (AAE) (Makhzani et al., 2016). Our experiments show that WAE shares many of the properties of VAEs (stable training, encoder-decoder architecture, nice latent manifold structure) while generating samples of better quality, as measured by the FID score.

# 1 INTRODUCTION

The field of representation learning was initially driven by supervised approaches, with impressive results using large labelled datasets. Unsupervised generative modeling, in contrast, used to be a domain governed by probabilistic approaches focusing on low-dimensional data. Recent years have seen a convergence of those two approaches. In the new field that formed at the intersection, variational auto-encoders (VAEs) (Kingma & Welling, 2014) constitute one well-established approach, theoretically elegant yet with the drawback that they tend to generate blurry samples when applied to natural images. In contrast, generative adversarial networks (GANs) (Goodfellow et al., 2014) turned out to be more impressive in terms of the visual quality of images sampled from the model, but come without an encoder, have been reported harder to train, and suffer from the "mode collapse" problem where the resulting model is unable to capture all the variability in the true data distribution. There has been a flurry of activity in assaying numerous configurations of GANs as well as combinations of VAEs and GANs. A unifying framework combining the best of GANs and VAEs in a principled way is yet to be discovered.

Following Arjovsky et al. (2017), we approach generative modeling from the optimal transport (OT) point of view. The OT cost (Villani, 2003) is a way to measure a distance between probability distributions and provides a much weaker topology than many others, including  $f$ -divergences associated with the original GAN algorithms (Nowozin et al., 2016). This is particularly important in applications, where data is usually supported on low dimensional manifolds in the input space  $\mathcal{X}$ . As a result, stronger notions of distances (such as  $f$ -divergences, which capture the density ratio between distributions) often max out, providing no useful gradients for training. In contrast, OT was claimed to have a nicer behaviour (Arjovsky et al., 2017; Gulrajani et al., 2017) although it requires, in its GAN-like implementation, the addition of a constraint or a regularization term into the objective.

In this work we aim at minimizing OT  $W_{c}(P_{X}, P_{G})$  between the true (but unknown) data distribution  $P_{X}$  and a latent variable model  $P_{G}$  specified by the prior distribution  $P_{Z}$  of latent codes  $Z \in \mathcal{Z}$  and the generative model  $P_{G}(X|Z)$  of the data points  $X \in \mathcal{X}$  given  $Z$ . Our main contributions are listed below (cf. also Figure 1):

- A new family of regularized auto-encoders (Algorithms 1, 2 and Eq.4), which we call Wasserstein Auto-Encoders (WAE), that minimize the optimal transport  $W_{c}(P_{X},P_{G})$  for any cost function  $c$ . Similarly to VAE, the objective of WAE is composed of two terms: the  $c$ -reconstruction cost and a regularizer  $\mathcal{D}_Z(P_Z,Q_Z)$  penalizing a discrepancy between two distributions in  $\mathcal{Z}$ :  $P_Z$  and a distribution of encoded data points, i.e.  $Q_{Z}\coloneqq \mathbb{E}_{P_{X}}[Q(Z|X)]$ .

![](images/0b55ca86694e2e82497ec090abb8c5aa4d86a7c1f1bb118088bbd92dcfbe2061.jpg)  
(a) VAE

![](images/a42423c93498e80821e97af46f056dff2321d096a30b2d575ca78ab584bda723.jpg)  
(b) WAE  
Figure 1: Both VAE and WAE minimize two terms: the reconstruction cost and the regularizer penalizing discrepancy between  $P_Z$  and distribution induced by the encoder  $Q$ . VAE forces  $Q(Z|X = x)$  to match  $P_Z$  for all the different input examples  $x$  drawn from  $P_X$ . This is illustrated on picture (a), where every single red ball is forced to match  $P_Z$  depicted as the white shape. Red balls start intersecting, which leads to problems with reconstruction. In contrast, WAE forces the continuous mixture  $Q_Z \coloneqq \int Q(Z|X)dP_X$  to match  $P_Z$ , as depicted with the green ball in picture (b). As a result, latent codes of different examples get a chance to stay far away from each other, promoting a better reconstruction.

When  $c$  is the squared cost and  $\mathcal{D}_Z$  is the GAN objective, WAE coincides with adversarial auto-encoders of Makhzani et al. (2016).

- Empirical evaluation of WAE on MNIST and CelebA datasets with squared cost  $c(x, y) = \| x - y\|_2^2$ . Our experiments show that WAE keeps the good properties of VAEs (stable training, encoder-decoder architecture, and a nice latent manifold structure) while generating samples of better quality, approaching those of GANs.  
- We propose and examine two different regularizers  $\mathcal{D}_Z(P_Z, Q_Z)$ . One is based on GANs and adversarial training in the latent space  $\mathcal{Z}$ . The other uses the maximum mean discrepancy, which is known to perform well when matching high-dimensional standard normal distributions  $P_Z$  (Gretton et al., 2012). Importantly, the second option leads to a fully adversary-free min-min optimization problem.  
- Finally, the theoretical considerations used to derive the WAE objective might be interesting in their own right. We prove in particular (Theorem 1) that in the case of generative models, the primal form of  $W_{c}(P_{X}, P_{G})$  is equivalent to a problem involving the optimization of a probabilistic encoder  $Q(Z|X)$ .

The paper is structured as follows. In Section 2 we derive a novel auto-encoder formulation for OT between  $P_{X}$  and the latent variable model  $P_{G}$ . Relaxing the resulting constrained optimization problem we arrive at an objective of Wasserstein auto-encoders. We propose two different regularizers, leading to WAE-GAN and WAE-MMD algorithms. Section 3 discusses the related work. We present the experimental results in Section 4 and conclude by pointing out some promising directions for future work.

# 2 PROPOSED METHOD

Our new method minimizes the optimal transport cost  $W_{c}(P_{X}, P_{G})$  based on the novel auto-encoder formulation derived in Theorem 1. In the resulting optimization problem the decoder tries to accurately reconstruct the encoded training examples as measured by the cost function  $c$ . The encoder tries to simultaneously achieve two conflicting goals: it tries to match the encoded distribution of training examples  $Q_{Z} \coloneqq \mathbb{E}_{P_{X}}[Q(Z|X)]$  to the prior  $P_{Z}$  as measured by any specified divergence  $\mathcal{D}_Z(Q_Z, P_Z)$ , while making sure that the latent codes provided to the decoder are informative enough to reconstruct the encoded training examples. This is schematically depicted on Fig. 1.

# 2.1 PRELIMINARIES AND NOTATIONS

We use calligraphic letters (i.e.  $\mathcal{X}$ ) for sets, capital letters (i.e.  $X$ ) for random variables, and lower case letters (i.e.  $x$ ) for their values. We denote probability distributions with capital letters (i.e.  $P(X)$ ) and corresponding densities with lower case letters (i.e.  $p(x)$ ). In this work we will consider several measures of discrepancy between probability distributions  $P_{X}$  and  $P_{G}$ . The class of  $f$ -divergences (Liese & Miescke, 2008) is defined by  $D_{f}(P_{X} \| P_{G}) := \int f\left(\frac{p_{X}(x)}{p_{G}(x)}\right) p_{G}(x) dx$ , where  $f: (0, \infty) \to \mathcal{R}$  is any convex function satisfying  $f(1) = 0$ . Classical examples include the Kullback-Leibler  $D_{\mathrm{KL}}$  and Jensen-Shannon  $D_{\mathrm{JS}}$  divergences.

# 2.2 OPTIMAL TRANSPORT AND ITS DUAL FORMULATIONS

A rich class of divergences between probability distributions is induced by the optimal transport (OT) problem (Villani, 2003). Kantorovich's formulation of the problem is given by

$$
W _ {c} \left(P _ {X}, P _ {G}\right) := \inf  _ {\Gamma \in \mathcal {P} \left(X \sim P _ {X}, Y \sim P _ {G}\right)} \mathbb {E} _ {(X, Y) \sim \Gamma} [ c (X, Y) ], \tag {1}
$$

where  $c(x,y)\colon \mathcal{X}\times \mathcal{X}\to \mathcal{R}_{+}$  is any measurable cost function and  $\mathcal{P}(X\sim P_X,Y\sim P_G)$  is a set of all joint distributions of  $(X,Y)$  with marginals  $P_{X}$  and  $P_{G}$  respectively. A particularly interesting case is when  $(\mathcal{X},d)$  is a metric space and  $c(x,y) = d^p (x,y)$  for  $p\geq 1$  . In this case  $W_{p}$  , the  $p$  -th root of  $W_{c}$  , is called the  $p$  -Wasserstein distance.

When  $c(x,y) = d(x,y)$  the following Kantorovich-Rubinstein duality holds:

$$
W _ {1} \left(P _ {X}, P _ {G}\right) = \sup  _ {f \in \mathcal {F} _ {L}} \mathbb {E} _ {X \sim P _ {X}} [ f (X) ] - \mathbb {E} _ {Y \sim P _ {G}} [ f (Y) ], \tag {2}
$$

where  $\mathcal{F}_L$  is the class of all bounded 1-Lipschitz functions on  $(\mathcal{X},d)$ .

# 2.3 APPLICATION TO GENERATIVE MODELS: WASSERSTEIN AUTO-ENCODERS

One way to look at modern generative models like VAEs and GANs is to postulate that they are trying to minimize certain discrepancy measures between the data distribution  $P_{X}$  and the model  $P_{G}$ . Unfortunately, most of the standard divergences known in the literature, including those listed above, are hard or even impossible to compute, especially when  $P_{X}$  is unknown and  $P_{G}$  is parametrized by deep neural networks. Previous research provides several tricks to address this issue.

In case of minimizing the KL-divergence  $D_{\mathrm{KL}}(P_X, P_G)$ , or equivalently maximizing the marginal log-likelihood  $E_{P_X}[\log p_G(X)]$ , the famous variational lower bound provides a theoretically grounded framework successfully employed by VAEs (Kingma & Welling, 2014; Mescheder et al., 2017). More generally, if the goal is to minimize the  $f$ -divergence  $D_f(P_X, P_G)$  (with one example being  $D_{\mathrm{KL}}$ ), one can resort to its dual formulation and make use of  $f$ -GANs and the adversarial training (Nowozin et al., 2016). Finally, OT cost  $W_c(P_X, P_G)$  is yet another option, which can be, thanks to the celebrated Kantorovich-Rubinstein duality (2), expressed as an adversarial objective as implemented by the Wasserstein-GAN (Arjovsky et al., 2017). We include an extended review of all these methods in Supplementary A.

In this work we will focus on latent variable models  $P_{G}$  defined by a two-step procedure, where first a code  $Z$  is sampled from a fixed distribution  $P_{Z}$  on a latent space  $\mathcal{Z}$  and then  $Z$  is mapped to the image  $X \in \mathcal{X} = \mathcal{R}^{d}$  with a (possibly random) transformation. This results in a density of the form

$$
p _ {G} (x) := \int_ {\mathcal {Z}} p _ {G} (x | z) p _ {z} (z) d z, \quad \forall x \in \mathcal {X}, \tag {3}
$$

assuming all involved densities are properly defined. For simplicity we will focus on non-random decoders, i.e. generative models  $P_{G}(X|Z)$  deterministically mapping  $Z$  to  $X = G(Z)$  for a given map  $G\colon \mathcal{Z}\to \mathcal{X}$ . In Supplementary B we present similar results for random decoders.

It turns out that under this model, the OT cost takes a simpler form as the transportation plan factors through the map  $G$ : instead of finding a coupling  $\Gamma$  in (1) between two random variables living in

the  $\mathcal{X}$  space, one distributed according to  $P_{X}$  and the other one according to  $P_{G}$ , it is sufficient to find a conditional distribution  $Q(Z|X)$  such that its  $Z$  marginal  $Q_{Z}(Z) \coloneqq \mathbb{E}_{X \sim P_{X}}[Q(Z|X)]$  is identical to the prior distribution  $P_{Z}$ . This is the content of our main theorem below.

Theorem 1 For any function  $G\colon \mathcal{Z}\to \mathcal{X}$  we have

$$
\inf _ {\Gamma \in \mathcal {P} (X \sim P _ {X}, Y \sim P _ {G})} \mathbb {E} _ {(X, Y) \sim \Gamma} [ c (X, Y) ] = \inf _ {Q: Q _ {Z} = P _ {Z}} \mathbb {E} _ {P _ {X}} \mathbb {E} _ {Q (Z | X)} [ c (X, G (Z)) ],
$$

where  $Q_Z$  is the marginal distribution of  $Z$  when  $X \sim P_X$  and  $Z \sim Q(Z|X)$ .

Proof The proof is reported in Supplementary B.

This result allows us to optimize over random encoders  $Q(Z|X)$  instead of optimizing over all couplings between  $X$  and  $Y$ . Of course, both problems are still constrained. In order to implement a numerical solution we relax the constraints on  $Q_Z$  by adding a penalty to the objective. This finally leads us to the WAE objective:

$$
D _ {\mathrm {W A E}} \left(P _ {X}, P _ {G}\right) := \inf  _ {Q (Z | X) \in \mathcal {Q}} \mathbb {E} _ {P _ {X}} \mathbb {E} _ {Q (Z | X)} [ c (X, G (Z)) ] + \lambda \cdot \mathcal {D} _ {Z} \left(Q _ {Z}, P _ {Z}\right), \tag {4}
$$

where  $\mathcal{Q}$  is any nonparametric set of probabilistic encoders,  $\mathcal{D}_Z$  is an arbitrary divergence between  $Q_{Z}$  and  $P_Z$ , and  $\lambda >0$  is a hyperparameter. Similarly to VAE, we propose to use deep neural networks to parametrize both encoders  $Q$  and decoders  $G$ . Note that as opposed to VAEs, the WAE formulation allows for non-random encoders deterministically mapping inputs to their latent codes.

We propose two different penalties  $\mathcal{D}_Z(Q_Z, P_Z)$ :

GAN-based  $\mathcal{D}_Z$ . The first option is to choose  $\mathcal{D}_Z(Q_Z, P_Z) = D_{\mathrm{JS}}(Q_Z, P_Z)$  and use the adversarial training to estimate it. Specifically, we introduce an adversary (discriminator) in the latent space  $\mathcal{Z}$  trying to separate<sup>2</sup> "true" points sampled from  $P_Z$  and "fake" ones sampled from  $Q_Z$  (Goodfellow et al., 2014). This results in the WAE-GAN described in Algorithm 1. Even though WAE-GAN falls back to the min-max problem, we move the adversary from the input (pixel) space  $\mathcal{X}$  to the latent space  $\mathcal{Z}$ . On top of that,  $P_Z$  may have a nice shape with a single mode (for a Gaussian prior), in which case the task should be easier than matching an unknown, complex, and possibly multi-modal distributions as usually done in GANs. This is also a reason for our second penalty:

MMD-based  $\mathcal{D}_{\mathcal{Z}}$ . For a positive-definite reproducing kernel  $k: \mathcal{Z} \times \mathcal{Z} \to \mathcal{R}$  the following expression is called the maximum mean discrepancy (MMD):

$$
\mathrm {M M D} _ {k} (P _ {Z}, Q _ {Z}) = \left\| \int_ {\mathcal {Z}} k (z, \cdot) d P _ {Z} (z) - \int_ {\mathcal {Z}} k (z, \cdot) d Q _ {Z} (z) \right\| _ {\mathcal {H} _ {k}},
$$

where  $\mathcal{H}_k$  is the RKHS of real-valued functions mapping  $\mathcal{Z}$  to  $\mathcal{R}$ . If  $k$  is characteristic then  $\mathrm{MMD}_k$  defines a metric and can be used as a divergence measure. We propose to use  $\mathcal{D}_Z(P_Z, Q_Z) = \mathrm{MMD}_k(P_Z, Q_Z)$ . Fortunately, MMD has an unbiased U-statistic estimator, which can be used in conjunction with stochastic gradient descent (SGD) methods. This results in the WAE-MMD described in Algorithm 2. It is well known that the maximum mean discrepancy performs well when matching high-dimensional standard normal distributions (Gretton et al., 2012) so we expect this penalty to work especially well working with the Gaussian prior  $P_Z$ .

# 3 RELATED WORK

Literature on auto-encoders Classical unregularized auto-encoders minimize only the reconstruction cost. This results in different training points being encoded into non-overlapping zones chaotically scattered all across the  $\mathcal{Z}$  space with "holes" in between where the decoder mapping  $P_{G}(X|Z)$  has never been trained. Overall, the encoder  $Q(Z|X)$  trained in this way does not provide a useful representation and sampling from the latent space  $\mathcal{Z}$  becomes hard (Bengio et al., 2013).

Variational auto-encoders (Kingma & Welling, 2014) minimize a variational bound on the KL-divergence  $D_{\mathrm{KL}}(P_X, P_G)$  which is composed of the reconstruction cost plus

$\mathbb{E}_{P_X}[D_{\mathrm{KL}}(Q(Z|X),P_Z)]$  which captures how distinct the image by the encoder of each training example is from the prior  $P_Z$ , which is not guaranteeing that the overall encoded distribution  $\mathbb{E}_{P_X}[Q(Z|X)]$  matches  $P_Z$  like WAE does. Also, VAEs require non-degenerate Gaussian encoders and random decoders for which  $\log p_G(x|z)$  can be computed and differentiated with respect to the parameters. Later Mescheder et al. (2017) proposed a way to use VAE with non-Gaussian encoders. WAE minimizes OT  $W_{c}(P_{X},P_{G})$  and allows both probabilistic and deterministic encoder-decoder pairs of any kind.

# ALGORITHM 1 Wasserstein Auto-Encoder with GAN-based penalty (WAE-GAN).

Require: Regularization coefficient  $\lambda >0$  Initialize the parameters of the encoder  $Q_{\phi}$  decoder  $G_{\theta}$  , and latent discriminator  $D_{\gamma}$  while  $(\phi ,\theta)$  not converged do Sample  $\{x_1,\dots ,x_n\}$  from the training set Sample  $\{z_{1},\ldots ,z_{n}\}$  from the prior  $P_Z$  Sample  $\tilde{z}_i$  from  $Q_{\phi}(Z|x_i)$  for  $i = 1,\dots ,n$  Update  $D_{\gamma}$  by ascending:

$$
\frac {\lambda}{n} \sum_ {i = 1} ^ {n} \log D _ {\gamma} (z _ {i}) + \log \left(1 - D _ {\gamma} (\tilde {z} _ {i})\right)
$$

Update  $Q_{\phi}$  and  $G_{\theta}$  by descending:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} c \left(x _ {i}, G _ {\theta} \left(\tilde {z} _ {i}\right)\right) - \lambda \cdot \log D _ {\gamma} \left(\tilde {z} _ {i}\right)
$$

# end while

# ALGORITHM 2 Wasserstein Auto-Encoder with MMD-based penalty (WAE-MMD).

Require: Regularization coefficient  $\lambda >0$  ,characteristic positive-definite kernel  $k$  Initialize the parameters of the encoder  $Q_{\phi}$  decoder  $G_{\theta}$  , and latent discriminator  $D_{\gamma}$  while  $(\phi ,\theta)$  not converged do Sample  $\{x_1,\ldots ,x_n\}$  from the training set Sample  $\{z_{1},\dots,z_{n}\}$  from the prior  $P_Z$  Sample  $\tilde{z}_i$  from  $Q_{\phi}(Z|x_i)$  for  $i = 1,\dots ,n$  Update  $Q_{\phi}$  and  $G_{\theta}$  by descending:

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} c \left(x _ {i}, G _ {\theta} \left(\tilde {z} _ {i}\right)\right)
$$

$$
+ \frac {\lambda}{n (n - 1)} \sum_ {\ell \neq j} k \left(z _ {\ell}, z _ {j}\right) + k \left(\tilde {z} _ {\ell}, \tilde {z} _ {j}\right) - 2 k \left(z _ {\ell}, \tilde {z} _ {j}\right)
$$

# end while

When used with  $c(x, y) = \| x - y\|_2^2$  WAE-GAN is equivalent to adversarial auto-encoders (AAE) proposed by Makhzani et al. (2016). Our theory thus suggests that AAEs minimize the 2-Wasserstein distance between  $P_X$  and  $P_G$ . This provides the first theoretical justification for AAEs known to the authors. WAE generalizes AAE in two ways: first, it can use any cost function  $c$  in the input space  $\mathcal{X}$ ; second, it can use any discrepancy measure  $\mathcal{D}_Z$  in the latent space  $\mathcal{Z}$  (for instance MMD), not necessarily the adversarial one of WAE-GAN.

Literature on OT Geneva et al. (2016) address computing the OT cost in large scale using SGD and sampling. They approach this task either through the dual formulation, or via a regularized version of the primal. They do not discuss any implications for generative modeling. Our approach is based on the primal form of OT, we arrive at regularizers which are very different, and our main focus is on generative modeling.

The WGAN (Arjovsky et al., 2017) minimizes the 1-Wasserstein distance  $W_{1}(P_{X}, P_{G})$  for generative modeling. The authors approach this task from the dual form. Their algorithm comes without an encoder and can not be readily applied to any other cost  $W_{c}$ , because the neat form of the Kantorovich-Rubinstein duality (2) holds only for  $W_{1}$ . WAE approaches the same problem from the primal form, can be applied for any cost function  $c$ , and comes naturally with an encoder.

In order to compute the values (1) or (2) of OT we need to handle non-trivial constraints, either on the coupling distribution  $\Gamma$  or on the function  $f$  being considered. Various approaches have been proposed in the literature to circumvent this difficulty. For  $W_{1}$  Arjovsky et al. (2017) tried to implement the constraint in the dual formulation (2) by clipping the weights of the neural network  $f$ . Later Gulrajani et al. (2017) proposed to relax the same constraint by penalizing the objective of (2) with a term  $\lambda \cdot \mathbb{E}(\| \nabla f(X)\| -1)^2$  which should not be greater than 1 if  $f\in \mathcal{F}_L$ . In a more general OT setting of  $W_{c}$  Cuturi (2013) proposed to penalize the objective of (1) with the KL-divergence  $\lambda \cdot D_{\mathrm{KL}}(\Gamma ,P\otimes Q)$  between the coupling distribution and the product of marginals. Genevay et al. (2016) showed that this entropic regularization drops the constraints on functions in the dual formulation as opposed to (2). Finally, in the context of unbalanced optimal transport it

has been proposed to relax the constraint in (1) by regularizing the objective with  $\lambda \cdot \left(D_{f}(\Gamma_{X},P) + D_{f}(\Gamma_{Y},Q)\right)$  (Chizat et al., 2015; Liero et al., 2015), where  $\Gamma_{X}$  and  $\Gamma_{Y}$  are marginals of  $\Gamma$ . In this paper we propose to relax OT in a way similar to the unbalanced optimal transport, i.e. by adding additional divergences to the objective. However, we show that in the particular context of generative modeling, only one extra divergence is necessary.

![](images/52d015f35eaecf9514e200f46dd425ea0a8cb1e3aecaf91b5092b3bfc7c3f53d.jpg)  
Figure 2: VAE (left column), WAE-MMD (middle column), and WAE-GAN (right column) trained on MNIST dataset. In "test reconstructions" odd rows correspond to the real test points.

Literature on GANs Many of the GAN variations (including  $f$ -GAN and WGAN) come without an encoder. Often it may be desirable to reconstruct the latent codes and use the learned manifold, in which cases these models are not applicable.

There have been many other approaches trying to blend the adversarial training of GANs with autoencoder architectures (Zhao et al., 2017; Dumoulin et al., 2017; Ulyanov et al., 2017; Berthelot et al., 2017). The approach proposed by Ulyanov et al. (2017) is perhaps the most relevant to our work. The authors use the discrepancy between  $Q_{Z}$  and the distribution  $\mathbb{E}_{Z^{\prime}\sim P_Z}[Q(Z|G(Z^{\prime}))]$  of auto-encoded noise vectors as the objective for the max-min game between the encoder and decoder respectively. While the authors showed that the saddle points correspond to  $P_{X} = P_{G}$ , they admit that encoders and decoders trained in this way have no incentive to be reciprocal. As a workaround they propose to include an additional reconstruction term to the objective. WAE does not necessarily lead to a min-max game, uses a different penalty, and has a clear theoretical foundation.

Several works used reproducing kernels in context of GANs. Li et al. (2015); Dziugaite et al. (2015) use MMD with a fixed kernel  $k$  to match  $P_{X}$  and  $P_{G}$  directly in the input space  $\mathcal{X}$ . These methods have been criticised to require larger mini-batches during training: estimating  $\mathrm{MMD}_k(P_X, P_G)$  requires number of samples roughly proportional to the dimensionality of the input space  $\mathcal{X}$  (Reddi

et al., 2015) which is typically larger than  $10^{3}$ . Li et al. (2017) take a similar approach but further train  $k$  adversarially so as to arrive at a meaningful loss function. WAE-MMD uses MMD to match  $Q_{Z}$  to the prior  $P_{Z}$  in the latent space  $\mathcal{Z}$ . Typically  $\mathcal{Z}$  has no more than 100 dimensions and  $P_{Z}$  is Gaussian, which allows us to use regular mini-batch sizes to accurately estimate MMD.

![](images/3ce7541bca87b86741433787e8d715984b5fa153e575d20bae04e7b6a039b298.jpg)  
Figure 3: VAE (left column), WAE-MMD (middle column), and WAE-GAN (right column) trained on CelebA dataset. In "test reconstructions" odd rows correspond to the real test points.

# 4 EXPERIMENTS

In this section we empirically evaluate the proposed WAE model. We would like to test if WAE can simultaneously achieve (i) accurate reconstructions of data points, (ii) reasonable geometry of the latent manifold, and (iii) random samples of good (visual) quality. Importantly, the model should generalize well: requirements (i) and (ii) should be met on both training and test data. We trained WAE-GAN and WAE-MMD (Algorithms 1 and 2) on two real-world datasets: MNIST (LeCun et al., 1998) consisting of  $70\mathrm{k}$  images and CelebA (Liu et al., 2015) containing roughly  $203\mathrm{k}$  images.

Experimental setup In all reported experiments we used Euclidean latent spaces  $\mathcal{Z} = \mathcal{R}^{d_z}$  for various  $d_{z}$  depending on the complexity of the dataset, isotropic Gaussian prior distributions  $P_Z(Z) = \mathcal{N}(Z;0,\sigma_z^2\cdot I_d)$  over  $\mathcal{Z}$ , and a squared cost function  $c(x,y) = \| x - y\| _2^2$  for data points  $x,y\in \mathcal{X} = \mathcal{R}^{d_x}$ . We used deterministic encoder-decoder pairs, Adam (Kingma & Lei, 2014) with  $\beta_{1} = 0.5$ ,  $\beta_{2} = 0.999$ , and convolutional deep neural network architectures for encoder mapping  $Q_{\phi}\colon \mathcal{X}\to \mathcal{Z}$  and decoder mapping  $G_{\theta}\colon \mathcal{Z}\rightarrow \mathcal{X}$  similar to the DCGAN ones reported by Radford et al. (2016) with batch normalization (Ioffe & Szegedy, 2015). We tried various values of  $\lambda$  and noticed that  $\lambda = 10$  seems to work good across all datasets we considered. All reported experiments use this value.

Since we are using deterministic encoders, choosing  $d_{z}$  larger than intrinsic dimensionality of the dataset would force the encoded distribution  $Q_{Z}$  to live on a manifold in  $\mathcal{Z}$ . This would make matching  $Q_{Z}$  to  $P_{Z}$  impossible if  $P_{Z}$  is Gaussian and may lead to numerical instabilities. We use  $d_{z} = 8$  for MNIST and  $d_{z} = 64$  for CelebA which seems to work reasonably well.

We also report results of VAEs. VAEs used the same latent spaces as discussed above and standard Gaussian priors  $P_Z = \mathcal{N}(\mathbf{0},\mathbf{I}_d)$ . We used Gaussian encoders  $Q(Z|X) = \mathcal{N}\bigl (Z;Q_{\phi}(X),\Sigma (X)\bigr)$  with mean  $Q_{\phi}$  and diagonal covariance  $\boldsymbol{\Sigma}$ . For MNIST we used Bernoulli decoders parametrized by  $G_{\theta}$  and for CelebA the Gaussian decoders  $P_G(X|Z) = \mathcal{N}\bigl (Z;G_\theta (X),\sigma_G^2\cdot I_d\bigr)$  with mean  $G_{\theta}(Z)$ . Functions  $Q_{\phi}$ ,  $\boldsymbol{\Sigma}$ , and  $G_{\theta}$  were parametrized by deep nets of the same architectures as used in WAE.

WAE-GAN and WAE-MMD specifics In WAE-GAN we used discriminator  $D$  composed of several fully connected layers with ReLu. We tried WAE-MMD with the RBF kernel but observed that it fails to penalize the outliers of  $Q_{Z}$  because of the quick tail decay. If the codes  $\tilde{z} = Q_{\phi}(x)$  for some of the training points  $x \in \mathcal{X}$  end up far away from the support of  $P_{Z}$  (which may happen in the early stages of training) the corresponding terms in the U-statistic  $k(z, \tilde{z}) = e^{-\|\tilde{z} - z\|_2^2 / \sigma_k^2}$  will quickly approach zero and provide no gradient for those outliers. This could be avoided by choosing the kernel bandwidth  $\sigma_k^2$  in a data-dependent manner, however in this case per-minibatch U-statistic would not provide an unbiased estimate for the gradient. Instead, we used the inverse multiquadratic kernel  $k(x, y) = C / (C + \|x - y\|_2^2)$  which is also characteristic and has much heavier tails. In all experiments we used  $C = 2d_z\sigma_z^2$ , which is the expected squared distance between two multivariate Gaussian vectors drawn from  $P_{Z}$ . This significantly improved the performance compared to the RBF kernel (even the one with  $\sigma_k^2 = 2d_z\sigma_z^2$ ). Trained models are presented in Figures 2 and 3. Further details are presented in Supplementary E.

Random samples are generated by sampling  $P_Z$  and decoding the resulting noise vectors  $z$  into  $G_{\theta}(z)$ . As expected, in our experiments we observed that for both WAE-GAN and WAE-MMD the quality of samples strongly depends on how accurately  $Q_Z$  matches  $P_Z$ . To see this, notice that while training the decoder function  $G_{\theta}$  is presented only with encoded versions  $Q_{\phi}(X)$  of the data points  $X \sim P_X$ . Indeed, the decoder is trained on samples from  $Q_Z$  and thus there is no reason to expect good results when feeding it with samples from  $P_Z$ . In our experiments we noticed that even slight differences between  $Q_Z$  and  $P_Z$  may affect the quality of samples. In some cases WAE-GAN seems to lead to a better matching and generates better samples than WAE-MMD. However, due to adversarial training WAE-GAN is highly unstable, while WAE-MMD has a very stable training much like VAE.

In order to quantitatively assess the quality of the generated images, we use the Fréchet Inception Distance introduced by Heusel et al. (2017) and report the results on CelebA in Table 1. These results confirm that the sampled images from WAE are of better quality than from VAE, and WAE-GAN gets a slightly better score than WAE-MMD, which correlates with visual inspection of the images.

<table><tr><td>Algorithm</td><td>FID</td></tr><tr><td>VAE</td><td>98</td></tr><tr><td>WAE-MMD</td><td>55</td></tr><tr><td>WAE-GAN</td><td>42</td></tr></table>

Table 1: FID scores for samples on CelebA (smaller is better).

Test reconstructions and interpolations. We take random points  $x$  from the held out test set and report their auto-encoded versions  $G_{\theta}(Q_{\phi}(x))$ . Next, pairs  $(x,y)$  of different data points are sampled randomly from the held out test set and encoded:  $z_{x} = Q_{\phi}(x),z_{y} = Q_{\phi}(y)$ . We linearly interpolate between  $z_{x}$  and  $z_{y}$  with equally-sized steps in the latent space and show decoded images.

# 5 CONCLUSION

Using the optimal transport cost, we have derived Wasserstein auto-encoders—a new family of algorithms for building generative models. We discussed their relations to other probabilistic modeling techniques. We conducted experiments using two particular implementations of the proposed method, showing that in comparison to VAEs, the images sampled from the trained WAE models are of better quality, without compromising the stability of training and the quality of reconstruction. Future work will include further exploration of the criteria for matching the encoded distribution  $Q_{Z}$  to the prior distribution  $P_{Z}$ , assaying the possibility of adversarially training the cost function  $c$  in the input space  $\mathcal{X}$ , and a theoretical analysis of the dual formulations for WAE-GAN and WAE-MMD.

# REFERENCES

M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein GAN, 2017.  
Y. Bengio, A. Courville, and P. Vincent. Representation learning: A review and new perspectives. Pattern Analysis and Machine Intelligence, 35, 2013.  
D. Berthelot, T. Schumm, and L. Metz. Began: Boundary equilibrium generative adversarial networks, 2017.  
Lenaic Chizat, Gabriel Peyré, Bernhard Schmitzer, and François-Xavier Vialard. Unbalanced optimal transport: geometry and kantorovich formulation. arXiv preprint arXiv:1508.05216, 2015.  
M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, pp. 2292-2300, 2013.  
V. Dumoulin, I. Belghazi, B. Poole, A. Lamb, M. Arjovsky, O. Mastropietro, and A. Courville. Adversarily learned inference. In *ICLR*, 2017.  
G. K. Dziugaite, D. M. Roy, and Z. Ghahramani. Training generative neural networks via maximum mean discrepancy optimization. In UAI, 2015.  
A. Genevay, M. Cuturi, G. Peyré, and F. R. Bach. Stochastic optimization for large-scale optimal transport. In Advances in Neural Information Processing Systems, pp. 3432-3440, 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, pp. 2672-2680, 2014.  
A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Scholkopf, and A. J. Smola. A kernel two-sample test. Journal of Machine Learning Research, 13:723-773, 2012.  
I. Gulrajani, F. Ahmed, M. Arjovsky, V. Domoulin, and A. Courville. Improved training of wasserstein GANs, 2017.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a nash equilibrium. arXiv preprint arXiv:1706.08500, 2017.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift, 2015.  
D. P. Kingma and J. Lei. Adam: A method for stochastic optimization, 2014.  
D. P. Kingma and M. Welling. Auto-encoding variational Bayes. In ICLR, 2014.  
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. In Proceedings of the IEEE, volume 86(11), pp. 2278-2324, 1998.  
C. L. Li, W. C. Chang, Y. Cheng, Y. Yang, and B. Poczos. Mmd gan: Towards deeper understanding of moment matching network, 2017.  
Y. Li, K. Swersky, and R. Zemel. Generative moment matching networks. In ICML, 2015.  
Matthias Liero, Alexander Mielke, and Giuseppe Savare. Optimal entropy-transport problems and a new hellinger-kantorovich distance between positive measures. arXiv preprint arXiv:1508.07941, 2015.  
F. Liese and K.-J. Miescke. Statistical Decision Theory. Springer, 2008.  
J. Lin. Divergence measures based on the shannon entropy. Information Theory, 37, 1991.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015.  
A. Makhzani, J. Shlens, N. Jaitly, and I. Goodfellow. Adversarial autoencoders. In ACLR, 2016.

L. Mescheder, S. Nowozin, and A. Geiger. Adversarial variational bayes: Unifying variational autoencoders and generative adversarial networks, 2017.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In NIPS, 2016.  
B. Poole, A. Alemi, J. Sohl-Dickstein, and A. Angelova. Improved generator objectives for GANs, 2016.  
A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016.  
R. Reddi, A. Ramdas, A. Singh, B. Poczos, and L. Wasserman. On the high-dimensional power of a linear-time two sample test under mean-shift alternatives. In AISTATS, 2015.  
A. B. Tsybakov. Introduction to Nonparametric Estimation. Springer, NY, 2008.  
D. Ulyanov, A. Vedaldi, and V. Lempitsky. It takes (only) two: Adversarial generator-encoder networks, 2017.  
C. Villani. Topics in Optimal Transportation. AMS Graduate Studies in Mathematics, 2003.  
J. Zhao, M. Mathieu, and Y. LeCun. Energy-based generative adversarial network. In ICLR, 2017.
