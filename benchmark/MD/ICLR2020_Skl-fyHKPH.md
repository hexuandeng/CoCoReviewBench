# A MEAN-FIELD THEORY FOR KERNEL ALIGNMENT WITH RANDOM FEATURES IN GENERATIVE ADVERSE-RIAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a novel supervised learning method to optimize the kernel in maximum mean discrepancy generative adversarial networks (MMD GANs). Specifically, we characterize a distributionally robust optimization problem to compute a good distribution for the random feature model of Rahimi and Recht to approximate a good kernel function. Due to the fact that the distributional optimization is infinite dimensional, we consider a Monte-Carlo sample average approximation (SAA) to obtain a more tractable finite dimensional optimization problem. We subsequently leverage a particle stochastic gradient descent (SGD) method to solve finite dimensional optimization problems. Based on a mean-field analysis, we then prove that the empirical distribution of the interactive particles system at each iteration of the SGD follows the path of the gradient descent flow on the Wasserstein manifold. We also establish the non-asymptotic consistency of the finite sample estimator. Our empirical evaluation on synthetic data-set as well as MNIST and CIFAR-10 benchmark data-sets indicates that our proposed MMD GAN model with kernel learning indeed attains higher inception scores well as Fréchet inception distances and generates better images compared to the generative moment matching network (GMMN) and MMD GAN with untrained kernels.

# 1 INTRODUCTION

A fundamental and long-standing problem in unsupervised learning systems is to capture the underlying distribution of data. While deep generative models such as Boltzmann machines Salakhutdinov & Hinton (2009) and auto-encoding variational Bayes Kingma & Welling (2013) learn the data distribution, they are inadequate for many intractable probabilistic computations that arise in maximum likelihood estimation. Moreover, in many machine learning tasks such as caption generation Xu et al. (2015), the main objective is to obtain new samples rather than to accurately estimate the underlying data distribution. Generative adversarial network (GAN) Goodfellow et al. (2014) provides a framework to directly draw new samples without estimating data distribution. It consists of a deep feedforward network to generate new samples from a base distribution (e.g. Gaussian distribution), and a discriminator network to accept or reject the generated samples. However, training GAN requires finding a Nash equilibrium of a non-convex minimax game with continuous, high-dimensional parameters. Consequently, it is highly unstable and prone to miss modes Salimans et al. (2016); Che et al. (2016). To obtain more stable models, the generative moment matching networks (GMMNs) Li et al. (2015) are proposed, wherein instead of training a discriminator network, a nonparametric statistical hypothesis test is performed to accept or reject the generated samples via the computation of the kernel maximum mean discrepancy Gretton et al. (2007). While leveraging a statistical test simplifies the loss function for training GMMN, in practice, the diversity of generated samples by GMMN is highly sensitive to the choice of the kernel. Thus, to improve the sampling

performance, the kernel function also needs to be jointly optimized with the generator. Rather than optimizing the kernel directly, the MMD GAN model Li et al. (2017) is proposed in which an embedding function is optimized in conjunction with a fixed user-defined kernel (e.g. RBF Gaussian kernel). However, there are no theoretical guarantees that the user-defined kernel is the 'right' kernel for embedded features.

Contributions. To address the kernel model selection problem in MMD GAN Li et al. (2017), in this paper we put forth a novel framework to learn a good kernel function from training data. Our kernel learning approach is based on a distributional optimization problem to learn a good distribution for the random feature model of Rahimi and Recht Rahimi & Recht (2008; 2009) to approximate the kernel. Since optimization with respect to the distribution of random features is infinite dimensional, we consider a Monte Carlo approximation to obtain a more tractable finite dimensional optimization problem with respect to the samples of the distribution. We then use a particle stochastic gradient descent (SGD) to solve the approximated finite dimensional optimization problem. We provide a theoretical guarantee for the consistency of the finite sample-average approximations. Based on a mean-field analysis, we also show the consistency of the proposed particle SGD. In particular, we show that when the number of particles tends to infinity, the empirical distribution of the particles in SGD follows the path of the gradient descent flow of the distributional optimization problem on the Wasserstein manifold.

# 2 PRELIMINARIES OF MMD GANS

Assume we are given data  $\{\pmb{v}_i\}_{i=1}^n$  that are sampled from an unknown distribution  $P_V$  with the support  $\mathcal{V}$ . In many unsupervised tasks, we wish to attain new samples from the distribution  $P_V$  without directly estimating it. Generative Adversarial Network (GAN) Goodfellow et al. (2014) provides such a framework. In vanilla GAN, a deep network  $\mathcal{G}(\cdot; \omega)$  parameterized by  $\omega \in \Omega$  is trained as a generator to transform the samples  $\pmb{Z} \sim P_Z$ ,  $\pmb{Z} \in \mathcal{Z}$  from a user-defined distribution  $P_Z$  (e.g. Gaussian) into a new sample  $\mathcal{G}(\pmb{Z}; \omega) \sim P_W$ , such that the distributions  $P_W$  and  $P_V$  are close. In addition, a discriminator network  $\mathcal{D}(\cdot; \delta)$  parameterized by  $\delta \in \Delta$  is also trained to reject or accept the generated samples as a realization of the data distribution. The training of the generator and discriminator networks is then accomplished via solving a minimax optimization problem as below

$$
\min  _ {\boldsymbol {\omega} \in \Omega} \max  _ {\boldsymbol {\delta} \in \Delta} \mathbb {E} _ {P _ {V}} [ \mathcal {D} (\boldsymbol {X}; \boldsymbol {\delta}) ] + \mathbb {E} _ {P _ {Z}} [ \log (1 - \mathcal {D} (\mathcal {G} (\boldsymbol {Z}; \boldsymbol {\omega}); \boldsymbol {\delta})) ]. \tag {1}
$$

In the high dimensional settings, the generator trained via the min-max program of equation 1 can potentially collapse to a single mode of distribution where it always emits the same point Che et al. (2016). To overcome this shortcoming, other adversarial generative models are proposed in the literature, which propose to modify or replace the discriminator network by a statistical two-sample test based on the notion of the maximum mean discrepancy which is defined below:

Definition 2.1. (MAXIMUM MEAN DISCREPANCY GRETTON ET AL. (2007)) Let  $(\mathcal{X},d)$  be a metric space,  $\mathcal{F}$  be a class of functions  $f:\mathcal{X}\to \mathbb{R}$ , and  $P,Q\in \mathcal{B}(\mathcal{X})$  be two probability measures from the set of all Borel probability measures  $\mathcal{B}(\mathcal{X})$  on  $\mathcal{X}$ . The maximum mean discrepancy (MMD) between the distributions  $P$  and  $Q$  with respect to the function class  $\mathcal{F}$  is defined below

$$
D _ {\mathcal {F}} [ P, Q ] \stackrel {\text {d e f}} {=} \sup  _ {f \in \mathcal {F}} \int_ {\mathcal {X}} f (\boldsymbol {x}) (P - Q) (\mathrm {d} \boldsymbol {x}). \tag {2}
$$

Different choices of the function class  $\mathcal{F}$  in equation 2 yield different adversarial models such as Wasserstein GANs (WGAN) Arjovsky et al. (2017),  $f$ -GANs Nowozin et al. (2016), and GMMN and MMD GAN Li et al. (2017; 2015). In the latter two cases, the function class  $\mathcal{F}$  corresponds to a reproducing kernel Hilbert space (RKHS) of functions with a kernel  $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ , denoted by  $(\mathcal{H}_{\mathcal{X}}, K)$ . Then, the squared MMD loss in equation 2 as a measure of the distance between the

distributions  $P_{\mathbf{V}}$  and  $P_{\mathbf{W}}$  has the following expression

$$
\begin{array}{l} D _ {K} \left[ P _ {\mathbf {V}}, P _ {\mathbf {W}} \right] \stackrel {\text {d e f}} {=} \sup  _ {f \in \mathcal {H} _ {\mathcal {X}}} \left\| \int_ {\mathcal {X}} K (\cdot , \mathbf {x}) \left(P _ {\mathbf {V}} - P _ {\mathbf {W}}\right) (\mathrm {d} \mathbf {x}) \right\| _ {\mathcal {H} _ {\mathcal {X}}} ^ {2} (3) \\ = \mathbb {E} _ {\boldsymbol {V}, \boldsymbol {V} ^ {\prime} \sim P _ {\boldsymbol {V}}} [ K (\boldsymbol {V}; \boldsymbol {V} ^ {\prime}) ] + \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {W} ^ {\prime} \sim P _ {\boldsymbol {W}}} [ K (\boldsymbol {W}, \boldsymbol {W} ^ {\prime}) ] - 2 \mathbb {E} _ {\boldsymbol {V} \sim P _ {\boldsymbol {V}}, \boldsymbol {W} \sim P _ {\boldsymbol {W}}} [ K (\boldsymbol {V}; \boldsymbol {W}) ], (4) \\ \end{array}
$$

where  $\mathcal{X} = \mathcal{V}\cup \Omega$ . Instead of training the generator via solving the minimax optimization in equation 1, the MMD GAN model of Li et al. (2015) proposes to optimize the discrepancy between two distributions via optimization of an embedding function  $\iota :\mathbb{R}^d\mapsto \mathbb{R}^p,p\leq d$ , i.e.,

$$
\min  _ {\boldsymbol {W} \in \mathcal {W}} \max  _ {\iota \in \mathcal {Q}} \operatorname {M M D} _ {k \circ \iota} [ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} ], \tag {5}
$$

where  $k: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$  is a user-defined fixed kernel. In Li et al. (2015), the proposal for the kernel  $k: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$  is a mixture of the Gaussians,

$$
k \circ \iota (\boldsymbol {x}, \boldsymbol {y}) = k (\iota (\boldsymbol {x}), \iota (\boldsymbol {y})) = \sum_ {i = 1} ^ {m} \left(\frac {\| \iota (\boldsymbol {x}) - \iota (\boldsymbol {y}) \| _ {2} ^ {2}}{\sigma_ {i} ^ {2}}\right), \tag {6}
$$

where the bandwidth parameters  $\sigma_1, \dots, \sigma_m > 0$  are manually selected. Nevertheless, in practice there is no guarantee that the user-defined kernel  $k(\iota(\pmb{x}), q(\pmb{y}))$  can capture the structure of the embedded features  $\iota(\pmb{x})$ .

# 3 PROPOSED APPROACH: KERNEL LEARNING WITH RANDOM FEATURES FOR MMD GANS

In this section, we first expound our kernel learning approach. Then, we describe a novel MMD GAN model based on the proposed kernel learning approach.

# 3.1 ROBUST DISTRIBUTIONAL OPTIMIZATION FOR KERNEL LEARNING

To address the kernel model selection issue in MMD GAN Li et al. (2017), we consider a kernel optimization scheme with random features Rahimi & Recht (2008; 2009). Let  $\varphi : \mathbb{R}^d \times \mathbb{R}^D \to [-1,1]$  denotes the explicit feature maps and  $\mu \in \mathcal{M}(\mathbb{R}^D)$  denotes a probability measure from the space of probability measures  $\mathcal{M}(\mathbb{R}^D)$  on  $\mathbb{R}^D$ . The kernel function is characterized via the explicit feature maps using the following integral equation

$$
K _ {\mu} (\boldsymbol {x}, \boldsymbol {y}) = \mathbb {E} _ {\mu} [ \varphi (\boldsymbol {x}; \boldsymbol {\xi}) \varphi (\boldsymbol {y}; \boldsymbol {\xi}) ] = \int_ {\Xi} \varphi (\boldsymbol {x}; \boldsymbol {\xi}) \varphi (\boldsymbol {y}; \boldsymbol {\xi}) \mu (\mathrm {d} \boldsymbol {\xi}). \tag {7}
$$

Let  $\mathrm{MMD}_{\mu}[P_V, P_W] \stackrel{\mathrm{def}}{=} \mathrm{MMD}_{K_{\mu}}[P_V, P_W]$ . Then, the kernel optimization problem in can be formulated as a distribution optimization for random features, i.e,

$$
\min  _ {\boldsymbol {W} \in \mathcal {W}} \sup  _ {\mu \in \mathcal {P}} \mathrm {M M D} _ {\mu} \left[ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} \right]. \tag {8}
$$

Here,  $\mathcal{P}$  is the set of probability distributions corresponding to a kernel class  $\kappa$ . In the sequel, we consider  $\mathcal{P}$  to be the distribution ball of radius  $R$  as below

$$
\mathcal {P} \stackrel {\text {d e f}} {=} \mathbb {B} _ {R} ^ {p} (\mu_ {0}) \stackrel {\text {d e f}} {=} \left\{\mu \in \mathcal {M} \left(\mathbb {R} ^ {D}\right): d \left(\mu , \mu_ {0}\right) \leq R \right\}, \tag {9}
$$

where  $\mu_0$  is a user-defined base distribution, and  $d(\cdot ,\cdot):\mathcal{M}(\mathbb{R}^D)\times \mathcal{M}(\mathbb{R}^D)\to \mathbb{R}$  is a distance on the measure space  $\mathcal{M}(\mathbb{R}^D)$ .

The kernel MMD loss function in equation 8 is defined with respect to the unknown distributions of the data-set  $P_V$  and the model  $P_W$ . Therefore, we construct an unbiased estimator for the MMD

loss function in equation 8 based on the training samples. To describe the estimator, sample the labels from a uniform distribution  $y_{1},\dots ,y_{n}\sim \mathrm{i.i.d.}$  Uniform  $\{-1, + 1\}$ , where we assume that the number of positive and negative labels are balanced. In particular, consider the set of positive labels  $\mathcal{I} = \{i\in \{1,2,\dots ,n\} :y_i = +1\}$ , and negative labels  $\mathcal{J} = \{1,2,\dots ,n\} /\mathcal{I}$ , where their cardinality is  $|\mathcal{I}| = |\mathcal{J}| = \frac{n}{2}$ . We consider the following assignment of labels:

- Positive class labels: If  $y_{i} = +1$ , sample the corresponding feature map from data-distribution  $\pmb{x}_i = \pmb{v}_i \sim P_V$ .  
- Negative class labels: If  $y_{i} = -1$ , sample from the corresponding feature map from the generated distribution  $\pmb{x}_i = \mathcal{G}(\pmb{Z}_i, \pmb{W}) \sim P_{\pmb{W}}, \pmb{Z}_i \sim P_{\pmb{Z}}$ .

By this construction, the joint distribution of features and labels  $P_{Y,X}$  has the marginals  $P_{X|Y = +1} = P_V$ , and  $P_{X|Y = -1} = P_W$ . Moreover, the following statistic, known as the kernel alignment in the literature (see, e.g., Sinha & Duchi (2016); Cortes et al. (2012)), is an unbiased estimator of the MMD loss in equation 8,

$$
\min  _ {\boldsymbol {W} \in \mathcal {W}} \sup  _ {\mu \in \mathcal {P}} \widehat {\operatorname {M M D}} _ {\mu} \left[ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} \right] \stackrel {\text {d e f}} {=} \frac {8}{n (n - 1)} \sum_ {1 \leq i <   j \leq n} y _ {i} y _ {j} K _ {\mu} \left(\boldsymbol {x} _ {i}, \boldsymbol {x} _ {j}\right). \tag {10}
$$

See Appendix B.1 for the related proof. The kernel alignment in equation 10 can also be viewed through the lens of the risk minimization

$$
\begin{array}{l} \min  _ {\boldsymbol {W} \in \mathcal {W}} \inf  _ {\mu \in \mathcal {P}} \widehat {\operatorname {M M D}} _ {\mu} ^ {\alpha} \left[ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} \right] \stackrel {\text {d e f}} {=} \frac {8}{n (n - 1) \alpha} \sum_ {1 \leq i <   j \leq n} \left(\alpha y _ {i} y _ {j} - K _ {\mu} \left(\boldsymbol {x} _ {i}, \boldsymbol {x} _ {j}\right)\right) ^ {2} (11a) \\ = \frac {8}{n (n - 1) \alpha} \sum_ {1 \leq i <   j \leq n} \left(\alpha y _ {i} y _ {j} - \mathbb {E} _ {\mu} \left[ \varphi \left(\boldsymbol {x} _ {i}; \boldsymbol {\xi}\right) \varphi \left(\boldsymbol {x} _ {j}; \boldsymbol {\xi}\right) \right]\right) ^ {2}. (11b) \\ \end{array}
$$

Here,  $\alpha > 0$  is a scaling factor that determines the separation between feature vectors, and  $\mathbf{K}_* \stackrel{\mathrm{def}}{=} \alpha \mathbf{y} \mathbf{y}^T$  is the ideal kernel that provides the maximal separation between the feature vectors over the training data-set, i.e.,  $K_*(\boldsymbol{x}_i, \boldsymbol{x}_j) = \alpha$  when features have identical labels  $y_i = y_j$ , and  $K_*(\boldsymbol{x}_i, \boldsymbol{x}_j) = -\alpha$  otherwise. Upon expansion of the risk function in equation 11, it can be easily shown that it reduces to the kernel alignment in equation 10 when  $\alpha \rightarrow +\infty$ . Intuitively, the risk minimization in equation 11 gives a feature space in which pairwise distances are similar to those in the output space  $\mathcal{Y} = \{-1, +1\}$ .

# 3.2 SAA FOR DISTRIBUTIONAL OPTIMIZATION

The distributional optimization problem in equation 8 is infinite dimensional, and thus cannot be solved directly. To obtain a tractable optimization problem, instead of optimizing with respect to the distribution  $\mu$  of random features, we optimize the i.i.d. samples (particles)  $\xi^1,\dots ,\xi^N\sim_{\mathrm{i.i.d.}}\mu$  generated from the distribution. The empirical distribution of these particles is accordingly defined as follows

$$
\widehat {\mu} ^ {N} (\boldsymbol {\xi}) \stackrel {\text {d e f}} {=} \frac {1}{N} \sum_ {k = 1} ^ {N} \delta \left(\boldsymbol {\xi} - \boldsymbol {\xi} ^ {k}\right), \tag {12}
$$

where  $\delta (\cdot)$  is the Dirac's delta function concentrated at zero. In practice, the optimization problem in equation 11 is solved via the Monte-Carlo sample average approximation of the objective function,

$$
\min  _ {\boldsymbol {W} \in \mathcal {W}} \min  _ {\hat {\mu} ^ {N} \in \mathcal {P} _ {N}} \widehat {\mathrm {M M D}} _ {\hat {\mu} ^ {N}} ^ {\alpha} \left[ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} \right] = \frac {8}{n (n - 1) \alpha} \sum_ {1 \leq i <   j \leq n} \left(\alpha y _ {i} y _ {j} - \frac {1}{N} \sum_ {k = 1} ^ {N} \varphi (\boldsymbol {x} _ {i}; \boldsymbol {\xi} ^ {k}) \varphi (\boldsymbol {x} _ {j}; \boldsymbol {\xi} ^ {k})\right) ^ {2},
$$

where  $\mathcal{P}_N\stackrel {\mathrm{def}}{=}\mathbb{B}_R^N (\widehat{\mu}_0^N) = \Big\{\widehat{\mu}^N\in \mathcal{M}(\mathbb{IR}^D):d(\widehat{\mu}^N,\widehat{\mu}_0^N)\leq R\Big\}$ , and  $\widehat{\mu}_0^N$  is the empirical measure associated with the samples  $\pmb{\xi}_0^1,\dots ,\pmb{\xi}_0^N\sim \mathrm{i.i.d.}\mu_0$ . The empirical objective function in equation 13 can be optimized with respect to the samples  $\pmb{\xi}^{1},\dots ,\pmb{\xi}^{N}$  using the particle stochastic gradient descent. For the optimization problem in equation 13, the (projected) stochastic gradient descent (SGD) takes the following recursive form,

$$
\boldsymbol {\xi} _ {m + 1} ^ {k} = \boldsymbol {\xi} _ {m} ^ {k} - \frac {\eta}{N} \left(y _ {m} \widetilde {y} _ {m} - \frac {1}{\alpha N} \sum_ {k = 1} ^ {N} \varphi \left(\boldsymbol {x} _ {m}; \boldsymbol {\xi} _ {m} ^ {k}\right) \varphi \left(\widetilde {\boldsymbol {x}} _ {m}; \boldsymbol {\xi} _ {m} ^ {k}\right)\right) \nabla_ {\boldsymbol {\xi}} \left(\varphi \left(\boldsymbol {x} _ {m}; \boldsymbol {\xi} _ {m} ^ {k}\right) \varphi \left(\widetilde {\boldsymbol {x}} _ {m}; \boldsymbol {\xi} _ {m} ^ {k}\right)\right), \tag {14a}
$$

for  $k = 1,2,\dots ,N$ , where  $(y_{m},\pmb{x}_{m})$ ,  $(\widetilde{y}_m,\widetilde{\pmb{x}}_m) \sim_{\mathrm{i.i.d}} P_{\pmb{x},y}$  and  $\eta \in \mathbb{R}_{>0}$  denotes the learning rate of the algorithm, and the initial particles are  $\pmb{\xi}_0^1,\dots ,\pmb{\xi}_0^N\sim_{\mathrm{i.i.d.}}\mu_0$ . At each iteration of the SGD dynamic in equation 14, a feasible solution for the inner optimization of the empirical risk function in equation 13 is generated via the empirical measure

$$
\widehat {\mu} _ {m} ^ {N} (\boldsymbol {\xi}) = \frac {1}{N} \sum_ {k = 1} ^ {N} \delta \left(\boldsymbol {\xi} - \boldsymbol {\xi} _ {m} ^ {k}\right). \tag {15}
$$

Indeed, we prove in Section 4 that for an appropriate choice of the learning rate  $\eta >0$ , the empirical measure in equation 15 remains inside the distribution ball  $\widehat{\mu}_m^N\in \mathcal{P}_N$  for all  $m\in [0,NT]\cap \mathbb{N}$  under an appropriate metric  $d$  on  $\mathcal{M}(\mathbb{R}^D)$ , and is thus a feasible solution for the empirical risk minimization equation 13 (see Corollary 4.2.1 in Section 4).

# 3.3 PROPOSED MMD GAN WITH KERNEL LEARNING

In Algorithm 1, we describe the proposed method MMD GAN model with the kernel learning approach described earlier. Algorithm 1 has an inner loop for the kernel training and an outer loop for training the generator, where we employ RMSprop Tieleman & Hinton (2012). Our proposed MMD GAN model is distinguished from MMD GAN of Li et al. (2017) in that we learn a good kernel function in equation 16 of the inner loop instead of optimizing the embedding function that is implemented by an auto-encoder. However, we mention that our kernel learning approach is compatible with the auto-encoder implementation of Li et al. (2017) for dimensionality reduction of features (and particles). In the case of including an auto-encoder, the inner loop in Algorithm 1 must be modified to add an additional step for training the auto-encoder. However, to convey the main ideas more clearly, the training step of the auto-encoder is omitted from Algorithm 1.

# 4 CONSISTENCY AND A MEAN-FIELD ANALYSIS

In this section, we provide theoretical guarantees for the consistency of various approximations we made to optimize the population MMD loss function in equation 8. We defer the proofs of the following theoretical results to Appendix B. The main assumptions ((A.1), (A.2), and (A.3)) underlying our theoretical results are also stated in the same section.

Consistency of finite-sample estimate: In this part, we prove that the solution to finite sample optimization problem in equation 13 approaches its population optimum in equation 8 as the number of data points as well as the number of random feature samples tends to infinity. To establish the proof, we consider the  $d_{W_p}(\cdot ,\cdot)$  is the  $p$  -Wasserstein (a.k.a. Kantorovich-Rubinstein metric) distance

# Algorithm 1 MMD GAN with a supervised kernel learning Method (Monte-Carlo Approach)

Inputs: The learning rates  $\tilde{\eta}$ ,  $\eta > 0$ , the number of iterations of discriminator per generator update  $T \in \mathbb{N}$ , the batch-size  $n$ , the number of random features  $N \in \mathbb{N}$ . Regularization parameter  $\alpha > 0$ .

while  $\omega$  has not converged do

for  $t = 1,2,\dots ,T$  do

Sample the labels  $y, \widetilde{y} \sim_{\mathrm{i.i.d}}$  Uniform  $\{-1, 1\}$ .

Sample the features  $\pmb{x}|y = +1\sim P_V$  , and  $\pmb {x}|y = -1\sim P_W$  . Similarly,  $\tilde{\pmb{x}} |\tilde{y} = +1\sim P_V$  , and  $\tilde{\pmb{x}} |\tilde{y} = -1\sim P_W$

For all  $k = 1,2,\dots ,N$  ,update the particles,

$$
\boldsymbol {\xi} ^ {k} \leftarrow \boldsymbol {\xi} ^ {k} - \frac {\eta}{N} \left(\alpha y \widetilde {y} - \frac {1}{N} \sum_ {k = 1} ^ {N} \varphi (\boldsymbol {x}; \boldsymbol {\xi} ^ {k}) \varphi (\widetilde {\boldsymbol {x}}; \boldsymbol {\xi} ^ {k})\right) \nabla_ {\boldsymbol {\xi}} \left(\varphi (\boldsymbol {x}; \boldsymbol {\xi} ^ {k}) \varphi (\widetilde {\boldsymbol {x}}; \boldsymbol {\xi} ^ {k})\right), \tag {16}
$$

end for

Sample a balanced minibatch of labels  $\{y_{i}\}_{i = 1}^{n}\sim_{\mathrm{i.i.d.}}$  Uniform{-1,+1}.

Sample the minibatch  $\{\pmb{x}\}_{i=1}^{n}$  such that  $\pmb{x}_i|y_i = +1 \sim P_V$ , and  $\pmb{x}_i|y_i = -1 \sim P_W$  for all  $i = 1,2,\dots,n$ .

Update the generator

$$
\boldsymbol {g} _ {\omega} \leftarrow \nabla_ {\omega} \widehat {D} _ {\widehat {\mu} ^ {N}} ^ {\alpha} \left[ P _ {V}, P _ {W} \right], \quad \widehat {\mu} ^ {N} = \frac {1}{N} \sum_ {k = 1} ^ {N} \delta \left(\boldsymbol {\xi} - \boldsymbol {\xi} ^ {k}\right). \tag {17a}
$$

$$
\boldsymbol {w} \leftarrow \boldsymbol {w} - \tilde {\eta} \operatorname {R M S p r o p} \left(\boldsymbol {g} _ {\omega}, \boldsymbol {\omega}\right). \tag {17b}
$$

# end while

defined as below

$$
d _ {W _ {p}} \left(\mu_ {1}, \mu_ {2}\right) \stackrel {\text {d e f}} {=} \left(\inf  _ {\pi \in \Pi \left(\mu_ {1}, \mu_ {2}\right)} \int_ {\mathbb {R} ^ {D} \times \mathbb {R} ^ {D}} \| \boldsymbol {\xi} _ {1} - \boldsymbol {\xi} _ {2} \| _ {2} ^ {p} \mathrm {d} \pi \left(\boldsymbol {\xi} _ {1}, \boldsymbol {\xi} _ {2}\right)\right) ^ {\frac {1}{p}}, \tag {18}
$$

where the infimum is taken with respect to all couplings  $\pi$  of the measures  $\mu, \mu_0 \in \mathcal{M}(\mathbb{R}^D)$ , and  $\Pi(\mu, \mu_0)$  is the set of all such couplings.

Theorem 4.1. (NON-ASYMPTOTIC CONSISTENCY OF FINITE-SAMPLE ESTIMATOR) Suppose conditions (A.1)-(A.3) of Appendix B are satisfied. Consider the distribution balls  $\mathcal{P}$  and  $\mathcal{P}_N$  that are defined with respect to the 2-Wasserstein distance. Furthermore, consider the optimal MMD values of the population optimization and its finite sample estimate

$$
\left(\boldsymbol {W} _ {*}, \mu_ {*}\right) \stackrel {\text {d e f}} {=} \arg \min  _ {\boldsymbol {W} \in \mathcal {W}} \arg \sup  _ {\mu \in \mathcal {P}} \mathrm {M M D} _ {\mu} \left[ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} \right]. \tag {19a}
$$

$$
\left(\widehat {\boldsymbol {W}} _ {*} ^ {N}, \widehat {\mu} _ {*} ^ {N}\right) \stackrel {\text {d e f}} {=} \arg \min  _ {\boldsymbol {W} \in \mathcal {W}} \arg \inf  _ {\widehat {\mu} ^ {N} \in \mathcal {P} _ {N}} \widehat {\mathrm {M M D}} _ {\widehat {\mu} ^ {N}} ^ {\alpha} \left[ P _ {\boldsymbol {V}}, P _ {\boldsymbol {W}} \right], \tag {19b}
$$

respectively. Then, with the probability of (at least)  $1 - 3\varrho$  over the training data samples  $\{(\pmb{x}_i,y_i)\}_{i = 1}^n$  and the random feature samples  $\{\pmb{\xi}_0^k\}_{k = 1}^N$ , the following non-asymptotic bound holds

$$
\begin{array}{l} \left| \mathrm {M M D} _ {\mu_ {*}} \left[ P _ {\mathbf {V}}, P _ {\mathbf {W} _ {*}} \right] - \mathrm {M M D} _ {\widehat {\mu} _ {*} ^ {N}} \left[ P _ {\mathbf {V}}, P _ {\widehat {\mathbf {W}} _ {*} ^ {N}} \right] \right| \tag {20} \\ \leq \sqrt {\frac {L ^ {2} (d + 2)}{N}} \ln^ {\frac {1}{2}} \left(\frac {2 ^ {8} N \mathrm {d i a m} ^ {2} (\mathcal {X})}{\varrho}\right) + 2 \max \left\{\frac {c _ {1} L ^ {2}}{n} \ln^ {\frac {1}{2}} \left(\frac {4}{\varrho}\right), \frac {c _ {2} R L ^ {4}}{n ^ {2}} \ln \left(\frac {4 e ^ {\frac {L ^ {4}}{9}}}{\varrho}\right) \right\} + \frac {8 L ^ {2}}{\alpha}, \\ \end{array}
$$

where  $c_{1} = 3^{\frac{1}{4}} \times 2^{4}$ , and  $c_{2} = 9 \times 2^{11}$ .

The proof of Theorem 4.1 is presented in Appendix B.1.

Notice that there are three key parameters involved in the upper bound of Theorem 4.1. Namely, the number of training samples  $n$ , the number of random feature samples  $N$ , and the regularization parameter  $\alpha$ . The upper bound in equation 20 thus shows that when  $n, N, \alpha \to +\infty$ , the solution obtained from solving the empirical risk minimization in equation 11 yields a MMD population value tending to the optimal value of the distributional optimization in equation 8.

Consistency of particle SGD for solving distributional optimization. The consistency result of Theorem 4.1 is concerned with the MMD value of the optimal empirical measure  $\widehat{\mu}_{*}^{N}(\pmb {\xi}) = \frac{1}{N}\sum_{k = 1}^{N}\delta (\pmb {\xi} - \pmb{\xi}_{*}^{k})$  of the empirical risk minimization equation 13. In practice, the particle SGD is executed for a few iterations and its values are used as an estimate for  $(\pmb{\xi}_{*}^{1},\dots ,\pmb{\xi}_{*}^{N})$ . Consequently, it is desirable to establish a similar consistency type result for the particle SGD estimates  $(\pmb{\xi}_{m}^{1},\dots ,\pmb{\xi}_{m}^{N})$  at the  $m$ -th iteration. To reach this objective, we define the scaled empirical measure as follows

$$
\mu_ {t} ^ {N} = \widehat {\mu} _ {\lfloor N t \rfloor} ^ {N} = \frac {1}{N} \sum_ {k = 1} ^ {N} \delta (\boldsymbol {\xi} - \boldsymbol {\xi} _ {\lfloor N t \rfloor}), \quad 0 \leq t \leq T. \tag {21}
$$

At any time  $t$ , the scaled empirical measure  $\mu_t^N$  is a random element, and thus  $(\mu_t^N)_{0 \leq t \leq T}$  is a measured-valued stochastic process. Therefore, we characterize the evolution of its Lebesgue density  $p_t^N(\pmb{\xi}) \stackrel{\mathrm{def}}{=} \mu_t^N(\mathrm{d}\pmb{\xi}) / \mathrm{d}\pmb{\xi}$  in the following theorem:

Theorem 4.2. (MCKEAN-VLASOV MEAN-FIELD PDE) Suppose conditions (A.1)-(A.3) of Appendix B are satisfied. Further, suppose that the Radon-Nikodym derivative  $q_{0}(\pmb{\xi}) = \mu_{0}(\mathrm{d}\pmb{\xi}) / \mathrm{d}\pmb{\xi}$  exists. Then, there exists a unique solution  $(p_t^* (\pmb {\xi}))_{0\leq t\leq T}$  to the following non-linear partial differential equation

$$
\left\{ \begin{array}{l l} \frac {\partial p _ {t} (\boldsymbol {\xi})}{\partial t} & = - \frac {\eta}{\beta} \iint_ {\mathcal {X} \times \mathcal {Y}} \left(\int_ {\mathbb {R} ^ {p}} \varphi (\boldsymbol {x}, \widetilde {\boldsymbol {\xi}}) \varphi (\widetilde {\boldsymbol {x}}, \widetilde {\boldsymbol {\xi}}) p _ {t} (\widetilde {\boldsymbol {\xi}}) \mathrm {d} \widetilde {\boldsymbol {\xi}} - \beta y \widetilde {y}\right) \nabla_ {\boldsymbol {\xi}} \left(p _ {t} (\boldsymbol {\xi}) \nabla_ {\boldsymbol {\xi}} \left(\varphi (\boldsymbol {x}; \boldsymbol {\xi}) \varphi (\widetilde {\boldsymbol {x}}; \boldsymbol {\xi})\right)\right) \mathrm {d} P _ {\boldsymbol {X}, Y} ^ {\otimes 2}, \\ p _ {0} (\boldsymbol {\xi}) & = q _ {0} (\boldsymbol {\xi}). \end{array} \right. \tag {22}
$$

Moreover, the measure-valued process  $\{(\mu_t^N)_{0\leq t\leq T}\}_{N\in \mathbb{N}}$  defined in equation 21 converges (weakly) to the unique solution  $\mu_t^* (\pmb {\xi}) = p_t^* (\pmb {\xi})\mathrm{d}\pmb{\xi}$  as the number of particles tend to infinity  $N\to \infty$

As a by product of the mean-field analysis of Theorem 4.2, we can prove that the empirical measure  $\widehat{\mu}_m^N$  of the particles in SGD dynamic equation 14 remains inside the feasible distribution ball  $\mathcal{P}_N$  defined with respect to the Radon distance

$$
d _ {\mathrm {R}} \left(\mu_ {1}, \mu_ {2}\right) = \sup  \left\{\int_ {\mathbb {R} ^ {D}} f (\boldsymbol {\xi}) \mathrm {d} \left(\mu_ {1} - \mu_ {2}\right) (\boldsymbol {\xi}): f \in \mathcal {F} _ {c} \right\}, \tag {23}
$$

where  $\mathcal{F}_c$  is the class of continuous functions  $f:\mathbb{R}^D\to [-1,1]\subset \mathbb{R}$

Corollary 4.2.1. Consider the learning rate  $\eta = \mathcal{O}\left(\frac{R}{T + T\sqrt{NT}\log(2 / \delta)}\right)$  for the SGD in equation 14.

Then, the empirical measure  $\widehat{\mu}_m^N$  of the particles remains inside the distributional ball  $\widehat{\mu}_N^m\in \mathcal{P}_N\stackrel {\text{def}}{=}$ $\mathbb{B}_R^N (\widehat{\mu}_0)\stackrel {\text{def}}{=} \{\widehat{\mu}^N\in \mathcal{M}(\mathbb{R}^D):d_{\mathrm{R}}(\widehat{\mu}^N,\widehat{\mu}_0)\leq R\}$  for all  $m\in [0,NT]\cap \mathbb{N}$ , with the probability of (at least)  $1 - \delta$ .

Let us make two remarks about the PDE in equation 22. First, the seminal works of Otto Otto (2001), and Jordan, et al. Jordan et al. (1998) establish a deep connection between the McKean-Vlasov type PDEs specified in equation 22 and the gradient flow on the Wasserstein manifolds. More specifically, the PDE equation in equation 22 can be thought of as the minimization of the energy functional

$$
\inf  _ {\mu \in \mathcal {M} (\mathbb {R} ^ {p})} E _ {\beta} \left(p _ {t} (\boldsymbol {\xi})\right) \stackrel {\text {d e f}} {=} \frac {1}{\beta} \int_ {\mathbb {R} ^ {p}} R _ {\beta} (\boldsymbol {\xi}, p _ {t} (\boldsymbol {\xi})) p _ {t} (\boldsymbol {\xi}) \mathrm {d} \boldsymbol {\xi} \tag {24a}
$$

$$
\left. \right. R _ {\beta} (\boldsymbol {\xi}, p _ {t} (\boldsymbol {\xi})) \stackrel {\text {d e f}} {=} - \beta \left(\mathbb {E} _ {P _ {\mathbf {X}}, Y} [ y \varphi (\boldsymbol {x}; \boldsymbol {\xi}) ]\right) ^ {2} + \mathbb {E} _ {\widetilde {\boldsymbol {\xi}} \sim p _ {t}} \left[\left(\mathbb {E} _ {P _ {\mathbf {X}}} [ \varphi (\boldsymbol {x}; \boldsymbol {\xi}) \varphi (\boldsymbol {x}; \widetilde {\boldsymbol {\xi}}) ]\right) ^ {2} \right], \tag {24b}
$$

using the following gradient flow dynamics

$$
\frac {\mathrm {d} p _ {t} (\boldsymbol {\xi})}{\mathrm {d} t} = - \eta \cdot \operatorname {g r a d} _ {p _ {t}} E _ {\beta} \left(p _ {t} (\boldsymbol {\xi})\right), \quad p _ {0} (\boldsymbol {\xi}) = q _ {0} (\boldsymbol {\xi}), \tag {25}
$$

where  $\mathrm{grad}_{p_t}E(p_t(\pmb {\xi})) = \nabla_{\pmb{\xi}}\cdot (p_t(\pmb {\xi})\nabla_{\pmb{\xi}}R_\beta (p_t(\pmb {\xi})))$  is the Riemannian gradient of  $R_{\beta}(\mu_t(\pmb {\xi}))$  with respect to the metric of the Wasserstein manifold. This shows that when the number of particles in particle SGD equation 14 tends to infinity  $(N\to +\infty)$ , their empirical distribution follows a gradient descent path for minimization of the population version (with respect to data samples) of the distributional risk optimization in equation 11. In this sense, the particle SGD is the 'consistent' approximation algorithm for solving the distributional optimization.

# 4.1 RELATED WORKS

The mean-field description of SGD dynamics has been studied in several prior works for different information processing tasks. Wang et al. Wang et al. (2017) consider the problem of online learning for the principal component analysis (PCA), and analyze the scaling limits of different online learning algorithms based on the notion of finite exchangeability. In their seminal papers, Montanari and co-authors Mei et al. (2018); Javanmard et al. (2019); Mei et al. (2019) consider the scaling limits of SGD for training a two-layer neural network, and characterize the related Mckean-Vlasov PDE for the limiting distribution of the empirical measure associated with the weights of the input layer. Similar mean-field type results for two-layer neural networks are also studied recently in Rotskoff & Vanden-Eijnden (2018); Sirignano & Spiliopoulos (2018). Our work is also related to the unpublished work of Wang, et al. Wang et al., which proposes a solvable model of GAN and analyzes the scaling limits. However, our GAN model is different from Wang et al. and is based on the notion of the kernel MMD. Our work is also closely related to the recent work of Li, et al Li et al. (2019) which proposes an implicit kernel learning method based on the following kernel definition

$$
K _ {h} (\iota (\boldsymbol {x}), \iota (\boldsymbol {y})) = \mathbb {E} _ {\boldsymbol {\xi} \sim \mu_ {0}} \left[ e ^ {(i h (\boldsymbol {\xi}) (\iota (\boldsymbol {x}) - \iota (\boldsymbol {y})))} \right], \tag {26}
$$

where  $\mu_0$  is a user defined base distribution, and  $h\in \mathcal{H}$  is a functions that transforms the base distribution  $\mu_0$  into a distribution  $\mu$  that provides a better kernel. Therefore, the work of Li, et al Li et al. (2019) implicitly optimizes the distribution of random features through a function. In contrast, the proposed distributional optimization framework in this paper optimizes the distribution of random feature explicitly, via optimizing their empirical measures. Perhaps more importantly from a practical perspective is the fact that our kernel learning approach does not require the user-defined function class  $\mathcal{H}$ . Moreover, our particle SGD method in equation 14 obviates tuning hyperparameters related to the implicit kernel learning method such as the gradient penalty factor and the variance constraint factor (denoted by  $\lambda_{GP}$  and  $\lambda_h$ , respectively, in Algorithm 1 of Li et al. (2019)).

# 5 EMPIRICAL EVALUATION

# 5.1 SYNTHETIC DATA-SET

Due to the space limitation, the experiments on the synthetic data are deferred to Appendix A.

# 5.2 PERFORMANCE ON BENCHMARK DATASETS

We evaluate our kernel learning approach on large-scale benchmark data-sets. We train our MMD GAN model on two distinct types of datasets, namely on MNIST LeCun et al. (1998) and CIFAR-10 LeCun et al. (1998), where the size of training instances are  $60 \times 10^{3}$  and  $50 \times 10^{3}$ , respectively. All the generated samples are from a fixed noise random vectors and are not singled out.

Implementation and hyper-parameters. We implement Algorithm 1 as well as MMD GAN Li et al. (2017) in Pytorch using NVIDIA Titan V100 32GB graphics processing units (GPUs). The source code of Algorithm 1 is built upon the code of Li et al. (2017), and retains the auto-encoder implementation. In particular, we use a sequential training of the auto-encoder and kernel as explained in the Synthetic data in Section A of Supplementary. For a fair comparison, our hyper-parameters are adjusted as in Li et al. (2017), i.e., the learning rate of 0.00005 is considered for RMSProp Tieleman & Hinton (2012). Moreover, the batch-size for training the generator and auto-encoder is  $n = 64$ . The learning rate of particle SGD is tuned to  $\eta = 10$ .

Random Feature Maps. To approximate the kernel, we use the explicit feature map of Rahimi and Recht Rahimi & Recht (2008; 2009), where  $\varphi(\boldsymbol{x};\boldsymbol{\xi}) = \sqrt{2}\cos(\boldsymbol{x}^T\boldsymbol{\xi} + b)$ . Here  $b \sim \mathrm{Uniform}\{-1, +1\}$  is a random bias term.

Practical considerations. When data-samples  $\{\pmb{V}_i\} \in \mathbb{R}^d$  are high dimensional (as in CIFAR-10), the particles  $\pmb{\xi}^{1},\dots ,\pmb{\xi}^{N}\in \mathbb{R}^{p},p = d$  in SGD equation 14 are also high-dimensional. To reduce the dimensionality of the particles, we apply an auto-encoder architecture similar to Li et al. (2017), and train our kernel on top of learned embedded features. More specifically, in our simulations, we train an auto-encoder where the dimensionality of the latent space is  $h = 10$  for MNIST, and  $h = 128$  (thus  $p = d = 128$ ) for CIFAR-10. Therefore, the particles  $\pmb{\xi}^{1},\dots ,\pmb{\xi}^{N}$  in subsequent kernel training phase have the dimension of  $p = 10$ , and  $p = 128$ , respectively.

Choice of the scaling parameter  $\beta$ . There is a trade-off in the choice of  $\beta$ . While for large values of  $\beta$ , the kernel is better able to separate data-samples from generated samples, in practice, a large value of  $\beta$  slows down the convergence of particle SGD. This is due to the fact that the coupling strength between the particles in equation 14 decreases as  $\beta$  increases. The scaling factor is set to be  $\beta = 1$  in all the following experiments.

Qualitative comparison. We now show that without the bandwidth tuning for Gaussian kernels and using the particle SGD to learn the kernel, we can attain better visual results on benchmark data-sets. In Figure 1, we show the generated samples on CIFAR-10 and MNIST data-sets, using our Algorithm 1, MMD GAN Li et al. (2017) with a mixed and homogeneous Gaussian RBF kernels, and GMMN Li et al. (2015).

Figure 1(a) shows the samples from Algorithm 1, Figure 1(b) shows the samples from MMD GAN Li et al. (2017) with a mixture RBF Gaussian kernel  $\kappa(\boldsymbol{x},\boldsymbol{y}) = \sum_{k=1}^{5}\kappa_{\sigma_k}(\boldsymbol{x},\boldsymbol{y})$ , where  $\sigma_k \in \{1,2,4,8,16\}$  are the bandwidths of the Gaussian kernels that are fine tuned and optimized. We observe that our MMD GAN with automatic kernel learning visually attains similar results to MMD GAN Li et al. (2017) which requires manual tuning of the hyper-parameters. In Figure 1(c), we show the MMD GAN result with a single kernel RBF Gaussian kernel whose bandwidth is manually tuned at  $\sigma = 16$ . Lastly, in Figure 1(d), we show the samples from GMMN Li et al. (2015) which does not exploit an auto-encoder or kernel training. Clearly, GMMN yield a poor results compared to other methods due to high dimensionality of features, as well as the lack of an efficient method to train the kernel.

On MNIST data-set in Figure 1,(e)-(h), the difference between our method and MMD GAN Li et al. (2017) is visually more pronounced. We observe that without a manual tuning of the kernel bandwidth and by using the particle SGD equation 14 to optimize the kernel, we attain better generated images in Figure 1(e), compared to MMD GAN with mixed RBF Gaussian kernel and manual bandwidth tuning in Figure 1(f). Moreover, using a single RBF Gaussian kernel yields a poor result regardless of the choice of its bandwidth. The generated images from GMMN is also shown in Figure 1(h).

Quantitative comparison. To quantitatively measure the quality and diversity of generated samples, we compute the inception score (IS) Salimans et al. (2016) as well as Fréchet Inception Distance (FID) Heusel et al. (2017) on CIFAR-10 images. Intuitively, the inception score is used for

![](images/c0fc4f902a2beb866159a1eaba149f9419f109ff8905bc40959ef649d4c31267.jpg)  
(a)

![](images/03a131b8866fc49b48e49b272d84610f364c631a76e100c09cbee5e2de0ec2dc.jpg)  
(b)

![](images/209916e4539d3619686e55835a6855c731ee40905a157810848108133f8cf40e.jpg)  
(c)

![](images/1d18eeff4f16d7c3de519cd7992f0086921293ab70f1e4d798c237c28f16e17b.jpg)  
(d)

![](images/4ab91ede2fbb6b79dc22b66df89f08e3146d5abc1a59c7708e91fc98f776cb4f.jpg)  
(e)

![](images/96c834823b4b1049a899afb2406717fa4230ef3f8612a1c75bee54635aec2c6c.jpg)  
(f)

![](images/6c66819c3365f25d0b5312efdaf31e5b67c4547a557aa367de9c545e49cbef28.jpg)  
(g)

![](images/b2d5739432195bb10d8c05c9fc45225329eb201b68fcc0d2d898bb8022fca0f9.jpg)  
(h)  
Figure 1: Sample generated images using CIFAR-10 (top row), and MNIST (bottom row) data-sets.Panels (a)-(e): Proposed MMD GAN with an automatic kernel selection via the particle SGD (Algorithm 1), Panels (b)-(f): MMD GAN Li et al. (2017) with an auto-encoder for dimensionality reduction in conjunction with a mixed RBF Gaussian kernel whose bandwidths are manually tuned, Panels (c)-(g): MMD GAN in Li et al. (2017) with a single RBF Gaussian kernel with an auto-encoder for dimensionality reduction in conjunction with a single RBF Gaussian kernel whose bandwidth is manually tuned, Panel (d)-(g): GMMN without an auto-encoder Li et al. (2015).

GANs to measure samples quality and diversity. Accordingly, for generative models that are collapsed into a single mode of distribution, the inception score is relatively low. The FID improves on IS by actually comparing the statistics of generated samples to real samples, instead of evaluating generated samples independently.

In Table 1, we report the quantitative measures for different MMD GAN model using different scoring metric. Note that in Table 1 lower FID scores and higher IS scores indicate a better performance. We observe from Table 1 that our approach attain lower FID score, and higher IS score compared to MMD GAN with single Gaussian kernel (bandwidth  $\sigma = 16$ ), and a mixture Gaussian kernel (bandwidths  $\{1,2,4,8,16\}$ ).

<table><tr><td>Method</td><td>FID (↓)</td><td>IS (↑)</td></tr><tr><td>MMD GAN (Gaussian) Li et al. (2017)</td><td>67.244 ± 0.134</td><td>5.608±0.051</td></tr><tr><td>MMD GAN (Mixture Gaussian) Li et al. (2017)</td><td>67.129 ± 0.148</td><td>5.850±0.055</td></tr><tr><td>SGD Alg. 1</td><td>65.059 ± 0.153</td><td>5.97 ± 0.046</td></tr><tr><td>Benchmark</td><td>-</td><td>11.237±0.116</td></tr></table>

Table 1: Comparison of the quantitative performance measures of MMD GANs with different kernel learning approaches.

# REFERENCES

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein GAN. arXiv preprint arXiv:1701.07875, 2017.  
Patrick Billingsley. Convergence of probability measures. John Wiley & Sons, 2013.  
Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration inequalities: A nonasymptotic theory of independence. Oxford university press, 2013.  
Tong Che, Yanran Li, Athul Paul Jacob, Yoshua Bengio, and Wenjie Li. Mode regularized generative adversarial networks. arXiv preprint arXiv:1612.02136, 2016.  
Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. Journal of Machine Learning Research, 13(Mar):795-828, 2012.  
Joseph Leo Doob. Stochastic processes, volume 101. New York Wiley, 1953.  
Rui Gao and Anton J Kleywegt. Distributionally robust stochastic optimization with Wasserstein distance. arXiv preprint arXiv:1604.02199, 2016.  
Kay Giesecke, Konstantinos Spiliopoulos, Richard B Sowers, et al. Default clustering in large portfolios: Typical events. The Annals of Applied Probability, 23(1):348-385, 2013.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Arthur Gretton, Karsten Borgwardt, Malte Rasch, Bernhard Scholkopf, and Alex J Smola. A kernel method for the two-sample-problem. In Advances in neural information processing systems, pp. 513-520, 2007.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, pp. 6626-6637, 2017.  
Adam Jakubowski. On the skorokhod topology. In Annales de l'IHP Probabilités et statistiques, volume 22, pp. 263-285, 1986.  
Adel Javanmard, Marco Mondelli, and Andrea Montanari. Analysis of a two-layer neural network via displacement convexity. arXiv preprint arXiv:1901.01375, 2019.  
Richard Jordan, David Kinderlehrer, and Felix Otto. The variational formulation of the Fokker-Planck equation. SIAM journal on mathematical analysis, 29(1):1-17, 1998.  
Masoud Badiei Khuzani and Na Li. Stochastic primal-dual method on riemannian manifolds of bounded sectional curvature. In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA), pp. 133-140. IEEE, 2017.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.  
Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Chun-Liang Li, Wei-Cheng Chang, Yu Cheng, Yiming Yang, and Barnabás Póczos. Mmd GAN: Towards deeper understanding of moment matching network. In Advances in Neural Information Processing Systems, pp. 2203-2213, 2017.

Chun-Liang Li, Wei-Cheng Chang, Youssef Mroueh, Yiming Yang, and Barnabás Póczos. Implicit kernel learning. arXiv preprint arXiv:1902.10214, 2019.  
Yujia Li, Kevin Swersky, and Rich Zemel. Generative moment matching networks. In International Conference on Machine Learning, pp. 1718-1727, 2015.  
Shishi Luo and Jonathan C Mattingly. Scaling limits of a model for selection at two scales. Nonlinearity, 30(4):1682, 2017.  
Colin McDiarmid. On the method of bounded differences. Surveys in combinatorics, 141(1):148-188, 1989.  
Song Mei, Andreaa Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences, 115(33):E7665-E7671, 2018.  
Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Mean-field theory of two-layers neural networks: dimension-free bounds and kernel limit. arXiv preprint arXiv:1902.06015, 2019.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka.  $f$ -GAN: Training generative neural samplers using variational divergence minimization. In Advances in neural information processing systems, pp. 271-279, 2016.  
Felix Otto. The geometry of dissipative evolution equations [poiu]: the porous medium equation. 2001.  
Neal Parikh, Stephen Boyd, et al. Proximal algorithms. Foundations and Trends® in Optimization, 1(3):127-239, 2014.  
David Pollard. Empirical processes: theory and applications. In NSF-CBMS regional conference series in probability and statistics, pp. i-86. JSTOR, 1990.  
Yu V Prokhorov. Convergence of random processes and limit theorems in probability theory. Theory of Probability & Its Applications, 1(2):157-214, 1956.  
Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in neural information processing systems, pp. 1177-1184, 2008.  
Ali Rahimi and Benjamin Recht. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. In Advances in neural information processing systems, pp. 1313-1320, 2009.  
Philippe Robert. Stochastic networks and queues, volume 52. Springer Science & Business Media, 2013.  
Grant M Rotskoff and Eric Vanden-Eijnden. Neural networks as interacting particle systems: Asymptotic convexity of the loss landscape and universal scaling of the approximation error. arXiv preprint arXiv:1805.00915, 2018.  
W Rudin. Real and complex analysis mcraw-hill book co. New York 3rd ed., xiv, 1987.  
Ruslan Salakhutdinov and Geoffrey Hinton. Deep boltzmann machines. In Artificial intelligence and statistics, pp. 448-455, 2009.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In Advances in neural information processing systems, pp. 2234-2242, 2016.

Aman Sinha and John C Duchi. Learning kernels with random features. In Advances in Neural Information Processing Systems, pp. 1298-1306, 2016.  
Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of neural networks. arXiv preprint arXiv:1805.01053, 2018.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2):26-31, 2012.  
VS Varadarajan. On the theorem of friesz concerning the form of linear functional. 1958.  
Chuang Wang, Hong Hu, and Yue M Lu. A solvable high-dimensional model of GAN.  
Chuang Wang, Jonathan Mattingly, and Yue Lu. Scaling limit: Exact and tractable analysis of online learning algorithms with applications to regularized regression and PCA. arXiv preprint arXiv:1712.04332, 2017.  
Jeff Webb. Extensions of Gronwall's inequality with quadratic growth terms and applications. *Electronic Journal of Qualitative Theory of Differential Equations*, 2018(61):1-12, 2018.  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In International conference on machine learning, pp. 2048-2057, 2015.
