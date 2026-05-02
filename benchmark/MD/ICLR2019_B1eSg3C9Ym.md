# MEAN-FIELD ANALYSIS OF BATCH NORMALIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Batch Normalization (BatchNorm) is an extremely useful component of modern neural network architectures, enabling optimization using higher learning rates and achieving faster convergence. In this paper, we use mean-field theory to analytically quantify the impact of BatchNorm on the geometry of the loss landscape for multi-layer networks consisting of fully-connected and convolutional layers. We show that it has a flattening effect on the loss landscape, as quantified by the maximum eigenvalue of the Fisher Information Matrix. These findings are then used to justify the use of larger learning rates for networks that use BatchNorm, and we provide quantitative characterization of the maximal allowable learning rate to ensure convergence. Experiments support our theoretically predicted maximum learning rate, and furthermore suggest that networks with smaller values of the BatchNorm parameter  $\gamma$  achieve lower loss after the same number of epochs of training.

# 1 INTRODUCTION

Deep neural networks have achieved remarkable success in the past decade on tasks that were out of reach prior to the era of deep learning (Krizhevsky et al., 2012; He et al., 2016). Amongst the myriad reasons for these successes are powerful computational resources, large datasets, new optimization algorithms, and modern architecture designs (Russakovsky et al., 2015; Kingma & Ba, 2015). In many modern deep learning architectures, one key component is batch normalization (BatchNorm). BatchNorm is a module that can be introduced in layers of deep neural networks that normalizes hidden layer outputs to have a common first and second moment. Empirically, BatchNorm enables optimization using much larger learning rates, and achieves better convergence (Ioffe & Szegedy, 2015).

Despite significant practical success, a theoretical understanding of BatchNorm is still lacking. A widely held view is that BatchNorm improves training by "reducing of internal covariate shift" (ICF) (Ioffe & Szegedy, 2015). Internal covariate shift refers to the change in the input distribution of internal layers of the deep network due to changes of the weights. Recent results (Santurkar et al., 2018), however, cast doubt on the ICF explanation, by demonstrating that noisy BatchNorm increases ICF yet still improves training as in regular BatchNorm. This raises the question of whether the utility of BatchNorm is indeed related to the reduction of ICF. Instead, it is argued by Santurkar et al. (2018) that BatchNorm actually improves the Lipschitzness of the loss and gradient.

Meanwhile, dynamical mean-field theory (Sompolinsky & Zippelius, 1982), a powerful theoretical technique, has recently been applied by Poole et al. (2016) to ensembles of multi-layer random neural networks. This theory studies networks with an i.i.d. Gaussian distribution of weights and biases. Most recent work focuses on the analysis of order parameter flows and their fixed points (Schoenholz et al., 2017; Xiao et al., 2018; Yang & Schoenholz, 2017), including their stability and decay rates. Importantly, Karakida et al. (2018) also successfully used mean-field analysis to estimate the spectral properties of the Fisher Information Matrix.

In this paper, we analytically quantify the impact of BatchNorm on the landscape of the loss function, by using mean-field theory to estimate the spectral properties of the Fisher Information Matrix (FIM) for typical batch-normalized neural networks. In particular, it is shown that BatchNorm reduces the maximal eigenvalue of the FIM provided that the normalization coefficient  $\gamma$  is not too large. By drawing on results linking Fisher Information to the geometry of the loss function, we explain how BatchNorm neural networks can be trained with a larger learning rate without leading to parameter

explosion, and provide upper bounds on the learning rate in terms of the BatchNorm parameters. As an additional contribution motivated by our theoretical findings, we demonstrate an empirical correlation between the BatchNorm parameter  $\gamma$  and test loss. In particular, networks with smaller  $\gamma$  achieve lower loss after a fixed number of training epochs.

# 2 PRELIMINARIES

In our theoretical analysis, we employ the recent application of mean-field theory to neural networks which studies an ensemble of random neural networks with pre-defined i.i.d. Gaussian weights and biases. In this section, we provide background information and briefly recall the formalism of Karakida et al. (2018) which first computes spectral properties of the Fisher Information of a neural network and then relates it to the maximal stable learning rate.

# 2.1 FISHER INFORMATION MATRIX AND LEARNING DYNAMICS

Given a data distribution  $\mathcal{D}$  over the set of instance-label pairs  $\mathcal{X} \times \mathcal{Y}$ , a family of parametrized functions  $f_{\theta}: \mathcal{X} \to \mathcal{Y}$  and a loss function  $l(f, y)$ , our focus will be to ensure convergence of the following gradient descent with momentum update rule:

$$
\theta_ {t + 1} = \theta_ {t} - \eta \nabla L (\theta_ {t}) + \mu \left(\theta_ {t} - \theta_ {t - 1}\right), \tag {1}
$$

where  $L(\theta)$  is the unobserved population loss,

$$
L (\theta) := \mathbb {E} _ {(x, y) \sim \mathcal {D}} [ l (f _ {\theta} (x), y) ]. \tag {2}
$$

In practice, the parameters are determined by minimizing an empirical estimate of equation 2 using a stochastic generalization (SGD) of the update rule equation 1. We neglect this difference by always working in the asymptotic limit of large sample size and moreover assuming full-batch gradient updates.

Suppose the loss function can be expressed in terms of a parametric family of positive densities as  $l(f_{\theta}(x),y) \eqqcolon -\log p_{\theta}(x,y)$ . This assumption holds true for a large class of losses including squared loss and cross-entropy loss. Let  $I_{\theta}$  denote the Fisher Information Matrix (FIM) associated with the parametric family induced by the loss,

$$
I _ {\theta} := \mathbb {E} _ {(x, y) \sim \mathbb {P} _ {\theta}} \left[ \nabla_ {\theta} \log p _ {\theta} (x, y) \otimes \nabla_ {\theta} \log p _ {\theta} (x, y) \right]. \tag {3}
$$

Recall that under suitable regularity conditions the following identity holds:

$$
I _ {\theta} = - \mathbb {E} _ {(x, y) \sim \mathbb {P} _ {\theta}} \left[ \underset {\theta} {\operatorname {H e s s}} \log p _ {\theta} (x, y) \right]. \tag {4}
$$

The above right-hand side is closely related to the Hessian of the population loss,

$$
\operatorname {H e s s} (L (\theta)) = - \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \underset {\theta} {\operatorname {H e s s}} \log p _ {\theta} (x, y) \right] \tag {5}
$$

where we interchanged the Hessian with the expectation value. In fact, if we assume that the estimation problem is well-specified so that there exist parameters  $\theta_{*}$  such that the data distribution is generated by  $\mathbb{P}_{\theta_*} = \mathcal{D}$ , then we obtain the following equality between the Hessian of the population loss and the FIM evaluated at the optimal parameters,

$$
\operatorname {H e s s} \left(L \left(\theta_ {*}\right)\right) = I _ {\theta_ {*}}. \tag {6}
$$

If  $\theta$  is initialized in a sufficiently small neighborhood of  $\theta_{*}$ , then by expanding the population loss  $L(\theta)$  to quadratic order about  $\theta_{*}$  one can show that a necessary condition for convergence is that the step size is bounded from above by (LeCun et al., 2012; Karakida et al., 2018)<sup>1</sup>,

$$
\eta <   \eta_ {*} := \frac {2 (1 + \mu)}{\lambda_ {\max } \left(\operatorname {H e s s} \left(L \left(\theta_ {*}\right)\right)\right)} = \frac {2 (1 + \mu)}{\lambda_ {\max } \left(I _ {\theta_ {*}}\right)}, \tag {7}
$$

where  $\lambda_{\mathrm{max}}(M)$  denotes the largest eigenvalue of the matrix  $M$ . Rather than computing the optimal parameters  $\theta_*$  directly, we follow the strategy of Karakida et al. (2018) by estimating the following

quantity and arguing that the distribution of the weights and biases is not significantly impacted by the training dynamics,

$$
\bar {\lambda} _ {\max } := \lambda_ {\max } \left(\mathbb {E} _ {\theta} \left[ I _ {\theta} \right]\right), \tag {8}
$$

where  $\mathbb{E}_{\theta}$  denotes the expectation value with respect to the weights and biases. This heuristic was shown to yield a remarkably accurate prediction of the maximal learning rate in (Karakida et al., 2018).

In this paper we adopt the data modeling assumption that  $\mathbb{P}_{\theta} = \mathbb{P}_{\mathcal{X}}\otimes \mathbb{P}_{\mathcal{Y}\mid \mathcal{X}}^{\theta}$  where  $\mathbb{P}_{\mathcal{X}}$  is the marginal distribution of the covariates, which is independent of  $\theta$ . Under this factorization assumption, the FIM simplifies to

$$
I _ {\theta} = \mathbb {E} _ {(x, y) \sim \mathbb {P} _ {\theta}} \left[ \nabla \log p _ {\theta} (y \mid x) \otimes \nabla_ {\theta} \log p _ {\theta} (y \mid x) \right]. \tag {9}
$$

Focusing on the Gaussian conditional model  $p_{\theta}(y|x) \propto \exp \left(\frac{1}{2}\| f_{\theta}(x) - y\| _2^2\right)$ , the FIM further simplifies to

$$
I _ {\theta} = \mathbb {E} _ {x \sim \mathcal {D}} \left[ \nabla_ {\theta} f _ {\theta} (x) \otimes \nabla_ {\theta} f _ {\theta} (x) \right]. \tag {10}
$$

The family of parametrized functions  $f_{\theta} : \mathbb{R}^{N_0} \to \mathbb{R}^{N_L}$  is chosen to be the family of functions computed by a multi-layer neural network architecture with  $N_0$  input nodes,  $N_L$  output nodes and  $L \geq 1$  layers. In this paper, we consider neural networks consisting of fully-connected (FC) and convolutional (Conv) layers, with and without batch normalization. The pointwise activation is denoted by  $\sigma$ , which is taken to be the rectified linear unit (ReLU) in this paper. Our analysis can be straightforwardly extended to other architectures and non-linearities. We use  $h_{\theta}^{l}(x)$  to denote the output of layer  $l$  and the input to layer  $l + 1$ . Clearly we have  $h_{\theta}^{0}(x) = x$  and  $h_{\theta}^{L}(x) = f_{\theta}(x)$ .

# 3 THEORY

In this section we focus on applying dynamical mean-field theory to study the effect of introducing batch normalization modules into a deep neural network by estimating the largest eigenvalue of the FIM. This estimate, in turn, provides an upper bound on the largest learning rate for which the learning dynamics is stable. This section is structured as follows: We first define various thermodynamic quantities (order parameters, 6 for fully-connected layers and 9 for convolutional layers) that satisfy recursion relations in the mean-field approximation. Then we present an estimate of  $\bar{\lambda}_{\mathrm{max}}$  in terms of these order parameters, generalizing a result of Karakida et al. (2018). Using this estimate, we study how  $\bar{\lambda}_{\mathrm{max}}$  and  $\eta_{*}$  are affected by BatchNorm and calculate their dependence on the Batch-Norm coefficient  $\gamma$ . Detailed derivations of the order parameters, their recursions, and the associated eigenvalue bound are deferred to the Supplementary Material.

# 3.1 FULLY CONNECTED LAYERS

# 3.1.1 STANDARD CASE

A general fully connected layer with input activation  $h^l (x)$  and output pre-activation  $z^{l + 1}(x)$  is described by the affine transformation,

$$
z ^ {l + 1} (x) := W ^ {l + 1} h ^ {l} (x) + b ^ {l + 1}, \tag {11}
$$

where  $W^{l + 1} \in \mathbb{R}^{N_{l + 1} \times N_l}$ ,  $b^{l + 1} \in \mathbb{R}^{N_{l + 1}}$  and  $N_{l}$  denotes the number of units in layer  $l$ . In the framework of mean-field theory, we will consider an ensemble of neural networks with Gaussian random weights and biases distributed as follows,

$$
\left[ W ^ {l + 1} \right] _ {i j} \sim N \left(0, \sigma_ {\mathrm {w}} ^ {2} / N _ {l}\right), \quad b _ {i} ^ {l + 1} \sim N \left(0, \sigma_ {\mathrm {b}} ^ {2}\right). \tag {12}
$$

In the case of a standard fully connected layer, the input activation satisfies the recursions  $h^l(x) = \sigma(z^l(x))$ , where  $\sigma$  denotes the pointwise activation.

A batch-normalized fully connected layer, in contrast, satisfies the following recursion,

$$
h ^ {l} (x) := \sigma \left(\frac {z ^ {l} (x) - \mu^ {l}}{s ^ {l}} \odot \gamma_ {l} + \beta_ {l}\right), \tag {13}
$$

where  $\mu^l\in \mathbb{R}^{N_l}$  and  $(s^l)^2\coloneqq s^l\odot s^l\in \mathbb{R}^{N_l}$  denote the mean and variance of the pre-activation layers with respect to the data distribution,

$$
\mu^ {l} := \underset {x} {\mathbb {E}} \left[ z ^ {l} (x) \right], \tag {14}
$$

$$
(s ^ {l}) ^ {2} := \underset {x} {\mathbb {E}} \left[ \left(z ^ {l} (x) - \mu^ {l}\right) ^ {2} \right]. \tag {15}
$$

The weights and biases are drawn from the same distributions as in the standard, no BatchNorm, case. In addition, we now have the BatchNorm parameters  $\gamma^{l + 1},\beta^{l + 1}\in \mathbb{R}^{N_{l + 1}}$  which are assumed to be non-random for simplicity,

$$
\gamma_ {l} [ i ] = \gamma_ {l}, \quad \beta_ {l} [ i ] = 0. \tag {16}
$$

# 3.1.2 ORDER PARAMETERS AND THEIR RECURSIONS

To investigate the spectral properties of the FIM, we define the following order parameters,

$$
q ^ {l} := \frac {1}{N _ {l}} \underset {x, \theta} {\mathbb {E}} \left[ \| z ^ {l} (x) \| ^ {2} \right], \quad \hat {q} ^ {l} := \frac {1}{N _ {l}} \underset {x, \theta} {\mathbb {E}} \left[ \| h ^ {l} (x) \| ^ {2} \right], \tag {17}
$$

$$
q _ {x y} ^ {l} := \frac {1}{N _ {l}} \underset {x, y, \theta} {\mathbb {E}} \left[ \langle z ^ {l} (x), z ^ {l} (y) \rangle \right], \quad \hat {q} _ {x y} ^ {l} := \frac {1}{N _ {l}} \underset {x, y, \theta} {\mathbb {E}} \left[ \langle h ^ {l} (x), h ^ {l} (y) \rangle \right], \tag {18}
$$

$$
\tilde {q} ^ {l} := \underset {x, \theta} {\mathbb {E}} \left[ \| \delta^ {l} (x) \| ^ {2} \right], \quad \tilde {q} _ {x y} ^ {l} := \underset {x, y, \theta} {\mathbb {E}} \left[ \langle \delta^ {l} (x), \delta^ {l} (y) \rangle \right], \tag {19}
$$

where  $\delta^l (x)\coloneqq \frac{\partial f_\theta}{\partial z^l} (x)$ . Here we assume that the data  $x$  are drawn i.i.d. from a distribution with mean 0 and variance 1, and also that the last layer is linear for classification. We then have the base cases:  $\hat{q}^{0} = 0$ ,  $\hat{q}_{xy}^{0} = 1$ ,  $\tilde{q}^{L} = \tilde{q}_{xy}^{L} = 1$ . The order parameters in the absence of BatchNorm satisfy the following recursions derived in Karakida et al. (2018),

$$
q ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} ^ {l - 1}, \quad \hat {q} ^ {l} = \frac {q ^ {l}}{2}, \tag {20}
$$

$$
q _ {x y} ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {x y} ^ {l - 1}, \quad \hat {q} _ {x y} ^ {l} = \frac {q ^ {l}}{2 \pi} \left(\sqrt {1 - c _ {x y} ^ {2}} + \frac {c _ {x y} \pi}{2} + c _ {x y} \sin^ {- 1} c _ {x y}\right), \tag {21}
$$

$$
\tilde {q} ^ {l} = \frac {\sigma_ {\mathrm {w}} ^ {2}}{2} \tilde {q} ^ {l + 1}, \quad \tilde {q} _ {x y} ^ {l} = \frac {\sigma_ {\mathrm {w}} ^ {2} \tilde {q} _ {x y} ^ {l + 1}}{2 \pi} \left(\frac {\pi}{2} + \sin^ {- 1} c _ {x y}\right), \tag {22}
$$

where  $c_{xy} \coloneqq q_{xy}^l / q^l$ . In the case of batch normalization we find the following recursions, which are derived in the Supplementary Material,

$$
q ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} ^ {l - 1}, \quad \hat {q} ^ {l} = \frac {\gamma_ {l} ^ {2}}{2}, \tag {23}
$$

$$
q _ {x y} ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {x y} ^ {l - 1}, \quad \hat {q} _ {x y} ^ {l} = \frac {\gamma l ^ {2}}{2 \pi}, \tag {24}
$$

$$
\tilde {q} ^ {l} = \frac {\gamma_ {l} ^ {2} \sigma_ {\mathrm {w}} ^ {2}}{2} \frac {\tilde {q} ^ {l + 1}}{q ^ {l}}, \quad \tilde {q} _ {x y} ^ {l} = \frac {\gamma_ {l} ^ {2} \sigma_ {\mathrm {w}} ^ {2}}{4} \frac {\tilde {q} _ {x y} ^ {l + 1}}{q ^ {l}}. \tag {25}
$$

# 3.2 CONVOLUTIONAL LAYER

The results of the preceding section also apply to structured affine transformations including convolutional layers. Let  $\mathcal{K}_l$  denote the set of allowable spatial locations of the  $l$ th layer feature map and let  $\mathcal{F}_{l + 1}$  index the sites of the convolutional kernel applied to that layer. Let  $C_l$  denote the number of input channels. The output of a general convolutional layer is of the form,

$$
z _ {\alpha} ^ {l + 1} (x) = \sum_ {\beta \in \mathcal {F} _ {l + 1}} W _ {\beta} ^ {l + 1} h _ {\alpha + \beta} ^ {l} (x) + b ^ {l + 1}, \tag {26}
$$

where  $\alpha \in \mathcal{K}_{l + 1}$ ,  $W_{\beta}^{l + 1} \in \mathbb{R}^{C_{l + 1} \times C_l}$  and  $b^{l + 1} \in \mathbb{R}^{C_{l + 1}}$ . The weights and biases are now distributed as

$$
\left[ W _ {\alpha} ^ {l + 1} \right] _ {i j} \sim N \left(0, \sigma_ {\mathrm {w}} ^ {2} / N _ {l}\right), \quad b _ {i} ^ {l + 1} \sim N \left(0, \sigma_ {\mathrm {b}} ^ {2}\right). \tag {27}
$$

where now  $N_{l} \coloneqq C_{l}|\mathcal{F}_{l + 1}|$ . As in the fully connected case, we consider convolutional layers with both vanilla activation functions of the form  $h_{\alpha}^{l}(x) \coloneqq \sigma (z_{\alpha}^{l}(x))$  as well as batch normalized convolutional layers, for which the input activations satisfy the recursive identity,

$$
h _ {\alpha} ^ {l} (x) := \sigma \left(\frac {z _ {\alpha} ^ {l} (x) - \mu_ {\alpha} ^ {l}}{s _ {\alpha} ^ {l}} \odot \gamma_ {l} + \beta_ {l}\right), \tag {28}
$$

# 3.2.1 ORDER PARAMETERS AND THEIR RECURSIONS

Similar to the definitions for fully connected layer, we define the following set of order parameters:

$$
q ^ {l} := \frac {1}{C _ {l}} \underset {\alpha x, \theta} {\mathbb {E}} \underset {\alpha x, \theta} {\mathbb {E}} \left[ \| z _ {\alpha} ^ {l} (x) \| ^ {2} \right], \quad \hat {q} ^ {l} := \frac {1}{C _ {l}} \underset {\alpha x, \theta} {\mathbb {E}} \underset {\alpha x, \theta} {\mathbb {E}} \left[ \| h _ {\alpha} ^ {l} (x) \| ^ {2} \right], \tag {29}
$$

$$
q _ {x y} ^ {l} := \frac {1}{C _ {l}} \underset {\alpha} {\mathbb {E}} \underset {x, y, \theta} {\mathbb {E}} \left[ \langle z _ {\alpha} ^ {l} (x), z _ {\alpha} ^ {l} (y) \rangle \right], \quad \hat {q} _ {x y} ^ {l} := \frac {1}{C _ {l}} \underset {\alpha} {\mathbb {E}} \underset {x, y, \theta} {\mathbb {E}} \left[ \langle h _ {\alpha} ^ {l} (x), h _ {\alpha} ^ {l} (y) \rangle \right], \tag {30}
$$

$$
q _ {\alpha \beta , x y} ^ {l} := \frac {1}{C _ {l}} \underset {\alpha \neq \beta} {\mathbb {E}} \left[ \underset {x, y, \theta} {\mathbb {E}} \langle z _ {\alpha} ^ {l} (x), z _ {\beta} ^ {l} (y) \rangle \right], \quad \hat {q} _ {\alpha \beta , x y} ^ {l} := \frac {1}{C _ {l}} \underset {\alpha \neq \beta} {\mathbb {E}} \left[ \underset {x, y, \theta} {\mathbb {E}} \langle h _ {\alpha} ^ {l} (x), h _ {\beta} ^ {l} (y) \rangle \right], \tag {31}
$$

$$
\tilde {q} ^ {l} := \underset {\alpha} {\mathbb {E}} \underset {x, \theta} {\mathbb {E}} \left[ \left\| \delta_ {\alpha} ^ {l} (x) \right\| ^ {2} \right], \quad \tilde {q} _ {x y} ^ {l} := \underset {\alpha} {\mathbb {E}} \left[ \underset {x, y, \theta} {\mathbb {E}} \left\langle \delta_ {\alpha} ^ {l} (x), \delta_ {\alpha} ^ {l} (y) \right\rangle \right], \tag {32}
$$

$$
\tilde {q} _ {\alpha \beta , x y} ^ {l} := \underset {\alpha \neq \beta} {\mathbb {E}} \left[ \underset {x, y, \theta} {\mathbb {E}} \left\langle \delta_ {\alpha} ^ {l} (x), \delta_ {\beta} ^ {l} (y) \right\rangle \right], \tag {33}
$$

where now  $\delta_{\alpha}^{l} \coloneqq \frac{\partial f_{\theta}}{\partial z_{\alpha}^{l}}$  in analogy with the fully connected layer. The expectations over  $\alpha$  and  $\beta$  are with respect to the uniform measure over the set of allowed indices. For a standard convolutional layer without BatchNorm, the order parameters can be shown to satisfy the following recursion relations:

$$
q ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {l - 1}, \quad \hat {q} ^ {l} = \frac {q l}{2}, \tag {34}
$$

$$
q _ {x y} ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {x y} ^ {l - 1}, \quad \hat {q} _ {x y} ^ {l} = \frac {q ^ {l}}{2 \pi} \left(\sqrt {1 - c _ {x y} ^ {2}} + \frac {c _ {x y} \pi}{2} + c _ {x y} \sin^ {- 1} c _ {x y}\right), \tag {35}
$$

$$
q _ {\alpha \beta , x y} ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {\alpha \beta , x y} ^ {l - 1}, \quad \hat {q} _ {\alpha \beta , x y} ^ {l} = \frac {q ^ {l}}{2 \pi} \left(\sqrt {1 - c _ {\alpha \beta} ^ {2}} + \frac {c _ {\alpha \beta} \pi}{2} + c _ {\alpha \beta} \sin^ {- 1} c _ {\alpha \beta}\right), \tag {36}
$$

$$
\tilde {q} ^ {l} = \frac {\sigma_ {\mathrm {w}} ^ {2}}{2} \tilde {q} ^ {l + 1}, \quad \tilde {q} _ {x y} ^ {l} = \frac {\sigma_ {\mathrm {w}} ^ {2} \tilde {q} _ {x y} ^ {l + 1}}{2 \pi} \left(\frac {\pi}{2} + \sin^ {- 1} c _ {x y}\right), \tag {37}
$$

$$
\tilde {q} _ {\alpha \beta , x y} ^ {l} = \frac {\sigma_ {\mathrm {w}} ^ {2} \tilde {q} _ {\alpha \beta , x y} ^ {l + 1}}{2 \pi} \left(\frac {\pi}{2} + \sin^ {- 1} c _ {\alpha \beta}\right). \tag {38}
$$

In the case of convolutional layers with BatchNorm, the following recursions hold:

$$
q ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {l - 1}, \quad \hat {q} ^ {l} = \frac {\gamma_ {l} ^ {2}}{2}, \tag {39}
$$

$$
q _ {x y} ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {x y} ^ {l - 1}, \quad \hat {q} _ {x y} ^ {l} = \frac {\gamma_ {l} ^ {2}}{2 \pi}, \tag {40}
$$

$$
q _ {\alpha \beta , x y} ^ {l} = \sigma_ {\mathrm {b}} ^ {2} + \sigma_ {\mathrm {w}} ^ {2} \hat {q} _ {\alpha \beta , x y} ^ {l - 1}, \quad \hat {q} _ {\alpha \beta , x y} ^ {l} = \frac {\gamma_ {l} ^ {2}}{2 \pi}, \tag {41}
$$

$$
\tilde {q} ^ {l} = \frac {\gamma_ {l} ^ {2} \sigma_ {\mathrm {w}} ^ {2}}{2} \frac {\tilde {q} ^ {l + 1}}{q ^ {l}}, \quad \tilde {q} _ {x y} ^ {l} = \frac {\gamma_ {l} ^ {2} \sigma_ {\mathrm {w}} ^ {2}}{4} \frac {\tilde {q} _ {x y} ^ {l + 1}}{q ^ {l}}, \tag {42}
$$

$$
\tilde {q} _ {\alpha \beta , x y} ^ {l} = \frac {\gamma_ {l} ^ {2} \sigma_ {\mathrm {w}} ^ {2}}{4} \frac {\tilde {q} _ {\alpha \beta , x y} ^ {l + 1}}{q ^ {l}}, \tag {43}
$$

where  $c_{xy} \coloneqq q_{xy}^l / q^l$  and  $c_{\alpha \beta} \coloneqq q_{\alpha \beta, xy}^l / q^l$ . The derivations of the recursion relations for both vanilla and batch-normalized convolutional layers are deferred to the Supplementary Material.

# 3.3 EIGENVALUE BOUND AND THERMODYNAMIC VARIABLES

The order parameters derived in the previous section are useful because they allow us to gain information about the maximal eigenvalue  $\bar{\lambda}_{\mathrm{max}}$  of the FIM. We derived a generalization of (Karakida et al., 2018, Theorem 6) to allow for the inclusion of batch normalization and convolutional layers. In particular, we obtain a lower bound on the maximal eigenvalue  $\bar{\lambda}_{\mathrm{max}}$  in terms of the previously introduced order parameters which satisfy the stated recursion relations in the mean-field approximation.

Claim 3.1. If the layer dimension  $N_{l}$  of the fully connected layers and the number of channels  $C_{l}$  of the convolutional layers satisfy  $N_{l} \gg 1$  and  $C_{l} \gg 1$  for  $0 < l < L$  and the number of samples  $n$  in the training dataset satisfies  $n \gg 1$ , we have,

$$
\bar {\lambda} _ {\max } \geq \sum_ {l \in [ L ]} f _ {l}, \tag {44}
$$

where  $f_{l} = N_{l - 1}\tilde{q}_{xy}^{l - 1}\tilde{q}_{xy}^{l}$  for fully connected layers and

$$
f _ {l} = N _ {l - 1} \left[ \left(\left| \mathcal {K} _ {l} \right| - 1\right) \tilde {q} _ {\alpha \beta , x y} ^ {l} + \tilde {q} _ {x y} ^ {l} \right] \left[ \left(\left| \mathcal {K} _ {l} \right| - 1\right) \hat {q} _ {\alpha \beta , x y} ^ {l - 1} + \hat {q} _ {x y} ^ {l - 1} \right], \tag {45}
$$

for convolutional layers, where recall  $N_{l-1} = C_{l-1}|\mathcal{F}_l|$ . The index sets  $\mathcal{F}_l$  and  $\mathcal{K}_l$  are defined in section 3.2. The order parameters are defined in the previous subsection.

Now we are ready to calculate the lower bound on  $\bar{\lambda}_{\mathrm{max}}$  for a given model architecture by calculating the order parameters using their recursions. In the next section, we will focus on the numerical analysis of these recursion relations as well as present experiments that support our calculation.

# 4 NUMERICAL ANALYSIS AND EXPERIMENTS

In order to understand the effect of BatchNorm on the loss landscape, we numerically compute  $\bar{\lambda}_{\mathrm{max}}$  as a function of the BatchNorm parameter  $\gamma$ , for both fully connected and convolutional architectures (Fig. 1) with and without BatchNorm. For  $\gamma \lesssim 3$  (typical for deep network initialization (Ioffe & Szegedy, 2015)) BatchNorm significantly reduces  $\bar{\lambda}_{\mathrm{max}}$  compared to the vanilla networks. As a direct consequence of this, the theory predicts that batch normalized networks can be trained using significantly higher learning rates than their vanilla counterparts.

We tested the above theoretical prediction by training the same architectures on MNIST and CIFAR-10 datasets, for different values of  $\eta$  and  $\gamma$ , starting from randomly initialized networks with same variances employed in the mean-field theory calculations. As shown in Fig. 2, the  $(\gamma, \log_{10} \eta)$ -plane clearly partitions into distinct phases characterized by convergent and non-convergent optimization dynamics, and our theoretically predicted upper bound  $\eta_*$  closely agrees with the experimentally determined phase boundary.

In addition to the striking match between our theoretical prediction and the experimentally determined phase boundaries, the experimental results also suggest a tendency for smaller  $\gamma$ -initiations to produce lower values of test loss after a fixed number of epochs, i.e. faster convergence. We leave detailed investigation of this initialization scheme to future work. Also, dark strips can be observed in the heatmaps indicate the optimal learning rates for optimization, which is around  $\eta_{*} / 2$  and consistent with LeCun et al. (2012) in the quadratic approximation to the loss.

The architectural design for our experiments is as follows. In the fully connected architecture, we choose  $L = 4$  layers with  $N_{l} = 1000$  hidden units per layer except the final (linear) layer which has  $N_{L} = 10$  outputs. Batch normalization is applied after each linear operation except for the final linear output layer. The convolutional network has a similar structure with  $L = 4$  layers. The first three are convolutional layers with filter size 3, stride 2, and number of channels  $C_1 = 30$ ,  $C_2 = 60$ ,  $C_3 = 90$ . The final layer is a fully connected output layer to perform classification. The other architectural/optimization hyperparameters were chosen to be  $\sigma_w^2 = 2$ ,  $\sigma_b^2 = 0.5$ ,  $\beta = 0$  and  $\mu = 0.9$ .

![](images/6b77add4e486ae441e2356a7ba267a6049c28994ac74f37e185db07c23f1a011.jpg)  
Figure 1: The maximum eigenvalue  $\bar{\lambda}_{\mathrm{max}}$  and associated critical learning rate  $\eta_{*}$  for vanilla (blue) and BatchNorm networks (red) as a function of the BatchNorm parameter  $\gamma$  for different choices of architecture (fully-connected and convolutional). (a, c) shows the flattening effect of BatchNorm on the loss function for a wide range of hyperparameters and (b, d) further show that for sufficiently small  $\gamma$  BatchNorm enables optimization with much higher learning rate than vanilla networks.

![](images/1a9edf769b3a8e8bdd31fe84c1a8c28d87a5a4494f7c39c09890b47f28402552.jpg)

![](images/a3a0ef39156e729df5383951fb4ed1ed113b86fb9c6020b68a5bd29fb06e1e8d.jpg)

![](images/4f819815c8ad35626fc85d929467b44b0fb738205e990baca617a340a5918db7.jpg)

![](images/2c9f82bb15248b32cd4e296ce30b1fbd566c853a36c5c2667a501ae23bb6dc2e.jpg)  
Figure 2: Heatmaps showing test loss as a function of  $(\log_{10}\eta ,\gamma)$  after 5 epochs of training for different choices of dataset and architecture. Results were obtained by averaging 5 random restarts. The white region indicates parameter explosion for at least one of the runs. The red line shows the theoretical prediction for the maximal learning rate  $\eta_{*}$ . The dark band on the heatmaps for CIFAR-10 approximately tracks the optimal learning rate  $\eta_{*} / 2$  in the quadratic approximation to the loss. Note the log scale for the learning rate, so the theory matches the experiments over three orders of magnitude for  $\eta$ .

![](images/46fa84d988c20aa35c9d1553b58eff4df32a71e786c8df01620b9edc11549d09.jpg)

![](images/7e5b363e0283ef167e2db8f898bdc61281e9d380bed1c2d005aae7b477a767b7.jpg)

# 5 CONCLUSION AND FUTURE WORK

In this paper, we studied the impact of BatchNorm on the loss surface of multi-layer neural networks and its implication for training dynamics. By developing recursion relations for the relevant order parameters, the maximum eigenvalue of the Fisher Information matrix  $\bar{\lambda}_{\mathrm{max}}$  can be estimated and related to the maximal learning rate. The theory correctly predicts that adding BatchNorm with small  $\gamma$  allows the training algorithm to exploit much larger learning rates, which speeds up convergence. The experiments also suggest that using a smaller  $\gamma$  results in a lower test loss for a fixed number of training epochs. This suggests that initialization with smaller  $\gamma$  may help the optimization process in deep learning models, which will be interesting for future study.

The close agreement between theoretical predictions and the experimentally determined phase boundaries strongly supports the validity of our analysis, despite the non-rigorous nature of the derivations. Although similar approaches have been used in other work (Poole et al., 2016; Schoenholz et al., 2017; Yang & Schoenholz, 2017; Xiao et al., 2018; Karakida et al., 2018), we hope that future work will place these results on a firmer mathematical footing. Furthermore, our BatchNorm analysis is not limited to the convolutional and fully-connected architectures we considered in this paper and can be extended to arbitrary feedforward architectures such as ResNets.

# REFERENCES

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Ryo Karakida, Shotaro Akaho, and Shun ichi Amari. Universal statistics of fisher information in deep neural networks: Mean field approach. arXiv preprint arXiv:1806.01316, 2018.  
Diederik P. Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. pp. 10971105, 2012.  
Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pp. 9-48. Springer, 2012.  
Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. pp. 3360-3368, 2016.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. 2015.  
Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch normalization help optimization? (no, it is not about internal covariate shift). arXiv preprint arXiv:1805.11604, 2018.  
Samuel S Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. In International Conference on Learning Representations (ICLR), 2017.  
Haim Sompolinsky and Annette Zippelius. Relaxational dynamics of the edwards-anderson model and the mean-field theory of spin-glasses. Physical Review B, 25(11):6860, 1982.  
Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel S. Schoenholz, and Jeffrey Pennington. Dynamical isometry and a mean field theory of cnns: How to train 10,000-layer vanilla convolutional neural networks. In International Conference on Machine Learning (ICML), 2018.  
Greg Yang and Samuel S. Schoenholz. Mean field residual networks: On the edge of chaos. In Advances in Neural Information Processing Systems (NIPS), 2017.
