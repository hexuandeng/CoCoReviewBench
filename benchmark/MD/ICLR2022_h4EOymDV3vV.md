# DIFFUSION-BASED REPRESENTATION LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Score-based methods represented as stochastic differential equations on a continuous time domain have recently proven successful as a non-adversarial generative model. Training such models relies on denoising score matching, which can be seen as multi-scale denoising autoencoders. Here, we augment the denoising score-matching framework to enable representation learning without any supervised signal. GANs and VAEs learn representations by directly transforming latent codes to data samples. In contrast, the introduced diffusion-based representation learning relies on a new formulation of the denoising score-matching objective and thus encodes information needed for denoising. We illustrate how this difference allows for manual control of the level of details encoded in the representation. Using the same approach, we propose to learn an infinite-dimensional latent code which achieves improvements of state-of-the-art models on semi-supervised image classification. As a side contribution, we show how adversarial training in score-based models can improve sample quality and improve sampling speed using a new approximation of the prior at smaller noise scales.

# 1 INTRODUCTION

Diffusion-based models have recently proven successful for generating images (Sohl-Dickstein et al. (2015); Song & Ermon (2020); Song et al. (2020)), graphs (Niu et al. (2020)), shapes (Cai et al. (2020)), and audio (Chen et al. (2020b); Kong et al. (2021)). Two promising approaches apply step-wise perturbations to samples of the data distribution until the perturbed distribution matches a known prior (Song & Ermon (2019); Ho et al. (2020)). A model is trained to estimate the reverse process, which transforms samples of the prior to samples of the data distribution (Saremi et al. (2018)). These diffusion models were further refined (Nichol & Dhariwal (2021); Luhman & Luhman (2021)) and even achieved better image sample quality than GANs (Dhariwal & Nichol (2021); Ho et al. (2021); Mehrjou et al. (2017); Sajjadi et al. (2018)). Further, Song et al. showed that these frameworks are discrete versions of continuous-time perturbations by stochastic differential equations and proposed a score-based generative modeling framework on continuous time. Unlike generative models such as GANs and VAEs and various versions of autoencoders, where the latent code is a fixed part of the architecture, the original form of diffusion models does not come with such a fixed architectural module that captures the representation. Notice that because the diffusion-based models use a stochastic process to learn the score functions of the stochastically transformed distributions, we use the terms diffusion-based and score-based models interchangeably throughout this manuscript.

Learning desirable representations has been an integral component of generative models such as GANs and VAEs (Bengio et al. (2013); Radford et al. (2016); Chen et al. (2016); van den Oord et al. (2017); Donahue & Simonyan (2019); Chen et al. (2020a); Scholkopf et al. (2021)). Considering score-based methods as promising and theoretically grounded generative models, here we propose a method to augment the underlying SDE for learning a latent data-generating code. The key idea is to provide a representation of the clean data as additional input to the model estimating the score function. Our approach is illustrated in Figure 1. Even though diffusion models are considered non-adversarial, in this work we show that it can actually be an adversarial process by means of the weighting function  $\lambda$  which has positive effect in the sampling quality of the score-based models.

We begin by briefly revisiting the foundations of score-based generative models in Section 1.1. In Section 2 we present our method for representation learning, propose how to apply adversarial training in score-based models in Section 3.1, and motivate the use of fewer sampling steps in Section 3.2. We follow up with experimental results and evaluations of our proposed methods in Section 4.

![](images/74225f55aff51fcf58e1119ab12bdddb75e45b1fff9c8022408eb16986a53088.jpg)  
Figure 1: Conditional score matching with a parametrized latent code is representation learning. Denoising score matching estimates the score at each  $x_{t}$ ; we add a latent representation  $z$  of the clean data  $x_{0}$  as additional input to the score estimator.

For further clarity, the main contributions of this work are itemized in the following.

- We present an alternative formulation of the denoising score matching objective, showing that the objective cannot be reduced to zero.  
- We introduce and evaluate Diffusion-based Representation Learning (DRL), a new framework for representation learning in score-based generative models. We show how this framework allows for manual control of the level of details encoded in the representation. We extend our approach to an infinite-dimensional code and evaluate it on the downstream task of semi-supervised image classification, improving state-of-the-art approaches.  
- Unlike the widely admitted non-adversarial nature of these models, we show there exists an inherent component in the formulation that acts adversarially and can be leveraged to improve the sample quality.  
- We evaluate the effect of the initial noise scale and achieve significant improvements in sampling speed, which is a bottleneck in diffusion-based generative models compared with GANs and VAEs, without sacrificing image quality.

# 1.1 SCORE-BASED GENERATIVE MODELING

In the following, we give a brief overview of the technical background for the framework of the diffusion-based generative model, for example, as described in (Song et al., 2021b). The forward diffusion process of the data is modeled as a Stochastic Differential Equation (SDE) on a continuous time domain  $t \in [0,T]$ . Let  $x_0 \in \mathbb{R}^d$  denote a sample of the data distribution  $x_0 \sim p_0$ , where  $d$  is the data dimension. The trajectory  $(x_{t})_{t \in [0,T]}$  of data samples is a function of time determined by the diffusion process. The SDE is chosen such that the distribution  $p_{0T}(x_T|x_0)$  for any sample  $x_0 \sim p_0$  can be approximated by a known prior distribution. Notice that the subscript  $0T$  of  $p_{0T}$  refers to the conditional distribution of the diffused data at time  $T$  given the data at time 0. For simplicity we limit the remainder of this paper to the so-called Variance Exploding SDE (Song et al. (2021b)), that is,

$$
\mathrm {d} x = f (x, t) \mathrm {d} t + g (t) \mathrm {d w} := \sqrt {\frac {\mathrm {d} [ \sigma^ {2} (t) ]}{\mathrm {d} t}} \mathrm {d w}, \tag {1}
$$

where  $\mathrm{w}$  is the standard Wiener process. The perturbation kernel of this diffusion process has a closed-form solution being  $p_{0t}(x_t|x_0) = \mathcal{N}(x_t;x_0, [\sigma^2(t) - \sigma^2(0)]I)$ . It was shown by Anderson (1982) that the reverse diffusion process is the solution to the following SDE:

$$
\mathrm {d} x = \left[ f (x, t) - g ^ {2} (t) \nabla_ {x} \log p _ {t} (x) \right] \mathrm {d} t + g (t) \mathrm {d} \bar {\mathrm {w}}, \tag {2}
$$

where  $\overline{\mathbf{w}}$  is the standard Wiener process when the time moves backwards. Thus, given the score function  $\nabla_{x}\log p_{t}(x)$  for all  $t\in [0,T]$ , we can generate samples from the data distribution  $p_0(x)$ . In order to learn the score function, the simplest objective is Explicit Score Matching (ESM) (Hyvarinen & Dayan (2005)), that is,

$$
\mathbf {E} _ {x _ {t}} \left[ \| s _ {\theta} \left(x _ {t}, t\right) - \nabla_ {x _ {t}} \log p _ {t} (x _ {t}) \| _ {2} ^ {2} \right]. \tag {3}
$$

Since the ground-truth score function  $\nabla_{x_t}\log p_t(x_t)$  is generally not known, one can apply denoising score matching (DSM) (Vincent (2011)), which is defined as the following:

$$
J _ {t} ^ {D S M} (\theta) = \mathbf {E} _ {x _ {0}} \left\{\mathbf {E} _ {x _ {t} | x _ {0}} \left[ \left\| s _ {\theta} (x _ {t}, t) - \nabla_ {x _ {t}} \log p _ {0 t} (x _ {t} | x _ {0}) \right\| _ {2} ^ {2} \right] \right\}. \tag {4}
$$

The training objective over all  $t$  is augmented by Song et al. (2021b) with a time-dependent positive weighting function  $\lambda(t)$ , that is,  $J^{DSM}(\theta) = \mathbf{E}_t[\lambda(t)J_t^{DSM}(\theta)]$ .

In the next section, we present an alternative formulation of this objective function that provides a deeper insight and motivates the representation learning in diffusion models.

# 2 DIFFUSION-BASED REPRESENTATION LEARNING

# 2.1 ALTERNATIVE FORMULATION OF DENOISING SCORE MATCHING

We begin this section by presenting an alternative formulation of the Densoising Score Matching (DSM) objective, which shows that this objective cannot be made arbitrarily small.

Proposition 1. The formula of the Denoising Score Matching objective can be rearranged as

$$
\begin{array}{l} J _ {t} ^ {D S M} (\theta) = \mathbf {E} _ {x _ {0}} \left\{\mathbf {E} _ {x _ {t} | x _ {0}} \left[ \| \nabla_ {x _ {t}} \log p _ {0 t} (x _ {t} | x _ {0}) - \nabla_ {x _ {t}} \log p _ {t} (x _ {t}) \| _ {2} ^ {2} \right. \right. \tag {5} \\ \left. + \left\| s _ {\theta} (x _ {t}, t) - \nabla_ {x _ {t}} \log p _ {t} (x _ {t}) \right\| _ {2} ^ {2} \right] \}. \\ \end{array}
$$

Proof sketch. The DSM objective in equation 4 is minimized when  $\forall x_{t}:s_{\theta}(x_{t},t) = \nabla_{x_{t}}\log p_{t}(x_{t})$  and differs from ESM in equation 3 only by a constant (Vincent (2011)). Hence, the constant is equal to the minimum achievable value of the DSM objective.

The full proof for Proposition 1 is included in the Appendix (A.1). It is noteworthy that the first term in the rhs of the equation 5 does not depend on the learned score function of  $x_{t}$  for every  $t \in [0, T]$ . Rather, it is influenced by the diffusion process that generates  $x_{t}$  from  $x_{0}$ . This observation has not been emphasized previously, probably because it has no direct effect on the learning of the score function that is handed by the second term in the rhs of equation 5. However, the additional constant has major implications for finding other hyperparameters such as the function  $\lambda(t)$  and the choice of  $\sigma(t)$  in the forward SDE. To the best of our knowledge, there is no known theoretical justification for the values of  $\sigma(t)$ . While these hyperparameters could be optimized in ESM using gradient-based learning, this ability is severely limited by the non-vanishing constant in equation 5. Similarly, it impacts the behaviour of adversarial training in diffusion models, which we analyze in Section 4.2.

Even though the non-vanishing constant in the denoising score matching objective presents a burden in multiple ways such as hyperparameter search and model evaluation, it provides an opportunity for latent representation learning, which will be described in the following sections.

# 2.2 CONDITIONAL SCORE MATCHING

Class-conditional generation can be achieved in score-based models by training an additional time-dependent classifier  $p_t(y|x_t)$  (Song et al. (2021b)). In particular, the conditional score for a fixed  $y$  can be expressed as the sum of the unconditional score and the score of the classifier, that is,  $\nabla_{x_t}\log p_t(x_t|y) = \nabla_{x_t}\log p_t(x_t) + \nabla_{x_t}\log p_t(y|x_t)$ .

We propose conditional score matching as an alternative way to allow for controllable generation. Given supervised samples  $(x,y(x))$ , the new training objective for each time  $t$  becomes

$$
J _ {t} ^ {C S M} (\theta) = \mathbf {E} _ {x _ {0}} \left\{\mathbf {E} _ {x _ {t} | x _ {0}} \left[ \| s _ {\theta} \left(x _ {t}, t, y \left(x _ {0}\right)\right) - \nabla_ {x _ {t}} \log p _ {0 t} \left(x _ {t} \mid x _ {0}\right) \| _ {2} ^ {2} \right] \right\}. \tag {6}
$$

The objective in equation 6 is minimized if and only if the model equals the conditional score function  $\nabla_{x_t}\log p_t(x_t|y(x_0) = \hat{y})$  for all labels  $\hat{y}$ . Note that conditional score matching is directly performed during training and does not require to train an additional classifier over the whole time domain.

# 2.3 LEARNING LATENT REPRESENTATIONS

Since supervised data is limited and rarely available, we propose to learn a labeling function  $y(x_0)$  at the same time as optimizing the conditional score matching objective in equation 6. In particular, we represent the labeling function as a trainable encoder  $E_{\phi}:\mathbb{R}^{d}\to \mathbb{R}^{c}$ , where  $E_{\phi}(x_0)$  maps the data sample  $x_0$  to its corresponding code in the  $c$ -dimensional latent space. The code is then used as additional input to the model. Formally, the proposed learning objective for Diffusion-based Representation Learning (DRL) is the following:

$$
J ^ {D R L} (\theta , \phi) = \mathbf {E} _ {t, x _ {0}, x _ {t}} [ \lambda (t) \| s _ {\theta} (x _ {t}, t, E _ {\phi} (x _ {0})) - \nabla_ {x _ {t}} \log p _ {0 t} (x _ {t} | x _ {0}) \| _ {2} ^ {2} ]. \tag {7}
$$

To get a better idea of the above objective, we provide an intuition for the role of  $E_{\phi}(x_0)$  in the input of the model. The score model  $s_{\theta}(\cdot ,\cdot ,\cdot):\mathbb{R}^d\times \mathbb{R}\times \mathbb{R}^c\to \mathbb{R}^d$  is a vector-valued function whose output points to different directions based on the value of its third argument. In fact,  $E_{\phi}(x_0)$  selects the direction that best recovers  $x_0$  from  $x_{t}$ . Hence, when optimizing over  $\phi$ , the encoder learns to extract the information from  $x_0$  in a reduced-dimensional space that helps recover  $x_0$  by denoising  $x_{t}$ . Notice that finding the denoising direction requires information from both  $x_0$  and  $x_{t}$  and  $E_{\phi}$  can only extract the partial information from the source  $x_0$ .

We show in the following that equation 7 is a valid representation learning objective. The score of the perturbation kernel  $\nabla_{x_t}\log p_{0t}(x_t|x_0)$  is a function of only  $t$ ,  $x_{t}$  and  $x_0$ . Thus the objective can be reduced to zero if all information about  $x_0$  is contained in the latent representation  $E_{\phi}(x_0)$ . When  $E_{\phi}(x_0)$  has no mutual information with  $x_0$ , the objective can only be reduced up to the constant in equation 5. Hence, our proposed formulation takes advantage of the non-zero lower-bound of equation 5, which can only vanish when data information is distilled in a code provided as input to the model. These properties show that equation 7 is a valid objective for representation learning.

Our proposed representation learning objective enjoys the continuous nature of SDEs, a property that is not available in many previous representation learning methods (Radford et al. (2016); Chen et al. (2016); Locatello et al. (2019)). In DRL, the encoder is trained to represent the information needed to denoise  $x_0$  for different levels of noise  $\sigma(t)$ . We hypothesize that by adjusting the weighting function  $\lambda(t)$ , we can manually control the granularity of the features encoded in the representation and provide empirical evidence as support. When  $t \to T$  that is associated to high noise level, the mutual information of  $x_t$  and  $x_0$  starts to vanish, thus denoising requires all information about  $x_0$  to be contained in the code. In contrast, when  $t \to 0$  that corresponds to low noise levels,  $x_t$  contains coarse-grained features of  $x_0$  and only fine-grained properties may have been washed out due to the small magnitude of noise. Hence, the representation learns to keep the information needed to recover these fine-grained details. We provide empirical evidence to support this hypothesis in Section 4.

It is noteworthy that  $E_{\phi}$  does not need to be a deterministic function. In principle, it can be viewed as an information channel that controls the amount of information that the score model receives from the initial point of the diffusion process. With this perspective, any deterministic or stochastic function that can manipulate  $I(x_{t},x_{0})$ , the mutual information between  $x_0$  and  $x_{t}$ , can be used. This opens up the room for stochastic encoders similar to VAEs that we call Variational Diffusion-based Representation Learning (VDLR) from this point onward.

# 2.4 INFINITE-DIMENSIONAL REPRESENTATION OF FINITE-DIMENSIONAL DATA

We now present an alternative version of DRL where the representation is a function of time. Instead of emphasizing on different noise levels by weighting the training objective, as done in the previous section, we can provide the time  $t$  as input to the encoder. Formally, the new objective is the following:

$$
\mathbf {E} _ {t, x _ {0}, x _ {t}} [ \lambda (t) \| s _ {\theta} (x _ {t}, t, E _ {\phi} (x _ {0}, t)) - \nabla_ {x _ {t}} \log p _ {0 t} (x _ {t} | x _ {0}) \| _ {2} ^ {2} ], \tag {8}
$$

where  $E_{\phi}(x_0,t)$  in equation 7 is replaced by  $E_{\phi}(x_0,t)$ . Intuitively, it allows the encoder to extract the necessary information of  $x_0$  required to denoise  $x_t$  for any noise level. This can be seen as a rich representation learning in the following way. Normally in autoencoders or other static representation learning methods, the input data  $x_0 \in \mathbb{R}^d$  is mapped to a single point  $z \in \mathbb{R}^c$  in the code space. We propose a richer representation where the input  $x_0$  is mapped to a curve in  $\mathbb{R}^c$  instead of a single point. Hence, the learned code is produced by the map  $x_0 \to (E_{\phi}(x_0,t))_{t \in [0,T]}$  where the infinite-dimensional object  $(E_{\phi}(x_0,t))_{t \in [0,T]}$  is the encoding for  $x_0$ .

Proposition 2. For any downstream task, the infinite-dimensional code  $(E_{\phi}(x_0,t))_{t\in [0,T]}$  learned using the objective in equation 8 is at least as good as finite-dimensional static codes learned by the reconstruction of  $x_0$ .

Proof sketch. The score matching objective can be seen as a reconstruction objective of  $x_0$  conditioned on  $x_t$ . The terminal time  $T$  is chosen large enough so that  $x_T$  is independent of  $x_0$ , hence the objective for  $t = T$  is equal to a reconstruction objective without conditioning. Therefore, there exists a  $t \in [0, T]$  where the learned representation  $E_{\phi}(x_0, t)$  is the same representation learned by the reconstruction objective of a vanilla autoencoder.

The full proof for Proposition 2 can be found in the Appendix (2). A downstream task can leverage this rich encoding in various ways, including the use of either the static code for a fixed  $t$ , or the use of the whole trajectory  $(E_{\phi}(x_0,t))_{t\in [0,T]}$  as input. We posit the conjecture that the proposed rich representation is helpful for downstream tasks when used for pretraining and evaluate it on semi-supervised image classification in Section 4.1.1. The current state-of-the-art model for many semi-supervised image classification benchmarks is LaplaceNet (Sellars et al. (2021)). The approach alternates between assigning pseudo-labels to samples and standard supervised training of a classifier. The key idea is to assign pseudo-labels by minimizing the graphical Laplacian of the prediction matrix, where similarities of data samples are calculated based on a hidden layer representation in the classifier. Note that LaplaceNet applies mixup (Zhang et al. (2017)) that changes the input distribution of the classifier. We evaluate our method both with and without mixup to investigate its effect. We evaluate our approach on CIFAR-10 (Krizhevsky et al. (a)), CIFAR-100 (Krizhevsky et al. (b)) and MinilImageNet (Vinyals et al. (2016)).

# 3 REMARKS ON DIFFUSION-BASED GENERATIVE MODELS

The following sections contain side-contributions on training and sampling in diffusion-based models. First we want to note that diffusion models are claimed to minimize KL-Divergence (Song et al. (2021a)). However, we noticed that the assumption of the model being curl-free is usually not enforced and thus this statement might not hold in practice.

# 3.1 ADVERSARIAL TRAINING IN SCORE-BASED GENERATIVE MODELS

In general, score-based generative models enjoy the advantages of non-adversarial training. Hence, they do not suffer from mode collapse, a common problem observed in GANs (Thanh-Tung et al. (2018)), but instead minimize KL-Divergence. However, it has been shown that KL-Divergence is not a good indicator of perceptual image quality (Theis et al. (2016); Gatys et al. (2017); Arjovsky et al. (2017)). Hence, GANs were extended to maximize the variation lower bound of the family of  $f$ -divergences (Nowozin et al. (2016)). Similarly, Song et al. introduced the set of  $\lambda$ -divergences between two probability distributions  $p$  and  $q$ , defined as

$$
D _ {\lambda} (p | | q) = \frac {1}{2} \int_ {0} ^ {T} \mathbf {E} _ {x \sim p _ {t} (x)} [ \lambda (x, t) \| \nabla_ {x} \log p _ {t} (x) - \nabla_ {x} \log q _ {t} (x) \| _ {2} ^ {2} ] d t. \tag {9}
$$

Note that  $\lambda$ -divergences can express any  $f$ -divergence and that we can train on the respective divergence solely by adjusting  $\lambda(x, t)$  in the denoising score matching objective. However, minimizing a specific  $f$ -divergence requires the knowledge of the ratio of densities  $p_t(x) / q_t(x)$  that is not at hand.

We form a min-max game to minimize the worst-case  $\lambda$ -divergence defined by an additional adversary, which is trained alternately with the model. Since we do not have access to the score function  $\nabla_{x_t}\log p_t(x_t)$ , we approximate the  $\lambda$ -divergence using the denoising score matching objective, and hence solve the following optimization problem:

$$
\min  _ {\theta} \max  _ {\lambda} \frac {1}{2} \int_ {0} ^ {T} \mathbf {E} _ {x _ {0}} \left\{\mathbf {E} _ {x _ {t} | x _ {0}} [ \lambda (x _ {t}, t) (\| \nabla_ {x _ {t}} \log p _ {0 t} (x _ {t} | x _ {0}) - \nabla_ {x _ {t}} \log p _ {t} (x _ {t}) \| _ {2} ^ {2} \right. \tag {10}
$$

$$
\left. + \left\| s _ {\theta} \left(x _ {t}, t\right) - \nabla_ {x _ {t}} \log p _ {t} \left(x _ {t}\right) \left\| _ {2} ^ {2}\right) \right] \right\} d t.
$$

As a result,  $\lambda$  is biased by the non-vanishing constant and might not explicitly focus on regions where the model is distant from the true score function. We hypothesize that the denoising score matching

approximation can still yield the improvements of an adversarial divergence, and report the empirical evidence to support it in Section 4.2.

For simplicity, we propose to limit  $\lambda$  to the set of linear functions on  $t$ , thus removing the dependence on  $x$ . Details are described in the Appendix in A.7. In order to prevent omission of any values of  $t$  in the training, we additionally interpolate between training on the  $D_{KL}$  and  $D_{\lambda}$  objective with a hyperparameter  $p_{KL}$  which determines the percentage of training on  $D_{KL}$ .

# 3.2 THE CHOICE OF INITIAL NOISE SCALE

The initial noise scale controls the quality and diversity of the generated samples. It is proposed by Song & Ermon (2020) that the initial noise scale has to be chosen so that the sampling trajectory can traverse from every mode to the other one. Even though this looks good in terms of diversity, it can be quite wasteful because the noise must be large enough so that the whole empirical data distribution is covered. In fact, it is not necessary to traverse from every mode to every other mode directly. We can traverse from a mode to a number of nearest modes and then from those modes to the farther ones. This way, we can choose a significantly smaller initial noise scale that saves us many steps in the sampling trajectory.

According to Song & Ermon (2020),  $\sigma(T)$  should be set numerically equal to the maximum pairwise distance of images, which is approximately 170 for CIFAR-10 training images. However, recent high quality results have been achieved with  $\sigma(T) = 50$  (Song et al. (2021b)). When using smaller initial noise scales, we can further approximate the noise distribution using the sum of a uniform random variable in the image domain  $u \sim U([0,1]^d)$  and gaussian noise  $z \sim \mathcal{N}(0,\sigma(T))$ . We evaluate qualitative diversity and FID of generated images for various initial noise scales in Section 4.3.

# 4 RESULTS

# 4.1 DIFFUSION-BASED REPRESENTATION LEARNING

For all experiments, we use the same function  $\sigma(t), t \in [0,1]$  as in Song et al. (2021b), which is  $\sigma(t) = \sigma_{\min} \left( \frac{\sigma_{\max}}{\sigma_{\min}} \right)^t$ , where  $\sigma_{\min} = 0.01$  and  $\sigma_{\max} = 50$ . Further, we use a 2d latent space and set  $\lambda(t) = \sigma^2(t)$ , which has been shown to yield the KL-Divergence objective (Song et al. (2021a)). Our goal is not to produce state-of-the-art image quality, rather showcase the representation learning method. Because of that and also limited computational resources, we did not carry out an extensive hyperparameter sweep (cf. A.4 for details). We illustrate how the representation encodes information for denoising in Appendix in Figure 9.

We first train a DRL model with L1-regularization on the latent code on MNIST (LeCun & Cortes (2010)) and CIFAR-10 (Krizhevsky et al. (a)). Figure 2 shows samples from a grid over the latent space and a point cloud visualization of the latent values  $z = E_{\phi}(x_0)$ . For MNIST, we can see that the value of  $z_1$  controls the stroke width, while  $z_2$  weakly indicates the class. The latent code of CIFAR-10 samples mostly encodes information about the background color, which is weakly correlated to the class. The use of a probabilistic encoder (VDRL) leads to similar representations (cf. 5, 6). We further want to point out that the generative process using the reverse SDE involves randomness and thus generates different samples for a single latent representation. The diversity of samples however steadily decreases with the dimensionality of the latent space, which is empirically shown in Figure 10 of the Appendix.

Next, we analyze the behavior of the representation when adjusting the weighting function  $\lambda(t)$  to focus on higher noise levels, which can be done by changing the sampling distribution of  $t$ . To this end, we sample  $t \in [0,1]$  such that  $\sigma(t)$  is uniformly sampled from the interval  $[\sigma_{\min}, \sigma_{\max}] = [0.01, 50]$ . Figure 3 shows the resulting representation of VDRL (cf. Fig. 7, 8 for DRL results). As expected, the latent representation for MNIST encodes information about classes rather than fine-grained features such as stroke width. This validates our hypothesis of Section 2.3 that we can control the granularity of features encoded in the latent space. For CIFAR-10, the model again only encodes information about the background. This is not surprising, since the background contains most information about the image. A detailed analysis of class separation in the extreme case of training on single timescales is included in the Appendix (A.3).

![](images/258cff37c05310b61387b046933b89728d5fa436b68fe390545aefce4dbcd90c.jpg)  
(a) Generated samples

![](images/5a6ea2bf1d5f020e4d3d2c4ea82a102e3a96557c0bd0896659ff462791da753e.jpg)  
(b) Latent codes

![](images/c28f43ddbf9aba0f2820a08ddd53f673fc19eab566aa8d9e1b722bfa2ae16b21.jpg)  
(c) Generated samples

![](images/bf82e6781f086bde953ad92c76ce0a181476a10cf192c5a193080275aab438ae.jpg)  
(d) Latent codes

![](images/8e2ef1eaab634f940852f26b1ce0cda3312c837838dc1d253a75866617abed50.jpg)  
Figure 2: Results of a DRL model trained on MNIST (a-b) and CIFAR-10 (c-d) using uniform sampling of  $t$ . Samples are generated from a grid of latent values ranging from -1 to 1. The point clouds visualize the latent representation of test samples, colored according to the digit class.  
(a) Generated samples  
Figure 3: Results of a VDRL model trained with a focus on high noise levels on MNIST (a-b) and CIFAR-10 (c-d). Samples are generated from a grid of latent values ranging from  $-2$  to 2. The point clouds visualize the latent representation of test samples, colored according to the digit class.

![](images/40e45082dc8e75f0ea40d1e257a5f8ce20548d9691c2ef59a14e50dd24802143.jpg)  
(b) Latent codes

![](images/ced8cea194b333549a52fd99673d781ecd1a0401e1af1b1c56acfd37f59f5a93.jpg)  
(c) Generated samples

![](images/d92680b55c87d882f5b342ff00a68ac61ae83f43d6d000f0df8c3b96f5f40b4f.jpg)  
(d) Latent codes

Overall, the difference in the latent codes for varying  $\lambda(t)$  shows that we can control the granularity encoded in the representation of DRL. While this ability does not exist in previously proposed models for representation learning, it provides a significant advantage when there exists some prior information about the level of detail that we intend to encode in the target representation.

# 4.1.1 APPLICATION TO SEMI-SUPERVISED IMAGE CLASSIFICATION

In the following, we evaluate the infinite-dimensional representation  $(E_{\phi}(x_0,t))_{t\in [0,T]}$  on semi-supervised image classification, where we use DRL and VDRL as pretraining for the LaplaceNet classifier. Table 1 depicts the classifier accuracy on test data for different pretraining settings. Details for architecture and hyperparameters are described in A.6.

Our proposed pretraining using DRL significantly improves the baseline and often surpasses the state-of-the-art performance of LaplaceNet. Most notable are the results of DRL and VDRL without mixup, which achieve high accuracies without being specifically tailored to the downstream task of classification. Note that pretraining the classifier as part of an autoencoder did not yield any improvements (cf. 6). Combining DRL with mixup yields inconsistent improvements, results are reported in Table 7. In addition, DRL pretraining achieves much better performances when only limited computational resources are available (cf. 4, 5).

We further evaluate the infinite-dimensional representation on few-shot image classification using the representation at different timescales as input. The detailed results are shown in the Appendix (A.5). In summary, the representations of DRL and VDRL achieve significant improvements compared to that of an autoencoder or VAE for several values of  $t$  and are similar for  $t$  close to 1.

Overall the results align with the theoretical foundation in Proposition 2 that the rich representation of DRL is at least as good as the static code learned using a reconstruction objective. It further shows that in practice, the infinite-dimensional code is superior to the static representation for the application to downstream tasks by a significant margin.

Table 1: Comparison of classifier accuracy in % for different pretraining settings. Scores better than the SOTA model (LaplaceNet) are in bold. "DRL" pretraining is our proposed representation learning, and "VDRL" the respective version which uses a probabilistic encoder.  

<table><tr><td>Pretraining Mixup</td><td></td><td>LaplaceNet None No</td><td>LaplaceNet None Yes</td><td>Ours DRL No</td><td>Ours DRL Yes</td><td>Ours VDRL No</td></tr><tr><td>Dataset</td><td>#labels</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">CIFAR-10</td><td>100</td><td>73.68</td><td>75.29</td><td>74.31</td><td>64.67</td><td>81.63</td></tr><tr><td>500</td><td>91.31</td><td>92.53</td><td>92.70</td><td>92.31</td><td>92.79</td></tr><tr><td>1000</td><td>92.59</td><td>93.13</td><td>93.24</td><td>93.42</td><td>93.60</td></tr><tr><td>2000</td><td>94.00</td><td>93.96</td><td>94.18</td><td>93.91</td><td>93.96</td></tr><tr><td>4000</td><td>94.73</td><td>94.97</td><td>94.75</td><td>95.22</td><td>95.00</td></tr><tr><td rowspan="4">CIFAR-100</td><td>1000</td><td>55.58</td><td>55.24</td><td>55.85</td><td>55.74</td><td>56.47</td></tr><tr><td>4000</td><td>67.07</td><td>67.25</td><td>67.22</td><td>67.47</td><td>67.54</td></tr><tr><td>10000</td><td>73.19</td><td>72.84</td><td>73.31</td><td>73.66</td><td>73.50</td></tr><tr><td>20000</td><td>75.80</td><td>76.07</td><td>76.46</td><td>76.88</td><td>76.64</td></tr><tr><td rowspan="2">MiniImageNet</td><td>4000</td><td>58.40</td><td>58.84</td><td>58.95</td><td>59.29</td><td>59.14</td></tr><tr><td>10000</td><td>66.65</td><td>66.80</td><td>67.31</td><td>66.63</td><td>67.46</td></tr></table>

Table 2: FID and Inception Score for different interpolations between maximum likelihood training and training based on the adversarial  $\lambda^{\prime}$ .  $p_{KL} = 1.0$  corresponds to original training.  

<table><tr><td>pKL</td><td>0</td><td>0.05</td><td>0.5</td><td>0.95</td><td>1</td></tr><tr><td>FID ↓</td><td>3.40</td><td>3.36</td><td>3.33</td><td>3.37</td><td>3.62</td></tr><tr><td>IS ↑</td><td>9.73</td><td>9.77</td><td>9.79</td><td>9.50</td><td>9.44</td></tr></table>

# 4.2 ADVERSARIAL TRAINING IN SCORE-BASED GENERATIVE MODELS

We evaluate our approach of optimizing the adversarial  $\lambda$ -Divergence on the task of synthetic image generation of CIFAR-10 images. We evaluate our approach for different values for  $p_{KL}$ , which is used to interpolate between the original and adversarial training objective. The resulting FID and Inception Scores are displayed in Table 2. The results show that for any value of  $p_{KL} \leq 0.95$ , the sample quality is improved significantly. However, in contrast to our intentions, the adversary converges to the extreme value of  $\alpha \approx -1$  in the first 10k iterations and does not change afterwards (cf. 14). While our adversary learns to put focus on values of  $t$  where the model is bad, the model apparently cannot improve in this region (cf. 13).

Overall, the higher image quality after training with the adversary indicates that diffusion models can be improved using adversarial training. Since the loss is not significantly reduced, more complex adversaries as in Hardy et al. (2018) will only have the same effect as a predefined  $\lambda$ . As shown in our experiments, there is much room for improvement by searching for a good  $\lambda$ . Furthermore, additional improvements might be achieved in the future by introducing inductive biases allowing the model to capture more high-frequency information, thus being able to reduce the loss for small values of  $t$ .

# 4.3 THE CHOICE OF INITIAL NOISE SCALE

In the following, we evaluate image quality and diversity for different initial noise scales. Note that we do not change  $\sigma (T)$ , but instead evaluate generated images for different initial times  $t_{init}$  which implicitly define the initial noise scale  $\sigma (t_{init})$ . This reduces the number of sampling steps per image, which is  $1000\cdot t_{init}$  and thus directly proportional to  $t_{init}$ . Table 3 shows the FID of generated images for various values of  $t_{init}$ . As we can see, the first 200 sampling steps can safely be replaced by approximating the prior directly either with the gaussian or the additional uniform distribution. Interestingly, using the sum of the uniform and gaussian random variables as a prior seed leads to improved image quality. This approximation for  $p_{0.7}(x)$  allows us to reduce the number of sampling steps by  $30\%$  without sacrificing image quality. Further, note that FID is occasionally lower for values of  $t_{init} < 1.0$  than for  $t_{init} = 1$ . This suggests that up to these timescales, our

Table 3: FID for different initial noise scales evaluated on  ${20}\mathrm{k}$  generated samples.  

<table><tr><td>tinit</td><td>σ(tinit)</td><td>Gaussian FID ↓</td><td>Uniform + Gaussian FID ↓</td></tr><tr><td>0.5</td><td>0.71</td><td>218.95</td><td>25.02</td></tr><tr><td>0.6</td><td>1.66</td><td>75.11</td><td>5.15</td></tr><tr><td>0.7</td><td>3.88</td><td>12.57</td><td>2.98</td></tr><tr><td>0.8</td><td>9.10</td><td>3.05</td><td>2.99</td></tr><tr><td>0.9</td><td>21.33</td><td>2.97</td><td>2.94</td></tr><tr><td>1.0</td><td>50.00</td><td>3.01</td><td>2.99</td></tr></table>

![](images/3945db03f4554625ebb42a351b4e1fa58ab8a3b59b0da1048fe302e7a5e53831.jpg)  
(a)  $t_{init} = 0.5$

![](images/9d001359daa278443c740975ddf4628847d17a2e5370a75ff9bbbcafe400ebd7.jpg)  
(b)  $t_{init} = 0.6$

![](images/f4680404ca05fe250d9c29644cf188e60d94db87c4605719c4c4a5e368ca3004.jpg)  
(c)  $t_{init} = 0.7$

![](images/bb4beb463dd8811856537ec0faa7444c4653ce5c91c57f8b47a83e3071a9ce92.jpg)  
(d)  $t_{init} = 0.8$

![](images/2b226ab946cae60ef81b470f99991795dbb05859d6d98c1af4f4ebddc7a032a0.jpg)  
(e)  $t_{init} = 0.9$

![](images/0f7f8eeb89d32bcce3cbbbd9c0e909629d00d12f3da46cfc3bb6004dc853563e.jpg)  
(f)  $t_{init} = 1.0$

![](images/dd8dc811b539a11dfd79fd5956740b5d0a6e2904a3708b8feebd38e73ce548d4.jpg)  
(g)  $t_{init} = 0.5$

![](images/368cc8d4b86d2759d3f0e2b60d82d4e27a944b351a78c5d8366dfebf4b07911b.jpg)  
(h)  $t_{init} = 0.6$

![](images/ba2c27a975e58b0774eed76f542d6bdbe4124bb7481d44bb382239debd5cf0f9.jpg)  
(i)  $t_{init} = 0.7$

![](images/80f6aede8a3bb5193a1901b0855962276fa082c5a55693aed7dc7a6bcbc04bfa.jpg)  
Figure 4: Generated image samples for different values of  $t_{init}$ . Top row ((a)-(f)) uses the gaussian prior, bottom row ((g)-(l)) uses the version with an additional uniform random variable in the prior.  
(j)  $t_{init} = 0.8$

![](images/b276964b05ecf0cbf3dbe05041ae4160a5310ae3c46335eab28d715e85b56967.jpg)  
(k)  $t_{init} = 0.9$

![](images/094e10a9bbf7abc5a4eee75c82a5a74944a8c680a7fdf9a3934a66822ca6a810.jpg)  
(1)  $t_{init} = 1.0$

prior approximates the distribution better than the diffusion model when starting at  $t_{init} = 1.0$ . The generated samples shown in Figure 4 align with the quantitative results and further support our hypothesis that a significant portion of the initial steps can be replaced by an approximate prior, particularly when the latter includes the additional uniform distribution.

# 5 CONCLUSION

We presented Diffusion-based Representation Learning (DRL), a new objective for representation learning based on conditional denoising score matching. In doing so, we turned the original nonvanishing objective function into one that can be reduced arbitrarily close to zero by the learned representation. We showed that the proposed method learns interpretable features in the latent space. In contrast to previous approaches, denoising score matching as a foundation comes with the ability to manually control the granularity of features encoded in the representation. We demonstrated that the encoder can learn to separate classes when focusing on high noise levels and encodes fine-grained features such as stroke-width when mainly trained on low level noise. In addition, we proposed an infinite-dimensional representation and demonstrated its effectiveness for downstream tasks such as few-shot classification. Using the representation learning as pretraining for a classifier, we were able to improve the results of LaplaceNet, a state-of-the-art model on semi-supervised image classification. As side-contributions, we further showed how adversarial training in score-based models can improve sample quality and were able to increase sampling speed using an approximation of the density at smaller noise scales.

# 6 REPRODUCIBILITY STATEMENT

In order to ensure reproducibility, the code used to run all experiments is attached as supplementary material and will be published based upon acceptance.

# REFERENCES

Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313-326, 1982. ISSN 0304-4149. doi: https://doi.org/10.1016/0304-4149(82)90051-5. URL https://www.sciencedirect.com/science/article/pii/0304414982900515.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan, 2017.  
Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798-1828, 2013.  
Ruojin Cai, Guandao Yang, Hadar Averbuch-Elor, Zekun Hao, Serge Belongie, Noah Snavely, and Bharath Hariharan. Learning gradient fields for shape generation, 2020.  
Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1691-1703. PMLR, 13-18 Jul 2020a. URL http://proceedings.mlr.press/v119/chen20s.html.  
Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation, 2020b.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets, 2016.  
Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021.  
Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning, 2019.  
Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann, and Eli Shechtman. Controlling perceptual factors in neural style transfer, 2017.  
Coretin Hardy, Erwan Le Merrer, and Bruno Sericola. Md-gan: Multi-discriminator generative adversarial networks for distributed datasets. CoRR, abs/1811.03850, 2018. URL http:// arxiv.org/abs/1811.03850.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. CoRR, abs/2006.11239, 2020. URL https://arxiv.org/abs/2006.11239.  
Jonathan Ho, Chitwan Sahara, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. arXiv preprint arXiv:2106.15282, 2021.  
Aapo Hyvarinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005.  
Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis, 2021.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research). a. URL http://www.cs.toronto.edu/~kriz/cifar.html.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-100 (canadian institute for advanced research). b. URL http://www.cs.toronto.edu/~kriz/cifar.html.  
Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL http://yann.lecun.com/exdb/mnist/.  
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Scholkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In international conference on machine learning, pp. 4114-4124. PMLR, 2019.

Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed, 2021.  
Arash Mehrjou, Bernhard Scholkopf, and Saeed Saremi. Annealed generative adversarial networks. arXiv preprint arXiv:1705.07505, 2017.  
Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. CoRR, abs/2102.09672, 2021. URL https://arxiv.org/abs/2102.09672.  
Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling, 2020.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization, 2016.  
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks, 2016.  
Peter J. Rousseeuw. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20:53-65, 1987. ISSN 0377-0427. doi: https://doi.org/10.1016/0377-0427(87)90125-7. URL https://www.sciencedirect.com/science/article/pii/0377042787901257.  
Mehdi SM Sajjadi, Giambattista Parascandolo, Arash Mehrjou, and Bernhard Scholkopf. Tempered adversarial networks. In International Conference on Machine Learning, pp. 4451-4459. PMLR, 2018.  
Saeed Saremi, Arash Mehrjou, Bernhard Schölkopf, and Aapo Hyvärinen. Deep energy estimator networks. arXiv preprint arXiv:1805.08306, 2018.  
Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612-634, 2021.  
Philip Sellars, Angelica I. Aviles-Rivero, and Carola-Bibiane Schonlieb. Laplacenet: A hybrid energy-neural model for deep semi-supervised classification. CoRR, abs/2106.04527, 2021. URL https://arxiv.org/abs/2106.04527.  
Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2020.  
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. CoRR, abs/1907.05600, 2019. URL http://arxiv.org/abs/1907.05600.  
Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. CoRR, abs/2006.09011, 2020. URL https://arxiv.org/abs/2006.09011.  
Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models, 2021a.  
Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2021b.  
Hoang Thanh-Tung, Truyen Tran, and Svetha Venkatesh. On catastrophic forgetting and mode collapse in generative adversarial networks. CoRR, abs/1807.04015, 2018. URL http:// arxiv.org/abs/1807.04015.

Lucas Theis, Aaron van den Oord, and Matthias Bethge. A note on the evaluation of generative models, 2016.  
Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. CoRR, abs/1711.00937, 2017. URL http://arxiv.org/abs/1711.00937.  
Pascal Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661-1674, 2011. doi: 10.1162/NECO_a_00142.  
Oriol Vinyals, Charles Blundell, Timothy P. Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. CoRR, abs/1606.04080, 2016. URL http://arxiv.org/abs/1606.04080.  
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. CoRR, abs/1710.09412, 2017. URL http://arxiv.org/abs/1710.09412.
