# A DIRT-T APPROACH TO UNSUPERVISED DOMAIN ADAPTATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Domain adaptation refers to the problem of leveraging labeled data in a source domain to learn an accurate model in a target domain where labels are scarce or unavailable. A recent approach for finding a common representation of the two domains is via domain adversarial training (Ganin & Lempitsky, 2015), which attempts to induce a feature extractor that matches the source and target feature distributions in some feature space. However, domain adversarial training faces two critical limitations: 1) if the feature extraction function has high-capacity, then feature distribution matching is a weak constraint, 2) in non-conservative domain adaptation (where no single classifier can perform well in both the source and target domains), training the model to do well on the source domain hurts performance on the target domain. In this paper, we address these issues through the lens of the cluster assumption, i.e., decision boundaries should not cross high-density data regions. We propose two novel and related models: 1) the Virtual Adversarial Domain Adaptation (VADA) model, which combines domain adversarial training with a penalty term that punishes the violation the cluster assumption; 2) the Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T)<sup>1</sup> model, which takes the VADA model as initialization and employs natural gradient steps to further minimize the cluster assumption violation. Extensive empirical results demonstrate that the combination of these two models significantly improve the state-of-the-art performance on several visual domain adaptation benchmarks.

# 1 INTRODUCTION

The development of deep neural networks has enabled impressive performance in a wide variety of machine learning tasks. However, these advancements often rely on the existence of a large amount of labeled training data. In many cases, direct access to vast quantities of labeled data for the task of interest (the target domain) is either costly or otherwise absent, but labels are readily available for related training sets (the source domain). A notable example of this scenario occurs when the source domain consists of richly-annotated synthetic or semi-synthetic data, but the target domain consists of unannotated real-world data (Sun & Saenko, 2014; Vazquez et al., 2014). However, the source data distribution is often dissimilar to the target data distribution, and the resulting significant covariate shift is often detrimental to the performance of the source-trained model when applied to the target domain (Shimodaira, 2000).

Solving the covariate shift problem of this nature is an instance of domain adaptation (Ben-David et al., 2010b). In this paper, we consider a challenging setting of domain adaptation where 1) we are provided with fully-labeled source samples and completely unlabeled target samples, and 2) the existence of a classifier in the hypothesis space with low generalization error in both source and target domains is not guaranteed. Borrowing approximately the terminology from Ben-David et al. (2010b), we refer to this setting as unsupervised, non-conservative domain adaptation. We note that this is in contrast to conservative domain adaptation, where we assume our hypothesis space contains a classifier that performs well in both the source and target domains.

To tackle unsupervised domain adaptation, Ganin & Lempitsky (2015) proposed to constrain the classifier to only rely on domain-invariant features. This is achieved by training the classifier to

perform well on the source domain while minimizing the divergence between features extracted from the source versus target domains. To achieve divergence minimization, Ganin & Lempitsky (2015) employ domain adversarial training. We highlight two issues with this approach: 1) when the feature function has high-capacity and the source-target supports are disjoint, the domain-invariance constraint is potentially very weak (see Section 3), and 2) good generalization on the source domain hurts target performance in the non-conservative setting.

Saito et al. (2017) addressed these issues by replacing domain adversarial training with asymmetric tri-training (ATT), which relies on the assumption that target samples that are labeled by a source-trained classifier with high confidence are correctly labeled by the source classifier. In this paper, we consider an orthogonal assumption: the cluster assumption (Chapelle & Zien, 2005), that the input distribution contains separated data clusters and that data samples in the same cluster share the same class label. This assumption introduces an additional bias where we seek decision boundaries that do not go through high-density regions. Based on this intuition, we propose two novel models: 1) the Virtual Adversarial Domain Adaptation (VADA) model which incorporates an additional virtual adversarial training (Miyato et al., 2017) and conditional entropy loss to push the decision boundaries away from the empirical data, and 2) the Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T) model which uses natural gradients to further refine the output of the VADA model while focusing purely on the target domain. We demonstrate that

1. In conservative domain adaptation, where the classifier is trained to perform well on the source domain, VADA can be used to further constrain the hypothesis space by penalizing violations of the cluster assumption, thereby improving domain adversarial training.  
2. In non-conservative domain adaptation, where we account for the mismatch between the source and target optimal classifiers, DIRT-T allows us to transition from a joint (source and target) classifier (VADA) to a better target domain classifier. Interestingly, we demonstrate the advantage of natural gradients in DIRT-T refinement steps.

We report results for domain adaptation in digits classification (MNIST-M, MNIST, SYN DIGITS, SVHN), traffic sign classification (SYN SIGNS, GTSRB), general object classification (STL-10, CIFAR-10), and Wi-Fi activity recognition (Yousefi et al., 2017). We show that, in nearly all experiments, VADA improves upon previous methods and that DIRT-T improves upon VADA, setting new state-of-the-art performances across a wide range of domain adaptation benchmarks. In adapting MNIST  $\rightarrow$  SVHN, a very challenging task, we out-perform ATT by over  $20\%$ .

# 2 RELATED WORK

Given the extensive literature on domain adaptation, we highlight several works most relevant to our paper. Shimodaira (2000); Mansour et al. (2009) proposed to correct for covariate shift by re-weighting the source samples such that the discrepancy between the target distribution and re-weighted source distribution is minimized. Such a procedure is problematic, however, if the source and target distributions do not contain sufficient overlap. Huang et al. (2007); Long et al. (2015); Ganin & Lempitsky (2015) proposed to instead project both distributions into some feature space and encourage distribution matching in the feature space. Ganin & Lempitsky (2015) in particular encouraged feature matching via domain adversarial training, which corresponds approximately to Jensen-Shannon divergence minimization (Goodfellow et al., 2014). To better perform non-conservative domain adaptation, Saito et al. (2017) proposed to modify tri-training (Zhou & Li, 2005) for domain adaptation, leveraging the assumption that highly-confident predictions are correct predictions (Zhu, 2005). Several of aforementioned methods are based on Ben-David et al. (2010a)'s theoretical analysis of domain adaptation, which states the following,

Theorem 1 (Ben-David et al., 2010a) Let  $\mathcal{H}$  be the hypothesis space and let  $(X_s, \epsilon_s)$  and  $(X_t, \epsilon_t)$  be the two domains and their corresponding generalization error functions. Then for any  $h \in \mathcal{H}$ ,

$$
\epsilon_ {t} (h) \leq \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} \left(X _ {s}, X _ {t}\right) + \epsilon_ {s} (h) + \min  _ {h ^ {\prime} \in \mathcal {H}} \epsilon_ {t} \left(h ^ {\prime}\right) + \epsilon_ {s} \left(h ^ {\prime}\right), \tag {1}
$$

where  $d_{\mathcal{H}\Delta \mathcal{H}}$  denotes the  $\mathcal{H}\Delta \mathcal{H}$  -distance between the domains  $X_{s}$  and  $X_{t}$

$$
d _ {\mathcal {H} \Delta \mathcal {H}} = 2 \sup  _ {h, h ^ {\prime} \in \mathcal {H}} | \mathbb {E} _ {x \sim X _ {s}} [ h (x) \neq h ^ {\prime} (x) ] - \mathbb {E} _ {x \sim X _ {t}} [ h (x) \neq h ^ {\prime} (x) ] |. \tag {2}
$$

Intuitively,  $d_{\mathcal{H}\Delta \mathcal{H}}$  measures the extent to which small changes to the hypothesis in the source domain can lead to large changes in the target domain. It is evident that  $d_{\mathcal{H}\Delta \mathcal{H}}$  relates intimately to the complexity of the hypothesis space and the divergence between the source and target domains. For disjoint domains and infinite-capacity models,  $d_{\mathcal{H}\Delta \mathcal{H}}$  is maximal.

A critical component to our paper is the cluster assumption, which states that decision boundaries should not cross high-density regions (Chapelle & Zien, 2005). This assumption has been extensively studied and leveraged for semi-supervised learning, leading to proposals such as conditional entropy minimization (Grandvalet & Bengio, 2005) and pseudo-labeling (Lee, 2013). More recently, the cluster assumption has led to many successful deep semi-supervised learning algorithms such as semi-supervised generative adversarial networks (Dai et al., 2017), virtual adversarial training (Miyato et al., 2017), and self/temporal-ensembling (Laine & Aila, 2016; Tarvainen & Valpola, 2017). Given the success of the cluster assumption in semi-supervised learning, it is natural to consider its application to domain adaptation. Indeed, Ben-David & Urner (2014) formalized the cluster assumption through the lens of probabilistic Lipschitzness and proposed a nearest-neighbors model for domain adaptation. Our work extends this line of research by showing that the cluster assumption can be applied to deep neural networks to solve complex, high-dimensional domain adaptation problems. Independently of our work, French et al. (2017) demonstrated the application of self-ensembling to domain adaptation. However, our work additionally considers the application of the cluster assumption to non-conservative domain adaptation.

# 3 LIMITATION OF DOMAIN ADVERSARIAL TRAINING

Before describing our model, we first highlight that domain adversarial training may not be sufficient for domain adaptation if the feature extraction function has high-capacity. Consider a classifier  $h_{\theta}$ , parameterized by  $\theta$ , that maps inputs to the  $K$ -simplex, where  $K$  is the number of classes. Suppose the classifier  $h = g \circ f$  can be decomposed as the composite of an embedding function  $f: \mathcal{X} \to \mathcal{Z}$  and embedding classifier  $g: \mathcal{Z} \to \mathcal{Y}$ . For the source domain, let  $\mathcal{D}_s$  be the joint distribution over input  $x$  and one-hot label  $y$  and let  $X_s$  be the marginal input distribution.  $(\mathcal{D}_t, X_t)$  are analogously defined for the target domain. Let  $(\mathcal{L}_s, \mathcal{L}_d)$  be the loss functions

$$
\mathcal {L} _ {y} (\theta ; \mathcal {D} _ {s}) = \mathbb {E} _ {x, y \sim \mathcal {D} _ {s}} \left[ y ^ {\top} \ln h _ {\theta} (x) \right] \tag {3}
$$

$$
\mathcal {L} _ {d} (\theta ; \mathcal {D} _ {s}, \mathcal {D} _ {t}) = \sup  _ {D} \mathbb {E} _ {x \sim \mathcal {D} _ {s}} [ \ln D (f _ {\theta} (x)) ] + \mathbb {E} _ {x \sim \mathcal {D} _ {t}} [ \ln (1 - D (f _ {\theta} (x))) ], \tag {4}
$$

where the supremum ranges over discriminators  $D: \mathcal{Z} \to (0,1)$ . Then  $\mathcal{L}_y$  is the cross-entropy objective and  $D$  is a domain discriminator. Domain adversarial training minimizes the objective

$$
\min  _ {\theta} \mathcal {L} _ {y} (\theta ; \mathcal {D} _ {s}) + \lambda_ {d} \mathcal {L} _ {d} (\theta ; \mathcal {D} _ {s}, \mathcal {D} _ {t}), \tag {5}
$$

where  $\lambda_{d}$  is a weighting factor. Minimization of  $\mathcal{L}_d$  encourages the learning of a feature extractor  $f$  for which the Jensen-Shannon divergence between  $f(X_s)$  and  $f(X_t)$  is small.2 Ganin & Lempitsky (2015) suggest that successful adaptation tends to occur when the source generalization error and feature divergence are both small.

It is easy, however, to construct situations where this suggestion does not hold. In particular, if  $f$  has infinite-capacity and the source-target supports are disjoint, then  $f$  can employ arbitrary transformations to the target domain so as to match the source feature distribution. A formal statement and proof is provided in Appendix E. We verify empirically that, for sufficiently deep layers, jointly achieving small source generalization error and feature divergence does not imply high accuracy on the target task (Table 5). Given the inherent limitations of domain adversarial training, we wish to identify additional constraints that one can place on the model to achieve better, more reliable domain adaptation.

# 4 CONSTRAINING VIA CONDITIONAL ENTROPY MINIMIZATION

In this paper, we apply the cluster assumption to domain adaptation. The cluster assumption states that the input distribution  $X$  contains clusters and that points in the same cluster come from the same

![](images/5809942a829033ec42702debf64a66d93f9c93c8ee485ffe14357f99b08b4385.jpg)  
Figure 1: VADA improves upon domain adversarial training by additionally penalizing violations of the cluster assumption.

class. This assumption has been extensively studied and applied successfully to a wide range of classification tasks (see Section 2). If the cluster assumption holds, the optimal decision boundaries should occur far away from data-dense regions in the space of  $\mathcal{X}$  (Chapelle & Zien, 2005). Following Grandvalet & Bengio (2005), we achieve this behavior via minimization of the conditional entropy with respect to the target distribution,

$$
\mathcal {L} _ {c} (\theta ; \mathcal {D} _ {t}) = - \mathbb {E} _ {x \sim \mathcal {D} _ {t}} \left[ h _ {\theta} (x) ^ {\top} \ln h _ {\theta} (x) \right]. \tag {6}
$$

Intuitively, minimizing the conditional entropy forces the classifier to be confident on the unlabeled target data, thus driving the classifier's decision boundaries away from the target data (Grandvalet & Bengio, 2005). In practice, the conditional entropy must be empirically estimated using the available data. However, Grandvalet & Bengio (2005) note that this approximation breaks down if the classifier  $h$  is not locally-Lipschitz. Without the locally-Lipschitz constraint, the classifier is allowed to abruptly change its prediction in the vicinity of the training data points, which 1) results in a unreliable empirical estimate of conditional entropy and 2) allows placement of the classifier decision boundaries close to the training samples even when the empirical conditional entropy is minimized. To prevent this, we propose to explicitly incorporate the locally-Lipschitz constraint via virtual adversarial training (Miyato et al., 2017) and add to the objective function the additional term

$$
\mathcal {L} _ {v} (\theta ; \mathcal {D}) = \mathbb {E} _ {x \sim \mathcal {D}} \left[ \max  _ {\| r \| \leq \epsilon} \mathrm {D} _ {\mathrm {K L}} \left(h _ {\theta} (x) \| h _ {\theta} (x + r)\right) \right], \tag {7}
$$

which enforces classifier consistency within the norm-ball neighborhood of each sample  $x$ . Note that virtual adversarial training can be applied with respect to either the target or source distributions. We can combine the conditional entropy minimization objective and domain adversarial training to yield

$$
\min  _ {\theta} \mathcal {L} _ {y} (\theta ; \mathcal {D} _ {s}) + \lambda_ {d} \mathcal {L} _ {d} (\theta ; \mathcal {D} _ {s}, \mathcal {D} _ {t}) + \lambda_ {s} \mathcal {L} _ {v} (\theta ; \mathcal {D} _ {s}) + \lambda_ {t} \left[ \mathcal {L} _ {v} (\theta ; \mathcal {D} _ {t}) + \mathcal {L} _ {c} (\theta ; \mathcal {D} _ {t}) \right], \tag {8}
$$

a basic combination of domain adversarial training and semi-supervised training objectives. We refer to this as the Virtual Adversarial Domain Adaptation (VADA) model. Empirically, we observed that the hyperparameters  $(\lambda_d,\lambda_s,\lambda_t)$  are easy to choose and work well across multiple tasks (Appendix B).

$\mathcal{H}\Delta \mathcal{H}$ -Distance Minimization. VADA aligns well with the theory of domain adaptation provided in Theorem 1. Let the loss,

$$
\mathcal {L} _ {t} (\theta) = \mathcal {L} _ {v} (\theta ; \mathcal {D} _ {t}) + \mathcal {L} _ {c} (\theta ; D _ {t}), \tag {9}
$$

denote the degree to which the target-side cluster assumption is violated. Modulating  $\lambda_{t}$  enables VADA to trade-off between hypotheses with low target-side cluster assumption violation and hypotheses with low source-side generalization error. Setting  $\lambda_{t} > 0$  allows rejection of hypotheses with high target-side cluster assumption violation. By rejecting such hypotheses from the hypothesis space  $\mathcal{H}$ , VADA reduces  $d_{\mathcal{H}\Delta \mathcal{H}}$  and yields a tighter bound on the target generalization error. We verify empirically that VADA achieves significant improvements over existing models on multiple domain adaptation benchmarks (Table 1).

![](images/b893f745296f39c8d5d0e6ce7466de61baf4278be4299bd56b6fa6f0b6e45ef0.jpg)

![](images/b7bf1aa524086d50c46b4994f8e1dc5904048d058261674004e7ad5214128726.jpg)

![](images/c89dee17b8fb5cf6f8cce8acdfd2a85f8f939adc1739885d064119514fd26da3.jpg)

![](images/6186f550e19a4e447b4aab4644096e3072a1476be6ae9177af3ff40df0002201.jpg)  
VADA

![](images/8efe19e4981d2863f6a53a39b79f7c1206ef453dd2628506ac0586e5a7695483.jpg)  
DIRT-T  
Figure 2: DIRT-T uses VADA as initialization. After removing the source training signal, DIRT-T minimizes cluster assumption violation in the target domain through a series of natural gradient steps.

![](images/2421a1060cb9a651e41595c17ab1e8ac41fd997c52019755d5d078a8ca0fb66a.jpg)

# 5 DECISION-BOUNDARY ITERATIVE REFINEMENT TRAINING

In non-conservative domain adaptation, we account for the following inequality,

$$
\min  _ {h \in \mathcal {H}} \epsilon_ {t} (h) <   \epsilon_ {t} \left(h ^ {a}\right) \text {w h e r e} h ^ {a} = \underset {h \in \mathcal {H}} {\arg \min } \epsilon_ {s} (h) + \epsilon_ {t} (h), \tag {10}
$$

where  $(\epsilon_s, \epsilon_t)$  are generalization error functions for the source and target domains. This means that, for a given hypothesis class  $\mathcal{H}$ , the optimal classifier in the source domain does not coincide with the optimal classifier in the target domain.

We assume that the optimality gap in Eq. (10) results from violation of the cluster assumption. In other words, we suppose that any source-optimal classifier drawn from our hypothesis space necessarily violates the cluster assumption in the target domain. Insofar as VADA is trained on the source domain, we hypothesize that a better hypothesis is achievable by introducing a secondary training phase that solely minimizes the target-side cluster assumption violation.

Under this assumption, the natural solution is to initialize with the VADA model and then further minimize the cluster assumption violation in the target domain. In particular, we first use VADA to learn an initial classifier  $h_{\theta_0}$ . Next, we incrementally push the classifier's decision boundaries away from data-dense regions by minimizing the target-side cluster assumption violation loss  $\mathcal{L}_t$  in Eq. (9). We denote this procedure Decision-boundary Iterative Refinement Training (DIRT).

# 5.1 DECISION-BOUNDARY ITERATIVE REFINEMENT TRAINING WITH A TEACHER

Stochastic gradient descent minimizes the loss  $\mathcal{L}_t$  by selecting gradient steps  $\Delta \theta$  according to the following objective,

$$
\min  _ {\Delta \theta} \mathcal {L} _ {t} (\theta + \Delta \theta) \tag {11}
$$

$$
\text {s . t .} \| \Delta \theta \| \leq \epsilon , \tag {12}
$$

which defines the neighborhood in the parameter space. This notion of neighborhood is sensitive to the parameterization of the model; depending on the parameterization, a seemingly small step  $\Delta \theta$  may result in a vastly different classifier. This contradicts our intention of incrementally and locally pushing the decision boundaries to a local conditional entropy minimum, which requires that the decision boundaries of  $h_{\theta + \Delta \theta}$  stay close to that of  $h_\theta$ . It is therefore important to define a neighborhood that is parameterization-invariant. Following Pascanu & Bengio (2013), we instead select  $\Delta \theta$  using the following objective,

$$
\min _ {\Delta \theta} \mathcal {L} _ {t} (\theta + \Delta \theta)
$$

$$
\mathrm {s . t .} \mathbb {E} _ {x \sim D _ {t}} \left[ \mathrm {D} _ {\mathrm {K L}} \left(h _ {\theta} (x) \| h _ {\theta + \Delta \theta} (x)\right) \right] \leq \epsilon . \tag {13}
$$

Each optimization step now solves for a gradient step  $\Delta \theta$  that minimizes the conditional entropy, subject to the constraint that the Kullback-Leibler divergence between  $h_\theta(x)$  and  $h_{\theta + \Delta \theta}(x)$  is small for  $x \sim \mathcal{X}_t$ . The corresponding Lagrangian suggests that one can instead minimize a sequence of optimization problems

$$
\min  _ {\theta_ {n}} \lambda_ {t} \mathcal {L} _ {t} \left(\theta_ {n}\right) + \beta_ {t} \mathbb {E} \left[ \mathrm {D} _ {\mathrm {K L}} \left(h _ {\theta_ {n - 1}} (x) \| h _ {\theta_ {n}} (x)\right) \right], \tag {14}
$$

that approximates the application of a series of natural gradient steps.

In practice, each of optimization problems in Eq. (14) can be solved approximately via a finite number of stochastic gradient descent steps. We denote the number of steps taken to be the refinement interval  $B$ . Similar to Tarvainen & Valpola (2017), we use the Adam Optimizer with Polyak averaging (Polyak & Juditsky, 1992). We interpret  $h_{\theta_{n-1}}$  as a (sub-optimal) teacher for the student model  $h_{\theta_n}$ , which is trained to stay close to the teacher model while seeking to reduce the cluster assumption violation. As a result, we denote this model as Decision-boundary Iterative Refinement Training with a Teacher (DIRT-T).

Weakly-Supervised Learning. This sequence of optimization problems has a natural interpretation that exposes a connection to weakly-supervised learning. In each optimization problem, the teacher model  $h_{\theta_{n-1}}$  pseudo-labels the target samples with noisy labels. Rather than naively training the student model  $h_{\theta_n}$  on the noisy labels, the additional training signal  $\mathcal{L}_t$  allows the student model to place its decision boundaries further from the data. If the clustering assumption holds and the initial noisy labels are sufficiently similar to the true labels, conditional entropy minimization can improve the placement of the decision boundaries (Reed et al., 2014).

Domain Adaptation. An alternative interpretation is that DIRT-T is the recursive extension of VADA, where the act of pseudo-labeling of the target distribution constructs a new "source" domain (i.e. target distribution  $X_{t}$  with pseudo-labels). The sequence of optimization problems can then be seen as a sequence of non-conservative domain adaptation problems in which  $X_{s} = X_{t}$  but  $p_{s}(y\mid x)\neq p_{t}(y\mid x)$ , where  $p_{s}(y\mid x) = h_{\theta_{n - 1}}(x)$  and  $p_t(y\mid x)$  is the true conditional label distribution in the target domain. Since  $d_{\mathcal{H}\Delta \mathcal{H}}$  is strictly zero in this sequence of optimization problems, domain adversarial training is no longer necessary. Furthermore, if  $\mathcal{L}_t$  minimization does improve the student classifier, then the gap in Eq. (10) should get smaller each time the source domain is updated.

# 6 EXPERIMENTS

In principle, our method can be applied to any domain adaptation tasks so long as one can define a reasonable notion of neighborhood for virtual adversarial training (Miyato et al., 2016). For comparison against Saito et al. (2017) and French et al. (2017), we focus on visual domain adaptation and evaluate on MNIST, MNIST-M, Street View House Numbers (SVHN), Synthetic Digits (SYN DIGITS), Synthetic Traffic Signs (SYN SIGNS), the German Traffic Signs Recognition Benchmark (GTSRB), CIFAR-10, and STL-10. For non-visual domain adaptation, we evaluate on Wi-Fi activity recognition.

# 6.1 IMPLEMENTATION DETAIL

Architecture We use a small CNN for the digits, traffic sign, and Wi-Fi domain adaptation experiments, and a larger CNN for domain adaptation between CIFAR-10 and STL-10. Both architectures are available in Appendix A. For fair comparison, we additionally report the performance of source-only baseline models and demonstrate that the significant improvements are attributable to our proposed method.

Replacing gradient reversal. In contrast to Ganin & Lempitsky (2015), which proposed to implement domain adversarial training via gradient reversal, we follow the suggestion in Goodfellow et al. (2014) and instead optimize via alternating updates to the discriminator and encoder. Further details are provided in Appendix C

Instance normalization. We explored the application of instance normalization as a pre-processing step to the input image. This procedure makes the classifier invariant to channel-wide shifts and

rescaling of pixel intensities. A discussion of instance normalization for domain adaptation is provided in Appendix D. We show in Figure 3 the effect of applying instance normalization to the input image.

![](images/e4897ca79c9838194df69bfcd14fe763ddd539d1553e9e0d414fc64a0b6246a1.jpg)  
Figure 3: Effect of applying instance normalization to the input image. In clockwise direction: MNIST-M, GTSRB, SVHN, and CIFAR-10. In each quadrant, the top row is the original image, and the bottom row is the instance-normalized image.

Hyperparameters. For each task, we tuned the four hyperparameters  $(\lambda_d,\lambda_s,\lambda_t,\beta)$  by randomly selecting 1000 labeled target samples from the training set and using that as our validation set. We observed that extensive hyperparameter-tuning is not necessary to achieve state-of-the-art performance. In all experiments with instance-normalized inputs, we restrict our hyperparameter search for each task to  $\lambda_{d} = \{0,10^{-2}\} ,\lambda_{s} = \{0,1\} ,\lambda_{t} = \{10^{-2},10^{-1}\}$ . We fixed  $\beta = 10^{-2}$ . Note that the decision to turn  $(\lambda_d,\lambda_s)$  on or off that can often be determined a priori. A complete list of the hyperparameters is provided in Appendix B.

# 6.2 MODEL EVALUATION

<table><tr><td>Source Target</td><td>MNIST MNIST-M</td><td>SVHN MNIST</td><td>MNIST SVHN</td><td>DIGITS SVHN</td><td>SIGNS GTSRB</td><td>CIFAR STL</td><td>STL CIFAR</td></tr><tr><td>MMD (Long et al., 2015)</td><td>76.9</td><td>71.1</td><td>-</td><td>88.0</td><td>91.1</td><td>-</td><td>-</td></tr><tr><td>DANN (Ganin &amp; Lempitsky, 2015)</td><td>81.5</td><td>71.1</td><td>35.7</td><td>90.3</td><td>88.7</td><td>-</td><td>-</td></tr><tr><td>DRCN (Ghifary et al., 2016)</td><td>-</td><td>82.0</td><td>40.1</td><td>-</td><td>-</td><td>66.4</td><td>58.7</td></tr><tr><td>DSN (Bousmalis et al., 2016b)</td><td>83.2</td><td>82.7</td><td>-</td><td>91.2</td><td>93.1</td><td>-</td><td>-</td></tr><tr><td>kNN-Ad (Sener et al., 2016)</td><td>86.7</td><td>78.8</td><td>40.3</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PixelDA (Bousmalis et al., 2016a)</td><td>98.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ATT (Saito et al., 2017)</td><td>94.2</td><td>86.2</td><td>52.8</td><td>92.9</td><td>96.2</td><td>-</td><td>-</td></tr><tr><td>II-model (aug) (French et al., 2017)</td><td>-</td><td>92.0</td><td>71.4</td><td>94.2</td><td>98.4</td><td>76.3</td><td>64.2</td></tr><tr><td colspan="8">Without Instance-Normalized Input:</td></tr><tr><td>Source-Only</td><td>58.5</td><td>77.0</td><td>27.9</td><td>86.9</td><td>79.6</td><td>76.3</td><td>63.6</td></tr><tr><td>VADA</td><td>97.7</td><td>97.9</td><td>47.5</td><td>94.8</td><td>98.8</td><td>80.0</td><td>73.5</td></tr><tr><td>DIRT-T</td><td>98.9</td><td>99.4</td><td>54.5</td><td>96.1</td><td>99.5</td><td>-</td><td>75.3</td></tr><tr><td colspan="8">With Instance-Normalized Input:</td></tr><tr><td>Source-Only</td><td>59.9</td><td>82.4</td><td>40.9</td><td>88.6</td><td>86.2</td><td>77.0</td><td>62.6</td></tr><tr><td>VADA</td><td>95.7</td><td>94.5</td><td>73.3</td><td>94.9</td><td>99.2</td><td>78.3</td><td>71.4</td></tr><tr><td>DIRT-T</td><td>98.7</td><td>99.4</td><td>76.5</td><td>96.2</td><td>99.6</td><td>-</td><td>73.3</td></tr></table>

Table 1: Results of the domain adaptation experiments. In all settings, both VADA and DIRT-T achieve state-of-the-art performance in all settings.  

<table><tr><td>Source Target</td><td>MNIST MNIST-M</td><td>SVHN MNIST</td><td>MNIST SVHN</td><td>DIGITS SVHN</td><td>SIGN GTSRB</td><td>CIFAR STL</td><td>STL CIFAR</td></tr><tr><td>ATT</td><td>37.1</td><td>16.1</td><td>17.9</td><td>9.0</td><td>20.5</td><td>-</td><td>-</td></tr><tr><td>II-model (aug)</td><td>-</td><td>3.7</td><td>18.1</td><td>10.6</td><td>1.0</td><td>4.5</td><td>7.4</td></tr><tr><td>DIRT-T</td><td>40.4</td><td>22.4</td><td>26.6</td><td>9.2</td><td>19.9</td><td>-</td><td>11.7</td></tr><tr><td>DIRT-T (W.I.N.I.)</td><td>38.8</td><td>17.0</td><td>35.6</td><td>7.6</td><td>13.4</td><td>-</td><td>10.7</td></tr></table>

Table 2: Additional comparison of the margin of improvement computed by taking the reported performance of each model and subtracting the reported source-only performance in the respective papers. W.I.N.I. indicates "with instance-normalized input."

MNIST  $\rightarrow$  MNIST-M. We first evaluation the adaptation from MNIST to MNIST-M. MNIST-M is constructed by blending MNIST digits with random color patches from the BSDS500 dataset.

MNIST  $\leftrightarrow$  SVHN. The distribution shift is exacerbated when adapting between MNIST and SVHN. Whereas MNIST consists of black-and-white handwritten digits, SVHN consists of crops of colored,

street house numbers. Because MNIST has a significantly lower intrinsic dimensionality that SVHN, the adaptation from MNIST  $\rightarrow$  SVHN is especially challenging when the input is not pre-processed via instance normalization. When instance normalization is applied, we achieve a strong state-of-the-art performance  $76.5\%$  and an equally impressive margin-of-improvement over source-only of  $35.6\%$ . Interestingly, by reducing the refinement interval  $B$  and taking noisier natural gradient steps, we were occasionally able to achieve accuracies as high as  $87\%$ . However, due to the high-variance associated with this, we omit reporting this configuration in Table 1.

SYN DIGITS  $\rightarrow$  SVHN. The adaptation from SYN DIGITS  $\rightarrow$  SVHN reflect a common adaptation problem of transferring from synthetic images to real images. The SYN DIGITS dataset consists of 500000 images generated from Windows fonts by varying the text, positioning, orientation, background, stroke color, and the amount of blur.

SYN SIGNS  $\rightarrow$  GTSRB. This setting provides an additional demonstration of adapting from synthetic images to real images. Unlike SYN DIGITS  $\rightarrow$  SVHN, SYN SIGNS  $\rightarrow$  GTSRB contains 43 classes instead of 10.

STL  $\leftrightarrow$  CIFAR. Both STL-10 and CIFAR-10 are 10-class image datasets. These two datasets contain nine overlapping classes. Following the procedure in French et al. (2017), we removed the non-overlapping classes ("frog" and "monkey") and reduce to a 9-class classification problem. We achieve state-of-the-art performance in both adaptation directions. In STL  $\rightarrow$  CIFAR, we achieve a  $11.7\%$  margin-of-improvement and a performance accuracy of  $73.3\%$ . Note that because STL-10 contains a very small training set, it is difficult to estimate the conditional entropy, thus making DIRT-T unreliable for CIFAR  $\rightarrow$  STL.

<table><tr><td>Source
Target</td><td>Room A
Room B</td></tr><tr><td colspan="2">With Instance-Normalized Input:</td></tr><tr><td>Source-Only</td><td>35.7</td></tr><tr><td>DANN</td><td>38.0</td></tr><tr><td>VADA</td><td>53.0</td></tr><tr><td>DIRT-T</td><td>53.0</td></tr></table>

Table 3: Results of the domain adaptation experiments on Wi-Fi Activity Recognition Task

Wi-Fi Activity Recognition. To evaluate the performance of our models on a non-visual domain adaptation task, we applied VADA and DIRT-T to the Wi-Fi Activity Recognition Dataset (Yousefi et al., 2017). The Wi-Fi Activity Recognition Dataset is a classification task that takes the Wi-Fi Channel State Information (CSI) data stream as input  $x$  to predict motion activity within an indoor area as output  $y$ . Domain adaptation is necessary when the training and testing data are collected from different rooms, which we denote as Rooms A and B. Table 3 shows that VADA significantly improves classification accuracy compared to Source-Only and DANN. However, DIRT-T does not lead to further improvements on this dataset. We perform experiments in Appendix F which suggests that VADA already achieves strong clustering in the target domain for this dataset, and therefore DIRT-T is not expected to yield further performance improvement.

Overall. We achieve state-of-the-art results across all tasks. For a fairer comparison against ATT and the  $\Pi$ -model, Table 2 provides the improvement margin over the respective source-only performance reported in each paper. In four of the tasks (MNIST  $\rightarrow$  MNIST-M, SVHN  $\rightarrow$  MNIST, MNIST  $\rightarrow$  SVHN, STL  $\rightarrow$  CIFAR), we achieve substantial margin of improvement compared to previous models. In the remaining three tasks, our improvement margin over the source-only model is competitive against previous models. Our closest competitor is the  $\Pi$ -model. However, unlike the  $\Pi$ -model, we do not perform data augmentation.

It is worth noting that DIRT-T consistently improves upon VADA. Since DIRT-T operates by incrementally pushing the decision boundaries away from the target domain data, it relies heavily on the cluster assumption. DIRT-T's empirical success therefore demonstrates the effectiveness of leveraging the cluster assumption in unsupervised domain adaptation with deep neural networks.

# 6.3 ANALYSIS OF VADA AND DIRT-T

# 6.3.1 ROLE OF VIRTUAL ADVERSARIAL TRAINING

To study the relative contribution of the virtual adversarial training in the VADA and DIRT-T objectives (Eq. (8) and Eq. (14) respectively), we perform an extensive ablation analysis in Table 4. The removal of the virtual adversarial training component is denoted by the "no-vat" subscript. Our results show that  $\mathrm{VADAn_{no - vat}}$  is sufficient for out-performing DANN in all but one task. The further ability for DIRT- $\mathrm{T_{no - vat}}$  to improve upon  $\mathrm{VADAn_{no - vat}}$  demonstrates the effectiveness of conditional entropy minimization. Ultimately, in six of the seven tasks, both virtual adversarial training and conditional entropy minimization are essential for achieving the best performance. The empirical importance of incorporating virtual adversarial training shows that the locally-Lipschitz constraint is beneficial for pushing the classifier decision boundaries away from data.

<table><tr><td>Source Target</td><td>MNIST MNIST-M</td><td>SVHN MNIST</td><td>MNIST SVHN</td><td>DIGITS SVHN</td><td>SIGNS GTSRB</td><td>CIFAR STL</td><td>STL CIFAR</td></tr><tr><td colspan="8">With Instance-Normalized Input:</td></tr><tr><td>Source-Only</td><td>59.9</td><td>82.4</td><td>40.9</td><td>88.6</td><td>86.2</td><td>77.0</td><td>62.6</td></tr><tr><td>DANN (our implementation)</td><td>94.6</td><td>68.3</td><td>60.6</td><td>90.1</td><td>97.5</td><td>78.1</td><td>62.7</td></tr><tr><td>VADAno-vat</td><td>93.8</td><td>83.1</td><td>66.8</td><td>93.4</td><td>98.4</td><td>79.1</td><td>68.6</td></tr><tr><td>VADAno-vat → DIRT-Tno-vat</td><td>94.8</td><td>96.3</td><td>68.6</td><td>94.4</td><td>99.1</td><td>-</td><td>69.2</td></tr><tr><td>VADAno-vat → DIRT-T</td><td>98.3</td><td>99.4</td><td>69.8</td><td>95.3</td><td>99.6</td><td>-</td><td>71.0</td></tr><tr><td>VADA</td><td>95.7</td><td>94.5</td><td>73.3</td><td>94.9</td><td>99.2</td><td>78.3</td><td>71.4</td></tr><tr><td>VADA → DIRT-T</td><td>98.7</td><td>99.4</td><td>76.5</td><td>96.2</td><td>99.6</td><td>-</td><td>73.3</td></tr></table>

Table 4: Results of ablation experiment, starting from the DANN model. The "no-vat" subscript denote models where the virtual adversarial training component is removed.

# 6.3.2 ROLE OF TEACHER MODEL IN DIRT-T

![](images/06ab6b875218294d5bcb5c3de716be95e442f7b51d258d1b41723c514bcd6a58.jpg)

![](images/c1dca1c9848df1405e5cf188134bb0b27d58c0e14fdbb0cc7acb0d0cf7210256.jpg)

![](images/69796eb7c855ff39070e1085ebd0b06f6761d3a6e0664630a0aa001a58c0f012.jpg)  
(a) SVHN  $\rightarrow$  MNIST  
(b) STL  $\rightarrow$  CIFAR

![](images/54a7fd902f83d58135eaa8a1d4717f52407af5285419e2c0cc38f1153a3fd916.jpg)  
Figure 4: Comparing model behavior with and without the application of the KL-term. At iteration 0, we begin with the VADA initialization and apply the DIRT-T algorithm.

When considering Eq. (14), it is natural to ask whether defining the neighborhood with respect to the classifier is truly necessary. In Figure 4, we demonstrate in SVHN  $\rightarrow$  MNIST and STL  $\rightarrow$  CIFAR that removal of the KL-term negatively impacts the model. Since the MNIST data manifold is low-dimensional and contains easily identifiable clusters, applying naive gradient descent (Eq. (12)) can also boost the test accuracy during initial training. However, without the KL constraint, the classifier can sometimes deviate significantly from the neighborhood of the previous classifier, and the resulting spikes in the KL-term correspond to sharp drops in target test accuracy. In STL  $\rightarrow$  CIFAR, where the data manifold is much more complex and contains less obvious clusters, naive gradient descent causes immediate decline in the target test accuracy.

# 6.3.3 VISUALIZATION OF REPRESENTATION

![](images/39aadb6bdd922a774bad500b978061d5d3aa77291c158cbf4b6c0ba5b8704b1c.jpg)  
(a) Source-Only

![](images/035c799ebb18d4658293ddab2e9279ea8fda53a8c7ec417c5f5bee6014f51558.jpg)  
(b) VADA  
Figure 5: T-SNE plot of the last hidden layer for MNIST (blue)  $\rightarrow$  SVHN (red). We used the model without instance normalization to highlight the further improvement that DIRT-T provides.

![](images/f18e5ecaf6e418db8b75437f9f3f2ffc9590beb0d9f189b78658d41002d47eec.jpg)  
(c)DIRT-T

We further analyze the behavior of VADA and DIRT-T by showing T-SNE embeddings of the last hidden layer of the model trained to adapt from MNIST  $\rightarrow$  SVHN. In Figure 5, source-only training shows strong clustering of the MNIST samples (blue) and performs poorly on SVHN (red). VADA offers significant improvement and exhibits signs of clustering on SVHN. DIRT-T begins with the VADA initialization and further enhances the clustering, resulting in the best performance on MNIST  $\rightarrow$  SVHN.

# 6.4 DOMAIN ADVERSARIAL TRAINING: LAYER ABLATION

<table><tr><td>Layer</td><td>JSD ≥</td><td>DANN Source Accuracy</td><td>Target Accuracy</td><td>JSD ≥</td><td>VADA Source Accuracy</td><td>Target Accuracy</td></tr><tr><td>L-0</td><td>0.001</td><td>78.0</td><td>24.7</td><td>0.001</td><td>24.9</td><td>18.4</td></tr><tr><td>L-1</td><td>0.002</td><td>98.6</td><td>35.0</td><td>0.007</td><td>12.0</td><td>11.6</td></tr><tr><td>L-2</td><td>0.353</td><td>16.4</td><td>10.3</td><td>0.383</td><td>11.5</td><td>9.9</td></tr><tr><td>L-3</td><td>0.036</td><td>94.8</td><td>33.8</td><td>0.034</td><td>67.8</td><td>37.1</td></tr><tr><td>L-4</td><td>0.012</td><td>97.0</td><td>40.0</td><td>0.020</td><td>96.8</td><td>61.5</td></tr><tr><td>L-5</td><td>0.235</td><td>99.3</td><td>57.9</td><td>0.244</td><td>99.4</td><td>73.3</td></tr><tr><td>L-6</td><td>0.486</td><td>99.2</td><td>60.3</td><td>0.509</td><td>99.3</td><td>70.4</td></tr><tr><td>L-7</td><td>0.644</td><td>99.0</td><td>52.5</td><td>0.608</td><td>99.1</td><td>70.5</td></tr></table>

Table 5: Comparison of model behavior when domain adversarial training is applied to various layers. We denote the very last (simplex) layer of the neural network as  $L$  and ablatively domain adversarial training to the last eight layers. A lower bound on the Jensen-Shannon Divergence is computed by training a logistic regression model to predict domain origin when given the layer embeddings.

In Table 5, we applied domain adversarial training to various layers of a Domain Adversarial Neural Network (Ganin & Lempitsky, 2015) trained to adapt MNIST  $\rightarrow$  SVHN. With the exception of layers  $L - 2$  and  $L - 0$ , which experienced training instability, the general observation is that as the layer gets deeper, the additional capacity of the corresponding embedding function allows better matching of the source and target distributions without hurting source generalization accuracy. This demonstrates that the combination of low divergence and high source accuracy does not imply better adaptation to the target domain. Interestingly, when the classifier is regularized to be locally-Lipschitz via VADA, the combination of low divergence and high source accuracy appears to correlate more strongly with better adaptation.

# 7 CONCLUSION

In this paper, we presented two novel models for domain adaptation inspired by the cluster assumption. Our first model, VADA, performs domain adversarial training with an added term that penalizes violations of the cluster assumption. Our second model, DIRT-T, is an extension of VADA that recursively refines the VADA classifier by untethering the model from the source training signal and applying approximate natural gradients to further minimize the cluster assumption violation. Our experiments demonstrate the effectiveness of the cluster assumption: VADA achieves strong

performance across several domain adaptation benchmarks, and DIRT-T further improves VADA performance. Our proposed models open up several possibilities for future work. One possibility is to apply DIRT-T to weakly supervised learning; another is to improve the natural gradient approximation via K-FAC (Martens & Grosse, 2015) and PPO (Schulman et al., 2017). Given the strong performance of our models, we also recommend them for other downstream domain adaptation applications.

# REFERENCES

Shai Ben-David and Ruth Urner. Domain adaptation-can quantity compensate for quality? Annals of Mathematics and Artificial Intelligence, 70(3):185-202, 2014.  
Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine learning, 79(1):151-175, 2010a.  
Shai Ben-David, Tyler Lu, Teresa Luu, and David Pál. Impossibility theorems for domain adaptation. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 129-136, 2010b.  
Konstantinos Bousmalis, Nathan Silberman, David Dohan, Dumitru Erhan, and Dilip Krishnan. Unsupervised pixel-level domain adaptation with generative adversarial networks. arXiv preprint arXiv:1612.05424, 2016a.  
Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, and Dumitru Erhan. Domain separation networks. In Advances in Neural Information Processing Systems, pp. 343-351, 2016b.  
Olivier Chapelle and Alexander Zien. Semi-supervised classification by low density separation. In AISTATS, pp. 57-64, 2005.  
Zihang Dai, Zhilin Yang, Fan Yang, William W Cohen, and Ruslan Salakhutdinov. Good semi-supervised learning that requires a bad gan. arXiv preprint arXiv:1705.09783, 2017.  
William Fedus, Mihaela Rosca, Balaji Lakshminarayanan, Andrew M Dai, Shakir Mohamed, and Ian Goodfellow. Many paths to equilibrium: Gans do not need to decrease advergence at every step. arXiv preprint arXiv:1710.08446, 2017.  
Geoffrey French, Michal Mackiewicz, and Mark Fisher. Self-ensembling for domain adaptation. arXiv preprint arXiv:1706.05208, 2017.  
Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In International Conference on Machine Learning, pp. 1180-1189, 2015.  
Muhammad Ghifary, W Bastiaan Kleijn, Mengjie Zhang, David Balduzzi, and Wen Li. Deep reconstruction-classification networks for unsupervised domain adaptation. In European Conference on Computer Vision, pp. 597-613. Springer, 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Yves Grandvalet and Yoshua Bengio. Semi-supervised learning by entropy minimization. In Advances in neural information processing systems, pp. 529-536, 2005.  
Jiayuan Huang, Arthur Gretton, Karsten M Borgwardt, Bernhard Scholkopf, and Alex J Smola. Correcting sample selection bias by unlabeled data. In Advances in neural information processing systems, pp. 601-608, 2007.  
Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. arXiv preprint arXiv:1610.02242, 2016.

Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on Challenges in Representation Learning, ICML, volume 3, pp. 2, 2013.  
Mingsheng Long, Yue Cao, Jianmin Wang, and Michael Jordan. Learning transferable features with deep adaptation networks. In International Conference on Machine Learning, pp. 97-105, 2015.  
Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation: Learning bounds and algorithms. arXiv preprint arXiv:0902.3430, 2009.  
James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International Conference on Machine Learning, pp. 2408-2417, 2015.  
Takeru Miyato, Andrew M Dai, and Ian Goodfellow. Virtual adversarial training for semi-supervised text classification. stat, 1050:25, 2016.  
Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. arXiv preprint arXiv:1704.03976, 2017.  
Razvan Pascanu and Yoshua Bengio. Revisiting natural gradient for deep networks. arXiv preprint arXiv:1301.3584, 2013.  
Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization, 30(4):838-855, 1992.  
Scott Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, and Andrew Rabinovich. Training deep neural networks on noisy labels with bootstrapping. arXiv preprint arXiv:1412.6596, 2014.  
Kuniaki Saito, Yoshitaka Ushiku, and Tatsuya Harada. Asymmetric tri-training for unsupervised domain adaptation. arXiv preprint arXiv:1702.08400, 2017.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Ozan Sener, Hyun Oh Song, Ashutosh Saxena, and Silvio Savarese. Learning transferrable representations for unsupervised domain adaptation. In Advances in Neural Information Processing Systems, pp. 2110-2118, 2016.  
Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference, 90(2):227-244, 2000.  
Baochen Sun and Kate Saenko. From virtual to reality: Fast adaptation of virtual object detectors to real domains. In BMVC, volume 1, pp. 3, 2014.  
Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. 2017.  
Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016.  
David Vazquez, Antonio M Lopez, Javier Marin, Daniel Ponsa, and David Geronimo. Virtual and real world adaptation for pedestrian detection. IEEE transactions on pattern analysis and machine intelligence, 36(4):797-809, 2014.  
Siamak Yousefi, Hirokazu Narui, Sankalp Dayal, Stefano Ermon, and Shahrokh Valaee. A survey on behavior recognition using wifi channel state information. IEEE Communications Magazine, 55(10):98-104, 2017.  
Zhi-Hua Zhou and Ming Li. Tri-training: Exploiting unlabeled data using three classifiers. IEEE Transactions on knowledge and Data Engineering, 17(11):1529-1541, 2005.  
Xiaojin Zhu. Semi-supervised learning literature survey. 2005.

A ARCHITECTURES  

<table><tr><td>Layer Index</td><td>Small CNN</td><td>Large CNN</td></tr><tr><td>L-18</td><td colspan="2">32 x 32 x 3 Image</td></tr><tr><td>L-17</td><td colspan="2">Instance Normalization (optional)</td></tr><tr><td>L-16</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 96 lReLU</td></tr><tr><td>L-15</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 96 lReLU</td></tr><tr><td>L-14</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 96 lReLU</td></tr><tr><td>L-13</td><td colspan="2">2 x 2 max-pool, stride 2</td></tr><tr><td>L-12</td><td colspan="2">dropout, p = 0.5</td></tr><tr><td>L-11</td><td colspan="2">gaussian dropout, σ = 1</td></tr><tr><td>L-10</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 192 lReLU</td></tr><tr><td>L-9</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 192 lReLU</td></tr><tr><td>L-8</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 192 lReLU</td></tr><tr><td>L-7</td><td colspan="2">2 x 2 max-pool, stride 2</td></tr><tr><td>L-6</td><td colspan="2">dropout, p = 0.5</td></tr><tr><td>L-5</td><td colspan="2">gaussian dropout, σ = 1</td></tr><tr><td>L-4</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 192 lReLU</td></tr><tr><td>L-3</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 192 lReLU</td></tr><tr><td>L-2</td><td>3 x 3 conv. 64 lReLU</td><td>3 x 3 conv. 192 lReLU</td></tr><tr><td>L-1</td><td colspan="2">global average pool</td></tr><tr><td>L-0</td><td colspan="2">10 dense, softmax</td></tr></table>

Table 6: Small and Large CNN architectures. Leaky ReLU parameter  $a = 0.1$ . All convolutional and dense layers in the classifier are pre-activation batch-normalized. All images are resized to  $32x32x3$ . Note the use of Gaussian dropout: this addition was motivated by initial experiments in which we observed that domain adversarial training appears to contract the feature space.  

<table><tr><td>Domain Discriminator</td></tr><tr><td>Layer L - 5 Output</td></tr><tr><td>100 dense, ReLU</td></tr><tr><td>1 dense, sigmoid</td></tr></table>

Table 7: Domain discriminator architecture.
