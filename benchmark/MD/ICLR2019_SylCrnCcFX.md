# TOWARDS ROBUST, LOCALLY LINEAR DEEP NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep networks realize complex mappings that are often understood by their locally linear behavior around or at points of interest. For example, we use the derivative of the mapping with respect to its inputs for sensitivity analysis, or to explain (obtain coordinate relevance for) a prediction. One key challenge is that such derivates are themselves inherently unstable. In this paper, we propose a new learning problem to encourage deep networks to have stable derivatives over larger regions. While the problem is challenging in general, we focus on networks with piecewise linear activation functions. Our algorithm consists of an inference step that identifies a region around a point where linear approximation is provably stable, and an optimization step to expand such regions. We propose a novel relaxation to scale the algorithm to realistic models. We illustrate our method with residual and recurrent networks on image and sequence datasets.

# 1 INTRODUCTION

Complex mappings are often characterized by their derivatives at points of interest. Such derivatives with respect to the inputs play key roles across learning problems, including sensitivity analysis. The associated local linearization is frequently used to obtain explanations for model predictions (Baehrens et al., 2010; Simonyan et al., 2013; Sundararajan et al., 2017; Smilkov et al., 2017); explicit first-order local approximations (Rifai et al., 2012; Goodfellow et al., 2014; Wang & Liu, 2016; Koh & Liang, 2017; Alvarez-Melis & Jaakkola, 2018b); or used to guide learning through regularization of functional classes controlled by derivatives (Gulrajani et al., 2017; Bellemare et al., 2017; Mroueh et al., 2018). We emphasize that the derivatives discussed in this paper are with respect to the input coordinates rather than parameters.

The key challenge lies in the fact that derivatives of functions parameterized by deep learning models are not stable in general (Ghorbani et al., 2017). State-of-the-art deep learning models (He et al., 2016; Huang et al., 2017) are typically over-parametrized (Zhang et al., 2017), leading to unstable functions as a by-product. The instability is reflected in both the function values (Goodfellow et al., 2014) as well as the derivatives (Ghorbani et al., 2017; Alvarez-Melis & Jaakkola, 2018a). Due to unstable derivatives, first-order approximations used for explanations therefore also lack robustness (Ghorbani et al., 2017; Alvarez-Melis & Jaakkola, 2018a).

We note that gradient stability is a notion different from adversarial examples. A stable gradient can be large or small, so long as it remains approximately invariant within a local region. Adversarial examples, on the other hand, are small perturbations of the input that change the predicted output (Goodfellow et al., 2014). A large local gradient, whether stable or not in our sense, is likely to contribute to finding an adversarial example. Robust estimation techniques used to protect against adversarial examples (e.g., (Madry et al., 2017)) focus on stable function values rather than stable gradients but can nevertheless indirectly impact (potentially help) gradient stability. A direct extension of robust estimation to ensure gradient stability would involve finding maximally distorted derivatives and require access to approximate Hessian of deep networks.

In this paper, we focus on deep networks with piecewise linear activations to make the problem tractable. The special structure of this class of networks (functional characteristics) allows us to infer lower bounds on the  $\ell_p$  margin — the maximum radius of  $\ell_p$ -norm balls around a point where derivatives are provably stable. In particular, we investigate the special case of  $p = 2$  since the lower bound has an analytical solution, and permits us to formulate a regularization problem to maximize

it. The resulting objective is, however, rigid and non-smooth, and we further relax the learning problem in a manner resembling (locally) support vector machines (SVM) (Vapnik, 1995; Cortes & Vapnik, 1995).

Both the inference and learning problems in our setting require evaluating the gradient of each neuron with respect to the inputs which poses a significant computational challenge. For piecewise linear networks, given  $D$ -dimensional data, we propose a novel perturbation algorithm that collects all the exact gradients by means of forward propagating  $O(D)$  carefully crafted samples in parallel without any back-propagation. When the GPU memory cannot fit  $O(D)$  samples in one batch, we develop an unbiased approximation to the objective with a random subset of such samples.

Empirically, we examine our inference and learning algorithms with fully-connected (FC), residual (ResNet) (He et al., 2016), and recurrent (RNN) networks on image and time-series datasets with quantitative and qualitative experiments. The main contributions of this work are as follows:

- Inference algorithms that identify input regions of neural networks, with piecewise linear activation functions, that are provably stable.  
- A novel learning criterion that effectively expand regions of provably stable derivatives.  
- Novel perturbation algorithms that scale computation to high dimensional data.  
- Empirical evaluation with several types of networks.

# 2 RELATED WORK

For tractability reasons, we focus in this paper on neural networks with piecewise linear activation functions, such as ReLU (Glorot et al., 2011) and its variants (Maas et al., 2013; He et al., 2015; Arjovsky et al., 2016). Since the nonlinear behavior of deep models is mostly governed by the activation function, a neural network defined with affine transformations and piecewise linear activation functions is inherently piecewise linear (Montufar et al., 2014). For example, FC, convolutional neural networks (CNN) (LeCun et al., 1998), RNN, and ResNet (He et al., 2016) are all plausible candidates under our consideration. We will call this kind of networks piecewise linear networks, throughout the paper.

The proposed approach is based on a mixed integer linear representation of piecewise linear networks, activation pattern (Raghu et al., 2017), which encodes the active linear piece (integer) of the activation function for each neuron; once an activation pattern is fixed, the network degenerates to a linear model (linear). Thus the feasible set corresponding to an activation pattern in the input space is a natural region where derivatives are provably stable (same linear function). Note the possible degenerate case where neighboring regions (with different activation patterns) nevertheless have the same end-to-end linear coefficients (Serra et al., 2018). We call the feasible set induced by an activation pattern (Serra et al., 2018) a linear region, and a maximal connected subset of the input space subject to the same derivatives of the network (Montufar et al., 2014) a complete linear region. Activation pattern has been studied in various contexts, such as visualizing neurons (Fischetti & Jo, 2017), reachability of a specific output value (Lomuscio & Maganti, 2017), counting the number of linear regions of piecewise linear networks (Raghu et al., 2017; Montúfar, 2017; Serra et al., 2018), and adversarial attacks (Cheng et al., 2017; Fischetti & Jo, 2017; Weng et al., 2018) or defense (Wong & Kolter, 2018).

Here we elaborate differences between our work and the two most relevant categories above. In contrast to quantifying the number of linear regions as a measure of complexity, we focus on the local linear regions, and try to expand them via learning. The notion of stability we consider differs from adversarial examples. The methods themselves are also different. Finding the exact adversarial example is in general NP-complete (Katz et al., 2017; Sinha et al., 2018), and mixed integer linear programs that compute the exact adversarial example do not scale (Cheng et al., 2017; Fischetti & Jo, 2017). Layer-wise relaxations of ReLU activations (Weng et al., 2018; Wong & Kolter, 2018) are more scalable but yield bounds instead exact solutions. Empirically, even relying on relaxations, the defense (learning) methods (Wong & Kolter, 2018; Wong et al., 2018) are still intractable on ImageNet scale images (Deng et al., 2009). In contrast, our inference algorithm certifies the exact  $\ell_2$  margin around a point subject to its activation pattern with a complexity linear in the number of input dimensions. In a high-dimensional setting, where it is computationally challenging to compute the learning objective, we develop an unbiased estimation by a simple sub-sampling procedure, which scales to ResNet (He et al., 2016) on  $299 \times 299 \times 3$  dimensional images in practice.

The proposed learning algorithm is based on the inference problem with  $\ell_2$  margins. The derivation is reminiscent of the SVM objective (Vapnik, 1995; Cortes & Vapnik, 1995), but differs in its purpose; while SVM training seeks to maximize the  $\ell_2$  margin between data points and a linear classifier, our approach instead maximizes the  $\ell_2$  margin of linear regions around each data point. Since there is no label information to guide the learning algorithm for each linear region, the objective is unsupervised and more akin to transductive/semi-supervised SVM (TSVM) (Vapnik & Sterin, 1977; Bennett & Demiriz, 1999).

The problem we tackle has implications to interpretability and transparency of complex models. The gradient has been a building block for various explaining methods for deep models, including gradient saliency map (Simonyan et al., 2013) and its variants (Springenberg et al., 2014; Sundararajan et al., 2017; Smilkov et al., 2017), which apply a gradient-based attribution of the prediction to the input with nonlinear post-processings for visualization (e.g., normalizing and clipping by the  $99^{\text{th}}$  percentile (Smilkov et al., 2017; Sundararajan et al., 2017)). While curing unstable derivatives is motivated by the instability of gradient-based explanations (Ghorbani et al., 2017; Alvarez-Melis & Jaakkola, 2018a), we focus on the fundamental problem of establishing robust derivatives over larger regions.

# 3 METHODOLOGY

To simplify the exposition, the approaches are developed under the notation of FC networks with ReLU activations, which naturally generalizes to other settings. We will first introduce notations, and then present our inference and learning algorithms. All the proofs are available in Appendix A.

# 3.1 NOTATION

We consider a neural network  $\theta$  with  $M$  hidden layers and  $N_{i}$  neurons in the  $i^{\mathrm{th}}$  layer, and the corresponding function  $f_{\theta}:\mathbb{R}^{D}\to \mathbb{R}^{L}$  it represents. We use  $\mathbf{z}^i\in \mathbb{R}^{N_i}$  and  $\mathbf{a}^i\in \mathbb{R}^{N_i}$  to denote the vector of (raw) neurons and activated neurons in the  $i^{\mathrm{th}}$  layer, respectively. We will use  $\mathbf{x}$  and  $\mathbf{a}^0$  interchangeably to represent an input instance from  $\mathbb{R}^D = \mathbb{R}^{N_0}$ . With an FC architecture and ReLU activations, each  $\mathbf{a}^i$  and  $\mathbf{z}^i$  are computed with the transformation matrix  $\mathbf{W}^i\in \mathbb{R}^{N_i\times N_{i - 1}}$  and bias vector  $\mathbf{b}^i\in \mathbb{R}^{N_i}$  as

$$
\mathbf {a} ^ {i} = \operatorname {R e L U} (\mathbf {z} ^ {i}) := \max  (\mathbf {0}, \mathbf {z} ^ {i}), \quad \mathbf {z} ^ {i} = \mathbf {W} ^ {i} \mathbf {a} ^ {i - 1} + \mathbf {b} ^ {i}, \forall i \in [ M ], \quad \mathbf {a} ^ {0} = \mathbf {x}, \tag {1}
$$

where  $[M]$  denotes the set  $\{1,\dots ,M\}$ . We use subscript to further denote a specific neuron. To avoid confusion from other instances  $\bar{\mathbf{x}}\in \mathbb{R}^{D}$ , we assert all the neurons  $\mathbf{z}_j^i$  are functions of the specific instance denoted by  $\mathbf{x}$ . The output of the network is a linear transformation of the last hidden layer  $f_{\theta}(\mathbf{x}) = \mathbf{W}^{M + 1}\mathbf{a}^{M} + \mathbf{b}^{M + 1}$  with  $\mathbf{W}^{M + 1}\in \mathbb{R}^{L\times N_M}$  and  $\mathbf{b}^{M + 1}\in \mathbb{R}^{L}$ . The output can be further processed by a nonlinearity such as softmax for classification problems. However, we focus on the piecewise linear property of neural networks represented by  $f_{\theta}(\mathbf{x})$ , and leverage a generic loss function  $\mathcal{L}(f_{\theta}(\mathbf{x}),\mathbf{y})$  to fold such nonlinear mechanism.

We use  $\mathcal{D}$  to denote the set of training data  $(\mathbf{x},\mathbf{y})$ ,  $\mathcal{D}_{\mathbf{x}}$  to denote the same set without labels  $\mathbf{y}$ , and  $\mathcal{B}_{\epsilon,p}(\mathbf{x}) \coloneqq \{\bar{\mathbf{x}} \in \mathbb{R}^D : \| \bar{\mathbf{x}} - \mathbf{x} \|_p \leq \epsilon\}$  to denote the  $\ell_p$ -ball around  $\mathbf{x}$  with radius  $\epsilon$ .

The activation pattern (Raghu et al., 2017) used in this paper is defined as:

Definition 1. (Activation Pattern) An activation pattern is a set of indicators for neurons  $\mathcal{O} = \{\mathbf{o}^i\in$ $-1,1\}^{N_i}|i\in [M]\}$  that specifies the following functional constraints:

$$
\mathbf {z} _ {j} ^ {i} \geq 0, \text {i f} \mathbf {o} _ {j} ^ {i} = 1; \mathbf {z} _ {j} ^ {i} \leq 0, \text {i f} \mathbf {o} _ {j} ^ {i} = - 1. \tag {2}
$$

Each  $\mathbf{o}_j^i$  is called an activation indicator. Note that a point on the boundary of a linear region is feasible for multiple activation patterns. The definition fits the property of the activation pattern discussed in §2. We define  $\nabla_{\mathbf{x}}\mathbf{z}_j^i$  to be the sub-gradient found by back-propagation using  $\partial \mathbf{a}_{j^{\prime}}^{i^{\prime}} / \partial \mathbf{z}_{j^{\prime}}^{i^{\prime}}\coloneqq \max (\mathbf{o}_{j^{\prime}}^{i^{\prime}},0),\forall j^{\prime}\in [N_{i^{\prime}}],i^{\prime}\in [i - 1]$  whenever  $\mathbf{o}_{j^{\prime}}^{i^{\prime}}$  is defined in the context.

# 3.2 INFERENCE FOR REGIONS WITH STABLE DERIVATIVES

Although the activation pattern implicitly describes a linear region, it does not yield explicit constraints on the input space, making it hard to develop algorithms directly. Hence, we first derive an explicit characterization of the feasible set on the input space  $\mathbb{R}^D$  with Lemma 2.

Lemma 2. Given an activation pattern  $\mathcal{O}$  with any feasible point  $\mathbf{x}$ , each activation indicator  $\mathbf{o}_j^i \in \mathcal{O}$  induces a feasible set  $S_j^i(\mathbf{x}) = \{\bar{\mathbf{x}} \in \mathbb{R}^D : \mathbf{o}_j^i[(\nabla_{\mathbf{x}}\mathbf{z}_j^i)^T\bar{\mathbf{x}} + (\mathbf{z}_j^i - (\nabla_{\mathbf{x}}\mathbf{z}_j^i)^T\mathbf{x})] \geq 0\}$ , and the feasible set of the activation pattern is equivalent to  $S(\mathbf{x}) = \cap_{i=1}^{M} \cap_{j=1}^{N_i} S_j^i(\mathbf{x})$ .

Remark 3. Lemma 2 characterizes each linear region of  $h_\theta$  as the feasible set  $S(\mathbf{x})$  with a set of linear constraints with respect to the input space  $\mathbb{R}^D$ , and thus  $S(\mathbf{x})$  is a convex polyhedron.

The aforementioned linear property of an activation pattern equipped with the input space constraints from Lemma 2 yield the definition of  $\hat{\epsilon}_{\mathbf{x},p}$ , the  $\ell_p$  margin of  $\mathbf{x}$  subject to its activation pattern:

$$
\hat {\epsilon} _ {\mathbf {x}, p} := \max  _ {\epsilon \geq 0: \mathbf {x} ^ {\prime} \in S (\mathbf {x}), \forall \mathbf {x} ^ {\prime} \in \mathcal {B} _ {\epsilon , p} (\mathbf {x})} \epsilon = \max  _ {\epsilon \geq 0: \mathcal {B} _ {\epsilon , p} (\mathbf {x}) \subseteq S (\mathbf {x})} \epsilon , \tag {3}
$$

where  $S(\mathbf{x})$  can be based on any feasible activation pattern  $\mathcal{O}$  on  $\mathbf{x}^1$ ; therefore,  $\partial \mathbf{a}_j^i / \partial \mathbf{z}_j^i$  at  $\mathbf{z}_j^i = 0$  from now on can take 0 or 1 arbitrarily as long as consistency among sub-gradients  $\{\nabla_{\mathbf{x}} \mathbf{z}_j^i | j \in [N_i], i \in [M]\}$  is ensured with respect to some feasible activation pattern  $\mathcal{O}$ . Note that  $\hat{\epsilon}_{\mathbf{x},p}$  is a lower bound of the  $\ell_p$  margin subject to a derivative specification (i.e., a complete linear region).

# 3.2.1 DIRECTIONAL VERIFICATION, THE CASES  $p = 1$  AND  $p = \infty$

We first exploit the convexity of  $S(\mathbf{x})$  to check the feasibility of a directional perturbation.

Proposition 4. (Directional Feasibility) Given a point  $\mathbf{x}$ , a feasible set  $S(\mathbf{x})$  and a unit vector  $\Delta \mathbf{x}$  if  $\exists \bar{\epsilon} \geq 0$  such that  $\mathbf{x} + \bar{\epsilon} \Delta \mathbf{x} \in S(\mathbf{x})$ , then  $f_{\theta}$  is linear in  $\{\mathbf{x} + \epsilon \Delta \mathbf{x} : 0 \leq \epsilon \leq \bar{\epsilon}\}$ .

The feasibility of  $\mathbf{x} + \bar{\epsilon}\Delta \mathbf{x} \in S(\mathbf{x})$  can be computed by simply checking whether  $\mathbf{x} + \bar{\epsilon}\Delta \mathbf{x}$  satisfies the activation pattern  $\mathcal{O}$  in  $S(\mathbf{x})$ . Proposition 4 can be applied to the feasibility problem on  $\ell_1$ -balls.

Proposition 5. ( $\ell_1$ -ball Feasibility) Given a point  $\mathbf{x}$ , a feasible set  $S(\mathbf{x})$ , and an  $\ell_1$ -ball  $\mathcal{B}_{\epsilon,1}(\mathbf{x})$  with extreme points  $\mathbf{x}^1, \ldots, \mathbf{x}^{2D}$ , if  $\mathbf{x}^i \in S(\mathbf{x}), \forall i \in [2D]$ , then  $f_\theta$  is linear in  $\mathcal{B}_{\epsilon,1}(\mathbf{x})$ .

Proposition 5 can be generalized for an  $\ell_{\infty}$ -ball. However, in high dimension  $D$ , the number of extreme points of an  $\ell_{\infty}$ -ball is exponential to  $D$ , making it intractable. Instead, the number of extreme points of an  $\ell_{1}$ -ball is only linear to  $D$  ( $+\epsilon$  and  $-\epsilon$  for each dimension). With the above methods to verify feasibility, we can do binary searches to find the certificates of the margins for directional perturbations  $\hat{\epsilon}_{\mathbf{x},\Delta \mathbf{x}} \coloneqq \max_{\{\epsilon \geq 0:\mathbf{x} + \epsilon \Delta \mathbf{x} \in S(\mathbf{x})\}} \epsilon$  and  $\ell_{1}$ -balls  $\hat{\epsilon}_{\mathbf{x},1}$ . The details are in Appendix B.

# 3.2.2 THE CASE  $p = 2$

The feasibility on  $\hat{\epsilon}_{\mathbf{x},1}$  is tractable due to convexity of  $S(\mathbf{x})$  and its certification is efficient by a binary search; by further exploiting the polyhedron structure of  $S(\mathbf{x})$ ,  $\hat{\epsilon}_{\mathbf{x},2}$  can be certified analytically.

Proposition 6. ( $\ell_2$ -ball Certificate) Given a point  $\mathbf{x}$ ,  $\hat{\epsilon}_{\mathbf{x},2}$  is the minimum  $\ell_2$  distance between  $\mathbf{x}$  and the union of hyperplanes  $\cup_{i=1}^{M} \cup_{j=1}^{N_i}\left\{\bar{\mathbf{x}} \in \mathbb{R}^D : \left(\nabla_{\mathbf{x}} \mathbf{z}_j^i\right)^T \bar{\mathbf{x}} + \left(\mathbf{z}_j^i - \left(\nabla_{\mathbf{x}} \mathbf{z}_j^i\right)^T \mathbf{x}\right) = 0\right\}$ .

To compute the  $\ell_2$  distance between  $\mathbf{x}$  and the hyperplane induced by a neuron  $\mathbf{z}_j^i$ , we evaluate  $|\left(\nabla_{\mathbf{x}}\mathbf{z}_j^i\right)^T\mathbf{x} + (\mathbf{z}_j^i - (\nabla_{\mathbf{x}}\mathbf{z}_j^i)^T\mathbf{x})| / \| \nabla_{\mathbf{x}}\mathbf{z}_j^i\|_2 = |\mathbf{z}_j^i| / \| \nabla_{\mathbf{x}}\mathbf{z}_j^i\|_2$ . If we denote  $\mathcal{I}$  as the set of hidden neuron indices  $\{(i,j)|i\in [M], j\in [N_i]\}$ , then  $\hat{\epsilon}_{\mathbf{x},2}$  can be computed as  $\hat{\epsilon}_{\mathbf{x},2} = \min_{(i,j)\in \mathcal{I}}|\mathbf{z}_j^i| / \| \nabla_{\mathbf{x}}\mathbf{z}_j^i\|_2$ , where all the  $\mathbf{z}_j^i$  can be computed by a single forward pass. We will show in §4.1 that all the  $\nabla_{\mathbf{x}}\mathbf{z}_j^i$  can also be computed efficiently by forward passes in parallel. We refer readers to Figure 1c to see a visualization of the certificates on  $\ell_2$  margins.

# 3.2.3 THE NUMBER OF COMPLETE LINEAR REGIONS

A related problem to measure the linearities is to count the number of linear regions in  $f_{\theta}$ , which is intractable due to the combinatorial nature of activation patterns (Serra et al., 2018). However, we argue that counting the number of linear regions on the whole space does not capture the structure of data manifold, and we propose to certify the number of complete linear regions (#CLR) of  $f_{\theta}$  among the empirical points  $\mathcal{D}_{\mathbf{x}}$ , which turns out to be efficient to compute given a mild condition as revealed by the following Lemma.

Lemma 7. (Complete Linear Region Certificate) If every empirical point  $\mathbf{x} \in \mathcal{D}_{\mathbf{x}}$  has only one feasible activation pattern denoted as  $\mathcal{O}(\mathbf{x})$ , the number of complete linear regions of  $f_{\theta}$  among  $\mathcal{D}_{\mathbf{x}}$  is upper-bounded by the number of different activation patterns  $|\{\mathcal{O}(\mathbf{x}) | \mathbf{x} \in \mathcal{D}_{\mathbf{x}}\}|$ , and lower-bounded by the number of different Jacobians  $|\{J_{\mathbf{x}} f_{\theta}(\mathbf{x}) | \mathbf{x} \in \mathcal{D}_{\mathbf{x}}\}|$ .

# 3.3 LEARNING: MAXIMIZING THE MARGINS OF STABLE DERIVATIVES

In this section, we focus on learning techniques to maximize the  $\ell_2$  margin  $\hat{\epsilon}_{\mathbf{x},2}$  since it is (sub-)differentiable. We first formulate a regularization problem in the objective to maximize the margin:

$$
\min  _ {\theta} \sum_ {(\mathbf {x}, \mathbf {y}) \in \mathcal {D}} \left[ \mathcal {L} \left(f _ {\theta} (\mathbf {x}), \mathbf {y}\right) - \lambda \min  _ {(i, j) \in \mathcal {I}} \frac {\left| \mathbf {z} _ {j} ^ {i} \right|}{\left\| \nabla_ {\mathbf {x}} \mathbf {z} _ {j} ^ {i} \right\| _ {2}} \right] \tag {4}
$$

However, the objective itself is rather rigid due to the inner-minimization and the reciprocal of  $\|\nabla_{\mathbf{x}} \mathbf{z}_j^i\|_2$ . Qualitatively, such rigid loss surface hinders optimization and may attend infinity. To alleviate the problem, we do a hinge-based relaxation to the distance function similar to SVM.

# 3.3.1 RELAXATION

An ideal relaxation of Eq. (4) is to disentangle  $|\mathbf{z}_j^i|$  and  $\| \nabla_{\mathbf{x}}\mathbf{z}_j^i\| _2$  for a smoother problem. Our first attempt is to formulate an equivalent problem with special constraints which we can leverage.

Lemma 8. If any (global) optimal solution in Eq. (4) satisfies  $\min_{(i,j)\in \mathcal{I}}|\mathbf{z}_j^i | > 0,\forall (\mathbf{x},\mathbf{y})\in \mathcal{D}$ , then any optimal solutions for the problem is also optimal for Eq. (4).

$$
\min  _ {\theta} \sum_ {(\mathbf {x}, \mathbf {y}) \in \mathcal {D}} \mathcal {L} \left(f _ {\theta} (\mathbf {x}), \mathbf {y}\right) - \lambda \min  _ {(i, j) \in \mathcal {I}} \frac {\left| \mathbf {z} _ {j} ^ {i} \right|}{\left\| \nabla_ {\mathbf {x}} \mathbf {z} _ {j} ^ {i} \right\| _ {2}}, \quad s. t. \min  _ {(i, j) \in \mathcal {I}} \left| \mathbf {z} _ {j} ^ {i} \right| \geq 1, \forall (\mathbf {x}, \mathbf {y}) \in \mathcal {D}. \tag {5}
$$

If the condition in Lemma 8 does not hold, Eq. (5) is still a valid upper bound of Eq. (4) due to a smaller feasible set. An upper bound of Eq. (5) can be obtained consequently due to the constraints:

$$
\min  _ {\theta} \sum_ {(\mathbf {x}, \mathbf {y}) \in \mathcal {D}} \mathcal {L} \left(f _ {\theta} (\mathbf {x}), \mathbf {y}\right) - \lambda \min  _ {(i, j) \in \mathcal {I}} \frac {1}{\| \nabla_ {\mathbf {x}} \mathbf {z} _ {j} ^ {i} \| _ {2}}, \quad s. t. \min  _ {(i, j) \in \mathcal {I}} \left| \mathbf {z} _ {j} ^ {i} \right| \geq 1, \forall (\mathbf {x}, \mathbf {y}) \in \mathcal {D}. \tag {6}
$$

We then derive a relaxation that solves a smoother problem by relaxing the squared root and reciprocal on the  $\ell_2$  norm as well as the hard constraint with a hinge loss to a soft regularization problem:

$$
\min  _ {\theta} \sum_ {(\mathbf {x}, \mathbf {y}) \in \mathcal {D}} \mathcal {L} \left(f _ {\theta} (\mathbf {x}), \mathbf {y}\right) + \lambda \max  _ {(i, j) \in \mathcal {I}} \left[ \| \nabla_ {\mathbf {x}} \mathbf {z} _ {j} ^ {i} \| _ {2} ^ {2} + C \max  (0, 1 - | \mathbf {z} _ {j} ^ {i} |) \right], \tag {7}
$$

where  $C$  is a hyper-parameter. The relaxed regularization problem can be regarded as a maximum aggregation of TSVM losses among all the neurons, where a TSVM loss with only unannotated data  $\mathcal{D}_{\mathbf{x}}$  can be written as:

$$
\min  _ {\mathbf {w}, b} \sum_ {\mathbf {x} \in \mathcal {D} _ {\mathbf {x}}} \| \mathbf {w} \| _ {2} ^ {2} + C \max  (0, 1 - | \mathbf {w} ^ {T} \mathbf {x} + b |), \tag {8}
$$

which pursues a similar goal to maximize the  $\ell_2$  margin in a linear model scenario, where the margin is computed between a linear hyperplane (the classifier) and the training points.

To visualize the effect of the proposed methods, we make a toy 2D binary classification dataset, and train a 4-layer fully connected network with 1) (vanilla) binary cross-entropy loss  $\mathcal{L}(\cdot ,\cdot)$ , 2) distance regularization as in Eq. (4), and 3) relaxed regularization as in Eq. (7). Implementation details are in Appendix F. The resulting piecewise linear regions and prediction heatmaps along with gradient  $\nabla_{\mathbf{x}}f_{\theta}(\mathbf{x})$  annotations are shown in Figure 1. The distance regularization enlarges the linear regions around each training point, and the relaxed regularization further generalizes the property to the whole space; the relaxed regularization possesses a smoother prediction boundary, and has a special central region where the gradients are 0 to allow gradients to change directions smoothly.

![](images/4edaf15a4716f9a08b0783b0723d631c2e893db15658ec71b82a0c940aca0bb7.jpg)  
(a) Vanilla loss

![](images/a65f57357b4455f5a3232e1c99d763e29cb42fe98093511818a8445d51072c22.jpg)  
(b) Distance regularization  
Figure 1: Toy Examples. The boundary of each linear region is plotted with line segments, and each circle indicates the  $\ell_2$  margin  $\hat{\epsilon}_{\mathbf{x},2}$  around the training point. The prediction heatmap is shown aside each figure. The gradient is annotated as arrows with length proportional to its  $\ell_2$  norm.

![](images/bc61fe85b7758392e7d4a6d2d8ad9d053680c32897bd80ef6a3a034f8aab376d.jpg)  
(c) Relaxed regularization

# 3.3.2 IMPROVING SPARSE LEARNING SIGNALS

Since a linear region is shaped by a set of neurons that are "close" to a given a point, a noticeable problem of Eq. (7) is that it only focuses on the "closest" neuron, making it hard to scale the effect to large networks. Hence, we make a generalization to the relaxed loss in Eq. (7) with a set of neurons that incur high losses to the given point. We denote  $\hat{\mathcal{L}} (\mathbf{x},\gamma)$  as the set of neurons with top  $\gamma$  percent relaxed loss (TSVM loss) on  $\mathbf{x}$ . The generalized loss is our final objective for learning RObust Local Linearity (ROLL) and is written as:

$$
\min  _ {\theta} \sum_ {(\mathbf {x}, \mathbf {y}) \in \mathcal {D}} \mathcal {L} \left(f _ {\theta} (\mathbf {x}), \mathbf {y}\right) + \frac {\lambda}{| \hat {\mathcal {I}} (\mathbf {x} , \gamma) |} \sum_ {(i, j) \in \hat {\mathcal {I}} (\mathbf {x}, \gamma)} \left[ \| \nabla_ {\mathbf {x}} \mathbf {z} _ {j} ^ {i} \| _ {2} ^ {2} + C \max  (0, 1 - | \mathbf {z} _ {j} ^ {i} |) \right]. \tag {9}
$$

A special case of Eq. (9) is when  $\gamma = 100$  (i.e.  $\hat{\mathcal{I}} (\mathbf{x},100) = \mathcal{I}$ ), where the nonlinear sorting step effectively disappears. Such simple additive structure without a nonlinear sorting step can stabilize the training process, is simple to parallelize computation, and allows for an approximate learning algorithm as will be developed in §4.2. Besides, taking  $\gamma = 100$  can induce a strong synergy effect, as all the gradient norms  $\| \nabla_{\mathbf{x}}\mathbf{z}_j^i\| _2^2$  in Eq. (9) between any two layers are highly correlated.

# 4 COMPUTATION, APPROXIMATE LEARNING, AND COMPATIBILITY

# 4.1 PARALLEL COMPUTATION OF GRADIENTS

The  $\ell_2$  margin  $\hat{\epsilon}_{\mathbf{x},2}$  and the ROLL loss in Eq. (9) demands heavy computation on gradient norms. While calling back-propagation  $|\mathcal{I}|$  times is intractable, we develop a parallel algorithm without calling a single back-propagation by exploiting the functional structure of  $f_{\theta}$ .

Given an activation pattern, we know that each hidden neuron  $\mathbf{z}_j^i$  is also a linear function of  $\mathbf{x} \in S(\mathbf{x})$ . We can construct another linear network  $g_{\theta}$  that is identical to  $f_{\theta}$  in  $S(\mathbf{x})$  based on the same set of parameters but fixed linear activation functions constructed to mimic the behavior of  $f_{\theta}$  in  $S(\mathbf{x})$ . Due to the linearity of  $g_{\theta}$ , the derivatives of all the neurons to an input axis can be computed by forwarding two samples: subtracting the neurons with an one-hot input from the same neurons with a zero input. The procedure can be amortized and parallelized to all the dimensions by feeding  $D + 1$  samples to  $g_{\theta}$  in parallel. We remark that the algorithm generalizes to all the piecewise linear networks, and refer readers to Appendix C for algorithmic details. When the network is restricted to be FC, we have a more efficient dynamic programming procedure in Appendix D.

# 4.2 APPROXIMATE LEARNING

Despite the parallelizable computation of  $\nabla_{\mathbf{x}}\mathbf{z}_j^i$ , it is still challenging to compute the loss for large networks in a high dimension setting, where even calling  $D + 1$  forward passes in parallel as used in §4.1 is infeasible due to memory constraints. Hence we propose an unbiased estimator of the ROLL loss in Eq. (9) when  $\hat{\mathcal{I}} (\mathbf{x},\gamma) = \mathcal{I}$ . Note that  $\sum_{(i,j)\in \mathcal{I}}C\max (0,1 - |\mathbf{z}_j^i |)$  is already computable in one single forward pass. For the sum of gradient norms, we use the following equivalent decoupling:

$$
\frac {1}{| \mathcal {I} |} \sum_ {(i, j) \in \mathcal {I}} \| \nabla_ {\mathbf {x}} \mathbf {z} _ {j} ^ {i} \| _ {2} ^ {2} = \frac {1}{| \mathcal {I} |} \sum_ {k = 1} ^ {D} \sum_ {(i, j) \in \mathcal {I}} \left(\frac {\partial \mathbf {z} _ {j} ^ {i}}{\partial \mathbf {x} _ {k}}\right) ^ {2} = \frac {D}{| \mathcal {I} |} \mathbb {E} _ {k \sim \operatorname {U n i f} ([ D ])} \left[ \sum_ {(i, j) \in \mathcal {I}} \left(\frac {\partial \mathbf {z} _ {j} ^ {i}}{\partial \mathbf {x} _ {k}}\right) ^ {2} \right], \tag {10}
$$

Table 1: FC networks on MNIST dataset. #CLR is the number of complete linear regions among the 10K testing points, and  $\hat{e}_{\mathbf{x},p}$  shows the  $\ell_p$  margin for each  $r \in \{25, 50, 75, 100\}$  percentile  $P_r$ .  

<table><tr><td rowspan="2">Loss</td><td rowspan="2">C</td><td rowspan="2">ACC</td><td rowspan="2">#CLR</td><td colspan="4">ˆx,1(×10-4)</td><td colspan="4">ˆx,2(×10-4)</td></tr><tr><td>P25</td><td>P50</td><td>P75</td><td>P100</td><td>P25</td><td>P50</td><td>P75</td><td>P100</td></tr><tr><td>Vanilla</td><td></td><td>98%</td><td>10000</td><td>22</td><td>53</td><td>106</td><td>866</td><td>3</td><td>6</td><td>13</td><td>91</td></tr><tr><td>ROLL</td><td>0.25</td><td>98%</td><td>9986</td><td>219</td><td>530</td><td>1056</td><td>6347</td><td>37</td><td>92</td><td>182</td><td>1070</td></tr><tr><td>ROLL</td><td>1.00</td><td>97%</td><td>8523</td><td>665</td><td>1593</td><td>3175</td><td>21825</td><td>125</td><td>297</td><td>604</td><td>4345</td></tr></table>

![](images/0771d79b9a7657bdb0ce1f73a8dfeaee1606c5e8f3c824842fbb4582d49647e3.jpg)  
(a)  $\lambda = 0.5$

![](images/69b63e3559676512e77e9b0a41d5a9b655b74bd8ff19f16d71981216b1d35c61.jpg)  
(b)  $\lambda = 0.5$

![](images/ec0c5b09959948e81c80617b92e81348e78dad7f3a8238dcbdd46999ac295a4d.jpg)  
(c)  $C = 1$  
Figure 2: Parameter analysis on MNIST dataset.  $P_{50}$  of  $\hat{\epsilon}_{\mathbf{x},2}$  is the median of  $\hat{\epsilon}_{\mathbf{x},2}$  in the testing data.

![](images/f62a0326b2bc43ca4361fec14b3b9ef3775d03eafacee587c650f4453421d398.jpg)  
(d)  $C = 1$

where the summation inside the expectation in the last equation can be efficiently computed using the procedure in §4.1 and is in general storable within GPU memory. In practice, we can uniformly sample  $D'$  ( $1 \leq D' \ll D$ ) input axes to have an unbiased approximation to Eq. (10), where computing all the partial derivatives with respect to  $D'$  axes only requires  $D' + 1$  times memory (one hot vectors and a zero vector) than a typical forward pass for  $\mathbf{x}$ .

# 4.3 COMPATIBILITY

The proposed algorithms can be used on all the deep learning models with affine transformations and piecewise linear activation functions by enumerating every neuron that will be imposed an ReLU-like activation function as  $\mathbf{z}_j^i$ . They do not immediately generalize to the nonlinearity of maxout/max-pooling (Goodfellow et al., 2013) that also yields a piecewise linear function. We provide an initial step towards doing so in the Appendix E, but we suggest to use an average-pooling or convolution with large strides instead, since they do not induce extra linear constraints as max-pooling and do not in general yield significant difference in performance (Springenberg et al., 2014).

# 5 EXPERIMENTS

In this section, we compare our approach ('ROLL') with a baseline model with the same training procedure except the regularization ('vanilla') in several scenarios. All the reported quantities are computed on a testing set. Experiments are run on single GPU with 12G memory.

# 5.1 MNIST

Evaluation Measures: 1) accuracy (ACC), 2) number of complete linear regions (#CLR), and 3)  $\ell_p$  margins of linear regions  $\hat{\epsilon}_{\mathbf{x},p}$ . We compute the margin  $\hat{\epsilon}_{\mathbf{x},p}$  for each testing point  $\mathbf{x}$  with  $p \in \{1,2\}$ , and we evaluate  $\hat{\epsilon}_{\mathbf{x},p}$  on 4 different percentiles  $P_{25}, P_{50}, P_{75}, P_{100}$  among the testing data.

We use a 55,000/5,000/10,000 split of MNIST dataset for training/validation/testing. Experiments are conducted on a 4-layer FC model with ReLU activations. The implementation details are in Appendix G. We report the two models with the largest median  $\hat{\epsilon}_{\mathbf{x},2}$  among validation data given the same and  $1\%$  less validation accuracy compared to the baseline model.

The results are shown in Table 1. The tuned models have  $\gamma = 100$ ,  $\lambda = 2$ , and different  $C$  as shown in the table. The condition in lemma 7 for certifying #CLR is satisfied with tight upper bound and lower bound, so a single number is reported. Given the same performance, the ROLL loss achieves about 10 times larger margins for most of the percentiles than the vanilla loss. By trading-off  $1\%$  accuracy, about 30 times larger margins can be achieved. The Spearman's rank correlation between  $\hat{\epsilon}_{\mathbf{x},1}$  and  $\hat{\epsilon}_{\mathbf{x},2}$  among testing data is at least 0.98 for all the cases. The lower #CLR in our approach than the baseline model reflects the existence of certain larger linear regions that span across different

Table 2: RNNs on the Japanese Vowel dataset.  $\hat{\epsilon}_{\mathbf{x},p}$  shows the  $\ell_p$  margin for each  $r \in \{25, 50, 75, 100\}$  percentile  $P_r$  in the testing data (the larger the better).  

<table><tr><td rowspan="2">Loss</td><td rowspan="2">λ</td><td rowspan="2">C</td><td rowspan="2">ACC</td><td colspan="4">ˆx,1(×10-6)</td><td colspan="4">ˆx,2(×10-6)</td></tr><tr><td>P25</td><td>P50</td><td>P75</td><td>P100</td><td>P25</td><td>P50</td><td>P75</td><td>P100</td></tr><tr><td>Vanilla</td><td></td><td></td><td>98%</td><td>66</td><td>177</td><td>337</td><td>1322</td><td>23</td><td>61</td><td>113</td><td>438</td></tr><tr><td>ROLL</td><td>2-5</td><td>24</td><td>98%</td><td>264</td><td>562</td><td>1107</td><td>5227</td><td>95</td><td>207</td><td>407</td><td>1809</td></tr><tr><td>ROLL</td><td>2-1</td><td>22</td><td>97%</td><td>1284</td><td>2898</td><td>6086</td><td>71235</td><td>544</td><td>1249</td><td>2644</td><td>22968</td></tr></table>

Figure 3: Stability bounds for stable derivatives on Japanese Vowel dataset.  
![](images/0b4ac96ecce28b2653f7eef1bbba901083d9a5f6050b20d665dd9d22ae6419b5.jpg)  
(a) The  $9^{\mathrm{th}}$  channel of the sequence that yields  $P_{50}$  of  $\hat{e}_{\mathbf{x},2}$  on the ROLL model.

![](images/cc9884daccb057c1dd7604081ef12fcc31142e61211dabf132fe1827c2cbbf5b.jpg)  
(b) The  $9^{\mathrm{th}}$  channel of the sequence that yields  $P_{75}$  of  $\hat{e}_{\mathbf{x},2}$  on the ROLL model.

testing points. All the points inside the same linear region in the ROLL model with  $\mathrm{ACC} = 98\%$  have the same label, while there are visually similar digits (e.g., 1 and 7) in the same linear region in the other ROLL model. We do a parameter analysis in Figure 2 with the ACC and  $P_{50}$  of  $\hat{\epsilon}_{\mathbf{x},2}$  under different  $C, \lambda$  and  $\gamma$  when the other hyper-parameters are fixed. As expected, with increased  $C$  and  $\lambda$ , the accuracy decreases with an increased  $\ell_2$  margin. Due to the smoothness of the curves, higher  $\gamma$  values reflect less sensitivity to hyper-parameters  $C$  and  $\lambda$ .

# 5.2 SPEAKER IDENTIFICATION

We train RNNs for speaker identification on a Japanese Vowel dataset from the UCI machine learning repository (Dheeru & Karra Taniskidou, 2017) with the official training/testing split<sup>2</sup>. The dataset has variable sequence length between 7 and 29 with 12 channels and 9 classes. We implement the network with the state-of-the-art scaled Cayley orthogonal RNN (scoRNN) (Helfrich et al., 2018), which parameterizes the transition matrix in RNN using orthogonal matrices to prevent gradient vanishing/exploding, with LeakyReLU activation. The implementation details are in Appendix H. The reported models are based on the same criterion as §5.1.

The results are reported in Table 2. With the same  $1\%$  inferior ACC, our approach leads to a model with about 4/20 times larger margins among the percentiles on testing data, compared to the vanilla loss. The Spearman's rank correlation between  $\hat{\epsilon}_{\mathbf{x},1}$  and  $\hat{\epsilon}_{\mathbf{x},2}$  among all the cases are 0.98. We also conduct sensitivity analysis on the derivatives by finding  $\hat{\epsilon}_{\mathbf{x},\Delta \mathbf{x}}$  along each coordinate  $\Delta \mathbf{x} \in \cup_{i} \cup_{j=1}^{12} \{-\mathbf{e}^{i,j}, \mathbf{e}^{i,j}\}$  ( $\mathbf{e}_{k,l}^{i,j} = 0, \forall k, l$  except  $\mathbf{e}_{i,j}^{i,j} = 1$ ), which identifies the stability bounds  $[\hat{\epsilon}_{\mathbf{x}, -\mathbf{e}^{i,j}}, \hat{\epsilon}_{\mathbf{x},\mathbf{e}^{i,j}}]$  at each timestamp  $i$  and channel  $j$  that guarantees stable derivatives. The visualization using the vanilla and our ROLL model with  $98\%$  ACC is in Figure 3. Qualitatively, the stability bound of the ROLL regularization is consistently larger than the vanilla model.

# 5.3 CALTECH-256

We conduct experiments on Caltech-256 (Griffin et al., 2007), which has 256 classes, each with at least 80 images. We downsize the images to  $299 \times 299 \times 3$  and train a 18-layer ResNet (He et al., 2016) with initializing from parameters pre-trained on ImageNet (Deng et al., 2009). The approximate ROLL loss in Eq. (10) is used with 120 random samples on each channel. We randomly select 5 and 15 samples in each class as the validation and testing set, respectively, and put the remaining data into the training set. The implementation details are in Appendix I.

Evaluation Measures: Due to high input dimensionality ( $D \approx 270K$ ), computing the certificates  $\hat{\epsilon}_{\mathbf{x},1}, \hat{\epsilon}_{\mathbf{x},2}$  is computationally challenging without a cluster of GPUs. Hence, we turn to a sample

Table 3: ResNet on Caltech0256. Here  $\Delta (\mathbf{x},\mathbf{x}^{\prime},\mathbf{y})$  denotes  $\ell_1$  gradient distortion  $\| \nabla_{\mathbf{x}'}f_{\theta}(\mathbf{x}')_{\mathbf{y}} - \nabla_{\mathbf{x}}f_{\theta}(\mathbf{x})_{\mathbf{y}}\| _1$  (the smaller the better for each  $r$  percentile  $P_r$  among the testing data).  

<table><tr><td rowspan="2">Loss</td><td rowspan="2">P@1</td><td rowspan="2">P@5</td><td colspan="4">Ex&#x27;~Unif(Bε,∞(x))[Δ(x, x&#x27;, y)]</td><td colspan="4">maxx&#x27; ∈ Bε,∞(x) [Δ(x, x&#x27;, y)]</td></tr><tr><td>P25</td><td>P50</td><td>P75</td><td>P100</td><td>P25</td><td>P50</td><td>P75</td><td>P100</td></tr><tr><td>Vanilla</td><td>80.7%</td><td>93.4%</td><td>583.8</td><td>777.4</td><td>1041.9</td><td>3666.7</td><td>840.9</td><td>1118.2</td><td>1477.6</td><td>5473.5</td></tr><tr><td>ROLL</td><td>80.8%</td><td>94.1%</td><td>540.6</td><td>732.0</td><td>948.7</td><td>2652.2</td><td>779.9</td><td>1046.7</td><td>1368.2</td><td>3882.8</td></tr></table>

![](images/32d7e07671d15d1b5d9bc25714d38ffdca72655ca7d616000d2b4cc64f24b84a.jpg)

![](images/9be042d701a1d167f2e5dc20e9ad2ab545d37fb0ff7c8861eea7aaaafa1bdea7.jpg)  
(a) Image (Laptop)  
(f) Image (Bear)  
Figure 4: Visualization of the examples in Caltech-256 that yield the  $P_{50}$  (above) and  $P_{75}$  (below) of the maximum  $\ell_1$  gradient distortions among the testing data on our ROLL model.

![](images/9c948be265a2151fc912ed49417ce4829953c88ab680047ae91714d31dc70e1b.jpg)

![](images/a668cbd67eb447e4f3f9ea7a1b098604a5eb6f92f4a82f583824ec6ee5816ccb.jpg)  
(b) Orig. gradient (ROLL)  
(g) Orig. gradient (ROLL)

![](images/848191a62bc2aae2ed960634bb8788d0be6740e94f0a7ffbc16508be8962ff32.jpg)

![](images/e270c87fb154502ced0bde1ef2a63373168bb6ea2ceead674611ae690e55796b.jpg)  
(c) Adv. gradient (ROLL)  
(h) Adv. gradient (ROLL)

![](images/b7ba5358696f104d0f7b2b9c0a641f869ac5e09d784a41e7f6c52d622c727aa5.jpg)

![](images/44ee675d93c8267ab492f3b8b075344a08b3894659ea5baa9be48cec90286da4.jpg)  
(d) Orig. gradient (Vanilla)  
(i) Orig. gradient (Vanilla)

![](images/beb085e1f111b4a141d0dea35a662334bb1f136bcb25a3f5f68116f095afc78e.jpg)

![](images/fe7de6f83868de1d33d296f2d44dd16e0b1fbef808a65bb47b0158306fd46519.jpg)  
(e) Adv. gradient (Vanilla)  
(j) Adv. gradient (Vanilla)

based approach to evaluate the stability of the gradients  $f_{\theta}(\mathbf{x})_{\mathbf{y}}$  for the ground-truth label in a local region with a goal to reveal the stability across different linear regions. Note that evaluating the gradient of the prediction instead is problematic to compare different models in this case.

Given labeled data  $(\mathbf{x},\mathbf{y})$ , we evaluate the stability of gradient  $\nabla_{\mathbf{x}}f_{\theta}(\mathbf{x})_{\mathbf{y}}$  in terms of expected  $\ell_1$  distortion (over a uniform distribution) and the maximum  $\ell_1$  distortion within the intersection  $\bar{\mathcal{B}}_{\epsilon,\infty}(\mathbf{x}) = \mathcal{B}_{\epsilon,\infty}(\mathbf{x}) \cap \mathcal{X}$  of an  $\ell_{\infty}$ -ball and the domain of images  $\mathcal{X} = [0,1]^{299 \times 299 \times 3}$ . The  $\ell_1$  gradient distortion is defined as  $\Delta(\mathbf{x},\mathbf{x}',\mathbf{y}) := \|\nabla_{\mathbf{x}'}f_{\theta}(\mathbf{x}')\mathbf{y} - \nabla_{\mathbf{x}}f_{\theta}(\mathbf{x})\mathbf{y}\|_1$ , where we called the maximizer of  $\ell_1$  gradient distortion the adversarial gradient. Computation of the maximum  $\ell_1$  distortion requires optimization, but gradient-based optimization is not applicable since the gradient of the loss involves the Hessian  $\nabla_{\mathbf{x}'}^2 f_{\theta}(\mathbf{x}')\mathbf{y}$  which is either 0 or ill-defined due to piecewise linearity. Hence, we use a genetic algorithm (Whitley, 1994) for black-box optimization. Implementation details are provided in Appendix J. We use 8000 samples to approximate the expected  $\ell_1$  distortion. Due to computational limits, we only evaluate 1024 random images  $\mathbf{x}$  in the testing set for both maximum and expected  $\ell_1$  gradient distortions. The  $\ell_{\infty}$ -ball radius  $\epsilon$  is set to  $8/256$ .

The results along with precision at 1 and 5 (P@1 and P@5) are presented in Table 3. The ROLL loss yields more stable gradients than the vanilla loss with marginally superior precisions. Out of 1024 examined examples  $\mathbf{x}$ , only 40 and 42 gradient-distorted images change prediction labels in the ROLL and vanilla model, respectively. We visualize some examples in Figure 4 with the original and adversarial gradients for each loss. Qualitatively, the ROLL loss yields stable shapes and intensities of gradients, while the vanilla loss does not. More examples with integrated gradient attributions (Sundararajan et al., 2017) are provided in Appendix K.

# 6 CONCLUSION

This paper introduces a new learning problem to endow deep learning models with robust local linearity. The central attempt is to construct locally transparent neural networks, where the derivatives faithfully approximate the underlying function and lends itself to be stable tools for further applications. We focus on piecewise linear networks and solve the problem based on a margin principle similar to SVM. Empirically, the proposed ROLL loss expands regions with provably stable derivatives, and further generalize the stable gradient property across linear regions.

# REFERENCES

David Alvarez-Melis and Tommi S Jaakkola. On the robustness of interpretability methods. 2018 ICML Workshop on Human Interpretability in Machine Learning (WHI 2018), 2018a.  
David Alvarez-Melis and Tommi S Jaakkola. Towards robust interpretability with self-explaining neural networks. arXiv preprint arXiv:1806.07538, 2018b.  
Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. In Proceedings of the International Conference on Machine Learning, pp. 1120-1128, 2016.  
David Baehrens, Timon Schroeter, Stefan Harmeling, Motoaki Kawanabe, Katja Hansen, and Klaus-Robert Mäzller. How to explain individual classification decisions. Journal of Machine Learning Research, 11(Jun):1803-1831, 2010.  
Marc G Bellemare, Ivo Danihelka, Will Dabney, Shakir Mohamed, Balaji Lakshminarayanan, Stephan Hoyer, and Rémi Munos. The cramer distance as a solution to biased Wasserstein gradients. arXiv preprint arXiv:1705.10743, 2017.  
Kristin P Bennett and Ayhan Demiriz. Semi-supervised support vector machines. In Advances in Neural Information processing systems, pp. 368-374, 1999.  
Chih-Hong Cheng, Georg Nuhrenberg, and Harald Ruess. Maximum resilience of artificial neural networks. In International Symposium on Automated Technology for Verification and Analysis, pp. 251-268. Springer, 2017.  
Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine learning, 20(3):273-297, 1995.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE international conference on computer vision, pp. 248-255. IEEE, 2009.  
Dua Dheeru and Efi Karra Taniskidou. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.  
Matteo Fischetti and Jason Jo. Deep neural networks as 0-1 mixed integer linear programs: A feasibility study. arXiv preprint arXiv:1712.06174, 2017.  
Amirata Ghorbani, Abubakar Abid, and James Zou. Interpretation of neural networks is fragile. arXiv preprint arXiv:1710.10547, 2017.  
Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 315-323, 2011.  
Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. 12 2014.  
Ian J Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. Maxout networks. arXiv preprint arXiv:1302.4389, 2013.  
Gregory Griffin, Alex Holub, and Pietro Perona. Caltech-256 object category dataset. 2007.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pp. 5767-5777, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026-1034, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.

Kyle Helfrich, Devin Willmott, and Qiang Ye. Orthogonal recurrent neural networks with scaled cayley transform. Proceedings of the International Conference on Machine Learning, 2018.  
Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE international conference on computer vision, volume 1, pp. 3, 2017.  
Guy Katz, Clark Barrett, David L Dill, Kyle Julian, and Mykel J Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In International Conference on Computer Aided Verification, pp. 97-117. Springer, 2017.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. arXiv preprint arXiv:1703.04730, 2017.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Alessio Lomuscio and Lalit Maganti. An approach to reachability analysis for feed-forward relu neural networks. arXiv preprint arXiv:1706.07351, 2017.  
Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. volume 30, pp. 3, 2013.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Guido Montúfar. Notes on the number of linear regions of deep neural networks. 2017.  
Guido F Montufar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of linear regions of deep neural networks. In Advances in neural information processing systems, pp. 2924-2932, 2014.  
Youssef Mroueh, Chun-Liang Li, Tom Sercu, Anant Raj, and Yu Cheng. Sobolev gan. International Conference on Learning Representations, 2018.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the expressive power of deep neural networks. Proceedings of the International Conference on Machine Learning, 2017.  
Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=ryQu7f-RZ.  
Salah Rifai, Yoshua Bengio, Yann Dauphin, and Pascal Vincent. A generative process for sampling contractive auto-encoders. arXiv preprint arXiv:1206.6434, 2012.  
Thiago Serra, Christian Tjandraatmadja, and Srikumar Ramalingam. Bounding and counting linear regions of deep neural networks. Proceedings of the International Conference on Machine Learning, 2018.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.  
Aman Sinha, Hongseok Namkoong, and John Duchi. Certifying some distributional robustness with principled adversarial training. 2018.

Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017.  
Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.  
Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. arXiv preprint arXiv:1703.01365, 2017.  
Vladimir N. Vapnik. Estimation of dependences based on empirical data. 1982. NY: Springer-Verlag, 1995.  
Vladimir N. Vapnik and A. Sterin. On structural risk minimization or overall risk in a problem of pattern recognition. Automation and Remote Control, 10(3):14951503, 1977.  
Dilin Wang and Qiang Liu. Learning to draw samples: With application to amortized mle for generative adversarial learning. arXiv preprint arXiv:1611.01722, 2016.  
Tsui-Wei Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane Boning, Inderjit S Dhillon, and Luca Daniel. Towards fast computation of certified robustness for relu networks. Proceedings of the International Conference on Machine Learning, 2018.  
Darrell Whitley. A genetic algorithm tutorial. Statistics and computing, 4(2):65-85, 1994.  
Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In Proceedings of the International Conference on Machine Learning, pp. 5283-5292, 2018.  
Eric Wong, Frank Schmidt, Jan Hendrik Metzen, and J Zico Kolter. Scaling provable adversarial defenses. arXiv preprint arXiv:1805.12514, 2018.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. 2017.
