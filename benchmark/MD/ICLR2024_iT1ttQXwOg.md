# EQUIVARIANT DEEP WEIGHT SPACE ALIGNMENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Permutation symmetries of deep networks make simple operations like model averaging and similarity estimation challenging. In many cases, aligning the weights of the networks, i.e., finding optimal permutations between their weights, is necessary. More generally, weight alignment is essential for a wide range of applications, from model merging, through exploring the optimization landscape of deep neural networks, to defining meaningful distance functions between neural networks. Unfortunately, weight alignment is an NP-hard problem. Prior research has mainly focused on solving relaxed versions of the alignment problem, leading to either time-consuming methods or sub-optimal solutions. To accelerate the alignment process and improve its quality, we propose a novel framework aimed at learning to solve the weight alignment problem, which we name DEEP-ALIGN. To that end, we first demonstrate that weight alignment adheres to two fundamental symmetries and then, propose a deep architecture that respects these symmetries. Notably, our framework does not require any labeled data. We provide a theoretical analysis of our approach and evaluate DEEP-ALIGN on several types of network architectures and learning setups. Our experimental results indicate that a feed-forward pass with DEEP-ALIGN produces better or equivalent alignments compared to those produced by current optimization algorithms. Additionally, our alignments can be used as an initialization for other methods to gain even better solutions with a significant speedup in convergence.

# 1 INTRODUCTION

The space of deep network weights has a complex structure since networks maintain their function under certain permutations of their weights. This fact makes it hard to perform simple operations over deep networks, such as averaging their weights or estimating similarity. It is therefore highly desirable to "align" networks - find optimal permutations between the weight matrices of two networks. Weight Alignment is critical to many tasks that involve weight spaces. One key application is model merging and editing (Ainsworth et al., 2022; Wortsman et al., 2022; Stoica et al., 2023; Ilharco et al., 2022), in which the weights of two or more models are (linearly) combined into a single model to improve their performance or enhance their capabilities. Weight alignment algorithms are also vital to the study of the loss landscape of deep networks (Entezari et al., 2022), a recent research direction that has gained increasing attention. Moreover, weight alignment induces an invariant distance function on the weight space that can be used for clustering and visualization.

Since weight alignment is NP-hard (Ainsworth et al., 2022), current approaches rely primarily on local optimization of the alignment objective which is time-consuming and may lead to suboptimal solutions. Therefore, identifying methods with faster run time and improved alignment quality is an important research objective. A successful implementation of such methods would allow practitioners to perform weight alignment in real-time, for example, when merging models in federated or continual learning setups, or to perform operations that require computing many alignments in a reasonable time, such as weight space clustering.

Following a large body of works that suggested learning to solve combinatorial optimization problems using deep learning architectures (Khalil et al., 2017; Bengio et al., 2021; Cappart et al., 2021), we propose the first learning-based approach to weight alignment, called DEEP-ALIGN. DEEP-ALIGN is a neural network with a specialized architecture to predict high-quality weight alignments for a given distribution of data. A major benefit of our approach is that after a model has been trained, predicting the alignment between two networks amounts to a simple feed-forward pass through the

network followed by an efficient projection step, as opposed to solving an optimization problem in other methods.

This paper presents a principled approach to designing a deep architecture for the weight alignment problem. We first formulate the weight-alignment problem and prove it adheres to a specific equivariance structure. We then propose a neural architecture that respects this structure, based on newly suggested equivariant architectures for deep-weight spaces (Navon et al., 2023) called Deep Weight Space Networks (DWSNets). The architecture is based on a Siamese application of DWSNets to a pair of input networks, mapping the outputs to a lower dimensional space we call activation space, and then using a generalized outer product layer to generate candidates for optimal permutations.

Theoretically, we prove that our architecture can approximate the Activation Matching algorithm Tatro et al. (2020); Ainsworth et al. (2022), which computes the activations of the two networks on some pre-defined input data and aligns their weights by solving a sequence of linear assignment problems. This theoretical analysis suggests that DEEP-ALIGN can be seen as a learnable generalization of this algorithm. Furthermore, we show that DEEP-ALIGN has a valuable theoretical property called Exactness, which guarantees that it always outputs the correct alignment when there is a solution with zero objective.

Obtaining labeled training data is one of the greatest challenges when learning to solve combinatorial optimization problems. To address this challenge, we generate labeled examples on the fly by applying random permutations and noise to our unlabeled data. We then train our network using a combination of supervised and unsupervised loss functions without relying on any labeled examples.

Our experimental results indicate that DEEP-ALIGN produces better or comparable alignments relative to those produced by slower optimization-based algorithms, when applied to both MLPs and CNNs. Furthermore, we show that our alignments can be used as an initialization for other methods that result in even better alignments, as well as significant speedups in their convergence. Lastly, we show that our trained networks produce meaningful alignments even when applied to out-of-distribution weight space data.

Previous work. Several algorithms have been proposed for weight-alignment (Tatro et al., 2020; Ainsworth et al., 2022; Pena et al., 2023; Akash et al., 2022). Ainsworth et al. (2022) presented three algorithms: Activation Matching, Weight Matching, and straight-through estimation. Pena et al. (2023) improved upon these algorithms by incorporating a Sinkhorn-based projection method. In part, these works were motivated by studying the loss landscapes of deep neural networks. It was conjectured that deep networks exhibit a property called linear mode connectivity: for any two trained weight vectors (i.e., a concatenation of all the parameters of neural architecture), a linear interpolation between the first vector and the optimal alignment of the second, yields very small increases in the loss (Entezari et al., 2022; Garipov et al., 2018; Draxler et al., 2018; Freeman & Bruna, 2016; Tatro et al., 2020). Another relevant research direction is the growing area of research that focuses on applying neural networks to neural network weights. Early methods proposed using simple architectures (Unterthiner et al., 2020; Andreis et al., 2023; Eilertsen et al., 2020). Several recent papers exploit the symmetry structure of the weight space in their architectures (Navon et al., 2023; Zhou et al., 2023a;b; Zhang et al., 2023). A comprehensive survey of relevant previous work can be found in Appendix A.

# 2 PRELIMINARIES

Equivalence Let  $G$  be a group acting on  $\mathcal{V}$  and  $\mathcal{W}$ . We say that a function  $L: \mathcal{V} \to \mathcal{W}$  is equivariant if  $L(gv) =gL(v)$  for all  $v \in \mathcal{V}, g \in G$ .

MultiLayer Perceptrons and weight spaces. The following definition follows the notation in Navon et al. (2023). An  $M$ -layer MultiLayer Perceptron (MLP)  $f_{v}$  is a parametric function of the following form:

$$
f (x) = x _ {M}, \quad x _ {m + 1} = \sigma \left(W _ {m + 1} x _ {m} + b _ {m + 1}\right), \quad x _ {0} = x \tag {1}
$$

Here,  $x_{m} \in \mathbb{R}^{d_{m}}$ ,  $W_{m} \in \mathbb{R}^{d_{m} \times d_{m-1}}$ ,  $b_{m} \in \mathbb{R}^{d_{m}}$ , and  $\sigma$  is a pointwise activation function. Denote by  $v = [W_{m}, b_{m}]_{m \in [M]}$  the concatenation of all (vectorized) weight matrices and bias vectors. We define the weight-space of an  $M$ -layer MLP as:  $\mathcal{V} = \bigoplus_{m=1}^{M} (\mathcal{W}_{m} \oplus \mathcal{B}_{m})$ , where  $\mathcal{W}_{m} := \mathbb{R}^{d_{m} \times d_{m-1}} \mathcal{B}_{m} = \mathbb{R}^{d_{m}}$  and  $\bigoplus$  denotes the direct sum (concatenation) of vector spaces. A vector in

this space represents all the learnable parameters on an MLP. We define the activation space of an MLP as  $\mathcal{A} = \bigoplus_{m=1}^{M} \mathbb{R}^{d_m} := \bigoplus_{m=1}^{M} \mathcal{A}_m$ . The activation space, as its name implies, represents the concatenation of network activations at all layers. i.e.,  $\mathcal{A}_m$  is the space in which  $x_m$  resides.

Symmetries of weight spaces. The permutation symmetries of the weight space are a result of the equivariance of pointwise activations: for every permutation matrix  $P$  we have that  $P\sigma(x) = \sigma(Px)$ . Thus for example, a shallow network defined by weight matrices  $W_{1}, W_{2}$  will represent the same function as the network defined by  $PW_{1}, W_{2}P^{T}$ , since the permutations cancel each other. The same idea can be used to identify permutation symmetries of general MLPs of depth  $M$ . In this case, the weight space's symmetry group is the direct product of symmetric groups for each intermediate dimension  $m \in [1, M-1]$  namely,  $S_{d_1} \times \dots \times S_{d_{M-1}}$ . For clarity, we formally define the symmetry group as a product of matrix groups:  $G = \Pi_{d_1} \times \dots \times \Pi_{d_{M-1}}$ , where  $\Pi_d$  is the group of  $d \times d$  permutation matrices (which is isomorphic to  $S_d$ ). For  $v \in \mathcal{V}$ ,  $v = [W_m, b_m]_{m \in [M]}$ , a group element  $g = (P_1, \ldots, P_{M-1})$  acts on  $v$  via a group action  $v' = g_\#(v)$ , where  $v' = [W_m', b_m']_{m \in [M]}$  is defined by:

$$
\tilde {W} _ {1} ^ {\prime} = P _ {1} W _ {1}, W _ {M} ^ {\prime} = W _ {M} P _ {M - 1} ^ {T}, \text {a n d} W _ {m} ^ {\prime} = P _ {m} W _ {m} P _ {m - 1} ^ {T}, \forall m \in [ 2, M - 1 ]
$$

$$
b _ {1} ^ {\prime} = P _ {1} b _ {1}, b _ {M ^ {\prime}} = b _ {M}, \text {a n d} b _ {m} ^ {\prime} = P _ {m} b _ {m}, \forall m \in [ 2, M - 1 ].
$$

By construction,  $v$  and  $v' = g_{\#}(v)$  define the same function  $f_v = f_{v'}$ . The group product  $g \cdot g'$  and group inverse  $g^{-1} = g^T$  are naturally defined as the elementwise matrix product and transpose operations  $g \cdot g' = (P_1 P_1', \ldots, P_M P_M')$ ,  $g^T = (P_1^T, \ldots, P_m^T)$ . Note that the elementwise product and transpose operations are well defined even if the  $P_m$  and  $P_m'$  matrices are not permutations.

# 3 THE WEIGHT ALIGNMENT PROBLEM AND ITS SYMMETRIES

The weight alignment problem. Given an MLP architecture as in equation 1 and two weight-space vectors  $v, v' \in \mathcal{V}$ , where  $v = [W_m, b_m]_{m \in [M]}$ ,  $v' = [W_m', b_m']_{m \in [M]}$ , the weight alignment problem is defined as the following optimization problem:

$$
\mathcal {G} (v, v ^ {\prime}) = \operatorname {a r g m i n} _ {k \in G} \| v - k _ {\#} v ^ {\prime} \| _ {2} ^ {2} \tag {2}
$$

In other words, the problem seeks a sequence of permutations  $k = (P_{1},\ldots ,P_{M - 1})$  that will make  $v^{\prime}$  as close as possible to  $v$ . The optimization problem in equation 2 always admits a minimizer since  $G$  is finite. For some  $(v,v^{\prime})$  it may have several minimizers, in which case  $\mathcal{G}(v,v^{\prime})$  is a set of elements. To simplify our discussion we will sometimes consider the domain of  $\mathcal{G}$  to be only the set  $\mathcal{V}_{\mathrm{unique}}^{2}$  of pairs  $(v,v^{\prime})$  for which a unique minimizer exists. On this domain we can consider  $\mathcal{G}$  as a function to the unique minimizer in  $G$ , that is  $\mathcal{G}:\mathcal{V}_{\mathrm{unique}}^{2}\to G$ .

Our goal in this paper is to devise an architecture that can learn the function  $\mathcal{G}$ . As a guiding principle for devising this architecture, we would like this function to be equivariant to the symmetries of  $\mathcal{G}$ . We describe these symmetries next.

The symmetries of  $\mathcal{G}$ . One important property of the function  $\mathcal{G}$  is that it is equivariant to the action of the group  $H = G \times G$  which consists of two independent copies of the permutation symmetry group for the MLP architecture we consider. Here, the action  $h = (g, g') \in H$  on the input space  $\mathcal{V} \times \mathcal{V}$  is simply  $(v, v') \mapsto (g_{\#}v, g_{\#}'v')$ , and the action of  $h = (g, g') \in H$  on an element  $k \in G$  in the output space is given by  $g \cdot k \cdot g'^T$ . This equivariance property is summarized and proved in the proposition below and visualized using the commutative diagram in Figure 1: applying  $\mathcal{G}$  and then  $(g, g')$  results in exactly the same output as applying  $(g, g')$  and then  $\mathcal{G}$ .

Proposition 1. The map  $\mathcal{G}$  is  $H$ -equivariant, namely, for all  $(v,v^{\prime})\in \mathcal{V}_{\mathrm{unique}}^{2}$  and  $(g,g^{\prime})\in H$

$$
\mathcal {G} \left(g _ {\#} v, g _ {\#} ^ {\prime} v ^ {\prime}\right) = g \cdot \mathcal {G} \left(v, v ^ {\prime}\right) \cdot g ^ {\prime T}
$$

The function  $\mathcal{G}$  exhibits another interesting property: swapping the order of the inputs  $v, v'$  corresponds to inverting the optimal alignment  $\mathcal{G}(v, v')$ :

Proposition 2. Let  $(v,v^{\prime})\in \mathcal{V}_{\text{unique}}^{2}$  then  $\mathcal{G}(v',v) = \mathcal{G}(v,v')^T$

Extension to multiple minimizers. For simplicity the above discussion focused on the case where  $(v, v') \in \mathcal{V}_{\mathrm{unique}}^2$ . We can also state analogous claims for the general case where multiple minimizers

![](images/e017db84fbd00518ca80c7e0d8aace99ae6d17664364230ce61c5e2415ca1fcb.jpg)  
Figure 1: The equivariance structure of the alignment problem. The function  $\mathcal{G}$  takes as input two weight space vectors  $v, v'$  and outputs a sequence of permutation matrices that aligns them denoted  $\mathcal{G}(v, v')$ . In case we reorder the input using  $(g, g')$  where  $g = (P_1, P_2)$ ,  $g' = (P_1', P_2')$ , the optimal alignment undergoes a transformation, namely  $\mathcal{G}(g_{\#}v, g_{\#}'v') = g \cdot \mathcal{G}(v, v') \cdot g'^T$ .

are possible. In this case we will have that the equalities  $g \cdot \mathcal{G}(v, v') \cdot g'^T = \mathcal{G}(gv, g'v')$  and  $\mathcal{G}(v, v')^T = \mathcal{G}(v', v)$  still hold as equalities between subsets of  $G$ .

Extension to other optimization objectives. In Appendix B we show that the equivariant structure of the function  $\mathcal{G}$  occurs not only for the objective in equation 2, but also when the objective  $\| v - k_{\#}v'\|_2^2$  is replaced with any scalar function  $E(v,k_{\#}v')$  that satisfies the following properties: (1)  $E$  is invariant to the action of  $G$  on both inputs; and (2)  $E$  is invariant to swapping its arguments.

# 4 DEEP-ALIGN

# 4.1 ARCHITECTURE

Here, we define a neural network architecture  $F = F(v,v^{\prime};\theta)$  for learning the weight-alignment problem. The output of  $F$  will be a sequence of square matrices  $(P_{1},\ldots ,P_{M - 1})$  that represents a (sometimes approximate) group element in  $G$ . In order to provide an effective inductive bias, we will ensure that our architecture meets both properties: 1,2, namely  $F(g_{\#}v,g_{\#}^{\prime}v^{\prime}) = g\cdot F(v,v^{\prime})\cdot g^{\prime T}$  and  $F(v,v^{\prime}) = F(v^{\prime},v)^{T}$ . The architecture we propose is composed of four functions:

$$
F = F _ {p r o j} \circ F _ {p r o d} \circ F _ {\mathcal {V} \rightarrow \mathcal {A}} \circ F _ {D W S}: \mathcal {V} \times \mathcal {V} ^ {\prime} \to \bigoplus_ {m = 1} ^ {M - 1} \mathbb {R} ^ {d _ {m} \times d _ {m}},
$$

where the equivariance properties we require are guaranteed by constructing each of the four functions composing  $F$  to be equivariant with respect to an appropriate action of  $H = G \times G$  and the transposition action  $(v, v') \mapsto (v', v)$ . In general terms, we choose  $F_{DWS}$  to be a siamese weight space encoder,  $F_{\mathcal{V} \to \mathcal{A}}$  is a siamese function that maps the weight space to the activation space,  $F_{prod}$  is a function that performs (generalized) outer products between corresponding activation spaces in both networks and  $F_{proj}$  performs a projection of the resulting square matrices on the set of doubly stochastic matrices (the convex hull of permutation matrices). The architecture is illustrated in Figure 2. We now describe our architecture in more detail.

Weight space encoder.  $F_{DWS} : \mathcal{V} \times \mathcal{V}' \to \mathcal{V}^d \times \mathcal{V}^{d'}$ , where  $d$  represents the number of feature channels, is implemented as a Siamese DWSNet (Navon et al., 2023). This function outputs two weight-space embeddings in  $\mathcal{V}^d$ , namely,  $F_{DWS}(v, v') = (\mathcal{E}(v), \mathcal{E}(v'))$ , for a DWS network  $\mathcal{E}$ . The Siamese structure of the network guarantees equivariance to transposition. This

![](images/784780026093f4d9e9905627f17e1d9e955183932ad1a01d990bc071bd4f68aa.jpg)  
Figure 2: Our architecture is a composition of four blocks: The first block,  $F_{DWS}$  generates weight space embedding for both inputs. The second block  $F_{\mathcal{V}\to \mathcal{A}}$  maps these to the activation spaces. The third block,  $F_{Prod}$ , generates square matrices by applying an outer product between the activation vector of one network to the activation vectors of the other network. Lastly, the fourth block,  $F_{Proj}$  projects these square matrices on the (convex hull of) permutation matrices.

is because the same encoder is used for both inputs, regardless of their input order. The  $G$ -equivariance of DWSNet, on the other hand, implies equivariance to the action of  $G \times G$ , that is  $(\mathcal{E}(g_{\#}v), \mathcal{E}(g_{\#}'v')) = (g_{\#}\mathcal{E}(v), g_{\#}'\mathcal{E}(v'))$ .

Mapping the weight space to the activation space. The function  $F_{\mathcal{V}\to \mathcal{A}}:\mathcal{V}^d\times \mathcal{V}^{\prime d}\to \mathcal{A}^d\times \mathcal{A}^{\prime d}$  maps the weight spaces  $\mathcal{V}^d,\mathcal{V}^{\prime d}$  to the corresponding Activation Spaces (see preliminaries section). There are several ways to implement  $F_{\mathcal{V}\rightarrow \mathcal{A}}$ . As the bias space,  $\mathcal{B} = \bigoplus_{m = 1}^{M}\mathcal{B}_{m}$ , and the activation space have a natural correspondence between them, perhaps the simplest way, which we use in this paper, is to map a weight space vector  $v = (w,b)\in \mathcal{V}^d$  to its bias component  $b\in \mathcal{B}^d$ . We emphasize that the bias representation is extracted from the previously mentioned weight space decodes, and in that case, it depends on and represents both the weights and the biases in the input. This operation is again equivariant to transposition and the action of  $G\times G$ , where the action of  $G\times G$  on the input space is the more complicated action (by  $(g_{\#},g_{\#}^{\prime})$ ) on  $\mathcal{V}\times \mathcal{V}$  and the action on the output space is the simpler action of  $G\times G$  on the activation spaces.

Generalized outer product.  $F_{prod}:\mathcal{A}^d\times \mathcal{A}'^d\to \bigoplus_{m = 1}^{M}\mathbb{R}^{d_m\times d_m}$  is a function that takes the activation space features and performs a generalized outer product operation as defined below:

$$
F _ {p r o d} \left(a, a ^ {\prime}\right) _ {m, i, j} = \phi \left(\left[ a _ {m, i}, a _ {m, j} ^ {\prime} \right]\right)
$$

where the subscripts  $m, i, j$  represent the  $(i, j)$ -th entry of the  $m$ -th matrix, and  $a_{m,i}, a_{m,j}' \in \mathbb{R}^d$  are the rows of  $a, a'$ . Here, the function  $\phi$  is a general (parametric or nonparametric) symmetric function in the sense that  $\phi(a, b) = \phi(b, a)$ . In this paper, we use  $\phi(a, b) = s^2 \langle a / \|a\|_2, b / \|b\|_2 \rangle$  where  $s$  is a trainable scalar scaling factor. The equivariance with respect to the action of  $G \times G$  and transposition is guaranteed by the fact that  $\phi$  is applied elementwise, and is symmetric, respectively.

Projection layer. The output of  $F_{prod}$  is a sequence of matrices  $Q_{1},\ldots ,Q_{M - 1}$  which in general will not be permutation matrices. To bring the outputs closer to permutation matrices,  $F_{proj}$  implements a approximate projection onto the convex hull of the permutation matrices, i.e., the space of doubly stochastic matrices. In this paper, we use two different projection operations, depending on whether the network is in training or inference mode. At training time, to ensure differentiability, we implement  $F_{proj}$  as an approximation of a matrix-wise projection  $Q_{m}$  to the space of doubly stochastic matrices using several iterations of the well-known Sinkhorn projection (Mena et al., 2018; Sinkhorn, 1967). Since the set of doubly stochastic matrices is closed under the action of  $G\times G$  on the output space, and under matrix transposition, and since the Sinkhorn iterations are composed of elementwise, row-wise, or column-wise operations, we see that this operation is equivariant as well. At inference time, we obtain permutation matrices from  $Q_{i}$  by finding the permutation matrix  $P_{i}$  which has the highest correlation with  $Q_{i}$ , that is  $P_{i} = \arg \max_{P\in S_{d_{i}}}\langle Q_{i},P\rangle$

![](images/83de9a1219d58ef6ce5a2d65176f4db6fdafb039b43008ad1470c8efe508c01e.jpg)  
(a) CIFAR10 MLPs.

![](images/fb8db511f147797f2f5eb4469fc0014455b900ae20de1b6e9a74f7516a5035bc.jpg)  
Figure 3: Merging image classifiers: the plots illustrate the values of the loss function used for training the input networks when evaluated on a line segment connecting  $v$  and  $g_{\#}v'$ , where  $g$  is the output of each method. Values are averaged over all test images and networks and 3 random seeds.  
(b) CIFAR10 CNNs.

![](images/6edc95fa20303e0ca428a96f1eb23283517b1f38fc7c12a26f477ba6f7bc2e98.jpg)  
(c) STL10 CNNs.

where the inner product is the standard Frobenius inner product. The optimization problem, known as the linear assignment problem can be solved using the Hungarian algorithm.

As we carefully designed the components of  $F$  so that they are all equivariant to transposition and the action of  $G \times G$ , we obtain the following proposition:

Proposition 3. The architecture  $F$  satisfies the conditions specified in 1,2, namely for all  $(v,v^{\prime})\in \mathcal{V}\times \mathcal{V}$  and  $(g,g^{\prime})\in H$  we have:  $F(g_{\#}v,g_{\#}^{\prime}v^{\prime}) = g\cdot F(v,v^{\prime})\cdot g^{\prime T}$  and  $F(v,v^{\prime}) = F(v^{\prime},v)^{T}$ .

# 4.2 DATA GENERATION AND LOSS FUNCTIONS

Generating labeled data for the weight-alignment problem is hard due to the intractability of the problem. Therefore, we propose a combination of both unsupervised and supervised loss functions where we generate labeled examples synthetically from unlabeled examples, as specified below.

Data generation. Our initial training data consists of a finite set of weight space vectors  $D \subset \mathcal{V}$ . From that set, we generate two datasets consisting of pairs of weights for the alignment problem. First, we generate a labeled training set,  $D_{\mathrm{labeled}} = \{(v^j,v^{\prime j},t^j)\}_{j = 1}^{N_{\mathrm{labeled}}}$  for  $t^j = (T_1^j,\dots ,T_{M - 1}^j)\in G$ . This is done by sampling  $v^{j}\in D$  and defining  $v^{\prime j}$  as a permuted and noisy version of  $v^{j}$ . More formally, we sample a sequence of permutations  $t\in G$  and define  $v^{\prime j} = t_{\#}f_{\mathrm{aug}}(v^{j})$ , where  $f_{\mathrm{aug}}$  applies several weight-space augmentations, like adding binary and Gaussian noise, scaling augmentations for ReLU networks, etc. We then set the label of this pair to be  $t$ . In addition, we define an unlabeled dataset  $D_{\mathrm{unlabeled}} = \{(v^{j},v^{\prime j})\}_{j = 1}^{N_{\mathrm{unlabeled}}}$  where  $v^{j},v^{\prime j}\in \mathcal{V}$ .

Loss functions. The datasets above are used for training our architecture using the following loss functions. The labeled training examples in  $D_{\mathrm{labeled}}$  are used by applying a cross-entropy loss for each row  $i = 1, \dots, d_m$  in each output matrix  $m = 1, \dots, M - 1$ . This loss is denoted as  $\ell_{\mathrm{supervised}}(F(v, v'; \theta), t)$ . The unlabeled training examples are used in combination with two unsupervised loss functions. The first loss function aims to minimize the alignment loss in equation 2 directly by using the network output  $F(v, v'; \theta)$  as the permutation sequence. This loss is denoted as  $\ell_{\mathrm{alignment}}(v, v', \theta) = \| v - F(v, v'; \theta)_{\#} v' \|_2^2$ . The second unsupervised loss function aims to minimize the original loss function used to train the input networks on a line segment connecting the weights  $v$  and the transformed version of  $v'$  using the network output  $F(v, v'; \theta)$  as the permutation sequence. Concretely, let  $\mathcal{L}$  denote the original loss function for the weight vectors  $v, v'$ , the loss is defined as  $\ell_{\mathrm{LMC}}(v, v', \theta) = \mathcal{L}(\lambda v + (1 - \lambda) F(v, v'; \theta)_{\#} v')$  for  $\lambda$  sampled uniformly  $\lambda \sim U(0, 1)^1$ . This loss is similar to the STE method in Ainsworth et al. (2022) and the differentiable version in Peña et al. (2023). Our final goal is to minimize the parameters of  $F$  with respect to a linear (positive) combination of  $\ell_{\mathrm{alignment}}$ ,  $\ell_{\mathrm{LMC}}$  and  $\ell_{\mathrm{supervised}}$  applied to the appropriate datasets described above.

# 5 THEORETICAL ANALYSIS

Relation to the activation matching algorithm. In this subsection, we prove that our proposed architecture can simulate the activation matching algorithm, a heuristic for solving the weight align-

Table 1: MLP image classifiers: Results on aligning MNIST and CIFAR10 MLP image classifiers.  

<table><tr><td rowspan="2"></td><td colspan="2">MNIST (MLP)</td><td colspan="2">CIFAR10 (MLP)</td></tr><tr><td>Barrier ↓</td><td>AUC ↓</td><td>Barrier ↓</td><td>AUC ↓</td></tr><tr><td>Naive</td><td>2.007 ± 0.00</td><td>0.835 ± 0.00</td><td>0.927 ± 0.00</td><td>0.493 ± 0.00</td></tr><tr><td>Weight Matching</td><td>0.047 ± 0.00</td><td>0.011 ± 0.00</td><td>0.156 ± 0.00</td><td>0.068 ± 0.00</td></tr><tr><td>Activation Matching</td><td>0.024 ± 0.00</td><td>0.007 ± 0.00</td><td>0.066 ± 0.00</td><td>0.024 ± 0.00</td></tr><tr><td>Sinkhorn</td><td>0.027 ± 0.00</td><td>0.002 ± 0.00</td><td>0.183 ± 0.00</td><td>0.072 ± 0.00</td></tr><tr><td>WM + Sinkhorn</td><td>0.012 ± 0.00</td><td>0.000 ± 0.00</td><td>0.137 ± 0.00</td><td>0.050 ± 0.00</td></tr><tr><td>DEEP-ALIGN</td><td>0.005 ± 0.00</td><td>0.000 ± 0.00</td><td>0.078 ± 0.01</td><td>0.029 ± 0.00</td></tr><tr><td>DEEP-ALIGN + Sinkhorn</td><td>0.000 ± 0.00</td><td>0.000 ± 0.00</td><td>0.037 ± 0.00</td><td>0.004 ± 0.00</td></tr></table>

Table 2: CNN image classifiers: Results on aligning CIFAR10 and STL10 CNN image classifiers.  

<table><tr><td rowspan="2"></td><td colspan="2">CIFAR10 (CNN)</td><td colspan="2">STL10 (CNN)</td><td rowspan="2">Runtime (Sec) ↓</td></tr><tr><td>Barrier ↓</td><td>AUC ↓</td><td>Barrier ↓</td><td>AUC ↓</td></tr><tr><td>Naive</td><td>1.124 ± 0.01</td><td>0.524 ± 0.00</td><td>1.006 ± 0.00</td><td>0.650 ± 0.00</td><td>—</td></tr><tr><td>Weight Matching</td><td>0.661 ± 0.02</td><td>0.178 ± 0.01</td><td>0.859 ± 0.00</td><td>0.453 ± 0.00</td><td>0.21</td></tr><tr><td>Activation Matching</td><td>0.238 ± 0.01</td><td>0.000 ± 0.00</td><td>0.479 ± 0.00</td><td>0.250 ± 0.00</td><td>7.52</td></tr><tr><td>Sinkhorn</td><td>0.313 ± 0.01</td><td>0.000 ± 0.00</td><td>0.366 ± 0.00</td><td>0.163 ± 0.00</td><td>79.81</td></tr><tr><td>WM + Sinkhorn</td><td>0.333 ± 0.01</td><td>0.000 ± 0.00</td><td>0.371 ± 0.00</td><td>0.165 ± 0.00</td><td>80.02 = 0.21 + 79.81</td></tr><tr><td>DEEP-ALIGN</td><td>0.237 ± 0.01</td><td>0.000 ± 0.00</td><td>0.382 ± 0.01</td><td>0.182 ± 0.00</td><td>0.44</td></tr><tr><td>DEEP-ALIGN + Sinkhorn</td><td>0.081 ± 0.00</td><td>0.000 ± 0.00</td><td>0.232 ± 0.00</td><td>0.097 ± 0.00</td><td>80.25 = 0.44 + 79.81</td></tr></table>

ment problem suggested in Ainsworth et al. (2022). In a nutshell, this algorithm works by evaluating two neural networks on a set of inputs and finding permutations that align their activations by solving a linear assignment problem using the outer product matrix of the activations as a cost matrix for every layer  $m = 1,\dots ,M - 1$

Proposition 4. (DEEP-ALIGN can simulate activation matching) For any compact set  $K \subset \mathcal{V}$  and  $x_{1},\ldots ,x_{N} \in \mathbb{R}^{d_{0}}$ , there exists an instance of our architecture  $F$  and weights  $\theta$  such that for any  $v,v^{\prime}\in K$  for which the activation matching algorithm has a single optimal solution  $g\in G$  and another minor assumption specified in the appendix,  $F(v,v^{\prime};\theta)$  returns  $g$ .

This result offers an interesting interpretation of our architecture: the architecture can simulate activation matching while optimizing the input vectors  $x_{1},\ldots ,x_{N}$  as a part of their weights  $\theta$ .

Exactness. We now discuss the exactness of our algorithms. An alignment algorithm is said to be exact on some input  $(v, v')$  if it can be proven to successfully return the correct minimizer  $\mathcal{G}(v, v')$ . For NP-hard alignment problems such as weight alignment, exactness can typically be obtained when restricting it to 'tame' inputs  $(v, v')$ . Examples of exactness results in the alignment literature can be found in Aflalo et al. (2015); Dym & Lipman (2017); Dym (2018). The following proposition shows that (up to probability zero events) when  $v, v'$  are exactly related by some  $g \in G$ , our algorithm will retrieve  $g$  exactly:

Proposition 5 (DEEP-ALIGN is exact for perfect alignments). Let  $F$  denote the DEEP-ALIGN architecture with non-constant analytic activations and  $d \geq 2$  channels. Then, for Lebesgue almost every  $v \in \mathcal{V}$  and parameter vector  $\theta$ , and for every  $g \in G$ , we have that  $F(v, g_{\#}v, \theta) = g$ .

# 6 EXPERIMENTS

In this section, we evaluate DEEP-ALIGN on the task of aligning and merging neural networks. To support future research and the reproducibility of the results, we will make our source code and datasets publicly available upon publication.

Evaluation metrics. We use the standard evaluation metrics for measuring model merging (Ainsworth et al., 2022; Peña et al., 2023): Barrier and Area Under the Curve (AUC). For two inputs  $v, v'$  the Barrier is defined by  $\max_{\lambda \in [0,1]} \psi(\lambda) \equiv \mathcal{L}(\lambda v + (1 - \lambda)v') - (\lambda \mathcal{L}(v) + (1 - \lambda)\mathcal{L}(v'))$  where  $\mathcal{L}$  denote the loss function on the original task. Similarly, the AUC is defined as the integral

![](images/3441072c817ebfcd986f2b9de4d9ff555f65333a2e63232df896fa39dadd8835.jpg)  
(a) Sine Wave INRs.

![](images/004fd29374e84a8a40ce5542501a0fd18b487f8c651e46a12e7bb11036e248b7.jpg)  
Figure 4: Aligning INRs: The test barrier vs. the number of Sinkhorn iterations (relevant only for Sinkhorn or DEEP-ALIGN + Sinkhorn), using (a) sine wave and (b) CIFAR10 INRs. DEEP-ALIGN outperforms baseline methods or achieves on-par results.  
(b) CIFAR10 INRs.

of  $\psi$  over  $[0,1]$ . Lower is better for both metrics. Following previous works (Ainsworth et al., 2022; Peña et al., 2023), we bound both metrics by taking the maximum between their value and zero.

Compared methods. We compare the following approaches: (1) Naive: where two models are merged by averaging the models' weights without alignment. The (2) Weight matching and (3) Activation matching approaches proposed in Ainsworth et al. (2022). (4) Sinkhorn (Peña et al., 2023): This approach directly optimizes the permutation matrices using the task loss on the line segment between the aligned models (denoted  $\mathcal{C}_{Rnd}$  in Peña et al. (2023)). (5) WM + Sinkhorn: using the weight matching solution to initialize the Sinkhorn method. (6) DEEP-ALIGN: Our proposed method described in Section 4. (7) DEEP-ALIGN + Sinkhorn: Here, the output from the DEEP-ALIGN is used as an initialization for the Sinkhorn method.

Experimental details. Our method is first trained on a dataset of weight vectors and then applied to unseen weight vectors at test time, as is standard in learning setups. In contrast, baseline methods are directly optimized using the test networks. For the Sinkhorn and DEEP-ALIGN + Sinkhorn methods, we optimize the permutations for 1000 iterations. For the Activation Matching method, we calculate the activations using the entire train dataset. We repeat all experiments using 3 random seeds and report each metric's mean and standard deviation. For full experimental details see Appendix E.

# 6.1 RESULTS

Aligning classifiers. Here, we evaluate our method on the task of aligning image classifiers. We use four network datasets. Two datasets consist of MLP classifiers for MNIST and CIFAR10, and two datasets consist of CNN classifiers trained using CIFAR10 and STL10. This collection forms a diverse benchmark for aligning NN classifiers. The results are presented in Figure 7, Table 1 and Table 2. The alignment produced through a feed-forward pass with DEEP-ALIGN performs on par or outperforms all baseline methods. Initializing the Sinkhorn algorithm with our alignment (DEEP-ALIGN + Sinkhorn) further improves the results, and significantly outperforms all other methods. For the CNN alignment experiments, we report the averaged alignment time using 1K random pairs.

Aligning INRs. We use two datasets consisting of implicit neural representations (INRs). The first consists of Sine waves INRs of the form  $f(x) = \sin(ax)$  on  $[- \pi, \pi]$ , where  $a \sim U(0.5, 10)$ , similarly to the data used in Navon et al. (2023). We fit two views (independently trained weight vectors) for each value of  $a$  starting from different random initializations and the task is to align and merge the two INRs. We train our network to align pairs of corresponding views. The second dataset consists of INRs fitted to CIFAR10 images. We fit five views per image. The results are presented in Figure 4. DEEP-ALIGN, performs on par or outperforms all baseline methods. Moreover, using the output from the DEEP-ALIGN to initialize the Sinkhorn algorithm further improves this result, with a large improvement over the Sinkhorn baseline with random initialization.

Generalization to out-of-distribution data (OOD). Here, we evaluate the generalization capabilities of DEEP-ALIGN under distribution shift. We use the DEEP-ALIGN model trained on CIFAR10 CNN image classifiers and evaluate the generalization on two datasets. The first dataset consists of CNN classifiers trained on a version of CIFAR10 in which each image is rotated by a rotation

Table 3: Aligning OOD image classifiers, using a DEEP-ALIGN network trained on CIFAR10.  

<table><tr><td rowspan="2"></td><td colspan="2">Rotated CIFAR10</td><td colspan="2">STL10</td></tr><tr><td>Barrier ↓</td><td>AUC ↓</td><td>Barrier ↓</td><td>AUC ↓</td></tr><tr><td>Naive</td><td>1.077 ± 0.01</td><td>0.714 ± 0.00</td><td>1.006 ± 0.00</td><td>0.650 ± 0.00</td></tr><tr><td>Weight Matching</td><td>0.945 ± 0.02</td><td>0.550 ± 0.01</td><td>0.859 ± 0.00</td><td>0.453 ± 0.00</td></tr><tr><td>Activation Matching</td><td>0.586 ± 0.00</td><td>0.336 ± 0.00</td><td>0.479 ± 0.00</td><td>0.250 ± 0.00</td></tr><tr><td>Sinkhorn</td><td>0.596 ± 0.01</td><td>0.321 ± 0.00</td><td>0.366 ± 0.00</td><td>0.163 ± 0.00</td></tr><tr><td>DEEP-ALIGN</td><td>0.769 ± 0.01</td><td>0.453 ± 0.00</td><td>0.686 ± 0.01</td><td>0.373 ± 0.01</td></tr><tr><td>DEEP-ALIGN + Sinkhorn</td><td>0.430 ± 0.01</td><td>0.245 ± 0.00</td><td>0.357 ± 0.00</td><td>0.165 ± 0.00</td></tr></table>

degree sampled uniformly from  $U(-45, 45)$ . The second dataset consists of CNN image classifiers trained on the STL10 dataset. Importantly, we note that DEEP-ALIGN is evaluated on a distribution of models that is different than the one observed during training. In contrast, the baselines directly solve an optimization problem for each model pair within the test datasets. While DEEP-ALIGN significantly outperforms the Naive and WM baselines, it falls short in comparison to the Sinkhorn and AM methods, both of which are directly optimized using data from the new domain (NNs and images). Employing DEEP-ALIGN as an initialization for the Sinkhorn method consistently proves beneficial, with the DEEP-ALIGN + Sinkhorn approach yielding the most favorable results.

Aligning networks trained on disjoint datasets. Following Ainsworth et al. (2022), we experiment with aligning networks trained on disjoint datasets. One major motivation for such a setup is Federated learning (McMahan et al., 2017). In Federated Learning, the goal is to construct a unified model from multiple networks trained on separate and distinct datasets.

To that end, we split the CIFAR10 dataset into two splits. The first consists of  $95\%$  images from classes 0-4 and  $5\%$  of classes 5-9, and the second split is constructed accordingly with  $95\%$  of classes 5-9. We train the DEEP-ALIGN model to align CNN networks trained using the different datasets. For Sinkhorn and Activation Matching, we assume full access to the training data in the

optimization stage. For DEEP-ALIGN, we assume this data is accessible in the training phase. The results are presented in Figure 5. DEEP-ALIGN, along with the Sinkhorn and Activation Matching approaches, are able to align and merge the networks to obtain a network with lower loss compared to the original models. However, our approach is significantly more efficient at inference.

![](images/ada126328ce3ca5987dcb28514a81088a48a361554e27024fec8b0ebf27aa8ad.jpg)  
Figure 5: Merging networks trained on distinct subsets of CIFAR10.

# 7 CONCLUSION

We investigate the challenging problem of weight alignment in deep neural networks. The key to our approach, DEEP-ALIGN, is an equivariant architecture that respects the natural symmetries of the problem. At inference time DEEP-ALIGN can align unseen network pairs without the need for performing expensive optimization. DEEP-ALIGN, performs on par or outperforms optimization-based approaches while significantly reducing the runtime or improving the quality of the alignments. Furthermore, we demonstrate that the alignments of our method can be used to initialize optimization-based approaches. One limitation of our approach is the need for training a network. Although this can be a relatively time-consuming process, we only have to perform it once for each weight distribution. Furthermore, this procedure does not require labeled examples. To summarize, DEEP-ALIGN is the first architecture designed for weight alignment. It demonstrates superior performance over existing methods. The generalization capabilities of DEEP-ALIGN make it a promising and practical solution for applications that require weight alignment.

# REFERENCES

Yonathan Aflalo, Alexander Bronstein, and Ron Kimmel. On convex relaxation of graph isomorphism. Proceedings of the National Academy of Sciences, 112(10):2942-2947, 2015.  
Samuel K Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. arXiv preprint arXiv:2209.04836, 2022.  
Aditya Kumar Akash, Sixu Li, and Nicolas Garcia Trillos. Wasserstein barycenter-based model fusion and linear mode connectivity of neural networks. arXiv preprint arXiv:2210.06671, 2022.  
Bruno Andreis, Soro Bedionita, and Sung Ju Hwang. Set-based neural network encoding. arXiv preprint arXiv:2305.16625, 2023.  
Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: a methodological tour d'horizon. European Journal of Operational Research, 290(2): 405-421, 2021.  
Quentin Cappart, Didier Chételat, Elias B. Khalil, Andrea Lodi, Christopher Morris, and Petar Velickovic. Combinatorial optimization and reasoning with graph neural networks. CoRR, abs/2102.09544, 2021. URL https://arxiv.org/abs/2102.09544.  
Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred Hamprecht. Essentially no barriers in neural network energy landscape. In International conference on machine learning, pp. 1309-1318. PMLR, 2018.  
Nadav Dym. Exact recovery with symmetries for the doubly stochastic relaxation. SIAM Journal on Applied Algebra and Geometry, 2(3):462-488, 2018.  
Nadav Dym and Yaron Lipman. Exact recovery with symmetries for procrustes matching. SIAM Journal on Optimization, 27(3):1513-1530, 2017.  
Gabriel Eilertsen, Daniel Jonsson, Timo Ropinski, Jonas Unger, and Anders Ynnerman. Classifying the classifier: dissecting the weight space of neural networks. arXiv preprint arXiv:2002.05688, 2020.  
Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=dNigytemkL.  
Matthias Fey, Jan E Lenssen, Christopher Morris, Jonathan Masci, and Nils M Kriege. Deep graph matching consensus. arXiv preprint arXiv:2001.09621, 2020.  
C Daniel Freeman and Joan Bruna. Topology and geometry of half-rectified network optimization. arXiv preprint arXiv:1611.01540, 2016.  
Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson. Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31, 2018.  
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089, 2022.  
Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari, and Behnam Neyshabur. Repair: Renormalizing permuted activations for interpolation repair. arXiv preprint arXiv:2211.08403, 2022.  
Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. Advances in neural information processing systems, 30, 2017.  
Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie Jegelka. Sign and basis invariant networks for spectral graph representation learning. arXiv preprint arXiv:2202.13013, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.  
Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph networks. arXiv preprint arXiv:1812.09902, 2018.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273-1282. PMLR, 2017.  
Gonzalo Mena, David Belanger, Scott Linderman, and Jasper Snoek. Learning latent permutations with gumbel-sinkhorn networks. arXiv preprint arXiv:1802.08665, 2018.  
Boris Mityagin. The zero set of a real analytic function. arXiv preprint arXiv:1512.07276, 2015.  
Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, and Haggai Maron. Equivariant architectures for learning in deep weight spaces. arXiv preprint arXiv:2301.12780, 2023.  
Alex Nowak, Soledad Villar, Afonso S Bandeira, and Joan Bruna. A note on learning algorithms for quadratic assignment with graph neural networks. stat, 1050:22, 2017.  
Alex Nowak, Soledad Villar, Afonso S Bandeira, and Joan Bruna. Revised note on learning quadratic assignment with graph neural networks. In 2018 IEEE Data Science Workshop (DSW), pp. 1-5. IEEE, 2018.  
Fidel A Guerrero Pña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric Granger, and Marco Pedersoli. Re-basin via implicit sinkhorn differentiation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20237-20246, 2023.  
Daniel Selsam, Matthew Lamm, Benedikt Bunz, Percy Liang, Leonardo de Moura, and David L. Dill. Learning a sat solver from single-bit supervision. arXiv preprint arXiv:1802.03685, 2018.  
Richard Sinkhorn. Diagonal equivalence to matrices with prescribed row and column sums. The American Mathematical Monthly, 74(4):402-405, 1967.  
George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, and Judy Hoffman. Zipit! merging models from different tasks without training. arXiv preprint arXiv:2305.03053, 2023.  
Norman Tatro, Pin-Yu Chen, Payel Das, Igor Melnyk, Prasanna Sattigeri, and Rongjie Lai. Optimizing mode connectivity via neuron alignment. Advances in Neural Information Processing Systems, 33:15300-15311, 2020.  
Thomas Unterthiner, Daniel Keysers, Sylvain Gelly, Olivier Bousquet, and Ilya Tolstikhin. Predicting neural network accuracy from weights. arXiv preprint arXiv:2002.11448, 2020.  
Natalia Vesselinova, Rebecca Steinert, Daniel F Perez-Ramirez, and Magnus Boman. Learning combinatorial optimization on graphs: A survey with applications to networking. IEEE Access, 8:120388-120416, 2020.  
Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning, pp. 23965-23998. PMLR, 2022.  
Junchi Yan, Shuang Yang, and Edwin R Hancock. Learning for graph matching and related combinatorial optimization problems. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pp. 4988-4996. International Joint Conferences on Artificial Intelligence Organization, 2020.  
Tianshu Yu, Runzhong Wang, Junchi Yan, and Baoxin Li. Learning deep graph matching with channel-independent embedding and hungarian attention. In International conference on learning representations, 2019.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. Advances in neural information processing systems, 30, 2017.  
Andrei Zanfir and Cristian Sminchisescu. Deep learning of graph matching. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2684-2693, 2018.  
David W Zhang, Miltiadis Kofinas, Yan Zhang, Yunlu Chen, Gertjan J Burghouts, and Cees GM Snoek. Neural networks are graphs! graph neural networks for equivariant processing of neural networks. 2023.  
Allan Zhou, Kaien Yang, Kaylee Burns, Yiding Jiang, Samuel Sokota, J Zico Kolter, and Chelsea Finn. Permutation equivariant neural functionals. arXiv preprint arXiv:2302.14040, 2023a.  
Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J Zico Kolter, and Chelsea Finn. Neural functional transformers. arXiv preprint arXiv:2305.13546, 2023b.
