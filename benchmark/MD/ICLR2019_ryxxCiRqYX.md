# DEEP LAYERS AS STOCHASTIC SOLVERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We provide a novel perspective on the forward pass through a block of layers in a deep network. In particular, we show that a forward pass through a standard dropout layer followed by a linear layer and a non-linear activation is equivalent to optimizing a convex optimization objective with a single iteration of a  $\tau$ -nice Proximal Stochastic Gradient method. We further show that replacing standard Bernoulli dropout with additive dropout is equivalent to optimizing the same convex objective with a variance-reduced proximal method. By expressing both fully-connected and convolutional layers as special cases of a high-order tensor product, we unify the underlying convex optimization problem in the tensor setting and derive a formula for the Lipschitz constant  $L$  used to determine the optimal step size of the above proximal methods. We conduct experiments with standard convolutional networks applied to the CIFAR-10 and CIFAR-100 datasets, and show that replacing a block of layers with multiple iterations of the corresponding solver, with step size set via  $L$ , consistently improves classification accuracy.

# 1 INTRODUCTION

Deep learning has revolutionized computer vision and natural language processing and is increasingly applied throughout science and engineering (LeCun et al., 2015). This has motivated mathematical analysis of various aspects of deep networks, such as the capacity and uniqueness of their representations (Soatto & Chiuso, 2014; Papyan et al., 2018) and their global training convergence properties (Haeffele & Vidal, 2017). However, a complete characterization of deep networks remains elusive. For example, Bernoulli dropout layers are widely celebrated in improving generalization (Srivastava et al., 2014); however, a complete theoretical understanding of their behavior remains an open problem. While basic dropout layers have proven to be effective, there are many other types of dropout with various desirable properties that have been proposed (Molchanov et al., 2017). This raises many questions. Can the fundamental block of layers that consists of a dropout layer followed by a linear transformation and a non-linear activation be further improved for better generalization? Can the choice of dropout layer be made independently from the linear transformation and non-linear activation? Are there systematic ways to propose new types of dropout?

We attempt to address some of these questions by establishing a strong connection between the forward pass through a block of layers in a deep network and the solution of convex optimization problems of the following form:

$$
\underset {\mathbf {x} \in \mathbb {R} ^ {d}} {\text {m i n i m i z e}} F (\mathbf {x}) + g (\mathbf {x}), \quad F (\mathbf {x}) \stackrel {\text {d e f}} {=} \frac {1}{n} \sum_ {i} ^ {n} f _ {i} \left(\mathbf {a} _ {i} ^ {\top} \mathbf {x}\right). \tag {1}
$$

In particular, we show that a block of layers that consists of dropout followed by a linear transformation (fully-connected or convolutional) and a non-linear activation has close connections to applying stochastic solvers to (1). Interestingly, the choice of the stochastic optimization algorithm gives rise to commonly used dropout layers, such as Bernoulli and additive dropout, and to a family of other types of dropout layers that have not been explored before. As a special case, the stochastic algorithm reduces to a deterministic one, when the block in question does not include dropout.

Our contributions can be summarized as follows. (i) We show that a forward pass through a block that consists of Bernoulli dropout followed by a linear transformation and a non-linear activation is equivalent to a single iteration of  $\tau$ -nice Proximal Stochastic Gradient, Prox-SG (Xiao & Zhang, 2014) when it is applied to an instance of (1). We provide various conditions on  $g$  that recover (either

exactly or approximately) common non-linearities used in practice. (ii) We show that the same block with an additive dropout instead of Bernoulli dropout is equivalent to a single iteration of mS2GD (Konečný et al., 2016) – a mini-batching form of variance-reduced SGD (Johnson & Zhang, 2013) – applied to an instance of (1). (iii) By expressing both fully-connected and convolutional layers (referred to as linear throughout) as special cases of a high-order tensor product (Bibi & Ghanem, 2017), we derive a formula for the Lipschitz constant  $L$  of  $\nabla F(\mathbf{x})$ . As a consequence, we can compute the optimal step size for the stochastic solvers that correspond to blocks of layers. (iv) We validate our theoretical analysis experimentally by replacing blocks of layers in standard image classification networks with corresponding solvers, with principled setting of the step size as implied by our derivation of  $L$ , and showing that this improves the accuracy of the models.

# 2 RELATED WORK

Optimization algorithms can provide insight and guidance in the design of deep network architectures (Vogel & Pock, 2017; Yang et al., 2016; Zhang & Ghanem, 2018). For example, Yang et al. (2016) have proposed a deep network architecture for compressed sensing. Their network, dubbed ADMM-Net, is inspired by ADMM updates (Boyd et al., 2011) on the compressed sensing objective. Similarly, Zhang & Ghanem (2018) demonstrated that unrolling a proximal gradient descent solver (Beck & Teboulle, 2009) on the same problem can further improve performance. Amos & Kolter (2017) proposed to embed optimization problems, in particular linearly-constrained quadratic programs, as structured layers in deep networks. Meinhardt et al. (2017) replaced proximal operators in optimization algorithms by neural networks. Huang & Van Gool (2017) proposed a new matrix layer, dubbed ReEig, that applies a thresholding operation to the eigenvalues of intermediate feature representations that are stacked in matrix form. ReEig can be tightly connected to a proximal operator of the set of positive semi-definite matrices. Sulam et al. (2018) proposed a new architecture based on a sparse representation construct, Multi-Layer Convolutional Sparse Coding (ML-CSC), initially introduced by Papyan et al. (2016). Sparsity on the intermediate representations was enforced by a multi-layer form of basis pursuit.

This body of work has demonstrated the merits of connecting the design of deep networks with optimization algorithms in the form of structured layers. However, with few exceptions (Amos & Kolter, 2017; Sulam et al., 2018), previous works propose specialized architectures for specific tasks. Our work aims to contribute to a unified framework that connects optimization algorithms to deep layers.

A line of work aims to provide rigorous interpretation for dropout layers. For example, Wager et al. (2013) showed that dropout is connected to an adaptively balanced  $\ell_2$ -regularized loss. Wang & Manning (2013) showed that approximating the loss with a normal distribution leads to a faster form of dropout. Gal & Ghahramani (2016a;b) developed an insightful framework that connects dropout with approximate variational inference in Bayesian models. We provide a complementary perspective, in which dropout layers arise naturally in an optimization-driven framework for neural network design.

# 3 UNIFIED FRAMEWORK

This section is organized as follows. We introduce our notation and preliminaries in Section 3.1. In Section 3.2, we present a motivational example relating a single iteration of proximal gradient descent (Prox-GD) on (1) to the forward pass through a fully-connected layer followed by a nonlinear activation. We will show that several commonly used non-linear activations can be exactly or approximately represented as proximal operators of  $g(\mathbf{x})$ . In Section 3.3, we unify fully-connected and convolutional layers as special cases of a high-order tensor product. We propose a generic instance of (1) in a tensor setting, where we provide a formula for the Lipschitz constant  $L$  of the finite sum structure of (1). In Section 3.4, we derive an intimate relation between stochastic solvers, namely  $\tau$ -nice Prox-SG and mS2GD, and two types of dropout layers. Figure 1 shows an overview of the connections that will be developed.

![](images/a0020fa3a385a4e721e051bcd35c90172f0999f6adfdcfaf3e8b3ee65f8e2404.jpg)  
Figure 1: An overview of the tight relation between a single iteration of a stochastic solver and the forward pass through the  $l^{\mathrm{th}}$  layer in a network that consists of dropout followed by a linear transformation and a non-linear activation. We study an instance of problem (1) with quadratic  $F(\mathbf{x})$ , where  $\mathbf{x}^{l-1}$  are the input activations and  $\mathbf{x}^l$ , the variables being optimized, correspond to the output activations. Varying the type of stochastic solver changes the nature of the dropout layer, while the prior  $g(\mathbf{x})$  on the output activations determines the non-linearity  $\operatorname{Prox}_{\frac{1}{L} g}(.)$ .

![](images/62de63d7abb2e83a864e70e6b70e622d02fbd12df5b67af2469802cc4f491625.jpg)

![](images/da27df4f6d172699d6838d72ceb5621b8a3191c3fa1d9c9a907469a1db8100a8.jpg)

# 3.1 NOTATION AND PRELIMINARIES

As we will be working with tensors, we will follow the tensor notation of Kolda & Bader (2009). The order of a tensor is the number of its dimensions. In particular, scalars are tensors of order zero, vectors are tensors of order one, and matrices are tensors of order two. We denote scalars by lowercase letters  $a$ , vectors by bold lowercase letters  $\mathbf{a}$ , and matrices by bold capital letters  $\mathbf{A}$ . We use subscripts  $\mathbf{a}_i$  to refer to individual elements in a vector. Tensors of order three or more will be denoted by cursive capital letters  $\mathcal{A} \in \mathbb{R}^{J_1 \times J_2 \times \dots \times J_n}$ . Throughout the paper, we will handle tensors that are of at most order four. High-order tensors with a second dimension of size equal to one are traditionally called vector tensors and denoted as  $\vec{\mathcal{A}} \in \mathbb{R}^{J_1 \times 1 \times J_3 \times J_4}$ . We use  $\mathcal{A}(i,j,k,z)$  to refer to an element in a tensor and  $\mathcal{A}(i,j,k,:)$  to refer to a slice of a tensor. The inner product between tensors of the same size is denoted as  $\langle \mathcal{A},\mathcal{B}\rangle = \sum_{i_1,\ldots ,i_N}\mathcal{A}(i_1,\ldots ,i_N)\mathcal{B}(i_1,\ldots ,i_N)$ . The squared Frobenius norm of a tensor  $\mathcal{A}$  is defined as  $\| \mathcal{A}\| _F^2 = \langle \mathcal{A},\mathcal{A}\rangle$ . Lastly, the superscripts  $\top$  and  $\mathbf{H}$  are used to denote the transpose and the Hermitian transpose, respectively.

# 3.2 MOTIVATIONAL INSIGHT: NON-LINEAR ACTIVATIONS AS PROXIMAL OPERATORS

As a motivating example, we consider a simple linear layer in a deep network that is followed by a non-linear activation  $\rho$ , i.e.  $\mathbf{x}^l = \rho (\mathbf{A}\mathbf{x}^{l - 1} + \mathbf{b})$ , where  $\mathbf{A}\in \mathbb{R}^{n_2\times n_1}$  and  $\mathbf{b}\in \mathbb{R}^{n_2}$  are the weights and biases of the layer and  $\mathbf{x}^{l - 1}$  and  $\mathbf{x}^l$  are the input and output activations, respectively. Now consider an instance of (1) with a convex function  $g(\mathbf{x})$  and

$$
F (\mathbf {x} ^ {l}) = \frac {1}{2} \| \mathbf {A} ^ {\top} \mathbf {x} ^ {l} - \mathbf {x} ^ {l - 1} \| ^ {2} - \mathbf {b} ^ {\top} \mathbf {x} = \frac {1}{2} \sum_ {i} ^ {n _ {1}} \left(\mathbf {A} ^ {\top} (i,:) \mathbf {x} ^ {l} - \mathbf {x} _ {i} ^ {l - 1}\right) ^ {2} - \mathbf {b} ^ {\top} \mathbf {x} ^ {l}, \tag {2}
$$

where  $\mathbf{A}^{\top}(i,:)$  is the  $i^{\text{th}}$  row of  $\mathbf{A}^{\top}$ . Such an objective can be optimized iteratively using Prox-GD with the following update equation:

$$
\mathbf {x} ^ {l} \leftarrow \operatorname {P r o x} _ {\frac {1}{L} g} \left(\left(\mathbf {I} - \frac {1}{L} \mathbf {A} \mathbf {A} ^ {\top}\right) \mathbf {x} ^ {l} + \frac {1}{L} \left(\mathbf {A} \mathbf {x} ^ {l - 1} + \mathbf {b}\right)\right), \tag {3}
$$

where the Lipschitz constant  $L = \lambda_{\mathrm{max}}\left(\mathbf{A}\mathbf{A}^{\top}\right)$

By initializing the iterative optimization at  $\mathbf{x}^l = \mathbf{0}$ , it becomes clear that a single iteration of (3) is equivalent to a fully-connected layer followed by a non-linearity that is implemented by the proximal operator (Fawzi et al., 2015). The choice of  $g(\mathbf{x})$  determines the specific form of the non-linearity  $\rho$ . Several popular activation functions can be traced back to their corresponding  $g(\mathbf{x})$ . The ReLU, which enforces non-negative output activations, corresponds to  $g(\mathbf{x}) = \mathbb{1}_{\mathbf{x} \geq 0}$ ; the corresponding instance of problem (1) is a non-negative quadratic program. Similar observations for the ReLU have been made in other contexts (Amos & Kolter, 2017; Papyan et al., 2017). We observe that

<table><tr><td>g(x)</td><td>1x≥0</td><td>γ/2 ∑i max2 (−xi − λ, 0)</td><td>-γlog(xi)</td><td>-γlog(1 − xi2)</td></tr><tr><td rowspan="2">Proxg(η)</td><td rowspan="2">max(0, η)</td><td>{η-γλ/1+γ if η ≤ -λ</td><td rowspan="2">1/2η + √1/4η2+γ</td><td rowspan="2">Root of cubic polynomial</td></tr><tr><td>η if η ≥ -λ</td></tr><tr><td>Shape</td><td>/</td><td>/</td><td>/</td><td>/</td></tr><tr><td>Activation</td><td>=ReLU(η)</td><td>=LeakyReLU(η)</td><td>≈Softplus(η)</td><td>≈Tanh(η)</td></tr></table>

Table 1: Different choices of  $g(\mathbf{x})$ , their corresponding proximal operators, and their relation to common activation functions. Squared hinge loss regularization of the activations yields a generalized Leaky ReLU. Log-barriers recover smooth activations, such as SoftPlus, Tanh, or Sigmoid. Derivations can be found in supplementary material.

many other activation functions fit this framework. For example, when  $g(\mathbf{x})$  is a squared hinge loss, i.e.  $\frac{\gamma}{2}\sum_{i}\max^{2}(-\mathbf{x}_{i} - \lambda ,0)$ , a single update of (3) is equivalent to a linear layer followed by a Leaky ReLU. Table 1 lists some other choices of  $g(\mathbf{x})$  and their induced activations.

Note that  $g(\mathbf{x})$  is not required to exhibit a simple, coordinate-wise separable structure. More complex functions can be used, as long as the proximal operator is easy to evaluate. Interesting examples arise when the output activations have matrix structure. For instance, one can impose nuclear norm regularization  $g(\mathbf{X}) = \| \mathbf{X}\|_{*}$  to encourage  $\mathbf{X}$  to be low rank. Alternatively, one can enforce positive semi-definite structure on the matrix  $\mathbf{X}$  by defining  $g(\mathbf{X}) = \mathbb{1}_{\mathbf{X}\succeq 0}$ . A similar activation has been used for higher-order pooling (Huang & Van Gool, 2017).

In what follows, we will show that this connection can be further extended to explain dropout layers. Interestingly, specific forms of dropout do not arise from particular forms of objective (1), but from different stochastic optimization algorithms that are applied to it.

# 3.3 UNIFYING FULLY-CONNECTED AND CONVOLUTIONAL LAYERS

Before presenting our main results on the equivalence between a forward pass through a block of layers and solving (1) with stochastic algorithms, we provide some key lemmas. These lemmas will be necessary for a unified treatment of fully-connected and convolutional layers as generic linear layers. This generic treatment will enable efficient computation of the Lipschitz constant for both fully-connected and convolutional layers.

Lemma 1. Consider the  $l^{\mathrm{th}}$  convolutional layer in a deep network with some non-linear activation, e.g.  $\operatorname{Prox}_g(.)$ , where the weights  $\mathcal{A} \in \mathbb{R}^{n_2 \times n_1 \times W \times H}$ , biases  $\vec{B} \in \mathbb{R}^{n_2 \times 1 \times W \times H}$ , and input activations  $\vec{x}^{l-1} \in \mathbb{R}^{n_1 \times 1 \times W \times H}$  are stacked into  $4^{\mathrm{th}}$ -order tensors. We can describe the layer as

$$
\vec {\mathcal {X}} ^ {l} = \operatorname {P r o x} _ {g} \left(\mathcal {A} * _ {\mathrm {H O}} \vec {\mathcal {X}} ^ {l - 1} + \vec {\mathcal {B}}\right), \tag {4}
$$

where  $\oplus_{\mathrm{HO}}$  is the high-order tensor product. Here  $n_1$  is the number of input features,  $n_2$  is the number of output features (number of filters), and  $W$  and  $H$  are the spatial dimensions of the features. As a special case, a fully-connected layer follows naturally, since  $\oplus_{\mathrm{HO}}$  reduces to a matrix-vector multiplication when  $W = H = 1$ .

The proof can be found in supplementary material. Note that the order of the dimensions is essential in this notation, as the first dimension in  $\mathcal{A}$  corresponds to the number of independent filters while the second corresponds to the input features that will be aggregated after the 2D convolutions. Also note that according to the definition of  $\text{串}_{\mathrm{HO}}$  in (Bibi & Ghanem, 2017), the spatial size of the filters in  $\mathcal{A}$ , namely  $W$  and  $H$ , has to match the spatial dimensions of the input activations  $\vec{x}^{l - 1}$ , since the operator  $\text{串}_{\mathrm{HO}}$  performs 2D circular convolutions while convolutions in deep networks are 2D linear convolutions. This is not a restriction, since one can perform linear convolution through a zero-padded circular convolution. Lastly, we assume that the values in  $\vec{B}$  are replicated along the spatial dimensions  $W$  and  $H$  in order to recover the behaviour of biases in deep networks.

Given this notation, we will refer to either a fully-connected or a convolutional layer as a linear layer throughout the rest of the paper. Since we are interested in a generic linear layer followed by

a non-linearity, we will consider the tensor quadratic version of  $F(\mathbf{x})$ , denoted as  $F(\vec{\mathcal{X}})$ :

$$
\underset {\vec {\mathcal {X}}} {\arg \min } \frac {1}{n _ {1}} \sum_ {i} ^ {n _ {1}} \underbrace {\frac {n _ {1}}{2} \| \mathcal {A} ^ {\mathbf {H}} (i , : , : , :) * _ {\mathrm {H O}} \vec {\mathcal {X}} - \vec {\mathcal {X}} ^ {l - 1} \| _ {F} ^ {2} - \langle \vec {\mathcal {B}}, \vec {\mathcal {X}} \rangle} _ {f _ {i} (\vec {\mathcal {X}})} + g (\vec {\mathcal {X}}). \tag {5}
$$

Note that if  $\mathcal{A} \in \mathbb{R}^{n_2 \times n_1 \times W \times H}$ , then  $\mathcal{A}^{\mathrm{H}} \in \mathbb{R}^{n_1 \times n_2 \times W \times H}$ , where each of the frontal slices of  $\mathcal{A}(:, :, i, j)$  is transposed and each filter,  $\mathcal{A}(i, j, :, :)$ , is rotated by  $180^\circ$ . This means that  $\mathcal{A}^{\mathrm{H}} \otimes_{\mathrm{HO}} \vec{\mathcal{X}}$  aggregates the  $n_2$  filters after performing 2D correlations. This is performed  $n_1$  times independently. This operation is commonly referred to as a transposed convolution. Details can be found in supplementary material.

Next, the following lemma provides a practical formula for the computation of the Lipschitz constant  $L$  of the finite sum part of (5):

Lemma 2. The Lipschitz constant  $L$  of  $\nabla F(\vec{\mathcal{X}})$  as defined in (5) is given by

$$
L = \max  _ {i \in \{1, 2, \dots , W \}, j \in \{1, 2, \dots , H \}} \left\{\lambda_ {\max } \left(\hat {\mathcal {A}} (:,:) i, j) \hat {\mathcal {A}} ^ {\mathbf {H}} (:,:) i, j)\right) \right\}, \tag {6}
$$

where  $\hat{\mathcal{A}}$  is the  $2D$  discrete Fourier transform along the spatial dimensions  $W$  and  $H$ .

The proof can be found in supplementary material. Lemma 2 states that the Lipschitz constant  $L$  is the maximum among the set of maximum eigenvalues of all the possible  $W \times H$  combinations of the outer product of frontal slices  $\hat{\mathcal{A}}(:, :, i, j) \hat{\mathcal{A}}^{\mathrm{H}}(:, :, i, j)$ . Note that if  $W = H = 1$ , then  $\hat{\mathcal{A}} = \mathcal{A} \in \mathbb{R}^{n_2 \times n_1}$  since the 2D discrete Fourier transform of scalars (i.e. matrices of size  $1 \times 1$ ) is an identity mapping. As a consequence, we can simplify (6) to  $L = \max_{i = j = 1} \{ \lambda_{\max}(\mathcal{A}(:, :, i, j) \mathcal{A}^{\mathrm{H}}(:, :, i, j)) \} = \lambda_{\max}(\mathcal{A} \mathcal{A}^\top)$ , which recovers the Lipschitz constant for fully-connected layers.

# 3.4 DROPOUT LAYERS AS VARIANTS OF STOCHASTIC SOLVERS

In this subsection, we present two propositions. The first shows the relation between standard Bernoulli dropout, BerDropout $_p$  (Srivastava et al., 2014), and  $\tau$ -nice Prox-SG. The second proposition relates additive dropout, AddDropout, to mS2GD (Konečný et al., 2016). We will first introduce a generic notion of sampling from a set. This is essential as the stochastic algorithms sample unbiased function estimates from the set of  $n_1$  functions in (5).

Definition 3.1. (Gower et al., 2018). A sampling is a random set-valued mapping with values being the subsets of  $[n_1] = \{1, \dots, n_1\}$ . A sampling  $S$  is  $\tau$ -nice if it is uniform, i.e.  $\operatorname{Prob}(i \in S) = \operatorname{Prob}(j \in S) \forall i, j$ , and assigns equal probabilities to all subsets of  $[n_1]$  of cardinality  $\tau$  and zero probability to all others.

Various other types of sampling can be found in (Gower et al., 2018). We are now ready to present our first proposition.

Proposition 1. A single iteration of Prox-SG with  $\tau$ -nice sampling  $S$  on (5) with  $\tau = (1 - p)n_{1}$ , zero initialization, and unit step size can be shown to exhibit the update

$$
\operatorname {P r o x} _ {\frac {1}{L} g} \left(\frac {n _ {1}}{\tau} \sum_ {i \in S} \mathcal {A} (:, i,:) * _ {H O} \vec {\mathcal {X}} ^ {l - 1} (i,:),:,:) + \vec {\mathcal {B}}\right), \tag {7}
$$

which is equivalent to a forward pass through a BerDropout $_p$  layer that drops exactly  $n_1p$  input activations followed by a linear layer and a non-linear activation.

We provide a simplified sketch for fully-connected layers here. The detailed proof is in the supplement. To see how (7) reduces to the functional form of  $\mathrm{BerDropout}_p$  followed by a fully-connected layer and a non-linear activation, consider  $W = H = 1$ . The argument of  $\mathrm{Prox}_{\frac{1}{L} g}$  in (7) (without the bias term) reduces to

$$
\frac {n _ {1}}{\tau} \sum_ {i \in S} \mathcal {A} (,: i, :,:) \circledast_ {H O} \vec {\mathcal {X}} ^ {l - 1} (i, :,:,:) = \frac {n _ {1}}{\tau} \sum_ {i \in S} \mathcal {A} (,: i) \vec {\mathcal {X}} ^ {l - 1} (i) = \frac {n _ {1}}{\tau} \mathcal {A} \operatorname {B e r D r o p o u t} _ {p} (\vec {\mathcal {X}} ^ {l - 1}). \tag {8}
$$

The first equality follows from the definition of  $\ast_{HO}$ , while the second equality follows from trivially reparameterizing the sum, with  $\mathrm{BerDropout}_p(.)$  being equivalent to a mask that zeroes out exactly  $pn_1$  input activations. Note that if  $\tau$ -nice Prox-SG was replaced with Prox-GD, i.e.  $\tau = n_1$ , then this corresponds to having a  $\mathrm{BerDropout}_p$  layer with dropout rate  $p = 0$ ; thus, (8) reduces to  $\mathcal{A}$ $\mathrm{BerDropout}_p(\vec{\mathcal{X}}^{l - 1}) = \mathcal{A}\vec{\mathcal{X}}^{l - 1}$ , which recovers our motivating example (3) that relates Prox-GD with the forward pass through a fully-connected layer followed by a non-linearity. Note that Proposition 1 directly suggests how to apply dropout to convolutional layers. Specifically, complete input features from  $n_1$  should be dropped and the 2D convolutions should be performed only on the  $\tau$ -sampled subset, where  $\tau = (1 - p)n_1$ .

Similarly, the following proposition shows that a form of additive dropout, AddDropout, can be recovered from a different choice of stochastic solver.

Proposition 2. A single outer-loop iteration of mS2GD (Konečný et al., 2016) with unit step size and zero initialization is equivalent to a forward pass through an AddDropout layer followed by a linear layer and a non-linear activation.

The proof is given in the supplement. It is similar to Proposition 1, with mS2GD replacing  $\tau$ -nice Prox-SG. Note that any variance-reduced algorithm where one full gradient is computed at least once can be used here as a replacement for mS2GD. For instance, one can show that the serial sampling version of mS2GD, S2GD (Konečný et al., 2016), and SVRG (Johnson & Zhang, 2013) can also be used. Other algorithms such as Stochastic Coordinate Descent (Richtárik & Takáč, 2016) with arbitrary sampling are discussed in the supplement.

# 4 EXPERIMENTS

A natural question arises as a consequence of our framework: If common layers in deep networks can be understood as a single iteration of an optimization algorithm, what happens if the algorithm is applied for multiple iterations? We empirically answer this question in our experiments. In particular, we embed solvers as a replacement to their corresponding blocks of layers and show that this improves the accuracy of the models without an increase in the number of network parameters.

Experimental setup. We perform experiments on CIFAR-10 and CIFAR-100 (Krizhevsky & Hinton, 2009). In all experiments, training was conducted on  $90\%$  of the training set while  $10\%$  was left for validation. The networks used in the experiments are variants of LeNet (LeCun et al., 1999), AlexNet (Krizhevsky et al., 2012), and VGG16 (Simonyan & Zisserman, 2014). We used stochastic gradient descent with a momentum of 0.9 and a weight decay of  $5 \times 10^{-4}$ . The learning rate was set to  $(10^{-2}, 10^{-3}, 10^{-4})$  for the first, second, and third 100 epochs, respectively. For finetuning, the learning rate was initially set to  $10^{-3}$  and reduced to  $10^{-4}$  after 100 epochs. Moreover, when a block of layers is replaced with a deterministic solver, i.e. Prox-GD, the step size is set to the optimal constant  $1 / L$ , where  $L$  is computed according to Lemma 2 and updated every epoch. In Prox-SG, a decaying step size is necessary for convergence; therefore, the step size is exponentially decayed as suggested by Bottou (2012), where the initial step size is again set according to Lemma 2. Finally, to guarantee convergence of the stochastic solvers, we add the strongly convex function  $\frac{\lambda}{2} \| \vec{\mathcal{X}} \|_F^2$  to the finite sum in (5), where we set  $\lambda = 10^{-3}$  in all experiments. Note that for networks that include a stochastic solver, the network will be stochastic at test time. We thus report the average accuracy and standard deviation over 20 trials.

Replacing fully-connected layers with solvers. In this experiment, we demonstrate that (i) training networks with solvers replacing one or more blocks of layers can improve accuracy when trained from scratch. (ii) The improvement is consistently present when one or more blocks are replaced with solvers at different layers in the network. To do so, we train a variant of LeNet on the CIFAR-10 dataset with two BerDropout $_p$  layers. The last two layers are fully-connected layers with ReLU activation. We consider three variants of this network: Both fully-connected layers are augmented with BerDropout $_p$  (LeNet-D-D), only the last layer is augmented with BerDropout $_p$  (LeNet-ND-D), and finally only the penultimate layer is augmented with BerDropout $_p$  (LeNet-D-ND). In all cases, we set the dropout rate to  $p = 0.5$ . We replace the BerDropout $_p$  layers with their corresponding stochastic solvers and run them for 10 iterations with  $\tau = n_1 / 2$  (the setting corresponding to a dropout rate of  $p = 0.5$ ). We train these networks from scratch using the same procedure as the baseline networks.

The results are summarized in Table 2. It can be seen that replacing  $\mathrm{BerDropout}_p$  with the corresponding stochastic solver ( $\tau$ -nice Prox-SG) improves performance significantly, for any choice of layer. The results indicate that networks that incorporate stochastic solvers can be trained stably and achieve desirable generalization performance.

<table><tr><td></td><td colspan="2">LeNet-D-D</td><td colspan="2">LeNet-D-ND</td><td colspan="2">LeNet-ND-D</td></tr><tr><td>Baseline</td><td colspan="2">64.39%</td><td colspan="2">71.72%</td><td colspan="2">68.54%</td></tr><tr><td>Prox-SG</td><td colspan="2">72.86% ± 0.177</td><td colspan="2">75.20% ± 0.205</td><td colspan="2">76.23% ± 0.206</td></tr></table>

Table 2: Comparison in accuracy between variants of the LeNet architecture on the CIFAR-10 dataset. The variants differ in the location (D or ND) and number of BerDropout $_p$  layers for both the baseline networks and their stochastic solver counterpart Prox-SG. Accuracy consistently improves when Prox-SG is used. Accuracy is reported on the test set.

Convolutional layers and larger networks. We now demonstrate that solvers can be used to improve larger networks. We conduct experiments with variants of AlexNet $^1$  and VGG16 on both CIFAR-10 and CIFAR-100. We start by training strong baselines for both AlexNet and VGG16, achieving  $77.3\%$  and  $92.56\%$  test accuracy on CIFAR-10, respectively. Note that performance on this dataset is nearly saturated. We then replace the first convolutional layer in AlexNet with the deterministic Prox-GD solver, since this layer is not preceded by a dropout layer. The results are summarized in Table 3. We observe that finetuning the baseline network with the solver leads to an improvement of  $\approx 1.2\%$ , without any change in the network's capacity. A similar improvement is observed on the harder CIFAR-100 dataset.

<table><tr><td></td><td>AlexNet</td><td>AlexNet-Prox-GD</td></tr><tr><td>CIFAR-10</td><td>77.30%</td><td>78.51%</td></tr><tr><td>CIFAR-100</td><td>44.20%</td><td>45.53%</td></tr></table>

Table 3: Replacing the first convolutional layer of AlexNet by the deterministic Prox-GD solver yields consistent improvement in test accuracy on CIFAR-10 and CIFAR-100.

Results on VGG16 are summarized in Table 4. Note that VGG16 has two fully-connected layers, which are preceded by a BerDropout layer with dropout rate  $p = 0.5$ . We start by replacing only the last layer with Prox-SG with 30 iterations and  $\tau = n_1 / 2$  (VGG16-Prox-SG-ND-D). We further replace both fully-connected layers that include BerDropout  $p$  with solvers (VGG16-Prox-SG-D-D). We observe comparable performance for both settings on CIFAR-10. We conjecture that this might be due to the dataset being close to saturation. On CIFAR-100, a more pronounced increase in accuracy is observed, where VGG-16-Prox-SG-ND-D outperforms the baseline by about  $0.7\%$ .

We further replace the stochastic solver with a deterministic solver and leave the dropout layers unchanged. We denote this setting as VGG16-Prox-GD in Table 4. Interestingly, this setting performs the best on CIFAR-10 and comparably to VGG16-Prox-SG-ND-D on CIFAR-100.

<table><tr><td></td><td>VGG16</td><td>VGG16-Prox-SG-ND-D</td><td>VGG16-Prox-SG-D-D</td><td>VGG16-Prox-GD</td></tr><tr><td>CIFAR-10</td><td>92.56%</td><td>92.44% ± 0.028</td><td>92.57% ± 0.029</td><td>92.80%</td></tr><tr><td>CIFAR-100</td><td>70.27%</td><td>70.95% ± 0.042</td><td>70.44% ± 0.077</td><td>71.10%</td></tr></table>

Table 4: Experiments with the VGG16 architecture on CIFAR-10 and CIFAR-100. Accuracy is reported on the test set.

Dropout rate vs.  $\tau$ -nice sampling. In this experiment, we demonstrate that the improvement in performance is still consistently present across varying dropout rates. Since Proposition 1 has established a tight connection between the dropout rate  $p$  and the sampling rate  $\tau$  in (5), we show that for different choices of dropout rate the baseline performance can always be improved by replacing a

block of layers with a stochastic solver with the corresponding sampling rate  $\tau$ . We conduct experiments with VGG16 on CIFAR-100. We train four different baseline models with varying choices of dropout rate  $p \in \{0.1, 0.9, 0.95\}$  for the last layer. We then replace this block with a stochastic solver with a sampling rate  $\tau$  and finetune the network.

Table 5 reports the accuracy of the baselines (BerDropout $_p$ ) for varying dropout rates and compares to the accuracy of the stochastic solver with corresponding  $\tau$  (Prox-SG). Note that since the last layer has  $n_1 = 512$  input activations, the corresponding sampling rates  $\tau$  are  $\{461, 256, 51, 26\}$ . With a high dropout rate  $p$ , the performance of the baseline network drops drastically. When using the stochastic solver, we observe a much more graceful drop in performance. For example, with a sampling rate  $\tau$  that corresponds to an extreme dropout rate of  $p = 0.95$  (i.e.  $95\%$  of all input activations are masked out), the baseline network with BerDropout $_p$  suffers a  $56\%$  reduction in accuracy while the stochastic solver declines by only  $5\%$ .

<table><tr><td colspan="2">Baseline</td><td colspan="2">Prox-SG</td></tr><tr><td>Dropout rate p</td><td>Accuracy</td><td>Sampling rate τ</td><td>Accuracy</td></tr><tr><td>0.10</td><td>70.56%</td><td>461</td><td>70.51% ± 0.0198</td></tr><tr><td>0.50</td><td>70.27%</td><td>256</td><td>70.95% ± 0.0419</td></tr><tr><td>0.90</td><td>68.34%</td><td>51</td><td>69.19% ± 0.0589</td></tr><tr><td>0.95</td><td>30.61%</td><td>26</td><td>67.42% ± 0.0774</td></tr></table>

Table 5: Comparison of the VGG16 architecture trained on CIFAR-100 with varying dropout rates  $p$  in the last BerDropout  ${}_{p}$  layer. We compare the baseline to its stochastic solver counterpart with corresponding  $\tau  = \left( {1 - p}\right) {n}_{1}$  . Accuracy is reported on the test set.

In summary, our experiments show that replacing common layers in deep networks with stochastic solvers can lead to better performance without increasing the number of parameters in the network. The resulting networks are stable to train and exhibit high accuracy in cases where standard dropout has proven to be problematic, such as high dropout rates.

# 5 DISCUSSION

We have presented equivalences between layers in deep networks and stochastic solvers, and have shown that this can be leveraged to improve accuracy. The presented relationships open many doors for future work. For instance, our framework shows an intimate relation between a dropout layer and the sampling  $S$  from the set  $[n_1]$  in a stochastic algorithm. As a consequence, one can borrow theory from the stochastic optimization literature to propose new types of dropout layers. For example, consider a serial importance sampling strategy with Prox-SG to solve (5) (Zhao & Zhang, 2015; Xiao & Zhang, 2014), where serial sampling is the sampling that satisfies  $\operatorname{Prob}(i \in S, j \in S) = 0$ . A serial importance sampling  $S$  from the set of functions  $f_i(\vec{\mathcal{X}})$  is the sampling such that  $\operatorname{Prob}(i \in S) \propto \| \nabla f_i(\vec{\mathcal{X}}) \| \propto L_i$ , where  $L_i$  is the Lipschitz constant of  $\nabla f_i(\vec{\mathcal{X}})$ , i.e., each function from the set  $[n_1]$  is sampled with a probability proportional to the norm of the gradient of the function. This sampling strategy is the optimal serial sampling  $S$  that maximizes the rate of convergence solving (5) (Zhao & Zhang, 2015). From a deep layer perspective, performing Prox-SG with importance sampling for a single iteration is equivalent to a forward pass through the same block of layers with a new dropout layer. Such a dropout layer will keep each input activation with a non-uniform probability proportional to the norm of the gradient. This is in contrast to BerDropout $_p$  where all input activations are kept with an equal probability  $1 - p$ . Other types of dropout arise when considering non-serial importance sampling where  $|S| = \tau > 1$ .

In summary, we have presented equivalences between stochastic solvers on a particular class of convex optimization problems and a forward pass through a dropout layer followed by a linear layer and a non-linear activation. Inspired by these equivalences, we have demonstrated empirically on multiple datasets and network architectures that replacing such network blocks with their corresponding stochastic solvers improves the model's accuracy. We hope that the presented framework will contribute to a principled understanding of the theory and practice of deep network architecture.

# REFERENCES

Brandon Amos and J. Zico Kolter. OptNet: Differentiable optimization as a layer in neural networks. In International Conference on Machine Learning (ICML), 2017.  
Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2009.  
Adel Bibi and Bernard Ghanem. High order tensor formulation for convolutional sparse coding. In International Conference on Computer Vision (ICCV), 2017.  
Léon Bottou. Stochastic gradient descent tricks. In Neural networks: Tricks of the trade. Springer, 2012.  
Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein, et al. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine learning, 2011.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition (CVPR), 2009.  
Alhussein Fawzi, Mike Davies, and Pascal Frossard. Dictionary learning for fast classification based on soft-thresholding. International Journal of Computer Vision, 2015.  
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning (ICML), 2016a.  
Yarin Gal and Zoubin Ghahramani. A theoretically grounded application of dropout in recurrent neural networks. In Advances in Neural Information Processing Systems (NIPS), 2016b.  
Robert M Gower, Peter Richtárik, and Francis Bach. Stochastic quasi-gradient methods: Variance reduction via jacobian sketching. arXiv preprint arXiv:1805.02632, 2018.  
Benjamin D Haeffele and René Vidal. Global optimality in neural network training. In Computer Vision and Pattern Recognition (CVPR), 2017.  
Zhiwu Huang and Luc J Van Gool. A riemannian network for spd matrix learning. In Association for the Advancement of Artificial Intelligence (AAAI), 2017.  
Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in Neural Information Processing Systems (NIPS), 2013.  
Misha E Kilmer and Carla D Martin. Factorization strategies for third-order tensors. Linear Algebra and its Applications, 2011.  
Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM review, 2009.  
Jakub Konečny, Jie Liu, Peter Richtárik, and Martin Takáč. Mini-batch semi-stochastic gradient descent in the proximal setting. Journal of Selected Topics in Signal Processing, 2016.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NIPS), 2012.  
Yann LeCun, Patrick Haffner, Léon Bottou, and Yoshua Bengio. Object recognition with gradient-based learning. In Shape, Contour and Grouping in Computer Vision. Springer, 1999.  
Yann LeCun, Joshua Bengio, and Geoffrey E. Hinton. Deep learning. Nature, 2015.  
Tim Meinhardt, Michael Möller, Caner Hazirbas, and Daniel Cremers. Learning proximal operators: Using denoising networks for regularizing inverse imaging problems. In International Conference on Computer Vision (ICCV), 2017.

Dmitry Molchanov, Arsenii Ashukha, and Dmitry P. Vetrov. Variational dropout sparsifies deep neural networks. In International Conference on Machine Learning (ICML), 2017.  
Vardan Papyan, Yaniv Romano, and Michael Elad. Convolutional neural networks analyzed via convolutional sparse coding. arXiv preprint arXiv:1607.08194, 2016.  
Vardan Papyan, Yaniv Romano, and Michael Elad. Convolutional neural networks analyzed via convolutional sparse coding. The Journal of Machine Learning Research, 2017.  
Vardan Papyan, Yaniv Romano, Jeremias Sulam, and Michael Elad. Theoretical foundations of deep learning via sparse representations: A multilayer sparse model and its connection to convolutional neural networks. IEEE Signal Processing Magazine, 2018.  
Peter Richtárik and Martin Takáč. On optimal probabilities in stochastic coordinate descent methods. Optimization Letters, 2016.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Stefano Soatto and Alessandro Chiuso. Visual representations: Defining properties and deep approximations. arXiv preprint arXiv:1411.7676, 2014.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 2014.  
Jeremias Sulam, Aviad Aberdam, and Michael Elad. On multi-layer basis pursuit, efficient algorithms and convolutional neural networks. arXiv preprint arXiv:1806.00701, 2018.  
Christoph Vogel and Thomas Pock. A primal dual network for low-level vision problems. In German Conference on Pattern Recognition. Springer, 2017.  
Stefan Wager, Sida Wang, and Percy S Liang. Dropout training as adaptive regularization. In Advances in Neural Information Processing Systems (NIPS), 2013.  
Sida Wang and Christopher Manning. Fast dropout training. In International Conference on Machine Learning (ICML), 2013.  
Lin Xiao and Tong Zhang. A proximal stochastic gradient method with progressive variance reduction. SIAM Journal on Optimization, 2014.  
Yan Yang, Jian Sun, Huibin Li, and Zongben Xu. Deep ADMM-Net for compressive sensing MRI. In Advances in Neural Information Processing Systems (NIPS), 2016.  
Jian Zhang and Bernard Ghanem. Ista-net: Iterative shrinkage-thresholding algorithm inspired deep network for image compressive sensing. In Computer Vision and Pattern Recognition (CVPR), 2018.  
Peilin Zhao and Tong Zhang. Stochastic optimization with importance sampling for regularized loss minimization. In International Conference on Machine Learning (ICML), 2015.
