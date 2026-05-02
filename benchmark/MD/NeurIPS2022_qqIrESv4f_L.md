# Signal Processing for Implicit Neural Representations

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Implicit Neural Representations (INR) encoding continuous multi-media data via multi-layer perceptrons has shown unbeatable promise in various computer vision tasks. Despite many successful applications, editing and processing an INR remains intractable as signals are represented by agnostic parameters of a neural network. Existing works manipulate such continuous representations via processing on their discretized instance, which breaks down the compactness and continuous nature of INR. In this work, we present a pilot study on the question: how to directly modify an INR without explicit decoding? We answer this question by proposing an implicit neural signal processing network, dubbed INSP-Net, via differential operators on INR. Our key insight is that spatial gradients of neural networks can be computed analytically and invariant to translation, while mathematically we show that any continuous convolution filter can be uniformly approximated by a linear combination of high-order differential operators. With these two knobs, we instantiate the INR signal operator as a composition of computational graphs corresponding to the high-order derivatives, where the weighting parameters can be either handcrafted or data-driven learned. Based on our proposed INSP-Net, we further build the first Convolutional Neural Network (CNN) that implicitly runs on INRs, named INSP-ConvNet. Our experiments validate the expressiveness of INSP-Net and INSP-ConvNet in fitting low-level image processing kernels (e.g. edge detection, blurring, deblurring, denoising, inpainting) as well as for high-level tasks on implicit fields such as image classification. We will release all codes.

# 1 Introduction

The idea that our visual world can be presented continuously has attracted increasing popularity in the field of implicit neural representations (INR). Also known as coordinate-based neural representations, INRs study to encode a coordinate-to-value mapping for continuous multi-media data. Instead of storing the discrete signal values in a grid of pixels or voxels, INRs represent discrete data as samples of a continuous manifold. Using multi-layer perceptrons, INRs bring practical benefits to various computer vision applications, such as image and video compression [1, 2, 3], 3D shape representation [4, 5, 6, 7, 8, 9, 10, 11], inverse problems [12, 2, 13, 14], and generative models [15, 16, 17, 18, 19, 20, 21, 22].

Despite their recent success, INRs are not yet amenable to flexible editing and processing as the standard images could do. The encoded coordinate-to-value mapping is too complex to comprehend and the parameter stored in multi-layer perceptrons (MLPs) remains less explored. One direction of existing approaches enables editing on INRs by training them with conditional input. For example, [23, 24, 25, 20, 21, 26] utilize conditional codes to indicate different characteristics of the scene including shape and color. Another main direction benefits from existing image editing techniques and operates on discretized instances of continuous INRs such as pixels [27, 28] or voxels [29]. However, such solutions break down the continuous characteristic of INR and are not due to the prerequisite of decoding and discretizing before editing and processing.

![](images/4e7197bf07939669a7644d93a41754759b8918432ac9718940fba77d52451f27.jpg)  
Figure 1: An illustration of implicit neural signal processing. Given an INR representing digital signals, our INSP-Net is capable of direct signal processing without needing to explicitly decode it. Our model first constructs derivative computation graphs of the original INR and then generates a linear combination of them into a new INR. It can be later decoded into discretized forms such as image pixels. The framework is capable of fitting low-level image processing kernels as well as performing high-level processing such as image classification.

In this paper, we conduct the first pilot study on the question: how to generally modify an INR without explicit decoding? The major challenge is that one cannot directly interpret what the parameters in an INR stand for, not to mention editing them correctly. Our key motivation is that spatial gradients can be served as a favorable tool to tackle this problem as they can be computed analytically, and possess desirable invariant properties. Theoretically, we prove that any continuous convolution filter can be uniformly approximated by a linear combination of high-order differential operators. Based on the above two rationales, we propose an Implicit Neural Signal Processing Network, dubbed INSP-Net, which processes INR utilizing high-order differential operators. The proposed INSP-Net is composed of an inception fusion block connecting computational graphs corresponding to derivatives of INRs. The weights in the branchy part are loaded from the INR being processing, while the weights in the fusion block are parameters of the represented operator, which can be either hand-crafted or learned by the data-driven algorithm. Even though we are not able to surgery on neural network parameters, we can implicitly process them by retrofitting their architecture and reorganizing the spatial gradients.

We further extend our framework to build the first Convolutional Neural Network (CNN) operating directly on INRs, dubbed INSP-ConvNet. Each layer of INSP-ConvNet is constructed by linearly combining the derivative computational graphs of the former layers. Nonlinear activation and normalization are naturally supported as they are element-wise functions. Data augmentation can be also implemented by augmenting the input coordinates of INRs. Under this pipeline (shown in Fig. 1), we demonstrate the expressiveness of our INSP-Net framework in fitting low-level image processing kernels including edge detection, blurring, deblurring, denoising, and image inpainting. We also successfully apply our INSP-ConvNet to high-level tasks on implicit fields such as classification.

Our main contributions can be summarized as follows:

- We propose a novel signal processing framework, dubbed INSP-Net, that operates on INRs analytically and continuously by closed-form high-order differential operators. Repeatedly cascading the computational paradigm of INSP-Net, we also build a convolutional network, called INSP-ConvNet, which directly runs on implicit fields for high-level tasks.  
- We illustrate the advantage of adopting differential operators by revealing their inherent group invariance. Furthermore, we rigorously prove that the convolution operator in the continuous regime can be uniformly approximated by a linear combination of the gradients.  
- Extensive experiments demonstrate the effectiveness of our approach in both low-level processing (e.g. edge detection, blurring, deblurring, denoising, image inpainting) and high-level processing such as image classification.

# 2 Preliminaries: Implicit Neural Representation

Implicit Neural Representation (INR) parameterizes continuous multi-media signals or vector fields with neural networks. Formally, we consider an INR as a continuous function  $\Phi : \mathbb{R}^m \to \mathbb{R}$  that maps low-dimension spatial/temporal coordinates to the value space<sup>1</sup>. For example, to represent

![](images/70d2bf30eee673092395653b1dd34ccfcc350d5147417d9f06cab51fc54cca06.jpg)  
Figure 2: The left image provides an overview of our INSP-Net framework. Each layer combines the high-order derivative computational graphs of the original INR network. The right image illustrates the weight sharing scheme in calculating the derivative sub-networks.

![](images/ed67204b09d29465707eb48576e6f3e1c2c726a30f9ba718b2be4940beecd633.jpg)

2D image signals, the domain of  $\Phi$  is  $(x,y)$  spatial coordinates, and the range of  $\Phi$  are the pixel intensities. The typical use of INR is to solve a feasibility problem where  $\Phi$  is sought to to satisfy a set of  $M$  constraints  $\{\mathcal{C}(\Phi ,a_m|\Omega_m)\}_{m = 1}^M$ , where  $\mathcal{C}$  is a functional that relates function  $\Phi$  to some observable quantities  $a_{m}$  evaluating over a measurable domain  $\Omega_{m}\subseteq \mathbb{R}^{m}$ . This problem can be cast into an optimization problem that minimizes deviations from each of the constraints:

$$
\Phi^ {*} = \underset {\Phi} {\arg \max } \sum_ {m = 1} ^ {M} \| \mathcal {C} (\Phi , a _ {m} | \Omega_ {m}) \| _ {2}. \tag {1}
$$

For instance, we can let  $\mathcal{C} = \Phi(\boldsymbol{x}_m) - a_m$  with  $\Omega_m = \{\boldsymbol{x}_m\}$ , then our objective is reduced to a point-to-point supervision which memorizes a signal into  $\Phi$  [30]. When functional  $\mathcal{C}$  is a combination of differential operators taking values in a point set, i.e.,  $\mathcal{C}(a(\boldsymbol{x}), \Phi(\boldsymbol{x}), \nabla \Phi(\boldsymbol{x}), \dots), \forall \boldsymbol{x} \in \Omega_m$ , Eq. 1 is objective to solving a bunch of differential equations [27, 7, 31]. Note that in this paper, without particular specification, the gradients are all computed with respect to the input coordinate  $\boldsymbol{x}$ .  $\mathcal{C}$  can also form an integral equation system over some intervals  $\Omega_m$  [12]. In practice of computer vision, we reconstruct a signal by capturing sparse observations  $\mathcal{D} = \{(\Omega_m, a_m)\}_{m=1}^M$  from unknown function  $\Phi$ , and dynamically sampling a mini-batch from  $\mathcal{D}$  to minimize Eq. 1 to obtain a feasible  $\Phi$ .

A handy parameterization of function  $\Phi$  is a fully-connected neural network, which enables solving Eq. 1 via gradient descent through a differentiable  $\mathcal{C}$ . Common INR networks consist of pure Multi-Layer Perceptrons (MLP) with periodic activation functions. Fourier Feature Mapping (FFM) [30] places a sinusoidal transformation before the MLP, while Sinusoidal Representation Network (SIREN) [27] replaces every piece-wise linear activation with a sinusoidal function. Below we give a unified formulation of INR networks:

$$
\Phi (\boldsymbol {x}) = \boldsymbol {W} _ {n} \left(\phi_ {n - 1} \circ \phi_ {n - 2} \circ \dots \circ \phi_ {1}\right) (\boldsymbol {x}), \quad \phi_ {i} (\boldsymbol {x}) = \sigma_ {i} \left(\boldsymbol {W} _ {i} \boldsymbol {x} + \boldsymbol {b} _ {i}\right), \tag {2}
$$

where  $\pmb{W}_i\in \mathbb{R}^{d_{i - 1}\times d_i}$ ,  $\pmb{b}_i\in \mathbb{R}^{d_i}$  are the weight matrix and bias of the  $i$ -th layer, respectively,  $n$  is the number of layers, and  $\sigma_{i}(\cdot)$  is an element-wise nonlinear activation function. For FFM architecture,  $\sigma_{i} = \sin (\cdot)$  when  $i = 1$  denotes the positional encoding layer [12, 32] and otherwise  $\sigma_{i} = \mathrm{ReLU}(\cdot)$ . For SIREN,  $\sigma_{i} = \sin (\cdot)$  for every layer  $i = 1,\dots ,n - 1$ .

# 3 Implicit Representation Processing via Differential Operators

Digital Signal Processing (DSP) techniques have been widely applied in computer vision tasks, such as image restoration [33], signal enhancement [34] and geometric processing [35]. Even modern deep learning models are consisting of the most basic signal processing operators. Suppose we already acquire an Implicit Neural Representation (INR)  $\Phi : \mathbb{R}^m \to \mathbb{R}$ , now we are interested in whether we can run a signal processing program on the implicitly represented signals. One straightforward solution is to rasterize the implicit field with a 2D/3D lattice and run a typical kernel on the pixel/voxel grids. However, this decoding strategy produces a finite resolution and discretizes signals, which is memory inefficient and unfriendly to modeling fine details. In this section, we introduce a computation paradigm that can process an INR analytically with spatial/temporal derivatives. We show that our proposed method serves as a universal operator that can represent any continuous convolutional kernels.

# 3.1 Computational Paradigm

It has not escaped our notice that spatial/temporal gradients on INRs  $\nabla^k\Phi$  can be computed analytically due to the differentiable characteristics of neural networks. Inspired by this, we propose an Implicit Neural Signal Processing (INsP) framework that composes a class of closed-form operators for INRs using functional combinations of high-order derivatives.

We denote our proposed signal processing operator by  $\mathcal{A}$  built upon high-order derivatives. Given an acquired INR  $\Phi$ , we denote the resultant INR processed by operator  $\mathcal{A}$  as  $\Psi = \mathcal{A}\Phi : \mathbb{R}^m \to \mathbb{R}$ . To evaluate point  $\pmb{x} \in \mathbb{R}^m$  of processed INR, we propose the following computational paradigm:

$$
\Psi (\boldsymbol {x}) := \mathcal {A} \Phi (\boldsymbol {x}) = \Pi (\Phi (\boldsymbol {x}), \nabla \Phi (\boldsymbol {x}), \nabla^ {2} \Phi (\boldsymbol {x}), \dots , \nabla^ {k} \Phi (\boldsymbol {x}), \dots), \tag {3}
$$

where  $\Pi : \mathbb{R}^M \to \mathbb{R}$  can be arbitrary continuous functions, which can be either handcrafted or learned from data. To learn an operator  $\mathcal{A}$  from data, we represent  $\Pi$  by Multi-Layer Perceptrons (MLP) with parameters  $\theta$ . Here we can slightly abuse the notation of  $\nabla^k$  to be a flattened vector of high-order derivatives without multiplicity since differential operators defined over continuous functions form a commutative ring. The input dimension of  $\Pi$  depends on the highest order of used derivatives. Suppose we compute derivatives up to  $K$ -th order, then  $M = \sum_{k=0}^{K} \binom{k+m-1}{k} = (K+1) \binom{K+M}{K+1}/m$ , where  $\binom{k+m-1}{k}$  is the number of distinctive  $k$ -th order differential operators<sup>2</sup>. As we will show in Sec. 3.2,  $\Pi$  can construct both shift-invariant and rotation-invariant operators, which introduces favorable inductive bias to images and 3D geometry processing. More importantly, we prove that Eq. 3 is also a universal approximator of arbitrary convolutional operators.

We note that  $\Psi(\pmb{x})$  as a whole can also be regarded as a neural network. Recall the architecture of  $\Phi(\pmb{x})$  in Eq. 2, its  $k$ -th order derivative is another computational graph parameterized by  $\pmb{W}_i$  and  $\pmb{b}_i$  that maps  $\pmb{x}$  to  $\nabla^k\Phi(\pmb{x})$ . For example, the first-order gradient will have the following form:

$$
\nabla \Phi (\boldsymbol {x}) = \hat {\phi} _ {n - 1} \circ \left(\phi_ {n - 2} \circ \dots \circ \phi_ {1}\right) (\boldsymbol {x}) \odot \dots \odot \hat {\phi} _ {2} \circ \phi_ {1} (\boldsymbol {x}) \odot \boldsymbol {W} _ {1} \tag {4}
$$

where  $\hat{\phi}_i(\pmb{y}) = \pmb{W}_i^\top \sigma_{i-1}'(\pmb{W}_{i-1}\pmb{y} + \pmb{b}_{k-1})$ , and  $\sigma_i'(\cdot)$  is the first-order derivative of  $\sigma_i(\cdot)$ . Since  $\hat{\phi}_i$  shares the weights with  $\hat{\phi}_i$ ,  $\nabla \Phi$  is represented by a closed-form computational network re-using the weights from  $\Phi$ , which we refer to as the first-order derivative network. The higher-order derivatives should induce the derivative network of similar forms. Therefore, the processed INR  $\Psi$  will have an Inception-like architecture, namely, a multi-branch structure connecting the original INR network and weight-sharing derivative subnetworks followed by a fusion layer  $\Pi$ . We call the entire model ( $\Psi = \mathcal{A}\Phi$  or Eq. 3) an Implicit Neural Signal Processing Network or an INSP-Net. Note that the only parameters of INSP-Net  $\theta$  are located at the last fusion layer, and can be trained in an end-to-end manner.

We illustrate an INSP-Net in Fig. 2 where the color indicates the weight-sharing scheme. In practice, we employ auto-differentiation in PyTorch [36] to automatically create such derivatives networks and reassemble them parallelly to constitute the architecture of an INSP-Net. When inputting an INR, we load the weights of the INR to our model following the weight-sharing scheme, and then we obtain an INSP-Net, which implicitly and continuously represents the processed INR  $\Psi(x)$ . To effectively express high-order derivatives, we choose SIREN as the base model [27].

# 3.2 Theoretical Analysis

In this section, we provide a theoretical justification for the design of our INSP-Net. We will focus on discussing the latent invariance property and the expressive power of INSP-Net.

Translation and Rotation Invariance. Group invariance has testified as a favorable inductive bias for image [37], video [38], and geometry processing [39]. It has also been well-known that group invariance is an intrinsic property of Partial Differential Equations (PDEs) [40, 41]. Since our INSP-Net is built using differential operators, we are motivated to reveal its hidden invariance property to demonstrate its advantage in processing visual signals.

In this section, we only consider two transformation groups: translation group  $\mathbb{T}(m)$  and the special orthogonal group  $\mathbb{SO}(m)$  (a.k.a. rotation group). Elements  $T_{\pmb{v}} \in \mathbb{T}(m)$  in translation group shifts the function  $\Phi$  by some offset  $\pmb{v} \in \mathbb{R}^m$ . The shifted function can be denoted as  $\Phi \circ T_{\pmb{v}}(\pmb{x}) = \Phi(\pmb{x} + \pmb{v})$ .

Similarly, elements in rotation group perform a coordinate transformation on function  $\Phi$  by a rotation matrix  $\pmb{R} \in \mathbb{S}\mathbb{O}(m)$ . The transformed function can be written as  $\Phi \circ \pmb{R}(\pmb{x}) = \Phi(\pmb{R}\pmb{x})$ . Group invariance means deforming the input space of a function first and then processing it via an operator is equivalent to directly applying the transformation to the processed function. For a more rigorous argument,  $\mathcal{A}$  is said to be translation-invariant if  $\forall T_{\pmb{v}} \in \mathcal{T}(m)$ ,  $\Psi(\pmb{x} + \pmb{v}) = \mathcal{A}[\Phi \circ T_{\pmb{v}}](\pmb{x})$ . Likewise,  $\mathcal{A}$  is rotation-invariant if  $\forall \pmb{R} \in \mathbb{S}\mathbb{O}(m)$  we have  $\Psi(\pmb{R}\pmb{x}) = \mathcal{A}[\Phi \circ \pmb{R}](\pmb{x})$ . Below we provide Theorem 1 to characterize the invariance property for our model.

Theorem 1. Given function  $\Pi : \mathbb{R}^M \to \mathbb{R}$ , the composed operator  $\mathcal{A}$  (Eq. 3) can satisfy:

1. shift invariance for every  $\Pi$ .  
2. rotation invariance if  $\Pi$  has the form:  $\Pi (\pmb {y}) = f(\| \pmb {y}\| _2)$  for some  $f:\mathbb{R}\to \mathbb{R}$

We prove Theorem 1 in Appendix A. Our Theorem 1 implies that operator  $\mathcal{A}$  is inherently shift-invariant. This is due to the shift-invariant intrinsic of differential operators as we show in the proof. Rotation invariance is not guaranteed in general. However, if one carefully designs  $\Pi$ , it can also be achieved via our framework. Moreover, we also suggest a feasible solution to constructing a rotation-invariant operator  $\mathcal{A}$  in Theorem 1. In our construction,  $\Pi$  first isotropically pools over the squares of all directional derivatives, and then maps the summarized information through another function  $f$ , which can be either handcrafted or a trainable neural network. We refer interested readers [40] for more group invariance in differential forms.

Universal Approximation. Convolution, formally known as the linear shift-invariant operator, has served as one of the most prevalent signal processing tools in the vision domain. Given two (real-valued) signals  $f$  and  $g$ , we denote their convolution as  $g \star f = f \star g$ . In this section, we examine the expressiveness of our INSP-Net (Eq. 3) by showing it can represent any convolutional filter. To draw this conclusion, we first present an informal version of our main results as follows:

Theorem 2. (Informal statement) For every real-valued function  $g: \mathbb{R} \to \mathbb{R}$ , there exists a polynomial  $p(x) = a_0 + a_1x + \dots$  where  $a_0, a_1, \dots \in \mathbb{R}$ , such that  $p(\nabla) f$  can uniformly approximate  $g \star f$  by arbitrary precision for all real-valued signals  $f$ .

The formal statement and proof can be found in Appendix B. Theorem 2 involves the notion of polynomials in differential operators. We define  $p(\nabla) = a_0 + a_1\nabla + a_2\nabla^2 + \dots$ , where the map that takes polynomial  $p(x)$  to  $p(\nabla)$  is a ring homomorphism from polynomial ring  $\mathbb{R}[x]$  to the ring of endomorphism of a function space. Theorem 2 can be regarded as an application of the Stone-Weierstrass approximation theorem on the Fourier domain. However, we note that functions obtained by the Fourier transform are generally complex functions. The prominence of our results is that we can constrain the range of the polynomial coefficients into the real domain, which makes it implementable via a common deep learning infrastructure. The implication of Theorem 2 is that the mapping between convolution and derivative is as simple as a linear transformation. Recent works [42, 43, 44] show similar results that derivatives can be computed via a linear combination of discrete convolution. Theorem 2 can be served as their converse argument but in the continuous regime. In our proof,  $k$ -th order derivatives correspond to  $k$ -th order monomial function in spectral space. This indicates that differential operators form a (wavelet) basis of the linear shift-invariant operator space. In other words, any convolution can be written as a linear combination of differential operators, i.e.,  $p(\nabla)$ . Since a linear function is not difficult to be approximated by a neural network  $\Pi_{\theta}$ , we can derive the next straightforward result Corollary 3.

Corollary 3. For every real-valued function  $g$ , there exists a neural network  $\Pi_{\theta}$  such that  $\Psi = \mathcal{A}\Phi$  (Eq. 3) can uniformly approximate  $g \star \Phi$  by arbitrary precision for every real-valued signals  $\Phi$ .

As we discussed in Theorem 1,  $\mathcal{A}$  are constantly shift-invariant. This means when approximating a convolutional kernel, the trajectory of  $\mathcal{A}$  is restricted into the shift-invariant space. Moreover, we emphasize that INSP-Net is far more expressive than convolutional kernels since  $\Pi_{\theta}$  can also fit any nonlinear continuous functions due to the universal approximation theorem [45, 46].

In fact, to simulate exact convolution, i.e.,  $\mathcal{A} = p(\nabla)$ , our Theorem 2 suggests simplify  $\Pi_{\theta}$  to a linear mapping. Then our former computational paradigm Eq. 3 is cast to:

$$
\Psi (\boldsymbol {x}) := p (\nabla) \Phi (\boldsymbol {x}) = \theta_ {0} \Phi (\boldsymbol {x}) + \boldsymbol {\theta} _ {1} ^ {\top} \nabla \Phi (\boldsymbol {x}) + \boldsymbol {\theta} _ {2} ^ {\top} \nabla^ {2} \Phi (\boldsymbol {x}) + \dots + \boldsymbol {\theta} _ {k} ^ {\top} \nabla^ {k} \Phi (\boldsymbol {x}) + \dots , \tag {5}
$$

where  $\theta_{k}\in \mathbb{R}^{\binom{k + m - 1}{k}}$  are parameters of the operator  $p(\nabla)$ . One plausible implementation of this model is to employ a one-layer MLP to represent  $\Pi_{\theta}$ . When  $\mathcal{A} = p(\nabla)$ , it preserves both linearity and shift-invariance when evolving during the training.

# 3.3 Building CNNs for Implicit Neural Representations

Convolutional Neural Networks (CNN) are capable of extracting informative semantics by only piling up basic signal processing operators. This motivates us to build CNNs based on INSP-Net that can directly run on INRs for high-level downstream tasks. We name this class of CNNs stacking multi-layer INSP-Net as INSP-ConvNet. Previous works [47, 48] extracting semantic features from INR either lack local information by point-wisely mapping INR's intermediate representation to a semantic space or explicitly rasterize INR into regular grids. To the best of our knowledge, it is the first time that one can run a CNN directly on an implicit representation thanks to our INSP-Net. The overall architecture of INSP-ConvNet can be formulated as:

$$
\operatorname {C o n v N e t} [ \Phi ] (\boldsymbol {x}) = \mathcal {A} ^ {(L)} \cdot \sigma \circ \mathcal {A} ^ {(L - 1)} \cdot \sigma \circ \dots \circ \mathcal {A} ^ {(2)} \cdot \sigma \circ \mathcal {A} ^ {(1)} \cdot \Phi (\boldsymbol {x}), \tag {6}
$$

where  $\sigma$  is an element-wise non-linear activation,  $L$  is the number of INSP-Net layers, and  $\Phi$  is an input INR. We use the symbol  $\cdot$  to denote operator functioning, and  $\circ$  to denote function composition. Below we elaborate on each main ingredient:

Convolutional Layer. Each  $\mathcal{A}^{(l)}$  represents an implicit convolution layer. We follow the closed-form solution in Eq. 5 to parameterize  $\mathcal{A}^{(l)}$  with  $\theta^{(l)}$ . We point out that  $\mathrm{ConvNet}[\Phi]$  also corresponds to a computational graph, which can continuously map coordinates to the output features. To construct this computational graph, we recursively call for gradient networks of the previous layer until the first layer. For example,  $\mathcal{A}^{(l)}$  will request the gradient network of  $\mathcal{A}^{(l-1)} \cdot \sigma \circ \dots \circ \mathcal{A}^{(1)} \cdot \Phi$ , and then  $\mathcal{A}^{(l-1)}$  will request the gradient network of the rest part. This procedure will proceed until the first layer, which directly returns the derivative network of  $\Phi$ . Kernels in CNNs typically perform multi-channel convolution. However, this is not memory friendly to gradient computing in our framework. To this end, we run channel-wise convolution first and then employ a linear layer to mix channels [49].

Nonlinear Activation and Normalization. Nonlinear activation and normalization are naturally element-wise functions. They are applied to the implicit signal after being processed by an INSP-Net. In our implementation, they participate the computational graph construction process described above.

**Training Recipe.** Given a dataset  $\mathcal{D} = \{(\Phi_i, y_i)\}$  with a set of pre-trained INRs  $\Phi_i$  and their corresponding semantic labels  $y_i$ , our goal is to learn a ConvNet[·] that can process each example. In contrast to standard ConvNets that are designed for grid-based images, the computational graph of INSP-ConvNet contains parameters of both the input INR  $\Phi_i$  and learnable kernels  $\mathcal{A}^{(l)}$ . During the training stage, we randomly sample a mini-batch  $(\Phi_i, y_i)$  from  $\mathcal{D}$  to optimize INSP-ConvNet. The corresponding loss will be evaluated according to the network output, and then back-propagate the calculated gradients to the learnable parameters in  $\mathcal{A}^{(l)}$ , using the stochastic gradient descent optimization. Along the whole process, the parameters of  $\Phi_i$  is fixed and only the parameters in  $\mathcal{A}^{(l)}$  is optimized. Standard data augmentations are included by default, including rotation, zoom in/out, etc. In practice, we implement these augmentations by using affine transformation on the coordinates of INRs.

# 4 Related Works

# 4.1 Implicit Neural Representation

Implicit Neural Representation (INR) represents signals by continuous functions parameterized by multi-layer perceptrons (MLPs) [27, 30], which is different from traditional discrete representations (e.g., pixel, mesh). Compared with other representations, the continuous implicit representations are capable of representing signals theoretically at infinite resolution and have become prevailing to be applied upon image fitting [27], image compression [1] and video compressing [3]. In addition, INR has been applied to more efficient and effective shape representation [4, 5, 6, 7, 8, 9, 10, 11], texture mapping [50, 51], inverse problems [12, 2, 13, 14] and generative models [15, 16, 17, 18, 19,

20, 21, 22]. Nowadays, editing and manipulating multi-media objects gains increasing interest and demand. Thus, signal processing on implicit neural representation is essentially an important task worth investigating.

# 4.2 Editable Implicit Fields

Editing implicit fields has recently attracted much research interest. Several methods have been proposed to allow editing the reconstructed 3D scenes by rearranging the objects or manipulating the shape and appearance. One line of work alters the structure and color of objects by conditioning latent codes for different characteristics of the scene [25, 20, 21, 26]. Another direction involves discretizing the continuous fields. By converting the implicit fields into pixels [27, 28] or voxels [29], traditional image and voxel editing techniques can be applied effortlessly. Recently, NFGP [52] proposes to use neural fields for geometry processing. These approaches, however, are not capable of directly performing signal processing on continuous INRs. Our INSP-Net makes smart use of closed-form differential operators and thus does not require explicit decoding.

# 4.3 PDE based Image Processing

Partial differential equations (PDEs) have been successfully applied to many tasks in image processing and computer vision, such as image enhancement [53, 54, 55], segmentation [56, 41], image registration [57], saliency detection [58] and optical flow computation [59]. Early traditional PDEs are written directly based on mathematical and physical understanding of the PDEs (e.g., anisotropic diffusion [53], shock filter [54] and curve evolution based equations [60, 61, 62]). Variational design methods [55, 63, 62] start from an energy function describing the desired properties of output images and compute the Euler-Lagrange equation to derive the evolution equations. Learning-based attempts [41, 58] build PDEs from image pairs based on the assumption (without proof) that PDEs could be written as linear combinations of fundamental differential invariants. Although it might be feasible to let INRs solve this bunch of signal processing PDEs, one needs to per-case re-fit an INR with an additional temporal axis, which is severely sampling inefficient. The multi-layer structure appearing in INSP-Net can be viewed as an unfolding algorithm [64, 65] that discretizes the evolution of time-variant PDEs [66].

# 5 Experiments

In this section, we evaluate the proposed INsP framework on several challenging tasks, using different combinations of  $\Pi$ . First, we build low-level image processing filters using either hand-crafted or learnable  $\Pi$ . Then, we construct convolutional neural networks with our INsP-ConvNet framework and validate its performance on image classification. More results and implementation details are provided in the Appendix.

![](images/c86e3cdde09652d27f33e522fd7e9171357063c97240b0597a4acb709af5033d.jpg)  
Figure 3: Edge detection. We fit the natural images with SIREN and use our INSP-Net to process implicitly into a new INR that can be decoded into edge maps.

For low-level image processing, we operate on natural images from Set5 dataset [67], Set14 dataset [68], and DIV-2k dataset [69]. Originally designed for super-resolution, the images are diverse in style and content. In our experiments, we construct SIREN [27] on each image. For efficiency, we resized the images to  $256 \times 256$ .

# 5.1 Handcrafted II for Edge Detection

We demonstrate the proposed INSP framework is capable of expressing low-level image processing filters such as for edge detection. Since the edges correspond to gradients in the images, using

![](images/b4b0e51a02da2f67f2938e37549b47663709f094ca39c0c719c2811fde7ad856.jpg)  
Figure 4: Image denoising. We fit the noisy images with SIREN and train our INSP-Net to process implicitly into a new INR that can be decoded into natural clear images.

![](images/f6163d5668d284f1c61391a0701dc50dc74274eae6dcb1704c8c0a933265c8c6.jpg)  
Figure 5: Image deblurring. We fit the blurred images with SIREN and train our INSP-Net to process implicitly into a new INR that can be decoded into clear natural images.

gradients of INRs to obtain edges is straightforward.  $\theta_{1}$  is set to 1 while other coefficients are set to 0. In this way, we implicitly obtain a new INR which can be decoded to obtain the edges. We provide visual comparisons against Sobel filter [70], Canny detector [71] and Prewitt operator [72] in Fig. 3.

# 5.2 Learnable II for Image Blurring, Denoising, Deblurring, and Inpainting

Since our method operates directly on INRs, we firstly fit the images with INRs and then feed the INRs into our framework. The final output is another INR which can be decoded into desired images. The training set of our method consists of 90 examples of INRs, where each INR is built on SIREN [27] architectures.

Image Denoising For classical image denoising filters, we compare against the median filter and mean filter. We use DnCNN [73] as a baseline learning-based method. DnCNN is a pioneer work of using residual CNN for denoising. The input noisy images are synthesized using additive gaussian noise. Visual results are provided in Fig. 4.

Image Blurring Image blurring is a low-pass filtering operation. We provide a visual comparison against classical filters including  $3 \times 3$  box filter and  $3 \times 3$  gaussian filter. The target images used for training our INSP-Net are the results of the Gaussian filter. Visual results are provided in Fig. 6.

Image Deblurring We compare the proposed method with both traditional algorithms (e.g., wiener filter [74]) and learning-based algorithms(e.g., MPRNet [75]). We first synthesize a blurry image using Gaussian filters and then apply each of them respectively. As shown in Fig. 5, Wiener Filter produce severe artifacts and MPRNet successfully reconstructs clear textures. INSP-Net is capable of generating competitive results against MPRNet and outperforms the Wiener Filter.

Image Inpainting We conduct two kinds of experiments in image inpainting, to inpaint  $30\%$  random masked pixels or to remove the texts ("INSP-Net"). Comparison methods include mean filter, median filter, and LaMa [76]. LaMa is a learning-based method using Fourier convolution for inpainting. As shown in Fig. 7, mean filter and median filter partially restore the masked pixels, but severely hurt the visual quality of the rest parts. Also, they can not handle the text region. LaMa successfully removes the text and inpaint the masked pixels. Our proposed method largely outperforms the filter-based algorithms and performs as well as the LaMa.

![](images/b17019ebf352ba0994a24a97d08d62c0ebfa709a5a93dc2868410df1ca7ea293.jpg)  
Figure 6: Image blurring. We fit the natural images with SIREN and train our INSP-Net to process implicitly into a new INR that can be decoded into blurred images.

![](images/8ef22eb6a60589c08351f829fd3fbbbe0e42f59e020e8a2402ffad0456053c1f.jpg)

![](images/629ba3a18dd75f4fb8830381102528efeecb2c2d4b276bfb85944fa975135cdc.jpg)

![](images/c0f1893b861ff1f9204a7160d1af7151037e1bbb93095efb9fb4ae0006180f91.jpg)

![](images/69e20990ff002cfa89aba6b4ff1638fd43f7966e82f0bdc7cd578d304fd4d504.jpg)  
Figure 7: Image inpainting. We fit the input images with SIREN and train our INSP-Net to process implicitly into a new INR that can be decoded into natural images. Note that LaMa requires explicit masks to select the regions for inpainting and the masks are roughly provided.

# 5.3 Convolutional Neural Networks for Image Classification

We demonstrate that the proposed INSP framework is not only capable to express low-level image processing filters, but also supports high-level tasks such as image classification. To achieve this goal, we construct a 2-layer INSP-ConvNet. The INSP-ConvNet consists of 2 INSP-Net layers. Each of them decomposes the INR via the differential operator and combines them with learnable II. Since each INSP-Net layer combines the derivative computational graphs of the former layers and approximates a convolution filter, we build another 2-layer depthwise ConvNets running on pixels as the baseline, for a fair comparison.

We evaluate the proposed INSP-ConvNet on MNIST  $(28 \times 28$  resolution) and CIFAR-10  $(32 \times 32$  resolution) datasets, respectively. For each dataset, we will firstly fit each image into an implicit representation using SIREN [27]. Both experiments take 1000 epochs to optimize, using AdamW optimizer [77] and a learning rate of  $1e - 4$ . Results are shown in Tab. 1.

<table><tr><td>Accuracy</td><td>Depthwise CNN</td><td>INSP-ConvNet</td></tr><tr><td>MNIST</td><td>87.6%</td><td>88.1%</td></tr><tr><td>CIFAR-10</td><td>59.5%</td><td>62.5%</td></tr></table>

Table 1: Quantitative Results of Image Classification.

# 6 Conclusion

We present INSP-Net framework, an implicit neural signal processing network that is capable of directly modifying an INR without explicit decoding. By incorporating differential operators on INR, we can instantiate the INR signal operator as a composition of computational graphs approximating any continuous convolution filter. Furthermore, we make the first effort to build a convolutional neural network that implicitly runs on INRs. While all other methods run on discrete grids, our experiment demonstrates our INSP-Net can achieve competitive results with entirely implicit operations. Future works may involve more complex parameterizations which potentially improve the performance.

# References

[1] Emilien Dupont, Adam Golinski, Milad Alizadeh, Yee Whye Teh, and Arnaud Doucet. Coin: Compression with implicit neural representations. arXiv preprint arXiv:2103.03123, 2021.  
[2] Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser Nam Lim, and Abhinav Shrivastava. Nerv: Neural representations for videos. Advances in Neural Information Processing Systems, 34, 2021.  
[3] Yunfan Zhang, Ties van Rozendaal, Johann Brehmer, Markus Nagel, and Taco Cohen. Implicit neural video compression. arXiv preprint arXiv:2112.11312, 2021.  
[4] Kyle Genova, Forrester Cole, Daniel Vlasic, Aaron Sarna, William T Freeman, and Thomas Funkhouser. Learning shape templates with structured implicit functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7154-7164, 2019.  
[5] Matan Atzmon, Niv Haim, Lior Yariv, Ofer Israelov, Haggai Maron, and Yaron Lipman. Controlling neural level sets. Advances in Neural Information Processing Systems, 32, 2019.  
[6] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5939-5948, 2019.  
[7] Amos Gropp, Lior Yariv, Niv Haim, Matan Atzmon, and Yaron Lipman. Implicit geometric regularization for learning shapes. arXiv preprint arXiv:2002.10099, 2020.  
[8] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4460-4470, 2019.  
[9] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Occupancy flow: 4d reconstruction by learning particle dynamics. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5379-5389, 2019.  
[10] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 165-174, 2019.  
[11] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In European Conference on Computer Vision, pages 523-540. Springer, 2020.  
[12] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European conference on computer vision, pages 405-421. Springer, 2020.  
[13] Vincent Sitzmann, Semon Rezchikov, William T Freeman, Joshua B Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. arXiv preprint arXiv:2106.02634, 2021.  
[14] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul Srinivasan, and Jonathan T Barron. Nerf in the dark: High dynamic range view synthesis from noisy raw images. arXiv preprint arXiv:2111.13679, 2021.  
[15] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5799-5809, 2021.  
[16] Terrance DeVries, Miguel Angel Bautista, Nitish Srivastava, Graham W Taylor, and Joshua M Susskind. Unconstrained scene generation with locally conditioned radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14304-14313, 2021.

[17] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. Stylenerf: A style-based 3d-aware generator for high-resolution image synthesis. arXiv preprint arXiv:2110.08985, 2021.  
[18] Zekun Hao, Arun Mallya, Serge Belongie, and Ming-Yu Liu. Gancraft: Unsupervised 3d neural rendering of apache worlds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14072-14082, 2021.  
[19] Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, and Jingyi Yu. Gnerf: Gan-based neural radiance field without posed camera. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6351-6361, 2021.  
[20] Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11453-11464, 2021.  
[21] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems, 33:20154-20166, 2020.  
[22] Peng Zhou, Lingxi Xie, Bingbing Ni, and Qi Tian. Cips-3d: A 3d-aware generator of gans based on conditionally-independent pixel synthesis. arXiv preprint arXiv:2110.09788, 2021.  
[23] Steven Liu, Xiuming Zhang, Zhoutong Zhang, Richard Zhang, Jun-Yan Zhu, and Bryan Russell. Editing conditional radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5773-5783, 2021.  
[24] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. arXiv preprint arXiv:2112.05139, 2021.  
[25] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 165-174, 2019.  
[26] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5932-5941, 2019.  
[27] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in Neural Information Processing Systems, 33:7462-7473, 2020.  
[28] James W. Hennessey, Wilmot Li, Bryan Russell, Eli Shechtman, and Niloy J. Mitra. Transferring image-based edits for multi-channel compositing. ACM Transactions on Graphics, 36(6), 2017.  
[29] Jerry Liu, Fisher Yu, and Thomas Funkhouser. Interactive 3d modeling with a generative adversarial network. In 2017 International Conference on 3D Vision (3DV), pages 126-134. IEEE, 2017.  
[30] Matthew Tancik, Pratul P Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. arXiv preprint arXiv:2006.10739, 2020.  
[31] Jiequn Han, Arnulf Jentzen, and E Weinan. Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34):8505-8510, 2018.  
[32] Ellen D Zhong, Tristan Bepler, Bonnie Berger, and Joseph H Davis. Cryodrgn: reconstruction of heterogeneous cryo-em structures using neural networks. Nature Methods, 18(2):176-185, 2021.

[33] Dejia Xu, Yihao Chu, and Qingyan Sun. Moiré pattern removal via attentive fractal network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 472-473, 2020.  
[34] Yaqian Xu, Wenqing Zheng, Jingchen Qi, and Qi Li. Blind image blur assessment based on markov-constrained fcm and blur entropy. In 2019 IEEE international conference on image processing (icip), pages 4519-4523. IEEE, 2019.  
[35] Peng-Shuai Wang, Xiao-Ming Fu, Yang Liu, Xin Tong, Shi-Lin Liu, and Baining Guo. Rolling guidance normal filter for geometric processing. ACM Transactions on Graphics (TOG), 34(6):1-9, 2015.  
[36] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.  
[37] Jan J Koenderink and Andrea J van Doorn. Image processing done right. In European Conference on Computer Vision, pages 158-172. Springer, 2002.  
[38] Hae-Kwang Kim and Jong-Deuk Kim. Region-based shape descriptor invariant to rotation, scale and translation. Signal Processing: Image Communication, 16(1-2):87-93, 2000.  
[39] Niloy J Mitra, Mark Pauly, Michael Wand, and Duygu Ceylan. Symmetry in 3d geometry: Extraction and applications. In Computer Graphics Forum, volume 32, pages 1-23. Wiley Online Library, 2013.  
[40] Peter J Olver. Applications of Lie groups to differential equations, volume 107. Springer Science & Business Media, 2000.  
[41] Risheng Liu, Zhouchen Lin, Wei Zhang, and Zhixun Su. Learning pdes for image restoration via optimal control. In European Conference on Computer Vision, pages 115-128. Springer, 2010.  
[42] Bin Dong, Qingtang Jiang, and Zuowei Shen. Image restoration: Wavelet frame shrinkage, nonlinear evolution pdes, and beyond. Multiscale Modeling & Simulation, 15(1):606-660, 2017.  
[43] Zichao Long, Yiping Lu, Xianzhong Ma, and Bin Dong. Pde-net: Learning pdes from data. In International Conference on Machine Learning, pages 3208-3216. PMLR, 2018.  
[44] Zichao Long, Yiping Lu, and Bin Dong. Pde-net 2.0: Learning pdes from data with a numeric-symbolic hybrid deep network. Journal of Computational Physics, 399:108925, 2019.  
[45] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 1989.  
[46] Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural Networks, 1991.  
[47] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J Davison. In-place scene labelling and understanding with implicit scene representation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15838-15847, 2021.  
[48] Suhani Vora, Noha Radwan, Klaus Greff, Henning Meyer, Kyle Genova, Mehdi SM Sajjadi, Etienne Pot, Andrea Tagliasacchi, and Daniel Duckworth. Nesf: Neural semantic fields for generalizable semantic segmentation of 3d scenes. arXiv preprint arXiv:2111.13260, 2021.  
[49] François Chollet. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1251-1258, 2017.  
[50] Michael Oechsle, Lars Mescheder, Michael Niemeyer, Thilo Strauss, and Andreas Geiger. Texture fields: Learning texture representations in function space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4531-4540, 2019.

[51] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2304-2314, 2019.  
[52] Guandao Yang, Serge Belongie, Bharath Hariharan, and Vladlen Koltun. Geometry processing with neural fields. Advances in Neural Information Processing Systems, 34, 2021.  
[53] Pietro Perona and Jitendra Malik. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7):629-639, 1990.  
[54] Stanley Osher and Leonid I Rudin. Feature-oriented image enhancement using shock filters. SIAM Journal on numerical analysis, 27(4):919-940, 1990.  
[55] Xue-Cheng Tai, Stanley Osher, and Randi Holm. Image inpainting using a tv-stokes equation. In Image Processing based on partial differential equations, pages 3-22. Springer, 2007.  
[56] Zhouchen Lin, Wei Zhang, and Xiaou Tang. Designing partial differential equations for image processing by combining differential invariants, 2009.  
[57] Lars Hörme, Claudia Frohn-Schauf, Stefan Henn, and Kristian Witsch. Total variation based image registration. In Image processing based on partial differential equations, pages 343-361. Springer, 2007.  
[58] Risheng Liu, Junjie Cao, Zhouchen Lin, and Shiguang Shan. Adaptive partial differential equation learning for visual saliency detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3866-3873, 2014.  
[59] Adam Rabcewicz. Clg method for optical flow estimation based on gradient constancy assumption. In Image Processing Based on Partial Differential Equations, pages 57-66. Springer, 2007.  
[60] Guillermo Sapiro. Geometric partial differential equations and image analysis. Cambridge university press, 2006.  
[61] Frédéric Cao. Geometric curve evolution and image processing. Springer Science & Business Media, 2003.  
[62] Bart M Haar Romeny. Geometry-driven diffusion in computer vision, volume 1. Springer Science & Business Media, 2013.  
[63] Leonid I Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4):259-268, 1992.  
[64] Karol Gregor and Yann LeCun. Learning fast approximations of sparse coding. In Proceedings of the 27th international conference on international conference on machine learning, pages 399-406, 2010.  
[65] Jialin Liu and Xiaohan Chen. Alist: Analytic weights are as good as learned weights in list. In International Conference on Learning Representations (ICLR), 2019.  
[66] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.  
[67] Marco Bevilacqua, Aline Roumy, Christine Guillemot, and Marie line Alberi Morel. Low-complexity single-image super-resolution based on nonnegative neighbor embedding. In Proceedings of the British Machine Vision Conference, pages 135.1-135.10. BMVA Press, 2012.  
[68] Roman Zeyde, Michael Elad, and Matan Protter. On single image scale-up using sparse-representations. In International conference on curves and surfaces, pages 711-730. Springer, 2010.

[69] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.  
[70] I Sobel. An isotropic  $3 \times 3$  image gradient operator, presentation at stanford ai project (1968), 2014.  
[71] John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679-698, 1986.  
[72] Judith MS Prewitt et al. Object enhancement and extraction. Picture processing and Psychopictorics, 10(1):15-19, 1970.  
[73] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142-3155, 2017.  
[74] Saeed V Vaseghi. Advanced digital signal processing and noise reduction. John Wiley & Sons, 2008.  
[75] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Multi-stage progressive image restoration. In CVPR, 2021.  
[76] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161, 2021.  
[77] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
