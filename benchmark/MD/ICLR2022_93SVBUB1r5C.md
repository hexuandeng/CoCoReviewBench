# LEARNING WITH CONVOLUTION AND POOLING OPERATIONS IN KERNEL METHODS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent empirical work has shown that hierarchical convolutional kernels inspired by convolutional neural networks (CNNs) significantly improve the performance of kernel methods in image classification tasks. A widely accepted explanation for the success of these architectures is that they encode hypothesis classes that are suitable for natural images. However, understanding the precise interplay between approximation and generalization in convolutional architectures remains a challenge. In this paper, we consider the stylized setting of covariates (image pixels) uniformly distributed on the hypercube, and fully characterize the RKHS of kernels composed of single layers of convolution, pooling, and downsampling operations. We then study the gain in sample efficiency of kernel methods using these kernels over standard inner-product kernels. In particular, we show that 1) the convolution layer breaks the curse of dimensionality by restricting the RKHS to 'local' functions; 2) local pooling biases learning towards low-frequency functions, which are stable by small translations; 3) downsampling may modify the high-frequency eigenspaces but leaves the low-frequency part approximately unchanged. Notably, our results quantify how choosing an architecture adapted to the target function leads to a large improvement in the sample complexity.

# 1 INTRODUCTION

Convolutional neural networks (CNNs) have become essential elements of the deep learning toolbox, achieving state-of-the-art performance in many computer vision tasks (Krizhevsky et al., 2012; He et al., 2016). CNNs are constructed by stacking convolution and pooling layers, which were shown to be paramount to their empirical success (LeCun et al., 2015). A widely accepted hypothesis to explain their favorable properties is that these architectures successfully encode useful properties of natural images: locality and compositionality of the data, stability by local deformations, and translation invariance. While some theoretical progress has been made in studying the approximation and generalization benefits brought by convolution and pooling operations (Cohen & Shashua, 2016a;b; Bietti, 2021), our mathematical understanding of the interaction between network architecture, image distribution, and efficient learning remains limited.

Consider  $\pmb{x} \in \mathbb{R}^d$  an input signal, which we can think of as a grayscale pixel representation of an image. For mathematical convenience, we will consider one-dimensional images with cyclic convention  $x_{d+i} := x_i$ , and denote  $\pmb{x}_{(k)} = (x_k, x_{k+1}, \dots, x_{k+q-1})$  the  $k$ -th patch of the signal  $\pmb{x}$ ,  $k \in [d]$ , with patch size  $q \leq d$ . Most of our results can be extended to two-dimensional images.

We further consider a simple convolutional neural network composed of a single convolution layer followed by local average pooling and downsampling. The network first computes the nonlinear convolution of  $N$  filters  $\pmb{w}_1, \dots, \pmb{w}_N \in \mathbb{R}^q$  with the image patches  $\pmb{x}_{(k)}$ . The outputs of the convolution operation  $\sigma(\langle \pmb{w}_i, \pmb{x}_{(k)} \rangle)$  are then averaged locally over segments of length  $\omega$  (local average pooling). This pooling operation is followed by downsampling which extracts one out of every  $\Delta$  output coordinates (for simplicity,  $\Delta$  is assumed to be a divisor of  $d$ ). Finally, the results are combined linearly using coefficients  $(a_{ik})_{i \in [N], k \in [d / \Delta]}$ :

$$
f _ {\mathrm {C N N}} (\boldsymbol {x}; \boldsymbol {a}, \boldsymbol {\Theta}) = \sqrt {\frac {\Delta}{N \omega d}} \sum_ {i \in [ N ]} \sum_ {k \in [ d / \Delta ]} a _ {i k} \sum_ {s \in [ \omega ]} \sigma \left(\langle \boldsymbol {w} _ {i}, \boldsymbol {x} _ {(k \Delta + s)} \rangle\right). \quad \text {(C N N - A P - D S)}
$$

Note that pooling and downsampling operations are often tied together in the literature. However in this work we will treat these two operations separately.

In the formula above, different values for  $q,\omega ,\Delta$  lead to different architectures with vastly different behaviors. For example, when  $q = \Delta = d$  and  $\omega = 1$ , we recover a two-layer fully-connected neural network  $f_{\mathsf{FC}}(\pmb {x};\pmb {\alpha},\Theta) = N^{-1 / 2}\sum_{i\in [N]}a_i\sigma (\langle \pmb {w}_i,\pmb {x}\rangle)$  which has the universal approximation property at large  $N$ . When  $\omega = \Delta = 1$  and  $q < d$ , the network is "locally connected"  $f_{\mathsf{LC}}(\pmb {x};\pmb {\alpha},\Theta) = N^{-1 / 2}\sum_{i\in [N],k\in [d]}a_{ik}\sigma (\langle \pmb {w}_i,\pmb {x}_{(k)}\rangle)$ , and not a universal approximator anymore: however,  $f_{\mathsf{LC}}$  vastly outperforms  $f_{\mathsf{FC}}$  in some cases (Li et al., 2020). For  $\omega >1$ , local pooling enables learning functions that are locally invariant by translations more efficiently than without pooling. For  $\omega = d$  (global pooling), the network only fits functions fully invariant by cyclic translations.

The aim of this paper is to formalize and quantify the trade-off between the target function class and the statistical efficiency brought by these different architectures. As a concrete first step in this direction, we consider kernel models that are naturally associated with the convolutional neural networks (CNN-AP-DS) through the neural tangent kernel perspective (Daniely et al., 2016; Jacot et al., 2018). Kernel methods have the advantage of 1) being tractable—leaving the computational issue of learning CNNs aside; 2) having well-understood approximation and generalization properties, which depends on the eigendecomposition of the kernel and the alignment between the target function and associated RKHS (Caponnetto & De Vito, 2007; Wainwright, 2019) (see Appendices B and C for background). While kernel models only describe neural networks in the lazy training regime (Chizat et al., 2019; Du et al., 2019b;a; Allen-Zhu et al., 2019; Zou et al., 2018) and miss important properties of deep learning, such as feature learning, we see that architecture choice is already crucial to enable efficient learning of 'image-like' functions in the fixed-feature regime.

Neural tangent kernels are obtained by linearizing the associated neural networks. Here we consider the tangent kernel associated to the network  $f_{\mathrm{CNN}}$  (c.f. Appendix A.2 for a detailed derivation):

$$
H _ {\omega , \Delta} ^ {\mathrm {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \frac {\Delta}{d \omega} \sum_ {k \in [ d / \Delta ]} \sum_ {s, s ^ {\prime} \in [ \omega ]} h \left(\left\langle \boldsymbol {x} _ {(k \Delta + s)}, \boldsymbol {y} _ {(k \Delta + s ^ {\prime})} \right\rangle / q\right), \tag {CK-AP-DS}
$$

where  $h: \mathbb{R} \to \mathbb{R}$  is related to the activation function  $\sigma$  in (CNN-AP-DS). As a linearization of CNNs, the kernel (CK-AP-DS) inherits some of the favorable properties of convolution, pooling, and downsampling operations. Indeed, a line of work (Mairal et al., 2014; Mairal, 2016; Arora et al., 2019; Li et al., 2019; Shankar et al., 2020) showed that, though performing slightly worse than CNNs, such (hierarchical) convolutional kernels have empirically outperformed the former state-of-the-art kernels. For instance, these kernels achieved test accuracy around  $87\% - 90\%$  on CIFAR-10, against  $79.6\%$  for the best former unsupervised feature-extraction method (Coates et al., 2011) (currently, the state-of-the-art CNNs can achieve test accuracy  $99\%$ ).

In this paper, we will further consider a stylized setting with input signal distribution  $\pmb{x} \sim \mathrm{Unif}(\mathcal{Q}^d)$  (uniform distribution over  $\mathcal{Q}^d \coloneqq \{-1, +1\}^d$  the discrete hypercube in  $d$  dimensions). This simple choice allows for a complete characterization of the eigendecomposition of  $H_{\omega,\Delta}^{\mathrm{CK}}$ , thanks to all patches having same marginal distribution  $\pmb{x}_{(k)} \sim \mathrm{Unif}(\mathcal{Q}^q)$ . We will be particularly interested in four specific choices of  $(q,\omega,\Delta)$  in (CK-AP-DS):

$$
H ^ {\mathrm {F C}} (\boldsymbol {x}, \boldsymbol {y}) = h \left(\langle \boldsymbol {x}, \boldsymbol {y} \rangle / d\right), \tag {FC}
$$

$$
H ^ {\mathrm {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \frac {1}{d} \sum_ {k \in [ d ]} h \left(\left\langle \boldsymbol {x} _ {(k)}, \boldsymbol {y} _ {(k)} \right\rangle / q\right), \tag {CK}
$$

$$
H _ {\omega} ^ {\mathrm {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \frac {1}{d \omega} \sum_ {k \in [ d ]} \sum_ {s, s ^ {\prime} \in [ \omega ]} h \left(\left\langle \boldsymbol {x} _ {(k + s)}, \boldsymbol {y} _ {(k + s ^ {\prime})} \right\rangle / q\right), \tag {CK-AP}
$$

$$
H _ {\mathrm {G P}} ^ {\mathrm {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \frac {1}{d} \sum_ {k, k ^ {\prime} \in [ d ]} h \left(\left\langle \boldsymbol {x} _ {(k)}, \boldsymbol {y} _ {(k ^ {\prime})} \right\rangle / q\right). \tag {CK-GP}
$$

These kernels are respectively the neural tangent kernels of a fully-connected network  $f_{\mathrm{FC}}$  (FC), a convolutional network  $f_{\mathrm{LC}}$  (CK), a convolutional network followed by local average pooling (CK-AP) and a convolutional network followed by global pooling (CK-GP). We will further be interested in (CK-GP) with patch size  $q = d$ , which we denote  $H_{\mathrm{GF}}^{\mathrm{FC}}$ : this corresponds to a convolutional kernel with full-size patches  $q = d$ , followed by global pooling.

In this paper, we first characterize the reproducing kernel Hilbert space (RKHS) of these convolutional kernels, and then investigate their generalization properties in the regression setup. More specifically, assume  $\{(x_i,y_i)\}_{i\leq n}$  are  $n$  i.i.d. samples with  $\pmb {x}_i\sim \mathrm{Unif}(\mathcal{Q}^d)$  and  $y_{i} = f_{\star}(\pmb {x}_{i}) + \varepsilon_{i}$ . Here  $f_{\star}\in L^{2}(\mathcal{Q}^{d})$  and  $(\varepsilon_{i})_{i\leq n}$  are independent errors with mean zero and variance bounded by  $\sigma_{\varepsilon}^{2}$ . We will focus on the generalization error of kernel ridge regression (KRR) (see Appendix B.1 for general kernel methods). In particular, given a kernel function  $H:\mathcal{Q}^d\times \mathcal{Q}^d\to \mathbb{R}$  and a regularization parameter  $\lambda \geq 0$ , the KRR estimator is the solution of the tractable convex problem

$$
\hat {f} _ {\lambda} = \underset {f \in \mathcal {H}} {\arg \min } \left\{\sum_ {i \in [ n ]} \left(y _ {i} - f \left(\boldsymbol {x} _ {i}\right)\right) ^ {2} + \lambda \| f \| _ {\mathcal {H}} ^ {2} \right\}, \tag {KRR}
$$

where  $\mathcal{H}$  is the RKHS associated to  $H$  with RKHS norm  $\| \cdot \|_{\mathcal{H}}$ . We denote the test error with square loss by  $R(f_{\star},\hat{f}_{\lambda}) = \mathbb{E}_{\boldsymbol{x}}\{(f_{\star}(\boldsymbol {x}) - \hat{f}_{\lambda}(\boldsymbol {x}))^{2}\}$ . We will sometimes consider the expected test error  $\mathbb{E}_{\varepsilon}\{R(f_{\star},\hat{f}_{\lambda})\}$ , where expectation is taken with respect to noise  $\varepsilon = (\varepsilon_{i})_{i\leq n}$  in the training data.

The generalization properties of the kernels  $H^{\mathrm{FC}}$  and  $H_{\mathrm{GP}}^{\mathrm{FC}}$  were recently studied in Mei et al. (2021b); Bietti et al. (2021). In particular, they showed that global pooling (kernel  $H_{\mathrm{GP}}^{\mathrm{FC}}$ ) leads to a gain of a factor  $d$  in sample complexity when fitting cyclic invariant functions, but still suffers from the curse of dimensionality ( $H_{\mathrm{GP}}^{\mathrm{FC}}$  only fits very smooth functions in high-dimension). More precisely, Mei et al. (2021b) considered the high-dimensional framework of Mei et al. (2021a) and showed the following: KRR with  $H^{\mathrm{FC}}$  requires  $n \approx d^{\ell}$  samples to fit degree- $\ell$  cyclic polynomials, while KRR with  $H_{\mathrm{GP}}^{\mathrm{FC}}$  only needs  $n \approx d^{\ell-1}$ . To enable milder dependence on the dimension  $d$ , further structural assumptions on the kernel and the target function should be considered (for instance, in this paper, we use the kernel  $H^{\mathrm{CK}}$  and consider 'local' functions).

# 1.1 SUMMARY OF MAIN RESULTS

Our contributions are two-fold. First, we describe the RKHS associated to the convolutional kernel (CK-AP-DS) in the stylized setting  $\pmb{x} \sim \mathrm{Unif}(\mathcal{Q}^d)$ , which provides a fully explicit illustration of the roles of convolution, pooling and downsampling operations in learning specific classes of functions. Second, based on the eigendecomposition of the kernels, we provide generalization bounds on kernel ridge regression both in the fixed and high-dimension settings. In particular, we quantify the gain in sample complexity achieved by different architectures when learning target functions with corresponding structures.

Let us define the  $q$ -local function class  $L^2(\mathcal{Q}^d, \mathrm{Loc}_q)$  and the cyclic  $q$ -local function class  $L^2(\mathcal{Q}^d, \mathrm{CycLoc}_q)$  (subspace of  $L^2(\mathcal{Q}^d, \mathrm{Loc}_q)$  consisting of cyclic-invariant functions) as follow:

$$
L ^ {2} \left(\mathcal {Q} ^ {d}, \operatorname {L o c} _ {q}\right) = \left\{f \in L ^ {2} \left(\mathcal {Q} ^ {d}\right): \exists \left\{g _ {k} \right\} _ {k \in [ d ]} \subseteq L ^ {2} \left(\mathcal {Q} ^ {q}\right), f (\boldsymbol {x}) = \sum_ {k \in [ d ]} g _ {k} \left(\boldsymbol {x} _ {(k)}\right) \right\}, \quad (\text {L O C})
$$

$$
L ^ {2} \left(\mathcal {Q} ^ {d}, \operatorname {C y c L o c} _ {q}\right) = \left\{f \in L ^ {2} \left(\mathcal {Q} ^ {d}\right): \exists g \in L ^ {2} \left(\mathcal {Q} ^ {q}\right), f (\boldsymbol {x}) = \sum_ {k \in [ d ]} g \left(\boldsymbol {x} _ {(k)}\right) \right\}. \quad (\text {C Y C - L O C})
$$

We summarize below some of the insights that follow from our results.

One-layer convolutional layer. The RKHS of  $H^{\mathsf{CK}}$  is constituted of  $q$ -local functions  $L^{2}(\mathcal{Q}^{d},\mathrm{Loc}_{q})$ . Furthermore, the decay of the eigenvalues of  $H^{\mathsf{CK}}$  is controlled by a  $q$ -dimensional kernel, instead of a  $d$ -dimensional kernel for  $H^{\mathsf{FC}}$ . In particular, this implies that the test error of kernel ridge regression decays at rate  $n^{-O(1 / q)}$  instead of  $n^{-O(1 / d)}$ , when the target function is in  $L^{2}(\mathcal{Q}^{d},\mathrm{Loc}_{q})$ . Hence, when  $q \ll d$ , kernel methods with convolutional kernel  $H^{\mathsf{CK}}$  breaks the curse of dimensionality.

Average pooling. The RKHS of  $H_{\omega}^{\mathsf{CK}}$  is still constituted of  $q$ -local functions  $f \in L^{2}(\mathcal{Q}^{d}, \mathrm{Loc}_{q})$ , but penalizes differently the frequency components  $f_{j}(\pmb{x})$  by reweighting their eigenspaces by a factor  $\kappa_{j}$ , where  $f_{j}(\pmb{x}) = \sum_{k \in [d]} \rho_{j}^{k} f(t_{k} \cdot \pmb{x})$  with  $\rho_{j} = e^{\frac{2i\pi j}{d}}$  and  $t_{k} \cdot \pmb{x} = (x_{k+1}, \dots, x_{d}, x_{1}, \dots, x_{k})$  denotes the  $k$ -shift. As  $\omega$  increases, local pooling penalizes more and more heavily the high-frequency components  $(\kappa_{j} \ll 1)$ , while making low-frequency components statistically easier to learn  $(\kappa_{j} \gg 1)$ . For global pooling  $\omega = d$ ,  $H_{\mathsf{GP}}^{\mathsf{CK}}$  only learns cyclic local functions  $L^{2}(\mathcal{Q}^{d}, \mathrm{CycLoc}_{q})$  and enjoy a factor  $d$  gain in statistical complexity compared to  $H^{\mathsf{CK}}$ .

Table 1: Sample size  $n$  required to fit a  $q$ -local cyclic-invariant polynomial of degree  $\ell$  using kernel ridge regression (KRR) with the 5 different kernels of interest in this paper.  

<table><tr><td>To fit a degree ℓ polynomial</td><td>HFC</td><td>HFCGP</td><td>HCK</td><td>HCKω</td><td>HCKGP</td></tr><tr><td>Sample complexity</td><td>dℓ</td><td>dℓ-1</td><td>dqℓ-1</td><td>dqℓ-1/ω</td><td>qℓ-1</td></tr></table>

Downsampling. When  $\Delta \leq \omega$ , downsampling after average pooling leaves the low-frequency eigenspaces of  $H_{\omega}^{\mathsf{CK}}$  stable. In particular, the downsampling operation does not modify the statistical complexity of learning low-frequency functions in one-layer kernels, while being potentially beneficial in further layers in deep convolutional kernels.

In Table 1, we report our high-dimensional predictions for the sample complexity of learning  $q$ -local cyclic-invariant polynomials of degree  $\ell$ . These results are proven within the framework of Mei et al. (2021a) (see Appendix C for background).

The rest of the paper is organized as follows. We discuss related work in Section 1.2. In Section 2, we present our main results on convolutional kernels and describe precisely the roles of convolution, pooling and downsampling operations. We briefly mention the multilayer case in Section 2.4. Finally, we present a numerical simulation on synthetic data in Section 3 and conclude in Section 4. Some details and discussions are deferred to Appendix A.

# 1.2 RELATED WORK

Convolutional kernels have been considered in Mairal et al. (2014); Mairal (2016); Li et al. (2019); Shankar et al. (2020); Bietti (2021); Thiry et al. (2021). In particular, they showed that these architectures achieve good results in image classification (90% accuracy on Cifar10) and that pooling and downsampling were necessary for their good performance (Li et al., 2019).

The generalization error of kernel ridge regression (KRR) has been well-studied in both the fixed dimension regime (Wainwright, 2019, Chap. 13), Caponnetto & De Vito (2007) and the high-dimensional regimes El Karoui (2010); Liang et al. (2020); Ghorbani et al. (2020; 2021); Mei et al. (2021b). These results show that the generalization error depends on the eigenvalues and eigenfunctions of the kernel, and the alignment of the kernel with the target function.

Recently, a few theoretical work have considered the generalization properties of invariant kernels and convolutional kernels (Scetbon & Harchaoui, 2020; Mei et al., 2021b; Bietti et al., 2021; Favero et al., 2021). In particular, Mei et al. (2021b) consider convolutional kernel with global pooling and full-size patches  $q = d$ , and show a gain of factor  $d$  in sample complexity when learning cyclic functions, compared to inner-product kernels. Bietti et al. (2021) considers additionally kernels that are stable with respect to local deformations, and similarly quantify the sample complexity gain.

See Malach & Shalev-Shwartz (2020); Li et al. (2020) for more theoretical results on the separation between convolutional and fully connected neural networks, and Boureau et al. (2010); Cohen & Shashua (2016b) for the inductive bias of pooling operations in convolutional neural networks.

# 2 MAIN RESULTS

We start by introducing some basic background on functions on the hypercube and eigendecomposition of kernel operators in Section 2.1. We first consider a kernel with a single convolution layer in Section 2.2, and characterize its eigendecomposition and generalization properties. We then show how these results are modified when applying local average pooling and downsampling in Section 2.3. Finally, we briefly discuss multilayer convolutional kernels in Section 2.4.

# 2.1 FUNCTIONS ON THE HYPERCUBE AND EIGENDECOMPOSITION OF KERNEL OPERATORS

Recall that we work on the  $d$ -dimensional hypercube  $\mathcal{Q}^d \coloneqq \{-1, +1\}^d$ . Let  $L^2(\mathcal{Q}^d) = L^2(\mathcal{Q}^d, \mathrm{Unif})$  be the  $2^d$ -dimensional vector space of all functions  $f: \mathcal{Q}^d \to \mathbb{R}$ , with scalar product  $\langle f, g \rangle_{L^2} \coloneqq \mathbb{E}_{\boldsymbol{x} \sim \mathrm{Unif}(\mathcal{Q}^d)}[f(\boldsymbol{x})g(\boldsymbol{x})]$ . Let  $\| \cdot \|_{L^2}$  be the norm associated with the scalar product. We

introduce the set of Fourier functions  $\{Y_S^{(d)}(\pmb{x})\}_{S\subseteq [d]}$  which forms an orthonormal basis of  $L^2 (\mathcal{Q}^d)$ . For any subset  $S\subseteq [d]$ , the Fourier function is defined as  $Y_{S}^{(d)}(\pmb {x})\coloneqq \prod_{i\in S}x_{i}$  with the convention that  $Y_{\emptyset}^{(d)}\coloneqq 1$  (it is easy to verify that  $\langle Y_S^{(d)},Y_{S'}^{(d)}\rangle_{L^2} = \mathbf{1}_{S = S'}$ ). We will omit the superscript  $(d)$  which will be clear from context and write  $Y_{S}\coloneqq Y_{S}^{(d)}$ .

Consider a nonnegative definite kernel function  $H: \mathcal{Q}^p \times \mathcal{Q}^p \to \mathbb{R}$  ( $p = d$  or  $q$  in this paper) with associated integral operator  $\mathbb{H}: L^2(\mathcal{Q}^p) \to L^2(\mathcal{Q}^p)$  defined as  $\mathbb{H}f(\boldsymbol{u}) = \mathbb{E}_\boldsymbol{v}\{h(\boldsymbol{u}, \boldsymbol{v})f(\boldsymbol{v})\}$  with  $\boldsymbol{v} \sim \mathrm{Unif}(\mathcal{Q}^p)$ . By spectral theorem of compact operators, there exists an orthonormal basis  $\{\psi_j\}_{j \geq 1}$  of  $L^2(\mathcal{Q}^p)$  and nonnegative eigenvalues  $(\lambda_j)_{j \geq 1}$  such that  $\mathbb{H} = \sum_{j \geq 1} \lambda_j \psi_j \psi_j^*$  (i.e.,  $H(\boldsymbol{u}, \boldsymbol{v}) = \sum_{j \geq 1} \lambda_j \psi_j(\boldsymbol{u}) \psi_j(\boldsymbol{v})$  for any  $\boldsymbol{u}, \boldsymbol{v} \in L^2(\mathcal{Q}^p)$ ).

The most widespread example are inner-product kernels defined as  $H(\mathbf{u},\mathbf{v}) \coloneqq h(\langle \mathbf{u},\mathbf{v}\rangle /p)$  for some function  $h:\mathbb{R}\to \mathbb{R}$ . Inner-product kernels have the following simple eigendecomposition in  $L^2 (\mathcal{Q}^p)$  (taking here  $\mathbf{u},\mathbf{v}\in \mathcal{Q}^p$ ):

$$
h \left(\langle \boldsymbol {u}, \boldsymbol {v} \rangle / p\right) = \sum_ {\ell = 0} ^ {p} \xi_ {p, \ell} (h) \sum_ {S \subseteq [ p ], | S | = \ell} Y _ {S} (\boldsymbol {u}) Y _ {S} (\boldsymbol {v}), \tag {1}
$$

where  $\xi_{p,\ell}(h)$  is the  $\ell$ -th Gegenbauer coefficient of  $h(\cdot/\sqrt{p})$  in dimension  $p$ , i.e.,

$$
\xi_ {p, \ell} (h) = \mathbb {E} _ {\boldsymbol {u} \sim \operatorname {U n i f} \left(\mathcal {Q} ^ {p}\right)} \left[ h \left(\langle \boldsymbol {u}, \boldsymbol {e} \rangle / p\right) Q _ {\ell} ^ {(p)} \left(\langle \boldsymbol {u}, \boldsymbol {e} \rangle\right) \right], \tag {2}
$$

for  $e \in \mathcal{Q}^p$  arbitrary and  $Q_{\ell}^{(p)}$  the degree- $\ell$  Gegenbauer polynomial on  $\mathcal{Q}^p$  (see Appendix D for details). Note that  $(\xi_{p,\ell})_{0 \leq \ell \leq q}$  are non-negative by positive semidefiniteness of the kernel. We will write  $\xi_{p,\ell} \coloneqq \xi_{p,\ell}(h)$  and use extensively the decomposition identity (1) in the rest of the paper.

# 2.2 ONE-LAYER CONVOLUTIONAL KERNEL

We first consider the convolutional kernel  $H^{\mathsf{CK}}$  (CK) given by a one-layer convolution layer with patch size  $q$  and inner-product kernel function  $h: \mathbb{R} \to \mathbb{R}$ :

$$
H ^ {\mathrm {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \frac {1}{d} \sum_ {k = 1} ^ {d} h \left(\left\langle \boldsymbol {x} _ {(k)}, \boldsymbol {y} _ {(k)} \right\rangle / q\right), \tag {3}
$$

where we recall that  $\pmb{x}_{(k)} = (x_k,\dots ,x_{k + q - 1})\in \mathcal{Q}^q$  is the  $k$  th patch of the image with size  $q$

Before stating the eigendecomposition of  $H^{\mathsf{CK}}$ , we introduce some notations. For any subset  $S \subseteq [d]$ , denote  $\gamma(S)$  the diameter of  $S$  with cyclic convention, i.e.,  $\gamma(S) = \max \{ \min \{ \mod(j - i, d) + 1, \mod(i - j, d) + 1 \} : i, j \in S \}$  (e.g.,  $\gamma(\{2, d\}) = 3$ ). For any integer  $\ell \leq q$ , consider the set  $\mathcal{E}_{\ell} = \{ S \subseteq [d] : |S| = \ell, \gamma(S) \leq q \}$  of all subsets of  $[d]$  of size  $\ell$  with diameter less or equal to  $q$ . We will assume throughout this paper that  $q \leq d/2$  to avoid additional overlap between sets.

Proposition 1 (Eigendecomposition of  $H^{\mathsf{CK}}$ ). Let  $H^{\mathsf{CK}}$  be a convolutional kernel as defined in Eq. (3). Then  $H^{\mathsf{CK}}$  admits the following eigendecomposition:

$$
H ^ {\mathbb {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \xi_ {q, 0} + \sum_ {\ell = 1} ^ {q} \sum_ {S \in \mathcal {E} _ {\ell}} \frac {r (S) \xi_ {q , \ell}}{d} \cdot Y _ {S} (\boldsymbol {x}) Y _ {S} (\boldsymbol {y}), \tag {4}
$$

where  $r(S) = q + 1 - \gamma (S)$  and  $\xi_{q,\ell}\geq 0$  is defined in Eq. (2).

Notice that  $Y_{S}$  with  $\gamma(S) > q$  (monomials with support not contained in a segment of size  $q$ ) are in the null space of  $H^{\mathsf{CK}}$ . Hence (as long as  $\xi_{q,\ell} > 0$  for all  $0 \leq \ell \leq q$ ), the RKHS associated to  $H^{\mathsf{CK}}$  exactly contains all the functions in the  $q$ -local function class  $L^{2}(\mathcal{Q}^{d},\mathrm{Loc}_{q})$  (c.f. Eq. (LOC)). In words,  $L^{2}(\mathcal{Q}^{d},\mathrm{Loc}_{q})$  consists of functions that are localized on patches, with no long-range interactions between different parts of the image. An example of local function with  $q = 3$  is given by  $f(\pmb{x}) = x_{1}x_{2}x_{3} + x_{4}x_{6} + x_{5}$ .

On the other hand, the RKHS associated to the fully-connected kernel  $H^{\mathrm{FC}}$  (FC) typically contains all the functions in  $L^2(\mathcal{Q}^d)$  (under genericity assumptions on  $h$ ). The RKHS with convolution

$\dim(L^2(\mathcal{Q}^d, \operatorname{Loc}_q)) = d2^{q-1} + 1$  is significantly smaller than  $\dim(L^2(\mathcal{Q}^d)) = 2^d$ , which prompts the following question: what is the statistical advantage of using  $H^{\mathsf{CK}}$  over  $H^{\mathsf{FC}}$  when learning functions in  $L^2(\mathcal{Q}^d, \operatorname{Loc}_q)$ ?

We first consider the classical approach to bounding the test error of Caponnetto & De Vito (2007); Wainwright (2019); Bach (2021) which relies on the following two standard assumptions:

(A1) Capacity condition: we assume  $\mathcal{N}(h,\lambda)\coloneqq \mathrm{Tr}[h / (h + \lambda \mathbf{I})^{-1}]\leq C_h\lambda^{-1 / \alpha}$  with  $\alpha >1$  
(A2) Source condition:  $\| h^{-\beta /2}g\|_{L^2}\leq B$  with  $\beta >\frac{\alpha - 1}{\alpha}$  and  $B\geq 0$

The capacity condition (A1) characterizes the size of the RKHS: for increasing  $\alpha$ , the RKHS contains less and less functions. The source condition (A2) characterizes the regularity of the target function (the 'source') with respect to the kernel: increasing  $\beta$  corresponds to smoother and smoother functions. See Appendix B.2 for more discussions.

Based on these two assumptions, we can apply standard bounds on the KRR test error and obtain:

Theorem 1 (Generalization error of KRR with  $H^{\mathsf{CK}}$ ). Let  $h: \mathbb{R} \to \mathbb{R}$  be an inner-product kernel satisfying (A1). Let  $f_{\star} \in L^{2}(\mathcal{Q}^{d},\mathrm{Loc}_{q})$  with  $f(\pmb {x}) = \sum_{k\in [d]}g_k(\pmb{x}_{(k)})$  satisfying (A2) with  $\sum_{k\in [d]}\| h^{-\beta /2}g_k\|_{L^2}^2\leq B^2$ . Then there exists  $C_1,C_2,C_3 > 0$  constants that only depend on (A1) and (A2) (and independent of  $d$ ), such that for  $n\geq C_1\max (\| f_\star \|_{L^\infty}^2,d)$  and  $\lambda_{*} = \frac{C_{2}}{d} (d / n)^{\frac{\alpha}{\alpha\beta + 1}}$ ,

$$
\mathbb {E} _ {\varepsilon} \left\{R \left(f _ {\star}, \hat {f} _ {\lambda_ {\star}}\right) \right\} \leq C _ {3} \left(\frac {d}{n}\right) ^ {\frac {\alpha \beta}{\alpha \beta + 1}}. \tag {5}
$$

Note that the exponent  $\frac{\beta\alpha}{\beta\alpha + 1}$  only depends on the  $q$ -dimensional kernel  $h$ . Hence, the generalization bound with respect to  $(n / d)$  is independent of the dimension  $d$  of the image. Let's compare to KRR with inner-product kernel  $H^{\mathrm{FC}}$  (FC): from Caponnetto & De Vito (2007), we have the minmax rate  $\mathbb{E}_{\varepsilon}\{R(f_{\star},\hat{f}_{\lambda})\} \asymp n^{-\frac{\tilde{\alpha}\tilde{\beta}}{\tilde{\alpha}\tilde{\beta} + 1}}$  where  $h$  is now defined in  $d$  dimension and verifies (A1) and (A2) with constants  $\tilde{\alpha},\tilde{\beta}$ . Typically, if  $f_{\star}$  is only assumed Lipschitz, then  $\tilde{\beta}\tilde{\alpha} = O(1 / d)$ , which leads to a minmax rate  $n^{-O(1 / d)}$  for  $H^{\mathrm{FC}}$ , while for  $H^{\mathrm{CK}}$ ,  $\beta \alpha = O(1 / q)$ , which leads to a minmax rate  $n^{-O(1 / q)}$ . Hence, for  $q \ll d$ ,  $H^{\mathrm{CK}}$  breaks the curse of dimensionality by restricting the RKHS to 'local' functions. This was already noticed in Scetbon & Harchaoui (2020); Favero et al. (2021) for slightly simplified architectures.

Theorem 1 and results of this type suffers from several limitations: 1) they are tight only in terms of the exponent of  $n$  in a minmax sense; 2) they do not provide comparisons for specific subclasses of functions; 3) in order to obtain the minmax rate, the regularization parameter  $\lambda$  has to be carefully tuned to balance the bias and variance terms, which is in contrast to modern practice where often the model is trained until interpolation. This led several groups to consider instead the test error of KRR in a high-dimensional limit Ghorbani et al. (2021); Mei et al. (2021a); Canatar et al. (2021) and derive exact asymptotic predictions correct up to an additive vanishing constant for any  $f_{\star} \in L^{2}$  (see Appendix C for more details).

Using the general framework in Mei et al. (2021a), we get the following result for  $q, d$  large:

Theorem 2 (Generalization error of KRR with  $H^{\mathsf{CK}}$  in high-dimension (informal)). Let  $f_{\star} \in L^{2}(\mathcal{Q}^{d}, \mathrm{Loc}_{q})$  and  $h: \mathbb{R} \to \mathbb{R}$  verifying some 'genericity condition'. Then for  $n = dq^{s - 1 + \nu}$  with  $0 < \nu < 1$ , and  $\lambda = O(1)$  (in particular  $\lambda = 0$  works), we have

$$
\hat {f} _ {\lambda} = \mathrm {P} _ {\mathcal {E} _ {\leq s, \nu}} f _ {\star} + o _ {q} (1), \tag {6}
$$

where  $\mathsf{P}_{\mathcal{E}_{\leq s,\nu}}$  is the projection on the span of  $Y_{S}$  with either  $|S| < \mathsf{s}$  and  $S \in \mathcal{E}_{|S|}$  or  $|S| = \mathsf{s}$  and  $\gamma(S) \leq q(1 - q^{-\nu})$ .

See Appendix C.1 for a rigorous statement. In words, when  $dq^{\mathbf{s} - 1} \ll n \ll dq^{\mathbf{s}}$ , KRR with  $H^{\mathsf{CK}}$  only learns a degree- $\mathbf{s}$  polynomial approximation to  $f_{\star}$ .

On the other hand, when considering the standard inner-product kernel  $H^{\mathrm{FC}}$  (FC) we get:

Theorem 3 (Generalization error of KRR with  $H^{\mathsf{FC}}$  in high-dimension (informal)). Let  $f_{\star} \in L^{2}(\mathcal{Q}^{d})$  and  $\tilde{h} : \mathbb{R} \to \mathbb{R}$  with some 'genericity condition'. Then for  $d^s \ll n \ll d^{s+1}$  and  $\lambda = O(1)$ ,

$$
\hat {f} _ {\lambda} = \mathrm {P} _ {\leq s} f _ {\star} + o _ {d} (1), \tag {7}
$$

where  $\mathsf{P}_{\leq s}$  is the projection on the subspace of degree- $\mathsf{s}$  polynomials.

This theorem was proved in Ghorbani et al. (2021); Mei et al. (2021a). Notice that Eq. (7) does not depend on the structure of  $f_{\star}$ . Hence, when  $f_{\star} \in L^{2}(\mathcal{Q}^{d}, \mathrm{Loc}_{q})$ , Theorems 2 and 3 show a clear statistical advantage of  $H^{\mathsf{CK}}$  over  $H^{\mathsf{FC}}$  when  $q \ll d$  (and therefore of one-layer CNNs over fully-connected neural networks in the kernel regime).

# 2.3 LOCAL AVERAGE POOLING AND DOWNSAMPLING

In many applications such as object recognition, we expect the target function to depend mildly on the absolute spatial position of an object and to be stable under small shifts of the input. To take this local invariance into account, convolution layers are often followed by a pooling operation. Here we consider local average pooling on a segment of length  $\omega$  and obtain the kernel

$$
H _ {\omega} ^ {\mathsf {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \frac {1}{d \omega} \sum_ {k \in [ d ]} \sum_ {s, s ^ {\prime} \in [ \omega ]} h \left(\left\langle \boldsymbol {x} _ {(k + s)}, \boldsymbol {y} _ {(k + s ^ {\prime})} \right\rangle / q\right). \tag {8}
$$

Define  $\mathcal{S}_{\ell} = \{S \subseteq [q] : |S| = \ell\}$  as the collection of sets of size  $\ell$ . We further define an equivalence relation  $\sim$  on  $\mathcal{S}_{\ell}$ :  $S \sim S'$  if  $S'$  is a translated subset of  $S$  in  $[q]$  (without cyclic convention). We denote  $\mathcal{C}_{\ell}$  the quotient set of  $\mathcal{A}_{\ell}$  under the equivalence relation  $\sim$ .

Proposition 2 (Eigendecomposition of  $H_{\omega}^{\mathsf{CK}}$ ). Let  $H_{\omega}^{\mathsf{CK}}$  be a convolutional kernel with local average pooling as defined in Eq. (8). Then  $H_{\omega}^{\mathsf{CK}}$  admits the following eigendecomposition:

$$
H _ {\omega} ^ {\mathsf {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \omega \xi_ {q, 0} + \sum_ {\ell = 1} ^ {q} \sum_ {S \in \mathcal {C} _ {\ell}} \sum_ {j \in [ d ]} \frac {\kappa_ {j} r (S) \xi_ {q , \ell}}{d} \cdot \psi_ {j, S} (\boldsymbol {x}) \psi_ {j, S} (\boldsymbol {y}), \tag {9}
$$

where (denoting  $k + S$  the translated set  $S$  by  $k$  positions with cyclic convention in  $[d]$ )

$$
\kappa_ {j} = 1 + 2 \sum_ {k = 1} ^ {\omega - 1} (1 - k / \omega) \cos \left(\frac {2 \pi j k}{d}\right), \quad \psi_ {j, S} (\boldsymbol {x}) = \frac {1}{\sqrt {d}} \sum_ {k = 1} ^ {d} e ^ {\frac {2 i \pi j k}{d}} Y _ {k + S} (\boldsymbol {x}). \tag {10}
$$

First notice that, as long as  $\gcd(\omega, d) = 1$ , the RKHS associated to  $H^{\mathsf{CK}}$  contains the same set of functions as the RKHS of  $H^{\mathsf{CK}}$ , i.e., all local functions  $L^2(\mathcal{Q}^d, \mathrm{Loc}_q)$ . (There are  $\gcd(\omega, d) - 1$  number of zero weights:  $\kappa_j = 0$  for all  $j \in [d-1]$  such that  $d$  is a divisor of  $j\omega$ . See Appendix A.3 for details.) However  $H^{\mathsf{CK}}$  will penalize different frequency components of the functions differently. Denote  $f_j(\pmb{x})$  the  $j$ -th component of the discrete Fourier transform of the function, i.e.,  $f_j(\pmb{x}) = \frac{1}{\sqrt{d}} \sum_{k \in [d]} \rho_j^k f(t_k \cdot \pmb{x})$  where  $\rho_j = e^{2i\pi j/d}$  and  $t_k \cdot \pmb{x} = (x_{k+1}, \dots, x_d, x_1, \dots, x_k)$  is the cyclic shift by  $k$  pixels. Then  $H^{\mathsf{CK}}$  reweights the eigenspaces associated with  $f_j(\pmb{x})$  by a factor  $\kappa_j$ , promoting low-frequency components ( $\kappa_j > 1$ ) and penalizing the high-frequencies ( $\kappa_j < 1$ ). In words, pooling biases the learning towards low-frequency functions, which are stable by small shifts.

Let us focus on two special choices here: the pooling parameter  $\omega = 1$  and  $\omega = d$ . When  $\omega = 1$ ,  $H_{\omega}^{\mathsf{CK}}$  reduces to  $H^{\mathsf{CK}}$  ( $\kappa_{j} = 1$  for all  $j \in [d]$ ) which does not bias towards either low or high frequency components. When  $\omega = d$ , we denote such kernel  $H_{\omega = d}^{\mathsf{CK}}$  by  $H_{\mathsf{GP}}^{\mathsf{CK}}$  which corresponds to global average pooling. In this case, we have  $\kappa_{d} = d$  and  $\kappa_{j} = 0$  for  $j < d$  which enforces exact invariance under the group of cyclic translations. More precisely,  $H_{\mathsf{GP}}^{\mathsf{CK}}$  has RKHS that contains all cyclic q-local functions  $f(\pmb{x}) = \sum_{k \in [d]} g(\pmb{x}_{(k)}) \in L^{2}(\mathcal{Q}^{d}, \mathrm{CycLoc}_{q})$  (c.f. Eq. (CYC-LOC)).

We obtain a bound on the test error of KRR with  $H_{\omega}^{\mathrm{CK}}$  similar to Theorem 1, but with  $d$  replaced by an effective dimension  $d^{\mathrm{eff}}$ .

Theorem 4 (Generalization of KRR with average pooling (fixed  $d, q$ )). Assume that  $h: \mathbb{R} \to \mathbb{R}$  has  $\xi_{q,0} = 0$  and satisfies (A1). Further assume  $(A2')$  that  $\|(H_{\omega}^{\mathsf{CK}} / \omega)^{-\beta /2}f_{\star}\|_{L^2} \leq B$ . Define

$d^{\mathrm{eff}} = \sum_{j\in [d]:\kappa_j > 0}(\kappa_j / \omega)^{1 / \alpha}$ . Then there exists  $C_1,C_2,C_3 > 0$  constants independent of  $d$ , such that for  $n\geq C_1\max (\| f_\star \| _L^2,d_{\mathrm{eff}})$  and setting  $\lambda_{*} = C_{2}(d_{\mathrm{eff}} / n)^{\frac{\alpha}{\alpha\beta + 1}}$ , we get

$$
\mathbb {E} _ {\varepsilon} \left\{R \left(f _ {\star}, \hat {f} _ {\lambda_ {\star}}\right) \right\} \leq C _ {3} \left(\frac {d _ {\text {e f f}}}{n}\right) ^ {\frac {\alpha \beta}{\alpha \beta + 1}}. \tag {11}
$$

By Jensen's inequality, we have  $d^{\mathrm{eff}} \leq d / \omega^{1 / \alpha}$ . In particular, for global pooling,  $d^{\mathrm{eff}} = 1$  and the bound (11) does not depend on  $d$  at all. Adding average pooling improves by a factor  $\omega^{1 / \alpha}$  the upper bound on the sample complexity for fitting low-frequency functions. Can we confirm this statistical advantage using the predictions for KRR in high dimension? Consider first the case of global pooling:

Theorem 5 (Generalization of KRR with  $H_{\mathsf{GP}}^{\mathsf{CK}}$  in high-dimension (informal)). Let  $f_{\star} \in L^{2}(\mathcal{Q}^{d},\mathrm{CycLoc}_{q})$  and  $h:\mathbb{R}\to \mathbb{R}$  verifying some 'genericity condition'. Then for  $n = q^{\mathsf{s} - 1 + \nu}$  with  $0 < \nu < 1$ , and  $\lambda = O(1)$ , we have  $(\mathsf{P}_{\mathcal{E}_{\leq s,\nu}}$  is defined as in Theorem 2)

$$
\hat {f} _ {\lambda} = \mathrm {P} _ {\mathcal {E} _ {\leq s, \nu}} f _ {\star} + o _ {q} (1). \tag {12}
$$

Hence, global average pooling results in an improvement by a factor  $d$  in statistical efficiency when fitting cyclic local functions, compared to  $H^{\mathsf{CK}}$ . This improvement was already noticed in Mei et al. (2021b); Bietti et al. (2021) but in the case of  $q = d$  (fully connected neural networks).

For  $\omega < d$ , the asymptotic framework Mei et al. (2021a) is more challenging to implement. However, we present in Appendix C.1 a simplified kernel with non-overlapping local pooling which we believe captures the statistical behavior of local pooling. In this case, we show that Theorem 5 holds with  $n = (d / \omega) \cdot q^{s - 1 + \nu}$ , which interpolates between Theorem 2 ( $\omega = 1$ ) and Theorem 5 ( $\omega = d$ ).

Downsampling: Often pooling is associated with a downsampling operation, which subsample one every  $\Delta$  output coordinates. In Appendix A.4, we characterize the eigendecomposition of  $H_{\omega ,\Delta}^{\mathrm{CK}}$  (Proposition 4) and prove for the popular choice  $\omega = \Delta$ , that downsampling does not modify the cyclic invariant subspace  $j = d$  (Proposition 5). More generally, we conjecture and check numerically that downsampling with  $\Delta \leq \omega$  leaves the low-frequency eigenspaces approximately unchanged. In particular, the statistical complexity of learning low-frequency functions is not modified by downsampling operation in the one-layer case (while downsampling is potentially beneficial in further layers).

# 2.4 MULTILAYER CONVOLUTIONAL KERNELS

For completeness, we briefly discuss here some intuitions of multilayer convolutional kernels. The benefit of depth in convolutional kernels has been investigated in Cohen & Shashua (2016b); Mhaskar & Poggio (2016); Scutbon & Harchaoui (2020); Bietti (2021). In particular, Bietti (2021) observed that the top layer operation of a two-layers convolutional kernel can be replaced by a low-degree polynomial without a performance change.

Here, as a concrete example, we consider a two-layers convolutional kernel with  $\omega$ -local pooling on first layer, quadratic kernel and global pooling on the second layer (see Appendix A.5 for details):

$$
H _ {\omega} ^ {2 \mathrm {C K}} (\boldsymbol {x}, \boldsymbol {y}) = \sum_ {\substack {k, k ^ {\prime} \in [ d ] \\ u, u ^ {\prime} \in [ q ]}} \sum_ {\substack {t, t ^ {\prime} \in [ \omega ] \\ r, r ^ {\prime} \in [ \omega ]}} h \left(\frac {\left\langle \boldsymbol {x} _ {(k + u + t)}, \boldsymbol {y} _ {(k ^ {\prime} + u + t ^ {\prime})} \right\rangle}{q}\right) h \left(\frac {\left\langle \boldsymbol {x} _ {(k + u ^ {\prime} + r)}, \boldsymbol {y} _ {(k ^ {\prime} + u ^ {\prime} + r ^ {\prime})} \right\rangle}{q}\right). \tag{13}
$$

While we believe techniques contained in this paper could be used to study kernels of the type (13), we leave it to future work. Here we only comment on the structure of  $H_{\omega}^{2CK}$ : 1) Including a second convolutional layer allows interactions between patches: the RKHS of (13) contains functions of the form  $f(\pmb{x}) = \sum_{|k - l| \leq q} g_{kl}(\pmb{x}_{(k)}, \pmb{x}_{(l)})$ . 2) Pooling on the first layer encourages interactions that do not rely too much on the relative position of the two patches. 3) Pooling on the second layer penalizes functions that depend on the absolute position of the interaction. For more layers and higher degree kernels, one obtain hierarchical interactions of higher-order, with multi-scale absolute and relative local invariances brought by pooling layers.

![](images/2a7b130b14b5504d410ffcd6599ed6c3aa616125e22296a116600b6315ffa18a.jpg)  
Figure 1: Learning low-frequency (left) and high-frequency (right) cubic polynomials over the hypercube  $d = 30$ , using KRR with  $H^{\mathrm{FC}}(\mathrm{FC})$ ,  $H_{\mathrm{GP}}^{\mathrm{FC}}(\mathrm{FC - GP})$ ,  $H^{\mathrm{CK}}(\mathrm{CK})$ ,  $H_{\omega}^{\mathrm{CK}}(\mathrm{CK - LP})$  and  $H_{\mathrm{GP}}^{\mathrm{CK}}(\mathrm{CK - GP})$ , and regularization parameter  $\lambda = 0^{+}$ . We report the average and the standard deviation of the test error over 5 realizations, against the sample size  $n$ .

![](images/5ff1f4fa346d181df384b5ec9121073e23d032b2632c1e110447b27c4eb551c8.jpg)

# 3 NUMERICAL SIMULATIONS

In order to check our theoretical predictions, we perform a simple numerical experiment on simulated data. We take  $\pmb{x} \sim \mathrm{Unif}(\mathcal{D}^d)$  with  $d = 30$ , and consider two target functions:

$$
f _ {\mathrm {L F}, 3} (\boldsymbol {x}) = \frac {1}{\sqrt {d}} \sum_ {i \in [ d ]} x _ {i} x _ {i + 1} x _ {i + 2}, \quad f _ {\mathrm {H F}, 3} (\boldsymbol {x}) = \frac {1}{\sqrt {d}} \sum_ {i \in [ d ]} (- 1) ^ {i} \cdot x _ {i} x _ {i + 1} x _ {i + 2}. \tag {14}
$$

Here  $f_{\mathrm{LF},3}$  is a cyclic-invariant local polynomial ( $f_{\mathrm{LF},3}$  is 'low-frequency'). The function  $f_{\mathrm{HF},3}$  is a high-frequency local polynomial, and is orthogonal to the space of cyclic invariant functions. On these target functions, we compare the test error of kernel ridge regression with 5 different kernels: a standard inner-product kernel  $H^{\mathrm{FC}}(\pmb{x},\pmb{y}) = h(\langle \pmb{x},\pmb{y}\rangle /d)$ ; a cyclic invariant kernel  $H_{\mathsf{GP}}^{\mathsf{FC}}(\pmb{x},\pmb{y})$  (convolutional kernel with global pooling and full-size patches  $q = d$ ); a convolutional kernel  $H^{\mathsf{CK}}$  with patch size  $q = 10$ ; a convolutional kernel with local pooling  $H_{\omega}^{\mathsf{CK}}$  with  $q = 10$  and  $\omega = 5$ ; and a convolutional kernel with global pooling  $H_{\mathsf{GP}}^{\mathsf{CK}}$  with  $q = 10$ . In all these kernels, we choose a common  $h(t) = \sum_{i\in [5]}0.2*t^{i}$  which is a degree 5-polynomial.

In Figure 1, we report the test errors of fitting  $f_{\mathrm{LF},3}$  (left) and  $f_{\mathrm{HF},3}$  (right) using kernel ridge regression with these 5 kernels. We choose a small regularization parameter  $\lambda = 10^{-6}$ , and the noise level  $\sigma_{\varepsilon} = 0$ . The curves are averaged over 5 independent instances and the error bar stands for the standard deviation of these instances. The results match well our theoretical predictions. For the function  $f_{\mathrm{LF},3}$ , the sample sizes required to achieve vanishing test errors are ordered as  $H_{\mathrm{GP}}^{\mathrm{CK}} < H_{\omega}^{\mathrm{CK}} < H_{\mathrm{GP}}^{\mathrm{FC}} < H_{\mathrm{FC}}^{\mathrm{FC}}$  and are around the predicted thresholds  $q^{2} < dq^{2} / \omega < d^{2} < dq^{2} < d^{3}$  respectively. Next we look at the test error of fitting the high frequency local function  $f_{\mathrm{HF},3}$ . The test errors of  $H^{\mathrm{CK}}$  and  $H^{\mathrm{FC}}$  are the same for  $f_{\mathrm{HF},3}$  and  $f_{\mathrm{LF},3}$ : this is because these kernels do not have bias towards either high-frequency or low-frequency functions. The kernel  $H_{\omega}^{\mathrm{CK}}$  performs worse on  $f_{\mathrm{HF},3}$  than on  $f_{\mathrm{LF},3}$ : this is because the eigenspaces of  $H_{\omega}^{\mathrm{CK}}$  are biased towards low-frequency polynomials. The kernels  $H_{\mathrm{GP}}^{\mathrm{CK}}$  and  $H_{\mathrm{GP}}^{\mathrm{FC}}$  do not fit  $f_{\mathrm{HF},3}$  at all (test error greater than or equal to 1): this is because the RKHS of these two kernels only contain cyclic polynomials, but  $f_{\mathrm{HF},3}$  is orthogonal to the space of cyclic polynomials.

# 4 DISCUSSION AND FUTURE WORK

In this paper, we characterized in a stylized setting how convolution, average pooling and downsampling operations modify the RKHS, by restricting it to  $q$ -local functions and then biasing the RKHS towards low-frequency components. We quantified precisely the gain in statistical efficiency of KRR using these operations. Beyond illustrating the 'RKHS engineering' of image-like function classes, these results can further provide intuition and a rigorous foundation for convolution and pooling operations in kernels and CNNs. A natural extension would be to study the multilayer convolutional kernels in details and consider other pooling operations such as max-pooling. Another important question is how anisotropy of the data impacts the results of this paper: in particular, it was shown that pre-processing (whitening of the patches) greatly improves the performance of convolutional kernels Thiry et al. (2021); Bietti (2021). A more challenging question is to study how training and feature learning can further improve the performance of CNNs outside the kernel regime.

# REFERENCES

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. On the convergence rate of training recurrent neural networks. In Advances in Neural Information Processing Systems, pp. 6676-6688. 2019.  
Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. In Advances in Neural Information Processing Systems, pp. 8139-8148, 2019.  
Francis Bach. Learning Theory from First Principles. 2021.  
William Beckner. Inequalities in Fourier analysis. Annals of Mathematics, pp. 159-182, 1975.  
William Beckner. Sobolev inequalities, the Poisson semigroup, and analysis on the sphere  $S^n$ . Proceedings of the National Academy of Sciences, 89(11):4816-4819, 1992.  
Alberto Bietti. Approximation and learning with deep convolutional models: a kernel perspective. arXiv preprint arXiv:2102.10032, 2021.  
Alberto Bietti, Luca Venturi, and Joan Bruna. On the sample complexity of learning with geometric stability. arXiv preprint arXiv:2106.07148, 2021.  
Aline Bonami. Etude des coefficients de Fourier des fonctions de  $L^p (G)$ . In Annales de l'institut Fourier, volume 20, pp. 335-402, 1970.  
Stéphane Boucheron, Olivier Bousquet, and Gábor Lugosi. Theory of classification: A survey of some recent advances. *ESAIM: probability and statistics*, 9:323-375, 2005.  
Y-Lan Boureau, Jean Ponce, and Yann LeCun. A theoretical analysis of feature pooling in visual recognition. In Proceedings of the 27th international conference on machine learning (ICML-10), pp. 111-118, 2010.  
Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature communications, 12(1):1-12, 2021.  
Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7(3):331-368, 2007.  
Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In Advances in Neural Information Processing Systems, pp. 2933-2943, 2019.  
Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In Geoffrey Gordon, David Dunson, and Miroslav Dudík (eds.), Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pp. 215-223, Fort Lauderdale, FL, USA, 11-13 Apr 2011. PMLR.  
Nadav Cohen and Amnon Shashua. Convolutional rectifier networks as generalized tensor decompositions. In International Conference on Machine Learning, pp. 955-963. PMLR, 2016a.  
Nadav Cohen and Amnon Shashua. Inductive bias of deep convolutional networks through pooling geometry. arXiv preprint arXiv:1605.06743, 2016b.  
Hugo Cui, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborova. Generalization error rates in kernel regression: The crossover from the noiseless to noisy regime. arXiv preprint arXiv:2105.15004, 2021.  
Amit Daniely, Roy Frostig, and Yoram Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity. In Advances in Neural Information Processing Systems, pp. 2253-2261, 2016.  
Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 1675-1685. PMLR, 09-15 Jun 2019a.

Simon S Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations, 2019b. URL https://openreview.net/forum?id=S1eK3i09YQ.  
Noureddine El Karoui. The spectrum of kernel random matrices. The Annals of Statistics, 38(1): 1-50, 2010.  
Alessandro Favero, Francesco Cagnetta, and Matthieu Wyart. Locality defeats the curse of dimensionality in convolutional teacher-student scenarios. arXiv preprint arXiv:2106.08619, 2021.  
Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. When do neural networks outperform kernel methods? Advances in Neural Information Processing Systems, 33, 2020.  
Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. The Annals of Statistics, 49(2):1029-1054, 2021.  
Leonard Gross. Logarithmic sobolev inequalities. American Journal of Mathematics, 97(4):1061-1083, 1975.  
Zaid Harchaoui, Francis R Bach, and Eric Moulines. Testing for homogeneity with kernel fisher discriminant analysis. In NIPS, pp. 609-616. CiteSeer, 2007.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, pp. 8580-8589, 2018.  
Arthur Jacot, Berfin Şimşek, Francesco Spadaro, Clément Hongler, and Franck Gabriel. Kernel alignment risk estimator: Risk prediction from training data. arXiv preprint arXiv:2006.09796, 2020.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, pp. 1097-1105, 2012.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, 2015.  
Zhiyuan Li, Ruosong Wang, Dingli Yu, Simon S Du, Wei Hu, Ruslan Salakhutdinov, and Sanjeev Arora. Enhanced convolutional neural tangent kernels. arXiv preprint arXiv:1911.00809, 2019.  
Zhiyuan Li, Yi Zhang, and Sanjeev Arora. Why are convolutional nets more sample-efficient than fully-connected nets? arXiv preprint arXiv:2010.08515, 2020.  
Tengyuan Liang, Alexander Rakhlin, et al. Just interpolate: Kernel "ridgeless" regression can generalize. Annals of Statistics, 48(3):1329-1347, 2020.  
Julien Mairal. End-to-end kernel learning with supervised convolutional kernel networks. arXiv preprint arXiv:1605.06265, 2016.  
Julien Mairal, Piotr Koniusz, Zaid Harchaoui, and Cordelia Schmid. Convolutional kernel networks. arXiv preprint arXiv:1406.3332, 2014.  
Eran Malach and Shai Shalev-Shwartz. Computational separation between convolutional and fully-connected networks. arXiv preprint arXiv:2010.01369, 2020.  
Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random features and kernel methods: hypercontractivity and kernel matrix concentration. arXiv preprint arXiv:2101.10588, 2021a.

Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Learning with invariances in random features and kernel models. arXiv preprint arXiv:2102.13219, 2021b.  
Hrushikesh N Mhaskar and Tomaso Poggio. Deep vs. shallow networks: An approximation theory perspective. Analysis and Applications, 14(06):829-848, 2016.  
Ryan O'Donnell. Analysis of boolean functions. Cambridge University Press, 2014.  
Mark Rudelson and Roman Vershynin. Hanson-wright inequality and sub-gaussian concentration. Electronic Communications in Probability, 18, 2013.  
Meyer Sctbon and Zaid Harchaoui. Harmonic decompositions of convolutional networks. In International Conference on Machine Learning, pp. 8522-8532. PMLR, 2020.  
Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge University Press, 2014.  
Vaishaal Shankar, Alex Fang, Wenshuo Guo, Sara Fridovich-Keil, Jonathan Ragan-Kelley, Ludwig Schmidt, and Benjamin Recht. Neural kernels without tangents. In International Conference on Machine Learning, pp. 8614-8623. PMLR, 2020.  
Louis Thiry, Michael Arbel, Eugene Belilovsky, and Edouard Oyallon. The unreasonable effectiveness of patches in deep convolutional kernels methods. arXiv preprint arXiv:2101.07528, 2021.  
Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge University Press, 2019.  
Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes over-parameterized deep relu networks. arXiv:1811.08888, 2018.
