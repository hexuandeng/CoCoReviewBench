# SPHERICAL CNNS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Convolutional Neural Networks (CNNs) have become the method of choice for learning problems involving 2D planar images. However, a number of problems of recent interest have created a demand for models that can analyze spherical images. Examples include omnidirectional vision for drones, robots, and autonomous cars, molecular regression problems, and global weather and climate modelling. A naive application of convolutional networks to a planar projection of the spherical signal is destined to fail, because the space-varying distortions introduced by such a projection will make translational weight sharing ineffective.

In this paper we introduce the building blocks for constructing spherical CNNs. We propose a definition for the spherical cross-correlation that is both expressive and rotation-equivariant. The spherical correlation satisfies a generalized Fourier theorem, which allows us to compute it efficiently using a generalized (non-commutative) Fast Fourier Transform (FFT) algorithm. We demonstrate the computational efficiency, numerical accuracy, and effectiveness of spherical CNNs applied to 3D model recognition and atomization energy regression.

# 1 INTRODUCTION

Convolutional networks are able to detect local patterns regardless of their position in the image. Like patterns in a planar image, patterns on the sphere can move around, but in this case the "move" is a 3D rotation instead of a translation. In analogy to the planar CNN, we would like to build a network that can detect patterns regardless of how they are rotated over the sphere.

As shown in Figure 1, there is no good way to use translational convolution or cross-correlation<sup>1</sup> to analyze spherical signals. The most obvious approach, then, is to change the definition of cross-correlation by replacing filter translations by rotations. Doing so, we run into a subtle but important difference between the plane and the sphere: whereas the space of moves for the plane (2D translations) is itself isomorphic to the plane, the space of moves for the sphere (3D rotations) is a different, three-dimensional manifold called  $\mathrm{SO}(3)^2$ . It follows that the result of a spherical correlation (the output feature map) is to be considered a signal on  $\mathrm{SO}(3)$ , not a signal on the sphere,  $S^2$ . For this reason, we deploy  $\mathrm{SO}(3)$  group correlation in the higher layers of a spherical CNN (Cohen and Welling, 2016).

The implementation of a spherical CNN ( $S^2$ -CNN) involves two major challenges. Whereas a square grid of pixels has discrete

![](images/f668d65d0b12d801c8892dbdb4dec157313decfaa0f560d0d4a68470bb2089f6.jpg)  
Figure 1: Any planar projection of a spherical signal will result in distortions. A rotation of a spherical signal cannot be emulated by translation of its planar projection.

translation symmetries, no perfectly symmetrical grids for the sphere exist. This means that there is no simple way to define the rotation of a spherical filter by one pixel. Instead, in order to rotate a filter we would need to perform some kind of interpolation. The other challenge is computational efficiency;  $\mathrm{SO}(3)$  is a three-dimensional manifold, so a naive implementation of  $\mathrm{SO}(3)$  correlation is  $O(n^6)$ .

We address both of these problems using techniques from non-commutative harmonic analysis (Chirikjian and Kyatkin, 2001; Folland, 1995). This field presents us with a far-reaching generalization of the Fourier transform, which is applicable to signals on the sphere as well as the rotation group. It is known that the SO(3) correlation satisfies a Fourier theorem with respect to the SO(3) Fourier transform, and we show that the same is true for our definition of  $S^2$  correlation. Hence, the  $S^2$  and SO(3) correlation can be implemented efficiently using generalized FFT algorithms.

Because we are the first to use cross-correlation on a continuous group inside a multi-layer neural network, we rigorously evaluate the degree to which the mathematical properties predicted by the continuous theory hold in practice for our discretized implementation.

Furthermore, we demonstrate the utility of spherical CNNs for rotation invariant classification and regression problems by experiments on three datasets. First, we evaluate the stability of our cross-correlation under random rotations on a variant of MNIST projected on the sphere. Second, we use the CNN for classifying 3D shapes. In a third experiment we use the model for molecular energy regression.

# CONTRIBUTIONS

The main contributions of this work are the following:

1. The theory of spherical CNNs. We show these networks are equivariant to 3D rotations of the input.  
2. The first automatically differentiable implementation of the generalized Fourier transform for  $S^2$  and SO(3). Our PyTorch implementation is easy to use, fast, and memory efficient.  
3. The first empirical support for the utility of spherical CNNs for rotation-invariant learning problems.

# 2 RELATED WORK

It is well understood that the power of CNNs stems in large part from their ability to exploit (translational) symmetries though a combination of weight sharing and translation equivariance. It thus becomes natural to consider generalizations that exploit larger groups of symmetries, and indeed this has been the subject of several recent papers by Gens and Domingos (2014); Olah (2014); Dieleman et al. (2015; 2016); Cohen and Welling (2016); Ravanbakhsh et al. (2017); Zaheer et al. (2017b); Guttenberg et al. (2016); Cohen and Welling (2017). With the exception of SO(2)-steerable networks (Worrall et al., 2017; Weiler et al., 2017), these networks are all limited to discrete groups, such as discrete rotations acting on planar images or permutations acting on point clouds. Other very recent work is concerned with the analysis of spherical images, but does not define an equivariant architecture (Su and Grauman, 2017; Boomsma and Frellsen, 2017). Our work is the first to achieve equivariance to a continuous, non-commutative group  $(\mathrm{SO}(3))$ , and the first to use the generalized Fourier transform for fast group correlation.

To efficiently perform cross-correlations on the sphere and rotation group, we use generalized FFT algorithms. Generalized Fourier Analysis, sometimes called abstract- or noncommutative harmonic analysis, has a long history in mathematics and many books have been written on the subject (Sugiura, 1990; Taylor, 1986; Folland, 1995). For a good engineering-oriented treatment which covers generalized FFT algorithms, see (Chirikjian and Kyatkin, 2001). Other important works include (Driscoll and Healy, 1994; Healy et al., 2003; Potts et al., 1998; Kunis and Potts, 2003; Drake et al., 2008; Maslen, 1998; Rockmore, 2004; Kostelec and Rockmore, 2007; 2008; Potts et al., 2009; Makadia et al., 2007; Gutman et al., 2008).

# 3 CORRELATION ON THE SPHERE AND ROTATION GROUP

We will explain the  $S^2$  and  $\mathrm{SO}(3)$  correlation by analogy to the classical planar  $\mathbb{Z}^2$  correlation. The planar correlation can be understood as follows:

The value of the output feature map at translation  $x \in \mathbb{Z}^2$  is computed as an inner product between the input feature map and a filter, shifted by  $x$ .

Similarly, the spherical correlation can be understood as follows:

The value of the output feature map evaluated at rotation  $R \in \mathrm{SO}(3)$  is computed as an inner product between the input feature map and a filter, rotated by  $R$ .

Because the output feature map is indexed by a rotation, it is modelled as a function on  $\mathrm{SO}(3)$ . We will discuss this issue in more detail shortly.

The above definition refers to various concepts that we have not yet defined mathematically. In what follows, we will go through the required concepts one by one and provide a precise definition. Our goal for this section is only to present a mathematical model of spherical CNNs. Generalized Fourier theory and implementation details will be treated later.

The Unit Sphere  $S^2$  can be defined as the set of points  $x \in \mathbb{R}^3$  with norm 1. It is a two-dimensional manifold, which can be parameterized by spherical coordinates  $\alpha \in [0, 2\pi]$  and  $\beta \in [0, \pi]$ .

Spherical Signals We model spherical images and filters as continuous functions  $f: S^2 \to \mathbb{R}^K$ , where  $K$  is the number of channels.

Rotations The set of rotations in three dimensions is called SO(3), the "special orthogonal group". Rotations can be represented by  $3 \times 3$  matrices that preserve distance (i.e.  $||Rx|| = ||x||$ ) and orientation  $(\operatorname{det}(R) = +1)$ . If we represent points on the sphere as 3D unit vectors  $x$ , we can perform a rotation using the matrix-vector product  $Rx$ . The rotation group SO(3) is a three-dimensional manifold, and can be parameterized by ZYZ-Euler angles  $\alpha \in [0,2\pi]$ ,  $\beta \in [0,\pi]$ , and  $\gamma \in [0,2\pi]$ .

Rotation of Spherical Signals In order to define the spherical correlation, we need to know not only how to rotate points  $x \in S^2$  but also how to rotate filters (i.e. functions) on the sphere. To this end, we introduce the rotation operator  $L_{R}$  that takes a function  $f$  and produces a rotated function  $L_{R}f$  by composing  $f$  with the rotation  $R^{-1}$ :

$$
\left[ L _ {R} f \right] (x) = f \left(R ^ {- 1} x\right). \tag {1}
$$

Due to the inverse on  $R$ , we have  $L_{RR'} = L_R L_{R'}$ .

Inner products The inner product on the vector space of spherical signals is defined as:

$$
\langle \psi , f \rangle = \int_ {S ^ {2}} \sum_ {k = 1} ^ {K} \psi_ {k} (x) f _ {k} (x) d x, \tag {2}
$$

The integration measure  $dx$  denotes the standard rotation invariant integration measure on the sphere, which can be expressed as  $d\alpha \sin(\beta)d\beta / 4\pi$  in spherical coordinates (see Appendix A). The invariance of the measure ensures that  $\int_{S^2} f(Rx)dx = \int_{S^2} f(x)dx$ , for any rotation  $R \in \mathrm{SO}(3)$ . That is, the volume under a spherical heightmap does not change when rotated. Using this fact, we can show that  $L_{R^{-1}}$  is adjoint to  $L_R$ , which implies that  $L_R$  is unitary:

$$
\begin{array}{l} \langle L _ {R} \psi , f \rangle = \int_ {S ^ {2}} \sum_ {k = 1} ^ {K} \psi_ {k} (R ^ {- 1} x) f _ {k} (x) d x \\ = \int_ {S ^ {2}} \sum_ {k = 1} ^ {K} \psi_ {k} (x) f _ {k} (R x) d x \tag {3} \\ = \langle \psi , L _ {R ^ {- 1}} f \rangle . \\ \end{array}
$$

Spherical Correlation With these ingredients in place, we are now ready to state mathematically what was stated in words before. For spherical signals  $f$  and  $\psi$ , we define the correlation as:

$$
[ \psi \star f ] (R) = \langle L _ {R} \psi , f \rangle = \int_ {S ^ {2}} \sum_ {k = 1} ^ {K} \psi_ {k} \left(R ^ {- 1} x\right) f _ {k} (x) d x. \tag {4}
$$

As mentioned before, the output of the spherical correlation is a function on  $\mathrm{SO}(3)$ . This is perhaps somewhat counterintuitive, and indeed the conventional definition of spherical convolution gives as output a function on the sphere. However, as shown in Appendix B, the conventional definition effectively restricts the filter to be circularly symmetric about the  $Z$  axis, which would greatly limit the expressive capacity of the network.

Rotation of SO(3) Signals We defined the rotation operator  $L_{R}$  for spherical signals (eq. 1), and used it to define spherical cross-correlation (eq. 4). To define the SO(3) correlation, we need to generalize the rotation operator so that it can act on signals defined on SO(3). As we will show, naively reusing eq. 1 is the way to go. That is, for  $f: \mathrm{SO}(3) \to \mathbb{R}^{K}$ , and  $R, Q \in \mathrm{SO}(3)$ :

$$
\left[ L _ {R} f \right] (Q) = f \left(R ^ {- 1} Q\right). \tag {5}
$$

Note that while the argument  $R^{-1}x$  in Eq. 1 denotes the rotation of  $x \in S^2$  by  $R^{-1} \in \mathrm{SO}(3)$ , the analogous term  $R^{-1}Q$  in Eq. 5 denotes to the composition of rotations.

Rotation Group Correlation Using the same analogy as before, we can define the correlation of two signals on the rotation group,  $f, \psi : \mathrm{SO}(3) \to \mathbb{R}^K$ , as follows:

$$
[ \psi \star f ] (R) = \langle L _ {R} \psi , f \rangle = \int_ {\mathrm {S O} (3)} \sum_ {k = 1} ^ {K} \psi_ {k} \left(R ^ {- 1} Q\right) f _ {k} (Q) d Q. \tag {6}
$$

The integration measure  $dQ$  is the invariant measure on  $\mathrm{SO}(3)$ , which may be expressed in ZYZ-Euler angles as  $d\alpha \sin (\beta)d\beta d\gamma /(8\pi)^2$  (see Appendix A).

Equivalence As we have seen, correlation is defined in terms the rotation operator  $L_{R}$ . This operator acts naturally on the input space of the network, but what justification do we have for using it in the second layer and beyond?

The justification is provided by an important property, shared by all kinds of convolution and correlation, called equivariance. A layer  $\Phi$  is equivariant if  $\Phi \circ L_R = T_R \circ \Phi$ , for some operator  $T_R$ . Using the definition of correlation and the unitarity of  $L_R$ , showing equivariance is a one liner:

$$
[ \psi \star [ L _ {Q} f ] ] (R) = \langle L _ {R} \psi , L _ {Q} f \rangle = \langle L _ {Q ^ {- 1} R} \psi , f \rangle = [ \psi \star f ] (Q ^ {- 1} R) = [ L _ {Q} [ \psi \star f ] ] (R). \tag {7}
$$

The derivation is valid for spherical correlation as well as rotation group correlation.

# 4 FAST SPHERICAL CORRELATION WITH G-FFT

It is well known that correlations and convolutions can be computed efficiently using the Fast Fourier Transform (FFT). This is a result of the Fourier theorem, which states that  $\widehat{f*\psi} = \hat{f}\cdot \hat{\psi}$ . Since the FFT can be computed in  $O(n\log n)$  time and the product has linear complexity, implementing the correlation using FFTs is asymptotically faster than the naive  $O(n^{2})$  spatial implementation.

For functions on the sphere and rotation group, there is an analogous transform, which we will refer to as the generalized Fourier transform (GFT) and a corresponding fast algorithm (GFFT). This transform finds it roots in the representation theory of groups, but due to space constraints we will not go into details here and instead refer the interested reader to Sugiura (1990) and Folland (1995).

Conceptually, the GFT is nothing more than the linear projection of a function onto a set of orthogonal basis functions called "matrix element of irreducible unitary representations". For the circle  $(S^1)$  or line  $(\mathbb{R})$ , these are the familiar complex exponentials  $\exp (in\theta)$ . For SO(3), we have the Wigner D-functions  $D_{mn}^{l}(R)$  indexed by  $l\geq 0$  and  $-l\leq m,n\leq l$ . For  $S^2$ , these are the spherical harmonics  $Y_{m}^{l}(x)$  indexed by  $l\geq 0$  and  $-l\leq m\leq l$ .

Denoting the manifold  $(S^2$  or  $\mathrm{SO}(3))$  by  $X$  and the corresponding basis functions by  $U^{l}$  (which is either vector-valued  $(Y^{l})$  or matrix-valued  $(D^{l}))$ , we can write the GFT of a function  $f:X\to \mathbb{R}$  as

$$
\hat {f} ^ {l} = \int_ {X} f (x) \overline {{U ^ {l} (x)}} d x. \tag {8}
$$

This integral can be computed efficiently using a GFFT algorithm (see Section 4.1).

The inverse SO(3) Fourier transform is defined as:

$$
f (R) = \sum_ {l = 0} ^ {b} (2 l + 1) \sum_ {m = - l} ^ {l} \sum_ {n = - l} ^ {l} \hat {f} _ {m n} ^ {l} U _ {m n} ^ {l} (R), \tag {9}
$$

and similarly for  $S^2$ . The maximum frequency  $b$  is known as the bandwidth, and is related to the resolution of the spatial grid (Kostelec and Rockmore, 2007).

Using the well-known (in fact, defining) property of the Wigner D-functions that  $D^{l}(R)D^{l}(R^{\prime}) = D^{l}(RR^{\prime})$  and  $D^{l}(R^{-1}) = D^{l}(R)^{\dagger}$ , it can be shown (see Appendix D) that the SO(3) correlation satisfies a Fourier theorem:  $\widehat{\psi \star f} = \hat{f} \cdot \hat{\psi}^{\dagger}$ , where  $\cdot$  denotes matrix multiplication of the two block matrices  $\hat{f}$  and  $\hat{\psi}^{\dagger}$ .

Similarly, using  $Y(Rx) = D(R)Y(x)$  and  $Y_{m}^{l} = D_{m0}^{l}|_{S^{2}}$ , one can derive an analogous  $S^2$  convolution theorem:  $\widehat{\psi \star f}^{l} = \hat{f}^{l}\cdot \hat{\psi}^{l\dagger}$ , where  $\hat{f}^l$  and  $\hat{\psi}^l$  are now vectors. This says that the SO(3)-FT of the  $S^2$  correlation of two spherical signals can be computed by taking the outer product of the  $S^2$ -FTs of the signals. This is shown in figure 2.

![](images/11e9c510a6652ad294afbca29f4f390ecb84724d8119b0d60ae821a7f2ca5833.jpg)  
Figure 2: Spherical correlation in the spectrum. The signal  $f$  and the locally-supported filter  $\psi$  are Fourier transformed, block-wise tensored, summed over input channels, and finally inverse transformed. Note that because the filter is locally supported, it is faster to use a matrix multiplication (DFT) than an FFT algorithm for it. We parameterize the sphere using spherical coordinates  $\alpha, \beta$ , and SO(3) with ZYZ-Euler angles  $\alpha, \beta, \gamma$ .

# 4.1 IMPLEMENTATION OF G-FFT AND SPECTRAL G-CONV

Here we sketch the implementation of GFFTs. For details, see (Kostelec and Rockmore, 2007).

The input of the SO(3) FFT is a spatial signal  $f$  on SO(3), sampled on a discrete grid and stored as a 3D array. The axes correspond to the ZYZ-Euler angles  $\alpha, \beta, \gamma$ . The first step of the SO(3)-FFT is to perform a standard 2D translational FFT over the  $\alpha$  and  $\gamma$  axes. The FFT'ed axes correspond to the  $m, n$  axes of the result. The second and last step is a linear contraction of the  $\beta$  axis of the FFT'ed array with a precomputed array of samples from the Wigner-d functions  $d_{mn}^{l}(\beta)$ . Because the shape of  $d^{l}$  depends on  $l$  (it is  $(2l + 1) \times (2l + 1)$ ), this linear contraction is implemented as a custom GPU kernel. The output is a set of Fourier coefficients  $\hat{f}_{mn}^{l}$  for  $l \geq n, m \geq -l$  and  $l = 0, \dots, L_{\max}$ .

The algorithm for the  $S^2$ -FFTs is very similar, only in this case we FFT over the  $\alpha$  axis only, and do a linear contraction with precomputed Legendre functions over the  $\beta$  axis.

Our code is available at TODO:URL.

# 5 EXPERIMENTS

In a first sequence of experiments, we evaluate the numerical stability and accuracy of our algorithm. In a second sequence of experiments, we showcase that the new cross-correlation layers we have

introduced are indeed useful building blocks for several real problems involving spherical signals. Our examples for this are recognition of 3D shapes and predicting the atomization energy of molecules.

# 5.1 EQUIVARIANCE ERROR

In this paper we have presented the first instance of a group equivariant CNN for a continuous, non-commutative group. In the discrete case, one can prove that the network is exactly equivariant, but although we can prove  $[L_Rf]*\psi = L_R[f*\psi]$  for continuous functions  $f$  and  $\psi$  on the sphere or rotation group, this is not exactly true for the discretized version that we actually compute. Hence, it is reasonable to ask if there are any significant discretization artifacts and whether they affect the equivariance properties of the network. If equivariance can not be maintained for many layers, one may expect the weight sharing scheme to become much less effective.

We first tested the equivariance of a single SO(3) correlation at various resolutions  $b$ . We do this by first sampling  $n = 500$  random rotations  $R_{i}$  as well as  $n$  feature maps  $f_{i}$  with  $K = 10$  channels. Then we compute  $\Delta = \frac{1}{n}\sum_{i = 1}^{n}\mathrm{std}(L_{R_i}\Phi (f_i) - \Phi (L_{R_i}f_i)) / \mathrm{std}(\Phi (f_i))$ , where  $\Phi$  is a composition of SO(3) correlation layers with randomly initialized filters. In case of perfect equivariance, we expect this quantity to be zero. The results

(figure 3 (top)), show that although the approximation error  $\Delta$  grows with the resolution and the number of layers, it stays manageable for the range of resolutions of interest.

![](images/16c691dbbcff95b6ee5a407894f5857ecb3f30d9f98ee1064e9cba13d3ebabba.jpg)

![](images/c11ff91e1187c6ccc350c6ae09a063706958fa0ca1ec7fc99c850f879b79326d.jpg)

![](images/37cc0dfbe5ca6f3a642164b7dec3cfbeb4e8aadfa3f99769d6ff3994835d3033.jpg)  
Figure 3:  $\Delta$  as a function of the resolution and the number of layers.

![](images/f5e7c0ee1e53b0de65c3774b37955414d17a1a92b9d0cdda617292b4457d2d3c.jpg)

We repeat the experiment with ReLU activation function after each correlation operation. As shown in figure 3 (bottom), the error is higher but stays flat. This indicates that the error is not due to the network layers, but due to the feature map rotation, which is exact only for bandlimited functions.

# 5.2 ROTATED MNIST ON THE SPHERE

In this experiment we evaluate the generalization performance with respect to rotations of the input. For testing we propose a version MNIST dataset projected on the sphere (see fig. 4). The dataset is well understood for planar CNNs. Thus it seems well suited to compare planar CNNs with our  $S^2$  CNN. We created two instances of this dataset: one in which each digit is projected on the northern hemisphere and one in which each projected digit is additionally randomly rotated.

Architecture and Hyperparameters As a baseline model, we use a simple CNN with layers conv-ReLU-conv-ReLU-FC-softmax, with filters of size  $5 \times 5$ ,  $k = 57,114,10$  channels, and stride 3 in both layers. We compare to a spherical CNN with layers  $S^2$  conv-ReLUSO(3)conv-ReLU-FC-softmax, bandwidth  $b = 30,10,5$  and  $k = 100,200,10$  channels. Both models have about 165K parameters.

![](images/b015c6618f19199db8e30077d5de67a685ab225964da01f78eba1080fc3c7f53.jpg)  
Figure 4: Two SO(3)-rotated MNIST digits (4/9) on  $S^2$  using stereographic projection onto a Driscoll-Healy grid. Mapping back to the plane results into non-linear distortions.

![](images/0d9de56e3a4b26871161df9a5bba9a19a85d1a3cfce49e5cd1abb15b4354d896.jpg)

![](images/144424d7e6c101ae2de026b81028d01dde39d93078b3bb88496265e67882c233.jpg)

![](images/43ff6ff6bb9f4ba8e4cfa6fa728ef31aa5c7529ce2b462d63ddafe604fddd86a.jpg)

Results We trained each model on the nonrotated (NR) and the rotated (R) training set and

evaluated it on the non-rotated and rotated test set. See table 1. While the planar CNN achieves

![](images/903219ba9b597a0878a5c3d0347366dfa39fc2e06fae489b62190acb8d80c153.jpg)  
Figure 5: The ray line is cast from the surface of the sphere in direction of its center. The first intersection with the model gives the values of the signal on the sphere. The two images of the right represent two spherical signals in  $(\alpha, \beta)$  representation. They contain respectively the distance from the sphere and the cosinus of the ray with the normal of the model. The red dot corresponds to the pixel set by the red line.

![](images/e88944496ed82c3fc31469cd29140ed956d88828b1c9f63e889f63fdba9e1404.jpg)

high accuracy in the NR / NR regime, its performance in the R / R regime is much worse, while the spherical CNN is unaffected. When trained on the non-rotated dataset and evaluated on the rotated dataset (NR / R), the planar CNN does no better than random chance. The spherical CNN shows a slight decrease in performance compared to  $R / R$ , but still performs quite well.

<table><tr><td></td><td>NR / NR</td><td>R / R</td><td>NR / R</td></tr><tr><td>planar</td><td>0.99</td><td>0.45</td><td>0.09</td></tr><tr><td>spherical</td><td>0.91</td><td>0.91</td><td>0.85</td></tr></table>

Table 1: Test accuracy for the networks evaluated on the spherical MNIST dataset. Here  $\mathrm{R} =$  rotated,  $\mathrm{{NR}} =$  non-rotated and  $\mathrm{X}/\mathrm{Y}$  denotes,that the network was trained on  $\mathrm{X}$  and evaluated on  $\mathrm{Y}$  .

# 5.3 RECOGNITION OF 3D SHAPES

Next, we applied  $S^2$  CNN to 3D shape classification. The SHREC17 task (Savva et al., 2017) contains 51300 3D models taken from the ShapeNet dataset (Chang et al., 2015) which have to be classified into 55 common categories (tables, airplanes, persons, etc.). There is a consistently aligned regular dataset and a version in which all models are randomly perturbed by rotations. We concentrate on the latter to test the quality of our rotation equivariant representations learned by  $S^2$  CNN.

Representation We project the 3D meshes onto an enclosing sphere using a straightforward raycasting scheme (see Fig. 5). For each point on the sphere we send a ray towards the origin and collect 3 types of information from the intersection: ray length and  $\cos / \sin$  of the surface angle. We further augment this information with raycasting information for the convex hull of the model, which in total gives us 6 channels for the signal. This signal is discretized using a Driscoll-Healy grid (Driscoll and Healy, 1994) with bandwidth  $b = 128$ . Ignoring non-convexity of surfaces we assume this projection captures enough information of the shape to be useful for the recognition task.

Architecture and Hyperparameters Our network consists of an initial  $S^2$  conv-BN-ReLU block followed by two SO(3) conv-BN-ReLU blocks. The resulting filters are pooled using a max pooling layer followed by a last batch normalization and then fed into a linear layer for the final classification. It is important to note that the max pooling happens over the group SO(3): if  $f_{k}$  is the  $k$ -th filter in the final layer (a function on SO(3)) the result of the pooling is  $\max_{x\in SO(3)}f_k(x)$ . We used 50, 70, and 350 features for the  $S^2$  and the two SO(3) layers, respectively. Further, in each layer we reduce the resolution  $b$ , from 128, 32, 22 to 7 in the final layer. Each filter kernel  $\psi$  on SO(3) has non-local support, where  $\psi (\alpha ,\beta ,\gamma)\neq 0$  iff  $\beta = \frac{\pi}{2}$  and  $\gamma = 0$  and the number of points of the discretization is

<table><tr><td>Method</td><td>P@N</td><td>R@N</td><td>F1@N</td><td>mAP</td><td>NDCG</td></tr><tr><td>Tatsuma_ReVGG</td><td>0.705</td><td>0.769</td><td>0.719</td><td>0.696</td><td>0.783</td></tr><tr><td>Furuya_DLAN</td><td>0.814</td><td>0.683</td><td>0.706</td><td>0.656</td><td>0.754</td></tr><tr><td>SHREC16-Bai_GIFT</td><td>0.678</td><td>0.667</td><td>0.661</td><td>0.607</td><td>0.735</td></tr><tr><td>Deng_CM-VGG5-6DB</td><td>0.412</td><td>0.706</td><td>0.472</td><td>0.524</td><td>0.624</td></tr><tr><td>Ours</td><td>0.701 (3rd)</td><td>0.711 (2nd)</td><td>0.699 (3rd)</td><td>0.676 (2nd)</td><td>0.756 (2nd)</td></tr></table>

Table 2: Results and best competing methods for the SHREC17 competition.

proportional to the bandwidth in each layer. The final network contains  $\approx 1.4\mathrm{M}$  parameters, takes 8GB of memory at batchsize 16, and takes 50 hours to train.

Results We evaluated our trained model using the official metrics and compared to the top three competitors in each category (see table 2 for results). Except for precision and F1@N, in which our model ranks third, it is the runner up on each other metric. The main competitors, Tatsuma_ReVGG and Furuya_DLAN use input representations and network architectures that are highly specialized to the SHREC17 task. Given the rather task agnostic architecture of our model and the lossy input representation we use, we interpret our models performance as strong empirical support for the effectiveness of Spherical CNNs.

# 5.4 PREDICTION OF ATOMIZATION ENERGIES FROM MOLECULAR GEOMETRY

Finally, we apply  $S^2$  CNN on molecular energy regression. In the QM7 task (Blum and Reymond, 2009; Rupp et al., 2012) the atomization energy of molecules has to be predicted from geometry and charges. Molecules contain up to  $N = 23$  atoms of  $T = 5$  types (H, C, N, O, S). They are given as a list of positions  $p_i$  and charges  $z_i$  for each atom  $i$ .

Representation by Coulomb matrices Rupp et al. (2012) propose a rotation and translation invariant representation of molecules by defining the Coulomb matrix  $C \in \mathbb{R}^{N \times N}$  (CM). For each pair of atoms  $i \neq j$  they set  $C_{ij} = (z_i z_j) / (|p_i - p_j|)$  and  $C_{ii} = 0.5z_i^{2.4}$ . Diagonal elements encode the atomic energy by nuclear charge, while other elements encode Coulomb repulsion between atoms. This representation is not permutation invariant. To this end Rupp et al. (2012) propose a distance measure between Coulomb matrices used within Gaussian kernels whereas Montavon et al. (2012) propose sorting  $C$  or random sampling index permutations.

Representation as a spherical signal We utilize spherical symmetries in the geometry by defining a sphere  $S_{i}$  around around  $p_i$  for each atom  $i$ . The radius is kept uniform across atoms and molecules and chosen minimal such that no intersections among spheres in the training set happen. Generalizing the Coulomb matrix approach we define for each possible  $z$  and for each point  $x$  on  $S_{i}$  potential functions  $U_{z}(x) = \sum_{j\neq i,z_{j} = z}\frac{z_{i}\cdot z}{|x - p_{i}|}$  producing a  $T$  channel spherical signal for each atom in the molecule (see figure 6). This representation is invariant with respect to translations and equivariant with respect to rotations. However, it is still not permutation invariant. The signal is discretized using a Driscoll-Healy (Driscoll and Healy, 1994) grid with bandwidth  $b = 10$  representing the molecule as a sparse  $N\times T\times 2b\times 2b$  tensor.

Architecture and Hyperparameters We use a deep ResNet style  $S^2$  CNN. Each ResNet block is made of  $S^2 / \mathrm{SO}(3)$  conv-BN-ReLU-SO(3) conv-BN after which the input is added to the result. We share weights among atoms making filter permutation invariant, by pushing the atom dimension into the batch dimesion. In each layer we downsample the bandwidth, while increasing the number of features  $F$ . After integrating the signal over SO(3) each molecule becomes a  $N \times F$  tensor. For permutation invariance over atoms we follow Zaheer et al. (2017a) and embed each resulting feature vector of an atom into a latent space using a MLP  $\phi$ . Then we sum these latent representations over the atom dimension and get our final regression value for the molecule by mapping with another MLP  $\psi$ . Both  $\phi$  and  $\psi$  are jointly optimized. Training a simple MLP only on the 5 frequencies of atom types in a molecule already gives a RMSE of  $\sim 19$ . Thus, we train the  $S^2$  CNN on the residual only, which improved convergence speed and stability over direct training. The final architecture is

![](images/5b154d62184e03f5e361e3c71b99cdf5e21f48d7c078f9b90826fa2c67c643e3.jpg)

![](images/0ebb251f40daafcd18dd952209720437a297c1cc593be054006cc955a951bbd0.jpg)

![](images/04c4ece15cd0ee7b742090de315a5e9da036af7a5bc45414ab3a89d4f23eeb38.jpg)

![](images/60530b96cb74f7e02f770b4fb9e2c8ad39a45ebce79753fd4f9e66e803ef2722.jpg)  
Figure 6: The five potential channels  $U_{z}$  with  $z \in \{1, 6, 7, 8, 16\}$  for a molecule containing atoms H (red), C (green), N (orange), O (brown), S (gray).

![](images/62c843f7bf3729ba2d7839b298621cc23945989e26e14aea33b3eff9b72ce3c4.jpg)

![](images/ff4fafab26764a8c6f9d3adfa116b655787ed0d487be4372602364b3ce4b88e6.jpg)

<table><tr><td>Method</td><td>Author</td><td>RMSE</td><td>S2CNN</td><td>Layer</td><td>Bandwidth</td><td>Features</td></tr><tr><td>MLP / random CM</td><td>(a)</td><td>5.96</td><td></td><td>Input</td><td></td><td>5</td></tr><tr><td>LGIKA(RF)</td><td>(b)</td><td>10.82</td><td></td><td>ResBlock</td><td>10</td><td>20</td></tr><tr><td>RBF kernels / random CM</td><td>(a)</td><td>11.40</td><td></td><td>ResBlock</td><td>8</td><td>40</td></tr><tr><td>RBF kernels / sorted CM</td><td>(a)</td><td>12.59</td><td></td><td>ResBlock</td><td>6</td><td>60</td></tr><tr><td>MLP / sorted CM</td><td>(a)</td><td>16.06</td><td></td><td>ResBlock</td><td>4</td><td>80</td></tr><tr><td></td><td></td><td>8.47</td><td></td><td>ResBlock</td><td>2</td><td>160</td></tr><tr><td></td><td></td><td></td><td>DeepSet</td><td>Layer</td><td>Input/Hidden</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>φ (MLP)</td><td>160/150</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>ψ (MLP)</td><td>100/50</td><td></td></tr></table>

Table 3: Left: Experiment results for the QM7 task: (a) Montavon et al. (2012) (b) Raj et al. (2016). Right: ResNet architecture for the molecule task.

sketched in table 3. It has about 1.4M parameters, consumes 7GB of memory at batchsize 20, and takes 3 hours to train.

Results We evaluate by RMSE and compare our results to Montavon et al. (2012) and Raj et al. (2016) (see table 3). Our learned representation outperforms all kernel-based approaches and a MLP trained on sorted Coulomb matrices. Superior performance could only be achieved for an MLP trained on randomly permuted Coulomb matrices. However, sufficient sampling of random permutations grows exponentially with  $N$ , so this method is unlikely to scale to large molecules.

# 6 DISCUSSION & CONCLUSION

In this paper we have presented the theory of Spherical CNNs and evaluated them on two important learning problems. We have defined  $S^2$  and  $\mathrm{SO}(3)$  cross-correlations, analyzed their properties, and implemented a Generalized FFT-based correlation algorithm. Our numerical results confirm the stability and accuracy of this algorithm, even for deep networks. Furthermore, we have shown that Spherical CNNs can effectively generalize across rotations, and achieve near state-of-the-art results on competitive 3D Model Recognition and Molecular Energy Regression challenges, without excessive feature engineering and task-tuning.

For intrinsically volumetric tasks like 3D model recognition, we believe that further improvements can be attained by generalizing further beyond SO(3) to the roto-translation group SE(3). The development of Spherical CNNs is an important first step in this direction. Another interesting generalization is the development of a Steerable CNN for the sphere (Cohen and Welling, 2017), which would make it possible to analyze vector fields such as global wind directions, as well as other sections of vector bundles over the sphere.

Perhaps the most exciting future application of the Spherical CNN is in omnidirectional vision. Although very little omnidirectional image data is currently available in public repositories, the

increasing prevalence of omnidirectional sensors in drones, robots, and autonomous cars makes this a very compelling application of our work.

# REFERENCES

L. C. Blum and J.-L. Reymond. 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. J. Am. Chem. Soc., 131:8732, 2009.  
W. Boomsma and J. Frellsen. Spherical convolutions and their application in molecular modelling. In I Guyon, U V Luxburg, S Bengio, H Wallach, R Fergus, S Vishwanathan, and R Garnett, editors, Advances in Neural Information Processing Systems 30, pages 3436-3446. Curran Associates, Inc., 2017.  
A.X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.  
G.S. Chirikjian and A.B. Kyatkin. Engineering Applications of Noncommutative Harmonic Analysis. CRC Press, 1 edition, may 2001. ISBN 9781420041767.  
T.S. Cohen and M. Welling. Group equivariant convolutional networks. In Proceedings of The 33rd International Conference on Machine Learning (ICML), volume 48, pages 2990-2999, 2016.  
T.S. Cohen and M. Welling. Steerable CNNs. In ICLR, 2017.  
S. Dieleman, K. W. Willett, and J. Dambre. Rotation-invariant convolutional neural networks for galaxy morphology prediction. Monthly Notices of the Royal Astronomical Society, 450(2), 2015.  
S. Dieleman, J. De Fauw, and K. Kavukcuoglu. Exploiting Cyclic Symmetry in Convolutional Neural Networks. In International Conference on Machine Learning (ICML), 2016.  
J.B. Drake, P.H. Worley, and E.F. D'Azevedo. Algorithm 888: Spherical harmonic transform algorithms. ACM Trans. Math. Softw., 35(3):23:1-23:23, 2008. doi: 10.1145/1391989.1404581.  
J.R. Driscoll and D.M. Healy. Computing Fourier transforms and convolutions on the 2-sphere. Advances in applied mathematics, 1994.  
G.B. Folland. A Course in Abstract Harmonic Analysis. CRC Press, 1995.  
R. Gens and P. Domingos. Deep Symmetry Networks. In Advances in Neural Information Processing Systems (NIPS), 2014.  
B. Gutman, Y. Wang, T. Chan, P.M. Thompson, and others. Shape registration with spherical cross correlation. 2nd MICCAI workshop, 2008.  
N. Guttenberg, N. Virgo, O. Witkowski, H. Aoki, and R. Kanai. Permutation-equivariant neural networks applied to dynamics prediction. 2016.  
D. Healy, D. Rockmore, P. Kostelec, and S. Moore. FFTs for the 2-Sphere - Improvements and Variations. The journal of Fourier analysis and applications, 9(4):340-385, 2003.  
P.J. Kostelec and D.N. Rockmore. SOFT: SO(3) Fourier Transforms. 2007. URL http://www.cs.dartmouth.edu/~geelong/soft/soft20_fx.pdf.  
P.J. Kostelec and D.N. Rockmore. FFTs on the rotation group. Journal of Fourier Analysis and Applications, 14(2):145-179, 2008.  
S. Kunis and D. Potts. Fast spherical Fourier algorithms. Journal of Computational and Applied Mathematics, 161:75-98, 2003.  
A. Makadia, C. Geyer, and K. Daniilidis. Correspondence-free structure from motion. Int. J. Comput. Vis., 75(3):311-327, December 2007.  
D.K. Maslen. Efficient Computation of Fourier Transforms on Compact Groups. Journal of Fourier Analysis and Applications, 4(1), 1998.

G. Montavon, K. Hansen, S. Fazli, M. Rupp, F. Biegler, A. Ziehe, A. Tkatchenko, O.A. von Lilienfeld, and K. Müller. Learning invariant representations of molecules for atomization energy prediction. In P. Bartlett, F.C.N. Pereira, C.J.C. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 449-457. 2012.  
L. Nachbin. The Haar Integral. 1965.  
C. Olah. Groups and Group Convolutions, 2014. URL https://colah.github.io/posts/2014-12-Groups-Convolution/.  
D. Potts, G. Steidl, and M. Tasche. Fast and stable algorithms for discrete spherical Fourier transforms. Linear Algebra and its Applications, 275:433-450, 1998.  
D. Potts, J. Prestin, and A. Vollrath. A fast algorithm for nonequispaced Fourier transforms on the rotation group. Numerical Algorithms, pages 1-28, 2009.  
A. Raj, A. Kumar, Y. Mroueh, P.T. Fletcher, et al. Local group invariant representations via orbit embeddings. arXiv preprint arXiv:1612.01988, 2016.  
S. Ravanbakhsh, J. Schneider, and B. Poczos. Deep learning with sets and point clouds. In International Conference on Learning Representations (ICLR) - workshop track, 2017.  
D.N. Rockmore. Recent Progress and Applications in Group FFTs. NATO Science Series II: Mathematics, Physics and Chemistry, 136:227-254, 2004.  
M. Rupp, A. Tkatchenko, K.-R. Müller, and O. A. von Lilienfeld. Fast and accurate modeling of molecular atomization energies with machine learning. Physical Review Letters, 108:058301, 2012.  
M. Savva, F. Yu, H. Su, A. Kanezaki, T. Furuya, R. Ohbuchi, Z. Zhou, R. Yu, S. Bai, X. Bai, M. Aono, A. Tatsuma, S. Thermos, A. Axenopoulos, G. Th. Papadopoulos, P. Daras, X. Deng, Z. Lian, B. Li, H. Johan, Y. Lu, and S. Mk. Large-Scale 3D Shape Retrieval from ShapeNet Core55. In Ioannis Pratikakis, Florent Dupont, and Maks Ovsjanikov, editors, Eurographics Workshop on 3D Object Retrieval. The Eurographics Association, 2017. ISBN 978-3-03868-030-7. doi: 10.2312/3dor.20171050.  
Y.C. Su and K. Grauman. Learning spherical convolution for fast features from 360 imagery. Adv. Neural Inf. Process. Syst., 2017.  
M. Sugiura. Unitary Representations and Harmonic Analysis. John Wiley & Sons, New York, London, Sydney, Toronto, 2nd edition, 1990.  
M.E. Taylor. Noncommutative Harmonic Analysis. American Mathematical Society, 1986. ISBN 0821815237.  
M. Weiler, F.A. Hamprecht, and M. Storath. Learning steerable filters for rotation equivariant CNNs. 2017.  
D.E. Worrall, S.J. Garbin, D. Turmukhambetov, and G.J. Brostow. Harmonic networks: Deep translation and rotation equivariance. In CVPR, 2017.  
M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola. Deep sets. arXiv preprint arXiv:1703.06114, 2017a.  
M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R.R. Salakhutdinov, and A.J. Smola. Deep sets. In Advances in Neural Information Processing Systems 30, pages 3393-3403, 2017b.
