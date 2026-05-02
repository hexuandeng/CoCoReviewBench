# FILTRA: RETHINKING STEERABLE CNN BY FILTER TRANSFORM

Anonymous authors

Paper under double-blind review

# ABSTRACT

Steerable CNN imposes the prior knowledge of transformation invariance or equivariance in the network architecture to enhance the network robustness on geometry transformation of data and reduce overfitting. Filter transform has been an intuitive and widely used technique to construct steerable CNN in the past decades. Recently, group representation theory is used to analyze steerable CNN and reveals the function space structure of a steerable kernel function. However, it is not yet clear on how this theory is related to the filter transform technique. In this paper, we show that kernel constructed by filter transform can also be interpreted in the group representation theory. Meanwhile, we show that filter transformed kernels can be used to convolve input/output features in different group representation. This interpretation helps complete the puzzle of steerable CNN theory and provides a novel and simple approach to implement steerable convolution operators. Experiments are executed on multiple datasets to verify the feasibility of the proposed approach.

# 1 INTRODUCTION

Beyond the well-known property of equivariance under translation, there has been substantial recent interest in CNN architectures that are equivariant with respect to other transformation groups, e.g. reflection and rotation. Applications of such architectures range over scenarios where object orientation might variate, including OCR, aerial image processing, 3D point cloud processing, medical image processing, texture analysis and etc.

Previous works on constructing equivariant CNN can be coarsely categorized as two aspects. The first aspect designs special steerable filters so that the convolutional output is hard-baked to transform accordingly when the input reflects or rotates. A plenty of works develop this idea by filter rotation, including hand-crafted filters (Oyallon & Mallat, 2015) and learned filters (Laptev et al., 2016; Zhou et al., 2017; Cheng et al., 2018; Marcos et al., 2017). TI-Pooling (Laptev et al., 2016) produces invariant output as input rotates. ORN (Zhou et al., 2017) and RotDCF (Cheng et al., 2018) produces output which circularly shifted as input rotates. Since each dimension of such permutable output corresponds to a uniformly discrete rotation angle, RotEqNet (Marcos et al., 2017) propose to extract rotation angle from the permutable features. Another approach to construct steerable filters is to linearly combine a set of steerable bases. These basis can be solved in discrete function space (Cohen & Welling, 2014; 2016) or continuous function space (Worrall et al., 2017; Weiler & Cesa, 2019). Weiler & Cesa (2019) comprehensively summarize works on steerable bases using polar Fourier basis.

The second aspect exploits specific transforms to act on input. Spatial Transformer Network (STN) is a well-known representative, which predicts an affine matrix to transform its input to the canonical form. Tai et al. (2019) inherits this idea to design equivariant network. Another choice of transform is to the polar coordinate system (Henriques & Vedaldi, 2017; Esteves et al., 2018). Since 2D rotation in Cadestrian coordinate system corresponds to 2D translation in polar coordinate system, rotation equivariance can be achieved by conventional translation equivariant CNN.

The approach proposed in this paper falls into the first category. Weiler & Cesa (2019) proves that all steerable convolutional operator could be denoted as the combination of a specific set of polar Fourier bases. However, it is not clear yet how this interpretation is related with the widely used filter transform scheme. In this paper, we aim to establish the missing connection between the

![](images/3aa6051f3dcb4e4eec94722550e62e42ef42663e7f85e22d703a0d6873a169b3.jpg)  
Figure 1: Illustrations for (8) for  $g = (0,1)$  at left and  $g = (1,1)$  at right. Red, light yellow and green denotes negative, 0 and positive values, respectively.

group representation based analysis for steerable filters and filter transform scheme. To this end, we propose a new approach (FILTRA) to use filter transform to establish steerability between features in different group representation in cyclic group  $\mathrm{C}_N$  and dihedral group  $\mathrm{D}_N$ . We verify the feasibility of FILTRA for the classification and regression tasks on different datasets.

# 2 PRELIMINARIES

We make use of several NumPy or SciPy functions in equations including roll, flipud and circulant. We omit the variable in bracket sometimes by writing  $\kappa_{*}^{*} = \kappa_{*}^{*}(g)$  and  $\mathsf{K}_{*}^{*} = \mathsf{K}_{*}^{*}(\phi)$ .

# 2.1 REFLECTION GROUP, CYCLIC GROUP AND DIHEDRAL GROUP

We consider steerable filters on reflection group  $(\{\pm 1\},*)$ , cyclic group  $C_N$  and dihedral group  $D_N = (\{\pm 1\},*) \ltimes C_N$ . To unify the notations in derivation, we interpret  $C_N = (\{1\},*) \ltimes C_N$  and  $(\{\pm 1\},*) = (\{\pm 1\},*) \ltimes C_1 = D_1$  so that a element in these three groups can always be denoted as a pair  $g = (i_0, i_1)$ , whose range is  $\mathbb{Z}_2 \times \mathbb{Z}_1$  for reflection group,  $\mathbb{Z}_1 \times \mathbb{Z}_N$  for cyclic group and  $\mathbb{Z}_2 \times \mathbb{Z}_N$  for dihedral group. Each element in  $C_N$  corresponds to rotation angle  $\theta_{i_1} = \frac{2i_1\pi}{N}$ .

# 2.2 GROUP REPRESENTATION

A linear representation  $\rho$  of a group  $G$  on a vector space  $\mathbb{R}^n$  is a group homomorphism from  $G$  to the general linear group  $\mathrm{GL}(n)$ , denoted as

$$
\rho : G \mapsto \operatorname {G L} (n) \quad \text {s . t .} \quad \rho \left(g g ^ {\prime}\right) = \rho (g) \rho \left(g ^ {\prime}\right), \quad \forall g, g ^ {\prime} \in G. \tag {1}
$$

We consider three types of linear representation in this paper, i.e. trivial representation, regular representation and irreducible representation (irrep). Readers can refer to Serre (1977) for further background for these concepts.

The trivial representation of a group element is always  $\rho_{\mathrm{tri}}(g) \equiv 1$ . The regular representation of a finite group  $G$  acts on a vector space  $\mathbb{R}^{|G|}$  by permuting its axis. Therefore, for a rotation element  $g = (0, i_1) \in \mathrm{C}_N$  or  $\mathrm{D}_N$ , we get

$$
\rho_ {\operatorname {r e g}} ^ {\mathrm {C} _ {N}} (g) = P \left(i _ {1}\right), \quad \rho_ {\operatorname {r e g}} ^ {\mathrm {D} _ {N}} (g) = \left[ \begin{array}{c c} P \left(i _ {1}\right) & 0 \\ 0 & P \left(i _ {1}\right) \end{array} \right], \text {w h e r e} \quad P \left(i _ {1}\right) = \operatorname {r o l l} \left(\mathbf {I} _ {N}, i _ {1}, 0\right). \tag {2}
$$

For a reflected element  $g = (1,i_1)\in \mathrm{D}_N$  , we get

$$
\rho_ {\text {r e g}} ^ {\mathrm {D} _ {N}} (g) = \left[ \begin{array}{c c} 0 & B (i _ {1}) \\ B (i _ {1}) & 0 \end{array} \right], \text {w h e r e} \quad B (i _ {1}) = f l i p u d (P (- i _ {1} - 1)). \tag {3}
$$

By selecting suitable change of basis of the vector space, a representation can be converted to a equivalent representation, which is the direct sum of several independent representations on the orthogonal subspaces. A representation is called irreducible representation if no non-trivial decomposition exists. This conversion is denoted as

$$
\rho (g) = Q \left[ \bigoplus_ {\left(i _ {0}, i _ {1}\right) \in I} \psi_ {i} (g) \right] Q ^ {- 1}, \tag {4}
$$

where  $I$  is an index set specifying the irreducible representations  $\psi_{i}$  and  $Q$  is the change of basis.

# 2.3 DECOMPOSING REGULAR REPRESENTATION

We decompose the regular representation into a set of irreps. Define the following base irrep

$$
\psi_ {j, k} \left(i _ {0}, i _ {1}\right) = \left\{ \begin{array}{l l} {\left((- 1) ^ {j}\right) ^ {i _ {0}}} & k = 0 \\ {\left(- 1\right) ^ {i _ {1}} \cdot \left((- 1) ^ {j}\right) ^ {i _ {0}}} & k = \frac {N}{2}, N \text {i s e v e n} \\ {\left[ \begin{array}{c c} \cos \left(k \theta_ {i _ {1}}\right) & - \sin \left(k \theta_ {i _ {1}}\right) \\ \sin \left(k \theta_ {i _ {1}}\right) & \cos \left(k \theta_ {i _ {1}}\right) \end{array} \right] \left[ \begin{array}{c c} 1 & 0 \\ 0 & (- 1) ^ {i _ {0}} \end{array} \right] \cdot \left((- 1) ^ {j}\right) ^ {i _ {0}}} & \text {o t h e r w i s e} \end{array} , \right. \tag {5}
$$

where  $j, k$  are referred as the reflection and rotation frequency of the irrep. Concretely, if the action  $g$  reflects/rotates an object once,  $\psi_{j,k}(g)$  will reflects/rotates in vector space  $j / k$  times. We also define the following discrete cosine transform basis

$$
\begin{array}{l} V = \left[ \begin{array}{c c c c} \beta_ {0} & \beta_ {1} & \dots & \beta_ {\lfloor \frac {N}{2} \rfloor} \end{array} \right], \\ \text {w h e r e} \quad \beta_ {k} = \left\{ \begin{array}{l l l l} \mathbf {1} _ {N} & & & k = 0 \\ {\left[ \cos (k \theta_ {0}) \right.} & \cos (k \theta_ {1}) & \dots & \cos (k \theta_ {N - 1}) \end{array} \right] ^ {\top} \quad k = \frac {N}{2}, N \text {i s e v e n}. \tag {6} \\ \end{array}
$$

The following decomposition for  $\rho_{\mathrm{reg}}^{\mathrm{C}_N}(0,i_1)$  holds

$$
\rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (g) = V D ^ {\mathrm {C} _ {N}} V ^ {\top}, \quad D ^ {\mathrm {C} _ {N}} = \bigoplus_ {0 \leq k \leq \lfloor \frac {N}{2} \rfloor} \psi_ {0, k} (0, i _ {1}). \tag {7}
$$

The decomposition for  $\rho_{\mathrm{reg}}^{\mathrm{D}_N}(i_0,i_1)$  holds in a bit more complicated form, i.e.

$$
\rho_ {\mathrm {r e g}} ^ {\mathrm {D} _ {N}} \left(i _ {0}, i _ {1}\right) = W D ^ {\mathrm {D} _ {N}} W ^ {\top},
$$

$$
\text {w h e r e} \quad W = \left[ \begin{array}{c c} V & V \\ V & - V \end{array} \right], \quad D ^ {\mathrm {D} _ {N}} = \bigoplus_ {0 \leq j \leq 1, 0 \leq k \leq \lfloor \frac {N}{2} \rfloor} \psi_ {j, k} \left(i _ {0}, i _ {1}\right), \tag {8}
$$

and each column of  $W$  is referred by  $\beta_{j,k} = \left[\beta_k^\top \quad (-1)^j\beta_k^\top \right]^\top$ . See Fig. 1 for a visualization of this decomposition.

We also mention a property of  $\beta_{k}$  that is easy to verify and will be useful in our derivation.

$$
\psi_ {0, k} \left(0, i _ {1}\right) \beta_ {k} ^ {\top} = \beta_ {k} ^ {\top} P \left(i _ {1}\right), \quad \psi_ {1, k} \left(0, i _ {1}\right) \beta_ {k} ^ {\top} = \beta_ {k} ^ {\top} P \left(i _ {1}\right), \tag {9a}
$$

$$
\psi_ {0, k} (1, i _ {1}) \beta_ {k} ^ {\top} = \beta_ {k} ^ {\top} B (i _ {1}), \quad \psi_ {1, k} (1, i _ {1}) \beta_ {k} ^ {\top} = - \beta_ {k} ^ {\top} B (i _ {1}), \tag {9b}
$$

where  $\psi_{0,k}(i_0,i_1)$  rotates column vectors of  $\beta_k^\top$  as if they are circularly shifted.

# 2.4 STEERABLE CNN AND HARMONIC FILTERS

Weiler et al. (2018) proposes the condition of a filter kernel  $\kappa$  to be equivariant under the action  $g\in G$ .

Lemma 1. The map  $f \mapsto \kappa \cdot f$  is equivariant under  $G$  if and only if for all  $g \in G$ ,

$$
\kappa (g x) = \rho_ {\text {o u t}} (g) \kappa (x) \rho_ {\text {i n}} (g) ^ {- 1}. \tag {10}
$$

Weiler & Cesa (2019) proves that such filters can be denoted by a series of harmonic bases  $b(\phi)$ , i.e.

$$
\kappa (r, \phi) = \sum_ {b \in \mathcal {K}} \omega_ {b} (r) b (\phi), \tag {11}
$$

where  $\omega_{b}(r)$  is the per radial weights. For example, consider  $\rho_{\mathrm{in}} = \psi_{i,m}$  and  $\rho_{\mathrm{out}} = \psi_{j,n}$  in  $\mathrm{D}_N$

$$
\mathcal {K} _ {\psi_ {j, m} \leftarrow \psi_ {i, n}} = \left\{b _ {\mu , \gamma , s} (\phi) = \psi (\mu \phi) \xi (s) \mid \mu = m - s n, s \in \{\pm 1 \} \right\}. \tag {12}
$$

![](images/e5536f3bb378bf27e5cda2f989d3c04d8475ee9b76413d3914b6e5e4ccac5428.jpg)  
Figure 2: Visualization of FILTRA filter examples. Based on a same weight kernel  $\mathsf{K}$ , we generate filters  $\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{C}_N},\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{D}_N},\mathsf{K}_{k\to \mathrm{reg}}^{\mathrm{C}_N}$  and  $\mathsf{K}_{j,k\to \mathrm{reg}}^{\mathrm{D}_N}$ . In this example we set  $j = 1, k = 1, N = 8$ . The two-columns of matrix  $\beta_{k}$  is splitted as  $\beta_{k}^{0}$  and  $\beta_{k}^{1}$  for visualization. Red, light yellow and green denotes negative, 0 and positive values, respectively. Please view this figure in color.

# 3 MAIN RESULTS

(10) and (11) provide a general approach to verify and construct steerable CNN with different representations. In this section, we relate these theories with filter transform and show how to use filter transform to construct steerable filters with input/output of different representations. Please also refer to Fig. 2 for illustration of the filters described as follows.

In our derivation, we mainly consider the angular coordinate of polar coordinate functions  $\kappa(r,\phi)$  and write them  $\kappa(\phi)$ . We will also frequently make use of the following property:

$$
\kappa (\phi - \theta_ {0}) = \kappa (\phi + \theta_ {0}), \quad \kappa (\phi - \theta_ {i}) = \kappa (\phi + \theta_ {N - i}). \tag {13}
$$

# 3.1 FROM TRIVIAL REPRESENTATION TO REGULAR REPRESENTATION

Rotation Group  $C_N$  Consider the rotating filter  $\mathsf{K}$  and its reflected version  $\overline{\mathsf{K}}$  which are commonly used in previous works, e.g. TI-Pooling, ORN, RotEqNet and RotDCF:

$$
\mathrm {K} (\phi) = \left[ \begin{array}{l l l l} \kappa^ {0} & \kappa^ {1} & \dots & \kappa^ {N - 1} \end{array} \right] ^ {\top}, \quad \kappa^ {n} (\phi) = \kappa (\phi - \theta_ {n}), \tag {14}
$$

$$
\overline {{\mathsf {K}}} (\phi) = \left[ \begin{array}{c c c c} \overline {{\kappa}} ^ {0} & \overline {{\kappa}} ^ {1} & \dots & \overline {{\kappa}} ^ {N - 1} \end{array} \right] ^ {\top}, \quad \overline {{\kappa}} ^ {n} (\phi) = \kappa (\theta_ {n} - \phi).
$$

The output of convolution with the above kernels naturally permutes as the input rotates in  $\mathrm{C}_N$ . This intuitively corresponds to property of a steerable filter transforming from trivial representation to regular representation. In this paper, we use  $\mathsf{K}$  and  $\overline{\mathsf{K}}$  as the basic filters to construct different types of steerable filters in  $\mathrm{C}_N$  and  $\mathrm{D}_N$ . We verify the observation of the above steerability by substituting  $\mathsf{K}$  into the lhs of Lemma 1 with  $g = (0,1)$  and write:

$$
\begin{array}{l} \mathsf {K} (\phi + \theta_ {1}) = \left[ \begin{array}{l l l l} \kappa (\phi + \theta_ {1}) & \kappa^ {0} & \dots & \kappa^ {N - 2} \end{array} \right] ^ {\top} = \left[ \begin{array}{l l l l} \kappa^ {N - 1} & \kappa^ {0} & \dots & \kappa^ {N - 2} \end{array} \right] ^ {\top} (15a) \\ = \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (0, 1) \mathrm {K} \rho_ {\text {t r i}} (0, 1) ^ {- 1}. (15b) \\ \end{array}
$$

The above equation can be similarly verified for other  $g = (0,i_1)$  and also on  $\overline{\mathbb{K}}$ . Thus WLOG we select the steerable filter which transforms trivial representation to regular representation on  $\mathrm{C}_N$  as

$$
\boxed {K _ {0 \rightarrow \operatorname {r e g}} ^ {C _ {N}} = K.} \tag {16}
$$

Dihedral Group  $\mathrm{D}_N$  The steerable filter that transforms trivial representation to regular representation on  $\mathrm{D}_N$  can be constructed as

$$
\boxed {K _ {0 \rightarrow \operatorname {r e g}} ^ {\mathrm {D} _ {N}} (\phi) = \left[\begin{array}{l l}K ^ {\top}&\overline {{K}} ^ {\top}\end{array}\right] ^ {\top},} \tag {17}
$$

which corresponds to enumerating each  $\mathrm{D}_N$  element and act on the kernel  $\kappa$ . For  $g = (0, i_1)$ ,  $\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{D}_N}$  can be verified to follow (10) in the same way as (15a), i.e.  $\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{D}_N}(\phi + \theta) = \rho_{\mathrm{reg}}^{\mathrm{D}_N}(g)\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{D}_N}\rho_{\mathrm{tri}}(g)^{-1}$ .

For reflected action, when  $g = (1, 1)$ , we write:

$$
\begin{array}{l} \mathsf {K} (- \phi + \theta_ {1}) = \left[ \kappa (- \phi + \theta_ {1}) \quad \kappa (- \phi - \theta_ {0}) \quad \kappa (- \phi - \theta_ {1}) \quad \dots \quad \kappa (- \phi - \theta_ {N - 2}) \right] ^ {\top} \tag {18} \\ = \left[ \begin{array}{c c c c c} \overline {{\kappa}} ^ {1} & \overline {{\kappa}} ^ {0} & \overline {{\kappa}} ^ {N - 1} & \dots & \overline {{\kappa}} ^ {2} \end{array} \right] ^ {\top} = B (1) \overline {{\mathsf {K}}}. \\ \end{array}
$$

Similarly, we can show for  $g = (1,i_1)$ ,

$$
\overline {{\mathsf {K}}} (- \phi + \theta_ {i _ {1}}) = B (i _ {1}) \mathsf {K}, \quad \mathsf {K} (- \phi + \theta_ {i _ {1}}) = B (i _ {1}) \overline {{\mathsf {K}}}. \tag {19}
$$

Thus we verify (10) for the reflected actions  $g = (1, i_1)$  by summarizing the above as  $\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{D}_N}(-\phi + \theta_{i_1}) = \rho_{\mathrm{reg}}^{\mathrm{D}_N}(g)\mathsf{K}_{0\to \mathrm{reg}}^{\mathrm{D}_N}\rho_{\mathrm{tri}}(g)^{-1}$ .

# 3.2 FROM IRREP TO REGULAR REPRESENTATION

Rotation Group  $C_N$  Consider a  $C_N$  irrep  $\psi_{0,k}(g)$  with frequency  $(0,k)$ . We show that the following kernel

$$
\boxed {K _ {k \rightarrow \operatorname {r e g}} ^ {C _ {N}} = \operatorname {d i a g} (K) \beta_ {k},} \tag {20}
$$

transforms from  $\psi_{0,k}(g)$  to regular representation for the action  $g = (0,i_1)$ :

$$
\begin{array}{l} \mathrm {K} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} (\phi + \theta_ {i _ {1}}) = \operatorname {d i a g} (P (i _ {1}) \mathrm {K}) \beta_ {k}, \quad \text {c . f .} (1 5 \mathrm {b}) \\ = P \left(i _ {1}\right) \operatorname {d i a g} (K) P \left(i _ {1}\right) ^ {- 1} \beta_ {k} = \rho_ {\operatorname {r e g}} ^ {\mathrm {C} _ {N}} (g) \mathrm {K} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} \psi_ {0, k} (g) ^ {- 1}, \quad \text {c . f .} (9 \mathrm {a}). \tag {21} \\ \end{array}
$$

We can also verify this for

$$
\overline {{\mathrm {K}}} _ {k \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} = \operatorname {d i a g} (\overline {{\mathrm {K}}}) \beta_ {k}. \tag {22}
$$

Dihedral Group  $\mathrm{D}_N$  Consider a  $\mathrm{D}_N$  irrep  $\psi_{j,k}(i_0,i_1)$  with frequency  $(j,k)$ . We show that the following kernel:

$$
\boxed {\mathsf {K} _ {j, k \rightarrow \mathrm {r e g}} ^ {\mathrm {D} _ {N}} = \left[\begin{array}{l l}\mathsf {K} _ {k \rightarrow \mathrm {r e g}} ^ {\mathrm {C} _ {N}} ^ {\top}&(- 1) ^ {j} \cdot \overline {{\mathsf {K}}} _ {k \rightarrow \mathrm {r e g}} ^ {\mathrm {C} _ {N}} ^ {\top}\end{array}\right] ^ {\top}} \tag {23}
$$

transforms from  $\psi_{j,k}(i_0,i_1)$  to regular representation for the action  $g = (i_0,i_1)\in \mathrm{D}_N$ . First note it is easy to verify that for  $i_0 = 0$ , i.e.  $g = (0,i_1)$ , the equation holds in the same way as (21),

$$
\mathrm {K} _ {j, k \rightarrow \operatorname {r e g}} ^ {\mathrm {D} _ {N}} (\phi + \theta) = \rho_ {\operatorname {r e g}} ^ {\mathrm {D} _ {N}} (g) \mathrm {K} _ {j, k \rightarrow \operatorname {r e g}} ^ {\mathrm {D} _ {N}} \psi_ {j, k} (g) ^ {- 1}. \tag {24}
$$

We then generalize (18) on  $\mathsf{K}_{k\to \mathrm{reg}}^{\mathrm{C}_N}$  and  $\overline{\mathsf{K}}_{k\to \mathrm{reg}}^{\mathrm{C}_N}$  given a reflected action  $g = (1,i_1)$ :

$$
\begin{array}{l} \mathrm {K} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} (- \phi + \theta_ {i _ {1}}) = \operatorname {d i a g} \left(\mathrm {K} (- \phi + \theta_ {i _ {1}})\right) \beta_ {k} = B (i _ {1}) \operatorname {d i a g} (\overline {{\mathrm {K}}}) B (i _ {1}) ^ {- 1} \beta_ {k}, \quad \text {c . f .} (9 \mathrm {b}) (25a) \\ = B \left(i _ {1}\right) \operatorname {d i a g} (\overline {{\mathrm {K}}}) \beta_ {k} \psi_ {0, k} (g) ^ {- 1} = - B \left(i _ {1}\right) \operatorname {d i a g} \left(\overline {{\mathrm {K}}} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {1, k} (g) ^ {- 1} (25b) \\ = B \left(i _ {1}\right) \operatorname {d i a g} \left(\mathrm {K} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {0, k} (g) ^ {- 1} = - B \left(i _ {1}\right) \operatorname {d i a g} \left(\mathrm {K} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {1, k} (g) ^ {- 1}. (25c) \\ \end{array}
$$

Note that (25b) and (25c) both have two equivalent forms denoted with  $\psi_{0,k}(g)$  and  $\psi_{1,k}(g)$  respectively. Now we can show  $\mathsf{K}_{j,k\to \mathrm{reg}}^{\mathrm{D}_N}$  follows Lemma 1 for  $j = 0$ ,  $i_0 = 1$ , i.e.  $g = (1,i_1)$  as:

$$
\begin{array}{l} \mathsf {K} _ {j, k \rightarrow \operatorname {r e g}} ^ {\mathrm {D} _ {N}} (- \phi + \theta_ {i _ {1}}) = \left[ \mathsf {K} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} (- \phi + \theta_ {i _ {1}}) ^ {\top} \quad \overline {{\mathsf {K}}} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} (- \phi + \theta_ {i _ {1}}) ^ {\top} \right] ^ {\top} (26a) \\ = \left[ B (i _ {1}) \operatorname {d i a g} \left(\overline {{\mathrm {K}}} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {0, k} (g) ^ {- 1} \quad B (i _ {1}) \operatorname {d i a g} \left(\mathrm {K} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {0, k} (g) ^ {- 1} \right] ^ {\top} \quad \text {c . f .} (2 5 b) (26b) \\ = \rho_ {\text {r e g}} ^ {\mathrm {D} _ {N}} (g) \left[ \mathsf {K} _ {k \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} ^ {\top} \quad \overline {{\mathsf {K}}} _ {k \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} ^ {\top} \right] ^ {\top} \psi_ {0, k} (g) ^ {- 1} \quad \text {c . f .} (2 0), (2 2) (26c) \\ = \rho_ {\text {r e g}} ^ {\mathrm {D} _ {N}} (g) \mathrm {K} _ {j, k \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} \psi_ {0, k} (g) ^ {- 1}. (26d) \\ \end{array}
$$

The verification is similar for  $j = 1$ ,  $i_0 = 1$ , i.e.  $g = (1, i_1)$ :

$$
\begin{array}{l} \mathsf {K} _ {j, k \rightarrow \operatorname {r e g}} ^ {\mathrm {D} _ {N}} (- \phi + \theta_ {i _ {1}}) = \left[\begin{array}{l l}\mathsf {K} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} (- \phi + \theta_ {i _ {1}}) ^ {\top}&- \overline {{\mathsf {K}}} _ {k \rightarrow \operatorname {r e g}} ^ {\mathrm {C} _ {N}} (- \phi + \theta_ {i _ {1}}) ^ {\top}\end{array}\right] ^ {\top} (27a) \\ = \left[ - B \left(i _ {1}\right) \operatorname {d i a g} \left(\overline {{\mathsf {K}}} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {1, k} (g) ^ {- 1} \quad B \left(i _ {1}\right) \operatorname {d i a g} \left(\mathsf {K} ^ {\mathrm {C} _ {N}}\right) \beta_ {k} \psi_ {1, k} (g) ^ {- 1} \right] ^ {\top} \quad \text {c . f .} (2 5 \mathrm {b}) (27b) \\ = \rho_ {\text {r e g}} ^ {\mathrm {D} _ {N}} (g) \left[ \mathsf {K} _ {k \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} ^ {\top} - \overline {{\mathsf {K}}} _ {k \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} ^ {\top} \right] ^ {\top} \psi_ {0, k} (g) ^ {- 1} \quad \text {c . f .} (2 0), (2 2) (27c) \\ = \rho_ {\text {r e g}} ^ {\mathrm {D} _ {N}} (g) \mathrm {K} _ {j, k \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} \psi_ {0, k} (g) ^ {- 1}. (27d) \\ \end{array}
$$

# 3.3 FROM REGULAR REPRESENTATION TO REGULAR REPRESENTATION

Regular representation possesses a nice property that it can be averaged, pooled or activated channelwise without violating steerability (Weiler & Cesa, 2019). Thus it is convenient to used regular representation for the intermediate features of a steerable CNN. We show in this subsection that the following kernels can be used to construct a steerable kernel whose input and output features are both in regular representation.

Rotation Group  $C_N$

$$
\boxed {K _ {\mathrm {r e g} \rightarrow \mathrm {r e g}} ^ {\mathrm {C} _ {N}} = \left[ K _ {0 \rightarrow \mathrm {r e g}} ^ {\mathrm {C} _ {N}}, \dots , K _ {\lfloor \frac {N}{2} \rfloor \rightarrow \mathrm {r e g}} ^ {\mathrm {C} _ {N}} \right] V ^ {- 1}.} \tag {28}
$$

This kernel can be verified as follows for  $g = (0,i_1)$ :

$$
\begin{array}{l} \mathrm {K} _ {\text {r e g} \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} (\phi + \theta_ {i _ {1}}) = \left[ \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (g) \mathrm {K} _ {0 \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \psi_ {0, 0} (g) ^ {- 1}, \dots , \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (g) \mathrm {K} _ {\lfloor \frac {N}{2} \rfloor \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \psi_ {0, \lfloor \frac {N}{2} \rfloor} (g) ^ {- 1} \right] V ^ {- 1} (29a) \\ = \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (g) \left[ \mathrm {K} _ {0 \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \dots \mathrm {K} _ {\lfloor \frac {N}{2} \rfloor \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \right] D ^ {\mathrm {C} _ {N}} V ^ {- 1} (29b) \\ = \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (g) \left[ \mathrm {K} _ {0 \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \dots \mathrm {K} _ {\lfloor \frac {N}{2} \rfloor \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \right] V ^ {- 1} V D ^ {\mathrm {C} _ {N}} V ^ {- 1} = \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} (g) \mathrm {K} _ {\text {r e g} \rightarrow \text {r e g}} ^ {\mathrm {C} _ {N}} \rho_ {\text {r e g}} ^ {\mathrm {C} _ {N}} ^ {- 1}. (29c) \\ \end{array}
$$

Dihedral Group  $\mathrm{D}_N$

$$
\boxed {K _ {\text {r e g} \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} = \left[ K _ {0, 0 \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} \dots K _ {0, \lfloor \frac {N}{2} \rfloor \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} \quad K _ {1, 0 \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} \dots K _ {1, \lfloor \frac {N}{2} \rfloor \rightarrow \text {r e g}} ^ {\mathrm {D} _ {N}} \right] W ^ {- 1}.} \tag {30}
$$

This kernel can be verified to transform regular representation to regular representation in the same way as Subsect. 3.3.

# 3.4 REVERSED TRANSFORM OF REPRESENTATIONS

It is obvious to find that for (10), if  $\rho_{\mathrm{in}}, \rho_{\mathrm{out}}$  are orthogonal matrices, i.e.  $\rho_{\mathrm{in}}^{-1} = \rho_{\mathrm{in}}^{\top}, \rho_{\mathrm{out}}^{-1} = \rho_{\mathrm{out}}^{\top}$ , the transpose of (10) naturally proves the equivariance of  $\kappa^{\top}$  under a reversed representation transform direction, i.e. from  $\rho_{\mathrm{out}}$  to  $\rho_{\mathrm{in}}$ . Thus we can easily obtain equivariance kernel from regular representation to trivial/irreducible representation by simply transposing (16), (17), (20) and (23).

# 3.5 CONVENTIONAL ROTATING FILTERS

We comprehensively study the approach to use filter rotation to form steerable convolutional kernels with regular representation features as input or output. Conventional filter rotation based networks exploit some basic forms introduced in this section. TI-Pooling (Laptev et al., 2016) exploits kernel  $\mathsf{K}^{\mathsf{C}_N}$  to transform trivial to regular representation, executes orientation pooling to convert regular to trivial representation and loses orientation information. RotDCF and ORN exploits a kernel of form

$$
\mathrm {K} _ {\mathrm {O R N}} ^ {\mathrm {C} _ {N}} = \text {c i r c u l a n t} (\mathrm {K}). \tag {31}
$$

It is easy to verify that  $K_{\mathrm{ORN}}^{\mathrm{C}_N}$  also follows Lemma 1 to be a steerable filter. However, compared to  $K_{\mathrm{reg}\to \mathrm{reg}}^{\mathrm{C}_N}$ ,  $K_{\mathrm{ORN}}^{\mathrm{C}_N}$  consumes same filter storage but has less weight capacity ( $N$  v.s.  $N\lfloor \frac{N}{2}\rfloor$ ). RotEqNet constructs 2D vector field which could rotate as its input rotates but regards the 2D vector field as independent trivial representation in convolution. As shown in this paper, it preserves better steerability to regards the vector field as irrep representation with frequency 1.

Table 1: Network structure in experiments  

<table><tr><td>layer</td><td>k</td><td>s</td><td>output</td><td>δt (FIL)</td><td>δt (R2)</td></tr><tr><td>conv+relu</td><td>5</td><td>1</td><td>128 (reg)</td><td>0.12</td><td>0.17</td></tr><tr><td>conv+relu</td><td>5</td><td>1</td><td>192 (reg)</td><td>0.13</td><td>0.13</td></tr><tr><td>pool</td><td>3</td><td>2</td><td>256 (reg)</td><td>-</td><td>-</td></tr><tr><td>conv+relu</td><td>5</td><td>1</td><td>256 (reg)</td><td>0.13</td><td>0.13</td></tr><tr><td>conv+relu</td><td>5</td><td>1</td><td>384 (reg)</td><td>0.23</td><td>0.23</td></tr><tr><td>pool</td><td>3</td><td>2</td><td>384 (reg)</td><td>-</td><td>-</td></tr><tr><td>conv+relu</td><td>5</td><td>1</td><td>512 (reg)</td><td>0.32</td><td>0.48</td></tr><tr><td>conv+relu</td><td>5</td><td>1</td><td>768 (reg)</td><td>0.62</td><td>0.91</td></tr><tr><td>pool</td><td>3</td><td>2</td><td>768 (reg)</td><td>-</td><td>-</td></tr></table>

(a) The backbone network structure used in our experiments is composed by convolution, ReLU and pooling layers. The convolution layers are realized by FILTRA, R2Conv and conventional convolution respectively while the rest layers remain the same. Three realizations have the same number of output channels in each layer but organize the channels to be follow regular representation for FILTRA and R2Conv. k: kernel size. s: stride.  $\delta t$ : filter generation time in ms.

<table><tr><td>layer</td><td>k</td><td>s</td><td>output</td></tr><tr><td>GroupPool</td><td>-</td><td>1</td><td>24 (reg)</td></tr><tr><td>fc+relu</td><td>-</td><td>-</td><td>16 (reg)</td></tr><tr><td>fc</td><td>-</td><td>-</td><td>10 (tri)</td></tr></table>

<table><tr><td>layer</td><td>k</td><td>s</td><td>output</td></tr><tr><td>PAMaxPool</td><td>-</td><td>-</td><td>24 (reg)</td></tr><tr><td>conv+relu</td><td>1</td><td>1</td><td>16 (reg)</td></tr><tr><td>conv</td><td>1</td><td>1</td><td>2 (irrep)</td></tr></table>

(b) The classification head network structure used in our experiments uses a Grouping Pooling (Weiler & Cesa, 2019) to generate transform invariant features.  
(c) The regression head network structure used in our experiments uses a PointwiseAdaptiveMaxPool (PAMax-Pool) (Weiler & Cesa, 2019) to summarize feature in regular representation.

# 3.6 NUMERICAL ACCURACY FOR DISCRETE KERNELS

Note that when implementing discrete convolution, the equality of (15a) does not perfectly hold. For example, consider  $\kappa^n (\phi) = \kappa (\phi -\theta_n)$ ,  $\kappa^n (\theta_n) = \kappa (0)$  holds for a continuous  $\kappa$ . However, for discrete  $\kappa$ ,  $\kappa^n (\phi)$  is a rotated interpolation of  $\kappa (\phi)$  and this equity does not precisely hold in general. There exist some exceptions where the equity can be achieved for discrete  $\kappa$ . One example is when  $\kappa^n (\phi)$  is a  $90^{\circ}$  rotation of  $\kappa$  and it can be precisely constructed from  $\kappa$ . Another example is when  $\kappa^n$  is a  $45^{\circ}$  rotation interpolated by nearest pixel from a  $\kappa$  of size  $3\times 3$ .

# 4 EXPERIMENTS

The proposed equivariant convolution, referred as FILTRA, can be interpreted as an alternative formulation for the harmonic based (Weiler & Cesa, 2019) implementation of steerable convolution. In this section we show the pros and cons of each implementation by experiments. We make use of the framework E2CNN (Weiler & Cesa, 2019) for our experiments as it provides the general interface and operations for steerable CNN network. Experiments are executed on the MNIST, KM-NIST (Clanuwat et al., 2018), FashionMNIST (Xiao et al., 2017), EMNIST (Cohen et al., 2017) and CIFAR10 datasets.

We compare FILTRA against two convolution operations, i.e. the representative harmonic based convolution R2Conv and the conventional vanilla convolution. All MNIST-like datasets are experimented on a same feature extraction backbone as described in Table 1a, with convolution operator realized by the three experimented approaches. CIFAR10 is experimented with WideResNet (Zagoruyko & Komodakis, 2016) in the setting similar to Weiler & Cesa (2019). We found that on CIFAR10,  $\mathrm{C_4}$  steerable network performs better than  $\mathrm{C_8}$  for both approaches. For all experiments, we randomly rotate or reflect according to the experiment settings. The settings and evaluation results are listed in Table 2. Different from Weiler et al. (2018), we force the three convolution kernels to output same number of channels. For example, compared to vanilla convolution, the number of free weights for a  $\mathrm{C_8}$  FILTRA is reduced to  $1/8$  and for a  $\mathrm{D_8}$  is reduced to  $1/16$ . The filters for all the approaches will thus have exactly same shape at the deploy stage.

Experiments are executed on GTX 2070. The training procedure of FILTRA and R2Conv can both be implemented as a vanilla convolution plus a filter generation step. For  $C_8$  case the runtime of both generator is similar and for  $D_8$  case FILTRA is slightly faster. We show runtime of  $D_8$  case in Table 1a at training stage. R2Conv additionally requires a initialization of about  $2\mathrm{min}$ . Both of the approaches consume the same inference time as of vanilla convolution.

Table 2: Performance on MNIST and CIFAR10. S: randomly augmented over SO(2). O: randomly augmented over O(2). wrn: WideResNet. Zagoruyko & Komodakis (2016).  

<table><tr><td rowspan="2">Tasks</td><td colspan="8">Classification (acc)</td><td colspan="7">Regression (angle err deg)</td><td></td></tr><tr><td colspan="2">mnist</td><td colspan="2">kmnist</td><td colspan="2">fmnist</td><td colspan="2">emnist</td><td colspan="2">cifar10</td><td colspan="2">mnist</td><td colspan="2">kmnist</td><td colspan="2">fmnist</td></tr><tr><td>Aug</td><td>S</td><td>O</td><td>S</td><td>O</td><td>S</td><td>O</td><td>S</td><td>O</td><td>wrn</td><td>wrn</td><td>S</td><td>O</td><td>S</td><td>O</td><td>S</td><td>O</td></tr><tr><td>Net equiv</td><td>C8</td><td>D8</td><td>C8</td><td>D8</td><td>C8</td><td>D8</td><td>C8</td><td>D8</td><td>C4</td><td>D4</td><td>C8</td><td>D8</td><td>C8</td><td>D8</td><td>C8</td><td>D8</td></tr><tr><td>FILTRA</td><td>98.9</td><td>98.1</td><td>97.1</td><td>97.0</td><td>90.5</td><td>90.8</td><td>77.1</td><td>80.5</td><td>93.4</td><td>92.8</td><td>3.3</td><td>5.4</td><td>3.2</td><td>3.6</td><td>2.6</td><td>2.8</td></tr><tr><td>R2Conv</td><td>98.8</td><td>98.1</td><td>97.3</td><td>96.8</td><td>90.5</td><td>90.8</td><td>76.7</td><td>80.1</td><td>93.6</td><td>92.7</td><td>4.8</td><td>8.9</td><td>3.4</td><td>4.5</td><td>2.9</td><td>3.7</td></tr><tr><td>Conv</td><td>98.5</td><td>98.0</td><td>96.4</td><td>95.2</td><td>89.3</td><td>88.3</td><td>72.6</td><td>80.1</td><td>93.2</td><td>-</td><td>6.6</td><td>10.6</td><td>4.8</td><td>6.4</td><td>3.1</td><td>3.6</td></tr></table>

# 4.1 CLASSIFICATION TASK

The most typical experiment used in previous works on conventional steerable CNN is the classification task. We follow this convention and compare the classification performance of the experimented three approaches in Table 2. FILTRA show comparable performance to R2Conv and slightly improves accuracy for OCR-like (*MNIST) tasks where high frequency texture is limited. On CIFAR10, the performance of FILTRA is minorly disadvantageous. The explanation comes in the interpolation artifacts mentioned in Subsect. 3.6. As the interpolation of high frequency components deviates more, this harms the performance on CIFAR10 with high frequency texture.

# 4.2 REGRESSION TASK

Besides the typical classification task, we find that the property of steerability is naturally advantageous for many regression tasks whose input might rotate or reflect. In this paper, we evaluate the regression performance with an example task to predict the character direction. Similar tasks are commonly used in OCR techniques. When the character rotates, the predicted direction should rotate with the same rotating frequency. This means the predicted 2D direction vector is following a irrep  $\psi_{0,1}$  for  $C_N$ . We reuse the backbone in Table 1a to extract features and use a regression head in Table 1c to predict a unit 2D vector denoting the direction. The network is trained with MSE loss. Note that the images should be masked by a disk to avoid the network to overfit the direction from rotated black boundary. Different approaches are evaluated by the mean included angle between the predicted and groundtruth directions as shown in Table 2. FILTRA with  $C_8$  steerability performs best when trained on data augmented over  $\mathrm{SO}(2)$ . We owe this to the fact that FILTRA weight is naturally organized by the discrete grid layout. Each element of discrete weight matrix contribute to one more DoF of the filters. In contrast, R2Conv uses filters parameterized with a polar coordinate. The DoF of the filters is slightly reduced due to the discretization.

# 5 CONCLUSIONS

In this paper, we establish the connection between the recent steerable CNN structure based on group representation theory and the conventional transformed filters. To this end, we propose an approach to construct steerable convolution filters, which transform between features in trivial, irreducible and regular representations. We verify the feasibility of FILTRA for the classification and regression tasks on several datasets.

# REFERENCES

Xiuyuan Cheng, Qiang Qiu, Robert Calderbank, and Guillermo Sapiro. Rotdcf: Decomposition of convolutional filters for rotation-equivariant deep networks. arXiv preprint arXiv:1805.06846, 2018.  
Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. arXiv preprint arXiv:1812.01718, 2018.  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emmist: Extending mnist to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 2921-2926. IEEE, 2017.

Taco Cohen and Max Welling. Learning the irreducible representations of commutative lie groups. In International Conference on Machine Learning, pp. 1755-1763, 2014.  
Taco S Cohen and Max Welling. Steerable cnns. arXiv preprint arXiv:1612.08498, 2016.  
Carlos Esteves, Christine Allen-Blanchette, Xiaowei Zhou, and Kostas Daniilidis. Polar transformer networks. In International Conference on Learning Representations, 2018.  
Joao F Henriques and Andrea Vedaldi. Warped convolutions: Efficient invariance to spatial transformations. In International Conference on Machine Learning, pp. 1461-1469. PMLR, 2017.  
Dmitry Laptev, Nikolay Savinov, Joachim M Buhmann, and Marc Pollefeys. Ti-pooling: transformation-invariant pooling for feature learning in convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 289-297, 2016.  
Diego Marcos, Michele Volpi, Nikos Komodakis, and Devis Tuia. Rotation equivariant vector field networks. In Proceedings of the IEEE International Conference on Computer Vision, pp. 5048-5057, 2017.  
Edouard Oyallon and Stephane Mallat. Deep roto-translation scattering for object classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2865-2873, 2015.  
Jean-Pierre Serre. Linear representations of finite groups, volume 42. Springer, 1977.  
Kai Sheng Tai, Peter Bailis, and Gregory Valiant. Equivariant transformer networks. In International Conference on Machine Learning (ICML), 2019.  
Maurice Weiler and Gabriele Cesa. General e (2)-equivariant steerable cnns. In Advances in Neural Information Processing Systems, pp. 14334-14345, 2019.  
Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco S Cohen. 3d steerable cnns: Learning rotationally equivariant features in volumetric data. In Advances in Neural Information Processing Systems, pp. 10381-10392, 2018.  
Daniel E Worrall, Stephan J Garbin, Daniyar Turmukhambetov, and Gabriel J Brostow. Harmonic networks: Deep translation and rotation equivariance. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5028-5037, 2017.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Yanzhao Zhou, Qixiang Ye, Qiang Qiu, and Jianbin Jiao. Oriented response networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 519-528, 2017.