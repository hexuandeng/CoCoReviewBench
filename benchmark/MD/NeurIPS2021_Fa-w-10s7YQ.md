# Sparse Steerable Convolutions: An Efficient Learning of SE(3)-Equivariant Features for Estimation and Tracking of Object Poses in 3D Space

Anonymous Author(s)

Affiliation

Address

email

# Abstract

As a basic component of SE(3)-equivariant deep feature learning, steerable convolution has recently demonstrated its advantages for 3D semantic analysis. The advantages are, however, brought by expensive computations on dense, volumetric data, which prevent its practical use for efficient processing of 3D data that are inherently sparse. In this paper, we propose a novel design of Sparse Steerable Convolution (SS-Conv) to address the shortcoming; SS-Conv greatly accelerates steerable convolution with sparse tensors, while strictly preserving the property of SE(3)-equivariance. Based on SS-Conv, we propose a general pipeline for precise estimation of object poses, wherein a key design is a Feature-Steering module that takes the full advantage of SE(3)-equivariance and is able to conduct an efficient pose refinement. To verify our designs, we conduct thorough experiments on three tasks of 3D object semantic analysis, including instance-level 6D pose estimation, category-level 6D pose and size estimation, and category-level 6D pose tracking. Our proposed pipeline based on SS-Conv outperforms existing methods on almost all the metrics evaluated by the three tasks. Ablation studies also show the superiority of our SS-Conv over alternative convolutions in terms of both accuracy and efficiency. We will make the implementation of SS-Conv publicly available.

# 1 Introduction

SE(3)-equivariant deep networks [19, 24, 7] have shown the promise recently in some tasks of 3D semantic analysis, among which 3D Steerable CNN [24] is a representative one. 3D Steerable CNNs employ steerable convolutions (termed as  $ST-Conv$ ) to learn pose-equivariant features in a layer-wise manner, thus preserving the pose information of the 3D input. Intuitively speaking, for a layer of ST-Conv, any SE(3) transformation  $(r, t)$  applied to its 3D input would induce a pose-synchronized transformation to its output features, where  $r \in \mathrm{SO}(3)$  stands for a rotation and  $t \in \mathbb{R}^3$  for a translation. Fig. 1 gives an illustration where given an SE(3) transformation of the input, the locations at which feature vectors are defined are rigidly transformed with respect to  $(r, t)$ , and the feature vectors themselves are also rotated by  $\rho(r)$  ( $\rho(r)$  is an representation of rotation  $r$ ). This property of SE(3)-equivariance enables the steerability of feature space. For example, without transforming the input, SE(3) transformation can be directly realized by steering in the feature space. To produce steerable features, ST-Conv confines its feature domain to regular grids of 3D volumetric data; it can thus be conveniently supported by 3D convolution routines. This compatibility with 3D convolutions eases the implementation of ST-Conv, but at the sacrifice of efficiently processing 3D data (e.g., point clouds) that are typically irregular and sparse; consequently, ST-Conv is still less widely used in broader areas of 3D semantic analysis.

![](images/5d1fc2601a9022eef4f307d3ac33df0ec2726119f238b3aee25736bfcf04741a.jpg)  
(a)

![](images/3c79a53a3ede2baf7498896fa64f3e2ad4e388b1b10f3cb8a7d25c932e91d86e.jpg)  
Figure 1: An illustration of SE(3)-equivariance achieved by (a) Steerable Convolution (ST-Conv), and (b) our Sparse Steerable Convolution (SS-Conv), where arrows defined on the 3D fields denote vector-formed, oriented features. Best view in the electronic version.  
(b)

In this paper, we propose a novel design of Sparse Steerable Convolution (SS-Conv) to address the aforementioned shortcoming faced by ST-Conv. SS-Conv can greatly accelerate steerable convolutions with sparse tensors, while strictly preserving the SE(3)-equivariance in feature learning; Fig. 1(b) gives the illustration. To implement SS-Conv, we construct convolutional kernels as linear combinations of spherical harmonics, which satisfy the rotation-steering constraint of SE(3)-equivariant convolutions [24], and implement the convolution as matrix-matrix multiply-add operations on GPUs only at active grids, which are recorded along with their features as sparse tensors.

Although SE(3)-equivariant feature learning is widely used in 3D object recognition, its potentials for other tasks of 3D semantic analysis have not been well explored yet. In this work, we make the attempt to apply our proposed SS-Conv to object pose estimation in 3D space. To this end, we propose a general pipeline based on SS-Conv, which stacks layers of SS-Conv as the backbone, and decodes object poses directly from the learned SE(3)-equivariant features. A novel Feature-Steering module is also designed into the pipeline to support iterative pose refinement, by taking advantage of the steerability of the learned features. We conduct thorough experiments on three tasks of pose-related, 3D object semantic analysis, including instance-level 6D pose estimation, category-level 6D pose and size estimation, and category-level 6D pose tracking. Our proposed pipeline based on SS-Conv outperforms existing methods on almost all the metrics evaluated by the three tasks; the gaps are clearer in the regimes of high-precision pose estimation. Ablation studies also show the superiority of our SS-Conv over alternative convolutions in terms of both accuracy and efficiency. We will make the implementation of SS-Conv publicly available.

# 2 Related Works

SE(3)-Equivariant Representation Learning SE(3)-equivariance is an important property in 3D computer vision. In earlier works, researchers ease the problem by focusing on SO(3)-equivariance, and design Spherical CNNs [6, 5] by stacking SO(3)-equivariant spherical convolutions which are implemented in the spherical harmonic domain. Recently, a series of works [19, 24, 7] build deep SE(3)-equivariant network based on steerable kernels, which are parameterized as linear combinations of basis kernels. Thomas et al. firstly propose Tensor Field Network (TFN) [19] to learn SE(3)-equivariant features on irregular point clouds, and later, Fuchs et al. present SE(3)-Transformer, which extends TFN with attention mechanism. However, those networks working on point clouds are required to compute kernels with respect to different input points inefficiently. To tackle this problem, steerable convolution [24] is proposed to work on regular volumetric data, so that basis kernels with respect to regular grids could be pre-computed; however, it still encounters challenging computational demands due to the ignorance of data sparsity. Compared to the above methods, our proposed Sparse Steerable Convolution aims at efficient SE(3)-equivariant representation learning for volumetric data, which is realized with sparse tensors to accelerate the computation.

Estimation and Tracking of Object Poses in 3D Space In the context of pose estimation, instance-level 6D pose estimation is a classical and well-developed task, for which a body of works are proposed. These works can be broadly categorized into three types: i) template matching [12] by constructing templates to search for the best matched poses; ii) 2D-3D correspondence methods [1, 14, 15, 18, 16], which establish 2D-3D correspondence via 2D keypoint detection [18, 16] or dense 3D coordinate predictions [1, 14, 15], followed by a PnP algorithm to obtain the target pose;

iii) direct pose regression [25, 13, 22] via deep networks. Recently, a more challenging task of category-level 6D pose and size estimation is formally introduced in [23], aiming at estimating poses of 3D unknown objects with respect to a categorical normalized object coordinate space (NOCS). The early work [23, 20] focuses on regression of NOCS maps, and the poses can be obtained by aligning them with the observed depth maps. Later, methods of direct pose regression are proposed based on fusion of pose-dependent and pose-independent features [2], or on decoupled rotation mechanism [4]. Motivated by [23], Wang et. al propose the task of category-level 6D pose tracking, aiming for the small change of object poses between two adjacent frames in a sequence. In the paper, they also present 6-PACK, a tracker estimating the change of poses by matching keypoints of two frames.

# 3 Sparse Steerable Convolutional Networks

# 3.1 Background

3D Convolution A conventional 3D convolution can be formulated as follows:

$$
f _ {n + 1} (\boldsymbol {x}) = [ \kappa \star f _ {n} ] (\boldsymbol {x}) = \int_ {\mathbb {R} ^ {3}} \kappa (\boldsymbol {x} - \boldsymbol {y}) f _ {n} (\boldsymbol {y}) d \boldsymbol {y}, \tag {1}
$$

where  $f_{n}(\pmb {x})\in \mathbb{R}^{K_{n}}$ $f_{n + 1}(\pmb {x})\in \mathbb{R}^{K_{n + 1}}$  , and  $\kappa :\mathbb{R}^3\to \mathbb{R}^{K_{n + 1}\times K_n}$  is a continuous learnable kernel.

SE(3)-Equivariance Given a transformation  $\pi_n(\pmb{g}) : \mathbb{R}^{K_n} \to \mathbb{R}^{K_n}$  for a 3D rigid motion  $\pmb{g} \in \mathrm{SE}(3)$ , a 3D convolution in Eq. (1) is SE(3)-equivariant if there exists a transformation  $\pi_{n+1}(\pmb{g}) : \mathbb{R}^{K_{n+1}} \to \mathbb{R}^{K_{n+1}}$  such that

$$
[ \pi_ {n + 1} (\boldsymbol {g}) f _ {n + 1} ] (\boldsymbol {x}) = [ \kappa \star [ \pi_ {n} (\boldsymbol {g}) f _ {n} ] ] (\boldsymbol {x}). \tag {2}
$$

Such a SE(3)-equivariant convolution is the so-called steerable convolution (ST-Conv) [24], since the features can be steered by  $\pi_{n + 1}(\pmb {g})$  in the output feature space.

In general, the transformation  $\pi_n(\pmb{g})$  is a group representation of SE(3), which satisfies  $\pi_n(\pmb{g}_1\pmb{g}_2) = \pi_n(\pmb{g}_1)\pi_n(\pmb{g}_2)$ . If  $\pmb{g}$  is decomposed into a 3D rotation  $\pmb{r} \in \mathrm{SO}(3)$  and a 3D translation  $\pmb{t} \in \mathbb{R}^3$ , written as  $\pmb{g} = \pmb{tr}$ ,  $\pi_n(\pmb{g})$  can be defined as follows:

$$
[ \pi_ {n} (\boldsymbol {g}) f _ {n} ] (\boldsymbol {x}) = [ \pi_ {n} (\boldsymbol {t r}) f _ {n} ] (\boldsymbol {x}) := \rho_ {n} (\boldsymbol {r}) f _ {n} (\boldsymbol {r} ^ {- 1} (\boldsymbol {x} - \boldsymbol {t})), \tag {3}
$$

where  $\rho_n(\boldsymbol{r}) : \mathbb{R}^{K_n} \to \mathbb{R}^{K_n}$  is a SO(3) representation.

Rotation-Steering Constraint To guarantee SE(3)-equivariance in Eq. (2), it can be derived that the kernel  $\kappa$  of 3D convolution must be rotation-steerable [24], which satisfies the following constraint:

$$
\kappa (\boldsymbol {r} \boldsymbol {x}) = \rho_ {n + 1} (\boldsymbol {r}) \kappa (\boldsymbol {x}) \rho_ {n} (\boldsymbol {r}) ^ {- 1}. \tag {4}
$$

Irreducible Feature  $\rho_{n}(\boldsymbol{r})$  is a SO(3) representation, which can be decomposed into  $F_{n}$  irreducible representations as follows:

$$
\rho_ {n} (\boldsymbol {r}) = \boldsymbol {Q} ^ {T} \left[ \bigoplus_ {i = 0} ^ {F _ {n}} D ^ {l _ {i}} (\boldsymbol {r}) \right] \boldsymbol {Q}, \tag {5}
$$

where  $Q$  is a  $K_{n} \times K_{n}$  change-of-basis matrix,  $D^{l_i}(\boldsymbol{r})$  is the  $(2l_i + 1) \times (2l_i + 1)$  irreducible Wigner-D matrix [8] of order  $l_i$  ( $l_i = 0, 1, 2, \ldots$ ), and  $\bigoplus$  represents block-diagonal construction of  $\{D^{l_i}(\boldsymbol{r})\}$ , so that  $K_n = \sum_{i=0}^{F_n} 2l_i + 1$ . Based on Eq. (3),  $f_n(\boldsymbol{x})$  can be constructed by stacking  $F_n$  irreducible features  $\{f_n^i(\boldsymbol{x}) \in \mathbb{R}^{2l_i + 1}\}$ ; each  $f_n^i(\boldsymbol{x})$  is associated with a  $D^{l_i}(\boldsymbol{r})$ . When  $l_i = 0$ ,  $D^0(\boldsymbol{r}) = 1$ , so that  $f_n^i(\boldsymbol{x})$  is a scalar, invariant to any rotation; when  $l_i > 0$ ,  $f_n^i(\boldsymbol{x})$  is a vector which can be rotated by  $D^{l_i}(\boldsymbol{r})$ .

# 3.2 Sparse Steerable Convolution

The ST-Conv introduced above enjoys the property of SE(3)-equivariance. However, as discussed in Sec. 1, it suffers from heavy computations as conventional 3D convolution does. Motivated by recent success of sparse convolution (SP-Conv) [9], we propose a novel design of Sparse Steerable Convolution (SS-Conv) with sparse tensors, which takes the natural sparsity of 3D data into account while strictly keeping the steerability of features.

Specifically, assuming  $\kappa$  is a discretized,  $s \times s \times s$ , cubic kernel with grid sites  $S = \{-\frac{s - 1}{2}, \dots, -1, 0, 1, \dots, \frac{s - 1}{2}\}^3$  ( $s$  is odd), our proposed SS-Conv can be formulated as follows:

$$
f _ {n + 1} (\boldsymbol {x}) = \left[ \kappa \star f _ {n} \right] (\boldsymbol {x}) = \left\{ \begin{array}{l l} \sum_ {\boldsymbol {x} - \boldsymbol {y} \in S, \sigma_ {n} (\boldsymbol {y}) = 1} \kappa (\boldsymbol {x} - \boldsymbol {y}) f _ {n} (\boldsymbol {y}), & \text {i f} \sigma_ {n + 1} (\boldsymbol {x}) = 1 \\ \mathbf {0}, & \text {i f} \sigma_ {n + 1} (\boldsymbol {x}) = 0 \end{array} \right. \tag {6}
$$

$$
s. t. \quad \forall \boldsymbol {r} \in S O (3), \kappa (\boldsymbol {r x}) = \rho_ {n + 1} (\boldsymbol {r}) \kappa (\boldsymbol {x}) \rho_ {n} (\boldsymbol {r}) ^ {- 1},
$$

where  $\sigma_n(\pmb{x})$  represents the state of site  $\pmb{x}$  in the feature space  $\mathbb{R}^{K_n}$ .  $\sigma_n(\pmb{x}) = 0$  denotes an inactive state at  $\pmb{x}$ , where  $f_{n}(\pmb{x})$  is in its ground state; when  $f_{n}(\pmb{x})$  is beyond the ground state, this site would be activated as  $\sigma_n(\pmb{x}) = 1$ . In SS-Conv, we set the ground state as a zero vector.

Compared with ST-Conv, our sparse version is accelerated in two ways: i) convolutions are conducted at activated output sites, not on the whole 3D volume, where the number of active sites only takes a small proportion; ii) in the receptive field of each activated output site, only input features at sites with active state are convolved. We introduce the detailed implementation of our SS-Conv as follows.

# 3.2.1 Rotation-Steering kernel Construction

The key to satisfy the rotation-steering constraint (4) is to control the angular directions of feature vectors, and recent research shows that the spherical harmonics  $Y^{J} = \{Y_{j}^{J}\}_{j = -J}^{J}$  give the unique and complete solution [24]. Linear combination of the basis kernels based on spherical harmonics produces the rotation-steering convolutional kernel  $\kappa$ .

For simplicity, we firstly consider both input and output features as individual irreducible ones of orders  $l$  and  $k$ , respectively, and the kernel  $\kappa^{kl}$  is a linear combination of basis kernels  $\kappa^{kl,Jm}$ :

$$
\kappa^ {k l} (\boldsymbol {x}) = \sum_ {J = | k - l |} ^ {k + l} \sum_ {m} w ^ {k l, J m} \kappa^ {k l, J m} (\boldsymbol {x}), \tag {7}
$$

where

$$
\kappa^ {k l, J m} (\boldsymbol {x}) = \sum_ {j = - J} ^ {J} \varphi^ {m} (\| \boldsymbol {x} \|) Y _ {j} ^ {J} \left(\frac {\boldsymbol {x}}{\| \boldsymbol {x} \|}\right) \boldsymbol {Q} _ {j} ^ {k l}. \tag {8}
$$

$Q_{j}^{kl}$  is a  $(2k + 1)\times (2l + 1)$  change-of-basis matrix, also known as Clebsch-Gordan coefficients, and  $\varphi^m$  is a continuous Gaussian radial function:  $\varphi^m (\| \pmb {x}\|) = e^{-\frac{1}{2} (\| \pmb {x}\| -m)^2 /\epsilon^2}$ . In the basis kernel  $\kappa^{kl,Jm}$  (8),  $Y^{J}$  controls the angular direction, while  $\varphi^m$  controls the radial direction; then  $\{\kappa^{kl,Jm}\}$  are linearly combined by learnable coefficients  $\{w^{kl,Jm}\}$  as in Eq. (7), to further adjust the radial direction, the only degree of freedom in the process of optimization. Accordingly, the angular direction is totally controlled by  $Y^{J}$ , such that the kernel constraint is strictly followed. In addition, the total number of learnable parameters in Eq. (7) is  $[2min(k,l) + 1]m$ , which is, in practice, marginally less than that of conventional 3D convolution, which has  $(2k + 1)(2l + 1)$  parameters.

Finally, assuming that the input and output features are stacked irreducible features, whose orders are  $\{l_1,\dots ,l_{F_n}\}$  and  $\{k_{1},\dots ,k_{F_{n + 1}}\}$  respectively, the whole rotation-steering kernel can be constructed as follows:

$$
\kappa (\boldsymbol {x}) = \left[ \begin{array}{c c c} \kappa^ {k _ {1} l _ {1}} (\boldsymbol {x}) & \dots & \kappa^ {k _ {1} l _ {F _ {n}}} (\boldsymbol {x}) \\ \vdots & \ddots & \vdots \\ \kappa^ {k _ {F _ {n + 1}} l _ {1}} (\boldsymbol {x}) & \dots & \kappa^ {k _ {F _ {n + 1}} l _ {F _ {n}}} (\boldsymbol {x}) \end{array} \right]. \tag {9}
$$

# 3.2.2 Site State Definition

The key enabling the efficiency of SS-Conv lies in the definition of active sites. In general, for a grid site  $x$ , if any of input sites in its receptive field are active, this site will be activated, and convolution at this site will be conducted; otherwise, this site will keep inactive, meaning that its feature will be directly set as a zero vector (representing the ground state) without convolutional operation. We formulate the above definition of site state at  $x$  as follows:

$$
\sigma_ {n + 1} (\boldsymbol {x}) = \left\{ \begin{array}{l l} 1, & \text {i f} \sum_ {\boldsymbol {x} - \boldsymbol {y} \in S} \sigma_ {n} (\boldsymbol {y}) > 0 \\ 0. & \text {i f} \sum_ {\boldsymbol {x} - \boldsymbol {y} \in S} \sigma_ {n} (\boldsymbol {y}) = 0 \end{array} \right. \tag {10}
$$

The number of active sites defined in this way will increase layer-by-layer, enabling long-range message transfer. However, when stacking dozens or even hundreds of convolutions, the rapid growth rate of active sites would result in heavy computational burden and the so-called "submanifold dilation problem" [9]. To alleviate this problem, we follow [9] and consider another choice of state definition in our SS-Conv, which keeps the output state consistent with the input one at a same grid site, i.e.,  $\sigma_{n + 1}(\pmb {x}) = \sigma_n(\pmb {x})$ . This kind of SS-Conv without dilation makes it possible to construct a deep but efficient network for sparse volumetric data, and we term it as "Submanifold SS-Conv". In practice, we mix general SS-Convs and Submanifold SS-Convs in an alternating manner to achieve high performance and efficiency.

# 3.2.3 Sparse Convolutional Operation

SS-Conv is achieved with sparse tensors. A sparse tensor can be represented as  $(\pmb{H}_{n + 1},\pmb{F}_{n + 1})$  where  $\pmb{H}_{n + 1}$  is a hash table recording the coordinates of those sites with active state, and  $\pmb{F}_{n + 1}$  is a feature matrix.  $\pmb{H}_{n + 1}$  and  $\pmb{F}_{n + 1}$  correspond to each other row-by-row; that is, if  $r_{n + 1,\pmb{x}}$  is the row number of  $\pmb{x}$  in  $\pmb{H}_{n + 1}$ , then  $\pmb{F}_{n + 1}[r_{n + 1,\pmb{x}}] = f_{n + 1}(\pmb{x})$ .

After obtaining  $\{\sigma_{n + 1}(\pmb {x})\}$ , the output hash table  $H_{n + 1}$  can be generated; the next target is to compute the values of  $F_{n + 1}$ . Specifically, we firstly initialize  $F_{n + 1}$  to zeros; then the feature vectors in  $F_{n + 1}$  are updated via the following algorithm:

ALGORITHM 1: Sparse Steerable Convolution  
Input:  $(H_{n},F_{n}),(H_{n + 1},F_{n + 1}),\{\kappa (s):s\in S\}$   
Output:  $(H_{n + 1},F_{n + 1})$   
1:  $R = \{R_s = \varnothing :s\in S\}$  // Initialize the rule book  $R$   
2: for  $x$  in  $H_{n + 1}$  do // Construct the rule book  $R$   
3: for  $y$  in  $H_{n}$  do  
4: if  $s = x - y\in S$   
5: Append  $(r_{n + 1,x},r_{n,y})$  to  $R_{s}$ . //  $r_{n + 1,x}$  is the row number of  $x$  in  $H_{n + 1}$   
6: end for //  $r_{n,y}$  is the row number of  $y$  in  $H_{n}$   
7: end for  
8: for  $R_{s}$  in  $R$  do // Update  $F_{n + 1}$   
9: for  $(r_{n + 1,x},r_{n,y})$  in  $R_{s}$  do  
10:  $F_{n + 1}[r_{n + 1,x}] = F_{n + 1}[r_{n + 1,x}] + \kappa (s)\times F_n[r_n,y]$   
11: end for  
12: end for

This process can be divided into two steps. The first step is to construct a rule book  $\mathbf{R} = \{R_s : s \in S\}$ , where an active output site  $\mathbf{x}$  is paired with an active input  $\mathbf{y}$  in each  $R_s$ , if  $\mathbf{x} - \mathbf{y} = s$ . The second step is to update  $F_{n+1}$  according to the paired relationships recorded in  $\mathbf{R}$ ; for example, if the paired relationship of output  $\mathbf{x}$  and input  $\mathbf{y}$  is recorded in  $\mathbf{R}_s$ , the current  $f_{n+1}(\mathbf{x})$  will be updated by adding the multiplication of  $f_n(\mathbf{y})$  and  $\kappa(\mathbf{s})$ . In this process, the construction of  $\mathbf{R}$  is very critical, which helps to implement the second step by matrix-matrix multiply-add operations on GPUs efficiently.

# 3.3 Normalization and Activation

As conventional CNNs do, SS-Convs are also followed by normalization and activation, i.e., Activation(Norm  $([\kappa \star f_n](\pmb{x}))$ ). Those operations of normalization and activation are required to be specially designed, not to break the SE(3)-equivariance of features. Since each SE(3)-equivariant feature is formed by stacking irreducible ones, without loss of generality, we take as an example an irreducible feature  $f(\pmb{x})$  with order  $l$ , so that the normalization can be formulated as follows:

$$
\operatorname {N o r m} (f (\boldsymbol {x})) = \left\{ \begin{array}{l l} (f (\boldsymbol {x}) - \mu) / \left(\sqrt {\frac {1}{N} \sum_ {\sigma (\boldsymbol {x}) = 1} \| f (\boldsymbol {x}) - \mu \| ^ {2} + \epsilon}\right), & l = 0 \\ f (\boldsymbol {x}) / \left(\sqrt {\frac {1}{N} \sum_ {\sigma (\boldsymbol {x}) = 1} \| f (\boldsymbol {x}) \| ^ {2} + \epsilon}\right), & l > 0 \end{array} \right. \tag {11}
$$

where  $\mu = \frac{1}{N}\sum_{\sigma (\pmb {x}) = 1}f(\pmb {x}),N$  is the number of active sites, and  $\epsilon$  is a very small constant. For the activation of  $f(x)$ , if  $l = 0$ , ReLU can be chosen to increase non-linearity; if  $l > 0$ , we follow [24] and multiply to  $f(\pmb {x})$  a scalar, which is learned by a SS-Conv and applied to the Sigmoid function:

$$
\operatorname {A c t i v a t i o n} (f (\boldsymbol {x})) = \left\{ \begin{array}{l l} R e L U (f (\boldsymbol {x})), & l = 0 \\ S i g m o i d \left(\left[ \kappa^ {0 l} \star f \right] (\boldsymbol {x})\right) f (\boldsymbol {x}). & l > 0 \end{array} \right. \tag {12}
$$

![](images/f26ff85cfabbc7e5199b93bc276647bea157d73894ddf67f1ac7dc335a2e4efe.jpg)  
Figure 2: An illustration of network architecture for instance-level 6D object pose estimation.

The above normalization and activation operations are both SE(3)-equivariant, since a feature vector multiplying any scalar keeps its equivariance; when applying them to features formed by numerous irreducible ones, we treat each irreducible member individually to ensure the equivariance.

# 4 Applications for Estimation and Tracking of Object Poses in 3D Space

# 4.1 Instance-level 6D Object Pose Estimation

Given an RGB-D image of a cluttered scene, instance-level 6D pose estimation is to estimate the 6D poses of known 3D objects with respect to the camera coordinate system. As introduced in Sec. 3.1, a 6D pose  $g \in \mathrm{SE}(3)$  can be decomposed into a 3D rotation  $r \in \mathrm{SO}(3)$  and a 3D translation  $t \in \mathbb{R}^3$ , which makes SS-Conv based network well suited for this task, due to: i) SS-Convs provide strong SE(3)-equivariant features to decode a precise 6D pose; ii) the steerability of feature maps helps to enable a second stage of pose refinement. Therefore, we propose an efficient SS-Conv based general pipeline for 6D pose estimation, as depicted in Fig. 2.

Specifically, we firstly segment out the objects of interest by an off-the-shelf model of instance segmentation, assigning each object with an RGB segment and a cropped point cloud; then each 3D object is voxelized and represented by a sparse tensor  $(\pmb{H}_0, \pmb{F}_0)$ , where each feature in  $\pmb{F}_0$  is a 4-dimensional vector, containing RGB values and a constant "1". For the input tensor, we set the site active, if the quantified grid centered at this site encloses any points, and average point features of those enclosed by a same grid.  $(\pmb{H}_0, \pmb{F}_0)$  is then fed into our pose estimation network, where the estimation could be achieved in the following two stages.

In the first stage, we construct an efficient SS-Conv based backbone, which extracts hierarchical SE(3)-equivariant feature maps, represented in the form of sparse tensors  $\{(H_n,F_n)\}$ . Those feature tensors are used for interpolation of multi-level point-wise features by using Tensor-to-Point Modules, proposed in [10], transforming features of discretized grid sites to those of real-world point coordinates. Each point feature is fed into two separate MLPs, regressing a point offset and a rotation, respectively, where the addition of the point coordinate and its offset generates a translation. The initially predicted pose  $(r_1,t_1)$  of this stage is obtained by averaging point-wise predictions.

In the second stage, we refine the pose  $(\pmb{r}_1, \pmb{t}_1)$  by learning a residual pose  $(\pmb{r}_2, \pmb{t}_2)$ , wherein a Feature-Steering module is designed, generating transformed features  $\{(H_n', F_n')\}$  by efficiently steering hierarchical backbone features  $\{(H_n, F_n)\}$  individually with  $(\pmb{r}_1, \pmb{t}_1)$ . Again we interpolate point-wise features from  $\{(H_n', F_n')\}$ , and average point-wise predictions to obtain  $(\pmb{r}_2, \pmb{t}_2)$ . Finally, the predicted 6D pose is then updated as  $(\pmb{r}_1 \pmb{r}_2, \pmb{t}_1 + \pmb{r}_1 \pmb{t}_2)$ . Additionally, owing to the novel SE(3)-Steering modules, this stage can be iteratively repeated, generating finer and finer poses.

# 4.1.1 The Feature-Steering Module

Feature-Steering module in the pipeline is to transform  $(\pmb{H}_n, \pmb{F}_n)$  of the backbone to  $(\pmb{H}_n', \pmb{F}_n')$ , where a rigid transformation of  $\pmb{H}_n$  with  $(r, t)$  and a rotation of  $\pmb{F}_n$  with  $\rho(\pmb{r})$  are included. Specifically, for  $\pmb{F}_n$ , we compute  $\rho(\pmb{r})$  as defined in (5) and rotate  $\pmb{F}_n$  by matrix multiplication; for  $\pmb{H}_n$ , we convert the sites in it to the real-world point coordinates, which are then applied to a rigid transformation of  $(r, t)$  and re-voxelized as grid sites. The same new sites are merged to a unique one, while their features are averaged. We also use two another SS-Convs, each followed by steerable normalization and activation, to enrich the new features and generate the final steered  $(\pmb{H}_n', \pmb{F}_n')$ .

# 4.2 Category-level 6D Object Pose and Size Estimation

Category-level 6D pose and size estimation is formally introduced in [23]. This is a more challenging task, which aims to estimate categorical 6D poses of unknown objects, and also the 3D object sizes. To tackle this problem, we use a similar network as that in Fig. 2, and make some adaptive modifications: i) for each stage in Fig. 2, we add another two separate MLPs for point-wise predictions of 3D sizes and point coordinates in the canonical space, respectively; ii) in each Feature-Steering module, the real-world coordinates of all 3D objects are also scaled by their predicted 3D sizes to be enclosed within a unit cube, for estimating more precise poses.

# 4.3 Category-level 6D Object Pose Tracking

Motivated by the above task of categorical pose estimation, category-level 6D pose tracking is also proposed to estimate the small change of 6D poses in two adjacent RGB-D frames of an image sequence [21]. Due to the available pose of the previous frame, the target object can be roughly located in the current frame, avoiding the procedures of object detection or instance segmentation in images. However, without a precise mask, the estimation of small pose change from noisy 3D data is a big challenge for deep networks. Our SS-Conv based network also surprisingly performs well in such noisy data, even though we only conduct one-stage pose estimation that achieves real-time tracking. For more details, one may refer to the supplementary material.

# 5 Experiments

Datasets We conduct experiments on the benchmark LineMOD dataset [11] for instance-level 6D pose estimation, which consists of 13 different objects. For both category-level 6D pose estimation and tracking, we experiment on REAL275 dataset [23], which is a more challenging real-world dataset with 4, 300 training images and 2, 750 testing ones, containing object instances of 6 categories. Following [23, 21], we augment the training data of REAL275 with synthetic RGB-D images.

Evaluation Metrics For instance-level task, we follow [22] and evaluate the results of LineMOD dataset on ADD(S) metric. For the category-level task, we report the mean Average Precision (mAP) of intersection over union (IoU) and  $n^{\circ} m$  cm, following [23]; mean rotation error  $(r_{err})$  in degrees and mean translation error  $(t_{err})$  in centimeters are also reported for pose tracking. Additionally, we compare the numbers of parameters (#Param) and the running speeds (FPS) for different models. Testing is conducted on a server with GeForce RTX 2080ti GPU for a batchsize of 32, and FPS is computed by averaging the time cost of forward propagation on the whole dataset.

# 5.1 Comparisons with Different 3D Convolutions

We firstly conduct experiments to compare our proposed SS-Conv with other kinds of 3D convolutions, including conventional 3D convolution (Dense-Conv), sparse convolution (SP-Conv) [9], and steerable convolution (ST-Conv) [24], on the LineMOD dataset for instance-level 6D pose estimation. Among those convolutions, SP-Conv improves the speed of Dense-Conv by considering data sparsity and turns out to be efficient in 3D object detection, while ST-Conv constructs rotation-steering kernels and realizes the convolutional operation based on Dense-Conv.

To meet various computational demands of different convolutions, those experiments are conducted on a light plain architecture in the same experimental settings, for a fair comparison. The architecture consists of 12 convolutional layers, of which the kernel sizes are all set as  $3 \times 3 \times 3$ ; feature channels are kept consistent when applying different convolutions. We use ADAM to train the networks for a

Table 1: Quantitative comparisons of different 3D convolutions on the LineMOD dataset [11].  

<table><tr><td>Conv</td><td>ADD(S) ↑</td><td>FPS ↑</td><td>#Param ↓</td></tr><tr><td>Dense-Conv</td><td>46.5</td><td>224</td><td>26.2 M</td></tr><tr><td>SP-Conv</td><td>62.8</td><td>486</td><td>26.2 M</td></tr><tr><td>ST-Conv</td><td>92.8</td><td>148</td><td>3.6 M</td></tr><tr><td>SS-Conv</td><td>93.5</td><td>293</td><td>3.6 M</td></tr></table>

![](images/e1188ff30ae5996274bd4550345e3db5dc6e87de82bc5ed5b32b4a1b1938e492.jpg)  
(a)

![](images/9da668e6d863b1c4d00157e7099497473b9559517cd1d3eaafd6cd32a689f466.jpg)  
Figure 3: Training of plain networks with different convolutions on the LineMOD dataset [11].

![](images/fe41e53b400141230d291b1646fa5b4c4ef7f6e38a5bc286581626e9bf18bcac.jpg)  
(b)

![](images/31d5d1686a3f917c916e4d6aa861c1d2803061fed9c93dd7207be59136e574aa.jpg)

total of 30,000 iterations, with an initial learning rate of 0.01; learning rate is halved every 1,500 iterations. We voxelize the input segmented objects into  $64 \times 64 \times 64$  dense/sparse grids, and set the training batchsize as 16.

Quantitative results of different convolutions are listed in Table 1, which confirms the advantages of our SS-Conv in both performance and efficiency. In terms of performance, SS-Conv achieves comparable results on LineMOD dataset as ST-Conv does, which marginally outperforms those of Dense-Conv and SP-Conv, indicating the importance of SE(3)-equivariant feature learning on pose estimation. Although the convolutional kernel of SS-Conv is restricted under the rotation-steering constraint, its capacity of feature extraction is not weakened, and instead, strengthened by constantly preserving the relative poses of features layer-by-layer; as a result, the property of SE(3)-equivariance makes it natural for learned features to extract more information of object poses. Additionally, due to the kernel constraint, the number of learnable parameters of SS-Conv/ST-Conv is much smaller than those of the other two convolutions, which is also listed in Table 1. In terms of efficiency, SS-Conv based network doubles the FPS of ST-Conv based one, and becomes more efficient and flexible to be applied to complex systems. The efficiency of SS-Conv can also be verified in Fig. 3, where we visualize the behaviors of the four convolutions in the process of training; as shown in the figure, the training time of SS-Conv based network is marginally shortened, compared to that of ST-Conv based one, although their loss curves both decrease consistently over iterations.

# 5.2 Comparisons with Existing Methods

Instance-level 6D Object Pose Estimation For the instance-level task, we compare the results of our SS-Conv based pipeline with existing methods on LineMOD dataset [11]. Quantitative results are shown in Table 2, where our two-stage pipeline outperforms all the existing methods and achieves a new state-of-the-art result of  $99.2\%$  on mean ADD(S) metric. We can also observe that the second stage of pose refinement with Feature-Steering modules in our pipeline indeed improves the predictions in the first stage, benefitting from the steerability of the feature spaces in SS-Convs.

Category-level 6D Object Pose and Size Estimation We conduct experiments on REAL275 [23] for the more challenging category-level task. Quantitative results in Table 3 confirm the advantage of our pipeline in the high-precision regime, especially on the precise metric of  $5^{\circ}5\mathrm{cm}$ , where we improve the state-of-the-art result in [4] from  $28.5\%$  to  $41.7\%$ . The second stage of pose refinement also plays an important role in this task, achieving remarkable improvements over the first stage.

Category-level 6D Object Pose Tracking We compare the results of our one-stage tracking pipeline with the baseline of 6-PACK [21] on REAL275 [23]. In 6-PACK, the relative pose between two frames is computed based on predicted keypoint pairs inefficiently, while our pipeline regresses the pose in a direct way. The results in Table 4 show that our pipeline outperforms 6-PACK on all the evaluation metrics, demonstrating the ability of SS-Conv based network for fine-grained pose estimation in noisy input data.

More implementation details and qualitative results are shown in the supplementary material.

Table 2: Quantitative comparisons of different methods on the LineMOD dataset [11] for instance-level 6D object pose estimation. The evaluation metric is ADD(S).  

<table><tr><td></td><td>Implicit[17] +ICP</td><td>SSD6D[13] +ICP</td><td>PointFusion [26]</td><td>DenseFusion [22]</td><td>DenseFusion (Iterative)[22]</td><td>G2L[3]</td><td>Ours w/o second stage</td><td>Ours</td></tr><tr><td>ape</td><td>20.6</td><td>65</td><td>70.4</td><td>79.5</td><td>92.3</td><td>96.8</td><td>92.9</td><td>97.4</td></tr><tr><td>bench.</td><td>64.3</td><td>80</td><td>80.7</td><td>84.2</td><td>93.2</td><td>96.1</td><td>97.4</td><td>99.3</td></tr><tr><td>camera</td><td>63.2</td><td>78</td><td>60.8</td><td>76.5</td><td>94.4</td><td>98.2</td><td>97.7</td><td>99.5</td></tr><tr><td>can</td><td>76.1</td><td>86</td><td>61.1</td><td>86.6</td><td>93.1</td><td>98.0</td><td>96.1</td><td>99.6</td></tr><tr><td>cat</td><td>72.0</td><td>70</td><td>79.1</td><td>88.8</td><td>96.5</td><td>99.2</td><td>98.8</td><td>99.8</td></tr><tr><td>driller</td><td>41.6</td><td>73</td><td>47.3</td><td>77.7</td><td>87.0</td><td>99.8</td><td>98.7</td><td>99.6</td></tr><tr><td>duck</td><td>32.4</td><td>66</td><td>63.0</td><td>76.3</td><td>92.3</td><td>97.7</td><td>91.1</td><td>97.8</td></tr><tr><td>egg.</td><td>98.6</td><td>100</td><td>99.9</td><td>99.9</td><td>99.8</td><td>100.0</td><td>100.0</td><td>99.9</td></tr><tr><td>glue</td><td>96.4</td><td>100</td><td>99.3</td><td>99.4</td><td>100.0</td><td>100.0</td><td>98.6</td><td>99.6</td></tr><tr><td>hole.</td><td>49.9</td><td>49</td><td>71.8</td><td>79.0</td><td>92.1</td><td>99.0</td><td>96.3</td><td>99.4</td></tr><tr><td>iron</td><td>63.1</td><td>78</td><td>83.2</td><td>92.1</td><td>97.0</td><td>99.3</td><td>98.7</td><td>99.2</td></tr><tr><td>lamp</td><td>91.7</td><td>73</td><td>62.3</td><td>92.3</td><td>95.3</td><td>99.5</td><td>99.5</td><td>99.7</td></tr><tr><td>phone</td><td>71.0</td><td>79</td><td>78.8</td><td>88.0</td><td>92.8</td><td>98.9</td><td>97.5</td><td>98.2</td></tr><tr><td>MEAN</td><td>64.7</td><td>79</td><td>73.7</td><td>86.2</td><td>94.3</td><td>98.7</td><td>97.2</td><td>99.2</td></tr></table>

Table 3: Quantitative comparisons of different methods on REAL275 dataset [23] for category-level 6D object pose and size estimation.  

<table><tr><td rowspan="2">Method</td><td colspan="6">mAP</td></tr><tr><td>IoU50</td><td>IoU75</td><td>5°2cm</td><td>5°5cm</td><td>10°2cm</td><td>10°5cm</td></tr><tr><td>NOCS [23]</td><td>78.0</td><td>30.1</td><td>7.2</td><td>10.0</td><td>13.8</td><td>25.2</td></tr><tr><td>SPD [20]</td><td>77.3</td><td>53.2</td><td>19.3</td><td>21.4</td><td>43.2</td><td>54.1</td></tr><tr><td>CASS [2]</td><td>77.7</td><td>-</td><td>-</td><td>23.5</td><td>-</td><td>58.0</td></tr><tr><td>FS-Net [4]</td><td>92.2</td><td>63.5</td><td>-</td><td>28.2</td><td>-</td><td>60.8</td></tr><tr><td>Ours w/o second stage</td><td>79.5</td><td>58.7</td><td>19.2</td><td>25.2</td><td>35.1</td><td>49.9</td></tr><tr><td>Ours</td><td>79.1</td><td>63.7</td><td>31.5</td><td>41.7</td><td>45.8</td><td>61.7</td></tr></table>

# 6 Discussions

The studied problems of object pose estimation and tracking in 3D space are very important to many real-world applications, including augmented reality, robotic grasping, and autonomous driving. By precisely predicting object poses in the 3D space, virtual contents could be seamlessly embedded in real environments, creating fascinating personal experience; on the contrary, less precise predictions may cause property loss and even life threat, especially in autonomous driving. The contributed solution based on SS-Conv would improve the overall level of safety.

In the paper, while we only demonstrate the advantages of SS-Conv on an object-centric pipeline, its capacity for processing 3D scenes with small regions is also revealed in tracking. In future work, we will work on the investigation of its potentials for tasks on large-scale scenes.

Table 4: Quantitative comparisons of different methods on REAL275 dataset [23] for category-level 6D object pose tracking.  

<table><tr><td>Method</td><td>Metric</td><td>bottle</td><td>bow</td><td>camera</td><td>can</td><td>laptop</td><td>mug</td><td>MEAN</td></tr><tr><td rowspan="4">6PACK [21]</td><td>5°5cm ↑</td><td>24.5</td><td>55.0</td><td>10.1</td><td>22.6</td><td>63.5</td><td>24.1</td><td>33.3</td></tr><tr><td>IoU25 ↑</td><td>91.1</td><td>100.0</td><td>87.6</td><td>92.6</td><td>98.1</td><td>95.2</td><td>94.1</td></tr><tr><td>Rerr ↓</td><td>15.6</td><td>5.2</td><td>35.7</td><td>13.9</td><td>4.7</td><td>21.3</td><td>16.1</td></tr><tr><td>Terr ↓</td><td>4.0</td><td>1.7</td><td>5.6</td><td>4.8</td><td>2.5</td><td>2.3</td><td>3.5</td></tr><tr><td rowspan="4">Ours</td><td>5°5cm ↑</td><td>70.3</td><td>60.6</td><td>10.6</td><td>49.9</td><td>87.7</td><td>47.9</td><td>54.5</td></tr><tr><td>IoU25 ↑</td><td>93.5</td><td>99.9</td><td>99.9</td><td>99.8</td><td>99.8</td><td>99.9</td><td>98.8</td></tr><tr><td>Rerr ↓</td><td>3.7</td><td>4.6</td><td>9.8</td><td>4.6</td><td>3.0</td><td>5.6</td><td>5.2</td></tr><tr><td>Terr ↓</td><td>1.9</td><td>1.2</td><td>2.0</td><td>2.7</td><td>2.4</td><td>1.1</td><td>1.9</td></tr></table>

# References

[1] Eric Brachmann, Alexander Krull, Frank Michel, Stefan Gumhold, Jamie Shotton, and Carsten Rother. Learning 6d object pose estimation using 3d object coordinates. In European conference on computer vision, pages 536-551. Springer, 2014.  
[2] Dengsheng Chen, Jun Li, Zheng Wang, and Kai Xu. Learning canonical shape space for category-level 6d object pose and size estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11973-11982, 2020.  
[3] Wei Chen, Xi Jia, Hyung Jin Chang, Jinming Duan, and Ales Leonardis. G21-net: global to local network for real-time 6d pose estimation with embedding vector features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4233-4242, 2020.  
[4] Wei Chen, Xi Jia, Hyung Jin Chang, Jinming Duan, Linlin Shen, and Ales Leonardis. Fs-net: Fast shape-based network for category-level 6d object pose estimation with decoupled rotation mechanism. arXiv preprint arXiv:2103.07054, 2021.  
[5] Taco S Cohen, Mario Geiger, Jonas Kohler, and Max Welling. Spherical cnns. In International Conference on Learning Representations, 2018.  
[6] Carlos Esteves, Christine Allen-Blanchette, Ameesh Makadia, and Kostas Daniilidis. Learning so (3) equivariant representations with spherical cnns. In Proceedings of the European Conference on Computer Vision (ECCV), pages 52–68, 2018.  
[7] Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling. Se (3)-transformers: 3d roto-translation equivariant attention networks. Advances in Neural Information Processing Systems, 33, 2020.  
[8] Robert Gilmore. Lie groups, physics, and geometry: an introduction for physicists, engineers and chemists. Cambridge University Press, 2008.  
[9] Benjamin Graham, Martin Engelcke, and Laurens Van Der Maaten. 3d semantic segmentation with submanifold sparse convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9224-9232, 2018.  
[10] Chenhang He, Hui Zeng, Jianqiang Huang, Xian-Sheng Hua, and Lei Zhang. Structure aware single-stage 3d object detection from point cloud. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11873-11882, 2020.  
[11] Stefan Hinterstoisser, Stefan Holzer, Cedric Cagniart, Slobodan Ilic, Kurt Konolige, Nassir Navab, and Vincent Lepetit. Multimodal templates for real-time detection of texture-less objects in heavily cluttered scenes. In 2011 international conference on computer vision, pages 858-865. IEEE, 2011.  
[12] Stefan Hinterstoisser, Vincent Lepetit, Slobodan Ilic, Stefan Holzer, Gary Bradski, Kurt Konolige, and Nassir Navab. Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. In Asian conference on computer vision, pages 548-562. Springer, 2012.  
[13] Wadim Kehl, Fabian Manhardt, Federico Tombari, Slobodan Ilic, and Nassir Navab. Ssd-6d: Making rgb-based 3d detection and 6d pose estimation great again. In Proceedings of the IEEE international conference on computer vision, pages 1521–1529, 2017.  
[14] Alexander Krull, Eric Brachmann, Frank Michel, Michael Ying Yang, Stefan Gumhold, and Carsten Rother. Learning analysis-by-synthesis for 6d pose estimation in rgb-d images. In Proceedings of the IEEE international conference on computer vision, pages 954-962, 2015.  
[15] Frank Michel, Alexander Kirillov, Eric Brachmann, Alexander Krull, Stefan Gumhold, Bogdan Savchynskyy, and Carsten Rother. Global hypothesis generation for 6d object pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 462-471, 2017.  
[16] Sida Peng, Yuan Liu, Qixing Huang, Xiaowei Zhou, and Hujun Bao. Pvnet: Pixel-wise voting network for 6dof pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4561-4570, 2019.  
[17] Martin Sundermeyer, Zoltan-Csaba Marton, Maximilian Durner, Manuel Brucker, and Rudolph Triebel. Implicit 3d orientation learning for 6d object detection from rgb images. In Proceedings of the European Conference on Computer Vision (ECCV), pages 699-715, 2018.  
[18] Bugra Tekin, Sudipta N Sinha, and Pascal Fua. Real-time seamless single shot 6d object pose prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 292-301, 2018.

[19] Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. arXiv preprint arXiv:1802.08219, 2018.  
[20] Meng Tian, Marcelo H Ang, and Gim Hee Lee. Shape prior deformation for categorical 6d object pose and size estimation. In European Conference on Computer Vision, pages 530-546. Springer, 2020.  
[21] Chen Wang, Roberto Martin-Martin, Danfei Xu, Jun Lv, Cewu Lu, Li Fei-Fei, Silvio Savarese, and Yuke Zhu. 6-pack: Category-level 6d pose tracker with anchor-based keypoints. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 10059–10066. IEEE, 2020.  
[22] Chen Wang, Danfei Xu, Yuke Zhu, Roberto Martin-Martin, Cewu Lu, Li Fei-Fei, and Silvio Savarese. Densefusion: 6d object pose estimation by iterative dense fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3343–3352, 2019.  
[23] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2642-2651, 2019.  
[24] Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco Cohen. 3d steerable cnns: Learning rotationally equivariant features in volumetric data. arXiv preprint arXiv:1807.02547, 2018.  
[25] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. Poseconn: A convolutional neural network for 6d object pose estimation in cluttered scenes. arXiv preprint arXiv:1711.00199, 2017.  
[26] Danfei Xu, Dragomir Anguelov, and Ashesh Jain. Pointfusion: Deep sensor fusion for 3d bounding box estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 244-253, 2018.
