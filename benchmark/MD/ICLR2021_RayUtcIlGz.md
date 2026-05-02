# TRAINING NEURAL NETWORKS WITH PROPERTY-PRESERVING PARAMETER PERTURBATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Many types of neural network layers rely on matrix properties such as invertibility or orthogonality. Retaining such properties during optimization with gradient-based stochastic optimizers is a challenging task, which is usually addressed by either reparameterization of the affected parameters or by directly optimizing on the manifold. In contrast, this work presents a novel, general approach of preserving matrix properties by using parameterized perturbations. In lieu of directly optimizing the network parameters, the introduced  $P^4$  update optimizes perturbations and merges them into the actual parameters infrequently such that the desired property is preserved. As a demonstration, we use this concept to preserve invertibility of linear layers during training. This  $P^4$ Inv update allows keeping track of inverses and determinants using rank-one updates without ever explicitly computing them. We show how such invertible blocks improve the mixing of coupling layers and thus the mode separation of the resulting normalizing flows.

# 1 INTRODUCTION

Many deep learning applications depend critically on the neural network parameters having a certain mathematical structure. As an important example, reversible generative models rely on invertibility and, in the case of normalizing flows, efficient computation of the Jacobian determinant (Papamakarios et al., 2019). Other models require (or benefit from) orthogonal linear layers or bounded Lipschitz constants. Finally, applications in physics often rely on networks that obey the relevant physical invariances and equivariances (e.g. Kohler et al. (2020); Boyda et al. (2020); Kanwar et al. (2020); Hermann et al. (2020); Pfau et al. (2020); Rezende et al. (2019)).

Preserving parameter properties during training can be challenging and many approaches are currently in use. The most basic way of incorporating constraints is by network design. Many examples could be listed, like defining convolutional layers to obtain equivariances, constraining network outputs to certain intervals through bounded activation func

tions, Householder flows (Tomczak & Welling, 2016) to enforce layer-wise orthogonality, or coupling layers (Dinh et al., 2014; 2016) that enforce tractable inversion through their two-channel structure. A second approach concerns the optimizers used for training. Optimization routines have been tailored for example to enforce Lipschitz bounds (Yoshida & Miyato, 2017) or efficiently optimize orthogonal linear layers (Choromanski et al., 2020).

![](images/c8fd2d0e3a15e524d97f87816e84cb2953159d0e9b37b2a938ba48c05a67e370.jpg)  
Figure 1: Training of deep neural networks (DNN). Standard DNN transform inputs  $x$  into outputs  $y$  through activation functions and linear layers, which are tuned by an optimizer. In contrast,  $\mathbf{P}^4$  training operates on perturbations to the parameters. Those are defined to retain certain network properties. The perturbed parameters are merged in regular intervals.

The present work introduces a novel algorithmic concept for training neural networks in a property-preserving manner that is orthogonal to existing approaches, see Figure 1. In lieu of directly changing the network parameters, the optimizer operates on perturbations to these parameters. The actual network parameters are frozen, while a parameterized perturbation serves as a proxy for optimization. Inputs are passed through the perturbed network during training. In regular intervals, the perturbed parameters are merged into the actual network and the perturbation is reset to the identity.

This stepwise reparameterization trick has several advantages for optimizing neural networks under constraints. First, it is usually easier to generate new sets of parameters that obey the constraints through suitable perturbations than from scratch. Therefore, we will refer to these updates as property-preserving parameter perturbations, or  $P^4$  updates. Second, the merging step can occur infrequently (e.g. every 100 iterations) and perform the heavy-lifting so that the perturbations occurring in every step are kept computationally efficient. Specifically, the constraints need not be rigorously obeyed in every optimization step. Rather, numerical inaccuracies can be corrected before merging, which avoids propagation of errors into the actual (frozen) network parameters. Third, the perturbative update does not affect the network structure or computational cost outside of training. Finally, the general method can be used to retain desirable properties of either individual layers or the deep neural network as a whole, depending on how the perturbation is defined.

To demonstrate these benefits in a practical example, we efficiently train invertible linear layers while keeping track of their inverses and determinants using rank-one updates. Previous works (see Section 3) have mostly focused on optimizing orthogonal matrices, which can be trivially inverted and have unity determinant. Only most recently, Gresele et al. (2020) presented a first method to optimize general invertible matrices implicitly using relative gradients, thereby providing greater flexibility and expressivity.

The novel  $\mathrm{P^4Inv}$  scheme presents an alternative approach to train arbitrary invertible matrices  $A\in \mathrm{GL}(n)$ . Interestingly, our stepwise perturbation even allows sign changes in the determinants and recovers the correct inverse after emerging from the ill-conditioned regime. Furthermore, it avoids any explicit computations of inverses or determinants. All operations occurring in standard optimization steps have complexity of at most  $\mathcal{O}(n^2)$ . Some operations in the merging step are  $\mathcal{O}(n^3)$  but embarrassingly parallel. Finally, we show how such invertible blocks can be utilized in normalizing flows by combining them with nonlinear, bijective activation functions and with coupling layers. The resulting neural networks are validated for density estimation and as deep generative models.

# 2  $\mathbf{P}^{4}$  UPDATES: PRESERVING PROPERTIES THROUGH PERTURBATIONS

# 2.1 GENERAL CONCEPT

A deep neural network is a parameterized function  $M_{\mathbf{A}}: \mathbb{R}^n \to \mathbb{R}^m$  with a high-dimensional parameter tensor  $\mathbf{A}$ . Now, let  $\mathbb{S}$  define the subset of feasible parameter tensors so that the network satisfies a certain desirable property. In many situations, generating elements of  $\mathbb{S}$  from scratch is much harder than transforming any  $\mathbf{A} \in \mathbb{S}$  into other elements  $\mathbf{A}' \in \mathbb{S}$ , i.e. to move within  $\mathbb{S}$ . For instance, parameterizing the whole orthogonal group (via the Cayley parameterization or matrix exponentials of skew-symmetric matrices) is computationally expensive. In contrast, orthogonal matrices can be cheaply perturbed into other orthogonal matrices using double Householder transforms or Givens rotations.

The efficiency of perturbative updates can be leveraged as an incremental approach to retain certain desirable properties of the network parameters during training. Rather than optimizing the parameter tensors directly, we instead use a transformation  $R_{\mathbf{B}}: \mathbb{S} \to \mathbb{S}$ , which we call a property-preserving parameter perturbation (P<sup>4</sup>). A P<sup>4</sup> transforms a given parameter tensor  $\mathbf{A} \in \mathbb{S}$  into another tensor with the desired property  $\mathbf{A}' \in \mathbb{S}$ . The P<sup>4</sup> itself is also parameterized, by a tensor  $\mathbf{B}$ . We demand that the identity  $id_{\mathbb{S}}: \mathbf{A} \mapsto \mathbf{A}$  be included in the set of these transformations, i.e. there exists a  $\mathbf{B}_0$  such that  $\mathbf{B}_0 = id_{\mathbb{S}}$ .

During training, the network is evaluated using the perturbed parameters  $\tilde{\mathbf{A}} = R_{\mathbf{B}}(\mathbf{A})$ . The parameter tensor of the perturbation,  $\mathbf{B}$ , is trainable via gradient-based stochastic optimizers, while the actual parameters  $\mathbf{A}$  are frozen. In regular intervals, every  $N$  iterations of the optimizer, the optimized

parameters of the  $\mathbf{P}^4$ ,  $\mathbf{B}$ , are merged into  $\mathbf{A}$  as follows:

$$
\mathbf {A} _ {\text {n e w}} \leftarrow R _ {\mathbf {B}} (\mathbf {A}), \tag {1}
$$

$$
\mathbf {B} _ {\text {n e w}} \leftarrow \mathbf {B} _ {0}. \tag {2}
$$

This update does not modify the effective (perturbed) parameters of the network  $\tilde{\mathbf{A}}$ , since

$$
\tilde {\mathbf {A}} _ {\text {n e w}} = R _ {\mathbf {B} _ {\text {n e w}}} (\mathbf {A} _ {\text {n e w}}) = R _ {\mathbf {B} _ {0}} \left(R _ {\mathbf {B}} (\mathbf {A})\right) = R _ {\mathbf {B}} (\mathbf {A}) = \tilde {\mathbf {A}}.
$$

Hence, this procedure enables a steady, iterative transformation of the effective network parameters and stochastic gradient descent methods can be used for training without major modifications. Furthermore, given a reasonable  $\mathbf{P}^4$ , the iterative update of  $\mathbf{A}$  can produce increasingly non-trivial transformations, thereby enabling high expressivity of the resulting neural networks. This concept is summarized in Algorithm 1.

Algorithm 1:  $\mathrm{P^4}$  Training  
Input: Model  $M$ , training data, loss function  $J$ , number of optimization steps  $N_{\mathrm{steps}}$ , merge interval  $N$ , perturbation  $R$ , optimizer OPT  
initialize  $\mathbf{A} \in \mathbb{S}$ ;  
initialize  $\mathbf{B} := \mathbf{B}_0$ ;  
for  $i := 1 \dots N_{\mathrm{steps}}$  do  
 $\mathbf{X}, \mathbf{Y}_0 := i$ -th batch from training data;  
 $\tilde{\mathbf{A}} := R_{\mathbf{B}}(\mathbf{A})$ ; // perturb parameters  
 $\mathbf{Y} := M_{\tilde{\mathbf{A}}}(\mathbf{X})$ ; // evaluate perturbed model  
 $\mathbf{j} := J(\mathbf{Y}, \mathbf{Y}_0)$ ; // evaluate loss function  
gradient :=  $\partial j / \partial \mathbf{B}$ ; // backpropagation  
 $\mathbf{B} := \mathrm{OPT}(\mathbf{B}, \mathrm{gradient})$ ; // optimization step  
if  $i \mod N = 0$  then  
 $\mathbf{A} := R_{\mathbf{B}}(\mathbf{A})$ ; // merging step: update frozen parameters  
 $\mathbf{B} := \mathbf{B}_0$ ; // merging step: reset perturbation  
end

Further extensions to stabilize the merging step will be exemplified in Section 2.4. In order to parameterize invertible matrices, the next section revisits rank-one perturbation.

# 2.2 RANK-ONE PERTURBATION OF REGULAR MATRICES

Rank-one updates are defined as transformations  $\mathbf{A} \mapsto \mathbf{A} + \mathbf{u}\mathbf{v}^T$  with  $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ . If  $\mathbf{A} \in \mathrm{GL}(n)$  and  $1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u} \neq 0$ , the updated matrix is also invertible and its inverse can be computed by the Sherman-Morrison formula

$$
\left(\boldsymbol {A} + \boldsymbol {u} \boldsymbol {v} ^ {T}\right) ^ {- 1} = \boldsymbol {A} ^ {- 1} - \frac {1}{1 + \boldsymbol {v} ^ {T} \boldsymbol {A} ^ {- 1} \boldsymbol {u}} \boldsymbol {A} ^ {- 1} \boldsymbol {u} \boldsymbol {v} ^ {T} \boldsymbol {A} ^ {- 1}. \tag {3}
$$

Furthermore, the determinant is given by the matrix determinant lemma

$$
\det  (\boldsymbol {A} + \boldsymbol {u} \boldsymbol {v} ^ {T}) = (1 + \boldsymbol {v} ^ {T} \boldsymbol {A} ^ {- 1} \boldsymbol {u}) \det  (\boldsymbol {A}). \tag {4}
$$

Both these equations are widely used in numerical mathematics, since they sidestep the  $\mathcal{O}(n^3)$  cost and poor parallelization of both matrix inversion and determinant computation. The present work leverages these perturbation formulas to keep track of the inverses and determinants of weight matrices during training of invertible neural networks.

# 2.3  $\mathbf{P}^4\mathrm{INV}$  : FULLY FLEXIBLE INVERTIBLE LAYERS VIA RANK-ONE UPDATES

The  $\mathrm{P^4}$  algorithm can in principle be applied to properties concerning either individual blocks or the whole network. Here we train individual invertible linear layers via rank-one perturbations. Each of these  $\mathrm{P^4}$  Inv layers is an affine transformation  $Ax + b$ . In this context, the weight matrix  $\mathbf{A}$  is handled by the  $\mathrm{P^4}$  update and the bias  $\pmb{b}$  is optimized without perturbation. Without loss of generality, we present the method for layers  $Ax$ .

We define  $\mathbb{S}$  as the set of invertible matrices, for which we know the inverse and determinant. Then the rank-one update

$$
R _ {\boldsymbol {u}, \boldsymbol {v}} (\boldsymbol {A}) = \boldsymbol {A} + \boldsymbol {u} \boldsymbol {v} ^ {T} \tag {5}
$$

with  $\mathbf{B} = (\pmb{u},\pmb{v})\in \mathbb{R}^{2n}$  is a  $\mathrm{P^4}$  on  $\mathbb{S}$  due to equations 3 and 4, which also define the inverse pass and determinant computation of the perturbed layer, see Appendix B for details. The perturbation can be reset by setting  $\pmb{u}$ ,  $\pmb{v}$ , or both to zero. In subsequent parameter studies, a favorable training efficiency was obtained by setting  $\pmb{u}$  to zero and reinitializing  $\pmb{v}$  from noise. The inverse matrix  $A_{\mathrm{inv}}$  and determinant  $d$  are stored in the  $\mathrm{P^4}$  layer alongside  $\pmb{A}$  and updated according to the merging step in Algorithm 2. Merges are skipped whenever the determinant update signals ill conditioning of the inversion. This is further explained in the following subsection.

Algorithm 2:  $\mathrm{P^4}$  Inv Merging Step  
```latex
Input : Matrix  $A$  ,Inverse  $A_{\mathrm{inv}}$  ,Determinant  $d$    
det_factor:  $\coloneqq (1 + \pmb {v}^T\pmb{A}_{\mathrm{inv}}\pmb {u})$    
new_DET:=det_factor·d;   
if ln |det_factor| and ln |new_DET| are sane then   
/\*update frozen parameters (equation 1) /\*   
 $d\coloneqq$  new_DET;   
 $\mathbf{A}\coloneqq R_{\mathbf{u},\mathbf{v}}(\mathbf{A})$ $\mathbf{A}_{\mathrm{inv}}\coloneqq \mathbf{A}_{\mathrm{inv}} - \frac{1}{1 + v^T\mathbf{A}_{\mathrm{inv}}\mathbf{u}}\mathbf{A}_{\mathrm{inv}}\mathbf{u}\mathbf{v}^T\mathbf{A}_{\mathrm{inv}};$    
/\*reset perturbation (equation 2) /\*   
 $\pmb {u}\coloneqq 0$  .   
 $\pmb {v}\coloneqq \mathcal{N}(0,\pmb {I}_n)$  // random reinitialization   
end
```

# 2.4 NUMERICAL STABILIZATION

The update to the inverse and determinant can become ill-conditioned if the denominator in equation 3 is close to zero. Thankfully, the determinant lemma from equation 4 also provides an indicator for ill-conditioned updates (if absolute determinants become very small or very large). This indicator in combination with the stepwise merging approach can be used to tame potential numerical issues. Concretely, the following additional strategies were applied to ensure stable optimizations.

- Skip Merges: Merges are skipped whenever the determinant update falls out of predefined bounds, see Appendix A for details. This allows the optimization to continue without propagating numerical errors into the actual weight matrix  $\mathbf{A}$ . Note that numerical errors in the perturbed parameters  $\tilde{\mathbf{A}}$  are instantaneous and vanish when the optimization leaves the ill-conditioned regime. As shown in our experiments in Section 4.1, merging steps that occur relatively infrequently without drastically hurting the efficiency of the optimization.  
- Penalization: The objective function can be augmented by a penalty function  $g(\pmb{u}, \pmb{v})$  in order to prevent entering the ill-conditioned regime  $\{(u, v) : \det(R_{\pmb{u}, \pmb{v}}(\pmb{A})) = 0\}$ , see Appendix A.  
- Iterative Inversion: In order to maintain a small error of the inverse throughout training, the inverse is corrected after every  $N_{\mathrm{correct}}$ -th merging step by one iteration of an iterative matrix inversion (Soleymani, 2014). This operation is  $O(n^{3})$  yet is highly parallel.

# 2.5 USE IN INVERTIBLE NETWORKS

Our invertible linear layers can be employed in normalizing flows thanks to having access to the determinant at each update step. We tested them in two different application scenarios:

$\mathbf{P}^4\mathrm{Inv}$  Swaps In a first attempt, we integrate  $\mathrm{P^4Inv}$  layers with RealNVP coupling layers by substituting the simple coordinate swaps with general linear layers (see Figure 7 in the appendix). Fixed coordinate swaps span a tiny subset of  $\mathrm{O}(n)$ . In contrast,  $\mathrm{P^4Inv}$  can express all of  $\mathrm{GL}(n)$ . We thus expect more expressivity with the help of better mixing. The parameter matrix  $\mathbf{A}$  is initialized with a permutation matrix. Note that the  $\mathrm{P^4}$  training is exclusively applied to the  $\mathrm{P^4Inv}$  layers.

Nonlinear invertible layer In a second attempt, we follow the approach of Gresele et al. (2020) and stack  $\mathrm{P^4Inv}$  layers with intermediate bijective nonlinear activation functions. Here we use the elementwise-wise Bent identity which is a  $\mathbb{R}$ -diffeomorphism:

$$
B (x) = \frac {\sqrt {x ^ {2} + 1} - 1}{2} + x.
$$

# 3 RELATED WORK

A reparameterization trick for general Lie groups has been introduced in Falorsi et al. (2019). For the unitary / orthogonal group there are multiple more specialized approaches, including using the Cayley transform (Helfrich et al., 2018), Householder Reflections (Mhammedi et al., 2017; Meng et al., 2020; Tomczak & Welling, 2016), Givens rotations (Lezcano-Casado & Martínez-Rubio, 2019; Pevny et al., 2020) or the exponential map (Golinski et al., 2019).

Maintaining invertibility of linear layers has been studied in the context of convolution operators (Kingma & Dhariwal, 2018; Karami et al., 2019; Hoogeboom et al., 2019; 2020) and using Sylvester's theorem (Berg et al., 2018). More closely to our work, Gresele et al. (2020) introduce a relative gradient optimization scheme for invertible matrices. In contrast to this related work our method reparameterizes the update step during gradient descent rather than the parameter matrix itself. Beyond reparameterization, constrained matrices can also be optimized using Riemannian gradient descent on the manifold (e.g. see Li et al. (2020); Choromanski et al. (2020)) which is orthogonal to our work. Optimizers that constrain a joint property of multiple layers are used to enforce Lipschitz bounds (Gouk et al. (2018), Yoshida & Miyato (2017)).

Normalizing flows were introduced in Tabak et al. (2010); Tabak & Turner (2013) and are commonly used, either in variational inference (Rezende & Mohamed, 2015; Tomczak & Welling, 2016; Louizos & Welling, 2017; Berg et al., 2018) or for approximate sampling from distributions given by an energy function (Oord et al., 2017; Müller et al., 2018; Noé et al., 2019; Kohler et al., 2020). They can roughly be categorized in two families: (1) Coupling layers (Dinh et al., 2014; Kingma & Dhariwal, 2018; Müller et al., 2018), which are a subclass of autoregressive flows (Germain et al., 2015; Papamakarios et al., 2017; Huang et al., 2018; De Cao et al., 2019), and (2) residual flows (Chen et al., 2018; Zhang et al., 2018; Grathwohl et al., 2018; Behrmann et al., 2018; Chen et al., 2019). A comprehensive survey on flows can be found in Papamakarios et al. (2019).

# 4 EXPERIMENTS

$\mathrm{P^4Inv}$  updates are demonstrated in three steps. First, as a basic validation, single  $\mathrm{P^4Inv}$  layers are fit to linear problems to explore their general capabilities and limitations. Second, to show their performance in deep architectures,  $\mathrm{P^4Inv}$  blocks are used in combination with the Bent identity to perform density estimation of common two-dimensional distributions. Third, to study the generative performance of normalizing flows that use  $\mathrm{P^4Inv}$  blocks, we train a RealNVP normalizing flow with  $\mathrm{P^4}$  swaps as a Boltzmann generator (Noé et al., 2019). One important feature of this test problem is the availability of a ground truth energy function that is highly sensitive to any numerical problems in the network inversion.

# 4.1 LINEAR PROBLEMS

Fitting linear layers to linear training data is trivial in principle using basic linear algebra methods. However, the optimization with stochastic gradient descent at a small learning rate will help illuminate some important capabilities and limitations of  $\mathrm{P^4Inv}$  layers. It will also help answer the open question if gradient-based optimization of an invertible matrix  $\mathbf{A}$  allows crossing the ill-conditioned regime  $\{\mathbf{A} \in \mathbb{R}^{n \times n} : \det \mathbf{A} = 0\}$ . Furthermore, the training efficiency of perturbation updates can be compared to arbitrary linear layers that are optimized without perturbations.

Specifically, each target problem is defined by a quadratic matrix  $\pmb{T}$ . The training data is generated by sampling random vectors  $\pmb{x}$  and computing targets  $\pmb{y} = \pmb{T}\pmb{x}$ . Linear layers are then initialized as the identity matrix  $\pmb{A} \coloneqq \pmb{I}$  and the loss function  $J(\pmb{A}) = \mathbb{E}\| \pmb{A}\pmb{x} - \pmb{y}\|^2$  is minimized in three ways:

1. by directly updating the matrix elements (standard training of linear layers),  
2. through  $\mathrm{P^4Inv}$  updates, and  
3. through the inverses of  $\mathrm{P^4Inv}$  updates, i.e., by training  $\mathbf{A}$  through the updates in equation 3.

![](images/a157a9ea8da61222ba3597d8751439133cea9d328fe5d891ead4fa6c1bef0152.jpg)  
Figure 2: Training towards a 32-dimensional positive definite target matrix  $\pmb{T}$ . Left: Losses during training. Right: Eigenvalues during training. Final eigenvalues are shown as red crosses. Eigenvalues of the target matrix are shown as black squares.

![](images/055e78e9c4466ede72b3279e8a37365c101248f8cf8e0a7915c2ea7cc591b862.jpg)

The first linear problem is a 32-dimensional positive definite matrix with eigenvalues close to 1. Figure 2 shows the evolution of eigenvalues and losses during training. All three methods of optimization successfully recovered the target matrix. While training  $\mathrm{P^4Inv}$  via the inverse led to slower convergence, the forward training of  $\mathrm{P^4Inv}$  converged in the same number of iterations as an unconstrained linear layer for a merge interval  $N = 1$ . Increasing the merge interval to  $N = 10$  only affected the convergence minimally. Even for  $N = 50$ , the optimizer took only twice as many iterations as for an unconstrained linear layer.

![](images/9bb7407f931355557bcbc863461dacac09f3a7b07028abba64c29a5dde7ddfd6.jpg)  
Figure 3: Training towards a orthogonal target matrix  $T \in \mathrm{SO}(128)$ . Left: Losses during training. Right: Eigenvalues during training. Final eigenvalues are shown as red crosses. Eigenvalues of the target matrix are shown as black squares.

![](images/a2168968dbf5d3f15d1fb276913cab7963da38c0ed71bd5a61c62a9c22e821f5.jpg)

The second target matrix was a 128-dimensional special orthogonal matrix. As shown in Figure 3, the direct optimization converged to the target matrix in a linear fashion. In contrast, the matrices generated by the  $\mathrm{P^4Inv}$  update avoided the region around the origin. This detour led to a slower convergence in the initial phase of the optimization. Notably, the inverse stayed accurate up to 5 decimals throughout training. Training an inverse  $\mathrm{P^4Inv}$  was not successful for this example. This shows that the inverse  $\mathrm{P^4Inv}$  update can easily get stuck in local minima. This is not surprising as the elements of the inverse (equation 3) are parameterized by  $\mathbb{R}^{2n}$ -dimensional rational quadratic functions. When training against linear training data with a unique global optimum and a constant descent direction, the multimodality can prevent convergence. When training against more complex target data, no such problems were encountered. However, this result suggests that the efficiency of the optimization may be perturbed by very complex nonlinear parameterizations.

The final target matrix was  $B = -I_{101}$ , a matrix with determinant -1. In order to iteratively converge to the target matrix, the set of singular matrices has to be crossed. As expected, using a nonzero penalty parameter prevented the  $\mathrm{P^4Inv}$  update from converging to the target. However, when no penalty was applied, the matrix converged almost as fast as the usual linear training, see

# 4.2 2D DISTRIBUTIONS

The next step was to assess the effectiveness of  $\mathrm{P^4Inv}$  layers in deep networks. This was particularly important to rule out a potentially harmful accumulation of rounding errors. Density estimation of common 2D toy distributions was performed by stacking  $\mathrm{P^4Inv}$  layers with Bent identities and their inverses. For comparison, an RNVP flow was constructed with the same number of tunable parameters as the  $\mathrm{P^4Inv}$  flow, see Appendix G for details.

![](images/49a03737517974f22e588b580794a2890af4f46137501ba37394b0f7d21cdef5.jpg)  
Figure 4. When the determinant approached zero, inversion became ill-conditioned and residues increased. However, after reaching the other side, the inverse was quickly recovered up to 5 decimal digits. Notably, the determinant also converged to the correct value despite never being explicitly corrected.  
Figure 5: Density estimation for two-dimensional distributions from RealNVP (RNVP) and  $\mathrm{P^4Inv}$  networks with similar numbers of tunable parameters.

Figure 5 compares the generated distributions from the two models. The samples from the  $\mathrm{P^4Inv}$  model aligned favorably with the ground truth. In particular, it reproduced the multimodality of the data. In contrast to RNVP,  $\mathrm{P^4Inv}$  cleanly separated the modes, which underlines the favorable mixing achieved by general linear layers with elementwise nonlinear activations.

# 4.3 BOLTZMANN GENERATORS OF ALANINE DIPEPTIDE

Boltzmann generators (Noé et al., 2019) combine normalizing flows with statistical mechanics in order to draw direct samples from a given target density, e.g. given by a many-body physics system. This setup is ideally suited to assess the inversion of normalizing flows, since it with a given physical potential energy the target density is defined and we therefore have a quantitative measure for the sample quality. In molecular examples, specifically, the target densities are multimodal, contain singularities, and are highly sensitive to small perturbations in the atomic positions. Therefore, the generation of the 66-dimensional alanine dipeptide conformations is a highly nontrivial test for generative models.

The training efficiency and expressiveness of Boltzmann Generators (see Appendix E for details) were compared between pure RNVP baseline models as used in Noé et al. and models augmented

by  $\mathrm{P^4Inv}$  swaps as previously shown in Figure 7. The deep neural network architecture and training strategy are described in Appendix H. Both flows had 25 blocks as from Figure 7 in the appendix, resulting in 735,050 RNVP parameters. In contrast, the  $\mathrm{P^4Inv}$  blocks had only 9,000 tunable parameters. Due to this discrepancy and the depth of the network, we cannot expect dramatic improvements from adding  $\mathrm{P^4Inv}$  swaps. However, significant numerical errors in the inversion would definitely show in such a setup due to the highly sensitive potential energy.

![](images/de36fb94cdd8531ad33b4b14509e0ee045583c5aa299094728b4404e566fe123.jpg)  
Figure 6: Left: Energy distributions of generated samples; the second (orange) violin plot shows energies when the training data was perturbed by normal distributed random noise with  $0.004\mathrm{nm}$  standard deviation. The low-energy fraction for each column denotes the fraction of samples that had potential energy  $u$  lower than the maximum energy from the training set  $(\approx 120\mathrm{kJ / mol})$ . Right: Joint marginal distribution of the backbone torsions  $\varphi$  and  $\psi$ : training data compared to samples from RealNVP Boltzmann generators with and without  $\mathrm{P^4Inv}$  swaps (denoted  $P^4\text{Inv}$  and  $RNVP$ , respectively).

![](images/b5cfc181d6cdaacb743d3fe1db074cf9ed6565cba489cc0b47f555802b3aaae0.jpg)

Figure 6 (left) shows the energy statistics of generated samples. To demonstrate the sensitivity of the potential energy, the training data was first perturbed by  $0.004\mathrm{nm}$  (less than  $1\%$  of the total length of the molecule) and energies were evaluated for the perturbed data set. As a consequence, the mean of the potential energy distribution increased by  $80\mathrm{kJ/mol}$ .

In comparison, the Boltzmann generators produced much more accurate samples. The energy distributions from RNVP and  $\mathrm{P^4Inv}$  blocks were only shifted upward by  $\approx 15\mathrm{kJ / mol}$  and rarely generated samples with infeasibly large energies. The performance of both models was comparable with slight advantages for models with  $\mathrm{P^4Inv}$  swaps. This shows that the  $\mathrm{P^4Inv}$  inverses remained intact during training. Finally, Figure 6 (right) shows the joint distribution of the two backbone torsions. Both Boltzmann generators reproduced the most important local minima of the potential energy. As in the 2D toy problems, the  $\mathrm{P^4Inv}$  layers provided a cleaner separation of modes.

# 5 CONCLUSIONS

We have introduced  $\mathrm{P^4}$  updates, a novel algorithmic concept to preserve properties of neural networks using parameterized perturbations. As an example for this concept, general invertible linear layers  $(\mathrm{P^4Inv})$  were trained with stochastic optimizers while efficiently keeping track of their inverses and determinants. Applications to normalizing flows proved the accuracy of the inverses and determinants during training.

A crucial aspect of the  $P^4$  method is its decoupled merging step, which allows stable and efficient updates. As a consequence, the invertible linear  $\mathrm{P}^4\mathrm{Inv}$  layers can approximate any well-conditioned regular matrix. This feature might open up new avenues to parameterize useful subsets of  $\operatorname{GL}(n)$  through penalty functions.

Since perturbation theorems like the rank-one update exist for many classes of linear and nonlinear functions, we believe that the  $\mathbf{P}^4$  concept presents an efficient and widely applicable way of preserving desirable network properties during training.

# REFERENCES

Jens Behrmann, Will Grathwohl, Ricky TQ Chen, David Duvenaud, and Jorn-Henrik Jacobsen. Invertible residual networks. arXiv preprint arXiv:1811.00995, 2018.  
Rianne van den Berg, Leonard Hasenclever, Jakub M Tomczak, and Max Welling. Sylvester normalizing flows for variational inference. arXiv preprint arXiv:1803.05649, 2018.  
Denis Boyda, Gurtej Kanwar, Sébastien Racanière, Danilo Jimenez Rezende, Michael S Albergo, Kyle Cranmer, Daniel C Hackett, and Phiala E Shanahan. Sampling using  $su(n)$  gauge equivariant flows. arXiv preprint arXiv:2008.05456, 2020.  
Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in neural information processing systems, pp. 6571-6583, 2018.  
Tian Qi Chen, Jens Behrmann, David K Duvenaud, and Jorn-Henrik Jacobsen. Residual flows for invertible generative modeling. In Advances in Neural Information Processing Systems, pp. 9913-9923, 2019.  
Krzysztof Choromanski, David Cheikhi, Jared Davis, Valerii Likhosherstov, Achille Nazaret, Achraf Bahamou, Xingyou Song, Mrugank Akarte, Jack Parker-Holder, Jacob Bergquist, et al. Stochastic flows and geometric optimization on the orthogonal group. arXiv preprint arXiv:2003.13563, 2020.  
Nicola De Cao, Ivan Titov, and Wilker Aziz. Block neural autoregressive flow. arXiv preprint arXiv:1904.04676, 2019.  
Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.  
Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In Advances in Neural Information Processing Systems, pp. 7509-7520, 2019.  
Luca Falorsi, Pim de Haan, Tim R Davidson, and Patrick Forre. Reparameterizing distributions on lie groups. arXiv preprint arXiv:1903.02958, 2019.  
Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. Made: Masked autoencoder for distribution estimation. In International Conference on Machine Learning, pp. 881-889, 2015.  
Adam Golinski, Mario Lezcano-Casado, and Tom Rainforth. Improving normalizing flows via better orthogonal parameterizations. In ICML Workshop on Invertible Neural Networks and Normalizing Flows, 2019.  
Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael Cree. Regularisation of Neural Networks by Enforcing Lipschitz Continuity. apr 2018. URL http://arxiv.org/abs/1804.04368.  
Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367, 2018.  
Luigi Gresele, Giancarlo Fissore, Adrián Javaloy, Bernhard Schölkopf, and Aapo Hyvarinen. Relative gradient optimization of the jacobian term in unsupervised deep learning. arXiv preprint arXiv:2006.15090, 2020.  
Kyle Helfrich, Devin Willmott, and Qiang Ye. Orthogonal recurrent neural networks with scaled cayley transform. In International Conference on Machine Learning, pp. 1969-1978. PMLR, 2018.  
Jan Hermann, Zeno Schatzle, and Frank Noé. Deep-neural-network solution of the electronic Schrödinger equation. Nat. Chem., 12(10):891-897, sep 2020. ISSN 17554349. doi: 10.1038/s41557-020-0544-y.

Emiel Hoogeboom, Rianne van den Berg, and Max Welling. Emerging convolutions for generative normalizing flows. arXiv preprint arXiv:1901.11137, 2019.  
Emiel Hoogeboom, Victor Garcia Satorras, Jakub M Tomczak, and Max Welling. The convolution exponential and generalized sylvester flows. arXiv preprint arXiv:2006.01910, 2020.  
Chin-Wei Huang, David Krueger, Alexandre Lacoste, and Aaron Courville. Neural autoregressive flows. arXiv preprint arXiv:1804.00779, 2018.  
Gurtej Kanwar, Michael S. Albergo, Denis Boyda, Kyle Cranmer, Daniel C. Hackett, Sebastien Racanière, Danilo Jimenez Rezende, and Phiala E. Shanahan. Equivariant flow-based sampling for lattice gauge theory. Phys. Rev. Lett., 125:121601, Sep 2020. doi: 10.1103/PhysRevLett.125.121601. URL https://link.aps.org/doi/10.1103/PhysRevLett.125.121601.  
Mahdi Karami, Dale Schuurmans, Jascha Sohl-Dickstein, Laurent Dinh, and Daniel Duckworth. Invertible convolutional flow. In Advances in Neural Information Processing Systems, pp. 5635-5645, 2019.  
Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible  $1 \times 1$  convolutions. In Advances in neural information processing systems, pp. 10215-10224, 2018.  
Jonas Köhler, Leon Klein, and Frank Noé. Equivariant flows: exact likelihood generative learning for symmetric densities. arXiv preprint arXiv:2006.02425, 2020.  
Mario Lezcano-Casado and David Martínez-Rubio. Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group. arXiv preprint arXiv:1901.08428, 2019.  
Jun Li, Li Fuxin, and Sinisa Todorovic. Efficient riemannian optimization on the stiefel manifold via the Cayley transform. arXiv preprint arXiv:2002.01113, 2020.  
Christos Louizos and Max Welling. Multiplicative normalizing flows for variational bayesian neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 2218-2227. JMLR.org, 2017.  
Chenlin Meng, Yang Song, Jiaming Song, and Stefano Ermon. Gaussianization flows. arXiv preprint arXiv:2003.01941, 2020.  
Zakaria Mhammedi, Andrew Hellicar, Ashfaqur Rahman, and James Bailey. Efficient orthogonal parametrisation of recurrent neural networks using householder reflections. In International Conference on Machine Learning, pp. 2401-2409. PMLR, 2017.  
Thomas Müller, Brian McWilliams, Fabrice Rousselle, Markus Gross, and Jan Novák. Neural importance sampling. arXiv preprint arXiv:1808.03856, 2018.  
Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019.  
Aaron van den Oord, Yazhe Li, Igor Babuschkin, Karen Simonyan, Oriol Vinyals, Koray Kavukcuoglu, George van den Driessche, Edward Lockhart, Luis C Cobo, Florian Stimberg, et al. Parallel wavenet: Fast high-fidelity speech synthesis. arXiv preprint arXiv:1711.10433, 2017.  
George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. In Advances in Neural Information Processing Systems, pp. 2338-2347, 2017.  
George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. arXiv preprint arXiv:1912.02762, 2019.  
Tomas Pevny, Vasek Smidl, Martin Trapp, Ondrej Polacek, and Tomas Oberhuber. Sum-product-transform networks: Exploiting symmetries using invertible transformations. arXiv preprint arXiv:2005.01297, 2020.

David Pfau, James S. Spencer, Alexander G. D. G. Matthews, and W. M. C. Foulkes. Ab initio solution of the many-electron schrödinger equation with deep neural networks. Phys. Rev. Research, 2:033429, Sep 2020. doi: 10.1103/PhysRevResearch.2.033429. URL https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. arXiv preprint arXiv:1505.05770, 2015.  
Danilo Jimenez Rezende, Sébastien Racanière, Irina Higgins, and Peter Toth. Equivariant hamiltonian flows. arXiv preprint arXiv:1909.13739, 2019.  
Fazlollah Soleymani. A fast convergent iterative solver for approximate inverse of matrices. Numerical Linear Algebra with Applications, 21(3):439-452, 2014.  
Esteban G Tabak and Cristina V Turner. A family of nonparametric density estimation algorithms. Communications on Pure and Applied Mathematics, 66(2):145-164, 2013.  
Esteban G Tabak, Eric Vanden-Eijnden, et al. Density estimation by dual ascent of the log-likelihood. Communications in Mathematical Sciences, 8(1):217-233, 2010.  
Jakub M Tomczak and Max Welling. Improving variational auto-encoders using householder flow. arXiv preprint arXiv:1611.09630, 2016.  
Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability of deep learning, 2017.  
Linfeng Zhang, Lei Wang, et al. Monge-ampere flow for generative modeling. arXiv preprint arXiv:1809.10188, 2018.
