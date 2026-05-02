# GEODIFF: A GEOMETRIC DIFFUSION MODEL FOR MOLECULAR CONFORMATION GENERATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Predicting molecular conformations from molecular graphs is a fundamental problem in cheminformatics and drug discovery. Recently, significant progress has been achieved with machine learning approaches, especially with deep generative models. Inspired by the diffusion process in classical non-equilibrium thermodynamics where heated particles will diffuse from original states to a noise distribution, in this paper, we propose a novel generative model named GEODIFF for molecular conformation prediction. GEODIFF treats each atom as a particle and learns to directly reverse the diffusion process (i.e., transforming from a noise distribution to stable conformations) as a Markov chain. Modeling such a generation process is however very challenging as the likelihood of conformations should be roto-translational invariant. We theoretically show that Markov chains evolving with equivariant Markov kernels can induce an invariant distribution by design, and further propose building blocks for the Markov kernels to preserve the desirable equivariance property. The whole framework can be efficiently trained in an end-to-end fashion by optimizing a weighted variational lower bound to the (conditional) likelihood. Experiments on multiple benchmarks show that GEODIFF is superior or comparable to existing state-of-the-art approaches, especially on large molecules.

# 1 INTRODUCTION

Graph representation learning has achieved huge success for molecule modeling in various tasks ranging from property prediction (Gilmer et al., 2017; Duvenaud et al., 2015) to molecule generation (Jin et al., 2018; Shi et al., 2020), where typically a molecule is represented as an atom-bond graph. Despite its effectiveness in various applications, a more intrinsic and informative representation for molecules is the 3D geometry, also known as conformation, where atoms are represented as their Cartesian coordinates. The 3D structures determine the biological and physical properties of molecules and hence play a key role in many applications such as computational drug and material design (Thomas et al., 2018; Gebauer et al., 2021; Jing et al., 2021; Batzner et al., 2021). Unfortunately, how to predict stable molecular conformation remains a challenging problem. Traditional methods based on molecular dynamics (MD) or Markov chain Monte Carlo (MCMC) are very computationally expensive, especially for large molecules (Hawkins, 2017).

Recently, significant progress has been made with machine learning approaches, especially with deep generative models. For example, Simm & Hernandez-Lobato (2020); Xu et al. (2021a) studied predicting atomic distances with variational autoencoders (VAEs) (Kingma & Welling, 2013) and flow-based models (Dinh et al., 2017) respectively. Shi et al. (2021) proposed to use denoising score matching (Song & Ermon, 2019; 2020) to estimate the gradient fields over atomic distances, through which the gradient fields over atomic coordinates can be calculated. Ganea et al. (2021) studied generating conformations by predicting both bond lengths and angles. As molecular conformations are roto-translational invariant, these approaches circumvent directly modeling atomic coordinates by leveraging intermediate geometric variables such as atomic distances, bond and torsion angles, which are roto-translational invariant. As a result, they are able to achieve very compelling performance. However, as all these approaches seek to indirectly model the intermediate geometric variables, they have inherent limitations in either training or inference process (see Sec. 2 for a detailed description). Therefore, an ideal solution would still be directly modeling the atomic coordinates and at the same time taking the roto-translational invariance property into account.

In this paper, we propose such a solution called GEODIFF, a principled probabilistic framework based on denoising diffusion models (Sohl-Dickstein et al., 2015). Our approach is inspired by the diffusion process in nonequilibrium thermodynamics (De Groot & Mazur, 2013). We view atoms as particles in a thermodynamic system, which gradually diffuse from the original states to a noisy distribution in contact with a heat bath. At each time step, stochastic noises are added to the atomic positions. Our high-level idea is learning to reverse the diffusion process, which recovers the target geometric distribution from the noisy distribution. In particular, inspired by recent progress of denoising diffusion models on image generation (Ho et al., 2020; Song et al., 2020), we view the noisy geometries at different timesteps as latent variables, and formulate both the forward diffusion and reverse denoising process as Markov chains. Our goal is to learn the transition kernels such that the reverse process can recover realistic conformations from the chaotic positions sampled from a noise distribution. However, extending existing methods to geometric generation is highly non-trivial: a direct application of diffusion models on the conformation generation task leads to poor generation quality. As mentioned above, molecular conformations are roto-translational invariant, i.e., the estimated (conditional) likelihood should be unaffected by translational and rotational transformations (Kohler et al., 2020). To this end, we first theoretically show that a Markov process starting from an roto-translational invariant prior distribution and evolving with roto-translational equivariant Markov kernels can induce an roto-translational invariant density function. We further provide practical parameterization to define a roto-translational invariant prior distribution and a Markov kernel imposing the equivariance constraints. In addition, we derive a weighted variational lower bound of the conditional likelihood of molecular conformations, which also enjoys the roto-translational invariance and can be efficiently optimized.

A unique strength of GEODIFF is that it directly acts on the atomic coordinates and entirely bypasses the usage of intermediate elements for both training and inference. This general formulation enjoys several crucial advantages. First, the model can be naturally trained end-to-end without involving any sophisticated techniques like bilevel programming (Xu et al., 2021b), which benefits from small optimization variances. Besides, instead of solving geometries from bond lengths or angles, the one-stage sampling fashion avoids accumulating any intermediate error, and therefore leads to more accurate predicted structures. Moreover, GEODIFF enjoys a high model capacity to approximate the complex distribution of conformations. Thus, the model can better estimate the highly multi-modal distribution and generate structures with high quality and diversity.

We conduct comprehensive experiments on multiple benchmarks, including conformation generation and property prediction tasks. Numerical results show that GEODIFF consistently outperforms existing state-of-the-art machine learning approaches, and by a large margin on the more challenging large molecules. The significantly superior performance demonstrate the high capacity to model the complex distribution of molecular conformations and generate both diverse and accurate molecules.

# 2 RELATED WORK

Recently, various deep generative models have been proposed for conformation generation. Among them, CVGAE (Mansimov et al., 2019) first proposed a VAE model to directly generate 3D atomic coordinates, which fails to preserve the roto-translation equivariance property of conformations and suffers from poor performance. To address this problem, the majority of subsequent models are based on intermediate geometric elements such as atomic distances and torsion angles. A favorable property of these elements is the roto-translational invariance, (e.g. atomic distances does not change when rotating the molecule), which has been shown to be an important inductive bias for molecular geometry modeling (Köhler et al., 2020). However, such a decomposition suffers from several drawbacks for either training or sampling. For example, GRAPHDG (Simm & Hernandez-Lobato, 2020) and CGCF (Xu et al., 2021a) proposed to predict the interatomic distance matrix by VAE and Flow respectively, and then solve the geometry through the Distance Geometry (DG) technique (Liberti et al., 2014), which searches reasonable coordinates that matches with the predicted distances. CONFVAE further improves this pipeline by designing an end-to-end framework via bilevel optimization (Xu et al., 2021b). However, all these approaches suffer from the accumulated error problem, meaning that the noise in the predicted distances will misguide the coordinate searching process and lead to inaccurate or even erroneous structures. To overcome this problem, CONFGF (Shi et al., 2021) proposed to learn the gradient of the log-likelihood w.r.t coordinates. However, in practice the model is still aided by intermediate geometric elements, in that it first estimates the gradient w.r.t

interatomic distances via denoising score matching (DSM) (Song & Ermon, 2019; 2020), and then derives the gradient of coordinates using the chain rule. The problem is, by learning the distance gradient via DSM, the model is fed with perturbed distance matrices, which may violate the triangular inequality or even contain negative values. As a consequence, the model is actually learned over invalid distance matrices but tested with valid ones calculated from coordinates, making it suffer from serious out-of-distribution (Hendrycks & Gimpel, 2016) problem. Most recently, Ganea et al. (2021) proposed a highly systematic (rule-based) pipeline named GEOMOL, which learns to predict a set of geometric quantities (i.e. length and angles) and then reconstruct the local and global structures of the conformation in a sophisticated procedure. Despite its effectiveness, this method can be only adopted to molecules with a specific pattern, which limits it from wide applications. In our experiments, the pattern requirement filtered out nearly one third of molecular graphs, making it incomparable in our setting. Besides, there has also been efforts to use reinforcement learning for conformation search Gogineni et al. (2020). Nevertheless, this method relies on rigid rotor approximation and can only model the torsion angles, and thus fundamentally differs from other approaches.

# 3 PRELIMINARIES

# 3.1 NOTATIONS AND PROBLEM DEFINITION

Notations. In this paper each molecule with  $n$  atoms is represented as an undirected graph  $\mathcal{G} = \langle \mathcal{V},\mathcal{E}\rangle$  where  $\mathcal{V} = \{v_i\}_{i = 1}^n$  is the set of vertices representing atoms and  $\mathcal{E} = \{e_{ij}\mid (i,j)\subseteq |\mathcal{V}|\times |\mathcal{V}|\}$  is the set of edges representing inter-atomic bonds. Each node  $v_{i}\in \mathcal{V}$  describes the atomic attributes, e.g., the element type. Each edge  $e_{ij}\in \mathcal{E}$  describes the corresponding connection between  $v_{i}$  and  $v_{j}$ , and is labeled with its chemical type. In addition, we also assign the unconnected edges with a virtual type. For the geometry, each atom in  $\mathcal{V}$  is embedded by a coordinate vector  $\pmb {c}\in \mathbb{R}^3$  into the 3-dimensional space, and the full set of positions (i.e., the conformation) can be represented as a matrix  $\mathcal{C} = [c_1,c_2,\dots ,c_n]\in \mathbb{R}^{n\times 3}$ .

Problem Definition. The task of molecular conformation generation is a conditional generative problem, where we are interested in generating stable conformations for a provided graph  $\mathcal{G}$ . Given multiple graphs  $\mathcal{G}$ , and for each  $\mathcal{G}$  given its conformations  $\mathcal{C}$  as i.i.d samples from an underlying Boltzmann distribution (Noé et al., 2019), our goal is learning a generative model  $p_{\theta}(\mathcal{C}|\mathcal{G})$ , which is easy to draw samples from, to approximate the Boltzmann function.

# 3.2 EQUIVARIANCE

Equivalence is ubiquitous in machine learning for atomic systems, e.g., the vectors of atomic dipoles or forces should rotate accordingly w.r.t. the conformation coordinates (Thomas et al., 2018; Weiler et al., 2018; Fuchs et al., 2020; Miller et al., 2020; Simm et al., 2021; Batzner et al., 2021). It has been shown effective to integrate such inductive bias into model parameterization for modeling 3D geometry, which is critical for the generalization capacity (Köhler et al., 2020; Satorras et al., 2021a). Formally, a function  $\mathcal{F}:\mathcal{X}\to \mathcal{Y}$  is equivariant w.r.t a group  $G$  if:

$$
\mathcal {F} \circ T _ {g} (x) = S _ {g} \circ \mathcal {F} (x), \tag {1}
$$

where  $T_{g}$  and  $S_{g}$  are transformations for an element  $g \in G$ , acting on the vector spaces  $\mathcal{X}$  and  $\mathcal{Y}$ , respectively. In this work, we consider the SE(3) group, i.e., the group of rotation, translation in 3D space. This requires the estimated likelihood unaffected with translational and rotational transformations, and we will elaborate on how our method satisfy this property in Sec. 4.

# 4 GEODIFF METHOD

In this section, we elaborate on the proposed equivariant diffusion framework. We first present a high level description of our 3D diffusion formulation in Sec. 4.1, based on recent progress of denoising diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020). Then we emphasize several non-trivial challenges of building diffusion models for geometry generation scenario, and show how we technically tackle these issues. Specifically, in Sec. 4.2, we present how we parameterize  $p_{\theta}(\mathcal{C}|\mathcal{G})$  so that the conditional likelihood is roto-translational invariant, and in Sec. 4.3, we introduce our surgery of the training objective to make the optimization also invariant of translation and rotation. Finally, we briefly show how to draw samples from our model in Sec. 4.4.

![](images/1485814bf9f9a6eb7595ea9cae4793622ad3ab6b801f14eb58995518b4517a29.jpg)  
Figure 1: Illustration of the diffusion and reverse process of GEODIFF. For diffusion process, noise from fixed posterior distributions  $q(\mathcal{C}^t | \mathcal{C}^{t-1})$  is gradually added until the conformation is destroyed. Symmetrically, for generative process, an initial state  $\mathcal{C}^T$  is sampled from standard Gaussian distribution, and the conformation is progressively refined via the Markov kernels  $p_{\theta}(\mathcal{C}^{t-1} | \mathcal{G}, \mathcal{C}^t)$ .

# 4.1 FORMULATION

Let  $\mathcal{C}^0$  denotes the ground truth conformations and let  $\mathcal{C}^t$  for  $t = 1,\dots ,T$  be a sequence of latent variables with the same dimension, where  $t$  is the index for diffusion steps. Then a diffusion probabilistic model (Sohl-Dickstein et al., 2015) can be described as a latent variable model with two processes: the forward diffusion process, and the reverse generative process. Intuitively, the diffusion process progressively injects small noises to the data  $\mathcal{C}^0$ , while the generative process learns to revert the diffusion process by gradually eliminating the noise to recover the ground truth. We provide a high-level schematic of the processes in Fig. 1.

Diffusion process. Following the physical insight, we model the particles  $\mathcal{C}$  as an evolving thermodynamic system. With time going by, the equilibrium conformation  $\mathcal{C}^0$  will gradually diffuse to the next chaotic states  $\mathcal{C}^t$ , and finally converge into a white noise distribution after  $T$  iterations. Different from typical latent variable models, in diffusion model this forward process is defined as a fixed (rather than trainable) posterior distribution  $q(\mathcal{C}^{1:T}|\mathcal{C}^0)$ . Specifically, we define it as a Markov chain according to a fixed variance schedule  $\beta_1,\ldots ,\beta_T$ :

$$
q \left(\mathcal {C} ^ {1: T} \mid \mathcal {C} ^ {0}\right) = \prod_ {t = 1} ^ {T} q \left(\mathcal {C} ^ {t} \mid \mathcal {C} ^ {t - 1}\right), \quad q \left(\mathcal {C} ^ {t} \mid \mathcal {C} ^ {t - 1}\right) = \mathcal {N} \left(\mathcal {C} ^ {t}; \sqrt {1 - \beta_ {t}} \mathcal {C} ^ {t - 1}, \beta_ {t} I\right). \tag {2}
$$

Note that, in this work we do not impose specific (invariance) requirement upon the diffusion process, as long as it can efficiently draw noisy samples for training the generative process  $p_{\theta}(\mathcal{C}^0)$ .

Let  $\alpha_{t} = 1 - \beta_{t}$  and  $\bar{\alpha}_{t} = \prod_{s = 1}^{t}\alpha_{s}$ , a special property of the forward process is that  $q(\mathcal{C}^t |\mathcal{C}^0)$  of arbitrary timestep  $t$  can be calculated in closed form  $q(\mathcal{C}^t |\mathcal{C}^0) = \mathcal{N}(\mathcal{C}^t;\sqrt{\bar{\alpha}_t}\mathcal{C}^0,(1 - \bar{\alpha}_t)I)^1$ . This indicates with sufficiently large  $T$ , the whole forward process will convert  $\mathcal{C}^0$  to whitened isotropic Gaussian, and thus it is natural to set  $p(\mathcal{C}^T)$  as a standard Gaussian distribution.

Reverse Process. Our goal is learning to recover conformations  $\mathcal{C}^0$  from the white noise  $\mathcal{C}^T$ , given specified molecular graphs  $\mathcal{G}$ . We consider this generative procedure as a reverse dynamics of the above diffusion process, starting from the noisy particles  $\mathcal{C}^T \sim p(\mathcal{C}^T)$ . We formulate this reverse dynamics as a conditional Markov chain with learnable transitions:

$$
p _ {\theta} \left(\mathcal {C} ^ {0: T - 1} | \mathcal {G}, \mathcal {C} ^ {T}\right) = \prod_ {t = 1} ^ {T} p _ {\theta} \left(\mathcal {C} ^ {t - 1} | \mathcal {G}, \mathcal {C} ^ {t}\right), \quad p _ {\theta} \left(\mathcal {C} ^ {t - 1} | \mathcal {G}, \mathcal {C} ^ {t}\right) = \mathcal {N} \left(\mathcal {C} ^ {t - 1}; \mu_ {\theta} \left(\mathcal {G}, \mathcal {C} ^ {t}, t\right), \sigma_ {t} ^ {2} I\right). \tag {3}
$$

Herein  $\mu_{\theta}$  are parameterized neural networks to estimate the means, and  $\sigma_t$  can be any user-defined variance. The initial distribution  $p(\mathcal{C}^T)$  is set as a standard Gaussian. Given a graph  $\mathcal{G}$ , its 3D structure is generated by first drawing chaotic particles  $\mathcal{C}^T$  from  $p(\mathcal{C}^T)$ , and then iteratively refined through the reverse Markov kernels  $p_{\theta}(\mathcal{C}^{t - 1}|\mathcal{G},\mathcal{C}^t)$ .

Having formulated the reverse dynamics, the marginal likelihood can be calculated by  $p_{\theta}(\mathcal{C}^0|\mathcal{G}) = \int p(\mathcal{C}^T)p_{\theta}(\mathcal{C}^{0:T - 1}|\mathcal{G},\mathcal{C}^T)\mathrm{d}\mathcal{C}^{1:T}$ . Herein a non-trivial problem is that the likelihood should be invariant w.r.t translation and rotation, which has proved to be a critical inductive bias for 3D object generation (Köhler et al., 2020; Satorras et al., 2021a). In the following subsections, we will elaborate on how we parameterize the Markov kernels  $p_{\theta}(\mathcal{C}^{t - 1}|\mathcal{G},\mathcal{C}^t)$  to achieve this desired property, and also how to maximize this likelihood by taking the invariance into account.

# 4.2 EQUIVARIANT REVERSE GENERATIVE PROCESS

Instead of directly leveraging existing methods, we consider building the density  $p_{\theta}(\mathcal{C}^0)$  that is invariant to rotation and translation transformations. Intuitively, this requires the likelihood to be unaffected by translations and rotations. Formally, let  $T_g$  be some roto-translational transformations of a group element  $g \in \mathrm{SE}(3)$ , then we have the following statement:

Proposition 1. Let  $p(x_{T})$  be a SE(3)-invariant density function, i.e.,  $p(x_{T}) = p(T_{g}(x_{T}))$ . If Markov transitions  $p(x_{t - 1}|x_t)$  are SE(3)-equivariant, i.e.,  $p(x_{t - 1}|x_t) = p(T_g(x_{t - 1})|T_g(x_t))$ , then we have that the density  $p_{\theta}(x_0) = \int p(x_T)p_{\theta}(x_{0:T - 1}|x_T)\mathrm{d}\pmb{x}_{1:T}$  is also SE(3)-invariant.

This proposition indicates that the dynamics starting from an invariant standard density along an equivariant Gaussian Markov kernel can result in an invariant density. Now we provide a practical implementation of GEODIFF based on the recent denoising diffusion framework (Ho et al., 2020).

Invariant Initial Density  $p(\mathcal{C}^T)$ . We first introduce the invariant distribution  $p(\mathcal{C}^T)$ , which will also be employed in the equivariant Markov chain. We borrow the idea from Kohler et al. (2020) to consider systems with zero center of mass (CoM), termed CoM-free systems. We define  $p(\mathcal{C}^T)$  as a "CoM-free standard density"  $\hat{\rho}(\mathcal{C})$ , built upon an isotropic normal density  $\rho(\mathcal{C})$ : for evaluating the likelihood  $\hat{\rho}(\mathcal{C})$  we can firstly translate  $\mathcal{C}$  to zero CoM and then calculate  $\rho(\mathcal{C})$ , and for sampling from  $\hat{\rho}(\mathcal{C})$  we can first sample from  $\rho(\mathcal{C})$  and then move the CoM to zero.

We provide a formal theoretical analysis of  $\hat{\rho} (\mathcal{C})$  in Appendix A. Intuitively, the isotropic Gaussian is manifestly invariant to rotations around the zero CoM. And by considering CoM-free system, moving the particles to zero CoM can always ensure the translational invariance. Consequently,  $\hat{\rho} (\mathcal{C})$  is constructed as a roto-transitional invariant density.

Equivariant Markov Kernels  $p(\mathcal{C}^{t - 1}|\mathcal{G},\mathcal{C}^t)$ . Similar to the prior density, we also consider equipping transitions  $p(\mathcal{C}^{t - 1}|\mathcal{G},\mathcal{C}^t)$  with CoM-free Gaussians  $\hat{\rho} (\mathcal{C})$ . Specifically, given mean  $\mu_{\theta}(\mathcal{G},\mathcal{C}^t,t)$  and variance  $\sigma_t$ , the likelihood of  $\mathcal{C}^{t - 1}$  will be calculated by  $\hat{\rho} (\frac{\mathcal{C}^{t - 1} - \mu_{\theta}(\mathcal{G},\mathcal{C}^t,t)}{\sigma_t})$ . The CoM-free Gaussian ensures the translation invariance in the Markov kernels. Consequently, to achieve the equivariant property defined in Proposition 1, we focus on the rotation equivariance.

Then in general, the key requirement is to ensure the means  $\mu_{\theta}(\mathcal{G},\mathcal{C}^t,t)$  to be roto-translation equivariant w.r.t  $\mathcal{C}^t$ . Following Ho et al. (2020), we consider the following parameterization of  $\mu_{\theta}$ :

$$
\mu_ {\theta} \left(\mathcal {C} ^ {t}, t\right) = \frac {1}{\sqrt {\alpha_ {t}}} \left(\mathcal {C} ^ {t} - \frac {\beta_ {t}}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {\theta} \left(\mathcal {G}, \mathcal {C} ^ {t}, t\right)\right), \tag {4}
$$

where  $\epsilon_{\theta}$  are neural networks with trainable parameters  $\theta$ . Intuitively, the model  $\epsilon_{\theta}$  learns to predict the noise necessary to decorrupt the conformations. This is analogous to the physical force fields (Schütt et al., 2017; Zhang et al., 2018; Hu et al., 2021; Shuaibi et al., 2021), which also gradually push particles towards convergence around the equilibrium states.

Now the problem is transformed to constructing  $\epsilon_{\theta}$  to be roto-translational equivariant. We draw inspirations from recent equivariant networks (Thomas et al., 2018; Satorras et al., 2021b) to design an equivariant convolutional layer, named graph field network (GFN). In the  $l$ -th layer, GFN takes node embeddings  $\mathbf{h}^l \in \mathbb{R}^{n \times b}$  ( $b$  denotes the feature dimension) and corresponding coordinate embeddings  $\mathbf{x}^l \in \mathbb{R}^{n \times 3}$  as inputs, and outputs  $\mathbf{h}^{l+1}$  and  $\mathbf{x}^{l+1}$  as follows:

$$
\mathbf {m} _ {i j} = \Phi_ {m} \left(\mathbf {h} _ {i} ^ {l}, \mathbf {h} _ {j} ^ {l}, \left\| \mathbf {x} _ {i} ^ {l} - \mathbf {x} _ {j} ^ {l} \right\| ^ {2}, e _ {i j}; \theta_ {m}\right) \tag {5}
$$

$$
\mathbf {h} _ {i} ^ {l + 1} = \Phi_ {h} \left(\mathbf {h} _ {i} ^ {l}, \sum_ {j \in \mathcal {N} (i)} \mathbf {m} _ {i j}; \theta_ {h}\right) \tag {6}
$$

$$
\mathbf {x} _ {i} ^ {l + 1} = \sum_ {j \in \mathcal {N} (i)} \frac {1}{d _ {i j}} \left(\mathbf {c} _ {i} - \mathbf {c} _ {j}\right) \Phi_ {x} \left(\mathbf {m} _ {i j}; \theta_ {x}\right) \tag {7}
$$

where  $\Phi$  are feed-forward networks and  $d_{ij}$  denotes interatomic distances.  $\mathcal{N}(i)$  denotes the neighborhood of  $i^{th}$  node, including both connected atoms and other ones within a radius threshold  $\tau$ , which enables the model to explicitly capture long-range interactions and support molecular graphs with disconnected components. Initial embeddings  $\mathbf{h}^0$  are combinations of atom and timestep embeddings, and  $\mathbf{x}^0$  are atomic coordinates. The main difference between proposed GFN and other GNNs lies

in equation 7, where  $\mathbf{x}$  is updated as a combination of radial directions weighted by  $\Phi_x:\mathbb{R}^b\to \mathbb{R}$ . Such vector field  $\mathbf{x}^{L}$  enjoys the roto-translation equivariance property. Formally, we have:

Proposition 2. Parameterizing  $\epsilon_{\theta}(\mathcal{G},\mathcal{C},t)$  as a composition of  $L$  GFN layers, and take the  $\mathbf{x}^L$  after  $L$  updates as the output. Then the noise vector field  $\epsilon_{\theta}$  is SE(3) equivariant w.r.t the 3D system  $\mathcal{C}$ .

Intuitively, given  $\mathbf{h}^l$  already invariant and  $\mathbf{x}^l$  equivariant, the message embedding  $\mathbf{m}$  will also be invariant since it only depends on invariant features. Since  $\mathbf{x}$  is updated with the relative differences  $\mathbf{c}_i - \mathbf{c}_j$  weighted by invariant features, it will be translation-invariant and rotation-equivariant. Then inductively, composing  $\epsilon_{\theta}$  with  $L$  GFN layers enables equivariance with  $\mathcal{C}^t$ . We provide the formal proof of equivariance properties in Appendix A.

# 4.3 IMPROVED TRAINING OBJECTIVE

Having formulated the generative process and the model parameterization, now we consider the practical training objective for the reverse dynamics. Since directly optimizing the exact log-likelihood is intractable, we instead maximize the usual variational lower bound (ELBO) $^2$ :

$$
\begin{array}{l} \mathbb {E} \left[ \log p _ {\theta} (\mathcal {C} ^ {0} | \mathcal {G}) \right] = \mathbb {E} \left[ \log \mathbb {E} _ {q (\mathcal {C} ^ {1: T} | \mathcal {C} ^ {0})} \frac {p _ {\theta} (\mathcal {C} ^ {0 : T} | \mathcal {G})}{q (\mathcal {C} ^ {1 : T} | \mathcal {C} ^ {0})} \right] \\ \geq - \mathbb {E} _ {q} \left[ \sum_ {t = 1} ^ {T} D _ {\mathrm {K L}} \left(q \left(\mathcal {C} ^ {t - 1} \mid \mathcal {C} ^ {t}, \mathcal {C} ^ {0}\right) \| p _ {\theta} \left(\mathcal {C} ^ {t - 1} \mid \mathcal {C} ^ {t}, \mathcal {G}\right)\right) \right] := - \mathcal {L} _ {\mathrm {E L B O}} \tag {8} \\ \end{array}
$$

where  $q(\mathcal{C}^{t - 1}|\mathcal{C}^t,\mathcal{C}^0)$  is analytically tractable as  $\mathcal{N}(\frac{\sqrt{\bar{\alpha}_{t - 1}}\beta_t}{1 - \bar{\alpha}_t}\mathcal{C}^0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t - 1})}{1 - \bar{\alpha}_t}\mathcal{C}^t, \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}\beta_t)^2$ . Most recently, Ho et al. (2020) showed that under the parameterization in equation 4, the ELBO of the diffusion model can be further simplified by calculating the KL divergences between Gaussians as weighted  $\mathcal{L}_2$  distances between the means  $\epsilon_{\theta}$  and  $\epsilon^2$ . Formally, we have:

Proposition 3. (Ho et al., 2020) Under the parameterization in equation 4, we have:

$$
\mathcal {L} _ {\mathrm {E L B O}} = \sum_ {t = 1} ^ {T} \gamma_ {t} \mathbb {E} _ {\left\{\mathcal {C} ^ {0}, \mathcal {G} \right\}} \sim q \left(\mathcal {C} ^ {0}, \mathcal {G}\right), \epsilon \sim \mathcal {N} (0, I) \left[ \| \epsilon - \epsilon_ {\theta} (\mathcal {G}, \mathcal {C} ^ {t}, t) \| _ {2} ^ {2} \right] \tag {9}
$$

where  $\mathcal{C}^t = \sqrt{\bar{\alpha}_t}\mathcal{C}^0 +\sqrt{1 - \bar{\alpha}_t}\epsilon$ . The weights  $\gamma_{t} = \frac{\beta_{t}}{2\alpha_{t}(1 - \bar{\alpha}_{t - 1})}$  for  $t > 1$ , and  $\gamma_{1} = \frac{1}{2\alpha_{1}}$ .

The intuition of this objective is to independently sample chaotic conformations of different timesteps from  $q(\mathcal{C}^{t - 1}|\mathcal{C}^t,\mathcal{C}^0)$ , and use  $\epsilon_{\theta}$  to model the noise vector  $\epsilon$ . To yield a better empirical performance, Ho et al. (2020) suggests to set all weights  $\gamma_{t}$  as 1, which is in line with the objectives of recent noise conditional score networks (Song & Ermon, 2019; 2020).

As  $\epsilon_{\theta}$  is designed to be equivariant, it is natural to require its supervision signal  $\epsilon$  to be equivariant with  $\mathcal{C}^t$ . Note that once this is achieved, the ELBO will also become invariant. However, the  $\epsilon$  in the forward diffusion process is not imposed with such equivariance, violating the above properties. Here we propose two approaches to obtain the modified noise vector  $\hat{\epsilon}$ , which, after replacing  $\epsilon$  in the  $\mathcal{L}_2$  distance calculation in equation 9, achieves the desired equivariance:

Alignment approach. Considering the fact that  $\epsilon$  can be calculated by  $\frac{\mathcal{C}^t - \sqrt{\bar{\alpha}_t}\mathcal{C}^0}{\sqrt{1 - \bar{\alpha}_t}}$ , we can first rotate and translate  $\mathcal{C}^0$  to  $\hat{\mathcal{C}}^0$  by aligning w.r.t  $\mathcal{C}^t$ , and then compute  $\hat{\epsilon}$  as  $\frac{\mathcal{C}^t - \sqrt{\bar{\alpha}_t}\hat{\mathcal{C}}^0}{\sqrt{1 - \bar{\alpha}_t}}$ . Since the aligned conformation  $\hat{\mathcal{C}}^0$  is equivariant with  $\mathcal{C}^t$ , the processed  $\hat{\epsilon}$  will also enjoy the equivariance. Specifically, the alignment is implemented by first translating  $\mathcal{C}^0$  to the same CoM of  $\mathcal{C}^t$  and then solve the optimal rotation matrix by Kabsch alignment algorithm (Kabsch, 1976).

Chain-rule approach. Another meaningful observation is that by reparameterizing the Gaussian distribution  $q(\mathcal{C}^t | \mathcal{C}^0)$  as  $\mathcal{C}^t = \sqrt{\bar{\alpha}_t} \mathcal{C}^0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ ,  $\epsilon$  can be viewed as a weighted score function  $\sqrt{1 - \bar{\alpha}_t} \nabla_{\mathcal{C}^t} q(\mathcal{C}^t | \mathcal{C}^0)$ . Shi et al. (2021) recently shows that generally this score function  $\nabla_{\mathcal{C}^t} q(\mathcal{C}^t | \cdot)$  can be designed to be equivariant by decomposing it into  $\partial_{\mathcal{C}^t} \mathbf{d}^t \nabla_{\mathbf{d}^t} q(\mathcal{C}^t | \cdot)$  with the chain rule, where  $\mathbf{d}^t$  can be any invariant features of the structures  $\mathcal{C}^t$  such as the inter-atomic distances. We refer

readers to Shi et al. (2021) for more details. The insight is that as gradient of invariant variables  $w.r.t$  equivariant variables, the partial derivative  $\partial_{\mathcal{C}^t}\mathbf{d}^t$  will always be equivalent with  $\mathcal{C}^t$ . In this work, under the common assumption that  $\mathbf{d}$  also follows a Gaussian distribution (Kingma & Welling, 2013), our practical implementation is to first approximately calculate  $\nabla_{\mathbf{d}^t}q(\mathcal{C}^t|\mathcal{C}^0)$  as  $\frac{\mathbf{d}^t - \sqrt{\bar{\alpha}_t}\mathbf{d}^0}{1 - \bar{\alpha}_t}$ , and then compute the modified noise vector  $\hat{\epsilon}$  as  $\sqrt{1 - \bar{\alpha}_t}\partial_{\mathcal{C}^t}\mathbf{d}^t (\frac{\mathbf{d}^t - \sqrt{\bar{\alpha}_t}\mathbf{d}^0}{1 - \bar{\alpha}_t}) = \frac{\partial_{\mathcal{C}^t}\mathbf{d}^t\cdot(\mathbf{d}^t - \sqrt{\bar{\alpha}_t}\mathbf{d}^0)}{\sqrt{1 - \bar{\alpha}_t}}$ .

# 4.4 SAMPLING

With a learned reverse dynamics  $\epsilon_{\theta}(\mathcal{G},\mathcal{C}^t,t)$  , the transition means  $\mu_{\theta}(\mathcal{G},\mathcal{C}^{t},t)$  can be calculated by equation 4. Thus, given a graph  $\mathcal{G}$  , its geometry  $\mathcal{C}^0$  is generated by first sampling chaotic particles  $\mathcal{C}^T\sim p(\mathcal{C}^T)$  , and then progressively sample  $\mathcal{C}^{t - 1}\sim$ $p_{\theta}(\mathcal{C}^{t - 1}| \mathcal{G},\mathcal{C}^t)$  for  $t = T,T-$ $1,\dots ,1$  . This process is Markovian, which gradually shifts the previous noisy positions towards

# Algorithm 1 Sampling Algorithm of GEODIFF.

Input: the molecular graph  $\mathcal{G}$ , the learned reverse model  $\epsilon_{\theta}$ .

Output: the molecular conformation  $\mathcal{C}$

1: Sample  $\mathcal{C}^T\sim p(\mathcal{C}^T) = \mathcal{N}(0,I)$  
2: for  $s = T, T - 1, \dots, 1$  do  
3: Shift  $\mathcal{C}^s$  to zero CoM  
4: Compute  $\mu_{\theta}(\mathcal{C}^s,\mathcal{G},s)$  from  $\epsilon_{\theta}(\mathcal{C}^{s},\mathcal{G},s)$  using equation 4  
5: Sample  $\mathcal{C}^{s - 1}\sim \mathcal{N}(\mathcal{C}^{s - 1};\mu_{\theta}(\mathcal{C}^{s},\mathcal{G},s),\sigma_{t}^{2}I)$  
6: end for  
7: return  $\mathcal{C}^0$  as  $\mathcal{C}$

equilibrium states. We provide the pseudo code of the whole sampling process in Algorithm 1.

# 5 EXPERIMENT

In this section, we empirically evaluate GEODIFF on the task of equilibrium conformation generation for both small and drug-like molecules. Following existing work (Shi et al., 2021; Ganea et al., 2021), we test the proposed method as well as the competitive baselines on two standard benchmarks: Conformation Generation (Sec. 5.2) and Property Prediction (Sec. 5.3). We first present the general experiment setups, and then describe task-specific evaluation protocols and discuss the results in each section. The implementation details are provided in Appendix B.

# 5.1 EXPERIMENT SETUP

Datasets. Following prior works (Xu et al., 2021a;b), we also use the recent GEOM-QM9 (Ramakrishnan et al., 2014) and GEOM-Drugs (Axelrod & Gomez-Bombarelli, 2020) datasets. The former one contains small molecules while the latter one are medium-sized organic compounds. We borrow the data split produced by Shi et al. (2021). For both datasets, the training split consists of 40,000 molecules with 5 conformations for each, resulting in 200,000 conformations in total. The valid split share the same size as training split. The test split contains 200 distinct molecules, with 22,408 conformations for QM9 and 14,324 ones for Drugs.

Baselines. We compare GEODIFF with 6 recent or established state-of-the-art baselines. For the ML approaches, we test the following models with highest reported performance: CVGAE (Mansimov et al., 2019), GRAPHDG (Simm & Hernandez-Lobato, 2020), CGCF (Xu et al., 2021a), CONFVAE (Xu et al., 2021b) and CONFGF (Shi et al., 2021). We also test the classic RDKIT (Riniker & Landrum, 2015) method, which is arguably the most popular open-source software for conformation generation. Some other recent works (TorsionNet (Gogineni et al., 2020) and GEOMOL (Ganea et al., 2021)) are omitted since these approaches are not directly comparable in our setting. We refer readers to Sec. 2 for a detailed discussion of these models.

# 5.2 CONFORMATION GENERATION

Evaluation metrics. The task aims to measure both quality and diversity of generated conformations by different models. We follow Ganea et al. (2021) to evaluate 4 metrics built upon root-mean-square deviation (RMSD), which is defined as the normalized Frobenius norm of two atomic coordinates matrices, after alignment by Kabsch algorithm (Kabsch, 1976). Formally, let  $S_{g}$  and  $S_{r}$  denote the sets of generated and reference conformers respectively, then the Coverage and Matching metrics (Xu

Table 1: Results on the GEOM-Drugs dataset, without FF optimization.  

<table><tr><td rowspan="2">Models</td><td colspan="2">COV-R (%) ↑</td><td colspan="2">MAT-R (Å) ↓</td><td colspan="2">COV-P (%) ↑</td><td colspan="2">MAT-P (Å) ↓</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>CVGAE (ML)</td><td>10.37</td><td>0.00</td><td>1.950</td><td>1.933</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GRAPHDG (ML)</td><td>8.27</td><td>0.00</td><td>1.9722</td><td>1.9845</td><td>2.08</td><td>0.00</td><td>2.4340</td><td>2.4100</td></tr><tr><td>CGCF (ML)</td><td>53.95</td><td>57.05</td><td>1.2487</td><td>1.2247</td><td>21.68</td><td>13.72</td><td>1.8571</td><td>1.8066</td></tr><tr><td>CONFVAE (ML)</td><td>55.20</td><td>59.43</td><td>1.2380</td><td>1.1417</td><td>22.96</td><td>14.05</td><td>1.8287</td><td>1.8159</td></tr><tr><td>CONFGF (ML)</td><td>62.14</td><td>70.92</td><td>1.1629</td><td>1.1596</td><td>23.42</td><td>15.52</td><td>1.7219</td><td>1.6863</td></tr><tr><td>GEODIFF-A</td><td>88.36</td><td>96.09</td><td>0.8704</td><td>0.8628</td><td>60.14</td><td>61.25</td><td>1.1864</td><td>1.1391</td></tr><tr><td>GEODIFF-C</td><td>89.13</td><td>97.88</td><td>0.8629</td><td>0.8529</td><td>61.47</td><td>64.55</td><td>1.1712</td><td>1.1232</td></tr></table>

Table 2: Results on the GEOM-QM9 dataset, without FF optimization.  

<table><tr><td rowspan="2">Models</td><td colspan="2">COV-R (%) ↑</td><td colspan="2">MAT-R (Å) ↓</td><td colspan="2">COV-P (%) ↑</td><td colspan="2">MAT-P (Å) ↓</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>CVGAE (ML)</td><td>0.09</td><td>0.00</td><td>1.6713</td><td>1.6088</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GRAPHDG (ML)</td><td>73.33</td><td>84.21</td><td>0.4245</td><td>0.3973</td><td>43.90</td><td>35.33</td><td>0.5809</td><td>0.5823</td></tr><tr><td>CGCF (ML)</td><td>78.05</td><td>82.48</td><td>0.4206</td><td>0.3903</td><td>36.49</td><td>33.57</td><td>0.6615</td><td>0.6427</td></tr><tr><td>CONFVAE (ML)</td><td>77.84</td><td>88.20</td><td>0.4154</td><td>0.3739</td><td>38.02</td><td>34.67</td><td>0.6215</td><td>0.6091</td></tr><tr><td>CONFGF (ML)</td><td>88.49</td><td>94.31</td><td>0.2673</td><td>0.2685</td><td>46.43</td><td>43.41</td><td>0.5224</td><td>0.5124</td></tr><tr><td>GEODIFF-A</td><td>90.54</td><td>94.61</td><td>0.2104</td><td>0.2021</td><td>52.35</td><td>50.10</td><td>0.4539</td><td>0.4399</td></tr><tr><td>GEODIFF-C</td><td>91.68</td><td>95.82</td><td>0.2099</td><td>0.2026</td><td>53.67</td><td>52.00</td><td>0.4488</td><td>0.4299</td></tr></table>

et al., 2021a) following the conventional Recall measurement can be defined as:

$$
\operatorname {C O V} - \mathrm {R} \left(S _ {g}, S _ {r}\right) = \frac {1}{\left| S _ {r} \right|} \left| \left\{\mathcal {C} \in S _ {r} \mid \operatorname {R M S D} (\mathcal {C}, \hat {\mathcal {C}}) \leq \delta , \hat {\mathcal {C}} \in S _ {g} \right\} \right|, \tag {10}
$$

$$
\operatorname {M A T - R} \left(S _ {g}, S _ {r}\right) = \frac {1}{\left| S _ {r} \right|} \sum_ {\mathcal {C} \in S _ {r}} \min  _ {\hat {\mathcal {C}} \in S _ {g}} \operatorname {R M S D} \left(\mathcal {C}, \hat {\mathcal {C}}\right), \tag {11}
$$

where  $\delta$  is a pre-defined threshold. The other two metrics COV-P and MAT-P inspired by Precision can be defined similarly but with the generated and reference sets exchanged. In practice,  $S_{g}$  is set as twice of the size of  $S_{r}$  for each molecule. Intuitively, the COV scores measure the percentage of structures in one set covered by another set, where covering means the RMSD between two conformations is within a certain threshold  $\delta$ . By contrast, the MAT scores measure the average RMSD of conformers in one set with its closest neighbor in another set. In general, higher COV rates or lower MAT score suggest that more realistic conformations are generated. Besides, the Precision metrics depend more on the quality, while the Recall metrics concentrate more on the diversity. Either metrics can be more appealing considering the specific scenario.

Results & discussion. The results are reported in Tab. 1 and Tab. 2. As noted in Sec. 4.3, GEODIFF can be trained with two types of modified ELBO, named alignment and chain-rule approaches. We denote models learned by these two objectives as GEODIFF-A and GEODIFF-C respectively. As shown in the tables, GEODIFF consistently outperform the state-of-the-art ML models on all datasets and metrics, especially by a significant margin for more challenging large molecules (Drugs dataset). The results demonstrate the superior capacity of GEODIFF to model the multi modal distribution, and generative both accurate and diverse conformations. We also notice that in general GEODIFF +C performs slightly better than GEODIFF +A, which suggests that chain-rule approach leads to a better optimization procedure. We thus take GEODIFF +C as the representative in the following comparisons. We visualize samples generated by different models in Fig. 2 to provide a qualitative comparison, where GEODIFF is shown to capture better both local and global structures.

On the more challenging Drugs dataset, we further test RDKIT. As shown in Tab. 3, our observation is in line with previous studies (Shi et al., 2021) that the state-of-the-art ML models (shown in Tab. 1) perform better on COV-R and MAT-R. However, for the new Precision-based metrics we found that ML models are still not comparable. This indicates that ML models tend to explore more possible representatives while RDKIT concentrates on a few most common ones, prioritizes quality over diversity. Previous works (Mansimov et al., 2019; Xu et al., 2021b) suggest that this is because

Figure 2: Examples of generated structures from Drugs dataset. For every model, we show the conformation best-aligned with the ground truth. More examples are provided in Appendix C.  

<table><tr><td>Graph</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GeoDiff</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ConfGF</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphDG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Results on the GEOM-Drugs dataset, with FF optimization.  

<table><tr><td rowspan="2">Models</td><td colspan="2">COV-R (%) ↑</td><td colspan="2">MAT-R (Å) ↓</td><td colspan="2">COV-P (%) ↑</td><td colspan="2">MAT-P (Å) ↓</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>RDKIT</td><td>60.91</td><td>65.69</td><td>1.2026</td><td>1.1252</td><td>72.22</td><td>88.72</td><td>1.0976</td><td>0.9539</td></tr><tr><td>GEODIFF + FF</td><td>92.27</td><td>100.00</td><td>0.7618</td><td>0.7340</td><td>84.51</td><td>95.86</td><td>0.9834</td><td>0.9221</td></tr></table>

RDKIT involves an additional empirical force field (FF) (Halgren, 1996) to optimize the structure, and we follow them to also combine GEODIFF with FF to yield a more fair comparison. Results in Tab. 3 demonstrate that GEODIFF +FF can stay the superior diversity (Recall metrics) while also enjoy significantly improved accuracy ((Precision metrics)).

# 5.3 PROPERTY PREDICTION

Evaluation metrics. This task estimates the molecular ensemble properties (Axelrod & Gomez-Bombarelli, 2020) over a set of generated conformations. This can provide an direct assessment on the quality of generated samples. In specific, we follow (Shi et al., 2021) to extract a split from GEOM-QM9 covering 30 molecules, and generate 50 samples for each. Then we use

Table 4: MAE of predicted ensemble properties in eV.  

<table><tr><td>Method</td><td>E̅</td><td>Emin</td><td>Δε</td><td>Δεmin</td><td>Δεmax</td></tr><tr><td>RDKIT</td><td>0.9233</td><td>0.6585</td><td>0.3698</td><td>0.8021</td><td>0.2359</td></tr><tr><td>GRAPHDG</td><td>9.1027</td><td>0.8882</td><td>1.7973</td><td>4.1743</td><td>0.4776</td></tr><tr><td>CGCF</td><td>28.9661</td><td>2.8410</td><td>2.8356</td><td>10.6361</td><td>0.5954</td></tr><tr><td>CONFVAE</td><td>0.8208</td><td>0.6100</td><td>1.6080</td><td>3.9111</td><td>0.2429</td></tr><tr><td>CONFGF</td><td>2.7886</td><td>0.1765</td><td>0.4688</td><td>2.1843</td><td>0.1433</td></tr><tr><td>GEODIFF</td><td>0.2574</td><td>0.1550</td><td>0.1758</td><td>0.7032</td><td>0.1039</td></tr></table>

the chemical toolkit Ps14 (Smith et al., 2020) to calculate each conformer's energy  $E$  and HOMO-LUMO gap  $\epsilon$ , and compare the average energy  $\overline{E}$ , lowest energy  $E_{\mathrm{min}}$ , average gap  $\overline{\Delta\epsilon}$ , minimum gap  $\Delta \epsilon_{\mathrm{min}}$ , and maximum gap  $\Delta \epsilon_{\mathrm{max}}$  with the ground truth.

Results & discussions. The mean absolute errors (MAE) between calculated properties and the ground truth are reported in Tab. 4. CVGAE is excluded due to the poor performance, which is also reported in Simm & Hernandez-Lobato (2020); Shi et al. (2021). The properties are highly sensitive to geometric structure, and thus the superior performance demonstrate that GEODIFF can consistently predict more accurate conformations across different molecules.

# 6 CONCLUSION

We propose GEODIFF, a novel probabilistic model for generating molecular conformations. GEODIFF marries denoising diffusion models with geometric representations, where we parameterize the reverse generative dynamics as a Markov chain, and Novelly impose roto-translational invariance into the density with equivariant Markov kernels. We derive a tractable invariant objective from the variational lower bound to optimize the likelihood. Comprehensive experiments over multiple tasks demonstrate that GEODIFF is competitive with the existing state-of-the-art models. Future work includes further improving or accelerating the model with other recent progress of diffusion models, and extending our method to other challenging structures such as proteins.

# REFERENCES

Simon Axelrod and Rafael Gomez-Bombarelli. Geom: Energy-annotated molecular conformations for property prediction and molecular generation. arXiv preprint arXiv:2006.05531, 2020.  
Simon Batzner, Tess E Smidt, Lixin Sun, Jonathan P Mailoa, Mordechai Kornbluth, Nicola Molinari, and Boris Kozinsky. Se (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. arXiv preprint arXiv:2101.03164, 2021.  
Sybren Ruurds De Groot and Peter Mazur. Non-equilibrium thermodynamics. Courier Corporation, 2013.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. In ICLR, 2017.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in neural information processing systems, pp. 2224-2232, 2015.  
Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling. Se(3)-transformers: 3d roto-translation equivariant attention networks. NeurIPS, 2020.  
Octavian-Eugen Ganea, Lagnajit Pattanaik, Connor W Coley, Regina Barzilay, Klavs F Jensen, William H Green, and Tommi S Jaakkola. Geomol: Torsional geometric generation of molecular 3d conformer ensembles. arXiv preprint arXiv:2106.07802, 2021.  
Niklas WA Gebauer, Michael Gastegger, Stefan SP Hessmann, Klaus-Robert Muller, and Kristof T Schütt. Inverse design of 3d molecular structures with conditional generative neural networks. arXiv preprint arXiv:2109.04824, 2021.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1263–1272. JMLR.org, 2017.  
T. Gogineni, Ziping Xu, Exequiel Punzalan, Runxuan Jiang, Joshua A Kammeraad, Ambuj Tewari, and P. Zimmerman. Torsionnet: A reinforcement learning approach to sequential conformer search. ArXiv, abs/2006.07078, 2020.  
Thomas A Halgren. Merck molecular force field. v. extension of mmff94 using experimental data, additional computational data, and empirical rules. Journal of Computational Chemistry, 17(5-6): 616-641, 1996.  
Paul CD Hawkins. Conformation generation: the state of the art. Journal of Chemical Information and Modeling, 57(8):1747-1756, 2017.  
Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239, 2020.  
Weihua Hu, Muhammed Shuaibi, Abhishek Das, Siddharth Goyal, Anuroop Sriram, Jure Leskovec, Devi Parikh, and Larry Zitnick. Forcenet: A graph neural network for large-scale quantum chemistry simulation. 2021.  
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. arXiv preprint arXiv:1802.04364, 2018.  
Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael John Lamarre Townshend, and Ron Dror. Learning from protein structure with geometric vector perceptrons. In International Conference on Learning Representations, 2021.  
Wolfgang Kabsch. A solution for the best rotation to relate two sets of vectors. Acta Crystallographica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography, 32(5):922-923, 1976.

Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, 2013.  
Jonas Kohler, Leon Klein, and Frank Noe. Equivariant flows: Exact likelihood generative learning for symmetric densities. In Proceedings of the 37th International Conference on Machine Learning, 2020.  
Leo Liberti, Carlile Lavor, Nelson Maculan, and Antonio Mucherino. Euclidean distance geometry and applications. SIAM review, 56(1):3-69, 2014.  
Elman Mansimov, Omar Mahmood, Seokho Kang, and Kyunghyun Cho. Molecular geometry prediction using a deep generative graph neural network. arXiv preprint arXiv:1904.00314, 2019.  
B. Miller, M. Geiger, T. Smidt, and F. Noé. Relevance of rotationally equivariant convolutions for predicting molecular properties. ArXiv, abs/2008.08461, 2020.  
Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457), 2019.  
Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1-7, 2014.  
Sereina Riniker and Gregory A. Landrum. Better informed distance geometry: Using what we know to improve conformation generation. Journal of Chemical Information and Modeling, 55(12): 2562-2574, 2015.  
Victor Garcia Satorras, Emiel Hoogeboom, Fabian B Fuchs, Ingmar Posner, and Max Welling. E (n) equivariant normalizing flows for molecule generation in 3d. arXiv preprint arXiv:2105.09016, 2021a.  
Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks, 2021b.  
Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. Schnet: A continuous-filter convolutional neural network for modeling quantum interactions. In Advances in Neural Information Processing Systems, pp. 991-1001. Curran Associates, Inc., 2017.  
Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. Graphaf: a flow-based autoregressive model for molecular graph generation. arXiv preprint arXiv:2001.09382, 2020.  
Chence Shi, Shitong Luo, Minkai Xu, and Jian Tang. Learning gradient fields for molecular conformation generation. ArXiv, 2021.  
Muhammed Shuaibi, Adeesh Kolluru, Abhishek Das, Aditya Grover, Anuroop Sriram, Zachary Ulissi, and C Lawrence Zitnick. Rotation invariant graph neural networks using spin convolutions. arXiv preprint arXiv:2106.09575, 2021.  
Gregor Simm and Jose Miguel Hernandez-Lobato. A generative model for molecular distance geometry. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119, pp. 8949-8958. PMLR, 2020.  
Gregor N. C. Simm, Robert Pinsler, Gábor Csányi, and José Miguel Hernández-Lobato. Symmetry-aware actor-critic for 3d molecular design. In International Conference on Learning Representations, 2021.  
Daniel G. A. Smith, L. Burns, A. Simmonett, R. Parrish, M. C. Schieber, Raimondas Galvelis, P. Kraus, H. Kruse, Roberto Di Remigio, Asem Alenaizan, A. M. James, S. Lehtola, Jonathon P Misiewicz, et al. Psi4 1.4: Open-source software for high-throughput quantum chemistry. The Journal of chemical physics, 2020.  
Jascha Sohl-Dickstein, Eric A Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. arXiv preprint arXiv:1503.03585, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.  
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, pp. 11918-11930, 2019.  
Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. NeurIPS, 2020.  
N. Thomas, T. Smidt, Steven M. Kearnes, Lusann Yang, L. Li, Kai Kohlhoff, and P. Riley. Tensor field networks: Rotation- and translation-equivariant neural networks for 3d point clouds. *ArXiv*, 2018.  
M. Weiler, M. Geiger, M. Welling, W. Boomsma, and T. Cohen. 3d steerable cnns: Learning rotationally equivariant features in volumetric data. In NeurIPS, 2018.  
Minkai Xu, Shitong Luo, Yoshua Bengio, Jian Peng, and Jian Tang. Learning neural generative dynamics for molecular conformation generation. In International Conference on Learning Representations, 2021a.  
Minkai Xu, Wujie Wang, Shitong Luo, Chence Shi, Yoshua Bengio, Rafael Gomez-Bombarelli, and Jian Tang. An end-to-end framework for molecular conformation generation via bilevel programming. arXiv preprint arXiv:2105.07246, 2021b.  
Linfeng Zhang, Jiequn Han, Han Wang, Roberto Car, and Weinan E. Deep Potential Molecular Dynamics: A Scalable Model with the Accuracy of Quantum Mechanics. Physical Review Letters, 120(14):143001, 2018.
