# ADVECTIVE DIFFUSION TRANSFORMERS FOR TOPOLOGICAL GENERALIZATION IN GRAPH LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph diffusion equations are intimately related to graph neural networks (GNNs) and have recently attracted attention as a principled framework for analyzing GNN dynamics, formalizing their expressive power, and justifying architectural choices. One key open questions in graph learning is the generalization capabilities of GNNs. A major limitation of current approaches hinges on the assumption that the graph topologies in the training and test sets come from the same distribution. In this paper, we make steps towards understanding the generalization of GNNs by exploring how graph diffusion equations extrapolate and generalize in the presence of varying graph topologies. We first show deficiencies in the generalization capability of existing models built upon local diffusion on graphs, stemming from the exponential sensitivity to topology variation. Our subsequent analysis reveals the promise of non-local diffusion, which advocates for feature propagation over fully-connected latent graphs, under the assumption of a specific data-generating condition. In addition to these findings, we propose a novel graph encoder backbone, Advective Diffusion Transformer (ADiT), inspired by advective graph diffusion equations that have a closed-form solution backed up with theoretical guarantees of desired generalization under topological distribution shifts. The new model, functioning as a versatile graph Transformer, demonstrates superior performance across a wide range of graph learning tasks. Source codes will be made publicly available.

# 1 INTRODUCTION

Learning representations for non-Euclidean data is essential for geometric deep learning. Graph-structured data in particular has attracted increasing attention, as graphs are a very popular mathematical abstraction for systems of relations and interactions that can be applied from microscopic scales (e.g. molecules) to macroscopic ones (social networks). The most common framework for learning on graphs is graph neural networks (GNNs), which operate by propagating information between adjacent nodes of the graph networks (Scarselli et al., 2008; Gilmer et al., 2017; Kipf & Welling, 2017). GNNs are intimately related to graph diffusion equations (Atwood & Towsley, 2016; Klicpera et al., 2019; Chamberlain et al., 2021a) and can be seen as discretized versions thereof. Considering GNNs as diffusion equations offers powerful tools from the domain of partial differential equations (PDEs) allowing to study the expressive power (Bodnar et al., 2022), behaviors such as over-smoothing (Rusch et al., 2023; Di Giovanni et al., 2022) and over-squashing (Topping et al., 2022), the settings of missing features (Rossi et al., 2022), and guide architectural choices (Di Giovanni et al., 2022).

While significant efforts have been devoted to understanding the expressive power of GNNs and similar architectures for graph learning, the generalization capabilities of such methods are largely an open question. In many important real-world settings, the training and testing graph topologies can be generated from different distributions (a phenomenon referred to as "topological shift") (Koh et al., 2021; Hu et al., 2021; Bazhenov et al., 2023; Zhang et al., 2023).

Generalization to testing data with new unseen topological patterns can be highly challenging when training observations are insufficient. One of the established principles by prior works resorts to the invariant underlying mechanism (Rojas-Carulla et al., 2018; Arjovsky et al., 2019; Scholkopf et al., 2021) that governs the shared data-generating process and enables generalization across environments. However, unlike in Euclidean space, in the case of graphs, the invariant topological features can be more abstract and complex, making it hard to come up with a single model to resolve the challenge.

Contributions We explore how graph diffusion equations (and derived GNN architectures) generalize in the presence of topological shifts. We show that current models relying on local graph diffusion suffer from undesirable sensitivity to variations in graph structure, making it difficult to achieve stable and reliable predictions and potentially tampering generalization. Extending the diffusion operators to latent fully-connected graphs in principle allows ideal generalization if the ground-truth labels are independent of the observed graphs in data generation, which is however often violated in practice.

To overcome this problem, we introduce a novel method for learning graph representations based on advective diffusion equations. We connect advective diffusion with a Transformer-like architecture particularly designed for the challenging topological generalization: the non-local diffusion term (instantiated as global attention) aims to capture invariant latent interactions that are insensitive to the observed graphs; the advection term (instantiated as local message passing) accommodates the observed topological patterns specific to environments. We prove that the closed-form solution of this new diffusion system possesses the capability to control the rate of change in node representations w.r.t. topological variations at arbitrary orders. This further produces a guarantee of the desired level of generalization under topological shifts.

For efficiently calculating the solution of the diffusion equation, we use the numerical scheme based on the Padé-Chebyshev theory (Golub & Van Loan, 1989). Experiments show that our model, which we call *Adjective Diffusion Transformer* (ADiT), offers superior generalization across a broad spectrum of graph ML tasks in diverse domains, including social and citation networks, molecular screening, and protein interactions.

# 2 BACKGROUND AND PRELIMINARIES

As building blocks of our methodology, we first recapitulate diffusion equations on manifolds (Freidlin & Wentzell, 1993; Medvedev, 2014) and its established connection with graph representations.

Diffusion on Riemannian manifolds. Let  $\Omega$  denote an abstract domain, which we assume here to be a Riemannian manifold (Eells & Sampson, 1964). A key feature distinguishing an  $n$ -dimensional Riemannian manifold from a Euclidean space is the fact that it is only locally Euclidean, in the sense that at every point  $u \in \Omega$  one can construct  $n$ -dimensional Euclidean tangent space  $T_u\Omega \cong \mathbb{R}^n$  that locally models the structure of  $\Omega$ . The collection of such spaces (referred to as the tangent bundle and denoted by  $T\Omega$ ) is further equipped with a smoothly-varying inner product (Riemannian metric).

Now consider some quantity (e.g., temperature) as a function of the form  $q: \Omega \to \mathbb{R}$ , which we refer to as a scalar field. Similarly, we can define a (tangent) vector field  $Q: \Omega \to T\Omega$ , associating to every point  $u$  on a manifold a tangent vector  $Q(u) \in T_u\Omega$ , which can be thought of as a local infinitesimal displacement. We use  $\mathcal{Q}(\Omega)$  and  $\mathcal{Q}(T\Omega)$  to denote the functional spaces of scalar and vector fields, respectively. The gradient operator  $\nabla: \mathcal{Q}(\Omega) \to \mathcal{Q}(T\Omega)$  takes scalar fields into vector fields representing the local direction of the steepest change of the field. The divergence operator is the adjoint of the gradient and maps in the opposite direction,  $\nabla^*: \mathcal{Q}(T\Omega) \to \mathcal{Q}(\Omega)$ .

A manifold diffusion process models the evolution of a quantity (e.g., temperature or chemical concentration) due to its difference across spatial locations on  $\Omega$ . Denoting by  $q(u,t):\Omega \times [0,\infty)\to \mathbb{R}$  the quantity over time  $t$ , the process is described by a PDE (diffusion equation) (Romeny, 2013):  $\frac{\partial q(u,t)}{\partial t} = \nabla^{*}\left(S(u,t)\odot \nabla q(u,t)\right)$ ,  $t\geq 0,u\in \Omega$  with initial conditions  $q(u,0) = q_0(u)$ , (1)

and possibly additional boundary conditions if  $\Omega$  has a boundary.  $S$  denotes the diffusivity of the domain. It is typical to distinguish between an isotropic (location-independent diffusivity), non-homogeneous (location-dependent diffusivity  $S = s(u)\in \mathbb{R}$ ), and anisotropic (location- and direction-dependent  $S(u)\in \mathbb{R}^{n\times n}$ ) settings. In the cases studied below, we will assume the dependence of the diffusivity on the location is via a function of the quantity itself, i.e.,  $S = S(q(u,t))$ .

Diffusion on Graphs. Recent works leverage diffusion equations as a foundation principle for learning graph representations (Chamberlain et al., 2021a;b; Thorpe et al., 2022; Bodnar et al., 2022; Choi et al., 2023; Rusch et al., 2023), employing analogies between calculus on manifolds and graphs. Let  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  be a graph with nodes  $\mathcal{V}$  and edges  $\mathcal{E}$ , represented by the  $|\mathcal{V}| \times |\mathcal{V}|$  adjacency matrix  $\mathbf{A}$ . Let  $\mathbf{X} = [\mathbf{x}_u]_{u \in \mathcal{V}}$  denote a  $|\mathcal{V}| \times D$  matrix of node features, analogous to scalar fields on manifolds. The graph gradient  $(\nabla \mathbf{X})_{uv} = \mathbf{x}_v - \mathbf{x}_u$  defines edge features for  $(u,v) \in \mathcal{E}$ , analogous to a vector field on a manifold. Similarly, the graph divergence of edge features  $\mathbf{E} = [\mathbf{e}_{uv}]_{(u,v) \in \mathcal{E}}$ , defined as the adjoint  $(\nabla^*\mathbf{E})_u = \sum_{v:(u,v) \in \mathcal{E}} \mathbf{e}_{uv}$ , produces node features.

Diffusion-based approaches replace discrete GNN layers with continuous time-evolving node embeddings  $\mathbf{Z}(t) = [\mathbf{z}_u(t)]$ , where  $\mathbf{z}_u(t):[0,\infty)\to \mathbb{R}^D$  is driven by the graph diffusion equation,

$\partial \mathbf{Z}(t) / \partial t = \nabla^{*}\left(\mathbf{S}(\mathbf{Z}(t),t;\mathbf{A})\odot \nabla \mathbf{Z}(t)\right), t\geq 0,$  with initial conditions  $\mathbf{Z}(0) = \phi_{enc}(\mathbf{X})$  (2) where  $\phi_{enc}$  is a node-wise MLP encoder and w.l.o.g., the diffusivity  $\mathbf{S}(\mathbf{Z}(t),t;\mathbf{A})$  over the graph can be defined as a  $|\mathcal{V}|\times |\mathcal{V}|$  matrix-valued function dependent on  $\mathbf{A}$ , which measures the rate of information flows between node pairs. With the graph gradient and divergence, Eqn. 2 becomes

$\partial \mathbf{Z}(t) / \partial t = (\mathbf{C}(\mathbf{Z}(t), t; \mathbf{A}) - \mathbf{I})\mathbf{Z}(t)$ ,  $0 \leq t \leq T$ , with initial conditions  $\mathbf{Z}(0) = \phi_{enc}(\mathbf{X})$ , (3) where  $\mathbf{C}(\mathbf{Z}(t), t; \mathbf{A})$  is a  $|\mathcal{V}| \times |\mathcal{V}|$  coupling matrix associated with the diffusivity. Eqn. 3 yields a dynamics from  $t = 0$  to an arbitrary given stopping time  $T$ , where the latter gives node representations for prediction, e.g.,  $\hat{\mathbf{Y}} = \phi_{dec}(\mathbf{Z}(T))$ . The coupling matrix determines the interactions between different nodes in the graph, and its common instantiations include the normalized adjacency (non-parametric) and learnable attention matrix (parametric), in which cases the finite-difference numerical iterations for solving Eqn. 3 correspond to the discrete propagation layers of common GNNs (Chamberlain et al., 2021a) and Transformers (Wu et al., 2023) (see Appendix A for details).

It is typical to tacitly make a closed-world assumption, i.e., the graph topologies of training and testing data are generated from the same distribution. The challenge of generalization arises when the testing graph topology is different from the training one. In such an open-world regime, it still remains unexplored how graph diffusion equations extrapolate and generalize to new unseen structures.

# 3 CAN GRAPH DIFFUSION GENERALIZE?

As a prerequisite for analyzing the generalization behaviors of graph diffusion models, we need to characterize how topological shifts happen in nature. In general sense, extrapolation is impossible without any exposure to the new data or prior knowledge about the data-generating mechanism. In our work, we assume testing data is strictly unknown during training, in which case structural assumptions become necessary for authorizing generalization.

# 3.1 PROBLEM FORMULATION: GRAPH DATA GENERATION

We present the underlying data-generating mechanism of graph data in Fig. 1, inspired by the graph limits (Lovasz & Szegedy, 2006; Medvedev, 2014) and random graph models (Snijders & Nowicki, 1997). In graph theory, the topology of a graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  can be assumed to be generated by a graphon (or continuous graph limit), a random symmetric measurable function  $W:[0,1]^2\to [0,1]$ , which is an unobserved latent variable. In our work, we generalize this data-generating mechanism to include alongside graph adjacency also node features and labels, as follows:

![](images/f97c05d5f0fdced3a4e6e7d28ff45b97e704719fa5fda98b418880493f3c616b.jpg)  
Figure 1: The data-generating mechanism with topological shifts caused by environment  $E$ . The solid (resp. dashed) nodes represents observed (resp. latent) random variables.

i) Each node  $u\in \mathcal{V}$  has a latent i.i.d. variable  $U_{u}\sim U[0,1]$ . The node features are a random variable  $X = [X_{u}]$  generated from each  $U_{u}$  through a certain node-wise function

$X_{u} = g(U_{u};W)$ . We denote by matrix  $\mathbf{X}$  a particular realization of the random variable  $X$ .

ii) Similarly, the graph adjacency  $A = [A_{uv}]$  is a random variable generated through a pairwise function  $A_{uv} = h(U_u, U_v; W, E)$  additionally dependent on the environment  $E$ . The change of  $E$  happens when it transfers from training to testing, resulting in a different distribution of  $A$ . We denote by  $\mathbf{A}$  a particular realization of the adjacency matrix.  
iii) The label  $Y$  can be specified in certain forms. In graph-level tasks (as we assume in below),  $Y$  is generated by a function over sets,  $Y = r(\{U_{v\in \mathcal{V}}\}, A; W)$ . Denote by  $\mathbf{Y}$  a realization of  $Y$ .

The above process formalizes the data-generating mechanism behind various data of inter-dependent nature. It boils down to finding parameters  $\theta$  of a parametric function  $\Gamma_{\theta}(\mathbf{A},\mathbf{X})$  that establishes the predictive mapping from observed node features  $\mathbf{X}$  and graph adjacency  $\mathbf{A}$  to the label  $\mathbf{Y}$ .  $\Gamma_{\theta}$  is typically implemented as a GNN, which is expected to possess sufficient expressive power (in the sense that  $\exists \theta$  such that  $\Gamma_{\theta}(\mathbf{A},\mathbf{X})\approx \mathbf{Y}$ ) as well as generalization capability under topological distribution shift (i.e., when the observed graph topology varies from training to testing, which in our model amounts to the change in  $E$ ). While significant attention in the literature has been devoted to the former property (Morris et al., 2019; Xu et al., 2019; Bouritsas et al., 2023; Papp et al., 2021; Balcilar et al., 2021; Bodnar et al., 2022); the latter is largely an open question.

# 3.2 GRAPH DIFFUSION UNDER TOPOLOGICAL SHIFTS

Building upon the connection between GNNs and diffusion equations, we next study the behavior of diffusion equation (i.e., Eqn. 3) under topological shifts, which will shed lights on GNN generalization. The effect of  $\mathbf{A}$  on node representations (solution of the diffusion equation  $\mathbf{Z}(T)$ ) stems from the coupling matrix  $\mathbf{C}(\mathbf{Z}(t), t; \mathbf{A})$ . Thereby, the output of the diffusion process can be expressed as  $\mathbf{Z}(T) = f(\mathbf{Z}(0), \mathbf{A})$ . We are interested in the extrapolation behavior of graph diffusion models that can be reflected by the change of  $\mathbf{Z}(T)$  w.r.t. small perturbation centered at  $\mathbf{A}$ .

Linear Diffusion. We first consider the constant diffusivity setting inducing  $\mathbf{C}(\mathbf{Z}(t), t; \mathbf{A}) = \mathbf{C}$ . In this case, Eqn. 3 becomes a linear diffusion equation with a closed-form solution  $\mathbf{Z}(t) = e^{-(\mathbf{I} - \mathbf{C})t}\mathbf{Z}(0)$ . In this case, using the numerical scheme to solve the PDE would induce the discrete propagation layers akin to SGC (Wu et al., 2019), where the non-linearity in-between layers is omitted for acceleration (see more illustration on this connection in Appendix A). The following proposition shows that the variation magnitude of  $\mathbf{Z}(T)$  can be significant for small change of input graphs.

Proposition 1. If the coupling matrix  $\mathbf{C}$  is set as the normalized adjacency  $\tilde{\mathbf{A}} = \mathbf{D}^{-1}\mathbf{A}$  or  $\tilde{\mathbf{A}} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ , where  $\mathbf{D}$  denotes the diagonal degree matrix of  $\mathbf{A}$ , then the change of  $\mathbf{Z}(T; \tilde{\mathbf{A}})$  given by Eqn. 3 w.r.t. a small perturbation  $\Delta \tilde{\mathbf{A}}$  is  $\| \mathbf{Z}(T; \tilde{\mathbf{A}} + \Delta \tilde{\mathbf{A}}) - \mathbf{Z}(T; \tilde{\mathbf{A}}) \|_2 = O(\exp (\| \Delta \tilde{\mathbf{A}} \|_2 T))$ .

The consequence of this result is that the label prediction  $\hat{\mathbf{Y}} = \phi_{dec}(\mathbf{Z}(T; \tilde{\mathbf{A}}))$  can be highly (exponentially) sensitive to the change of the graph topology. Under the assumption of our graph generation model in which the graph adjacency is a realization of a random variable  $A = h(U_u, U_v; W, E)$  dependent on a varying environment  $E$ , this may result in poor generalization. Proposition 1 can be extended to the multi-layer model comprised of multiple piece-wise diffusion dynamics with feature transformations (e.g., neural networks) in-between layers (see Appendix B.2).

Non-Linear Diffusion. In a more general setting, the diffusivity can be time-dependent. The analogy in GNN architectures e.g. GAT (Velickovic et al., 2018) is layer-wise propagation that can aggregate neighboring nodes' signals with adaptive strengths across edges. Consider the time-dependent case used in (Chamberlain et al., 2021a), where  $\mathbf{C}(t)$  depends on  $\mathbf{Z}(t)$  throughout the diffusion process:

$$
\mathbf {C} (\mathbf {Z} (t); \mathbf {A}) = \left[ c _ {u v} (t) \right] _ {u, v \in \mathcal {V}}, \quad c _ {u v} (t) = \mathbb {I} [ (u, v) \in \mathcal {E} ] \cdot \frac {\eta \left(\mathbf {z} _ {u} (t) , \mathbf {z} _ {v} (t)\right)}{\sum_ {w , (u , w) \in \mathcal {E}} \eta \left(\mathbf {z} _ {u} (t) , \mathbf {z} _ {w} (t)\right)}, \tag {4}
$$

where  $\eta : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$  denotes a pairwise function ("attention"). While such a non-linear diffusion equation has no closed-form solution anymore, we can generalize our previous result as follows:

Proposition 2. For arbitrary time limit  $T$  and bounded function  $\eta$ , the change of  $\mathbf{Z}(T)$  by the diffusion model Eqn. 3 with  $\mathbf{C}(\mathbf{Z}(t); \mathbf{A})$  by Eqn. 4 w.r.t. a small perturbation  $\Delta \mathbf{A}$  is  $\mathcal{O}\left(\exp \left(\| \Delta \mathbf{A} \|_2 T\right)\right)$ .

The analysis so far suggests the common limitation of local graph diffusion equations with different instantiations, i.e., the sensitivity of the output states w.r.t. the change of graph topology. This implies the potential failure of such a model class for the challenge of generalization where the graph topology varies from training to testing. Moreover, the analysis enlightens that the crux of the matter lies in the diffusion operators which determine the effect of graph structures throughout the diffusion process.

# 3.3 NON-LOCAL GRAPH DIFFUSION AND GENERALIZATION WITH CONDITIONS

We proceed to extend our discussion to another class of neural diffusion models that resort to non-local diffusion operators allowing instantaneous information flows among arbitrary locations (Chasseigne et al., 2006). In the context of learning on graphs, the non-local diffusion can be seen as generalizing the feature propagation to a complete or fully-connected (latent) graph (Wu et al., 2023), in contrast with common GNNs that allow message passing only between neighboring nodes. Formally speaking, we can define the gradient and divergence operators on a complete graph:  $(\nabla \mathbf{X})_{uv} = \mathbf{x}_v - \mathbf{x}_u$ $(u,v\in \mathcal{V})$  and  $(\nabla^{*}\mathbf{E})_u = \sum_{v\in \mathcal{V}}\mathbf{e}_{uv}$ $(u\in \mathcal{V})$ . The corresponding diffusion equation still exhibits the form of Eqn. 3. Nevertheless, unlike the models studied in Sec. 3.2 assuming that  $\mathbf{C}(t)$  only has non-zero entries  $c_{uv}(t)\neq 0$  for neighboring node pairs  $(u,v)\in \mathcal{E}$ , the non-local diffusion model allows non-zero  $c_{uv}(t)$  for arbitrary  $(u,v)$ 's to accommodate the all-pair information flows. For example, the coupling matrix can be instantiated as the global attention  $\mathbf{C}(\mathbf{Z}(t)) = [c_{uv}(t)]_{u,v\in \mathcal{V}}$  with

$c_{uv}(t) = \frac{\eta(\mathbf{z}_u(t),\mathbf{z}_v(t))}{\sum_{w\in\mathcal{V}}\eta(\mathbf{z}_u(t),\mathbf{z}_w(t))}$ , in which case the finite-difference iteration of the non-local diffusion equation corresponds to a Transformer layer (Vaswani et al., 2017) (see details in Appendix A).

The non-local diffusion model essentially learns latent interaction graphs among nodes from input data and is agnostic to observed graph. For the predictive function  $\Gamma_{\theta}$  built by the diffusion equation along with the encoder  $\phi_{enc}$  and decoder  $\phi_{dec}$ , we can theoretically guarantee topological generalization when  $Y$  is conditionally independent from  $A$  within the data-generating process in Sec. 3.1.

Proposition 3. Suppose the label  $Y$  is conditionally independent from  $A$  with given  $\{U_u\}_{u \in \mathcal{V}}$  in the data generation hypothesis of Sec. 3.1, then for non-local diffusion model  $\Gamma_{\theta}$  minimizing the empirical risk  $\mathcal{R}_{emp}(\Gamma_{\theta}; E_{tr}) = \frac{1}{N_{tr}} \sum_{i}^{N_{tr}} l(\Gamma_{\theta}(\mathbf{X}^{(i)}, \mathbf{A}^{(i)}), \mathbf{Y}^{(i)})$  over training data  $\{(\mathbf{X}^{(i)}, \mathbf{A}^{(i)}, \mathbf{Y}^{(i)})\}$  generated from  $p(X, A, Y | E = E_{tr})$ , it holds with confidence  $1 - \delta$  for the bounded generalization error on unseen data  $(\mathbf{X}', \mathbf{A}', \mathbf{Y}')$  from a new environment  $E_{te} \neq E_{tr}: \mathcal{R}(\Gamma_{\theta}; E_{te}) \triangleq$

$$
\mathbb {E} _ {\left(\mathbf {X} ^ {\prime}, \mathbf {A} ^ {\prime}, \mathbf {Y} ^ {\prime}\right) \sim p (X, A, Y | E = E _ {t e})} \left[ l \left(\Gamma_ {\theta} \left(\mathbf {X} ^ {\prime}, \mathbf {A} ^ {\prime}\right), \mathbf {Y} ^ {\prime}\right) \right] \leq \mathcal {R} _ {e m p} \left(\Gamma_ {\theta}; E _ {t r}\right) + \mathcal {D} _ {1} (\Gamma , N _ {t r}), \tag {5}
$$

where  $\mathcal{D}_1(\Gamma, N_{tr}) = 2\mathcal{H}(\Gamma) + \mathcal{O}\left(\sqrt{(1 / N_{tr})\log(1 / \delta)}\right)$ ,  $\mathcal{H}(\Gamma)$  denotes the Rademacher complexity of the function class of  $\Gamma$ ,  $N_{tr}$  is the size of the training set, and  $l$  denotes any bounded loss function.

The conditional independence between  $Y$  and  $A$ , however, can be violated in many situations where labels strongly correlate with observed graph structures. In such cases, the non-local diffusion alone, discarding any observed structural information, could be insufficient for generalization.

# 4 GRAPH ADVECTIVE DIFFUSION FOR TOPOLOGICAL GENERALIZATION

The preceding analysis reveals that the obstacles for graph diffusion models to achieving generalization arise from the non-fulfillment of two critical criteria: i) the diffusion process is capable of learning useful topological patterns; ii) the node representations are insensitive to variation of graph structures. While balancing these two objectives can be challenging due to the inherent trade-off, we present a novel graph diffusion model in this section that offers a provable level of generalization. The new model is inspired by a different class of diffusion equations, advective diffusion.

# 4.1 MODEL FORMULATION: GRAPH ADVECTIVE DIFFUSION

Adjective Diffusion Equations. We first introduce the classic advective diffusion commonly used for characterizing physical systems with convoluted quantity transfers, where the term advection (or convection) refers to the evolution caused by the movement of the diffused quantity (Chandrasekhar, 1943). Consider the abstract domain  $\Omega$  of our interest defined in Sec. 2, and assume  $V(u,t)\in T_u\Omega$  (a vector field in  $\Omega$ ) to denote the velocity of the particle at location  $u$  and time  $t$ . The advective diffusion of the physical quantity  $q$  on  $\Omega$  is governed by the PDE as (Leveque, 1992)

$$
\frac {\partial q (u , t)}{\partial t} = \underbrace {\nabla^ {*} (S (u , t) \odot \nabla q (u , t))} _ {\text {d i f f u s i o n}} + \beta \underbrace {\nabla^ {*} (V (u , t) \cdot q (u , t))} _ {\text {a d v e c t i o n}}, \quad t \geq 0, u \in \Omega ; \quad q (u, 0) = q _ {0} (u), \tag {6}
$$

where  $\beta \geq 0$  is a weight. For example, if we consider  $q(u,t)$  as the water salinity in a river, then Eqn. 6 describes the temporal evolution of salinity at each location that equals to the spatial transfers of both diffusion process (caused by the concentration difference of salt and  $S$  reflects the molecular diffusivity in the water) and advection process (caused by the movement of the water and  $V$  characterizes the flowing directions).

Similarly, on a graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$ , we can define the velocity for each node  $u$  as a  $|\mathcal{V}|$ -dimensional vector-valued function  $\mathbf{V}(t) = [\mathbf{v}_u(t)]$ . Then, we have  $(\nabla^{*}(\mathbf{V}(t)\cdot \mathbf{Z}(t)))_{u} = \sum_{v\in \mathcal{V}}v_{uv}(t)\mathbf{z}_{v}(t)$ , giving rise to the graph advective diffusion equation:

$$
\frac {\partial \mathbf {Z} (t)}{\partial t} = [ \mathbf {C} (\mathbf {Z} (t), t) + \beta \mathbf {V} (t) - \mathbf {I} ] \mathbf {Z} (t), \quad 0 \leq t \leq T. \tag {7}
$$

Graph Advective Diffusion. We proceed to discuss how to properly define the coupling matrix  $\mathbf{C}$  and the velocity  $\mathbf{V}$  to ensure that advective diffusion equations are stable under topological shifts. Our inspiration stems from the recent research line in the pursuit of invariance in data generation (Rojas-Carulla et al., 2018; Arjovsky et al., 2019; Scholkopf et al., 2021), where the

principle of (out-of-distribution) generalization lies in enforcing proper inductive bias that guides the model to capture the invariant underlying mechanism shared across environments. Different from natural data in Euclidean space (e.g., images), the invariant topological patterns in graphs can be much more difficult to capture given their abstract and versatile characteristics. We next generalize the invariance principle as an important inductive bias integrated into the advective diffusion for generalization purpose (with illustration in Fig. 2).

Non-local diffusion as global attention. The diffusion process led by the concentration gradient acts as an internal driving force, where the diffusivity keeps invariant across environments (e.g., the molecular diffusivity stays constant in different rivers). This resonates with the environment-invariant latent interactions among nodes, determined by the underlying data manifold, that induce all-pair information flows over a complete graph. We thus follow Sec. 3.3 and instantiate  $\mathbf{C}$  as a global attention that computes the similarities between arbitrary node pairs.

Advection as local message passing. The advection process driven by the directional movement belongs to an external force, with the velocity depending on

contexts (e.g., different rivers). This is analogous to the environment-sensitive graph topology that is informative for prediction in specific environments. We instantiate the velocity as the normalized adjacency  $\mathbf{V} = \tilde{\mathbf{A}}$  that reflects graph structures. With the above definitions, our graph advective diffusion model can be formulated as:

![](images/4ed29c84f56519996496bee12673e4f92b03663a5119cba9e30de1606b326a74.jpg)  
Figure 2: Illustration of the proposed model.

$$
\frac {\partial \mathbf {Z} (t)}{\partial t} = \left[ \mathbf {C} + \beta \tilde {\mathbf {A}} - \mathbf {I} \right] \mathbf {Z} (t), 0 \leq t \leq T \text {w i t h i n i t i a l c o n d i t i o n s} \mathbf {Z} (0) = \phi_ {e n c} (\mathbf {X}),
$$

$$
\text {w h e r e} \quad \mathbf {C} = \left[ c _ {u v} \right] _ {u, v \in \mathcal {V}}, \quad c _ {u v} = \frac {\eta \left(\mathbf {z} _ {u} (0) , \mathbf {z} _ {v} (0)\right)}{\sum_ {w \in \mathcal {V}} \eta \left(\mathbf {z} _ {u} (0) , \mathbf {z} _ {w} (0)\right)}. \tag {8}
$$

Here  $\beta \in [0,1]$  is a weight hyper-parameter and  $\eta$  is a learnable pairwise similarity function. The two mechanisms of non-local diffusion (implemented through attention akin to Transformers) and advection (implemented like message passing neural networks) give rise to a new architecture, which we call the Advective Diffusion Transformer, or ADiT for short.

Remark. Eqn. 8 has a closed-form solution  $\mathbf{Z}(t) = e^{-(\mathbf{I} - \mathbf{C} - \beta \bar{\mathbf{A}})t}\mathbf{Z}(0)$ , and as we will show in the next subsection, it allows generalization guarantees with topological distribution shifts. A special case of  $\beta = 0$  (no advection) can be used in situations where the graph structure is not useful. Moreover, one can extend Eqn. 8 to a non-linear equation with time-dependent  $\mathbf{C}(\mathbf{Z}(t), t)$ , in which situation the equation will have no closed-form solution and need numerical schemes for solving. Similarly to Di Giovanni et al. (2022), we found in our experiments a simple linear diffusion to be sufficient to yield promising performance. We therefore leave the study of the non-linear variant for the future.

# 4.2 HOW GRAPH ADVECTIVE DIFFUSION HANDLES TOPOLOGICAL SHIFTS

We proceed to analyze the behavior of our proposed model w.r.t. topological shifts to demonstrate its capability of generalizing to out-of-distribution (OOD) data. Our first main result is derived based on the universal approximation power of neural networks and the data generation hypothesis in Sec. 3.1.

Theorem 1. For the model Eqn. 7 with  $\mathbf{C}$  pre-computed by global attention over  $\mathbf{Z}(0)$  and fixed velocity  $\mathbf{V} = \tilde{\mathbf{A}}$ , the change rate of node representations  $\mathbf{Z}(T; \tilde{\mathbf{A}})$  w.r.t. a small perturbation  $\Delta \tilde{\mathbf{A}}$  can be reduced to  $\mathcal{O}(\psi(\|\Delta \tilde{\mathbf{A}}\|_2))$  where  $\psi$  denotes an arbitrary polynomial function.

Theorem 1 suggests that the advective diffusion model with observed structural information incorporated is capable of controlling the impact of topology variation on node representations to arbitrary rates. We can further derive the generalization error that is decomposed into the in-distribution generalization (ID) error  $\mathcal{D}_1(\Gamma, N_{tr})$  and the topological distribution gap between ID and OOD data.

Theorem 2. Assume  $l$  and  $\phi_{dec}$  are Lipschitz continuous. Then for data generated with the data generation hypothesis of Sec. 3.1 from arbitrary  $E_{tr}$  and  $E_{te}$ , we have the generalization error bound of the model  $\Gamma_{\theta}$  with confidence  $1 - \delta$ :

$$
\mathcal {R} \left(\Gamma_ {\theta}; E _ {t e}\right) \leq \mathcal {R} _ {e m p} \left(\Gamma_ {\theta}; E _ {t r}\right) + \mathcal {D} _ {1} \left(\Gamma , N _ {t r}\right) + \mathcal {D} _ {2} \left(E _ {t r}, E _ {t e}, W\right), \tag {9}
$$

$$
w h e r e \mathcal {D} _ {2} (E _ {t r}, E _ {t e}, W) = \mathcal {O} (\mathbb {E} _ {\mathbf {A} \sim p (A | E _ {t r}), \mathbf {A} ^ {\prime} \sim p (A | E _ {t e})} [ \psi (\| \Delta \tilde {\mathbf {A}} \| _ {2}) ]).
$$

Theorem 2 implies that the generalization error can be controlled with the adaptive change rate yielded by the model. The model possesses provable potential for achieving a desired level of generalization with topological shifts. Furthermore, our model only requires trainable parameters for two shallow MLPs  $\phi_{enc}$  and  $\phi_{dec}$  and the attention network  $\eta$ , which is highly parameter-efficient. This helps to reduce the model complexity measured by  $\mathcal{H}(\Gamma)$  that impacts  $\mathcal{D}_1$  and is beneficial for generalization.

# 4.3 NUMERICAL SOLVERS FOR GRAPH ADVECTIVE DIFFUSION

We next delve into the model implementation, with a key question how to compute the closed-form solution  $e^{-(\mathbf{I} - \mathbf{C} - \beta \tilde{\mathbf{A}})t}$ . Direct computation of the matrix exponential through eigendecomposition is computationally intractable for large matrices. As an alternative, we explore several numerical approximation techniques based on series expansion.

ADIT-INVERSE uses a numerical method based on the extension of Padé-Chebyshev theory to rational fractions (Golub & Van Loan, 1989; Gallopoulos & Saad, 1992), which has shown empirical success in 3D shape analysis (Patané, 2014). The matrix exponential is approximated by solving multiple linear systems (see more details and derivations in Appendix D) and we generalize it as a flexible multi-head network where each head propagates in parallel:

$$
\mathbf {Z} (T) \approx \sum_ {h = 1} ^ {H} \phi_ {F C} ^ {(h)} (\mathbf {Z} _ {h}), \quad \mathbf {Z} _ {h} = \operatorname {l i n s o l v e r} (\mathbf {L} _ {h}, \mathbf {Z} (0)), \quad \mathbf {L} _ {h} = (1 + \theta) \mathbf {I} - \mathbf {C} _ {h} - \beta \tilde {\mathbf {A}}, \tag {10}
$$

where the linsolver computes the matrix inverse  $\mathbf{Z}_h = (\mathbf{L}_h)^{-1}\mathbf{Z}(0)$  and can be efficiently implemented via torch.linalg.solve() that supports automated differentiation. Each head contributes to propagation with the pre-computed attention  $\mathbf{C}_h$  and node-wise transformation  $\phi_{FC}^{(h)}$ .

ADIT-SERIES approximates the matrix inverse via finite geometric series (see Appendix D for detailed derivations)

$$
\mathbf {Z} (T) \approx \sum_ {h = 1} ^ {H} \phi_ {F C} ^ {(h)} (\mathbf {Z} _ {h}), \quad \mathbf {Z} _ {h} = [ \mathbf {Z} (0), \mathbf {P} _ {h} \mathbf {Z} (0), \dots , (\mathbf {P} _ {h}) ^ {K} \mathbf {Z} (0) ], \quad \mathbf {P} _ {h} = \mathbf {C} _ {h} + \beta \tilde {\mathbf {A}}, \tag {11}
$$

for better scalability. This model resorts to aggregation of  $K$ -order propagation with the propagation matrix  $\mathbf{P}_h$  in each head. The feed-forward of the model can be efficiently computed within linear complexity w.r.t. the number of nodes (see how we achieve this acceleration in Appendix E.1.2).

The node representations obtained by approximate solution of the diffusion equation  $\mathbf{Z}(T)$  are then fed into  $\phi_{dec}$  for prediction and loss computation (e.g., cross-entropy for classification or mean square loss for regression). Due to space limit, we defer details of model architectures to Appendix E.1. Moreover, in Appendix E.2 we discuss how to extend our model to accommodate edge attributes.

# 5 EXPERIMENTS

We apply our model to synthetic and real-world datasets that involve various topological distribution shifts. We consider a wide variety of graph-based downstream tasks of disparate scales and granularities. More detailed dataset information is provided in Appendix F.1. In each case, we compare with different sets of competitors that are suitable for the tasks. Details on baselines and implementation are deferred to Appendix F.2 and F.3, respectively.

# 5.1 SYNTHETIC DATASETS

We create synthetic datasets that simulate the data generation in Sec. 3.1 to validate our model. We instantiate  $h$  as a stochastic block model which generates edges  $A_{uv}$  according to block numbers  $(b)$ , intra-block edge probability  $(p_1)$  and inter-block edge probability  $(p_2)$ . Then we study three types of topological distribution shifts: homophily shift (changing  $p_2$  with fixed  $p_1$ ); density shift (changing  $p_1$  and  $p_2$ ); and block shift (varying  $b$ ). The predictive task is node regression and we use RMSE to measure the performance. Details for dataset generation is presented in Appendix F.1.1.

Fig. 3 plots RMSE on training/validation/testing graphs in three cases. We compare our model (ADIT-INVERSE and ADIT-SERIES) with diffusion-based models analyzed in Sec. 3. The latter includes Diff-Linear (graph diffusion with constant C), Diff-MultiLayer (the extension of Diff-Linear with intermediate feature transformations), Diff-Time (graph diffusion with time-dependent  $\mathbf{C}(\mathbf{Z}(t)))$

![](images/f990a8dac622c9e6af169efd7884e313e5ccf8cedd3b6db8edbacb8307610d43.jpg)  
Figure 3: Results of RMSE  $(\downarrow)$  on synthetic datasets that simulate the topological shifts caused by the environment  $E$  in Fig. 1. We consider three types of shifts w.r.t. homophily levels, edge densities, and block numbers, respectively. In each case, the validation and  $\#1 \sim \#10$  testing sets are generated with different configurations introducing increasing distribution gaps from the training set.

and Diff-NonLocal (non-local diffusion with global attentive diffusivity  $\mathbf{C}(\mathbf{Z}(t))$ ). Three local graph diffusion models exhibit clear performance degradation w.r.t. topological shifts exacerbated from #1 to #10 testing graphs, while our two models yield consistently low RMSE across environments. In contrast, the non-local diffusion model produces comparably stable performance yet inferior to our models due to its failure of utilizing the observed topological information.

Table 1: Results on Arxiv and Twitch, where we use time and spatial contexts for data splits, respectively. We report the Accuracy  $(\uparrow)$  for three testing sets of Arxiv and average ROC-AUC  $(\uparrow)$  for all testing graphs of Twitch (results for each case are reported in Appendix G.1). Top performing methods are marked as first/second/third. OOM indicates out-of-memory error.  

<table><tr><td></td><td>Arxiv (2018)</td><td>Arxiv (2019)</td><td>Arxiv (2020)</td><td>Twitch (avg)</td></tr><tr><td>MLP (Rumelhart et al., 1986)</td><td>49.91 ± 0.59</td><td>47.30 ± 0.63</td><td>46.78 ± 0.98</td><td>61.12 ± 0.16</td></tr><tr><td>GCN (Kipf &amp; Welling, 2017)</td><td>50.14 ± 0.46</td><td>48.06 ± 1.13</td><td>46.46 ± 0.85</td><td>59.76 ± 0.34</td></tr><tr><td>GAT (Velickovic et al., 2018)</td><td>51.60 ± 0.43</td><td>48.60 ± 0.28</td><td>46.50 ± 0.21</td><td>59.14 ± 0.72</td></tr><tr><td>SGC (Wu et al., 2019)</td><td>51.40 ± 0.10</td><td>49.15 ± 0.16</td><td>46.94 ± 0.29</td><td>60.86 ± 0.13</td></tr><tr><td>GDC (Klicpera et al., 2019)</td><td>51.53 ± 0.42</td><td>49.02 ± 0.51</td><td>47.33 ± 0.60</td><td>61.36 ± 0.10</td></tr><tr><td>GRAND (Chamberlain et al., 2021a)</td><td>52.45 ± 0.27</td><td>50.18 ± 0.18</td><td>48.01 ± 0.24</td><td>61.65 ± 0.23</td></tr><tr><td>GraphTrans (Wu et al., 2021)</td><td>OOM</td><td>OOM</td><td>OOM</td><td>61.65 ± 0.23</td></tr><tr><td>GraphGPS (Rampásek et al., 2022)</td><td>51.11 ± 0.19</td><td>48.91 ± 0.34</td><td>46.46 ± 0.95</td><td>62.13 ± 0.34</td></tr><tr><td>DIFFformer (Wu et al., 2023)</td><td>50.45 ± 0.94</td><td>47.37 ± 1.58</td><td>44.30 ± 2.02</td><td>62.11 ± 0.11</td></tr><tr><td>ADIT-SERIES</td><td>53.41 ± 0.48</td><td>51.53 ± 0.60</td><td>49.64 ± 0.54</td><td>62.51 ± 0.07</td></tr></table>

# 5.2 REAL-WORLD DATASETS

We proceed to evaluate ADiT beyond the synthetic cases and experiment on real-world datasets with more complex shifts in graph topologies encountered in diverse and broad applications.

Information Networks. We first consider node classification on citation networks Arxiv (Hu et al., 2020) and social networks Twitch (Rozemberczki et al., 2021) with graph sizes ranging from 2K to 0.2M, where we use the scalable version ADIT-SERIES. To introduce topological shifts, we partition the data according to publication years and geographic information for Arxiv and Twitch, respectively. The predictive task is node classification, and we follow the common practice comparing Accuracy (resp. ROC-AUC) for Arxiv (resp. Twitch). We compare with three types of state-of-the-art baselines: (i) classical GNNs (GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2018) and SGC (Wu et al., 2019)); (ii) diffusion-based GNNs (GDC (Klicpera et al., 2019) and GRAND (Chamberlain et al., 2021a)), and (iii) graph Transformers (GraphTrans (Wu et al., 2021), GraphGPS (Rampasek et al., 2022), and the diffusion-based DIFFormer (Wu et al., 2023)). Appendix F.2 presents detailed descriptions for these models. Table 1 reports the results, showing that our model offers significantly superior generalization for node classification.

Molecular Property Prediction. We next study graph classification for predicting molecular properties on OGB-BACE and OGB-SIDER. We follow the scaffold-based splits by Hu et al. (2020), which guarantee structural diversity across training and test sets and provide a realistic estimate of model generalization in prospective experimental settings (Yang et al., 2019). The performance is measured by ROC-AUC. Table 2 reports the results, showing that our model outperforms classical GNNs and powerful graph Transformers² that use the same input data and training loss.

Protein Interactions. We then test on protein-protein interactions of yeast cells (Fu & He, 2022). Each node denotes a protein with a time-aware gene expression value and the edges indicate co-expressed protein pairs at each time. The dataset consists of 12 dynamic networks each of which is

Table 2: ROC-AUC (↑) on two molecule datasets OGB-BACE and OGB-SIDER with scaffold splits for training/validation/testing, where the task is to predict molecular graph properties.  

<table><tr><td rowspan="2"></td><td colspan="3">OGB-BACE</td><td colspan="3">OGB-SIDER</td></tr><tr><td>Train</td><td>Valid</td><td>Test</td><td>Train</td><td>Valid</td><td>Test</td></tr><tr><td>MLP</td><td>67.78 ± 0.01</td><td>65.31 ± 0.00</td><td>66.80 ± 0.01</td><td>71.83 ± 2.07</td><td>57.72 ± 0.16</td><td>57.98 ± 0.23</td></tr><tr><td>GCN</td><td>93.58 ± 0.43</td><td>67.83 ± 0.39</td><td>80.93 ± 0.59</td><td>76.21 ± 0.10</td><td>61.84 ± 0.18</td><td>59.87 ± 0.14</td></tr><tr><td>GAT</td><td>91.67 ± 1.85</td><td>79.31 ± 1.27</td><td>78.18 ± 1.43</td><td>80.26 ± 0.03</td><td>61.88 ± 0.10</td><td>58.99 ± 0.06</td></tr><tr><td>GraphTrans</td><td>96.96 ± 0.59</td><td>71.76 ± 1.53</td><td>80.12 ± 0.58</td><td>97.67 ± 1.22</td><td>62.46 ± 0.85</td><td>60.73 ± 1.97</td></tr><tr><td>GraphGPS</td><td>68.24 ± 2.18</td><td>66.54 ± 2.44</td><td>73.46 ± 0.30</td><td>74.97 ± 1.06</td><td>60.87 ± 0.07</td><td>61.71 ± 0.07</td></tr><tr><td>DIFFformer</td><td>95.97 ± 0.97</td><td>74.48 ± 1.31</td><td>79.67 ± 0.87</td><td>89.94 ± 3.57</td><td>64.13 ± 0.58</td><td>60.94 ± 2.17</td></tr><tr><td>ADiT-INVERSE</td><td>97.39 ± 1.67</td><td>73.82 ± 1.45</td><td>80.38 ± 1.40</td><td>83.67 ± 0.09</td><td>60.85 ± 0.22</td><td>65.29 ± 0.16</td></tr><tr><td>ADiT-SERIES</td><td>93.58 ± 0.46</td><td>67.03 ± 0.53</td><td>82.03 ± 0.42</td><td>80.24 ± 0.23</td><td>59.70 ± 0.35</td><td>62.28 ± 0.36</td></tr></table>

Table 3: Results on dynamic protein interaction networks DDPIN with splits by different protein identification methods. The predictive tasks span node regression, edge regression and link prediction.  

<table><tr><td></td><td colspan="2">Node Regression (RMSE) (↓)</td><td colspan="2">Edge Regression (RMSE) (↓)</td><td colspan="2">Link Prediction (ROC-AUC) (↑)</td></tr><tr><td></td><td>Valid</td><td>Test</td><td>Valid</td><td>Test</td><td>Valid</td><td>Test</td></tr><tr><td>MLP</td><td>2.44 ± 0.02</td><td>2.34 ± 0.03</td><td>0.163 ± 0.004</td><td>0.185 ± 0.003</td><td>0.658 ± 0.014</td><td>0.616 ± 0.117</td></tr><tr><td>GCN</td><td>3.74 ± 0.01</td><td>3.40 ± 0.01</td><td>0.170 ± 0.004</td><td>0.184 ± 0.004</td><td>0.673 ± 0.088</td><td>0.683 ± 0.062</td></tr><tr><td>GAT</td><td>3.10 ± 0.09</td><td>2.86 ± 0.06</td><td>0.164 ± 0.001</td><td>0.176 ± 0.001</td><td>0.765 ± 0.023</td><td>0.687 ± 0.031</td></tr><tr><td>SGC</td><td>3.66 ± 0.00</td><td>3.40 ± 0.02</td><td>0.177 ± 0.016</td><td>0.190 ± 0.004</td><td>0.658 ± 0.044</td><td>0.775 ± 0.042</td></tr><tr><td>GraphTrans</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>GraphGPS</td><td>1.80 ± 0.01</td><td>1.65 ± 0.02</td><td>0.165 ± 0.016</td><td>0.159 ± 0.007</td><td>0.604 ± 0.029</td><td>0.673 ± 0.068</td></tr><tr><td>DIFFformer</td><td>2.06 ± 0.04</td><td>2.04 ± 0.02</td><td>0.173 ± 0.012</td><td>0.155 ± 0.002</td><td>0.935 ± 0.030</td><td>0.902 ± 0.054</td></tr><tr><td>ADiT-INVERSE</td><td>1.83 ± 0.02</td><td>1.75 ± 0.02</td><td>0.146 ± 0.002</td><td>0.147 ± 0.002</td><td>0.946 ± 0.027</td><td>0.957 ± 0.018</td></tr><tr><td>ADiT-SERIES</td><td>1.56 ± 0.02</td><td>1.49 ± 0.03</td><td>0.146 ± 0.002</td><td>0.144 ± 0.001</td><td>0.828 ± 0.026</td><td>0.866 ± 0.036</td></tr></table>

![](images/31672ef05fc8482c7648477f29fa29470218764122b87f5206887468bf8be366.jpg)

![](images/0a6114c823c1ad6afa6f2aeab76566cc5ccf51426bcb82a9f7c21ddbbd63cb12.jpg)

![](images/e0127c4868932b1abbe5a9a1efba3bf04584bb760e807c462f18e2a23270f45e.jpg)

![](images/bb3198d5cbf171b4f89266079c707a002d17be1c6d3c142beb3d39888af98e67.jpg)

![](images/105816f48e7e280c789ef63fac381f951d73854e280961a1cf54c77c922d0464.jpg)

![](images/04c4797977bf9cdfebaf8c69a672d3ed8e3612754af2582d0368d26bee90a6c4.jpg)

![](images/f1570699002a2f67035c7043f0836d4f84683fc122cf87d6132b99e444033035.jpg)  
Ground Truth

![](images/4779b71dcf37e74c3746402620834d4f2da33be7ff3aa5ae828758e0cf2896ad.jpg)  
ADI T (0.697)

![](images/83c44cff1f1ee6cc22246add1fa62e9c7c7f76a1ccbabbb904aac33075aceb75.jpg)  
Figure 4: Testing cases for molecular mapping operators generated by different models with averaged testing Accuracy  $(\uparrow)$  reported. The task is to generate subgraph-level partitions resembling expert annotations (ground-truth) for each molecule instance. See more results in Appendix G.1.  
GCN (0.685)

![](images/66d032793f0d13fe38c3677436646c94cde542b33f83890b8457880ac1b36e2f.jpg)  
GAT (0.664)

![](images/c54788d488cb5ca18f171b3509a6b2636274202c82ae2c0f99aeed0f849a1251.jpg)  
GraphGPS (0.694) Differmer (0.674)

![](images/b2d92fb36cb5e85c41be845bbff0deb76f904e5dfcadd46ab31be02149fc9cc5.jpg)

obtained by one protein identification method and records the metabolic cycles of yeast cells. The networks have distinct topological features (e.g., distribution of cliques) as observed by (Fu & He, 2022), and we use 6/1/5 networks for train-valid/test. To test the generalization of the model across different tasks, we consider: i) node regression for gene expression values (measured by RMSE); 2) edge regression for predicting the co-expression correlation coefficients (measured by RMSE); 3) link prediction for identifying co-expressed protein pairs (measured by ROC-AUC). Table 3 shows that our models yield the first-ranking results in three tasks. In contrast, ADiT-SERIES performs better in node/edge regression tasks, while ADiT-INVERSE exhibits better competitiveness for link prediction. The possible reason might be that ADiT-INVERSE can better exploit high-order structural information as the matrix inverse can be treated as ADiT-SERIES with  $K \to \infty$ .

Molecular Mapping Operator Generation. Finally we investigate on the generation of molecular coarse-grained mapping operators, an important step for molecular dynamics simulation, aiming to find a representation of how atoms are grouped in a molecule (Li et al., 2020). The task is a graph segmentation problem which can be modeled as predicting edges that indicate where to partition the graph. We use the relative molecular mass to split the data and test the model's extrapolation ability for larger molecules. Fig. 4 compares the testing cases (with more cases in Appendix G.1) generated by different models, which shows the more accurate estimation of our model (we use ADIT-SERIES for experiments) that demonstrates desired generalization.

Additional Experimental Results. Due to space limit, we defer more results such as ablation studies and hyper-parameter analysis (for  $\beta, \theta$  and  $K$ ) along with more discussions to Appendix G.2.

# 6 CONCLUSIONS AND DISCUSSIONS

This paper has systematically studied the generalization capabilities of graph diffusion equations under topological shifts, and shed lights on building generalizable GNNs in the open-world regime. The latter remains a largely under-explored question in graph ML community. Our new model, inspired by advective diffusion equations, has provable topological generalization capability and is implemented as a Transformer-like architecture. It shows superior performance in various graph learning tasks. Our analysis and proposed methodology open new possibilities of leveraging established PDE techniques for building generalizable GNNs.

Reproducibility Statement. We supplement the complete proofs for all the theoretical results and detailed information for model implementations and experiments, with references below:

- The proofs for technical results in Sec. 3 are presented in Appendix B.  
- The proofs for technical results in Sec. 4 are presented in Appendix C.  
- The detailed derivations for our proposed models in Sec. 4.3 are shown in Appendix D.  
- The architectures of our models along with pseudo codes are illustrated in Appendix E.  
- The detailed information for all experimental datasets is presented in Appendix F.1.  
- The details for competitors are provided in Appendix F.2.  
- The implementation details for experiments are provided in Appendix F.3.

The source codes will be made publicly available.

# REFERENCES

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.  
James Atwood and Don Towsley. Diffusion-convolitional neural networks. In Advances in Neural Information Processing Systems, pp. 1993-2001, 2016.  
Muhammet Balcilar, Guillaume Renton, Pierre Héroux, Benoit Gauzère, Sébastien Adam, and Paul Honeine. Analyzing the expressive power of graph neural networks in a spectral perspective. In International Conference on Learning Representations, 2021.  
Gleb Bazhenov, Denis Kuznedev, Andrey Malinin, Artem Babenko, and Liudmila Prokhorenkova. Evaluating robustness and uncertainty of graph models under structural distributional shifts. arXiv preprint arXiv:2302.13875, 2023.  
Cristian Bodnar, Francesco Di Giovanni, Benjamin Chamberlain, Pietro Liò, and Michael Bronstein. Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns. Advances in Neural Information Processing Systems, 35:18527-18541, 2022.  
Giorgos Bouritsas, Fabrizio Frasca, Stefanos Zafeiriou, and Michael M. Bronstein. Improving graph neural network expressivity via subgraph isomorphism counting. IEEE Trans. Pattern Anal. Mach. Intell., 45(1):657-668, 2023.  
Ben Chamberlain, James Rowbottom, Maria I. Gorinova, Michael M. Bronstein, Stefan Webb, and Emanuele Rossi. GRAND: graph neural diffusion. In International Conference on Machine Learning (ICML), pp. 1407-1418, 2021a.  
Benjamin Paul Chamberlain, James Rowbottom, Davide Eynard, Francesco Di Giovanni, Xiaowen Dong, and Michael M. Bronstein. Beltrami flow and neural diffusion on graphs. In Advances in Neural Information Processing Systems (NeurIPS), 2021b.  
Subrahmanyan Chandrasekhar. Stochastic problems in physics and astronomy. Reviews of modern physics, 15(1):1, 1943.  
Emmanuel Chasseigne, Manuela Chaves, and Julio D Rossi. Asymptotic behavior for nonlocal diffusion equations. Journal de mathématiques pures et appliquées, 86(3):271-291, 2006.  
Jeongwhan Choi, Seoyoung Hong, Noseong Park, and Sung-Bae Cho. Gread: Graph neural reaction-diffusion equations. In International Conference on Machine Learning, 2023.  
Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021.

Francesco Di Giovanni, James Rowbottom, Benjamin Paul Chamberlain, Thomas Markovich, and Michael M Bronstein. Graph neural networks as gradient flows: understanding graph convolutions via energy. 2022.  
Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. CoRR, abs/2012.09699, 2020.  
James Eells and Joseph H Sampson. Harmonic mappings of riemannian manifolds. American journal of mathematics, 86(1):109-160, 1964.  
Mark I Freidlin and Alexander D Wentzell. Diffusion processes on graphs and the averaging principle. The Annals of probability, pp. 2215-2245, 1993.  
Dongqi Fu and Jingrui He. Dppin: A biological repository of dynamic protein-protein interaction network data. In 2022 IEEE International Conference on Big Data (Big Data), pp. 5269-5277. IEEE, 2022.  
Efstratios Gallopoulos and Yousef Saad. Efficient solution of parabolic equations by krylov approximation methods. SIAM journal on scientific and statistical computing, 13(5):1236-1264, 1992.  
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In International Conference on Machine Learning, pp. 1263-1272, 2017.  
Gene H Golub and Charles F Van Loan. Matrix computations. John Hopkins University Press, 1989.  
Roger A Horn and Charles R Johnson. Matrix analysis. Cambridge university press, 2012.  
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989.  
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, 2020.  
Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. Ogb-lsc: A large-scale challenge for machine learning on graphs. arXiv preprint arXiv:2103.09430, 2021.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.  
Johannes Klicpera, Stefan Weißenberger, and Stephan Gunnemann. Diffusion improves graph learning. In Advances in neural information processing systems, 2019.  
Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M. Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (ICML), pp. 5637-5664, 2021.  
Randall J Leveque. Numerical methods for conservation laws, volume 214. Springer, 1992.  
Zhiheng Li, Geemi P Wellawatte, Maghesree Chakraborty, Heta A Gandhi, Chenliang Xu, and Andrew D White. Graph neural network based coarse-grained mapping prediction. Chemical science, 11(35):9524-9531, 2020.  
László Lovász and Balázs Szegedy. Limits of dense graph sequences. Journal of Combinatorial Theory, Series B, 96(6):933-957, 2006.  
Georgi S Medvedev. The nonlinear heat equation on dense graphs and graph limits. SIAM Journal on Mathematical Analysis, 46(4):2743-2766, 2014.

Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In AAAI Conference on Artificial Intelligence, pp. 4602-4609, 2019.  
Pál András Papp, Karolis Martinkus, Lukas Faber, and Roger Wattenhofer. *Dropgnn: Random dropouts increase the expressiveness of graph neural networks*. In *Advances in Neural Information Processing Systems*, pp. 21997–22009, 2021.  
Giuseppe Patané. Laplacian spectral distances and kernels on 3d shapes. Pattern Recognition Letters, 47:102-110, 2014.  
Ladislav Rampásek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2022.  
Mateo Rojas-Carulla, Bernhard Schölkopf, Richard E. Turner, and Jonas Peters. Invariant models for causal transfer learning. Journal of Machine Learning Research, 19:36:1-36:34, 2018.  
Bart M Haar Romeny. Geometry-driven diffusion in computer vision, volume 1. Springer Science & Business Media, 2013.  
Emanuele Rossi, Henry Kenlay, Maria I Gorinova, Benjamin Paul Chamberlain, Xiaowen Dong, and Michael M Bronstein. On the unreasonable effectiveness of feature propagation in learning on graphs with missing node features. In Learning on Graphs Conference, pp. 11-1. PMLR, 2022.  
Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-scale attributed node embedding. Journal of Complex Networks, 9(2), 2021.  
David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533-536, 1986.  
T Konstantin Rusch, Benjamin P Chamberlain, Michael W Mahoney, Michael M Bronstein, and Siddhartha Mishra. Gradient gating for deep multi-rate learning on graphs. In International Conference on Learning Representations, 2023.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008.  
Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612-634, 2021.  
Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.  
Tom AB Snijders and Krzysztof Nowicki. Estimation and prediction for stochastic blockmodels for graphs with latent block structure. Journal of classification, 14(1):75-100, 1997.  
Matthew Thorpe, Hedi Xia, Tan Nguyen, Thomas Strohmer, Andrea L. Bertozzi, Stanley J. Osher, and Bao Wang. GRAND++: graph neural diffusion with a source term. In International Conference on Learning Representations (ICLR), 2022.  
Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, and Michael M. Bronstein. Understanding over-squashing and bottlenecks on graphs via curvature. In International Conference on Learning Representations, 2022.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2017.  
Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations (ICLR), 2018.

Felix Wu, Amauri H. Souza Jr., Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Q. Weinberger. Simplifying graph convolutional networks. In International Conference on Machine Learning, pp. 6861-6871, 2019.  
Qitian Wu, Hengrui Zhang, Junchi Yan, and David Wipf. Handling distribution shifts on graphs: An invariance perspective. In International Conference on Learning Representations, 2022a.  
Qitian Wu, Wentao Zhao, Zenan Li, David Wipf, and Junchi Yan. Nodeformer: A scalable graph structure learning transformer for node classification. In Advances in Neural Information Processing Systems, 2022b.  
Qitian Wu, Chenxiao Yang, Wentao Zhao, Yixuan He, David Wipf, and Junchi Yan. Differmer: Scalable (graph) transformers induced by energy constrained diffusion. In International Conference on Learning Representations, 2023.  
Zhanghao Wu, Paras Jain, Matthew A. Wright, Azalia Mirhoseini, Joseph E. Gonzalez, and Ion Stoica. Representing long-range context for graph neural networks with global attention. In Advances in Neural Information Processing Systems, 2021.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.  
Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao, Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, and et al. Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling, 59(8): 3370-3388, 2019.  
Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform bad for graph representation? In Advances in Neural Information Processing Systems, 2021.  
Xuan Zhang, Limei Wang, Jacob Helwig, Youzhi Luo, Cong Fu, Yaochen Xie, Meng Liu, Yuchao Lin, Zhao Xu, Keqiang Yan, et al. Artificial intelligence for science in quantum, atomistic, and continuum systems. arXiv preprint arXiv:2307.08423, 2023.
