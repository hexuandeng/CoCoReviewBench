# LEARNING DIRECTED GRAPHICAL MODELS WITH OPTIMAL TRANSPORT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Estimating the parameters of a probabilistic directed graphical model from incomplete data remains a long-standing challenge. This is because, in the presence of latent variables, both the likelihood function and posterior distribution are intractable without further assumptions about structural dependencies or model classes. While existing learning methods are fundamentally based on likelihood maximization, here we offer a new view of the parameter learning problem through the lens of optimal transport. This perspective licenses a general framework that operates on any directed graphs without making unrealistic assumptions on the posterior over the latent variables or resorting to black-box variational approximations. We develop a theoretical framework and support it with extensive empirical evidence demonstrating the flexibility and versatility of our approach. Across experiments, we show that not only can our method recover the ground-truth parameters but it also performs comparably or better on downstream applications, notably the non-trivial task of discrete representation learning.

# 1 INTRODUCTION

Learning probabilistic directed graphical models (DGMs, also known as Bayesian networks) with latent variables is an important ongoing challenge in machine learning and statistics. This paper focuses on parameter learning, i.e., estimating the parameters of a DGM given its known structure. Learning DGMs has a long history, dating back to classical indirect likelihood-maximization approaches such as expectation maximization (EM, Dempster et al., 1977). However, despite all its success stories, EM is well-known to suffer from local optima issues. More importantly, EM becomes inapplicable when the posterior distribution is intractable, which arises fairly often in practice.

A large family of related methods based on variational inference (VI, Jordan et al., 1999; Hoffman et al., 2013) have demonstrated tremendous potential in this case, where the evidence lower bound (ELBO) is not only used for posterior approximation but also for point estimation of the model parameters. Such an approach has proved surprisingly effective and robust to overfitting, especially when having a small number of parameters. From a high-level perspective, both EM and VI are based on likelihood maximization in the presence of latent variables, which ultimately requires carrying out expectations over the commonly intractable posterior. In order to address this challenge, a large spectrum of methods have been proposed in the literature and we refer the reader to Ambrogioni et al. (2021) for an excellent discussion of these approaches. Here we characterize them between two extremes. At one extreme, restrictive assumptions about the structure (e.g., as in mean-field approximations) or the model class (e.g., using conjugate exponential families) must be made to simplify the task. At the other extreme, when no assumptions are made, most existing black-box methods exploit very little information about the structure of the known probabilistic model (e.g., in black-box and stochastic VI (Ranganath et al., 2014; Hoffman et al., 2013), hierarchical approaches (Ranganath et al., 2016) or normalizing flows (Papamakarios et al., 2021)). Recently, VI has taken a significant leap forward by embracing amortized inference (Amos, 2022), which allows black-box optimization to be done in a considerably more efficient way.

Since the ultimate goal of VI is posterior inference, parameter estimation has been treated as a by-product of the optimization process where the model parameters are jointly updated with the variational parameters. As the complexity of the graph increases, despite the current advancements, parameter estimation in VI becomes less straightforward and computationally challenging.

Bridging this gap, we propose a scalable framework dedicated to learning parameters of a general directed graphical model. This alternative strategy inherits the flexibility of amortized optimization while eliminating the need to estimate expectations over the posterior distribution. Concretely, parameter learning is now viewed through the lens of optimal transport (Villani et al., 2009), where the data distribution is the source and the true model distribution is the target. Instead of minimizing a Kullback-Leibler (KL) divergence (which likelihood maximization methods are essentially doing), we aim to find a point estimate  $\theta^{*}$  that minimizes the Wasserstein (WS) distance (Kantorovich, 1960) between these two distributions.

This perspective allows us to leverage desirable properties of WS distance in comparison with other metrics. These properties have motivated the recent surge in generative models, e.g., Wasserstein GANs (Adler & Lunz, 2018; Arjovsky et al., 2017) and Wasserstein Auto-encoders (WAE, Tolstikhin et al., 2017). Indeed, WS distance is shown to be well-behaved in situations where standard metrics such as the KL or JS (Jensen-Shannon) divergences are either infinite or undefined (Peyre et al., 2017; Ambrogioni et al., 2018). WS distance thus characterizes a more meaningful distance, especially when the two distributions reside in low-dimensional manifolds (Arjovsky et al., 2017).

Interestingly, akin to how Variational Auto-encoders (VAE, Kingma & Welling, 2013) is related to VI, our framework can be viewed as an extension of WAE for learning the parameters of a directed graphical model that can effectively exploit its structure. The parameter learning landscape is summarized in Figure 1.

Contributions. We present an entirely different view that casts parameter estimation as an optimal transport problem (Villani et al., 2009), where the goal is to find the optimal plan transporting "mass" from the data distribution to the model distribution. This permits a flexible framework applicable to any type of variable and graphical structure. In summary, we make the following contributions:

- We introduce OTP-DAG - an Optimal Transport framework for Parameter Learning in Directed Acyclic Graphical models<sup>1</sup>. OTP-DAG is an alternative line of thinking about parameter learning. Diverging from the existing frameworks, the underlying idea is to find the parameter set associated with the distribution that yields the lowest transportation cost from the data distribution.  
- We present theoretical developments showing that minimizing the transport cost is equivalent to minimizing the reconstruction error between the observed data and the model generation. This renders a tractable training objective to be solved efficiently with stochastic gradient descent.  
- We provide empirical evidence demonstrating the versatility of our method on various graphical structures. OTP-DAG is shown to successfully recover the ground-truth parameters and achieve comparable or better performance than competing methods across a range of downstream applications.

# 2 PRELIMINARIES

We first introduce the notations and basic concepts used throughout the paper. We reserve bold capital letters (i.e.,  $\mathbf{G}$ ) for notations related to graphs. We use calligraphic letters (i.e.  $\mathcal{X}$ ) for spaces, italic capital letters (i.e.  $X$ ) for random variables, and lower case letters (i.e.  $x$ ) for their values.

A directed graph  $\mathbf{G} = (\mathbf{V},\mathbf{E})$  consists of a set of nodes  $\mathbf{V}$  and an edge set  $\mathbf{E} \subseteq \mathbf{V}^2$  of ordered pairs of nodes with  $(v,v) \notin \mathbf{E}$  for any  $v \in \mathbf{V}$  (one without self-loops). For a pair of nodes  $i,j$  with  $(i,j) \in \mathbf{E}$ , there is an arrow pointing from  $i$  to  $j$  and we write  $i \rightarrow j$ . Two nodes  $i$  and  $j$  are adjacent if either  $(i,j) \in \mathbf{E}$  or  $(j,i) \in \mathbf{E}$ . If there is an arrow from  $i$  to  $j$  then  $i$  is a parent of  $j$  and  $j$  is a child of  $i$ . A Bayesian network structure  $\mathbf{G} = (\mathbf{V},\mathbf{E})$  is a directed acyclic graph (DAG), in which the nodes represent random variables

![](images/f9abbe1137dcba05145115242a4d29f7258db39b50995f9bb601d4e69e7564f8.jpg)  
Figure 1: Notable parameter learning methods along the two lines of approaches. OTP-DAG can be viewed as an extension of WAE for learning parameters of a general directed graph, laying a foundation stone for a new paradigm of learning and, potentially, inference of graphical models.

$X = [X_{i}]_{i=1}^{n}$  with index set  $\mathbf{V} := \{1, \dots, n\}$ . Let  $\mathrm{PA}_{X_i}$  denote the set of variables associated with parents of node  $i$  in  $\mathbf{G}$ . In this work, we tackle the classic yet important problem of learning the parameters of a directed graph from partially observed data. Let  $\mathbf{O} \subseteq \mathbf{V}$  and  $X_{\mathbf{O}} = [X_i]_{i \in \mathbf{O}}$  be the set of observed nodes and  $\mathbf{H} := \mathbf{V} \backslash \mathbf{O}$  be the set of hidden nodes. Let  $P_{\theta}$  and  $P_d$  respectively denote the distribution induced by the graphical model and the empirical one induced by the complete (yet unknown) data. Given a fixed graphical structure  $\mathbf{G}$  and some set of i.i.d data points, we aim to find the point estimate  $\theta^*$  that best fits the observed data  $X_{\mathbf{O}}$ . The conventional approach is to minimize the KL divergence between the model distribution and the empirical data distribution over observed data i.e.,  $D_{\mathrm{KL}}(P_d(X_{\mathbf{O}}), P_{\theta}(X_{\mathbf{O}}))$ , which is equivalent to maximizing the likelihood  $P_{\theta}(X_{\mathbf{O}})$  w.r.t  $\theta$ . In the presence of latent variables, the marginal likelihood, given as  $P_{\theta}(X_{\mathbf{O}}) = \int_{X_{\mathbf{H}}} P_{\theta}(X) dX_{\mathbf{H}}$ , is generally intractable. Standard approaches then resort to maximizing a bound on the marginal log-likelihood, known as the evidence lower bound (ELBO), which is essentially the objective of EM (Moon, 1996) and VI (Jordan et al., 1999). Optimization of the ELBO for parameter learning in practice requires many considerations. We refer readers to Appendix B for a review of these intricacies.

# 3 OPTIMAL TRANSPORT FOR LEARNING DIRECTED GRAPHICAL MODELS

We begin by explaining how parameter learning can be reformulated into an optimal transport problem. Villani (2003) and thereafter introduce our novel theoretical contribution.

We consider a DAG  $\mathbf{G}(\mathbf{V},\mathbf{E})$  over random variables  $X = [X_{i}]_{i = 1}^{n}$  that represents the data generative process of an underlying system. The system consists of  $X$  as the set of endogenous variables and  $U = \{U_i\}_{i = 1}^n$  as the set of exogenous variables representing external factors affecting the system. Associated with every  $X_{i}$  is an exogenous variable  $U_{i}$  whose values are sampled from a prior distribution  $P(U)$  independently from the other exogenous variables. For the purpose of theoretical development, our framework operates on an extended graph consisting of both endogenous and exogenous nodes (See Figure 2b). In the graph  $\mathbf{G}$ ,  $U_{i}$  is represented by a node with no ancestors that has an outgoing arrow towards node  $i$ . Every distribution  $P_{\theta_i}(X_i|\mathrm{PA}_{X_i})$  henceforth can be reparameterized into a deterministic assignment

$$
X _ {i} = \psi_ {i} \left(\mathrm {P A} _ {X _ {i}}, U _ {i}\right), \text {f o r} i = 1, \dots , n.
$$

The ultimate goal is to estimate  $\theta = \{\theta_i\}_{i=1}^n$  as the parameters of the set of deterministic functions  $\psi = \{\psi_i\}_{i=1}^n$ . We will use the notation  $\psi_\theta$  to emphasize this connection from now on. Given the empirical data distribution  $P_d(X_{\mathbf{O}})$  and the model distribution  $P_\theta(X_{\mathbf{O}})$  over the observed set  $\mathbf{O}$ , the optimal transport (OT) goal is to find the parameter set  $\theta$  that minimizes the cost of transport between these two distributions. The Kantorovich's formulation of the problem is given by

$$
W _ {c} \left(P _ {d}; P _ {\theta}\right) := \inf  _ {\Gamma \sim \mathcal {P} (X \sim P _ {d}, Y \sim P _ {\theta})} \mathbb {E} _ {(X, Y) \sim \Gamma} [ c (X, Y) ], \tag {1}
$$

where  $\mathcal{P}(X\sim P_d,Y\sim P_\theta)$  is a set of all joint distributions of  $(P_d;P_\theta)$ ;  $c:\mathcal{X}_{\mathbf{O}}\times \mathcal{X}_{\mathbf{O}}\mapsto \mathcal{R}_+$  is any measurable cost function over  $\mathcal{X}_{\mathbf{O}}$  (i.e., the product space of the spaces of observed variables) defined as  $c(X_{\mathbf{O}},Y_{\mathbf{O}})\coloneqq \sum_{i\in \mathbf{O}}c_i(X_i,Y_i)$  where  $c_{i}$  is a measurable cost function over a space of an observed variable. Let  $P_{\theta}(\mathrm{PA}_{X_i},U_i)$  denote the joint distribution of  $\mathrm{PA}_{X_i}$  and  $U_{i}$  factorized according to the graphical model. Let  $\mathcal{U}_i$  denote the space over random variable  $U_{i}$ . The key ingredient of our theoretical development is local backward mapping. For every observed node  $i\in \mathbf{O}$ , we define a stochastic "backward" map  $\phi_i:\mathcal{X}_i\mapsto \Pi_{k\in \mathrm{PA}_{X_i}}\mathcal{X}_k\times \mathcal{U}_i$  such that  $\phi_i\in \mathfrak{C}(X_i)$  where  $\mathfrak{C}(X_i)$  is the constraint set given as

$$
\mathfrak {C} \left(X _ {i}\right) := \left\{\phi_ {i}: \phi_ {i} \# P _ {d} \left(X _ {i}\right) = P _ {\theta} \left(\mathrm {P A} _ {X _ {i}}, U _ {i}\right) \right\};
$$

that is, every backward  $\phi_i\#$  defines a push forward operator such that the samples from  $\phi_i(X_i)$  follow the marginal distribution  $P_{\theta}(\mathrm{PA}_{X_i},U_i)$ .

![](images/ddac29266c4620b0eb2715728f7a190b56bdf3310b211acf236208db4e3a8c4f.jpg)  
(a) DAG

![](images/f89f0c4bca12264457949ac7efa816c45afdf542d8c6949d3ed267443824171d.jpg)  
(b) Extended DAG

![](images/277575754945c3d048fad89fa5110d12adbf360ce71e2bb25a3c1c0986a7f525.jpg)  
Figure 2: (a) A DAG represents a system of 4 endogenous variables where  $X_{1}, X_{3}$  are observed (black-shaded) and  $X_{2}, X_{4}$  are hidden variables (non-shaded). (b): The extended DAG includes an additional set of independent exogenous variables  $U_{1}, U_{2}, U_{3}, U_{4}$  (grey-shaded) acting on each endogenous variable.  $U_{1}, U_{2}, U_{3}, U_{4} \sim P(U)$  where  $P(U)$  is a prior product distribution. (c) Visualization of our backward-forward algorithm, where the dashed arcs represent the backward maps involved in optimization.  
(c) Algorithmic DAG

Theorem 1 presents the main theoretical contribution of our paper. Our OT problem seeks to find the optimal set of deterministic "forward" maps  $\psi_{\theta}$  and stochastic "backward" maps  $\{\phi_i \in \mathfrak{C}(X_i)\}_{i \in \mathbf{O}}$  that minimize the cost of transporting the mass from  $P_d$  to  $P_{\theta}$  over  $\mathbf{O}$ . While the formulation in Eq. (1) is not trainable, we show that the problem is reduced to minimizing the reconstruction error between the data generated from  $P_{\theta}$  and the observed data. To understand how reconstruction works, let us examine Figure 2c.

With a slight abuse of notations, for every  $X_{i}$ , we extend its parent set  $\mathrm{PA}_{X_i}$  to include an exogenous variable and possibly some other endogenous variables. Given  $X_{1}$  and  $X_{3}$  as observed nodes, we first sample  $X_{1} \sim P_{d}(X_{1}), X_{3} \sim P_{d}(X_{3})$  and evaluate the local densities  $P_{\phi_1}(\mathrm{PA}_{X_1}|X_1)$ ,  $P_{\phi_3}(\mathrm{PA}_{X_3}|X_3)$  where  $\mathrm{PA}_{X_1} = \{X_2, X_4, U_1\}$  and  $\mathrm{PA}_{X_3} = \{X_4, U_3\}$ . The next step is to sample  $\mathrm{PA}_{X_1} \sim P_{\phi_1}(\mathrm{PA}_{X_1}|X_1)$  and  $\mathrm{PA}_{X_3} \sim P_{\phi_3}(\mathrm{PA}_{X_3}|X_3)$ , which are plugged back to the model  $\psi_\theta$  to obtain the reconstructions  $\widetilde{X_1} = \psi_{\theta_1}(\mathrm{PA}_{X_1})$  and  $\widetilde{X_3} = \psi_{\theta_3}(\mathrm{PA}_{X_3})$ . We wish to learn  $\theta$  such that  $X_{1}$  and  $X_{3}$  are reconstructed correctly. For a general graphical model, this optimization objective is formalized as

Theorem 1. For every  $\phi_i$  as defined above and fixed  $\psi_{\theta}$

$$
W _ {c} \left(P _ {d} \left(X _ {\mathbf {O}}\right); P _ {\theta} \left(X _ {\mathbf {O}}\right)\right) = \inf  _ {\left[ \phi_ {i} \in \mathfrak {C} \left(X _ {i}\right) \right] _ {i \in \mathbf {O}}} \mathbb {E} _ {X _ {\mathbf {O}} \sim P _ {d} \left(X _ {\mathbf {O}}\right), \mathrm {P A} X _ {\mathbf {O}} \sim \phi \left(X _ {\mathbf {O}}\right)} \left[ c \left(X _ {\mathbf {O}}, \psi_ {\theta} \left(\mathrm {P A} X _ {\mathbf {O}}\right)\right) \right], \tag {2}
$$

where  $\mathrm{PA}_{X_{\mathbf{O}}} := \left[[X_{ij}]_{j \in \mathrm{PA}_{X_i}}\right]_{i \in \mathbf{O}}$ .

The proof is provided in Appendix A. While Theorem 1 set ups a tractable form for our optimization solution, the quality of the reconstruction hinges on how well the backward maps approximate the true local densities. To ensure approximation fidelity, every backward function  $\phi_{i}$  must satisfy its push-forward constraint defined by  $\mathfrak{C}$ . In the above example, the backward maps  $\phi_{i}$  and  $\phi_{3}$  must be constructed such that

$\phi_1 \# P_d(X_1) = P_\theta(X_2, X_4, U_1)$  and  $\phi_3 \# P_d(X_3) = P_\theta(X_4, U_3)$ . This results in a constraint optimization problem, and we relax the constraints by adding a penalty to the above objective.

The final optimization objective is therefore given as

$$
J _ {\mathrm {W S}} = \inf  _ {\psi , \phi} \mathbb {E} _ {X _ {\mathbf {O}} \sim P _ {d} \left(X _ {\mathbf {O}}\right), \mathrm {P A} _ {X _ {\mathbf {O}}} \sim \phi \left(X _ {\mathbf {O}}\right)} \left[ c \left(X _ {\mathbf {O}}, \psi_ {\theta} \left(\mathrm {P A} _ {X _ {\mathbf {O}}}\right)\right) \right] + \eta D \left(P _ {\phi}, P _ {\theta}\right), \tag {3}
$$

where  $D$  is any arbitrary divergence measure and  $\eta > 0$  is a trade-off hyper-parameter.  $D(P_{\phi}, P_{\theta})$  is a short-hand for divergence between all pairs of backward and forward distributions.

Connection with Auto-encoders. OTP-DAG is an optimization-based approach in which we leverage reparameterization and amortized inference (Amos, 2022) for solving it efficiently via stochastic gradient descent. This theoretical result provides us with two interesting properties: (1) all model parameters are optimized simultaneously within a single framework whether the variables are continuous or discrete, and (2) the computational process can be automated without the need for analytic lower bounds (as in EM and traditional VI), specific graphical structures (as in mean-field VI), or priors over variational distributions on latent variables (as in hierarchical VI). The flexibility our method exhibits is akin to VAE, and OTP-DAG in fact serves as an extension of WAE for learning general directed graphical models. Our formulation thus inherits a desirable characteristic from that of WAE, which specifically helps mitigate the posterior collapse issue notoriously occurring to VAE. Appendix D explains this in more detail. Particularly, in the next section, we will empirically show that OTP-DAG effectively alleviates the codebook collapse issue in discrete representation learning. Details on our algorithm can be found in Appendix C.

# 4 APPLICATIONS

In this section, we illustrate the practical application of the OTP-DAG algorithm. Instead of achieving state-of-the-art performance on specific applications, our key objective is to demonstrate the versatility of OTP-DAG: our method can be harnessed for a wide range of purposes in a single learning procedure. In terms of estimation accuracy, OTP-DAG is capable of recovering the ground-truth parameters while achieving the comparable or better performance level of existing frameworks across downstream tasks.

Experimental setup. We consider various directed probabilistic models with either continuous or discrete variables. We begin with (1) Latent Dirichlet Allocation Blei et al. (2003) for topic modeling and (2) Hidden Markov Model (HMM) for sequential modeling. We conclude with a more challenging setting: (3) Discrete Representation Learning (Discrete RepL) that cannot simply be solved by EM or MAP (maximum a posteriori). It in fact invokes deep generative modeling via a pioneering development called Vector Quantization Variational Auto-Encoder (VQ-VAE, Van Den Oord et al., 2017). We attempt to apply OTP-DAG for learning discrete representations by grounding it into a parameter learning problem. Figure 3 illustrates the empirical DAG structures of 3 applications. Unlike the standard visualization where the parameters are considered hidden nodes, our graph separates model parameters from latent variables and only illustrates random variables and their dependencies (except the special setting of Discrete RepL). We also omit the exogenous variables associated with the hidden nodes for visibility, since only those acting on the observed nodes are relevant for computation. There is also a noticeable difference between Figure 3 and Figure 2c: the empirical version does not require learning the backward maps for the exogenous variables. It is observed across our experiments that sampling the noise from an appropriate prior distribution suffices to yield accurate estimation, which is in fact beneficial in that training time can be greatly reduced.

Baselines. We compare OTP-DAG with two groups of parameter learning methods towards the two extremes: (1) MAP, EM and SVI where analytic derivation is required; (2) variational auto-encoding frameworks (the closest baseline to ours) where black-box optimization is permissible. For the latter, we here report the performance of vanilla VAE-based models, while providing additional experiments with some advances in Appendix E. We also leave the discussion of the formulation and technicalities in Appendix E. In all tables, we report the average results over 5 random initializations and the best/second-best ones are bold/underlined. In addition,  $\uparrow$ ,  $\downarrow$  indicate higher/lower performance is better, respectively.

![](images/1f0e27f2c75ea4245f3136d94c64939e2acdd6892024794cfb3f8c928fc364c7.jpg)  
(a) LDA

![](images/090a11a38246ff114757552e47e738ccf599c2fe7af46bb68b085c1ac31eddb5.jpg)  
Figure 3: Empirical structures of (a) latent Dirichlet allocation model (in plate notation), (b) standard hidden Markov model, and (c) discrete representation learning.  
(b) HMM

![](images/8a37ad98767d8dc5688a4f122b8eff6edfe3bc6fb89f3b7317a33073169155fb.jpg)  
(c) Discrete RepL

# 4.1 LATENT DIRICHLET ALLOCATION

Let us consider a corpus  $\mathcal{D}$  of  $M$  independent documents where each document is a sequence of  $N$  words denoted by  $W = (W_{1}, W_{2}, \dots, W_{N})$ . Documents are represented as random mixtures over  $K$  latent topics, each of which is characterized by a distribution over words. Let  $V$  be the size of a vocabulary indexed by  $\{1, \dots, V\}$ . Latent Dirichlet Allocation (LDA) (Blei et al., 2003) dictates the following generative process for every document in the corpus:

1. Sample  $\theta \sim \mathrm{Dir}(\alpha)$  with  $\alpha < 1$  
2. Sample  $\gamma_{k}\sim \mathrm{Dir}(\beta)$  where  $k\in \{1,\dots ,K\}$  
3. For each of the word positions  $n \in \{1, \dots, N\}$ ,

- Sample a topic  $Z_{n} \sim$  Multi-nominal(θ),  
- Sample a word  $W_{n} \sim$  Multi-nominal  $(\gamma_{k})$ ,

where  $\mathrm{Dir}(.)$  is a Dirichlet distribution.  $\theta$  is a  $K$ -dimensional vector that lies in the  $(K - 1)$ -simplex and  $\gamma_{k}$  is a  $V$ -dimensional vector represents the word distribution corresponding to topic  $k$ . In the standard model,  $\alpha, \beta, K$  are hyper-parameters and  $\theta, \gamma$  are learnable parameters. Throughout the experiments, the number of topics  $K$  is assumed known and fixed.

Parameter Estimation. To test whether OTP-DAG can recover the true parameters, we generate synthetic data in the setting: the word probabilities are parameterized by a  $K \times V$  matrix  $\gamma$  where  $\gamma_{kn} \coloneqq P(W_n = 1|Z_n = 1)$ ;  $\gamma$  is now a fixed quantity to be estimated. We set  $\alpha = 1 / K$  uniformly and generate small datasets for different numbers of topics  $K$  and sample size  $N$ . Following Griffiths & Steyvers (2004), for every topic  $k$ , the word distribution  $\gamma_k$  can be represented as a square grid where each cell, corresponding to a word, is assigned an integer value of either 0 and 1, indicating whether a certain word is allocated to the  $k^{th}$  topic or not. As a result, each topic is associated with a specific pattern. For simplicity, we represent topics using horizontal or vertical patterns (See Figure 4a). According to the above generative model, we sample data w.r.t 3 sets of configuration triplets  $\{K,M,N\}$ . We compare OTP-DAG with Batch EM and SVI and Prod LDA - a variational auto-encoding topic model (Srivastava & Sutton, 2017).

Table 1: Fidelity of estimates of the topic-word distribution  $\gamma$  across 3 settings. Fidelity is measured by KL divergence, Hellinger (HL) (Hellinger, 1909) and Wasserstein distance with the ground-truth distributions.  

<table><tr><td>Metric ↓</td><td>K</td><td>M</td><td>N</td><td>OTP-DAG (Ours)</td><td>Batch EM</td><td>SVI</td><td>Prod LDA</td></tr><tr><td>HL</td><td>10</td><td>1,000</td><td>100</td><td>2.327 ± 0.009</td><td>2.807 ± 0.189</td><td>2.712 ± 0.087</td><td>2.353 ± 0.012</td></tr><tr><td>KL</td><td>10</td><td>1,000</td><td>100</td><td>1.701 ± 0.005</td><td>1.634 ± 0.022</td><td>1.602 ± 0.014</td><td>1.627 ± 0.027</td></tr><tr><td>WS</td><td>10</td><td>1,000</td><td>100</td><td>0.027 ± 0.004</td><td>0.058 ± 0.000</td><td>0.059 ± 0.000</td><td>0.052 ± 0.001</td></tr><tr><td>HL</td><td>20</td><td>5,000</td><td>200</td><td>3.800 ± 0.058</td><td>4.256 ± 0.084</td><td>4.259 ± 0.096</td><td>3.700 ± 0.012</td></tr><tr><td>KL</td><td>20</td><td>5,000</td><td>200</td><td>2.652 ± 0.080</td><td>2.304 ± 0.004</td><td>2.305 ± 0.003</td><td>2.316 ± 0.026</td></tr><tr><td>WS</td><td>20</td><td>5,000</td><td>200</td><td>0.010 ± 0.001</td><td>0.022 ± 0.000</td><td>0.022 ± 0.001</td><td>0.018 ± 0.000</td></tr><tr><td>HL</td><td>30</td><td>10,000</td><td>300</td><td>4.740 ± 0.029</td><td>5.262 ± 0.077</td><td>5.245 ± 0.035</td><td>4.723 ± 0.017</td></tr><tr><td>KL</td><td>30</td><td>10,000</td><td>300</td><td>2.959 ± 0.015</td><td>2.708 ± 0.002</td><td>2.709 ± 0.001</td><td>2.746 ± 0.034</td></tr><tr><td>WS</td><td>30</td><td>10,000</td><td>300</td><td>0.005 ± 0.001</td><td>0.012 ± 0.000</td><td>0.012 ± 0.000</td><td>0.009 ± 0.000</td></tr></table>

Table 1 reports the fidelity of the estimation of  $\gamma$ . OTP-DAG consistently achieves high-quality estimates by both Hellinger and Wasserstein distances. It is straightforward that the baselines are superior by the KL metric, as it is what they implicitly minimize. While it is inconclusive from the numerical estimations, the qualitative results complete the story. Figure 4a illustrates the distributions of individual words to the topics from each method after 300 training epochs. OTP-DAG successfully recovers the true patterns and as well as EM and SVI. More qualitative examples for the other settings are presented in Figures 7 and 8 where OTP-DAG is shown to recover almost all true patterns.

Topic Inference. We now demonstrate the effectiveness of OTP-DAG on downstream applications. We here use OTP-DAG to infer the topics of 3 real-world datasets:2 20 News Group, BBC News and DBLP. We revert to the original generative process where the topic-word distribution follows a Dirichlet distribution parameterized by the concentration parameters  $\beta$ , instead of having  $\gamma$  as a fixed quantity.  $\beta$  is now initialized as a matrix of real values  $(\beta \in \mathbb{R}^{K\times V})$  representing the log concentration values.

Table 5 reports the quality of the inferred topics, which is evaluated via the diversity and coherence of the selected words. Diversity refers to the proportion of unique words, whereas Coherence is measured with normalized pointwise mutual information (Aletras & Stevenson, 2013), reflecting the extent to which the words in a topic are associated with a common theme. There exists a trade-off between Diversity and Coherence: words that are excessively diverse greatly reduce coherence, while a set of many duplicated words yields higher coherence yet harms diversity. A well-performing topic model would strike a good balance between these metrics. If we consider two metrics comprehensively, our method achieves comparable or better performance than the other learning algorithms

# 4.2 HIDDEN MARKOV MODELS

This application deals with time-series data following a Poisson hidden Markov model (See Figure 3b). Given a time series of  $T$  steps, the task is to segment the data stream into  $K$  different states, each of which follows a Poisson distribution with rate  $\lambda_{k}$ . The observation at each step  $t$  is given as

$$
P \left(X _ {t} | Z _ {t} = k\right) = \operatorname {P o i} \left(X _ {t} \mid \lambda_ {k}\right), \quad \text {f o r} k = 1, \dots , K.
$$

Following Murphy (2023), we use a uniform prior over the initial state. The Markov chain stays in the current state with probability  $p$  and otherwise transitions to one of the other  $K - 1$  states uniformly at random. The transition distribution is given as

$$
Z _ {1} \sim \operatorname {C a t} \left(\left\{\frac {1}{4}, \frac {1}{4}, \frac {1}{4}, \frac {1}{4} \right\}\right), \quad Z _ {t} | Z _ {t - 1} \sim \operatorname {C a t} \left(\left\{ \begin{array}{c c} p & \text {i f} Z _ {t} = Z _ {t - 1} \\ \frac {1 - p}{4 - 1} & \text {o t h e r w i s e} \end{array} \right\}\right)
$$

Let  $P(Z_{1})$  and  $P(Z_{t}|Z_{t - 1})$  respectively denote these prior transition distributions. We generate a synthetic dataset  $\mathcal{D}$  of 200 observations at rates  $\lambda = \{12,87,60,33\}$  with change points occurring at times (40, 60, 55). We would like to learn the concentration parameters  $\lambda_{1:K} = [\lambda_k]_{k = 1}^K$  through which segmentation can be realized, assuming that the number of states  $K = 4$  is known.

The true transition probabilities are generally unknown. The value  $p$  is treated as a hyper-parameter and we fit HMM with 6 choices of  $p$ . Table 2 demonstrates the quality of our estimates, in comparison with MAP estimates. Our estimation approaches the ground-truth values comparably to MAP. We note that the MAP solution requires the analytical marginal likelihood of the model, which is not necessary for our method. Figure 4b reports the most probable state for each observation, inferred from our backward distribution  $\phi(X_{1:T})$ . It can be seen that the partition overall aligns with the true generative process of the data.

By observing the data, one can assume  $p$  should be relatively high,  $0.75 - 0.95$  seems most reasonable. This explains why the MAP estimation at  $p = 0.05$  is terrible. Meanwhile, for our OTP-DAG, the effect of  $p$  is controlled by the trade-off coefficient  $\eta$ . We here fix  $\eta = 0.1$ . Since the effect is fairly minor, OTP-DAG estimates across  $p$  are less variant. Table 7 additionally analyzes the model performance when  $\eta$  varies.

Table 2: Estimates of  ${\lambda }_{1 : 4}$  at various transition probabilities  $p$  and mean absolute reconstruction error.  

<table><tr><td>p</td><td>λ1=12</td><td>λ2=87</td><td>λ3=60</td><td>λ4=33</td><td>λ1=12</td><td>λ2=87</td><td>λ3=60</td><td>λ4=33</td></tr><tr><td colspan="5">OTP-DAG Estimates (Ours)</td><td colspan="4">MAP Estimates</td></tr><tr><td>0.05</td><td>11.83</td><td>87.20</td><td>60.61</td><td>33.40</td><td>14.88</td><td>85.22</td><td>71.42</td><td>40.39</td></tr><tr><td>0.15</td><td>11.62</td><td>87.04</td><td>59.69</td><td>32.85</td><td>12.31</td><td>87.11</td><td>61.86</td><td>33.90</td></tr><tr><td>0.35</td><td>11.77</td><td>86.76</td><td>60.01</td><td>33.26</td><td>12.08</td><td>87.28</td><td>60.44</td><td>33.17</td></tr><tr><td>0.55</td><td>11.76</td><td>86.98</td><td>60.15</td><td>33.38</td><td>12.05</td><td>87.12</td><td>60.12</td><td>33.01</td></tr><tr><td>0.75</td><td>11.63</td><td>86.46</td><td>60.04</td><td>33.57</td><td>12.05</td><td>86.96</td><td>59.98</td><td>32.94</td></tr><tr><td>0.95</td><td>11.57</td><td>86.92</td><td>60.36</td><td>33.06</td><td>12.05</td><td>86.92</td><td>59.94</td><td>32.93</td></tr><tr><td>MAE ↓</td><td>0.30</td><td>0.19</td><td>0.25</td><td>0.30</td><td>0.57</td><td>0.40</td><td>2.32</td><td>1.43</td></tr></table>

![](images/488109c8d5ac0d40fde5f06dcb404c3fc38de8868664dc270d07e83737ae3429.jpg)  
(a) LDA topic modeling

![](images/7f8b7d09e232fc6629318682d83351fd4f49fd40b3afca88d3743e7c6851e5e9.jpg)  
Figure 4: (a) The topic-word distributions recovered from each method after 300-epoch training. (b) Segmentation of Poisson time series inferred from the backward distribution  $\phi(X_{1:T})$ .  
(b) Poisson time-series segmentation

# 4.3 LEARNING DISCRETE REPRESENTATIONS

Many types of data exist in discrete symbols e.g., words in texts, or pixels in images. This motivates the need to explore the latent discrete representations of the data, which can be useful for planning and symbolic reasoning tasks. Viewing discrete representation learning as a parameter learning problem, we endow it with a probabilistic generative process as illustrated in Figure 3c. The problem deals with a latent space  $\mathcal{C} \in \mathbb{R}^{K \times D}$  composed of  $K$  discrete latent sub-spaces of  $D$  dimensionality. The probability a data point belongs to a discrete sub-space  $c \in \{1, \dots, K\}$  follows a  $K$ -way categorical distribution  $\pi = [\pi_1, \dots, \pi_K]$ . In the language of VQ-VAE, each  $c$  is referred to as a codeword and the set of codewords is called a codebook. Let  $Z \in \mathbb{R}^D$  denote the latent variable in a sub-space. On each sub-space, we impose a Gaussian distribution parameterized by  $\mu_c, \Sigma_c$  where  $\Sigma_c$  is diagonal. The data generative process is described as follows:

1. Sample  $c\sim \mathrm{Cat}(\pi)$  and  $Z\sim \mathcal{N}(\mu_c,\Sigma_c)$  
2. Quantize  $\mu_c = Q(Z)$  
3. Generate  $X = \psi_{\theta}(Z,\mu_c)$

where  $\psi$  is a highly non-convex function with unknown parameters  $\theta$  and often parameterized with a deep neural network.  $Q$  refers to the quantization of  $Z$  to  $\mu_c$  defined as  $\mu_c = Q(Z)$  where  $c = \operatorname{argmin}_c d_z(Z; \mu_c)$  and  $d_z = \sqrt{(Z - \mu_c)^T \Sigma_c^{-1}(Z - \mu_c)}$  is the Mahalanobis distance.

The goal is to learn the parameter set  $\{\pi, \mu, \Sigma, \theta\}$  with  $\mu = [\mu_k]_{k=1}^K$ ,  $\Sigma = [\Sigma_k]_{k=1}^K$  such that the learned representation captures the key properties of the data. Following VQ-VAE, our practical implementation considers  $Z$  as an  $M$ -component latent embedding. We experiment with images in this application and compare OTP-DAG with VQ-VAE on CIFAR10, MNIST, SVHN and CELEBA datasets. Since the true parameters are unknown, we assess how well the latent space characterizes the input data through the quality

of the reconstruction of the original images. Table 3 reports our superior performance in preserving high-quality information of the input images. VQ-VAE suffers from poorer performance mainly due to codebook collapse (Yu et al., 2021) where most of latent vectors are quantized to limited discrete codewords. Meanwhile, our framework allows for control over the number of latent representations, ensuring all codewords are utilized. In Appendix E.3, we detail the formulation of our method and provide qualitative examples. We also showcase therein our competitive performance against a recent advance called SQ-VAE (Takida et al., 2022) without introducing any additional complexity.

Table 3: Quality of the image reconstructions  $\left( {K = {512}}\right)$  .  

<table><tr><td>Dataset</td><td>Method</td><td>Latent Size</td><td>SSIM ↑</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>rFID ↓</td><td>Perplexity ↑</td></tr><tr><td rowspan="2">CIFAR10</td><td>VQ-VAE</td><td>8 × 8</td><td>0.70</td><td>23.14</td><td>0.35</td><td>77.3</td><td>69.8</td></tr><tr><td>OTP-DAG (Ours)</td><td>8 × 8</td><td>0.80</td><td>25.40</td><td>0.23</td><td>56.5</td><td>498.6</td></tr><tr><td rowspan="2">MNIST</td><td>VQ-VAE</td><td>8 × 8</td><td>0.98</td><td>33.37</td><td>0.02</td><td>4.8</td><td>47.2</td></tr><tr><td>OTP-DAG (Ours)</td><td>8 × 8</td><td>0.98</td><td>33.62</td><td>0.01</td><td>3.3</td><td>474.6</td></tr><tr><td rowspan="2">SVHN</td><td>VQ-VAE</td><td>8 × 8</td><td>0.88</td><td>26.94</td><td>0.17</td><td>38.5</td><td>114.6</td></tr><tr><td>OTP-DAG (Ours)</td><td>8 × 8</td><td>0.94</td><td>32.56</td><td>0.08</td><td>25.2</td><td>462.8</td></tr><tr><td rowspan="2">CELEBA</td><td>VQ-VAE</td><td>16 × 16</td><td>0.82</td><td>27.48</td><td>0.19</td><td>19.4</td><td>48.9</td></tr><tr><td>OTP-DAG (Ours)</td><td>16 × 16</td><td>0.88</td><td>29.77</td><td>0.11</td><td>13.1</td><td>487.5</td></tr></table>

# 5 DISCUSSION AND CONCLUSION

The key message across our experiments is that OTP-DAG is a scalable and versatile framework readily applicable to learning any directed graphs with latent variables. OTP-DAG is consistently shown to perform comparably and in some cases better than MAP, EM and SVI which are well-known for yielding reliable estimates. Similar to amortized VI, on one hand, our method employs amortized optimization and assumes one can sample from the priors or more generally, the model margins over latent parents. OTP-DAG requires continuous relaxation through reparameterization of the underlying model distribution to ensure the gradients can be back-propagated effectively. The specification is also not unique to OTP-DAG: VAE also relies on reparameterization trick to compute the gradients w.r.t the variational parameters. For discrete distributions and for some continuous ones (e.g., Gamma distribution), this is not easy to attain. To this end, a proposal on Generalized Reparameterization Gradient (Ruiz et al., 2016) is a viable solution. On the other hand, different from VI, our global OT cost minimization is achieved by characterizing local densities through backward maps from the observed nodes to their parents. This localization strategy makes it easier to find a good approximation compared to VI, where the variational distribution is defined over all hidden variables and should ideally characterize the entire global dependencies in the graph. A popular method called Semi-amortized VAE (SA-VAE, Kim et al., 2018) is proposed to tackle this sub-optimality issue of the inference network in VI. In Appendix D, we compare OTP-DAG with this model on parameter estimation task, where ours competes on par with SA-VAE under the usual OTP-DAG learning procedure that comes with no extra overhead. To model the backward distributions, we utilize the expressivity of deep neural networks. Based on the universal approximation theorem (Hornik et al., 1989), the gap between the model distribution and the true conditional can be assumed to be smaller than an arbitrary constant  $\epsilon$  given enough data, network complexity, and training time.

Future Research. The proposed algorithm lays the cornerstone for an exciting paradigm shift in the realm of graphical learning and inference. Looking ahead, this fresh perspective unlocks a wealth of promising avenues for future application of OTP-DAG to large-scale inference problems or other learning tasks such as for undirected graphical models, or structural learning where edge existence and directionality can be parameterized as part of the model parameters.

# REFERENCES

Jonas Adler and Sebastian Lunz. Banach wasserstein gan. Advances in neural information processing systems, 31, 2018. 2  
Nikolaos Aletras and Mark Stevenson. Evaluating topic coherence using distributional semantics. In Proceedings of the 10th international conference on computational semantics (IWCS 2013)-Long Papers, pp. 13-22, 2013. 7, 22  
Luca Ambrogioni, Umut Güçlü, Yagmur Güçlütürk, Max Hinne, Marcel A.J. Van Gerven, and Eric Maris. Wasserstein variational inference. Advances in Neural Information Processing Systems, 2018-December (NeurIPS):2473-2482, 2018. ISSN 10495258. 2  
Luca Ambrogioni, Kate Lin, Emily Fertig, Sharad Vikram, Max Hinne, Dave Moore, and Marcel van Gerven. Automatic structured variational inference. In Arindam Banerjee and Kenji Fukumizu (eds.), Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pp. 676-684. PMLR, 13-15 Apr 2021. URL https://proceedings.mlr.press/v130/ambrogioni21a.html.1,17  
Brandon Amos. Tutorial on amortized optimization for learning to optimize over continuous domains. arXiv preprint arXiv:2202.00665, 2022. 1, 5  
Animashree Anandkumar, Daniel Hsu, and Sham M Kakade. A method of moments for mixture models and hidden markov models. In Conference on Learning Theory, pp. 33-1. JMLR Workshop and Conference Proceedings, 2012. 17  
Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. Journal of machine learning research, 15:2773-2832, 2014. 17  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International conference on machine learning, pp. 214-223. PMLR, 2017. 2  
Matthew J Beal and Zoubin Ghahramani. Variational bayesian learning of directed graphical models with hidden variables. 2006. 17  
Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006. 17  
David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan):993-1022, 2003. 5, 6, 21  
Olivier Cappé and Eric Moulines. On-line expectation-maximization algorithm for latent data models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 71(3):593-613, 2009. 17  
Bin Dai, Ziyu Wang, and David Wipf. The usual suspects? reassessing blame for vae posterior collapse. In International conference on machine learning, pp. 2313-2322. PMLR, 2020. 20  
Bernard Delyon, Marc Lavielle, and Eric Moulines. Convergence of a stochastic approximation version of the em algorithm. Annals of statistics, pp. 94-128, 1999. 17  
A. P. Dempster, N. M. Laird, and D. B. Rubin. Maximum Likelihood from Incomplete Data Via the EM Algorithm. Journal of the Royal Statistical Society: Series B (Methodological), 39(1):1-22, 1977. doi: 10.1111/j.2517-6161.1977.tb01600.x. 1

Nick Foti, Jason Xu, Dillon Laird, and Emily Fox. Stochastic variational inference for hidden markov models. Advances in neural information processing systems, 27, 2014. 17  
Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, et al. Deep end-to-end causal inference. arXiv preprint arXiv:2202.02195, 2022. 17  
Alan E Gelfand and Adrian FM Smith. Sampling-based approaches to calculating marginal densities. Journal of the American statistical association, 85(410):398-409, 1990. 17  
Walter R Gilks, Sylvia Richardson, and David Spiegelhalter. Markov chain Monte Carlo in practice. CRC press, 1995. 17  
Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pp. 1440-1448, 2015. 25  
Thomas L Griffiths and Mark Steyvers. Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl_1):5228-5235, 2004. 6  
John Hammersley. Monte carlo methods. Springer Science & Business Media, 2013. 17  
Ernst Hellinger. Neue begründung der theorie quadratischer formen von unendlichvielen veränderlichen. Journal für die reine und angewandte Mathematik, 1909(136):210-271, 1909. 6  
James Hensman, Magnus Rattray, and Neil Lawrence. Fast variational inference in the conjugate exponential family. Advances in neural information processing systems, 25, 2012. 17  
Jose Hernandez-Lobato, Yingzhen Li, Mark Rowland, Thang Bui, Daniel Hernandez-Lobato, and Richard Turner. Black-box alpha divergence minimization. In International conference on machine learning, pp. 1511-1520. PMLR, 2016. 17  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 27  
Matthew D Hoffman and David M Blei. Structured stochastic variational inference. In Artificial Intelligence and Statistics, pp. 361-369, 2015. 17  
Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic variational inference. Journal of Machine Learning Research, 2013. 1, 17  
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989. 9  
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016. 21  
Matthew Johnson and Alan Willsky. Stochastic variational inference for bayesian time series models. In International Conference on Machine Learning, pp. 1854-1862. PMLR, 2014. 17  
Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. Machine learning, 37:183-233, 1999. 1, 3  
Leonid V Kantorovich. Mathematical methods of organizing and planning production. Management science, 6(4):366-422, 1960. 2

Yoon Kim, Sam Wiseman, Andrew Miller, David Sontag, and Alexander Rush. Semi-amortized variational autoencoders. In International Conference on Machine Learning, pp. 2678-2687. PMLR, 2018. 9, 20  
Nathaniel J King and Neil D Lawrence. Fast variational inference for gaussian process models through k-cl-correction. In Machine Learning: ECML 2006: 17th European Conference on Machine Learning Berlin, Germany, September 18-22, 2006 Proceedings 17, pp. 270-281. Springer, 2006. 17  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2  
Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems, 29, 2016. 17  
Miguel Lázaro-Gredilla, Steven Van Vaerenbergh, and Neil D Lawrence. Overlapping mixtures of gaussian processes for the data association problem. Pattern recognition, 45(4):1386-1395, 2012. 17  
Yingzhen Li and Richard E Turner. Rényi divergence variational inference. Advances in neural information processing systems, 29, 2016. 17  
Percy Liang and Dan Klein. Online em for unsupervised models. In Proceedings of human language technologies: The 2009 annual conference of the North American chapter of the association for computational linguistics, pp. 611-619, 2009. 17  
David JC MacKay. Choice of basis for laplace approximation. Machine learning, 33:77-86, 1998. 21  
Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016. 21  
Dmitry Molchanov, Valery Kharitonov, Artem Sobolev, and Dmitry Vetrov. Doubly semi-implicit variational inference. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2593–2602. PMLR, 2019. 17  
Todd K Moon. The expectation-maximization algorithm. IEEE Signal processing magazine, 13(6):47-60, 1996. 3  
Kevin P Murphy. Probabilistic machine learning: Advanced topics. MIT Press, 2023. 7  
Radford M Neal and Geoffrey E Hinton. A view of the em algorithm that justifies incremental, sparse, and other variants. Learning in graphical models, pp. 355-368, 1998. 17  
Ronald C Neath et al. On convergence properties of the monte carlo em algorithm. Advances in modern statistical theory and applications: a Festschrift in Honor of Morris L. Eaton, pp. 43-62, 2013. 17  
George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22:1-64, 2021. ISSN 15337928. 1  
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pp. 4055-4064. PMLR, 2018. 27  
Gabriel Peyre, Marco Cuturi, et al. Computational optimal transport. Center for Research in Economics and Statistics Working Papers, (2017-86), 2017. 2

Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational inference. In Artificial intelligence and statistics, pp. 814-822. PMLR, 2014. 1, 17  
Rajesh Ranganath, Dustin Tran, and David Blei. Hierarchical variational models. In International conference on machine learning, pp. 324-333. PMLR, 2016. 1, 17  
Francisco R Ruiz, Titsias RC AUEB, David Blei, et al. The generalized reparameterization gradient. Advances in neural information processing systems, 29, 2016. 9  
Filippo Santambrogio. Optimal transport for applied mathematicians. Birkhäuser, NY, 55(58-63):94, 2015. 15  
Lawrence Saul and Michael Jordan. Exploiting tractable substructures in intractable networks. Advances in neural information processing systems, 8, 1995. 17  
Akash Srivastava and Charles Sutton. Autoencoding variational inference for topic models. arXiv preprint arXiv:1703.01488, 2017. 6, 21, 22  
Yuhta Takida, Takashi Shibuya, Weihsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational bayes on discrete representation with self-annealed stochastic quantization. In International Conference on Machine Learning, pp. 20987-21012. PMLR, 2022. 9, 27  
Yee Teh, David Newman, and Max Welling. A collapsed variational bayesian inference algorithm for latent dirichlet allocation. Advances in neural information processing systems, 19, 2006. 17  
Michalis K Titsias and Francisco Ruiz. Unbiased implicit variational inference. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 167-176. PMLR, 2019. 17  
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein auto-encoders. arXiv preprint arXiv:1711.01558, 2017. 2, 18, 19  
Dustin Tran, David Blei, and Edo M Airoldi. Copula variational inference. Advances in neural information processing systems, 28, 2015. 17  
Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 5, 25, 26, 27  
Cédric Villani. Topics in optimal transportation, volume 58. AMS Graduate Studies in Mathematics, 2003. 3  
Cédric Villani et al. Optimal transport: old and new, volume 338. Springer, 2009. 2  
Neng Wan, Dapeng Li, and Naira Hovakimyan. F-divergence variational inference. Advances in neural information processing systems, 33:17370-17379, 2020. 17  
Greg CG Wei and Martin A Tanner. A monte carlo implementation of the em algorithm and the poor man's data augmentation algorithms. Journal of the American statistical Association, 85(411):699-704, 1990. 17  
Ming Xu, Matias Quiroz, Robert Kohn, and Scott A Sisson. Variance reduction properties of the repa-rameterization trick. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2711-2720. PMLR, 2019. 17  
Mingzhang Yin and Mingyuan Zhou. Semi-implicit variational inference. In International Conference on Machine Learning, pp. 5660-5669. PMLR, 2018. 17

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.9  
Yue Yu, Jie Chen, Tian Gao, and Mo Yu. Dag-gnn: Dag structure learning with graph neural networks. In International Conference on Machine Learning, pp. 7154-7163. PMLR, 2019. 17  
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586-595, 2018. 27
