# Network Lasso Bandits

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We consider a multi-task contextual bandit setting, where the learner is given a graph encoding relations between the bandit tasks. The tasks' preference vectors are assumed to be piecewise constant over the graph, forming clusters. At every round, we estimate the preference vectors by solving an online network lasso problem with a suitably chosen, time-dependent regularization parameter. We establish a novel oracle inequality relying on a convenient restricted eigenvalue assumption. Our theoretical findings highlight the importance of dense intra-cluster connections and sparse inter-cluster ones. That results in a sublinear regret bound significantly lower than its counterpart in the independent task learning setting. Finally, we support our theoretical findings by experimental evaluation against graph bandit multi-task learning and online clustering of bandits algorithms.

# 1 Introduction

Online commercial websites aim to properly recommend their products to their customers, and the performance of these recommendations depends on the knowledge of users' preferences. Unlike traditional collaborative-filtering-based methods [Su and Khoshgoftaar, 2009], such knowledge is initially unavailable. Therefore, the online recommender systems need to recommend various items to the users and observe their ratings to explore their preferences. At the same time, the recommender system should be able to recommend items that attract users' attention and receive high ratings by exploiting the learned knowledge. The contextual bandits frameworks [Li et al., 2010] have been popularly used to formalize and address this exploration-exploitation trade-off.

However, the classical form of contextual bandits [Li et al., 2010, Chu et al., 2011, Abbasi-Yadkori et al., 2011] ignores the availability of social networks amongst users and solves the problem for each user separately. Consequently, such algorithms have some drawbacks when applied to problems with a large number of users. First, such a large number hinders the computational efficiency of such algorithms. Second, the partial feedback of the bandit settings exposes the algorithms to have weak estimations and impair their decision-making ability [Yang et al., 2020]. Consequently, to improve bandit algorithms' performance for large-scale applications, structural assumptions that link the different users are usually integrated within bandit algorithms [Cesa-Bianchi et al., 2013, Gentile et al., 2014, Li et al., 2019, Herbster et al., 2021].

The papers of Cesa-Bianchi et al. [2013], Yang et al. [2020] attempt to integrate the prior knowledge of social networks into their contextual bandit algorithms. Both papers proposed UCB-style algorithms and exhibited the importance of using the social network graph to achieve lower regrets using Laplacian regularization. Consequently, both methods promote smoothness among the preference vectors of users in order to transfer the collected information between them. However, the Laplacian regularization does not account for the smoothness heterogeneity introduced by a piecewise constant behavior over the graph [Wang et al., 2016]. On the other hand, algorithms of online clustering of bandits [Gentile et al., 2014, Li et al., 2019] start from a graph and gradually add or remove edges to

form clusters as connected components. However, their clustering can cause overconfidence in the constructed clusters, potentially leading to error accumulation.

In this paper, we assume access to a graph encoding relations between bandit tasks, and that the task parameter vectors are piecewise constant over the graph. That means that tasks form clusters. We propose an algorithm that integrates the prior knowledge of the piecewise constant structure to update tasks rather than finding the clusters explicitly. That way, we mitigate the limitations mentioned above: the piecewise constant smoothness is naturally integrated into our regularizer, and we do not estimate the clusters so our algorithm does not suffer from overconfidence drawbacks.

More precisely, we provide the following contributions

- We analyze an instance of the Network Lasso problem [Hallac et al., 2015], where every vertex's preference vector is estimated using data generated during the interaction between users and the bandit. We provide the first oracle inequality in this setting and link it to fundamental quantities characterizing the relation between the graph and the true preference vectors of the users. Our result relies on our novel restricted eigenvalue (RE) condition, which we assume for our setting. This result is of independent interest and can be applied to independently generated data as a special case.  
- We prove how the empirical multi-task Gram matrix of the data inherits the RE condition from its true counterpart. Both this result and the previous one depend on the sparsity of inter-cluster connections and the density of intra-cluster ones.  
- We provide a regret upper bound for our setting. Our bound highlights the advantage of our algorithm in high dimensional settings, and for large graphs.  
- We support our theoretical findings by extensive numerical experiments on simulated data that prove the advantage of our algorithm compared to other approaches used for online clustering of bandits.

The rest of the paper is organized as follows. Section 2 discusses the relation of our work to the literature. We formulate our problem and state some of our assumptions in Section 3, then we present our bandit algorithm in Section 4. We analyze the problem theoretically in Section 5, and finally, we demonstrate its practical interest via numerical experiments in Section 6.

# 2 Related work

Lasso contextual bandits To address the high dimensional setting for linear bandits, several multi-armed bandit papers solve a LASSO [Tibshirani, 1996] problem under different assumptions [Bastani and Bayati, 2019, Kim and Paik, 2019, Oh et al., 2021, Ariu et al., 2022]. They all rely on a previously established compatibility or RE condition [Buhlmann and van de Geer, 2011], that they adapt to the non-i.i.d case. Such assumptions were also used in the multi-task setting by Cella and Pontil [2021] with a Group Lasso regularization [Yuan and Lin, 2006], and to impose a low rank structure on the task preference vectors in Cella et al. [2023]. In our case, we provide a novel oracle inequality, rather than just generalize an existing one to the non-i.i.d setting, with a newly introduced RE assumption.

**Clustering of bandits** Sequentially clustering bandit tasks was introduced in Gentile et al. [2014] with CLUB algorithm. In CLUB, starting with a fully connected graph, an iterative graph learning process is performed, where edges between users are deleted if their preference vectors are significantly different. As a result, any connected component is seen as a cluster and only one recommendation per cluster is developed. In another work, Li et al. [2019] generalize the setting of Gentile et al. [2014] and address its limitations via including merging operations in addition to splitting. In contrast to these approaches, the algorithm in Nguyen and Lauw [2014] groups users via K-means clustering, and the algorithm in Cheng et al. [2023] relies on hedonic games for online clustering of bandits. Furthermore, Yang and Toni [2018] make use of community detection techniques on graphs to find user clusters. Gentile et al. [2017] study the clustering of the contextual bandit problem where their proposed algorithm, named CAB, adaptively matches user preferences in the face of constantly evolving items. Our work fundamentally differs from the previous ones on two aspects. First, we assume access to a graph encoding relations between users, which is more informative than a complete graph. Second, we do not keep track of a model for each cluster, but rather we integrate a prior over

the graph via a graph total variation regularizer that enforces a piecewise constant behaviour for the estimated preference vectors.

Multi-task learning Several contributions assume some underlying structure that links the bandit tasks. In Cella and Pontil [2021], task preference vectors are assumed to be sparse and to share their sparsity support, implying that they lie in a low-dimensional subspace with dimensions aligning with the canonical basis vectors. This idea is further generalized in Cella et al. [2023], where the tasks are assumed to be confined to an arbitrary unknown low-dimensional subspace. That work improves upon Hu et al. [2021] by not requiring the knowledge of the small dimension of the task space. The underlying structure linking tasks can also be a graph encoding relations between them [Cesa-Bianchi et al., 2013, Yang and Toni, 2018], which is our case. However, while they assume smoothness as a prior, we assume piecewise constant behavior.

# 3 Problem setting

We consider a linear bandit setting, with a finite number of tasks representing users in a recommendation system for example. For each task the agent has to choose among  $K$  arms, each associated to a  $d$ -dimensional context vector. All interactions over a horizon of  $T$  time steps. We further assume that we have access to an undirected graph  $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ , with vertex set  $\mathcal{V}$  representing the tasks and edge set  $\mathcal{E}$  encoding the relationships between them. We identify the vertex set  $\mathcal{V}$  with the set of vertex indices  $[|\mathcal{V}|]$ . Thus, we consider  $\mathcal{E}$  to be a subset of  $\mathcal{V}^2$ , where every edge  $(m,n) \in \mathcal{E}$  has weight  $w_{mn} > 0$ , with  $m < n$ . The tasks' preference vectors are denoted by  $\{\pmb{\theta}_m\}_{m \in \mathcal{V}} \subset \mathbb{R}^d$  verifying  $\| \pmb{\theta}_m \| \leq 1 \forall m \in \mathcal{V}$ , which we concatenate as row vectors into matrix  $\Theta \in \mathbb{R}^{|\mathcal{V}| \times d}$ . The latter represents a graph vector signal, assumed to be piecewise constant over  $\mathcal{G}$ .

At a round  $t \in \mathbb{N}^{\star}$ , a user  $m(t) \in \mathcal{V}$  is selected uniformly at random and served an arm with context vector  $\mathbf{x}(t)$  from a finite action set  $\mathcal{A}(t) \subset \mathbb{R}^{d}$  with size  $K$ , depending on their estimated preference vector  $\hat{\theta}_{m(t)}(t) \in \mathbb{R}^{d}$ . We assume the expected reward to be linear, with an additive,  $\sigma$ -sub-Gaussian noise conditionally on the past. Formally, denoting by  $\mathcal{F}_0$  the trivial sigma-algebra, and for all  $t \geq 1$  by  $\mathcal{F}_t$  the sigma-algebra generated by history set  $\{m(1), \mathbf{x}(1), y(1), \dots, m(t), \mathbf{x}(t), y(t), m(t + 1)\}$ , the received reward  $y(t)$  is given by  $y(t) = \left\langle \pmb{\theta}_{m(t)}(t), \mathbf{x}(t) \right\rangle + \eta(t)$ , where  $\eta(t)$  is  $\mathcal{F}_t$ -measurable and

$$
\mathbb {E} [ \eta (t) | \mathcal {F} _ {t - 1} ] = 0, \quad \mathbb {E} [ \exp (s \eta (t)) | \mathcal {F} _ {t - 1} ] \leq \exp \left(\frac {1}{2} \sigma^ {2} s ^ {2}\right) \quad \forall t \geq 1, \forall s \in \mathbb {R}. \tag {1}
$$

At the end of a round  $t$ , all preference vectors are updated into a new estimation  $\hat{\Theta}(t)$  while leveraging the structure of graph  $\mathcal{G}$ , formally by solving the following optimization problem:

$$
\hat {\boldsymbol {\Theta}} (t) = \underset {\tilde {\boldsymbol {\Theta}} \in \mathbb {R} ^ {| \mathcal {V} | \times d}} {\arg \min } \frac {1}{2 t} \sum_ {\tau = 1} ^ {t} \left(\left\langle \tilde {\boldsymbol {\theta}} _ {m (\tau)}, \mathbf {x} (\tau) \right\rangle - y (\tau)\right) ^ {2} + \alpha (t) \sum_ {(m, n) \in \mathcal {E}} w _ {m n} \left\| \tilde {\boldsymbol {\theta}} _ {m} - \tilde {\boldsymbol {\theta}} _ {n} \right\|, \tag {2}
$$

where  $\|\cdot\|$  denotes the Euclidean norm for vectors. The performance of our policy is assessed by the expected regret over the  $T$  interaction rounds for all tasks:

$$
\mathcal {R} (T) = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \left\langle \boldsymbol {\theta} _ {m (t)}, \mathbf {x} ^ {\star} (t) - \mathbf {x} (t) \right\rangle \right], \tag {3}
$$

where  $\mathbf{x}^{\star}(t)\in \arg \max_{\tilde{\mathbf{x}}\in \mathcal{A}(t)}\left\langle \pmb{\theta}_{m(t)},\tilde{\mathbf{x}}\right\rangle$

The Optimization problem in (2) is an instance of the Network Lasso [Hallac et al., 2015]. Other instances of the same type were studied by Jung et al. [2018], Jung and Vesselinova [2019], Jung [2020]. The objective is characterized by its second term that, while being just the Laplacian regularization without squaring the norms, promotes a piecewise constant behavior rather than smoothness. For real-valued signals  $(d = 1)$ , this regularization has been extensively studied for image and graph signal denoising, for the problem of trend filtering on graphs [Wang et al., 2016]. According to Wang et al. [2016], that regularization better adapts to the heterogeneity of smoothness of the signal and induces a cluster structure in the data: similar users will not only have similar models but the same model, which offers a compression of the overall model over the graph. Note

that our setting is cluster agnostic; our algorithm does not aim to learn the cluster structure explicitly but to exploit it implicitly using the total variation semi-norm as regularization. The latter's strength is controlled via a time-dependent regularization coefficient  $\alpha(t)$ , which we will express later in the analysis.

We formalize our assumption on the context generation as follows.

Assumption 1 (i.i.d action sets). Context sets  $\{\mathcal{A}(t)\}_{t=1}^{T}$  are generated i.i.d. from a distribution  $p$  over  $\mathbb{R}^{K \times d}$ , such that  $\| \mathbf{x} \| \leq 1 \forall \mathbf{x} \in \mathcal{A}(t) \forall t \geq 1$ .

In addition to the i.i.d assumption, we assume more regularity.

Assumption 2 (Relaxed symmetry and balanced covariance). There exists a constant  $\nu \geq 1$  such that for all  $\mathbf{X} \in \mathbb{R}^{K \times d}$ ,  $p(-\mathbf{X}) \leq \nu p(\mathbf{X})$ . Furthermore, there exists  $\omega > 0$ , such that for any permutation  $(a_1, \dots, a_K)$  of  $[K]$ , for any  $i \in \{2, \dots, K-1\}$ , and for any  $\mathbf{w} \in \mathbb{R}^d$ , we have

$$
\mathbb {E} \left[ \mathbf {x} _ {a _ {i}} \mathbf {x} _ {a _ {i}} ^ {\top} \left[ \mathbf {w} ^ {\top} \mathbf {x} _ {a _ {1}} <   \dots <   \mathbf {w} ^ {\top} \mathbf {x} _ {a _ {K}} \right] \right] \preccurlyeq \omega \mathbb {E} \left[ \left(\mathbf {x} _ {a _ {1}} \mathbf {x} _ {a _ {1}} ^ {\top} + \mathbf {x} _ {a _ {K}} \mathbf {x} _ {a _ {K}} ^ {\top}\right) \left[ \mathbf {w} ^ {\top} \mathbf {x} _ {a _ {1}} <   \dots <   \mathbf {w} ^ {\top} \mathbf {x} _ {a _ {K}} \right] \right],
$$

where  $\mathbf{M} \precsim \mathbf{N}$  means that  $\mathbf{N} - \mathbf{M}$  is a PSD matrix.

This assumption was introduced in Oh et al. [2021], and has already been used in a multi-task setting by Cella et al. [2023]. Parameter  $\nu$  controls the skewness, as  $\nu = 1$  corresponds to a symmetric distribution.  $\omega$  decreases with increasing positive correlation between arms. It verifies  $\omega = O(1)$  for multi-variate Gaussians and uniform distributions over the unit sphere [Oh et al., 2021]. The piecewise constant behaviour of the graph signal  $\Theta$  is formalized in the next assumption.

Assumption 3 (Piecewise constant signal). There exists a partition  $\mathcal{P}$  of  $\mathcal{V}$ , such that for any cluster  $\mathcal{C} \in \mathcal{P}$ , signal  $\Theta$  is constant on  $\mathcal{C}$ , and the graph obtained by taking the vertices in  $\mathcal{C}$  and the edges linking them is connected.

Assumption 3 basically states that the true preference vectors are clustered and that the given graph induces the cluster structure. It is required for our approach to be beneficial, as we will detail in the analysis section. For the sake of clarity, we defer the statement of other technical assumptions to Section 5.

# 4 Algorithm

Our policy in Algorithm 1 follows a greedy arm selection rule in a multi-task setting, in the same vein as those presented in Oh et al. [2021], Cella et al. [2023]. Indeed, as pointed out in Oh et al. [2021], exploration is implicitly incorporated into regularization parameter  $\alpha(t)$ 's time dependence. It has the following expression

$$
\alpha (t) := \frac {\alpha_ {0} \sigma}{t} \sqrt {t + \sqrt {2 \sum_ {m \in \mathcal {V}} \left| \mathcal {T} _ {m} (t) \right| ^ {2} \log \frac {1}{\delta (t)}} + 2 \max  _ {m \in \mathcal {V}} \left| \mathcal {T} _ {m} (t) \right| \log \frac {1}{\delta (t)}}, \tag {4}
$$

where the set of time steps a task  $m$  has been selected up to time  $t$  is denoted by  $\mathcal{T}_m(t)$ .

# 5 Analysis

This section provides the main steps of the analysis. One of the paper's contribution lies in finding an oracle inequality of the network lasso problem given a restricted eigenvalue condition holding for the true multi-task Gram matrix. In this regard, the next major challenge and contribution is to show that the empirical multi-task Gram matrix, estimated in the algorithm, satisfies the restricted eigenvalue condition. We start by proving an oracle inequality for the estimation error of  $\Theta$ , assuming that the condition given by Definition 2 is verified by the empirical data Gram matrix. Then, we prove that the latter assumption actually holds with high probability given that true multi-task Gram matrix satisfies it. Our final contribution in this work is the establishment of a regret bound for our algorithm.

# 5.1 Notation and technical assumptions

We provide additional notations required for the analysis. We denote by  $\partial \mathcal{P}$  the set of all edges in  $\mathcal{E}$  connecting vertices from different clusters from partition  $\mathcal{P}$  (Assumption 3), and we call it the

# Algorithm 1: Network Lasso Policy

Input :  $T,\alpha_0 > 0,\mathcal{G}$  ,function  $\delta$

Initialization :  $\hat{\Theta} (0) = \mathbf{0}\in \mathbb{R}^{|\mathcal{V}|\times d}$

for  $t\in [1,T]$  do

1. Draw a user  $m(t)\in \mathcal{V}$  uniformly at random.  
2. Observe context set  $\mathcal{A}(t)$ .  
3. Select  $\mathbf{x}(t) \in \arg \max_{\tilde{\mathbf{x}} \in \mathcal{A}(t)} \left\langle \hat{\boldsymbol{\theta}}_{m(t-1)}, \tilde{\mathbf{x}} \right\rangle$ , breaking ties arbitrarily.  
4. Receive payoff  $y(t)$  
5. Update  $\alpha(t)$  via Equation (4)  
6. Update  $\hat{\Theta}(t)$  via solving the network Lasso problem (2)

# end

boundary of  $\mathcal{P}$ . Thus,  $\partial \mathcal{P}^c$ , the complementary set of  $\partial \mathcal{P}$ , is formed by edges connecting vertices of the same cluster. The total weight of the boundary, i.e. the sum of its edges' weights, is referred to as  $w(\partial \mathcal{P})$ . Given a signal  $\mathbf{Z} \in \mathbb{R}^{|\mathcal{V}| \times d}$ , we denote by  $\mathbf{Z}_{\mathcal{P}}$  the signal obtained by setting row vectors of  $\mathbf{Z}$  to their mean-per-cluster value w.r.t.  $\mathcal{P}$ . For any edge subset  $I \in \mathcal{E}$ , we denote the following norms:  $\| \cdot \|_F$  as the Frobenius norm,  $\| \mathbf{z} \|_{\mathbf{M}} = \sqrt{\mathbf{z}^\top \mathbf{M} \mathbf{z}}$  as the weighted norm of vector  $\mathbf{z} \in \mathbb{R}^d$  induced by matrix  $\mathbf{M} \in \mathbb{R}^{d \times d}$  and  $\| \Theta \|_I := \sum_{(m,n) \in I} w_{mn} \| \theta_m - \theta_n \|$  as the total variation semi-norm of  $\Theta \in \mathbb{R}^{|\mathcal{V}| \times d}$  over  $I$ . Thus, the regularization term of Problem (2) is equal to  $\| \Theta \|_{\mathcal{E}}$ . Also, we define the incidence matrix  $\mathbf{B}_I \subset \mathbb{R}^{|\mathcal{E}| \times |\mathcal{V}|}$  restricted to  $I \subseteq \mathcal{E}$  to be null except at rows with index  $i \in I$  corresponding to edge  $(m,n)$ , where it equals  $w_{mn} (\mathbf{e}_m - \mathbf{e}_n)$ , where  $\mathbf{e}_m$  is the  $m^{\text{th}}$  canonical basis vector of  $\mathbb{R}^{|\mathcal{V}|}$ . We define  $\mathbf{A}_{\mathcal{V}}(t) := \mathrm{diag}\left(\mathbf{X}_1(t)^\top \mathbf{X}_1(t), \ldots, \mathbf{X}_{|\mathcal{V}|}(t)^\top \mathbf{X}_{|\mathcal{V}|}(t)\right) \in \mathbb{R}^{d|\mathcal{V}| \times d|\mathcal{V}|}$ , and subsequently the empirical multi-task Gram matrix up to time step  $t$  is given by  $\frac{1}{t} \mathbf{A}_{\mathcal{V}}(t)$ . The following definition introduces quantities related to the clusters defined by partition  $\mathcal{P}$ , with crucial roles that we will elucidate throughout the analysis.

Definition 1 (Cluster content constants). Let  $\mathcal{C} \in \mathcal{P}$  be a cluster.

- We denote by  $\partial_v\mathcal{C}$  the inner boundary of  $\mathcal{C}$ , i.e. the vertices of  $\mathcal{C}$  that are connected to its complementary. We define the inner isoperimetric ratio of  $\mathcal{C}$  as  $\iota_{\mathcal{G}}(\mathcal{C}) \coloneqq \frac{|\partial_v\mathcal{C}|}{|\mathcal{C}|}$ .  
- By abuse of notation, we denote as  $\mathbf{B}_{\mathcal{C}}$  the incidence matrix restricted to edges linking vertices of  $\mathcal{C}$ , its associated Laplacian matrix by  $\mathbf{L}_{\mathcal{C}} \coloneqq \mathbf{B}_{\mathcal{C}}^{\top} \mathbf{B}_{\mathcal{C}}$ , and its pseudo-inverse by  $\mathbf{L}_{\mathcal{C}}^{\dagger}$ . The topological centrality index of node  $m \in \mathcal{C}$  w.r.t  $\mathcal{C}$  is equal to  $(\mathbf{L}_{\mathcal{C}}^{\dagger})_{mm}^{-1}$ . We define the topological centrality index of  $\mathcal{C}$  by  $c_{\mathcal{G}}(\mathcal{C}) \coloneqq \min_{m \in \mathcal{C}} (\mathbf{L}_{\mathcal{C}}^{\dagger})_{mm}^{-1}$ .

The inner isoperimetric ratio of a cluster measures how many "interior" nodes a cluster contains, in the sense that they are not connected to its complementary. It is at most equal to the isoperimetric ratio for weightless graphs as the size of the inner boundary is at most equal to that of the edge boundary, the latter being connected to the algebraic connectivity via the Cheeger inequality [Cheeger, 1970].

The topological centrality index measures the overall connectedness of a vertex in a network and indicates how robust a node is to edge failures [Ranjan and Zhang, 2013]. Also, it can be tied to electricity spreading in a network according to Van Mieghem et al. [2017]. We refer the interested reader to the two previously mentioned works for a detailed account of the properties of the topological centrality index. In the appendix, we show that for binary weights graphs the minimum topological centrality index is at least equal to the algebraic connectivity theoretically and experimentally, where we showcase that the difference between the two can be significant.

To proceed, we will need the following definition that introduces several notations to reduce the clutter.

Definition 2 (Restricted Eigenvalue (RE) condition and norm). Let  $\{\mathbf{M}_i\}_{i=1}^{|\mathcal{V}|} \subset \mathbb{R}^{d \times d}$  be a set of positive semi-definite matrices. We say that the matrix  $\mathbf{M}_{\mathcal{V}} := \mathrm{diag}(\mathbf{M}_1, \dots, \mathbf{M}_{|\mathcal{V}|})$  verifies the restricted eigenvalue condition with constants  $\kappa \geq 0$  and  $\phi > 0$  if

$$
\phi^ {2} \| \mathbf {Z} \| _ {\mathrm {R E}} ^ {2} \leq \sum_ {i \in \mathcal {V}} \| \mathbf {z} _ {i} \| _ {\mathbf {M} _ {i}} ^ {2} \quad \forall \mathbf {Z} \in \mathcal {S} \text {w i t h r o w s} \left\{\mathbf {z} _ {i} \right\} _ {i \in \mathcal {V}},
$$

where  $S$  is the cone defined by:

$$
\mathcal {S} := \left\{\mathbf {Z} \in \mathbb {R} ^ {| \mathcal {V} | \times d}; a _ {1} (\mathcal {G}, \boldsymbol {\Theta}) \| \mathbf {Z} \| _ {\partial \mathcal {P} ^ {c}} \leq a _ {2} (\mathcal {G}, \boldsymbol {\Theta}) \| \overline {{\mathbf {Z}}} _ {\mathcal {P}} \| _ {F} + (1 - \kappa) ^ {+} \| \mathbf {Z} \| _ {\partial \mathcal {P}} \right\},
$$

$$
a _ {1} (\mathcal {G}, \boldsymbol {\Theta}) := 1 - \frac {\frac {1}{\alpha_ {0}} + 2 \kappa w (\partial \mathcal {P})}{\underset {\mathcal {C} \in \mathcal {P}} {\min } \sqrt {c _ {\mathcal {G}} (\mathcal {C})}}, \quad a _ {2} (\mathcal {G}, \boldsymbol {\Theta}) := \frac {1}{\alpha_ {0}} + \sqrt {2} \kappa w (\partial \mathcal {P}) \underset {\mathcal {C} \in \mathcal {P}} {\max } \sqrt {\iota_ {\mathcal {G}} (\mathcal {C})},
$$

and the  $RE$  semi-norm is defined by  $\| \mathbf{Z} \|_{\mathrm{RE}} \coloneqq \left\| \overline{\mathbf{Z}}_{\mathcal{P}} \right\|_F \vee (1 - \kappa)^+ \left\| \mathbf{B}_{\partial \mathcal{P}}^\dagger \mathbf{B}_{\partial \mathcal{P}} \mathbf{Z} \right\|$ .

To interpret the previous definition, we point out that the sum on the right-hand side of Definition 2 can be written as  $\left\| \operatorname{vec}(\mathbf{Z}^{\top}) \right\|_{\mathbf{M}_{\mathcal{V}}}$ , where  $\operatorname{vec}$  denotes the operation of stacking a matrix's columns vertically. As a result, the condition is analogous to requiring that  $\mathbf{M}_{\mathcal{V}}$  is invertible with minimum eigenvalue  $\phi^2$ , but weaker since it holds only for signals  $\mathbf{Z} \in S$  and for the  $\| \cdot \|_{\mathrm{RE}}$  norm. This requirement has the same form as the compatibility assumption for the Lasso [Buhlmann and van de Geer, 2011, Oh et al., 2021] or the restricted strong convexity assumption [Cella et al., 2023].

We further make the following assumption on the true multi-task Gram matrix:

Assumption 4 (RE condition for the true multi-task Gram matrix). For  $k \in [K]$ , let  $\boldsymbol{\Sigma}_k \coloneqq \mathbb{E}\left[\mathbf{x}_k\mathbf{x}_k^\top\right]$  be the Gram matrix of the  $k^{th}$  context vector's marginal distribution, let  $\boldsymbol{\Sigma}_{\mathcal{V}}$  be the true multi-task Gram matrix of the context vector generating distribution, given by

$$
\boldsymbol {\Sigma} _ {\mathcal {V}} := \mathbf {I} _ {| \mathcal {V} |} \otimes \overline {{\boldsymbol {\Sigma}}}, \quad \text {w h e r e} \quad \overline {{\boldsymbol {\Sigma}}} = \frac {1}{K} \sum_ {k = 1} ^ {K} \boldsymbol {\Sigma} _ {k}. \tag {5}
$$

We assume that  $\Sigma_{\mathcal{V}}$  verifies RE condition (Definition 2) with some problem dependent constants  $\kappa \in \left[0, \frac{1}{2w(\partial\mathcal{P})} \min_{\mathcal{C} \in \mathcal{P}} \sqrt{c_{\mathcal{G}}(\mathcal{C})}\right)$  and  $\phi > 0$ .

This assumption is common to make for Lasso-like bandit problems [Oh et al., 2021, Ariu et al., 2022, Cella et al., 2023]. We will later show that it can be transferred to empirical multi-task Gram matrix.

# 5.2 Oracle inequality

This section is dedicated to provide a bound on the estimation error of the Network Lasso problem given in Equation (2) at a particular step  $t$  of Algorithm 1. We assume fixed design, meaning that the context vectors are given and fixed, and we are not concerned by their randomness (due to the context generating distribution), nor by the randomness of their number for each user (due to random selection at each time step).

For a time step  $t$ , we deliver the oracle inequality controlling the deviation between the estimated preference vectors  $\hat{\Theta}(t)$  and the true ones  $\Theta$ . For the sake of simplicity, we provisionally assume that the RE condition holds for the empirical multi-task Gram matrix  $\mathbf{A}_{\mathcal{V}}(t)$ .

Theorem 1 (Oracle inequality). Assume that the  $RE$  assumption holds for the empirical multi-task Gram matrix with constants  $\kappa \in \left[0, \frac{1}{2w(\partial\mathcal{P})} \min_{\mathcal{C} \in \mathcal{P}} \sqrt{c_{\mathcal{G}}(\mathcal{C})}\right)$  and  $\phi > 0$ . Suppose that  $\max_{m \in \mathcal{V}} |\mathcal{T}_m(t)| \leq bt$  for some  $b > 0$ . Then, with a probability at least  $1 - \delta(t)$ , we have

$$
\left\| \boldsymbol {\Theta} - \hat {\boldsymbol {\Theta}} (t) \right\| _ {F} \leq 2 \frac {\sigma}{\phi^ {2} \sqrt {t}} f (\mathcal {G}, \boldsymbol {\Theta}) \sqrt {1 + 2 b \sqrt {| \mathcal {V} | \log \frac {1}{\delta (t)}} + 2 b \log \frac {1}{\delta (t)}},
$$

where

$$
f (\mathcal {G}, \boldsymbol {\Theta}) := \alpha_ {0} \left(a _ {2} (\mathcal {G}, \boldsymbol {\Theta}) + \sqrt {2} \mathbb {1} _ {\leq 1} (\kappa) w (\partial \mathcal {P})\right) \left(\frac {a _ {2} (\mathcal {G} , \boldsymbol {\Theta}) + \sqrt {2} \mathbb {1} _ {\leq 1} (\kappa) w (\partial \mathcal {P})}{a _ {1} (\mathcal {G} , \boldsymbol {\Theta}) \min  _ {\mathcal {C} \in \mathcal {P}} \sqrt {c _ {\mathcal {G}} (\mathcal {C})}} + 1\right).
$$

The proof of the previous theorem mainly relies on a decomposition of the estimation error signal into two parts: one is the projection of the error onto its mean per cluster value, that is, every node within the same cluster is mapped to the mean estimation error of its cluster. The second part of the decomposition is simply the residual part i.e. the deviation from the mean per cluster value, which is related to the incidence matrices of each cluster. The probabilistic statement comes from a high probability bound on the Euclidean norm of an empirical vector process associated with our problem, using a generalization of the Hanson-Wright inequality to the subgaussian case [Hsu et al., 2012, Theorem 2.1]. Compared to the bound of Jung [2020, Theorem 1], we bound a norm of the estimation error rather than just the total variation semi-norm. Additionally, the bound exhibits different behavior depending on whether  $\kappa > 1$ . Indeed, due to the expressions of  $a_1(\Theta, \mathcal{G})$  and  $a_2(\Theta, \mathcal{G})$ , in the case where  $\kappa > 1$ , the bound significantly decreases with the products  $w(\partial \mathcal{P}) \min_{\mathcal{C} \in \mathcal{P}} \sqrt{\iota(\mathcal{C})}$  and  $w(\partial \mathcal{P}) \max_{\mathcal{C} \in \mathcal{P}} c_{\mathcal{G}}(\mathcal{C})^{-\frac{1}{2}}$ , which are both small enough for dense intra-cluster edge links and sparse inter-cluster ones. However, when  $\kappa < 1$ , the  $w(\partial \mathcal{P})$  term might dominate if it is moderately large, and its effect can only be mitigated via a small subgaussianity constant  $\sigma$  or a large enough RE condition constant  $\phi$ .

# 5.3 RE condition for the empirical multi-task Gram matrix

To establish the oracle inequality, we assumed that the RE condition holds for the empirical multi-task Gram matrix. The goal of this section is to prove this holds with high probability. To this end, we use the same strategy as in Oh et al. [2021], Cella et al. [2023]. We prove that on the one hand, given the empirical multi-task Gram matrix inherits the RE condition from its adapted counterpart since it concentrates around it. On the other hand, we prove that the adapted Gram matrix verifies the RE condition due to Assumption 1, 2 and 4 made on the context generation distribution.

Theorem 2 (RE condition holding for the empirical multi-task Gram matrix). Under assumptions 2 and 4, let  $t \geq 1$ , and let  $\kappa, \phi$  be the constants from Assumption 4. Assume that  $\max_{m \in \mathcal{V}} |\mathcal{T}_m(t)| \leq bt$ . Then, for any  $\gamma \in \left(0, \left(1 + \frac{a_2(\mathcal{G},\Theta) + (1 - \kappa)^+ \sqrt{2} w(\partial \mathcal{P})}{a_1(\mathcal{G},\Theta)}\right)^{-2}\right)$ , the empirical multi-task Gram matrix verifies the RE condition with constants  $\kappa$  and  $\hat{\phi}$ , with

$$
\hat {\phi} = \tilde {\phi} \sqrt {1 - \gamma \left(1 + \frac {a _ {2} (\mathcal {G} , \boldsymbol {\Theta}) + (1 - \kappa) ^ {+} \sqrt {2} w (\partial \mathcal {P})}{a _ {1} (\mathcal {G} , \boldsymbol {\Theta})}\right) ^ {2}}, \tag {6}
$$

with a probability at least equal to  $1 - 6d|\mathcal{V}|$  exp  $\left(\frac{-3\gamma^2\tilde{\phi}^4(\min_{\mathcal{C}\in\mathcal{P}}(\tilde{c}_{\mathcal{G}}(\mathcal{C})\wedge\tilde{c}_{\mathcal{G}}(\mathcal{C})^2)t}{6b + 2\sqrt{2}\gamma\tilde{\phi}^2}\right)$ , where

$\tilde{\phi} \coloneqq \frac{\phi}{\sqrt{2\nu\omega}}$  and  $\tilde{c}_{\mathcal{G}}(\mathcal{C}) \coloneqq c_{\mathcal{G}}(\mathcal{C}) \wedge |\mathcal{C}| \quad \forall \mathcal{C} \in \mathcal{P}$ .

The proof follows the same approach as in Oh et al. [2021], Cella et al. [2023]; we prove that the RE condition transfers from the true multi-task Gram matrix to its adapted counterpart  $\mathbf{V}_{\mathcal{V}}(t)$ , defined as follows:

$$
\mathbf {V} _ {\mathcal {V}} (t) = \operatorname {d i a g} \left(\mathbf {V} _ {1} (t), \dots , \mathbf {V} _ {| \mathcal {V} |} (t)\right), \tag {7}
$$

where

$$
\mathbf {V} _ {m} (t) = \frac {1}{t} \sum_ {\tau \in \mathcal {T} _ {m} (t)} \mathbb {E} [ \mathbf {x} (\tau) \mathbf {x} (\tau) ^ {\top} | \mathcal {F} _ {\tau - 1} ]. \tag {8}
$$

This transfer relies on the work of Oh et al. [2021, lemma 10]. The other step of the proof is showing that the empirical multi-task Gram matrix and  $\mathbf{V}_{\mathcal{V}}(t)$  become close to each other with high probability after sufficiently many time steps, the respective distance between the two is measured with a matrix norm induced by the RE semi-norm and the restriction to set  $S$  (Definition 2). The bound showcases a dependence on  $\min_{\mathcal{C} \in \mathcal{P}} c_{\mathcal{G}}(\mathcal{C}) \wedge |\mathcal{C}|$ , which is of the same order as  $|\mathcal{C}|$  for a fully connected cluster with vertices  $\mathcal{C}$ . It is also clear that with a higher minimum centrality of a cluster, the probability of satisfying the RE condition increases.

# 5.4 Regret bound

To bound the regret, we bound the expected instantaneous regret for each round  $t \geq 1$ . This bound relies on the oracle inequality holding and on the RE condition being satisfied for the empirical Gram matrix, both with high probability. These two conditions are ensured and Theorem 1 and Theorem 2.

Theorem 3 (Regret bound). Let the mean horizon per node be  $\overline{T} = \frac{T}{|\mathcal{V}|}$ . Let  $\min_{\mathcal{C} \in \mathcal{P}} \sqrt{c_{\mathcal{G}}(\mathcal{C})}$  going asymptotically to infinity and  $\max_{\mathcal{C} \in \mathcal{P}} \sqrt{\iota_{\mathcal{G}}(\mathcal{C})}$  going asymptotically to zero as well as  $\max_{\mathcal{C} \in \mathcal{P}} \sqrt{\iota_{\mathcal{G}}(\mathcal{C})} w(\partial \mathcal{P})$  and  $\frac{w(\partial \mathcal{P})}{\min_{\mathcal{C} \in \mathcal{P}} \sqrt{c_{\mathcal{G}}(\mathcal{C})}}$  going asymptotically to zero. Under assumptions1 to 4 and  $\kappa < 1$ , the expected regret of the Network Lasso Bandit algorithm is upper bounded as follows:

$$
\mathcal {R} (| \mathcal {V} | \overline {{T}}) = \mathcal {O} \left(\sqrt {\frac {\bar {T}}{\underset {\mathcal {C} \in \mathcal {P}} {\min } c _ {\mathcal {G}} (\mathcal {C})}} \left(\sqrt {| \mathcal {V} |} + \sqrt {\log (\bar {T} | \mathcal {V} |)} + \sqrt [ 4 ]{| \mathcal {V} \log (\bar {T} | \mathcal {V} |) |}\right) + \frac {1}{A} \log (d | \mathcal {V} |)\right),
$$

$$
w i t h A = \frac {3 \gamma^ {2} \min _ {\mathcal {C} \in \mathcal {P}} (\tilde {c} _ {\mathcal {G}} (\mathcal {C}) \wedge \tilde {c} _ {\mathcal {G}} ^ {2} (\mathcal {C}))}{6 \frac {\log (| \mathcal {V} |)}{\sqrt {| \mathcal {V} |}} + 2 \sqrt {2} \gamma}.
$$

Our regret is mainly formed of two parts. The first one is the sublinear time-dependent term and represents the bulk of horizon dependence. Interestingly, it does not depend on the dimension, which is a consequence of using the concentration inequality from Hsu et al. [2012]. Interestingly, it decreases as the topological centrality index grows with the graph size, which proves the importance of intra-cluster high connectivity.

The second significant term comes from ensuring the RE condition for the empirical multi-task Gram matrix, and can be interpreted as the number of time steps necessary for it to hold, as pointed out by Oh et al. [2021]. It has a logarithmic dependence in the graph size and in the dimension, which is a characteristic of regret bound of the "lasso type". Also noteworthy is that the regret grows with  $\log (d)$  only in the time-independent term, making our policy useful in high-dimensional settings.

# 6 Experiments

We provide experiments to showcase the effect on the problem's parameters on our algorithm's performance as well as highlighting its advantageous performance compared to other algorithms. At each time step, the algorithm solves the network lasso problem (2) via a primal-dual algorithm used in Jung [2020].

We compare our algorithm to several baselines of the literature. On the one hand, baselines relying on a given graph, GOBLin [Cesa-Bianchi et al., 2013] and GraphUCB [Yang et al., 2020] that use the Laplacian to smooth the preference vectors. On the other hand, we consider online clustering of bandits baselines, namely CLUB [Gentile et al., 2014] and SCLUB [Li et al., 2019]. Since these latter approaches start with a fully connected graph, we provide them the known graph for a fair comparison. As a sanity check, we also compare the independent task learning case with LinUCB (LinUcbITL) where each task is solved independently, and to the case of a LinUCB agent for each cluster (LinUcbOracle). The graph used is generated using stochastic block models in order to ensure that the generated graph induces a cluster structure, where an edge is constructed with probability  $p$  within clusters and  $q$  between clusters.

Experimentally, we found that normalizing the adjacency matrix, that is we utilize the following normalized edges:  $w_{mn} = \frac{1}{\sqrt{\deg(m)\deg(n)}}$ , where  $\deg(m)$  denotes the degree of node  $m$ , yields significantly better results. Indeed, such a normalization makes the algorithm focus more on edges between low-degree nodes, which improves the propagation of the collected information within the graph. In all experiments we have set  $\alpha_0 = 0.1$ .

Our results clearly showcase an improvement compared to the other baselines. Apart from the oracle that has complete knowledge of all clusters from the beginning, our policy performs significantly better than the rest beyond the error margins, covering one standard deviation at ten repetitions. We

![](images/1f5680964180c69ed01842a6b56a70d954c5c3861af31376c009f5079fae3389.jpg)  
(a)  $|\mathcal{V}| = 100, d = 20, p = 0.4, q = 0.1$

![](images/ea4c66c6fda6770fcf18e0250ffdbb5aff0cfcd01ccb41ee4cc12bda18b3c354.jpg)  
(b)  $|\mathcal{V}| = 100, d = 10, p = 0.5, q = 0.1$

![](images/2a13d99152492fd326c5ea8a75365325ed1d509508458d9460939a51c5b8ab05.jpg)  
(c)  $|\mathcal{V}| = 50, d = 80, p = 0.8, q = 0.2$

![](images/715307fd9af74a99c57320dd998b5bbf7b18828ac9025f34e18452a0f4c5c9b8.jpg)  
Figure 1: Synthetic data experiment showing the cumulative regret of Network Lasso Policy as a function of time-steps compared to other baselines, for different choices of  $|\mathcal{V}|$ ,  $d$ ,  $p$  and  $q$ .  
(d)  $|\mathcal{V}| = 200, d = 20, p = 0.5, q = 0.05$

provide results for up to  $|\mathcal{V}| = 500$  nodes showing the effective transfer of knowledge within the graph.

# 7 Conclusion and future perspectives

In this work, we proposed a multi-task bandit framework that solves the case where the task preference vectors are piecewise constant over a graph. To this end, we used the Network Lasso policy to estimate the task parameters, which bypasses explicit clustering procedures. We showed a sublinear regret bound and as a byproduct, we proved a novel oracle inequality that relies on the small size of the boundary as well as on the high value of the topological centrality index of each node within its cluster. Our experimental evaluations highlight the advantage of our method, especially when either the number of dimensions or nodes increases.

Due to the technical similarity of our problem with the Lasso, a natural extension would be to extend it to a thresholded approach, in the same vein as [Ariu et al., 2022]. Another possible extension would be to use regularization with higher order total variation terms that impose a piecewise polynomial signal on a graph, as explained for scalar signals in Wang et al. [2016], Ortelli and van de Geer [2019].

# References

Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011.

K. Ariu, K. Abe, and A. Proutiere. Thresholded Lasso Bandit. In Proceedings of the 39th International Conference on Machine Learning, pages 878-928. PMLR, 2022.  
H. Bastani and M. Bayati. Online Decision Making with High-Dimensional Covariates. Operations Research, 2019. doi: 10.1287/opre.2019.1902.  
S. Basu, B. Kveton, M. Zaheer, and C. Szeptesvari. No Regrets for Learning the Prior in Bandits. In Advances in Neural Information Processing Systems, 2021.  
S. Bilaj, S. Dhouib, and S. Maghsudi. Meta learning in bandits within shared affine subspaces. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics. PMLR, 2024.  
J. Borge-Holthoefer, A. Rivero, I. Garcia, E. Cauhe, A. Ferrer, D. Ferrer, D. Francois, D. Iniguez, M. P. Pérez, G. Ruiz, et al. Structural and dynamical patterns on online social networks: the spanish may 15th movement as a case study. *PloS one*, 6(8), 2011.  
P. Buhlmann and S. van de Geer. Statistics for high-dimensional data. Springer Series in Statistics. Springer, Heidelberg, 2011. ISBN 978-3-642-20191-2.  
L. Cella and M. Pontil. Multi-task and meta-learning with sparse linear bandits. In Uncertainty in Artificial Intelligence. PMLR, 2021.  
L. Cella, A. Lazaric, and M. Pontil. Meta-learning with stochastic linear bandits. In Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020.  
L. Cella, K. Lounici, G. Pacreau, and M. Pontil. Multi-task representation learning with stochastic linear bandits. In International Conference on Artificial Intelligence and Statistics, 2023.  
N. Cesa-Bianchi, C. Gentile, and G. Zappella. A gang of bandits. Advances in neural information processing systems, 26, 2013.  
J. Cheeger. A lower bound for the smallest eigenvalue of the laplacian. Problems in analysis, 1970.  
X. Cheng, C. Pan, and S. Maghsudi. Parallel online clustering of bandits via hedonic game. In International Conference on Machine Learning, pages 5485-5503. PMLR, 2023.  
W. Chu, L. Li, L. Reyzin, and R. Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings, 2011.  
X. Dong, D. Thanou, M. Rabbat, and P. Frossard. Learning graphs from data: A signal representation perspective. IEEE Signal Processing Magazine, 2019.  
D. Easley, J. Kleinberg, et al. Networks, crowds, and markets: Reasoning about a highly connected world, volume 1. Cambridge university press Cambridge, 2010.  
A. Fontan and C. Altafini. On the properties of laplacian pseudoinverses. In 2021 60th IEEE Conference on Decision and Control (CDC). IEEE, 2021.  
C. Gentile, S. Li, and G. Zappella. Online clustering of bandits. In International Conference on Machine Learning, pages 757-765. PMLR, 2014.  
C. Gentile, S. Li, P. Kar, A. Karatzoglou, G. Zappella, and E. Etrue. On context-dependent clustering of bandits. In International Conference on machine learning, pages 1253-1262. PMLR, 2017.  
D. Hallac, J. Leskovec, and S. Boyd. Network lasso: Clustering and optimization in large graphs. In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pages 387-396, 2015.  
M. Herbster, S. Pasteris, F. Vitale, and M. Pontil. A gang of adversarial bandits. Advances in Neural Information Processing Systems, 34, 2021.  
D. Hsu, S. Kakade, and T. Zhang. A tail inequality for quadratic forms of subgaussian random vectors. Electronic Communications in Probability, 17, 2012.

J. Hu, X. Chen, C. Jin, L. Li, and L. Wang. Near-optimal representation learning for linear bandits and linear rl. In International Conference on Machine Learning. PMLR, 2021.  
A. Jung. Networked Exponential Families for Big Data Over Networks. IEEE Access, 8, 2020. ISSN 2169-3536.  
A. Jung and N. Vesselinova. Analysis of network lasso for semi-supervised regression. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 380-387. PMLR, 2019.  
A. Jung, N. Tran, and A. Mara. When Is Network Lasso Accurate? Frontiers in Applied Mathematics and Statistics, 3, 2018. ISSN 2297-4687.  
G.-S. Kim and M. C. Paik. Doubly-robust lasso bandit. Advances in Neural Information Processing Systems, 32, 2019.  
B. Kveton, M. Konobeev, M. Zaheer, C.-w. Hsu, M. Mladenov, C. Boutilier, and C. Szepesvari. Meta-thompson sampling. In International Conference on Machine Learning. PMLR, 2021.  
L. Li, W. Chu, J. Langford, and R. E. Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pages 661-670, 2010.  
S. Li, W. Chen, and K.-S. Leung. Improved algorithm on online clustering of bandits. arXiv preprint arXiv:1902.09162, 2019.  
M. McPherson, L. Smith-Lovin, and J. M. Cook. Birds of a feather: Homophily in social networks. Annual review of sociology, 27(1):415-444, 2001.  
M. E. Newman. Modularity and community structure in networks. Proceedings of the national academy of sciences, 103(23):8577-8582, 2006.  
T. T. Nguyen and H. W. Lauw. Dynamic clustering of contextual multi-armed bandits. In Proceedings of the 23rd ACM international conference on conference on information and knowledge management, pages 1959–1962, 2014.  
B. Nourani-Koliji, S. Bilaj, A. R. Balef, and S. Maghsudi. Piecewise-stationary combinatorial semi-bandit with causally related rewards. arXiv preprint arXiv:2307.14138, 2023.  
M.-H. Oh, G. Iyengar, and A. Zeevi. Sparsity-Agnostic Lasso Bandit. In Proceedings of the 38th International Conference on Machine Learning, pages 8271-8280. PMLR, 2021.  
F. Ortelli and S. van de Geer. Synthesis and analysis in total variation regularization. arXiv preprint arXiv:1901.06418, 2019.  
A. Peleg, N. Pearl, and R. Meir. Metalearning linear bandits by prior update. In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics. PMLR, 2022.  
G. Ranjan and Z.-L. Zhang. Geometry of complex networks and topological centrality. Physica A: Statistical Mechanics and its Applications, 2013.  
X. Su and T. M. Khoshgoftaar. A survey of collaborative filtering techniques. Advances in artificial intelligence, 2009, 2009.  
R. Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 1996.  
J. Tropp. Freedman's inequality for matrix martingales. Electronic Communications in Probability, 16:262 - 270, 2011.  
P. Van Mieghem, K. Devriendt, and H. Cetinay. Pseudoinverse of the laplacian and best spreader node in a network. Physical Review E, 2017.  
Y.-X. Wang, J. Sharpnack, A. J. Smola, and R. J. Tibshirani. Trend filtering on graphs. Journal of Machine Learning Research, 17(105):1-41, 2016. URL http://jmlr.org/papers/v17/15-147.html.

K. Yang and L. Toni. Graph-based recommendation system. In 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP), pages 798-802. IEEE, 2018.  
K. Yang, L. Toni, and X. Dong. Laplacian-regularized graph bandits: Algorithms and theoretical analysis. In International Conference on Artificial Intelligence and Statistics, pages 3133-3143. PMLR, 2020.  
M. Yuan and Y. Lin. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society Series B: Statistical Methodology, 2006.
