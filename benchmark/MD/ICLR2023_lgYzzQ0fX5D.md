# GRAPH NEURAL NETWORKS ARE MORE POWERFUL THAN WE THINK

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph Neural Networks (GNNs) are powerful convolutional architectures that have shown remarkable performance in various node-level and graph-level tasks. Despite their success, the common belief is that the expressive power of standard GNNs is limited and that they are at most as discriminative as the Weisfeiler-Lehman (WL) algorithm. In this paper we argue the opposite and show that the WL algorithm is the upper bound only when the input to the GNN is the vector of all ones. In this direction, we derive an alternative analysis that employs linear algebraic tools and characterize the representational power of GNNs with respect to the eigenvalue decomposition of the graph operators. We show that GNNs can distinguish between any graphs that differ in at least one eigenvalue and design simple GNN architectures that are provably more expressive than the WL algorithm. Thorough experimental analysis on graph isomorphism and graph classification datasets corroborates our theoretical results and demonstrates the effectiveness of the proposed architectures.

# 1 INTRODUCTION

Graph Neural Networks (GNNs) have emerged in the field of machine learning and artificial intelligence as powerful tools that process network structures and network data. Their convolutional architecture allows them to inherit all the favorable properties of convolutional neural networks (CNNs), while they also exploit the graph structure. Biology (Gainza et al., 2020; Strokach et al., 2020; Jiang et al., 2021), quantum chemistry (Gilmer et al., 2017), as well as robotics (Lima et al., 2020; Cranmer et al., 2021), social networks and recommender systems (Ying et al., 2018; Wu et al.) are typical fields where GNNs have been applied. GNNs have demonstrated state-of-the-art performance in various downstream tasks, associated with these fields, that include (but are not limited to) node and graph classification, link prediction and network regression.

Despite their remarkable performance, the success of GNNs is still to be demystified. A lot of research has been conducted to theoretically support the experimental developments, focusing on understanding the functionality of GNNs and analyzing their properties. In particular, permutation invariance-equivariance (Maron et al., 2018), stability to perturbations (Gama et al., 2020) and transferability (Ruiz et al., 2020a; Levie et al., 2021) are properties tantamount to the success of the GNNs. Lately, the research focus has been shifted towards analyzing their expressive power, since the universality of GNNs depends on their ability to produce different outputs for different graphs. The common belief is that standard GNNs have limited expressive power (Xu et al., 2019) and that it is upper bounded by the expressive power of the Weisfeiler-Lehman (WL) algorithm (Weisfeiler & Leman, 1968). This induced increased research activity towards building complex and more expressive GNNs. In this work we argue the opposite. We prove that standard graph convolutional structures are capable of distinguishing between graphs that the WL algorithm cannot and therefore complex GNNs are not necessary to break the WL limits.

Our work is motivated by the following questions. How expressive are GNNs? Can simple convolutional architectures be more expressive than the WL algorithm? The answer to both questions is definitive. Our analysis utilizes spectral decomposition tools to show that the source of the WL test as a limit for the expressive power of GNNs is the use of the all-one vector as an input. Our spectral analyses corroborate that, indeed, if a GNN is initialized with the all-one vector as an input, the WL test is a limit on the expressive power of GNNs. However, if we initialize a GNN with white

random inputs, it is possible to discriminate, at least, any pair of graphs with at least one different eigenvalue. This implies that standard GNNs are provably more expressive than the WL algorithm as they discriminate between graphs that fail the WL test, yet have different eigenvalues. In fact, having at least one different eigenvalue is a very mild condition that is rarely not met in practice.

Using white noise as an input to a GNN may be computationally costly. We show, however, that there are two alternative GNN architectures that are equivalent to a GNN architecture with white random inputs: (i) A GNN that operates on matrix representations of the graph without requiring any input. (ii) A GNN in which input features are derived from powers of matrix representations of the graph. Our numerical results show that our proposed GNNs are better discriminator in some graph classification problems.

Our contribution is summarized as follows:

(C1) We characterize the expressive power of GNNs employing spectral decomposition tools.  
(C2) We explain that the WL algorithm is a limit on the expressive power of GNNs only when we use the all-one vector as an input.  
(C3) We show that standard GNNs can distinguish between any pair of graphs with at least one different eigenvalue if node features are initialized with white random noise. This implies that standard GNNs are provably more expressive than the WL algorithm.  
(C4) We design equivalent architectures that circumvent the use of random input features. These architectures can use features derived from powers of matrix representations or can avoid the use of features altogether.  
(C5) We demonstrate the effectiveness of using GNNs with white random inputs, or the proposed alternatives, in graph isomorphism and graph classification datasets.

Related work: The first work to study the approximation properties of the GNNs was by (Scarselli et al., 2008a). Along the same lines (Maron et al., 2019b; Keriven & Peyre, 2019) discuss the universality of GNNs for permutation invariant or equivariant functions. Then the scientific attention focused on the ability of GNNs to distinguish between non-isomorphic graphs. The works of (Morris et al., 2019; Xu et al., 2019) place the expressive power of GNNs with respect to that of the WL algorithm and prompted various follow-up works in the area. Specifically, (Abboud et al., 2021; Sato et al., 2021) use random features to increase the separation capabilities of GNNs, whereas (Tahmasebi et al., 2020) produce features related to the subgraph information by adding a neighborhood pooling layer. Furthermore (Corso et al., 2020; Beini et al., 2021) use multiple and directional aggregators, respectively, to increase the GNN expressivity. GNNs that use k-tuple and k-subgraph information have been designed by (Maron et al., 2019a; Murphy et al., 2019; Azizian et al., 2020; Morris et al., 2020; Geerts & Reutter, 2021). These works use a tensor framework, and employ more expressive structures compared to simple GNNs. However, they are usually computationally heavier to implement and also prone to overfitting. Moreover, (Balcilar et al., 2021) design convolutions in the spectral domain to produce powerful GNNs, whereas (Loukas, 2019) studies the learning capabilities of a GNN with respect to its width and depth. Finally, (Chen et al., 2019) reveal a connection between the universal approximation and the capacity capabilities of GNNs.

# 2 ON THE EXPRESSIVE POWER OF GNNS

Studying the expressive power of GNNs has attracted significant attention, since it sheds light on the success and general functionality of graph convolutional architectures. One of the most influential works by (Xu et al., 2019) compares the representational capabilities of GNNs with those of the WL algorithm (color refinement algorithm). The claim is that GNNs are at most as powerful as the WL algorithm in distinguishing between different graphs. This is indeed true when the input to the GNN is the vector of all ones, i.e.,  $\mathbf{x} = \mathbf{1}$  and therefore the propagated graph signals are of the form  $S^k\mathbf{1}$ , where  $S \in \{0,1\}^{N\times N}$  is the graph adjacency and  $S^k$  is the  $k$ -th power of  $S$ .

A question that naturally arises is 'Why limit attention to input features  $\mathbf{x} = \mathbf{1}$ ?'. On the one hand, various systems of practical interest provide access to a set of attributes or graph signals,  $\mathbf{X} \in \mathbb{R}^{N \times D}$ , that are associated with each node. These attributes contain rich information beyond the connectivity

provided by the graph, as opposed to  $\pmb{x} = \mathbf{1}$  inputs. On the other hand, when attributes are not available, we can design artificial features from the graph that incorporate valuable knowledge, not captured by  $S^k\mathbf{1}$ . The need for further analysis with general input signals is therefore clear.

To better understand the expressive power of GNNs we study general inputs. Consider then graphs  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  with graph operators  $S$ ,  $\hat{S}$ . In this paper  $S$ ,  $\hat{S}$  denote the graph adjacencies, but other choices of graph operators can be used, e.g., graph Laplacians or weighted graph adjacencies/Laplacians. We assume that  $S$ ,  $\hat{S}$  are both symmetric and thus admit eigenvalue decompositions as:

$$
\boldsymbol {S} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {T}, \quad \hat {\boldsymbol {S}} = \hat {\boldsymbol {U}} \hat {\boldsymbol {\Lambda}} \hat {\boldsymbol {U}} ^ {T}, \tag {1}
$$

The equations in (1) represent the spectral decompositions of  $\mathcal{G}$  and  $\hat{\mathcal{G}}$ , where  $\pmb{U}$ ,  $\hat{\pmb{U}}$  are orthogonal matrices containing the eigenvectors and  $\Lambda$ ,  $\hat{\Lambda}$  are the diagonal matrices of corresponding eigenvalues. The graphs  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  are non-isomorphic if and only if there does not exist a permutation matrix  $\Pi$  such that  $\pmb{S} = \pmb{\Pi}\hat{\pmb{S}}\pmb{\Pi}^T$ . A broad class of non-isomorphic graphs have at least one different eigenvalue. To be more precise, let  $\mathcal{S} = \{\lambda_1,\dots ,\lambda_N\}$  be the multiset containing the eigenvalues of  $\pmb{S}$  and  $\hat{\mathcal{S}} = \{\hat{\lambda}_1,\dots ,\hat{\lambda}_N\}$  be the multiset containing the eigenvalues of  $\hat{\mathcal{S}}$ . The following assumption is heavily used in the main part of this paper:

Assumption 2.1  $S$ ,  $\hat{S}$  have at least one different eigenvalue, i.e., there exists  $\mu_k$  with multiplicity  $m$  and corresponding eigenspace  $\mathbf{V} \in \mathbb{R}^{N \times m}$ , such that  $\mu_k \in S$  but  $\mu_k \notin \hat{S}$ .

When Assumption 2.1 holds,  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  are always non-isomorphic. Assumption 2.1 is not restrictive. Real non-isomorphic graphs have different eigenvalues with very high probability (Haemers & Spence, 2004). Corner cases where Assumption 2.1 doesn't hold are studied in Appendix H.

First, we consider GNNs that are constructed by the following modules, corresponding to the neurons of a typical (non-graph) neural network:

$$
\boldsymbol {Y} = \sigma \left(\sum_ {k = 0} ^ {K - 1} \boldsymbol {S} ^ {k} \boldsymbol {X} \boldsymbol {H} _ {k}\right). \tag {2}
$$

The module in (2) is composed by a graph filter of length  $K$  followed by a nonlinearity  $\sigma(\cdot)$ .  $H_{k}$  represents the filter parameters and can be a matrix, a vector, or a scalar. In order to characterize the representational power of GNNs with general input, we provide the following theorem:

Theorem 2.2 Let  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  be non-isomorphic graphs with graph signals  $\pmb{X}$ ,  $\hat{\pmb{X}}$ . There exist a GNN that tells  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  apart if:

1. There does not exist permutation matrix  $\Pi$  such that  $X = \Pi \hat{X}$ , or  
2. Assumption 2.1 holds and  $V^T X \neq 0$ .

Theorem 2.2 highlights the importance of the input  $X$  in the representational capabilities of a GNN. For problems in which inputs are given, it states that a GNN can distinguish between non-isomorphic graphs if they have different graph signals or their signals are not orthogonal to the eigenspace associated with the eigenvalue that differentiates them. In problems where inputs are not available, Theorem 2.2 provides guidelines on how to design input  $X$  from the graph.

Theorem 2.2 also indicates that the limitations of GNNs discussed in (Xu et al., 2019) are not due to the architecture but they are limitations associated with the input. In particular,  $x = 1$  fails to satisfy condition 1, while it is also prone to fail condition 2, since the majority of real graphs have eigenvectors that are orthogonal to 1. Thus, the challenge lies in designing GNN inputs that satisfy conditions 1 and 2. This is accomplished in section 5, where we propose to construct  $X$  as:

$$
\boldsymbol {X} = \left[ \operatorname {d i a g} \left(\boldsymbol {S} ^ {0}\right), \operatorname {d i a g} \left(\boldsymbol {S} ^ {1}\right), \operatorname {d i a g} \left(\boldsymbol {S} ^ {2}\right), \dots , \operatorname {d i a g} \left(\boldsymbol {S} ^ {D - 1}\right) \right], \tag {3}
$$

where  $\operatorname{diag}\left(\boldsymbol{S}^k\right)$  refers to vector containing the diagonal entries of matrix  $\boldsymbol{S}^k$ . Proper choice of  $D$  guarantees that the graph signal  $\mathbf{X}$  in (3) satisfies both conditions of Theorem 2.2 and enables GNNs

to distinguish between any non-isomorphic graphs that have at least one different eigenvalue. A nice interpretation of this result is given in section 5 and shows that  $\mathbf{X}$  in (3) combines information from both the k-hop degrees  $(S^k\mathbf{1})$  and the high-order subgraphs that appear in the network. As shown in the next section, the WL algorithm cannot always tell graphs with different eigenvalues apart, which implies that GNNs are more expressive than the WL algorithm for this class of graphs.

# 3 LIMITATIONS OF GNNS WITH  $\pmb{x} = \mathbf{1}$  INPUT AND THE WL ALGORITHM

Using Theorem 2.2 we can explain why feeding a GNN with  $\pmb{x} = \mathbf{1}$  is limiting. The limitations associated with input  $\pmb{x} = \mathbf{1}$  are also highly related to the limitations of the WL algorithm. The problem appears in graphs that admit a spectral decomposition with eigenvectors that are orthogonal to  $\mathbf{1}$  (they sum up to zero). Following condition 2 in Theorem 2.2, if two graphs are the same except eigenvalues that correspond to eigenvectors that sum up to zero, then GNNs with input  $\pmb{x} = \mathbf{1}$  will fail to tell the two graphs apart. To see this consider the graphs  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  with spectral decompositions:

$$
\boldsymbol {S} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {T} = \lambda_ {1} \boldsymbol {u} _ {1} \boldsymbol {u} _ {1} ^ {T} + \lambda_ {2} \boldsymbol {u} _ {2} \boldsymbol {u} _ {2} ^ {T} + \lambda_ {3} \boldsymbol {u} _ {3} \boldsymbol {u} _ {3} ^ {T}, \tag {4}
$$

$$
\hat {\boldsymbol {S}} = \hat {\boldsymbol {U}} \hat {\boldsymbol {\Lambda}} \hat {\boldsymbol {U}} ^ {T} = \lambda_ {1} \boldsymbol {u} _ {1} \boldsymbol {u} _ {1} ^ {T} + \lambda_ {2} \boldsymbol {u} _ {2} \boldsymbol {u} _ {2} ^ {T} + \hat {\lambda} _ {3} \boldsymbol {u} _ {3} \boldsymbol {u} _ {3} ^ {T}, \tag {5}
$$

where  $\lambda_3\neq \hat{\lambda}_3$  .If  $\pmb{u}_{3}$  is orthogonal to 1 then:

$$
\boldsymbol {S} ^ {k} \mathbf {1} = \boldsymbol {U} \boldsymbol {\Lambda} ^ {k} \boldsymbol {U} ^ {T} \mathbf {1} = \lambda_ {1} ^ {k} \boldsymbol {u} _ {1} \boldsymbol {u} _ {1} ^ {T} \mathbf {1} + \lambda_ {2} ^ {k} \boldsymbol {u} _ {2} \boldsymbol {u} _ {2} ^ {T} \mathbf {1} + \lambda_ {3} ^ {k} \boldsymbol {u} _ {3} \boldsymbol {u} _ {3} ^ {T} \mathbf {1} = \lambda_ {1} ^ {k} \left(\boldsymbol {u} _ {1} ^ {T} \mathbf {1}\right) \boldsymbol {u} _ {1} + \lambda_ {2} ^ {k} \left(\boldsymbol {u} _ {2} ^ {T} \mathbf {1}\right) \boldsymbol {u} _ {2} \tag {6}
$$

$$
\hat {\boldsymbol {S}} ^ {k} \mathbf {1} = \hat {\boldsymbol {U}} \hat {\boldsymbol {\Lambda}} ^ {k} \hat {\boldsymbol {U}} ^ {T} \mathbf {1} = \lambda_ {1} ^ {k} \mathbf {u} _ {1} \mathbf {u} _ {1} ^ {T} \mathbf {1} + \lambda_ {2} ^ {k} \mathbf {u} _ {2} \mathbf {u} _ {2} ^ {T} \mathbf {1} + \hat {\lambda} _ {3} ^ {k} \mathbf {u} _ {3} \mathbf {u} _ {3} ^ {T} \mathbf {1} = \lambda_ {1} ^ {k} \left(\mathbf {u} _ {1} ^ {T} \mathbf {1}\right) \mathbf {u} _ {1} + \lambda_ {2} ^ {k} \left(\mathbf {u} _ {2} ^ {T} \mathbf {1}\right) \mathbf {u} _ {2} \tag {7}
$$

The diffused information in GNNs with this naive input is of the form  $S^k \mathbf{1}$  and therefore in the above example the decisive information that differentiates the two graphs is omitted.

Graphs with eigenvectors orthogonal to  $\mathbf{1}$  can also affect the performance of the WL algorithm. In the absence of features the WL algorithm is initialized with  $x = S\mathbf{1}$ , which is propagated through the nodes iteratively. In graphs with eigenvectors orthogonal to  $\mathbf{1}$ , the propagated degrees have suffered critical information loss in the initialization, which in certain graph structures is impossible to recover, as WL iterations progress. Further analysis on this subject can be found in Appendix C.

Classic examples of graphs with different eigenvalues, that the WL algorithm and GNNs with  $x = 1$  input cannot tell apart, are presented in Figs. 1, 2. In particular, these approaches decide that  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  in Fig. 1 and  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  in Fig. 2 are the same. This is due to the fact that these graphs contain eigenvectors that are orthogonal to 1. The case of Fig. 1 is straightforward. All the nodes of  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  have the same degree,

i.e.,  $\pmb{x} = \mathbf{1}$  is an eigenvector in both graphs and thus orthogonal to all the remaining eigenvectors. As a result, the node degrees (which are the same for both graphs) are the only information that the WL algorithm and GNNs with 1 input are able to process. The case of Fig. 2 is more complicated;  $\pmb{x} = \mathbf{1}$  is not an eigenvector in any of the graphs, but it is orthogonal to the eigenvectors corresponding to the eigenvalues that differentiate the two graphs. Consequently, the operation  $S1$  negates vital information and the two approaches fail.

![](images/d833751154d08eb67d9df6a44aa498999c8d73af1ae4f1798452faaf7507d44e.jpg)  
(a)  $\mathcal{G}$

![](images/b2c62e11f98342ba667a9dc937b1e850415c904b9c434901c793cf8df3c6d853.jpg)  
Figure 1: WL indistinguishable graphs.  
(b)  $\hat{\mathcal{G}}$

Detailed information about the eigenvalues and eigenvectors of the graphs in Figs. 1, 2 can be found in Tables 7, 8 of Appendix K. This information corroborates the issues discussed in the previous paragraph. As noted earlier and will be explained in more detail in the upcoming sections, carefully designed GNNs overcome these issues and decide that  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  in both Figs. 1, 2 are non-isomorphic.

![](images/2afd1b07f45508348e834ade8b5e3b2ae11f67a2ee480cd3157e7fe601cb7de4.jpg)  
(a)  $\mathcal{G}$  
Figure 2: WL indistinguishable graphs

![](images/c681ee5ce5dfe26dbf26167da8c59b3c1769d5fd8cb60b75717df061ff9ce966.jpg)  
(b)  $\hat{\mathcal{G}}$

![](images/4779570dd78d7bff9281baa605dfe45078b294c60765bafa16a55fd7d06d42e6.jpg)  
(a) Stochastic GNN module

![](images/195c024c968544d41692f67694b0ff732ebb89ca1d5eec8134fb918f9500406f.jpg)  
Figure 3: GNN with random Gaussian input  
(b) equivalent model

# 4 FEEDING THE GNN WITH RANDOM INPUT

In this section we overcome the GNN limitations associated with  $\pmb{x} = \mathbf{1}$  by feeding a GNN with white Gaussian input. We consider again the GNN module in (2) where  $H_{k}$  is a scalar, i.e.,  $\pmb{y} = \sigma \left( \sum_{k=0}^{K-1} h_{k} S^{k} \pmb{x} \right)$ . Before choosing an appropriate nonlinearity, let us focus on the linear convolutional graph filter of length  $K$ :

$$
\boldsymbol {z} = \sum_ {k = 0} ^ {K - 1} h _ {k} \boldsymbol {S} ^ {k} \boldsymbol {x}, \tag {8}
$$

which we load with random input  $\pmb{x} \in \mathbb{R}^N$  that is drawn from a Gaussian distribution, i.e.,  $\pmb{x} \sim \mathcal{N}(\mathbf{0}, \pmb{I})$ . Since  $\pmb{x}$  is a random vector with  $\mathbb{E}[x] = 0$ ,  $z$  is also a random vector with  $\mathbb{E}[z] = 0$ . Thus, the expected value provides no information about the network. Measuring the covariance, on the other hand, yields:

$$
\begin{array}{l} \operatorname {c o v} [ \boldsymbol {z} ] = \mathbb {E} [ \boldsymbol {z} \boldsymbol {z} ^ {T} ] = \mathbb {E} \left[ \sum_ {k = 0} ^ {K - 1} h _ {k} \boldsymbol {S} ^ {k} \boldsymbol {x} \boldsymbol {x} ^ {T} \sum_ {m = 0} ^ {K - 1} h _ {m} \boldsymbol {S} ^ {m ^ {T}} \right] = \sum_ {k = 0} ^ {K - 1} h _ {k} \boldsymbol {S} ^ {k} \mathbb {E} [ \boldsymbol {x} \boldsymbol {x} ^ {T} ] \sum_ {m = 0} ^ {K - 1} h _ {m} \boldsymbol {S} ^ {m} \\ = \sum_ {k = 0} ^ {K - 1} h _ {k} \boldsymbol {S} ^ {k} \sum_ {m = 0} ^ {K - 1} h _ {m} \boldsymbol {S} ^ {m} = \sum_ {k = 0} ^ {K - 1} \sum_ {m = 0} ^ {K - 1} h _ {k} h _ {m} \boldsymbol {S} ^ {k} \boldsymbol {S} ^ {m} = \sum_ {k = 0} ^ {2 K - 2} h _ {k} ^ {\prime} \boldsymbol {S} ^ {k}, \tag {9} \\ \end{array}
$$

where  $h_k' = \sum_{m,l} h_m h_l$ , such that  $m + l = k$ . The results of equation (9) are noteworthy. We have shown that the covariance of a graph filter with random uncorrelated input corresponds to a different graph filter with no input. Furthermore, the resulting filter has length  $2K - 1$ , whereas the original filter has length  $K$ . In other words the nonlinearity introduced by the covariance computation enables the filter to gather information from a broader neighborhood compared to the initial filter. However, there is a caveat that the degrees of freedom for  $h'$  are  $K$  and not  $2K - 1$ . Further discussion on the subject can be found in Appendix D.

In practice we want to associate the output of a GNN with a feature for each node that is permutation equivariant. This is not the case with the rows or columns of the covariance matrix in (9). Therefore we choose  $\sigma(\cdot)$  to be the variance of each node i.e.,

$$
\boldsymbol {y} = \sigma (\boldsymbol {z}) = \operatorname {v a r} [ \boldsymbol {z} ] = \mathbb {E} \left[ \boldsymbol {z} ^ {2} \right] = \operatorname {d i a g} (\operatorname {c o v} [ \boldsymbol {z} ]) = \operatorname {d i a g} \left(\sum_ {k = 0} ^ {2 K - 2} h _ {k} ^ {\prime} \boldsymbol {S} ^ {k}\right) = \sum_ {k = 0} ^ {2 K - 2} h _ {k} ^ {\prime} \operatorname {d i a g} \left(\boldsymbol {S} ^ {k}\right). \tag {10}
$$

The proposed stochastic GNN module is illustrated in Fig. 3a. Regarding its expressive power, we present the following theorem:

Theorem 4.1 Let  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  be non-isomorphic graphs. If Assumption 2.1 holds, there exists a GNN with modules as in Fig. 3a that tells the two graphs apart.

In simple words, a GNN with modules as in Fig. 3a can always distinguish between graphs that have at least one different eigenvalue.

Proposition 4.1 The GNN module in Fig. 3a with random input  $\pmb{x} \sim \mathcal{N}(\pmb{0}, \pmb{I})$  is equivalent to the GNN module in Fig. 3b with no input up to degrees of freedom (dependencies) in the filter parameters.

The proof of Proposition 4.1 is the combination of equations (9), (10). The claim is eminent. It proves equivalence of two GNN architectures; a stochastic graph filter with Gaussian input followed by a variance nonlinearity with a deterministic graph filter followed by a diagonal operator. Depending on the problem and the variance of the system one has the option to choose either of them. Further discussion on the stochastic approach can be found in Appendix D.

# 5 THE DIAGONAL MODULE

Proposition 4.1 proved the equivalence of the two GNN modules in Fig. 3. In this section we focus on the module in 3a and analyze its unique properties. To be more precise, we study the following diagonal GNN module:

$$
\boldsymbol {y} = \sigma \left(\sum_ {k = 0} ^ {K - 1} h _ {k} \operatorname {d i a g} \left(\boldsymbol {S} ^ {k}\right)\right), \tag {11}
$$

Note that the module in (11) is not exactly the same as the one in Fig. 3b, since a nonlinearity is added and the filter is of length  $K$ . As an example, we test the proposed diagonal module on the graphs of Figs. 1, 2, and present the output  $y$  of (11) with parameters  $(h_0, h_1, h_2, h_3, h_4, h_5) = (10, 1, -\frac{1}{2}, \frac{1}{3}, -\frac{1}{4}, \frac{1}{5})$  and ReLU nonlinearity, in Table 1.

Table 1: Outputs  $\pmb{y}$  of  $\mathcal{G}$  and  $\hat{\pmb{y}}$  of  $\hat{\mathcal{G}}$  of the proposed diagonal module for the graphs in Figs. 1, 2.  

<table><tr><td rowspan="2" colspan="2">GRAPH</td><td colspan="10">NODE</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td><td>I</td><td>J</td></tr><tr><td rowspan="2">FIG. 1</td><td>y</td><td>10.42</td><td>10.42</td><td>10.42</td><td>10.42</td><td>10.42</td><td>10.42</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>\(\hat{y}\)</td><td>1.75</td><td>1.75</td><td>1.75</td><td>1.75</td><td>1.75</td><td>1.75</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="2">FIG. 2</td><td>y</td><td>7.5</td><td>7.5</td><td>7.25</td><td>7.25</td><td>5.25</td><td>5.25</td><td>7.25</td><td>7.25</td><td>7.5</td><td>7.5</td></tr><tr><td>\(\hat{y}\)</td><td>7.9</td><td>7.9</td><td>7.65</td><td>7.65</td><td>5.65</td><td>5.65</td><td>7.65</td><td>7.65</td><td>7.9</td><td>7.9</td></tr></table>

We observe that the output (11) of the proposed diagonal module produces embeddings that are different for the nodes of  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  in both Figs. 1, 2. Therefore, there does not exist permutation matrix  $\Pi$  such that  $\pmb{y} = \pmb{\Pi}\hat{\pmb{y}}$  and the proposed architecture is able to tell  $\mathcal{G}$  and  $\hat{\mathcal{G}}$  apart in both Figs. 1, 2. This is in stark contrast to GNNs with  $\pmb{x} = \pmb{1}$  input and the WL algorithm that fail to distinguish between these graphs (as discussed in section 3). The success of the proposed diagonal module lies in the spectral decomposition of (11):

$$
\boldsymbol {y} = \sigma \left(\sum_ {k = 0} ^ {K - 1} h _ {k} \operatorname {d i a g} \left(\sum_ {n = 1} ^ {N} \lambda_ {n} ^ {k} \boldsymbol {u} _ {n} \boldsymbol {u} _ {n} ^ {T}\right)\right) = \sigma \left(\sum_ {k = 0} ^ {K - 1} \sum_ {n = 1} ^ {N} h _ {k} \lambda_ {n} ^ {k} | \boldsymbol {u} _ {n} | ^ {2}\right). \tag {12}
$$

In simple words, the frequency response of the proposed GNN module depends on the absolute values of the graph adjacency eigenvectors. On the contrary, the modules of GNNs, as in (2), admit a different frequency representation when loaded with  $\mathbf{x} = \mathbf{1}$  input:

$$
\boldsymbol {y} _ {1} = \sigma \left(\sum_ {k = 0} ^ {K - 1} h _ {k} \boldsymbol {S} ^ {k} \mathbf {1}\right) = \sigma \left(\sum_ {k = 0} ^ {K - 1} h _ {k} \sum_ {n = 1} ^ {N} \lambda_ {n} ^ {k} \boldsymbol {u} _ {n} \boldsymbol {u} _ {n} ^ {T} \mathbf {1}\right) = \sigma \left(\sum_ {k = 0} ^ {K - 1} \sum_ {n = 1} ^ {N} h _ {k} \lambda_ {n} ^ {k} \boldsymbol {u} _ {n} \boldsymbol {u} _ {n} ^ {T} \mathbf {1}\right), \tag {13}
$$

where  $\pmb{y}_1$  denotes the output of a GNN module with  $\pmb{x} = \mathbf{1}$  input. As we can see both outputs  $\pmb{y}$ ,  $\pmb{y}_1$  are functions of the graph eigenvectors. The question that arises is which function,  $|\pmb{u}_n|$  or  $(\pmb{u}_n^T\pmb{1})\pmb{u}_n$ , results in more expressive GNNs. The naive answer is that depending on the graph, there is a trade-off between the information loss caused by  $|\pmb{u}_n|$  or  $(\pmb{u}_n^T\pmb{1})\pmb{u}_n$ . However, after adding a second layer, GNNs with diagonal modules are always more powerful than GNNs initialized by 1. This will be explained in more detail in the next section.

Remark 5.1 A closer look at equations (11) and (12), reveals further insights regarding the proposed architecture. In particular, the proposed diagonal module is constructed by the diagonal elements of

![](images/e5d1a0a1a2f2d338f925e6eed8d86a791b631d84e3c631373279094729dfc8fd.jpg)  
(a) Type-1 GNN module

![](images/8edb411493ce667b59e5329bbd2b20e60bce58d18a001e2891a2a3e0904a37a8.jpg)  
Figure 4: Proposed GNN modules  
(b) Type-2 GNN module

the graph adjacency powers and thus we study the vector:

$$
\boldsymbol {d} ^ {k} = \operatorname {d i a g} \left(\boldsymbol {S} ^ {k}\right) = \sum_ {n = 1} ^ {N} \lambda_ {n} ^ {k} \left| \boldsymbol {u} _ {n} \right| ^ {2}. \tag {14}
$$

Since  $S$  is the adjacency of the graph,  $d^{k}$  counts the number of  $k$ -length self loops of each node. For instance, when  $k = 2$ ,  $d^{k}$  indicates the degree of each node, whereas for  $k = 3$ , it counts the number of triangles each node is involved in, multiplied by a constant factor. For  $k = 4$ ,  $d^{k}$  holds information about the degrees of 1-hop and 2-hop neighbors as well as the 4-th order cycles. Similar observations are derived by considering larger values of  $k$ . Graph adjacency diagonals are not only associated with  $k$ -hop neighbor degrees but also with motifs that are present in the graph. Overall,  $d^{k}$  combines  $k$ -th order degree and subgraph information and the proposed GNN module provides additional knowledge about each node, compared to GNNs with 1 inputs. This observation becomes even more valuable, if we consider the significance of subgraph mining in graph theory (Kuramochi & Karypis, 2001; Danisch et al., 2018). Finally, the combined  $k$ -th order degree and subgraph information, provided by  $d^{k}$ , is associated with the absolute values of the graph adjacency eigenvectors  $|u_{n}|$ , whereas degrees alone are connected with  $(\mathbf{u}_n^T\mathbf{1})\mathbf{u}_n$ .

The following theorem characterizes the expressive power of GNNs with modules as in (11):

Theorem 5.2 Let  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  be non-isomorphic graphs. If Assumption 2.1 holds, there exists a GNN with diagonal modules as in (11) that tells the two graphs apart.

# 6 DESIGNING POWERFUL GNN ARCHITECTURES

After introducing and analyzing the GNN module in (11), it is time to place it in a broader perspective as part of a GNN architecture. The modules we employ to build the proposed GNN architecture are presented in Fig. 4. Regarding their functionality we provide the following result:

Proposition 6.1 A GNN designed with the diagonal modules of Fig. 4a in the input layer is equivalent to a standard GNN designed with the modules of Fig. 4b in the input layer, if the input to the modules of Fig. 4b is designed according to:

$$
\boldsymbol {X} = \left[ \operatorname {d i a g} \left(\boldsymbol {S} ^ {0}\right), \operatorname {d i a g} \left(\boldsymbol {S} ^ {1}\right), \operatorname {d i a g} \left(\boldsymbol {S} ^ {2}\right), \dots , \operatorname {d i a g} \left(\boldsymbol {S} ^ {D - 1}\right) \right]. \tag {15}
$$

The claim of Proposition 6.1 is fundamental. The diagonal GNN module in (11) is equivalent to a standard GNN module with proper input design. Furthermore, combining propositions 4.1 and 6.1 yields a direct connection between the three considered architectures; GNNs with Gaussian input and variance nonlinearity, GNNs with no input and diagonal operator, and standard GNNs with input as in (15). Guided by these findings we design the GNN architectures presented in Fig. 5. The architecture on the left uses one type of GNN blocks (type-2) and the input is designed by equation (15). Furthermore, it is a symmetric architecture and admits all the favorable properties of symmetric designs. On the other hand, the architecture on the right uses a combination of type-1 and type-2 GNN blocks and designing an input is not necessary. Although the design is not symmetric, it offers reduced number of trainable parameters and reuse of first layer features, which has been observed to benefit convolutional architectures. The expressive power of the proposed architectures is demonstrated in the following theorem:

![](images/dcd41704178e9f98940dd71557bb6c9d93e79d22abde2090cda0ba79cd30acc3.jpg)  
(a) Type-2 architecture

![](images/b178cea015f24b1e2cb80e03580190b389b204ebaff8c4572546404381aecc23.jpg)  
(b) Type-1 and type-2 architecture  
Figure 5: Proposed GNN architectures

Theorem 6.1 Let  $\mathcal{G}$ ,  $\hat{\mathcal{G}}$  be non-isomorphic graphs with graph signals  $X$ ,  $\hat{X}$  designed according to (15). If Assumption 2.1 holds, then the proposed GNNs in Fig. 5 can tell the two graphs apart.

Corollary 6.2 The proposed architectures in Fig. 5 are more expressive compared to GNNs with  $x = 1$  or  $x = S1$  inputs.

Corollary 6.2 follows from Theorem 6.1 and the fact that both  $\operatorname{diag}\left(\boldsymbol{S}^0\right) = \mathbf{1}$ ,  $\operatorname{diag}\left(\boldsymbol{S}^2\right) = \boldsymbol{S}\mathbf{1}$  are included in the proposed input  $\boldsymbol{X}$ , defined in (15).

# 7 SIMULATIONS

In this section we assess the performance of the proposed GNN architectures in the task of graph classification. In particular, we use graph isomorphism and graph classification datasets and compare against GIN initialized with  $x = 1$  (Xu et al., 2019), denoted as  $\mathrm{GIN}_1$  and GIN modified according to our proposed architectures, i.e., initialized according to equation (15), denoted as  $\mathrm{GIN}_{\mathrm{plus}}$ .

# 7.1 THE CSL DATASET

Our first experiment involves the Circular Skip Link (CSL) dataset, which was introduced in (Murphy et al., 2019) to test the expressiveness of GNNs; it is the golden standard when it comes to benchmarking GNNs for isomorphism (Dwivedi et al., 2020). CSL is a symmetric graph dataset. It contains 150 4-regular graphs, where the edges form a cycle and contain skip-links between nodes. A schematic representation of the CSL graphs can be found in Appendix K. Each graph consists of 41 nodes and 164 edges and belongs to one of 10 classes. All the nodes have degree 4 and thus  $x = 1$  is an eigenvector of every graph and orthogonal to all the remaining eigenvectors. As a result the degree vector is uninformative and so is any message passing operation of the degree.

GNNs initialized with  $x = 1$  and the WL algorithm fail to provide any essential information for this set of graphs and the classification task is completely random, as shown in Table 4. The proposed GNN architectures, on the other hand, have no issue in dealing with this dataset. In particular a single diagonal GNN module with parameters  $(h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9) = (0, 1, -\frac{1}{2}, \frac{1}{3}, -\frac{1}{4}, \frac{1}{5}, -\frac{1}{6}, \frac{1}{7}, -\frac{1}{8}, \frac{1}{9})$  and  $\sigma(\cdot)$  being the linear function, is able to classify these graphs with  $100\%$  accuracy. To see this, we present in Table 2 the output  $\mathbf{1}^T \mathbf{y}$  for every class, where  $\mathbf{y}$  is defined in (11) with the aforementioned parameters. The output is the same for each graph in the same class but different for graphs that belong to different classes. Therefore, perfect classification accuracy is achieved by passing the GNN output to a simple linear classifier or even a linear assignment algorithm.

Table 2: GNN output  $y$  for every class of the CSL graphs.  

<table><tr><td colspan="10">CLASS</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>73616</td><td>-45968</td><td>1059</td><td>-30593</td><td>-25345</td><td>-26001</td><td>-17555</td><td>-28543</td><td>16065</td><td>-21163</td></tr></table>

# 7.2 SOCIAL AND BIOLOGICAL NETWORKS

Next, we test the performance of the proposed architecture with standard social, chemical and bioinformatics graph classification datasets (Errica et al., 2019). The details of each dataset can be found in Table 3. To perform the graph classification task, we train a GNN with 4 layers, each layer consisting of the same number of neurons. The input to each GNN is designed by equation (15) with  $K = 10$  and we also pass the  $k$ -th degree vector. Apart from feeding the output of each layer to the next layer, we also apply a readout function that performs graph pooling. The graph pooling layer generates a global graph embedding from the node representations and passes it to a linear classifier. The nonlinearity is chosen to be the ReLU. An illustration of the used architecture, as well as a detailed description of the experiments, is presented in Appendix K.

To test the performance of the proposed architectures and the baseline we divide each dataset into 50 - 50 training-testing splits and perform 10-fold cross validation. We measure the micro F1 and macro F1 score for each epoch and present the epoch with the best average result among the 10 folds. The mean and standard deviation of the testing results over 10 shuffles are presented in Table 4.

Table 3: Datasets  

<table><tr><td>Dataset</td><td># Graphs</td><td>Average # Vertices</td><td>Average # Edges</td><td># Classes</td><td>Network Type</td></tr><tr><td>CSL</td><td>150</td><td>41</td><td>164</td><td>10</td><td>Circulant</td></tr><tr><td>IMDBBINARY</td><td>1,000</td><td>20</td><td>193</td><td>2</td><td>Social</td></tr><tr><td>IMDBMULTI</td><td>1,500</td><td>13</td><td>132</td><td>3</td><td>Social</td></tr><tr><td>REDDITBNNARY</td><td>2000</td><td>430</td><td>498</td><td>2</td><td>Social</td></tr><tr><td>REDDITMULTI</td><td>5000</td><td>509</td><td>595</td><td>5</td><td>Social</td></tr><tr><td>PTC</td><td>344</td><td>26</td><td>52</td><td>3</td><td>Bioinformatic</td></tr><tr><td>PROTEINS</td><td>1,113</td><td>39</td><td>146</td><td>2</td><td>Bioinformatic</td></tr><tr><td>MUTAG</td><td>188</td><td>18</td><td>20</td><td>2</td><td>Chemical</td></tr><tr><td>NCI1</td><td>4110</td><td>39</td><td>73</td><td>2</td><td>Chemical</td></tr></table>

Table 4: Average testing score and standard deviation over 10 shuffles  

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Proposed</td><td colspan="2">GIN</td><td colspan="2">GINplus (proposed+GIN)</td></tr><tr><td>micro F1</td><td>macro F1</td><td>micro F1</td><td>macro F1</td><td>micro F1</td><td>macro F1</td></tr><tr><td>CSL</td><td>100 ± 0</td><td>100 ± 0</td><td>10 ± 3.3</td><td>1.8 ± 0.6</td><td>100 ± 0</td><td>100 ± 0</td></tr><tr><td>IMDBBINARY</td><td>71.7 ± 2.5</td><td>71.3 ± 2.7</td><td>74.7 ± 3.2</td><td>74.6 ± 3.2</td><td>71.6 ± 3.4</td><td>71 ± 3.8</td></tr><tr><td>IMDBMULTI</td><td>46.1 ± 2.8</td><td>44.2 ± 3.2</td><td>50.3 ± 2.8</td><td>48 ± 3.4</td><td>48.6 ± 2.9</td><td>46.1 ± 4.2</td></tr><tr><td>REDDITBINARY</td><td>87.2 ± 4.1</td><td>87.1 ± 4.3</td><td>81.6 ± 5.6</td><td>81.5 ± 5.7</td><td>89.8 ± 2.3</td><td>89.7 ± 2.3</td></tr><tr><td>REDDITMULTI</td><td>54 ± 2.2</td><td>52.4 ± 2.1</td><td>52.4 ± 2.4</td><td>50.9 ± 2.4</td><td>55 ± 1.5</td><td>53.6 ± 1.7</td></tr><tr><td>PTC</td><td>63.6 ± 4.9</td><td>61.4 ± 6.9</td><td>65.7 ± 8.8</td><td>65.1 ± 9.1</td><td>62.5 ± 5.1</td><td>61.4 ± 5.5</td></tr><tr><td>PROTEINS</td><td>74.2 ± 4.2</td><td>73 ± 4</td><td>74 ± 4.6</td><td>72.3 ± 4.5</td><td>74.3 ± 4.8</td><td>73.1 ± 4.5</td></tr><tr><td>MUTAG</td><td>89.3 ± 7.3</td><td>87.2 ± 9.3</td><td>89.8 ± 7.6</td><td>88.6 ± 8.8</td><td>89.8 ± 8</td><td>88.7 ± 8.6</td></tr><tr><td>NCI1</td><td>74.5 ± 2.1</td><td>74.3 ± 2.1</td><td>77.2 ± 1.9</td><td>77.2 ± 1.9</td><td>76.3 ± 3.7</td><td>76.2 ± 3.8</td></tr></table>

In Table 4 we observe that the proposed architecture and  $\mathrm{GIN}_{\mathrm{plus}}$  markedly outperform  $\mathrm{GIN}_1$  in the REDDITBINARY dataset, and also show notable improvement in the REDDITMULTI dataset.  $\mathrm{GIN}_1$ , on the other hand, has a  $3\%$  advantage in the IMDBBINARY dataset, whereas in the remaining datasets the performances of the competing algorithms are statistically similar. The latter can be explained, since the vital classification components, of these datasets, are not orthogonal to  $\pmb{x} = \mathbf{1}$  and  $\mathrm{GIN}_1$  is not undergoing critical information loss. Overall, we conclude that properly designed GNNs, as the proposed and  $\mathrm{GIN}_{\mathrm{plus}}$  can not only demonstrate remarkable performance in graph classification tasks, but can also handle pathological datasets such as the CSL.

# 8 CONCLUSION

In this paper we studied the expressive power of GNNs with spectral decomposition tools. We showed that, contrary to common belief, the WL algorithm is not the real limit and proved that GNNs can distinguish between any graphs with at least one different eigenvalue. Furthermore, we explained the limitations of GNNs with all-one inputs and designed GNN architectures that overcome these limitations. Experiments with graph isomorphism and graph classification datasets demonstrated the effectiveness of the proposed architectures. With this work we move one step closer to understanding the properties of GNNs and analyzing their functionality.

# REFERENCES

Ralph Abboud, Ismail Ilkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. The surprising power of graph neural networks with random node initialization. In *IJCAI*, 2021.  
Waiss Azizian et al. Expressive power of invariant and equivariant graph neural networks. In International Conference on Learning Representations, 2020.  
Muhammet Balcilar, Pierre Héroux, Benoit Gauzere, Pascal Vasseur, Sébastien Adam, and Paul Honeine. Breaking the limits of message passing graph neural networks. In International Conference on Machine Learning, pp. 599-608. PMLR, 2021.  
Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, et al. Interaction networks for learning about objects, relations and physics. Advances in neural information processing systems, 29, 2016.  
Dominique Beaini, Saro Passaro, Vincent Létourneau, Will Hamilton, Gabriele Corso, and Pietro Lio. Directional graph networks. In International Conference on Machine Learning, pp. 748-758. PMLR, 2021.  
Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph isomorphism testing and function approximation with gnns. Advances in neural information processing systems, 32, 2019.  
Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, and Petar Velicković. Principal neighbourhood aggregation for graph nets. Advances in Neural Information Processing Systems, 33:13260-13271, 2020.  
Miles Cranmer, Peter Melchior, and Brian Nord. Unsupervised resource allocation with graph neural networks. Proceedings of Machine Learning Research, 1:1-13, June 2021. URL http://arxiv.org/abs/2106.09761.  
Maximilien Danisch, Oana Balalau, and Mauro Sozio. Listing k-cliques in sparse real-world graphs. In Proceedings of the 2018 World Wide Web Conference, pp. 589-598, 2018.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29:3844-3852, 2016.  
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.  
Federico Errica, Marco Podda, Davide Bacciu, and Alessio Micheli. A fair comparison of graph neural networks for graph classification. arXiv preprint arXiv:1912.09893, 2019.  
P. Gainza, F. Sverrisson, F. Monti, E. Rodola, D. Boscaini, M. M. Bronstein, and B. E. Correia. Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. Nature Methods, 17(2):184-192, February 2020.  
Fernando Gama, Joan Bruna, and Alejandro Ribeiro. Stability properties of graph neural networks. IEEE Transactions on Signal Processing, 68:5680-5695, 2020.  
Floris Geerts and Juan L Reutter. Expressiveness and approximation properties of graph neural networks. In International Conference on Learning Representations, 2021.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pp. 1263-1272. PMLR, 2017.  
Samar Hadou, Charilaos I Kanatsoulis, and Alejandro Ribeiro. Space-time graph neural networks. In International Conference on Learning Representations, 2021.  
Willem H Haemers and Edward Spence. Enumeration of cospectral graphs. European Journal of Combinatorics, 25(2):199-211, 2004.

Ehsan Hajiramezanali, Arman Hasanzadeh, Nick Duffield, Krishna R Narayanan, Mingyuan Zhou, and Xiaoning Qian. Variational graph recurrent neural networks. In Neural Information Processing Systems (NeurIPS), 2019.  
William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 1025-1035, 2017.  
Dejun Jiang, Zhenxing Wu, Chang Yu Hsieh, Guangyong Chen, Ben Liao, Zhe Wang, Chao Shen, Dongsheng Cao, Jian Wu, and Tingjun Hou. Could graph neural networks learn better molecular representation for drug discovery? a comparison study of descriptor-based and graph-based models. Journal of Cheminformatics, 13(1):12, dec 2021.  
Nicolas Keriven and Gabriel Peyré. Universal invariant and equivariant graph neural networks. Advances in Neural Information Processing Systems, 32, 2019.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Michihiro Kuramochi and George Karypis. Frequent subgraph discovery. In Proceedings 2001 IEEE international conference on data mining, pp. 313-320. IEEE, 2001.  
Ron Levie, Wei Huang, Lorenzo Bucci, Michael Bronstein, and Gitta Kutyniok. Transferability of spectral graph convolutional neural networks. Journal of Machine Learning Research, 22(272): 1-59, 2021.  
Yujia Li, Richard Zemel, Marc Brockschmidt, and Daniel Tarlow. Gated graph sequence neural networks. In Proceedings of ICLR'16, 2016.  
Vinicius Lima, Mark Eisen, Konstatinos Gatsis, and Alejandro Ribeiro. Resource allocation in large-scale wireless control systems with graph neural networks. IFAC-PapersOnLine, 53(2): 2634-2641, 2020.  
Xiaorui Liu, Wei Jin, Yao Ma, Yaxin Li, Hua Liu, Yiqi Wang, Ming Yan, and Jiliang Tang. Elastic graph neural networks. In International Conference on Machine Learning, pp. 6837-6849. PMLR, 2021.  
Andreas Loukas. What graph neural networks cannot learn: depth vs width. In International Conference on Learning Representations, 2019.  
Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph networks. In International Conference on Learning Representations, 2018.  
Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. Advances in neural information processing systems, 32, 2019a.  
Haggai Maron, Ethan Fetaya, Nimrod Segol, and Yaron Lipman. On the universality of invariant networks. In International conference on machine learning, pp. 4363-4371. PMLR, 2019b.  
Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: higher-order graph neural networks. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, pp. 4602-4609, 2019.  
Christopher Morris, Gaurav Rattan, and Petra Mutzel. Weisfeiler and leman go sparse: Towards scalable higher-order graph embeddings. Advances in Neural Information Processing Systems, 33: 21824-21840, 2020.  
Ryan Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Relational pooling for graph representations. In International Conference on Machine Learning, pp. 4663-4673. PMLR, 2019.

Andrei Nicolicioiu, Iulia Duta, and Marius Leordeanu. Recurrent space-time graph neural networks. Advances in Neural Information Processing Systems, 32, apr 2019.  
Luana Ruiz, Luiz Chamon, and Alejandro Ribeiro. Graphon neural networks and the transferability of graph neural networks. In Advances in Neural Information Processing Systems, volume 33, pp. 1702-1712, 2020a.  
Luana Ruiz, Fernando Gama, and Alejandro Ribeiro. Gated graph recurrent neural networks. IEEE Transactions on Signal Processing, 68:6303-6318, 2020b.  
Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random features strengthen graph neural networks. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM), pp. 333-341. SIAM, 2021.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. Computational capabilities of graph neural networks. IEEE Transactions on Neural Networks, 20 (1):81-102, 2008a.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008b.  
Youngjoo Seo, Michael Defferrard, Pierre Vandergheynst, and Xavier Bresson. Structured sequence modeling with graph convolutional recurrent networks. In Advances in Neural Information Processing Systems, pp. 362-373, 2018.  
Alexey Strokach, David Becerra, Carles Corbi-Verge, Albert Perez-Riba, and Philip M. Kim. Fast and flexible protein design using deep graph neural networks. Cell Systems, 11(4):402-411.e4, October 2020.  
Behrooz Tahmasebi, Derek Lim, and Stefanie Jegelka. Counting substructures with higher-order graph neural networks: Possibility and impossibility results. arXiv preprint arXiv:2012.03174, 2020.  
Petar Velicković, Arantxa Casanova, Pietro Lio, Guillem Cucurull, Adriana Romero, and Yoshua Bengio. Graph attention networks. In 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceedings. International Conference on Learning Representations, ICLR, 2018.  
Yanbang Wang, Pan Li, Chongyang Bai, and Jure Leskovec. Tedic: Neural modeling of behavioral patterns in dynamic social interaction networks. In Proceedings of the Web Conference 2021, WWW '21, pp. 693-705, New York, NY, USA, 2021.  
Boris Weisfeiler and Andrei Leman. The reduction of a graph to canonical form and the algebra which appears therein. NTI, Series, 2(9):12-16, 1968.  
Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. Graph neural networks in recommender systems: a survey. ACM Computing Surveys (CSUR).  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=ryGs6iA5Km.  
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 10:974-983, June 2018.
