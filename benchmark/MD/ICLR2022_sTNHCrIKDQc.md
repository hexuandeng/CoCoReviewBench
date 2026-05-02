# GRAPHON BASED CLUSTERING AND TESTING OF NETWORKS: ALGORITHMS AND THEORY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Network-valued data are encountered in a wide range of applications, and pose challenges in learning due to their complex structure and absence of vertex correspondence. Typical examples of such problems include classification or grouping of protein structures and social networks. Various methods, ranging from graph kernels to graph neural networks, have been proposed that achieve some success in graph classification problems. However, most methods have limited theoretical justification, and their applicability beyond classification remains unexplored. In this work, we propose methods for clustering multiple graphs, without vertex correspondence, that are inspired by the recent literature on estimating graphons—symmetric functions corresponding to infinite vertex limit of graphs. We propose a novel graph distance based on sorting-and-smoothing graphon estimators. Using the proposed graph distance, we present two clustering algorithms and show that they achieve state-of-the-art results. We prove the statistical consistency of both algorithms under Lipschitz assumptions on the graph degrees. We further study the applicability of the proposed distance for graph two-sample testing problems.

# 1 INTRODUCTION

Machine learning on graphs has evolved considerably over the past two decades. The traditional view towards network analysis is limited to modelling interactions among entities of interest, for instance social networks or world wide web, and learning algorithms based on graph theory have been commonly used to solve these problems (Von Luxburg, 2007; Yan et al., 2006). However, recent applications in bioinformatics and other disciplines require a different perspective, where the networks are the quantities of interest. For instance, it is of practical interest to classify protein structures as enzyme or non-enzyme (Dobson & Doig, 2003) or detect topological changes in brain networks caused by Alzheimer's disease (Stam et al., 2007). We refer to such problems as learning from network-valued data to distinguish from the traditional network analysis problems, involving a single network of interactions (Newman, 2003).

Machine learning on network-valued data has been an active area of research in recent years, although most works focus on the network classification problem. The generic approach is to convert the network-valued data into a standard representation. Graph neural networks are commonly used for network embedding, that is, finding Euclidean representations of each network that can be further used in standard machine learning models (Narayanan et al., 2017; Xu et al., 2019). In contrast, graph kernels capture similarities between pairs of networks that can be used in kernel based learning algorithms (Shervashidze et al., 2011; Kondor & Pan, 2016; Togninalli et al., 2019). In particular, the graph neural tangent kernel defines a graph kernel that corresponds to infinitely wide graph neural networks, and typically outperforms neural networks in classification tasks (Du et al., 2019). A more classical equivalent for graph kernels is to define metrics that characterise the distances between pairs of graphs (Bunke & Shearer, 1998), but there has been limited research on designing efficient and useful graph distances in the machine learning literature.

The motivation for this paper stems from two shortcomings in the literature on network-valued data analysis: first, the efficacy of existing kernels or embeddings have not been studied beyond network classification, and second is the lack of theoretical analysis of these methods, particularly in the small sample setting. Generalisation error bounds for graph kernel based learning exist (Du et al., 2019), but these bounds, based on learning theory, are meaningful only when many networks

are available. However, in many applications, one needs to learn from a small population of large networks and, in such cases, an informative statistical analysis should consider the small sample, large graph regime. To address this issue, we take inspiration from the recent statistics literature on graph two-sample testing—given two (populations of) large graphs, the goal is to decide if they are from same statistical model or not. Although most theoretical studies in graph two-sample testing focus on graph with vertex correspondence (Tang et al., 2017a; Ghoshdastidar & von Luxburg, 2018), some works address the problem of testing graphs on different vertex sets either by defining distances between graphs (Tang et al., 2017b; Agterberg et al., 2020) or by representing networks in terms of pre-specified network statistics (Ghoshdastidar et al., 2017). The use of network statistics for clustering network-valued data is studied in Mukherjee et al. (2017). Another fundamental approach for dealing with graphs of different sizes is graph matching, where the objective is to determine the vertex correspondence. Graph matching is often solved by formulating it as an optimization problem (Zaslavskiy et al., 2008; Guo et al., 2019) or defining graph edit distance between the graphs (Riesen & Bunke, 2009; Gao et al., 2010). Although, there is extensive research on graph matching, the efficacy of these methods in learning from network-valued data remains unexplored.

Contribution and organisation. In this work, we follow the approach of defining meaningful graph distances based on statistical models, and use the proposed graph distance in the context of learning from networks without vertex correspondence. In particular, we propose graph distances based on graphons. Graphons are symmetric bivariate functions that represent the limiting structure for a sequence of graphs with increasing number of nodes (Lovasz & Szegedy, 2006), but can be also viewed as a nonparametric statistical model for exchangeable random graphs (Diaconis & Janson, 2007; Bickel & Chen, 2009). The latter perspective is useful for the purpose of machine learning since it allows us to view the multiple graphs as random samples drawn from one or more graphon models. This perspective forms the basis of our contributions, which are listed below:

1) In Section 2, we propose a distance between two networks, that do not have vertex correspondence and could have different number of vertices. We view the networks as random samples from (unknown) graphons, and propose a graph distance that estimates the  $L_{2}$ -distance between the graphons. The distance is inspired by the sorting-and-smoothing graphon estimator (Chan & Airoldi, 2014).  
2) In Section 3, we present two algorithms for clustering network-valued data based on the proposed graph distance: a distance-based spectral clustering algorithm, and a similarity based semi-definite programming (SDP) approach. We derive performance guarantees for both algorithms under the assumption that the networks are sampled from graphons satisfying certain smoothness conditions.  
3) We empirically compare the performance of our algorithms with other clustering strategies based on graph kernels, graph matching, network statistics etc. and show that, on both simulated and real data, our graph distance-based spectral clustering algorithm outperforms others while the SDP approach also shows reasonable performance, and they also scale to large networks (Section 3.3).  
4) Inspired by the success of the proposed graph distance in clustering, we use the distance for graph two-sample testing. In Section 4, we show that the proposed two-sample test is statistically consistent for large graphs, and also demonstrate the efficacy of the test through numerical simulation.

We provide further discussion in Section 5 and present the proofs of theoretical results in Appendix.

# 2 GRAPH DISTANCE BASED ON GRAPHONS

Clustering or testing of multiple networks requires a notion of distance between the networks. In this section, we present a transformation that converts graphs of different sizes into a fixed size representation, and subsequently, propose a graph distance inspired by the theory of graphons. We first provide some background on graphons and graphon estimation. Graphon has been studied in the literature from two perspectives: as limiting structure for infinite sequence of growing graphs (Lovasz & Szegedy, 2006), or as exchangeable random graph model. In this paper, we follow the latter perspective. A random graph is said to be exchangeable if its distribution is invariant under permutation of nodes. Diaconis & Janson (2007) showed that any statistical model that generates exchangeable random graphs can be characterised by graphons, as introduced by Lovász & Szegedy (2006). Formally, a graphon is a symmetric measurable continuous function  $w:[0,1]^2\to [0,1]$  where  $w(x,y)$  can be interpreted as the link probability between two nodes of the graph that are assigned values  $x$  and  $y$ , respectively. This interpretation propounds the following two stage sampling

procedure for graphons. To sample a random graph  $G$  with  $n$  nodes from a graphon  $w$ , in the first stage, one samples  $n$  variables  $U_{1},\ldots ,U_{n}$  uniformly from  $[0,1]$  and constructs a latent mapping between the sampled points and the node labels. In the second stage, edges between any two nodes  $i,j$  are randomly added based on the link probability  $w(U_i,U_j)$ . Mathematically, if we abuse notation to denote the adjacency matrix by  $G\in \{0,1\}^{n\times n}$ , we have

$$
U _ {1}, \dots , U _ {n} \stackrel {{\text {i i d}}} {{\sim}} \text {U n i f o r m} [ 0, 1 ] \qquad \text {a n d} \qquad G _ {i j} | U _ {i}, U _ {j} \sim \text {B e r n o u l l i} (w (U _ {i}, U _ {j})) \text {f o r a l l} i <   j.
$$

We consider problems involving multiple networks sampled independently from the same (or different) graphons. We make the following smoothness assumptions on the graphons.

Assumption 1 (Lipschitz continuous) A graphon  $w$  is Lipschitz continuous with constant  $L$  if

$$
| w (u, v) - w (u ^ {\prime}, v ^ {\prime}) | \leq L \sqrt {(u - u ^ {\prime}) ^ {2} + (v - v ^ {\prime}) ^ {2}} \quad f o r e v e r y u, v, u ^ {\prime}, v ^ {\prime} \in [ 0, 1 ].
$$

Assumption 2 (Two-sided Lipschitz degree) A graphon  $w$  has two-sided Lipschitz degree with constants  $\lambda_1, \lambda_2 > 0$  if its degree distribution  $g$ , defined by  $g(u) = \int_0^1 w(u,v)\mathrm{d}v$ , satisfies

$$
\lambda_ {2} | u - u ^ {\prime} | \leq | g (u) - g (u ^ {\prime}) | \leq \lambda_ {1} | u - u ^ {\prime} | \quad f o r e v e r y u, u ^ {\prime} \in [ 0, 1 ].
$$

One of the challenges in graphon estimation is due to the issue of non-identifiability, that is, different graphon functions  $w$  can generate the same random graph model. In particular, two graphons  $w$  and  $w'$  generate the same random graph model if they are weakly isomorphic—there exist two measures preserving transformations  $\phi, \phi': [0,1] \to [0,1]$  such that  $w(\phi(u), \phi(v)) = w'(\phi'(u), \phi'(v))$ . Moreover, the converse also holds meaning that such transformations are known to be the only source of non-identifiability (Diaconis & Janson, 2007). This weak isomorphism induces equivalence classes on the space of graphons. Since our goal is only to cluster graphs belonging to random graph models, we simply make the following assumption on our graphons.

Assumption 3 (Equivalence classes) Any reference to  $K$  graphons,  $w_{1},\ldots ,w_{K}$ , assumes that, for every  $i,j$ , either  $w_{i} = w_{j}$  or  $w_{i}$  and  $w_{j}$  belong to different equivalence classes. Furthermore, without loss of generality, we assume that every graphon  $w_{i}$  is represented such that the corresponding degree function  $g_{i}$  is non-decreasing.

Remark on the necessity of Assumptions 1-3. Assumption 1 is standard in graphon estimation literature (Klopp et al., 2017) since it avoids graphons corresponding to inhomogeneous random graph models. It is known that two graphs from widely separated inhomogeneous models (in  $L_{2}$ -distance) are statistically indistinguishable (Ghoshdastidar et al., 2020), and hence, it is essential to ignore such models to derive meaningful guarantees. Assumption 2 ensures that, under a measure-preserving transformation, the graphon has strictly increasing degree function, which is a canonical representation of an equivalence class of graphons (Bickel & Chen, 2009). Assumption 3 is needed since graphons can only be estimated up to measure-preserving transformation. As noted above, it is inconsequential for all practical purposes but simplifies the theoretical exposition.

Graph transformation. In order to deal with multiple graphs and measure distances among pairs of graphs, we require a transformation that maps all graphs into a common metric space—the space of all  $n_0 \times n_0$  symmetric matrices for some integer  $n_0$ . While the graphon estimation literature provides several consistent estimators (Klopp et al., 2017; Zhang et al., 2017), only the histogram based sorting-and-smoothing graphon estimator of Chan & Airoldi (2014) can be adapted to meet the above requirement. We use the following graph transformation, inspired by Chan & Airoldi (2014). The adjacency matrix  $G$  of size  $n \times n$  is first reordered based on permutation  $\sigma$ , such that the empirical degree based on this permutation is monotonically increasing. The degree sorted adjacency matrix is denoted by  $G^{\sigma}$ . It is then transformed to a 'histogram'  $A \in \mathbb{R}^{n_0 \times n_0}$  given by

$$
A _ {i j} = \frac {1}{h ^ {2}} \sum_ {i _ {1} = 1} ^ {h} \sum_ {j _ {1} = 1} ^ {h} G _ {i h + i _ {1}, j h + j _ {1}} ^ {\sigma}, \text {w h e r e} h = \left\lfloor \frac {n}{n _ {0}} \right\rfloor \text {a n d} \left\lfloor \cdot \right\rfloor \text {i s t h e f l o o r f u n c t i o n}. \tag {1}
$$

Proposed graph distance. Given two graphs  $G_{1}$  and  $G_{2}$  with  $n_{1}$  and  $n_{2}$  nodes, respectively, we apply the transformation (1) to both the graphs with  $n_{0} \leq \min \{n_{1}, n_{2}\}$ . We propose to use the graph distance

$$
d \left(G _ {1}, G _ {2}\right) = \frac {1}{n _ {0}} \| A _ {1} - A _ {2} \| _ {F}, \tag {2}
$$

where  $A_{1}$  and  $A_{2}$  denote the transformed matrices and  $\| \cdot \| _F$  denotes the matrix Frobenius norm. Proposition 1 shows that, if  $G_{1}$  and  $G_{2}$  are sampled from two graphons, then the graph distance (2) consistently estimates the  $L_{2}$ -distance between the two graphons, which is defined as

$$
\left\| w _ {1} - w _ {2} \right\| _ {L _ {2}} ^ {2} = \int_ {0} ^ {1} \int_ {0} ^ {1} \left(w _ {1} (x, y) - w _ {2} (x, y)\right) ^ {2} \mathrm {d} x \mathrm {d} y. \tag {3}
$$

Proposition 1 (Graph distance is consistent) Let  $w_{1}$  and  $w_{2}$  satisfy Assumptions 1-3. Let  $G_{1}$  and  $G_{2}$  be random graphs with at least  $n$  nodes sampled from the graphons  $w_{1}$  and  $w_{2}$ , respectively. If  $n \to \infty$  and  $n_{0}$  is chosen such that  $\frac{n_{0}^{2}\log n}{n} \to 0$ , then with high probability (w.h.p.),

$$
\left| \left\| w _ {1} - w _ {2} \right\| _ {L _ {2}} - d \left(G _ {1}, G _ {2}\right) \right| = \mathcal {O} \left(\frac {1}{n _ {0}}\right). \tag {4}
$$

Proof sketch. We define a novel technique for approximating the graphon. The proof in Appendix A.1 first establishes that the approximation error is bounded using Assumption 1. Consequently, a relation between approximated graphons and transformed graphs is derived using lemmas from Chan & Airoldi (2014). Proposition 1 is subsequently proved using the above two results.  $\square$

Notation. For ease of exposition, Proposition 1 as well as main results are stated asymptotically using the standard  $\mathcal{O}(\cdot)$  and  $\Omega (\cdot)$  notations, which subsume absolute and Lipschitz constants. We use "with high probability" (w.h.p.) to state that the probability of an event converges to 1 as  $n\to \infty$

# 3 GRAPH CLUSTERING

We now present the first application of the proposed graph distance (2) in the context of clustering network-valued data. We are particularly interested in the setting where one needs to cluster a small population of large graphs, that is, minimum graph size  $n$  grows faster than the sample size  $m$ . This scenario is relevant in practice as bioinformatics or neuroscience application often deals with very few graphs (see real datasets in Section 3.3). Theoretically, this perspective complements guarantees for (graph) kernels that are applicable only in supervised setting and large sample regime,  $m \to \infty$ . In contrast, our guarantees are more conclusive for bounded  $m$  and large graph size,  $n \to \infty$ .

Strategy for clustering. Since our aim is to cluster graphs of varying sizes, we transform the graphs to a common representation of  $n_0 \times n_0$  matrices, and use the graph distance function in (2). We then use two different approaches for clustering: spectral clustering based on distances (Mukherjee et al., 2017), and similarity-based semi-definite programming (Perrot et al., 2020). We discuss the methods below, and prove statistical consistency, assuming that the graphs are sampled from graphons.

# 3.1 DISTANCE BASED SPECTRAL CLUSTERING (DSC)

Given  $m$  graphs with adjacency matrices  $G_{1},\ldots ,G_{m}$ , we propose a distance based clustering algorithm where we apply spectral clustering to an estimated distance matrix. The distance matrix  $\widehat{D}\in \mathbb{R}^{m\times m}$  is computed on all pairs of graphs using the defined estimator function (2), that is  $\widehat{D}_{ij} = d(G_i,G_j)$ . Unlike the standard Laplacian based spectral clustering, which is applicable for adjacency or similarity matrices, we use the method suggested by Mukherjee et al. (2017) that computes the  $K$  leading eigenvectors of  $\widehat{D}$  (corresponding to the  $K$  smallest eigenvalues in magnitude) and applies k-means clustering to the rows of the eigenvector matrix resulting in  $K$  number of clusters. We refer to this distance based clustering algorithm as DSC, described in Algorithm 1 of Appendix. To derive the statistical consistency of DSC, we consider the problem of clustering  $m$  random graphs of potentially different sizes, each sampled from one of  $K$  graphons. We establish the consistency in Theorem 1 by proving that the number of misclustered graphs goes to zero asymptotically (for large graphs).

Theorem 1 (Consistency of DSC) Consider  $K$  graphons satisfying Assumptions 1-3, and  $m$  random graphs  $G_{1},\ldots ,G_{m}$ , each sampled from one of the  $K$  graphons (assume there is at least one graph from each graphon). Define the distance matrix  $D\in \mathbb{R}^{m\times m}$  such that  $D_{ij} = \| w_i - w_j\|_{L_2}$  where  $w_{i}$  and  $w_{j}$  are the graphons from which  $G_{i}$  and  $G_{j}$  are generated. Let  $n$  be the size of the smallest graph, and  $\gamma$  be the  $K$ -th smallest eigenvalue value of  $D$  in magnitude. As  $n\to \infty$ , if  $n_0$  is chosen such that  $\frac{m^2n_0^2\log n}{n}\rightarrow 0$ , then DSC misclusters at most  $\mathcal{O}\left(\frac{m^3}{\gamma^2n_0^2}\right)$  graphs w.h.p.

Proof sketch. The proof, given in Appendix A.2, uses Davis-Kahan spectral perturbation theorem to bound the error in terms of  $\| \widehat{D} - D \|_F$ , which is further bounded using Proposition 1.

While the number of misclustered graphs seem to depend on  $m^3$ , we note that there is an inverse dependence on  $\gamma^2$  which has dependence on  $m$  (see Corollary 1 that illustrates it for a specific case). Moreover, our focus is on the setting where  $m = \mathcal{O}(1)$  and  $n, n_0 \to \infty$ , in which case, the error asymptotically vanishes. It is natural to wonder whether the dependence on  $m$  and  $n_0$  is tight in the above bounds. Currently, we do not know the optimal rates, but deriving this would be difficult due to the strong dependency of entries in  $\hat{D}$  and slow rate of convergence of the graph distance in Proposition 1. The presence of  $\gamma$  in the above clustering error bound makes Theorem 1 less interpretable. Hence, we also consider the specific case of  $K = 2$  (two graphons) in the following result, along with the assumption that equal number of graphs are generated from both graphons.

Corollary 1 Let  $w \neq w'$  be two graphons satisfying Assumptions 1-3, and  $m$  is a bounded even number. Assume that equal number of graphs are generated from  $w$  and  $w'$ . For any  $n_0$  and large enough constant  $C$  such that  $\| w - w' \|_{L_2} \geq C \frac{m}{n_0}$  and  $\frac{m^2 n_0^2 \log n}{n} \to 0$  as  $n \to \infty$ , the number of graphs misclustered by Algorithm 1 goes to zero w.h.p.

The corollary implies that given the observed graphs are large enough, and if the choice of  $n_0$  is relatively small,  $n_0 \ll \sqrt{n / \log n}$ , and the graphons are  $\Omega\left(\frac{1}{n_0}\right)$  apart in  $L_2$ -distance, then the clustering is consistent. Intuitively, it can be understood that if we condense large graphs to a small representation (small  $n_0$ ), then the clusters can be identified only if the models are quite dissimilar.

# 3.2 SIMILARITY BASED SEMI-DEFINITE PROGRAMMING (SSDP)

We propose another algorithm for clustering  $m$  graphs based on similarity between pairs of graphs. The pairwise similarity matrix  $\widehat{S} \in \mathbb{R}^{m \times m}$  is computed by applying Gaussian kernel on the distance between the graphs, that is  $\widehat{S}_{ij} = \exp \left(-\frac{d(G_i,G_j)}{\sigma_i\sigma_j}\right)$ , where  $\sigma_1, \ldots, \sigma_n$  are parameters. For theoretical analysis, we assume  $\sigma_{1} = \ldots = \sigma_{n}$  is fixed, but in experiments, the parameters are chosen adaptively. We use the following semi-definite program (SDP) (Yan et al., 2018; Perrot et al., 2020) to find membership of the observed graphs. Let  $X \in \mathbb{R}^{m \times m}$  be the normalised clustering matrix, that is,  $X_{ij} = 1 / |\mathcal{C}|$  if  $i$  and  $j$  belong to the same cluster  $\mathcal{C}$ , and 0 otherwise. Then, the SDP for estimating  $X$  is as follows:

$$
\max  _ {X} \operatorname {t r a c e} (\widehat {S} X) \quad \text {s . t .} X \geq 0, X \succeq 0, X \mathbf {1} = \mathbf {1}, \operatorname {t r a c e} (X) = K, \tag {5}
$$

where  $X \geq 0$ ,  $X \succeq 0$  ensure that  $X$  is a non-negative, positive semi-definite matrix, and  $\mathbf{1}$  denotes the vector of all ones. We denote the optimal  $X$  from the SDP as  $\widehat{X}$ . Once we have  $\widehat{X}$ , we apply standard spectral clustering on  $\widehat{X}$  to obtain a clustering of the graphs. We refer to this algorithm as SSDP, described in Algorithm 2 of Appendix. We present strong consistency result for SSDP below.

Theorem 2 (Consistency of SSDP) Consider  $K$  graphons,  $w_{1},\ldots ,w_{K}$ , satisfying Assumptions 1-3, and  $m$  random graphs, each sampled from one of the  $K$  graphons. Let  $n$  be the size of the smallest graph. As  $n\to \infty$ , if  $n_0$  is chosen such that  $\frac{m^2n_0^2\log n}{n}\rightarrow 0$  and  $\min_{l\neq l'}\| w_l - w_{l'}\|_{L_2} = \Omega \left(\frac{m}{n_0}\right)$ , then the number of graphs misclustered by SSDP is zero w.h.p.

Proof sketch. The proof in Appendix A.3 adapts Perrot et al. (2020, Proposition 1) to the present setting and combines it with Proposition 1 to derive the stated condition for zero error.  $\square$

Theorem 2 is slightly stronger than Theorem 1, or Corollary 1, since SSDP achieve a zero clustering error for large enough graphs. This theoretical merit of SDP over spectral clustering is known in the statistics literature. Similar to Corollary 1, the choice of  $n_0$  is important such that it does not violate the minimum  $L_{2}$ -distance condition in the theorem to ensure consistency.

Remark on the knowledge of  $K$ . Above discussions assume that the number of clusters  $K$  is known, which is not necessarily the case in practice. To tackle this issue, one can estimate  $K$  using Elbow method (Thorndike, 1953) or approach from Perrot et al. (2020), and then use it as input in our algorithms, DSC and SSDP. One can modify the SDP (5) and Theorem 2 to the case where  $K$

is adaptively estimated. However, we found the corresponding algorithm, adapted from Perrot et al. (2020), to be empirically unstable in the present context. Hence, the knowledge of  $K$  is assumed in the following experiments, which also allows the efficacy of the proposed algorithms and graph distance to be evaluated without the error induced by incorrect estimation of  $K$ .

# 3.3 EXPERIMENTAL ANALYSIS

In this section, we evaluate the performance of our algorithms DSC and SSDP, both in terms of accuracy and computational efficacy. We measure the performance of the algorithms in terms of error rate, that is, the fraction of misclustered graphs by using the source of the graphs as labels. Since clustering provides labels up to permutation, we use the Hungarian method (Kuhn, 1955) to match the labels. The performance can also be measured in terms of Adjusted Rand Index (results in Appendix C.3). We use both simulated and real datasets for evaluation and obtain all our experimental results using Tesla K80 GPU instance with 12GB memory from Google Colab.

Simulated data. We generate graphs of varied sizes from four graphons,  $W_{1}(u,v) = uv$ ,  $W_{2}(u,v) = \exp \left\{-\max (u,v)^{0.75}\right\}$ ,  $W_{3}(u,v) = \exp \left\{-0.5*(\min (u,v) + u^{0.5} + v^{0.5})\right\}$  and  $W_{4}(u,v) = |u - v|$ . The simulated graphs are dense and the graph sizes are controlled to study how algorithms scale. Their corresponding  $L_{2}$  distances between pairs of graphons is shown later in Figure 3 and the heatmap of the graphons are visualised in Figure 4 in Appendix.

Real data. We analyse the performance of algorithms using datasets from two contrasting domains: molecule datasets from Bioinformatics and network datasets from Social Networks. The Bioinformatics networks are smaller whereas the latter has relatively larger graphs. We use Proteins (Borgwardt et al., 2005), KKI (Pan et al., 2016), OHSU (Pan et al., 2016) and Peking_1 (Pan et al., 2016) datasets from Bioinformatics, and Facebook_Ct1 (Oettershagen et al., 2020), Github_Stargazers (Rozemberczki et al., 2020), Deezer_Ego_Nets (Rozemberczki et al., 2020) and Reddit_Binary (Yanardag & Vishwanathan, 2015) datasets from Social Networks. We sub-sample a few graphs from each dataset by setting a minimum number of nodes to validate the case of clustering small number of large graphs (small  $m$ , large  $n$ ). The number and size of the graphs sampled from each dataset are listed in '#graphs' and '#nodes' columns of tables in Figure 1. We evaluate the clustering performance on all combinations of the datasets for three and four clusters in both the domains separately.

Choice of  $n_0$  and  $\sigma_i$ . As noted in our algorithms DSC and SSDP,  $n_0$  is an input parameter. Theorems 1 and 2 show that the performance of both DSC and SSDP depend on the choice of  $n_0 = \mathcal{O}\big(\sqrt{n / \log n}\big)$ . In the experiments, we set  $n_0 = \sqrt{n / \log n}$  where  $n$  is the minimum number of nodes. In Appendix C.2, we use simulated data to show that the above choice of  $n_0$  is reasonable (if not the best) for both DSC and SSDP. Furthermore, the similarity matrix  $\widehat{S}$  in SSDP is computed using parameters  $\sigma_1, \ldots, \sigma_n$ . In the experiments, we set  $\sigma_i = d(G_i, G_{5nn})$  where  $G_{5nn}$  is the fifth nearest neighbour of  $G_i$ . Hence, apart from knowledge of  $K$ , our algorithms are parameter-free.

Performance comparison with existing methods. We compare our algorithms with a range of approaches for measuring similarity or distance among multiple networks. Most methods discussed below provide a kernel or distance matrix to which we apply spectral clustering to obtain the clusters:

1) Network Clustering based on Log-Moments (NCLM) is the only known clustering strategy for graphs of different sizes (Mukherjee et al., 2017). It is based on network statistics called log moments. Log moments for a graph with adjacency matrix  $A$  and number of nodes  $n$  is obtained by  $(\log(m_1(A)), \log(m_2(A)), \dots, \log(m_J(A)))$  where  $m_i(A) = \text{trace}(A/n)^i$  and  $J$  is a parameter.  
2) Wasserstein Weisfeiler-Lehman Graph Kernels (WWLGK) is a recent graph kernel that is based on the Wasserstein distance between the node feature vector distributions of two graphs proposed by Togninalli et al. (2019).  
3) Graph Neural Tangent Kernel (GNTK) is another graph kernel that describes infinitely wide graph neural networks derived by Du et al. (2019). Both WWLGK and GNTK provide state-of-the-art performance in graph classification with GNTK outperforming most graph neural networks.  
4) Network Clustering algorithm based on Maximum Mean Discrepancy (NCMMD) considers a graph metric (MMD) to cluster the graphs. MMD distance between random graphs is proposed as an efficient test statistic for random dot product graphs (Agterberg et al., 2020). We compute

![](images/928f36f270a2a124803331d781e0a322547c651a75ec516e3be68dff259f02c5.jpg)

![](images/8000b3c6b4d6fcd7995c984da3fc6230bb7798bff132e2b01cbcfd3348e62e78.jpg)

![](images/1732fe49a0e7b42f40b159371b57f549d60a62358c4b5739b8edeaabe1a93174.jpg)

<table><tr><td></td><td>Bioinformatics</td><td>#graphs</td><td>#nodes</td></tr><tr><td>B1</td><td>Proteins</td><td>7</td><td>[273, 620]</td></tr><tr><td>B2</td><td>KKI</td><td>7</td><td>[62, 90]</td></tr><tr><td>B3</td><td>OHSU</td><td>9</td><td>[140, 171]</td></tr><tr><td>B4</td><td>Peking_1</td><td>7</td><td>[86, 134]</td></tr></table>

![](images/0faa927f1ce878d4eb9909cf8cfe9628cd6a3e8575a2ac1ca05b9ac3ffd981e0.jpg)  
Figure 1: Evaluation of DSC and SSDP with other methods. (row 1) Results on simulated data. (rows 2 and 3) Results on real data from Bioinformatics and Social Networks, respectively. DSC outperforms in majority of the cases. Tables in rows 2 and 3 show details of the considered datasets.

<table><tr><td></td><td>Social Networks</td><td>#graphs</td><td>#nodes</td></tr><tr><td>S1</td><td>Facebook Ct1</td><td>10</td><td>[100, 100]</td></tr><tr><td>S2</td><td>Github Stargazers</td><td>8</td><td>[951, 957]</td></tr><tr><td>S3</td><td>Deezer Ego Nets</td><td>10</td><td>[208, 363]</td></tr><tr><td>S4</td><td>Reddit_Binary</td><td>12</td><td>[3219, 3782]</td></tr></table>

MMD between the graphs that are represented by latent finite dimensional embedding called spectral adjacency embedding with the dimension  $r$  as a parameter.

5) In Network Clustering algorithm based on Graph Matching Metric (NCGMM), we match two graphs of different sizes by appending null nodes to the small graph as described in Guo et al. (2019) and compute Frobenius norm between the matched graphs as their distance. Although both the considered graph metrics (MMD and graph matching) are for different purposes, we evaluate their efficacy in the context of clustering.

The different parameters to tune in the algorithms include  $n_0$  and  $\sigma_i$  in our algorithms DSC and SSDP,  $J$  in NCLM, number of iterations (#itr) to perform in WWLGK, number of layers (#layer) in graph neural networks for GNTK,  $r$  in NCMMD and none in NCGMM. We fix  $n_0$  in our algorithms using the theoretical bound and  $\sigma_i$  is set adaptively as discussed, whereas we tune the parameters for other algorithms by grid search over a set of values.

Evaluation on simulated data. We sample 10 graphs of varied sizes between 50 and 100 nodes from each of the four graphons in Figure 4, and evaluate the performances of all the seven algorithms. We perform the experiments by considering all combinations of three and four clusters of the chosen graphons. Based on the theoretical bound,  $n_0$  is fixed to 5 since minimum number of nodes is 50. We report the performance for  $J = 8$ ,  $r = 3$ , #itr = 1 and #layer = 2 as these produce the best results. The first row of Figure 1 shows the average performance of the algorithms computed over 5 independent runs. We observe that our algorithm DSC outperforms all the other algorithms, achieving nearly zero error in all cases, and SSDP also performs competitively by standing second or third best. The graph kernels, WWLGK and GNTK, and the graph metric based method NCGMM typically do not perform well. NCMMD either performs very well or quite poorly. We sample small graphs since otherwise GNTK cannot run due to memory requirement for dense large graphs and NCGMM has high computation time. Appendix C.4 includes evaluation of the algorithms except GNTK and NCGMM on larger graphs, where we observe similar behaviour.

Evaluation on real data. We consider all combinations of three and four clusters of both Bioinformatics and Social Networks separately, and evaluate the performance of the discussed seven algorithms. The second and third rows of Figure 1 show the performance with  $n_0 = 30$ ,  $J = 8$ ,  $r = 3$ , #itr = 1 and #layer = 2, and the upper limit of 7200 seconds (2 hours) as running time of algorithms. We observe DSC outperforms other algorithms by a large margin in majority of the combinations, while in the other combinations like {Proteins,KKI,Peking_1}, DSC performs well with a very small margin to the best performing one. Although NCLM and GNTK compare favorably in Social Networks datasets, they typically have high error rate in Bioinformatics datasets or simulated data, suggesting that they could be well suited for large networks, whereas DSC is more

versatile and suitable for all networks. The performance of SSDP is moderate on real data, but it achieves the smallest error in some cases, implying that SSDP is suited for certain types of networks.

Computation time comparison. Figure 2 shows the time (measured in seconds) taken by each algorithm for four clusters case, plotted in log scale. Similar behavior is observed in three clusters case also and the result can be found in Appendix C.5. Our algorithms, DSC and SSDP, perform competitively with respect to time as well. In addition, it scales effectively for large graphs unlike other algorithms. It is worth noting that although NCLM takes lesser time than DSC and SSDP for small graphs, it takes longer for large social networks datasets, thus favoring our methods over NCLM in terms of both accuracy and scalability. Graph matching based

![](images/02a09212cf81571ad7986198081d9b66dfcb502678bb2d98f34579208da8702d.jpg)  
Figure 2: Computation time

algorithm, NCGMM, has severe scalability issue demonstrating the inapplicability of such methods to learning problems. We also evaluate the scalability of the considered algorithms by measuring the time taken for clustering different sets of varied sized graphs from graphons  $W_{1}, W_{2}, W_{3}$  and  $W_{4}$ . Detailed discussion on the experiment is provided in Appendix C.6. The experimental results also illustrate the high scalability of DSC and SSDP compared to the other algorithms.

# 4 GRAPH TWO-SAMPLE TESTING

Inspired by the remarkable performance of the proposed graph distance (2) in clustering, we analyse the applicability of the distance for graph two-sample testing. Two-sample testing is usually studied in the large sample case  $m \to \infty$ , and several nonparametric tests are known that could also be applied to graphs. However, in the context of graphs, it is relevant to study the small sample setting, particularly  $m = 2$ , that is, the problem of deciding if two large graphs are statistically identical or not (Ghoshdastidar et al., 2020; Agterberg et al., 2020).

We consider the following formulation of the graph two-sample problem, stated under the assumption that the graphs are sampled from graphons. Given two random graphs,  $G_{1}$  sampled from some model (here, graphon  $w_{1}$ ), and  $G_{2}$  from another model  $w_{2}$ , the goal is to determine which of the following hypothesis is true:  $H_{0}: \{w_{1} = w_{2}\}$  or  $H_{a}: \{w_{1} \neq w_{2}: \| w_{1} - w_{2} \|_{L_{2}} \geq \phi\}$  for some  $\phi > 0$ . Existing works consider alternative random graph models, such as inhomogeneous Erdős-Rényi models or random dot product graph models, which are more restrictive. The condition  $\phi > 0$  is necessary if one only has access to finitely many independent samples (Ghoshdastidar et al., 2020). A two-sample test  $T$  is a binary function of the given samples such that  $T = 1$  denotes that the test rejects the null hypothesis  $H_{0}$  and  $T = 0$  implies that the test rejects the alternate hypothesis  $H_{a}$ . The goodness of a two-sample test is measured in terms of the Type-I and Type-II errors, which denote the probabilities of incorrectly rejecting the null and alternate hypotheses, respectively.

The goal of this section is to show that one can construct a test  $T$  that has arbitrarily small Type-I and Type-II errors. For this purpose, we consider the test

$$
T: \mathbb {I} \left\{d \left(G _ {1}, G _ {2}\right) \geq \xi \right\} \tag {6}
$$

for some  $\xi > 0$ , where  $\mathbb{I}\{\cdot\}$  is the indicator function and  $d(G_1, G_2)$  is the proposed graph distance for some choice of integer  $n_0$ . We state the following theoretical guarantee for the two-sample test  $T$ , where the performance is quantified in terms of Type-I and Type-II errors.

Theorem 3 Assume that the graphons  $w_{1}, w_{2}$  satisfy Assumptions 1-3, and let the graphs  $G_{1} \sim w_{1}$  and  $G_{2} \sim w_{2}$  have at least  $n$  nodes. As  $n \to \infty$ , there is a choice of  $\xi$  such that the Type-I and Type-II errors of the test  $T$  in (6) go to 0 if  $\frac{n_0^2 \log n}{n} \to 0$  and  $\phi \geq \frac{C}{n_0}$ , where the constant  $C$  depends only on the Lipschitz constants.

Theorem 3 shows that the test  $T$  in (6) can distinguish between any pair of graphons that have separation  $\| w_{1} - w_{2}\|_{L_{2}} = \Omega (1 / n_{0})$  with arbitrarily small error, if the graphs are large enough.

Empirical analysis. We empirically validate the consistency result in Theorem 3 by computing power of the proposed two-sample test  $T$ , which measures the probability of rejecting the null hypothesis  $H_0$ . Intuitively, power of the test for graphs sampled from same graphons should be small

![](images/190ebfa2b4e3e4f19c392ac9212e1e55f7697dee3d7dd72deefe87210de25cf2.jpg)  
Figure 3: (left)  $L_{2}$  distance between the graphons  $W_{1}, W_{2}, W_{3}$  and  $W_{4}$ . (other plots) Average power of the test (6) for graph pairs of varying sizes, sampled from every pair of graphons.

![](images/6d4e037d77509bc67fc4ff5046e7728540d9ed9baeab0c25330d68d95e222f30.jpg)

![](images/d069c282a0e6ea3aba1eb549944d280a375ceb1d3145c6b1759a83776f9ecd85.jpg)

![](images/f1befc8eae17deaa5ccaa3411b1718435fa92c9cf17b3959a0bb5bb8db0c9f61.jpg)

(close to a pre-specified significance level) since  $H_0$  must not be rejected, whereas, it should be close to 1 for graphs sampled from different graphons. As known in the testing literature, theoretical threshold,  $\xi$  in (6), is typically conservative in practice and the rejection/acceptance is decided based on  $p$ -values, computed using bootstrap samples. To this end, we follow the bootstrapping strategy in Ghoshdastidar & von Luxburg (2018, Boot-ASE algorithm). We perform the experiment by sampling two graphs  $G_1 \sim w_1$  and  $G_2 \sim w_2$  of size  $n$  and  $2n$ , respectively, where  $w_1$  and  $w_2$  are chosen from the graphons  $W_1, W_2, W_3, W_4$  discussed in Section 3.3. We consider  $n = \{50, 100, 150\}$  and fix  $n_0 = 10$  for evaluating the test  $T$ . The power is computed using the test  $T$  for the significance level 0.05, and the plots in Figure 3 show the average power computed over 500 trials of bootstrapping 100 samples generated from all pairs of graphons. From the result for graph sizes (50, 100), we observe that the graphon pair  $(W_2, W_3)$  is not easily distinguishable (low  $H_0$  rejection probability), which can be explained by their respective  $L_2$  distance that is shown in the left plot of Figure 3. This issue does not arise in testing larger graphs as the result shows for graph sizes (100, 200) and (150, 300). Therefore, the test can distinguish between pairs of graphons that are quite close provided that the observed graphs are sufficiently large, thus proving to be consistent.

# 5 CONCLUSION

There has been significant progress in learning on complex data, including network-valued data. However, much of the theoretical and algorithmic development have been in large sample problems, where one has access to  $m \to \infty$  independent samples. Practical applications of network-valued data analysis often leads to small sample, large graph problems—a setting where the machine learning literature is quite limited. Inspired by graph limits and high-dimensional statistics, this paper proposes a simple graph distance (2) based on non-parametric graph models (graphons).

Sections 3-4 demonstrate that the proposed graph distance leads to provable and practically effective algorithms for clustering (DSC and SSDP) as well as two-sample testing (6). Extensive empirical studies on simulated and real data show that the clustering based on the graph distance (2) outperforms methods based on more complex graph similarities or metrics, both in terms of accuracy and scalability. Figures 1-2 show that DSC achieves best performance for both small dense graphs (simulated graphons) as well as large sparse graphs (social networks). On the other hand, popular machine learning approaches—graph kernels or graph matching—can be computationally expensive in large graphs and their performance may not improve as  $n \to \infty$ , see WWLGK in Figure 7.

Statistical approaches, such as the proposed clustering algorithms and two-sample test, show better performance on large graphs (Figures 1, 3 and Appendix C.4). Theorems 1-3 theoretically support this observation by showing consistency of the clustering and testing methods in the limit of  $n \to \infty$ . The theoretical results, however, hinge on Assumptions 1-3. We remark that such smoothness and equivalence assumptions could be necessary for meaningful non-parametric approaches, which is also supported by the graph testing and graphon estimation literature. Further insights about the necessity of smoothness assumptions would aid in theoretical and algorithmic development.

The poor performance of graph kernels and graph matching in clustering and small sample problems calls for further studies on these methods, which have shown success in network classification. Fundamental research, combining graphon based approaches and kernels, could lead to improved techniques. Algorithmic modifications, such as estimation of  $K$ , would be also useful in practice.

# 6 ETHICS STATEMENT

The proposed clustering algorithms for network-valued data can naturally be evaluated on real-world data, but no social interpretation about the results will be drawn. Moreover, development of fair clustering algorithms is not the focus of this project.

# 7 REPRODUCIBILITY STATEMENT

The assumptions for the theory are stated clearly in Assumptions 1-3 and all the theoretical results, Proposition 1, Theorems 1-3, Corollary 1, are proved in detail in Appendix A. The implementation of the considered algorithms are provided in graph_clustering.zip with the filename as the corresponding algorithm. Datasets used in the experiments are public and the links are provided in downloadDatasets function of utils.py. The experimental results can be reproduced by following graph_clustering.ipynb.

# REFERENCES

Joshua Agterberg, Minh Tang, and Carey Priebe. Nonparametric two-sample hypothesis testing for random graphs with negative and repeated eigenvalues. arXiv preprint arXiv:2012.09828, 2020.  
Peter J Bickel and Aiyou Chen. A nonparametric view of network models and newman-girvan and other modularities. Proceedings of the National Academy of Sciences, 106(50):21068-21073, 2009.  
Karsten M Borgwardt, Cheng Soon Ong, Stefan Schonauer, SVN Vishwanathan, Alex J Smola, and Hans-Peter Kriegel. Protein function prediction via graph kernels. Bioinformatics, 21(suppl_1): i47-i56, 2005.  
Horst Bunke and Kim Shearer. A graph distance metric based on the maximal common subgraph. Pattern recognition letters, 19(3-4):255-259, 1998.  
Stanley Chan and Edoardo Airoldi. A consistent histogram estimator for exchangeable graph models. In International Conference on Machine Learning, pp. 208-216, 2014.  
Persi Diaconis and Svante Janson. Graph limits and exchangeable random graphs. arXiv preprint arXiv:0712.2749, 2007.  
Paul D Dobson and Andrew J Doig. Distinguishing enzyme structures from non-enzymes without alignments. Journal of molecular biology, 330(4):771-783, 2003.  
Simon S Du, Kangcheng Hou, Russ R Salakhutdinov, Barnabas Poczos, Ruosong Wang, and Keyulu Xu. Graph neural tangent kernel: Fusing graph neural networks with graph kernels. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.  
Xinbo Gao, Bing Xiao, Dacheng Tao, and Xuelong Li. A survey of graph edit distance. Pattern Analysis and applications, 13(1):113-129, 2010.  
Debarghya Ghoshdastidar and Ulrike von Luxburg. Practical methods for graph two-sample testing. In Advances in Neural Information Processing Systems, pp. 3019-3028, 2018.  
Debarghya Ghoshdastidar, Maurilio Gutzeit, Alexandra Carpentier, and Ulrike von Luxburg. Two-sample tests for large random graphs using network statistics. In Conference on Learning Theory, pp. 954-977, 2017.  
Debarghya Ghoshdastidar, Maurilio Gutzeit, Alexandra Carpentier, Ulrike von Luxburg, et al. Two-sample hypothesis testing for inhomogeneous random graphs. Annals of Statistics, 48(4):2208-2229, 2020.  
Xiaoyang Guo, Anuj Srivastava, and Sudeep Sarkar. A quotient space formulation for generative statistical analysis of graphical data. arXiv preprint arXiv:1909.12907, 2019.  
Olga Klopp, Alexandre B Tsybakov, Nicolas Verzelen, et al. Oracle inequalities for network models and sparse graphon estimation. The Annals of Statistics, 45(1):316-354, 2017.  
Risi Kondor and Horace Pan. The multiscale laplacian graph kernel. In Advances in Neural Information Processing Systems, pp. 2990-2998, 2016.  
Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83-97, 1955.  
László Lovász and Balázs Szegedy. Limits of dense graph sequences. Journal of Combinatorial Theory, Series B, 96(6):933-957, 2006.  
Soumendu Sundar Mukherjee, Purnamrita Sarkar, and Lizhen Lin. On clustering network-valued data. In Advances in neural information processing systems, pp. 7071-7081, 2017.  
Annamalai Narayanan, Mahinthan Chandramohan, Rajasekar Venkatesan, Lihui Chen, Yang Liu, and Shantanu Jaiswal. graph2vec: Learning distributed representations of graphs. arXiv preprint arXiv:1707.05005, 2017.

Mark EJ Newman. The structure and function of complex networks. SIAM review, 45(2):167-256, 2003.  
Lutz Oettershagen, Nils M Kriege, Christopher Morris, and Petra Mutzel. Temporal graph kernels for classifying dissemination processes. In Proceedings of the 2020 SIAM International Conference on Data Mining, pp. 496-504. SIAM, 2020.  
Shirui Pan, Jia Wu, Xingquan Zhu, Guodong Long, and Chengqi Zhang. Task sensitive feature exploration and learning for multitask graph classification. IEEE transactions on cybernetics, 47 (3):744-758, 2016.  
Michaël Perrot, Pascal Esser, and Debarghya Ghoshdastidar. Near-optimal comparison based clustering. In Advances in Neural Information Processing Systems (NeurIPS), 2020.  
Kaspar Riesen and Horst Bunke. Approximate graph edit distance computation by means of bipartite graph matching. Image and Vision computing, 27(7):950-959, 2009.  
Benedek Rozemberczki, Oliver Kiss, and Rik Sarkar. Karate club: An api oriented open-source python framework for unsupervised learning on graphs. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pp. 3125-3132, 2020.  
Nino Shervashidze, Pascal Schweitzer, Erik Jan Van Leeuwen, Kurt Mehlhorn, and Karsten M Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(9), 2011.  
Cornelis J Stam, BF Jones, G Nolte, M Breakspear, and Ph Scheltens. Small-world networks and functional connectivity in alzheimer's disease. Cerebral cortex, 17(1):92-99, 2007.  
Minh Tang, Avanti Athreya, Daniel L Sussman, Vince Lyzinski, Youngser Park, and Carey E Priebe. A semiparametric two-sample hypothesis testing problem for random graphs. Journal of Computational and Graphical Statistics, 26(2):344-354, 2017a.  
Minh Tang, Avanti Athreya, Daniel L Sussman, Vince Lyzinski, Carey E Priebe, et al. A nonparametric two-sample hypothesis testing problem for random graphs. Bernoulli, 23(3):1599-1630, 2017b.  
Robert L Thorndike. Who belongs in the family? Psychometrika, 18(4):267-276, 1953.  
Matteo Togninalli, Elisabetta Ghisu, Felipe Llinares-López, Bastian Rieck, and Karsten Borgwardt. Wasserstein weisfeiler-lehman graph kernels. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.  
Ulrike Von Luxburg. A tutorial on spectral clustering. Statistics and computing, 17(4):395-416, 2007.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In 7th International Conference on Learning Representations, ICLR. OpenReview.net, 2019.  
Bowei Yan, Purnamrita Sarkar, and Xiuyuan Cheng. Provable estimation of the number of blocks in block models. In International Conference on Artificial Intelligence and Statistics, pp. 1185-1194, 2018.  
Shuicheng Yan, Dong Xu, Benyu Zhang, Hong-Jiang Zhang, Qiang Yang, and Stephen Lin. Graph embedding and extensions: A general framework for dimensionality reduction. IEEE transactions on pattern analysis and machine intelligence, 29(1):40-51, 2006.  
Pinar Yanardag and SVN Vishwanathan. Deep graph kernels. In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1365-1374, 2015.  
Mikhail Zaslavskiy, Francis Bach, and Jean-Philippe Vert. A path following algorithm for the graph matching problem. IEEE Transactions on Pattern Analysis and Machine Intelligence, 31(12): 2227-2242, 2008.  
Yuan Zhang, Elizaveta Levina, and Ji Zhu. Estimating network edge probabilities by neighbourhood smoothing. Biometrika, 104(4):771-783, 2017.
