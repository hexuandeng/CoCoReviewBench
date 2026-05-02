# HENCLER: NODE CLUSTERING IN HETEROPHIOUS GRAPHS VIA LEARNED ASYMMETRIC SIMILARITY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Clustering nodes in heterophilous graphs is challenging as traditional methods assume that effective clustering is characterized by high intra-cluster and low inter-cluster connectivity. To address this, we introduce HeNCler—a novel approach for Heterophilous Node Clustering. HeNCler learns a similarity graph by optimizing a clustering-specific objective based on weighted kernel singular value decomposition. Our approach enables spectral clustering on an asymmetric similarity graph, providing flexibility for both directed and undirected graphs. By solving the primal problem directly, our method overcomes the computational difficulties of traditional adjacency partitioning-based approaches. Experimental results show that HeNCler significantly improves node clustering performance in heterophilous graph settings, highlighting the advantage of its asymmetric graph-learning framework.

# 1 INTRODUCTION

Graph neural networks (GNNs) have substantially advanced machine learning applications to graph-structured data by effectively propagating node attributes end-to-end. Typically, GNNs rely on the assumption of homophily, where nodes with similar labels are more likely to be connected (Zheng et al., 2024; Wu et al., 2021). The homophily assumption holds true in contexts such as social networks and citation graphs, where models like GCN (Kipf & Welling, 2017), GIN (Xu et al., 2019), and GraphSAGE (Hamilton et al., 2017) excel at tasks like node classification and graph prediction.

However, in heterophilous datasets, such as web page and transaction networks, edges often link nodes with differing labels. Models like GAT (Velicković et al., 2018) and various graph transformers (Ying et al., 2022; Dwivedi & Bresson, 2021) have demonstrated improved performance on these datasets. Their attention mechanisms learning edge importance provide a straightforward way to reduce the reliance on homophily for supervised tasks.

Our work specifically addresses unsupervised attributed node clustering tasks. Such tasks necessitate entirely unsupervised or self-supervised learning approaches. For instance, auto-encoder type models (Park et al., 2019; Pan et al., 2020) are primarily focused on node representation learning rather than clustering, making them less suited for directly improving cluster-ability. Various self-supervised, contrastive learning techniques (Hassani & Ahmadi, 2020; You et al., 2020) enhance node representation learning in homophilous settings only and lack a specific clustering objective. At the same time, several self-supervised methods have been developed to handle heterophilous graphs (Chen et al., 2022; Xiao et al., 2022; Yuan et al., 2023). For example, MUSE (Yuan et al., 2023) extracts semantic and contextual views for contrastive learning. However, these methods are designed for the general node representation learning task and lack a clustering objective.

In contrast,  $\mathrm{S}^3\mathrm{GC}$  (Devvrit et al., 2022) employs a self-supervised approach specifically designed for clustering. It however assumes homophily by leveraging random walk co-occurrences to infer proximity-based similarities. MinCutPool (Bianchi et al., 2020) and DMoN (Tsitsulin et al., 2023) introduce unsupervised losses linked to graph structure, with strong theoretical ties to spectral clustering and graph modularity, respectively. These methods are suited for undirected graphs only, and moreover rely on partitioning the adjacency matrix where effective clustering correlates with high intra-cluster and low inter-cluster similarity—a premise often invalid in heterophilous graphs.

This paper introduces HeNCler, a novel approach for node clustering in heterophilous graphs, illustrated in Figure 1. Existing works overlook the asymmetric relationships in heterophilous

Table 1: Qualitative comparison of HeNCler with several baselines. In the table,  $|\mathcal{V}|$ ,  $|\mathcal{B}|$ , and  $|\mathcal{E}|$  denote the total number of nodes, the mini-batch size, and the number of edges respectively.  

<table><tr><td rowspan="2"></td><td colspan="4">BASELINES</td><td>OURS</td></tr><tr><td>MINCUTP.</td><td>DMON</td><td>S3GC</td><td>MUSE</td><td>HENCLER</td></tr><tr><td>CAN HANDLE HETEROPHILY</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>✓</td></tr><tr><td>DIRECTED GRAPHS</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>HAS CLUSTERING OBJECTIVE</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>✓</td></tr><tr><td>SPACE COMPLEXITY</td><td>O(|V|2)</td><td>O(|V| + |E|)</td><td>O(|B|)</td><td>O(|V| + |E|)</td><td>O(|B|)</td></tr><tr><td>TIME COMPLEXITY</td><td>O(|V| + |E|)</td><td>O(|V| + |E|)</td><td>O(|V|)</td><td>O(|V| + |E|)</td><td>O(|V|)</td></tr></table>

![](images/6a06a2cbc475d85c7483876b81f84a04dcafdf2d51785c918cf05fbaa40f111d.jpg)  
Figure 1: HeNCler Overview. Starting from a heterophilous graph, where nodes with the same label are not close to each other (left), HeNCler learns two sets of node representations,  $\{\phi(\boldsymbol{x}_v)\}_{v\in \mathcal{V}}$  and  $\{\psi(\boldsymbol{x}_v)\}_{v\in \mathcal{V}}$ , forming a bipartite graph  $S$  (middle), where the similarity between nodes is defined as  $S_{uv} = \mathrm{sim}(u,v) = \phi(\boldsymbol{x}_u)^{\top}\psi(\boldsymbol{x}_v)$ . Due to the clustering objective, nodes that should belong to the same cluster are positioned closer together in the learned graph. These clusters are then identified using spectral biclustering through wKSVD (right).

graphs, as shown in Table 1. HeNCler addresses this by using weighted kernel singular value decomposition (wKSVD) to induce a learned asymmetric similarity graph for both directed and undirected graphs. The dual problem of wKSVD aligns with asymmetric kernel spectral clustering, enabling the interpretation of similarities without homophily. By solving the primal problem directly, HeNCler overcomes computational difficulties and shows superior performance in node clustering tasks within heterophilous graphs.

Contributions: Our contributions in this work can be summarized as follows:

- We introduce HeNCler, a kernel spectral biclustering framework designed to learn an induced asymmetric similarity graph suited for node clustering of heterophilous graphs, applicable to both directed and undirected graphs.  
- We develop a primal-dual framework for a generic weighted kernel singular value decomposition (wKSVD) model.  
- We show that the dual wKSVD formulation allows for biclustering of bipartite/asymmetric graphs, while we employ a computationally feasible implementation in the primal wKSVD formulation.  
- We further generalize our approach with trainable feature mappings, using node and edge decoders, such that the similarity matrix to cluster is learned.  
- We train HeNCler in the primal setting and demonstrate its superior performance on the node clustering task for heterophilous attributed graphs. Our implementation is available in supplementary materials.

# 2 PRELIMINARIES AND RELATED WORK

We use lowercase symbols (e.g.,  $x$ ) for scalars, lowercase bold (e.g.,  $\pmb{x}$ ) for vectors and uppercase bold (e.g.,  $\pmb{X}$ ) for matrices. A single entry of a matrix is represented by  $X_{ij}$ .  $\phi(\cdot)$  denotes a mapping and  $\phi_v = \phi(\pmb{x}_v)$  represents the mapping of node  $v$  in the induced feature space. We represent a graph  $\mathcal{G}$  by its vertices (i.e., nodes)  $\mathcal{V}$  and edges  $\mathcal{E}$ ,  $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ , or by its node feature matrix and adjacency matrix  $\mathcal{G} = (\pmb{X}, \pmb{A})$ . For a bipartite graph, we have  $\mathcal{G} = (\mathcal{I}, \mathcal{J}, \mathcal{E})$  or  $\mathcal{G} = (\pmb{X}_{\mathcal{I}}, \pmb{X}_{\mathcal{J}}, \pmb{S})$  where  $S_{ij}$  is the edge weight between nodes  $i \in \mathcal{I}$  and  $j \in \mathcal{J}$ . Note that  $\pmb{S}$  is generally asymmetric and rectangular, and that the adjacency matrix of the bipartite graph is given by  $\pmb{A} = \begin{bmatrix} 0 & S \\ S^{\top} & 0 \end{bmatrix}$ .

Kernel singular value decomposition KSVD (Suykens, 2016) sets up a primal-dual framework, based on Lagrange duality, that formulates a variational principle in the primal formulation that corresponds to the matrix singular value decomposition (SVD) in the dual for linear feature maps. By employing non-linear feature mappings or asymmetric kernel functions, this framework allows for nonlinear extensions of the SVD problem. The KSVD framework can be applied on data structures such as row and column features, directed graphs, and/or can exploit asymmetric similarity information such as conditional probabilities (He et al., 2023). Interestingly, KSVD often outperforms the similar though symmetric kernel principal component analysis model on tasks where the asymmetry is not immediately apparent (Tao et al., 2024). A different connection is shown in Primal-Attention (Chen et al., 2023), where the authors demonstrate the relation between canonical self-attention, which is asymmetric, and KSVD. They show how to gain computational efficiency by considering a primal equivalent of the attention mechanism.

Spectral clustering generalizations have been proposed in many settings. Spectral graph biclustering (Dhillon, 2001) formulates the spectral clustering problem of a bipartite graph  $\mathcal{G} = (\mathcal{I},\mathcal{J},\mathcal{S})$  and shows the equivalence with the SVD of the normalized matrix  $S_{n} = D_{1}^{-1 / 2}SD_{2}^{-1 / 2}$ , where  $D_{1,ii} = \sum_{j}S_{ij}$  and  $D_{2,jj} = \sum_{i}S_{ij}$ . Cluster assignments for nodes  $\mathcal{I}$  and nodes  $\mathcal{J}$  can be inferred from the left and right singular vectors respectively. Further, kernel spectral clustering (KSC) (Alzate & Suykens, 2010) proposes a weighted kernel principal component analysis in which the dual formulation corresponds to the random walks interpretation of the spectral clustering problem. KSC and the aforementioned spectral biclustering formulation lack asymmetry and a primal formulation respectively, which are limitations that our model will address.

Restricted kernel machines (RKM) (Suykens, 2017) possess primal and dual model formulations, based on the concept of conjugate feature duality. It is an energy-based framework for (deep) kernel machines, that shows relations with least-squares support vector machines (Suykens et al., 2002) and restricted Boltzmann machines (Salakhutdinov, 2015). The RKM framework encompasses many model classes, including classification, regression, kernel principal component analysis and KSVD, and allows for deep kernel learning (Tonin et al., 2021) and deep kernel learning on graphs (Achten et al., 2024). One possibility to represent the feature maps in RKMs is by means of deep neural networks, e.g., for unsupervised representation learning (Pandey et al., 2021; 2022). RKM models can work in either primal or dual setting, and with decomposition or gradient based algorithms (Achten et al., 2023).

Homophilous node clustering methods like MinCutPool (Bianchi et al., 2020) and DMoN (Tsitsulin et al., 2023) introduce unsupervised loss functions within a graph neural network framework. MinCutPool employs a relaxed version of the minimal cut loss applied to the adjacency matrix, while DMoN optimizes the modularity score of clustering labels with respect to the adjacency structure. Both of these methods rely on partitioning the adjacency matrix and inherently assume homophily. Additionally, due to their theoretical underpinnings, these losses are only applicable to undirected graphs. Beyond these adjacency partitioning-based approaches, self-supervised or contrastive methods have also been proposed (You et al., 2020; Hassani & Ahmadi, 2020; Devvrit et al., 2022). These methods typically use graph proximity as their supervision signal, which similarly assumes homophily. For example,  $\mathrm{S}^3\mathrm{GC}$  (Devvrit et al., 2022) employs a self-supervised loss based on random walk co-occurrences.

Heterophilous node clustering methods typically rely on self-supervised or contrastive techniques. Gong et al. (Gong et al., 2023) propose Sparse Graph Anomaly Detection (SparseGAD), a method that sparsifies graph structures to effectively reduce noise from irrelevant edges and enhance the detection of closely related nodes. This technique reveals underlying node dependencies, accommodating both homophilous and heterophilous relationships. Similarly, HGRL (Chen et al., 2022) employs

self-supervised learning on heterophilous graphs by utilizing graph augmentation techniques to capture global and higher-order structural information. MUSE (Yuan et al., 2023), on the other hand, constructs semantic and contextual views to capture both node-level and neighborhood information for contrastive learning, subsequently integrating these multi-view representations through a fusion controller.

While adjacency partitioning-based methods have demonstrated both theoretical and empirical success for homophilous graphs, they have not been effectively extended to heterophilous graph learning. On the other hand, self-supervised clustering approaches, though promising, often lack a clear clustering interpretation. In the following section, we introduce HeNCler, which bridges these gaps.

# 3 METHOD

Model motivation Our approach employs an RKM auto-encoder framework, which has been shown to be effective in unsupervised representation learning by jointly optimizing feature mappings and projection matrices within a kernel-based setting (Pandey et al., 2022). To capture long-range relational dependencies in heterophilous graphs, we utilize a KSVD loss, where a double feature mapping yields a learned asymmetric similarity matrix. To further enhance the cluster-ability of this matrix, we extend the loss function to a weighted KSVD (wKSVD) loss, which not only boosts clustering performance but also offers a spectral graph biclustering interpretation. We next introduce a general wKSVD framework, after which we introduce our HeNCLer model that operates in the primal setting while jointly learning the feature mappings end-to-end.

# 3.1 KERNEL SPECTRAL BICLUSTERING WITH ASYMMETRIC SIMILARITIES

Consider a dataset with two, possibly different, input sources  $\{\pmb{x}_i\}_{i=1}^n$  and  $\{\pmb{z}_j\}_{j=1}^m$ , on which we want to define an unsupervised learning task. To this end, we introduce a weighted kernel singular value decomposition model (wKSVD), starting from the following primal optimization problem, which is a weighted variant of the KSVD formulation:

$$
\min  _ {\boldsymbol {U}, \boldsymbol {V}, \boldsymbol {e}, \boldsymbol {r}} J \triangleq \operatorname {T r} (\boldsymbol {U} ^ {\top} \boldsymbol {V}) - \frac {1}{2} \sum_ {i = 1} ^ {n} w _ {1, i} \boldsymbol {e} _ {i} ^ {\top} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {e} _ {i} - \frac {1}{2} \sum_ {j = 1} ^ {m} w _ {2, j} \boldsymbol {r} _ {j} ^ {\top} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {r} _ {j}
$$

$$
\text {s . t .} \left\{\boldsymbol {e} _ {i} = \boldsymbol {U} ^ {\top} \phi \left(\boldsymbol {x} _ {i}\right), \forall i = 1, \dots , n; \quad \boldsymbol {r} _ {j} = \boldsymbol {V} ^ {\top} \psi \left(\boldsymbol {z} _ {j}\right), \forall j = 1, \dots , m \right\}, \tag {1}
$$

with projection matrices  $\pmb{U}$ ,  $\pmb{V} \in \mathbb{R}^{d_f \times s}$ ; strictly positive weighting scalars  $w_{1,i}, w_{2,j}$ ; latent variables  $\pmb{e}_i, \pmb{r}_j \in \mathbb{R}^s$ ; diagonal and positive definite hyperparameter matrix  $\pmb{\Sigma} \in \mathbb{R}^{s \times s}$ ; and centered feature maps  $\phi(\cdot): \mathbb{R}^{d_x} \mapsto \mathbb{R}^{d_f}$  and  $\psi(\cdot): \mathbb{R}^{d_z} \mapsto \mathbb{R}^{d_f}$ ; details on centering of the feature maps are provided in Appendix A. The following derivation shows the equivalence with the spectral biclustering problem.

Proposition 1. The solution to the primal problem (1) can be obtained by solving the singular value decomposition of

$$
\boldsymbol {W} _ {1} ^ {1 / 2} \boldsymbol {S} \boldsymbol {W} _ {2} ^ {1 / 2} = \boldsymbol {H} _ {e} \boldsymbol {\Sigma} \boldsymbol {H} _ {r} ^ {\top}, \tag {2}
$$

where  $\mathbf{W}_1$  and  $\mathbf{W}_2$  are diagonal matrices such that  $W_{1,ii} = w_{1,i}$  and  $W_{2,jj} = w_{2,j}$ ,  $S = \Phi \Psi^\top$  is an asymmetric similarity matrix where  $S_{ij} = \phi(\boldsymbol{x}_i)^\top \psi(\boldsymbol{z}_j)$ ,  $\Phi = [\phi(\boldsymbol{x}_1) \ldots \phi(\boldsymbol{x}_n)]^\top$ ,  $\Psi = [\psi(\boldsymbol{z}_1) \ldots \psi(\boldsymbol{z}_m)]^\top$ , and where  $\mathbf{H}_e = [\mathbf{h}_{\boldsymbol{e}_1} \ldots \mathbf{h}_{\boldsymbol{e}_n}]^\top$ , and  $\mathbf{H}_r = [\mathbf{h}_{\boldsymbol{r}_1} \ldots \mathbf{h}_{\boldsymbol{r}_m}]^\top$  are the left and right singular vectors respectively; and by applying  $r_j = \Sigma h_{\boldsymbol{r}_j} / \sqrt{w_{2,j}}$  and  $e_i = \Sigma h_{e_i} / \sqrt{w_{1,i}}$ .

Proof. We now introduce dual variables  $h_{e_i}$  and  $h_{r_j}$  using a case of Fenchel-Young inequality (Rockafellar, 1974):

$$
\frac {1}{2} w _ {1, i} \boldsymbol {e} _ {i} ^ {\top} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {e} _ {i} + \frac {1}{2} \boldsymbol {h} _ {\boldsymbol {e} _ {i}} ^ {\top} \boldsymbol {\Sigma} \boldsymbol {h} _ {\boldsymbol {e} _ {i}} \geq \sqrt {w _ {1 , i}} \boldsymbol {e} _ {i} ^ {\top} \boldsymbol {h} _ {\boldsymbol {e} _ {i}}, \quad \frac {1}{2} w _ {2, j} \boldsymbol {r} _ {j} ^ {\top} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {r} _ {j} + \frac {1}{2} \boldsymbol {h} _ {\boldsymbol {r} _ {j}} ^ {\top} \boldsymbol {\Sigma} \boldsymbol {h} _ {\boldsymbol {r} _ {j}} \geq \sqrt {w _ {2 , j}} \boldsymbol {r} _ {j} ^ {\top} \boldsymbol {h} _ {\boldsymbol {r} _ {j}}, \tag {3}
$$

$\forall \pmb{e}_i, \pmb{r}_j, \pmb{h}_{\pmb{e}_i}, \pmb{h}_{\pmb{r}_j} \in \mathbb{R}^s, \forall w_{1,i}, w_{2,j} \in \mathbb{R}_{>0}, \forall \pmb{\Sigma} \in \mathbb{R}_{>0}^{s \times s}$ . The above inequalities can be verified by writing it in quadratic form:  $\frac{1}{2} [e_i^\top h_{e_i}^\top] \begin{bmatrix} w_{1,i} \pmb{\Sigma}^{-1} & -\sqrt{w_{1,i}} \pmb{I}_s \\ -\sqrt{w_{1,i}} \pmb{I}_s & \pmb{\Sigma} \end{bmatrix} [e_i h_{e_i}] \geq 0, \forall i$ , with  $I_s$  the  $s$ -dimensional identity matrix, which follows immediately from the Schur complement form:

![](images/59cbb8f165c53c92856ea7616a5432af2ef9b877ca1832ed3a4c87269de2c4e1.jpg)  
Figure 2: The HeNCler model. HeNCler operates in the primal setting (top of the figure in red) and uses a double multilayer perceptron (MLP) to map node representations to a feature space. The obtained representations  $\phi_v$  and  $\psi_v$  are then projected to latent representations  $e_v$  and  $r_v$  respectively. The wKSVD loss ensures that these latent representations correspond to the dual equivalent (bottom of the figure in blue) i.e., a biclustering of the asymmetric similarity graph defined by  $S$ . The node and edge reconstructions (dashed arrows) aid in the feature map learning.

for a matrix  $\mathbf{Q} = \left[ \begin{array}{cc} Q_1 & Q_2 \\ Q_2^{\top} & Q_3 \end{array} \right]$ , one has  $\mathbf{Q} \succeq 0$  if and only if  $Q_1 \succ 0$  and the Schur complement  $Q_3 - Q_2^{\top} Q_1^{-1} Q_2 \succeq 0$  (Boyd & Vandenberghe, 2004).

By substituting the constraints of (1) and inequalities (3) into the objective function of (1), we obtain an objective in primal and dual variables as an upper bound on the primal objective  $\bar{J} \geq J$ :

$$
\begin{array}{l} \min  _ {\boldsymbol {U}, \boldsymbol {V}, \boldsymbol {h} _ {e}, \boldsymbol {h} _ {r}} \bar {J} \triangleq \operatorname {T r} (\boldsymbol {U} ^ {\top} \boldsymbol {V}) - \sum_ {i = 1} ^ {n} \sqrt {w _ {1 , i}} \phi (\boldsymbol {x} _ {i}) ^ {\top} \boldsymbol {U} \boldsymbol {h} _ {\boldsymbol {e} _ {i}} + \frac {1}{2} \sum_ {i = 1} ^ {n} \boldsymbol {h} _ {\boldsymbol {e} _ {i}} ^ {\top} \boldsymbol {\Sigma} \boldsymbol {h} _ {\boldsymbol {e} _ {i}} \\ - \sum_ {j = 1} ^ {m} \sqrt {w _ {2 , j}} \psi \left(\boldsymbol {z} _ {j}\right) ^ {\top} \boldsymbol {V h} _ {\boldsymbol {r} _ {i}} + \frac {1}{2} \sum_ {j = 1} ^ {m} \boldsymbol {h} _ {\boldsymbol {r} _ {j}} ^ {\top} \boldsymbol {\Sigma h} _ {\boldsymbol {r} _ {j}}. \tag {4} \\ \end{array}
$$

Next, we formulate the stationarity conditions of problem (4):

$$
\frac {\partial \bar {J}}{\partial \boldsymbol {V}} = 0 \Rightarrow \boldsymbol {U} = \sum_ {j = 1} ^ {m} \sqrt {w _ {2 , j}} \psi (\boldsymbol {z} _ {j}) \boldsymbol {h} _ {\boldsymbol {r} _ {j}} ^ {\top}, \quad \frac {\partial \bar {J}}{\partial \boldsymbol {h} _ {\boldsymbol {e} _ {i}}} = 0 \Rightarrow \boldsymbol {\Sigma} \boldsymbol {h} _ {\boldsymbol {e} _ {i}} = \sqrt {w _ {1 , i}} \boldsymbol {U} ^ {\top} \phi (\boldsymbol {x} _ {i}), \tag {5}
$$

$$
\frac {\partial \bar {J}}{\partial \boldsymbol {U}} = 0 \Rightarrow \boldsymbol {V} = \sum_ {i = 1} ^ {n} \sqrt {w _ {1 , i}} \phi (\boldsymbol {x} _ {i}) \boldsymbol {h} _ {\boldsymbol {e} _ {i}} ^ {\top}, \quad \frac {\partial \bar {J}}{\partial \boldsymbol {h} _ {\boldsymbol {r} _ {j}}} = 0 \Rightarrow \boldsymbol {\Sigma} \boldsymbol {h} _ {\boldsymbol {r} _ {j}} = \sqrt {w _ {2 , j}} \boldsymbol {V} ^ {\top} \psi (\boldsymbol {z} _ {j}),
$$

from which we then eliminate the primal variables  $\mathbf{U}$  and  $\mathbf{V}$ . This yields the eigenvalue problem:

$$
\left[ \begin{array}{c c} 0 & W _ {1} ^ {1 / 2} S W _ {2} ^ {1 / 2} \\ W _ {2} ^ {1 / 2} S ^ {\top} W _ {1} ^ {1 / 2} & 0 \end{array} \right] \left[ \begin{array}{l} H _ {e} \\ H _ {r} \end{array} \right] = \left[ \begin{array}{l} H _ {e} \\ H _ {r} \end{array} \right] \boldsymbol {\Sigma}, \tag {6}
$$

where  $\mathbf{0}$  is an all-zeros matrix. Note that, by Lanczos' Theorem (Lanczos, 1958), the above eigenvalue problem is equivalent with (2), and that the stationarity conditions (5) provide the relationships between primal and dual variables, which concludes the proof.

We have thus shown the connection between the primal (1) and dual formulation (2). Similarly to the KSVD framework, the wKSVD framework can be used for learning with asymmetric kernel functions and/or rectangular data sources. The spectral biclustering problem can now easily be obtained by choosing the weights  $w_{1,i}$  and  $w_{2,j}$  appropriately.

Corollary 2. Given Proposition 1, and by choosing  $W_{1}$  and  $W_{2}$  to equal  $D_{1}^{-1 / 2}$  and  $D_{2}^{-1 / 2}$ , where  $D_{1,ii} = \sum_{j}S_{ij}$  and  $D_{2,jj} = \sum_{i}S_{ij}$ , we obtain the random walk interpretation  $D_1^{-1 / 2}\pmb {S}\pmb {D}_2^{-1 / 2} = \pmb{H}_e\pmb {\Sigma}\pmb {H}_r^\top$  of the spectral graph bipartitioning problem for the bipartite graph  $S = (\Phi ,\Psi ,S)$ .

Moreover, the wKSVD framework is more general as, on the one hand, one can use a given similarity matrix (e.g. adjacency matrix of a graph) or (asymmetric) kernel function in the dual, or, on the other hand, one can choose to use explicitly defined (deep) feature maps in both primal or dual.

# 3.2 THE HENCLER MODEL

HeNCLer employs the wKSVD framework in a graph setting, where the dataset is a node set  $\mathcal{V}$  and where the asymmetry arises from employing to different mappings that operate on the nodes given the entire graph  $\mathcal{G} = (\mathcal{X},\mathcal{A})$ . Our method is visualized in Figure 2, where red indicates the primal setting of the framework and blue the dual.

In the preceding subsection, we showed that problem (1) has an equivalent dual problem corresponding to the graph bipartitioning problem, when  $w_{1,i}$  and  $w_{2,j}$  are chosen to equal the square root of the inverse of the out-degree and in-degree of a similarity graph  $S$  respectively. This similarity graph  $S$  depends on the feature mappings  $\phi(\cdot)$  and  $\psi(\cdot)$ , which for our method does not only depend on the node of interest, but also on the rest of the input graph and the learnable parameters. The mappings for node  $v$  thus become  $\phi(\boldsymbol{x}_v, \mathcal{G}; \boldsymbol{\theta}_{\phi})$  and  $\psi(\boldsymbol{x}_v, \mathcal{G}; \boldsymbol{\theta}_{\psi})$  and we will ease these notations to  $\phi(\boldsymbol{x}_v)$  and  $\psi(\boldsymbol{x}_v)$ . The ability of our method to learn these feature mappings is an important aspect of our contribution, as a key motivation behind our model is that we need to learn new similarities for clustering heterophilous graphs. The loss function is comprised of three terms: the wKSVD-loss, a node-reconstruction loss, and an edge-reconstruction loss:

$$
\mathcal {L} _ {\mathrm {w K S V D}} (\boldsymbol {U}, \boldsymbol {V}, \boldsymbol {\Sigma}, \boldsymbol {\theta} _ {\phi}, \boldsymbol {\theta} _ {\psi}) + \mathcal {L} _ {\mathrm {N o d e R e c}} (\boldsymbol {U}, \boldsymbol {V}, \boldsymbol {\theta} _ {\phi}, \boldsymbol {\theta} _ {\psi}, \boldsymbol {\theta} _ {\mathrm {r e c}}) + \mathcal {L} _ {\mathrm {E d g e R e c}} (\boldsymbol {U}, \boldsymbol {V}, \boldsymbol {\theta} _ {\phi}, \boldsymbol {\theta} _ {\psi}),
$$

where the trainable parameters of the model are in the multilayer perceptron (MLP) feature maps  $(\theta_{\phi}$  and  $\theta_{\psi})$ , the MLP node decoder  $(\theta_{\mathrm{rec}})$ , in the  $U$  and  $V$  projection matrices, and in the singular values  $\Sigma$ . All these parameters are trained end-to-end and we next explain the losses in more detail.

wKSVD-loss Instead of solving the SVD in the dual formulation, HeNCler leverages the primal formulation (1) of the wKSVD framework for greater computational efficiency. While equation (1) assumes that the feature maps  $\phi(\cdot)$  and  $\psi(\cdot)$  are fixed, HeNCler utilizes parametric functions  $\phi(\cdot; \pmb{\theta}_{\phi})$  and  $\psi(\cdot; \pmb{\theta}_{\psi})$ , enabling it to learn new similarities between nodes. By incorporating regularization terms for these functions and defining the weighting scalars as  $w_{1,v} = D_{1,vv}^{-1} = 1 / \sum_u \phi(\pmb{x}_v)^\top \psi(\pmb{x}_u)$  and  $w_{2,v} = D_{2,vv}^{-1} = 1 / \sum_u \phi(\pmb{x}_u)^\top \psi(\pmb{x}_v)$ , we derive the wKSVD-loss:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {w K S V D}} \triangleq - \sum_ {v = 1} ^ {| \mathcal {V} |} D _ {1, v v} ^ {- 1} \phi (\boldsymbol {x} _ {v}) ^ {\top} \boldsymbol {U} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {U} ^ {\top} \phi (\boldsymbol {x} _ {v}) - \sum_ {v = 1} ^ {| \mathcal {V} |} D _ {2, v v} ^ {- 1} \psi (\boldsymbol {x} _ {v}) ^ {\top} \boldsymbol {V} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {V} ^ {\top} \psi (\boldsymbol {x} _ {v}) \\ + \operatorname {T r} \left(\mathrm {U} ^ {\top} \mathrm {V}\right) + \sum_ {\mathrm {v} = 1} ^ {| \mathcal {V} |} \sqrt {\mathrm {D} _ {1 , \mathrm {v v}} ^ {- 1} \mathrm {D} _ {2 , \mathrm {v v}} ^ {- 1}} \phi \left(\mathrm {x} _ {\mathrm {v}}\right) ^ {\top} \psi \left(\mathrm {x} _ {\mathrm {v}}\right). \tag {7} \\ \end{array}
$$

The primal formulation of HeNCler (7) can be understood as follows: The first two terms aim to maximize the weighted variance of the learned node representations  $e$  and  $r$ . The third and fourth terms act as regularizers, encouraging asymmetry by penalizing the similarity between  $U$  and  $V$ , and between  $\phi(\boldsymbol{x}_v)$  and  $\psi(\boldsymbol{x}_v)$ , respectively.

For the two feature maps  $\phi (\cdot)$  and  $\psi (\cdot)$ , we employ two MLPs:  $\phi (\pmb {x}_v,\mathcal{G};\pmb {\theta}_\phi)\equiv \mathrm{MLP}_{\phi}(\pmb {x}_v||\mathrm{PE}_v;\pmb {\theta}_\phi)$  and  $\psi (\pmb {x}_v,\mathcal{G};\pmb {\theta}_\psi)\equiv \mathrm{MLP}_{\psi}(\pmb {x}_v||\mathrm{PE}_v;\pmb {\theta}_\psi)$ . We construct a random walks positional encoding (PE) (Dwivedi et al., 2022) to embed the network's structure and concatenate this encoding with the node attributes. The MLPs have two linear layers with a LeakyReLU activation function in between, followed by a batch normalization layer. The singular values in  $\Sigma$  are jointly learned, constrained to lie between 0 and 1, with the additional condition that  $\mathrm{Tr}(\Sigma^{-\frac{1}{2}}) = 1$ .

Reconstruction losses Since the feature maps  $\phi(\cdot)$  and  $\psi(\cdot)$  need to be learned, an additional loss function beyond the above regularization term is required to effectively optimize the parameters of the MLPs. As the node clustering setting is completely unsupervised, we add a decoder network and

a reconstruction loss. This technique has been proven to be effective for unsupervised learning in the RKM-framework (Pandey et al., 2022), as well as for unsupervised node representation learning (Sun et al., 2021). For heterophilous graphs, we argue that it is particularly important to also reconstruct node features and not only the graph structure.

For the node reconstruction, we first project the  $e$  and  $r$  variables back to feature space, concatenate these and then map to input space with another MLP. This MLP has also two layers and a leaky ReLU activation function. The hidden layer size is set to the average of the latent dimension and input dimension. With the mean-squared-error as the associated loss, this gives:

$$
\mathcal {L} _ {\text {N o d e R e c}} = \frac {1}{| \mathcal {V} |} \sum_ {v \in \mathcal {V}} \left\| \mathrm {M L P} _ {\text {r e c}} \left(\boldsymbol {U} \boldsymbol {e} _ {v} \| \boldsymbol {V} \boldsymbol {r} _ {v}; \boldsymbol {\theta} _ {\text {r e c}}\right) - \boldsymbol {x} _ {v} \right\| ^ {2}. \tag {8}
$$

To reconstruct edges, we use a simple dot-product decoder  $\sigma(e_u^\top U^\top V r_v)$  where  $\sigma$  is the sigmoid function. By using the  $e$  representation for source nodes and  $r$  for target nodes, this reconstruction is asymmetric and can reconstruct directed graphs. We use a binary cross-entropy loss:

$$
\mathcal {L} _ {\text {E d g e R e c}} = \frac {1}{| \mathcal {U} |} \sum_ {(u, v) \in \mathcal {U}} \operatorname {B C E} \left(\sigma \left(\boldsymbol {e} _ {u} ^ {\top} \boldsymbol {U} ^ {\top} \boldsymbol {V} \boldsymbol {r} _ {v}\right), \mathcal {E} _ {u v}\right), \tag {9}
$$

where  $\mathcal{U}$  is a node-tuple set, resampled every epoch, containing  $2|\mathcal{V}|$  positive edges from  $\mathcal{E}$  and  $2|\mathcal{V}|$  negative edges from  $\mathcal{E}^C$ , and  $\mathcal{E}_{uv} \in \{0,1\}$  indicates whether an edge  $(u,v)$  exists:  $(u,v) \in \mathcal{E}$ .

Optimizer, constraints, and cluster assignment We use Adam (Kingma & Ba, 2015) for the training of all parameters. The batch normalization in the MLP's keeps the wKSVD-loss bounded and the constraints on the singular values is enforced with a softmax function. Cluster assignments are obtained by KMeans clustering on the concatenation of learned  $e$  and  $r$  node representations.

HeNCler jointly learns the wKSVD projection matrices,  $U$  and  $V$ , along with the feature map parameters,  $\theta_{\phi}$  and  $\theta_{\psi}$ . The wKSVD loss improves the cluster-ability of the learned similarity graph, ensuring that  $e$  and  $r$  function as spectral biclustering embeddings. The two distinct feature maps enable asymmetric learning, effectively capturing potential asymmetric relationships in the data, while the reconstruction losses ensure robust and meaningful representation learning.

Table 2: Dataset statistics of the employed heterophilous graphs.  

<table><tr><td>Dataset</td><td>short</td><td># Nodes</td><td># Edges</td><td># Classes</td><td>Directed</td><td>H(G)</td></tr><tr><td>Texas</td><td>tex</td><td>183</td><td>325</td><td>5</td><td>✓</td><td>0.000</td></tr><tr><td>Cornell</td><td>corn</td><td>183</td><td>298</td><td>5</td><td>✓</td><td>0.150</td></tr><tr><td>Wisconsin</td><td>wis</td><td>251</td><td>515</td><td>5</td><td>✓</td><td>0.084</td></tr><tr><td>Chameleon</td><td>cha</td><td>2,277</td><td>31,371</td><td>5</td><td>✗</td><td>0.042</td></tr><tr><td>Squirrel</td><td>squi</td><td>5,201</td><td>198,353</td><td>5</td><td>✗</td><td>0.031</td></tr><tr><td>Roman-empire</td><td>rom</td><td>22,662</td><td>32,927</td><td>18</td><td>✗</td><td>0.021</td></tr><tr><td>Minesweeper</td><td>mine</td><td>10,000</td><td>39,402</td><td>2</td><td>✗</td><td>0.009</td></tr><tr><td>Tolokers</td><td>tol</td><td>11,758</td><td>519,000</td><td>2</td><td>✗</td><td>0.180</td></tr></table>

# 4 EXPERIMENTS

Datasets We assess the performance of HeNCLer on heterophilous attributed graphs that are available in literature. we use Texas, Cornell, and Wisconsin (Pei et al., 2020)<sup>1</sup>, which are directed webpage networks where edges encode hyperlinks between pages. Next, we use Chameleon and Squirrel (Rozemberczki et al., 2021), which are undirected Wikipedia webpage networks where edges encode mutual links. We further assess our model on the undirected graphs: Roman-empire, Minesweeper, and Tolokers (Platonov et al., 2023), which are a graph representation of a Wikipedia article, a grid graph based on the minesweeper game, and a crowd-sourcing network respectively. We include

Table 3: Experimental results on heterophilous graphs. We report NMI and F1 scores for 10 runs (mean ± standard deviation), where higher values indicate better performance. The best results for each metric are highlighted in bold.  

<table><tr><td rowspan="2" colspan="2">Dataset</td><td colspan="5">Baselines</td><td>Ours</td></tr><tr><td>KMeans</td><td>MinCutPool</td><td>DMoN</td><td>S3GC</td><td>MUSE</td><td>HeNCler</td></tr><tr><td rowspan="2">tex</td><td>NMI</td><td>4.97±1.00</td><td>11.60±2.19</td><td>9.06±2.11</td><td>11.56±1.46</td><td>39.23±4.91</td><td>43.65±2.52</td></tr><tr><td>F1</td><td>59.27±0.83</td><td>55.26±0.56</td><td>47.76±4.79</td><td>43.69±2.74</td><td>65.96±3.52</td><td>71.39±2.16</td></tr><tr><td rowspan="2">corn</td><td>NMI</td><td>5.42±2.04</td><td>17.04±1.61</td><td>12.49±2.51</td><td>14.48±1.79</td><td>38.99±2.73</td><td>41.52±4.35</td></tr><tr><td>F1</td><td>52.97±0.24</td><td>51.21±5.06</td><td>43.83±6.23</td><td>33.13±0.83</td><td>60.58±3.61</td><td>63.40±3.67</td></tr><tr><td rowspan="2">wis</td><td>NMI</td><td>6.84±4.39</td><td>13.38±2.36</td><td>12.56±1.23</td><td>13.07±0.61</td><td>39.71±2.22</td><td>47.13±1.76</td></tr><tr><td>F1</td><td>56.16±0.58</td><td>55.63±2.96</td><td>45.72±7.85</td><td>31.71±2.25</td><td>58.94±3.09</td><td>68.30±2.17</td></tr><tr><td rowspan="2">cha</td><td>NMI</td><td>0.44±0.11</td><td>11.88±1.99</td><td>12.87±1.86</td><td>15.83±0.26</td><td>23.06±0.28</td><td>23.89±0.84</td></tr><tr><td>F1</td><td>53.23±0.07</td><td>50.40±5.65</td><td>45.05±4.30</td><td>36.51±0.24</td><td>52.10±0.48</td><td>44.14±1.83</td></tr><tr><td rowspan="2">squi</td><td>NMI</td><td>1.40±2.12</td><td>6.35±0.32</td><td>3.08±0.38</td><td>3.83±0.11</td><td>8.30±0.23</td><td>9.67±0.13</td></tr><tr><td>F1</td><td>54.05±2.72</td><td>55.26±0.57</td><td>49.21±2.74</td><td>35.08±0.18</td><td>50.07±5.99</td><td>36.51±2.39</td></tr><tr><td rowspan="2">rom</td><td>NMI</td><td>35.20±1.79</td><td>9.97±2.02</td><td>13.14±0.53</td><td>14.48±0.21</td><td>40.50±0.73</td><td>36.99±0.61</td></tr><tr><td>F1</td><td>37.17±2.12</td><td>42.19±0.26</td><td>22.69±3.91</td><td>17.76±0.53</td><td>38.34±0.35</td><td>35.43±1.07</td></tr><tr><td rowspan="2">mine</td><td>NMI</td><td>0.02±0.02</td><td>6.16±2.17</td><td>6.87±2.91</td><td>6.53±0.17</td><td>0.06±0.01</td><td>0.06±0.00</td></tr><tr><td>F1</td><td>73.63±3.58</td><td>71.76±8.86</td><td>70.42±9.47</td><td>48.78±0.63</td><td>75.77±2.24</td><td>76.48±1.56</td></tr><tr><td rowspan="2">tol</td><td>NMI</td><td>3.04±2.83</td><td>6.68±0.98</td><td>6.69±0.20</td><td>5.99±0.05</td><td>6.67±0.55</td><td>6.73±0.59</td></tr><tr><td>F1</td><td>65.56±10.49</td><td>72.10±10.38</td><td>67.87±4.74</td><td>59.17±0.27</td><td>73.56±1.94</td><td>73.66±2.10</td></tr></table>

experimental results for additional homophilous datasets in Appendix B. The dataset statistics can be consulted in Table 2, where the class insensitive edge homophily ratio  $\mathcal{H}(\mathcal{G})$  (Lim et al., 2021) is a homophily measure.

Model selection and metrics Model selection in this unsupervised setting is non-trivial, and the best metric depends on the task at hand. Therefore, this is not the scope of this paper and we assess our model agnostically to the model selection, and fairly w.r.t. to the baselines. We fix the hyperparameter configuration of the models across all datasets. We train for a fixed number of epochs and keep track of the evaluation metrics to report the best observed result. We repeat the training process 10 times and report average best results with standard deviations. We report the normalized mutual information (NMI) and pairwise F1-scores, based on the class labels.

Baselines and hyperparameters We compare our model against several methods, including a simple KMeans based on node attributes, adjacency partitioning-based approaches such as MinCutPool (Bianchi et al., 2020) and DMoN (Tsitsulin et al., 2023), as well as  $\mathbf{S}^3\mathrm{GC}$  (Devvrit et al., 2022) and MUSE (Yuan et al., 2023), which represent the current state-of-the-art in homophilous and heterophilous node clustering, respectively. For HeNCler, we fix the hyperparameters to: MLP hidden dimensions 256, output dimensions 128, latent dimension  $s = 2 \times \#$  classes, learning rate 0.01, and epochs 300. For the baselines, we used their code implementations and the default hyperparameter settings as proposed by the authors. The number of clusters to infer is set to the number of classes cfr. Table 2 for all methods. The experiments are run on a Nvidia V100 GPU.

Experimental results Table 3 presents the experimental results for heterophilous graphs. HeNCler consistently demonstrates superior performance, significantly outperforming KMeans, MinCutPool, DMoN,  $S^3GC$ , and MUSE, especially on the directed graphs. For undirected graphs, HeNCler also shows strong results, achieving the best performance in 5 out of 10 cases, compared to KMeans (1/10), MinCutPool (2/10), DMoN (1/10),  $S^3GC$  (0/10), and MUSE (1/10). These results highlight HeNCler's versatility and effectiveness in handling heterophilous graph structures.

Ablation studies We conduct several ablation studies, presented in Table 4. The 'Undirected' variant refers to a simplified, symmetric version of the model that uses a single MLP for both the  $\phi(\cdot)$  and  $\psi(\cdot)$  mappings, i.e.,  $\phi(\cdot) \equiv \psi(\cdot)$ . In this version, the model loses its asymmetry. The

Table 4: Ablation study results. We report mean NMI and F1 scores for 10 runs (higher is better) for different model configurations. Best results are highlighted in bold.  

<table><tr><td rowspan="2">Metric</td><td colspan="2">tex</td><td colspan="2">corn</td><td colspan="2">cha</td><td colspan="2">rom</td><td colspan="2">tol</td></tr><tr><td>NMI</td><td>F1</td><td>NMI</td><td>F1</td><td>NMI</td><td>F1</td><td>NMI</td><td>F1</td><td>NMI</td><td>F1</td></tr><tr><td>Undirected</td><td>27.58</td><td>65.20</td><td>18.12</td><td>53.69</td><td>19.91</td><td>44.08</td><td>33.17</td><td>33.57</td><td>6.33</td><td>73.89</td></tr><tr><td>Reconstr only</td><td>29.54</td><td>66.64</td><td>27.76</td><td>54.70</td><td>22.02</td><td>43.42</td><td>40.05</td><td>35.16</td><td>6.18</td><td>68.60</td></tr><tr><td>wKSVD only</td><td>31.64</td><td>62.83</td><td>20.63</td><td>47.12</td><td>22.60</td><td>42.98</td><td>35.99</td><td>35.30</td><td>4.42</td><td>68.45</td></tr><tr><td>HeNCler</td><td>43.65</td><td>71.39</td><td>41.52</td><td>63.40</td><td>23.89</td><td>44.14</td><td>36.99</td><td>35.43</td><td>6.73</td><td>73.66</td></tr></table>

'wKSVD only' and 'Reconstr only' variations reflect models that incorporate only the wKSVD loss  $(\mathcal{L}_{\mathrm{wKSVD}})$  and the reconstruction losses  $(\mathcal{L}_{\mathrm{NodeRec}} + \mathcal{L}_{\mathrm{EdgeRec}})$ , respectively. Interestingly, as shown in Table 4, even for undirected graphs, introducing asymmetry in HeNCler enhances clustering performance. Furthermore, all loss components are shown to contribute positively to HeNCler's overall performance. For a comprehensive analysis, including results across all datasets and standard deviations, we refer the reader to Table 7 in Appendix B.

# 5 DISCUSSION

A key motivation behind HeNCler is to learn a new graph representation where nodes belonging to the same cluster are positioned closer together, driven by the clustering objective. This results in spectral biclustering embeddings that exhibit improved cluster-ability. Note that HeNCler uses KMeans to obtain cluster assignments. Therefore, the comparisons between HeNCler and KMeans, as shown in Tables 3 and 6, demonstrate that our model enhances the cluster-ability of the node representations relative to the original input features.

The asymmetry in HeNCler eliminates the undirected constraints of traditional adjacency partitioning-based models, enabling superior performance on directed graphs, as shown in Table 3. Furthermore, our ablation study in Table 4 shows that, while most of the performance on undirected graphs stem from the graph learning component, HeNCler is additionally able to capture and learn meaningful asymmetric information. This capacity to extract valuable asymmetric insights from symmetric data is a common occurrence in KSVD frameworks (He et al., 2023; Tao et al., 2024). Importantly, thanks to the added performance boost from asymmetry, on top of the benefits from similarity learning, HeNCler outperforms state-of-the-art models, even when applied to undirected graphs.

We visualize the learned similarity matrix  $S = \Phi \Psi^{\top}$  for two datasets in Figure 3. These matrices are generally asymmetric, with the asymmetry particularly pronounced in the directed graph of the Wisconsin dataset. In contrast, the Roman-Empire dataset, which is represented by an undirected graph, exhibits less asymmetry in the learned similarity matrix. This demonstrates the adaptability of HeNCLer to handle both directed and undirected graphs. Further, given the observable block structures, the learned similarities are meaningful w.r.t. to the ground truth node labels. Note however that our model operates in the primal setting and directly projects

![](images/5844a4a7a2e1d8b86937553c0244516e33462212204047e89222c174312da779.jpg)  
Figure 3: The learned matrix  $S = \Phi \Psi^{\top}$  for the Wisconsin (left) and Roman-empire (right) dataset. Rows and columns are grouped according to ground-truth node labels.

the learned mappings  $\phi$  and  $\psi$  to their final embeddings  $e$  and  $r$  using  $U$  and  $V$  respectively, avoiding quadratic space complexity and cubic time complexity of the SVD. This is the motivation of employing a kernel based method, and exploiting the primal-dual framework that comes with it. In fact, the matrices in Figure 3 are only constructed for the sake of this visualization.

Computational complexity The space and time complexity of the current implementation of HeNCler are both linear w.r.t. the number of nodes  $\mathcal{O}(|\mathcal{V}|)$ . Whereas MinCutPool and DMoN need

all the node attributes in memory to calculate the loss w.r.t. the full adjacency matrix, HeNCler is easily adaptable to work with minibatches which reduces space complexity to the minibatch size  $\mathcal{O}(|\mathcal{B}|)$ . Although HeNCler relies on edge reconstruction, the edge sampling avoids quadratic complexity w.r.t. number of nodes, and is specifically designed to scale with the number of nodes, rather than the number of edges. Assuming the graphs are sparse, we add an overview of space and time complexity w.r.t. the number of nodes and edges for all methods in Table 1. A detailed table with measured computation times is provided in Appendix C.

# 6 CONCLUSION AND FUTURE WORK

We tackle three limitations of current node clustering algorithms, that prevent these methods from effectively clustering nodes in heterophilous graphs: they assume homophily in their loss, they are only defined for undirected graphs and/or they lack a specific focus on clustering.

To this end, we introduce a weighted kernel SVD framework and harness its primal-dual equivalences. HeNCler relies on the dual interpretation for its theoretical motivation, while it benefits from the computational advantages of its implementation in the primal. In an end-to-end fashion, it learns new similarities, which are asymmetric where necessary, and node embeddings resulting from the spectral biclustering interpretation of these learned similarities. As empirical evidence shows, our approach effectively eliminates the aforementioned limitations, significantly outperforming current state-of-the-art alternatives.

HeNCler is the first heterophilous node clustering model that does not rely on contrastive learning techniques. Future research could explore the integration of contrastive learning into HeNCler, potentially combining the strengths of both approaches. Another next step can be to investigate how to do the cluster assignments in a graph pooling setting (i.e., differentiable graph coarsening), to enable end-to-end learning for downstream graph prediction tasks.

# REFERENCES

Sonny Achten, Arun Pandey, Hannes De Meulemeester, Bart De Moor, and Johan A. K. Suykens. Duality in Multi-View Restricted Kernel Machines. ICML Workshop on Duality for Modern Machine Learning, 2023. arXiv:2305.17251 [cs].  
Sonny Achten, Francesco Tonin, Panagiotis Patrinos, and Johan A.K. Suykens. Unsupervised Neighborhood Propagation Kernel Layers for Semi-supervised Node Classification. Proceedings of the AAAI Conference on Artificial Intelligence, 38(10):10766-10774, Mar. 2024.  
Carlos Alzate and Johan A. K. Suykens. Multiway Spectral Clustering with Out-of-Sample Extensions through Weighted Kernel PCA. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(2):335-347, 2010.  
Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Spectral Clustering with Graph Neural Networks for Graph Pooling. In International Conference on Machine Learning, 2020.  
Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, New York, NY, USA, 2004. ISBN 0521833787.  
Jingfan Chen, Guanghui Zhu, Yifan Qi, Chunfeng Yuan, and Yihua Huang. Towards self-supervised learning on graphs with heterophily. In ACM International Conference on Information & Knowledge Management, pp. 201-211, 2022.  
Yingyi Chen, Qinghua Tao, Francesco Tonin, and Johan Suykens. Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation. In Advances in Neural Information Processing Systems, 2023.  
Fnu Devvrit, Aditya Sinha, Inderjit Dhillon, and Prateek Jain. S3GC: Scalable Self-Supervised Graph Clustering. In Advances in Neural Information Processing Systems, 2022.  
Inderjit S. Dhillon. Co-clustering documents and words using bipartite spectral graph partitioning. In ACM SIGKDD international conference on Knowledge discovery and data mining, 2001.

Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. AAAI Workshop on Deep Learning on Graphs: Methods and Applications, 2021. arXiv:2012.09699 [cs].  
Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. In International Conference on Learning Representations, 2022.  
Zheng Gong, Guifeng Wang, Ying Sun, Qi Liu, Yuting Ning, Hui Xiong, and Jingyu Peng. Beyond Homophily: Robust Graph Anomaly Detection via Neural Sparsification. In International Joint Conference on Artificial Intelligence, 2023.  
William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive Representation Learning on Large Graphs. In Advances in Neural Information Processing Systems, 2017.  
Kaveh Hassani and Amir Hosein Khas Ahmadi. Contrastive multi-view representation learning on graphs. In International Conference on Machine Learning, 2020.  
Mingzhen He, Fan He, Lei Shi, Xiaolin Huang, and Johan A. K. Suykens. Learning With Asymmetric Kernels: Least Squares and Feature Interpretation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):10044-10054, 2023.  
Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations, 2015.  
Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations, 2017.  
Cornelius Lanczos. Linear systems in self-adjoint form. The American Mathematical Monthly, 9(65): 665-679, 1958.  
Derek Lim, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, and Ser-Nam Lim. Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods. In Advances in Neural Information Processing Systems, 2021.  
Shirui Pan, Ruiqi Hu, Sai-Fu Fung, Guodong Long, Jing Jiang, and Chengqi Zhang. Learning graph embedding with adversarial training methods. IEEE Transactions on Cybernetics, 50(6): 2475-2487, June 2020. ISSN 2168-2275.  
Arun Pandey, Joachim Schreurs, and Johan A.K. Suykens. Generative restricted kernel machines: A framework for multi-view generation and disentangled feature learning. *Neural Networks*, 135: 177–191, 2021.  
Arun Pandey, Michael Fanuel, Joachim Schreurs, and Johan A. K. Suykens. Disentangled representation learning and generation with manifold optimization. *Neural Computation*, 34(10):2009-2036, 09 2022. ISSN 0899-7667.  
Jiwoong Park, Minsik Lee, Hyung Jin Chang, Kyuewang Lee, and Jin Young Choi. Symmetric graph convolutional autoencoder for unsupervised graph representation learning. In IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.  
Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. Geom-GCN: Geometric Graph Convolutional Networks. In International Conference on Learning Representations, 2020.  
Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In International Conference on Learning Representations, 2023.  
R. Tyrrell Rockafellar. Conjugate Duality and Optimization. SIAM, 1974.  
Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-Scale attributed node embedding. Journal of Complex Networks, 9(1):1-22, 2021.  
Ruslan Salakhutdinov. Learning Deep Generative Models. Annual Review of Statistics and Its Application, 2(1):361-385, 2015.

Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective Classification in Network Data. AI magazine, 29(3):93-93, 2008.  
Dengdi Sun, Dashuang Li, Zhuanlian Ding, Xingyi Zhang, and Jin Tang. Dual-decoder graph autoencoder for unsupervised graph representation learning. Knowledge-Based Systems, 234: 107564, December 2021. ISSN 09507051.  
Johan A. K. Suykens. SVD revisited: A new variational principle, compatible feature maps and nonlinear extensions. Applied and Computational Harmonic Analysis, 40(3):600-609, May 2016. ISSN 1063-5203.  
Johan A. K. Suykens. Deep Restricted Kernel Machines Using Conjugate Feature Duality. Neural Computation, 29(8):2123-2163, 2017.  
Johan A. K. Suykens, Tony Van Gestel, Jos De Brabanter, Bart De Moor, and Joos Vandewalle. Least Squares Support Vector Machines. World Scientific, Singapore, 2002.  
Qinghua Tao, Francesco Tonin, Alex Lambert, Yingyi Chen, Panagiotis Patrinos, and Johan A. K. Suykens. Learning in Feature Spaces via Coupled Covariances: Asymmetric Kernel SVD and Nyström method. International Conference on Machine Learning, 2024.  
Francesco Tonin, Panagiotis Patrinos, and Johan A. K. Suykens. Unsupervised learning of disentangled representations in deep restricted kernel machines with orthogonality constraints. *Neural Networks*, 142:661-679, 2021.  
Anton Tsitsulin, John Palowitch, Bryan Perozzi, and Emmanuel Müller. Graph Clustering with Graph Neural Networks. Journal of Machine Learning Research, 24(127):1-21, 2023.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph Attention Networks. In International Conference on Learning Representations, 2018.  
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. A Comprehensive Survey on Graph Neural Networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1):4-24, 2021.  
Teng Xiao, Zhengyu Chen, Zhimeng Guo, Zeyang Zhuang, and Suhang Wang. Decoupled self-supervised learning for graphs. In Advances in Neural Information Processing Systems, 2022.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural Networks? In International Conference on Learning Representations, 2019.  
Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting Semi-Supervised Learning with Graph Embeddings. In International Conference on Machine Learning, 2016.  
Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do Transformers Really Perform Badly for Graph Representation? In Advances in Neural Information Processing Systems, 2022.  
Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. In Advances in Neural Information Processing Systems, 2020.  
Mengyi Yuan, Minjie Chen, and Xiang Li. MUSE: Multi-View Contrastive Learning for Heterophilic Graphs. In ACM International Conference on Information and Knowledge Management, pp. 3094-3103, 2023.  
Xin Zheng, Yi Wang, Yixin Liu, Ming Li, Miao Zhang, Di Jin, Philip S. Yu, and Shirui Pan. Graph neural networks for graphs with heterophily: A survey, 2024. arXiv:2202.07082 [cs].
