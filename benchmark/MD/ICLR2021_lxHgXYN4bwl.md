# EXPRESSIVE POWER OF INVARIANT AND EQUIVARIANT GRAPH NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Various classes of Graph Neural Networks (GNN) have been proposed and shown to be successful in a wide range of applications with graph structured data. In this paper, we propose a theoretical framework able to compare the expressive power of these GNN architectures. The current universality theorems only apply to intractable classes of GNNs. Here, we prove the first approximation guarantees for practical GNNs, paving the way for a better understanding of their generalization. Our theoretical results are proved for invariant GNNs computing a graph embedding (permutation of the nodes of the input graph does not affect the output) and equivariant GNNs computing an embedding of the nodes (permutation of the input permutes the output). We show that Folklore Graph Neural Networks (FGNN), which are tensor based GNNs augmented with matrix multiplication are the most expressive architectures proposed so far for a given tensor order. We illustrate our results on the Quadratic Assignment Problem (a NP-Hard combinatorial problem) by showing that FGNNs are able to learn how to solve the problem, leading to much better average performances than existing algorithms (based on spectral, SDP or other GNNs architectures). On a practical side, we also implement masked tensors to handle batches of graphs of varying sizes.

# 1 INTRODUCTION

Graph Neural Networks (GNN) are designed to deal with graph structured data. Since a graph is not changed by permutation of its nodes, GNNs should be either invariant if they return a result that must not depend on the representation of the input (typically when building a graph embedding) or equivariant if the output must be permuted when the input is permuted (typically when building an embedding of the nodes). More fundamentally, incorporating symmetries in machine learning is a fundamental problem as it allows to reduce the number of degree of freedom to be learned.

Deep learning on graphs. This paper focuses on learning deep representation of graphs with network architectures, namely GNN, designed to be invariant to permutation or equivariant by permutation. From a practical perspective, various message passing GNNs have been proposed, see Dwivedi et al. (2020) for a recent survey and benchmarking on learning tasks. In this paper, we study 3 architectures: Message passing GNN (MGNN) which is probably the most popular architecture used in practice, order- $k$  Linear GNN ( $k$ -LGNN) proposed in Maron et al. (2018) and order- $k$  Folklore GNN ( $k$ -FGNN) first introduced by Maron et al. (2019a). MGNN layers are local thus highly parallelizable on GPUs which make them scalable for large sparse graphs.  $k$ -LGNN and  $k$ -FGNN are dealing with representations of graphs as tensors of order  $k$  which make them of little practical use for  $k \geq 3$ .

In order to compare these architectures, the separating power of these networks has been compared to a hierarchy of graph invariants developed for the graph isomorphism problem. Namely, for  $k \geq 2$ ,  $k$ -WL( $G$ ) are invariants based on the Weisfeiler-Lehman tests (described in Section 4.1). For each  $k \geq 2$ ,  $(k + 1)$ -WL has strictly more separating power than  $k$ -WL (in the sense that there is a pair of non-isomorphic graphs distinguishable by  $(k + 1)$ -WL and not by  $k$ -WL). GIN (which are invariant MGNN) introduced in Xu et al. (2018) are shown to be as powerful as 2-WL. In Maron et al. (2019a), Geerts (2020b) and Geerts (2020a),  $k$ -LGNN are shown to be as powerful as  $k$ -WL and 2-FGNN is shown to be as powerful as 3-WL. In this paper, we extend this last result about  $k$ -FGNN to general values of  $k$ . So in term of separating power, when restricted to tensors of order  $k$ ,  $k$ -FGNN is the most powerful architecture. This means that for a given pair of graphs  $G$  and

$G^{\prime}$  , if  $(k + 1)$  -WL  $(G)\neq (k + 1)$  -WL  $(G^{\prime})$  , then there exists a  $k$  -FGNN, say  $\mathbf{GNN}_{G,G^{\prime}}$  such that  $\mathbf{GNN}_{G,G^{\prime}}(G)\neq \mathbf{GNN}_{G,G^{\prime}}(G^{\prime})$

Approximation results for GNNs. Results on the separating power of GNNs only deal with pairwise comparison of graphs: we need a priori a different GNNs for each pair of graphs in order to distinguish them. Such results are of little help in a practical learning scenario. Our main contribution in this paper overcomes this issue and we show that a single GNN can give a meaningful representation for all graphs. More precisely, we characterize the set of functions that can be approximated by MGNNs,  $k$ -LGNNs and  $k$ -FGNNs respectively. Standard Stone-Weierstrass theorem shows that if an algebra  $\mathcal{A}$  of real continuous functions separates points, then  $\mathcal{A}$  is dense in the set of continuous function on a compact. Here we extend such a theorem to general functions with symmetries and apply it to invariant and equivariant functions to get our main result for GNNs. As a consequence, we show that  $k$ -FGNN has the best approximation power among architectures dealing with tensors of order  $k$ .

Universality results for GNNs. Universal approximation theorem (similar to Cybenko (1989) for multi-layers perceptron) have been proved for linear GNNs in Maron et al. (2019b); Keriven & Peyré (2019); Chen et al. (2019). They show that some classes of GNNs can approximate any function defined on graphs. To be able to approximate any invariant function, they require the use of very complex networks, namely  $k$ -LGNN where  $k$  tends to infinity with  $n$  the number of nodes. Since we prove that any invariant function less powerful than  $(k + 1)$ -WL can be approximated by a  $k$ -FGNN, letting  $k$  tends to infinity directly implies universality. Universality results for  $k$ -FGNN is another contribution of our work.

Equivariant GNNs. Our second set of results extends previous analysis from invariant functions to equivariant functions. There are much less results about equivariant GNNs: Keriven & Peyré (2019) proves the universality of linear equivariant GNNs, and Maehara & Hoang (2019) shows the universality of a new class of networks they introduced. Here, we consider a natural equivariant extension of  $k$ -WL and prove that equivariant  $(k + 1)$ -LGNNs and  $k$ -FGNN can approximate any equivariant function less powerful than this equivariant  $(k + 1)$ -WL for  $k \geq 1$ . At this stage, we should note that all universality results for GNNs by Maron et al. (2019b); Keriven & Peyré (2019); Chen et al. (2019) are easily recovered from our main results. Also our analysis is valid for graphs of varying sizes.

Empirical results for the Quadratic Assignment Problem (QAP). To validate our theoretical contributions, we empirically show that 2-FGNN outperforms classical MGNN. Indeed, Maron et al. (2019a) already demonstrate state of the art results for the invariant version of 2-FGNNs (for graph classification or graph regression). Here we consider the graph alignment problem and show that the equivariant 2-FGNN is able to learn a node embedding which beats by a large margin other algorithms (based on spectral method, SDP or GNNs).

Outline and contribution. After reviewing more previous works and notations in the next section, we define the various classes of GNNs studied in this paper in Section 3: message passing GNN, linear GNN and folklore GNN. Section 4 contains our main theoretical results for GNNs. First in Section 4.2 we describe the separating power of each GNN architecture with respect to the Weisfeiler-Lehman test. In Section 4.3, we give approximation guarantees for MGNNs, LGNNs and FGNNs at fixed order of tensor. They cover both the invariant and equivariant cases and are our main theoretical contribution. For this, we develop in Section D a fine-grained Stone-Weierstrass approximation theorem for vector-valued functions with symmetries. Our theorem handles both invariant and equivariant cases and is inspired by recent works in approximation theory. In Section 5, we illustrate our theoretical results on a practical application: the graph alignment problem, a well-known NP-hard problem. We highlight a previously overlooked implementation question: the handling of batches of graphs of varying sizes. A PyTorch implementation of the code necessary to reproduce the results is available in the supplementary.

# 2 RELATED WORK

The pioneering works that applied neural networks to graphs are Gori et al. (2005) and Scarselli et al. (2009) that learn node representation with recurrent neural networks. More recent message passing architectures make use of non-linear functions of the adjacency matrix (Kipf & Welling, 2016), for example polynomials (Defferrard et al., 2016). For regular-grid graphs, they match classical

convolutional networks which by design can only approximate translation-invariant functions and hence have limited expressive power. In this paper, we focus instead on more expressive architectures.

Following the recent surge in interest in graph neural networks, some works have tried to extend the pioneering work of Cybenko (1989); Hornik et al. (1989) for various GNN architectures. Among the first ones is Scarselli et al. (2009), which studied invariant message-passing GNNs. They showed that such networks can approximate, in a weak sense, all functions whose discriminatory power is weaker than 1-WL. Yarotsky (2018) described universal architectures which are invariant or equivariant to some group action. These models rely on polynomial intermediate layers of arbitrary degrees, which would be prohibitive in practice. Maron et al. (2019b) leveraged classical results about the polynomials invariant to a group action to show that  $k$ -LGNN are universal as  $k$  tends to infinity with the number of nodes. Keriven & Peyré (2019) derived a similar result, in the more complicated equivariant case by introducing a new Stone-Weierstrass theorem. Similarly to Maron et al. (2019b), they require the order of tensors to go to infinity. Another route towards universality is the one of Chen et al. (2019). In the invariant setting, they show for a class of GNN that universality is equivalent to being able to discriminate between (non-isomorphic) graphs. However, the only way to achieve such discriminatory power is to use tensors of arbitrary high order, see also Ravanbakhsh (2020). Our work encompasses and precise these results using high-order tensors as it yields approximation guarantees even at fixed order of tensor.

CPNGNN in Sato et al. (2019) and DimeNet in Klicpera et al. (2020) are message passing GNN incorporating more information than those studied here. Partial results about their separating power follows from Garg et al. (2020) which provides impossibility results to decide graph properties including girth, circumference, diameter, radius, conjoint cycle, total number of cycles, and  $k$ -cliques. Note that 2-WL solves  $k$ -cliques for  $k \leq 6$  Fürer (2017) so that these networks are probably not comparable to 2-WL. Chen et al. (2020) studies the ability of GNNs to count graph substructures. Though our theorems are much more general, note that their results are improved by the present work. For example, Corrolary 6 in Chen et al. (2020) implies that 6-LGNN can count the number of 6-cycles, whereas our Theorem 4 implies that 3-LGNN or 2-FGNN can already count the number of 6-cycles. Note also, that if the nodes are given distinct features, MGNNs become much more expressive Loukas (2019) but this is meaningless in some problems such as our graph alignment problem.

Note that for neural networks on sets, the situation is a bit simpler. Efficient architectures such as DeepSets (Zaheer et al., 2017) or PointNet (Qi et al., 2017) have been shown to be invariant universal. Similar results exist in the equivariant case (Segol & Lipman, 2020; Maron et al., 2020), whose proofs rely on polynomial arguments. Though this is not our main motivation, our approximation theorems could also be applied in this context.

# 2.1 NOTATIONS: GRAPHS AS TENSORS

We denote by  $\mathbb{F},\mathbb{F}_0,\mathbb{F}_1,\ldots$  arbitrary finite-dimensional spaces of the form  $\mathbb{R}^p$  (for various values of  $p$ ) typically representing the space of features. Product of vectors in  $\mathbb{R}^p$  always refer to componentwise product. There are two ways to see graphs with features. First, graphs can be seen as tensors of order  $k$ $G\in \mathbb{F}^{n^k}$ . The classical representation of a graph by its (weighted) adjacency matrix for  $k = 2$  is a tensor of order 2 in  $\mathbb{R}^{n^2}$ . This case allows for features on edges by replacing  $\mathbb{R}^{n^2}$  with  $\mathbb{F}^{n^2}$  where  $\mathbb{F}$  is some  $\mathbb{R}^p$ . Second, graphs can also be represented by their discrete structure with an additional feature vector. More exactly, denote by  $\mathcal{G}_n$  the set of discrete graphs  $G = (V,E)$  with  $n$  nodes  $V = [n]$  and edges  $E\subset V^2$ . Such a  $G\in \mathcal{G}_n$  with a vector  $h^0\in \mathbb{F}^n$  represents a graphs with features on the vertices.

# 2.2 DEFINITIONS: INVARIANT AND EQUIVARIANT OPERATORS

Let  $[n] = \{1, \dots, n\}$ . The set of permutations on  $[n]$  is denoted by  $S_{n}$ . For  $G \in \mathbb{F}^{n^k}$  and  $\sigma \in S_{n}$ , we define:  $(\sigma \star G)_{\sigma(i_1), \dots, \sigma(i_k)} = G_{i_1, \dots, i_k}$ . Note that the  $\star$  operation is valid between a permutation in  $S_{n}$  and a graph  $G$  as soon as the number of nodes of  $G$  is  $n$ , i.e. it is valid for any order  $k$  of the graph. Two graphs  $G_1, G_2$  are said isomorphic if they have the same number of nodes and there exists a permutation  $\sigma$  such that  $G_1 = \sigma \star G_2$ .

Definition 1. A function  $f: \mathbb{F}_0^{n^k} \to \mathbb{F}_1$  is said to be invariant if  $f(\sigma \star G) = f(G)$  for every permutation  $\sigma \in S_n$  and every  $G \in \mathbb{F}_0^{n^k}$ . A function  $f: \mathbb{F}_0^{n^k} \to \mathbb{F}_1^{n^\ell}$  is said to be equivariant if  $f(\sigma \star G) = \sigma \star f(G)$  for every permutation  $\sigma \in S_n$  and every  $G \in \mathbb{F}_0^{n^k}$ .

Note that composing an equivariant function with an invariant function gives an invariant function. For  $k \geq 1$ , we define the invariant summation layer  $S^k: \mathbb{F}^{n^k} \to \mathbb{F}$  by  $S^k(G) = \sum_{\mathbf{i} \in [n]^k} G_{\mathbf{i}}$  for  $G \in \mathbb{F}^{n^k}$ . We also define the equivariant reduction layer  $S_1^k: \mathbb{F}^{n^k} \to \mathbb{F}^n$  as follows:  $S_1^k(G)_i = \sum_{1 \leq i_2 \ldots i_k \leq n} G_{i, i_2, \ldots i_k}$ . For message passing GNN, we will use the equivariant layer  $\operatorname{Id} + \lambda S^1: \mathbb{F}^n \to \mathbb{F}^n$  defined by,  $(\operatorname{Id} + \lambda S^1)(G)_i = G_i + \lambda S^1(G)$ , where  $\lambda \in \mathbb{R}$  is a learnable parameter.

In the sequel, we will need a mapping  $I^k$  lifting the input graph to a higher order tensor. We denote by  $I^k: \mathbb{F}_0^{n^2} \to \mathbb{F}_1^{n^k}$  the initialization function mapping for a given graph each  $k$ -tuple to its isomorphism type. We refer to the appendix §C.3 for a precise description of this linear equivariant function. Note at this stage that  $I^2$  is given by, for  $G \in \mathbb{F}^{n^2}$ ,  $I(G)_{i,j} = (G_{i,j}, \delta_{i,j})$  where  $\delta_{i,j}$  is 0 if  $i \neq j$  and 1 otherwise. Indeed for a pair of nodes  $i, j$  in a graph (without features), there are only three isomorphism types:  $i = j, i \neq j$  and  $(i,j)$  is an edge and  $i \neq j$  but  $(i,j)$  is not an edge.

# 3 GNN DEFINITIONS

In this section, we define the various GNN architectures studied in this paper. In all architectures, there is a main building block or layer mapping  $\mathbb{F}_t^{n^k}$  to  $\mathbb{F}_{t + 1}^{n^k}$  where  $\mathbb{F}_t^{n^k}$  can be seen as the space for the representation of the graph at layer  $t$ . We will define three different types of layers for message passing GNN, linear GNN and folklore GNN. The case  $k = 2$  is probably the most interesting case from a practical point view and corresponds to a case where a layer takes as input a graph (with features on nodes and edges) and produces as output a graph (with new features on nodes and edges). For each type of GNN, there will be an invariant and an equivariant version. All architectures will share the last function:  $m_I:\mathbb{F}_{T + 1}\to \mathbb{F}$  for the invariant case and  $m_E:\mathbb{F}_{T + 1}^n\to \mathbb{F}^n$  for the equivariant case which are continuous functions. It is typically modeled by a Multi Layer Perceptron, which is applied on each component for the equivariant case. In words, each network takes as input a graph  $G\in \mathbb{F}_0^{n^2}$ , produces in the invariant case a graph embedding in  $\mathbb{F}_{T + 1}$  and in the equivariant case a node embedding in  $\mathbb{F}_{T + 1}^n$ , then these embeddings are passed through the function  $m_I$  or  $m_E$  respectively to get a feature in  $\mathbb{F}$  or  $\mathbb{F}^n$  for the learning task.

# 3.1MESSAGEPASSINGGNN

Message passing GNN (MGNN) are defined for classical graphs  $G$  with features on the nodes. More exactly they take as input a discrete graph  $G = (V,E)\in \mathcal{G}_n$  and features on the nodes  $h^0\in \mathbb{F}^n$ . MGNN are then defined inductively as follows: let  $h_i^\ell \in \mathbb{F}_\ell$  denote the feature at layer  $\ell$  associated with node  $i$ , the updated features  $h_i^{\ell +1}$  are obtained as:  $h_i^{\ell +1} = f\left(h_i^\ell ,\{\{h_j^\ell \} \}_{j\sim i}\right)$ , where  $j\sim i$  means that nodes  $j$  and  $i$  are neighbors in the graph  $G$ , i.e.  $(i,j)\in E$ , and the function  $f$  is a learnable function taking as input the feature vector of the center vertex  $h_i^\ell$  and the multiset of features of the neighboring vertices  $\{\{h_j^\ell \} \}_{j\sim i}$ . Indeed, it follows from Lem. 33 in Appendix, that any such function  $f$  can be approximated by a layer of the form,

$$
h _ {i} ^ {\ell + 1} = f _ {0} \left(h _ {i} ^ {\ell}, \sum_ {j \sim i} f _ {1} \left(h _ {i} ^ {\ell}, h _ {j} ^ {\ell}\right)\right), \tag {1}
$$

where  $f_0: \mathbb{F}_\ell \times \mathbb{F} \to \mathbb{F}_{\ell+1}$  and  $f_1: \mathbb{F}_\ell \times \mathbb{F}_\ell \to \mathbb{F}$ , so that  $\mathbb{F}_\ell$  is the field for the features at the  $\ell$ -th layer. We call such a function a message passing layer and denote it by  $F: \mathbb{F}_\ell^n \to \mathbb{F}_{\ell+1}^n$  (note that  $F$  depends implicitly from the graph). Then an equivariant message passing GNN is simply obtained by the composition of message passing layers:  $F_T \circ \ldots \circ F_2 \circ F_1$ , where each  $F_i$  is a message passing layer. Clearly since each  $F_i$  is equivariant, this message passing GNN is also equivariant and produces features on each node in the space  $\mathbb{F}_T$ . In order to obtain an invariant GNN, we apply an invariant function to the output of an equivariant message passing GNN from  $\mathbb{F}_T^{n^2} \to \mathbb{F}_{T+1}$ . In practice, a symmetric function is applied on the vectors of features indexed by the nodes, typically

the sum of the features  $\sum_{i}(F_{T}\circ \ldots F_{2}\circ F_{1}(G))_{i}$  is taken as an invariant feature for the graph  $G$ . With our notation,  $S^1\circ F_T\circ \ldots F_2\circ F_1$  (where  $S^1$  was defined in Section2.2) defines an invariant message passing GNN.

Hence, we define the sets of message passing GNNs as follows:

$$
\operatorname {M G N N} _ {I} = \left\{m _ {I} \circ S ^ {1} \circ F _ {T} \circ \dots F _ {2} \circ F _ {1}, \forall T \right\}
$$

$$
\mathrm {M G N N} _ {E} = \left\{m _ {E} \circ \left(\operatorname {I d} + \lambda S ^ {1}\right) \circ F _ {T} \circ \dots F _ {2} \circ F _ {1}, \forall T \right\}
$$

where  $F_{t}:\mathbb{F}_{t}^{n}\to \mathbb{F}_{t + 1}^{n}$  are message passing layers.

# 3.2 LINEAR GNN

We define the linear graph layer of order  $k$  as  $F: \mathbb{F}_{\ell}^{n^k} \to \mathbb{F}_{\ell+1}^{n^k}$ , where for all  $G \in \mathbb{F}_{\ell}^{n^k}$ ,  $F(G) = f(L[G])$  where  $L: \mathbb{F}_{\ell}^{n^k} \to \mathbb{F}_{\ell}^{n^k}$  is a linear equivariant function, and  $f: \mathbb{F}_{\ell} \to \mathbb{F}_{\ell+1}$  is a learnable function applied on each of the  $n^k$  features and  $\mathbb{F}_{\ell}$  is the field for the features at the  $\ell$ -th layer.

We then define the sets of linear GNNs as follows:

$$
k \text {- L G N N} _ {I} = \left\{m _ {I} \circ S ^ {k} \circ F _ {T} \circ \dots F _ {2} \circ F _ {1} \circ I ^ {k}, \forall T \right\}
$$

$$
k \text {- L G N N} _ {E} = \left\{m _ {E} \circ S _ {1} ^ {k} \circ F _ {T} \circ \dots F _ {2} \circ F _ {1} \circ I ^ {k}, \forall T \right\}
$$

where  $I^{k}:\mathbb{F}_{0}^{n^{2}}\to \mathbb{F}_{1}^{n^{k}}$  defined in §2.2 and for  $t\geq 1$ ,  $F_{t}:\mathbb{F}_{t}^{n^{k}}\to \mathbb{F}_{t + 1}^{n^{k}}$  are linear equivariant layers.

# 3.3 FOLKLORE GNN

The main building block of Folklore GNN (FGNN) is what we call the folklore graph layer (FGL) of order  $k$  defined as follows: for  $k \geq 1$ ,  $F: \mathbb{F}_{\ell}^{n^k} \to \mathbb{F}_{\ell + 1}^{n^k}$  where for all  $G \in \mathbb{F}_{\ell}^{n^k}$  and all  $\mathbf{i} \in [n]^k$ ,

$$
F (G) _ {\mathbf {i}} = f _ {0} \left(G _ {\mathbf {i}}, \sum_ {j = 1} ^ {n} \prod_ {w = 1} ^ {k} f _ {w} \left(G _ {i _ {1}, \dots , i _ {w - 1}, j, i _ {w + 1}, \dots , i _ {k}}\right)\right), \tag {2}
$$

where  $f_0: \mathbb{F}_\ell \times \mathbb{F} \to \mathbb{F}_{\ell+1}$  and  $f_k: \mathbb{F}_\ell \to \mathbb{F}$  are learnable functions. As shown in Lem. 33 in Appendix, FGL is an equivariant function which is indeed very expressive.

For classical graphs  $G \in \mathbb{F}_0^{n^2}$ , we can now define 2-FGNN by composing folklore graph layers  $F_t: \mathbb{F}_t^{n^2} \to \mathbb{F}_{t+1}^{n^2}$ , so that  $F_T \circ \ldots \circ F_1 \circ F_0$  is an equivariant GNN producing a graph in  $\mathbb{F}_{T+1}^{n^2}$ . To obtain an invariant feature of the graph, we use the summation layer  $S^2$  defined in Section 2.2 so that  $S^2 \circ F_T \circ \ldots \circ F_1 \circ F_0$  is now an invariant 2-FGNN. In order to define general  $k$ -FGNN, we first need to lift the classical graph to a tensor in  $\mathbb{F}^{n^k}$ , then we apply folklore graph layers of order  $k$  and finally we need to project the tensor in  $\mathbb{F}^{n^k}$  to a tensor in  $\mathbb{F}^n$  for the equivariant version and to a tensor in  $\mathbb{F}$  for the invariant version. The first step is done with the linear equivariant function  $I^k: \mathbb{F}_0^{n^2} \to \mathbb{F}_1^{n^k}$  defined in Section 2.2. The last step is done with the reduction layer  $S_1^k$  for the equivariant case and the summation layer  $S^k$  for the invariant case, both defined in Section 2.2.

We define the sets of folklore GNNs as follows:

$$
k \text {- F G N N} _ {I} = \left\{m _ {I} \circ S ^ {k} \circ F _ {T} \circ \dots F _ {2} \circ F _ {1} \circ I ^ {k}, \forall T \right\}
$$

$$
k \text {- F G N N} _ {E} = \left\{m _ {E} \circ S _ {1} ^ {k} \circ F _ {T} \circ \dots F _ {2} \circ F _ {1} \circ I ^ {k}, \forall T \right\}
$$

where  $F_{t}:\mathbb{F}_{t}^{n^{k}}\to \mathbb{F}_{t + 1}^{n^{k}}$  are FGLs.

# 4 THEORETICAL RESULTS FOR GNNS

# 4.1 WEISFEILER-LEHMAN INVARIANT AND EQUIVARIANT VERSIONS

We introduce a family of functions on graphs parametrized by integers  $k \geq 2$  developed for the graph isomorphism problem and working with tuples of  $k$  vertices. Each  $k$ -tuple  $\mathbf{i} \in V^k = [n]^k$  is given

a color  $c^0 (\mathbf{i})$  corresponding to its isomorphism type (see Section B.2). The  $k$ -WL test relies on the following notion of neighborhood, defined by, for any  $w \in [k]$ , and  $\mathbf{i} = (i_1, \dots, i_k) \in V^k$ ,  $N_w(\mathbf{i}) = \{(i_1, \dots, i_{w-1}, j, i_{w+1}, \dots, i_k) : j \in V\}$ . Then, the colors of the  $k$ -tuples are refined as follows,  $c^{t+1}(\mathbf{i}) = \operatorname{Lex}(c^t(\mathbf{i}), (C_1^t(\mathbf{i}), \dots, C_k^t(\mathbf{i})))$  where, for  $w \in [k]$ ,  $C_w^t(\mathbf{i}) = \left\{\left\{c^t(\tilde{\mathbf{i}}) : \tilde{\mathbf{i}} \in N_w(\mathbf{i})\right\}\right\}$  and the function Lex means that all occurring colors are lexicographically ordered and replaced by an initial segment of the natural numbers.

For a graph  $G$ , let  $k\text{-WL}_I^T(G)$  denote the multiset of colors of the  $k$ -WL algorithm at the  $T^{th}$  iteration. After a finite number of steps (which depends on the number of vertices in the graph), the algorithm stops because a stable coloring is reached (no color class of  $k$ -tuples is further divided). We denote by  $k\text{-WL}_I(G)$  the multiset of colors in the stable coloring. This is a graph invariant that is usually used to test if graphs are isomorphic. Clearly, the power of this invariant increases with  $k$ .

We now define an equivariant version of  $k$ -WL test to express the discriminatory power of equivariant architectures. For this, we construct a coloring of the vertices from the coloring of the  $k$ -tuples given by the standard  $k$ -WL algorithm. Formally, define  $k\text{-WL}_E^T:\mathbb{F}_0^{n^2}\to \mathbb{F}^n$  by, for  $i\in V$ :  $k\text{-WL}_E^T (G)_i = \{\{c^T (\mathbf{i}):\mathbf{i}\in V^k,i_1 = i\} \}$ . Similarly, define  $k\text{-WL}_E(G) = \{\{c(\mathbf{i}):\mathbf{i}\in V^k,i_1 = i\}\}$  where  $c(\mathbf{i})$  is the stable coloring obtained by the algorithm.

# 4.2 SEPARATING POWER OF GNNS

We formulate our results using the equivalence relation introduced by Timofte (2005), which characterizes the separating power of a set of functions.

Definition 2. Let  $\mathcal{F}$  be a set of functions  $f$  defined on a set  $X$ , where each  $f$  takes its values in some  $Y_{f}$ . The equivalence relation  $\rho(\mathcal{F})$  defined by  $\mathcal{F}$  on  $X$  is: for any  $x, x' \in X$ ,

$$
(x, x ^ {\prime}) \in \rho (\mathcal {F}) \iff \forall f \in \mathcal {F}, f (x) = f \left(x ^ {\prime}\right).
$$

Given two sets of functions  $\mathcal{F}$  and  $\mathcal{E}$ , we say that  $\mathcal{F}$  is more separating (resp. strictly more separating) than  $\mathcal{E}$  if  $\rho(\mathcal{F}) \subset \rho(\mathcal{E})$  (resp.  $\rho(\mathcal{F}) \subsetneq \rho(\mathcal{E})$ ). Note that all the functions in  $\mathcal{F}$  and  $\mathcal{E}$  need to be defined on the same set but can take values in different sets. For example, we can easily see that for the  $k$ -WL algorithm defined above, the equivariant version is more separating than the invariant one.

Some properties of the WL hierarchy of tests can be rephrased with the notion of separating power. In particular, Cai et al. (1989) showed that  $(k + 1)\text{-WL}_I$  distinguishes strictly more than  $k\text{-WL}_I$ , which can be rewritten simply as (for a function  $f$ , we write  $\rho(f)$  for  $\rho(\{f\}))$

$$
\rho \left(\left(k + 1\right) - \mathrm {W L} _ {I}\right) \subsetneq \rho (k - \mathrm {W L} _ {I}). \tag {3}
$$

This notion of separating power enables us to concisely summarize the current knowledge about the discriminatory power of classes of GNN.

Proposition 3. We have, for  $k \geq 2$ ,

$$
\rho \left(M G N N _ {I}\right) = \rho (2 - W L _ {I}) \quad \rho \left(M G N N _ {E}\right) = \rho (2 - W L _ {E}) \tag {4}
$$

$$
\rho (k - L G N N _ {I}) = \rho (k - W L _ {I}) \quad \rho (k - L G N N _ {E}) \subset \rho (k - W L _ {E}) \tag {5}
$$

$$
\rho (k - F G N N _ {I}) = \rho ((k + 1) - W L _ {I}) \quad \rho (k - F G N N _ {E}) = \rho ((k + 1) - W L _ {E}) \tag {6}
$$

Only results about the invariant cases were previously known: (4) comes from Xu et al. (2018), (5) from Maron et al. (2018) Geerts (2020a) and one inclusion of (6) comes from Maron et al. (2019a). The equality in (6) for general  $k \geq 2$  is proved in Section C.

Note that for  $k = 2$ , all GNNs are dealing with tensors of order 2 i.e. with the adjacency matrix of the graph. However, the complexities of the various layers are quite different: for the message passing GNN, all computations are local (scaling with the maximum degree in the graph) and can be done in parallel; for the linear layer, there are only 15 linear functions from  $\mathbb{R}^{n^2} \to \mathbb{R}^{n^2}$  for all values of  $n$  (Maron et al., 2018); the folklore layer involves a (dense) matrix multiplication of shape  $n \times n$ . If 2-FGNN is the most complex architecture, we see that it has the best separating power among all architectures proposed so far dealing with tensors of order 2.

# 4.3 APPROXIMATION RESULTS FOR GNNS

For  $X,Y$  finite-dimensional spaces, let us denote by  $\mathcal{C}_I(X,Y),\mathcal{C}_E(X,Y)$ , respectively, the set invariant continuous functions and equivariant continuous functions from  $X$  to  $Y$ . The closure of a class of function  $\mathcal{F}$  for the uniform norm is denoted by  $\overline{\mathcal{F}}$ . Our result extends easily to graphs of varying sizes but this is deferred to Section F.2 for clarity.

The theorem below states in particular that the class  $k$ -FGNN can approximate any continuous function that is less separating than  $(k + 1)$ -WL in the invariant and in the equivariant cases.

Theorem 4. Let  $K_{discr} \subset \mathcal{G}_n \times \mathbb{F}_0^n$ ,  $K \subset \mathbb{F}_0^{n^2}$  be compact sets. For the invariant case, we have:

$$
\overline {{M G N N _ {I}}} = \left\{f \in \mathcal {C} _ {I} \left(K _ {\text {d i s c r}}, \mathbb {F}\right): \rho (2 - W L _ {I}) \subset \rho (f) \right\}
$$

$$
\overline {{k - L G N N _ {I}}} = \{f \in \mathcal {C} _ {I} (K, \mathbb {F}): \rho (k - W L _ {I}) \subset \rho (f) \}
$$

$$
\overline {{k - F G N N _ {I}}} = \{f \in \mathcal {C} _ {I} (K, \mathbb {F}): \rho ((k + 1) - W L _ {I}) \subset \rho (f) \}
$$

For the equivariant case, we have:

$$
\overline {{M G N N _ {E}}} = \left\{f \in \mathcal {C} _ {E} \left(K _ {\text {d i s c r}}, \mathbb {F} ^ {n}\right): \rho (2 - W L _ {E}) \subset \rho (f) \right\}
$$

$$
\overline {{k - L G N N _ {E}}} = \{f \in \mathcal {C} _ {E} (K, \mathbb {F} ^ {n}): \rho (k - L G N N _ {E}) \subset \rho (f) \} \supset \{f \in \mathcal {C} _ {E} (K, \mathbb {F} ^ {n}): \rho (k - W L _ {E}) \subset \rho (f) \}
$$

$$
\overline {{k - F G N N _ {E}}} = \{f \in \mathcal {C} _ {E} (K, \mathbb {F} ^ {n}): \rho ((k + 1) - W L _ {E}) \subset \rho (f) \}
$$

In the invariant case for  $k = 2$ , we have  $\overline{\mathrm{MGNN}_I} = \overline{2\text{-LGNN}_I} \subsetneq \overline{2\text{-FGNN}_I}$  where the strictness of the last inclusion comes from (3). In other words, 2-FGNN has a better power of approximation than the other architectures working with tensors of order 2. We already knew by Proposition 3 that 2-FGNN is the best separating architecture among those studied in this paper and dealing with tensor of order 2 and our theorem implies that this is also the case for the approximation power.

To clarify the meaning of theses statements, we explain why the inclusions “ $\subset$ ” are actually straightforward. For concreteness, we focus on  $\overline{k - \mathrm{FGNN}_I} \subset \{f \in \mathcal{C}_I(K, \mathbb{F}) : \rho((k + 1) - \mathrm{WL}_I) \subset \rho(f)\}$ . Take  $h \in \overline{k - \mathrm{FGNN}_I}$ , this means that there is a sequence  $\mathbf{GNN}_j \in k\text{-FGNN}_I$  such that,  $\sup_{G \in K} \| h(G) - \mathbf{GNN}_j(G) \|$  goes to zero when  $j$  goes to infinity.

Therefore,  $h$  is continuous and constant on each  $\rho (k\text{-FGNN}_I)$ -class. Indeed, for any  $(G,G^{\prime}) \in \rho (k\text{-FGNN}_I)$ ,  $\mathbf{G}\mathbf{N}\mathbf{N}_j(G) = \mathbf{G}\mathbf{N}\mathbf{N}_j(G^{\prime})$  so that  $h(G) = \lim_{i} \mathbf{G}\mathbf{N}\mathbf{N}_{j}(G) = \lim_{j} \mathbf{G}\mathbf{N}\mathbf{N}_{j}(G^{\prime}) = h(G^{\prime})$ . Hence we have  $\rho (k\text{-FGNN}_I) \subset \rho (h)$  and by Prop. 3,  $\rho (k\text{-FGNN}_I) = \rho ((k + 1)\text{-WL}_I)$ , allowing us to get the inclusion above.

On the contrary, the reverse inclusions “ $\supset$ ” are much more intricate but they are also the most valuable. For instance, consider the inclusion  $\overline{k - \mathrm{FGNN}_I} \supset \{f \in \mathcal{C}_I(K, \mathbb{F}) : \rho((k + 1) - \mathrm{WL}_I) \subset \rho(f)\}$ . If one wishes to learn a function  $h \in \mathcal{C}_I(K, \mathbb{F})$  with  $k\text{-FGNN}_I$ , this function must at least be approximable by the class of  $k\text{-FGNN}_I$ . Our theorem precisely guarantees that if  $h$  is less separating than  $k\text{-WL}_I$ , it can be approximated by  $k\text{-FGNN}_I$ :

$$
\forall \epsilon > 0, \exists \mathbf {G N N} \in k \text {- F G N N} _ {I}, \sup  _ {G \in K} \| h (G) - \mathbf {G N N} (G) \| \leq \epsilon .
$$

For this, we show a much more general version of the famous Stone-Weierstrass theorem (see Section D) which relates the separating power with the approximation power. Following the elegant idea of Maehara & Hoang (2019), we augment the input space to transform vector-valued equivariant functions into scalar invariant maps. Then, we apply a fine-grained approximation theorem from Timofte (2005). We also provide specialized versions of our abstract theorem in  $\S E$ , which can be easily used to determine the approximation capabilities of any deep learning architecture.

Our theorem has also implications for universality results like Maron et al. (2019b); Keriven & Peyre (2019). A class of GNN is said to be universal if its closure on a compact  $K$  is the whole  $\mathcal{C}_I(K,\mathbb{F})$  (or  $\mathcal{C}_E(K,\mathbb{F}^n)$ ). In particular, Thm. 4 implies that  $n$ -LGNN and  $n$ -FGNN are universal as  $n$ -WL distinguishes non-isomorphic graphs of size  $n$ . This recovers a result of Ravanbakhsh (2020) for LGNN. Moreover, we can leverage the extensive literature on the WL tests to give more subtle results. For instance, Cai et al. (1989, §8.2) show that, for planar graphs,  $O(\sqrt{n})$ -WL can distinguish non-isomorphic instances. Therefore,  $O(\sqrt{n})$ -LGNN or  $O(\sqrt{n})$ -FGNN achieve universality in the particular, yet common, case of planar graphs. On a more practical side, Fürer (2010, Thm. 4.5) shows that the spectrum of a graph is less separating than 3-WL so that functions of the spectrum can actually be well approximated by 2-FGNN.

# 5 QUADRATIC ASSIGNMENT PROBLEM

To empirically evaluate our results, we study the Quadratic Assignment Problem (QAP), a classical problem in combinatorial optimization. For  $A$ ,  $B$ $n \times n$  symmetric matrices, it consists in solving

$$
\operatorname {m a x i m i z e} \operatorname {t r a c e} \left(A X B X ^ {\top}\right), \text {s u b j e c t t o} X \in \Pi ,
$$

where  $\Pi$  is the set of  $n\times n$  permutation matrices. Many optimization problems can be formulated as QAP. An example is the network alignment problem, which consists in finding the best matching between two graphs, represented by their adjacency matrices  $A$  and  $B$ . Though QAP is known to be NP-hard, recent works such as Nowak et al. (2018) have investigated whether it can be solved efficiently w.r.t. a fixed input distribution. More precisely, Nowak et al. (2018) studied whether one can learn to solve this problem using a MGNN trained on a dataset of already solved instances. However, as shown below, both the baselines and their approach fail on regular graphs, a class of graph considered as particularly hard for isomorphism testing.

To remedy this weakness, we consider 2-FGNN $_E$ . We then follow the siamese method of (Nowak et al., 2018): given two graphs, our system produces an embedding in  $\mathbb{F}^n$  for each graph, where  $n$  is the number of nodes, which are then multiplied together to obtain a  $n \times n$  similarity matrix on nodes. A permutation is finally computed by solving a Linear Assignment Problem (LAP) with this resulting  $n \times n$  as cost matrix. We tested our architecture on two distributions: the Erdős-Rényi model and random regular graphs. The accuracy in matching the graphs is much improved compared to previous works. The experimental setup is described more precisely in §A.1.

![](images/176ca0cfda8379755b64c625a67abbac865ed8cd81b35ee79f5055165b85961d.jpg)  
Figure 1: Fraction of matched nodes for pairs of correlated graphs (with edge density 0.2) as a function of the noise, see Section A.1 for details.

![](images/ed509bed391d8c51763b69f054ab68b0ff6c76e77e358d5d84ea886f946e5572.jpg)

<table><tr><td>This work</td><td>SDP (Peng et al., 2010)</td></tr><tr><td>LowRankAlign (Feizi et al., 2016)</td><td>GNN (Nowak et al., 2018)</td></tr></table>

# 6 CONCLUSION

We derived the expressive power of various practical GNN architectures: message passing GNNs, linear GNNs and folklore GNN; both for their invariant and equivariant counterparts. Our results unify and extend the recent works in this direction. In particular, we are able to recover all the universality results proved for GNNs so far. Similarly to existing results in the literature, we do not deal here with the sizes of the embeddings constructed at different layers, i.e. the sizes of the spaces  $\mathbb{F}_{\ell}$ , and these sizes are supposed to grow to infinity with the number of nodes  $n$  in the graph. Obtaining bounds on the scaling of the sizes of the features to ensure that the results presented here are still valid is an interesting open question. We show that folklore GNNs have the best power of approximation among all GNNs studied here dealing with tensors of order 2. From a practical perspective, we demonstrate their improved performance on the QAP with a significant gap in performances compared to other approaches.

# REFERENCES

J.-Y. Cai, M. Furer, and N. Immerman. An optimal lower bound on the number of variables for graph identification. In Proceedings of the 30th Annual Symposium on Foundations of Computer Science, SFCS '89, pp. 612-617, USA, 1989. IEEE Computer Society. ISBN 0818619821. doi: 10.1109/SFCS.1989.63543. URL https://doi.org/10.1109/SFCS.1989.63543.  
Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph isomorphism testing and function approximation with gnns. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada, pp. 15868-15876, 2019. URL http://papers.nips.cc/paper/9718-on-the-equivalence-between-graph-isomorphism-testing-and-fur  
Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count sub-structures? CoRR, abs/2002.04025, 2020. URL https://arxiv.org/abs/2002.04025.  
George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in neural information processing systems, pp. 3844-3852, 2016.  
Brendan L Douglas. The weisfeiler-lehman method and graph isomorphism testing. arXiv preprint arXiv:1101.5211, 2011.  
Vijay Prakash Dwivedi, Chaitanya K Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.  
Soheil Feizi, Gerald T. Quon, Mariana Recamonde Mendoza, Muriel Medard, Manolis Kellis, and Ali Jabbabaie. Spectral alignment of networks. CoRR, abs/1602.04181, 2016. URL http://arxiv.org/abs/1602.04181.  
Martin Fürer. On the combinatorial power of the weisfeiler-lehman algorithm. In International Conference on Algorithms and Complexity, pp. 260-271. Springer, 2017.  
Martin Fürer. On the power of combinatorial and spectral invariants. Linear Algebra and its Applications, 432(9):2373 - 2380, 2010. ISSN 0024-3795. doi: https://doi.org/10.1016/j.laa.2009.07.019. URL http://www.sciencedirect.com/science/article/pii/S0024379509003620. Special Issue devoted to Selected Papers presented at the Workshop on Spectral Graph Theory with Applications on Computer Science, Combinatorial Optimization and Chemistry (Rio de Janeiro, 2008).  
Vikas K Garg, Stefanie Jegelka, and Tommi Jaakkola. Generalization and representational limits of graph neural networks. arXiv preprint arXiv:2002.06157, 2020.  
Floris Geerts. The expressive power of kth-order invariant graph networks. arXiv preprint arXiv:2007.12035, 2020a.  
Floris Geerts. Walk message passing neural networks and second-order graph neural networks. arXiv preprint arXiv:2006.09499, 2020b.  
Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., volume 2, pp. 729-734. IEEE, 2005.  
Martin Grohe. Descriptive Complexity, Canonisation, and Definable Graph Structure Theory. Lecture Notes in Logic. Cambridge University Press, 2017. doi: 10.1017/9781139028868.  
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359 - 366, 1989. ISSN 0893-6080. doi: https://doi.org/10.1016/0893-6080(89)90020-8. URL http://www.sciencedirect.com/science/article/pii/0893608089900208.

Nicolas Keriven and Gabriel Peyre. Universal invariant and equivariant graph neural networks. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada, pp. 7090-7099, 2019. URL http://papers.nips.cc/paper/8931-universal-invariant-and-equivariant-graph-neural-networks.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Johannes Klicpera, Janek Groß, and Stephan Gunnemann. Directional message passing for molecular graphs. arXiv preprint arXiv:2003.03123, 2020.  
Andreas Loukas. What graph neural networks cannot learn: depth vs width. arXiv preprint arXiv:1907.03199, 2019.  
Takanori Maehara and NT Hoang. A simple proof of the universality of invariant/equivariant graph neural networks. ArXiv, abs/1910.03802, 2019.  
Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph networks. arXiv preprint arXiv:1812.09902, 2018.  
Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. In Advances in Neural Information Processing Systems, pp. 2153-2164, 2019a.  
Haggai Maron, Ethan Fetaya, Nimrod Segol, and Yaron Lipman. On the universality of invariant networks. arXiv preprint arXiv:1901.09342, 2019b.  
Haggai Maron, Or Litany, Gal Chechik, and Ethan Fetaya. On learning sets of symmetric elements. CoRR, abs/2002.08599, 2020. URL https://arxiv.org/abs/2002.08599.  
J.R. Munkres. Topology. Featured Titles for Topology. Prentice Hall, Incorporated, 2000. ISBN 9780131816299. URL https://books.google.fr/books?id=XjoZAQAAIAAJ.  
Alex Nowak, Soledad Villar, Afonso S. Bandeira, and Joan Bruna. Revised note on learning quadratic assignment with graph neural networks. 2018 IEEE Data Science Workshop (DSW), pp. 1-5, 2018.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems 32, pp. 8024-8035. Curran Associates, Inc., 2019. URL http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.  
Jiming Peng, Hans D. Mittelmann, and Xiaoxue Li. A new relaxation framework for quadratic assignment problems based on matrix splitting. Mathematical Programming Computation, 2: 59-77, 2010.  
Charles Ruizhongtai Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 77-85. IEEE Computer Society, 2017. doi: 10.1109/CVPR.2017.16. URL https://doi.org/10.1109/CVPR.2017.16.  
Siamak Ravanbakhsh. Universal equivariant multilayer perceptrons. arXiv preprint arXiv:2002.02912, 2020.  
W. Rudin. Functional Analysis. International series in pure and applied mathematics. McGraw-Hill, 1991. ISBN 9780070542365. URL https://books.google.fr/books?id=Sh_vAAAAMAAJ.

Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Approximation ratios of graph neural networks for combinatorial problems. In Advances in Neural Information Processing Systems, pp. 4081-4090, 2019.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. Computational capabilities of graph neural networks. IEEE Trans. Neural Networks, 20(1):81-102, 2009. doi: 10.1109/TNN.2008.2005141. URL https://doi.org/10.1109/TNN.2008.2005141.  
Nimrod Segol and Yaron Lipman. On universal equivariant set networks. ArXiv, abs/1910.02421, 2020.  
Vlad Timofte. Stone--weierstrass theorems revisited. Journal of Approximation Theory, 136(1): 45 - 59, 2005. ISSN 0021-9045. doi: https://doi.org/10.1016/j.jat.2005.05.004. URL http://www.sciencedirect.com/science/article/pii/S0021904505001097.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018.  
Dmitry Yarotsky. Universal approximations of invariant maps by neural networks. CoRR, abs/1804.10306, 2018. URL http://arxiv.org/abs/1804.10306.  
Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. Deep sets. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, 4-9 December 2017, Long Beach, CA, USA, pp. 3391-3401, 2017. URL http://papers.nips.cc/paper/6931-deep-sets.
