# HOW POWERFUL ARE GRAPH NEURAL NETWORKS?

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph Neural Networks (GNNs) for representation learning of graphs broadly follow a neighborhood aggregation framework, where the representation vector of a node is computed by recursively aggregating and transforming feature vectors of its neighboring nodes. Many GNN variants have been proposed and have achieved state-of-the-art results on both node and graph classification tasks. However, despite GNNs revolutionizing graph representation learning, there is limited understanding of their representational properties and limitations. Here, we present a theoretical framework for analyzing the expressive power of GNNs in capturing different graph structures. Our results characterize the discriminative power of popular GNN variants, such as Graph Convolutional Networks and GraphSAGE, and show that they cannot learn to distinguish certain simple graph structures. We then develop a simple architecture that is provably the most expressive among the class of GNNs and is as powerful as the Weisfeiler-Lehman graph isomorphism test. We empirically validate our theoretical findings on a number of graph classification benchmarks, and demonstrate that our model achieves state-of-the-art performance

# 1 INTRODUCTION

Learning with graph structured data, such as molecules, social, biological, and financial networks, requires effective representation of their graph structure (Hamilton et al., 2017b). Recently, there has been a surge of interest in Graph Neural Network (GNN) approaches for representation learning of graphs (Li et al., 2016; Hamilton et al., 2017a; Kipf & Welling, 2017; Velickovic et al., 2018; Xu et al., 2018). GNNs broadly follow a recursive neighborhood aggregation (or message passing) scheme, where each node aggregates feature vectors of its neighbors to compute its new feature vector (Gilmer et al., 2017; Xu et al., 2018). After  $k$  iterations of aggregation, a node is represented by its transformed feature vector, which captures the structural information within the node's  $k$ -hop network neighborhood. The representation of an entire graph can then be obtained through pooling, for example, by summing the representation vectors of all nodes in the graph.

Many GNN variants with different neighborhood aggregation and graph-level pooling schemes have been proposed (Battaglia et al., 2016; Defferrard et al., 2016; Duvenaud et al., 2015; Hamilton et al., 2017a; Kearnes et al., 2016; Kipf & Welling, 2017; Li et al., 2016; Velickovic et al., 2018; Verma & Zhang, 2018; Ying et al., 2018; Zhang et al., 2018). Empirically, these GNNs have achieved state-of-the-art performance in many tasks such as node classification, link prediction, and graph classification. However, the design of new GNNs is mostly based on empirical intuition, heuristics, and experimental trial-and-error. There is little theoretical understanding of the properties and limitations of GNNs, and formal analysis of GNNs' representational capacity is limited.

Here, we present a theoretical framework for analyzing the representational power of GNNs. We formally characterize how expressive different GNN variants are in learning to represent and distinguish between different graph structures. Our framework is inspired by the close connection between GNNs and the Weisfeiler-Lehman (WL) graph isomorphism test (Weisfeiler & Lehman, 1968), a powerful test known to distinguish a broad class of graphs (Babai & Kucera, 1979). Similar to GNNs, the WL test iteratively updates a given node's feature vector by aggregating feature vectors of its network neighbors. What makes the WL test so powerful is its injective aggregation update that maps different node neighborhoods to different feature vectors. Our key insight is that a GNN can have as large discriminative power as the WL test if the GNN's aggregation scheme is highly expressive and can model injective functions.

To mathematically formalize the above insight, our framework first abstracts the feature vectors of a node's neighbors as a multiset, i.e., a set with possibly repeating elements. Then, the neighbor aggregation in GNNs can be abstracted as a function over the multiset. We rigorously study different variants of multiset functions and theoretically characterize their discriminative power, i.e., how well different aggregation functions can distinguish different multisets. The more discriminative the multiset function is, the more powerful the representational power of the underlying GNN.

Our main results are summarized as follows:

1) We show that GNNs are at most as powerful as the WL test in distinguishing graph structures.  
2) We establish conditions on the neighbor aggregation and graph pooling functions under which the resulting GNN is as powerful as the WL test.  
3) We identify graph structures that cannot be distinguished by popular GNN variants, such as GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017a), and we precisely characterize the kinds of graph structures such GNN-based models can capture.  
4) We develop a simple neural architecture, Graph Isomorphism Network (GIN), and show that its discriminative/representational power is equal to the power of the WL test.

We validate our theory via experiments on graph classification datasets, where the expressive power of GNNs is crucial to capture graph structures. In particular, we compare the performance of GNNs with various aggregation functions. Our results confirm that the most powerful GNN (our Graph Isomorphism Network (GIN)) has high representational power as it almost perfectly fits the training data, whereas the less powerful GNN variants often severely underfit the training data. In addition, the representationally more powerful GNNs outperform the others by test set accuracy and achieve state-of-the-art performance on many graph classification benchmarks.

# 2 PRELIMINARIES

We begin by summarizing some of the most common GNN models and, along the way, introduce our notation. Let  $G = (V,E)$  denote a graph with node feature vectors  $X_{v}$  for  $v\in V$ . Then are two tasks of interest: (1) Node classification, where each node  $v\in V$  has an associated label  $y_{v}$  and the goal is to learn a representation vector  $h_v$  of  $v$  such that  $v$ 's label can be predicted as  $y_{v} = f(h_{v})$ ; (2) Graph classification, where, given a set of graphs  $\{G_1,\dots,G_N\} \subseteq \mathcal{G}$  and their labels  $\{y_1,\dots,y_N\} \subseteq \mathcal{Y}$ , we aim to learn a representation vector  $h_G$  that helps predict the label of an entire graph,  $y_{G} = g(h_{G})$ .

Graph Neural Networks. GNNs use the graph structure and node features  $X_{v}$  to learn a representation vector of a node,  $h_v$ , or the entire graph,  $h_G$ . Modern GNNs follow a neighborhood aggregation strategy, where we iteratively update the representation of a node by aggregating representations of its neighbors. After  $k$  iterations of aggregation, a node's representation captures the structural information within its  $k$ -hop network neighborhood. Formally, the  $k$ -th layer of a GNN is

$$
a _ {v} ^ {(k)} = \operatorname {A G G R E G A T E} ^ {(k)} \left(\left\{h _ {u} ^ {(k - 1)}: u \in \mathcal {N} (v) \right\}\right), \quad h _ {v} ^ {(k)} = \operatorname {C O M B I N E} ^ {(k)} \left(h _ {v} ^ {(k - 1)}, a _ {v} ^ {(k)}\right), \tag {2.1}
$$

where  $h_v^{(k)}$  is the feature vector of node  $v$  at the  $k$ -th iteration/layer. We initialize  $h_v^{(0)} = X_v$ , and  $\mathcal{N}(v)$  is a set of nodes adjacent to  $v$ . The choice of AGGREGATE $^{(k)}(\cdot)$  and COMBINE $^{(k)}(\cdot)$  in GNNs is crucial. A number of architectures for AGGREGATE have been proposed. In Graph Convolutional Networks (GCN) (Kipf & Welling, 2017), AGGREGATE has been formulated as

$$
a _ {v} ^ {(k)} = \operatorname {M E A N} \left(\left\{\operatorname {R e L U} \left(W \cdot h _ {u} ^ {(k - 1)}\right), \forall u \in \mathcal {N} (v) \right\}\right) \tag {2.2}
$$

where  $W$  is a learnable matrix. In the pooling variant of GraphSAGE (Hamilton et al., 2017a), the mean operation in Eq. 2.2 is replaced by an element-wise max-pooling. The COMBINE step could be a concatenation followed by a linear mapping  $W \cdot \left[h_v^{(k-1)} \mid a_v^{(k)}\right]$  as in GraphSAGE. In GCN, the COMBINE step is omitted and the model simply aggregates node  $v$  with its neighbors as  $h_v^{(k)} = \text{AGGREGATE}\left(\left\{h_v^{(k-1)}, h_u^{(k-1)} : u \in \mathcal{N}(v)\right\}\right)$ .

For node classification, the node representation  $h_v^{(K)}$  of the final iteration is used for prediction. For graph classification, the READOUT function aggregates node features from the final iteration to

![](images/d9f9da95aebed701cf9104e899d989acfce04cf79ec3fe43f11023effb6dce37.jpg)  
Figure 1: Subtree structures at the blue nodes in Weisfeiler-Lehman graph isomorphism test. Two WL iterations can capture and distinguish the structure of rooted subtrees of height 2.

![](images/7072747ca43a9dfaec03b4cbca625139e34b6db3adb90e8d9482a0f60208be5b.jpg)

obtain the entire graph's representation  $h_G$ :

$$
h _ {G} = \operatorname {R E A D O U T} \left(\left\{h _ {v} ^ {(K)} \mid v \in G \right\}\right). \tag {2.3}
$$

READOUT can be a simple permutation invariant function such as summation or a more sophisticated graph-level pooling function (Ying et al., 2018; Zhang et al., 2018).

Weisfeiler-Lehman test. The graph isomorphism problem asks whether two graphs are topologically identical. This is a challenging problem: no polynomial-time algorithm is known for it yet (Garey, 1979; Garey & Johnson, 2002; Babai, 2016). Despite some corner cases (Cai et al., 1992), the Weisfeiler-Lehman (WL) test of graph isomorphism (Weisfeiler & Lehman, 1968) is an effective and computationally efficient test that distinguishes a broad class of graphs (Babai & Kucera, 1979). Its 1-dimensional form, "naive vertex refinement", is analogous to neighborhood aggregation in GNNs. Assuming each node has a categorical label<sup>1</sup>, the WL test iteratively (1) aggregates the labels of nodes and their neighborhoods, and (2) hashes the aggregated labels into unique new labels. The algorithm decides that two graphs are different if at some iteration their node labels are different.

Based on the WL test, Shervashidze et al. (2011) proposed the WL subtree kernel that measures the similarity between graphs. The kernel uses the counts of node labels at different iterations of the WL test as the feature vector of a graph. Intuitively, a node's label at the  $k$ -th iteration of WL test represents a subtree structure of height  $k$  rooted at the node (Figure 1). Thus, the graph features considered by the WL subtree kernel are essentially counts of different rooted subtrees in the graph.

# 3 THEORETICAL FRAMEWORK: OVERVIEW

We start with an overview of our framework for analyzing the expressive power of GNNs. A GNN recursively updates each node's feature vector to capture the network structure and features of other nodes around it, i.e., its rooted subtree structures (Figure 1). For notational simplicity, we can assign each feature vector a unique label  $\in \{a,b,c\ldots \}$ . Then, feature vectors of a set of neighboring nodes form a multiset: the same element can appear multiple times since different nodes can have identical feature vectors.

Definition 1 (Multiset). A multiset is a generalized concept of a set that allows multiple instances for its elements. More formally, a multiset is a 2-tuple  $X = (S, m)$  where  $S$  is the underlying set of  $X$  that is formed from its distinct elements, and  $m : S \to \mathbb{N}_{\geq 1}$  gives the multiplicity of the elements.

In order to analyze the representational power of a GNN, we analyze when a GNN maps two nodes into the same location in the embedding space. Intuitively, the most powerful GNN maps two nodes to the same location only if they have identical subtree structures with identical features on the corresponding nodes. Since subtree structures are defined recursively via node neighborhoods (Figure 1), we can reduce our analysis recursively to the question when a GNN maps two neighborhoods to the same embedding. The most powerful GNN would never map two different neighborhoods, i.e., multiset of feature vectors, to the same location. This means its aggregation scheme is injective. Thus, we abstract a GNN's aggregation scheme as a class of functions over multisets that its neural networks can represent, and analyze whether they are able to represent injective multiset functions. Next, we use this reasoning to develop a maximally powerful GNN. In Section 5, we study popular GNN variants and see that their aggregation schemes are inherently not injective and thus less powerful, but that they can capture other interesting properties of graphs.

# 4 GENERALIZING THE WL TEST WITH GRAPH NEURAL NETWORKS

Ideally, a representationally powerful GNN could distinguish different graphs by mapping them to different locations in the embedding space. This is, however, equivalent to solving graph isomorphism.

In our analysis, we characterize the representational capacity of GNNs via a slightly weaker criterion: the Weisfeiler-Lehman (WL) graph isomorphism test that is known to work well in general, with some few exceptions. Proofs of all lemmas and theorems can be found in the appendix.

Lemma 2. Let  $G_{1}$  and  $G_{2}$  be any non-isomorphic graphs. If a graph neural network  $\mathcal{A} : \mathcal{G} \to \mathbb{R}^{d}$  following the neighborhood aggregation scheme maps  $G_{1}$  and  $G_{2}$  to different embeddings, the Weisfeiler-Lehman graph isomorphism test also decides  $G_{1}$  and  $G_{2}$  are not isomorphic.

Hence, any aggregation-based GNN is at most as powerful as the WL test in distinguishing different graphs. A natural follow-up question is whether there exist GNNs that are, in principle, as powerful as the WL test? Our answer, in Theorem 3, is yes: if the neighbor aggregation and graph pooling functions are injective, then the resulting GNN is as powerful as the WL test.

Theorem 3. Let  $\mathcal{A}:\mathcal{G}\to \mathbb{R}^d$  be a GNN following the neighborhood aggregation scheme. With sufficient iterations,  $\mathcal{A}$  maps any graphs  $G_{1}$  and  $G_{2}$  that the Weisfeiler-Lehman test of isomorphism decides as non-isomorphic, to different embeddings if the following conditions hold:

a)  $\mathcal{A}$  aggregates and updates node features iteratively with

$$
h _ {v} ^ {(k)} = \phi \left(h _ {v} ^ {(k - 1)}, f \left(\left\{h _ {u} ^ {(k - 1)}: u \in \mathcal {N} (v) \right\}\right)\right) o r h _ {v} ^ {(k)} = f \left(\left\{h _ {v} ^ {(k - 1)}, h _ {u} ^ {(k - 1)}: u \in \mathcal {N} (v) \right\}\right)
$$

where the functions  $f$ , which operates on multiset, and  $\phi$  are injective.

b)  $\mathcal{A}$ 's graph-level readout, which operates on the multiset of node features  $\left\{h_v^{(k)}\right\}$ , is injective.

We prove Theorem 3 in the appendix. Generally note that GNNs have an important benefit over the WL test: node feature vectors in the WL test are one-hot encodings and thus cannot capture the similarity between subtrees. In contrast, a GNN satisfying the criteria in Theorem 3 generalizes the WL test by learning to embed the subtrees to continuous space. This enables GNNs to not only discriminate different structures, but also to learn to map similar graph structures to similar embeddings and capture dependencies between graph structures. Such learned embeddings are particularly helpful for generalization when the co-occurrence of subtrees is sparse across different graphs or there are noisy edges (Yanardag & Vishwanathan, 2015).

# 4.1 GRAPH ISOMORPHISM NETWORK (GIN)

Next we develop a model that provably satisfies the conditions in Theorem 3 and thus generalizes the WL test. We name the resulting architecture Graph Isomorphism Network (GIN).

To model injective multiset functions for the neighbor aggregation, we develop a theory of "deep multisets", i.e., parameterizing universal multiset functions with neural networks. Our next lemma states that sum aggregators can represent injective, in fact, universal functions over multisets.

Lemma 4. Assume  $\mathcal{X}$  is countable. There exists a function  $f:\mathcal{X}\to \mathbb{R}^n$  so that  $h(X) = \sum_{x\in X}f(x)$  is unique for each finite multiset  $X\subset \mathcal{X}$ . Moreover, any multiset function  $g$  can be decomposed as  $g(X) = \phi \left(\sum_{x\in X}f(x)\right)$  for some function  $\phi$ .

We prove Lemma 4 in the appendix. The proof extends the setting in (Zaheer et al., 2017) from sets to multisets. An important distinction between deep multisets and sets is that certain popular injective set functions, such as the mean aggregator, are not injective multiset functions. Thanks to the universal approximation theorem (Hornik et al., 1989; Hornik, 1991), we can use multi-layer perceptrons (MLPs) to model and learn  $f$  and  $\phi$  in Lemma 4 for universal injective embeddings. In practice, we model  $f^{(k + 1)}\circ \phi^{(k)}$  with one MLP, because MLPs can represent the composition of functions. In the first iteration, we do not need MLPs before summation if input features are one-hot encodings as their summation alone is injective. GIN updates node representations as

$$
h _ {v} ^ {(k)} = \operatorname {M L P} ^ {(k)} \left(h _ {v} ^ {(k - 1)} + \sum_ {u \in \mathcal {N} (v)} h _ {u} ^ {(k - 1)}\right) \tag {4.1}
$$

In contrast to GNNs, which combine a node's feature with its aggregated neighborhood feature as in Eq. 2.1, GIN does not have the combine step and simply aggregates the node along with its neighbors. Although not intuitive, Theorem 3 suggests this simple scheme is as powerful as more complicated ones. Experimentally we observe that such simplicity improves performance.

![](images/3418b2668c84f6815c84482e4ef9fb0f104f54361adf30194849e8a18d86c77c.jpg)  
Input

![](images/17673ad5e6b8bc3f00836fd494eeef1264f02ced0e8fa28f4f7e7fd1a9bc00a4.jpg)  
sum - multiset

![](images/772616208c1772e230d08096013477dd02d9264c89bd72499194a7bb7d6271fd.jpg)  
mean - distribution

![](images/89e257b82c3f8e36b71a44507377eb7d08643283a579f154d1a2899a20d0fd3f.jpg)  
max - set

![](images/7433486f3e216287f01d06da5afdad0927ac59f219fcd5f5c852fb6cdc5d83dd.jpg)  
Figure 2: Ranking by expressive power for sum, mean and max-pooling aggregators over a multiset. Left panel shows the input multiset and the three panels illustrate the aspects of the multiset a given aggregator is able to capture: sum captures the full multiset, mean captures the proportion/distribution of elements of a given type, and the max aggregator ignores multiplicities (reduces the multiset to a simple set).  
(a) Mean and Max both fail  
Figure 3: Examples of simple graph structures that mean and max-pooling aggregators fail to distinguish. Figure 2 gives reasoning about how different aggregators "compress" different graph structures/multisets.

![](images/64afdf1f5f686096c7373a8b89f4a7b6a51779bc6f1c16bef087221c764670e5.jpg)  
(b) Max fails

![](images/6b4c761433769df4f2a91b07d6451da13694cddfe4a3c18e4c442654f9dc53ef.jpg)  
(c) Mean and Max both fail

# 4.2 READOUT SUBTREE STRUCTURES OF DIFFERENT DEPHS

An important aspect of the graph-level readout is that node representations, corresponding to subtree structures, get more refined and global as the number of iterations increases. A sufficient number of iterations is key to achieving good discriminative power. Yet, features from earlier iterations may sometimes generalize better. To consider all structural information, GIN uses information from all depths/iterations of the model. We achieve this by an architecture similar to Jumping Knowledge Networks (JK-Nets) (Xu et al., 2018), where we replace Eq. 2.3 with graph representations concatenated across all iterations:

$$
h _ {G} = \operatorname {C O N C A T} \left(\operatorname {R E A D O U T} \left(\left\{h _ {v} ^ {(k)} | v \in G \right\}\right) \mid k = 0, 1, \dots , K\right). \tag {4.2}
$$

By Theorem 3 and Lemma 4, if GIN replaces READOUT in Eq. 4.2 with summing all node features from the same iterations (we do not need an extra MLP before summation for the same reason as in Eq. 4.1), it provably generalizes the WL test and the WL subtree kernel.

# 5 LESS POWERFUL BUT STILL INTERESTING GNNS

Next we study GNNs that do not satisfy the conditions in Theorem 3, including GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017a). We conduct ablation studies on two aspects of the aggregator in Eq. 4.1: (1) 1-layer perceptrons instead of MLPs and (2) mean or max-pooling instead of the sum. We will see that these GNN variants get confused by surprisingly simple graphs and are less powerful than the WL test. Nonetheless, models with mean aggregators like GCN perform well for node classification tasks. To better understand this, we precisely characterize what different GNN variants can and cannot capture about a graph and discuss the implications for learning with graphs.

# 5.1 1-LAYER PERCEPTRON IS INSUFFICIENT FOR CAPTURING STRUCTURES

The function  $f$  in Lemma 4 helps map distinct multisets to unique embeddings. It can be parameterized by an MLP by the universal approximation theorem (Hornik, 1991). Nonetheless, many existing GNNs instead use a 1-layer perceptron  $\sigma \circ W$  (Duvenaud et al., 2015; Kipf & Welling, 2017; Zhang et al., 2018), a linear mapping followed by a non-linear activation function such as a ReLU. Such 1-layer mappings are examples of Generalized Linear Models (Nelder & Wedderburn, 1972). Therefore, we are interested in understanding whether 1-layer perceptrons are enough for graph learning. Lemma 5 suggests that there are indeed network neighborhoods (multisets) that models with 1-layer perceptrons can never distinguish.

Lemma 5. There exist finite multiset  $X_{1} \neq X_{2}$  so that for any linear mapping  $W$ ,  $\sum_{x \in X_{1}} \operatorname{ReLU}(Wx) = \sum_{x \in X_{2}} \operatorname{ReLU}(Wx)$ .

The main idea of the proof for Lemma 5 is that 1-layer perceptrons can behave much like linear mappings, so the GNN layers degenerate into simply summing over neighborhood features. GNNs with 1-layer perceptrons lack representational capacity, and, as we will later see empirically, when applied to graph classification they may severely underfit, whereas GNNs with MLPs usually do not.

# 5.2 STRUCTURES THAT CONFUSE MEAN AND MAX-POOLING

What happens if we replace the sum in  $h(X) = \sum_{x \in X} f(x)$  with mean or max-pooling as in GCN and GraphSAGE? Mean and max-pooling aggregators are still well-defined multiset functions because they are permutation invariant. But, they are not injective. Figure 2 ranks the three aggregators by their representational power, and Figure 3 illustrates pairs of structures that the mean and max-pooling aggregators fail to distinguish. Here, node colors denote different node features, and we assume the GNNs aggregate neighbors first before combining them with the central node.

In Figure 3a, every node has the same feature  $a$  and  $f(a)$  is the same across all nodes (for any function  $f$ ). When performing neighborhood aggregation, the mean or maximum over  $f(a)$  remains  $f(a)$  and, by induction, we always obtain the same node representation everywhere. Thus, mean and max-pooling aggregators fail to capture any structural information. In contrast, a sum aggregator distinguishes the structures because  $2 \cdot f(a)$  and  $3 \cdot f(a)$  give different values. The same argument can be applied to any unlabeled graph. If node degrees instead of a constant value is used as node input features, in principle, mean can recover sum, but max-pooling cannot.

Fig. 3a suggests that mean and max have trouble distinguishing graphs with nodes that have repeating features. Let  $h_{\mathrm{color}}$  ( $r$  for red,  $g$  for green) denote node features transformed by  $f$ . Fig. 3b shows that maximum over the neighborhood of the blue nodes yields  $\max(h_{\mathrm{g}}, h_{\mathrm{r}})$  and  $\max(h_{\mathrm{g}}, h_{\mathrm{r}}, h_{\mathrm{r}})$ , which collapse to the same representation. Thus, max-pooling fails to distinguish them. In contrast, the sum aggregator still works because  $\frac{1}{2}(h_{\mathrm{g}} + h_{\mathrm{r}})$  and  $\frac{1}{3}(h_{\mathrm{g}} + h_{\mathrm{r}} + h_{\mathrm{r}})$  are in general not equivalent. Similarly, in Fig. 3c, both mean and max fail as  $\frac{1}{2}(h_{\mathrm{g}} + h_{\mathrm{r}}) = \frac{1}{4}(h_{\mathrm{g}} + h_{\mathrm{g}} + h_{\mathrm{r}} + h_{\mathrm{r}})$ .

# 5.3 MEAN LEARNS DISTRIBUTIONS

To characterize the class of multisets that the mean aggregator can distinguish, consider the example  $X_{1} = (S,m)$  and  $X_{2} = (S,k\cdot m)$ , where  $X_{1}$  and  $X_{2}$  have the same set of distinct elements, but  $X_{2}$  contains  $k$  copies of each element of  $X_{1}$ . Any mean aggregator maps  $X_{1}$  and  $X_{2}$  to the same embedding, because it simply takes averages over individual element features. Thus, the mean captures the distribution (proportions) of elements in a multiset, but not the exact multiset.

Corollary 6. Assume  $\mathcal{X}$  is countable. There exists a function  $f: \mathcal{X} \to \mathbb{R}^n$  so that for  $h(X) = \frac{1}{|X|} \sum_{x \in X} f(x)$ ,  $h(X_1) = h(X_2)$  if and only if finite multiset  $X_1$  and  $X_2$  have the same distribution. That is, assuming  $|X_2| \geq |X_1|$ , we have  $X_1 = (S, m)$  and  $X_2 = (S, k \cdot m)$  for some  $k \in \mathbb{N}_{\geq 1}$ .

The mean aggregator may perform well if, for the task, the statistical and distributional information in the graph is more important than the exact structure. Moreover, when node features are diverse and rarely repeat, the mean aggregator is as powerful as the sum aggregator. This may explain why, despite the limitations identified in Section 5.2, GNNs with mean aggregators are effective for node classification tasks, such as classifying article subjects and community detection, where node features are rich and the distribution of the neighborhood features provides a strong signal for the task.

# 5.4 MAX-POOLING LEARNS SETS WITH DISTINCT ELEMENTS

The examples in Figure 3 illustrate that max-pooling considers multiple nodes with the same feature as only one node (i.e., treats a multiset as a set). Max-pooling captures neither the exact structure nor the distribution. However, it may be suitable for tasks where it is important to identify representative elements or the "skeleton", rather than to distinguish the exact structure or distribution. Qi et al. (2017) empirically show that the max-pooling aggregator learns to identify the skeleton of a 3D point cloud and that it is robust to noise and outliers. For completeness, the next corollary shows that the max-pooling aggregator captures the underlying set of a multiset.

Corollary 7. Assume  $\mathcal{X}$  is countable. Then there exists a function  $f: \mathcal{X} \to \mathbb{R}^{\infty}$  so that for  $h(X) = \max_{x \in X} f(x)$ ,  $h(X_1) = h(X_2)$  if and only if  $X_1$  and  $X_2$  have the same underlying set.

# 6 EXPERIMENTS

We evaluate and compare the training and test performance of GIN and less powerful GNN variants.

Datasets. We use 9 graph classification benchmarks: 4 bioinformatics datasets (MUTAG, PTC, NCI1, PROTEINS) and 5 social network datasets (COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY and REDDIT-MULTI5K) (Kersting et al., 2016). In the bioinformatic graphs, the nodes have categorical input features; in the social networks, they have no features. For the REDDIT datasets, we set all node feature vectors to be the same (thus, features here are uninformative); for the other social graphs, we use one-hot encodings of node degrees. Dataset statistics are summarized in Table 1, and more details of the data can be found in Appendix G.

Models and configurations. We evaluate GIN (Eqs. 4.1 and 4.2) and five less powerful GNN variants: We consider architectures that replace the sum in Eq. 4.1 with mean or max-pooling $^2$ , or replace MLPs with 1-layer perceptrons, i.e., a linear mapping followed by ReLU. In Figure 4 and Table 1, a model is named by the aggregator/perceptron it uses. We apply the same graph-level readout (READOUT in Eq. 4.2) for GINs and all GNN variants, specifically, sum readout on bioinformatics datasets and mean readout on social datasets due to better test performance.

Following (Yanardag & Vishwanathan, 2015; Niepert et al., 2016), we perform 10-fold cross-validation with LIB-SVM (Chang & Lin, 2011), using 9 folds for training and 1 for testing. For all configurations, 5 GNN layers (including the input layer) are applied, and all MLPs have 2 layers. Batch normalization (Ioffe & Szegedy, 2015) is applied on every hidden layer. We use the Adam optimizer (Kingma & Ba, 2015) with initial learning rate 0.01 and decay the learning rate by 0.5 every 50 epochs. The hyper-parameters we tune for each dataset are: (1) The number of hidden units  $\in \{16,32\}$  for bioinformatics graphs and 64 for social graphs; (2) batch size  $\in \{32,128\}$ ; (3) dropout ratio  $\in \{0,0.5\}$  after the dense layer (Srivastava et al., 2014); (4) the number of epochs.

Baselines. We compare the GNNs above with a number of state-of-the-art baselines for graph classification: (1) The WL subtree kernel (Shervashidze et al., 2011), where  $C$ -SVM (Chang & Lin, 2011) was used as a classifier. The hyper-parameters we tune are  $C$  in SVM and the number of WL iterations  $\in \{1,2,\dots,6\}$ ; (2) state-of-the-art deep learning architectures Diffusion-convolutional neural networks (DCNN) (Atwood & Towsley, 2016), PATCHY-SAN (Niepert et al., 2016) and Deep Graph CNN (DGCNN) (Zhang et al., 2018); (3) Anonymous Walk Embeddings (AWL) (Ivanov & Burnaev, 2018). For the deep learning methods and AWL, we report the accuracies reported in the original papers.

# 6.1 RESULTS

Training set performance. We validate our theoretical analysis of the representational power of GNNs by comparing their training accuracies. Figure 4 shows training curves of GINs and less powerful GNN variants with the same hyper-parameter settings. First, the theoretically most powerful GNN, i.e. GIN (Sum-MLP), is able to almost perfectly fit all training sets. In comparison, the less powerful GNN variants severely underfit on many datasets. In particular, the training accuracy patterns align with our ranking by the models' representational power: GNN variants with MLPs tend to have higher training accuracies than those with 1-layer perceptrons, and GNNs with sum aggregators tend to fit the training sets better than those with mean and max-pooling aggregators.

On our datasets training accuracies of the GNNs, however, never exceed those of the WL subtree kernel, which has the same discriminative power as the WL test. For example, on IMDBBINARY, none of the models can perfectly fit the training set, and the GNNs achieve at most the same training accuracy as the WL kernel. This pattern aligns with our result that the WL test provides an upper bound for the representational capacity of the aggregation-based GNNs. Our theoretical results focus on representational power and do not yet take into account optimization (e.g., local minima). Nonetheless, the empirical results align very well with our theory.

![](images/aa013c2b3e1d4007e26533dc57952cfb032657139ab3b3003abb3ffffffd133d.jpg)

![](images/a46c2a731f421462b0e073f1c0df521e5b7e3be6ae33531129908367e07cc0fb.jpg)

![](images/f0a5fdd2732ee100450ce5692bc0eb53fd4695a0500d22d7f8241c9c7caa7737.jpg)  
WL kernel and GNN variants

![](images/68c190209f3b12cd1f405833b76cdfaf093573eab555b63b5130224d01503fcc.jpg)

![](images/8113c6c5ffd273c0241a5d523712bfce609d6c646a0c0fa13890355909faf3ef.jpg)

![](images/60cd8d904f8378e59f6ffc8f300f44342dbc65b94113bc84ed102b50fc21ae6a.jpg)

Figure 4: Training set performance of GINs, less powerful GNN variants, and the WL subtree kernel.  

<table><tr><td rowspan="4">Datasets</td><td>Datasets</td><td>IMDB-B</td><td>IMDB-M</td><td>RDT-B</td><td>RDT-M5K</td><td>COLLAB</td><td>MUTAG</td><td>PROTEINS</td><td>PTC</td><td>NCI1</td></tr><tr><td># graphs</td><td>1000</td><td>1500</td><td>2000</td><td>5000</td><td>5000</td><td>188</td><td>1113</td><td>344</td><td>4110</td></tr><tr><td># classes</td><td>2</td><td>3</td><td>2</td><td>5</td><td>3</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Avg # nodes</td><td>19.8</td><td>13.0</td><td>429.6</td><td>508.5</td><td>74.5</td><td>17.9</td><td>39.1</td><td>25.5</td><td>29.8</td></tr><tr><td rowspan="5">Baselines</td><td>WL subtree</td><td>73.8</td><td>50.9</td><td>81.0</td><td>52.5</td><td>78.9</td><td>90.4</td><td>75.0</td><td>59.9</td><td>86.0*</td></tr><tr><td>DCNN</td><td>49.1</td><td>33.5</td><td>-</td><td>-</td><td>52.1</td><td>67.0</td><td>61.3</td><td>56.6</td><td>62.6</td></tr><tr><td>PATCHYSAN</td><td>71.0</td><td>45.2</td><td>86.3</td><td>49.1</td><td>72.6</td><td>92.6*</td><td>75.9</td><td>60.0</td><td>78.6</td></tr><tr><td>DGCNN</td><td>70.0</td><td>47.8</td><td>-</td><td>-</td><td>73.7</td><td>85.8</td><td>75.5</td><td>58.6</td><td>74.4</td></tr><tr><td>AWL</td><td>74.6</td><td>51.6</td><td>87.9</td><td>54.7</td><td>73.9</td><td>87.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="6">GNN variants</td><td>GIN (SUM-MLP)</td><td>75.1</td><td>52.3</td><td>92.4</td><td>57.5</td><td>80.2</td><td>89.4</td><td>76.2</td><td>64.6</td><td>82.7</td></tr><tr><td>SUM-1-LAYER</td><td>74.1</td><td>52.2</td><td>90.0</td><td>55.1</td><td>80.6</td><td>90.0</td><td>76.2</td><td>63.1</td><td>82.0</td></tr><tr><td>MEAN-MLP</td><td>73.7</td><td>52.3</td><td>50.0†(71.2)</td><td>20.0†(41.3)</td><td>79.2</td><td>83.5</td><td>75.5</td><td>66.6</td><td>80.9</td></tr><tr><td>MEAN-1-LAYER</td><td>74.0</td><td>51.9</td><td>50.0†(69.7)</td><td>20.0†(39.7)</td><td>79.0</td><td>85.6</td><td>76.0</td><td>64.2</td><td>80.2</td></tr><tr><td>MAX-MLP</td><td>73.2</td><td>51.1</td><td>-</td><td>-</td><td>-</td><td>84.0</td><td>76.0</td><td>64.6</td><td>77.8</td></tr><tr><td>MAX-1-LAYER</td><td>72.3</td><td>50.9</td><td>-</td><td>-</td><td>-</td><td>85.1</td><td>75.9</td><td>63.9</td><td>77.7</td></tr></table>

Table 1: Classification accuracies  $(\%)$ .  $\dagger$  indicate test accuracies (equal to chance rates) when all nodes have the same feature vector. We also report in the parentheses the test accuracies when the node degrees are used as input node features. The best-performing GNNs are highlighted with boldface. On datasets where GIN's accuracy is not strictly the highest among GNN variants, GIN is comparable to the best because paired t-test at significance level  $10\%$  does not distinguish GIN from the best. If a baseline performs better than all GNNs, we highlight it with boldface and asterisk.

Test set performance. Next, we compare test accuracies. Although our theoretical results do not directly speak about generalization ability of GNNs, it is reasonable to expect that GNNs with strong expressive power can accurately capture graph structures of interest and thus generalize well. Table 1 compares test accuracies of GINs (Sum-MLP), other GNN variants, as well as the state-of-the-art baselines.

First, GINs outperform (or achieve comparable performance as) the less powerful GNN variants on all the 9 datasets, achieving state-of-the-art performance. In particular, GINs shine on the social network datasets, which contain a relatively large number of training graphs. On the Reddit datasets, a random vector was used as a node feature. Here GINs and sum-aggregation GNNs accurately capture the graph structure (as predicted in Section 5.2) and significantly outperform other models. Mean-aggregation GNNs, however, fail to capture structural information and do not perform better than random guessing. Even if node degrees are provided as input features, mean-based GNNs perform much worse than sum-based GNNs.

# 7 CONCLUSION

In this paper, we developed theoretical foundations for reasoning about expressive power of GNNs and proved tight bounds on the representational capacity of popular GNN variants. Along the way, we also designed a provably most powerful GNN under the neighborhood aggregation framework. An interesting direction for future work is to go beyond the neighborhood aggregation (or message passing) framework in order to pursue even more powerful architectures for learning with graphs. It would also be interesting to understand and improve the generalization properties of GNNs.

# REFERENCES

James Atwood and Don Towsley. Diffusion-convolitional neural networks. In Advances in Neural Information Processing Systems (NIPS), pp. 1993-2001, 2016.  
László Babai. Graph isomorphism in quasipolynomial time. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pp. 684-697. ACM, 2016.  
László Babai and Ludik Kucera. Canonical labelling of graphs in linear average time. In Foundations of Computer Science, 1979., 20th Annual Symposium on, pp. 39-46. IEEE, 1979.  
Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, et al. Interaction networks for learning about objects, relations and physics. In Advances in Neural Information Processing Systems (NIPS), pp. 4502-4510, 2016.  
Jin-Yi Cai, Martin Fürer, and Neil Immerman. An optimal lower bound on the number of variables for graph identification. Combinatorica, 12(4):389-410, 1992.  
Chih-Chung Chang and Chih-Jen Lin. Libsvm: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST), 2(3):27, 2011.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems (NIPS), pp. 3844-3852, 2016.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. pp. 2224-2232, 2015.  
Michael R Garey. A guide to the theory of np-completeness. Computers and intractability, 1979.  
Michael R Garey and David S Johnson. Computers and intractability, volume 29. wh freeman New York, 2002.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International Conference on Machine Learning (ICML), pp. 1273-1272, 2017.  
William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems (NIPS), pp. 1025-1035, 2017a.  
William L Hamilton, Rex Ying, and Jure Leskovec. Representation learning on graphs: Methods and applications. IEEE Data Engineering Bulletin, 40(3):52-74, 2017b.  
Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks, 4(2): 251-257, 1991.  
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (ICML), pp. 448-456, 2015.  
Sergey Ivanov and Evgeny Burnaev. Anonymous walk embeddings. In International Conference on Machine Learning (ICML), pp. 2191-2200, 2018.  
Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, and Patrick Riley. Molecular graph convolutions: moving beyond fingerprints. Journal of computer-aided molecular design, 30(8): 595-608, 2016.  
Kristian Kersting, Nils M. Kriege, Christopher Morris, Petra Mutzel, and Marion Neumann. Benchmark data sets for graph kernels, 2016. URL http://graphkernels.cs.tu-dortmund.de.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.  
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. In International Conference on Learning Representations (ICLR), 2016.  
J. A. Nelder and R. W. M. Wedderburn. Generalized linear models. Journal of the Royal Statistical Society, Series A, General, 135:370-384, 1972.  
Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. Learning convolutional neural networks for graphs. In International Conference on Machine Learning (ICML), pp. 2014-2023, 2016.  
Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 1(2):4, 2017.  
Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(Sep): 2539-2561, 2011.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations (ICLR), 2018.  
Saurabh Verma and Zhi-Li Zhang. Graph capsule convolutional neural networks. arXiv preprint arXiv:1805.08090, 2018.  
Boris Weisfeiler and AA Lehman. A reduction of a graph to a canonical form and an algebra arising during this reduction. *Nauchno-Technicheskaya Informatsia*, 2(9):12-16, 1968.  
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In International Conference on Machine Learning (ICML), pp. 5453-5462, 2018.  
Pinar Yanardag and SVN Vishwanathan. Deep graph kernels. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1365-1374. ACM, 2015.  
Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In Advances in Neural Information Processing Systems (NIPS), 2018.  
Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan R Salakhutdinov, and Alexander J Smola. Deep sets. In Advances in Neural Information Processing Systems, pp. 3391-3401, 2017.  
Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen. An end-to-end deep learning architecture for graph classification. In AAAI Conference on Artificial Intelligence, pp. 4438-4445, 2018.
