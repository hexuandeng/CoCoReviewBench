# MEMORY-BASED GRAPH NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph Neural Networks (GNNs) are deep models that operate on data with arbitrary topology represented as graphs. We introduce an efficient memory layer for GNNs that can jointly learn node representations and coarsen the graph. We also introduce two new networks based on this layer: Memory-Based Graph Neural Network (MemGNN) and Graph Memory Network (GMN) that can learn hierarchical graph representations. The experimental results show that the proposed models achieve state-of-the-art results in eight out of nine graph classification and regression benchmarks. We also show that the learned representations could correspond to chemical features in the molecule data.

# 1 INTRODUCTION

Graph Neural Networks (GNNs) (Wu et al., 2019; Zhou et al., 2018; Zhang et al., 2018) are a class of deep architectures that operate on data with arbitrary topology represented as graphs such as social networks (Kipf & Welling, 2016), knowledge graphs (Schlichtkrull et al., 2018), molecules (Duvenaud et al., 2015), point clouds (Hassani & Haley, 2019), and robots (Wang et al., 2019). Unlike regular-structured inputs with spatial locality such as grids (e.g., images and volumetric data) and sequences (e.g., speech and text), GNN inputs are variable-size graphs consisting of permutation-invariant nodes and interactions among them. GNNs such as Gated GNN (GGNN) (Li et al., 2015), Message Passing Neural Network (MPNN) (Gilmer et al., 2017), Graph Convolutional Network (GCN) (Kipf & Welling, 2016), and Graph Attention Network (GAT) (Velikovi et al., 2018) learn node embeddings through an iterative process of transferring, transforming, and aggregating the node embeddings from topological neighbors. Each iteration expands the receptive field by one hop and after  $k$  iterations the nodes within  $k$  hops influence the node embeddings of one another. GNNs are shown to learn better representations compared to random walks (Grover & Leskovec, 2016; Perozzi et al., 2014), matrix factorization (Belkin & Niyogi, 2002; Ou et al., 2016), kernel methods (Shervashidze et al., 2011; Kriege et al., 2016), and probabilistic graphical models (Dai et al., 2016).

These models, however, cannot learn hierarchical representation as they do not exploit the graph compositionality. Recent work such as Differentiable Pooling (DiffPool) (Ying et al., 2018), Top-KPool (Gao & Ji, 2019), and Self-Attention Graph Pooling (SAGPool) (Lee et al., 2019) define parametric graph pooling layers that let models learn hierarchical graph representation by stacking interleaved layers of GNN and pooling layers. These layers cluster nodes in the latent space such that the clusters are meaningful with respect to the task. These clusters might be communities in a social network or potent functional groups within a chemical dataset. Nevertheless, these models are not efficient as they require an iterative process of message passing after each pooling layer.

In this paper, we introduce a memory layer for joint graph representation learning and graph coarsening that consists of a multi-head array of memory keys and a convolution operator to aggregate the soft cluster assignments from different heads. The queries to a memory layer are node embeddings from the previous layer and the outputs are the node embeddings of the coarsened graph. The memory layer does not explicitly require connectivity information and unlike GNNs relies on the global information rather than local topology. These properties make them more efficient and improve their performance. We also introduce two networks based on the proposed layer: Memory-based Graph Neural Network (MemGNN) and Graph Memory Network (GMN). MemGNN consists of a GNN that learns the initial node embeddings, and a stack of memory layers that learns hierarchical graph representation up to the global graph embedding. GMN, on the other hand, learns the hierarchical representation purely based on memory layers and hence does not require message passing.

# 2 RELATED WORK

Memory Augmented Neural Networks (MANNs) utilize external memory with differentiable read-write operators allowing them to explicitly access the past experiences and are shown to enhance reinforcement learning (Pritzel et al., 2017), meta learning (Santoro et al., 2016), few-shot learning (Vinyals et al., 2016), and multi-hop reasoning (Weston et al., 2015). Unlike RNNs, in which the memory is represented within their hidden states, the decoupled memory in MANNs lets them to store and retrieve longer term memories with less parameters. The memory can be implemented as a key-value memory such as neural episodic control (Pritzel et al., 2017) and product-key memory layers (Lample et al., 2019) or as a array-structured memory such as Neural Turing Machine (NTM) (Graves et al., 2014), prototypical networks (Snell et al., 2017), memory networks (Weston et al., 2015), and Sparse Access Memory (SAM) (Rae et al., 2016). Our memory layer consists of a multi-head array of memory keys.

Graph Neural Networks (GNNs) use message passing to learn node embeddings over graphs. GraphSAGE (Hamilton et al., 2017b) learns embedding by sampling and aggregating neighbor nodes whereas GAT (Velikovi et al., 2018) uses attention to aggregate embeddings from all neighbors. GCN models extend the convolution to arbitrary topology. Spectral GCNs (Bruna et al., 2014; Defferrard et al., 2016; Kipf & Welling, 2016) use spectral filters over graph Laplacian to define the convolution in the Fourier domain. These models are less efficient compared to spatial GCNs (Schlichtkrull et al., 2018; Ma et al., 2019) which directly define the convolution on graph patches centered on nodes. Our memory layer uses a feed-forward network to learn the node embeddings.

Graph pooling can be done globally or hierarchically. In former, node embeddings are aggregated into a graph embedding using arithmetic operators such as sum or max (Hamilton et al., 2017a; Kipf & Welling, 2016) or set neural networks such as Set2Set (Vinyals et al., 2015) and SortPool (Morris et al., 2019). In latter, graphs are coarsened in each layer to capture the hierarchical structure. Non-parametric methods such as clique pooling (Luzhnica et al., 2019), kNN pooling (Wang et al., 2018), and Graclus (Dhillon et al., 2007) rely on topological information and are efficient, but are outperformed by parametric models such as edge contraction pooling (Diehl, 2019).

DiffPool (Ying et al., 2018) trains two parallel GNNs to compute node embeddings and cluster assignments using a combination of classification loss, link prediction loss, and entropy loss, whereas Mincut pool (Bianchi et al., 2019) trains a sequence of a GNN and an MLP using classification loss and the minimum cut objective. TopKPool (Cangea et al., 2018; Gao & Ji, 2019) computes a node score by learning a projection vector and then drops all the nodes except the top scoring nodes. SAGPool (Lee et al., 2019) extends the TopKPool by using graph convolutions to take neighbor node features into account. We use a clustering-friendly distribution to compute the attention scores between nodes and clusters.

# 3 METHOD

# 3.1 MEMORY LAYER

We define a memory layer  $\mathcal{M}^{(l)}: \mathbb{R}^{n_l \times d_l} \longmapsto \mathbb{R}^{n_{l+1} \times d_{l+1}}$  in layer  $l$  as a parametric function that takes in  $n_l$  query vectors of size  $d_l$  and generates  $n_{l+1}$  query vectors of size  $d_{l+1}$  such that  $n_{l+1} < n_l$ . The input and output queries represent the node features of the input graph and the coarsened graph, respectively. The memory layer learns to jointly coarsen the input nodes (i.e., pooling) and transform their features (i.e., representation learning). As shown in Figure 1, it consists of arrays of memory keys (i.e., multi-head memory) and a convolutional layer. Assuming  $|h|$  memory heads, a shared input query is compared against all the keys in each head resulting in  $|h|$  attention matrices which are then aggregated into a single attention matrix using the convolution layer.

In a content addressable memory (Graves et al., 2014; Sukhbaatar et al., 2015; Weston et al., 2015), the task of attending to memory (i.e., addressing scheme) is formulated as computing the similarly between memory keys to a given query  $q$ . Specifically, the attention weight of key  $k_{j}$  for query  $q$  is defined as  $w_{j} = \text{softmax}(d(q, k_{j}))$  where  $d$  is a similarity measure, typically Euclidean distance or cosine similarity (Rae et al., 2016). The soft read operation on memory is defined as a weighted average over the memory keys:  $r = \sum_{j} w_{j} k_{j}$ .

![](images/b95573fdff0ec12c9b1c18f1700b81e68ac6584772d00d0f0c06047ed38d64a9.jpg)  
Figure 1: The proposed Architecture for hierarchical graph representation learning using the introduced memory layer. The query network projects the initial node attributes to a query embedding space and each memory layer jointly coarsens the input queries and transforms them into a new query space.

In this work, we treat the input queries  $\mathbf{Q}^{(l)}\in \mathbb{R}^{n_l\times d_l}$  as the node embeddings of an input graph and treat the keys  $\mathbf{K}^{(l)}\in \mathbb{R}^{n_{l + 1}\times d_l}$  as the cluster centroids of the queries. To satisfy this assumption, we impose a clustering-friendly distribution as the distance metric between keys and a query. Following (Xie et al., 2016; Maaten & Hinton, 2008), we use the Student's t-distribution as a kernel to measure the normalized similarity between query  $q_{i}$  and key  $k_{j}$  as follows:

$$
C _ {i, j} = \frac {\left(1 + \left\| q _ {i} - k _ {j} \right\| ^ {2} / \tau\right) ^ {- \frac {\tau + 1}{2}}}{\sum_ {j ^ {\prime}} \left(1 + \left\| q _ {i} - k _ {j ^ {\prime}} \right\| ^ {2} / \tau\right) ^ {- \frac {\tau + 1}{2}}} \tag {1}
$$

where  $C_{ij}$  is the normalized score between query  $q_i$  and key  $k_j$  (i.e., probability of assigning node  $i$  to cluster  $j$  or attention score between query  $q_i$  and memory key  $k_j$ ) and  $\tau$  is the degree of freedom of the Student's t-distribution (i.e., temperature).

To increase the model capacity, we model the memory keys as a multi-head array. Applying a shared input query against the memory keys produces a tensor of cluster assignments  $\left[\mathbf{C}_0^{(l)}\dots \mathbf{C}_{|h|}^{(l)}\right] \in \mathbb{R}^{|h|\times n_{l + 1}\times n_l}$  where  $|h|$  denotes the number of heads. To aggregate the heads into a single assignment matrix, we treat the heads and the matrix rows and columns as depth, height, and width in standard convolution analogy and apply a convolution operator over them. Because there is no spatial structure, we use  $[1\times 1]$  convolution to aggregate the information across heads and therefore the convolution behaves as a weighted pooling that reduces the heads to a single matrix. The aggregated assignment matrix is computed as follows:

$$
\mathbf {C} ^ {(l)} = \operatorname {s o f t m a x} \left(\Gamma_ {\phi} \binom {| h |} {\|} \mathbf {C} _ {k} ^ {(l)}\right) \in \mathbb {R} ^ {n _ {l} \times n _ {l + 1}} \tag {2}
$$

where  $\Gamma_{\phi}$  is a  $[1\times 1]$  convolutional operator parametrized by  $\phi, ||$  is the concatenation operator, and  $\mathbf{C}^{(l)}$  is the aggregated soft assignment matrix.

A memory read generates a value matrix  $\mathbf{V}^{(l)}\in \mathbb{R}^{n_{l + 1}\times d_l}$  that represents the coarsened node embeddings in the same space as the input queries and is defined as the product of the soft assignment scores and the original queries as follows:

$$
\mathbf {V} ^ {(l)} = \mathbf {C} ^ {(l) \top} \mathbf {Q} ^ {(l)} \in \mathbb {R} ^ {n _ {l + 1} \times d _ {l}} \tag {3}
$$

The value matrix is fed to a single layer neural network consisting of a weight matrix  $\mathbf{W} \in \mathbb{R}^{d_l \times d_{l+1}}$  and a LeakyReLU activation function to project the coarsened embeddings from  $\mathbb{R}^{n_{l+1} \times d_l}$  into  $\mathbb{R}^{n_{l+1} \times d_{l+1}}$  representing the output queries  $\mathbf{Q}^{(l+1)}$ :

$$
\mathbf {Q} ^ {(l + 1)} = \text {L e a k y R e L U} \left(\mathbf {V} ^ {(l)} \mathbf {W}\right) \in \mathbb {R} ^ {n _ {l + 1} \times d _ {l + 1}} \tag {4}
$$

Thanks to these parametrized transformations, a memory layer can jointly learn the node embeddings and coarsens the graph end-to-end. The computed queries  $\mathbf{Q}^{(l + 1)}$  are the input queries to the

subsequent memory layer  $\mathcal{M}^{(l + 1)}$ . For graph classification, one can simply stack layers of memory up to the level where the input graph is coarsened into a single node representing the global graph embedding and then feed it to a fully-connected layer to predict the graph class as follows:

$$
\mathcal {Y} = \operatorname {s o f t m a x} \left(\mathrm {M L P} \left(\mathcal {M} ^ {(l)} \left(\mathcal {M} ^ {(l - 1)} \left(\dots . \mathcal {M} ^ {(0)} \left(\mathbf {Q} _ {0}\right)\right)\right)\right)\right) \tag {5}
$$

where  $\mathbf{Q}_0 = f_q(g)$  is the initial query embedding<sup>1</sup> generated by the query network  $f_{q}$  over graph  $g$ . We introduce two architectures based on the memory layer: GMN and MemGNN. These two architectures are different in the way that the query network is implemented. More specifically, GMN uses a feed-forward network for initializing the query:  $f_{q}(g) = \mathrm{FFN}_{\theta}(g)$ , whereas MemGNN implements the query network as a message passing GNN:  $f_{q}(g) = \mathrm{GNN}_{\theta}(g)$ .

# 3.2 GMN ARCHITECTURE

A GMN is a stack of memory layers on top of a query network  $f_{q}(g)$  that generates the initial query embeddings without any message passing. Similar to set neural networks (Vinyals et al., 2015) and transformers (Vaswani et al., 2017), graph nodes in a GMN are treated as a permutation-invariant set of embeddings. The query network projects the initial node attributes into an embedding space that represents the initial query space.

Assume a training set  $\mathcal{D} = [g_1, g_2, \dots, g_N]$  of  $N$  graphs where each graph is represented as  $g = (\mathbf{A}, \mathbf{X}, Y)$  and  $\mathbf{A} \in \{0, 1\}^{n \times n}$  denotes the adjacency matrix,  $\mathbf{X} \in \mathbb{R}^{n \times d_{in}}$  is the initial node attribute, and  $Y \in \mathbb{R}^n$  is the graph label. Considering that the GMN model treats a graph as a set of permutation-invariant nodes and does not use message passing, and also considering that the memory layers do not rely on connectivity information, the topological information of each node should be somehow encoded into its initial embedding. Inspired by transformers (Vaswani et al., 2017), we encode this information along with the initial attribute into the initial query embeddings using a query network  $f_q$  implemented as a two-layer feed-forward neural network:

$$
\mathbf {Q} ^ {(0)} = \text {L e a k y R e L U} \left(\left[ \text {L e a k y R e L U} \left(\mathbf {A} \mathbf {W} _ {0}\right) \| \mathbf {X} \right] \mathbf {W} _ {1}\right) \tag {6}
$$

where  $\mathbf{W}_0\in \mathbb{R}^{n\times d_{in}}$  and  $\mathbf{W}_1\in \mathbb{R}^{2d_{in}\times d_0}$  are the parameters of the query networks, and  $||$  is the concatenation operator.

# 3.3 MEMGNN ARCHITECTURE

Unlike the GMN architecture, the query network in MemGNN relies on the iterative process of passing messages and aggregating them to compute the initial query  $\mathbf{Q}_0$ :

$$
\mathbf {Q} ^ {(0)} = G _ {\theta} (\mathbf {A}, \mathbf {X}) \tag {7}
$$

where query network  $G_{\theta}$  is an arbitrary parameterized message passing GNN (Gilmer et al., 2017; Li et al., 2015; Kipf & Welling, 2016; Velikovi et al., 2018). In our implementation of MemGNN, we use a modified variant of GAT (Velikovi et al., 2018). Specifically, we introduce an extension to the original GAT model called edge-based GAT (e-GAT) and use it as the query network. Unlike GAT, e-GAT learns attention weights not only from the neighbor nodes but also from the input edge attributes. This is especially important for data containing edge information (e.g., various bonds among atoms represented as edges in molecule datasets). In an e-GAT layer, attention score between two neighbor nodes is computed as follows.

$$
\alpha_ {i j} = \frac {\exp \left(\text {L e a k y R e L U} \left(\mathbf {W} \left[ \mathbf {W} _ {n} h _ {i} ^ {(l)} \| \mathbf {W} _ {n} h _ {j} ^ {(l)} \| \mathbf {W} _ {e} h _ {i \rightarrow j} ^ {(l)} \right]\right)\right)}{\sum_ {k \in \mathcal {N} _ {i}} \exp \left(\text {L e a k y R e L U} \left(\mathbf {W} \left[ \mathbf {W} _ {n} h _ {i} ^ {(l)} \| \mathbf {W} _ {n} h _ {k} ^ {(l)} \| \mathbf {W} _ {e} h _ {i \rightarrow k} ^ {(l)} \right]\right)\right)} \tag {8}
$$

where  $h_i^{(l)}$  and  $h_{i\rightarrow j}^{(l)}$  denote the embedding of node  $i$  and the embedding of the edge connecting node  $i$  to its one-hop neighbor node  $j$  in layer  $l$ , respectively.  $\mathbf{W}_n$  and  $\mathbf{W}_e$  are trainable node and edge weights and  $\mathbf{W}$  is the parameter of a single-layer feed-forward network that computes the attention score.

# 3.4 TRAINING

We jointly train the model using two loss functions: a supervised classification loss and an unsupervised clustering loss. The supervised loss denoted as  $\mathcal{L}_{ent}$  is defined as the cross-entropy loss between the predicted and true graph class labels. The unsupervised clustering loss is inspired by deep clustering methods (Razavi et al., 2019; Xie et al., 2016; Aljalbout et al., 2018). It encourages the model to learn clustering-friendly embeddings in the latent space by urging it to learn from high confidence assignments with the help of an auxiliary target distribution. The unsupervised loss is defined as the Kullback-Leibler (KL) divergence loss between the soft assignments  $\mathbf{C}^{(l)}$  and the auxiliary distribution  $\mathbf{P}^{(l)}$  as follows:

$$
\mathcal {L} _ {K L} ^ {(l)} = \mathrm {K L} \left(\mathbf {P} ^ {(l)} | | \mathbf {C} ^ {(l)}\right) = \sum_ {i} \sum_ {j} P _ {i j} ^ {(l)} \log \frac {P _ {i j} ^ {(l)}}{C _ {i j} ^ {(l)}} \tag {9}
$$

For the target distributions  $\mathbf{P}^{(l)}$ , we use the distribution proposed in (Xie et al., 2016) which normalizes the loss contributions and improves the cluster purity while emphasizing on the samples with higher confidence. This distribution is defined as follows:

$$
P _ {i j} ^ {(l)} = \frac {\left(C _ {i j} ^ {(l)}\right) ^ {2} / \sum_ {i} C _ {i j} ^ {(l)}}{\sum_ {j ^ {\prime}} \left(C _ {i j ^ {\prime}} ^ {(l)}\right) ^ {2} / \sum_ {i} C _ {i j ^ {\prime}} ^ {(l)}} \tag {10}
$$

We define the total loss as follows where  $L$  is the number of memory layers and  $\lambda$  is a scalar weight.

$$
\mathcal {L} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\lambda \mathcal {L} _ {\text {e n t}} + (1 - \lambda) \sum_ {l = 1} ^ {L} \mathcal {L} _ {K L} ^ {(l)}\right) \tag {11}
$$

We initialize the model parameters, the keys, and the queries randomly and optimize them jointly with respect to  $\mathcal{L}$  using mini-batch stochastic gradient descent. To stabilize the training, the gradients of  $\mathcal{L}_{ent}$  are back-propagated batch-wise while the gradients of  $\mathcal{L}_{KL}^{(l)}$  are applied epoch-wise by periodically switching  $\lambda$  between 0 and 1. Moreover, updating the centroids (i.e., memory keys) with the same frequency as the network parameters can destabilize the training. To address this, we optimize all model parameters and the queries in each batch with respect to  $\mathcal{L}_{ent}$  and in each epoch with respect to  $\mathcal{L}_{KL}^{(l)}$ . Memory keys, on the other hand, are only updated at the end of each epoch by the gradients of  $\mathcal{L}_{KL}^{(l)}$ . This technique has also been applied in (Hassani & Haley, 2019; Caron et al., 2018) to avoid trivial solutions in deep clustering problem.

# 4 EXPERIMENTS

# 4.1 DATASETS

We use nine graph benchmarks including seven classification and two regression datasets to evaluate the proposed method. These datasets are commonly used in both graph kernel (Borgwardt & Kriegel, 2005; Yanardag & Vishwanathan, 2015; Shervashidze et al., 2009; Ying et al., 2018; Shervashidze et al., 2011; Kriege et al., 2016) and GNN (Cangea et al., 2018; Ying et al., 2018; Lee et al., 2019; Gao & Ji, 2019) literature. The summary of these datasets is as follows (i.e., first two benchmarks are regression tasks and the rest are classification tasks):

ESOL (Delaney, 2004) contains water solubility data for compounds.

Lipophilicity (Gaulton et al., 2016) contains experimental results of octanol/water distribution of compounds.

Bace (Subramanian et al., 2016) provides quantitative binding results for a set of inhibitors of human  $\beta$ -secretase 1 (BACE-1).

DD (Dobson & Doig, 2003) is used to distinguish enzyme structures from non-enzymes.

Enzymes (Schomburg et al., 2004) is for predicting functional classes of enzymes.

Proteins (Dobson & Doig, 2003) is used to predict the protein function from structure.

COLLAB (Yanardag & Vishwanathan, 2015) is for predicting the field of a researcher given her

Table 1: Mean validation accuracy over 10-folds.  

<table><tr><td rowspan="2">Method</td><td colspan="4">Dataset</td></tr><tr><td>Enzymes</td><td>Proteins</td><td>DD</td><td>COLLAB</td></tr><tr><td>Graphlet (Shervashidze et al., 2009)</td><td>41.03</td><td>72.91</td><td>64.66</td><td>64.66</td></tr><tr><td>ShortestPath (Borgwardt &amp; Kriegel, 2005)</td><td>42.32</td><td>76.43</td><td>78.86</td><td>59.10</td></tr><tr><td>WL (Shervashidze et al., 2011)</td><td>53.43</td><td>73.76</td><td>74.02</td><td>78.61</td></tr><tr><td>WL Optimal (Kriege et al., 2016)</td><td>60.13</td><td>75.26</td><td>79.04</td><td>80.74</td></tr><tr><td>PatchySan (Niepert et al., 2016)</td><td>-</td><td>75.00</td><td>76.27</td><td>72.60</td></tr><tr><td>GraphSage (Hamilton et al., 2017a)</td><td>54.25</td><td>70.48</td><td>75.42</td><td>68.25</td></tr><tr><td>ECC (Simonovsky &amp; Komodakis, 2017)</td><td>53.50</td><td>72.65</td><td>74.10</td><td>67.79</td></tr><tr><td>Set2Set (Vinyals et al., 2015)</td><td>60.15</td><td>74.29</td><td>78.12</td><td>71.75</td></tr><tr><td>SortPool (Morris et al., 2019)</td><td>57.12</td><td>75.54</td><td>79.37</td><td>73.76</td></tr><tr><td>DiffPool (Ying et al., 2018)</td><td>60.53</td><td>76.25</td><td>80.64</td><td>75.48</td></tr><tr><td>CliquePool (Luzhnica et al., 2019)</td><td>60.71</td><td>72.59</td><td>77.33</td><td>74.50</td></tr><tr><td>Sparse HGC (Cangea et al., 2018)</td><td>64.17</td><td>75.46</td><td>78.59</td><td>75.46</td></tr><tr><td>TopKPool (Gao &amp; Ji, 2019)</td><td>-</td><td>77.68</td><td>82.43</td><td>77.56</td></tr><tr><td>SAGPool (Lee et al., 2019)</td><td>-</td><td>71.86</td><td>76.45</td><td>-</td></tr><tr><td>GMN (ours)</td><td>78.66</td><td>82.07</td><td>82.24</td><td>77.26</td></tr><tr><td>MemGNN (ours)</td><td>75.50</td><td>81.35</td><td>82.92</td><td>77.0</td></tr></table>

ego-collaboration graph.

REDDIT-Binary (Yanardag & Vishwanathan, 2015) is for predicting the type of community given a graph of online discussion threads.

Tox21 (Challenge, 2014) is for predicting toxicity on 12 different targets.

For more information about the datasets and implementation details refer to Appendix A.2 and A.1, respectively.

# 4.2 RESULTS

To evaluate the performance of our models on DD, Enzymes, Proteins, and COLLAB datasets, we follow the experimental protocol in (Ying et al., 2018) and perform 10-fold cross-validation and report the mean accuracy over all folds. We also report the performance of four kernel-based methods including Graphlet (Shervashidze et al., 2009), shortest path (Borgwardt & Kriegel, 2005), Weisfeiler-Lehman (WL) (Shervashidze et al., 2011), and WL Optimal Assignment (Kriege et al., 2016), and ten deep models. The results shown in Table 1 suggest that: (i) our models significantly improve the performance on DD, Enzymes, and Proteins datasets by absolute margins of  $14.49\%$ ,  $4.75\%$ , and  $0.49\%$  accuracy, respectively, (ii) both proposed models achieve better performance on these three datasets compared to the baselines, (iii) MemGNN outperforms GMN on COLLAB whereas GMN achieves better result on the Enzymes, Proteins, and DD datasets. On COLLAB, our models are outperformed by a variant of DiffPpool (i.e., diffpool-det) (Ying et al., 2018) and WL Optimal Assignment (Kriege et al., 2016). The former is a GNN augmented with deterministic clustering algorithm $^{2}$ , whereas the latter is a graph kernel method. We speculate that because of the high edge-to-node ratio of COLLAB, these augmentations help in extracting near-optimal cliques.

For the ESOL and Lipophilicity datasets, we follow the evaluation protocol in (Wu et al., 2018) and report the Root-Mean-Square Error (RMSE) for these regression benchmarks. Considering that these datasets contain initial edge attributes (refer to Appendix A.2 for further details), we train the MemGNN model and compare the results to the baseline models reported in (Wu et al., 2018) including graph-based methods such as GCN, MPNN, Directed Acyclic Graph (DAG) based models, Weave as well as other conventional methods such as Kernel Ridge Regression (KRR) and Influence Relevance Voting (IRV). Tables 2 and 5 show that our MemGNN model achieves state-of-the-art results by absolute margin of 0.04 and 0.1 RMSE on ESOL and Lipophilicity benchmarks, respectively. For further details on these datasets and the baselines see (Wu et al., 2018).

We also achieve state-of-the-art results on the Bace, Reddit-Binary, and Tox21 datasets. For more details see Appendix A.3.

Table 2: RMSE on ESOL and Lipophilicity.  

<table><tr><td rowspan="3">Method</td><td colspan="4">Dataset</td></tr><tr><td colspan="2">ESOL</td><td colspan="2">Lipophilicity</td></tr><tr><td>validation</td><td>test</td><td>validation</td><td>test</td></tr><tr><td>Multitask</td><td>1.17 ± 0.13</td><td>1.12 ± 0.19</td><td>0.852 ± 0.048</td><td>0.859 ± 0.013</td></tr><tr><td>Random Forest</td><td>1.16 ± 0.15</td><td>1.07 ± 0.19</td><td>0.835 ± 0.036</td><td>0.876 ± 0.040</td></tr><tr><td>XGBoost</td><td>1.05 ± 0.10</td><td>0.99 ± 0.14</td><td>0.783 ± 0.021</td><td>0.799 ± 0.054</td></tr><tr><td>GCN</td><td>1.05 ± 0.15</td><td>0.97 ± 0.01</td><td>0.678 ± 0.040</td><td>0.655 ± 0.036</td></tr><tr><td>MPNN</td><td>0.55 ± 0.02</td><td>0.58 ± 0.03</td><td>0.757 ± 0.030</td><td>0.715 ± 0.035</td></tr><tr><td>KRR</td><td>1.65 ± 0.19</td><td>1.53 ± 0.06</td><td>0.889 ± 0.009</td><td>0.899 ± 0.043</td></tr><tr><td>DAG</td><td>0.74 ± 0.04</td><td>0.82 ± 0.08</td><td>0.857 ± 0.050</td><td>0.835 ± 0.039</td></tr><tr><td>Weave</td><td>0.57 ± 0.04</td><td>0.61 ± 0.07</td><td>0.734 ± 0.011</td><td>0.715 ± 0.035</td></tr><tr><td>MemGNN (ours)</td><td>0.53 ± 0.03</td><td>0.54 ± 0.01</td><td>0.555 ± 0.039</td><td>0.556 ± 0.023</td></tr></table>

# 4.3 ABLATION STUDY

# 4.3.1 EFFECT OF EDGE FEATURES

To investigate the effect of the proposed e-GAT model, we train our MemGNN model using both GAT and e-GAT models as the query network. Considering that the ESOL, Lipophilicity, and BACE datasets contain edge attributes, we use them as the benchmarks. Since nodes have richer features compared to edges, we set the node and edge feature dimensions to 16 and 4, respectively. The comparative performance evaluation of the two models on the ESOL dataset is shown in Appendix A.4 demonstrating that e-GAT achieves better results on the validation set in each epoch compared to the standard GAT model. We observed the same effect on Lipophilicity and BACE datasets too.

# 4.3.2 EFFECT OF TOPOLOGICAL EMBEDDING

To investigate the effect of the topological embeddings on the GMN model, we evaluated three initial topological features including adjacency matrix, normalized adjacency matrix, and Random Walk with Restart (RWR). For further details on RWR, see section A.5. The results suggested that using the adjacency matrix as the initial feature achieves the best performance. For instance, 10-fold cross validation accuracy of a GMN model trained on ENZYMES with adjacency matrix, normalized adjacency matrix, and RWR is  $78.66\%$ ,  $77.16\%$ , and  $77.33\%$ , respectively.

# 4.3.3 DOWN-SAMPLING NEIGHBORS WITH RANDOM WALKS

We investigated two methods to down-sample the neighbors in dense datasets such as COLLAB (i.e., average of 66 neighbors per node) to enhance the memory and computation. The first method randomly selects  $10\%$  of the edges whereas the second method ranks the neighbors based on their RWR scores with respect to the center node and then keeps the top  $10\%$  of the edges. We trained the MemGNN model on COLLAB using both sampling methods which resulted in  $73.9\%$  and  $73.1\%$  10-fold cross validation accuracy for random and RWR-based sampling methods respectively, suggesting that random sampling performs slightly better than a random walk sampling.

# 4.3.4 EFFECT OF NUMBER OF KEYS AND HEADS

We stipulate that although keys represent the clusters, the number of keys is not necessarily proportional to the number of the nodes in the input graphs. In fact, datasets with smaller graphs might have more meaningful clusters to capture. For example, molecules are comprised of numerous functional groups and yet the average number of nodes in the ESOL dataset is 13.3. Moreover, our experiments show that for ENZYMES with average number of 32.69 nodes, the best performance is achieved with 10 keys whereas for the ESOL dataset 64 keys results in the best performance. In ESOL 8, 64, and 160 keys result in RMSE of 0.56, 0.52, and 0.54, respectively. We also observed that keeping the number of parameters fixed, increasing the number of memory heads improves the performance. For instance, when the model is trained on ESOL with 160 keys and 1 head, it achieves RMSE of 0.54, whereas when trained with 32 keys of 5 heads, the same model achieves RMSE of 0.53.

![](images/938444516b3ecc5f73448e0b200ed75afa43ef7cc95182caecedae0b9a0c34df.jpg)  
(a)

![](images/d6c90861b7fc402222cb5fca99c1a0cbdd0d1a0a1b4ba9ef5c272a6f981482b6.jpg)  
(b)  
Figure 2: Visualization of the learned clusters of two molecules instances from (a) ESOL and (b) Lipophilicity datasets. The visualizations show that the learned clusters correspond to known chemical groups. Note that a node without label represents a carbon atom. For more visualizations and discussion see section A.6

# 4.3.5 WHAT DO THE KEYS REPRESENT?

Intuitively, the memory keys represent the cluster centroids and enhance the model performance by capturing meaningful structures and coarsening the graph. To investigate this intuition, we used the learned keys to interpret the knowledge learned by the models through visualizations. Figure 2 visualizes the learned clusters over atoms (i.e., atoms with same color are within the same cluster) indicating that the clusters mainly consist of meaningful chemical substructures such as a carbon chain and a Hydroxyl group (OH) (i.e., Figure 2a), as well as a Carboxyl group (COOH) and a benzene ring (i.e., Figure 2b). From a chemical perspective, Hydroxyl and Carboxyl groups, and carbon chains have a significant impact on the solubility of the molecule in water or lipid. This confirms that the network has learned chemical features that are essential for determining the molecule solubility. It is noteworthy that we tried initializing the memory keys using K-Means algorithm over the initial node embeddings to warm-start them but we did not observe any significant improvement over the randomly selected keys.

# 5 CONCLUSION

We proposed an efficient memory layer and two deep models for hierarchical graph representation learning. We evaluated the proposed models on nine graph classification and regression tasks and achieved state-of-the-art results on eight of them. We also experimentally showed that the learned representations can capture the well-known chemical features of the molecules. Our study indicated that node attributes concatenated with corresponding topological embeddings in combination with one or more memory layers achieves notable results without using message passing. We also showed that for the topological embeddings, the binary adjacency matrix is sufficient and thus no further preprocessing step is required for extracting them. Finally, we showed that although connectivity information is not directly imposed on the model, the memory layer can process node embeddings and properly cluster and aggregate the learned embeddings.

Limitations: In section 4.2, we discussed that on the COLLAB dataset, kernel methods or deep models augmented with deterministic clustering algorithm achieve better performance compared to our models. Analyzing samples in this dataset suggests that in graphs with dense communities, such as cliques, our model lacks the ability to properly detect these dense sub-graphs. Moreover, the results of the DD dataset reveals that our MemGNN model outperforms the GMN model which implies that we need message passing to perform better on this dataset. We speculate that this is because the DD dataset relies more on local information. The most important features to train an SVM on this dataset are surface features which have local behavior. This suggest that for data with strong local interactions, message passing is required to improve the performance.

Future Directions: We are planning to introduce a model based on the MemGNN and GMN architectures that can perform node classification by attending to the node embeddings and centroids of the clusters from different layers of hierarchy that the node belongs to. We are also planning to investigate the representation learning capabilities of the proposed models in self-supervised setting.

# REFERENCES

Elie Aljalbout, Vladimir Golkov, Yawar Siddiqui, Maximilian Strobel, and Daniel Cremers. Clustering with deep learning: Taxonomy and new methods. arXiv preprint arXiv:1801.07648, 2018.  
Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps and spectral techniques for embedding and clustering. In Advances in Neural Information Processing Systems 14, pp. 585-591. MIT Press, 2002.  
Filippo M. Bianchi, Daniele Grattarola, and Cesare Alippi. Mincut pooling in graph neural networks. arXiv preprint arXiv:1907.00481, 2019.  
Karsten M Borgwardt and Hans-Peter Kriegel. Shortest-path kernels on graphs. In Fifth IEEE international conference on data mining, pp. 8-pp. IEEE, 2005.  
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. In International Conference on Learning Representation (ICLR), 2014.  
Catalina Cangea, Petar Velicković, Nikola Jovanović, Thomas Kipf, and Pietro Lio. Towards sparse hierarchical graph classifiers. In Advances in neural information processing systems, Relational Representation Learning Workshop, 2018.  
Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 132-149, 2018.  
Tox21 Data Challenge. Tox21 data challenge 2014, 2014.  
Hanjun Dai, Bo Dai, and Le Song. Discriminative embeddings of latent variable models for structured data. In International conference on machine learning, pp. 2702-2711, 2016.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in neural information processing systems, pp. 3844-3852, 2016.  
John S Delaney. Esol: estimating aqueous solubility directly from molecular structure. Journal of chemical information and computer sciences, 44(3):1000-1005, 2004.  
Inderjit S Dhillon, Yuqiang Guan, and Brian Kulis. Weighted graph cuts without eigenvectors a multilevel approach. IEEE transactions on pattern analysis and machine intelligence, 29(11): 1944-1957, 2007.  
Frederik Diehl. Edge contraction pooling for graph neural networks. arXiv preprint arXiv:1905.10990, 2019.  
Paul D Dobson and Andrew J Doig. Distinguishing enzyme structures from non-enzymes without alignments. Journal of molecular biology, 330(4):771-783, 2003.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in neural information processing systems, pp. 2224-2232, 2015.  
Hongyang Gao and Shuiwang Ji. Graph u-nets. In International Conference on Machine Learning, pp. 2083-2092, 2019.  
Anna Gaulton, Anne Hersey, Michal Nowotka, A Patrónica Bento, Jon Chambers, David Mendez, Prudence Mutowo, Francis Atkinson, Louisa J Bellis, Elena Cibrián-Uhalte, et al. The chembl database in 2017. Nucleic acids research, 45(D1):D945–D954, 2016.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1263-1272. JMLR.org, 2017.

Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014.  
Aditya Grover and Jure Leskovec. Node2vec: Scalable feature learning for networks. In Proceedings of the 22nd International Conference on Knowledge Discovery and Data Mining, pp. 855-864. ACM, 2016.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pp. 1024-1034, 2017a.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pp. 1024-1034, 2017b.  
Kaveh Hassani and Mike Haley. Unsupervised multi-task feature learning on point clouds. In The IEEE International Conference on Computer Vision (ICCV), pp. 8160-8171, 2019.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, Proceedings of Machine Learning Research, pp. 448-456, Jul 2015.  
Diederik P Kingma and Jimmy Lei Ba. Adam: Amethod for stochastic optimization. In International Conference on Learning Representation (ICLR), 2014.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2016.  
Nils M Kriege, Pierre-Louis Giscard, and Richard Wilson. On valid optimal assignment kernels and applications to graph classification. In Advances in Neural Information Processing Systems, pp. 1623-1631, 2016.  
Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Large memory layers with product keys. arXiv preprint arXiv:1907.05242, 2019.  
Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 3734-3743, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. In International Conference on Learning Representations, 2015.  
Enxhell Luzhnica, Ben Day, and Pietro Lio. Clique pooling for graph classification. In International Conference on Learning Representations Workshop, Representation Learning on Graphs and Manifolds, 2019.  
Jianxin Ma, Peng Cui, Kun Kuang, Xin Wang, and Wenwu Zhu. Disentangled graph convolutional networks. In Proceedings of the 36th International Conference on Machine Learning, pp. 4212-4221, 2019.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9:2579-2605, 2008.  
Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 4602-4609, 2019.  
Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. Learning convolutional neural networks for graphs. In International conference on machine learning, pp. 2014-2023, 2016.  
Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. Asymmetric transitivity preserving graph embedding. In Proceedings of the 22nd International Conference on Knowledge Discovery and Data Mining, pp. 1105-1114. ACM, 2016.

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th International Conference on Knowledge Discovery and Data Mining, pp. 701-710. ACM, 2014.  
Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adria Puigdomenech Badia, Oriol Vinyals, Demis Hassabis, Daan Wierstra, and Charles Blundell. Neural episodic control. In Proceedings of the 34th International Conference on Machine Learning, pp. 2827-2836, 2017.  
Jack Rae, Jonathan J Hunt, Ivo Danihelka, Timothy Harley, Andrew W Senior, Gregory Wayne, Alex Graves, and Timothy Lillicrap. Scaling memory-augmented neural networks with sparse reads and writes. In Advances in Neural Information Processing Systems, pp. 3621-3629, 2016.  
Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. arXiv preprint arXiv:1906.00446, 2019.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Metalearning with memory-augmented neural networks. In Proceedings of the 33th International Conference on Machine Learning, pp. 1842-1850, 2016.  
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In European Semantic Web Conference, pp. 593-607. Springer, 2018.  
Ida Schomburg, Antje Chang, Christian Ebeling, Marion Gremse, Christian Heldt, Gregor Huhn, and Dietmar Schomburg. Brenda, the enzyme database: updates and major new developments. Nucleic acids research, 32(suppl_1):D431-D433, 2004.  
Nino Shervashidze, SVN Vishwanathan, Tobias Petri, Kurt Mehlhorn, and Karsten Borgwardt. Efficient graphlet kernels for large graph comparison. In Artificial Intelligence and Statistics, pp. 488-495, 2009.  
Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(Sep):2539-2561, 2011.  
Martin Simonovsky and Nikos Komodakis. Dynamic edge-conditioned filters in convolutional neural networks on graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3693-3702, 2017.  
Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, pp. 4077-4087, 2017.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15:1929-1958, 2014.  
Govindan Subramanian, Bharath Ramsundar, Vijay Pande, and Rajiah Aldrin Denny. Computational modeling of  $\beta$ -secretase 1 (bace-1) inhibitors using ligand based approaches. Journal of chemical information and modeling, 56(10):1936-1949, 2016.  
Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. In Advances in neural information processing systems, pp. 2440-2448, 2015.  
Hanghang Tong, Christos Faloutsos, and Jia-Yu Pan. Fast random walk with restart and its applications. In Sixth International Conference on Data Mining (ICDM'06), pp. 613-622. IEEE, 2006.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.

Petar Velikovi, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.  
Oriol Vinyals, Samy Bengio, and Manjunath Kudlur. Order matters: Sequence to sequence for sets. In International Conference on Learning Representations, 2015.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in neural information processing systems, pp. 3630-3638, 2016.  
Chu Wang, Babak Samari, and Kaleem Siddiqi. Local spectral graph convolution for point set feature learning. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 52-66, 2018.  
Tingwu Wang, Yuhao Zhou, Sanja Fidler, and Jimmy Ba. Neural graph evolution: Automatic robot design. In International Conference on Learning Representations, 2019.  
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. In International Conference on Learning Representation (ICLR), 2015.  
Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Genisses, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513-530, 2018.  
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. A comprehensive survey on graph neural networks. arXiv preprint arXiv:1901.00596, 2019.  
Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In International conference on machine learning, pp. 478-487, 2016.  
Pinar Yanardag and S.V.N. Vishwanathan. Deep graph kernels. In Proceedings of the 21th International Conference on Knowledge Discovery and Data Mining, pp. 1365-1374. ACM, 2015.  
Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In Advances in Neural Information Processing Systems, pp. 4800-4810, 2018.  
Ziwei Zhang, Peng Cui, and Wenwu Zhu. Deep learning on graphs: A survey. arXiv preprint arXiv:1812.04202, 2018.  
Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Graph neural networks: A review of methods and applications. arXiv preprint arXiv:1812.08434, 2018.
