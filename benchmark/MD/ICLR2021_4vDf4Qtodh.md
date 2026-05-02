# INSTANTEMBEDDING: EFFICIENT LOCAL NODE REPRESENTATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this paper, we introduce InstantEmbedding, an efficient method for generating single-node representations using local PageRank computations. We theoretically prove that our approach produces globally consistent representations in sublinear time. We demonstrate this empirically by conducting extensive experiments on real-world datasets with over a billion edges. Our experiments confirm that InstantEmbedding requires drastically less computation time (over 9,000 times faster) and less memory (by over 8,000 times) to produce a single node's embedding than traditional methods including DeepWalk, node2vec, VERSE, and FastRP. We also show that our method produces high quality representations, demonstrating results that meet or exceed the state of the art for unsupervised representation learning on tasks like node classification and link prediction.

# 1 INTRODUCTION

Graphs are widely used to represent data when are objects connected to each other, such as social networks, chemical molecules, and knowledge graphs. A widely used approach in dealing with graphs is learning compact representations of graphs (Perozzi et al., 2014; Grover & Leskovec, 2016; Abu-El-Haija et al., 2018), which learns a  $d$ -dimensional embedding vector for each node in a given graph. Unsupervised embeddings in particular have shown improvements in many downstream machine learning tasks, such as visualization (Maaten & Hinton, 2008), node classification (Perozzi et al., 2014) and link prediction (Abu-El-Haija et al., 2018). Importantly, since such embeddings are learned solely from the structure of the graph, they can be used across multiple tasks and applications.

Typically, graph embedding models often assume that graph data fits in memory (Perozzi et al., 2014) and require representations for all nodes to be generated. However, in many real-world applications, it is often the case that graph data is large but also scarcely annotated. For example, the Friendster social graph (Yang & Leskovec, 2015) has only  $30\%$  nodes assigned to a community, from its total 65M entries. At the same time, many applications of graph embeddings such as classifying a data item only require one current representation for the item itself, and eventually representations of labeled nodes. Therefore, computing a full graph embedding is at worst infeasible and at best inefficient.

These observations motivate the problem which we study in this paper – the Local Node Embedding problem. In this setting, the embedding for a node is restricted to using only local structural information, and can not access the representations of other nodes in the graph or rely on trained global model state. In addition, we require that a local method needs to produce embeddings which are consistent with all other node's representations, so that the final representations can be used in the same downstream tasks that graph embeddings have proved adapt at in the past.

In this work, we introduce InstantEmbedding, an efficient method to generate local node embeddings on the fly in sublinear time which are globally consistent. Considering previous work that links embedding learning methods to matrix factorization (Tsitsulin et al., 2018; Qiu et al., 2018), our method leverages a high-order similarity matrix based on Personalized PageRank (PPR) as foundations on which local node embeddings are computed via hashing. We offer theoretical guarantees on the locality of the computation, as well as the proof of the global consistency of the generated embeddings. We show empirically that our method is able to produce high-quality representations on par with state of the art methods, with efficiency several orders of magnitude better in clock time and memory consumption: running 9,000 times faster and using 8,000 times less memory on the largest graphs that contenders can process.

Table 1: Related work in terms of desirable properties and complexities. Analysis in Section 3.2.1.  

<table><tr><td rowspan="2">method</td><td colspan="4">Properties</td><td colspan="2">Complexities</td></tr><tr><td>Local Inference</td><td>No Global Training</td><td>Unsupervised Embedding</td><td>Attribute-Free</td><td>Time O</td><td>Memory O</td></tr><tr><td>DeepWalk</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>dn log n</td><td>dn + m</td></tr><tr><td>node2vec</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>dbn</td><td>n3</td></tr><tr><td>VERSE</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>dbn</td><td>dn + m</td></tr><tr><td>FastRP</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>dm√n</td><td>dn + m</td></tr><tr><td>GCN</td><td>X</td><td>X</td><td>X</td><td>X</td><td>dm</td><td>dn + m</td></tr><tr><td>DGI</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>dm</td><td>dn + m</td></tr><tr><td>InstantEmbedding</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>1/α(1-α)ε + d</td><td>1/α(1-α)ε + d</td></tr></table>

# 2 PRELIMINARIES & RELATED WORK

# 2.1 GRAPH EMBEDDING

Let  $G = (V, E)$  represent an unweighted graph, which contains a set of nodes  $V$ ,  $|V| = n$ , and edges  $E \subseteq (V \times V)$ ,  $|E| = m$ . A graph can also be represented as an adjacency matrix  $\mathbf{A} \in \{0, 1\}^{n \times n}$  where  $\mathbf{A}_{u,v} = 1$  iff  $(u, v) \in E$ . The task of graph embedding then, is to learn a  $d$ -dimensional node embedding matrix  $\mathbf{X} \in \mathbb{R}^{n \times d}$  where  $\mathbf{X}_v$  serves as the embedding for any node  $v \in G$ . We note that  $d \ll n$ , i.e. the learned representations are low-dimensional, and the challenge of graph embedding is to best preserve graph properties (such as node similarities) in this space. Following the formalization in Abu-El-Haija et al. (2018), many graph embeddings can be thought of minimizing an objective in the general form:  $\min_{\mathbf{X}} L(f(\mathbf{X}), g(\mathbf{A}))$ , where  $f: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times n}$  is a pairwise distance function on the embedding space,  $g: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$  is a distance function on the (possibly transformed) adjacency matrix, and  $L$  is a loss function over all  $(u, v) \in (V \times V)$  pairs.

A number of graph embedding methods have been proposed. One family of these methods simply learn  $\mathbf{X}$  as a lookup dictionary of embeddings and calculate the loss via distance (Kruskal, 1964), or matrix factorization (either implicit (Perozzi et al., 2014; Grover & Leskovec, 2016) or explicit (Ou et al., 2016)). On attributed structured data, Graph Convolutional Networks (Kipf & Welling, 2016) have been successfully applied to both supervised and unsupervised tasks (Veličković et al., 2018). However, in the absence of node-level features, Duong et al. (2019) demonstrated that these methods do not produce meaningful representations.

Graph Embedding via Random Projection The computational efficiency brought by advances in random projection (Achlioptas, 2003; Dasgupta et al., 2010) paved the way for its adaption in graph embedding to allow direct construction of the embedding matrix  $\mathbf{X}$ . Two recent works, RandNE (Zhang et al., 2018) and FastRP (Chen et al., 2019) iteratively project the adjacency matrix to simulate the higher-order interactions between nodes. As we show in the experiments, these methods suffer from high memory requirements and are not always competitive with other methods.

# 2.2 LOCAL ALGORITHMS ON GRAPHS

Local algorithms on graphs (Suomela, 2013) solve graph without using the full graph. A well-studied problem in this space is personalized recommendation (Jeh & Widom, 2003) where users are represented as nodes in a graph and the goal is to recommend items to specific users leveraging the graph structure. Classic solutions to this problem are Personalized PageRank (Gupta et al., 2013) and Collaborative Filtering (Schafer et al., 2007; He et al., 2017). Interestingly, these methods have been recently applied to graph neural networks (Klicpera et al., 2019; He et al., 2020). We now recall the definition of Personalized PageRank that is one of the main ingredients in our embedding algorithm.

Definition (Personalized PageRank (PPR)). Given  $\mathbf{s} \in \mathbb{R}^n$  ( $\mathbf{s}_i \geq 0$ ,  $\sum_{i} \mathbf{s}_i = 1$ ), a distribution of the starting node of random walks, and  $\alpha \in (0,1)$ , a decay factor, the Personalized PageRank vector  $\pi(\mathbf{s}) \in \mathbb{R}^n$  is defined recursively as:

$$
\pi (\mathbf {s}) = \alpha \mathbf {s} + (1 - \alpha) \pi (\mathbf {s}) ^ {\top} \mathbf {D} ^ {- 1} \mathbf {A}, \tag {1}
$$

where  $\mathbf{D}^{-1}\mathbf{A}$  is the transition matrix.

PPR takes as input a distribution of starting nodes  $\mathbf{s}$ , which is typically a  $n$  dimensional one-hot vector  $\mathbf{e}_i$  with 1 in the  $i$ -th coordinate, enforcing a local random walks starting from node  $i$ . Following this practice, we denote  $\pi_i \in \mathbb{R}^n$ , the PPR vector starting from a single node  $i$ , and  $\mathbf{PPR} \in \mathbb{R}^{n \times n}$ , the full PPR matrix for all nodes in the graph, where  $\mathbf{PPR}_{i,:} = \pi(\mathbf{e}_i)$ . VERSE (Tsitsulin et al., 2018) proposes to learn node embeddings by implicitly factorizing PPR. Its stochastic approach can perform well, but lacks guarantees of stability and convergence. The idea of learning embeddings based on local random walks has also been used in the property testing framework, a direction in graph algorithm aiming at analyzing the clustering structure of a graph (Kale & Seshadhri, 2008; Czumaj & Sohler, 2010; Czumaj et al., 2015; Chiplunkar et al., 2018).

# 2.3 PROBLEM STATEMENT

In this work, we consider the problem of embedding a single node in a graph quickly. More formally, we consider what we term the Local Node Embedding problem: given a graph  $G$  and any node  $v$ , return a globally consistent structural representation for  $v$  using only local information around  $v$ , in time sublinear to the size of the graph.

A solution to the local node embedding problem should possess two following properties:

1. Locality. The embeddings for a node are computed locally, i.e. the embedding for a node can be produced using only local information and in time independent of the total graph size.  
2. Global Consistency. A local method must produce embeddings that are globally consistent (i.e. able to relate each embedding to each other, s.t. distances in the space preserve proximity).

While many node embedding approaches have been proposed (Chen et al., 2018), to the best of our knowledge we are the first to examine the local embedding problem. Furthermore, no existing methods for positional representations of nodes meet these requirements. We briefly discuss these requirements in detail below, and put the related work in terms of these properties in Table 1.

Locality. While classic node embedding methods, such as DeepWalk (Perozzi et al., 2014), node2vec (Grover & Leskovec, 2016), or VERSE (Tsitsulin et al., 2018) rely on information aggregated from local subgraphs (e.g. sampled by a random walk), they do not meet our locality requirement. Specifically, they also require the representations of all the nodes around them, resulting in a dependency on information from all nodes in the graph (in addition to space complexity  $O(nd)$  where  $d$  is the embedding dimension) to compute a single representation. Classical random-projection based methods also require access to the full adjacency matrix in order to compute the higher-order ranking matrix. We briefly remark that even methods capable of local attributed subgraph embedding (such as GCN or DGI) also do not meet this definition of locality, as they require a global training phase to calibrate their graph pooling functions.

Global Consistency. This property allows embeddings produced by local node embedding to be used together, perhaps as features in a model. While existing methods for node embeddings are global ones which implicitly have global consistency, this property is not trivial for a local method to achieve. One exciting implication of a local method which is globally consistent is that it can wait to compute a representation until it is actually required for a task. For example, in a production system, one might only produce representations for immediate classification when they are requested.

In the rest of this paper, we propose our approach satisfying these properties in Section 3, and experimentally illustrate the satisfied properties in Section 4, followed by conclusions in Section 5.

# 3 METHOD

Here we outline our proposed approach for local node embedding. We begin by discussing the connection between a recent embedding approach and matrix factorization. Then using this analysis, we propose an embedding method based on randomly hashing the PPR matrix. We note that this approach has a tantalizing property - it can be decomposed into entirely local operations per node. With this observation in hand, we present our solution, InstantEmbedding. Finally, we analyze the algorithmic complexity of our approach, showing that it is both a local algorithm (which runs in time sublinear to the size of  $G$ ) and that the local representations are globally consistent.

# 3.1 GLOBAL EMBEDDING USING PPR

A recently proposed method for node embedding, VERSE (Tsitsulin et al., 2018), learns node embeddings using a neural network which encodes Personalized PageRank similarities. Their objective function, in the form of Noise Contrastive Estimation (NCE) (Gutmann & Hyvarinen, 2010), is:

$$
\mathcal {L} = \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} \left[ \mathbf {P P R} _ {i j} \log \sigma \left(\mathbf {x} _ {i} ^ {\top} \mathbf {x} _ {j}\right) + b \mathbb {E} _ {j ^ {\prime} \sim \mathcal {U}} \log \sigma \left(- \mathbf {x} _ {i} ^ {\top} \mathbf {x} _ {j ^ {\prime}}\right) \right], \tag {2}
$$

where PPR is the Personalized PageRank matrix,  $\sigma$  is the sigmoid function,  $b$  is the number of negative samples, and  $\mathcal{U}$  is a uniform noise distribution from which negative samples are drawn. Like many SkipGram-style methods (Mikolov et al., 2013), this learning process can be linked to matrix factorization by the following lemma:

Lemma 3.1 (Tsitsulin et al. (2020)). VERSE implicitly factorizes the matrix  $\log (\mathbf{P}\mathbf{P}\mathbf{R}) + \log n - \log b$  into  $\mathbf{X}\mathbf{X}^{\top}$ , where  $n$  is the number of nodes in the graph and  $b$  is the number of negative samples.

# 3.1.1 HASHING FOR GRAPH EMBEDDING

Lemma 3.1 provides an incentive to find an efficient alternative to factorizing the dense similarity matrix  $\mathbf{M} = \log (\mathbf{P}\mathbf{P}\mathbf{R}) + \log n - \log b$ . Our choice of the algorithm requires two important properties: a) providing an unbiased estimator for the inner product, and b) requiring less than  $\mathcal{O}(n)$  memory. The first property is essential to ensure we have a good sketch of  $\mathbf{M}$  for the embedding, while the second one keeps our complexity per node sublinear.

In order to meet both requirements we propose to use hashing (Weinberger et al., 2009) to preserve the essential similarities of PPR in expectation. We leverage two global hash functions  $h_{\mathrm{d}} \colon \mathbb{N} \to \{0, \dots, d - 1\}$  and  $h_{\mathrm{sgn}} \colon \mathbb{N} \to \{-1, 1\}$  sampled from universal hash families  $\mathbb{U}_d$  and  $\mathbb{U}_{-1,1}$  respectively, to define the hashing kernel  $H_{h_{\mathrm{d}}, h_{\mathrm{sgn}}} : \mathbb{R}^n \to \mathbb{R}^d$ . Applied to an input vector  $\mathbf{x}$ , it yields  $\mathbf{h} = H_{h_{\mathrm{d}}, h_{\mathrm{sgn}}}(\mathbf{x})$ , where  $\mathbf{h}_i = \sum_{k \in h_d^{-1}(i)} \mathbf{x}_k h_{\mathrm{sgn}}(k)$ .

We note that although  $H_{h_{\mathrm{d}}, h_{\mathrm{sgn}}}$  is proposed for vectors, it can be trivially extended to matrix  $\mathbf{M}$  when applied to each row vector of that matrix, e.g. by defining  $H_{h_{\mathrm{d}}, h_{\mathrm{sgn}}}(\mathbf{M})_{i,:} \equiv H_{h_{\mathrm{d}}, h_{\mathrm{sgn}}}(\mathbf{M}_{i,:})$ . In the appendix we prove the next lemma that follows from (Weinberger et al., 2009) and highlights both the aforementioned properties:

Lemma 3.2. The space complexity of  $H_{h_{\mathrm{d}}, h_{\mathrm{sgn}}}$  is  $\mathcal{O}(1)$  and:

$$
\mathbb {E} _ {h _ {\mathrm {d}} \sim \mathbb {U} _ {d}, h _ {\mathrm {s g n}} \sim \mathbb {U} _ {- 1, 1}} \left[ H _ {h _ {\mathrm {d}}, h _ {\mathrm {s g n}}} (\mathbf {M}) H _ {h _ {\mathrm {d}}, h _ {\mathrm {s g n}}} (\mathbf {M}) ^ {\top} \right] = \mathbf {M M} ^ {\top}
$$

Our algorithm for global node embedding is presented in Algorithm 1. First, we compute the PPR matrix PPR (Line 2) with a generic approach (CreatePPRMatrix), which takes a graph and  $\epsilon$ , the desired precision of the approximation. We remark that any of the many proposed approaches for computing such a matrix (e.g. from Jeh & Widom (2003); Andersen et al. (2007); Lofgren et al. (2014)) can be used for this calculation. As the PPR could be dense, the same could be said about the implicit matrix M. Thus, we filter the signal from non-significant PPR values by applying the max operator. We remove the constant  $\log b$  from the implicit target matrix. In lines (4-6), the provided hash function accumulates each value in the corresponding embedding dimension.

# Algorithm 1 Global Node Embedding using Personalized PageRank

Input: graph  $G$  embedding dimension  $d$  , PPR precision  $\epsilon$  , hash functions  $h_d,h_{sgn}$  Output: embedding matrix W   
1: function GRAPHEMBEDDING(G,d,  $\epsilon ,h_d,h_{\mathrm{sgn}})$    
2: PPR  $\leftarrow$  CreatePPRMatrix(G,  $\epsilon$  1   
3:  $\mathbf{W} = \mathbf{0}_{n\times d}$    
4: for  $\pi_i$  in PPR do   
5: for  $r_j$  in  $\pi_i$  do   
6:  $\mathbf{W}_{i,h_{\mathrm{d}}(j)} + = h_{\mathrm{sgn}}(j)\times \max (\log (r_j*n),0)$    
7: return W

Interestingly, the projection operation only uses information from each node's individual PPR vector  $\pi_{i}$  to compute its representation. In the following section, we will show that local calculation of the PPR can be utilized to develop an entirely local algorithm for node embedding.

# 3.2 LOCAL NODE EMBEDDING VIA INSTANTEEMBEDDING

Having a local projection method, all that we require is a procedure that can calculate the PPR vector for a node in time sublinear to size of the graph. Specifically, for InstantEmbedding we propose that the CreatePPRMatrix operation consists of invoking the SparsePPR routine from Andersen et al. (Andersen et al., 2007) once for each node  $i$ . This routine is an entirely local algorithm for efficiently constructing  $\pi_i$ , the PPR vector for node  $i$ , which offers strong guarantees. The following lemma formalizes the result (proof in Appendix A.4).

Lemma 3.3. The INSTANTEMBEDING  $(v,G,d,\epsilon)$  algorithm computes the local embedding of a node  $v$  by exploring at most the  $O\left(1 / (1 - \alpha)\epsilon\right)$  nodes in the neighborhood of  $v$ .

We present InstantEmbedding, our algorithm for local node embedding, in Algorithm 2. As we will show, it is a self-contained solution for the local node embedding problem that can generate embeddings for individual nodes extremely efficiently. Notably, per Lemma 3.3, the local area around  $v$  explored by InstantEmbedding is independent of  $n$ . Therefore the algorithm is strictly local.

Algorithm 2 InstantEmbedding  
Input: node  $v$ , graph  $G$ , embedding dimension  $d$ , PPR precision  $\epsilon$ , hash functions  $h_d, h_{sgn}$   
Output: embedding vector  $\mathbf{w}$   
1: function INSTANTEMBEDGING(v, G, d,  $\epsilon$ , h_d, h_{sgn})  
2:  $\pi_v \gets \text{SparsePPR}(v, G, \epsilon)$   
3:  $\mathbf{w} \gets \mathbf{0}_d$   
4: for  $r_j$  in  $\pi_v$  do  
5:  $\mathbf{w}_{h_d(j)} + = h_{sgn}(j) \times \max(\log(r_j * n), 0)$   
6: return  $\mathbf{w}$

# 3.2.1 ANALYSIS

We now prove some basic properties of our proposed approach. First, we show that the runtime of our algorithm is local and independent of  $n$ , the number of nodes in the graph. Then, we show that our local computations are globally consistent, i.e., the embedding of a node  $v$  is the same independently if we compute it locally or if we recompute the embeddings for all nodes in the graph at the same time. Note that we focus on bounding the running time to compute the embedding for a single node in the graph. Nonetheless, the global complexity to compute all the embeddings can be obtained by multiplying our bound by  $n$ , although it is not the focus of this work. We state the following theorem and prove it in Appendix A.5.

Theorem 3.4. The InstantEmbedding  $(v,G,d,\epsilon)$  algorithm has running time  $\mathcal{O}\left(d + 1 / \alpha (1 - \alpha)\epsilon\right)$ .

Besides the embedding size  $d$ , both the time and space complexity of our algorithm depend only on the approximation factor  $\epsilon$  and the decay factor  $\alpha$ . Both are independent of  $n$ , the size of the graph, and  $m$ , the size of the edge set. Notably, if  $\mathcal{O}\left(1 / \alpha (1 - \alpha)\epsilon\right) \in o(n)$ , as commonly happen in real world applications, our algorithm has sublinear time w.r.t. the graph size. Lastly, we note that the space complexity is also sublinear (due to Lemma 3.3), which we show in the appendix.

Now we turn our attention to the consistency of our algorithm, by showing that for a node  $v$  the embeddings computed by InstantEmbedding and GraphEmbedding are identical. In the following we denote the graph embedding computed by GraphEmbedding $(G, d, \epsilon)$  for node  $v$  by GraphEmbedding $(G, d, \epsilon)_v$ , and we prove the following theorem (Appendix A.6).

Theorem 3.5 (Global Consistency). InstantEmbedding  $(v, G, d, \epsilon)$  output equals one of GraphEmbedding  $(G, d, \epsilon)$  at position  $v$ .

Complexity Comparison. Table 1 compares the complexity of InstantEmbedding with that of previous works:  $d, n, m$  stands for embedding dimension, size of graph and number of edges respectively. Specifically,  $b \geq 1$  stands for the number of samples used in node2vec and VERSE. It is noteworthy that all the previous works have time complexity depending on  $n$ , and perform at least linear w.r.t. size of the graph. In contrast, our algorithm depends only on  $\epsilon$  and  $\alpha$ , and has sublinear time w.r.t.  $n$ , the graph size. In Section 4, we experimentally verify the advantages of our principled method.

# 4 EXPERIMENTS

In the light of the theoretical guarantees about the proposed method, we perform extended experiments in order to verify our two main hypotheses:

1. H1. Computing local node-embedding is more efficient than generating a global embedding.  
2. H2. The local representations are consistent and of high-quality, being competitive with and even surpassing state-of-the-art methods on several tasks.

We assess H1 in Section 4.2, in which we measure the efficiency of generating a single node embedding for each method. Then in Section 4.3 we validate H2 by comparing our method against the baselines on multiple datasets using tasks of node classification, link prediction and visualization.

# 4.1 DATASETS AND EXPERIMENTAL SETTINGS

To ensure a relevant and fair evaluation, we compare our method against multiple strong baselines, including DeepWalk (Perozzi et al., 2014), node2vec (Grover & Leskovec, 2016), VERSE (Tsitsulin et al., 2018), and FastRP (Chen et al., 2019). Each method was run on a virtual machine hosted on the Google Cloud Platform, with a  $2.3\mathrm{GHz}$  16-core CPU and 128GB of RAM. All reported results use dimensionality  $d = 512$  for every method. We provide additional experiments for multiple dimensions, along with full details regarding each method and its parameterization in the Appendix B.1. For reproducibility, we release an implementation of our method.

Table 2: Dataset attributes: size of vertices  $\left| V\right|$  ,edges  $\left| E\right|$  ,labeled vertices  $\left| S\right|$  .  

<table><tr><td>Dataset</td><td>|V|</td><td>|E|</td><td>|S|</td></tr><tr><td>PPI</td><td>3.8k</td><td>38k</td><td>3.8k</td></tr><tr><td>BlogCatalog</td><td>10k</td><td>334k</td><td>10k</td></tr><tr><td>CoCit</td><td>44k</td><td>195k</td><td>44k</td></tr><tr><td>CoAuthor</td><td>52k</td><td>356k</td><td>—</td></tr><tr><td>Flickr</td><td>81k</td><td>5.9M</td><td>81k</td></tr><tr><td>YouTube</td><td>1.1M</td><td>3.0M</td><td>32k</td></tr><tr><td>Amazon2M</td><td>2.4M</td><td>62M</td><td>—</td></tr><tr><td>Orkut</td><td>3.0M</td><td>117M</td><td>110k</td></tr><tr><td>Friendster</td><td>66M</td><td>1806M</td><td>—</td></tr></table>

InstantEmbedding Instantiation. As presented in Section 3, our implementation of the presented method relies on the choice of PPR approximation used. For instant single-node embeddings, we use the highly efficient PushFlow (Andersen et al., 2007) approximation that enables us to dynamically load into memory at most  $\frac{2}{(1 - \alpha)\epsilon}$  nodes from the full graph to compute a single PPR vector  $\pi$ . This is achieved by storing graphs in binarized compressed sparse row format that allows selective reads for nodes of interest. In the special case when a full graph embedding is requested, we have the freedom to approximate the PPR in a distributed manner (we omit this from runtime analysis, as we had no distributed implementations for the baselines, but we note our local method is trivially parallelizable). We refer to Appendix B.5 for the study of the influence of  $\epsilon$  on runtime and quality.

Datasets. We perform our evaluations on 10 datasets, as presented in Table 2. Detailed descriptions of these datasets are available in the supplementary material. Note that on YouTube and Orkut the number of labeled nodes is much smaller than the total. We observe this behavior in several real-world application scenarios, where our method shines the most.

# 4.2 PERFORMANCE CHARACTERISTICS

We report the mean wall time and total memory consumption (Wolff) required to generate an embedding  $(d = 512)$  for a node in the given dataset. We repeat the experiment 1,000 times for InstantEmbedding due to its locality property; for the baselines, we measure the time 5 times, and memory once. Complete results for all results and method can be found in Appendix B.3.

Running Time. As Figure 1(a) shows, InstantEmbedding is the most scalable method, drastically outperforming all the other methods. We are over 9,000 times faster than the next fastest baseline in the largest graph both methods can process, and can scale to graphs of any size.

Memory Consumption. As Figure 1(b) shows, InstantEmbedding is the most efficient method having been able to run in all datasets using negligible memory compared to the other methods. Compared to the next most memory-efficient baseline (VERSE) we are over 8,000 times more efficient in the largest graph both methods can process.

![](images/1d54d4d06588c3329bfc58ebcc9de0d6cc440a6fbc6463330990299ca38ebc92.jpg)

![](images/db995782ccc80ce2d2d12bb25be47e7e4faa258c191283aae450246e08544617.jpg)  
(a) Running Time

![](images/79d0d2f4e8b8c4901c9215066d415af4b7aa5c63e3351275161b2a7a3b3dad01.jpg)  
Figure 1: Required (a) running time and (b) memory consumption to generate a node embedding  $(d = 512)$  based on the edge count of each graph  $(|E|)$ , with the best line fit drawn. Our method is over 9,000 times faster than FastRP and uses over 8,000 times less memory than VERSE, the next most efficient baselines respectively, in the largest graph that these baseline methods can process.  
(b) Memory Usage

Table 3: Average Micro-F1 classification scores and confidence intervals. Our method is marked as follows: * - above baselines; bold - no other method is statistically significant better.  

<table><tr><td>Method\Dataset</td><td>PPI</td><td>BlogCatalog</td><td>CoCit</td><td>Flickr</td><td>YouTube</td><td>Orkut</td></tr><tr><td>DeepWalk</td><td>16.08 ± 0.64</td><td>32.48 ± 0.35</td><td>37.44 ± 0.67</td><td>31.22 ± 0.38</td><td>38.69 ± 1.17</td><td>87.67 ± 0.23</td></tr><tr><td>node2vec</td><td>15.03 ± 3.18</td><td>33.67 ± 0.93</td><td>38.35 ± 1.75</td><td>29.80 ± 0.67</td><td>36.02 ± 2.01</td><td>DNC</td></tr><tr><td>VERSE</td><td>12.59 ± 2.54</td><td>24.64 ± 0.85</td><td>38.22 ± 1.34</td><td>25.22 ± 0.20</td><td>36.74 ± 1.05</td><td>81.52 ± 1.11</td></tr><tr><td>FastRP</td><td>15.74 ± 2.19</td><td>33.54 ± 0.96</td><td>26.03 ± 2.10</td><td>29.85 ± 0.26</td><td>22.83 ± 0.41</td><td>DNC</td></tr><tr><td>InstantEmbedding</td><td>17.67* ± 1.22</td><td>33.36 ± 0.67</td><td>39.95* ± 0.67</td><td>30.43 ± 0.79</td><td>40.04* ± 0.97</td><td>76.83 ± 1.16</td></tr></table>

The results of running time and memory analysis confirm hypothesis H1 and show that Instant-Embedding has a significant speed and space advantage versus the baselines. The relative speedup continues to grow as the size of the datasets increase. On a dataset with over 1 billion edges (Friendster), we can compute an embedding in  $80\mathrm{ms}$  - fast enough for a real-time application!

# 4.3 EMBEDDING QUALITY

Node Classification. This task measures the semantic information preserved by the embeddings by training a simple classifier on a small fraction of labeled representations. For each method, we perform three different random splits of the data. More details are available in the Appendix B.4.1.

In Table 3 we report the mean Micro F1 scores with their respective confidence intervals (corresponding Macro-F1 scores in the supplementary material). For each dataset, we perform Welch's t-test between our method and the best performing contender. We observe that InstantEmbedding is remarkably good on these node classification, despite its several approximations and locality restriction. Specifically, on four out of five datasets, no other method is statistically significant above ours, and three of these (PPI, CoCit and YouTube) we achieve the best classification results.

In Figure 2, we study how our hyperparameter, the PPR approximation error  $\epsilon$ , influences both the classification performance, running time, and memory consumption. There is a general sweet spot (around  $\epsilon = 10^{-5}$ ) across datasets where InstantEmbedding outperforms competing methods while being orders of magnitude faster. Data on the other datasets is available in Section B.5.

![](images/c073bc734d749b5a36218a4b5150762e06d56b239f91471e6205dcdbd04023c9.jpg)  
Figure 2: The impact of the choice of  $\epsilon$  on the quality of the resulting embedding (through the Micro-F1 score), average running time and peak memory increase for the YouTube dataset.

Table 4: Average ROC-AUC scores and confidence intervals for the link prediction task. Our method is marked as follows: * - above baselines; bold - no other method is statistically significant better.  

<table><tr><td>Method \ Dataset</td><td>CoAuthor</td><td>Blogcatalog</td><td>Youtube</td><td>Amazon2M</td></tr><tr><td>DeepWalk</td><td>88.43 ± 1.08</td><td>91.41 ± 0.67</td><td>82.17 ± 1.02</td><td>98.79 ± 0.41</td></tr><tr><td>node2vec</td><td>86.09 ± 0.85</td><td>92.18 ± 0.12</td><td>81.27 ± 1.58</td><td>DNC</td></tr><tr><td>VERSE</td><td>92.75 ± 0.73</td><td>93.42 ± 0.35</td><td>80.03 ± 0.99</td><td>99.67 ± 0.18</td></tr><tr><td>FastRP</td><td>82.19 ± 2.22</td><td>88.68 ± 0.70</td><td>76.30 ± 1.46</td><td>92.12 ± 0.61</td></tr><tr><td>InstantEmbedding</td><td>90.44 ± 0.48</td><td>92.74 ± 0.60</td><td>82.89* ± 0.83</td><td>99.15 ± 0.18</td></tr></table>

Link prediction. We conduct link prediction experiments to assess the capability of the produced representations to model hidden connections in the graph. For the dataset which has temporal information (CoAuthor), we select data until 2014 as training data, and split co-authorship links between 2015-2016 in two balanced partitions that we use as validation and test. For the other datasets, we uniformly sample  $80\%$  of the available edges as training (to learn embeddings on), and use the rest for validation  $(10\%)$  and testing  $(10\%)$ . Over repeated runs, we vary the splits. More details about the experimental design are available in the supplementary material. We report results for each method in in Table 4, which shows average ROC-AUC and confidence intervals for each method. Across the datasets, our proposed method beats all baselines except VERSE, however we do achieve the best performance on YouTube by a statistically significant margin.

Visualization. Figure 3 presents UMAP (McInnes et al., 2018) projections on the CoCit dataset, where we grouped together similar conferences. We note that our sublinear approach is especially well suited to visualizing graph data, as visualization algorithms only require a small subset of points (typically downsampling to only thousands) to generate a visualization for datasets.

The experimental analysis of node classification, link prediction, and visualization show that despite relying on two different approximations (PPR & random projection), InstantEmbedding is able to very quickly produce representations which meet or exceed the state of the art in unsupervised representation learning for graph structure, confirming hypothesis H2. We remark that interestingly InstantEmbedding seems slightly better at node classifications than link prediction. We suspect that the randomization may effectively act as a regularization which is more useful on classification.

![](images/68532a4cd08df5ee02c8f405e48884947770c5be86fdba55bc5ba1d87e242e29.jpg)  
(a) DeepWalk

![](images/1dc3ab86b901725046e7be6dce349c9487803ac891f6564923dd899cf88e8526.jpg)  
(b) VERSE

![](images/fe9e39ac9f496e0cca52eabfa972ac27ecfa117c2981f85b102ad83db4917100.jpg)  
Figure 3: UMAP visualization of CoCit  $(d = 512)$ . Research areas (■ ML, □ DM, ▢ DB, ▣ IR).  
(c) FastRP

![](images/a4c173571a20d1fd47ca58eb58e9661b022415a49ea7ffa970b3b24ff43626f2.jpg)  
(d) InstantEmbedding

# 5 CONCLUSION

The present work has two main contribution: a) introducing and formally defining the Local Node Embedding problem and b) presenting InstantEmbedding, a highly efficient method that selectively embeds nodes using only local information, effectively solving the aforementioned problem. As existing graph embedding methods require accessing the global graph structure at least once during the representation generating process, the novelty brought by InstantEmbedding is especially impactful in real-world scenarios where graphs outgrow the capabilities of a single machine, and annotated data is scarce or expensive to produce. Embedding selectively only the critical subset of nodes for a task makes many more applications feasible in practice, while reducing the costs for others.

Furthermore, we show theoretically that our method embeds a single node in space and time sublinear to the size of the graph. We also empirically prove that InstantEmbedding is capable of surpassing state-of-the-art methods, while being many orders of magnitude faster than them - our experiments show that we are over 9,000 times faster on large datasets and on a graph over 1 billion edges we can compute a representation in 80ms.

# REFERENCES

Microsoft academic graph (mag) - kkd cup 2016. https://www.kdd.org/kdd-cup/view/kdd-cup-2016/Data, 2016.  
Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou, and Alexander A Alemi. Watch your step: Learning node embeddings via graph attention. In Advances in Neural Information Processing Systems, pp. 9180–9190, 2018.  
Dimitris Achlioptas. Database-friendly random projections: Johnson-lindenstrauss with binary coins. Journal of computer and System Sciences, 66(4):671-687, 2003.  
Reid Andersen, Fan Chung, and Kevin Lang. Using pagerank to locally partition a graph. *Internet Mathematics*, 4(1):35-64, 2007.  
Haochen Chen, Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. A tutorial on network embeddings. arXiv preprint arXiv:1808.02590, 2018.  
Haochen Chen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and Steven Skiena. Fast and accurate network embeddings via very sparse random projection. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pp. 399-408, 2019.  
Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2019. URL http://web.cs.ucla.edu/~chohsieh/data/Amazon2M.tar.gz.  
Ashish Chiplunkar, Michael Kapralov, Sanjeev Khanna, Aida Mousavifar, and Yuval Peres. Testing graph clusterability: Algorithms and lower bounds. In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS), pp. 497-508. IEEE, 2018.  
Thomas H Cormen, Charles E Leiserson, Ronald L Rivest, and Clifford Stein. Introduction to algorithms. MIT press, 2009.  
Artur Czumaj and Christian Sohler. Testing expansion in bounded-degree graphs. Combinatorics, Probability and Computing, 19(5-6):693-709, 2010.  
Artur Czumaj, Pan Peng, and Christian Sohler. Testing cluster structure of graphs. In Proceedings of the forty-seventh annual ACM symposium on Theory of Computing, pp. 723-732, 2015.  
Anirban Dasgupta, Ravi Kumar, and Tamás Sarlós. A sparse Johnson: Lindenstrauss transform. In Proceedings of the forty-second ACM symposium on Theory of computing, pp. 341-350, 2010.  
Chi Thang Duong, Thanh Dat Hoang, Ha The Hien Dang, Quoc Viet Hung Nguyen, and Karl Aberer. On node features for graph neural networks. arXiv preprint arXiv:1911.08795, 2019.  
Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 855-864, 2016.  
Pankaj Gupta, Ashish Goel, Jimmy Lin, Aneesh Sharma, Dong Wang, and Reza Zadeh. Wtf: The who to follow service at twitter. In Proceedings of the 22nd international conference on World Wide Web, pp. 505-514, 2013.  
Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 297-304, 2010.  
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web, pp. 173-182, 2017.  
Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. Lightgen: Simplifying and powering graph convolution network for recommendation. arXiv preprint arXiv:2002.02126, 2020.

Glen Neh and Jennifer Widom. Scaling personalized web search. In WWW, 2003.  
Satyen Kale and C Seshadhri. Testing expansion in bounded degree graphs. 35th ICALP, pp. 527-538, 2008.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Johannes Klicpera, Aleksandar Bojchevski, and Stephan Gunnemann. Combining neural networks with personalized pagerank for classification on graphs. In International Conference on Learning Representations, 2019.  
Joseph B Kruskal. Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29(1):1-27, 1964.  
Omer Levy and Yoav Goldberg. Neural word embedding as implicit matrix factorization. In Advances in neural information processing systems, pp. 2177-2185, 2014.  
Peter A Lofgren, Siddhartha Banerjee, Ashish Goel, and C Seshadhri. Fast-ppr: Scaling personalized pagerank estimation for large graphs. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 1436–1445, 2014.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579-2605, 2008.  
Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger. Umap: Uniform manifold approximation and projection. Journal of Open Source Software, 2018.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 26, pp. 3111-3119. 2013.  
Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. Asymmetric transitivity preserving graph embedding. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 1105-1114, 2016.  
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.  
Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 701-710, 2014.  
Jiezhong Qiu, Yuxiao Dong, Hao Ma, Jian Li, Kuansan Wang, and Jie Tang. Network embedding as matrix factorization: Unifying deepwalk, line, pte, and node2vec. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pp. 459-467, 2018.  
J Ben Schafer, Dan Frankowski, Jon Herlocker, and Shilad Sen. Collaborative filtering recommender systems. In The adaptive web, pp. 291-324. Springer, 2007.  
Chris Stark, Bobby-Joe Breitkreutz, Teresa Reguly, Lorrie Boucher, Ashton Breitkreutz, and Mike Tyers. Biogrid: a general repository for interaction datasets. *Nucleic acids research*, 34(suppl_1): D535–D539, 2006. https://snap.stanford.edu/node2vec/Homo_sapiens.mat.  
Jukka Suomela. Survey of local algorithms. ACM Computing Surveys (CSUR), 45(2):1-40, 2013.  
Lei Tang and Huan Liu. Social dimension approach to classification in large-scale networks. 2010. URL http://leitang.net/social_dimention.html.  
Anton Tsitsulin, Davide Mottin, Panagiotis Karras, and Emmanuel Müller. Verse: Versatile graph embeddings from similarity measures. In Proceedings of the 2018 World Wide Web Conference, pp. 539-548, 2018.

Anton Tsitsulin, Marina Munkhoeva, Davide Mottin, Panagiotis Karras, Ivan Oseledets, and Emmanuel Müller. Frede: Linear-space anytime graph embeddings. arXiv preprint arXiv:2006.04746, 2020.  
Petar Velicković, William Fedus, William L Hamilton, Pietro Lio, Yoshua Bengio, and R Devon Hjelm. Deep graph infomax. arXiv preprint arXiv:1809.10341, 2018.  
Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola, and Josh Attenberg. Feature hashing for large scale multitask learning. In Proceedings of the 26th annual international conference on machine learning, pp. 1113-1120, 2009.  
Milian Wolff. A heap memory profiler for linux, 2018. https://github.com/KDE/heaptrack.  
Jaewon Yang and Jure Leskovec. Defining and evaluating network communities based on ground-truth. Knowledge and Information Systems, 42(1):181-213, 2015. https://snap.stanford.edu/data/com-Orkut.html https://snap.stanford.edu/data/com-Friendster.html.  
Ziwei Zhang, Peng Cui, Haoyang Li, Xiao Wang, and Wenwu Zhu. Billion-scale network embedding with iterative random projection. In 2018 IEEE International Conference on Data Mining (ICDM), pp. 787-796. IEEE, 2018.
