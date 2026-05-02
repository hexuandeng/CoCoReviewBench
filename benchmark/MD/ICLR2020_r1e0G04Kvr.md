# DEEP GRAPH TRANSLATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep graph generation models have achieved great successes recently, among which however, are typically unconditioned generative models that have no control over the target graphs given an input graph. In this paper, we propose a novel Graph-Translation-Generative-Adversarial-Networks (GT-GAN) that transforms the input graphs into their target output graphs. GT-GAN consists of a graph translator equipped with innovative graph convolution and deconvolution layers to learn the translation mapping considering both global and local features. A new conditional graph discriminator is proposed to classify the target graphs by conditioning on input graphs while training. Extensive experiments on multiple synthetic and real-world datasets demonstrate that our proposed GT-GAN significantly outperforms other baseline methods in terms of both effectiveness and scalability. For instance, GT-GAN achieves at least 10X and 15X faster runtimes than GraphRNN and RandomVAE, respectively, when the size of the graph is around 50.

# 1 INTRODUCTION

In recent years, deep learning on graphs has seen a surge of interests, especially for graph representation and recognition tasks such as node-level classification (Li et al., 2016; Kipf & Welling, 2017; Velicković et al., 2017; Gilmer et al., 2017; Hamilton et al., 2017) and graph-level classification (Niepert et al., 2016; Atwood et al., 2016; Wu et al., 2017). Because of the successes in graph neural networks, researchers have recently started to explore the use of deep generative models for graph synthesis on practical applications such as designing new chemical molecular structures (Simonovsky & Komodakis, 2018; You et al., 2018). This has led to many of the recent advances in deep graph generative models where some of these approaches are domain dependent models (Kusner et al., 2017; Dai et al., 2018) for generating graphs with physical constrains, while some others consider the generation of generic graphs (Li et al., 2018; Samanta et al., 2018; Jin et al., 2018a).

However, there are two main drawbacks of existing deep graph generative models. First, one significant limitation of the previous approaches is that most of these models are only suitable for small graphs with 40 or fewer nodes, which is mainly due to their one-node-per-step generation manner. More importantly, most of the existing graph generation models are unconditioned and thus ignore rich input graph information for generating a new graph. In many applications, it is crucial to guide the graph generation process by conditioning on an input graph, which can be cast as a graph translation learning problem – translating the input graph to the output graph.

One straightforward way is to build a translation system by using a graph encoder-decoder architecture. However, there are several challenges for this type of approaches: 1) how to learn one-to-more mapping between the input graph and the target graphs. Different from the plain graph generation problem, a conditional graph synthesis task is to learn a distribution of target graphs conditioning on the input graph, which aims to capture the underlying implicit properties of the graphs, such as their scale-free characteristic. 2) how to jointly learn both local and global information for translation. One needs to not only learn the translation mapping in the local information (i.e. neighborhood pattern of each node), but also in the global property of the whole graph (e.g., node degree distribution or graph density).

To address the aforementioned challenges, we present a novel neural network architecture - Graph-Translation-Generative-Adversarial-Nets (GT-GAN). We first propose a conditional graph GAN architecture that consists of an encoder-decoder translator and a conditional graph discriminator to learn the one-to-more mapping (a conditional distribution) for graph translation. To jointly embed

the local and global information, we present a novel graph encoder including both the edge and the node convolution layers. In addition, we further propose a novel graph U-net with graph skips and dedicated graph deconvolution layers including both the edge and the node deconvolution layers. Finally, GT-GAN is scalable with at most quadratic computation and memory consumption in terms of the number of nodes in a graph, making it suitable for at least modest-scale graphs (with hundreds of nodes, compared to the tens of nodes in most of existing graph generative models).

We highlight our main contributions as follows:

- We develop a generic framework GT-GAN consisting of a novel graph translator and conditional graph discriminator for learning a conditional distribution of target graphs given the input graphs.  
- We propose a novel graph encoder consisting of "edge convolution" layers that extract various relations among nodes containing both local and global information, and "node convolution" layers that embed the node representations based on the extracted relations.  
- We propose a novel graph decoder consisting of the "edge deconvolution" and "node deconvolution" layers, which can decode the node representations first into the latent relations of the target graph and then generate the final target graph. The graph skip-connection is also utilized to map the learned latent relations between the input and target graphs.  
- Extensive experiments have been conducted on both synthetic and real-world datasets on eight performance metrics to demonstrate the effectiveness and efficiency of the proposed model.

# 2 RELATED WORKS

Graph Neural Networks. The recent surge of research into GNN (Graph Neural Networks) can be generally divided into two categories: Graph Recurrent Networks and Graph Convolutional Networks. Graph Recurrent Networks originate from early work by Gori et al. (2005); Scarselli et al. (2009) and are based on recursive neural networks that have been extended by modern deep learning techniques such as gated recurrent units (Li et al., 2016). The other category, Graph Convolutional Networks, originate from spectral graph convolutional neural networks (Bruna et al., 2014), which were then extended by Defferrard et al. (2016) using fast localized convolutions, and further approximated by an efficient architecture for a semi-supervised setting proposed by Kipf & Welling (2017). Self-attention mechanism and subgraph-level information are also explored later to further improve the representation power of learned node embeddings (Veličković et al., 2017; 2018; Bai et al., 2019).

Graph generation. Most of the existing GNN based graph generation for general graphs have been proposed in the last two years and are based on VAE (Simonovsky & Komodakis, 2018; Samanta et al., 2018) and generative adversarial nets (GANs) (Bojchevski et al., 2018), among others (Li et al., 2018; You et al., 2018). Most of these approaches generate nodes and edges sequentially to form a whole graph, leading to the issues of being sensitive to the generation order and very time-consuming for large graphs. Differently, GraphRNN (You et al., 2018) builds an autoregressive generative model on these sequences with LSTM model and has demonstrated its good scalability.

Data Translation involved Graphs. A variety of graph-to-sequence models have been proposed to cope with different tasks including machine translation (Beck et al., 2018; Bastings et al., 2017), semantic parsing (Xu et al., 2018a;b; Song et al., 2018), and question generation (Chen et al., 2019), and health status prediction (Gao et al., 2019). The sequence-to-graph algorithms are generally popular with those working on NLP methods, including generating dependency graphs (Gildea et al., 2018; Wang et al., 2018) and AMR structures (Peng et al., 2018). A few of very recent attempts have also been made to develop graph-to-graph translation models. Jin et al. (2018b) proposed a domain-specific graph translation model to deal with molecular optimization task by utilizing the domain knowledge - junction tree and molecule graph. Do et al. (2019) dealt with the chemical reaction product prediction problem by predicting the reaction sequences based on the input graph of molecules. Sun & Li (2019) proposed a RNN based model for encoding and decoding the directed acyclic graph (converted from regular graphs), which can be viewed as a contemporary work to our work. However, this method is trained following the encoder-decoder architecture but in a supervised setting instead of learning a distribution of graphs. More importantly, it is difficult to scale to even modest-scale graph due to its one-node-per-step generation manner.

![](images/29a072129adccf1e7511ee0f9b8dac7454c903d1183ef1cfcbfc9e9f09ed41f8.jpg)  
Figure 1: GT-GANs consisting of a graph translator and a conditional graph discriminator. A novel graph encoder and decoder are designed for the graph translation problem.

# 3 THE OVERALL ARCHITECTURE OF GT-GAN

In this section, we first present our problem formulation of graph translation problem. We then propose our new GT-GAN model for graph translation and discuss each component in detail in the subsequent sections.

# 3.1 PROBLEM FORMULATION FOR DEEP GRAPH TRANSLATION

Our goal is to learn an end-to-end translation mapping from an input graph to a target graph. Let an input graph  $G_{X} = (\mathcal{V}, \mathcal{E}, A, S)$  such that  $\mathcal{V}$  is the set of  $N$  nodes,  $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$  is the set of edges, and  $A \in \mathbb{R}^{N \times N}$  is an adjacency matrix (binary or weighted), where  $G_{X}$  can be weighted or unweighted, directed or undirected. Let  $S \in \mathbb{R}^{N \times F}$  be a node feature matrix with each row representing a node feature vector  $S_{i}$ . Denote  $e_{i,j} \in \mathcal{E}$  as an edge from the node  $v_{i} \in \mathcal{V}$  to  $v_{j} \in \mathcal{V}$ ;  $A_{i,j} \in A$  therefore denotes the corresponding weight of the edge  $e_{i,j}$ . Similarly, we define a target graph  $G_{Y} = (\mathcal{V}', \mathcal{E}', A', S')$  that shares the same node sets and node features with  $G_{X}$  but with different topology and connection weights. Formally, graph translation is to learn a translator from an input graph  $G_{X} \in \mathcal{G}_{X}$  with a random noise  $U$  to generate a target graph  $G_{Y} \in \mathcal{G}_{Y}$ , where  $\mathcal{G}_{X}$  and  $\mathcal{G}_{Y}$  denote the domains of the input and target graphs, respectively. The translation mapping is denoted as  $T: U, G_{X} \to G_{Y}$ .

Note that since our aim is to learn a conditional distribution of the target graphs given an input graph, we can cast the graph translation as a conditional graph generation problem, where an input graph can be mapped into any target graph that may have different topologies yet follow the same distribution. In contrast, the graph generation, that are designed to learn a distribution of graphs and generate a new graph sample based on this distribution, typically uses variational autoencoder framework for graph generation. Therefore, the previous graph generation frameworks such as graphVAE (Simonovsky & Komodakis, 2018) and GraphRNN (You et al., 2018) do not directly fit into "translation" setting.

The Proposed GT-GAN Framework. Fig.1 shows our proposed generic GAN framework for graph translation that consists of a graph translator  $\mathcal{T}$  and a conditional graph discriminator  $\mathcal{D}$ . In this figure we assume the node feature has only one dimension for simplicity. Since our task is to train a conditional generator with "one-to-many mapping" instead of a deterministic one, the noise  $U$  is introduced by the dropout function (Seltzer et al., 2013) in each convolution and deconvolution layer, as shown (in green lines) in Fig.1. Our graph translator  $\mathcal{T}$  is trained to produce target graphs that cannot be distinguished from "real" ones by our conditional graph discriminator  $\mathcal{D}$ . Specifically, the generated target graph  $G_{Y'} = \mathcal{T}(G_X, U)$  cannot be distinguished from the real one,  $G_Y$ , based on the current input graph  $G_X$ .  $\mathcal{T}$  and  $\mathcal{D}$  undergo an adversarial training process based on input and target graphs by solving the following the loss function:

$$
\mathcal {L} (\mathcal {T}, \mathcal {D}) = \mathbb {E} _ {G _ {X}, G _ {Y}} [ \log \mathcal {D} (G _ {Y} | G _ {X}) ] + \mathbb {E} _ {G _ {X}, U} [ \log (1 - \mathcal {D} (\mathcal {T} (G _ {X}, U) | G _ {X})) ], \tag {1}
$$

where  $\mathcal{T}$  tries to minimize this objective while an adversarial  $\mathcal{D}$  tries to maximize it, i.e.  $\mathcal{T}^{*} = \arg \min_{\mathcal{T}}\max_{\mathcal{D}}\mathcal{L}(\mathcal{T},\mathcal{D})$ . We also mix the GAN loss with the L1 loss to enforce sparsity similarity, which is also found useful in image translation problem (Isola et al., 2017),

$$
\mathcal {L} _ {l 1} (\mathcal {T}) = \mathbb {E} _ {A, A ^ {\prime}, U} [ \| A ^ {\prime} - T (G _ {X}, U) \| _ {1} ], \tag {2}
$$

where  $T(G_{X}, U)$  refers to the adjacent matrix of generated graph. The training process is a trade-off between  $\mathcal{L}_{l1}$  and  $\mathcal{L}(\mathcal{T}, \mathcal{D})$ , which jointly enforces  $\mathcal{T}(G_{X}, U)$  and  $G_{Y}$  to follow a similar, but not necessarily identical topological pattern. Specifically,  $\mathcal{L}_{l1}$  makes  $\mathcal{T}(G_{X}, U)$  share the same rough outline of sparsity pattern as  $G_{Y}$ , while  $\mathcal{L}(\mathcal{T}, \mathcal{D})$  allows  $\mathcal{T}(G_{X}, U)$  to vary to some degree. Thus, the optimal objective  $\mathcal{T}^*$  of the translator, which generates graphs that are as "real" as possible, is defined as:

$$
\mathcal {T} ^ {*} = \arg \min  _ {\mathcal {T}} \max  _ {\mathcal {D}} \mathcal {L} (\mathcal {T}, \mathcal {D}) + \mathcal {L} _ {l 1} (\mathcal {D}), \tag {3}
$$

The graph translator  $\mathcal{T}$  is an encoder-decoder architecture, where we propose a new graph encoder to obtain the node representations of the input graph and propose the graph deconvolution with skips to generate the target graph, as shown in Fig.1, which we elaborated in the followings sections.

# 3.2 GRAPH ENCODER

The graph encoder aims to learn the representations of nodes based on the node features and graph topology of the input graph. One of crucial challenges is to learn both local and global information in the graph embedding. For instance, when learning translation between two scale-free graphs, one needs to translate both the local information (i.e. n-hop neighborhood of each node) and the scale-free property (i.e. node degree distributions of whole graph) from an input graph to a target graph.

The Proposed Graph Convolution. To learn the local information, the proposed encoder learns each

node representation based on its n-hop neighbors. To learn the global information, it learns each node representation by looking for more "virtual neighbors" regarding the latent relations from the aspect of the whole graph. Thus, we first propose the "edge convolution" layers to learn a group of multi-mode relations from the topology of the input graph, which can include both the n-hop connections and the latent relations that are derived from their adjacent edges/relations as shown in Fig.2(a). And then the "node convolution" layer is used to embed each node representations by aggregating its "virtual neighbors" that related to each latent relations, as shown in Fig.2(b).

In each "edge convolution" layer, each node pair's latent relation is computed by its adjacent edges or the extracted adjacent relations from the last layer. In the directed graph, each node have incoming edge(s) and out-going edge(s). Thus, there are two learnable parametric vectors  $\phi$  and  $\psi$  as convolution filters for two directions to convolute the adjacent edges/relations for each node pairs. The relation  $E_{i,j}^{l,m}$  in the  $m$ th relation mode of the  $l$ th layer is learned by the out-going edges/relations of node  $v_{i}$  and the in-coming edges/relations of node  $v_{j}$ ,

$$
E _ {i, j} ^ {l, m} = \sum_ {n = 1} ^ {R _ {l - 1}} \left(\sigma \left(\sum_ {k _ {1} = 1} ^ {N} E _ {i, k _ {1}} ^ {l - 1, n} \phi_ {k _ {1}} ^ {l, m}\right) + \sigma \left(\sum_ {k _ {2} = 1} ^ {N} E _ {k _ {2}, j} ^ {l - 1, n} \psi_ {k _ {2}} ^ {l, m}\right)\right) \tag {4}
$$

where  $E_{i,j}^{1,1} \equiv A$  and  $\phi^{l,m} \in \mathbb{R}^{N \times 1}$  refers to the filter vector to be learned and  $\phi_{k_1}^{l,m}$  refers to the element of  $\phi^{l,m}$  that is related to node  $v_{k_1}$ .  $R_{l-1}$  refers to the number of relation modes extracted for the  $(l-1)$ th layer of the graph encoder.

After learning the various modes of relations, the "node convolution" layer learns each node's representations by aggregating its "virtual neighbors" in terms of each mode of relation. The  $m$ th feature vector of node representation tensor  $\bar{H}_i^m\in \mathbb{R}^{1\times F}$  for node  $v_{i}$  is computed as:

$$
\bar {H} _ {i} ^ {m} = \sum_ {n = 1} ^ {R _ {l - 1}} \left(\sigma \left(\sum_ {k _ {1} = 1} ^ {N} E _ {i, k _ {1}} ^ {l - 1, n} \mu_ {k _ {1}} ^ {m} S _ {k _ {1}}\right) + \sigma \left(\sum_ {k _ {2} = 1} ^ {N} E _ {k _ {2}, i} ^ {l - 1, n} \nu_ {k _ {2}} ^ {m} S _ {k _ {2}}\right)\right), \tag {5}
$$

where  $\bar{H}_i\in \mathbb{R}^{R_l\times F}$  and  $R_{l}$  refers to the number of feature vectors in the "node convolution" layer. Here  $\mu^m,\nu^m\in \mathbb{R}^{N\times 1}$  refer to the filter vectors for the two directions to be learned and  $\mu_{k_1}^m$  refers to the element of  $\mu^m$  that is related to node  $v_{k_1}$ .  $\bar{H}_i$  is then flattened and transformed into a node representation vector  $H_{i}\in \mathbb{R}^{1\times C}$  by a fully connected layer.  $C$  is the length of the node representation. Note that our graph encoder is designed for a directed graph, and it is easily generalized to an undirected graph, where the weight vector is shared by both directions.

![](images/1d1edd1e3ecb611511339fecaf2fd2f85e949cd4eade43380e4d5de43bc6e939.jpg)  
Figure 2: Graph convolution and deconvolution

# 3.3 GRAPH DECODER

The decoder aims to generate the edges of the target graph by taking the extracted latent information of the input graph. It is straightforward to directly use the embedded node representation of the last layer to generate the target graph. However, the extracted information from each layer in the encoder could also be useful for generating the target graph. Thus, we consider all possible information learned in the encoder to be fed into a graph decoder.

Motivated by these observations, we propose a graph U-Net consisting of graph skips and dedicated graph deconvolution layers. The graph deconvolution decodes the single node (or edge) information to yield its incoming and outgoing adjacent edges as a mirrored graph convolution process. In addition, several skips are implemented to map the learned information of each layer in the encoder to mirror the corresponding layers in the decoder. Similar Graph U-Net was proposed in (Gao & Ji, 2019). The key difference is that their U-Net is barely a graph embedding method by using the old graph topology from pooling part to embed nodes during unpooling part. However, our Graph U-Net can not only do node embedding in graph encoder but also generate the new graph's topology in the graph decoder, which is necessary for the graph translation problem.

The proposed Graph Deconvolution. The proposed graph deconvolution technique incorporates both "node deconvolution" and "edge deconvolution" layers. First, the "node deconvolution" layer are used to generate the latent multi-mode relations of the target graph based on the learned latent node representations. As shown in Fig. 2(c), "node deconvolution" is a reversed process of the "node" convolution. Since each node has an influence to its relations connecting to other nodes. Then the relation  $E_{i,j}^{1,m}$  between node  $v_{i}$  and node  $v_{j}$  in the  $m$ th relation mode of the  $l$ th "node" deconvolution layer in the decoder can be computed as follows:

$$
E _ {i, j} ^ {1, m} = \sum_ {n = 1} ^ {C} \left(\sigma \left(H _ {i} ^ {n} \bar {\mu} _ {j} ^ {m}\right) + \sigma \left(H _ {j} ^ {n} \bar {\nu} _ {i} ^ {m}\right)\right), \tag {6}
$$

where  $\sigma(H_i^n \bar{\mu}_j^m)$  means the deconvolution contribution of node  $v_i$  to its relation with node  $v_j$  made by the  $n$ th element of its node representations, and  $\bar{\mu}_j^m$  represents the element of the deconvolution filter vector  $\bar{\mu}^m \in \mathbb{R}^{1 \times N}$  that is related to node  $v_j$ .

We can now recursively apply our proposed "edge deconvolution" layer to decode the latent relation between each pair of nodes from the upper layer to those of lower layer. As a reversed way of "edge" convolution, the relation of each pair of nodes in the  $(l - 1)$ th layer can make contribution to generating itself and its adjacent relations in the  $l$ th layer, as shown in Fig. 2(d). Thus, the relation  $E_{i,j}^{l,m}$  between node  $v_{i}$  and node  $v_{j}$  in the  $l$ th layer is computed as follows:

$$
E _ {i, j} ^ {l, m} = \sum_ {n = 1} ^ {R _ {l - 1} ^ {\prime}} \left(\sigma \left(\bar {\phi} _ {j} ^ {l, m} \sum_ {k _ {1} = 1} ^ {N} E _ {i, k _ {1}} ^ {l - 1, n}\right) + \sigma \left(\bar {\psi} _ {j} ^ {l, m} \sum_ {k _ {2} = 1} ^ {N} E _ {k _ {2}, j} ^ {l - 1, n}\right)\right), \tag {7}
$$

where  $\bar{\phi}^{l,m}\sum_{k_1 = 1}^N E_{i,k_1}^{l - 1,n}$  is interpreted as the decoded contribution of node  $i$  to its relations with node  $v_{j}$ , and  $\bar{\phi}^{l,m}$  refers to the element of deconvolution filter vector that is related to node  $v_{j}$ .  $R_{l - 1}^{\prime}$  refers to the number of relation modes extracted by the  $(l - 1)\mathrm{th}$  layer in the graph decoder. The output of the last "edge" deconvolution layer denotes the edges of the target graph.

Skipping for graph deconvolution. Based on the graph deconvolution above, it is possible to utilize skipps to link the extracted latent relation sets of each layers in the graph encoder with those in the graph decoder. Specifically, the output of the  $l$ th "edge deconvolution" layer with  $R_{l}$  channels in the decoder is concatenated with the output of the  $l$ th "edge convolution" layer with  $R_{l}^{\prime}$  channels in encoder to form joint  $R_{l} + R_{l}^{\prime}$  channels, which are then input into the  $(l + 1)th$  deconvolution layer.

# 3.4 CONDITIONAL GRAPH DISCRIMINATOR

The graph discriminator must distinguish between the "translated" target graph and the "real" ones based on the input graphs, as this helps to train the generator in an adversarial way. Technically, this requires the discriminator to accept two graphs simultaneously as inputs (a target graph and an input graph or a generated graph and an input graph), and classify the two graphs as either related or not. Thus, we propose a conditional graph discriminator (CGD) which leverages the same graph convolution layers in the translator for the graph classification, as shown in Fig.1. Specifically, the input and target graphs are both ingested by CGD and stacked into a  $N \times N \times 2$  tensor which can be considered a 2-channel input. After obtaining the node representations, the graph-level embedding is computed by summing these node embeddings. Finally, a softmax layer is implemented to distinguish the input graph-pair from the real graph or generated graph.

# 3.5 COMPUTATIONAL COMPLEXITY ANALYSIS

The graph encoder and decoder shares the same time complexity. Without loss of generality, we assume all the hidden layers have the same number of feature maps as  $M$ .  $P$  is the length of the fully connected layer in CGD. The worst-case total complexity of GT-GAN (i.e., the dense graph) is now  $O(9N^2M^2 + 3N^2M^2 + N^2MP)$ , where the first, second, and third terms represent "edge convolutions", "node convolutions", and fully connected layers in the graph discriminator, respectively. Similarly, the total memory consumption for GT-GAN is  $O((9NM^2 + 9N^2M) + (3NM^2 + 3NM) + (N^2MP + P))$ . In practice, many graphs are likely to be sparse, thus it further reduces the computational and memory cost to  $O(N)$  by using sparse matrix-vector operations (You et al., 2018), which paves the way toward modest scale graphs with hundreds or thousands of nodes.

# 4 EXPERIMENT

This section reports the results of extensive experiments and ablation studies carried out to test the performance of GT-GAN on two synthetic and two real-world datasets. All experiments were conducted on a 64-bit machine with Nvidia GPU (GTX 1070,1683 MHz, 8 GB GDDR5). The code and data utilized are available at https://github.com/anonymous1025/Deep-Graph-Translation-.

# 4.1 DATASETS

The experimental settings for each dataset were as follows. The rules for generating synthetic input-target graph pairs and the process of collecting the real-world graphs is provided in Appendix.

Two synthetic datasets: Two groups of synthetic datasets were used to validate the performance of the proposed GT-GAN: a scale-free graph dataset and a Poisson-random graph dataset. Each group has five subsets with different graph sizes (number of nodes): 10, 20, 50, 100 and 150. Each subset consists of 5000 input-target graph pairs; 2500 pairs were used for training and the remaining 2500 for testing.

User authentication datasets. The goal of this application was to forecast future potential malicious authentication graphs given the user's normal authentication graph. Each user authentication graph is a directed weighted graph, where nodes represent computers and the weights of the edges represent the authentication activities at certain frequencies. There are 78 pairs of graphs (malicious and normal behavior) of graph size 50 and 315 pairs of graphs of graph size 300 from 97 users in two subsets. We performed a 2-fold cross-validations and 3-fold cross-validation, respectively, for the two subsets.

Internet of Things (IOT) datasets. This application focused IOT network malware confinement prediction (predicting optimal network operation given a compromised one). There are three subsets of graph pairs with different sizes (20, 40 and 60), where the nodes represent devices and the node attributes indicating the compromised status of the nodes. The weights of the edges represent the distance between two devices. There are 334 pairs of input (compromised IOT) and target graphs (optimal IOT) in each subset and each is divided into two parts for the 2-fold cross-validation.

# 4.2 BASELINE METHODS

We compare our GT-GAN against five state-of-the-art graph generation methods: 1) GraphRNN (You et al., 2018) is a new graph generation method based on sequential generation with the LSTM model; 2) GraphVAE (Simonovsky & Komodakis, 2018) is a probability-based graph generation method for small graphs; 3) GraphGMG (Li et al., 2018) is a framework based upon graph neural networks for small single graphs; 4) RandomVAE (Samanta et al., 2018) was described earlier; and 5) S-Generator is the part of our full model GT-GAN, which essentially is a graph translator with L1 loss but no discriminator. We propose this S-Generator model in order to evaluate the necessity of the proposed GT-GAN framework to learn the one-to-many mappings. All the comparison methods were trained on the malicious graphs without conditioning on the input graphs due to the models' inherent capability limitations. The datasets were assigned to each comparison model for the experiment based on their scalability in terms of graph size.

Table 1: Node degree distribution distance between the generated and real graphs scale-free graphs  

<table><tr><td>Size</td><td>Methods</td><td>JS</td><td>HD</td><td>BD</td><td>WD</td><td>En-dist</td><td>C-dist</td><td>wl-sim</td><td>lt-sim</td></tr><tr><td rowspan="6">10</td><td>Random-VAE</td><td>0.42</td><td>0.98</td><td>Inf</td><td>7.58</td><td>0.3787</td><td>0.4528</td><td>0.3333</td><td>0.2494</td></tr><tr><td>GraphRNN</td><td>0.47</td><td>0.98</td><td>Inf</td><td>1.64</td><td>0.7226</td><td>0.5319</td><td>0.2470</td><td>0.0055</td></tr><tr><td>GraphVAE</td><td>0.67</td><td>1.00</td><td>Inf</td><td>2.85</td><td>0.6849</td><td>0.6664</td><td>0.3723</td><td>0.1576</td></tr><tr><td>GraphGMG</td><td>0.43</td><td>0.98</td><td>Inf</td><td>1.69</td><td>0.6849</td><td>0.4763</td><td>0.3701</td><td>0.0120</td></tr><tr><td>S-Generator</td><td>0.35</td><td>0.98</td><td>3.45</td><td>0.80</td><td>0.2097</td><td>0.2465</td><td>0.4185</td><td>0.5431</td></tr><tr><td>GT-GAN</td><td>0.35</td><td>0.98</td><td>3.44</td><td>0.77</td><td>0.2034</td><td>0.2379</td><td>0.4195</td><td>0.5469</td></tr><tr><td rowspan="4">20</td><td>RandomVAE</td><td>0.51</td><td>0.97</td><td>Inf</td><td>1.74</td><td>0.4513</td><td>0.5400</td><td>0.3333</td><td>0.3813</td></tr><tr><td>GraphRNN</td><td>0.50</td><td>0.98</td><td>Inf</td><td>1.44</td><td>0.7222</td><td>0.6087</td><td>0.2652</td><td>0.2373</td></tr><tr><td>S-Generator</td><td>0.36</td><td>0.96</td><td>2.84</td><td>0.67</td><td>0.1367</td><td>0.1903</td><td>0.4665</td><td>0.7017</td></tr><tr><td>GT-GAN</td><td>0.35</td><td>0.96</td><td>2.74</td><td>0.66</td><td>0.1367</td><td>0.1894</td><td>0.4681</td><td>0.7018</td></tr><tr><td rowspan="3">100</td><td>GraphRNN</td><td>0.48</td><td>0.88</td><td>Inf</td><td>0.90</td><td>0.7147</td><td>0.6519</td><td>0.2713</td><td>0.2138</td></tr><tr><td>S-Generator</td><td>0.14</td><td>0.68</td><td>0.64</td><td>0.30</td><td>0.1149</td><td>0.1501</td><td>0.3522</td><td>0.8891</td></tr><tr><td>GT-GAN</td><td>0.15</td><td>0.43</td><td>0.24</td><td>0.31</td><td>0.1153</td><td>0.2087</td><td>0.4078</td><td>0.9217</td></tr><tr><td rowspan="3">150</td><td>GraphRNN</td><td>0.42</td><td>0.74</td><td>Inf</td><td>0.95</td><td>0.7494</td><td>0.6266</td><td>0.2891</td><td>0.1874</td></tr><tr><td>S-Generator</td><td>0.08</td><td>0.31</td><td>0.11</td><td>0.29</td><td>0.0949</td><td>0.1101</td><td>0.3493</td><td>0.8493</td></tr><tr><td>GT-GAN</td><td>0.07</td><td>0.30</td><td>0.11</td><td>0.27</td><td>0.0931</td><td>0.2105</td><td>0.3926</td><td>0.8714</td></tr></table>

# 4.3 EVALUATION RESULTS ON SYNTHETIC DATASETS

Results for the synthetic datasets. To evaluate the similarity between the generated and real target graphs for scale-free dataset, we selected eight performance metrics: 1) two metrics are distances between generated and real graph in terms of Eigenvector centrality (En-dist) (Bonacich, 1987) and Closeness centrality (C-dist) (Freeman, 1978), where the lower the distance, the better the performance; 2) two metrics are similarity score based on the graph kernels of Weisfeiler Lehman kernel(wl-sim) (Shervashidze et al., 2011) and Lovasz Theta Kernel(lt-sim) (Johansson & et al, 2014), where the higher the score, the better the performance; 3) four metrics are used to evaluate the the node degree distribution correlation between the generated and real target graphs by: Jensen-Shannon distances (JS), the Hellinger Distance (HD), the Bhattacharyya Distance (BD) and the Wasserstein Distances (WD), where the lower the score, the better the performance.

As shown in table 1, our GT-GAN consistently outperforms all other baselines by a large margin, especially when the graph size becomes large (i.e. having the superiority of  $34.6\%$  than other methods when size is 150). The "Inf" entries represent distance over 1000. S-Generator is generally the second best methods in terms of these four evaluation metrics, highlighting the effectiveness of our proposed graph encoder and decoder.

To verify whether GT-GAN can indeed discover the underlying ground-truth translation rules between input-target pairs, we draw the node degree distribution curve for three pairs of generated and real target graphs by GT-GAN, as shown in Fig. 3. The curves of the generated graphs closely follow the power-law rule and become even closer to the real graphs as the graph size increases, which is consistent with the findings in Table 1. This demonstrates that our GT-GAN model successfully learns the inherent properties of scale-free graphs during translation. Similar observations for the evaluation metrics (e.g. average degree, repository and density) of the Poisson random datasets and remaining scale-free subsets can be found in Appendixes B and C.

![](images/eb4f3ac1375c19699bbfb4683200d43aaf570ecf6df4ab5e24982b1ba8206935.jpg)  
Figure 3: Examples of node degree distributions of generated and target graphs for scale-free graphs

![](images/97f8113027ae37d1c2bfa4039c2ba5baaaff7573742c34aae0e91af9844283d9.jpg)

![](images/78c2f08f1984e708fd721101f86b22fcadac04af2de74d0251ae93beae61d978.jpg)

# 4.4 EVALUATION RESULTS ON REAL APPLICATION DATASETS

Results for the user authentication datasets. For the real world dataset, we design an indirect evaluation metric inspired from a real-world classification problem: label imbalance issues. For

example, we may want to build a classifier to determine whether an authentication graph of a user is malicious (positive) or normal (negative), but this user has few malicious records. For this difficult task, the graphs (i.e., malicious graphs) generated by GT-GAN, which has been trained on other users' records, can be utilized as positive samples to train the classifier. Specifically, when evaluating, the test set is further split evenly into two subsets. The first subset is used to train a graph classifier, as proposed by Nikolentzos et al. (2017), using only the normal graphs plus the generated malicious graphs. The second subset, which contains both the normal and real malicious graphs can then be used to validate the trained classifier. In addition, a "gold standard" classifier trained on both normal and real malicious graphs acts as the "best-possible-performer" and is used to evaluate all the different generative models to judge how "real" the graphs they generate are. We refer readers to the detailed evaluations in Appendix E.

Table 2: User authentication datasets  

<table><tr><td>Size</td><td>Method</td><td>P</td><td>R</td><td>AUC</td><td>F1</td></tr><tr><td rowspan="5">50</td><td>RandomVAE</td><td>0.32</td><td>0.51</td><td>0.26</td><td>0.39</td></tr><tr><td>GraphRNN</td><td>0.34</td><td>0.36</td><td>0.50</td><td>0.36</td></tr><tr><td>S-Generator</td><td>0.72</td><td>0.61</td><td>0.74</td><td>0.66</td></tr><tr><td>GT-GAN</td><td>0.79</td><td>0.68</td><td>0.78</td><td>0.73</td></tr><tr><td>Gold Standard</td><td>0.97</td><td>0.97</td><td>0.97</td><td>0.97</td></tr><tr><td rowspan="3">300</td><td>S-Generator</td><td>0.77</td><td>0.58</td><td>0.62</td><td>0.66</td></tr><tr><td>GT-GAN</td><td>0.84</td><td>0.66</td><td>0.79</td><td>0.74</td></tr><tr><td>Gold Standard</td><td>0.98</td><td>0.96</td><td>0.97</td><td>0.97</td></tr></table>

Table 3: IOT datasets  

<table><tr><td>Size</td><td>Method</td><td>R2</td><td>MSE</td><td>P</td><td>ACC</td></tr><tr><td rowspan="3">20</td><td>GraphRNN</td><td>0.16</td><td>1775.58</td><td>0.23</td><td>83.97%</td></tr><tr><td>GraphVAE</td><td>0.39</td><td>2109.64</td><td>0.32</td><td>81.19%</td></tr><tr><td>GT-GAN</td><td>0.67</td><td>370.91</td><td>0.85</td><td>92.00%</td></tr><tr><td rowspan="3">40</td><td>GraphRNN</td><td>0.44</td><td>1950.46</td><td>0.29</td><td>70.54%</td></tr><tr><td>GraphVAE</td><td>0.73</td><td>2410.57</td><td>0.16</td><td>66.60%</td></tr><tr><td>GT-GAN</td><td>0.69</td><td>408.50</td><td>0.86</td><td>93.94%</td></tr><tr><td rowspan="3">60</td><td>GraphRNN</td><td>0.52</td><td>1831.43</td><td>0.04</td><td>61.07%</td></tr><tr><td>GraphVAE</td><td>0.00</td><td>2453.61</td><td>0.04</td><td>50.64%</td></tr><tr><td>GT-GAN</td><td>0.62</td><td>566.88</td><td>0.80</td><td>94.63%</td></tr></table>

As shown in Table 2, classifiers trained by the graphs generated by GT-GAN can classify normal and hacked behaviors effectively with AUC above 0.78, which is well above the 0.5 obtained using a random model. GT-GAN significantly outperforms other methods by around  $25\%$ ,  $16\%$ ,  $24.5\%$  and  $22.1\%$ , respectively, on the four metrics: precision (P), recall (R), AUC and F1-score for the trained classifier. GT-GAN performs consistently better than other methods when the graph size rises from 50 to 300. In addition, GT-GAN clearly outperformed the S-Genertor in this evaluation setting. This confirms that using a translator alone to learn a deterministic output given an input graph is not sufficient to capture the generic distribution of the target graphs. In addition, the four direct evaluation mentioned above are also tested and the results can be found in Appendix C.

Results on IOT dataset. Table 3 compared the performance of GT-GAN and other comparison methods for the IOT dataset by examining the edges of the generated and real target graphs for four metrics: MSE (mean squared error), R2 (coefficient of determination score), Pearson Correlation (P) of adjacent matrix, and ACC (Accuracy) for the correct existence of edges among all the pairs of nodes. The results show that GT-GAN performed almost the best for all the three subsets. GT-GAN got highest Pearson Correlation of around 0.8 for all three subsets compared to the other methods which had Pearson Correlations below 0.4. Due to the L1-loss required to maintain the topology pattern similarity, GT-GAN also outperformed the comparison methods with around  $8\%$ ,  $26\%$  and  $40\%$  superiority in ACC for the three subsets, respectively, and had the smallest MSE, at just one tenth of those achieved by comparison methods.

# 4.5 ABLATION STUDY ON THE GRAPH ENCODERS AND DECODER

To further validate the superiority of the proposed graph convolution and deconvolution layers, an ablation experiment was conducted by replacing the encoder and decoder with node embedding and decoder methods normally used. The graph encoder was replaced by the GCN (Kipf & Welling, 2017), DCNN (Atwood & Towsley, 2016) and Graph U-NET (Gao & Ji, 2019), both of which consider edge and node features for graph embedding. The graph decoder was replaced by the decoder in VGAE (Kipf & Welling, 2016). There were thus three method combinations for comparison.

Table. 4 shows the results of the ablation study of the proposed encoder and decoder on part of the scale-free (Scale), user authentication (Auth) and IOT datasets. There are two major findings here. First, the encoder of GT-GAN outperformed both the GCN- and DCNN- based encoders by a large margin on these datasets, especially for the real-world datasets, where the edges of the graphs can have a very complex meaning. For example, on Auth-I, GT-GAN performed  $43\%$ ,  $50\%$ ,  $31\%$ , and  $38\%$  better on average, when compared with the GCN and DCNN encoders in terms of precision, recall, AUC and F1-scores, respectively. Second, the proposed decoder in GT-GAN was deemed both effective and irreplaceable for graph generation. For example, on IOT-III, GT-GAN performed  $6.97\%$ ,

Table 4: Ablation study on four datasets  

<table><tr><td>Dataset</td><td>Method</td><td>JS</td><td>HD</td><td>BD</td><td>WD</td><td>En-dist</td><td>C-dist</td><td>wl-sim</td></tr><tr><td rowspan="5">Scale-III</td><td>GCN+decoder</td><td>0.18</td><td>0.48</td><td>0.27</td><td>18.84</td><td>0.6903</td><td>0.6751</td><td>0.4031</td></tr><tr><td>DCNN+decoder</td><td>0.65</td><td>0.96</td><td>Inf</td><td>0.77</td><td>0.6907</td><td>0.6745</td><td>0.4032</td></tr><tr><td>Graph-U+decoder</td><td>0.69</td><td>0.99</td><td>Inf</td><td>5.77</td><td>0.6931</td><td>0.6496</td><td>0.4040</td></tr><tr><td>Encoder+VGAE</td><td>0.31</td><td>0.63</td><td>0.51</td><td>43.78</td><td>0.0922</td><td>0.2559</td><td>0.4003</td></tr><tr><td>GT-GAN</td><td>0.15</td><td>0.43</td><td>0.24</td><td>0.31</td><td>0.1153</td><td>0.2087</td><td>0.4078</td></tr><tr><td></td><td></td><td>P</td><td>R</td><td>AUC</td><td>F1</td><td>En-dist</td><td>C-dist</td><td>wl-sim</td></tr><tr><td rowspan="5">Auth-I</td><td>GCN+decoder</td><td>0.31</td><td>0.35</td><td>0.52</td><td>0.33</td><td>0.7394</td><td>0.7494</td><td>0.6632</td></tr><tr><td>DCNN+decoder</td><td>0.59</td><td>0.55</td><td>0.55</td><td>0.57</td><td>0.0186</td><td>0.3349</td><td>0.6851</td></tr><tr><td>Graph-U+decoder</td><td>0.41</td><td>0.60</td><td>0.30</td><td>0.49</td><td>0.6789</td><td>0.6859</td><td>0.9239</td></tr><tr><td>Encoder+VGAE</td><td>0.49</td><td>0.46</td><td>0.61</td><td>0.47</td><td>0.0231</td><td>0.3129</td><td>0.6111</td></tr><tr><td>GT-GAN</td><td>0.79</td><td>0.68</td><td>0.78</td><td>0.73</td><td>0.0134</td><td>0.1924</td><td>0.9439</td></tr><tr><td rowspan="3">Auth-II</td><td>DCNN+decoder</td><td>0.58</td><td>0.42</td><td>0.62</td><td>0.51</td><td>0.0007</td><td>0.1896</td><td>0.7033</td></tr><tr><td>Graph-U+decoder</td><td>0.42</td><td>0.44</td><td>0.23</td><td>0.32</td><td>0.6931</td><td>0.6842</td><td>0.9744</td></tr><tr><td>GT-GAN</td><td>0.84</td><td>0.66</td><td>0.79</td><td>0.74</td><td>0.0054</td><td>0.0681</td><td>0.9864</td></tr><tr><td></td><td></td><td>R2</td><td>MSE</td><td>P</td><td>ACC</td><td>En-dist</td><td>C-dist</td><td>wl-sim</td></tr><tr><td rowspan="5">IOT-III</td><td>GCN+decoder</td><td>0.46</td><td>818.25</td><td>0.71</td><td>92.69</td><td>0.4990</td><td>0.4349</td><td>0.3304</td></tr><tr><td>DCNN+decoder</td><td>0.52</td><td>721.98</td><td>0.74</td><td>93.26</td><td>0.3596</td><td>0.3217</td><td>0.3292</td></tr><tr><td>Graph-U+decoder</td><td>0.45</td><td>826.63</td><td>0.70</td><td>92.46</td><td>0.3526</td><td>0.2771</td><td>0.3310</td></tr><tr><td>Encoder+VGAE</td><td>0.12</td><td>1337.16</td><td>0.44</td><td>88.14</td><td>0.4811</td><td>0.4876</td><td>0.3333</td></tr><tr><td>GT-GAN</td><td>0.62</td><td>566.88</td><td>0.80</td><td>94.63</td><td>0.3350</td><td>0.3051</td><td>0.3899</td></tr></table>

45.00%, and 83.33% better than the decoder in VGAE in terms of ACC, P and R2, respectively, as well as a low MSE below 1000.

# 4.6 MODEL SCALABILITY ANALYSIS

We compare the scalability of GT-GAN against three graph generation methods as shown in Fig.4. Our GT-GAN model significantly outperforms other state-of-the-art baselines in terms of both computational time and memory consumption. As the graph size increases up to 50, both computational time and the memory consumption of the GT-GAN remains almost constant. In contrast, the runtime and memory consumption of RandomVAE and the runtime of GraphVAE increase super-linearly as the graph size increases, making it hard to scale even to a graph size of 50. Interestingly, the runtime and memory consumption of GraphRNN also increases only slightly as the graph size increases. However, our GT-GAN model achieves around ten times speedups while requiring almost half of memory, compared to GraphRNN, highlighting the strong linear complexity of GT-GAN in practice.

![](images/1e79daeaa41f3c7e08baf664719d6d47745372af9c9e762204b41fbba0994ad4.jpg)  
(a) Time Cost

![](images/64319d032be2658620a3f25ad9c1adf693961d9968305dea0ecab228140afdad.jpg)  
(b) Memory Cost  
Figure 4: Scalability plots for memory and time cost of GT-GAN, RandomVAE, GraphVAE and GraphRNN

# 5 CONCLUSIONS

This paper focuses on a new problem: deep graph translation. To achieve this, we propose a novel GT-GAN which translates an input graph to a target graph. To learn both global and local mapping between graphs, a new graph encoder-decoder model have been proposed while preserving the graph patterns in various scales. Extensive experiments have been conducted on the synthetic and real-world dataset to compare with the state-of-the-art graph generation models. Experimental results show that our GT-GAN can discover the ground-truth translation rules, and significantly outperform other

basielines in terms of both effectiveness and scalability. This paper opens a thread of research for deep graph translation in many practical applications.

# REFERENCES

James Atwood and Don Towsley. Diffusion-convolitional neural networks. In Advances in Neural Information Processing Systems, pp. 1993-2001, 2016.  
James Atwood, Siddharth Pal, Don Towsley, and Ananthram Swami. Sparse diffusion-convolutional neural networks. In NIPS, 2016.  
Yunsheng Bai, Hao Ding, Yang Qiao, Agustin Marinovic, Ken Gu, Ting Chen, Yizhou Sun, and Wei Wang. Unsupervised inductive graph-level representation learning via graph-graph proximity. International Joint Conferences on Artificial Intelligence, 2019.  
Joost Bastings, Ivan Titov, Wilker Aziz, Diego Marcheggiani, and Khalil Sima'an. Graph convolutional encoders for syntax-aware neural machine translation. arXiv preprint arXiv:1704.04675, 2017.  
Daniel Beck, Gholamreza Haffari, and Trevor Cohn. Graph-to-sequence learning using gated graph neural networks. arXiv preprint arXiv:1806.09835, 2018.  
Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, and Stephan Gunnemann. Netgan: Generating graphs via random walks. arXiv preprint arXiv:1803.00816, 2018.  
Béla Bollobás, Christian Borgs, Jennifer Chayes, and Oliver Riordan. Directed scale-free graphs. In Proceedings of the fourteenth annual ACM-SIAM symposium on Discrete algorithms, pp. 132-139. Society for Industrial and Applied Mathematics, 2003.  
Phillip Bonacich. Power and centrality: A family of measures. American journal of sociology, 92(5): 1170-1182, 1987.  
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann Lecun. Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations (ICLR2014), CBLS, April 2014, 2014.  
Yu Chen, Lingfei Wu, and Mohammed J Zaki. Reinforcement learning based graph-to-sequence model for natural question generation. arXiv preprint arXiv:1908.04942, 2019.  
Hanjun Dai, Yingtao Tian, Bo Dai, Steven Skiena, and Le Song. Syntax-directed variational autoencoder for structured data. arXiv preprint arXiv:1802.08786, 2018.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems, pp. 3844-3852, 2016.  
Kien Do, Truyen Tran, and Svetha Venkatesh. Graph transformation policy network for chemical reaction prediction. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 750-760. ACM, 2019.  
Linton C Freeman. Centrality in social networks conceptual clarification. Social networks, 1(3): 215-239, 1978.  
Hongyang Gao and Shuiwang Ji. Graph u-nets. arXiv preprint arXiv:1905.05178, 2019.  
Yuyang Gao, Lingfei Wu, Houman Homayoun, and Liang Zhao. Dyngraph2seq: Dynamic-graph-to-sequence interpretable learning for health stage prediction in online health forums. arXiv preprint arXiv:1908.08497, 2019.  
Daniel Gildea, Giorgio Satta, and Xiaochang Peng. Cache transition systems for graph parsing. Computational Linguistics, 44(1):85-118, 2018.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212, 2017.

Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Neural Networks, 2005. IJCNN'05. Proceedings. 2005 IEEE International Joint Conference on, volume 2, pp. 729-734. IEEE, 2005.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pp. 1025-1035, 2017.  
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1125-1134, 2017.  
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. arXiv preprint arXiv:1802.04364, 2018a.  
Wengong Jin, Kevin Yang, Regina Barzilay, and Tommi Jaakkola. Learning multimodal graph-to-graph translation for molecular optimization. arXiv preprint arXiv:1812.01070, 2018b.  
Fredrik Johansson and et al. Global graph kernels using geometric embeddings. In ICML 2014, 2014.  
Thomas N Kipf and Max Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308, 2016.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.  
Paul L Krapivsky and Sidney Redner. Organization of growing random networks. Physical Review E, 63(6):066123, 2001.  
Matt J Kusner, Brooks Paige, and José Miguel Hernández-Lobato. Grammar variational autoencoder. In International Conference on Machine Learning, pp. 1945-1954, 2017.  
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. International Conference on Learning Representations, 2016.  
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, and Peter Battaglia. Learning deep generative models of graphs. arXiv preprint arXiv:1803.03324, 2018.  
Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. Learning convolutional neural networks for graphs. In ICML, pp. 2014-2023, 2016.  
Giannis Nikolentzos, Polykarpos Meladianos, Antoine Jean-Pierre Tixier, Konstantinos Skianis, and Michalis Vazirgiannis. Kernel graph convolutional neural networks. arXiv preprint arXiv:1710.10689, 2017.  
Xiaochang Peng, Daniel Gildea, and Giorgio Satta. Amr parsing with cache transition systems. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Bidisha Samanta, Abir De, Niloy Ganguly, and Manuel Gomez-Rodriguez. Designing random graph models using variational autoencoders with applications to chemical design. arXiv preprint arXiv:1802.05283, 2018.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61-80, 2009.  
Michael L Seltzer, Dong Yu, and Yongqiang Wang. An investigation of deep neural networks for noise robust speech recognition. In Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on, pp. 7398-7402. IEEE, 2013.  
Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(Sep): 2539-2561, 2011.  
Martin Simonovsky and Nikos Komodakis. Graphvae: Towards generation of small graphs using variational autoencoders. arXiv preprint arXiv:1802.03480, 2018.

Linfeng Song, Yue Zhang, Zhiguo Wang, and Daniel Gildea. A graph-to-sequence model for amr-to-text generation. arXiv preprint arXiv:1805.02473, 2018.  
Mingming Sun and Ping Li. Graph to graph: a topology aware approach for graph structures learning and generation. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2946-2955, 2019.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
Petar Velicković, William Fedus, William L Hamilton, Pietro Lio, Yoshua Bengio, and R Devon Hjelm. Deep graph infomax. arXiv preprint arXiv:1809.10341, 2018.  
Yuxuan Wang, Wanxiang Che, Jiang Guo, and Ting Liu. A neural transition-based approach for semantic dependency graph parsing. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Bo Wu, Yang Liu, Bo Lang, and Lei Huang. Dgcnn: Disordered graph convolutional neural network based on the gaussian mixture model. arXiv:1712.03563, 2017.  
Kun Xu, Lingfei Wu, Zhiguo Wang, and Vadim Sheinin. Graph2seq: Graph to sequence learning with attention-based neural networks. arXiv preprint arXiv:1804.00823, 2018a.  
Kun Xu, Lingfei Wu, Zhiguo Wang, Mo Yu, Liwei Chen, and Vadim Sheinin. Exploiting rich syntactic information for semantic parsing with graph-to-sequence model. arXiv preprint arXiv:1808.07624, 2018b.  
Jiaxuan You, Rex Ying, Xiang Ren, William Hamilton, and Jure Leskovec. Graphnn: Generating realistic graphs with deep auto-regressive models. In International Conference on Machine Learning, pp. 5694-5703, 2018.
