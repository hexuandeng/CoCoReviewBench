# KOLMOGOROV-ARNOLD GRAPH NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Graph neural networks (GNNs) excel in learning from network-like data but often lack interpretability, making their application challenging in domains requiring transparent decision-making. We propose the Kolmogorov-Arnold Network for Graphs (KANG), a novel GNN model leveraging spline-based activation functions on edges to enhance both accuracy and interpretability. Our experiments on five benchmark datasets demonstrate that KANG outperforms state-of-the-art GNN models in node classification, link prediction, and graph classification tasks. In addition to the improved accuracy, KANG's design inherently provides insights into the model's decision-making process, eliminating the need for post-hoc explainability techniques. This paper discusses the methodology, performance, and interpretability of KANG, highlighting its potential for applications in domains where interpretability is crucial.

# 1 INTRODUCTION

Neural networks and deep learning have clearly revolutionized countless fields, encompassing the domains of image, text, audio, and medical data. The basic building block of neural network models is the multilayer perceptron (MLP) (Haykin, 1998). Very recently, a promising alternative to MLPs was proposed, namely the Kolmogorov-Arnold Network (KAN) (Liu et al., 2024). MLPs are proved to be universal function approximators (Hornik et al., 1989), whereas KANs are inspired by the Kolmogorov-Arnold representation theorem (Kolmogorov, 1956; 1957), which states that a smooth multivariate function can be written as the finite sum of the composition of univariate functions; this has a similar structure to a two-layer neural network.

The idea of Liu et al. (2024) is to generalize this two-level structure to more levels. This extension allows the definition of a novel framework that proved increased accuracy and interpretability with respect to standard MLPs. Indeed, lack of interpretability in neural networks is a major issue, which limits its usage in domains in which interpretable results are crucial (e.g., medical and financial scenarios). To cope with this flaw, research focuses on the development of explainability techniques that try to unveil what was learned by those models in terms of salient features (Burbart & Huber, 2021; Saleem et al., 2022). In the domain of neural networks, models specialized in learning from graph-structure data have emerged. Graph neural networks (GNNs) (Scarselli et al., 2008) are able to leverage the connectivity structure of network-like data, learning representative topology-aware embeddings. In literature, we find different architectures addressing different aspects and characteristics of graph data. Graph convolutional networks (GCNs) (Kipf & Welling, 2017) extend the concept of convolution to graph structures, aggregating information from each node's neighbors using spectral graph convolutions. GraphSAGE (Hamilton et al., 2017) is another GNN model that generates node embeddings by sampling and aggregating information from neighbor nodes. It is based on an inductive learning procedure that allows embedding generation also for previously unseen nodes. Graph attention networks (GATs) (Velickovic et al., 2018) incorporate attention mechanisms into GNNs, allowing the model to focus on the most salient parts of a node's neighborhood. An additional GNN model is offered by GINs (graph isomorphism networks) (Xu et al., 2019). Such models are designed to have an effective discriminative ability, theoretically proven to be as powerful as the Weisfeiler-Lehman graph isomorphism test (Weisfeiler & Leman, 1968). GNNs have also been extended to relational structures, with the introduction of relational graph convolutional networks (RGCNs). They are designed to handle graphs with multiple types of edges and nodes by incorporating relations explicitly into the model. Furthermore, variational graph autoencoders (VGAE) (Kipf & Welling, 2016) aim to encode graph data into a latent space and then reconstruct the graph from the generated embeddings.

Similarly to classic multilayer perceptron models, GNNs are not immune to the interpretability curse; to shed light on their predictions, several explainability methodologies have been developed. Such methods explain GNN predictions in terms of importance subgraphs built of salient nodes, edges, node features, connected subgraphs, or a combination of those elements. The pioneering work in eXplainable Artificial Intelligence (XAI) for GNNs is GNNExplainer (Ying et al., 2019), which determined explanations by generating important subgraphs using a mask on the adjacency matrix able to maximize the mutual information between the prediction and the distribution of the possible explanation subgraphs. GraphSVX (Duval & Malliaros, 2021) in another method, which determines explanation in terms of important nodes and node features relying on the theoretical background behind Shapley values (Shapley, 1953). It uses a decomposition technique relying on a surrogate linear model for approximating Shapley values. A methodology targeting edges as a means for determining explanation subgraphs is EdgeSHAPer (Mastropietro et al., 2022). It employs Monte Carlo sampling to approximate Shapley values determining salient edges forming relevant subgraphs driving predictions. One additional XAI tool is SubgraphX (Yuan et al., 2021). Also exploiting Shapley value approximation, it looks for explanations only in terms of connected subgraphs by using a Monte Carlo Tree Search approach.

In this work, we extend the Kolmogorov-Arnold representation theorem to GNNs, introducing the Kolmogorov-Arnold Network for Graphs (KANG). Our main contributions are as follows:

- Novel GNN Architecture: We propose KANG, a novel GNN model that employs spline-based activation functions on graph edges. This design enhances the model's flexibility and interpretability while retaining the efficiency of message-passing mechanisms in GNNs.  
- Enhanced Interpretability: KANG provides inherent interpretability by design, eliminating the need for external explainability techniques. This feature is crucial for applications in domains requiring transparent decision-making processes.  
- Performance Improvement: We demonstrate that KANG outperforms state-of-the-art GNN models in node classification, link prediction, and graph classification tasks on benchmark datasets (Cora, PubMed, CiteSeer, MUTAG, and PROTEINS).

Recently, the integration of KANs with graph-structured data has attracted growing interest from researchers (Kiamari et al., 2024; Bresson et al., 2024; Zhang & Zhang, 2024). Specifically, Kiamari et al. (2024) proposed two KAN-based architectures: one where node embeddings are aggregated before applying the learnable spline-based KAN layers, and another where the KAN layers are applied prior to aggregation. They compared their models to GCNs using a reduced subset of the Cora dataset features (200 out of 1433). Bresson et al. (2024) introduced two GNN variants utilizing KAN layers for node representation updates: KAGIN (based on GIN) and KAGCN (based on GCN). Additionally, Ahmed & Sifat (2024) applied a similar architecture to molecular data for protein-ligand affinity prediction.

In the following sections, we detail the KANG architecture and its components, present experimental results to validate our approach, and discuss the interpretability of KANG. Our findings suggest that KANG outperforms existing GNNs while providing interpretable outcomes.

# 2 METHODOLOGY

# 2.1 KOLMOGOROV-ARNOLD NETWORKS

This section details the construction and operation of our proposed KANG model. We begin by revisiting the key elements of Kolmogorov-Arnold Networks (KANs), upon which KANG is built. Kolmogorov-Arnold theorem states that a multivariate continuous function in a bounded domain can be rewritten using a finite composition of continuous functions on one single variable and the addition operation. Given  $\mathbf{x}$  a vector of dimension  $n$ ,  $f$  a function such that  $f:[0,1]^n\to \mathbb{R}$ , it is thus possible to write

$$
f (\mathbf {x}) = \sum_ {q = 1} ^ {2 n + 1} \Phi_ {q} \left(\sum_ {p = 1} ^ {n} \phi_ {q, p} \left(x _ {p}\right)\right), \tag {1}
$$

where  $\phi_{q,p}:[0,1]\to \mathbb{R}$  and  $\Phi_q:\mathbb{R}\rightarrow \mathbb{R}$ . Liu et al. (2024) extended Equation 1, representing a two-layer KAN with  $2n + 1$  terms in the hidden layer, to larger depths and widths, parametrizing each one-dimensional function as a B-spline curve. This kind of neural network has an activation function on the edges instead of nodes; the latter simply performs a summation. From the implementation side, KANs activation functions  $\phi (x)$  are built as a sum of a basis function  $b(x)$  and the spline function such that

$$
\phi (x) = w _ {b} (x) + w _ {s} \operatorname {s p l i n e} (x). \tag {2}
$$

The original KAN model uses

$$
b (x) = \operatorname {s i l u} (x) = \frac {x}{\left(1 + e ^ {- x}\right)}, \operatorname {s p l i n e} (x) = \sum_ {i} c _ {i} B _ {i} (x). \tag {3}
$$

At initialization, all activation functions are such that  $w_{s} = 1$  and  $\mathrm{spline}(x)\approx 0$ , and  $w_{b}$  are initialized using Xavier initialization.

# 2.2 KOLMOGOROV-ARNOLD GRAPH NEURAL NETWORK

Building upon the strengths of KANs, we introduce the Kolmogorov-Arnold Network for Graphs (KANG), a novel GNN architecture (Figure 1) designed for processing graph-structured data. KANG leverages the flexibility, accuracy, and interpretability of KANs while retaining the efficient message passing mechanisms of GNNs.

![](images/595e559046642e62903f5e1c6d544c8b494882f3aa3c5ed4501b1a01dea75230.jpg)  
Figure 1: A simplified graphical representation of the KANG model is shown.  $\mathbf{X}$  and  $\mathbf{A}$  represent the feature matrix of the nodes and the adjacency matrix, respectively. The hidden layers consist of KANG convolutional layers, where messages are propagated and then aggregated. The output layer is a KAN linear layer. Each neuron has its own set of learnable splines. Although the figure provides a simplified view of the splines, the actual learned splines, responsible for transforming input values, can be visualized, as explained also in the original KAN paper (Appendix A.3). This allows for a clearer interpretation of the nonlinear transformations that contribute to the model's final predictions.

# 2.2.1 KANG ARCHITECTURE

KANG employs learnable spline-based activation functions on the edges of the graph, allowing for flexible nonlinear transformations of node features based on their connections. The architecture is composed of multiple layers:

- KAN-based convolutional layer: Each layer efficiently handles the propagation and aggregation of messages between nodes by applying a KAN-based transformation to the features of each node, taking into account information from its neighbors.  
- KAN-based linear layer: A final linear layer performs a linear transformation on the aggregated node features, producing the final node representations.

The Xavier uniform initialization, also known as Glorot initialization, is a widely used method for initializing the weights of neural networks. This technique aims to maintain the variance of the

gradients approximately the same across all layers, thereby mitigating the vanishing and exploding gradient problems. The weights are initialized by sampling from a uniform distribution in the range  $[-r,r]$ , with  $r = \sqrt{\frac{6}{n_{in} + n_{out}}}$ , where  $n_{in}$  and  $n_{out}$  denote the number of input and output units in the weight tensor, respectively (Glorot & Bengio, 2010). As suggested in the KAN paper, also in KANG the basis function weights are initialized using the Xavier initialization, facilitating effective training and optimization.

# 2.2.2 MATHEMATICAL FORMULATION OF KANG

In the next part of this section we will go through the basic mathematical formulation of the constituent steps of KANG.

Message Passing: Each node  $i$  in the graph has an initial feature vector  $\mathbf{x}_i$ . For each layer  $l$  in the KANG, the node representations are updated through message passing and aggregation. The message from a node  $j$  to its neighbors at layer  $l$  is denoted as  $\mathbf{m}_j^{(l)}$ .

$$
\mathbf {m} _ {j} ^ {(l)} = \left[ \mathrm {s p l i n e} _ {1} ^ {(l)} (x _ {j, 1} ^ {(l - 1)}) \dots \mathrm {s p l i n e} _ {H} ^ {(l)} (x _ {j, H} ^ {(l - 1)}) \right]
$$

Spline-Based Activation Function: The spline-based activation function  $\varphi^{(l)}$  at layer  $l$  used in KANG is defined as

$$
\varphi^ {(l)} (\mathbf {x}) = w _ {b} ^ {(l)} b (\mathbf {x}) + w _ {s} ^ {(l)} \mathrm {S P L I N E} ^ {(l)} (\mathbf {x})
$$

where  $b(\cdot)$  is a basis function (e.g., SiLU), and  $\mathrm{SPLINE}^{(l)}(\mathbf{x})$  applies  $\mathrm{spline}_h^{(l)}(\cdot)$  to each element  $h$  of vector  $\mathbf{x}$ .

KANG Convolutional Layer: Each KANG layer  $l$  combines these steps, resulting in the following layer-wise update rule for node  $i$ :

$$
\mathbf{x}_{i}^{(l)} = \operatorname{AGGR}_{j\in \mathcal{N}(i)}\varphi^{(l - 1)}(\mathbf{x}_{j}^{(l - 1)}),
$$

where AGGR is aggregation function (we consider average, sum, max) which combines the messages that node  $i$  receives from its neighbors<sup>2</sup>

Output Layer: After passing through multiple KANG convolutional layers, the final node representations are obtained using a KAN-based linear layer:

$$
\mathbf {z} _ {i} = \mathrm {K A N L i n e a r} (\mathbf {x} _ {i} ^ {(L)})
$$

where  $L$  is the number of layers and KANlinear is a KAN layer as defined by Liu et al. (2024), which applies a final spline-based transformation.

# 2.2.3 OVERALL MODEL

The overall model can be summarized as:

1. Initialize spline weights with Xavier uniform initialization.  
2. For each KANG layer  $l = 1, \dots, L$ :

(a) Compute messages  $\mathbf{m}_j^{(l)}$  for each node.  
(b) Aggregate messages  $\mathbf{a}_i^{(l)}$  for each neighboring node.  
(c) Update node representation  $\mathbf{x}_i^{(l)}$  aggregating the spline-activated messages.

3. Apply a KAN-based linear layer to obtain the final node representation  $\mathbf{z}_i$ .

# 2.3 INTERPRETABILITY OF KANG

KANG allows for understanding its predictions without relying on external explainers, which may need to be trained (e.g., GNNExplainer) or whose computations are expensive (e.g., Shapley value-based explainers). KANG's interpretability involves determining the influence of input features and edge importance by analyzing the information flow across the graph.

# 2.3.1 FEATURE INFLUENCE

Motivated by previous studies (Baehrens et al., 2010; Simonyan et al., 2014; Hechtlinger, 2016; Sundararajan et al., 2017), to compute the inculence of features in KANG, we leverage the interaction between the gradients and the spline weights, capturing both feature sensitivity and nonlinear transformations. Specifically, let  $\mathbf{x} \in \mathbb{R}^{N \times F}$  represent the input features for  $N$  nodes, each with  $F$  features. Given a hidden layer, the first step involves calculating the gradient of the output prediction with respect to each input feature, which gives us a matrix  $\mathbf{G} \in \mathbb{R}^{N \times F}$ , where each element  $g_{i,f}$  reflects the sensitivity of the output for node  $i$  to its corresponding feature  $f$ .

However, gradients alone do not provide a complete picture of how the features are processed by KANG, and can fail to determine a correct measure of feature importance (Sundararajan et al., 2017; Shrikumar et al., 2017). In our model, the spline function weights  $\mathbf{S}_{\mathrm{spline}} \in \mathbb{R}^{H \times F \times B}$ , where  $H$  is the number of hidden units at the current hidden layer,  $F$  is the input feature size, and  $B$  is the number of spline basis coefficients, modulate the feature transformation along the edges in a nonlinear fashion. These splines are critical because they allow KANG to adaptively adjust how features are processed based on the local graph structure, capturing important nonlinear interactions. We aggregate the spline weights by averaging over the  $B$  coefficients, resulting in a reduced matrix  $\mathbf{S}_{\mathrm{mean}}^{(l)} \in \mathbb{R}^{H \times F}$ , which can then be interpreted as a set of adaptive nonlinear weights acting on each feature across the hidden dimensions.

Next, to calculate the influence of feature  $f$  of node  $i$  on neuron  $h$  at a given hidden layer  $l$ , we multiply the corresponding gradient  $g_{i,f}$  by the mean spline weight  $\mathbf{S}_{\mathrm{mean}_h,f}^{(l)}$ . This product reflects how a change in the input feature is transformed nonlinearly by the model's internal structure. The overall feature importance  $\mathrm{I}_{i,f}^{(l)}$  for node  $i$  at layer  $l$  is computed by summing these contributions across all  $H$  hidden units, producing a scalar value:

$$
\mathbf {I} _ {i, f} ^ {(l)} = \sum_ {h = 1} ^ {H} g _ {i, f} \cdot \mathbf {S} _ {\mathrm {m e a n} _ {h, f}} ^ {(l)}
$$

This methodology integrates both the sensitivity of the features and the local transformations modeled by the splines, capturing the complex interactions between features that are fundamental to KANG's nonlinear structure. By doing so, we account not only for the direction of the change in prediction but also for the adaptive scaling that each feature undergoes along the graph edges.

# 2.3.2 EDGE IMPORTANCE

In KANG, edge importance captures how information flows between nodes, influencing predictions. By focusing on edge importance, we capture the interaction between nodes as modulated by the internal spline weights, which are central to the nonlinear transformations occurring along the edges. For a given node  $i$ , the importance of an edge  $(i,j)$ , where  $j \in \mathcal{N}(i)$  is a direct neighbor of  $i$ , is determined by analyzing how the features of node  $j$  are transformed and passed along the edge to node  $i$ .

This approach is particularly insightful because the spline weights  $\mathbf{S}_{\mathrm{spline}}$  play a crucial role in modulating the feature propagation along edges, allowing for adaptive nonlinear transformations of features as they flow through the graph. This makes the spline weights crucial for interpreting edge importance, as they capture complex feature interactions.

The spline weights in KANG serve a dual purpose: they not only determine how features are transformed between nodes but also govern the overall influence of edges on the model's prediction. When computing the influence of features on a node's prediction, the spline-modulated transformations provide insight into how individual features contribute to the target node. However, this

feature influence is inherently tied to the edge through which the features propagate. By extending this concept, we can derive the importance of an edge by aggregating the spline-modulated feature transformations and combining them with the feature activations. Essentially, edge importance emerges as a natural extension of feature influence, encapsulating how the transformation of each feature contributes to the overall prediction via the edge connecting two nodes.

Importances can be computed on a layer-by-layer basis. To determine the importance of an edge  $(i,j)$  between a target node  $i$  and a neighboring node  $j\in \mathcal{N}(i)$ , for the convolutional layer  $l$ , we first reduce the corresponding spline weights by averaging over the  $B$  spline coefficients, yielding the reduced matrix  $\mathbf{S}_{\mathrm{mean}}^{(l)}$ . This matrix encapsulates the non-linear transformation applied at layer  $l$  as features propagate along the edges. We then incorporate the feature activations  $\mathbf{a}_j^{(l)}\in \mathbb{R}^H$  and  $\mathbf{a}_i^{(l)}\in \mathbb{R}^H$ , where  $H$  is the hidden dimension of layer  $l$ , representing the transformed feature vectors for the neighbor  $j$  and the target node  $i$  respectively, after layer  $l$  of KANG. These activations reflect the local embeddings of the features post-convolution and are combined with the spline weights to assess the importance of edge  $(i,j)$ .

The edge importance,  $\Xi_{i,j}^{(l)}$ , for the edge  $(i,j)$  at layer  $l$  is computed by multiplying the spline-modulated feature weights  $\mathbf{S}_{\mathrm{mean}}^{(l)}$  with the weights  $\mathbf{W}^{(l)}$  of the convolutional layer  $l$ , for which we are computing the importance. This product is then multiplied by the signals  $\mathbf{a}_j^{(l)}$  and  $\mathbf{a}_i^{(l)}$ , summed over the hidden units  $H$ :

$$
\Xi_ {i, j} ^ {(l)} = \mathrm {m e a n} (\mathbf {W} ^ {(l)} \cdot \mathbf {S} _ {\mathrm {m e a n}} ^ {(l)}) \sum_ {h = 1} ^ {H} \left(a _ {h, i} ^ {(l)} \cdot a _ {h, j} ^ {(l)}\right)
$$

This formulation directly links nonlinear feature propagation along an edge to node  $i$ 's prediction. Since the spline weights remain static on the graph, they offer a consistent, interpretable framework for understanding how features are propagated through the edges, which are the fundamental pathways of information in a graph.

This approach to edge importance is highly beneficial because it breaks down the prediction process into localized interactions, helping us understand which edges are most responsible for driving the prediction. Unlike standard GNNs, where edges may simply aggregate features, KANG's spline-based transformations allow us to precisely identify how each edge modifies the feature representations in a nonlinear manner.

This section provided a comprehensive overview of the KANG architecture, its core components, and its interpretability capabilities. The next sections will delve into the experimental results and demonstrate the effectiveness of this new approach to GNNs.

# 3 EXPERIMENTS

# 3.1 DATASETS

We evaluated KANG on node classification, graph classification, and link prediction tasks using the benchmark datasets summarized in Appendix A.1. In the following section, we will discuss the experimental setup and the results obtained by KANG in comparison to state-of-the-art methodologies.

# 3.2 KANG PERFORMANCES

We compared the performances of our proposed model against established GNN architectures, namely GCN, GAT, using the GATv2 PyTorch Geometric implementation (Fey & Lenssen, 2019), GraphSAGE, and GIN. KANG was able to outperform all the mentioned GNN models in all tasks with all datasets, with the single exception of the link prediction with PubMed, in which GCN performed better.

Each method is unique, and the hyperparameters chosen for training are as crucial as the implementation itself. To ensure a fair comparison, each model has been trained following the guidelines

Table 1: Results in terms of average accuracy and standard deviation  $(\%)$  on 10 runs (node and graph classification) and average AUC-ROC (link prediction) on the test set obtained by KANG and the compared architectures. Our framework delivers higher accuracy and ROC, being the top-performing architecture.  

<table><tr><td>Dataset</td><td>GCN</td><td>GAT</td><td>GraphSAGE</td><td>GIN</td><td>KANG (Ours)</td></tr><tr><td colspan="6">Node Classification</td></tr><tr><td>Cora</td><td>77.5±1.0</td><td>78.7±1.1</td><td>73.6±2.5</td><td>75.5±1.2</td><td>79.5±0.8</td></tr><tr><td>PubMed</td><td>77.9±1.0</td><td>78.8±1.3</td><td>75.1±1.2</td><td>77.7±1.5</td><td>80.7±0.9</td></tr><tr><td>CiteSeer</td><td>67.6±1.5</td><td>68.7±0.5</td><td>63.1±2.3</td><td>63.1±1.9</td><td>69.1±1.5</td></tr><tr><td colspan="6">Link Prediction</td></tr><tr><td>Cora</td><td>87.0±8.9</td><td>89.8±0.6</td><td>82.0±6.8</td><td>75.0±1.1</td><td>90.4±0.5</td></tr><tr><td>PubMed</td><td>94.2±0.4</td><td>88.9±1.1</td><td>84.8±3.7</td><td>89.5±0.5</td><td>85.8±0.4</td></tr><tr><td>CiteSeer</td><td>81.2±3.3</td><td>82.0±4.4</td><td>77.9±1.4</td><td>83.4±1.8</td><td>84.7±0.6</td></tr><tr><td colspan="6">Graph Classification</td></tr><tr><td>MUTAG</td><td>67.5±4.0</td><td>64.0±5.4</td><td>74.0±8.3</td><td>74.5±6.5</td><td>93.0±4.6</td></tr><tr><td>PROTEINS</td><td>71.3±2.4</td><td>71.0±2.3</td><td>71.4±2.5</td><td>72.4±1.2</td><td>73.7±3.0</td></tr></table>

and hyperparameters provided by their authors. For training KANG, we conducted hyperparameter tuning to determine the best set of hyperparameter for each dataset and each task.

Hyperparameter tuning was conducted using a grid search over a broad range of potential values: learning rate [0.01, 0.001, 0.005], weight decay [1e-4, 1e-5], hidden channels [8, 16, 32, 64], dropout rate [0, 0.3, 0.6, 0.7], number of layers [1, 2, 3, 4], spline grid size [2, 3, 4, 8, 10, 20, 30], splines degree [1, 2, 4, 8, 10, 15], aggregation function [add, mean, max], and L2-regularization [0.0001, 0.00001]. All GNN models were trained for a maximum of 600 epochs for the Node Classification task, 700 epochs for Link Prediction, and 200 epochs for Graph Classification, with early stopping applied based on validation loss (node and graph classification) and validation AUC-ROC (link prediction). The optimal hyperparameters that we found for KANG are reported in Appendix A.2.

We report in Table 1 summarized results obtained by averaging over 10 runs the test accuracy (or AUC-ROC for link prediction) of the models that achieved the highest validation accuracy during training over each run. For each run, to achieve unbiased outcomes, we randomly split the datasets utilizing  $80\%$  as training set,  $10\%$  as validation set, and the remaining  $10\%$  as test set.

Performance metrics alone do not fully capture a model's capabilities; it is equally important to evaluate its scalability to larger datasets. To assess how the models perform as graph sizes increase, we compared them on synthetically generated datasets under consistent conditions, ensuring all architectures had the same number of layers and hidden units, leading to comparable numbers of trainable parameters. The graphs varied in size (1000, 5000, 10,000, and 20,000 nodes) and edge density (with probabilities of 0.05, 0.25, and 0.5 for edge creation). Although KANG provides inherent interpretability (Section 2.3), this advantage comes with a slight increase in computational cost due to the additional parameters and weights introduced by the splines. As a result, the models were evaluated in two scenarios: 1) where only the training time is considered and 2) where all models were trained alongside an explainer, specifically GNNExplainer. KANG does not need an additional explainer, as it provides direct interpretability. The results of the study can be found in Appendix A.4, where we show that KANG can scale efficiently to large graphs.

# 3.3 INTERPRETABILITY

The added value of KANG lies not only in its higher accuracy but also in its inherent interpretability. As pointed out in Section 2.3, the interpretability of KANG is two-fold: it provides 1) a means for node feature influence and 2) a measure for edge importance, accounting for the information flow throughout the graph. As a representative example, we determined the most influencing features for a node in the Cora dataset, computed as shows in Section 2.3.1. This information can be used to understand the most important features for a particular node's prediction, critical in scenarios not suitable for black-box predictions (medicine, life sciences, and finance among others). We evaluated

the most influent features obtained in terms of Fidelity  $(\mathrm{FID}^{+})$  and Infidelity  $(\mathrm{FID}^{-})$  of prediction accuracy (Yuan et al., 2022), adapted for node classification.

The Fidelity metric is defined as

$$
\mathbf {F I D} ^ {+} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\mathbb {1} \left(\hat {y} _ {i} = y _ {i}\right) - \mathbb {1} \left(\hat {y} _ {i} ^ {1 - m _ {i}} = y _ {i}\right)\right)
$$

where  $\hat{y}_i$  is the predicted label for node  $i$  using the original graph with all the features and  $y_i$  is its correct class,  $\hat{y}_i^{1 - m_i}$  is the predicted label for node  $i$  using the graph with the  $m_i$  most important features removed,  $\mathbb{1}(\cdot ,\cdot)$  is an indicator function and  $N$  is the number of nodes for which the metric is computed. Analogously, the Infidelity metric is defined as

$$
\mathrm {F I D} ^ {-} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\mathbb {1} \left(\hat {y} _ {i} = y _ {i}\right) - \mathbb {1} \left(\hat {y} _ {i} ^ {m _ {i}} = y _ {i}\right)\right)
$$

where  $\hat{y}_i^{m_i}$  is predicted lable for node  $i$  when only the  $m_i$  most important features retained. A good method should achieve high Fidelity and low Infidelity values. Table 2 shows the results considering different cutoffs for the top  $k$  features.

Table 2: Feature influence analysis on the Cora dataset. The table presents the  $\mathrm{FID^{+}}$  and  $\mathrm{FID^{-}}$  scores for various top-  $k$  feature cutoffs, computed on the correctly predicted samples from the test set.  $\mathrm{FID^{+}}$  and  $\mathrm{FID^{-}}$  values are stable across different cutoffs.

<table><tr><td>Top k</td><td>FID+</td><td>FID-</td></tr><tr><td>10%</td><td>0.31</td><td>0.84</td></tr><tr><td>20%</td><td>0.31</td><td>0.85</td></tr><tr><td>30%</td><td>0.31</td><td>0.85</td></tr></table>

We notice how the removal of features for a single node leads to marginal changes in the prediction, identified by low  $\mathrm{FID^{+}}$  and high  $\mathrm{FID^{-}}$  scores, indicating that the behavior of GNNs is not solely dependent on node features, but the overall structure and topology of the graph plays a crucial role. Indeed, the prediction of a node heavily relies on the messages passed by its neighbors and not only on its own features.

After analyzing feature influence, it is also possible to interpret KANG to understand the importance of the edges in the graph. As shown in Section 2.3.2, is it possible to determine the importance of the information flowing on the egdes of the graph, in order to analyze the messages passed to a target node and determine the most influent neighbors impacting on the its prediction. We show a representative example on the Cora dataset in Figure 2, comparing the results against GNNExplainer.

In KANG, the values (which have been normalized for comparison) genuinely represent the network message, meaning they are the actual values used in making the prediction. In contrast, GN-Explainer provides importance scores that are calculated post hoc on a subgraph optimized to maximize mutual information. We notice that both strategies prioritized similar edges. In particular, node with ID 2176 appears to be carrying the most important message for the target node 4 in both methodologies. Analogously, node 1761 is the least important neighbor (also belonging to a different class). This highlights that KANG interpretability is consistent with the explainability provided by GNExplainer, validating the usage of the network messages as a means for edge importance. The added value brought by KANG is that it does not need an additional explainer to be trained or used, thereby saving computation time and avoiding possible approximations introduced by such methodologies (Rudin, 2019).

Indeed, spline-based activation functions, which are central to KAN and consequently KANG, are inherently more interpretable compared to traditional neural network activation functions. Traditional GNNs use fixed, nonlinear activation functions like ReLU or Sigmoid, which can make it challenging to understand the decision boundaries or the transformations applied to the input features.

![](images/82704169071783736f91abc6ef8a26c33364d701aaa88db2378b3dc9c6ad0fe3.jpg)  
(a) Direct interpretation of KANG.

![](images/4fecddbd3bc09b3b1b3f17df3085c7754b59d10e0cb47fc3b1741b1d69d89516.jpg)  
Figure 2: Interpretability vs. explainability. Comparison of the direct interpretation of KANG (Figure 2a) and GNNExplainer applied to KANG (Figure 2b), trained on the Cora dataset for node classifier task. The explanation focuses on the node with ID 4. In Figure 2a, the output of the direct interpretation of the gradients and the weights of the splines of the neurons in the last convolutional layer (just before the KAN-based linear layer used for classification) is plotted. In Figure 2b, the edge mask returned by GNNExplainer is shown. The scores are normalized for visualization and comparison purposes. Additional examples can be found in Appendix A.5.  
(b) Output of GNNExplainer applied to KANG.

In contrast, spline functions are defined as piecewise polynomials which can be easily visualized and understood. The smooth and continuous nature of splines allows for a clear representation of how inputs are mapped to outputs. By examining the spline functions, one can see exactly how each input feature contributes to the final prediction and consequently how messages flowing throughout the graph influence the final output. This transparency makes it possible to trace the influence of individual features and edges, and understand the model's decision-making process.

# 4 CONCLUSIONS

In this paper, we propose KANG, a novel GNN architecture inspired by the Kolmogorov-Arnold theorem and based on the recently introduced KAN model. The added value of KANG is two-fold. First, it is more accurate than established state-of-the-art GNNs in node and graph classification and link prediction tasks. Second, thanks to the usage of splines and simple aggregation functions, KANG models are more interpretable. While KANG provides significant advancements in terms of interpretability, this is only a preliminary step. Future work should focus on enhancing the interpretability capabilities and exploring their application to even more complex graph structures, and compare the outcomes with state-of-art explainability strategies. Moreover, we showed that KANG can scale to large graphs.

We want highlight that KANG is not fully interpretable in every scenario, particularly in deeper networks where some information may be lost. However, KANG provides significantly greater transparency than other models by offering direct interpretability from the model itself, without relying on external explainability methods.

Looking forward, several research directions could further enhance KANG's performance and applicability. First, optimizing computational efficiency is essential, particularly by reducing memory usage and improving training and inference speed through techniques such as more efficient spline implementations, approximation methods, and parallelization. Second, incorporating edge features is a priority for extending KANG's capabilities. Developing methods to integrate edge information into the spline-based message passing and aggregation process could significantly improve predictive power.

Furthermore, applying KANG to real-world problems—such as biomedical research, where interpretability is crucial, or financial analytics, where it can aid in regulatory compliance—could demonstrate its practical utility and further validate the results with the aid of domain experts. Finally,

hybrid models combining KANG with advanced GNN architectures, such as attention mechanisms or variational techniques, may further boost performance while preserving interpretability.  
By addressing these limitations and pursuing these research directions, KANG can be further extended into a more robust and versatile tool for graph machine learning applications.

# REFERENCES

Tashin Ahmed and Md Habibur Rahman Sifat. GraphKAN: Graph kolmogorov arnold network for small molecule-protein interaction predictions. In ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications, 2024.  
David Baehrens, Timon Schroeter, Stefan Harmeling, Motoaki Kawanabe, Katja Hansen, and Klaus-Robert Müller. How to explain individual classification decisions. The Journal of Machine Learning Research, 11:1803-1831, 2010.  
Roman Bresson, Giannis Nikolentzos, George Panagopoulos, Michail Chatzianastasis, Jun Pang, and Michalis Vazirgiannis. KAGNNs: Kolmogorov-Arnold networks meet graph learning. arXiv preprint arXiv:2406.18380, 2024.  
Nadia Burkart and Marco F Huber. A survey on the explainability of supervised machine learning. Journal of Artificial Intelligence Research, 70:245-317, 2021.  
Alexandre Duval and Fragkiskos D Malliaros. GraphSVX: Shapley value explanations for graph neural networks. In Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference, ECML PKDD 2021, Proceedings, Part II 21, pp. 302-318. Springer, 2021.  
Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-256. JMLR Workshop and Conference Proceedings, 2010.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in Neural Information Processing Systems, 30, 2017.  
Simon Haykin. Neural networks: a comprehensive foundation. Prentice Hall PTR, 1998.  
Yotam Hechtlinger. Interpretation of prediction models using the input gradient. CoRR, abs/1611.07634, 2016. URL http://arxiv.org/abs/1611.07634.  
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989.  
Mehrdad Kiamari, Mohammad Kiamari, and Bhaskar Krishnamachari. GKAN: Graph kolmogorov-arnold networks. arXiv preprint arXiv:2406.06470, 2024.  
Thomas N. Kipf and Max Welling. Variational graph auto-encoders. CoRR, abs/1611.07308, 2016. URL http://arxiv.org/abs/1611.07308.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In 5th International Conference on Learning Representations, ICLR 2017, Conference Track Proceedings, 2017.  
A.N. Kolmogorov. On the representation of functions of several variables as a superposition of functions of a smaller number of variables. Dokl. Akad. Nauk, 108(2), 1956.  
Andrey Nikolaevich Kolmogorov. On the representation of continuous functions of several variables in the form of superpositions of continuous functions of one variable and addition. In *Reports of the Academy of Sciences*, volume 114, pp. 953-956. Russian Academy of Sciences, 1957.  
Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y Hou, and Max Tegmark. KAN: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756, 2024.

Andrea Mastropietro, Giuseppe Pasculli, Christian Feldmann, Raquel Rodríguez-Pérez, and Jürgen Bajorath. EdgeSHAPer: bond-centric Shapley value-based explanation method for graph neural networks. iScience, 25(10):105043, 2022.  
Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5):206-215, 2019.  
Rabia Saleem, Bo Yuan, Fatih Kurugollu, Ashiq Anjum, and Lu Liu. Explaining deep neural networks: A survey on the global interpretation methods. Neurocomputing, 513:165-180, 2022.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008.  
Lloyd S Shapley. A value for  $n$ -person games. In Harold W. Kuhn and Albert W. Tucker (eds.), Contributions to the Theory of Games II, pp. 307-317. Princeton University Press, 1953.  
Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In International conference on machine learning, pp. 3145-3153. PMIR, 2017.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: visualising image classification models and saliency maps. In Yoshua Bengio and Yann LeCun (eds.), 2nd International Conference on Learning Representations, ICLR 2014, Workshop Track Proceedings, 2014.  
Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International conference on machine learning, pp. 3319-3328. PMLR, 2017.  
Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In 6th International Conference on Learning Representations, ICLR 2018, Conference Track Proceedings, 2018.  
Boris Weisfeiler and Andrei Leman. The reduction of a graph to canonical form and the algebra which appears therein. nti, Series, 2(9):12-16, 1968.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In 7th International Conference on Learning Representations, ICLR 2019, 2019.  
Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. GNNExplainer: generating explanations for graph neural networks. Advances in Neural Information Processing Systems, 32, 2019.  
Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, and Shuiwang Ji. On explainability of graph neural networks via subgraph explorations. In International conference on machine learning, pp. 12241-12252. PMLR, 2021.  
Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in graph neural networks: A taxonomic survey. IEEE transactions on pattern analysis and machine intelligence, 45(5): 5782-5799, 2022.  
Fan Zhang and Xin Zhang. GraphKAN: Enhancing feature extraction with graph kolmogorov arnold networks. arXiv preprint arXiv:2406.13597, 2024.
