# CHARACTERIZE AND TRANSFER ATTENTION IN GRAPH NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Does attention matter and, if so, when and how? Our study on both inductive and transductive learning suggests that datasets have a strong influence on the effects of attention in graph neural networks. Independent of learning setting, task and attention variant, attention mostly degenerate to simple averaging for all three citation networks, whereas they behave strikingly different in the protein-protein interaction networks and molecular graphs: nodes attend to different neighbors per head and get more focused in deeper layers. Consequently, attention distributions become telltale features of the datasets themselves. We further explore the possibility of transferring attention for graph sparsification and show that, when applicable, attention-based sparsification retains enough information to obtain good performance while reducing computational and storage costs. Finally, we point out several possible directions for further study and transfer of attention.

# 1 INTRODUCTION

The modeling of graphs has become an active research topic in deep learning (Bronstein et al., 2017). Dozens of neural network models have been developed for exploiting the structural information of graphs (Scarselli et al., 2009; Bruna et al., 2014; Henaff et al., 2015; Duvenaud et al., 2015; Niepert et al., 2016; Defferrard et al., 2016), now collectively referred to as graph neural networks (GNNs).

Built upon the success of attention in NLP (Vaswani et al., 2017), Velicković et al. (2018) proposed the graph attention networks (GATs) to integrate multi-head self-attention into node feature update for adaptive weighting, with several extensions (Thekumparampil et al., 2018; Zhang et al., 2018; Monti et al., 2018; Svoboda et al., 2019; Trivedi et al., 2019). While the use of attention in GNNs is an attractive direction, several works also report that attention contributes little to the performance of GNNs (Zhang et al., 2018; Shchur et al., 2019). Considering the high computational cost of attention, the question is then that does attention help and, if so, when and how?

In this paper, we take a first step towards the question. We first identify the key questions for understanding attention and propose an analytical paradigm. With extensive experiments, our findings suggest that, although attention is motivated by inductive learning, its functionality depends highly on the characteristics of the datasets. The attention distributions across heads and layers are near uniform for all citation networks (Cora, Citeseer and Pubmed) while they get more concentrated over layers on the protein-protein interaction networks (PPI) and molecular graphs, with significant diversity among heads. That the attention distribution is a telltale sign of the nature of graph class is further verified with a meta graph classification experiment. With attention features as inputs, citation networks are indistinguishable whereas PPI and molecule graphs are.

Inspired by these findings, we hypothesize that attention carry semantic meanings when they are non-uniform and can be helpful for transfer learning. This has been the case in the NLP community (Radford et al., 2019), and is motivating many research efforts on understanding multi-head attention (Jain & Wallace, 2019; Clark et al., 2019; Voita et al., 2019). We attempt the idea of attention based sparsification – sparsifying a graph by retaining edges where attention are higher, with the intuition being that the resulting graph preserves enough information. We find that not only such attention-based sparsification is transferable (meaning, it can work on unseen graphs), it also affords us to train a cheaper model without using attention to fit the downstream task. Finally, we discuss several possible fruitful directions for further exploration, including theory, interpretability, and unsupervised learning.

# 2 RELATED WORK

Visualize and understand attention Several works attempted to visualize the learned attention by coloring edges or nodes based on the attention magnitudes (Velicković et al., 2018; Qiu et al., 2018). Thekumparamil et al. (2018) studied the averaged attention values between nodes with different or the same class labels. Shanthamallu et al. (2018) studied the attention GAT learned on two citation networks Cora and Citeseer with interquartile range metric and showed that they are near uniform. Knyazev et al. (2019) investigated the effectiveness of attentional pooling and find that attention is only effective when it is close to optimal. Our work differs in that we propose a general paradigm for analyzing graph attention, including how to characterize the overall attention statistics and how to measure the layer-wise and head-wise differences of the learned attention.

Transfer attention In the computer vision community, transferring the attention maps using a teacher-student network to improve the downstream tasks is a well-studied technique (Zagoruyko & Komodakis, 2017; Li et al., 2019). Our approach for transferring attention is different from these works in that we use the trained graph attention network as a graph sparsifier instead of a teaching signal. Yang et al. (2018) proposed to transfer the relational structure within the data, which is represented as a set of attention weights, to boost the performance of other tasks. Our transferring strategy is different from their work because we reduce the sparsity of the affinity matrix by removing the entries with smaller attention weights. Thus, we can train a cheap graph sparsifier to accelerate the training and testing speed.

# 3 BACKGROUND

# 3.1 GRAPH NEURAL NETWORKS

Let  $G$  be an undirected graph with node set  $\mathcal{V}$  and edge set  $\mathcal{E}$ , where each node  $i\in \mathcal{V}$  has a feature  $h_i^0\in \mathbb{R}^{n_0}$ . In a wide class of GNNs (Kipf & Welling, 2017; Hamilton et al., 2017; Velickovic et al., 2018), the basic feature update function for node  $i\in \mathcal{V}$  at the  $l + 1$ -th GNN layer takes the form of

$$
h _ {i} ^ {l + 1} = \sigma \left(\sum_ {j \in \mathcal {N} (i)} \alpha_ {i, j} ^ {l + 1} \mathbf {W} ^ {l + 1} h _ {j} ^ {l}\right),
$$

where  $\sigma$  is an activation function,  $\mathcal{N}(i)$  is a set containing  $i$  and its neighbors,  $\alpha_{i,j}^{l+1} \in \mathbb{R}$  is the attention weight of edge  $(j,i)$  in updating the feature of node  $i$ ,  $\mathbf{W}^{l+1} \in \mathbb{R}^{n_{l+1} \times n_l}$  is the projection matrix, and  $h_i^l, h_i^{l+1}$  are corresponding node features after the  $l$ -th and the  $l+1$ -th layer. With a sparse implementation, it has a time complexity of  $O(|\mathcal{V}|n_{l+1}n_l + |\mathcal{E}|n_{l+1})$ .

Graph Convolutional Network (GCN) (Kipf & Welling, 2017) and the mean variant of GraphSAGE (Hamilton et al., 2017) use static attention  $\frac{1}{\sqrt{|\mathcal{N}(i)|}}\frac{1}{\sqrt{|\mathcal{N}(j)|}}$  and  $\frac{1}{|\mathcal{N}(i)|}$ , which we separately refer to as GCN and uniform attention.

GAT (Veličković et al., 2018) uses a parameterized subnetwork to output the attention weights  $\alpha_{i,j}\mathrm{s}$ . Rather than using a single attention head as in Eqn. equation 3.1, GAT aggregates the outputs of multiple heads:

$$
\begin{array}{l} \alpha_ {i, j} ^ {l + 1, k} = \frac {\exp \left(\operatorname {s c o r e} \left(h _ {i} ^ {l} , h _ {j} ^ {l}\right)\right)}{\sum_ {j ^ {\prime} \in \mathcal {N} (i)} \exp \left(\operatorname {s c o r e} \left(h _ {i} ^ {l} , h _ {j ^ {\prime}} ^ {l}\right)\right)}, \qquad h _ {i} ^ {l + 1, k} = \sigma \left(\sum_ {j \in \mathcal {N} (i)} \alpha_ {i, j} ^ {l + 1, k} \mathbf {W} ^ {l + 1, k} h _ {j} ^ {l}\right), \\ h _ {i} ^ {l + 1} = \sigma \left(\operatorname {A g g r e g a t e} ^ {l + 1} \left(h _ {i} ^ {l + 1, 1}, \dots , h _ {i} ^ {l + 1, K ^ {l + 1}}\right)\right), \\ \end{array}
$$

where  $k$  is the index of the attention head and  $K^{l + 1}$  is the number of attention heads in the  $l + 1$ -th layer. Aggregate<sup>l+1</sup> aggregates all head results in the  $l + 1$ -th layer and we follow the approach of Velicković et al. (2018) to use concatenation for intermediate layers and average for the final layer.

Attention variants As in Luong et al. (2015), there are multiple ways to calculate the attention scores. In this paper, we focus on the following three types of attention, namely concat, dot product,

Table 1: Dataset task and learning setting  

<table><tr><td></td><td></td><td>Cora</td><td>Citeseer</td><td>Pubmed</td><td>PPI</td><td>CEP</td><td>HIV</td></tr><tr><td rowspan="2">Task</td><td>Node Classification</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>Graph Prediction</td><td></td><td></td><td></td><td></td><td>✓</td><td>✓</td></tr><tr><td rowspan="2">Setting</td><td>Transductive Learning</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>Inductive Learning</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td></tr></table>

and general:

LReLU  $(\mathbf{a}^T[\mathbf{W}h_i||\mathbf{W}h_j])$  (concat),  $(\mathbf{W}h_i)^T\mathbf{W}h_j$  (dot product),  $(\mathbf{W}h_i)^T\mathbf{B}\mathbf{W}h_j$  (general)

GATs use the concat attention. Zhang et al. (2018) and Ryu et al. (2018) separately explores dot product and general attention in GNNs. Also, LReLU( $\cdot$ ) means the leaky ReLU activation.

Graph-level prediction Based on the node representations generated by GNNs, we can also compute a graph representation (Li et al., 2018) for graph-level prediction problems like graph classification and regression:

$$
h _ {G} = \sum_ {v \in \mathcal {V}} \operatorname {S i g m o i d} \left(g \left(h _ {v} ^ {L}\right)\right) \operatorname {R e L U} \left(f \left(h _ {v} ^ {L}\right)\right),
$$

where  $L$  is the number of GNN layers,  $h_v^L$  is the representation of node  $v$  output by the last layer of GNN,  $g(\cdot)$  calculates the impact of node  $v$  on the graph representation, and  $f(\cdot)$  is a linear projection.

# 3.2 TASKS AND DATASETS

We consider the tasks of node classification and graph-level prediction. For modeling, we treat all graphs as undirected with untyped nodes and edges. Self loops are added to preserve information from previous node features. As in Velicković et al. (2018), we consider four datasets - citation networks Cora, Citeseer (Sen et al., 2008), PubMed (Namata et al., 2012) and PPI (Zitnik & Leskovec, 2017). Additionally, we include two more datasets of molecular graphs for graph-level prediction.

The Harvard Clean Energy Project (CEP) (Hachmann et al., 2011) estimates the photovoltaic efficiency of organic molecules. We use a subset of it pre-processed by Duvenaud et al. (2015); Ryu et al. (2018). The HIV dataset was initially introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen HIV for testing the ability of compounds to inhibit HIV replication. It was later included in the MoleculeNet benchmark (Wu et al., 2018) as a binary classification task. For both datasets, the node features are extracted based on DeepChem (Ramsundar et al., 2019) and RDKit (Landrum), which includes atom type, degree, and many other chemical properties.

Since GAT was partially motivated to work on unseen data, we consider two learning settings: transductive learning and inductive learning. In the transductive learning setting, the model can access the features of all nodes in the graph. However, only a fraction of the nodes are labeled in the training phase and the model is asked to predict the missing labels. In the inductive learning setting, we have two mutually exclusive sets of nodes separately for training and testing. The model is trained only on the features and labels of the nodes in the training set and is asked to predict the labels of the nodes in the testing set. A summary of the tasks and learning settings can be found in Table 1. We leave more detailed information like dataset statistics, training/testing split and features in Appendix A.

# 4 METHODOLOGY

The introduction of multi-head attention into multi-layer GNNs poses many interesting questions, we investigate five in this paper. Q1: In the GAT model, all nodes have different attention distributions on their incoming edges. How should we characterize the overall statistics of these learned attention distributions? Q2: How do attention distributions differ across different heads and layers? Q3: How does the choice of dataset, attention variant, and learning setting affect the learned attention? Q4:

Table 2: Discrepancy between learned and static attention  

<table><tr><td></td><td>Cora</td><td>Citeseer</td><td>Pubmed</td><td>PPI</td><td>CEP</td><td>HIV</td></tr><tr><td>uniform vs learned</td><td>0.0083</td><td>0.0020</td><td>0.0059</td><td>0.5442</td><td>0.1754</td><td>0.2376</td></tr><tr><td>GCN vs learned</td><td>0.1118</td><td>0.0796</td><td>0.1999</td><td>0.5791</td><td>0.1759</td><td>0.2258</td></tr></table>

Is the statistics of the learned attention related to the intrinsic properties of the graph? Q5: How to transfer attention for further usage?

To answer Q1, we propose multiple metrics for characterizing attention distributions. For Q2, we examine the metrics at different layers and compare the change of them over layers. To answer Q3, we run experiments to see how varying the dataset, attention variant and the learning setting impacts the learned attention. Previous works (Kipf & Welling, 2017; Hamilton et al., 2017; Velicković et al., 2018) only perform transductive learning on the citation networks and inductive learning on PPI. To fill in the gap, we perform transductive learning on PPI and inductive learning on the citation networks (see data processing strategy in Appendix B.1). We show that learning tasks are largely irrelevant. To answer Q4, we propose a new task called Meta Graph Classification which asks the model to distinguish the type of the graphs by the characteristics of the attention distributions. For Q5, we transfer attention for graph sparsification and examine whether we can preserve enough task-related information with a significant number of edges removed.

# 5 CHARACTERIZING ATTENTION

# 5.1 ANALYZING ATTENTION METRICS

Experiment settings Our attention study is completely based on the GAT architecture except that we try different types of attention mentioned in Section 3.1. We follow the experiment settings of the original authors whenever possible and perform a hyperparameter search otherwise. To make a fair comparison between attention variants, we use the same hidden size for each layer output across attention variants. The detailed settings can be found in Appendix B.2. Unless explicitly mentioned, we perform 100 random runs for Cora, Citeseer and Pubmed and 10 random runs for PPI, CEP and HIV. The test performance is mostly consistent across attention variants and is comparable to the original reported numbers, which we report in Appendix B.4.

Learned attention vs. static attention Attention-free GNNs employ static weights in updating node features. The first question is then how does learned attention differ from these static weights? If the learned attention are almost the same as the static ones, then there is no point in performing costly attention computation and we do not need to proceed with the analysis. We quantify the discrepancy between learned attention and the static one with  $\frac{1}{2|\mathcal{V}|}\sum_{i\in \mathcal{V}}\sum_{j\in \mathcal{N}(i)}|\alpha_{i,j}^{\mathrm{learned}} - \alpha_{i,j}^{\mathrm{static}}|$ . With a range of  $[0,1]$ , the larger the value, the larger the discrepancy is. Table 2 shows the discrepancy between the static attention and the learned attention for the first head with the 'concat' variant. Surprisingly, the discrepancy against the uniform attention is very small for the three citation networks, suggesting that each node attends similarly to different neighbors.

Head-wise and layer-wise differences Figure 2 visualizes the attention of a node over its incoming edges in Cora and PPI, based on three heads in the last layer. We can find that different heads behave distinctively in the PPI case and they are all uniform in the Cora case. We summarize the change of attention over heads and layers with several metrics. To quantify the variance between two head-wise distributions, we compute the averaged  $L_{1}$  norm of the difference between the mean distribution over all heads and the learned distribution for each head.

$$
\alpha_ {i, j} ^ {\text {m e a n}} = \frac {1}{K} \sum_ {k = 1} ^ {K} \alpha_ {i, j} ^ {k}, j \in \mathcal {N} (i), \quad \text {H e a d - w i s e V a r i a n c e} = \frac {1}{2 K} \frac {1}{| \mathcal {V} |} \sum_ {k = 1} ^ {K} \sum_ {i \in \mathcal {V}} \sum_ {j \in \mathcal {N} (i)} \left| \alpha_ {i, j} ^ {k} - \alpha_ {i, j} ^ {\text {m e a n}} \right|.
$$

To probe the concentration of attention, we compute the maximum pairwise difference of them within one-hop neighborhoods. To verify whether attention are on self-loops when they are concentrated and GNNs degenerate to MLPs, we monitor the self-loop attention values. We average the metrics

![](images/52e65c485febde060926e6834cedaae83c453916b71a47e8e25f1bfd8cd1f534.jpg)

![](images/f564da60164fba2e049f630a2a9f7a5e0f5b6189c2a77df222aa0ea0b41b03dc.jpg)  
Figure 1: Comparison of the learned attention across layers, datasets and attention variants

![](images/023244a109ff043897565b37136caceec6c9fa2c6ced54abf47a93a7c0494e93.jpg)  
Figure 2: Visualization of the learned attention of one node. Nodes are colored by labels and edges are colored by attention magnitude.  
Figure 3: Attention learned by training on Cora and PPI with inductive and transductive settings respectively.

over all nodes and heads for a layer in each run, and compute the mean and standard deviation of the averaged metrics in all runs.

Varying settings and attention variants We leverage the defined metrics to compare the learned attention by different attention variants. Figure 1 visualizes the head-wise and layer-wise metrics for Cora, Pubmed, PPI and CEP. Independent of the attention variant, the learned attention change little across layers for Cora and Pubmed while the increasing max pairwise difference indicates that they get more concentrated with deeper layers for PPI and CEP. Besides, these attention do not get concentrated on self loops based on relatively stable values. We also experiment with different training settings on these graphs. Figure 3 shows the learned attention when training inductively on Cora while training transductively on PPI. We observe the similar phenomenon of the learned attention, which eliminates the effect of training settings.

# 5.2 META GRAPH CLASSIFICATION

Previous experiments suggest that the attention learned are highly graph-dependent and their characteristics can be predicted with proper knowledge of graph semantics. To verify it, we attempt to infer the graph types based on the attention learned. Specifically, we perform graph classification with attention-based features.

Table 3: Graph Classification Accuracy  

<table><tr><td></td><td>Concat</td><td>General</td><td>Dot product</td></tr><tr><td>All Layers</td><td>94.1 ± 0.5%</td><td>95.6 ± 0.7%</td><td>95.5 ± 0.4%</td></tr><tr><td>First Layer</td><td>81.3 ± 1.1%</td><td>88.4 ± 0.8%</td><td>91.3 ± 0.6%</td></tr><tr><td>Second Layer</td><td>83.5 ± 0.6%</td><td>89.7 ± 0.6%</td><td>85.3 ± 0.5%</td></tr></table>

![](images/9d21770279428b79f3ba947549b7f3dd1d6eacfae7c4c2e7efc0807ab1368f1f.jpg)  
All layers

![](images/2f7761d47f5dc67ee4d1ecffefddf36f977346ebb99e9d65116275cf740b2f69.jpg)  
Layer 1

![](images/afe4c6ab498999416bf165e02b19a0843b48e099e87e1c14fcab749e26435e6e.jpg)  
Layer 2

Figure 4: t-SNE visualization of 'concat' attention based features. From left to right, the features are from all layers, the first, and the second layer, respectively. See Appendix B.5 for more results.  
![](images/4e4223b0b887d78c5792c9383b7cc96ad941d7e0721cbaad361870c4a1114344.jpg)  
cora  
citeseer  
pubmed  
- ppi

![](images/3d685481b09b660562b4f9b9e53e6768df8bebc2cd6430a014028db5a9ca69db.jpg)  
CEP  
HIV

Synthetic dataset We construct a synthetic dataset for graph classification with two steps. First, we collect 480 samples/subgraphs of 20 to 30 nodes from each dataset. For HIV and CEP, we choose 480 graphs and construct a balanced subset. For the rest four datasets, we sample 480 graphs for each using random walk as in the case of inductive learning on citation networks. Second, we separately train a 2-layer GAT for each collected dataset and compute the attention metrics for each layer. The mean and standard deviation of the metrics are then used as the graph features. We leave more details in Appendix B.5.

We train a logistic regression classifier for graph classification, where  $20\%$  of the graphs are used for training and the rest are used for testing. We have separately experimented with attention metrics from all layers, the first layer, and the second layer. The classification performance is reported in Table 3, with all experiments repeated for 10 times.

In the experiments, we find that more than  $80\%$  of the incorrect classifications happen within citation subgraphs. This is also verified by our t-SNE (van der Maaten & Hinton, 2008; Pedregosa et al., 2011) visualization of the attention metrics in Figure 4. We can see that the attention metrics of citation networks tend to be indistinguishable while the attention metrics of other datasets are better separated and clustered.

# 6 ATTENTION-BASED GRAPH SPARSIFICATION

# 6.1 PPI SPARSIFICATION FOR GAT PREDICTION

We consider two intuitive heuristics for attention based sparsification: 1) local top- $k$  sparsification selects up to  $k$  incoming edges for each node with highest attention values; 2) global threshold sparsification selects edges whose attention value exceeds a pre-specified threshold over the entire graph. Across all GNN layers, we perform attention-based sparsification for each attention head to get a subset of edges. We then take the union of the edge subsets to get the edge set for the sparsified graph. All self loops are selected to preserve the information of original node features.

Inspired by recent research on sampling based training of GNNs (Hamilton et al., 2017; Huang et al., 2018), we consider two baseline random sparsification for comparison: 1) uniform neighbor sparsification uniformly samples at most  $k$  incoming edges for each node without replacement and we compare it against local top-  $k$  sparsification; 2) uniform graph-wise sparsification uniformly

![](images/4b86a89e97f98a60d9afe06f1d48efa4fcb87c1f51fd81d6f3c674587bf0eaa7.jpg)  
Figure 5: Comparison of attention based sparsification against baseline random sparsification. The result of training on unsparsified graphs is included for reference. For local top- $k$  sparsification we consider  $1 \leq k \leq 8$ . For global threshold sparsification, we consider threshold in  $\{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005\}$ .

![](images/1803ec83284fe9258e5f5c83454e1df3034afd64e2ae2a11c6ee284f35570138.jpg)

![](images/7fb3db953c96ed4a3e4a612e4993a040dd052c0baf25bef2e00fa5b6cd008c02.jpg)  
Figure 6: Sparsify graphs with one GAT and predict on sparsified graphs with another GNN

samples a proportion of edges over the entire graph(s) without replacement and we compare it against threshold sparsification.

We first experiment attention-based graph sparsification with  $PPI$ , which tends to have sharp attention-based on our previous study and is relatively dense (56944 nodes and 1644208 edges with self loops and bi-directional edges added). Despite similar sharp attention, molecules are not good candidates for the experiment as they are already sparse and on average each atom only has less than three neighbors. As illustrated in figure 6, after training a GAT on PPI, we perform attention-based sparsification and re-train a GAT on the sparsified training and validation graphs.

Figure 5 compares the results of attention-based sparsification with concat variant against random sparsification. 1) With a similar degree of sparsification, the attention-based sparsification consistently performs better than the baseline in terms of test metric and its variance across runs; 2) We can reach a test accuracy comparable to the original result with only  $40\% \sim 50\%$  edges left in the training and validation graphs.

# 6.2 SPARSIFICATION WITH GENTLE ATTENTION

What if the attention are neither uniform nor sharp? Our previous study shows that the attention in Pubmed are not completely uniform in the last GAT layer with dot product and general attention variants so we use it as a testbed for the study. In cases where attention are not very sharp, threshold sparsification is not very useful and we consider only top- $k$  sparsification.

Table 4 compares top-  $k$  sparsification with dot product attention against uniform neighbor sparsification. With similar proportion of edges left in the training and validation graphs, the top-  $k$  sparsification consistently outperforms uniform neighbor sparsification and achieves a performance comparable to that of training on the raw graphs.

Table 4: Attention-based sparsification against random sparsification on Pubmed  

<table><tr><td>Sparsification</td><td>% edges left</td><td>Test score</td></tr><tr><td>None</td><td>100%</td><td>78.20 ± 0.70%</td></tr><tr><td>local top-k (k=1)</td><td>57.37 ± 0.78%</td><td>77.57 ± 0.12%</td></tr><tr><td>local top-k (k=2)</td><td>71.44 ± 0.74%</td><td>78.33 ± 0.29%</td></tr><tr><td>uniform neighbor</td><td>59.13%</td><td>74.40 ± 0.36%</td></tr><tr><td>uniform neighbor</td><td>72.72%</td><td>75.47 ± 1.03%</td></tr></table>

Table 5: Varying GAT hidden sizes for local top-1 sparsified prediction with GraphSAGE on PPI. For reference, originally GraphSAGE can reach a test score of 0.9802 without graph sparsification. Using baseline sparsification methods, we reach at best a test score of 0.5431 with about  $30\%$  edges left.  

<table><tr><td># hidden</td><td>GAT test score</td><td># GAT parameters</td><td>% edges left</td><td>GraphSAGE test score</td></tr><tr><td>4</td><td>0.5208</td><td>27574</td><td>29.55%</td><td>0.9700</td></tr><tr><td>16</td><td>0.7718</td><td>107686</td><td>30.06%</td><td>0.9776</td></tr><tr><td>64</td><td>0.9704</td><td>520294</td><td>30.95%</td><td>0.9802</td></tr></table>

# 6.3 SPARSIFICATION WITH HARD PREDICTION

As the attention-based sparsification does manage to preserve enough task-related information, it has the potential to speed up the computation by significantly reducing the number of edges. The most aggressive approach is to use a light GAT with a lot fewer parameters for graph sparsification and an attention-free GNN for prediction on the sparsified graphs.

We explore this idea by varying the hidden sizes of attention heads in GAT and use a GraphSAGE for prediction on  $PPI$ . The GraphSAGE has 3 layers and each layer has a hidden size of 512. We consider the mean aggregator for it with skip connection added. The results are summarized in Table 5. Our experiments show that a GraphSAGE model can achieve a good performance with training on sparsified graphs as long as we sparsify test graphs. Surprisingly, while a smaller hidden size in GAT does harm its own prediction performance, it has little effect on capturing important edges for GraphSAGE prediction even with top-1 sparsification. We note that the number of parameters in GraphSage and the baseline GAT is 1.2M and 3.6M, respectively, i.e. a 3-fold reduction.

# 7 CONCLUSIONS AND DISCUSSIONS

In this work, we propose an analytical paradigm that can summarize the characteristics of multi-head attention learned over graphs and compare them against topology-based static attention. This allows a deeper understanding of attention beyond simply comparing the performance of models and motivates further use of attention like transfer learning. In addition to the attention-based sparsification we explored, we believe below are several interesting directions that are underexplored:

- Theory. Many efforts have performed to theoretically understand and explain GNNs, particularly their connection to kernel methods and Weisfeiler-Lehman tests (Morris et al., 2019; Xu et al., 2019; Maron et al., 2019), but few of them have considered attention.  
- Interpretability. The use of attention can add interpretability. This is particularly valued in risk-sensitive scenarios like medicine. Several efforts have been made in the chemistry community (Ryu et al., 2018; Preuer et al., 2019; Xiong et al., 2019).  
- Unsupervised learning. Unsupervised representation learning is an important approach when the training of a model is expensive and the labeled data is scarce. This is particularly the case for biological networks and molecular graphs due to the need of wet-lab experiments. The NLP community has witnessed the success of unsupervised learning with attention (Radford et al., 2019) and we might expect the same for GNNs. Weihua Hu (2019) demonstrates the effectiveness of training GNNs with unsupervised learning for chemistry and biology, but did not employ attention.

# REFERENCES

Aids antiviral screen data. https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data. Accessed: 2017-09-27.  
Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: Going beyond euclidean data. IEEE Signal Processing Magazine, 34:18-42, 2017.  
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations, 2014.  
Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. What does bert look at? an analysis of bert's attention. arXiv preprint arXiv:1906.04341, 2019.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems 29, pp. 3844-3852. 2016.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alan Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in Neural Information Processing Systems 28, pp. 2224-2232. 2015.  
Johannes Hachmann, Roberto Olivares-Amaya, Sule Atahan-Evrenk, Carlos Amador-Bedolla, Roel S Sánchez-Carrera, Aryeh Gold-Parker, Leslie Vogt, Anna M Brockway, and Alán Aspuru-Guzik. The Harvard clean energy project: large-scale computational screening and design of organic photovoltaics on the world community grid. *The Journal of Physical Chemistry Letters*, 2(17): 2241-2251, 2011.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems 30, pp. 1024-1034. 2017.  
Mikael Henaff, Joan Bruna, and Yann LeCun. Deep convolutional networks on graph-structured data. arXiv preprint arXiv:1506.05163, 2015.  
Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. Adaptive sampling towards fast graph representation learning. In Advances in Neural Information Processing Systems 31, pp. 4558-4567. 2018.  
Sarthak Jain and Byron C. Wallace. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.  
Boris Knyazev, Graham W. Taylor, and Mohamed R. Amer. Understanding attention in graph neural networks. In Workshop on Representation Learning on Graphs and Manifolds, International Conference on Learning Representations, 2019.  
Greg Landrum. Rdkit: open-source cheminformatics. http://www.rdkit.org/. Accessed: 2019-05-16.  
Jure Leskovec and Christos Faloutsos. Sampling from large graphs. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 631-636, 2006.  
Junying Li, Deng Cai, and Xiaofei He. Learning graph-level representation for drug discovery. 2017.  
Xingjian Li, Haoyi Xiong, Hanchao Wang, Yuxuan Rao, Liping Liu, and Jun Huan. DELTA: Deep learning transfer using feature map with attention for convolutional networks. In ICLR. 2019.  
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, and Peter Battaglia. Learning deep generative models of graphs. 2018.

Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, 2015.  
Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. In Advances in Neural Information Processing Systems 32, 2019.  
Federico Monti, Oleksandr Shchur, Aleksandar Bojchevski, Or Litany, Stephan Gunnemann, Michael, and Bresson. Dual-primal graph convolutional networks. arXiv preprint arXiv:1806.00770, 2018.  
Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence, 2019.  
Galileo Namata, Ben London, Lise Getoor, and Bert Huang. Query-driven active surveying for collective classification. In Proceedings of the Workshop on Mining and Learning with Graphs, 2012.  
Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. Learning convolutional neural networks for graphs. In Proceedings of The 33rd International Conference on Machine Learning, pp. 2014-2023, 2016.  
Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Eduard Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.  
Kristina Preuer, Günter Klambauer, Friedrich Rippmann, Sepp Hochreiter, and Thomas Unterthiner. Interpretable deep learning in drug discovery. arXiv preprint arXiv:1903.02788, 2019.  
Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang. Deepinf: Social influence prediction with deep learning. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery, pp. 2110-2119, 2018.  
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.  
Bharath Ramsundar, Peter Eastman, Patrick Walters, Vijay Pande, Karl Leswing, and Zhenqin Wu. Deep Learning for the Life Sciences. O'Reilly Media, 2019.  
Seongok Ryu, Jaechang Lim, Seung Hwan Hong, and Woo Youn Kim. Deeply learning molecular structure-property relationships using attention- and gate-augmented graph convolutional network. arXiv preprint arXiv:1805.10988, 2018.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20:61-80, 2009.  
Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, 29, 2008.  
Uday Shankar Shanthamalli, Jayaraman J. Thiagarajan, and Andreas Spanias. Improving robustness of attention models on graphs. arXiv preprint arXiv:1811.00181, 2018.  
Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan Gunnemann. Pitfalls of graph neural network evaluation. arXiv preprint arXiv:1811.05868, 2019.  
Aravind Subramanian, Pablo Tamayo, Vamsi K. Mootha, Sayan Mukherjee, Benjamin L. Ebert, Michael A. Gillette, Amanda Paulovich, Scott L. Pomeroy, Todd R. Golub, Eric S. Lander, and Jill P. Mesirov. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. In Proceedings of the National Academy of Sciences, volume 102, pp. 15545-15550, 2005.

Jan Svoboda, Jonathan Masci, Federico Monti, Michael Bronstein, and Leonidas Guibas. Peernets: Exploiting peer wisdom against adversarial attacks. In International Conference on Learning Representations, 2019.  
Kiran K. Thekumparampil, Chong Wang, Sewoong Oh, and Li-Jia Li. Attention-based graph neural network for semi-supervised learning. arXiv preprint arXiv:1803.03735, 2018.  
Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. Dyrep: Learning representations over dynamic graphs. In International Conference on Learning Representations, 2019.  
L.J.P. van der Maaten and G.E. Hinton. Visualizing high-dimensional data using t-sne. Journal of Machine Learning Research, 9:2579-2605, 2008.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, pp. 5998-6008. 2017.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.  
Elena Voita, Rico Sennrich, and Ivan Titov. The bottom-up evolution of representations in the transformer: A study with machine translation and language modeling objectives. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.  
Joseph Gomes Marinka Zitnik Percy Liang Vijay Pande Jure Leskovec Weihua Hu, Bowen Liu. Pre-training graph neural networks. arXiv preprint arXiv:1903.02788, 2019.  
Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Genisses, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Molecularnet: a benchmark for molecular machine learning. Chemical science, 9(2):513-530, 2018.  
Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, and Mingyue Zheng. Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism. Journal of Medical Chemistry, 2019.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.  
Zhilin Yang, Jake Junbo Zhao, Bhuwan Dhingra, Kaiming He, William W Cohen, Ruslan Salakhutdinov, and Yann LeCun. GLoMo: Unsupervisedly learned relational graphs as transferable representations. In NIPS. 2018.  
Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In *ICLR*. 2017.  
Jiani Zhang, Xingjian Shi, Junyuan Xie, Hao Ma, Irwin King, and Dit-Yan Yeung. Gaan: Gated attention networks for learning on large and spatiotemporal graphs. In The Conference on Uncertainty in Artificial Intelligence, 2018.  
Marinka Zitnik and Jure Leskovec. Predicting multicellular function through multi-layer tissue networks. Bioinformatics, 33(14):190-198, 2017.
