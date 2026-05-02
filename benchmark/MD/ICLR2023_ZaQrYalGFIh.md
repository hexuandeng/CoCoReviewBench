# GLINKX: A SCALABLE UNIFIED FRAMEWORK FOR HOMOPHIOUS AND HETEROPHIOUS GRAPHS

Anonymous authors

Paper under double-blind review

# ABSTRACT

In graph learning, there have been two predominant inductive biases regarding graph-inspired architectures: On the one hand, higher-order interactions and message passing work well on homophilous graphs and are leveraged by GCNs and GATs. Such architectures, however, cannot easily scale to large real-world graphs. On the other hand, shallow (or node-level) models using ego features and adjacency embeddings work well in heterophilous graphs. In this work, we propose a novel scalable shallow method – GLINKX – that can work both on homophilous and heterophilous graphs. GLINKX leverages (i) novel monophilous label propagations (ii) ego/node features, (iii) knowledge graph embeddings as positional embeddings, (iv) node-level training, and (v) low-dimensional message passing. Formally, we prove novel error bounds and justify the components of GLINKX. Experimentally, we show its effectiveness of it on several homophilous and heterophilous datasets.

# 1 INTRODUCTION

In recent years, graph learning methods have emerged with strong performance for various ML tasks. Graph ML methods leverage the topology of graphs underlying the data (Battaglia et al., 2018) to improve their performance. Two very important design options for proposing graph ML based architectures in the context of node classification are related to whether the data is homophilous or heterophilous.

For homophilous data – where neighboring nodes share similar labels (McPherson et al., 2001; Altenburger & Ugander, 2018a) – Graph Convolutional Network (GCN)-based methods (GCN, GAT, etc.) (Kipf & Welling, 2016; Velivcković et al., 2017; Zhu et al., 2020) are able to achieve high accuracy. In the GCN paradigm, message passing and higher-order interactions help node classification tasks in the homophilous setting since such inductive biases tend to bring the learned representations of linked nodes close to each other. However, GCN-based architectures suffer from scalability issues. Performing (higher-order) propagations during the training stage is hard to scale in large graphs because the number of nodes grows exponentially with the increase of the filter receptive field. Thus, for practical purposes, GCN-based methods require node sampling which substantially increases their training time. For this reason, architectures (Huang et al., 2020; Zhang et al., 2022b; Sun et al., 2021; Maurya et al., 2021; Rossi et al., 2020) that leverage propagations outside of the training loop (as a preprocessing step) have shown promising results in terms of scaling to large graphs.

In heterophilous datasets (Rogers et al., 2014), the nodes that are connected tend to have different labels. Currently, many works that address heterophily can be classified into two categories concerning scale. On the one hand, recent successful architectures (in terms of accuracy) (Jin et al., 2022a; Di Giovanni et al., 2022; Zheng et al., 2022b; Luan et al., 2021; Chien et al., 2020) that address heterophily resemble GCNs in terms of design and thus suffer from the same scalability issues. On the other hand, shallow or node-level models (see e.g. (Lim et al., 2021; Zhong et al., 2022)), i.e. models that are treating graph data as tabular data and do not involve propagations during training, have shown a lot of promise for large heterophilous graphs. In (Lim et al., 2021), it is shown that combining ego embeddings (node features) and adjacency embeddings works in the heterophilous setting. One element that LINKX exploits via the adjacency embeddings is monophily (Altenburger & Ugander, 2018a;b), namely the similarity of the labels of a node's neighbors. However, their design is still impractical in real-world data since the method (LINKX) is not inductive (see Section 2),

# Algorithm 1 GLINKX Algorithm

Input: Graph  $G(V,E)$  with train set  $V_{\mathrm{train}} \subseteq V$ , node features  $X$ , labels  $Y$

Output: Label Predictions  $Y_{\mathrm{final}}$

1st Stage. Pre-train knowledge graph embeddings  $P$  with Pytorch Biggraph.

2nd Stage. Propagate labels and predict neighbor distribution

1. Forward Propagation: Calculate  $\hat{\pmb{y}}_i = \frac{\sum_{j\in V_{\mathrm{train}}:j\to i}\pmb{y}_j}{|\{j\in V_{\mathrm{train}}:j\to i\}|}$  for all  $i\in V_{\mathrm{train}}$  
2. Learn distribution of a node's neighbors: For each epoch, calculate  $\tilde{\pmb{y}}_i = f_1(\pmb {\xi}_i,\pmb {p}_i;\pmb {\theta}_1)$  for  $i\in V_{\mathrm{train}}$

Update the parameters s.t.  $\mathcal{L}_{\mathrm{CE},1}(\pmb{\theta}_1) = \sum_{i \in V_{\mathrm{train}}} \mathrm{CE}(\hat{\pmb{y}}_i, \tilde{\pmb{y}}_i; \pmb{\theta}_1)$  is maximized.

Let  $\theta_1^*$  be the parameters at the end of the training that correspond to the epoch with the best validation accuracy.

3rd Stage. Propagate labels backwards and train model to predict a node's own labels

1. Backward Propagation: Calculate  $\pmb{y}_i' = \frac{\sum_{j \in V : j \to i} \tilde{\pmb{y}}_j}{|\{j \in V : i \to j\}|}$  for all  $i \in V_{\mathrm{train}}$ , where  $\tilde{\pmb{y}}_j = f_1(\pmb{\xi}_j, \pmb{p}_j; \pmb{\theta}_1^*)$  
2. Learn a node's own distribution: For each epoch, calculate  $y_{\text{final},i} = f_2(\pmb{\xi}_i, \pmb{p}_i, \pmb{y}_i'; \pmb{\theta}_2)$ .

Update the parameters s.t.  $\mathcal{L}_{\mathrm{CE},2}(\pmb{\theta}_2) = \sum_{i \in V_{\mathrm{train}}} \mathrm{CE}(y_i, y_{\mathrm{final},i}; \pmb{\theta}_2)$  is maximized.

Return  $Y_{\mathrm{final}}$

and embedding the adjacency matrix directly requires many parameters in a model. In LINKX, the adjacency embedding of a node can alternatively be thought of as a positional embedding (PE) of the node in the graph, and recent developments (Kim et al., 2022; Dwivedi et al., 2021; Lim et al., 2021) have shown the importance of PEs in both homophilous and heterophilous settings. However, most of these works suggest PE parametrizations that are difficult to compute in large-scale settings. Accordingly, more scalable ways of computing PEs via knowledge graph embeddings (El-Kishky et al., 2022; Lerer et al., 2019; Bordes et al., 2013; Yang et al., 2014) are useful in practical settings.

Goal & Contribution: In this work, we develop a scalable method for node classification that: (i) works both on homophilous and heterophilous graphs (ii) is simpler and faster than conventional message passing networks (by avoiding the neighbor sampling and message passing overhead during training), and (iii) can work in both a transductive and an inductive setting. For a method to be scalable, we argue that it should: (i) run models on node-scale (thus leveraging i.i.d. minibatching), (ii) avoid doing message passing during training and do it a constant number of times before training, and (iii) transmit small messages along the edges. Our proposed method - GLINKX (see Section 3) - combines all the above desiderata. GLINKX has three components: (i) ego embeddings $^2$ , (ii) PEs inspired by architectures suited for heterophilous settings, and (iii) scalable 2nd-hop-neighborhood propagations inspired by architectures suited for monophilous settings. We provide novel theoretical error bounds and justify components of our method (Section 3.4). Finally, we evaluate GLINKX's empirical effectiveness on several homophilous and heterophilous datasets (Section 4).

# 2 PRELIMINARIES

# 2.1 NOTATION

We denote scalars with lower-case, vectors with bold lower-case letters, and bold upper-case letters are used to denote matrices. We consider a directed graph  $G = G(V,E)$  with vertex set  $V$  with  $|V| = n$  nodes, and edge set  $E$  with  $|E| = m$  edges, and adjacency matrix  $A$ . Let  $X \in \mathbb{R}^{n \times d_X}$  represent the  $d_X$ -dimensional node feature matrix and  $P \in \mathbb{R}^{n \times d_P}$  represent the  $d_P$ -dimensional node positional embedding matrix. A node  $i$  has a feature vector  $\boldsymbol{x}_i \in \mathbb{R}^{d_X}$  and a positional embedding  $\boldsymbol{p}_i \in \mathbb{R}^{d_P}$  and belongs to a class  $y_i \in \{1,\dots,c\}$ . The training set is denoted by  $V_{\mathrm{train}}$ , validation set by  $V_{\mathrm{valid}}$ , and test set by  $V_{\mathrm{test}}$ .  $\mathbb{I}\{\cdot\}$  denotes the indicator function.  $\Gamma_c$  is the  $c$ -dimensional simplex.

# 2.2 GRAPH CONVOLUTIONAL NEURAL NETWORKS

In homophilous datasets, GCN-based methods have been used for node classification. GCNs (Kipf & Welling, 2016) utilize feature propagations together with non-linearities to produce node embeddings. More specifically, a GCN consists of multiple layers where each layer  $i$  collects  $i$ -th hop information from the nodes, through propagations, and forwards this information to the  $i + 1$ -th layer. More specifically, if  $G$  has a symmetrically-normalized adjacency matrix  $A_{sym}$  (ignoring the directionality of edges), then a GCN layer has the form

$$
\boldsymbol {H} ^ {(0)} = \boldsymbol {\xi}, \boldsymbol {H} ^ {(i + 1)} = \sigma (\boldsymbol {A} _ {s y m} \boldsymbol {H} ^ {(i)} \boldsymbol {W} ^ {(i)}) \forall i \in [ L ], \boldsymbol {Y} = \operatorname {s o f t m a x} (\boldsymbol {H} ^ {(L)}).
$$

Here  $\pmb{H}^{(i)}$  is the embedding from the previous layer,  $\pmb{W}^{(i)}$  is a learnable projection matrix and  $\sigma(\cdot)$  is a non-linearity (e.g. ReLU, sigmoid, etc.).

# 2.3 LINKX

In heterophilous datasets, the simple method of LINKX has been shown to perform well. LINKX combines two components – MLP on the node features  $\mathbf{X}$  and LINK regression (Altenburger & Ugander, 2018a) on the adjacency matrix – as follows:

$$
\boldsymbol {H} _ {X} = \operatorname {M L P} _ {X} (\boldsymbol {X}), \boldsymbol {H} _ {A} = \operatorname {M L P} _ {A} (\boldsymbol {A}), \boldsymbol {Y} = \operatorname {R e s N e t} (\boldsymbol {H} _ {X}, \boldsymbol {H} _ {A}).
$$

# 2.4 NODE CLASSIFICATION

In node classification problems on graphs, we have a model  $f(\mathbf{X}, \mathbf{Y}_{\mathrm{train}}, \mathbf{A}; \boldsymbol{\theta})$  that takes as an input the node features  $\mathbf{X}$ , the training labels  $\mathbf{Y}_{\mathrm{train}}$  and the graph topology  $\mathbf{A}$  and produces a prediction for each node  $i$  of  $G$ , which corresponds to the probability that a given node belongs to any of  $c$  classes (with the sum of such probabilities being one). The model is trained with back-propagation. Once trained, the model can be used for the prediction of labels of nodes in the test set.

There are two training regimes: transductive and inductive. In the transductive training regime, we have full knowledge of the graph topology (for the train, test, and validation sets) and the node features and the task is to predict the labels of the validation and test set. In the inductive regime, only the graph induced by  $V_{\text{train}}$  is known at the time of training and then the full graph is revealed for prediction on the validation and test sets. In real-world scenarios, such as online social networks, the dynamic nature of problems makes the inductive regime particularly useful.

# 2.5 HOMOPHILY, HETEROPHILY & MONOPHILY

Homophily and Heterophily: There are various measures of homophily in the GNN literature like node homophily, and/or edge homophily (Lim et al., 2021). Intuitively, homophily in a graph implies that nodes with similar labels are connected to each other. GNN based approaches like GCN, GAT, etc. leverage this property to improve the node classification performance. Alternatively, if a graph has a low homophily – namely nodes that connect to each other tend to have different labels – it is said to be heterophilous. In other words, a graph is said to be heterophilous if neighboring node do not share similar labels.

Monophily: Generally, a graph is said to be monophilous if the label of a node is similar to that of its neighbors' neighbors. In the context of a directed graph, monophily can be thought of a structure that resembles Fig. 2(a) where similar nodes (in this case 3 green nodes connected to a yellow node) are connected to a node with a different label. We argue that encoding monophily into a model can be useful for both heterophilous and homophilous graphs. In homophilous graphs, monophily will fundamentally encode 2nd-hop neighbor's label information and since in such graphs neighboring nodes have similar labels it can provide useful signal for node classification. In heterophily, neighboring nodes have dissimilar labels but the 2nd-hop neighbors may share the same label thus providing useful information for node classification. In fact, monophily has been shown

![](images/fb45bc2468bb536b3171484233efeb6130ef54d61f792ac2d9b4c0d12b076d4c.jpg)  
(a) 1st Stage

![](images/51356aa5d3e706c0fc4258ac17577b82d138f1bdfb79fe9290b4d01a9778fb61.jpg)  
(b) 2nd Stage

![](images/ef2ae9860f030d7d98c913c867447b0fbaa4101b5643198538380048f7488921.jpg)  
Figure 1: Block Diagrams of GLINKX stages.  
(c) 3rd Stage

to be effective for heterophilous graphs (Lim et al., 2021). Thus, an approach encoding monophily has an advantage over methods designed specifically for homophilous and heterophilous graphs especially when varying levels of homophily can exist between different sub-regions in the same graph (see Section 3.3).

# 2.6 GRAPH MINIBATCHING & SHALLOW METHODS

In order to train a GCN-based model (or generally, whenever propagations based on the graph topology are involved in the model) on a large-network (that cannot fit in the GPU memory), one has to do minibatching through neighbor sampling. For large-scale networks, minibatching takes much longer than the full-batch training, which is one of the reasons that graph GCNs are not preferred in real-world settings (see e.g. (Jin et al., 2022b; Zhang et al., 2022a; Zheng et al., 2022a; Lim et al., 2021; Maurya et al., 2021; Rossi et al., 2020)).

For this reason, most methods that can scale on real-world settings are shallow. Shallow (or node-level) models are based on manipulating the node features  $X$  and the graph topology  $A$  in a way that propagations do not take place during training. Examples of these methods are LINKX, FSGNN (Maurya et al., 2021), and SIGN (Rossi et al., 2020). Such methods treat the input embeddings as tabular data and pass them through a feed-forward neural network (MLP) to produce the predictions. Thus, they avoid the need for neighborhood sampling, and instead rely on simple tabular minibatching.

# 3 METHOD

# 3.1 COMPONENTS

GLINKX is described in Alg. 1 and consists of three main components which are detailed as block diagrams in Fig. 1. Fig. 2 shows the GLINKX stages from Alg. 1 on a toy graph.

Positional Embeddings: We use PEs to provide our model information about the position of each node and hypothesize that PEs are an important piece of information in the context of large-scale node classification. PEs have been used to help discriminate isomorphic graph (sub)-structures (Kim et al., 2022; Dwivedi et al., 2021; Srinivasan & Ribeiro, 2019). This is useful for both homophily (Kim et al., 2022; Dwivedi et al., 2021) and heterophily (Lim et al., 2021) because isomorphic (sub)-structures can exist in both the settings. In the homophilous case, adding positional information can help distinguish nodes that have the same neighborhood but distinct position (Dwivedi et al., 2021; Morris et al., 2019; Xu et al., 2019), circumventing the to do higher-order propagations (Dwivedi et al., 2021; Li et al., 2019; Bresson & Laurent, 2017) which are prone to over-squashing (Alon & Yahav, 2021). In heterophily, structural similarity among nodes is important for classification as in the case of LINKX – where adjacency embedding can be thought of as a PE. However, in large

graphs, using adjacency embeddings or Laplacian eigenvectors (as methods such as (Kim et al., 2022) suggest) can be a computational bottleneck and may be infeasible.

In this work, we leverage knowledge graph embeddings (KGEs) to encode positional information about the nodes. Using KGEs has the following two benefits: Firstly, KGEs can be trained quickly for large graphs. This is because KGEs essentially compress the adjacency matrix into a fixed sized embedding and adjacency matrices have been shown to be effective in heterophilous cases. Further, KGEs are low-dimensional compared to the adjacency matrix (e.g.  $d_P \sim 10^2$ ) which allows for faster training and inference times. Secondly, KGEs can be pre-trained efficiently on such graphs (Lerer et al., 2019) and can be used off-the-shelf for other downstream tasks including node classification (see e.g. (El-Kishky et al., 2022)). So, in the 1st Stage of our methods in Alg. 1 (??) we train KGEs model on the available graph structure. In this work, we fix this positional encoding once they are pre-trained for downstream usage. One can fine-tune these along with learning (Dwivedi et al., 2021) in the downstream task but we leave this for future work.

Ego Embeddings: We obtain ego embeddings from the node features. Such embeddings have been used both in homophilous and heterophilous settings (e.g. (Lim et al., 2021; Zhu et al., 2020)). Node embeddings are useful for tasks where the graph structure provides little/no information for the task.

Monophilous Label Propagations: We now propose a novel monophily (refer Section 2.5) inspired label propagation which we refer to as Monophilous Label Propagation (MLaP). MLaP has the advantage that it can be used both for hetero(homo)philous graphs or in a scenario with varying levels of graph homophily (see Section 3.3) as it encodes monophily (Section 2.5).

In order to understand how MLaP encodes monophily, we consider the example in Fig. 2. In this example, we have 3 green nodes connected to a yellow node and two nodes of different color connected to the yellow node. Then, one way to encode monophily in Fig. 2(a) while predicting label for  $j_{\ell}, \ell \in [5]$ , is to get a distribution of labels of nodes connected to node  $i$  thus encoding its neighbors' distribution. The fact that there are more nodes with green color than other colors can be used by the model to make a prediction. But this information may not always be present or there may be few labelled nodes around node  $i$ . Consequently, we propose to use a model that predicts the label distribution of nodes connected to  $i$ . We use the node features  $(x_{i})$  and PE  $(p_i)$  of node  $i$  to build this model, since nodes that are connected to node  $i$  share similar labels and, thus, the features of node  $i$  must be predictive of its neighbors. So, in Fig. 2(a) we train a model to predict a distribution of  $i$ 's neighbors. Next, we provide  $j_{\ell}$  the learned distribution of  $i$ 's neighbors by propagating the learned distribution from  $i$  back to  $j_{\ell}$ . Eqs. (1) to (3) correspond to MLaP. We train a final model that leverages this information together with node features and PEs (Fig. 2(b)).

# 3.2 GLINKX

Following, we present the individual stages of GLINKX (see Alg. 1):

1st Stage: We train KGEs as PEs by using Pytorch-Biggraph and the DistMult method (Yang et al., 2014).

2nd Stage: First (2nd Stage in Alg. 1, Fig. 1(b), and Fig. 2(a)), for a node we want to learn the distribution of its neighbors. To achieve this, we propagate the labels from a node's neighbors, i.e. calculate

$$
\hat {\boldsymbol {y}} _ {i} = \frac {\sum_ {j \in V _ {\text {t r a i n}} : j \rightarrow i} \boldsymbol {y} _ {j}}{| \{j \in V _ {\text {t r a i n}} : j \rightarrow i \} |} \quad \forall i \in V _ {\text {t r a i n}}. \tag {1}
$$

We train a model that predicts the distribution of neighbors, which we denote with  $\tilde{\pmb{y}}_i$  using the ego features  $\{\pmb{x}_i\}_{i\in V_{train}}$  and the PEs  $\{\pmb{p}_i\}_{i\in V_{train}}$  and maximize the cross-entropy with treating  $\{\hat{\pmb{y}}_i\}_{i\in V_{train}}$  as ground truth labels, namely we maximize

$$
\mathcal {L} _ {\mathrm {C E}, 1} \left(\boldsymbol {\theta} _ {1}\right) = \sum_ {i \in V _ {\text {t r a i n}}} \sum_ {l \in [ c ]} \hat {\boldsymbol {y}} _ {i, l} \log \left(\tilde {\boldsymbol {y}} _ {i, l}\right), \tag {2}
$$

where  $\tilde{\pmb{y}}_i = f_1(\pmb{x}_i, \pmb{p}_i; \pmb{\theta}_1)$  and  $\pmb{\theta}_1 \in \Theta_1$  is a learnable parameter vector. Although in this paper we assume to be in the transductive setting, this step allows us to be inductive (see App. B). In

![](images/f75c1bf0fe3151dc896ed23d71a98258c87652785dc722777b59e3719ffc0ca8.jpg)  
(a) 2nd Stage

![](images/d2b473dbea8944d47cc8ae4bbf02547b733bf2e0926320f298b4fb8e0ba85f6f.jpg)  
Figure 2: Example. For node  $i$  we want to learn a model that takes  $i$ 's features  $\boldsymbol{x}_i \in \mathbb{R}^{d_X}$ , and PEs  $\boldsymbol{p}_i \in \mathbb{R}^{d_P}$  and predict a value  $\widetilde{\boldsymbol{y}}_i \in \mathbb{R}^c$  that matches the label distribution of it's neighbors neighbors  $\hat{\boldsymbol{y}}_i$  using a shallow model. Next, we want to propagate (outside the training loop) the (predicted) distribution of a node back to its neighbors and use it together with the ego features and the PEs to make a prediction about a node's own label. We propagate  $\widetilde{\boldsymbol{y}}_i$  to its neighbors  $j_1$  to  $j_5$ . For example, for  $j_1$ , we encode the propagated distribution estimate  $\widetilde{\boldsymbol{y}}_i$  from  $i$  to form  $\boldsymbol{y}_{j_1}'$ . We predict the label by using  $\boldsymbol{y}_{j_1}', \boldsymbol{x}_{j_1}, \boldsymbol{p}_{j_1}$ .  
(b) 3rd Stage

Section 3.4 we give a theoretical justification of this step, namely "why is it good to use a parametric model to predict the distribution of neighbors?".

3rd Stage: Then (3rd Stage in Alg. 1, Fig. 1(c), and Fig. 2(b)), we propagate the predicted soft-labels  $\widetilde{\pmb{y}}_i$  back to the original nodes, i.e. calculate

$$
\boldsymbol {y} _ {i} ^ {\prime} = \frac {\sum_ {j \in V : j \rightarrow i} \tilde {\boldsymbol {y}} _ {j}}{| \{j \in V : i \rightarrow j \} |} \quad \forall i \in V _ {\text {t r a i n}}, \tag {3}
$$

where the soft labels  $\{\tilde{y}_i\}_{i\in V_{train}}$  have been computed with the parameter  $\pmb{\theta}_1^*$  of the epoch with the best validation accuracy of Stage 2. Then, we make the final predictions  $\pmb{y}_{\mathrm{final},\mathrm{i}} = f_2(\pmb{x}_i,\pmb{p}_i,\pmb{y}_i';\pmb{\theta}_2)$  by combining the ego embeddings, PEs, and the (back)-propagated soft labels  $(\pmb{\theta}_2$  is a learnable parameter vector). We use the soft-labels  $\tilde{y}_i$  instead of the actual labels one-hot  $(y_i)$  in order to avoid label leakage, which hurts performance (see also (Shi et al., 2020) for a different way to combat label leakage). Finally, we maximize the cross-entropy with respect to a node's own labels,

$$
\mathcal {L} _ {\mathrm {C E}, 2} \left(\boldsymbol {\theta} _ {2}\right) = \sum_ {i \in V _ {\text {t r a i n}}} \sum_ {l \in [ c ]} \mathbb {I} \left\{y _ {i} = l \right\} \log \left(\boldsymbol {y} _ {\text {f i n a l}, 1, l}\right), \tag {4}
$$

Overall, Stage 2 corresponds to learning the neighbor distributions and Stage 3 uses these distributions to train a new model which predicts a node's own labels. In Section 3.4, we prove that such a procedure incurs lower error than using the features to predict a node's own labels.

Relationship to Label Propagation: From a label propagation perspective, one could argue that our method is related to the HITS algorithm of Kleinberg (1999). Besides, on a directed graph a potential solution would be to perform label propagation with the Gram matrix  $AA^T$  and use this information for as the propagated labels instead of  $\pmb{y}_i'$ . This, however, would not be inductive.

**Scalability:** GLINKX is highly scalable as it performs message passing a constant number of times by paying an  $O(mc)$  cost, where the dimensionality of classes  $c$  is usually small (compared to  $d_X$  that GCNs rely on). In both Stages 2 and 3 of Alg. 1 we train node-level MLPs which allow us to leverage i.i.d. (row-wise) minibatching, like tabular data, and thus our complexity is similar to other shallow methods (LINKX, FSGNN) (Lim et al., 2021; Maurya et al., 2021). This, combined with the propagations that happen outside of the training loops, circumvent the scalability issues of GCNs. For more details, refer App. A.1.

# 3.3 VARYING HOMOPHILY

Graphs with monophily experience homophily, heterophily, or both. For instance, in the yelp-chi dataset – where we classify a review as spam/non-spam (see Fig. 3) – we observe a case of monophily together with varying homophily. Specifically in this dataset, spam reviews are linked to non-spam

![](images/4d4663a289bb14319a3844144a2ab4885b864ea36b4cecf5a5c9f81b64d1e01f.jpg)  
Figure 3: Node Homophily Distribution and Class (or Edge-insensitive) Homophily (Lim et al., 2021).

reviews and non-spam reviews usually connect to other non-spam reviews, which makes the node homophily distribution bimodal. Here the 2nd-order similarity makes both the MLaP mechanism particularly effective and PEs, since the PEs can be used to distinguish nodes that have similar features but are located in different regions of the graph (homophilous/heterophilous region).

# 3.4 THEORETICAL ANALYSIS

Justification of Stage 2: In Stage 2, we train a parametric model to learn the distribution of a node's neighbors from the node features  $\xi_{i}^{3}$ . Arguably, such a distribution can be learned naively by counting the neighbors  $i$  that belong to each class. This motivates our first theoretical result. In a nutshell, we show that training a parametric model for learning the distribution of a node's neighbors (as we do in Stage 2) yields a lower error than the naive solution. Below we present the Thm. 1 (proof in App. F), for undirected graphs (the case of directed graphs is the same, but we omit it for simplicity of exposition):

Theorem 1. Let  $G([n], E)$  be an undirected graph of minimum degree  $K > c^2$  and let  $Q_i \in \Gamma_c$  be the likelihood, from the viewpoint of node  $i$ , of any node in its neighborhood  $\mathcal{N}(i)$  to be assigned to different classes for every node  $i \in [n]$ . The following two facts are true (under standard assumptions for SGD and the losses):

1. Let  $\widehat{\mathbf{Q}}_i$  be the sample average of  $\mathbf{Q}_i$ , i.e.  $\widehat{Q}_{i,j} = \frac{1}{|\mathcal{N}(i)|}\sum_{k\in \mathcal{N}(i)}\mathbb{I}\{y_k = j\}$ . Then, for every  $i\in [n]$ , we have that  $\max_{j\in [c]}\mathbb{E}[|Q_{i,j} - \widehat{Q}_{i,j}|]\leq \mathbb{E}[||\mathbf{Q}_i - \widehat{\mathbf{Q}}_i||_\infty ]\leq O\left(\sqrt{\frac{\log(Kc)}{K}}\right)$ .  
2. Let  $q(\cdot |\pmb{\xi}_i;\pmb{\theta})$  be a model parametrized by  $\pmb{\theta} \in \mathbb{R}^D$  that uses the features  $\pmb{\xi}_i$  of each node  $i$  to predict  $\pmb{Q}_i$ . We estimate the parameter  $\pmb{\theta}_1$  by running SGD for  $t = n$  steps to maximize  $\mathcal{L}(\pmb{\theta}) = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{c}Q_{i,j}\log q(j|\pmb{\xi}_i;\pmb{\theta})$ . Then, for every  $i \in [n]$ , we have that  $\max_{j\in [c]}\mathbb{E}[|q(j;\pmb{\xi}_i;\pmb{\theta}_1) - Q_{i,j}|]\leq O\left(\sqrt{\frac{\log n}{n}}\right)$ .

It is evident here that if the minimum degree  $K$  is much smaller than  $n$ , then the parametric model has lower error than the naive approach, namely  $\tilde{O}(n^{-1/2})$  compared to  $\tilde{O}(K^{-1/2})$ .

Justification of Stages 2 and 3: We now provide theoretical foundations for the 2 stage (stages 2 and 3) approach. Specifically, we argue that a two-stage procedure involving learning the distribution of a node's 2nd-hop neighbor distributions (we assume for simplicity, again, that the graph is undirected) first with a parametric model such as in Thm. 1, and then running a two-phase algorithm to learn a parametric model that predicts a node's own label, yields a lower error than naively training a shallow parametric model to learn a node's own labels. The first phase of the two-phase algorithm involves training the model first by minimizing the cross-entropy between the predictions and the 2nd-hop neighborhood distribution. Then the model trains a joint objective which uses the learned neighbor distributions and the true labels starting from the model learned in the previous phase. Our result follows (proof in App. F):

Theorem 2. Let  $G([n], E)$  be an undirected graph of minimum degree  $K > c^2$  and, let  $P_i$  be the likelihood of node  $i$  to be assigned to a different class, and let  $Q_i, q(\cdot | \xi_i; \theta_1)$  defined as in

Thm. 1. Let  $p(\cdot |\pmb{\xi}_i; \pmb{w})$  be a model parametrized by  $\pmb{w} \in \mathbb{R}^D$  that is used to predict the class assignments  $y_i \sim p(\cdot |\pmb{\xi}_i; \pmb{w})$ . Let  $\pmb{w}_*$  be the optimal parameter. The following are true (under standard assumptions for SGD and the losses):

1. The naive optimization scheme that runs SGD to maximize  $\mathcal{G}(\pmb{w}) = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{c}P_{i,j}\log p(j|\pmb{\xi}_i;\pmb{w})$  for  $n$  steps. Then  $\mathbb{E}[\mathcal{G}(\pmb{w}_{n+1}) - \mathcal{G}(\pmb{w}_*)] \leq O\left(\frac{\log n}{n}\right)$ .  
2. The two-phase optimization scheme that runs SGD to maximize  $\widehat{\mathcal{G}}(\boldsymbol{w}) = \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{c} \left( \frac{1}{|\mathcal{N}(i)|} \sum_{k \in \mathcal{N}(i)} q(j|\boldsymbol{\xi}_k; \boldsymbol{\theta}_1) \right) \log p(j|\boldsymbol{\xi}_i; \boldsymbol{w})$  for  $n_1$  steps, to estimate a solution  $\boldsymbol{w}'$  and then runs SGD on the objective  $\lambda \widehat{\mathcal{G}}(\boldsymbol{w}) + (1 - \lambda) \mathcal{G}(\boldsymbol{w})$  for  $n$  steps starting from  $\boldsymbol{w}'$ , achieves error  $\mathbb{E}[\mathcal{G}(\boldsymbol{w}_{n+1}) - \mathcal{G}(\boldsymbol{w}_*)] \leq O\left( \frac{\sqrt{\log n \log \log n}}{n} \right)$ .

We observe that the two-phase optimization scheme is able to reduce the error by a factor of  $\sqrt{\log n / \log\log n}$  highlighting the importance of using the distribution of the 2nd-hop neighbors of a node in order to predict its own label. Also, note that the above two-phase optimization scheme is different from the description of the method we gave in Alg. 1. The difference is that the distribution of a node's neighbors is embedded into the model in the case of Alg. 1, and the distribution of a node's neighbors is embedded into the loss function in Thm. 2 as a regularizer. In Alg. 1, we chose to incorporate this information in the model because using multiple losses harms scalability and makes training harder in practice. In the same spirit, the conception of GCNs (Kipf & Welling, 2016) replaces explicit regularization with the graph Laplacian with the graph topology into the model (see also (Hamilton et al., 2017; Yang et al., 2016)), and, similarly to GCNs, label information as a feature still acts a regularizer in our model.

# 3.5 COMPLEMENTARITY

Different components of GLINKX provide a complementary signal to components proposed in the GNN literature (Maurya et al., 2021; Zhang et al., 2022b; Rossi et al., 2020). One can combine GLINKX with existing architectures (e.g. feature propagations (Maurya et al., 2021; Rossi et al., 2020), label propagations (Zhang et al., 2022b)) for potential metric gains. For example, SIGN computes a series of  $r \in \mathbb{N}$  feature propagations  $[X, \Phi X, \Phi^2 X, \dots, \Phi^r X]$  where  $\Phi$  is a matrix (e.g. normalized adjacency or normalized Laplacian) as a preprocessing step. We can include this complementary signal, namely embed each of the propagated features and combine them in the final layer of the 3rd Stage, to GLINKX. Overall, even though in this paper we want to keep GLINKX simple to highlight its main components, we conjecture that adding more components to GLINKX would improve its performance on datasets with highly variable homophily (see Section 3.3).

# 4 EXPERIMENTS

# 4.1 COMPARISONS

We experiment with homophilous and heterophilous datasets (see Tab. 1 and App. D.3). We train KGEs with Pytorch-Biggraph (Lerer et al., 2019; Yang et al., 2014). For homophilous datasets we compare with vanilla GCN and GAT, FSGNN and Label Propagation (LP). For a fair comparison, we compare with one-layer GCN/GAT/FSGNN/LP since our method is one-hop. We also compare with higher-order (h.o.) GCN/GAT/FSGNN/LP with 2 and 3 layers. In the heterophilous case, we compare with  $\mathrm{LINKX^4}$  because it is scalable and is shown to work better than other baselines (e.g. H2GCN), as well as with FSGNN. Note that we do not compare GLINKX with other more complex methods because GLINKX is complementary to methods (see Section 3), and design principles from these methods can be incorporated into GLINKX. We use a ResNet module to combine the components from Stages 2 and 3 of our algorithm. Details about the hyperparameters we use are in App. C.

In the heterophilous datasets, GLINKX outperforms LINKX (except arxiv-year where we are within the confidence interval). Moreover, the performance gap between using KGEs and adjacency

Table 2: Ablation Study. We use the hyperparameters of the best run from Tab. 1 with KGEs.  
Table 1: Experimental results.  $(^{*}) =$  results from the OGB leaderboard.  

<table><tr><td></td><td colspan="2">Homophilous Datasets</td><td colspan="3">Heterophilous Datasets</td></tr><tr><td></td><td>PubMed</td><td>ogbn-arxiv</td><td>squirrel</td><td>yelp-chi</td><td>arxiv-year</td></tr><tr><td>n</td><td>19.7K</td><td>169.3K</td><td>5.2K</td><td>169.3K</td><td>45.9K</td></tr><tr><td>m</td><td>44.3K</td><td>1.16M</td><td>216.9K</td><td>7.73M</td><td>1.16M</td></tr><tr><td>Edge-insensitive homophily (Lim et al., 2021)</td><td>0.66</td><td>0.41</td><td>0.02</td><td>0.05</td><td>0.27</td></tr><tr><td>dX/c</td><td>500/27</td><td>128/40</td><td>2089/5</td><td>32/2</td><td>128/5</td></tr><tr><td>GLINKX w/KGEs</td><td>87.95±0.30</td><td>69.27±0.25</td><td>45.83±2.89</td><td>87.82±0.20</td><td>54.09±0.61</td></tr><tr><td>GLINKX w/ Adjacency</td><td>88.03±0.30</td><td>69.09±0.13</td><td>69.15±1.87</td><td>89.32±0.45</td><td>53.07±0.29</td></tr><tr><td>Label Propagation (1-hop)</td><td>83.02±0.35</td><td>69.59±0.00</td><td>32.22±1.45</td><td>85.98±0.28</td><td>43.71±0.22</td></tr><tr><td>LINKX (from (Lim et al., 2021))</td><td>87.86±0.77</td><td>67.32±0.24</td><td>61.81±1.80</td><td>85.86±0.40</td><td>56.00±1.34</td></tr><tr><td>LINKX (our runs)</td><td>87.55±0.37</td><td>63.91±0.18</td><td>61.46±1.60</td><td>88.25±0.24</td><td>53.78±0.06</td></tr><tr><td>GCN w/1 Layer</td><td>86.43±0.74</td><td>50.76±0.20</td><td></td><td>N/A</td><td></td></tr><tr><td>GAT w/1 Layer</td><td>86.41±0.53</td><td>54.42±0.10</td><td></td><td>N/A</td><td></td></tr><tr><td>FSGNN w/1 Layer</td><td>88.93±0.31</td><td>61.82±0.84</td><td>64.06±2.69</td><td>86.36±0.36</td><td>42.86±0.22</td></tr><tr><td>Higher-order GCN</td><td>86.29±0.46</td><td>71.18±0.27(*)</td><td></td><td>N/A</td><td></td></tr><tr><td>Higher-order GAT</td><td>86.64±0.40</td><td>73.66±0.11(*)</td><td></td><td>N/A</td><td></td></tr><tr><td>Higher-order FSGNN</td><td>89.37±0.49</td><td>69.26±0.36</td><td>68.04±2.19</td><td>86.33±0.30</td><td>44.89±0.29</td></tr><tr><td>Label Propagation (2-hop)</td><td>83.44±0.35</td><td>69.78±0.00</td><td>43.41±1.44</td><td>85.95±0.26</td><td>46.30±0.27</td></tr><tr><td>Label Prop. on II[A2-A-I≥0]</td><td>82.14±0.33</td><td>9.87±0.00</td><td>24.43±1.18</td><td>85.68±0.32</td><td>23.08±0.13</td></tr></table>

<table><tr><td></td><td>Ablation Type</td><td>Stages</td><td>All</td><td>Remove ego embeddings</td><td>Remove propagation</td><td>Remove PEs</td></tr><tr><td rowspan="2">Heterophilous</td><td>arxiv-year</td><td>All Stages</td><td>54.09 ±0.61</td><td>53.52 ±0.77</td><td>50.83 ±0.24</td><td>39.06 ±0.35</td></tr><tr><td>arxiv-year</td><td>3rd Stage</td><td>54.09 ±0.61</td><td>53.69 ±0.65</td><td>50.83 ±0.24</td><td>49.13 ±1.10</td></tr><tr><td rowspan="2">Homophilous</td><td>ogbn-arxiv</td><td>All Stages</td><td>69.27 ±0.25</td><td>61.26 ±0.33</td><td>62.70 ±0.34</td><td>65.64 ±0.18</td></tr><tr><td>ogbn-arxiv</td><td>3rd Stage</td><td>69.27 ±0.25</td><td>67.60 ±0.39</td><td>62.70 ±0.34</td><td>69.62 ±0.15</td></tr></table>

embeddings shrinks as the dataset grows. In the homophilous datasets GLINKX outperforms 1-layer GCN/GAT/LP/FSGNN and LINKX. In PubMed, GLINKX outperforms h.o. GCN/GAT and in arxiv-year GLINKX is very close to the performance of GCN/GAT.

Finally, we note that our method produces consistent results across regime shifts. In detail, in the heterophilous regime our method performs on par with LINKX, however when we shift to the homophilous regime, LINKX's performance drops, whereas our method's performance continues to be high. Similarly, while FSGNN performs similar to GLINKX on the homophilous datasets, we observe a big performance drop on the heterophilous datasets (see arxiv-year).

# 4.2 ABLATION STUDY

We ablate each component of Alg. 1 to see the performance contribution from each of the components. We use the hyperparameters of the best model from Tab. 1. We perform two types of ablations: (i) we remove each of the components from all stages of the training, and (ii) we remove the corresponding components only from the 3rd Stage. With the exception of removing the PEs from the 3rd Stage only on ogbn-arxiv, all components contribute to increasing performance on both datasets. Note that adding PEs on the 1st Stage does improve performance, suggesting the primary use case of PEs.

# 5 CONCLUSION

We present GLINKX, a scalable method for node classification in homophilous and heterophilous graphs that combines 3 components: (i) ego embeddings, (ii) PEs, and (iii) monophilous propagations. Our method is complementary to what other methods propose, since we can incorporate extra components such as feature propagations, label propagations, and attention to GLINKX. As future work, (i) GLINKX can be extended in heterogeneous graphs, (ii) use more expressive methods such as attention or Wasserstein barycenters (Cuturei & Doucet, 2014) for averaging the low-dimensional messages, and (iii) add complimentary signals.

# REFERENCES

Marjan Albooyeh, Rishab Goel, and Seyed Mehran Kazemi. Out-of-sample representation learning for knowledge graphs. In *Findings of the Association for Computational Linguistics: EMNLP* 2020, pp. 2657-2666, 2020.  
Uri Alon and Eran Yahav. On the bottleneck of graph neural networks and its practical implications. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=i800PhOCVH2.  
Kristen M Altenburger and Johan Ugander. Monophily in social networks introduces similarity among friends-of-friends. Nature human behaviour, 2(4):284-290, 2018a.  
Kristen M Altenburger and Johan Ugander. Node attribute prediction: An evaluation of within-versus across-network tasks. In NeurIPS Workshop on Relational Representation Learning, 2018b.  
Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.  
K. Bhatia, K. Dahiya, H. Jain, P. Kar, A. Mittal, Y. Prabhu, and M. Varma. The extreme classification repository: Multi-label datasets and code, 2016. URL http://manikvarma.org/downloads/XC/XMLRepository.html.  
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. Advances in neural information processing systems, 26, 2013.  
Xavier Bresson and Thomas Laurent. Residual gated graph convnets. arXiv preprint arXiv:1711.07553, 2017.  
Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. arXiv preprint arXiv:2006.07988, 2020.  
Marco Cuturi and Arnaud Doucet. Fast computation of wasserstein barycenters. In Eric P. Xing and Tony Jebara (eds.), Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pp. 685-693, Beijing, China, 22-24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/cuturi14.html.  
Francesco Di Giovanni, James Rowbottom, Benjamin P Chamberlain, Thomas Markovich, and Michael M Bronstein. Graph neural networks as gradient flows. arXiv preprint arXiv:2206.10991, 2022.  
Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, and Philip S Yu. Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pp. 315-324, 2020.  
Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. arXiv preprint arXiv:2110.07875, 2021.  
Ahmed El-Kishky, Thomas Markovich, Serim Park, Chetan Verma, Baekjin Kim, Ramy Eskander, Yury Malkov, Frank Portman, Sofia Samaniego, Ying Xiao, et al. Twhin: Embedding the twitter heterogeneous information network for personalized recommendation. arXiv preprint arXiv:2202.05387, 2022.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.  
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems, 33:22118-22133, 2020.

Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin R Benson. Combining label propagation and simple models out-performs graph neural networks. arXiv preprint arXiv:2010.13993, 2020.  
Di Jin, Rui Wang, Meng Ge, Dongxiao He, Xiang Li, Wei Lin, and Weixiong Zhang. Raw-gnn: Random walk aggregation based graph neural network. arXiv preprint arXiv:2206.13953, 2022a.  
Wei Jin, Lingxiao Zhao, Shichang Zhang, Yozen Liu, Jiliang Tang, and Neil Shah. Graph condensation for graph neural networks. In International Conference on Learning Representations, 2022b. URL https://openreview.net/forum?id=WLEx3Jo4QaB.  
Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, and Seunghoon Hong. Pure transformers are powerful graph learners. arXiv preprint arXiv:2207.02505, 2022.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Jon M Kleinberg. Authoritative sources in a hyperlinked environment. Journal of the ACM (JACM), 46(5):604-632, 1999.  
Adam Lerer, Ledell Wu, Jiajun Shen, Timothee Lacroix, Luca Wehrstedt, Abhijit Bose, and Alex Peysakhovich. Pytorch-biggraph: A large scale graph embedding system. Proceedings of Machine Learning and Systems, 1:120-131, 2019.  
Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deep GCs: Can GCs go as deep as cnns? In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9267-9276, 2019.  
Derek Lim, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, and Ser Nam Lim. Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. Advances in Neural Information Processing Systems, 34:20887-20902, 2021.  
Sitao Luan, Chenqing Hua, Qincheng Lu, Jiaqi Zhu, Mingde Zhao, Shuyuan Zhang, Xiao-Wen Chang, and Doina Precup. Is heterophily a real nightmare for graph neural networks to do node classification? arXiv preprint arXiv:2109.05641, 2021.  
Sunil Kumar Maurya, Xin Liu, and Tsuyoshi Murata. Improving graph neural networks with simple architecture design. arXiv preprint arXiv:2105.07634, 2021.  
Miller McPherson, Lynn Smith-Lovin, and James M Cook. Birds of a feather: Homophily in social networks. Annual review of sociology, pp. 415-444, 2001.  
Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 4602-4609, 2019.  
Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. Geom-gcn: Geometric graph convolutional networks. arXiv preprint arXiv:2002.05287, 2020.  
Everett M Rogers, Arvind Singhal, and Margaret M Quinlan. Diffusion of innovations. In An integrated approach to communication theory and research, pp. 432-448. Routledge, 2014.  
Emanuele Rossi, Fabrizio Frasca, Ben Chamberlain, Davide Eynard, Michael Bronstein, and Federico Monti. Sign: Scalable inception graph neural networks. arXiv preprint arXiv:2004.11198, 7:15, 2020.  
Benedek Rozemberczki, Carl Allen, and Rik Sarkar. Multi-scale attributed node embedding. Journal of Complex Networks, 9(2):cnab014, 2021.  
Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93-93, 2008.

Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjin Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification. arXiv preprint arXiv:2009.03509, 2020.  
Arnab Sinha, Zhihong Shen, Yang Song, Hao Ma, Darrin Eide, Bo-June Hsu, and Kuansan Wang. An overview of microsoft academic service (mas) and applications. In Proceedings of the 24th international conference on world wide web, pp. 243-246, 2015.  
Balasubramaniam Srinivasan and Bruno Ribeiro. On the equivalence between positional node embeddings and structural graph representations. arXiv preprint arXiv:1910.00452, 2019.  
Chuxiong Sun, Hongming Gu, and Jie Hu. Scalable and adaptive graph neural networks with self-label-enhanced training. arXiv preprint arXiv:2104.09376, 2021.  
Petar Velivcković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=ryGs6iA5Km.  
Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. arXiv preprint arXiv:1412.6575, 2014.  
Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In International conference on machine learning, pp. 40-48. PMLR, 2016.  
Shichang Zhang, Yozen Liu, Yizhou Sun, and Neil Shah. Graph-less neural networks: Teaching old MLPs new tricks via distillation. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?id=4p6_5HBWPCw.  
Wentao Zhang, Ziqi Yin, Zeang Sheng, Yang Li, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, and Bin Cui. Graph attention multi-layer perceptron. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022b.  
Wenqing Zheng, Edward W Huang, Nikhil Rao, Sumeet Katariya, Zhangyang Wang, and Karthik Subbian. Cold brew: Distilling graph node representations with incomplete or missing neighborhoods. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?id=1ugNpm7W6E.  
Xin Zheng, Yixin Liu, Shirui Pan, Miao Zhang, Di Jin, and Philip S Yu. Graph neural networks for graphs with heterophily: A survey. arXiv preprint arXiv:2202.07082, 2022b.  
Zhiqiang Zhong, Sergey Ivanov, and Jun Pang. Simplifying node classification on heterophilous graphs with compatible label propagation. arXiv preprint arXiv:2205.09389, 2022.  
Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. Advances in Neural Information Processing Systems, 33:7793-7804, 2020.  
Jiong Zhu, Ryan A Rossi, Anup Rao, Tung Mai, Nedim Lipka, Nesreen K Ahmed, and Danai Koutra. Graph neural networks with heterophily. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11168-11176, 2021.