# GREED: A Neural Framework for Learning Graph Distance Functions

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Among various distance functions for graphs, graph and subgraph edit distances (GED and SED respectively) are two of the most popular and expressive measures. Unfortunately, exact computations for both are NP-hard. To overcome this computational bottleneck, neural approaches to learn and predict edit distance in polynomial time have received much interest. While considerable progress has been made, there exist limitations that need to be addressed. First, the efficacy of an approximate distance function lies not only in its approximation accuracy, but also in the preservation of its properties. To elaborate, although GED is a metric, its neural approximations do not provide such a guarantee. This prohibits their usage in higher order tasks that rely on metric distance functions, such as clustering or indexing. Second, several existing frameworks for GED do not extend to SED due to SED being asymmetric. In this work, we design a novel siamese graph neural network called GreED, which through a carefully crafted inductive bias, learns GED and SED in a property-preserving manner. Through extensive experiments across 10 real graph datasets containing up to 7 million edges, we establish that GreED is not only more accurate than the state of the art, but also up to 3 orders of magnitude faster. Even more significantly, due to preserving the triangle inequality, the generated embeddings are indexable and consequently, even in a CPU-only environment, GreED is up to 50 times faster than GPU-powered baselines.

# 1 Introduction and Related Work

A distance function on any dataset, including graphs, is a fundamental operator. Among several distance measures on graphs, edit distance is one of the most powerful and popular mechanisms [28, 42, 40, 19]. Edit distance can be posed in two forms: graph edit distance (GED) and subgraph edit distance (SED). Given two graphs  $\mathcal{G}_1$  and  $\mathcal{G}_2$ ,  $\mathrm{GED}(\mathcal{G}_1,\mathcal{G}_2)$  returns the minimum cost of edits needed to convert  $\mathcal{G}_1$  to  $\mathcal{G}_2$ , i.e., for  $\mathcal{G}_1$  to become isomorphic to  $\mathcal{G}_2$ . An edit can be the addition or deletion of edges and nodes, or the replacement of edge or node labels, with an associated cost. In  $\mathrm{SED}(\mathcal{G}_1,\mathcal{G}_2)$ , the goal is to identify the minimum cost of edits so that  $\mathcal{G}_1$  is a subgraph (subgraph isomorphic) of  $\mathcal{G}_2$ . For examples, see Fig. 1.

GED is typically restricted to graph databases containing small graphs to facilitate distance computation with queries of similar sizes. As an example, given a repository of molecules, and a query molecule, we may want to identify the closest molecule in the repository that is similar to the query. SED, on the other hand, is useful when the database has large graphs and the query is a comparatively smaller graph. As examples, subgraph queries are used on knowledge graphs for analogy reasoning [16]. In PPI and chemical compounds, SED is of central importance to identify functional motifs and binding pockets [33, 19, 15]. Unfortunately, both GED and SED are NP-hard to compute [40, 19]. To mitigate this computational bottleneck, several heuristics [9, 12] and index structures [19, 40, 28, 42] have been proposed. Recently, graph neural networks have been shown to be effective in learning and predicting GED [2, 36, 27, 3, 41, 37, 13, 1]. The basic goal in all these algorithms is to learn a neural model from a training set of graph pairs and their distances, such that, at inference time, given an unseen graph pair, we are able to predict its distance accurately. In the

![](images/e96e760891f8b1bc74d1c9fe0982ac2474abf3d87d95ab835da9cb005a023767.jpg)

![](images/c4fb9fc4bd6589942da99dad40a397b6f60a327a553301fa33740a54b8f998f7.jpg)

![](images/80b8425969ea7dd77d5744dd2d4be21981dca9c140c0d26ed1c4ae2dbb53b11f.jpg)

![](images/2b447249869fe4bc86f20f936d102931c843961240f06284ec253208d38921e5.jpg)

![](images/353839624dbedb267eb2685c4b139843741e76bdc21d7c8a570fb73b66d4610b.jpg)

![](images/4e56ab76520073380b00f0457c6b69b147d2839d2e2076c5e7ab2d1e1920b4a9.jpg)

![](images/0b073b5fd188b2c2e580e34b3de4d649608ce31becc90d0903bceda405ece6f9.jpg)  
Figure 1: A sample set of graphs  $(g_{1} - g_{5})$ , their corresponding GED and SED matrices and an example of a graph mapping from  $g_{1}$  to  $g_{2}$ . The dashed nodes and edges in the mapping represent dummy nodes and edges. The red arrows denote either insertion or change of label.

![](images/0382e48f984b37e8d1d3a67df062a75a25a8be7ed176530b57bcfa4577d14245.jpg)

domain of subgraphs, NEUROMATCH [32] generates embeddings to detect subgraph isomorphism, wherein if  $\mathcal{G}_1$  is a subgraph of  $\mathcal{G}_2$ , then it is embedded on the lower right space of  $\mathcal{G}_2$ . NSC [35] generates subgraph level embeddings that can count the number of subgraph instances of a query graph on a target graph. While the progress made is impressive, there is scope to do more.

- Preservation of theoretical properties: GED is a metric distance function. While SED is not metric due to being asymmetric, it satisfies the triangle inequality, non-negativity, and subgraph-identity. Several higher-order tasks such as clustering and indexing rely on such properties [18, 14, 20, 34, 11]. Existing neural approaches do not preserve these properties, which limits their usability for these higher-order tasks.  
- Indexable embeddings: Given graph pair  $\mathcal{G}_1$  and  $\mathcal{G}_2$ , neural approaches first embed them into a feature space. Next, they compute a distance on these feature vectors, which is an approximation of the distance in the original graph space. The literature on indexing range and  $k$ -NN queries over feature vectors is rich [17, 23, 11]. Index structures typically allow sub-linear computation costs with respect to the database size. Unfortunately, none of the neural approaches generate indexable feature vectors since they perform pair-dependent computations. Specifically, the neural computations on  $\mathcal{G}_1$  depend on both  $\mathcal{G}_1$  and  $\mathcal{G}_2$  (and same for  $\mathcal{G}_2$ ). Consequently, the computations can only be done at query-time and thereby negating the possibility of indexing pre-computed feature space embeddings.  
- Modeling SED: Existing neural approaches to learning GED cannot easily be adapted to learn SED. While GED is symmetric, SED is not. Several neural architectures for GED have the assumption of symmetry at its core and hence modeling SED is non-trivial [2, 3, 27].  
- Exponential Search Space: Computing  $\operatorname{SED}(\mathcal{G}_1, \mathcal{G}_2)$  conceptually requires us to compare the query graph  $\mathcal{G}_1$  with the exponentially many subgraphs of the target graph  $\mathcal{G}_2$ . Therefore, it is imperative that the model has an efficient and effective mechanism to prune the search space without compromising on the prediction accuracy.

In this work, we address the above limitations through the following contributions.

- Novel neural architecture: We address the above mentioned challenges through a novel architecture called GreED: GRaph Embeddings for Edit Distances. GreED utilizes a siamese graph isomorphism network [38] to embed graphs in a pair-independent fashion. A simple, but theoretically well-characterized, function on this embedding space predicts the SED and GED. The carefully crafted prediction function serves as an inductive bias for the model, which, in addition to enabling high generalization accuracy, preserves the metric property of GED and the triangle inequality of SED in the embedding space.  
- Indexable embeddings: Owing to pair-independent embeddings and preservation of the triangle inequality over the embedding space for both SED and GED, GreED can exploit the rich literature on index structures [17, 23, 11] to boost efficiency.  
- Accurate, Fast and Scalable: Extensive experiments on real graph datasets containing up to a million nodes establish that GreED is more accurate in both GED and SED when compared to the state of the art algorithms and is more than 3 orders of magnitude faster in range and  $k$ -NN queries. Furthermore, owing to indexable embeddings, even in a CPU-only environment, GreED is up to 50 times faster than the closest baseline run on a GPU.

# 2 Preliminaries and Problem Formulation

We denote a labeled undirected graph as  $\mathcal{G} = (\mathcal{V},\mathcal{E},\mathcal{L})$  where  $\mathcal{V}$  is the node set,  $\mathcal{E}$  is the edge set and  $\mathcal{L}:\mathcal{V}\cup \mathcal{E}\to \Sigma$  is the labeling function over nodes and edges.  $\Sigma$  is the universe of all labels and contains a special empty label  $\epsilon$ .  $\mathcal{L}(v)$  and  $\mathcal{L}(e)$  denote the labels of node  $v$  and edge  $e$  respectively.  $\mathcal{G}_1\subseteq \mathcal{G}_2$  denotes that  $\mathcal{G}_1$  is a subgraph of  $\mathcal{G}_2$ . The computation of GED relies on a graph mapping.

Definition 1 (Graph Mapping) Given two graphs  $\mathcal{G}_1$  and  $\mathcal{G}_2$ , let  $\tilde{\mathcal{G}}_1 = (\tilde{\mathcal{V}}_1, \tilde{\mathcal{E}}_1, \tilde{\mathcal{L}}_1)$  and  $\tilde{\mathcal{G}}_2 = (\tilde{\mathcal{V}}_2, \tilde{\mathcal{E}}_2, \tilde{\mathcal{L}}_2)$  be obtained by adding dummy nodes and edges (labeled with  $\epsilon$ ) to  $\mathcal{G}_1$  and  $\mathcal{G}_2$  respectively,

![](images/bd1a6b799a9a7f38ebea49d80c763f038ef5f847c538a0dc0f94b5d854e0401c.jpg)  
(a) Siamese architecture of GreED  
Figure 2: The architecture of GreED.

![](images/10fa40df31a3fdd979cd4b7182f47151b32f07e2e1532fbd490c1e63d991de06.jpg)  
(b) The GNN component in GreED.

such that  $|\mathcal{V}_1| = |\mathcal{V}_2|$  and  $|\mathcal{E}_1| = |\mathcal{E}_2|$ . A node mapping between  $\mathcal{G}_1$  and  $\mathcal{G}_2$  is a bijection  $\pi : \tilde{\mathcal{G}}_1 \to \tilde{\mathcal{G}}_2$  where  $(\pmb{i}) \forall v \in \tilde{\mathcal{V}}_1, \pi(v) \in \tilde{\mathcal{V}}_2$  and at least one of  $v$  and  $\pi(v)$  is not a dummy;  $(\pmb{ii}) \forall e = (v_1, v_2) \in \tilde{\mathcal{E}}_1, \pi(e) = (\pi(v_1), (\pi(v_2))) \in \tilde{\mathcal{E}}_2$  and at least one of  $e$  and  $\pi(e)$  is not a dummy.

Example 1 Fig. 1 shows a graph mapping. Edge mappings can be trivially inferred.

Definition 2 (Graph Edit Distance (GED) under mapping  $\pi$ ) GED between  $\mathcal{G}_1$  and  $\mathcal{G}_2$  under  $\pi$  is

$$
\operatorname {G E D} _ {\pi} \left(\mathcal {G} _ {1}, \mathcal {G} _ {2}\right) = \sum_ {v \in \tilde {V} _ {1}} d \left(\mathcal {L} (v), \mathcal {L} (\pi (v))\right) + \sum_ {e \in \tilde {E} _ {1}} d \left(\mathcal {L} (e), \mathcal {L} (\pi (e))\right) \tag {1}
$$

where  $d:\Sigma \times \Sigma \to \mathbb{R}_0^+$  is a distance function over the label set.  $d(\ell_1,\ell_2)$  models an insertion if  $\ell_1 = \epsilon$ , deletion if  $\ell_2 = \epsilon$  and replacement if  $\ell_1\neq \ell_2$  and neither  $\ell_{1}$  nor  $\ell_{2}$  is a dummy.

We assume  $d$  to be a binary function, where  $d(\ell_1,\ell_2) = 1$  if  $\ell_1\neq \ell_2$  , otherwise, 0.

Definition 3 (Graph Edit Distance (GED)) GED is the minimum distance under all mappings.

$$
\operatorname {G E D} \left(\mathcal {G} _ {1}, \mathcal {G} _ {2}\right) = \min  _ {\forall \pi \in \Phi \left(\mathcal {G} _ {1}, \mathcal {G} _ {2}\right)} \operatorname {G E D} _ {\pi} \left(\mathcal {G} _ {1}, \mathcal {G} _ {2}\right) \tag {2}
$$

$\Phi (\mathcal{G}_1,\mathcal{G}_2)$  denotes the set of all possible node maps from  $\mathcal{G}_1$  to  $\mathcal{G}_2$

Definition 4 (Subgraph Edit Distance (SED)) SED is the minimum GED over all subgraphs of  $\mathcal{G}_2$ .

$$
\operatorname {S E D} \left(\mathcal {G} _ {1}, \mathcal {G} _ {2}\right) = \min  _ {\mathcal {S} \subseteq \mathcal {G} _ {2}} \operatorname {G E D} \left(\mathcal {G} _ {1}, \mathcal {S}\right) \tag {3}
$$

Problem 1 (Learning GED/SED) Given a training set of tuples of the form  $\langle \mathcal{G}_1, \mathcal{G}_2, \mathrm{GED}(\mathcal{G}_1, \mathcal{G}_2) \rangle$  (or  $\mathcal{G}_1, \mathcal{G}_2, \mathrm{SED}(\mathcal{G}_1, \mathcal{G}_2) \rangle$ ), learn a neural model to predict  $\mathrm{GED}(\mathcal{Q}_1, \mathcal{Q}_2)$  (or  $\mathrm{SED}(\mathcal{Q}_1, \mathcal{Q}_2)$ ) on unseen graphs  $\mathcal{Q}_1$  and  $\mathcal{Q}_2$ .

# 2.1 Properties of GED and SED

Observation 1 (i)  $\operatorname{GED}(\mathcal{G}_1, \mathcal{G}_2) \geq 0$ , (ii)  $\operatorname{SED}(\mathcal{G}_1, \mathcal{G}_2) \geq 0$ .

Observation 2 (i)  $\operatorname{GED}(\mathcal{G}_1, \mathcal{G}_2) = 0$  iff  $\mathcal{G}_1$  is isomorphic to  $\mathcal{G}_2$ , (ii)  $\operatorname{SED}(\mathcal{G}_1, \mathcal{G}_2) = 0$  iff  $\mathcal{G}_1$  is subgraph isomorphic to  $\mathcal{G}_2$ .

Theorem 1 Let  $\widehat{d}:\Sigma \times \Sigma \to \mathbb{R}_0^+$  be a distance function over  $\Sigma$ , where (i)  $\widehat{d}(\ell_1,\ell_2) = 0$  if  $\ell_1 = \epsilon$ , and (ii)  $\widehat{d}(\ell_1,\ell_2) = d(\ell_1,\ell_2)$  otherwise; the following holds:  $\operatorname{SED}(\mathcal{G}_1,\mathcal{G}_2) = \widehat{\operatorname{GED}}(\mathcal{G}_1,\mathcal{G}_2)$ , where  $\widehat{\operatorname{GED}}$  denotes  $\operatorname{GED}$  with  $\widehat{d}$  as the label set distance function. In simple words, the  $\operatorname{SED}$  between two graphs is equivalent to  $\operatorname{GED}$  with a label set distance function where we ignore insertion costs.

PROOF. See App. A.1.

Observation 3 GED satisfies the triangle inequality if the distance function  $d$  over label set  $\Sigma$  satisfies the triangle inequality [19]. As defined in the paragraph following Def. 2,  $d$  satisfies the triangle inequality [19]. Furthermore, it is trivial to see that GED is symmetric, non-negative and satisfies identity as long as  $d$  satisfies the analogues. Hence, GED with distance function  $d$  is metric.

Theorem 2 SED is not metric due to violating the properties of symmetry and identity. However, it satisfies the triangle inequality, i.e.,  $\operatorname{SED}(\mathcal{G}_1,\mathcal{G}_3)\leq \operatorname{SED}(\mathcal{G}_1,\mathcal{G}_2) + \operatorname{SED}(\mathcal{G}_2,\mathcal{G}_3)$ .

PROOF: See App. A.2 for details.

Observation 4 Computing GED and SED is NP-hard [40].

Hereon, we use GED as the illustrative distance function being modeled. The architecture trivially extends to SED. The specific places that need separate treatment will be discussed explicitly.

# 3 GreED: The Proposed Architecture

Fig. 2 presents the architecture of GreED. The input to our learning framework is a pair of graphs  $\mathcal{G}_{\mathcal{Q}}$  (query),  $\mathcal{G}_{\mathcal{T}}$  (target) along with the supervision data  $\mathrm{GED}(\mathcal{G}_Q,\mathcal{G}_T)$  (or  $\mathrm{SED}(\mathcal{G}_Q,\mathcal{G}_T)$ ). Our objective is to train a model that can predict GED on unseen query and target graphs. The design of our model must be cognizant of the fact that computing GED is NP-hard and high quality training data is scarce. Thus, we use a Siamese architecture [10], where there are two networks with shared parameters applied to two inputs independently to compute representations.

# 3.1 Siamese Graph Neural Network

As depicted in Fig. 2a, we use a siamese graph neural network (GNN) with shared parameters to embed both  $\mathcal{G}_{\mathcal{Q}}$  and  $\mathcal{G}_{\mathcal{T}}$ . While one could use two different GNN models for the query and the target, this design increases the model parameters and consequently, the training time. Furthermore, an architecture with higher number of parameters also requires larger amount of training data, which is difficult due to GED being NP-hard.

Fig. 2b focuses on the GNN component of GreED. We next discuss each of its individual components.

Pre-MLP: The primary task of the Pre-MLP is to learn representations for the node labels (or features). Towards that end, let  $\mathbf{x}_v$  denote the initial feature set of node  $v$ . The MLP learns a hidden representation  $\pmb{\mu}_v^G = \mathbf{MLP}(\mathbf{x}_v)$ . In our implementation,  $\mathbf{x}_v$  is a one-hot encoding of the categorical node labels. We do not explicitly model edge labels in our experiments. GreED can easily be extended to edge labels by using GINE [21] instead of GIN.

Graph Isomorphism Network (GIN): GIN [38] consumes the information from the Pre-MLP to learn hidden representations that encode both the graph structure as well as the node feature information. GIN is as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test [24] in distinguishing graph structures. Since our goal is to accurately characterize graph topology and learn similarity, GIN emerges as the natural choice. GIN develops its expressive power by using an injective aggregation function. Specifically, in the initial layer, each node  $v$  in graph  $\mathcal{G}$  is characterized by the representation learned by the MLP, i.e.,  $\mathbf{h}_{v,0}^{\mathcal{G}} = \boldsymbol{\mu}_v^{\mathcal{G}}$ . Subsequently, in each hidden layer  $i$ , we learn an embedding through the following transformation.

$$
\mathbf {h} _ {v, i} ^ {\mathcal {G}} = \operatorname {M L P} \left((1 + \epsilon^ {i}) \cdot \mathbf {h} _ {v, i - 1} ^ {\mathcal {G}} + \sum_ {u \in \mathcal {N} _ {\mathcal {G}} (v)} \mathbf {h} _ {u, i - 1} ^ {\mathcal {G}}\right) \tag {4}
$$

Here,  $\epsilon^i$  is a layer-specific learnable parameter,  $\mathcal{N}_{\mathcal{G}}(v)$  is one-hop neighbourhood of the node  $v$ , and  $\mathbf{h}_{v,0}^{\mathcal{G}} = \boldsymbol{\mu}_v^{\mathcal{G}}$ . The  $k$ -th layer embedding is  $\mathbf{h}_{v,k}^{\mathcal{G}}$ , where  $k$  is final hidden layer.

Concatenation, Pool and Post-MLP: Intuitively,  $\mathbf{h}_{v,i}^{\mathcal{G}}$  captures a feature-space representation of the  $i$ -hop neighborhood of  $v$ . Typically, GNNs operate on node or edge level predictive tasks, such as node classification or link prediction, and hence, the node representations are passed through an MLP for the final prediction task. In our problem, we need to capture a graph level representation. Furthermore, the representation should be rich enough to also capture the various subgraphs within the input graph so that SED can be predicted accurately. To fulfil these requirements, we first concatenate the representation of a node across all hidden layers, i.e., the final node embedding is  $\mathbf{z}_v^{\mathcal{G}} = \mathrm{CONCAT}\left(\mathbf{h}_{v,i}^{\mathcal{G}}, \forall i \in \{1,2,\dots,k\}\right)$ . This allows us to capture a multi-granular view of the subgraphs centered on  $v$  at different radii in the range  $[1,k]$ . Next, to construct the graph-level representation, we perform a sum-pool, which adds the node representations to give a single vector. This information is then fed to the Post-MLP to enable post-processing. Mathematically:

$$
\mathbf {Z} _ {\mathcal {G}} = \operatorname {M L P} \left(\mathbf {z} ^ {\mathcal {G}}\right) = \operatorname {M L P} \left(\sum_ {v \in \mathcal {V}} \mathbf {z} _ {v} ^ {\mathcal {G}}\right) \tag {5}
$$

GED and SED Prediction: The final task is to predict the GED (and SED) as a function of query graph embedding  $\mathbf{Z}_{\mathcal{G}_Q}$  and target graph embedding  $\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}}$ . The natural choice would be to feed these embeddings into another MLP to learn  $\mathrm{GED}(\mathbf{Z}_{\mathcal{G}_Q},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}})$ . This MLP can then be trained jointly with the graph embedding model in an end-to-end fashion. However, an MLP prediction does not have any theoretical guarantees with respect to the preservation of metric properties of GED and the triangle inequality of SED. We, therefore, focus on learning prediction functions  $\mathcal{F}_g(\mathbf{Z}_{\mathcal{G}_Q},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}})$  and  $\mathcal{F}_s(\mathbf{Z}_{\mathcal{G}_Q},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}})$  for GED and SED respectively, such that they are accurate and respects the desirable

properties from the original graph space. As we will empirically substantiate in  $\S 4.5$ , the inductive bias injected through the prediction functions also lead to more effective learning over low volumes of training data than an MLP.

# 3.1.1 GED

We require the following four properties to ensure that the prediction is also a metric.

$$
\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\varrho}}, \mathbf {Z} _ {\mathcal {G} _ {\tau}}\right) \geq 0 \tag {6}
$$

$$
\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}\right) = 0 \Longleftrightarrow \forall i: \mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}} [ i ] = \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} [ i ] \tag {7}
$$

$$
\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} \right) = \mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}\right) \tag {8}
$$

$$
\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\varrho}}, \mathbf {Z} _ {\mathcal {G} _ {\tau}}\right) \leq \mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\varrho}}, \mathbf {Z} _ {\mathcal {G} ^ {\prime}}\right) + \mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} ^ {\prime}}, \mathbf {Z} _ {\mathcal {G} _ {\tau}}\right) \tag {9}
$$

To achieve this, we establish an important connection of metrics on vector spaces to norms. Every norm  $\| .\|$  gives a metric  $(\mathbf{x},\mathbf{y})\mapsto \| \mathbf{x} - \mathbf{y}\|$ . Moreover for a metric, there exists a norm  $\| .\|$  such that the metric can be expressed as  $(\mathbf{x},\mathbf{y})\mapsto \| \mathbf{x} - \mathbf{y}\|$ , iff the metric is translation invariant and homogeneous. Thus, we add these properties to the desiderata for  $\mathcal{F}_g$ :

$$
\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}} + \mathbf {k}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} + \mathbf {k}\right) = \mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}\right), \forall \mathbf {k} \in \mathbb {R} ^ {d} \tag {10}
$$

$$
\mathcal {F} _ {g} \left(r \mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, r \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}\right) = | r | \mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}\right), \forall r \in \mathbb {R} \tag {11}
$$

Armed with these observations, we define the class of functions that may be used for  $\mathcal{F}_g$ .

Observation 5  $\mathcal{F}_g$  may be defined as any function  $(x,y)\mapsto \| x - y\|$  for some norm  $\| .\|$  over the vector space  $\mathbb{R}^d$  such that  $\mathcal{F}_g$  satisfies Eqs. 10 - 11.

The  $L_{p}$  norm satisfies Obs. 5. Hence, we define  $\mathcal{F}_g$  as:

$$
\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}\right) = \left\| \mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}} - \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} \right\| _ {p} \tag {12}
$$

In our implementation we use the  $L_{2}$  norm. Finally, the parameters of the entire model are learned by minimizing the mean squared error (here  $\mathbb{T}$  is the training set).

$$
\mathcal {L} = \frac {1}{| \mathbb {T} |} \sum_ {\forall \langle \mathcal {G} _ {\mathcal {Q}}, \mathcal {G} _ {\mathcal {T}} \rangle \in \mathbb {T}} \left(\mathcal {F} _ {g} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}}\right) - \operatorname {G E D} \left(\mathcal {G} _ {\mathcal {Q}}, \mathcal {G} _ {\mathcal {T}}\right)\right) ^ {2} \tag {13}
$$

Intuition: Regardless of the graph representations generated by our model,  $\mathcal{F}_g$  ensures that the predicted distance is a metric. One the other hand, by training the model to produce embeddings  $\mathbf{Z}_{\mathcal{G}_{\mathcal{Q}}}$  and  $\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}}$  such that  $\mathcal{F}_g(\mathbf{Z}_{\mathcal{G}_{\mathcal{Q}}},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}})\approx \mathrm{GED}(\mathbf{Z}_{\mathcal{G}_{\mathcal{Q}}},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}})$ , we enforce a rich structure on the embedding space such that  $\mathcal{F}_g$  is also accurate. Thus,  $\mathcal{F}_g$  injects an inductive bias satisfying the dual needs of accuracy and preservation of original space properties.

# 3.1.2 SED

SED satisfies non-negativity and triangle inequality. Following a similar reasoning as above, we define  $\mathcal{F}_s$  as follows:

$$
\left. \right. \mathcal {F} _ {s} \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}}, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} \right) = \left\| R e L U \left(\mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}} - \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} \right)\right\| _ {2} = \left\| \max  \left\{0, \mathbf {Z} _ {\mathcal {G} _ {\mathcal {Q}}} - \mathbf {Z} _ {\mathcal {G} _ {\mathcal {T}}} \right\}\right\| _ {2} \tag {14}
$$

Intuitively, for those co-ordinates where the value of  $\mathbf{Z}_{\mathcal{G}_Q}$  is greater than  $\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}}$ , a distance penalty is accounted by  $\mathcal{F}_s$  in terms of how much those values differ; otherwise  $\mathcal{F}_s$  considers 0. This follows the intuition that the SED accounts for those features of  $\mathcal{G}_Q$  that are not in  $\mathcal{G}_{\mathcal{T}}$ . Moreover, consistent with SED, the additional features in  $\mathcal{G}_{\mathcal{T}}$  that are not in  $\mathcal{G}_Q$ , do not incur any cost.

Lemma 1 The following properties hold on predicted SED.

1.  $\mathcal{F}_s(\mathbf{Z}_{\mathcal{G}_\mathcal{O}}, \mathbf{Z}_{\mathcal{G}_\tau}) \geq 0$  
2.  $\mathcal{F}_s(\mathbf{Z}_{\mathcal{G}_\mathcal{O}},\mathbf{Z}_{\mathcal{G}_\tau}) = 0\iff \mathbf{Z}_{\mathcal{G}_\mathcal{O}}\leq \mathbf{Z}_{\mathcal{G}_\tau}$  
3.  $\mathcal{F}_s(\mathbf{Z}_{\mathcal{G}_Q}, \mathbf{Z}_{\mathcal{G}_T}) \leq \mathcal{F}(\mathbf{Z}_{\mathcal{G}_Q}, \mathbf{Z}_{\mathcal{G}_{\mathcal{T}'}}) + \mathcal{F}(\mathbf{Z}_{\mathcal{G}_{\mathcal{T}'}}, \mathbf{Z}_{\mathcal{G}_T})$

Proof. Properties (1) and (2) follow from the definition of  $\mathcal{F}$  itself. Property (3) follows from the fact that we take the  $L_{2}$  norm. Formally, we state it as follows.

Lemma 2  $\mathcal{F}_s(\mathbf{Z}_{\mathcal{G}_Q},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}})\leq \mathcal{F}(\mathbf{Z}_{\mathcal{G}_Q},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}'}}) + \mathcal{F}(\mathbf{Z}_{\mathcal{G}_{\mathcal{T}'}},\mathbf{Z}_{\mathcal{G}_{\mathcal{T}}}),$  where  $\mathcal{F}(\mathbf{x},\mathbf{y}) = \| ReLU(\mathbf{x} - \mathbf{y})\|$  for any monotonic norm.

PROOF. See App. A.4.

# 3.2 Characterization of GreED

Importance of pair-independence and siamese architecture: If we do not use a siamese architecture, then the embedding model for the query and the target graphs would be different. Hence, the predicted distance would violate symmetry, and therefore, would not be a metric. Furthermore, if the distance computations are pair-dependent [27, 41, 2], i.e., it jointly learns the embedding of the query and the target, then a single graph may correspond to multiple representations. Hence, it would not be a metric or satisfy the triangle inequality.

Complexity Analysis: The complexity of GED and SED inference in GreED is linear to the number of nodes and edges in the query and target graphs (See App. B for derivation). This computation cost is drastically lower than the factorial computation cost of optimal GED and SED. With respect to neural methods for graph similarity [2, 3, 27, 36, 41], all have at least quadratic computation cost, i.e.,  $O(|\mathcal{V}|^2)$ .

Indexing Embeddings: Since the generated embeddings for both GED and SED satisfy triangle inequality, they are indexable leading to fast querying times. We develop an index structure to exploit this property. Due to space limitations, the details are included in App. C. In addition, we also design a neighborhood decomposition scheme, which enables fast pruning of the exponential search space § C.4. In § 4.3, we empirically analyze the impact of index structures on querying time.

# 4 Empirical Evaluation

In this section, we establish the following:

- Efficacy: GreED is more accurate than the state of the art approaches for both GED and SED.  
- Efficiency: GreED is orders of magnitude faster than existing approaches and scales well to graphs with millions of nodes.  
- Scalability: Pair-independence and indexability further enhances the scalability of GreED and enables it to be run on CPU-only platforms.

Our code base and datasets are available at https://anonymous.4open.science/r/greed/.

# 4.1 Experimental Setup

We use a machine with an Intel Xeon Gold 6142 processor and GeForce GTX 1080 Ti GPU for all our experiments.

Datasets: Table 1 lists the datasets used for benchmarking. Further details on the dataset semantics are provided in the App. D. We include a mixture of both graph databases (#graphs >1), as well as single large graphs (#graphs = 1). Linux and IMDB are unlabeled. We note that this is the first study to evaluate neural graph distance approaches on million-scale graphs.

Baselines: To evaluate performance in GED, we compare with GENN-A* [36] and H $^2$ MN[41]. These are two of the most recent neural frameworks and have shown better efficacy than other neural approaches such as SIMGNN [2], GRAPHSIM [3], and GMN [27].

For SED, no neural approaches exist. However,  $\mathrm{H}^2\mathrm{MN}$  and SIMGNN can be trained by replacing GED with SED along with minor modifications in training. NSC [29] is a method for counting subgraphs using graph embeddings. Since this is a related operation, we use NSC as a baseline by changing the loss function to minimize the RMSE between true and predicted SED. We also use NEUROMATCH [32] as a baseline, which was originally designed to detect subgraph isomorphism. While NEUROMATCH cannot predict SED, it generates a violation score, which can be interpreted as the likelihood of the query being subgraph isomorphic to the target. The violation score can be used as a proxy for SED and used in ranking of  $k$ -NN ( $k$ -NearestNeighbour) queries. Thus, NEUROMATCH comparisons are limited to  $k$ -NN queries on SED. GMN [27], GRAPHSIM [3] and GENN-A* are not included since they cannot be easily adapted for SED. See App. E for details.

In the non-neural category, we use mixed integer programming based method MIP-F2 [25] with a time bound of 0.1 seconds per pair for both GED and SED. MIP-F2 provides the optimal solution given infinite time. We also compare with BRANCH [4], which achieves an excellent trade-off between accuracy and time [5]. BRANCH uses linear sum assignment problem with error-correction (LSAPE) to process the search space. We use GEDLIB's [6] implementation of these methods.

Training (and Test) Data Generation: For GED, we use <query,target> graph pairs from IMDB, AIDS', and Linux. Our setup is identical to SIMGNN [2] and  $\mathrm{H}^2\mathrm{MN}$  [41].

For SED, the target graphs are taken from datasets listed in Table 1. For the query graph, in AIDS, we use known functional groups [31]. In the rest of the graph datasets, queries are sampled by

![](images/38977388c9251b7fdd371dfa6b87ac254fa856528ec5d246afe6ea764613e4f4.jpg)  
(a) PubMed

![](images/deb10f3710d6f52d956f1abd9028c2972faf4798a0f0e19309e2684da5728cd1.jpg)  
(c) Cora_ML

![](images/5bd781c6c07ab5ad9cba59287bc8ee3a5702efa7473211b922ed993514e247b6.jpg)  
(d) Protein

![](images/5e9dac8953940ff2dd0b0f25e5b84b02171ac3225fd77715d0d561111bb7d95f.jpg)

![](images/3d07eda50cb07b3bfd85cbf27456c1df50511738f17ca3f3095a81e9d378dbf7.jpg)

![](images/55d54557b561e95f656bbd70c5d0deaf0bb56b572593a24dc90ad497f1bbf72b.jpg)  
(f) AIDS

![](images/45fe3e253ef1283fc988b6f7f145fd0176e571906ddb725b339bda326a1c2990.jpg)  
(b)CiteSeer  
(g) Linux

![](images/70f46c9f7bd705891b840f78279f8f53e8c5572f7183c7f66b839cbaf78e4212.jpg)  
Figure 3: F1-score in range queries on SED (a-e) and GED (f-h). The range threshold is set as a percentage of the max distance observed in the test set. The legend for Figs. (a)-(e) is provided in (a) and for (f-h) is provided in (h). (i-k) Ablation study to analyze the impact of siamese architecture and function  $\mathcal{F}$ . The legend for Figs. (i)-(k) is provided in (i).  
(h)IMDB

![](images/a58ab4072edc7db6070a736e88a87cc9d558b384aa48ff648cb71704af013d3d.jpg)  
(i) Dblp

![](images/c4016bbd80131e6923c6355b200e170ebcfbb6772ddf17b87d226fdf2e32ad59.jpg)  
(e) AIDS  
(j) PubMed

![](images/0257c9817803db9e20f717e25410372e289c357a9463256d0d7b706b71f334cb.jpg)  
(k)CiteSeer

performing a random BFS traversal (depth up to 5). Table 1 shows the average query sizes  $(|\mathcal{V}_{\mathcal{Q}}|, |\mathcal{E}_{\mathcal{Q}}|)$ . We use mixed integer programming method F2 [25] implemented in GEDLIB [6] with a large time limit to generate ground-truth data.

Train-Validation-Test: We use  $100K$  query-target pairs for training and  $10K$  pairs each for validation and test. All models are trained till validation loss is minimized or there is less than  $0.05\%$  change in validation loss over a number of extended epochs. The codebase of all baselines have been obtained from the respective authors. For all baselines, we use the default parameters suggested by the authors. We also conducted experiments on the datasets to ensure that the default parameters indeed provide the best performance. For GreED, we set the number of layers in GIN to 8. The hidden layer dimension is set to 64.

# 4.2 Prediction Accuracy of SED and GED

Tables 2a and 2b present the accuracy of all techniques on GED and SED in terms of Root Mean Square Error (RMSE). GreED outperforms all other techniques in 9 out of 10 settings across GED and SED. While  $\mathrm{H}^2\mathrm{MN}$  and NSC are the second best performers in SED, GENN-A* performs well in GED. GENN-A*, however, is extremely slow and does not scale on graphs of size beyond 10.  $\mathrm{H}^2\mathrm{MN}$ , thus, provides the second best balance between efficacy and efficiency after GreED. The gap in accuracy is the highest in IMDB for GED, where GreED is more than 10 times better than the neural baselines. IMDB graphs are significantly denser and larger than AIDS' or Linux. Thus, computing the optimal GED is harder. While all techniques have higher errors in IMDB, the deterioration is more severe in the baselines indicating that GreED scales better with graph sizes.

Range and k-NN queries: To quantify performance in range and  $k$ -NN queries, we measure F1-score (Range query) and Kendalls's tau ( $k$ -NN) [22] of the predicted answer set, when compared against the ground truth. In Figs. 3a-3h, we measure the performance in range queries. In SED, GreED consistently outperforms all baselines in F1-score. In GED, the trend remains similar. Although, GENN-A* outperforms GreED for a brief region in Linux, overall, GreED has the highest F1-score. We also note the GENN-A* is not included in Fig. 3h since it fails to scale on IMDB. In  $k$ -NN queries (Tables 1a and 1b), GreED outperforms all algorithms in SED. In GED, similar to the trend in range queries, GreED is the dominant method and GENN-A* marginally outperforms GreED in Linux.

Impact of Query Size: We next investigate how the accuracy varies against the query size. Intuitively, the task gets harder with query size since the combinatorial space of possible maps increases exponentially. For this analysis, we compare GREED with  $\mathbf{H}^2\mathbf{MN}$  in IMDB and Db1p for GED and SED respectively. GENN-A* fails to scale on both datasets.

(a) Ranking in GED.  
Table 1: Kendall's tau scores (higher is better).  

<table><tr><td>Methods</td><td>AIDS&#x27;</td><td>Linux</td><td>IMDB</td></tr><tr><td>GREED</td><td>0.80</td><td>0.89</td><td>0.87</td></tr><tr><td>H2MN</td><td>0.74</td><td>0.88</td><td>0.80</td></tr><tr><td>GENN-A*</td><td>0.75</td><td>0.90</td><td>NA</td></tr><tr><td>SIMGNN</td><td>0.72</td><td>0.86</td><td>0.67</td></tr></table>

(b) Ranking in SED.  

<table><tr><td>Methods</td><td>PubMed</td><td>CiteSeer</td><td>Cora_ML</td><td>Protein</td><td>AIDS</td></tr><tr><td>GREED</td><td>0.90</td><td>0.90</td><td>0.91</td><td>0.75</td><td>0.80</td></tr><tr><td>H2MN</td><td>0.87</td><td>0.88</td><td>0.88</td><td>0.70</td><td>0.72</td></tr><tr><td>Nsc</td><td>0.89</td><td>0.88</td><td>0.88</td><td>0.74</td><td>0.78</td></tr><tr><td>SIMGNN</td><td>0.85</td><td>0.87</td><td>0.86</td><td>0.63</td><td>0.73</td></tr><tr><td>NEUROMATCH</td><td>0.70</td><td>0.75</td><td>0.73</td><td>0.57</td><td>0.59</td></tr></table>

![](images/a6145bbdd8dde1c2f7f649d7d4b58837335f699d1770317b03ad7185ebad4f6f.jpg)  
(a) GreED,IMDB

![](images/8ab1fe7cf8723d748d725f40c5a67224fd3424585bb3d5ab8a3b0aac9c229521.jpg)  
(b)  $\mathrm{H}^2\mathrm{MN}$ , IMDB

![](images/5141bd0dd4b441b2fdc52dbb436271c2785dbc4450bb0c3e47ae47d56d7c65f5.jpg)  
Figure 4: Heat Map of RMSE in (a-b) GED and (c-d) SED against query size in IMDB and Dblp.  
(c) GreED, Dblp

![](images/a8c33215b752b6a6d23ed7a0ebfb5baac664813efc27d6f12d8f62e38b8d6d82.jpg)  
(d)  $\mathrm{H}^2\mathrm{MN}$ , IMDB

In Fig. 4, we plot the heat map of RMSE against query graph size. In this plot, each dot corresponds to a query graph  $\mathcal{G}_{\mathcal{Q}}$ . The co-ordinate of a query is  $(\mathrm{GED}(\mathcal{G}_{\mathcal{Q}},\mathcal{G}_{\mathcal{T}}),|\mathcal{V}_{\mathcal{Q}}|)$  (analogously defined for SED). The color of a dot represents the RMSE; the darker the color, the higher is the RMSE. When we compare the heat maps of GreED with  $\mathrm{H}^2\mathrm{MN}$ , we observe that  $\mathrm{H}^2\mathrm{MN}$  is noticeably darker. Furthermore, the concentration of dark colors is noticeably higher on the upper-right corner indicating deterioration with larger query sizes and higher distance values. This indicates that GreED scales better with query sizes and distances.

Visualization: A case study to visually illustrate the efficacy of GreED is provided in App. H.

# 4.3 Efficiency

Tables 2a-2b present the inference times per  $10K$  query-target pairs. In this experiment, we do not index embeddings by GreED so that the comparison unearths the raw difference in computation efficiency of solely the neural architectures. As visible, GreED is up to 1800 times faster than the non-neural baselines and up to 10 to 20 times faster than  $\mathrm{H}^2\mathrm{MN}$ , the current state of the art in GED prediction. Also note that GENN- $\mathbf{A}^{*}$  is exorbitantly slow (Table 2a). GENN- $\mathbf{A}^{*}$  is slower since it not only predicts the GED but also the alignment via an  $\mathbf{A}^{*}$  search. While the alignment information is indeed useful, computing this information across all graphs in the database may generate redundant information since an user is typically interested only on a small minority of graphs that are in the answer set. In App. G, we discuss this issue in detail.

# 4.4 Pair-independence and Indexability

Here, we showcase how pair-independent embeddings, and ensuring triangle inequality leads to further boost in scalability. For this experiment, we use the three largest datasets of PubMed, Amazon and Dblp. For each dataset, we pre-compute GreED embeddings of all database graphs by exploiting pair-independent embeddings. Such pre-computation is not possible in the neural or non-neural baselines. Furthermore, since the predictions of GreED satisfy triangle inequality, we index the pre-computed embeddings of the database graphs as discussed in § C.1. Consequently, for GreED, we only need to embed the query graph and evaluate  $\mathcal{F}$  to make predictions at query time. Table 2c presents the results on range and 10-NN queries. When computations are done on a GPU, GreED is more than 1000 times faster than  $\mathrm{H}^2\mathrm{MN}$ . In the absence of a GPU,  $\mathrm{H}^2\mathrm{MN}$  is practically infeasible since expensive pair-dependent computations are done at query time. In contrast, even on a CPU, through indexing, GreED is  $\approx 50$  times faster than GPU-based  $\mathrm{H}^2\mathrm{MN}$ . Note that indexing enables up to 3-times speed-up on GreED over linear scan, which demonstrates the gain from ensuring triangle inequality. These results establish that GreED breaks new ground in scalability of neural graph distance computations; not only is it faster, it overcomes the barrier of GPU-dependence and hence better suited for low-resource environments.

(a) GED  

<table><tr><td>Methods</td><td>AIDS&#x27;</td><td>Linux</td><td>IMDB</td></tr><tr><td>GREED</td><td>0.49</td><td>0.70</td><td>0.63</td></tr><tr><td>H2MN</td><td>9.50</td><td>8.74</td><td>8.83</td></tr><tr><td>GENN-A*</td><td>12190</td><td>1340</td><td>NA</td></tr><tr><td>BRANCH</td><td>10.70</td><td>8.24</td><td>127.90</td></tr><tr><td>MIP-F2</td><td>593.34</td><td>191.88</td><td>1173.548</td></tr></table>

<table><tr><td>Methods</td><td>Dblp</td><td>Amazon</td><td>PubMed</td><td>CiteSeer</td><td>Cora_ML</td><td>Protein</td><td>AIDS</td></tr><tr><td>GREED</td><td>6.84</td><td>1.46</td><td>1.30</td><td>1.28</td><td>1.25</td><td>0.86</td><td>0.84</td></tr><tr><td>H2MN</td><td>44.68</td><td>23.2</td><td>25.79</td><td>27.54</td><td>29.04</td><td>19.33</td><td>9.63</td></tr><tr><td>NSC</td><td>NA</td><td>21</td><td>35.05</td><td>24.46</td><td>70.59</td><td>21</td><td>4</td></tr><tr><td>SIMGNN</td><td>109.56</td><td>47.68</td><td>39.80</td><td>39.40</td><td>40.73</td><td>39.02</td><td>43.83</td></tr><tr><td>BRANCH</td><td>626.489</td><td>79.25</td><td>99.11</td><td>155.09</td><td>132.98</td><td>52.26</td><td>12.93</td></tr><tr><td>MIP-F2</td><td>1979.185</td><td>861.95</td><td>606.01</td><td>827.65</td><td>790.01</td><td>881.77</td><td>360.12</td></tr></table>

(b) SED  
(c) Scalability  

<table><tr><td rowspan="3">Datasets</td><td colspan="4">Range (θ = 2)</td><td colspan="4">10-NN</td></tr><tr><td colspan="2">CPU</td><td colspan="2">GPU</td><td colspan="2">CPU</td><td colspan="2">GPU</td></tr><tr><td>L-Scan</td><td>Indexed</td><td>L-Scan</td><td>H2MN</td><td>L-Scan</td><td>Indexed</td><td>L-Scan</td><td>H2MN</td></tr><tr><td>PubMed</td><td>0.693</td><td>0.56</td><td>0.004</td><td>26.6</td><td>1.01</td><td>0.49</td><td>0.004</td><td>27.5</td></tr><tr><td>Amazon</td><td>9.09</td><td>5.07</td><td>0.025</td><td>371</td><td>11.3</td><td>4.75</td><td>0.027</td><td>372</td></tr><tr><td>Dblp</td><td>48</td><td>20.9</td><td>0.070</td><td>696</td><td>50.4</td><td>18.6</td><td>0.126</td><td>698</td></tr></table>

Table 2: (a-b) Running times of all methods in seconds per 10k query-target pair. (c) Querying time (s) for SED in the three largest datasets. L-Scan indicates time taken by linear scan in GreED (times differ based on whether executed on CPU or GPU).

# 4.5 Ablation Study

In this study, we explore the impact of our inductive biases in learning from low-volume data. We create two variants of GreED: (1) GreED-Dual trains the two parallel GNN models separately without weight-sharing, and (2) GreED-NN uses an MLP instead of  $\mathcal{F}$ . Both have strictly better representational capacity than GreED, so are expected to match the performance with infinite data. Figs. 3i-3k present the results. The RMSE of GreED is consistently better than GreED-Dual, with the difference being more significant at low volumes. This indicates that siamese structure helps. Compared to GreED, GreED-NN achieves marginally better performance at larger train sizes in PubMed and CiteSeer. However, in Dblp, GreED is consistently better. The number of subgraphs in a dataset grows exponentially with the node set size. Hence, an MLP needs growing training data to accurately model the intricacies of this search space. In Dblp, even 100k pairs is not enough to improve upon  $\mathcal{F}$ . Overall, these trends indicate that  $\mathcal{F}$  enables better generalization and scalability with respect to accuracy. Furthermore, given that its performance is close to an MLP, and it enables indexing, the benefits outweigh the marginal reduction in accuracy.

More ablations studies justifying our choice of GIN and the sum-pool layer are provided in App. F.

# 4.6 Generalization to Unseen Query Distributions in SED

We train the model by sampling queries from the graph database through BFS enumerations. How does GreED generalize to unseen distributions? Towards that end, we generate queries from the three unseen distributions of (1) Random Walks (Rw), (2) Random Walks with Restarts (RWR), and (3) SHADOW [39] (See App. I for details on the sampling strategies). We first note that in AIDS, we use real queries of functional groups, and thus the good performance in AIDS indicates good generalizability. In Table 3a, we more exhaustively analyze this aspect. As visible, the errors remain low. Even more surprisingly, the errors on Rw and RWR are better than the train distribution of BFS itself. This indicates good generalization to unseen distributions.

# 4.7 Generalizability to Unseen, Larger Query Sizes:

Generating training data for learning GED and SED is expensive since optimal distance computations are NP-hard. Hence, a desirable property would be to learn from small graphs and then generalize to larger unseen graphs. We evaluate this ability for GreED and  $\mathrm{H}^2\mathrm{MN}$ , which are the two best performing algorithms, and show that more needs to be done. Table 3b provides the numbers. We notice that although there is some deterioration in the quality for query sizes in the range [25, 50] when compared to the entire set, it is not severe (GreED-50 in Table 3b). However, if the train set only contains queries till size 25 and we deploy the learned model to infer on queries of larger unseen sizes, the drop in quality is significant (GreED-25 in Table 3b). This drop is even more dramatic in  $\mathrm{H}^2\mathrm{MN}$ . On the positive side, GreED remains superior to the optimal non-neural approach (MIP-F2) when run with a generous time limit of 60 seconds per query. Overall, this experiment highlights one direction that needs further study and improvement.

# 5 Conclusions, Limitation, and Future Directions

The problem of learning graph distances from their embeddings has seen much interest over the last few years. This thread of research is important since it allows us to overcome the bottleneck of exponential graph alignment space. Our experiments clearly establish GreED as the state of the art for both GED and SED. In addition, it is significantly faster and provides better theoretical correspondence between properties of the original space and predicted space. One clear direction of future work that emerges from our experiments is that GreED, and existing methods of graph distance learning, do not generalize well to unseen larger query sizes. We hope to address this limitation next.

(a) Query distributions  

<table><tr><td>Sampler</td><td>PubMed</td><td>CiteSeer</td><td>Amazon</td></tr><tr><td>BFS</td><td>0.728</td><td>0.519</td><td>0.495</td></tr><tr><td>Rw</td><td>0.508</td><td>0.770</td><td>0.490</td></tr><tr><td>RWR</td><td>0.545</td><td>0.754</td><td>0.299</td></tr><tr><td>SHADOW</td><td>0.966</td><td>0.753</td><td>0.830</td></tr></table>

Table 4: (a) RMSE on unseen query distributions. BFS (seen) is the baseline to compare against. (b) RMSE against query sizes. GreED-50 indicates GreED trained on a dataset containing queries of size up to 50. GreED-25 is defined analogously.  
(b) Query size  

<table><tr><td rowspan="2">Method</td><td colspan="2">PubMed</td><td colspan="2">CiteSeer</td><td colspan="2">Amazon</td></tr><tr><td>VQ∈[0,50]</td><td>VQ∈[25,50]</td><td>VQ∈[0,50]</td><td>VQ∈[25,50]</td><td>VQ∈[0,50]</td><td>VQ∈[25,50]</td></tr><tr><td>GREED-50</td><td>1.294</td><td>1.917</td><td>0.728</td><td>0.948</td><td>0.638</td><td>0.782</td></tr><tr><td>GREED-25</td><td>2.824</td><td>4.999</td><td>4.740</td><td>9.052</td><td>1.152</td><td>1.724</td></tr><tr><td>H2MN-50</td><td>3.1334</td><td>5.112</td><td>4.9380</td><td>8.583</td><td>6.014</td><td>9.550</td></tr><tr><td>H2MN-25</td><td>7.417</td><td>13.366</td><td>10.459</td><td>19.787</td><td>5.72</td><td>9.462</td></tr><tr><td>MIP-F2</td><td>3.507</td><td>6.278</td><td>4.831</td><td>8.505</td><td>6.454</td><td>10.293</td></tr></table>

# References

[1] Jiyang Bai and Peixiang Zhao. Tagsim: Type-aware graph similarity learning and computation. Proc. VLDB Endow., 15(2):335-347, 2021.  
[2] Yunsheng Bai, Hao Ding, Song Bian, Ting Chen, Yizhou Sun, and Wei Wang. Simgnn: A neural network approach to fast graph similarity computation. In wSDM, WSDM '19, page 384-392, 2019.  
[3] Yunsheng Bai, Hao Ding, Ken Gu, Yizhou Sun, and Wei Wang. Learning-based efficient graph similarity computation via multi-scale convolutional set matching. AAAI, pages 3219-3226, Apr. 2020.  
[4] David B Blumenthal. New techniques for graph edit distance computation. arXiv preprint arXiv:1908.00265, 2019.  
[5] David B Blumenthal, Nicolas Boria, Johann Gamper, Sébastien Bougleux, and Luc Brun. Comparing heuristics for graph edit distance computation. The VLDB Journal, 29(1):419-458, 2020.  
[6] David B Blumenthal, Sébastien Bougleux, Johann Gamper, and Luc Brun. Gedlib: a c++ library for graph edit distance computation. In International Workshop on Graph-Based Representations in Pattern Recognition, pages 14-24. Springer, 2019.  
[7] Aleksandar Bojchevski and Stephan Gunnemann. Deep gaussian embedding of graphs: Unsupervised inductive learning via ranking. arXiv preprint arXiv:1707.03815, 2017.  
[8] Angela Bonifati, Wim Martens, and Thomas Timm. An analytical study of large sparql query logs. Proc. VLDB Endow., 11(2):149-161, 2017.  
[9] Sébastien Bougleux, Luc Brun, Vincenzo Carletti, Pasquale Foggia, Benoit Gaizère, and Mario Vento. Graph edit distance as a quadratic assignment problem. Pattern Recognition Letters, 87:38-46, 2017. Advances in Graph-based Pattern Recognition.  
[10] Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah. Signature verification using a" siamese" time delay neural network. Advances in neural information processing systems, 6:737-744, 1993.  
[11] Paolo Ciaccia, Marco Patella, and Pavel Zezula. M-tree: An efficient access method for similarity search in metric spaces. In VLDB, 1997.  
[12] Évariste Daller, Sébastien Bougleux, Benoit Gaizère, and Luc Brun. Approximate Graph Edit Distance by Several Local Searches in Parallel. In 7th International Conference on Pattern Recognition Applications and Methods, 2018.  
[13] Khoa D. Doan, Saurav Manchanda, Suchismit Mahapatra, and Chandan K. Reddy. Interpretable graph similarity computation via differentiable optimal alignment of node embeddings. In SIGIR, page 665-674, 2021.  
[14] Vlastislav Dohnal, Claudio Gennaro, Pasquale Savino, and Pavel Zezula. D-index: Distance searching index for metric data sets. Multim. Tools Appl., 21(1):9-33, 2003.  
[15] Chi Thang Duong, Trung Dung Hoang, Hongzhi Yin, Matthias Weidlich, Quoc Viet Hung Nguyen, and Karl Aberer. Efficient streaming subgraph isomorphism with graph neural networks. 14(5):730-742, January 2021.  
[16] Christopher L. Ebsch, Joseph A. Cottam, Natalie C. Heller, Rahul D. Deshmukh, and George Chin. Using graph edit distance for noisy subgraph matching of semantic property graphs. In 2020 IEEE International Conference on Big Data (Big Data), pages 2520-2525, 2020.  
[17] Charles Elkan. Using the triangle inequality to accelerate k-means. In ICML, page 147-153, 2003.  
[18] Ali S. Hadi. Finding groups in data: An introduction to chster analysis. Technometrics, 34:111-112, 1991.

[19] Huahai He and Ambuj K Singh. Closure-tree: An index structure for graph queries. In ICDE, pages 38-38. IEEE, 2006.  
[20] Gisli R. Hjaltason and Hanan Samet. Index-driven similarity search in metric spaces (survey article). ACM Trans. Database Syst., 28(4):517-580, 2003.  
[21] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. arXiv preprint arXiv:1905.12265, 2019.  
[22] M. G. Kendall. A new measure of rank correlation. Biometrika, 30(1/2):81-93, 1938.  
[23] Wojciech Kwedlo and Pawel J. Czochanski. A hybrid mpi/openmp parallelization of $k$ -means algorithms accelerated using the triangle inequality. IEEE Access, 7:42280-42297, 2019.  
[24] AA Leman and B Weisfeiler. A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsiya, 2(9):12-16, 1968.  
[25] Julien Lerouge, Zeina Abu-Aisheh, Romain Raveaux, Pierre Héroux, and Sébastien Adam. New binary linear programming formulation to compute the graph edit distance. Pattern Recognition, 72:254-265, 2017.  
[26] Jure Leskovec and Rok Sosic. Snap: A general-purpose network analysis and graph-mining library. ACM Transactions on Intelligent Systems and Technology (TIST), 8(1):1, 2016.  
[27] Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, and Pushmeet Kohli. Graph matching networks for learning the similarity of graph structured objects. In ICML, pages 3835-3845, 2019.  
[28] Yongjiang Liang and Peixiang Zhao. Similarity search in graph databases: A multi-layered indexing approach. In ICDE, pages 783-794, 2017.  
[29] Xin Liu, Haojie Pan, Mutian He, Yangqiu Song, Xin Jiang, and Lifeng Shang. Neural subgraph isomorphism counting. In KDD, page 1959-1969, 2020.  
[30] Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. In ICML workshop on Graph Representation Learning and Beyond, 2020.  
[31] Sayan Ranu and Ambuj K Singh. Mining statistically significant molecular substructures for efficient molecular classification. Journal of chemical information and modeling, 49(11):2537-2550, 2009.  
[32] Rex, Ying, Zhaoyu Lou, Jiaxuan You, Chengtao Wen, Arquimedes Canedo, and Jure Leskovec. Neural subgraph matching, 2020.  
[33] Aravind Sankar, Sayan Ranu, and Karthik Raman. Predicting novel metabolic pathways through subgraph mining. Bioinformatics, 33(24):3955-3963, 2017.  
[34] Jeffrey K. Uhlmann. Satisfying general proximity/similarity queries with metric trees. Inf. Process. Lett., 40:175-179, 1991.  
[35] Lichen Wang, Bo Zong, Qianqian Ma, Wei Cheng, Jingchao Ni, Wenchao Yu, Yanchi Liu, Dongjin Song, Haifeng Chen, and Yun Fu. Inductive and unsupervised representation learning on graph structured objects. In ICLR, 2020.  
[36] Runzhong Wang, Tianqi Zhang, Tianshu Yu, Junchi Yan, and Xiaokang Yang. Combinatorial learning of graph edit distance via dynamic embedding. In CVPR.  
[37] Haibo Xiu, Xiao Yan, Xiaogiang Wang, James Cheng, and Lei Cao. Hierarchical graph matching network for graph similarity computation, 2020.  
[38] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018.

[39] Hanqing Zeng, Muhan Zhang, Yinglong Xia, Ajitesh Srivastava, Andrey Malevich, Rajgopal Kannan, Viktor K. Prasanna, Long Jin, and Ren Chen. Deep graph neural networks with shallow subgraph samplers. In NeurIPS, 2021.  
[40] Zhiping Zeng, Anthony K. H. Tung, Jianyong Wang, Jianhua Feng, and Lizhu Zhou. Comparing stars: On approximating graph edit distance. Proc. VLDB Endow., 2(1):25-36, 2009.  
[41] Zhen Zhang, Jiajun Bu, Martin Ester, Zhao Li, Chengwei Yao, Zhi Yu, and Can Wang. H2mn: Graph similarity learning with hierarchical hypergraph matching networks. In KDD, page 2274-2284, 2021.  
[42] Xiang Zhao, Chuan Xiao, Xuemin Lin, Qing Liu, and Wenjie Zhang. A partition-based approach to structure similarity search. VLDB, 7(3):169-180, 2013.
