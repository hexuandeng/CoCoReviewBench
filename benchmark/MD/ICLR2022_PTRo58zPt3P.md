# INDUCTIVE RELATION PREDICTION USING ANALOGY SUBGRAPH EMBEDDINGS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Prevailing methods for relation prediction in heterogeneous graphs including knowledge graphs aim at learning the latent representations (i.e., embeddings) of observed nodes and relations, and are thus limited to the transductive setting where the relation types must be known during training. In this paper, we propose ANalogy SubGraph Embedding Learning (GraphANGEL), a novel relation prediction framework that predicts relations between each node pair by checking whether the subgraphs containing the pair are similar to other subgraphs containing the considered relation. Each graph pattern explicitly represents a specific logical rule, which contributes to an inductive bias that facilitates generalization to unseen relation types and leads to more explainable predictive models. Our model consistently outperforms existing models in terms of heterogeneous graph based recommendation as well as knowledge graph completion. We also empirically demonstrate the capability of our model in generalizing to new relation types while producing explainable heat maps of attention scores across the discovered logics.

# 1 INTRODUCTION

Relation modeling aims to learn the relations between nodes, leading to advances in a wide range of applications, e.g., recommender systems (Karen et al., 2009), knowledge graphs (Bordes et al., 2013), and biology (Yasunaga et al., 2021). As most relational data in the real world is heterogeneous, a principal way is to organize it into a heterogeneous graph. The dominant paradigms for relation prediction can be categorized into matrix factorization techniques (Nickel et al., 2011; 2012), statistical relational learning approaches (Richardson & Domingos, 2006; Singla & Domingos, 2005), and neural-embedding-based methods (Bordes et al., 2013; Dettmers et al., 2018). Among these, neural-embedding-based methods which learn to encode relational information using low-dimensional representations of nodes and relations, have shown good scalability (Bordes et al., 2013) and inductive learning capability (Battaglia et al., 2018) in terms of validating unseen nodes.

Results of such methods show that graph neural networks (GNNs) are able to condense the neighborhood connectivity pattern of each node into a node-specific low-dimensional embedding and successfully exploit such local connectivity patterns and homophily. Recent advances further reveal the logical expressiveness of GNNs (Barceló et al., 2019) and support their inductive ability to generalize to unseen nodes (Teru et al., 2020; Zhang & Chen, 2019).

In contrast, a limited study has been conducted (Yang et al., 2014) on the inductive learning capability for unseen relation types. Such inductive ability, if successfully exploited, can directly improve logical expressiveness of GNNs and enable them to effectively capture the underlying logical semantics (e.g., logical rules). Note that this is more challenging than unseen nodes, since it is usually hard to define the "neighborhood" of a relation type topologically.

In this paper, we propose ANalogy SubGraph Embedding Learning (GraphANGEL), a new relation prediction paradigm that holds a strong inductive bias to generalize to unseen relation types. Given a pair of nodes to predict the existence of a specific relation between them, the core idea is to extract some analogy subgraphs containing the pair, and compare them against other subgraphs sharing similar shapes. We call these shapes graph patterns.

Taking Figure 1 as an example, the task is to predict whether Person E lives in London. We construct three patterns involving the relation live. The first is target pattern containing the source

and target nodes, i.e., Person E and London, for which we are to predict the existence of the relation live. The second is supporting pattern that includes live as evidence supporting the existence. The third is refuting pattern that does not include live as a baseline for comparison. If an edge does exist between the two nodes, the subgraphs matching the target pattern should be more similar to those matching supporting patterns than the refuting patterns. Following the above intuition, we find a set of subgraphs that match each of the patterns. Then we compare the first set against the set matching supporting patterns, and also against the set matching refuting patterns, using a neural network. As shown in the bottom part of the figure, the subgraphs in the second set share the higher similarity with the one in the first set, so the prediction result in this case is that there exists a live.

Given a triplet  $\langle s, r, t \rangle$ , GraphANGEL consists of the following stages: (1) determining target patterns from  $s, t$  as well as supporting and refuting patterns from  $r$ , (2) retrieving subgraphs matching each pattern, (3) computing the representations of each set and then the similarity between the subgraph set matching target pattern and the set matching supporting/refuting patterns, and the final prediction based on the similarities. For the first stage, our architecture design only involves the graph patterns in pair, 3-cycle, and 4-cycle shapes, which is efficient to match and already shows good results. For the second stage, we introduce efficient searching and sampling techniques to find the subgraphs matching the patterns. We use a GNN with attention in the third stage to combine the sets of subgraphs as well as their node features. The

![](images/ce9ebf1a5d8bdf163492ba5d0ae16f6e244aabc90f995b348b8170903345ebe2.jpg)  
Figure 1: An illustrative example of motivation.

attention module can simultaneously produce an explainable heat map across the discovered patterns. Notably, none of the above stages requires explicitly learning a representation of  $r$ , the relation we are predicting, thus, GraphANGEL naturally generalizes to modeling unseen relation types.

We benchmark GraphANGEL on heterogeneous graph based recommendation and knowledge graph completion tasks with the state-of-the-art methods. For the evaluation of inductive capabilities, we construct several new inductive benchmarks by either removing or adding relations from knowledge graph datasets. Extensive experimental comparisons on these benchmarks exhibit the superiority of our method under both transductive and inductive settings.

# 2 BRIDGING LOGICAL EXPRESSIONS AND GRAPH PATTERNS

We begin with bridging logical expressions and graph patterns based on two intuitions, both of which come from real world experiences:

- A relation can often be inferred from other relations with a combination of simple logical rules that do not involve too many nodes.  
- One can predict relation existence by finding whether the subgraphs containing the pair are similar to the subgraphs containing an edge with the same relation.

Take Figure 1 as an example where we wish to predict whether Person E lives in London. An example of the first intuition goes as follows: if Person E watches the soccer matches with club Arsenal, which is based in London, then Person E may also live in London. However, such logical rules may not always hold, requiring probabilistic inference. For instance, Person G watches the matches with FC Barcelona, which is Based in Barcelona, but he does not live in the same city. To make more accurate predictions, we need to combine other possible logical rules, such as "Friends often live in the same city" - in this case, Person E and Person H are friends, and Person H lives in London.

The second intuition tells us that we can predict whether Person E lives in London by checking if the subgraphs containing the pair (Person E, London) (named target subgraphs, e.g. Person E - Arsenal - London) are similar to the subgraphs containing a live relation (named supporting

![](images/82b6a628af2e2d8b518e628ed9723094e4a3e03a7f9c9f55049ac6fb656629c2.jpg)  
Figure 2: Illustration of GraphANGEL's relation prediction workflow, where different edge colors in the graph  $\mathcal{G}$  represent different relation types, and dashed edges in  $\mathcal{G}$  represent the triplet  $\langle s,r,t\rangle$  we wish to predict. The left box shows the patterns considered in our implementation, where black edges mean matching edges irrespective of relation types. The bottom boxes show the logical function of the three patterns.

subgraphs, e.g. Person I - Chelsea - London). Similarity computation is done by a neural network. To determine whether the two sets of subgraphs are similar, we additionally compare the target subgraphs against another set of subgraphs that do not contain the relation (named refuting subgraphs, e.g. Person E - Person G - Barcelona) as a baseline. A variety of options exist for selecting the refuting subgraphs, ranging from selecting those having the same topology regardless of the actual relation types, to those having both the same shape and relation types.

In addition, we do not include the edge of live in the supporting subgraphs to avoid information leakage. We also require the supporting subgraphs and refuting subgraphs to have the same shape as that of the target subgraphs in order to make similarity computation focus more on the node features and relation types rather than the topology, and that is why we call both supporting subgraphs and refuting subgraphs analogy subgraphs. This enables us to predict live relation with the pipeline above without learning an explicit embedding for live relation, unlike prior works. The reason is that the information of live relation can be implicitly expressed by the other relations found in the supporting and refuting subgraphs, e.g. live can be expressed by watch and based. This serves as the basis for our model's generalizability to relations that have no occurrence in the training set.

We can describe what kind of subgraphs we are looking for in the above example with logical expressions. The target subgraphs above will have the logical expression  $\operatorname{Source}(x) \wedge \operatorname{Target}(y) \wedge \operatorname{Edge}(x, z) \wedge \operatorname{Edge}(z, y)$ , where  $\operatorname{Source}(x)$  means if node  $x$  is the source node  $s$ ,  $\operatorname{Target}(y)$  means if node  $y$  is the target node  $t$ , and  $\operatorname{Edge}(x, z)$  means if there exists an edge regardless of relation type between  $x$  and  $z$ . Our task is to determine whether an edge of relation type  $r$  exists between  $s$  and  $t$ . The supporting subgraphs will have  $\operatorname{Edge}(x, z) \wedge \operatorname{Edge}(z, y) \wedge \operatorname{Edge}_r(x, y)$ , meaning that the subgraph can be any 3-cycle except that there must be an edge with relation type  $r$ . The refuting subgraphs will have  $\operatorname{Edge}(x, z) \wedge \operatorname{Edge}(z, y) \wedge \neg \operatorname{Edge}_r(x, y)$ , meaning that it can be any 2-path except that the starting node and ending node must not have an edge with relation  $r$  in between. This leads to the concept of graph patterns, defined as follows:

Definition 1 (Graph Pattern). A graph pattern is a logical function that takes in a subgraph as input and returns a boolean value, consisting of logical operators  $(\neg ,\land ,\lor)$  as well as indicator operators determined by the existence of nodes and edges, the latter includes:

- Source(x) and Target(x), returning true iff the node  $x$  is the source node and target node, respectively.  
- Edge(x,y), returning true iff there exists an edge between node  $x$  and  $y$ .  
- Edge $_r(x, y)$ , returning true iff there exists an edge of relation  $r$  between node  $x$  and  $y$ .

We call a subgraph  $S$  matches a graph pattern  $\Pi$  if true is returned when  $S$  is applied to  $\Pi$ . Such operation is already well supported in graph databases (Francis et al., 2018), and in the following sections we give more efficient algorithms that perform matching with simple patterns.

# 3 MODELING RELATION WITH ANALOGY SUBGRAPH EMBEDDINGS

For each graph  $\mathcal{G} = (\mathcal{V},\mathcal{E},\mathcal{R})$  where  $\mathcal{V}$  denotes node set,  $\mathcal{E}$  denotes edge set and  $\mathcal{R}$  is relation set, as Figure 2 shows, we outline how GraphANGEL works for each triplet  $\langle s,r,t\rangle$  to predict the

Table 1: A summary of how to construct target, supporting, and refuting patterns.  

<table><tr><td>Base Pattern Πp</td><td>Target Pattern Πp*</td><td>Supporting Pattern Πp+</td><td>Refuting Pattern Πp-</td></tr><tr><td>Πp</td><td>Source(x) ∧ Target(y) ∧ Πp</td><td>Edge_r(x,y) ∧ Πp</td><td>∇Edge_r(x,y) ∧ Πp</td></tr></table>

existence for the edge of type  $r$  connecting source node  $s$  and target node  $t$  as follows. (a) We start by determining  $P$  target patterns denoted as  $\Pi_1^*, \dots, \Pi_P^*$ . For each  $\Pi_p^*$  we also determine its corresponding supporting pattern  $\Pi_p^+$  and refuting pattern  $\Pi_p^-$ . (b) We then sample a set of  $K$  target subgraphs  $\{S_{p,k}^*\}_{k=1}^K$  matching  $\Pi_p^*$ ,  $Q$  supporting subgraphs  $\{S_{p,q}^+\}_{q=1}^Q$  matching  $\Pi_p^+$ , and  $Q$  refuting subgraphs  $\{S_{p,q}^-\}_{q=1}^Q$  matching  $\Pi_p^-$ . (c) For each pattern  $\Pi_p$ , we next compute the representation of each sampled subgraph and obtain the set of target subgraph embeddings  $\{e_{p,k}^*\}_{k=1}^K$ , supporting subgraph embeddings  $\{e_{p,q}^+\}_{q=1}^Q$  and refuting subgraph embeddings  $\{e_{p,q}^-\}_{q=1}^Q$ . (d) We finally compute a similarity score between the set  $\{e_{p,k}^*\}_{p=1,k=1}^{P,K}$  and the set  $\{e_{p,q}^+\}_{p=1,q=1}^{P,Q}$ , as well as  $\{e_{p,k}^*\}_{p=1,k=1}^{P,K}$  and  $\{e_{p,q}^-\}_{p=1,q=1}^{P,Q}$ , to get the relation prediction result.

We summarize the notations in Appendix A1, show the training algorithm in Algorithm 1 and describe the details in the following subsections.

Pattern Construction. During training, we set for each  $\Pi_p$  the target pattern  $\Pi_p^*$ , the supporting pattern  $\Pi_p^+$  and the refuting pattern  $\Pi_p^-$  using Table 1. Since pattern matching is NP-complete like subgraph matching (Lewis, 1983), we first designate before training a set of patterns  $\{\Pi_p\}_{p=1}^P$  whose pattern matching can be computed in manageable time. In practical, we use Pairs, 3-cycles, and 4-cycles.

Subgraph Retrieval. Although general graph pattern matching is supported in graph databases (Francis et al., 2018), more efficient solutions

# Algorithm 1: GraphANGEL

Input: Graph  $\mathcal{G}$ , Patterns  $\Pi_1, \ldots, \Pi_P$ .

for each tuple  $\langle s,r,t\rangle$  do

for each pattern  $\Pi_p$  do

Construct  $\Pi_p^*$ ,  $\Pi_p^+$ ,  $\Pi_p^-$ .

Retrieve  $K$  subgraphs  $\{\mathcal{S}_{p,k}^{*}\}_{k = 1}^{K}$  matching  $\Pi_p^*$ .

Retrieve  $Q$  subgraphs  $\{S_{p,q}^{+}\}_{q = 1}^{Q}$  matching  $\Pi_p^+$ .

Retrieve  $Q$  subgraphs  $\{S_{p,q}^{-}\}_{q = 1}^{Q}$  matching  $\Pi_p^-$ .

Compute  $e_{p,k}^{*}, e_{p,q}^{+}, e_{p,q}^{-}$  via Eq. (1).

end

Compute  $s^+, s^-$  via Eq. (2).

Update parameters according to Eq. (3).

end

exist for simpler graph patterns, especially when we only consider pairs, 3-cycle and 4-cycle shapes. Searching and retrieving the subgraphs matching patterns in Pair shape is trivial since it reduces to finding edges with a given relation, so the following discussion only involves 3-cycle and 4-cycle patterns. We first precompute and store all subgraphs matching the patterns in 3-cycle and 4-cycle shapes by our algorithm, summarized as follows, with pseudocode, correctness proofs and complexity analysis in Appendix A2.

- 3-cycles. We first partition the node set by the node degrees into two sets  $\mathcal{V} = \mathcal{V}_1 \cup \mathcal{V}_2$ .  $\mathcal{V}_1$  contains the nodes whose degrees are less than  $|\mathcal{E}|^{1/2}$ , and  $\mathcal{V}_2$  contains the rest. For each  $u, v, w \in \mathcal{V}_1$ , we check if they form a 3-cycle. Then, for each  $u \in \mathcal{V}_2$ , we enumerate all pairs of its neighbors and see if they are connected. The overall complexity to find all the subgraphs is  $O(|\mathcal{E}|^{3/2})$ .  
- 4-cycles. For each node  $u \in \mathcal{V}$ , we find all 2-paths  $(u, v, w)$  whose starting node  $u$  has the largest degree, i.e.,  $d_u = \max(d_u, d_v, d_w)$ . Each pair of 2-paths that shares the same ending node forms a 4-cycle. In this way, all 4-cycles can be enumerated by the starting node with the largest degree. The overall complexity to find all subgraphs is  $O(\max(|\mathcal{E}|^{3/2}, n_{4-cycle}))$  where  $n_{4-cycle}$  is the number of the subgraphs.

Then, we perform pattern matching for 3-cycles and 4-cycles for each subgraph as follows.

- Target patterns. Matching 3-cycle target patterns reduces to finding common neighbors of  $s$  and  $t$ , which takes  $O(d_s + d_t)$  time where  $d_s$  and  $d_t$  are degrees of  $s$  and  $t$ . Matching 4-cycle target pattern reduces to finding whether the neighbors of  $s$  and  $t$  are connected, taking  $O(d_s d_t)$  time.  
- Supporting patterns. Matching supporting patterns reduces to finding a subgraph containing an edge of type  $r$  in the precomputed result. One can efficiently retrieve with a precomputed inverted map with relation  $r$  as key and the actual subgraphs containing it as value.  
- Refuting patterns. Matching refuting patterns reduces to random walks, followed by checking whether the starting node and the ending node has an edge of type  $r$ .

Table 2: Patterns  ${\Pi }_{p}$  considered in our experiments.  

<table><tr><td>Task</td><td>Pair</td><td>3-cycle (with type)</td><td>4-cycle (with type)</td></tr><tr><td>Knowledge Graph Completion</td><td>true</td><td>Edge(x,z) ∧ Edge(z,y)</td><td>Edge(x,z) ∧ Edge(z,w) ∧ Edge(w,y)</td></tr><tr><td>Heterogeneous Graph Recommendation</td><td>true</td><td>Edgea(x,z) ∧ Edgeb(z,y)</td><td>Edgea(x,z) ∧ Edgeb(z,w) ∧ Edgec(w,y)</td></tr></table>

We further design a series of novel uniform sampling algorithms such that the time complexity of sampling  $K$  supporting cases of  $\Pi_{4 - cycle}$  reduces to  $O(|\mathcal{E}|^{\frac{3}{2}} + K)$ , sampling  $Q$  refuting cases of  $\Pi_{3 - cycle}$  or  $\Pi_{4 - cycle}$  reduces to  $O(|\mathcal{V}| + |\mathcal{E}| + Q)$ . See detailed descriptions in Appendix A2.

Representation Computation. We apply a neural network  $\Phi(\cdot)$  over each subgraph  $S_{p,k}^{*}$ ,  $S_{p,q}^{+}$  and  $S_{p,q}^{-}$  to obtain graph-level representations  $e_{p,k}^{*}$ ,  $e_{p,q}^{+}$ ,  $e_{p,q}^{-}$ , following

$$
\boldsymbol {e} _ {p, k} ^ {*} = \Phi \left(\mathcal {S} _ {p, k} ^ {*}\right), \boldsymbol {e} _ {p, q} ^ {+} = \Phi \left(\mathcal {S} _ {p, q} ^ {+}\right), \boldsymbol {e} _ {p, q} ^ {-} = \Phi \left(\mathcal {S} _ {p, q} ^ {-}\right). \tag {1}
$$

In the implementation, we adopt single layer R-GCN (Schlichtkrull et al., 2018) followed by any readout function, e.g., Mean(·), Max(·) as  $\Phi(\cdot)$ .

Similarity Computation. We deploy a neural network  $\Psi(\cdot)$  to measure the similarity  $s^+$  between the set of subgraphs matching  $\Pi_1^*, \ldots, \Pi_P^*$  and the set of subgraphs matching  $\Pi_1^+, \ldots, \Pi_P^+$ ; and the similarity  $s^-$  between the set of subgraphs matching  $\Pi_1^*, \ldots, \Pi_P^*$  and the set of subgraphs matching  $\Pi_1^-, \ldots, \Pi_P^-$ , which can be formulated as

$$
\begin{array}{l} s ^ {+} = \Psi \left(\left\{\mathbf {e} _ {p, k} ^ {*}, p = 1, \dots , P; k = 1, \dots , K \right\}, \left\{\mathbf {e} _ {p, q} ^ {+}: p = 1, \dots , P; q = 1, \dots , Q \right\}\right), \\ s ^ {-} = \Psi \left(\left\{\boldsymbol {e} _ {p, k} ^ {*}: p = 1, \dots , P; k = 1, \dots , K \right\}, \left\{\boldsymbol {e} _ {p, q} ^ {-}: p = 1, \dots , P; q = 1, \dots , Q \right\}\right). \tag {2} \\ \end{array}
$$

There are many choices of measuring the similarity between two sets. In the implementation, we adopt a co-attention mechanism (Lu et al., 2016) as  $\Psi(\cdot)$ , because even within the same pattern, the subgraphs having similar node features are more important than others.

We put the concrete formulation of  $\Phi (\cdot)$  and  $\Psi (\cdot)$  in the Appendix A3.2 and A3.3.

Loss Function. For each tuple  $\langle s, r, t \rangle$ , we have the binary label  $y$  in the training dataset  $\mathcal{D}$  to denote the relation existence. We here train the model by logistic loss with negative sampling:

$$
L = - \sum_ {\langle s, r, t \rangle \in \mathcal {D}} (y \log \hat {y} + (1 - y) \log (1 - \hat {y})), \tag {3}
$$

where  $\hat{y}$  is the final prediction calculated with normalized similarity as  $\hat{y} = \frac{s^{+}}{s^{+} + s^{-}}$ .

Inference. We perform pattern matching to find the target, supporting, and refuting subgraphs on the testing graph, and compute the prediction score  $\hat{y}$  exactly as what we do in training. We predict that an edge exists if  $\hat{y}$  is larger than a threshold (as a hyper-parameter, 0.5 in our implementation).

Limitations. There are two main limitations of GraphANGEL. The first one is that GraphANGEL would not work reliably if the assumptions (intuitions) in Section 2 are violated. Examples include when node  $s$  and  $t$  are disconnected or topologically far away when  $\langle s, r, t \rangle$  is removed, so that one may not be able to find 3-cycle or 4-cycle target subgraphs. The prediction in this case can only rely on Pair patterns, i.e., comparing if the source-target pair is similar to the incident nodes of edges with relation  $r$  or not. The second one is that current online subgraph sampling algorithms are slow. To make sampling efficient, the subgraph retrieval stage requires finding and storing all 3-cycles and 4-cycles. It is a one-time preprocessing step and it can take lots of space, although the results can be stored on external storage. See further discussions on other limitations in Appendix A4.

# 4 RELATED WORK

Heterogeneous graphs such as knowledge graphs (Bordes et al., 2013; Schlichtkrull et al., 2018) and social networks (Zhang et al., 2019; Jin et al., 2020) that encode facts about the world surrounding us, have motivated work on automatically predicting new statements based on known ones. Roughly speaking, existing approaches can be summarized into three main branches, namely matrix factorization techniques (Nickel et al., 2011; 2012), statistical relational learning approaches (Richardson & Domingos, 2006; Singla & Domingos, 2005), and neural-embedding-based methods (Bordes et al., 2013; Dettmers et al., 2018). Our work focuses on the study of neural-embedding-based models as they embed nodes and relations into low-dimensional spaces (Bordes et al., 2013; Schlichtkrull et al., 2018; Trouillon et al., 2016; Yan et al., 2019; Teru et al., 2020), showing good scalability and strong generalization ability. One dominant paradigm (Ji et al., 2015; Lin et al., 2015) in this branch is

constructed based on translation (Bordes et al., 2013) or rotation (Sun et al., 2019) assumptions. The key idea behind this kind of models is that for a positive instance, the source node should be as close as possible to the target node through the relation, serving as a translation or rotation.

Although there are multiple successful stories in this line of works, these models are all trained on individual instances, regardless of their local neighborhood structures. As stated in (Schlichtkrull et al., 2018), explicitly modeling local structure can be an important supplement to help recover missing relations. Inspired by the success of GNNs in modeling structured neighborhood information, another line of literature (Schlichtkrull et al., 2018; Zhang & Chen, 2018) learns the relation embedding based on its neighborhood subgraph using graph convolution layers. With a similar approach, recent work (Zhang & Chen, 2019; Teru et al., 2020) illustrates the inductive capabilities of generalization to unseen nodes. However, these approaches fail to generalize to unseen relation types, as it is indirect to introduce the "neighborhood" for a relation type upon the graph structure.

Meanwhile, our work also relates to previous researches (Qu et al., 2019; Qu & Tang, 2019; Zhang et al., 2020; Qi et al., 2018; Zheng et al., 2019) studying to effectively combine GNNs techniques with symbolic logic rule-based approaches (Giarratano & Riley, 1998; Jackson, 1998; Lafferty et al., 2001; Taskar et al., 2012; Richardson & Domingos, 2006; Singla & Domingos, 2005). As logic rule-based approaches are able to leverage domain knowledge but suffer from complicated graph structures while graph embedding methods excel at learning effective relation embeddings but fail to leverage domain knowledge, the basic idea of these works is to combine the advantages of both methods. However, these methods, as originally proposed, are transductive in nature. Unlike our method, they still require learning relation type specific embeddings, whereas we treat the relation prediction as a graph pattern matching problem, independent of any particular relation type identity. There are also other work (Yang et al., 2014) that demonstrates the potential generalization ability of transition-based methods and designs a logical rule extraction approach to mine the underlying rules. As this method can incorporate with various neural-embedding-based models, this set of models constitute our baselines in the inductive setting.

Another popular link prediction approach is to directly infer the likelihood of relation existence using a local neighborhood or a local subgraph (Schlichtkrull et al., 2018; Zhang & Chen, 2018; Hu et al., 2020). A significant difference between those works and our approach is in the formulation of the loss function. Let  $f_{r}$  be any function that maps a vector to a scalar score parametrized by  $r$  and  $\ell_r$  be a scalar function, then the prior works' loss function often take the form  $L = \sum_{\langle s,r,t\rangle \in \mathcal{D}}\ell (f_r(\pmb {h}^*),y)$  meaning that the score computation explicitly depends on the embedding of relation to be predicted. Ours, in contrast, has the form  $L = \sum_{\langle s,r,t\rangle \in \mathcal{D}}\ell (d(\pmb {h}^*,\pmb {h}^+),d(\pmb {h}^*,\pmb {h}^-),y)$  where  $\ell$  is a scalar function and  $d$  is a distance function between two vectors, which are embeddings of a set of subgraphs. Consequently, our loss function  $\ell$  does not require representation of  $r$ .

# 5 EXPERIMENT

In this section, as the main advantage of our model is that it can generalize to relation types unseen during training without fine-tuning, we both evaluate the overall performance of GraphANGEL in standard relation prediction tasks on heterogeneous graphs and its generalizability to unseen relation types against other state-of-the-art methods. All code will be released at publication time.

# 5.1 EXPERIMENT SETUP AND COMPARED ALGORITHMS

Recommendation on Heterogeneous Graph. We evaluate our model on four heterogeneous graph benchmark datasets in various fields: LastFM (Hu et al., 2018a), Yelp (Hu et al., 2018b), Amazon (Ni et al., 2019), and Douban Book (Zheng et al., 2017). The baselines we compare against are: HetGNN (Zhang et al., 2019), HAN (Wang et al., 2019b), TAHIN (Bi et al., 2020), HGT (Hu et al., 2020), and R-GCN (Schlichtkrull et al., 2018). As the recommendation task can naturally be regarded as the relation predictions between each user and item pair, for each triplet, we formulate the task as a binary classification task. We split each dataset into  $60\%$ ,  $20\%$ , and  $20\%$  for training, validation and test sets, respectively. Following the setting of (Zhang et al., 2019), we generate an equal number of negative triplets with the same relation type in the test set, and report Area Under ROC Curve (AUC), Accuracy (ACC), and F1 score. We also compare the training and inference time of GraphANGEL against baseline models and report results in Appendix A6.4.

Table 3: Result comparisons with baselines on heterogeneous graph recommendation tasks.  

<table><tr><td rowspan="2">Models</td><td colspan="3">LastFM</td><td colspan="3">Yelp</td><td colspan="3">Amazon</td><td colspan="3">Douban Book</td></tr><tr><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td><td>AUC</td><td>ACC</td><td>F1</td></tr><tr><td>HetGNN</td><td>0.7936</td><td>0.7258</td><td>0.7177</td><td>0.9083</td><td>0.8297</td><td>0.8205</td><td>0.7744</td><td>0.7108</td><td>0.7109</td><td>0.8737</td><td>0.7912</td><td>0.7915</td></tr><tr><td>HAN</td><td>0.8915</td><td>0.8337</td><td>0.8296</td><td>0.9156</td><td>0.8488</td><td>0.8426</td><td>0.8487</td><td>0.7682</td><td>0.7572</td><td>0.9244</td><td>0.8501</td><td>0.8458</td></tr><tr><td>TAHIN</td><td>0.8910</td><td>0.8463</td><td>0.8337</td><td>0.9067</td><td>0.8490</td><td>0.8393</td><td>0.8535</td><td>0.7718</td><td>0.7644</td><td>0.9253</td><td>0.8497</td><td>0.8373</td></tr><tr><td>HGT</td><td>0.8394</td><td>0.7939</td><td>0.7882</td><td>0.9006</td><td>0.8375</td><td>0.8334</td><td>0.7125</td><td>0.6482</td><td>0.6296</td><td>0.9132</td><td>0.8364</td><td>0.8222</td></tr><tr><td>R-GCN</td><td>0.8526</td><td>0.8393</td><td>0.8341</td><td>0.9098</td><td>0.8427</td><td>0.8323</td><td>0.8130</td><td>0.7408</td><td>0.7366</td><td>0.9203</td><td>0.8413</td><td>0.8271</td></tr><tr><td>GraphANGEL3-cycle</td><td>0.8934</td><td>0.8519</td><td>0.8465</td><td>0.9167</td><td>0.8498</td><td>0.8514</td><td>0.8601</td><td>0.7746</td><td>0.7712</td><td>0.9256</td><td>0.8512</td><td>0.8479</td></tr><tr><td>GraphANGEL4-cycle</td><td>0.8961</td><td>0.8514</td><td>0.8467</td><td>0.9201</td><td>0.8506</td><td>0.8521</td><td>0.8609</td><td>0.7752</td><td>0.7716</td><td>0.9242</td><td>0.8502</td><td>0.8378</td></tr><tr><td>GraphANGEL</td><td>0.8979</td><td>0.8524</td><td>0.8469</td><td>0.9231</td><td>0.8512</td><td>0.8533</td><td>0.8611</td><td>0.7790</td><td>0.7753</td><td>0.9311</td><td>0.8601</td><td>0.8543</td></tr><tr><td>GraphANGEL*</td><td>0.9001</td><td>0.8611</td><td>0.8589</td><td>0.9337</td><td>0.8701</td><td>0.8577</td><td>0.8700</td><td>0.7810</td><td>0.7813</td><td>0.9410</td><td>0.8640</td><td>0.8591</td></tr></table>

Table 4: Result comparisons with baselines on knowledge graph completion task.  

<table><tr><td rowspan="2">Models</td><td colspan="5">FB15k-237</td><td colspan="5">WN18RR</td></tr><tr><td>MR</td><td>MRR</td><td>Hit@1</td><td>Hit@3</td><td>Hit@10</td><td>MR</td><td>MRR</td><td>Hit@1</td><td>Hit@3</td><td>Hit@10</td></tr><tr><td>pLogicNet</td><td>173</td><td>0.332</td><td>0.237</td><td>0.367</td><td>0.524</td><td>3408</td><td>0.441</td><td>0.398</td><td>0.446</td><td>0.537</td></tr><tr><td>TransE</td><td>181</td><td>0.326</td><td>0.229</td><td>0.363</td><td>0.521</td><td>3410</td><td>0.223</td><td>0.135</td><td>0.401</td><td>0.531</td></tr><tr><td>ConvE</td><td>244</td><td>0.325</td><td>0.237</td><td>0.356</td><td>0.501</td><td>4187</td><td>0.430</td><td>0.400</td><td>0.440</td><td>0.520</td></tr><tr><td>ComplEx</td><td>339</td><td>0.247</td><td>0.158</td><td>0.275</td><td>0.428</td><td>5261</td><td>0.440</td><td>0.410</td><td>0.460</td><td>0.510</td></tr><tr><td>MLN</td><td>1980</td><td>0.098</td><td>0.067</td><td>0.103</td><td>0.160</td><td>11549</td><td>0.259</td><td>0.191</td><td>0.322</td><td>0.361</td></tr><tr><td>RotatE</td><td>177</td><td>0.338</td><td>0.241</td><td>0.375</td><td>0.533</td><td>3340</td><td>0.476</td><td>0.428</td><td>0.492</td><td>0.571</td></tr><tr><td>RNNLogic</td><td>232</td><td>0.344</td><td>0.252</td><td>0.380</td><td>0.530</td><td>4615</td><td>0.483</td><td>0.446</td><td>0.497</td><td>0.558</td></tr><tr><td>ComplEx-N3</td><td>159</td><td>0.370</td><td>0.272</td><td>0.400</td><td>0.561</td><td>3452</td><td>0.491</td><td>0.440</td><td>0.500</td><td>0.581</td></tr><tr><td>GraIL</td><td>205</td><td>0.322</td><td>0.223</td><td>0.361</td><td>0.520</td><td>3539</td><td>0.401</td><td>0.352</td><td>0.438</td><td>0.501</td></tr><tr><td>GraphANGEL3-cycle</td><td>159</td><td>0.366</td><td>0.270</td><td>0.398</td><td>0.560</td><td>2919</td><td>0.492</td><td>0.463</td><td>0.497</td><td>0.590</td></tr><tr><td>GraphANGEL4-cycle</td><td>165</td><td>0.351</td><td>0.239</td><td>0.381</td><td>0.548</td><td>2914</td><td>0.493</td><td>0.465</td><td>0.502</td><td>0.587</td></tr><tr><td>GraphANGEL</td><td>152</td><td>0.372</td><td>0.275</td><td>0.407</td><td>0.563</td><td>2834</td><td>0.502</td><td>0.470</td><td>0.513</td><td>0.597</td></tr></table>

Knowledge Graph Completion. We compare different methods on two benchmark datasets: FB15k-237 (Toutanova & Chen, 2015) and WN18RR (Dettmers et al., 2018), which are constructed from Freebase (Bollacker et al., 2008) and WordNet (Miller, 1995), respectively. The baselines we compare against are: MLN (Singla & Domingos, 2005), TransE (Bordes et al., 2013), ConvE (Dettmers et al., 2018), ComplEx (Trouillon et al., 2016), pLogicNet (Qu & Tang, 2019), RotatE (Sun et al., 2019), RNNLogic (Qu et al., 2020), ComplEx-N3 (Lacroix et al., 2018) and GraIL (Teru et al., 2020). For each triplet, we mask the source or target node, and let each method predict the masked node. Following (Qu & Tang, 2019; Yang et al., 2014; Bordes et al., 2013), we use the filtered setting during evaluation on the standard training-validation-test split, and report Mean Rank (MR), Mean Reciprocal Rank (MRR), and Hit@K (K=1,3,10).

In each task, we implement GraphANGEL as we proposed in Section 3. Concretely, we imply GraphANGEL with  $\Pi_p$  upon the logical patterns shown in Table 2. For further investigations on the influence of different graph patterns, we here introduce GraphANGEL $_{3-cycle}$ , a variant of GraphANGEL without using patterns in 3-cycle shapes; and GraphANGEL $_{4-cycle}$ , another variant without using patterns in 4-cycle shapes. For recommendation tasks on heterogeneous graphs, since the number of edge types is usually small, we can enumerate all the relation type combination in each pattern. This allows us to make the pattern  $\Pi_p$  specific to relation types. We denote this variant as GraphANGEL*. More details of the datasets and experimental configurations as well as the implementation details for baselines are reported in Appendix A5.1.

# 5.2 RESULT ANALYSIS OF STANDARD TASKS

Heterogeneous Graph Based Recommendation. In recommendation scenarios, edges between user and item nodes are generally more likely to exist if they share neighboring users or items. In other words, users close in the graph may share similar interests and items close usually share similar attributes. Table 3 summarizes the performances of GraphANGEL and baselines on four different kinds of recommendation tasks. We observe that GraphANGEL significantly outperforms the baselines across all datasets in terms of AUC, ACC, and F1 metrics. Almost all prevailing baseline methods on heterogeneous graph are based on sampling through metapath. One explanation is that given a specific pattern, these metapaths can be roughly regarded as target patterns, but without constructing supporting and refuting patterns in Table 1.

Knowledge Graph Completion. In knowledge graphs, the connection between two nodes is determined by both logic and node attributes. Table 4 summarizes all experimental results. As can be seen, GraphANGEL outperforms all baselines across all datasets. One explanation is that most knowledge graph embedding techniques focus on mining the hidden information in each tuple, which

Table 5: Comparable results with baselines, where there are  $20\%$  least frequent relations. Here we only report the results in terms of Hit@K (K=1,3,10). See Appendix A6.1 for the full version. The numbers in brackets show the descent degree comparing to Table 5.  

<table><tr><td rowspan="2">Models</td><td colspan="3">FB15k-237</td><td colspan="3">WN18RR</td></tr><tr><td>Hit@1</td><td>Hit@3</td><td>Hit@10</td><td>Hit@1</td><td>Hit@3</td><td>Hit@10</td></tr><tr><td>pLogicNet</td><td>0.209(11.8%↓)</td><td>0.342(6.81%↓)</td><td>0.500(4.58%↓)</td><td>0.341(14.3%↓)</td><td>0.406(8.97%↓)</td><td>0.491(8.57%↓)</td></tr><tr><td>TransE</td><td>0.197(14.0%↓)</td><td>0.339(6.61%↓)</td><td>0.494(5.18%↓)</td><td>0.123(8.89%↓)</td><td>0.367(8.48%↓)</td><td>0.487(8.29%↓)</td></tr><tr><td>ConvE</td><td>0.207(12.7%↓)</td><td>0.324(8.99%↓)</td><td>0.478(4.59%↓)</td><td>0.364(9.00%↓)</td><td>0.391(11.1%↓)</td><td>0.479(7.88%↓)</td></tr><tr><td>ComplEx</td><td>0.140(11.4%↓)</td><td>0.261(5.09%↓)</td><td>0.409(4.44%↓)</td><td>0.375(8.54%↓)</td><td>0.428(6.96%↓)</td><td>0.475(6.86%↓)</td></tr><tr><td>MLN</td><td>0.051(23.9%↓)</td><td>0.077(25.2%↓)</td><td>0.143(10.6%↓)</td><td>0.166(13.1%↓)</td><td>0.285(11.5%↓)</td><td>0.333(7.76%↓)</td></tr><tr><td>RotatE</td><td>0.211(12.3%↓)</td><td>0.351(6.34%↓)</td><td>0.505(5.32%↓)</td><td>0.386(9.83%↓)</td><td>0.445(9.57%↓)</td><td>0.529(7.38%↓)</td></tr><tr><td>RNNLogic</td><td>0.219(13.2%↓)</td><td>0.333(12.5%↓)</td><td>0.499(5.76%↓)</td><td>0.407(8.61%↓)</td><td>0.444(10.7%↓)</td><td>0.511(8.42%↓)</td></tr><tr><td>ComplEx-N3</td><td>0.242(11.1%↓)</td><td>0.361(9.85%↓)</td><td>0.534(4.85%↓)</td><td>0.407(7.49%↓)</td><td>0.456(8.74%↓)</td><td>0.539(7.15%↓)</td></tr><tr><td>GraIL</td><td>0.197(11.5%↓)</td><td>0.315(12.8%↓)</td><td>0.484(6.84%↓)</td><td>0.307(12.7%↓)</td><td>0.387(11.6%↓)</td><td>0.458(8.52%↓)</td></tr><tr><td>GraphANGEL3-cycle</td><td>0.243(9.87%↓)</td><td>0.384(3.53%↓)</td><td>0.539(3.80%↓)</td><td>0.418(9.78%↓)</td><td>0.465(6.42%↓)</td><td>0.549(6.93%↓)</td></tr><tr><td>GraphANGEL4-cycle</td><td>0.215(9.96%↓)</td><td>0.366(4.04%↓)</td><td>0.526(3.94%↓)</td><td>0.421(9.37%↓)</td><td>0.467(6.92%↓)</td><td>0.549(6.47%↓)</td></tr><tr><td>GraphANGEL</td><td>0.248(9.62%↓)</td><td>0.394(3.27%↓)</td><td>0.541(3.84%↓)</td><td>0.429(8.74%↓)</td><td>0.481(6.21%↓)</td><td>0.557(6.75%↓)</td></tr></table>

is similar to only considering the patterns in Pair shapes in Table 1. However, other patterns contain information involving multiple relations, which enables to model the logics.

In order to better illustrate our performance of these relations with few occurrences in the training set, we solely report the results of testing each model on the  $20\%$  relations with few occurrence in Table 5. From the comparison between Tables 4 and 5, we can observe that with few shots of relations, GraphANGEL can have a better generalization ability. One reason is that embeddings of the relations with few occurrences cannot be trained with plenty of data samples, resulting in low expressive power of the relations. In contrast, GraphANGEL does not learn the embeddings directly, but learns to represent the relations of the related logics. However, it is still more challenging to model these relations that lead to a drop in performance.

![](images/da36389614c893ffb583cca87b841a9ae64aedcbe792fbdff50d2a4c41724b53.jpg)  
Figure 3: Performance change of GraphANGEL with different number of subgraphs in terms of ACC and AUC.

![](images/5d0f44d16b6e2503fb495e0ee9cd457402b108b602eebb5c0282fbb6881c87b3.jpg)

![](images/a26a09b433df06edd3977c98f4cb224421e6f27ea1997cc884608fcca4dcc2bf.jpg)

![](images/c8c9122e2dd9ac7d7f8bf00ed02ccc18f9201e9c757643f1cdcb138539638a3e.jpg)

# 5.3 EFFECT OF DIFFERENT PATTERNS

In GraphANGEL, three shapes of patterns are used. Next, we systematically investigate the effect of each graph pattern and different numbers of sampled subgraphs used for each pattern. For each dataset, we evaluate the influence of the patterns in 3-cycle and 4-cycle shapes by the performance of  $\mathrm{GraphANGEL}_{3 - cycle}$  and  $\mathrm{GraphANGEL}_{4 - cycle}$ . Since different shapes of patterns represent different composition logical rules, as shown in Table 1, patterns in Pair shape are the most general but include the least structure information, while those in 4-cycle shape are rich in the structure but less common. Hence, these patterns have their unique power in representing logics. Although it is hard to determine whether 3-cycle or 4-cycle shaped patterns are more powerful, as shown in Tables 3, 4 and 5, GraphANGEL with patterns in all shapes achieves the best performance. Besides the pattern type, we also investigate how the number of sampled subgraphs influences the performance. Taking the heterogeneous graph datasets Amazon and Douban Book as examples, we show the performance of GraphANGEL under the different  $K$  and  $Q$  in terms of ACC and AUC in Figure 3. One explanation is that the subgraphs following target patterns are constricted within the neighborhood of source and target nodes, the number of which is much smaller than subgraphs following (analogy) supporting and refuting patterns.

# 5.4 RESULT ANALYSIS OF GENERALIZATION STUDY

We further evaluate these models in a scenario where generalizing from existing relations to unseen relations is required. Concretely, we use the same datasets with knowledge graph completion and randomly split  $\mathcal{R}$  into two partitions  $\mathcal{R}_{\mathrm{seen}}$  and  $\mathcal{R}_{\mathrm{unseen}}$ . Each model is trained and validated only with the relations in  $\mathcal{R}_{\mathrm{seen}}$ . During testing, the training and validation triplets in  $\mathcal{R}_{\mathrm{unseen}}$  are added back to the original graph. We report results on test triples with relations in  $\mathcal{R}_{\mathrm{unseen}}$  only.

Table 6: Result comparisons with baselines on generalization setting by randomly removing  $20\%$  relations. Due to space limitation, we only report the results in terms of Hit@K (K=1,3,10). See Appendix A6.2 for full version and Appendix A6.2 for results of dropping  $5\%$ ,  $10\%$ ,  $15\%$ . The numbers in brackets show the descent degree.  

<table><tr><td rowspan="2">Models</td><td colspan="3">FB15k-237</td><td colspan="3">WN18RR</td></tr><tr><td>Hit@1</td><td>Hit@3</td><td>Hit@10</td><td>Hit@1</td><td>Hit@3</td><td>Hit@10</td></tr><tr><td>pLogicNet*</td><td>0.112(52.7%↓)</td><td>0.179(51.2%↓)</td><td>0.257(51.0%↓)</td><td>0.141(64.6%↓)</td><td>0.222(50.2%↓)</td><td>0.267(50.3%↓)</td></tr><tr><td>TransE*</td><td>0.101(55.9%↓)</td><td>0.163(55.1%↓)</td><td>0.246(52.8%↓)</td><td>0.072(46.7%↓)</td><td>0.200(50.1%↓)</td><td>0.260(51.0%↓)</td></tr><tr><td>ConvE*</td><td>0.104(56.1%↓)</td><td>0.178(50.0%↓)</td><td>0.247(50.7%↓)</td><td>0.201(49.8%↓)</td><td>0.223(49.3%↓)</td><td>0.268(48.5%↓)</td></tr><tr><td>ComplEx*</td><td>0.078(50.6%↓)</td><td>0.142(48.4%↓)</td><td>0.226(47.2%↓)</td><td>0.214(47.8%↓)</td><td>0.236(48.7%↓)</td><td>0.267(47.6%↓)</td></tr><tr><td>MLN*</td><td>0.031(53.7%↓)</td><td>0.049(52.4%↓)</td><td>0.070(56.3%↓)</td><td>0.092(51.8%↓)</td><td>0.154(52.2%↓)</td><td>0.178(50.7%↓)</td></tr><tr><td>RotatE*</td><td>0.121(49.8%↓)</td><td>0.187(50.1%↓)</td><td>0.271(49.1%↓)</td><td>0.238(44.3%↓)</td><td>0.260(47.1%↓)</td><td>0.296(48.2%↓)</td></tr><tr><td>RNNLogic*</td><td>0.124(50.7%↓)</td><td>0.172(54.7%↓)</td><td>0.240(54.6%↓)</td><td>0.244(45.2%↓)</td><td>0.260(47.6%↓)</td><td>0.281(49.7%↓)</td></tr><tr><td>ComplEx-N3*</td><td>0.142(47.2%↓)</td><td>0.208(49.6%↓)</td><td>0.289(48.5%↓)</td><td>0.250(43.2%↓)</td><td>0.269(46.2%↓)</td><td>0.311(46.4%↓)</td></tr><tr><td>GraIL*</td><td>0.125(43.9%↓)</td><td>0.185(48.8%↓)</td><td>0.263(49.4%↓)</td><td>0.195(44.7%↓)</td><td>0.222(49.3%↓)</td><td>0.267(46.8%↓)</td></tr><tr><td>GraphANGEL3-cycle</td><td>0.168(37.6%↓)</td><td>0.230(42.2%↓)</td><td>0.333(40.5%↓)</td><td>0.277(40.2%↓)</td><td>0.291(41.4%↓)</td><td>0.329(44.3%↓)</td></tr><tr><td>GraphANGEL4-cycle</td><td>0.147(38.7%↓)</td><td>0.222(41.7%↓)</td><td>0.328(40.2%↓)</td><td>0.278(40.2%↓)</td><td>0.291(42.1%↓)</td><td>0.326(44.4%↓)</td></tr><tr><td>GraphANGEL</td><td>0.173(37.2%↓)</td><td>0.238(41.5%↓)</td><td>0.337(40.1%↓)</td><td>0.284(39.5%↓)</td><td>0.299(41.8%↓)</td><td>0.334(44.1%↓)</td></tr></table>

We cannot directly use the baselines above for unseen relations since those relation embeddings are never trained. Therefore, we combine them with EmbedRule, which estimates the relation embedding by finding a number of most common relation sequences that cooccur with the unseen relation, and composing those embeddings thereafter (Yang et al., 2014). We superscript the name with an asterisk for models enhanced by EmbedRule (e.g.  $\mathrm{TransE^{*}}$ ).

Results reported in Table 6 illustrate that our model is significantly less affected than other models when we drop  $20\%$  relations from the training and validation sets. We additionally report the results of dropping or adding  $5\%$ ,  $10\%$ ,  $15\%$  relations in Appendix A6.2. These results on FB15k-237 dataset are summarized in Figure 4. From the better generalization ability against other baselines.

![](images/d9998eff6e59ad316731ae185299b09b6efaad1f429b2772c380f5f4cf52dbda.jpg)  
Figure 4: Illustrations of generalization ability for GraphANGEL against baselines.

# 5.5 RESULT ANALYSIS OF EXPLAINABLE ATTENTION MAP

Besides the performance, we further show that our model can produce explainable heat maps of attention scores across the discovered logic. We here provide an illustration on the recommendation task based on Douban Book graph, where we are required to predict the relation existence between each user and book pair. In Douban Book, we can define the graph patterns based on the node types, such as 4-cycle shaped patterns: User - Book - Author - Book denoted as  $(b, a)$ , User - Book - Year - Book denoted as  $(b, y)$ , and User - User - User - Book denoted as  $(u, u)$ . In Figure 5, the rows represent the supporting subgraphs while the columns represent the target subgraphs. Each cell represents the similarity between a target subgraph (at the top) and a supporting subgraph (at the bottom). The color of each cell shows the attention weight for corresponding pair of supporting and target subgraphs. We can

observe that the deep color of the cell located at target and supporting subgraphs following  $(b, a)$  patterns, which indicates that the logic User  $\wedge$  Book  $\wedge$  Author  $\wedge$  Book  $\Rightarrow$  User  $\wedge$  Book has high confidence and can be strong evidence to support the relation prediction.

![](images/8e0b388bf2ab24cc9e809603057276310673fc388a1c94514ee52af5267d5ea8.jpg)  
Figure 5: Illustrations of generated heat map of attention scores. See Appendix A6.3 for the full version with subgraph structure.

# 6 CONCLUSION

We propose a novel relation prediction framework that predicts the relations between each node pair based on the subgraph containing the pair and other subgraphs with identical graph patterns, and has a strong inductive bias for the generalization to unseen relation types. With these graph patterns, we introduce several graph pattern searching and sampling techniques, which can efficiently find subgraphs matching the patterns in triangle and quadrangle shapes. In the future, we plan to further extend GraphANGEL to more complex structures (i.e., compositional logical rules).

# REFERENCES

Pablo Barceló, Egor V Kostylev, Mikael Monet, Jorge Pérez, Juan Reutter, and Juan Pablo Silva. The logical expressiveness of graph neural networks. In International Conference on Learning Representations, 2019.  
Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.  
Ye Bi, Liqiang Song, Mengqiu Yao, Zhenyu Wu, Jianming Wang, and Jing Xiao. A heterogeneous information network based cross domain insurance recommendation system for cold start users. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2211-2220, 2020.  
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pp. 1247-1250, 2008.  
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Neural Information Processing Systems (NIPS), pp. 1-9, 2013.  
Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
Nadime Francis, Alastair Green, Paolo Guagliardo, Leonid Libkin, Tobias Lindaaker, Victor Marsault, Stefan Plantikow, Mats Rydberg, Petra Selmer, and Andrés Taylor. Cypher: An evolving query language for property graphs. In Proceedings of the 2018 International Conference on Management of Data, pp. 1433-1445, 2018.  
Joseph C Giarratano and Gary Riley. Expert systems. PWS publishing co., 1998.  
William Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf.  
Binbin Hu, Chuan Shi, Wayne Xin Zhao, and Tianchi Yang. Local and global information fusion for top-n recommendation in heterogeneous information network. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1683-1686, 2018a.  
Binbin Hu, Chuan Shi, Wayne Xin Zhao, and Philip S Yu. Leveraging meta-path based context for top-n recommendation with a neural co-attention model. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1531–1540, 2018b.  
Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. Heterogeneous graph transformer. In Proceedings of The Web Conference 2020, pp. 2704-2710, 2020.  
Peter Jackson. Introduction to expert systems. Addison-Wesley Longman Publishing Co., Inc., 1998.  
Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. Knowledge graph embedding via dynamic mapping matrix. In Proceedings of the 53rd annual meeting of the association for computational linguistics and the 7th international joint conference on natural language processing (volume 1: Long papers), pp. 687-696, 2015.  
Jiarui Jin, Jiarui Qin, Yuchen Fang, Kounianhua Du, Weinan Zhang, Yong Yu, Zheng Zhang, and Alexander J Smola. An efficient neighborhood-based interaction model for recommendation on heterogeneous graph. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 75-84, 2020.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Johannes Klicpera, Aleksandar Bojchevski, and Stephan Gunnemann. Predict then propagate: Graph neural networks meet personalized pagerank. arXiv preprint arXiv:1810.05997, 2018.  
Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30-37, 2009.  
Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. Canonical tensor decomposition for knowledge base completion. In International Conference on Machine Learning, pp. 2863-2872. PMLR, 2018.  
John Lafferty, Andrew McCallum, and Fernando CN Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. 2001.  
Harry R Lewis. Computers and intractability. a guide to the theory of np-completeness, 1983.  
Yankai Lin, Zhiyuan Liu, Huanbo Luan, Maosong Sun, Siwei Rao, and Song Liu. Modeling relation paths for representation learning of knowledge bases. arXiv preprint arXiv:1506.00379, 2015.  
Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Hierarchical question-image co-attention for visual question answering. arXiv preprint arXiv:1606.00061, 2016.  
George A Miller. Wordnet: a lexical database for english. Communications of the ACM, 38(11): 39-41, 1995.  
Pasquale Minervini, Matko Bošnjak, Tim Rosttäschel, Sebastian Riedel, and Edward Grefenstette. Differentiable reasoning on large knowledge bases and natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 5182-5190, 2020a.  
Pasquale Minervini, Sebastian Riedel, Pontus Stenetorp, Edward Grefenstette, and Tim Roktaschel. Learning reasoning strategies in end-to-end differentiable proving. In International Conference on Machine Learning, pp. 6938-6949. PMLR, 2020b.  
Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 188-197, 2019.  
Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. A three-way model for collective learning on multi-relational data. In Icml, 2011.  
Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. Factorizing yago: scalable machine learning for linked data. In WwW, 2012.  
Siyuan Qi, Wenguan Wang, Baoxiong Jia, Jianbing Shen, and Song-Chun Zhu. Learning human-object interactions by graph parsing neural networks. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 401-417, 2018.  
Meng Qu and Jian Tang. Probabilistic logic neural networks for reasoning. arXiv preprint arXiv:1906.08495, 2019.  
Meng Qu, Yoshua Bengio, and Jian Tang. Gmnn: Graph markov neural networks. arXiv preprint arXiv:1905.06214, 2019.  
Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, and Jian Tang. Rnnlogic: Learning logic rules for reasoning on knowledge graphs. arXiv preprint arXiv:2010.04029, 2020.  
Matthew Richardson and Pedro Domingos. Markov logic networks. Machine learning, 62(1-2): 107-136, 2006.  
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In European Semantic Web Conference, pp. 593-607. Springer, 2018.

Parag Singla and Pedro Domingos. Discriminative training of markov logic networks. In AAAI, volume 5, pp. 868-873, 2005.  
Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. arXiv preprint arXiv:1902.10197, 2019.  
Ben Taskar, Pieter Abbeel, and Daphne Koller. Discriminative probabilistic models for relational data. arXiv preprint arXiv:1301.0604, 2012.  
Komal Teru, Etienne Denis, and Will Hamilton. Inductive relation prediction by subgraph reasoning. In International Conference on Machine Learning, pp. 9448-9457. PMLR, 2020.  
Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd workshop on continuous vector space models and their compositionality, pp. 57-66, 2015.  
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In International Conference on Machine Learning, pp. 2071-2080. PMLR, 2016.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
Alastair J Walker. An efficient method for generating discrete random variables with general distributions. ACM Transactions on Mathematical Software (TOMS), 3(3):253-256, 1977.  
Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma, Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, and Zheng Zhang. Deep graph library: A graph-centric, highly-performant package for graph neural networks. arXiv preprint arXiv:1909.01315, 2019a.  
Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S Yu. Heterogeneous graph attention network. In The World Wide Web Conference, pp. 2022-2032, 2019b.  
Leon Weber, Pasquale Minervini, Jannes Münchmeyer, Ulf Leser, and Tim Rocktäschel. Nlprolog: Reasoning with weak unification for question answering in natural language. arXiv preprint arXiv:1906.06187, 2019.  
Bo Yan, Matthew Walker, and Krzysztof Janowicz. A time-aware inductive representation learning strategy for heterogeneous graphs. In Proceedings of the 15th International Workshop on Mining and Learning with Graphs (MLG'19) at KDD, volume 19, 2019.  
Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. arXiv preprint arXiv:1412.6575, 2014.  
Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. arXiv preprint arXiv:2104.06378, 2021.  
Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V Chawla. Heterogeneous graph neural network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 793-803, 2019.  
Muhan Zhang and Yixin Chen. Link prediction based on graph neural networks. Advances in Neural Information Processing Systems, 31:5165-5175, 2018.  
Muhan Zhang and Yixin Chen. Inductive matrix completion based on graph neural networks. arXiv preprint arXiv:1904.12058, 2019.  
Yuyu Zhang, Xinshi Chen, Yuan Yang, Arun Ramamurthy, Bo Li, Yuan Qi, and Le Song. Efficient probabilistic logic reasoning with graph neural networks. arXiv preprint arXiv:2001.11850, 2020.  
Jing Zheng, Jian Liu, Chuan Shi, Fuzhen Zhuang, Jingzhi Li, and Bin Wu. Recommendation in heterogeneous information network via dual similarity regularization. International Journal of Data Science and Analytics, 3(1):35-48, 2017.

Zilong Zheng, Wenguan Wang, Siyuan Qi, and Song-Chun Zhu. Reasoning visual dialogs with structural and partial observations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6669-6678, 2019.
