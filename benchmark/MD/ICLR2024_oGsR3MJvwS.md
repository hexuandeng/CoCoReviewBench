# GENERALIZABLE DEEP RL-BASED TSP SOLVER VIA APPROXIMATE INVARIANCE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently, deep reinforcement learning (DRL) has shown promising results for learning fast heuristics to solve traveling salesman problems (TSP). Meanwhile, most existing state-of-the-art (SOTA) DRL methods yield solvers that do not generalize well on TSP instances larger than those seen during training. However, such generalization ability is crucial in practice since training on large instances is impractical. To tackle this issue, we propose a novel DRL method, called  $\mathsf{T}\mathsf{S}^3$ , which is designed to enforce a variety of (possibly approximate) invariances to promote the generalizability of the learned solver. More specifically,  $\mathsf{T}\mathsf{S}^3$  applies a modified policy gradient algorithm enhanced with data augmentation to train a Transformer-based model to select the next city to visit among the k-nearest neighbors of the last visited city by integrating a local view and global view of a TSP instance. To further validate the capability of  $\mathsf{T}\mathsf{S}^3$ , we also propose its combination with Monte-Carlo Tree Search. Abundant experiments on random TSP and TSPLIB instances demonstrate that our propositions achieve a dominant performance when generalizing to large-sized TSPs.

# 1 INTRODUCTION

Among all combinatorial optimization problems, Traveling Salesman Problem (TSP) is arguably one of the most popular thanks notably to the simplicity of its formulation and its wide application range, such as logistics (Madani et al., 2020), electronic design automation (Alkaya & Duman, 2013), or bioinformatics (Matai et al., 2010). In this problem, given a graph, the goal is to find a shortest tour that visits all the nodes exactly once while returning to a starting node. Due to its NP-hard nature, exact algorithms are impracticable to solve large-sized instances, which motivates the active development of approximate heuristic methods. Although state-of-the-art (SOTA) heuristic methods, such as LKH3 (Helsgaun, 2009; 2017) have been designed to provide high-quality solution for large TSP instances faster than exact methods, they are still too computationally costly.

To obtain faster heuristics, researchers have started to actively explore the exploitation of deep learning, and especially deep reinforcement learning (DRL), to design TSP solvers, e.g., Attention Model (Kool et al., 2019), or PointerFormer (Jin et al., 2023), which are generally constructive (i.e., they generate the solution by iteratively select the next node to visit from the last visited one). Though this approach shows promising results, the proposed models do not generally reveal good generalization ability (Joshi et al., 2020). Indeed, most work can only achieve good performance on TSP instances whose size is close to the training instance sizes. Thus, models trained on small-sized instances are incapable of generating a satisfying solution on large-sized instances, which could only be tackled by a model trained on large-sized instances. However, such training would cost a large amount of time and computational resource, making it impractical.

Our work aims at better understanding how approximate invariance can promote cross-size generalization (omit cross-size in the rest of this article) in a DRL-based TSP solver. Figure 1 (Left) shows the distribution of the rank of the next node among the nearest neighbors of a current node in an "optimal" tour for different TSP instances. This figure suggests that an optimal tour can generally still be obtained by only focusing on the k-nearest neighbors (k-NNs) of the last visited node. Based on this observation, we propose (1) to directly restrict the action space of a DRL agent to the k-NNs of the last visited node and (2) to provide two views to this agent as its state: a local view focused on the k-NNs and a global view including all the unvisited nodes. The first idea simplifies

![](images/ef6f35faf46b4fd2eda5d4ce865c3ae6850501eb99469bc69f4b9eaff8ec2d4b.jpg)  
Figure 1: Empirical regularity observed on random TSP instances (results averaged over 1000 instances per TSP size). (Left) Distribution of the rank of the next node to visit from a node in a solution tour among the nearest neighbors of the latter node. (Right) Percentage change (gap) of the quality of solutions of instances perturbed by random noise. Note that the solutions here are produced by LKH3, which can output exact optimal solutions for small-sized instances and near-optimal solutions for large-sized instances. The dashlines in the right figure denotes the variation might be caused by the sub-optimality of LKH3, rather than the random noise.

![](images/94802ba6c7f8d1333457de49baa5680ac3b266a3dcc0f42e3ef4ea387be6bb92.jpg)

the decision-making problem by directly choosing among the most probable nodes, while the second idea allows the local and global views to be processed separately, which enables more efficient invariant preprocessing (e.g., scaling) of the k-NNs. Interestingly, the first idea can be understood as approximate invariance since focusing on the k-NNs amounts to expressing the independence with respect to the nodes that are farther away. Furthermore, Figure 1 (Right) shows how much the quality of an "optimal" solution changes (i.e., gap) when an instance is perturbed by random noise (i.e., all node positions are changed by small random noises). Note that while the solutions obtained by LKH3 are optimal with high probability, they are generally suboptimal for TSP500 and TSP1000, which indicates that the corresponding curves in dashed lines are less reliable. This figure shows that small perturbations introduce small gaps, which can be regarded as exploiting approximate invariance. This observation motivates us to apply various invariant transformations (e.g., rotation or reflection) with random perturbation to a TSP instance to generate many instances sharing similar optimal solutions.

To make these ideas operational for exploiting (approximate) invariance, we design a novel Transformer-based model for solving TSP and propose a modification of the REINFORCE algorithm to train it. In addition, we also combine our trained model with Monte-Carlo Tree Search (MCTS). Our contributions are summarized below.

- We propose a novel Transformer-based architecture combining local and global information enforcing approximate invariance in the policy. The local view includes the  $k$ -NN graph centered at the last visited node. The global view includes all nodes dependent on the optimal solution.  
- We propose a novel training method exploiting approximate invariance by involving exact-invariant operations (e.g. rotation) and approximate-invariant operation (e.g. noisy perturbation).  
- In addition, to demonstrate the quality of  $\mathsf{TS}^3$ , we formulate a generic approach  $\mathsf{TS}^4$  to derive a heatmap from a constructive method to be used in MCTS.  
- We perform comprehensive experiments to evaluate the generalizability of our method, compared with many other available methods. We also conduct an ablation study and sensitivity experiments to validate the positive effects of our design decisions.

# 2 RELATED WORK

In this section, we discuss related work and emphasize how it differs from our proposition. Here we only include those work having strong relationship with us, and more additional work over TSP can be found in Appendix B.

Transformer Structure. The Transformer (Vaswani et al., 2017) model has inspired multiple architectures proposed for solving TSP. Notably, Kool et al. (2019) design an attention model (AM) using attention layers, while Bresson & Laurent (2021) show that the original Transformer model works well on small-sized TSP. Jin et al. (2023) create a multi-pointer network, called Pointerformer, to aggregate information from all nodes. In contrast, our proposition, which exploits approximate invariance by notably focusing on nearest neighbors, uses attention layers to process and use local and global views when selecting the next node to visit.

Local and Global Information. Jiang et al. (2023) proposes a MVGCL to leverage local information using kNN on the whole graph for learning representative features by contrastive learning. This kind of  $k$ -NN usage is applied on the complete graph, which is different from the  $k$ -NN during tour construction. Gao et al. (2023) explores a local view based on a  $k$ -NN during construction and combines the outputs of a local policy and a global policy. Our design uses only one local policy, which enforces a  $k$ -NN approximate invariance over the policy, and the aggregation of the local and global views is performed in the embedding space.

Data Augmentation and Invariance. Kwon et al. (2020) achieve with POMO a great performance gain by applying data augmentation to handle multiple trajectories at both the training phase and the test phase. Ouyang et al. (2021) demonstrate with eMAGIC that exploiting invariance (via data augmentation and preprocessing) can help generalization. Kim et al. (2022) further develops a formal algorithm to learn invariant policy for combinatorial optimizations by rotation augmentation. In addition to exact invariance, our method also exploits approximate invariance both when applying data augmentation and in the architecture design to further enhance generalizability.

Monte-Carlo Tree Search. Fu et al. (2021) train a specific model to predict probability heatmaps of TSP instances, and then utilize MCTS to optimize the solutions. Their MCTS samples a k-opt local move according to a probability heatmap over all edges, which is updated by the performance of sampled local moves. Our proposition  $\mathsf{T}\mathsf{S}^4$  follows the same MCTS scheme, but we construct the initial solutions and the probability heatmaps from our model.

# 3 BACKGROUND

We first specify some mathematical notations used in the rest of this article. Basic knowledge of TSP and DRL is then recalled, and the problem is finally formalized.

Notations. For any positive integer  $n \in \mathbb{N}$ ,  $[n]$  denotes the set  $\{1, 2, \dots, n\}$ . Set  $\mathbb{S}_n$  represents the set of all permutations of  $[n]$ , where a permutation is denoted  $\sigma = (\sigma_1, \dots, \sigma_n)$ . By extension, for any finite set  $\mathcal{X} = \{x_1, x_2, \dots, x_n\}$ ,  $\sigma(\mathcal{X}) = (x_{\sigma_1}, \dots, x_{\sigma_n})$  denotes a permutation of the elements of  $\mathcal{X}$ . For any finite set  $\mathcal{X}$ ,  $\Delta(\mathcal{X})$  denotes the set of probability distributions over  $\mathcal{X}$ .

# 3.1 EUCLIDEAN TSP

A Euclidean TSP instance can be described as a set  $\mathcal{C} = \{c_1, c_2, \ldots, c_n\}$  of cities, where each city  $c_i$  has coordinates  $(x_i, y_i) \in [0, 1]^2$ . This set induces a graph  $\mathcal{G} = (\mathcal{C}, \mathcal{E})$ , where  $\mathcal{E} = \{e_{i,j} \mid i, j \in [n]\}$  denotes the set of edges, and each edge  $e_{i,j} = \{c_i, c_j\}$  has a cost defined as the Euclidean distance between cities  $c_i$  and  $c_j$ :  $D(e_{i,j}) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$ . In a TSP instance, the salesman has to visit all the cities exactly once and return to its initial city, while minimizing the travelled distance. Formally, such a sequence of city visits is called a tour, which can be encoded as a permutation  $\sigma$  of the city indices. The length of a tour can be calculated as:

$$
L _ {\mathcal {C}} (\boldsymbol {\sigma}) = D \left(c _ {\sigma_ {n}}, c _ {\sigma_ {1}}\right) + \sum_ {i = 1} ^ {n - 1} D \left(c _ {\sigma_ {i}}, c _ {\sigma_ {i + 1}}\right). \tag {1}
$$

Therefore, solving a TSP instance  $\mathcal{C}$  amounts to finding an optimal tour  $\sigma^{*}$  with minimal length.

# 3.2 MDP AND RL

Reinforcement learning is based on the Markov Decision Process (MDP) model, which can be described by a tuple  $\mathcal{M} = (\mathcal{S},\mathcal{A},P,r,\gamma ,d_0)$ , where  $\mathcal{S}$  and  $\mathcal{A}$  represent a state space and an action

space respectively,  $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$  is a transition function,  $r: \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$  is a reward function,  $\gamma \in (0,1]$  is a discount factor, and  $d_0 \in \Delta(\mathcal{S})$  is an initial state distribution. A (stochastic) policy  $\pi: \mathcal{S} \to \Delta(\mathcal{A})$  selects an action stochastically given a current state.

The objective of reinforcement learning is to find an optimal policy  $\pi^{*}$  to maximize the total expected discounted rewards (assuming episodic problems with horizon  $T$ ):

$$
J (\pi) = \mathbb {E} _ {s _ {0} \sim d _ {0}, a _ {t} \sim \pi (\cdot | s _ {t}), s _ {t + 1} \sim P (s _ {t}, a _ {t}, \cdot)} [ G ] \quad \text {w h e r e} \quad G = \sum_ {t = 0} ^ {T - 1} \gamma^ {t} r (s _ {t}, a _ {t}), \tag {2}
$$

where value  $G$  is the so-called episodic return.

# 3.3 FORMALIZING TSP AS AN RL PROBLEM

Constructive methods solve a combinatorial optimization problem by generating a whole solution step by step. In Euclidean TSP, such methods start from an initial node, and repeatedly select the next node to visit, until all nodes have been visited. This iterative process can be formalized as an MDP. At time step  $t$ , a state  $s_t = (\sigma(\mathcal{V}_t), \mathcal{C})$  consists of a current partial tour  $\sigma(\mathcal{V}_t) = (c_{\sigma_1}, \dots, c_{\sigma_t})$  (i.e., permutation of already-visited nodes in  $\mathcal{V}_t \subset \mathcal{C}$ ) and a TSP instance  $\mathcal{C}$ . Thus, the state space is defined as  $S = \{ (\sigma(\mathcal{V}_t), \mathcal{C}) \mid t \in [n], \mathcal{V}_t \subseteq \mathcal{C}, |\mathcal{V}_t| = t, \sigma \in \mathbb{S}_t \}$ . In a state  $s_t$ , an action is any node  $c_i$  in the set of unvisited nodes  $\mathcal{U}_t = \mathcal{C} \setminus \mathcal{V}_t$ . Thus, the action space is simply  $\mathcal{A} = \mathcal{C}$ , but in a state  $s_t$ , only actions in  $\mathcal{U}_t$  are allowed. In this MDP, transitions are deterministic: selecting  $c_i \in \mathcal{U}_t$  in state  $s = (\sigma(\mathcal{V}_t), \mathcal{C})$  leads to a new state where  $c_i$  is removed from  $\mathcal{U}_t$  and appended to current partial tour  $\sigma(\mathcal{V}_t)$ . The reward here is simply the negative cost of the newly-added edge  $e_{\sigma_t,i}$ . Thus, with the discount factor set to  $\gamma = 1$ , episodic returns equal negative tour distances. By maximizing the expected returns, we learn a policy to minimize the tour length, ensuring the consistency between the objective of MDP and the objective of TSP.

As noticed by Kool et al. (2019), the optimal choice for the next city to visit is independent of the intermediate cities visited between the first visited node  $c_{\sigma_1}$  and the last visited node  $c_{\sigma_t}$ , although they still use the previously-defined state space  $S$  as input since their encoder depends on the whole instance  $\mathcal{C}$ . Building on this idea, Ouyang et al. (2021) instead reduce the state space to  $S = \{(c_{\sigma_1}, c_{\sigma_t}, \mathcal{U}_t) \mid t \in [n], \mathcal{V}_t \subseteq \mathcal{C}, |\mathcal{V}_t| = t, \mathcal{U}_t = \mathcal{C} \setminus \mathcal{V}_t, \sigma \in \mathbb{S}_t\}$ , allowing the embeddings of unvisited nodes to be independent of visited ones. In this paper, we also use this reduced state space.

# 4 ARCHITECTURE

Our proposed deep RL method trains a differentiable architecture called  $\mathsf{TS}^2$  (Transformer Structured TSP Solver). It is composed of three components: local encoder, global encoder, and decoder. Figure 2 visualizes the model architecture, whose details are presented in the following paragraphs.

As explained in Section 1, an RL agent can be viewed as approximately invariant with respect to the  $k$ -NNs of the last visited node based on observations. Following this idea, we reduce the action set  $\mathcal{U}_t$  to a set  $\mathcal{U}_t^{\mathrm{knn}}$  containing the  $k$ -NN of the last visited node. To focus on local information, the local encoder processes as inputs a partial state, which we call local  $k$ -NNs, defined as  $(c_{\sigma_t}, \mathcal{U}_t^{\mathrm{knn}})$ . Since choosing the next node to visit only based on local information may be insufficient, we also introduce a global encoder. The global encoder, receiving the whole state  $s = (c_{\sigma_1}, c_{\sigma_t}, \mathcal{U}_t)$  as inputs, is designed to compensate for this information loss. We denote nodes in the  $k$ -NNs as  $\mathcal{U}_t^{\mathrm{knn}} = \{c_{\varsigma_1}, \dots, c_{\varsigma_k}\}$ . The decoder takes as inputs the embeddings obtained from the two encoders and selects the next node to visit from  $\mathcal{U}_t^{\mathrm{knn}}$ . Before explaining in details those components, we recall the definition of attention layer, which is the basic building blocks used in both encoders and decoder.

Attention Layer. Vaswani et al. (2017) propose a well-known encoder architecture built on attention layers. Each attention layer is equipped with residual mechanism and is composed of two key components: Multi-Head Attention (MHA) Layer and Feed Forward (FF) layer. Given the input embeddings  $E_{\mathrm{enc}}^{(\ell)}$  of the  $\ell$ -th layer and embeddings  $E_{Q}, E_{K}, E_{V}$ , the MHA layer and the FF layer

![](images/1ed51aec7fd567bf0bb9a3b01dfac2e88539544d98221a349a43715d63666dae.jpg)  
Figure 2: The overall architecture of  $\mathsf{TS}^2$ . The input state is processed by the local encoder and the global encoder. The local encoder, aimed to provide representative local information, takes the scaled local  $k$ -NNs as inputs. The global encoder, aimed to compensate the information loss of the local encoder, take the scaled state as inputs. The decoder processes the merged information from the two encoders and outputs a final probability. The  $\mathsf{TS}^2$  samples a node to visit by the output probabilities and updates the partial tour in an auto-regressive manner, until a complete feasible tour is constructed. As for the first visited node, a learnable start placeholder is applied to find the suitable alternative. Best viewed in colors.

process the information as follows:

$$
\boldsymbol {Q} = \operatorname {L i n e a r} \left(\boldsymbol {E} _ {\boldsymbol {Q}}\right), \quad \boldsymbol {K} = \operatorname {L i n e a r} \left(\boldsymbol {E} _ {\boldsymbol {K}}\right), \quad \boldsymbol {V} = \operatorname {L i n e a r} \left(\boldsymbol {E} _ {\boldsymbol {V}}\right) \tag {3}
$$

$$
\boldsymbol {E} _ {\mathrm {a t t}} ^ {(\ell + 1)} = \mathrm {B N} \left(\boldsymbol {E} _ {\mathrm {e n c}} ^ {(\ell)} + \operatorname {S o f t m a x} \left(\frac {\boldsymbol {Q} \boldsymbol {K} ^ {\top}}{\sqrt {d _ {\boldsymbol {K}}}}\right) \boldsymbol {V}\right), \tag {4}
$$

$$
\boldsymbol {E} _ {\text {e n c}} ^ {(\ell + 1)} = \mathrm {B N} \left(\boldsymbol {E} _ {\text {a t t}} ^ {(\ell + 1)} + \mathrm {F F} \left(\boldsymbol {E} _ {\text {a t t}} ^ {(\ell + 1)}\right)\right), \tag {5}
$$

where  $Q, K, V$ , called queries, keys and values, are calculated by a linear layer from  $E_{Q}, E_{K}, E_{V}$  respectively. Scaling factor  $d_{K}$  is the feature dimension of the keys to normalize the inputs  $QK^{\top}$  of the softmax function. The outputs of the attention layer can be viewed as information aggregation for  $Q$  from  $K, V$ . The aggregated information for  $Q$  is directly added by the inputs  $E_{\mathrm{enc}}^{(\ell)}$  following the residual mechanism. To make the two terms consistent, it is common to set  $E_{Q} = E_{\mathrm{enc}}^{(\ell)}$  when applying the attention layer.

Local Encoder. Guided by approximate invariance, the local encoder processes as inputs the local  $k$ -NNs  $(c_{\sigma_t},\mathcal{U}_t^{\mathrm{knn}})$ . This enables the local encoder to process a small fixed-sized group of nodes extracted from the original state. However, the density of nodes increases as the size of TSP increases. Higher density results in shorter distances between nodes, which makes it hard to generalize from small-sized TSP instances to large-sized TSP instances. To maintain the approximate invariance with respect to local information among different TSP sizes,  $\mathrm{TS}^2$  scales the coordinates of the nodes in the local  $k$ -NNs to a unit square  $[0,1]^2$ . The scaled coordinates of these nodes are then processed as inputs by a linear embedding layer followed by multiple attention layers shown as follows:

$$
\boldsymbol {E} ^ {(1)} = \text {L i n e a r} \left(\operatorname {S c a l e} \left(c _ {\sigma_ {t}}, \mathcal {U} _ {t} ^ {\mathrm {k n n}}\right)\right), \tag {6}
$$

$$
\boldsymbol {E} ^ {(\ell + 1)} = \text {A t t e n t i o n} \left(\boldsymbol {E} ^ {(\ell)}, \boldsymbol {E} _ {\boldsymbol {Q}} = \boldsymbol {E} _ {\boldsymbol {K}} = \boldsymbol {E} _ {\boldsymbol {V}} = \boldsymbol {E} ^ {(\ell)}\right). \tag {7}
$$

We call the outputs of the local encoder the local embeddings  $\pmb{E}^{\mathrm{loc}} = (\pmb{E}_{\sigma_t}^{\mathrm{loc}},\pmb{E}_{\varsigma_1}^{\mathrm{loc}},\dots ,\pmb{E}_{\varsigma_k}^{\mathrm{loc}})$  (assuming  $\mathcal{U}_t^{\mathrm{knn}} = \{c_{\varsigma_1},\dots ,c_{\varsigma_k}\}$ ), where  $\pmb{E}_{\sigma}$  is abused to represent the embeddings corresponding to node  $c_{\sigma}$ .

Global Encoder. The global encoder takes the whole state  $s = (c_{\sigma_1}, c_{\sigma_t}, \mathcal{U}_t)$  as inputs. Similar to the local encoder, the global encoder also scales the coordinates and processes them with a linear embedding layer followed by multiple attention layers. Since the final action space is reduced to

$\mathcal{U}_t^{\mathrm{knn}}$ , the global encoder only provides the embeddings for  $c_{\sigma_1}$ ,  $c_{\sigma_t}$  and nodes in  $\mathcal{U}_t^{\mathrm{knn}}$  to improve computation efficiency. To achieve this, the queries are reduced to the embeddings corresponding to these nodes, which leads to the absence of the embeddings corresponding to  $\mathcal{U}_t \setminus \mathcal{U}_t^{\mathrm{knn}}$  in the layer outputs. However, if we process the embeddings same as in the local encoder where  $E_K = E_V = E^{(\ell)}$ , the global encoder no longer aggregates the information from embeddings corresponding to  $\mathcal{U}_t \setminus \mathcal{U}_t^{\mathrm{knn}}$ , because they are not provided in the layer outputs  $E^{(\ell)}$ . To address this problem, we reuse the embeddings from the linear embedding layer to be keys and values for the attention layers. Formally, the whole state  $s = (c_{\sigma_1}, c_{\sigma_t}, \mathcal{U}_t)$  is processed by the global encoder as follows:

$$
\boldsymbol {E} ^ {(1)} = \operatorname {L i n e a r} \left(\operatorname {S c a l e} \left(c _ {\sigma_ {1}}, c _ {\sigma_ {t}}, \mathcal {U} _ {t}\right)\right), \tag {8}
$$

$$
\boldsymbol {E} ^ {(\ell + 1)} = \text {A t t e n t i o n} (\boldsymbol {E} ^ {(\ell)}, \boldsymbol {E} _ {\boldsymbol {Q}} = \left(\boldsymbol {E} _ {\sigma_ {1}} ^ {(\ell)}, \boldsymbol {E} _ {\sigma_ {t}} ^ {(\ell)}, \boldsymbol {E} _ {\varsigma_ {1}} ^ {(\ell)}, \dots , \boldsymbol {E} _ {\varsigma_ {k}} ^ {(\ell)}\right), \boldsymbol {E} _ {\boldsymbol {K}} = \boldsymbol {E} _ {\boldsymbol {V}} = \boldsymbol {E} ^ {(1)}. \tag {9}
$$

The global encoder provides the global embeddings  $\pmb{E}^{\mathrm{glo}} = (\pmb{E}_{\sigma_1}^{\mathrm{glo}}, \pmb{E}_{\sigma_t}^{\mathrm{glo}}, \pmb{E}_{\zeta_1}^{\mathrm{glo}}, \dots, \pmb{E}_{\zeta_k}^{\mathrm{glo}})$ .

Decoder. The decoder merges the local embeddings and the global embeddings, and outputs the final probabilities for node selection by a linear embedding layer followed by multiple attention layers. The decoder concatenates the local embeddings and the global embeddings corresponding to each node. Specifically, the global embeddings of the first visited node are directly concatenated with the embeddings of the last visited node. Then taking the concatenated embeddings of the first/last visited nodes as queries and the remaining concatenated embeddings as keys and values, the decoder applies attention layers.

$$
\boldsymbol {E} ^ {(1)} = \operatorname {L i n e a r} \left(\operatorname {C o n c a t} \left(\boldsymbol {E} _ {\sigma_ {t}} ^ {\text {l o c}}, \boldsymbol {E} _ {\sigma_ {t}} ^ {\text {g l o}}, \boldsymbol {E} _ {\sigma_ {1}} ^ {\text {g l o}}\right)\right), \tag {10}
$$

$$
\boldsymbol {E} _ {\mathrm {k n n}} = \operatorname {L i n e a r} \left(\left\{\operatorname {C o n c a t} \left(\boldsymbol {E} _ {\varsigma_ {i}} ^ {\text {l o c}}, \boldsymbol {E} _ {\varsigma_ {i}} ^ {\text {g l o}}\right) \right\} _ {i = 1} ^ {k}\right), \tag {11}
$$

$$
\boldsymbol {E} ^ {(\ell + 1)} = \text {A t t e n t i o n} \left(\boldsymbol {E} ^ {(\ell)}, \boldsymbol {E} _ {\boldsymbol {Q}} = \boldsymbol {E} ^ {(\ell)}, \boldsymbol {E} _ {\boldsymbol {K}} = \boldsymbol {E} _ {\boldsymbol {V}} = \boldsymbol {E} _ {\mathrm {k m n}}\right), \tag {12}
$$

and the probabilities for next node selection are computed by a softmax over the output embeddings  $\pmb{E}^{\mathrm{dec}} = (E_{\varsigma_1}^{\mathrm{dec}}, \dots, E_{\varsigma_k}^{\mathrm{dec}})$ .

# 5 ALGORITHM

We propose  $\mathsf{TS}^3$  (for  $\mathsf{TS}^2 +$  Transformed Samples) by developing a deep RL algorithm to train our  $\mathsf{TS}^2$  model. We also develop  $\mathsf{TS}^4$  (for  $\mathsf{TS}^3 +$  Tree Search) by integrating the MCTS algorithm with our proposed deep RL methods. Appendix A.2 provides the algorithms written in pseudo-codes, which will be presented in this section.

# 5.1 ALGORITHM FOR TS

As explained in Section 1, the RL agent should be (approximately) invariant with respect to some transformations (e.g., Euclidean symmetry or random noise). Therefore, we utilize data augmentation to learn approximate invariance. Our algorithm is based on the REINFORCE algorithm, which is adopted in many previous papers on TSP.

REINFORCE. Assuming a parametric policy space  $\{\pi_{\theta} \mid \theta \in \Theta\}$ , the REINFORCE (Williams, 1992) algorithm optimizes eq. (2) via gradient ascent using the following gradient (so called stochastic policy gradient (Sutton et al., 1999)):

$$
\nabla_ {\theta} J (\pi_ {\theta}) = \mathbb {E} _ {\eta} \left[ (G - B) \nabla_ {\theta} \log p _ {\theta} (\eta) \right] \tag {13}
$$

where  $\eta$  is the episodic trajectory with respect to distributions defined by  $\pi_{\theta}, P, d_0; p_{\theta}(\eta)$  is the joint probability of the  $\eta$ ; and  $B$ , called baseline, is to reduce the variance of the policy gradient.

Similar to Kool et al. (2019), a baseline model with parameters  $\theta^{\mathrm{BL}}$ , which shares the same architecture with the train model with parameters  $\theta$ , is used to calculate baseline  $B$ .

Data Augmentation. The augmentation function  $f \in \mathcal{F}$  includes operations of rotation, reflection, scaling, and noisy perturbation, with detailed presentations available in Appendix C.1. Given an original instance, we sample an augmentation function from  $\mathcal{F}$  randomly for every generation of

![](images/0452d77060f554b910ab330e2a82a5d07d92519ae60b4cd332f11736694001e2.jpg)  
Figure 3: Data augmentation for  $\mathsf{T}\mathbb{S}^3$ . For a given TSP instance  $\mathcal{C}$ , more instances are generated by sampled augmentation functions. Solutions of all instances inferred by our  $\mathsf{T}\mathbb{S}^2$  model are evaluated over  $\mathcal{C}$ . Average (resp. minimum) tour length represents the episodic return (resp. best result) of the input instance  $\mathcal{C}$  for training (resp. evaluation). Notations on the figure are the same as in the text.

augmented instances. Augmented instances of number  $\omega$  are generated for each original instance by  $\omega$  sampled augmentation functions.

To generate the tours for TSP instances during training, the train model applies probabilistic sampling and the baseline model applies deterministic policy. In order to learn approximate invariance with respect to our defined data augmentation, tours for augmented instances are evaluated on there corresponding original instances. As discussed in Section 3.3, the episodic return and the baseline are derived by the negative tour lengths, with respect to tours generated by the train model and the baseline model respectively. Figure 3 visualizes the procedures described above. Formally, given a sampled TSP instance  $\mathcal{C}$ , our algorithm calculates:

$$
G = L _ {\mathcal {C}} \left(\boldsymbol {\sigma} \left(\pi_ {\theta}, \mathcal {C}\right)\right) + \omega \mathbb {E} _ {f \in \mathcal {F}} \left[ L _ {\mathcal {C}} \left(\boldsymbol {\sigma} \left(\pi_ {\theta}, f (\mathcal {C})\right)\right) \right], \tag {14}
$$

$$
B = L _ {\mathcal {C}} \left(\sigma \left(\mu_ {\theta^ {\mathrm {B L}}}, \mathcal {C}\right)\right) + \omega \mathbb {E} _ {f \in \mathcal {F}} \left[ L _ {\mathcal {C}} \left(\sigma \left(\mu_ {\theta^ {\mathrm {B L}}}, f (\mathcal {C})\right)\right) \right]. \tag {15}
$$

We use  $\sigma(\pi, \mathcal{C})$  to denote the tour generated by policy  $\pi$  on instance  $\mathcal{C}$ . Policy  $\mu_{\theta^{\mathrm{BL}}}$  is the deterministic policy induced by  $\pi_{\theta^{\mathrm{BL}}}$  by selecting the node with maximum probability.

Baseline Model Update. After several policy updates, we would utilize a batch of instances to evaluate the performance of the models. By comparing the average performance of the train model and the baseline model, the baseline model will copy the parameters from the train model if the performance difference is less than a tolerance  $\varepsilon$ . This implicitly indicates that the baseline model is the best-so-far train model during training.

To evaluate one instance by our model, number of  $\omega_{\mathrm{eval}}$  extra instances are augmented using sampled augmentation functions. We use the model to generate  $\omega_{\mathrm{eval}} + 1$  tours on the original instance and its augmented instances by deterministic policy. Calculating the tour lengths over the original instance, we select the best tour to be the final solution.

# 5.2 ALGORITHM FOR TS

Local search methods can further improve the performance of solutions given by constructive methods. We adapt the Monte-Carlo Tree Search (MCTS) based scheme (Fu et al., 2021) into  $\mathrm{TS}^3$ .

Heatmap Generation. To adapt probabilistic sampling in MCTS, we need to define a probability heatmap for the given instance, assigned to every edge belonging to the complete graph. Given a set of solutions  $\{\sigma_q\}_{q=0}^{\omega_{\mathrm{eval}}}$  obtained from model evaluation over augmented instances, the heatmap is calculated as:

$$
\Pr \left(e _ {i j}\right) \propto \mathbf {1} _ {i j} ^ {\mathrm {k n n}} + \frac {\beta}{\omega_ {\mathrm {e v a l}} + 1} \sum_ {q = 0} ^ {\omega_ {\mathrm {e v a l}}} \mathbf {1} _ {i j} ^ {\sigma_ {q}}, \quad \sum_ {j = 1} ^ {n} \Pr \left(e _ {i j}\right) = 1, \tag {16}
$$

where  $\mathbf{1}_{ij}^{\mathrm{knn}}$  is an indicator function, indicating whether node  $c_{j}$  is in the top- $k$ -th nearest node of  $c_{i}$ , and  $\mathbf{1}_{ij}^{\sigma_q}$  is also an indicator function indicating whether edge  $e_{ij}$  exists in tour  $\sigma_q$ . We use a weighting parameter  $\beta$  to balance the two terms.

Table 1: Comparisons on TSP-Random evaluations by model trained on TSP50 for TS  ${}^{3}$  .  

<table><tr><td>Category</td><td colspan="2">TSP-20</td><td colspan="2">TSP-50</td><td colspan="2">TSP-100</td><td colspan="2">TSP-200</td><td colspan="2">TSP-500</td><td colspan="2">TSP-1000</td></tr><tr><td>LKH3(100)</td><td colspan="2">0.13788 s</td><td colspan="2">1.78567 s</td><td colspan="2">11.34500 s</td><td colspan="2">65.01146 s</td><td colspan="2">553.47642 s</td><td colspan="2">2754.31533 s</td></tr><tr><td>Measurements</td><td>gap (%)</td><td>time (s)</td><td>gap (%)</td><td>time (s)</td><td>gap (%)</td><td>time (s)</td><td>gap (%)</td><td>time (s)</td><td>gap (%)</td><td>time (s)</td><td>gap (%)</td><td>time (s)</td></tr><tr><td>*DIMES</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>/</td><td>14.38</td><td>0.497</td><td>14.97</td><td>1.116</td></tr><tr><td>AM</td><td>0.97</td><td>0.009</td><td>1.72</td><td>0.022</td><td>4.87</td><td>0.043</td><td>13.54</td><td>0.084</td><td>30.37</td><td>0.212</td><td>45.04</td><td>0.432</td></tr><tr><td>POMO</td><td>0.01</td><td>0.007</td><td>0.03</td><td>0.015</td><td>0.68</td><td>0.027</td><td>9.08</td><td>0.111</td><td>30.91</td><td>1.319</td><td>45.04</td><td>9.688</td></tr><tr><td>TSP Transformer</td><td>0.31</td><td>0.009</td><td>0.27</td><td>0.021</td><td>2.57</td><td>0.040</td><td>14.42</td><td>0.076</td><td>43.35</td><td>0.186</td><td>68.20</td><td>0.391</td></tr><tr><td>PointerFormer</td><td>0.01</td><td>0.020</td><td>0.02</td><td>0.054</td><td>0.53</td><td>0.086</td><td>8.18</td><td>0.151</td><td>19.97</td><td>1.089</td><td>27.03</td><td>7.909</td></tr><tr><td>AttGCRN</td><td>0.60</td><td>0.013</td><td>28.21</td><td>0.031</td><td>50.97</td><td>0.061</td><td>56.27</td><td>0.237</td><td>85.43</td><td>0.558</td><td>118.05</td><td>1.070</td></tr><tr><td>TS3</td><td>0.03</td><td>0.099</td><td>0.51</td><td>0.190</td><td>1.82</td><td>0.444</td><td>3.98</td><td>0.735</td><td>6.68</td><td>1.926</td><td>8.10</td><td>5.578</td></tr></table>

The first term forces the MCTS algorithm to consider all nodes in the  $k$ -NNs. The second term represents the frequency of each edge selected by the models among different augmented instances. Intuitively, we construct the heatmap to guide the search algorithm over the reduced action space, based on the preferences by the model.

# 6 EXPERIMENTAL RESULTS

# 6.1 EXPERIMENTAL SETUP

To demonstrate the performance of our method, we design a series of experiments on different datasets and compare the performance with other baselines. Codes can be found in Appendix A.1 for reproduction.

In our experiments, we consider two metrics: the solution gap and the inference time. The solution gap is computed by the average gap to the optimal solutions among all generated instances, and the inference time is computed by the average inference time per instance. Parallelization is not explored, meaning that all methods only proceed with one instance at one run. To ensure the fairness of comparisons, all baseline methods and our method are tested under the same machine (a single Intel Core i7-12700 CPU and a single RTX 3060 GPU). We use the same algorithm used in baseline model update to evaluate instances for  $\mathsf{TS}^3$ . Full configurations of our settings can be found in Appendix A.3, and it takes about 3 days to train a model following. For some baseline methods marked with *, it means that we directly take the reported data from others' paper.

Dataset. Our experiments focus on two datasets: TSP-Random and TSPLIB. TSP-Random consists of instances generated by uniformly sampling a specific number of nodes within the range of  $[0,1]^2$ . It includes six sets of TSPs, with different sizes  $n = 20,50,100,200,500,1000$ , each with 1000 instances. We utilize LKH3 (Helsgaun, 2009) to solve those random instances and to calculate the gap. TSPLIB (Reinelt, 1991), as a well-known TSP library that contains different sizes of TSP instances for practical applications, is also included. In our experiments, we consider the same test set as PointerFormer (Jin et al., 2023).

Baselines. To get a comparison of generalization ability, all of the models is trained on TSP50 in our experiments. Our baselines include AM (Kool et al., 2019), POMO (Kwon et al., 2020), TSP Transformer (Bresson & Laurent, 2021), PointerFormer (Jin et al., 2023), Att-GCRN (Fu et al., 2021) and DIMES (Qiu et al., 2022), compared with our  $\mathsf{T}\mathsf{S}^3$ . For those models considering multiple decoding strategies, we assume to use multiple optima (if available) or sampling.

# 6.2 PERFORMANCE ANALYSIS

Performance on TSP-Random. As shown in Table 1, PointerFormer achieves the best performance when the size of TSP is less than or equal to 100. Compared to PointerFormer, our model,  $\mathsf{T}\mathsf{S}^3$ , achieves close but worse performance on small-sized TSP. However, when applied to large-sized TSP,  $\mathsf{T}\mathsf{S}^3$  can achieve much better performance with low increase on inference time.

Performance on TSPLIB. Table 6.2 demonstrates the overall performance of baseline models and our models on TSPLIB. Results marked with \* are those reported by Jin et al. (2023), and the TSPLIB results for PointerFormer corresponds to Model100, instead of Model50 in TSP-Random. Similar to the results on TSP-Random, our  $\mathrm{TS}^3$  achieves a

Table 2: TSPLIB evaluations.  

<table><tr><td>Model</td><td>1~100</td><td>101~500</td><td>501~1002</td></tr><tr><td>*AM</td><td>15.36%</td><td>78.18%</td><td>139.02%</td></tr><tr><td>*POMO</td><td>1.20%</td><td>6.99%</td><td>26.93%</td></tr><tr><td>*PointerFormer</td><td>1.33%</td><td>5.43%</td><td>18.65%</td></tr><tr><td>TS³</td><td>2.04%</td><td>4.73%</td><td>8.57%</td></tr></table>

slight worse performance on small-sized TSPs while keeping a dominant performance on large-sized TSPs. The detailed results for TSPLIB instances can be checked in Appendix A.4.

# 6.3 ABLATION STUDY AND SENSITIVITY ANALYSIS

Ablation Study. As shown in Section 4 and Section 5, our  $\mathrm{TS}^3$  involves different components and mechanisms to improve the performance. Table 3 gives a view of the effects of using those components and mechanisms. The dramatic decrease of the performance when dropping the components in  $\mathrm{TS}^2$  reflects their significance, especially for the scale operation and the local encoder. For other mechanisms in  $\mathrm{TS}^3$ , it can be observed that the overall performance increases after applying this mechanism.

Sensitivity Analysis. Several hyperparameters are involved in our model training, including the number of global layers  $\gamma_{\mathrm{global}}$ , the number of local layers  $\gamma_{\mathrm{local}}$ , the number of

Table 3: Ablation study and sensitivity analysis.  

<table><tr><td>Hyperparameters</td><td>TSP50</td><td>TSP200</td><td>TSP1000</td></tr><tr><td>TS3</td><td>0.51%</td><td>3.98%</td><td>8.10%</td></tr><tr><td>w/o Scale</td><td>0.59%</td><td>4.92%</td><td>23.51%</td></tr><tr><td>w/o Global Encoder</td><td>0.84%</td><td>6.50%</td><td>12.39%</td></tr><tr><td>w/o Local Encoder</td><td>1.50%</td><td>11.39%</td><td>36.55%</td></tr><tr><td>w/o Augmentation</td><td>0.61%</td><td>4.65%</td><td>9.95%</td></tr><tr><td>γlocal = 4</td><td>0.63%</td><td>4.24%</td><td>8.99%</td></tr><tr><td>γglobal = 2</td><td>0.61%</td><td>4.19%</td><td>8.63%</td></tr><tr><td>γheads = 4</td><td>0.66%</td><td>4.22%</td><td>8.44%</td></tr><tr><td>ω = 15</td><td>0.70%</td><td>4.27%</td><td>8.20%</td></tr><tr><td>k = 15</td><td>0.53%</td><td>4.11%</td><td>8.48%</td></tr><tr><td>ntrain = 30</td><td>0.56%</td><td>4.55%</td><td>10.10%</td></tr><tr><td>ntrain = 100</td><td>0.52%</td><td>2.59%</td><td>7.76%</td></tr></table>

heads  $\gamma_{\mathrm{heads}}$  in the attention layers, the augmentation size  $\omega$  during training, the parameter  $k$  for the local  $k$ -NNs, and the size of training instances  $n_{\mathrm{train}}$ . Following a quick grid search, we set the hyperparameters as follows:  $\gamma_{\mathrm{global}} = 4$ ,  $\gamma_{\mathrm{local}} = 6$ ,  $\gamma_{\mathrm{heads}} = 8$ ,  $\omega = 7$ , and  $k = 12$ . However, as suggested by Table 3, the performance of our model is very stable to changes of hyperparameters.

# 6.4 ANALYSIS OVER TS

We evaluate  $\mathrm{TS}^4$  on the same dataset, with the best solution generated by  $\mathrm{TS}^3$  as the initial solution of MCTS.  $\mathrm{TS}^4$ -uniform is  $\mathrm{TS}^4$  with the heatmap replaced by a uniform one. Although our performance is the best shown in the table, we do not conclude that  $\mathrm{TS}^4$  is dominant to other methods, since we do not include a systematic evaluation nor we also do not include other model-search papers. We just use this to

Table 4: TSP-Random evaluations with MCTS.  

<table><tr><td>Methods</td><td>TSP200</td><td>TSP500</td><td>TSP1000</td></tr><tr><td>*DIMES+MCTS</td><td>/</td><td>2.64%</td><td>3.98%</td></tr><tr><td>*DIMES+AS+MCTS</td><td>/</td><td>1.76%</td><td>2.46%</td></tr><tr><td>AttGCRN+MCTS</td><td>0.64%</td><td>2.10%</td><td>2.69%</td></tr><tr><td>TS4-uniform</td><td>0.10%</td><td>2.91%</td><td>5.39%</td></tr><tr><td>TS4</td><td>0.04%</td><td>0.45%</td><td>2.05%</td></tr></table>

show that our  $\mathsf{TS}^4$  can at least show a competitive result compared with other SOTA works. Detailed experimental results for  $\mathsf{TS}^4$  are available in Appendix A.4.

# 7 CONCLUSION

We demonstrated that approximate invariance can be exploited to improve generalization by proposing an architecture called  $\mathsf{TS}^2$ , training it on small-sized instances with data augmentation to form  $\mathsf{TS}^3$ , and evaluating it on larger instances. Our  $\mathsf{TS}^3$  achieves a dominant performance among all end-to-end methods considering cross-size generalization. In addition, we also propose a simple but generic method to adapt Monte-Carlo Tree Search. Furthermore, we performed multiple experiments to investigate the effects of our designed components and the sensitivity to the hyperparameters. As future work, we plan to extend our architecture to more complex routing problems, such as Vehicle Routing Problem (VRP).

# REFERENCES

Ali Fuat Alkaya and Ekrem Duman. Application of sequence dependent traveling salesman problem in printed circuit board assembly. 3(6):1063-1076, 2013. URL https://doi.org/10.1109/TCPMT.2013.2252429.  
David Applegate, Robert Bixby, Václav Chvátal, and William J. Cook. Concorde TSP Solver, 2015. URL https://www.math.uwaterloo.ca/tsp/concorde/index.html.  
Irwan Bello, Hieu Pham, Quoc V Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940, 2016. URL https://arxiv.org/abs/1611.09940.  
Jieyi Bi, Yining Ma, Jiahai Wang, Zhiguang Cao, Jinbiao Chen, Yuan Sun, and Yeow Meng Chee. Learning generalizable models for vehicle routing problems via knowledge distillation. 35:31226-31238, 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/bit/70528fb11dc8086c6a623da9f3fee6-AAbstract-Conference.html.  
Jakob Bossek, Pascal Kerschke, Aneta Neumann, Markus Wagner, Frank Neumann, and Heike Trautmann. Evolving diverse tsp instances by means of novel and creative mutation operators. In Proceedings of the 15th ACM/SIGEVO conference on foundations of genetic algorithms, pp. 58-71, 2019. URL https://dl.acm.org/doi/10.1145/3299904.3340307.  
Xavier Bresson and Thomas Laurent. The Transformer Network for the Traveling Salesman Problem, March 2021. URL http://arxiv.org/abs/2103.03012.  
Hanni Cheng, Haosi Zheng, Ya Cong, Weihao Jiang, and Shiliang Pu. Select and optimize: Learning to solve large-scale TSP instances. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 1219-1231. PMLR, 2023. URL https://proceedings.mlr.press/v206/cheng23a.html.  
Jinho Choo, Yeong-Dae Kwon, Jihoon Kim, Jeongwoo Jae, André Hottung, Kevin Tierney, and Youngjune Gwon. Simulation-guided beam search for neural combinatorial optimization. Advances in Neural Information Processing Systems, 35:8760-8772, 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/ hash/39b9b60f0d149eabd1fff2d7c7d5afc4-Abstract-Conference.html.  
Paulo da Costa, Jason Rhuggenaath, Yingqian Zhang, Alp Akcay, and Uzay Kaymak. Learning 2-Opt Heuristics for Routing Problems via Deep Reinforcement Learning. SN Computer Science, 2(5):388, July 2021. ISSN 2661-8907. doi: 10.1007/s42979-021-00779-2. URL https://doi.org/10.1007/s42979-021-00779-2.  
Zhang-Hua Fu, Kai-Bin Qiu, and Hongyuan Zha. Generalize a Small Pre-trained Model to Arbitrarily Large TSP Instances. Proceedings of the AAAI Conference on Artificial Intelligence, 35 (8):7474-7482, May 2021. ISSN 2374-3468, 2159-5399. doi: 10.1609/aaai.v35i8.16916. URL https://ojs.aaai.org/index.php/AAAI/article/view/16916.  
Chengrui Gao, Haopu Shang, Ke Xue, Dong Li, and Chao Qian. Towards generalizable neural solvers for vehicle routing problems via ensemble with transferrable local policy. 2023. URL https://arxiv.org/abs/2308.14104v1.  
Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2023. URL https://www.gurobi.com.  
Keld Helsgaun. General k-opt submoves for the Lin-Kernighan TSP heuristic. Mathematical Programming Computation, 1(2):119-163, October 2009. ISSN 1867-2957. doi: 10.1007/s12532-009-0004-6. URL https://doi.org/10.1007/s12532-009-0004-6.  
Keld Helsgaun. An extension of the lin-kernighan-helsgaun tsp solver for constrained traveling salesman and vehicle routing problems. Roskilde: Roskilde University, 12, 2017. URL https://forskning.ruc.dk/en/publications/an-extension-of-the-lin-kernighan-helsgaun-tsp-solver-for-constra.

André Hottung, Yeong-Dae Kwon, and Kevin Tierney. Efficient active search for combinatorial optimization problems. 2021. URL https://openreview.net/forum?id=nO5caZwFwYu.  
Yuan Jiang, Yaoxin Wu, Zhiguang Cao, and Jie Zhang. Learning to solve routing problems via distributionally robust optimization, 2022. URL https://arxiv.org/abs/2202.07241v1.  
Yuan Jiang, Zhiguang Cao, Yaoxin Wu, and Jie Zhang. Multi-view graph contrastive learning for solving vehicle routing problems. In Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, pp. 984-994. PMLR, 2023. URL https://proceedings.mlr.org/press/v216/jiang23a.html.  
Yan Jin, Yuandong Ding, Xuanhao Pan, Kun He, Li Zhao, Tao Qin, Lei Song, and Jiang Bian. Pointerformer: Deep Reinforced Multi-Pointer Transformer for the Traveling Salesman Problem. Proceedings of the AAAI Conference on Artificial Intelligence, 37(7):8132-8140, June 2023. ISSN 2374-3468. doi: 10.1609/aaai.v37i7.25982. URL https://ojs.aaai.org/index.php/AAAI/article/view/25982.  
Chaitanya K. Joshi, Thomas Laurent, and Xavier Bresson. An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem, October 2019. URL http://arxiv.org/abs/1906.01227.  
Chaitanya K. Joshi, Quentin Cappart, Louis-Martin Rousseau, Thomas Laurent, and Xavier Bresson. Learning tsp requires rethinking generalization. arXiv:2006.07054 [cs, stat], Jun 2020. URL http://arxiv.org/abs/2006.07054.  
Elias Khalil, Hanjun Dai, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning Combinatorial Optimization Algorithms over Graphs. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit/bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-bit-
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- - - - - - - - - - - - - - - - - - - - - - - -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
-
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- -
- #
.
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
#
...
#
#
#