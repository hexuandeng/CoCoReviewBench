# OOD Link Prediction Generalization Capabilities of Message-Passing GNNs in Larger Test Graphs

Anonymous Author(s)

Affiliation

Address

email

# Abstract

This work provides the first theoretical study on the ability of graph Message Passing Neural Networks (gMPNNs)—such as Graph Neural Networks (GNNs)—to perform inductive out-of-distribution (OOD) link prediction tasks, where deployment (test) graph sizes are larger than training graphs. We first prove non-asymptotic bounds showing that link predictors based on permutation-equivariant (structural) node embeddings obtained by gMPNNs can converge to a random guess as test graphs get larger. We then propose a theoretically-sound gMPNN that outputs structural pairwise (2-node) embeddings and prove non-asymptotic bounds showing that, as test graphs grow, these embeddings converge to embeddings of a continuous function that retains its ability to predict links OOD. Empirical results on random graphs show agreement with our theoretical results.

# 1 Introduction

13 Link prediction is the task of predicting whether two nodes likely have a missing link [1, 10, 27, 34, 58]. Link prediction tasks arise in many settings, ranging from predicting edges on bipartite graphs between users and products or content in recommender systems [5, 9, 28, 29, 35, 54], to knowledge graph reconstruction [4, 12, 18, 47, 58, 59], to predicting protein-protein interactions [49].

In recent years, there has been growing interest in applying neural network models to inductive link prediction tasks. Inductive link prediction considers methods trained on a graph  $G^{\mathrm{tr}}$  and deployed at test time on another graph  $G^{\mathrm{te}}$ . It also encompasses the task of training the method on a smaller induced subgraph  $G^{\mathrm{tr}}$  of a larger graph  $G^{\mathrm{te}}$ , then deploying it on the entire graph. In particular, our work focuses on graph message-passing Neural Networks (gMPNNs) [19, 52] or, more precisely, the widely used Graph Neural Network (GNN) framework [7, 8, 11, 20, 22, 25, 53, 56, 61].

Our work asks the following questions: Are link prediction methods able to cope with the task of inductive out-of-distribution (OOD) link prediction, where (unseen) test graphs are significantly larger than training graphs? How can these OOD link prediction tasks be theoretically defined? Can we obtain non-asymptotic bounds on the generalization capabilities of these methods?

The majority of today's link prediction methods are based on a similar principle. Consider an attributed graph  $G = (V, E)$ , with node set  $V = \{1, \dots, N\}$ , edge set  $E \subseteq V \times V$ , and node features  $\pmb{F} \in \mathbb{R}^{N \times F_0}$ ,  $F_0 \geq 1$ . Then, given a pair of nodes  $i, j \in V$ , after  $T \geq 1$  iterations over  $G$ , these methods produce associated node embeddings (representation vectors)  $\Theta_i^{\bullet}, \Theta_j^{\bullet} \in \mathbb{R}^{F_T}$ ,  $F_T \geq 1$ , which are then used in a link function  $\eta^{\bullet}: \mathbb{R}^{F_T} \times \mathbb{R}^{F_T} \to [0,1]$  such that  $\eta^{\bullet}(\Theta_i^{\bullet}, \Theta_j^{\bullet})$  predicts the probability that  $i$  and  $j$  have a missing link in  $G$ . In our notation we will denote all node embeddings and associated functions with the superscript “ $\bullet$ ”. Henceforth we denote gMPNNs that output structural node embeddings as gMPNNs $\bullet$ .

Node embeddings. The first part of our work considers a subset of these methods, where the output node embeddings are permutation equivariant (a.k.a. structural node embeddings [57]). Informally, a sequence of node embeddings  $\Theta^{\bullet} \in \mathbb{R}^{N \times F_T}$  given by an embedding method is permutation-equivariant if for any arbitrary graph  $G$  and any permutation  $\pi \in \mathbb{S}_N$  of the node indices, where  $\mathbb{S}_N$  is the symmetric group, the resulting isomorphic graph  $G' = (\pi \circ V, \pi \circ E, \pi \circ F)$  gets permuted node embeddings  $\Theta^{\bullet'} = \pi \circ \Theta^{\bullet}$ , where  $\pi \circ M$  defines the action of  $\pi$  on  $M$  (we will provide a formal definition in Section 2). We leave the study of OOD link prediction with positional node embeddings (a.k.a. permutation-sensitive node embeddings [57]) to future work, since each such method has a different procedure to induce permutation sensitivity: Adding features with one-hot encodings [25], the node's own eigenvectors as positional encoding [14, 30], or random node features [33, 45, 68] that are used to break the permutation equivariance of gMPNNs and force positional nodes embeddings.

The application of GNNs to link prediction tasks is made difficult by the fact that, by construction, permutation-equivariant GNNs give the same embeddings  $\Theta_i^{\bullet},\Theta_j^{\bullet}$  to any isomorphic nodes  $i,j$  in  $G$ , as noted by You et al. [68] and Srinivasan and Ribeiro [57]. Isomorphic nodes are nodes that are structurally indistinguishable in  $G$  (even when considering node features) except by their (assumed arbitrary) node indices  $i,j\in V$ . Thankfully, both large real-world and large random graphs (see the Appendix for a discussion) tend to have only very few isomorphic nodes. Therefore, in practice, node isomorphism in large graphs  $G$  is a non-issue for link prediction (as we see in our experiments).

Pairwise embeddings. Taking a different route, Srinivasan and Ribeiro [57] shows that the link prediction task between  $i$  and  $j$  can always be performed by pairwise embeddings  $\Theta_{ij}^{\bullet \bullet}(G)$ , i.e., for any pair of nodes  $i,j$  in a graph  $G$ , there exists a pairwise embedding  $\Theta_{ij}^{\bullet \bullet}(G)$  and a link function  $\eta^{\bullet \bullet}:\mathbb{R}^{F_T}\to [0,1]$  such that  $\eta^{\bullet \bullet}(\Theta_{ij}^{\bullet \bullet})$  approximates the probability that  $i$  and  $j$  have a hidden link. In our notation we will denote all pairwise (joint 2-node) embeddings and associated functions with the superscript “ $\bullet \bullet$ ”. And while one could still apply standard GNNs to obtain pairwise embeddings by defining a dual graph where all edges in  $G$  become nodes in  $G^{\mathrm{dual}}$  [42], there are new purposefully-built gMPNNs that generate pairwise embeddings, such as [64, 70, 72, 73]. Unfortunately, as the test graph grows, we were unable to prove these methods converge to a procedure that is able to perform OOD link prediction tasks. Hence, we propose a different family of gMPNNs for pairwise embeddings, denoted gMPNNs $\bullet \bullet$  henceforth. The second part of our work considers the OOD generalization capability of these gMPNNs $\bullet \bullet$ .

Contributions. In this work we study inductive OOD link prediction tasks for larger test graphs using permutation-equivariant node and pairwise embeddings,  $\Theta^{\bullet}$  and  $\Theta^{\bullet \bullet}$ , respectively. Our work makes the following contributions:

1. We provide a theoretical framework defining OOD inductive link prediction tasks, where test graphs are significantly larger than training graphs.  
2. We show that structural node embeddings from message-passing GNNs can fail in OOD link prediction tasks if the test graph (from the same graph family) is significantly larger than the training graph. Our work fills an important gap in the literature, where Bevilacqua et al. [6] studied the OOD capabilities of GNNs for graph classification using random graph models. Our work studies the OOD capabilities of GNNs for inductive link prediction in a similar setting.  
3. We provide non-asymptotic bounds on the convergence of node and pairwise gMPNNs embeddings to an easy-to-analyze continuous model. Extensive empirical experiments using stochastic block models (SBMs [55]) validate our theoretical results. Our work focuses on providing a theoretical understanding of the challenges of OOD link prediction tasks rather than propose real-world link prediction tasks and compare baselines. Our work, however, lays the theoretical foundation (and challenges) for future application-focused works.

# 2 Preliminaries

Given an attributed graph  $G = (V, E)$ , with node set  $V = \{1, \dots, N\}$ , edge set  $E \subseteq V \times V$ , adjacency matrix  $A \in \{0, 1\}^{N \times N}$ , where  $A_{ij} = \mathbb{1}_{\{(i,j) \in E\}}$ , and node features  $F \in \mathbb{R}^{N \times F_0}$ ,  $F_0 > 0$ . Let  $P_{\pi} \in \mathcal{B}_N$  be a permutation matrix associated with permutation  $\pi \in \mathbb{S}_N$  (where  $\mathbb{S}_N$  is the symmetric group), where  $\mathcal{B}_N$  denotes the Birkhoff polytope of  $N \times N$  doubly-stochastic matrices. Doubly-sctochastic matrices are non-negative square matrices whose rows and columns sum to one. The matrix  $P_{\pi}$  defines the action of permutation  $\pi$  on these matrices, e.g.,  $\pi \circ A = P_{\pi}AP_{\pi}^T$ . We denote a pair of nodes  $i, j \in V$  as isomorphic in  $G$  if exists  $\pi \in \mathbb{S}_N$  such that  $\pi_i = j, \pi_j = i, A = P_{\pi}AP_{\pi}^T$ , and  $F = P_{\pi}F$ . Node features can be defined by the graph signal  $f: V \to \mathbb{R}^{F_0}$  as a function that

maps a node to an  $F_0$ -dimensional feature in  $\mathbb{R}^{F_0}$ . Then the signal of the graph  $\pmb{F}$  can be represented by a matrix  $\pmb{F} = [\mathbf{f}_1, \dots, \mathbf{f}_N]^T \in \mathbb{R}^{N \times F_0}$ , where  $\mathbf{f}_i \in \mathbb{R}^{F_0}$  are the features of node  $i \in V$ .

Random graph model for  $G$ . Let  $(\mathcal{X}, d, \mu)$  be a metric-measure space, where  $\mathcal{X}$  is a set,  $d$  is a metric, and  $\mu$  is a probability Borel measure. A graphon is defined as a mapping  $W: \mathcal{X} \times \mathcal{X} \to [0,1]$  [13, 66]. In what follows we define how the graph  $G$  is sampled from the graphon models. The signal definition follows Maskey et al. [39] and the edge samples follow Airoldi et al. [3], Lawrence and Hyvarinen [31].

Definition 1 (Random graph model). A random graph model for  $G$  on  $(\mathcal{X},d,\mu)$  is defined as a pair  $(W,f)$  of a graphon  $W:\mathcal{X}\times$ $\mathcal{X}\rightarrow [0,1]$  and a metric-space signal  $f:\mathcal{X}\to \mathbb{R}^{F_0}$ , where  $f\in$ $L^{\infty}(\mathcal{X})$  is an essentially bounded measurable function with the essential supremum norm. We obtain  $(G,F)$  by first sampling  $N$  i.i.d. random points  $X_{1},\ldots ,X_{N}$  from  $\mathcal{X}$  with probability density  $\mu$ , as the nodes of  $G$ . Then the edge  $(i,j)$  between nodes  $i$  and  $j$  is sampled with probability  $W(X_{i},X_{j})$ , i.e., the adjacency matrix  $\mathbf{A} = (\mathbf{A}_{i,j})_{i,j}$  of  $G$  is defined as  $\mathbf{A}_{i,j} = \mathbb{1}(Z_{i,j} < W(X_i,W_j))$  for  $i,j = 1,\dots,N$ , where  $\{Z_{i,j}\}_{i,j = 1}^{N}$  are sampled i.i.d. from Uniform(0,1). The graph signal  $\mathbf{F} = [\mathbf{f}_1,\dots,\mathbf{f}_N]^T\in \mathbb{R}^{N\times F_0}$  is defined as  $\mathbf{f}_i = f(X_i)$ . We say  $(G,F)\sim (W,f)$ . Further, we restrict our attention to graphons  $W$  such that there exists a

![](images/9d39c73ba1ed3ae710cbf3eb196edfebc3db188db63a2de45fc1af77ae72b14a.jpg)  
Figure 1: Templated causal DAG of  $G$ . Hidden and observed variables are shaded white and gray, respectively.

constant  $d_{\min}$  satisfying the graphon degree  $d_W(x)\coloneqq \int_{\mathcal{X}}W(x,y)d\mu (y)\geq d_{\min} > 0,\forall x\in \mathcal{X}.$

In an abuse of notation we identify node  $i \in V$  with the sampled value  $X_{i} \sim \mu, \forall i \in \{1, \dots, N\}$ , since generally  $\mu$  is such that  $P(X_{i} = X_{j}) = 0$  almost everywhere for  $i \neq j$  (e.g.,  $\mu$  is uniform). The causal DAG of the data generation process of  $G$  is given in Figure 1. Furthermore, we note that all proofs are relegated to the Appendix due to space constraints. In what follows we use the terms node embeddings and node representations interchangeably.

# 2.1 Inductive structural node representations with graph message-passing neural networks

Graph message-passing Neural Network (gMPNN\* ) is defined by realizing a message-passing Neural Network (MPNN) on a graph.

Definition 2 (MPNN). Let  $T \in \mathbb{N}$  denote the number of layers. For  $t = 1, \dots, T$ , let  $\Phi^{(t)}: \mathbb{R}^{2F_{t-1}} \to \mathbb{R}^{H_{t-1}}$  and  $\Psi^{(t)}: \mathbb{R}^{F_{t-1} + H_{t-1}} \to \mathbb{R}^{F_t}$  be functions, where  $F_t \in \mathbb{N}$  is called the feature dimension of layer  $t$ . The corresponding MPNN  $\Theta$  is defined by the sequence of message functions  $(\Phi^{(t)})_{t=1}^T$  and update functions  $(\Psi^{(t)})_{t=1}^T$ , i.e.  $\Theta = ((\Phi^{(t)})_{t=1}^T, (\Psi^{(t)})_{t=1}^T)$ .

The message and update functions in Definition 2 are usually given by MLPs. We now introduce the gMPNN $\bullet$  with  $T$  message-passing layers. For each node  $i \in V$ ,  $\mathbf{f}_i^{\bullet(t)}$  at layer  $t \in \{1, \dots, T\}$  is defined recursively using (a) its own representation at layer  $t - 1$  ( $\mathbf{f}_i^{\bullet(t - 1)}$ ) and (b) an aggregated representation of its neighbors  $m_i^{(t)}$  (commonly) defined via two distinct permutation-invariant aggregation functions as follows:

Definition 3 (gMPNN $\bullet$ ). Let  $(G,F)$  be a graph with graph signals as in Definition 1 and  $\Theta$  be a MPNN as in Definition 2. For layer  $t = 1,\dots,T$ , define  $\overline{\Theta}_{A}^{\bullet(t)}$  and  $\Theta_{A}^{\bullet(t)}$  as maps from the input graph  $G$  and graph signals  $F^{(0)} = F \in \mathbb{R}^{N \times F_0}$  to the features in the t-th neural layer by

$$
\overline {{\Theta}} _ {\boldsymbol {A}} ^ {\bullet (t)}: \mathbb {R} ^ {N \times F _ {0}} \to \mathbb {R} ^ {N \times F _ {t}}, \quad \Theta_ {\boldsymbol {A}} ^ {\bullet (t)}: \mathbb {R} ^ {N \times F _ {0}} \to \mathbb {R} ^ {N \times F _ {t}}, \quad \boldsymbol {F} \mapsto \boldsymbol {F} ^ {(t)} = (\mathbf {f} _ {i} ^ {\bullet (t)}) _ {i = 1} ^ {N}
$$

where  $\pmb{F}^{(t)}$  is defined by either the neighbor-average aggregation procedure (a),  $\forall i\in V$ , for  $\overline{\Theta}_{A}^{\bullet (t)}$

$$
\overline {{m}} _ {i} ^ {(t)} := \frac {1}{N} \sum_ {j = 1} ^ {N} \frac {\boldsymbol {A} _ {i , j}}{d _ {i}} \Phi^ {(t)} (\overline {{\mathbf {f}}} _ {i} ^ {\bullet (t - 1)}, \overline {{\mathbf {f}}} _ {j} ^ {\bullet (t - 1)}),
$$

$$
\overline {{\mathbf {f}}} _ {i} ^ {\bullet (t)} := \Psi^ {(t)} (\overline {{\mathbf {f}}} _ {i} ^ {\bullet (t - 1)}, \overline {{m}} _ {i} ^ {(t)}),
$$

where the  $N$ -normalized degree of node  $i$  is  $d_{i} = \frac{1}{N}\sum_{j=1}^{N}\mathbf{A}_{i,j}$ , or the (N-normalized) sum aggregation procedure (b),  $\forall i \in V$ , for  $\Theta_{\mathbf{A}}^{\bullet(t)}$ ,  $m_{i}^{(t)} := \frac{1}{N}\sum_{j=1}^{N}A_{i,j}\Phi^{(t)}(\mathbf{f}_{i}^{\bullet(t-1)}, \mathbf{f}_{j}^{\bullet(t-1)})$ , and  $\mathbf{f}_{i}^{\bullet(t)} := \Psi^{(t)}(\mathbf{f}_{i}^{\bullet(t-1)}, m_{i}^{(t)})$ .

Given a  $\mathrm{gMPNN}^{\bullet}$ ,  $\overline{\Theta}_{A}^{\bullet (T)}$  or  $\Theta_{A}^{\bullet (T)}$ , with  $T\geq 1$  layers as in Definition 3, their outputs are the graph signals  $\overline{\Theta}_{A}^{\bullet (T)}(F)\in \mathbb{R}^{N\times F_T}$  for the neighbor-average aggregation and  $\Theta_{A}^{\bullet (T)}(F)\in \mathbb{R}^{N\times F_T}$  for the  $(N$ -normalized) sum aggregation, and are henceforth denoted as node embedding outputs of the  $\mathrm{gMPNN}^{\bullet}$ . We denote  $\overline{\Theta}_{A}^{\bullet (T)}(F)_i$  as the node embedding for node  $i\in V$ .

# 2.2 Node embeddings with continuous message passing neural networks

Here we define continuous message passing neural networks for structural node embeddings.

Definition 4 (Continuous message-passing). Given a MPNN  $\Theta$  as in Definition 2, the node continuous message passing neural network (cMPNN $\bullet$ ) on graphons and metric-space signals  $f: \mathcal{X} \to \mathbb{R}^{F_0}$  can be defined by replacing the graph node features and the aggregation scheme in Definition 3 by the following continuous counterparts. Using a message signal  $U: \mathcal{X} \times \mathcal{X} \to \mathbb{R}^H$ , the continuous mean aggregation of  $U$  is defined by  $\overline{M}_W^\bullet(U)(x) = \int_{\mathcal{X}} \frac{W(x,y)}{d_W(x)} U(x,y)d\mu(y)$ , while the integral aggregation is defined as  $M_W^\bullet(U)(x) = \int_{\mathcal{X}} W(x,y)U(x,y)d\mu(y)$ , where  $W$  is a graphon and  $d_W(x) = \int_{\mathcal{X}} W(x,y)d\mu(y)$  is the graphon degree as in Definition 1.

By replacing the discrete aggregations in Definition 3 by the above continuous aggregations, the same MPNN  $\Theta$  can process metric-space signals instead of graph signals in the following definition.

Definition 5 (cMPNN $\bullet$ ). Let  $(W, f)$  be a random graph model as in Definition 1 and  $\Theta$  be a MPNN as in Definition 2. For  $t = 1, \dots, T$ , define  $\overline{\Theta}_W^{\bullet(t)}$  and  $\Theta_W^{\bullet(t)}$  as maps from input metric-space signal  $f^{\bullet(0)} = f: \mathcal{X} \to \mathbb{R}^{F_0}$  to the features in the  $t$ -th layer by

$$
\overline {{\Theta}} _ {W} ^ {\bullet (t)}: L ^ {2} (\mathcal {X}) \to L ^ {2} (\mathcal {X}), \quad \Theta_ {W} ^ {\bullet (t)}: L ^ {2} (\mathcal {X}) \to L ^ {2} (\mathcal {X}), \quad f ^ {\bullet} \mapsto f ^ {\bullet (t)},
$$

where  $\vec{f}^{\bullet (t)}$  are defined sequentially either through the degree-average aggregation (a) for  $\overline{\Theta}_{W}^{\bullet (t)}$ :

$$
\bar {g} ^ {\bullet (t)} (x) := \overline {{M}} _ {W} ^ {\bullet} (\Phi^ {(t)} (\bar {f} ^ {\bullet (t - 1)}, \bar {f} ^ {\bullet (t - 1)})) (x) = \int_ {\mathcal {X}} \frac {W (x , y)}{d _ {W} (x)} \Phi^ {(t)} (\bar {f} ^ {\bullet (t - 1)} (x), \bar {f} ^ {\bullet (t - 1)} (y)) d \mu (y),
$$

$$
\bar {f} ^ {\bullet (t)} (x) := \Psi^ {(t)} (\bar {f} ^ {\bullet (t - 1)} (x), \bar {g} ^ {\bullet (t)} (x)),
$$

or through the integral aggregation  $(b)$  for  $\Theta_W^{\bullet (t)}\colon g^{\bullet (t)}(x)\coloneqq M_W^{\bullet}(\Phi^{(t)}(f^{\bullet (t - 1)},f^{\bullet (t - 1)}))(x) =$ $\int_{\mathcal{X}}W(x,y)\Phi^{(t)}(f^{\bullet (t - 1)}(x),f^{\bullet (t - 1)}(y))d\mu (y),$  with  $f^{\bullet (t)}(x)\coloneqq \Psi^{(t)}(f^{\bullet (t - 1)}(x),g^{\bullet (t)}(x)).$

# 3 Size-stability of node representation and its drawbacks

We now present our main theorems about convergence of  $\mathrm{gMPNN}^{\bullet}$  to cMPNN\* for test graphs  $G^{\mathrm{te}}$  sampled from the graphon random graph model (see Definition 1), and how it leads to size-stability of  $\mathrm{gMPNN}^{\bullet}$  for nodes that have the same representation under cMPNNs\*. In what follows we will focus on the neighbor-average aggregation procedure (a) of Definition 3, since this is the more difficult case to prove. Similar results for the  $(N$ -normalized) sum aggregation procedure (Definition 3(b)) are shown in the Appendix due to space constraints. Moreover, common definitions (e.g., Lipschitz continuous functions) are also relegated to the Appendix to save space.

# 3.1 Convergence of gMPNNs towards cMPNNs as test graph size increase

We now prove that, with high probability, the maximum infinity difference between the gMPNN\* and cMPNN node representations decreases with  $N^{\mathrm{te}}$  , the size of  $G^{\mathrm{te}}$

Theorem 1. Let  $\Theta = ((\Phi^{(l)})_{l=1}^T, (\Psi^{(l)})_{l=1}^T)$  be a MPNN as in Definition 2 with  $T$  layers such that  $\Phi^{(l)}: \mathbb{R}^{2F_{l-1}} \to \mathbb{R}^{H_{l-1}}$  and  $\Psi^{(l)}: \mathbb{R}^{F_{l-1} + H_{l-1}} \to \mathbb{R}^{F_l}$  are Lipschitz continuous with Lipschitz constants  $L_{\Phi}^{(l)}$  and  $L_{\Psi}^{(l)}$ . Let  $gMPNN^\bullet \overline{\Theta}_A^{\bullet(T)}$  and  $cMPNN^\bullet \overline{\Theta}_W^{\bullet(T)}$  be as in Definitions 3 and 5. For a random graph model  $(W, f)$  satisfying Definition 1, consider training  $(G^{tr}, F^{tr}) \sim (W, f)$  and test  $(G^{te}, F^{te}) \sim (W, f)$  graphs, where  $N^{tr}$  and  $N^{te}$  are their respective number of nodes,  $N^{te} > N^{tr}$ . Let  $X_1^{te}, \ldots, X_{N^{te}}^{te}$  and  $A^{te}$  be as in Definition 1. Let  $p \in (0, \frac{1}{\sum_{l=1}^T 2(H_l + 1)})$ . Then, if

$$
\frac {\sqrt {N ^ {t e}}}{\sqrt {\log \left(2 N ^ {t e} / p\right)}} \geq \frac {4 \sqrt {2}}{d _ {\min }}, \tag {1}
$$

we have with probability at least  $1 - \sum_{l=1}^{T} 2(H_l + 1)p$ ,

$$
\delta_ {A \cdot W} ^ {\bullet} := \max  _ {i = 1, \dots , N ^ {t e}} \| \overline {{\Theta}} _ {\boldsymbol {A} ^ {t e}} ^ {\bullet (T)} (\boldsymbol {F} ^ {t e}) _ {i} - \overline {{\Theta}} _ {W} ^ {\bullet (T)} (f) (X _ {i} ^ {t e}) \| _ {\infty} \leq (C _ {1} + C _ {2} \| f \| _ {\infty}) \frac {\sqrt {\log (2 N ^ {t e} / p)}}{\sqrt {N ^ {t e}}},
$$

where the constants  $C_1$  and  $C_2$  are defined in the Appendix and depend on  $\{L_{\Phi}^{(l)}, L_{\Psi}^{(l)}\}_{l=1}^T$  and  $(G^{tr}, F^{tr})$ ,  $N^{tr}$ .

Theorem 1 above shows that as the test graph size  $N^{\mathrm{te}}$  grows, the node representations from the discrete gMPNNs converge to the continuous cMPNNs. This has profound consequences when it comes to predicting links using the node representations obtained by a gMPNN. Next, Corollary 1 shows that for any two nodes  $i,j\in V^{\mathrm{te}}$  that are indistinguishable in the degree-aggregated cMPNN (defined as  $\overline{\Theta}_{W}^{\bullet (T)}(f)(X_{i}^{\mathrm{te}}) = \overline{\Theta}_{W}^{\bullet (T)}(f)(X_{j}^{\mathrm{te}}))$ , they will get increasingly similar representations in the discrete gMPNN as  $N^{\mathrm{te}}$  grows.

Corollary 1. Let  $\Theta = ((\Phi^{(l)})_{l=1}^T, (\Psi^{(l)})_{l=1}^T)$ ,  $\overline{\Theta}_A^{\bullet(T)}$ ,  $\overline{\Theta}_W^{\bullet(T)}$ ,  $p$ ,  $(W,f)$ ,  $(G^{tr},F^{tr})$ ,  $(G^{te},F^{te})$ ,  $N^{tr}$ ,  $N^{te}$ ,  $A^{te}$ , and  $X_1^{te},\ldots,X_{N^{te}}^{te}$  be as in Theorem 1. If there exists  $i,j\in V^{te},i\neq j$ , s.t.  $\overline{\Theta}_W^{\bullet(T)}(X_i) = \overline{\Theta}_W^{\bullet(T)}(X_j)$  and Equation (1) is satisfied, then, with  $C_1$  and  $C_2$  as in Theorem 1, we have that with probability at least  $1 - \sum_{l=1}^{T}2(H_l + 1)p$ ,

$$
\| \overline {{\Theta}} _ {\boldsymbol {A} ^ {t e}} ^ {\bullet (T)} (\boldsymbol {F} ^ {t e}) _ {i} - \overline {{\Theta}} _ {\boldsymbol {A} ^ {t e}} ^ {\bullet (T)} (\boldsymbol {F} ^ {t e}) _ {j} \| _ {\infty} \leq \left(C _ {1} + C _ {2} \| f \| _ {\infty}\right) \frac {2 \sqrt {\log \left(2 N ^ {t e} / p\right)}}{\sqrt {N ^ {t e}}}.
$$

Implications of Corollary 1 on Stochastic Block Models (SBMs). In what follows, we will discuss circumstances where two nodes  $i,j\in V$  get the same cMPNN  $\bullet$  representations (i.e., both  $\overline{\Theta}_W^{\bullet (T)}(f)(X_i) = \overline{\Theta}_W^{\bullet (T)}(f)(X_j)$  and  $\Theta_W^{\bullet (T)}(f)(X_i) = \Theta_W^{\bullet (T)}(f)(X_j))$ . In what follows we restrict our results to an important family of graphon models: Stochastic Block Models (SBMs) [55], where we also model node attributes. SBMs were chosen because they can consistently model large graphs generated by any piecewise Lipschitz graphon model [3]. SBMs are also intuitive models, which makes them useful to illustrate our results.

Definition 6 (Stochastic Block Model (SBM)). An SBM  $(W, f)$  is a random graph model (Definition 1) with cluster structures in  $W$  and  $f$ . Partition the node set into  $r \geq 2$  disjoint subsets  $S_{1}, S_{2}, \ldots, S_{r} \subseteq V$  (known as blocks or communities) with an associated  $r \times r$  symmetric matrix  $\mathbf{S}$ , where the probability of an edge  $(i,j)$ ,  $i \in S_{\mathbf{A}}$  and  $j \in S_{b}$  is  $\mathbf{S}_{ab}$ , for  $a,b \in \{1,\dots,r\}$ . Let  $\mathcal{X} = [0,1]$ , and  $\mu$  be the uniform distribution on  $[0,1]$ . By dividing  $\mathcal{X} = [0,1]$  into disjoint convex sets  $[t_0,t_1],[t_1,t_2],\ldots,[t_{r-1},t_r]$ , where  $t_0 = 0$  and  $t_r = 1$ , node  $i$  belongs to block  $S_{a}$  if  $X_{i} \sim \text{Uniform}(0,1)$  satisfies  $X_{i} \in [t_{a-1},t_{a})$ . The graphon function  $W$  is defined as  $W(X_{i},X_{j}) = \sum_{a,b \in \{1,\dots,r\}} \mathbf{S}_{ab} \mathbb{1}(X_{i} \in [t_{a-1},t_{a})) \mathbb{1}(X_{j} \in [t_{b-1},t_{b}))$ . We take the liberty to also define node signals in our SBM model, where for  $\mathbf{B} = [B_{1},\dots,B_{r}]^{T} \in \mathbb{R}^{r \times F_{0}}$  the metric-space signal  $f: \mathcal{X} \to \mathbb{R}^{F_{0}}$  is defined as  $f(x) = \sum_{a \in \{1,\dots,r\}} \mathbb{1}(x \in [t_{a-1},t_{a})) B_{a}$ .

We define the action of permutation  $\pi$  on  $B$  of Definition 6 as  $\pi \circ B$ , where  $(\pi \circ B)_{\pi_a} = B_a$ .

Definition 7 (Isomorphic SBM blocks). For the SBM model  $(W, f)$  in Definition 6, we say two blocks  $a, b \in \{1, \ldots, r\}$  are isomorphic if the SBM satisfies the following two conditions: (a)  $t_a - t_{a-1} = t_b - t_{b-1}$ , and (b) exists  $\pi \in \mathbb{S}_r$ , such that  $\pi_a = b$ ,  $\pi_b = a$ ,  $S = \pi \circ S$ , and  $B = \pi \circ B$ .

A similar definition can be obtained for the general graphons in Definition 1 using the isomorphic graphon definition of Lovász and Szegedy [36]. Now that we have the definition for isomorphic blocks in SBM models, we can prove that all nodes in these isomorphic blocks will obtain the same representations under degree-average aggregation and integral aggregation cMPNNs\*.

Lemma 1. Let  $\Theta = ((\Phi^{(l)})_{l=1}^T, (\Psi^{(l)})_{l=1}^T)$  be a MPNN as in Definition 2, and  $\overline{\Theta}_W^{\bullet(T)}$ ,  $\Theta_W^{\bullet(T)}$  as in Definition 5. For the SBM model  $(W, f)$  in Definition 6 with  $N^{te}$  nodes  $X_1, \ldots, X_{N^{te}}$ . If there exists  $i, j \in V^{te}$  such that  $X_i^{te}$ ,  $X_j^{te}$  are nodes that belong to isomorphic SBM blocks (Definition 7), then  $\overline{\Theta}_W^{\bullet(T)}(f)(X_i^{te}) = \overline{\Theta}_W^{\bullet(T)}(f)(X_j^{te})$  and  $\Theta_W^{\bullet(T)}(f)(X_i^{te}) = \Theta_W^{\bullet(T)}(f)(X_j^{te})$ .

Note that even though any two nodes in isomorphic SBM blocks get the same cMPNN\* representations per Lemma 1, these nodes are likely not isomorphic in  $G^{\mathrm{te}}$  (as shown in Proposition 1 in

Appendix) and, hence, they get different  $\mathrm{gMPNN}^{\bullet}$  representations. However, Corollary 1 shows that these representations become increasingly similar as the test graph size grows ( $N^{\mathrm{te}} \gg 1$ ). We use this observation to understand the ability of  $\mathrm{gMPNNs}^{\bullet}$  to perform link prediction tasks next.

# 3.2 The hardness of OOD inductive link prediction using structural node embeddings

The convergence of gMPNNs\* to cMPNNs\* as the test graph size  $N^{\mathrm{te}}$  grows (Theorem 1) implies through Corollary 1 and Lemma 1 that node representations of distinct SBM blocks can become increasingly similar as the test graph size grows  $(N^{\mathrm{te}}\gg 1)$ , even though these nodes are not isomorphic in  $G^{\mathrm{te}}$  with high probability (see Proposition 1 in the Appendix).

Definition 8 (Link prediction function from structural node embeddings). An inductive link prediction function  $\eta^{\bullet}:\mathbb{R}^{F_T}\times \mathbb{R}^{F_T}\to [0,1]$  takes the gMPNN node representations of two nodes  $i,j\in V^{te}$  and predicts the edge probability  $P(A_{ij}^{te} = 1)$ . We assume  $\eta^{\bullet}$  is Lipschitz continuous with Lipschitz constant  $L_{\eta}$ . In the context of graphon random graph models (Definition 1), we aim to learn  $\eta^{\bullet}(\overline{\Theta}_{A^{te}}^{\bullet (T)}(F^{te})_i,\overline{\Theta}_{A^{te}}^{\bullet (T)}(F^{te})_j)\approx W(i,j)$ . We further assume we predict a link if  $\eta^{\bullet}(\cdot ,\cdot) > \tau$  while no link if  $\eta^{\bullet}(\cdot ,\cdot) < \tau$ , for some (arbitrary) threshold  $\tau \in [0,1]$  chosen by the user of such system.

The next corollary showcases the difficulty in OOD predicting links using structural node representations as  $N^{\mathrm{te}}$  grows.

Corollary 2. Let  $\Theta = ((\Phi^{(l)})_{l=1}^T, (\Psi^{(l)})_{l=1}^T)$  be the MPNN with  $T$  layers and  $\overline{\Theta}_A^{\bullet(T)}, \overline{\Theta}_W^{\bullet(T)}$  as in Theorem 1. Let  $\eta^\bullet : \mathbb{R}^{F_T} \times \mathbb{R}^{F_T} \to [0,1]$  be as in Definition 8. Consider the SBM  $(W,f)$  in Definition 6 with isomorphic blocks (Definition 7). Let  $(G^{tr}, F^{tr}) \sim (W,f)$  and  $(G^{te}, F^{te}) \sim (W,f)$  be the training and test graphs with  $N^{tr}$  and  $N^{te}$  nodes, respectively. Consider any two test nodes  $i,j \in \{1,\dots,N^{te}\}$ ,  $i \neq j$ , for which we can make a link prediction decision with  $\eta^\bullet$  (i.e.,  $\eta^\bullet(\overline{\Theta}_{A^te}^{\bullet(T)}(F^{te})_i, \overline{\Theta}_A^{\bullet(T)}(F^{te})_j) \neq \tau$ ). Let  $G^{te}$  be large enough to satisfy both Equation (1) and

$$
\frac {\sqrt {N ^ {t e}}}{\sqrt {\log (2 N ^ {t e} / p)}} > \frac {2 (C _ {1} + C _ {2} \| f \| _ {\infty})}{| \eta^ {\bullet} (\overline {{\Theta}} _ {\boldsymbol {A} ^ {t e}} ^ {\bullet (T)} (\boldsymbol {F} ^ {t e}) _ {i} , \overline {{\Theta}} _ {\boldsymbol {A} ^ {t e}} ^ {\bullet (T)} (\boldsymbol {F} ^ {t e}) _ {j} - \tau | / L _ {\eta} ^ {\bullet}},
$$

where  $p$ ,  $C_1$ , and  $C_2$  are as given in Corollary 1. Then, if  $i$  and  $j$  belong to isomorphic blocks (i.e.,  $\overline{\Theta}_W^{\bullet (T)}(f)(X_i^{te}) = \overline{\Theta}_W^{\bullet (T)}(f)(X_j^{te})$ ), with probability at least  $1 - \sum_{l = 1}^{T}2(H_l + 1)p$  the link prediction method in Definition 8 will make the same link prediction regardless of the SBM probability matrix  $\mathbf{S}$  (Definition 6) and whether  $i$  and  $j$  are in the same block or distinct isomorphic blocks.

Corollary 2 proves that link prediction with structural node embeddings form gMPNNs\* is unreliable. That is, for any link prediction method satisfying Definition 8, as the test graph grows  $N^{\mathrm{te}}\gg 1$  , the method will increasingly struggle to give different predictions within and across isomorphic SBM blocks, even when these probabilities are arbitrarily different in the underlying graph model. In what follows we show that pairwise embeddings can address this challenge.

# 4 Size-stability of structural pairwise embeddings and its advantages

We have discussed the limitation of gMPNNs\* on node representation for link prediction. Now we claim that a joint continuous message passing graph neural network is capable of link prediction in graphon random graph models (Definition 1). We define the joint continuous message passing graph neural network inspired by the cMPNNs\* for node representations (Definition 5). First, we need to define the graphon fraction of common neighbors for graphon nodes  $x$  and  $y$ ,  $c_{W}(x,y) \coloneqq \int_{\mathcal{X}} W(x,z)W(y,z)d\mu(z)$ . We only consider graphons  $W$  such that there exists  $d_{cmin}$  satisfying  $c_{W}(x,y) \geq d_{cmin} > 0, \forall x,y \in \mathcal{X}$  in this section. Since we do not have edge feature as in Definition 1, we define the metric-space pair-wise signal as  $f^{\bullet \bullet}(x,y) = 1, \forall x,y \in \mathcal{X}$ .

Definition 9 (cMPNN $\bullet$ ). Let  $(W, f)$  be a random graph model as in Definition 1 and  $\Theta$  be a MPNN as in Definition 2. For  $t = 1, \dots, T$ , define the continuous (pairwise) cMPNN $\bullet$ $\Theta_W^{\bullet\bullet(t)}$  as the mapping that maps input pairwise metric-space signals  $f^{\bullet\bullet(0)} = f^{\bullet\bullet}$  to the features in the  $t$ -th layer by

$$
\Theta_ {W} ^ {\bullet \bullet (t)}: L ^ {2} (\mathcal {X}, \mathcal {X}) \to L ^ {2} (\mathcal {X}, \mathcal {X}), \quad f ^ {\bullet \bullet^ {(0)}} \mapsto f ^ {\bullet \bullet^ {(t)}},
$$

where  $f^{\bullet \bullet (t)}$  are defined recursively by

$$
\begin{array}{l} g ^ {\bullet \bullet (t)} (x, y) := M _ {W} ^ {\bullet \bullet} (\Phi^ {(t)} (f ^ {\bullet \bullet (t - 1)})) (x, y) = \frac {1}{2} \int_ {\mathcal {X}} \left(\frac {W (y , z)}{c _ {W} (x , y)} \Phi^ {(t)} (f ^ {\bullet \bullet (t - 1)} (x, y), f ^ {\bullet \bullet (t - 1)} (x, z)) \right. \\ + \frac {W (x , z)}{c _ {W} (x , y)} \Phi^ {(t)} \left(f ^ {\bullet \bullet (t - 1)} (x, y), f ^ {\bullet \bullet (t - 1)} (y, z)\right)) d \mu (z), \\ \end{array}
$$

$$
\begin{array}{l} g ^ {\bullet \bullet (t)} (x, y) := M _ {W} ^ {\bullet \bullet} (\Phi^ {(t)} (f ^ {\bullet \bullet (t - 1)})) (x, y) = \frac {1}{2} \int_ {\mathcal {X}} \left(\frac {W (y , z)}{c _ {W} (x , y)} \Phi^ {(t)} (f ^ {\bullet \bullet (t - 1)} (x, y), f ^ {\bullet \bullet (t - 1)} (x, z)) \right. \\ + \frac {W (x , z)}{c _ {W} (x , y)} \Phi^ {(t)} \left(f ^ {\bullet \bullet (t - 1)} (x, y), f ^ {\bullet \bullet (t - 1)} (y, z)\right)) d \mu (z), \\ f ^ {\bullet \bullet (t)} (x, y) := \Psi^ {(t)} \left(f ^ {\bullet \bullet (t - 1)} (x, y), g ^ {\bullet \bullet (t)} (x, y)\right). \\ \end{array}
$$

The intuition of the aggregation function is that two edges with one same node is considered neighbors in a higher-order graph [43], and to go from  $(x,y)$  to  $(x,z)$ , we need to transition from  $y$  to  $z$ , which has probability  $W(y,z)$ . The same holds for going from  $(x,y)$  to  $(y,z)$ .

Lemma 2. If  $\Phi(x, y) = y$  and  $\Psi(x, y) = x / y$ , then  $f^{\bullet \bullet(t)}(x, y) = W(x, y)$ ,  $\forall x, y \in \mathcal{X}$  is a stationary point in the cMPNN $^{\bullet \bullet}$ , i.e. if  $f^{\bullet \bullet(t-1)}(x, y) = W(x, y)$ , then  $f^{\bullet \bullet(t)}(x, y) = W(x, y)$ ,  $\forall x, y \in \mathcal{X}$ .

We define the corresponding gMPNN\* as follows. First we define the fraction of common neighbors between nodes  $i$  and  $j$  as  $c_{A_i,j} = \frac{1}{N}\sum_{z=1}^N A_{i,z} \cdot A_{j,z}$ . If two nodes do not have common neighbors, then we set  $c_{A_i,j} = \frac{1}{N}$  to avoid computation error. Further, we define  $\mathbf{f}^{\bullet \bullet}_{i,j} = 1 \forall i, j \in V$  for any graph  $G$ , and  $\mathbf{F}^{\bullet \bullet} = (\mathbf{f}^{\bullet \bullet}_{i,j})_{i,j \in V}$  as the pair-wise graph signals.

Definition 10 (gMPNN $\bullet$ ). Let  $(G,F)$  be a graph with graph signals as in Definition 1 and  $\Theta$  be a MPNN as in Definition 2. For  $t = 1,\dots,T$  layers we define the gMPNN $\bullet$ $\Theta_A^{\bullet\bullet(t)}$  as the mapping that maps input pairwise graph signals  $F^{\bullet\bullet(0)} = F^{\bullet\bullet}$  to the features in the  $t-th$  layer by

$$
\Theta_ {A} ^ {\bullet \bullet (t)}: \mathbb {R} ^ {N ^ {2} \times F _ {0}} \to \mathbb {R} ^ {N ^ {2} \times F _ {t}}, \boldsymbol {F} ^ {\bullet \bullet (0)} \mapsto \boldsymbol {F} ^ {\bullet \bullet (t)} = (\mathbf {f} _ {i, j} ^ {\bullet \bullet (t)}) _ {i, j = 1} ^ {N}
$$

where  $\mathbf{f}^{\bullet \bullet (t)}$  are defined recursively by the following function,

$$
\begin{array}{l} m ^ {\bullet \bullet (t)} _ {i, j} := \frac {1}{2 N} \sum_ {z = 1} ^ {N} \frac {A _ {j , z}}{c _ {A _ {i , j}}} \Phi^ {(t)} (\mathbf {f} _ {i, j} ^ {\bullet \bullet (t - 1)}, \mathbf {f} _ {i, z} ^ {\bullet \bullet (t - 1)}) + \frac {A _ {i , z}}{c _ {A _ {i , j}}} \Phi^ {(t)} (\mathbf {f} _ {i, j} ^ {\bullet \bullet (t - 1)}, \mathbf {f} _ {j, z} ^ {\bullet \bullet (t - 1)}), \\ \mathbf {f} _ {i, j} ^ {\bullet \bullet (t)} := \Psi^ {(t)} (\mathbf {f} _ {i, j} ^ {\bullet \bullet (t - 1)}, m _ {i, j} ^ {\bullet \bullet (t)}. \\ \end{array}
$$

Next, Theorem 2 proves a similar convergence results for cMPNN\* as Theorem 1 for cMPNN\*.

Theorem 2. Let  $\Theta = ((\Phi^{(l)})_{l=1}^T, (\Psi^{(l)})_{l=1}^T)$  be a MPNN as in Definition 2 with  $T$  layers such that  $\Phi^{(l)}$  and  $\Psi^{(l)}$  are Lipschitz continuous with Lipschitz constants  $L_{\Phi}^{(l)}$  and  $L_{\Psi}^{(l)}$ . Let  $gMPNN^{\bullet \bullet}$ ,  $\Theta_A^{\bullet \bullet (T)}$  and  $cMPNN^{\bullet \bullet}$ ,  $\Theta_W^{\bullet \bullet (T)}$  be as in Definitions 9 and 10. For a random graph model  $(W, f)$  as in Definition 1 with  $d_{cmin}$ , consider training  $(G^{tr}, F^{tr}) \sim (W, f)$  and test  $(G^{te}, F^{te}) \sim (W, f)$  graphs, where  $N^{te} > N^{tr}$ . Let  $X_1^{te}, \ldots, X_{N^{te}}^{te}$  and  $A^{te}$  be as in Definition 1. Let  $p \in (0, \frac{1}{\sum_{l=1}^T 2(H_l + 1)})$ . Then, if  $\frac{\sqrt{N^{te}}}{\sqrt{\log(2(N^{te})^2 / p)}} \geq \frac{4\sqrt{2}}{d_{cmin}}$ , we have with probability at least  $1 - \sum_{l=1}^T 2(H_l + 1)p$ ,

$$
\delta_ {A - W} ^ {\bullet \bullet} = \max  _ {i, j = 1, \dots , N ^ {t e}} \| \Theta_ {A} ^ {\bullet \bullet (T)} (\boldsymbol {F} ^ {\bullet \bullet}) _ {i, j} - \Theta_ {W} ^ {\bullet \bullet (T)} (f ^ {\bullet \bullet}) (X _ {i} ^ {t e}, X _ {j} ^ {t e}) \| _ {\infty} \leq (C _ {3} + C _ {4} \| f ^ {\bullet \bullet} \| _ {\infty}) \frac {\sqrt {\log (2 (N ^ {t e}) ^ {2} / p)}}{\sqrt {N ^ {t e}}},
$$

where the constants  $C_3$  and  $C_4$  are defined in the Appendix and depend on  $\{L_{\Phi}^{(l)}, L_{\Psi}^{(l)}\}_{l=1}^T$  and  $(G^{tr}, \mathbf{F}^{tr})$ ,  $N^{tr}$ .

Hence, as  $N^{\mathrm{te}}$  gets larger, the link predictor using gMPNN $^{\bullet \bullet}$  will converge to a continuous method (cMPNN $^{\bullet \bullet}$ ) that can predict links in OOD tasks (i.e.,  $W(X_{i}^{\mathrm{te}}, X_{j}^{\mathrm{te}})$  is a stationary solution of cMPNN $^{\bullet \bullet}$  per Lemma 2). This convergence is observed in our empirical results.

# 5 Further Related Work

In what follows we describe works related to learning transferability in GNNs. The concept of transferability of GNN is introduced by Levie et al. [32], Ruiz et al. [50], which state that if two graphs represent same phenomena (e.g., are sampled from the same distribution), then a transferable

![](images/e268d33425d41ee9212c3fbb279ffe462dcfaae683637c38a85eb018e53dc41a.jpg)  
Figure 2: Experimental agreement with theory: (a) shows  $\delta_{\mathrm{A - W}}^{\bullet}$  (Theorem 1) of a GraphSAGE GNN as a function of  $N^{\mathrm{te}}$ ; (b) shows  $\delta_{\mathrm{A - W}}^{\bullet \bullet}$  (Theorem 2) with the gMPNN $^{\bullet \bullet}$  of Lemma 2 as a function of  $N^{\mathrm{te}}$ ; (c) replicates (b) with  $\Psi$  as a randomly-initialized neural network. Results show close agreement with Theorems 1 and 2 that predicts slope  $\approx -1 / 2$  in log-log scale for large  $N^{\mathrm{te}}$ ; (d) shows stable node representations between isomorphic SBM blocks, while (e) shows constant difference in node representations between non-isomorphic SBM blocks, which validate Corollary 1.

![](images/7b1f611f4a32c3cbcf82d0d0cd05f5b8cefb4c25bd3fce4a672e87f5c705622d.jpg)

![](images/60df9b00894870afa7ded73b576baf985bf46e2cbfda5e012d473d03bfe66c96.jpg)

![](images/57ad838ce57c7b0eeead6a92f9c677b5cbd76fe4289751c3673341e0e9ba5495.jpg)

![](images/8dfbd7d9cab23430959e2dbfff16b25810e82e3351aeb188332e5a7368979931.jpg)

GNN has approximately the same predictive performance on both graphs. This is closely related to in-distribution generalization capabilities of GNNs to unseen test data, i.e., generalization error when train and test data come from the same distribution. Existing works [24, 38, 50, 51] prove the transferability for spectral-based GCNs under graphon models, and Maskey et al. [39] extends these results to more general message passing GNNs. Our results are also based on general message passing GNNs. Our goal (OOD link prediction) is, however, significantly different than these prior works, which focus on in-distribution graph and node classification. Related works relating to the representation power, higher order structural and positional link prediction methods (not already covered in our introduction) can be found in Appendix E due to space constraints.

# 6 Empirical Evaluation

In what follows we empirically validate our theoretical results in two parts. We implement all our models in Pytorch Geometric [17] and make it available<sup>1</sup>. Due to space constraints we relegate a detailed description of our experiments to the Appendix.

Convergence and stability. First we will empirically validate Theorems 1 and 2 and Corollary 1. Consider an SBM (Definition 6) with three blocks  $(r = 3)$  and  $S_{a,a} = 0.55$ ,  $a = 1,2,3$ ,  $S_{1,2} = S_{2,1} = 0.05$ ,  $S_{1,3} = S_{3,1} = 0.02$ . The probability a node belongs to block one or three is 0.45, while for block two it is 0.1. Note that one and three are isomorphic blocks (see Definition 7). Since our results are valid for any gMPNN functions  $\Theta$ , for our first experiment with node embeddings we use a randomly initialized GraphSAGE [22] GNN model, where following standard GNN procedures we initialize node features as size-normalized degrees (see  $d_i$  in Definition 3). For the experiment with pairwise embeddings, we test both the  $\Phi$  and  $\Psi$  of Lemma 2, and a scenario where  $\Psi$  is a randomly-initialized feedforward neural network. In the Appendix we show how to efficiently compute the exact cMPNN\* and cMPNN\*\* embeddings of our GraphSAGE and gMPNN\*\* models.

Figures 2(a-c) show log-log plots of the convergence of gMPNNs to their continuous cMPNN counterparts as the test graph size  $N^{\mathrm{te}}$  increases. The empirical approximation errors  $\delta_{\mathrm{A - W}}^{\bullet}$  (Theorem 1) (Figure 2(a)) and  $\delta_{\mathrm{A - W}}^{\bullet \bullet}$  (Theorem 2) are shown as a function of the test graph size  $N^{\mathrm{te}} = 2^{n}$ ,  $n = 5, \dots, 13$ . The empirical results show agreement with the theory since  $\delta_{\mathrm{A - W}}^{\bullet}$  and  $\delta_{\mathrm{A - W}}^{\bullet \bullet}$  are bounded above by  $O(\sqrt{\log N^{\mathrm{te}}} / \sqrt{N^{\mathrm{te}}})$ , which is approximated by the slope  $-1/2$  in a log-log plot.

Figures 2(d-e) show histograms of the difference between gMPNN\* embeddings of different nodes in  $G^{\mathrm{te}}$  . Let  $\Delta_{i,j}^{\bullet}\coloneqq \overline{\Theta}_{A}^{\bullet (T)}(F)_{i} - \overline{\Theta}_{A}^{\bullet (T)}(F)_{j}$  for  $i,j\in V^{\mathrm{te}}$ $\Delta_{i,j}^{\bullet}\in \mathbb{R}^{F_T}$  and further define  $\Delta_{\mathrm{iso(non - iso)}}^{\bullet}{}_{i,j}\coloneqq (\Delta_{i,j}^{\bullet})_{\arg \max_k|(\Delta_{i,j}^{\bullet})_k|}$  , where  $k\in \{1,\ldots ,F_T\}$  is the dimension of the embedding. We use subscript iso (resp. non-iso) when  $i,j\in V^{\mathrm{te}}$  are in isomorphic (resp. non-isomorphic) SBM blocks (Definition 7). As  $N^{\mathrm{te}}$  increases, Figure 2(d) shows that embeddings between isomorphic blocks converge, validating Corollary 1, while Figure 2(e) shows that non-isomorphic blocks do not.

Link prediction performance evaluation with SBMs (in-distribution and OOD). In what follows we introduce empirical results using the SBM described earlier. We start by sampling the training graph  $(G^{\mathrm{tr}}, F^{\mathrm{tr}})$  with  $N^{\mathrm{tr}} = 10^{3}$  nodes. We randomly split  $E^{\mathrm{tr}}$  into positive train  $(80\%)$  and validation  $(10\%)$  edges (we reserve  $10\%$  of  $E^{\mathrm{tr}}$  for the transductive test scenario), and uniformly sample the same number of across-block non-edges as negative train and validation edges. The embedding method gMPNN\* (resp. gMPNN\*\*) along with link predictor  $\eta^{\bullet}$  (resp.  $\eta^{\bullet \bullet}$ ) are trained in an end-to-end

Table 1: Test performance over 50 runs of node and pairwise gMPNNs for in-distribution and OOD link prediction over SBM graphs. Methods marked with * indicate best result out of distinct configurations detailed in the Appendix.  

<table><tr><td rowspan="2" colspan="2">Tasks</td><td rowspan="2">Model</td><td colspan="5">Training graph size Ntr = 103</td></tr><tr><td>Hit@10(%)</td><td>Hit@50(%)</td><td>Hit@100(%)</td><td>mcc(%)</td><td>balanced acc.(%)</td></tr><tr><td rowspan="14">In-distribution link prediction</td><td rowspan="7">Transductive</td><td>GraphSAGE*</td><td>95.12(0.46)</td><td>95.33(0.59)</td><td>95.47(0.64)</td><td>94.98(0.12)</td><td>97.43(0.06)</td></tr><tr><td>GCN*</td><td>95.47(0.55)</td><td>95.58(0.55)</td><td>95.66(0.54)</td><td>94.17(6.64)</td><td>96.96(3.80)</td></tr><tr><td>GAT*</td><td>87.71(22.14)</td><td>88.17(21.10)</td><td>88.67(19.84)</td><td>87.46(21.49)</td><td>93.46(11.09)</td></tr><tr><td>GIN*</td><td>94.97(1.35)</td><td>95.24(0.99)</td><td>95.41(0.80)</td><td>94.58(3.14)</td><td>97.20(1.81)</td></tr><tr><td>gMPNN** (fixed Ψ)</td><td>93.06(0.22)</td><td>93.13(0.22)</td><td>93.20(0.21)</td><td>93.22(0.20)</td><td>96.49(0.11)</td></tr><tr><td>gMPNN** (learn Ψ)</td><td>94.29(0.26)</td><td>94.69(0.26)</td><td>94.91(0.25)</td><td>93.43(0.14)</td><td>96.61(0.07)</td></tr><tr><td>Oracle</td><td>96.69(0.15)</td><td>96.69(0.15)</td><td>96.69(0.15)</td><td>93.25(0.17)</td><td>96.51(0.09)</td></tr><tr><td rowspan="7">Inductive Ntr = Ntr</td><td>GraphSAGE*</td><td>55.03(44.14)</td><td>55.99(43.95)</td><td>56.72(43.84)</td><td>25.83(46.11)</td><td>64.78(21.26)</td></tr><tr><td>GCN*</td><td>70.02(34.65)</td><td>71.74(33.88)</td><td>72.65(33.72)</td><td>21.80(44.01)</td><td>62.52(20.55)</td></tr><tr><td>GAT*</td><td>34.59(39.49)</td><td>34.91(39.76)</td><td>35.08(39.87)</td><td>22.62(39.14)</td><td>61.41(19.08)</td></tr><tr><td>GIN*</td><td>37.22(38.87)</td><td>37.92(38.89)</td><td>38.31(38.89)</td><td>21.99(40.54)</td><td>61.56(18.81)</td></tr><tr><td>gMPNN** (fixed Ψ)</td><td>93.03(0.17)</td><td>93.10(0.17)</td><td>93.17(0.18)</td><td>93.20(0.16)</td><td>96.48(0.09)</td></tr><tr><td>gMPNN** (learn Ψ)</td><td>96.03(0.18)</td><td>96.30(0.16)</td><td>96.41(0.15)</td><td>94.74(0.20)</td><td>97.30(0.10)</td></tr><tr><td>Oracle</td><td>96.68(0.11)</td><td>96.68(0.11)</td><td>96.68(0.11)</td><td>93.26(0.15)</td><td>96.52(0.08)</td></tr><tr><td rowspan="7">OOD link prediction</td><td rowspan="7">Inductive Ntr = 104</td><td>GraphSAGE*</td><td>9.52(21.73)</td><td>10.24(22.53)</td><td>11.09(24.18)</td><td>-6.72(4.87)</td><td>49.32(0.57)</td></tr><tr><td>GCN*</td><td>27.12(31.14)</td><td>29.24(30.83)</td><td>30.07(31.02)</td><td>-6.98(4.18)</td><td>49.35(0.41)</td></tr><tr><td>GAT*</td><td>13.35(22.55)</td><td>14.10(22.97)</td><td>14.55(23.20)</td><td>-2.41(3.95)</td><td>49.79(0.37)</td></tr><tr><td>GIN*</td><td>0.04(0.26)</td><td>0.04(0.26)</td><td>0.04(0.26)</td><td>-2.44(5.05)</td><td>49.73(0.59)</td></tr><tr><td>gMPNN** (fixed Ψ)</td><td>96.18(0.07)</td><td>96.48(0.02)</td><td>96.56(0.02)</td><td>93.26(0.01)</td><td>96.52(0.01)</td></tr><tr><td>gMPNN** (learn Ψ)</td><td>96.70(0.01)</td><td>96.71(0.01)</td><td>96.72(0.01)</td><td>94.39(0.44)</td><td>97.12(0.23)</td></tr><tr><td>Oracle</td><td>96.70(0.01)</td><td>96.70(0.01)</td><td>96.70(0.01)</td><td>93.26(0.02)</td><td>96.51(0.01)</td></tr></table>

manner for predicting positive and negative edges in training using cross-entropy loss. The model achieving the best validation accuracy is then chosen to be applied in test. Our experiments consider three scenarios (in all scenarios we use the same number of negative test edges as positive test edges, sampled from non-edges in  $G^{\mathrm{te}}$  with endpoints in different isomorphic blocks): (i) (In-distribution) transductive scenario where  $G^{\mathrm{te}} = G^{\mathrm{tr}}$ , where positive test edges are the  $10\%$  reserved in  $E^{\mathrm{tr}}$  not used in training or validation; (ii) In-distribution inductive scenario where  $G^{\mathrm{te}}$  is sampled from the same SBM with  $N^{\mathrm{te}} = N^{\mathrm{tr}}$ , where we sample  $0.1|E^{\mathrm{tr}}|$  positive test edges from  $E^{\mathrm{te}}$  (for fair comparison across all scenarios); (c) OOD inductive scenario where  $G^{\mathrm{te}}$  is sampled from the same SBM with  $N^{\mathrm{te}} = 10 \times N^{\mathrm{tr}}$ , where we sample  $0.1|E^{\mathrm{tr}}|$  positive test edges from  $E^{\mathrm{te}}$ .

For structural node embeddings we consider GraphSAGE [22], GCN [25] (without positional features), GAT [62] and GIN [67] as the representatives of gMPNN\* models. The link predictor  $\eta^{\bullet}$  is as feedforward network (with 3 hidden layers and 10 neurons each) that receives the two node embeddings as input, and has link prediction threshold  $\tau = 0.5$  (see Definition 8 for details).

For structural pairwise embeddings we choose our proposed gMPNN\*\* method of Definition 10, since we can prove that our approach is theoretically sound in Lemma 2. We test  $\mathrm{gMPNN}^{\bullet \bullet}$  in two versions: The  $\Phi$  and  $\Psi$  functions in Lemma 2 (denoted fixed  $\Psi$ ) and a feedforward neural network for  $\Psi$  with 2 hidden layers and 5 neurons each (denoted learn  $\Psi$ ). The link predictor  $\eta^{\bullet \bullet}$  is the same as  $\eta^{\bullet}$  except it just takes one pairwise embedding as input, rather than two node embeddings.

Table 1 presents our empirical results. The oracle predictor knows the graphon values  $W(X_{i}^{\mathrm{te}},X_{j}^{\mathrm{te}})$ . Our evaluation metrics include the Matthews correlation coefficient (mcc) [40], balanced accuracy, and Hits@K for  $K = 10,50,100$  that counts the ratio of positive edges ranked at the  $k$ -th place or above against all negative edges. Note that gMPNN\* structural node representations can very accurately predict links in the transductive tasks, and still performs reasonably well in inductive in-distribution tasks. However, as expected from Corollary 2, this performance suffers significantly as  $N^{\mathrm{te}}$  becomes  $10\times$  larger than  $N^{\mathrm{tr}}$ . Now all gMPNN\* methods produce predictors that are no better than a random guess over all metrics (e.g., see OOD mcc and accuracy (in red)). In contrast, the gMPNN\* is able to consistently offer good performance on both in-distribution and OOD tasks.

# 7 Conclusions

This work studied and provided the first theoretical framework for the task of out-of-distribution (OOD) link prediction, where test graphs are larger than training graphs. Using non-asymptotic bounds, this work showed that OOD link prediction methods using structural node embeddings given by message-passing GNNs converge to link predictors that may perform no better than random guesses. The work also proposed a theoretically-sound structural pairwise embedding with a message-passing algorithm which converges to its continuous version that is proven to be able to perform OOD link prediction tasks. Extensive empirical evaluation showed agreement with these theoretical results. We do not foresee adverse social impacts for this theoretical work.

# References

[1] Lada A Adamic and Eytan Adar. Friends and neighbors on the web. Social networks, 25(3): 211-230, 2003.  
[2] Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy, Vanja Josifovski, and Alexander J Smola. Distributed large-scale natural graph factorization. In Proceedings of the 22nd international conference on World Wide Web, pages 37-48, 2013.  
[3] Edo M Airoldi, Thiago B Costa, and Stanley H Chan. Stochastic blockmodel approximation of a graphon: Theory and consistent estimation. Advances in Neural Information Processing Systems, 26, 2013.  
[4] Gabor Angeli and Christopher D Manning. Philosophers are mortal: Inferring the truth of unseen facts. In Proceedings of the seventeenth conference on computational natural language learning, pages 133-142, 2013.  
[5] Robert M Bell and Yehuda Koren. Lessons from the netflix prize challenge. Acm Sigkdd Explorations Newsletter, 9(2):75-79, 2007.  
[6] Beatrice Bevilacqua, Yangze Zhou, and Bruno Ribeiro. Size-invariant graph representations for graph classification extrapolations. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 837-851. PMLR, 2021.  
[7] Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34 (4):18-42, 2017.  
[8] Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations, 2014.  
[9] Abhinandan S Das, Mayur Datar, Ashutosh Garg, and Shyam Rajaram. Google news personalization: scalable online collaborative filtering. In Proceedings of the 16th international conference on World Wide Web, pages 271-280, 2007.  
[10] Luc De Raedt. Logical and relational learning. Springer Science & Business Media, 2008.  
[11] Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29, 2016.  
[12] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
[13] Persi Diaconis and Svante Janson. Graph limits and exchangeable random graphs. arXiv preprint arXiv:0712.2749, 2007.  
[14] Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. arXiv preprint arXiv:2012.09699, 2020.  
[15] Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs. AAAI Workshop on Deep Learning on Graphs: Methods and Applications, 2021.  
[16] Vijay Prakash Dwivedi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Graph neural networks with learnable structural and positional representations. In International Conference on Learning Representations, 2022.  
[17] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
[18] Lise Getoor and Christopher P Diehl. Link mining: a survey. Acm Sigkdd Explorations Newsletter, 7(2):3-12, 2005.

[19] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1263–1272. PMLR, 2017.  
[20] Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Proceedings. 2005 IEEE international joint conference on neural networks, volume 2, pages 729-734, 2005.  
[21] Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In Proc. of KDD, pages 855-864. ACM, 2016.  
[22] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.  
[23] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems, 33:22118-22133, 2020.  
[24] Nicolas Keriven, Alberto Bietti, and Samuel Vaiter. Convergence and stability of graph convolutional networks on large random graphs. Advances in Neural Information Processing Systems, 33:21512-21523, 2020.  
[25] Thomas Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.  
[26] Thomas N Kipf and Max Welling. Variational graph auto-encoders. NIPS Workshop on Bayesian Deep Learning, 2016.  
[27] Daphne Koller, Nir Friedman, Sašo Džeroski, Charles Sutton, Andrew McCallum, Avi Pfeffer, Pieter Abbeel, Ming-Fai Wong, Chris Meek, Jennifer Neville, et al. Introduction to statistical relational learning. MIT press, 2007.  
[28] Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30-37, 2009.  
[29] Yehuda Koren, Steffen Rendle, and Robert Bell. Advances in collaborative filtering. Recommender systems handbook, pages 91-142, 2022.  
[30] Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Létourneau, and Prudencio Tossou. Rethinking graph transformers with spectral attention. Advances in Neural Information Processing Systems, 34, 2021.  
[31] Neil Lawrence and Aapo Hyvarinen. Probabilistic non-linear principal component analysis with gaussian process latent variable models. Journal of machine learning research, 6(11), 2005.  
[32] Ron Levie, Wei Huang, Lorenzo Bucci, Michael Bronstein, and Gitta Kutyniok. Transferability of spectral graph convolutional neural networks. Journal of Machine Learning Research, 22 (272):1-59, 2021.  
[33] Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec. Distance encoding: Design provably more powerful neural networks for graph representation learning. Advances in Neural Information Processing Systems, 33:4465-4478, 2020.  
[34] David Liben-Nowell and Jon Kleinberg. The link-prediction problem for social networks. Journal of the American society for information science and technology, 58(7):1019-1031, 2007.  
[35] Greg Linden, Brent Smith, and Jeremy York. Amazon.com recommendations: Item-to-item collaborative filtering. IEEE Internet computing, 7(1):76-80, 2003.  
[36] László Lovász and Balázs Szegedy. The automorphism group of a graphon. Journal of Algebra, 421:136-166, 2015.

[37] Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph networks. In Advances in Neural Information Processing Systems, pages 2156-2167, 2019.  
[38] Sohir Maskey, Ron Levie, and Gitta Kutyniok. Transferability of graph neural networks: an extended graphon approach. arXiv preprint arXiv:2109.10096, 2021.  
[39] Sohir Maskey, Yunseok Lee, Ron Levie, and Gitta Kutyniok. Stability and generalization capabilities of message passing graph neural networks. arXiv preprint arXiv:2202.00645, 2022.  
[40] Brian W Matthews. Comparison of the predicted and observed secondary structure of t4 phage lysozyme. Biochimica et Biophysica Acta (BBA)-Protein Structure, 405(2):442-451, 1975.  
[41] Andriy Mnih and Russ R Salakhutdinov. Probabilistic matrix factorization. Advances in neural information processing systems, 20, 2007.  
[42] Federico Monti, Oleksandr Shchur, Aleksandar Bojchevski, Or Litany, Stephan Gunnemann, and Michael M Bronstein. Dual-primal graph convolutional networks. arXiv preprint arXiv:1806.00770, 2018.  
[43] Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 4602-4609, 2019.  
[44] Christopher Morris, Yaron Lipman, Haggai Maron, Bastian Rieck, Nils M Kriege, Martin Grohe, Matthias Fey, and Karsten Borgwardt. Weisfeiler and leman go machine learning: The story so far. arXiv preprint arXiv:2112.09992, 2021.  
[45] R. Murphy, B. Srinivasan, V. Rao, and B. Ribeiro. Janossy pooling: Learning deep permutation-invariant functions for variable-size inputs. In International Conference on Learning Representations, 2019.  
[46] Ryan Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Relational pooling for graph representations. In Proceedings of the 36th International Conference on Machine Learning, 2019.  
[47] Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 104(1):11-33, 2015.  
[48] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 701-710, 2014.  
[49] Yanjun Qi, Ziv Bar-Joseph, and Judith Klein-Seetharaman. Evaluation of different biological data and computational classification methods for use in protein interaction prediction. Proteins: Structure, Function, and Bioinformatics, 63(3):490-500, 2006.  
[50] Luana Ruiz, Luiz Chamon, and Alejandro Ribeiro. Graphon neural networks and the transferability of graph neural networks. Advances in Neural Information Processing Systems, 33: 1702-1712, 2020.  
[51] Luana Ruiz, Fernando Gama, and Alejandro Ribeiro. Graph neural networks: architectures, stability, and transferability. Proceedings of the IEEE, 109(5):660-682, 2021.  
[52] Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning. Advances in neural information processing systems, 30, 2017.  
[53] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008.  
[54] Brent Smith and Greg Linden. Two decades of recommender systems at amazon.com. IEEE internet computing, 21(3):12-18, 2017.

[55] Tom AB Snijders and Krzysztof Nowicki. Estimation and prediction for stochastic blockmodels for graphs with latent block structure. Journal of classification, 14(1):75-100, 1997.  
[56] Alessandro Sperduti and Antonina Starita. Supervised neural networks for the classification of structures. IEEE Transactions on Neural Networks, 8(3):714-735, 1997.  
[57] Balasubramaniam Srinivasan and Bruno Ribeiro. On the equivalence between positional node embeddings and structural graph representations. ICLR, 2020.  
[58] Ben Taskar, Ming-Fai Wong, Pieter Abbeel, and Daphne Koller. Link prediction in relational data. Advances in neural information processing systems, 16, 2003.  
[59] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In International conference on machine learning, pages 2071–2080. PMLR, 2016.  
[60] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.  
[61] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
[62] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. *ICLR*, 2018.  
[63] Haorui Wang, Haoteng Yin, Muhan Zhang, and Pan Li. Equivariant and stable positional encoding for more powerful graph neural networks. In International Conference on Learning Representations, 2022.  
[64] Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. Inductive representation learning in temporal networks via causal anonymous walks. In International Conference on Learning Representations, 2021.  
[65] Boris Weisfeiler and AA Lehman. A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsia, 2(9):12-16, 1968.  
[66] Patrick J Wolfe and Sofia C Olhede. Nonparametric graphon estimation. arXiv preprint arXiv:1309.5936, 2013.  
[67] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.  
[68] Jiaxuan You, Rex Ying, and Jure Leskovec. Position-aware graph neural networks. In International Conference on Machine Learning, pages 7134-7143. PMLR, 2019.  
[69] Jiaxuan You, Jonathan Gomes-Selman, Rex Ying, and Jure Leskovec. Identity-aware graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.  
[70] Muhan Zhang and Yixin Chen. Weisfeiler-lehman neural machine for link prediction. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '17, page 575-583, 2017.  
[71] Muhan Zhang and Yixin Chen. Link prediction based on graph neural networks. Advances in neural information processing systems, 31, 2018.  
[72] Muhan Zhang, Pan Li, Yinglong Xia, Kai Wang, and Long Jin. Labeling trick: A theory of using graph neural networks for multi-node representation learning. Advances in Neural Information Processing Systems, 34, 2021.  
[73] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. Neural bellman-ford networks: A general graph neural network framework for link prediction. Advances in Neural Information Processing Systems, 34, 2021.
