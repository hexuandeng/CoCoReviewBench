# Can Hybrid Geometric Scattering Networks Help Solve the Maximal Clique Problem?

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We propose a geometric scattering-based graph neural network (GNN) for approximating solutions of the NP-hard maximal clique (MC) problem. We construct a loss function with two terms, one which encourages the network to find a large set of nodes and the other which acts as a surrogate for the constraint that the nodes form a clique. We then use this loss to train a novel GNN architecture that outputs a vector representing the probability for each node to be part of the MC and apply a rule-based decoder to make our final prediction. The incorporation of the scattering transform alleviates the so-called oversmoothing problem that is often encountered in GNNs and would degrade the performance of our proposed setup. Our empirical results demonstrate that our method outperforms representative GNN baselines in terms of solution accuracy and inference speed as well as conventional solvers like GUROBI with limited time budgets.

# 1 Introduction

The success of Graph Neural Networks (GNNs) for a variety of machine learning tasks [Scarselli et al., 2008, Gori et al., 2005, Kipf and Welling, 2016, Gilmer et al., 2017] has sparked interests in using GNNs to solve graph combinatorial optimization (CO) problems [Li et al., 2018a, Karalias and Loukas, 2020]. For example, Joshi et al. [2019] introduces a GNN-based method for approximately solving the travelling salesman problem (TSP), while Karalias and Loukas [2020] approximate solutions of the maximal clique (MC) problem. Solving such problems in an end-to-end fashion via GNNs is challenging for several reasons. First of all, many CO problems are provably either NP-Hard or NP-complete. Therefore, learning CO solvers in a supervised or semi-supervised fashion can be computationally infeasible since the cost of generating the truth labels grows exponentially with the problem size. Second, the solution to the CO problems must satisfy a number of constraints or limitations (e.g. belonging to a clique). Although these constraints can be imposed during training time, there is still no guarantee that these constraints are still satisfied at test time. Furthermore, solving graph combinatorial problems largely depends on the expressive power of GNNs. Many GNNs aggregate information via local averaging, which can be interpreted as a smoothing operation. This degrades the expressive power of GNNs and results in the so-called oversmoothing problem [Li et al., 2018b] where neighboring nodes have similar representations and are difficult to distinguish from each other. This is problematic if a node that is not in the solution set borders many points that are in the solution set.

The purpose of this paper is to use a geometric scattering-based GNN to approximate the solution of the maximal clique (MC) problem, that is to find the largest complete subgraph contained within a large graph  $G$ . The use of geometric scattering, rather than a more traditional GNN is motivated by recent work [Wenkel et al., 2022] showing that the geometric scattering transform can help overcome the oversmoothing problem via the use of bandpass wavelet filters in conjunction with GCN-type

![](images/bd1ec4dfa5084a1c6310a8bb1557164a42902ef932308faf6916256282bd3ed6.jpg)  
Figure 1: Illustration of our model pipeline. We use a scattering-based GNN to learn a discriminative node representation and use a decoder to extract the MC from the learned representation.

filters. This is particularly important in the context of the MC problem because it is critical to distinguish a point which connects to many members of the clique from an actual member of the clique. As we shall show, our method, which utilizes bandpass wavelet filters is able to better detect the border between the MC and the rest of the graph. This parallels the traditional use of wavelets as edge detectors in image processing[Grossmann, 1988].  
Our method uses a two-phase strategy shown in Figure 1. We first use an unsupervised learning model to generate an efficient representation of how likely each node is to be part of the maximal clique. Then, this representations is fed into a constraint-preserving decoder to generate the maximal clique. Ideally, given a graph  $G$  with  $n$  nodes, we would build a representation  $p \in [0,1]^n$  where the nodes in the maximal clique are assigned the value 1 (True) while all other nodes are assigned 0 (False), so that we can 'cut' the MC from the graph via  $O(n)$  operations. In practice, we will build a representation which assigns large probabilities to the maximal clique nodes and small probabilities to the other nodes. Figure 2 (a) presents a situation where oversmoothing does not occur. The signal  $p$  is much larger on the nodes  $(1,2,3,4,6)$ , which form the maximal clique, than it is on the other nodes  $(5,7,8)$ . Therefore, the decoder can easily find the MC by, e.g., setting a threshold. Figure 2 (b), on the other hand, illustrates a situation in which oversmoothing does occur and  $p$  has less variability. Indeed, there are no blue colored nodes at all. In this setting, it is nearly impossible to distinguish node 7, which does not belong to the maximal clique, from the nodes 1, 6, and therefore extracting the MC from  $p$  is difficult if not impossible.

![](images/558286b83b37af49d6287313c79991ffc5f39742688951a5935932107d6f963b.jpg)  
(a)  
Figure 2: Illustration of the oversmoothing problem. Left: A discriminative representation, which can be interpreted as a probability for each node to belong to the maximal clique or not. Right: Due to oversmoothing problem, the GNN generates a smooth representation, making MC and non-MC nodes difficult to discriminate.

![](images/06809957b7f3370c664ffd8bd3827b8589bee7e4ac827e97282fb74cb1fac849.jpg)

![](images/253ef4e495d30b8856b55b64c31ef825ec8737a88826c3dbfed5588acd507bdd.jpg)  
(b)

![](images/9b9b58cfa83896886acb3dc93105e51c8812b6327ff93f90eb0601e1e8cb52c9.jpg)

# 2 Background and Related Work

# 2.1 Graph Signal Processing

Consider a graph  $G = (V, E)$  with a set of nodes (or vertices)  $V := \{v_1, \ldots, v_n\}$  and a set of undirected edges  $E \subset V \times V$ . For a function  $x: V \to \mathbb{R}$  defined on  $V$ , we will, in minor abuse of notation identify  $x$  with the vector where  $x[i] = x(v_i)$ . We let  $W \in \mathbb{R}^{n \times n}$  denote the adjacency matrix of  $G$  and define the symmetric normalized grah Laplacian by

$$
\boldsymbol {\mathcal {L}} := \boldsymbol {I} _ {n} - \boldsymbol {D} ^ {- 1 / 2} \boldsymbol {W} \boldsymbol {D} ^ {1 / 2},
$$

where  $d_{i} \coloneqq \sum_{j=1}^{n} W[v_{i}, v_{j}]$  is the degree of the node  $v_{i}$  and  $D \coloneqq \mathrm{diag}(d_{1}, \ldots, d_{n}) \in \mathbb{R}^{n \times n}$  is the degree matrix. We will let  $\mathbf{q}_{i}$  and  $\lambda_{i}$  denote the (normalized) eigenvectors and eigenvalues of  $\mathcal{L}$  with  $\mathcal{L}\mathbf{q}_{i} = \lambda_{i}\mathbf{q}_{i}$ , and make use of the eigendecomposition

$$
\boldsymbol {\mathcal {L}} = \boldsymbol {Q} \boldsymbol {\Lambda} \boldsymbol {Q} ^ {\top},
$$

where  $Q$  is the orthogonal matrix whose  $i$ -th column is the normalized eigenvector  $\pmb{q}_i$ , and  $\Lambda := \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ . Notably, the eigenvectors  $\pmb{q}_1, \dots, \pmb{q}_n$  can be seen as a generalization of Fourier modes in the context of graph domains with the corresponding eigenvalues representing the increasing frequencies  $0 \leq \lambda_1 \leq \dots \leq \lambda_n \leq 2$ . This link may be established by viewing frequency as a measure of variation, in particular the variation of (degree-normalized) Fourier modes across edges is  $\lambda_i = \pmb{q}_i^\top \pmb{\mathcal{L}} \pmb{q}_i = \sum_{\{u,v\} \in E} (\tilde{\pmb{q}}_i[u] - \tilde{\pmb{q}}_i[v])^2$  for  $\tilde{\pmb{q}}_i := D^{-1/2} \pmb{q}_i$ . The graph Fourier transform of a signal vector  $\pmb{x}$  is then defined as by  $\hat{\pmb{x}}[i] = \langle \pmb{x}, \pmb{q}_i \rangle$  and the inverse Fourier is given by  $\pmb{x} = \sum_{i=1}^{n} \hat{\pmb{x}}[i] \pmb{q}_i$ . Compactly, we write  $\hat{\pmb{x}} = Q^\top \pmb{x}$  and  $\pmb{x} = Q \hat{\pmb{x}}$ .

In the Euclidean setting, it is known that convolution in the spatial domain corresponds to multiplication in the Fourier domain. This fact has motivated work such as Shuman et al. [2016], which defines the convolution of a signal  $\pmb{x}$  with a filter  $\pmb{g}$  to be the unique vector verifying  $(\widehat{\pmb{g} \star \pmb{x}})[i] = \widehat{\pmb{g}}[i]\widehat{\pmb{x}}[i]$ , which implies that

$$
\boldsymbol {g} \star \boldsymbol {x} = \sum_ {i = 1} ^ {n} \hat {\boldsymbol {g}} [ i ] \hat {\boldsymbol {x}} [ i ] \boldsymbol {q} _ {i} = \sum_ {i = 1} ^ {n} \hat {\boldsymbol {g}} [ i ] \langle \boldsymbol {q} _ {i}, \boldsymbol {x} \rangle \boldsymbol {q} _ {i} = Q \widehat {\boldsymbol {G}} \boldsymbol {Q} ^ {\top} \boldsymbol {x},
$$

where  $\widehat{G} \coloneqq \mathrm{diag}(\widehat{\pmb{g}}) = \mathrm{diag}(\widehat{\pmb{g}}[1], \dots, \widehat{\pmb{g}}[n])$ . When incorporating this notion of convolution into a graph neural network, a common choice [e.g., Defferrard et al., 2016] is to require the coefficients  $\widehat{\pmb{g}}[i]$  to be polynomials of the eigenvalues  $\lambda_i, i \in [n]$ , i.e.,  $\widehat{\pmb{g}}[i] \coloneqq \sum_k \gamma_k \lambda_i^k$  in which case we have  $\widehat{G} = \sum_k \gamma_k \Lambda^k$ . This allows one to implement convolution in the spatial domain by verifying that

$$
\boldsymbol {g} \star \boldsymbol {x} = \sum_ {k} \gamma_ {k} \mathcal {L} ^ {k} \boldsymbol {x}.
$$

In particular, the entire method may be implemented without the need to diagonalize a matrix which is extremely expensive for large graphs.

# 2.2 Graph Convolutional Networks

In Kipf and Welling [2016], the authors set  $\hat{\pmb{g}}[i] \coloneqq \theta(2 - \lambda_i)$ , where  $\theta$  is a real number, which yields

$$
\boldsymbol {g} _ {\theta} \star \boldsymbol {x} = \theta (2 \boldsymbol {I} _ {n} - \mathcal {L}) = \theta (\boldsymbol {I} _ {n} + \boldsymbol {D} ^ {- 1 / 2} \boldsymbol {W} \boldsymbol {D} ^ {- 1 / 2}) \boldsymbol {x}. \tag {1}
$$

As the eigenvalues of the above convolutional filter lie in [0,2] and could lead to vanishing or exploding gradients, Kipf and Welling [2016] apply a renormalization trick which replaces  $I_n + D^{-1/2}WD^{-1/2}$  with

$$
\boldsymbol {A} := (D + \boldsymbol {I}) ^ {- 1 / 2} (\boldsymbol {W} + \boldsymbol {I}) (D + \boldsymbol {I}) ^ {- 1 / 2}.
$$

The layer-wise propagation rule of GCN is then defined by  $\pmb{x}_j^\ell = \sigma \left( \sum_{i=1}^{N_{\ell-1}} \theta_{ij}^\ell \pmb{A} \pmb{x}_i^{\ell-1} \right)$ , where  $\pmb{x}_i^{\ell-1} \in \mathbb{R}^n$  is the  $i$ -th feature vector in layer  $\ell-1$ ,  $\theta_{ij}^\ell$  is a trainable parameter and  $\sigma(.)$  is a nonlinear activation function,  $\pmb{x}_j^\ell$  is the  $j$ -th feature activation vector in layer  $\ell$ . We can further write this compactly in matrix notation, this gives

$$
\boldsymbol {X} ^ {\ell} = \sigma (\boldsymbol {A} \boldsymbol {X} ^ {\ell - 1} \boldsymbol {\Theta} ^ {\ell}), \tag {2}
$$

where  $\Theta^{\ell} \in \mathbb{R}^{N_{\ell-1} \times N_{\ell}}$  is the weight-matrix of layer  $\ell$  and  $X^{\ell} \in \mathbb{R}^{n \times N_{\ell}}$  contains the activations output by layer  $\ell$ . The GCN model updates node features at every node by averaging over the node features of the node itself and its neighbors, which enforces similarity throughout node neighborhoods. This makes nodes increasingly difficult to discriminate for 'deeper' models and is widely referred to as the oversmoothing problem [Li et al., 2018b]. From a graph spectral theory perspective, oversmoothing is related to lowpass filtering as the filter in Eq. 1 puts larger weights on the low-frequency spectrum (as  $0 \leq \lambda_i \leq 2$ ). This motivates the idea of using GCN-type lowpass filters in conjunction with bandpass filters, that can be implemented, for example, using graph scattering [Min et al., 2020], which we discuss in the following.

# 2.3 Graph Scattering

The graph scattering transform [Gama et al., 2019, Zou and Lerman, 2019, Gao et al., 2019] is a wavelet-based model for machine learning on graphs. Unlike the one-hop localized low-pass filters used in GCN, which aim to promote smoothness between neighboring nodes, these wavelets are band-pass filters, which in turn incorporate long-range dependencies through the large spatial support of the used aggregations. The scattering transform is based upon raising the lazy random walk matrix

$$
\boldsymbol {P} := \frac {1}{2} \left(\boldsymbol {I} _ {n} + \boldsymbol {W} \boldsymbol {D} ^ {- 1}\right)
$$

to different powers in order to capture the diffusion geometry of the graph  $G$  at various time scales. In particular, subtracting such powers allows us to detect changes in these diffusion geometries. Following the lead of Coifman and Maggioni [2006], for  $k \in \mathbb{N}_0$ , we define a wavelet matrix  $\Psi_k \in \mathbb{R}^{n \times n}$  at scale  $2^k$  by

$$
\Psi_ {0} := I _ {n} - P, \quad \Psi_ {k} := P ^ {2 ^ {k - 1}} - P ^ {2 ^ {k}}, \quad k \geq 1. \tag {3}
$$

Intuitively, at every node, these diffusion wavelets can be seen as a comparison operation that calculates the difference of the averaged features of two neighborhoods of different sizes (namely sizes  $2^{k-1}$  and  $2^k$ ). The geometric scattering transform is constructed through an alternating cascade on wavelet filterings and pointwise nonlinearities. Towards this end, we let  $\sigma: \mathbb{R} \to \mathbb{R}$  be a nonlinear function which is nonexpansive in the sense that  $|\sigma(x) - \sigma(y)| \leq |x - y|$ , and for a scattering path  $p := (k_1, \ldots, k_m)$ , we define

$$
\boldsymbol {U} _ {p} \boldsymbol {x} := \boldsymbol {\Psi} _ {k _ {m}} \circ \sigma \circ \boldsymbol {\Psi} _ {k _ {m - 1}} \dots \sigma \circ \boldsymbol {\Psi} _ {k _ {2}} \circ \sigma \circ \boldsymbol {\Psi} _ {k _ {1}} \boldsymbol {x}. \tag {4}
$$

# 3 Model

Our method is centered around three main components, (i) a hybrid scattering-GCN model  $M$  (Sec. 3.1-3.3) that transforms a small set of simple node-level statistics represented by the matrix  $\mathbf{X} \in \mathbb{R}^{n \times d}$  to a probability vector  $\mathbf{p} \in [0,1]^n$  representing the probabilities that each node is part of the maximal clique, (ii) an easy-to-optimize unsupervised loss function  $L$  (Sec. 3.4) that is small for "good" probability vectors  $\mathbf{p}^{\star}$ ; and (iii) a rule-based decoder  $D$  (Sec. 3.5) that maps the probability vector to the maximal clique  $C^{\star} \subset V$  of  $G$ .

$$
X \xrightarrow {M} p \xleftrightarrow {L} p ^ {\star} \xrightarrow {D} C ^ {\star}
$$

# 3.1 Embedding Module

The input of our model is a node feature matrix  $\mathbf{X} \in \mathbb{R}^{n \times d}$ , containing  $d$  features for each node. In our experiments, we set  $d = 3$  and let the features be the eccentricity, the clustering coefficient, and the logarithm of the degree of each vertex. The encoder transforms the node features to  $d_h$ -dimensional embeddings  $H^0$  using a multi-layer perceptron (MLP)  $\mathrm{m_{emb}}: \mathbb{R}^d \to \mathbb{R}^{d_h}$ , i.e.,

$$
\boldsymbol {H} ^ {0} := \operatorname {m} _ {\mathrm {e m b}} (\boldsymbol {X}).
$$

We will use  $H^0$  as the input to the diffusion model introduced in the next subsection.

# 3.2 Diffusion Module

The aggregation module consists of a cascade of  $K \in \mathbb{N}$  aggregation (or diffusion) layers with operations that are chosen for each node via an attention mechanism. Our network is based upon the framework introduced in Wenkel et al. [2022]. However, our method differs from Wenkel et al. [2022] by storing every intermediate node representation  $H^{\ell}, 0 \leq \ell \leq K$ , in a list of readouts.  
In each layer  $\ell$ , every node has access to node representations from set a of filters  $\mathcal{F}$  that contains a selection of lowpass and bandpass filters. Similar to Wenkel et al. [2022], our bandpass filters  $f_{\mathrm{low},r}$  are modified GCN filters that have the form

$$
f _ {\text {l o w}, r} \left(\boldsymbol {H} ^ {\ell - 1}\right) = \boldsymbol {A} ^ {r} \boldsymbol {H} ^ {\ell - 1} \tag {5}
$$

and are parameterized by the power  $r \geq 1$  of the matrix  $\mathbf{A}$ . Similarly, we define the scattering filter  $f_{\mathrm{band},k}$  of order  $k$  according to

$$
f _ {\text {b a n d}, k} \left(\boldsymbol {H} ^ {\ell - 1}\right) = \boldsymbol {\Psi} _ {k} \boldsymbol {H} ^ {\ell - 1}. \tag {6}
$$

Next, we want to obtain data-driven scores  $s_f(v)$  that determine the importance of filter  $f \in \mathcal{F}$  for node  $v \in V$ . We set  $H_f^\ell \coloneqq f(H^{\ell - 1})$  and calculate the scores via an attention mechanism, setting

$$
\boldsymbol {s} _ {f} ^ {\ell} := \sigma \left(\boldsymbol {H} _ {f} ^ {\ell} \| \boldsymbol {H} ^ {\ell - 1}\right) \boldsymbol {a} ^ {\ell}, \tag {7}
$$

which, at every node, calculates the dot product of the learnable attention vector  $\pmb{a}^{\ell} \in \mathbb{R}^{2d_h}$  with the filtered node representation of the node concatenated to its previous representation. We normalize the scores across the filters using the softmax function, i.e.,  $\alpha_{f}(v) = \mathrm{softmax}_{\mathcal{F}}(s_{f}(v))$ , and use the corresponding attention vectors  $\alpha_{f}^{\ell} \in \mathbb{R}^{n}$ ,  $f \in \mathcal{F}$  to update node representations via

$$
\boldsymbol {H} _ {\text {a g g}} ^ {\ell} := \sum_ {f \in \mathcal {F}} \alpha_ {f} ^ {\ell} \odot \boldsymbol {H} _ {f} ^ {\ell}. \tag {8}
$$

Notably, the softmax function enforces that every node is likely to focus on one single filter. We note that the method is compatible with multi-head attention but omit further details as it is not necessary for the present experiments. Finally, we transform the aggregated node representations via an MLP  $\mathbf{m}^{\ell}:\mathbb{R}^{d_h}\to \mathbb{R}^{d_h}$ , i.e.,  $\pmb{H}^{\ell}\coloneqq \mathrm{m}^{\ell}(\pmb{H}_{\mathrm{agg}}^{\ell})$

# 3.3 Output Module

In the output module, we combine the information from the readouts  $H^{\ell}, 0 \leq \ell \leq K$ . We first concatenate the readouts horizontally to

$$
\boldsymbol {H} _ {\text {c a t}} := \| _ {\ell = 0} ^ {K} \boldsymbol {H} ^ {\ell},
$$

and then transform  $H_{\mathrm{cat}}$  to a vector  $\pmb{h} \in \mathbb{R}^n$  using an MLP, i.e.,

$$
\boldsymbol {h} := \operatorname {m o u t} \left(\boldsymbol {H} _ {\text {c a t}}\right).
$$

Finally, to obtain a probability vector, we apply min-max normalization, i.e.,

$$
\boldsymbol {p} := (\boldsymbol {h} - \min  (\boldsymbol {h}) \cdot \mathbf {1} _ {n}) / (\max  (\boldsymbol {h}) - \min  (\boldsymbol {h})),
$$

and interpret  $\pmb{p}[i]$  as the probability that the  $i$ -th node is part of the maximal clique.

# 3.4 Training Loss

We now derive an unsupervised training loss for the maximal clique problem. Letting  $\Omega$  denote the set of cliques, we note that the task of finding the maximal clique is equivalent to identifying the clique  $C\in \Omega$  that contains the most edges, i.e., maximizes the objective function

$$
L ^ {\star} (C) := \sum_ {u, v \in C} w _ {u, v}.
$$

Our approach relies on producing a vector  $\pmb{p}$  such that  $p_v\approx 1$  if  $v$  is in the maximal clique and  $\pmb {p}_v\approx 0$  otherwise. We consider the vector  $\pmb {x}\in \mathbb{R}^n$  where  $\pmb {x}_v = 1$  if  $v$  is in the maximal clique and  $\pmb {x}_v = 0$  otherwise. We then model  $\pmb {x}_v$  as a Bernoulli random variable where  $\mathbb{P}(\pmb {x}_v = 1) = \mathbf{p}_v$ . Our goal then becomes to maximize the expectation

$$
\mathbb {E} \left[ L ^ {\star} (C) \right] = \mathbb {E} \left[ \sum_ {(u, v) \in E} w _ {u, v} \boldsymbol {x} _ {u} \boldsymbol {x} _ {v} \right] = \sum_ {(u, v) \in E} w _ {u, v} \boldsymbol {p} _ {u} \boldsymbol {p} _ {v} = \boldsymbol {p} ^ {\top} \boldsymbol {W} \boldsymbol {p}. \tag {9}
$$

124 Maximizing (9) encourages  $\pmb{p}$  to be large on the support of the largest clique. However, we also the support of  $\pmb{p}$  to be concentrated within a clique. This naturally motivates the constrained optimization problem of minimizing

$$
L _ {1} (\boldsymbol {p}) := - \boldsymbol {p} ^ {\top} \boldsymbol {W} \boldsymbol {p} \quad \text {s u b j e c t t o :} \operatorname {s u p p} (\boldsymbol {p}) \text {i s c o n t a i n e d i n a c l i q u e .} \tag {10}
$$

However, the constraint that  $\mathbf{p}$  is contained in a clique is hard to enforce directly. In order to construct a surrogate, we consider the complement graph  $\overline{G} = (\overline{V},\overline{E})$ . The vertices of  $\overline{G}$  are taken to be the same as  $G$ , i.e.,  $\overline{V} = V$ , while the edges are defined by the rule that for  $u\neq v$  there is an edge between  $u$  and  $v$  in  $\overline{G}$  if and only if there is not an edge between them in  $G$ , i.e.,  $\overline{E} = \{(u,v)\in V\times V:u\neq v\} \backslash E$ . We denote the adjacency matrix of  $\overline{G}$  by  $\overline{W} = (\overline{w}_{u,v})_{u,v\in V}$ , and note that

$$
\overline {{w}} _ {u, v} := \left\{ \begin{array}{l l} 1 & \text {i f} (u, v) \notin E \text {a n d} u \neq v, \\ 0 & \text {o t h e r w i s e}. \end{array} \right.
$$

In particular,  $\overline{W}$  has zeros on the diagonal and can be computed from  $\mathbf{W}$  via  $\overline{\mathbf{W}} = \mathbf{1}_{n\times n} - (\mathbf{I} + \mathbf{W})$ . The following Lemma indicates that a second loss term  $L_{2}(\pmb {p})\coloneqq \pmb{p}^{\top}\overline{\mathbf{W}}\pmb{p}$  can be used to ensure that mass is primarily concentrated in a set of nodes that form a clique.

Lemma 1. Consider a graph  $G = (V, E)$ , a signal  $\pmb{p} \geq 0$  and define  $\operatorname{supp}(\pmb{p}) := \{v \in V : \pmb{p}_v > 0\}$ . Then,  $L_2(\pmb{p}) = \pmb{p}^\top \overline{\pmb{W}} \pmb{p} = 0$  if and only if there exists a clique  $C \subset V$  such that the support of  $\mathbf{p}$  is contained in  $C$ .

Proof. We first assume  $\pmb{p}^{\top}\overline{\pmb{W}}\pmb{p} = 0$  and suppose for the sake of a contradiction that there exists no clique  $C\subset V$  with  $\mathrm{supp}(\pmb {p})\subset C$ . Then, there exist at least two nodes  $u_0,v_0\in \operatorname {supp}(\pmb {p})$  with  $\{u_0,v_0\} \notin E$ . Now,

$$
\boldsymbol {p} ^ {\top} \overline {{\boldsymbol {W}}} \boldsymbol {p} = \sum_ {\{u, v \} \notin E} \boldsymbol {p} _ {u} \boldsymbol {p} _ {v} \geq \boldsymbol {p} _ {u _ {0}} \boldsymbol {p} _ {v _ {0}} > 0,
$$

which is a contradiction.

For the opposite direction, we assume that there exists a clique  $C \subset V$  such that  $\operatorname{supp}(\pmb{p}) \subset C$ . Hence, every pair of distinct nodes in  $\operatorname{supp}(\pmb{p})$  is connected by an edge and consequently  $\overline{w}_{u,v} = 0$  for all  $u, v \in \operatorname{supp}(\pmb{p})$ . Thus,

$$
\boldsymbol {p} ^ {\top} \overline {{\boldsymbol {W}}} \boldsymbol {p} = \sum_ {u, v \in V} \overline {{w}} _ {u, v} \boldsymbol {p} _ {u} \boldsymbol {p} _ {v} = \sum_ {u, v \in \operatorname {s u p p} (\boldsymbol {p})} \overline {{w}} _ {u, v} \boldsymbol {p} _ {u} \boldsymbol {p} _ {v} = 0.
$$

140

In light of (9) and Lemma 1, we now define our training loss as

$$
L (\boldsymbol {p}) := L _ {1} (\boldsymbol {p}) + \beta L _ {2} (\boldsymbol {p}) = - \boldsymbol {p} ^ {T} \boldsymbol {W} \boldsymbol {p} + \beta \boldsymbol {p} ^ {T} \overline {{\boldsymbol {W}}} \boldsymbol {p}. \tag {11}
$$

We note that the loss function decomposes in two terms. The first term,  $L_{1}$ , encourages the allocation of mass at highly connected nodes, while the second term  $L_{2}$  encourages most of the mass of  $\mathbf{p}$  to be contained in a clique.  $\beta$  is a hyper-parameter which is used to balance the contributions of the two terms.

# 3.5 Decoder

In order to extract the MC from  $\pmb{p}$ , we use a greedy decoder detailed in in Alg. 1. Our algorithm creates a set  $\Omega = \{\hat{C}_j\}_{j=1}^{\kappa}$  of  $\kappa$  cliques and then selects the  $\hat{C}_j$  with largest cardinality. To construct the  $\hat{C}_j$ , we first create a sorting function  $\pi: V \to \{1, \dots, |V|\}$  which puts the nodes in descending order, so

$$
\mathbf {p} _ {\pi^ {- 1} (1)} \geq \dots \geq \mathbf {p} _ {\pi^ {- 1} (n)}.
$$

Each  $\hat{C}_j$  starts as a single vertex  $\pi^{-1}(j)$ . We then consider  $\tau - 1$ , additional candidate,  $\pi^{-1}(j + 1), \ldots, \pi^{-1}(\tau)$  and accept each candidate vertex  $v_{\mathrm{candidate}}$  if  $\hat{C}_j \cup \{v_{\mathrm{candidate}}\}$  forms a clique. This is tested by verifying that  $\boldsymbol{x}^{\top} \overline{\boldsymbol{W}} \boldsymbol{x} = 0$  for

$$
\boldsymbol {x} := \sum_ {v \in \hat {C} _ {j} \cup \{v _ {\text {c a n d i d a t e}} \}} \mathbf {1} _ {v}.
$$

We remark that in order for our decoder to predict true maximal clique exactly, we must choose the parameters  $\tau$  and  $\kappa$  such that there exists a  $j$ ,  $1 \leq j \leq \kappa$ , such that  $j \leq \pi^{-1}(i) \leq j + \tau$  for all  $v_{i}$  in the MC. On simple datasets, such as IMDB, it suffices to set  $\kappa = 1$ . More complex datasets

Algorithm 1 Maximal Clique Decoder  
Input: Probability vector  $\pmb{p}$  graph  $G$  with  $n$  nodes, number of samplers  $\kappa$  sample length  $\tau$    
Output: Predicted maximal clique  $\hat{C}$  of size  $|\hat{C} |$    
Order nodes via permutation  $\pi :V\to [n]$  such that  $\pmb{p}_{\pi^{-1}(1)}\geq \pmb{p}_{\pi^{-1}(2)}\geq \dots \geq \pmb{p}_{\pi^{-1}(n)}$    
for  $j = 1$  to  $\kappa$  do   
 $\hat{C}_j\gets \{\pi^{-1}(j)\}$    
for  $i = 2$  to  $\tau$  do if  $\hat{C}_j\cup \{\pi^{-1}(j + i)\}$  forms a clique then   
 $\hat{C}_j\gets \hat{C}_j\cup \{\pi^{-1}(j + i)\}$    
end if   
end for   
end for   
 $\hat{\Omega}\gets \{\hat{C}_1,\ldots ,\hat{C}_\kappa \}$ $\hat{C}\gets \arg \max_{C\in \hat{\Omega}}|C|$

such as Twitter require larger values of  $\kappa$ . Importantly, we note that the  $\hat{C}_j$  can be computed in parallel, so increasing the value of  $\kappa$  does not create major scalability issues. We note that setting the hyperparameters of the decoder requires some prior knowledge of the largest sizes of the maximal cliques present in the data, which we tune using the validation set.

# 4 Results

We empirically experiment on three popular graph learning baselines, namely IMDB, COLLAB [Yanardag and Vishwanathan, 2015] and TWITTER [Yan et al., 2008]. We compare both against other neural network based methods and against an integer-programming-based solver. For the neural networks, Erdős' GNN [Karalias and Loukas, 2020] and RUN-CSP [Toenhoff et al., 2019], the authors provide two implementations, one optimized towards computational speed (fast), and the other optimized for best approximation score. The integer-programming-based solver, GUROBI 9.0[Gurobi Optimization, LLC, 2022], is in general more accurate than neural network based methods. However, it is slower on larger graphs. In order to highlight the contribution of our hybrid, scattering model, we also provide results for a pure lowpass model, which only uses GCN-type layers [Kipf and Welling, 2016] followed by our decoder module. We provide our results in Tab. 1, where we track each model's test approximation score and the average time needed to classify one graph of the dataset. Dataset statistics can be found in Tab. 2. For a given graph, the approximation score is computed by dividing the size of the clique found by the algorithm by the true MC. So if the MC of a graph has size 10 and the algorithm returns a clique of size 9, it would be given a score of  $90\%$ . We then average these scores over all graphs in each dataset.

For the IMDB dataset, our hybrid approach and most other methods (except RUN-CSP) solve the MC problem with a perfect  $(100\%)$  approximation score. However, our model significantly outperforms the other neural network-based models (Erdős' GNN and RUN-CSP) with respect to computation time (although the highly optimized GUROBI solver is slightly faster than our method on this dataset due to the small graph size). On COLLAB, our model almost reaches perfect performance  $(99.8\%)$  in  $0.035~\mathrm{S / G}$ , outperforming the other neural network based methods in both speed and approximation score. Some versions of the Gurobi solver are able to achieve  $99.9\%$  and  $100\%$  approximation scores, however, they are slower than our method on this dataset. Lastly, we experimented on the TWITTER dataset, the most challenging of the datasets considered here. Among neural networks, RUN-CSP (accurate) does exceptionally well at  $98.7\%$  approximation score our method is second best with  $95\%$ , while running faster than all compared neural networks, in particular, 3 times faster than RUN-CSP.

Figure 3 illustrates the differences between our hybrid scattering model and the low-pass GCN-based model[Kipf and Welling, 2016] and helps explain why the scattering-based model performs better across all three datasets. It shows that the probability vector  $p$  generated by our hybrid scattering model is a more discriminative representation than the one produced by the low-pass model. When using the low-pass model, most nodes are assigned high probabilities (marked in red). The decoder cannot tell which nodes have higher priorities given such a smooth output. This leads to the decoder accepting nodes which are not part of the MC. If the new clique is not a sub-clique of the MC, this

Table 1: Maximal clique test approximation ratio and average prediction time measured in seconds per graph (in brackets) on test graphs for the different baseline methods compared on across datasets. We set the number of samplers,  $\kappa$ , equal to 1, 3 and 10 on IMDB, COLLAB and TWITTER, respectively. We mark the two best neural network models with highest prediction accuracy bold and underlined and the best one. In case of the same performance, we use lower computation time as a tiebreaker. We provide results for GUROBI with 4 different time budgets.  

<table><tr><td>DATASET</td><td>IMDB</td><td>COLLAB</td><td>TWITTER</td></tr><tr><td>SCATTERINGCLIENTE</td><td>1.000 (8E-3)</td><td>0.998 ± 0.011 (0.035)</td><td>0.950 ± 0.062(0.12)</td></tr><tr><td>GCN (LOWPASS)</td><td>0.956 ± 0.109 (4E-3)</td><td>0.981 ± 0.085 (0.033)</td><td>0.887 ± 0.111(0.11)</td></tr><tr><td>ERDŐS (FAST)</td><td>1.000 (0.08)</td><td>0.982 ± 0.063 (0.10)</td><td>0.924 ± 0.133 (0.17)</td></tr><tr><td>ERDŐS (ACCURATE)</td><td>1.000 (0.10)</td><td>0.990 ± 0.042 (0.15)</td><td>0.942 ± 0.111 (0.42)</td></tr><tr><td>RUN-CSP (FAST)</td><td>0.823 ± 0.191 (0.11)</td><td>0.912 ± 0.188 (0.14)</td><td>0.909 ± 0.145 (0.21)</td></tr><tr><td>RUN-CSP (ACCURATE)</td><td>0.957 ± 0.089 (0.12)</td><td>0.987 ± 0.074 (0.19)</td><td>0.987 ± 0.063 (0.39)</td></tr><tr><td>GUROBI 9.0 (0.1s)</td><td>1.000 (1E-3)</td><td>0.982 ± 0.101 (0.05)</td><td>0.803 ± 0.258 (0.21)</td></tr><tr><td>GUROBI 9.0 (0.5s)</td><td>1.000 (1E-3)</td><td>0.997 ± 0.035 (0.06)</td><td>0.996 ± 0.019 (0.34)</td></tr><tr><td>GUROBI 9.0 (1s)</td><td>1.000 (1E-3)</td><td>0.999 ± 0.015 (0.06)</td><td>1.000 (0.34)</td></tr><tr><td>GUROBI 9.0 (5s)</td><td>1.000 (1E-3)</td><td>1.000 (0.06)</td><td>1.000 (0.35)</td></tr></table>

Table 2: Dataset statistics of IMDB, COLLAB and TWITTER.  

<table><tr><td>Dataset</td><td># Graphs</td><td>Avg. # Nodes</td><td>Avg. # Edges</td><td>Avg. MC Size</td></tr><tr><td>IMDB</td><td>1,000</td><td>19.8 ± 10.0</td><td>96.5 ± 105.6</td><td>10.2 ± 5.3</td></tr><tr><td>COLLAB</td><td>5,000</td><td>74.5 ± 62.3</td><td>2457.2 ± 6438.9</td><td>41.5 ± 48.5</td></tr><tr><td>TWITTER</td><td>973</td><td>131.8 ± 64.4</td><td>1709.3 ± 1559.7</td><td>15.2 ± 7.1</td></tr></table>

in turn can lead to the decoder rejecting nodes which are part of the MC. The scattering model on the other hand produces a less smooth representation. The nodes outside the MC are assigned lower probabilities so that our decoder can successfully find the MC. This is consistent with previous work [Wenkel et al., 2022] showing that the graph scattering transform helps overcome the oversmoothness problem in node classification. We emphasize that both of these models are trained with the same loss function and the same rule-based decoder to predict the maximal clique. Therefore, the differences illustrated in Figure 3 are the direct result using scattering. More comparison between the scattering model and low-pass model can be found in the appendix.

![](images/2637ace1306ad7f05cd1acd123320c4026576692c97b94326ac9730ff189e2f8.jpg)  
Figure 3: Comparison of the output probability vector  $\pmb{p}$  for our hybrid scattering model (left) and the low-pass model (right) on a graph taken from the TWITTER dataset with ground truth maximal clique size of 10. Our hybrid scattering model yields followed by our decoder yields the correct clique, while the lowpass model only concludes with a clique of size 8.

Table 3: Maximal clique test approximation ratio and average prediction time (measured in seconds per graph) on test graphs for our hybrid scattering model compared to the GUROBI solver with different time budgets. We use 1 sampler for the easy and medium difficulty and 10 samplers on hard dataset. Fastest computation time is marked in bold, while we underline the GUROBI solver with lowest time budget that outperforms our model.  

<table><tr><td>DATASET</td><td>EASY</td><td>MEDIUM</td><td>HARD</td></tr><tr><td>SCATTERINGCLIQUE</td><td>0.993 ±0.017 (0.050)</td><td>0.935 ± 0.102 (0.021)</td><td>0.846 ± 0.177 (0.184)</td></tr><tr><td>GUROBI 9.0 (0.1s)</td><td>0.991 (0.42)</td><td>0.806 ± 0.101 (0.41)</td><td>0.742 ± 0.258 (0.42)</td></tr><tr><td>GUROBI 9.0 (0.2s)</td><td>0.998 (0.49)</td><td>0.878 ± 0.035 (0.51G)</td><td>0.843 ± 0.019 (0.48)</td></tr><tr><td>GUROBI 9.0 (0.5s)</td><td>1.000 (0.55)</td><td>0.977 ± 0.015 (0.68)</td><td>0.962 (0.60)</td></tr><tr><td>GUROBI 9.0 (1s)</td><td>1.000 (0.65)</td><td>0.991 (0.91)</td><td>0.989 (0.70)</td></tr></table>

To further explore how our model is compared to the current state-of-the-art commercial solver GUROBI 9.0, we introduce a new dataset that contains three classes of graphs with different difficulties (easy, medium and hard) for the maximal clique prediction. The dataset generation is based on the approach in Xu et al. [2005], where we set the hardness parameter to 0.2, 0.5 and 0.8, respectively, to generate the easy, medium and hard cases, as shown in Table 4. Each hardness class case contains 1,000 graphs. We notice that our model outperforms GUROBI under limited time budget. The results are shown in Table 3. Notably, the scattering model is very efficient when dealing with medium hardness cases, where our model is 20 times faster than GUROBI when achieving similar approximation score.

Table 4: Dataset statistics of easy, medium and hard cases.  

<table><tr><td>Dataset</td><td># Graphs</td><td>Avg # Nodes</td><td>Avg # Edges</td><td>Avg. MC Size</td></tr><tr><td>Easy</td><td>1,000</td><td>209.0 ± 49.9</td><td>2293.6 ± 1147.8</td><td>16.0 ± 3.6</td></tr><tr><td>Medium</td><td>1,000</td><td>187.6 ± 62.7</td><td>3378.2 ± 2077.9</td><td>17.7 ± 8.2</td></tr><tr><td>Hard</td><td>1,000</td><td>181.3 ± 62.2</td><td>4165.3 ± 2780.2</td><td>21.0 ± 9.0</td></tr></table>

Overall, we remark that our hybrid model provides considerable speedup compared to both the other neural networks and to GUROBI for large graphs. We also note that our model takes fewer parameters than the Erdős' GNN model, which uses MLPs with 512 hidden units for each layer, while in our model, whereas the MLPs used in our method have either 8, 16, or 20 hidden units on IMDB, COLLAB, and TWITTER, respectively. This in turn suggests that the use of both GCN and scattering filters leads to a powerful feature extractors that help learn informative node representations with relatively few parameters.

# 5 Conclusion

In this paper, we show the expressive power of GNNs can be critical for solving graph CO problems. Our results suggest that low-pass models that generally enforce smoothness over graph neighborhoods due to oversmoothing, make the nodes indistinguishable, which leads to unfavorable node representations for the MC problem. Inspired by the geometric scattering transform, we propose a novel scattering-based GNN to overcome the oversmoothing problem. We further construct a novel two-term loss function which encourages our network both to find a large set of nodes and to find a set of nodes which are contained within a clique, which, in conjunction mimic the otherwise hard to optimize problem of finding maximal cliques. Our scattering model is able to generate efficient representations that enable us to learn probabilities for each node to be part of the maximal clique. Based on these node representations we approximate the correct maximal cliques using a constraint-preserving greedy decoder. Overall, our performance is competitive with other neural networks-based methods and commercial solvers in terms of time and accuracy.

# References

Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008.  
Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., volume 2, pages 729-734. IEEE, 2005.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263-1272. PMLR, 2017.  
Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Combinatorial optimization with graph convolutional networks and guided tree search. arXiv preprint arXiv:1810.10659, 2018a.  
Nikolaos Karalias and Andreas Loukas. Erdos goes neural: an unsupervised learning framework for combinatorial optimization on graphs. Advances in Neural Information Processing Systems, 33: 6659-6672, 2020.  
Chaitanya K Joshi, Thomas Laurent, and Xavier Bresson. An efficient graph convolutional network technique for the travelling salesman problem. arXiv preprint arXiv:1906.01227, 2019.  
Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Thirty-Second AAAI conference on artificial intelligence, 2018b.  
Frederik Wenkel, Yimeng Min, Matthew Hirn, Michael Perlmutter, and Guy Wolf. Overcoming oversmoothness in graph convolutional networks via hybrid scattering networks. arXiv preprint arXiv:2201.08932, 2022.  
Alex Grossmann. Wavelet transforms and edge detection. In Stochastic processes in physics and engineering, pages 149-157. Springer, 1988.  
David I Shuman, Benjamin Ricaud, and Pierre Vandergheynst. Vertex-frequency analysis on graphs. Applied and Computational Harmonic Analysis, 40(2):260-291, 2016.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29, 2016.  
Yimeng Min, Frederik Wenkel, and Guy Wolf. Scattering gcn: Overcoming oversmoothness in graph convolutional networks. Advances in Neural Information Processing Systems, 33, 2020.  
Fernando Gama, Alejandro Ribeiro, and Joan Bruna. Diffusion scattering transforms on graphs. In International Conference on Learning Representations, 2019.  
Dongmian Zou and Gilad Lerman. Graph convolutional neural networks via scattering. Applied and Computational Harmonic Analysis, 49(3)(3):1046-1074, 2019.  
Feng Gao, Guy Wolf, and Matthew Hiram. Geometric scattering for graph data analysis. In Proceedings of the 36th International Conference on Machine Learning, PMLR, volume 97, pages 2122-2131, 2019.  
Ronald R Coifman and Mauro Maggioni. Diffusion wavelets. Applied and computational harmonic analysis, 21(1):53-94, 2006.  
Pinar Yanardag and SVN Vishwanathan. Deep graph kernels. In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pages 1365-1374, 2015.  
Xifeng Yan, Hong Cheng, Jiawei Han, and Philip S Yu. Mining significant graph patterns by leap search. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pages 433-444, 2008.

Jan Toenhoff, Martin Ritzert, Hinrikus Wolf, and Martin Grohe. Run-csp: Unsupervised learning of message passing networks for binary constraint satisfaction problems. arXiv preprint arXiv:1909.08387, 2019.  
Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2022. URL https://www.gurobi.com.  
Ke Xu, Fr' ed' eric Boussemart, Fred Hemery, and Christophe Lecoutre. A simple model to generate hard satisfiable instances. In Proceedings of the 19th international joint conference on Artificial intelligence, pages 337-342, 2005.
