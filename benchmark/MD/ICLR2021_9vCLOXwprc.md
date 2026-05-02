# ITERATED GRAPH NEURAL NETWORK SYSTEM

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present Iterated Graph Neural Network System (IGNNNS), a new framework of Graph Neural Networks(GNN), which can deal with undirected graph, directed graph and multi-relational graph in a unified way. The core component of IGNNS is the Iterated Function System (IFS), which is an important research field in fractal geometry. The key idea of IGNNS is to use a pair of affine transformations to characterize the process of message passing between graph nodes and assign an adjoint probability vector to them to form an IFS layer with probability. After embedding in the latent space, the node features are sent to IFS layer for iterating, and then obtain the high-level representation of graph nodes. We also analyze the geometric properties of IGNNS from the perspective of dynamical system. We prove that if the IFS induced by IGNNS is contractive, then the fractal representation of graph nodes converges to the fractal set of IFS in Hausdorff distance and the ergodic representation of that converges to a constant matrix in Frobenius norm. A number of semi-supervised node classification experiments on citation network datasets such as Citeseer, Cora and Pubmed, we demonstrate that our approach outperforms related methods by a significant margin.

# 1 INTRODUCTION

GNN (Scarselli et al., 2009) has been proved to be effective in processing graph structured data, and has been widely used in natural language processing, computer vision, data mining, social network and biochemistry. In recent years, GNN has developed a variety of architectures, such as GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017), GAT (Velicković et al., 2018), DGI (Velicković et al., 2019), GIN (Xu et al., 2019), APPNP (Klicpera et al., 2019a) and GCNII (Ming Chen et al., 2020). These architectures have a common feature, that is, the representation of each node is updated using messages from its neighbors but without distinguishing the direction (or angle) of message passing between two nodes. Therefore, the above architectures can not deal with directed graph directly. In order to adapt to directed graph, Bi-GCN (Marcheggiani & Titov, 2017; Fu et al., 2019) and R-GCN (Schlichtkrull et al., 2017) are proposed. However, the message passing in the two directions is independent and lacks of interaction. Only in the output layer, they are concatenated together.

But in real life, message passing is interactive in different directions. For example, node A obtains a message from node B. After processing the message, node A not only passes it to the next node C, but also feeds back to node B. Suppose there are only two directions for message passing, forward and backward, represented by 0 or 1, respectively. The symbol space of the first generation message passing path is  $\{0,1\} = \{0,1\}^1$ , and that of the second generation message passing path is  $\{00,01,10,11\} = \{0,1\}^2$ . Generally, the symbol space of the  $n$ -th generation message passing path is  $\{0,1\}^n$  and the size of the symbol space is  $2^n$ . This means that the scope of message passing spreads with exponent 2. However, in Bi-GCN (similar to Bi-LSTM) and R-GCN architectures, the symbol space is  $\{\{0\}^n, \{1\}^n\}$ , and its size is 2, which indicates that a lot of information will be lost in the process of message passing.

How to characterize the above message passing patterns? We use two mappings to represent message passing process in two directions. Then the interactive passing of messages in different directions is equivalent to the composite operation of corresponding mappings. In addition, the direction of message passing is often random, so we endow the two mappings with an adjoint probability vector to reflect the randomness. Because the symbol space of the iterative path of the Iterated Function System (IFS) with two mappings is also  $\{0,1\}^n$  and the mapping is selected with a certain probabil-

![](images/5c685efb28ce21d6f3b11fcd47e97f63321f1afea3c8dbeffa3389b601546710.jpg)  
Figure 1: Message passing patterns. Where the symbol  $H$  is the representations of all the nodes. (a) An undirected graph is transformed into a directed graph in a natural way. (b) Regardless of direction, simply gather information from neighbors. (c) Message is passed in the same direction (forward or backward), and get two hidden representations independently. (d) Message passing not only occurs in the same direction, but also occurs interactively in different directions, which is more in line with the actual situation. For example, in layer 1, node 2 passes the processed message  $f_{1}(m_{2})$  to node 1, and then, in layer2, node 1 processes the received message  $f_{1}(m_{2})$  and returns the processed message  $f_{0}(f_{1}(m_{2}))$  to node 2.

ity, the iterative process of IFS is similar to the message passing process. In other words, the above message passing pattern can be described perfectly by IFS with probabilities. We naturally present the Iterative Graph Neural Network System (IGNNS), whose core layer is constructed by IFS. Figure 1 describes the differences in message passing patterns among GCN, Bi-GCN and IGNNS. At the same time, we regard undirected graph as a directed graph with equal probability of bidirectional message passing (see Figure 1(a)), so the IGNNS architecture can handle directed graph, undirected graph and multi-relational graph (see Section 5) in a unified way.

# 2 PRELIMINARIES

A graph  $\mathcal{G} = (V,E)$  is defined by its note set  $V = \{v_{1},v_{2},\dots,v_{N}\}$  and edge set  $E = \{(v_{i},v_{j})|v_{i},v_{j}\in V\}$ . Let  $\pmb{A}\in \mathbb{R}^{N}$  denote the adjacency matrix of  $\mathcal{G}$ , providing with relational information between nodes.  $\pmb{A}[i,j]$  denote  $i,j$ th element of  $\pmb{A}$ ,  $\pmb{A}[i,:]$  means the  $i$ th row, and  $\pmb{A}(:,j]$  means the  $j$ th column. In this paper, we assume that all nodes of  $\mathcal{G}$  are self adjacent, that is  $\pmb{A}[i,i] = 1,i = 1,2,\dots,N$ . Let  $\pmb{D} = \text{diag}(d_1,d_2,\dots,d_N)$  be the degree matrix of  $\pmb{A}$ , where  $d_{i} = \sum_{j = 1}^{N}\pmb{A}[i,j]$ .

Neighborhood Normalization. There are two ways to normalize  $A$ . One approach is the following mean-pooling employed by Hamilton et al. (2017) and Velicković et al. (2019) for inductive learning:

$$
\boldsymbol {A} _ {m p} = \boldsymbol {D} ^ {- 1} \boldsymbol {A}.
$$

Another approach is the following symmetric normalization employed by Kipf & Welling (2017):

$$
\boldsymbol {A} _ {s y m} = \boldsymbol {D} ^ {- \frac {1}{2}} \boldsymbol {A} \boldsymbol {D} ^ {- \frac {1}{2}}.
$$

Iterated Function System. An iterated function system (Hutchinson, 1981) is defined by

$$
\operatorname {I F S} = \left\{\mathbb {R} ^ {N}; f _ {1}, f _ {2}, \dots , f _ {n}; \boldsymbol {p} \right\},
$$

where each  $f_{i}$  is a contractive mapping on  $\mathbb{R}^N$ , and  $\pmb{p} = (p_1, p_2, \dots, p_n)$  is an adjoint probability vector meaning that  $f_{i}$  is selected by probability  $p_i$  for each iteration. Hutchinson (1981) showed that there exists a unique nonempty compact set  $\mathbb{F}$  such that  $\mathbb{F} = \bigcup_{i=1}^{N} f_i(\mathbb{F})$ . We call  $\mathbb{F}$  the fractal set or invariant set of IFS. More conclusions on IFS can be found in the Appendix B. It is well known that there exists a unique probability measure  $\mu$  satisfying the equation

$$
\mu = \sum_ {j = 1} ^ {N} p _ {j} \mu \circ f _ {j} ^ {- 1}. \tag {1}
$$

The probability measure  $\mu$  in (1) is called the self-similar measure of IFS with probability vector  $\pmb{p}$ .

![](images/0089099858dd65f52430bd789b87371e3ce24fd0000cc19e46eeca9d4d4378b2.jpg)  
Figure 2: An overview of IGNNS. The upper part of the Figure describes how to generate two affine transformations on  $\mathbb{R}^4$ , where we use the mean-pooling method to normalize  $A$ ,  $p_0 = 0.6$ ,  $p_1 = 0.4$

$$
\boldsymbol {A} = \left( \begin{array}{c c c c} 1 & 1 & 0 & 0 \\ 1 & 1 & 0 & 1 \\ 0 & 1 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{array} \right),   \boldsymbol {A} _ {0} = \left( \begin{array}{c c c c} \frac {1}{2} & \frac {1}{2} & 0 & 0 \\ 0 & \frac {1}{2} & 0 & \frac {1}{2} \\ 0 & 0 & \frac {1}{2} & \frac {1}{2} \\ 0 & 0 & 0 & 1 \end{array} \right) \text {a n d}   \boldsymbol {A} _ {1} = \left( \begin{array}{c c c c} 1 & 0 & 0 & 0 \\ \frac {1}{2} & \frac {1}{2} & 0 & 0 \\ 0 & \frac {1}{2} & \frac {1}{2} & 0 \\ 0 & \frac {1}{3} & \frac {1}{3} & \frac {1}{3} \end{array} \right).
$$

# 3 IGNNS ARCHITECTURE

In this section, we will introduce the architecture of the IGNNS according to the input layer, IFS layer, representation layer and output layer, which is described in Figure 2.

# 3.1 INPUT LAYER

Given a graph structure data  $\mathbf{X} \in \mathbb{R}^{N \times F}$  of  $\mathcal{G} = (V, E)$ , called as the feature matrix of node set  $V$ . A row of  $\mathbf{X}$  represents the  $F$ -dimensional feature vector of a node in  $V$ . Let  $W^{\mathrm{int}} \in \mathbb{R}^{F \times H}$  be a learnable parameter matrix, where  $H$  is the dimension of the latent space. Then  $XW^{\mathrm{int}} \in \mathbb{R}^{N \times H}$ . The output of the input layer is defined by

$$
\boldsymbol {X} ^ {\text {i n t}} = \sigma (\boldsymbol {X} \boldsymbol {W} ^ {\text {i n t}}) \in \mathbb {R} ^ {N \times H}, \tag {2}
$$

where  $\sigma(\cdot)$  is the activation function. Generally,  $\mathrm{ReLU}(x) = \max(0, x)$  is used as the nonlinear activation function. Here, each column of  $X^{\mathrm{int}}$  is regarded as a point in  $\mathbb{R}^N$ , so  $X^{\mathrm{int}}$  is the set of  $H$  points in  $\mathbb{R}^N$  and arranged in a certain order. The vector composed of the  $i$ th component of these points (the  $i$ th row of  $X^{\mathrm{int}}$ ) is a feature representation of the  $i$ th node of graph  $\mathcal{G}$ .

# 3.2 IFS LAYER

Let  $\mathbf{A}$  be the adjacency matrix of  $\mathcal{G}$ . Let  $triu(\mathbf{A})$  denote the upper triangular matrix of  $\mathbf{A}$  and  $tril(\mathbf{A})$  denote the lower triangular matrix of  $\mathbf{A}$ . The symmetric normalization of  $triu(\mathbf{A})$  and  $tril(\mathbf{A})$  are

$$
\boldsymbol {A} _ {0} = \boldsymbol {D} _ {0} ^ {- \frac {1}{2}} t r i u (\boldsymbol {A}) \boldsymbol {D} _ {0} ^ {- \frac {1}{2}} \text {a n d} \boldsymbol {A} _ {1} = \boldsymbol {D} _ {1} ^ {- \frac {1}{2}} t r i l (\boldsymbol {A}) \boldsymbol {D} _ {1} ^ {- \frac {1}{2}},
$$

where  $D_0$  and  $D_{1}$  are degree matrices of triu(A) and tril(A) respectively. Sometimes, we use the mean-pooling of triu(A) and tril(A), i.e.  $\pmb{A}_{0} = D_{0}^{-1}\text{triu} (\pmb {A}),\pmb{A}_{1} = D_{1}^{-1}\text{tril} (\pmb {A})$ . Let  $f_{0},f_{1}$  be the two affine transformations on  $\mathbb{R}^N$ , induced by  $\pmb{A}_0,\pmb{A}_1$  respectively, defined as follows:

$$
f _ {0}: x \to \boldsymbol {A} _ {0} x + b _ {0}, x \in \mathbb {R} ^ {N}, b _ {0} \in \mathbb {R}, f _ {1}: x \to \boldsymbol {A} _ {1} x + b _ {1}, x \in \mathbb {R} ^ {N}, b _ {1} \in \mathbb {R},
$$

where  $b_{0}$  and  $b_{1}$  are learnable biases, namely add constants  $b_{0}$  and  $b_{1}$  to each component of  $A_{0}x$  and  $A_{1}x$  respectively. Constructing iterated function system

$$
\operatorname {I F S} = \left\{\mathbb {R} ^ {N}; f _ {0}, f _ {1}; \boldsymbol {p} \right\},
$$

where  $\pmb{p} = (p_0, p_1)$  is a learnable adjoint probability vector, satisfying  $p_0 > 0$ ,  $p_1 > 0$  and  $p_0 + p_1 = 1$ . Using symbol space  $\Omega_n = \{0, 1\}^n$ , then for each  $\mathbf{i} = (i_1, i_2, \dots, i_n) \in \Omega_n$  the length of  $\mathbf{i}$  is  $n$ , denoted as  $|\mathbf{i}| = n$ , and defining  $\pmb{p_i} = p_{i_1}p_{i_2}\cdots p_{i_n}$  and  $f_i = f_{i_1} \circ f_{i_2} \circ \cdots \circ f_{i_n}$ . The iterative process of IFS is described as follows:

The first iteration  $(|\mathbf{i}| = 1)$ . The result of the first iteration is denoted by

$$
\mathbb {H} ^ {(1)} = \left\{f _ {0} \left(\boldsymbol {X} ^ {\text {i n t}}\right), f _ {1} \left(\boldsymbol {X} ^ {\text {i n t}}\right) \right\} = \left\{\boldsymbol {H} _ {\mathbf {i}} \right\} _ {| \mathbf {i} | = 1},
$$

where  $H_{\mathbf{i}} = f_{\mathbf{i}}(X^{\mathrm{int}})$ ,  $\forall \mathbf{i} \in \Omega_1$ . Since IFS selects the iteration branch  $f_{i}$  with probability  $p_i$ , the mathematical expectation of  $\mathbb{H}^{(1)}$  is computed by

$$
\boldsymbol {E} _ {1} = p _ {0} f _ {0} \left(\boldsymbol {X} ^ {\text {i n t}}\right) + p _ {1} f _ {1} \left(\boldsymbol {X} ^ {\text {i n t}}\right) = p _ {0} \boldsymbol {H} _ {0} + p _ {1} \boldsymbol {H} _ {1} = \sum_ {| \mathbf {i} | = 1} \boldsymbol {p} _ {\mathbf {i}} \boldsymbol {H} _ {\mathbf {i}}.
$$

If choose to use bias in iterations, then  $H_0 = A_0 X^{\mathrm{int}} + b_0$ ,  $H_1 = A_1 X^{\mathrm{int}} + b_1$ , where  $b_0$  and  $b_1$  are learnable  $H$ -dimensional vectors.

The second iteration  $(|\mathbf{i}| = 2)$ . Using the results of the first iteration as the input of the second iteration, then the result of the second iteration is denoted by

$$
\begin{array}{l} \mathbb {H} ^ {(2)} = \left\{f _ {0} \left(f _ {0} \left(\boldsymbol {X} ^ {\text {i n t}}\right)\right), f _ {0} \left(f _ {1} \left(\boldsymbol {X} ^ {\text {i n t}}\right)\right), f _ {1} \left(f _ {0} \left(\boldsymbol {X} ^ {\text {i n t}}\right)\right), f _ {1} \left(f _ {1} \left(\boldsymbol {X} ^ {\text {i n t}}\right)\right) \right\} \\ = \left\{f _ {0 0} \left(\boldsymbol {X} ^ {\text {i n t}}\right), f _ {0 1} \left(\boldsymbol {X} ^ {\text {i n t}}\right), f _ {1 0} \left(\boldsymbol {X} ^ {\text {i n t}}\right), f _ {1 1} \left(\boldsymbol {X} ^ {\text {i n t}}\right) \right\} = \left. \boldsymbol {H} _ {\mathbf {i}} \right\rangle_ {| \mathbf {i} | = 2}, \\ \end{array}
$$

where  $H_{\mathbf{i}} = f_{\mathbf{i}}(X^{\mathrm{int}})$ ,  $\forall \mathbf{i} \in \Omega_2$ . Note that IFS selects the iteration path  $f_{\mathbf{i}}$  with probability  $p_{\mathbf{i}}$ , then the mathematical expectation of  $\mathbb{H}^{(2)}$  is computed by

$$
\boldsymbol {E} _ {2} = \sum_ {| \mathbf {i} | = 2} \boldsymbol {p} _ {\mathbf {i}} f _ {\mathbf {i}} (\boldsymbol {X} ^ {\text {i n t}}) = \sum_ {| \mathbf {i} | = 2} \boldsymbol {p} _ {\mathbf {i}} H _ {\mathbf {i}}.
$$

We expand the expression of  $E_{2}$  and perceive its feature representation ability. First,

$$
\boldsymbol {H} _ {0 0} = f _ {0} (\boldsymbol {H} _ {0}) = \boldsymbol {A} _ {0} (\boldsymbol {A} _ {0} \boldsymbol {X} ^ {\text {i n t}} + \boldsymbol {b} _ {0}) + \boldsymbol {b} _ {0}, \boldsymbol {H} _ {0 1} = f _ {0} (\boldsymbol {H} _ {1}) = \boldsymbol {A} _ {0} (\boldsymbol {A} _ {1} \boldsymbol {X} ^ {\text {i n t}} + \boldsymbol {b} _ {1}) + \boldsymbol {b} _ {0},
$$

$$
\boldsymbol {H} _ {1 0} = f _ {1} (\boldsymbol {H} _ {0}) = \boldsymbol {A} _ {1} (\boldsymbol {A} _ {0} \boldsymbol {X} ^ {\text {i n t}} + \boldsymbol {b} _ {0}) + \boldsymbol {b} _ {1}, \boldsymbol {H} _ {1 1} = f _ {1} (\boldsymbol {H} _ {1}) = \boldsymbol {A} _ {1} (\boldsymbol {A} _ {1} \boldsymbol {X} ^ {\text {i n t}} + \boldsymbol {b} _ {1}) + \boldsymbol {b} _ {1}.
$$

Then

$$
\begin{array}{l} \boldsymbol {E} _ {2} = p _ {0 0} \boldsymbol {H} _ {0 0} + p _ {0 1} \boldsymbol {H} _ {0 1} + p _ {1 0} \boldsymbol {H} _ {1 0} + p _ {1 1} \boldsymbol {H} _ {1 1} \\ = \left(p _ {0 0} \boldsymbol {A} _ {0 0} + p _ {0 1} \boldsymbol {A} _ {0 1} + p _ {1 0} \boldsymbol {A} _ {1 0} + p _ {1 1} \boldsymbol {A} _ {1 1}\right) \boldsymbol {X} ^ {\text {i n t}} \\ + \left(p _ {0 0} \boldsymbol {A} _ {0} \boldsymbol {b} _ {0} + p _ {0 1} \boldsymbol {A} _ {0} \boldsymbol {b} _ {1} + p _ {1 0} \boldsymbol {A} _ {1} \boldsymbol {b} _ {0} + p _ {1 1} \boldsymbol {A} _ {1} \boldsymbol {b} _ {1}\right) + \left(p _ {0 0} \boldsymbol {b} _ {0} + p _ {0 1} \boldsymbol {b} _ {0} + p _ {1 0} \boldsymbol {b} _ {1} + p _ {1 1} \boldsymbol {b} _ {1}\right), \\ \end{array}
$$

where  $A_{\mathbf{i}} = A_{i_1}A_{i_2},\forall \mathbf{i} = (i_1,i_2)\in \Omega_2$

The  $n$ -th iteration  $(|\mathbf{i}| = n)$ . Inductively, we have

$$
\mathbb {H} ^ {(n)} = \left\{\boldsymbol {H} _ {\mathrm {i}} \right\} _ {| \mathrm {i} | = n}, \quad \boldsymbol {H} _ {\mathrm {i}} = f _ {\mathrm {i}} \left(\boldsymbol {X} ^ {\text {i n t}}\right), \quad \boldsymbol {E} _ {n} = \sum_ {| \mathrm {i} | = n} \boldsymbol {p} _ {\mathrm {i}} \boldsymbol {H} _ {\mathrm {i}}. \tag {3}
$$

Note that  $\pmb{H}_{\mathbf{i}} \in \mathbb{R}^{N \times H}$  and each column of  $\pmb{H}_{\mathbf{i}}$  is a point in  $\mathbb{R}^N$ , so we regard it as a subset of  $\mathbb{R}^N$  with  $H$  elements. Thus  $\mathbb{H}^{(n)}$  is a subset of  $\mathbb{R}^N$  with  $H \times 2^n$  elements (including duplicate elements). Because of Theorem 4.1, we call  $\mathbb{H}^{(n)}$  the fractal representation with depth  $n$  of nodes.

# 3.3 REPRESENTATION LAYER

After  $n$  iterations of IFS layer, the dynamic trajectory of IFS is obtained:

$$
\mathcal {O} = \left\{\boldsymbol {E} _ {1}, \boldsymbol {E} _ {2},..., \boldsymbol {E} _ {n} \right\}.
$$

In general, the global representation  $\pmb{R}$  of nodes is obtained by time mean or concatenation operations on  $\mathcal{O}$ .

$$
\boldsymbol {R} = \frac {1}{n} \sum_ {i = 1} ^ {n} \boldsymbol {E} _ {i} \in \mathbb {R} ^ {N \times H} \quad \text {o r} \quad \boldsymbol {R} = \| _ {i = 1} ^ {n} \boldsymbol {E} _ {i} \in \mathbb {R} ^ {N \times n H},
$$

where  $\parallel$  is the concatenation operator. Because of Theorem 4.2, we call  $R$  the ergodic representation of nodes. In practice, we adopt weighted time mean or weighted concatenation. According to the Theorem A.1, we use heuristic weights (Here, we understand it as the average expansion factor of the distance between two points after affine transformation). Let  $r = \sqrt{\ln(N) + \gamma}$ , where  $\gamma \approx 0.577215664$  is the Euler constant. Suppose  $r = (r_1, r_2, \dots, r_n)$  is a learnable n-dimensional vector with initial value  $r_i = \left(\frac{1}{r}\right)^{i-1}$ . Then the ergodic representation of nodes is

$$
\boldsymbol {R} = \sum_ {i = 1} ^ {n} r _ {i} \boldsymbol {E} _ {i} \in \mathbb {R} ^ {N \times H} \quad \text {o r} \quad \boldsymbol {R} = \| _ {i = 1} ^ {n} r _ {i} \boldsymbol {E} _ {i} \in \mathbb {R} ^ {N \times n H}.
$$

# 3.4 OUTPUT LAYER

Let  $\mathbf{W}^{\mathrm{out}} \in \mathbb{R}^{H \times P}$  be a learnable parameter matrix, where  $P$  is the dimension of the output layer (such as the number of class lebels). If  $\mathbf{R}$  is generated by  $\mathcal{O}$  concatenation, then let  $\mathbf{W}^{\mathrm{out}} \in \mathbb{R}^{nH \times P}$ . There are two ways to construct output layer, one is to use a multi-layer perception as output, that is  $\mathbf{O} = \mathrm{MLP}(\mathbf{R}, \mathbf{W}^{\mathrm{out}})$ ; the other is to use  $f_{0}, f_{1}$  for mixed propagation, that is, let  $\mathbf{R}_{0} = f_{0}(\mathbf{R}\mathbf{W}^{\mathrm{out}})$  and  $\mathbf{R}_{1} = f_{1}(\mathbf{R}\mathbf{W}^{\mathrm{out}})$ , where the biases of  $f_{0}, f_{1}$  are removed, then the output

$$
\boldsymbol {O} = p _ {0} \boldsymbol {R} _ {0} + p _ {1} \boldsymbol {R} _ {1} + \boldsymbol {b} _ {\text {o u t}},
$$

where the bias  $\pmb{b}_{\mathrm{out}} \in \mathbb{R}^{P}$  is an optional learnable parameter vector.

# 3.5 INITIALIZATION OF LEARNABLE VARIABLES

The learnable parameters of IGNNS include input layer matrix  $\mathbf{W}^{\mathrm{int}} \in \mathbb{R}^{F \times H}$ , adjoint probability vector  $\pmb{p} = (p_0, p_1) \in \mathbb{R}^2$  of IFS, biases  $\pmb{b}_0, \pmb{b}_1 \in \mathbb{R}^H$  of IFS layer, weight coefficient  $r = (r_1, r_2, \dots, r_n)$  of representation layer, matrix  $\mathbf{W}^{\mathrm{out}} \in \mathbb{R}^{H \times P}$  of output layer and bias  $\pmb{b}_{\mathrm{out}} \in \mathbb{R}^P$  of output layer. Among them,  $\mathbf{W}^{\mathrm{int}}$  and  $\mathbf{W}^{\mathrm{out}}$  are the required learnable parameters, using the initialization described in Glorot & Bengio (2010);  $\pmb{b}_0, \pmb{b}_1$  and  $\pmb{b}_{\mathrm{out}}$  are optional learnable parameters with an initial value 0;  $\pmb{p}$  is a optional learnable parameter, for undirected graph, setting  $p_0 \in [0.5 - 0.05, 0.5 + 0.05]$ , and for directed graph, setting (For the reasons, see Appendix E)

$$
p _ {0} = \frac {\det  D _ {1}}{\det  D _ {0} + \det  D _ {1}}, p _ {1} = \frac {\det  D _ {0}}{\det  D _ {0} + \det  D _ {1}};
$$

$r$  is a optional learnable parameter, and its initial value as defined in 3.3. Thus, IGNNS is denoted as

$$
\boldsymbol {O} = \operatorname {I G N N S} (\boldsymbol {X}, \boldsymbol {A}; \boldsymbol {W} ^ {\text {i n t}}, \boldsymbol {p}, \boldsymbol {b} _ {0}, \boldsymbol {b} _ {1}, \boldsymbol {r}, \boldsymbol {W} ^ {\text {o u t}}, \boldsymbol {b} _ {\text {o u t}}) \quad \text {o r s i m p l y} \quad \boldsymbol {O} = \operatorname {I G N N S} (\boldsymbol {X}, \text {I F S}),
$$

where IFS is induced by  $\mathbf{A}$ . The output of IGNNS can be used as the input of downstream tasks, and can also be connected to other network architectures.

# 4 GEOMETRIC PROPERTIES OF IGGNS

The discussion here assumes that affine  $f_0, f_1$  are contractive, that is, there exists a constant  $0 < c < 1$  such that  $\| f_0(x_1) - f_0(x_2)\|_2 < c\| x_1 - x_2\|_2$  and  $\| f_1(x_1) - f_1(x_2)\|_2 < c\| x_1 - x_2\|_2$ . Otherwise, let  $f_0: x \to \frac{1}{\|\mathbf{A}_0\|_{F+1}}\mathbf{A}_0x + b_0$  and  $f_1: x \to \frac{1}{\|\mathbf{A}_1\|_{F+1}}\mathbf{A}_1x + b_1$ . In practice, IGNNS does not use contractive affine. If contractive affine is used in IGNNS, it can be seen from the following theorems that the characterization ability of IGNNS decreases with the increase of IFS iterations, which is similar to the performance of Graph Convolution Network (GCN). Such a phenomenon is called over-smoothing (Li et al., 2018b; Xu et al., 2019; Chen et al., 2020), which suggests that as the number of layers increases, the representations of the nodes in GCN are inclined to converge to a certain value and thus become indistinguishable.

Theorem 4.1 (Fractal generation) Let  $\mathbb{H}^{(n)} = \{\pmb{H}_i\}_{|i| = n}$ , which is a subset of  $\mathbb{R}^N$  with  $H\times 2^n$  elements (including duplicate elements), then

$$
d _ {H} (\mathbb {H} ^ {(n)}, \mathbb {F}) \to 0, n \to \infty ,
$$

where  $d_H$  is the Hausdorff distance defined on  $\mathcal{H}(\mathbb{R}^N)$ , the set of all nonempty compact subsets of  $\mathbb{R}^N$ , and  $\mathbb{F}$  is the fractal set of IFS in IGNNS. In other words, as the number of iterations increases,  $\mathbb{H}^{(n)}$  will be independent of node feature  $\pmb{X}$ , only related to the graph structure described by  $\pmb{A}$ .

Let  $T$  be the Hutchinson operator on  $\mathcal{H}(\mathbb{R}^N)$ , defined by  $T(B) = f_0(B) \bigcup f_1(B), \forall B \in \mathcal{H}(\mathbb{R}^N)$ . Then the updated rule of  $\mathbb{H}^{(n)}$  satisfying

$$
\mathbb {H} ^ {(n)} = T (\mathbb {H} ^ {(n - 1)}) = \dots = T ^ {n} (\mathbb {H} ^ {(0)}),
$$

where  $\mathbb{H}^{(0)} = \{\pmb{X}^{\mathrm{int}}\}$  is a subset of  $\mathbb{R}^N$  with  $H$  elements. In fractal geometry,  $\mathbb{H}^{(n)}$  is used to draw the fractal image on the plane. First, taking initial value  $\mathbb{H}^{(0)} = \{x_0\}$ , where  $x_0$  is a point in plane. For enough  $n$ , printing all the points of  $\mathbb{H}^{(n)}$  on the screen to obtain the approximate fractal image.

Theorem 4.2 (Ergodic property) Let  $\pmb{E}_n = \sum_{|\pmb{i}|=n} \pmb{p}_i \pmb{H}_i$  be the mathematical expectation of  $\mathbb{H}^{(n)} = \{\pmb{H}_i\}_{|i|=n}$ , then  $\pmb{E}_n$  converges to a constant matrix  $\pmb{E}$  in Frobenius norm, i.e.

$$
\lim _ {n \to \infty} \boldsymbol {E} _ {n} = \boldsymbol {E} := \left( \begin{array}{c c c c} e _ {1} & e _ {1} & \dots & e _ {1} \\ e _ {2} & e _ {2} & \dots & e _ {2} \\ \vdots & \vdots & \ldots & \vdots \\ e _ {N} & e _ {N} & \dots & e _ {N} \end{array} \right) _ {N \times H},
$$

where  $e_i \in \mathbb{R}$  is a constant. Further more, the time mean of the dynamic trajectory  $\mathcal{O}$  of IFS satisfies

$$
\lim  _ {n \rightarrow \infty} \frac {1}{n} \sum_ {i = 1} ^ {n} \boldsymbol {E} _ {i} = \lim  _ {n \rightarrow \infty} \boldsymbol {E} _ {n} = \boldsymbol {E},
$$

and the series  $\sum_{i=1}^{\infty} r_i E_i$  converges in Frobenius norm.

Theorem 4.2 shows that as long as the number of iterations is large enough, the embeddings of nodes will be close to linear correlation, and the representation ability of IGNNS will decline. However, in the framework of IGNNS, because the spectral radius  $\rho(A_0) = \rho(A_1) = 1$ , IFS is not contractive in general, and IGNNS still has the ability of depth feature representation.

# 5 GENERAL FRAMEWORK

The IGNNS described above is only suitable for graphs with only one relationship (undirected graph or directed graph). We only need to extend the function set of IFS a little, and IGNNS can be applied to multi-relational graph (undirected graph or directed graph) in a natural way. A multi-relational graph is defined as  $\mathcal{G} = (V,E,R)$ , where the edges are defined as tuples  $e = (u,r,v) \in E$  indicating the presence of a particular relation  $r \in R$  holding between two nodes  $u,v \in V$ . For any relation  $r \in R$ , there exists a corresponding relational matrix  $\mathbf{A}_r$ . Using the same decomposition techniques described above,  $\mathbf{A}_r$  is decomposed into a standardized upper triangle matrix  $\mathbf{A}_{r0}$  and a lower triangle matrix  $\mathbf{A}_{r1}$ , respectively. Define affine set as

$$
\mathcal {F} = \{f _ {r 0}: x \to \boldsymbol {A} _ {r 0} x + b _ {r 0}, x \in \mathbb {R} ^ {N}, b _ {r 0} \in \mathbb {R}, f _ {r 1}: x \to \boldsymbol {A} _ {r 1} x + b _ {r 1}, x \in \mathbb {R} ^ {N}, b _ {r 1} \in \mathbb {R} | r \in R \}.
$$

The adjoint probability set corresponding to the affine set is

$$
\mathcal {P} = \{p _ {r 0}, p _ {r 1} | p _ {r 0} > 0, p _ {r 1} > 0, r \in R \},
$$

where  $\sum_{r\in R}(p_{r0} + p_{r1}) = 1$ . Then IFS =  $\{\mathbb{R}^N\mathcal{F};\mathcal{P}\}$ . Its iterative calculation method is the same as that of normal IFS. Thus the Multi-relational Iterative Graph Neural Network System is denoted as

$$
\boldsymbol {O} = \operatorname {R - I G N N S} (\boldsymbol {X}, \operatorname {I F S}).
$$

Table 1: Summary statistics of the benchmark datasets used in the experiment.  

<table><tr><td>Data</td><td>Nodes</td><td>Edges</td><td>Features</td><td>Classes</td><td>Training</td><td>Validation</td><td>Testing</td></tr><tr><td>Cora</td><td>2708</td><td>5429</td><td>1433</td><td>7</td><td>140</td><td>500</td><td>1000</td></tr><tr><td>Citeseer</td><td>3327</td><td>4732</td><td>3703</td><td>6</td><td>120</td><td>500</td><td>1000</td></tr><tr><td>Pubmed</td><td>19717</td><td>44338</td><td>500</td><td>3</td><td>60</td><td>500</td><td>1000</td></tr></table>

Table 2: hyper-parameters in experiment.  

<table><tr><td>Setting</td><td>Cora</td><td>Citeseer</td><td>Pubmed</td></tr><tr><td>Neighborhood Normalization</td><td>symmetric</td><td>mean-pooling</td><td>symmetric</td></tr><tr><td>Learning rate</td><td>0.005</td><td>0.002</td><td>0.01</td></tr><tr><td>Initial value of p0</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Dropout</td><td>0.9</td><td>0.9</td><td>0.8</td></tr><tr><td>Weight decay</td><td>5e-3</td><td>5e-3</td><td>5e-3</td></tr><tr><td>Epochs</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Hidden dimensions</td><td>48</td><td>72</td><td>72</td></tr><tr><td>Number of iterations in IFS layer</td><td>5</td><td>4</td><td>4</td></tr><tr><td>Learnable adjoint probability vector</td><td>False</td><td>True</td><td>False</td></tr><tr><td>Use bias for IFS layer</td><td>True</td><td>False</td><td>False</td></tr><tr><td>Use bias for output layer</td><td>False</td><td>False</td><td>False</td></tr></table>

# 6 EXPERIMENTS

# 6.1 EXPERIMENTAL TASK: SEMI-SUPERVISED NODE CLASSIFICATION

Let  $Z = \mathrm{softmax}(O)$ , where  $Z \in \mathbb{R}^N$  and  $\mathrm{softmax}(\cdot)$  is the softmax activation function, defined as  $\mathrm{softmax}(x_i) = \frac{1}{\mathcal{Z}} \exp(x_i)$  with  $\mathcal{Z} = \sum_{i} \exp(x_i)$ , is applied row-wise. For semi-supervised multiclass classification, we employ the following cross-entropy to evaluate error over all labeled examples:  $L = -\sum_{l \in \mathbb{Y}_L} \sum_{i=1}^{P} Y[l, i] \ln Z[l, i]$ , where  $\mathbb{Y}_L$  is the set of node indices that have labels with  $P$  classes,  $Y[l, :]$  is a one-hot vector of size  $P$  representing the class of node  $l$  and  $Z[l, :]$  is the row  $l$  of the matrix  $Z$ .

# 6.2 EXPERIMENTAL SETUP

Dataset. In our experiment, we use three benchmark datasets for evaluation, including Cora, Cite-seer, Pubmed and apply the standard fixed training/validation/testing split (Yang et al., 2016; Kipf & Welling, 2017; Velicković et al., 2018) on above datasets, with 20 nodes per class for training, 500 nodes for validation and 1,000 nodes for testing. See Table 1 for more details.

Parameter Setting. Random seed for Tensorflow and Numpy is set to 1234. ReLU (Nair & Hinton, 2010) is used as the activation function in input layer and output layer. Dropout (Srivastava et al., 2014) is applied to input layer, IFS layer and output layer. In representation layer, we adopt weighted time mean to get the ergodic representation of nodes. In output layer, we adopt the method of mixed propagation to get the output of IGNNS. We use the AdamOptimizer (P.Kingma & Ba, 2015) during training. More details of hyper-parameters are shown in Table 2. During training stage, we select the best model to maximize the accuracy of the validation set and use early stopping with a patience of 100 epochs.

# 6.3 EXPERIMENTAL RESULT

We compare with those models that strictly follow the standard of experiment setup of semi-supervised node classification, i.e. the standard fixed training/validation/testing split (Yang et al., 2016; Kipf & Welling, 2017) is applied on dataset. For baselines, we include recent deep GN-

Table 3: Summary of classification accuracy (%) results on Cora, Citeseer and Pubmed. The results are taken from the corresponding papers.  

<table><tr><td>Method</td><td>Cora</td><td>Citeseer</td><td>Pubmed</td></tr><tr><td>Planetoid (Yang et al., 2016)</td><td>75.7</td><td>64.7</td><td>77.2</td></tr><tr><td>GCN (Kipf &amp; Welling, 2017)</td><td>81.5</td><td>70.3</td><td>79.0</td></tr><tr><td>GAT (Veličković et al., 2018)</td><td>83.0</td><td>72.5</td><td>79.0</td></tr><tr><td>TAGCN (Du et al., 2017)</td><td>83.3</td><td>71.4</td><td>81.1</td></tr><tr><td>JKNet (Xu et al., 2018)</td><td>81.1</td><td>69.8</td><td>78.1</td></tr><tr><td>AGNN (Thekumparampil et al., 2018)</td><td>83.1</td><td>71.7</td><td>79.9</td></tr><tr><td>N-GCN (Xhonneux et al., 2018)</td><td>83.0</td><td>72.2</td><td>79.5</td></tr><tr><td>DGCN (Zhuang &amp; Ma, 2018)</td><td>83.5</td><td>72.6</td><td>80.0</td></tr><tr><td>APPNP (Klicpera et al., 2019a)</td><td>83.3</td><td>71.8</td><td>81.1</td></tr><tr><td>H-GAT (Gulcehre et al., 2019)</td><td>83.5</td><td>72.9</td><td>-</td></tr><tr><td>IGNNS (ours)</td><td>86.3</td><td>75.1</td><td>80.5</td></tr></table>

N models such as JKNet (Xu et al., 2018) and APPNP (Klicpera et al., 2019a), Attention-based models such as GAT (Velicković et al., 2018), AGNN (Thekumparampil et al., 2018) and H-GAT (Gulcehre et al., 2019), and other models such as TAGCN (Du et al., 2017) and N-GCN (Xhonneux et al., 2018). We also include three state-of-the-art shallow GNN models: Planetoid (Yang et al., 2016), GCN (Kipf & Welling, 2017) and DGCN (Zhuang & Ma, 2018). The detailed results are shown in Table 3.

# 7 CONCLUSION

In this paper, we propose a new framework of graph neural networks, Iterated Graph Neural Network System (IGNNS), which give a connection between GNN and IFS. We use IFS to simulate the bidirectional message passing process of GNN, and obtain the fractal representation and ergodic representation of graph nodes, which are very helpful for downstream tasks. The experiments show that we have achieved good results in semi-supervised node classification task. Interesting directions for future work include pruning the iterative path space  $\{0,1\}^n$  to reduce the computational complexity, coding graph structured data with IFS, and establishing more interesting connections between IFS and GNN.

# REFERENCES

M.F.Barnsley.Fractals Everywhere.Academic Press,1988.  
Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In International Conference on Machine Learning, 2020.  
Jian Du, Shanghang Zhang, Guanhang Wu, José MF Moura, and Soummya Kar. Topology adaptive graph convolutional networks. 2017. URL arXivpreprintarXiv:1710.10370.  
J.H. Elton. An ergodic theorem for iterated maps. Journal of Ergodic theory and Dynamical Systems, 7:481-488, 1987.  
K. Falconer. Fractal Geometry: Mathematical Foundations and Applications. Wiley, 1990.  
Tsu-Jui Fu, Peng-Hsuan Li, and Wei-Yun Ma. GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction. Association for Computational Linguistics, July 2019. URL https://www.aclweb.org/anthology/P19-1136.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, volume 9, pp. 249-256, 2010.

Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, and Nando de Freitas. Hyperbolic attention networks. In ICLR, 2019.  
William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NIPS, 2017.  
J. Hutchinson. Fractals and self-similarity. Indiana University Journal of Mathematics, 30:713-747, 1981.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.  
J. Klicpera, A. Bojchevski, and S. Gunnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In ICLR, 2019a.  
Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In AAAI, 2018b.  
Diego Marcheggiani and Ivan Titov. Encoding sentences with graph convolutional networks for semantic role labeling. In EMNLP, 2017.  
P.R. Massopust. Fractal Functions, Fractal Surfaces, and Wavelets. Academic Press, 2017.  
Zhewei Wei Ming Chen, Bolin Ding Zengfeng Huang, and Yaliang Li. Simple and deep graph convolutional networks. 2020.  
V. Nair and G. E Hinton. Rectified linear units improve restricted boltzmann machines. In ICML, 2010.  
Diederik P.Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
F. Scarselli, M. Gori, A.C. Tsoi, M. Hagenbuchner, and G. Monfardini. The graph neural network model. IEEE Trans. Neural Netw. Learn. Syst, 20(1):61-80, 2009.  
Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. arXiv preprint arXiv:1703.06103, 2017.  
N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1): 1929-1958, 2014.  
Kiran K Thekumparamil, Chong Wang, Sewoong Oh, and Li-Jia Li. Attention-based graph neural network for semi-supervised learning. 2018. URL arXivpreprintarXiv:1803.03735.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph Attention Networks. International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJXMpikCZ. accepted as poster.  
Petar Velicković, William Fedus, William L. Hamilton, Pietro Lio, Yoshua Bengio, and R Devon Hjelm. Deep Graph Infomax. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rklz9iAcKQ.  
Louis-Pascal A. C. Xhonneux, Meng Qu, and Jian Tang. N-gcn: Multi-scale graph convolution for semi-supervised node classification. 2018. URL arXivpreprintarXiv:1802.08888.  
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In ICML, 2018.  
Keyulu Xu, Weihua Hu, and Jure Leskovec. How powerful are graph neural networks. In ICLR, 2019.

Zhilin Yang, William W. Cohen, and Ruslan Salakhutdinov. Revisiting semi-supervised learning with graph embeddings. In International Conference on Machine Learning, 2016.

Chenyi Zhuang and Qiang Ma. Dual graph convolutional networks for graph-based semi-supervised classification. In WWW, 2018.
