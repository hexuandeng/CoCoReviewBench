# SPSC: A FAST AND PROVABLE ALGORITHM FOR SAMPLING-BASED GNN TRAINING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Neighbor sampling is a commonly used technique for training Graph Neural Networks (GNNs) on large graphs. Previous work has shown that sampling-based GNN training can be considered as Stochastic Compositional Optimization (SCO) problems and can be better solved by SCO algorithms. However, we find that SCO algorithms are impractical for training GNNs on large graphs because they need to store the moving averages of the aggregated features of all nodes in the graph. The moving averages can easily exceed the GPU memory limit and even the CPU memory limit. In this work, we propose a variant of SCO algorithms with sparse moving averages for GNN training. By storing the moving averages in the most recent iterations, our algorithm only requires a fixed size buffer, regardless of the graph size. We show that our algorithm preserves the convergence rate of the original SCO algorithm when the buffer size satisfies certain conditions. Our experiments validate our theoretical results and show that our algorithm outperforms the traditional Adam SGD for GNN training with a small memory overhead.

# 1 INTRODUCTION

Graph Neural Networks (GNNs) have become the state-of-the-art models for machine learning tasks on graph-structured data. By recursively aggregating the features of neighboring nodes, GNNs learn an embedding of the nodes and use the embedding for downstream tasks such as node classification (Kipf & Welling, 2017; Duran & Niepert, 2017) or link prediction (Zhang & Chen, 2017; 2018).

Due to the recursive neighbor aggregation, training GNNs on large graphs is computationally challenging. To alleviate the computation burden, various neighbor sampling methods have been proposed (Hamilton et al., 2017; Ying et al., 2018; Chen et al., 2018; Zou et al., 2019; Li et al., 2018; Chiang et al., 2019; Zeng et al., 2020). The idea is to compute an unbiased estimation of the aggregation result in each layer based on a sampled subset of neighbors. These sampling techniques enable GNN training on large graphs. However, due to the composition of the aggregation functions in multiple layers, the stochastic gradient obtained with sampled neighbor aggregation is not an unbiased estimation of the true gradient, which undermines the convergence property of SGD-based training algorithms.

Previous work has shown that sampling-based GNN training is actually a Stochastic Compositional Optimization (SCO) problem (Cong et al., 2020; 2021). Cong et al. (2021) show that SCO algorithms can achieve faster convergence than the commonly used Adam SGD for GNN training on small graphs. Despite their good convergence property, SCO algorithms are not widely adopted for GNN training due to two reasons. First, although SCO algorithms achieve smaller training losses, the obtained GNN models usually have poor generalization – the validation and test accuracy are lower than the models trained by Adam SGD. Second, SCO algorithms need to maintain the moving averages of aggregation results of all nodes in the graph. For large graphs, the moving averages may exceed the memory capacity of the GPU. While it is possible to store the moving averages in CPU memory, copying the data from CPU to GPU in each iteration is expensive, which may negate the benefits of the faster convergence of SCO algorithms.

To address the above issues, we propose a Sparse Stochastic Compositional (SpSC) gradient method in this work. Our main idea is to store the moving averages for nodes sampled in the most recent

![](images/534e6b6b7631c982fb3da96113749e1978eac9340ee689e558579958eca4a8ee.jpg)  
(a) SCGD

![](images/dca8d279d8eb43039131e102f322e7984fa921f5b7e85f4c32b922449227d94d.jpg)  
Figure 1: Updating the moving average of  $\widetilde{Z}^{(l)}$ . SCGD needs to store the moving averages for all nodes in the graph. In SpSC, we only stores the data for nodes sampled in the past  $t$  iterations.  
(b) SpSC

iterations instead of all nodes. As only a small number of nodes are stored, our algorithm has small memory consumption even for large graphs. We provide a convergence analysis on SpSC and show that, when the number of stored iterations satisfies certain constraints, SpSC can preserve the asymptotic convergence rate of the original SCO algorithm. In practice, the sparse moving averaging slightly slows down the convergence of SCO algorithm, but it surprisingly overcomes its poor generalization problem and achieves higher accuracy for sampling-based GNN training. Compared with Adam SGD, our algorithm incurs a small overhead for updating the moving averages in each iteration, but the overhead can be easily justified by the faster convergence of our algorithm.

Our experiments with two GNN models on different input graphs validate our theoretical results and show that our algorithm achieves higher accuracy than Adam SGD with the same or less amount of training time.

# 2 BACKGROUND AND MOTIVATION

To facilitate our discussion, we first give background on sampling-based GNN training and its relation to stochastic compositional optimization.

# 2.1 GNN COMPUTATION

The computation at each layer of a GNN is conducted in two steps: aggregate and update. For each node  $v$  in the graph, the aggregate function gathers data from its neighboring nodes and returns the aggregation result as

$$
z _ {v} = \operatorname {A g g} _ {v} \left(h _ {n e [ v ]}, x _ {v}, x _ {n e [ v ]}\right). \tag {1}
$$

Here,  $h_{ne[v]}$  is the intermediate features of  $v$ 's neighbors from the previous layer,  $x_v$  is the input feature of  $v$ , and  $x_{ne[v]}$  is the input features of  $v$ 's neighbors. The update function uses the aggregated value to produce the intermediate features of  $v$  as

$$
h _ {v} = \operatorname {U p d} _ {v} \left(z _ {v}, x _ {v}\right). \tag {2}
$$

By stacking the intermediate features and the input features of all nodes, the computation at layer  $l$  can be written as

$$
Z ^ {(l)} = \operatorname {A g g} \left(H ^ {(l - 1)}, X\right), \quad H ^ {(l)} = \operatorname {U p d} \left(Z ^ {(l)}, X, W ^ {(l)}\right). \tag {3}
$$

Here,  $H^{(l-1)} = [h_1^{(l-1)}, \ldots, h_N^{(l)}]$  denotes the intermediate features of all nodes at layer  $l-1$ ,  $Z^{(l)} \in \mathbb{R}^{N \times d_l}$  is the aggregated features of all nodes at layer  $l$ ,  $X = [x_1, \ldots, x_N]$  is the input features of all nodes, and  $W^{(l)}$  is the learnable weights. As an example, Graph Convolutional Network (GCN) (Kipf & Welling, 2017) has  $\mathrm{Agg}(H^{(l-1)}, X) = PH^{(l-1)}$  where  $P$  is the normalized Laplacian matrix of the graph, and  $\mathrm{Upd}(Z^{(l)}, X, W^{(l)}) = \sigma(Z^{(l)}W^{(l)})$  where  $\sigma$  is a non-linear activation function. Many other GNNs can be expressed in this form with different definitions of Agg and Upd (Zhou et al., 2018).

# 2.2 SAMPLING-BASED GNN TRAINING AS STOCHASTIC COMPOSITIONAL OPTIMIZATION

When the graph is large, the neighbor aggregation operation  $\mathrm{Agg}$  incurs a large overhead, making the training of GNNs computationally challenging. Therefore, prior work has proposed to replace the  $\mathrm{Agg}$  function with a sampled neighbor aggregation operation  $\widetilde{\mathrm{Agg}}$ . By sampling the neighboring nodes, an unbiased estimate of  $Z^{(l)}$  is computed at each layer, i.e.,

$$
\widetilde {Z} ^ {(l)} = \widetilde {\operatorname {A g g}} (H ^ {(l - 1)}, X) \tag {4}
$$

with  $\mathbb{E}[\tilde{Z}^{(l)}] = Z^{(l)}$ . If we define the computation at layer  $l$  of the original GNN as a function

$$
\begin{array}{l} f ^ {(l)} \left(Z ^ {(l - 1)}, W ^ {(l - 1)}, \dots , W ^ {(T)}\right) = \left[ Z ^ {(l)}, W ^ {(l)}, \dots , W ^ {(T)} \right] \tag {5} \\ = \left[ \operatorname {A g g} \left(\operatorname {U p d} \left(Z ^ {(l - 1)}, X, W ^ {(l - 1)}\right)\right), W ^ {(l)}, \dots , W ^ {(T)} \right] ], \\ \end{array}
$$

the computation with sampled neighbor aggregation can be written as a stochastic function

$$
\begin{array}{l} f _ {\xi_ {l}} ^ {(l)} \left(\widetilde {Z} ^ {(l - 1)}, W ^ {(l - 1)}, \dots , W ^ {(T)}\right) = \left[ \widetilde {Z} ^ {(l)}, W ^ {(l)}, \dots , W ^ {(T)} \right] \tag {6} \\ = \left[ \widetilde {\mathrm {A g g}} \left(\operatorname {U p d} \left(\widetilde {Z} ^ {(l - 1)}, X, W ^ {(l - 1)}\right)\right), W ^ {(l)}, \dots , W ^ {(T)} \right] \\ \end{array}
$$

where  $\xi_{l}$  represents the sampled neighbors at layer  $l$ . Since  $\widetilde{Z}^{(l)}$  is an unbiased estimate of  $Z^{(l)}$ , we have  $\mathbb{E}[f_{\xi_l}^{(l)}] = f^{(l)}$ , and the computation of a  $T$ -layer GNN can be written as

$$
F (\theta) = \mathbb {E} _ {\xi_ {T + 1}} \left[ f _ {\xi_ {T + 1}} ^ {(T + 1)} \left(\mathbb {E} _ {\xi_ {T}} \left[ f _ {\xi_ {T}} ^ {(T)} \left(\dots E _ {\xi_ {1}} [ f ^ {(1)} (\theta) ] \dots\right) \right]\right) \right] \tag {7}
$$

where  $\theta = [X, W^{(1)}, \dots, W^{(T)}]$ ,  $f^{(T + 1)}$  is the loss function, and  $f_{\xi_{T + 1}}^{(T + 1)}$  corresponds to the estimated loss with mini-batch sampling. Note that we put all the learnable weights in  $\theta$  to formulate the computation as a stochastic compositional function. Our goal is to minimize  $F(\theta)$ , which is exactly a multi-level SCO problem.

# 2.3 LARGE MEMORY CONSUMPTION ISSUE WITH A NAIVE IMPLEMENTATION

SCO has been well studied in the past few years, and many algorithms with guaranteed convergence have been proposed (Zhang & Xiao, 2019; Yang et al., 2019; Chen et al., 2020; Yang et al., 2019; Balasubramanian et al., 2020; Chen et al., 2020; Lian et al., 2017; Wang et al., 2017b; Ghadimi et al., 2020). It seems straightforward to adopt these SCO algorithms to achieve faster training of GNNs. However, these algorithms have large memory consumption when applied to GNN training and cannot run on GPUs for large graphs.

To see the problem, let us consider the implementation of the SCGD algorithm (Yang et al., 2019) for GNN training. Formally, the algorithm is written as

$$
y _ {k + 1} ^ {(1)} = \left(1 - \beta_ {k}\right) y _ {k} ^ {(1)} + \beta_ {k} f _ {\xi_ {1, k}} ^ {(1)} \left(\theta_ {k}\right), \tag {8}
$$

$$
y _ {k + 1} ^ {(l)} = \left(1 - \beta_ {k}\right) y _ {k} ^ {(l)} + \beta_ {k} f _ {\xi_ {l, k}} ^ {(l)} \left(y _ {k + 1} ^ {(l - 1)}\right), \quad 2 \leq l \leq T, \tag {9}
$$

$$
\theta_ {k + 1} = \theta_ {k} - \alpha_ {k} \nabla f _ {\xi_ {1, k}} ^ {(1)} \left(\theta_ {k}\right) \nabla f _ {\xi_ {2, k}} ^ {(2)} \left(y _ {k} ^ {(1)}\right) \dots \nabla f _ {\xi_ {T + 1, k}} ^ {(T + 1)} \left(y _ {k} ^ {(T)}\right). \tag {10}
$$

The key idea is to store an auxiliary variable  $y^{(i)}$  to maintain the moving average of each composite function. Since  $f_{\xi_l}^{(l)}$  returns the exact values of  $W^{(l)}, \ldots, W^{(T)}$ , we only need to maintain a moving average of  $\widetilde{Z}^{(l)}$  for each layer. The computation is shown in Figure 1a. The moving average of the aggregated features is stored in  $Y^{(l)}$  with each row for one node. In each iteration, some nodes (rows) are sampled, and the estimated aggregation results  $\widetilde{Z}^{(l)}$  are merged into  $Y^{(l)}$  based on Formula (8) and (9). For the nodes that are not sampled, we simply multiply the corresponding rows of  $\bar{Z}^{(l)}$  by  $(1 - \beta_k)$ . Since the number of rows in  $Y^{(l)}$  is the number of nodes in the graph,  $Y^{(l)}$  takes a lot of memory when the graph is large. For example, for training a 3-layer GCN on a graph with two million nodes, suppose the hidden state dimension  $d_l = 512$  and a floating point has 4 bytes,  $Y$  takes  $3 \times 2\mathrm{M} \times 512 \times 4 = 12\mathrm{GB}$  of memory. All of the existing SCO algorithms need to maintain this moving average, which impedes their application to large-scale GNN training.

# 3 SPARSE STOCHASTIC COMPOSITIONAL GRADIENT DESCENT

To reduce the memory consumption of SCO algorithms for GNN training, we propose a Sparse Stochastic Compositional (SpSC) gradient method. Instead of storing the moving averages of all nodes in the graph, we only store the moving averages of nodes that are sampled in the most recent iterations.

As shown in Figure 1b, we maintain a fixed size buffer of the moving averages. The buffer is divided into  $t$  chunks with each chunk for the  $\widetilde{Z}^{(l)}$  of one iteration. The size of each chunk is  $m_l\cdot d_l$  where  $m_{l}$  is the maximum number of the nodes that can be sampled at layer  $l$  and  $d_{l}$  is the hidden state dimension. Initially, the buffer is empty. In every iteration, we first check if the sampled nodes are in the buffer. For the nodes that are found in the buffer, we collect the corresponding rows of the buffer and add them to  $\widetilde{Z}_k^{(l)}$  based on Formula (8) and (9). For the nodes that are not found in the buffer, we multiply the corresponding rows of  $\widetilde{Z}_k^{(l)}$  by  $\beta_{k}$ . The updated  $\widetilde{Z}_k^{(l)}$  is then written to chunk-  $(k\bmod t)$ . All the other chunks are multiplied by  $(1 - \beta_{k})$ . Since the sampled nodes found in the buffer are updated to chunk-  $(k\bmod t)$ , the original values in chunk-0 and chunk-1 are invalidated, as shown by the shadowed rows in Figure 1b. As the buffer size is a constant  $(T\cdot t\cdot m_l\cdot d_l)$  regardless of the graph size, our algorithm can be employed to train GNN on very large graphs.

Our algorithm overwrites chunk-  $(k\mod t)$  in iteration  $k$  . The information of the overwritten nodes is lost. The update of the moving averages can be written as

$$
y _ {k + 1} ^ {(l)} = \left(1 - \beta_ {k}\right) y _ {k} ^ {(l)} + \beta_ {k} f _ {\xi_ {l, k}} ^ {(l)} \left(y _ {k + 1} ^ {(l - 1)}\right) - \prod_ {j = k - t + 1} ^ {k - 1} \left(1 - \beta_ {j}\right) u _ {k} ^ {(l)} \tag {11}
$$

where

$$
u _ {k} ^ {(l)} = P \left(\xi_ {l, k - t} / \left(\xi_ {l, k - t + 1} \cup \dots \cup \xi_ {l, k}\right)\right) y _ {k - t + 1} ^ {(l)}. \tag {12}
$$

$P(\xi_{l,k - t} / (\xi_{l,k - t + 1}\cup \dots \cup \xi_{l,k}))$  is a projection matrix representing the overwritten nodes in iteration  $k$ , i.e., the nodes that are sampled in iteration  $k - t$  and are not sampled in the following  $t$  iterations. Since these nodes are multiplied by  $(1 - \beta_j)$  in every iteration after iteration  $k - t$ , the values of the overwritten rows are  $\prod_{j = k - t + 1}^{k - 1}(1 - \beta_j)u_k^{(l)}$ . Our algorithm simply replaces Formula (8) and (9) in the SCGD algorithm with Formula (11).

To study the convergence property of SpSC, we make the following assumptions that are commonly used in the analysis of SCO algorithms (Yang et al., 2019; Balasubramanian et al., 2020).

Assumption 1. The composite functions  $f^{(l)}$  are  $L_{l}$ -smooth. That is, for any  $y$  and  $y'$ , we have  $\| \nabla f_{\xi_l}^{(l)}(y) - \nabla f_{\xi_l}^{(l)}(y') \| \leq L_l \| y - y' \|$ .

Assumption 2. The stochastic gradients of the composite functions  $f^{(l)}$  are bounded in expectation, i.e.,  $\mathbb{E}[\| \nabla f_{\xi_l}^{(l)}(y)\|^2 ]\leq C_l^2$ .

Assumption 3. The estimated aggregation results obtained by sampled neighbor aggregation is unbiased, i.e.,  $\mathbb{E}[f_{\xi_l}^{(l)}(y)] = f^{(l)}(y)$ , and the stochastic gradient of  $f^{(l)}$  is unbiased, i.e.,  $\mathbb{E}[\nabla f_{\xi_l}^{(l)}(y)] = \nabla f^{(l)}(y)$ .

Following the single-timescale analysis of the algorithm (Balasubramanian et al., 2020), we use large batches for estimating the composite functions and assume that the estimation variance is small.

Assumption 4. The estimated aggregation results have small bounded variance, i.e.,  $\mathbb{E}[\| \nabla f_{\xi_l,k}^{(l)}(y)] - \nabla f^{(l)}(y)\| \leq \beta_kV^2$

This is a reasonable assumption for GNN training on GPUs as we always sample a batch of nodes for neighbor aggregation to achieve better utilization of the GPU parallelism.

In additional to the conventional assumptions, we make an assumption on the moving averages.

Assumption 5. The moving average of the aggregated features are bounded, i.e.,  $\mathbb{E}[\| y^{(l)}\| ^2 ]\leq D_l^2$

The convergence rate of our algorithm is summarized in the following theorem.

Theorem 1. Under Assumptions 1-5, if we choose  $\alpha_{k} = \alpha = \frac{c_{\alpha}}{\sqrt{K}}$  and  $\beta_{k} = \beta = \alpha (\sum_{l=1}^{T-1} A_{l}^{2}) / 2 = \frac{c_{\beta}}{\sqrt{K}}$ , the model parameters  $\{\theta_{k}\}$  of our training algorithm with (11) for updating sparse moving averages satisfy

$$
\begin{array}{l} \frac {1}{K} \sum_ {k = 0} ^ {K - 1} \mathbb {E} [ \| \nabla F (\theta_ {k}) \| ^ {2} ] \leq O (\beta) + \lambda_ {1} \frac {(1 - \beta) ^ {t + 1}}{\beta^ {3}} D ^ {2} (13) \\ + \lambda_ {2} \sum_ {l = 2} ^ {T} \left(\frac {(1 - \beta) ^ {2 (t - 1)} D ^ {2}}{\beta^ {3}} + \frac {C _ {l} ^ {2} (1 - \beta) ^ {2 (t - 1)} D ^ {2}}{\beta^ {3}}\right) (14) \\ \end{array}
$$

where  $\lambda_{1}$  and  $\lambda_{2}$  are constants, and  $t$  is the number of buffer chunks used in our algorithm.

The last term on the RHS of (13) reveals how the convergence of the SCGD algorithm is affected by the sparse moving averages. The larger  $t$  we use (i.e., the more chunks we have in the buffer), the smaller  $(1 - \beta)^{2(t-1)}$  and  $(1 - \beta)^{(t+1)}$  we have, and the faster convergence we achieve. In theory, if we can make  $(1 - \beta)^{2(t-1)} = O(\beta^4)$  and  $(1 - \beta)^{(t+1)} = O(\beta^4)$ , the algorithm will achieve  $O(\sqrt{1/K})$  convergence rate. As  $\beta \to 0$ , we need larger  $t$  to maintain the convergence rate, and eventually, we will need to store the moving average of all nodes in the graph.

Applying Sparse Moving Averages to SCSC. Our sparse moving average can also be applied to other SCO algorithms. For example, the Stochastically Corrected Stochastic Compositional gradient method (SCSC) (Chen et al., 2020) has a correction term in the update of the moving averages. Because of the correction term, SCSC needs a relaxed assumption on the estimation error of the composite functions. More specifically, if we change (11) to

$$
y _ {k + 1} ^ {(l)} = \left(1 - \beta_ {k}\right) y _ {k} ^ {(l)} + f _ {\xi_ {l, k}} ^ {(l)} \left(y _ {k + 1} ^ {(l - 1)}\right) - \left(1 - \beta_ {k}\right) f _ {\xi_ {l, k}} ^ {(l)} \left(y _ {k} ^ {(l - 1)}\right) - \prod_ {j = k - t + 1} ^ {k - 1} \left(1 - \beta_ {j}\right) u _ {k} ^ {(l)} \tag {15}
$$

and replace (8) and (9) with (15), we can relax Assumption 4 as

Assumption 6. The estimated aggregation results have bounded variance, i.e.,  $\mathbb{E}[\| \nabla f_{\xi_l,k}^{(l)}(y)] - \nabla f^{(l)}(y)\| \leq V^2$

The convergence rate of SCSC with sparse moving averages is summarized as follows.

Theorem 2. Under Assumptions 1-3 and 5-6, if we choose  $\alpha_{k} = \alpha = \frac{c_{\alpha}}{\sqrt{K}}$  and  $\beta_{k} = \beta = \alpha \sum_{l=1}^{N-1} A_{l}^{2} = \frac{c_{\beta}}{\sqrt{K}}$ , the model parameters  $\{\theta_{k}\}$  of SCSC with (15) for updating sparse moving averages satisfy

$$
\begin{array}{l} \frac {1}{K} \sum_ {k = 0} ^ {K - 1} \mathbb {E} [ \| \nabla F (\theta_ {k}) \| ^ {2} ] \leq O \left(\frac {1}{\sqrt {K}}\right) + \frac {2 (1 + \beta) (1 - \beta) ^ {2 (t - 1)}}{\alpha \beta} \sum_ {l = 1} ^ {T} D _ {l} ^ {2} \\ + \frac {6 (1 - \beta) ^ {2 (t - 1)}}{\alpha} C _ {u} \sum_ {l = 1} ^ {T} D _ {l} ^ {2} \\ \end{array}
$$

where  $C_u = \max \left(4C_l^2 +\gamma_l\right)$  for  $l = 1\dots T$

The result suggests that, if we can set  $t$  such that  $(1 - \beta)^{2(t - 1)} = O(\beta^3)$ , the algorithm will achieve  $O(\sqrt{1 / K})$  convergence rate.

# 4 IMPLEMENTATION DETAILS

Algorithm 1 describes an implementation of Formula (11) in our algorithm. For each layer, we allocate a buffer  $(buf)$  of size  $tm_l\times d_l$  for the moving averages. The buffered nodes are maintained in a list node_list  $\in \mathbb{R}^{tm_l}$  where node_list[i] is the index of the node stored at  $buf[i]$ . If  $buf[i]$  is empty, node_list[i] is set to -1. In every iteration  $k$ , we first get the location of chunk- $(k\bmod t)$ . Then, we look up each of the sampled nodes in the buffer (line 3). The LookUp function computes

Algorithm 1: Updating sparse moving average of aggregated features at layer  $l$  in iteration  $k$  
Input: Sampled nodes  $S$  ,Buffered nodes node_list  $\in \mathbb{R}^{tm_l}$ $buf^{(l)}\in \mathbb{R}^{tm_l\times d_l}$ $\widetilde{Z}_k^{(l)}\in \mathbb{R}^{|S|\times d_l}$  //Get the location of chunk-(k mod t)   
1 start  $= (k$  mod  $t)*m_{l}$  .   
2 end  $=$  chunk_start  $+|S|$  .   
// Look up the sampled nodes in the buffer   
3 idx_in_buf,idx_in_z  $=$  LookUp(S,node_list); // Update the moving average for the sampled nodes   
4  $\widetilde{Z}_k^{(l)} = \beta_k*\widetilde{Z}_k^{(l)}$  ..  $\widetilde{Z}_k^{(l)}[idx\_ in\_ z] = (1 - \beta_k)*buf^{(l)}[idx\_ in\_ buf] + \widetilde{Z}_k^{(l)}[idx\_ in\_ z];$  //Update the moving average for all buffered nodes   
5  $buf^{(l)} = (1 - \beta_k)*buf^{(l)}$  ..  $buf^{(l)}[start:end] = \widetilde{Z}_k^{(l)}$  . // Invalidate the old buffer for the sampled nodes   
6 node_list{idx_in_buf]  $= -1$  .   
// Add the sampled nodes to node_list   
7 node_list[ start : end]  $= S$

the intersection of  $S$  and node_list and returns the indices of the overlapping nodes in the two arrays. If a sampled node is not found in the buffer, we multiply the corresponding row of  $\widetilde{Z}_k^{(l)}$  by  $\beta_k$ . If a sampled node is in the buffer, we read in its current moving average and update the corresponding row of  $\widetilde{Z}_k^{(l)}$  (line 4). For buffered nodes that are not sampled, we simply multiply their moving averages by  $(1 - \beta_k)$  (line 5). For buffered nodes that are sampled, we invalidate their original buffer by setting node_list[idx_in_buf] to -1 (line 6). Last, we add the sampled nodes to the node_list.

Most of the operations in Algorithm 1 are simple vector operations, and they incur little overhead. The performance bottleneck is the LookUp function. With a naive implementation, it has  $O(tm_{l}\log |S|)$  time complexity, assuming  $S$  is sorted. In our implementation, we use an auxiliary array node\_loc  $\in \mathbb{R}^N$  to store the locations of all nodes in the buffer and accelerate the LookUp function. Specifically, if node- $i$  is in the buffer, we store its location in buffer in node\_loc[i]; otherwise, node\_loc[i] is set to -1. With the auxiliary array, the idx\_in\_z can be obtained by comparing node\_loc[S] with zero, and the idx\_in\_buf is simply node\_loc[S][idx\_in\_z]. Before updating the node_list at line 7 of Algorithm 1, we remove the overwritten nodes from node\_loc by setting node\_loc[node_list[ start : end]] to -1. Finally, we store the locations of the newly sampled nodes to node\_loc by setting node\_loc[S] to [start, start + 1, ..., end]. It is easy to see that all these operations have  $O(|S|)$  time complexity.

# 5 EVALUATION

# 5.1 EXPERIMENTAL SETUP

We conduct our experiments on a workstation with an Nvidia RTX 3090 GPU, an Intel Xeon Gold 6226R CPU, and 512GB RAM. Our code is implemented with PyTorch 1.8.0 and PyTorch Geometric 1.7.0.

We evaluate our algorithm on five graphs as listed in Table 1. The reddit and yelp graph are adopted from GraphSAINT (Zeng et al., 2020), and the arxiv, proteins, products are from the Open Graph Benchmark (Hu et al., 2020).

We apply our algorithm to two GNN models: GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017). Both models have three convolutional layers. We use Formula (11) for updating the moving average instead of Formula (15). This is because Formula (15) requires two forward passes which incurs extra overheads. The algorithm is run for 50 epochs. We set  $\beta$  to 0.2 initially, and decrease it to 0.1 at epoch 20, and further decrease it to 0.05 at epoch 40. The number of buffered chunks  $(t)$  is set to 8. We adopt the layer-wise sampling method in Zou et al. (2019) for neighbor sampling. The batch size is set to 4096, and the number of sampled neighbors in each layer is set to 8192.

# 5.2 TRAINING RESULTS

Validation Accuracy. Figure 3a shows the validation accuracy of different algorithms for training a GCN on arxiv. We can see that full neighbor aggregation (Adam_Full) achieves the highest

Table 1: Graph datasets ('m' stands for multi-label classification).  

<table><tr><td></td><td>reddit</td><td>yelp</td><td>arxiv</td><td>proteins</td><td>products</td></tr><tr><td>#nodes</td><td>233K</td><td>717K</td><td>169K</td><td>132K</td><td>2.4M</td></tr><tr><td>#edges</td><td>11.6M</td><td>7.0M</td><td>1.2M</td><td>79M</td><td>123M</td></tr><tr><td>#classes</td><td>41</td><td>100 (m)</td><td>40</td><td>112 (m)</td><td>47</td></tr></table>

![](images/f267f170fa46b4a2cb38265d7aa92e2a0fb84f64186c80fb6b9f245ef7ecaafc.jpg)  
(a) arxiv GCN

![](images/fb15403af37d10ebe6fef271a5bda9ad384b177fcead422fa190cf3decffefce.jpg)  
(b) reddit GraphSAGE

![](images/f2a8abe0b74e04b4d1be569b4295136fc3ae5bc7fe74d196284f04ebe863ec05.jpg)  
(c) proteins GraphSAGE

![](images/49484faec94f65dd3f2ad9f9d8f07da4e5c6820c53b0291d1d78a1ca780558f5.jpg)  
(d) products GCN

![](images/acdc8fe4aadc4ed5293e430be39322ca014a2010268568fd468a74eb372d59a9.jpg)  
(e) yelp GraphSAGE

![](images/57b5ea2c4174b7df9b900278759f01bca0bc575822df3aa6776f294b5778e6bc.jpg)  
Figure 2: Validation accuracy over epochs.

accuracy. This is reasonable because full neighbor aggregation returns unbiased estimates of gradients. If neighbor sampling is used, the accuracy clearly drops with the same training algorithm (Adam_Sample). Our algorithm (Sparse_SCO) is able to improve the convergence of sampling-based GNN training and achieves almost the same accuracy as full neighbor aggregation. The results on reddit and yelp are similar. On products graph, Adam_Full runs out of memory, so we only show the results of sampling-based training in Figure 2d. Interestingly, we find that our algorithm with neighbor sampling achieves even higher accuracy than Adam_Full on proteins graph, as shown in Figure 3c. The original SCO algorithm (which stores the moving averages for all nodes in the graph) also has lower accuracy than Sparse_SCO. This is probably due to the overfitting of models by Adam_Full and SCO.

Training Loss. Figure 3 shows the training loss of different algorithms on different graphs. For reddit and yelp, we are able to run full neighbor aggregation on our GPU. As expected, Adam_Full achieves the smallest training loss. Adam_Sample, however, has the slowest convergence. SCO achieves training loss close to Adam_Full. We run our algorithm with different  $t$ 's. The larger  $t$  we use, the smaller training loss we obtain. The results are consistent with our theoretical analysis and also suggesting that the poor accuracy of the original SCO algorithm is probably due to overfitting. For products graph, we are not able to run full neighbor aggregation. The results show a clearly faster convergence of our algorithm than Adam SGD for sampling-based training.

Test Accuracy. Table 2 lists the test accuracy of the models trained by different algorithms. For reddit and yelp, we follow the GraphSAINT paper (Zeng et al., 2020) and report the F1-micro score. For proteins, we follow the OGB (Hu et al., 2020) and report the ROC-AUC. We can see that our training algorithm achieves the highest test accuracy for both GCN and GraphSAGE on almost all the graphs. We do not include the results for GCN on yelp graph because its accuracy is apparently lower than GraphSAGE with all training algorithms, probably due to the limited expressiveness of the GCN model. While it is hard to draw a direct comparison with GraphSAINT because the sampling methods and the model architectures are different, our test accuracy on reddit and yelp matches the best accuracy reported by GraphSAINT (Zeng et al., 2020). It is worth noting that our algorithm achieves higher accuracy than Adam SGD with both full neighbor aggregation and neighbor sampling on proteins and products graph. The numbers are higher than the accuracy

![](images/f81e3d3c3bce8be8ab8a59e6fe6b434890186f7d4c520e9fdef09c78d3edf40a.jpg)  
(a) reddit GraphSAGE

![](images/4fb31998378f754af8c147a8b328d3911e716e347501934682340e08c1830eaa.jpg)  
(b) yelp GraphSAGE

![](images/c99e9444da92cf4cd42b970e926cf2c1235eb50313717a9701a5be873f8db2b7.jpg)  
Figure 3: Training loss over epochs.  
(c) products GCN

Table 2: Test accuracy of models trained by different algorithms on different graphs ('- ' means not available due to out of memory).  
(a) GCN  

<table><tr><td></td><td>reddit</td><td>arxiv</td><td>proteins</td><td>products</td></tr><tr><td>Adam_Full</td><td>0.961</td><td>0.712</td><td>0.737</td><td>-</td></tr><tr><td>Adam_Sample</td><td>0.957</td><td>0.663</td><td>0.713</td><td>0.790</td></tr><tr><td>SCO</td><td>0.945</td><td>0.698</td><td>0.715</td><td>-</td></tr><tr><td>Sparse_SCO</td><td>0.961</td><td>0.711</td><td>0.770</td><td>0.802</td></tr></table>

(b) GraphSAGE  

<table><tr><td></td><td>reddit</td><td>yelp</td><td>arxiv</td><td>proteins</td><td>products</td></tr><tr><td>Adam_Full</td><td>0.963</td><td>0.632</td><td>0.714</td><td>0.753</td><td>-</td></tr><tr><td>Adam_Sample</td><td>0.956</td><td>0.631</td><td>0.685</td><td>0.726</td><td>0.787</td></tr><tr><td>SCO</td><td>0.913</td><td>0.628</td><td>0.644</td><td>0.709</td><td>-</td></tr><tr><td>Sparse_SCO</td><td>0.966</td><td>0.651</td><td>0.713</td><td>0.779</td><td>0.801</td></tr></table>

of the same models reported in OGB Leaderboards (ogb, 2021). The results suggest that there is an improvement space for the accuracy of GNN models by using better training algorithms.

Execution Time. Table 3 lists the time per epoch of different training algorithms on different graphs. Although full neighbor aggregation achieves good convergence in many cases, it incurs a much large computation overhead than sampled neighbor aggregation. The execution time of Adam_Full is  $7\mathrm{x}$  to  $37\mathrm{x}$  longer than that of Adam_Sample. Compared with Adam_Sample, the SCO algorithm incurs a small overhead for updating the moving average of aggregation results. Our algorithm runs slightly slower than SCO because the LookUp operation in Algorithm 1 incurs an extra overhead. However, the execution time is still much smaller than full neighbor aggregation.

Memory Consumption. Figure 4 shows the memory consumption of different algorithms. We collect the numbers by calling the max_memory_allocated function in PyTorch at the end of the first epoch. For  $\mathsf{arxiv}$ , since it is a small graph, full neighbor aggregation has almost the same memory consumption as sampled neighbor aggregation. For reddit, proteins, and yelp, full neighbor aggregation takes much larger memory space than sampled aggregation. SCO and Sparse_SCO require additional memory for storing the moving averages. Because we only store the moving average of nodes sampled in recent iterations, Sparse_SCO uses less memory than SCO. On products graph, SCO aborts due to out of memory error, while our Sparse_SCO can run with only 2GB of GPU memory.

# 6 RELATED WORK

To overcome the scalability limitation of GNN training, various neighbor sampling methods have been proposed, including node-wise sampling (Hamilton et al., 2017; Ying et al., 2018), layer-wise sampling (Chen et al., 2018; Huang et al., 2018; Zou et al., 2019), and subgraph sampling (Zeng

Table 3: Time per epoch of different training algorithms in seconds ('-' means not available due to out of memory).  

<table><tr><td colspan="5">(a) GCN</td></tr><tr><td></td><td>reddit</td><td>arxiv</td><td>proteins</td><td>products</td></tr><tr><td>Adam_Full</td><td>28.9</td><td>2.89</td><td>19.88</td><td>-</td></tr><tr><td>Adam_Sample</td><td>1.43</td><td>0.50</td><td>0.84</td><td>0.66</td></tr><tr><td>SCO</td><td>1.85</td><td>0.65</td><td>1.03</td><td>-</td></tr><tr><td>Sparse_SCO</td><td>2.05</td><td>0.79</td><td>1.20</td><td>0.83</td></tr></table>

(b) GraphSAGE  

<table><tr><td></td><td>reddit</td><td>yelp</td><td>arxiv</td><td>proteins</td><td>products</td></tr><tr><td>Adam_Full</td><td>39.2</td><td>23.6</td><td>3.13</td><td>36.8</td><td>-</td></tr><tr><td>Adam_Sample</td><td>1.28</td><td>3.31</td><td>0.45</td><td>1.51</td><td>0.95</td></tr><tr><td>SCO</td><td>1.70</td><td>4.56</td><td>0.59</td><td>1.66</td><td>-</td></tr><tr><td>Sparse_SCO</td><td>1.82</td><td>5.05</td><td>0.62</td><td>1.89</td><td>1.16</td></tr></table>

![](images/c47d2e68dec91027d8d7cc8995823a45f478787d021530b25a2c77b9befc3867.jpg)  
(a) arxiv GCN

![](images/993648b347180573c24cef327e1a86aeee7ea99ce185807d4a20651a733ca66d.jpg)  
(b) reddit GraphSAGE

![](images/772f1e466cb5fcec443dc12cf0e95d129376e1796b0d1b9d3b75d741eef7c8a9.jpg)  
(c) proteins GraphSAGE

![](images/1344d7122e39c3ff8267f5c0a52e9ffa331f0ba5a501eaa0a35c33e060882e1d.jpg)  
Figure 4: GPU memory consumption of different algorithms.  
(d) products GCN

![](images/318c21104a2df99727a40e0878c16ed735bb2a4eafdc216c733b8b0c7e1e866a.jpg)  
(e) yelp GraphSAGE

et al., 2020; Chiang et al., 2019). These sampling-based training methods achieve good accuracy in practice (particularly on small graphs), but they lack theoretical justification.

Some recent works (Cong et al., 2020; 2021) point out that sampling-based GNN training is actually multi-level Stochastic Compositional Optimization (SCO). However, they either use this connection to justify their sampling techniques and fall back to Adam SGD for training (Cong et al., 2020), or they directly adopt an SCO algorithm without considering the large memory consumption issue (Cong et al., 2021).

The research on SCO traces back to (Ermoliev, 1976) where a two-timescale stochastic approximation scheme was proposed for two-level problems. Various SCO algorithms and convergence analyses have been proposed ever since (Wang et al., 2017a;b; Lian et al., 2017; Yang et al., 2019; Zhang & Xiao, 2019; Chen et al., 2020; Balasubramanian et al., 2020; Ghadimi et al., 2020; Hu et al., 2019; Lin et al., 2018). Despite a substantial volume of work on SCO in recent years, none of the existing work has considered the large memory consumption issue and the data movement overhead of the algorithms. Our work is the first to establish a convergence analysis for SCO algorithms with sparse moving averages.

# 7 CONCLUSION

In this work, we propose a new variant of SCO algorithm for training graph neural networks on large graphs. Our main idea is to maintain a sparse moving average of the aggregation results in each convolutional layer. We study the convergence property of our algorithm and show that the algorithm can achieve  $O(\sqrt{1 / K})$  convergence rate when a sufficient amount of moving averages are maintained. Our experiments with two GNN models on different graphs validate our theoretical results and show a clear advantage of our algorithm against Adam SGD for GNN training.

# REFERENCES

Ogb leaderboards for node classification, 2021. URL https://ogb.stanford.edu/docs/leader_nodeprop/.  
Krishnakumar Balasubramanian, Saeed Ghadimi, and Anthony Nguyen. Stochastic multi-level composition optimization algorithms with level-independent convergence rates. arXiv preprint arXiv:2008.10526, 2020.  
Jie Chen, Tengfei Ma, and Cao Xiao. FastGCN: Fast learning with graph convolutional networks via importance sampling. In International Conference on Learning Representations, 2018.  
Tianyi Chen, Yuejiao Sun, and Wotao Yin. Solving stochastic compositional optimization is nearly as easy as solving stochastic optimization. arXiv preprint arXiv:2008.10847, 2020.  
Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 257-266, 2019.  
Weilin Cong, Rana Forsati, Mahmut Kandemir, and Mehrdad Mahdavi. Minimal variance sampling with provable guarantees for fast training of graph neural networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1393-1403, 2020.  
Weilin Cong, Morteza Ramezani, and Mehrdad Mahdavi. On the importance of sampling in learning graph convolutional networks. arXiv preprint arXiv:2103.02696, 2021.  
Alberto Garcia Duran and Mathias Niepert. Learning graph representations with embedding propagation. In Advances in neural information processing systems, pp. 5119-5130, 2017.  
Yu. Ermoliev. Methods of stochastic programming. Monographs in Optimization and OR, 1976. In Russian.  
Saeed Ghadimi, Andrzej Ruszczyński, and Mengdi Wang. A single timescale stochastic approximation method for nested stochastic optimization. SIAM Journal on Optimization, 30(1):960-979, 2020.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in neural information processing systems, pp. 1024-1034, 2017.  
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. arXiv preprint arXiv:2005.00687, 2020.  
Wenqing Hu, Chris Junchi Li, Xiangru Lian, Ji Liu, and Huizhuo Yuan. Efficient smooth non-convex stochastic compositional optimization via stochastic recursive gradient descent. In Advances in Neural Information Processing Systems, pp. 6929–6937, 2019.  
Wenbing Huang, Tong Zhang, Yu Rong, and Junzhou Huang. Adaptive sampling towards fast graph representation learning. In Advances in neural information processing systems, pp. 4558-4567, 2018.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.  
Ruoyu Li, Sheng Wang, Feiyun Zhu, and Junzhou Huang. Adaptive graph convolutional neural networks. In AAAI Conference on Artificial Intelligence, 2018.  
Xiangru Lian, Mengdi Wang, and Ji Liu. Finite-sum composition optimization via variance reduced gradient descent. In Artificial Intelligence and Statistics, pp. 1159-1167, 2017.  
Tianyi Lin, Chenyou Fan, and Mengdi Wang. Improved oracle complexity of variance reduced methods for nonsmooth convex stochastic composition optimization. arXiv, pp. arXiv-1802, 2018.

Mengdi Wang, Ethan X Fang, and Han Liu. Stochastic compositional gradient descent: algorithms for minimizing compositions of expected-value functions. Mathematical Programming, 161(1-2): 419-449, 2017a.  
Mengdi Wang, Ji Liu, and Ethan X Fang. Accelerating stochastic composition optimization. The Journal of Machine Learning Research, 18(1):3721-3743, 2017b.  
Shuoguang Yang, Mengdi Wang, and Ethan X Fang. Multilevel stochastic gradient methods for nested composition optimization. SIAM Journal on Optimization, 29(1):616-659, 2019.  
Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. Graph convolutional neural networks for web-scale recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 974-983, 2018.  
Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, Rajgopal Kannan, and Viktor Prasanna. Graph-saint: Graph sampling based inductive learning method. In International Conference on Learning Representations, 2020.  
Junyu Zhang and Lin Xiao. Multi-level composite stochastic optimization via nested variance reduction. arXiv preprint arXiv:1908.11468, 2019.  
Muhan Zhang and Yixin Chen. Weisfeiler-lehman neural machine for link prediction. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 575-583, 2017.  
Muhan Zhang and Yixin Chen. Link prediction based on graph neural networks. In Advances in Neural Information Processing Systems, pp. 5165-5175, 2018.  
Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. Graph neural networks: A review of methods and applications. arXiv preprint arXiv:1812.08434, 2018.  
Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. Layer-dependent importance sampling for training deep and large graph convolutional networks. In Advances in Neural Information Processing Systems, pp. 11249-11259, 2019.
