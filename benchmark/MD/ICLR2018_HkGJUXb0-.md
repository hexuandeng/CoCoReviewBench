# LEARNING EFFICIENT TENSOR REPRESENTATIONS WITH RING STRUCTURE NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Tensor train (TT) decomposition is a powerful representation for high-order tensors, which has been successfully applied to various machine learning tasks in recent years. In this paper, we propose a more generalized tensor decomposition with ring structure network by employing circular multilinear products over a sequence of lower-order core tensors, which is termed as TR representation. Several learning algorithms including blockwise ALS with adaptive tensor ranks and SGD with high scalability are presented. Furthermore, the mathematical properties are investigated, which enables us to perform basic algebra operations in a computationally efficiently way by using TR representations. Experimental results on synthetic signals and real-world datasets demonstrate the effectiveness of TR model and the learning algorithms. In particular, we show that the structure information and high-order correlations within a 2D image can be captured efficiently by employing an appropriate tensorization and TR decomposition.

# 1 INTRODUCTION

Tensor decompositions aim to represent a higher-order (or multi-dimensional) data as a multilinear product of several latent factors, which attracted considerable attentions in machine learning (Yu & Liu, 2016; Anandkumar et al., 2014; Romera-Paredes et al., 2013; Kanagawa et al., 2016; Yang et al., 2017) and signal processing (Zhou et al., 2016) in recent years. For a  $d$ th-order tensor with "square" core tensor of size  $r$ , standard tensor decompositions are the canonical polyadic  $(CP)$  decomposition (Goulart et al., 2015) which represents data as a sum of rank-one tensors by  $\mathcal{O}(dnr)$  parameters and Tucker decomposition (De Lathauwer et al., 2000; Xu et al., 2012; Wu et al., 2014; Zhe et al., 2016) which represents data as a core tensor and several factor matrices by  $\mathcal{O}(dnr + r^d)$  parameters. In general, CP decomposition provides a compact representation but with difficulties in finding the optimal solution, while Tucker decomposition is stable and flexible but its number of parameters scales exponentially to the tensor order.

Recently, tensor networks have emerged as a powerful tool for analyzing very high-order tensors (Cichocki et al., 2016). A powerful tensor network is tensor train / matrix product states (TT/MPS) representation (Oseledets, 2011), which requires  $\mathcal{O}(dnr^2)$  parameters and avoid the curse of dimensionality through a particular geometry of low-order contracted tensors. TT representation has been applied to model weight parameters in deep neural network and nonlinear kernel learning (Novikov et al., 2015; Stoudenmire & Schwab, 2016; Tsai et al., 2016), achieving a significant compression factor and scalability. It also has been successfully used for feature learning and classification (Bengua et al., 2015). To fully explore the advantages of tensor algebra, the key step is to efficiently represent the real-world dataset by tensor networks, which is not well studied. In addition, there are some limitations of TT including that i) the constraint on TT-ranks, i.e.,  $r_1 = r_{d+1} = 1$ , leads to the limited representation ability and flexibility; ii) TT-ranks are bounded by the rank of k-unfolding matricization, which might not be optimal; iii) the permutation of data tensor will yield an inconsistent solution, i.e., TT representations and TT-ranks are sensitive to the order of tensor dimensions. Hence, finding the optimal permutation remains a challenging problem.

In this paper, we introduce a new structure of tensor networks, which can be considered as a generalization of TT representations. First of all, we relax the condition over TT-ranks, i.e.,  $r_1 = r_{d+1} = 1$ , leading to an enhanced representation ability. Secondly, the strict ordering of multilinear products between cores should be alleviated. Third, the cores should be treated equivalently by

![](images/484de92940d0b7a3cfb5b93df701c0722718cc60f050e33586e86250c9d51471.jpg)  
Figure 1: The effects of noise corrupted tensor cores. From left to right, each figure shows noise corruption by adding noise to one specific tensor core.

![](images/95229245942c8f9b5f00a59cca18ae3046fbb27b66da266e383b6f96cf9b8638.jpg)

![](images/11897bef650495d4448f552d7482bdf67be7a39580bcee535d73dcf77634269b.jpg)

![](images/0db59f72e7ceac5e6517890ebc1c823018a49bc54b62cbabed174e7001156daa.jpg)

![](images/8b525d9e31ad1228689a4a064fc9ff97d294b2cac4cb8a68c3803750d6e3f640.jpg)  
Figure 2: A graphical representation of tensor ring decomposition.

making the model symmetric. To this end, we add a new connection between the first and the last core tensors, yielding a circular tensor products of a set of cores (see Fig. 2). More specifically, we consider that each tensor element is approximated by performing a trace operation over the sequential multilinear products of cores. Since the trace operation ensures a scalar output,  $r_1 = r_{d+1} = 1$  is not necessary. In addition, the cores can be circularly shifted and treated equivalently due to the properties of the trace operation. We call this model tensor ring (TR) decomposition and its cores tensor ring (TR) representations. To learn TR representations, we firstly develop a non-iterative TR-SVD algorithm that is similar to TT-SVD algorithm (Oseledets, 2011). To find the optimal lower TR-ranks, a block-wise ALS algorithms is presented. Finally, we also propose a scalable algorithm by using stochastic gradient descend, which can be applied to handling large-scale datasets.

Another interesting contribution is that we show the intrinsic structure or high order correlations within a 2D image can be captured more efficiently than SVD by converting 2D matrix to a higher order tensor. For example, given an image of size  $I \times J$ , we can apply an appropriate tensorization operation (see details in Sec. 5.2) to obtain a fourth order tensor, of which each mode controls one specific scale of resolution. To demonstrate this, Fig. 1 shows the effects caused by noise corruption of specific tensor cores. As we can see, the first mode corresponds to the small-scale patches, while the 4th-mode corresponds to the large-scale partitions. We have shown in Sec. 5.2 that TR model can represent the image more efficiently than the standard SVD.

# 2 TENSOR RING DECOMPOSITION

The TR decomposition aims to represent a high-order (or multi-dimensional) tensor by a sequence of 3rd-order tensors that are multiplied circularly. Specifically, let  $\mathcal{T}$  be a  $d$ th-order tensor of size  $n_1\times n_2\times \dots \times n_d$ , denoted by  $\mathcal{T}\in \mathbb{R}^{n_1\times \dots \times n_d}$ , TR representation is to decompose it into a sequence of latent tensors  $\mathcal{Z}_k\in \mathbb{R}^{r_k\times n_k\times r_{k + 1}}$ ,  $k = 1,2,\ldots ,d$ , which can be expressed in an element-wise form given by

$$
T \left(i _ {1}, i _ {2}, \dots , i _ {d}\right) = \operatorname {T r} \left\{\mathbf {Z} _ {1} \left(i _ {1}\right) \mathbf {Z} _ {2} \left(i _ {2}\right) \dots \mathbf {Z} _ {d} \left(i _ {d}\right) \right\} = \operatorname {T r} \left\{\prod_ {k = 1} ^ {d} \mathbf {Z} _ {k} \left(i _ {k}\right) \right\}. \tag {1}
$$

$T(i_{1},i_{2},\ldots ,i_{d})$  denotes the  $(i_1,i_2,\dots ,i_d)$  th element of the tensor.  $\mathbf{Z}_k(i_k)$  denotes the  $i_k$  th lateral slice matrix of the latent tensor  $\mathcal{Z}_k$ , which is of size  $r_k\times r_{k + 1}$ . Note that any two adjacent latent tensors,  $\mathcal{Z}_k$  and  $\mathcal{Z}_{k + 1}$ , have a common dimension  $r_{k + 1}$  on their corresponding modes. The last latent tensor  $\mathcal{Z}_d$  is of size  $r_d\times n_d\times r_1$ , i.e.,  $r_{d + 1} = r_1$ , which ensures the product of these matrices is a square matrix. These prerequisites play key roles in TR decomposition, resulting in some important numerical properties. For simplicity, the latent tensor  $\mathcal{Z}_k$  can also be called the kth-core (or node). The size of cores,  $r_k,k = 1,2,\ldots ,d$ , collected and denoted by a vector  $\mathbf{r} = [r_1,r_2,\dots ,r_d]^T$ , are called TR-ranks. From (1), we can observe that  $T(i_{1},i_{2},\ldots ,i_{d})$  is equivalent to the trace of a

sequential product of matrices  $\{\mathbf{Z}_k(i_k)\}$ . Based on (1), we can also express TR decomposition in the tensor form, given by

$$
\boldsymbol {\mathcal {T}} = \sum_ {\alpha_ {1}, \dots , \alpha_ {d} = 1} ^ {r _ {1}, \dots , r _ {d}} \mathbf {z} _ {1} (\alpha_ {1}, \alpha_ {2}) \circ \mathbf {z} _ {2} (\alpha_ {2}, \alpha_ {3}) \circ \dots \circ \mathbf {z} _ {d} (\alpha_ {d}, \alpha_ {1}),
$$

where the symbol  $\circ$  denotes the outer product of vectors and  $\mathbf{z}_k(\alpha_k, \alpha_{k+1}) \in \mathbb{R}^{n_k}$  denotes the  $(\alpha_k, \alpha_{k+1})$ th mode-2 fiber of tensor  $\mathcal{Z}_k$ . The number of parameters in TR representation is  $\mathcal{O}(dnr^2)$ , which is linear to the tensor order  $d$  as in TT representation.

The TR representation can also be illustrated graphically by a linear tensor network as shown in Fig. 2. A node represents a tensor (including a matrix and a vector) whose order is denoted by the number of edges. The number by an edge specifies the size of each mode (or dimension). The connection between two nodes denotes a multilinear product operator between two tensors on a specific mode. This is also called tensor contraction, which corresponds to the summation over the indices of that mode. It should be noted that  $\mathcal{Z}_d$  is connected to  $\mathcal{Z}_1$  by the summation over the index  $\alpha_{1}$ , which is equivalent to the trace operation. For simplicity, we denote TR decomposition by  $\mathcal{T} = \Re (\mathcal{Z}_1,\mathcal{Z}_2,\ldots ,\mathcal{Z}_d)$ .

Theorem 1 (Circular dimensional permutation invariance). Let  $\mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \ldots \times n_d}$  be a dth-order tensor and its TR decomposition is given by  $\mathcal{T} = \Re(\mathcal{Z}_1, \mathcal{Z}_2, \ldots, \mathcal{Z}_d)$ . If we define  $\overleftarrow{\mathcal{T}}^k \in \mathbb{R}^{n_{k+1} \times \cdots \times n_d \times n_1 \times \cdots \times n_k}$  as the circularly shifted version along the dimensions of  $\mathcal{T}$  by  $k$ , then we have  $\overleftarrow{\mathcal{T}}^k = \Re(\mathcal{Z}_{k+1}, \ldots, \mathcal{Z}_d, \mathcal{Z}_1, \ldots, \mathcal{Z}_k)$ .

A proof of Theorem 1 is provided in Appendix B.1.

It should be noted that circular dimensional permutation invariance is an essential feature that distinguishes TR decomposition from TT decomposition. For TT decomposition, the product of matrices must keep a strictly sequential order, yielding that the tensor with a circular dimension shifting does not correspond to the shifting of tensor cores.

# 3 LEARNING ALGORITHMS

# 3.1 SEQUENTIAL SVDS

We propose the first algorithm for computing the TR decomposition using  $d$  sequential SVDs. This algorithm will be called the TR-SVD algorithm.

Theorem 2. Let us assume  $\mathcal{T}$  can be represented by a TR decomposition. If the  $k$ - unfolding matrix  $\mathbf{T}_{\langle k\rangle}$  has  $\mathrm{Rank}(\mathbf{T}_{\langle k\rangle}) = R_{k + 1}$ , then there exists a TR decomposition with TR-ranks  $\mathbf{r}$  which satisfies that  $\exists k, r_1r_{k + 1} \leq R_{k + 1}$ .

Proof. We can express TR decomposition in the form of  $k$ - unfolding matrix,

$$
T _ {\langle k \rangle} \left(\overline {{i _ {1} \cdots i _ {k}}}, \overline {{i _ {k + 1} \cdots i _ {d}}}\right) = \operatorname {T r} \left\{\prod_ {j = 1} ^ {k} \mathbf {Z} _ {j} \left(i _ {j}\right) \prod_ {j = k + 1} ^ {d} \mathbf {Z} _ {j} \left(i _ {j}\right) \right\} = \left\langle \operatorname {v e c} \left(\prod_ {j = 1} ^ {k} \mathbf {Z} _ {j} \left(i _ {j}\right)\right), \operatorname {v e c} \left(\prod_ {j = d} ^ {k + 1} \mathbf {Z} _ {j} ^ {T} \left(i _ {j}\right)\right) \right\rangle . \tag {2}
$$

It can also be rewritten as

$$
T _ {\langle k \rangle} \left(\overline {{i _ {1} \cdots i _ {k}}}, \overline {{i _ {k + 1} \cdots i _ {d}}}\right) = \sum_ {\alpha_ {1} \alpha_ {k + 1}} Z ^ {\leq k} \left(\overline {{i _ {1} \cdots i _ {k}}}, \overline {{\alpha_ {1} \alpha_ {k + 1}}}\right) Z ^ {> k} \left(\overline {{\alpha_ {1} \alpha_ {k + 1}}}, \overline {{i _ {k + 1} \cdots i _ {d}}}\right), \tag {3}
$$

where we defined the subchain by merging multiple linked cores as  $\mathbf{Z}^{<k}(\overline{i_1 \cdots i_{k-1}}) = \prod_{j=1}^{k-1} \mathbf{Z}_j(i_j)$  and  $\mathbf{Z}^{>k}(\overline{i_{k+1} \cdots i_d}) = \prod_{j=k+1}^d \mathbf{Z}_j(i_j)$ . Hence, we can obtain  $\mathbf{T}_{\langle k \rangle} = \mathbf{Z}_{(2)}^{<k}(\mathbf{Z}_{[2]}^{>k})^T$ , where the subchain  $\mathbf{Z}_{(2)}^{<k}$  is of size  $\prod_{j=1}^k n_j \times r_1 r_{k+1}$ , and  $\mathbf{Z}_{[2]}^{>k}$  is of size  $\prod_{j=k+1}^d n_j \times r_1 r_{k+1}$ . Since the rank of  $\mathbf{T}_{\langle k \rangle}$  is  $R_{k+1}$ , we can obtain  $r_1 r_{k+1} \leq R_{k+1}$ .

According to (2) and (3), TR decomposition can be written as

$$
T _ {\langle 1 \rangle} (i _ {1}, \overline {{i _ {2} \cdots i _ {d}}}) = \sum_ {\alpha_ {1}, \alpha_ {2}} Z ^ {\leq 1} (i _ {1}, \overline {{\alpha_ {1} \alpha_ {2}}}) Z ^ {> 1} (\overline {{\alpha_ {1} \alpha_ {2}}}, \overline {{i _ {2} \cdots i _ {d}}}).
$$

Since the low-rank approximation of  $\mathbf{T}_{\langle 1\rangle}$  can be obtained by the truncated SVD, which is  $\mathbf{T}_{\langle 1\rangle} = \mathbf{U}\Sigma \mathbf{V}^T +\mathbf{E}_1$ , the first core  $\mathcal{Z}_1(i.e.,\mathcal{Z}^{\leq 1})$  of size  $r_1\times n_1\times r_2$  can be obtained by the proper reshaping and permutation of  $\mathbf{U}$  and the subchain  $\mathcal{Z}^{>1}$  of size  $r_2\times \prod_{j = 2}^d n_j\times r_1$  is obtained by the proper reshaping and permutation of  $\boldsymbol{\Sigma}\mathbf{V}^{T}$ , which corresponds to the remaining  $d - 1$  dimensions of  $\mathcal{T}$ . Note that this algorithm uses the similar strategy with TT-SVD (Oseledets, 2011), but the reshaping and permutations are totally different between them. Subsequently, we can further reshape the subchain  $\mathcal{Z}^{>1}$  as a matrix  $\mathbf{Z}^{>1}\in \mathbb{R}^{r_2n_2\times \prod_{j = 3}^d n_jr_1}$  which thus can be written as

$$
Z ^ {> 1} (\overline {{\alpha_ {2} i _ {2}}}, \overline {{i _ {3} \cdots i _ {d} \alpha_ {1}}}) = \sum_ {\alpha_ {3}} Z _ {2} (\overline {{\alpha_ {2} i _ {2}}}, \alpha_ {3}) Z ^ {> 2} (\alpha_ {3}, \overline {{i _ {3} \cdots i _ {d} \alpha_ {1}}}).
$$

By applying truncated SVD, i.e.,  $\mathbf{Z}^{>1} = \mathbf{U}\Sigma \mathbf{V}^T +\mathbf{E}_2$  , we can obtain the second core  $\mathcal{Z}_2$  of size  $(r_2\times n_2\times r_3)$  by appropriately reshaping U and the subchain  $\mathcal{Z}^{>2}$  by proper reshaping of  $\boldsymbol{\Sigma}\mathbf{V}^{T}$  This procedure can be performed sequentially to obtain all  $d$  cores  $\mathcal{Z}_k,k = 1,\ldots ,d$

As proved in (Oseledets, 2011), the approximation error by using such sequential SVDs is given by

$$
\left\| \boldsymbol {\mathcal {T}} - \Re (\boldsymbol {\mathcal {Z}} _ {1}, \boldsymbol {\mathcal {Z}} _ {2}, \dots , \boldsymbol {\mathcal {Z}} _ {d}) \right\| _ {F} \leq \sqrt {\sum_ {k = 1} ^ {d - 1} \left\| \mathbf {E} _ {k} \right\| _ {F} ^ {2}}.
$$

Hence, given a prescribed relative error  $\epsilon_{p}$ , the truncation threshold  $\delta$  can be set to  $\frac{\epsilon_p}{\sqrt{d - 1}}\|\mathcal{T}\|_F$ . However, considering that  $\|\mathbf{E}_1\|_F$  corresponds to two ranks including both  $r_1$  and  $r_2$ , while  $\|\mathbf{E}_k\|_F$ ,  $\forall k > 1$  correspond to only one rank  $r_{k + 1}$ . Therefore, we modify the truncation threshold as

$$
\delta_ {k} = \left\{ \begin{array}{l l} \sqrt {2} \epsilon_ {p} \| \boldsymbol {\mathcal {T}} \| _ {F} / \sqrt {d} & k = 1, \\ \epsilon_ {p} \| \boldsymbol {\mathcal {T}} \| _ {F} / \sqrt {d} & k > 1. \end{array} \right. \tag {4}
$$

A pseudocode of the TR-SVD algorithm is summarized in Alg. 1. Note that the cores obtained by the TR-SVD algorithm are left-orthogonal, which is  $\mathbf{Z}_{k\langle 2\rangle}^{T}\mathbf{Z}_{k\langle 2\rangle} = \mathbf{I}$  for  $k = 2,\dots ,d - 1$

# 3.2 BLOCK-WISE ALTERNATING LEAST-SQUARES (ALS)

The ALS algorithm has been widely applied to various tensor decomposition models such as CP and Tucker decompositions (Kolda & Bader, 2009; Holtz et al., 2012). The main concept of ALS is optimizing one core while the other cores are fixed, and this procedure will be repeated until some convergence criterion is satisfied. Given a  $d$ th-order tensor  $\mathcal{T}$ , our goal is to optimize the error function as

$$
\min  _ {\boldsymbol {Z} _ {1}, \dots , \boldsymbol {Z} _ {d}} \| \boldsymbol {\mathcal {T}} - \Re (\boldsymbol {Z} _ {1}, \dots , \boldsymbol {Z} _ {d}) \| _ {F}. \tag {5}
$$

According to the TR definition in (1), we have

$$
\begin{array}{l} T (i _ {1}, i _ {2}, \dots , i _ {d}) = \sum_ {\alpha_ {1}, \dots , \alpha_ {d}} Z _ {1} (\alpha_ {1}, i _ {1}, \alpha_ {2}) Z _ {2} (\alpha_ {2}, i _ {2}, \alpha_ {3}) \dots Z _ {d} (\alpha_ {d}, i _ {d}, \alpha_ {1}) \\ = \sum_ {\alpha_ {k}, \alpha_ {k + 1}} \left\{Z _ {k} (\alpha_ {k}, i _ {k}, \alpha_ {k + 1}) Z ^ {\neq k} (\alpha_ {k + 1}, \overline {{i _ {k + 1} \cdots i _ {d} i _ {1} \cdots i _ {k - 1}}}, \alpha_ {k}) \right\}, \\ \end{array}
$$

where  $\mathbf{Z} \neq k(\overline{i_{k+1} \cdots i_d i_1 \cdots i_{k-1}}) = \prod_{j=k+1}^d \mathbf{Z}_j(i_j) \prod_{j=1}^{k-1} \mathbf{Z}_j(i_j)$  denotes a slice matrix of subchain tensor by merging all cores except  $k$ th core  $\mathcal{Z}_k$ . Hence, the mode- $k$  unfolding matrix of  $\mathcal{T}$  can be expressed by

$$
T _ {[ k ]} (i _ {k}, \overline {{i _ {k + 1} \cdots i _ {d} i _ {1} \cdots i _ {k - 1}}}) = \sum_ {\alpha_ {k} \alpha_ {k + 1}} \Big \{Z _ {k} (i _ {k}, \overline {{\alpha_ {k} \alpha_ {k + 1}}}) Z ^ {\neq k} (\overline {{\alpha_ {k} \alpha_ {k + 1}}}, \overline {{i _ {k + 1} \cdots i _ {d} i _ {1} \cdots i _ {k - 1}}}) \Big \}.
$$

By applying different mode- $k$  unfolding operations, we can obtain that  $\mathbf{T}_{[k]} = \mathbf{Z}_{k(2)}\left(\mathbf{Z}_{[2]}^{\neq k}\right)^T$ , where  $\mathcal{Z}^{\neq k}$  is a subchain obtained by merging  $d - 1$  cores.

The objective function in (5) can be optimized by solving  $d$  subproblems alternatively. More specifically, having fixed all but one core, the problem reduces to a linear least squares problem, which is

$$
\min  _ {\mathbf {Z} _ {k (2)}} \left\| \mathbf {T} _ {[ k ]} - \mathbf {Z} _ {k (2)} \left(\mathbf {Z} _ {[ 2 ]} ^ {\neq k}\right) ^ {T} \right\| _ {F}, \quad k = 1, \dots , d.
$$

This algorithm is called TT-ALS.

Here, we propose a computationally efficient block-wise ALS (BALS) algorithm by utilizing truncated SVD, which facilitates the self-adaptation of ranks. The main idea is to perform the blockwise optimization followed by the separation of a block into individual cores. To achieve this, we consider merging two linked cores, e.g.,  $\mathcal{Z}_k$ ,  $\mathcal{Z}_{k+1}$ , into a block (or subchain)  $\mathcal{Z}^{(k,k+1)} \in \mathbb{R}^{r_k \times n_k n_{k+1} \times r_{k+2}}$ . Thus, the subchain  $\mathcal{Z}^{(k,k+1)}$  can be optimized while leaving all cores except  $\mathcal{Z}_k$ ,  $\mathcal{Z}_{k+1}$  fixed. Subsequently, the subchain  $\mathcal{Z}^{(k,k+1)}$  can be reshaped into  $\tilde{\mathbf{Z}}^{(k,k+1)} \in \mathbb{R}^{r_k n_k \times n_{k+1} r_{k+2}}$  and separated into a left-orthonormal core  $\mathcal{Z}_k$  and  $\mathcal{Z}_{k+1}$  by a truncated SVD:

$$
\tilde {\mathbf {Z}} ^ {(k, k + 1)} = \mathbf {U} \boldsymbol {\Sigma} \mathbf {V} ^ {T} = \mathbf {Z} _ {k \langle 2 \rangle} \mathbf {Z} _ {k + 1 \langle 1 \rangle}, \tag {6}
$$

where  $\mathbf{Z}_{k\langle 2\rangle}\in \mathbb{R}^{r_kn_k\times r_{k + 1}}$  is the 2-unfolding matrix of core  $\mathcal{Z}_k$ , which can be set to  $\mathbf{U}$ , while  $\mathbf{Z}_{k + 1\langle 1\rangle}\in \mathbb{R}^{r_{k + 1}\times n_{k + 1}r_{k + 2}}$  is the 1-unfolding matrix of core  $\mathcal{Z}_{k + 1}$ , which can be set to  $\boldsymbol{\Sigma}\mathbf{V}^T$ . This procedure thus moves on to optimize the next block cores  $\mathcal{Z}^{(k + 1,k + 2)},\ldots ,\mathcal{Z}^{(d - 1,d)},\mathcal{Z}^{(d,1)}$  successively in the similar way. Note that since the TR model is circular, the  $d$ th core can also be merged with the first core yielding the block core  $\mathcal{Z}^{(d,1)}$ .

The key advantage of our BALS algorithm is the rank adaptation ability which can be achieved simply by separating the block core into two cores via truncated SVD, as shown in (6). The truncated rank  $r_{k+1}$  can be chosen such that the approximation error is below a certain threshold. One possible choice is to use the same threshold as in the TR-SVD algorithm, i.e.,  $\delta_k$  described in (4). However, the empirical experience shows that this threshold often leads to overfitting and the truncated rank is higher than the optimal rank. This is because the updated block  $\mathcal{Z}^{(k,k+1)}$  during ALS iterations is not a closed form solution and many iterations are necessary for convergence. To relieve this problem, we choose the truncation threshold based on both the current and the desired approximation errors, which is

$$
\delta = \max  \left\{\epsilon \| \boldsymbol {\mathcal {T}} \| _ {F} / \sqrt {d}, \epsilon_ {p} \| \boldsymbol {\mathcal {T}} \| _ {F} / \sqrt {d} \right\}.
$$

A pseudo code of the BALS algorithm is described in Alg. 2.

# 3.3 STOCHASTIC GRADIENT DESCENT

For large-scale dataset, the ALS algorithm is not scalable due to the cubic time complexity in the target rank, while Stochastic Gradient Descent (SGD) shows high efficiency and scalability for matrix/tensor factorization (Gemulla et al., 2011; Maehara et al., 2016; Wang & Anandkumar, 2016). In this section, we present a scalable and efficient TR decomposition by using SGD, which is also suitable for online learning and tensor completion problems. To this end, we first provide the element-wise loss function, which is

$$
L \left(\boldsymbol {Z} _ {1}, \boldsymbol {Z} _ {2}, \dots , \boldsymbol {Z} _ {d}\right) = \frac {1}{2} \sum_ {i _ {1}, \dots , i _ {d}} \left\{T \left(i _ {1}, i _ {2}, \dots , i _ {d}\right) - \operatorname {T r} \left(\prod_ {k = 1} ^ {d} \mathbf {Z} _ {k} \left(i _ {k}\right)\right) \right\} ^ {2} + \frac {1}{2} \lambda_ {k} \| \mathbf {Z} _ {k} \left(i _ {k}\right) \| ^ {2}, \tag {7}
$$

where  $\lambda_{k}$  is the regularization parameters. The core idea of SGD is to randomly select one sample  $\pmb{\mathcal{T}}(i_1,i_2,\dots ,i_d)$ , then update the corresponding slice matrices  $\mathbf{Z}_k(i_k), k = 1,\dots ,d$  from each latent core tensor  $\pmb{z}_{k}$  based on the noisy gradient estimates by scaling up just one of local gradients, i.e.  $\forall k = 1,\dots ,d$ ,

$$
\frac {\partial L}{\partial \mathbf {Z} _ {k} \left(i _ {k}\right)} = - \left\{T \left(i _ {1}, i _ {2}, \dots , i _ {d}\right) - \operatorname {T r} \left(\prod_ {k = 1} ^ {d} \mathbf {Z} _ {k} \left(i _ {k}\right)\right) \right\} \left(\prod_ {j = 1, j \neq k} ^ {d} \mathbf {Z} _ {j} \left(i _ {j}\right)\right) ^ {T} + \lambda_ {k} \mathbf {Z} _ {k} \left(i _ {k}\right), \tag {8}
$$

We employ Adaptive Moment Estimation (Adam) method to compute adaptive learning rates for each parameter. Thus, the update rule for each core tensor is given by

$$
\mathbf {Z} _ {k} \left(i _ {k}\right) ^ {t} = \mathbf {Z} _ {k} ^ {t - 1} \left(i _ {k}\right) - \frac {\eta}{\sqrt {\mathbf {V} _ {t}} + \epsilon} \mathbf {M} _ {t} - \lambda_ {k} \mathbf {Z} _ {k} ^ {t - 1} \left(i _ {k}\right), \quad \forall k = 1, \dots , d, \tag {9}
$$

where  $\mathbf{M}_t = \beta_1\mathbf{M}_{t - 1} + (1 - \beta_1)\frac{\partial L}{\partial\mathbf{Z}_k^t(i_k)}$  denotes an exponentially decaying average of past gradients and  $\mathbf{V}_t = \beta_2\mathbf{V}_{t - 1} + (1 - \beta_2)(\frac{\partial L}{\partial\mathbf{Z}_k^t(i_k)})^2$  denotes exponentially decaying average of second moment of the gradients.

The SGD algorithm can be naturally applied to tensor completion problem, when the data points are sampled only from a sparse tensor. Furthermore, this also naturally gives an online TR decomposition. The batched versions, in which multiple local losses are averaged, are also feasible but often have inferior performance in practice. For each element  $T(i_{1},i_{2},\ldots ,i_{d})$ , the computational complexity of SGD is  $\mathcal{O}(r^3)$ . If we define  $N = \prod_{k = 1}^{d}n_{k}$  consecutive updates as one iteration of SGD, the computational complexity per SGD iteration is thus only  $\mathcal{O}(r^3 N)$ , which linearly scales to data size and independent with the order of tensor. As compared to ALS, which needs  $\mathcal{O}(Ndr^4 +dr^6)$ , it is more efficient in terms of computational complexity for one iteration. The convergence condition of SGD algorithm follows other stochastic tensor decompositions (Ge et al., 2015; Maehara et al., 2016).

# 4 PROPERTIES OF TR REPRESENTATION

By assuming that tensor data have been already represented as TR decompositions, i.e., a sequence of third-order cores, we justify and demonstrate that the basic operations on tensors, such as the addition, multilinear product, Hadamard product, inner product and Frobenius norm, can be performed efficiently by the appropriate operations on each individual cores. We have the following theorems:

Theorem 3. Let  $\mathcal{T}_1$  and  $\mathcal{T}_2$  be dth-order tensors of size  $n_1 \times \dots \times n_d$ . If TR decompositions of these two tensors are  $\mathcal{T}_1 = \Re(\mathcal{Z}_1, \ldots, \mathcal{Z}_d)$  where  $\mathcal{Z}_k \in \mathbb{R}^{r_k \times n_k \times r_{k+1}}$  and  $\mathcal{T}_2 = \Re(\mathcal{Y}_1, \ldots, \mathcal{Y}_d)$  where  $\mathcal{Y}_k \in \mathbb{R}^{s_k \times n_k \times s_{k+1}}$ , then the addition of these two tensors,  $\mathcal{T}_3 = \mathcal{T}_1 + \mathcal{T}_2$ , can also be represented in the TR format given by  $\mathcal{T}_3 = \Re(\mathcal{X}_1, \ldots, \mathcal{X}_d)$ , where  $\mathcal{X}_k \in \mathbb{R}^{q_k \times n_k \times q_{k+1}}$  and  $q_k = r_k + s_k$ . Each core  $\mathcal{X}_k$  can be computed by

$$
\mathbf {X} _ {k} \left(i _ {k}\right) = \left( \begin{array}{c c} \mathbf {Z} _ {k} \left(i _ {k}\right) & 0 \\ 0 & \mathbf {Y} _ {k} \left(i _ {k}\right) \end{array} \right), \quad \begin{array}{l} i _ {k} = 1, \dots , n _ {k}, \\ k = 1, \dots , d. \end{array} \tag {10}
$$

A proof of Theorem 3 is provided in Appendix B.2. Note that the sizes of new cores are increased and not optimal in general. This problem can be solved by the rounding procedure (Oseledets, 2011).

Theorem 4. Let  $\mathcal{T} \in \mathbb{R}^{n_1 \times \dots \times n_d}$  be a dth-order tensor whose TR representation is  $\mathcal{T} = \Re(\mathcal{Z}_1, \ldots, \mathcal{Z}_d)$  and  $\mathbf{u}_k \in \mathbb{R}^{n_k}$ ,  $k = 1, \ldots, d$  be a set of vectors, then the multilinear products, denoted by  $c = \mathcal{T} \times_1 \mathbf{u}_1^T \times_2 \dots \times_d \mathbf{u}_d^T$ , can be computed by the multilinear product on each cores, which is

$$
c = \Re \left(\mathbf {X} _ {1}, \dots , \mathbf {X} _ {d}\right) \text {w h e r e} \mathbf {X} _ {k} = \sum_ {i _ {k} = 1} ^ {n _ {k}} \mathbf {Z} _ {k} \left(i _ {k}\right) u _ {k} \left(i _ {k}\right). \tag {11}
$$

A proof of Theorem 4 is provided in Appendix B.3. It should be noted that the computational complexity in the original tensor form is  $\mathcal{O}(dn^d)$ , while it reduces to  $\mathcal{O}(dnr^2 +dr^3)$  that is linear to tensor order  $d$  by using TR representation.

Theorem 5. Let  $\mathcal{T}_1$  and  $\mathcal{T}_2$  be dth-order tensors of size  $n_1\times \dots \times n_d$ . If the TR decompositions of these two tensors are  $\mathcal{T}_1 = \Re (\mathcal{Z}_1,\ldots ,\mathcal{Z}_d)$  where  $\mathcal{Z}_k\in \mathbb{R}^{r_k\times n_k\times r_{k + 1}}$  and  $\mathcal{T}_2 = \Re (\mathcal{Y}_1,\ldots ,\mathcal{Y}_d)$  where  $\mathcal{V}_k\in \mathbb{R}^{s_k\times n_k\times s_{k + 1}}$ , then the Hadamard product of these two tensors,  $\mathcal{T}_3 = \mathcal{T}_1*\mathcal{T}_2$ , can also be represented in the TR format given by  $\mathcal{T}_3 = \Re (\mathcal{X}_1,\ldots ,\mathcal{X}_d)$ , where  $\mathcal{X}_k\in \mathbb{R}^{q_k\times n_k\times q_{k + 1}}$  and  $q_{k} = r_{k}s_{k}$ . Each core  $\mathcal{X}_k$  can be computed by

$$
\mathbf {X} _ {k} \left(i _ {k}\right) = \mathbf {Z} _ {k} \left(i _ {k}\right) \otimes \mathbf {Y} _ {k} \left(i _ {k}\right), \quad k = 1, \dots , d. \tag {12}
$$

![](images/fc8af8f2a53281e1659dc253821ace4a529a71b6baf2489e2c65e3da2c2f857b.jpg)  
Figure 3: Highly oscillated functions. The left panel is  $f_{1}(x) = (x + 1)\sin (100(x + 1)^{2})$ . The middle panel is Airy function:  $f_{2}(x) = x^{-\frac{1}{4}}\sin \left(\frac{2}{3} x^{\frac{3}{2}}\right)$ . The right panel is Chirp function  $f_{3}(x) = \sin \frac{x}{4}\cos (x^{2})$ .

A proof of Theorem 5 is provided in Appendix B.4. Furthermore, one can compute the inner product of two tensors in TR representations. For two tensors  $\mathcal{T}_1$  and  $\mathcal{T}_2$ , it is defined as  $\langle \mathcal{T}_1,\mathcal{T}_2\rangle = \sum_{i_1,\ldots ,i_d}T_3(i_1,\ldots ,i_d)$ , where  $\mathcal{T}_3 = \mathcal{T}_1\circledast \mathcal{T}_2$ . Thus, the inner product can be computed by applying the Hadamard product and then computing the multilinear product between  $\mathcal{T}_3$  and vectors of all ones, i.e.,  $\mathbf{u}_k = \mathbf{1},k = 1,\dots ,d$ . In contrast to  $\mathcal{O}(n^d)$  in the original tensor form, the computational complexity is equal to  $\mathcal{O}(dnq^2 +dq^3)$  that is linear to  $d$  by using TR representation. Similarly, we can also compute the Frobenius norm  $\| \pmb {\tau}\| _F = \sqrt{\langle\pmb{\tau},\pmb{\tau}\rangle}$  in the TR representation.

# 5 EXPERIMENTAL RESULTS

# 5.1 NUMERICAL ILLUSTRATION

We consider highly oscillating functions that can be approximated perfectly by a low-rank TT format (Khoromskij, 2015), as shown in Fig. 3. We firstly tensorize the functional vector resulting in a  $d$ th-order tensor of size  $n_1 \times n_2 \times \dots \times n_d$ , where isometric size is usually preferred, i.e.,  $n_1 = n_2 = \dots = n_d = n$ , with the total number of elements denoted by  $N = n^d$ . The error bound (tolerance), denoted by  $\epsilon_p = 10^{-3}$ , is given as the stopping criterion for all compared algorithms. As shown in Table 1, TR-SVD and TR-BALS can obtain comparable results with TT-SVD in terms of compression ability. However, when noise is involved, TR model significantly outperforms TT model, indicating its more robustness to noises.

Table 1: The functional data  $f_{1}(x), f_{2}(x), f_{3}(x)$  is tensorized to 10th-order tensor  $(4 \times 4 \times \dots \times 4)$ . In the table,  $\epsilon, \bar{r}, N_{p}$  denote relative error, average rank, and the total number of parameters, respectively.  

<table><tr><td rowspan="2"></td><td colspan="4">f1(x)</td><td colspan="4">f2(x)</td><td colspan="4">f3(x)</td><td colspan="4">f1(x) + N(0,σ), SNR = 60dB</td></tr><tr><td>ε</td><td>r</td><td>Np</td><td>Time (s)</td><td>ε</td><td>r</td><td>Np</td><td>Time (s)</td><td>ε</td><td>r</td><td>Np</td><td>Time (s)</td><td>ε</td><td>r</td><td>Np</td><td>Time (s)</td></tr><tr><td>TT-SVD</td><td>3e-4</td><td>4.4</td><td>1032</td><td>0.17</td><td>3e-4</td><td>5</td><td>1360</td><td>0.16</td><td>3e-4</td><td>3.7</td><td>680</td><td>0.16</td><td>1e-3</td><td>16.6</td><td>13064</td><td>0.5</td></tr><tr><td>TR-SVD</td><td>3e-4</td><td>4.4</td><td>1032</td><td>0.17</td><td>3e-4</td><td>5</td><td>1360</td><td>0.28</td><td>5e-4</td><td>3.6</td><td>668</td><td>0.15</td><td>1e-3</td><td>9.7</td><td>4644</td><td>0.4</td></tr><tr><td>TR-ALS</td><td>3e-4</td><td>4.4</td><td>1032</td><td>13.2</td><td>3e-4</td><td>5</td><td>1360</td><td>18.6</td><td>8e-4</td><td>3.6</td><td>668</td><td>4.0</td><td>1e-3</td><td>4.4</td><td>1032</td><td>11.8</td></tr><tr><td>TR-BALS</td><td>9e-4</td><td>4.3</td><td>1052</td><td>4.6</td><td>8e-4</td><td>4.9</td><td>1324</td><td>5.7</td><td>5e-4</td><td>3.7</td><td>728</td><td>3.4</td><td>1e-3</td><td>4.2</td><td>1000</td><td>6.1</td></tr></table>

We also tested TR-ALS and TR-SGD algorithms on datasets which are generated by a TR model, in which the core tensors are randomly drawn from  $\mathcal{N}(0,1)$ . As shown in Table 2, TR-SGD can achieve similar performance as TR-ALS in all cases. In particular, when data is relatively large-scale ( $10^{8}$ ), TR-SGD can achieve relative error  $\epsilon = 0.01$  by using  $1\%$  of data points only once.

Table 2: Results on synthetic data with fixed ranks  $r_1 = r_2 = \dots = 2$  

<table><tr><td>Tensor size</td><td>TR-ALS</td><td>TR-SGD</td></tr><tr><td>n = 10, d = 4</td><td>(ε = 0.01, Iteration = 19)</td><td>(ε = 0.01, Iteration = 10 )</td></tr><tr><td>n = 10, d = 6</td><td>(ε = 0.01, Iteration = 10)</td><td>(ε = 0.01, Iteration = 0.4 )</td></tr><tr><td>n = 10, d = 8</td><td>(ε = 0.05, Iteration = 9)</td><td>(ε = 0.01, Iteration = 0.01 )</td></tr></table>

# 5.2 IMAGE REPRESENTATION BY HIGHER-ORDER TENSOR DECOMPOSITIONS

An image is naturally represented by a 2D matrix, on which SVD can provide the best low-rank approximation. However, the intrinsic structure and high-order correlations within the image is not well exploited by SVD. In this section, we show the tensorization of an image, yielding a higher-order

![](images/0e9798a7485e674cc8e8fb807d9634d527282e60137f3ee2ff504a2ea53fe655.jpg)  
Figure 4: TR-SGD decomposition with TR-ranks of 12 on the 8th-order tensorization of an image. Iter:  $50\%$  indicates that only  $50\%$  elements are sampled for learning its TR representation. RSE indicates root relative square error  $\| \hat{\mathcal{V}} -\mathcal{V}\| _F / \| \mathcal{V}\| _F$ .

tensor, and TR decomposition enable us to represent the image more efficiently than SVD. Given an image (e.g. 'Peppers') denoted by  $\mathcal{Y}$  of size  $I\times J$ , we can reshape it as  $I_{1}\times I_{2}\times \ldots \times I_{d}\times$ $J_{1}\times J_{2}\times \ldots \times J_{d}$  followed by an appropriate permutation to  $I_{1}\times J_{1}\times I_{2}\times J_{2}\ldots \times I_{d}\times J_{d}$  and thus reshape it again to  $I_1J_1\times I_2J_2\times \ldots \times I_dJ_d$ , which is a dth-order tensor. The first mode corresponds to small-scale patches of size  $I_{1}\times J_{1}$ , while the dth-mode corresponds to large-scale partition of whole image as  $I_d\times J_d$ . Based on this tensorization operations, TR decomposition is able to capture the intrinsic structure information and provides a more compact representation. As shown in Table 3, for 2D matrix case, SVD, TT and TR give exactly same results. In contrast, for 4th-order tensorization cases, TT needs only half number of parameters (2 times compression rate) while TR achieves 3 times compression rate, given the same approximation error 0.1. It should be noted that TR representation provides significantly high compression ability as compared to TT. In addition, Fig. 4 shows TR-SGD results on 'Lena' image by sampling different fraction of data points.

Table 3: Image representation by using tensorization and TR decomposition. The number of parameters is compared for SVD, TT and TR given the same approximation errors.  

<table><tr><td>Data</td><td colspan="2">ε = 0.1</td><td colspan="2">ε = 0.01</td><td colspan="2">ε = 9e-4</td><td colspan="2">ε = 2e - 15</td></tr><tr><td rowspan="2">n = 256, d = 2</td><td>SVD</td><td>TT/TR</td><td>SVD</td><td>TT/TR</td><td>SVD</td><td>TT/TR</td><td>SVD</td><td>TT/TR</td></tr><tr><td>9.7e3</td><td>9.7e3</td><td>7.2e4</td><td>7.2e4</td><td>1.2e5</td><td>1.2e5</td><td>1.3e5</td><td>1.3e5</td></tr><tr><td rowspan="2">Tensorization</td><td colspan="2">ε = 0.1</td><td colspan="2">ε = 0.01</td><td colspan="2">ε = 2e - 3</td><td colspan="2">ε = 1e - 14</td></tr><tr><td>TT</td><td>TR</td><td>TT</td><td>TR</td><td>TT</td><td>TR</td><td>TT</td><td>TR</td></tr><tr><td>n = 16, d = 4</td><td>5.1e3</td><td>3.8e3</td><td>6.8e4</td><td>6.4e4</td><td>1.0e5</td><td>7.3e4</td><td>1.3e5</td><td>7.4e4</td></tr><tr><td>n = 4, d = 8</td><td>4.8e3</td><td>4.3e3</td><td>7.8e4</td><td>7.8e4</td><td>1.1e5</td><td>9.8e4</td><td>1.3e5</td><td>1.0e5</td></tr><tr><td>n = 2, d = 16</td><td>7.4e3</td><td>7.4e3</td><td>1.0e5</td><td>1.0e5</td><td>1.5e5</td><td>1.5e5</td><td>1.7e5</td><td>1.7e5</td></tr></table>

# 5.3 CIFAR-10

The CIFAR-10 dataset consists of  $60000 \times 32$  colour images. We randomly pick up 1000 images for testing of TR decomposition algorithms. As shown in Table 4, TR-SVD outperforms TT-SVD in terms of compression rate given the same approximation error, which is caused by strict limitation that the mode-1 rank must be 1 for TT model. In addition, TR is a more generalized model, which contains TT as a special case, thus yielding better low-rank approximation. Moreover, all other TR algorithms can also achieve similar results. Note that TR-SGD can achieve the same performance as TR-ALS, which demonstrates its effectiveness on real-world dataset. Due to high computational efficiency of TR-SGD per iteration, it can be potentially applied to very large-scale dataset. For visualization, TR-SGD results after 10 and 100 iterations are shown in Fig. 5.

# 6 CONCLUSION

We have proposed a novel tensor decomposition model, which provides an efficient representation for a very high-order tensor by a sequence of low-dimensional cores. The number of parameters in our model scales only linearly to the tensor order. To optimize the latent cores, we have presented several different algorithms: TR-SVD is a non-recursive algorithm that is stable and efficient, while TR-BALS can learn a more compact representation with adaptive TR-ranks, TR-SGD is a scalable algorithm which can be also used for tensor completion and online learning. Furthermore, we have

Table 4: Results on CIFAR-10 images.  

<table><tr><td></td><td>€</td><td>Ranks</td><td>Np</td><td>Iterations</td></tr><tr><td>TT-SVD</td><td>0.092</td><td>(177967)</td><td>66099</td><td>NaN</td></tr><tr><td>TR-SVD</td><td>0.095</td><td>(5,3,49,58)</td><td>42710</td><td>NaN</td></tr><tr><td>TR-BALS</td><td>0.094</td><td>(61,13,3,6)</td><td>63278</td><td>23</td></tr><tr><td>TR-ALS</td><td>0.1076</td><td>(5,3,49,58)</td><td>42710</td><td>10</td></tr><tr><td>TR-SGD</td><td>0.1041</td><td>(5,3,49,58)</td><td>42710</td><td>100</td></tr></table>

![](images/619475fded0d83d750d6ef328f6db3fa20479046371a3de87a78e7db4f29691b.jpg)  
(a)  $\mathrm{RSE} = 0.18$  ,Iter  $= 10$

![](images/6200448fe90c77c006d2e5ab1c113072577d1636b7b489337a11fa46a6ebe59b.jpg)  
(b)  $\mathrm{RSE} = 0.10$  ,Iter  $= 100$  
Figure 5: The reconstructed images by using TR-SGD after 10 and 100 iterations.

investigated the properties on how the basic multilinear algebra can be performed efficiently by operations over TR representations (i.e., cores), which provides a powerful framework for processing large-scale data. The experimental results verified the effectiveness of our proposed algorithms.

# REFERENCES

Animashree Anandkumar, Rong Ge, Daniel J Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. Journal of Machine Learning Research, 15(1): 2773-2832, 2014.  
J. A. Bengua, H. N. Phien, and H. D. Tuan. Optimal feature extraction and classification of tensors via matrix product state decomposition. In 2015 IEEE International Congress on Big Data, pp. 669-672, June 2015. doi: 10.1109/BigDataCongress.2015.105.  
Andrzej Cichocki, Namgil Lee, Ivan Oseledets, Anh-Huy Phan, Qibin Zhao, Danilo P Mandic, et al. Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions. Foundations and Trends® in Machine Learning, 9(4-5):249-429, 2016.  
L. De Lathauwer, B. De Moor, and J. Vandewalle. On the best rank-1 and rank-(R1,R2,...,RN) approximation of higher-order tensors. SIAM J. Matrix Anal. Appl., 21:1324-1342, 2000. ISSN 0895-4798.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points? online stochastic gradient for tensor decomposition. In Conference on Learning Theory, pp. 797-842, 2015.  
Rainer Gemulla, Erik Nijkamp, Peter J Haas, and Yannis Sismanis. Large-scale matrix factorization with distributed stochastic gradient descent. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 69-77. ACM, 2011.  
J. Goulart, M. Boizard, R. Boyer, G. Favier, and P. Comon. Tensor cp decomposition with structured factor matrices: Algorithms and performance. IEEE Journal of Selected Topics in Signal Processing, 2015.  
S. Holtz, T. Rohwedder, and R. Schneider. The alternating linear scheme for tensor optimization in the tensor train format. SIAM J. Scientific Computing, 34(2), 2012.

Heishiro Kanagawa, Taiji Suzuki, Hayato Kobayashi, Nobuyuki Shimizu, and Yukihiro Tagami. Gaussian process nonparametric tensor estimator and its minimax optimality. In International Conference on Machine Learning (ICML2016), pp. 1632-1641, 2016.  
Boris N Khoromskij. Tensor numerical methods for multidimensional PDEs: theoretical analysis and initial applications. *ESAIM: Proceedings and Surveys*, 48:1-28, 2015.  
T.G. Kolda and B.W. Bader. Tensor decompositions and applications. SIAM Review, 51(3):455-500, 2009.  
Ivan Laptev and Tony Lindeberg. Local descriptors for spatio-temporal recognition. In Spatial Coherence for Visual Motion Analysis, pp. 91-103. Springer, 2006.  
Takanori Maehara, Kohei Hayashi, and Ken-ichi Kawarabayashi. Expected tensor decomposition with stochastic gradient descent. In AAAI, pp. 1919-1925, 2016.  
S Nayar, S Nene, and Hiroshi Murase. Columbia object image library (coil 100). Department of Comp. Science, Columbia University, Tech. Rep. CUCS-006-96, 1996.  
Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. In Advances in Neural Information Processing Systems, pp. 442-450, 2015.  
Ivan V Oseledets. Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5): 2295-2317, 2011.  
Bernardino Romera-Paredes, Hane Aung, Nadia Bianchi-Berthouze, and Massimiliano Pontil. Multilinear multitask learning. In International Conference on Machine Learning, pp. 1444-1452, 2013.  
Edwin Stoudenmire and David J Schwab. Supervised learning with tensor networks. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems 29, pp. 4799-4807. Curran Associates, Inc., 2016. URL http://papers.nips.cc/paper/6211-supervised-learning-with-tensor-networks.pdf.  
Chuan-Yung Tsai, Andrew M Saxe, and David Cox. Tensor switching networks. In Advances in Neural Information Processing Systems, pp. 2038-2046, 2016.  
Yining Wang and Anima Anandkumar. Online and differentially-private tensor decomposition. In Advances in Neural Information Processing Systems, pp. 3531-3539, 2016.  
Qiang Wu, Liqing Zhang, and Andrzej Cichocki. Multifactor sparse feature extraction using convolutional nonnegative tucker decomposition. Neurocomputing, 129:17-24, 2014.  
Zenglin Xu, Feng Yan, and Alan Qi. Infinite Tucker decomposition: Nonparametric Bayesian models for multiway data analysis. In Proceedings of the 29th International Conference on Machine Learning (ICML-12), pp. 1023-1030, 2012.  
Yinchong Yang, Denis Krompass, and Volker Tresp. Tensor-train recurrent neural networks for video classification. arXiv preprint arXiv:1707.01786, 2017.  
Rose Yu and Yan Liu. Learning from multiway data: Simple and efficient tensor regression. In International Conference on Machine Learning, pp. 373-381, 2016.  
Shandian Zhe, Kai Zhang, Pengyuan Wang, Kuang-chih Lee, Zenglin Xu, Yuan Qi, and Zoubin Ghahramani. Distributed flexible nonlinear tensor factorization. In Advances in Neural Information Processing Systems, pp. 928-936, 2016.  
G. Zhou, Q. Zhao, Y. Zhang, T. Adali, S. Xie, and A. Cichocki. Linked component analysis from matrices to high-order tensors: Applications to biomedical data. Proceedings of the IEEE, 104(2): 310-331, 2016.
