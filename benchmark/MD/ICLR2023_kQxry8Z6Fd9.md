# STATISTICAL GUARANTEES FOR CONSENSUS CLUSTERING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Consider the problem of clustering  $n$  objects. One can apply multiple algorithms to produce  $N$  potentially different clusterings of the same objects, that is, partitions of the  $n$  objects into  $K$  groups. Even a single randomized algorithm can output different clusterings. This often happens when one samples from the posterior of a Bayesian model, or runs multiple MCMC chains from random initializations. A natural task is then to form a consensus among these different clusterings. The challenge in an unsupervised setting is that the optimal matching between clusters of different inputs is unknown. We model this problem as finding a barycenter (also known as Fréchet mean) relative to the misclassification rate. We show that by lifting the problem to the space of association matrices, one can derive aggregation algorithms that circumvent the knowledge of the optimal matchings. We analyze the statistical performance of aggregation algorithms under a stochastic label perturbation model, and show that a  $K$ -means type algorithm followed by a local refinement step can achieve near optimal performance, with a rate that decays exponentially fast in  $N$ . Numerical experiments show the effectiveness of the proposed methods.

# 1 INTRODUCTION

Clustering is a fundamental task in machine learning and data analysis. Given data on each of the  $n$  objects in a set, there are numerous algorithms to produce a clustering of these  $n$  objects, which is formally a partitioning of  $\{1, \dots, n\}$  into  $K$  disjoint sets. A natural problem that arises in practice is how to form a consensus among these clusterings. This is especially important if the different clusterings are produced by a single randomized algorithm. This situation often arises in Bayesian modeling, where the posterior naturally encodes the variability of the clustering problem. Finding a consensus clustering then corresponds to finding the center of the posterior, from which we can also obtain estimates of the variability of the posterior.

A clustering of  $n$  objects can be viewed as a label vector in  $[K]^n$  where  $[K] = \{1, \dots, K\}$ . We assume that we are given  $N$  label vectors  $z_j \in [K_j]^n$  for  $j = 1, \dots, N$ , with potentially different number of clusters each. Let  $K = \max_j K_j$  and note that we can view all  $z_j$  as vectors in  $[K]^n$ . The task is to obtain a consensus  $K$ -clustering, that is, a label vector  $z \in [K]^n$  which is close to all  $z_1, \dots, z_N$  at the same time. We also refer to this task as the label aggregation problem.

In the context of clustering, there is no meaning to the label of each cluster, that is, the label aggregation problem is unsupervised, in the sense that there is no natural correspondence between labels of different clusterings. This is in contrast to label aggregation in classification in which the labels have a common meaning among different input classifications. We refer to the latter task as supervised label aggregation.

In the unsupervised setting, forming a consensus label is a nontrivial task due the label-switching problem. Consider for example, the case  $n = 5$  and the two label vectors  $z_{1} = (1,1,1,2,2)$  and  $z_{2} = (2,2,2,1,1)$ . These two vectors are different in all 5 positions but they define the same clusterings of the objects. In this case, the consensus label  $\widehat{z}$  can be taken to be either  $z_{1}$  or  $z_{2}$ . More generally, for every  $z_{j}$ , there could be a permutation  $\pi_{j}$  on  $[K]$ , such that the permuted vectors  $\pi_{j} \circ z_{j} := (\pi_{j}(z_{ji}))_{i=1}^{n}$ , are closer to each other than the original  $z_{j}$ s.

To formalize the above idea, we recall the definition of the misclassification rate between two label vectors,  $z, y \in [K]^n$ :

$$
\operatorname {M i s} (z, y) = \min  _ {\pi} \frac {1}{n} \sum_ {i = 1} ^ {n} 1 \left\{z _ {i} \neq \pi \left(y _ {i}\right) \right\} \tag {1}
$$

where the minimum is taken over all the permutations  $\pi : [K] \to [K]$ .  $\mathrm{Mis}(\cdot, \cdot)$  is a proper metric on the space of  $K$ -clusterings of  $n$  objects. It is also a metric on  $[K]^n$  if we identify vectors that are obtained from each other by label-switching. We can now define the consensus label as the barycenter of  $z_1, \ldots, z_N$  in  $\mathrm{Mis}(\cdot, \cdot)$  metric, that is,

$$
\widehat {z} \in \operatorname {a r g m i n} _ {z \in [ K ] ^ {n}} \sum_ {j = 1} ^ {N} w _ {j} \operatorname {M i s} (z, z _ {j}) \tag {2}
$$

where  $w_{j} \geq 0$  are a given set of weights. We often assume uniform weights:  $w_{j} = 1$  for all  $j$ . The barycenter  $\widehat{z}$  is also known as the Frechet mean. Solving (2) is complicated by the presence of the permutation in the definition of Mis function. More explicitly, we need to solve

$$
\widehat {z} \in \underset {\pi_ {1}, \dots , \pi_ {N}} {\operatorname {a r g m i n}} \sum_ {j = 1} ^ {N} \sum_ {i = 1} ^ {n} w _ {j} 1 \left\{z _ {i} \neq \pi_ {j} \left(z _ {j i}\right) \right\} \tag {3}
$$

showing that in addition to  $z$ , we have to optimize over  $N$  permutations  $\pi_j, j = 1,\dots ,N$ . In this paper, we provide alternative solutions that avoid optimizing over these permutations.

Our contributions The unsupervised version of the label aggregation problem is the realistic and practical one when dealing with aggregating labels from Bayesian clustering algorithms, since the posterior has  $K!$  modes corresponding to all possible label permutations, and the output will be near an arbitrary mode in each run of the algorithm. The main contributions of this paper to unsupervised aggregation are the following:

1. We show that by lifting the barycenter problem to the space of association matrices, one can derive algorithms that avoid optimizing over the unknown permutations (Section 2.1). In particular, we propose both a basic and a spectral  $K$ -means type aggregation algorithm.  
2. We propose a random perturbation model (RPM) under which we can study the theoretical performance of both supervised and unsupervised aggregation algorithms. We prove the statistical consistency of the basic aggregation algorithm under RPM (Section 2.2).  
3. Under RPM, the supervised setting corresponds to an oracle that knows the true matching permutations. By studying this oracle, we derive the optimal statistical misclassification rate for supervised aggregation (Section 3.1).  
4. We propose an efficient local refinement step on the output of any consistent aggregation algorithm in the unsupervised setting, and show that the updated labels achieve nearly the same misclassification rate as the above oracle (Section 3.2).

Our theoretical analysis illustrates how different parameters affect the difficulty of the label aggregation problem. In Section 4, we provide numerical experiments comparing the performance of the proposed algorithms against each other and existing methods.

Related work In the supervised setting, the problem of label aggregation is to combine multiple annotated dataset. The label inferred for each item from those produced by multiple annotators acts as the ground truth for the classification task. Various probabilistic models have been proposed for aggregating annotations, with parameters to account for the expertise of the annotators and the noise in the labeling process (47; 37). The unsupervised setting is more challenging as there is no meaning to the cluster labels (the label-switching issue) and the clusterings can have potentially different number of clusters. The idea of passing to association matrices to get around the label-switching issue, has been leveraged in several existing approaches (12; 24; 43; 13; 21; 29), although the connection we make to the lifted barycenter problem and the resulting spectral methods is new to the best of our knowledge. In (24; 43), the authors employ an Expectation-Maximization strategy to obtain a

nonnegative matrix factorization of the combined association matrix. The authors of (41) provide several approaches, using the hypergraph representation of the clusterings, that have shown promising results in the context of image segmentation (22). A set of fuzzy consensus algorithms is proposed in (40) that generate soft consensus partitions by combining a collection of fuzzy clusterings.

What we referred to as unsupervised label aggregation problem has appeared under many different names in the literature, including but not limited to, cluster ensembles (41), clustering/cluster ensemble problem (44; 9), ensemble clustering (1), clustering aggregation (17), combining clusterings (42; 14; 27), consensus clustering (48; 18) and the median partition problem (11; 46; 18). As can be surmised from the variety of names, there is a copious literature on the subject, spanning over multiple fields, with many ideas rediscovered time after time. We refer to the excellent surveys (48; 44; 18) for a more exhaustive list of references and historical discussions. There is also a parallel line of work in the Bayesian clustering literature on aggregating the posterior clusterings (31; 6; 8; 25; 15; 45; 7).

The barycentric view to aggregation that we take in (2) has appeared in many previous work, but often with a different distance in place of Mis, including but not limited to the symmetric difference distance (SDD), a.k.a. the Mirkin metric (up to a constant), in the median partition problem (39; 23; 11; 17; 18), the Binder loss and variational information (VI) in (45; 15), the normalized mutual information (NMI) in the pioneering work of (41), the adjusted Rand index (15) and the category utility function (42; 35). After introducing our methods, in Section 2.3, we give a more detailed comparison with the literature. We choose Mis as the distance in the present work for a better comparison with the oracle problem in Section 3.1. We also show in the supplementary material that consistency in Mis implies the consistency in other distances.

Despite the voluminous literature on the subject, statistical analysis of the methods under a statistical model for the input clusterings has not been undertaken before. This is the gap that we fill in this paper, by providing the first consistency and optimality results under a statistical model (the RPM) for a method of clustering aggregation that we propose. To the best of our knowledge, the question of consistency, let alone optimality, has not been considered for any method of aggregation before. We also shed more light on the relation between the barycenteric approach and those based on association matrices (Section 2.3), and how convex relaxation leading to a spectral method can be used to approximate the median partition. To illustrate the importance of statistical analysis, we also show that a simple common approach to the median partition problem, known as the BestOfK (18), is in general inconsistent under RPM, despite being shown to be a 2-factor approximation of the median partition problem (11). This further highlights the key insights that statistical analysis under a model can provide which is not possible to obtain by CS-type theory on approximation algorithms.

# 2 LIFTED AGGREGATION ALGORITHMS

We start by introducing some notation. Let  $\mathcal{E}_K = \{e_k\}_{k=1}^K$  be the set of standard basis vectors of  $\mathbb{R}^K$ . The elements of  $\mathcal{E}_K$  can be considered one-hot encodings of the labels from  $[K]$ . From now on, instead of encoding labels as element of  $[K]$ , we encode them as element of  $\mathcal{E}_k$ . We can then view labelings of  $n$  objects as elements of the following set

$$
\mathcal {E} _ {K} ^ {n} = \left\{z = \left(z _ {1}, \dots , z _ {n}\right): z _ {i} \in \mathcal {E} _ {K} \forall i \in [ n ] \right\}. \tag {4}
$$

Each  $z_{i}$  is viewed as a  $K\times 1$  vector and each element of  $\mathcal{E}_K^n$  as  $K\times n$  matrices, which we refer to as label matrices. For  $Z\in \mathcal{E}_K$ , permuting the cluster labels is equivalent to pre-multiplication by a  $K\times K$  permutation matrix  $P$ , that is,  $PZ$ .

The label aggregation problem can be restated as follows: Given label matrices  $Z_{1},\ldots ,Z_{N}\in \mathcal{E}_{K}^{n}$  find a consensus label matrix  $Z\in \mathcal{E}_K^n$  , by solving the barycenter problem:

$$
\widehat {Z} \in \underset { \begin{array}{c} Z \in \mathcal {E} _ {K} ^ {n}, \\ P _ {1}, \dots , P _ {N} \end{array} } {\operatorname* {a r g m i n}} \sum_ {j = 1} ^ {N} w _ {j} \| Z - P _ {j} Z _ {j} \| _ {F} ^ {2} \tag {5}
$$

where  $P_{1},\ldots ,P_{N}$  are  $K\times K$  permutation matrices and  $\| X\| _F\coloneqq \left(\sum_{i,j}X_{ij}^2\right)^{1 / 2}$  is the Frobenius norm of matrix  $X$ . One can verify that problem (5) is equivalent to (3). The following result shows that if we know the optimal permutations  $P_{j}\mathrm{s}$ , we can easily find the barycenter  $\widehat{Z}$ :

Proposition 1. Let  $\{\widehat{P}_1,\dots ,\widehat{P}_N\}$  be an optimal set of permutation matrices in (5). Then, the optimal solution  $\widehat{Z}$  of (5) is the columnwise "argmax" of  $\sum_{j}w_{j}\widehat{P}_{j}Z_{j}$

# Algorithm 1 Basic label aggregation algorithm.

1: Form association matrices  $X_{j} = Z_{j}^{T}Z_{j}$  
2: Form the average association matrix  $\bar{X} = \sum_{j=1}^{N} w_j X_j$ .  
3: Perform  $K$ -means on the rows of  $\bar{X}$ .

# Algorithm 2 Spectral label aggregation algorithm.

1: Define the same average association matrix  $\bar{X}$  as in Algorithm 1.  
2: Perform  $K$ -truncated eigendecomposition of  $\bar{X} = U\bar{\Lambda} U^T$ , where  $\Lambda \in \mathbb{R}^{K\times K}$  contains top- $K$  eigenvalues on the diagonal and the columns of  $U \in \mathbb{R}^{n\times K}$  are the corresponding eigenvectors.  
3: Perform  $K$ -means on the rows of  $U$ .

# 2.1 LIFTING TO ASSOCIATION MATRICES

The difficulty in the unsupervised setting is that the optimal permutations  $\{\widehat{P}_j\}$  are unknown. To get around this issue, we lift the barycenter problem to the space of association matrices. For a label matrix  $Z \in \mathcal{E}_K^n$ , we define the corresponding association matrix as  $X = Z^T Z \in \{0,1\}^{n \times n}$ . Note that  $X_{ij} = 1$  iff  $i$  and  $j$  are in the same cluster according to  $Z$ , otherwise  $X_{ij} = 0$ . The advantage of  $X$  is that it is invariant to label switching:  $X = Z^T Z = (PZ)^T PZ$  for any permutation matrix  $P$ . This suggests solving the following lifted barycenter problem instead of (3):

$$
\widehat {X} \in \underset {X \in \mathcal {X} _ {K}} {\operatorname {a r g m i n}} \sum_ {j = 1} ^ {N} w _ {j} \| X - X _ {j} \| _ {F} ^ {2} \tag {6}
$$

where  $X_{j} = Z_{j}^{T}Z_{j}$  and  $\mathcal{X}_K = \{Z^T Z:Z\in \mathcal{E}_K^n\}$ , that is, the set of (binary) association matrices with at most  $K$  clusters.

Semidefinite relaxation Problem (6) is still hard to solve due to the combinatorial nature of  $\mathcal{X}_K$ . A common approach to solving problems over  $\mathcal{X}_K$  is to relax them to a semidefinite program. In particular,  $\mathcal{X}_K$  is inside the doubly nonnegative cone  $\{X:X\succeq 0,X\geq 0\}$ , where  $X\succeq 0$  means that  $X$  is positive semidefinite and  $X\geq 0$  means that it is elementwise nonnegative. We also note that  $X_{ii} = 1$  for all  $i$ . This suggests relaxing to the following problem

$$
\hat {X} \in \operatorname {a r g m i n} \left\{\sum_ {j = 1} ^ {N} w _ {j} \| X - X _ {j} \| _ {F} ^ {2}: X \succeq 0, X \geq 0, X _ {i i} = 1 \forall i \right\}. \tag {7}
$$

Problem (7) has a simple solution. The solution of the unconstrained version of (7) over  $\mathbb{R}^{n \times n}$  is  $X' := \sum_{j=1}^{n} w_j X_j / \sum_{j=1}^{n} w_j$ . Since  $\{X_j\}$  belong to the constraint set of (7) and this set is convex,  $X'$  too belongs to the constraint set. Hence,  $X'$  is the solution of (7), that is,  $\widehat{X} = X'$ . It remains to translate  $\widehat{X}$  back to labels, for which we can preform rowwise  $K$ -means, leading to Algorithm 1.

Since elements of  $\mathcal{X}_K$  are of rank at most  $K$ , to get a solution which is closer to that of the lifted barycenter problem (6), we can perform a spectral trunction of  $\widehat{X}$  to its  $K$  top eigenvectors, before applying the rowwise  $K$ -means. This leads to the spectral aggregation Algorithm 1. Other variants of spectral clustering on  $\widehat{X}$  are also possible, e.g., using the normalized Laplacian, etc. In the  $K$ -means step of Algorithm 1 and 2, any constant-factor approximation to the  $K$ -means problem can be used.

# 2.2 CONSISTENCY

In order to study the statistical performance of different aggregation algorithms, we propose a random perturbation model (RPM), where both the clusters and the labels of the clusters can undergo random perturbations, allowing us to study the difficulty of the unsupervised aggregation problem. Let  $Z^{*} \in \mathcal{E}_{K}^{n}$  be the "true" label matrix with columns  $z_{i}^{*} \in \mathcal{E}_{K}, i = 1, \dots, n$ .

Definition 1 (RPM). We write  $Z \sim \mathcal{L}(Z^{*}, p)$  if  $Z = (z_{1}, \ldots, z_{n}) \in \mathcal{E}_{K}^{n}$  with columns  $z_{i}$  drawn independently as follows:

$$
z _ {i} = P z _ {i} ^ {\prime}, \quad z _ {i} ^ {\prime} \sim (1 - p) \delta_ {z _ {i} ^ {*}} + p \operatorname {U n i f} \left(\mathcal {E} _ {K}\right) \tag {8}
$$

where  $\delta_{z_i^*}$  is the point mass at  $z_i^*$  and  $P$  is an independent  $K\times K$  permutation matrix.

Under RPM, the observed label matrices  $Z_{1}, \ldots, Z_{N}$  are i.i.d., so it is reasonable to let  $w_{1} = \dots = w_{N} = 1$  in Algorithms 1 and 2. We will discuss the algorithm for weighted samples in Section 5. Let  $n_{k}(Z^{*})$  be the number of objects in cluster  $k$  according to  $Z^{*}$ . We make the following assumption:

$$
n _ {k} \left(Z ^ {*}\right) \leq \beta n / K, \quad k \in [ K ] \tag {9}
$$

for some  $\beta \in [1, K]$ . Here,  $\beta$  measures how much the true clustering deviates from being balanced. For  $\beta = 1$ , we have  $n_k(Z^*) = n / K$  for all  $k$ , while  $\beta = K$  corresponds to no restriction on the sizes of the true clusters. The following result shows that the basic aggregation algorithm is statistically consistent under the RPM:

Theorem 1 (Consistency). Let  $Z^{*} \in \mathcal{E}_{K}^{n}$  be a label matrix satisfying (9). Assume that  $Z_{1},\ldots ,Z_{N}$  are i.i.d. draws from  $\mathcal{L}(Z^{*},p)$  and let  $\xi \coloneqq p(2 - p)$ . Let Mis be the misclassification rate between the true label matrix  $Z^{*}$ , and the output of Algorithm 1 with  $w_{j} = 1$  for all  $j \in [N]$ . Then, there exists a universal constant  $C > 0$ , such that

$$
\mathbb {E} [ \text {M i s} ] \leq \frac {C \xi \beta^ {2} K}{(1 - \xi) ^ {2}} \left(\frac {2 \beta^ {2}}{N} + \frac {\xi K}{n}\right). \tag {10}
$$

Consistency of Algorithm 1 follows from (10) and the Markov inequality: For any  $\delta > 0$ , we have  $\mathbb{P}(\mathrm{Mis} \geq \delta) \leq \delta^{-1} \mathbb{E}[\mathrm{Mis}] \to 0$  as  $n, N \to \infty$  and  $p$  is bounded away from 1. We note that in this and subsequent results all the parameters, such as  $K$  and  $p$ , are allowed to change as  $n, N \to \infty$ , subject to the conditions of the theorems.

The first term inside the parentheses in (10) is the dominant one. Assume for simplicity that  $\beta \asymp 1$ . If the model has low noise, then  $p$  is small and so is  $\xi$  since  $\xi \asymp p$ . Then, the dominant term is  $O(pK / N)$ , that is, a smaller number of clusters,  $K$ , and a larger sample size,  $N$ , improve the performance. The second term is independent of the sample size, but vanishes at the rate  $O(p^{2}K^{2} / n)$  in the low noise setting, as the number of objects,  $n$ , grows.

# 2.3 LITRATURE COMPARISON

The relation between the metrics on clusterings is discussed in (34; 33). Let  $d_M'(Z, Z_j)$  be the Mirkin metric between clusterings  $Z$  and  $Z_j$ , as in (34, Eqn (6)). As we show in the Supplement, it turns out that  $d_M'(Z, Z_j) = \| X - X_j\|_F^2$ . It is also not hard to see that  $d_M' = 2 \cdot \mathrm{SDD} = 2 \cdot \mathrm{Binder} = \binom{n}{2}(1 - \mathrm{Rand})$  where Binder denotes the Binder loss (4) and Rand, the Rand index (36). It follows that the lifted barycenter (6) that we derived is equivalent to the median partition (11) and the Binder loss barycenter of (45) as well as the Rand barycenter (15). This problem is often solved by greedy search starting from a random initialization (45; 7). Our Algorithm 2 then provides a fast scalable spectral method of obtaining an approximate solution to this ubiquitous problem. A lot of algorithms proposed for consensus clustering operate on the average association matrix  $\bar{X}$ , by treating it as a similarity matrix and performing usual clustering on it, for example, by performing agglomerative clustering (31; 18; 14; 28). Wade and Ghahramani (45) criticize these approaches as being ad-hoc compared to the decision-theoretic approach of finding a barycenter. However, we show  $\bar{X}$  is indeed a solution to the relaxed version (7) of the barycenter (6) which is the same as the Binder barycenter in (45), hence clustering  $\bar{X}$  is effectively solving the same problem with a different method.

A  $k$ -means based aggregation algorithm, called KCC, has been proposed in (48). In our notation, this is equivalent to concatenating  $Z_{j}$ 's row-wise to form an  $NK \times n$  matrix and running  $k$ -means on the columns. This is different from our Algorithm (1) that operates on  $\bar{X}$ . We compare with KCC in Section 4. Unlike KCC, Algorithm (1) comes with a consistency guarantee under our model assumptions. The BestOfK (11) essentially solves (6) by restricting the feasible region to  $\{X_{1}, \ldots, X_{N}\}$ , hence picking the lowest-scoring input clustering. The approach proposed in (6; 8) (see also (15)) is to find the input clustering that minimizes the cost  $X \mapsto \|X - \bar{X}\|_{F}$ . It is not hard to see that the barycenter cost (6) is equal to  $\|X - \bar{X}\|_{F}^{2}$  plus a constant (essentially a bias-variance decomposition; see (26)). Hence, the approach of (6; 8) is equivalent to the BestOfK. As we show in the Supplement, BestOfK is, in general, inconsistent under RPM unless  $N$  grows exponentially in  $n$ , a very strong condition not needed by our algorithms.

The  $K$ -means step 3 in Algorithm (1) can be replaced by other clustering algorithms, e.g. average-linkage clustering, BestOfK and CC pivot algorithm of (18; 11). Besides global clustering algorithms, many authors, including (41; 17; 18; 45; 7) also propose local search, a.k.a. greedy, algorithms which

![](images/35d5325026c08bb0d79c35c84bf30b8fd56fc9a9ec27815100f888a4119d765c.jpg)  
Figure 1: Left: Chernoff divergence  $I$  in equation (11) as a function of  $p$ , the label perturbation probability. Right: How fast  $-(\log P_N) / N$  converges to the Chernoff divergence for  $K = 2$ .

![](images/b16518cd96477f179f33cd7096841eef90ded66277ca5910a4867d81b2bffd0d.jpg)

update one label at a time to minimize the barycenter loss, based on any number of metrics discussed earlier, including the information-theoretic ones (NMI and VI (32)). In (26), after reducing (6) to minimizing  $X \mapsto \|X - \bar{X}\|_F^2$ , they write  $X = Z^T Z$  (in our notation) and then relax  $Z$  to a general nonnegative matrix and solve the problem with nonnegative matrix factorization. Their work has the flavor of our Algorithm 2, although our approach, being based on regular spectral decomposition, is highly robust and scalable.

The RPM, with  $P$  set equal to the identity, is closely related to the artificial data model considered in (18), with the difference that RPM does not potentially create a new cluster after perturbation, and introduces a random permutation of the clusters labels after perturbation (via  $P$ ). In the Supplement, we argue that RPM is a good model of a concentrated posterior, hence consistency under RPM is relevant to Bayesian aggregation problems for which posterior consistency has been shown.

# 3 OPTIMAL RATE

Theorem 1 guarantees an  $O(N^{-1})$  rate of misclassification for Algorithm 1. A natural question is whether we can do better. To answer this question, we first consider what is the best an oracle, with the knowledge of the random permutations in (8), can do. This oracle is effectively solving the supervised version of the problem. We then show that a refinement step allows us to achieve nearly the same as the optimal oracle rate, without knowing the matching permutations.

# 3.1 SUPERVISED ORACLE

Let  $Z'$  be a label matrix with the  $i$ th column  $z_i' \sim (1 - p)\delta_{z_i^*} + p\mathrm{Unif}(\mathcal{E}_K)$ , and let  $Z_1, \ldots, Z_N$  be independent copies of  $Z'$ . We would like to recover the true label matrix  $Z^*$ . This is the oracle version of model (8), since  $Z_j$ s are label matrices without random permutations. In this case, the problem decouples to  $n$  independent label recovery problems. We further simplify the problem to that of deciding between  $z_1^* = e_1$  and  $z_1^* = e_2$ . This problem is equivalent the hypothesis testing:

$$
H _ {0}: \text {M u l t i n o m i a l} (N, (1 - \tilde {p}, q, \dots , q)) \quad \text {v e r s u s} \quad H _ {1}: \text {M u l t i n o m i a l} (N, (q, 1 - \tilde {p}, \dots , q)),
$$

where  $q \coloneqq p / K$  and  $\tilde{p} \coloneqq (K - 1)q \coloneqq p - q$ . A classical result from information theory (5, Theorem 11.9.1) allows us to determine the optimal performance in this case:

Proposition 2. The Bayesian error probability,  $P_N$ , for testing  $H_0$  against  $H_1$ , with positive prior probabilities, is bounded by  $e^{-NI}$ , where

$$
I := - \log (2 \sqrt {(1 - \tilde {p}) q} + (K - 2) q) \tag {11}
$$

is the best achievable error exponent in the sense that  $-\frac{1}{N}\log P_N\to I$  as  $N\to \infty$

The left panel in Figure 1 shows plots of  $I$  as a function of  $p$ , for various  $K$ , and the right panel shows the convergence of the exponent of  $P_N$  for  $K = 2$ . The Bayesian error probability  $P_N$  can be achieved by performing a likelihood ratio test between the two hypotheses. We can generalize this result to testing between  $K$  hypotheses, in which case the Bayesian error probability is bounded by

Algorithm 3 Local Refinement

1: Input: Average association matrix  $\bar{X}$ , initial label matrix  $\tilde{Z}$ .  
2: Output: An updated label matrix  $\widehat{Z}$ .  
3: for  $i = 1$  to  $N$  do  
4: Let  $\bar{X}_i$  be the  $i$ th column of  $\bar{X} = \sum_{j} w_j X_j$ .  
5: Replace the  $i$ th column of  $\tilde{Z}$  by zeros and denote this matrix by  $\tilde{Z}_{-i}$ .  
6: Let  $(n_{1},\ldots ,n_{K})$  be the row sums of  $\widetilde{Z}_{-i}$  
7: Let  $(b_{1},\ldots ,b_{K}) = \widetilde{Z}_{-i}\bar{X}_{i}$  
8: Update the  $i$ th label by  $\arg \max_k (b_k / n_k)$ .  
9: end for

$(K - 1)e^{-NI}$ . The oracle algorithm that achieves this bound is the one that finds the columnwise "argmax" of the average of  $Z_{j}$ s. In light of Proposition 2, the bound in Theorem 1 is far from optimal since it guarantees a linear decay of the error in  $N$ , that is  $O(N^{-1})$ , rather than the exponential decay  $e^{-NI}$ . The question is whether this gap can be filled by a non-oracle algorithm.

# 3.2 LOCAL REFINEMENT

To approach the oracle rate, we propose a fast local refinement on the label of each object, as outlined in Algorithm 3. This refinement can be performed on the output of any reasonable aggregation algorithm. The idea of performing a local refinement to boost the performance of clustering algorithms has been employed in various settings, including community detection in stochastic block models (3; 16) and clustering of sub-Gaussian mixtures (30).

Algorithm 3 requires a good initial label matrix  $\tilde{Z}$ , with a small number of mismatches relative to the true matrix  $Z^{*}$ . The algorithm focuses on the local information of the  $i$ th object. With  $\bar{X}$  the average of the association matrices,  $\bar{X}_{ij}$  is the sample proportion of objects  $i$  and  $j$  appearing in the same cluster. Viewing  $\bar{X}$  as the adjacency matrix of a weighted graph, one can verify that  $b_{k} = \sum_{j\neq i}w_{j}X_{ji}1\{\tilde{Z}_{j} = k\}$  is the weighted number of connections between object  $i$  and objects in cluster  $k$ . The algorithm then normalizes the number of connections by the cluster size  $n_k$ . Thus,  $b_{k} / n_{k}$  estimates the probability that object  $i$  is connected to another object in cluster  $k$ . The higher this probability is, the higher the chance that object  $i$  belongs to cluster  $k$ . The last step in the for loop, updates the  $i$ th label according to these statistics.

It is possible to repeat the local refinement procedure, by feeding its output  $\widehat{Z}$  back as an initial label matrix. Given a good initialization, local refinements usually converge in constant or  $O(\log n)$  number of steps (30).

# 3.3 ACHIEVING THE SUPERVISED RATE WITHOUT SUPERVISION

From the oracle result (Proposition 2), we expect the optimal misclassification rate to be close to  $e^{-NI}$ , which is exponential in  $N$ . The is verified by the next result, showing that a single local refinement step applied to a consistent aggregation algorithm, such as Algorithm 1, can get us nearly to the optimal rate:

Theorem 2. Assume that  $Z_{1},\ldots ,Z_{N}$  are i.i.d. draws from the random perturbation model (8). Let  $\widehat{Z}$  be the output of Algorithm 3 initialized by some, possibly data-dependent,  $\widetilde{Z}$ . Let  $n_{min} = \min_{k\in [K]}n_k(Z^*)$  and assume that

(a)  $n_{\min}p(1 \wedge I) / K \to \infty$  and  $\frac{NI}{\log K} \to \infty$ ,

and there exists  $\delta$  satisfying  $Kn\delta /(n_{\min}pI) = o(1)$  such that one of the followings holds:

(b1)  $\mathbb{P}(\mathrm{Mis}(\widetilde{Z},Z^{*})\leq \delta) = 1 - o(1),$  or (b2)  $\mathbb{E}[\mathrm{Mis}(\widetilde{Z},Z^{*})]\leq \delta .$

Then, for some  $\eta = o(1)$ , the misclassification rate satisfies

$$
\mathbb {P} \left(\operatorname {M i s} (\widehat {Z}, Z ^ {*}) \leq e ^ {- (1 - \eta) N I}\right) = 1 - o (1). \tag {12}
$$

The assumptions of Theorem 2 are mild. Suppose that  $p$  is bounded away from 0 or 1, say  $p \in [0.01, 0.99]$ ,  $K = O(1)$  and the cluster sizes are similar. Then, the assumptions can be simplified to  $n_{\min}, N \to \infty$  and  $\delta = o(1)$ . The theorem is most interesting when  $p \to 1$  and  $K$  is large. In this case,  $I \to 0$  and the first requirement of assumption (a) becomes  $n_{\min}I / K \to \infty$ . This assumption guarantees that the samples provide sufficient information to recover the permutations, although we have not attempted to do so in our algorithm. The second requirement of assumption (a) provides evidence to distinguish the true labels from the other  $K - 1$  labels.

Under the assumptions of Theorem 2, Algorithm 3 initialized with input satisfying (12), will have an output satisfying (b1). Hence, Theorem 2 also guarantees rate-optimality of an iterative Algorithm 3.

# 4 EXPERIMENTAL RESULTS

We now present empirical results comparing the performances of the proposed aggregation algorithms, with additional results provided in the supplemental material (Appendix C). The ground truth label matrix  $Z^{*}$  is generated by randomly assigning each of the  $n$  objects to one of the  $K$  labels. The  $N$  input clusterings  $Z_{j}, j \in [N]$  are generated from model (8). We measure the performance of an algorithm by the adjusted Rand index (ARI) of its output against the ground truth. We consider seven different aggregation algorithms: (1) Our Algorithm 1, referred to as "Basic" in the plots; (2) Our Algorithm 2, referred to as SC; (3) KCC algorithm (48); (4) CC Pivot algorithm (2; 18) with threshold 0.25; (5) Best One Element Move (BOEM) algorithm (10; 18); (6) the EM algorithm of (24); and (7) BestOfK algorithm (11; 18). In addition, we consider variants of algorithms (1)-(5) where we apply our refinement step to their output. This gives us a total of 12 methods. In the plots, the refined version is denoted with a solid line and the original version (without refinement) with a dotted line. We also use the average ARI of the  $N$  input clusterings, denoted by the INPUT, as a baseline.

Balanced setting with varying  $n$  and  $N$ . Figure 2 depicts plots of ARI versus the noise probability  $p$  in model (8), for various methods. The results are averaged over 40 replications. The settings in Figure 2 all correspond to balanced cluster sizes. Generally, our proposed Basic and SC algorithms outperform the EM, KCC, CCPivot and BOEM algorithms, with the failure thresholds occurring at larger values of  $p$  (harder problems). In some settings, the refinement shows some improvement, but in others, the output of the refinement applied to Basic and SC nearly coincides with the original algorithm. This shows that in some settings the original algorithm implicitly performs the refinement itself. We note that increasing  $N$  shifts the failure thresholds to the right as expected, as does the increasing of  $n$ , both consistent with our theory. Note that BestOfK performs no better than INPUT. Moreover, refined versions of KCC and BOEM outperform their original versions.

Unbalanced setting. Figure 3 depicts the results obtained with disproportionate cluster sizes, specifically with  $p_1$  proportion of the objects in one cluster and the rest distributed uniformly to the remaining clusters. Local refinement over Basic and SC performs significantly better as we deal with input clusterings of disproportionate cluster sizes, especially at lower noise probabilities.

# 5 CONCLUSION AND DISCUSSION

In the present paper, we defined the random perturbation model to study the label aggregation problem. We developed a  $K$ -means type algorithm followed by an efficient local refinement step to achieve the optimal misclassification rate under the assumptions of the model. Numerical experiments also show the effectiveness of our proposed methods. Let us also discuss possible avenues for future work.

A single-stage algorithm. Two-stage algorithms have been popular in many clustering problems (3; 30; 16). Numerical experiments show that, in many cases, a  $K$ -means type algorithm performs sufficiently close to the local refinement with good initialization (Section 4). The  $K$ -means type algorithm assigns labels based on the distances between the objects and the centers. This criterion is different from the likelihood ratio test in many cases, and so the output of the algorithm will not achieve the misclassification rate of the oracle problem. It will be a novel improvement if the  $K$ -means type algorithm can be generalized to an EM-type algorithm so that the "distance" between the parameter and the object is defined by the likelihood. Whether such an algorithm exists, and how efficient it is statistically and computationally, are interesting open questions.

A robust algorithm. We observe i.i.d. label matrices from the random perturbation model defined in Definition 1. As long as  $p < 1$ , every label matrix from this model provides the same amount

![](images/46043e5bda5cb18bc3f3725e10c301db45686f199847928e6190495cd358654c.jpg)  
(a)  $n = 100, N = 20, K = 6$

![](images/5c0d4f8dab4e4c6f4d163aa8979e3fced9f9454b179a58d0e9425c9c15df7b04.jpg)  
(b)  $n = 100, N = 200, K = 6$

![](images/2ef1ebcd64825cdcb84c4d3eae95a1c5b74dbbb4ccab9245f426bcfc98c6e4f4.jpg)  
(c)  $n = 500, N = 20, K = 6$

![](images/8e2f1a2b7a4bf75ac045f6e45ee59e930ec59cec309787e2d67e914afedb568f.jpg)  
Figure 2: Performance impact with the increase in  $n$  and  $N$  
(d)  $n = 500, N = 200, K = 6$

![](images/ea1330d3ef6cd16885ce4f95b24f71691a6b00a8eeb9889b60c68a287c99a2e2.jpg)  
(a)  $n = 100, N = 20, K = 6, p_1 = 0.5$

![](images/355187c4ac9ad7d6fec6d10993213c9ca6a57b2960061b10755b0db27be0d4ab.jpg)  
(b)  $n = 100, N = 20, K = 6, p_1 = 0.75$  
Figure 3: Significant improvements due to local refinement in the case of unbalanced cluster sizes.

of information in expectation, so there is no reason to assign different weights to label matrices. However, in practice, the samples may not be i.i.d.

# REFERENCES

[1] Daniel Duarte Abdala, Pakaket Wattuya, and Xiaoyi Jiang. Ensemble clustering via random walker consensus strategy. In 2010 20th International Conference on Pattern Recognition, pages 1433-1436. IEEE, 2010.  
[2] Nir Ailon, Moses Charikar, and Alantha Newman. Aggregating inconsistent information: ranking and clustering. Journal of the ACM (JACM), 55(5):1-27, 2008.  
[3] Arash A Amini, Aiyou Chen, Peter J Bickel, and Elizaveta Levina. Pseudo-likelihood methods for community detection in large sparse networks. The Annals of Statistics, 41(4):2097-2122, 2013.  
[4] David A Binder. Bayesian cluster analysis. Biometrika, 65(1):31-38, 1978.  
[5] Thomas M Cover. Elements of information theory. John Wiley & Sons, 1999.  
[6] David B Dahl. Model-based clustering for expression data via a dirichlet process mixture model. Bayesian inference for gene expression and proteomics, 4:201-218, 2006.  
[7] David B Dahl, Devin J Johnson, and Peter Müller. Search algorithms and loss functions for bayesian clustering. Journal of Computational and Graphical Statistics, pages 1-13, 2022.  
[8] David B Dahl and Michael A Newton. Multiple hypothesis testing by clustering treatment effects. Journal of the American Statistical Association, 102(478):517-526, 2007.  
[9] Xiaoli Zhang Fern and Carla E Brodley. Solving cluster ensemble problems by bipartite graph partitioning. In Proceedings of the twenty-first international conference on Machine learning, page 36, 2004.  
[10] Vladimir Filkov and Steven Skiena. Heterogeneous data integration with the consensus clustering formalism. In International Workshop on Data Integration in the Life Sciences, pages 110-123. Springer, 2004.  
[11] Vladimir Filkov and Steven Skiena. Integrating microarray data by consensus clustering. International Journal on Artificial Intelligence Tools, 13(04):863-880, 2004.  
[12] A.L.N. Fred and A.K. Jain. Data clustering using evidence accumulation. 4:276-280 vol.4, 2002.  
[13] Ana LN Fred and Anil K Jain. Data clustering using evidence accumulation. In Object recognition supported by user interaction for service robots, volume 4, pages 276-280. IEEE, 2002.  
[14] Ana LN Fred and Anil K Jain. Combining multiple clusterings using evidence accumulation. IEEE transactions on pattern analysis and machine intelligence, 27(6):835-850, 2005.  
[15] Arno Fritsch and Katja Ickstadt. Improved criteria for clustering based on the posterior similarity matrix. Bayesian analysis, 4(2):367-391, 2009.  
[16] Chao Gao, Zongming Ma, Anderson Y Zhang, and Harrison H Zhou. Achieving optimal misclassification proportion in stochastic block models. The Journal of Machine Learning Research, 18(1):1980-2024, 2017.  
[17] Aristides Gionis, Heikki Mannila, and Panayiotis Tsaparas. Clustering aggregation. Acm transactions on knowledge discovery from data (tkdd), 1(1):4-es, 2007.  
[18] Andrey Goder and Vladimir Filkov. Consensus clustering algorithms: Comparison and refinement. In 2008 Proceedings of the Tenth Workshop on Algorithm Engineering and Experiments (ALENEX), pages 109-117. SIAM, 2008.  
[19] David Inouye, Pradeep Ravikumar, and Inderjit Dhillon. Square root graphical models: Multivariate generalizations of univariate exponential families that permit positive dependencies. In International conference on machine learning, pages 2445-2453. PMLR, 2016.

[20] David I Inouye, Eunho Yang, Genevera I Allen, and Pradeep Ravikumar. A review of multivariate distributions for count data derived from the poisson distribution. Wiley Interdisciplinary Reviews: Computational Statistics, 9(3):e1398, 2017.  
[21] Paul Kellam, Xiaohui Liu, Nigel Martin, Christine Orengo, Stephen Swift, and Allan Tucker. Comparing, contrasting and combining clusters in viral gene expression data. In Proceedings of 6th workshop on intelligent data analysis in medicine and pharmacology, pages 56-62, 2001.  
[22] Jens Keuchel and Daniel Kuttel. Efficient combination of probabilistic sampling approximations for robust image segmentation. In Joint Pattern Recognition Symposium, pages 41-50. Springer, 2006.  
[23] Ludmila I Kuncheva and Dmitry P Vetrov. The problems of approximation in spaces of relations and qualitative data analysis. Information and Remote Control, 35(11):1424-1431, 2006.  
[24] Tilman Lange and Joachim M Buhmann. Combining partitions by probabilistic label aggregation. In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining, pages 147-156, 2005.  
[25] John W Lau and Peter J Green. Bayesian model-based clustering procedures. Journal of Computational and Graphical Statistics, 16(3):526-558, 2007.  
[26] Tao Li and Chris Ding. Weighted consensus clustering. In Proceedings of the 2008 SIAM International Conference on Data Mining, pages 798-809. SIAM, 2008.  
[27] Tao Li, Mitsunori Ogihara, and Sheng Ma. On combining multiple clusterings: an overview and a new perspective. Applied Intelligence, 33(2):207-219, 2010.  
[28] Yan Li, Jian Yu, Pengwei Hao, and Zhulin Li. Clustering ensembles based on normalized edges. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 664–671. Springer, 2007.  
[29] Yuan Li, Benjamin Rubinstein, and Trevor Cohn. Exploiting worker correlation for label aggregation in crowdsourcing. In International Conference on Machine Learning, pages 3886-3895. PMLR, 2019.  
[30] Yu Lu and Harrison H Zhou. Statistical and computational guarantees of lloyd's algorithm and its variants. arXiv preprint arXiv:1612.02099, 2016.  
[31] Mario Medvedovic and Siva Sivaganesan. Bayesian infinite mixture model based clustering of gene expression profiles. Bioinformatics, 18(9):1194-1206, 2002.  
[32] Marina Meilă. Comparing clusterings by the variation of information. In Learning theory and kernel machines, pages 173-187. Springer, 2003.  
[33] Marina Meilă. Comparing clusterings—an information based distance. Journal of multivariate analysis, 98(5):873-895, 2007.  
[34] Marina Meilă. Comparing clusterings: an axiomatic view. In Proceedings of the 22nd international conference on Machine learning, pages 577-584, 2005.  
[35] Boris Mirkin. Reinterpreting the category utility function. Machine Learning, 45(2):219-228, 2001.  
[36] William M Rand. Objective criteria for the evaluation of clustering methods. Journal of the American Statistical association, 66(336):846-850, 1971.  
[37] Vikas C Raykar, Shipeng Yu, Linda H Zhao, Gerardo Hermosillo Valadez, Charles Florin, Luca Bogoni, and Linda Moy. Learning from crowds. Journal of machine learning research, 11(4), 2010.  
[38] Zahra Razaee and Arash Amini. The potts-ising model for discrete multivariate data. Advances in Neural Information Processing Systems, 33:13727–13737, 2020.

[39] Simon Régnier. Sur quelques aspects mathématiques des problèmes de classification automatique. Mathématiques et Sciences humaines, 82:13-29, 1983.  
[40] Xavier Sevillano, Francesc Alías, and Joan Claudi Socorro. Positional and confidence voting-based consensus functions for fuzzy cluster ensembles. Fuzzy Sets and Systems, 193:1-32, 2012.  
[41] Alexander Strehl and Joydeep Ghosh. Cluster ensembles—a knowledge reuse framework for combining multiple partitions. Journal of machine learning research, 3(Dec):583-617, 2002.  
[42] Alexander Topchy, Anil K Jain, and William Punch. Combining multiple weak clusterings. In Third IEEE international conference on data mining, pages 331-338. IEEE, 2003.  
[43] Alexander Topchy, Anil K Jain, and William Punch. A mixture model for clustering ensembles. In Proceedings of the 2004 SIAM international conference on data mining, pages 379-390. SIAM, 2004.  
[44] Sandro Vega-Pons and José Ruiz-Shulcloper. A survey of clustering ensemble algorithms. International Journal of Pattern Recognition and Artificial Intelligence, 25(03):337-372, 2011.  
[45] Sara Wade and Zoubin Ghahramani. Bayesian cluster analysis: Point estimation and credible balls (with discussion). *Bayesian Analysis*, 13(2):559–626, 2018.  
[46] Yoshiko Wakabayashi. The complexity of computing medians of relations. *Resenhas do Instituto de Matemática e Estatística da Universidade de São Paulo*, 3(3):323–349, 1998.  
[47] Jacob Whitehill, Ting-fan Wu, Jacob Bergsma, Javier Movellan, and Paul Ruvolo. Whose vote should count more: Optimal integration of labels from labelers of unknown expertise. Advances in neural information processing systems, 22, 2009.  
[48] Junjie Wu, Hongfu Liu, Hui Xiong, Jie Cao, and Jian Chen. K-means-based consensus clustering: A unified view. IEEE transactions on knowledge and data engineering, 27(1):155-169, 2014.  
[49] Zhixin Zhou and Arash A Amini. Analysis of spectral clustering algorithms for community detection: the general bipartite setting. The Journal of Machine Learning Research, 20(1):1774-1820, 2019.
