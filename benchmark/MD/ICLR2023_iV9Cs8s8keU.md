# LEARNING THE POSITIONS IN COUNTSKETCH

Anonymous authors

Paper under double-blind review

# ABSTRACT

We consider sketching algorithms which first compress data by multiplication with a random sketch matrix, and then apply the sketch to quickly solve an optimization problem, e.g., low-rank approximation and regression. In the learning-based sketching paradigm proposed by (Indyk et al., 2019), the sketch matrix is found by choosing a random sparse matrix, e.g., CountSketch, and then the values of its nonzero entries are updated by running gradient descent on a training data set. Despite the growing body of work on this paradigm, a noticeable omission is that the locations of the non-zero entries of previous algorithms were fixed, and only their values were learned. In this work, we propose the first learning-based algorithms that also optimize the locations of the non-zero entries. Our first proposed algorithm is based on a greedy algorithm. However, one drawback of the greedy algorithm is its slower training time. We fix this issue and propose approaches for learning a sketching matrix for both low-rank approximation and Hessian approximation for second order optimization. The latter is helpful for a range of constrained optimization problems, such as LASSO and matrix estimation with a nuclear norm constraint. Both approaches achieve good accuracy with a fast running time. Moreover, our experiments suggest that our algorithm can still reduce the error significantly even if we only have a very limited number of training matrices.

# 1 INTRODUCTION

The work of (Indyk et al., 2019) investigated learning-based sketching algorithms for low-rank approximation. A sketching algorithm is a method of constructing approximate solutions for optimization problems via summarizing the data. In particular, linear sketching algorithms compress data by multiplication with a sparse "sketch matrix" and then use just the compressed data to find an approximate solution. Generally, this technique results in much faster or more space-efficient algorithms for a fixed approximation error. The pioneering work of (Indyk et al., 2019) shows it is possible to learn sketch matrices for low-rank approximation (LRA) with better average performance than classical sketches.

In this model, we assume inputs come from an unknown distribution and learn a sketch matrix with strong expected performance over the distribution. This distributional assumption is often realistic – there are many situations where a sketching algorithm is applied to a large batch of related data. For example, genomics researchers might sketch DNA from different individuals, which is known to exhibit strong commonalities. The high-performance computing industry also uses sketching, e.g., researchers at NVIDIA have created standard implementations of sketching algorithms for CUDA, a widely used GPU library. They investigated the (classical) sketched singular value decomposition (SVD), but found that the solutions were not accurate enough across a spectrum of inputs (Chien & Bernabeu, 2019). This is precisely the issue addressed by the learned sketch paradigm where we optimize for "good" average performance across a range of inputs.

While promising results have been shown using previous learned sketching techniques, notable gaps remain. In particular, all previous methods work by initializing the sketching matrix with a random sparse matrix, e.g., each column of the sketching matrix has a single non-zero value chosen at a uniformly random position. Then, the values of the non-zero entries are updated by running gradient descent on a training data set, or via other methods. However, the locations of the non-zero entries are held fixed throughout the entire training process.

Clearly this is sub-optimal. Indeed, suppose the input matrix  $A$  is an  $n \times d$  matrix with first  $d$  rows equal to the  $d \times d$  identity matrix, and remaining rows equal to 0. A random sketching matrix  $S$  with a

single non-zero per column is known to require  $m = \Omega(d^2)$  rows in order for  $S \cdot A$  to preserve the rank of  $A$  (Nelson & Nguyen, 2014); this follows by a birthday paradox argument. On the other hand, it is clear that if  $S$  is a  $d \times n$  matrix with first  $d$  rows equal to the identity matrix, then  $\| S \cdot Ax \|_2 = \| Ax \|_2$  for all vectors  $x$ , and so  $S$  preserves not only the rank of  $A$  but all important spectral properties. A random matrix would be very unlikely to choose the non-zero entries in the first  $d$  columns of  $S$  so perfectly, whereas an algorithm trained to optimize the locations of the non-zero entries would notice and correct for this. This is precisely the gap in our understanding that we seek to fill.

Learned CountSketch Paradigm of Indyk et al. (2019). Throughout the paper, we assume our data  $A \in \mathbb{R}^{n \times d}$  is sampled from an unknown distribution  $\mathcal{D}$ . Specifically, we have a training set  $\mathrm{Tr} = \{A_1, \ldots, A_N\} \in \mathcal{D}$ . The generic form of our optimization problems is  $\min_X f(A, X)$ , where  $A \in \mathbb{R}^{n \times d}$  is the input matrix. For a given optimization problem and a set  $\mathcal{S}$  of sketching matrices, define  $\mathrm{ALG}(S, A)$  to be the output of the classical sketching algorithm resulting from using  $\mathcal{S}$ ; this uses the sketching matrices in  $\mathcal{S}$  to map the given input  $A$  and construct an approximate solution  $\hat{X}$ . We remark that the number of sketches used by an algorithm can vary and in its simplest case,  $\mathcal{S}$  is a single sketch, but in more complicated sketching approaches we may need to apply sketching more than once—hence  $\mathcal{S}$  may also denote a set of more than one sketching matrix.

The learned sketch framework has two parts: (1) offline sketch learning and (2) "online" sketching (i.e., applying the learned sketch and some sketching algorithm to possibly unseen data). In offline sketch learning, the goal is to construct a CountSketch matrix (abbreviated as CS matrix) with the minimum expected error for the problem of interest. Formally, that is,

$$
\arg \min  _ {\mathcal {C S} S} \mathbf {E} _ {A \in \operatorname {T r}} f (A, \mathsf {A L G} (S, A)) - f (A, X ^ {*}) = \arg \min  _ {\mathcal {C S} S} \mathbf {E} _ {A \in \operatorname {T r}} f (A, \mathsf {A L G} (S, A)),
$$

where  $X^{*}$  denotes the optimal solution. Moreover, the minimum is taken over all possible constructions of CS. We remark that when ALG needs more than one CS to be learned (e.g., in the sketching algorithm we consider for LRA), we optimize each CS independently using a surrogate loss function.

In the second part of the learned sketch paradigm, we take the sketch from part one and use it within a sketching algorithm. This learned sketch and sketching algorithm can be applied, again and again, to different inputs. Finally, we augment the sketching algorithm to provide worst-case guarantees when used with learned sketches. The goal is to have good performance on  $A \in \mathcal{D}$  while the worst-case performance on  $A \notin \mathcal{D}$  remains comparable to the guarantees of classical sketches. We remark that the learned matrix  $S$  is trained offline only once using the training data. Hence, no additional computational cost is incurred when solving the optimization problem on the test data.

Our Results. In this work, in addition to learning the values of the non-zero entries, we learn the locations of the non-zero entries. Namely, we propose three algorithms that learn the locations of the non-zero entries in CountSketch. Our first algorithm (Section 4) is based on a greedy search. The empirical result shows that this approach can achieve a good performance. Further, we show that the greedy algorithm is provably beneficial for LRA when inputs follow a certain input distribution (Section G). However, one drawback of the greedy algorithm is its much slower training time. We then fix this issue and propose two specific approaches for optimizing the positions for the sketches for low-rank approximation and second-order optimization, which run much faster than all previous algorithms while achieving better performance.

For low-rank approximation, our approach is based on first sampling a small set of rows based on their ridge leverage scores, assigning each of these sampled rows to a unique hash bucket, and then placing each non-sampled remaining row in the hash bucket containing the sampled row for which it is most similar to, i.e., for which it has the largest dot product with. We also show that the worst-case guarantee of this approach is strictly better than that of the classical Count-Sketch (see Section 5).

For sketch-based second-order optimization where we focus on the case that  $n \gg d$ , we observe that the actual property of the sketch matrix we need is the subspace embedding property. We next optimize this property of the sketch matrix. We provably show that the sketch matrix  $S$  needs fewer rows, with optimized positions of the non-zero entries, when the input matrix  $A$  has a small number of rows with a heavy leverage score. More precisely, while CountSketch takes  $O(d^{2} / (\delta \epsilon^{2}))$  rows with failure probability  $\delta$ , in our construction,  $S$  requires only  $O((d\log (1 / \epsilon) + \log (1 / \delta)) / \epsilon^2)$  rows if  $A$  has at most  $d\log (1 / \epsilon) / \epsilon^2$  rows with leverage score at least  $\epsilon / d$ . This is a quadratic improvement in  $d$  and an exponential improvement in  $\delta$ . In practice, it is not necessary to calculate the

leverage scores. Instead, we show in our experiments that the indices of the rows of heavy leverage score can be learned and the induced  $S$  is still accurate. We also consider a new learning objective, that is, we directly optimize the subspace embedding property of the sketching matrix instead of optimizing the error in the objective function of the optimization problem in hand. This demonstrates a significant advantage over non-learned sketches, and has a fast training time (Section 6).

We show strong empirical results for real-world datasets. For low-rank approximation, our methods reduce the errors by  $70\%$  than classical sketches under the same sketch size, while we reduce the errors by  $30\%$  than previous learning-based sketches. For second-order optimization, we show that the convergence rate can be reduced by  $87\%$  over the non-learned CountSketch for the LASSO problem on a real-world dataset. We also evaluate our approaches in the few-shot learning setting where we only have a limited amount of training data (Indyk et al., 2021). We show our approach reduces the error significantly even if we only have one training matrix (Sections 7 and 8). This approach clearly runs faster than all previous methods.

Additional Related Work. In the last few years, there has been much work on leveraging machine learning techniques to improve classical algorithms. We only mention a few examples here which are based on learned sketches. One related body of work is data-dependent dimensionality reduction, such as an approach for pair-wise/multi-wise similarity preservation for indexing big data (Wang et al., 2017), learned sketching for streaming problems (Indyk et al., 2019; Aamand et al., 2019; Jiang et al., 2020; Cohen et al., 2020; Eden et al., 2021; Indyk et al., 2021), learned algorithms for nearest neighbor search (Dong et al., 2020), and a method for learning linear projections for general applications (Hegde et al., 2015). While we also learn linear embeddings, our embeddings are optimized for the specific application of low rank approximation. In fact, one of our central challenges is that the theory and practice of learned sketches generally needs to be tailored to each application. Our work builds off of the work of (Indyk et al., 2019), which introduced gradient descent optimization for LRA, but a major difference is that we also optimize the locations of the non-zero entries in the sketching matrix. We also extend the learned sketch to second-order optimization methods.

# 2 PRELIMINARIES

Notation. Denote the canonical basis vectors of  $\mathbb{R}^n$  by  $e_1, \ldots, e_n$ . Suppose that  $A$  has singular value decomposition (SVD)  $A = U\Sigma V^\top$ . Define  $[A]_k = U_k\Sigma_kV_k^\top$  to be the optimal rank- $k$  approximation to  $A$ , computed by the truncated SVD. Also, define the Moore-Penrose pseudo-inverse of  $A$  to be  $A^\dagger = V\Sigma^{-1}U^\top$ , where  $\Sigma^{-1}$  is constructed by inverting the non-zero diagonal entries. Let  $\operatorname{row}(A)$  and  $\operatorname{col}(A)$  be the row space and the column space of  $A$ , respectively.

CountSketch. We define  $S_{C} \in \mathbb{R}^{m \times n}$  as a classical CountSketch (abbreviated as CS). It is a sparse matrix with one nonzero entry from  $\{\pm 1\}$  per column. The position and value of this nonzero entry are chosen uniformly at random. CountSketch matrices can be succinctly represented by two vectors. We define  $p \in [m]^n, v \in \mathbb{R}^n$  as the positions and values of the nonzero entries, respectively. Further, we let  $\mathrm{CS}(p, v)$  be the CountSketch constructed from vectors  $p$  and  $v$ .

Below we define the objective function  $f(\cdot, \cdot)$  and a classical sketching algorithm  $\mathrm{ALG}(S, A)$  for each individual problem.

Low-rank approximation (LRA). In LRA, we find a rank- $k$  approximation of our data that minimizes the Frobenius norm of the approximation error. For  $A \in \mathbb{R}^{n \times d}$ ,  $\min_{\mathrm{rank} - k B} f_{\mathrm{LRA}}(A, B) = \min_{\mathrm{rank} - k X} \| A - B \|_F^2$ . Usually, instead of outputting the whole  $B \in \mathbb{R}^{n \times d}$ , the algorithm outputs two factors  $Y \in \mathbb{R}^{n \times k}$  and  $X \in \mathbb{R}^{k \times d}$  such that  $B = YX$  for efficiency.

The authors of Indyk et al. (2019) considered Algorithm 1, which only compresses one side of the input matrix  $A$ . However, in practice often both dimensions of the matrix  $A$  are large. Hence, in this work we consider the following algorithm that compresses both sides of  $A$ . We defer the full explanation of the algorithm to Appendix D.

Constrained regression. Given a vector  $b \in \mathbb{R}^n$ , a matrix  $A \in \mathbb{R}^{n \times d}$  ( $n \gg d$ ) and a convex set  $\mathcal{C}$  we want to find  $x$  to minimize the squared error

$$
\min  _ {x \in \mathcal {C}} f _ {\mathrm {R E G}} ([ A b ], X) = \min  _ {x \in \mathcal {C}} \| A x - b \| _ {2} ^ {2}. \tag {2.1}
$$

Iterative Hessian Sketch. The Iterative Hessian Sketching (IHS) method (Pilanci & Wainwright, 2016) solves the constrained least-squares problem by iteratively performing the update

$$
x _ {t + 1} = \underset {x \in \mathcal {C}} {\arg \min } \left\{\frac {1}{2} \| S _ {t + 1} A (x - x _ {t}) \| _ {2} ^ {2} - \langle A ^ {\top} (b - A x _ {t}), x - x _ {t} \rangle \right\}, \tag {2.2}
$$

where  $S_{t + 1}$  is a sketching matrix. It is not difficult to see that for the unsketched version ( $S_{t + 1}$  is the identity matrix) of (2.2), the optimal solution  $x^{t + 1}$  coincides with the optimal solution to the original constrained regression problem (2.1). The IHS approximates the Hessian  $A^\top A$  by a sketched version ( $S_{t + 1}A$ ) $\top$  ( $S_{t + 1}A$ ) to improve runtime, as  $S_{t + 1}A$  typically has very few rows.

Algorithm 1 Rank- $k$  approximation of a matrix  $A$  using a sketch matrix  $S$  (see (Clarkson & Woodruff, 2009, Sec. 4.1.1))

Input:  $\mathbf{A} \in \mathbb{R}^{n \times d}$ ,  $S \in \mathbb{R}^{m \times n}$

1:  $U, \Sigma, V^{\top} \leftarrow \text{COMPACTSVD}(SA) \quad \triangleright$

$$
\{r = \operatorname {r a n k} (S A), U \in \mathbb {R} ^ {m \times r}, V \in \mathbb {R} ^ {d \times r} \}
$$

2: Return:  $[AV]_k V^\top$

Algorithm 2 Position optimization: Greedy Search

Input:  $f, \mathsf{ALG}, \mathsf{Tr} = \{A_1, \dots, A_N \in \mathbb{R}^{n \times d}\}$ ;

sketch dimension  $m$

1: initialize  $S_{L} = \mathbb{O}^{m\times n}$  
2: for  $i = 1$  to  $n$  do  
3:  $\overline{j}\gets \underset {j\in [m]}{\arg \min}\sum_{A\in \mathrm{Tr}}f(A,\mathsf{ALG}(S_L\pm e_je_i^\top ,A))$  
4:  $S_{L}\gets S_{L}\pm (e_{\overline{j}}e_{i}^{\top})$  
5: end for  
6: return  $p$  for  $S_L = \mathsf{CS}(p, v)$

Learning-Based Algorithms in the Few-Shot Setting. Recently, Indyk et al. (2021) studied learning-based algorithms for LRA in the setting where we have access to limited data or computing resources. We provide a brief explanation of learning-based algorithms in the Few-Shot setting in Appendix C.

Leverage Scores and Ridge Leverage Scores. Given a matrix  $A$ , the leverage score of the  $i$ -th row  $a_{i}$  of  $A$  is defined to be  $\tau_{i} \coloneqq a_{i}(A^{\top}A)^{\dagger}a_{i}^{\top}$ , which is the squared  $\ell_{2}$ -norm of the  $i$ -th row of  $U$ , where  $A = U\Sigma V^{T}$  is the singular value decomposition of  $A$ . Given a regularization parameter  $\lambda$ , the ridge leverage score of the  $i$ -th row  $a_{i}$  of  $A$  is defined to be  $\tau_{i} \coloneqq a_{i}(A^{\top}A + \lambda I)^{\dagger}a_{i}^{\top}$ . Our learning-based algorithms employs the ridge leverage score sampling technique proposed in (Cohen et al., 2017), which shows that sampling proportional to ridge leverage scores gives a good solution to LRA.

# 3 DESCRIPTION OF OUR APPROACH

We describe our contributions to the learning-based sketching paradigm which, as mentioned, is to learn the locations of the non-zero values in the sketch matrix. To learn a CountSketch for the given training data set, we locally optimize the following in two stages:

$$
\min  _ {S} \mathbf {E} _ {A \in \mathcal {D}} [ f (A, \mathrm {A L G} (S, A)) ]. \tag {3.1}
$$

(1) compute the positions of the non-zero entries, then (2) fix the positions and optimize their values.

Stage 1: Optimizing Positions. In Section 4, we provide a greedy search algorithm for this stage, as our starting point. In Section 5 and 6, we provide our specific approaches for optimizing the positions for the sketches for low-rank approximation and second-order optimization.

Stage 2: Optimizing Values. This stage is similar to the approach of Indyk et al. (2019). However, instead of the power method, we use an automatic differentiation package, PyTorch (Paszke et al., 2019), and we pass it our objective

$$
\min  _ {v \in \mathbb {R} ^ {n}} \mathbf {E} _ {A \in \mathcal {D}} [ f (A, \mathrm {A L G} (\mathrm {C S} (p, v), A)) ], \tag {3.2}
$$

implemented as a chain of differentiable operations. It will automatically compute the gradient using the chain rule. We also consider new approaches to optimize the values for LRA (proposed in Indyk et al. (2021), see Appendix C for details) and second-order optimization (proposed in Section 6).

Worst-Cases Guarantees. In Appendix E, we show that both of our approaches for the above two problems can perform no worse than a classical sketching matrix when  $A$  does not follow the distribution  $\mathcal{D}$ . In particular, for LRA, we show that the sketch monotonicity property holds for the time-optimal sketching algorithm for low rank approximation. For second-order optimization, we propose an algorithm which runs in input-sparsity time and can test for and use the better of a random sketch and a learned sketch.

# 4 SKETCH LEARNING: GREEDY SEARCH

When  $S$  is a CountSketch, computing  $SA$  amounts to hashing the  $n$  rows of  $A$  into the  $m \ll n$  rows of  $SA$ . The optimization is a combinatorial optimization problem with an empirical risk minimization (ERM) objective. The naive solution is to compute the objective value of the exponentially many  $(m^n)$  possible placements, but this is clearly intractable. Instead, we iteratively construct a full placement in a greedy fashion. We start with  $S$  as a zero matrix. Then, we iterate through the columns of  $S$  in an order determined by the algorithm, adding a nonzero entry to each. The best position in each column is the one that minimizes Eq. (3.1) if an entry were to be added there. For each column, we evaluate Eq. (3.1)  $\mathcal{O}(m)$  times, once for each prospective half-built sketch.

While this greedy strategy is simple to state, additional tactics are required for each problem to make it more tractable. Usually the objective evaluation (Algorithm 2, line 3) is too slow, so we must leverage our insight into their sketching algorithms to pick a proxy objective. Note that we can reuse these proxies for value optimization, since they may make gradient computation faster too.

Proxy objective for LRA. For the two-sided sketching algorithm, we can assume that the two factors  $X, Y$  has the form  $Y = AR^{\top}\tilde{Y}$  and  $X = \tilde{X}SA$ , where  $S$  and  $R$  are both CS matrices, so we optimize the positions in both  $S$  and  $R$ . We cannot use  $f(A, \mathrm{ALG}(S, R, A))$  as our objective because then we would have to consider combinations of placements between  $S$  and  $R$ . To find a proxy, we note that a prerequisite for good performance is for  $\mathrm{row}(SA)$  and  $\mathrm{col}(AR^{\top})$  to both contain a good rank-  $k$  approximation to  $A$  (see proof of Lemma D.5). Thus, we can decouple the optimization of  $S$  and  $R$ . The proxy objective for  $S$  is  $\left\| [AV]_k V^\top - A \right\|_F^2$  where  $SA = U\Sigma V^\top$ . In this expression,  $\hat{X} = [AV]_k V^\top$  is the best rank-  $k$  approximation to  $A$  in  $\mathrm{row}(SA)$ . The proxy objective for  $R$  is defined analogously.

# Algorithm 3 Position optimization: Inner Product

Input:  $A \in \mathbb{R}^{n \times d}$  induced by  $\operatorname{Tr}$ ; sketch dimension  $m$

1: initialize  $S_{1}, S_{2} = \mathbb{O}^{m \times n}$  
2: Sample a set  $C$  of  $m$  rows using the ridge leverage score sampling technique defined in Section 2, where  $C_p$  is the  $p$ -th sampled row.

3: for  $i = 1$  to  $n$  do  
4:  $p_i, v_i \gets \underset{p \in [m], v \in \{\pm 1\}}{\arg \max} \left\langle \frac{C_p}{\|C_p\|_2}, v \frac{A_i}{\|A_i\|_2} \right\rangle$  
5:  $S_{1}[p_{i},i]\gets v_{i}$  
6: end for  
7: for  $i = 1$  to  $m$  do  
8:  $I_{i}\gets \{j\mid p_{j} = i\}$  
9:  $A^{(i)}\gets$  restriction of  $A$  to rows in  $I_{i}$  
0:  $u_{i}\gets$  the top left singular vector of  $A^{(i)}$  
1:  $S_{1}[i,I_{i}]\gets \bar{u}_{i}^{\top}$  
2: end for  
3: for  $i = 1$  to  $m$  do  
4:  $q_{i}\gets$  index such that  $C_i$  is the  $q_{i}$  -th row of  $A$  
5:  $S_{2}[i,q_{i}]\gets 1$  
6: end for  
7: return  $S_{1}$  or  $[S_{2}]$

In Appendix G, we show that the greedy algorithm is provably beneficial for LRA when inputs follow a certain distribution (in particular, the spiked covariance and the Zipfian distribution). Despite the good empirical performance we present in Section 7, one drawback is its much slower training time. Also, for the iterative sketching method for second-order optimization, it is non-trivial to find a proxy objective because the input of the  $i$ -th iteration depends on the solution to the  $(i - 1)$ -th iteration, for which the greedy approach sometimes does not give a good solution. In the next section, we will propose our specific approach for optimizing the positions of the sketches for low-rank approximation and second-order optimization, both of which achieve a very high accuracy and can finish in a very short amount of time.

# 5 SKETCH LEARNING:

# LOW-RANK APPROXIMATION

Now we present a conceptually new algorithm for similar error bounds as the greedy search approach. A better guarantees than the classical Count-Sketch.

To achieve this, we need a more careful analysis. To provide some intuition, if  $\mathrm{rank}(SA) = k$  and  $SA = U\Sigma V^{\top}$ , then the rank- $k$  approximation cost is exactly  $\left\|AVV^{\top} - A\right\|_F^2$ , the projection cost onto  $\operatorname{col}(V)$ . Minimizing it is equivalent to maximizing the sum of squared projection coefficients:

$$
\underset {S} {\arg \min } \left\| A - A V V ^ {\top} \right\| _ {F} ^ {2} = \underset {S} {\arg \min } \sum_ {i \in [ n ]} (\| A _ {i} \| _ {2} ^ {2} - \sum_ {j \in [ k ]} \langle A _ {i}, v _ {j} \rangle^ {2}) = \underset {S} {\arg \max } \sum_ {i \in [ n ]} \sum_ {j \in [ k ]} \langle A _ {i}, v _ {j} \rangle^ {2}.
$$

As mentioned, computing  $SA$  actually amounts to hashing the  $n$  rows of  $A$  to the  $m$  rows of  $SA$ . Hence, intuitively, if we can put similar rows into the same bucket, we may get a smaller error.

Our algorithm is given in Algorithm 3. Suppose that we want to form the matrix  $S$  with  $m$  rows. At the beginning of the algorithm, we sample  $m$  rows according to the ridge leverage scores of  $A$ . By the property of the ridge leverage score, the subspace spanned by this set of sampled rows contains an approximately optimal solution to the low rank approximation problem. Hence, we map these rows to separate " buckets" of  $SA$ . Then, we need to decide the locations of the remaining rows (i.e., the non-sampled rows). Ideally, we want similar rows to be mapped into the same bucket. To achieve this, we use the  $m$  sampled rows as reference points and assign each (non-sampled) row  $A_{i}$  to the  $p$ -th bucket in  $SA$  if the normalized row  $A_{i}$  and  $C_p$  have the largest inner product (among all possible buckets).

Once the locations of the non-zero entries are fixed, the next step is to determine the values of these entries. We follow the same idea proposed in (Indyk et al., 2021): for each block  $A^{(i)}$ , one natural approach is to choose the unit vector  $s_i \in \mathbb{R}^{|I_i|}$  that preserves as much of the Frobenius norm of  $A^{(i)}$  as possible, i.e., to maximize  $\left\| s_i^\top A^{(i)} \right\|_2^2$ . Hence, we set  $s_i$  to be the top left singular vector of  $A^{(i)}$ . In our experiments, we observe that this step can help the downstream value optimizations performed by SGD achieve smaller error than before.

To obtain a worst-case theoretical guarantee, we note that the row span of the sampled rows  $C_i$  is a good subspace with high probability. Hence, here we set the matrix  $S_2$  to be the sampling matrix that samples the  $C_i$ . The final output of our algorithm is the vertical concatenation of  $S_1$  and  $S_2$ . Here  $S_1$  is good empirically, while  $S_2$  has a worst-case guarantee for any input. Combining Lemma F.2 and the sketch monotonicity for low rank approximation in Section E, we get that  $O(k\log k + k / \epsilon)$  rows is enough for a  $(1 \pm \epsilon)$ -approximation for the input matrix  $A$  induced by Tr, which is better than the  $\Omega(k^2)$  rows required of a non-learned Count-Sketch, even if its non-zero values have been further improved by the previous learning-based algorithms in (Indyk et al., 2019) and (Indyk et al., 2021). As a result, under the assumption of the input data, we may expect that  $S$  will still be good for the test data. We defer the proof to Appendix F.1.

In Appendix A, we shall show that the assumptions we make in Theorem 5.1 are reasonable. We also provide an empirical comparison between Algorithm 3 and some of its variants, as well as some adaptive sketching methods on the training sample. The evaluation result shows that only our algorithm has a significant improvement for the test data, which suggests that both ridge leverage score sampling and row bucketing are essential.

Theorem 5.1. Let  $S \in \mathbb{R}^{2m \times n}$  be given by concatenating the sketching matrices  $S_1, S_2$  computed by Algorithm 3 with input  $A$  induced by  $\operatorname{Tr}$  and let  $B \in \mathbb{R}^{n \times d}$ . Then with probability at least  $1 - \delta$ , we have  $\min_{\mathrm{rank} - k} X: \operatorname{row}(X) \subseteq \operatorname{row}(SB)} \| B - X \|_F^2 \leq (1 + \epsilon) \| B - B_k \|_F^2$  if one of the following holds:

1.  $m = O(\beta \cdot (k\log k + k / \epsilon)),\delta = 0.1$  , and  $\tau_{i}(B)\geq \frac{1}{\beta}\tau_{i}(A)$  for all  $i\in [n]$  
2.  $m = O(k\log k + k / \epsilon),\delta = 0.1 + 1.1\beta$  , and the total variation distance  $d_{\mathrm{tv}}(p,q)\leq \beta$  , where  $p,q$  are sampling probabilities defined as  $p_i = \frac{\tau_i(A)}{\sum_i\tau_i(A)}$  and  $q_{i} = \frac{\tau_{i}(B)}{\sum_{i}\tau_{i}(B)}$

Time Complexity. As mentioned, an advantage of our second approach is that it significantly reduces the training time. We now discuss the training times of different algorithms. For the value-learning algorithms in (Indyk et al., 2019), each iteration requires computing a differentiable SVD to perform gradient descent, hence the runtime is at least  $\Omega(n_{it} \cdot T)$ , where  $n_{it}$  is the number of iterations (usually set  $>500$ ) and  $T$  is the time to compute an SVD. For the greedy algorithm, there are  $m$  choices for each column, hence the runtime is at least  $\Omega(m^n \cdot T)$ . For our second approach, the most complicated step is to compute the ridge leverage scores of  $A$  and then the SVD of each submatrix. Hence, the total runtime is at most  $O(T)$ . We note that the time complexities discussed here are all for training time. There is no additional runtime cost for the test data.

# 6 SKETCH LEARNING: SECOND-ORDER OPTIMIZATION

In this section, we consider optimizing the sketch matrix in the context of second-order methods. The key observation is that for many sketching-based second-order methods, the crucial property of the sketching matrix is the so-called subspace embedding property: for a matrix  $A \in \mathbb{R}^{n \times d}$ , we say a matrix  $S \in \mathbb{R}^{m \times n}$  is a  $(1 \pm \epsilon)$ -subspace embedding for the column space of  $A$  if  $(1 - \epsilon) \| Ax \|_2 \leq \| SAx \|_2 \leq (1 + \epsilon) \| Ax \|_2$  for all  $x \in \mathbb{R}^d$ . For example, consider the iterative Hessian sketch, which performs the update (2.2) to compute  $\{x_t\}_t$ . Pilanci & Wainwright (2016) showed that if

$S_{1},\ldots ,S_{t + 1}$  are  $(1 + O(\rho))$  -subspace embeddings of  $A$  , then  $\| A(x^{t} - x^{*})\|_{2}\leq \rho^{t}\| Ax^{*}\|_{2}$  . Thus, if  $S_{i}$  is a good subspace embedding of  $A$  , we will have a good convergence guarantee. Therefore, unlike the previous work of (Indyk et al., 2019), which treats the training objective in a black-box manner, we shall optimize the subspace embedding property of the matrix  $A$

Optimizing positions. We consider the case that  $A$  has a few rows of large leverage score, as well as access to an oracle which reveals a superset of the indices of such rows. Formally, let  $\tau_{i}(A)$  be the leverage score of the  $i$ -th row of  $A$  and  $I^{*} = \{i : \tau_{i}(A) \geq \nu\}$  be the set of rows with large leverage score. Suppose that a superset  $I \supseteq I^{*}$  is known to the algorithm. In the experiments we train an oracle to predict such rows. We can maintain all rows in  $I$  explicitly and apply a Count-Sketch to the remaining rows, i.e., the rows in  $[n] \setminus I$ . Up to permutation of the rows, we can write

$$
A = \left( \begin{array}{c} A _ {I} \\ A _ {I ^ {c}} \end{array} \right) \quad \text {a n d} \quad S = \left( \begin{array}{c c} I & 0 \\ 0 & S ^ {\prime} \end{array} \right), \tag {6.1}
$$

where  $S'$  is a random Count-Sketch matrix of  $m$  rows. Clearly  $S$  has a single non-zero entry per column. We have the following theorem, whose proof is postponed to Section F.2. Intuitively, the proof for Count-Sketch in (Clarkson & Woodruff, 2017) handles rows of large leverage score and rows of small leverage score separately. The rows of large leverage score are to be perfectly hashed while the rows of small leverage score will concentrate in the sketch by the Hanson-Wright inequality.

Theorem 6.1. Let  $\nu = \epsilon /d$ . Suppose that  $m = O((d / \epsilon^2)(\mathrm{polylog}(1 / \epsilon) + \log (1 / \delta)))$ ,  $\delta \in (0,1 / m]$  and  $d = \Omega ((1 / \epsilon)\mathrm{polylog}(1 / \epsilon)\log^{2}(1 / \delta))$ . Then, there exists a distribution on  $S$  of the form in (6.1) with  $m + |I|$  rows such that  $\operatorname*{Pr}\left\{\forall x\in \operatorname {col}(A),\left|\| Sx\| _2^2 -\| x\| _2^2\right| > \epsilon \| x\| _2^2\right\} \leq \delta$ . In particular, when  $\delta = 1 / m$ , the sketching matrix  $S$  has  $O((d / \epsilon^{2})\mathrm{polylog}(d / \epsilon))$  rows.

Hence, if there happen to be at most  $d\log (1 / \epsilon) / \epsilon^2$  rows of leverage score at least  $\epsilon /d$ , the overall sketch length for embedding  $\operatorname {colsp}(A)$  can be reduced to  $O((d\log (1 / \epsilon) + \log (1 / \delta)) / \epsilon^2)$ , a quadratic improvement in  $d$  and an exponential improvement in  $\delta$  over the original sketch length of  $O(d^{2} / (\epsilon^{2}\delta))$  for Count-Sketch. In the worst case there could be  $O(d^{2} / \epsilon)$  such rows, though empirically we do not observe this. In Section 8, we shall show it is possible to learn the indices of the heavy rows for real-world data.

Optimizing values. When we fix the positions of the non-zero entries, we aim to optimize the values by gradient descent. Rather than the previous black-box way in (Indyk et al., 2019) that minimizes  $\sum_{i}f(A,\mathsf{ALG}(S,A))$ , we propose the following objective loss function for the learning algorithm  $\mathcal{L}(S,\mathcal{A}) = \sum_{A_i\in \mathcal{A}}\| (A_iR_i)^\top A_iR_i - I\| _F$ , over all the training data, where  $R_{i}$  comes from the QR decomposition of  $SA_{i} = Q_{i}R_{i}^{-1}$ . The intuition for this loss function is given by the lemma below, whose proof is deferred to Section F.3.

Lemma 6.2. Suppose that  $\epsilon \in (0,\frac{1}{2})$ ,  $S \in \mathbb{R}^{m \times n}$ ,  $A \in \mathbb{R}^{n \times d}$  of full column rank, and  $SA = QR$  is the QR-decomposition of  $SA$ . If  $\|(AR^{-1})^\top AR^{-1} - I\|_{\mathrm{op}} \leq \epsilon$ , then  $S$  is a  $(1 \pm \epsilon)$ -subspace embedding of  $\operatorname{col}(A)$ .

Lemma 6.2 implies that if the loss function over  $\mathcal{A}_{\mathrm{train}}$  is small and the distribution of  $\mathcal{A}_{\mathrm{test}}$  is similar to  $\mathcal{A}_{\mathrm{train}}$ , it is reasonable to expect that  $S$  is a good subspace embedding of  $\mathcal{A}_{\mathrm{test}}$ . Here we use the Frobenius norm rather than operator norm in the loss function because it will make the optimization problem easier to solve, and our empirical results also show that the performance of the Frobenius norm is better than that of the operator norm.

# 7 EXPERIMENTS: LOW-RANK APPROXIMATION

In this section, we evaluate the empirical performance of our learning-based approach for LRA on three datasets. For each, we fix the sketch size and compare the approximation error  $\| A - X\| _F - \| A - A_k\| _F$  averaged over 10 trials. In order to make position optimization more efficient, we noted for LRA that placing each entry (line 3 in Algorithm 2) involves computing many rank-1 SVD updates. Instead of doing this naively, we use formulas for fast rank-1 SVD updates (Brand, 2006), which improved the runtime significantly. For the greedy method, we used several Nvidia GeForce GTX 1080 Ti machines. For the maximum inner product method, the experiments are conducted on a laptop with a  $1.90\mathrm{GHz}$  CPU and 16GB RAM.

Datasets. For a direct comparison, we use the three datasets from (Indyk et al., 2019): (1, 2) Friends, Logo (image): frames from videos of a scene from the TV show *Friends* and of a logo being painted; (3) Hyper (image): hyperspectral images of outdoor environments. Additional details (data dimension,  $N_{\mathrm{train}}$ , etc.) can be found in Table A.1.

Table 7.4: Runtime (in seconds) of LRA on Logo with  $k = {30},m = {60}$  

<table><tr><td></td><td>Offline learning</td><td>Online solving</td></tr><tr><td>Ours (inner product)</td><td>5</td><td>0.166</td></tr><tr><td>Ours (greedy)</td><td>6300 (1.75h)</td><td>0.172</td></tr><tr><td>IVY19</td><td>193 (3min)</td><td>0.168</td></tr><tr><td>Classical CS</td><td>X</td><td>0.166</td></tr></table>

Baselines. We describe the sketches used in the algorithms we compare. Classical CS is a random Count-Sketch. IVY19 is a sparse sketch with learned values, and random positions for the non-zero entries. Ours (greedy) is a sparse sketch where both the values and positions of the non-zero entries are learned. The positions are learned by the Algorithm 2. The values are learned by the same

method in (Indyk et al., 2019). Ours (inner product) is a sparse sketch where both the values and the positions of the non-zero entries are learned. The positions are learned by  $S_{1}$  in Algorithm 3.

We also give a sensitivity analysis for our algorithm, where we compare our algorithm with the following variants: Only row sampling (perform projection by ridge leverage score sampling),  $\ell_2$  sampling (Replace leverage score sampling with  $\ell_2$  (Euclidean row norm) sampling and maintain the same downstream step), and Randomly Grouping (Use ridge leverage score sampling but randomly distribute the remaining rows). Both our Algorithm 3 and these variants take the input as the average over the entire training matrix. The result shows that all of these variants are worse than or the same as standard non-learned sketching. We defer the results of this part to Appendix A.1.

Result Summary. Our empirical results are provided in Table 7.1 for both Algorithm 4 and Algorithm 1, where the errors take an average over 10 trials. We use the average of all training matrices from  $\mathrm{Tr}$ , as the input to the algorithm 3. We note that all the steps of our training algorithms are done on the training data. Hence, no additional computational cost is incurred for the sketching algorithm on the test data. Experimental parameters (i.e., learning rate for gradient descent) can be found in Appendix H. For both sketching algorithms, Ours are always the best of the four sketches. It is significantly better than Classical CS, obtaining improvements of around  $70\%$ . It also obtains a roughly  $30\%$  improvement over IVY19.

Wall-Clock Times. The offline learning runtime is in Table 7.4, which is the time to train a sketch on  $\mathcal{A}_{\mathrm{train}}$ . We can see that although the greedy method will take much longer (1h  $45\mathrm{min}$ ), our second approach is much faster (5 seconds) than the previous algorithm in (Indyk et al., 2019) (3 min) and can still achieve a similar error as the greedy algorithm. The reason is that Algorithm 3 only needs to compute the ridge leverage scores on the training matrix once, which is actually much cheaper than IVY19 which needs to compute a differentiable SVD many times during gradient descent.

In the rest of this section, we study the performance of our second approach in the few-shot learning setting. We first consider the case where we only have one training matrix randomly sampled from  $\mathrm{Tr}$ . Here, we compare our method with the 1Shot2Vec method proposed in (Indyk et al., 2021) in the same setting ( $k = 10$ ,  $m = 40$ ) as in their empirical evaluation. The result is shown in Table 7.2. Compared to 1Shot2Vec, our method reduces the error by around  $50\%$ , and has an even slightly faster runtime.

Table 7.1: Test errors for LRA. (Left: two-side sketch. Right: one-side sketch)  

<table><tr><td>k, m, Sketch</td><td>Logo</td><td>Friends</td><td>Hyper</td></tr><tr><td>20, 40, Classical CS</td><td>2.371</td><td>4.073</td><td>6.344</td></tr><tr><td>20, 40, IVY19</td><td>0.687</td><td>1.048</td><td>3.764</td></tr><tr><td>20, 40, Ours (greedy)</td><td>0.500</td><td>0.899</td><td>2.497</td></tr><tr><td>20, 40, Ours (inner product)</td><td>0.532</td><td>0.733</td><td>2.975</td></tr><tr><td>30, 60, Class CS</td><td>1.642</td><td>2.683</td><td>5.390</td></tr><tr><td>30, 60, IVY19</td><td>0.734</td><td>1.077</td><td>3.748</td></tr><tr><td>30, 60, Ours (greedy)</td><td>0.492</td><td>0.794</td><td>2.492</td></tr><tr><td>30, 60, Ours (inner product)</td><td>0.436</td><td>0.733</td><td>2.409</td></tr></table>

<table><tr><td>k, m, Sketch</td><td>Logo</td><td>Friends</td><td>Hyper</td></tr><tr><td>20, 40, Classical CS</td><td>0.930</td><td>1.542</td><td>2.971</td></tr><tr><td>20, 40, IVY19</td><td>0.255</td><td>0.723</td><td>1.273</td></tr><tr><td>20, 40, Ours (greedy)</td><td>0.196</td><td>0.407</td><td>0.784</td></tr><tr><td>20, 40, Ours (inner product)</td><td>0.205</td><td>0.407</td><td>1.223</td></tr><tr><td>30, 60, Classical CS</td><td>0.650</td><td>1.0575</td><td>2.315</td></tr><tr><td>30, 60, IVY19</td><td>0.290</td><td>0.713</td><td>1.274</td></tr><tr><td>30, 60, Ours(greedy)</td><td>0.197</td><td>0.406</td><td>0.717</td></tr><tr><td>30, 60, Ours(inner product)</td><td>0.201</td><td>0.340</td><td>0.943</td></tr></table>

Table 7.2: Test errors and training times for LRA in the one-shot setting (using Alg. 1 with one sketch)  

<table><tr><td>Algorithm</td><td>Dataset</td><td>Few-shot Error</td><td>Training Time</td></tr><tr><td rowspan="3">Classical CS</td><td>Logo</td><td>0.331</td><td></td></tr><tr><td>Friends</td><td>0.524</td><td>X</td></tr><tr><td>Hyper</td><td>1.082</td><td></td></tr><tr><td rowspan="3">1shot2Vec</td><td>Logo</td><td>0.171</td><td>5.682</td></tr><tr><td>Friends</td><td>0.306</td><td>5.680</td></tr><tr><td>Hyper</td><td>0.795</td><td>1.054</td></tr><tr><td rowspan="3">Ours (inner product)</td><td>Logo</td><td>0.065</td><td>4.515</td></tr><tr><td>Friends</td><td>0.139</td><td>4.773</td></tr><tr><td>Hyper</td><td>0.535</td><td>0.623</td></tr></table>

Table 7.3: Test errors for LRA in the few-shot setting (using Alg. 1 from Indyk et al. (2019) with one sketch)  

<table><tr><td>Sketch</td><td>Logo</td><td>Friends</td><td>Hyper</td></tr><tr><td>Ours (Initialization only)</td><td>0.065</td><td>0.139</td><td>0.535</td></tr><tr><td>Ours + FewShotSGD</td><td>0.048</td><td>0.125</td><td>0.443</td></tr><tr><td>1Shot1Vec only</td><td>0.171</td><td>0.306</td><td>0.795</td></tr><tr><td>1Shot1Vec + FewShot SGD</td><td>0.104</td><td>0.229</td><td>0.636</td></tr><tr><td>Classical CS</td><td>0.331</td><td>0.524</td><td>1.082</td></tr><tr><td>Classical CS + FewShot SGD</td><td>0.173</td><td>0.279</td><td>0.771</td></tr></table>

![](images/bef70c240b8b0e6ec31a48686d6a7fe06ffa5bdcaa540d68b848967de5dece04.jpg)

![](images/3da51b43183eafbb47fa954f7cb485097bc65ab13bd3669e838b6b8235250dcc.jpg)

![](images/d23c8ef7a6d54e6c72fa82139a64f664595fdefc7679fce7cbd3246eb6eb2da6.jpg)  
Figure 7.1: Test error of LASSO in Electric dataset.

Indyk et al. (2021) also proposed a FewShotSGD algorithm which further improves the non-zero values of the sketches after different initialization methods. We compare the performance of this approach for different initialization methods: in all initialization methods, we only use one training matrix and we use three training matrices for the FewShotSGD step. The results are shown in Table 7.3. We report the minimum error of 50 iterations of the FewShotSGD because we aim to compare the computational efficiency for different methods. From the table we see that our approach plus the FewShotSGD method can achieve a much smaller error, with around a  $50\%$  improvement upon (Indyk et al., 2021). Moreover, even without further optimization by FewShotSGD, our initialization method for learning the non-zero locations in CountSketch obtains a smaller error than other methods (even when they are optimized with 1ShotSGD or FewShotSGD learning).

# 8 EXPERIMENTS: SECOND-ORDER OPTIMIZATION

In this section, we consider the IHS on the following instance of LASSO regression:

$$
x ^ {*} = \arg \min  _ {\| x \| _ {1} \leq \lambda} f (x) = \arg \min  _ {\| x \| _ {1} \leq \lambda} \frac {1}{2} \| A x - b \| _ {2} ^ {2}, \tag {8.1}
$$

where  $\lambda$  is a parameter. We also study the performance of the sketches on the matrix estimation with a nuclear norm constraint problem, the fast regression solver (van den Brand et al. (2021)), as well as the use of sketches for first-order methods. The results can be found in Appendix B. All of our experiments are conducted on a laptop with a 1.90GHz CPU and 16GB RAM. The offline training is done separately using a single GPU. The details of the implementation are deferred to Appendix H.

Dataset. We use the Electric $^1$  dataset of residential electric load measurements. Each row of the matrix corresponds to a different residence. Matrix columns are consecutive measurements at different times. Here  $A^i \in \mathbb{R}^{370 \times 9}$ ,  $b^i \in \mathbb{R}^{370 \times 1}$ , and  $|(A, b)_{\mathrm{train}}| = 320$ ,  $|(A, b)_{\mathrm{test}}| = 80$ . We set  $\lambda = 15$ .

Experiment Setting. We compare the learned sketch against the classical Count-Sketch $^2$ . We choose  $m = 6d, 8d, 10d$  and consider the error  $f(x) - f(x^{*})$ . For the heavy-row Count-Sketch, we allocate  $30\%$  of the sketch space to the rows of the heavy row candidates. For this dataset, each row represents a specific residence and hence there is a strong pattern of the distribution of the heavy rows. We select the heavy rows according to the number of times each row is heavy in the training data. We give a detailed discussion about this in Appendix B.1. We highlight that it is still possible to recognize the pattern of the rows even if the row orders of the test data are permuted. We also consider optimizing the non-zero values after identifying the heavy rows, using our new approach in Section 6.

Results. We plot in Figures 7.1 the mean errors on a logarithmic scale. The average offline training time is 3.67s to find a superset of the heavy rows over the training data and 66s to optimize the values when  $m = 10d$ , which are both faster than the runtime of Indyk et al. (2019) with the same parameters. Note that the learned matrix  $S$  is trained offline only once using the training data. Hence, no additional computational cost is incurred when solving the optimization problem on the test data.

We see all methods display linear convergence, that is, letting  $e_k$  denote the error in the  $k$ -th iteration, we have  $e_k \approx \rho^k e_1$  for some convergence rate  $\rho$ . A smaller convergence rate implies a faster convergence. We calculate an estimated rate of convergence  $\rho = (e_k / e_1)^{1/k}$  with  $k = 7$ . We can see both sketches, especially the sketch that optimizes both the positions and values, show significant improvements. When the sketch size is small (6d), this sketch has a convergence rate that is just  $13.2\%$  of that of the classical Count-Sketch, and when the sketch size is large (10d), this sketch has a smaller convergence rate that is just  $12.1\%$ .

# REFERENCES

Anders Aamand, Piotr Indyk, and Ali Vakilian. (learned) frequency estimation algorithms under zipfian distribution. arXiv preprint arXiv:1908.05198, 2019.  
Akshay Agrawal, Brandon Amos, Shane T. Barratt, Stephen P. Boyd, Steven Diamond, and J. Zico Kolter. Differentiable convex optimization layers. In Advances in Neural Information Processing Systems, pp. 9558-9570, 2019.  
Haim Avron, Kenneth L. Clarkson, and David P. Woodruff. Sharper bounds for regularized data fitting. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques, (APPROX/RANDOM), pp. 27:1-27:22, 2017.  
Jean Bourgain, Sjoerd Dirksen, and Jelani Nelson. Toward a unified theory of sparse dimensionality reduction in Euclidean space. Geometric and Functional Analysis, pp. 1009-1088, 2015.  
Matthew Brand. Fast low-rank modifications of the thin singular value decomposition. Linear Algebra and its Applications, 415.1, 2006.  
Lung-Sheng Chien and Samuel Rodriguez Bernabeu. Fast singular value decomposition on gpu. NVIDIA presentation at GPU Technology Conference, 2019. URL https://developerdownload.nvidia.com/video/gputechconf/gtc/2019/presentation/s9226-fast-singular-value-decomposition-on-gpus-v2.pdf.  
Kenneth L Clarkson and David P Woodruff. Numerical linear algebra in the streaming model. In Proceedings of the forty-first annual symposium on Theory of computing (STOC), pp. 205-214, 2009.  
Kenneth L Clarkson and David P Woodruff. Low-rank approximation and regression in input sparsity time. Journal of the ACM (JACM), 63(6):54, 2017.  
Edith Cohen, Ofir Geri, and Rasmus Pagh. Composable sketches for functions of frequencies: Beyond the worst case. In International Conference on Machine Learning, pp. 2057-2067. PMLR, 2020.  
Michael B. Cohen, Cameron Musco, and Christopher Musco. Input sparsity time low-rank approximation via ridge leverage score sampling. In Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, (SODA), pp. 1758-1777, 2017.  
Graham Cormode and Charlie Dickens. Iterative hessian sketch in input sparsity time. In Proceedings of 33rd Conference on Neural Information Processing Systems (NeurIPS), Vancouver, Canada, 2019.  
Yihe Dong, Piotr Indyk, Ilya Razenshteyn, and Tal Wagner. Learning sublinear-time indexing for nearest neighbor search. In International Conference on Learning Representations, 2020.  
Talya Eden, Piotr Indyk, Shyam Narayanan, Ronitt Rubinfeld, Sandeep Silwal, and Tal Wagner. Learning-based support estimation in sublinear time. In International Conference on Learning Representations, 2021.  
Chinmay Hegde, Aswin C. Sankaranarayanan, Wotao Yin, and Richard G. Baraniuk. Numax: A convex approach for learning near-isometric linear embeddings. In IEEE Transactions on Signal Processing, pp. 6109-6121, 2015.  
Piotr Indyk, Ali Vakilian, and Yang Yuan. Learning-based low-rank approximations. In Advances in Neural Information Processing Systems, pp. 7400-7410, 2019.  
Piotr Indyk, Tal Wagner, and David Woodruff. Few-shot data-driven algorithms for low rank approximation. Advances in Neural Information Processing Systems, 34, 2021.  
Tanqiu Jiang, Yi Li, Honghao Lin, Yisong Ruan, and David P. Woodruff. Learning-augmented data stream algorithms. In International Conference on Learning Representations, 2020.  
Xiangrui Meng and Michael W Mahoney. Low-distortion subspace embeddings in input-sparsity time and applications to robust linear regression. In Proceedings of the forty-fifth annual ACM symposium on Theory of computing, pp. 91-100, 2013.

Jelani Nelson and Huy L Nguyen. Osnap: Faster numerical linear algebra algorithms via sparser subspace embeddings. In Foundations of Computer Science (FOCS), 2013 IEEE 54th Annual Symposium on, pp. 117-126, 2013.  
Jelani Nelson and Huy L. Nguyen. Lower bounds for oblivious subspace embeddings. In Automata, Languages, and Programming - 41st International Colloquium (ICALP), pp. 883-894, 2014.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, and Trevor Killeen. Pytorch: An imperative style, high-performance deep learning library. 2019.  
Mert Pilanci and Martin J. Wainwright. Iterative Hessian sketch: Fast and accurate solution approximation for constrained least-squares. J. Mach. Learn. Res., 17:53:1-53:38, 2016.  
Tamas Sarlos. Improved approximation algorithms for large matrices via random projections. In 47th Annual IEEE Symposium on Foundations of Computer Science (FOCS), pp. 143-152, 2006.  
Jan van den Brand, Binghui Peng, Zhao Song, and Omri Weinstein. Training (overparametrized) neural networks in near-linear time. In James R. Lee (ed.), 12th Innovations in Theoretical Computer Science Conference, ITCS, volume 185, pp. 63:1-63:15, 2021.  
Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. In Yonina C. Eldar and Gitta Kutyniok (eds.), Compressed Sensing: Theory and Applications, pp. 210-268. Cambridge University Press, 2012. doi: 10.1017/CBO9780511794308.006.  
Jingdong Wang, Ting Zhang, Nicu Sebe, and Heng Tao ShenWang. A survey on learning to hash. In IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 769-790, 2017.
