# Approximate Euclidean lengths and distances beyond Johnson-Lindenstrauss

Anonymous Author(s)

Affiliation

Address

email

# Abstract

A classical result of Johnson and Lindenstrauss states that a set of  $n$  high dimensional data points can be projected down to  $O(\log n / \epsilon^2)$  dimensions such that the square of their pairwise distances is preserved up to a small distortion  $\epsilon \in (0,1)$ . It has been proved that the JL lemma is optimal for the general case, therefore, improvements can only be explored for special cases. This work aims to improve the  $\epsilon^{-2}$  dependency based on techniques inspired by the Hutch++ Algorithm [24], which reduces  $\epsilon^{-2}$  to  $\epsilon^{-1}$  for the related problem of implicit matrix trace estimation. For  $\epsilon = 0.01$ , for example, this translates to 100 times less matrix-vector products in the matrix-vector query model to achieve the same accuracy as other previous estimators. We first present an algorithm to estimate the Euclidean lengths of the rows of a matrix. We prove element-wise probabilistic bounds that are at least as good as standard JL approximations in the worst-case, but are asymptotically better for matrices with decaying spectrum. Moreover, for any matrix, regardless its spectrum, the algorithm achieves  $\epsilon$ -accuracy for the total, Frobenius norm-wise relative error using only  $O(\epsilon^{-1})$  queries. This is a quadratic improvement over the norm-wise error of standard JL approximations. We finally show how these results can be extended to estimate the Euclidean distances between data points and to approximate the statistical leverage scores of a tall-and-skinny data matrix, which are ubiquitous for many applications. Proof-of-concept numerical experiments are presented to validate the theoretical analysis.

# 1 Introduction

The Johnson-Lindenstrauss (JL) lemma [20] is a fundamental concept in dimensionality reduction and data science. Given a set of  $n$  high dimensional data points  $X = \{x_{1},\ldots ,x_{n}\}$ , where each  $x_{i}\in \mathbb{R}^{d}$ , the goal is to find a projection  $f:\mathbb{R}^d\to \mathbb{R}^k$  that maps the vectors to a much smaller dimension  $k\ll d$  such that the geometry of the original set is approximately preserved. Specifically, the projection should preserve the pairwise distances up to a small distortion  $\epsilon \in (0,1)$ , that is

$$
(1 - \epsilon) \| x _ {i} - x _ {j} \| ^ {2} \leq \| f (x _ {i}) - f (x _ {j}) \| ^ {2} \leq (1 + \epsilon) \| x _ {i} - x _ {j} \| ^ {2}, \tag {1}
$$

for all  $i, j \in [n]$ . If  $f$  satisfies this property, then it is called an  $\epsilon$ -isometry. Johnson and Lindenstrauss proved that, given  $\epsilon$ , such an  $f$  can be found in randomized polynomial time and that the projected dimension is no larger than  $O(\log n / \epsilon^2)$ . In the last decades the JL lemma has made an impact in many areas, including Graph Algorithms [3, 31], Machine Learning [2, 6, 16, 11], Numerical Linear Algebra [29, 9, 23, 33] and Optimization [15, 27, 13].

In this work we study the problem of approximating the Euclidean row norms of an arbitrary matrix  $A \in \mathbb{R}^{n \times d}$  in the so-called matrix-vector query model. In this model, the matrix  $A$  might not be explicitly available, but we have access to a linear operator that computes the product  $Ax$ , for an

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

arbitrary vector  $x$ . For non-symmetric and rectangular matrices we assume that we can compute both  $Ax$  and  $A^\top x$ . An example application of this model is the case where we want to compute the Euclidean row norms of a function  $f(A)$ . If it is expensive to compute the true Euclidean row norms of  $f(A)$ , one would seek for a fast approximation. One way to do this, up to a relative error  $\epsilon \in (0,1/2)$ , is to use the aforementioned JL lemma. We first give some definitions.

Definition 1 (Johnson-Lindenstrauss transform [29]). A random matrix  $S \in \mathbb{R}^{r \times d}$  forms a Johnson-Lindenstrauss transform with parameters  $\epsilon, \delta \in (0,1/2)$  and positive integer  $n$ , or  $(\epsilon, \delta, n)$ -JLT for short, if with probability at least  $1 - \delta$ , for any fixed set  $V \subseteq \mathbb{R}^d$  with  $n$  elements it holds that  $(1 - \epsilon) \| v \|^2 \leq \| Sv \|^2 \leq (1 + \epsilon) \| v \|^2$  for all  $v \in V$ .

It is known that Gaussian matrices can provide JLTs; c.f. [20, 2].

Lemma 1 (Gaussian random projections [20, 2]). Let  $G \in \mathbb{R}^{r \times d}$  with i.i.d. elements from  $\mathcal{N}(0,1/\sqrt{r})$  and  $\epsilon \in (0,1/2)$ . For a fixed  $x \in \mathbb{R}^d$  it holds that

$$
\Pr \left[ \left| \| x \| ^ {2} - \| G x \| ^ {2} \right| \leq \epsilon \| x \| ^ {2} \right] \geq 1 - 2 \exp \left(- \frac {r \left(\epsilon^ {2} - \epsilon^ {3}\right)}{4}\right).
$$

For a set  $X \subset \mathbb{R}^d$  of  $n$  vectors and for  $\delta \in (0,1/2)$ , as long as  $r \geq \frac{4\log(2n/\delta)}{\epsilon^2 - \epsilon^3}$ , then  $G$  forms an  $(\epsilon, \delta, n)$ -JLT.

The dimension  $r$  of  $G$  depends on  $1 / \epsilon^2$ , which can quickly become very large if a high accuracy is needed. If  $r$  is very large, then it is also very expensive to compute the product  $GA$ . There is therefore no advantage in taking an approximate solution over computing the true solution. Here, we would like to improve this dependency. Note that vector norm estimation indirectly provides the classical Johnson-Lindenstrauss result, since the distance between  $x$  and  $y$  is just the length of  $x - y$ . The concept of Oblivious Subspace Embeddings [29] generalizes the JLT definition for an entire subspace. We use the definitions from [33].

Definition 2 (Oblivious Subspace Embedding). Let  $\mathcal{D}$  be a distribution on  $r\times n$  matrices  $S$ , where  $r$  is a function of  $n,d$  and  $\epsilon, \delta \in (0,1/2)$ . We call  $S$  an  $(\epsilon, \delta)$  Oblivious Subspace Embedding, or  $(\epsilon, \delta)$ -OSE if for any fixed  $n\times d$  matrix  $A$ ,  $S \sim \mathcal{D}$  is a  $(1\pm \epsilon)l_{2}$ -Subspace Embedding for  $A$  with probability at least  $1 - \delta$ , that is, for all  $x \in \mathbb{R}^d$  it holds that

$$
(1 - \epsilon) \| A x \| ^ {2} \leq \| S A x \| ^ {2} \leq (1 + \epsilon) \| A x \| ^ {2}.
$$

Notation. Throughout the paper, unless stated otherwise, the following notation is used. We follow the Householder notation, denoting matrices with capital letters, vectors with small letters, and scalars with Greek letters. All vectors are assumed to be columns.  $[n]$  is the set  $\{1,2,\dots,n\}$ , where  $n\in \mathbb{N}$ .  $I_{n}$  is the identity matrix of size  $n\times n$  and  $e_i$  the  $i$ -th column of the standard basis. For a matrix  $A$ ,  $A_{i,j}$  and  $A_{:,j}$  are the  $i$ -th row and  $j$ -th column, respectively, both assumed to be column vectors, and  $A_{i,j}$  is the element in row  $i$  and column  $j$ . For a set  $\mathcal{K}\subseteq [d]$ ,  $A_{:,K}$  denotes the submatrix of  $A$  containing the columns defined by  $\mathcal{K}$ .  $A_{k}$  denotes the best rank- $k$  approximation of  $A$  in the 2-norm. For a vector  $x$ ,  $\| x\| _A = \sqrt{x^\top Ax}$ . The 2-norm is assumed for matrices and vectors when the subscript is omitted.  $A^{\top}$  is the transpose of  $A$  and  $A^{\dagger}$  is the pseudoinverse.  $\mathbb{P}[\alpha ]\in [0,1]$  is the probability of occurrence of an event  $\alpha$ .  $\mathcal{N}(\mu ,\sigma)$  is the normal distribution with mean value  $\mu$  and standard deviation  $\sigma$ .  $\sigma_{i}(A)$  denotes the  $i$ -th largest singular value of  $A$ .  $\mathfrak{nnz}(x)$  is the number of nonzeros of  $x$ , where  $x$  can be either a vector or a matrix.  $\tilde{O} (k)\coloneqq O(k\log^c (k))$  for some constant  $c$ . We refer to matrices with i.i.d. elements from  $\mathcal{N}(0,1)$  as Gaussian matrices.  $A\succeq B$  denotes that  $A - B$  is Positive Semi-Definite (PSD). In the complexity analysis we denote by  $\omega$  the fast matrix multiplication exponent, where  $2\leq \omega < 2.37286$  [1].

Related work. A related topic is stochastic matrix trace estimation [19, 4, 24, 28, 26]. Intuitively, a set of data points can be seen as the columns of a matrix. In various applications the trace of such a matrix contains useful information like triangle counts in graphs [3]. Hutchinson [19] proposed a randomized algorithm to rapidly approximate the trace of such a matrix, which uses similar ideas to JL: it projects the rows of the matrix onto a low-dimensional subspace so that the trace can be quickly computed. Avron and Toledo showed that the dimension of that subspace needs to be proportional to  $O(1 / \epsilon^2)$  in order to guarantee a worst-case  $\epsilon$ -approximation for the trace [4]. This condition matches the requirements for the  $\epsilon$ -isometry of JL. The  $\epsilon^{-2}$  overhead can be prohibitive when  $\epsilon$  is small, i.e. in applications where high accuracy is needed. Recently, in their seminal work, Meyer, Musco,

Musco, and Woodruff [24] proved a remarkable result: their Hutch++ algorithm is the first to obtain  $\epsilon$ -accuracy for stochastic trace estimation while requiring only  $1 / \epsilon$  matrix-vector queries. For the related problem of estimating the diagonal elements of a matrix, which was also recently studied in depth [18, 5], Baston and Nakatsukasa [5] achieved  $\epsilon$ -accuracy for the total, norm-wise error of the entire diagonal using  $O(1 / \epsilon)$  matrix-vector queries, but not for each individual diagonal element, which should not be possible due to the optimality of the JL lemma [22]. It is worth noting that the squared row norms of a matrix  $A$  can be found in the diagonal of  $A^\top A$ , therefore, our work is closely connected the diagonal estimation literature. Our results for the total norm-wise error of our estimators (see e.g. Theorem 2), are tighter than simply using [5] on  $A^\top A$ , since we are exploiting the special structure of  $A^\top A$ . In a Fine-Grained complexity perspective, estimating row norms can be easily reduced to diagonal estimation, but the opposite reduction is not straightforward, therefore, one can argue that diagonal estimation is harder, which justifies our tighter bounds.

Contributions. In Algorithm 1, we present the main algorithm of this work to approximate the Euclidean row norms of a real matrix  $A$ , which is inspired by the Hutch++ algorithm [24]. Following [24], this algorithm is called "Adaptive", since it needs to make two passes over the input matrix  $A$ . Here, by  $T_{\mathrm{MM}}(A,m)$  we denote the complexity of computing the product  $AB$  for a matrix  $B$  with  $m$  columns. A brief summary of our main results regarding the approximation bounds of Algorithm 1,

Algorithm 1 Adaptive Euclidean Norm Estimation  
Input: Matrix  $A\in \mathbb{R}^{n\times d}$ $n\geq d$  , positive integer  $m <   d$    
Output:  $\mathrm{Alg1}(e_i^\top A)\approx \| e_i^\top A\| ^2$  # Step 1: Low-rank approximation 1: Construct two random matrices  $S,G\in \mathbb{R}^{d\times m}$  with i.i.d. elements from  $\mathcal{N}(0,1)$  .  $\triangleright O(dm)$  2: Compute  $B = A^{\top}(AS)$  .  $\triangleright O(T_{\mathrm{MM}}(A,m))$  3: Compute an orthonormal basis  $Q\in \mathbb{R}^{d\times m}$  for range(B) (e.g., via QR).  $\triangleright O(dm^2)$  # Step 2: Project and compute row norms 4: Compute  $\tilde{A} = AQ$  and  $C = AG$  .  $\triangleright O(T_{\mathrm{MM}}(A,m))$  5: Compute  $\tilde{\Delta} = A(I - QQ^{\top})G = C - \tilde{A} (Q^{\top}G)$  .  $\triangleright O(nm^2)$  6: return  $\mathrm{Alg1}(e_i^\top A) = \| e_i^\top \tilde{A}\| ^2 +\| e_i^\top \tilde{\Delta}\| ^2$  , for all  $i\in [n]$

as well as for Algorithms 2 and 3 for Euclidean distance and leverage scores estimation, respectively, can be found in Table 1. For the precise, tighter bounds we refer to the corresponding sections.

Table 1: Bounds for different problems using  $m = O(1 / \epsilon^2)$  matrix-vector queries (ignoring logarithmic factors on  $n, \delta$ ), for  $\epsilon \in (0,1/2)$ . Here  $m > k$ ,  $\bar{A}_k = A - A_k$  and  $\bar{M}_k = M - M_k$ , where  $M = BA$  and the rows of  $B$  define the pairwise distance vectors, and  $\theta_i$  are the leverage scores of  $A$ .  

<table><tr><td></td><td colspan="2">Element-wise</td><td colspan="3">Frobenius norm-wise</td></tr><tr><td></td><td>This work</td><td>JL</td><td>This work</td><td>JL</td><td>ref.</td></tr><tr><td>Row norms</td><td>ε||eiT A||| |eTiAik||</td><td>ε||eiT A||2</td><td>ε2||A||2F</td><td>ε||A||2F</td><td>Thms. 1 &amp; 2</td></tr><tr><td>Distances</td><td>ε||eiT M||| |eTiMk||</td><td>ε||eiT M||2</td><td>ε2||M||2F</td><td>ε||M||2F</td><td>Thm. 5</td></tr><tr><td>Leverage scores</td><td>εθi</td><td>εθi</td><td>ε2d</td><td>εd</td><td>Thm. 3</td></tr></table>

Outline. The analysis of Algorithm 1 is given in Section 2. In Sections 3 and 4 we show two important applications of the main results, namely for the estimation of the pairwise Euclidean distances between a set of data points and for the estimation of the statistical leverage scores of a tall-and-skinny data matrix. In Section 5 we present indicative experiments to validate the theoretical analysis, before finally giving concluding remarks and future directions in Section 6.

# 2 Analysis of Algorithm 1

In this section we provide the analysis of Algorithm 1. Preliminary results and long proofs which were omitted from the main text and can be found in the Appendix. We first prove element-wise

approximation guarantees for Algorithm 1, and we also argue that these bounds are tight, that is, any method which is based on low-rank projections cannot do much better than Algorithm 1. We state the following general result for the element-wise bounds of Algorithm 1.

Lemma 2. (Proof in the Appendix) Let  $A \in \mathbb{R}^{n \times d}$ . If we use Algorithm 1 with  $m$  matrix-vector queries to estimate the Euclidean lengths of the rows  $e_i^\top A$ ,  $i \in [n]$ , then as long as  $m \geq l \geq 32\log(4n/\delta)$  it holds that

$$
\left| A l g I \left(e _ {i} ^ {\top} A\right) - \left\| e _ {i} ^ {\top} A \right\| ^ {2} \right| \leq \sqrt {\frac {8 \log \left(\frac {2 n}{\delta}\right)}{l}} \| e _ {i} ^ {\top} A (I - Q Q ^ {\top}) \| ^ {2}, f o r a l l i \in [ n ],
$$

with probability at least  $1 - \delta$  for all  $i\in [n]$  simultaneously.

Evidently, this result implies that if we can determine a suitable bound for  $\| e_i^\top A(I - QQ^\top)\|^2$  then we automatically get a proper bound for the element-wise approximations of Algorithm 1. If  $A$  has a fast decaying spectrum and  $Q$  captures the dominant eigenspace of  $A$  we can expect that our approximations are very accurate, even for small  $l$ . For the general case, however, the following Lemma 3 as well as the optimality of the JL lemma [22] already hint that this is not possible (see also Appendix II, Limitations of low-rank projections).

Lemma 3.  $\| e_i^\top (A - A_k)\| _2^2\leq \sigma_{k + 1}^2 (A)\leq \frac{\|A_k\|_F^2}{k}$

Proof. Clearly,  $\| e_i^\top (A - A_k)\| _2^2\leq \max_{\| x\| = 1}\| x^\top (A - A_k)\| _2^2 = \sigma_{k + 1}^2 (A)$ . For the second part we have that  $\sigma_{k + 1}^2 (A)\leq \frac{1}{k}\sum_{i = 1}^{k}\sigma_i^2 (A) = \frac{\|A_k\|_F^2}{k}$ .

# 2.1 Projecting rows on randomly chosen subspaces

To proceed further with the analysis, we show some length-preserving properties of the orthogonal projector  $QQ^{\top}$ , which is an orthogonal projector on a random subspace as obtained in Algorithm 1. Note that Corollary 1 is stated for constant factor approximations. Here we provide a brief proof sketch. For the main result we refer to Lemma 8 in Appendix III.

Corollary 1 (Projection on rowspace  $(SA^{\top}A)$ ). (Proof in the Appendix) Let  $\delta \in (0, \frac{1}{2})$ ,  $\bar{A}_k = A - A_k$ , and  $S$  be such that

(i)  $S\sim \mathcal{D}$ , where  $\mathcal{D}$  is an  $(1 / 3,\delta)$ -OSE for any fixed  $k$ -dimensional subspace;

(ii)  $S$  is a  $(1/3, \delta, 2n)$ -JLT.

If  $Q$  is a matrix that forms an orthonormal basis for  $\mathrm{rowspace}(SA^{\top}A)$ , then, with probability at least  $1 - 2\delta$ , for all  $i \in [n]$  simultaneously, it holds that

$$
\| e _ {i} ^ {\top} A (I - Q Q ^ {\top}) \| ^ {2} \leq \| e _ {i} ^ {\top} (\bar {A} _ {k}) \| ^ {2} + \frac {1}{2} \frac {\sigma_ {k + 1} ^ {2} (A)}{\sigma_ {k} ^ {2} (A)} \| e _ {i} ^ {\top} A _ {k} \| \| e _ {i} ^ {\top} \bar {A} _ {k} \| \leq \frac {3}{2} \| e _ {i} ^ {\top} A \| \| e _ {i} ^ {\top} \bar {A} _ {k} \|.
$$

Proof sketch. To prove the result it suffices to find a projector within  $\mathrm{rowspace}(SA^{\top}A)$  with the desired properties. To do this, we consider the matrix  $\Pi_k = V_k(SV_k\Sigma_k^2)^\dagger SA^\top A$ , where  $V_{k},\Sigma_{k}$  originate from the SVD of  $A_{k} = U_{k}\Sigma_{k}V_{k}^{\top}$ . Clearly, this  $\Pi_k$  is a rank- $k$  matrix within  $\mathrm{rowspace}(SA^{\top}A)$ . After some algebra, the problem reduces to get a bound for the quantities  $|e_i^\top AV_k\Sigma_k^2 C^{-1}V_k^\top S^\top SV_k\bar{\Sigma}_k^2\bar{V}_k^\top A^\top e_i|$ , for all  $i\in [n]$ , where the existence of  $C^{-1}$  is guaranteed due to the  $(1/3,\delta)$ -OSE property of  $S$ . For each  $i$ , this quantity is the absolute value of the inner product  $\langle S(V_kC^{-1}V_k^\top A^\top)e_i,S(\bar{V}_k\bar{V}_k^\top A^\top)e_i\rangle$ , which can be written in a simplified form as  $\langle Sx_k,S\bar{x}_k\rangle$ . Therefore, we use an  $(1/3,\delta,2n)$ -JLT to bound the inner products between the vectors of the set

$$
V = \left\{e _ {i} ^ {\top} A V _ {k} \Sigma_ {k} ^ {2} C ^ {- 1} V _ {k} ^ {\top} | i \in [ n ] \right\} \bigcup \left\{e _ {i} ^ {\top} A \bar {V} _ {k} \bar {\Sigma} _ {k} ^ {2} \bar {V} _ {k} ^ {\top} | i \in [ n ] \right\}.
$$

Having all pieces in-place, we can finally bound the element-wise approximations of Algorithm 1.

Theorem 1. (Proof in the Appendix) Let  $A \in \mathbb{R}^{n \times d}$  and  $n \geq d$ . If we use Algorithm 1 with  $m$  matrix-vector queries to estimate the Euclidean lengths of the rows of  $A$ , then there exist global constants  $c, C$  such that, as long as

(i)  $m\geq l\geq O(\log (n / \delta))$  , such that  $G$  satisfies Lemma 1 and  $S$  forms an  $(1 / 3,\delta ,2n)$  -JLT,  
(ii)  $m \geq O(k + \log(1/\delta))$ , such that  $S$  forms an  $(1/3, \delta)$ -OSE for a  $k$ -dimensional subspace,

then it holds that

$$
\left| A l g 1 (e _ {i} ^ {\top} A) - \| e _ {i} ^ {\top} A \| ^ {2} \right| \leq C \sqrt {\frac {\log (\frac {n}{\delta})}{l}} \| e _ {i} ^ {\top} (A - A _ {k}) \| \| e _ {i} ^ {\top} A \| \leq C \sqrt {\frac {\log (\frac {n}{\delta})}{l k}} \| A _ {k} \| _ {F} \| e _ {i} ^ {\top} A \|,
$$

for all  $i \in [n]$  with probability at least  $1 - 3\delta$ .

Discussion. We can investigate the bounds for special matrix cases. We highlight the approximation power of Algorithm 1 for matrices with decaying spectrum. For matrices with a linear decay it suffices to take  $m \gtrsim O(\epsilon^{-1}\sqrt{\log(n / \delta)})$  queries to achieve an almost  $\epsilon$ -accuracy. For matrices with exponential decay we can use as few as  $m \geq O(\log (1 / \epsilon))$  matrix-vector queries. For matrices with no decay, e.g., for orthogonal projector matrices, Lemma 2 already guarantees that Algorithm 1 provides at least as accurate element-wise approximations as standard JL projections. We recall once more that the JL lemma is optimal in the general case [22], therefore, improvements can only be derived for special cases, like the ones considered here.

# 2.2 Frobenius norm bounds

If we carefully examine the total Frobenius norm-wise error, we can in fact obtain a true  $\epsilon$ -relative error approximation. This cannot be done by "simply" adding together all element-wise bounds, i.e., we must use a different "collective" approach. We note that this is a quadratic improvement over classical JL projections. This result also makes the element-wise bounds more appealing: even if there remain few outliers that violate the element-wise  $\epsilon$ -approximation, the total error is still very small. Due to the tightness of Lemma 3, which is crucial, it is highly unlikely that low-rank projection-based methods can generally achieve better element-wise bounds.

Theorem 2. (Proof in the Appendix) In Algorithm 1, for some absolute constants  $c, C$ , if  $l > c \log(1/\delta)$ , it holds that

$$
\left| A \operatorname {l g} I (A) - \| A \| _ {F} ^ {2} \right| \leq C \sqrt {\frac {\log (\frac {1}{\delta})}{l k}} \| A \| _ {F} ^ {2},
$$

where  $\operatorname{Alg1}(A)$  is the sum of the returned approximations. For  $l = k = O\left(\frac{\sqrt{\log\left(\frac{1}{\delta}\right)}}{\epsilon}\right)$ , where  $\epsilon \in (0, 1/2)$ , setting  $m \geq O(k / \delta + \log \left(\frac{1}{\delta}\right))$ , it follows that

$$
\left| A \operatorname {l g} I (A) - \| A \| _ {F} ^ {2} \right| \leq \epsilon \| A \| _ {F} ^ {2}.
$$

# 2.3 Complexity

The complexity of Algorithm 1 is as follows. In the first step two matrices  $S$  and  $G$  must be generated with  $d \times m$  random elements each. Hence,  $O(dm)$  calls to a random number generator are required. In the second step, the products  $A^{\top}(AS)$  and  $A^{\top}(AG)$  can be both computed in  $O(T_{\mathrm{MM}}(A,m) + T_{\mathrm{MM}}(A^{\top},m))$ . Next we need to create an orthonormal basis for  $A^{\top}AS$  which has size  $d \times m$ . This can be done with a standard Householder QR or another orthogonal factorization in  $O(dm^{2})$  [17, Chapter 5]. The complexity of this operation can be improved using fast matrix multiplication primitives [12]. The product  $AQ$  costs  $O(T_{\mathrm{MM}}(A,m))$ . We then have to compute  $\tilde{A}(Q^{\top}G)$ , which takes  $O(dm^{2})$  or  $O(dm^{\omega - 1})$  to first obtain  $Q^{\top}G$  and then the same cost to get  $\tilde{A}(Q^{\top}G)$ . Finally, for the last step the squared row norms of  $2n$  vectors, the rows of  $\tilde{A}$ , and the rows of  $\tilde{\Delta}$ , are needed. For each row of  $\tilde{A}$  and  $\tilde{\Delta}$  the cost of computing the squared Euclidean norm is  $O(m)$ , therefore the cost for the last step is  $O(nm)$ . Summing up, the total cost of Algorithm 1 is  $O(d^{2}m + T_{\mathrm{MM}}(A,m) + nm)$ .

# 3 Euclidean distances

In many applications it is desired to find an approximate isometry for a set of data points. Let  $A \in \mathbb{R}^{n \times d}$  be a matrix whose rows define these  $d$ -dimensional data points. Assume we are interested

to estimate the distances between all the  $\binom{n}{2}$  rows of  $A$ . Let  $B$  be a matrix with size  $\binom{n}{2} \times n$  and each row of  $B$  is equal to the vector  $(e_i - e_j)^\top$  for some  $i, j \in [n]$ . Each row  $(e_i - e_j)^\top$  of  $B$ , when multiplied with  $A$ , gives the difference vector  $e_i^\top A - e_j^\top A$ . Therefore, to estimate the Euclidean distances between the rows of  $A$ , it is sufficient to estimate the Euclidean lengths of the rows of the matrix  $BA$ . In Algorithm 2 we describe this procedure for a general "incidence matrix"  $B$ , e.g., when one wants to estimate only a subset of the pairwise distances. Since  $B$  has in general more rows than  $A$ , the matrix multiplications must be computed in the correct order to minimize their complexity.

Algorithm 2 Adaptive Euclidean Distance Estimation  
Input: Data matrix  $A\in \mathbb{R}^{t\times d}$ $t\geq d$  , incidence matrix  $B\in \mathbb{R}^{n\times t}$  , positive integer  $m <   d$    
Output: Approximate pairwise distances  $\mathrm{Alg2}(e_i^\top BA))\approx \| e_i^\top BA\| ^2,i\in [n].$  # Step 1: Low-rank approximation 1: Construct two random matrices  $S,G\in \mathbb{R}^{d\times m}$  with i.i.d. elements from  $\mathcal{N}(0,1)$  .  $\triangleright O(dm)$  2: Compute the product  $\tilde{S} = A^{\top}(B^{\top}(B(AS)))$ $\triangleright O(T_{\mathrm{MM}}(A,m) + T_{\mathrm{MM}}(A^{\top},m) + nm)$  3: Compute an orthonormal basis  $Q\in \mathbb{R}^{d\times m}$  for range(S) (e.g., via QR).  $\triangleright O(dm^2)$  # Step 2: Project and compute row norms 4: Compute  $\tilde{A} = AQ$  and  $C = AG$  .  $\triangleright O(T_{\mathrm{MM}}(A,m))$  5: Compute  $\tilde{\Delta} = A(I - QQ^{\top})G = C - \tilde{A} (Q^{\top}G)$  .  $\triangleright O(tm^2)$  6: return  $\mathrm{Alg2}(e_i^\top BA) = ||(e_i^\top B)\tilde{A} ||^2 +||(e_i^\top B)\tilde{\Delta} ||^2$  , for all  $i\in [n]$  .  $\triangleright O(nm)$

Bounds. Approximation bounds can be directly derived from Theorems 1 and 2, replacing  $A$  with  $BA$ . For completeness, they can be found in Theorem 5 in the Appendix.

Complexity. The complexity of Algorithm 2 is as follows.  $O(dm)$  operations are needed to generate  $G$  and  $S$ . The product  $\tilde{S} = A^\top B^\top BAS$  is evaluated in three steps. We first compute  $AS$  in  $O(T_{\mathrm{MM}}(A,m))$ , then  $B(AS)$  and  $B^\top (BAS)$  in  $O(nm)$ , and finally  $A^\top (B^\top BAS)$  in  $O(T_{\mathrm{MM}}(A^\top ,m))$ . The intermediate products can be calculated in batches to save memory. The QR factorization of  $\tilde{S}$  requires  $O(dm^{2})$ . The products  $\tilde{A} = AQ$  and  $C = AG$  both require  $O(T_{\mathrm{MM}}(A,m))$ , whereas the product  $Q^\top G$  can be performed in  $O(dm^2)$ . Accordingly, the product  $\tilde{A}(Q^\top G)$  needs  $O(tm^2)$  and  $C - \tilde{A}(Q^\top G)O(tm)$ . In the last step each row norm costs  $O(m)$  operations, resulting in  $O(nm)$ . The total complexity of Algorithm 2 is therefore

$$
O \left(T _ {\mathrm {M M}} (A, m) + T _ {\mathrm {M M}} \left(A ^ {\top}, m\right) + n m + d m ^ {2} + t d m\right).
$$

# 4 Statistical leverage scores

We next consider the problem of approximating the leverage scores of a tall-and-skinny matrix. The leverage scores of the rows of  $A$  can be found in the diagonal of the orthogonal projector matrix  $P = AA^{\dagger} = UU^{\top}$ , where  $U$  is any orthonormal basis for range  $(A)$ . Specifically, the leverage score  $\theta_{i}$  of the  $i$ -th row of  $A$  is equal to all the following quantities

$$
\theta_ {i} = \| e _ {i} ^ {\top} A A ^ {\dagger} \| ^ {2} = e _ {i} ^ {\top} A A ^ {\dagger} e _ {i} = e _ {i} U U ^ {\top} e _ {i} = \| e _ {i} ^ {\top} U \| ^ {2}.
$$

It is known that the leverage scores of a  $n \times d$  matrix  $A$  with  $\mathrm{rank}(A) = r \leq d$  sum to  $r \colon \sum_{i=1}^{n} \theta_i = r$ . Leverage scores are important in outlier detection, graph sparsification, and numerical linear algebra. We consider the general case where  $U$  is not explicitly available, and we only have access to  $A$ .

To simplify the analysis, we assume that the matrix  $A$  has full column rank. The rank  $r$  of  $A$  as well as a set of  $r$  linearly independent columns of  $A$  can be computed in  $O(\mathsf{nnz}(A) + r^{\omega}\mathrm{poly}(\log \log d))$  [8, 7]. Finding less than  $\operatorname {rank}(A)$  columns with provable approximation bounds is also possible in similar time [30]. Therefore, from now on we assume that  $A$  has full column rank.

To use Algorithm 1 to estimate leverage scores, we first need a linear operator that computes  $AA^{\dagger}v$ , for an arbitrary vector  $v$ . Since evaluating  $(A^{\top}A)$  in order to compute its pseudoinverse is expensive,

we opt for a fast approximate operator. For this we can use standard techniques from the literature. One of the first approximation algorithms for tall-and-skinny leverage scores was proposed in [14]. Note that in [30] it was shown that this algorithm is only efficient for dense matrices, or more specifically for matrices with at least  $\omega (\log n)$  nonzeros per row. Given  $A\in \mathbb{R}^{n\times d}$  with  $n\gg d$  and  $\omega (\log n)$  nonzeros per row, the idea consists of approximating the leverage scores of the rows of  $A$  with the squared Euclidean row norms of the matrix

$$
A (\Pi_ {1} A) ^ {\dagger} \Pi_ {2}.
$$

Here,  $\Pi_1$  is a subspace embedding for range  $(A)$  and  $\Pi_2$  is an  $\epsilon$ -JLT. It can be proved that  $(\Pi_1 A)^\dagger$  is in fact an approximate "orthogonalizer" for  $A$ , a property that we can leverage in our algorithm. Specifically, we apply Algorithm 1 to approximate the Euclidean row norms of  $A(\Pi_1 A)^\dagger$ , instead of multiplying with  $\Pi_2$ . This procedure is described in Algorithm 3.

# Algorithm 3 Adaptive Leverage Scores Estimation

Input:  $A \in \mathbb{R}^{n \times d}$ , with  $n \gg d$  and  $\omega(\log n)$  nonzeros per row, positive integer  $m < d$ .

Output: Approximate leverage scores  $\tilde{\theta}_i\approx \| e_i^\top AA^\dagger \| ^2,i\in [n]$  # Step 1: Construct approximate pseudoinverse operator

1: Construct  $\Pi_1$ , an  $(\epsilon_1, \delta)$ -OSE for range  $(A)$ .  
2: Compute  $R$  from a QR factorization of  $\Pi_1A$ , i.e.  $\Pi_1A = QR$  and use  $R^{-1}$  as a substitute for  $(\Pi_1A)^\dagger$ .  
# Step 2: Low-rank approximation

3: Construct two random matrices  $S, G \in \mathbb{R}^{d \times m}$  with i.i.d. elements from  $\mathcal{N}(0,1)$ .  
4: Compute the product  $\tilde{S} = R^{-T}(A^{\top}(A(R^{-1}S)))$ .  
5: Compute an orthonormal basis  $Q\in \mathbb{R}^{d\times m}$  for range(S) (e.g., via QR).  $\triangleright O(dm^2)$  # Step 3: Project and compute row norms

6: Compute  $\tilde{A} = A(R^{-1}Q)$  and  $C = A(R^{-1}G)$ .  
7: Compute  $\tilde{\Delta} = A(I - QQ^{\top})G = C - \tilde{A} (Q^{\top}G).$  
8: return  $\operatorname{Alg}3(A, i) = \tilde{\theta}_i = \| (e_i^\top B) \tilde{A} \|^2 + \| (e_i^\top B) \tilde{\Delta} \|^2$ , for all  $i \in [n]$ .

The following theorem gives approximation bounds for the leverage scores returned by Algorithm 3.

Theorem 3. Let  $A \in \mathbb{R}^{n \times d}$ ,  $\theta_{i} = \| e_{i}^{\top} AA^{\dagger} \|^{2}$  and  $\tilde{\theta}_{i} = Alg3(A, i)$ . The following hold:

$$
\left| \tilde {\theta} _ {i} - \theta_ {i} \right| \leq \left(\epsilon_ {1} + \sqrt {\epsilon_ {2}}\right) \theta_ {i}, \quad a n d \quad \left| \sum_ {i = 1} ^ {n} \tilde {\theta} _ {i} - d \right| \leq \left(\epsilon_ {1} + \epsilon_ {2}\right) d.
$$

Proof. Let  $\hat{\theta}_i = \| e_i^\top A(\Pi_1A)^\dagger\|^2$ , so that

$$
\left| \tilde {\theta} _ {i} - \theta_ {i} \right| = \left| \tilde {\theta} _ {i} - \hat {\theta} _ {i} + \hat {\theta} _ {i} - \theta_ {i} \right| \leq \left| \tilde {\theta} _ {i} - \hat {\theta} _ {i} \right| + \left| \hat {\theta} _ {i} - \theta_ {i} \right|.
$$

From [14, Lemma 9] it follows that  $|\hat{\theta}_i - \theta_i| \leq \frac{\epsilon_1}{1 - \epsilon_1} \theta_i$ . Subsequently,  $\hat{\theta}_i \leq (1 + \frac{\epsilon_1}{1 - \epsilon_1}) \theta_i = \frac{1}{1 - \epsilon_1} \theta_i$ . From Theorem 1 we recall that for appropriate  $m, l, k$ ,

$$
\left| \hat {\theta} _ {i} - \tilde {\theta} _ {i} \right| \leq \sqrt {\epsilon_ {2}} \hat {\theta} _ {i} \leq \left(\frac {\sqrt {\epsilon_ {2}}}{1 - \epsilon_ {1}}\right) \theta_ {i}.
$$

Combining all these observations we find that

$$
\left| \tilde {\theta} _ {i} - \theta_ {i} \right| \leq \left| \tilde {\theta} _ {i} - \hat {\theta} _ {i} \right| + \left| \hat {\theta} _ {i} - \theta_ {i} \right| \leq \frac {\epsilon_ {1} + \sqrt {\epsilon_ {2}}}{1 - \epsilon_ {1}} \theta_ {i} \leq 2 \left(\epsilon_ {1} + \sqrt {\epsilon_ {2}}\right) \theta_ {i}.
$$

Rescaling  $\epsilon_{1}$  and  $\epsilon_{2}$  gives the element-wise bounds. For the Frobenius norm bounds we can use similar arguments in combination with Theorem 2.

Complexity and choice of  $\epsilon_1, \epsilon_2$  The complexity of Algorithm 3 can be split into two parts: (i) the complexity of obtaining an  $\epsilon_1$ -approximate orthonormal basis for  $A$ , and (ii) the complexity of estimating the row norms of this basis. The complexity of the former has been heavily studied in the literature and depends on the choice of the subspace embedding. For very tall-and-skinny matrices, an efficient construction is to use a combination of a CountSketch [25, 10], a Subsampled Randomized

Hadamard Transform (SRHT) [32] and a Gaussian subspace embedding; see e.g. [10]. This provides a sketch  $\Pi_1A$  with dimension  $O(d / \epsilon^2)\times d$  in  $T(\Pi_1A)$  time. Computing the QR factorization of the sketch requires  $O(d^{3} / \epsilon^{2})$  to obtain  $R$ . Since  $R$  is upper triangular, the computation of  $R^{-1}G$  and  $R^{-1}Q$  both take  $O(d^{2}m)$ . The products  $A(R^{-1}Q)$  and  $A(R^{-1}G)$  cost  $O(T_{\mathrm{MM}}(A,m))$  each. The last step takes  $O(nm)$ . The total complexity is

$$
O \left(T \left(\Pi_ {1} A\right) + d ^ {3} / \epsilon_ {1} ^ {2} + T _ {\mathrm {M M}} (A, m) + n m\right).
$$

To achieve  $\epsilon$ -accuracy in the Frobenius norm, it suffices to use  $m = O(\sqrt{\log(n / \delta)} / \epsilon_2)$ . With  $\epsilon_1 = \epsilon_2 = \epsilon$ , the total complexity becomes

$$
O (T (\Pi_ {1} A) + d ^ {3} / \epsilon^ {2} + T _ {\mathrm {M M}} (A, \sqrt {\log (n / \delta)} / \epsilon) + n \sqrt {\log (n / \delta)} / \epsilon).
$$

If, instead, we use standard JL projections to estimate the row norms of  $AR^{-1}$ , the total complexity is

$$
O \left(T \left(\Pi_ {1} A\right) + d ^ {3} / \epsilon^ {2} + T _ {\mathrm {M M}} (A, \log (n / \delta) / \epsilon^ {2}) + n \log (n / \delta) / \epsilon^ {2}\right),
$$

to achieve the same Frobenius norm-wise accuracy. For any matrix which is "tall-enough" such that the  $O(T_{\mathrm{MM}}(A,\log (n / \delta) / \epsilon^2))$  factor dominates the complexity, Algorithm 3 achieves a quadratic improvement over standard estimators.

# 5 Numerical experiments

Algorithm 1 was implemented in Python using NumPy. We conducted experiments to verify the approximation guarantees and the performance improvements against standard Gaussian random projections. Following [24], we generated synthetic matrices with decay in the spectrum. Specifically,  $d \times d$  matrices  $A$ , with  $d = 5000$ , were created as follows. We drew a random orthogonal  $d \times d$  matrix  $Q$ . We then fixed a diagonal  $d \times d$  matrix  $\Lambda$  which defines the eigenvalues of the matrix. Each element  $\Lambda_{i,i}$ ,  $i \in [d]$  is set to  $i^{-c}$  for a given  $c \geq 0$ . The larger the  $c$ , the faster the spectral decay. We finally constructed the symmetric  $A = Q\Lambda Q^{\top}$  which were used in the numerical experiments. Following [24], we applied four different decay factors, specifically  $c = \{0.5, 1, 1.5, 2\}$ .

The approximation errors of standard JL projections versus Algorithm 1 are compared in Figure 1. We plot the approximation errors of both methods as the number of samples increases. We plot two types of errors, the maximum element-wise and the Frobenius norm-wise errors

$$
\max _ {i \in [ d ]} \frac {| \tilde {x} _ {i} - \| e _ {i} ^ {\top} A \| ^ {2} |}{\| e _ {i} ^ {\top} A \| ^ {2}} \quad \text {a n d} \quad \frac {\left| \tilde {X} - \| A \| _ {F} ^ {2} \right|}{\| A \| _ {F} ^ {2}},
$$

where  $\tilde{x}_i$  are the approximated row norms and  $\tilde{X}$  is their sum returned by either Algorithm 1 or standard JL projections. The exact same number of matrix vector queries is used in both methods. Standard JL projections involve only one random matrix  $G$ , which is multiplies  $A$  from the right.  $G$  has size  $d \times m$ ,  $m$  being the number of samples. In Algorithm 1, on the other hand,  $A$  is multiplied four times with a  $d \times m$  matrix. Therefore, we set  $G$ ,  $S$ , and  $Q$  in Algorithm 1 to have size  $d \times m/4$ , so that both algorithms are tested with the same number of matrix-vector products. In each plot we illustrate the mean error over 10 independent runs and the standard deviation. Standard JL approximations perform marginally better than Algorithm 1 only for the element-wise errors and only for the matrix with very slow decay. In all other cases, Algorithm 1 performs significantly better.

# 6 Conclusion

We proposed an adaptive algorithm to estimate the Euclidean row norms of a matrix  $A$ . This algorithm improves standard Johnson-Lindenstrauss estimators in the following aspects: (i) Quadratically less matrix-vector queries are required to achieve the same Frobenius norm-wise accuracy for all matrices; (ii) Asymptotically less matrix-vector queries are needed to achieve the same element-wise accuracy for matrices with decaying spectrum; (iii) At least as accurate element-wise approximations as standard JL are achieved for worst-case input matrices, that is, for matrices with flat spectrum. We also showed how these results can be applied to other important problems, specifically to estimate Euclidean distances between data points, which is related to the fundamental concept of approximate isometries that has many applications in data science, as well as for statistical leverage scores

![](images/98cb9c2d3b56a99b5e87b4fd2cd83e26ae752dfe00ac876738f0362158af781a.jpg)  
(a) Very slow eigenvalue decay  $(c = 0.5)$

![](images/cd945dc885eb9eb7e79b4c9fb5d6f949a5da2db2fd4cd345ea4a3177aa8904b4.jpg)  
(b) Slow eigenvalue decay  $(c = 1.0)$

![](images/289114ec09cb72c9f5a26371249b83bba383498f1602c1f55a452d4ebb70a252.jpg)  
(c) Moderate eigenvalue decay  $(c = 1.5)$

![](images/4f959cd2efac422bc5e38fe7c0b36fb3d016934524b56356cdc5e5971db2698d.jpg)  
Figure 1: Comparison between the element-wise (dashed curves with “ $\times$ ” marker) and norm-wise (solid curves with “star” marker) relative errors of Algorithm 1 (blue) and standard Gaussian random projections (red) versus number of matrix-vector multiplication queries (x-axis) ran on random matrices with power law spectra. The mean relative error of the approximation averaged over 10 independent runs is plotted. The upper and lower bounds around each curve represent the standard deviation. As expected, for matrices with a very slow decay standard JL projections perform marginally better with respect to the element-wise errors, but Algorithm 1 performs significantly better for all other cases.  
(d) Fast eigenvalue decay  $(c = 2.0)$

estimations, which are ubiquitous quantities not only in data science and statistics, but also in numerical linear algebra and spectral graph theory.

As future work, several directions can be envisioned. Most prominently, it would be interesting to determine whether the studied techniques can be used to improve Oblivious Subspace Embeddings [29, 33]. Such improvements would have an immediate impact in many problems in NLA, e.g. least squares regression, low-rank approximations and column subset selection. Two other relevant topics concern  $(a)$  the possibility to derive lower bounds similar to [24] for Euclidean row norms estimation and  $(b)$  to make the algorithms non-adaptive, like the non-adaptive version of Hutch++ [24] which is based on results from [9], or the Nyström++ of [26].

# References

[1] Josh Alman and Virginia Vassilevska Williams. A refined laser method and faster matrix multiplication. In Proc. 2021 ACM-SIAM SODA, pages 522-539. SIAM, 2021.  
[2] Rosa I Arriaga and Santosh Vempala. An algorithmic theory of learning: Robust concepts and random projection. Machine learning, 63(2):161-182, 2006.  
[3] Haim Avron. Counting triangles in large graphs using randomized matrix trace estimation. In Workshop on Large-scale Data Mining: Theory and Applications, volume 10, pages 10-9, 2010.

[4] Haim Avron and Sivan Toledo. Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix. JACM, 58(2):1-34, 2011.  
[5] Robert A Baston and Yuji Nakatsukasa. Stochastic diagonal estimation: probabilistic bounds and an improved algorithm. arXiv preprint arXiv:2201.10684, 2022.  
[6] Christos Boutsidis, Anastasios Zouzias, and Petros Drineas. Random projections for  $k$ -means clustering. Advances in neural information processing systems, 23, 2010.  
[7] Nadiia Chepurko, Kenneth L Clarkson, Praneeth Kacham, and David P Woodruff. Near-optimal algorithms for linear algebra in the current matrix multiplication time. In Proc. 2022 ACM-SIAM SODA, pages 3043-3068. SIAM, 2022.  
[8] Ho Yee Cheung, Tsz Chiu Kwok, and Lap Chi Lau. Fast matrix rank algorithms and applications. JACM, 60(5):1-25, 2013.  
[9] Kenneth L Clarkson and David P Woodruff. Numerical linear algebra in the streaming model. In Proc. 41st ACM STOC, pages 205-214, 2009.  
[10] Kenneth L Clarkson and David P Woodruff. Low-rank approximation and regression in input sparsity time. JACM, 63(6):1-45, 2017.  
[11] Michael B Cohen, Sam Elder, Cameron Musco, Christopher Musco, and Madalina Persu. Dimensionality reduction for k-means clustering and low rank approximation. In Proc. 47th ACM STOC, pages 163-172, 2015.  
[12] James Demmel, Ioana Dumitriu, and Olga Holtz. Fast linear algebra is stable. Numerische Mathematik, 108(1):59-91, 2007.  
[13] Michal Derezinski, Jonathan Lacotte, Mert Pilanci, and Michael W Mahoney. Newton-less: Sparsification without trade-offs for the sketched newton update. Advances in Neural Information Processing Systems, 34, 2021.  
[14] Petros Drineas, Malik Magdon-Ismail, Michael W Mahoney, and David P Woodruff. Fast approximation of matrix coherence and statistical leverage. JMLR, 13(1):3475-3506, 2012.  
[15] Petros Drineas, Michael W Mahoney, Shan Muthukrishnan, and Tamás Sarlós. Faster least squares approximation. Numerische mathematik, 117(2):219-249, 2011.  
[16] Alex Gittens and Michael W Mahoney. Revisiting the Nyström method for improved large-scale machine learning. ICML, 28:567-575, 2013.  
[17] G.H. Golub and C.F. Van Loan. Matrix Computations. Johns Hopkins Studies in the Mathematical Sciences. Johns Hopkins University Press, 2013.  
[18] Eric Hallman, Ilse CF Ipsen, and Arvind Saibaba. Monte carlo methods for estimating the diagonal of a real symmetric matrix. arXiv preprint arXiv:2202.02887, 2022.  
[19] Michael F Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 18(3):1059-1076, 1989.  
[20] William B Johnson and Joram Lindenstrauss. Extensions of Lipschitz mappings into a Hilbert space. Contemp. Math., 26(1):189-206, 1984.  
[21] Daniel M. Kane and Jelani Nelson. Sparser Johnson-Lindenstrauss Transforms. JACM, 61(1):1-23, January 2014.  
[22] Kasper Green Larsen and Jelani Nelson. Optimality of the johnson-lindenstrauss lemma. In 58th IEEE FOCS, pages 633-638. IEEE, 2017.  
[23] Michael W. Mahoney. Randomized algorithms for matrices and data. Foundations and Trends in Machine Learning, 3(2):123-224, 2011.

[24] Raphael A. Meyer, Cameron Musco, Christopher Musco, and David P. Woodruff. *Hutch++: Optimal stochastic trace estimation*. In *Symposium on Simplicity in Algorithms (SOSA)*, pages 142-155. Society for Industrial and Applied Mathematics, January 2021.  
[25] Jelani Nelson and Huy L Nguyen. Osnap: Faster numerical linear algebra algorithms via sparser subspace embeddings. In 54th IEEE FOCS, pages 117-126. IEEE, 2013.  
[26] David Persson, Alice Cortinovis, and Daniel Kressner. Improved variants of the hutch++ algorithm for trace estimation. arXiv preprint arXiv:2109.10659, 2021.  
[27] Mert Pilanci and Martin J Wainwright. Newton sketch: A near linear-time optimization algorithm with linear-quadratic convergence. SIAM J. Optim., 27(1):205-245, 2017.  
[28] Farbod Roosta-Khorasani and Uri Ascher. Improved bounds on sample size for implicit matrix trace estimators. Foundations of Computational Mathematics, 15(5):1187-1212, 2015.  
[29] Tamas Sarlos. Improved approximation algorithms for large matrices via random projections. In 47th IEEE FOCS, pages 143-152. IEEE, 2006.  
[30] Aleksandros Sobczyk and Efstratios Gallopoulos. Estimating leverage scores via rank revealing methods and randomization. SIAM J. Matrix Anal. Appl., 42(3):1199-1228, 2021.  
[31] Daniel A Spielman and Nikhil Srivastava. Graph sparsification by effective resistances. SIAM J. Comput., 40(6):1913-1926, 2011.  
[32] Joel A Tropp. Improved analysis of the subsampled randomized hadamard transform. Advances in Adaptive Data Analysis, 3(01n02):115-126, 2011.  
[33] David P. Woodruff. Sketching as a tool for numerical linear algebra. Foundations and Trends® in Theoretical Computer Science, 10(1-2):1-157, 2014.
