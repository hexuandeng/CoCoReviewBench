# STOCHASTIC OPTIMIZATION OF SORTING NETWORKS VIA CONTINUOUS RELAXATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Sorting input objects is an important step within many machine learning pipelines. However, the sorting operator is non-differentiable w.r.t. its inputs, which prohibits end-to-end gradient-based optimization. In this work, we propose a general-purpose continuous relaxation of the output of the sorting operator from permutation matrices to the set of unimodal matrices. Further, we use this relaxation to enable more efficient stochastic optimization over the combinatorially large space of permutations. In particular, we derive a reparameterized gradient estimator for the widely used Plackett-Luce family of distributions. We demonstrate the usefulness of our framework on three tasks that require learning semantic orderings of high-dimensional objects.

# 1 INTRODUCTION

In machine learning, learning to automatically sort objects is useful in many applications, such as top- $k$  losses for classification (Berrada et al., 2018), ranking documents for information retrieval (Liu et al., 2009), and multi-object target tracking in computer vision (Bar-Shalom & Li, 1995). Many such algorithms involve learning permutations of complex, high-dimensional objects, e.g., images. Such algorithms typically require learning informative representations of data prior to sorting and subsequent downstream processing. For instance, the  $k$ -nearest neighbors image classification algorithm, which orders the neighbors based on distances in the canonical pixel basis, can be highly suboptimal for classification. While gradient-based optimization methods for deep neural networks excel at representation learning, the non-differentiability of sorting w.r.t. the input representations makes it challenging to use the sort operator in conjunction with automatic differentiation tools.

In this work, we propose a relaxation to the sorting operator that is continuous everywhere and differentiable almost everywhere w.r.t. the inputs. The outcome of any sorting algorithm can be viewed as a permutation matrix, which is a square matrix with 0/1 entries such that every row and every column sums to 1. We propose a continuous relaxation of permutation matrices onto the set of unimodal row stochastic matrices. The relaxation includes a temperature knob that controls the degree of approximation, recovering a permutation matrix in the limit of zero temperature. Besides every row summing to 1, unimodal matrices additionally can be transformed to a permutation matrix that sorts the inputs by simply taking the arg max within each row. Hence, such matrices are also suitable for efficient straight-through gradient optimization (Bengio et al., 2013) that require "hard" permutation matrices for evaluating learning objectives during the forward pass.

Our second contribution extends the use of the proposed deterministic sorting relaxation to stochastic optimization over permutations. In many cases, such as latent variable models, the permutations may not be directly observed. By allowing for distributions over permutations, we can account for the uncertainty due to the unobserved permutations in a principled manner. However, the challenge with including stochastic nodes over discrete objects lies in gradient estimation with respect to the parameters of the discrete distribution. Vanilla REINFORCE estimators are impractical for most cases, or necessitate custom control variates for low-variance gradient estimation (Glasserman, 2013; Schulman et al., 2015).

In this regard, we consider the Plackett-Luce (PL) family of distributions over permutations (Plackett, 1975; Luce, 1959), a common modeling choice for ranking models. Appealingly, we can also derive a reparameterizable sampler for stochastic optimization with respect to this distribution, based on Gumbel perturbations to the distribution parameters. The reparameterized sampler, however requires a sorting operation which makes the objective non-differentiable w.r.t. these parameters. Our

proposed continuous relaxation to sorting allows us to approximate the objective and obtain well-defined reparameterized gradient estimates.

Finally, we apply this framework to tasks that require us to learn semantic orderings of the complex, high-dimensional input data. First, we consider sorting of images of handwritten digits, where the goal is to learn to sort images by their unobserved labels. Our second task extends the first one to identifying the 50-th quantile (a.k.a. the median) of a set of handwritten numbers. In addition to identifying the index of the median image in the sequence, we need to learn to map the inferred median digit to its scalar representation. In the third task, we propose an algorithm that learns a basis representation for the  $k$ -nearest neighbors (kNN) classifier in an end-to-end procedure. Because the choice of the  $k$  nearest neighbors requires a non-differentiable sorting, we use our relaxation to obtain an approximate, differentiable surrogate. On all tasks, we observe significant empirical improvements over the competing benchmarks.

# 2 PRELIMINARIES

An  $n$ -dimensional permutation  $\mathbf{z} = [z_1, z_2, \ldots, z_n]^T$  is a list of unique indices  $\{1, 2, \ldots, n\}$ . Every permutation  $\mathbf{z}$  is associated with a permutation matrix  $P_{\mathbf{z}} \in \{0, 1\}^{n \times n}$  with entries given as:

$$
P _ {\mathbf {z}} [ i, j ] = \left\{ \begin{array}{l} 1 \text {i f} j = z _ {i} \\ 0 \text {o t h e r w i s e .} \end{array} \right.
$$

Let  $\mathcal{Z}_n$  denote the set of all possible  $n!$  permutations in the symmetric group. We define the sort:  $\mathbb{R}^n\to \mathcal{Z}_n$  operator as a mapping of  $n$  real-valued inputs to a permutation corresponding to a descending ordering of these inputs. E.g., if the input vector  $\mathbf{s} = [9,1,5,2]^T$ , then  $\operatorname {sort}(\mathbf{s}) = [1,3,4,2]^T$  since the largest element is at the first index, second largest element is at the third index and so on. In case of ties, elements are assigned indices in the order they appear. We can obtain the sorted vector simply via  $P_{\mathrm{sort}(\mathbf{s})}\mathbf{s}$ .

# 2.1 PLACKETT-LUCE DISTRIBUTIONS

The family of Plackett-Luce distributions over permutations is best described via a generative process: Consider a sequence of  $n$  items, each associated with a canonical index  $i = 1,2,\ldots ,n$ . A common assumption in ranking models is that the underlying generating process for any observed permutation of  $n$  items satisfies Luce's choice axiom (Luce, 1959). Mathematically, this axiom defines the 'choice' probability of an item with index  $i$  as:  $q(i)\propto s_i$  where  $s_i > 0$  is interpreted as the score of item with index  $i$ . The normalization constant is given by  $Z = \sum_{i\in \{1,2,\dots,n\}}s_i$ .

If we choose the  $n$  items one at a time (without replacement) based on these choice probabilities, we obtain a discrete distribution over all possible permutations. This distribution is referred to as the Plackett-Luce (PL) distribution, and its probability mass function for any  $\mathbf{z} \in \mathcal{Z}_n$  is given by:

$$
q (\mathbf {z} | \mathbf {s}) = \frac {s _ {z _ {1}}}{Z} \frac {s _ {z _ {2}}}{Z - s _ {z _ {1}}} \dots \frac {s _ {z _ {n}}}{Z - \sum_ {i = 1} ^ {n - 1} s _ {z _ {i}}} \tag {1}
$$

where  $\mathbf{s} = \{s_1, s_2, \ldots, s_n\}$  is the vector of scores parameterizing this distribution (Plackett, 1975).

# 2.2 STOCHASTIC COMPUTATION GRAPHS

The abstraction of stochastic computation graphs (SCG) compactly specifies the forward value and the backward gradient computation for computational circuits. An SCG is a directed acyclic graph that consists of three kinds of nodes: input nodes which specify external inputs (including parameters), deterministic nodes which are deterministic functions of their parents, and stochastic nodes which are distributed conditionally on their parents. See Schulman et al. (2015) for a review.

To define gradients of an objective function w.r.t. any node in the graph, the chain rule necessitates that the gradients w.r.t. the intermediate nodes are well-defined. This is not the case for the sort operator. In Section 3, we will propose to extend stochastic computation graphs with nodes corresponding to a relaxation of the deterministic sort operator. In Section 4, we will further use this relaxation to extend computation graphs to include stochastic nodes corresponding to distributions over permutations. The proofs of all theoretical results in this work are deferred to Appendix B.

![](images/0c5b1b0be4b1918c4bc3765d4936fb372aa39f897ca7c09621372d9a13d9af06.jpg)  
Figure 1: Center: Venn Diagram relationships between permutation matrices  $(\mathcal{P})$ , doubly stochastic matrices  $(\mathcal{D})$ , unimodal row stochastic matrices  $(\mathcal{U})$ , and row stochastic matrices  $(\mathcal{R})$ . Left: A doubly stochastic matrix that is not unimodal. Right: A unimodal matrix that is not doubly stochastic.

# 3 THE RELAXED SORTING OPERATOR

Our goal in this section is to derive surrogates to the sort operator that have well-defined gradients w.r.t. the inputs. The general recipe to relax non-differentiable operators with discrete codomains  $\mathcal{N}$  is to consider differentiable alternatives that map the input to a larger continuous codomain  $\mathcal{M}$  with desirable properties. For gradient-based optimization, we are interested in two key properties:

1. The relaxation is continuous everywhere and differentiable (almost-)everywhere with respect to elements in the input domain.  
2. There exists a computationally efficient projection from  $\mathcal{M}$  to  $\mathcal{N}$ .

Relaxations satisfying the first requirement are amenable to the powerful machinery of automatic differentiation for optimizing stochastic computational graphs. The second requirement is useful for evaluating metrics and losses that necessarily require a discrete output akin to the one obtained from the original, non-relaxed operator. E.g., in straight-through gradient estimation (Bengio et al., 2013; Jang et al., 2017), the non-relaxed operator is used for evaluating the learning objective in the forward pass and the relaxed operator is used in the backward pass for gradient estimation.

The canonical example is the  $0/1$  loss used for binary classification. While the  $0/1$  loss is discontinuous w.r.t. real-valued model predictions, surrogates such as the logistic and hinge losses used for logistic regression and support vector machines respectively are continuous everywhere and differentiable (almost-)everywhere, and can give hard binary predictions via thresholding.

Note: For brevity, we assume that the arg max operator is applied over a set of elements with a unique maximizer and hence, the operator has well-defined semantics. With some additional bookkeeping for resolving ties, the results in this section hold even if the elements are not unique. See Appendix C.

Unimodal Row Stochastic Matrices. The sort operator maps the input vector to a permutation, or equivalently a permutation matrix. Our relaxation to sort is motivated by the geometric structure of permutation matrices. The set of permutation matrices is a subset of doubly stochastic matrices, which allow for arbitrary non-negative matrix entries such that the every row and column sums to one. If we remove the requirement that every column should sum to one, we obtain a larger set of row stochastic matrices. In this work, we propose a relaxation to sort that maps inputs to an alternate subset of row stochastic matrices, which we refer to as the unimodal row stochastic matrices.

Definition 1 (Unimodal Row Stochastic Matrices). An  $n \times n$  matrix is Unimodal Row Stochastic if it satisfies the following conditions:

1. Non-negativity:  $U[i,j] \geq 0 \quad \forall i,j \in \{1,2,\dots,n\}$ .  
2. Row Affinity:  $\sum_{j=1}^{n} U[i,j] = 1 \quad \forall i \in \{1,2,\dots,n\}$ .  
3. Argmax Permutation: Let  $\mathbf{u}$  denote an  $n$ -dimensional vector with entries such that  $u_{i} = \arg \max_{j} U[i,j] \forall i \in \{1,2,\ldots,n\}$ . Then,  $\mathbf{u} \in \mathcal{Z}_n$ .

We denote  $\mathcal{U}_n$  as the set of  $n\times n$  unimodal row stochastic matrices.

All row stochastic matrices satisfy the first two conditions. The third condition is useful for gradient based optimization involving sorting based losses. The condition provides a straightforward mechanism for extracting a permutation from a unimodal row stochastic matrix via a row-wise arg max operation. Figure 1 shows the relationships between the different subsets of square matrices.

Sorting Relaxation. Our relaxation to the sort operator is based on a standard identity for evaluating the sum of the  $k$  largest elements in any input vector.

Lemma 2. [Lemma 1 in Ogryczak & Tamir (2003)] For an input vector  $\mathbf{s} = [s_1, s_2, \ldots, s_n]^T$  that is sorted as  $s_{[1]} \geq s_{[2]} \geq \ldots \geq s_{[n]}$ , we have the sum of the  $k$ -largest elements given as:

$$
\sum_ {i = 1} ^ {k} s _ {[ i ]} = \min  _ {\lambda \in \{s _ {1}, s _ {2}, \dots , s _ {n} \}} \lambda k + \sum_ {i = 1} ^ {n} \max  (s _ {i} - \lambda , 0). \tag {2}
$$

The identity in Lemma 2 outputs the sum of the top- $k$  elements. The  $k$ -th largest element itself can be recovered by taking the difference of the sum of top- $k$  elements and the top- $(k - 1)$  elements.

Corollary 3. Let  $\mathbf{s} = [s_1, s_2, \dots, s_n]^T$  be a real-valued vector of length  $n$ . Let  $A_{\mathbf{s}}$  denote the matrix of absolute pairwise differences of the elements of  $\mathbf{s}$  such that  $A_{\mathbf{s}}[i,j] = |s_i - s_j|$ . The permutation matrix  $P_{\text{sort}(\mathbf{s})}$  corresponding to  $\text{sort}(\mathbf{s})$  is given by:

$$
P _ {\text {s o r t} (\mathbf {s})} [ i, j ] = \left\{ \begin{array}{l} 1 \text {i f} j = \arg \max  \left[ (n + 1 - 2 i) \mathbf {s} - A _ {\mathbf {s}} \mathbb {1} \right] \\ 0 \text {o t h e r w i s e} \end{array} \right. \tag {3}
$$

where  $\mathbb{1}$  denotes the column vector of all ones.

E.g., if we set  $i = \lfloor (n + 1) / 2 \rfloor$  then the non-zero entry in the  $i$ -th row  $P_{\mathrm{sort}(\mathbf{s})}[i,:]$  corresponds to the element with the minimum sum of (absolute) distance to the other elements. As desired, this corresponds to the median element. The relaxation requires  $O(n^2)$  operations to compute  $A_{\mathbf{s}}$ , as opposed to the  $O(n \log n)$  overall complexity for the best known comparator based sorting algorithms. In practice however, it is highly parallelizable and can be implemented efficiently on GPU hardware.

The arg max operator is non-differentiable which prohibits the direct use of Corollary 3 for gradient computation. Instead, we propose to replace the arg max operator with soft max to obtain a continuous relaxation  $\hat{P}_{\mathrm{sort}(\mathbf{s})}(\tau)$ . In particular, the  $i$ -th row of  $\hat{P}_{\mathrm{sort}(\mathbf{s})}(\tau)$  is given by:

$$
\widehat {P} _ {\text {s o r t} (\mathbf {s})} [ i,: ] (\tau) = \operatorname {s o f t} \max  \left[ \left(\left(n + 1 - 2 i\right) \mathbf {s} - A _ {\mathbf {s}} \mathbb {1}\right) / \tau \right] \tag {4}
$$

where  $\tau > 0$  is a temperature parameter. Our relaxation is continuous everywhere and differentiable almost everywhere with respect to the elements of  $s$ . Furthermore, we have the following result.

Theorem 4. Let  $\widehat{P}_{\text{sort}(s)}$  denote the continuous relaxation to the permutation matrix  $P_{\text{sort}(s)}$  for an arbitrary input vector  $s$  and temperature  $\tau$ . Then, we have:

1. Unimodality:  $\forall \tau > 0$ ,  $\widehat{P}_{sort(s)}$  is a unimodal row stochastic matrix. Further, let  $\mathbf{u}$  denote the permutation obtained by applying arg max row-wise to  $\widehat{P}_{sort(s)}$ . Then,  $\mathbf{u} = sort(s)$ .  
2. Limiting behavior: If we assume that the entries of  $\mathbf{s}$  are drawn independently from a distribution that is absolutely continuous w.r.t. the Lebesgue measure in  $\mathbb{R}$ , then the following convergence holds almost surely:

$$
\lim  _ {\tau \rightarrow 0 ^ {+}} \widehat {P} _ {s o r t (\mathbf {s})} [ i,: ] (\tau) = P _ {s o r t (\mathbf {s})} [ i,: ] \quad \forall i \in \{1, 2, \dots , n \}. \tag {5}
$$

Unimodality allows for efficient projection of the relaxed permutation matrix  $\widehat{P}_{\mathrm{sort}(\mathbf{s})}$  to the hard matrix  $P_{\mathrm{sort}(\mathbf{s})}$  via a row-wise arg max, e.g., for straight-through gradients. For analyzing limiting behavior, independent draws ensure that the elements of s are distinct almost surely. The temperature  $\tau$  controls the degree of smoothness of our approximation. At one extreme, the approximation becomes tighter as the temperature is reduced. In practice however, the trade-off is in the variance of these estimates, which is typically lower for larger temperatures.

# 4 STOCHASTIC OPTIMIZATION OVER PERMUTATIONS

In many scenarios, we would like the ability to specify stochastic nodes corresponding to distributions over permutations in our computation graph, e.g., latent variable models with latent nodes corresponding to permutation matrices. Consider the following stochastic optimization objective:

$$
L (\theta , \mathbf {s}) = \mathbb {E} _ {q (\mathbf {z}; \mathbf {s})} [ f (P _ {\mathbf {z}}; \theta , \mathbf {s}) ] \tag {6}
$$

![](images/4e6d2c3250c349a8ffa0677ebca8f274be96e04bb67e57a3ac6c06272e5ef88e.jpg)  
(a) Default

![](images/d603dbb6546f062d2cf167b5525f3b09c1075b71d35caef5c586a37eea1920c0.jpg)  
(b) Reparameterized (Plackett-Luce)  
Figure 2: Stochastic computation graphs with stochastic nodes corresponding to permutations. Squares denote deterministic nodes and circles denote stochastic nodes.

where  $\theta$  and  $s$  denote sets of parameters,  $P_{\mathbf{z}}$  is the permutation matrix corresponding to the permutation  $\mathbf{z}$ ,  $q(\cdot)$  is a parameterized distribution over the elements of the symmetric group  $\mathcal{Z}_n$ , and  $f(\cdot)$  is an arbitrary function of interest assumed to be differentiable in  $\theta$  and  $\mathbf{z}$ . While such objectives are typically intractable to evaluate exactly since they require summing over a combinatorially large set, we can obtain unbiased estimates efficiently via Monte Carlo. The SCG is shown in Figure 2a.

Monte Carlo estimates of gradients w.r.t.  $\theta$  can be derived simply via linearity of expectation. The gradient estimates w.r.t.  $\mathbf{s}$  cannot be obtained directly since the sampling distribution depends on  $\mathbf{s}$ . The REINFORCE gradient estimator (Glynn, 1990; Williams, 1992; Fu, 2006) uses the fact that  $\nabla_{\mathbf{s}}q(\mathbf{z};\mathbf{s}) = q(\mathbf{z};\mathbf{s})\nabla_{\mathbf{s}}\log q(\mathbf{z};\mathbf{s})$  to derive the following Monte Carlo gradient estimates:

$$
\nabla_ {\mathbf {s}} L (\theta , \mathbf {s}) = \mathbb {E} _ {q (\mathbf {z}; \mathbf {s})} \left[ f (P _ {\mathbf {z}}; \theta , \mathbf {s}) \nabla_ {\mathbf {s}} \log q (\mathbf {z}; \mathbf {s}) \right] + \mathbb {E} _ {q (\mathbf {z}; \mathbf {s})} \left[ \nabla_ {\mathbf {s}} f (P _ {\mathbf {z}}; \theta , \mathbf {s}) \right]. \tag {7}
$$

# 4.1 REPARAMETERIZED GRADIENT ESTIMATORS FOR PL DISTRIBUTIONS

REINFORCE gradient estimators typically suffer from high variance (Schulman et al., 2015; Glasserman, 2013). Reparameterized samplers provide an alternate gradient estimator by expressing samples from a distribution as a deterministic function of its parameters and a fixed source of randomness (Kingma & Welling, 2014; Rezende et al., 2014; Titsias & Lázaro-Gredilla, 2014). Since the randomness is from a fixed distribution, Monte Carlo gradient estimates can be derived by pushing the gradient operator inside the expectation (via linearity). In this section, we will derive a reparameterized sampler and gradient estimator for the Plackett-Luce (PL) family of distributions.

Let the score  $s_i$  for an item  $i \in \{1, 2, \dots, n\}$  be an unobserved random variable drawn from some underlying score distribution (Thurstone, 1927). Now for each item, we draw a score from its corresponding score distribution. Next, we generate a permutation by applying the deterministic sort operator to these  $n$  randomly sampled scores. Interestingly, prior work has shown that the resulting distribution over permutations corresponds to a PL distribution if and only if the scores are sampled independently from Gumbel distributions with identical scales.

Proposition 5. [adapted from Yellott Jr (1977)] Let  $\mathbf{s}$  be a vector of scores for the  $n$  items. For each item  $i$ , sample  $g_{i} \sim \mathrm{Gumbel}(0, \beta)$  independently with zero mean and a fixed scale  $\beta$ . Let  $\tilde{\mathbf{s}}$  denote the vector of Gumbel perturbed log-scores with entries such that  $\tilde{s}_{i} = \beta \log s_{i} + g_{i}$ . Then:

$$
q \left(\tilde {s} _ {z _ {1}} \geq \dots \geq \tilde {s} _ {z _ {n}}\right) = \frac {s _ {z _ {1}}}{Z} \frac {s _ {z _ {2}}}{Z - s _ {z _ {1}}} \dots \frac {s _ {z _ {n}}}{Z - \sum_ {i = 1} ^ {n - 1} s _ {z _ {i}}}. \tag {8}
$$

For ease of presentation, we assume  $\beta = 1$  in the rest of this work. Proposition 5 provides a method for sampling from PL distributions with parameters  $s$  by adding Gumbel perturbations to the log-scores and applying the sort operator to the perturbed log-scores. This procedure can be seen as a reparameterization trick that expresses a sample from the PL distribution as a deterministic function of the scores and a fixed source of randomness (Figure 2b). Letting  $\mathbf{g}$  denote the vector of i.i.d. Gumbel perturbations, we can express the objective in Eq. 6 as:

$$
L (\theta , \mathbf {s}) = \mathbb {E} _ {\mathbf {g}} \left[ f \left(P _ {\text {s o r t} (\log \mathbf {s} + \mathbf {g})}; \theta , \mathbf {s}\right) \right]. \tag {9}
$$

While the reparameterized sampler removes the dependence of the expectation on the parameters  $s$ , it introduces a sort operator in the computation graph such that the overall objective is non-differentiable in  $s$ . In order to obtain a differentiable surrogate, we approximate the objective based on the continuous relaxation to the sort operator proposed previously:

$$
\mathbb {E} _ {\mathbf {g}} \left[ f \left(P _ {\text {s o r t} (\log \mathbf {s} + \mathbf {g})}; \theta , \mathbf {s}\right) \right] \approx \mathbb {E} _ {\mathbf {g}} \left[ f \left(\widehat {P} _ {\text {s o r t} (\log \mathbf {s} + \mathbf {g})}; \theta , \mathbf {s}\right) \right] := \widehat {L} (\theta , \mathbf {s}). \tag {10}
$$

Accordingly, we get the following reparameterized gradient estimates for the approximation:

$$
\nabla_ {\mathbf {s}} \widehat {L} (\theta , \mathbf {s}) = \mathbb {E} _ {\mathbf {g}} \left[ \nabla_ {\mathbf {s}} f \left(\widehat {P} _ {\text {s o r t} (\log \mathbf {s} + \mathbf {g})}; \theta , \mathbf {s}\right) \right] \tag {11}
$$

which can be estimated efficiently via Monte Carlo.

![](images/b4a387e2594d3e9925c4844de80db3515e3ffdfffb2674db50e018f74b2f8f45.jpg)  
Figure 3: Illustrative example for the (a) input sequence, (b) corresponding ground-truth label for sorting, and (c) quantile (median) regression label in the large-MNIST dataset.

![](images/47e6e74873ed2330817d7c88d8b87b58c2a725525f4ae916eccc8341e9523964.jpg)  
(c) 2960.0

# 5 DISCUSSION AND RELATED WORK

The problem of learning to rank documents based on relevances has been studied extensively in the context of information retrieval. In particular, the listwise approaches learn functions that map objects to scores. Much of this work concerns the PL distribution: the RankNet algorithm (Burges et al., 2005) can be interpreted as maximizing the PL likelihood of pairwise comparisons between items, while the ListMLE ranking algorithm in Xia et al. (2008) extends this with a loss that maximizes the PL likelihood of ground-truth permutations directly. The differentiable pairwise approaches to ranking, such as Rigutini et al. (2011), learn to approximate the comparator between pairs of objects. Our work considers a generalized setting where sorting based operators can be inserted anywhere in computation graphs to extend traditional algorithmic pipelines e.g., kNN.

Recent works have proposed relaxations of permutation matrices to the Birkhoff polytope, which is defined as the set of doubly stochastic matrices (Adams & Zemel, 2011; Mena et al., 2018; Linderman et al., 2018). Adams & Zemel (2011) proposed the use of the Sinkhorn operator to map any square matrix to the Birkhoff polytope. They interpret the resulting doubly-stochastic matrix as the marginals of a distribution over permutations. Mena et al. (2018) propose an alternate method where the square matrix defines a latent distribution over the doubly-stochastic matrices themselves. These distributions can be sampled from by adding elementwise Gumbel perturbations. Linderman et al. (2018) propose a rounding procedure that uses the Sinkhorn operator to directly sample matrices near the Birkhoff polytope. Unlike Mena et al. (2018), the resulting distribution over matrices has a tractable density. In practice, however, the approach of Mena et al. (2018) performs better and will be the main baseline we will be comparing against in our experiments in Section 6.

As discussed in Section 3, our relaxation maps permutation matrices to the set of unimodal row stochastic matrices. For the stochastic setting, the PL distribution permits efficient sampling, exact and tractable density estimation, making it an attractive choice for several applications, e.g., variational inference over latent permutations. Our reparameterizable sampler, while also making use of the Gumbel distribution, is based on a result unique to the PL distribution (Proposition 5).

The use of the Gumbel distribution for defining continuous relaxations to discrete distributions was first proposed concurrently by Jang et al. (2017) and Maddison et al. (2017) for categorical variables, referred to as Gumbel-Softmax. The number of possible permutations grow factorially with the dimension, and thus any distribution over  $n$ -dimensional permutations can be equivalently seen as a distribution over  $n!$  categories. Gumbel-softmax does not scale to a combinatorially large number of categories, necessitating the use of alternate relaxations, such as the one considered in this work.

# 6 EXPERIMENTS

We refer to the two approaches proposed in Sections 3, 4 as Deterministic Sortnet and Stochastic Sortnet, respectively. For additional hyperparameter details and analysis, see Appendix D.

# 6.1 SORTING HANDWRITTEN NUMBERS

Dataset. We create the large-MNIST dataset, which extends the MNIST dataset of handwritten digits. The dataset consists of multi-digit images, each a concatenation of randomly selected individual images from MNIST. Figure 3a shows an example sequence of 5 such images.

Setup. We are given a dataset of sequences. Each sequence contains  $n$  images, and each image corresponds to an integer label. Our goal is to learn to predict the permutation that sorts these labels, given a training set of ground-truth permutations, e.g., Figure 3b. This task is a challenging extension of the one considered by Mena et al. (2018) in sorting scalars, since it involves learning the semantics of high-dimensional objects prior to sorting. A good model needs to learn to dissect the individual digits in an image, rank these digits, and finally, compose such rankings based on the digit positions within an image. The available supervision, in the form of the ground-truth permutation, is very weak compared to a classification setting that gives direct access to the image labels.

Table 1: Average sorting accuracy on the test set. First value is proportion of permutations correctly identified; value in parentheses is the proportion of individual element ranks correctly identified.  

<table><tr><td>Algorithm</td><td>n = 3</td><td>n = 5</td><td>n = 7</td><td>n = 9</td><td>n = 15</td></tr><tr><td>Vanilla RS</td><td>0.467 (0.801)</td><td>0.093 (0.603)</td><td>0.009 (0.492)</td><td>0. (0.113)</td><td>0. (0.067)</td></tr><tr><td>Sinkhorn</td><td>0.462 (0.561)</td><td>0.038 (0.293)</td><td>0.001 (0.197)</td><td>0. (0.143)</td><td>0. (0.078)</td></tr><tr><td>Gumbel-Sinkhorn</td><td>0.484 (0.575)</td><td>0.033 (0.295)</td><td>0.001 (0.189)</td><td>0. (0.146)</td><td>0. (0.078)</td></tr><tr><td>Deterministic Sortnet</td><td>0.930 (0.951)</td><td>0.837 (0.927)</td><td>0.738 (0.909)</td><td>0.649 (0.896)</td><td>0.386 (0.857)</td></tr><tr><td>Stochastic Sortnet</td><td>0.927 (0.950)</td><td>0.835 (0.926)</td><td>0.741 (0.909)</td><td>0.646 (0.895)</td><td>0.418 (0.862)</td></tr></table>

Table 2: Test mean squared error  $\left( {\times {10}^{-4}}\right)$  and  ${R}^{2}$  values (in parenthesis) for quantile regression.  

<table><tr><td>Algorithm</td><td>n = 5</td><td>n = 9</td><td>n = 15</td></tr><tr><td>Constant (Simulated)</td><td>356.79 (0.00)</td><td>227.31 (0.00)</td><td>146.94 (0.00)</td></tr><tr><td>Vanilla NN</td><td>1004.70 (0.85)</td><td>699.15 (0.82)</td><td>562.97 (0.79)</td></tr><tr><td>Sinkhorn</td><td>343.60 (0.25)</td><td>231.87 (0.19)</td><td>156.27 (0.04)</td></tr><tr><td>Gumbel-Sinkhorn</td><td>344.28 (0.25)</td><td>232.56 (0.23)</td><td>157.34 (0.06)</td></tr><tr><td>Deterministic Sortnet</td><td>45.50 (0.95)</td><td>34.98 (0.94)</td><td>34.78 (0.92)</td></tr><tr><td>Stochastic Sortnet</td><td>33.80 (0.94)</td><td>31.43 (0.93)</td><td>29.34 (0.90)</td></tr></table>

Baselines. All baselines use a CNN that is shared across all images in a sequence to map each large-MNIST image to a feature space. The vanilla row stochastic (RS) baseline concatenates the CNN representations for  $n$  images into a single vector that is fed into a multilayer perceptron that outputs  $n$  multiclass predictions of the image probabilities for each rank. The Sinkhorn and Gumbel-Sinkhorn baselines, as discussed in Section 5, use the Sinkhorn operator to map the stacked CNN representations for the  $n$  objects into a doubly stochastic matrix. For all methods, we minimized the cross-entropy loss between the predicted matrix and the ground-truth permutation matrix.

Results. Following Mena et al. (2018), our evaluation metric is the proportion of correctly predicted permutations on a test set of sequences. Additionally, we evaluate the proportion of individual elements ranked correctly. Table 1 demonstrates that the approaches based on the proposed sorting relaxation significantly outperform the baseline approaches for all  $n$  considered. The performance of the deterministic and stochastic variants are comparable. The vanilla RS baseline performs well in ranking individual elements, but is not good at recovering the overall square matrix.

We believe the poor performance of the Sinkhorn baselines is partly since these methods were designed and evaluated for matchings. Like the output of sort, matchings can also be represented as permutation matrices. However, distributions over matchings need not satisfy Luce's choice axiom or imply a total ordering, which could explain the poor performance on the tasks considered.

# 6.2 QUANTILE REGRESSION

Setup. In this experiment, we extend the sorting task to regression. Again, each sequence contains  $n$  large-MNIST images, and the regression target for each sequence is the 50-th quantile (i.e., the median) of the  $n$  labels of the images in the sequence. Figure 3c illustrates this task for one such sequence and  $n = 5$ . The design of this task highlights two key challenges since it explicitly requires learning both a suitable representation for sorting high-dimensional inputs and a secondary function that approximates the label itself (regression). Again, the supervision available in the form of the label of only a single image at an arbitrary and unknown location in the sequence is very weak.

Baselines. In addition to Sinkhorn and Gumbel-Sinkhorn, we design two more baselines. The Constant baseline always returns the median of the full range of possible outputs, ignoring the input sequence. This corresponds to 4999.5 since we are sampling large-MNIST images uniformly in the range of four-digit numbers. The vanilla neural net (NN) baseline directly maps the input sequence of images to a real-valued prediction for the median.

Results. Our evaluation metric is the mean squared error (MSE) and  $R^2$  on a test set of sequences. Results for  $n = \{5, 9, 15\}$  images are shown in Table 2. The Vanilla NN baseline while incurring a large MSE, is competitive on the  $R^2$  metric. The other baselines give comparable performance on the MSE metric. The proposed sortnet approaches outperform the competing methods on both the metrics considered. The stochastic sortnet approach is the consistent best performer on MSE, while the deterministic sortnet is slightly better on the  $R^2$  metric.

Table 3: Average test kNN classification accuracies from  $n$  neighbors for best value of  $k$  .  

<table><tr><td>Algorithm</td><td>MNIST</td><td>CIFAR-10</td></tr><tr><td>kNN</td><td>96.7%</td><td>35.4%</td></tr><tr><td>kNN+PCA</td><td>97.5%</td><td>40.9%</td></tr><tr><td>kNN+CAE</td><td>97.6%</td><td>44.2%</td></tr><tr><td>Deterministic Sortnet</td><td>98.7%</td><td>66.4%</td></tr><tr><td>Stochastic Sortnet</td><td>98.7%</td><td>66.5%</td></tr><tr><td>CNN (w/o kNN)</td><td>99.2%</td><td>77.2%</td></tr></table>

![](images/5fa8d867372a31bab08cdcad2aaf4d37b58bf9f58b12e276b3229a8668ef252c.jpg)  
Figure 4: Differentiable kNN. The model is trained to learn a feature space such that training points in  $\{x_{1}, x_{2}, \ldots, x_{n}\}$  that have the same label  $y$  are closer to  $x$  (included in top- $k$ ) than others.

# 6.3 END-TO-END, DIFFERENTIABLE  $k$ -NEAREST NEIGHBORS

Setup. In this experiment, we design a fully differentiable, end-to-end  $k$ -nearest neighbors (kNN) classifier. Unlike a standard kNN classifier which computes distances between points in a predefined space, we learn a representation of the data points before evaluating the  $k$ -nearest neighbors.

Every sequence of items here consists of a query point and a randomly sampled subset of  $n$  candidate nearest neighbors from the training set. In principle, we could use the entire training set (excluding the query point) as candidate points, but this can hurt the learning both computationally and statistically. The query points are randomly sampled from the train/validation/test sets as appropriate but the nearest neighbors are always sampled from the training set. The loss function optimizes for a representation spaces such that the top-  $k$  candidate points (with the minimum Euclidean distance to the query point) have the same label as the query point. Figure 4 illustrates the proposed algorithm.

Datasets. We consider the benchmark MNIST dataset of handwritten digits and the CIFAR-10 dataset of natural images (no data augmentation) with the canonical splits for training and testing.

Baselines. We consider kNN baselines that operate in three standard representation spaces: the canonical pixel basis, the basis specified by the top 50 principal components (PCA), a convolutional automencoder (CAE). Additionally, we experimented with  $k = 1,3,5,9$  nearest neighbors and across two distance metrics: uniform weighting of all  $k$ -nearest neighbors and weighting nearest neighbors by the inverse of their distance. For completeness, we trained a CNN using the cross-entropy loss with the same architecture as the one used for sortnet (except the final layer).

Results. We report the classification accuracies on the standard test sets in Table 3. On both datasets, the differentiable kNN classifier outperforms all the baseline kNN variants including the convolutional autoencoder approach. The performance is much closer to the accuracy of a standard CNN.

# 7 CONCLUSION

In this paper, we propose a continuous relaxation of the sorting operator to the set of unimodal row stochastic matrices. Our relaxation facilitates gradient estimation on stochastic computation graphs and can be extended to include stochastic nodes corresponding to distributions over permutations. Further, we derived a reparameterized gradient estimator for the Plackett-Luce distribution for efficient stochastic optimization. On three illustrative tasks, our proposed relaxations outperform prior work in end-to-end learning of semantic orderings of high-dimensional objects.

In the future, we would like to explore alternate relaxations to sorting as well as applications that extend widely-used algorithms such as beam search (Goyal et al., 2018). Both our relaxed sort operator and the reparameterizable sampler are easy to implement. We provide reference implementations in Tensorflow (Abadi et al., 2016) and PyTorch (Paszke et al., 2017) in Appendix A.

# REFERENCES

Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: a system for large-scale machine learning. In OSDI, 2016.  
Ryan Prescott Adams and Richard S Zemel. Ranking via sinkhorn propagation. arXiv preprint arXiv:1106.1925, 2011.  
Matej Balog, Nilesh Tripuraneni, Zoubin Ghahramani, and Adrian Weller. Lost relatives of the Gumbel trick. In International Conference on Machine Learning, 2017.  
Yaakov Bar-Shalom and Xiao-Rong Li. Multitarget-multisensor tracking: principles and techniques. 1995.  
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Leonard Berrada, Andrew Zisserman, and M Pawan Kumar. Smooth loss functions for deep top-k classification. In International Conference on Learning Representations, 2018.  
Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hul-lender. Learning to rank using gradient descent. In International Conference on Machine learning, 2005.  
Michael C Fu. Gradient estimation. *Handbooks in operations research and management science*, 13:575-616, 2006.  
B. Gao and L. Pavel. On the Properties of the Softmax Function with Application in Game Theory and Reinforcement Learning. *ArXiv e-prints*, April 2017.  
Paul Glasserman. Monte Carlo methods in financial engineering, volume 53. Springer Science & Business Media, 2013.  
Peter W Glynn. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM, 33(10):75-84, 1990.  
Kartik Goyal, Graham Neubig, Chris Dyer, and Taylor Berg-Kirkpatrick. A continuous relaxation of beam search for end-to-end training of neural sequence models. In AAAI Conference on Artificial Intelligence, 2018.  
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-softmax. In International Conference on Learning Representations, 2017.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations, 2014.  
Scott W Linderman, Gonzalo E Mena, Hal Cooper, Liam Paninski, and John P Cunningham. Reparameterizing the birkhoff polytope for variational permutation inference. In International Conference on Artificial Intelligence and Statistics, 2018.  
Tie-Yan Liu et al. Learning to rank for information retrieval. Foundations and Trends in Information Retrieval, 3(3):225-331, 2009.  
R Duncan Luce. Individual choice behavior: A theoretical analysis. Courier Corporation, 1959.  
Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations, 2017.  
Gonzalo Mena, David Belanger, Scott Linderman, and Jasper Snoek. Learning latent permutations with gumbel-sinkhorn networks. In International Conference on Learning Representations, 2018.  
Wlodzimierz Ogryczak and Arie Tamir. Minimizing the sum of the  $k$  largest functions in linear time. Information Processing Letters, 85(3):117-122, 2003.

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Robin L Plackett. The analysis of permutations. Applied Statistics, pp. 193-202, 1975.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, 2014.  
Leonardo Rigutini, Tiziano Papini, Marco Maggini, and Franco Scarselli. Sortnet: Learning to rank by a neural preference function. IEEE transactions on neural networks, 22(9):1368-1380, 2011.  
John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel. Gradient estimation using stochastic computation graphs. In Advances in Neural Information Processing Systems, 2015.  
Louis L Thurstone. A law of comparative judgment. Psychological review, 34(4):273, 1927.  
Michalis Titsias and Miguel Lázaro-Gredilla. Doubly stochastic variational Bayes for non-conjugate inference. In International Conference on Machine Learning, 2014.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.  
Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, and Hang Li. Listwise approach to learning to rank: theory and algorithm. In International Conference on Machine Learning, 2008.  
John I Yellott Jr. The relationship between luce's choice axiom, thurstone's theory of comparative judgment, and the double exponential distribution. Journal of Mathematical Psychology, 15(2): 109-144, 1977.
