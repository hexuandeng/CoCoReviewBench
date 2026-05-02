# ADAPTIVE SINGLE-PASS STOCHASTIC GRADIENT DESCENT IN INPUT SPARSITY TIME

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study sampling algorithms for variance reduction methods for stochastic optimization. Although stochastic gradient descent (SGD) is widely used for large scale machine learning, it sometimes experiences slow convergence rates due to the high variance from uniform sampling. In this paper, we introduce an algorithm that approximately samples a gradient from the optimal distribution for a common finite-sum form with  $n$  terms, while just making a single pass over the data, using input sparsity time, and  $\tilde{\mathcal{O}}(Td)$  space. Our algorithm can be implemented in big data models such as the streaming and distributed models. Moreover, we show that our algorithm can be generalized to approximately sample Hessians and thus provides variance reduction for second-order methods as well. We demonstrate the efficiency of our algorithm on large-scale datasets.

# 1 INTRODUCTION

There has recently been tremendous progress in variance reduction methods for stochastic gradient descent (SGD) methods for the standard convex finite-sum form optimization problem  $\min_{\mathbf{x} \in \mathbb{R}^d} F(\mathbf{x}) := \frac{1}{n} \sum_{i=1}^{n} f_i(\mathbf{x})$ , where  $f_1, \ldots, f_n: \mathbb{R}^d \to \mathbb{R}$  is a set of convex functions that commonly represent loss functions. Whereas gradient descent (GD) performs the update rule  $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta_t \nabla F(\mathbf{x}_t)$  on the iterative solution  $\mathbf{x}_t$  at iterations  $t = 1, 2, \ldots$ , SGD (Robbins & Monro, 1951; Nemirovsky & Yudin, 1983; Nemirovski et al., 2009) picks  $i_t \in [n]$  in iteration  $t$  with probability  $p_{i_t}$  and performs the update rule  $\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\eta_t}{np_{i_t}} \nabla f_{i_t}(\mathbf{x}_t)$ , where  $\nabla f_{i_t}$  is the gradient (or a subgradient) of  $f_{i_t}$  and  $\eta_t$  is some predetermined learning rate. Effectively, training example  $i_t$  is sampled with probability  $p_{i_t}$  and the model parameters are updated using the selected example.

The SGD update rule only requires the computation of a single gradient at each iteration and provides an unbiased estimator to the full gradient, compared to GD, which evaluates  $n$  gradients at each iteration and is prohibitively expensive for large  $n$ . However, since SGD is often performed with uniform sampling so that the probability  $p_{i,t}$  of choosing index  $i \in [n]$  at iteration  $t$  is  $p_{i,t} = \frac{1}{n}$  at all times, the variance introduced by the randomness of sampling a specific vector function can be a bottleneck for the convergence rate of the iterative process. Thus the subject of variance reduction beyond uniform sampling has been well-studied in recent years (Roux et al., 2012; Johnson & Zhang, 2013; Defazio et al., 2014; Reddi et al., 2015; Zhao & Zhang, 2015; Daneshmand et al., 2016; Needell et al., 2016; Stich et al., 2017; Johnson & Guestrin, 2018; Katharopoulos & Fleuret, 2018; Salehi et al., 2018; Qian et al., 2019).

A common technique to reduce variance is importance sampling, where the probabilities  $p_{i,t}$  are chosen so that vector functions with larger gradients are more likely to be sampled. Thus for  $\mathrm{Var}(\mathbf{v}) \coloneqq \mathbb{E}\left[\| \mathbf{v}\| _2^2\right] - \| \mathbb{E}\left[\mathbf{v}\right]\| _2^2$ , for a random vector  $\mathbf{v}$ , then  $p_{i,t} = \frac{1}{n}$  for uniform sampling implies

$$
\sigma_ {t} ^ {2} = \mathrm {V a r} \left(\frac {1}{n p _ {i _ {t} , t}} \nabla f _ {i _ {t}}\right) = \frac {1}{n ^ {2}} \left(n \sum_ {i = 1} ^ {n} \| \nabla f _ {i} (\mathbf {x} _ {t}) \| ^ {2} - \| \nabla F (\mathbf {x} _ {t}) \| ^ {2}\right),
$$

whereas importance sampling with  $p_{i,t} = \frac{\|\nabla f_i(\mathbf{x}_t)\|}{\sum_{j=1}^{n}\|\nabla f_j(\mathbf{x}_t)\|}$  gives

$$
\sigma_ {t} ^ {2} = \mathrm {V a r} \left(\frac {1}{n p _ {i _ {t} , t}} \nabla f _ {i _ {t}}\right) = \frac {1}{n ^ {2}} \left(\left(\sum_ {i = 1} ^ {n} \| \nabla f _ {i} (\mathbf {x} _ {t}) \|\right) ^ {2} - \| \nabla F (\mathbf {x} _ {t}) \| ^ {2}\right),
$$

which is at most  $\frac{1}{n^2} \left( n \sum \| \nabla f_i(\mathbf{x}_t) \|^2 - \| \nabla F(\mathbf{x}_t) \|^2 \right)$ , by the Root-Mean Square-Arithmetic Mean Inequality, and can be significantly less. Hence the variance at each step is reduced, possibly substantially, e.g., Example A.1 and Example A.2, by performing importance sampling instead of uniform sampling. In fact, it follows from the Cauchy-Schwarz inequality that the above importance sampling probability distribution is the optimal distribution for variance reduction. However, computing the probability distribution for importance sampling requires computing the gradients in each round, which is too expensive in the first place.

Second-Order Methods. Although first-order methods such as SGD are widely used, they do sometimes have issues such as sensitivity to the choice of hyperparameters, stagnation at high training errors, and difficulty in escaping saddle points. By considering second-order information such as curvature, second-order optimization methods are known to be robust to several of these issues, such as ill-conditioning. For example, Newton's method can achieve a locally super-linear convergence rate under certain conditions, independent of the problem. Although naive second-order methods are generally too slow compared to common first-order methods, stochastic Newton-type methods such as Gauss-Newton have shown to be scalable in the scientific computing community (Roosta-Khorasani et al., 2014; Roosta-Khorasani & Mahoney, 2016a;b; Xu et al., 2019; 2020).

Our Contributions. We give a time efficient algorithm that provably approximates the optimal importance sampling using a small space data structure. Remarkably, our data structure can be implemented in big data models such as the streaming model, which just takes a single pass over the data, and the distributed model, which requires just a single round of communication between parties holding each loss function. For  $\nabla F = \frac{1}{n}\sum \nabla f_i(\mathbf{x})$ , where each  $\nabla f_{i} = f(\langle \mathbf{a}_{i},\mathbf{x}\rangle)\cdot \mathbf{a}_{i}$  for some polynomial  $f$  and vector  $\mathbf{a}_i\in \mathbb{R}^d$ , let  $\mathrm{nnz}(\mathbf{A})$  be the number of nonzero entries of  $\mathbf{A}\coloneqq \mathbf{a}_1\circ \dots \circ \mathbf{a}_n^1$ . Thus for  $T$  iterations, where  $d\ll T\ll n$ , GD has runtime  $\tilde{\mathcal{O}} (T\cdot \mathrm{nnz}(\mathbf{A}))$  while our algorithm has runtime  $T\cdot \mathrm{poly}(d,\log n) + \tilde{\mathcal{O}} (\mathrm{nnz}(\mathbf{A}))$ , where we use  $\tilde{\mathcal{O}} (\cdot)$  to suppress polylogarithmic terms.

Theorem 1.1 Let  $\nabla F = \frac{1}{n}\sum \nabla f_{i}(\mathbf{x})$ , where each  $\nabla f_{i} = f(\langle \mathbf{a}_{i},\mathbf{x}\rangle)\cdot \mathbf{a}_{i}$  for some polynomial  $f$  and vector  $\mathbf{a}_i\in \mathbb{R}^d$  and let  $\mathrm{nnz}(\mathbf{A})$  be the number of nonzero entries of  $\mathbf{A}\coloneqq \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$ . For  $d\ll T\ll n$ , there exists an algorithm that performs  $T$  steps of SGD and at each step samples a gradient within a constant factor of the optimal probability distribution. The algorithm requires a single pass over  $\mathbf{A}$  and uses  $\tilde{\mathcal{O}}$  (nnz(A)) pre-processing time and  $\tilde{\mathcal{O}}$  ( $Td$ ) space.

Theorem 1.1 can be used to immediately obtain improved convergence guarantees for a class of functions whose convergence rate depends on the variance  $\sigma_t^2$ , such as  $\mu$ -smooth functions or strongly convex functions. Unlike a number of previous variance reduction methods, we do not require distributional assumptions (Bouchard et al., 2015; Frostig et al., 2015; Gopal, 2016; Jothimirugesan et al., 2018) or offline access to the data (Roux et al., 2012; Johnson & Zhang, 2013; Defazio et al., 2014; Reddi et al., 2015; Zhao & Zhang, 2015; Daneshmand et al., 2016; Needell et al., 2016; Stich et al., 2017; Johnson & Guestrin, 2018; Katharopoulos & Fleuret, 2018; Salehi et al., 2018; Qian et al., 2019). On the other hand, for applications such as neural nets in which the parameters in the loss function can change, we can use a second-order approximation for a number of iterations, then reread the data to build a new second-order approximation when necessary.

We complement our main theoretical result with empirical evaluations comparing our algorithm to SGD with uniform sampling for logistic regression on the a9a Adult dataset collected by UCI and retrieved from LibSVM (Chang & Lin, 2011). Our evaluations demonstrate that for various step-sizes, our algorithm has significantly better performance than uniform sampling across both the number of SGD iterations and surprisingly, wall-clock time.

We then show that our same framework can also be reworked to approximate importance sampling for the Hessian, thereby performing variance reduction for second-order optimization methods. (Xu et al., 2016) reduce the bottleneck of many second-order optimization methods to the task of sampling  $s$  rows of  $\mathbf{A} = \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$  so that a row  $\mathbf{a}_i$  is sampled with probability  $\frac{\left\|f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\cdot\mathbf{a}_i^\top\mathbf{a}_i\right\|_F^2}{\sum_{i = 1}^n\left\|f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\cdot\mathbf{a}_i^\top\mathbf{a}_i\right\|_F^2}$ , for some fixed function  $f$  so that the Hessian  $\mathbf{H}$  has the form  $\mathbf{H} := \nabla^2 F = \frac{1}{n}\sum \nabla f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\mathbf{a}_i^\top\mathbf{a}_i$ . (Xu et al., 2016) show that this finite-sum form arises frequently in machine learning problems such as logistic regression with least squares loss.

Theorem 1.2 Let  $\nabla^2 F = \frac{1}{n}\sum \nabla f_i(\mathbf{x})$ , where each  $\nabla f_{i} = f(\langle \mathbf{a}_{i},\mathbf{x}\rangle)\cdot \mathbf{a}_{i}^{\top}\mathbf{a}_{i}$  for some polynomial  $f$  and vector  $\mathbf{a}_i\in \mathbb{R}^d$  and let  $\mathrm{nnz}(\mathbf{A})$  be the number of nonzero entries of  $\mathbf{A}\coloneqq \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$ . For  $d\ll T\ll n$ , there exists an algorithm that subsamples  $T$  Hessian within a constant factor of the optimal probability distribution. The algorithm requires a single pass over  $\mathbf{A}$  and uses  $\tilde{\mathcal{O}}$  (nnz(A)) pre-processing time and  $\tilde{\mathcal{O}}$  ( $Td$ ) space.

# 2 SGD ALGORITHM

We first introduce a number of algorithms that will be used in our final SGD algorithm, along with their guarantees. We defer all formal proofs to the appendix.

$L_{2}$  polynomial inner product sketch. For a fixed polynomial  $f$ , we first require a constant-factor approximation to  $\|\sum_{i=1}^{n} f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2$  for any query  $\mathbf{x} \in \mathbb{R}^d$ ; we call such an algorithm an  $L_{2}$  polynomial inner product sketch and give such an algorithm with the following guarantee:

Theorem 2.1 For a fixed  $\epsilon >0$  and polynomial  $f$ , there exists a data structure ESTIMATOR that outputs a  $(1 + \epsilon)$ -approximation to  $\sum_{i = 1}^{n}\| f(\langle \mathbf{a}_i,\mathbf{x}\rangle)\cdot \mathbf{a}_i\| _2^2$  for any query  $\mathbf{x}\in \mathbb{R}^d$ . The data structure requires a single pass over  $\mathbf{A} = \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$  (possibly through turnstile updates²), can be built in  $\tilde{\mathcal{O}}\left(\mathrm{nnz}(\mathbf{A}) + \frac{d}{\epsilon^2}\right)$  time and  $\tilde{\mathcal{O}}\left(\frac{d}{\epsilon^2}\right)$  space, uses query time poly  $\left(d,\frac{1}{\epsilon},\log n\right)$ , and succeeds with probability  $1 - \frac{1}{\mathrm{poly}(n)}$ .

The  $L_{2}$  polynomial inner product sketch ESTIMATOR is a generalization of AMS variants Alon et al. (1999); Mahabadi et al. (2020) and is simple to implement. For intuition, observe that for  $d = 1$  and the identity function  $f$ , the matrix  $\mathbf{A} \in \mathbb{R}^{n \times d}$  reduces to a vector of length  $n$  so that estimating  $\sum_{i=1}^{n} \|f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2^2$  is just estimating the squared norm of a vector in sublinear space. For a degree  $p$  polynomial  $f$ , ESTIMATOR generates random sign matrices  $\mathbf{S}_0, \mathbf{S}_1, \ldots, \mathbf{S}_p$  with  $\tilde{\mathcal{O}}\left(\frac{1}{\epsilon^2}\right)$  rows and maintains  $\mathbf{S}_0\mathbf{A}, \ldots, \mathbf{S}_p\mathbf{A}$ . To estimate  $\sum_{i=1}^{n} \| \alpha_q \cdot (\langle \mathbf{a}_i, \mathbf{x} \rangle)^q \cdot \mathbf{a}_q\|_2^2$  for an integer  $q \in [0, p]$  and scalar  $\alpha_q$  on a given query  $\mathbf{x}$ , ESTIMATOR creates the  $q$ -fold tensor  $\mathbf{Y} = \mathbf{y}^{\otimes q}$  for each row  $\mathbf{y}$  of  $\mathbf{S}_q\mathbf{A}$  and the  $(q-1)$ -fold tensor  $\mathbf{X} = \mathbf{x}^{\otimes (q-1)}$ . Note that  $\mathbf{X}$  and  $\mathbf{Y}$  can be refolded into dimensions  $\mathbb{R}^{d^q-1}$  and  $\mathbb{R}^{d \times d^{q-1}}$  so that  $\mathbf{Y}\mathbf{X} \in \mathbb{R}^d$  and  $\| \alpha_q \cdot \mathbf{Y}\mathbf{X}\|_2^2$  is an unbiased estimator of  $\sum_{i=1}^{n} \| \alpha_q \cdot (\langle \mathbf{a}_i, \mathbf{x} \rangle)^q \cdot \mathbf{a}_q\|_2^2$ . We give this algorithm in full in Algorithm 1. Thus, taking the average over  $\mathcal{O}\left(\frac{1}{\epsilon^2}\right)$  instances of the sums of the tensor products for rows  $\mathbf{y}$  across the sketches  $\mathbf{S}_0\mathbf{A}, \ldots, \mathbf{S}_p\mathbf{A}$  gives a  $(1+\epsilon)$ -approximation to  $\sum_{i=1}^{n} \| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2^2$  with constant probability. The success probability of success can then be boosted to  $1 - \frac{1}{\mathrm{poly}(n)}$  by taking the median of  $\mathcal{O}(\log n)$  such outputs.

$L_{2}$  polynomial inner product sampler. Given a matrix  $\mathbf{A} = \mathbf{a}_1\circ \dots \circ \mathbf{a}_n\in \mathbb{R}^{n\times d}$  and a fixed function  $f$ , a data structure that takes query  $\mathbf{x}\in \mathbb{R}^d$  and outputs an index  $i\in [n]$  with probability roughly

$$
\frac {\left\| f \left(\langle \mathbf {a} _ {i} , \mathbf {x} \rangle\right) \cdot \mathbf {a} _ {i} \right\| _ {2} ^ {2}}{\sum_ {i = 1} ^ {n} \left\| f \left(\langle \mathbf {a} _ {i} , \mathbf {x} \rangle\right) \cdot \mathbf {a} _ {i} \right\| _ {2} ^ {2}}
$$

is called an  $L_{2}$  polynomial inner product sampler. We give such a data structure in Section A.1:

Algorithm 1 Basic algorithm ESTIMATOR that outputs  $(1 + \epsilon)$ -approximation to  $\sum_{i=1}^{n} \left\| (\langle \mathbf{a}_i, \mathbf{x} \rangle)^p \cdot \mathbf{a}_i \right\|_2^2$ , where  $\mathbf{x}$  is a post-processing vector  
Input: Matrix  $\mathbf{A} = \mathbf{a}_1\circ \dots \circ \in \mathbb{R}^{n\times d}$  , post-processing vector  $\mathbf{x}\in \mathbb{R}^d$  , integer  $p\geq 0$  , constant parameter  $\epsilon >0$    
Output:  $(1 + \epsilon)$  -approximation to  $\sum_{i = 1}^{n}\| (\langle \mathbf{a}_{i},\mathbf{x}\rangle)^{p}\cdot \mathbf{a}_{i}\|_{2}^{2}$    
1:  $r\gets \Theta (\log n)$  with a sufficiently large constant.   
2:  $b\gets \Omega \left(\frac{1}{\epsilon^2}\right)$  with a sufficiently large constant.   
3: Let  $\mathcal{T}$  be an  $r\times b$  table of buckets, where each bucket stores an  $\mathbb{R}^d$  vector, initialized to the zeros vector.   
4: Let  $s_i\in \{-1, + 1\}$  be 4-wise independent for  $i\in [n]$    
5: Let  $h_i:[n]\to [b]$  be 4-wise independent for  $i\in [r]$    
6: Let  $\mathbf{u}_{i,j}$  be the all zeros vector for each  $i\in [r],j\in [b]$    
7: for each  $j = 1$  to  $n$  do   
8: for each  $i = 1$  to  $r$  do   
9: Add  $s_j\mathbf{a}_j$  to the vector in bucket  $h_i(j)$  of row i.   
10: Let  $\mathbf{v}_{i,j}$  be the vector in row i, bucket  $j$  of  $\mathcal{T}$  for  $i\in [r],j\in [b]$    
11: Process x:   
12: for  $i\in [r],j\in [b]$  do   
13:  $\mathbf{u}_{i,j}\gets \mathbf{v}_{i,j}^{\otimes p}\mathbf{x}^{\otimes (p - 1)}$    
14: return median  $_{i\in [r]}\frac{1}{b}\sum_{j\in [b]}\| \mathbf{u}_{i,j}\| _2^2.$

Theorem 2.2 For a fixed  $\epsilon >0$  and polynomial  $f$ , there exists a data structure SAMPLER that takes any query  $\mathbf{x}\in \mathbb{R}^d$  and outputs an index  $i\in [n]$  with probability  $\frac{(1\pm\epsilon)\cdot\|f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\cdot\mathbf{a}_i\|_2^2}{\sum_{i = 1}^n\|f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\cdot\mathbf{a}_i\|_2^2} +\frac{1}{\mathrm{poly}(n)}$ , along with a vector  $\mathbf{u}\coloneqq f(\langle \mathbf{a}_i,\mathbf{x}\rangle)\cdot \mathbf{a}_i + \mathbf{v}$ , where  $\mathbb{E}[\mathbf{v}] = 0$  and  $\| \mathbf{v}\| _2\leq \epsilon \cdot \| f(\langle \mathbf{a}_i,\mathbf{x}\rangle)\cdot \mathbf{a}_i\| _2$ . The data structure requires a single pass over  $\mathbf{A} = \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$  (possibly through turnstile updates), can be built in  $\tilde{\mathcal{O}}\left(\mathrm{nnz}(\mathbf{A}) + \frac{d}{\epsilon^2}\right)$  time and  $\tilde{\mathcal{O}}\left(\frac{d}{\epsilon^2}\right)$  space, uses query time poly  $(d,\frac{1}{\epsilon},\log n)$ , and succeeds with probability  $1 - \frac{1}{\mathrm{poly}(n)}$ .

We remark that  $T$  independent instances of SAmPLER provide an oracle for  $T$  steps of SGD with importance sampling, but the overall runtime would be  $T\cdot \mathrm{mz}(\mathbf{A})$  so it would be just as efficient to run  $T$  iterations of GD. The subroutine SAmPLER is significantly more challenging to describe and analyze, so we defer its discussion to Section A.1, though it can be seen as a combination of ESTIMATOR and a generalized CountSketch Charikar et al. (2004); Nelson & Nguyen (2013); Mahabadi et al. (2020) variant and is nevertheless relatively straightforward to implement.

Leverage score sampler. Although SAmPLER outputs a (noisy) vector according to the desired probability distribution, we also require an algorithm that automatically does this for indices  $i \in [n]$  that are likely to be sampled multiple times across the  $T$  iterations. Equivalently, we require explicitly storing the rows with high leverage scores, but we defer the formal discussion and algorithmic presentation to Section A.2. For our purposes, the following suffices:

Theorem 2.3 There exists an algorithm LEVERAGE that returns all indices  $i \in [n]$  such that  $\frac{(1 \pm \epsilon) \cdot \| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2^2}{\sum_{i=1}^{n} \| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2^2} \geq \frac{1}{200Td}$  for some  $\mathbf{x} \in \mathbb{R}^n$ , along with a vector  $\mathbf{u}_i := f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i + \mathbf{v}_i$ , where  $\|\mathbf{v}_i\|_2 \leq \epsilon$  and  $\|f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2$ . The algorithm requires a single pass over  $\mathbf{A} = \mathbf{a}_1 \circ \ldots \circ \mathbf{a}_n$  (possibly through turnstile updates), uses  $\tilde{\mathcal{O}}\left(\mathrm{nnz}(\mathbf{A}) + \frac{d^\omega}{\epsilon^2}\right)$  runtime (where  $\omega$  denotes the exponent of square matrix multiplication) and  $\tilde{\mathcal{O}}\left(\frac{d}{\epsilon^2}\right)$  space, and succeeds with probability  $1 - \frac{1}{\mathrm{poly}(n)}$ .

# 2.1 SGD ALGORITHM AND ANALYSIS

For the finite-sum optimization problem  $\min_{\mathbf{x} \in \mathbb{R}^d} F(\mathbf{x}) \coloneqq \frac{1}{n} \sum_{i=1}^{n} f_i(\mathbf{x})$ , where each  $\nabla f_i = f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i$ , recall that we could simply an instance of SAMPLER as an oracle for SGD with importance sampling. However, naively running  $T$  SGD steps requires  $T$  independent instances, which uses

$T \cdot \mathrm{nnz}(\mathbf{A})$  runtime by Theorem 2.2. Thus we use a two level data structure by first implicitly partition the rows of matrix  $\mathbf{A} = \mathbf{a}_1 \circ \ldots \circ \mathbf{a}_n$  into  $\beta := \Theta(Td)$  buckets  $B_1, \ldots, B_\beta$  and creating an instance of ESTIMATOR and SAMPLER for each bucket. The idea is that for a given query  $\mathbf{x}_t$  in SGD iteration  $t \in [T]$ , we first query  $\mathbf{x}_t$  to each of the ESTIMATOR data structures to estimate  $\sum_{i \in B_j} \|f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i\|_2^2$  for each  $j \in [\beta]$ . We then sample index  $j \in [\beta]$  among the buckets  $B_1, \ldots, B_\beta$  with probability roughly  $\frac{\sum_{i \in B_j} \|f(\langle \mathbf{a}_i, \mathbf{x}_t \rangle) \cdot \mathbf{a}_i\|_2^2}{\sum_{i=1}^n \|f(\langle \mathbf{a}_i, \mathbf{x}_t \rangle) \cdot \mathbf{a}_i\|_2^2}$ . Once we have sampled index  $j$ , it would seem that querying the instance SAMPLER corresponding to  $B_j$  simulates SGD, since SAMPLER now performs importance sampling on the rows in  $B_j$ , which gives the correct overall probability distribution for each row  $i \in [n]$ . Moreover, SAMPLER has runtime proportional to the sparsity of  $B_j$ , so the total runtime across the  $\beta$  instances of SAMPLER is  $\tilde{\mathcal{O}}(\mathrm{nnz}(\mathbf{A}))$ .

However, an issue arises when the same bucket  $B_{j}$  is sampled multiple times, as we only create a single instance of SAMPLER for each bucket. We avoid this issue by explicitly accounting for the buckets that are likely to be sampled multiple times. Namely, we show that if  $\frac{\|f(\langle\mathbf{a}_i,\mathbf{x}_t\rangle)\cdot\mathbf{a}_i\|_2^2}{\sum_{i = 1}^n\|f(\langle\mathbf{a}_i,\mathbf{x}_t\rangle)\cdot\mathbf{a}_i\|_2^2} < \frac{1}{200Td}$  for all  $t\in [T]$  and  $i\in [n]$ , then by Bernstein's inequality, the probability that no bucket  $B_{j}$  is sampled multiple times is at least  $\frac{99}{100}$ . Thus we use LEVERAGE to separate all such rows  $\mathbf{a}_i$  that violate this property from their respective buckets and explicitly track the SGD steps in which these rows are sampled. We give the algorithm in full in Algorithm 2.

Algorithm 2 Approximate SGD with Importance Sampling  
Input: Matrix  $\mathbf{A} = \mathbf{a}_1\circ \dots \circ \mathbf{a}_n\in \mathbb{R}^{n\times d}$  , parameter  $T$  for number of SGD steps.   
Output:  $T$  gradient directions.   
1: Preprocessing Stage:   
2:  $\beta \gets \Theta (Td)$  with a sufficiently large constant.   
3: Let  $h:[n]\to [\beta ]$  be a uniformly random hash function.   
4: Let  $\mathbf{B}_j$  be the matrix formed by the rows  $\mathbf{a}_i$  of A with  $h(i) = j$  , for each  $j\in [\beta ]$    
5: Create an instance ESTIMATOR and SAMPLER for each  $\mathbf{B}_j$  with  $j\in [\beta ]$  with  $\epsilon = \frac{1}{2}$    
6: Run LEVERAGE to find a set  $L_{0}$  of row indices and corresponding (noisy) vectors.   
7: Gradient Descent Stage:   
8: Randomly pick starting location  $\mathbf{x}_0$    
9: for  $t = 1$  to  $T$  do   
10: Let  $q_{i}$  be the output of ESTIMATOR on query  $\mathbf{x}_{t - 1}$  for each  $i\in [\beta ]$    
11: Sample  $j\in [\beta ]$  with probability  $p_j = \frac{q_j}{\sum_{i\in[\beta]}q_i}$    
12: if there exists  $i\in L_0$  with  $h(i) = j$  then   
13: Use ESTIMATOR, LEVERAGE, and SAMPLER to sample gradient  $\mathbf{w}_t = \widehat{\nabla f_{it}(\mathbf{x}_t)}$    
14: else   
15: Use SAMPLER to sample gradient  $\mathbf{w}_t = \widehat{\nabla f_{it}(\mathbf{x}_t)}$    
16:  $\widehat{p_{i,t}}\gets \frac{\|\mathbf{w}_t\|_2^2}{\sum_{j\in[\beta]}q_j}$    
17:  $\mathbf{x}_{t + 1}\leftarrow \mathbf{x}_t - \frac{\eta_t}{n\bar{p}_{i,t}}\cdot \mathbf{w}_t$

The key property achieved by Algorithm 2 in partitioning the rows and removing the rows that are likely to be sampled multiple times is that each of the SAMPLER instances are queried at most once.

Lemma 2.4 With probability at least  $\frac{98}{100}$ , each  $t \in [T]$  uses a different instance of  $\mathrm{SAMPLER}_j$ .

Proof of Theorem 1.1: Consider Algorithm 2. By Lemma 2.4, each time  $t \in [T]$  uses a fresh instance of  $\mathrm{SAMPLER}_j$ , so that independent randomness is used. A possible concern is that each instance  $\mathrm{ESTIMATOR}_j$  is not using fresh randomness, but we observe that ESTIMATOR procedures is only used in sampling a bucket  $j \in [\beta]$  as an  $L_2$  polynomial inner product sketch; otherwise the sampling uses fresh randomness whereas the sampling is built into each instance of  $\mathrm{SAMPLER}_j$ . By Theorem 2.2, each index  $i$  is sampled with probability within a factor 2 of the importance sampling probability distribution. By Theorem 2.1, we have that  $\widehat{p_{i,t}}$  is within a factor 4 of the probability  $p_{i,t}$  induced by optimal importance sampling SGD. Note that  $\mathbf{w}_t = \nabla f_i(\mathbf{x}_t)$  is an

unbiased estimator of  $\nabla f_{i}(\mathbf{x}_{t})$  and  $\| \mathbf{w}_t\|$  is a 2-approximation to  $\| \nabla f_i(\mathbf{x}_t)\|$  by Theorem 2.2. Hence, the variance at each time  $t\in [T]$  of Algorithm 2 is within a constant factor of the variance  $\sigma^2 = (\sum \| \nabla f_i(\mathbf{x}_t)\|)^2 -\| \nabla F(\mathbf{x}_t)\|^2$  of optimal importance sampling SGD.

By Theorem 2.1, Theorem 2.2, and Theorem 2.3, the preprocessing time is  $\tilde{\mathcal{O}} (\mathrm{nnz}(\mathbf{A})) + T\cdot$  poly  $(d,\log n)$  due to the choices of  $\epsilon = \mathcal{O}(1)$  and  $\beta = \Theta (Td)$ , but partitioning the nonzero entries of  $\mathbf{A}$  across the  $\beta$  buckets. Similarly, the space used by the algorithm is  $\tilde{\mathcal{O}} (Td)$ . Once the gradient descent stage of Algorithm 2 begins, it takes poly  $(d)$  time in each step  $t\in [T]$  to query the  $\beta = \Theta (Td)$  instances of SAMPLER and ESTIMATOR, for total time  $T\cdot \mathrm{poly}(d,\log n)$ .

# 3 SECOND-ORDER OPTIMIZATION

In this section, we repurpose our data structure that performs importance sampling for SGD to instead perform importance sampling for second-order optimization. Given a second-order optimization algorithm that requires a sampled Hessian  $\mathbf{H}_t$ , possibly along with additional inputs such as the current iterate  $\mathbf{x}_t$  and the gradient  $\mathbf{g}_t$  of  $F$ , we model the update rule by an oracle  $\mathbb{O}(\mathbf{H}_t)$ , suppressing other inputs to the oracle in the notation. For example, the oracle  $\mathbb{O}$  corresponding to the canonical second-order algorithm Newton's method can be formulated as

$$
\mathbf {x} _ {t + 1} = \mathbb {O} (\mathbf {x} _ {t}) := \mathbf {x} _ {t} - [ \mathbf {H} _ {t} ] ^ {- 1} \mathbf {g} _ {t}.
$$

By black-boxing the update rule of any second-order optimization algorithm into the oracle, we can focus our attention to the running time of sampling a Hessian with nearly the optimal probability distribution. Thus we prove generalizations of the  $L_{2}$  polynomial inner product sketch, the  $L_{2}$  polynomial inner product sampler, and the leverage score sampler for Hessians.

Theorem 3.1 For a fixed  $\epsilon >0$  and polynomial  $f$ , there exists a data structure HESTIMATOR that outputs a  $(1 + \epsilon)$ -approximation to  $\sum_{i = 1}^{n}\left\| f(\langle \mathbf{a}_i,\mathbf{x}\rangle)\cdot \mathbf{a}_i^\top \mathbf{a}_i\right\| _F^2$  for any query  $\mathbf{x}\in \mathbb{R}^d$ . The data structure requires a single pass over  $\mathbf{A} = \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$  (possibly through turnstile updates), can be built in  $\tilde{\mathcal{O}}\left(\mathrm{nnz}(\mathbf{A}) + \frac{d}{\epsilon^2}\right)$  time and  $\tilde{\mathcal{O}}\left(\frac{d}{\epsilon^2}\right)$  space, uses query time poly  $\left(d,\frac{1}{\epsilon},\log n\right)$ , and succeeds with probability  $1 - \frac{1}{\mathrm{poly}(n)}$ .

Theorem 3.2 For a fixed  $\epsilon >0$  and polynomial  $f$ , there exists a data structure HSAMPLER that takes any query  $\mathbf{x}\in \mathbb{R}^d$  and outputs an index  $i\in [n]$  with probability  $\frac{(1\pm\epsilon)\cdot\left\|f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\cdot\mathbf{a}_i^\top\mathbf{a}_i\right\|_F^2}{\sum_{i = 1}^n\left\|f(\langle\mathbf{a}_i,\mathbf{x}\rangle)\cdot\mathbf{a}_i^\top\mathbf{a}_i\right\|_F^2} + \frac{1}{\mathrm{poly}(n)}$ , along with a matrix  $\mathbf{U}\coloneqq f(\langle \mathbf{a}_i,\mathbf{x}\rangle)\cdot \mathbf{a}_i^\top \mathbf{a}_i + \mathbf{V}$ , where  $\mathbb{E}[\mathbf{V}] = 0$  and  $\| \mathbf{V}\| _F\leq \epsilon \cdot \left\| f(\langle \mathbf{a}_i,\mathbf{x}\rangle)\cdot \mathbf{a}_i^\top \mathbf{a}_i\right\| _F$ . The data structure requires a single pass over  $\mathbf{A} = \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n$  (possibly through turnstile updates), can be built in  $\tilde{\mathcal{O}}\left(\mathrm{nnz}(\mathbf{A}) + \frac{d}{\epsilon^2}\right)$  time and  $\tilde{\mathcal{O}}\left(\frac{d}{\epsilon^2}\right)$  space, uses query time poly  $(d,\frac{1}{\epsilon},\log n)$ , and succeeds with probability  $1 - \frac{1}{\mathrm{poly}(n)}$ .

Theorem 3.3 There exists an algorithm HLEVERAGE that returns all indices  $i \in [n]$  such that  $\frac{(1 \pm \epsilon) \cdot \| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i^\top \mathbf{a}_i \|_F^2}{\sum_{i=1}^n \| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i^\top \mathbf{a}_i \|_F^2} \geq \frac{1}{200Td}$  for some  $\mathbf{x} \in \mathbb{R}^n$ , along with a matrix  $\mathbf{U}_i := f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i^\top \mathbf{a}_i + \mathbf{V}_i$ , where  $\| \mathbf{V}_i \|_F \leq \epsilon$  and  $\| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i^\top \mathbf{a}_i \|_F$ . The algorithm uses a single pass over  $\mathbf{A} = \mathbf{a}_1 \circ \ldots \circ \mathbf{a}_n$  (possibly through turnstile updates), uses  $\tilde{\mathcal{O}}\left(\mathrm{nnz}(\mathbf{A}) + \frac{d^\omega}{\epsilon^2}\right)$  runtime (where  $\omega$  denotes the exponent of square matrix multiplication) and  $\tilde{\mathcal{O}}\left(\frac{d}{\epsilon^2}\right)$  space, and succeeds with probability  $1 - \frac{1}{\mathrm{poly}(n)}$ .

We remark that HSAMPLER and LEVERAGE are generalizations of ESTIMATOR and SAMPLER that simply return an outer product of a noisy vector rather than the noisy vector itself.

As before, observe that we could simply run an instance of HSAMPLER to sample a Hessian through importance sampling, but sampling  $T$  Hessians requires  $T$  independent instances, significantly increasing the total runtime. We thus use the same two level data structure that partitions the rows of matrix  $\mathbf{A} = \mathbf{a}_1 \circ \ldots \circ \mathbf{a}_n$  into  $\beta \coloneqq \Theta(Td)$  buckets  $B_1, \ldots, B_\beta$ . We then create an instance of HESTIMATOR and HSAMPLER for each bucket. For an iterate  $\mathbf{x}_t$ , we sample  $j \in [\beta]$  among the

buckets  $B_{1},\ldots ,B_{\beta}$  with probability roughly  $\frac{\sum_{i\in B_j}\left\|f(\langle\mathbf{a}_i,\mathbf{x}_t\rangle)\cdot\mathbf{a}_i^\top\mathbf{a}_i\right\|_F^2}{\sum_{i = 1}^n\left\|f(\langle\mathbf{a}_i,\mathbf{x}_t\rangle)\cdot\mathbf{a}_i^\top\mathbf{a}_i\right\|_F^2}$  using HESTIMATOR and then querying HSAMPLER at  $\mathbf{x}_t$  to sample a Hessian among the indices partitioned into bucket  $B_{j}$ . As before, this argument fails when the same bucket  $B_{j}$  is sampled multiple times, due to dependencies in randomness, but this issue can be avoided by using HLEVERAGE to decrease the probability that each bucket is sampled. We give the algorithm in full in Algorithm 3.

# Algorithm 3 Second-Order Optimization with Importance Sampling

Input: Matrix  $\mathbf{A} = \mathbf{a}_1\circ \ldots \circ \mathbf{a}_n\in \mathbb{R}^{n\times d}$ , parameter  $T$  for number of sampled Hessians, oracle  $\mathbb{O}$  that performs the update rule.

Output:  $T$  approximate Hessian.

1: Preprocessing Stage:

2:  $\beta \gets \Theta(Td)$  with a sufficiently large constant.

3: Let  $h:[n]\to [\beta ]$  be a uniformly random hash function.

4: Let  $\mathbf{B}_j$  be the matrix formed by the rows  $\mathbf{a}_i$  of  $\mathbf{A}$  with  $h(i) = j$ , for each  $j \in [\beta]$ .

5: Create an instance HESTIMATOR and HSAMPLER for each  $\mathbf{B}_j$  with  $j\in [\beta ]$  with  $\epsilon = \frac{1}{2}$

6: Run HLEVERAGE to find a set  $L_{0}$  of row indices and corresponding (noisy) outer products.

7: Second-Order Optimization Stage:

8: Randomly pick starting location  $\mathbf{x}_0$

9: for  $t = 1$  to  $T$  do

10: Let  $q_{i}$  be the output of HESTIMATOR on query  $\mathbf{x}_{t - 1}$  for each  $i\in [\beta ]$ .

11: Sample  $j \in [\beta]$  with probability  $p_j = \frac{q_j}{\sum_{i \in [\beta]} q_i}$ .

12: if there exists  $i \in L_0$  with  $h(i) = j$  then

13: Use HESTIMATOR, HLEVERAGE, and HSAMPLER to sample Hessian  $\mathbf{H}_t$ .

14: else

15: Use HSAMPLER  $j$  to sample Hessian  $\mathbf{H}_t = \nabla f_{i_t}(\mathbf{x}_t)$

16:  $\widehat{p_{i,t}}\gets \frac{\|\mathbf{H}_t\|_F^2}{\sum_{j\in[\beta]}q_j}$

17:  $\mathbf{x}_{t + 1}\gets \mathbb{O}\left(\frac{1}{n\widehat{p_{i,t}}}\mathbf{H}_t\right)$

We remark that Algorithm 3 can be generalized to handle oracles  $\mathbb{O}$  corresponding to second-order methods that require batches of subsampled Hessians in each iteration. For example, if we want to run  $T$  iterations of a second-order method that requires  $s$  subsampled Hessians in each batch, we can simply modify Algorithm 3 to sample  $s$  Hessians in each iteration as input to  $\mathbb{O}$  and thus  $Ts$  Hessians in total.

# 4 EMPIRICAL EVALUATIONS

Our primary contribution is the theoretical design of a nearly input sparsity time algorithm that approximates optimal importance sampling SGD. In this section we implement a scaled-down version of our algorithm and compare its performance on large-scale real world datasets to SGD with uniform sampling on logistic regression. We also consider both linear regression and support-vector machines (SVMs) in the supplementary material. Because most rows have roughly uniformly small leverage scores in real-world data, we assume that no bucket contains a row with a significantly large leverage score and thus the implementation of our importance sampling algorithm does not create multiple samplers for any buckets. By similar reasoning, our implementation uniformly samples a number of indices  $i$  and estimates  $\sum_{i=1}^{n} \| f(\langle \mathbf{a}_i, \mathbf{x} \rangle) \cdot \mathbf{a}_i \|^2_2$  by rescaling. Observe that although these simplifications to our algorithm decrease the wall-clock running time and the total space used by our algorithm, they only decrease the quality of our solution for each SGD iteration. We also consider two hybrid SGD sampling algorithms; the first takes the better gradient obtained at each iteration from both uniform sampling and importance sampling while the second performs 25 iterations of importance sampling before using uniform sampling for the remaining iterations. Surprisingly, our SGD importance sampling implementation not only significantly improves upon SGD with uniform sampling, but are also competitive with the two hybrid algorithms. We do not consider other SGD variants due to either their distributional assumptions or lack of known flexibility to big data models. The experiments were performed in Python 3.6.9 on an Intel Core i7-8700K 3.70 GHz CPU with

12 cores and 64GB DDR4 memory, using a Nvidia Geforce GTX 1080 Ti 11GB GPU. Our code is publicly available at https://github.com/SGD-adaptive-importance/code.

Logistic Regression. We performed logistic regression on the a9a Adult data set collected by UCI and retrieved from LibSVM (Chang & Lin, 2011). The features correspond to responses from the 1994 Census database and the prediction task is to determine whether a person makes over 50K USD a year. We trained using a data batch of 32581 points and 123 features and tested the performance on a separate batch of 16281 data points. For each evaluation, we generated 10 random initial positions shared for importance sampling and uniform sampling. We then ran 250 iterations of SGD for each of the four algorithms, creating only 250 buckets for the importance sampling algorithm and computed the average performance on each iteration across these 10 separate instances. The relative average performance of all algorithms was relatively robust to the step-size. Although uniform sampling used significantly less time overall, our importance sampling SGD algorithm actually had better performance when considering either number of iterations or wall-clock time across all tested step-sizes. For example, uniform sampling had average objective value 20680 at iteration 250 using 0.0307 seconds with step-size 0.1, but importance sampling had average objective value 12917 at iteration 5 using 0.025 seconds. We give our results for logistic regression in Figure 1. For additional experiments, see Section B.

![](images/5d5ec2ac59a8ea5c5ccffa7e010aedbb2f406fd6a4cd86db7c3a38878c7db822.jpg)

![](images/704ee5ab6d7b461656b6cec561481c22d4c5b001de8c94a25f54275986be3813.jpg)

![](images/a915c90206d48a9f6b50426940b4f9893bd94fe5c6e0294df3228983ca28f62f.jpg)

![](images/16b75a193407b88930a8855c960f10eab8845a382fcbf6a05dc315afde26a928.jpg)  
(a) Step size 0.1  
(d) Step size 0.1  
Fig. 1: Comparison of objective values and runtimes for importance sampling (in blue squares), uniform sampling (in red triangles), hybrid sampling that chooses the better gradient at each step (in purple circles), and hybrid sampling that performs 25 steps of importance sampling followed by uniform sampling (in teal X's) over various step-sizes for logistic regression on a9a Adult dataset from UCI, across 250 iterations, averaged over 10 repetitions.

![](images/7ee5a742cd235bacd9a4ee0cb57bdcad5bc25a3a9d03aa7ecc21160553d5e3d1.jpg)  
(b) Step size 0.01  
(e) Step size 0.01

![](images/ed1c9d70e51e842a1ddabb6b4ce92f30d7009a1c438c9dce1ce9a296f780e158.jpg)  
(c) Step size 0.001  
(f) Step size 0.001

# 5 CONCLUSION AND FUTURE WORK

We have given variance reduction methods for both first-order and second-order stochastic optimization. Our algorithms require a single pass over the data, which may even arrive implicitly in the form of turnstile updates, and use input sparsity time and  $\tilde{\mathcal{O}}(Td)$  space. Our algorithms are also amenable to big data models such as the streaming and distributed models and are supported by empirical evaluations on large-scale datasets. We believe there are many interesting future directions to explore. For example, can we generalize our techniques to show provable guarantees for other SGD variants and accelerated methods? A very large-scale empirical study of these methods would also be quite interesting.

# REFERENCES

Noga Alon, Yossi Matias, and Mario Szegedy. The space complexity of approximating the frequency moments. J. Comput. Syst. Sci., 58(1):137-147, 1999. 3, 11  
Alexandr Andoni, Robert Krauthgamer, and Krzysztof Onak. Streaming algorithms via precision sampling. In IEEE 52nd Annual Symposium on Foundations of Computer Science, FOCS, pp. 363-372, 2011. 13  
Guillaume Bouchard, Théo Trouillon, Julien Perez, and Adrien Gaidon. Online learning to sample. CoRR, abs/1506.09016, 2015. 2  
Chih-Chung Chang and Chih-Jen Lin. LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1-27:27, 2011. 2, 8, 17  
Moses Charikar, Kevin C. Chen, and Martin Farach-Colton. Finding frequent items in data streams. Theor. Comput. Sci., 312(1):3-15, 2004. 4, 11  
Kenneth L. Clarkson and David P. Woodruff. Low rank approximation and regression in input sparsity time. In Symposium on Theory of Computing Conference, STOC, pp. 81-90, 2013. 16  
Hadi Daneshmand, Aurélien Lucchi, and Thomas Hofmann. Starting small - learning with adaptive sample sizes. In Proceedings of the 33nd International Conference on Machine Learning, ICML, pp. 1463-1471, 2016. 1, 2  
Aaron Defazio, Francis R. Bach, and Simon Lacoste-Julien. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. In Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems, pp. 1646-1654, 2014. 1, 2  
Petros Drineas, Malik Magdon-Ismail, Michael W. Mahoney, and David P. Woodruff. Fast approximation of matrix coherence and statistical leverage. J. Mach. Learn. Res., 13:3475-3506, 2012. 16  
Roy Frostig, Rong Ge, Sham M. Kakade, and Aaron Sidford. Competing with the empirical risk minimizer in a single pass. In Proceedings of The 28th Conference on Learning Theory, COLT, pp. 728-763, 2015. 2  
Siddharth Gopal. Adaptive sampling for SGD by exploiting side information. In Proceedings of the 33nd International Conference on Machine Learning, ICML, pp. 364-372, 2016. 2  
Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in Neural Information Processing Systems 26, Proceedings., pp. 315-323, 2013. 1, 2  
Tyler B. Johnson and Carlos Guestrin. Training deep models faster with robust, approximate importance sampling. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems, NeurIPS, pp. 7276-7286, 2018. 1, 2  
Ellango Jothimurugesan, Ashraf Tahmasbi, Phillip B. Gibbons, and Srikanta Tirthapura. Variance-reduced stochastic gradient descent on streaming data. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems, NeurIPS, pp. 9928-9937, 2018. 2  
Hossein Jowhari, Mert Saglam, and Gábor Tardos. Tight bounds for lp samplers, finding duplicates in streams, and related problems. In Proceedings of the 30th ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS, pp. 49-58, 2011. 13  
Angelos Katharopoulos and François Fleuret. Not all samples are created equal: Deep learning with importance sampling. In Proceedings of the 35th International Conference on Machine Learning, ICML, pp. 2530-2539, 2018. 1, 2  
Sepideh Mahabadi, Ilya P. Razenshteyn, David P. Woodruff, and Samson Zhou. Non-adaptive adaptive sampling on turnstile streams. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing, STOC, pp. 1251-1264, 2020. 3, 4, 11, 13, 16

Deanna Needell, Nathan Srebro, and Rachel Ward. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. Math. Program., 155(1-2):549-573, 2016. 1, 2  
Jelani Nelson and Huy L. Nguyen. OSNAP: faster numerical linear algebra algorithms via sparser subspace embeddings. In 54th Annual IEEE Symposium on Foundations of Computer Science, FOCS, pp. 117-126. IEEE Computer Society, 2013. 4, 16  
Arkadi Nemirovski, Anatoli B. Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574-1609, 2009. 1  
Arkadi Semenovich Nemirovsky and David Borisovich Yudin. Problem complexity and method efficiency in optimization, 1983. 1  
Xun Qian, Peter Richtárik, Robert M. Gower, Alibek Sailanbayev, Nicolas Loizou, and Egor Shulgin. SGD with arbitrary sampling: General analysis and improved rates. In Proceedings of the 36th International Conference on Machine Learning, ICML, volume 97, pp. 5200-5209, 2019. 1, 2  
Sashank J. Reddi, Ahmed Hefny, Suvrit Sra, Barnabás Póczos, and Alexander J. Smola. On variance reduction in stochastic gradient descent and its asynchronous variants. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems, pp. 2647-2655, 2015. 1, 2  
Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pp. 400-407, 1951. 1  
Farbod Roosta-Khorasani and Michael W. Mahoney. Sub-sampled newton methods I: globally convergent algorithms. CoRR, abs/1601.04737, 2016a. 2  
Farbod Roosta-Khorasani and Michael W. Mahoney. Sub-sampled newton methods II: local convergence rates. CoRR, abs/1601.04738, 2016b. 2  
Farbod Roosta-Khorasani, Kees Van Den Doel, and Uri Ascher. Stochastic algorithms for inverse problems involving pdes and many measurements. SIAM Journal on Scientific Computing, 36(5): S3-S22, 2014. 2  
Nicolas Le Roux, Mark Schmidt, and Francis R. Bach. A stochastic gradient method with an exponential convergence rate for finite training sets. In Advances in Neural Information Processing Systems 25: 26th Annual Conference on Neural Information Processing Systems., pp. 2672-2680, 2012. 1, 2  
Farnood Salehi, Patrick Thiran, and L. Elisa Celis. Coordinate descent with bandit sampling. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems, NeurIPS, pp. 9267-9277, 2018. 1, 2  
Sebastian U. Stich, Anant Raj, and Martin Jaggi. Safe adaptive importance sampling. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems, pp. 4381-4391, 2017. 1, 2  
Peng Xu, Jiyan Yang, Farbod Roosta-Khorasani, Christopher Ré, and Michael W. Mahoney. Subsampled newton methods with non-uniform sampling. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems, pp. 3000-3008, 2016. 3  
Peng Xu, Fred Roosta, and Michael W Mahoney. Newton-type methods for non-convex optimization under inexact hessian information. Mathematical Programming, pp. 1-36, 2019. 2  
Peng Xu, Fred Roosta, and Michael W. Mahoney. Second-order optimization for non-convex machine learning: an empirical study. In Proceedings of the 2020 SIAM International Conference on Data Mining, SDM 2020, Cincinnati, Ohio, USA, May 7-9, 2020, pp. 199-207. SIAM, 2020. 2  
Peilin Zhao and Tong Zhang. Stochastic optimization with importance sampling for regularized loss minimization. In Proceedings of the 32nd International Conference on Machine Learning ICML, pp. 1-9, 2015. 1, 2
