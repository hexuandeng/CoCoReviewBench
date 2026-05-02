# Practical Schemes for Finding Near-Stationary Points of Convex Finite-Sums

Anonymous Author(s)

Affiliation

Address

email

# Abstract

The problem of finding near-stationary points in convex optimization has not been adequately studied yet, unlike other optimality measures such as the function value. Even in the deterministic case, the optimal method (OGM-G, due to Kim and Fessler [33]) has just been discovered recently. In this work, we conduct a systematic study of algorithmic techniques for finding near-stationary points of convex finite-sums. Our main contributions are several algorithmic discoveries: (1) we discover a memory-saving variant of OGM-G based on the performance estimation problem approach [19]; (2) we design a new accelerated SVRG variant that can simultaneously achieve fast rates for minimizing both the gradient norm and function value; (3) we propose an adaptively regularized accelerated SVRG variant, which does not require the knowledge of some unknown initial constants and achieves near-optimal complexities. We put an emphasis on the simplicity and practicality of the new schemes, which could facilitate future developments.

# 1 Introduction

Classic convex optimization usually focuses on providing guarantees for minimizing function value. For this task, the optimal (up to constant factors) Nesterov's accelerated gradient method (NAG) [40, 41] has been known for decades, and there are even methods that can exactly match the lower complexity bounds [30, 17, 55, 18]. On the other hand, in general non-convex optimization, near-stationarity is the typical optimality measure, and there has been a flurry of recent research devoted to this topic [25, 26, 23, 28, 21, 60]. Recently, there has been growing interest on devising fast schemes for finding near-stationary points in convex optimization [42, 2, 22, 7, 31, 32, 33, 27, 15, 14]. This line of research is basically driven by the following facts.

- Nesterov [42] studied the problem with a linear constraint:  $f(x^{\star}) = \min_{x\in Q}\{f(x):Ax = b\}$ , where  $Q$  is a convex set and  $f$  is strongly convex. Assuming that  $Q$  and  $f$  are simple, we can focus on the dual problem  $\phi (y^{\star}) = \max_y\{\phi (y)\triangleq \min_{x\in Q}\{f(x) + \langle y,b - Ax\rangle \} \}$ . Clearly, the dual objective  $-\phi (y)$  is smooth convex. Letting  $x_{y}$  be the unique solution to the inner problem, we have  $\nabla \phi (y) = b - Ax_{y}$ . Note that  $f(x_{y}) - f(x^{\star}) = \phi (y) - \langle y,\nabla \phi (y)\rangle -\phi (y^{\star})\leq \| y\| \| \nabla \phi (y)\|$ . Thus, in this problem, the quantity  $\| \nabla \phi (y)\|$  serves as a measure of both primal optimality  $f(x_{y}) - f(x^{\star})$  and feasibility  $\| b - Ax_{y}\|$ , which is better than just measuring the function value.  
- Matrix scaling [50] is a convex problem and its goal is to find near-stationary points [4, 9].  
- Gradient norm is readily available, unlike other optimality measures  $(f(x) - f(x^{\star})$  and  $\| x - x^{\star}\|$ ), and is thus usable as a stopping criterion. This fact motivates the design of several parameter-free algorithms [43, 39, 27], and their guarantees are established on the gradient norm.  
- Designing schemes for minimizing the gradient norm can inspire new non-convex optimization methods. For example, SARAH [46] was designed for convex finite-sums with gradient-norm measure, but was later discovered to be the near-optimal method for non-convex finite-sums [21, 47].

Table 1: Finding near-stationary points  $\| \nabla f(x) \| \leq \epsilon$  of convex finite-sums.  

<table><tr><td></td><td>Algorithm</td><td>Complexity</td><td>Remark</td></tr><tr><td rowspan="7">IFC</td><td>GD [33]</td><td>O( n/ε2)</td><td></td></tr><tr><td>Regularized NAG* [7]</td><td>O( n/ε log 1/ε)</td><td></td></tr><tr><td>OGM-G [33]</td><td>O( n/ε)</td><td>O( 1/ε + d) memory, optimal in ε</td></tr><tr><td>M-OGM-G [Section 3.1]</td><td>O( n/ε)</td><td>O(d) memory, optimal in ε</td></tr><tr><td>L2S [37]</td><td>O(n + √n/ε2)</td><td>Loopless variant of SARAH [46]</td></tr><tr><td>Regularized Katyusha* [2]</td><td>O((n + √n/ε) log 1/ε)</td><td>Requires the knowledge of Δ0</td></tr><tr><td>R-Acc-SVRG-G* [Section 5]</td><td>O((n log 1/ε + √n/ε) log 1/ε)</td><td>Without the knowledge of Δ0</td></tr><tr><td rowspan="9">IDC</td><td>GD [42, 54]</td><td>O( n/ε)</td><td></td></tr><tr><td>NAG/NAG+GD [32] / [42]</td><td>O( n/ε2/3)</td><td></td></tr><tr><td>Regularized NAG* [42, 27]</td><td>O( n/√ε log 1/ε)</td><td></td></tr><tr><td>NAG+OGM-G [45]</td><td>O( n/√ε)</td><td>O( 1/√ε + d) memory, optimal in ε</td></tr><tr><td>NAG+M-OGM-G [Section 3.1]</td><td>O( n/√ε)</td><td>O(d) memory, optimal in ε</td></tr><tr><td>Katyusha+L2S [Appendix E]</td><td>O(n log 1/ε + √n/ε2/3)</td><td></td></tr><tr><td>Acc-SVRG-G [Section 4]</td><td>O(n log 1/ε + n2/3/ε2/3)1</td><td>O(n log 1/ε + √n/ε) for function at the same time, simple and elegant</td></tr><tr><td>Regularized Katyusha* [2]</td><td>O((n + √n/ε) log 1/ε)</td><td>Requires the knowledge of R0</td></tr><tr><td>R-Acc-SVRG-G* [Section 5]</td><td>O((n log 1/ε + √n/ε) log 1/ε)</td><td>Without the knowledge of R0</td></tr></table>

* Indirect methods (using regularization).

Moreover, finding near-stationary points is a harder task than minimizing function value, because NAG has the optimal guarantee for  $f(x) - f(x^{\star})$  but is only suboptimal for minimizing  $\| \nabla f(x)\|$ .

In this work, we consider the problem  $\min_{x\in \mathbb{R}^d}f(x) = \frac{1}{n}\sum_{i = 1}^{n}f_i(x)$ , where each  $f_{i}$  is  $L$ -smooth and convex. We focus on finding an  $\epsilon$ -stationary point of this objective, i.e., a point with  $\| \nabla f(x)\| \leq \epsilon$ . We use  $\mathcal{X}^{\star}$  to denote the set of optimal solutions, which is assumed to be nonempty. There are two different assumptions on the initial point  $x_0$ , namely, the Initial bounded-Function Condition (IFC):  $f(x_0) - f(x^{\star})\leq \Delta_0$ , and the Initial bounded-Distance Condition (IDC):  $\| x_0 - x^{\star}\| \leq R_0$  for some  $x^{\star}\in \mathcal{X}^{\star}$ . This subtlety results in drastically different best achievable rates as studied in [7, 22]. Below we categorize existing algorithmic techniques into three classes (relating to Table 1).

(i) "IDC + IFC". Nesterov [42] showed that we can combine the guarantees of a method minimizing function value under IDC and a method finding near-stationary points under IFC to produce a faster one for minimizing gradient norm under IDC. For example, NAG produces  $f(x_{K_1}) - f(x^{\star}) = O\left(\frac{LR_0^2}{K_1^2}\right)$  [40] and GD produces  $\| \nabla f(x_{K_2})\| ^2 = O\left(\frac{L(f(x_0) - f(x^{\star}))}{K_2}\right)$  [33] under IFC. Letting  $x_0 = x_{K_1}$  and  $K = K_{1} + K_{2}$ , by balancing the ratio of  $K_{1}$  and  $K_{2}$ , we obtain the guarantee  $\| \nabla f(x_K)\|^2 = O\left(\frac{L^2R_0^2}{K^3}\right)$  for "NAG + GD". We point out that we can use this technique to combine the guarantees of Katyusha [1] and SARAH[46]; see Appendix E.

(ii) Regularization. Nesterov [42] used NAG (strongly convex variant) to solve the regularized objective, and showed that it achieves near-optimal complexity (optimal up to logarithmic factors). Inspired by this technique, Allen-Zhu [2] proposed recursive regularization for stochastic approximation algorithms, which also achieves near-optimal complexities [22].

(iii) Direct methods. Due to the lack of insight, existing direct methods are mostly derived or analyzed with the help of computer-aided tools [31, 32, 54, 33]. The computer-aided approach was pioneered by Drori and Teboulle [19], who introduced the performance estimation problem (PEP). The only known optimal method OGM-G [33] was designed based on the PEP approach.

Observe that since  $f(x) - f(x^{\star}) \leq \| \nabla f(x) \| \| x - x^{\star} \|$ , the lower bound for finding near-stationary points must be of the same order as for minimizing function value [44]. Thus, under IDC, the lower bound is  $\Omega \left( n + \sqrt{\frac{n}{\epsilon}} \right)$  due to [58]. Under IFC, we can establish an  $\Omega \left( n + \frac{\sqrt{n}}{\epsilon} \right)$  lower bound using the techniques in [7, 58]. The main contributions of this work are three new algorithmic schemes that improve the practicalities of existing methods as summarized below (highlighted in Table 1).

- (Section 3) We propose a memory-saving variant of OGM-G for the deterministic case ( $n = 1$ ), which does not require a pre-computed and stored parameter sequence. The derivation of the new variant is inspired by the numerical solution to a PEP problem.  
- (Section 4) We propose a new accelerated SVRG [29, 59] variant that can simultaneously achieve fast convergence rates for minimizing both the gradient norm and function value, that is,  $O\left(n \log \frac{1}{\epsilon} + \frac{n^{2/3}}{\epsilon^{2/3}}\right)$  complexity for gradient norm and  $O\left(n \log \frac{1}{\epsilon} + \sqrt{\frac{n}{\epsilon}}\right)$  complexity for function value. Note that other stochastic approaches in Table 1 do not have this property.  
- (Section 5) We propose an adaptively regularized accelerated SVRG variant, which does not require the knowledge of  $R_0$  or  $\Delta_0$  and achieves a near-optimal complexity under IDC or IFC.

We put in extra efforts to make the proposed schemes as simple and elegant as possible. We believe that the simplicity makes the extensions of the new schemes easier.

# 2 Preliminaries

Throughout this paper, we use  $\langle \cdot ,\cdot \rangle$  and  $\| \cdot \|$  to denote the inner product and the Euclidean norm, respectively. We let  $[n]$  denote the set  $\{1,2,\dots ,n\}$ ,  $\mathbb{E}$  denote the total expectation and  $\mathbb{E}_{i_k}$  denote the expectation with respect to a random sample  $i_k$ . We say that a function  $f:\mathbb{R}^d\to \mathbb{R}$  is  $L$ -smooth if it has  $L$ -Lipschitz continuous gradients, i.e.,

$$
\forall x, y \in \mathbb {R} ^ {d}, \| \nabla f (x) - \nabla f (y) \| \leq L \| x - y \|.
$$

A continuously differentiable  $f$  is called  $\mu$ -strongly convex if

$$
\forall x, y \in \mathbb {R} ^ {d}, f (x) - f (y) - \langle \nabla f (y), x - y \rangle \geq \frac {\mu}{2} \| x - y \| ^ {2}.
$$

Other equivalent definitions of these two assumptions can be found in the textbook [44]. The following is an important consequence of a function  $f$  being  $L$ -smooth and convex:

$$
\forall x, y \in \mathbb {R} ^ {d}, f (x) - f (y) - \langle \nabla f (y), x - y \rangle \geq \frac {1}{2 L} \| \nabla f (x) - \nabla f (y) \| ^ {2}. \tag {1}
$$

We call (1) the interpolation condition at  $(x,y)$  following [56]. If  $f$  is both  $L$ -smooth and  $\mu$ -strongly convex, we can define a "shifted" function  $h(x) = f(x) - f(x^{\star}) - \frac{\mu}{2}\| x - x^{\star}\|^2$  following [63]. It can be easily verified that  $h$  is  $(L - \mu)$ -smooth and convex, and thus from (1),

$$
\forall x, y \in \mathbb {R} ^ {d}, h (x) - h (y) - \langle \nabla h (y), x - y \rangle \geq \frac {1}{2 (L - \mu)} \| \nabla h (x) - \nabla h (y) \| ^ {2}, \tag {2}
$$

which is equivalent to the strongly convex interpolation condition discovered in [56].

Oracle complexity (or simply complexity) refers to the required number of stochastic gradient  $\nabla f_{i}$  computations to find an  $\epsilon$ -accurate solution.

# 3 OGM-G: "Momentum" Reformulation and a Memory-Saving Variant

In this section, we focus on the IFC case, i.e.,  $f(x_0) - f(x^{\star}) \leq \Delta_0$ . We use  $N$  to denote the total iteration number to prevent confusion (in other sections, we use  $K$ ). Proofs in this section are given in

# Algorithm 1 OGM-G: "Momentum" reformulation

Input: initial guess  $x_0 \in \mathbb{R}^d$ , total iteration number  $N$ .

Initialize: vector  $v_{0} = \mathbf{0}$ , scalars  $\theta_{N} = 1$  and  $\theta_{k}^{2} - \theta_{k} = \theta_{k + 1}^{2}$ , for  $k = 0\ldots N - 1$ .

1: for  $k = 0, \dots, N - 1$  do

2:  $v_{k + 1} = v_k + \frac{1}{L\theta_k\theta_{k + 1}^2}\nabla f(x_k).$  
3:  $x_{k + 1} = x_{k} - \frac{1}{L}\nabla f(x_{k}) - (2\theta_{k + 1}^{3} - \theta_{k + 1}^{2})v_{k + 1}.$  
4: end for

Output:  $x_{N}$ .

Appendix B. Recall that OGM-G has the following updates [33]. Let  $y_0 = x_0$ . For  $k = 0, \dots, N - 1$ ,

$$
\begin{array}{l} y _ {k + 1} = x _ {k} - \frac {1}{L} \nabla f (x _ {k}), \\ x _ {k + 1} = y _ {k + 1} + \frac {\left(\theta_ {k} - 1\right) \left(2 \theta_ {k + 1} - 1\right)}{\theta_ {k} \left(2 \theta_ {k} - 1\right)} \left(y _ {k + 1} - y _ {k}\right) + \frac {2 \theta_ {k + 1} - 1}{2 \theta_ {k} - 1} \left(y _ {k + 1} - x _ {k}\right), \tag {3} \\ \end{array}
$$

where  $\{\theta_k\}$  is recursively defined:  $\theta_N = 1$  and  $\begin{cases} \theta_k^2 - \theta_k = \theta_{k+1}^2 & k = 1 \dots N-1, \\ \theta_0^2 - \theta_0 = 2\theta_1^2 & \text{otherwise.} \end{cases}$

OGM-G was discovered from the numerical solution to an SDP problem and its analysis is to show that the step coefficients in (3) specify a feasible solution to the SDP problem. While this analysis is natural for the PEP approach, it is hard to understand how each coefficient affects the rate, especially if one wants to generalize the scheme. Here we provide a simple algebraic analysis for OGM-G.

We start with a reformulation $^3$  of OGM-G in Algorithm 1, which aims to simplify the proof. We adopt a consistent  $\{\theta_k\} : \theta_N = 1$  and  $\theta_k^2 - \theta_k = \theta_{k+1}^2, k = 0 \dots N-1$ , which only costs a constant factor. $^4$  Interestingly, the reformulated scheme resembles the heavy-ball momentum method [49]. However, it can be shown that Algorithm 1 is not covered by the heavy-ball momentum scheme. Defining  $\theta_{N+1}^2 = \theta_N^2 - \theta_N = 0$ , we provide the one-iteration analysis in the following proposition:

Proposition 3.1. In Algorithm 1, the following holds at any iteration  $k \in \{0, \dots, N - 1\}$ :

$$
\begin{array}{l} A _ {k} + B _ {k + 1} + C _ {k + 1} + E _ {k + 1} \leq A _ {k + 1} + B _ {k} + C _ {k} + E _ {k} - \theta_ {k + 1} \langle \nabla f (x _ {k + 1}), v _ {k + 1} \rangle \\ + \sum_ {i = k + 1} ^ {N} \frac {\theta_ {i}}{L \theta_ {k} \theta_ {k + 1} ^ {2}} \langle \nabla f (x _ {k}), \nabla f (x _ {i}) \rangle , \tag {4} \\ \end{array}
$$

with  $A_{k}\triangleq \frac{1}{\theta_{k}^{2}} (f(x_{N}) - f(x^{\star}) - \frac{1}{2L}\| \nabla f(x_{N})\|^{2}),B_{k}\triangleq \frac{1}{\theta_{k}^{2}} (f(x_{k}) - f(x^{\star})),C_{k}\triangleq \frac{1}{2L\theta_{k}^{2}}\| \nabla f(x_{k})\|^{2},$

$$
E _ {k} \triangleq \frac {\theta_ {k + 1} ^ {2}}{\theta_ {k}} \langle \nabla f (x _ {k}), v _ {k} \rangle .
$$

Remark 3.1.1. A recent work [15] also conducted an algebraic analysis of OGM-G under a potential function framework. Their potential function decrease can be directly obtained from Proposition 3.1 by summing up (4). By contrast, our "momentum" vector  $\{v_{k}\}$  naturally merges into the analysis, which significantly simplifies the analysis. Moreover, it provides a better interpretation on how OGM-G utilizes the past gradients to achieve acceleration.

From (4), we see that only the last two terms do not telescope. Note that the "momentum" vector is a weighted sum of the past gradients, i.e.,  $v_{k + 1} = \sum_{i = 0}^{k}\frac{1}{L\theta_{i}\theta_{i + 1}^{2}}\nabla f(x_{i})$ . If we sum the terms up from  $k = 0,\dots ,N - 1$ , it can be verified that they exactly sum up to 0. The presence of these special terms prevents OGM-G to have a usual potential function (e.g., those in [6]). Then, by telescoping the remaining terms, we obtain the final convergence guarantee.

Theorem 3.1. The output of Algorithm 1 satisfies  $\| \nabla f(x_N)\|^2 \leq \frac{8L\Delta_0}{(N + 2)^2}$ .

We observe two drawbacks of OGM-G (same as the algorithm description in [15]): (1) it requires storing a pre-computed parameter sequence, which costs  $O\left(\frac{1}{\epsilon}\right)$  floats; (2) except for the last iterate,

# Algorithm 2 M-OGM-G: Memory-saving OGM-G

Input: initial guess  $x_0 \in \mathbb{R}^d$ , total iteration number  $N$ .

Initialize: vector  $v_{0} = 0$

1: for  $k = 0, \dots, N - 1$  do

2:  $v_{k + 1} = v_k + \frac{12}{L(N - k + 1)(N - k + 2)(N - k + 3)}\nabla f(x_k).$  
3:  $x_{k + 1} = x_k - \frac{1}{L}\nabla f(x_k) - \frac{(N - k)(N - k + 1)(N - k + 2)}{6} v_{k + 1}.$  
4: end for

Output:  $x_{N}$  or  $\arg \min_{x\in \{x_0,\dots ,x_N\}}\| \nabla f(x)\|$

all other iterates are not known to have guarantees. We resolve these issues by proposing another parameterization of Algorithm 1 in the next subsection.

# 3.1 Memory-Saving OGM-G

A straightforward idea to resolve the aforementioned issues is to generalize Algorithm 1. However, we find it rather difficult since the parameters in the analysis are rather strict (despite that the proof is already simple). We choose to rely on computer-aided techniques [19]. The derivation of this variant (Algorithm 2) is based on the following numerical experiment.

Numerical experiment. OGM-G was discovered when considering the relaxed PEP problem [33]:

$$
\max_{\substack{\nabla f(x_{0}),\ldots ,\nabla f(x_{N})\in \mathbb{R}^{d}\\ f(x_{0}),\ldots ,f(x_{N}),f(x^{\star})\in \mathbb{R}}}\| \nabla f(x_{N})\|^{2}
$$

$$
\text {s u b j e c t} \left\{ \begin{array}{l} \text {i n t e r p o l a t i o n c o n d i t i o n (1) a t} \left(x _ {k}, x _ {k + 1}\right), k = 0, \dots , N - 1, \\ \text {i n t e r p o l a t i o n c o n d i t i o n (1) a t} \left(x _ {N}, x _ {k}\right), k = 0, \dots , N - 1, \\ \text {i n t e r p o l a t i o n c o n d i t i o n (1) a t} \left(x _ {N}, x ^ {\star}\right), f \left(x _ {0}\right) - f \left(x ^ {\star}\right) \leq \Delta_ {0}, \end{array} \right. \tag {P}
$$

where the sequence  $\{x_{k}\}$  is defined as  $x_{k + 1} = x_{k} - \frac{1}{L}\sum_{i = 0}^{k}h_{k + 1,i}\nabla f(x_{i}), k = 0,\dots ,N - 1$  for some step coefficients  $h\in \mathbb{R}^{N(N + 1) / 2}$ . Given  $N$ , the step coefficients of OGM-G correspond to a numerical solution to the problem:  $\arg \min_h\{\mathrm{Lagrangian~dual~of~(P)}\}$ , which is denoted as (HD). Conceptually, solving problem (HD) would give us the fastest possible step coefficients under the constraints. We expect there to be some constant-time slower schemes, which are neglected when solving (HD). To identify such schemes, we relax a set of interpolation conditions in problem (P):

$$
f (x _ {N}) - f (x _ {k}) - \langle \nabla f (x _ {k}), x _ {N} - x _ {k} \rangle \geq \frac {1}{2 L} \| \nabla f (x _ {N}) - \nabla f (x _ {k}) \| ^ {2} - \rho \| \nabla f (x _ {k}) \| ^ {2},
$$

for  $k = 0, \dots, N - 1$  and some  $\rho > 0$ . After this relaxation, solving (HD) will no longer give us the step coefficients of OGM-G. By trying different  $\rho$  and checking the dependence on  $\bar{N}$ , we discover Algorithm 2 when  $\rho = \frac{1}{2L}$ . Similar to our analysis of OGM-G, we provide a simple algebraic analysis for the new variant in the following theorem.

Theorem 3.2. Define  $\delta_{k + 1} \triangleq \frac{12}{(N - k + 1)(N - k + 2)(N - k + 3)}$ ,  $k = 0, \ldots, N$ . In Algorithm 2, it holds that

$$
\sum_ {k = 0} ^ {N} \frac {\delta_ {k + 1}}{2} \| \nabla f (x _ {k}) \| ^ {2} \leq \frac {1 2 L \Delta_ {0}}{(N + 2) (N + 3)}. \tag {5}
$$

Remark 3.2.1. Algorithm 2 converges optimally on the last iterate (note that  $\delta_{N + 1} = 2$ ) and the minimum gradient since

$$
\min  _ {k \in \{0, \dots , N \}} \| \nabla f (x _ {k}) \| ^ {2} \leq \frac {1}{\sum_ {k = 0} ^ {N} \frac {\delta_ {k + 1}}{2}} \sum_ {k = 0} ^ {N} \frac {\delta_ {k + 1}}{2} \| \nabla f (x _ {k}) \| ^ {2} \leq \frac {8 L \Delta_ {0}}{(N + 2) (N + 3) - 2}.
$$

Clearly, the parameters of this variant can be computed on the fly and from (5), each iterate has a guarantee (although the guarantee degenerates quickly as  $k \to 0$  since  $1 / \delta_{k + 1} = \Omega ((N - k)^3)$ ). Moreover, we can extend the benefits into the IDC case using the ideas in [42] as summarized below.

Algorithm 3 Acc-SVRG-G: Accelerated SVRG for Gradient minimization

Input: parameters  $\{\tau_k\}$ ,  $\{p_k\}$ , initial guess  $x_0 \in \mathbb{R}^d$ , total iteration number  $K$ . Initialize: vectors  $z_0 = \tilde{x}_0 = x_0$  and scalars  $\alpha_k = \frac{L\tau_k}{1 - \tau_k}, \forall k$  and  $\widetilde{\tau} = \sum_{k=0}^{K-1} \tau_k^{-2}$ .

1: for  $k = 0, \dots, K - 1$  do

2:  $y_{k} = \tau_{k}z_{k} + (1 - \tau_{k})\left(\tilde{x}_{k} - \frac{1}{L}\nabla f(\tilde{x}_{k})\right)$

3:  $z_{k + 1} = \arg \min_{x}\left\{\langle \mathcal{G}_{k},x\rangle +(\alpha_{k} / 2)\| x - z_{k}\|^{2}\right\} .$

4: //  $\mathcal{G}_k\triangleq \nabla f_{i_k}(y_k) - \nabla f_{i_k}(\tilde{x}_k) + \nabla f(\tilde{x}_k)$ , where  $i_k$  is sampled uniformly in  $[n]$ .

5:  $\tilde{x}_{k + 1} = \left\{ \begin{array}{ll}y_k & \text{with probability } p_k,\\ \tilde{x}_k & \text{with probability } 1 - p_k. \end{array} \right.$

6: end for

Output (for gradient):  $x_{\mathrm{out}}$  is sampled from  $\left\{\mathrm{Prob}\{x_{\mathrm{out}} = \tilde{x}_k\} = \frac{\tau_k^{-2}}{\tilde{\tau}} \mid k \in \{0, \dots, K - 1\} \right\}$ .

Output (for function value):  $\tilde{x}_K$ .

Corollary 3.2.1 (IDC case). If we first run  $N / 2$  iterations of NAG and then continue with  $N / 2$  iterations of Algorithm 2, we obtain an output satisfying  $\| \nabla f(x_N)\| = O(\frac{LR_0}{N^2})$ .

# 4 Accelerated SVRG: Fast Rates for Both Gradient Norm and Objective

In this section, we focus on the IDC case, i.e.,  $\| x_0 - x^\star \| \leq R_0$  for some  $x^\star \in \mathcal{X}^\star$ . From the development in the previous section, it is natural to ask whether we can use the PEP approach to motivate new stochastic schemes. However, due to the exponential growth of the number of possible states  $(i_0, i_1, \ldots)$ , we cannot directly adopt this approach. A feasible alternative is to first fix an algorithmic framework and a family of potential functions, and then use the potential-based PEP approach in [54]. However, this approach is much more restrictive. For example, it cannot identify special constructions like (4) in OGM-G. Fortunately, as we will see, we can get some inspiration from the recent development of deterministic methods. Proofs in this section are given in Appendix C.

Our proposed scheme is given in Algorithm 3. We adopt the elegant loopless design of SVRG in [34]. Note that the full gradient  $\nabla f(\tilde{x}_k)$  is computed and stored only when  $\tilde{x}_{k + 1} = y_k$  at Step 5. We summarize our main technical novelty as follows.

Main algorithmic novelty. The design of stochastic accelerated methods is largely inspired by NAG. To make it clear, by setting  $n = 1$ , we see that Katyusha [1], MiG [61], SSNM [62], Varag [36], VRADA [52], ANITA [38], the acceleration framework in [16] and AC-SA [35, 24] all reduce to one of the following variants of NAG. We say that these methods are under the NAG framework.

$$
\left\{ \begin{array}{l} x _ {k} = \tau_ {k} z _ {k} + (1 - \tau_ {k}) y _ {k}, \\ z _ {k + 1} = z _ {k} - \alpha_ {k} \nabla f (x _ {k}), \\ y _ {k + 1} = \tau_ {k} z _ {k + 1} + (1 - \tau_ {k}) y _ {k}. \end{array} \right.
$$

Auslender and Teboulle [5]

$$
\left\{ \begin{array}{l} x _ {k} = \tau_ {k} z _ {k} + (1 - \tau_ {k}) y _ {k}, \\ z _ {k + 1} = z _ {k} - \alpha_ {k} \nabla f (x _ {k}), \\ y _ {k + 1} = x _ {k} - \eta_ {k} \nabla f (x _ {k}). \end{array} \right.
$$

Linear Coupling [64]

See [57, 12] for other variants of NAG. When  $n = 1$ , Algorithm 3 reduces to the following scheme:

$$
\left\{ \begin{array}{l} y _ {k} = \tau_ {k} z _ {k} + \left(1 - \tau_ {k}\right) \left(y _ {k - 1} - \frac {1}{L} \nabla f (y _ {k - 1})\right), \\ z _ {k + 1} = z _ {k} - \frac {1}{\alpha_ {k}} \nabla f (y _ {k}). \end{array} \right.
$$

Optimized Gradient Method (OGM) [19, 30]

Algorithm 3 reduces to the scheme of OGM when  $n = 1$  (this point is clearer in the formulation of ITEM in [55]). OGM has a constant-time faster worst-case rate than NAG, which exactly matches the lower complexity bound established in [17]. In the following proposition, we show that the OGM framework helps us conduct a tight one-iteration analysis, which gives room for achieving our goal.

Proposition 4.1. In Algorithm 3, the following holds at any iteration  $k \geq 0$  and  $\forall x^{\star} \in \mathcal{X}^{\star}$ :

$$
\begin{array}{l} \left(\frac {1 - \tau_ {k}}{\tau_ {k} ^ {2} p _ {k}} \mathbb {E} \left[ f (\tilde {x} _ {k + 1}) - f (x ^ {\star}) \right] + \frac {L}{2} \mathbb {E} \left[ \| z _ {k + 1} - x ^ {\star} \| ^ {2} \right]\right) + \frac {(1 - \tau_ {k}) ^ {2}}{2 L \tau_ {k} ^ {2}} \mathbb {E} \left[ \| \nabla f (\tilde {x} _ {k}) \| ^ {2} \right] \\ \leq \left(\frac {\left(1 - \tau_ {k} p _ {k}\right) \left(1 - \tau_ {k}\right)}{\tau_ {k} ^ {2} p _ {k}} \mathbb {E} \left[ f \left(\tilde {x} _ {k}\right) - f \left(x ^ {\star}\right) \right] + \frac {L}{2} \mathbb {E} \left[ \| z _ {k} - x ^ {\star} \| ^ {2} \right]\right). \\ \end{array}
$$

The terms inside the parentheses form the commonly used potential function of SVRG variants. The additional  $\mathbb{E}[\| \nabla f(\tilde{x}_k)\|^2]$  term is created by adopting the OGM framework. In other words, we use the following potential function for Algorithm 3 ( $a_k, b_k, c_k \geq 0$ ):

$$
T _ {k} = a _ {k} \mathbb {E} \left[ f (\tilde {x} _ {k}) - f (x ^ {\star}) \right] + b _ {k} \mathbb {E} \left[ \| z _ {k} - x ^ {\star} \| ^ {2} \right] + \sum_ {i = 0} ^ {k - 1} c _ {i} \mathbb {E} \left[ \| \nabla f (\tilde {x} _ {i}) \| ^ {2} \right].
$$

We first provide a simple parameter choice, which leads to a simple and clean analysis.

Theorem 4.1 (Single-stage parameter choice). In Algorithm 3, if we choose  $p_k \equiv \frac{1}{n}$ ,  $\tau_k = \frac{3}{k / n + 6}$ , then the following holds at the outputs:

$$
\begin{array}{l} \mathbb {E} \left[ \| \nabla f (x _ {\text {o u t}}) \| ^ {2} \right] = O \left(\frac {n ^ {3} L \left(f \left(x _ {0}\right) - f \left(x ^ {\star}\right)\right) + n ^ {2} L ^ {2} R _ {0} ^ {2}}{K ^ {3}}\right), \tag {7} \\ \mathbb {E} \left[ f (\tilde {x} _ {K}) \right] - f (x ^ {\star}) = O \left(\frac {n ^ {2} \big (f (x _ {0}) - f (x ^ {\star}) \big) + n L R _ {0} ^ {2}}{K ^ {2}}\right). \\ \end{array}
$$

In other words, to guarantee that  $\mathbb{E}\left[\|\nabla f(x_{\mathrm{out}})\|\right] \leq \epsilon_g$  and  $\mathbb{E}\left[f(\tilde{x}_K)\right] - f(x^{\star}) \leq \epsilon_f$ , the oracle complexities are  $O\left(\frac{n(L(f(x_0) - f(x^{\star})))^{1/3}}{\epsilon_g^{2/3}} + \frac{(nLR_0)^{2/3}}{\epsilon_g^{2/3}}\right)$  and  $O\left(n\sqrt{\frac{f(x_0) - f(x^{\star})}{\epsilon_f}} + \frac{\sqrt{nL}R_0}{\sqrt{\epsilon_f}}\right)$ , respectively.  
From (7), we see that Algorithm 3 achieves fast  $O\left(\frac{1}{K^{1.5}}\right)$  and  $O\left(\frac{1}{K^2}\right)$  rates for minimizing the gradient norm and function value at the same time. However, despite being a simple choice, the oracle complexities are not better than the deterministic methods in Table 1. Below we provide a two-stage parameter choice, which is inspired by the idea of including a "warm-up phase" in [3, 36, 52, 38]. This theorem corresponds to the reported result in Table 1.  
Theorem 4.2 (Two-stage parameter choice). In Algorithm 3, let  $p_k = \max \left\{\frac{6}{k + 8}, \frac{1}{n}\right\}$ ,  $\tau_k = \frac{3}{p_k(k + 8)}$ . The oracle complexities needed to guarantee  $\mathbb{E}\left[\|\nabla f(x_{\mathrm{out}})\|\right] \leq \epsilon_g$  and  $\mathbb{E}\left[f(\tilde{x}_K)\right] - f(x^{\star}) \leq \epsilon_f$  are

$$
O \left(n \min  \left\{\log \frac {L R _ {0}}{\epsilon_ {g}}, \log n \right\} + \frac {(n L R _ {0}) ^ {2 / 3}}{\epsilon_ {g} ^ {2 / 3}}\right) a n d O \left(n \min  \left\{\log \frac {L R _ {0} ^ {2}}{\epsilon_ {f}}, \log n \right\} + \frac {\sqrt {n L} R _ {0}}{\sqrt {\epsilon_ {f}}}\right),
$$

respectively.

If  $\epsilon$  is large or  $n$  is very large, the recently proposed ANITA [38] achieves an  $O(n)$  complexity, which matches the lower complexity bound  $\Omega (n)$  in this case [58]. Since ANITA uses the NAG framework, we show that similar results can be derived under the OGM framework in the following theorem:

Theorem 4.3 (Low accuracy parameter choice). In Algorithm 3, let iteration  $N$  be the first time Step 5 updates  $\tilde{x}_{k + 1} = y_k$ . If we choose  $p_k \equiv \frac{1}{n}$ ,  $\tau_k \equiv 1 - \frac{1}{\sqrt{n + 1}}$  and terminate Algorithm 3 at iteration  $N$ , then the following holds at  $\tilde{x}_{N + 1}$ :

$$
\mathbb {E} \left[ \| \nabla f (\tilde {x} _ {N + 1}) \| ^ {2} \right] \leq \frac {8 L ^ {2} R _ {0} ^ {2}}{5 (\sqrt {n + 1} + 1)} a n d \mathbb {E} \left[ f (\tilde {x} _ {N + 1}) \right] - f (x ^ {\star}) \leq \frac {L R _ {0} ^ {2}}{\sqrt {n + 1} + 1}.
$$

In particular, if the required accuracies are low (or  $n$  is very large), i.e.,  $\epsilon_g^2 \geq \frac{8L^2R_0^2}{5(\sqrt{n + 1} + 1)}$  and  $\epsilon_f \geq \frac{LR_0^2}{\sqrt{n + 1} + 1}$ , then Algorithm 3 only has an  $O(n)$  oracle complexity.

In the low accuracy region (specified above), the choice in Theorem 4.3 removes the  $O(\log \frac{1}{\epsilon})$  factor in the complexity of Theorem 4.2. We include some numerical justifications of Algorithm 3 in Appendix A. We believe that the potential-based PEP approach in [54] can help us identify better parameter choices of Algorithm 3, which we leave for future work.

Algorithm 4 R-Acc-SVRG-G  
Input: accuracy  $\epsilon >0$  parameters  $\delta_0 = L,\beta >1$  , initial guess  $x_0\in \mathbb{R}^d$    
1: for  $t = 0,1,2,\ldots$  do   
2: Define  $f^{\delta_t}(x) = (1 / n)\sum_{i = 1}^n f_i^{\delta_t}(x)$  , where  $f_{i}^{\delta_{t}}(x) = f_{i}(x) + (\delta_{t} / 2)\| x - x_{0}\|^{2}$    
3: Initialize vectors  $z_0 = \tilde{x}_0 = x_0$  and set  $\tau_x,\tau_z,\alpha ,p,C_{\mathrm{IDC}},C_{\mathrm{IFC}}$  according to Proposition 5.1.   
4: for  $k = 0,1,2,\ldots$  do   
5:  $y_{k} = \tau_{x}z_{k} + (1 - \tau_{x})\tilde{x}_{k} + \tau_{z}\left(\delta_{t}(\tilde{x}_{k} - z_{k}) - \nabla f^{\delta_{t}}(\tilde{x}_{k})\right).$    
6:  $z_{k + 1} = \arg \min_{x}\left\{\left\langle \mathcal{G}_{k}^{\delta_{t}},x\right\rangle +(\alpha /2)\left\| x - z_{k}\right\|^{2} + (\delta_{t} / 2)\left\| x - y_{k}\right\|^{2}\right\} .$    
7: //G'  $\begin{array}{r}\triangleq \nabla f_{i_k}^{\delta_t}(y_k) - \nabla f_{i_k}^{\delta_t}(\tilde{x}_k) + \nabla f^{\delta_t}(\tilde{x}_k), \end{array}$  where  $i_k$  is sampled uniformly in [n].   
8:  $\tilde{x}_{k + 1} = \left\{ \begin{array}{ll}y_k & \text{with probability} p,\\ \tilde{x}_k & \text{with probability} 1 - p. \end{array} \right.$    
9: if  $^6 ||\nabla f(\tilde{x}_k)||\leq \epsilon$  then output  $\tilde{x}_k$  and terminate the algorithm.   
10: if under IDC and  $(1 + \frac{\delta_t}{\alpha})^k\geq \sqrt{C_{\mathrm{IDC}}} /\delta_t$  then break the inner loop.   
11: if under IFC and  $(1 + \frac{\delta_t}{\alpha})^k\geq \sqrt{C_{\mathrm{IFC}} / 2\delta_t}$  then break the inner loop.   
12: end for   
13:  $\delta_{t + 1} = \delta_t / \beta$    
14: end for

# 5 Near-Optimal Accelerated SVRG with Adaptive Regularization

Currently, there is no known stochastic method that directly achieves the optimal rate in  $\epsilon$ . To get near-optimal rates, the existing strategy is to use a carefully designed regularization technique [42, 2] with a method that solves strongly convex problems; see, e.g., [42, 2, 22, 11]. However, the regularization parameter requires the knowledge of  $R_0$  or  $\Delta_0$ , which significantly limits its practicality.

Inspired by the recently proposed adaptive regularization technique [27], we develop a near-optimal accelerated SVRG variant (Algorithm 4) that does not require the knowledge of  $R_0$  or  $\Delta_0$ . Note that this technique was originally proposed for NAG under the IDC assumption. Our development extends this technique to the stochastic setting, which brings an  $O(\sqrt{n})$  rate improvement. Moreover, we consider both IFC and IDC cases. Proofs in this section are provided in Appendix D.

Detailed design. Algorithm 4 has a "guess-and-check" framework. In the outer loop, we first define the regularized objective  $f^{\delta_t}$  using the current estimate of regularization parameter  $\delta_t$ , and then we initialize an accelerated SVRG method (the inner loop) to solve the  $\delta_t$ -strongly convex  $f^{\delta_t}$ . If the inner loop breaks at Step 10 or 11, indicating the poor quality of the current estimate  $\delta_t$ ,  $\delta_t$  will be divided by a fixed  $\beta$ . Thus, conceptually, we can adopt any method that solves strongly convex finite-sums at the optimal rate as the inner loop. However, since the constructions of Step 10 or 11 require some algorithm-dependent constants, we have to fix one method as the inner loop.

The inner loop we adopted is a loopless variant of BS-SVRG [63]. This is because (i) BS-SVRG is the fastest known accelerated SVRG variant (for ill-conditioned problems) and (ii) it has a simple scheme, especially after using the loopless construction [34]. However, its original guarantee is built upon  $\{z_k\}$ . Clearly, we cannot implement the stopping criterion (Step 9) on  $\|\nabla f(z_k)\|$ . Interestingly, we discover that its sequence  $\{\tilde{x}_k\}$  works perfectly in our regularization framework, even if we can neither establish convergence on  $f(\tilde{x}_k) - f(x^\star)$  nor on  $\|\tilde{x}_k - x^\star\|^2$ . Moreover, we find that the loopless construction significantly simplifies the parameter constraints of BS-SVRG, which originally involves  $\Theta(n)$ th-order inequality. We provide the detailed parameter choice as follows:

Proposition 5.1 (Parameter choice). In Algorithm 4, we set  $\tau_{x} = \frac{\alpha + \delta_{t}}{\alpha + L + \delta_{t}}$ ,  $\tau_{z} = \frac{\tau_{x}}{\delta_{t}} - \frac{\alpha(1 - \tau_{x})}{\delta_{t}L}$  and  $p = \frac{1}{n}$ . We set  $\alpha$  as the (unique) positive root of the cubic equation  $\left(1 - \frac{p(\alpha + \delta_t)}{\alpha + L + \delta_t}\right)\left(1 + \frac{\delta_t}{\alpha}\right)^2 = 1$  and specify  $C_\mathrm{IDC} = L^2 +\frac{L\alpha^2p}{L + (1 - p)(\alpha + \delta_t)}$ ,  $C_\mathrm{IFC} = 2L + \frac{2L\alpha^2p}{(L + (1 - p)(\alpha + \delta_t))\delta_t}$ . Under these choices, we have  $\frac{\alpha}{\delta_t} = O\big(n + \sqrt{n(L / \delta_t + 1)}\big)$ ,  $C_\mathrm{IDC} = O\big((L + \delta_t)^2\big)$ , and  $C_\mathrm{IFC} = O(L)$ .

Under the choices of  $\tau_{x}$  and  $\tau_{z}$ , the  $\alpha$  above is the optimal choice in our analysis. Then, we can characterize the progress of the inner loop in the following proposition:

Proposition 5.2 (The inner loop of Algorithm 4). Using the parameters specified in Proposition 5.1, after running the inner loop (Step 4-12) of Algorithm 4 for  $k$  iterations, we can conclude that

(i) under IDC, i.e.,  $\| x_0 - x^\star \| \leq R_0$  for some  $x^{\star} \in \mathcal{X}^{\star}$ ,

$$
\mathbb {E} \left[ \| \nabla f (\tilde {x} _ {k}) \| \right] \leq \left(\delta_ {t} + \left(1 + \frac {\delta_ {t}}{\alpha}\right) ^ {- k} \sqrt {C _ {\mathrm {I D C}}}\right) R _ {0},
$$

(ii) under IFC, i.e.,  $f(x_0) - f(x^{\star})\leq \Delta_0$

$$
\mathbb {E} \left[ \| \nabla f (\tilde {x} _ {k}) \| \right] \leq \left(\sqrt {2 \delta_ {t}} + \left(1 + \frac {\delta_ {t}}{\alpha}\right) ^ {- k} \sqrt {C _ {\mathrm {I F C}}}\right) \sqrt {\Delta_ {0}}.
$$

The above results motivate the design of Step 10 and 11. For example, in the IDC case, when the inner loop breaks at Step 10, using (i) above, we obtain  $\mathbb{E}\left[\|\nabla f(\tilde{x}_k)\|\right] \leq 2\delta_t R_0$ . Then, by discussing the relative size of  $\delta_t$  and a certain constant, we can estimate the complexity of Algorithm 4. The same methodology is used for the IFC case.

Theorem 5.1 (IDC case). Denote  $\delta_{\mathrm{IDC}}^{\star} = \frac{\epsilon q}{2R_0}$  for some  $q \in (0,1)$  and let the outer iteration  $t = \ell$  be the first time  $\delta_{\ell} \leq \delta_{\mathrm{IDC}}^{\star}$ . The following assertions hold:

(i) At outer iteration  $\ell$ , Algorithm 4 terminates with probability at least  $1 - q$ .<sup>9</sup>  
(ii) The total expected oracle complexity of the  $\ell + 1$  outer loops is

$$
O \left(\left(n \log \frac {L R _ {0}}{\epsilon q} + \sqrt {\frac {n L R _ {0}}{\epsilon q}}\right) \log \frac {L R _ {0}}{\epsilon q}\right).
$$

Theorem 5.2 (IFC case). Denote  $\delta_{\mathrm{IFC}}^{\star} = \frac{\epsilon^{2}q^{2}}{8\Delta_{0}}$  for some  $q\in (0,1)$  and let the outer iteration  $t = \ell$  be the first time  $\delta_{\ell}\leq \delta_{\mathrm{IFC}}^{\star}$ . The following assertions hold:

(i) At outer iteration  $\ell$ , Algorithm 4 terminates with probability at least  $1 - q$ .  
(ii) The total expected oracle complexity of the  $\ell + 1$  outer loops is

$$
O \left(\left(n \log \frac {\sqrt {L \Delta_ {0}}}{\epsilon q} + \frac {\sqrt {n L \Delta_ {0}}}{\epsilon q}\right) \log \frac {\sqrt {L \Delta_ {0}}}{\epsilon q}\right).
$$

Compared with regularized Katyusha in Table 1, the adaptive regularization approach drops the need to estimate  $R_0$  or  $\Delta_0$  at the cost of a mere log  $\frac{1}{\epsilon}$  factor in the non-dominant term (if  $\epsilon$  is small).

# 6 Discussion

In this work, we proposed several simple and practical schemes that complement existing works (Table 1). Admittedly, the new schemes are currently only limited to the unconstrained Euclidean setting, because our techniques heavily rely on the interpolation conditions (1) and (2). On the other hand, methods such as OGM [30], TM [51] and ITEM [55, 10], which also rely on these conditions, are still not known to have their proximal variants. We list a few future directions as follows.

(1) It is not clear how to naturally connect the parameters of M-OGM-G (Algorithm 2) to OGM-G (Algorithm 1). The parameters of both algorithms seem to be quite restrictive and hardly generalizable due to the special construction in (4). Does there exist an optimal method for minimizing the gradient norm that has a proper potential function (at each iteration)?  
(2) Is this new "momentum" in OGM-G beneficial for training neural nets? Other classic momentum schemes such as NAG [40] or heavy-ball momentum method [49] are extremely effective for this task [53], and they were also originally proposed for convex objectives.  
(3) Can we directly accelerate SARAH (L2S)? By extending OGM-G? It seems that existing stochastic acceleration techniques fail to accelerate SARAH (or result in poor dependence on  $n$  as in [16]).

# References

[1] Z. Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. Journal of Machine Learning Research, 18(1):8194-8244, 2017. 2, 6, 26  
[2] Z. Allen-Zhu. How to make the gradients small stochastically: Even faster convex and nonconvex sgd. In Advances in Neural Information Processing Systems, pages 1157-1167, 2018. 1, 2, 8  
[3] Z. Allen-Zhu and Y. Yuan. Improved SVRG for Non-Strongly-Convex or Sum-of-Non-Convex Objectives. In Proceedings of The 33rd International Conference on Machine Learning, pages 1080–1089, 2016. 7  
[4] Z. Allen-Zhu, Y. Li, R. M. de Oliveira, and A. Wigderson. Much Faster Algorithms for Matrix Scaling. In C. Umans, editor, 58th IEEE Annual Symposium on Foundations of Computer Science, pages 890-901, 2017. 1  
[5] A. Auslender and M. Teboulle. Interior gradient and proximal methods for convex and conic optimization. SIAM Journal on Optimization, 16(3):697-725, 2006. 6  
[6] N. Bansal and A. Gupta. Potential-Function Proofs for Gradient Methods. Theory of Computing, 15(4):1-32, 2019. 4  
[7] Y. Carmon, J. C. Duchi, O. Hinder, and A. Sidford. Lower bounds for finding stationary points ii: first-order methods. Mathematical Programming, 185(1-2), 2021. 1, 2, 3  
[8] C.-C. Chang and C.-J. Lin. LIBSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1-27:27, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm.13,14  
[9] M. B. Cohen, A. Madry, D. Tsipras, and A. Vladu. Matrix Scaling and Balancing via Box Constrained Newton's Method and Interior Point Methods. In IEEE 58th Annual Symposium on Foundations of Computer Science, pages 902–913. IEEE, 2017. 1  
[10] A. d'Aspremont, D. Scieur, and A. Taylor. Acceleration methods. arXiv preprint arXiv:2101.09545, 2021. 9  
[11] D. Davis and D. Drusvyatskiy. Complexity of finding near-stationary points of convex functions stochastically. arXiv preprint arXiv:1802.08556, 2018. 8  
[12] A. Defazio. On the Curved Geometry of Accelerated Optimization. In Advances in Neural Information Processing Systems, volume 32, pages 1764-1773, 2019. 6  
[13] A. Defazio, F. R. Bach, and S. Lacoste-Julien. SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. In Advances in Neural Information Processing Systems, pages 1646–1654, 2014. 14  
[14] J. Diakonikolas and C. Guzmán. Complementary Composite Minimization, Small Gradients in General Norms, and Applications to Regression Problems. arXiv preprint arXiv:2101.11041, 2021. 1  
[15] J. Diakonikolas and P. Wang. Potential Function-based Framework for Making the Gradients Small in Convex and Min-Max Optimization. arXiv preprint arXiv:2101.12101, 2021. 1, 4  
[16] D. Driggs, M. J. Ehrhardt, and C.-B. Schonlieb. Accelerating variance-reduced stochastic gradient methods. Mathematical Programming, 2020. doi: 10.1007/s10107-020-01566-2. 6, 9  
[17] Y. Drori. The exact information-based complexity of smooth convex minimization. Journal of Complexity, 39:1-16, 2017. 1, 6  
[18] Y. Drori and A. Taylor. On the oracle complexity of smooth strongly convex minimization. arXiv preprint arXiv:2101.09740, 2021. 1  
[19] Y. Drori and M. Teboulle. Performance of first-order methods for smooth convex minimization: a novel approach. Mathematical Programming, 145(1-2):451-482, 2014. 1, 3, 5, 6  
[20] D. Dua and C. Graff. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml. 13, 14  
[21] C. Fang, C. J. Li, Z. Lin, and T. Zhang. SPIDER: Near-Optimal Non-Convex Optimization via Stochastic Path-Integrated Differential Estimator. In Advances in Neural Information Processing Systems, pages 687–697, 2018. 1

[22] D. J. Foster, A. Sekhari, O. Shamir, N. Srebro, K. Sridharan, and B. Woodworth. The Complexity of Making the Gradient Small in Stochastic Convex Optimization. In Proceedings of the Thirty-Second Conference on Learning Theory, pages 1319–1345, 2019. 1, 2, 8  
[23] R. Ge, F. Huang, C. Jin, and Y. Yuan. Escaping From Saddle Points — Online Stochastic Gradient for Tensor Decomposition. In Proceedings of The 28th Conference on Learning Theory, pages 797–842, 2015. 1  
[24] S. Ghadimi and G. Lan. Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization: A generic algorithmic framework. SIAM Journal on Optimization, 22(4):1469-1492, 2012. 6  
[25] S. Ghadimi and G. Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341-2368, 2013. 1  
[26] S. Ghadimi and G. Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming, 156(1-2):59-99, 2016. 1  
[27] M. Ito and M. Fukuda. Nearly optimal first-order methods for convex optimization under gradient norm measure: An adaptive regularization approach. Journal of Optimization Theory and Applications, 188(3):770-804, 2021. 1, 2, 8  
[28] C. Jin, R. Ge, P. Netrapalli, S. M. Kakade, and M. I. Jordan. How to Escape Saddle Points Efficiently. In Proceedings of the 34th International Conference on Machine Learning, pages 1724-1732, 2017. 1  
[29] R. Johnson and T. Zhang. Accelerating Stochastic Gradient Descent using Predictive Variance Reduction. In Advances in Neural Information Processing Systems, pages 315-323, 2013. 3, 14  
[30] D. Kim and J. A. Fessler. Optimized first-order methods for smooth convex minimization. Mathematical Programming, 159(1):81-107, 2016. 1, 6, 9  
[31] D. Kim and J. A. Fessler. Another Look at the Fast Iterative Shrinkage/Thresholding Algorithm (FISTA). SIAM Journal on Optimization, 28(1):223-250, 2018. 1, 3  
[32] D. Kim and J. A. Fessler. Generalizing the optimized gradient method for smooth convex minimization. SIAM Journal on Optimization, 28(2):1920-1950, 2018. 1, 2, 3  
[33] D. Kim and J. A. Fessler. Optimizing the efficiency of first-order methods for decreasing the gradient of smooth convex functions. Journal of Optimization Theory and Applications, 188(1): 192-219, 2021. 1, 2, 3, 4, 5  
[34] D. Kovalev, S. Horváth, and P. Richtárik. Don't jump through hoops and remove those loops: SVRG and Katyusha are better without the outer loop. In Algorithmic Learning Theory, pages 451-467. PMLR, 2020. 6, 8  
[35] G. Lan. An optimal method for stochastic composite optimization. Mathematical Programming, 133(1-2):365-397, 2012. 6  
[36] G. Lan, Z. Li, and Y. Zhou. A unified variance-reduced accelerated gradient method for convex optimization. In Advances in Neural Information Processing Systems, volume 32, pages 10462-10472, 2019. 6, 7  
[37] B. Li, M. Ma, and G. B. Giannakis. On the Convergence of SARAH and Beyond. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, pages 223–233, 2020. 2, 14, 27  
[38] Z. Li. ANITA: An Optimal Loopless Accelerated Variance-Reduced Gradient Method. arXiv preprint arXiv:2103.11333, 2021. 6, 7  
[39] Q. Lin and L. Xiao. An Adaptive Accelerated Proximal Gradient Method and its Homotopy Continuation for Sparse Optimization. In Proceedings of the 31th International Conference on Machine Learning, pages 73-81, 2014. 1  
[40] Y. Nesterov. A method for solving the convex programming problem with convergence rate  $O(1 / k^2)$ . In Dokl. akad. nauk Sssr, volume 269, pages 543-547, 1983. 1, 2, 9  
[41] Y. Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2003. 1  
[42] Y. Nesterov. How to make the gradients small. Optima. Mathematical Optimization Society Newsletter, (88):10-11, 2012. 1, 2, 5, 8, 22, 27

[43] Y. Nesterov. Gradient methods for minimizing composite functions. Mathematical Programming, 140(1):125-161, 2013. 1  
[44] Y. Nesterov. Lectures on convex optimization, volume 137. Springer, 2018. 3, 18  
[45] Y. Nesterov, A. Gasnikov, S. Guminov, and P. Dvurechensky. Primal-dual accelerated gradient methods with small-dimensional relaxation oracle. Optimization Methods and Software, pages 1-38, 2020. 2  
[46] L. M. Nguyen, J. Liu, K. Scheinberg, and M. Takáč. SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient. In Proceedings of the 34th International Conference on Machine Learning, pages 2613-2621, 2017. 1, 2  
[47] N. H. Pham, L. M. Nguyen, D. T. Phan, and Q. Tran-Dinh. ProxSARAH: An efficient algorithmic framework for stochastic composite nonconvex optimization. Journal of Machine Learning Research, 21(110):1-48, 2020. 1  
[48] J. Platt. Sequential minimal optimization: A fast algorithm for training support vector machines. 1998. 14  
[49] B. T. Polyak. Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics, 4(5):1-17, 1964. 4, 9  
[50] U. G. Rothblum and H. Schneider. Scalings of matrices which have prespecified row sums and column sums via optimization. Linear Algebra and its Applications, 114:737-764, 1989. 1  
[51] B. V. Scoy, R. A. Freeman, and K. M. Lynch. The Fastest Known Globally Convergent First-Order Method for Minimizing Strongly Convex Functions. IEEE Control Systems Letters, 2(1): 49–54, 2017. 9  
[52] C. Song, Y. Jiang, and Y. Ma. Variance Reduction via Accelerated Dual Averaging for Finite-Sum Optimization. In Advances in Neural Information Processing Systems, volume 33, pages 833-844, 2020. 6, 7  
[53] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning, pages 1139–1147, 2013. 9  
[54] A. Taylor and F. Bach. Stochastic first-order methods: non-asymptotic and computer-aided analyses via potential functions. In Conference on Learning Theory, pages 2934–2992, 2019. 2, 3, 6, 7  
[55] A. Taylor and Y. Drori. An optimal gradient method for smooth strongly convex minimization. arXiv preprint arXiv:2101.09741, 2021. 1, 6, 9  
[56] A. B. Taylor, J. M. Hendrickx, and F. Glineur. Smooth strongly convex interpolation and exact worst-case performance of first-order methods. Mathematical Programming, 161(1-2):307-345, 2017. 3  
[57] P. Tseng. On accelerated proximal gradient methods for convex-concave optimization. https://www.mit.edu/~dimitrib/PTseng/papers/apgm.pdf, 2008. Accessed May 1, 2020. 6  
[58] B. E. Woodworth and N. Srebro. Tight Complexity Bounds for Optimizing Composite Objectives. In Advances in Neural Information Processing Systems, pages 3639-3647, 2016. 3, 7  
[59] L. Xiao and T. Zhang. A Proximal Stochastic Gradient Method with Progressive Variance Reduction. SIAM Journal on Optimization, 24(4):2057-2075, 2014. 3, 14  
[60] D. Zhou, P. Xu, and Q. Gu. Stochastic Nested Variance Reduction for Nonconvex Optimization. Journal of Machine Learning Research, 21:103:1-103:63, 2020. 1  
[61] K. Zhou, F. Shang, and J. Cheng. A Simple Stochastic Variance Reduced Algorithm with Fast Convergence Rates. In Proceedings of the 35th International Conference on Machine Learning, pages 5980-5989, 2018. 6  
[62] K. Zhou, Q. Ding, F. Shang, J. Cheng, D. Li, and Z.-Q. Luo. Direct Acceleration of SAGA using Sampled Negative Momentum. In Proceedings of the Twenty Second International Conference on Artificial Intelligence and Statistics, pages 1602-1610, 2019. 6  
[63] K. Zhou, A. M.-C. So, and J. Cheng. Boosting First-Order Methods by Shifting Objective: New Schemes with Faster Worst-Case Rates. In Advances in Neural Information Processing Systems, pages 15405–15416, 2020. 3, 8, 22

[64] Z. A. Zhu and L. Orecchia. Linear Coupling: An Ultimate Unification of Gradient and Mirror Descent. In 8th Innovations in Theoretical Computer Science Conference, volume 67 of LIPIcs, pages 3:1-3:22, 2017. 6
