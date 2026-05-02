# Tighter Analysis of Alternating Stochastic Gradient Method for Stochastic Nested Problems

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Stochastic nested optimization, including stochastic compositional, min-max and bilevel optimization, is gaining popularity in many machine learning applications. While the three problems share the common nested structure, existing works often treat them separately, and thus develop problem-specific algorithms and their analyses. Among various exciting developments, simple SGD-type updates (potentially on multiple variables) are still prevalent in solving this class of nested problems, but they are believed to have slower convergence rate compared to that of the non-nested problems. This paper unifies several SGD-type updates for stochastic nested problems into a single SGD approach that we term ALternating Stochastic gradient dEscenT (ALSET) method. By leveraging the hidden smoothness of the problem, this paper presents a tighter analysis of ALSET for stochastic nested problems. Under the new analysis, to achieve an  $\epsilon$ -stationary point of the nested problem, it requires  $\mathcal{O}(\epsilon^{-2})$  samples in total. Under certain regularity conditions, applying our results to stochastic compositional, min-max and reinforcement learning problems either improves or matches the best-known sample complexity in the respective cases. Our results explain why simple SGD-type algorithms in stochastic nested problems all work very well in practice without the need for further modifications.

# 1 Introduction

Stochastic gradient descent (SGD) methods [1] are prevalent in solving large-scale machine learning problems. Oftentimes, SGD is being applied to solve stochastic problems with a relatively simple structure. Specifically, applying SGD to minimize the function  $\mathbb{E}_{\xi}[f(x;\xi)]$  over the variable  $x \in \mathbb{R}^d$ , we have the iterative update  $x^{k+1} = x^k - \alpha \nabla f(x^k; \xi^k)$ , where  $\alpha > 0$  is the stepsize and  $\nabla f(x^k; \xi^k)$  is the stochastic gradient at the iterate  $x^k$  and the sample  $\xi^k$ . However, many problems in machine learning today, such as meta learning, deep learning, hyper-parameter optimization, and reinforcement learning, go beyond the above simple minimization structure (termed the non-nested problem thereafter). For example, the objective function may be the compositions of multiple functions, where each composition may introduce an additional expectation [2]; and, the objective function may depend on the solution of another optimization problem [3]. In these problems, how to apply SGD and what is the efficiency of running SGD is not fully-understood.

To answer these questions, in this paper, we consider the following form of stochastic nested optimization problems, which is a generalization of the non-nested problems, given by

$$
\min  _ {x \in \mathbb {R} ^ {d}} F (x) := \mathbb {E} _ {\xi} [ f (x, y ^ {*} (x); \xi) ] \quad (\text {u p p e r}) \tag {1a}
$$

$$
\text {s . t .} \quad y ^ {*} (x) = \underset {y \in \mathbb {R} ^ {d ^ {\prime}}} {\arg \min } \mathbb {E} _ {\phi} [ g (x, y; \phi) ] \quad (\text {l o w e r}) \tag {1b}
$$

where  $f$  and  $g$  are differentiable functions; and,  $\xi$  and  $\phi$  are random variables. In the optimization literature [4-6], the problem (1) is referred to as the stochastic bilevel problem, where the upper-level optimization problem depends on the solution of the lower-level optimization over  $y \in \mathbb{R}^{d'}$ , denoted as  $y^*(x)$ , which depends on the value of upper-level variable  $x \in \mathbb{R}^d$ .

The stochastic bilevel nested problem (1) encompasses two popular formulations with the nested structure: stochastic min-max problems and stochastic compositional problems. Therefore, results on the general nested problem (1) will also imply the results in the special cases. For example, if the lower-level objective  $g$  is the negative of the upper-level objective  $f$ , i.e.,  $g(x,y;\phi) := -f(x,y;\xi)$ , the stochastic bilevel problem (1) reduces to the stochastic min-max problem

$$
\text {I f} g (x, y; \phi) := - f (x, y; \xi) \Rightarrow \min  _ {x \in \mathbb {R} ^ {d}} F (x) := \max  _ {y \in \mathbb {R} ^ {d ^ {\prime}}} \mathbb {E} _ {\xi} [ f (x, y; \xi) ]. \tag {2}
$$

Motivated by applications in zero-sum games, adversarial learning and training GANs, significant efforts have been recently made for solving the stochastic min-max problem; see e.g., [7-11].

For example, if the upper-level objective  $f$  is only a function of  $y$ , i.e.,  $f(x,y;\xi) \coloneqq f(y;\xi)$ , and the lower-level objective  $g$  is a quadratic function of  $y$ , i.e.,  $g(x,y;\phi) \coloneqq \| y - h(x;\phi)\|^2$  with a smooth function  $h$  of  $x$ , then the variable  $y^{*}(x)$  admits a closed-form solution, and thus the stochastic bilevel problem (1) reduces to the stochastic compositional problem [12-14]

$$
\text {I f} g (x, y; \phi) := \| y - h (x; \phi) \| ^ {2} \quad \Rightarrow \quad \min  _ {x \in \mathbb {R} ^ {d}} F (x) := \mathbb {E} _ {\xi} \left[ f \left(\mathbb {E} _ {\phi} [ h (x; \phi) ]; \xi\right) \right]. \tag {3}
$$

Stochastic compositional problems in the form of (3) have been studied in the applications in model-agnostic meta learning and policy evaluation in reinforcement learning; see e.g., [2, 15].

To solve the nested problem (1) by SGD, one natural solution is to apply alternating SGD updates on  $x$  and  $y$  based on their stochastic gradients

$$
y ^ {k + 1} = y ^ {k} - \beta_ {k} h _ {g} ^ {k} \quad \text {a n d} \quad x ^ {k + 1} = x ^ {k} - \alpha_ {k} h _ {f} ^ {k} \tag {4}
$$

where  $h_g^k$  is the unbiased stochastic gradient of  $\mathbb{E}_{\phi}[g(x^k, y^k; \phi)]$  and  $h_f^k$  is the (possibly biased) stochastic gradient of  $F(x^k)$ ; and,  $\beta_k$  and  $\alpha_k$  are the stepsizes. A key challenge of running (4) for the nested problem is that (stochastic) gradient of the upper-level variable  $x$  is prohibitively expensive to compute. As we will show later, computing an unbiased stochastic gradient of  $F(x)$  requires solving the lower-level problem exactly to obtain  $y^*(x)$ .

To obtain an accurate stochastic gradient  $h_f^k$ , there are roughly three ways in the literature. One way is to run SGD updates on  $y^{k}$  multiple times before updating  $x^{k}$ , which yields a double-loop algorithm. To guarantee convergence, it typically requires either the increasing number of lower-level  $y$ -update or the increasing number of batch size to estimate  $h_g^k$ ; see e.g., [16, 17]. The second way is to update  $y^{k}$  in a timescale faster than that of  $x^{k}$  so that  $x^{k}$  is relatively static with respect to  $y^{k}$ ; i.e.,  $\lim_{k\to \infty}\alpha_k / \beta_k = 0$ ; see e.g., [18]. The third way is to modify the direction  $h_g^k$  of  $y^{k}$  by incorporating additional correction term, which adds extra computation burden; see e.g., [19]. At a high level, these modifications either deviate from the originally light-weight implementation of SGD or sacrifice the sample complexity of SGD.

To this end, the main goal of this paper is to study the efficiency of running the vanilla alternating SGD (4) for the nested problem (1) and its implications on the special problem classes (2)-(3).

# 1.1 Main results

This paper aims to analyze a unifying algorithm for the stochastic bilevel problems that runs SGD on each variable in an alternating fashion, and provide sample complexity that matches the complexity of SGD for single-level stochastic problems. Our results explain why SGD-type algorithms in stochastic bilevel, min-max, and compositional problems all work very well in practice without the need for modifications such as correction, increasing batch size and two-timescale step sizes.

In the context of existing methods, our contributions can be summarized as follows.

C1) We connect three different classes of stochastic nested optimization problems (namely, stochastic compositional, min-max, and bilevel optimization), and unify three popular SGD-type updates for the respective problems into a single SGD-type approach that we term ALternating Stochastic gradient dEscenT (ALSET) method.

Table 1: Sample complexity of stochastic bilevel algorithms (BSA in [16], TTSA in [18], stocBiO in [17], STABLE in [19], SUSTAIN in [25], RSVRB in [26]) to achieve an  $\epsilon$ -stationary point of  $F(x)$ .  

<table><tr><td></td><td>ALSET</td><td>BSA</td><td>TTSA</td><td>stocBiO</td><td>STABLE</td><td>SUSTAIN/RSVRB</td></tr><tr><td>batch size</td><td>O(1)</td><td>O(1)</td><td>O(1)</td><td>O(ε-1)</td><td>O(1)</td><td>O(1)</td></tr><tr><td>y-update</td><td>SGD</td><td>O(ε-1) SGD steps</td><td>SGD</td><td>SGD</td><td>correction</td><td>momentum</td></tr><tr><td>samples inξ</td><td>O(ε-2)</td><td>O(ε-2)</td><td>O(ε-2.5)</td><td>O(ε-2)</td><td>O(ε-2)</td><td>O(ε-3/2)</td></tr><tr><td>samples inφ</td><td>O(ε-2)</td><td>O(ε-3)</td><td>O(ε-2.5)</td><td>O(ε-2)</td><td>O(ε-2)</td><td>O(ε-3/2)</td></tr></table>

C2) Under the same assumptions made in most of the previous work, we discover that the solution of the lower-level problem is smooth - a property that is overlooked by the previous analyses. By leveraging the hidden smoothness, we present a tighter analysis of ALSET for the stochastic bilevel problems. Under the new analysis, to achieve an  $\epsilon$ -stationary point of the nested problem, ALSET requires  $\mathcal{O}(\epsilon^{-2})$  samples in total, rather than the  $\mathcal{O}(\epsilon^{-2.5})$  sample complexity in the existing literature.  
C3) We further customize the analysis to the two special cases – the compositional and min-max problems, and establish the improved sample complexity relative to that in the literature. We apply our a new analysis to the celebrated actor-critic method for reinforcement learning problems. Under some regularity conditions, our new analysis implies that to achieve an  $\epsilon$ -stationary point, the single-loop actor-critic method requires  $\mathcal{O}(\epsilon^{-2})$  samples with i.i.d. sampling, which improves the  $\mathcal{O}(\epsilon^{-2.5})$  sample complexity in the existing literature.

# 1.2 Other related works

To put our work in context, we review prior art that we group in the following three categories.

Stochastic bilevel optimization. The study of bilevel optimization can be traced back to 1950s [20]. Many recent efforts have been made to solve the bilevel problems. One successful approach is to reformulate the bilevel problem as a single-level problem by replacing the lower-level problem by its optimality conditions [4, 5]. Recently, gradient-based methods for bilevel optimization have gained popularity, where the idea is to iteratively approximate the (stochastic) gradient of the upper-level problem either in forward or backward manner [21, 3, 22, 23]. Recent work has also studied the case where the lower-level problem does not have a unique solution [24].

The non-asymptotic analysis of bilevel optimization algorithms has been recently studied in some pioneering works, e.g., [16, 18, 17], just to name a few. In both [16, 17], bilevel stochastic optimization algorithms have been developed that run in a double-loop manner. To achieve an  $\epsilon$ -stationary point, they only need the sample complexity  $\mathcal{O}(\epsilon^{-2})$  that is comparable to the complexity of SGD for the single-level case. Recently, a single-loop two-timescale stochastic approximation algorithm has been developed in [18] for the bilevel problem (1). Due to the nature of two-timescale update, it incurs the sub-optimal sample complexity  $\mathcal{O}(\epsilon^{-2.5})$ . A single-loop single-timescale stochastic bilevel optimization method has been recently developed in [19]. While the method can achieve the sample complexity  $\mathcal{O}(\epsilon^{-2})$ , the resultant update on  $y$  needs extra matrix projection, which can be costly. Very recently, the momentum-based acceleration has been incorporated into the  $x-$  and  $y$ -updates in [25, 26], where the new algorithms enjoy an improved sample complexity  $\mathcal{O}(\epsilon^{-3/2})$ . However, these results cannot imply the  $\mathcal{O}(\epsilon^{-2})$  sample complexity of the alternating SGD update (4), and are orthogonal to our results. A comparison of our results with prior work can be found in Table 1.

Stochastic min-max optimization. In the context of min-max problems, the alternating version of the stochastic gradient descent ascent (GDA) method can be viewed as the alternating SGD updates (4) for the special nested problem (2). To mitigate the cycling behavior of GDA for convex-concave

Table 2: Sample complexity of stochastic min-max algorithms (BSA in [16], GDA in [27], SMD in [9]) to achieve an  $\epsilon$ -stationary point of  $F(x)$ .  

<table><tr><td></td><td>ALSET</td><td>SGDA</td><td>SMD</td></tr><tr><td>batch size</td><td>O(1)</td><td>O(ε-1)</td><td>/</td></tr><tr><td>y-update</td><td>SGD</td><td>SGD</td><td>subproblem</td></tr><tr><td>samples</td><td>O(κ3ε-2)</td><td>O(κ3ε-2)</td><td>O(κ3ε-2)</td></tr></table>

Table 3: Sample complexity of stochastic compositional algorithms (SCGD in [12], NASA in [14]) to achieve an  $\epsilon$ -stationary point of  $F(x)$ .  

<table><tr><td></td><td>ALSET</td><td>SCGD</td><td>NASA</td></tr><tr><td>batch size</td><td>O(1)</td><td>O(1)</td><td>O(1)</td></tr><tr><td>y-update</td><td>SGD</td><td>SGD</td><td>correction</td></tr><tr><td>samples</td><td>O(ε-2)</td><td>O(ε-4)</td><td>O(ε-2)</td></tr></table>

min-max problems, several variants have been developed by incorporating the idea of optimism; see e.g., [7, 8, 11, 28]. The analysis of stochastic GDA in the nonconvex-strongly concave setting is closely related to this paper; e.g., [9, 10, 29, 27]. Specifically, for stochastic GDA (SGDA), the  $\mathcal{O}(\epsilon^{-2})$  sample complexity has been established in [27] under an increasing batch size  $\mathcal{O}(\epsilon^{-1})$ . As highlighted in [27], how to achieve the  $\mathcal{O}(\epsilon^{-2})$  sample complexity under an  $\mathcal{O}(1)$  constant batch size remains open. The reduction of our results to the min-max setting will provide an answer to this open question. In the same setting, accelerated GDA algorithms have been developed in [30-32]. Going beyond the one-side concave settings, algorithms and their convergence analysis have been studied for nonconvex-nonconcave min-max problems with certain benign structure; see e.g., [8, 33-35]. A comparison of our results with prior work can be found in Table 2.

Stochastic compositional optimization. Stochastic compositional gradient algorithms developed in [12, 36] can be viewed as the alternating SGD updates (4) for the special compositional problem (3). However, to ensure convergence, the algorithms [12, 36] use two sequences of variables being updated in two different time scales, and thus the complexity of [12] and [36] is worse than  $\mathcal{O}(\epsilon^{-2})$  of SGD for the non-compositional case. While most of existing algorithms rely on either two-timescale updates, the single-timescale single-loop approaches have been recently developed in [14, 37, 38], which achieve the sample complexity  $\mathcal{O}(\epsilon^{-2})$ , same as SGD for the non-nested problems. However, the algorithms proposed therein are not the vanilla alternating SGD update in the sense of (4). Other related compositional algorithms also include [39-41]. A comparison can be found in Table 3.

Organization. The basic background of bilevel optimization is reviewed, and the tighter analysis of the unifying ALSET method is presented in Section 2. The reduction of the main results to the special stochastic nested problems are provided in Section 3, and its applications to the actor-critic method is discussed in Section 4, followed by the conclusions in Section 5.

# 2 Improved Analysis of Alternating Stochastic Gradient Method

In this section, we will first provide background of bilevel problems and then introduce the general alternating stochastic gradient descent (ALSET) method for stochastic nested problems.

# 2.1 Preliminaries

We use  $\| \cdot \|$  to denote the  $\ell_2$  norm for vectors and Frobenius norm for matrices. For convenience, we define the deterministic functions as  $g(x,y) \coloneqq \mathbb{E}_{\phi}[g(x,y;\phi)]$  and  $f(x,y) \coloneqq \mathbb{E}_{\xi}[f(x,y;\xi)]$ .

We also define  $\nabla_{yy}^2 g(x,y)$  as the Hessian matrix of  $g$  with respect to  $y$  and define  $\nabla_{xy}^2 g(x,y)$  as

$$
\nabla_ {x y} ^ {2} g \big (x, y \big) := \left[ \begin{array}{c c c} \frac {\partial^ {2}}{\partial x _ {1} \partial y _ {1}} g \big (x, y \big) & \dots & \frac {\partial^ {2}}{\partial x _ {1} \partial y _ {d ^ {\prime}}} g \big (x, y \big) \\ & \dots & \\ \frac {\partial^ {2}}{\partial x _ {d} \partial y _ {1}} g \big (x, y \big) & \dots & \frac {\partial^ {2}}{\partial x _ {d} \partial y _ {d ^ {\prime}}} g \big (x, y \big) \end{array} \right].
$$

We make the following assumptions that are common in bilevel optimization literature [16-18, 26].

Assumption 1 (Lipschitz continuity). Assume that  $f, \nabla f, \nabla g, \nabla^2 g$  are respectively  $\ell_{f,0}, \ell_{f,1}, \ell_{g,1}, \ell_{g,2}$ -Lipschitz continuous; that is, for  $z_1 := [x_1; y_1]$ ,  $z_2 := [x_2; y_2]$ , we have  $\| f(x_1, y_1) - f(x_2, y_2) \| \leq \ell_{f,0} \| z_1 - z_2 \|$ ,  $\| \nabla f(x_1, y_1) - \nabla f(x_2, y_2) \| \leq \ell_{f,1} \| z_1 - z_2 \|$ ,  $\| \nabla g(x_1, y_1) - \nabla g(x_2, y_2) \| \leq \ell_{g,1} \| z_1 - z_2 \|$ ,  $\| \nabla^2 g(x_1, y_1) - \nabla^2 g(x_2, y_2) \| \leq \ell_{g,2} \| z_1 - z_2 \|$ .

Assumption 2 (Strong convexity of  $g$  in  $y$ ). For any fixed  $x$ ,  $g(x, y)$  is  $\mu_g$ -strongly convex in  $y$ .

Assumptions 1 and 2 together ensure that the first- and second-order derivations of  $f(x,y), g(x,y)$  as well as the solution mapping  $y^{*}(x)$  are well-behaved. Define the condition number  $\kappa \coloneqq \ell_{g,1} / \mu_g$ .

Assumption 3 (Stochastic derivatives). The stochastic derivatives  $\nabla f(x,y;\xi)$ ,  $\nabla g(x,y;\phi)$ ,  $\nabla^2 g(x,y,\phi)$  are unbiased estimators of  $\nabla f(x,y)$ ,  $\nabla g(x,y)$ ,  $\nabla^2 g(x,y)$ , respectively; and their variances are bounded by  $\sigma_f^2$ ,  $\sigma_{g,1}^2$ ,  $\sigma_{g,2}^2$ , respectively.

Assumptions 2 and 3 together imply that the second moments are bounded by

$$
\mathbb {E} _ {\xi} \left[ \| \nabla f (x, y; \xi) \| ^ {2} \right] \leq \ell_ {f, 0} ^ {2} + \sigma_ {f} ^ {2} := C _ {f} ^ {2} \tag {5a}
$$

$$
\mathbb {E} _ {\phi} \left[ \| \nabla^ {2} g (x, y; \phi) \| ^ {2} \right] \leq \ell_ {g, 1} + \sigma_ {g, 2} ^ {2} := C _ {g} ^ {2}. \tag {5b}
$$

Assumption 3 is the counterpart of the unbiasedness and bounded variance assumption in the single-level stochastic optimization. In addition, the bounded moments in Assumption 3 ensure the Lipschitz continuity of the upper-level gradient  $\nabla F(x)$ .

We first highlight the inherent challenge of directly applying the alternating SGD method to the bilevel problem (1). To illustrate this point, we derive the gradient of the upper-level function  $F(x)$  in the next proposition; see the proof in the supplementary document.

Proposition 1. Under Assumptions 1-3, we have the gradients

$$
\nabla F (x) = \nabla_ {x} f \left(x, y ^ {*} (x)\right) - \nabla_ {x y} ^ {2} g \left(x, y ^ {*} (x)\right) \left[ \nabla_ {y y} ^ {2} g \left(x, y ^ {*} (x)\right) \right] ^ {- 1} \nabla_ {y} f \left(x, y ^ {*} (x)\right). \tag {6}
$$

Furthermore,  $\nabla F(x)$  and  $y^{*}(x)$  are Lipschitz continuous with constants  $L_{F}, L_{y}$ , respectively.

Notice that obtaining an unbiased stochastic estimate of  $\nabla F(x)$  and applying SGD on  $x$  face two main difficulties: i) the gradient  $\nabla F(x)$  at  $x$  depends on the minimizer of the lower-level problem  $y^{*}(x)$ ; ii) even if  $y^{*}(x)$  is known, it is hard to apply the stochastic approximation to obtain an unbiased estimate of  $\nabla F(x)$  since  $\nabla F(x)$  is nonlinear in  $\nabla_{yy}^2 g(x,y^* (x))$ .

Similar to some existing stochastic bilevel algorithms [16, 18, 17], we evaluate  $\nabla F(x)$  on a certain vector  $y$  in place of  $y^{*}(x)$ . Replacing the  $y^{*}(x)$  in definition (6) by  $y$ , we define

$$
\overline {{\nabla}} _ {x} f (x, y) := \nabla_ {x} f (x, y) - \nabla_ {x y} ^ {2} g (x, y) [ \nabla_ {y y} ^ {2} g (x, y) ] ^ {- 1} \nabla_ {y} f (x, y). \tag {7}
$$

And to reduce the bias in (7), we estimate  $\left[\nabla_{yy}^2 g(x,y)\right]^{-1}$  via

$$
\left[ \nabla_ {y y} ^ {2} g (x, y) \right] ^ {- 1} \approx \left[ \frac {N}{\ell_ {g , 1}} \prod_ {n = 1} ^ {N ^ {\prime}} \left(I - \frac {1}{\ell_ {g , 1}} \nabla_ {y y} ^ {2} g (x, y; \phi_ {(n)})\right) \right] \tag {8}
$$

where  $N'$  is drawn from  $\{1,2,\ldots,N\}$  uniformly at random and  $\{\phi^{(1)},\dots,\phi^{(N')}\}$  are i.i.d. samples. It has been shown in [16] that using (8), the estimation bias of  $\left[\nabla_{yy}^2 g(x,y)\right]^{-1}$  exponentially decreases with the number of samples  $N$ .

# 2.2 Main results: Tighter analysis of ALSET

In this subsection, we first describe the general ALSET algorithm for the stochastic bilevel problem, and then present its new convergence result.

This algorithm is very simple to implement. At each iteration  $k$ , ALSET alternates between the stochastic gradient update on  $y^{k}$  and that on  $x^{k}$ . Although it is possible that  $T = 1$ , for generality, we run  $T$  steps of SGD on

the lower-level variable  $y^{k}$  before updating upper-level variable  $x^{k}$ . With  $\alpha_{k}$  and  $\beta_{k}$  denoting the stepsizes of  $x^{k}$  and  $y^{k}$  that decrease at the same rate as SGD, the ALSET update is

$$
y ^ {k, t + 1} = y ^ {k, t} - \beta_ {k} h _ {g} ^ {k, t}, t = 0, \dots , T \quad \text {w i t h} \quad y ^ {k, 0} := y ^ {k}; y ^ {k + 1} := y ^ {k, T} \tag {9a}
$$

$$
x ^ {k + 1} = x ^ {k} - \alpha_ {k} h _ {f} ^ {k} \tag {9b}
$$

where the update direction of  $y$  is the stochastic gradient  $h_{g}^{k,t} \coloneqq \nabla_{y} g(x^{k}, y^{k,t}; \phi^{k,t})$ ; and, with the Hessian inverse estimator (8), the update direction of  $x$  is the slightly biased gradient

$$
\begin{array}{l} h _ {f} ^ {k} := \nabla_ {x} f (x ^ {k}, y ^ {k + 1}; \xi^ {k}) \\ \left. - \nabla_ {x y} ^ {2} g \left(x ^ {k}, y; \phi_ {(0)} ^ {k}\right) \left[ \frac {N}{\ell_ {g , 1}} \prod_ {n = 1} ^ {N ^ {\prime}} \left(I - \frac {1}{\ell_ {g , 1}} \nabla_ {y y} ^ {2} g \left(x ^ {k}, y ^ {k + 1}; \phi_ {(n)} ^ {k}\right)\right) \right] \nabla_ {y} f \left(x ^ {k}, y ^ {k + 1}; \xi^ {k}\right). \right. \tag {10} \\ \end{array}
$$

The alternating update (9) serves as a template of running SGD on stochastic nested problems. As we will show in the subsequent sections, we can generate stochastic algorithms for min-max,

compositional, and even reinforcement learning problems following (9) as a template, but they differ in the particular forms of the stochastic gradients  $h_g^k$ ,  $h_f^k$  for the specific upper- and lower-level objective functions. See a summary of ALSET for the bilevel problem in Algorithm 1.

Comparison between ALSET with existing works. Readers who are familiar with recent developments on stochastic optimization for bilevel problems may readily recognize the similarities between the general ALSET update (1) that we will analyze and the SGD-based updates in BSA [16], TTSA [18] and stocBiO [17]. However, the update (1) is different from BSA in that the number of  $y$ -update, denoted as  $T$ , is a constant in (1) that does not grow with the accuracy  $\epsilon^{-1}$ ; the update (1) is different from stocBiO in that the stochastic gradient  $h_{g}^{k,t}$  used in the  $y$ -update (9a) is obtained by a fixed batch size that does not depend on the accuracy  $\epsilon^{-1}$ ; and, the update (1) is different from TTSA in that the stepsizes  $\alpha_{k}$  and  $\beta_{k}$  in (9) decrease at the same timescale.

We next present the convergence result of ALSET.

Theorem 1 (Nonconvex). Under Assumptions 1-3, define the constants as

$$
\bar {\alpha} _ {1} = \frac {1}{2 L _ {F} + 4 L _ {f} L _ {y} + \frac {L _ {f} L _ {y x}}{L _ {y} \eta}}, \quad \bar {\alpha} _ {2} = \frac {1 6 T \mu_ {g} \ell_ {g , 1}}{\left(\mu_ {g} + \ell_ {g , 1}\right) ^ {2} \left(8 L _ {f} L _ {y} + \eta L _ {y x} \tilde {C} _ {f} ^ {2} \bar {\alpha} _ {1}\right)} \tag {11}
$$

where  $\eta > 0$  is a control constant that will be specified in each special case to achieve the best sample complexity, choose the stepsizes as

$$
\alpha_ {k} = \min  \left\{\bar {\alpha} _ {1}, \bar {\alpha} _ {2}, \frac {\alpha}{\sqrt {K}} \right\} \quad \text {a n d} \quad \beta_ {k} = \frac {8 L _ {f} L _ {y} + \eta L _ {y x} \tilde {C} _ {f} ^ {2} \bar {\alpha} _ {1}}{4 T \mu_ {g}} \alpha_ {k} \tag {12}
$$

then for any  $T \geq 1$ , the iterates  $\{x^k\}$  and  $\{y^k\}$  generated by Algorithm 1 satisfy

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \left\| \nabla F \left(x ^ {k}\right) \right\| ^ {2} \right] = \mathcal {O} \left(\frac {1}{\sqrt {K}}\right) \text {a n d} \mathbb {E} \left[ \left\| y ^ {K} - y ^ {*} \left(x ^ {K}\right) \right\| ^ {2} \right] = \mathcal {O} \left(\frac {1}{\sqrt {K}}\right) \tag {13}
$$

where  $y^{*}(x^{K})$  is the minimizer of the lower-level problem in (1b).

Proposition 2. Under the same assumptions and the choice of parameters of Theorem 1, with  $\kappa := \frac{\ell_{g,1}}{\mu_g}$  being the condition number, select  $\alpha = \Theta(\kappa^{-2.5})$ ,  $T = \Theta(\kappa^4)$ ,  $\eta = \mathcal{O}(\kappa)$  in (12), and then

$$
\frac {1}{K} \sum_ {k = 0} ^ {K - 1} \mathbb {E} [ \| \nabla F (x ^ {k}) \| ^ {2} ] = \mathcal {O} \left(\frac {\kappa^ {3}}{K} + \frac {\kappa^ {2 . 5}}{\sqrt {K}}\right). \tag {14}
$$

Discussion of Theorem 1. Theorem 1 implies that the convergence rate of ALSET to the stationary point of (1) is  $\mathcal{O}(K^{-0.5})$ . Since each iteration of ALSET only uses  $\widetilde{\mathcal{O}}(1)$  samples (see Algorithm 1), the sample complexity to achieve an  $\epsilon$ -stationary point of (1) is  $\mathcal{O}(\epsilon^{-2})$ , which is on the same order of SGD's sample complexity for the single-level nonconvex problems [42], and improves the state-of-the-art single-loop TTSA's sample complexity  $\mathcal{O}(\epsilon^{-2.5})$  [18]. Compared to [17], ALSET achieves the same sample complexity both in terms of  $\epsilon$  and  $\kappa$ , without using an increasing batch size. Importantly, we obtain this tighter bound without introducing additional assumptions.

# 2.3 Proof sketch

In this subsection, we highlight the key steps of the proof towards Theorem 1, and highlight the differences between our analysis and the existing ones.

For simplicity of the convergence analysis, we define the following Lyapunov function as  $\mathbb{V}^k \coloneqq F(x^k) + \frac{L_f}{L_y} \| y^k - y^* (x^k) \|^2$ . We first quantify the difference between two Lyapunov functions as

$$
\mathbb {V} ^ {k + 1} - \mathbb {V} ^ {k} = \underbrace {F \left(x ^ {k + 1}\right) - F \left(x ^ {k}\right)} _ {\text {L e m m a 1}} + \frac {L _ {f}}{L _ {y}} (\underbrace {\| y ^ {k + 1} - y ^ {*} \left(x ^ {k + 1}\right) \| ^ {2} - \| y ^ {k} - y ^ {*} \left(x ^ {k}\right) \| ^ {2}} _ {\text {L e m m a 3}}). \tag {15}
$$

The difference in (15) consists of two difference terms: the first term quantifies the descent of the overall objective functions; the second term characterizes the descent of the lower-level errors.

We will first analyze the descent of the upper-level objective in the next lemma.

Lemma 1 (Descent of upper level). Suppose Assumptions 1-3 hold. Define  $\bar{h}_f^k \coloneqq \mathbb{E}[h_f^k | x^k, y^{k+1}]$  and  $\| \bar{h}_f^k - \overline{\nabla} f(x^k, y^{k+1}) \| \leq b_k$ . The sequence of  $x^k$  generated by Algorithm 1 satisfies

$$
\begin{array}{l} \mathbb {E} [ F (x ^ {k + 1}) ] - \mathbb {E} [ F (x ^ {k}) ] \leq - \frac {\alpha_ {k}}{2} \mathbb {E} [ \| \nabla F (x ^ {k}) \| ^ {2} ] - \left(\frac {\alpha_ {k}}{2} - \frac {L _ {F} \alpha_ {k} ^ {2}}{2}\right) \mathbb {E} [ \| \bar {h} _ {f} ^ {k} \| ^ {2} ] \\ + L _ {f} ^ {2} \alpha_ {k} \mathbb {E} [ \| y ^ {k + 1} - y ^ {*} (x ^ {k}) \| ^ {2} ] + \alpha_ {k} b _ {k} ^ {2} + \frac {L _ {F} \alpha_ {k} ^ {2}}{2} \tilde {\sigma} _ {f} ^ {2} \tag {16} \\ \end{array}
$$

where constants  $L_{f}, L_{F}, \sigma_{f}^{2}$  are defined in Lemma 4 of the supplementary document.

Lemma 1 implies that the descent of the upper-level objective functions depends on the error of the lower-level variable  $y^{k}$ . We will next analyze the error of the lower-level variable, which is the key step to improving the existing results.

Before we analyze the error of  $y^{k}$ , we introduce a lemma that characterizes the smoothness of  $y^{*}(x)$  and the bounded moments of  $h_f^k$ . The smoothness and the bounded moments have not been explored by previous analysis such as [16-18], and they play an essential role in our improved analysis of  $y^{k}$ .

Lemma 2 (Smoothness and boundedness). Under Assumptions 1 and 2, we have

$$
\left\| \nabla y ^ {*} \left(x _ {1}\right) - \nabla y ^ {*} \left(x _ {2}\right) \right\| \leq L _ {y x} \| x _ {1} - x _ {2} \|; \quad \mathbb {E} [ \| h _ {f} ^ {k} \| ^ {2} | x ^ {k}, y ^ {k + 1} ] \leq \tilde {C} _ {f} ^ {2} \tag {17}
$$

where  $L_{yx}$  and  $\tilde{C}_f^2$  depend on the constants defined in Assumptions 1-2.

Building upon Lemma 2, we establish the progress of the lower-level update.

Lemma 3 (Error of lower level). Suppose that Assumptions 1-3 hold, and  $y^{k + 1}$  is generated by running iteration (9) given  $x^k$ . If we choose  $\beta_{k} \leq \frac{2}{\mu_{g} + \ell_{g,1}}$ , then  $y^{k + 1}$  satisfies

$$
\mathbb {E} \left[ \| y ^ {k + 1} - y ^ {*} \left(x ^ {k}\right) \| ^ {2} \right] \leq \left(1 - \mu_ {g} \beta_ {k}\right) ^ {T} \mathbb {E} \left[ \| y ^ {k} - y ^ {*} \left(x ^ {k}\right) \| ^ {2} \right] + T \beta_ {k} ^ {2} \sigma_ {g, 1} ^ {2} \tag {18a}
$$

$$
\begin{array}{l} \mathbb {E} [ \| y ^ {k + 1} - y ^ {*} (x ^ {k + 1}) \| ^ {2} ] \leq \left(1 + L _ {f} L _ {y} \alpha_ {k} + \frac {\eta L _ {y x} \tilde {C} _ {\bar {f}} ^ {2}}{4} \alpha_ {k} ^ {2}\right) \mathbb {E} [ \| y ^ {k + 1} - y ^ {*} (x ^ {k}) \| ^ {2} ] \\ + \left(L _ {y} ^ {2} + \frac {L _ {y}}{4 L _ {f} \alpha_ {k}} + \frac {L _ {y x}}{4 \eta}\right) \alpha_ {k} ^ {2} \mathbb {E} [ \| \bar {h} _ {f} ^ {k} \| ^ {2} ] + \left(L _ {y} ^ {2} + \frac {L _ {y x}}{4 \eta}\right) \alpha_ {k} ^ {2} \tilde {\sigma} _ {f} ^ {2} \tag {18b} \\ \end{array}
$$

where  $\eta > 0$  is a fixed constant that will be chosen to obtain the tighter complexity bound.

Plugging (18a) into (18b), and selecting stepsizes  $\alpha_{k},\beta_{k}$  properly, we can show that

$$
\mathbb {E} \left[ \| y ^ {k + 1} - y ^ {*} \left(x ^ {k + 1}\right) \| ^ {2} \right] \leq (1 - \delta_ {1}) \mathbb {E} \left[ \| y ^ {k} - y ^ {*} \left(x ^ {k}\right) \| ^ {2} \right] + \delta_ {2} \mathbb {E} \left[ \| \bar {h} _ {f} ^ {k} \| ^ {2} \right] + \delta_ {3} T \sigma_ {g, 1} ^ {2} + \delta_ {4} \tilde {\sigma} _ {f} ^ {2} \tag {19}
$$

where the constants are  $\delta_1\in [0,1)$ ,  $\delta_{2} = \mathcal{O}(\alpha_{k})$ ,  $\delta_{3} = \mathcal{O}(\beta_{k}^{2})$ ,  $\delta_{4} = \mathcal{O}(\alpha_{k}^{2})$ . In our tighter analysis, the term  $\mathbb{E}[\| \bar{h}_f^k\| ^2 ]$  will be canceled, so choosing  $\alpha_{k} = \mathcal{O}(k^{-0.5})$  and  $\beta_{k} = \mathcal{O}(k^{-0.5})$  makes the variance terms decrease at the same order as SGD for stochastic non-nested problems.

As a comparison, the progress of the lower-level problem in [18, 17] can be summarized as

$$
\mathbb {E} \left[ \| y ^ {k + 1} - y ^ {*} \left(x ^ {k + 1}\right) \| ^ {2} \right] \leq \left(1 - \delta_ {1}\right) \mathbb {E} \left[ \| y ^ {k} - y ^ {*} \left(x ^ {k}\right) \| ^ {2} \right] + \delta_ {5} \sigma^ {2} \tag {20}
$$

where  $\sigma^2$  is some variance term, and the constant is  $\delta_5 = \mathcal{O}(\beta_k^2 +\alpha_k^2 /\beta_k)$  or  $\mathcal{O}(1 / B_k)$  with  $B_{k}$  being the batch size at iteration  $k$ . To balance the two terms in  $\delta_5 = \mathcal{O}(\beta_k^2 +\alpha_k^2 /\beta_k)$ , two timescales of stepsizes are needed; and to reduce  $\delta_5 = \mathcal{O}(1 / B_k)$ , a growing batch size  $B_{k} = \mathcal{O}(k)$  is needed.

# 3 Applications to Stochastic Min-Max and Compositional Problems

Building upon the general results for the bilevel problems in Section 2, this section will identify special features of the stochastic min-max and stochastic compositional problems, and customize the general results to yield state-of-the-art convergence results for two special nested problems.

# 3.1 Stochastic min-max problems

We first apply our results to the stochastic min-max problem (2). In this special case, the lower-level function is  $g(x,y;\phi) = -f(x,y;\xi)$ , and the bilevel gradient in (6) reduces to

$$
\nabla F (x) := \nabla_ {x} f (x, y ^ {*} (x)) + \nabla_ {x} y ^ {*} (x) ^ {\top} \nabla_ {y} f (x, y ^ {*} (x)) = \nabla_ {x} f (x, y ^ {*} (x)) \tag {21}
$$

where the second equality follows from the optimality condition of the lower-level problem, i.e.,  $\nabla_y f(x,y^* (x)) = 0$ . Similar to Section 2, we again approximate  $\nabla F(x)$  on a certain vector  $y$  in place of  $y^{*}(x)$ . Therefore, the alternating stochastic gradients for this special case are given by

$$
h _ {g} ^ {k, t} = - \nabla_ {y} f \left(x ^ {k}, y ^ {k, t}; \xi_ {1} ^ {k}\right) \text {a n d} h _ {f} ^ {k} = \nabla_ {x} f \left(x ^ {k}, y ^ {k + 1}; \xi_ {2} ^ {k}\right). \tag {22}
$$

Plugging the stochastic gradient into the general update (9), we summarize the update in Algorithm 2. When the number of  $y$ -update is  $T = 1$ , the ALSET algorithm reduces to the SGDA method in [27].

Proposition 3. Under the same assumptions and the choice of parameters as those in Theorem 1, select  $\alpha = \Theta (\kappa^{-1})$ ,  $T = \Theta (\kappa)$ ,  $\eta = 1$  in (12), we have

$$
\frac {1}{K} \sum_ {k = 0} ^ {K - 1} \mathbb {E} [ \| \nabla F (x ^ {k}) \| ^ {2} ] = \mathcal {O} \left(\frac {\kappa^ {2}}{K} + \frac {\kappa}{\sqrt {K}}\right). \tag {23}
$$

Proposition 3 implies that for the minmax problem, the convergence rate of ALSET to the stationary point of  $F(x) \coloneqq \max_{y \in \mathbb{R}^{d'}} \mathbb{E}_{\xi}[f(x,y;\xi)]$  is  $\mathcal{O}(K^{-0.5})$ . Since each iteration of ALSET only uses  $\mathcal{O}(1)$  samples (see Algorithm 2), the sample complexity to achieve an  $\epsilon$ -stationary point of (2) is  $\mathcal{O}(\epsilon^{-2})$ . Comparing with the results in [27], we achieve the same sample complexity without an increasing batch size  $\mathcal{O}(\epsilon^{-1})$ , and improve their sample com

plexity  $\mathcal{O}(\epsilon^{-2.5})$  under a fixed batch size. However, it is also worth mentioning that compared with [27], our analysis requires the additional Lipschitz continuity assumption of  $f(x,y)$ , which inherits from the analysis for the general bilevel problem.

# Algorithm 2 ALSET for min-max problems

1: initialize:  $x^0, y^0$ , stepsizes  $\{\alpha_k, \beta_k\}$ .  
2: for  $k = 0,1,\ldots ,K - 1$  do  
3: set  $y^{k,0} = y^k$  
4: for  $t = 0,1,\dots ,T - 1$  do  
5: update  $y^{k,t + 1} = y^{k,t} - \beta_k\nabla_yf(x^k,y^{k,t};\xi_1^k)$  
6: end for  
7: set  $y^{k + 1} = y^{k,T}$  
8: update  $x^{k + 1} = x^k -\alpha_k\nabla_xf(x^k,y^{k + 1};\xi_2^k)$  
9: end for

# 3.2 Stochastic compositional problems

In this section, we apply our results to the stochastic compositional problem (3). In this special case, the upper-level function is  $f(x,y;\xi) \coloneqq f(y;\xi)$ , and the lower-level function is  $g(x,y;\phi) = \| y - h(x;\phi)\|^{2}$ , and the bilevel gradient in (6) reduces to

$$
\begin{array}{l} \nabla F (x): = \nabla_ {x} f (x, y ^ {*} (x)) - \nabla_ {x y} ^ {2} g (x, y ^ {*} (x)) [ \nabla_ {y y} ^ {2} g (x, y ^ {*} (x)) ] ^ {- 1} \nabla_ {y} f (x, y ^ {*} (x)) \\ = \nabla h (x; \phi) ^ {\top} \nabla_ {y} f \left(y ^ {*} (x)\right) \tag {24} \\ \end{array}
$$

where we use the fact that  $\nabla_{yy}g(x,y;\phi) = \mathbf{I}_{d'\times d'}$ ,  $\nabla_{xy}g(x,y;\phi) = -\nabla h(x;\phi)^\top$ . Similar to Section 2, we again evaluate  $\nabla F(x)$  on a certain vector  $y$  in place of  $y^{\ast}(x)$ . Therefore, the alternating stochastic gradients  $h_f^k,h_g^{k,t}$  for this special case are much simpler in this case, given by

$$
h _ {g} ^ {k, t} = y ^ {k, t} - h \left(x ^ {k}; \phi^ {k, t}\right) \text {a n d} h _ {f} ^ {k} = \nabla h \left(x ^ {k}; \phi^ {k}\right) \nabla f \left(y ^ {k + 1}; \xi^ {k}\right). \tag {25}
$$

Plugging the stochastic gradient into the general update (9), we summarize the update in Algorithm 3. When  $T = 1$ , the ALSET algorithm reduces to SCGD proposed in [12].

In the supplementary document, we have verified that the standard assumptions of stochastic compositional optimization in [12, 36, 14, 40, 37] are sufficient for Assumptions 1-3 to hold.

Proposition 4. Under the same assumptions and the choice of parameters as those in Theorem 1, select  $T = 1$ ,  $\alpha = 1$ ,  $\eta = \frac{1}{L_{yx}}$  in (12), and then it holds

# Algorithm 3 ALSET for compositional problems

1: initialize:  $x^0, y^0$ , stepsizes  $\{\alpha_k, \beta_k\}$ .  
2: for  $k = 0,1,\ldots ,K - 1$  do  
3: update  $y^{k + 1} = y^k - \beta_k(y^k - g(x^k; \phi^k))$  
4: update  $x^{k + 1} = x^k -\alpha_k\nabla f(y^{k + 1};\xi^k)\nabla g(x^k;\phi^k)$  
5: end for

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \left\| \nabla F \left(x ^ {k}\right) \right\| ^ {2} \right] = \mathcal {O} \left(\frac {1}{\sqrt {K}}\right). \tag {26}
$$

Since each iteration of ALSET only uses  $\mathcal{O}(1)$  samples (see Algorithm 3), Proposition 4 implies that the sample complexity to achieve an  $\epsilon$ -stationary point of (3) is  $\mathcal{O}(\epsilon^{-2})$ . Comparing with the results of the SCGD method in [12], our result improves the sample complexity  $\mathcal{O}(\epsilon^{-4})$  under a fixed batch size. Importantly, our analysis does not introduce additional assumption compared to [12].

# 4 Applications to Single-Timescale Actor-Critic Method

In this section, we apply our tighter analysis to the actor-critic (AC) method with linear value function approximation [43], which can be viewed as a special case of the stochastic bilevel algorithm.

Consider a Markov decision process described by  $\mathcal{M} = \{\mathcal{S},\mathcal{A},\mathcal{P},R,\gamma \}$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $\mathcal{P}(s'|s,a)$  is the probability of transitioning to  $s' \in \mathcal{S}$  given state  $s \in \mathcal{S}$  and action  $a \in \mathcal{A}$ , and  $R(s,a,s')$  is the reward associated with  $(s,a,s')$ , and  $\gamma \in [0,1)$  is a discount factor. For a policy  $\pi_{\theta}$ , define the value function  $V_{\pi_{\theta}}(s)$  that satisfies the Bellman equation [44]

$$
V _ {\pi_ {\theta}} (s) = \mathbb {E} _ {a \sim \pi_ {\theta} (\cdot | s), s ^ {\prime} \sim \mathcal {P} (\cdot | s, a)} [ r (s, a, s ^ {\prime}) + \gamma V _ {\pi_ {\theta}} (s ^ {\prime}) ]. \tag {27}
$$

Given the state feature mapping  $\phi(\cdot): \mathcal{S} \to \mathbb{R}^{d_y}$ , we approximate the value function linearly as  $V_{\pi_\theta}(s) \approx \hat{V}_y(s) \coloneqq \phi(s)^\top y$ , where  $y \in \mathbb{R}^{d_y}$  is the critic parameter. The task of finding the best  $y$  such that  $V_{\pi_\theta}(s) \approx \hat{V}_y(s)$  is usually addressed by TD learning [45].

Defining the stationary distribution induced by the policy parameter  $\theta_{k}$  as  $\mu_{\theta_k}$  and the  $k$ th transition as  $\xi_{k} := (s_{k}, a_{k}, s_{k+1})$ , which is sampled from  $s_{k} \sim \mu_{\theta_{k}}$ ,  $a \sim \pi_{\theta_{k}}$ ,  $s_{k+1} \sim \mathcal{P}$ , the TD-error is

$$
\hat {\delta} \left(\xi_ {k}, y _ {k}\right) := r \left(s _ {k}, a _ {k}, s _ {k + 1}\right) + \gamma \phi \left(s _ {k + 1}\right) ^ {\top} y _ {k} - \phi \left(s _ {k}\right) ^ {\top} y _ {k} \tag {28}
$$

and the critic gradient  $h_{g}(\xi_{k},y_{k})\coloneqq \hat{\delta} (\xi_{k},y_{k})\nabla \hat{V}_{y_{k}}(s_{k})$ . We update the parameter  $y$  via

$$
y _ {k + 1} = \Pi_ {R _ {y}} \left(y _ {k} + \beta_ {k} h _ {g} \left(\xi_ {k}, y _ {k}\right)\right), \tag {29}
$$

where  $\beta_{k}$  is the critic stepsize, and  $\Pi_{R_y}$  is the projection to control the norm of the gradient. A pre-defined constant  $R_{y}$  will be specified in the supplementary document.

The goal of policy optimization is to solve  $\max_{\theta \in \mathbb{R}^d} F(\theta)$  with  $F(\theta) \coloneqq \mathbb{E}_{s \sim \eta}[V_{\pi_\theta}(s)]$ , where  $\eta$  is the initial distribution. Leveraging the value function approximation and the policy gradient theorem [46], we have the policy gradient  $h_f(\xi, \theta, y) \coloneqq \hat{\delta}(\xi, y) \psi_\theta(s, a)$ , which gives the policy update

$$
\theta_ {k + 1} = \theta_ {k} + \alpha_ {k} h _ {f} \left(\xi_ {k} ^ {\prime}, \theta_ {k}, y _ {k + 1}\right), \tag {30}
$$

where  $\alpha_{k}$  is the stepsize and  $\psi_{\theta}(s,a)\coloneqq \nabla \log \pi_{\theta}(a|s)$ . Note that the sample  $\xi_k^\prime \coloneqq (s_k^\prime ,a_k^\prime ,s_{k + 1}^\prime)$  used in (30) is independent from  $\xi_{k}$  in (29). Specifically,  $\xi_k^\prime$  is sampled from  $s_k^\prime \sim d_{\theta_k},a_k^\prime \sim$ $\pi_{\theta_k},s_{k + 1}^\prime \sim \mathcal{P}$  with  $d_{\theta_k}$  being the discounted state action visitation measure under  $\theta_{k}$ .

The alternating AC update (29)-(30) is a special case of ALSET, where the critic update is the lower-level update, and the actor update is the upper-level update.

Due to space limitation, we will directly present the results of the alternating AC next, and defer presentation of the proof and the corresponding assumptions, which are the counterparts of Assumptions 1-3 in the context of AC, to the supplementary document.

Theorem 2 (Actor-critic). Under the some regularity conditions that are specified in the supplementary document, selecting step size  $\alpha_{k} = \alpha = \mathcal{O}\left(\frac{1}{\sqrt{K}}\right)$ ,  $\beta_{k} = \beta = \mathcal{O}\left(\frac{1}{\sqrt{K}}\right)$ , it holds

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \mathbb {E} \| \nabla F \left(\theta_ {k}\right) \| _ {2} ^ {2} = \mathcal {O} \left(\frac {1}{\sqrt {K}}\right) + \epsilon_ {\text {a p p}} \tag {31}
$$

where  $\epsilon_{\mathrm{app}}$  defined in the supplementary document, captures the richness of the linear function class.

Both sides of Theorem 2. As an application of our tighter analysis, Theorem 2 establishes for the first time that the sample complexity of the single-loop alternating actor-critic method is  $\mathcal{O}(\epsilon^{-2})$ . On the positive side, this new result improves the previous complexity  $\mathcal{O}(\epsilon^{-2.5})$  for the single-loop AC [47], and  $\mathcal{O}(\epsilon^{-2}\log \epsilon^{-1})$  for the nested-loop AC [48], and matches  $\mathcal{O}(\epsilon^{-2})$  for AC with an exact critic oracle [49]. In addition to using two independent samples, one limitation of our result is that inheriting from the analysis for the general bilevel case, our analysis of AC requires the smoothness of the critic fixed-point  $y^{*}(\theta)$ . As shown in the supplementary document, this implicitly requires the additional bounded and Lipschitz continuity assumption on the stationary distribution  $\mu_{\theta}$ . The removal of this assumption and the extension to Markovian sampling are left for future research.

# References

[1] H. Robbins and S. Monro, “A stochastic approximation method,” Annals of Mathematical Statistics, vol. 22, no. 3, pp. 400–407, Sep. 1951.  
[2] C. Finn, P. Abbeel, and S. Levine, "Model-agnostic meta-learning for fast adaptation of deep networks," in Proc. Intl. Conf. Machine Learn., Sydney, Australia, Jun. 2017, pp. 1126-1135.  
[3] L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel programming for hyperparameter optimization and meta-learning," in Proc. Intl. Conf. Machine Learn., Vienna, Austria, Jun. 2018, pp. 1568-1577.  
[4] B. Colson, P. Marcotte, and G. Savard, “An overview of bilevel optimization,” Annals of operations research, vol. 153, no. 1, pp. 235–256, 2007.  
[5] G. Kunapuli, K. P. Bennett, J. Hu, and J.-S. Pang, "Classification model selection via bilevel programming," Optimization Methods & Software, vol. 23, no. 4, pp. 475-489, 2008.  
[6] S. Dempe and A. Zemkoho, Bilevel Optimization. Springer, 2020.  
[7] C. Daskalakis and I. Panageas, "The limit points of (optimistic) gradient descent in min-max optimization," in Proc. Advances in Neural Info. Process. Syst., Montreal, Canada, Dec. 2018, pp. 9256-9266.  
[8] G. Gidel, H. Berard, G. Vignoud, P. Vincent, and S. Lacoste-Julien, “A variational inequality perspective on generative adversarial networks,” in Proc. Intl. Conf. Learn. Representations, Vancouver, Canada, Apr. 2018.  
[9] H. Rafique, M. Liu, Q. Lin, and T. Yang, "Non-convex min-max optimization: Provable algorithms and applications in machine learning," Optimization Methods and Software, Mar. 2021.  
[10] K. K. Thekumparampil, P. Jain, P. Netrapalli, and S. Oh, "Efficient algorithms for smooth minimax optimization," Dec. 2019.  
[11] A. Mokhtari, A. Ozdaglar, and S. Pattathil, "A unified analysis of extra-gradient and optimistic gradient methods for saddle point problems: Proximal point approach," in Proc. Intl. Conf. on Artif. Intell. and Stat., Palermo, Italy, Aug. 2020, pp. 1497-1507.  
[12] M. Wang, E. X. Fang, and H. Liu, "Stochastic compositional gradient descent: algorithms for minimizing compositions of expected-value functions," Mathematical Programming, vol. 161, no. 1-2, pp. 419-449, Jan. 2017.  
[13] B. Dai, N. He, Y. Pan, B. Boots, and L. Song, "Learning from conditional distributions via dual embeddings," in Proc. Intl. Conf. on Artif. Intell. and Stat., Fort Lauderdale, FL, Apr. 2017, pp. 1458-1467.  
[14] S. Ghadimi, A. Ruszczynski, and M. Wang, "A single timescale stochastic approximation method for nested stochastic optimization," SIAM Journal on Optimization, vol. 30, no. 1, pp. 960-979, Mar. 2020.  
[15] K. Ji, J. Yang, and Y. Liang, "Multi-step model-agnostic meta-learning: Convergence and improved algorithms," arXiv preprint:2002.07836, Feb. 2020.  
[16] S. Ghadimi and M. Wang, “Approximation methods for bilevel programming,” arXiv preprint:1802.02246, 2018.  
[17] K. Ji, J. Yang, and Y. Liang, "Provably faster algorithms for bilevel optimization and applications to meta-learning," in Proc. Intl. Conf. Machine Learn., Virtual, Jul. 2021.  
[18] M. Hong, H.-T. Wai, Z. Wang, and Z. Yang, “A two-timescale framework for bilevel optimization: Complexity analysis and application to actor-critic,” arXiv preprint:2007.05170, 2020.

[19] T. Chen, Y. Sun, and W. Yin, “A single-timescale stochastic bilevel optimization method,” arXiv preprint arXiv:2102.04671, 2021.  
[20] H. V. Stackelberg, The Theory of Market Economy. Oxford University Press, 1952.  
[21] S. Sabach and S. Shtern, "A first order method for solving convex bilevel optimization problems," SIAM Journal on Optimization, vol. 27, no. 2, pp. 640-660, 2017.  
[22] A. Shaban, C.-A. Cheng, N. Hatch, and B. Boots, "Truncated back-propagation for bilevel optimization," in Proc. Intl. Conf. on Artif. Intell. and Stat., Naha, Okinawa, Japan, Apr. 2019, pp. 1723-1732.  
[23] R. Grazzi, L. Franceschi, M. Pontil, and S. Salzo, "On the iteration complexity of hypergradient computation," in Proc. Intl. Conf. Machine Learn., virtual, Jul. 2020, pp. 3748-3758.  
[24] R. Liu, P. Mu, X. Yuan, S. Zeng, and J. Zhang, "A generic first-order algorithmic framework for bi-level programming beyond lower-level singleton," in Proc. of International Conference on Machine Learning, Virtual, July 2020, pp. 6305-6315.  
[25] P. Khanduri, S. Zeng, M. Hong, H.-T. Wai, Z. Wang, and Z. Yang, "A momentum-assisted single-timescale stochastic approximation algorithm for bilevel optimization," arXiv preprint arXiv:2102.07367, Feb. 2021.  
[26] Z. Guo and T. Yang, "Randomized stochastic variance-reduced methods for stochastic bilevel optimization," arXiv preprint arXiv:2105.02266, May 2021.  
[27] T. Lin, C. Jin, and M. Jordan, "On gradient descent ascent for nonconvex-concave minimax problems," in Proc. Intl. Conf. Machine Learn., virtual, Jul. 2020, pp. 6083-6093.  
[28] T. Yoon and E. K. Ryu, "Accelerated algorithms for smooth convex-concave minimax problems with  $O(1 / k^2)$  rate on squared gradient norm," in Proc. Intl. Conf. Machine Learn., Virtual, Jul. 2021.  
[29] M. Nouiehed, M. Sanjabi, T. Huang, J. D. Lee, and M. Razaviyayn, "Solving a class of nonconvex min-max games using iterative first order methods," in Proc. Advances in Neural Info. Process. Syst., Vancouver, Canada, Dec. 2019, pp. 14 934-14 942.  
[30] L. Luo, H. Ye, Z. Huang, and T. Zhang, "Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems," in Proc. Advances in Neural Info. Process. Syst., Virtual, Dec. 2020.  
[31] Y. Yan, Y. Xu, Q. Lin, W. Liu, and T. Yang, "Optimal epoch stochastic gradient descent ascent methods for min-max optimization," Proc. Advances in Neural Info. Process. Syst., vol. 33, Dec. 2020.  
[32] Q. Tran Dinh, D. Liu, and L. Nguyen, "Hybrid variance-reduced sgd algorithms for minimax problems with nonconvex-linear function," in Proc. Advances in Neural Info. Process. Syst., Virtual, Dec. 2020.  
[33] M. Liu, Y. Mroueh, J. Ross, W. Zhang, X. Cui, P. Das, and T. Yang, "Towards better understanding of adaptive gradient algorithms in generative adversarial nets," in Proc. Intl. Conf. Learn. Representations, Virtual, Apr. 2020.  
[34] J. Yang, N. Kiyavash, and N. He, "Global convergence and variance reduction for a class of nonconvex-nonconcave minimax problems," in Proc. Advances in Neural Info. Process. Syst., Virtual, Dec. 2020.  
[35] J. Diakonikolas, C. Daskalakis, and M. Jordan, "Efficient methods for structured nonconvex-nonconcave min-max optimization," in Proc. Intl. Conf. on Artif. Intell. and Stat., Virtual, Apr. 2021, pp. 2746-2754.  
[36] M. Wang, J. Liu, and E. Fang, "Accelerating stochastic composition optimization," Journal of Machine Learning Research, vol. 18, no. 1, pp. 3721-3743, 2017.

[37] T. Chen, Y. Sun, and W. Yin, "Solving stochastic compositional optimization is nearly as easy as solving stochastic optimization," arXiv preprint:2008.10847, Aug. 2020.  
[38] A. Ruszczyński, “A stochastic subgradient method for nonsmooth nonconvex multi-level composition optimization,” arXiv preprint:2001.10669, Jan. 2020.  
[39] X. Lian, M. Wang, and J. Liu, "Finite-sum composition optimization via variance reduced gradient descent," in Proc. Intl. Conf. on Artif. Intell. and Stat., Fort Lauderdale, FL, Apr. 2017.  
[40] J. Zhang and L. Xiao, "A stochastic composite gradient method with incremental variance reduction," in Proc. Advances in Neural Info. Process. Syst., Vancouver, Canada, Dec. 2019, pp. 9075-9085.  
[41] Q. Tran-Dinh, N. Pham, and L. Nguyen, "Stochastic gauss-newton algorithms for nonconvex compositional optimization," in Proc. Intl. Conf. Machine Learn., Virtual, Jul. 2020, pp. 9572-9582.  
[42] S. Ghadimi and G. Lan, "Stochastic first-and zeroth-order methods for nonconvex stochastic programming," SIAM Journal on Optimization, vol. 23, no. 4, pp. 2341-2368, 2013.  
[43] V. Konda and V. Borkar, "Actor-critic-type learning algorithms for markov decision processes," SIAM Journal on Control and Optimization, vol. 38, no. 1, pp. 94-123, 1999.  
[44] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT Press, 2018.  
[45] R. Sutton, “Learning to predict by the methods of temporal differences,” Machine Learning, vol. 3, pp. 9–44, 1988.  
[46] R. Sutton, D. McAllester, S. Singh, and Y. Mansour, "Policy gradient methods for reinforcement learning with function approximation," in Proc. Advances in Neural Info. Process. Syst., 2000.  
[47] Y. Wu, W. Zhang, P. Xu, and Q. Gu, "A finite time analysis of two time-scale actor critic methods," in Proc. Advances in Neural Info. Process. Syst., 2020.  
[48] T. Xu, Z. Wang, and Y. Liang, "Improving sample complexity bounds for (natural) actor-critic algorithms," in Proc. Advances in Neural Info. Process. Syst., 2020.  
[49] Z. Fu, Z. Yang, and Z. Wang, "Single-timescale actor-critic provably finds globally optimal policy," in Proc. Intl. Conf. Learn. Representations, 2020.  
[50] Y. Nesterov, Introductory Lectures on Convex Optimization: A basic course. Berlin, Germany: Springer, 2013, vol. 87.  
[51] J. Bhandari, D. Russo, and R. Singal, “A finite time analysis of temporal difference learning with linear function approximation.” 2018.  
[52] T. Xu, Z. Wang, Y. Zhou, and Y. Liang, "Reanalysis of variance reduced temporal difference learning," in Proc. Intl. Conf. Learn. Representations, 2020.  
[53] K. Zhang, A. Koppel, H. Zhu, and T. Bāşar, “Global convergence of policy gradient methods to (almost) locally optimal policies,” arXiv preprint: 1906.08383, 2019.  
[54] A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan, "Optimality and approximation with policy gradient methods in markov decision processes," in Proc. of Thirty Third Conference on Learning Theory, 2020.  
[55] K. Doya, "Reinforcement learning in continuous time and space," Neural Computation, vol. 12, no. 1, pp. 219-245, 2000.  
[56] S. Qiu, Z. Yang, J. Ye, and Z. Wang, "On the finite-time convergence of actor-critic algorithm," in Optimization Foundations for Reinforcement Learning Workshop at Advances in Neural Information Processing Systems, 2019.
