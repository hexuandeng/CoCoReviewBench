# ON THE MARGINAL REGRET BOUND MINIMIZATION OF ADAPTIVE METHODS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Numerous adaptive algorithms such as AMSGrad and Radam have been proposed and applied to deep learning recently. However, these modifications do not improve the convergence rate of adaptive algorithms and whether a better algorithm exists still remains an open question. In this work, we propose a new motivation for designing the proximal function of adaptive algorithms, named as marginal regret bound minimization. Based on such an idea, we propose a new class of adaptive algorithms that not only achieves marginal optimality, but can also potentially converge much faster than any existing adaptive algorithms in the long term. We show the superiority of the new class of adaptive algorithms both theoretically and empirically using experiments in deep learning.

# 1 INTRODUCTION

Accelerating the convergence speed of optimization algorithms is one main concern of the machine learning community. After stochastic gradient descent (SGD) was introduced, quite a few variants of SGD have become popular, such as momentum (Polyak, 1964) and AdaGrad (Duchi et al., 2011). Instead of directly moving parameters in the negative direction of the gradient, AdaGrad proposed to scale the gradient by a matrix, which was the matrix in the proximal function of the composite mirror descent rule (Duchi et al., 2011). The diagonal version of AdaGrad designed this matrix to be the square root of the global average of the squared gradients. Duchi et al. (2011) proved that this algorithm could be faster than SGD when the gradients were sparse.

However, AdaGrad's performance is known to deteriorate when the gradients are dense, especially in high dimensional problems such as deep learning (Reddi et al., 2018). To tackle this issue, many new algorithms were proposed to boost the performances of AdaGrad. Most of these algorithms focused on changing the design of the matrix in the proximal function. For example, RMSProp (Tieleman & Hinton, 2012) and Adam (Kingma & Ba, 2015) changed the global average design in AdaGrad to the exponential moving average. However, Reddi et al. (2018) proved that such a modification had convergence issues in the presence of high frequency noises and added a max operation to the matrix of Adam, leading to the AMSGrad algorithm. Other modifications, such as Padam (Chen & Gu, 2018), AdaShift (Zhou et al., 2019), NosAdam (Huang et al., 2019), and Radam (Liu et al., 2019), were based on various designs of this matrix as well. However, all aforementioned works did not improve the convergence rate of AdaGrad and simply supported their designs using experiments and synthetic examples. A theoretical foundation for the design of this matrix that improves the convergence and guides future adaptive algorithms is very much needed.

In this work, we bring new insights to the design of the matrix in the proximal function. In particular, our major contributions in this paper are listed as follows

- We propose a new motivation for designing the proximal function in adaptive algorithms. Specifically, we have found a marginally optimal design, which is the best matrix at each time step through minimizing the marginal increment of the regret bound.

- Based on our proposal of marginal regret bound minimization, we create a new class of adaptive algorithms, named as AMX. We prove theoretically that AMX can converge with a regret bound of size  $\tilde{O}(\sqrt{\tau})$ , where  $\tau$  is smaller than  $T$ . Such a regret bound is potentially much smaller than those of common adaptive algorithms and can make AMX converge

much faster than any existing adaptive algorithms, depending on  $\tau$ . In the worst case, we show it is at least as fast as AMSGrad and AdaGrad under the same assumptions

- We evaluate AMX's empirical performance on different tasks in deep learning. All experiments show our algorithm can converge fast and achieve good testing performances.

# 2 BACKGROUND

Notation: We denote the set of all positive definite matrices in  $\mathbb{R}^{d\times d}$  by  $S_d^+$ . For any two vectors  $a, b \in \mathbb{R}^d$ , we use  $\sqrt{a}$  for element-wise square root,  $a^2$  for element-wise square,  $|a|$  for element-wise absolute value,  $a / b$  for element-wise division, and  $\max(a, b)$  for element-wise maximum between  $a$  and  $b$ . We also frequently use the notation  $g_{1:T,i} = [g_{1,i}, g_{2,i}, \dots, g_{T,i}]$ , i.e. the vector of all the  $i$ -th elements of vectors  $g_1, g_2, \dots, g_T$ . For a vector  $a$ , we use  $\mathrm{diag}(a)$  to represent the diagonal matrix whose diagonal entries are  $a$ . For two functions  $f(t), g(t), f(t) = o(g(t))$  means  $f(t) / g(t) \to 0$  as  $t$  goes to infinity. We use  $\tilde{O}(\cdot)$  to omit logarithm terms in big- $O$  notations. We say a space  $\mathcal{X}$  has a bounded diameter  $D_\infty$  if  $\|x - y\|_\infty \leq D_\infty, \forall x, y \in \mathcal{X}$ .

Online Learning Framework. We choose the online learning framework to analyze all the algorithms in this paper. In this framework, an algorithm picks a new  $x_{t} \in \mathcal{X}$  according to its update rule at each iteration  $t$ , where  $X \subseteq \mathbb{R}^{d}$  is the set of feasible values of  $x_{t}$ . The composite loss function  $f_{t} + \phi$  is then revealed, where  $\phi$  is the regularization function that controls the complexity of  $x$  and  $f_{t}$  can be considered as the instantaneous loss at  $t$ . In the convex setting,  $f_{t}$  and  $\phi$  are both convex functions. The regularized regret function is defined with respect to an optimal predictor  $x^{*}$  as

$$
R (T) = \sum_ {t = 1} ^ {T} f _ {t} \left(x _ {t}\right) - f _ {t} \left(x ^ {*}\right) + \phi \left(x _ {t}\right) - \phi \left(x ^ {*}\right).
$$

Our goal is to find algorithms that ensures a sub-linear regret, i.e.  $R(T) = o(T)$ , which means that the average regret converges to zero. For example, online gradient descent is proved to have a regret of  $O(\sqrt{dT})$  (Zinkevich, 2003), where  $d$  is the dimension size of  $\mathcal{X}$ . Note that stochastic optimization and online learning are basically interchangeable (Cesa-Bianchi et al., 2004). Therefore, we will refer to online algorithms and their stochastic counterparts using the same names. For example, we will use stochastic gradient descent (SGD) to represent online gradient descent as it is more well-known.

Composite Mirror Descent Setup. In this paper, we will revisit the general composite mirror descent method (Duchi et al., 2010b) used in the creation of the first adaptive algorithm, AdaGrad, to bring new insights into adaptive methods. Such a general framework is preferred because it covers a wide range of algorithms, including both SGD and all the adaptive methods, and thus simplifies the discussions. The composite mirror descent rule at the time step  $t + 1$  is to solve for

$$
x _ {t + 1} = \operatorname {a r g m i n} _ {x \in \mathcal {X}} \left\{\alpha_ {t} \left\langle g _ {t}, x \right\rangle + \alpha_ {t} \phi (x) + B _ {\psi_ {t}} \left(x, x _ {t}\right) \right\}, \tag {1}
$$

where  $g_{t}$  is the gradient,  $\phi (x)$  is the regularization function in the dual space, and  $\alpha_{t}$  is the step size. Also,  $\psi_t$  is a strongly convex and differentiable function, named as the proximal function and  $B_{\psi_t}(x,x_t)$  is the Bregman divergence associated with  $\psi_t$  defined as

$$
B _ {\psi_ {t}} (x, y) = \psi_ {t} (x) - \psi_ {t} (y) - \langle \nabla \psi_ {t} (y), x - y \rangle .
$$

The general update rule (1) is mostly determined by the function  $\psi_t$ . We first observe that it becomes the projected SGD algorithm when  $\psi_t(x) = x^T x$  and  $\phi(x) = 0$ .

$$
x _ {t + 1} = \operatorname * {a r g m i n} _ {x \in \mathcal {X}} \left\{\alpha_ {t} \left\langle g _ {t}, x \right\rangle + \| x - x _ {t} \| _ {2} ^ {2} \right\} = \Pi_ {\mathcal {X}} \left(x _ {t} - \alpha_ {t} g _ {t}\right), \tag {SGD}
$$

where  $\Pi_{\mathcal{X}}(x) = \operatorname*{argmin}_{y\in \mathcal{X}}\| x - y\| _2$  is the projection operation that ensures the updated parameter is in the original space. On the other hand, adaptive algorithms choose different proximal functions  $\psi_t(x) = \langle x,H_t x\rangle$ , where  $H_{t}$  can be any full or diagonal symmetric positive definite matrix.

$$
x _ {t + 1} = \operatorname {a r g m i n} _ {x \in \mathcal {X}} \left\{\alpha_ {t} \langle g _ {t}, x \rangle + \alpha_ {t} \phi (x) + \langle x - x _ {t}, H _ {t} (x - x _ {t}) \rangle \right\}, \tag {Adaptive}
$$

Another popular representation of adaptive algorithms is the generalized projection rule  $x_{t+1} = \Pi_{\mathcal{X},H_t}(x_t - \alpha_t H_t^{-1} g_t)$ , where  $\Pi_{\mathcal{X},H_t}(x) = \operatorname*{argmin}_{y \in \mathcal{X}} \| H_t^{1/2}(x - y) \|_2$ , which is used in a lot of

recent literature such as Reddi et al. (2018); Huang et al. (2019). We show that these two rules are actually equivalent when  $\phi(x) = 0$  in the Appendix A.1, so that the regret bounds found in different representations can be generalized. A few recent works have shown that adaptive algorithms work well with special designs of the step sizes  $\alpha_{t}$  (Choi et al., 2019; Vaswani et al., 2020). In this work, we choose the more standard  $\alpha_{t} = \alpha / \sqrt{t}$  as it is used in most analysis. Also, we restrain our discussions to diagonal matrix proximal functions in the main text, i.e.  $H_{t} = \mathrm{diag}(h_{t}), h_{t} \in \mathbb{R}^{d}$ . Discussions on extending our results to full matrix proximal functions are provided in Appendix B.

Different Designs of the Proximal Function. Recently, researchers have proposed numerous designs of  $H_{t} = \mathrm{diag}(h_{t})$ , such as AdaGrad (Duchi et al., 2011), Adam (Kingma & Ba, 2015), AMSGrad (Reddi et al., 2018) and NosAdam (Huang et al., 2019), to name a few. It's impossible to go over all the proposed designs in this section so we choose the two most famous designs to review. The first adaptive algorithm, AdaGrad, used the square root of the average of past gradient squares as the diagonal  $h_t$  of the matrix in the proximal function (Duchi et al., 2011), i.e.

$$
h _ {t} = (\frac {\sum_ {i = 1} ^ {t} g _ {i} ^ {2}}{t}) ^ {1 / 2} \quad \mathrm {(A d a G r a d)}
$$

Normally, a small constant  $\epsilon$  is added to  $h_t$  at each iteration. Some recent work have shown that tuning this constant can benefit the performance of adaptive algorithms (Zaheer et al., 2018; Savarese et al., 2019). However, in this work, we assume it is small and fixed for simplicity, as it is originally designed to compute the pseudo-inverse, or equivalently, avoid division by zero in the generalized projected descent. Kingma & Ba (2015) proposed Adam to replace the simple average by exponential moving average in AdaGrad, but Reddi et al. (2018) showed that there was a mistake in Adam's convergence analysis, which lead to divergence of Adam even in simple convex problems. They therefore proposed the following simple modification of Adam to ensure its convergence.

$$
h _ {t} = \max  _ {t} \left\{\frac {\sum_ {i = 1} ^ {t} \left(1 - \beta_ {2}\right) \beta_ {2} ^ {t - i} g _ {i} ^ {2}}{t}\right) ^ {1 / 2} \} \tag {AMSGrad}
$$

where  $\beta_{2} \in (0,1)$  is a constant. We propose the following theorem that generalizes the regret bounds for most of the designs of the diagonal matrix proximal function.

Theorem 2.1 Let the sequence  $\{x_{t}\}$  be defined by the update rule (1) and for any  $x^{*}$ , denote  $D_{t,\infty}^{2} = \| x_{t} - x^{*}\|_{\infty}^{2}$ . When  $\psi_t(x) = \langle x,H_tx\rangle$ , where  $H_{t} = \mathrm{diag}(h_{t,1},h_{t,2},\dots ,h_{t,d})$ , assume without loss of generality that  $\phi (x_1) = 0$ ,  $H_0 = 0$ , if  $(h_{t,i} / \alpha_t)\geq (h_{t - 1,i} / \alpha_{t - 1})$ , then

$$
R (T) \leq \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {d} \left(\frac {h _ {t , i}}{\alpha_ {t}} - \frac {h _ {t - 1 , i}}{\alpha_ {t - 1}}\right) D _ {t, \infty} ^ {2} + \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {d} \frac {\alpha_ {t} g _ {t , i} ^ {2}}{2 h _ {t , i}}. \tag {2}
$$

The proof is relegated to Appendix A.3. The above regret bound is suitable for any designs of  $h_t$  that satisfy the constraint condition  $(h_{t,i} / \alpha_t) \geq (h_{t - 1,i} / \alpha_{t - 1})$ . Such a condition is crucial because if it is unsatisfied, the regret  $R(T)$  might diverge. In fact, the divergence of Adam in simple convex problems results from not satisfying this constraint, which is proved by Reddi et al. (2018). With  $\alpha_{t} = \alpha /\sqrt{t}$  in Theorem 2.1, most of recent adaptive algorithms have a regret bound of the following form (Duchi et al., 2011; Reddi et al., 2018; Huang et al., 2019; Luo et al., 2019).

$$
R (T) \leq C _ {1} \sqrt {T} \sum_ {i = 1} ^ {d} h _ {T, i} + f (T) \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2} + C _ {2} \tag {3}
$$

where  $C_1, C_2$  are constants and  $f(T) = o(\sqrt{T})$ . These algorithms are supposed to converge faster than SGD when the gradients are sparse or small, i.e. when  $\sum_{i=1}^{d} h_{T,i} \ll \sqrt{d}$  and  $\sum_{i=1}^{d} \|g_{1:T,i}\|_2 \ll \sqrt{dT}$ . However, all existing regret bounds are still  $O(\sqrt{T})$ , which makes it hard to compare different proximal functions. Whether the best proximal function exists and whether the  $O(\sqrt{T})$  regret bound can be further improved still remain open questions.

# 3 THE MOTIVATION-MARGINAL REGRET BOUND MINIMIZATION

In this section, we introduce the motivation behind our new class of algorithms. Although we find it difficult to determine the optimal proximal function globally, we show that it is possible to find the best proximal function at each iteration through marginal regret bound minimization. Denote  $\tilde{R}(T)$  to be the regret upper bound (the right hand side of inequality (2)) in Theorem 2.1. At time step  $T$ , we define the marginal regret bound increment  $\Delta \tilde{R}(T)$  as follows.

$$
\Delta \tilde {R} (T) := \tilde {R} (T) - \tilde {R} (T - 1) = \sum_ {i = 1} ^ {d} (\frac {h _ {T , i}}{\alpha_ {T}} - \frac {h _ {T - 1 , i}}{\alpha_ {T - 1}}) D _ {T, \infty} ^ {2} + \sum_ {i = 1} ^ {d} \frac {\alpha_ {T} g _ {T , i} ^ {2}}{2 h _ {T , i}}.
$$

As shown in the definition,  $\Delta \tilde{R}(T)$  will be the increment in the regret bound  $\tilde{R}(T)$  after  $h_T$  is determined. An important observation here is that  $h_{T-1}$  is a given constant at  $T$ , so  $\Delta \tilde{R}(T)$  is only a function of  $h_T$ . Therefore, the best design of  $h_T$  we can find at this moment is the one that minimizes  $\Delta \tilde{R}(T)$  and satisfy the constraint in Theorem 2.1. Consider the minimization problem

$$
\min  _ {h _ {T}} \Delta \tilde {R} (T), \text {s . t .} \frac {h _ {T , i}}{\alpha_ {T}} \geq \frac {h _ {T - 1 , i}}{\alpha_ {T - 1}} \geq 0, \tag {4}
$$

We propose the following proposition that solves the problem above

Proposition 3.1 With  $\alpha_{t} = \alpha /\sqrt{t}$ , the minimum of problem (4) is obtained at

$$
h _ {T} ^ {*} = \max  \left\{\sqrt {\frac {T - 1}{T}} h _ {T - 1}, \left(\frac {\alpha^ {2}}{2 T D _ {T , \infty} ^ {2}}\right) ^ {1 / 2} | g _ {T} | \right\}. \tag {5}
$$

To see this, set the function  $L$  with the Lagrangian multiplier  $\mu$  as follows

$$
L (h, \alpha) = \sum_ {i = 1} ^ {d} \left(\frac {h _ {T , i}}{\alpha_ {T}} - \frac {h _ {T - 1 , i}}{\alpha_ {T - 1}}\right) D _ {T, \infty} ^ {2} + \sum_ {i = 1} ^ {d} \frac {\alpha_ {T} g _ {T , i} ^ {2}}{2 h _ {T , i}} - \langle \frac {h _ {T}}{\alpha_ {T}} - \frac {h _ {T - 1}}{\alpha_ {T - 1}}, \mu \rangle .
$$

Take partial derivatives with respect to each  $h_{T,i}$ , we can see that  $D_{T,\infty}^{2} = (\alpha_{T}^{2}g_{T,i}^{2}) / 2h_{T,i}^{2} + \mu_{i}$ . By the complementary slackness conditions, either  $\mu_{i} = 0$  or  $h_{T,i} = (\alpha_{T} / \alpha_{T - 1})h_{T - 1,i}$ . When  $\mu_{i} = 0$ ,  $h_{T,i} = (\alpha_{T}^{2} / 2D_{T,\infty}^{2})^{1 / 2}|g_{T}|$  and the constraint condition  $h_{T,i}\geq (\alpha_T / \alpha_T - 1)h_{T - 1,i}$  needs to be satisfied. Hence, by setting  $\alpha_{T} = \alpha /\sqrt{T}$ , we can get the solution in (5).

Solution (5) is the best diagonal matrix proximal function in terms of regret bound increment at time  $T$ . Therefore, if we replace  $T$  by  $t$  in the subscripts, we can obtain a greedy choice of the proximal function  $h_t$  that minimizes the marginal regret bound increment at each time step. Intuitively, the reason this solution achieves the minimum is that it balances the two terms of  $\Delta \tilde{R}(T)$ . On each dimension  $(i)$ , it makes the first term of  $\Delta \tilde{R}(T)$  zero when the derivative is small  $^2$ , i.e. when we don't need to have an even larger  $h_{T,i}$  to slow down. When the derivative is too large,  $h_{T,i}$  adapts to the size of the derivative so that the second term of  $\Delta \tilde{R}(T)$  is constrained.

However, similar to the other greedy algorithms, solution (5) is only suboptimal as it minimizes the regret bound marginally instead of globally. Moreover, the parameter  $D_{t,\infty}$  is often unknown during the optimization process because  $x^{*}$  is usually unknown. Therefore, stronger theoretical motivation is needed to trust that solution (5) or similar algorithms can be useful and beneficial.

# 4 A NEW CLASS OF ADAPTIVE ALGORITHMS - AMX

Now, motivated by the greedy choice of  $h_t$  in section 3, we focus on a more general design of the diagonal matrix in the proximal functions that has the following form and show why such greedy designs can be beneficial in the long term. Consider

$$
h _ {t} = \max  \left\{\sqrt {\frac {t - 1}{t}} h _ {t - 1}, c _ {t} | g _ {t} | \right\}. \tag {AMX}
$$

Algorithm 1 AMX Algorithm (Diagonal, Composite Mirror Descent Form)  
1: Input:  $x \in \mathcal{F}$ ,  $\alpha_{t} = \alpha / \sqrt{t}$ ,  $\{c_{t}\}_{t=1}^{T}$ ,  $\phi(x)$ ,  $\epsilon$   
2: Initialize  $h_{0} = 0$ ,  $H_{0} = 0$   
3: for  $t = 1$  to  $T$  do  
4:  $g_{t} = \nabla f_{t}(x_{t})$   
5:  $h_{t} = \max(\sqrt{\frac{t - 1}{t}} h_{t - 1}, c_{t}|g_{t}|) + \epsilon$   
6:  $x_{t + 1} = \operatorname{argmin}_{x \in \mathcal{X}} \{\alpha_{t}\langle g_{t}, x \rangle + \alpha_{t}\phi(x) + \langle x - x_{t}, \operatorname{diag}(h_{t})(x - x_{t}) \rangle\}$   
7: end for

where  $c_{t}$  is an arbitrary function of  $t$ , for example solution (5) or as simple as  $c_{t} = 1$ . The corresponding new class of adaptive algorithms is given in Algorithm 1, which we name as AMX. Note that the diagonal proximal function performs all operations coordinate-wisely, therefore we start our analysis from one dimension  $(i)$ . Denote  $(i)$ -th dimension component of the regret bound as

$$
\tilde {R} ^ {(i)} (T) := \sum_ {t = 1} ^ {T} \left(\frac {h _ {t , i}}{\alpha_ {t}} - \frac {h _ {t - 1 , i}}{\alpha_ {t - 1}}\right) D _ {t, \infty} ^ {2} + \sum_ {t = 1} ^ {T} \frac {\alpha_ {t} g _ {t , i} ^ {2}}{2 h _ {t , i}}. \tag {6}
$$

For a sequence of gradients  $g_{1}, g_{2}, \dots, g_{T}$ , denote the time steps  $t$  when  $h_{t,i} = c_t |g_{t,i}|$  as  $\tau_1^{(i)}, \tau_2^{(i)}, \dots, \tau_{m_i}^{(i)}$ . Note that  $\tau_j^{(i)}$ 's may be different across different dimensions, so the analysis applies to each dimension independently. These are the time steps when the gradient term  $c_t |g_t|$  dominates  $h_t$  on the  $(i)$ -th dimension and  $h_{t,i} = c_t |g_{t,i}|$  when  $t = \tau_j^{(i)}, \forall j = 1, \dots, m_i$ . Since  $h_{t,i}$  is equal to  $\sqrt{\frac{t - 1}{t}} h_{t - 1,i}$  between  $\tau_j^{(i)}$ 's, which is exactly the same as in section 3, the increment in the first term of the right hand side of (6) is always 0. Therefore, we have the following proposition

Proposition 4.1 For any  $\tau \in (\tau_j^{(i)},\tau_{j + 1}^{(i)})$ , the regret bound increment on the  $(i)$ -th dimension is

$$
\tilde {R} ^ {(i)} (\tau) - \tilde {R} ^ {(i)} (\tau_ {j} ^ {(i)}) = \sum_ {t = \tau_ {j} ^ {(i)} + 1} ^ {\tau} \frac {\alpha_ {t} g _ {t , i} ^ {2}}{2 h _ {t , i}}.
$$

The above proposition indicates that the regret increments between the  $\tau_{j}^{(i)}$ 's are only related to the second term of  $\tilde{R}^{(i)}(T)$ . Note that the first term of  $\tilde{R}^{(i)}(T)$  is a major reason why the regret bound is  $O(\sqrt{T})$  because there is a  $1 / \alpha_{T}$  term in the summation. Therefore, such designs of  $h_{t,i}$  try to constrain the regret increments between  $\tau_{j}^{(i)}$ 's and hence can potentially make the regret bound small. More importantly, Proposition 4.1 is true for the time steps between the last  $\tau_{m_i}^{(i)}$  and  $T + 1$ . Denote  $D_{\infty}$  to be the bounded diameter of the parameter space  $\mathcal{X}$ , because the regret bound increment is only related to the second term after the last  $\tau_{m_i}^{(i)}$ , the bound in equation (6) becomes

$$
\tilde {R} ^ {(i)} (T) \leq \sum_ {t = 1} ^ {T} \left(\frac {h _ {t , i}}{\alpha_ {t}} - \frac {h _ {t - 1 , i}}{\alpha_ {t - 1}}\right) D _ {\infty} ^ {2} + \sum_ {t = 1} ^ {T} \frac {\alpha_ {t} g _ {t , i} ^ {2}}{2 h _ {t , i}} \leq \frac {D _ {\infty} ^ {2} \sqrt {\tau_ {m _ {i}} ^ {(i)}}}{\alpha} h _ {\tau_ {m _ {i}} ^ {(i)}, i} + \sum_ {t = 1} ^ {T} \frac {\alpha}{2 \sqrt {t} c _ {t}} | g _ {t, i} | \tag {7}
$$

Since  $\tilde{R}(T) = \sum_{i=1}^{d} \tilde{R}^{(i)}(T)$ , so the total regret upper bound across all the dimensions is

$$
R (T) \leq \sum_ {i = 1} ^ {d} \tilde {R} ^ {(i)} (T) \leq \sum_ {i = 1} ^ {d} \frac {D _ {\infty} ^ {2} \sqrt {\tau_ {m _ {i}} ^ {(i)}}}{\alpha} h _ {\tau_ {m _ {i}, i} ^ {(i)}, i} + \sum_ {i = 1} ^ {d} \sum_ {t = 1} ^ {T} \frac {\alpha}{2 \sqrt {t} c _ {t}} | g _ {t, i} | \tag {8}
$$

The first term of the right hand side can be considered as better than the  $O(\sqrt{T})$  regret bound of SGD or common adaptive algorithms when some or all of the  $\tau_{mi}^{(i)}$  are much smaller than  $T$  and  $h_{t,i}$ 's are bounded. Therefore if we can ensure the second term also increases much slower than  $O(\sqrt{T})$ , which is decided by the design of  $c_{t}$ , then the AMX class of algorithms is potentially much faster than SGD and the other adaptive algorithms. Note that we need  $h_{t,i}$  to be bounded in the above arguments, therefore  $c_{t}$  can be at most a constant. Fortunately, one simple yet very effective design that we have found is  $c_{t} = 1$ . We formalize the above statements in the following theorem.

Theorem 4.1 Let  $\{x_{t}\}$  and  $\{h_t\}$  be the sequences obtained from Algorithm 1,  $\alpha_{t} = \alpha /\sqrt{t},c_{t} = 1$  Let  $\{\tau_{m_i}^{(i)}\}_{i = 1}^d$  be the largest time steps  $t$  on each dimension when  $h_{t,i} = c_t|g_{t,i}|$ . Assume that  $\mathcal{F}$  has bounded diameter  $\| x - y\|_{\infty}\leq D_{\infty},\forall x,y\in \mathcal{F}$ . Then we have the following bound on the regret.

$$
R (T) \leq \sum_ {i = 1} ^ {d} \frac {D _ {\infty} ^ {2} \sqrt {\tau_ {m _ {i}} ^ {(i)}}}{\alpha} h _ {\tau_ {m _ {i}} ^ {(i)}, i} + \frac {\alpha}{2} \sum_ {i = 1} ^ {d} \sqrt {1 + \log \tau_ {m _ {i}} ^ {(i)}} \| g _ {1: \tau_ {m _ {i}} ^ {(i)}, i} \| _ {2} + \frac {\alpha}{2} \sum_ {i = 1} ^ {d} \sqrt {\tau_ {m _ {i}} ^ {(i)}} \log \left(\frac {T}{\tau_ {m _ {i}} ^ {(i)}}\right) | g _ {\tau_ {m _ {i}} ^ {(i)}, i} |, \tag {9}
$$

An important remark here is that using a different constant  $c_{t} = c$  in Theorem 4.1 is equivalent to tuning the step size by  $1 / c$ , as it magnifies all  $|g_t|$  at the same time in the algorithm. Using a decaying  $c_{t}$  can further improve the first term in Theorem 4.1, but it also enlarges the second and the third term so the regret may not be  $O(\sqrt{T})$  anymore. We will focus on  $c_{t} = 1$  to prove AMX can potentially converge faster in the rest of this paper, but a detailed discussion about the possible choices of  $c_{t}$  for future designs of the AMX algorithm is provided in Appendix A.5, which proves that  $c_{t}$  cannot be  $O(1 / \sqrt{t})$  if we do not impose any further assumptions.

Now, since most of the bound in Theorem 4.1 depends on the time steps  $\{\tau_{m_i}^{(i)}\}_{i = 1}^d$  instead of  $T$  (only a log term), the specific AMX algorithm can be potentially much faster than common adaptive algorithms. To further make our argument clearer, we propose the following corollary.

Corollary 4.1 Let  $\tau = \max_{i}\{\tau_{m_i}^{(i)}\}$  in Theorem 4.1, under the same assumptions as AdaGrad and AMSGrad, Algorithm 1 converges with regret bound

$$
O \left(\max  \left(\sqrt {\tau}, \sqrt {\tau} \log \frac {T}{\tau}\right)\right) \tag {10}
$$

The corollary indicates that the regret bound is approximately of size  $\tilde{O} (\sqrt{\tau})$  if we omit the log terms. As far as we are aware, this is the first algorithm that generates a regret bound that is not asymptotically  $O(\sqrt{T})$ , so AMX can be potentially much faster than any existing algorithms. The time step  $\tau$  can be understood as "the time when the gradients start to converge" and whether it makes the convergence faster depends on the distribution of the gradients. For example, if  $\tau = \sqrt{T}$ , the regret bound is only of size  $O(T^{1 / 4}\log T)$ . We emphasize that a small  $\tau$  is not an assumption on the gradient distribution, but rather a condition that once satisfied, the regret bound will only increase logarithmically and hence the algorithm converges very fast. Moreover, the regret bounds of the other adaptive algorithms go to  $O(\sqrt{T})$  even under such conditions, because their regret bound increments are not minimized. We use a rather simple example to illustrate why AMX has this unique advantage.

Example. Suppose that the domain is a hyper-cube  $\mathcal{X} = \{\| x\|_{\infty}\leq 1\}$ , then  $D_{\infty} = 2$ . Assume that on each dimension, the gradient decreases as  $|g_{t,i}| = (1 / \sqrt{t})|g_{1,i}|$ , and  $|g_{1,i}|\ll 1,\forall i$ . Note that this is one example where adaptive algorithms should work well since  $\| g_{1:T,i}\| _2\leq |g_{1,i}|\sqrt{(1 + \log T)}\ll \sqrt{T}$ . A very important property for AMX in this case is that  $\tau$  is the first time step, so its regret bound only increases logarithmically. However, the regret bounds of the other algorithms still goes to  $O(\sqrt{T})$ . We plot the regret bounds of AMX, AMSGrad and AdaGrad in Figure 4. Note that the regret bound of AMX increases much slower than AdaGrad and AMSGrad, hence it is much faster than these algorithms in this example. One may argue that the example is extreme since  $\tau = 1$  rarely happens in real situations. However, the regret in this example can be understood as the regret increment after  $\tau$  in real training processes, i.e. before  $\tau$ , AMX is only asymptotically as fast as the

other adaptive algorithms, but after  $\tau$ , since the regret increment of AMX is very small, it converges very fast. More details of this example can be found in Appendix D.1.

![](images/5c87b59eb067a9e90147b3ca840bd9e43f588d69b286079fddf2e630f02dbeed.jpg)  
Figure 1: The regret bounds of AMX, AdaGrad, AMSGrad in the example.

Besides, since the term  $\sqrt{\tau}\log (T / \tau)$  in Corollary 4.1 is at most  $O(\sqrt{T})^3$ , the AMX algorithm is at least as fast as AdaGrad and AMSGrad under the same assumptions. We propose the following theorem that corresponds to the general results in section 2 to prove our claim:

Theorem 4.2 Let  $\{x_{t}\}$  and  $\{h_t\}$  be the sequences obtained from Algorithm 1,  $\alpha_{t} = \alpha /\sqrt{t},c_{t} = 1$  Assume that  $\mathcal{F}$  has bounded diameter  $\| x - y\|_{\infty}\leq D_{\infty},\forall x,y\in \mathcal{F}$  . Then we have the following bound on the regret.

$$
R (T) \leq \frac {D _ {\infty} ^ {2} \sqrt {T}}{\alpha} \sum_ {i = 1} ^ {d} h _ {T, i} + \frac {\alpha}{2} \sqrt {1 + \log T} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2}, \tag {11}
$$

The above bound can be considered as being better than the regret of SGD, i.e.,  $O(\sqrt{dT})$ , when  $\sum_{i=1}^{d} h_{T,i} \ll \sqrt{d}$  and  $\sum_{i=1}^{d} \| g_{1:T,i} \|_2 \ll \sqrt{dT}$  (Duchi et al., 2011). Therefore, AMX can be at least much faster than SGD when the gradients are sparse or small, and it can be potentially even faster. To keep up with the current popular adaptive algorithms such as Adam, we also provide the detailed implementation of adding first order momentum into AMX and include some discussions on its convergence properties in Appendix C. Similar to Algorithm 1, the AMX with momentum algorithm has a regret bound that (mostly) depends on  $\tau$  instead of  $T$  and hence enjoys the acceleration.

# 5 EXPERIMENTS

In this section, we evaluate the effectiveness of the specific AMX algorithm proposed in Section 4 (i.e.  $c_{t} = 1$ ) on different deep learning tasks. We relegate more details of parameter tuning and step size decay strategies to Appendix D.2-D.5. Moreover, an empirical analysis for different designs of  $\{c_t\}_{t = 1}^T$  that show different  $\{c_t\}_{t = 1}^T$ 's generate different performances is provided in Appendix D.6.

Figure 2: Training and Testing Top-1 accuracy on CIFAR-10 and CIFAR-100.  
![](images/3de03294673a14b7fe003064cd6d26483e6d1a172e0d4de75104a75bc1e70cfa.jpg)  
(a) CIFAR-10 Training Acc.(b) CIFAR-10 Testing Acc(c) CIFAR-100 Training Acc(d) CIFAR-100 Testing Acc.

![](images/67d0388bbb56f72fc3d026247ec09d707526899c95dd9b3ad1c384d29e31a679.jpg)

![](images/98268fd69c88eab1dede4bbc830d1c530e9fe3f8a84a6d61e9b62d34ec4d9879.jpg)

![](images/2f039a9627d1d050f3f8cfc6e2b59cce44524ff32cdf6cf51ebde8a2d3c34a85.jpg)

Table 1: Testing Top-1 accuracy on the CIFAR-10, CIFAR-100 datasets and testing IoU on the VOC2012 Segmentation dataset. The results were averaged over 5 independent runs. Our results were shown in bold.  
Table 2: Validation perplexity on the character Penn Tree Bank (PTB) dataset and BLEU score on the IWSLT'14 DE-EN dataset. The results were averaged over three independent runs. Our results were shown in bold.  

<table><tr><td>OPTIMIZER</td><td>CIFAR-10</td><td>CIFAR-100</td><td>VOC2012</td><td rowspan="2">OptIMIZER</td><td rowspan="2">CHAR-PTB</td><td rowspan="2">IWSLT&#x27;14</td></tr><tr><td>SGDM</td><td>92.40 ± 0.06</td><td>77.80 ± 0.08</td><td>76.10 ± 0.09</td></tr><tr><td>ADAGRAD</td><td>85.60 ± 0.14</td><td>72.84± 0.06</td><td>71.28 ± 0.18</td><td>ADAGRAD</td><td>2.63 ± 0.04</td><td>25.56 ± 0.05</td></tr><tr><td>ADAM</td><td>91.85± 0.03</td><td>74.51± 0.05</td><td>73.32± 0.21</td><td>ADAM</td><td>2.48 ± 0.08</td><td>28.01 ± 0.07</td></tr><tr><td>AMSGrad</td><td>91.97 ± 0.07</td><td>75.75 ± 0.04</td><td>73.66 ± 0.13</td><td>AMSGrad</td><td>2.46 ± 0.05</td><td>28.15 ± 0.06</td></tr><tr><td>AMX</td><td>92.42± 0.08</td><td>77.65± 0.10</td><td>76.04± 0.16</td><td>AMX</td><td>2.32 ± 0.04</td><td>28.29 ± 0.03</td></tr></table>

We compared our AMX algorithm with SGD with momentum (SGDM), Adam, AdaGrad and AMSGrad on different tasks in deep learning. The hyper-parameters in AMX were set to be  $c_t = 1$  in this subsection. For the language modeling and the neural machine translation tasks, because SGDM typically performs much worse than adaptive algorithms, we did not include it in the comparisons. Following Loshchilov & Hutter (2019), we used decoupled weight decay in all the algorithms.

Image Classification. We first conducted some experiments where  $\tau$  was possibly very large and AMX was only as fast as the other adaptive algorithms, but it still achieved better testing performances. The image classification task was performed on the CIFAR (Krizhevsky et al., 2009) datasets. We used the publicly available code by Li et al. (2020) and DeVries & Taylor (2017) to train ResNet-20 and ResNet-18 (He et al., 2016) on CIFAR-10 and CIFAR-100 respectively using batch size of 128.

We summarized the performances of different algorithms in Figure 2 and Table 1. As observed, AMX started slightly slowly, but it quickly caught up with the other adaptive algorithms and converged much faster than SGDM. This was possibly because the time when gradients start to converge (the  $\tau$  in section 4) was large in image classification tasks, and AMX could only converge asymptotically as fast as the other adaptive algorithms, corresponding to Theorem 4.1. However, its final testing performance was as good as SGDM, so it converged both fast and well at the same time. The other adaptive algorithms such as Adam and AMSGrad had faster training performances in the beginning, but they ended up with much worse final accuracy than SGDM and AMX.

Image Segmentation. Next, more experiments proved our claim that AMX could be potentially much faster and generate even better testing performances. For the segmentation task, we used the publicly released implementation of the Deeplab model (Chen et al., 2016) by Kazuto1011 (2016) and evaluated the performances of different algorithms on the PASCAL VOC2012 Segmentation dataset (Everingham et al., 2014). We used a small batch size of 4 and a polynomially decaying step size in  $20\mathrm{k}$  iterations. The trained models were evaluated at the 5k, 10k, 15k and  $20\mathrm{k}$  iterations and we used mean Intersection over Union (IoU) as the evaluation metric. The results were provided in Figure 3(a), 3(b) and Table 1. As shown in the figures and the table, AMX was not only the fastest adaptive algorithm but also achieved the best IoU score, which was comparable to that of SGDM. The other algorithms were not able to perform similarly.

![](images/9c64f27e47f9b8dc794b5dda487f5494040412b6b3da2592ad4daacbf4064abf.jpg)  
(a) VOC Training Loss Curve

![](images/7eb7e728fc83d06c8ef55da1e4dc49c93dd6cb3589a91eeb99b390e78d1d1f39.jpg)  
(b) VOC Testing IoU

![](images/b15381eba2b71e9a2e0e7943883eb5f45ad2fefd85dbc7491939371e3bd70835.jpg)  
Figure 3: (a), (b). Training Loss and Testing IoU curves on the VOC2012 Segmentation dataset. (c). Validation perplexity curve on the Penn Tree Bank (PTB) dataset. (d). Validation Perplexity curve on the IWSLT'14 DE-EN machine translation dataset.  
(d) IWSLT'14 Test Perplexity

![](images/3e659439259d0853747c4ada0b7c261ccdbfc71583764db0ddf384f1abd9f2c1.jpg)  
(c) PTB Test Perplexity

Language Modeling. We trained three-layer LSTMs (Hochreiter & Schmidhuber, 1997) on the character level Penn Tree Bank (PTB) dataset. The general setup in Merity et al. (2017) was adopted in our experiments. Specifically, we trained the model for 500 epochs with batch size 128. The validation perplexity curve and the final validation perplexity were shown in Figure 3(c) and Table 2. It can be observed that AMX was the fastest algorithm and achieved the lowest perplexity among all the adaptive algorithms, which proved our claim that AMX was potentially much faster.

Neural Machine Translation. We utilized the publicly released code by pcyin (2018) and trained the basic attentional neural machine translation models (Luong et al., 2015) on the IWSLT'14 DE-EN (Ranzato et al., 2015) dataset. We used 64 as the batch size and decreased the step size by 2 every 5 iterations. The validation perplexity curve and the final BLEU score were reported in Figure 3(d) and Table 2. AMX not only had a much smoother validation perplexity curve, but also achieved the best BLEU score among all the adaptive algorithms, showing that AMX was indeed a better choice.

# 6 CONCLUSION

In this paper, we propose our design of the best proximal functions at each time step based on marginal regret bound minimization. We then show that a more general class of adaptive algorithms can not only achieve marginal optimality in some sense, but also converge much faster than any existing adaptive algorithms, depending on the distribution of the gradients. We evaluate one particular case of our new class of algorithms on different tasks in deep learning and show its effectiveness. This work provides a new framework for adaptive algorithms and can hopefully prevent the random searching process for better designs of the proximal function. Future researchers can concentrate on finding better choices of the sequence  $\{c_t\}_{t=1}^T$  to find better algorithms.

# REFERENCES

Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, USA, 2004. ISBN 0521833787.  
Nicolò Cesa-Bianchi, Alex Conconi, and Claudio Gentile. On the generalization ability of on-line learning algorithms. IEEE Transacitons on Information Theory, 2004.  
Jinghui Chen and Quanquan Gu. Closing the generalization gap of adaptive gradient methods in training deep neural networks. arXiv preprint arXiv:1806.06763, 2018.  
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40:834-848, 2016.  
Dami Choi, Christopher J. Shallue, Zachary Nado, Jaehoon Lee, Chris J. Maddison, and George E. Dahl. On empirical comparisons of optimizers for deep learning, 2019.  
Terrance DeVries and Graham W. Taylor. Improved regularization of convolutional neural networks with cutout, 2017.  
John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Ambuj Tewari. Composite objective mirror descent. In Proceedings of the Twenty Third Annual Conference on Computational Learning Theory, 2010b.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research (JMLR), pp. 12:2121-2159, 2011.  
Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The Pascal visual object classes challenge: A retrospective. International Journal of Computer Vision(IJCV), 2014.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, pp. 9(8):1735-1780, 1997.  
Haiwen Huang, Chang Wang, and Bin Dong. Nostalgic adam: Weighting more of the past gradients when designing the adaptive learning rate. arXiv preprint arXiv: 1805.07557, 2019.  
Kazuto1011. Deeplab with pytorch. https://github.com/kazuto1011/deeplab-pytorch, 2016.  
Diederik P Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research). 2009.  
Wenjie Li, Zhaoyang Zhang, Xinjiang Wang, and Ping Luo. Adax: Adaptive gradient descent with exponential long term memory. arXiv preprint arXiv:2004.09740, 2020.  
Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312, 2014.  
Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. arXiv preprint arXiv:1908.03265, 2019.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. Proceedings of 7th International Conference on Learning Representations (ICLR), 2019.

Liangchen Luo, Yuanhao Xiong, Yan Liu, and Xu Sun. Adaptive gradient methods with dynamic bound of learning rate. Proceedings of 7th International Conference on Learning Representations, 2019.  
Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine translation, 2015.  
Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and optimizing LSTM language models. arXiv preprint arXiv:1708.02182, 2017.  
pcyin. Basic pytorch implementation of attentional neural machine translation. https://github.com/pcyin/pytorch基礎_nmt, 2018.  
Boris Polyak. Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, pp. 4(5):1-17, 1964.  
Marc'Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks, 2015.  
Sashank J. Reddi, Stayen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. Proceedings of the 6th International Conference on Learning Representations (ICLR), 2018.  
Pedro Savarese, David McAllester, Sudarshan Babu, and Michael Maire. Domain-independent dominance of adaptive methods, 2019.  
Tijmen Tieleman and Geoffrey Hinton. Rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, pp. 4(2):26-31, 2012.  
Sharan Vaswani, Frederik Kunstner, Issam Laradji, Si Yi Meng, Mark Schmidt, and Simon Lacoste-Julien. Adaptive gradient methods converge faster with over-parameterization (and you can do a line-search), 2020.  
Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive methods for nonconvex optimization. Advances in Neural Information Processing Systems 31, 2018.  
Zhiming Zhou, Qingru Zhang, Guansong Lu, Hongwei Wang, Weinan Zhang, and Yong Yu. Adashift: Decorrelation and convergence of adaptive learning rate methods. Proceedings of 7th International Conference on Learning Representations (ICLR), 2019.  
Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. International Conference on Machine Learning (ICML), 2003.
