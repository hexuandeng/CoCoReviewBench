# A FIRST-ORDER MULTI-GRADIENT ALGORITHM FOR MULTI-OBJECTIVE BI-LEVEL OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this paper, we study the Multi-Objective Bi-Level Optimization (MOBLO) problem, where the upper-level subproblem is a multi-objective optimization problem and the lower-level subproblem is for scalar optimization. Existing gradient-based MOBLO algorithms need to compute the Hessian matrix, causing the computational inefficient problem. To address this, we propose an efficient first-order multi-gradient method for MOBLO, called FORUM. Specifically, we reformulate MOBLO problems as a constrained multi-objective optimization (MOO) problem via the value-function approach. Then we propose a novel multi-gradient aggregation method to solve the challenging constrained MOO problem. Theoretically, we provide the complexity analysis to show the efficiency of the proposed method and a non-asymptotic convergence result. Empirically, extensive experiments demonstrate the effectiveness and efficiency of the proposed FORUM method in different learning problems. In particular, it achieves state-of-the-art performance on three multi-task learning benchmark datasets.

# 1 INTRODUCTION

In this work, we study the Multi-Objective Bi-Level Optimization (MOBLO) problem, which is formulated as

$$
\min  _ {\alpha \in \mathbb {R} ^ {n}, \omega \in \mathbb {R} ^ {p}} F (\alpha , \omega) \quad \text {s . t .} \quad \omega \in \mathcal {S} (\alpha) = \underset {\omega} {\arg \min } f (\alpha , \omega), \tag {1}
$$

where  $\alpha$  and  $\omega$  denote the Upper-Level (UL) and Lower-Level (LL) variables, respectively. The UL subproblem,  $F\coloneqq (F_{1},F_{2},\ldots ,F_{m})^{\top}:\mathbb{R}^{n}\times \mathbb{R}^{p}\to \mathbb{R}^{m}$ , is a vector-valued jointly continuous function for  $m$  desired objectives.  $S(\alpha)$  denotes the optimal solution set (which is usually assumed to be a singleton set (Franceschi et al., 2017; Ye et al., 2021)) of the LL subproblem by minimizing a continuous function  $f(\alpha ,\omega)$  w.r.t.  $\omega$ . In this work, we focus on MOBLO with a singleton  $S(\alpha)$  and a non-convex UL subproblem, where  $F_{i}$  is a non-convex function for all  $i$ . MOBLO has demonstrated its superiority in various learning problems such as neural architecture search (Elsken et al., 2018; Lu et al., 2020; Liu & Jin, 2021; Yue et al., 2022), reinforcement learning (Chen et al., 2019; Yang et al., 2019; Abdolmaleki et al., 2020), multi-task learning (Ye et al., 2021; Mao et al., 2022), and meta-learning (Ye et al., 2021; Yu et al., 2023).

Recently, MOML (Ye et al., 2021) and MoCo (Fernando et al., 2023) are proposed as effective gradient-based MOBLO algorithms, which hierarchically optimize the UL and LL variables based on IIterative Differentiation (ITD) based Bi-Level Optimization (BLO) approach (Maclaurin et al., 2015; Franceschi et al., 2017; 2018; Grazzi et al., 2020). Specifically, given  $\alpha$ , both MOML and MoCo first compute the LL solution  $\omega^{*}(\alpha)$  by solving LL subproblem with  $T$  iterations and then update  $\alpha$  via the combination of the hypergradients  $\{\nabla_{\alpha}F_{i}(\alpha,\omega^{*}(\alpha))\}_{i=1}^{m}$ . Note that they need to calculate the complex gradient  $\nabla_{\alpha}\omega^{*}(\alpha)$ , which requires to compute many Hessian-vector products via the chain rule. Besides, their time and memory costs grow significantly fast with respect to the dimension of  $\omega$  and  $T$ . Therefore, existing gradient-based methods to solve MOBLO problems could suffer from the inefficiency problem, especially in deep neural networks.

To address this limitation, we propose an efficient First-Order mUlti-gradient method for MOBLO (FORUM). Specifically, we reformulate MOBLO as an equivalent constrained multi-objective optimization (MOO) problem by the value-function-based approach (Liu et al., 2021c; 2022a; Sow et al., 2022). Then, we propose a multi-gradient aggregation method to solve the challenging

constrained MOO problem. Different from MOML and MoCo, FORUM is a fully first-order algorithm and does not need to calculate the high-order Hessian matrix. Theoretically, we provide the complexity analysis showing that FORUM is more efficient than MOML and MoCo in both time and memory costs. In addition, we provide a non-asymptotic convergence analysis for FORUM. Empirically, we evaluate the effectiveness and efficiency of FORUM on two learning problems, i.e., multi-objective data hyper-cleaning and multi-task learning on three benchmark datasets.

The main contributions of this work are three-fold:

- We propose the FORUM method, an efficient gradient-based algorithm for the MOBLO problem;  
- We demonstrate FORUM is more efficient than existing MOBLO methods from the perspective of complexity analysis and provide a non-asymptotic convergence analysis;  
- Extensive experiments demonstrate the effectiveness and efficiency of the proposed FORUM method. In particular, it achieves state-of-the-art performance on three benchmark datasets under the setting of multi-task learning.

# 2 RELATED WORKS

Multi-Objective Optimization. MOO aims to solve multiple objectives simultaneously and its goal is to find the Pareto-optimal solution. MOO algorithms can be divided into three categories: population-based (Angus, 2007), evolutionary-based (Zhou et al., 2011), and gradient-based (Désidéri, 2012; Mahapatra & Rajan, 2020). In this paper, we focus on the last category. MGDA algorithm (Désidéri, 2012) is a representative gradient-based MOO method, which finds a gradient update direction to make all the objectives decrease in every training iteration by solving a quadratic programming problem. Compared with the widely-used linear scalarization approach which linearly combines multiple objectives to a single objective, MGDA and its variants (Fernando et al., 2023; Zhou et al., 2022) have shown their superiority in many learning problems such as multi-task learning (Sener & Koltun, 2018) and reinforcement learning (Yu et al., 2020), especially when some objectives are conflicting.

Bi-Level Optimization. BLO (Liu et al., 2021b) is a type of optimization problem with a hierarchical structure, where one subproblem is nested within another subproblem. The MOBLO problem (1) reduces degrades to BLO problem when  $m$  equals 1. One representative category of the BLO method is the ITD-based methods (Maclaurin et al., 2015; Franceschi et al., 2017; 2018; Grazzi et al., 2020) that use approximated hypergradient to optimize the UL variable, which is computed by the automatic differentiation based on the optimization trajectory of the LL variable. Some value-function-based algorithms (Liu et al., 2022a; 2021c; Sow et al., 2022) have been proposed recently to solve BLO by reformulating the original BLO to an equivalent optimization problem with a simpler structure. The value-function-based reformulation strategy naturally yields a first-order algorithm, hence it has high computational efficiency.

Multi-Objective Bi-Level Optimization. MOML (Ye et al., 2021) is proposed as the first gradient-based MOBLO algorithm. However, MOML needs to calculate the complex Hessian matrix to obtain the hypergradient, causing the computationally inefficient problem. MoCo (Fernando et al., 2023) also employs the ITD-based approach like MOML for hypergradient calculation. It uses a momentum-like gradient approximation approach for hypergradient and a one-step approximation method to update the weights. It has the same inefficiency problem as the MOML method. Yu et al. (2023) propose a mini-batch approach to optimize the UL subproblem in the MOBLO. However, it aims to generate weights for a huge number of UL objectives and is different from what we focus on. MORBiT (Gu et al., 2023) studies a BLO problem with multiple objectives in its UL subproblem but it formulates the UL subproblem as a min-max problem, which is different from problem (1) we focus on in this paper.

# 3 THE FORUM ALGORITHM

In this section, we introduce the proposed FORUM method. Firstly, we reformulate MOBLO as an equivalent constrained multi-objective problem via the value-function-based approach in Section 3.1.

Next, we provide a novel multi-gradient aggregation method to solve the constrained multi-objective problem in Section 3.2.

# 3.1 REFORMULATION OF MOBLO

Based on the value-function-based approach (Liu et al., 2021c; 2022a; Sow et al., 2022; Kwon et al., 2023), we reformulate MOBLO problem (1) as an equivalent single-level constrained multi-objective optimization problem:

$$
\min  _ {\alpha \in \mathbb {R} ^ {n}, \omega \in \mathbb {R} ^ {p}} F (\alpha , \omega) \quad \text {s . t .} f (\alpha , \omega) \leq f ^ {*} (\alpha), \tag {2}
$$

where  $f^{*}(\alpha) = \min_{\omega} f(\alpha, \omega) = f(\alpha, \omega^{*}(\alpha))$  is the value function, which represents the lower bound of  $f(\alpha, \omega)$  w.r.t.  $\omega$ . To simplify the notation, we define  $z \equiv (\alpha, \omega) \in \mathbb{R}^{n+p}$  and  $\mathcal{Z} \equiv \mathbb{R}^n \times \mathbb{R}^p$ . Then, we have  $F(z) \equiv F(\alpha, \omega)$  and  $f(z) \equiv f(\alpha, \omega)$ . Thus, problem (2) can be rewritten as

$$
\min  _ {z \in \mathcal {Z}} F (z) \quad \text {s . t .} q (z) \leq 0, \tag {3}
$$

where  $q(z) = f(z) - f^{*}(\alpha)$  is the constraint function. Since the gradient of the value function  $f^{*}(\alpha)$  is  $\nabla_{\alpha}f^{*}(\alpha) = \nabla_{\alpha}f(\alpha ,\omega^{*}(\alpha)) = \nabla_{\alpha}f(\alpha ,\omega^{*})$  by the chain rule and  $\nabla_{\omega}f(\alpha ,\omega)|_{\omega = \omega^{*}(\alpha)} = 0$ , we do not need to compute the complex Hessian matrix  $\nabla_{\alpha}\omega^{*}(\alpha)$  like MOML and MoCo.

However, solving problem (3) is challenging for two reasons. One reason is that the Slater's condition (Chen et al., 2023), which is required for duality-based optimization methods, does not hold for problem (3), since the constraint  $q(z) \leq 0$  is ill-posed (Liu et al., 2021c; Jiang et al., 2023) and does not have an interior point. To see this, we assume  $z_0 = (\alpha_0, \omega_0) \in \mathcal{Z}$  and  $q(z_0) \leq 0$ . Then the constraint  $q(z) \leq 0$  is hard to be satisfied at the neighborhood of  $\alpha_0$ , unless  $f^*(\alpha)$  is a constant function around  $\alpha_0$ , which rarely happens. Therefore, problem (3) cannot be treated as classic constrained optimization and we propose a novel gradient method to solve it in Section 3.2. Another reason is that for given  $\alpha$ , the computation of  $\omega^*(\alpha)$  is intractable. Thus, we approximate it by  $\tilde{\omega}^T$  computed by  $T$  steps of gradient descent. Specifically, given  $\alpha$  and an initialization  $\tilde{\omega}^0$  of  $\omega$ , we have

$$
\tilde {\omega} ^ {t + 1} = \tilde {\omega} ^ {t} - \eta \nabla_ {\omega} f (\alpha , \tilde {\omega} ^ {t}), \quad t = 0, \dots , T - 1, \tag {4}
$$

where  $\eta$  is the step size. Then, the constraint function  $q(z)$  is approximated by  $\widetilde{q}(z) = f(z) - f(\alpha, \tilde{\omega}^T)$  and its gradient  $\nabla_z q(z)$  is approximated by  $\nabla_z \widetilde{q}(z)$ . We show that the approximation error of gradient exponentially decays w.r.t. the LL iterations  $T$  in Appendix A.1. Hence, problem (3) is modified to

$$
\min  _ {z \in \mathcal {Z}} F (z) \quad \text {s . t .} \quad \widetilde {q} (z) = f (z) - f (\alpha , \tilde {\omega} ^ {T}) \leq 0. \tag {5}
$$

# 3.2 MULTI-GRADIENT AGGREGATION METHOD

In this section, we introduce the proposed multi-gradient aggregation method for solving problem (5) iteratively. Specifically, at  $k$ -th iteration, assume  $z_{k}$  is updated by  $z_{k + 1} = z_{k} + \mu d_{k}$  where  $\mu$  is the step size and  $d_{k}$  is the update direction for  $z_{k}$ . Then, we expect  $d_{k}$  can simultaneously minimize the UL objective  $F(z)$  and the constraint function  $\widetilde{q}(z)$ . Note that the minimum of the approximated constraint function  $\widetilde{q}(z)$  converges to the minimum of  $q(z)$ , i.e., as  $T \to +\infty$ . Thus, we expect  $d_{k}$  to decrease  $\widetilde{q}(z)$  consistently such that the constraint  $\widetilde{q}(z) \leq 0$  is satisfied.

Note that there are multiple potentially conflicting objectives  $\{F_i\}_{i=1}^m$  in the UL subproblem. Hence, we expect  $d_k$  can decrease every objective  $F_i$ , which can be formulated as the following problem to find  $d_k$  to maximize the minimum decrease across all objectives as

$$
\max  _ {d} \min  _ {i \in [ m ]} \left(F _ {i} \left(z _ {k}\right) - F _ {i} \left(z _ {k} + \mu d\right)\right) \approx - \mu \min  _ {d} \max  _ {i \in [ m ]} \langle \nabla F _ {i} \left(z _ {k}\right), d \rangle . \tag {6}
$$

To regularize the update direction, we add a regularization term  $\frac{1}{2}\|d\|^2$  to problem (6) and compute  $d_k$  by solving  $\min_d \max_{i \in [m]} \langle \nabla F_i(z_k), d \rangle + \frac{1}{2}\|d\|^2$ .

To decrease the constraint function  $\widetilde{q}(z)$ , we expect the inner product of  $-d$  and  $\nabla \widetilde{q}(z_k)$  to hold positive during the optimization process, i.e.,  $\langle \nabla \widetilde{q}(z_k), -d \rangle \geq \phi$ , where  $\phi$  is non-negative constant. To further guarantee that  $\widetilde{q}(z)$  can be optimized such that the constraint  $\widetilde{q}(z) \leq 0$  can be satisfied, we introduce a dynamic  $\phi_k$  here. Specifically, inspired by Gong et al. (2021), we set  $\phi_k = \frac{\rho}{2} \| \nabla \widetilde{q}(z_k) \|^2$ , where  $\rho$  is a positive constant. When  $\phi_k > 0$ , it means that  $\| \nabla \widetilde{q}(z) \| \neq 0$  and  $\widetilde{q}(z)$  should be further

Algorithm 1 The FORUM Method  
Require: number of iterations  $(K,T)$  , step size  $(\mu ,\eta)$  , coefficient  $\beta_{k}$  , constant  $\rho$  1: Randomly initialize  $z_0 = (\alpha_0,\omega_0)$  .   
2: Initialize  $\tilde{\lambda}_i^{-1} = 0,i = 1,\dots,m;$    
3: for  $k = 0$  to  $K - 1$  do   
4: Set  $\tilde{\omega}^0 = \omega_0$  or  $\tilde{\omega}^0 = \omega_k$  .   
5: for  $t = 0$  to  $T - 1$  do   
6: Update  $\tilde{\omega}$  as  $\tilde{\omega}^{t + 1} = \tilde{\omega}^t -\eta \nabla_\omega f(\alpha_k,\tilde{\omega}^t)$  .   
7: end for   
8: Set  $\widetilde{q} (z_k) = f(z_k) - f(\alpha_k,\tilde{\omega}^T)$  .   
9: Compute gradient  $\nabla_z\widetilde{q} (z_k) = \nabla_zf(z_k) - \nabla_\alpha f(\alpha_k,\tilde{\omega}^T)$  .   
10: Compute gradients  $\nabla_{z}F_{i}(z_{k})$ $i = 1,\ldots ,m$  .   
11: Compute  $\lambda^k$  by solving problem (11);   
12: Compute the momentum update  $\tilde{\lambda}^k = (1 - \beta_k)\tilde{\lambda}^{k - 1} + \beta_k\lambda^k$  .   
13: Compute  $\nu (\tilde{\lambda}^k)$  via Eq. (9);   
14: Compute  $d_{k}$  via Eq. (8);   
15: Update  $z$  as  $z_{k + 1} = z_{k} + \mu d_{k}$  .   
16: end for   
17: return  $z_K$

optimized, and  $\langle \nabla \widetilde{q}(z_k), -d \rangle \geq \phi_k > 0$  can enforce  $\widetilde{q}(z)$  to decrease. When  $\phi_k$  equals 0, it indicates that the optimum of  $\widetilde{q}(z)$  is reached and  $\langle \nabla \widetilde{q}(z_k), -d \rangle \geq \phi_k = 0$  also holds. Thus, the dynamic  $\phi_k$  can ensure  $d_k$  to iteratively decrease  $\widetilde{q}(z)$  such that the constraint  $\widetilde{q}(z) \leq 0$  is satisfied.

Therefore, at  $k$ -th iteration, we can find  $d_{k}$  by solving the following problem,

$$
\min  _ {d} \max  _ {i \in [ m ]} \left\langle \nabla F _ {i} \left(z _ {k}\right), d \right\rangle + \frac {1}{2} \| d \| ^ {2}, \text {s . t .} \left\langle \nabla \widetilde {q} \left(z _ {k}\right), d \right\rangle \leq - \phi_ {k}. \tag {7}
$$

Based on the Lagrangian multiplier method, problem (7) has a solution as

$$
d _ {k} = - \left(\sum_ {i = 1} ^ {m} \lambda_ {i} ^ {k} \nabla F _ {i} \left(z _ {k}\right) + \nu \left(\lambda^ {k}\right) \nabla \widetilde {q} \left(z _ {k}\right)\right), \tag {8}
$$

where Lagrangian multipliers  $\lambda^k = (\lambda_1^k, \dots, \lambda_m^k) \in \Delta^{m-1}$  (i.e.,  $\sum_{i=1}^m \lambda_i^k = 1$  and  $\lambda_i^k \geq 0$ ) and  $\nu(\lambda)$  is a function of  $\lambda$  as

$$
\nu (\lambda) = \max  \left(\sum_ {i = 1} ^ {m} \lambda_ {i} \pi_ {i} \left(z _ {k}\right), 0\right) \text {w i t h} \pi_ {i} \left(z _ {k}\right) = \frac {2 \phi_ {k} - \langle \nabla \widetilde {q} \left(z _ {k}\right) , \nabla F _ {i} \left(z _ {k}\right) \rangle}{\| \nabla \widetilde {q} \left(z _ {k}\right) \| ^ {2}}. \tag {9}
$$

Here  $\lambda_i^k$  can be obtained by solving the following dual problem as

$$
\lambda^ {k} = \underset {\lambda \in \Delta^ {m - 1}} {\arg \min } \frac {1}{2} \left\| \sum_ {i = 1} ^ {m} \lambda_ {i} \nabla F _ {i} \left(z _ {k}\right) + \nu (\lambda) \nabla \tilde {q} \left(z _ {k}\right) \right\| ^ {2} - \nu (\lambda) \phi_ {k}. \tag {10}
$$

The detailed derivations of the above procedure are put in Appendix A.2. Problem (10) can be reformulated as

$$
\min  _ {\lambda \in \Delta^ {m - 1}, \gamma} \frac {1}{2} \left\| \sum_ {i = 1} ^ {m} \lambda_ {i} \nabla F _ {i} \left(z _ {k}\right) + \gamma \nabla \widetilde {q} \left(z _ {k}\right) \right\| ^ {2} - \gamma \phi_ {k} \quad \text {s . t .} \gamma \geq 0, \gamma \geq \sum_ {i = 1} ^ {m} \lambda_ {i} \pi_ {i} \left(z _ {k}\right). \tag {11}
$$

The first term of the objective function in problem (11) can be simplified to  $R^{\top}\Lambda^{\top}\Lambda R$ , where  $R = (\lambda_1,\dots ,\lambda_m,\gamma)^\top$  and  $\Lambda = (\nabla F_{1},\ldots ,\nabla F_{m},\nabla \widetilde{q})$ . Note that the dimension of the matrix  $\Lambda^{\top}\Lambda$  is  $(m + 1)\times (m + 1)$ , which is independent with the dimension of  $z$ . As the number of UL objectives  $m$  is usually small, solving problem (11) does not incur too much computational cost. In practice, we can use the open-source CVXPY library (Diamond & Boyd, 2016) to solve problem (11).

To ensure convergence, the sequence of  $\{\lambda^k\}_{k=1}^K$  should be a convergent sequence (refer to the discussion in Appendix A.3). However,  $\{\lambda^k\}_{k=1}^K$  obtained by directly solving the problem (11) in each

iteration cannot ensure such properties. Therefore, we apply a momentum strategy (Zhou et al., 2022) to  $\lambda$  to generate a stable sequence and further guarantee the convergence. Specifically, in  $k$ -th iteration, we first solve the problem (11) to obtain  $\lambda^k$ , then update the weights by  $\tilde{\lambda}^k = (1 - \beta_k)\tilde{\lambda}^{k-1} + \beta_k\lambda^k$ , where  $\beta_k \in (0,1]$  is set to 1 at the beginning and asymptotically convergent to 0 as  $k \to +\infty$ .

After obtaining  $\tilde{\lambda}^k$  with the momentum update, we can compute the corresponding  $\nu(\tilde{\lambda}^k)$  via Eq. (9). Then we obtain the update direction  $d_k$  by Eq. (8) and update  $z_k$  as  $z_{k+1} = z_k + \mu d_k$ . The entire FORUM algorithm is shown in Algorithm 1.

# 4 ANALYSIS

In this section, we provide complexity analysis and convergence analysis for the FORUM method.

# 4.1 COMPLEXITY ANALYSIS

For the proposed FORUM method, it takes time  $\mathcal{O}(pT)$  and space  $\mathcal{O}(p)$  to obtain  $\widetilde{q}(z)$ , and then the computations of all the gradients including  $\nabla_z F_i(z)$  and  $\nabla_z \widetilde{q}(z)$  require time  $\mathcal{O}((n+p)(m+1))$  and space  $\mathcal{O}((n+p)(m+1))$ . When  $m \ll \min\{n,p\}$ , the time and space costs of solving the quadratic programming problem (11), which only depends on  $m$ , can be negligible. Therefore, FORUM runs in time  $\mathcal{O}(mn+p(m+T))$  and space  $\mathcal{O}(mn+mp)$  in total for each UL iteration.

For existing MOBLO methods (i.e., MOML and MoCo), after  $T$ -iteration update for the LL subproblem in time  $\mathcal{O}(pT)$  and space  $\mathcal{O}(p)$ , calculating the Hessian-matrix product via backward propagation can be evaluated in time  $\mathcal{O}(p(n + p)T)$  and space  $\mathcal{O}(n + pT)$ . Similar to FORUM, the cost of solving the quadratic programming problem in MOML is also negligible. Note that MoCo applies a momentum update to the UL variables, which causes an additional  $\mathcal{O}(mn)$  space cost. Thus, for each UL iteration, MOML and MoCo require  $\mathcal{O}(mp(n + p)T)$  time in total, and they require  $\mathcal{O}(mn + mpT)$  and  $\mathcal{O}(2mn + mpT)$  space, respectively.

In summary, the above analysis indicates FORUM is more efficient than MOML and MoCo in terms of time and space complexity.

# 4.2 CONVERGENCE ANALYSIS

In this section, we analyze the convergence property of FORUM. Firstly, we make an assumption for the UL subproblem.

Assumption 4.1. For  $i = 1, \dots, m$ , it is assumed that  $\nabla F_i(\alpha, \omega)$  is  $L_F$ -Lipschitz continuous with respect to  $z := (\alpha, \omega)$ . The  $\ell_2$  norm of  $\nabla F_i(z)$  and  $|F_i(z)|$  are upper-bounded by a constant  $M$ .

The smoothness and the boundedness assumptions in Assumption 4.1 are widely adopted in nonconvex multi-objective optimization (Zhou et al., 2022; Fernando et al., 2023). Then we make an assumption for the LL subproblem.

Assumption 4.2. The function  $f(\alpha, \omega)$  is  $c$ -strongly convex with respect to  $\omega$ , and  $\nabla f(\alpha, \omega)$  is  $L_{f}$ -Lipschitz continuous with respect to  $z \coloneqq (\alpha, \omega)$ .

The strongly convexity assumption in Assumption 4.2 is commonly used in the analysis for the BLO (Maclaurin et al., 2015; Franceschi et al., 2017; 2018) and MOBLO problems (Fernando et al., 2023; Ye et al., 2021). The proposed FORUM algorithm focuses on generating one Karush-Kuhn-Tucker (KKT) stationary point of the original constrained multi-objective optimization problem (3). Following (Gong et al., 2021; Liu et al., 2022a), we measure the convergence of problem (3) by both its KKT stationary condition and the feasibility condition, where detailed definitions are provided in Appendix B.1. Specifically, we denote by  $\mathcal{K}(z_k) = \left\| \sum_{i=1}^m \tilde{\lambda}_i^k \nabla F_i(z_k) + \nu_k \nabla q(z_k) \right\|^2$  the measure of KKT stationary condition in the  $k$ -th iteration, where  $\nu_k = \nu(\tilde{\lambda}^k)$ . To satisfy the feasibility condition of problem (3), the non-negative function  $q(z_k)$  should decrease to 0. Then, with a non-convex multi-objective UL subproblem, we have the following convergence result.

Theorem 4.3. Suppose that Assumptions 4.1 and 4.2 hold, and the sequence  $\{z_k\}_{k=0}^K$  generated by Algorithm 1 satisfies  $q(z_k) \leq B$ , where  $B$  is a positive constant. Then if  $\eta \leq 1/L_f$ ,  $\mu = \mathcal{O}(K^{-1/2})$ ,

and  $\beta = \mathcal{O}(K^{-3 / 4})$ , there exists a constant  $C > 0$  such that when  $T\geq C$ , for any  $K > 0$ , we have

$$
\max  \left\{\min  _ {k <   K} \mathcal {K} \left(z _ {k}\right), q \left(z _ {k}\right) \right\} = \mathcal {O} \left(K ^ {- 1 / 4} + \Gamma (T)\right), \tag {12}
$$

where  $\Gamma (T)$  represents exponential decays with respect to  $T$

The proof is put in Appendix B.3. Theorem 4.3 gives a non-asymptotic convergence result for Algorithm 1, which depends on both numbers of steps in the UL and LL subproblems (i.e.,  $K$  and  $T$ ).

# 5 EXPERIMENTS

In this section, we empirically evaluate the proposed FORUM method on different learning problems. All experiments are conducted on a single NVIDIA GeForce RTX 3090 GPU.

# 5.1 DATA HYPER-CLEANING

Setup. Data hyper-cleaning (Bao et al., 2021; Franceschi et al., 2017; Liu et al., 2022a;b; Shaban et al., 2019) is a specific hyperparameter optimization problem, where a model is trained on a dataset with part of corrupted training labels. Thus, it aims to reduce the influence of noisy examples by learning to weigh the train samples in a bi-level optimization manner. Here we extend it to a multi-objective setting, where we aim to train a model on multiple corrupted datasets.

Specifically, suppose that there are  $m$  corrupted datasets.  $\mathcal{D}_i^{\mathrm{tr}} = \{x_{i,j},y_{i,j}\}_{j = 1}^{N_i}$  and  $\mathcal{D}_i^{\mathrm{val}}$  denote the noisy training set and the clean validation set for the  $i$ -th dataset, respectively, where  $x_{i,j}$  denotes the  $j$ -th training sample in the  $i$ -th dataset,  $y_{i,j}$  is the corresponding label, and  $N_{i}$  denotes the size of the  $i$ -th training dataset. Let  $\omega$  denote the model parameters and  $\alpha_{i,j}$  denotes the weight of  $x_{i,j}$ . Let  $\mathcal{L}_i^{\mathrm{val}}(\omega; \mathcal{D}_i^{\mathrm{val}})$  be the average loss of model  $\omega$  on the clean validation set of the  $i$ -th dataset and  $\mathcal{L}_i^{\mathrm{tr}}(\alpha, \omega; \mathcal{D}_i^{\mathrm{tr}}) = \frac{1}{N_i} \sum_{j=1}^{N_i} \sigma(\alpha_{i,j}) \ell(\omega; x_{i,j}, y_{i,j})$  be the weighted average loss on the noisy training set of the  $i$ -th dataset, where  $\sigma(\cdot)$  is an element-wise sigmoid function to constrain each weight in the range [0, 1] and  $\ell(\omega; x,y)$  denotes the loss of model  $\omega$  on sample  $(x,y)$ . Therefore, the objective function of this multi-objective data hyper-cleaning is formulated as

$$
\min _ {\alpha , \omega} \left(\mathcal {L} _ {1} ^ {\mathrm {v a l}} (\omega ; \mathcal {D} _ {1} ^ {\mathrm {v a l}}), \dots , \mathcal {L} _ {m} ^ {\mathrm {v a l}} (\omega ; \mathcal {D} _ {m} ^ {\mathrm {v a l}})\right) ^ {\top} \text {s . t .} \omega \in \mathcal {S} (\alpha) = \arg \min _ {\omega} \sum_ {i = 1} ^ {m} \mathcal {L} _ {i} ^ {\mathrm {t r}} (\alpha , \omega ; \mathcal {D} _ {i} ^ {\mathrm {t r}}).
$$

Datasets. We conduct experiments on the MNIST (LeCun et al., 1998) and FashionMNIST (Xiao et al., 2017) datasets. Each dataset corresponds to a 10-class image classification problem. All the images have the same size of  $28 \times 28$ . Following Bao et al. (2021), we randomly sample 5000, 1000, 1000, and 5000 images from each dataset as the training set, validation set 1, validation set 2, and test set, respectively. The training set and validation set 1 are used to formulate the LL and UL subproblems, respectively. The validation set 2 is used to select the best model and the testing evaluation is conducted on the test set. Half of the samples in the training set are contaminated by assigning them to another random class. Due to page limit, implementation details are put in Appendix D.1.

Results. Table 1 shows the results on both two datasets under different numbers of LL iterations (i.e.,  $T = 16, 32, 64, 128$ ). The classification accuracy and F1 score computed on the test set are used as the evaluation metrics. As can be seen, the proposed FORUM method outperforms the MOML and MoCo in all the settings, which demonstrates the effectiveness of the proposed FORUM method.

Figures 1(a) and 1(b) show that MOML and MoCo need longer running time than FORUM in every configuration of the UL iteration  $T$  and the number of LL parameters  $p$ , respectively, which implies FORUM has a lower time complexity. Figures 1(c) and 1(d) show the change of memory cost per iteration with respect to the LL iteration  $T$  and the number of LL parameters  $p$ , respectively. As can be seen, the memory cost remains almost constant with different  $T$ 's for FORUM and increases faster for MOML and MoCo. Moreover, the memory cost slightly increases in FORUM with increasing  $p$ , while it linearly increases in MOML and MoCo. In summary, the results in Figure 1 match the complexity analysis in Section 4.1 and demonstrate that FORUM is more efficient than MOML and MoCo.

Table 1: Performance of different methods with different numbers of LL iterations  $T$  on the MNIST and FashionMNIST datasets for the multi-objective data hyper-cleaning problem. Each experiment is repeated over 3 random seeds, and the mean as well as the standard deviation is reported. The best result for each  $T$  is marked in bold.  

<table><tr><td rowspan="2">T</td><td rowspan="2">Methods</td><td colspan="2">MNIST</td><td colspan="2">FashionMNIST</td></tr><tr><td>Accuracy (%)</td><td>F1 Score</td><td>Accuracy (%)</td><td>F1 Score</td></tr><tr><td rowspan="3">16</td><td>MOML</td><td>88.81±0.17</td><td>88.78±0.16</td><td>79.98±0.21</td><td>79.59±0.40</td></tr><tr><td>MoCo</td><td>88.25±0.31</td><td>88.22±0.30</td><td>80.09±0.25</td><td>79.65±0.59</td></tr><tr><td>FORUM (ours)</td><td>90.79±0.33</td><td>90.79±0.33</td><td>82.37±1.00</td><td>82.10±1.16</td></tr><tr><td rowspan="3">32</td><td>MOML</td><td>87.29±0.72</td><td>87.26±0.71</td><td>80.63±0.58</td><td>80.50±0.28</td></tr><tr><td>MoCo</td><td>87.59±0.42</td><td>87.56±0.42</td><td>80.42±0.47</td><td>80.41±0.14</td></tr><tr><td>FORUM (ours)</td><td>90.65±0.44</td><td>90.63±0.47</td><td>82.11±0.72</td><td>81.79±1.01</td></tr><tr><td rowspan="3">64</td><td>MOML</td><td>88.64±0.94</td><td>88.61±0.98</td><td>80.64±0.35</td><td>80.60±0.49</td></tr><tr><td>MoCo</td><td>88.05±1.21</td><td>88.03±1.27</td><td>80.94±0.19</td><td>80.67±0.25</td></tr><tr><td>FORUM (ours)</td><td>90.81±0.14</td><td>90.81±0.15</td><td>82.07±0.38</td><td>81.72±0.57</td></tr><tr><td rowspan="3">128</td><td>MOML</td><td>88.88±0.33</td><td>88.86±0.36</td><td>80.31±0.45</td><td>80.10±0.33</td></tr><tr><td>MoCo</td><td>88.21±0.33</td><td>88.20±0.36</td><td>80.31±0.30</td><td>79.81±0.50</td></tr><tr><td>FORUM (ours)</td><td>90.13±0.37</td><td>90.11±0.36</td><td>82.07±0.73</td><td>81.79±0.97</td></tr></table>

![](images/92aa2aee1a9ce6bdb5df67df27ac4ca19da9080a231d53fcd510fb106b7b5c98.jpg)  
(a) Running time vs.  $T$

![](images/94e08c1a9403c75ed13473856b17cd94a09c638c23ece759ff367a3b9802fe02.jpg)  
(b) Running time vs.  $p$ .

![](images/440e78006a59059d8bf99dc82cea18c87fbf3229a00ab7121fd9bd4af86766cb.jpg)  
Figure 1: Results of different methods on the multi-objective data hyper-cleaning problem. (a): The running time per iteration varies over different LL update steps  $T$ . (b): The running time per iteration varies over the different numbers of LL parameters  $p$  with  $T = 64$ . (c): The memory cost varies over different LL update steps  $T$ . (d): The memory cost varies over the different numbers of LL parameters  $p$  with  $T = 64$ .  
(c) Memory cost vs.  $T$

![](images/8e83def430cea7b4baba1f8f1c9772d8ff3f7b86cd3415743bf0f20e62b95a64.jpg)  
(d) Memory cost vs.  $p$ .

# 5.2 MULTI-TASK LEARNING

Setup. Multi-Task Learning (MTL) (Caruana, 1997; Zhang & Yang, 2022) aims to train a single model to solve several tasks simultaneously. Following Ye et al. (2021), we aim to learn the loss weights to balance different tasks and improve the generalization performance by casting MTL as a MOBLO problem. Specifically, suppose there are  $m$  tasks and the  $i$ -th task has its corresponding dataset  $\mathcal{D}_i$  that contains a training set  $\mathcal{D}_i^{\mathrm{tr}}$  and a validation set  $\mathcal{D}_i^{\mathrm{val}}$ . The MTL model is parameterized by  $\omega$  and  $\alpha \in \Delta^{m - 1}$  denotes the loss weights for the  $m$  tasks. Let  $\mathcal{L}(\omega ;\mathcal{D})$  represent the average loss of model  $\omega$  on the dataset  $\mathcal{D}$ . The MOBLO formulation for MTL is as

$$
\min  _ {\alpha , \omega} \left(\mathcal {L} (\omega ; \mathcal {D} _ {1} ^ {\mathrm {v a l}}), \dots , \mathcal {L} (\omega ; \mathcal {D} _ {m} ^ {\mathrm {v a l}})\right) ^ {\top} \text {s . t .} \omega \in \mathcal {S} (\alpha) = \arg \min  _ {\omega} \sum_ {i = 1} ^ {m} \alpha_ {i} \mathcal {L} (\omega ; \mathcal {D} _ {i} ^ {\mathrm {t r}}).
$$

We conduct experiments on three benchmark datasets among three different task categories, i.e., the Office-31 (Saenko et al., 2010) dataset for image classification, the NYUv2 (Silberman et al., 2012) dataset for scene understanding, and the QM9 dataset for molecular property prediction.

Datasets. (i) The Office-31 dataset (Saenko et al., 2010) includes images from three different sources: Amazon (A), digital SLR cameras (D), and Webcam (W). It contains 31 categories for

each source and a total of 4652 labeled images. We use the data split in Lin et al. (2022):  $60\%$  for training,  $20\%$  for validation, and  $20\%$  for testing. (ii) The NYUv2 dataset (Silberman et al., 2012), an indoor scene understanding dataset, has 795 and 654 images for training and testing, respectively. It has three tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. (iii) The QM9 dataset (Ramakrishnan et al., 2014), a molecular property prediction dataset. We use the commonly-used split as in Fey & Lenssen (2019); Navon et al. (2022): 110,000 for training, 10,000 for validation, and 10,000 for testing. The QM9 dataset contains 11 tasks and each task is a regression task for one property. Due to page limit, implementation details are put in Appendix D.2.

Baselines. The proposed FORUM method is compared with: (i) single-task learning (STL) that trains each task independently; (ii) a comprehensive set of state-of-the-art MTL methods, including Equal Weighting (EW) (Zhang & Yang, 2022), UW (Kendall et al., 2018), PCGrad (Yu et al., 2020), GradDrop (Chen et al., 2020), GradVac (Wang et al., 2021), CAGrad (Liu et al., 2021a), Nash-MTL (Navon et al., 2022), and RLW (Lin et al., 2022); (iii) two first-order BLO methods: BVFIM (Liu et al., 2021c) and BOME (Liu et al., 2022a), where we simply transform MOBLO to BLO by aggregating multiple objectives in the UL subproblem with equal weights into a single objective so that we can apply those BLO methods to solve the MOBLO problem; (iv) two MOBLO method: MOML (Ye et al., 2021) and MoCo (Fernando et al., 2023).

Table 2: Classification accuracy (%) on the Office-31 dataset. Each experiment is repeated over 3 random seeds and the average is reported. The best results over baselines except STL are marked in bold.  

<table><tr><td>Methods</td><td>A</td><td>D</td><td>W</td><td>Avg</td><td>Δp↑</td></tr><tr><td>STL</td><td>86.61</td><td>95.63</td><td>96.85</td><td>93.03</td><td>0.00</td></tr><tr><td colspan="6">multi-task learning methods</td></tr><tr><td>EW</td><td>83.53</td><td>97.27</td><td>96.85</td><td>92.55</td><td>-0.61</td></tr><tr><td>UW</td><td>83.82</td><td>97.27</td><td>96.67</td><td>92.58</td><td>-0.56</td></tr><tr><td>PCGrad</td><td>83.59</td><td>96.99</td><td>96.85</td><td>92.48</td><td>-0.68</td></tr><tr><td>GradDrop</td><td>84.33</td><td>96.99</td><td>96.30</td><td>92.54</td><td>-0.59</td></tr><tr><td>GradVac</td><td>83.76</td><td>97.27</td><td>96.67</td><td>92.57</td><td>-0.58</td></tr><tr><td>CAGrad</td><td>83.65</td><td>95.63</td><td>96.85</td><td>92.04</td><td>-1.13</td></tr><tr><td>Nash-MTL</td><td>85.01</td><td>97.54</td><td>97.41</td><td>93.32</td><td>+0.24</td></tr><tr><td>RLW</td><td>83.82</td><td>96.99</td><td>96.85</td><td>92.55</td><td>-0.59</td></tr><tr><td colspan="6">first-order bi-level optimization methods</td></tr><tr><td>BVFIM</td><td>84.84</td><td>96.99</td><td>97.78</td><td>93.21</td><td>+0.11</td></tr><tr><td>BOME</td><td>85.53</td><td>96.72</td><td>98.15</td><td>93.47</td><td>+0.41</td></tr><tr><td colspan="6">multi-objective bi-level optimization methods</td></tr><tr><td>MOML</td><td>84.67</td><td>96.72</td><td>96.85</td><td>92.75</td><td>-0.36</td></tr><tr><td>MoCo</td><td>84.38</td><td>97.26</td><td>97.03</td><td>92.89</td><td>-0.22</td></tr><tr><td>FORUM (ours)</td><td>85.64</td><td>98.63</td><td>97.96</td><td>94.07</td><td>+0.96</td></tr></table>

Table 3: Results on the NYUv2 dataset. Each experiment is repeated over 3 random seeds and the average is reported. The best results over baselines except STL are marked in **bold.**  $\uparrow (\downarrow)$  indicates that the higher (lower) the result, the better the performance.  

<table><tr><td rowspan="3">Methods</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="5">Surface Normal Prediction</td><td rowspan="3">Δp↑</td></tr><tr><td rowspan="2">mIoU↑</td><td rowspan="2">PAcc↑</td><td rowspan="2">AErr↓</td><td rowspan="2">RErr↓</td><td colspan="2">Angle Distance</td><td colspan="3">Within t°</td></tr><tr><td>Mean↓</td><td>Median↓</td><td>11.25↑</td><td>22.5↑</td><td>30↑</td></tr><tr><td>STL</td><td>53.50</td><td>75.39</td><td>0.3926</td><td>0.1605</td><td>21.9896</td><td>15.1641</td><td>39.04</td><td>65.00</td><td>75.16</td><td>0.00</td></tr><tr><td colspan="11">multi-task learning methods</td></tr><tr><td>EW</td><td>53.93</td><td>75.53</td><td>0.3825</td><td>0.1577</td><td>23.5691</td><td>17.0149</td><td>35.04</td><td>60.99</td><td>72.05</td><td>-1.78</td></tr><tr><td>UW</td><td>54.29</td><td>75.64</td><td>0.3815</td><td>0.1583</td><td>23.4805</td><td>16.9206</td><td>35.26</td><td>61.17</td><td>72.21</td><td>-1.52</td></tr><tr><td>PCGrad</td><td>53.94</td><td>75.62</td><td>0.3804</td><td>0.1578</td><td>23.5226</td><td>16.9276</td><td>35.19</td><td>61.17</td><td>72.19</td><td>-1.57</td></tr><tr><td>GradDrop</td><td>53.73</td><td>75.54</td><td>0.3837</td><td>0.1580</td><td>23.5392</td><td>16.9587</td><td>35.17</td><td>61.06</td><td>72.07</td><td>-1.85</td></tr><tr><td>GradVac</td><td>54.21</td><td>75.67</td><td>0.3859</td><td>0.1583</td><td>23.5804</td><td>16.9055</td><td>35.34</td><td>61.15</td><td>72.10</td><td>-1.75</td></tr><tr><td>CAGrad</td><td>53.97</td><td>75.54</td><td>0.3885</td><td>0.1588</td><td>22.4701</td><td>15.7139</td><td>37.77</td><td>63.82</td><td>74.30</td><td>-0.27</td></tr><tr><td>Nash-MTL</td><td>53.41</td><td>74.95</td><td>0.3867</td><td>0.1612</td><td>22.5662</td><td>15.9365</td><td>37.30</td><td>63.40</td><td>74.09</td><td>-1.01</td></tr><tr><td>RLW</td><td>54.13</td><td>75.72</td><td>0.3833</td><td>0.1590</td><td>23.2125</td><td>16.6166</td><td>35.88</td><td>61.84</td><td>72.74</td><td>-1.27</td></tr><tr><td colspan="11">first-order bi-level optimization methods</td></tr><tr><td>BVFIM</td><td>53.29</td><td>75.07</td><td>0.3981</td><td>0.1632</td><td>22.3552</td><td>15.9710</td><td>37.15</td><td>63.44</td><td>74.27</td><td>-1.68</td></tr><tr><td>BOME</td><td>54.15</td><td>75.79</td><td>0.3831</td><td>0.1578</td><td>23.3378</td><td>16.8828</td><td>35.29</td><td>61.31</td><td>72.40</td><td>-1.45</td></tr><tr><td colspan="11">multi-objective bi-level optimization methods</td></tr><tr><td>MOML</td><td>53.59</td><td>75.48</td><td>0.3839</td><td>0.1577</td><td>23.1487</td><td>16.5319</td><td>36.06</td><td>62.05</td><td>72.89</td><td>-1.26</td></tr><tr><td>MoCo</td><td>53.73</td><td>75.63</td><td>0.3838</td><td>0.1560</td><td>23.1922</td><td>16.5737</td><td>36.02</td><td>61.93</td><td>72.82</td><td>-1.06</td></tr><tr><td>FORUM (ours)</td><td>54.04</td><td>75.64</td><td>0.3795</td><td>0.1555</td><td>22.1870</td><td>15.6815</td><td>37.71</td><td>64.04</td><td>74.67</td><td>+0.65</td></tr></table>

Evaluation Metrics. (i) For the Office-31 dataset, following Lin et al. (2022), we use classification accuracy as the evaluation metric for each task and the average accuracy as the overall metric. (ii) For the NYUv2 dataset, following Liu et al. (2019); Lin et al. (2022), we use the mean intersection

Table 4: Results on the QM9 dataset. Each experiment is repeated over 3 random seeds and the average is reported.  $\uparrow (\downarrow)$  indicates that the higher (lower) the result, the better the performance. The best results over baselines except STL are marked in bold.  

<table><tr><td rowspan="2">Methods</td><td>μ</td><td>α</td><td>εHOMO</td><td>εLUMO</td><td>〈R2〉</td><td>ZPVE</td><td>U0</td><td>U</td><td>H</td><td>G</td><td>cv</td><td rowspan="2">Δp↑</td></tr><tr><td colspan="11">MAE↓</td></tr><tr><td>STL</td><td>0.062</td><td>0.192</td><td>58.82</td><td>51.95</td><td>0.529</td><td>4.52</td><td>63.69</td><td>60.83</td><td>68.33</td><td>60.31</td><td>0.069</td><td>0.00</td></tr><tr><td colspan="13">multi-task learning methods</td></tr><tr><td>EW</td><td>0.096</td><td>0.286</td><td>67.46</td><td>82.80</td><td>4.655</td><td>12.4</td><td>128.3</td><td>128.8</td><td>129.2</td><td>125.6</td><td>0.116</td><td>-146.3</td></tr><tr><td>UW</td><td>0.336</td><td>0.382</td><td>155.1</td><td>144.3</td><td>0.965</td><td>4.58</td><td>61.41</td><td>61.79</td><td>61.83</td><td>61.40</td><td>0.116</td><td>-92.35</td></tr><tr><td>PCGrad</td><td>0.104</td><td>0.293</td><td>75.29</td><td>88.99</td><td>3.695</td><td>8.67</td><td>115.6</td><td>116.0</td><td>116.2</td><td>113.8</td><td>0.109</td><td>-117.8</td></tr><tr><td>GradDrop</td><td>0.114</td><td>0.349</td><td>75.94</td><td>94.62</td><td>5.315</td><td>15.8</td><td>155.2</td><td>156.1</td><td>156.6</td><td>151.9</td><td>0.136</td><td>-191.4</td></tr><tr><td>GradVac</td><td>0.100</td><td>0.299</td><td>68.94</td><td>84.14</td><td>4.833</td><td>12.5</td><td>127.3</td><td>127.8</td><td>128.1</td><td>124.7</td><td>0.117</td><td>-150.7</td></tr><tr><td>CAGrad</td><td>0.107</td><td>0.296</td><td>75.43</td><td>88.59</td><td>2.944</td><td>6.12</td><td>93.09</td><td>93.68</td><td>93.85</td><td>92.32</td><td>0.106</td><td>-87.25</td></tr><tr><td>Nash-MTL</td><td>0.115</td><td>0.263</td><td>85.54</td><td>86.62</td><td>2.549</td><td>5.85</td><td>83.49</td><td>83.88</td><td>84.05</td><td>82.96</td><td>0.097</td><td>-73.92</td></tr><tr><td>RLW</td><td>0.112</td><td>0.331</td><td>74.59</td><td>90.48</td><td>6.015</td><td>15.6</td><td>156.0</td><td>156.8</td><td>157.3</td><td>151.6</td><td>0.133</td><td>-200.9</td></tr><tr><td colspan="13">first-order bi-level optimization methods</td></tr><tr><td>BVFIM</td><td>0.107</td><td>0.325</td><td>73.18</td><td>98.97</td><td>5.336</td><td>21.4</td><td>200.1</td><td>201.2</td><td>201.8</td><td>195.5</td><td>0.148</td><td>-228.5</td></tr><tr><td>BOME</td><td>0.105</td><td>0.318</td><td>72.10</td><td>88.52</td><td>4.984</td><td>12.6</td><td>138.8</td><td>139.4</td><td>140.0</td><td>136.1</td><td>0.124</td><td>-164.1</td></tr><tr><td colspan="13">multi-objective bi-level optimization methods</td></tr><tr><td>MOML</td><td>0.083</td><td>0.347</td><td>74.87</td><td>80.57</td><td>3.813</td><td>8.64</td><td>191.9</td><td>192.6</td><td>192.8</td><td>188.9</td><td>0.135</td><td>-165.1</td></tr><tr><td>MoCo</td><td>0.086</td><td>0.427</td><td>69.60</td><td>79.00</td><td>5.693</td><td>10.2</td><td>295.5</td><td>296.6</td><td>297.0</td><td>290.1</td><td>0.169</td><td>-267.6</td></tr><tr><td>FORUM (ours)</td><td>0.104</td><td>0.266</td><td>85.37</td><td>82.15</td><td>2.126</td><td>6.49</td><td>96.97</td><td>97.53</td><td>97.69</td><td>95.88</td><td>0.097</td><td>-73.36</td></tr></table>

over union (MIoU) and the class-wise pixel accuracy (PAcc) for the semantic segmentation task, the relative error (RErr) and the absolute error (AErr) for the depth estimation task, and the mean and median angle error as well as the percentage of normals within  $t^{\circ}$  ( $t = 11.25, 22.5, 30$ ) for the surface normal prediction task. (iii) For the QM9 dataset, following Fey & Lenssen (2019); Navon et al. (2022), we use mean absolute error (MAE) as the evaluation metric. (iv) Following Maninis et al. (2019); Lin et al. (2022), we use  $\Delta_{\mathrm{p}}$  as a metric to evaluate the overall performance on all the tasks. It is defined as the mean of the relative improvement of each task over the STL method, which is formulated as

$$
\Delta_ {\mathrm {p}} = 100 \% \times \frac {1}{m} \sum_ {i = 1} ^ {m} \frac {1}{N _ {i}} \sum_ {j = 1} ^ {N _ {i}} \frac {(- 1) ^ {s _ {i , j}} \left(M _ {i , j} - M _ {i , j} ^ {\mathrm {STL}}\right)}{M _ {i , j} ^ {\mathrm {STL}}},
$$

where  $N_{i}$  denotes the number of metrics for  $i$ -th task,  $M_{i,j}$  denotes the performance of an MTL method for the  $j$ -th metric in the  $i$ -th task,  $M_{i,j}^{\mathrm{STL}}$  is defined in the same way for the STL method, and  $s_{i,j}$  is set to 0 if a larger value represents better performance for the  $j$ -th metric in  $i$ -th task and otherwise  $s_{i,j} = 1$ .

Results. Table 2 shows the results on Office-31 dataset. The proposed FORUM method achieves the best performance in terms of average classification accuracy and  $\Delta_{\mathrm{p}}$ . The results on NYUv2 dataset are shown in Table 3. As can be seen, only FORUM achieves better performance than STL in terms of  $\Delta_{\mathrm{p}}$ . Moreover, FORUM performs well in the depth estimation and surface normal prediction tasks. Table 4 shows the results on QM9 dataset. FORUM again outperforms all the baselines in terms of  $\Delta_{\mathrm{p}}$ . Those results consistently demonstrate FORUM achieves state-of-the-art performance and is more effective than previous MOBLO methods such as MOML and MoCo.

# 6 CONCLUSION

In this paper, we propose FORUM, an efficient fully first-order gradient-based method for solving the MOBLO problem. Specifically, we reformulate the MOBLO problem to a constrained MOO problem and we propose a novel multi-gradient aggregation method to solve it. Compared with the existing MOBLO methods, FORUM does not require any hypergradient computation and thus is efficient. Theoretically, we provide a complexity analysis to show the efficiency of the proposed method and a non-asymptotic convergence guarantee for FORUM. Moreover, empirical studies demonstrate the proposed FORUM method is effective and efficient.

# REFERENCES

Abbas Abdelmaleki, Sandy Huang, Leonard Hasenclever, Michael Neunert, Francis Song, Martina Zambelli, Murilo Martins, Nicolas Heess, Raia Hadsell, and Martin Riedmiller. A distributional view on multi-objective policy optimization. In International Conference on Machine Learning, 2020.  
Daniel Angus. Crowding population-based ant colony optimisation for the multi-objective travelling salesman problem. In IEEE Symposium on Computational Intelligence in Multi-Criteria Decision-Making, 2007.  
Fan Bao, Guoqiang Wu, Chongxuan Li, Jun Zhu, and Bo Zhang. Stability and generalization of bilevel programming in hyperparameter optimization. In Neural Information Processing Systems, 2021.  
Rich Caruana. Multitask learning: A knowledge-based source of inductive bias. In International Conference on Machine Learning, 1993.  
Rich Caruana. Multitask learning. Machine learning, 28(1):41-75, 1997.  
Lesi Chen, Jing Xu, and Jingzhao Zhang. On bilevel optimization without lower-level strong convexity. arXiv preprint arXiv:2301.00712, 2023.  
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In European Conference on Computer Vision, 2018.  
Xi Chen, Ali Ghadirzadeh, Mårten Björkman, and Patric Jensfelt. Meta-learning for multi-objective reinforcement learning. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2019.  
Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In Neural Information Processing Systems, 2020.  
Jean-Antoine Désideri. Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. Comptes Rendus Mathematique, 350(5):313-318, 2012.  
Steven Diamond and Stephen Boyd. CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17(83):1-5, 2016.  
Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Efficient multi-objective neural architecture search via lamarckian evolution. arXiv preprint arXiv:1804.09081, 2018.  
Min Feng and Shengjie Li. An approximate strong kkt condition for multiobjective optimization. *TOP*, 26(3):489-509, 2018.  
Heshan Devaka Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen. Mitigating gradient bias in multi-objective learning: A provably convergent approach. In International Conference on Learning Representations, 2023.  
Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.  
Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning, 2017.  
Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning, 2018.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International Conference on Machine Learning, 2017.

Chengyue Gong, Xingchao Liu, and Qiang Liu. Automatic and harmless regularization with constrained and lexicographic optimization: A dynamic barrier approach. In Neural Information Processing Systems, 2021.  
Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning, 2020.  
Alex Gu, Songtao Lu, Parikshit Ram, and Tsui-Wei Weng. Min-max multi-objective bilevel optimization with applications in robust machine learning. In International Conference on Learning Representations, 2023.  
Ruichen Jiang, Nazanin Abolfazli, Aryan Mokhtari, and Erfan Yazdandoost Hamedani. A conditional gradient-based method for simple bilevel optimization with convex lower-level problem. In International Conference on Artificial Intelligence and Statistics, 2023.  
Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.  
Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning, 2023.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Baijiong Lin and Yu Zhang. LibMTL: A Python library for multi-task learning. Journal of Machine Learning Research, 24(209):1-7, 2023.  
Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. Transactions on Machine Learning Research, 2022.  
Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Neural Information Processing Systems, 2021a.  
Bo Liu, Mao Ye, Stephen Wright, Peter Stone, et al. Bome! bilevel optimization made easy: A simple first-order approach. In Neural Information Processing Systems, 2022a.  
Jia Liu and Yaochu Jin. Multi-objective search of robust neural architectures against multiple types of adversarial attacks. Neurocomputing, 453:73-84, 2021.  
Risheng Liu, Jiaxin Gao, Jin Zhang, Deyu Meng, and Zhouchen Lin. Investigating bi-level optimization for learning and vision from a unified perspective: A survey and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):10045-10067, 2021b.  
Risheng Liu, Xuan Liu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. A value-function-based interior-point method for non-convex bi-level optimization. In International Conference on Machine Learning, 2021c.  
Risheng Liu, Pan Mu, Xiaoming Yuan, Shangzhi Zeng, and Jin Zhang. A general descent aggregation framework for gradient-based bi-level optimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(1):38-57, 2022b.  
Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.  
Zhichao Lu, Kalyanmoy Deb, Erik Goodman, Wolfgang Banzhaf, and Vishnu Naresh Boddeti. Nsganetv2: Evolutionary multi-objective surrogate-assisted neural architecture search. In European Conference Computer Vision, 2020.  
Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning, 2015.

Debabrata Mahapatra and Vaibhav Rajan. Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization. In International Conference on Machine Learning, 2020.  
Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.  
Yuren Mao, Zekai Wang, Weiwei Liu, Xuemin Lin, and Pengtao Xie. Metaweighting: Learning to weight tasks in multi-task learning. In Findings of the Association for Computational Linguistics, 2022.  
Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In International Conference on Machine Learning, 2022.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Neural Information Processing Systems, 2019.  
Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data, 1(1):1-7, 2014.  
Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell. Adapting visual category models to new domains. In European Conference on Computer Vision, 2010.  
Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Neural Information Processing Systems, 2018.  
Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated back-propagation for bilevel optimization. In International Conference on Artificial Intelligence and Statistics, 2019.  
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgb images. In European Conference on Computer Vision, 2012.  
Daouda Sow, Kaiyi Ji, Ziwei Guan, and Yingbin Liang. A constrained optimization approach to bilevel optimization with multiple inner minima. arXiv preprint arXiv:2203.01123, 2022.  
Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In International Conference on Learning Representations, 2021.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
Runzhe Yang, Xingyuan Sun, and Karthik Narasimhan. A generalized algorithm for multi-objective reinforcement learning and policy adaptation. In Neural Information Processing Systems, 2019.  
Feiyang Ye, Baijiong Lin, Zhixiong Yue, Pengxin Guo, Qiao Xiao, and Yu Zhang. Multi-objective meta learning. In Neural Information Processing Systems, 2021.  
Runsheng Yu, Weiyu Chen, Xinrun Wang, and James Kwok. Enhancing meta learning via multi-objective soft improvement functions. In International Conference on Learning Representations, 2023.  
Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Neural Information Processing Systems, 2020.  
Zhixiong Yue, Baijiong Lin, Yu Zhang, and Christy Liang. Effective, efficient and robust neural architecture search. In International Joint Conference on Neural Networks, 2022.  
Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering, 34(12):5586-5609, 2022.

Aimin Zhou, Bo-Yang Qu, Hui Li, Shi-Zheng Zhao, Ponnuthurai Nagaratnam Suganthan, and Qingfu Zhang. Multiobjective evolutionary algorithms: A survey of the state of the art. Swarm and evolutionary computation, 1(1):32-49, 2011.  
Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu. On the convergence of stochastic multi-objective gradient manipulation and beyond. In Neural Information Processing Systems, 2022.
