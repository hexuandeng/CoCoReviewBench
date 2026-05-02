# OTCOP: LEARNING OPTIMAL TRANSPORT MAP VIA CONSTRAINT OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

The approximation power of the neural network makes it an ideal tool to learn optimal transport maps. However, existing methods are mostly based on the Kantorovich duality and require regularizations and/or special network structures. In this paper, we propose a direct constraint optimization algorithm for the computation of optimal transport maps based on the Monge formulation. We solve this constraint optimization problem by using three different methods: the Langrangian multiplier method, the augmented Lagrangian method, and the alternating direction method of multipliers (ADMM). We demonstrate a significant accuracy of learned optimal transport maps on high dimensional benchmarks. Moreover, we show that our methods reduce the regularization effects and accurately learn the target distributions at a lower transport cost.

# 1 INTRODUCTION

There has been a great interest in applying modern machine learning techniques for finding optimal transport maps between two distributions. Different from traditional computational methods that solve PDEs for optimal transport maps (Benamou & Brenier (2000); Angenent et al. (2003); Li et al. (2018)), modern machine learning techniques aim to solve the problem directly by optimizations. The Sinkhorn Distance method Cuturi (2013); Peyre et al. (2019), the regularized OT dual Seguy et al. (2017) have been used to find large scale optimal transport maps between discrete probability distributions and have been used to train generative networks Geneva et al. (2018); Sanjabi et al. (2018). The Input Convex Neural Network (ICNN) is used to construct a convex Brenier potential for finding optimal transport maps Makkuva et al. (2020) between continuous distributions and is recently used in population dynamics Bunne et al. (2022), which combines the ICNN and Sinkhorn distance methods Amos et al. (2022). Despite these successes, most methods are based on the duality formulation and avoids the direct treatment on the Monge problem.

In this paper, we focus on the direct solution of the Monge problem. The Monge problem (Monge (1781)) directly seeks to identify the optimal transport maps and is a nonlinear constraint optimization problem. The major difficulty in solving the problem numerically is that it is nonlinear and includes a constraint that the push-forward distribution is equal to the target distribution, which is difficult to implement. Therefore, most optimal transport algorithms avoid directly solving the Monge problem but use the Kantorovich duality (Kantorovich (1942)), for which the objective function is linear and the transport map is obtained by taking the gradient of the Brenier potential for the quadratic cost. However, these two problems are not always identical (Villani (2009)) and it is desirable to find a direct approach for the Monge problem.

The Monge problem has been solved numerically using optimization based methods with polynomial approximations. For example, a Lagrangian penalty method was used to find optimal transport maps approximated by polynomials for Bayesian inference El Moselhy & Marzouk (2012) and space discretization was used in Haber et al. (2010) to calculate the Jacobian matrix of the transport maps and transferred the optimization to finite dimensional spaces. However, their approaches are limited to low dimensions as number of grids expands exponentially as dimensions become large. Considering the success of deep neural networks in approximating high dimensional data, the integration of classical constraint optimization methods and neural networks holds a promise.

One successful application of the optimal transport theory to deep learning is the Wasserstein Generative Adversarial Network (WGAN) Arjovsky et al. (2017). However, WGAN only use the optimal

transport distance as a loss function and does not target at finding the optimal transport maps. It is desirable to study whether it is possible to lower the transport cost of the map learned by WGAN or other networks using the algorithm for finding optimal transport maps.

This paper presents a new approach for finding optimal transport maps between two continuous distributions. We make the following contributions:

- We integrate three constraint optimization algorithms including the Standard Lagrangian (SL), the Augmented Lagrangian method (AL) and the Alternating Direction Method of Multipliers (ADMM) with neural networks to solve the Monge problem of optimal transport with provable guarantees (Theorem 1-3).  
- We show that our method is able to find an accurate optimal transport map between Gaussian distributions, both theoretically (Theorem 2) and experimentally. Moreover, we apply our method to WGAN and show that our method can find a generative map with lower transport cost while not sacrificing the quality of outputs.  
- We compare the three algorithms and find the SL algorithm introduces errors but is simple and easy to implement, while AL and ADMM algorithms can find exact results and are more robust, and ADMM gives a lower transport cost in general.

Notations. We use the notations  $\alpha_{d} = (\alpha ,\dots ,\alpha)\in \mathbb{R}^{d}$  and  $\alpha_{d\times d}$  for the constant  $d\times d$  matrix. The transport cost of a map  $T$ , which pushes distribution  $\mu$  to  $\nu$ , is defined to be  $\mathbb{E}_{x\sim \mu}[|x - Tx|^2]$ .

# 2 THE MONGE PROBLEM AS CONSTRAINT OPTIMIZATION

# 2.1 THE MONGE PROBLEM

Let  $(X,\mu)$ ,  $(Y,\nu)$  be two separable metric probability spaces. The Monge problem is to find a transport map  $T:X\mapsto Y$  that realizes the infimum

$$
\inf  \left\{\int_ {X} c (x, T x) d \mu (x) \mid T _ {\#} \mu = \nu \right\} \tag {1}
$$

where  $T_{\#}\mu$  denotes the push forward of  $\mu$  and  $c: X \times Y \to \mathbb{R}_+$  is a Borel measurable function which is lower semicontinuous. In this paper, we simply take the distance  $|x - y|^2$  but our method applies to other distance functions.

The existence of the Monge problem is difficult and does not hold always. However, under suitable conditions, for example for continuous distributions without atoms, the existence and uniqueness of the Monge problem is guaranteed (see for example, (Villani, 2009, Theorem 5.30). Therefore, here we focus on learning transport maps between continuous distributions. For discrete distributions, one can apply dequantization techniques to transform them to continuous distributions Ho et al. (2019).

# 2.2 THE MONGE PROBLEM AS CONSTRAINT OPTIMIZATION

In order to solve the Monge problem, we use a generative network, denoted by  $T_{\theta}$  with parameter set  $\theta$ , which inputs random samples from the distribution  $\mu$  and generates samples representing the target distributions  $\nu$ . As can be seen from the definition, the Monge problem is a constraint optimization problem. However, the constraint  $T_{\#}\mu = \nu$  is a highly nonlinear constraint. In order to impose this constraint, we take  $\mathfrak{d}(\cdot |\cdot)$  to be a distance function (such as the Wasserstein distance, the MMD (Gretton et al. (2012)) or the IPM (Müller (1997))) or a probability divergence (such as the Kullback-Leibler (KL) divergence). The constraint optimization problem reads as

$$
\min  _ {\theta} \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta} x | ^ {2} \right], \quad \text {s . t .} \quad \mathfrak {d} \left(T _ {\theta \#} \mu | \nu\right) = 0. \tag {2}
$$

The objective of this paper is to solve the above problem using techniques from the constraint optimization theory (Bertsekas (2014)).

Since a neural network may not fully reveal the target distributions  $\nu$ , the above problem can be relaxed to

$$
\min  _ {\theta} \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta} x | ^ {2} \right], \quad \text {s . t .} \quad \mathfrak {d} \left(T _ {\theta \#} \mu | \nu\right) \leq \alpha , \tag {3}
$$

When  $\alpha$  goes towards zero, we can prove that the solution  $T_{\theta_{\alpha}}$  to the problem (3) converges to the solution of the original Monge problem (1). The following theorem holds:

Theorem 1. Let  $\mu, \nu$  be two probability measures on  $\mathbb{R}^d$  with finite second moments and are absolutely continuous. Let  $T_{\theta}$  be given by a neural network with bounded width (each with at least  $2d + 2$  neurons) and arbitrary depth, and with non-affine activation functions. Suppose for any  $\alpha > 0$ , there exists a solution  $\theta_{\alpha}^{*}$  to problem (2), then as  $\alpha \to 0$ ,  $T_{\theta_{\alpha}^{*}} \to T$  where  $T$  is a solution of the Monge problem (1). Moreover,  $\sup_{x \in \mathcal{X}} |T_{\alpha}(x) - T(x)|_{\mathcal{C}} \leq C\alpha$  for some constant  $C$  for any compact subset  $\mathcal{X} \in \mathbb{R}^d$ .

Proof of the above theorem follows from the universal approximation theorem (Kidger & Lyons (2020)) and the existence theorem of the Monge problem (Villani (2009)), and is given in Appendix A.1.

# 2.3 EXAMPLE: THE MONGE PROBLEM FROM GAUSSIAN TO GAUSSIAN

For the case when  $\mu \sim \mathcal{N}(X_1,\Sigma_1)$  and  $\nu \in \mathcal{N}(X_2,\Sigma_2)$  are two multivariate normal distributions, the optimal transport map is unique and can be explicitly given by  $T^{*}:x\mapsto X_{2} + A^{*}(x - X_{1})$  with  $A^{*} = \Sigma_{1}^{-1 / 2}(\Sigma_{1}^{1 / 2}\Sigma_{2}\Sigma_{1}^{1 / 2})^{1 / 2}\Sigma_{1}^{-1 / 2}$  (Olkin & Pukelsheim (1982)). Taking  $\mathfrak{d} = D_{\mathrm{KL}}$  to be the KL-divergence, we prove that the solution to problem (2) with  $T_{\theta}x = Ax + b$  ( $\theta = \{A\in \mathbb{R}^{d\times d},b\in \mathbb{R}^d\}$ ) is the optimal transport map  $T^{*}$  (Theorem 2 in Appendix A.1).

# 3 CONSTRAINT OPTIMIZATION FOR OPTIMAL TRANSPORT

We propose to leverage three different algorithms to solve the constraint problem (2).

# 3.1 PENALTY METHOD (OTCOP-P)

Standard Lagrangian (SL). We introduce a Lagrangian multiplier  $\lambda$  and take the Lagrangian function as

$$
\mathcal {L} _ {S L} (\theta , \lambda) = \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta} x | ^ {2} \right] + \lambda \mathfrak {d} \left(T _ {\theta \#} \mu | \nu\right). \tag {4}
$$

Then the solution to the problem (2) is a saddle point of the above Lagrangian. By duality theory, for each  $\alpha \geq 0$ , the problem (3) corresponds to the duality problem  $\min_{\theta} \mathcal{L}_{SL}(\theta, \lambda, 0)$  for a  $\lambda \in [0, \infty]$ . Hence we can take a suitable  $\lambda$  to solve the problem (2) approximately.

According to the Brenier's polar factorization theorem Brenier (1991), the optimal transport map should satisfy  $\nabla \times T = 0$ , hence we can add an additional term  $|\nabla \times T|^2$  into the above Lagrangian to impose this constraint. We will show experimentally that without this term, this constraint is almost satisfied and we will not included in our implementations.

Quadratic penalty  $(QP)$ . Instead of taking  $\mathfrak{d}(T_{\theta \#}|\nu)$ , we can take a quadratic penalty loss

$$
\mathcal {L} _ {Q P} (\theta , \rho) = \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta} x | ^ {2} \right] + \frac {1}{2} \rho \left(\mathfrak {d} \left(T _ {\theta \#} \mu | \nu\right)\right) ^ {2}. \tag {5}
$$

As  $\rho$  goes towards infinity, the constraint violations is penalized with increasing severity. For example, we can take  $\rho_{k}$  at the  $k$ th training step to be increased by multiplying by a constant bigger than 1 and parameter  $\theta$  can be updated using gradient descent considering  $\rho$  as a constant.

Convergence. Suppose there exists a global minimizer to the problem (2), and  $\theta_{k}$  is the exact minimizer of  $\mathcal{L}_{QP}(\theta, \rho_{k})$  and  $\rho_{k} \uparrow \infty$ . Then any limit point of the sequence  $\{\theta_{k}\}$  is a solution to problem (2). Moreover, for any  $\varepsilon > 0$ , there exists a sufficient large  $K > 0$ ,  $|\theta_{k} - \theta^{*}| \leq \varepsilon$  for  $k \geq K$  (see (Nocedal & Wright, 1999, Theorem 17.1)). In addition, without assuming a global minimizer, for a sequence  $\theta_{k}$  such that  $\nabla_{\theta_{k}} \mathcal{L}_{QP}(\theta_{k}; \rho_{k}) \to 0$ , its all limit points  $\theta^{*}$  satisfy the Karush-Kuhn-Tucker (KKT) conditions and there exists a subsequence such that  $\lim_{k \to \infty} (\rho_{k} \mathfrak{d}(T_{\theta_{k} \#} | \nu) = \lambda^{*}$ , where  $\lambda^{*}$  is the multiplier that satisfies the KKT condition (see Appendix A.4 for the KKT condition and see (Nocedal & Wright, 1999, Theorem 17.2) for the proof).

Advantages and disadvantages. The penalty method is simple and easy to implement. However, since the optimal value of the Lagrangian multiplier  $\lambda$  is unknown (SL) or the optimal condition

for the Lagrangian multiplier  $\rho$  is infinite (QP), the penalty always introduces errors and the exact solution to the problem (2) cannot be reached. Moreover, the Hessian of the Lagrangian  $\nabla_{\theta \theta}^{2}\mathcal{L}_{QP}$  becomes singular as  $\rho$  goes towards infinity and cause ill-condition problems. These issues can be solved by the methods below, but at the expense of a more computational cost.

# 3.2 THE AUGMENTED LAGRANGIAN METHOD (OCTOP-AL)

In order to overcome the above issues, we can use the augmented Lagrangian method, by taking the loss function as

$$
\mathcal {L} _ {A L} (\theta , \lambda , \rho) = \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta} x | ^ {2} \right] + \lambda \mathfrak {d} \left(T _ {\theta \#} \mu | \nu\right) + \frac {\rho}{2} \left(\mathfrak {d} \left(T _ {\theta \#} \mu | \nu\right)\right) ^ {2}. \tag {6}
$$

The above function combines standard Lagrangian penalty (4) and quadratic Lagrangian penalty (5). At the  $k$ th iteration, fix  $\lambda_k, \rho_k$  and solve  $\theta_k = \arg \min_{\theta} \mathcal{L}_{AL}(\theta, \lambda_k, \rho_k)$ . After the minimization, we update  $\lambda$  by  $\lambda_{k+1} = \lambda_k + \rho_k \mathfrak{d}(T_{\theta_k \#} \mu | \nu)$ . Comparing the KKT conditions of the SL and the AL (see Appendix A.4) implies  $\lambda_k + \rho_k \mathfrak{d}(T_{\theta_k \#} \mu | \nu) \approx \lambda^*$  when  $\lambda_k$  is taken close to  $\lambda^*$ . Hence  $\mathfrak{d}(T_{\theta_k \#} \mu | \nu) \approx (\lambda^* - \lambda^k) / \rho$ . Compared to the quadratic penalty method that  $\mathfrak{d}(T_{\theta_k \#} \mu | \nu) \approx \lambda^* / \rho$ , the infeasibility in  $\theta_k$  will be much smaller. Moreover, for certain choice of  $\rho$ , the local solution of (2) is a strict local minimizer of  $\mathcal{L}_{AL}(\theta, \lambda, \rho)$  ((Nocedal & Wright, 1999, Chapter 17)).

Convergence. One of the nice properties of the AL method is that for the exact Lagrangian multiplier  $\lambda^{*}$ , the solution  $\theta^{*}$  of the problem (2) is a strict minimizer of  $\mathcal{L}_{AL}(\theta, \lambda^{*}, \rho)$  for all  $\rho$  sufficiently large. The existence of a threshold is proved under the condition that  $\nabla_{\theta}^{2}\mathcal{L}_{SL}(\theta^{*}, \lambda^{*})$  is locally strictly positive ((Nocedal & Wright, 1999, Theorem 17.6)). Thus we can take  $\rho$  to be increasing at each minimizing step and when  $\rho$  becomes bigger than some threshold value  $\bar{\rho}$ , gradient descent methods could find the local minimizer around  $\theta^{*}$ .

Advantages and disadvantages. AL method introduces the multiplier estimates and reduces the likelihood that large values of  $\rho$  will be needed to obtain good feasibility and accuracy. The method is also simple and easy to implement. However, since this is a min-max method, training may experience oscillations and slower convergence rates.

# 3.3 ADMM METHOD (OCTOP-ADMM)

The ADMM method blends the decomposition techniques and the AL method and provides an efficient way for constraint optimizations Boyd et al. (2011). Let  $S$  be the set  $S = \{T_{\theta} : T_{\theta \#} \mu = \nu\}$ , problem (2) can be rewritten into the form

$$
\min  _ {\theta} \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta} x | ^ {2} \right] + \mathbf {1} _ {S} \left(T _ {\theta}\right), \tag {7}
$$

where  $\mathbf{1}_S$  is the indicator function that equals 0 if  $T_{\theta} \in S$  and equals  $\infty$  if  $T_{\theta} \notin S$ . In order to apply the ADMM method, we rewrite the above problem into the form

$$
\min  _ {\theta_ {1}, \theta_ {2}} \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta_ {1}} x | ^ {2} \right] + \mathbf {1} _ {S} \left(T _ {\theta_ {2}}\right), \quad \text {s . t .} \quad T _ {\theta_ {1}} = T _ {\theta_ {2}}. \tag {8}
$$

In the ADMM method, we alternatively update  $\theta_{1}$  and  $\theta_{2}$ . First we take  $\theta_{2}$  to be constant and take the minimization of the above problem over  $\theta_{1}$ , and then we project  $\theta_{2}$  onto the space  $S$ . In detail, we introduce the loss function

$$
\begin{array}{l} \mathcal {L} _ {A D M M} (\theta_ {1}, \theta_ {2}, \Lambda , \rho) = \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta_ {1}} x | ^ {2} \right] + \mathbf {1} _ {\mathfrak {d} (T _ {\theta_ {2} \#} \mu | \nu) = 0} \\ + \Lambda^ {T} \left(T _ {\theta_ {1}} x - T _ {\theta_ {2}} x\right) + \frac {\rho}{2} \left(\mathbb {E} _ {x \sim \mu} \left[ \left| T _ {\theta_ {1}} x - T _ {\theta_ {2}} x \right| ^ {2} \right]\right). \tag {9} \\ \end{array}
$$

Here  $\Lambda \in \mathbb{R}^d$  is the multiplier. The training procedure is given by

1.  $\theta_1^k = \arg \min_{\theta_1}\mathcal{L}_{ADMM}(\theta_1,\theta_2^k,\Lambda^k,\rho)$  (assuming  $\mathbf{1}_{\mathfrak{d}(T_{\theta_2\#}\mu |\nu) = 0} = 0$  
2.  $\theta_2^k = \arg \min_{\theta_2}\mathfrak{d}(T_{\theta_2\#}\mu |\nu)$  
3.  $\Lambda_{k + 1} = \Lambda_k + \rho \mathbb{E}_{x\sim \mu}(T_{\theta_1^{k + 1}\#}(x) - T_{\theta_2^{k + 1}\#}(x)).$

Convergence. The convergence of ADMM method is only known to hold under convex conditions or for some non-convex problems (Boyd et al. (2011)). Using results of Wang et al. (2019), we can

prove the convergence of the ADMM method if we modify the above method by relaxing problem (8) to

$$
\min  _ {\theta} \mathbb {E} _ {x \sim \mu} \left[ | x - T _ {\theta_ {1}} x | ^ {2} \right] + \eta_ {\varepsilon} (\mathfrak {d} (T _ {\theta_ {2}}, S)), \quad \text {s . t .} \quad \theta_ {1} = \theta_ {2}, \tag {10}
$$

where  $\eta_{\varepsilon}$  is the mollifier function which converges to the  $\delta$ -function as  $\varepsilon \to 0$ . We show that the ADMM method converges to the KKT points and if the corresponding Lagrangian is a Kurdyka-Łojasiewicz function, the ADMM method converges globally to the unique solution (see Appendix A.3 for details). As a consequence, there exists a convergence sequence of the ADMM method for the problem (8) approximately.

Advantage and disadvantages. The advantage of ADMM is that it decomposes the Monge problem into two sub-problems: minimizing the transport cost and minimizing the  $D_{\mathrm{KL}}$ . Compared to the AL method, AMDD solves two decomposed minimization problems and AMDD may also converge faster than the AL method in some situations (Wang et al. (2019)). However, the method also solves a min-max problem and may facing oscillations in the training process.

# 3.4 IMPLEMENTATION OF THE ALGORITHMS

The implementations of the algorithms are given in Algorithm 1, 2, 3. When the distributions  $\mu$  and  $\nu$  are given, we may use the divergence  $D_{\mathrm{KL}}$  and the metric  $\mathfrak{d} = D_{\mathrm{KL}}$ . For the general case, we can take  $\mathfrak{d} = \mathfrak{d}_w$  to be a discriminator network with parameter set  $w$ . The discriminator network is updated each training step after the parameter of the generative network is updated.

Algorithm 1 Solving the Monge problem with the penalty method  
Input Data:  $X\sim \mu ,Y\sim \nu$    
Constants:  $\lambda_0,\rho_0,\alpha >1$    
Training step:  $\eta$    
for number of training iterations   
do for m steps do  $\theta \leftarrow \theta -\eta \nabla_{\theta}\mathcal{L}_{SL}$  (or  $\theta \leftarrow \theta -\eta \nabla_{\theta}\mathcal{L}_{QP})$    
end for Update  $\rho \leftarrow \alpha \rho$    
end for

Algorithm 2 Solving the Monge problem with the augmented Lagrangian method  
Input Data:  $X\sim \mu ,Y\sim \nu$    
Constants:  $\lambda_0,\rho_0,\alpha >1$    
Training step:  $\eta$    
for number of training iterations do for m steps do  $\theta \gets \theta -\eta \nabla_{\theta}\mathcal{L}_{AL}$ $w\gets w + \eta \nabla_w\mathcal{L}_{AL}$  (if  $\mathfrak{d} = \mathfrak{d}_w$  end for Update  $\lambda \leftarrow \lambda +\rho \mathfrak{d}(T_{\theta \#}\mu |\nu)$  Update  $\rho \leftarrow \alpha \rho$    
end for

Algorithm 3 Solving the Monge problem with the ADMM  
Input Data:  $X\sim \mu ,Y\sim \nu$  Constants:  $\lambda_0,\rho_0,\alpha >1$  ,Training step:  $\eta$    
for number of training iterations do   
for  $m_{1}$  steps do   
 $\theta_{1}\gets \theta_{1} - \eta \nabla_{\theta}\mathcal{L}_{ADMM}(\theta_{1},\theta_{2},\lambda ,\rho)$    
end for   
for  $m_{2}$  steps do   
 $\theta_{2}\leftarrow \theta_{2} - \eta \nabla_{\theta}\mathfrak{d}(T_{\theta_{2}\#}\mu |\nu)$ $w_{2}\gets w_{2} + \eta \nabla_{w}\mathfrak{d}(T_{\theta_{2}\#}\mu |\nu)$  (if  $\mathfrak{d} = \mathfrak{d}_w$    
end for   
Update  $\lambda \gets \lambda +\rho \mathfrak{d}(T_{\theta_1\#}\mu |T_{\theta_2\#}\mu)$    
end for

# 4 EXPERIMENT

# 4.1 MULTIVARIATE NORMAL DISTRIBUTIONS

Linear maps. First, we consider the optimal transport between multivariate normal distributions  $\mu = \mathcal{N}(X_1,\Sigma_1)$  and  $\nu = \mathcal{N}(X_2,\Sigma_2)$ . Considering the map given by  $T_{\theta} = Ax + b$  with  $\theta = \{A\in$

$\mathbb{R}^{d\times d}, b \in \mathbb{R}^d\}$ , we prove in Theorem 2 that the solution to problem (2) gives the correct solution to the Monge problem. Indeed, in this case, the SL, AL and ADMM algorithms reduce to optimization of linear objective function constraint by a nonlinear function and it could be theoretically analyzed by the constraint optimization theory Bertsekas (2014). Here we take  $X_1 = 0_2$ ,  $\Sigma_1 = I_2$  and  $X_2 = 0_2$ ,  $\Sigma_2 = [[4,1],[1,4]]$ , let  $T_{\theta}x = Ax$  with  $A = [[a,b],[b,a]]$  ( $\theta = \{a,b\}$ ). Then the problem (2) reduces to

$$
\begin{array}{l} \min _ {a, b} 2 (1 - a) ^ {2} + 2 b ^ {2}, \\ \mathrm {s . t .} D _ {\mathrm {K L}} (T _ {\theta} \mu | \nu) = \frac {1}{3 0} (- 1 5 \log \left(\frac {1}{1 5} \left(a ^ {2} - b ^ {2}\right) ^ {2}\right) + 8 a ^ {2} - 4 a b + 8 b ^ {2} - 3 0) = 0. \\ \end{array}
$$

The landscapes of  $D_{\mathrm{KL}}$  and  $\mathcal{L}_{SL}(\theta, \lambda = 1)$  as well as the value of  $\min_{\theta} \mathcal{L}_{SL}(\theta, \lambda)$  as functions of  $\lambda$ , are plotted in Figure 1. As can be seen from the figure, the function  $D_{\mathrm{KL}}(T_{\theta \#} \mu | \nu)$  has multiple minimizers (red points), whereas the Lagrangian  $\mathcal{L}_{SP}(\theta, 1)$  has a unique global minimizer (blue point). Hence minimizing the Lagrangian  $\mathcal{L}_{SL}$  helps to find the optimal transport maps by finding the approximate map with the lowest transport cost among all maps that realize the target distributions. More importantly, from the figure, the whole domain can be divided into four pieces by the landscape of  $D_{\mathrm{KL}}$ , and starting in each piece, gradient descent method will converge to one of the four different points. In contrast, the landscape of the Lagrangian changes dramatically and starting from any point, gradient descent method will only converge to one point.

The Langrangian introduces errors for  $\lambda$  finite. As can be seen from the leftmost figure, the constraint  $D_{\mathrm{KL}} = 0$  is more relaxed as  $\lambda$  becomes smaller. However, for large  $\lambda$ , the barrier region becomes wider and the gradient descent may converge to a local minimizer away from the optimal result. For small  $\lambda$ , the gradient descent is easy to find the global minimizer, but since penalty introduces error, the relaxation effect of the constraints can also push the minimizer away from the solution of the Monge problem. The choice of  $\lambda$  needs to take this tradeoff into considerations. Training using a one layer linear neural network confirms the above analysis.

![](images/4516bbb594ee3ce5ac36d1b55e1222b76bc42f6684bf529dd40f2382a29af6be.jpg)  
Figure 1: Graph of the KL error and the loss function of the SP method (left:  $\min_{\theta} \mathcal{L}_{AL}(\theta, \lambda)$  as well as the corresponding  $D_{\mathrm{KL}}$  and transport cost as functions of  $\lambda$ ; middle: landscape of  $D_{\mathrm{KL}}$  as functions of  $a, b$ ; right: landscape of  $\mathcal{L}_{SL}(\theta, 1)$  as functions of  $a, b$ ). Red points: minimizers of the  $D_{\mathrm{KL}}$ , blue point: minimizer of  $\mathcal{L}_{SL}(\theta, 1)$ .

Training with neural networks with nonlinear activation function. Next we present our results on the training of Gaussian to Gaussian distributions with neural networks with nonlinear activation function. We take  $X_{1} = 0_{d}$ ,  $X_{1} = 1_{d}$  and  $\Sigma_{1} = I_{d}$ ,  $\Sigma_{2} = 3I_{d} + 1_{d\times d}$ . Theoretical analysis gives that the optimal transport distance is  $2d$  (see Theorem 2). Since here we use the  $D_{\mathrm{KL}}$  as distance function which is always positive, so the QP method behaves similarly as the SL method with different multiplier. Hence the results of the QP method is not presented here. The results are given in Table 1. and the 784D Gaussian is taken for  $X_{1} = 0_{784}$ ,  $X_{1} = 2\cdot 1_{784}$  with  $\Sigma_{1} = \Sigma_{2} = I_{784}$  (theoretical result of the optimal transport distance is  $784*4$ ). As can be seen from the table, all three algorithms give nice result and learn approximately the optimal transport map with a high accuracy. Training is done by using a  $d$  width and 10 depth neural network with tanh activation for all cases except for 78D Gaussian, for which a 100 depth neural network is used in order to learn the correlations correctly. Remarkably, for the highly correlated distributions in high dimensions, our method gives a nice result.

ADMM learns a lower transport cost. From the figure, we can see that the ADMM method learns a lower transport cost compared to other methods. This is because during training, minimizing the

Table 1: Training results on Gaussian and Gaussian mixtures  

<table><tr><td>benchmark</td><td>method</td><td>DKL</td><td>Transport cost/d</td><td>benchmark</td><td>DKL</td><td>Transport cost/d</td></tr><tr><td rowspan="3">2D Gaussian</td><td>SL</td><td>0.002</td><td>2.018</td><td rowspan="3">78D Gaussian</td><td>0.228</td><td>1.866</td></tr><tr><td>AL</td><td>0.002</td><td>2.021</td><td>0.260</td><td>2.290</td></tr><tr><td>ADMM</td><td>0.003</td><td>1.950</td><td>0.805</td><td>2.250</td></tr><tr><td rowspan="3">784D Gaussian</td><td>SL</td><td>0.365</td><td>3.981</td><td rowspan="3">2D mixture</td><td>0.034</td><td>0.048</td></tr><tr><td>AL</td><td>0.333</td><td>4.001</td><td>0.021</td><td>0.066</td></tr><tr><td>ADMM</td><td>0.399</td><td>3.998</td><td>0.059</td><td>0.035</td></tr></table>

$D_{\mathrm{KL}}$  between the generated and target distribution may not converge to the solution to the Monge problem, as illustrated in the simple 2D case above. Hence, the splitting feature enables ADMM to learn a transport map with lower transport cost. This can also be seen from the learning curve in Figure 2. The transport of the second network  $(T_{\theta_2})$  has a higher transport cost when learning the target distribution, whereas the first network  $(T_{\theta_1})$  has a lower transport cost, while the learned target distribution remains accurate.

![](images/165878f8fa7c6275d3e6929ef017780dcf762b1af4f21ce4f116e2f921d3ccc3.jpg)  
Figure 2: Training curves of the three algorithms for 78D Gaussian benchmark (the increasing line is the transport cost, and the decreasing line is the  $D_{\mathrm{KL}}$ . Left: the AL method; right: the ADMM method, orange line is for the first network and blue line is for the second network).

![](images/5302e78f0252e562038b251d9e98d93f5941c9ef35e3c2dbc2fdf7cae64d2c2c.jpg)

![](images/32b93e589c614b3b1efbc6b2d848d679aaa834727b84df5abc1da61c4cbc7af5.jpg)

# 4.2 GAUSSIAN TO GAUSSIAN MIXTURES

We take a four component Gaussian mixture, each with variance matrix  $0.5I_2$ , centers lying on the four corners of the square  $[-1,1]^2$  and learn the optimal transport map between two dimensional standard Gaussian to this mixture distribution. We plot the Jacobian graph of the learned map and the value of the  $D_{\mathrm{KL}}$  between target distribution and the prediction by the network in Figure 3. As can be seen from the figure, solely minimizing the  $D_{\mathrm{KL}}$  does not fulfil the right directions of the optimal transport maps. Using our method, the learned map is more balanced and approximately satisfies the condition  $\nabla \times Tx = 0$ . Note that here we donot include the penalty of  $|\nabla \times Tx|^2$ . From the learning curves, we confirm the findings above that ADMM learns a lower transport cost than the other methods. Here the transport cost of the second network converges to around 0.07, while the transport cost of the first network converges to about one half of that of the second network.

AL and ADMM methods are more robust. Compared to the SL, AL gives a more robust result with respect to the value of  $\lambda$ . For a well chosen  $\lambda$ , SL performs as good as AL. However, if  $\lambda$  is not taken properly, the obtained transport cost will be higher or the target distribution is not well realized. For ADMM method, we can see the choice of  $\rho$  affects the training process, but in a big range, the choice of  $\rho$  has little effects on the final result (see appendix A.5 for the graphs indicating this finding). This confirms the benefits of the AL and ADMM method in the literature (Nocedal & Wright (1999); Boyd et al. (2011)).

# 4.3 IMPROVED TRANSPORT COST OF GANS

We test our method on the Wasserstein GANs and show our method lowers the transport cost of WGANs.

![](images/915e8085fe27f3e9361e8dcd11427c2bdefb1105b5314e41b78925a629eabd33.jpg)

![](images/0d4f0052bbc36b9ce63879cdc0bf652547a44b8a089f8e07a11f6c679ba7fa52.jpg)  
Figure 3: Graph of the KL error and the loss function of the minimization of  $D_{\mathrm{KL}}$ , min  $D_{\mathrm{KL}}$ , the SP, AL and the ADMM method (from left to right). The  $D_{\mathrm{KL}}$  curve is the decreasing line and the transport cost is the increasing line. For the bottom right figure, orange line is for the first network and blue line is for the second network

![](images/1424286b525464249772176d8ba3dc0b6908794688ed0bdf98043b8408b87e9d.jpg)

![](images/6b09316f2f958a8c2c08e90d68fbed4213a34283f2f47f143e5e8d00708fe222.jpg)

![](images/8afc46f82158a2dac4a8a5995b4100fe10e55d1a8f1113b00be752ef449c3155.jpg)

![](images/a36a2b874fbf1906059811d5d4536885e3d92ab941294b6ab0300b97ca4757d4.jpg)

![](images/1d02d872ec427adb6bf7951afaed7c3d3a630053582868f76fe2f19589c8b8f5.jpg)

![](images/d5e6ea2b0e2ebd800c713e9a0901e2199b042f32c10f3da84bdd718cde601d8f.jpg)

Gaussian mixtures. First we use the Wasserstein GANs to learn the Gaussian mixture distributions described in the previous subsection. We train a GAN with a generator and discriminator both with 4 linear layers, each hidden layer with width 400 and activation function ReLU. The generated samples are plotted in Figure 4. As can be seen from the results, classical WGAN gives a transport cost 2.5054, and the WGAN with SL and ADMM give significantly lower values, with 0.0915 and 0.0758, respectively.

![](images/a7790f07c3908850b00135dc06a0639d96fc42e01feda62f75393ae98d171a59.jpg)

![](images/975b7e0378bd77399b846c6ae553ca203d31df9dd4d3c5a08539467b233cca4e.jpg)

![](images/4c1f56a14983bd4f686233cde747305ef279f60280d60dae403d3d069fb41d7d.jpg)

![](images/4a53193ae97df73fe71b2cbb9f6033ed8e19193436c0094bc1406772cb6bd80a.jpg)

![](images/1d9a0abc859527eb3a75010bb5d79e1873cbfb3b137893510fc4e4281546252f.jpg)  
Figure 4: Graph of the generated results and the transport distances. Top (left to right): generate samples from target distribution, WGAN, WGAN with SL, WGAN with ADMM; Bottom (left to right): transport cost during training with WGAN, WGAN with SL, WGAN with ADMM. For the last figure, orange line is for the first network and blue line is for the second network

![](images/4b53a8c380ee1aee6e8ba25ba0cbbb75a7e91be3dc8944e2dcce626096795709.jpg)

![](images/6c44a819dc69c22a6b6f108b009e7f924399a22dcdc921f61c541efd7f3fa676.jpg)

MNIST. We also train the WGANs on the MNIST dataset. By using WGAN with  $\mathrm{SL}(\lambda = 1)$ , we obtain a transport cost around 1.81 compared to 1.87 with only WGAN. MNIST like samples generated by the learned optimal is plotted in Figure 5. Therefore, our method lowers the transport cost of WGAN while keeping the quality of the generated distributions.

# 5 CONCLUSION

We have shown that the incorporation of constraint optimization tools provides a direct and efficient way for computing optimal transport maps. By solving the Monge problem directly, our method avoids using special network structures or solving the dual problem. Moreover, applying our method to WGAN shows a lower transport cost for the generative networks without sacrificing the quality of the generated data.

![](images/12587fad48e33d5411d76e412495ef27adccfaf93fb71bab92c9d7e242eaa378.jpg)  
Figure 5: MNIST like samples generated by WGAN with SL penalty

# REFERENCES

Brandon Amos, Samuel Cohen, Giulia Luise, and Ievgen Redko. Meta optimal transport. arXiv preprint arXiv:2206.05262, 2022.  
Sigurd Angenent, Steven Haker, and Allen Tannenbaum. Minimizing flows for the Monge-Kantorovich problem. SIAM Journal on Mathematical Analysis, 35(1):61-97, 2003.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International Conference on Machine Learning, pp. 214-223. PMLR, 2017.  
Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem. Numerische Mathematik, 84(3):375-393, 2000.  
Dimitri P Bertsekas. Constrained optimization and Lagrange multiplier methods. Academic press, 2014.  
Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, Jonathan Eckstein, et al. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1):1-122, 2011.  
Yann Brenier. Polar factorization and monotone rearrangement of vector-valued functions. Communications on Pure and Applied Mathematics, 44(4):375-417, 1991.  
Charlotte Bunne, Laetitia Papaxanthos, Andreas Krause, and Marco Cuturi. Proximal optimal transport modeling of population dynamics. In International Conference on Artificial Intelligence and Statistics, pp. 6511-6528. PMLR, 2022.  
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in Neural Information Processing Systems, 26, 2013.  
Tarek A El Moselhy and Youssef M Marzouk. Bayesian inference with optimal maps. Journal of Computational Physics, 231(23):7815-7850, 2012.  
Aude Geneva, Gabriel Peyre, and Marco Cuturi. Learning generative models with sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics, pp. 1608-1617. PMLR, 2018.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. The Journal of Machine Learning Research, 13(1):723-773, 2012.  
Eldad Haber, Tauseef Rehman, and Allen Tannenbaum. An efficient numerical method for the solution of the 1-2 optimal mass transfer problem. SIAM Journal on Scientific Computing, 32(1): 197-211, 2010.  
Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, and Pieter Abbeel. Flow++: Improving flow-based generative models with variational dequantization and architecture design. In International Conference on Machine Learning, pp. 2722-2730. PMLR, 2019.  
Leonid V Kantorovich. On the translocation of masses. In Dokl. Akad. Nauk. USSR (NS), volume 37, pp. 199-201, 1942.

Patrick Kidger and Terry Lyons. Universal approximation with deep narrow networks. In Conference on learning theory, pp. 2306-2327. PMLR, 2020.  
Wuchen Li, Penghang Yin, and Stanley Osher. Computations of optimal transport distance with fisher information regularization. Journal of Scientific Computing, 75(3):1581-1595, 2018.  
Ashok Makkuva, Amirhossein Taghvaei, Sewoong Oh, and Jason Lee. Optimal transport mapping via input convex neural networks. In International Conference on Machine Learning, pp. 6672-6681. PMLR, 2020.  
Gaspard Monge. Mémoire sur la théorie des déblais et des remblais. Mem. Math. Phys. Acad. Royale Sci., pp. 666-704, 1781.  
Alfred Müller. Integral probability metrics and their generating classes of functions. Advances in Applied Probability, 29(2):429-443, 1997.  
Jorge Nocedal and Stephen J Wright. Numerical optimization. Springer, 1999.  
Ingram Olkin and Friedrich Pukelsheim. The distance between two random vectors with given dispersion matrices. Linear Algebra and its Applications, 48:257-263, 1982.  
Gabriel Peyre, Marco Cuturi, et al. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355-607, 2019.  
Maziar Sanjabi, Jimmy Ba, Meisam Razaviyayn, and Jason D Lee. On the convergence and robustness of training GANs with regularized optimal transport. Advances in Neural Information Processing Systems, 31, 2018.  
Vivien Seguy, Bharath Bhushan Damodaran, Rémi Flamary, Nicolas Courty, Antoine Rolet, and Mathieu Blondel. Large-scale optimal transport and mapping estimation. arXiv preprint arXiv:1711.02283, 2017.  
Cédric Villani. Optimal transport: old and new, volume 338. Springer, 2009.  
Yu Wang, Wotao Yin, and Jinshan Zeng. Global convergence of ADMM in nonconvex nonsmooth optimization. Journal of Scientific Computing, 78(1):29-63, 2019.
