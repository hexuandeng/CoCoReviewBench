# ESCAPING SADDLE POINT EFFICIENTLY IN MINIMAX AND BILEVEL OPTIMIZATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Hierarchical optimization (including minimax optimization and bilevel optimization) is attracting significant attention as it can be broadly applied to many machine learning tasks such as adversarial training, policy optimization, meta-learning and hyperparameter optimization. Recently, many algorithms have been studied to improve the theoretical analysis results of minimax and bilevel optimizations. Among these works, one of the most crucial issues is to escape saddle point and find local minimum, which is also of importance in conventional nonconvex optimization. In this paper, thus, we focus on investigating the methods to achieve second-order stationary point for nonconvex-strongly-concave minimax optimization and nonconvex-strongly-convex bilevel optimization. Specifically, we propose a new algorithm named PRGDA via perturbed stochastic gradient which does not require the computation of second order derivatives. In stochastic nonconvex-strongly-concave minimax optimization, we prove that our algorithm can find an  $O(\epsilon, \sqrt{\rho_{\Phi}\epsilon})$  second-order stationary point within gradient complexity of  $\tilde{O}(\kappa^3\epsilon^{-3})$ , which matches state-of-the-art to find first-order stationary point. To our best knowledge, our algorithm is the first stochastic algorithm that is guaranteed to obtain the second-order stationary point for nonconvex minimax problems. Besides, in stochastic nonconvex-strongly-convex bilevel optimization, our method also achieves better gradient complexity of  $Gc(f,\epsilon) = \tilde{O}(\kappa^3\epsilon^{-3})$  and  $Gc(g,\epsilon) = \tilde{O}(\kappa^7\epsilon^{-3})$  to find local minimum. Finally, we conduct a numerical experiment to validate the performance of our new method.

# 1 INTRODUCTION

Hierarchical optimization (including minimax and bilevel optimization) is a popular and important optimization framework which has been applied to a wide range of machine learning problems, such as Generative Adversarial Net (Goodfellow et al. (2014)), adversarial training (Madry et al. (2018)), multi-agent reinforcement learning (Wai et al. (2018)), meta-learning (Franceschi et al. (2018); Bertinetto et al. (2018)) and hyperparameter optimization (Shaban et al. (2019); Feurer & Hutter (2019)). In this paper, we study the following stochastic hierarchical optimization problem

$$
\min  _ {x \in \mathbb {R} ^ {d _ {1}}} \Phi (x) := f (x, y ^ {*} (x)) = \mathbb {E} _ {\xi \in \mathcal {D}} [ F (x, y ^ {*} (x); \xi) ] \tag {1}
$$

$$
\mathrm {s . t .} y ^ {*} (x) = \arg \min  _ {y \in \mathbb {R} ^ {d _ {2}}} g (x, y) = \mathbb {E} _ {\zeta \in \mathcal {D} ^ {\prime}} [ G (x, y; \zeta) ],
$$

where the upper-level function  $f(x, y^*(x)) = \mathbb{E}_{\xi \in \mathcal{D}}[F(x, y^*(x); \xi)]$  is smooth and possibly nonconvex, and the lower-level function  $g(x, y) = \mathbb{E}_{\zeta \in \mathcal{D}'}[G(x, y; \zeta)]$  is smooth and strongly-convex in variable  $y$  so that  $y^*(x)$  and  $\Phi(x)$  can be well defined.  $\xi$  and  $\zeta$  are samples drawn from data distribution  $\mathcal{D}$  and  $\mathcal{D}'$ . Stochastic problem is a general form that covers a couple of optimization tasks, including online optimization and finite-sum optimization. When  $g(x, y) = -f(x, y)$ ,  $\xi = \zeta$  and  $\mathcal{D} = \mathcal{D}'$ , the above hierarchical optimization (i.e., bilevel optimization) is reduced to a standard minimax optimization which can be rewritten as Eq. (2)

$$
\min  _ {x \in \mathbb {R} ^ {d _ {1}}} \max  _ {y \in \mathcal {Y} \subseteq \mathbb {R} ^ {d _ {2}}} f (x, y) = \mathbb {E} _ {\xi \in \mathcal {D}} [ F (x, y; \xi) ] \tag {2}
$$

where  $\mathcal{V}$  is a convex domain (not required to be compact). The loss function  $f(x,y)$  is smooth and possibly nonconvex w.r.t.  $x$ , and is smooth and strongly-concave w.r.t.  $y$ .

# 1.1 MINIMAX OPTIMIZATION

Recently, there are plenty of works studying minimax optimization problem in a variety of research fields in machine learning. Many deterministic and stochastic algorithms with asymptotic or non-asymptotic convergence analysis have been developed, such as Gradient Descent Ascent (GDA)

Table 1: Comparison of properties between related algorithms for minimax optimization.  

<table><tr><td>Name</td><td>Reference</td><td>Stochastic</td><td>Local Minimum</td><td>Pure First-Order</td></tr><tr><td>SGDA</td><td>(Lin et al. (2020a))</td><td>✓</td><td>×</td><td>✓</td></tr><tr><td>Cubic-GDA</td><td>(Chen et al. (2021b))</td><td>×</td><td>✓</td><td>×</td></tr><tr><td>MCN</td><td>(Luo &amp; Chen (2021))</td><td>×</td><td>✓</td><td>×</td></tr><tr><td>Perturbed GDmax</td><td>(Huang et al. (2022b))</td><td>×</td><td>✓</td><td>✓</td></tr><tr><td>PRGDA</td><td>(ours)</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

(Du & Hu (2019); Nemirovski (2004)) and Stochastic Gradient Descent Ascent (SGDA) (Lin et al. (2020a)). Some algorithms adopt a single loop structure (Heusel et al. (2017); Lin et al. (2020a); Xu et al. (2020)) while the others use a nested loop to update  $y$  more frequently so that they can obtain a better estimation of the maximum  $y^{*}(x)$  (Jin et al. (2019); Nouiehed et al. (2019)). Besides, some algorithms have been proposed to improve the theoretical results of minimax optimization, such as SREDA (Luo et al. (2020)) and Acc-MDA (Huang et al. (2022a)) which take advantage of variance reduction to accelerate the convergence rate and reduce the gradient complexity. Moreover, on deterministic setting some recently proposed algorithms (Lin et al. (2020b)) have already matched the optimal lower bound (Zhang et al. (2021)).

However, most of these works only consider the criterion of finding first-order stationary point. In nonconvex setting, convergence to first-order stationary point is not always satisfactory because a first-order stationary point could be a local minimum, saddle point or even local maximum. Therefore, second-order stationary point that reaches local minimum becomes a popular and important issue in nonconvex optimization. Since finding global minimum in nonconvex optimization is usually an NP-hard problem (Hillar & Lim (2013)), in some situations we attempt to find a local minimum instead. Moreover, in some machine learning tasks such as tensor decomposition (Ge et al. (2015)), matrix sensing (Bhojanapalli et al. (2016); Park et al. (2017)), and matrix completion (Ge et al. (2016)), finding local minimum is equivalent to finding global minimum, which makes second-order stationary point more crucial.

Therefore, we are motivated to study the method that obtains second-order stationary point for minmax (and bilevel) optimization which captures local minimum and escapes saddle point of  $\Phi(x)$ . In section 3 we can see that under certain conditions the objective function  $\Phi(x)$  is twice differentiable and  $\nabla^2\Phi(x)$  is Lipschitz continuous. An  $O(\epsilon, \epsilon_H)$  second-order stationary point satisfies  $\|\nabla\Phi(x)\| \leq O(\epsilon)$  and  $\lambda_{min}(\nabla^2\Phi(x)) \geq -\epsilon_H$  where  $\lambda_{min}(\cdot)$  means the smallest eigenvalue.

Although several recent works have been proposed to study the second-order stationary point for nonconvex-strongly-concave minimax optimization based on cubic-regularized gradient descent ascent (Chen et al. (2021b); Luo & Chen (2021)) or perturbed gradient (Huang et al. (2022b)), they are only adaptive to deterministic gradient oracle and finite-sum problem. The study of the second-order stationary point for stochastic nonconvex minimax problem where the full gradient is not available is still limited. A comparison of properties between related works for minimax optimization is demonstrated in Table 1.

Thus, to fill this gap, we propose a new algorithm named Perturbed Recursive Gradient Descent Ascent (PRGDA) to search second-order stationary point for stochastic nonconvex problem (2). To our best knowledge, PRGDA is the first algorithm that is guaranteed to obtain second-order stationary point for stochastic nonconvex minimax optimization problems. Furthermore, our method is a pure first-order algorithm that only requires the computation of gradient oracle. Neither Hessian matrix nor Hessian vector product is required, which makes our method more efficient to implement. We will also provide the analysis results to show that the gradient complexity of our algorithm is  $\tilde{O} (\kappa^3\epsilon^{-3})$  to achieve  $O(\epsilon ,\sqrt{\rho_{\Phi}\epsilon})$  second-order stationary point where  $\kappa$  is the condition number and  $\rho_{\Phi}$  is the Lipschitz constant of  $\nabla^2\Phi (x,y)$  (defined in section 3), which matches the best result of finding first-order stationary point for the same minimax optimization problem.

# 1.2 BILEVEL OPTIMIZATION

Recently, many algorithms have been studied to solve bilevel optimization. Some optimization algorithms are deterministic such as AID-BiO and ITD-BiO (Ji et al. (2021)) while the others consider stochastic algorithms including BSA (Ghadimi & Wang (2018)), TTSA (Hong et al. (2020)) and StocBiO (Ji et al. (2021)). These methods are proposed to improve the convergence analysis of bilevel optimization since most earlier works (Domke (2012); Pedregosa (2016)) only provide the asymptotic convergence analysis without specific convergence rates.

Table 2: Comparison of complexity between related algorithms for bilevel optimization. We use  $p(\kappa)$  for some algorithms that do not provide the explicit dependence on  $\kappa$ .  

<table><tr><td>Name</td><td>Reference</td><td>Gc(f,ε)</td><td>Gc(g,ε)</td><td>Local Minimum</td></tr><tr><td>StocBiO</td><td>(Ji et al. (2021))</td><td>O(κ5ε-4)</td><td>O(κ9ε-4)</td><td>×</td></tr><tr><td>SUSTAIN</td><td>(Khanduri et al. (2021))</td><td>O(p(κ)ε-3)</td><td>O(p(κ)ε-3)</td><td>×</td></tr><tr><td>MRBO/VRBO</td><td>(Yang et al. (2021))</td><td>O(p(κ)ε-3)</td><td>O(p(κ)ε-3)</td><td>×</td></tr><tr><td>StocBiO + iNEON</td><td>(Huang et al. (2022b))</td><td>O(κ5ε-4)</td><td>O(κ10ε-4)</td><td>√</td></tr><tr><td>PRGDA</td><td>(ours)</td><td>O(κ3ε-3)</td><td>O(κ7ε-3)</td><td>√</td></tr></table>

StocBiO algorithm (Ji et al. (2021)) is a recent work to solve stochastic nonconvex-strongly-convex bilevel optimization via AID. In this paper, we also study the convergence of our method under this condition where  $\Phi(x)$  is stochastic and probably nonconvex. According to previous studies of bilevel optimization, when  $f(x,y)$  and  $g(x,y)$  are differentiable and  $g(x,y)$  is strongly-convex with respect to  $y$ ,  $\Phi(x)$  is also differentiable and automatically  $\|\nabla \Phi(x)\| \leq \epsilon$  is a criterion of first-order stationary point. Notice that in (Ji et al. (2021))  $\|\nabla \Phi(x)\|^2 \leq \epsilon$  is used as the criterion. In this paper, we will uniformly adopt  $\|\nabla \Phi(x)\| \leq \epsilon$  as the convergence criterion. More recently, many stochastic algorithms with variance reduction are proposed, such as RSVRB (Guo & Yang (2021)), SUSTAIN (Khanduri et al. (2021)), MRBO and VRBO (Yang et al. (2021)). The gradient complexity of bilevel optimization is enhanced to  $O(\epsilon^{-3})$ , which is the best theoretical result as far as we know. StocBiO with iNEON (Huang et al. (2022b)) is another recent work that combines StocBiO algorithm with pure first-order method inexact negative curvature originated from noise (iNEON) to escape saddle point and find second-order stationary point for nonconvex-strongly-convex bilevel optimization.

Although these works are proposed to improve the performance of algorithms for bilevel optimization, the complexity of current methods that achieve second-order stationary point are still high. Actually, the complexity of StocBiO with iNEON is even higher than the standard StocBiO algorithm in order to find a local minimum with high probability. Thus, to fill these gap, we are motivated to propose an accelerated algorithm with variance reduction that requires lower complexity to find second-order stationary point for stochastic nonconvex-strongly-convex bilevel optimization.

The comparison of gradient complexity between our method and related works to find  $O(\epsilon)$  first-order stationary point or  $O(\epsilon, \sqrt{\rho_{\Phi}\epsilon})$  second-order stationary point is shown in Table 2. In Table 2,  $Gc(f,\epsilon)$  and  $Gc(g,\epsilon)$  are the numbers of gradient evaluations of function  $f(x,y)$  and  $g(x,y)$  respectively. The last column represents whether the algorithm is able to escape saddle point and find local minimum. Notation  $\tilde{O}$  hides the logarithm term. StocBiO with iNEON and our PRGDA algorithm involve a logarithm term in the complexity because they converge to second-order stationary point with high probability, considering all randomness including the stochastic gradient while other algorithms only consider the expectation over stochastic gradients. From Table 2 we can see our PRGDA algorithm improves the gradient complexity  $Gc(f,\epsilon)$  and  $Gc(g,\epsilon)$  of StocBiO with iNEON algorithm significantly and matches state-of-the-art complexity  $O(\epsilon^{-3})$ , which is one of the most important contribution of this paper.

# 1.3 CONTRIBUTIONS

We summarize our main contributions as follows:

- We propose a new PRGDA algorithm which is the first algorithm to reach second-order stationary point for stochastic nonconvex minimax optimization problem. Our method is pure first-order and does not require any calculation of second-order derivatives. Our method does not involve nested loops either, which makes it more efficient to implement.  
- We prove that the gradient complexity of our algorithm is  $\tilde{O}(\kappa^3\epsilon^{-3})$  to achieve  $O(\epsilon, \sqrt{\epsilon})$  second-order stationary point in stochastic nonconvex minimax optimization, which matches the best result of finding first-order stationary point in the same problem.  
- Our PRGDA algorithm can also be applied to nonconvex bilevel optimization and we can prove that the gradient complexity is  $Gc(f, \epsilon) = \tilde{O}(\kappa^3 \epsilon^{-3})$  and  $Gc(g, \epsilon) = \tilde{O}(\kappa^7 \epsilon^{-3})$  to find  $O(\epsilon, \sqrt{\epsilon})$  second-order stationary point in stochastic nonconvex bilevel optimization, which outperforms the previous best theoretical results and matches state-of-the-art to find first-order stationary point.

# 2 RELATED WORK

In this section we will summarize the background of related works and some details of methods that are important to our work will be further discussed in the Appendix.

# 2.1 STOCHASTIC MINIMAX OPTIMIZATION

Recently many algorithms for solving stochastic minimax optimization were proposed, and the majority of them were studied under the nonconvex-strongly-concave condition, including intuitive methods SGDmax (Jin et al. (2019)) and Stochastic Gradient Descent Ascent (SGDA) (Lin et al. (2020a)). More recently in (Yang et al. (2022)), a new method Stoc-Smoothed-AGDA is proposed to achieve better complexity with a weaker PL condition instead of strong concavity. Besides, some methods integrate variance reduction with a minimax problem to accelerate the convergence, such as Stochastic Recursive gradiEnt Descent Ascent (SREDA) (Luo et al. (2020)), Hybrid Variance-Reduced SGD (Tran-Dinh et al. (2020)) and Acc-MDA (Huang et al. (2022a)). There are also some works that study the weakly-convex concave minimax optimization such as (Rafique et al. (2021)) and (Yan et al. (2020)). More related to this work, Cubic-Regularized Gradient Descent Ascent (Cubic-GDA) (Chen et al. (2021b)) and Minimax Cubic Newton (MCN) (Luo & Chen (2021)) are two recent algorithms that can reach the second-order stationary point in nonconvex-strongly-concave minimax optimization.

# 2.2 PERTURBED GRADIENT DESCENT

Perturbed Gradient Descent (PGD) (Jin et al. (2017)) was proposed to find second-order stationary point for nonconvex optimization which introduces a perturbation under specific condition. It is a deterministic gradient based algorithm and only involves first-order oracle. To extend Perturbed Gradient Descent to the stochastic setting and incorporate it with variance reduction, SSRGD Li (2019) was proposed to reach second-order stationary point with SFO of  $O(\epsilon^{-3.5})$ . After that Pullback algorithm (Chen et al. (2021a)) was proposed to improve the complexity to  $O(\epsilon^{-3})$ .

# 2.3 STOCBIO WITH INEON

In (Huang et al. (2022b)), algorithms for both minimax and bilevel optimization are proposed to find second-order stationary point. However, for minimax optimization only the deterministic problem is studied. In the proposed Perturbed GDmax algorithm, perturbed gradient descent is used to solve the issue in this case. As we have mentioned, perturbed gradient descent in deterministic and stochastic are totally different. Therefore, it is essential to investigate the stochastic minimax optimization algorithm that converge to second-order stationary point. For bilevel optimization, the stochastic problem is considered and the StocBiO with iNEON algorithm is proposed. The algorithm is inspired by NEON (Xu et al. (2018); Allen-Zhu & Li (2018)), which is a method to find local minimum merely based on first-order oracles. Inexact NEON is a variant of NEON since the exact gradient in bilevel optimization is unavailable. However, it requires an extra nested loop to solve a subproblem that extracts a negative curvature descent direction. Besides, the gradient complexity of StocBiO with iNEON is also higher than the vanilla StocBiO. Therefore, we are motivated to propose a more efficient bilevel optimization algorithm to find second-order stationary point.

# 3 PRELIMINARY

In this section we will present the notations used in this paper and introduce some basic assumptions to further illustrate the problem setting. We assume that upper-level function  $f(x,y)$  is twice differentiable. Lower-level  $g(x,y)$  is three times differentiable (only required in bilevel optimization). The partial derivative is denoted by  $\nabla_{x}$  and  $\nabla_{y}$ , e.g.,  $\nabla f(x,y) = [\nabla_{x}f(x,y),\nabla_{y}f(x,y)]$ . Similarly,  $\nabla_{x}^{2}$  and  $\nabla_{y}^{2}$  represent the Hessian.  $\nabla_{xy}^{2}$  and  $\nabla_{yx}^{2}$  represent the Jacobian. We use  $\| \cdot \|_2$  and  $\| \cdot \|_F$  to denote the spectral norm and Frobenius norm of matrix respectively. Notation  $\tilde{O}(\cdot)$  means the complexity after hiding logarithm terms. First, we assume that lower-level function  $g(x,y)$  is strongly-convex with respect to  $y$  so that  $y^*(x)$  and  $\Phi(x)$  can be well defined.

Assumption 1. The lower-level function  $g(x,y)$  is  $\mu$ -strongly-convex with respect to  $y$ , i.e., there exists a constant  $\mu$  such that

$$
g (x, y) + \left\langle \nabla_ {y} g (x, y), y ^ {\prime} - y \right\rangle + \frac {\mu}{2} \| y ^ {\prime} - y \| ^ {2} \leq g (x, y ^ {\prime}) \tag {3}
$$

for any  $x, y$  and  $y'$ .

Notice that in minimax optimization  $g(x,y)$  is the same as  $-f(x,y)$  so we merge these two cases into one statement. With Assumption 1, objective function  $\Phi (x)$  is also differentiable and the gradient is formulated as follows (Ji et al. (2021)).

$$
\nabla \Phi (x) = \nabla_ {x} f (x, y ^ {*} (x)) - \nabla_ {x y} ^ {2} g (x, y ^ {*} (x)) [ \nabla_ {y} ^ {2} g (x, y ^ {*} (x)) ] ^ {- 1} \nabla_ {y} f (x, y ^ {*} (x)) \tag {4}
$$

In minimax optimization, since we always have  $\nabla_y f(x,y^* (x)) = 0$ , the expression of  $\nabla \Phi (x)$  is simplified by  $\nabla \Phi (x) = \nabla_{y}f(x,y^{*}(x))$  (5)

$$
\nabla \Phi (x) = \nabla_ {x} f (x, y ^ {*} (x)) \tag {5}
$$

Next, we introduce the following assumptions about Lipschitz continuity of first and second order derivatives. These assumptions are commonly used in the convergence analysis of minimax and bilevel optimization (Luo et al. (2020); Luo & Chen (2021); Ji et al. (2021); Huang et al. (2022b)).

Assumption 2. The gradients of component functions  $F(x,y;\xi)$  and  $G(x,y;\zeta)$  are  $L$ -Lipschitz continuous, i.e., there exists a constant  $L$  such that

$$
\left\| \nabla F (z; \xi) - \nabla F \left(z ^ {\prime}; \xi\right) \right\| \leq L \| z - z ^ {\prime} \|, \left\| \nabla G (z; \zeta) - \nabla G \left(z ^ {\prime}; \zeta\right) \right\| \leq L \| z - z ^ {\prime} \| \tag {6}
$$

for any  $z = (x,y)$  and  $z^{\prime} = (x^{\prime},y^{\prime})$

Assumption 3. The second order derivatives  $\nabla_x^2 f(x,y)$ ,  $\nabla_{xy}^2 f(x,y)$ ,  $\nabla_y^2 f(x,y)$ ,  $\nabla_{xy}^2 g(x,y)$  and  $\nabla_y^2 g(x,y)$  are  $\rho$ -Lipschitz continuous.

The condition number  $\kappa$  of the hierarchical optimization problem is defined by  $\kappa = L / \mu$ . According to previous works, in minimax optimization under Assumptions 1, 2 and 3,  $\Phi(x)$  is twice differentiable.  $y^{*}(x)$  is  $\kappa$ -Lipschitz continuous,  $\nabla \Phi(x)$  is  $L_{\Phi}$ -Lipschitz continuous and  $\nabla^2 \Phi(x)$  is  $\rho_{\Phi}$ -Lipschitz continuous, which is shown in the Appendix.

According to (Ghadimi & Wang (2018); Ji et al. (2021)), we know in bilevel optimization function  $y^{*}(x)$  is also  $\kappa$ -Lipschitz continuous, but we need an additional Assumptions 4 to guarantee  $\Phi(x)$  has  $L_{\Phi}$ -Lipschitz gradient, which is described in the Appendix.

Assumption 4. The upper-level function  $f(x,y)$  is  $M$ -Lipschitz continuous, i.e., there exists a constant  $M$  such that

$$
\left\| f (z) - f \left(z ^ {\prime}\right) \right\| \leq M \| z - z ^ {\prime} \| \tag {7}
$$

for any  $z = (x,y)$  and  $z^{\prime} = (x^{\prime},y^{\prime})$

Since in this paper we study the convergence to second-order stationary point, we also need the following Assumption 5 which is also assumed in (Huang et al. (2022b)) that makes function  $\Phi(x)$  twice differentiable and have  $\rho_{\Phi}$ -Lipschitz Hessian. We should notice that Assumption 4 and 5 are only used for bilevel optimization.

Assumption 5. The third order derivatives  $\nabla_{xyx}^{3}g$ ,  $\nabla_{yxy}^{3}g$  and  $\nabla_y^3 g$  are  $\nu$ -Lipschitz continuous.

# 4 PROPOSED ALGORITHM FOR MINIMAX OPTIMIZATION

In this section, we will propose our PRGDA algorithm for the special case of minimax optimization. The description of our PRGDA algorithm is demonstrated in Algorithm 1. Similar to SREDA, the initial value  $y_0$  is also yield by PiSARAH algorithm to make it close to  $y^{*}(x_{0})$ , which is a conventional strongly-convex optimization subproblem. In our convergence analysis this step costs the gradient complexity of  $\tilde{O}(\kappa^2\epsilon^{-2})$ . We use  $v_t$  and  $u_t$  to represent the gradient estimator of  $\nabla_x f(x_t,y_t)$  and  $\nabla_y f(x_t,y_t)$  respectively. In each iteration,  $y_{t+1}, v_t$  and  $u_t$  are computed by an inner loop updater with  $K$  iterations, which is shown in Algorithm 2. In Algorithm 2, we use the SPIDER gradient estimator to update  $y_{t,k}, v_{t,k}$  and  $u_{t,k}$ .  $S_1$  is the large batchsize that is loaded every  $q$  iterations of  $t$ .  $S_2$  is the small batchsize.  $\lambda$  is the stepsize to update variable  $y$ . The output of the inner loop updater depends on the minimum value of the norm of  $\tilde{\mathcal{G}}_{\lambda}(y_{t,k})$  and its corresponding index, which is defined by  $\tilde{\mathcal{G}}_{\lambda}(y_{t,k}) = (y_{t,k} - \Pi_{\mathcal{V}}(y_{t,k} + \lambda u_{t,k})) / \lambda$ . We will show that gradient estimator  $v_t$  satisfies  $\| v_t - \nabla \Phi(x_t)\| \leq O(\epsilon)$  based on this inner loop updater.

Inspired by perturbed gradient descent, our PRGDA is also composed of a descent phase and an escaping phase. In the descent phase our PRGDA algorithm follows the iterative update rule of SPIDER that  $x_{t + 1} = x_t - (\eta / \| v_t \|) v_t$  until the norm of  $v_t$  satisfies  $\| v_t \| \leq O(\epsilon)$ . After the descent phase is terminated, we use  $m_s$  to denote the current counter  $t$  and uniformly draw a perturbation  $\xi$  from ball  $B_0(r)$  where parameter  $r$  is the perturbation radius. We add the perturbation to the current status  $x_t$  and start the escaping phase. In the escaping phase, parameter  $t_{thres}$  is maximum number of iterations of the phase and  $\bar{D}$  is the average moving distance which is used to determine if the escaping phase should be stopped. The stepsize of  $x$  in this phase is denoted by  $\eta_H$  which is typically larger than  $\eta$  in the descent phase. We use  $D$  to denote the accumulated squared moving distance. If the averaged squared moving distance is larger than  $\bar{D}$  then we pull it back (line 17 in Algorithm 1) and break the escaping phase. In this case we consider  $x_{m_s}$  as a saddle point and continue to run next descent phase. Otherwise, if the escaping phase is not broken after  $t_{thres}$  iterations, we claim that  $x_{m_s}$  is a second-order stationary point with high probability. This is because when  $\lambda_{min}(\nabla^2 \Phi(x_{m_s})) < -\epsilon_H$ , the stuck region  $S$  defined by the area  $\{\xi \in B_0(r) | \text{the sequence started from } x_{m_s + 1} = x_{m_s} + \xi \text{ does not break the escaping phase}\}$  has a small volume. Specifically, similar to Lemma 6 in (Li (2019)) and Lemma D.3 in (Chen et al.

Algorithm 1 Perturbed Recursive Gradient Descent Ascent  
Input: initial value  $x_0, y_0$   
Parameter: stepsize  $\eta$  and  $\eta_H$ , perturbation radius  $r$ , escaping phase threshold  $t_{thres}$ , average movement  $\bar{D}$ , tolerance  $\epsilon$ , maximum iteration  $T$ .  
1: Set escape = false,  $s = 0$ , esc = 0.  
2: for  $t = 0, 1, \ldots, T - 1$  do  
3: Update  $y_{t+1}, v_t, u_t$  from Algorithm 2 (Minimax) or Algorithm 3 (Bilevel).  
4: if escape = false then  
5: if  $\|v_t\| \geq \epsilon$  then  
6: Update  $x_{t+1} = x_t - (\eta / \|v_t\|) v_t$ .  
7: else  
8: Let  $m_s = t$ ,  $s = s + 1$ , escape = true, esc = 0.  
9: Draw perturbation  $\xi \sim B_0(r)$  and update  $x_{t+1} = x_t + \xi$ .  
10: end if  
11: else  
12: Compute  $D = \sum_{j=m_s+1}^{t} \eta_H^2 \|v_j\|^2$ .  
13: if  $D > (t - m_s) \bar{D}$  then  
14: Set  $\eta_t$  s.t.  $\sum_{j=m_s+1}^{t} \eta_j^2 \|v_j\|^2 = (t - m_s) \bar{D}$ .  
15: Update  $x_{t+1} = x_t - \eta_t v_t$ . Set escape = false.  
16: else  
17: Set  $\eta_t = \eta_H$ . Update  $x_{t+1} = x_t - \eta_t v_t$ , esc = esc + 1.  
18: Return  $x_{m_s}$  if esc =  $t_{thres}$ .  
19: end if  
20: end if  
21: end for  
Output:  $x_{m_s}$

(2021a)), we can prove if we suppose after the perturbation there are two coupled sequences started from two points  $x_{m_s + 1}$  and  $x_{m_s + 1}'$  respectively within a small distance  $\| x_{m_s + 1} - x_{m_s + 1}' \| = r_0$  in the smallest eigenvector direction of Hessian matrix  $\nabla^2 \Phi(x_{m_s})$ , then there must be at least one sequence  $\{x_{m_s + 1}\}$  or  $\{x_{m_s + 1}'\}$  that breaks the escaping phase. Informally, this means the stuck region  $S$  must be contained in a "narrow band" or "thin disk" in a high dimensional space which cannot have a large measure. Since the perturbation  $\xi$  is uniformly generated from ball  $B_0(r)$ , the probability that  $\xi$  belongs to the stuck region is low.

# 5 PROPOSED ALGORITHM FOR BILEVEL OPTIMIZATION

In this section we propose our PRGDA algorithm to solve the more general bilevel optimization. Actually, we only need to switch the inner loop updater in Algorithm 2 to the bilevel mode, which is demonstrated in Algorithm 3 in Appendix. Similar to the case of minimax optimization, here we also need a initialization algorithm to initialize  $y_0$  with the cost of  $Gc(g,\epsilon) = \tilde{O}(\kappa^6\epsilon^{-2})$  in the convergence analysis. Next we will elaborate the inner loop updater for bilevel optimization. We also use the update rule of SPIDER to compute  $v_{t,k}^{(1)}$ ,  $v_{t,k}^{(2)}$  and  $u_{t,k}$ , which represent the estimator of  $\nabla_x f(x,y)$ ,  $\nabla_y f(x,y)$  and  $\nabla_y g(x,y)$  respectively. We should notice that the large and small batchsize of computing  $u_{t,k}$  are different from that of  $v_{t,k}^{(1)}$  or  $v_{t,k}^{(2)}$ . After the inner loop to compute  $y_{t+1}$ , we calculate the Jacobian  $J_t$  with a batch of size  $S_5$ . Then we compute  $v_t$ , the estimator of  $\nabla \Phi(x)$  via AID. Here we follow the method used in StocBiO, which is

$$
z _ {t} ^ {Q} = \alpha \sum_ {q = - 1} ^ {Q - 1} \prod_ {j = Q - q} ^ {Q} \left(I - \alpha \nabla_ {y} ^ {2} G \left(x _ {t}, y _ {t + 1}; \mathcal {B} _ {j}\right)\right) v _ {t} ^ {(2)}, v _ {t} = v _ {t} ^ {(1)} - J _ {t} z _ {t} ^ {Q} \tag {8}
$$

where  $\mathcal{B}_j$  is the set of samples to calculate the stochastic estimator of Hessian  $\nabla_y^2 g(x_t,y_{t + 1})$ .

# 6 CONVERGENCE ANALYSIS

In this section we will illustrate the main theorem and provide the convergence analysis of our algorithm. First, we need to assume that  $\Phi(x)$  is lower bounded by  $\Phi^{*}$ . Then we will present the main theorems of our PRGDA algorithm. In this paper, we set  $\epsilon_{H} = \sqrt{\rho_{\Phi}\epsilon}$  as the tolerance of the second-order stationary point. We leave the proof of Theorem 1 and 2 to the Appendix.

Algorithm 2 Updater of Inner Loop (Minimax)  
Input: status  $x_{t}, x_{t-1}, y_{t}, v_{t-1}, u_{t-1}$  and  $t$   
Parameter: stepsize  $\lambda$ , inner loop size  $K$ , batchsize  $S_{1}$  and  $S_{2}$ , period  $q$ .  
1: Set  $x_{t,-1} = x_{t-1}, x_{t,k} = x_{t}$  when  $k \geq 0$ ,  $y_{t,-1} = y_{t,0} = y_{t}$ .  
2: if mod  $(t,q) = 0$  then  
3: Draw  $S_{1}$  samples  $\{\xi_{1},\ldots,\xi_{S_{1}}\}$   
4: Compute  $v_{t,-1} = \frac{1}{S_{1}} \sum_{i=1}^{S_{1}} \nabla_{x} F(x_{t},y_{t};\xi_{i})$ ,  $u_{t,-1} = \frac{1}{S_{1}} \sum_{i=1}^{S_{1}} \nabla_{y} F(x_{t},y_{t};\xi_{i})$ .  
5: else  
6: Let  $v_{t,-1} = v_{t-1}$ ,  $u_{t,-1} = u_{t-1}$ .  
7: end if  
8: for  $k = 0$  to  $K - 1$  do  
9: Draw  $S_{2}$  samples  $\{\xi_{1},\ldots,\xi_{S_{2}}\}$   
10: Compute  $v_{t,k} = v_{t,k-1} + \frac{1}{S_{2}} \sum_{i=1}^{S_{2}} (\nabla_{x} F(x_{t,k},y_{t,k};\xi_{i}) - \nabla_{x} F(x_{t,k-1},y_{t,k-1};\xi_{i}))$   
11: Compute  $u_{t,k} = u_{t,k-1} + \frac{1}{S_{2}} \sum_{i=1}^{S_{2}} (\nabla_{y} F(x_{t,k},y_{t,k};\xi_{i}) - \nabla_{y} F(x_{t,k-1},y_{t,k-1};\xi_{i}))$   
12:  $y_{t,k+1} = \prod_{\mathcal{Y}} (y_{t,k} + \lambda u_{t,k})$ .  
13: end for  
14: Select  $s_t = \arg \min_k \| \tilde{\mathcal{G}}_\lambda(y_{t,k})\|$ . Let  $y_{t+1} = y_{t,s_t}$ ,  $v_t = v_{t,s_t}$ ,  $u_t = u_{t,s_t}$ .  
Output:  $y_{t+1}, v_t, u_t$ .

# 6.1 MAIN THEOREM FOR MINIMAX OPTIMIZATION

Theorem 1. Under Assumption 1, 2 and 3, we set stepsize  $\eta = \tilde{O}(\frac{\epsilon}{\kappa L})$ ,  $\eta_H = \tilde{O}(\frac{1}{\kappa L})$  and  $\lambda = O(\frac{1}{L})$ , batchsize  $S_1 = \tilde{O}(\kappa^2 \epsilon^{-2})$  and  $S_2 = \tilde{O}(\kappa \epsilon^{-1})$ , period  $q = O(\epsilon^{-1})$ , inner loop  $K = O(\kappa)$ , perturbation radius  $r = \min \{\tilde{O}(\sqrt{\frac{\epsilon}{\kappa^3 \rho}}), \tilde{O}(\frac{\epsilon}{\kappa L})\}$ , threshold  $t_{thres} = \tilde{O}(\frac{L}{\sqrt{\kappa \rho \epsilon}})$  and average movement  $\bar{D} = \tilde{O}(\frac{\epsilon^2}{\kappa^2 L^2})$ . Then our PRGDA algorithm requires  $\tilde{O}(\kappa^3 \epsilon^{-3})$  SFO complexity to achieve  $O(\epsilon, \sqrt{\rho \Phi \epsilon})$  second-order stationary point with high probability.

# 6.2 MAIN THEOREM FOR BILEVEL OPTIMIZATION

Theorem 2. Under Assumption 1, 2, 3, 4 and 5, we set stepsize  $\eta = \tilde{O}(\frac{\epsilon}{\kappa^3 L})$ ,  $\eta_H = \tilde{O}(\frac{1}{\kappa^3 L})$ ,  $\lambda = O\left(\frac{1}{L}\right)$  and  $\alpha = O\left(\frac{1}{L}\right)$ , batchsize  $S_1 = \tilde{O}(\kappa^2 \epsilon^{-2})$ ,  $S_2 = \tilde{O}(\kappa^{-1} \epsilon^{-1})$ ,  $S_3 = \tilde{O}(\kappa^6 \epsilon^{-2})$ ,  $S_4 = \tilde{O}(\kappa^3 \epsilon^{-1})$ ,  $S_5 = \tilde{O}(\kappa^2 \epsilon^{-2})$  and  $B = \tilde{O}(\kappa^2 \epsilon^{-1})$ , period  $q = O(\kappa^2 \epsilon^{-1})$ , inner loop  $K = O(\kappa)$  and  $Q = \tilde{O}(\kappa)$ , perturbation radius  $r = \min\{\tilde{O}(\sqrt{\frac{\epsilon}{\rho_{\Phi}}}), \tilde{O}(\frac{\epsilon}{\kappa^3 L})\}$ , threshold  $t_{thres} = \tilde{O}\left(\frac{\kappa^3 L}{\sqrt{\rho_{\Phi} \epsilon}}\right)$  and average movement  $\bar{D} = \tilde{O}\left(\frac{\epsilon^2}{\kappa^6 L^2}\right)$ . Then our PRGDA algorithm requires complexity of  $Gc(f, \epsilon) = \tilde{O}(\kappa^3 \epsilon^{-3})$ ,  $Gc(g, \epsilon) = \tilde{O}(\kappa^7 \epsilon^{-3})$ ,  $JV(g, \epsilon) = \tilde{O}(\kappa^5 \epsilon^{-4})$  and  $HV(g, \epsilon) = \tilde{O}(\kappa^6 \epsilon^{-4})$  to achieve  $O(\epsilon, \sqrt{\rho_{\Phi} \epsilon})$  second-order stationary point with high probability.

# 7 EXPERIMENTS

In this section we conduct the matrix sensing (Bhojanapalli et al. (2016); Park et al. (2017)) experiment to validate the performance of out PRGDA algorithm for solving both minimax and bilevel problem. As a result of existing study on matrix sensing problem (Ge et al. (2017)), there is no spurious local minimum in this circumstance, i.e., every local minimum is a global minimum. Therefore, the capability of escaping saddle points of our algorithm can be verified by this experiment. We follow the experiment setup of (Chen et al. (2021a)) to recover a low-rank symmetric matrix  $M^{*} = U^{*}(U^{*})^{T}$  where  $U^{*} \in \mathbb{R}^{d \times r}$ . Suppose we have  $n$  sensing matrices  $\{A_{i}\}_{i=1}^{n}$  with  $n$  observations  $b_{i} = \langle A_{i}, M^{*} \rangle$ . Here the inner product of two matrices is defined by the trace  $\langle X, Y \rangle = tr(X^T Y)$ . Then the optimization problem can be defined by

$$
\min  _ {U \in \mathbb {R} ^ {d \times r}} \frac {1}{2} \sum_ {i = 1} ^ {n} L _ {i} (U), L _ {i} (U) = \left(\left\langle A _ {i}, U U ^ {T} \right\rangle - b _ {i}\right) ^ {2} \tag {9}
$$

The code of our algorithms is uploaded in the Supplementary Material.

# 7.1 ROBUST OPTIMIZATION

Similar to the problem setting of (Yan et al. (2019)), we also introduce another variable  $y$  and add a robust term to make the model robust. Therefore, the optimization problem can be formulated by

$$
\min  _ {U \in \mathbb {R} ^ {d \times r}} \max  _ {y \in \Delta_ {n}} f (U, y) = \frac {1}{2} \sum_ {i = 1} ^ {n} y _ {i} L _ {i} (U) - \left(y _ {i} - \frac {1}{n}\right) ^ {2} \tag {10}
$$

![](images/2236cdb7ec9fac66f10df219b1ae79d0fe2947265f611de456e89b8c476636de.jpg)

![](images/faf891d966ef29beeb77f390c8df5d7a8a7ddc0adc745950480b27f978c1cae0.jpg)

![](images/d48cd2bb069b1b066f9b1767164e561b452fdcd3edd0ac3ac624ba8f9d7e1e3c.jpg)

![](images/f12c27f687bae6a96c12cd0a1aaad1cb930e233d3b9c10650169c7da2eb4218d.jpg)  
Figure 1: Experimental results of our robust low-rank matrix sensing task. Figure (a) to (c) show the loss function value of  $\Phi(U)$  against the number of gradient oracles with  $d = 50$ ,  $d = 75$ , and  $d = 100$  respectively. Figure (d) to (f) show the ratio of distance  $\|UU^T - M^*\|_F^2 / \|M^*\|_F^2$  against the number of gradient oracles with  $d = 50$ ,  $d = 75$ , and  $d = 100$  respectively.

![](images/2476c4ac1bb97624d59693a61bc945876b4a77f35afed16cdef45b32613b9d22.jpg)

![](images/12c632d0b89db72889f91fc2eea3dfbf6444940ead1a13639e2cf0fb2518497e.jpg)

![](images/f43effcc0ab4373170934178601e9b284366f4b0c46f2e891d5b5358a4040f66.jpg)  
Figure 2: Experimental results of our hyper-representation learning of low-rank matrix sensing task. The ratio of distance  $\| UU^T - M^* \|_F^2 / \| M^* \|_F^2$  is shown against the number of gradient oracles with  $d = 50$ ,  $d = 75$ , and  $d = 100$  respectively.

![](images/859d7c7c9d7e54713b43647fc119352d9d49ef63949d5b93f7cd7df89efdb17a.jpg)

![](images/2abdfb6dca0bf38cb00f696209daae8664f19cde80a2dd3d8d7dfde2d05b92d5.jpg)

where  $\Delta_{n} = \{y\in \mathbb{R}^{n}|0\leq y_{i}\leq 1,\sum_{i = 1}^{n}y_{i} = 1\}$  is the simplex in  $\mathbb{R}^n$  and  $L_{i}(U)$  is defined in Eq. (9). The number of rows of matrix  $U$  is set to  $d = 50$ ,  $d = 75$  and  $d = 100$  respectively and the number of columns is fixed as  $r = 3$  in the main manuscript. The results of different ranks will be shown in the Appendix. The ground truth low-rank matrix  $M^{*}$  is generated by  $M^{*} = U^{*}(U^{*})^{T}$  where each entry of  $U^{*}$  is drawn from Gaussian distribution  $\mathcal{N}(0,1 / d)$  independently. We randomly generate  $n = 20d$  samples of sensing matrices  $\{A_i\}_{i = 1}^n$ ,  $A_{i}\in \mathbb{R}^{d\times d}$  from standard Gaussian distribution and calculate the corresponding labels  $b_{i} = \langle A_{i},M^{*}\rangle$  hence there is no noise in the synthetic data. The global minimum of loss function value  $\Phi (U)$  should be 0 which can be achieved at point  $U = U^{*}$  and  $y = \mathbf{1} / n$ .

Following the setup in (Chen et al. (2021a)), we randomly generalize a vector  $u_0$  from Gaussian distribution and multiply it by a scalar such that it satisfies the condition  $\| u_0 \| \leq \lambda_{max}(M^*)$  where we denote  $\lambda_{max}(\cdot)$  as the maximum eigenvalue. The initial value is set to  $U = [u_0, 0, 0]$ . Each optimization algorithm shares the same initialization. Apart from our PRGDA algorithm, we run three baseline algorithms, SGDA, Acc-MDA and SREDA. The code is implemented on matlab. We choose  $\eta = 0.001$ ,  $\eta_H = 0.1$ ,  $\lambda = 0.01$ ,  $\bar{D} = r = 0.01$ ,  $t_{thres} = 20$ ,  $K = 5$ ,  $S_2 = 40$  and  $q = 25$ .

We evaluate the performance of each algorithm by two criteria, loss function value of  $\Phi(U)$  and the ratio of distance to the optimum  $\|UU^T - M^*\|_F^2/\|M^*\|_F^2$ . The experimental results of these two quantities versus the number of gradient oracles are shown in Figure 1.

From the experimental results we can see SGDA, Acc-MDA and SREDA cannot escape saddle points because the loss function value is far away from the global minimum 0, which is equivalent to

local minimum in this task because of the strict saddle property. In contrast, we can see our PRGDA algorithm eventually converges to the global optimum  $U^{*}$  and achieves the best loss function value that is close to 0, which indicates its ability to escape saddle point. Especially in the case of  $d = 50$ , we can see clearly that our PRGDA algorithm jumps out of the trap of saddle point. Besides, in our experiment we also list the smallest eigenvalue of the Hessian matrix  $\nabla^2\Phi (U)$  for each algorithm after they have converged. Each algorithm is run for 5 times and the mean value is reported in Table 3. We can see the value  $\lambda_{min}(\nabla^{2}\Phi (U))$  of our method is the closest to 0 in all cases, which also verifies the performance of our PRGDA algorithm to find second-order stationary point.

# 7.2 HYPER-REPRESENTATION LEARNING

We also conduct a hyper-representation learning experiment to reach second-order stationary point in bilevel optimization. Recently, many methods in meta learning Finn et al. (2017); Nichol & Schulman (2018) are designed to learn hyper-representations via two steps and separated dataset. The backbone is trained to extract better feature representations which can be applied to many different tasks. Based on these features a classifier is further learned on specific type of

Table 3: Smallest eigenvalue of  ${\nabla }^{2}\Phi \left( U\right)$  .  

<table><tr><td>Algorithm</td><td>d=50</td><td>d=75</td><td>d=100</td></tr><tr><td>SGDA</td><td>-0.0788</td><td>-0.0688</td><td>-0.0360</td></tr><tr><td>Acc-MDA</td><td>-0.0677</td><td>-0.0420</td><td>-0.0257</td></tr><tr><td>SREDA</td><td>-0.0746</td><td>-0.0414</td><td>-0.0259</td></tr><tr><td>PRGDA</td><td>-0.0018</td><td>-0.0074</td><td>-0.0071</td></tr></table>

training data, which eventually forms a bilevel problem. In this experiment we also consider the matrix sensing task but conduct it in the hyper-representation learning manner.

The generation of  $U^{*}$ ,  $M^{*}$ ,  $A_{i}$  and  $b_{i}$  are the same as Section 7.1. We also set  $d = 50$ ,  $d = 75$  and  $d = 100$ . The number of samples is  $n = 20d$ . We split all samples into two datasets: a train dataset  $D_{1}$  with  $70\%$  data and a validation dataset  $D_{2}$  with  $30\%$  data. We define variable  $x$  to be the first  $r - 1$  columns of  $U$  and variable  $y$  to be the last column. The objective function is formulated by

$$
\min  _ {x \in \mathbb {R} ^ {d \times (r - 1)}} \frac {1}{2 | D _ {1} |} \sum_ {i \in D _ {1}} L _ {i} \left(x, y ^ {*} (x)\right), \text {w h e r e} y ^ {*} (x) = \arg \min  _ {y \in \mathbb {R} ^ {d}} \frac {1}{2 | D _ {2} |} \sum_ {i \in D _ {2}} L _ {i} (x, y) \tag {11}
$$

Here  $L_{i}(\cdot)$  is defined in Eq. (9) since  $U$  is the concatenation of  $x$  and  $y$ .

We follow the initialization in Section 7.1 to set  $x = [u_0, \mathbf{0}]$  and  $y = \mathbf{0}$ . We compare our PRGDA algorithm with four baselines, StocBiO, MRBO, VRBO and StocBiO + iNEON. The code is implemented on matlab. We choose  $\eta = 0.001$ ,  $\eta_H = 0.1$ ,  $\lambda = 0.01$ ,  $\bar{D} = r = 0.01$ ,  $t_{thres} = 20$ ,  $K = 5$ ,  $S_2 = 40$  and  $q = 25$ . We also use the ration of distance to optimum, i.e.  $\| UU^T - M^* \|_F^2 / \| M^* \|_F^2$  as the metric to evaluate the performance. The experimental results are shown in Figure 2. We can see our PRGDA shows the best performance to find second-order stationary point and approach the expected optimum. MRBO and VRBO do not escape saddle points during the experiment. In the case of  $d = 50$ , StocBiO performs better than MRBO and VRBO because the randomness of stochastic gradient serves as a kind of perturbation. In variance-reduced algorithms, as the gradient estimator is closer to the full gradient, it suffers more from saddle point than SGD counterparts which indicates the necessity of our method that makes variance-reduced bilevel algorithm able to escape saddle points. StocBiO + iNEON shows poor ability to escape saddle point probably because it is too sensitive to the hyperparameters. This issue is also pointed out in Chen et al. (2021a). The second-order optimality of StocBiO + iNEON relies on the precision of the calculation of descent direction, which is computed by a subroutine. Tuning the parameters of the subroutine is a tradeoff between precision and complexity. Larger batchsize and more iterations lead to better precision but consume more time. In theoretical analysis these parameters can be large enough but in practice they are not. Our PRGDA also has some hyperparameters that are large in the theorem, e.g.,  $t_{thres} = O(\epsilon^{-0.5})$ , but it still works when we only set  $t_{thres} = 20$  in our experiment.

# 8 CONCLUSION

In this paper, we propose a new algorithm PRGDA for stochastic nonconvex hierarchical optimization which is as far as we know the first algorithm to find second-order stationary point for stochastic nonconvex minimax optimization. We prove that our method obtains the gradient complexity of  $\tilde{O} (\epsilon^{-3})$  to achieve  $O(\epsilon ,\sqrt{\rho_{\Phi}\epsilon})$  second-order stationary point, which matches the best results of searching first-order stationary point under same conditions. We also conduct a robust matrix sensing experiment to validate the performance of our algorithm to escape saddle point.

# REFERENCES

Zeyuan Allen-Zhu and Yanzhi Li. Neon2: Finding local minima via first-order oracles. Advances in Neural Information Processing Systems, 31, 2018.  
Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. arXiv preprint arXiv:1805.08136, 2018.  
Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Global optimality of local search for low rank matrix recovery. In Advances in Neural Information Processing Systems, pp. 3873-3881, 2016.  
Zixiang Chen, Dongruo Zhou, and Quanquan Gu. Faster perturbed stochastic gradient methods for finding local minima. arXiv preprint arXiv:2110.13144, 2021a.  
Ziyi Chen, Qunwei Li, and Yi Zhou. Escaping saddle points in nonconvex minimax optimization via cubic-regularized gradient descent-ascent. arXiv preprint arXiv:2110.07098, 2021b.  
Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex sgd. Neural Information Processing Systems (NeurIPS), 2019.  
Justin Domke. Generic methods for optimization-based modeling. In Artificial Intelligence and Statistics, pp. 318-326. PMLR, 2012.  
Simon S. Du and Wei Hu. Linear convergence of the primal-dual gradient method for convex-concave saddle point problems without strong convexity. International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.  
Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. Advances in Neural Information Processing Systems, 31, 2018.  
Matthias Feurer and Frank Hutter. Hyperparameter optimization. In Automated machine learning, pp. 3-33. Springer, Cham, 2019.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pp. 1126-1135. PMLR, 2017.  
Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning, pp. 1568-1577. PMLR, 2018.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In Conference on learning theory, pp. 797–842. PMLR, 2015.  
Rong Ge, Jason D Lee, and Tengyu Ma. Matrix completion has no spurious local minimum. arXiv preprint arXiv:1605.07272, 2016.  
Rong Ge, Chi Jin, and Yi Zheng. No spurious local minima in nonconvex low rank problems: A unified geometric analysis. In International Conference on Machine Learning, pp. 1233-1242. PMLR, 2017.  
Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246, 2018.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Neural Information Processing Systems (NeurIPS), 2014.  
Zhishuai Guo and Tianbao Yang. Randomized stochastic variance-reduced methods for stochastic bilevel optimization. arXiv e-prints, pp. arXiv-2105, 2021.  
Zhishuai Guo, Yi Xu, Wotao Yin, Rong Jin, and Tianbao Yang. A novel convergence analysis for algorithms of the adam family. arXiv preprint arXiv:2112.03459, 2021.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Neural Information Processing Systems (NeurIPS), 2017.  
Christopher J Hillar and Lek-Heng Lim. Most tensor problems are np-hard. Journal of the ACM (JACM), 60(6):1-39, 2013.  
Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale framework for bilevel optimization: Complexity analysis and application to actor-critic. arXiv preprint arXiv:2007.05170, 2020.  
Feihu Huang and Heng Huang. Adagda: Faster adaptive gradient descent ascent methods for minimax optimization. arXiv preprint arXiv:2106.16101, 2021.  
Feihu Huang, Shangqian Gao, Jian Pei, and Heng Huang. Accelerated zeroth-order and first-order momentum methods from mini to minimax optimization. Journal of Machine Learning Research, 23(36):1-70, 2022a.  
Minhui Huang, Kaiyi Ji, Shiqian Ma, and Lifeng Lai. Efficiently escaping saddle points in bilevel optimization. arXiv preprint arXiv:2202.03684, 2022b.  
Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning, pp. 4882-4892. PMLR, 2021.  
Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In International Conference on Machine Learning, pp. 1724-1732. PMLR, 2017.  
Chi Jin, Praneeth Netrapalli, and Michael I. Jordan. What is local optimality in nonconvex-nonconcave minimax optimization? arXiv:1902.00618v2, 2019.  
Prashant Khanduri, Siliang Zeng, Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A near-optimal algorithm for stochastic bilevel optimization via double-momentum. Advances in Neural Information Processing Systems, 34, 2021.  
Zhize Li. Ssrgd: Simple stochastic recursive gradient descent for escaping saddle points. arXiv preprint arXiv:1904.09265, 2019.  
Tianyi Lin, Chi Jin, and Michael Jordan. On gradient descent ascent for nonconvex-concave minimax problems. In International Conference on Machine Learning, pp. 6083-6093. PMLR, 2020a.  
Tianyi Lin, Chi Jin, and Michael I Jordan. Near-optimal algorithms for minimax optimization. In Conference on Learning Theory, pp. 2738-2779. PMLR, 2020b.  
Mingrui Liu, Wei Zhang, Youssef Mroueh, Xiaodong Cui, Jerret Ross, Tianbao Yang, and Payel Das. A decentralized parallel algorithm for training generative adversarial nets. Smooth Games Optimization and Machine Learning Workshop (NeurIPS), 2019.  
Luo Luo and Cheng Chen. Finding second-order stationary point for nonconvex-strongly-concave minimax problem. arXiv preprint arXiv:2110.04814, 2021.  
Luo Luo, Haishan Ye, and Tong Zhang. Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems. Neural Information Processing Systems (NeurIPS), 2020.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. International Conference on Learning Representations (ICLR), 2018.  
Arkadi Nemirovski. Prox-method with rate of convergence  $\mathrm{o}(1 / t)$  for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 2004.  
Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2003.

Yurii Nesterov and Boris T Polyak. Cubic regularization of newton method and its global performance. Mathematical Programming, 108(1):177-205, 2006.  
Lam M. Nguyen, Jie Liu, Katya Scheinberg, and Martin Takáč. Sarah: A novel method for machine learning problems using stochastic recursive gradient. arXiv:1703.00102v2, 2017.  
Lam M Nguyen, Katya Scheinberg, and Martin Takáč. Inexact sarah algorithm for stochastic optimization. Optimization Methods and Software, 36(1):237-258, 2021.  
Alex Nichol and John Schulman. Reptile: a scalable metalearning algorithm. arXiv preprint arXiv:1803.02999, 2(3):4, 2018.  
Maher Nouiehed, Maziar Sanjabi, Tianjian Huang, and Jason D. Lee. Solving a class of nonconvex min-max games using iterative first order methods. Neural Information Processing Systems (NeurIPS), 2019.  
Dohyung Park, Anastasios Kyrillidis, Constantine Carmanis, and Sujay Sanghavi. Non-square matrix sensing without spurious local minima via the burer-monteiro approach. In Artificial Intelligence and Statistics, pp. 65-74. PMLR, 2017.  
Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In International conference on machine learning, pp. 737-746. PMLR, 2016.  
Hassan Rafique, Mingrui Liu, Qihang Lin, and Tianbao Yang. Weakly-convex-concave min-max optimization: provable algorithms and applications in machine learning. Optimization Methods and Software, pp. 1-35, 2021.  
Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated back-propagation for bilevel optimization. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1723-1732. PMLR, 2019.  
Q. Tran-Dinh, D. Liu, and L.M. Nguyen. Hybrid variance-reduced sgd algorithms for minimax problems with nonconvex-linear function. Neural Information Processing Systems (NeurIPS), 2020.  
Hoi-To Wai, Zhuoran Yang, Zhaoran Wang, and Mingyi Hong. Multi-agent reinforcement learning via double averaging primal-dual optimization. Neural Information Processing Systems (NeurIPS), 2018.  
Wenhan Xian, Feihu Huang, Yanfu Zhang, and Heng Huang. A faster decentralized algorithm for nonconvex minimax problems. Advances in Neural Information Processing Systems, 34, 2021.  
Yi Xu, Rong Jin, and Tianbao Yang. First-order stochastic algorithms for escaping from saddle points in almost linear time. Advances in neural information processing systems, 31, 2018.  
Zi Xu, Huiling Zhang, Yang Xu, and Guanghui Lan. A unified single-loop alternating gradient projection algorithm for nonconvex-concave and convex-nonconcave minimax problems. arXiv preprint arXiv:2006.02032, 2020.  
Yan Yan, Yi Xu, Qihang Lin, Lijun Zhang, and Tianbao Yang. Stochastic primal-dual algorithms with faster convergence than  $O(1 / \sqrt{T})$  for problems without bilinear structure. arXiv:1904.10112, 2019.  
Yan Yan, Yi Xu, Qihang Lin, Wei Liu, and Tianbao Yang. Optimal epoch stochastic gradient descent ascent methods for min-max optimization. In Advances in Neural Information Processing Systems, volume 33, pp. 5789-5800. Curran Associates, Inc., 2020.  
Junchi Yang, Antonio Orvieto, Aurelien Lucchi, and Niao He. Faster single-loop algorithms for minimax optimization without strong concavity. In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pp. 5485-5517. PMLR, 28-30 Mar 2022.  
Junjie Yang, Kaiyi Ji, and Yingbin Liang. Provably faster algorithms for bilevel optimization. Advances in Neural Information Processing Systems, 34, 2021.  
Siqi Zhang, Junchi Yang, Cristóbal Guzmán, Negar Kiyavash, and Niao He. The complexity of nonconvex-strongly-concave minimax optimization. arXiv preprint arXiv:2103.15888, 2021.
