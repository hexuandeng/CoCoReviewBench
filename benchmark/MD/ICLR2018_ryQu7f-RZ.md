# ON THE CONVERGENCE OF ADAM AND BEYOND

Anonymous authors

Paper under double-blind review

# ABSTRACT

Several recently proposed stochastic optimization methods that have been successfully used in training deep networks such as RMSPROP, ADAM, ADADELTA, NADAM, etc are based on using gradient updates scaled by square roots of exponential moving averages of squared past gradients. It has been empirically observed that sometimes these algorithms fail to converge to an optimal solution (or a critical point in nonconvex settings). We show that one cause for such failures is the exponential moving average used in the algorithms. We provide an explicit example of a simple convex optimization setting where ADAM does not converge to the optimal solution, and describe the precise problems with the previous analysis of ADAM algorithm. Our analysis suggests that the convergence issues may be fixed by endowing such algorithms with "long-term memory" of past gradients, and propose new variants of the ADAM algorithm which not only fix the convergence issues but often also lead to improved empirical performance.

# 1 INTRODUCTION

Stochastic gradient descent (SGD) is the dominant method to train deep networks today. This method iteratively updates the parameters of a model by moving them in the direction of the negative gradient of the loss evaluated on a minibatch. In particular, variants of SGD that scale coordinates of the gradient by square roots of some form of averaging of the squared coordinates in the past gradients have been particularly successful, because they automatically adjust the learning rate on a per-feature basis. The first popular algorithm in this line of research is ADAGRAD (Duchi et al., 2011; McMahan & Streeter, 2010), which can achieve significantly better performance compared to vanilla SGD when the gradients are sparse, or in general small.

Although ADAGRAD works well for sparse settings, its performance has been observed to deteriorate in settings where the loss functions are nonconvex and gradients are dense due to rapid decay of the learning rate in these settings since it uses all the past gradients in the update. This problem is especially exacerbated in high dimensional problems arising in deep learning. To tackle this issue, several variants of ADAGRAD, such as RMSPROP (Tieleman & Hinton, 2012), ADAM (Kingma & Ba, 2015), ADADELTA (Zeiler, 2012), NADAM (Dozat, 2016), etc, have been proposed which mitigate the rapid decay of the learning rate using the exponential moving averages of squared past gradients, essentially limiting the reliance of the update to only the past few gradients. While these algorithms have been successfully employed in several practical applications, they have also been observed to not converge in some other settings. It has been typically observed that in these settings some minibatches provide large gradients but only quite rarely, and while these large gradients are quite informative, their influence dies out rather quickly due to the exponential averaging, thus leading to poor convergence.

In this paper, we analyze this situation in detail. We rigorously prove that the intuition conveyed in the above paragraph is indeed correct; that limiting the reliance of the update on essentially only the past few gradients can indeed cause significant convergence issues. In particular, we make the following key contributions:

- We elucidate how the exponential moving average in the RMSPROP and ADAM algorithms can cause non-convergence by providing an example of simple convex optimization problem where RMSPROP and ADAM provably do not converge to an optimal solution. Our analysis easily extends to other algorithms using exponential moving averages such as ADADELTA and NADAM as well, but we omit this for the sake of clarity. In fact, the

analysis is flexible enough to extend to other algorithms that employ averaging squared gradients over essentially a fixed size window (for exponential moving averages, the influences of gradients beyond a fixed window size becomes negligibly small) in the immediate past. We omit the general analysis in this paper for the sake of clarity.

- The above result indicates that in order to have guaranteed convergence the optimization algorithm must have "long-term memory" of past gradients. Specifically, we point out a problem with the proof of convergence of the ADAM algorithm given by Kingma & Ba (2015). To resolve this issue, we propose new variants of ADAM which rely on long-term memory of past gradients, but can be implemented in the same time and space requirements as the original ADAM algorithm. We provide a convergence analysis for the new variants in the convex setting, based on the analysis of Kingma & Ba (2015), and show a data-dependent regret bound similar to the one in ADAGRAD.  
- We provide a preliminary empirical study of one of the variants we proposed and show that it either performs similarly, or better, on some commonly used problems in machine learning.

# 2 PRELIMINARIES

Notation. We use  $S_d^+$  to denote the set of all positive definite  $d \times d$  matrices. With slight abuse of notation, for a vector  $a \in \mathbb{R}^d$  and a positive definite matrix  $M \in \mathbb{R}^d \times \mathbb{R}^d$ , we use  $a / M$  to denote  $M^{-1}a$  and  $\sqrt{M}$  to represent  $M^{1/2}$ . Furthermore, for any vectors  $a, b \in \mathbb{R}^d$ , we use  $\sqrt{a}$  for element-wise square root,  $a^2$  for element-wise square,  $a/b$  to denote element-wise division and  $\max(a, b)$  to denote element-wise maximum. For any vector  $\theta_i \in \mathbb{R}^d$ ,  $\theta_{i,j}$  denotes its  $j^{\text{th}}$  coordinate where  $j \in [d]$ . The projection operation  $\Pi_{\mathcal{F},A}(y)$  for  $A \in S_+^d$  is defined as  $\arg \min_{x \in \mathcal{F}} \|A^{1/2}(x - y)\|$  for  $y \in \mathbb{R}^d$ . Finally, we say  $\mathcal{F}$  has bounded diameter  $D_{\infty}$  if  $\|x - y\|_{\infty} \leq D_{\infty}$  for all  $x, y \in \mathcal{F}$ .

**Optimization setup.** A flexible framework to analyze iterative optimization methods is the online optimization problem in the full information feedback setting. In this online setup, at each time step  $t$ , the optimization algorithm picks a point (i.e. the parameters of the model to be learned)  $x_{t} \in \mathcal{F}$ , where  $\mathcal{F} \in \mathbb{R}^{d}$  is the feasible set of points. A loss function  $f_{t}$  (to be interpreted as the loss of the model with the chosen parameters in the next minibatch) is then revealed, and the algorithm incurs loss  $f_{t}(x_{t})$ . The algorithm's regret at the end of  $T$  rounds of this process is given by  $R_{T} = \sum_{i=1}^{T} f_{t}(x_{t}) - \min_{x \in \mathcal{F}} \sum_{i=1}^{T} f_{t}(x)$ .

Our aim is to devise an algorithm that ensures  $R_{T} = o(T)$ , which implies that on average, the model's performance converges to the optimal one. The simplest algorithm for this setting is the standard online gradient descent algorithm (Zinkevich, 2003), which moves the point  $x_{t}$  in the opposite direction of the gradient  $g_{t} = \nabla f_{t}(x_{t})$  while maintaining the feasibility by projecting onto the set  $\mathcal{F}$  via the update rule  $x_{t + 1} = \Pi_{\mathcal{F}}(x_{t} - \alpha_{t}g_{t})$ , where  $\Pi_{\mathcal{F}}(y)$  denotes the projection of  $y \in \mathbb{R}^{d}$  onto the set  $\mathcal{F}$  i.e.,  $\Pi_{\mathcal{F}}(y) = \min_{x \in \mathcal{F}} \| x - y \|$ , and  $\alpha_{t}$  is typically set to  $\alpha / \sqrt{t}$  for some constant  $\alpha$ . The aforementioned online learning problem is closely related to the empirical risk minimization (ERM):  $\min_{x \in \mathcal{F}} \mathbb{E}_{z}[f(x,z)]$ , where  $z$  is a training example drawn training sample over which a model with parameters  $x$  is to be learned, and  $f(x,z)$  is the loss of the model with parameters  $x$  on the sample  $z$ . In particular, an online optimization algorithm with vanishing average regret yields a stochastic optimization algorithm for the ERM problem (Cesa-Bianchi et al., 2004). Thus, we use online gradient descent and stochastic gradient descent (SGD) synonymously.

Generic adaptive methods setup. We now provide a framework of adaptive methods that gives us insights into the differences between different adaptive methods and is useful for understanding the flaws in a few popular adaptive methods. Algorithm 1 provides a generic adaptive framework that encapsulates many popular adaptive methods. Note the algorithm is still abstract because the "averaging" functions  $\phi_t$  and  $\psi_t$  have not been specified. Here  $\phi_t: \mathcal{F}^t \to \mathbb{R}^d$  and  $\psi_t: \mathcal{F}^t \to S_+^d$ . For ease of exposition, we refer to  $\alpha_t$  as step size and  $\alpha_t V_t^{-1/2}$  as learning rate of the algorithm and furthermore, restrict ourselves to diagonal variants of adaptive methods encapsulated by Algorithm 1 where  $V_t = \mathrm{diag}(v_t)$ . We first observe that standard stochastic gradient algorithm falls in this

Algorithm 1 Generic Adaptive Method Setup  
Input:  $x_{1}\in \mathcal{F}$  , step size  $\{\alpha_t > 0\}_{t = 1}^T$  , sequence of functions  $\{\phi_t,\psi_t\}_{t = 1}^T$    
for  $t = 1$  to  $T$  do  
 $g_{t} = \nabla f_{t}(x_{t})$ $m_{t} = \phi_{t}(g_{1},\dots,g_{t})$  and  $V_{t} = \psi_{t}(g_{1}\circ g_{1},\dots,g_{t}\circ g_{t})$ $\hat{x}_{t + 1} = x_{t} - \alpha_{t}m_{t} / \sqrt{V_{t}}$ $x_{t + 1} = \Pi_{\mathcal{F},\sqrt{V}_t}(\hat{x}_{t + 1})$   
end for

framework by using:

$$
\phi_ {t} \left(g _ {1}, \dots , g _ {t}\right) = g _ {t} \text {a n d} \psi_ {t} \left(g _ {1}, \dots , g _ {t}\right) = \mathbb {I}, \tag {SGD}
$$

and  $\alpha_{t} = \alpha /\sqrt{t}$  for all  $t\in [T]$ . While the decreasing step size is required for convergence, such an aggressive decay of learning rate typically translates into poor empirical performance. The key idea of adaptive methods is to choose averaging functions appropriately so as to entail good convergence. For instance, the first adaptive method ADAGRAD (Duchi et al., 2011), which propelled the research on adaptive methods, uses the following averaging functions:

$$
\phi_ {t} \left(g _ {1}, \dots , g _ {t}\right) = g _ {t} \text {a n d} \psi_ {t} \left(g _ {1}, \dots , g _ {t}\right) = \frac {\operatorname {d i a g} \left(\sum_ {i = 1} ^ {t} g _ {i} ^ {2}\right)}{t}, \tag {ADAGRAD}
$$

and step size  $\alpha_{t} = \alpha /\sqrt{t}$  for all  $t\in [T]$ . In contrast to a learning rate of  $\alpha /\sqrt{t}$  in SGD, such a setting effectively implies a modest learning rate decay of  $\alpha /\sqrt{\sum_{i}g_{i,j}^{2}}$  for  $j\in [d]$ . When the gradients are sparse, this can potentially lead to huge gains in terms of convergence (see Duchi et al. (2011)). These gains have also been observed in practice for even few non-sparse settings.

Adaptive methods based on Exponential Moving Averages. Exponential moving average variants of ADAGRAD are popular in the deep learning community. RMSPROP, ADAM, NADAM, and ADADELTA are some prominent algorithms that fall in this category. The key difference is to use an exponential moving average as function  $\psi_t$  instead of the simple average function used in ADAGRAD.  $\mathrm{ADAM}^1$ , a particularly popular variant, uses the following averaging functions:

$$
\phi_ {t} \left(g _ {1}, \dots , g _ {t}\right) = \left(1 - \beta_ {1}\right) \sum_ {i = 1} ^ {t} \beta_ {1} ^ {t - i} g _ {i} \text {a n d} \psi_ {t} \left(g _ {1}, \dots , g _ {t}\right) = \left(1 - \beta_ {2}\right) \operatorname {d i a g} \left(\sum_ {i = 1} ^ {t} \beta_ {2} ^ {t - i} g _ {i} ^ {2}\right), (\mathrm {A D A M})
$$

for some  $\beta_{1},\beta_{2}\in [0,1)$ . This update can alternatively be stated by the following simple recursion:

$$
m _ {t, i} = \beta_ {1} m _ {t - 1, i} + (1 - \beta_ {1}) g _ {t, i} \text {a n d} v _ {t, i} = \beta_ {2} v _ {t - 1, i} + (1 - \beta_ {2}) g _ {t, i} ^ {2} \tag {1}
$$

and  $m_{0,i} = 0$  and  $v_{0,i} = 0$  for all  $i \in [d]$ . and  $t \in [T]$ . Also, note the additional projection operation in Algorithm 1 in comparison to ADAM. When  $\mathcal{F} = \mathbb{R}^d$ , the projection operation is an identity operation and this corresponds to the algorithm in (Kingma & Ba, 2015). For theoretical analysis, one requires  $\alpha_t = 1 / \sqrt{t}$  for  $t \in [T]$ , although, a more aggressive choice of constant step size seems to work well in practice. RMSPROP, which appeared in an earlier unpublished work (Tieleman & Hinton, 2012) is essentially a variant of ADAM with  $\beta_1 = 0$ . In practice, especially in deep learning applications, the momentum term arising due to non-zero  $\beta_1$  appears to significantly boost the performance. We will mainly focus on ADAM algorithm due to this generality but our arguments also apply to RMSPROP.

# 3 THE NON-CONVERGENCE OF ADAM

With the problem setup in the previous section, we discuss fundamental flaw in the current exponential moving average methods like ADAM. We show that ADAM can fail to converge to an optimal

solution even in simple one-dimensional convex settings. These examples of non-convergence contradict the claim of convergence in (Kingma & Ba, 2015), and the main issue lies in the following quantity of interest:

$$
\Gamma_ {t + 1} = \left(\frac {\sqrt {V _ {t + 1}}}{\alpha_ {t + 1}} - \frac {\sqrt {V _ {t}}}{\alpha_ {t}}\right). \tag {2}
$$

This quantity essentially measures the change in the inverse of learning rate of the adaptive method with respect to time. One key observation is that for SGD and ADAGRAD,  $\Gamma_t \succeq 0$  for all  $t \in [T]$ . This simply follows from update rules of SGD and ADAGRAD in the previous section. In particular, update rules for these algorithms lead to "non-increasing" learning rates. However, this is not necessarily the case for exponential moving average variants like ADAM and RMSPROP i.e.,  $\Gamma_t$  can potentially be indefinite for  $t \in [T]$ . We show that this violation of positive definiteness can lead to undesirable convergence behavior for ADAM and RMSPROP. Consider the following simple sequence of linear functions for  $\mathcal{F} = [-1,1]$ :

$$
f _ {t} (x) = \left\{ \begin{array}{l l} C x, & \text {f o r t m o d 3 = 1} \\ - x, & \text {o t h e r w i s e}, \end{array} \right.
$$

where  $C > 2$ . For this function sequence, it is easy to see that the point  $x = -1$  provides the minimum regret. The following result shows that when  $\beta_{1} = 0$  and  $\beta_{2} = 1 / (1 + C^{2})$ , ADAM converges to a highly suboptimal solution of  $x = +1$ . Intuitively, the reasoning is as follows. The algorithm obtains the large gradient  $C$  once every 3 steps, and while the other 2 steps it observes the gradient  $-1$ , which moves the algorithm in the wrong direction. The large gradient  $C$  is unable to counteract this effect since it is scaled down by a factor of almost  $C$  for the given value of  $\beta_{2}$ , and hence the algorithm converges to 1 rather than  $-1$ .

Below we will also present a more general result for general constant  $\beta_{1}$  and  $\beta_{2}$ . We note here that while these examples use constant  $\beta_{1}$  and  $\beta_{2}$ , the proof of convergence of ADAM in (Kingma & Ba, 2015) actually relies on decreasing  $\beta_{1}$  over time. It is quite easy to extend our examples to the case where  $\beta_{1}$  is decreased over time, since the critical parameter is  $\beta_{2}$  rather than  $\beta_{1}$ , and as long as  $\beta_{2}$  is bounded away from 1, our analysis goes through. For the sake of clarity, we only analyze the setting where  $\beta_{1}$  is held constant (as is routinely done in practice) and prove non-convergence of ADAM.

Theorem 1. ADAM with parameter setting such that all the conditions in (Kingma & Ba, 2015) are satisfied can have non-zero average regret i.e.,  $R_{T} / T \nrightarrow 0$  as  $T \to \infty$  for convex  $\{f_i\}_{i=1}^{\infty}$  with bounded gradients on a feasible set  $\mathcal{F}$  having bounded  $D_{\infty}$  diameter.

We relegate all proofs to the appendix. A few remarks are in order. One might wonder if adding a small constant in the denominator of the update helps in circumventing this problem i.e., the update for ADAM in Algorithm 1 of  $\hat{x}_{t + 1}$  is modified as follows:

$$
\hat {x} _ {t + 1} = x _ {t} - \alpha_ {t} m _ {t} / \sqrt {V _ {t} + \epsilon \mathbb {I}}. \tag {3}
$$

The algorithm in (Kingma & Ba, 2015) uses such an update in practice, although their analysis does not. In practice, selection of the  $\epsilon$  parameter appears to be critical for the performance of the algorithm. However, we show that for any constant  $\epsilon > 0$ , there exists an online optimization setting where, again, ADAM has non-zero average regret asymptotically (see Theorem 6 in Section B of the appendix).

The above examples of non-convergence are catastrophic insofar that ADAM and RMSPROP converge to a point that is worst amongst all points in the set  $[-1, 1]$ . Note that above example also holds for constant step size  $\alpha_{t} = \alpha$ . Also note that classic SGD and ADAGRAD do not suffer from this problem and for these algorithms, average regret asymptotically goes to 0. This problem is especially aggravated in high dimensional settings and when the variance of the gradients with respect to time is large. This example also provides intuition for why large  $\beta_{2}$  is advisable while using ADAM algorithm, and indeed in practice using large  $\beta_{2}$  helps. However the following result shows that for any constant  $\beta_{1}$  and  $\beta_{2}$  with  $\beta_{1} < \sqrt{\beta_{2}}$ , we can design an example where ADAM has non-zero average rate asymptotically.

Theorem 2. For any constant  $0 \leq \beta_{1}, \beta_{2} \leq 1$  such that  $\beta_{1} < \sqrt{\beta_{2}}$ , ADAM with parameter setting such that all the conditions in (Kingma & Ba, 2015) are satisfied can have non-zero average regret i.e.,  $R_{T} / T \nrightarrow 0$  as  $T \to \infty$  for convex  $\{f_{i}\}_{i=1}^{\infty}$  with bounded gradients on a feasible set  $\mathcal{F}$  having bounded  $D_{\infty}$  diameter.

Algorithm 2 AMSGRAD  
Input:  $x_{1}\in \mathcal{F}$  step size  $\{\alpha_t\}_{t = 1}^T,\{\beta_{1t}\}_{t = 1}^T,\beta_2$  Set  $m_0 = 0$  and  $v_{0} = 0$    
for  $t = 1$  to  $T$  do   
 $g_{t} = \nabla f_{t}(x_{t})$ $m_t = \beta_{1t}m_{t - 1} + (1 - \beta_{1t})g_t$ $v_{t} = \beta_{2}v_{t - 1} + (1 - \beta_{2})g_{t}^{2}$ $\hat{v}_t = \max (\hat{v}_{t - 1},v_t)$  and  $\hat{V}_t = \mathrm{diag}(\hat{v}_t)$ $x_{t + 1} = \Pi_{\mathcal{F},\sqrt{\hat{V}_t}}(x_t - \alpha_t m_t / \sqrt{\hat{v}_t})$    
end for

The above results show that with constant  $\beta_{1}$  and  $\beta_{2}$ , momentum or regularization via epsilon will help in convergence of the algorithm to the optimal solution. Note that the condition  $\beta_{1} < \sqrt{\beta_{2}}$  is benign and is typically satisfied in the parameter settings used in practice. Furthermore, such condition is typically assumed for convergence of the algorithm (see (Kingma & Ba, 2015)). We can strengthen this result by providing a similar example of non-convergence even in the easier stochastic optimization setting:

Theorem 3. For any constant  $0 \leq \beta_{1}, \beta_{2} \leq 1$  such that  $\beta_{1} < \sqrt{\beta_{2}}$ , there is a stochastic convex optimization problem for which ADAM, with parameter settings such that all the conditions in (Kingma & Ba, 2015) are satisfied, does not converge to the optimal solution.

These results have important consequences insofar that one has to use "problem-dependent"  $\epsilon, \beta_{1}$  and  $\beta_{2}$  in order to avoid bad convergence behavior. In high-dimensional problems, this typically amounts to using, unlike the update in Equation (3), a different  $\epsilon, \beta_{1}$  and  $\beta_{2}$  for each dimension. However, this defeats the purpose of adaptive methods since it requires tuning a large set of parameters. We would also like to emphasize that while the example of non-convergence is carefully constructed to demonstrate the problems in ADAM, it is not unrealistic to imagine scenarios where such an issue can at the very least slow down convergence.

# 4 A NEW EXPONENTIAL MOVING AVERAGE VARIANT: AMSGRAD

In this section, we develop a new principled exponential moving average variant and provide its convergence analysis. Our aim is to devise a new strategy with guaranteed convergence while preserving the practical benefits of ADAM and RMSPROP. To understand the design of our algorithms, let us revisit the quantity  $\Gamma_{t}$  in (2). For ADAM and RMSPROP, this quantity can potentially be negative. The proof in the original paper of ADAM erroneously assumes that  $\Gamma_{t}$  is positive semi-definite and is hence, incorrect (refer to Appendix E for more details). For the first part, we modify these algorithms to satisfy this additional constraint. Later on, we also explore an alternative approach where  $\Gamma_{t}$  can be made positive semi-definite by using values of  $\beta_{1}$  and  $\beta_{2}$  that change with  $t$ .

AMSGrad uses a smaller learning rate in comparison to ADAM and yet incorporates the intuition of slowly decaying the effect of past gradients on the learning rate as long as  $\Gamma_t$  is positive semi-definite. Algorithm 2 presents the pseudocode for the algorithm. The key difference of AMSGrad with ADAM is that it maintains the maximum of all  $v_t$  until the present time step and uses this maximum value for normalizing the running average of the gradient instead of  $v_t$  in ADAM. By doing this, AMSGrad results in a non-increasing step size and avoids the pitfalls of ADAM and RMSPROP i.e.,  $\Gamma_t \succeq 0$  for all  $t \in [T]$  even with constant  $\beta_2$ . Also, in Algorithm 2, one typically uses a constant  $\beta_{1t}$  in practice (although, the proof requires a decreasing schedule for proving convergence of the algorithm).

To gain more intuition for the updates of AMSGRAD, it is instructive to compare its update with ADAM and ADAGRAD. Suppose at particular time step  $t$  and coordinate  $i \in [d]$ , we have  $v_{t-1,i} > g_{t,i}^2 > 0$ , then ADAM aggressively increases the learning rate, however, as we have seen in the previous section, this can be detrimental to the overall performance of the algorithm. On the other hand, ADAGRAD slightly decreases the learning rate, which often leads to poor performance in practice since such an accumulation of gradients over a large time period can significantly decrease the learning rate. In contrast, AMSGRAD neither increases nor decreases the learning rate and

![](images/55ccd88e936acbd4de73ba18712287219b12fbfcc56dbb62e0925c4b24cf3e96.jpg)  
Figure 1: Performance comparison of ADAM and AMSGRAD on synthetic example on a simple one-dimensional convex problem inspired by our examples of non-convergence. The first two plots (left and center) are for the online setting and the the last one (right) is for the stochastic setting.

![](images/b0ccfc8edb8342474cce4d84e507ff0a72b2d6b02aa1fd530b445a8691ffa98b.jpg)

![](images/090fac8d3adf13a38c9fddebda784f669018ccb349661e3e0e058ff10c9bad60.jpg)

furthermore, decreases  $v_{t}$  which can potentially lead to non-decreasing learning rate even if gradient is large in the future iterations. We prove the following key result for AMSGRAD algorithm.

Theorem 4. Let  $\{x_{t}\}$  and  $\{v_{t}\}$  be the sequences obtained from Algorithm 2,  $\alpha_{t} = \alpha / \sqrt{t}$ ,  $\beta_{1} = \beta_{11}$ ,  $\beta_{1t} \leq \beta_{1}$  for all  $t \in [T]$  and  $\gamma = \beta_{1} / \sqrt{\beta_{2}} < 1$ . Assume that  $\mathcal{F}$  has bounded diameter  $D_{\infty}$  and  $\|\nabla f_{t}(x)\|_{\infty} \leq G_{\infty}$  for all  $t \in [T]$  and  $x \in \mathcal{F}$ . For  $x_{t}$  generated using the AMSGRAD (Algorithm 2), we have the following bound on the regret

$$
R _ {T} \leq \frac {D _ {\infty} ^ {2} \sqrt {T}}{(1 - \beta_ {1})} \sum_ {i = 1} ^ {d} \hat {v} _ {T, i} ^ {1 / 2} + \frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {d} \frac {\beta_ {1 t} \hat {v} _ {t , i} ^ {1 / 2}}{\alpha_ {t}} + \frac {\alpha \sqrt {1 + \log T}}{(1 - \beta_ {1}) ^ {2} (1 - \gamma) \sqrt {(1 - \beta_ {2})}} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2}.
$$

The following result falls as an immediate corollary of the above result.

Corollary 1. Suppose  $\beta_{1t} = \beta_1\lambda^{t - 1}$  in Theorem 4, then we have

$$
R _ {T} \leq \frac {D _ {\infty} ^ {2} \sqrt {T}}{(1 - \beta_ {1})} \sum_ {i = 1} ^ {d} \hat {v} _ {T, i} ^ {1 / 2} + \frac {\beta_ {1} D _ {\infty} ^ {2} G _ {\infty}}{2 (1 - \beta_ {1}) (1 - \lambda) ^ {2}} + \frac {\alpha \sqrt {1 + \log T}}{(1 - \beta_ {1}) ^ {2} (1 - \gamma) \sqrt {(1 - \beta_ {2})}} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2}.
$$

The above bound can be considerably better than  $O(\sqrt{T})$  regret of SGD when  $\sum_{i=1}^{d} \hat{v}_{T,i}^{1/2} \ll dG_{\infty}$  and  $\sum_{i=1}^{d} \|g_{1:T,i}\|_2 \ll dG_{\infty}$ . Furthermore, in Theorem 4, one can use a much more modest momentum decay of  $\beta_{1t} = \beta_1 / t$  and still ensure a regret of  $O(\sqrt{T})$ . We would also like to point out that one could consider taking simple average of smoothed gradients instead of using the maximum of all the previous values of  $v_t$ . The resulting algorithm is very similar to ADAGRAD except for normalization with smoothed gradients rather than actual gradients and can be shown to have similar convergence as ADAGRAD.

# 5 EXPERIMENTS

In this section, we present empirical results on both synthetic and real-world datasets. For our experiments, we study the problem of multiclass classification using logistic regression and neural networks, representing convex and nonconvex settings, respectively.

Synthetic Experiments: To demonstrate the convergence issue of ADAM, we first consider the following simple convex setting inspired from our examples of non-convergence:

$$
f _ {t} (x) = \left\{ \begin{array}{l l} 1 0 1 0 x, & \text {f o r t m o d 1 0 1 = 1} \\ - 1 0 x, & \text {o t h e r w i s e}, \end{array} \right.
$$

with the constraint set  $\mathcal{F} = [-1,1]$ . We first observe that, similar to the examples of nonconvergence we have considered, the optimal solution is  $x = -1$ ; thus, for convergence, we expect the algorithms to converge to  $x = -1$ . For this sequence of functions, we investigate the regret and the value of the iterate  $x_{t}$  for ADAM and AMSGRAD. To enable fair comparison, we set  $\beta_{1} = 0.9$  and  $\beta_{2} = 0.99$  for ADAM and AMSGRAD algorithm, which are typically the parameters settings used for ADAM in practice. Figure 1 shows the average regret  $(R_{t} / t)$  and value of the iterate  $(x_{t})$  for

![](images/e72108febf8069b2cfbc83fee3f386b274ec025b662c1b31fccbeebdb5dc6c70.jpg)

![](images/23ca14f16fd3677adfa483dc4179c653bc1ec5e7f17ff4b67eb37f5d8c654ac1.jpg)

![](images/3098bc49014df07e254d688725a0cd0814782b34b3fee02285ff1b39d11d5665.jpg)

![](images/64ab71c5aa9901a46ebc6b428038b581cefdd29b2ce0164b28aabb351d543a14.jpg)  
Figure 2: Performance comparison of ADAM and AMSGRAD for logistic regression, feedforward neural network and CIFARNET. The top row shows performance of ADAM and AMSGRAD on logistic regression (left and center) and 1-hidden layer feedforward neural network (right) on MNIST. In the bottom row, the two plots compare the training and test loss of ADAM and AMSGRAD with respect to the iterations for CIFARNET.

![](images/27773ab5175a9fa5b9d352463ef607ce5908be5164de6d0dc761c66167ba2390.jpg)

this problem. We first note that the average regret of ADAM does not converge to 0 with increasing  $t$ . Furthermore, its iterates  $x_{t}$  converge to  $x = 1$ , which unfortunately has the largest regret amongst all points in the domain. On the other hand, the average regret of AMSGRAD converges to 0 and its iterate converges to the optimal solution. Figure 1 also shows the stochastic optimization setting:

$$
f _ {t} (x) = \left\{ \begin{array}{l l} 1 0 1 0 x, & \text {w i t h p r o b a b i l i t y 0 . 0 1} \\ - 1 0 x, & \text {o t h e r w i s e .} \end{array} \right.
$$

Similar to the aforementioned online setting, the optimal solution for this problem is  $x = -1$ . Again, we see that the iterate  $x_{t}$  of ADAM converges to the highly suboptimal solution  $x = 1$ .

Logistic Regression: To investigate the performance of the algorithm on convex problems, we compare AMSGRAD with ADAM on logistic regression problem. We use MNIST dataset for this experiment, the classification is based on 784 dimensional image vector to one of the 10 class labels. The step size parameter  $\alpha_{t}$  is set to  $\alpha / \sqrt{t}$  for both ADAM and AMSGRAD in for our experiments, consistent with the theory. We use a minibatch version of these algorithms with minibatch size set to 128. We set  $\beta_{1} = 0.9$  and  $\beta_{2}$  is chosen from the set  $\{0.99, 0.999\}$ , but they are fixed throughout the experiment. The parameters  $\alpha$  and  $\beta_{2}$  are chosen by grid search. We report the train and test loss with respect to iterations in Figure 2. We can see that AMSGRAD performs better than ADAM with respect to both train and test loss. We also observed that AMSGRAD is relatively more robust to parameter changes in comparison to ADAM.

Neural Networks: For our first experiment, we trained a simple 1-hidden fully connected layer neural network for the multiclass classification problem on MNIST. Similar to the previous experiment, we use  $\beta_{1} = 0.9$  and  $\beta_{2}$  is chosen from  $\{0.99, 0.999\}$ . We use a fully connected 100 rectified linear units (ReLU) as the hidden layer for this experiment. Furthermore, we use constant  $\alpha_{t} = \alpha$  throughout all our experiments on neural networks. Such a parameter setting choice of ADAM is consistent with the ones typically used in the deep learning community for training neural networks. A grid search is used to determine parameters that provides the best performance for the algorithm.

Finally, we consider the multiclass classification problem on the standard CIFAR-10 dataset, which consists of 60,000 labeled examples of  $32 \times 32$  images. We use CIFARNET, a convolutional neural network (CNN) with several layers of convolution, pooling and non-linear units, for training a multiclass classifier for this problem. In particular, this architecture has 2 convolutional layers with 64 channels and kernel size of  $6 \times 6$  followed by 2 fully connected layers of size 384 and 192. The network uses  $2 \times 2$  max pooling and layer response normalization between the convolutional layers (Krizhevsky et al., 2012). A dropout layer with keep probability of 0.5 is applied in between the fully connected layers (Srivastava et al., 2014). The minibatch size is also set to 128 similar to previous experiments. The results for this problem are reported in Figure 2. The parameters for ADAM and AMSGRAD are selected in a way similar to the previous experiments. We can see that

AMSGRAD performs considerably better than ADAM on train loss and accuracy. Furthermore, this performance gain also translates into good performance on test loss.

# 5.1 EXTENSION: ADAMNC ALGORITHM

An alternative approach is to use an increasing schedule of  $\beta_{2}$  in ADAM. This approach, unlike Algorithm 2 does not require changing the structure of ADAM but rather uses a non-constant  $\beta_{1}$  and  $\beta_{2}$ . The pseudocode for the algorithm, ADAMNC, is provided in the appendix (Algorithm 3). We show that by appropriate selection of  $\beta_{1t}$  and  $\beta_{2t}$ , we can achieve good convergence rates.

Theorem 5. Let  $\{x_{t}\}$  and  $\{v_{t}\}$  be the sequences obtained from Algorithm 3,  $\alpha_{t} = \alpha / \sqrt{t}$ ,  $\beta_{1} = \beta_{11}$  and  $\beta_{1t} \leq \beta_{1}$  for all  $t \in [T]$ . Assume that  $\mathcal{F}$  has bounded diameter  $D_{\infty}$  and  $\|\nabla f_{t}(x)\|_{\infty} \leq G_{\infty}$  for all  $t \in [T]$  and  $x \in \mathcal{F}$ . Furthermore, let  $\{\beta_{2t}\}$  be such that the following conditions are satisfied:

1.  $\frac{1}{\alpha_T}\sqrt{\sum_{j=1}^t\Pi_{k=1}^{t-j}\beta_{2(t-k+1)}(1-\beta_{2j})g_{j,i}^2}\geq\frac{1}{\zeta}\sqrt{\sum_{j=1}^tg_{j,i}^2}$  for some  $\zeta>0$  and all  $t\in[T]$ ,  
2.  $\frac{v_{t,i}^{1/2}}{\alpha_t} \geq \frac{v_{t-1,i}^{1/2}}{\alpha_{t-1}}$  for all  $t \in \{2, \dots, T\}$  and  $i \in [d]$ .

Then for  $x_{t}$  generated using the ADAMNC (Algorithm 3), we have the following bound on the regret

$$
R _ {T} \leq \frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \sum_ {i = 1} ^ {d} \sqrt {T} v _ {T, i} ^ {1 / 2} + \frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {d} \frac {\beta_ {1 t} v _ {t , i} ^ {1 / 2}}{\alpha_ {t}} + \frac {2 \zeta}{(1 - \beta_ {1}) ^ {3}} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2}.
$$

The above result assumes selection of  $\{(\alpha_{t},\beta_{2t})\}$  such that  $\Gamma_t\succeq 0$  for all  $t\in \{2,\dots ,T\}$ . However, one can generalize the result to deal with the case where this constraint is violated as long as the violation is not too large or frequent. Following is an immediate consequence of the above result.

Corollary 2. Suppose  $\beta_{1t} = \beta_1\lambda^{t - 1}$  and  $\beta_{2t} = 1 - 1 / t$  in Theorem 5, then we have

$$
\frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2} + \frac {\beta_ {1} D _ {\infty} ^ {2} G _ {\infty}}{2 (1 - \beta_ {1}) (1 - \lambda) ^ {2}} + \frac {2 \zeta}{(1 - \beta_ {1}) ^ {3}} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2}.
$$

The above corollary follows from a trivial fact that  $v_{t,i} = \sum_{j=1}^{t} g_{j,i}^{2} / t$  for all  $i \in [d]$  when  $\beta_{2t} = 1 - 1 / t$ . This corollary is interesting insofar that such a parameter setting effectively yields a momentum based variant of ADAGRAD. Similar to ADAGRAD, the regret is data-dependent and can be considerably better than  $O(\sqrt{T})$  regret of SGD when  $\sum_{i=1}^{d} \|g_{1:T,i}\|_2 \ll dG_\infty$ . It is easy to generalize this result for setting similar settings of  $\beta_{2t}$ . Similar to Corollary 1, one can again use a much more modest decay of  $\beta_{1t} = \beta_1 / t$  and still ensure a data-dependent regret of  $O(\sqrt{T})$ .

# 6 DISCUSSION

In this paper, we study exponential moving variants of ADAGRAD and identify an important flaw in these algorithms which can lead to undesirable convergence behavior. We demonstrate these problems through carefully constructed examples where RMSPROP and ADAM converge to highly suboptimal solutions. In general, any algorithm that relies on an essentially fixed sized window of past gradients to scale the gradient updates will suffer from this problem.

We proposed fixes to this problem by slightly modifying the algorithms, essentially endowing the algorithms with a long-term memory of past gradients. These fixes retain the good practical performance of the original algorithms, and in some cases actually show improvements.

The primary goal of this paper is to highlight the problems with popular exponential moving average variants of ADAGRAD from a theoretical perspective. RMSPROP and ADAM have been immensely successful in development of several state-of-the-art solutions for a wide range of problems. Thus, it is important to understand their behavior in a rigorous manner and be aware of potential pitfalls while using them in practice. We believe this paper is a first step in this direction and suggests good design principles for faster and better stochastic optimization.

# REFERENCES

Peter Auer and Claudio Gentile. Adaptive and self-confident on-line learning algorithms. In Proceedings of the 13th Annual Conference on Learning Theory, pp. 107-117, 2000.  
Nicol Cesa-Bianchi, Alex Conconi, and Claudio Gentile. On the generalization ability of on-line learning algorithms. IEEE Transactions on Information Theory, 50:2050-2057, 2004.  
Timothy Dozat. Incorporating Nesterov Momentum into Adam. In Proceedings of 4th International Conference on Learning Representations, Workshop Track, 2016.  
John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121-2159, 2011.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of 3rd International Conference on Learning Representations, 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pp. 1097-1105, 2012.  
H. Brendan McMahan and Matthew J. Streeter. In Proceedings of the 23rd Annual Conference on Learning Theory, pp. 244-256, 2010.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15:1929-1958, 2014.  
T. Tieleman and G. Hinton. RmsProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.  
Matthew D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. CoRR, abs/1212.5701, 2012.  
Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning, pp. 928-936, 2003.
