# ADAX: ADAPTIVE GRADIENT DESCENT WITH EXPONENTIAL LONG TERM MEMORY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Adaptive optimization algorithms such as RMSProp and Adam have fast convergence and smooth learning process. Despite their successes, they are proven to have non-convergence issue even in convex optimization problems as well as weak performance compared with the first order gradient methods such as stochastic gradient descent (SGD). Several other algorithms, for example AMSGrad and AdaShift, have been proposed to alleviate these issues but only minor effect has been observed. This paper further analyzes the performance of such algorithms in a non-convex setting by extending their non-convergence issue into a simple nonconvex case and show that Adam's design of update steps would possibly lead the algorithm to local minimums. To address the above problems, we propose a novel adaptive gradient descent algorithm, named AdaX, which accumulates the long-term past gradient information exponentially. We prove the convergence of AdaX in both convex and non-convex settings. Extensive experiments show that AdaX outperforms Adam in various tasks of computer vision and natural language processing and can catch up with SGD.

# 1 INTRODUCTION

In the era of deep learning, Stochastic Gradient Descent (SGD), though proposed in the last century, remains the most effective algorithm in training deep neural networks (Robbins & Monro, 1951). Many methods have been created to accelerate the training process and boost the performance of SGD, such as momentum (Polyak, 1964) and Nesterov's acceleration (Nesterov, 1983). Recently, adaptive optimization methods have become popular as they adjust parameters' learning rates in different scales, instead of directly controlling the overall step size. These algorithms schedule the learning rates using a weighted average of the past gradients. For example, AdaGrad (Duchi et al., 2011) chooses the square root of the global average of the past gradients as the denominator of the adaptive learning rates. It is shown that when the gradients are sparse or small, AdaGrad can converge faster than vanilla SGD. However, its performance is quite limited in non-sparse settings.

Other adaptive algorithms have been proposed to replace the global average in AdaGrad by using the exponential moving average of past gradients, such as RMSProp (Tieleman & Hinton, 2012), AdaDelta (Zeiler, 2012), and Adam (Kingma & Ba, 2015). Among all these variants, Adam (Kingma & Ba, 2015) is the most popular yet controversial optimization algorithm since it has faster convergence rate than the others. However, Adam has worse performance (i.e. generalization ability in testing stage) compared with SGD. Recent theories (Wilson et al., 2017; Reddi et al., 2018) have also shown that Adam suffers from non-convergence issue and weak generalization ability. For example, Reddi et al. (2018) proposed that Adam's non-convergence problem originate from a mistake in their proof of convergence. They constructed a counterexample for Adam and thoroughly proved that Adam did not guarantee convergence even in a simple convex setting.

In the meantime, Zhou et al. (2019) theoretically proved that Adam could make a large update when gradients are small, and a small update when gradients are large, which probably lead the optimization process to wrong directions. Shazeer & Stern (2018) also empirically showed that Adam's parameter updates are not stable and its second moment could be out of date. Luo et al. (2019) examined the effective learning rate of Adam in training and found that Adam would produce too large or too small extreme learning rates. All the above analyses have suggested that Adam's unstable design of adaptive learning rate may impair the optimization process.

To address the above issues, we propose a new adaptive optimization method, termed AdaX, which guarantees convergence in both convex and non-convex settings. This is done by memorizing the long-term square of gradients exponentially as the second moment. To achieve this, we first extend the convex counterexample in Reddi et al. (2018) to a non-convex setting. We then analyze how the second moment instability in Adam and RMSProp could cause the optimization process to converge to a local minimum even without noisy gradients, revealing that Adam produces update steps that are too large in optimization. We also show how AMSGrad (Reddi et al., 2018) is unable to solve Adam's problem completely, because its effectiveness relies heavily on the magnitude of the maximal second moment, for instance, too large second moment would lead to insufficient training and thus sub-optimal solutions. To address the above problems, we introduce a novel AdaX algorithm and theoretically prove that it converges with a similar speed to Adam, but gets rid of the second moment instability. Extensive experiments show that AdaX outperforms Adam and is comparable to SGD with momentum in many tasks of computer vision and natural language processing.

# 2 BACKGROUND AND NOTATION

Overview of Adaptive Methods. To compare AdaX with the previous optimization methods, we follow (Reddi et al., 2018) to present a generic framework of adaptive algorithms as shown in Algorithm 1. Let  $S_{d}^{+}$  be a set of positive definite matrices in  $\mathbb{R}^{d\times d}$ , and  $\mathcal{F}$  be the parameter domain. In line 1, we take  $\phi_t:\mathcal{F}\to \mathbb{R}^d$  and  $\psi_t:\mathcal{F}\rightarrow S_+^d$  as input, which are unspecified moment functions that vary among different optimization algorithms. After obtaining the gradient at time  $t$  in line 3, we can calculate the corresponding first and second moment  $m_t,V_t$ . In line 5,  $\alpha_{t} = \alpha /\sqrt{t}$  is chosen as the step size in each iteration for the convergence analysis of adaptive algorithms. The projection operation in line 6,  $\Pi_{\mathcal{F},M}(y)$  is defined as  $\mathrm{argmin}_{x\in \mathcal{F}}\| \sqrt{M} (x - y)\|$ , where  $M\in S_d^+$  and  $y\in \mathbb{R}^d$ .

Algorithm 1 Generic Adaptive Optimization Algorithm

1: Input:  $x \in  \mathcal{F}$  ,step size  $\alpha$  ,sequence of functions  ${\left\{  {\phi }_{t},{\psi }_{t}\right\}  }_{t = 1}^{T}$  
2: for  $t = 1$  to  $T$  do  
3:  $g_{t} = \nabla f_{t}(x_{t})$  
4:  $m_{t} = \phi_{t}(g_{1},g_{2},\ldots ,g_{t})$  and  $V_{t} = \psi_{t}(g_{1},g_{2},\dots,g_{t})$  
5:  $\alpha_{t} = \alpha /\sqrt{t}$  
6:  $x_{t + 1} = \Pi_{\mathcal{F},\sqrt{V_t}}(x_t - \alpha_t m_t / \sqrt{V_t})$  
7: end for

The main differences between the adaptive methods and the conventional SGD are in line 4 and 6, where the matrix  $V_{t}$  scales the overall step size  $\alpha_{t}$  element-wisely by  $1 / \sqrt{V_t}$ , known as the adaptive learning rate. Using the general framework in Algorithm 1, we are able to summarize many adaptive optimization algorithms proposed recently. For example, AdaGrad (Duchi et al., 2011) designed its  $V_{t}$  as the global average of past gradients, while Adam (Kingma & Ba, 2015) and RMSProp (Tieleman & Hinton, 2012) chose the exponential moving average as follows instead.

$$
V _ {t} = \left(\frac {1 - \beta_ {2}}{1 - \beta_ {2} ^ {t}}\right) \operatorname {d i a g} \left(\sum_ {i = 1} ^ {t} \beta_ {2} ^ {t - i} g _ {i} ^ {2}\right), \tag {Adam}
$$

where  $\beta_{2}$  is the fixed second moment coefficient and  $g_{i}^{2}$  denotes the element-wise square of the gradients. The diagonal operation diag() perform the dimension transformation from  $\mathbb{R}^d$  to  $S_{+}^{d}$ . In order to improve the performance of Adam, Reddi et al. (2018) proposed AMSGrad, which took max operation on the second moment. Zhou et al. (2019) argued that we could replace  $g_{t}^{2}$  in  $V_{t}$  with some past gradient squares  $g_{t-n}^{2}$  to temporarily remove the correlation between the first and second moment. Huang et al. (2019) constructed NosAdam, where a sequence of  $\beta_{2t}$ 's gave higher weights to past gradients. We provide a summary of different designs of adaptive learning rate in Table 1. It's noticeable that these algorithms, due to their exponential moving average design, still assign relatively high weights on recent gradients and past information is not emphasized. Besides, Loshchilov & Hutter (2019) noticed the difference between  $L_{2}$  regularization and weight decay in adaptive algorithms and improved Adam with standard weight decay by proposing AdamW.

Optimization Framework. A commonly used framework for analyzing convex optimization algorithms was constructed by Zinkevich (2003), named the online optimization problem. In this

Table 1: Comparisons of different designs of the second moment  

<table><tr><td></td><td>SGDM</td><td>AdaGrad</td><td>RMSProp</td></tr><tr><td>ψt</td><td>II</td><td>diag(∑i=1tgi2/t)</td><td>(1 - β2)diag(∑i=1tβ2t-i gi2)</td></tr><tr><td></td><td>Adam</td><td>AMSGrad</td><td>AdaShift</td></tr><tr><td>ψt</td><td>(1-β2/1-β2t)diag(∑i=1tβ2t-i gi2)</td><td>diag(max(vt-1, vt))</td><td>diag(β2vt-1 + (1 - β2)gt-n)</td></tr><tr><td></td><td>NosAdam</td><td>...</td><td>AdaX (ours)</td></tr><tr><td>ψt</td><td>diag(β2vt-1 + (1 - β2t)gt-n)</td><td>...</td><td>β2/(1+β2)t-1diag(∑i=1t(1 + β2)t-i gi2)</td></tr></table>

framework setting, the optimization algorithm chooses a parameter set  $\theta_t \in \mathcal{F}$  and an unknown cost function  $f_t(\theta)$  evaluates its performance at  $\theta_t$  in each iteration. Suppose that there exists a best parameter  $f_t(\theta^*)$  such that  $\theta^* = \operatorname{argmin}_{\theta \in \mathcal{F}} \left( \sum_{t=1}^{T} f_t(\theta) \right)$ , then a metric used to show the algorithm's performance is the regret function  $R_T = \sum_{t=1}^{T} f_t(\theta_t) - f_t(\theta^*)$  and we want to ensure that  $R_T = o(T)$  so that the algorithm will always converge to the optimal solution. (Zinkevich, 2003).

Non-convergence of Adam. Reddi et al. (2018) proposed that the matrix  $\Gamma_t$  defined as follows, was mistakenly assumed to be positive semi-definite in the original convergence proof of Adam.

$$
\Gamma_ {t} = \left(\frac {\sqrt {V _ {t + 1}}}{\alpha_ {t + 1}} - \frac {\sqrt {V _ {t}}}{\alpha_ {t}}\right), \tag {1}
$$

where  $V_{t}$  and  $\alpha_{t}$  are defined as in Algorithm 1. Based on such an observation, they constructed the following online convex optimization problem, in which Adam failed to converge to the optimal solution. Let  $C > 2$  be a fixed constant and  $\{f_t\}$  be the sequence of cost functions whose sum is to be minimized. Define  $f_{t}$  as follows

$$
f _ {t} (x) = \left\{ \begin{array}{l} C x, \text {f o r} t \bmod 3 = 1 \\ - x, \text {o t h e r w i s e} \end{array} \right. \tag {2}
$$

In this problem, Adam could not distinguish between the true large gradient direction  $(C)$  and the noisy gradient directions  $(-1)$  because its  $\sqrt{V_t}$  scales the gradients to be of similar sizes, which forces the algorithm to reach a highly suboptimal solution  $x = 1$  every three iterations. However, SGD and AdaGrad are both able to counteract the noisy gradients and converge to the optimum, which reveals the fact that Adam's design of adaptive learning rate is very unstable.

# 3 THE NONCONVERGENCE OF ADAM IN A NON-CONVEX SETTING

In this section, we extend the non-convergence problem in (2) to the non-convex setting and explain why the fast convergence of Adam impairs its performance in the long term. Let  $C \in (1, +\infty)$ ,  $\lambda \in (0, 1)$  be constants in  $\mathbb{R}$ , consider the following simple sequence of non-convex functions  $f_t$ .

$$
f _ {t} (x) = \left\{ \begin{array}{l l} C \lambda^ {t - 1} x, & \text {f o r} x \geq 0 \\ \frac {C ^ {2}}{1 - \lambda}, & \text {f o r} x <   0 \end{array} \quad \forall t \geq 1 \right. \tag {3}
$$

It can be easily observed that for the domain  $\mathcal{F} = [-2,C / (1 - \lambda)]$ , the minimum of each  $f_{t}$  is obtained at  $x = 0$ . Suppose we start from  $x_0 > 0$ , then this problem simulates a situation where the gradient decreases exponentially as time increases, implying that the algorithm is approaching the global minimum and smaller step sizes are needed. Compared with the non-convergence problem proposed by Reddi et al. (2018), no high-frequency noise exist in our gradients. However, next to the optimal solution, there is a local minimum trap where no gradients exist and thus no algorithm could escape. Let  $\alpha_{1} = \alpha$  be the initial step size, we are able to show that SGD is capable of avoiding the trap, even without the learning rate decrease  $\alpha_{t} = \alpha /\sqrt{t}$ , and will converge to  $x = 0$  if initialized well. However, Adam ignores the gradient decrease information and always enters the trap regardless of initialization. We summarize the above results in the following lemma

Lemma 3.1 In problem (3), with  $\beta_{1} = 0$ ,  $\beta_{2} \in (0, \lambda^{2})$  and  $\alpha_{t} \geq \alpha / t$ , Adam will always reach the local minimum, i.e.  $\sum_{t=1}^{T} f(t) / T \to \frac{C^{2}}{1 - \lambda}$ ,  $\forall x_{1}, \alpha_{1} > 0$ .

We provide all the proofs in the Appendix. This parameter setting of Adam is the same as RMSProp except for the bias correction term, and the condition  $\beta_{1} < \sqrt{\beta_{2}}$  mentioned by Kingma & Ba (2015) is automatically satisfied.  $\alpha_{t} \geq \alpha / t$  is a weak requirement for the step sizes and can be ensured with constant step sizes or  $\alpha_{t} = \alpha / \sqrt{t}$  as in the convergence analysis. Intuitively, Adam would scale the decreasing gradient by  $1 / \sqrt{V_t}$ , which approximately increases with the same speed. Therefore, its update steps would be larger than a fixed constant at any time step and would ultimately lead the parameters to the trap regardless of initialization. People may wonder whether the first moment design helps Adam in such a situation, but we can show that as long as the condition  $\beta_{1} < \sqrt{\beta_{2}}$  is satisfied, Adam would always reach the local minimum. Hence, Adam converges faster than SGD because of its large updates, but it cannot slow down when approaching the global minimum.

AMSGrad is constructed to address the problem of Adam's large steps during optimization. However, it suffers from two major issues in a non-convex situation. 1) The never decreasing  $V_{t}$  in AMSGrad could lead to early stops and therefore insufficient training during optimization, as revealed by Huang et al. (2019); 2) The time for achieving the maximum of  $V_{t}$  is uncontrollable. We show that for certain cases in our problem, AMSGrad is unable to help Adam.

Lemma 3.2 In problem (3), with  $\beta_{1} = 0$  and  $\alpha_{t}\geq \alpha /t$ ,  $\forall \beta_{2}\in (0,1)$ ,  $\exists \lambda \in (\sqrt{\beta_2},1)$ , such that AMSGrad will always reach the local minimum, i.e.  $\sum_{t = 1}^{T}f(t) / T\to \frac{C^2}{1 - \lambda},\forall x_1,\alpha_1 > 0$ .

The lemma essentially states for any fixed  $\beta_{2}$ , we can find a  $\lambda$  such that AMSGrad cannot help Adam because  $V_{t}$  keeps increasing before stepping into the trap. Therefore, we still need an algorithm that can generate stable learning rates and control the update steps effectively. We emphasize that although the functions in (3) are not smooth, the problem does successfully provide some intuition on why Adam variants trains much faster than SGD, but cannot have comparable testing performance.

# 4 ALGORITHM AND CONVERGENCE

Next, we introduce our novel optimization algorithm and present its special way of adjusting the adaptive learning rate. Based upon the above discussions that current gradients lead to unstable second moment and that long-term memory algorithms are preferred, we design our algorithm AdaX by weighting exponentially more on the history gradients and less on the current gradients, as shown in Algorithm 2. The most important differences between AdaX and Adam are in line 6 and 7, where instead of an exponential moving average, we change  $\beta_{2}$  to  $1 + \beta_{2}$  and accumulate the past gradients. Such a design guarantees that noisy and extreme gradients cannot greatly influence the update steps, and  $\hat{v}_{t}$  would gradually become stable. Similar to Kingma & Ba (2015)'s derivation, in order to

Algorithm 2 AdaX Algorithm  
1: Input:  $x \in \mathcal{F}$ , step size  $\{\alpha_t\}_{t=1}^T, \beta_1, \beta_2, \beta_3$   
2: Initialize  $m_0 = 0, v_0 = 0$   
3: for  $t = 1$  to  $T$  do  
4:  $g_t = \nabla f_t(x_t)$   
5:  $m_t = \beta_1 m_{t-1} + (1 - \beta_3) g_t$   
6:  $v_t = (1 + \beta_2) v_{t-1} + \beta_2 g_t^2$   
7:  $\hat{v}_t = v_t / [(1 + \beta_2)^t - 1]$  and  $V_t = \mathrm{diag}(\hat{v}_t)$   
8:  $x_{t+1} = \Pi_{\mathcal{F}, \sqrt{V_t}}(x_t - \alpha_t m_t / \sqrt{\hat{v}_t})$   
9: end for

achieve an unbiased estimate of second moment, we obtain our bias correction term as follows. Let  $g_{t}$  be the gradient at timestep  $t$  and further suppose  $g_{t}$ 's are drawn from a stationary distribution

$g_{t} \sim p(g_{t})$ . Take expectation on both sides of line 6 in Algorithm 2, we get

$$
\begin{array}{l} \mathbb {E} \left(v _ {t}\right) = \mathbb {E} \left(\left(1 + \beta_ {2}\right) v _ {t - 1} + \beta_ {2} g _ {t} ^ {2}\right) \\ = \sum_ {i = 1} ^ {t} \left(1 + \beta_ {2}\right) ^ {t - i} \beta_ {2} \mathbb {E} \left(g _ {t} ^ {2}\right) \tag {4} \\ = \left[ \left(1 + \beta_ {2}\right) ^ {t} - 1 \right] \mathbb {E} \left(g _ {t} ^ {2}\right) \\ \end{array}
$$

To maintain an accurate second moment, we would naturally divide  $v_{t}$  by  $(1 + \beta_{2})^{t} - 1$  in line 7. However, it's worth mentioning that we do not include a first moment correction term  $(1 - \beta_1^t)$  as Kingma & Ba (2015) did for the following reason. Consider the Stochastic Gradient Descent with momentum(SGDM) algorithm and Adam's first moment,

$$
\begin{array}{l} \text {S G D M :} \quad m _ {t} = \gamma m _ {t - 1} + g _ {t} = \sum_ {i = 1} ^ {t} \gamma^ {t - i} g _ {i} \\ \text {A d a m :} \quad m _ {t} = \beta_ {1} m _ {t - 1} + (1 - \beta_ {1}) g _ {t} = (1 - \beta_ {1}) \sum_ {i = 1} ^ {t} \beta_ {1} ^ {t - i} g _ {i} \\ \end{array}
$$

It can be observed that they have the same form except for the scaling constant  $1 - \beta_{1}$ , and therefore the first order bias correction term is counter intuitive. We change the scaling constant from  $1 - \beta_{1}$  to  $1 - \beta_{3}$  in line 5 in our algorithm to obtain a more general form of the moment expression, and when  $\beta_{3} \neq \beta_{1}$ , it helps to scale the hyper-parameters (such as step size, weight decay) of adaptive algorithms to the same size as SGD. Next we show that our algorithm ensures the positive semi-definiteness of  $\Gamma_{t}$  and hence does not have the non-convergence issue of Adam. Consider the following lemma which leads to the conclusion that  $\Gamma_{t}$  is positive semi-definite in our algorithm,

Lemma 4.1 Algorithm 2 ensures that the matrix  $\frac{V_t}{\alpha_t^2} - \frac{V_{t-1}}{\alpha_{t-1}^2} \succeq 0$

Finally, we provide the convergence analysis of our algorithm in both convex and non-convex settings. Using the analysis framework by Zinkevich (2003), the following theorem states that we are able to obtain a regret bound of  $\mathcal{O}(\sqrt{T})$ , which is the same as the results of Reddi et al. (2018). A domain  $\mathcal{F}$  is said to have bounded diameter if  $\| x - y\|_{\infty}\leq D_{\infty},\forall x,y\in \mathcal{F}$  for some  $D_{\infty}\in \mathbb{R}$ .

Theorem 4.1 Let  $\{x_{t}\}$  and  $\{v_{t}\}$  be the sequences obtained from Algorithm 2,  $\alpha_{t} = \alpha /\sqrt{t},\beta_{1,1} = \beta_{1},\beta_{1,t}\leq \beta_{1}$ , for all  $t\in [T]$  and  $\beta_{2t} = \beta_2 / t,\beta_{3t} = 1 - 1 / \sqrt{t}$ . Assume that  $\mathcal{F}$  has bounded diameter  $D_{\infty}$  and  $\| \nabla f_t(x)\| \leq G_\infty$  for all  $t\in [T]$  and  $x\in \mathcal{F}$ . Then for  $x_{t}$  generated using Algorithm 2, we have the following bound on the regret.

$$
R _ {T} \leq \frac {D _ {\infty^ {2}}}{2 \alpha_ {T} (1 - \beta_ {1})} \sum_ {i = 1} ^ {d} \hat {v} _ {T, i} ^ {1 / 2} + \frac {D _ {\infty} ^ {2}}{2 (1 - \beta_ {1})} \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {d} \frac {\beta_ {1 t} \hat {v} _ {t , i} ^ {1 / 2}}{\alpha_ {t}} + \frac {\alpha C \sqrt {1 + \log T}}{(1 - \beta_ {1}) ^ {3} \sqrt {\beta_ {2}}} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2} \quad (5)
$$

The following corollary follows naturally from the above theorem.

Corollary 4.1 Suppose  $\beta_{1t} = \beta_1\lambda^{t - 1}$  in Theorem 4.1, then we have

$$
R _ {T} \leq \frac {D _ {\infty} {} ^ {2} \sqrt {T}}{2 \alpha (1 - \beta_ {1})} \sum_ {i = 1} ^ {d} \hat {v} _ {T, i} ^ {1 / 2} + \frac {d \beta_ {1} D _ {\infty} ^ {2} G _ {\infty}}{2 \alpha (1 - \beta_ {1}) (1 - \lambda) ^ {2}} + \frac {\alpha C \sqrt {1 + \log T}}{(1 - \beta_ {1}) ^ {3} \sqrt {\beta_ {2}}} \sum_ {i = 1} ^ {d} \| g _ {1: T, i} \| _ {2} \tag {6}
$$

Analyzing optimization algorithms in a non-convex setting is slightly different from the convex case, where instead of the average regret, stationarity in gradient is utilized to show convergence in time. Following Chen et al. (2019), suppose we are minimizing a cost function  $f$  that satisfies the following three assumptions

A1.  $f$  is differentiable and has  $L$ -Lipschitz gradient, i.e.  $\forall x, y, \| \nabla f(x) - \nabla f(y) \| \leq L \| x - y \|$ . and  $f(x^{*}) > \infty$  where  $x^{*}$  is the optimal solution.

![](images/d69a6b93c9b5427de9bd6ef77cadb8eed2b1689e286af15c42da82028c59a1a5.jpg)  
(a) Training Accuracy for  $L_{2}$  Regularization

![](images/b2e2cd8f587b127e4a0b9926f6e56d885227354c888a6fd243b08bf6a94f2c26.jpg)  
(b) Testing Accuracy for  $L_{2}$  Regularization

![](images/433de200dd61a3042f7a641856b8f8035f8323c5a2b1b35701a50d2e696c533d.jpg)  
(c) Training Accuracy for Weight Decay

![](images/13ea8aedca82731b9085445b3484c328ad29c06621bdd3e50a742874ebddb317.jpg)  
(d) Testing Accuracy for Weight Decay  
Figure 1: Training and Testing Accuracy on CIFAR-10

<table><tr><td>Method</td><td>Top 1 Acc</td><td>Method</td><td>Top 1 Acc</td></tr><tr><td>Adam</td><td>90.96</td><td>AdamW, wd=1e-4</td><td>91.86</td></tr><tr><td>AdaGrad</td><td>86.56</td><td>AdamW, wd=5e-4</td><td>92.12</td></tr><tr><td>AMSGrad</td><td>90.98</td><td>SGDM</td><td>92.30</td></tr><tr><td>RMSProp</td><td>89.64</td><td>AdaX-W, wd=5e-4(ours)</td><td>92.32</td></tr><tr><td>AdaX(ours)</td><td>91.60</td><td>AdaX-W, wd=1e-3(ours)</td><td>92.51</td></tr></table>

Table 2: Validation accuracy on CIFAR-10.

A2. Suppose that the true gradients and the noisy gradients are bounded i.e.  $\| \nabla f(x_{t})\| \leq G_{\infty}, \| g_{t}\| \leq G_{\infty}, \forall t\geq 1$ . Also,  $\| \alpha_{t}\frac{m_{t}}{\sqrt{v_{t}}}\| \leq G$  for some  $G > 0$  
A3. The noisy gradient is unbiased and the noise is independent, i.e.  $g_{t} = \nabla f(x_{t}) + \eta_{t}, \mathbb{E}[\eta_{t}] = 0$  and  $\eta_{i}$  is independent of  $\eta_{j}$  if  $i \neq j$ .

then we would obtain the following theorem and corollary, which prove that AdaX converges with a speed close to AMSGrad as mentioned by Chen et al. (2019).

Theorem 4.2 Let  $\{x_{t}\}$  and  $\{v_{t}\}$  be the sequences obtained from Algorithm 2,  $\alpha_{t} = \alpha /\sqrt{t},\beta_{1,1} = \beta_{1},\beta_{1,t}\leq \beta_{1}$ , for all  $t\in [T]$  and  $\beta_{2t} = \beta_2 / t$ $\beta_{3t} = \beta_{1}$ . Assume that  $\| \nabla f_t(x)\| \leq G_\infty$  for all  $t\in [T]$  and  $x\in \mathcal{F}$ . Then for  $x_{t}$  generated using Algorithm 2, we have the following bound.

$$
\min  _ {t \in [ T ]} \mathbb {E} \left[ \| \nabla f (x _ {t}) \| ^ {2} \right] \leq \frac {G _ {\infty}}{\alpha \log (1 + T)} \left(\frac {C _ {1} G _ {\infty} ^ {2} \alpha^ {2}}{c ^ {2}} + \frac {C _ {2} d \alpha}{c} + \frac {C _ {3} d \alpha^ {2}}{c ^ {2}} + C _ {4}\right) \tag {7}
$$

where  $C_1, C_2, C_3, C_4$  are constants independent of  $T$

Corollary 4.2 Suppose  $\beta_{2t} = \beta_2$ , with the other assumptions same as in Theorem 4.2, we have

$$
\min  _ {t \in [ T ]} \mathbb {E} \left[ \| \nabla f (x _ {t}) \| ^ {2} \right] \leq \frac {G _ {\infty}}{\alpha \sqrt {T}} \left(\frac {C _ {1} G _ {\infty} ^ {2} \alpha^ {2}}{c ^ {2}} + \frac {C _ {2} d \alpha}{c} + \frac {C _ {3} d \alpha^ {2}}{c ^ {2}} + C _ {4}\right) \tag {8}
$$

# 5 EXPERIMENTS

In this section, we evaluate the performance of AdaX on various tasks in comparison with SGD with Nesterov momentum (SGDM), Adam(W), and many other common optimizers. The implementation of AdaX consists of two parts, AdaX and AdaX-W, representing using  $L_{2}$  regularization and standard weight decay in the algorithm respectively as discussed in Loshchilov & Hutter (2019). We relegate the detailed implementation of AdaX to the Appendix. We show that AdaX, combined with a proper weight decay, is capable of performing better than Adam and SGDM in many tasks.

# 5.1 CONVOLUTIONAL NEURAL NETWORK ON CIFAR-10

Using ResNet-20 proposed by He et al. (2016), we verified the performance of AdaX on CIFAR-10 (Krizhevsky et al., 2009) image classification task. In our experiments, we utilized a learning

![](images/aedf2fe2b34d55f9fb9f275d68e6c8679f441ed0848d833c25d2c66c5d3db731.jpg)  
(a) Training Top-1 Accuracy on ImageNet

![](images/d49539651ef2d8602655e7180c2d5d317495b314022c245feb69eb57b3ed2593.jpg)  
(b) Testing Top-1 Accuracy on ImageNet  
Figure 2: Training and Testing Results on ImageNet.

(a) Validation accuracy on ImageNet and IoU on VOC2012 Segmentation  

<table><tr><td>Method</td><td>ImageNet Top 1 Acc</td><td>VOC2012 IoU</td></tr><tr><td>AdamW</td><td>66.92</td><td>74.62</td></tr><tr><td>SGDM</td><td>69.90</td><td>76.28</td></tr><tr><td>AdaX-W(ours)</td><td>69.87</td><td>76.53</td></tr></table>

(b) Validation perplexity on One Billion Word for language modeling.  

<table><tr><td>Method</td><td>Validation PPL</td></tr><tr><td>Adam</td><td>36.90</td></tr><tr><td>AdaX(ours)</td><td>35.22</td></tr></table>

Table 3: Performance of AdaX on ImageNet, VOC2012 Segmentation and One Billion Words

rate schedule that the initial step size was scaled down by 0.1 and 0.01 at the 100-th and the 150-th epoch. The experimental results in Table 2 correspond to our theoretical finding that Adam is actually taking steps that are "too large" and will potentially converge to local minimum at the end.

$L_{2}$  Regularization. We used  $L_{2}$  regularization of strength 5e-4 for all the optimizers. As shown in Figure 1a and 1b, although AdaX was relatively slow at the beginning compared with other optimizers, its testing accuracy quickly caught up with the others after the first learning rate decrease and became the highest (91.6) at last.

Weight Decay. The baseline was trained with SGDM with weight decay 5e-4. Adam with weight decay, named AdamW (Loshchilov & Hutter, 2019) was also trained for comparisons. Although AdamW with 1e-4 weight decay converged much faster than SGDM and AdaX-W, its final accuracy could not catch up with the other two (see Figure 1c & 1d). A higher weight decay could potentially help AdamW achieve a better performance (92.1), but it was as slow as SGDM. On the other hand, AdaX-W with step size rate 0.5 and 5e-4 weight decay converged fast and yielded the same performance as SGDM(92.32). The best result we obtained was AdaX-W with step size 0.25 and 1e-3 weight decay, which resulted in 92.51 Top-1 accuracy, even slightly higher than SGDM.

# 5.2 CONVOLUTIONAL NEURAL NETWORK ON IMAGENET

We also conducted experiments to examine the performance of AdaX-W on ImageNet (Deng et al., 2009). SGDM, AdaX-W, and AdamW were used to train a ResNet-18 model on ImageNet, with a standard 1e-4 weight decay and 0.1, 0.5, 1e-3 step sizes as in CIFAR-10 respectively. A warm up scheme was applied in the initial 25k iterations (Goyal et al., 2017), and then the step size was multiplied by 0.1 at the 150k, 300k and 450k-th iteration steps. As observed from Figure 2, although AdamW was fast at the beginning of training, its test accuracy stagnated after the second learning rate decrease. AdaX-W, on the other hand, converged faster than SGDM without loss of testing accuracy (69.87), as shown in Table 3a.

# 5.3 RECURRENT NEURAL NETWORK ON LANGUAGE MODELING

AdaX has also been validated on Billion Word (Chelba et al., 2013) dataset of language modeling task. For the Billion Word, we used a two-layer LSTMs with 2048 hidden states and sampled softmax. The global experiment settings in the released public code Rdspring1 was adopted in this

![](images/546f74bccaed2e4b7843be3b224f01a1e0bd2f1558a15e79d76efde67d0816b8.jpg)  
(a) Training Dynamics on One Billion Word.

![](images/3dd91fb436ce48948b01ca8b110f05fc152c66f3f15926176e24bfa7084adc36.jpg)  
(b) Training Loss on VOC2012 Segmentation  
Figure 3: (a) Traning loss curves for Adam and AdaX on One Billion Word. (b, c) Training Loss and Testing Results on VOC2012 Segmentation task. In (c), dashed lines are mean accuracy values and solid lines are Intersection over Union (IoU) values

![](images/4a1fe496f99ba2360870c41de2867b67cea727db9eccd874b9287530b1d9dff1.jpg)  
(c) Testing IoU and Mean Accuracy

study. For both vanilla Adam and AdaX, the LSTMs were trained using for 5 epochs, with learning rate decaying to 1e-8 linearly. Similarly, the weight decay for AdaX was set to 0. Note that the regular SGD is not suitable in this task, so it is not included in the comparison.

The training loss and the validation perplexity is shown in Figure 3a and Table 3b. We can see that the AdaX outperforms the Adam baseline by a significant margin (35.22 vs. 36.90). Moreover, similar to the effect on image classification tasks described above, AdaX starts a little slower at the early stage, but it soon surpasses Adam on both training and validation performance.

# 5.4 TRANSFER LEARNING

Finally, to further examine the robustness of AdaX in transfer learnings such as semantic segmentation, we evaluated its performance on the PASCAL VOC2012 augmented dataset (Everingham et al., 2014) (Hariharan et al., 2011). The classic Deeplab-ASPP model (Chen et al., 2016) was adopted with a ResNet-101 backbone pretrained on MS-COCO dataset(Lin et al., 2014). Adaptive methods are seldom used in semantic segmentation tasks because of their bad performances, but AdaX can surprisingly be applied to train these models as well. The initial step sizes for SGDM, AdaX-W and AdamW were set to  $2.5\mathrm{e - }4$  , 1e-3, and 1e-6 respectively and all other parameters stuck to the default setting in Chen et al. (2016). We evaluated the model's performance at the 5k, 10k, 15k and 20k iterations using intersection over union (IoU) and mean accuracy. As can be observed in Figure 3c, AdaX-W trained faster than SGDM and obtained a higher IoU (76.5) at the same time. On the other hand, AdamW was not capable of obtaining comparable results.

The experiments shown above verify the effectiveness of AdaX, showing that the accumulated long-term past gradient information can enhance the model performance, by getting rid of the second moment instability in vanilla Adam. It is also worth noticing that the computational cost for each step of AdaX and Adam are approximately the same, as they both memorize the first and second momentum in the past. Therefore AdaX enables one to get higher performance than Adam in various tasks using the same training budget.

# 6 CONCLUSION

In this paper, we present a novel optimization algorithm named AdaX in order to improve the performance of traditional adaptive methods. We first extend the non-convergence issue of Adam to a non-convex case, and show that Adam's fast convergence impairs its convergence. We then propose our variant of Adam, analyze its convergence, and evaluate its performance on various learning tasks. Our theoretical analysis and experimental results both show that AdaX is more stable and performs better than Adam in various tasks. In the future, more experiments still need to be performed to evaluate the overall performance of AdaX and AdaX-W. Moreover, our paper is a first step into designing adaptive learning rates in ways different from simple and exponential average methods. Other new and interesting designs should also be examined. We believe that new adaptive algorithms that outperform AdaX in convergence rate and performance still exist and remain to explore.

# REFERENCES

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Philipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.  
Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40:834-848, 2016.  
Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of a class of adam-type algorithm for non-convex optimization. Proceedings of 7th International Conference on Learning Representations(ICLR), 2019.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. Imagenet: A large-scale hierarchical image database. in 2009 IEEE conference on computer vision and pattern recognition. IEEE, 40:248255, 2009.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research (JMLR), pp. 12:2121-2159, 2011.  
Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The Pascal visual object classes challenge: A retrospective. International Journal of Computer Vision(IJCV), 2014.  
Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. International Conference of Computer Vision (ICCV), 2011.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
Haiwen Huang, Chang Wang, and Bin Dong. Nostalgic adam: Weighting more of the past gradients when designing the adaptive learning rate. arXiv preprint arXiv: 1805.07557, 2019.  
Diederik P Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015.  
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research). 2009.  
Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312, 2014.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. Proceedings of 7th International Conference on Learning Representations (ICLR), 2019.  
Liangchen Luo, Yuanhao Xiong, Yan Liu, and Xu Sun. Adaptive gradient methods with dynamic bound of learning rate. Proceedings of 7th International Conference on Learning Representations, 2019.  
H Brendan Mcmahan and Matthew Streeter. Adaptive bound optimization for online convex optimization. Proceedings of the 23rd Annual Conference On Learning Theory (COLT), pp. 244-256, 2010.

Y. Nesterov. A method for unconstrained convex minimization problem with the rate of convergence  $o(1 / k^2)$ . Doklady AN USSR, pp. (269), 543-547, 1983.  
Boris Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, pp. 4(5):1-17, 1964.  
Rdspring1. Pytorch gbw lm. https://github.com/rdspring1/PyTorch_GBW_LM.  
Sashank J. Reddi, Stayen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. Proceedings of the 6th International Conference on Learning Representations (ICLR), 2018.  
Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, pp. 22(3):400-407, 1951.  
Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. arXiv preprint arXiv: 1804.04235), 2018.  
Tijmen Tieleman and Geoffrey Hinton. Rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, pp. 4(2):26-31, 2012.  
Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. Advances in Neural Information Processing Systems 30, pp. 4148-4158, 2017.  
Matthew D. Zeiler. Adadelta: An adaptive learning rate method. arXiv preprint arXiv:1212.5701, 2012.  
Zhiming Zhou, Qingru Zhang, Guansong Lu, Hongwei Wang, Weinan Zhang, and Yong Yu. Adashift: Decorrelation and convergence of adaptive learning rate methods. Proceedings of 7th International Conference on Learning Representations (ICLR), 2019.  
Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. International Conference on Machine Learning (ICML), 2003.
