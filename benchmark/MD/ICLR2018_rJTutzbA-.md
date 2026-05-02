# ON THE INSUFFICIENCY OF EXISTING MOMENTUM SCHEMES FOR STOCHASTIC OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Momentum based stochastic gradient methods such as heavy ball (HB) and Nesterov's accelerated gradient descent (NAG) method are widely used in practice for training deep networks and other supervised learning models, as they often provide significant improvements over stochastic gradient descent (SGD). Theoretically, these "fast gradient" methods have provable improvements over gradient descent only for the deterministic case, where the gradients are exact. In the stochastic case, the popular explanations for their wide applicability is that when these fast gradient methods are applied in the stochastic case, they partially mimic their exact gradient counterparts, resulting in some practical gain. This work provides a counterpoint to this belief by proving that there are simple problem instances where these methods cannot outperform SGD despite the best setting of its parameters. These negative problem instances are, in an informal sense, generic; they do not look like carefully constructed pathological instances. These results suggest (along with empirical evidence) that HB or NAG's practical performance gains are a by-product of minibatching.

Furthermore, this work provides a viable (and provable) alternative, which, on the same set of problem instances, significantly improves over HB, NAG, and SGD's performance. This algorithm, denoted as ASGD, is a simple to implement stochastic algorithm, based on a relatively less popular version of Nesterov's AGD. Extensive empirical results in this paper show that ASGD has performance gains over HB, NAG, and SGD.

# 1 INTRODUCTION

First order optimization methods, which access a function (to be optimized) through its gradient or an unbiased approximation of its gradient, are the workhorses for modern large scale optimization problems, which include training the current state-of-the-art deep neural networks. Gradient descent (Cauchy, 1847) is the simplest first order method that is used heavily in practice. However, it is known that for the class of smooth convex functions, gradient descent is suboptimal (Nesterov, 2004) and there exists a class of algorithms called fast gradient/momentum based methods which achieve optimal convergence guarantees. The heavy ball method (Polyak, 1964) and Nesterov's accelerated gradient descent (Nesterov, 1983) are two of the most popular methods in this category.

On the other hand, training deep neural networks on large scale datasets have been possible through the use of Stochastic Gradient Descent (SGD) (Robbins & Monro, 1951), which samples a random subset of training data to compute gradient estimates that are then used to optimize the objective function. The advantages of SGD for large scale optimization and the related issues of tradeoffs between computational and statistical efficiency was highlighted in Bottou & Bousquet (2007).

The above mentioned theoretical advantages of fast gradient methods (Polyak, 1964; Nesterov, 1983) (albeit for smooth convex problems) coupled with cheap to compute stochastic gradient estimates led to the influential work of Sutskever et al. (2013), which demonstrated the empirical advantages possessed by SGD when augmented with the momentum machinery. This work has led to the wide spread adoption of momentum methods for training deep neural networks; so much so that, in the context of neural network training, gradient descent often refers to momentum methods.

But, there is a subtle difference between classical momentum methods and their implementation in practice - classical momentum methods work in the exact first order oracle model (Nesterov, 2004),

i.e., they employ exact gradients (computed on the full training dataset), while in practice (Sutskever et al., 2013), they are implemented with stochastic gradients (estimated from a randomly sampled subset/mini-batch of the training data). This leads to a natural question:

Are momentum methods optimal even in the stochastic first order oracle (SFO) model, where we access stochastic gradients computed on a small constant sized minibatches (or a batchsize of 1?)

Even disregarding the question of optimality of momentum methods in the SFO model, it is not even known if momentum methods (say, Polyak (1964); Nesterov (1983)) provide any provable improvement over SGD in this model. While these are open questions, a recent effort of Jain et al. (2017) showed that improving upon SGD (in the stochastic first order oracle) is rather subtle as there exists problem instances in SFO model where it is not possible to improve upon SGD, even information theoretically. Jain et al. (2017) studied a variant of Nesterov's accelerated gradient updates (Nesterov, 2012) for stochastic linear regression and show that their method improves upon SGD wherever it is information theoretically admissible. Through out this paper, we refer to the algorithm of (Jain et al., 2017) as Accelerated Stochastic Gradient Method (ASGD) while we refer to a stochastic version of the most widespread form of Nesterov's method (Nesterov, 1983) as NAG; HB denotes a stochastic version of the heavy ball method. Critically, while the result of Jain et al. (2017) shows that ASGD improves on SGD in any information-theoretically admissible regime, it is still not known whether HB and NAG can achieve a similar performance gain.

A key contribution of this work is to show that HB does not provide similar performance gain to ASGD even when it is informationally-theoretically admissible. That is, we provide a problem instance where it is indeed possible to improve upon SGD (and in fact ASGD achieves this improvement), but HB cannot achieve any improvement over SGD. We validate this claim empirically as well. In fact, we provide empirical evidence to the claim that NAG also do not achieve any improvement over SGD for several problems where ASGD can still achieve better rates of convergence.

This raises a question about why HB and NAG provide better performance than SGD in practice (Sutskever et al., 2013), especially for training deep networks. Our conclusion (that is well supported by our theoretical result) is that HB and NAG's improved performance is attributed to mini-batching and hence, these methods will often struggle to improve over SGD with small constant batch sizes. Note that this result indicates that there is a natural tension between the gains offered by momentum methods with larger batches and the superior generalization properties offered by training with small mini-batches (Keskar et al., 2016), which is a regime that may not be amenable for HB/NAG to improve upon SGD. This is in stark contrast to methods like ASGD, which are designed to improve over SGD across small or large mini-batch sizes. In fact, based on our experiments, we observe that on the task of training deep residual networks (He et al., 2016a) on the cifar-10 dataset, we note that ASGD offers noticeable improvements by achieving  $5 - 7\%$  better test error over HB and NAG even with commonly used batch sizes like 128 during the initial stages of the optimization.

# 1.1 CONTRIBUTIONS

The contributions of this paper are as follows.

1. In Section 3, we prove that HB is not optimal in the SFO model. In particular, there exist linear regression problems for which the performance of HB (with any step size and momentum) is either the same or worse than that of SGD while ASGD improves upon both of them.  
2. Experiments on several different linear regression problems suggest that the suboptimality of HB in the SFO model is not restricted to special cases – it is rather widespread. Empirically, the same holds true for NAG as well (Section 5).  
3. The above observations suggest that the only reason for the superiority of momentum methods in practice is minibatching, which reduces the variance in stochastic gradients and moves closer to exact first order oracle. This conclusion is supported by empirical evidence through training deep residual networks on CIFar-10, with a batch size of 8 (see Section 5.3).  
4. We present an intuitive and easier to tune version of ASGD (see Section 4) and show that ASGD can provide significantly faster convergence to a reasonable accuracy than SGD, HB, NAG, while still providing asymptotically optimal accuracy.

Hence, the take-home message of this paper is: HB and NAG are not optimal in the SFO model. The only reason for the superiority of momentum methods in practice is minibatching. ASGD provides a distinct advantage in training deep networks over SGD, HB and NAG.

# Algorithm 1 HB: Heavy ball with a SFO

Require: Initial  $w_0$ , stepsize  $\delta$ , momentum  $\alpha$   
1:  $w_{-1} \gets w_0$ ;  $t \gets 0$  /*Set  $w_{-1}$  to  $w_0\*/$

2: while  $w_{t}$  not converged do

3:  $w_{t + 1}\gets w_t - \delta \cdot \widehat{\nabla} f_t(w_t) + \alpha \cdot (w_t - w_{t - 1})$  /*Sum of stochastic gradient step and momentum*/

4:  $t\gets t + 1$

Ensure:  $w_{t}$  /*Return the last iterate*/

# Algorithm 2 NAG: Nesterov's AGD with a SFO

Require: Initial  $w_0$ , stepsize  $\delta$ , momentum  $\alpha$

1:  $v_{0} \gets w_{0}; t \gets 0$  /*Set  $v_{0}$  to  $w_{0}^{*} /$  
2: while  $w_{t}$  not converged do  
3:  $v_{t + 1} \gets w_t - \delta \cdot \widehat{\nabla} f_t(w_t) / {}^*\mathrm{SGD~step}^* /$  
4:  $w_{t + 1} = (1 + \alpha)v_{t + 1} - \alpha v_t / * \text{Sum of SGD}$  step and previous iterate\*/  
5:  $t\gets t + 1$

Ensure:  $w_{t}$  /*Return the last iterate*/

# 2 NOTATION

We denote matrices by bold-face capital letters and vectors by lower-case letters.  $f(w) = 1 / n\sum_{i}f_{i}(w)$  denotes the function to optimize w.r.t.  $w$ .  $\nabla f(w)$  denote gradient of  $f$  at  $w$  while  $\widehat{\nabla} f_t(w)$  denote a stochastic gradient of  $f$ . That is,  $\widehat{\nabla} f_t(w_t) = \nabla f_{i_t}(w)$  where  $i_t$  is sampled uniformly at random from  $[1,\dots ,n]$ . For linear regression,  $f_{i}(w) = (y - \langle w,x_{i}\rangle)^{2}$  where  $y$  is the target variable and  $x\in \Re^d$  is the covariate, and  $\widehat{\nabla} f_{t}(w_{t}) = -(y_{t} - \langle w_{t},x_{t}\rangle)x_{t}$ . In this case,  $\mathbf{H} = \mathbb{E}\left[xx^{\top}\right]$  denotes the Hessian of  $f$  and  $\kappa = \frac{\lambda_1(\mathbf{H})}{\lambda_d(\mathbf{H})}$  denotes its condition number.

Algorithm 1 provides a pseudo-code of HB method (Polyak, 1964).  $w_{t} - w_{t-1}$  is the momentum term and  $\alpha$  denotes the momentum parameter. Next iterate  $w_{t+1}$  is obtained by a linear combination of the SGD update and the momentum term. Algorithm 2 provides pseudo-code of a stochastic version of the most commonly used form of Nesterov's accelerated gradient descent (Nesterov, 1983)

# 3 SUBOPTIMALITY OF HEAVY BALL METHOD

In this section, we show that there exist linear regression problems where the performance of HB (Algorithm 1) is no better than that of SGD, while ASGD significantly improves upon this performance. Let us now describe the problem instance.

Fix  $w^{*}\in \mathbb{R}^{2}$  and let  $(\mathbf{x},\mathbf{y})\sim \mathcal{D}$  be the following distribution over  $\mathbb{R}^2$  ..

$$
x = \left\{ \begin{array}{l} \sigma_ {1} \cdot z \cdot e _ {1} \text {w . p .} 0. 5 \\ \sigma_ {2} \cdot z \cdot e _ {2} \text {w . p .} 0. 5, \end{array} \right. \qquad \text {a n d} \qquad y = \langle w ^ {*}, x \rangle ,
$$

where  $e_1, e_2 \in \mathbb{R}^2$  are canonical basis vectors,  $\sigma_1 > \sigma_2 > 0$ . Let  $z$  be a random variable such that  $\mathbb{E}\left[z^2\right] = 2$  and  $\mathbb{E}\left[z^4\right] = 2c \geq 4$ . Hence, we have:  $\mathbb{E}\left[(x^{(i)})^2\right] = \sigma_i^2$ ,  $\mathbb{E}\left[(x^{(i)})^4\right] = c\sigma_i^4$ , for  $i = 1, 2$ . Now, our goal is to minimize:

$$
f (w) \stackrel {{\mathrm {d e f}}} {{=}} \mathbb {E} \left[ \left(\langle w ^ {*}, x \rangle - y\right) ^ {2} \right],   \text {H e s s i a n}   \mathbf {H} \stackrel {{\mathrm {d e f}}} {{=}} \mathbb {E} \left[ x x ^ {\top} \right] = \left[ \begin{array}{c c} \sigma_ {1} ^ {2} & 0 \\ 0 & \sigma_ {2} ^ {2} \end{array} \right].
$$

Let  $\kappa$  and  $\tilde{\kappa}$  denote the computational and statistical condition numbers - see Jain et al. (2017) for definitions. For the problem above, we have  $\kappa = \frac{2c\sigma_1^2}{\sigma_2^2}$  and  $\tilde{\kappa} = 2c$ . Then we obtain following convergence rates for SGD and ASGD when applied to the above given problem instance:

Corollary 1 (of Theorem 1 of Jain et al. (2016)). Let  $w_{t}^{SGD}$  be the  $t^{\text{th}}$  iterate of SGD on the above problem with starting point  $w_{0}$  and stepsize  $\frac{1}{c\sigma_{1}^{2}}$ . The error of  $w_{t}^{SGD}$  can be bounded as,

$$
\mathbb {E} \left[ f \left(w _ {t} ^ {S G D}\right) \right] - f \left(w _ {*}\right) \leq \exp \left(\frac {- t}{\kappa}\right) \left(f \left(w _ {0}\right) - f \left(w _ {*}\right)\right).
$$

On the other hand, ASGD achieves the following superior rate.

Corollary 2 (of Theorem 1 of Jain et al. (2017)). Let  $w_{t}^{ASGD}$  be the  $t^{\text{th}}$  iterate of ASGD on the above problem with starting point  $w_{0}$  and appropriate parameters. The error of  $w_{t}^{ASGD}$  can be bounded as,

$$
\mathbb {E} \left[ f \left(w _ {t} ^ {A S G D}\right) \right] - f \left(w _ {*}\right) \leq \mathrm {p o l y} (\kappa) \exp \left(\frac {- t}{\sqrt {\kappa \tilde {\kappa}}}\right) \left(f \left(w _ {0}\right) - f \left(w _ {*}\right)\right).
$$

# Algorithm 3 Accelerated stochastic gradient descent - ASGD

Input: Initial  $w_0$ , short step  $\delta$ , long step parameter  $\kappa \geq 1$ , statistical advantage parameter  $\xi \leq \sqrt{\kappa}$

1:  $\bar{w}_0\gets w_0;t\gets 0$

/*Set running average to  $w_0^*$

2:  $\alpha \gets 1 - \frac{\xi}{\kappa}$

/*Set momentum value*/

3: while  $w_{t}$  not converged do

4:  $\bar{w}_{t+1} \gets \alpha \cdot \bar{w}_t + (1 - \alpha) \cdot \left(w_t - \frac{\kappa \cdot \delta}{0.7} \cdot \widehat{\nabla} f_t(w_t)\right) / * \text{Update the running average as a weighted average of previous running average and a long step gradient */}$  
5:  $w_{t + 1}\gets \frac{0.7}{0.7 + (1 - \alpha)}\cdot \left(w_t - \delta \cdot \widehat{\nabla} f_t(w_t)\right) + \frac{1 - \alpha}{0.7 + (1 - \alpha)}\cdot \bar{w}_{t + 1}$  /*Update the iterate as weighted average of current running average and short step gradient*/  
6:  $t\gets t + 1$

Output:  $w_{t}$

/*Return the last iterate*/

Note that  $\tilde{\kappa} = 2c$  is a constant while  $\kappa = \frac{2c\sigma_1^2}{\sigma_2^2}$  can be arbitrarily large. Hence, ASGD improves upon rate of SGD by a factor of  $\sqrt{\kappa}$ . The following proposition, which is the main result of this section, establishes that HB (Algorithm 1) cannot provide a similar improvement. In fact, we show that despite selecting best parameters for HB, it's convergence rate is exactly same as that SGD (Corollary 1), up to constants.

Proposition 3. Let  $w_{t}^{HB}$  be the  $t^{th}$  iterate of HB (Algorithm 1) on the above problem with starting point  $w_{0}$ . For any choice of stepsize  $\delta$  and momentum  $\alpha \in [0,1]$ ,  $\exists T$  large enough such that  $\forall t \geq T$ , we have,

$$
\mathbb {E} \left[ f \left(w _ {t} ^ {H B}\right) \right] - f \left(w _ {*}\right) \geq C (\kappa , \delta , \alpha) \cdot \exp \left(\frac {- 5 0 0 t}{\kappa}\right) \left(f \left(w _ {0}\right) - f \left(w _ {*}\right)\right),
$$

where  $C(\kappa, \delta, \alpha)$  depends on  $\kappa, \delta$  and  $\alpha$  (but not on  $t$ ).

That is, to obtain  $\widehat{w}$  s.t.  $\| \widehat{w} - w^{*} \| \leq \epsilon$ , HB requires  $\Omega(\kappa \log \frac{1}{\epsilon})$  samples and iterations. On the other hand, ASGD can obtain  $\epsilon$ -approximation to  $w^{*}$  in  $\mathcal{O}(\sqrt{\kappa} \log \frac{1}{\epsilon})$  iterations. While we do not prove it theoretically, we observe empirically that for the same problem instance, NAG also obtains nearly same rate as HB and SGD.

# 4 ALGORITHM

In this section, we will present and explain an intuitive version of ASGD. Algorithm 3 presents the pseudocode. The algorithm takes three inputs: short step  $\delta$ , long step parameter  $\kappa$  and statistical advantage parameter  $\xi$ . The short step  $\delta$  is precisely the same as the step size in SGD, HB or NAG. For convex problems, this scales inversely with the smoothness of the function. The long step parameter  $\kappa$  is intended to give an estimate of the ratio of the largest and smallest curvatures of the function; for convex functions, this is just the condition number. The statistical advantage parameter  $\xi$  captures trade off between statistical and computational condition numbers – in the deterministic case,  $\xi = \sqrt{\kappa}$  and ASGD is equivalent to NAG, while in the high stochasticity regime,  $\xi$  is much smaller. The algorithm itself maintains two iterates: descent iterate  $w_{t}$  and a running average  $\bar{w}_{t}$ . The running average is a weighted average of the previous average and a long gradient step from the descent iterate, while the descent iterate is updated as a convex combination of short gradient step from the descent iterate and the running average. The idea is that since the algorithm takes a long step as well as short step and an appropriate average of both of them, it can make progress on different directions at a similar pace. Appendix B shows the equivalence between Algorithm 3 and ASGD as proposed in Jain et al. (2017). Note that the constant 0.7 appearing in Algorithm 3 has no special significance. Jain et al. (2017) require it to be smaller than  $\sqrt{1/6}$  but any constant smaller than 1 seems to work in experiments.

# 5 EXPERIMENTS

We now present our experimental results exploring performance of SGD, HB, NAG and ASGD. Our experiments are geared towards answering the following questions:

- Even for linear regression, is the suboptimality of HB restricted to specific distributions given in Section 3 or does it hold for more general distributions as well? Is the same true of NAG?

![](images/b2407dc5ebe344ad551c694c3d4c8304187fb16d819d80f2c0ed2a653ca2b3f8.jpg)  
Figure 1: Plot of  $1 / \mathrm{rate}$  (refer equation (1)) vs condition number  $(\kappa)$  for various methods for the linear regression problem. Discrete distribution in the left, Gaussian to the right.

![](images/a9fb4830fde3fe92f7e6e31b56617d136c065f61366efacbd78db0c50f9aa09a.jpg)

<table><tr><td>Algorithm</td><td>Slope – discrete</td><td>Slope – Gaussian</td></tr><tr><td>SGD</td><td>0.9302</td><td>0.8745</td></tr><tr><td>HB</td><td>0.8522</td><td>0.8769</td></tr><tr><td>NAG</td><td>0.98</td><td>0.9494</td></tr><tr><td>ASGD</td><td>0.5480</td><td>0.5127</td></tr></table>

Table 1: Slopes (i.e.  $\gamma$ ) obtained by fitting a line to the curves in Figure 1. A value of  $\gamma$  indicates that the error decays at a rate of  $\exp\left(\frac{-t}{\kappa^{\gamma}}\right)$ . A smaller value of  $\gamma$  indicates a faster rate of error decay.

- What is the reason for the superiority of HB and NAG in practice? Is it because momentum methods have better performance than SGD for stochastic gradients or due to minibatching? Does this superiority hold even for small minibatches?  
- How does the performance of ASGD compare to that of SGD, HB and NAG, especially while training deep networks?

Section 5.1 and parts of Section 5.2 address the first two questions. Section 5.2 and 5.3 address Question 2 partially and the last question. We use Matlab to conduct experiments presented in Section 5.1 and use PyTorch pyt for our deep networks related experiments.

# 5.1 LINEAR REGRESSION

In this section, we will present results on performance of the four optimization methods (SGD, HB, NAG, and ASGD) for linear regression problems. We consider two different class of linear regression problems, both of them in two dimensions. Given  $\kappa$  which stands for condition number, we consider the following two distributions:

Discrete:  $x = e_1$  w.p. 0.5 and  $x = \frac{2}{\kappa} \cdot e_2$  with 0.5;  $e_i$  is the  $i^{th}$  standard basis vector.

Gaussian:  $x \in \mathbb{R}^2$  is distributed as a Gaussian random vector with covariance matrix  $\begin{bmatrix} 1 & 0 \\ 0 & \frac{1}{\kappa} \end{bmatrix}$ .

We fix a randomly generated  $w^{*} \in \mathbb{R}^{2}$  and for both the distributions above, we let  $y = \langle w^{*}, x \rangle$ . We vary  $\kappa$  from  $\{2^4, 2^5, \dots, 2^{12}\}$  and for each  $\kappa$  in this set, we run 100 independent runs of all four methods, each for a total of  $t = 5\kappa$  iterations. We define that the algorithm converges if there is no error in the second half (i.e. after  $2.5\kappa$  updates) that exceeds the starting error - this is reasonable since we expect geometric convergence of the initial error.

Unlike ASGD and SGD, we do not know optimal learning rate and momentum parameters for NAG and HB in the stochastic gradient model. So, we perform a grid search over the values of the learning rate and momentum parameters. In particular, we lay a  $10 \times 10$  grid in  $[0,1] \times [0,1]$  for learning rate and momentum and run NAG and HB. Then, for each grid point, we consider the subset of 100 trials that converged and computed the final error using these. Finally, the parameters that yield the minimal error are chosen for NAG and HB, and these numbers are reported. We measure convergence performance of a method using:

$$
\operatorname {r a t e} = \frac {\log \left(f \left(w _ {0}\right)\right) - \log \left(f \left(w _ {t}\right)\right)}{t}, \tag {1}
$$

![](images/dc28a5545b4d6d404c3aa43be4c3e64cbcc60189f25de0cc9753fae7c574a191.jpg)  
Figure 2: Training loss (left) and test loss (right) while training deep autoencoder for mnist with minibatch size 8. Clearly, ASGD matches performance of NAG and outperforms SGD on the test data. HB also outperforms SGD.

![](images/194d518ccbcb34d4dffad329e9dcedc79b583b9f152670cb0985a814a73ae7a2.jpg)

We compute the rate (1) for all the algorithms with varying condition number  $\kappa$ . Given a rate vs  $\kappa$  plot for a method, we compute its slope (denoted as  $\gamma$ ) using linear regression. Table 1 presents the estimated slopes (i.e.  $\gamma$ ) for various methods for both the discrete and the Gaussian case. The slope values clearly show that the rate of SGD, HB and NAG have a nearly linear dependence on  $\kappa$  while that of ASGD seems to scale linearly with  $\sqrt{\kappa}$ .

# 5.2 DEEP AUTOENCODERS FOR MNIST

In this section, we present experimental results on training deep autoencoders for the mnist dataset, and we closely follow the setup of Hinton & Salakhutdinov (2006). This problem is a standard benchmark for evaluating the performance of different optimization algorithms e.g., Martens (2010); Sutskever et al. (2013); Martens & Grosse (2015); Reddi et al. (2017). The network architecture follows previous work Hinton & Salakhutdinov (2006) and is represented as  $784 - 1000 - 500 - 250 - 30 - 250 - 500 - 1000 - 784$  with the first and last 784 nodes representing the input and output respectively. All hidden/output nodes employ sigmoid activations except for the layer with 30 nodes which employs linear activations and we use MSE loss. Initialization follow the scheme of Martens (2010), also employed in Sutskever et al. (2013); Martens & Grosse (2015). We perform training with two minibatches sizes  $-1$  and 8. The runs with minibatch size of 1 were run for 30 epochs while the runs with minibatch size of 8 were run for 50 epochs. For each of SGD, HB, NAG and ASGD, a grid search over learning rate, momentum and long step parameter (whichever is applicable) was done and best parameters were chosen based on achieving the smallest training error in the same protocol followed by (say,)Sutskever et al. (2013). The grid was extended whenever the best parameter fell at the edge of a grid. For the parameters chosen by grid search, we perform 10 runs with different seeds and averaged the results. The results are presented in Figures 2 and 3. Note that the final loss values reported here are suboptimal compared to those in published literature e.g., Sutskever et al. (2013); while Sutskever et al. (2013) report results after 750000 updates with a large batch size of 200 (which implies a total of  $750000 \times 200 = 150\mathrm{M}$  gradient evaluations), whereas, our results are after 1.8M updates of SGD with a batch size 1 (which is just 1.8M gradient evaluations).

Effect of minibatch sizes: While HB and NAG decay the loss faster compared to SGD for a minibatch size of 8 (Figure 2), this superior decay rate does not hold for a minibatch size of 1 (Figure 3). This supports our intuitions from the stochastic linear regression setting, where we demonstrate that HB and NAG are suboptimal in the stochastic first order oracle model.

Comparison of ASGD with momentum methods: While ASGD performs slightly better than NAG for batch size 8 in the training error (Figure 2), ASGD decays the error at a faster rate compared to all the three other methods for a batch size of 1 (Figure 3).

# 5.3 DEEP RESIDUAL NETWORKS FOR CIFAR-10

In this section, we will present experimental results on training deep residual networks He et al. (2016b) with pre-activation blocks as introduced in He et al. (2016a) for classifying images in CIFar10 Krizhevsky & Hinton (2009); the network we use has 44 layers (dubbed preresnet-44). The code for these experiments was downloaded from pre. One of the most distinct characteristics of this

![](images/08beed2d4d168d51160d904736d941422b1c408d99c796e009182d1279b0c297.jpg)  
Figure 3: Training loss (left) and test loss (right) while training deep autoencoder for mnist with minibatch size 1. Interestingly, SGD, HB and NAG, all decrease the loss at a similar rate, while ASGD decays at a faster rate.

![](images/4edf4fc6dc7cbc76b3d708adb3d371e0e009278329d8cfa636d6ae2c3317ea24.jpg)

experiment compared to our previous experiments is learning rate decay. We use a validation set based decay scheme, wherein, after every 3 epochs, we decay the learning rate by a certain factor (which we grid search on) if the validation zero one error does not decrease by at least a certain amount (precise numbers are provided in the appendix since they vary across batch sizes). Due to space constraints, we present only a subset of training error plots. Please see Appendix C.3 for some more plots on training errors.

Effect of minibatch sizes: Our first experiment tries to understand how the performance of HB and NAG compare with that of SGD and how it varies with minibatch sizes. Figure 4 presents the test zero one error for minibatch sizes of 8 and 128. While training with batch size 8 was done for 40 epochs, with batch size 128, it was done for 120 epochs. We perform a grid search over all parameters for each of these algorithms. See Appendix C.3 for details on the grid search parameters. We observe that final error achieved by SGD, HB and NAG are all very close for both batch sizes. While NAG exhibits a superior rate of convergence compared to SGD and HB for batch size 128, this superior rate of convergence disappears for a batch size of 8.

![](images/e2b4a236165bfa86c0f385bddf43240914e8742cc98168ba0ae1d94065bf0156.jpg)  
Figure 4: Test zero one loss for batch size 128 (left), batch size 8 (center) and training function value for batch size 8 (right) for SGD, HB and NAG.

![](images/24c12459a32a323bc4750cb7495c4ee11e07f973c2fe7fc0cf94542510c7e253.jpg)

![](images/febc24ec8119f433d513c749fc500025c5f1e761ac0bbd967652f59dd83852a9.jpg)

Comparison of ASGD with momentum methods: The next experiment tries to understand how ASGD compares with HB and NAG. The errors achieved by various methods when we do grid search over all parameters are presented in Table 2. Note that the final test errors for batch size 128 are better than those for batch size 8 since the former was run for 120 epochs while the latter was run only for 40 epochs (due to time constraints).

<table><tr><td>Algorithm</td><td>Final test error - batch size 128</td><td>Final test error - batch size 8</td></tr><tr><td>SGD</td><td>8.32 ± 0.21</td><td>9.57 ± 0.18</td></tr><tr><td>HB</td><td>7.98 ± 0.19</td><td>9.28 ± 0.25</td></tr><tr><td>NAG</td><td>7.63 ± 0.18</td><td>9.07 ± 0.18</td></tr><tr><td>ASGD</td><td>7.23 ± 0.22</td><td>8.52 ± 0.16</td></tr></table>

Table 2: Final test errors achieved by various methods for batch sizes of 128 and 8. The hyperparameters have been chosen by grid search.

![](images/2a7b96160db7af2f0b0bae96685a84e2b9eb832121bbc0a3bb68358ff337d04e.jpg)  
Figure 5: Test zero one loss for batch size 128 (left), batch size 8 (center) and training function value for batch size 8 (right) for ASGD compared to HB. In the above plots, both ASGD and ASGD-Hb-Params refer to ASGD run with the learning rate and decay schedule of HB. ASGD-Fully-Optimized refers to ASGD where learning rate and decay schedule were also selected by grid search.

![](images/7c8ba5a986b8bb298b95451904f4a5469ac91508d77bf11f5d65b5a9520cb51c.jpg)

![](images/e46a16a15d434ea4ed03cb78195ba4cfbad488bde55582c444c016fff1ae1a8c.jpg)

![](images/c93598fcd29229ae6822de4945ed2bca0d38f734af55b310362c508a08601b0e.jpg)  
Figure 6: Test zero one loss for batch size 128 (left), batch size 8 (center) and training function value for batch size 8 (right) for ASGD compared to NAG. In the above plots, ASGD was run with the learning rate and decay schedule of NAG. Other parameters were selected by grid search.

![](images/b9f1cab2582635edc0a7e6861cd77f77a0bf3367299cf4bc9f5814be7c6dbd8d.jpg)

![](images/36b6398c0aa21cd2a8640b5c46ea9a988d3d59cbf956188cc98927304f8d8de5.jpg)

While the final error achieved by ASGD is similar to that of all other methods, we are more interested in understanding whether ASGD has a superior convergence speed. In order to do this experiment however, we need to address the issue of different learning rates used by various algorithms and different places where they decay learning rate. So, for each of HB and NAG, we choose the learning rate and decay factors by grid search, use these values for ASGD and do grid search only over long step parameter  $\kappa$  and momentum  $\alpha$  for ASGD. The results are presented in Figures 5 and 6. For batch size 128, ASGD decays error at a faster rate compared to both HB and NAG. For batch size 8, while we see a superior convergence of ASGD compared to NAG, we do not see this superiority over HB. The reason for this turns out to be that the learning rate for HB, which we also use for ASGD, turns out to be quite suboptimal for ASGD. So, for batch size 8, we also compare fully optimized (i.e., grid search over learning rate as well) ASGD with HB. The superiority of ASGD over HB is clear from this comparison. These results suggest that ASGD decays error at a faster rate compared to HB and NAG across different batch sizes.

# 6 RELATED WORK

First order oracle methods: The primary method in this family is Gradient Descent (GD) (Cauchy, 1847). As mentioned previously, GD is suboptimal for smooth convex optimization (Nesterov, 2004), and this is addressed using momentum methods such as the Heavy Ball method (Polyak, 1964) (for quadratics), and Nesterov's Accelerated gradient descent (Nesterov, 1983).

Stochastic first order methods: The simplest method employing the SFO is Stochastic Gradient Descent (Robbins & Monro, 1951)(SGD); the effectiveness of SGD has been immense, and its applicability goes well beyond optimizing convex objectives. Accelerating SGD is a tricky proposition given the instability of fast gradient methods in dealing with noise, as evidenced by several negative results which consider both statistical (Proakis, 1974; Polyak, 1987; Roy & Shynk, 1990) and adversarial errors (Devolder et al., 2014). A result of Jain et al. (2017) developed the first provably accelerated SGD method for linear regression inspired by a method of Nesterov (2012). Other schemes such as Ghadimi & Lan (2012; 2013); Dieuleveut et al. (2016), which indicate acceleration

is possible with noisy gradients do not hold in the SFO model satisfied by algorithms that are run in practice (see Jain et al. (2017) for more details).  
While HB (Polyak, 1964) and NAG (Nesterov, 1983) are known to be effective in case of exact first order oracle, for the SFO, the theoretical performance of HB and NAG is not well understood. Polyak (1987) describes HB to be rather brittle when provided with noisy gradient estimates.  
Practical methods for training deep networks: Momentum based methods employed with stochastic gradients (Sutskever et al., 2013) have become standard and very popular in practice. These schemes tend to outperform standard SGD on several important practical problems. As previously mentioned, we attribute this improvement to effect of minibatching rather than improvement by HB or NAG when working with stochastic gradients. Other schemes such as Adagrad (Duchi et al., 2011), RMSProp (Tieleman & Hinton, 2012), Adam (Kingma & Ba, 2014) represent an important and useful class of algorithms. The advantages offered by these methods are orthogonal to the advantages offered by fast gradient methods; it is an important direction to explore augmenting these methods with ASGD.

# 7 CONCLUSIONS AND FUTURE DIRECTIONS

In this paper, we show that the performance gain of HB over SGD in stochastic setting is attributable to minibatching rather than the algorithm's ability to accelerate with stochastic gradients. Concretely, we provide a formal proof that for several easy problem instances, HB does not outperform SGD despite large condition number of the problem; we observe this trend for NAG in our experiments. In contrast, ASGD (Jain et al., 2017) provides significant improvement over SGD for the same problem instances. We observe similar trend when training a resnet on CIFar-10 and an autoencoder on mnist. This work motivates several directions such as understanding the behavior of ASGD on other domains such as NLP, combining ASGD with adagrad (Duchi et al., 2011)/adam (Kingma & Ba, 2014) and possibly developing automatic tuning schemes similar to (Zhang et al., 2017).

# REFERENCES

Preresnet-44 for cifar-10. https://github.com/D-X-Y/ResNeXt-DenseNet. Accessed: 2017-10-25.  
Pytorch. https://github.com/pytorch. Accessed: 2017-10-25.  
Léon Bottou and Olivier Bousquet. The tradeoffs of large scale learning. In NIPS 20, 2007.  
Louis Augustin Cauchy. Méthode générale pour la résolution des systèmes d'équations simultanées. C. R. Acad. Sci. Paris, 1847.  
Olivier Devolder, Francois Glineur, and Yuri E. Nesterov. First-order methods of smooth convex optimization with inexact oracle. Mathematical Programming, 146:37-75, 2014.  
Aymeric Dieuleveut, Nicolas Flammarion, and Francis R. Bach. Harder, better, faster, stronger convergence rates for least-squares regression. CoRR, abs/1602.05419, 2016.  
John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121-2159, 2011.  
Saeed Ghadimi and Guanghui Lan. Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization: A generic algorithmic framework. SIAM Journal on Optimization, 2012.  
Saeed Ghadimi and Guanghui Lan. Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization, ii: shrinking procedures and optimal algorithms. SIAM Journal on Optimization, 2013.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In ECCV (4), Lecture Notes in Computer Science, pp. 630-645. Springer, 2016a.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pp. 770-778, 2016b.  
Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504-507, 2006.  
Prateek Jain, Sham M Kakade, Rahul Kidambi, Praneeth Netrapalli, and Aaron Sidford. Parallelizing stochastic approximation through mini-batching and tail-averaging. arXiv preprint arXiv:1610.03774, 2016.  
Prateek Jain, Sham M Kakade, Rahul Kidambi, Praneeth Netrapalli, and Aaron Sidford. Accelerating stochastic gradient descent. arXiv preprint arXiv:1704.08227, 2017.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. CoRR, abs/1609.04836, 2016.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
James Martens. Deep learning via hessian-free optimization. In International conference on machine learning, 2010.  
James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International conference on machine learning, 2015.  
Yurii Nesterov. A method of solving a convex programming problem with convergence rate o (1/k2). In Soviet Mathematics Doklady, volume 27, pp. 372-376, 1983.  
Yurii E. Nesterov. Introductory lectures on convex optimization: A basic course, volume 87 of Applied Optimization. Kluwer Academic Publishers, 2004.  
Yurii E. Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 22(2):341-362, 2012.  
Boris T Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.  
Boris T. Polyak. Introduction to Optimization. Optimization Software, 1987.  
John G. Proakis. Channel identification for high speed digital communications. IEEE Transactions on Automatic Control, 1974.  
Sashank Reddi, Manzil Zaheer, Suvrit Sra, Barnabas Poczos, Francis Bach, Ruslan Salakhutdinov, and Alexander Smola. A generic approach for escaping saddle points. arXiv preprint arXiv:1709.01434, 2017.  
Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, vol. 22, 1951.  
Sumit Roy and John J. Shynk. Analysis of the momentum lms algorithm. IEEE Transactions on Acoustics, Speech and Signal Processing, 1990.  
Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International conference on machine learning, pp. 1139-1147, 2013.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 2012.  
Jian Zhang, Ioannis Mitliagkas, and Christopher R. Yellowfin and the art of momentum tuning. CoRR, abs/1706.03471, 2017.
