# STRENGTH OF MINIBATCH NOISE IN SGD

Anonymous authors

Paper under double-blind review

# ABSTRACT

The noise in stochastic gradient descent (SGD), caused by minibatch sampling, is poorly understood despite its practical importance in deep learning. This work presents the first systematic study of the SGD noise and fluctuations close to a local minimum. We first analyze the SGD noise in linear regression in detail and then derive a general formula for approximating SGD noise in different types of minima. For application, our results (1) provide insight into the stability of training a neural network, (2) suggest that a large learning rate can help generalization by introducing an implicit regularization, (3) explain why the linear learning rate-batchsize scaling law fails at a large learning rate or at a small batchsize and (4) can provide an understanding of how discrete-time nature of SGD affects the recently discovered power-law phenomenon of SGD.

# 1 INTRODUCTION

Stochastic gradient descent (SGD) is the simple and efficient optimization algorithm behind the success of deep learning (Allen-Zhu et al., 2019; Xing et al., 2018; Zhang et al., 2018; Wang et al., 2020; He and Tao, 2020; Liu et al., 2021; Simsekli et al., 2019; Wu et al., 2020). minibatch noise, also known as the SGD noise, is the primary type of noise in the learning dynamics of neural networks. Practically, minibatch noise is unavoidable because a modern computer's memory is limited while the size of the datasets we use is large; this demands the dataset to be split into "minibatches" for training. At the same time, using minibatch is also a recommended practice because using a smaller batch size often leads to better generalization performance (Hoffer et al., 2017). Therefore, understanding minibatch noise in SGD has been one of the primary topics in deep learning theory. Dominantly many theoretical studies take two approximations: (1) the continuous-time approximation, which takes the infinitesimal step-size limit; (2) the Hessian approximation, which assumes that the covariance matrix of the SGD noise is equal to the Hessian  $H$ . While these approximations have been shown to provide some qualitative understanding, the limitation of these approximations is not well understood. For example, it is still unsure when such approximations are valid, which hinders our capability to assess the correctness of the results obtained by approximations.

In this work, we fill this gap by deriving analytical formulae for discrete-time SGD with arbitrary learning rates and exact minibatch noise covariance. In summary, the main contributions are: (1) we derive the strength and the shape of the minibatch SGD noise in the cases where the noise for discrete-time SGD is analytically solvable; (2) we show that the SGD noise takes a different form in different kinds of minima and propose general and more accurate approximations. This work is organized as follows: Sec. 2 introduces the background. Sec. 3 discusses the related works. Sec. 4 outlines our theoretical results. Sec. 5 derives new approximation formulae for SGD noises. In Sec. 6, we show how our results can provide practical and theoretical insights to problems relevant to contemporary machine learning research. For reference, the relationship of this work to the previous works is shown in Table 1.

# 2 BACKGROUND

In this section, we introduce the minibatch SGD algorithm. Let  $\{x_i,y_i\}_{i = 1}^N$  be a training set. We can define the gradient descent (GD) algorithm for a differentiable loss function  $L$  as  $\mathbf{w}_t = \mathbf{w}_{t - 1} - \lambda \nabla_{\mathbf{w}}L(\mathbf{w},\{\mathbf{x},\mathbf{y}\})$ , where  $\lambda$  is the learning rate and  $\mathbf{w}\in \mathbb{R}^{D}$  is the weights of the model. We consider an additive loss function for applying the minibatch SGD.

Definition 1. A loss function  $L(\{x_i, y_i\}_{i=1}^N, \mathbf{w})$  is additive if  $L(\{x_i, y_i\}_{i=1}^N, \mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \ell(x_i, y_i, \mathbf{w})$  for some differentiable, non-negative function  $\ell(\cdot)$ .

Table 1: Summary of related works on the noise and stationary distribution of SGD. This work fills the gap of the lack of theoretical results for the actual SGD dynamics, which is discrete-time and with minibatch noise.  

<table><tr><td>Setting</td><td>Artificial Noise</td><td>Hessian Approximation Noise</td><td>Minibatch Noise</td></tr><tr><td rowspan="2">Continuous-time</td><td>Sato and Nakagawa (2014); Welling and Teh (2011)</td><td>Jastrzebski et al. (2018); Zhu et al. (2019)</td><td rowspan="2">Blanc et al. (2020); Mori et al. (2021)</td></tr><tr><td>Mandt et al. (2017); Meng et al. (2020)</td><td>Wu et al. (2020); Xie et al. (2021)</td></tr><tr><td rowspan="2">Discrete-time</td><td>Yaida (2019); Gitman et al. (2019)</td><td rowspan="2">Liu et al. (2021)</td><td rowspan="2">This Work</td></tr><tr><td>Liu et al. (2021)</td></tr></table>

This definition is quite general. Most commonly studied and used loss functions are additive, e.g., the mean-square error (MSE) and cross-entropy loss. For an additive loss, the minibatch SGD with momentum algorithm can be defined.

Definition 2. The minibatch SGD with momentum algorithm by sampling with replacement computes the update to the parameter  $\mathbf{w}$  with the following set of equations:

$$
\left\{ \begin{array}{l} \hat {\mathbf {g}} _ {t} = \frac {1}{S} \sum_ {i \in B _ {t}} \nabla \ell \left(x _ {i}, y _ {i}, \mathbf {w} _ {t - 1}\right); \\ \mathbf {m} _ {t} = \mu \mathbf {m} _ {t - 1} + \hat {\mathbf {g}} _ {t}; \\ \mathbf {w} _ {t} = \mathbf {w} _ {t - 1} - \lambda \mathbf {m} _ {t}. \end{array} \right. \tag {1}
$$

where  $\mu \in [0,1)$  is the momentum hyperparameter,  $S \coloneqq |B_t|$  is the minibatch size, and the set  $B_{t} = \{i_{1},\dots i_{S}\}$  are  $S$  i.i.d. random integers sampled uniformly from  $[1,N]$ .

One can decompose the gradient into a deterministic plus a stochastic term. Note that  $\mathbb{E}_{\mathrm{B}}[\hat{\mathbf{g}}_t] = \nabla L$  is equal to the gradient for the GD algorithm. We use  $\mathbb{E}_{\mathrm{B}}(\cdot)$  to denote the expectation over batches, and use  $\mathbb{E}_{\mathbf{w}}(\cdot)$  to denote the expectation over the stationary distribution of the model parameters. Therefore, we can write  $\hat{\mathbf{g}}_t = \mathbb{E}_{\mathrm{B}}[\hat{\mathbf{g}}_t] + \eta_t$ , where  $\eta_t \coloneqq \frac{1}{S} \sum_{i \in B_t} \nabla \ell(x_i, y_i, \mathbf{w}_{t-1}) - \mathbb{E}_{\mathrm{B}}[\hat{\mathbf{g}}_t]$  is the noise term; the noise covariance is  $C(\mathbf{w}_t) \coloneqq \operatorname{cov}(\eta_t, \eta_t)$ . Of central importance to us is the averaged asymptotic noise covariance  $C \coloneqq \lim_{t \to \infty} \mathbb{E}_{\mathbf{w}_t}[C(\mathbf{w}_t)]$ . Also, we consider the asymptotic model fluctuation  $\Sigma \coloneqq \lim_{t \to \infty} \operatorname{cov}(\mathbf{w}_t, \mathbf{w}_t)$ .  $\Sigma$  gives the strength and shape of the fluctuation of  $\mathbf{w}$  around a local minimum and is another quantity of central importance to this work. Throughout this work,  $C$  is called the "noise" and  $\Sigma$  the "fluctuation".

# 3 RELATED WORKS

Noise and Fluctuation in SGD. Deep learning models are trained with SGD and its variants. To understand the parameter distribution in deep learning, one needs to understand the stationary distribution of SGD (Mamd et al., 2017). Sato and Nakagawa (2014) describes the stationary distribution of stochastic gradient Langevin dynamics using discrete-time Fokker-Planck equation. Yaida (2019) connects the covariance of parameter  $\Sigma$  to that of the noise  $C$  through the fluctuation-dissipation theorem. When  $\Sigma$  is known, one may obtain by Laplace approximation the stationary distribution of the model parameter around a local minimum  $w^{*}$  as  $\mathcal{N}(w^{*},\Sigma)$ . Therefore, knowing  $\Sigma$  can be of great practical use. For example, it has been used to estimate the local minimum escape efficiency (Zhu et al., 2019; Liu et al., 2021) and argue that SGD prefers a flatter minimum; it can also be used to assess parameter uncertainty and prediction uncertainty when a Bayesian prior is specified (Mamd et al., 2017; Gal and Ghahramani, 2016; Pearce et al., 2020). Empirically, both the fluctuation and the noise are known to crucially affect the generalization of a deep neural network. Wu et al. (2020) shows that the strength and shape of the  $\Sigma$  due to the minibatch noise lead to better generalization of neural networks in comparison to an artificially constructed noise.

Hessian Approximation of the Minibatch Noise. However, it is not yet known what form  $C$  and  $\Sigma$  actually take for SGD in a realistic learning setting. Early attempts assume isotropic noise in the continuous-time limit (Sato and Nakagawa, 2014; Mandt et al., 2017). In this setting, the noise is an isotropic Gaussian with  $C \sim I_D$ , and  $\Sigma$  is known to be proportional to the inverse Hessian  $H^{-1}$ . More recently, the importance of noise structure was realized (Hoffer et al., 2017; Jastrzebski et al., 2018; Zhu et al., 2019; HaoChen et al., 2020). "Hessian approximation", which assumes  $C \approx c_0H$  for some constant  $c_0$ , has often been adopted for understanding SGD (see Table 1); this assumption is often motivated by the fact that  $C = J_w \approx H$ , where  $J_w$  is the Fisher information matrix (FIM) (Zhu et al., 2019); the fluctuation can be solved to be isotropic:  $\Sigma \sim I_D$ . However, it is not known under what conditions the Hessian approximation is valid, while previous works have argued that it can be very inaccurate (Martens, 2014; Liu et al., 2021; Thomas et al., 2020; Kunstner et al., 2019). However, Martens (2014) and Kunstner et al. (2019) only focuses on the natural gradient descent (NGD) setting; Thomas et al. (2020) is closest to ours, but it does not apply to the case with momentum, a matrix learning rate, or regularization.

Discrete-time SGD with a Large Learning Rate. Recently, it has been realized that networks trained at a large learning rate have a dramatically better performance than networks trained with a vanishing learning rate (lazy training) (Chizat and Bach, 2018). Lewkowycz et al. (2020) shows that there is a qualitative difference between the lazy training regime and the large learning rate regime; the performance features two plateaus in testing accuracy in the two regimes, with the large learning rate regime performing much better. However, the theory regarding discrete-time SGD at a large learning rate is almost non-existent, and it is also not known what  $\Sigma$  may be when the learning rate is non-vanishing. Our work also sheds light on the behavior of SGD at a large learning rate. Some other works also consider discrete-time SGD in a similar setting (Fontaine et al., 2021; Dieuleveut et al., 2020; Toulis et al., 2017), but the focus is not on deriving analytical formulae or does not deal with the stationary distribution.

# 4 SGD NOISE AND FLUCTUATION IN LINEAR REGRESSION

This section derives the shape and strength of SGD noise and fluctuation for linear regression; concurrent to our work, Kunin et al. (2021) also studies the same problem but with continuous-time approximation; our result is thus more general. To emphasize the message, we discuss the label noise case in more detail. The other situations also deserve detailed analysis; we delay such discussion to the appendix due to space constraints. Notation:  $S$  denotes the minibatch size.  $\mathbf{w} \in \mathbb{R}^D$  is the model parameter viewed in a vectorized form;  $\lambda \in \mathbb{R}_+$  denotes a scalar learning rate; when the learning rate takes the form of a preconditioning matrix, we use  $\Lambda \in \mathbb{R}^{D \times D}$ .  $A \in \mathbb{R}^{D \times D}$  denotes the covariance matrix of the input data. When a matrix  $X$  is positive semi-definite, we write  $X \geq 0$ ; throughout, we require  $\Lambda \geq 0$ .  $\gamma \in \mathbb{R}$  denotes the weight decay hyperparameter; when the weight decay hyperparameter is a matrix, we write  $\Gamma \in \mathbb{R}^{D \times D}$ .  $\mu$  is the momentum hyperparameter in SGD. For two matrices  $X, Y$ , the commutator is defined as  $[X, Y] := XY - YX$ . Other notations are introduced in the context.

# 4.1 KEY PREVIOUS RESULTS

When  $N \gg S$ , the following proposition is well-known and gives the exact noise due to minibatch sampling. See Appendix E.1 for derivation.

Proposition 1. The noise covariance of SGD as defined in Definition 2 is

$$
C (\mathbf {w}) = \frac {1}{S N} \sum_ {i = 1} ^ {N} \nabla \ell_ {i} (\mathbf {w}) \nabla \ell_ {i} (\mathbf {w}) ^ {\mathrm {T}} - \frac {1}{S} \nabla L (\mathbf {w}) \nabla L (\mathbf {w}) ^ {\mathrm {T}}, \tag {2}
$$

where the notations  $\ell_i(\mathbf{w})\coloneqq l(x_i,y_i,\mathbf{w})$  and  $L(\mathbf{w})\coloneqq L(\{x_i,y_i\}_{i = 1}^N,\mathbf{w})$  are used.

This gradient covariance matrix  $C$  is crucial to understand the minibatch noise. The standard literature often assumes  $C(\mathbf{w}) \approx H(\mathbf{w})$ ; however, the following well-known proposition shows that this approximation can easily break down.

Proposition 2. Let  $\mathbf{w}_{*}$  be the solution such that  $L(\mathbf{w}_{*}) = 0$ , then  $C(\mathbf{w}_{*}) = 0$ .

Proof. Because  $\ell_{i}$  is non-negative for all  $i$ ,  $L(\mathbf{w}_{*}) = 0$  implies that  $\ell_{i}(\mathbf{w}_{*}) = 0$ . The differentiability in turn implies that each  $\nabla \ell_{i}(\mathbf{w}_{*}) = 0$ ; therefore,  $C = 0$ .  $\square$

This proposition implies that there is no noise if our model can achieve zero training loss (which is achievable for an overparametrized model). This already suggests that the Hessian approximation  $C \sim H$  is wrong since the Hessian is unlikely to vanish in any minimum. The fact that the noise strength vanishes at  $L = 0$  suggests that the SGD noise might at least be proportional to  $L(\mathbf{w})$ , which we will show to be true for many cases. The following theorem relates  $C$  and  $\Sigma$  of the discrete-time SGD algorithm with momentum for a matrix learning rate.

Theorem 1. (Liu et al., 2021) Consider running SGD on a quadratic loss function with Hessian  $H$ , learning rate matrix  $\Lambda$ , momentum  $\mu$ . Assuming ergodicity, then

$$
(1 - \mu) (\Lambda H \Sigma + \Sigma H \Lambda) - \frac {1 + \mu^ {2}}{1 - \mu^ {2}} \Lambda H \Sigma H \Lambda + \frac {\mu}{1 - \mu^ {2}} (\Lambda H \Lambda H \Sigma + \Sigma H \Lambda H \Lambda) = \Lambda C \Lambda . \tag {3}
$$

Proposition 1 and Theorem 1 allow one to solve  $C$  and  $\Sigma$ . Equation (3) can be seen as a general form of the Lyapunov equation (Lyapunov, 1992) and is hard to solve in general (Hammarling, 1982; Ye et al., 1998; Simoncini, 2016). Solving this analytical equation in settings of machine learning relevance is one of the main technical contributions of this work.

<sup>1</sup>We use the word global minimum to refer to the global minimum of the loss function, i.e., where  $L = 0$  and a local minimum refers to a minimum that has a non-negative loss, i.e.,  $L \geq 0$ .

# 4.2 RANDOM NOISE IN THE LABEL

We first consider the case when the labels contain noise. The loss function takes the form

$$
L (\mathbf {w}) = \frac {1}{2 N} \sum_ {i = 1} ^ {N} \left(\mathbf {w} ^ {\mathrm {T}} x _ {i} - y _ {i}\right) ^ {2}, \tag {4}
$$

where  $x_{i} \in \mathbb{R}^{D}$  are drawn from a zero-mean Gaussian distribution with feature covariance  $A \coloneqq \mathbb{E}_{\mathrm{B}}[xx^{\mathrm{T}}]$ , and  $y_{i} = \mathbf{u}^{\mathrm{T}}x_{i} + \epsilon_{i}$ , for some fixed  $\mathbf{u}$  and  $\epsilon_{i} \in \mathbb{R}$  is drawn from a distribution with zero mean and finite second momentum  $\sigma^2$ . We redefine  $\mathbf{w} - \mathbf{u} \rightarrow \mathbf{w}$  and let  $N \to \infty$  with  $D$  held fixed. The following lemma finds  $C$  as a function of  $\Sigma$ .

Lemma 1. (Covariance matrix for SGD noise in the label) Let the model be updated according to Eq. (1) with random noise in the label while  $N \to \infty$ . Then,

$$
C = \frac {1}{S} (A \Sigma A + \operatorname {T r} [ A \Sigma ] A + \sigma^ {2} A). \tag {5}
$$

The model fluctuation can be obtained using this lemma.

Theorem 2. (Fluctuation of model parameters with random noise in the label) Let the assumptions be the same as in Proposition 1 and  $[\Lambda, A] = 0$ . Then,

$$
\Sigma = \frac {\sigma^ {2}}{S} \left(1 + \frac {\kappa_ {\mu}}{S}\right) \Lambda G _ {\mu} ^ {- 1}, \tag {6}
$$

where  $\kappa_{\mu} := \frac{\mathrm{Tr}[\Lambda AG_{\mu}^{-1}]}{1 - \frac{1}{S}\mathrm{Tr}[\Lambda AG_{\mu}^{-1}]}$  with  $G_{\mu} := 2(1 - \mu)I_{D} - \left(\frac{1 - \mu}{1 + \mu} + \frac{1}{S}\right)\Lambda A.$

Remark. This result is numerically validated in Appendix A. The subscript  $\mu$  refers to momentum. To obtain results for vanilla SGD, one can set  $\mu = 0$ , which has the effect of reducing  $G_{\mu} \to G = 2I_D - \left(1 + \frac{1}{S}\right)\Lambda A$ . From now on, we focus on the case when  $\mu = 0$  for notational simplicity, but we note that the results for momentum can be likewise studied. The condition  $[\Lambda, A] = 0$  is not peculiar because this condition holds for a scalar learning rate and common second-order methods such as Newton's method.

If  $\sigma^2 = 0$ , then  $\Sigma = 0$ . This means that when there is no label noise, the model parameter has a vanishing stationary fluctuation, which corroborates Proposition 2. When a scalar learning rate  $\lambda \ll 1$  and  $1 \ll S$ , we have

$$
\Sigma \approx \frac {\lambda \sigma^ {2}}{2 S} I _ {D}, \tag {7}
$$

which is the result one would expect from the continuous-time theory with the Hessian approximation (Liu et al., 2021; Xie et al., 2021; Zhu et al., 2019), except for a correction factor of  $\sigma^2$ . Therefore, a Hessian approximation fails to account for the randomness in the data of strength  $\sigma^2$ . We provide a systematic and detailed comparison with the Hessian approximation in Table 2 of Sec. B.

Moreover, it is worth comparing the exact result in Theorem 2 with Eq. (7) in the regime of nonvanishing learning rate and small batch size. One notices two differences: (1) an anisotropic enhancement, appearing in the matrix  $G_{\mu}$  and taking the form  $-\lambda (1 + 1 / S)A$ ; compared with the result in Liu et al. (2021), this term is due to the compound effect of using a large learning rate and a small batchsize; (2) an isotropic enhancement term  $\kappa$ , which causes the overall magnitude of fluctuations to increase; this term does not appear in the previous works that are based on the Hessian approximation and is due to the minibatch sampling process alone. As the numerical example in Sec. A shows, at large batch size, the discrete-time nature of SGD is the leading source of fluctuation; at small batch size, the isotropic enhancement becomes the dominant source of fluctuation. Therefore, the minibatch sampling process causes two different kinds of enhancement to the fluctuation, potentially increasing the exploration power of SGD at initialization but reducing the convergence speed.

Now, combining Theorem 2 and Lemma 1, one can obtain an explicit form of the noise covariance.

Theorem 3. The noise covariance matrix of minibatch SGD with random noise in the label is

$$
C = \frac {\sigma^ {2}}{S} A + \frac {\sigma^ {2}}{S ^ {2}} \left(1 + \frac {\kappa_ {\mu}}{S}\right) \left(\Lambda A G _ {\mu} ^ {- 1} + \operatorname {T r} \left[ \Lambda A G _ {\mu} ^ {- 1} \right] I _ {D}\right) A. \tag {8}
$$

By definition,  $C = J$  is the FIM. The Hessian approximation, in sharp contrast, can only account for the term in orange. A significant modification containing both anisotropic and isotropic (up to Hessian) is required to fully understand SGD noise, even in this simple example. Additionally, comparing this result with the training loss (127), one can find that the noise covariance contains one term that is proportional to the training loss. In fact, we will derive in Sec. 5 that containing a term proportional to training loss is a general feature of the SGD noise. We also study the case when the input is contaminated with noise. Interestingly, the result is the same with the label noise case with  $\sigma^2$  replaced by a more complicated term of the form  $\mathrm{Tr}[AK^{-1}BU]$ . We thus omit this part from the main text. A detailed discussion can be found in Appendix E.3.1. In the next section, we study the effect of regularization on SGD noise and fluctuation.

# 4.3 LEARNING WITH REGULARIZATION

Now, we show that regularization also causes a unique SGD noise. The loss function for  $\Gamma - L_2$  regularized linear regression is

$$
L _ {\Gamma} (\mathbf {w}) = \frac {1}{2 N} \sum_ {i = 1} ^ {N} \left[ (\mathbf {w} - \mathbf {u}) ^ {\mathrm {T}} x _ {i} \right] ^ {2} + \frac {1}{2} \mathbf {w} ^ {\mathrm {T}} \Gamma \mathbf {w} = \frac {1}{2} (\mathbf {w} - \mathbf {u}) ^ {\mathrm {T}} A (\mathbf {w} - \mathbf {u}) + \frac {1}{2} \mathbf {w} ^ {\mathrm {T}} \Gamma \mathbf {w}, \tag {9}
$$

where  $\Gamma$  is a symmetric matrix; conventionally, one set  $\Gamma = \gamma I_D$  with a scalar  $\gamma > 0$ . For conciseness, we assume that there is no noise in the label, namely  $y_i = \mathbf{u}^{\mathrm{T}}x_i$  with a constant vector  $\mathbf{u}$ . One important quantity in this case will be  $\mathbf{uu}^{\mathrm{T}} := U$ . The noise for this form of regularization can be calculated but takes a complicated form.

Proposition 3. (Noise covariance matrix for learning with  $L_{2}$  regularization) Let the algorithm be updated according to Eq. (1) with  $L_{2}$  regularization with  $N \to \infty$  and  $[A, \Gamma] = 0$ . Then,

$$
C = \frac {1}{S} \left(A \Sigma A + \operatorname {T r} [ A \Sigma ] A + \operatorname {T r} \left[ \Gamma^ {\prime \mathrm {T}} A \Gamma^ {\prime} U \right] A + \Gamma A ^ {\prime} U A ^ {\prime} \Gamma\right), \tag {10}
$$

where  $A^{\prime}\coloneqq K^{-1}A$ $\Gamma^{\prime}\coloneqq K^{-1}\Gamma$  with  $K\coloneqq A + \Gamma$

Notice that the last term  $\Gamma A'UA'\Gamma$  in  $C$  is unique to the regularization-based noise: it is rank-1 because  $U$  is rank-1. This term is due to the mismatch between the regularization and the minimum of the original loss. Also, note that the term  $\mathrm{Tr}[A\Sigma]$  is proportional to the training loss. Define the test loss to be  $L_{\mathrm{test}} \coloneqq \lim_{t \to \infty} \mathbb{E}_{\mathbf{w}_t}\left[\frac{1}{2} (\mathbf{w}_t - \mathbf{u})^{\mathrm{T}} A (\mathbf{w}_t - \mathbf{u})\right]$ , we can prove the following theorem. We will show that one intriguing feature of discrete-time SGD is that the weight decay can be negative.

Theorem 4. (Test loss and model fluctuation for  $L_{2}$  regularization) Let the assumptions be the same as in Proposition 3. Then

$$
L _ {\text {t e s t}} = \frac {\lambda}{2 S} \left(\operatorname {T r} \left[ A K ^ {- 2} \Gamma^ {2} U \right] \kappa + r\right) + \frac {1}{2} \operatorname {T r} \left[ A K ^ {- 2} \Gamma^ {2} U \right], \tag {11}
$$

where  $\kappa := \frac{\operatorname{Tr}[A^2K^{-1}G^{-1}]}{1 - \frac{\lambda}{S}\operatorname{Tr}[A^2K^{-1}G^{-1}]}$ ,  $r := \frac{\operatorname{Tr}[A^3K^{-3}\Gamma^2G^{-1}U]}{1 - \frac{\lambda}{S}\operatorname{Tr}[A^2K^{-1}G^{-1}]}$ , with  $G := 2I_D - \lambda\left(K + \frac{1}{S}K^{-1}A^2\right)$ . Moreover, let  $[\Gamma, U] = 0$ , then

$$
\Sigma = \frac {\lambda}{S} \operatorname {T r} \left[ A K ^ {- 2} \Gamma^ {2} U \right] \left(1 + \frac {\lambda \kappa}{S}\right) A K ^ {- 1} G ^ {- 1} + \frac {\lambda}{S} \left(A ^ {2} K ^ {- 2} \Gamma^ {2} U + \frac {\lambda r}{S} A\right) K ^ {- 1} G ^ {- 1}. \tag {12}
$$

This result is numerically validated in Appendix A. The test loss (11) has an interesting consequence. One can show that there exist situations where the optimal  $\Gamma$  is negative. When discussing the test loss, we make the convention that if  $\mathbf{w}_t$  diverges, then  $L_{\mathrm{test}} = \infty$ .

Corollary 1. Let  $\gamma^{*} = \arg \min_{\gamma}L_{\mathrm{test}}$ . There exist  $a, \lambda$  and  $S$  such that  $\gamma^{*} < 0$ .

The proof shows that when the learning rate is sufficiently large, only negative weight decay is allowed. This agrees with the argument in Liu et al. (2021) that discrete-time SGD introduces an implicit  $L_{2}$  regularization that favors small norm solutions. A too-large learning rate requires a negative weight decay because a large learning rate already over-regularizes the model and one needs

to introduce an explicit negative weight decay to offset this over-regularization effect of SGD. This is a piece of direct evidence that using a large learning rate can help regularize the models. This result relates to the implicit regularization effect of SGD. It has been hypothesized that the dynamics of SGD implicitly regularizes neural networks such that the training favors simpler solutions (Kalimeris et al., 2019). Our result suggests one new mechanism for such a regularization.

# 5 NOISE STRUCTURE FOR GENERIC SETTINGS

The results in the previous sections suggest that (1) the SGD noises differ for different kinds of situations, and (2) SGD noise contains a term proportional to the training loss in general. These two facts motivate us to derive the noise covariance differently for different kinds of minima. Let  $f(\mathbf{w}, x)$  denote the output of the model for a given input  $x \in \mathbb{R}^D$ . Here, we consider a more general case;  $f(\mathbf{w}, x)$  may be any differentiable function, e.g., a non-linear deep neural network. The number of parameters in the model is denoted by  $P$ , and hence  $\mathbf{w} \in \mathbb{R}^P$ . For a training dataset  $\{x_i, y_i\}_{i=1,2,\dots,N}$ , the loss function with a  $L_2$  regularization is given by

$$
L _ {\Gamma} (\mathbf {w}) = L _ {0} (\mathbf {w}) + \frac {1}{2} \mathbf {w} ^ {\mathrm {T}} \Gamma \mathbf {w}, \tag {13}
$$

where  $L_0(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^{N}\ell(f(\mathbf{w},x_i),y_i)$  is the loss function without regularization, and  $H_0$  is the Hessian of  $L_0$ . We focus on the MSE loss  $\ell(f(\mathbf{w},x_i),y_i) = [f(\mathbf{w},x_i) - y_i]^2/2$ . Our result crucially relies on the following two assumptions, which relate to the conditions of different kinds of local minima.

Assumption 1. (Fluctuation decays with batch size)  $\Sigma$  is proportional to  $S^{-1}$ , i.e.  $\Sigma = O(S^{-1})$ .

This is justified by the results in all the related works (Liu et al., 2021; Xie et al., 2021; Meng et al., 2020; Mori et al., 2021), where  $\Sigma$  is found to be  $O(S^{-1})$ .

Assumption 2. (Weak homogeneity)  $|L - \ell_i|$  is small; in particular, it is of order  $o(L)$ .

This assumption amounts to assuming that the current training loss  $L$  reflects the actual level of approximation for each data point well. In fact, since  $L \geq 0$ , one can easily show that  $|L - \ell_i| = O(L)$ . Here, we require a slightly stronger condition for a more clean expression, when  $|L - \ell_i| = O(L)$  we can still get a similar expression but with some constant that hinders the clarity. Relaxing this condition can be an important and interesting future work. The above two conditions allow us to state our general theorem formally.

Theorem 5. Let the training loss be  $L_{\Gamma} = L_0 + \frac{1}{2}\mathbf{w}^{\mathrm{T}}\boldsymbol{\Gamma}\mathbf{w}$  and the models be optimized with SGD in the neighborhood of a local minimum  $\mathbf{w}^*$ . Then,

$$
C (\mathbf {w}) = \frac {2 L _ {0} (\mathbf {w})}{S} H _ {0} (\mathbf {w}) - \frac {1}{S} \nabla L _ {\Gamma} (\mathbf {w}) \nabla L _ {\Gamma} (\mathbf {w}) ^ {\mathrm {T}} + o (L _ {0}). \tag {14}
$$

The noise takes different forms for different kinds of local minima.

Corollary 2. Omitting the terms of order  $o(L_0)$ , when  $\Gamma \neq 0$ ,

$$
C = \frac {2 L _ {0} \left(\mathbf {w} ^ {*}\right)}{S} H _ {0} \left(\mathbf {w} ^ {*}\right) - \frac {1}{S} \Gamma \mathbf {w} ^ {*} \mathbf {w} ^ {* \mathrm {T}} \Gamma + O \left(S ^ {- 2}\right) + O \left(\left| \mathbf {w} - \mathbf {w} ^ {*} \right| ^ {2}\right). \tag {15}
$$

When  $\Gamma = 0$  and  $L_0(\mathbf{w}^*)\neq 0$

$$
C = \frac {2 L _ {0} \left(\mathbf {w} ^ {*}\right)}{S} H _ {0} \left(\mathbf {w} ^ {*}\right) + O \left(S ^ {- 2}\right) + O \left(\left| \mathbf {w} - \mathbf {w} ^ {*} \right| ^ {2}\right). \tag {16}
$$

When  $\Gamma = 0$  and  $L_0(\mathbf{w}^*) = 0$ ,

$$
C = \frac {1}{S} \left(\operatorname {T r} \left[ H _ {0} \left(\mathbf {w} ^ {*}\right) \boldsymbol {\Sigma} \right] I _ {D} - H _ {0} \left(\mathbf {w} ^ {*}\right) \boldsymbol {\Sigma}\right) H _ {0} \left(\mathbf {w} ^ {*}\right) + O \left(S ^ {- 2}\right) + O \left(\left| \mathbf {w} - \mathbf {w} ^ {*} \right| ^ {2}\right). \tag {17}
$$

Remark. Assumption 2 can be replaced by a weaker but more technical assumption called the "decoupling assumption", which has been used in recent works to derive the continuous-time distribution of SGD (Mori et al., 2021; Wojtowytsch, 2021). The Hessian approximation was invoked in most of the literature without considering the conditions of its applicability (Jastrzebski et al., 2018; Zhu et al., 2019; Liu et al., 2021; Wu et al., 2020; Xie et al., 2021). Our result does provide such

conditions for applicability. As indicated by the two assumptions, this theorem is applicable when the batch size is not too small and when the local minimum has a loss close to 0. The reason for the failure of the Hessian approximation is that, while the FIM is equal to the expected Hessian  $J = \mathbb{E}[H]$ , there is no reason to expect the expected Hessian to be close to the actual Hessian of the minimum.

The proof is given in Appendix C. Two crucial messages this corollary delivers are (1) the SGD noise is different in strength and shape in different kinds of local minima and that they need to be analyzed differently; (2) the SGD noise contains a term that is proportional to the training loss  $L_{0}$  in general. Recently, it has been experimentally demonstrated that the SGD noise is indeed proportional to the training loss in realistic deep neural network settings, both when the loss function is MSE and cross-entropy (Mori et al., 2021); our result offers a theoretical justification. The previous works all treat all the minima as if the noise is similar (Jastrzebski et al., 2018; Zhu et al., 2019; Liu et al., 2021; Wu et al., 2020; Xie et al., 2021), which can lead to inaccurate or even incorrect understanding. For example, Theorem 3.2 in Xie et al. (2021) predicts a high escape probability from a sharp local or global minimum. However, this is incorrect because a model at a global minimum has zero probability of escaping due to a vanishing gradient. In contrast, the escape rate results derived in Mori et al. (2021) correctly differentiate the local and global minima. We also note that these general formulae are consistent with the exact solutions we obtained in the previous section than the Hessian approximation. For example, the dependence of the noise strength on the training loss in Theorem 2, and the rank-1 noise of regularization are all well-reflected in these formulae. In contrast, the simple Hessian approximation misses these crucial distinctions. Lastly, combining Theorem 5 with Theorem 1, one can also find the fluctuation.

Corollary 3. Let the noise be as in Theorem 5, and omit the terms of order  $O(S^{-2})$  and  $O(|\mathbf{w} - \mathbf{w}^{*}|^{2})$ . Then, when  $\Gamma \neq 0$  and when  $\Lambda$ ,  $H_{0}(\mathbf{w}^{*})$  and  $\Gamma$  commute with each other,  $P_{r^{\prime}}\Sigma = \frac{1}{S}\frac{\Lambda}{1 - \mu}(2L_{0}H_{0} - \Gamma\mathbf{w}^{*}\mathbf{w}^{*\mathrm{T}}\Gamma)(H_{0} + \Gamma)^{+}\left[2I_{D} - \frac{\Lambda}{1 + \mu}(H_{0} + \Gamma)\right]^{-1}$ . When  $\Gamma = 0$  and  $L_{0}(\mathbf{w}^{*}) \neq 0$ ,  $P_{r}\Sigma = \frac{2L_{0}}{S(1 - \mu)}P_{r}\Lambda\left(2I_{D} - \frac{\Lambda}{1 + \mu}H_{0}\right)^{-1}$ . When  $\Gamma = 0$  and  $L_{0}(\mathbf{w}^{*}) = 0$ ,  $P_{r}\Sigma = 0$ . Here the superscript  $+$  is the Moore-Penrose pseudo inverse,  $P_{r} := \operatorname{diag}(1,\dots,1,0,\dots,0)$  is the projection operator with  $r$  non-zero entries,  $r \leq D$  is the rank of the Hessian  $H_{0}$ , and  $r' \leq D$  is the rank of  $H_{0} + \Gamma$ . For the null space  $H_{0}$ ,  $\Sigma$  can be arbitrary.

# 6 APPLICATIONS

One major advantage of analytical solutions is that they can be applied in a simple "plug-in" manner by the practitioners or theorists to analyze new problems they encounter. In this section, we briefly outline a few examples where the proposed theories can be relevant.

# 6.1 HIGH-DIMENSIONAL REGRESSION

We first apply our result to the high-dimensional regression problem and show how over-and-underparametrization might play a role in determining the minibatch noise. Here, we take  $N, D \to \infty$  with the ratio  $\alpha \coloneqq N / D$  held fixed. The loss function is  $L(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N} \left( \mathbf{w}^{\mathrm{T}} x_i - y_i \right)^2$ . As in the standard literature (Hastie et al., 2019), we assume the existence of label noise:  $y_i = \mathbf{u}^{\mathrm{T}} x_i + \epsilon_i$ , with  $\operatorname{Var}[\epsilon_i] = \sigma^2$ . A key difference between our setting and the standard high-dimensional setting is that, in the standard setting (Hastie et al., 2019), one uses the GD algorithm with vanishing learning rate  $\lambda$  instead of the minibatch SGD algorithm with a non-vanishing learning rate. Tackling the high-dimensional regression problem with non-vanishing  $\lambda$  and a minibatch noise is another main technical contribution of this work. In this setting, we can obtain the following result on the noise covariance matrix.

Proposition 4. Let  $\hat{A} = \frac{1}{N}\sum_{i}^{N}x_{i}x_{i}^{\mathrm{T}}$  and suppose assumptions 1 and 2 hold. With fixed  $S$ ,  $\lambda$ , then  $C = \frac{1}{S}\left(\operatorname{Tr}[\hat{A}\Sigma]I_D - \hat{A}\Sigma\right)\hat{A} + \max \left\{0, \frac{\sigma^2}{S}\left(1 - \frac{1}{\alpha}\right)\right\} \hat{A}$ .

We note that this proposition follows from an application of Theorem 5; this shows an important theoretical application of our general theory. An interesting observation is that one  $\Sigma$ -independent term proportional to  $\sigma^2$  emerges in the underparametrized regime ( $\alpha > 1$ ). However, for the overparametrized regime, the noise is completely dependent on  $\Sigma$ , which is a sign that the stationary solution has no fluctuation. This shows that the degree of underparametrization also plays a distinctive role in the fluctuation. In fact, one can prove the following theorem.

![](images/cd574577dcb4c0ff3fc978e677128d833d84ae57fa28f142fcda848594c1274e.jpg)  
Figure 1: Realistic learning settings with neural networks and logistic regression. Left: Variance of training loss of a neural network with width  $d$  and tanh activation on the MNIST dataset. We see that the variance explodes after  $d \geq 200$ . In contrast, rescaling the learning rate by  $1 / d$  results in a constant noise level in training. This suggests that the stability condition we derived for high-dimension regression is also useful for understanding deep learning. Middle: Stability of Adam with the same setting. Adam also experiences a similar stability problem when the model width increases. Right: Logistic regression on MNIST trained with SGD; with  $\lambda = 1.5$ ,  $S = 32$ . We see that the optimal performance is also achieved at negative weight decay strength  $\gamma$ , suggesting that a large learning rate can indeed introduce effective regularization.

![](images/01f285877699a5855a159ef1e1b6475074ebc490df90f3a9606a5019225387a4.jpg)

![](images/fa24fd466e51b91707229f7ff54230ea95887d9b1db72730c3377558678606b6.jpg)

Theorem 6. When a stationary solution exists for  $\mathbf{w}$ , we have  $\mathrm{Tr}[\hat{A}\Sigma] = \max \left\{0, \frac{\lambda \sigma^2}{S} \left(1 - \frac{1}{\alpha}\right) \hat{\kappa}\right\}$ , where  $\hat{\kappa} := \frac{\mathrm{Tr}[\hat{G}^{-1} \hat{A}]}{1 - \frac{\lambda}{S} \mathrm{Tr}[\hat{G}^{-1} \hat{A}]}$  with  $\hat{G} := 2I_D - \lambda \left(1 - \frac{1}{S}\right) \hat{A}$ .

# 6.2 IMPLICATION FOR NEURAL NETWORK TRAINING

It is commonly believed that the high-dimensional linear regression problem can be a minimal model for deep learning. Taking this stance, Theorem 6 suggests a technique for training neural networks. For SGD to converge, a positive semi-definite  $\Sigma$  must exist; however,  $\Sigma \geq 0$  if and only if  $\hat{\kappa} \geq 0$ . From  $\hat{\kappa} > 0$ , we have  $\sum_{i=1}^{D} \frac{1}{2 / \lambda a_i - 1 + 1 / S} < S$ , where  $a_i$  are the eigenvalues of  $\hat{A}$ . This means that each summand should have the order of  $D / S$ . Thus the upper bound of  $\lambda$  should have the order of  $2S / aD$ , where  $a$  is the typical value of  $a_i$ 's. One implication of the dependence on the dimension is that the stability of a neural network trained with SGD may strongly depend on its width  $d$ , and one may rescale the learning rate according to the width to stabilize neural network training. See Figure 1-Left and Middle. We train a two-layer tanh neural network on MNIST and plot the variance of its training loss in the first epoch with fixed  $\lambda = 0.5$ . We see that, when  $d \geq 200$ , the training starts to destabilize, and the training loss begins to fluctuate dramatically. When rescaling the learning rate by  $1 / d$ , we see that the variance of the training loss is successfully kept roughly constant across all  $d$ . This suggests a training technique worth being explored by practitioners in the field. In Figure 1-Middle, we also use Adam for training the same network and find a similar stabilizing trick to work for Adam.

# 6.3 A NATURAL LEARNING EXAMPLE WITH NEGATIVE WEIGHT DECAY

Sec. 4.3 shows that a too-large learning rate introduces an effective  $L_{2}$  regularization that can only be corrected by setting the weight decay to be negative. This effect can be observed in more realistic learning settings. We train a logistic regressor on the MNIST dataset with a large learning rate (of order  $O(1)$ ). Figure 1-Right confirms that, at a large learning rate, the optimal weight decay can indeed be optimal. This agrees with our argument that using a large learning rate can effectively regularize the training.

# 6.4 SECOND-ORDER METHODS

Understanding stochastic second-order methods (including the adaptive gradient methods) is also important for deep learning (Agarwal et al., 2017; Zhiyi and Ziyin, 2021; Martens, 2014; Kunstner et al., 2019). In this section, we apply our theory to two standard second-order methods: damped Newton's method (DNM) and natural gradient descent (NGD). We provide more accurate results than those derived in Liu et al. (2021). The derivations are given in Appendix D.2. For DNM, the preconditioning learning rate matrix is defined as  $\Lambda := \lambda A^{-1}$ . The model fluctuation is shown to be proportional to the inverse of the Hessian:  $\Sigma = \frac{\lambda\sigma^2}{gS - \lambda D} A^{-1}$ , where  $g := 2(1 - \mu) - \left(\frac{1 - \mu}{1 + \mu} + \frac{1}{S}\right)\lambda$ . The main difference with the previous results is that the fluctuation now depends explicitly on the dimension  $D$ , and implies a stability condition:  $S \geq \lambda D / g$ , corroborating the stability condition we derived above. For NGD, the preconditioning matrix is defined by the inverse of the Fisher information that  $\Lambda := \frac{\lambda}{S} J(\mathbf{w})^{-1} = \frac{\lambda}{S} C^{-1}$ . We show that  $\Sigma = \frac{\lambda}{2}\left(\frac{1}{1 + D}\frac{1}{1 + \mu} + \frac{1}{1 - \mu}\frac{1}{S}\right)A^{-1}$  is one solution when  $\sigma = 0$ , which also contains a correction related to  $D$  compared to the result in Liu et al. (2021) which is  $\Sigma = \frac{\lambda}{2}\left(\frac{1}{1 + \mu} + \frac{1}{1 - \mu}\frac{1}{S}\right)A^{-1}$ . A consequence is that  $J \sim \Sigma^{-1}$ . The surprising

fact is that the stability of both NGD and DNM now crucially depends on  $D$ ; combining with the results in Sec. 6.1, this suggests that the dimension of the problem may crucially affect the stability and performance of the minibatch-based algorithms. This result also implies that some features we derived are shared across many algorithms that depend on minibatch noise and that our results may be relevant to a broad class of optimization algorithms other than SGD.

# 6.5 FAILURE OF THE  $\lambda - S$  SCALING LAW

One well-known technique in deep learning training is that one can scale  $\lambda$  linearly as one increases the batch size  $S$  to achieve high-efficiency training without hindering the generalization performance; however, it is known that this scaling law fails when the learning rate is too large, or the batch size is too small. In Hoffer et al. (2017), this scaling law is established on the ground that  $\Sigma \sim \lambda / S$ . However, our result in Theorem 2 suggests the reason for the failure even for the simple setting of linear regression. Recall that the exact  $\Sigma$  takes the form:

$$
\Sigma = \frac {\lambda \sigma^ {2}}{S} \left(1 + \frac {\kappa_ {\mu}}{S}\right) G _ {\mu} ^ {- 1}
$$

for a scalar  $\lambda$ . One notices that the leading term is indeed proportional to  $\lambda / S$ . However, the discrete-time SGD results in a second-order correction in  $S$ , and the term proportional to  $1 / S^2$  does not contain a corresponding  $\lambda$ ; this explains the failure of the scaling law in small  $S$ , where the second-order contribution of  $S$  becomes significant. To understand the failure at large  $\lambda$ , we need to look at the term  $G_{\mu}$ :

$$
G _ {\mu} = 2 (1 - \mu) I _ {D} - \left(\lambda \frac {1 - \mu}{1 + \mu} + \frac {\lambda}{S}\right) A.
$$

One notices that the second term contains a part that only depends on  $\lambda$  but not on  $S$ . This part is negligible compared to the first term when  $\lambda$  is small; however, it becomes significant as the second term approaches the first term. Therefore, increasing  $\lambda$  changes this part of the fluctuation, and thus the scaling law no more holds if  $\lambda$  is large.

# 6.6 POWER LAW TAIL IN DISCRETE-TIME SGD

It has recently been discovered that the SGD noise causes a heavy-tail distribution (Simsekli et al., 2019; 2020), with a tail decaying like a power law with tail index  $\beta$  (Hodgkinson and Mahoney, 2020). In continuous-time, the stationary distribution has been found to obey a Student's t-like distribution,  $p(w) \sim L^{-(1 + \beta)/2} \sim \left(\sigma^2 + aw^2\right)^{-(1 + \beta)/2}$  (Meng et al., 2020; Mori et al., 2021; Wojtowitsch, 2021). However, this result is only established for continuous-time approximations to SGD and one does not know what affects the exponent  $\beta$  for discrete-time SGD. Our result in Theorem 2 can serve as a tool to find the discrete-time correction to the tail index of the stationary distribution. In Appendix D.3, we show that the tail index of discrete-time SGD in 1d can be estimated as  $\beta(\lambda, S) = \frac{2S}{a\lambda} - S$ . A clear discrete-time contribution is  $-(S + 1)$  which depends only on the batch size, while  $\frac{2S}{a\lambda} + 1$  is the tail index

![](images/35ec45be19ec2ce6bbe4892e543d394f7fbf23bf95af56a61466e57989bd4c9f.jpg)  
Figure 2: Comparison of the proposed theory with the continuous-time theory on the SGD stationary distribution for  $a\lambda = 1$ . The proposed theory agrees with the experiment exactly.

in the continuous-time limit (Mori et al., 2021). See Figure 2; the proposed formula agrees with the experiment. Knowing the tail index  $\beta$  is important for understanding the SGD dynamics because  $\beta$  is equal to the smallest moment of  $w$  that diverges. For example, when  $\beta \leq 4$ , then the kurtosis of  $w$  diverges, and one expects to see outliers of  $w$  very often during training; when  $\beta \leq 2$ , then the second moment of  $w$  diverges, and one does not expect  $w$  to converge in the minimum under consideration. Our result suggests that the discrete-time dynamics always leads to a heavier tail than the continuous-time theory expects, and therefore is more unstable.

# 7 OUTLOOK

In this work, we have presented a systematic analysis with a focus on exactly solvable results to promote our fundamental understanding of SGD, and our results are shown to provide a novel and accurate understanding of how SGD works for problems relevant to deep learning. One major limitation is that we have only focused on studying the asymptotic behavior of SGD, and one major future step is to investigate the dynamical, non-asymptotic noise and fluctuations of SGD.

# REFERENCES

Agarwal, N., Bullins, B., and Hazan, E. (2017). Second-order stochastic optimization for machine learning in linear time. The Journal of Machine Learning Research, 18(1):4148-4187.  
Allen-Zhu, Z., Li, Y., and Song, Z. (2019). A convergence theory for deep learning via overparameterization. In International Conference on Machine Learning, pages 242-252. PMLR.  
Amari, S.-I. (1998). Natural gradient works efficiently in learning. Neural Comput., 10(2):251-276.  
Blanc, G., Gupta, N., Valiant, G., and Valiant, P. (2020). Implicit regularization for deep neural networks driven by an Ornstein-uhlenbeck like process. In Conference on learning theory, pages 483-513. PMLR.  
Chizat, L. and Bach, F. (2018). A note on lazy training in supervised differentiable programming. arXiv preprint arXiv:1812.07956, 8.  
Clauset, A., Shalizi, C. R., and Newman, M. E. (2009). Power-law distributions in empirical data. SIAM review, 51(4):661-703.  
Dieuleveut, A., Durmus, A., Bach, F., et al. (2020). Bridging the gap between constant step size stochastic gradient descent and markov chains. Annals of Statistics, 48(3):1348-1382.  
Fontaine, X., Bortoli, V. D., and Durmus, A. (2021). Convergence rates and approximation results for sgd and its continuous-time counterpart.  
Gal, Y. and Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050-1059. PMLR.  
Gitman, I., Lang, H., Zhang, P., and Xiao, L. (2019). Understanding the role of momentum in stochastic gradient methods. In Advances in Neural Information Processing Systems, pages 9633-9643.  
Hammarling, S. J. (1982). Numerical solution of the stable, non-negative definite lyapunov equation lyapunov equation. IMA Journal of Numerical Analysis, 2(3):303-323.  
HaoChen, J. Z., Wei, C., Lee, J. D., and Ma, T. (2020). Shape matters: Understanding the implicit bias of the noise covariance. arXiv preprint arXiv:2006.08680.  
Hastie, T., Montanari, A., Rosset, S., and Tibshirani, R. J. (2019). Surprises in high-dimensional ridgeless least squares interpolation. arXiv preprint arXiv:1903.08560.  
He, F. and Tao, D. (2020). Recent advances in deep learning theory. arXiv preprint arXiv:2012.10931.  
Hodgkinson, L. and Mahoney, M. W. (2020). Multiplicative noise and heavy tails in stochastic optimization. arXiv preprint arXiv:2006.06293.  
Hoffer, E., Hubara, I., and Soudry, D. (2017). Train longer, generalize better: closing the generalization gap in large batch training of neural networks. In Advances in Neural Information Processing Systems, pages 1731-1741.  
Janssen, P. H. M. and Stoica, P. (1988). On the expectation of the product of four matrix-valued gaussian random variables. IEEE Transactions on Automatic Control, 33(9):867-870.  
Jastrzebski, S., Kenton, Z., Arpit, D., Ballas, N., Fischer, A., Storkey, A., and Bengio, Y. (2018). Three factors influencing minima in SGD.  
Kalimeris, D., Kaplun, G., Nakkiran, P., Edelman, B., Yang, T., Barak, B., and Zhang, H. (2019). Sgd on neural networks learns functions of increasing complexity. Advances in Neural Information Processing Systems, 32:3496-3506.  
Kunin, D., Sagastuy-Brena, J., Gillespie, L., Margalit, E., Tanaka, H., Ganguli, S., and Yamins, D. L. (2021). Rethinking the limiting dynamics of sgd: modified loss, phase space oscillations, and anomalous diffusion. arXiv preprint arXiv:2107.09133.

Kunstner, F., Balles, L., and Hennig, P. (2019). Limitations of the empirical fisher approximation for natural gradient descent. arXiv preprint arXiv:1905.12558.  
Levy, M. and Solomon, S. (1996). Power laws are logarithmic boltzmann laws. International Journal of Modern Physics C, 7(04):595-601.  
Lewkowycz, A., Bahri, Y., Dyer, E., Sohl-Dickstein, J., and Gur-Ari, G. (2020). The large learning rate phase of deep learning: the catapult mechanism. arXiv preprint arXiv:2003.02218.  
Liu, K., Ziyin, L., and Ueda, M. (2021). Noise and fluctuation of finite learning rate stochastic gradient descent. arXiv preprint arXiv:2012.03636.  
Lyapunov, A. M. (1992). The general problem of the stability of motion. International journal of control, 55(3):531-534.  
Mandt, S., Hoffman, M. D., and Blei, D. M. (2017). Stochastic gradient descent as approximate bayesian inference. J. Mach. Learn. Res., 18(1):4873-4907.  
Martens, J. (2014). New insights and perspectives on the natural gradient method. cite arxiv:1412.1193Comment: New title and abstract. Added multiple sections, including a proper introduction/outline and one on convergence speed. Many other revisions throughout.  
Meng, Q., Gong, S., Chen, W., Ma, Z.-M., and Liu, T.-Y. (2020). Dynamic of stochastic gradient descent with state-dependent noise. arXiv preprint arXiv:2006.13719.  
Mori, T., Ziyin, L., Liu, K., and Ueda, M. (2021). Logarithmic landscape and power-law escape rate of sgd. arXiv preprint arXiv:2105.09557.  
Pearce, T., Leibfried, F., and Brintrup, A. (2020). Uncertainty in neural networks: Approximately bayesian ensembling. In International conference on artificial intelligence and statistics, pages 234-244. PMLR.  
Sato, I. and Nakagawa, H. (2014). Approximation analysis of stochastic gradient Langevin dynamics by using fokker-planck equation and ito process. In International Conference on Machine Learning, pages 982-990. PMLR.  
Simoncini, V. (2016). Computational methods for linear matrix equations. SIAM Review, 58(3):377-441.  
Simsekli, U., Sagun, L., and Gurbuzbalaban, M. (2019). A tail-index analysis of stochastic gradient noise in deep neural networks. In International Conference on Machine Learning, pages 5827-5837. PMLR.  
Simsekli, U., Sener, O., Deligiannidis, G., and Erdogdu, M. A. (2020). Hausdorff dimension, heavy tails, and generalization in neural networks. Advances in Neural Information Processing Systems, 33.  
Thomas, V., Pedregosa, F., Merrienboer, B., Manzagol, P.-A., Bengio, Y., and Le Roux, N. (2020). On the interplay between noise and curvature and its effect on optimization and generalization. In International Conference on Artificial Intelligence and Statistics, pages 3503-3513. PMLR.  
Toulis, P., Airoldi, E. M., et al. (2017). Asymptotic and finite-sample properties of estimators based on stochastic gradients. Annals of Statistics, 45(4):1694-1727.  
Wang, X., Zhao, Y., and Pourpanah, F. (2020). Recent advances in deep learning. International Journal of Machine Learning and Cybernetics, 11(4):747-750.  
Welling, M. and Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 681-688. Citeseer.  
Wojtowtysch, S. (2021). Stochastic gradient descent with noise of machine learning type. part ii: Continuous time analysis. arXiv preprint arXiv:2106.02588.

Wu, J., Hu, W., Xiong, H., Huan, J., Braverman, V., and Zhu, Z. (2020). On the noisy gradient descent that generalizes as sgd. In International Conference on Machine Learning, pages 10367-10376. PMLR.  
Xie, Z., Sato, I., and Sugiyama, M. (2021). A diffusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima. In International Conference on Learning Representations.  
Xing, C., Arpit, D., Tsirigotis, C., and Bengio, Y. (2018). A walk with sgd. arXiv preprint arXiv:1802.08770.  
Yaida, S. (2019). Fluctuation-dissipation relations for stochastic gradient descent. In International Conference on Learning Representations.  
Ye, H., Michel, A. N., and Hou, L. (1998). Stability theory for hybrid dynamical systems. IEEE transactions on automatic control, 43(4):461-474.  
Zhang, C., Liao, Q., Rakhlin, A., Miranda, B., Golowich, N., and Poggio, T. (2018). Theory of deep learning iib: Optimization properties of sgd. arXiv preprint arXiv:1801.02254.  
Zhiyi, Z. and Ziyin, L. (2021). On the distributional properties of adaptive gradients.  
Zhu, Z., Wu, J., Yu, B., Wu, L., and Ma, J. (2019). The anisotropic noise in stochastic gradient descent: Its behavior of escaping from sharp minima and regularization effects. In International Conference on Machine Learning, pages 7654-7663. PMLR.

![](images/64d18d6788787978236f48747edae85698aeaab2bf03ab04f74565e90087ebc9.jpg)  
Figure 3: Left: 1d experiments with label noise. The parameters are set to be  $a = 1.5$  and  $\lambda = 1$ . Right: Experiments with  $L_{2}$  regularization with weight decay strength  $\gamma$ . The parameters are set to be  $a = 1$ ,  $\lambda = 0.5$ ,  $S = 1$ . This is the standard case with a vanishing optimal  $\gamma$ .

![](images/c88850aae0498ed70c4be266e966a8871648bb95a78b188aa1894ebb2d0d2e14.jpg)

![](images/d3731e37fd4db9051457a04dcedcc03beba4a7bf418127049b568427ff9d0358.jpg)  
(a)  $a = 1, S = 10$

![](images/42bc758092c79dc0cdc4783de223449cab048be37b6c58c52ecf1d8229bd4370.jpg)  
Figure 4: Comparison between theoretical predictions and experiments. (a) 1d experiment. We plot  $\Sigma$  as an increasing function of  $\lambda$ . We see that the continuous-time approximation fails to predict the divergence at a learning rate and the prediction in Liu et al. (2021) severely underestimates the model fluctuation. In contrast, our result is accurate throughout the entire range of learning rates. (b)-(c) 2d experiments. The Hessian has eigenvalues 1 and 0.5, and  $\lambda = 1.5$ . For a large batch size, the discrete-time Hessian approximation is quite accurate; for a small  $S$ , the Hessian approximation underestimates the overall strength of the fluctuation. In contrast, the continuous-time result is both inaccurate in shape and in strength.  
(b)  $S = 50$

![](images/4a7fd9a65a105fb4ff35df20b01b82e80b5aabf4aa6a101d749b4a1bbdbb3b3e.jpg)  
(c)  $S = 10$
