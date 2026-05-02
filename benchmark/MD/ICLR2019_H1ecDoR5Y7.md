# LOCAL STABILITY AND PERFORMANCE OF SIMPLE GRADIENT PENALTY  $\mu$ -WASSERSTEIN GAN

Anonymous authors

Paper under double-blind review

# ABSTRACT

Wasserstein GAN(WGAN) is a model that minimizes the Wasserstein distance between a data distribution and sample distribution. Recent studies have proposed stabilizing the training process for the WGAN and implementing the Lipschitz constraint. In this study, we prove the local stability of optimizing the simple gradient penalty  $\mu$ -WGAN(SGP  $\mu$ -WGAN) under suitable assumptions regarding the equilibrium and penalty measure  $\mu$ . The measure valued differentiation concept is employed to deal with the derivative of the penalty terms, which is helpful for handling abstract singular measures with lower dimensional support. Based on this analysis, we claim that penalizing the data manifold or sample manifold is the key to regularizing the original WGAN with a gradient penalty. Experimental results obtained with unintuitive penalty measures that satisfy our assumptions are also provided to support our theoretical results.

# 1 INTRODUCTION

Deep generative models reached a turning point after generative adversarial networks (GANs) were proposed by Goodfellow et al. (2014). GANs are capable of modeling data with complex structures. For example, DCGAN can sample realistic images using a convolutional neural network (CNN) structure(Radford et al., 2015). GANs have been implemented in many applications in the field of computer vision with good results, such as super-resolution, image translation, and text-to-image generation(Ledig et al., 2017; Isola et al., 2017; Zhang et al., 2017; Reed et al., 2016).

However, despite these successes, GANs are affected by training instability and mode collapse problems. GANs often fail to converge, which can result in unrealistic fake samples. Furthermore, even if GANs successfully synthesize realistic data, the fake samples exhibit little variability. This problem is due to Jensen-Shannon divergence and the low dimensionality of the data manifold.

A common solution to this problem is injecting an instance noise and finding different divergences. The injection of instance noise into real and fake samples during the training procedure was proposed by Sønderby et al. (2017), where its positive impact on the low dimensional support for the data distribution was shown to be a regularizing factor based on the Wasserstein distance, as demonstrated analytically by Arjovsky & Bottou (2017). In  $f$ -GAN,  $f$ -divergence between the target and generator distributions was suggested which generalizes the divergence between two distributions (Nowozin et al., 2016). In addition, a gradient penalty term which is related with Sobolev IPM(Integral Probability Metric) between data distribution and sample distribution was suggested by Mroueh et al. (2018).

The Wasserstein GAN (WGAN) is known to resolve the problems of generic GANs by selecting the Wasserstein distance as the divergence(Arjovsky et al., 2017). However, WGAN often fails with simple examples because the Lipschitz constraint on discriminator is rarely achieved during the optimization process and weight clipping. Thus, mimicking the Lipschitz constraint on the discriminator by using a gradient penalty was proposed by Gulrajani et al. (2017).

Noise injection and regularizing with a gradient penalty appear to be equivalent. The addition of instance noise in  $f$ -GAN can be approximated to adding a zero centered gradient penalty (Roth et al., 2017). Thus, regularizing GAN with a simple gradient penalty term was suggested by Mescheder et al. (2018) who provided a proof of its stability.

Based on a theoretical analysis of the dynamic system, Nagarajan & Kolter (2017) proved the local exponential stability of the gradient-based optimization dynamics in GANs by treating the simultaneous gradient descent algorithm with a dynamic system approach. These previous studies were useful because they showed that the local behavior of GANs can be explained using dynamic system tools and the related Jacobian's eigenvalues.

In this study, we aim to prove the convergence property of the simple gradient penalty  $\mu$ -Wasserstein GAN(SGP  $\mu$ -WGAN) dynamic system under general gradient penalty measures  $\mu$ . To the best of our knowledge, our study is the first theoretical approach to GAN stability analysis which deals with abstract singular penalty measure. In addition, measure valued differentiation(Heidergott & Vázquez-Abad, 2008) is applied to take the derivative on the integral with a parametric measure, which is helpful for handling an abstract measure and its integral in our proof.

The main contributions of this study are as follows.

- We prove the regularized effect and local stability of the dynamic system for a general penalty measure under suitable assumptions. The assumptions are written as both a tractable strong version and intractable weak version. To prove the main theorem, we also introduce the measure valued differentiation concept to handle the parametric measure.  
- Based on the proof of the stability, we explain the reason for the success of previous penalty measures. We claim that the support of a penalty measure will be strongly related to the stability, where the weight on the limiting penalty measure might affect the speed of convergence.  
- We experimentally examined the general convergence results by applying two test penalty measures to several examples. The proposed test measures are unintuitive but they still satisfy the assumptions and similar convergence results were obtained in the experiment.

# 2 PRELIMINARIES

First, we introduce our notations and basic measure-theoretic concepts. Second, we define our SGP  $\mu$ -WGAN optimization problem and treat this problem as a continuous dynamic system. Preliminary measure theoretic concepts are required to justify that the dynamic system changes in a sufficiently smooth manner as the parameter changes, so it is possible to use linearization theorem. They are also important for dealing with the parametric measure and its derivative. The problem setting with a simple gradient term is also discussed. The squared gradient size and simple gradient penalty term are used to build a differentiable dynamic system and to apply soft regularization as a resolving constraint, respectively. The continuous dynamic system approach, which is a so-called ODE method, is used to analyze the GAN optimization problem with the simultaneous gradient descent algorithm, as described by Nagarajan & Kolter (2017).

# 2.1 NOTATIONS AND PRELIMINARIES REGARDING MEASURE THEORY

$D(x;\psi):\mathcal{X}\to \mathbb{R}$  is a discriminator function with its parameter  $\psi$  and  $G(z;\theta):\mathcal{Z}\rightarrow \mathcal{X}$  is a generator function with its parameter  $\theta$ .  $p_d$  is the distribution of real data and  $p_g = p_\theta$  is the distribution of the generated samples in  $\mathcal{X}$ , which is induced from the generator function  $G(z;\theta)$  and a known initial distribution  $p_{\text{latent}}(z)$  in the latent space  $\mathcal{Z}$ .  $\|\cdot\|$  denotes the  $L^2$  Euclidean norm if no special subscript is present.

The concept of weak convergence for finite measures is used to ensure the continuity of the integral term over the measure in the dynamic system, which must be checked before applying the theorems related to stability. Throughout this study, we assume that the measures in the sample space are all finite and bounded.

Definition 1. For a set of finite measures  $\{\mu_i\}_{i\in \mathcal{I}}$  in the metric space  $(\mathcal{X},d)$  with metric  $d$  and Borel  $\sigma$ -algebra  $\mathcal{B}(\mathcal{X})$ ,  $\{\mu_i\}_{i\in \mathcal{I}}$  is referred to as bounded if there exists some  $M > 0$  such that for all  $i\in \mathcal{I}$ ,

$$
\mu_ {i} (\mathcal {X}) \leq M
$$

For instance,  $M$  can be set as 1 if  $\{\mu_i\}$  are probability measures on  $\mathbb{R}^n$ . Assuming that the penalty measures are bounded, Portmanteau theorem offers the equivalent definition of the weak conver

gence for finite measures. This definition is important for ensuring that the integrals over  $p_{\theta}$  and  $\mu$  in the dynamic system change continuously.

Definition 2. (Portmanteau Theorem) For a bounded sequence of finite measures  $\{\mu_n\}_{n\in \mathbb{N}}$  on the Euclidean space  $\mathbb{R}^n$  with a  $\sigma$ -field of Borel subsets  $\mathcal{B}(\mathbb{R}^n)$ ,  $\mu_{n}$  converges weakly to  $\mu$  if and only if for every continuous bounded function  $\phi$  on  $\mathbb{R}^n$ , its integrals with respect to  $\mu_{n}$  converge to  $\int \phi d\mu$ , i.e.,

$$
\mu_ {n} \rightarrow \mu \Longleftrightarrow \int \phi d \mu_ {n} \rightarrow \int \phi d \mu
$$

The most challenging problem in our analysis with the general penalty measure is taking the derivative of the integral, where the measure depends on the variable that we want to differentiate. If our penalty measure is either absolutely continuous or discrete, then it is easy to deal with the integral. However, in the case of singular penalty measure, dealing with the integral term is not an easy task. Therefore, we introduce the concept of a weak derivative of a probability measure in the following(Heidergott & Vázquez-Abad, 2008). The weak derivative of a measure is useful for handling a parametric measure that is not absolutely continuous with low dimensional support.

Definition 3. (Weak Derivatives of a Probability Measure) Consider the Euclidean space and its  $\sigma$ -field of Borel subsets  $(\mathbb{R}^d, \mathcal{B}(\mathbb{R}^d))$ . The probability measure  $P_{\theta}$  is called weakly differentiable at  $\theta$  if a signed finite measure  $P_{\theta}'$  exists where

$$
\frac {d}{d \theta} \int \phi (x) d P _ {\theta} = \lim _ {\Delta \to 0} \frac {1}{\Delta} \{\int \phi (x) d P _ {\theta + \Delta} - \int \phi (x) d P _ {\theta} \} = \int \phi (x) d P _ {\theta} ^ {\prime}
$$

is satisfied for every continuous bounded function  $\phi$  on  $\mathbb{R}^n$ . For the multidimensional parameter  $\theta$ , this can be defined similar manner.

We can show that the positive part and negative part of  $P_{\theta}^{\prime}$  have the same mass by putting  $\phi(x) = 1$  and the Hahn-Jordan decomposition on  $P_{\theta}^{\prime}$ . Therefore, the following triple  $(c_{\theta}, P_{\theta}^{+}, P_{\theta}^{-})$  is called a weak derivative of  $P_{\theta}$ , where  $P_{\theta}^{\pm}$  are probability measures and  $P_{\theta}^{\prime}$  is rewritten as:

$$
P _ {\theta} ^ {\prime} = c _ {\theta} P _ {\theta} ^ {+} - c _ {\theta} P _ {\theta} ^ {-}
$$

Therefore,

$$
\frac {d}{d \theta} \int \phi (x) d P _ {\theta} = \int \phi (x) d P _ {\theta} ^ {\prime} = c _ {\theta} \left(\int \phi (x) d P _ {\theta} ^ {+} - \int \phi (x) d P _ {\theta} ^ {-}\right)
$$

holds for every continuous bounded function  $\phi$  on  $\mathbb{R}^n$ . It is known that the representation of  $(c_{\theta}, P_{\theta}^{+}, P_{\theta}^{-})$  for  $P_{\theta}'$  is not unique because  $(c_{\theta} + C_{\theta}, P_{\theta}^{+} + q_{\theta}, P_{\theta}^{-} + q_{\theta})$  is also another representation of  $P_{\theta}'$ .

For the general finite measure  $Q_{\theta}$ , a normalizing coefficient  $M(\theta) < \infty$  can be introduced. The product rule for differentiating can also be applied in a similar manner to calculus.

$$
\frac {d}{d \theta} \int \phi (x; \theta) d P _ {\theta} = \int \nabla_ {\theta} \phi (x; \theta) d P _ {\theta} + \int \phi (x; \theta) d P _ {\theta} ^ {\prime}
$$

Therefore, for the general finite measure  $Q_{\theta} = M(\theta)P_{\theta}$ , its derivative  $Q_{\theta}'$  can be represented as below.

$$
Q _ {\theta} ^ {\prime} = M ^ {\prime} (\theta) P _ {\theta} + M (\theta) P _ {\theta} ^ {\prime} = M ^ {\prime} (\theta) P _ {\theta} + c _ {\theta} M (\theta) P _ {\theta} ^ {+} - c _ {\theta} M (\theta) P _ {\theta} ^ {-}
$$

# 2.2 PROBLEM SETTING AS A DYNAMIC SYSTEM

Previous work of Mescheder et al. (2018) showed that the dynamic system of WGAN-GP is not necessarily stable at equilibrium by demonstrating that the sequence of parameters is not Cauchy sequence. This is mainly due to the term  $\| x\|$  in the dynamic system which has a derivative  $\frac{x}{\|x\|}$  that is not defined at  $x = 0$ . WGAN-GP has a penalty term  $\mathbb{E}_{\mu_{GP}}[(\|\nabla_xD(x;\psi)\| - 1)^2]$  that can lead to a discontinuity in its dynamic system.

These problems can be avoided by using the squared value of the gradient's norm  $\| \nabla_x D \|^2$ , which is a differentiable function. In contrast to the WGAN-GP, recent methods based on a gradient penalty such as the simple gradient penalty employed by Mescheder et al. (2018) and the Sobolev GAN used

the average of the squared values for the penalty area, whereas the WGAN-GP penalizes the size of the discriminator's gradient  $\| \nabla_x D\|$  away from 1 in a pointwise manner.

This advantage of squared gradient term $^1$ ,  $\mathbb{E}_{\mu}[\| \nabla_x D\|^2]$ , makes the dynamic system differentiable and we define the WGAN problem with the square of the gradient's norm as a simple gradient penalty. This simple gradient penalty can be treated as soft regularization based on the size of the discriminator's gradient, especially in case where  $\mu$  is the probability measure (Roth et al., 2017). It is convenient to determine whether the system is stable by observing the spectrum of the Jacobian matrix. In the following,  $(D(x; \psi), p_d, p_\theta, \mu)$  is defined as an SGP  $\mu$ -WGAN optimization problem (SGP-form) with a simple gradient penalty term on the penalty measure  $\mu$ .

Definition 4. The WGAN optimization problem with a simple gradient penalty term  $\| \nabla_x D \|^2$ , penalty measure  $\mu$ , and penalty weight hyperparameter  $\rho > 0$  is given as follows, where the penalty term is only introduced to update the discriminator:

$$
\max  _ {\psi}: \mathbb {E} _ {p _ {d}} [ D (x; \psi) ] - \mathbb {E} _ {p _ {\theta}} [ D (x; \psi) ] - \frac {\rho}{2} \mathbb {E} _ {\mu} [ \| \nabla_ {x} D (x; \psi) \| ^ {2} ]
$$

$$
\min _ {\theta}: \mathbb {E} _ {p _ {d}} [ D (x; \psi) ] - \mathbb {E} _ {p _ {\theta}} [ D (x; \psi) ]
$$

According to Nagarajan & Kolter (2017) and many other optimization problem studies, the simultaneous gradient descent algorithm for GAN updating can be viewed as an autonomous dynamic system of discriminator parameters and generator parameters, which we denote as  $\psi$  and  $\theta$ . As a result, the related dynamic system is given as follows.

$$
\dot {\psi} = \mathbb {E} _ {p _ {d}} [ \nabla_ {\psi} D ] - \mathbb {E} _ {p _ {\theta}} [ \nabla_ {\psi} D ] - \frac {\rho}{2} \nabla_ {\psi} \mathbb {E} _ {\mu} [ \nabla_ {x} ^ {T} D \nabla_ {x} D ]
$$

$$
\dot {\theta} = \nabla_ {\theta} \mathbb {E} _ {p _ {\theta}} [ D ]
$$

# 3 TOY EXAMPLES

We investigate two examples considered in previous studies by Mescheder et al. (2018) and Nagarajan & Kolter (2017). We then generalize the results to a finite measure case. The first example is the univariate Dirac GAN, which was introduced by Mescheder et al. (2018).

Definition 5. (Dirac GAN) The Dirac GAN comprises a linear discriminator  $D(x; \psi) = \psi x$ , data distribution  $p_d = \delta_0$ , and sample distribution  $p_\theta = \delta_\theta$ .

The Dirac GAN with a gradient penalty with an arbitrary probability measure is known to be globally convergent(Mescheder et al., 2018). We argue that this result can be generalized to a finite penalty measure case.

Lemma 1. Consider the Dirac GAN problem with SGP form  $(D(x;\psi) = \psi x,\delta_0,\delta_\theta ,\mu_{\psi ,\theta})$  . Suppose that some small  $\eta >0$  exists such that its finite penalty measure  $\mu_{\psi ,\theta}$  with mass  $M(\psi ,\theta) =$ $\int 1d\mu_{\psi ,\theta}\geq 0$  satisfies either

-  $M(\psi, \theta) > 0$  for  $(\psi, \theta) \in B_{\eta}((0,0))$  or  
-  $M(0,0) = 0$  and  $\psi \nabla_{\psi} M(\psi, \theta) \geq 0$  for  $(\psi, \theta) \in B_{\eta}((0,0))$ .

Then, the SGP  $\mu$ -WGAN optimization dynamics with  $(D(x;\psi) = \psi x, \delta_0, \delta_\theta, \mu_{\psi,\theta})$  are locally stable at the origin and the basin of attraction  $B = B_R((0,0))$  is open ball with radius  $R$ . Its radius is given as follows.

$$
R = \max  \left\{\eta \geq 0 | 2 M (\psi , \theta) + \psi \nabla_ {\psi} M (\psi , \theta) \geq 0 f o r a l l (\psi , \theta) s u c h t h a t \psi^ {2} + \theta^ {2} \leq \eta^ {2} \right\}
$$

Motivated by this example, we can extend this idea to the other toy example given by Nagarajan & Kolter (2017), where WGAN fails to converge to the equilibrium points  $(\psi ,\theta) = (0,\pm 1)$ .

Lemma 2. Consider the toy example  $(D(x;\psi) = \psi x^2, U(-1,1), U(-|\theta|, |\theta|), \mu_\theta)$  where  $U(0,0) = \delta_0$  and the ideal equilibrium points are given by  $(\psi^*, \theta^*) = (0, \pm 1)$ . For a finite measure  $\mu = \mu_\theta$  on  $\mathbb{R}$  which is independent of  $\psi$ , suppose that  $\mu_\theta \to \mu^*$  with  $\mu^* \neq C\delta_0$  for  $C \geq 0$ . The dynamic system is locally stable near the desired equilibrium  $(0, \pm 1)$ , where the spectrum of the Jacobian at  $(0, \pm 1)$  is given by  $\lambda = -2\rho \mathbb{E}_{\mu^*}[x^2] \pm \sqrt{4\rho^2\mathbb{E}_{\mu^*}[x^2]^2 - \frac{4}{9}}$ .

# 4 MAIN CONVERGENCE THEOREM

We propose the convergence property of WGAN with a simple gradient penalty on an arbitrary penalty measure  $\mu$  for a realizable case:  $\theta = \theta^{*}$  with  $p_d = p_{\theta^*}$  exists. In subsection 4.1, we provide the necessary assumptions, which comprise our main convergence theorem. In subsection 4.2, we give the main convergence theorem with a sketch of the proof. A more rigorous analysis is given in the Appendix.

# 4.1 ASSUMPTIONS

The first assumption is made regarding the equilibrium condition for GANs, where we state the ideal conditions for the discriminator parameter and generator parameter. As the parameters converge to the ideal equilibrium, the sample distribution  $(p_{\theta})$  converges to the real data distribution  $(p_d)$  and the discriminator cannot distinguish the generated sample and the real data.

Assumption 1.  $p_{\theta} \to p_d$  as  $\theta \to \theta^{*}$  and  $D(x; \psi^{*}) = 0$  on  $\operatorname{supp}(p_d)$  and its small open neighborhood, i.e.,  $x \in \cup_{x' \in \operatorname{supp}(p_d)} B_{\epsilon_{x'}}(x')$  implies  $D(x; \psi^{*}) = 0$ . For simplicity, we denote  $\cup_{x' \in \operatorname{supp}(p_d)} B_{\epsilon_{x'}}(x')$  as  $B(\operatorname{supp}(p_d))$ .

The second assumption ensures that the higher order terms cannot affect the stability of the SGP  $\mu$ -WGAN. In the Appendix, we consider the case where the WGAN fails to converge when Assumption 2 is not satisfied. Compared with the previous study by Nagarajan & Kolter (2017), the conditions for the discriminator parameter are slightly modified.

# Assumption 2.

$$
g (\theta) = \| \mathbb {E} _ {p _ {d}} [ \nabla_ {\psi} D (x; \psi^ {*}) ] - \mathbb {E} _ {p _ {\theta}} [ \nabla_ {\psi} D (x; \psi^ {*}) ] \| ^ {2}, h (\psi) = \mathbb {E} _ {\mu_ {\psi , \theta^ {*}}} [ \| \nabla_ {x} D (x; \psi) \| ^ {2} ]
$$

are locally constant along the nullspace of the Hessian matrix.

The third assumption allows us to extend our results to discrete probability distribution cases, as described by Mescheder et al. (2018).

Assumption 3.  $\exists \epsilon_{g} > 0$  such that  $D(x;\psi^{*}) = 0$  on  $\cup_{|\theta -\theta^{*}| <   \epsilon_{g}}supp(p_{\theta})$

The fourth assumption indicates that there are no other "bad" equilibrium points near  $(\psi^{*},\theta^{*})$  which justifies the projection along the axis perpendicular to the null space.

Assumption 4. A bad equilibrium does not exist near the desired equilibrium point. Thus,  $(\psi^{*},\theta^{*})$  is an isolated equilibrium or there exist  $\delta_d,\delta_g > 0$  such that all equilibrium points in  $B_{\delta_d}(\psi^*)\times$ $B_{\delta_g}(\theta^*)$  satisfy the other assumptions.

The last assumption is related to the necessary conditions for the penalty measure. A calculation of the gradient penalty based on samples from the data manifold and generator manifold or the interpolation of both was introduced in recent studies (Gulrajani et al., 2017; Roth et al., 2017; Mescheder et al., 2018). First, we propose strong conditions for the penalty measure.

Assumption 5. The finite penalty measure  $\mu = \mu_{\theta}$  satisfies the followings:

a  $\mu_{\theta}\rightarrow \mu_{\theta^{*}} = \mu^{*}$  and  $\mu_{\theta}$  is independent of the discriminator parameter  $\psi$

$b\operatorname {supp}(p_d)\subset \operatorname {supp}(\mu^*)$

$c\exists \epsilon_{\mu} > 0$  such that  $supp(\mu_{\theta})\subset B(supp(p_d))$  for  $|\theta -\theta^{*}| <   \epsilon_{\mu}$

The assumption given above means that the support of the penalty measure  $\mu_{\theta}$  should approach the support of the data manifolds smoothly as  $\theta \rightarrow \theta^{*}$ . Thus, the gradient penalty should be evaluated based on the data manifold and some open neighborhood  $B(supp(p_d))$  near the equilibrium.

However, the penalty measure from WGAN-GP with a simple gradient penalty still reaches equilibrium without satisfying Assumption 5c. Therefore, we suggest Assumption 6, which is a weak version of Assumption 5. Assumption  $6a^2$  is technically required to take the derivative of the integral  $\mathbb{E}_{\mu_{\psi,\theta}}[\|\nabla_xD(x;\psi)\|^2]$  with respect to  $\psi$ .

Assumption 6. (Weak version of Assumption 5) The finite penalty measure  $\mu = \mu_{\psi, \theta}$  satisfies the following.

a  $\mu_{\psi,\theta} \to \mu_{\psi^*,\theta^*} = \mu^*$ , where  $\operatorname{supp}(\mu_{\psi,\theta})$  only depends on  $\theta$ . Near the equilibrium,  $\mu_{\psi,\theta}$  can be weakly differentiated twice with respect to  $\psi$ . In addition, its mass  $M(\psi,\theta) = \int 1d\mu_{\psi,\theta}$  is a twice-differentiable function of  $\psi$  and bounded by  $M_1 < \infty$  near the equilibrium.

b  $E_{\mu^{*}}[\nabla_{\psi x}D\nabla_{\psi x}^{T}D]$  is positive definite or  $\operatorname {supp}(p_d)\subset \operatorname {supp}(\mu^*)$

$c\exists \epsilon_{\mu} > 0$  such that  $supp(\mu_{\theta})\subset V$  for  $|\theta -\theta^{*}| <   \epsilon_{\mu}$  , where  $V = \{x|\nabla_{x}D(x;\psi^{*}) = 0\}$

In summary, the gradient penalty regularization term with any penalty measure where the support approaches  $B(supp(p_d))$  in a smooth manner works well and this main result can explain the regularization effect of previously proposed penalty measures such as  $\mu_{GP}$ ,  $p_d$ ,  $p_\theta$ , and their mixtures.

# 4.2 MAIN CONVERGENCE THEOREM

According to the modified assumptions given above, we prove that the related dynamic system is locally stable near the equilibrium. The tools used for analyzing stability are mainly based on those described by Nagarajan & Kolter (2017). Our main contributions comprise proposing the necessary conditions for the penalty measure and proving the local stability for all penalty measures that satisfy Assumption 6.

Theorem 1. Suppose that our SGP  $\mu$ -WGAN optimization problem  $(D, p_d, p_\theta, \mu)$  with equilibrium point  $(\psi^*, \theta^*)$  satisfies the assumptions given above. Then, the related dynamic system is locally stable at the equilibrium.

A detailed proof of the main convergence theorem is given in the Appendix. A sketch of the proof is given in three steps. First, the undesired terms in the Jacobian matrix of the system at the equilibrium are cancelled out. Next, the Jacobian matrix at equilibrium is given by  $\left[ \begin{array}{cc} -\rho Q & -R\\ R^T & 0 \end{array} \right]$ , where  $Q = \mathbb{E}_{\mu^{*}}[\nabla_{\psi x}D\nabla_{\psi x}^{T}D]$  and  $R = \nabla_{\theta}\mathbb{E}_{p_{\theta}}[\nabla_{\psi}D]|_{\theta = \theta^{*}}$ . The system is locally stable when both  $Q$  and  $R^T R$  are positive definite. We can complete the proof by dealing with zero eigenvalues by showing that  $N(Q^{T})\subset N(R^{T})$  and the projected system's stability implies the original system's stability.

Our analysis mainly focuses on WGAN, which is the simplest case of general GAN minimax optimization

$$
\max  _ {\psi}: \mathbb {E} _ {p _ {d}} [ f (D (x; \psi)) ] + \mathbb {E} _ {p _ {\theta}} [ f (- D (x; \psi)) ] - \frac {\rho}{2} \mathbb {E} _ {\mu} [ \| \nabla_ {x} D (x; \psi) \| ^ {2} ]
$$

$$
\min _ {\theta}: \mathbb {E} _ {p _ {d}} [ f (D (x; \psi)) ] + \mathbb {E} _ {p _ {\theta}} [ f (- D (x; \psi)) ]
$$

with  $f(x) = x$ . Similar approach is still valid for general GANs with concave function  $f$  with  $f''(x) < 0$  and  $f'(0) \neq 0$ .

# 5 EXPERIMENTAL RESULTS

We claim that every penalty measure that satisfies the assumptions can regularize the WGAN and generate similar results to the recently proposed gradient penalty methods. Several penalty measures were tested based on two-dimensional problems (mixture of 8 Gaussians, mixture of 25 Gaussians,

and swissroll), MNIST and CIFAR-10 datasets using a simple gradient penalty term. In the comparisons with WGAN, the recently proposed penalty measures and our test penalty measures used the same network settings and hyperparameters. The penalty measures and its detailed sampling methods are listed in Table 1, where  $x_{d} \sim p_{d}, x_{g} \sim p_{\theta}$ , and  $\alpha \sim U(0,1)$ .  $\mathcal{A}$  indicates fixed anchor point in  $\mathcal{X}$ .

Table 1: List of benchmark WGANs (WGAN and WGAN-GP with non-zero centered gradient penalty) and 5 penalty measures with a simple gradient penalty term. In this table, WGAN-GP represents the previous model proposed by (Gulrajani et al., 2017), which penalizes the WGAN with non-zero centered gradient penalty terms, whereas  $\mu_{GP}$  represents the simple method. In our experiment, no additional weights are applied on 5 penalty measures and they are all probability distributions.

<table><tr><td>Penalty</td><td>Penalty term</td><td>Penalty measure, sampling method</td></tr><tr><td>WGAN</td><td>None(Weight Clipping)</td><td>None</td></tr><tr><td>WGAN-GP</td><td>Eμ[||∇xD||-1]2</td><td>x̂ = αxd + (1 - α)xg</td></tr><tr><td>pg</td><td>Eμ[||∇xD||2]</td><td>x̂ = xg</td></tr><tr><td>pd</td><td>Eμ[||∇xD||2]</td><td>x̂ = xd</td></tr><tr><td>μGP</td><td>Eμ[||∇xD||2]</td><td>x̂ = αxd + (1 - α)xg</td></tr><tr><td>μmid</td><td>Eμ[||∇xD||2]</td><td>x̂ = 0.5xd + 0.5xg</td></tr><tr><td>μg,anc</td><td>Eμ[||∇xD||2]</td><td>x̂ = αA + (1 - α)xg</td></tr></table>

By setting the previously proposed WGAN with weight-clipping (Arjovsky et al., 2017) and WGAN-GP (Gulrajani et al., 2017) as the baseline models, SGP  $\mu$ -WGAN was examined with various penalty measures comprising three recently proposed measures and two artificially generated measures.  $p_{\theta}$  and  $p_d$  were suggested by Mescheder et al. (2018) and  $\mu_{GP}$  was introduced from the WGAN-GP. We analyzed the artificial penalty measures  $\mu_{mid}$  and  $\mu_{g,anc}$  as the test penalty measures.

The experiments were conducted based on the implementation of the Gulrajani et al. (2017). The hyperparameters, generator/discriminator structures, and related TensorFlow implementations can be found at https://github.com/igul222/improved_wgan_training (Gulrajani et al., 2017). Only the loss function was modified slightly from a non-zero centered gradient penalty to a simple penalty. For the CIFAR-10 image generation tasks, the inception score(Salimans et al., 2016) and FID(Heusel et al., 2017) were used as benchmark scores to evaluate the generated images.

# 5.1 2D EXAMPLES AND MNIST

We checked the convergence of  $p_{\theta}$  for the 2D examples (8 Gaussians, swissroll data, and 25 Gaussians) and MNIST digit generation for the SGP-WGANs with five penalty measures. MNIST and 25 Gaussians were trained over 200K iterations, the 8 Gaussians were trained for 30K iterations, and the Swiss Roll data were trained for 100K iterations. The anchor  $\mathcal{A}$  for  $\mu_{g,\text{anc}}$  was set as  $(2, -1)$  for the 2D examples and 784 gray pixels for MNIST. We only present the results obtained for the MNIST dataset with the penalty measures comprising  $\mu_{mid}$  and  $\mu_{g,\text{anc}}$  in Figure 1. The others are presented in the Appendix.

![](images/0e42548a3ebb376fcbacbee033bc3c999ee8bebee011c40248783f38f9279794.jpg)  
Figure 1: MNIST example. Images generated with  $\mu_{mid}$  (left) and  $\mu_{g,anc}$  (right).

# 5.2 CIFAR-10

DCGAN and ResNet architectures were tested on the CIFAR-10 dataset. The generators were trained for 200K iterations. The anchor  $\mathcal{A}$  for  $\mu_{g,anc}$  during CIFAR-10 generation was set as fixed random pixels. The WGAN, WGAN-GP, and five penalty measures were evaluated based on the inception score and FID, as shown in Table 2, which are useful tools for scoring the quality of generated images. The images generated from  $\mu_{mid}$  and  $\mu_{g,anc}$  with ResNet are shown in Figure 2. The others are presented in the Appendix.

Table 2: Benchmark score results obtained based on the CIFAR-10 dataset under DCGAN and ResNet architectures. The higher inception score and lower FID indicate the good quality of the generated images.  

<table><tr><td rowspan="2">Penalty</td><td colspan="2">DCGAN</td><td colspan="2">ResNet</td></tr><tr><td>Inception</td><td>FID</td><td>Inception</td><td>FID</td></tr><tr><td>WGAN3</td><td>5.64 ± 0.09</td><td>48.7</td><td>-</td><td>-</td></tr><tr><td>WGAN-GP</td><td>6.48 ± 0.10</td><td>35.0</td><td>7.82 ± 0.09</td><td>18.1</td></tr><tr><td>pg</td><td>6.46 ± 0.09</td><td>38.0</td><td>7.63 ± 0.10</td><td>20.9</td></tr><tr><td>pd</td><td>6.33 ± 0.07</td><td>38.9</td><td>7.63 ± 0.09</td><td>20.3</td></tr><tr><td>μGP</td><td>6.40 ± 0.08</td><td>35.4</td><td>7.60 ± 0.09</td><td>18.3</td></tr><tr><td>μmid</td><td>6.60 ± 0.07</td><td>33.9</td><td>7.86 ± 0.07</td><td>16.4</td></tr><tr><td>μg,anc</td><td>6.45 ± 0.07</td><td>33.7</td><td>7.36 ± 0.09</td><td>22.4</td></tr></table>

![](images/05d2ff5f4dbccd69ee7c56687e16791afa0add3be264c61b07705a533545dba4.jpg)  
Figure 2: CIFAR-10 example. Images generated with  $\mu_{mid}$  (left) and  $\mu_{g,anc}$  (right) under the ResNet architecture.

# 6 CONCLUSION

In this study, we proved the local stability of simple gradient penalty  $\mu$ -WGAN optimization for a general class of finite measure  $\mu$ . This proof provides insight into the success of regularization with previously proposed penalty measures. We explored previously proposed analyses based on various gradient penalty methods. Furthermore, our theoretical approach was supported by experiments using unintuitive penalty measures. In future research, our works can be extended to alternative gradient descent algorithm and its related optimal hyperparameters. Stability at non-realizable equilibrium points is one of the important topics on stability of GANs. Optimal penalty measure for achieving the best convergence speed can be also investigated using a spectral theory, which provides the mathematical analysis on stability of GAN with a precise information on the convergence theory.

# REFERENCES

Martín Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. In International Conference on Learning Representations, 2017.  
Martín Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In Proceedings of the 34th International Conference on Machine Learning, pp. 214-223, 2017.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C. Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pp. 5769-5779, 2017.  
B. Heidergott and F. J. Vázquez-Abad. Measure-valued differentiation for markov chains. Journal of Optimization Theory and Applications, 136:187-209, 2008. ISSN 1573-2878. doi: 10.1007/s10957-007-9297-7.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, pp. 6629-6640, 2017.  
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 5967-5976, 2017.  
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew P. Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photorealistic single image super-resolution using a generative adversarial network. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 105-114, 2017.  
Lars M. Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In Proceedings of the 35th International Conference on Machine Learning, pp. 3478-3487, 2018.  
Youssef Mroueh, Chun-Liang Li, Tom Sercu, Anant Raj, and Yu Cheng. Sobolev GAN. In International Conference on Learning Representations, 2018.  
Vaishnavh Nagarajan and J. Zico Kolter. Gradient descent GAN optimization is locally stable. In Advances in Neural Information Processing Systems, pp. 5591-5600, 2017.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. In Advances in Neural Information Processing Systems, pp. 271-279, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. CoRR, abs/1511.06434, 2015. URL http:// arxiv.org/abs/1511.06434.  
Scott E. Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, pp. 1060-1069, 2016. URL http://jmlr.org/proceedings/papers/v48/reed16.html.  
Kevin Roth, Aurélien Lucchi, Sebastian Nowozin, and Thomas Hofmann. Stabilizing training of generative adversarial networks through regularization. In Advances in Neural Information Processing Systems, pp. 2015-2025, 2017.  
Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2226-2234, 2016.

Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Huszár. Amortised MAP inference for image super-resolution. International Conference on Learning Representations, 2017.  
Han Zhang, Tao Xu, and Hongsheng Li. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pp. 5908-5916, 2017.
