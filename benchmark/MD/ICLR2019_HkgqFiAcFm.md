# MARGINAL POLICY GRADIENTS: A UNIFIED FAMILY OF ESTIMATORS FOR BOUNDED ACTION SPACES WITH APPLICATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Many complex domains, such as robotics control and real-time strategy (RTS) games, require an agent to learn a continuous control. In the former, an agent learns a policy over  $\mathbb{R}^d$  and in the latter, over a discrete set of actions each of which is parametrized by a continuous parameter. Such problems are naturally solved using policy based reinforcement learning (RL) methods, but unfortunately these often suffer from high variance leading to instability and slow convergence. Unnecessary variance is introduced whenever policies over bounded action spaces are modeled using distributions with unbounded support by applying a transformation  $T$  to the sampled action before execution in the environment. Recently, the variance reduced clipped action policy gradient (CAPG) was introduced for actions in bounded intervals, but to date no variance reduced methods exist when the action is a direction, something often seen in RTS games. To this end we introduce the angular policy gradient (APG), a stochastic policy gradient method for directional control. With the marginal policy gradients family of estimators we present a unified analysis of the variance reduction properties of APG and CAPG; our results provide a stronger guarantee than existing analyses for CAPG. Experimental results on a popular RTS game and a navigation task show that the APG estimator offers a substantial improvement over the standard policy gradient.

# 1 INTRODUCTION

Recent work in deep reinforcement learning (RL) has achieved human level-control for complex tasks like Atari 2600 games and the ancient game of Go. Mnih et al. (2015) show that it is possible to learn to play Atari 2600 games using end-to-end reinforcement learning. Other authors (Silver et al., 2014) derive algorithms tailored to continuous action spaces, such as appear in problems of robotics control. Today, solving RTS games is a major open problem in RL (Foerster et al., 2016; Usunier et al., 2017; Vinyals et al., 2017); these are more challenging than previously solved game domains because the action and state spaces are far larger. In RTS games, actions are no longer chosen from a relatively small discrete action set as in other game types. Neither is the objective solely learning a continuous control. Instead the action space typically consists of many discrete actions each of which has a continuous parameter. For example, a discrete action in an RTS game might be moving the player controlled by the agent with a parameter specifying the movement direction. Because the agent must learn a continuous parameter for each discrete action, a policy gradient method is a natural approach to an RTS game. Unfortunately, obtaining stable, sample-efficient performance from policy gradients remains a key challenge in model-free RL.

Just as robotics control tasks often have actions restricted to a bounded interval, Multi-player Online Battle Arena (MOBA) games, an RTS sub-genre, often have actions restricted to the unit sphere which specify a direction (e.g. to move or attack). The current practice, despite most continuous control problems having bounded action spaces, is to use a Gaussian distribution to model the policy and then apply a transformation  $T$  to the action  $a$  before execution in the environment. This support mismatch between the sampling action distribution (i.e. the policy  $\pi$ ), and the effective action distribution can both introduce bias to and increase the variance of policy gradient estimates (Chou et al., 2017; Fujita and Maeda, 2018). For an illustration of how the distribution over actions  $a$  is transformed under  $T(a) = a / ||a||$ , see Figure 1 in Section 3.

In this paper, motivated by an application to a MOBA game, we study policy gradient methods in the context of directional actions, something unexplored in the RL literature. Like CAPG for actions in an interval  $[\alpha, \beta]$ , our proposed algorithm, termed angular policy gradient (APG), uses a variance-reduced, unbiased estimated of the true policy gradient. Since the key step in APG is an update based on an estimate of the policy gradient, it can easily be combined with other state-of-the-art methodology including value function approximation and generalized advantage estimation (Sutton et al., 2000; Schulman et al., 2016), as well as used in policy optimization algorithms like TRPO, A3C, and PPO (Schulman et al., 2015; Mnih et al., 2016; Schulman et al., 2017).

Beyond new methodology, we also introduce the marginal policy gradients (MPG) family of estimators; this general class of estimators contains both APG and CAPG, and we present a unified analysis of the variance reduction properties of all such methods. Because marginal policy gradient methods have already been shown to provide substantial benefits for clipped actions (Fujita and Maeda, 2018), our experimental work focuses only on angular actions; we use a marginal policy gradient method to learn a policy for the 1 vs. 1 map of the King of Glory game and the Platform2D-v1 navigation task, demonstrating improvement over several baseline policy gradient approaches.

# 1.1 RELATED WORK

Model-Free RL. Policy based methods are appealing because unlike value based methods they can support learning policies over discrete, continuous and parametrized action spaces. It has long been recognized that policy gradient methods suffer from high variance, hence the introduction of trust region methods like TRPO and PPO (Schulman et al., 2015; 2017). Mnih et al. (2016) leverage the independence of asynchronous updating to improve stability in actor-critic methods. See Sutton and Barto (2018) for a general survey of reinforcement learning algorithms, including policy based and actor-critic methods. Recent works have applied policy gradient methods to parametrized action spaces in order to teach an agent to play RoboCup soccer (Hausknecht and Stone, 2016; Masson et al., 2016). Formally, a parametrized action space  $\mathcal{A}$  over  $K$  discrete, parametrized actions is defined as  $\mathcal{A} \coloneqq \bigcup_{k} \{(k,\omega) : \omega \in \Omega_k\}$ , where  $k \in [K]$  and  $\Omega_k$  is the parameter space for the  $k^{th}$  action. See Appendix B.5 for rigorous discussion of the construction of a distribution over parametrized action spaces and the corresponding policy gradient algorithms.

Bounded Action Spaces. Though the action space for many problems is bounded, it is nonetheless common to model a continuous action using the multivariate Gaussian, which has unbounded support (Hausknecht and Stone, 2016; Florensa et al., 2017; Finn et al., 2017). Until recently, the method for dealing with this type of action space was to sample according to a Gaussian policy and then either (1) allow the environment to clip the action and update according to the unclipped action or (2) clip the action and update according to the clipped action (Chou et al., 2017). The first approach suffers from unnecessarily high variance, and the second approach is off-policy.

Recent work considers variance reduction when actions are clipped to a bounded interval (Chou et al., 2017; Fujita and Maeda, 2018). Depending upon the way in which the  $Q$ -function is modeled, clipping has also been shown to introduce bias (Chou et al., 2017). Previous approaches are not applicable to the case when  $T$  is the projection onto the unit sphere; in the case of clipped actions, unlike previous work, we do not require that each component of the action is independent and obtain much stronger variance reduction results. Concurrent work (Fellows et al., 2018) also considers angular actions, but their method cannot be used as a drop in replacement in state of the art methods and the a special form of the critic  $q_{\pi}$  is required.

Integrated Policy Gradients. Several recent works have considered, as we do, exploiting an integrated form of policy gradient (Ciosek and Whiteson, 2018; Asadi et al., 2017; Fujita and Maeda, 2018; Tamar et al., 2012). Ciosek and Whiteson (2018) introduces a unified theory of policy gradients, which subsumes both deterministic (Silver et al., 2014) and stochastic policy gradients (Sutton et al., 2000). They characterize the distinction between different policy gradient methods as a choice of quadrature for the expectation. Their Expected Policy Gradient algorithm uses a new way of estimating the expectation for stochastic policies. They prove that the estimator has lower variance than stochastic policy gradients. Asadi et al. (2017) propose a similar method, but lack theoretical guarantees. Fujita and Maeda (2018) introduce the clipped action policy gradient (CAPG) which is a partially integrated form of policy gradient and provide a variance reduction guarantee, but their result is not tight. By viewing CAPG as a marginal policy gradient we obtain tighter results.

Variance Decomposition. The law of total variance, or variance decomposition, is given by  $\operatorname{Var}[Y] = \mathbb{E}[\operatorname{Var}(Y|X)] + \operatorname{Var}[\mathbb{E}[Y|X]]$ , where  $X$  and  $Y$  are two random variables on the same probability space. Our main result can be viewed as a special form of law of total variance, but it is highly non-trivial to obtain the result directly from the law of total variance. Also related to our approach is Rao-Blackwellization (Blackwell, 1947) of a statistic to obtain a lower variance estimator, though in our case it is the fact that  $T(a)$  is not sufficient for the sampling distribution that enables the variance reduction.

# 2 PRELIMINARIES

Notation and Setup. For MDP's we use the standard notation.  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $p$  denotes the transition probability kernel,  $p_0$  the initial state distribution,  $r$  the reward function. A policy  $\pi (a|s)$  is a distribution over actions given a state  $s\in S$ . A sample trajectory under  $\pi$  is denoted  $\tau_{\pi}\coloneqq (s_0,a_0,r_1,s_1,a_1,\ldots)$  where  $s_0\sim p_0$  and  $a_{t}\sim \pi (\cdot |s_{t})$ . The state-value function is defined as  $v_{\pi}(s)\coloneqq \mathbb{E}_{\pi}[\sum_{t = 0}^{\infty}\gamma^{t}r_{t + 1}|s_{0} = s]$  and the action-value function as  $q_{\pi}(s,a)\coloneqq \mathbb{E}_{\pi}[\sum_{t = 0}^{\infty}\gamma^{t}r_{t + 1}|s_{0} = s,a_{0} = a]$ . The objective is to maximize expected cumulative discounted reward,  $\eta (\pi) = \mathbb{E}_{p_0}[v_\pi (s_0)]$ .  $\rho_{\pi}$  denotes the improper discounted state occupancy distribution, defined as  $\rho_{\pi}\coloneqq \sum_{t}\gamma^{t}\mathbb{E}_{p_{0}}[\mathbb{P}(s_{t} = s|s_{0},\pi)]$ . We make the standard assumption of bounded rewards.

We consider the problem of learning a policy  $\pi$  parametrized by  $\theta \in \Theta$ . All gradients are with respect to  $\theta$  unless otherwise stated. By convention, we define  $0 \cdot \infty = 0$  and  $\frac{0}{0} = 0$ . A measurable space  $(\mathcal{A}, \mathcal{E})$  is a set  $\mathcal{A}$  with a sigma-algebra  $\mathcal{E}$  of subsets of  $\mathcal{A}$ . When we refer to a probability distribution of a random variable taking values in  $(\mathcal{A}, \mathcal{E})$  we will work directly with the probability measure on  $(\mathcal{A}, \mathcal{E})$  rather than the underlying sample space. For a measurable mapping  $T$  from measure space  $(\mathcal{A}, \mathcal{E}, \lambda)$  to measurable space  $(\mathcal{B}, \mathcal{F})$ , we denote by  $T_*\lambda$  the push-forward of  $\lambda$ .  $S^{d-1}$  denotes the unit sphere in  $\mathbb{R}^d$  and for any space  $\mathcal{A}$ ,  $B(\mathcal{A})$  denotes the Borel  $\sigma$ -algebra on  $\mathcal{A}$ . The notation  $\mu \ll \nu$  signifies the measure  $\mu$  is absolutely continuous with respect to  $\nu$ . The function  $\operatorname{clip}$  is defined as  $\operatorname{clip}(a, \alpha, \beta) = \min(\beta, \max(\alpha, a))$  for  $a \in \mathbb{R}$ . If  $a \in \mathbb{R}^d$ , it is interpreted element-wise.

Variance of Random Vectors. We define the variance of a random vector  $\mathbf{y}$  as  $\operatorname{Var}(\mathbf{y}) = \mathbb{E}[(\mathbf{y} - \mathbb{E}\mathbf{y})^\top (\mathbf{y} - \mathbb{E}\mathbf{y})]$ , i.e. the trace of the covariance of  $\mathbf{y}$ ; it is easy to verify standard properties of the variance still hold. This definition is often used to analyze the variance of gradient estimates (Greensmith et al., 2004).

Stochastic Policy Gradients. In Section 4 we present marginal policy gradient estimators and work in the very general setting described below. Let  $(\mathcal{A},\mathcal{E},\mu)$  be a measure space, where as before  $\mathcal{A}$  is the action space of the MDP. In practice, we often encounter  $(\mathcal{A},\mathcal{E}) = (\mathbb{R}^d,B(\mathbb{R}^d))$  with  $\mu$  as the Lebesgue measure. The types of policies for which there is a meaningful notation of stochastic policy gradients are  $\mu$ -compatible measures (see remarks 2.3 and 2.4).

Definition 2.1 ( $\mu$ -Compatible Measures). Let  $(\mathcal{A}, \mathcal{E}, \mu)$  be a measure space and consider a parametrized family of measures  $\Pi = \{\pi(\cdot, \theta) : \theta \in \Theta\}$  on the same space.  $\Pi$  is a  $\mu$ -compatible family of measures if for all  $\theta$ :

(a)  $\pi (\cdot ,\theta)\ll \mu$  with density of the form  $f_{\pi}(\cdot ,\theta)$  
(b)  $f_{\pi}$  is differentiable in  $\theta$  , and  
(c)  $\pi$  satisfies the conditions to apply the Leibniz integral rule for each  $\theta$ , so that  $\nabla \int_{\mathcal{A}} f_{\pi}(a) d\mu = \int_{\mathcal{A}} \nabla f_{\pi}(a) d\mu$ .

For  $\mu$ -compatible policies, Theorem 2.2 gives the stochastic policy gradient, easily estimable from samples. When  $\mu$  is the counting measure we recover the discrete policy gradient theorem (Sutton et al., 2000). See Appendix A.1 for a more in depth discussion and a proof of Theorem 2.2, which we include for completeness.

Theorem 2.2 (Stochastic Policy Gradient). Let  $(\mathcal{A},\mathcal{E},\mu)$  be a measure space and let  $\Pi = \{\pi (\cdot ,\theta |s):\theta \in \Theta \}$  be a family of  $\mu$  -compatible probability measures. Denoting by  $f_{\pi}$  the density with respect to  $\mu$  , we have that

$$
\nabla \eta = \int_ {\mathcal {S}} d \rho_ {\pi} (s) \int_ {\mathcal {A}} q _ {\pi} (s, a) \nabla \log f _ {\pi} (a | s) d \pi (\cdot | s).
$$

In general we want an estimate  $g$  of  $\nabla \eta$  such that it is unbiased  $(\mathbb{E}[g] = \nabla \eta)$  and that has minimal variance, so that convergence to a (locally) optimal policy is as fast as possible. In the following sections, we explore a general approach to finding a low variance, unbiased estimator.

Remark 2.3. Under certain choices of  $T$  (e.g. clipping) the effective action distribution is a mixture of a continuous distribution and point masses. Thus, although it adds some technical overhead, it is necessary that we take a measure theoretic approach in this work.

Remark 2.4. Definition 2.1 is required to ensure the policy gradient is well defined, as it stipulates the existence of an appropriate reference measure; it also serves to clarify notation and to draw a distinction between  $\pi$  and its density  $f_{\pi}$ . Though these details are often minimized they are important in analyzing the interaction between  $T$  and  $\pi$ .

# 3 ANGULAR POLICY GRADIENTS

Consider the task of learning a policy over directions in  $\mathcal{A} = \mathbb{R}^2$ , or equivalently learning a policy over angles  $[0, 2\pi)$ . A naive approach is to fit the mean  $\mu_{\theta}(s)$ , model the angle as normally distributed about  $\mu_{\theta}$ , and then clip the sampled angle before execution in the environment. However, this approach is asymmetric in that does not place similar probability on  $\mu_{\theta}(s) - \epsilon$  and  $\mu_{\theta}(s) + \epsilon$  for  $\mu_{\theta}(s)$  near to 0 and  $2\pi$ .

An alternative is to model  $\mu_{\theta}(s) \in \mathbb{R}^2$ , sample  $a \sim \mathcal{N}(\mu_{\theta}(s), \Sigma)$ , and then execute  $T(a) \coloneqq a / ||a||$  in the environment. This method also works for directional control in  $\mathbb{R}^d$ . The drawback of this approach is the following: informally speaking, we are sampling from a distribution with  $d$  degrees of freedom, but the environment is affected by an action with only  $d - 1$  degrees of freedom. This suggests, and indeed we later prove, that the variance of the stochastic policy gradient for this distribution is unnecessarily high. In this section we introduce the angular policy gradient which can be used as a drop-in replacement for the policy update step in existing algorithms.

![](images/ec37fde5796b08ce4a2cd8f4069ac6b28565f5bb026737e2988fb18eab031947.jpg)  
Figure 1: Transformation of a Gaussian policy - (left to right)  $\pi (\cdot |s)$ ,  $T = a / ||a||$ , and  $T_{*}\pi (\cdot |s)$ .

![](images/dd7bae4bf6ea6150e735f6f4967e02a1267a698854ab2cce75eca068a22b52c0.jpg)

![](images/1e328bedeaff76a8e021ef3a9ba26ad9c6dc9c25d28dc8a86013a1b7ffa9d672.jpg)

# ANGULAR GAUSSIAN DISTRIBUTION

Instead, we can directly model  $T(a) \in S^{d-1}$  instead of  $a \in \mathbb{R}^d$ . If  $a \sim \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$ , then  $T(a)$  is distributed according to what is known as the angular Gaussian distribution (Definition 3.1). It can be derived by a change of variables to spherical coordinates, followed by integration with respect to the magnitude of the random vector (Paine et al., 2018). Figure 1 illustrates the transformation of a Gaussian sampling policy  $\pi$  under  $T$ .

Definition 3.1 (Angular Gaussian Distribution). Let  $a \sim \mathcal{N}(\mu, \Sigma)$ . Then, with respect to the spherical measure  $\sigma$  on  $(S^{d-1}, B(S^{d-1}))$ ,  $x = a / ||a||$  has density

$$
f (x; \mu , \Sigma) = \left((2 \pi) ^ {d - 1} | \Sigma | \left(x ^ {\top} \Sigma^ {- 1} x\right) ^ {d}\right) ^ {- 1 / 2} \exp \left(\frac {1}{2} \left(\alpha^ {2} - \mu^ {\top} \Sigma^ {- 1} \mu\right)\right) \mathcal {M} _ {d - 1} (\alpha), \tag {3.1}
$$

where  $\alpha = \frac{x^{\top}\Sigma^{-1}\mu}{(x^{\top}\Sigma^{-1}x)^{1 / 2}}$  and  $\mathcal{M}_{d - 1}(x) = (2\pi)^{-\frac{1}{2}}\int_0^\infty u^{d - 1}\exp (-(u - x)^2 /2)du.$

# POLICY GRADIENT METHOD

Although the density in Definition 3.1 does not have a closed form, we can still obtain a stochastic policy gradient for this type of policy. Define the action space as  $\mathcal{A} \coloneqq S^{d-1}$  and consider angular Gaussian policies parametrized by  $\theta \coloneqq (\theta_{\mu}, \theta_{\Sigma})$ , where  $\theta_{\mu}$  parametrizes  $\mu$  and  $\theta_{\Sigma}$  parametrizes  $\Sigma$ . As before, denote the corresponding parametrized family of measures as  $\Pi \coloneqq \{\pi(\cdot, \theta|s) : \theta \in \Theta\}$ . Directly from Definition 3.1, we obtain

$$
\log f _ {\pi} = \frac {1}{2} \left(\alpha^ {2} - \mu^ {\top} \Sigma^ {- 1} \mu\right) + \log \mathcal {M} _ {d - 1} (\alpha) - \frac {1}{2} \left[ (d - 1) \log 2 \pi + \log | \Sigma | + d \log \left(x ^ {\top} \Sigma x\right) \right].
$$

Though this log-likelihood does not have a closed form, it turns out it is easy to compute the gradient in practice. It is only necessary that we can evaluate  $\mathcal{M}_{d - 1}^{\prime}(\alpha)$  and  $\mathcal{M}_d(\alpha)$  easily. Assuming for now that we can do so, denote by  $\theta_{i}$  the parameters after  $i$  gradient updates and define

$$
l_{i}(\theta):= \frac{1}{2}\left(\alpha^{2} - \mu^{\top}\Sigma^{-1}\mu\right) + \underbrace{\mathcal{M}_{d - 1}^{\prime}(\alpha(\theta_{i}))}_{\substack{\text{(i)}}}\alpha -\frac{1}{2}\left[(d - 1)\log 2\pi +\log |\Sigma | + d\log \left(x^{\top}\Sigma x\right)\right].
$$

By design,

$$
\nabla \log f _ {\pi} (\theta) | _ {\theta = \theta_ {i}} = \nabla l _ {i} (\theta) | _ {\theta = \theta_ {i}},
$$

thus at update  $i$  it suffices to compute the gradient of  $l_{i}$ , which can be done using standard auto-differentiation software (Paszke et al., 2017) since term (i) is a constant. From Paine et al. (2018), we have that  $\mathcal{M}_d'(\alpha) = d\mathcal{M}_{d-1}(\alpha)$ ,  $\mathcal{M}_{d+1}(\alpha) = \alpha \mathcal{M}_d(\alpha) + d\mathcal{M}_{d-1}(\alpha)$ ,  $\mathcal{M}_1(\alpha) = \alpha \Phi(\alpha) + \phi(\alpha)$  and  $\mathcal{M}_0(\alpha) = \Phi(\alpha)$ , where  $\Phi, \phi$  denote the PDF and CDF of  $\mathcal{N}(0,1)$ , respectively. Leveraging these properties, the integral  $\mathcal{M}_d(\alpha)$  can be computed recursively; Algorithm 1 in Appendix B.1 gives pseudo-code for the computation. Importantly it runs in  $\mathcal{O}(d)$  time and therefore does not effect the computational cost of the policy update since it is dominated by the cost of computing  $\nabla l_i$ . In addition, stochastic gradients of policy loss functions for TRPO or PPO Schulman et al. (2015; 2017) can be computed in a similar way since we can easily get the derivative of  $f_\pi(\theta)$  when  $\mathcal{M}_{d-1}(\alpha)$  and  $\mathcal{M}_{d-1}'(\alpha)$  are known.

# 4 MARGINAL POLICY GRADIENT ESTIMATORS

In Section 2, we described a general setting in which a stochastic policy gradient theorem holds on a measure space  $(\mathcal{A},\mathcal{E},\lambda)$  for a family of  $\lambda$ -compatible probability measures,  $\Pi = \{\pi (\cdot ,\theta |s):\theta \in \Theta \}$ . As before, we are interested in the case when the dynamics of the environment only depend on  $a\in \mathcal{A}$  via a function  $T$ . That is to say  $r(s,a)\coloneqq r(s,T(a))$  and  $p(s,a,s^{\prime})\coloneqq p(s,T(a),s^{\prime})$ .

The key idea in Marginal Policy Gradient is to replace the policy gradient estimate based on the log-likelihood of  $\pi$  with a lower variance estimate, which is based on the log-likelihood of  $T_{*}\pi$ .  $T_{*}\pi$  can be thought of as (and in some cases is) a marginal distribution, hence the name Marginal Policy Gradient. For this reason it can easily be used with value function approximation and GAE, as well as incorporated into algorithms like TRPO, A3C and PPO.

# 4.1 SETUP AND REGULARITY CONDITIONS

For our main results we need regularity Condition 4.1 on the measure space  $(\mathcal{A},\mathcal{E},\lambda)$ . Next, let  $(\mathcal{B},\mathcal{F})$  be another measurable space and  $T:\mathcal{A}\to \mathcal{B}$  be a measurable mapping.  $T$  induces a family of probability measures on  $(\mathcal{B},\mathcal{F})$ , denoted  $T_{*}\Pi \coloneqq \{T_{*}\pi (\cdot ,\theta |s):\theta \in \Theta \}$ . We also require regularity Conditions 4.2 and 4.3 regarding the structure of  $\mathcal{F}$  and the existence of a suitable reference measure  $\mu$  on  $(\mathcal{B},\mathcal{F})$ . These conditions are all quite mild and are satisfied in all practical settings, to the best of our knowledge.

Condition 4.1.  $\mathcal{A}$  is a metric space and  $\lambda$  is a Radon measure.

Condition 4.2.  $\mathcal{F}$  is countably generated and contains the singleton sets  $\{b\}$ , for all  $b \in \mathcal{B}$ .

Condition 4.3. There exists a  $\sigma$ -finite measure  $\mu$  on  $(\mathcal{B},\mathcal{F})$  such that  $T_{*}\lambda \ll \mu$  and  $T_{*}\Pi$  is  $\mu$ -compatible.

In statistics, Fisher information is used to capture the variance of a score function. In reinforcement learning, typically one encounters a score function that has been rescaled by a measurable function  $q(a)$ . Definition 4.4 provides a notion of Fisher information for  $\lambda$ -compatible distributions and rescaled score functions; we defer a discussion of the definition until Section 4.4 after we present our results in their entirety. If  $q(a) = 1$ , Definition 4.4 is the trace of the classical Fisher Information.

Definition 4.4 (Total Scaled Fisher Information). Let  $(\mathcal{A},\mathcal{E},\lambda)$  be a measure space,  $\Pi = \{\pi (\cdot ,\theta):$ $\theta \in \Theta \}$  be a family of  $\lambda$  -compatible probability measures, and  $q$  a measurable function on  $\mathcal{E}$  . The total scaled fisher information is defined as  $\mathcal{I}_{\pi ,\lambda}(q,\theta)\coloneqq \mathbb{E}[q(a)^2\nabla \log f_{\pi}(a)^{\top}\nabla \log f_{\pi}(a)]$

# 4.2 VARIANCE REDUCTION GUARANTEE

From Theorem 2.2 it is immediate that

$$
\begin{array}{l} \nabla \eta (\boldsymbol {\theta}) = \int_ {\mathcal {S}} d \rho (s) \int_ {\mathcal {A}} q (T (a), s) \nabla \log f _ {\pi} (a | s) d \pi (a | s) \\ = \int_ {\mathcal {S}} d \rho (s) \int_ {\mathcal {B}} q (b, s) \nabla \log f _ {T _ {*} \pi} (b | s) d (T _ {*} \pi) (b | s), \\ \end{array}
$$

where we dropped the subscripts on  $\rho$  and  $q$  because the two polices affect the environment in the same way, and thus have the same value function and discounted state occupancy measure. Denote the two alternative gradient estimators as  $g_{1} = q(T(a),s)\nabla \log f_{\pi}(a|s)$  and  $g_{2} = q(b,s)\nabla \log f_{T_s\pi}(b|s)$ . Just by definition, we have that  $\mathbb{E}_{\rho ,\pi}[g_1] = \mathbb{E}_{\rho ,\pi}[g_2]$ . Lemma 4.5 says something slightly different - it says that they are also equivalent in expectation conditional on the state  $s$ , a fact we use later.

Lemma 4.5. Let  $(\mathcal{A},\mathcal{E},\lambda)$  and  $(\mathcal{B},\mathcal{F},\mu)$  be measure spaces, and  $T:\mathcal{A}\to \mathcal{B}$  be measurable mapping. If  $\Pi$ , parametrized by  $\theta$ , is  $\lambda$ -compatible and  $T_{*}\Pi$  is  $\mu$ -compatible, then

$$
\mathbb {E} _ {\pi | s} [ g _ {1} ] = \mathbb {E} _ {\pi | s} [ g _ {2} ] = \mathbb {E} _ {T _ {*} \pi | s} [ g _ {2} ]. \tag {4.1}
$$

Proof. The result follows immediately from the proof of Theorem 2.2 in Appendix A.1.  $\square$

Because the two estimates  $g_{1}$  and  $g_{2}$  are both unbiased, it is always preferable to use whichever has lower variance. Theorem 4.6 shows that  $g_{2}$  is the lower variance policy gradient estimate. See Appendix B.3 for the proof. The implication of Theorem 4.6 is that if there is some information loss via a function  $T$  before the action interacts with the dynamics of the environment, then one obtains a lower variance estimator of the gradient by replacing the density of  $\pi$  with the density of  $T_{*}\pi$  in the expression for the policy gradient.

Theorem 4.6. Let  $g_{1}$  and  $g_{2}$  be as defined above. Then if Conditions 4.1-4.3 are satisfied,

$$
\operatorname {V a r} _ {\rho , \pi} (g _ {1}) - \operatorname {V a r} _ {\rho , T _ {*} \pi} (g _ {2}) = \mathbb {E} _ {\rho , T _ {*} \pi} \left[ \mathcal {I} _ {\pi | s | b, \lambda_ {b}} (q \circ T, \theta) \right] \geq 0,
$$

for some family of measures  $\{\lambda_b\}$  on  $\mathcal{A}$ . Furthermore, if  $T$  is not a sufficient statistic for  $\theta$ ,

$$
\mathbb {E} _ {\rho , T _ {*} \pi} \left[ \mathcal {I} _ {\pi | s | b, \lambda_ {b}} (q \circ T, \theta) \right] > 0.
$$

# 4.3 EXAMPLES OF MARGINAL POLICY GRADIENT ESTIMATORS

# CLIPPED ACTION POLICY GRADIENT

Consider a control problem where actions in  $\mathbb{R}$  are clipped to an interval  $[\alpha, \beta]$ . Let  $\lambda$  be an arbitrary measure on  $(\mathcal{A}, \mathcal{E}) \coloneqq (\mathbb{R}, B(\mathbb{R}))$ , and consider any  $\lambda$ -compatible family  $\Pi$ . Following Fujita and Maeda (2018), define the clipped score function

$$
\widetilde {\psi} (s, b, \theta) = \left\{ \begin{array}{l l} \nabla \log \int_ {(- \infty , \alpha ]} f _ {\pi} (a, \theta | s) d \lambda & b = \alpha \\ \nabla \log f _ {\pi} (b, \theta | s) & b \in (\alpha , \beta) \\ \nabla \log \int_ {[ \beta , \infty)} f _ {\pi} (a, \theta | s) d \lambda & b = \beta . \end{array} \right.
$$

We can apply Theorem 4.6 in this setting to obtain Corollary 4.7. It is a strict generalization of the results in Fujita and Maeda (2018) in that it applies to a larger class of measures and provides a much stronger variance reduction guarantee. It is possible to obtain this more powerful result precisely because we require minimal assumptions for Theorem 4.6. Note that the result can be extended to  $\mathbb{R}^d$ , but we stick to  $\mathbb{R}$  for clarity of presentation. See Appendix B.4 for a discussion of which distributions are  $\lambda$ -compatible and a proof of Corollary 4.7.

Corollary 4.7. Let  $\lambda$  be an arbitrary measure on  $(\mathcal{A},\mathcal{E})\coloneqq (\mathbb{R},B(\mathbb{R}))$ ,  $T(a)\coloneqq \mathrm{clip}(a,\alpha ,\beta)$ , and  $\psi (s,a,\theta)\coloneqq \nabla \log f_{\pi}(a,\theta |s)$ . If  $\Pi$  is a  $\lambda$ -compatible family parametrized by  $\theta$  and the dynamics of the environment depend only on  $T(a)$ , then

1.  $\mathbb{E}_{\pi |s}\left[q_{\pi}(s,a)\psi (s,a,\theta)\right] = \mathbb{E}_{\pi |s}\left[q_{\pi}(s,a)\widetilde{\psi} (s,T(a),\theta)\right]$  , and  
2.  $\operatorname{Var}_{\rho, \pi}(q_{\pi}(s, a)\psi(s, a, \theta)) - \operatorname{Var}_{\rho, \pi}(q_{\pi}(s, a)\widetilde{\psi}(s, T(a), \theta)) = \mathbb{E}_{\rho}\left[\mathbb{E}_{T_*\pi|s}\left[\mathcal{I}_{\pi|s|b, \lambda_b}(q \circ T, \theta)\right]\right]$ , for some family of measures  $\{\lambda_b\}$  on  $\mathcal{A}$ .

# ANGULAR POLICY GRADIENT

Now consider the case where we sample an action  $a \in \mathbb{R}^d$  and apply  $T(a) = a / ||a||$  to map into  $\mathcal{S}^{d-1}$ . Let  $(\mathcal{A}, \mathcal{E}) = (\mathbb{R}^d, B(\mathbb{R}^d))$  and let  $\lambda$  be the Lebesgue measure. When  $\Pi$  is a multivariate Gaussian family parametrized by  $\theta$ ,  $T_*\Pi$  is an angular Gaussian family also parametrized by  $\theta$  (Section 3). If  $\Pi$  is  $\lambda$ -compatible - here it reduces to ensuring the parametrization is such that  $f_\pi$  is differentiable in  $\theta$  - then  $T_*\Pi$  is  $\sigma$ -compatible, where  $\sigma$  denotes the spherical measure. Denoting by  $f_{MV}(a,\theta|s)$  and  $f_{AG}(b,\theta|s)$  the corresponding multivariate and angular Gaussian densities, respectively, we state the results for this setting as Corollary 4.8. See Appendix B.4 for a proof.

Corollary 4.8. Let  $\lambda$  be the Lebesgue measure on  $(\mathcal{A},\mathcal{E}) = (\mathbb{R}^d,B(\mathbb{R}^d))$ ,  $T(a)\coloneqq a / ||a||$  and  $\Pi$  be a multivariate Gaussian family on  $\mathcal{A}$  parametrized by  $\theta$ . If the dynamics of the environment only depend on  $T(a)$  and  $f_{MV}(\cdot ,\theta |s)$ , the density corresponding to  $\Pi$ , is differentiable in  $\theta$ , then

1.  $\mathbb{E}_{\pi |s}\left[q_{\pi}(s,a)\psi (s,a,\theta)\right] = \mathbb{E}_{\pi |s}\left[q_{\pi}(s,a)\widetilde{\psi} (s,T(a),\theta)\right]$  , and  
2.  $\operatorname{Var}_{\rho, \pi}(q_{\pi}(s, a) \psi(s, a, \theta)) - \operatorname{Var}_{\rho, \pi}(q_{\pi}(s, a) \widetilde{\psi}(s, T(a), \theta)) = \mathbb{E}_{\rho, T_* \pi}\left[\operatorname{Var}_{\pi|b}(q_{\pi}(s, a) \psi_r(s, r, \theta))\right]$ , where  $r = ||a||$ ,  $f_r$  is the conditional density of  $r$ ,  $\psi(s, a, \theta) := \nabla \log f_{MV}(a, \theta | s)$ ,  $\widetilde{\psi}(s, b, \theta) = \nabla \log f_{AG}(b, \theta | s)$ , and  $\psi_r(s, r, \theta) = \nabla \log f_r(r, \theta | s)$ .

# PARAMETRIZED ACTION SPACES

As one might expect, our variance reduction result applies to parametrized action spaces when a lossy transformation  $T_{i}$  is applied to the parameter for discrete action  $i$ . See Appendix B.5 for an in-depth discussion of policy gradient methods for parametrized action spaces.

# 4.4 DISCUSSION

Denoting by  $g_{1}$  the standard policy gradient estimator for a  $\lambda$ -compatible family  $\Pi$ , observe that  $\mathrm{Var}_{\rho, \pi}(g_1) = \mathcal{I}_{\pi, \lambda}(q, \theta)$ . We introduce the quantity  $\mathcal{I}_{\pi, \lambda}$  because unless  $T$  is a coordinate projection it is not straightforward to write Theorem 4.6 in terms of the density of a conditional distribution. Corollary 4.8 can be written this way because under a re-parametrization to polar coordinates,  $T(a) = a / ||a||$  can be written as a coordinate projection. In general, by using  $\mathcal{I}_{\pi, \lambda}$  we can phrase the result in terms of a quantity with an intuitive interpretation: a (q-weighted) measure of information contained in  $a$  that does not influence the environment.

Recalling the law of total variation (LOTV), we can observe that Theorem 4.6 is indeed specific version of that general result. We can not directly apply the LOTV because in the general setting, it is highly non-trivial to conclude that  $g_{2}$  is a version of the conditional expectation of  $g_{1}$ , and for arbitrary policies, one must be extremely careful when making the conditioning argument (Chang and Pollard, 1997). However for certain special cases, like CAPG, we can check fairly easily that  $g_{2} = \mathbb{E}[g_{1}|b]$ .

# 5 APPLICATIONS AND DISCUSSION

# 5.1 2D NAVIGATION TASK

Because relatively few existing reinforcement learning environments support angular actions, we implement a navigation task to benchmark our methods. In this navigation task, the agent is located

on a platform and must navigate from one location to another without falling off. The state space is  $S = \mathbb{R}^2$ , the action space is  $\mathcal{A} = \mathbb{R}^2$  and the transformation  $T(a) = a / ||a||$  is applied to actions before execution in the environment. Let  $s_G = (1,1)$  be the goal (terminal) state. Using the reward shaping approach (Ng et al., 1999), we define a potential function  $\phi(s) = ||s - s_G||_2$  and a reward function as  $r(s_t, a_t) = \phi(s_t) - \phi(s_t + a_t)$ . The start state is fixed at  $s_0 = (-1, -1)$ . One corner of the platform is located at  $(-1.5, -1.5)$  and the other at  $(1.5, 1.5)$ .

We compare angular Gaussian policies with (1) bivariate Gaussian policies and (2) a 1-dimensional Gaussian policy where we model the mean of the angle directly, treating angles that differ by  $2\pi$  as identical. For all candidate policies, we use A2C (the synchronous version of A3C (Mnih et al., 2016)) to learn the conditional mean  $\mu(s; \theta)$  of the sampling distribution by fitting a feed-forward neural network with tanh activations. The variance of the sampling distribution,  $\sigma^2 \mathbf{I}$ , is fixed. For the critic we estimate the state value function  $v_{\pi}(s)$ , again using a feed-forward neural network. Appendix C.1 for details on the hyper-parameter settings, network architecture and training procedure.

# 5.2 APPLICATION - KING OF GLORY

We implement a marginal policy gradient method for King of Glory (the North American release is titled Arena of Valor) by Tencent Games. King of Glory has several game types and we focus on the 1v1 version. Our work here is one of the first attempts to solve King of Glory, and MOBA games in general, using reinforcement learning. Similar MOBA games include Dota 2 and League of Legends.

Game Description. In King of Glory, players are divided into two "camps" located in opposite corners of the game map. Each player chooses a "hero", a character with unique abilities, and the objective is to destroy the opposing team's "crystal", located at their game camp. The path to each camp and crystal is guarded by towers which attack enemies when in range. Each team has a number of allied "minions", less powerful characters, to help them destroy the enemy crystal. Only the "hero" is controlled by the player. During game play, heroes increase in level and obtain gold by killing enemies. This allows the player to upgrade the level of their hero's unique skills and buy improved equipment, resulting in more powerful attacks, increased HP, and other benefits. Figure 2 shows King of Glory game play; in the game pictured, both players use the hero "Di Ren Jie".

Formulation as an MDP.  $\mathcal{A}$  is a parametrized action space with 7 discrete actions, 4 of which are parametrized by  $\omega \in \mathbb{R}^2$ . These actions include move, attack, and use skills; a detailed description of all actions and parameters is given in Table 3, Appendix C.2. In our setup, we use rules crafted by domain experts to manage purchasing equipment and learning skills. The transformation  $T(a) = a / ||a||$  is applied to the action parameter before execution in the environment, so the effective action parameter spaces are  $S^1$ .

Using information obtained directly from the game engine, we construct a 2701-dimensional state representation. Features extracted from the game engine include hero locations, hero health, tower health, skill availability and relative locations to towers and crystals - see Appendix C.2 for details on the feature extraction process. As in Section 5.1, we define rewards using a potential function. In particular we define a reward feature mapping  $\rho$  and a weighting vector  $w$ , and then a linear potential function as  $\phi_r(s) = w^T\rho(s)$ . Information extracted by  $\rho$  includes hero health, crystal health, and game outcome; see Table 5, Appendix C.2 for a complete description of  $w$  and  $\rho$ . Using  $\phi_r$ , we can define the reward as  $r_t = \phi_r(s_t) - \phi_r(s_{t-1})$ .

Implementation. We implement the A3C algorithm, and model both the policy  $\pi$  and the value function  $v_{\pi}$  using feed-forward neural networks. See Appendix C.2 for more details on how we model and learn the value function and policy. Using the setup described above, we compare:

1. a standard policy gradient approach for parametrized action spaces, and  
2. a marginal (angular) policy gradient approach, adapted to the parametrized action space where  $T_{i}(a) = a / ||a||$  is applied to parameter  $i$ .

Additional details on both approaches can be found in Appendix B.5.

# 5.3 RESULTS

For the navigation task, the top row of Figure 2 contains, from left to right, cumulative, discounted reward trajectories, and two plots showing the variances of the competing estimators. We see that the

![](images/dd46c89774805ac7801618eacb292cd9c7fff63d117745956e58a9e0c034e362.jpg)

![](images/43e06167678d58ac04298a024887eb654e0d2f370fe7187b04f3607fee6353ff.jpg)

![](images/dd526989b51920e08441926e3b136525a6f53450f550c0a30b78ec98d10f0bdb.jpg)

![](images/be64c7b9df523ed5282743d63e328203bf35342f725e6b0cef9bc804792b3af1.jpg)  
Figure 2: On top are results for Platform2D-v1; on bottom, results for King of Glory 1v1 and a screenshot of game play.

![](images/df70b682fd19ac99ebca9982fe43d1467f45cf7b62a0a781207724b7e1db0461.jpg)

![](images/709731e8fafbe30ee43fcf53cbd75bd10bd549e4733465b56b671dbe7b8560de.jpg)

agent using the angular policy gradient converges faster compared to the multivariate Gaussian due to the variance reduced gradient estimates. The second baseline also performs worse than APG, likely due in part to the fact that the critic must approximate a periodic function. Only APG achieves the maximum possible cumulative, discounted reward. On the King of Glory 1 vs. 1 task, the agent is trained to play as the hero Di Ren Jie and training occurs by competing with the game's internal AI, also playing as Di Ren Jie. The bottom row of Figure 2 shows the results, and as before, the angular policy gradient outperforms the standard policy gradient by a significant margin both in terms of win percentage and cumulative discounted reward.

In addition, Figure 2 highlights the effects of Theorem 4.6 in practice. The plot in the center shows the variance at the start of training, for a fixed random initialization, and the plot on the right shows the variance for a trained model that converged to the optimal policy. The main difference between the two settings is that the value function estimate  $\widehat{v}_{\pi}$  is highly accurate for the trained model (since both actor and critic have converged) and highly inaccurate for the untrained model. In both cases, we see that the variance of the marginal policy gradient estimator is roughly  $\frac{1}{2}$  that of the estimator using the sampling distribution.

# 5.4 DISCUSSION

Motivated by challenges found in complex control problems, we introduced a general family of variance reduced policy gradients estimators. This view provides the first unified approach to problems where the environment only depends on the action through some transformation  $T$ , and we demonstrate that CAPG and APG are members of this family corresponding to different choices of  $T$ . We also show that it can be applied to parametrized action spaces. Because thorough experimental work has already been done for the CAPG member of the family (Fujita and Maeda, 2018), confirming the benefits of MPG estimators, we do not reproduce those results here. Instead we focus on the case when  $T(a) = a / ||a||$  and demonstrate the effectiveness of the angular policy gradient approach on King of Glory and our own Platform2D-v1 environment. Although at this time few RL environments use directional actions, we anticipate the number will grow as RL is applied to newer and increasingly complex tasks like MOBA games where such action spaces are common. We also envision that our methods can be applied to autonomous vehicle, in particular quadcopter, control.

# REFERENCES

ASADI, K., ALLEN, C., RODERICK, M., MOHAMED, A.-R., KONIDARIS, G. and LITTMAN, M. (2017). Mean Actor Critic. arXiv:1709.00503.  
BLACKWELL, D. (1947). Conditional expectation and unbiased sequential estimation. Annals of Mathematical Statistics 18 105-110.  
CHANG, J. T. and POLLARD, D. (1997). Conditioning as disintegration. Statistica Neerlandica 51 287-317.  
CHOU, P.-W., MATURANA, D. and SCHERER, S. (2017). Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution. In ICML.  
CIOSEK, K. and WHITESON, S. (2018). Expected Policy Gradients for Reinforcement Learning. arXiv:1801.03326.  
FELLOWS, M., CIOSEK, K. and WHITESON, S. (2018). Fourier Policy Gradients. In ICML.  
FINN, C., ABBEEL, P. and LEVINE, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In ICML.  
FLORENSA, C., DUAN, Y. and ABBEEL, P. (2017). Stochastic Neural Networks for Hierarchical Reinforcement Learning. In ICLR.  
FOERSTER, J. N., ASSAEL, Y. M., DE FREITAS, N. and WHITESON, S. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. In NIPS.  
FUJITA, Y. and MAEDA, S.-I. (2018). Clipped Action Policy Gradient. In ICML.  
GREENSMITH, E., BARTLETT, P. L. and BAXTER, J. (2004). Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. Journal of Machine Learning Research 5 1471-1530.  
HAUSKNECHT, M. and STONE, P. (2016). Deep Reinforcement Learning In Parameterized Action Space. In ICLR.  
KINGMA, D. P. and BA, J. L. (2015). Adam: A Method for Stochastic Optimization. In ICLR.  
KLAMBAUER, G., UNTERTHINER, T., MAYR, A. and HOCHREITER, S. (2017). Self-Normalizing Neural Networks. In NIPS.  
MASSON, W., RANCHOD, P. and KONIDARIS, G. (2016). Reinforcement Learning with Parameterized Actions. In AAAI.  
MNIH, V., KAVUKCUOGLU, K., SILVER, D., RUSU, A. A., VENESS, J., BELLEMARE, M. G., GRAVES, A., RIEDMILLER, M., FIDJELAND, A. K., OSTROVSKI, G., PETERSEN, S., BEATTIE, C., SADIK, A., ANTONOGLOU, I., KING, H., KUMARAN, D., WIERSTRA, D., LEGG, S. and HASSABIS, D. (2015). Human-level control through deep reinforcement learning. Nature 529-533.  
MNIH, V., PUIGDOMENECH BADIA, A., MIRZA, M., GRAVES, A., HARLEY, T., LILLCRAP, T. P., SILVER, D., KAVUKCUOGLU, K., COM, K. and DEEPMIND, G. (2016). Asynchronous Methods for Deep Reinforcement Learning. In ICML.  
NG, A., HARADA, D. and RUSSELL, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. In ICML.  
PAINE, P. J., PRESTON, S. P., TSAGRIS, M. and WOOD, A. T. A. (2018). An elliptically symmetric angular Gaussian distribution. Statistics and Computing 28 689-697.  
PASZKE, A., CHANAN, G., LIN, Z., GROSS, S., YANG, E., ANTIGA, L. and DEVITO, Z. (2017). Automatic differentiation in PyTorch. In NIPS Workshop.  
SCHULMAN, J., LEVINE, S., MORITZ, P., JORDAN, M. and ABBEEL, P. (2015). Trust Region Policy Optimization. In ICML.

SCHULMAN, J., MORITZ, P., LEVINE, S., JORDAN, M. and ABBEEL, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. In ICLR.  
SCHULMAN, J., WOLSKI, F., DHARIWAL, P., RADFORD, A. and OPENAI, O. K. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.  
SILVER, D., HEESS, N., DEGRIS, T., WIERSTRA, D. and RIEDMILLER, M. (2014). Deterministic Policy Gradient Algorithms. In ICML.  
SUTTON, R. S. and BARTO, A. G. (2018). Reinforcement learning: an introduction.  
SUTTON, R. S., MCALLESTER, D., SINGH, S. and MANSOUR, Y. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. In NIPS.  
TAMAR, A., DI CASTRO, D. and MEIR, R. (2012). Integrating a partial model into model free reinforcement learning. Journal of Machine Learning Research 13 1927-1966.  
USUNIER, N., SYNNAEVE, G., LIN, Z. and CHINTALA, S. (2017). Episodic Exploration for Deep Deterministic Policies: An Application to StarCraft Micromanagement Tasks. In ICLR.  
VINYALS, O., Ewalds, T., BARTUNOV, S., GEORGIEV, P., VEZHNEVETS, A. S., YEO, M., MAKHZANI, A., UTTLER, H., AGAPIOU, J., SCHRITTWIESER, J., QUAN, J., GAFFNEY, S., PETERSEN, S., SIMONYAN, K., SCHAUL, T., VAN HASSELT, H., SILVER, D., LILICRAP, T., CALDERONE, D. K., KEET, P., BRUNASSO, A., LAWRENCE, D., EKERMO, A., REPP, J. and BLIZZARD, R. T. (2017). StarCraft II: A New Challenge for Reinforcement Learning. arXiv:.1708.04782.
