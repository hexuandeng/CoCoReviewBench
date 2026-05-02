# PROMP: PROXIMAL META-POLICY SEARCH

Anonymous authors

Paper under double-blind review

# ABSTRACT

Credit assignment in Meta-reinforcement learning (Meta-RL) is still poorly understood. Existing methods either neglect credit assignment to pre-adaptation behavior or implement it naively. This leads to poor sample-efficiency during meta-training as well as ineffective task identification strategies. This paper provides a theoretical analysis of credit assignment in gradient-based Meta-RL. Building on the gained insights we develop a novel meta-learning algorithm that overcomes both the issue of poor credit assignment and previous difficulties in estimating meta-policy gradients. By controlling the statistical distance of both pre-adaptation and adapted policies during meta-policy search, the proposed algorithm endows efficient and stable meta-learning. Our approach leads to superior pre-adaptation policy behavior and consistently outperforms previous Meta-RL algorithms in sample-efficiency, wall-clock time, and asymptotic performance.

# 1 INTRODUCTION

A remarkable trait of human intelligence is the ability to adapt to new situations in the face of limited experience. In contrast, our most successful artificial agents struggle in such scenarios. While achieving impressive results, they suffer from high sample complexity in learning even a single task, fail to generalize to new situations, and require large amounts of additional data to successfully adapt to new environments. Meta-learning addresses these shortcomings by learning how to learn. Its objective is to learn an algorithm that allows the artificial agent to succeed in an unseen task when only limited experience is available, aiming to achieve the same fast adaptation that humans possess (Schmidhuber, 1987; Thrun & Pratt, 1998).

Despite recent progress, deep reinforcement learning (RL) still relies heavily on hand-crafted features and reward functions as well as engineered problem specific inductive bias. Meta-RL aims to forego such reliance by acquiring inductive bias in a data-driven manner. Recent work proves this approach to be promising, demonstrating that Meta-RL allows agents to obtain a diverse set of skills, attain better exploration strategies, and learn faster through meta-learned dynamics models or synthetic returns (Duan et al., 2016; Xu et al., 2018; Gupta et al., 2018b; Saemundsson et al., 2018).

Meta-RL is a multi-stage process in which the agent, after a few sampled environment interactions, adapts its behavior to the given task. Despite its wide utilization, little work has been done to promote theoretical understanding of this process, leaving Meta-RL grounded on unstable foundations. Although the behavior prior to the adaptation step is instrumental for task identification, the interplay between pre-adaptation sampling and posterior performance of the policy remains poorly understood. In fact, prior work in gradient-based Meta-RL has either entirely neglected credit assignment to the pre-update distribution (Finn et al., 2017) or implemented such credit assignment in a naive way (Al-Shedivat et al., 2018; Stadie et al., 2018).

To our knowledge, we provide the first formal in-depth analysis of credit assignment w.r.t. pre-adaptation sampling distribution in Meta-RL. Based on our findings, we develop a novel Meta-RL algorithm. First, we analyze two distinct methods for assigning credit to pre-adaptation behavior. We show that the recent formulation introduced by Al-Shedivat et al. (2018) and Stadie et al. (2018) leads to poor credit assignment, while the MAML formulation (Finn et al., 2017) potentially yields superior meta-policy updates. Second, based on insights from our formal analysis, we highlight both the importance and difficulty of proper meta-policy gradient estimates. In light of this, we propose the low variance curvature (LVC) surrogate objective which yields gradient estimates with a favorable bias-variance trade-off. Finally, building upon the LVC estimator we develop Proximal Meta-Policy Search (ProMP), an efficient and stable meta-learning algorithm for RL. In our experiments,

we show that ProMP consistently outperforms previous Meta-RL algorithms in sample-efficiency, wall-clock time, and asymptotic performance.

# 2 RELATED WORK

Meta-Learning concerns the question of "learning to learn", aiming to acquire inductive bias in a data driven manner, so that the learning process in face of unseen data or new problem settings is accelerated (Schmidhuber, 1987; Schmidhuber et al., 1997; Thrun & Pratt, 1998).

This can be achieved in various ways. One category of methods attempts to learn the "learning program" of an universal Turing machine in form of a recurrent / memory-augmented model that ingests datasets and either outputs the parameters of the trained model (Hochreiter et al., 2001; Andrychowicz et al., 2016; Chen et al., 2017; Ravi & Larochelle, 2017) or directly outputs predictions for given test inputs (Duan et al., 2016; Santoro et al., 2016; Mishra et al., 2018). Though very flexible and capable of learning very efficient adaptations, such methods lack performance guarantees and are difficult to train on long sequences that arise in Meta-RL.

Another set of methods embeds the structure of a classical learning algorithm in the meta-learning procedure, and optimizes the parameters of the embedded learner during the meta-training (Husken & Goerick, 2000; Finn et al., 2017; Nichol et al., 2018; Miconi et al., 2018). A particular instance of the latter that has proven to be particularly successful in the context of RL is gradient-based meta-learning (Finn et al., 2017; Al-Shedivat et al., 2018; Stadie et al., 2018). Its objective is to learn an initialization such that after one or few steps of policy gradients the agent attains full performance on a new task. A desirable property of this approach is that even if fast adaptation fails, the agent just falls back on vanilla policy-gradients. However, as we show, previous gradient-based Meta-RL methods either neglect or perform poor credit assignment w.r.t. the pre-update sampling distribution.

A diverse set of methods building on meta-RL, has recently been introduced. This includes: learning exploration strategies (Gupta et al., 2018b), synthetic rewards (Sung et al., 2017; Xu et al., 2018), unsupervised policy acquisition (Gupta et al., 2018a), model-based RL (Claverna et al., 2018; Saemundsson et al., 2018), learning in competitive environments (Al-Shedivat et al., 2018) and meta-learning modular policies (Frans et al., 2018; Alet et al., 2018). Many of the mentioned approaches build on previous gradient-based meta-learning methods that insufficiently account for the pre-update distribution. ProMP overcomes these deficiencies, providing the necessary framework for novel applications of Meta-RL in unsolved problems.

# 3 BACKGROUND

Reinforcement Learning. A discrete-time finite Markov decision process (MDP),  $\mathcal{T}$ , is defined by the tuple  $(\mathcal{S},\mathcal{A},p,p_0,r,H)$ . Here,  $\mathcal{S}$  is the set of states,  $\mathcal{A}$  the action space,  $p(s_{t + 1}|s_t,a_t)$  the transition distribution,  $p_0$  represents the initial state distribution,  $r:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$  is a reward function, and  $H$  the time horizon. We omit the discount factor  $\gamma$  in the following elaborations for notational brevity. However, it is straightforward to include it by substituting the reward by  $r(s_{t},a_{t})\coloneqq \gamma^{t}r(s_{t},a_{t})$ . We define the return  $R(\tau)$  as the sum of rewards along a trajectory  $\tau \coloneqq (s_0,a_0,\dots,s_{H - 1},a_{H - 1},s_H)$ . The goal of reinforcement learning is to find a policy  $\pi (a|s)$  that maximizes the expected return  $\mathbb{E}_{\tau \sim P_{\tau}(\tau |\pi)}[R(\tau)]$ .

Meta-Reinforcement Learning goes one step further, aiming to learn a learning algorithm which is able to quickly learn the optimal policy for a task  $\mathcal{T}$  drawn from a distribution of tasks  $\rho(\mathcal{T})$ . Each task  $\mathcal{T}$  corresponds to a different MDP. Typically, it is assumed that the distribution of tasks share the action and state space, but may differ in their reward function or their dynamics.

Gradient-based meta-learning aims to solve this problem by learning the parameters  $\theta$  of a policy  $\pi_{\theta}$  such that performing a single or few steps of vanilla policy gradient (VPG) with the given task leads to the optimal policy for that task. This meta-learning formulation, also known under the name of MAML, was first introduced by Finn et al. (2017). We refer to it as formulation I which can be expressed as maximizing the objective

$$
J ^ {I} (\theta) = \mathbb {E} _ {\mathcal {T} \sim \rho (\mathcal {T})} \left[ \mathbb {E} _ {\boldsymbol {\tau} ^ {\prime} \sim P _ {\mathcal {T}} (\boldsymbol {\tau} ^ {\prime} | \theta^ {\prime})} \left[ R (\boldsymbol {\tau} ^ {\prime}) \right] \right] \quad \text {w i t h} \quad \theta^ {\prime} := U (\theta , \mathcal {T}) = \theta + \alpha \nabla_ {\theta} \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau} | \theta)} \left[ R (\boldsymbol {\tau}) \right]
$$

In that  $U$  denotes the update function which depends on the task  $\mathcal{T}$ , and performs one VPG step towards maximizing the performance of the policy in  $\mathcal{T}$ . For national brevity and conciseness we

assume a single policy gradient adaptation step. Nonetheless, all presented concepts can easily be extended to multiple adaptation steps.

Later work proposes a slightly different notion of gradient-based meta-RL, also known as E-MAML, that attempts to circumvent issues with the meta-gradient estimation in MAML (Al-Shedivat et al., 2018; Stadie et al., 2018):

$$
J^{II}(\theta) = \mathbb{E}_{\mathcal{T}\sim \rho (\mathcal{T})}\bigl[\mathbb{E}_{\substack{\boldsymbol{\tau}^{1:N}\sim P_{\mathcal{T}}(\boldsymbol{\tau}^{1:N}|\boldsymbol {\theta})\\ \boldsymbol{\tau}'\sim P_{\mathcal{T}}(\boldsymbol{\tau}'|\boldsymbol{\theta}')}}\bigl[R(\boldsymbol{\tau}^{\prime})\bigr ]\bigr ]\text{with}\theta^{\prime}:= U(\theta ,\boldsymbol{\tau}^{1:N}) = \theta +\alpha \nabla_{\theta}\sum_{n = 1}^{N}\Bigl[R(\boldsymbol{\tau}^{(n)})\Bigr]
$$

Formulation II views  $U$  as a deterministic function that depends on  $N$  sampled trajectories from a specific task. In contrast to formulation I, the expectation over pre-update trajectories  $\pmb{\tau}$  is applied outside of the update function. Throughout this paper we refer to  $\pi_{\theta}$  as pre-update policy, and  $\pi_{\theta'}$  as post-update policy.

# 4 SAMPLING DISTRIBUTION CREDIT ASSIGNMENT

This section analyzes the two gradient-based Meta-RL formulations introduced in Section 3. Figure 1 illustrates the stochastic computation graphs (Schulman et al., 2015b) of both formulations. The red arrows depict how credit assignment w.r.t the pre-update sampling distribution  $P_{\mathcal{T}}(\pmb{\tau}|\theta)$  is propagated. Formulation I (left) propagates the credit assignment through the update step, thereby exploiting the full problem structure. In contrast, formulation II (right) neglects the inherent structure, directly assigning credit from post-update return  $R'$  to the pre-update policy  $\pi_{\theta}$  which leads to noisier, less effective credit assignment.

![](images/a7d5e0726da95c66187386e533902ba1e7f77b5540ee93c3585864004e78b046.jpg)

![](images/4919bf3925ad0bed2fad748c0d585ffa0b452fb446cbb7dbae4ee3a18c278a5e.jpg)  
Figure 1: Stochastic computation graphs of meta-learning formulation I (left) and formulation II (right). The red arrows illustrate the credit assignment from the post-update returns  $R'$  to the pre-update policy  $\pi_{\theta}$  through  $\nabla_{\theta} J_{\mathrm{pre}}$ . (Deterministic nodes: Square; Stochastic nodes: Circle)

Both formulations optimize for the same objective, and are equivalent at the  $0^{th}$  order. However, because of the difference in their formulation and stochastic computation graph, their gradients and the resulting optimization step differs. In the following, we shed light on how and where formulation II loses signal by analyzing the gradients of both formulations, which can be written as (see Appendix A for more details and derivations)

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\mathcal {T} \sim \rho (\mathcal {T})} \left[ \underset {\boldsymbol {\tau} ^ {\prime} \sim P _ {\mathcal {T}} \left(\boldsymbol {\tau} ^ {\prime} \mid \theta^ {\prime}\right)} {\mathbb {E}} \left[ \nabla_ {\theta} J _ {\text {p o s t}} \left(\boldsymbol {\tau}, \boldsymbol {\tau} ^ {\prime}\right) + \nabla_ {\theta} J _ {\text {p r e}} \left(\boldsymbol {\tau}, \boldsymbol {\tau} ^ {\prime}\right) \right] \right] \tag {1}
$$

The first term  $\nabla_{\theta}J_{\mathrm{post}}(\tau ,\tau^{\prime})$  is equal in both formulations, but the second term,  $\nabla_{\theta}J_{\mathrm{pre}}(\tau ,\tau^{\prime})$  differs between them. In particular, they correspond to

$$
\nabla_ {\theta} J _ {\text {p o s t}} \left(\boldsymbol {\tau}, \boldsymbol {\tau} ^ {\prime}\right) = \underbrace {\left(I + \alpha R (\boldsymbol {\tau}) \nabla_ {\theta} ^ {2} \log \pi_ {\theta^ {\prime}} (\boldsymbol {\tau}))\right)} _ {\text {t r a n s f o r m a t i o n f r o m} \theta^ {\prime} \text {t o} \theta} \underbrace {\nabla_ {\theta^ {\prime}} \log \pi_ {\theta} \left(\boldsymbol {\tau} ^ {\prime}\right) R \left(\boldsymbol {\tau} ^ {\prime}\right)} _ {\nabla_ {\theta^ {\prime}} J _ {\text {o u t e r}}} \tag {2}
$$

$$
\nabla_ {\theta} J _ {\text {p r e}} ^ {I I} (\boldsymbol {\tau}, \boldsymbol {\tau} ^ {\prime}) = \alpha \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {\tau}) R \left(\boldsymbol {\tau} ^ {\prime}\right) \tag {3}
$$

$$
\nabla_ {\theta} J _ {\text {p r e}} ^ {I} (\boldsymbol {\tau}, \boldsymbol {\tau} ^ {\prime}) = \alpha \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {\tau}) \left(\underbrace {(\nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {\tau}) R (\boldsymbol {\tau})) ^ {\top}} _ {\nabla_ {\theta} J _ {\text {i n n e r}} ^ {\text {i n n e r}}} \underbrace {(\nabla_ {\theta^ {\prime}} \log \pi_ {\theta^ {\prime}} (\boldsymbol {\tau} ^ {\prime}) R (\boldsymbol {\tau} ^ {\prime}))}\right) \tag {4}
$$

$\nabla_{\theta}J_{\mathrm{post}}(\tau ,\tau^{\prime})$  simply corresponds to a policy gradient step on the post-update policy  $\pi_{\theta '}$  w.r.t  $\theta^\prime$  followed by a linear transformation from post- to pre-update parameters. It corresponds to increasing the likelihood of the trajectories  $\pmb{\tau}'$  that led to higher returns. However, this term does not optimize for the pre-update sampling distribution, i.e., which trajectories  $\pmb{\tau}$  led to better adaptation steps.

The credit assignment w.r.t. the pre-updated sampling distribution is carried out by the second term. In formulation II,  $\nabla_{\theta}J_{\mathrm{pre}}^{II}$  can be viewed as standard reinforcement learning on  $\pi_{\theta}$  with  $R(\pmb{\tau}')$  as

reward signal, treating the update function  $U$  as part of the unknown dynamics of the system. This shifts the pre-update sampling distribution to better adaptation steps.

Formulation I takes the causal dependence of  $P_{\mathcal{T}}(\pmb{\tau}'|\theta')$  on  $P_{\mathcal{T}}(\pmb{\tau}|\theta)$  into account. It does so by maximizing the inner product of pre-update and post-update policy gradients (see Eq. 4). This steers the pre-update policy towards 1) larger post-updates returns 2) larger adaptation steps  $\alpha \nabla_{\theta} J^{\text{inner}}$ , 3) better alignment of pre- and post-update policy gradients (Li et al., 2017; Nichol et al., 2018). When combined, these effects directly optimize for adaptation. As a result, we expect the first meta-policy gradient formulation,  $J^{I}$ , to yield superior learning properties.

# 5 LOW VARIANCE CURVATURE ESTIMATOR

In the previous section we show that the formulation introduced by Finn et al. (2017) results in superior meta-gradient updates, which should in principle lead to improved convergence properties. However, obtaining correct and low variance estimates of the respective meta-gradients proves challenging. As discussed by Foerster et al. (2018), and shown in Appendix B.3, the score function surrogate objective approach is ill suited for calculating higher order derivatives via automatic differentiation toolboxes. This important fact was overlooked in the original RL-MAML implementation (Finn et al., 2017) leading to incorrect meta-gradient estimates<sup>1</sup>. But, even when properly implemented, we show that these gradients exhibit high variance.

Specifically, the estimation of the hessian of the RL-objective, which is inherent in the meta-gradients, requires special consideration. In this section, we motivate and introduce the low variance curvature estimator (LVC): an improved estimator for the hessian of the RL-objective which promotes better meta-policy gradient updates. As we show in Appendix A.1, we can write the gradient of the meta-learning objective as

$$
\nabla_ {\theta} J ^ {I} (\theta) = \mathbb {E} _ {\mathcal {T} \sim \rho (\mathcal {T})} \left[ \mathbb {E} _ {\boldsymbol {\tau} ^ {\prime} \sim P _ {\mathcal {T}} \left(\boldsymbol {\tau} ^ {\prime} \mid \theta^ {\prime}\right)} \left[ \nabla_ {\theta^ {\prime}} \log P _ {\mathcal {T}} \left(\boldsymbol {\tau} ^ {\prime} \mid \theta^ {\prime}\right) R \left(\boldsymbol {\tau} ^ {\prime}\right) \nabla_ {\theta} U (\theta , \mathcal {T}) \right] \right] \tag {5}
$$

Since the update function  $U$  resembles a policy gradient step, its gradient  $\nabla_{\theta} U(\theta, \mathcal{T})$  involves computing the hessian of the reinforcement learning objective, i.e.,  $\nabla_{\theta}^{2} \mathbb{E}_{\tau \sim P_{\mathcal{T}}(\tau | \theta)}[R(\tau)]$ . Estimating this hessian has been discussed in Baxter & Bartlett (2001) and Furmston et al. (2016). In the infinite horizon MDP case, Baxter & Bartlett (2001) derived a decomposition of the hessian. We extend their finding to the finite horizon case, showing that the hessian can be decomposed into three matrix terms (see Appendix B.2 for proof):

$$
\nabla_ {\theta} U (\theta , \mathcal {T}) = I + \alpha \nabla_ {\theta} ^ {2} \mathbb {E} _ {\tau \sim P _ {\mathcal {T}} (\tau | \theta)} [ R (\tau) ] = I + \alpha \left(\mathcal {H} _ {1} + \mathcal {H} _ {2} + \mathcal {H} _ {1 2} + \mathcal {H} _ {1 2} ^ {\top}\right) \tag {6}
$$

whereby

$$
\mathcal {H} _ {1} = \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau} | \theta)} \left[ \sum_ {t = 0} ^ {H - 1} \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {a} _ {t}, \boldsymbol {s} _ {t}) \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {a} _ {t}, \boldsymbol {s} _ {t}) ^ {\top} \left(\sum_ {t ^ {\prime} = t} ^ {H - 1} r (\boldsymbol {s} _ {t ^ {\prime}}, \boldsymbol {a} _ {t ^ {\prime}})\right) \right]
$$

$$
\mathcal {H} _ {2} = \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau} | \theta)} \left[ \sum_ {t = 0} ^ {H - 1} \nabla_ {\theta} ^ {2} \log \pi_ {\theta} \left(\boldsymbol {a} _ {t}, \boldsymbol {s} _ {t}\right) \left(\sum_ {t ^ {\prime} = t} ^ {H - 1} r \left(\boldsymbol {s} _ {t ^ {\prime}}, \boldsymbol {a} _ {t ^ {\prime}}\right)\right) \right]
$$

$$
\mathcal {H} _ {1 2} = \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau} | \theta)} \left[ \sum_ {t = 0} ^ {H - 1} \nabla_ {\theta} \log \pi_ {\theta} (\boldsymbol {a} _ {t}, \boldsymbol {s} _ {t}) \nabla_ {\theta} Q _ {t} ^ {\pi_ {\theta}} (\boldsymbol {s} _ {t}, \boldsymbol {a} _ {t}) ^ {\top} \right]
$$

Here  $Q_{t}^{\pi_{\theta}}(\pmb{s}_{t}, \pmb{a}_{t}) = \mathbb{E}_{\pmb{\tau}^{t+1:H-1} \sim P_{\mathcal{T}}(\cdot|\theta)} \left[ \sum_{t'=t}^{H-1} r(\pmb{s}_{t'}, \pmb{a}_{t'}) |s_{t}, a_{t} \right]$  denotes the expected state-action value function under policy  $\pi_{\theta}$  at time  $t$ .

Computing the expectation of the RL-objective is in general intractable. Typically, its gradients are computed with a Monte Carlo estimate based on the policy gradient theorem (Eq. 82). In practical implementations, such an estimate is obtained by automatically differentiating a surrogate objective (Schulman et al., 2015b). However, this results in a highly biased hessian estimate which just computes  $\mathcal{H}_2$ , entirely dropping the terms  $\mathcal{H}_1$  and  $\mathcal{H}_{12} + \mathcal{H}_{12}^{\top}$ . In the notation of the previous section, it leads to neglecting the  $\nabla_{\theta}J_{\mathrm{pre}}$  term, ignoring the influence of the pre-update sampling distribution.

The issue can be overcome using the DiCE formulation, which allows to compute unbiased higher-order Monte Carlos estimates of arbitrary stochastic computation graphs (Foerster et al., 2018). The DiCE-RL objective can be rewritten as follows

$$
J ^ {\mathrm {D i C E}} (\boldsymbol {\tau}) = \sum_ {t = 0} ^ {H - 1} \left(\prod_ {t ^ {\prime} = 0} ^ {t} \frac {\pi_ {\theta} \left(\boldsymbol {a} _ {t ^ {\prime}} \mid \boldsymbol {s} _ {t ^ {\prime}}\right)}{\perp \left(\pi_ {\theta} \left(\boldsymbol {a} _ {t ^ {\prime}} \mid \boldsymbol {s} _ {t ^ {\prime}}\right)\right)}\right) r \left(\boldsymbol {s} _ {t}, \boldsymbol {a} _ {t}\right) \quad \boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau}) \tag {7}
$$

$$
\mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\boldsymbol {\tau}} (\boldsymbol {\tau} | \theta)} \left[ \nabla_ {\theta} ^ {2} J ^ {\mathrm {D i C E}} (\boldsymbol {\tau}) \right] = \mathcal {H} _ {1} + \mathcal {H} _ {2} + \mathcal {H} _ {1 2} + \mathcal {H} _ {1 2} ^ {\top} \tag {8}
$$

In that,  $\bot$  denotes the "stop-gradient" operator, i.e.,  $\bot (f_{\theta}(x))\to f_{\theta}(x)$  but  $\nabla_{\theta}\bot (f_{\theta}(x))\to 0$ . The sequential dependence of  $\pi_{\theta}(\pmb{a}_t|\pmb{s}_t)$  within the trajectory, manifesting itself through the product of importance weights in (7), results in high variance estimates of the hessian  $\nabla_{\theta}^{2}\mathbb{E}_{\tau \sim P_{\mathcal{T}}(\tau |\theta)}[R(\tau)]$ . As noted by Furmston et al. (2016),  $\mathcal{H}_{12}$  is particularly difficult to estimate, since it involves three nested sums along the trajectory. In section 7.2 we empirically show that the high variance estimates of the DiCE objective lead to noisy meta-policy gradients and poor learning performance.

To facilitate a sample efficient meta-learning, we introduce the low variance curvature (LVC) estimator:

$$
J ^ {\mathrm {L V C}} (\boldsymbol {\tau}) = \sum_ {t = 0} ^ {H - 1} \frac {\pi_ {\theta} \left(\boldsymbol {a} _ {t} \mid \boldsymbol {s} _ {t}\right)}{\perp \left(\pi_ {\theta} \left(\boldsymbol {a} _ {t} \mid \boldsymbol {s} _ {t}\right)\right)} \binom {H - 1} {\sum_ {t ^ {\prime} = t}} r \left(\boldsymbol {s} _ {t ^ {\prime}}, \boldsymbol {a} _ {t ^ {\prime}}\right) \quad \boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau}) \tag {9}
$$

$$
\mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\boldsymbol {\tau}} (\boldsymbol {\tau} | \theta)} \left[ \nabla_ {\theta} ^ {2} J ^ {\mathrm {L V C}} (\boldsymbol {\tau}) \right] = \mathcal {H} _ {1} + \mathcal {H} _ {2} \tag {10}
$$

By removing the sequential dependence of  $\pi_{\theta}(\pmb{a}_t|\pmb{s}_t)$  within trajectories, the hessian estimate neglects the term  $\mathcal{H}_{12} + \mathcal{H}_{12}^{\top}$  which leads to a variance reduction, but makes the estimate biased. The choice of this objective function is motivated by findings in Furmston et al. (2016): under certain conditions the term  $\mathcal{H}_{12} + \mathcal{H}_{12}^{\top}$  vanishes around local optima  $\theta^{*}$ , i.e.,  $\mathbb{E}_{\tau}[\nabla_{\theta}^{2}J^{\mathrm{LVC}}]\to \mathbb{E}_{\tau}[\nabla_{\theta}^{2}J^{\mathrm{DiCE}}]$  as  $\theta \rightarrow \theta^{*}$ . Hence, the bias of the LVC estimator becomes negligible close to local optima. The experiments in section 7.2 underpin the theoretical findings, showing that the low variance hessian estimates obtained through  $J^{\mathrm{LVC}}$  improve the sample-efficiency of meta-learning by a significant margin when compared to  $J^{\mathrm{DiCE}}$ . We refer the interested reader to Appendix B for derivations and a more detailed discussion.

# 6 PROMP: PROXIMAL META-POLICY SEARCH

Building on the previous sections, we develop a novel meta-policy search method based on the low variance curvature objective which aims to solve the following optimization problem:

$$
\max  _ {\theta} \mathbb {E} _ {\mathcal {T} \sim \rho (\mathcal {T})} \left[ \mathbb {E} _ {\boldsymbol {\tau} ^ {\prime} \sim P _ {\mathcal {T}} \left(\boldsymbol {\tau} ^ {\prime} \mid \theta^ {\prime}\right)} \left[ R \left(\boldsymbol {\tau} ^ {\prime}\right) \right] \right] \quad \text {w i t h} \quad \theta^ {\prime} := \theta + \alpha \nabla_ {\theta} \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau} \mid \theta)} \left[ J ^ {\mathrm {L V C}} (\boldsymbol {\tau}) \right] \tag {11}
$$

Prior work has optimized this objective using either vanilla policy gradient (VPG) or TRPO (Schulman et al., 2015a). TRPO holds the promise to be more data efficient and stable during the learning process when compared to VPG. However, it requires computing the Fisher information matrix (FIM). Estimating the FIM is particularly problematic in the meta-learning set up. The meta-policy gradients already involve second order derivatives; as a result, the time complexity of the FIM estimate is cubic in the number of policy parameters. Typically, the problem is circumvented using finite difference methods, which introduce further approximation errors.

The recently introduced PPO algorithm (Schulman et al., 2017) achieves comparable results to TRPO with the advantage of being a first order method. PPO uses a surrogate clipping objective which allows it to safely take multiple gradient steps without re-sampling trajectories.

$$
\mathbf {J} _ {\mathcal {T}} ^ {\mathrm {C L I P}} (\theta) = \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau}, \theta_ {o})} \left[ \sum_ {t = 0} ^ {H - 1} \min \left(\frac {\pi_ {\theta} (\boldsymbol {a} _ {t} | \boldsymbol {s} _ {t})}{\pi_ {\theta_ {o}} (\boldsymbol {a} _ {t} | \boldsymbol {s} _ {t})} A ^ {\pi_ {\theta_ {o}}} (\boldsymbol {s} _ {t}, \boldsymbol {a} _ {t}), \operatorname {c l i p} _ {1 - \epsilon} ^ {1 + \epsilon} \left(\frac {\pi_ {\theta} (\boldsymbol {a} _ {t} | \boldsymbol {s} _ {t})}{\pi_ {\theta_ {o}} (\boldsymbol {a} _ {t} | \boldsymbol {s} _ {t})}\right) A ^ {\pi_ {\theta_ {o}}} (\boldsymbol {s} _ {t}, \boldsymbol {a} _ {t})\right) \right]
$$

In case of Meta-RL, it does not suffice to just replace the post-update reward objective with  $J_{T}^{\mathrm{CLIP}}$ . In order to safely perform multiple meta-gradient steps based on the same sampled data from a recent policy  $\pi_{\theta_o}$ , we also need to 1) account for changes in the pre-update action distribution  $\pi_{\theta}(a_t|s_t)$ , and 2) bound changes in the pre-update state visitation distribution (Kakade & Langford, 2002).

We propose Proximal Meta-Policy Search (ProMP) which incorporates both the benefits of proximal policy optimization and the low variance curvature objective (see Alg. 1.) In order to comply with

Algorithm 1 Proximal Meta-Policy Search (ProMP)

Require: Task distribution  $\rho$ , step sizes  $\alpha$ ,  $\beta$ , KL-penalty coefficient  $\eta$ , clipping range  $\epsilon$

1: Randomly initialize  $\theta$  
2: while  $\theta$  not converged do  
3: Sample batch of tasks  $\mathcal{T}_i\sim \rho (\mathcal{T})$  
4: for step  $n = 0, \dots, N - 1$  do  
5: if  $n = 0$  then  
6: Set  $\theta_{o}\gets \theta$  
7: for all  $\mathcal{T}_i\sim \rho (\mathcal{T})$  do  
8: Sample pre-update trajectories  $\mathcal{D}_i = \{\tau_i\}$  from  $\mathcal{T}_i$  using  $\pi_{\theta}$  
9: Compute adapted parameters  $\theta_{o,i}^{\prime}\gets \dot{\theta} +\alpha \nabla_{\theta}J_{\tau_i}^{LR}(\theta)$  with  $\mathcal{D}_i = \{\tau_i\}$  
10: Sample post-update trajectories  $\mathcal{D}'_i = \{\tau_i'\}$  from  $\mathcal{T}_i$  using  $\pi_{\theta_{o,i}'}'$  
11: Update  $\theta \gets \theta +\beta \sum_{T_i}\nabla_\theta J_{T_i}^{\mathrm{ProMP}}(\theta)$  using each  $\mathcal{D}_i^\prime = \{\tau_i^\prime \}$

requirement 1), ProMP replaces the "stop gradient" importance weight  $\frac{\pi_{\theta}(a_t|s_t)}{\perp(\pi_{\theta}(a_t|s_t))}$  by the likelihood ratio  $\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_o}(a_t|s_t)}$ , which results in the following objective

$$
J _ {\mathcal {T}} ^ {L R} (\theta) = \mathbb {E} _ {\boldsymbol {\tau} \sim P _ {\mathcal {T}} (\boldsymbol {\tau}, \theta_ {o})} \left[ \sum_ {t = 0} ^ {H - 1} \frac {\pi_ {\theta} \left(\boldsymbol {a} _ {t} \mid \boldsymbol {s} _ {t}\right)}{\pi_ {\theta_ {o}} \left(\boldsymbol {a} _ {t} \mid \boldsymbol {s} _ {t}\right)} A ^ {\pi_ {\theta_ {o}}} \left(\boldsymbol {s} _ {t}, \boldsymbol {a} _ {t}\right) \right] \tag {12}
$$

An important feature of this objective is that its derivatives evaluated at  $\theta_{o}$  are identical to those of the LVC objective. To satisfy condition 2) we extend the clipped meta-objective with a KL-penalty term between  $\pi_{\theta}$  and  $\pi_{\theta_o}$ . This KL-penalty term enforces a soft local "trust region" around  $\pi_{\theta_o}$ , preventing the shift in state visitation distribution to become large during optimization. This enables us to take multiple meta-policy gradient steps without re-sampling. Altogether, ProMP optimizes

$$
J _ {\mathcal {T}} ^ {\text {P r o M P}} (\theta) = J _ {\mathcal {T}} ^ {\text {C L I P}} \left(\theta^ {\prime}\right) - \eta \bar {\mathcal {D}} _ {K L} \left(\pi_ {\theta_ {o}}, \pi_ {\theta}\right) \quad \text {s . t .} \quad \theta^ {\prime} = \theta + \alpha \nabla_ {\theta} J _ {\mathcal {T}} ^ {L R} (\theta), \quad \mathcal {T} \sim \rho (\mathcal {T}) \tag {13}
$$

ProMP consolidates the insights developed throughout the course of this paper, while at the same time making maximal use of recently developed policy gradients algorithms. First, its meta-learning formulation exploits the full structural knowledge of gradient-based meta-learning. Second, it incorporates a low variance estimate of the RL-objective hessian. Third, ProMP controls the statistical distance of both pre- and post-adaptation policies, promoting efficient and stable meta-learning. All in all, ProMP consistently outperforms previous gradient-based meta-RL algorithms in sample complexity, wall clock time, and asymptotic performance (see Section 7.1).

# 7 EXPERIMENTS

In order to empirically validate the theoretical arguments outlined above, this section provides a detailed experimental analysis that aims to answer the following questions: (i) How does ProMP perform against previous Meta-RL algorithms? (ii) How do the lower variance but biased LVC gradient estimates compare to the high variance, unbiased DiCE estimates? (iii) Do the different formulations result in different pre-update exploration properties? (iv) How do formulation I and formulation II differ in their meta-gradient estimates and convergence properties?

To answer the posed questions, we evaluate our approach on six continuous control Meta-RL benchmark environments based on OpenAI Gym and the Mujoco simulator (Brockman et al., 2016; Todorov et al., 2012). A description of the experimental setup is found in Appendix D. In all experiments, the reported curves are averaged over at least three random seeds. Returns are estimated based on sampled trajectories from the adapted post-update policies and averaged over sampled tasks. The source code and the experiments data are available on our supplementary website.

# 7.1 META-GRADIENT BASED COMPARISON

We compare our method, ProMP, in sample complexity and asymptotic performance to four other gradient-based approaches: TRPO-MAML (Finn et al., 2017), E-MAML-TRPO, E-MAML-VPG (Stadie et al., 2018), and LVC-VPG, an ablated version of our method that uses the LVC

objective in the adaptation step and meta-optimizes with vanilla policy gradient. These algorithms are benchmarked on six different locomotion tasks that require adaptation: the ant must learn to run in different directions in the 2D-plane, the half-cheetah and walker must switch between running forward and backward, the walker and hopper have to adapt to different configuration of their dynamics. Finally, we present how these algorithms perform in a high-dimensional environment, where a humanoid has to adapt to run in different directions.

![](images/e4578fa2475d61a1dcb91dbebc4c1c64efa1b741f189feffb39e78178735f736.jpg)

![](images/467cdb0d254ab0b1c85a8c610ea81fb636a2193dd2fd13504e5511f8850924e9.jpg)

![](images/b8ff08afb37f8e6c9a8ec48b9522eecf07d8d65499b7b1f05531b132a158d5a3.jpg)

![](images/82a8c6d2a064fb253c1b0858a23415d2042b57e67029aac671a5395a3cf03567.jpg)  
Figure 2: Meta-learning curves of ProMP and four other gradient-based meta-learning algorithms in six different Mujoco environments. ProMP outperforms previous work in all the the environments.

![](images/f7f2b5f4dc5b1eef3a6aede47378f86d6987d3a89ad3ebc827640680937c3a13.jpg)

![](images/eee64f246bc8a08a94e180d268f6aefefa654a968792362600a88664364acbff.jpg)

The results, shown in Figure 2, highlight the strength of ProMP in terms of sample efficiency and asymptotic performance. They also demonstrate the positive effect of the LVC objective: LVC-VPG, even though optimized with vanilla policy gradient, is often able to achieve comparable results to the prior methods that are optimized with TRPO. When compared to E-MAML-VPG, LVC proves strictly superior in performance which underpins the soundness of the theory developed throughout this paper. Results for four additional environments are displayed in Appendix D along with hyperparameter settings, environment specifications and a wall-clock time comparison of the algorithms.

# 7.2 ESTIMATOR VARIANCE AND ITS EFFECT ON META-LEARNING

In Section 5 we discussed how the DiCE formulation yields unbiased but high variance estimates of the RL-objective hessian and served as motivation for the low variance curvature (LVC) estimator. Here we investigate the meta-gradient variance of both estimators as well as its implication on the learning performance. Specifically, we report the relative standard deviation of the meta-policy gradients as well as the average return throughout the learning process in the HalfCheetahFwdBack environment. The results, depicted in Figure 3, highlight the advantage of the low variance curvature estimate. The trajectory level dependencies inherent in the DiCE estimator lead to a metagradient standard deviation that is on average two times higher when compared to LVC. As the learning curves indicate, the noisy gradients impede sample efficient meta-learning in case of DiCE. Meta-policy search based on the LVC estimator leads to substantially better learning properties.

# 7.3 COMPARISON OF INITIAL SAMPLING DISTRIBUTIONS

Here we evaluate the effect of the different objectives on the learned pre-update sampling distribution. We compare the low variance curvature (LVC) estimator with TRPO (LVC-TRPO)

![](images/53a955da4501142d9794d89b1ad0e2a3dc6535600433ef1c6ed19704bd288218.jpg)

![](images/71440b06c6f50cbbdfde10881021c6e681765194b600b5d2039c29e98b5dac69.jpg)  
Figure 3: Upper: Relative standard deviation of meta-policy gradients. Lower: Return in the HalfCheetah-FwdBack environment.

against MAML (Finn et al., 2017) and E-MAML-TRPO (Stadie et al., 2018) in a 2D environment on which the exploration behavior can be visualized. Each task of this environment corresponds to reaching a different corner location; however, the 2D agent only experiences reward when it is sufficiently close to the corner (translucent regions of Figure 4). Thus, to successfully identify the task, the agent must explore the different regions. We perform three inner adaptation steps on each task, allowing the agent to fully change its behavior from exploration to exploitation.

![](images/182dd3ebe03c4b70ec26cb7bc2d277959497d0804ee024805428b6e1fc01e0a8.jpg)  
Figure 4: Exploration patterns of the pre-update policy and exploitation post-update with different update functions. Through its superior credit assignment, the LVC objective learns a pre-update policy that is able to identify the current task and respectively adapt its policy, successfully reaching the goal (dark green circle).

![](images/48010782d3f5f300f31272c56e23b708487a050e3e5f626e34a6dfae894c3b0b.jpg)

![](images/cfd165f6cfcce087d2dcb9509e121cb2fd5d49786412a76de4d4fa6aa2548908.jpg)  
Pre-update Post-update

The different exploration-exploitation strategies are displayed in Figure 4. Since the MAML implementation does not assign credit to the pre-update sampling trajectory, it is unable to learn a sound exploration strategy for task identification and thus fails to accomplish the task. On the other hand, E-MAML, which corresponds to formulation II, learns to explore in long but random paths: because it can only assign credit to batches of pre-update trajectories, there is no notion of which actions in particular facilitate good task adaptation. As a consequence the adapted policy slightly misses the task-specific target. The LVC estimator, instead, learns a consistent pattern of exploration, visiting each of the four regions, which it harnesses to fully solve the task.

# 7.4 GRADIENT UPDATE DIRECTIONS OF THE TWO META-RL FORMULATIONS

To shed more light on the differences of the gradients of formulation I and formulation II, we evaluate the meta-gradient updates and the corresponding convergence to the optimum of both formulations in a simple 1D environment. In this environment, the agent starts in a random position in the real line and has to reach a goal located at the position 1 or -1. In order to visualize the convergence, we parameterize the policy with only two parameters  $\theta_0$  and  $\theta_{1}$ . We employ formulation I by optimizing the DiCE objective with VPG, and formulation II by optimizing its (E-MAML) objective with VPG.

Figure 5 depicts meta-gradient updates of the parameters  $\theta_{i}$  for both formulations. Formulation I (red) exploits the internal structure of the adaptation update yielding faster and steadier convergence to the optimum. Due to its inferior credit assignment, formulation II (green) produces noisier gradient estimates leading to worse convergence properties.

![](images/ee04386f245f334e0aa674f54fb863c43736a79f7b5d9386206d7612c3c854dd.jpg)  
Figure 5: Meta-gradient updates of policy parameters  $\theta_0$  and  $\theta_{1}$  in a 1D environment w.r.t Formulation I (red) and Formulation II (green).

# 8 CONCLUSION

In this paper we propose a novel Meta-RL algorithm, proximal meta-policy search (ProMP), which fully optimizes for the pre-update sampling distribution leading to effective task identification. Our method is the result of a theoretical analysis of gradient-based Meta-RL formulations, based on which we develop the low variance curvature (LVC) surrogate objective that produces low variance meta-policy gradient estimates. Experimental results demonstrate that our approach surpasses previous meta-reinforcement learning approaches in a diverse set of continuous control tasks. Finally, we underpin our theoretical contributions with illustrative examples which further justify the soundness and effectiveness of our method.

# REFERENCES

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained Policy Optimization. Technical report, 2017. URL https://arxiv.org/pdf/1705.10528.pdf.  
Maruan Al-Shedivat, Trapit Bansal, Umass Amherst, Yura Burda, Openai Ilya, Sutskever Openai, Igor Mordatch Openai, and Pieter Abbeel. Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments. In ICLR, 2018. URL https://goo.gl/tboqaN.  
Ferran Alet, Toms Lozano-Pérez, and Leslie P. Kaelbling. Modular meta-learning. Technical report, 6 2018. URL http://arxiv.org/abs/1806.10166.  
Marcin Andrychowicz, Misha Denil, Sergio Gmez Colmenarejo, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. Technical report, 2016. URL https://arxiv.org/pdf/1606.04474.pdf.  
Jonathan Baxter and Peter L Bartlett. Infinite-Horizon Policy-Gradient Estimation. Technical report, 2001. URL https://arxiv.org/pdf/1106.0665.pdf.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI Gym. Technical report, 6 2016. URL http://arxiv.org/abs/1606.01540.  
Yutian Chen, Matthew W Hoffman, Sergio Gmez Colmenarejo, Misha Denil, Timothy P Lillicrap, Matt Botvinick, and Nando De Freitas. Learning to Learn without Gradient Descent by Gradient Descent. In ICML, 2017.  
Ignasi Clavera, Jonas Rothfuss, John Schulman, Yasuhiro Fujita, Tamim Asfour, and Pieter Abbeel. Model-Based Reinforcement Learning via Meta-Policy Optimization. In CoRL, 2018. URL http://arxiv.org/abs/1809.05214.  
Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, and Pieter Abbeel. RL$^2$: Fast Reinforcement Learning via Slow Reinforcement Learning. CoRR, abs/1611.0:1-14, 2016. ISSN 0004-6361. doi: 10.1051/0004-6361/201527329. URL http://arxiv.org/abs/1611.02779.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In ICML, 2017.  
Jakob Foerster, Gregory Farquhar, Maruan Al-Shedivat, Tim Rocttäschel, Eric P Xing, and Shimon Whiteson. DiCE: The Infinitely Differentiable Monte Carlo Estimator. In ICML, 2018. URL https://goo.gl/xkkGxN.  
Kevin Frans, Jonathan Ho, Xi Chen, Pieter Abbeel, and John Schulman. Meta Learning Shared Hierarchies. In ICLR, 10 2018. URL http://arxiv.org/abs/1710.09767.  
Thomas Furmston, Guy Lever, David Barber, and Joelle Pineau. Approximate Newton Methods for Policy Search in Markov Decision Processes. Technical report, 2016. URL http://jmlr.org/papers/volume17/15-414/15-414.pdf.  
Abhishek Gupta, Benjamin Eysenbach, Chelsea Finn, and Sergey Levine. Unsupervised Meta-Learning for Reinforcement Learning. In ICML, 2018a.  
Abhishek Gupta, Russell Mendonca, Yuxuan Liu, Pieter Abbeel, and Sergey Levine. MetaReinforcement Learning of Structured Exploration Strategies. In ICML, 2018b. URL https://arxiv.org/pdf/1802.07245.pdf.  
Sepp Hochreiter, A. Steven Younger, and Peter R. Conwell. Learning To Learn Using Gradient Descent. In ICANN, pp. 87-94, 2001. URL http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.5.323.

Michael Husken and Christian Goerick. Fast learning for problem classes using knowledge based network initialization. In IJCNN. IEEE Computer Society Press, 2000. URL http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.9720&rep=rep1&type=pdf.  
Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. In ICML, 2002. URL https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf.  
Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Learning to Generalize: Meta-Learning for Domain Generalization. In AAAI, 2017. URL www.aaai.org.  
Thomas Miconi, Jeff Clune, and Kenneth O. Stanley. Differentiable plasticity: training plastic neural networks with backpropagation. In ICML, 4 2018. URL https://arxiv.org/abs/1804.02464.  
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A Simple Neural Attentive Meta-Learner. In ICLR, 7 2018. URL http://arxiv.org/abs/1707.03141.  
Alex Nichol, Joshua Achiam, and John Schulman. On First-Order Meta-Learning Algorithms. Technical report, 2018. URL http://arxiv.org/abs/1803.02999.  
Jan Peters and Stefan Schaal. Policy Gradient Methods for Robotics. In 2006 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 2219-2225. IEEE, 10 2006. ISBN 1-4244-0258-1. doi: 10.1109/IROS.2006.282564. URL http://ieeexplore.ieee.org/document/4058714/.  
Sachin Ravi and Hugo Larochelle. Optimization as a Model for Few-Shot Learning. In ICLR, 11 2017. URL https://openreview.net/forum?id=rJY0-Kcll.  
Steindr Saemundsson, Katja Hofmann, and Marc Peter Deisenroth. Meta Reinforcement Learning with Latent Variable Gaussian Processes. In UAI, 2018. URL https://arxiv.org/pdf/1803.07551.pdf.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, and Google Deepmind. Meta-Learning with Memory-Augmented Neural Networks. In ICML, 2016. URL http://proceedings.mlr.press/v48/santoro16.pdf.  
Juergen Schmidhuber. Evolutionary principles in self-referential learning. On learning how to learn: The meta-meta... hook. PhD thesis, Technische Universitaet Munchen, 1987. URL http://people.idsia.ch/~juergen/diploma.html.  
Jrgen Schmidhuber, Jieyu Zhao, and Marco Wiering. Shifting Inductive Bias with Success-Story Algorithm, Adaptive Levin Search, and Incremental Self-Improvement. Machine Learning, 28 (1):105-130, 1997. ISSN 08856125. doi: 10.1023/A:1007383707642. URL http://link.springer.com/10.1023/A:1007383707642.  
John Schulman, Nicolas Heess, Theophane Weber, and Pieter Abbeel. Gradient Estimation Using Stochastic Computation Graphs. In NIPS, 2015a. URL https://arxiv.org/pdf/1506.05254.pdf.  
John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. Trust Region Policy Optimization. ICML, 2015b. ISSN 2158-3226. doi: 10.1063/1.4927398. URL http://arxiv.org/abs/1502.05477.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov Openai. Proximal Policy Optimization Algorithms. CoRR, 2017. URL https://arxiv.org/pdf/1707.06347.pdf.  
Bradly C Stadie, Ge Yang, Rein Houthooft, Xi Chen, Yan Duan, Yuhuai Wu, Pieter Abbeel, and Ilya Sutskever. Some Considerations on Learning to Explore via Meta-Reinforcement Learning. Technical report, 2018. URL https://arxiv.org/pdf/1803.01118.pdf.

Flood Sung, Li Zhang, Tao Xiang, Timothy Hospedales, and Yongxin Yang. Learning to Learn: Meta-Critic Networks for Sample Efficient Learning. Technical report, 6 2017. URL http://arxiv.org/abs/1706.09529.  
Richard S. Sutton, David Mcallester, Satinder Singh, and Yishay Mansour. Policy Gradient Methods for Reinforcement Learning with Function Approximation. In NIPS, 2000. ISBN 0-262-19450-3. doi: 10.1.1.37.9714.  
Sebastian Thrun and Lorien Pratt. Learning to learn. 1998. ISBN 0792380479. URL https://dl.acm.org/citation.cfm?id=296639.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In IROS, pp. 5026-5033. IEEE, 10 2012. ISBN 978-1-4673-1736-8. doi: 10.1109/IROS.2012.6386109. URL http://ieeexplore.ieee.org/document/6386109/.  
Zhongwen Xu, Hado van Hasselt, and David Silver. Meta-Gradient Reinforcement Learning. Technical report, 5 2018. URL http://arxiv.org/abs/1805.09801.
