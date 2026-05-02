# ALGORITHMIC FRAMEWORK FOR MODEL-BASED DEEP REINFORCEMENT LEARNING WITH THEORETICAL GUARANTEES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Model-based reinforcement learning (RL) is considered to be a promising approach to reduce the sample complexity that hinders model-free RL. However, the theoretical understanding of such methods has been rather limited. This paper introduces a novel algorithmic framework for designing and analyzing model-based RL algorithms with theoretical guarantees. We design a meta-algorithm with a theoretical guarantee of monotone improvement to a local maximum of the expected reward. The meta-algorithm iteratively builds a lower bound of the expected reward based on the estimated dynamical model and sample trajectories, and then maximizes the lower bound jointly over the policy and the model. The framework extends the optimism-in-face-of-uncertainty principle to non-linear dynamical models in a way that requires no explicit uncertainty quantification. Instantiating our framework with simplification gives a variant of model-based RL algorithms Stochastic Lower Bounds Optimization (SLBO). Experiments demonstrate that SLBO achieves state-of-the-art performance when only one million or fewer samples are permitted on a range of continuous control benchmark tasks.

# 1 INTRODUCTION

In recent years deep reinforcement learning has achieved strong empirical success, including superhuman performances on Atari games and Go (Mnih et al., 2015; Silver et al., 2017) and learning locomotion and manipulation skills in robotics (Levine et al., 2016; Schulman et al., 2015b; Lillicrap et al., 2015). Many of these results are achieved by model-free RL algorithms that often require a massive number of samples, and therefore their applications are mostly limited to simulated environments. Model-based deep reinforcement learning, in contrast, exploits the information from state observations explicitly — by planning with an estimated dynamical model — and is considered to be a promising approach to reduce the sample complexity. Indeed, empirical results (Deisenroth & Rasmussen, 2011b; Deisenroth et al., 2013; Levine et al., 2016; Nagabandi et al., 2017; Kurutach et al., 2018; Pong et al., 2018a) have shown strong improvements in sample efficiency.

Despite promising empirical findings, many of theoretical properties of model-based deep reinforcement learning are not well-understood. For example, how does the error of the estimated model affect the estimation of the value function and the planning? Can model-based RL algorithms be guaranteed to improve the policy monotonically and converge to a local maximum of the value function? How do we quantify the uncertainty in the dynamical models?

It's challenging to address these questions theoretically in the context of deep RL with continuous state and action space and non-linear dynamical models. Due to the high-dimensionality, learning models from observations in one part of the state space and extrapolating to another part sometimes involves a leap of faith. The uncertainty quantification of the non-linear parameterized dynamical models is difficult — even without the RL components, it is an active but widely-open research area. Prior work in model-based RL mostly quantifies uncertainty with either heuristics or simpler models (Moldovan et al., 2015; Xie et al., 2016; Deisenroth & Rasmussen, 2011a).

Previous theoretical work on model-based RL mostly focuses on either the finite-state MDPs (Jaksch et al., 2010; Bartlett & Tewari, 2009; Fruit et al., 2018; Lakshmanan et al., 2015; Hinderer, 2005; Pirotta et al., 2015; 2013), or the linear parametrization of the dynamics, policy, or value func

tion (Abbasi-Yadkori & Szepesvári, 2011; Simchowitz et al., 2018; Dean et al., 2017; Sutton et al., 2012; Tamar et al., 2012), but not much on non-linear models. Even with an oracle prediction intervals<sup>1</sup> or posterior estimation, to the best of our knowledge, there was no previous algorithm with convergence guarantees for model-based deep RL.

Towards addressing these challenges, the main contribution of this paper is to propose a novel algorithmic framework for model-based deep RL with theoretical guarantees. Our meta-algorithm (Algorithm 1) extends the optimism-in-face-of-uncertainty principle to non-linear dynamical models in a way that requires no explicit uncertainty quantification of the dynamical models.

Let  $V^{\pi}$  be the value function  $V^{\pi}$  of a policy  $\pi$  on the true environment, and let  $\widehat{V}^{\pi}$  be the value function of the policy  $\pi$  on the estimated model  $\widehat{M}$ . We design provable upper bounds, denoted by  $D^{\pi, \widehat{M}}$ , on how much the error can compound and divert the expected value  $\widehat{V}^{\pi}$  of the imaginary rollouts from their real value  $V^{\pi}$ , in a neighborhood of some reference policy. Such upper bounds capture the intrinsic difference between the estimated and real dynamical model with respect to the particular reward function under consideration.

The discrepancy bounds  $D^{\pi, \widehat{M}}$  naturally leads to a lower bound for the true value function:

$$
V ^ {\pi} \geq \widehat {V} ^ {\pi} - D ^ {\pi , \widehat {M}}. \tag {1.1}
$$

Our algorithm iteratively collects batches of samples from the interactions with environments, builds the lower bound above, and then maximizes it over both the dynamical model  $\widehat{M}$  and the policy  $\pi$ . We can use any RL algorithms to optimize the lower bounds, because it will be designed to only depend on the sample trajectories from a fixed reference policy (as opposed to requiring new interactions with the policy iterate.)

We show that the performance of the policy is guaranteed to monotonically increase, assuming the optimization within each iteration succeeds (see Theorem 3.1.) To the best of our knowledge, this is the first theoretical guarantee of monotone improvement for model-based deep RL.

Readers may have realized that optimizing a robust lower bound is reminiscent of robust control and robust optimization. The distinction is that we optimistically and iteratively maximize the RHS of (1.1) jointly over the model and the policy. The iterative approach allows the algorithms to collect higher quality trajectory adaptively, and the optimism in model optimization encourages explorations of the parts of space that are not covered by the current discrepancy bounds.

To instantiate the meta-algorithm, we design a few valid discrepancy bounds in Section 4. In Section 4.1, we recover the norm-based model loss by imposing the additional assumption of a Lipschitz value function. The result suggests a norm is preferred compared to the square of the norm. Indeed in Section 6.2, we show that experimentally learning with  $\ell_2$  loss significantly outperforms the mean-squared error loss  $(\ell_2^2)$ .

In Section 4.2, we design a discrepancy bound that is invariant to the representation of the state space. Here we measure the loss of the model by the difference between the value of the predicted next state and the value of the true next state. Such a loss function is shown to be invariant to one-to-one transformation of the state space. Thus we argue that the loss is an intrinsic measure for the model error without any information beyond observing the rewards. We also refine our bounds in Section A by utilizing some mathematical tools of measuring the difference between policies in  $\chi^2$ -divergence (instead of KL divergence or TV distance).

Our analysis also sheds light on the comparison between model-based RL and on-policy model-free RL algorithms such as policy gradient or TRPO (Schulman et al., 2015a). The RHS of equation (1.1) is likely to be a good approximator of  $V^{\pi}$  in a larger neighborhood than the linear approximation of  $V^{\pi}$  used in policy gradient is (see Remark 4.5.)

Finally, inspired by our framework and analysis, we design a variant of model-based RL algorithms Stochastic Lower Bounds Optimization (SLBO). Experiments demonstrate that SLBO achieves state-of-the-art performance when only 1M samples are permitted on a range of continuous control benchmark tasks.

# 2 NOTATIONS AND PRELIMINARIES

We denote the state space by  $S$ , the action space by  $\mathcal{A}$ . A policy  $\pi(\cdot|s)$  specifies the conditional distribution over the action space given a state  $s$ . A dynamical model  $M(\cdot|s, a)$  specifies the conditional distribution of the next state given the current state  $s$  and action  $a$ . We will use  $M^{\star}$  globally to denote the unknown true dynamical model. Our target applications are problems with the continuous state and action space, although the results apply to discrete state or action space as well. When the model is deterministic,  $M(\cdot|s, a)$  is a dirac measure. In this case, we use  $M(s, a)$  to denote the unique value of  $s'$  and view  $M$  as a function from  $S \times \mathcal{A}$  to  $S$ . Let  $\mathcal{M}$  denote a (parameterized) family of models that we are interested in, and  $\Pi$  denote a (parameterized) family of policies.

Unless otherwise stated, for random variable  $X$ , we will use  $p_X$  to denote its density function.

Let  $S_0$  be the random variable for the initial state. Let  $S_t^{\pi, M}$  to denote the random variable of the states at steps  $t$  when we execute the policy  $\pi$  on the dynamic model  $M$  stating with  $S_0$ . Note that  $S_0^{\pi, M} = S_0$  unless otherwise stated. We will omit the subscript when it's clear from the context. We use  $A_t$  to denote the actions at step  $t$  similarly. We often use  $\tau$  to denote the random variable for the trajectory  $(S_0, A_1, \ldots, S_t, A_t, \ldots)$ . Let  $R(s, a)$  be the reward function at each step. We assume  $R$  is known throughout the paper, although  $R$  can be also considered as part of the model if unknown. Let  $\gamma$  be the discount factor.

Let  $V^{\pi, M}$  be the value function on the model  $M$  and policy  $\pi$  defined as:

$$
V ^ {\pi , M} (s) = \underset { \begin{array}{c} \forall t \geq 0, A _ {t} \sim \pi (\cdot | S _ {t}) \\ S _ {t + 1} \sim M (\cdot | S _ {t}, A _ {t}) \end{array} } {\mathbb {E}} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} R \left(S _ {t}, A _ {t}\right) \mid S _ {0} = s \right] \tag {2.1}
$$

We define  $V^{\pi, M} = \mathbb{E}\left[V^{\pi, M}(S_0)\right]$  as the expected reward-to-go at Step 0 (averaged over the random initial states). Our goal is to maximize the reward-to-go on the true dynamical model, that is,  $V^{\pi, M^{\star}}$ , over the policy  $\pi$ . For simplicity, throughout the paper, we set  $\kappa = \gamma(1 - \gamma)^{-1}$  since it occurs frequently in our equations. Every policy  $\pi$  induces a distribution of states visited by policy  $\pi$ :

Definition 2.1. For a policy  $\pi$ , define  $\rho^{\pi, M}$  as the discounted distribution of the states visited by  $\pi$  on  $M$ . Let  $\rho^{\pi}$  be a shorthand for  $\rho^{\pi, M^{\star}}$  and we omit the superscript  $M^{\star}$  throughout the paper. Concretely, we have  $\rho^{\pi} = (1 - \gamma)\sum_{t=0}^{\infty}\gamma^{t} \cdot pS_{t}^{\pi}$

# 3 ALGORTHMIC FRAMEWORK

As mentioned in the introduction, towards optimizing  $V^{\pi ,M^{\star}}$ , our plan is to build a lower bound for  $V^{\pi ,M^{\star}}$  of the following type and optimize it iteratively:

$$
V ^ {\pi , M ^ {\star}} \geq V ^ {\pi , \widehat {M}} - D (\widehat {M}, \pi) \tag {3.1}
$$

where  $D(\widehat{M}, \pi) \in \mathbb{R}_{\geq 0}$  bounds from above the discrepancy between  $V^{\pi, \widehat{M}}$  and  $V^{\pi, M^*}$ . Building such an estimizable discrepancy bound globally that holds for all  $\widehat{M}$  and  $\pi$  turns out to be rather difficult, if not impossible. Instead, we shoot for establishing such a bound over the neighborhood of a reference policy  $\pi_{\mathrm{ref}}$ .

$$
V ^ {\pi , M ^ {\star}} \geq V ^ {\pi , \widehat {M}} - D _ {\pi_ {\text {r e f}}, \delta} (\widehat {M}, \pi), \quad \forall \pi \text {s . t .} d (\pi , \pi_ {\text {r e f}}) \leq \delta \tag {R1}
$$

Here  $d(\cdot, \cdot)$  is a function that measures the closeness of two policies, which will be chosen later in alignment with the choice of  $D$ . We will mostly omit the subscript  $\delta$  in  $D$  for simplicity in the rest of the paper. We will require our discrepancy bound to vanish when  $\widehat{M}$  is an accurate model:

$$
\widehat {M} = M ^ {\star} \Longrightarrow D _ {\pi_ {\text {r e f}}} (\widehat {M}, \pi) = 0, \quad \forall \pi , \pi_ {\text {r e f}} \tag {R2}
$$

The third requirement for the discrepancy bound  $D$  is that it can be estimated and optimized in the sense that

$$
D _ {\pi_ {\text {r e f}}} (\widehat {M}, \pi) \text {i s o f t h e f o r m} \underset {\tau \sim \pi_ {\text {r e f}}, M ^ {\star}} {\mathbb {E}} [ f (\widehat {M}, \pi , \tau) ] \tag {R3}
$$

where  $f$  is a known differentiable function. We can estimate such discrepancy bounds for every  $\pi$  in the neighborhood of  $\pi_{\mathrm{ref}}$  by sampling empirical trajectories  $\tau^{(1)}, \ldots, \tau^{(n)}$  from executing policy  $\pi_{\mathrm{ref}}$  on the real environment  $M^{\star}$  and compute the average of  $f(\widehat{M}, \pi, \tau^{(i)})$ 's. We would have to insist that the expectation cannot be over the randomness of trajectories from  $\pi$  on  $M^{\star}$ , because then we would have to re-sample trajectories for every possible  $\pi$  encountered.

For example, assuming the dynamical models are all deterministic, one of the valid discrepancy bounds (under some strong assumptions) that will prove in Section 4 is a multiple of the error of the prediction of  $\widehat{M}$  on the trajectories from  $\pi_{\mathrm{ref}}$ :

$$
D _ {\pi_ {\text {r e f}}} (\widehat {M}, \pi) = L \cdot \underset {S _ {0}, \dots , S _ {t}, \sim \pi_ {\text {r e f}}, M ^ {\star}} {\mathbb {E}} \left[ \| \widehat {M} (S _ {t}) - S _ {t + 1} \| \right] \tag {3.2}
$$

Suppose we can establish such an discrepancy bound  $D$  (and the distance function  $d$ ) with properties (R1), (R2), and (R3), — which will be the main focus of Section 4 —, then we can devise the following meta-algorithm (Algorithm 1). We iteratively optimize the lower bound over the policy  $\pi_{k+1}$  and the model  $M_{k+1}$ , subject to the constraint that the policy is not very far from the reference policy  $\pi_k$  obtained in the previous iteration. For simplicity, we only state the population version with the exact computation of  $D_{\pi_{\mathrm{ref}}}(\widehat{M}, \pi)$ , though empirically it is estimated by sampling trajectories.

# Algorithm 1 Meta-Algorithm for Model-based RL

Inputs: Initial policy  $\pi_0$ . Discrepancy bound  $D$  and distance function  $d$  that satisfy equation (R1) and (R2).

For  $k = 0$  to  $T$ :

$$
\pi_ {k + 1}, M _ {k + 1} = \underset {\pi \in \Pi , M \in \mathcal {M}} {\operatorname {a r g m a x}} V ^ {\pi , M} - D _ {\pi_ {k}, \delta} (M, \pi) \tag {3.3}
$$

$$
\text {s . t .} d \left(\pi , \pi_ {k}\right) \leq \delta \tag {3.4}
$$

We first remark that the discrepancy bound  $D_{\pi_k}(M,\pi)$  in the objective plays the role of learning the dynamical model by ensuring the model to fit to the sampled trajectories. For example, using the discrepancy bound in the form of equation (3.2), we roughly recover the standard objective for model learning, with the caveat that we only have the norm instead of the square of the norm in MSE. Such distinction turns out to be empirically important for better performance (see Section 6.2).

Second, our algorithm can be viewed as an extension of the optimism-in-face-of-uncertainty (OFU) principle to non-linear parameterized setting: jointly optimizing  $M$  and  $\pi$  encourages the algorithm to choose the most optimistic model among those that can be used to accurately estimate the value function. (See (Jaksch et al., 2010; Bartlett & Tewari, 2009; Fruit et al., 2018; Lakshmanan et al., 2015; Pirotta et al., 2015; 2013) and references therein for the OFU principle in finite-state MDPs.) The main novelty here is to optimize the lower bound directly, without explicitly building any confidence intervals, which turns out to be challenging in deep learning. In other words, the uncertainty is measured straightforwardly by how the error would affect the estimation of the value function.

Thirdly, the maximization of  $V^{\pi ,M}$ , when  $M$  is fixed, can be solved by any model-free RL algorithms with  $M$  as the environment without querying any real samples. Optimizing  $V^{\pi ,M}$  jointly over  $\pi$ ,  $M$  can be also viewed as another RL problem with an extended actions space using the known "extended MDP technique". See (Jaksch et al., 2010, section 3.1) for details.

Our main theorem shows formally that the policy performance in the real environment is nondecreasing under the assumption that the real dynamics belongs to our parameterized family  $\mathcal{M}$ .<sup>3</sup>

Theorem 3.1. Suppose that  $M^{\star} \in \mathcal{M}$ , that  $D$  and  $d$  satisfy equation (R1) and (R2), and the optimization problem in equation (3.3) is solvable at each iteration. Then, Algorithm 1 produces a sequence of policies  $\pi_0, \ldots, \pi_T$  with monotonically increasing values:

$$
V ^ {\pi_ {0}, M ^ {*}} \leq V ^ {\pi_ {1}, M ^ {*}} \leq \dots \leq V ^ {\pi_ {T}, M ^ {*}} \tag {3.5}
$$

Moreover, as  $k \to \infty$ , the value  $V^{\pi_k, M^\star}$  converges to some  $V^{\bar{\pi}, M^\star}$ , where  $\bar{\pi}$  is a local maximum of  $V^{\pi, M^\star}$  in domain  $\Pi$ .

Proof of Theorem 3.1. Since  $D$  and  $d$  satisfy equation (R1), we have that

$$
V ^ {\pi_ {k + 1}, M ^ {\star}} \geq V ^ {\pi_ {k + 1}, M _ {k + 1}} - D _ {\pi_ {k}} \left(M _ {k + 1}, \pi_ {k + 1}\right)
$$

By the definition that  $\pi_{k + 1}$  and  $M_{k + 1}$  are the optimizers of equation (3.3), we have that

$$
V ^ {\pi_ {k + 1}, M _ {k + 1}} - D _ {\pi_ {k}} \left(M _ {k + 1}, \pi_ {k + 1}\right) \geq V ^ {\pi_ {k}, M ^ {\star}} - D _ {\pi_ {k}} \left(M ^ {\star}, \pi_ {k}\right) = V ^ {\pi_ {k}, M ^ {\star}} \quad (\text {b y})
$$

Combing the two equations above we complete the proof of equation (3.5).

For the second part of the theorem, by compactness, we have that a subsequence of  $\pi_k$  converges to some  $\bar{\pi}$ . By the monotonicity we have  $V^{\pi_k,M^\star} \leq V^{\bar{\pi},M^\star}$  for every  $k \geq 0$ . For the sake of contradiction, we assume  $\bar{\pi}$  is a not a local maximum, then in the neighborhood of  $\bar{\pi}$  there exists  $\pi'$  such that  $V^{\pi',M^\star} > V^{\bar{\pi},M^\star}$  and  $d(\bar{\pi},\pi') < \delta/2$ . Let  $t$  be such that  $\pi_t$  is in the  $\delta/2$ -neighborhood of  $\bar{\pi}$ . Then we see that  $(\pi',M^\star)$  is a better solution than  $(\pi_{t+1},M_{t+1})$  for the optimization problem (3.3) in iteration  $t$  because  $V^{\pi',M^\star} > V^{\bar{\pi},M^\star} \geq V^{\pi_{t+1},M_t} \geq D_{\pi_t}(M_{t+1},\pi_{t+1})$ . (Here the last inequality uses equation (R1) with  $\pi_t$  as  $\pi_{\mathrm{ref}}$ .) The fact  $(\pi',M^\star)$  is a strictly better solution than  $(\pi_{t+1},M_{t+1})$  contradicts the fact that  $(\pi_{t+1},M_{t+1})$  is defined to be the optimal solution of (3.3). Therefore  $\bar{\pi}$  is a local maximum and we complete the proof.

# 4 DISCREPANCY BOUNDS DESIGN

![](images/ae6b21e6f161c75fc5a73b44a0b4787cefb0a69d3e385611d16cf0090ffcaade.jpg)

In this section, we design discrepancy bounds that can provably satisfy the requirements (R1), (R2), and (R3). We design increasingly stronger discrepancy bounds from Section 4.1 to Section A.

# 4.1 NORM-BASED PREDICTION ERROR BOUNDS

In this subsection, we assume the dynamical model  $M^{\star}$  is deterministic and we also learn with a deterministic model  $\widehat{M}$ . Under assumptions defined below, we derive a discrepancy bound  $D$  of the form  $\| \widehat{M}(S, A) - M^{\star}(S, A) \|$  averaged over the observed state-action pair  $(S, A)$  on the dynamical model  $\widehat{M}$ . This suggests that the norm is a better metric than the mean-squared error for learning the model, which is empirically shown in Section 6.2. Through the derivation, we will also introduce a telescoping lemma, which serves as the main building block towards other finer discrepancy bounds.

We make the (strong) assumption that the value function  $V^{\pi, \widehat{M}}$  on the estimated dynamical model is  $L$ -Lipschitz w.r.t to some norm  $\| \cdot \|$  in the sense that

$$
\forall s, s ^ {\prime} \in \mathcal {S}, \left| V ^ {\pi , \widehat {M}} (s) - V ^ {\pi , \widehat {M}} (s ^ {\prime}) \right| \leq L \cdot \| s - s ^ {\prime} \| \tag {4.1}
$$

In other words, nearby starting points should give reward-to-go under the same policy  $\pi$ . We note that not every real environment  $M^{\star}$  has this property, let alone the estimated dynamical models. However, once the real dynamical model induces a Lipschitz value function, we may penalize the Lipschitz-ness of the value function of the estimated model during the training.

We start off with a lemma showing that the expected prediction error is an upper bound of the discrepancy between the real and imaginary values.

Lemma 4.1. Suppose  $V^{\pi, \overline{\mathcal{M}}}$  is  $L$ -Lipschitz (in the sense of (4.1)). Recall  $\kappa = \gamma(1 - \gamma)^{-1}$ .

$$
\left| V ^ {\pi , \widehat {M}} - V ^ {\pi , M ^ {\star}} \right| \leq \kappa L \underset { \begin{array}{c} S \sim \rho^ {\pi} \\ A \sim \pi (\cdot | S) \end{array} } {\mathbb {E}} \left[ \| \widehat {M} (S, A) - M ^ {\star} (S, A) \| \right] \tag {4.2}
$$

However, in RHS in equation 4.2 cannot serve as a discrepancy bound because it does not satisfy the requirement (R3) — to optimize it over  $\pi$  we need to collect samples from  $\rho^{\pi}$  for every iterate  $\pi$  — the state distribution of the policy  $\pi$  on the real model  $M^{\star}$ . The main proposition of this subsection stated next shows that for every  $\pi$  in the neighborhood of a reference policy  $\pi_{\mathrm{ref}}$ , we can replace the distribution  $\rho^{\pi}$  be a fixed distribution  $\rho^{\pi_{\mathrm{ref}}}$  with incurring only a higher order approximation. We use the expected KL divergence between two  $\pi$  and  $\pi_{\mathrm{ref}}$  to define the neighborhood:

$$
d ^ {\mathrm {K L}} (\pi , \pi_ {\mathrm {r e f}}) = \underset {S \sim \rho^ {\pi}} {\mathbb {E}} \left[ K L (\pi (\cdot | S), \pi_ {\mathrm {r e f}} (\cdot | S)) ^ {1 / 2} \right] \tag {4.3}
$$

Proposition 4.2. In the same setting of Lemma 4.1, assume in addition that  $\pi$  is close to a reference policy  $\pi_{\mathrm{ref}}$  in the sense that  $d^{\mathrm{KL}}(\pi ,\pi_{\mathrm{ref}})\leq \delta$ , and that the states in  $\mathcal{S}$  are uniformly bounded in the sense that  $\| s\| \leq B,\forall s\in \mathcal{S}$ . Then,

$$
\left| V ^ {\pi , \widehat {M}} - V ^ {\pi , M ^ {\star}} \right| \leq \kappa L _ {\substack {S \sim \rho^ {\pi_ {\text {ref}}} \\ A \sim \pi (\cdot | S)}} \mathbb {E} \left[ \| \widehat {M} (S, A) - M ^ {\star} (S, A) \| \right] + 2 \kappa^ {2} \delta B \tag{4.4}
$$

In a benign scenario, the second term in the RHS of equation (4.4) should be dominated by the first term when the neighborhood size  $\delta$  is sufficiently small. Moreover, the term  $B$  can also be replaced by  $\max_{S,A}\| \widehat{M} (S,A) - M^{\star}(S,A)\|$  (see the proof that is deferred to Section C.). The dependency on  $\kappa$  may not be tight for real-life instances, but we note that most analysis of similar nature loses the additional  $\kappa$  factor Schulman et al. (2015a); Achiam et al. (2017), and it's inevitable in the worst-case.

A telescoping lemma. Towards proving Propositions 4.2 and deriving stronger discrepancy bound, we define the following quantity that captures the discrepancy between  $\widehat{M}$  and  $M^{\star}$  on a single state-action pair  $(s,a)$ .

$$
G ^ {\pi , \widehat {M}} (s, a) = \underset {\hat {s} ^ {\prime} \sim \widehat {M} (\cdot | s, a)} {\mathbb {E}} V ^ {\pi , \widehat {M}} \left(\hat {s} ^ {\prime}\right) - \underset {s ^ {\prime} \sim M ^ {\star} (\cdot | s, a)} {\mathbb {E}} V ^ {\pi , \widehat {M}} \left(s ^ {\prime}\right) \tag {4.5}
$$

Note that if  $M, \widehat{M}$  are deterministic, then  $G^{\pi, \widehat{M}}(s, a) = V^{\pi, \widehat{M}}(\widehat{M}(s, a)) - V^{\pi, \widehat{M}}(M^{\star}(s, a))$ . We give a telescoping lemma that decompose the discrepancy between  $V^{\pi, M}$  and  $V^{\pi, M^{\star}}$  into the expected single-step discrepancy  $G$ .

Lemma 4.3. [Telescoping Lemma] Recall that  $\kappa \coloneqq \gamma (1 - \gamma)^{-1}$ . For any policy  $\pi$  and dynamical models  $M, \widehat{M}$ , we have that

$$
V ^ {\pi , \widehat {M}} - V ^ {\pi , M} = \kappa_ {\substack {S \sim \rho^ {\pi , M} \\ A \sim \pi (\cdot | S)}} \mathbb {E} \left[ G ^ {\pi , \widehat {M}} (S, A) \right] \tag{4.6}
$$

The proof is reminiscent of the telescoping expansion in Kakade & Langford (2002) (c.f. Schulman et al. (2015a)) for characterizing the value difference of two policies, but we apply it to deal with the discrepancy between models. The detail is deferred to Section B. With the telescoping Lemma 4.3, Proposition 4.1 follows straightforwardly from Lipschitzness of the imaginary value function. Proposition 4.2 follows from that  $\rho^{\pi}$  and  $\rho^{\pi_{\mathrm{ref}}}$  are close. We defer the proof to Appendix C.

# 4.2 REPRESENTATION-INVARIANT DISCREPANCY BOUNDS

The main limitation of the norm-based discrepancy bounds in previous subsection is that it depends on the state representation. Let  $\mathcal{T}$  be a one-to-one map from the state space  $S$  to some other space  $S^{\prime}$ , and for simplicity of this discussion let's assume a model  $M$  is deterministic. Then if we represent every state  $s$  by its transformed representation  $\mathcal{T}s$ , then the transformed model  $M^{\mathcal{T}}$  defined as  $M^{\mathcal{T}}(s,a)\triangleq \mathcal{T}M(\mathcal{T}^{-1}s,a)$  together with the transformed reward  $R^{\mathcal{T}}(s,a)\triangleq R(\mathcal{T}^{-1}s,a)$  and transformed policy  $\pi^{\mathcal{T}}(s)\triangleq \pi (\mathcal{T}^{-1}s)$  is equivalent to the original set of the model, reward, and policy in terms of the performance (Lemma C.1). Thus such transformation  $\mathcal{T}$  is not identifiable from only observing the reward. However, the norm in the state space is a notion that depends on the hidden choice of the transformation  $\mathcal{T}$ .<sup>4</sup>

Another limitation is that the loss for the model learning should also depend on the state itself instead of only on the difference  $\widehat{M}(S, A) - M^{\star}(S, A)$ . It is possible that when  $S$  is at a critical position, the prediction error needs to be highly accurate so that the model  $\widehat{M}$  can be useful for planning. On the other hand, at other states, the dynamical model is allowed to make bigger mistakes because they are not essential to the reward.

We propose the following discrepancy bound towards addressing the limitations above. Recall the definition of  $G^{\pi, \widehat{M}}(s, a) = V^{\pi, \widehat{M}}(\widehat{M}(s, a)) - V^{\pi, \widehat{M}}(M^{\star}(s, a))$  which measures the difference

between  $\widehat{M}(s, a)$  and  $M^{\star}(s, a)$  according to their imaginary rewards. We construct a discrepancy bound using the absolute value of  $G$ . Let's define  $\varepsilon_1$  and  $\varepsilon_{\mathrm{max}}$  as the average of  $|G^{\pi, \widehat{M}}|$  and its maximum:  $\varepsilon_1 = \mathbb{E}_{S \sim \rho^{\pi_{\mathrm{ref}}}}\left[\left|G^{\pi, \widehat{M}}(S, A)\right|\right]$  and  $\varepsilon_{\mathrm{max}} = \max_S\left|G^{\pi, \widehat{M}}(S)\right|$  where  $G^{\pi, \widehat{M}}(S) = \mathbb{E}_{A \sim \pi}\left[G^{\pi, \widehat{M}}(S, A)\right]$ . We will show that the following discrepancy bound  $D_{\pi_{\mathrm{ref}}}^{G}(\widehat{M}, \pi)$  satisfies the property (R1), (R2).

$$
D _ {\pi_ {\text {r e f}}} ^ {G} (\widehat {M}, \pi) = \kappa \cdot \varepsilon_ {1} + \kappa^ {2} \delta \varepsilon_ {\max } \tag {4.7}
$$

Proposition 4.4. Let  $d^{\mathrm{KL}}$  and  $D^{G}$  be defined as in equation (4.3) and (4.7). Then the choice  $d = d^{\mathrm{KL}}$  and  $D = D^{G}$  satisfies the basic requirements (equation (R1) and (R2)). Moreover,  $G$  is invariant w.r.t any one-to-one transformation of the state space (in the sense of equation C.3 in the proof).

The proof follows from the telescoping lemma (Lemma 4.3) and is deferred to Section C. We remark that the first term  $\kappa \varepsilon_{1}$  can in principle be estimated and optimized approximately: the expectation be replaced by empirical samples from  $\rho^{\pi_{\mathrm{ref}}}$ , and  $G^{\pi ,\hat{M}}$  is an analytical function of  $\pi$  and  $\widehat{M}$  when they are both deterministic, and therefore can be optimized by back-propagation through time (BPTT). (When  $\pi$  and  $\widehat{M}$  and are stochastic with a re-parameterizable noise such as Gaussian distribution Kingma & Welling (2013), we can also use back-propagation to estimate the gradient.) The second term in equation (4.7) is difficult to optimize because it involves the maximum. However, it can be in theory considered as a second-order term because  $\delta$  can be chosen to be a fairly small number. (In the refined bound in Section A, the dependency on  $\delta$  is even milder.)

Remark 4.5. Proposition 4.4 intuitively suggests a technical reason of why model-based approach can be more sample-efficient than policy gradient based algorithms such as TRPO or PPO (Schulman et al., 2015a; 2017). The approximation error of  $V^{\pi, \widehat{M}}$  in model-based approach decreases as the model error  $\varepsilon_1, \varepsilon_{\max}$  decrease or the neighborhood size  $\delta$  decreases, whereas the approximation error in policy gradient only linearly depends on the neighborhood size Schulman et al. (2015a). In other words, model-based algorithms can trade model accuracy for a larger neighborhood size, and therefore the convergence can be faster (in terms of outer iterations.) This is consistent with our empirical observation that the model can be accurate in a descent neighborhood of the current policy so that the constraint (3.4) can be empirically dropped. We also refine our bonds in Section A, where the discrepancy bounds is proved to decay faster in  $\delta$ .

# 5 ADDITIONAL RELATED WORK

Model-based reinforcement learning is expected to require fewer samples than model-free algorithms (Deisenroth et al., 2013) and has been successfully applied to robotics in both simulation and in the real world (Deisenroth & Rasmussen, 2011b; Morimoto & Atkeson, 2003; Deisenroth et al., 2011) using dynamical models ranging from Gaussian process (Deisenroth & Rasmussen, 2011b; Ko & Fox, 2009), time-varying linear models (Levine & Koltun, 2013; Lioutikov et al., 2014; Levine & Abbeel, 2014; Yip & Camarillo, 2014), mixture of Gaussians (Khansari-Zadeh & Billard, 2011), to neural networks (Hunt et al., 1992; Nagabandi et al., 2017; Kurutach et al., 2018; Tangkaratt et al., 2014; Sanchez-Gonzalez et al., 2018; Pascanu et al., 2017). In particular, the work of Kurutach et al. (2018) uses an ensemble of neural networks to learn the dynamical model, and significantly reduces the sample complexity compared to model-free approaches. The work of Chua et al. (2018) makes further improvement by using a probabilistic model ensemble. Clavera et al. (Clavera et al., 2018) extended this method with meta-policy optimization and improve the robustness to model error. In contrast, we focus on theoretical understanding of model-based RL and the design of new algorithms, and our experiments use a single neural network to estimate the dynamical model.

Our discrepancy bound in Section 4 is closely related to the work (Farahmand et al., 2017) on the value-aware model loss. Our approach differs from it in three details: a) we use the absolute value of the value difference instead of the squared difference; b) we use the imaginary value function from the estimated dynamical model to define the loss, which makes the loss purely a function of the estimated model and the policy; c) we show that the iterative algorithm, using the loss function as a building block, can converge to a local maximum, partly by cause of the particular choices made in a) and b). Asadi et al. (2018) also study the discrepancy bounds under Lipschitz condition of the MDP.

Prior work explores a variety of ways of combining model-free and model-based ideas to achieve the best of the two methods (Sutton, 1991; 1990; Racanière et al., 2017; Mordatch et al., 2016). For example, estimated models (Levine & Koltun, 2013; Gu et al., 2016; Kalweit & Boedecker, 2017) are used to enrich the replay buffer in the model-free off-policy RL. Pong et al. (2018b) proposes goal-conditioned value functions trained by model-free algorithms and uses it for model-based controls. Feinberg et al. (2018); Buckman et al. (2018) use dynamical models to improve the estimation of the value functions in the model-free algorithms.

On the control theory side, Dean et al. (2018; 2017) provide strong finite sample complexity bounds for solving linear quadratic regulator using model-based approach. Boczar et al. (2018) provide finite-data guarantees for the "coarse-ID control" pipeline, which is composed of a system identification step followed by a robust controller synthesis procedure. Our method is inspired by the general idea of maximizing a low bound of the reward in (Dean et al., 2017). By contrast, our work applies to non-linear dynamical systems. Our algorithms also estimate the models iteratively based on trajectory samples from the learned policies.

Strong model-based and model-free sample complexity bounds have been achieved in the tabular case (finite state space). We refer the readers to (Kakade et al., 2018; Dann et al., 2017; Szita & Szepesvári, 2010; Kearns & Singh, 2002; Jaksch et al., 2010) and the reference therein. Our work focuses on continuous and high-dimensional state space (though the results also apply to tabular case).

Another line of work of model-based reinforcement learning is to learn a dynamic model in a hidden representation space, which is especially necessary for pixel state spaces (Kakade et al., 2018; Dann et al., 2017; Szita & Szeptsvári, 2010; Kearns & Singh, 2002; Jaksch et al., 2010). Srinivas et al. (2018) shows the possibility to learn an abstract transition model to imitate expert policy. Oh et al. (2017) learns the hidden state of a dynamical model to predict the value of the future states and applies RL or planning on top of it. Serban et al. (2018); Ha & Schmidhuber (2018) learns a bottleneck representation of the states. Our framework can be potentially combined with this line of research.

# 6 PRACTICAL IMPLEMENTATION AND EXPERIMENTS

# 6.1 PRACTICAL IMPLEMENTATION

We design with simplification of our framework a variant of model-based RL algorithms, Stochastic Lower Bound Optimization (SLBO). First, we removed the constraints (3.4). Second, we stop the gradient w.r.t  $M$  (but not  $\pi$ ) from the occurrence of  $M$  in  $V^{\pi, M}$  in equation (3.3) (and thus our practical implementation is not optimism-driven.)

Extending the discrepancy bound in Section 4.1, we use a multi-step prediction loss for learning the models with  $\ell_2$  norm. For a state  $s_t$  and action sequence  $a_{t:t + h}$ , we define the  $h$ -step prediction  $\hat{s}_{t + h}$  as  $\hat{s}_t = s_t$ , and for  $h\geq 0$ ,  $\hat{s}_{t + h + 1} = \widehat{M}_{\phi}(\hat{s}_{t + h},a_{t + h})$ , The  $H$ -step loss is then defined as

$$
\mathcal {L} _ {\phi} ^ {(H)} \left(\left(s _ {t: t + h}, a _ {t: t + h}\right); \phi\right) = \frac {1}{H} \sum_ {i = 1} ^ {H} \| \hat {s} _ {t + i} - s _ {t + i} \| _ {2}. \tag {6.1}
$$

A similar loss is also used in Nagabandi et al. (2017) for validation. We note that we use  $\ell_2$ -norm instead of the square of  $\ell_2$  norm. The loss function we attempt to optimize at iteration  $k$  is thus<sup>5</sup>

$$
\max  _ {\phi , \theta} V ^ {\pi_ {\theta}, \operatorname {s g} (\widehat {M} _ {\phi})} - \lambda \underset {\left(s _ {t: t + h}, a _ {t: t + h}\right) \sim \pi_ {k}, M ^ {\star}} {\mathbb {E}} \left[ \mathcal {L} _ {\phi} ^ {(H)} \left(\left(s _ {t: t + h}, a _ {t: t + h}\right); \phi\right) \right] \tag {6.2}
$$

where  $\lambda$  is a tunable parameter and  $\mathrm{sg}$  denotes the stop gradient operation.

We note that the term  $V^{\pi_{\theta}, \mathrm{sg}(\widehat{M}_{\phi})}$  depends on both the parameter  $\theta$  and the parameter  $\phi$  but there is no gradient passed through  $\phi$ , whereas  $\mathcal{L}_{\phi}^{(H)}$  only depends on the  $\phi$ . We optimize equation (6.2) by alternatively maximizing  $V^{\pi_{\theta}, \mathrm{sg}(\widehat{M}_{\phi})}$  and minimizing  $\mathcal{L}_{\phi}^{(H)}$ : for the former, we use TRPO with samples from the estimated dynamical model  $\widehat{M}_{\phi}$  (by treating  $\widehat{M}_{\phi}$  as a fixed simulator), and for

the latter we use standard stochastic gradient methods. Algorithm 2 gives a pseudo-code for the algorithm. The  $n_{\mathrm{model}}$  and  $n_{\mathrm{policy}}$  iterations are used to balance the number of steps of TRPO and Adam updates within the loop indexed by  $n_{\mathrm{inner}}$ .<sup>6</sup>

Algorithm 2 Stochastic Lower Bound Optimization (SLBO)  
1: Initialize model network parameters  $\phi$  and policy network parameters  $\theta$   
2: Initialize dataset  $\mathcal{D} \gets \emptyset$   
3: for  $n_{\text{outer}}$  iterations do  
4:  $\mathcal{D} \gets \mathcal{D} \cup \{\text{collect } n_{\text{collect}}$  samples from real environment using  $\pi_{\theta}$  with noises  $\}$   
5: for  $n_{\text{inner}}$  iterations do  $\triangleright$  optimize (6.2) with stochastic alternating updates  
6: for  $n_{\text{model}}$  iterations do  
7: optimize (6.1) over  $\phi$  with sampled data from  $\mathcal{D}$  by one step of Adam  
8: for  $n_{\text{policy}}$  iterations do  
9:  $\mathcal{D}' \gets \{\text{collect } n_{\text{trpo}}$  samples using  $\widehat{M}_{\phi}$  as dynamics  $\}$   
10: optimize  $\pi_{\theta}$  by running TRPO on  $\mathcal{D}'$

Power of stochasticity and connection to standard MB RL: We identify the main advantage of our algorithms over standard model-based RL algorithms is that we alternate the updates of the model and the policy within an outer iteration. By contrast, most of the existing model-based RL methods only optimize the models once (for a lot of steps) after collecting a batch of samples (see Algorithm 3 for an example). The stochasticity introduced from the alternation with stochastic samples seems to dramatically reduce the overfitting (of the policy to the estimated dynamical model) in a way similar to that SGD regularizes ordinary supervised training. Another way to view the algorithm is that the model obtained from line 7 of Algorithm 2 at different inner iteration serves as an ensemble of models. We do believe that a cleaner and easier instantiation of our framework (with optimism) exists, and the current version, though performing very well, is not necessarily the best implementation.

Entropy regularization: An additional component we apply to SLBO is the commonly-adopted entropy regularization in policy gradient method (Williams & Peng, 1991; Mnih et al., 2016), which was found to significantly boost the performance in our experiments (ablation study in Appendix F.5). Specifically, an additional entropy term is added to the objective function in TRPO. We hypothesize that entropy bonus helps exploration, diversifies the collected data, and thus prevents overfitting.

# 6.2 EXPERIMENTAL RESULTS

We evaluate our algorithm SLBO (Algorithm 2) on five continuous control tasks from rllab (Duan et al., 2016), including Swimmer, Half Cheetah, Humanoid, Ant, Walker. All environments we test have a maximum horizon of 500, which is longer than most of the existing model-based RL work (Nagabandi et al., 2017; Kurutach et al., 2018). (Environments with longer horizons are commonly harder to train.) More details can be found in Appendix F.1.

Baselines. We compare our algorithm with 3 other algorithms including: (1) Soft Actor-Critic (SAC) (Haarnoja et al., 2018), the state-of-the-art model-free off-policy algorithm in sample efficiency; (2) Trust-Region Policy Optimization (TRPO) (Schulman et al., 2015a), a policy-gradient based algorithm; and (3) Model-Based TRPO, a standard model-based algorithm described in Algorithm 3. Details of these algorithms can be found in Appendix F.4. $^{8}$

The result is shown in Figure 1. In Fig 1, our algorithm shows superior convergence rate (in number of samples) than all the baseline algorithms while achieving better final performance with 1M samples.

Specifically, we mark model-free TRPO performance after 8 million steps by the dotted line in Fig 1 and find out that our algorithm can achieve comparable or better final performance in one million steps. For ablation study, we also add the performance of SLBO-MSE, which corresponds to running SLBO with squared  $\ell_2$  model loss instead of  $\ell_2$ . SLBO-MSE performs significantly worse than SLBO on four environments, which is consistent with our derived model loss in Section 4.1. Ablation study of multi-step model training can be found in Appendix F.5. Our code will be released soon.

![](images/e98c756d9b4413430caaee4c228b02b1e8c68418c8a0b2536e0e8c8c159dc132.jpg)  
(a) Swimmer

![](images/b2332cbd72c67875e40cb02095814142b0a788ffe3c1fbb5f32c0d2adf00979b.jpg)  
(b) Half Cheetah

![](images/3a1b882b6c253a45987e6c6b64f9e12abfa3ebfdfe5435f723bb83499236f3ef.jpg)

![](images/91a406cee949e4dd2a381cc2b7e2c05c8b35bf4a53b840a4d7b88008525b9d96.jpg)  
(d) Walker

![](images/43bb2eb4531e97d5377ac7579bf5bcb546863aa04a679989abcfd54b9fe9f028.jpg)  
(c) Ant

![](images/9558cefd6eb810cd6593012ac080495f5e9d25a9fc13ef118dfd8af772becd22.jpg)  
(e) Humanoid  
Figure 1: Comparison between SLBO (ours), SLBO with squared  $\ell^2$  model loss (SLBO-MSE), vanilla model-based TRPO (MB-TRPO), model-free TRPO (MF-TRPO), and Soft Actor-Critic (SAC). We average the results over 10 different random seeds, where the solid lines indicate the mean and shaded areas indicate one standard deviation. The dotted reference lines are the total rewards of MF-TRPO after 8 million steps.

# 7 CONCLUSIONS

We devise a novel algorithmic framework for designing and analyzing model-based RL algorithms with the guarantee to convergence monotonically to a local maximum of the reward. Experimental results show that our proposed algorithm (SLBO) achieves new state-of-the-art performance on several Mujoco benchmark tasks when one million or fewer samples are permitted.

A compelling (but obvious) empirical open question then given rise to is whether model-based RL can achieve near-optimal reward on other more complicated tasks or real-world robotic tasks with fewer samples. We believe that understanding the trade-off between optimism and robustness is essential to design more sample-efficient algorithms. Currently, we observed empirically that the optimism-driven part of our proposed meta-algorithm (optimizing  $V^{\pi,\widehat{M}}$  over  $\widehat{M}$ ) may lead to instability in the optimization, and therefore don't in general help the performance. It's left for future work to find practical implementation of the optimism-driven approach.

In our theory, we assume that the parameterized model class contains the true dynamical model. Removing this assumption is also another interesting open question. It would be also very interesting if the theoretical analysis can be applied other settings involving model-based approaches (e.g., model-based imitation learning).

# REFERENCES

Yasin Abbasi-Yadkori and Csaba Szepesvári. Regret bounds for the adaptive control of linear quadratic systems. In Proceedings of the 24th Annual Conference on Learning Theory, pp. 1-26, 2011.  
Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. arXiv preprint arXiv:1705.10528, 2017.  
Kavosh Asadi, Dipendra Misra, and Michael L Littman. Lipschitz continuity in model-based reinforcement learning. arXiv preprint arXiv:1804.07193, 2018.  
Peter L Bartlett and Ambuj Tewari. Regal: A regularization based algorithm for reinforcement learning in weakly communicating mdps. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, pp. 35-42. AUAI Press, 2009.  
Ross Boczar, Nikolai Matni, and Benjamin Recht. Finite-data performance guarantees for the output-feedback control of an unknown system. arXiv preprint arXiv:1803.09186, 2018.  
J. Buckman, D. Hafner, G. Tucker, E. Brevdo, and H. Lee. Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion. *ArXive* -prints, July 2018.  
Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. arXiv preprint arXiv:1805.12114, 2018.  
Ignasi Clavera, Jonas Rothfuss, John Schulman, Yasuhiro Fujita, Tamim Asfour, and Pieter Abbeel. Model-based reinforcement learning via meta-policy optimization. arXiv preprint arXiv:1809.05214, 2018.  
Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2012.  
Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems, pp. 5713-5723, 2017.  
Sarah Dean, Horia Mania, Nikolai Matni, Benjamin Recht, and Stephen Tu. On the sample complexity of the linear quadratic regulator. arXiv preprint arXiv:1710.01688, 2017.  
Sarah Dean, Horia Mania, Nikolai Matni, Benjamin Recht, and Stephen Tu. Regret bounds for robust adaptive control of the linear quadratic regulator. arXiv preprint arXiv:1805.09388, 2018.  
Marc Deisenroth and Carl E Rasmussen. *Pilco: A model-based and data-efficient approach to policy search*. In Proceedings of the 28th International Conference on machine learning (ICML-11), pp. 465–472, 2011a.  
Marc Deisenroth and Carl E Rasmussen. *Pilco: A model-based and data-efficient approach to policy search*. In Proceedings of the 28th International Conference on machine learning (ICML-11), pp. 465–472, 2011b.  
Marc Peter Deisenroth, Carl Edward Rasmussen, and Dieter Fox. Learning to control a low-cost manipulator using data-efficient reinforcement learning. 2011.  
Marc Peter Deisenroth, Gerhard Neumann, Jan Peters, et al. A survey on policy search for robotics. Foundations and Trends in Robotics, 2(1-2):1-142, 2013.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning, pp. 1329-1338, 2016.  
Amir-massoud Farahmand, Andre Barreto, and Daniel Nikovski. Value-aware loss function for model-based reinforcement learning. In Artificial Intelligence and Statistics, pp. 1486-1494, 2017.  
Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I Jordan, Joseph E Gonzalez, and Sergey Levine. Model-based value estimation for efficient model-free reinforcement learning. arXiv preprint arXiv:1803.00101, 2018.  
Ronan Fruit, Matteo Pirotta, Alessandro Lazaric, and Ronald Ortner. Efficient bias-span-constrained exploration-exploitation in reinforcement learning. arXiv preprint arXiv:1802.04020, 2018.  
Shixiang Gu, Timothy Lillicrap, Ilya Sutskever, and Sergey Levine. Continuous deep q-learning with model-based acceleration. In International Conference on Machine Learning, pp. 2829-2838, 2016.  
David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.

Karl Hinderer. Lipschitz continuity of value functions in markovian decision processes. Mathematical Methods of Operations Research, 62(1):3-22, 2005.  
K Jetal Hunt, D Sbarbaro, R Zbikowski, and Peter J Gawthrop. Neural networks for control systems—a survey. Automatica, 28(6):1083-1112, 1992.  
Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(Apr):1563-1600, 2010.  
S. Kakade, M. Wang, and L. F. Yang. Variance Reduction Methods for Sublinear Reinforcement Learning. ArXiv e-prints, February 2018.  
Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In ICML, volume 2, pp. 267-274, 2002.  
Gabriel Kalweit and Joschka Boedecker. Uncertainty-driven imagination for continuous deep reinforcement learning. In Conference on Robot Learning, pp. 195-206, 2017.  
Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine learning, 49(2-3):209-232, 2002.  
S Mohammad Khansari-Zadeh and Aude Billard. Learning stable nonlinear dynamical systems with gaussian mixture models. IEEE Transactions on Robotics, 27(5):943-957, 2011.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Jonathan Ko and Dieter Fox. Gp-bayesfilters: Bayesian filtering using gaussian process prediction and observation models. Autonomous Robots, 27(1):75-90, 2009.  
Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble trust-region policy optimization. arXiv preprint arXiv:1802.10592, 2018.  
Kailasam Lakshmanan, Ronald Ortner, and Daniil Ryabko. Improved regret bounds for undiscounted continuous reinforcement learning. In International Conference on Machine Learning, pp. 524-532, 2015.  
Sergey Levine and Pieter Abbeel. Learning neural network policies with guided policy search under unknown dynamics. In Advances in Neural Information Processing Systems, pp. 1071-1079, 2014.  
Sergey Levine and Vladlen Koltun. Guided policy search. In International Conference on Machine Learning, pp. 1-9, 2013.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334-1373, 2016.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
Rudolf Lioutikov, Alexandros Paraschos, Jan Peters, and Gerhard Neumann. Sample-based information-theoretic stochastic optimal control. In Robotics and Automation (ICRA), 2014 IEEE International Conference on, pp. 3896-3902. IEEE, 2014.  
Horia Mania, Aurelia Guy, and Benjamin Recht. Simple random search provides a competitive approach to reinforcement learning. arXiv preprint arXiv:1803.07055, 2018.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pp. 1928-1937, 2016.  
Teodor Mihai Moldovan, Sergey Levine, Michael I Jordan, and Pieter Abbeel. Optimism-driven exploration for nonlinear systems. In Robotics and Automation (ICRA), 2015 IEEE International Conference on, pp. 3239-3246. IEEE, 2015.  
Igor Mordatch, Nikhil Mishra, Clemens Eppner, and Pieter Abbeel. Combining model-based policy search with online model learning for control of physical humanoids. In *Robotics and Automation(ICRA)*, 2016 IEEE International Conference on, pp. 242-248. IEEE, 2016.

Jun Morimoto and Christopher G Atkeson. Minimax differential dynamic programming: An application to robust biped walking. In Advances in neural information processing systems, pp. 1563-1570, 2003.  
Anusha Nagabandi, Gregory Kahn, Ronald S Fearing, and Sergey Levine. Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. arXiv preprint arXiv:1708.02596, 2017.  
Frank Nielsen and Richard Nock. On the chi square and higher-order chi distances for approximating f-divergences. IEEE Signal Processing Letters, 21(1):10-13, 2014.  
Junhyuk Oh, Satinder Singh, and Honglak Lee. Value prediction network. In Advances in Neural Information-Processing Systems, pp. 6118-6128, 2017.  
Razvan Pascanu, Yujia Li, Oriol Vinyals, Nicolas Heess, Lars Buesing, Sebastien Racanière, David Reichert, Théophane Weber, Daan Wierstra, and Peter Battaglia. Learning model-based planning from scratch. arXiv preprint arXiv:1707.06170, 2017.  
Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Adaptive step-size for policy gradient methods. In Advances in Neural Information Processing Systems, pp. 1394-1402, 2013.  
Matteo Pirotta, Marcello Restelli, and Luca Bascetta. Policy gradient in lipschitz markov decision processes. Machine Learning, 100(2-3):255-283, 2015.  
Vitchyr Pong, Shixiang Gu, Murtaza Dalal, and Sergey Levine. Temporal difference models: Model-free deep rl for model-based control. arXiv preprint arXiv:1802.09081, 2018a.  
Vitchyr Pong, Shixiang Gu, Murtaza Dalal, and Sergey Levine. Temporal difference models: Model-free deep rl for model-based control. International Conference on Learning Representations, 2018b.  
Sebastien Racanière, Théophane Weber, David Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adrià Puigdomènech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, et al. Imagination-augmented agents for deep reinforcement learning. In AdvancesinNeuralInformationProcessingSystems, pp. 5690-5701, 2017.  
Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, and Peter Battaglia. Graph networks as learnable physics engines for inference and control. arXiv preprint arXiv:1806.01242, 2018.  
Igal Sason and Sergio Verdu.  $f$ -divergence inequalities. IEEE Transactions on Information Theory, 62(11): 5973-6006, 2016.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning, pp. 1889-1897, 2015a.  
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015b.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Iulian Vlad Serban, Chinnadhurai Sankar, Michael Pieper, Joelle Pineau, and Yoshua Bengio. The bottleneck simulator: A model-based deep reinforcement learning approach. arXiv preprint arXiv:1807.04723, 2018.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 550(7676):354, 2017.  
Max Simchowitz, Horia Mania, Stephen Tu, Michael I Jordan, and Benjamin Recht. Learning without mixing: Towards a sharp analysis of linear system identification. arXiv preprint arXiv:1802.08334, 2018.  
Aravind Srinivas, Allan Jabri, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Universal planning networks. arXiv preprint arXiv:1804.00645, 2018.  
Richard S Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In Machine Learning Proceedings 1990, pp. 216-224. Elsevier, 1990.  
Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM SIGART Bulletin, 2(4):160-163, 1991.  
Richard S Sutton, Csaba Szepesvári, Alborz Geramifard, and Michael P Bowling. Dyna-style planning with linear function approximation and prioritized sweeping. arXiv preprint arXiv:1206.3285, 2012.

István Szita and Csaba Szepesvári. Model-based reinforcement learning with nearly tight exploration complexity bounds. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pp. 1031-1038, 2010.  
Aviv Tamar, Dotan Di Castro, and Ron Meir. Integrating a partial model into model free reinforcement learning. Journal of Machine Learning Research, 13(Jun):1927-1966, 2012.  
Voot Tangkaratt, Syogo Mori, Tingting Zhao, Jun Morimoto, and Masashi Sugiyama. Model-based policy gradients with parameter-based exploration by least-squares conditional density estimation. Neural networks, 57:128-140, 2014.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026-5033. IEEE, 2012.  
Ronald J Williams and Jing Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241-268, 1991.  
Chris Xie, Sachin Patil, Teodor Moldovan, Sergey Levine, and Pieter Abbeel. Model-based reinforcement learning with parametrized physical models and optimism-driven exploration. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pp. 504-511. IEEE, 2016.  
Michael C Yip and David B Camarillo. Model-less feedback control of continuum manipulators in constrained environments. IEEE Transactions on Robotics, 30(4):880-889, 2014.
