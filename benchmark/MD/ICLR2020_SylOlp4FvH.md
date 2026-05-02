# V-MPO: ON-POLICY MAXIMUM A POSTERIOR POLICY OPTIMIZATION FOR DISCRETE AND CONTINUOUS CONTROL

Anonymous authors

Paper under double-blind review

# ABSTRACT

Some of the most successful applications of deep reinforcement learning to challenging domains in discrete and continuous control have used policy gradient methods in the on-policy setting. However, policy gradients can suffer from large variance that may limit performance, and in practice require carefully tuned entropy regularization to prevent policy collapse. As an alternative to policy gradient algorithms, we introduce V-MPO, an on-policy adaptation of Maximum a Posteriori Policy Optimization (MPO) that performs policy iteration based on a learned state-value function. We show that V-MPO surpasses previously reported scores for both the Atari-57 and DMLab-30 benchmark suites in the multi-task setting, and does so reliably without importance weighting, entropy regularization, or population-based tuning of hyperparameters. On individual DMLab and Atari levels, the proposed algorithm can achieve scores that are substantially higher than has previously been reported. V-MPO is also applicable to problems with high-dimensional, continuous action spaces, which we demonstrate in the context of learning to control simulated humanoids with 22 degrees of freedom from full state observations and 56 degrees of freedom from pixel observations, as well as example OpenAI Gym tasks where V-MPO achieves substantially higher asymptotic scores than previously reported.

# 1 INTRODUCTION

Deep reinforcement learning (RL) with neural network function approximators has achieved superhuman performance in several challenging domains (Mnih et al., 2015; Silver et al., 2016; 2018). Some of the most successful recent applications of deep RL to difficult environments such as Dota 2 (OpenAI, 2018a), Capture the Flag (Jaderberg et al., 2019), Starcraft II (DeepMind, 2019), and dexterous object manipulation (OpenAI, 2018b) have used policy gradient-based methods such as Proximal Policy Optimization (PPO) (Schulman et al., 2017) and the Importance-Weighted Actor-Learner Architecture (IMPALA) (Espeholt et al., 2018), both in the approximately on-policy setting.

Policy gradients, however, can suffer from large variance that may limit performance, especially for high-dimensional action spaces (Wu et al., 2018). In practice, moreover, policy gradient methods typically employ carefully tuned entropy regularization in order to prevent policy collapse. As an alternative to policy gradient-based algorithms, in this work we introduce an approximate policy iteration algorithm that adapts Maximum a Posteriori Policy Optimization (MPO) (Abdolmaleki et al., 2018a;b) to the on-policy setting. The modified algorithm, V-MPO, relies on a learned state-value function  $V(s)$  instead of the state-action value function used in MPO. Like MPO, rather than directly updating the parameters in the direction of the policy gradient, V-MPO first constructs a target distribution for the policy update subject to a sample-based KL constraint, then calculates the gradient that partially moves the parameters toward that target, again subject to a KL constraint.

As we are particularly interested in scalable RL algorithms that can be applied to multi-task settings where a single agent must perform a wide variety of tasks, we show for the case of discrete actions that the proposed algorithm surpasses previously reported performance in the multi-task setting for both the Atari-57 (Bellemare et al., 2012) and DMLab-30 (Beattie et al., 2016) benchmark suites, and does so reliably without population-based tuning of hyperparameters (Jaderberg et al., 2017a).

For a few individual levels in DMLab and Atari we also show that V-MPO can achieve scores that are substantially higher than has previously been reported, especially in the challenging Ms. Pacman.

V-MPO is also applicable to problems with high-dimensional, continuous action spaces. We demonstrate this in the context of learning to control both a 22-dimensional simulated humanoid from full state observations—where V-MPO reliably achieves higher asymptotic performance than previous algorithms—and a 56-dimensional simulated humanoid from pixel observations (Tassa et al., 2018; Merel et al., 2019). In addition, for several OpenAI Gym tasks (Brockman et al., 2016) we show that V-MPO achieves higher asymptotic performance than has previously been reported.

# 2 BACKGROUND AND SETTING

We consider the discounted RL setting, where we seek to optimize a policy  $\pi$  for a Markov Decision Process described by states  $s$ , actions  $a$ , initial state distribution  $\rho_0^{\mathrm{env}}(s_0)$ , transition probabilities  $\mathcal{P}^{\mathrm{env}}(s_{t + 1}|s_t,a_t)$ , reward function  $r(s_{t},a_{t})$ , and discount factor  $\gamma \in (0,1)$ . In deep RL, the policy  $\pi_{\theta}(a_{t}|s_{t})$ , which specifies the probability that the agent takes action  $a_{t}$  in state  $s_t$  at time  $t$ , is described by a neural network with parameters  $\theta$ . We consider problems where both the states  $s$  and actions  $a$  may be discrete or continuous. Two functions play a central role in RL: the state-value function  $V^{\pi}(s_t) = \mathbb{E}_{a_t,s_{t + 1},a_{t + 1},\ldots}\left[\sum_{k = 0}^{\infty}\gamma^k r(s_{t + k},a_{t + k})\right]$  and the state-action value function  $Q^{\pi}(s_t,a_t) = \mathbb{E}_{s_{t + 1},a_{t + 1},\ldots}\left[\sum_{k = 0}^{\infty}\gamma^k r(s_{t + k},a_{t + k})\right] = r(s_t,a_t) + \gamma \mathbb{E}_{s_{t + 1}}\left[V^{\pi}(s_{t + 1})\right]$ , where  $s_0\sim \rho_0^{\mathrm{env}}(s_0)$ ,  $a_{t}\sim \pi (a_{t}|s_{t})$ , and  $s_{t + 1}\sim \mathcal{P}^{\mathrm{env}}(s_{t + 1}|s_{t},a_{t})$ .

In the usual formulation of the RL problem, the goal is to find a policy  $\pi$  that maximizes the expected return given by  $J(\pi) = \mathbb{E}_{s_0,a_0,s_1,a_1,\dots}\left[\sum_{t = 0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right]$ . In policy gradient algorithms (Williams, 1992; Sutton et al., 2000; Mnih et al., 2016), for example, this objective is directly optimized by estimating the gradient of the expected return. An alternative approach to finding optimal policies derives from research that treats RL as a problem in probabilistic inference, including Maximum a Posteriori Policy Optimization (MPO) (Levine, 2018; Abdelmaleki et al., 2018a;b). Here our objective is subtly different, namely, given a suitable criterion for what are good actions to take in a certain state, how do we find a policy that achieves this goal?

As was the case for the original MPO algorithm, the following derivation is valid for any such criterion. However, the policy improvement theorem (Sutton & Barto, 1998) tells us that a policy update performed by exact policy iteration,  $\pi(s) = \arg \max_{a} [Q^{\pi}(s, a) - V^{\pi}(s)]$ , can improve the policy if there is at least one state-action pair with a positive advantage and nonzero probability of visiting the state. Motivated by this classic result, in this work we specifically choose an exponential function of the advantages  $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$ .

Notation. In the following we use  $\sum_{s,a}$  to indicate both discrete and continuous sums (i.e., integrals) over states  $s$  and actions  $a$  depending on the setting. A sum with indices only, such as  $\sum_{s,a}$ , denotes a sum over all possible states and actions, while  $\sum_{s,a\sim \mathcal{D}}$ , for example, denotes a sum over sample states and actions from a batch of trajectories (the "dataset")  $\mathcal{D}$ .

# 3 RELATED WORK

V-MPO shares many similarities, and thus relevant related work, with the original MPO algorithm (Abdolmaleki et al., 2018a;b). In particular, the general idea of using KL constraints to limit the size of policy updates is present in both Trust Region Policy Optimization (TRPO; Schulman et al., 2015) and Proximal Policy Optimization (PPO) (Schulman et al., 2017); we note, however, that this corresponds to the E-step constraint in V-MPO. Meanwhile, the introduction of the M-step KL constraint and the use of top- $k$  advantages distinguishes V-MPO from Relative Entropy Policy Search (REPS) (Peters et al., 2008). Interestingly, previous attempts to use REPS with neural network function approximators reported very poor performance, being particularly prone to local optima (Duan et al., 2016). In contrast, we find that the principles of EM-style policy optimization, when combined with appropriate constraints, can reliably train powerful neural networks, including transformers, for RL tasks.

Like V-MPO, Supervised Policy Update (SPU) (Vuong et al., 2019) seeks to exactly solve an optimization problem and fit the parametric policy to this solution. As we argue in Appendix D,

however, SPU uses this nonparametric distribution quite differently from V-MPO; as a result, the final algorithm is closer to a policy gradient algorithm such as PPO.

# 4 METHOD

V-MPO is an approximate policy iteration (Sutton & Barto, 1998) algorithm with a specific prescription for the policy improvement step. In general, policy iteration uses the fact that the true state-value function  $V^{\pi}$  corresponding to policy  $\pi$  can be used to obtain an improved policy  $\pi'$ . Thus we can

1. Generate trajectories  $\tau$  from an old "target" policy  $\pi_{\theta_{\mathrm{old}}}(a|s)$  whose parameters  $\theta_{\mathrm{old}}$  are fixed. To control the amount of data generated by a particular policy, we use a target network which is fixed for  $T_{\mathrm{target}}$  learning steps (Fig. 5a in the Appendix).  
2. Evaluate the policy  $\pi_{\theta_{\mathrm{old}}}(a|s)$  by learning the value function  $V^{\pi_{\theta_{\mathrm{old}}}}(s)$  from empirical returns and estimating the corresponding advantages  $A^{\pi_{\theta_{\mathrm{old}}}}(s,a)$  for the actions that were taken.  
3. Estimate an improved "online" policy  $\pi_{\theta}(a|s)$  based on  $A^{\pi_{\theta_{\mathrm{old}}}}(s,a)$ .

The first two steps are standard, and describing V-MPO's approach to step (3) is the essential contribution of this work. At a high level, our strategy is to first construct a nonparametric target distribution for the policy update, then partially move the parametric policy towards this distribution subject to a KL constraint. Ultimately, we use gradient descent to optimize a single, relatively simple loss, which we provide here in complete form in order to ground the derivation of the algorithm.

Consider a batch of data  $\mathcal{D}$  consisting of a number of trajectories, with  $|\mathcal{D}|$  total state-action samples. Each trajectory consists of an unroll of length  $n$  of the form  $\tau = [(s_t,a_t,r_{t + 1}),\dots ,(s_{t + n - 1},a_{t + n - 1},r_{t + n})$ $s_{t + n}]$  including the bootstrapped state  $s_{t + n}$ , where  $r_{t + 1} = r(s_t,a_t)$ . The total loss is the sum of a policy evaluation loss and a policy improvement loss,

$$
\mathcal {L} (\phi , \theta , \eta , \alpha) = \mathcal {L} _ {V} (\phi) + \mathcal {L} _ {\mathrm {V - M P O}} (\theta , \eta , \alpha), \tag {1}
$$

where  $\phi$  are the parameters of the value network,  $\theta$  the parameters of the policy network, and  $\eta$  and  $\alpha$  are Lagrange multipliers. In practice, the policy and value networks share most of their parameters in the form of a shared convolutional network (a ResNet) and recurrent LSTM core, and are optimized together (Fig. 5b in the Appendix) (Mnih et al., 2016). We note, however, that the value network parameters  $\phi$  are considered fixed for the policy improvement loss, and gradients are not propagated.

The policy evaluation loss for the value function,  $\mathcal{L}_V(\phi)$ , is the standard regression to  $n$ -step returns and is given by Eq. 6 below. The policy improvement loss  $\mathcal{L}_{\mathrm{V - MPO}}(\theta ,\eta ,\alpha)$  is given by

$$
\mathcal {L} _ {\mathrm {V - M P O}} (\theta , \eta , \alpha) = \mathcal {L} _ {\pi} (\theta) + \mathcal {L} _ {\eta} (\eta) + \mathcal {L} _ {\alpha} (\theta , \alpha). \tag {2}
$$

Here the policy loss is the weighted maximum likelihood loss

$$
\mathcal {L} _ {\pi} (\theta) = - \sum_ {s, a \sim \tilde {\mathcal {D}}} \psi (s, a) \log \pi_ {\theta} (a | s), \quad \psi (s, a) = \frac {\exp \left(\frac {A ^ {\text {t a r g e t}} (s , a)}{\eta}\right)}{\sum_ {s , a \sim \tilde {\mathcal {D}}} \exp \left(\frac {A ^ {\text {t a r g e t}} (s , a)}{\eta}\right)}, \tag {3}
$$

where the advantages  $A^{\mathrm{target}}(s,a)$  for the target network policy  $\pi_{\theta_{\mathrm{target}}}(a|s)$  are estimated according to the standard method described below. The tilde over the dataset,  $\tilde{D}$ , indicates that we take samples corresponding to the top half advantages in the batch of data. The  $\eta$ , or "temperature", loss is

$$
\mathcal {L} _ {\eta} (\eta) = \eta \epsilon_ {\eta} + \eta \log \left[ \frac {1}{| \tilde {\mathcal {D}} |} \sum_ {s, a \sim \tilde {\mathcal {D}}} \exp \left(\frac {A ^ {\text {t a r g e t}} (s , a)}{\eta}\right) \right]. \tag {4}
$$

The KL constraint, which can be viewed as a form of trust-region loss, is given by

$$
\mathcal {L} _ {\alpha} (\theta , \alpha) = \frac {1}{| \mathcal {D} |} \sum_ {s \in \mathcal {D}} \left[ \alpha \left(\epsilon_ {\alpha} - \operatorname {s g} \left[ \left[ D _ {\mathrm {K L}} \left(\pi_ {\theta_ {\text {t a r g e t}}} (a | s) \| \pi_ {\theta} (a | s)\right) \right] \right]\right) + \operatorname {s g} [ [ \alpha ] ] D _ {\mathrm {K L}} \left(\pi_ {\theta_ {\text {t a r g e t}}} (a | s) \| \pi_ {\theta} (a | s)\right) \right], \tag {5}
$$

where  $\mathrm{sg}[[\cdot]]$  indicates a stop gradient, i.e., that the enclosed term is assumed constant with respect to all variables. Note that here we use the full batch  $\mathcal{D}$ , not  $\hat{\mathcal{D}}$ .

We used the Adam optimizer (Kingma & Ba, 2015) with default TensorFlow hyperparameters to optimize the total loss in Eq. 1. In particular, the learning rate was fixed at  $10^{-4}$  for all experiments.

# 4.1 POLICY EVALUATION

In the present setting, policy evaluation means learning an approximate state-value function  $V^{\pi}(s)$  given a policy  $\pi(a|s)$ , which we keep fixed for  $T_{\mathrm{target}}$  learning steps (i.e., batches of trajectories). We note that the value function corresponding to the target policy is instantiated in the "online" network receiving gradient updates; bootstrapping uses the online value function, as it is the best available estimate of the value function for the target policy. Thus in this section  $\pi$  refers to  $\pi_{\theta_{\mathrm{old}}}$ , while the value function update is performed on the current  $\phi$ , which may share parameters with the current  $\theta$ .

We fit a parametric value function  $V_{\phi}^{\pi}(s)$  with parameters  $\phi$  by minimizing the squared loss

$$
\mathcal {L} _ {V} (\phi) = \frac {1}{2 | \mathcal {D} |} \sum_ {s _ {t} \sim \mathcal {D}} \left(V _ {\phi} ^ {\pi} \left(s _ {t}\right) - G _ {t} ^ {(n)}\right) ^ {2}, \tag {6}
$$

where  $G_{t}^{(n)}$  is the standard  $n$ -step target for the value function at state  $s_t$  at time  $t$  (Sutton & Barto, 1998). This return uses the actual rewards in the trajectory and bootstraps from the value function for the rest: for each  $\ell = t,\dots ,t + n - 1$  in an unroll,  $G_{\ell}^{(n)} = \sum_{k = \ell}^{t + n - 1}\gamma^{k - \ell}r_k + \gamma^{t + n - \ell}V_{\phi}^{\pi}(s_{t + n})$ . The advantages, which are the key quantity of interest for the policy improvement step in V-MPO, are then given by  $A^{\pi}(s_t,a_t) = G_t^{(n)} - V_\phi^\pi (s_t)$  for each  $s_t,a_t$  in the batch of trajectories.

PopArt normalization. As we are interested in the multi-task setting where a single agent must learn a large number of tasks with differing reward scales, we used PopArt (van Hasselt et al., 2016; Hessel et al., 2018) for the value function, even when training on a single task. Specifically, the value function outputs a separate value for each task in normalized space, which is converted to actual returns by a shift and scaling operation, the statistics of which are learned during training. We used a scale lower bound of  $10^{-2}$ , scale upper bound of  $10^{6}$ , and learning rate of  $10^{-4}$  for the statistics. The lower bound guards against numerical issues when rewards are extremely sparse.

Importance-weighting for off-policy data. It is possible to importance-weight the samples using V-trace to correct for off-policy data (Espeholt et al., 2018), for example when data is taken from a replay buffer. For simplicity, however, no importance-weighting was used for the experiments presented in this work, which were mostly on-policy.

# 4.2 POLICY IMPROVEMENT IN V-MPO

In this section we show how, given the advantage function  $A^{\pi_{\theta_{\mathrm{old}}}}(s,a)$  for the state-action distribution  $p_{\theta_{\mathrm{old}}}(s,a) = \pi_{\theta_{\mathrm{old}}}(a|s)p(s)$  induced by the old policy  $\pi_{\theta_{\mathrm{old}}}(a|s)$ , we can estimate an improved policy  $\pi_{\theta}(a|s)$ . More formally, let  $\mathcal{I}$  denote the binary event that the new policy is an improvement (in a sense to be defined below) over the previous policy:  $\mathcal{I} = 1$  if the policy is successfully improved and 0 otherwise. Then we would like to find the mode of the posterior distribution over parameters  $\theta$  conditioned on this event, i.e., we seek the maximum a posteriori (MAP) estimate

$$
\theta^ {*} = \arg \max  _ {\theta} \left[ \log p _ {\theta} (\mathcal {I} = 1) + \log p (\theta) \right], \tag {7}
$$

where we have written  $p(\mathcal{I} = 1|\theta)$  as  $p_{\theta}(\mathcal{I} = 1)$  to emphasize the parametric nature of the dependence on  $\theta$ . We use the well-known identity  $\log p(X) = \mathbb{E}_{\psi(Z)}\left[\log \frac{p(X,Z)}{\psi(Z)}\right] + D_{\mathrm{KL}}\big(\psi(Z)\| p(Z|X)\big)$  for any latent distribution  $\psi(Z)$ , where  $D_{\mathrm{KL}}(\psi(Z)\| p(Z|X))$  is the Kullback-Leibler divergence between  $\psi(Z)$  and  $p(Z|X)$  with respect to  $Z$ , and the first term is a lower bound because the KL divergence is always non-negative. Then considering  $s, a$  as latent variables,

$$
\log p _ {\theta} (\mathcal {I} = 1) = \sum_ {s, a} \psi (s, a) \log \frac {p _ {\theta} (\mathcal {I} = 1 , s , a)}{\psi (s , a)} + D _ {\mathrm {K L}} \left(\psi (s, a) \| p _ {\theta} (s, a | \mathcal {I} = 1)\right). \tag {8}
$$

Policy improvement in V-MPO consists of the following two steps which have direct correspondences to the expectation maximization (EM) algorithm (Neal & Hinton, 1998): In the expectation (E) step, we choose the variational distribution  $\psi(s, a)$  such that the lower bound on  $\log p_{\theta}(\mathcal{I} = 1)$  is as tight as possible, by minimizing the KL term. In the maximization (M) step we then find parameters  $\theta$  that maximize the corresponding lower bound, together with the prior term in Eq. 7.

# 4.2.1 E-STEP

In the E-step, our goal is to choose the variational distribution  $\psi(s,a)$  such that the lower bound on  $\log p_{\theta}(\mathcal{I} = 1)$  is as tight as possible, which is the case when the KL term in Eq. 8 is zero. Given the old parameters  $\theta_{\mathrm{old}}$ , this simply leads to  $\psi(s,a) = p_{\theta_{\mathrm{old}}}(s,a|\mathcal{I} = 1)$ , or

$$
\psi (s, a) = \frac {p _ {\theta_ {\text {o l d}}} (s , a) p _ {\theta_ {\text {o l d}}} (\mathcal {I} = 1 | s , a)}{p _ {\theta_ {\text {o l d}}} (\mathcal {I} = 1)}, \quad p _ {\theta_ {\text {o l d}}} (\mathcal {I} = 1) = \sum_ {s, a} p _ {\theta_ {\text {o l d}}} (s, a) p _ {\theta_ {\text {o l d}}} (\mathcal {I} = 1 | s, a). \tag {9}
$$

Intuitively, this solution weights the probability of each state-action pair with its relative improvement probability  $p_{\theta_{\mathrm{old}}}(\mathcal{I} = 1|s,a)$ . We now choose a distribution  $p_{\theta_{\mathrm{old}}}(\mathcal{I} = 1|s,a)$  that leads to our desired outcome. As we prefer actions that lead to a higher advantage in each state, we suppose that this probability is given by

$$
p _ {\theta_ {\text {o l d}}} (\mathcal {I} = 1 | s, a) \propto \exp \left(\frac {A ^ {\pi_ {\theta_ {\text {o l d}}}} (s , a)}{\eta}\right) \tag {10}
$$

for some temperature  $\eta > 0$ , from which we obtain the equation on the right in Eq. 3. This probability depends on the old parameters  $\theta_{\mathrm{old}}$  and not on the new parameters  $\theta$ . Meanwhile, the value of  $\eta$  allows us to control the diversity of actions that contribute to the weighting, but at the moment is arbitrary. It turns out, however, that we can tune  $\eta$  as part of the optimization, which is desirable since the optimal value of  $\eta$  changes across iterations. The convex loss that achieves this, Eq. 4, is derived in Appendix A by minimizing the KL term in Eq. 8 subject to a hard constraint on  $\psi(s, a)$ .

Top- $k$  advantages. We found that learning improves substantially if we take only the samples corresponding to the highest  $50\%$  of advantages in each batch for the E-step, corresponding to the use of  $\tilde{\mathcal{D}}$  rather than  $\mathcal{D}$  in Eqs. 3, 4. Importantly, these must be consistent between the maximum likelihood weights in Eq. 3 and the temperature loss in Eq. 4, since, mathematically, this is justified by choosing the corresponding policy improvement probability in Eq. 10 to only use the top half of the advantages. This is similar to the technique used in Covariance Matrix Adaptation - Evolutionary Strategy (CMA-ES) (Hansen et al., 1997; Abdelmaleki et al., 2017), and is a special case of the more general feature that any rank-preserving transformation is allowed under this formalism.

Importance weighting for off-policy corrections. As for the value function, importance weights can be used in the policy improvement step to correct for off-policy data. While not used for the experiments presented in this work, details for how to carry out this correction are given in Appendix E.

# 4.2.2 M-STEP: CONSTRAINED SUPERVISED LEARNING OF THE PARAMETRIC POLICY

In the E-step we found the nonparametric variational state-action distribution  $\psi (s,a)$ , Eq. 9, that gives the tightest lower bound to  $p_{\theta}(\mathcal{I} = 1)$  in Eq. 8. In the M-step we maximize this lower bound together with the prior term  $\log p(\theta)$  with respect to the parameters  $\theta$ , which effectively leads to a constrained weighted maximum likelihood problem. Thus the introduction of the nonparametric distribution in Eq. 9 separates the RL procedure from the neural network fitting.

We would like to find new parameters  $\theta$  that minimize

$$
\mathcal {L} (\theta) = - \sum_ {s, a} \psi (s, a) \log \frac {p _ {\theta} (\mathcal {I} = 1 , s , a)}{\psi (s , a)} - \log p (\theta). \tag {11}
$$

Note, however, that so far we have worked with the joint state-action distribution  $\psi(s,a)$  while we are in fact optimizing for the policy, which is the conditional distribution  $\pi_{\theta}(a|s)$ . Writing  $p_{\theta}(s,a) = \pi_{\theta}(a|s)p(s)$  since only the policy is parametrized by  $\theta$  and dropping terms that are not parametrized by  $\theta$ , the first term of Eq. 11 is seen to be the weighted maximum likelihood policy loss

$$
\mathcal {L} _ {\pi} (\theta) = - \sum_ {s, a} \psi (s, a) \log \pi_ {\theta} (a | s). \tag {12}
$$

In the sample-based computation of this loss, we assume that any state-action pairs not in the batch of trajectories have zero weight, leading to the normalization in Eq. 3.

As in the original MPO algorithm, a useful prior is to keep the new policy  $\pi_{\theta}(a|s)$  close to the old policy  $\pi_{\theta_{\mathrm{old}}}(a|s)$ :  $\log p(\theta) \approx -\alpha \mathbb{E}_{s \sim p(s)}\left[D_{\mathrm{KL}}\big(\pi_{\theta_{\mathrm{old}}}(a|s) \big|\big|\pi_{\theta}(a|s)\big)\right]$ . While intuitive, we motivate

this more formally in Appendix B. It is again more convenient to specify a bound on the KL divergence instead of tuning  $\alpha$  directly, so we solve the constrained optimization problem

$$
\theta^ {*} = \arg \min  _ {\theta} - \sum_ {s, a} \psi (s, a) \log \pi_ {\theta} (a | s) \quad \text {s . t .} \underset {s \sim p (s)} {\mathbb {E}} \left[ D _ {\mathrm {K L}} \left(\pi_ {\theta_ {\mathrm {o l d}}} (a | s) \| \pi_ {\theta} (a | s)\right) \right] <   \epsilon_ {\alpha}. \tag {13}
$$

Intuitively, the constraint in the E-step expressed by Eq. 19 in Appendix A for tuning the temperature only constrains the nonparametric distribution; it is the constraint in Eq. 13 that directly limits the change in the parametric policy, in particular for states and actions that were not in the batch of samples and which rely on the generalization capabilities of the neural network function approximator.

To make the constrained optimization problem amenable to gradient descent, we use Lagrangian relaxation to write the unconstrained objective as

$$
\mathcal {J} (\theta , \alpha) = \mathcal {L} _ {\pi} (\theta) + \alpha \left(\epsilon_ {\alpha} - \underset {s \sim p (s)} {\mathbb {E}} \left[ D _ {\mathrm {K L}} \left(\pi_ {\theta_ {\text {o l d}}} (a | s) \| \pi_ {\theta} (a | s)\right) \right]\right), \tag {14}
$$

which we can optimize by following a coordinate-descent strategy, alternating between the optimization over  $\theta$  and  $\alpha$ . Thus, in addition to the policy loss we arrive at the constraint loss

$$
\mathcal {L} _ {\alpha} (\theta , \alpha) = \alpha \left(\epsilon_ {\alpha} - \underset {s \sim p (s)} {\mathbb {E}} \left[ \operatorname {s g} \left[ \left[ D _ {\mathrm {K L}} \left(\pi_ {\theta_ {\mathrm {o l d}}} \| \pi_ {\theta}\right) \right] \right] \right]\right) + \operatorname {s g} [ [ \alpha ] ] \underset {s \sim p (s)} {\mathbb {E}} \left[ D _ {\mathrm {K L}} \left(\pi_ {\theta_ {\mathrm {o l d}}} \| \pi_ {\theta}\right) \right]. \tag {15}
$$

Replacing the sum over states with samples gives Eq. 5. Since  $\eta$  and  $\alpha$  are Lagrange multipliers that must be positive, after each gradient update we project the resulting  $\eta$  and  $\alpha$  to a small positive value which we choose to be  $\eta_{\mathrm{min}} = \alpha_{\mathrm{min}} = 10^{-8}$  throughout the results presented below.

For continuous action spaces parametrized by Gaussian distributions, we use decoupled KL constraints for the M-step in Eq. 15 as in Abdelmaleki et al. (2018b); the precise form is given in Appendix C.

# 5 EXPERIMENTS

Details on the network architecture and hyperparameters used for each task are given in Appendix F.

# 5.1 DISCRETE ACTIONS: DMLAB, ATARI

![](images/e55304f71db2d8de7ebba97c68c001d29ec85db7b744064da061d6c3e9252702.jpg)  
(a) Multi-task DMLab-30.

![](images/623d136d8deb080ea4da263ecc47a805b30e63d81e6a831cc2b359b9aa4892a7.jpg)  
(b) Multi-task Atari-57.  
Figure 1: (a) Multi-task DMLab-30. IMPALA results show 3 runs of 8 agents each; within a run hyperparameters were evolved via PBT. For V-MPO each line represents a set of hyperparameters that are fixed throughout training. The final result of R2D2+ trained for 10B environment steps on individual levels (Kapturowski et al., 2019) is also shown for comparison (orange line). (b) Multi-task Atari-57. In the IMPALA experiment, hyperparameters were evolved with PBT. For V-MPO each of the 24 lines represents a set of hyperparameters that were fixed throughout training, and all runs achieved a higher score than the best IMPALA run. Data for IMPALA ("Pixel-PopArt-IMPALA" for DMLab-30 and "PopArt-IMPALA" for Atari-57) was obtained from the authors of Hessel et al. (2018). Each environment frame corresponds to 4 agent steps due to the action repeat.

![](images/314ca0339697a493db96cc08c045016ea4e2c673dffaf1118d0ce26fe789a4be.jpg)  
Figure 2: Example levels from DMLab-30, compared to IMPALA and more recent results from R2D2+, the larger, DMLab-specific version of R2D2 (Kapturowski et al., 2019). The IMPALA results include hyperparameter evolution with PBT.

![](images/0f7cf677b13c2b265c3197b0fef695814bb4c740340113d2357a16f0cb3f31f5.jpg)

![](images/91ec305b2ac9efac11ba4617bf2f2141ea1ed58d553c9ace8631c3e70bf648df.jpg)

![](images/e3d6c25446f4e38113d5c2b89abba3b20e6395dc94945a82e29be082d0822e99.jpg)

![](images/3283943bbb5ee414321c8b0f168d92e7425d09c625a0c3dec63bdd701d4190fe.jpg)  
Figure 3: Example levels from Atari. In Breakout, V-MPO achieves the maximum score of 864 in every episode. No reward clipping was applied, and the maximum length of an episode was 30 minutes (108,000 frames). Supplementary video for Ms. Pacman: https://bit.ly/2lWQBy5

![](images/e810a3533e4457db8edf4a23391b1fb0f590022d15fce3650542b54a50ab8a6d.jpg)

![](images/e5124e93b8e366a26c3abbb14ae04d9302410ec6525e8a80d66f34ff11dd8e53.jpg)

![](images/fadfe6ab145ab39a78fd4a66521cfcc6627e6b4e889251314a63cc9ff413cbe0.jpg)

DMLab. DMLab-30 (Beattie et al., 2016) is a collection of visually rich, partially observable 3D environments played from the first-person point of view. Like IMPALA, for DMLab we used pixel control as an auxiliary loss for representation learning (Jaderberg et al., 2017b; Hessel et al., 2018). However, we did not employ the optimistic asymmetric reward scaling used by previous IMPALA experiments to aid exploration on a subset of the DMLab levels, by weighting positive rewards more than negative rewards (Espeholt et al., 2018; Hessel et al., 2018; Kapturowski et al., 2019). Unlike in Hessel et al. (2018) we also did not use population-based training (PBT) (Jaderberg et al., 2017a). Additional details for the settings used in DMLab can be found in Table 5 of the Appendix.

Fig. 1a shows the results for multi-task DMLab-30, comparing the V-MPO learning curves to data obtained from Hessel et al. (2018) for the PopArt IMPALA agent with pixel control. We note that the result for V-MPO at 10B environment frames across all levels matches the result for the Recurrent Replay Distributed DQN (R2D2) agent (Kapturowski et al., 2019) trained on individual levels for 10B environment steps per level. Fig. 2 shows example individual levels in DMLab where V-MPO achieves scores that are substantially higher than has previously been reported, for both R2D2 and IMPALA. The pixel-control IMPALA agents shown here were carefully tuned for DMLab and are similar to the "experts" used in Schmitt et al. (2018); in all cases these results match or exceed previously published results for IMPALA (Espeholt et al., 2018; Kapturowski et al., 2019).

Atari. The Atari Learning Environment (ALE) (Bellemare et al., 2012) is a collection of 57 Atari 2600 games that has served as an important benchmark for recent deep RL methods. We used the standard preprocessing scheme and a maximum episode length of 30 minutes (108,000 frames), see Table 6 in the Appendix. For the multi-task setting we followed Hessel et al. (2018) in setting the discount to zero on loss of life; for the example single tasks we did not employ this trick, since it can prevent the agent from achieving the highest score possible by sacrificing lives. Similarly, while in the multi-task setting we followed previous work in clipping the maximum reward to 1.0, no such clipping was applied in the single-task setting in order to preserve the original reward structure. Additional details for the settings used in Atari can be found in Table 6 in the Appendix.

Fig. 1b shows the results for multi-task Atari-57, demonstrating that it is possible for a single agent to achieve "superhuman" median performance on Atari-57 in approximately 4 billion ( $\sim 70$  million per level) environment frames.

![](images/25b5e9e293630fe82d04fc85400ca5df81e277ac3e60105de6a8567a670dab37.jpg)  
(a)

![](images/5fa692f982b4a7b88df859c81c3412998c5419e64366dc25bbd67730ed931bec.jpg)  
(b)

![](images/85bef559f7f028ce792dd2459b19db5c12790c2adc8ce58509d141c97110427b.jpg)  
(c)  
Figure 4: (a) Humanoid "run" from full state (Tassa et al., 2018) and (b) humanoid "gaps" from pixel observations (Merel et al., 2019). Purple curves are the same runs but without parametric KL constraints. Det. eval.: deterministic evaluation. Supplementary video for humanoid gaps: https://bit.ly/2L9KZdS. (c)-(d) Example OpenAI Gym tasks.

![](images/37e5558f22a4e4fbc614eaca53e63dd35926dc21ef4c99ad68e01adb54fda197.jpg)  
(d)

We also compare the performance of V-MPO on a few individual Atari levels to R2D2 (Kapturowski et al., 2019), which previously achieved some of the highest scores reported for Atari. Again, V-MPO can match or exceed previously reported scores while requiring fewer interactions with the environment. In Ms. Pacman, the final performance approaches 300,000 with a 30-minute timeout (and the maximum 1M without), effectively solving the game. Inspired by the argument in Kapturowski et al. (2019) that in a fully observable environment LSTMs enable the agent to utilize more useful representations than is available in the immediate observation, for the single-task setting we used a Transformer-XL (TrXL) (Dai et al., 2019) to replace the LSTM core. Unlike previous work for single Atari levels, we did not employ any reward clipping (Mnih et al., 2015; Espeholt et al., 2018) or nonlinear value function rescaling (Kapturowski et al., 2019).

# 5.2 CONTINUOUS CONTROL

To demonstrate V-MPO's effectiveness in high-dimensional, continuous action spaces, here we present examples of learning to control both a simulated humanoid with 22 degrees of freedom from full state observations and one with 56 degrees of freedom from pixel observations (Tassa et al., 2018; Merel et al., 2019). As shown in Fig. 4a, for the 22-dimensional humanoid V-MPO reliably achieves higher asymptotic returns than has previously been reported, including for Deep Deterministic Policy Gradients (DDPG) (Lillicrap et al., 2015), Stochastic Value Gradients (SVG) (Heess et al., 2015), and MPO. These algorithms are far more sample-efficient but reach a lower final performance.

In the "gaps" task the 56-dimensional humanoid must run forward to match a target velocity of  $4\mathrm{m / s}$  and jump over the gaps between platforms by learning to actuate joints with position-control (Merel et al., 2019). Previously, only an agent operating in the space of pre-learned motor primitives was able to solve the task from pixel observations (Merel et al., 2018; 2019); here we show that V-MPO can learn a challenging visuomotor task from scratch (Fig. 4b). For this task we also demonstrate the importance of the parametric KL constraint, without which the agent learns poorly.

In Figs. 4c-d we also show that V-MPO achieves the highest asymptotic performance reported for two OpenAI Gym tasks (Brockman et al., 2016). Again, MPO and Stochastic Actor-Critic (Haarnoja et al., 2018) are far more sample-efficient but reach a lower final performance.

# 6 CONCLUSION

In this work we have introduced a scalable on-policy deep reinforcement learning algorithm, V-MPO, that is applicable to both discrete and continuous control domains. For the results presented in this work neither importance weighting nor entropy regularization was used; moreover, since the size of neural network parameter updates is limited by KL constraints, we were also able to use the same learning rate for all experiments. This suggests that a scalable, performant RL algorithm may not require some of the tricks that have been developed over the past several years. Interestingly, both the original MPO algorithm for replay-based off-policy learning (Abdolmaleki et al., 2018a;b) and V-MPO for on-policy learning are derived from similar principles, providing evidence for the benefits of this approach as an alternative to popular policy gradient-based methods.

# REFERENCES

Abbas Abdelmaleki, Bob Price, Nuno Lau, Luis P Reis, and Gerhard Neumann. Deriving and Improving CMA-ES with Information Geometric Trust Regions. Proceedings of the Genetic and Evolutionary Computation Conference, 2017.  
Abbas Abdelmaleki, Jost Tobias Springenberg, Jonas Degrave, Steven Bohez, Yuval Tassa, Dan Belov, Nicolas Heess, and Martin Riedmiller. Relative Entropy Regularized Policy Iteration. arXiv preprint, 2018a. URL https://arxiv.org/pdf/1812.02256.pdf.  
Abbas Abdelmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a Posteriori Policy Optimisation. Int. Conf. Learn. Represent., 2018b. URL https://arxiv.org/pdf/1806.06920.pdf.  
Anonymous Authors. Off-Policy Actor-Critic with Shared Experience Replay. Under review, Int. Conf. Learn. Represent., 2019.  
Charles Beattie, Joel Z Leibo, Denis Teptyashin, Tom Ward, Marcus Wainwright, Heinrich Kuttler, Andrew Lefrancq, Simon Green, Víctor Valdés, Amir Sadik, et al. Deepmind Lab. arXiv preprint arXiv:1612.03801, 2016.  
Marc G. Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents. Journal of Artificial Intelligence Research, 47, 2012.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI Gym. arXiv preprint, 2016. URL http://arxiv.org/abs/1606.01540.  
Peter Buchlovsky, David Budden, Dominik Grewe, Chris Jones, John Aslanides, Frederic Besse, Andy Brock, Aidan Clark, Sergio Gomez Colmenarejo, Aedan Pope, Fabio Viola, and Dan Below. TF-Replicator: Distributed Machine Learning for Researchers. arXiv preprint, 2019. URL http://arxiv.org/abs/1902.00465.  
Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. arXiv preprint, 2019. URL http://arxiv.org/abs/1901.02860.  
DeepMind. AlphaStar: Mastering the Real-Time Strategy Game StarCraft II, 2019. URL https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking Deep Reinforcement Learning for Continuous Control. arXiv preprint, 2016. URL http://arxiv.org/abs/1604.06778.  
Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. arXiv preprint, 2018. URL http://arxiv.org/abs/1802.01561.  
Google. Cloud TPU, 2018. URL https://cloud.google.com/tpu/.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv preprint, 2018. URL http://arxiv.org/abs/1801.01290.  
Nikolaus Hansen, Andreas Ostermeier, and Andreas Ostermeier. Convergence Properties of Evolution Strategies with the Derandomized Covariance Matrix Adaptation: CMA-ES. 1997. URL http://www.cmap.polytechnique.fr/~nikolaus.hansen/CMAES2.pdf.  
Nicolas Heess, Greg Wayne, David Silver, Timothy P. Lillicrap, Yuval Tassa, and Tom Erez. Learning continuous control policies by stochastic value gradients. arXiv preprint, 2015. URL http://arxiv.org/abs/1510.09142.

Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, and Hado van Hasselt. Multi-task Deep Reinforcement Learning with PopArt. arXiv preprint, 2018. URL https://arxiv.org/pdf/1809.04474.pdf.  
Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. Population Based Training of Neural Networks. arXiv preprint, 2017a. URL http://arxiv.org/abs/1711.09846.  
Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z Leibo, David Silver, and Koray Kavukcuoglu. Reinforcement Learning with Unsupervised Auxiliary Tasks. Int. Conf. Learn. Represent., 2017b. URL https://openreview.net/pdf?id=SJ6yPD5xg.  
Max Jaderberg, Wojciech M. Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castañeda, Charles Beattie, Neil C. Rabinowitz, Ari S. Morcos, Avraham Ruderman, Nicolas Sonnerat, Tim Green, Louise Deason, Joel Z. Leibo, David Silver, Demis Hassabis, Koray Kavukcuoglu, and Thore Graepel. Human-level performance in 3d multiplayer games with population-based reinforcement learning. Science, 364:859-865, 2019. URL https://science.sciencemag.org/content/364/6443/859.  
Steven Kaptuowski, Georg Ostrovski, John Quan, Rémi Munos, and Will Dabney. Recurrent Experience Replay in Distributed Reinforcement Learning. Int. Conf. Learn. Represent., 2019. URL https://openreview.net/pdf?id=r11yTjAqYX.  
Diederik P. Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. Int. Conf. Learn. Represent., 2015. URL https://arxiv.org/abs/1412.6980.  
Sergey Levine. Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review. arXiv preprint, 2018. URL http://arxiv.org/abs/1805.00909.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint, 2015. URL http://arxiv.org/abs/1509.02971.  
Josh Merel, Leonard Hasenclever, Alexandre Galashov, Arun Ahuja, Vu Pham, Greg Wayne, Yee Whye Teh, and Nicolas Heess. Neural probabilistic motor primitives for humanoid control. arXiv preprint, 2018. URL http://arxiv.org/abs/1811.11711.  
Josh Merel, Arun Ahuja, Vu Pham, Saran Tunyasuvunakool, Siqi Liu, Dhruva Tirumala, Nicolas Heess, and Greg Wayne. Hierarchical Visuomotor Control of Humanoids. Int. Conf. Learn. Represent., 2019. URL https://openreview.net/pdf?id=BJfYvo09Y7.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-Level Control through Deep Reinforcement Learning. Nature, 518:529-533, 2015. URL http://dx.doi.org/10.1038/nature14236.  
Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy P Lillicrap, David Silver, and Koray Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. arXiv:1602.01783, 2016. URL http://arxiv.org/abs/1602.01783.  
Radford M. Neal and Geoffrey E. Hinton. A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants. In M.I. Jordan (ed.), Learn. Graph. Model. NATO ASI Ser. vol. 89. Springer, Dordrecht, 1998.  
OpenAI. OpenAI Five, 2018a. URL https://openai.com/blog/openai-five/.  
OpenAI. Learning Dexterity, 2018b. URL https://openai.com/blog/learning-dexterity/.  
Jan Peters, M Katharina, and Yasemin Altun. Relative Entropy Policy Search. Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, pp. 1607-1612, 2008.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. 2019. URL https://d4mucfpkseywv.cloudfront.net/better-language-models/language_models_are_unsupervisedMULTITASK_learners.pdf.  
Simon Schmitt, Jonathan J. Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M. Czarnecki, Joel Z. Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, and S. M. Ali Eslami. Kickstarting Deep Reinforcement Learning. arXiv preprint, 2018. URL http://arxiv.org/abs/1803.03835.  
John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. Trust Region Policy Optimization. arXiv preprint, 2015. URL http://arxiv.org/abs/1502.05477.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017. URL http://arxiv.org/abs/1707.06347.  
David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529:484-489, 2016. URL http://www.nature.com/doifinder/10.1038/nature16961.  
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362:1140-1144, 2018. URL https://science.sciencemag.org/content/362/6419/1140.  
Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 1998.  
Richard S Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. A. Solla, T. K. Leen, and K. Müller (eds.), Advances in Neural Information Processing Systems 12, pp. 1057-1063. MIT Press, 2000. URL http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf.  
Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, Timothy P. Lillicrap, and Martin A. Riedmiller. DeepMind Control Suite. arXiv preprint, 2018. URL http://arxiv.org/abs/1801.00690.  
Hado van Hasselt, Arthur Guez, Matteo Hessel, and David Silver. Learning functions across many orders of magnitudes. arXiv preprint, 2016. URL http://arxiv.org/abs/1602.07714.  
Quan Vuong, Keith Ross, and Yiming Zhang. Supervised Policy Update for Deep Reinforcement Learning. arXiv preprint, 2019. URL http://arxiv.org/abs/1805.11706.  
Ronald J. Williams. Simple statistical gradient-following methods for connectionist reinforcement learning. Mach. Learn., 8:229-256, 1992. URL http://dx.doi.org/10.1007/BF00992696.  
Cathy Wu, Aravind Rajeswaran, Yan Duan, Vikash Kumar, Alexandre M. Bayen, Sham Kakade, Igor Mordatch, and Pieter Abbeel. Variance reduction for policy gradient with action-dependent factorized baselines. arXiv preprint, 2018. URL http://arxiv.org/abs/1803.07246.
