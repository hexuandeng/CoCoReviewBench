# DECENTRALIZED DETERMINISTIC MULTI-AGENT REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

[Zhang, ICML 2018] provided the first decentralized actor-critic algorithm for multi-agent reinforcement learning (MARL) that offers convergence guarantees. In that work, policies are stochastic and are defined on finite action spaces. We extend those results to offer a provably-convergent decentralized actor-critic algorithm for learning deterministic policies on continuous action spaces. Deterministic policies are important in real-world settings. To handle the lack of exploration inherent in deterministic policies, we consider both off-policy and on-policy settings. We provide the expression of a local deterministic policy gradient, decentralized deterministic actor-critic algorithms and convergence guarantees for linearly-approximated value functions. This work will facilitate decentralized MARL in high-dimensional action spaces and pave the way for more widespread use of MARL.

# 1 INTRODUCTION

Cooperative multi-agent reinforcement learning (MARL) has seen considerably less use than its single-agent analog, in part because often no central agent exists to coordinate the cooperative agents. As a result, decentralized architectures have been advocated for MARL. Recently, decentralized architectures have been shown to admit convergence guarantees comparable to their centralized counterparts under mild network-specific assumptions (see Zhang et al. (2018); Suttle et al. (2019)). In this work, we develop a decentralized actor-critic algorithm with deterministic policies for multiagent reinforcement learning. Specifically, we extend results for actor-critic with stochastic policies (Bhatnagar et al. (2009); Degris et al. (2012); Maei (2018); Suttle et al. (2019)) to handle deterministic policies. Indeed, theoretical and empirical work has shown that deterministic algorithms outperform their stochastic counterparts in high-dimensional continuous action settings (Silver et al. (January 2014b); Lillicrap et al. (2015); Fujimoto et al. (2018)). Deterministic policies further avoid estimating the complex integral over the action space. Empirically this allows for lower variance of the critic estimates and faster convergence. On the other hand, deterministic policy gradient methods suffer from reduced exploration. For this reason, we provide both off-policy and on-policy versions of our results, the off-policy version allowing for significant improvements in exploration. The contributions of this paper are three-fold: (1) we derive the expression of the gradient in terms of the long-term average reward, which is needed in the undiscounted multi-agent setting with deterministic policies; (2) we show that the deterministic policy gradient is the limiting case, as policy variance tends to zero, of the stochastic policy gradient; and (3) we provide a decentralized deterministic multi-agent actor critic algorithm and prove its convergence under linear function approximation.

# 2 BACKGROUND

Consider a system of  $N$  agents denoted by  $\mathcal{N} = [N]$  in a decentralized setting. Agents determine their decisions independently based on observations of their own rewards. Agents may however communicate via a possibly time-varying communication network, characterized by an undirected graph  $\mathcal{G}_t = (\mathcal{N},\mathcal{E}_t)$ , where  $\mathcal{E}_t$  is the set of communication links connecting the agents at time  $t\in \mathbb{N}$ . The networked multi-agent MDP is thus characterized by a tuple  $(S,\{\mathcal{A}^i\}_{i\in \mathcal{N}},P,\{R^i\}_{i\in \mathcal{N}},\{\mathcal{G}_t\}_{t\geq 0})$  where  $S$  is a finite global state space shared by all agents in  $\mathcal{N}$ ,  $\mathcal{A}^i$  is the action space of agent  $i$ , and  $\{\mathcal{G}_t\}_{t\geq 0}$  is a time-varying communication network. In

addition, let  $\mathcal{A} = \prod_{i\in \mathcal{N}}\mathcal{A}^i$  denote the joint action space of all agents. Then,  $P:S\times \mathcal{A}\times \mathcal{S}\to [0,1]$  is the state transition probability of the MDP, and  $R^{i}:S\times \mathcal{A}\rightarrow \mathbb{R}$  is the local reward function of agent  $i$ . States and actions are assumed globally observable whereas rewards are only locally observable. At time  $t$ , each agent  $i$  chooses its action  $a_{t}^{i}\in \mathcal{A}^{i}$  given state  $s_t\in S$ , according to a local parameterized policy  $\pi_{\theta^i}^i:\mathcal{S}\times \mathcal{A}^i\to [0,1]$ , where  $\pi_{\theta^i}^i (s,a^i)$  is the probability of agent  $i$  choosing action  $a^i$  at state  $s$ , and  $\theta^i\in \Theta^i\subseteq \mathbb{R}^{m_i}$  is the policy parameter. We pack the parameters together as  $\theta = [(\theta^{1})^{\top},\dots ,(\theta^{N})^{\top}]^{\top}\in \Theta$  where  $\Theta = \prod_{i\in \mathcal{N}}\Theta^i$ . We denote the joint policy by  $\pi_{\theta}:S\times \mathcal{A}\to [0,1]$  where  $\pi_{\theta}(s,a) = \prod_{i\in \mathcal{N}}\pi_{\theta^i}^i (s,a^i)$ . Note that decisions are decentralized in that rewards are observed locally, policies are evaluated locally, and actions are executed locally. We assume that for any  $i\in \mathcal{N}$ ,  $s\in S$ ,  $a^i\in \mathcal{A}^i$ , the policy function  $\pi_{\theta^i}^i (s,a^i) > 0$  for any  $\theta^i\in \Theta^i$  and that  $\pi_{\theta^i}^i (s,a^i)$  is continuously differentiable with respect to the parameters  $\theta^i$  over  $\Theta^i$ . In addition, for any  $\theta \in \Theta$ , let  $P^\theta :S\times S\to [0,1]$  denote the transition matrix of the Markov chain  $\{s_t\}_{t\geq 0}$  induced by policy  $\pi_{\theta}$ , that is, for any  $s,s'\in S$ ,  $P^{\theta}(s'|s) = \sum_{a\in A}\pi_{\theta}(s,a)\cdot P(s'|s,a)$ . We make the standard assumption that the Markov chain  $\{s_t\}_{t\geq 0}$  is irreducible and aperiodic under any  $\pi_{\theta}$  and denote its stationary distribution by  $d_{\theta}$ .

Our objective is to find a policy  $\pi_{\theta}$  that maximizes the long-term average reward over the network. Let  $r_{t + 1}^{i}$  denote the reward received by agent  $i$  as a result of taking action  $a_{t}^{i}$ . Then, we wish to solve:

$$
\max _ {\theta} J (\pi_ {\theta}) = \lim _ {T \to \infty} \frac {1}{T} \mathbb {E} \left[ \sum_ {t = 0} ^ {T - 1} \frac {1}{N} \sum_ {i \in \mathcal {N}} r _ {t + 1} ^ {i} \right] = \sum_ {s \in S, a \in \mathcal {A}} d _ {\theta} (s) \pi_ {\theta} (s, a) \bar {R} (s, a),
$$

where  $\bar{R}(s, a) = (1/N) \cdot \sum_{i \in \mathcal{N}} R^i(s, a)$  is the globally averaged reward function. Let  $\bar{r}_t = (1/N) \cdot \sum_{i \in \mathcal{N}} r_t^i$ , then  $\bar{R}(s, a) = \mathbb{E}[\bar{r}_{t+1}|s_t = s, a_t = a]$ , and therefore, the global relative action-value function is:  $Q_\theta(s, a) = \sum_{t \geq 0} \mathbb{E}[\bar{r}_{t+1} - J(\theta)|s_0 = s, a_0 = a, \pi_\theta]$ , and the global relative state-value function is:  $V_\theta(s) = \sum_{a \in \mathcal{A}} \pi_\theta(s, a) Q_\theta(s, a)$ . For simplicity, we refer to  $V_\theta$  and  $Q_\theta$  as simply the state-value function and action-value function. We define the advantage function as  $A_\theta(s, a) = Q_\theta(s, a) - V_\theta(s)$ .

Zhang et al. (2018) provided the first provably convergent MARL algorithm in the context of the above model. The fundamental result underlying their algorithm is a local policy gradient theorem:

$$
\nabla_ {\theta^ {i}} J (\mu_ {\theta}) = \mathbb {E} _ {s \sim d _ {\theta}, a \sim \pi_ {\theta}} \left[ \nabla_ {\theta^ {i}} \log \pi_ {\theta^ {i}} ^ {i} (s, a ^ {i}) \cdot A _ {\theta} ^ {i} (s, a) \right],
$$

where  $A_{\theta}^{i}(s,a) = Q_{\theta}(s,a) - \tilde{V}_{\theta}^{i}(s,a^{-i})$  is a local advantage function and  $\tilde{V}_{\theta}^{i}(s,a^{-i}) = \sum_{a^{i}\in \mathcal{A}^{i}}\pi_{\theta^{i}}^{i}(s,a^{i})Q_{\theta}(s,a^{i},a^{-i})$ . This theorem has important practical value as it shows that the policy gradient with respect to each local parameter  $\theta^i$  can be obtained locally using the corresponding score function  $\nabla_{\theta^i}\log \pi_{\theta^i}^i$  provided that agent  $i$  has an unbiased estimate of the advantage functions  $A_{\theta}^{i}$  or  $A_{\theta}$ . With only local information, the advantage functions  $A_{\theta}^{i}$  or  $A_{\theta}$  cannot be well estimated since the estimation requires the rewards  $\{r_t^i\}_{i\in \mathcal{N}}$  of all agents. Therefore, they proposed a consensus based actor-critic that leverages the communication network to share information between agents by placing a weight  $c_{t}(i,j)$  on the message transmitted from agent  $j$  to agent  $i$  at time  $t$ . Their action-value function  $Q_{\theta}$  was approximated by a parameterized function  $\hat{Q}_{\omega}:S\times \mathcal{A}\to \mathbb{R}$ , and each agent  $i$  maintains its own parameter  $\omega^i$ , which it uses to form a local estimate  $\hat{Q}_{\omega^i}$  of the global  $Q_{\theta}$ . At each time step  $t$ , each agent  $i$  shares its local parameter  $\omega_{t}^{i}$  with its neighbors on the network, and the shared parameters are used to arrive at a consensual estimate of  $Q_{\theta}$  over time.

# 3 LOCAL GRADIENTS OF DETERMINISTIC POLICIES

While the use of a stochastic policy facilitates the derivations of convergence proofs, most real-world control tasks require a deterministic policy to be implementable. In addition, the quantities estimated in the deterministic critic do not involve estimation of the complex integral over the action space found in the stochastic version. This offers lower variance of the critic estimates and faster convergence. To address the lack of exploration that comes with deterministic policies, we provide both off-policy and on-policy versions of our results. Our first requirement is a local deterministic policy gradient theorem.

We assume that  $\mathcal{A}^i = \mathbb{R}^{n_i}$ . We make standard regularity assumptions on our MDP. That is, we assume that for any  $s, s' \in S$ ,  $P(s'|s, a)$  and  $R^i(s, a)$  are bounded and have bounded first and second derivatives. We consider local deterministic policies  $\mu_{\theta^i}^i: S \to \mathcal{A}^i$  with parameter vector  $\theta^i \in \Theta^i$ , and denote the joint policy by  $\mu_\theta: S \to \mathcal{A}$ , where  $\mu_\theta(s) = (\mu_{\theta^1}^1(s), \ldots, \mu_{\theta^N}^N(s))$  and  $\theta = [(\theta^1)^\top, \ldots, (\theta^N)^\top]^\top$ . We assume that for any  $s \in S$ , the deterministic policy function  $\mu_{\theta^i}^i(s)$  is twice continuously differentiable with respect to the parameter  $\theta^i$  over  $\Theta^i$ . Let  $P^\theta$  denote the transition matrix of the Markov chain  $\{s_t\}_{t \geq 0}$  induced by policy  $\mu_\theta$ , that is, for any  $s, s' \in S$ ,  $P^\theta(s'|s) = P(s'|s, \mu_\theta(s))$ . We assume that the Markov chain  $\{s_t\}_{t \geq 0}$  is irreducible and aperiodic under any  $\mu_\theta$  and denote its stationary distribution by  $d^{\mu_\theta}$ .

Our objective is to find a policy  $\mu_{\theta}$  that maximizes the long-run average reward:

$$
\max _ {\theta} J (\mu_ {\theta}) = \mathbb {E} _ {s \sim d ^ {\mu_ {\theta}}} [ \bar {R} (s, \mu_ {\theta} (s)) ] = \sum_ {s \in \mathcal {S}} d ^ {\mu_ {\theta}} (s) \bar {R} (s, \mu_ {\theta} (s)).
$$

Analogous to the stochastic policy case, we denote the action-value function by  $Q_{\theta}(s,a) = \sum_{t\geq 0}\mathbb{E}[\bar{r}_{t + 1} - J(\mu_{\theta})|s_0 = s,a_0 = a,\mu_{\theta}]$ , and the state-value function by  $V_{\theta}(s) = Q_{\theta}(s,\mu_{\theta}(s))$ . When there is no ambiguity, we will denote  $J(\mu_{\theta})$  and  $d^{\mu_{\theta}}$  by simply  $J(\theta)$  and  $d^{\theta}$ , respectively. We present three results for the long-run average reward: (1) an expression for the local deterministic policy gradient in the on-policy setting  $\nabla_{\theta^i}J(\mu_\theta)$ , (2) an expression for the gradient in the off-policy setting, and (3) we show that the deterministic policy gradient can be seen as the limit of the stochastic one.

# On-Policy Setting

Theorem 1 (Local Deterministic Policy Gradient Theorem - On Policy). For any  $\theta \in \Theta$ ,  $i \in \mathcal{N}$ ,  $\nabla_{\theta^i}J(\mu_\theta)$  exists and is given by

$$
\nabla_ {\theta^ {i}} J (\mu_ {\theta}) = \mathbb {E} _ {s \sim d ^ {\mu_ {\theta}}} \left[ \nabla_ {\theta^ {i}} \mu_ {\theta^ {i}} ^ {i} (s) \nabla_ {a ^ {i}} \left. Q _ {\theta} (s, \mu_ {\theta^ {- i}} ^ {- i} (s), a ^ {i}) \right| _ {a ^ {i} = \mu_ {\theta^ {i}} ^ {i} (s)} \right].
$$

The first step of the proof consists in showing that  $\nabla_{\theta}J(\mu_{\theta}) = \mathbb{E}_{s\sim d^{\theta}}\left[\nabla_{\theta}\mu_{\theta}(s)\nabla_{a}Q_{\theta}(s,a)|_{a = \mu_{\theta}(s)}\right]$ . This is an extension of the well-known stochastic case, for which we have  $\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}_{s\sim d_{\theta}}\left[\nabla_{\theta}\log (\pi_{\theta}(a|s))Q_{\theta}(s,a)\right]$ , which holds for a long-term averaged return with stochastic policy (e.g Theorem 1 of Sutton et al. (2000a)). See the Appendix for the details.

Off-Policy Setting In the off-policy setting, we are given a behavior policy  $\pi : S \to \mathcal{P}(\mathcal{A})$ , and our goal is to maximize the long-run average reward under state distribution  $d^{\pi}$ :

$$
J _ {\pi} \left(\mu_ {\theta}\right) = \mathbb {E} _ {s \sim d ^ {\pi}} \left[ \bar {R} (s, \mu_ {\theta} (s)) \right] = \sum_ {s \in \mathcal {S}} d ^ {\pi} (s) \bar {R} (s, \mu_ {\theta} (s)). \tag {1}
$$

Note that we consider here an excursion objective (Sutton et al. (2009); Silver et al. (January 2014a); Sutton et al. (2016)) since we take the average over the state distribution of the behaviour policy  $\pi$  of the state-action reward when selecting action given by the target policy  $\mu_{\theta}$ . We thus have:

Theorem 2 (Local Deterministic Policy Gradient Theorem - Off Policy). For any  $\theta \in \Theta$ ,  $i \in \mathcal{N}$ ,  $\pi : \mathcal{S} \to \mathcal{P}(\mathcal{A})$  a fixed stochastic policy,  $\nabla_{\theta^i} J_{\pi}(\mu_\theta)$  exists and is given by

$$
\nabla_ {\theta^ {i}} J _ {\pi} (\mu_ {\theta}) = \mathbb {E} _ {s \sim d ^ {\pi}} \left[ \nabla_ {\theta^ {i}} \mu_ {\theta^ {i}} ^ {i} (s) \nabla_ {a ^ {i}} \bar {R} (s, \mu_ {\theta^ {- i}} ^ {- i} (s), a ^ {i}) \big | _ {a ^ {i} = \mu_ {\theta^ {i}} ^ {i} (s)} \right].
$$

Proof. Since  $d^{\pi}$  is independent of  $\theta$  we can take the gradient on both sides of (1)

$$
\nabla_ {\theta} J _ {\pi} (\mu_ {\theta}) = \mathbb {E} _ {s \sim d ^ {\pi}} \left[ \nabla_ {\theta} \mu_ {\theta} (s) \left. \nabla_ {a} \bar {R} (s, \mu_ {\theta} (s)) \right| _ {a = \mu_ {\theta} (s)} \right].
$$

Given that  $\nabla_{\theta^i}\mu_\theta^j (s) = 0$  if  $i\neq j$ , we have  $\nabla_{\theta}\mu_{\theta}(s) = \mathrm{Diag}(\nabla_{\theta^1}\mu_{\theta_1}^1 (s),\ldots ,\nabla_{\theta^N}\mu_{\theta_N}^N (s))$  and the result follows.

This result implies that, off-policy, each agent needs access to  $\mu_{\theta_t^{-i}}^{-i}(s_t)$  for every  $t$

Limit Theorem As noted by Silver et al. (January 2014b), the fact that the deterministic gradient is a limit case of the stochastic gradient enables the standard machinery of policy gradient, such as compatible-function approximation (Sutton et al. (2000b)), natural gradients (Kakade (2001)), on-line feature adaptation (Prabuchandran et al. (2016),) and actor-critic (Konda (2002)) to be used with deterministic policies. We show that it holds in our setting. The proof can be found in the Appendix.

Theorem 3 (Limit of the Stochastic Policy Gradient for MARL). Let  $\pi_{\theta, \sigma}$  be a stochastic policy such that  $\pi_{\theta, \sigma}(a|s) = \nu_{\sigma}(\mu_{\theta}(s), a)$ , where  $\sigma$  is a parameter controlling the variance, and  $\nu_{\sigma}$  satisfy Condition 1 in the Appendix. Then,

$$
\lim  _ {\sigma \downarrow 0} \nabla_ {\theta} J _ {\pi_ {\theta , \sigma}} (\pi_ {\theta , \sigma}) = \nabla_ {\theta} J _ {\mu_ {\theta}} (\mu_ {\theta})
$$

where on the l.h.s the gradient is the standard stochastic policy gradient and on the r.h.s. the gradient is the deterministic policy gradient.

# 4 ALGORITHMS

We provide two decentralized deterministic actor-critic algorithms, one on-policy and the other off-policy and demonstrate their convergence in the next section; assumptions and proofs are provided in the Appendix.

# On-Policy Deterministic Actor-Critic

Algorithm 1 Networked deterministic on-policy actor-critic  
Initialize  $\hat{J}_0^i,\omega_0^i,\widetilde{\omega}_0^i,\theta_0^i,\forall i\in \mathcal{N}$  ; state  $s_0$  ; stepsizes  $\{\beta_{\omega ,t}\}_{t\geq 0},\{\beta_{\theta ,t}\}_{t\geq 0}$    
Draw  $a_0^i = \mu_{\theta_0^i}^i (s_0)$  and compute  $\widetilde{a}_0^i = \nabla_{\theta^i}\mu_{\theta_0^i}^i (s_0)$    
Observe joint action  $a_0 = (a_0^1,\dots ,a_0^N)$  and  $\widetilde{a}_0 = (\widetilde{a}_0^1,\dots ,\widetilde{a}_0^N)$    
repeat   
for  $i\in \mathcal{N}$  do Observe  $s_{t + 1}$  and reward  $r_{t + 1}^{i} = r^{i}(s_{t},a_{t})$  Update  $\hat{J}_{t + 1}^{i}\gets (1 - \beta_{\omega ,t})\cdot \hat{J}_{t}^{i} + \beta_{\omega ,t}\cdot r_{t + 1}^{i}$  Draw action  $a_{t + 1} = \mu_{\theta_t^i}^i (s_{t + 1})$  and compute  $\widetilde{a}_{t + 1}^{i} = \nabla_{\theta^{i}}\mu_{\theta_{t}^{i}}^{i}(s_{t + 1})$    
end for   
Observe joint action  $a_{t + 1} = (a_{t + 1}^{1},\ldots ,a_{t + 1}^{N})$  and  $\widetilde{a}_{t + 1} = (\widetilde{a}_{t + 1}^{1},\ldots ,\widetilde{a}_{t + 1}^{N})$    
for  $i\in \mathcal{N}$  do Update:  $\delta_t^i\gets r_{t + 1}^i -\hat{J}_t^i +\hat{Q}_{\omega_t^i}(s_{t + 1},a_{t + 1}) - \hat{Q}_{\omega_t^i}(s_t,a_t)$  Critic step:  $\begin{array}{r}\widetilde{\omega}_t^i\gets \omega_t^i +\beta_{\omega ,t}\cdot \delta_t^i\cdot \nabla_\omega \hat{Q}_{\omega^i}(s_t,a_t)\bigg|_{\omega = \omega_t^i}\\ \textbf{Actor step:}\left.\theta_{t + 1}^i = \theta_t^i +\beta_{\theta ,t}\cdot \nabla_\theta^i\mu_{\theta_t^i}^i (s_t)\nabla_{a^i}\hat{Q}_{\omega_t^i}(s_t,a_t^{-i},a^i)\right|_{a^i = a_t^i} \end{array}$  Send  $\widetilde{\omega}_t^i$  to the neighbors  $\{j\in \mathcal{N}:(i,j)\in \mathcal{E}_t\}$  over  $\mathcal{G}_t$  Consensus step:  $\omega_{t + 1}^{i}\gets \sum_{j\in \mathcal{N}}c_{t}^{ij}\cdot \widetilde{\omega}_{t}^{j}$    
end for   
until end

Consider the following on-policy algorithm. The actor step is based on an expression for  $\nabla_{\theta^i}J(\mu_\theta)$  in terms of  $\nabla_{a^i}Q_\theta$  (see Equation equation 15 in the Appendix). We approximate the action-value function  $Q_{\theta}$  using a family of functions  $\hat{Q}_{\omega}:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$  parameterized by  $\omega$ , a column vector in  $\mathbb{R}^K$ . Each agent  $i$  maintains its own parameter  $\omega^i$  and uses  $\hat{Q}_{\omega^i}$  as its local estimate of  $Q_{\theta}$ . The parameters  $\omega^i$  are updated in the critic step using consensus updates through a weight matrix  $C_t = \left(c_t^{ij}\right)_{i,j}\in \mathbb{R}^{N\times N}$  where  $c_t^{ij}$  is the weight on the message transmitted from  $i$  to  $j$  at time  $t$ ,

namely:

$$
\hat {J} _ {t + 1} ^ {i} = \left(1 - \beta_ {\omega , t}\right) \cdot \hat {J} _ {t} ^ {i} + \beta_ {\omega , t} \cdot r _ {t + 1} ^ {i} \tag {2}
$$

$$
\left. \widetilde {\omega} _ {t} ^ {i} = \omega_ {t} ^ {i} + \beta_ {\omega , t} \cdot \delta_ {t} ^ {i} \cdot \nabla_ {\omega} \hat {Q} _ {\omega^ {i}} \left(s _ {t}, a _ {t}\right) \right| _ {\omega = \omega_ {t} ^ {i}} \tag {3}
$$

$$
\omega_ {t + 1} ^ {i} = \sum_ {j \in \mathcal {N}} c _ {t} ^ {i j} \cdot \widetilde {\omega} _ {t} ^ {j} \tag {4}
$$

with

$$
\delta_ {t} ^ {i} = r _ {t + 1} ^ {i} - \hat {J} _ {t} ^ {i} + \hat {Q} _ {\omega_ {t} ^ {i}} (s _ {t + 1}, a _ {t + 1}) - \hat {Q} _ {\omega_ {t} ^ {i}} (s _ {t}, a _ {t}).
$$

For the actor step, each agent  $i$  improves its policy via:

$$
\left. \theta_ {t + 1} ^ {i} = \theta_ {t} ^ {i} + \beta_ {\theta , t} \cdot \nabla_ {\theta^ {i}} \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t}) \cdot \nabla_ {a ^ {i}} \hat {Q} _ {\omega_ {t} ^ {i}} \left(s _ {t}, a _ {t} ^ {- i}, a ^ {i}\right) \right| _ {a ^ {i} = a _ {t} ^ {i}}. \tag {5}
$$

Since Algorithm 1 is an on-policy algorithm, each agent updates the critic using only  $(s_t, a_t, s_{t+1})$ , at time  $t$  knowing that  $a_{t+1} = \mu_{\theta_t}(s_{t+1})$ . The terms in blue are additional terms that need to be shared when using compatible features (this is explained further in the next section).

Off-Policy Deterministic Actor-Critic We further propose an off-policy actor-critic algorithm, defined in Algorithm 2 to enable better exploration capability. Here, the goal is to maximize  $J_{\pi}(\mu_{\theta})$  where  $\pi$  is the behavior policy. To do so, the globally averaged reward function  $\bar{R}(s,a)$  is approximated using a family of functions  $\hat{\bar{R}}_{\lambda} : S \times \mathcal{A} \to \mathbb{R}$  that are parameterized by  $\lambda$ , a column vector in  $\mathbb{R}^K$ . Each agent  $i$  maintains its own parameter  $\lambda^i$  and uses  $\hat{\bar{R}}_{\lambda^i}$  as its local estimate of  $\bar{R}$ . Based on (1), the actor update is

$$
\theta_ {t + 1} ^ {i} = \theta_ {t} ^ {i} + \beta_ {\theta , t} \cdot \nabla_ {\theta^ {i}} \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t}) \cdot \nabla_ {a ^ {i}} \hat {\tilde {R}} _ {\lambda_ {t} ^ {i}} \left(s _ {t}, \mu_ {\theta_ {t} ^ {- i}} ^ {- i} (s _ {t}), a ^ {i}\right) \Big | _ {a ^ {i} = \mu_ {\theta_ {t} ^ {i}} (s _ {t})}, \tag {6}
$$

which requires each agent  $i$  to have access to  $\mu_{\theta_t^j}^j (s_t)$  for  $j\in \mathcal{N}$ .

The critic update is

$$
\left. \widetilde {\lambda} _ {t} ^ {i} = \lambda_ {t} ^ {i} + \beta_ {\lambda , t} \cdot \delta_ {t} ^ {i} \cdot \nabla_ {\lambda} \hat {\bar {R}} _ {\lambda^ {i}} \left(s _ {t}, a _ {t}\right) \right| _ {\lambda = \lambda_ {t} ^ {i}} \tag {7}
$$

$$
\lambda_ {t + 1} ^ {i} = \sum_ {j \in \mathcal {N}} c _ {t} ^ {i j} \widetilde {\lambda} _ {t} ^ {j}, \tag {8}
$$

with

$$
\delta_ {t} ^ {i} = r ^ {i} \left(s _ {t}, a _ {t}\right) - \hat {\bar {R}} _ {\lambda_ {t} ^ {i}} \left(s _ {t}, a _ {t}\right). \tag {9}
$$

In this case,  $\delta_t^i$  was motivated by distributed optimization results, and is not related to the local TD-error (as there is no "temporal" relationship for  $R$ ). Rather, it is simply the difference between the sample reward and the bootstrap estimate. The terms in blue are additional terms that need to be shared when using compatible features (this is explained further in the next section).

# 5 CONVERGENCE

To show convergence, we use a two-timescale technique where in the actor, updating deterministic policy parameter  $\theta^i$  occurs more slowly than that of  $\omega^i$  and  $\hat{J}^i$  in the critic. We study the asymptotic behaviour of the critic by freezing the joint policy  $\mu_{\theta}$ , then study the behaviour of  $\theta_t$  under convergence of the critic. To ensure stability, projection is often assumed since it is not clear how boundedness of  $\{\theta_t^i\}$  can otherwise be ensured (see Bhatnagar et al. (2009)). However, in practice, convergence is typically observed even without the projection step (see Bhatnagar et al. (2009); Degris et al. (2012); Prabuchandran et al. (2016); Zhang et al. (2018); Suttle et al. (2019)). Additional technical assumptions are required to show convergence and can be found in the Appendix.

Algorithm 2 Networked deterministic off-policy actor-critic  
Initialize  $\lambda_0^i,\widetilde{\lambda}_0^i,\theta_0^i,\forall i\in \mathcal{N}$  ; state  $s_0$  ; stepsizes  $\{\beta_{\lambda ,t}\}_{t\geq 0},\{\beta_{\theta ,t}\}_{t\geq 0}$    
Draw  $a_0^i\sim \pi^i (s_0)$  , compute  $\dot{a}_0^i = \mu_{\theta_0^i}^i (s_0)$  and  $\tilde{a}_0^i = \nabla_{\theta^i}\mu_{\theta_0^i}^i (s_0)$    
Observe joint action  $a_0 = (a_0^1,\ldots ,a_0^N),\dot{a}_0 = (\dot{a}_0^1,\ldots ,\dot{a}_0^N)$  and  $\widetilde{a}_0 = (\widetilde{a}_0^1,\dots ,\widetilde{a}_0^N)$    
repeat   
for  $i\in \mathcal{N}$  do Observe  $s_{t + 1}$  and reward  $r_{t + 1}^{i} = r^{i}(s_{t},a_{t})$    
end for   
for  $i\in \mathcal{N}$  do Update:  $\delta_t^i\gets r_{t + 1}^i -\hat{\bar{R}}_{\lambda_t^i}(s_t,a_t)$  Critic step:  $\begin{array}{r}\widetilde{\lambda}_t^i\gets \lambda_t^i +\beta_{\lambda ,t}\cdot \delta_t^i\cdot \nabla_\lambda \hat{\bar{R}}_{\lambda^i}(s_t,a_t)\bigg|_{\lambda = \lambda_t^i} \end{array}$  Actor step:  $\theta_{t + 1}^{i} = \theta_{t}^{i} + \beta_{\theta ,t}\cdot \nabla_{\theta^{i}}\mu_{\theta_{t}^{i}}^{i}(s_{t})\cdot \nabla_{a^{i}}\hat{\bar{R}}_{\lambda_{t}^{i}}(s_{t},\mu_{\theta_{t}^{-i}}^{-i}(s_{t}),a^{i})\Big|_{a^{i} = \mu_{\theta_{t}^{i}}(s_{t})}$  Send  $\widetilde{\lambda}_t^i$  to the neighbors  $\{j\in \mathcal{N}:(i,j)\in \mathcal{E}_t\}$  over  $\mathcal{G}_t$    
end for   
for  $i\in \mathcal{N}$  do Consensus step:  $\lambda_{t + 1}^{i}\gets \sum_{j\in \mathcal{N}}c_{t}^{ij}\cdot \widetilde{\lambda}_{t}^{j}$  Draw action  $a_{t + 1}\sim \pi (s_{t + 1})$  , compute  $\dot{a}_{t + 1}^{i} = \mu_{\theta_{t + 1}^{i}}^{i}(s_{t + 1})$  and compute  $\widetilde{a}_{t + 1}^{i} =$ $\nabla_{\theta^i}\mu_{\theta_{t + 1}^i}(s_{t + 1})$    
end for   
Observe joint action  $a_{t + 1} = (a_{t + 1}^{1},\dots ,a_{t + 1}^{N})$  .  $\dot{a}_{t + 1} = (\dot{a}_{t + 1}^{1},\dots ,\dot{a}_{t + 1}^{N})$  and  $\widetilde{a}_{t + 1} =$ $(\widetilde{a}_{t + 1}^{1},\dots ,\widetilde{a}_{t + 1}^{N})$    
until end

On-Policy Convergence To state convergence of the critic step, we define  $D_{\theta}^{s} = \mathrm{Diag}\bigl [d^{\theta}(s),s\in \mathcal{S}\bigr ]$ ,  $\bar{R}_{\theta} = \big[\bar{R} (s,\mu_{\theta}(s)),s\in \mathcal{S}\big]^\top \in \mathbb{R}^{|S|}$  and the operator  $T_{\theta}^{Q}:\mathbb{R}^{|S|}\to \mathbb{R}^{|S|}$  for any action-value vector  $Q\in \mathbb{R}^{|S|}$  (and not  $\mathbb{R}^{|S|\cdot |\mathcal{A}|}$  since there is a mapping associating an action to each state) as:

$$
T _ {\theta} ^ {Q} (Q ^ {\prime}) = \bar {R} _ {\theta} - J (\mu_ {\theta}) \cdot \mathbf {1} + P ^ {\theta} Q ^ {\prime}.
$$

Theorem 4. Under Assumptions 3, 4, and 5, for any given deterministic policy  $\mu_{\theta}$ , with  $\{\hat{J}_t\}$  and  $\{\omega_t\}$  generated from (2), we have  $\lim_{t\to \infty}\frac{1}{N}\sum_{i\in \mathcal{N}}\hat{J}_t^i = J(\mu_\theta)$  and  $\lim_{t\to \infty}\omega_t^i = \omega_\theta$  a.s. for any  $i\in \mathcal{N}$ , where

$$
J (\mu_ {\theta}) = \sum_ {s \in \mathcal {S}} d ^ {\theta} (s) \bar {R} (s, \mu_ {\theta} (s))
$$

is the long-term average return under  $\mu_{\theta}$ , and  $\omega_{\theta}$  is the unique solution to

$$
\Phi_ {\theta} ^ {\top} D _ {\theta} ^ {s} \left[ T _ {\theta} ^ {Q} \left(\Phi_ {\theta} \omega_ {\theta}\right) - \Phi_ {\theta} \omega_ {\theta} \right] = 0. \tag {10}
$$

Moreover,  $\omega_{\theta}$  is the minimizer of the Mean Square Projected Bellman Error (MSPBE), i.e., the solution to

$$
\underset {\omega} {\text {m i n i m i z e}} \| \Phi_ {\theta} \omega - \Pi T _ {\theta} ^ {Q} (\Phi_ {\theta} \omega) \| _ {D _ {\theta} ^ {s}} ^ {2},
$$

where  $\Pi$  is the operator that projects a vector to the space spanned by the columns of  $\Phi_{\theta}$ , and  $\| \cdot \|_{D_{\theta}^s}^2$  denotes the euclidean norm weighted by the matrix  $D_{\theta}^{s}$ .

To state convergence of the actor step, we define quantities  $\psi_{t,\theta}^i$ ,  $\xi_t^i$  and  $\xi_{t,\theta}^{i}$  as

$$
\psi_ {t, \theta} ^ {i} = \nabla_ {\theta^ {i}} \mu_ {\theta^ {i}} ^ {i} (s _ {t}) \quad \text {a n d} \quad \psi_ {t} ^ {i} = \psi_ {t, \theta_ {t}} ^ {i} = \nabla_ {\theta^ {i}} \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t}),
$$

$$
\xi_ {t, \theta} ^ {i} = \left. \nabla_ {a _ {i}} \hat {Q} _ {\omega_ {\theta}} (s _ {t}, a _ {t} ^ {- i}, a _ {i}) \right| _ {a _ {i} = a _ {i} = \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t})} = \left. \nabla_ {a _ {i}} \phi (s _ {t}, a _ {t} ^ {- i}, a _ {i}) \right| _ {a _ {i} = a _ {i} = \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t})} \omega_ {\theta},
$$

$$
\xi_ {t} ^ {i} = \left. \nabla_ {a _ {i}} \hat {Q} _ {\omega_ {t} ^ {i}} (s _ {t}, a _ {t} ^ {- i}, a _ {i}) \right| _ {a _ {i} = \mu_ {\theta i} ^ {i} (s _ {t})} = \left. \nabla_ {a _ {i}} \phi (s _ {t}, a _ {t} ^ {- i}, a _ {i}) \right| _ {a _ {i} = \mu_ {\theta^ {i}} ^ {i} (s _ {t})} \omega_ {t} ^ {i}.
$$

Additionally, we introduce the operator  $\hat{\Gamma} (\cdot)$  as

$$
\hat {\Gamma} ^ {i} [ g (\theta) ] = \lim  _ {0 <   \eta \rightarrow 0} \frac {\Gamma^ {i} [ \theta^ {i} + \eta \cdot g (\theta) ] - \theta^ {i}}{\eta} \tag {11}
$$

for any  $\theta \in \Theta$  and  $g:\Theta \to \mathbb{R}^{m_i}$  a continuous function. In case the limit above is not unique we take  $\hat{\Gamma}^i\left[g(\theta)\right]$  to be the set of all possible limit points of (11).

Theorem 5. Under Assumptions 2, 3, 4, and 5, the policy parameter  $\theta_t^i$  obtained from (5) converges a.s. to a point in the set of asymptotically stable equilibria of

$$
\dot {\theta} ^ {i} = \hat {\Gamma} ^ {i} \left[ \mathbb {E} _ {s _ {t} \sim d ^ {\theta}, \mu_ {\theta}} \left[ \psi_ {t, \theta} ^ {i} \cdot \xi_ {t, \theta} ^ {i} \right] \right], \quad \text {f o r a n y} i \in \mathcal {N}. \tag {12}
$$

In the case of multiple limit points, the above is treated as a differential inclusion rather than an ODE.

The convergence of the critic step can be proved by taking similar steps as that in Zhang et al. (2018). For the convergence of the actor step, difficulties arise from the projection (which is handled using Kushner-Clark Lemma Kushner & Clark (1978)) and the state-dependent noise (that is handled by "natural" timescale averaging Crowder (2009)). Details are provided in the Appendix.

Remark. Note that with a linear function approximator  $Q_{\theta}$ ,  $\psi_{t,\theta} \cdot \xi_{t,\theta} = \nabla_{\theta}\mu_{\theta}(s_t)\nabla_a\hat{Q}_{\omega_\theta}(s_t,a)\bigg|_{a = \mu_\theta (s_t)}$  may not be an unbiased estimate of  $\nabla_{\theta}J(\theta)$ :

$$
\mathbb {E} _ {s \sim d ^ {\theta}} \left[ \psi_ {t, \theta} \cdot \xi_ {t, \theta} \right] = \nabla_ {\theta} J (\theta) + \mathbb {E} _ {s \sim d ^ {\theta}} \left[ \nabla_ {\theta} \mu_ {\theta} (s) \cdot \left(\left. \nabla_ {a} \hat {Q} _ {\omega_ {\theta}} (s, a) \right| _ {a = \mu_ {\theta} (s)} - \left. \nabla_ {a} Q _ {\omega_ {\theta}} (s, a) \right| _ {a = \mu_ {\theta} (s)}\right) \right].
$$

A standard approach to overcome this approximation issue is via compatible features (see, for example, Silver et al. (January 2014a) and Zhang & Zavlanos (2019)), i.e.  $\phi(s, a) = a \cdot \nabla_{\theta} \mu_{\theta}(s)^{\top}$ , giving, for  $\omega \in \mathbb{R}^m$ ,

$$
\begin{array}{l} \hat {Q} _ {\omega} (s, a) = a \cdot \nabla_ {\theta} \mu_ {\theta} (s) ^ {\top} \omega = (a - \mu_ {\theta} (s)) \cdot \nabla_ {\theta} \mu_ {\theta} (s) ^ {\top} \omega + \hat {V} _ {\omega} (s), \\ \text {w i t h} \hat {V} _ {\omega} (s) = \hat {Q} _ {\omega} (s, \mu_ {\theta} (s)) \text {a n d} \left. \nabla_ {a} \hat {Q} _ {\omega} (s, a) \right| _ {a = \mu_ {\theta} (s)} = \nabla_ {\theta} \mu_ {\theta} (s) ^ {\top} \omega . \\ \end{array}
$$

We thus expect that the convergent point of equation 5 corresponds to a small neighborhood of a local optimum of  $J(\mu_{\theta})$ , i.e.,  $\nabla_{\theta^i}J(\mu_\theta) = 0$ , provided that the error for the gradient of the action-value function  $\left. \nabla_{a}\hat{Q}_{\omega}(s,a)\right|_{a = \mu_{\theta}(s)} - \left. \nabla_{a}Q_{\theta}(s,a)\right|_{a = \mu_{\theta}(s)}$  is small. However, note that using compatible features requires computing, at each step  $t$ ,  $\phi (s_t,a_t) = a_t\cdot \nabla_\theta \mu_\theta (s_t)^\top$ . Thus, in Algorithm 1, each agent observes not only the joint action  $a_{t + 1} = (a_{t + 1}^{1},\dots ,a_{t + 1}^{N})$  but also  $(\nabla_{\theta^1}\mu_{\theta_t^1}^1 (s_{t + 1}),\ldots ,\nabla_{\theta^N}\mu_{\theta_t^N}^N (s_{t + 1}))$  (see the parts in blue in Algorithm 1).

# Off-Policy Convergence

Theorem 6. Under Assumptions 1, 4, and 6, for any given behavior policy  $\pi$  and any  $\theta \in \Theta$ , with  $\{\lambda_t^i\}$  generated from (7), we have  $\lim_{t\to \infty}\lambda_t^i = \lambda_\theta$  a.s. for any  $i\in \mathcal{N}$ , where  $\lambda_{\theta}$  is the unique solution to

$$
B _ {\pi , \theta} \cdot \lambda_ {\theta} = A _ {\pi , \theta} \cdot d _ {\pi} ^ {s} \tag {13}
$$

where  $d_{\pi}^{s} = \left[d^{\pi}(s), s \in \mathcal{S}\right]^{\top}$ ,  $A_{\pi, \theta} = \left[\int_{\mathcal{A}} \pi(a|s)\bar{R}(s,a)w(s,a)^{\top}\mathrm{d}a, s \in \mathcal{S}\right] \in \mathbb{R}^{K \times |\mathcal{S}|}$  and  $B_{\pi, \theta} = \left[\sum_{s \in \mathcal{S}} d^{\pi}(s)\int_{\mathcal{A}} \pi(a|s)w_{i}(s,a) \cdot w(s,a)^{\top}\mathrm{d}a, 1 \leq i \leq K\right] \in \mathbb{R}^{K \times K}$ .

From here on we let

$$
\begin{array}{l} \xi_ {t, \theta} ^ {i} = \left. \nabla_ {a _ {i}} \hat {\bar {R}} _ {\lambda_ {\theta}} (s _ {t}, \mu_ {\theta_ {t} ^ {- i}} ^ {- i} (s _ {t}), a _ {i}) \right| _ {a _ {i} = \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t})} = \left. \nabla_ {a _ {i}} w (s _ {t}, \mu_ {\theta_ {t} ^ {- i}} ^ {- i} (s _ {t}), a _ {i}) \right| _ {a _ {i} = \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t})} \lambda_ {\theta} \\ \xi_ {t} ^ {i} = \left. \nabla_ {a _ {i}} \hat {\bar {R}} _ {\lambda_ {t} ^ {i}} (s _ {t}, \mu_ {\theta_ {t} ^ {- i}} ^ {- i} (s _ {t}), a _ {i}) \right| _ {a _ {i} = \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t})} = \left. \nabla_ {a _ {i}} w (s _ {t}, \mu_ {\theta^ {- i}} ^ {- i} (s _ {t}), a _ {i}) \right| _ {a _ {i} = \mu_ {\theta^ {i}} ^ {i} (s _ {t})} \lambda_ {t} ^ {i} \\ \end{array}
$$

and we keep

$$
\psi_ {t, \theta} ^ {i} = \nabla_ {\theta^ {i}} \mu_ {\theta^ {i}} ^ {i} (s _ {t}), \quad \mathrm {a n d} \quad \psi_ {t} ^ {i} = \psi_ {t, \theta_ {t}} ^ {i} = \nabla_ {\theta^ {i}} \mu_ {\theta_ {t} ^ {i}} ^ {i} (s _ {t}).
$$

Theorem 7. Under Assumptions 1, 3, 4, and 6, the policy parameter  $\theta_t^i$  obtained from (6) converges a.s. to a point in the asymptotically stable equilibria of

$$
\dot {\theta} ^ {i} = \Gamma^ {i} \left[ \mathbb {E} _ {s \sim d ^ {\pi}} \left[ \psi_ {t, \theta} ^ {i} \cdot \xi_ {t, \theta} ^ {i} \right] \right]. \tag {14}
$$

We define compatible features for the action-value and the average-reward function in an analogous manner:  $w_{\theta}(s,a) = (a - \mu_{\theta}(s))\cdot \nabla_{\theta}\mu_{\theta}(s)^{\top}$ . For  $\lambda \in \mathbb{R}^m$ ,

$$
\hat {\bar {R}} _ {\lambda , \theta} (s, a) = (a - \mu_ {\theta} (s)) \cdot \nabla_ {\theta} \mu_ {\theta} (s) ^ {\top} \cdot \lambda
$$

$$
\nabla_ {a} \hat {\dot {R}} _ {\lambda , \theta} (s, a) = \nabla_ {\theta} \mu_ {\theta} (s) ^ {\top} \cdot \lambda
$$

and we have that, for  $\lambda^{*} = \operatorname *{argmin}_{\lambda}\mathbb{E}_{s\sim d^{\pi}}\big[\| \nabla_{a}\hat{\bar{R}}_{\lambda ,\theta}(s,\mu_{\theta}(s)) - \nabla_{a}\bar{R} (s,\mu_{\theta}(s))\|^{2}\big]$ :

$$
\nabla_ {\theta} J _ {\pi} (\mu_ {\theta}) = \mathbb {E} _ {s \sim d ^ {\pi}} \left[ \nabla_ {\theta} \mu_ {\theta} (s) \cdot \nabla_ {a} \bar {R} (s, a) \Big | _ {a = \mu_ {\theta} (s)} \right] = \mathbb {E} _ {s \sim d ^ {\pi}} \left[ \nabla_ {\theta} \mu_ {\theta} (s) \cdot \nabla_ {a} \hat {\bar {R}} _ {\lambda^ {*}, \theta} (s, a) \Big | _ {a = \mu_ {\theta} (s)} \right].
$$

The use of compatible features requires each agent to observe not only the joint action taken  $a_{t+1} = (a_{t+1}^1, \ldots, a_{t+1}^N)$  and the "on-policy action"  $\dot{a}_{t+1} = (\dot{a}_{t+1}^1, \ldots, \dot{a}_{t+1}^N)$ , but also  $\widetilde{a}_{t+1} = (\nabla_{\theta^1} \mu_{\theta_t^1}^1(s_{t+1}), \ldots, \nabla_{\theta^N} \mu_{\theta_t^N}^N(s_{t+1}))$  (see the parts in blue in Algorithm 2).

We illustrate algorithm convergence on multi-agent extension of a continuous bandit problem from Sec. 5.1 of Silver et al. (January 2014b). Details are in the Appendix. Figure 2 shows the convergence of Algorithms 1 and 2 averaged over 5 runs. In all cases, the system converges and the agents are able to coordinate their actions to minimize system cost.

![](images/ec56991e62aefdc8563fa9ed54144af47f667c0dc2de9bdccdaf4a3bb9212be6.jpg)  
Figure 1: Convergence of Algorithms 1 and 2 on the multi-agent continuous bandit problem.

![](images/b031de9b6a1b3ddf21ec1ca6981265cace8fcf4f60f6ee986bae05007bc96d24.jpg)

![](images/eabd662400d2264c3b9e124f916734721c965371ba9861272fa0fc52c68b58a3.jpg)

# 6 CONCLUSION

We have provided the tools needed to implement decentralized, deterministic actor-critic algorithms for cooperative multi-agent reinforcement learning. We provide the expressions for the policy gradients, the algorithms themselves, and prove their convergence in on-policy and off-policy settings. We also provide numerical results for a continuous multi-agent bandit problem that demonstrates the convergence of our algorithms. Our work differs from Zhang & Zavlanos (2019) as the latter was based on policy consensus whereas ours is based on critic consensus. Our approach represents agreement between agents on every participants' contributions to the global reward, and as such, provides a consensus scoring function with which to evaluate agents. Our approach may be used in compensation schemes to incentivize participation. An interesting extension of this work would be to prove convergence of our actor-critic algorithm for continuous state spaces, as it may hold with assumptions on the geometric ergodicity of the stationary state distribution induced by the deterministic policies (see Crowder (2009)). The expected policy gradient (EPG) of Ciosek & Whiteson (2018), a hybrid between stochastic and deterministic policy gradient, would also be interesting to leverage. The Multi-Agent Deep Deterministic Policy Gradient algorithm (MADDPG) of Lowe et al. (2017) assumes partial observability for each agent and would be a useful extension, but it is likely difficult to extend our convergence guarantees to the partially observed setting.

# REFERENCES

Albert Benveniste, Pierre Priouret, and Michel Métivier. Adaptive Algorithms and Stochastic Approximations. Springer-Verlag, Berlin, Heidelberg, 1990. ISBN 0-387-52894-6.  
Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, and Mark Lee. Natural actor-critic algorithms. Automatica, 45(11):2471-2482, November 2009. ISSN 0005-1098. doi: 10. 1016/j.automatica.2009.07.008. URL http://dx.doi.org/10.1016/j.automatica. 2009.07.008.  
Kamil Ciosek and Shimon Whiteson. Expected Policy Gradients for Reinforcement Learning. arXiv e-prints, art. arXiv:1801.03326, Jan 2018.  
Martin Crowder. Stochastic approximation: A dynamical systems viewpoint by Vivek s. borkar. International Statistical Review, 77(2):306-306, 2009.  
Thomas Degris, Martha White, and Richard S. Sutton. Off-policy actor-critic. CoRR, abs/1205.4839, 2012. URL http://arxiv.org/abs/1205.4839.  
Scott Fujimoto, Herke van Hoof, and Dave Meger. Addressing function approximation error in actor-critic methods. CoRR, abs/1802.09477, 2018. URL http://arxiv.org/abs/1802.09477.  
Sham Kakade. A natural policy gradient. In Proceedings of the 14th International Conference on Neural Information Processing Systems: Natural and Synthetic, NIPS'01, pp. 1531-1538, Cambridge, MA, USA, 2001. MIT Press. URL http://dl.acm.org/citation.cfm?id=2980539.2980738.  
Vijaymohan Konda. Actor-critic Algorithms. PhD thesis, Cambridge, MA, USA, 2002. AAI0804543.  
Harold J. (Harold Joseph) Kushner and (joint author.) Clark, Dean S. Stochastic approximation methods for constrained and unconstrained systems. New York: Springer-Verlag, 1978. ISBN 0387903410.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. CoRR, abs/1509.02971, 2015.  
Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. Neural Information Processing Systems (NIPS), 2017.  
Hamid Reza Maei. Convergent actor-critic algorithms under off-policy training and function approximation. CoRR, abs/1802.07842, 2018. URL http://arxiv.org/abs/1802.07842.  
P. Marbach and J. N. Tsitsiklis. Simulation-based optimization of markov reward processes. IEEE Transactions on Automatic Control, 46(2):191-209, Feb 2001. ISSN 0018-9286. doi: 10.1109/9.905687.  
K. J. Prabuchandran, Shalabh Bhatnagar, and Vivek S. Borkar. Actor-critic algorithms with online feature adaptation. ACM Trans. Model. Comput. Simul., 26(4):24:1-24:26, February 2016. ISSN 1049-3301. doi: 10.1145/2868723. URL http://doi.acm.org/10.1145/2868723.  
Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, Inc., New York, NY, USA, 1st edition, 1994. ISBN 0471619779.  
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic Policy Gradient Algorithms. International Conference on Machine Learning, pp. 387-395, January 2014a.  
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic Policy Gradient Algorithms. International Conference on Machine Learning, pp. 387-395, January 2014b.

Wesley Suttle, Zhuoran Yang, Kaiqing Zhang, Zhaoran Wang, Tamer Basar, and Ji Liu. A multi-agent off-policy actor-critic algorithm for distributed reinforcement learning. CoRR, abs/1903.06372, 2019. URL http://arxiv.org/abs/1903.06372.  
Richard S Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. A. Solla, T. K. Leen, and K. Müller (eds.), Advances in Neural Information Processing Systems 12, pp. 1057-1063. MIT Press, 2000a.  
Richard S Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. A. Solla, T. K. Leen, and K. Müller (eds.), Advances in Neural Information Processing Systems 12, pp. 1057-1063. MIT Press, 2000b.  
Richard S. Sutton, Hamid Reza Maei, Doina Precup, Shalabh Bhatnagar, David Silver, Csaba Szepesvári, and Eric Wiewiora. Fast gradient-descent methods for temporal-difference learning with linear function approximation. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML '09, pp. 993-1000, New York, NY, USA, 2009. ACM. ISBN 978-1-60558-516-1.  
Richard S. Sutton, A. Rupam Mahmood, and Martha White. An emphatic approach to the problem of off-policy temporal-difference learning. J. Mach. Learn. Res., 17(1):2603-2631, January 2016. ISSN 1532-4435. URL http://dl.acm.org/citation.cfm?id=2946645. 3007026.  
Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar. Fully decentralized multiagent reinforcement learning with networked agents. 80:5872-5881, 10-15 Jul 2018.  
Yan Zhang and Michael M. Zavlanos. Distributed off-policy actor-critic reinforcement learning with policy consensus. CoRR, abs/1903.09255, 2019.
