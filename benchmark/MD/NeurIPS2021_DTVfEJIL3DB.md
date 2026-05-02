# MADE: Exploration via Maximizing Deviation from Explored Regions

Anonymous Author(s)

Affiliation

Address

email

# Abstract

In online reinforcement learning (RL), efficient exploration remains particularly challenging in high-dimensional environments with sparse rewards. In low-dimensional environments, where tabular parameterization is possible, count-based upper confidence bound (UCB) exploration methods achieve minimax near-optimal rates. However, it remains unclear how to efficiently implement UCB in realistic RL tasks that involve non-linear function approximation. To address this, we propose a new exploration approach via maximizing the deviation of the occupancy of the next policy from the explored regions. We add this term as an adaptive regularizer to the standard RL objective to balance exploration vs. exploitation. We pair the new objective with a provably convergent algorithm, giving rise to a new intrinsic reward that adjusts existing bonuses. The proposed intrinsic reward is easy to implement and combine with other existing RL algorithms to conduct exploration. As a proof of concept, we evaluate the new intrinsic reward on tabular examples across a variety of model-based and model-free algorithms, showing improvements over count-only exploration strategies. When tested on navigation and locomotion tasks from MiniGrid and DeepMind Control Suite benchmarks, our approach significantly improves sample efficiency over state-of-the-art methods.

# 1 Introduction

Online RL is a useful tool for an agent to learn how to perform tasks, particularly when expert demonstrations are unavailable and reward information needs to be used instead [83]. To learn a satisfactory policy, an RL agent needs to effectively balance between exploration and exploitation, which remains a central question in RL [22, 14]. Exploration is particularly challenging in environments with sparse rewards. One popular approach to exploration is based on intrinsic motivation, often applied by adding an intrinsic reward (or bonus) to the extrinsic reward provided by the environment. In provable exploration methods, bonus often captures the value estimate uncertainty and then the agent takes an action that maximizes the upper confidence bound (UCB) [4, 7, 36, 42, 39]. In tabular setting, UCB bonuses are often constructed based on either Hoeffding's inequality, which only uses visitation count, or Bernstein's inequality, which uses value function variance in addition to count. The latter is proved to be minimax near-optimal in environments with bounded rewards [39, 59] as well as bounded total reward [101] and reward-free settings [58, 43, 40, 102]. It remains an open question how one can efficiently compute confidence bounds to construct UCB bonus in non-linear function approximation. Furthermore, Bernstein-style bonuses are often hard to compute in practice beyond tabular setting.

In practice, various approaches are proposed to design intrinsic rewards: visitation pseudo-counts bonuses estimate count-based UCB bonus using function approximation [9, 14]; curiosity-based bonuses seek states where model prediction error is high; uncertainty-based bonuses [70, 78] adopt

![](images/c72d9a3365b608485a6aad60a82bf433bab69029e0388c6bca8804ae0c553388.jpg)  
Figure 1: Normalized samples use of different methods with respect to MADE (smaller values are better). MADE consistency achieves a better sample efficiency compared to all other baselines. Infinity means the method fails to achieve maximum reward in given steps.

![](images/b4033aeca9cb98c4830aca2d0e447190758abbe0f356c01c6613898e844a3cb6.jpg)

enssembles of networks for estimating variance of the Q-function; empowerment-based approaches [46, 30, 76, 62] lead the agent to states over which the agent has control, and information gain bonuses [45] that reward the agent based on the information gain between state-action pairs and next states.

Although the performance of practical intrinsic rewards is good in certain domains, empirically they are observed to suffer from issues such as detachment, derailment, and catastrophic forgetting [2, 22]. Moreover, these methods usually lack a clear objective and can get stuck in local optimum [2]. Indeed, the impressive performance currently achieved by some deep RL algorithms often revolves around manually designing dense rewards [12], complicated exploration strategies utilizing a significant amount of domain knowledge [22], or operating in the known environment regime [79, 63].

Motivated by current practical challenges and the gap between theory and practice, we propose a new algorithm for exploration by maximizing deviation from explored regions. This yields a practical algorithm with strong empirical performance. To be specific, we make the following contributions:

1. Exploration via maximizing deviation Our approach is based on modifying the standard RL objective (i.e. the cumulative reward) by adding a regularizer that adaptively changes across iterations. The regularizer can be a general function depending on the state-action visitation density and previous state-action coverage. We then choose a particular regularizer that MAximizes the DEviation (MADE) of the next policy visitation  $d^{\pi}$  from the regions covered by prior policies  $\rho_{\mathrm{cov}}^{k}$ :

$$
L _ {k} \left(d ^ {\pi}\right) = J \left(d ^ {\pi}\right) + \tau_ {k} \sum_ {s, a} \sqrt {\frac {d ^ {\pi} (s , a)}{\rho_ {\operatorname {c o v}} ^ {k} (s , a)}}. \tag {1}
$$

Here,  $k$  is the iteration number,  $J(d^{\pi})$  is the standard RL objective, and the regularizer encourages  $d^{\pi}(s,a)$  to be large when  $\rho_{\mathrm{cov}}^{k}(s,a)$  is small. We give an algorithm for solving the regularized objective and prove that with access to an approximate planning oracle, it converges to the global optimum. We show that objective (1) results in an intrinsic reward that can be easily added to any RL algorithm to improve performance, as suggested by our empirical studies. Furthermore, the intrinsic reward applies a simple modification to the UCB-style bonus that considers prior visitations. This simple modification can also be added to existing bonuses in practice.  
2. Tabular studies In the special case of tabular parameterization, we show that MADE only applies some simple adjustments to the Hoeffding-style count-based bonus. We compare the performance of MADE to Hoeffding and Bernstein bonuses in three different RL algorithms, for the exploration task in stochastic diabolical bidirectional lock [2, 60], which has sparse rewards and local optima. Our results show that MADE robustly improves over the Hoeffding bonus and is competitive to the Bernstein bonus, across all three RL algorithms. Interestingly, MADE bonus and exploration strategy appear to be very close to Bernstein bonus, without computing or estimating variance, suggesting that MADE potentially captures some environmental structures. Additionally, we empirically show that MADE regularizer can improve the optimization rate in policy gradient methods.  
3. Experiments on MiniGrid and DeepMind Control Suite We empirically show that MADE works both for model-free (IMAPLA [24], RAD [49]) and model-based (Dreamer [31]) RL algorithms, greatly improving the sample efficiency over existing baselines. When tested in MiniGrid environments, MADE manages to converge with 2x to 5x fewer samples compared to SOTA method BeBold [100]. In DeepMind Control Suite [86], we build upon the model-free method RAD [49] and the model-based method Dreamer [31], improving the return up to 150 in 500K steps compared to baselines. Figure 1 shows normalized sample size to achieve maximum reward w.r.t. our algorithm.

# 2 Background

Markov decision processes. An infinite-horizon discounted MDP is described by a tuple  $M = (S, \mathcal{A}, P, r, \rho, \gamma)$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $P: S \times \mathcal{A} \mapsto \Delta(S)$  is the transition kernel,  $r: S \times \mathcal{A} \mapsto [0,1]$  is the (extrinsic) reward function,  $\rho: S \mapsto \Delta(S)$  is the initial distribution, and  $\gamma \in [0,1)$  is the discount factor. A stationary (stochastic) policy  $\pi \in \Delta(\mathcal{A} \mid \mathcal{S})$  specifies a distribution over actions in each state. Each policy  $\pi$  induces a visitation density over state-action pairs  $d^{\pi}: S \times \mathcal{A} \mapsto [0,1]$  defined as  $d_{\rho}^{\pi}(s, a) := (1 - \gamma) \sum_{t=0}^{\infty} \gamma^{t} \mathbb{P}_{t}(s_{t} = s, a_{t} = a; \pi)$ , where  $\mathbb{P}_{t}(s_{t} = s, a_{t} = a; \pi)$  denotes  $(s, a)$  visitation probability at step  $t$ , starting at  $s_{0} \sim \rho(\cdot)$  and following  $\pi$ . An important quantity is the value a policy  $\pi$ , which is the discounted sum of rewards  $V^{\pi}(s) := \mathbb{E}[\sum_{t=0}^{\infty} \gamma^{t} r_{t} \mid s_{0} = s, a_{t} \sim \pi(\cdot \mid s_{t})$  for all  $t \geq 0]$  starting at state  $s \in S$ .

Policy mixture. For a sequence of policies  $\mathcal{C}^k = (\pi_1, \dots, \pi_k)$  with corresponding weights  $w^k \in \Delta_{k-1}$ , the policy mixture  $\pi_{\mathrm{mix}, k} = (\mathcal{C}^k, w^k)$  is obtained by first sampling a policy from  $w^k$  and then following that policy over subsequent steps [32]. The mixture policy induces a state-action visitation density according to  $d^{\pi_{\mathrm{mix}}}(s, a) = \sum_{i=1}^{k} w_i^k d^{\pi_i}(s, a)$ . While the  $\pi_{\mathrm{mix}}$  may not be stationary in general, there exists a stationary policy  $\pi'$  such that  $d^{\pi'} = d^{\pi_{\mathrm{mix}}}$ ; see [73] for details.

Online reinforcement learning. Online RL is the problem of finding a policy with maximum value from an unknown MDP using samples collected during exploration. Oftentimes, the following objective is considered, which is a scalar summary of the performance of policy  $\pi$ :

$$
J _ {M} (\pi) := \mathbb {E} _ {s \sim \rho} [ V ^ {\pi} (s) ] = (1 - \gamma) ^ {- 1} \mathbb {E} _ {(s, a) \sim d _ {\rho} ^ {\pi} (\cdot , \cdot)} [ r (s, a) ]. \tag {2}
$$

We drop index  $M$  when it is clear from context. We denote an optimal policy by  $\pi^{\star} \in \arg \max_{\pi} J(\pi)$  and use the shorthand  $V^{\star} \coloneqq V^{\pi^{\star}}$  to denote the optimal value function. It is straightforward to check that  $J(\pi)$  can equivalently be represented by the expectation of the reward over the visitation measure of  $\pi$  and slightly abuse the notation and sometimes write  $J(d^{\pi})$ .

# 3 Adaptive regularization of the RL objective

# 3.1 Regularization to guide exploration

In online RL, the agent faces a dilemma in each state: whether it should select a seemingly optimal policy (exploit) or it should explore different regions of the MDP. To allow flexibility in this choice and trade-off between exploration and exploitation, we propose to add a regularizer to the standard RL objective that changes throughout iterations of an online RL algorithm:

$$
L _ {k} \left(d ^ {\pi}\right) = \underbrace {J \left(d ^ {\pi}\right)} _ {\text {e x p l o i t a t i o n}} + \tau_ {k} \underbrace {R \left(d ^ {\pi} ; \left\{d ^ {\pi_ {i}} \right\} _ {i = 1} ^ {k}\right)} _ {\text {e x p l o r a t i o n}}. \tag {3}
$$

Here,  $R(d^{\pi}; \{d^{\pi_i}\}_{i=1}^{k})$  is a function of state-action visitation of  $\pi$  as well as the visitation of prior policies  $\pi_1, \ldots, \pi_k$ . The temperature parameter  $\tau_k$  determines the strength of regularization. Objective (3) is a population objective in the sense that it does not involve empirical estimations affected by the randomness in sample collection. In the following section, we give our particular choice of regularizer and discuss how this objective can describe some popular exploration bonuses. We then provide a convergence guarantee for the regularized objective in Section 3.2.

# 3.2 Exploration via maximizing deviation from policy cover

We develop our exploration strategy MADE based on a simple intuition: maximizing the deviation from the explored regions, i.e. all states and actions visited by prior policies. We define policy cover at iteration  $k$  to be the density over regions explored by policies  $\pi_1, \ldots, \pi_k$ , i.e.  $\rho_{\mathrm{cov}}^k(s, a) := \frac{1}{k} \sum_{i=1}^{k} d^{\pi_i}(s, a)$ . We then design our regularizer to encourage  $d^\pi$  to be different from  $\rho_{\mathrm{cov}}^k$ :

$$
R _ {k} \left(d ^ {\pi}; \left\{d ^ {\pi_ {i}} \right\} _ {i = 1} ^ {k}\right) = \sum_ {s, a} \sqrt {\frac {d ^ {\pi} (s , a)}{\rho_ {\mathrm {c o v}} ^ {k} (s , a)}}. \tag {4}
$$

It is easy to check that the maximizer of above function is  $d^{\pi}(s,a) \propto \frac{1}{\rho_{\mathrm{cov}}^{k}(s,a)}$ . Our motivation behind this particular deviation is that it results in a simple modification of UCB bonus in tabular case.

We now compute the reward yielded by the new objective. First, define a policy mixture  $\pi_{\mathrm{mix},k}$  with policy sequence  $(\pi ,\dots ,\pi_{k})$  and geometric weights  $((1 - \eta)^{k - 1}\eta ,\dots ,\eta)$  for  $\eta >0$ . Let  $d^{\pi_{\mathrm{mix},k}}$  be the visitation density of  $\pi_{\mathrm{mix},k}$ . We compute the total reward at iteration  $k$  by taking the gradient of new objective with respect to  $d^{\pi}$  at  $d^{\pi_{\mathrm{mix},k}}$ :

$$
\left. r _ {k} (s, a) = \nabla_ {d} L _ {k} (d) \right| _ {d = d ^ {\pi_ {\operatorname {m i x}, k}}} = r (s, a) + \tau_ {k} \nabla_ {d} R _ {k} \left(d; \left\{d ^ {\pi_ {i}} \right\} _ {i = 1} ^ {k}\right) \Big | _ {d = d ^ {\pi_ {\operatorname {m i x}, k}}}, \tag {5}
$$

which gives the following reward

$$
r _ {k} (s, a) = r (s, a) + \frac {\tau_ {k} / 2}{\sqrt {d ^ {\pi_ {\operatorname* {m i x} , k}} (s , a) \rho_ {\operatorname* {c o v}} ^ {k} (s , a)}}. \tag {6}
$$

The intrinsic reward above is constructed based on two densities:  $\rho_{\mathrm{cov}}^k$  a uniform combination of past visitation densities and  $\hat{d}^{\pi_{\mathrm{mix},k}}$  a geometric mixture of past visitation densities. As we will discuss shortly, policy cover  $\rho_{\mathrm{cov}}^k (s,a)$  is related to the visitation count of  $(s,a)$  pair in previous iterations and resembles count-based bonuses [9, 39] or their approximates such as RND [14]. Therefore, for an appropriate choice of  $\tau_{k}$ , MADE's intrinsic reward decreases as the number of visitations increases.

MADE's intrinsic reward is also proportional to  $1 / \sqrt{d^{\pi_{\mathrm{mix},k}}(s,a)}$ , which can be viewed as a correction implied to the count-based bonus. In effect, due to the geometric weights in  $\pi_{\mathrm{mix},k}$ , the above construction gives a higher reward to  $(s,a)$  pairs visited earlier. Experimental results suggest that this correction may alleviate major difficulties in sparse reward exploration, namely detachment and catastrophic forgetting, by encouraging the agent to revisit forgotten states and actions.

Empirically, MADE's intrinsic reward is computed based on estimates  $\hat{d}^{\pi_{\mathrm{mix},k}}$  and  $\hat{\rho}_{\mathrm{cov}}^{k}$  from data collected by iteration  $k$ . Furthermore, practically we consider a smoothed version of the above regularizer by adding  $\lambda > 0$  to both numerator and denominator; see (7).

MADE intrinsic reward in tabular case. In tabular empirical setting, the empirical estimation of policy cover is simply  $\hat{\rho}_{\mathrm{cov}}^k (s,a) = \frac{N_k(s,a)}{N_k}$ , where  $N_{k}(s,a)$  is  $(s,a)$  pair's visitation count and  $N_{k}$  is the total count, by iteration  $k$ . Thus, MADE simply modifies the Hoeffding-type bonus via the mixture density and has the following form:  $\propto 1 / \sqrt{\hat{d}^{\pi_{\mathrm{mix},k}}(s,a)N_k(s,a)}$

Bernstein bonus is another tabular UCB bonus that modifies Hoeffding bonus via value function variance. Bernstein bonus is shown to improve over Hoeffding count-only bonus by exploiting additional environment structure [95] and close the gap between algorithmic upper bounds and information-theoretic limits up to logarithmic factors [101, 102]. However, a practical and efficient implementation of a bonus that exploits variance information in non-linear function approximation parameterization still remains an open question; see Section 6 for further discussion. On the other hand, our proposed modification based on mixture density can be easily and efficiently incorporated with non-linear parameterization.

Deriving some popular bonuses from regularization. We now discuss how the regularization in (3) can describe some popular bonuses. Exploration bonuses that only depend on state-action visitations can be expressed in the form (3) by setting the regularizer a linear function of  $d^{\pi}$  and the exploration bonus  $r_i(s, a)$ , i.e.,  $R_k(d^\pi; \{d^{\pi_i}\}_{i=1}^k) = \sum_{s,a} d^\pi(s,a) r_i(s,a)$ . It is easy to check that taking the gradient of the regularizer with respect to  $d^\pi$  recovers  $r_i(s,a)$ . As another example, one can set the regularizer to Shannon entropy  $R_k(d^\pi; \{d^{\pi_i}\}_{i=1}^k) = -\sum_{s,a} d^\pi(s,a) \log d^\pi(s,a)$ , which gives the intrinsic reward  $-\log d^\pi(s,a)$  (up to an additive constant) and recovers the result in [98].

# 3.3 Solving the regularized objective

We pair MADE's objective with the algorithm proposed by Hazan et al. [32] extended to the adaptive objective. We provide convergence guarantee for Algorithm 1 in the following theorem whose proof is given in Appendix A.

Theorem 1 Consider the following regularization for  $\lambda >0$  for a valid visitation density  $d$

$$
R _ {\lambda} \left(d; \left\{\pi_ {i} \right\} _ {i = 1} ^ {k - 1}\right) = \frac {1}{k ^ {c}} \sum_ {s, a} \sqrt {\frac {d (s , a) + \lambda}{\rho_ {c o v} (s , a) + \lambda}}, \tag {7}
$$

Algorithm 1 Policy computation for adaptively regularized objective  
1: Inputs: Iteration count  $K$ , planning error  $\epsilon_{p}$ , visitation density error  $\epsilon_{d}$ .  
2: Initialize policy mixture  $\pi_{\mathrm{mix},1} =$  with  $\mathcal{C}_1 = (\pi_1)$  and  $w^{1} = (1)$   
3: for  $k = 1,\dots ,K$  do  
4: Estimate the visitation density  $\hat{d}^{\pi_{\mathrm{mix},k}}$  of  $\pi_{\mathrm{mix},k}$  via a visitation density oracle.  
5: Compute reward  $r_k(s,a) = r(s,a) + \tau_k\nabla_dR_k(d;\{\pi_i\}_{i = 1}^k)\big|_{d = \hat{d}^{\pi_{\mathrm{mix},k}}}$ .  
6: Run approximate planning on modified MDP  $M^{k} = (S,\mathcal{A},P,r_{k},\gamma)$  and return  $\pi_{k + 1}$ .  
7: Update policy mixture  $\mathcal{C}^{k + 1} = (C_k,\pi_{k + 1})$  and  $w^{k + 1} = ((1 - \eta)w^{k},\eta)$ .  
8: Return:  $\pi_{\mathrm{mix},K} = (\mathcal{C}^k,w^k)$ .

Then, for any  $\epsilon > 0$ , there exists  $\eta, \epsilon_p, \epsilon_d, c, B$  such that  $\pi_{\mathrm{mix},K}$  returned by Algorithm 1 after  $K \geq \eta^{-1} \log(10B\epsilon^{-1})$  iteration satisfies  $L_k(d^{\pi_{\mathrm{mix},K}}) \geq \max_{\pi} L_k(d^{\pi}) - \epsilon$ .

Remark 1 One does not need to maintain the functional forms of past policies to estimate  $\hat{d}^{\pi_{\mathrm{mix},k}}$ . Practically, one may truncate the dataset to a (prioritized) buffer and estimate the density over that.

# 4 A tabular study

# 4.1 Exploration in bidirectional lock

We consider a stochastic version of bidirectional diabolical combination lock (Figure 2), which is considered a particularly difficult exploration task in tabular setting [60, 2]. This environment is challenging because: (1) positive rewards are sparse, (2) a small negative reward is given when transiting to a good state and thus, moving to dead states is locally optimal, and (3) the agent may forget to explore one chain and get stuck in local minima upon receiving an end reward [2].

RL algorithms and exploration strategies. We compare Hoeffding and Bernstein bonuses [39] with MADE in three different RL algorithms. To implement MADE in tabular setting, we simply use two buffers: one stores all past state-action pairs to estimate  $\rho_{\mathrm{cov}}$ ; another one only maintains the most recent  $B$  pairs to estimate  $d_{\mu}^{\pi}$ . We use empirical counts to estimate both densities, which give a bonus  $\propto 1 / \sqrt{N_k(s,a)B_k(s,a)}$ , where  $N_{k}(s,a)$  is the total count and  $B_{k}(s,a)$  is the recent buffer count of  $(s,a)$  pair. We combine three bonuses with three RL algorithms: (1) value iteration with bonus [33], (2) proximal policy optimization with a model [15], and (3) Q-learning with bonus [39].

Results. Figure 2 summarizes our results showing MADE improves over Hoeffding bonus and is competitive to the Bernstein bonus in all three algorithms. Unlike Bernstein bonus that is hard to compute beyond tabular setting, MADE design is simple and can be effectively combined with any deep RL algorithms. The experimental results suggest several interesting properties for MADE. First, MADE applies a simple modification to Hoeffding bonus which improves the performance. Second, as illustrated in Figure 3, bonus values and exploration pattern of MADE is somewhat similar to the Bernstein bonus. This suggests that MADE may capture some structural information of the environment, similar to Bernstein bonus [95].

![](images/63b7fc98c8f579dd5eaf88c5fc5d01487d9e317cff96816d3e510c8cab079362.jpg)  
Figure 2: In a stochastic bidirectional lock, the agent starts at  $s_0$  and enters one of the chains based on the selected action. Each chain has a positive reward at the end,  $H$  good states, and  $H$  dead states. Both actions available to the agent lead it to the dead state, one with probability one and the other with probability  $p < 1$ . MADE performs better than Hoeffding-style bonus and comparable to Bernstein-style bonus across all three RL algorithms.

![](images/0462204d84717f13f8f5ef35ecc01279101682ed9f3e5871376b3dda24681c83.jpg)  
Figure 3: Hoeffding, Bernstein, and MADE intrinsic rewards over iterations in a bidirectional lock.

# 4.2 Policy gradient in chain MDP

We consider the chain MDP (Figure 4) presented in [1], which suffers from vanishing gradients with policy gradient approach as a positive reward is only achieved if the agent always takes action  $a_1$  [84]. This leads to an exponential iteration complexity lower bound even with access to exact gradients [1]. In this environment, the agent always starts at state  $s_0$ . Therefore, recent guarantees on the global convergence of exact policy gradients whose rate depends on the distribution shift [10, 1, 56], i.e. the ratio of the optimal visitation density and initial distribution, are vacuous.

RL algorithms. Since our goal in this experiment is to investigate the optimization effects and not exploration, we assume access to exact gradients. = In this setting, we consider MADE regularizer with the form  $\sum_{s,a}\sqrt{d^{\pi}(s,a)}$ . Note that policy gradients take gradient of the objective w.r.t. policy parameters  $\theta$  not  $d^{\pi}$ . We compare optimizing the policy gradient objective with four methods: vanilla version PG (e.g. uses policy gradient theorem [90, 84, 47]), relative policy entropy regularization PG+RE [1], policy entropy regularization PG+E [61, 56], and MADE regularization.

Results. Figure 4 illustrates our results on policy gradient methods. As expected [1], the vanilla version has a very slow convergence rate. Both entropy and relative entropy regularization are proved to achieve a linear convergence rate of  $\exp(-t)$  in the iteration count  $t$  [56, 1]. Interestingly, MADE seems to also enjoy a similar rate and outperforms the policy entropy regularizers.

# 5 Experiments on MiniGrid and DeepMind Control Suite

In addition to tabular setting, MADE can also be integrated with various model-free and model-based Deep RL training algorithms like IMPALA [24], RAD [51] and Dreamer [31]. The performance of MADE on MiniGrid [18] and DeepMind Control Suite [86] achieve SOTA sample efficiency.

To estimate  $N_{k}(s,a)$  and  $d(s,a)$ , we adopt the two buffer idea from tabular setting. However, since now the state space is high-dimensional, we use RND [14] to estimate  $N_{k}(s,a)$  and use a VAE to estimate  $d(s,a)$ . Specifically, for RND, we minimize the difference between a predictor network  $\phi^{\prime}(s,a)$  and a randomly initialized target network  $\phi (s,a)$  and train it in an online manner as the agent collects data. For VAE, we sample from the recent buffer  $B$  to train a VAE. The length of  $B$  is a design choice for which we do an ablation study. The intrinsic reward takes the form:  $\tau_{k}\| \phi (s,a) - \phi^{\prime}(s,a)\| \sqrt{1 / d(s,a)}$ .

Model-free RL baselines. We consider several baselines in MiniGrid: IMPALA [24] is a variant of policy gradient algorithms which we as the training baseline; ICM [69] learns a forward and

![](images/28e9c5d819c65a3a84c21c3eae8139936f97b6fdf10cc310858182443d857c9e.jpg)

![](images/28f252b393cf4a21431757d75d2617c4d2ac8531fdd31bfac65beebf24f7146d.jpg)  
Figure 4: A deterministic chain MDP that suffers from vanishing gradients [1]. We consider a constrained tabular policy parameterization with  $\pi(a|s) = \theta_{s,a}$  and  $\sum_{a} \theta_{s,a} = 1$ . The agent always starts from  $s_0$  and the only non-zero reward is  $r(s_{H+1}, a_1) = 1$ .

![](images/fef90799917d218fd19510187de22b6d2fa08332708763b899d7808ea34954de.jpg)  
Figure 5: Results for various hard exploration environments from MiniGrid. MADE successfully solves all the environments while all the baselines except for BeBold manage to solve all of them. MADE finds the optimal solution using  $2\mathrm{x} - 5\mathrm{x}$  fewer samples, yields a much better sample efficiency.

reverse model for predicting state transition and uses the forward model prediction error as intrinsic reward; RND [14] works as aforementioned; RIDE [74] learns a representation similar to ICM and uses the difference of learned representation along a trajectory as intrinsic reward; AMIGo [16] learns a teacher agent to assign intrinsic reward; BeBold [100] adopts a regulated difference of novelty measure using RND. In DeepMind Control Suite, we consider RE3 [77] as a baseline which uses a random encoder for state embedding followed by a  $k$ -nearest neighbour bonus for a maximum state coverage objective.

Model-based RL baselines. MADE can be combined with model-based RL algorithms and improve sample efficiency. For baselines: Dreamer is a well-known model-based RL algorithm used in DeepMind Control Suite; Dreamer+RE3 build upon Dreamer plus RE3 bonus.

MADE achieves strong SOTA results on both tasks, greatly improving the sample efficiency of the RL exploration in both model-free and model-based methods. Further details on experiments and exact hyperparameters are provided in Appendix B.

# 5.1 Model-free RL on MiniGrid

MiniGrid [18] is a widely used benchmark for exploration in RL. Despite having symbolic states and a discrete action space, MiniGrid tasks are quite challenging: the easiest task is MultiRoom(MR) in which the agent needs to navigate to the goal by going to different rooms connected by the doors; in KeyCorridor(KC), the agent needs to search around different rooms to find the key and then use it to open the door; in ObstructedMaze(OM) is a harder version of Kc where the key is hidden in a box and sometimes the door is blocked by an obstruct. In addition to that, the entire environment is procedurally-generated. This adds another layer of difficulty to the problem.

From Figure 5 we can see that MADE manages to solve all the challenging tasks within 90M steps while all the baseline (except BeBold) only solves up to  $50\%$  of them. Compared to BeBold, MADE uses significantly fewer  $(2\mathbf{x}$  to  $5\mathbf{x}$  reduction) samples.

# 5.2 Model-free RL on DeepMind Control

We also test MADE on image-based continuous control tasks of DeepMind Control Suite [86], which is a collection of diverse control tasks such as Pendulum, Hopper, and Acrobot with realistic simulations. Compared to MiniGrid, these tasks are more realistic and complex as they involve stochastic transitions, high-dimensional states, and continuous actions. For baselines, we build our algorithm on top of RAD [51], a strong model-free RL algorithm with a competitive sample efficiency. We compare with ICM, RND, and RE3 [77] (the SOTA algorithm)  $^{1}$ . Note that we compare MADE

![](images/2b43b125eb5bc43fc06eaa2cf5387b18def5b4003415bbe2a4bf35fc8416a4b4.jpg)  
Figure 6: Results for several DeepMind control suite locomotion tasks. Comparing to all baselines, the performance of MADE is consistently better. Sometimes baseline methods even fail to solve the task.

![](images/2f23cdfabe18f42da4491978ce26bf3bddd1a4ba66d50d839d3d75a0aaaf97cf.jpg)  
Figure 7: Ablation study on buffer size in MADE. The optimal buffer size varies in different tasks. We found buffer size of 10000 empirically works consistently reasonable.

to very strong baselines; other algorithms (e.g., DrQ [48], CURL [80], ProtoRL [94], SAC+AE [93]) perform worse.2 MADE show consistent improvement in sample efficiency: 2.6x over RAD+RE3, 3.3x over RAD+RND, 19.7x over CURL, 15.0x over DrQ and 3.8x over RAD.

From Figure 6, we can see that MADE consistently improves sample efficiency compared to all baselines. For these tasks, RND and ICM do not perform well and even fail on Cartpole-Swingup. RE3 achieves a comparable performance in two tasks, however, its performance on Pendulum-Swingup, Quadruped-Run, Hopper-Hop and Walker-Run is significantly worse than MADE. For example, in Pendulum-Swingup, MADE achieves a reward of around 800 in only 30K steps while RE3 requires 300k samples. In Quadruped-Run, there is a 150 reward gap between MADE and RE3 and it seems to be still enlarging. These tasks show the strong performance of MADE in model-free RL.

Ablation study. We study how the buffer length affects the performance of our algorithm in some DeepMind Control tasks. Results show that for different tasks the optimal length is slightly different. We empirically found that using a buffer length of 1000 consistently works well across different tasks.

# 5.3 Model-based RL on DeepMind Control

We also empirically verify the performance of MADE combined with the SOTA model-based RL algorithm Dreamer [31]. We compare MADE with Dreamer and Dreamer combined with RE3 in Figure 8. Results show that MADE has great sample efficiency in maps like Cheetah-Run-Sparse, Hopper-Hop and Pendulum-Swingup. For example, in Hopper-Hop, MADE achieves more than 100 higher return than RE3 and 250 higher than Dreamer, achieving a new SOTA result.

# 6 Related work

Provable optimistic exploration. Most provable exploration strategies are based on optimism in the face of uncertainty (OFU) principle. In tabular setting, model-based exploration algorithms include variants of UCB [44, 11], UCRL [50, 36, 95, 43, 58], and Thompson sampling [91, 4, 75] and value-based methods include optimistic Q-learning [39, 89, 81, 54, 59] and value-iteration with UCB [7, 101, 102, 40]. These methods are recently extended to linear MDP setting leading to a variety of model-based [103, 6, 37, 104], value-based [88, 41], and policy-based algorithms [15, 97, 2]. Going beyond linear function approximation, systematic exploration strategies are developed based on

![](images/c5dea02b9456973e27b40e115566d154cdd3d39518f1a9f087a9680c20fdb715.jpg)

![](images/828fb25e69ec9d1d6f403f8823a74ca2517a0cdb81c1da17ab501206990a1cc6.jpg)

![](images/8db842da163260f739bcdecacc732a55f6992239d3112639b302a9b9f3d28d06.jpg)  
Figure 8: Results for DeepMind control suite locomotion tasks in model-based RL setting. Comparing to all baselines, the performance of MADE is consistently better. Some baseline methods even fail to solve the task.

274 structural assumptions on MDP such as low Bellman rank [38] and block MDP [21]. These methods are either computationally intractable [38, 82, 6, 96, 92, 20, 87] or are only oracle efficient [25, 3]. The recent work [26] provides a sample efficient approach with non-linear policies, however, the algorithm requires maintaining the functional form of all prior policies.

Practical exploration via intrinsic reward. Apart from previously-discussed methods, other works give intrinsic reward based on the difference in (abstraction of) consecutive states [99, 55, 74]. However, this approach is inconsistent: the intrinsic reward does not converge to zero and thus, even with infinite samples, the final policy does not maximize the RL objective. Other intrinsic rewards try to estimate pseudo-counts [9, 85, 14, 13, 68, 8], inspired by provable count-based methods. Though favoring novel states, practically these methods might suffer from detachment and derailment [22, 23], and forgetting [2]. More recent works propose a combination of different criteria. RIDE [74] learns a representation using curiosity criterion and uses the difference of consecutive states along the trajectory as the bonus. AMiGo [16] learns a teacher agent for assigning rewards for exploration. GoExplore [22] explicitly decouples the exploration and exploitation stage, yields a more sophisticated algorithm with many hand-tuned hyperparameters.

Maximum entropy exploration. Another line of work encourages exploration via maximizing some type of entropy. One category maximizes policy entropy [61] or relative entropy [1] in addition to the RL objective. Recently, effects of policy entropy regularization have been studied theoretically [65, 29] in policy gradient methods with access to exact gradients, showing better convergence rate by improving optimization landscape [56, 57, 5, 17]. Another category considers maximizing the entropy of state or state-action occupancy densities such as Shannon entropy [32, 35, 52, 77] or Renyi entropy [98]. Empirically, our approach achieves better performance over entropy-based methods.

Other exploration strategies. Besides intrinsic motivation, other strategies are also fruitful in encouraging the RL agent to visit a wide range of states. One example is exploration by injecting noise to the action action space [53, 66, 34, 67] or parameter space [28, 71]. Another example is the reward-shaping category, in which diverse goals are set to guide exploration [19, 27, 64, 72].

# 7 Discussion

We introduce a new exploration strategy MADE based on maximizing deviation from explored regions. We show that by simply adding a regularizer to the original RL objective, we get an easy-to-implement intrinsic reward and it can be incorporated with any RL algorithm. We provide a policy computation algorithm for this objective and prove that it converges to a global optimum. In tabular setting, MADE consistently improves the Hoeffding-style bonus and shows competitive performance compared to Bernstein-style bonus, while the latter is impractical to compute beyond tabular. We conduct extensive experiments on MiniGrid, showing a significant reduction of the required sample size by over 5x. MADE also performs well in DeepMind Control Suite with both model-free and model-based RL algorithms, achieving SOTA sample efficiency results. One limitation of MADE is that it only uses the naive representations of states (e.g., one-hot representation in tabular case). In fact, the exploration could be much more efficient if MADE is implemented with a more compact representation of states. We leave such analysis to future work.

# References

[1] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. arXiv preprint arXiv:1908.00261, 2019.  
[2] Alekh Agarwal, Mikael Henaff, Sham Kakade, and Wen Sun. PC-PG: Policy cover directed exploration for provable policy gradient learning. arXiv preprint arXiv:2007.08459, 2020.  
[3] Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity and representation learning of low rank MDPs. arXiv preprint arXiv:2006.10814, 2020.  
[4] Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: Worst-case regret bounds. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pages 1184-1194, 2017.  
[5] Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding the impact of entropy on policy optimization. In International Conference on Machine Learning, pages 151-160. PMLR, 2019.  
[6] Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning, pages 463-474. PMLR, 2020.  
[7] Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning, pages 263-272. PMLR, 2017.  
[8] Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Bilal Piot, Steven Kapturowski, Olivier Tieleman, Martin Arjovsky, Alexander Pritzel, Andrew Bolt, et al. Never give up: Learning directed exploration strategies. arXiv preprint arXiv:2002.06038, 2020.  
[9] Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. In Advances in neural information processing systems, pages 1471-1479, 2016.  
[10] Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. arXiv preprint arXiv:1906.01786, 2019.  
[11] Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research, 3(Oct):213-231, 2002.  
[12] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.  
[13] Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A Efros. Large-scale study of curiosity-driven learning. arXiv preprint arXiv:1808.04355, 2018.  
[14] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.  
[15] Qi Cai, Zhuoran Yang, Chi Jin, and Zhaoran Wang. Provably efficient exploration in policy optimization. In International Conference on Machine Learning, pages 1283-1294. PMLR, 2020.  
[16] Andres Campero, Roberta Raileanu, Heinrich Kuttler, Joshua B Tenenbaum, Tim Roktaschel, and Edward Grefenstette. Learning with amigo: Adversarily motivated intrinsic goals. arXiv preprint arXiv:2006.12122, 2020.  
[17] Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. arXiv preprint arXiv:2007.06558, 2020.

[18] Maxime Chevalier-Boisvert, Lucas Willems, and Suman Pal. Minimalistic gridworld environment for openai gym. https://github.com/maximecb/gym-minigrid, 2018.  
[19] Cédric Colas, Pierre Fournier, Mohamed Chetouani, Olivier Sigaud, and Pierre-Yves Oudeyer. CURIOUS: Intrinsically motivated modular multi-goal reinforcement learning. In International conference on machine learning, pages 1331–1340. PMLR, 2019.  
[20] Kefan Dong, Jiaqi Yang, and Tengyu Ma. Provable model-based nonlinear bandit and reinforcement learning: Shelve optimism, embrace virtual curvature. arXiv preprint arXiv:2102.04168, 2021.  
[21] Simon Du, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal, Miroslav Dudik, and John Langford. Provably efficient RL with rich observations via latent state decoding. In International Conference on Machine Learning, pages 1665-1674. PMLR, 2019.  
[22] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Go-exlore: a new approach for hard-exploration problems. arXiv preprint arXiv:1901.10995, 2019.  
[23] Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O Stanley, and Jeff Clune. First return then explore. arXiv preprint arXiv:2004.12919, 2020.  
[24] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. arXiv preprint arXiv:1802.01561, 2018.  
[25] Fei Feng, Ruosong Wang, Wotao Yin, Simon S Du, and Lin Yang. Provably efficient exploration for reinforcement learning using unsupervised learning. Advances in Neural Information Processing Systems, 33, 2020.  
[26] Fei Feng, Wotao Yin, Alekh Agarwal, and Lin F Yang. Provably correct optimization and exploration with non-linear policies. arXiv preprint arXiv:2103.11559, 2021.  
[27] Carlos Florensa, David Held, Xinyang Geng, and Pieter Abbeel. Automatic goal generation for reinforcement learning agents. In International conference on machine learning, pages 1515-1528. PMLR, 2018.  
[28] Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, et al. Noisy networks for exploration. International Conference on Learning Representations, 2018.  
[29] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized Markov decision processes. In International Conference on Machine Learning, pages 2160-2169. PMLR, 2019.  
[30] Karol Gregor, Danilo Jimenez Rezende, and Daan Wierstra. Variational intrinsic control. arXiv preprint arXiv:1611.07507, 2016.  
[31] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.  
[32] Elad Hazan, Sham Kakade, Karan Singh, and Abby Van Soest. Provably efficient maximum entropy exploration. In International Conference on Machine Learning, pages 2681-2691, 2019.  
[33] Jiafan He, Dongruo Zhou, and Quanquan Gu. Nearly minimax optimal reinforcement learning for discounted MDPs. arXiv preprint arXiv:2010.00587, 2020.  
[34] Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. arXiv preprint arXiv:1710.02298, 2017.  
[35] Riashat Islam, Zafarali Ahmed, and Doina Precup. Marginalized state distribution entropy regularization in policy optimization. arXiv preprint arXiv:1912.05128, 2019.

[36] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(4), 2010.  
[37] Zeyu Jia, Lin Yang, Csaba Szepesvari, and Mengdi Wang. Model-based reinforcement learning with value-targeted regression. In Learning for Dynamics and Control, pages 666-686. PMLR, 2020.  
[38] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low Bellman rank are PAC-learnable. In International Conference on Machine Learning, pages 1704–1713. PMLR, 2017.  
[39] Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is Q-learning provably efficient? In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 4868-4878, 2018.  
[40] Chi Jin, Akshay Krishnamurthy, Max Simchowitz, and Tiancheng Yu. Reward-free exploration for reinforcement learning. In International Conference on Machine Learning, pages 4870-4879. PMLR, 2020.  
[41] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory, pages 2137-2143. PMLR, 2020.  
[42] Sham Kakade, Mengdi Wang, and Lin F Yang. Variance reduction methods for sublinear reinforcement learning. arXiv preprint arXiv:1802.09184, 2018.  
[43] Emilie Kaufmann, Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Edouard Leurent, and Michal Valko. Adaptive reward-free exploration. In Algorithmic Learning Theory, pages 865-891. PMLR, 2021.  
[44] Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time. Machine learning, 49(2):209-232, 2002.  
[45] Hyoungseok Kim, Jaekyeom Kim, Yeonwoo Jeong, Sergey Levine, and Hyun Oh Song. Emi: Exploration with mutual information. arXiv preprint arXiv:1810.01176, 2018.  
[46] Alexander S Klyubin, Daniel Polani, and Chrystopher L Nehaniv. All else being equal be empowered. In European Conference on Artificial Life, pages 744-753. Springer, 2005.  
[47] Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In Advances in neural information processing systems, pages 1008-1014. Citeseer, 2000.  
[48] Ilya Kostrikov, Denis Yarats, and Rob Fergus. Image augmentation is all you need: Regularizing deep reinforcement learning from pixels. arXiv preprint arXiv:2004.13649, 2020.  
[49] Michael Laskin, Kimin Lee, Adam Stooke, Lerrel Pinto, Pieter Abbeel, and Aravind Srinivas. Reinforcement learning with augmented data. arXiv preprint arXiv:2004.14990, 2020.  
[50] Tor Lattimore and Marcus Hutter. PAC bounds for discounted MDPs. In International Conference on Algorithmic Learning Theory, pages 320-334. Springer, 2012.  
[51] Kimin Lee, Kibok Lee, Jinwoo Shin, and Honglak Lee. Network randomization: A simple technique for generalization in deep reinforcement learning. arXiv preprint arXiv:1910.05396, 2019.  
[52] Lisa Lee, Benjamin Eysenbach, Emilio Parisotto, Eric Xing, Sergey Levine, and Ruslan Salakhutdinov. Efficient exploration via state marginal matching. arXiv preprint arXiv:1906.05274, 2019.  
[53] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
[54] Shuang Liu and Hao Su. Regret bounds for discounted MDPs. arXiv preprint arXiv:2002.05138, 2020.

[55] Kenneth Marino, Abhinav Gupta, Rob Fergus, and Arthur Szlam. Hierarchical RL using an ensemble of proprioceptive periodic policies. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=SJz1x20cFQ.  
[56] Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, and Dale Schuurmans. On the global convergence rates of softmax policy gradient methods. In International Conference on Machine Learning, pages 6820-6829. PMLR, 2020.  
[57] Jincheng Mei, Yue Gao, Bo Dai, Csaba Szepesvari, and Dale Schuurmans. Leveraging non-uniformity in first-order non-convex optimization. arXiv preprint arXiv:2105.06072, 2021.  
[58] Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Emilie Kaufmann, Edouard Leurent, and Michal Valko. Fast active learning for pure exploration in reinforcement learning. arXiv preprint arXiv:2007.13442, 2020.  
[59] Pierre Menard, Omar Darwiche Domingues, Xuedong Shang, and Michal Valko. UCB momentum Q-learning: Correcting the bias without forgetting. arXiv preprint arXiv:2103.01312, 2021.  
[60] Dipendra Misra, Mikael Henaff, Akshay Krishnamurthy, and John Langford. Kinematic state abstraction and provably efficient rich-observation reinforcement learning. In International conference on machine learning, pages 6961–6971. PMLR, 2020.  
[61] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pages 1928-1937, 2016.  
[62] Shakir Mohamed and Danilo Jimenez Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. In Advances in neural information processing systems, pages 2125-2133, 2015.  
[63] Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, and Michael Bowling. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337):508-513, 2017.  
[64] Ashvin V Nair, Vitchyr Pong, Murtaza Dalal, Shikhar Bahl, Steven Lin, and Sergey Levine. Visual reinforcement learning with imagined goals. Advances in Neural Information Processing Systems, 31:9191-9200, 2018.  
[65] Gergely Neu, Anders Jonsson, and Vicenç Gomez. A unified view of entropy-regularized Markov decision processes. arXiv preprint arXiv:1705.07798, 2017.  
[66] Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. In Advances in neural information processing systems, pages 4026-4034, 2016.  
[67] Ian Osband, Benjamin Van Roy, Daniel J Russo, and Zheng Wen. Deep exploration via randomized value functions. Journal of Machine Learning Research, 20(124):1-62, 2019.  
[68] Georg Ostrovski, Marc G Bellemare, Aaron van den Oord, and Rémi Munos. Count-based exploration with neural density models. arXiv preprint arXiv:1703.01310, 2017.  
[69] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 16-17, 2017.  
[70] Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Self-supervised exploration via disagreement. arXiv preprint arXiv:1906.04161, 2019.  
[71] Matthias Plappert, Rein Houthooft, Prafulla Dhariwal, Szymon Sidor, Richard Y Chen, Xi Chen, Tamim Asfour, Pieter Abbeel, and Marcin Andrychowicz. Parameter space noise for exploration. International Conference on Learning Representations, 2018.

[72] Vitchyr Pong, Murtaza Dalal, Steven Lin, Ashvin Nair, Shikhar Bahl, and Sergey Levine. Skew-Fit: State-covering self-supervised reinforcement learning. In International Conference on Machine Learning, pages 7783-7792. PMLR, 2020.  
[73] Martin L Puterman. Markov decision processes. *Handbooks in operations research and management science*, 2:331-434, 1990.  
[74] Roberta Raileanu and Tim Rocktäschel. Ride: Rewarding impact-driven exploration for procedurally-generated environments. arXiv preprint arXiv:2002.12292, 2020.  
[75] Daniel Russo. Worst-case regret bounds for exploration via randomized value functions. arXiv preprint arXiv:1906.02870, 2019.  
[76] Christoph Salge, Cornelius Glackin, and Daniel Polani. Empowerment—an introduction. In *Guided Self-Organization: Inception*, pages 67–114. Springer, 2014.  
[77] Younggyo Seo, Lili Chen, Jinwoo Shin, Honglak Lee, Pieter Abbeel, and Kimin Lee. State entropy maximization with random encoders for efficient exploration. arXiv preprint arXiv:2102.09430, 2021.  
[78] Pranav Shyam, Wojciech Jaskowski, and Faustino Gomez. Model-based active exploration. In International Conference on Machine Learning, pages 5779-5788, 2019.  
[79] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of Go without human knowledge. nature, 550(7676):354-359, 2017.  
[80] Aravind Srinivas, Michael Laskin, and Pieter Abbeel. Curl: Contrastive unsupervised representations for reinforcement learning. arXiv preprint arXiv:2004.04136, 2020.  
[81] Alexander L Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael L Littman. PAC model-free reinforcement learning. In Proceedings of the 23rd international conference on Machine learning, pages 881-888, 2006.  
[82] Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based RL in contextual decision processes: PAC bounds and exponential improvements over model-free approaches. In Conference on Learning Theory, pages 2898-2933. PMLR, 2019.  
[83] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.  
[84] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Proceedings of the 12th International Conference on Neural Information Processing Systems, pages 1057-1063, 1999.  
[85] Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. In Advances in neural information processing systems, pages 2753-2762, 2017.  
[86] Yuval Tassa, Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Piotr Trochim, Siqi Liu, Steven Bohez, Josh Merel, Tom Erez, Timothy Lillicrap, et al. dm_control: Software and tasks for continuous control. arXiv preprint arXiv:2006.12983, 2020.  
[87] Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded Eluder dimension. Advances in Neural Information Processing Systems, 33, 2020.  
[88] Yining Wang, Ruosong Wang, Simon S Du, and Akshay Krishnamurthy. Optimism in reinforcement learning with generalized linear function approximation. arXiv preprint arXiv:1912.04136, 2019.  
[89] Yuanhao Wang, Kefan Dong, Xiaoyu Chen, and Liwei Wang. Q-learning with UCB exploration is sample efficient for infinite-horizon MDP. In International Conference on Learning Representations, 2019.

[90] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.  
[91] Zhihan Xiong, Ruoqi Shen, and Simon S Du. Randomized exploration is near-optimal for tabular MDP. arXiv preprint arXiv:2102.09703, 2021.  
[92] Zhuoran Yang, Chi Jin, Zhaoran Wang, Mengdi Wang, and Michael I Jordan. Bridging exploration and general function approximation in reinforcement learning: Provably efficient kernel and neural value iterations. arXiv preprint arXiv:2011.04622, 2020.  
[93] Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, and Rob Fergus. Improving sample efficiency in model-free reinforcement learning from images. arXiv preprint arXiv:1910.01741, 2019.  
[94] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Reinforcement learning with prototypical representations. arXiv preprint arXiv:2102.11271, 2021.  
[95] Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning, pages 7304-7312. PMLR, 2019.  
[96] Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent Bellman error. In International Conference on Machine Learning, pages 10978-10989. PMLR, 2020.  
[97] Andrea Zanette, Ching-An Cheng, and Alekh Agarwal. Cautiously optimistic policy optimization and exploration with linear function approximation. arXiv preprint arXiv:2103.12923, 2021.  
[98] Chuheng Zhang, Yuanying Cai, and Longbo Huang Jian Li. Exploration by maximizing Rényi entropy for reward-free rl framework. 2021.  
[99] Jingwei Zhang, Niklas Wetzel, Nicolai Dorka, Joschka Boedecker, and Wolfram Burgard. Scheduled intrinsic drive: A hierarchical take on intrinsically motivated exploration. arXiv preprint arXiv:1903.07400, 2019.  
[100] Tianjun Zhang, Huazhe Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph E Gonzalez, and Yuandong Tian. BeBold: Exploration beyond the boundary of explored regions. arXiv preprint arXiv:2012.08621, 2020.  
[101] Zihan Zhang, Xiangyang Ji, and Simon S Du. Is reinforcement learning more difficult than bandits? a near-optimal algorithm escaping the curse of horizon. arXiv preprint arXiv:2009.13503, 2020.  
[102] Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learning via reference-advantage decomposition. Advances in Neural Information Processing Systems, 33, 2020.  
[103] Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement learning for linear mixture Markov decision processes. arXiv preprint arXiv:2012.08507, 2020.  
[104] Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for discounted MDPs with feature mapping. arXiv preprint arXiv:2006.13165, 2020.
