# LEARNING SELF-CORRECTABLE POLICIES AND VALUE FUNCTIONS FROM DEMONSTRATIONS WITH NEGATIVE SAMPLING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Imitation learning, followed by reinforcement learning algorithms, is a promising paradigm to solve complex control tasks sample-efficiently. However, learning from demonstrations often suffers from the covariate shift problem, which results in cascading errors of the learned policy. We introduce a notion of conservatively-extrapolated value functions, which provably lead to policies with self-correction. We design an algorithm Value Iteration with Negative Sampling (VINS) that practically learns such value functions with conservative extrapolation. We show that VINS can correct mistakes of the behavioral cloning policy on simulated robotics benchmark tasks. We also propose the algorithm of using VINS to initialize a reinforcement learning algorithm, which is shown to outperform prior works in sample efficiency.

# 1 INTRODUCTION

Reinforcement learning (RL) algorithms, especially with sparse rewards, often require a large amount of trial-and-errors. Imitation learning from a small number of demonstrations followed by RL finetuning is a promising paradigm to improve the sample efficiency (Rajeswaran et al., 2017; Večerík et al., 2017; Hester et al., 2018; Nair et al., 2018; Gao et al., 2018).

The key technical challenge of learning from demonstrations is the covariate shift: the distribution of the states visited by the demonstrations often has a low-dimensional support; however, knowledge learned from this distribution may not necessarily transfer to other distributions of interests. This phenomenon applies to both learning the policy and the value function. The policy learned from behavioral cloning has compounding errors after we execute the policy for multiple steps and reach unseen states (Bagnell, 2015; Ross & Bagnell, 2010). The value function learned from the demonstrations can also extrapolate falsely to unseen states. See Figure 1a for an illustration of the false extrapolation in a toy environment.

We develop an algorithm that learns a value function that extrapolates to unseen states more conservatively, as an approach to attack the optimistic extrapolation problem (Fujimoto et al., 2018a). Consider a state  $s$  in the demonstration and its nearby state  $\tilde{s}$  that is not in the demonstration. The key intuition is that  $\tilde{s}$  should have a lower value than  $s$ , because otherwise  $\tilde{s}$  likely should have been visited by the demonstrations in the first place. If a value function has this property for most of the pair  $(s, \tilde{s})$  of this type, the corresponding policy will tend to correct its errors by driving back to the demonstration states because the demonstration states have locally higher values. We formalize the intuition in Section 4 by defining the so-called conservatively-extrapolated value function, which is guaranteed to induce a policy that stays close to the demonstrations states (Theorem 4.4).

In Section 5, we design a practical algorithm for learning the conservatively-extrapolated value function by a negative sampling technique inspired by work on learning embeddings Mikolov et al. (2013); Gutmann & Hyvarinen (2012). We also learn a dynamical model by standard supervised learning so that we compute actions by maximizing the values of the predicted next states. This algorithm does not use any additional environment interactions, and we show that it empirically helps correct errors of the behavioral cloning policy.

![](images/c30c2cd024b436a43ec80a3f59a920a91af915199ebe86bf73266fcd024f4189.jpg)  
(a) The value function learned from the standard Bellman equation (or supervised learning) on the demonstration states. The value function falsely extrapolates to the unseen states. For example, the top left corner has erroneously the largest value. As a result, once the policy induced by the value function makes a mistake, the error will compound.

![](images/d1f71c2e4c2fc6c69a446a0de946f952f8347479020cbd14c97ad95668d732a4.jpg)  
(b) The conservatively-extrapolated value function (defined in equation (4.2)) learned with negative sampling (VINS, Algorithm 2 in Section 5). The values at unseen states tend to be lower than their nearby states in the demonstrations, and therefore the corresponding policy tends to correct itself towards the demonstration trajectories.  
Figure 1: A toy environment where the agent aims to walk from a starting state (the yellow entry) to a goal state (the green entry). The reward is sparse:  $\bar{R}(s, a) = -1$  unless  $s$  is at the goal (which is also the terminal state.) The colors of the entries show the learned value functions. Entries in black edges are states in demonstrations. The cyan arrows show the best actions according to the value functions.

When additional environment interactions are available, we use the learned value function and the dynamical model to initialize an RL algorithm. This approach relieves the inefficiency in the prior work (Hester et al., 2018; Nair et al., 2018; Rajeswaran et al., 2017) that the randomly-initialized  $Q$  functions require a significant amount of time and samples to be warmed up, even though the initial policy already has a non-trivial success rate. Empirically, the proposed algorithm outperforms the prior work in the number of environment interactions needed to achieve near-optimal success rate.

In summary, our main contributions are: 1) we formalize the notion of values functions with conservative extrapolation which are proved to induce policies that stay close to demonstration states and achieve near-optimal performances, 2) we propose the algorithm Value Iteration with Negative Sampling (VINS) that outperforms behavioral cloning on three simulated robotics benchmark tasks with sparse rewards, and 3) we show that initializing an RL algorithm from VINS outperforms prior work in sample efficiency on the same set of benchmark tasks.

# 2 RELATED WORK

Imitation learning. Imitation learning is commonly adopted as a standard approach in robotics (Pomerleau, 1989; Schaal, 1997; Argall et al., 2009; Osa et al., 2017; Ye & Alterovitz, 2017; Aleotti & Caselli, 2006; Lawitzky et al., 2012; Torabi et al., 2018; Le et al., 2017; 2018) and many other areas such as playing games (Mnih et al., 2013). Behavioral cloning (Bain & Sommut, 1999) is one of the underlying central approaches. See Osa et al. (2018) for a thorough survey and more references therein. If we are allowed to access an expert policy (instead of trajectories) or an approximate value function, in the training time or in the phase of collecting demonstrations, then, stronger algorithms can be designed, such as DAgger (Ross et al., 2011), AggreVaTe (Ross & Bagnell, 2014), AggreVaTeD (Sun et al., 2017), DART (Laskey et al., 2017), THOR Sun et al. (2018a). Our setting is that we have only clean demonstrations trajectories and a sparse reward (but we still hope to learn the self-correctable policy.)

Ho & Ermon (2016); Wang et al. (2017); Schroecker et al. (2018) successfully combine generative models in the setting where a large amount of environment interaction without rewards are allowed. By contrast, we would like to minimize the amount of environment interactions needed, but are allowed to access a sparse reward. The work (Schroecker & Isbell, 2017) also aims to learn policies that can stay close to the demonstration sets, but through a quite different approach of estimating the true MAP estimate of the policy. The algorithm also requires environment interactions, whereas one of our main goals is to improve upon behavioral cloning without any environment interactions.

Inverse reinforcement learning (e.g., see (Abbeel & Ng, 2004; Ng et al., 2000; Ziebart et al., 2008; Finn et al., 2016a;b; Fu et al., 2017)) is another important and successful line of ideas for imitation

learning. It relates to our approach in the sense that it aims to learn a reward function that the expert is optimizing. In contrast, we construct a model to learn the value function (of the trivial sparse reward  $R(s,a) = -1$ ), rather than the reward function. Some of these works (e.g., (Finn et al., 2016a;b; Fu et al., 2017)) use techniques that are reminiscent of negative sampling or contrastive learning, although unlike our methods, they use "negative samples" that are sampled from the environments.

Leveraging demonstrations for sample-efficient reinforcement learning. Demonstrations have been widely used to improve the efficiency of RL (Kim et al., 2013; Chemali & Lazaric, 2015; Piot et al., 2014), and a common paradigm for continuous state and action space is to initialize with RL algorithms with a good policy or  $Q$  function (Rajeswaran et al., 2017; Nair et al., 2018; Večérík et al., 2017; Hester et al., 2018; Gao et al., 2018). We experimentally compare with the previous state-of-the-art algorithm in Nair et al. (2018) on the same type of tasks. Gao et al. (2018) has introduced soft version of actor-critic to tackle the false extrapolation of  $Q$  in the argument of  $a$  when the action space is discrete. In contrast, we deal with the extrapolation of the states in a continuous state and action space.

Model-based reinforcement learning. Even though we will learn a dynamical model in our algorithms, we do not use it to generate fictitious samples for planning. Instead, the learned dynamics are only used in combination with the value function to get a  $Q$  function. Therefore, we do not consider our algorithm as model-based techniques. We refer to (Kurutach et al., 2018; Clavera et al., 2018; Sun et al., 2018b; Chua et al., 2018; Sanchez-Gonzalez et al., 2018; Pascanu et al., 2017; Khansari-Zadeh & Billard, 2011; Luo et al., 2018) and the reference therein for recent work on model-based RL.

Off-policy reinforcement learning There is a large body of prior works in the domain of off-policy RL, including extensions of policy gradient (Gu et al., 2016; Degris et al., 2012; Wang et al., 2016) or Q-learning (Watkins & Dayan, 1992; Haarnoja et al., 2018; Munos et al., 2016). Fujimoto et al. (2018a) propose to solve off-policy reinforcement learning by constraining the action space, and Fujimoto et al. (2018b) use double Q-learning (Van Hasselt et al., 2016) to alleviate the optimistic extrapolation issue. In contrast, our method adjusts the erroneously extrapolated value function by explicitly penalizing the unseen states (which is customized to the particular demonstration off-policy data). For most of the off-policy methods, their convergence are based on the assumption of visiting each state-action pair sufficiently many times. In the learning from demonstration setting, the demonstrations states are highly biased or structured; thus off-policy method may not be able to learn much from the demonstrations.

# 3 PROBLEM SETUP AND CHALLENGES

We consider a setting with a deterministic MDP with continuous state and action space, and sparse rewards. Let  $S = \mathbb{R}^d$  be the state space and  $\mathcal{A} = \mathbb{R}^k$  be the action space, and let  $M^{\star}: \mathbb{R}^{d} \times \mathbb{R}^{k} \to \mathbb{R}^{d}$  be the deterministic dynamics. At the test time, a random initial state  $s_0$  is generated from some distribution  $D_{s_0}$ . We assume  $D_{s_0}$  has a low-dimensional bounded support because typically initial states have special structures. We aim to find a policy  $\pi$  such that executing  $\pi$  from state  $s_0$  will lead to a set of goal states  $\mathcal{G}$ . All the goal states are terminal states, and we run the policy for at most  $T$  steps if none of the goal states is reached.

Let  $\tau = (s_0, a_1, s_1, \ldots)$  be the trajectory obtained by executing a deterministic policy  $\pi$  from  $s_0$ , where  $a_t = \pi(s_t)$ , and  $s_{t+1} = M^{\star}(s_t, a_t)$ . The success rate of the policy  $\pi$  is defined as

$$
\operatorname {s u c c} (\pi) = \mathbb {E} \left[ \mathbb {1} \left\{\exists t \leq T, s _ {t} \in \mathcal {G} \right\} \right] \tag {3.1}
$$

where the expectation is taken over the randomness of  $s_0$ . Note that the problem comes with a natural sparse reward:  $R(s, a) = -1$  for every  $s$  and  $a$ . This will encourage reaching the goal with as small number of steps as possible: the total payoff of a trajectory is equal to negative the number of steps if the trajectory succeeds, or  $-T$  otherwise.

Let  $\pi_{\mathrm{e}}$  be an expert policy from which a set of  $n$  demonstrations are sampled. Concretely,  $n$  independent initial states  $\{s_0^{(i)}\}_{i = 1}^n$  from  $D_{s_0}$  are generated, and the expert executes  $\pi_{\mathrm{e}}$  to collect a set of  $n$  trajectories  $\{\tau^{(i)}\}_{i = 1}^n$ . We only have the access to the trajectories but not the expert policy itself.

We will design algorithms for two different settings:

Imitation learning without environment interactions: The goal is to learn a policy  $\pi$  from the demonstration trajectories  $\{\tau^{(i)}\}_{i=1}^n$  without having any additional interactions with the environment.

Leveraging demonstrations in reinforcement learning: Here, in addition to the demonstrations, we can also interact with the environment (by sampling  $s_0 \sim D_{s_0}$  and executing a policy) and observe if the trajectory reaches the goal. We aim is to minimize the amount of environment interactions by efficiently leveraging the demonstrations.

Let  $\mathcal{U}$  be the set of states that can be visited by the demonstration policy from a random state  $s_0$  with positive probability. Throughout this paper, we consider the situation where the set  $\mathcal{U}$  is only a small subset or a low-dimensional manifold of the entire states space. This is typical for continuous state space control problems in robotics, because the expert policy may only visit a very special kind of states that are the most efficient for reaching the goal. For example, in the toy example in Figure 1, the set  $\mathcal{U}$  only contains those entries with black edges. $^1$

To put our theoretical motivation in Section 4 into context, next we summarize a few challenges of imitation learning that are particularly caused by that  $\mathcal{U}$  is only a small subset of the state space.

Cascading errors for behavioral cloning. As pointed out by Bagnell (2015); Ross & Bagnell (2010), the errors of the policy can compound into a long sequence of mistakes and in the worst case cascade quadratically in the number of time steps  $T$ . From a statistical point of view, the fundamental issue is that the distribution of the states that a learned policy may encounter is different from the demonstration state distribution. Concretely, the behavioral cloning  $\pi_{\mathrm{BC}}$  performs well on the states in  $\mathcal{U}$  but not on those states far away from  $\mathcal{U}$ . However, small errors of the learned policy can drive the state to leave  $\mathcal{U}$ , and then the errors compound as we move further and further away from  $\mathcal{U}$ . As shown in Section 4, our key idea is to design policies that correct themselves to stay close to the set  $\mathcal{U}$ .

Degeneracy in learning value or  $Q$  functions from only demonstrations. When  $\mathcal{U}$  is a small subset or a low-dimensional manifold of the state space, off-policy evaluation of  $V^{\pi_{\mathrm{e}}}$  and  $Q^{\pi_{\mathrm{e}}}$  is fundamentally problematic in the following sense. The expert policy  $\pi_{\mathrm{e}}$  is not uniquely defined outside  $\mathcal{U}$  because any arbitrary extension of  $\pi_{\mathrm{e}}$  outside  $\mathcal{U}$  would not affect the performance of the expert policy (because those states outside  $\mathcal{U}$  will never be visited by  $\pi_{\mathrm{e}}$  from  $s_0 \sim D_{s_0}$ ). As a result, the value function  $V^{\pi_{\mathrm{e}}}$  and  $Q^{\pi_{\mathrm{e}}}$  is not uniquely defined outside  $\mathcal{U}$ . In Section 4, we will propose a conservative extrapolation of the value function that encourages the policy to stay close to  $\mathcal{U}$ . Fitting  $Q^{\pi_{\mathrm{e}}}$  is in fact even more problematic. We refer to Section A for detailed discussions and why our approach can alleviate the problem.

Success and challenges of initializing RL with imitation learning. A successful paradigm for sample-efficient RL is to initialize the RL policy by some coarse imitation learning algorithm such as BC (Rajeswaran et al., 2017; Večérík et al., 2017; Hester et al., 2018; Nair et al., 2018; Gao et al., 2018). However, the authors suspect that the method can still be improved, because the value function or the  $Q$  function are only randomly initialized so that many samples are burned to warm up them. As alluded before and shown in Section 4, we will propose a way to learn a value function from the demonstrations so that the following RL algorithm can be initialized by a policy, value function, and  $Q$  function (which is a composition of value and dynamical model) and thus converge faster.

# 4 THEORETICAL MOTIVATIONS

In this section, we formalize our key intuition that the ideal extrapolation of the value function  $V^{\pi_e}$  should be that the values should decrease as we get further and further from the demonstrations. Recall that we use  $\mathcal{U}$  to denote the set of states reachable by the expert policy from any initial state  $s_0$  drawn with positive probability from  $D_{s_0}$ . We use  $\| \cdot \|$  to denote a norm in Euclidean space  $\mathbb{R}^d$ . Let  $\Pi_{\mathcal{U}}(s)$  be the projection of  $s \in \mathbb{R}^d$  to a set  $\mathcal{U} \subset \mathbb{R}^d$  (according to the norm  $\| \cdot \|$ ).

We introduce the notion of value functions with conservative extrapolation which matches  $V^{\pi_{\mathrm{e}}}$  on the demonstration states  $\mathcal{U}$  and has smaller values outside  $\mathcal{U}$ . As formally defined in equation (4.1)

![](images/8037ce8d2d0485f834427f64528ec4723a90c8cdcbd5a38b658bf0227ad0612d.jpg)  
Figure 2: Illustration of the correction effect. A conservatively-extrapolated value function  $V$ , as shown in the figure, has lower values further away from  $\mathcal{U}$ , and therefore the gradients of  $V$  point towards  $\mathcal{U}$ . With such a value function, suppose we are at state  $s$  which is  $\varepsilon$ -close to  $\mathcal{U}$ . The locally-correctable assumption of the dynamics assumes the existence of  $a_{\mathrm{cx}}$  that will drive us to state  $s_{\mathrm{cx}}$  that is closer to  $\mathcal{U}$  than  $s$ . Since  $s_{\mathrm{cx}}$  has a relatively higher value compared to other possible future states that are further away from  $\mathcal{U}$  (e.g.,  $s'$  shown in the figure),  $s_{\mathrm{cx}}$  will be preferred by the optimization (4.3). In other words, if an action  $a$  leads to state  $s$  with large distance to  $\mathcal{U}$ , the action won't be picked by (4.3) because it cannot beat  $a_{\mathrm{cx}}$ .

and (4.2) in Alg. 1, we extrapolate  $V^{\pi_{\mathrm{e}}}$  in a way that the value at  $s \notin \mathcal{U}$  is decided by the value of its nearest neighbor in  $\mathcal{U}$  (that is  $V^{\pi_{\mathrm{e}}}\big(\Pi_{\mathcal{U}}(s)\big)$ , and its distance to the nearest neighbor (that is,  $\| s - \Pi_{\mathcal{U}}(s)\|$ ). We allow a  $\delta_V > 0$  error because exact fitting inside or outside  $\mathcal{U}$  would be impossible.

# Algorithm 1 Self-correctable policy induced from a value function with conservative extrapolation

Require: conservatively-extrapolated values  $V$  satisfying

$$
V (s) = V ^ {\pi_ {e}} (s) \pm \delta_ {V}, \quad \text {i f} s \in \mathcal {U} \tag {4.1}
$$

$$
V (s) = V ^ {\pi_ {e}} \left(\Pi_ {\mathcal {U}} (s)\right) - \lambda \| s - \Pi_ {\mathcal {U}} (s) \| \pm \delta_ {V} \quad \text {i f} s \notin \mathcal {U} \tag {4.2}
$$

and a locally approximately correct dynamics  $M$  and BC policy  $\pi_{\mathrm{BC}}$  satisfying Assumption (4.1).

Self-correctable policy  $\pi$ :

$$
\pi (s) \triangleq \underset {a: \| a - \pi_ {\mathrm {B C}} (s) \| \leq \zeta} {\operatorname {a r g m a x}} V (M (s, a)) \tag {4.3}
$$

Besides a conservatively-extrapolated value function  $V$ , our Alg. 1 relies on a learned dynamical model  $M$  and a behavioral cloning policy  $\pi_{\mathrm{BC}}$ . With these, the policy returns the action with the maximum value of the predicted next state in around the action of the BC policy. In other words, the policy  $\pi$  attempts to re-adjust the BC policy locally by maximizing the value of the next state.

Towards analyzing Alg. 1, we will make a few assumptions. We first assume that the BC policy is correct in the set  $\mathcal{U}$ , and the dynamical model  $M$  is locally correct around the set  $\mathcal{U}$  and the BC actions. Note that these are significantly weaker than assuming that the BC policy is globally correct (which is impossible to ensure) and that the model  $M$  is globally correct.

Assumption 4.1 (Local errors in learned dynamics and BC policy). We assume the BC policy  $\pi_{\mathrm{BC}}$  makes at most  $\delta_{\pi}$  error in  $\mathcal{U}$ : for all  $s \in \mathcal{U}$ , we have  $\| \pi_{\mathrm{BC}}(s) - \pi_{\mathrm{e}}(s) \| \leq \delta_{\pi}$ . We also assume that the learned dynamics  $M$  has  $\delta_M$  error locally around  $\mathcal{U}$  and the BC actions in the sense that for all  $s$  that is  $\varepsilon$ -close to  $\mathcal{U}$ , and any action that is  $\zeta$ -close to  $\pi_{\mathrm{BC}}(s)$ , we have  $\| M(s, a) - M^{\star}(s, a) \| \leq \delta_M$ .

We make another crucial assumption on the stability/correctability of the true dynamics. The following assumption essentially says that if we are at a state that is near the demonstration set, then there exists an action that can drive us closer to the demonstration set. This assumption rules out certain dynamics that does not allow corrections even after the policy making a small error. For example, if a robot, unfortunately, falls off a cliff, then fundamentally it cannot recover itself — our algorithm cannot deal with such pathological situations.

Assumption 4.2 (Locally-correctable dynamics). For some  $\gamma \in (0,1)$  and  $\varepsilon > 0$ ,  $L_{c} > 0$ , we assume that the dynamics  $M^{\star}$  is  $(\gamma, L_{c}, \varepsilon)$ -locally-correctable w.r.t to the set  $\mathcal{U}$  in the sense that for all  $\varepsilon_0 \in (0, \varepsilon]$  and any tuple  $(\bar{s}, \bar{a}, \bar{s}')$  satisfying  $\bar{s}, \bar{s}' \in \mathcal{U}$  and  $\bar{s}' = M^{\star}(\bar{s}, \bar{a})$ , and any  $\varepsilon_0$ -perturbation  $s$  of  $\bar{s}$  (that is,  $s \in N_{\varepsilon_0}(\bar{s})$ ), there exists an action  $a_{\mathrm{cx}}$  that is  $L_{c} \varepsilon_0$  close to  $\bar{a}$ , such that it makes a correction in the sense that the resulting state  $s'$  is  $\gamma \varepsilon_0$ -close to the set  $\mathcal{U}$ :  $s' = M^{\star}(s, a_{\mathrm{cx}}) \in N_{\gamma \varepsilon_0}(\mathcal{U})$ . Here  $N_{\delta}(K)$  denotes the set of points that are  $\delta$ -close to  $K$ .

Finally, we will assume the BC policy, the value function, and the dynamics are all Lipschitz in their arguments. We also assume the projection operator to the set  $\mathcal{U}$  is locally Lipschitz. These are regularity conditions that provide loose local extrapolation of these functions, and they are satisfied by parameterized neural networks that are used to model these functions.

Assumption 4.3 (Lipschitz-ness of policy, value function, and dynamics). We assume that the policy  $\pi_{\mathrm{BC}}$  is  $L_{\pi}$ -Lipschitz. That is,  $\| \pi_{\mathrm{BC}}(s) - \pi_{\mathrm{BC}}(\tilde{s}) \| \leq L_{\pi} \| s - \tilde{s} \|$  for all  $s, \tilde{s}$ . We assume the value function  $V^{\pi_{\mathrm{e}}}$  and the learned value function  $V$  are  $L_{V}$ -Lipschitz, the model  $M^{\star}$  is  $L_{M,a}$ -Lipschitz w.r.t to the action and  $L_{M,s}$ -Lipschitz w.r.t to the state  $s$ . We also assume that the set  $\mathcal{U}$  has  $L_{\Pi}$ -Lipschitz projection locally: for all  $s, \hat{s}$  that is  $\varepsilon$ -close to  $\mathcal{U}$ ,  $\| \Pi_{\mathcal{U}}(s) - \Pi_{\mathcal{U}}(\hat{s}) \| \leq L_{\Pi} \| s - \hat{s} \|$ .

Under these assumptions, now we are ready to state our main theorem. It claims that 1) the induced policy  $\pi$  in Alg. 1 stays close to the demonstration set and performs similarly to the expert policy  $\pi_{\mathrm{e}}$ , and 2) following the induced policy  $\pi$ , we will arrive at a state with a near-optimal value.

Theorem 4.4. Suppose Assumption 4.1, 4.2, 4.3 hold with sufficiently small  $\varepsilon >0$  and errors  $\delta_{M},\delta_{\pi},\delta_{\pi} > 0$  so that they satisfy  $\zeta \geq L_c\varepsilon +\delta_\pi +L_\pi$  . Let  $\lambda$  be sufficiently large so that  $\lambda \geq$ $\frac{2L_VL_{\Pi}L_M\zeta + 2\delta_V + 2L_V\delta_M}{(1 - \gamma)\varepsilon}$  . Then, the policy  $\pi$  from equation (4.3) satisfies the following:

1. Starting from  $s_0 \in \mathcal{U}$  and executing policy  $\pi$  for  $T_0 \leq T$  steps, the resulting states  $s_1, \ldots, s_{T_0}$  are all  $\varepsilon$ -close to the demonstrate states set  $\mathcal{U}$ .  
2. In addition, suppose the expert policy makes at least  $\rho$  improvement every step in the sense that for every  $s\in \mathcal{U}$ , either  $V^{\pi_{\mathrm{e}}}(M^{\star}(s,\pi_{\mathrm{e}}(s)))\geq V^{\pi_{\mathrm{e}}}(s) + \rho$  or  $M^{\star}(s,\pi_{\mathrm{e}}(s))$  reaches the goal. Assume  $\varepsilon$  and  $\delta_M,\delta_V,\delta_\pi$  are small enough so that they satisfy  $\rho \gtrsim \varepsilon +\delta_{\pi}$ .

Then, the policy  $\pi$  will achieve a state  $s_T$  with  $T \leq 2|V^{\pi_{\mathrm{e}}}(s_0)| / \rho$  steps which is  $\varepsilon$ -close to a state  $\bar{s}_T$  with value at least  $V^{\pi_{\mathrm{e}}}(s_T) \gtrsim -(\varepsilon + \delta_{\pi})$ .<sup>5</sup>

The proof is deferred to Section B. The first bullet follows inductively invoking the following lemma which states that if the current state is  $\varepsilon$ -close to  $\mathcal{U}$ , then so is the next state. The proof of the Lemma is the most mathematically involved part of the paper and is deferred to the Section B. We demonstrate the key idea of the proof in Figure 2 and its caption.

Lemma 4.5. In the setting of Theorem 4.4, suppose  $s$  is  $\varepsilon$ -close to the demonstration states set  $\mathcal{U}$ . Suppose  $\mathcal{U}$ , and let  $a = \pi(s)$  and  $s' = M^{\star}(s, a)$ . Then,  $s'$  is also  $\varepsilon$ -close to the set  $\mathcal{U}$ .

We effectively represent the  $Q$  function by  $V(M(s,a))$  in Alg. 1. We argue in Section A.1 that this helps address the degeneracy issue when there are random goal states (which is the case in our experiments.)

Discussion: can we learn conservatively-extrapolated  $Q$ -function? We remark that we do not expect a conservative-extrapolated  $Q$ -functions would be helpful. The fundamental idea here is to penalize the value of unseen states so that the policy can self-correct. However, to learn a  $Q$  function that induces self-correctable policies, we should encourage unseen actions that can correct the trajectory, instead of penalize them just because they are not seen before. Therefore, it is crucial that the penalization is done on the unseen states (or  $V$ ) but not the unseen actions (or  $Q$ ).

# 5 MAIN APPROACH

Learning value functions with negative sampling from demonstration trajectories. As motivated in Section 4 by Algorithm 1 and Theorem 4.4, we first develop a practical method that can learn a value function with conservative extrapolation, without environment interaction. Let  $V_{\phi}$  be a value function parameterized by  $\phi$ . Using the standard TD learning loss, we can ensure the value function to be accurate on the demonstration states  $\mathcal{U}$  (i.e., to satisfy equation (4.1)). Let  $\bar{\phi}$  be the target value

Algorithm 2 Value Iteration on Demonstrations with Negative Sampling (VINS)  
1:  $\mathcal{R}\gets$  demonstration trajectories  $\triangleright$  No environment interaction will be used   
2: Initialize value parameters  $\bar{\phi} = \phi$  and model parameters  $\theta$  randomly   
3: for  $i = 1,\dots ,T$  do   
4: sample mini-batch  $\mathcal{B}$  of  $N$  transitions  $(s,a,r,s^{\prime})$  from  $\mathcal{R}$    
5: update  $\phi$  to minimize  $\mathcal{L}_{td}(\phi ;\mathcal{B}) + \mathcal{L}_{ns}(\phi ;\mathcal{B})$    
6: update  $\theta$  to minimize loss  $\mathcal{L}_{\mathrm{model}}(\theta ;\mathcal{B})$    
7: update target network:  $\bar{\phi}\gets \bar{\phi} +\tau (\phi -\bar{\phi})$    
8:   
9: function POLICY(s)   
10: Option 1:  $a = \pi_{\mathrm{BC}}(s)$  ; Option 2:  $a = 0$    
11: sample  $k$  noises  $\xi_1,\ldots ,\xi_k$  from Uniform[-1,1]m  $\triangleright m$  is the dimension of action space   
12:  $i^{*} = \operatorname {argmax}_{i}V_{\phi}(M_{\theta}(s,a + \alpha \xi_{i}))$ $\triangleright \alpha >0$  is a hyper-parameter   
13: return  $a + \alpha \xi_{i*}$

function, $^6$  the TD learning loss is defined as  $\mathcal{L}_{td}(\phi) = \mathbb{E}_{(s,a,s')\sim \rho^{\pi_{\mathrm{e}}}}\left[\left(r(s,a) + V_{\bar{\phi}}(s') - V_{\phi}(s)\right)^2\right]$  where  $r(s,a)$  is the (sparse) reward,  $\bar{\phi}$  is the parameter of the target network,  $\rho^{\pi_{\mathrm{e}}}$  is the distribution of the states-action-states tuples of the demonstrations. The crux of the ideas in this paper is to use a negative sampling technique to enforce the value function to satisfy conservative extrapolation requirement (4.2). It would be infeasible to enforce condition (4.2) for every  $s\notin \mathcal{U}$ . Instead, we draw random "negative samples"  $\tilde{s}$  from the neighborhood of  $\mathcal{U}$ , and enforce the condition (4.2). This is inspired by the negative sampling approach widely used in NLP for training word embeddings Mikolov et al. (2013); Gutmann & Hyvarinen (2012). Concretely, we draw a sample  $s\sim \rho^{\pi_{\mathrm{e}}}$ , create a random perturbation of  $s$  to get a point  $\tilde{s}\notin \mathcal{U}$  and construct the following loss function: $^7$

$$
\mathcal {L} _ {n s} (\phi) = \mathbb {E} _ {s \sim \rho^ {\pi_ {\mathrm {e}}}, \tilde {s} \sim \text {p e r t u r b} (s)} \left(V _ {\bar {\phi}} (s) - \lambda \| s - \tilde {s} \| - V _ {\phi} (\tilde {s})\right) ^ {2},
$$

The rationale of the loss function can be best seen in the situation when  $\mathcal{U}$  is assumed to be a low-dimensional manifold in a high-dimensional state space. In this case,  $\tilde{s}$  will be outside the manifold  $\mathcal{U}$  with probability 1. Moreover, the random direction  $\tilde{s} - s$  is likely to be almost orthogonal to the tangent space of the manifold  $\mathcal{U}$ , and thus  $s$  is a reasonable approximation of the projection of  $\tilde{s}$  back to the  $\mathcal{U}$ , and  $\|s - \tilde{s}\|$  is an approximation of  $\|\Pi_{\mathcal{U}}\tilde{s} - \tilde{s}\|$ . If  $\mathcal{U}$  is not a manifold but a small subset of the state space, these properties may still likely to hold for a good fraction of  $s$ .

We only attempt to enforce condition (4.2) for states near  $\mathcal{U}$ . This likely suffices because the induced policy is shown to always stay close to  $\mathcal{U}$ . Empirically, we perturb  $s$  by adding a Gaussian noise. The loss function to learn  $V_{\phi}$  is defined as  $\mathcal{L}(\phi) = \mathcal{L}_{td}(\phi) + \mu \mathcal{L}_{ns}(\phi)$  for some constant  $\mu > 0$ . For a mini-batch  $\mathcal{B}$  of data, we define the corresponding empirical loss by  $\mathcal{L}(\phi; \mathcal{B})$  (similarly we define  $\mathcal{L}_{td}(\phi; \mathcal{B})$  and  $\mathcal{L}_{ns}(\phi; \mathcal{B})$ ). The concrete iterative learning algorithm is described in line 1-7 of Algorithm 2 (except line 6 is for learning the dynamical model, described below.)

Learning the dynamical model. We use standard supervised learning to train the model. We use  $\ell_2$  norm as the loss for model parameters  $\theta$  instead of the more commonly used MSE loss, following the success of (Luo et al., 2018):  $\mathcal{L}_{\mathrm{model}}(\theta) = \mathbb{E}_{(s,a,s')\sim \rho^{\pi_{\mathrm{e}}}}\left[\| M_{\theta}(s,a) - s'\| _2\right]$ .

Optimization for policy. We don't maintain an explicit policy but use an induced policy from  $V_{\phi}$  and  $M_{\theta}$  by optimizing equation (4.3). A natural choice would be using projected gradient ascent to optimize equation (4.3). However, we found the random shooting suffices because the action space is relatively low-dimensional in our experiments. Moreover, the randomness introduced appears to reduce the overfitting of the model and value function slightly. As shown in line 10-13 of Alg. 2, we sample  $k$  actions in the feasible set and choose the one with maximum  $V_{\phi}(M_{\theta}(s,a))$ .

Value iteration with environment interaction. As alluded before, when more environment interactions are allowed, we initialize an RL algorithm by the value function, dynamics learned from VINS. Given that we have  $V$  and  $M$  in hand, we alternate between fitted value iterations for updating the

value function and supervised learning for updating the models. (See Algorithm 3 in Section C.) We do not use negative sampling here since the RL algorithms already collect bad trajectories automatically. We also do not hallucinate any goals as in HER (Andrychowicz et al., 2017).

# 6 EXPERIMENTS

**Environments.** We evaluate our algorithms in three simulated robotics environments designed by (Plappert et al., 2018) based on OpenAI Gym (Brockman et al., 2016) and MuJoCo (Todorov et al., 2012): Reach, Pick-And-Place, and Push. A detailed description can be found in Section D.1.

Demonstrations. For each task, we use Hindsight Experience Replay (HER) (Andrychowicz et al., 2017) to train a policy until convergence. The policy rolls out to collect 100/200 successful trajectories as demonstrations except for Reach environment where 100 successful trajectories are sufficient for most of the algorithms to achieve optimal policy. We filtered out unsuccessful trajectories during data collection.

We consider two settings: imitation learning from only demonstrations data, and leveraging demonstration in RL with a limited amount of interactions. We compare our algorithm with Behavioral Cloning and multiple variants of our algorithms in the first setting. We compare with the previous state-of-the-art by Nair et al. (2018), and GAIL Ho & Ermon (2016) in the second setting. We do not compare with (Gao et al., 2018) because it cannot be applied to the case with continuous actions.

Behavioral Cloning (Bain & Sommut, 1999). Behavioral Cloning (BC) learns a mapping from a state to an action on demonstration data using supervised learning. We use MSE loss for predicting the actions.

Nair et al.'18 (Nair et al., 2018). The previous state-of-the-art algorithm from Nair et al. (2018) combines HER (Andrychowicz et al., 2017) with BC and a few techniques: 1) an additional replay buffer filled with demonstrations, 2) an additional behavioral cloning loss for the policy, 3) a  $Q$ -filter for non-optimal demonstrations, 4) resets to states in the demonstrations to deal with long horizon tasks. We note that resetting to an arbitrary state may not be realistic for real-world applications in robotics. In contrast, our algorithm does not require resetting to a demonstration state.

GAIL (Ho & Ermon, 2016) Generative Adversarial Imitation Learning (GAIL) imitates the expert by matching the state-action distribution with a GAN-like framework.

VINS. As described in Section 5, in the setting without environment interaction, we use Algorithm 2; otherwise we use it to initialize an RL algorithm (see Algorithm 3). We use neural networks to parameterize the value function and the dynamics model. The granularity of the HER demonstration policy is very coarse, and we argue the data with additional linear interpolation between consecutive states. We also use only a subset of the states as inputs to the value function and the dynamics model, which apparently helps improve the training and generalization of them. Implementation details can be found in Section D.2.

<table><tr><td></td><td>VINS (ours)</td><td>BC</td></tr><tr><td>Reach 10</td><td>99.3 ± 0.1%</td><td>98.6 ± 0.1%</td></tr><tr><td>Pick 100</td><td>75.7 ± 1.0%</td><td>66.8 ± 1.1%</td></tr><tr><td>Pick 200</td><td>84.0 ± 0.5%</td><td>82.0 ± 0.8%</td></tr><tr><td>Push 100</td><td>44.0 ± 1.5%</td><td>37.3 ± 1.1%</td></tr><tr><td>Push 200</td><td>55.2 ± 0.7%</td><td>51.3 ± 0.6%</td></tr></table>

Table 1: The success rates of achieving the goals for VINS and BC in the setting without any environment interactions. A random policy has about  $5\%$  success rate at Pick and Push.

![](images/95169d6efe09126cf4af023a0435ece5363189e8c59c226ba659918cdf831c15.jpg)  
Figure 3: The learning curves of VINS+RL vs the prior state-of-the-art Nair et al.'18 on Pick-And-Place and Push. Shaded areas indicate one standard deviation estimated from 10 random seeds.

Our main results are reported in Table 1<sup>10</sup> for the setting with no environment interaction and Figure 3 for the setting with environment interactions. Table 1 shows that the Reach environment is too simple so that we do not need to run the RL algorithm. On the harder environments Pick-And-Place and Push, our algorithm VINS outperforms BC. We believe this is because our conservatively-extrapolated value function helps correct the mistakes in the policy. Here we use 2k trials to estimate the success rate (so that the errors in the estimation is negligible), and we run the algorithms with 10 different seeds. The error bars are for 1 standard deviation.

Figure 3 shows that VINS initialized RL algorithm outperforms prior state-of-the-art in sample efficiency. We believe the main reason is that due to the initialization of value and model, we pay less samples for warming up the value function. We note that our initial success rate in RL is slightly lower than the final result of VINS in Table 1. This is because in RL we implemented a slightly worse variant of the policy induced by VINS: in the policy of Algorithm 2, we use option 2 by search the action uniformly. This suffices because the additional interactions quickly allows us to learn a good model and the BC constraint is no longer needed.

Ablation studies. Towards understanding the effect of each component of VINS, we perform three ablative experiments to show the importance of negative sampling, searching in the neighborhood of Behavioral Cloned actions (option 1 in line 10 or Algorithm 2), and a good dynamics model. The results are shown in Table 2. We study three settings: (1) VINS without negative sampling (VINS w/o NS), where the loss  $\mathcal{L}_{ns}$  is removed; (2) VINS without BC (VINS w/o BC), where option 2 in line 10 or Algorithm 2 is used; (3) VINS with oracle model without BC (VINS w/ oracle w/o BC), where we use the true dynamics model to replace line 12 of Algorithm 2. Note that the last setting is only synthetic for ablation study because in the real-world we don't have access to the true dynamics model. Please see the caption of Table 2 for more interpretations. We use the same set of hyperparameters for the same environment, which may not be optimal: for example, with more expert trajectories, the negative sampling loss  $\mathcal{L}_{ns}$ , which can be seen as a regularization, should be assigned a smaller coefficient  $\mu$ .

# 7 CONCLUSION

We devise a new algorithm, VINS, that can learn self-correctable by learning value function and dynamical model from demonstrations. The key idea is a theoretical formulation of conservatively-extrapolated value functions that provably leads to self-correction. The empirical results show a promising performance of VINS and an algorithm that initializes RL with VINS. It's a fascinating direction to study other algorithms that may learn conservatively-extrapolated value functions in other real-world applications beyond the proof-of-concepts experiments in this paper. For example, the negative sampling by Gaussian perturbation technique in this paper may not make sense for high-dimensional pixel observation. The negative sampling can perhaps be done in the representation

<table><tr><td></td><td>Pick 100</td><td>Pick 200</td><td>Push 100</td><td>Push 200</td></tr><tr><td>BC</td><td>66.8 ± 1.1%</td><td>82.0 ± 0.8%</td><td>37.3 ± 1.1%</td><td>51.3 ± 0.6%</td></tr><tr><td>VINS</td><td>75.7 ± 1.0%</td><td>84.0 ± 0.5%</td><td>44.0 ± 0.8%</td><td>55.2 ± 0.7%</td></tr><tr><td>VINS w/o BC</td><td>28.5 ± 1.1%</td><td>43.6 ± 1.2%</td><td>14.3 ± 0.5%</td><td>24.9 ± 1.3%</td></tr><tr><td>VINS w/ oracle w/o BC</td><td>51.4 ± 1.4%</td><td>62.3 ± 1.1%</td><td>40.7 ± 1.4%</td><td>42.9 ± 1.3%</td></tr><tr><td>VINS w/ oracle</td><td>76.3 ± 1.4%</td><td>87.0 ± 0.7%</td><td>48.7 ± 1.2%</td><td>63.8 ± 1.3%</td></tr><tr><td>VINS w/o NS</td><td>48.5 ± 2.1%</td><td>71.6 ± 0.9%</td><td>29.3 ± 1.2%</td><td>38.7 ± 1.5%</td></tr></table>

Table 2: Ablation study of components of VINS in the setting without environment interactions. We reported the average performance of 10 runs (with different random seeds) and the empirical standard deviation of the estimator of the average performance. The success rate of VINS w/o NS is consistently worse than VINS, which suggests that NS is crucial for tackling the false extrapolation. From comparisons between VINS w/o BC and VINS w/ oracle w/o BC, and between VINS and VINS w/ oracle, we observe that if the learning of the dynamics can be improved (potentially by e.g., by collecting data with random actions), then VINS or VINS w/o BC can be improved significantly. We also suspect that the reason why we need to search over the neighborhood of BC actions is that the dynamics is not accurate at state-action pairs far away from the demonstration set (because the dynamics is only learned on the demonstration set.)

space (which might be learned by unsupervised learning algorithms) so that we can capture the geometry of the state space better.

# REFERENCES

Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning, pp. 1. ACM, 2004.  
Jacopo Aleotti and Stefano Caselli. Grasp recognition in virtual reality for robot pregrasp planning by demonstration. In Proceedings 2006 IEEE International Conference on Robotics and Automation, 2006. ICRA 2006., pp. 2801-2806. IEEE, 2006.  
Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Advances in Neural Information Processing Systems, pp. 5048-5058, 2017.  
Brenna D Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning from demonstration. Robotics and autonomous systems, 57(5):469-483, 2009.  
J Andrew Bagnell. An invitation to imitation. Technical report, CARNEGIE-MELLON UNIV PITTSBURGH PA ROBOTICS INST, 2015.  
Michael Bain and Claude Sommut. A framework for behavioural cloning. *Machine intelligence*, 15(15):103, 1999.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Jessica Chemali and Alessandro Lazaric. Direct policy iteration with demonstrations. In Twenty-Fourth International Joint Conference on Artificial Intelligence, 2015.  
Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Advances in Neural Information Processing Systems, pp. 4754-4765, 2018.  
Ignasi Clavera, Jonas Rothfuss, John Schulman, Yasuhiro Fujita, Tamim Asfour, and Pieter Abbeel. Model-based reinforcement learning via meta-policy optimization. arXiv preprint arXiv:1809.05214, 2018.  
Thomas Degris, Martha White, and Richard S Sutton. Off-policy actor-critic. arXiv preprint arXiv:1205.4839, 2012.  
Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, Yuhuai Wu, and Peter Zhokhov. Openai baselines. https://github.com/openai/baselines, 2017.

Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv preprint arXiv:1611.03852, 2016a.  
Chelsea Finn, Sergey Levine, and Pieter Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. In International Conference on Machine Learning, pp. 49-58, 2016b.  
Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. arXiv preprint arXiv:1710.11248, 2017.  
Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. arXiv preprint arXiv:1812.02900, 2018a.  
Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018b.  
Yang Gao, Ji Lin, Fisher Yu, Sergey Levine, Trevor Darrell, et al. Reinforcement learning from imperfect demonstrations. arXiv preprint arXiv:1802.05313, 2018.  
Shixiang Gu, Timothy Lillicrap, Zoubin Ghahramani, Richard E Turner, and Sergey Levine. Q-prop: Sample-efficient policy gradient with an off-policy critic. arXiv preprint arXiv:1611.02247, 2016.  
Michael U Gutmann and Aapo Hyvarinen. Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. Journal of Machine Learning Research, 13(Feb):307-361, 2012.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.  
Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, et al. Deep q-learning from demonstrations. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In Advances in Neural Information Processing Systems, pp. 4565-4573, 2016.  
S Mohammad Khansari-Zadeh and Aude Billard. Learning stable nonlinear dynamical systems with gaussian mixture models. IEEE Transactions on Robotics, 27(5):943-957, 2011.  
Beomjoon Kim, Amir-massoud Farahmand, Joelle Pineau, and Doina Precup. Learning from limited demonstrations. In Advances in Neural Information Processing Systems, pp. 2859-2867, 2013.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble trust-region policy optimization. arXiv preprint arXiv:1802.10592, 2018.  
Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. Dart: Noise injection for robust imitation learning. arXiv preprint arXiv:1703.09327, 2017.  
Martin Lawitzky, Jose Ramon Medina, Dongheui Lee, and Sandra Hirche. Feedback motion planning and learning from demonstration in physical robotic assistance: differences and synergies. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 3646-3652. IEEE, 2012.  
Hoang M Le, Yisong Yue, Peter Carr, and Patrick Lucey. Coordinated multi-agent imitation learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1995-2003. JMLR.org, 2017.  
Hoang M Le, Nan Jiang, Alekh Agarwal, Miroslav Dudík, Yisong Yue, and Hal Daumé III. Hierarchical imitation and reinforcement learning. arXiv preprint arXiv:1803.00590, 2018.  
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, and Tengyu Ma. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees. 2018.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pp. 3111-3119, 2013.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems, pp. 1054-1062, 2016.  
Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Overcoming exploration in reinforcement learning with demonstrations. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 6292-6299. IEEE, 2018.  
Andrew Y Ng, Stuart J Russell, et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, pp. 2, 2000.  
Takayuki Osa, Amir M Ghalamzan Esfahani, Rustam Stolkin, Rudolf Lioutikov, Jan Peters, and Gerhard Neumann. Guiding trajectory optimization by demonstrated distributions. IEEE Robotics and Automation Letters, 2(2):819-826, 2017.  
Takayuki Osa, Joni Pajarinen, Gerhard Neumann, J Andrew Bagnell, Pieter Abbeel, Jan Peters, et al. An algorithmic perspective on imitation learning. Foundations and Trends in Robotics, 7(1-2):1-179, 2018.  
Razvan Pascanu, Yujia Li, Oriol Vinyals, Nicolas Heess, Lars Buesing, Sebastien Racanière, David Reichert, Théophane Weber, Daan Wierstra, and Peter Battaglia. Learning model-based planning from scratch. arXiv preprint arXiv:1707.06170, 2017.  
Bilal Piot, Matthieu Geist, and Olivier Pietquin. Boosted bellman residual minimization handling expert demonstrations. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 549-564. Springer, 2014.  
Matthias Plappert, Marcin Andrychowicz, Alex Ray, Bob McGrew, Bowen Baker, Glenn Powell, Jonas Schneider, Josh Tobin, Maciek Chogiej, Peter Welinder, Vikash Kumar, and Wojciech Zaremba. Multi-goal reinforcement learning: Challenging robotics environments and request for research, 2018.  
Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In Advances in neural information processing systems, pp. 305-313, 1989.  
Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017.  
Stéphane Ross and Drew Bagnell. Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 661-668, 2010.  
Stephane Ross and J Andrew Bagnell. Reinforcement and imitation learning via interactive no-regret learning. arXiv preprint arXiv:1406.5979, 2014.  
Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627-635, 2011.  
Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, and Peter Battaglia. Graph networks as learnable physics engines for inference and control. arXiv preprint arXiv:1806.01242, 2018.  
Stefan Schaal. Learning from demonstration. In Advances in neural information processing systems, pp. 1040-1046, 1997.  
Yannick Schroeker and Charles L Isbell. State aware imitation learning. In Advances in Neural Information Processing Systems, pp. 2911-2920, 2017.  
Yannick Schroeker, Mel Vecerik, and Jon Scholz. Generative predecessor models for sample-efficient imitation learning. 2018.

Wen Sun, Arun Venkatraman, Geoffrey J Gordon, Byron Boots, and J Andrew Bagnell. Deeply aggravated: Differentiable imitation learning for sequential prediction. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 3309-3318. JMLR.org, 2017.  
Wen Sun, J Andrew Bagnell, and Byron Boots. Truncated horizon policy search: Combining reinforcement learning & imitation learning. arXiv preprint arXiv:1805.11240, 2018a.  
Wen Sun, Geoffrey J Gordon, Byron Boots, and J Bagnell. Dual policy iteration. In Advances in Neural Information Processing Systems, pp. 7059-7069, 2018b.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026-5033. IEEE, 2012.  
Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation. arXiv preprint arXiv:1805.01954, 2018.  
Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.  
Matej Večérík, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothörl, Thomas Lampe, and Martin Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards. arXiv preprint arXiv:1707.08817, 2017.  
Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224, 2016.  
Ziyu Wang, Josh S Merel, Scott E Reed, Nando de Freitas, Gregory Wayne, and Nicolas Heess. Robust imitation of diverse behaviors. In Advances in Neural Information Processing Systems, pp. 5320-5329, 2017.  
Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3-4):279-292, 1992.  
Gu Ye and Ron Alterovitz. guided motion planning. In Robotics research, pp. 291-307. Springer, 2017.  
Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pp. 1433-1438. Chicago, IL, USA, 2008.
