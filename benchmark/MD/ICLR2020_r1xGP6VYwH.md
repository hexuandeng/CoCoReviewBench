# OPTIMISTIC EXPLORATION EVEN WITH A PESSIMISTIC INITIALISATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Optimistic initialisation is an effective strategy for efficient exploration in reinforcement learning (RL). In the tabular case, all provably efficient model-free algorithms rely on it. However, model-free deep RL algorithms do not use optimistic initialisation despite taking inspiration from these provably efficient tabular algorithms. In particular, in scenarios with only positive rewards,  $Q$ -values are initialised at their lowest possible values due to commonly used network initialisation schemes, a pessimistic initialisation. Merely initialising the network to output optimistic  $Q$ -values is not enough, since we cannot ensure that they remain optimistic for novel state-action pairs, which is crucial for exploration. We propose a simple count-based augmentation to pessimistically initialised  $Q$ -values that separates the source of optimism from the neural network. We show that this scheme is provably efficient in the tabular setting and extend it to the deep RL setting. Our algorithm, Optimistic Pessimistically Initialised  $Q$ -Learning (OPIQ), augments the  $Q$ -value estimates of a DQN-based agent with count-derived bonuses to ensure optimism during both action selection and bootstrapping. We show that OPIQ outperforms non-optimistic DQN variants that utilise a pseudocount-based intrinsic motivation in hard exploration tasks, and that it predicts optimistic estimates for novel state-action pairs.

# 1 INTRODUCTION

In reinforcement learning (RL), exploration is crucial for gathering sufficient data to infer a good control policy. As environment complexity grows, exploration becomes more challenging and simple randomisation strategies become inefficient.

While most provably efficient methods for tabular RL are model-based (Brafman and Tennenholtz, 2002; Strehl and Littman, 2008; Azar et al., 2017), in deep RL, learning models that are useful for planning is notoriously difficult and often more complex (Hafner et al., 2019) than model-free methods. Consequently, model-free approaches have shown the best final performance on large complex tasks (Mnih et al., 2015; 2016; Hessel et al., 2018), especially those requiring hard exploration (Bellemare et al., 2016; Ostrovski et al., 2017). Therefore, in this paper, we focus on how to devise model-free RL algorithms for efficient exploration that scale to large complex state spaces and have strong theoretical underpinnings.

Despite taking inspiration from tabular algorithms, current model-free approaches to exploration in deep RL do not employ optimistic initialisation, which is crucial to provably efficient exploration in all model-free tabular algorithms. This is because deep RL algorithms do not pay special attention to the initialisation of the neural networks and instead use common initialisation schemes that yield initial  $Q$ -values around zero. In the common case of non-negative rewards, this means  $Q$ -values are initialised to their lowest possible values, i.e., a pessimistic initialisation.

While initialising a neural network optimistically would be trivial, e.g., by setting the bias of the final layer of the network, the uncontrolled generalisation in neural networks changes this initialisation quickly. Instead, to benefit exploration, we require the  $Q$ -values for novel state-action pairs must remain high until they are explored.

An empirically successful approach to exploration in deep RL, especially when reward is sparse, is intrinsic motivation (Oudeyer and Kaplan, 2009). A popular variant is based on pseudocounts

(Bellemare et al., 2016), which derive an intrinsic bonus from approximate visitation counts over states and is inspired by the tabular MBIE-EB algorithm (Strehl and Littman, 2008). However, adding a positive intrinsic bonus to the reward yields optimistic  $Q$ -values only for state-action pairs that have already been chosen sufficiently often. Incentives to explore unvisited states rely therefore on the generalisation of the neural network. Exactly how the network generalises to those novel state-action pairs is unknown, and thus it is unclear whether those estimates are optimistic when compared to nearby visited state-action pairs.

![](images/12139f4baec9ca03db4d9f63b1463fc8fc4af77c62dd395df5f045e3b4c6ae8e.jpg)  
Figure 1

Consider the simple example with a single state and two actions shown in Figure 1. The left action yields  $+0.1$  reward and the right action yields  $+1$  reward. An agent whose  $Q$ -value estimates have been zero-initialised must at the first time step select an action randomly. As both actions are underestimated, this will increase the estimate of the chosen action. Greedy agents always pick the action with the largest  $Q$ -value estimate and will select the same action forever, failing

to explore the alternative. Whether the agent learns the optimal policy or not is thus decided purely at random based on the initial  $Q$ -value estimates. This effect will only be amplified by intrinsic reward.

To ensure optimism in unvisited, novel state-action pairs, we introduce Optimistic Pessimistically Initialised  $Q$ -Learning (OPIQ). OPIQ does not rely on an optimistic initialisation to ensure efficient exploration, but instead augments the  $Q$ -value estimates with count-based bonuses in the following manner:

$$
Q ^ {+} (s, a) := Q (s, a) + \frac {C}{(N (s , a) + 1) ^ {M}}, \tag {1}
$$

where  $N(s, a)$  is the number of times a state-action pair has been visited and  $M, C > 0$  are hyperparameters. These  $Q^{+}$ -values are then used for both action selection and during bootstrapping, unlike the above methods which only utilise  $Q$ -values during these steps. This allows OPIQ to maintain optimism when selecting actions and bootstrapping, since the  $Q^{+}$ -values can be optimistic even when the  $Q$ -values are not.

In the tabular domain, we base OPIQ on UCB-H (Jin et al., 2018), a simple online  $Q$ -learning algorithm that uses count-based intrinsic rewards and optimistic initialisation. Instead of optimistically initialising the  $Q$ -values, we pessimistically initialise them and use  $Q^{+}$ -values during action selection and bootstrapping. Pessimistic initialisation is used to enable a worst case analysis where all of our  $Q$ -value estimates underestimate  $Q^{*}$  and is not a requirement for OPIQ. We show that these modifications retain the theoretical guarantees of UCB-H.

Furthermore, our algorithm easily extends to the Deep RL setting. The primary difficulty lies in obtaining appropriate state-action counts in high-dimensional and/or continuous state spaces, which has been tackled by a variety of approaches (Bellemare et al., 2016; Ostrovski et al., 2017; Tang et al., 2017; Machado et al., 2018) and is orthogonal to our contributions.

We demonstrate clear performance improvements in sparse reward tasks over 1) a baseline DQN that just uses intrinsic motivation derived from the approximate counts, 2) simpler schemes that aim for an optimistic initialisation when using neural networks, and 3) strong exploration baselines. We show the importance of optimism during action selection for ensuring efficient exploration. Visualising the predicted  $Q^{+}$ -values shows that they are indeed optimistic for novel state-action pairs.

# 2 BACKGROUND

We consider a Markov Decision Process (MDP) defined as a tuple  $(S, \mathcal{A}, P, R)$ , where  $S$  is the state space,  $\mathcal{A}$  is the discrete action space,  $P(\cdot | s, a)$  is the state-transition distribution,  $R(\cdot | s, a)$  is the distribution over rewards and  $\gamma \in [0,1)$  is the discount factor. The goal of the agent is then to maximise the expected discounted sum of rewards:  $\mathbb{E}[\sum_{t=0}^{\infty} \gamma^{t} r_{t} | r_{t} \sim R(\cdot | s_{t}, a_{t})]$ , in the discounted episodic setting. A policy  $\pi(\cdot | s)$  is a mapping from states to actions such that it is a valid probability distribution.  $Q^{\pi}(s, a) := \mathbb{E}[\sum_{t=0}^{\infty} \gamma^{t} r_{t} | a_{t} \sim \pi(\cdot | s_{t})]$  and  $Q^{*} := \max_{\pi} Q^{\pi}$ .

Deep  $Q$ -Network (DQN) (Mnih et al., 2015) uses a nonlinear function approximator (a deep neural network) to estimate the action-value function,  $Q(s,a;\theta)\approx Q^{*}(s,a)$ , where  $\theta$  are the parameters of the network. Exploration based on intrinsic rewards (e.g., Bellemare et al., 2016), which uses a DQN agent, additionally augments the observed rewards  $r_t$  with a bonus  $\beta /\sqrt{N(s_t,a_t)}$  based on pseudo-visitation-counts  $N(s_{t},a_{t})$ . The DQN parameters  $\theta$  are trained by gradient descent on the mean squared regression loss  $\mathcal{L}$  with bootstrapped 'target'  $y_{t}$ :

![](images/3232c9cce59b358f40e5cbb88635c82b6291eade92d7d8cc093a6dcb204f692d.jpg)  
Figure 2: A simple regression task to illustrate the effect of an optimistic initialisation in neural networks. Left: 10 different networks whose final layer biases are initialised at 3 (shown in green), and the same networks after training on the blue data points (shown in red). Right: One of the trained networks whose output has been augmented with an optimistic bias as in equation 1. The counts were obtained by computing a histogram over the input space  $[-2, 2]$  with 50 bins.

![](images/724197736131ca4ef7a096ce8afdd2aed8912f5240bdeece4c6dd8a5dc167b6d.jpg)

$$
\mathcal {L} [ \theta ] := \mathbb {E} \left[ \left(\overbrace {r _ {t} + \frac {\beta}{\sqrt {N (s _ {t} , a _ {t})}} + \gamma \max  _ {a ^ {\prime}} Q \left(s _ {t + 1} , a ^ {\prime} ; \theta^ {-}\right)} ^ {y _ {t}} - Q \left(s _ {t}, a _ {t}; \theta\right)\right) ^ {2} \mid \left(s _ {t}, a _ {t}, r _ {t}, s _ {t + 1}\right) \sim D \right]. \tag {2}
$$

The expectation is estimated with uniform samples from a replay buffer  $D$  (Lin, 1992).  $D$  stores past transitions  $(s_t, a_t, r_t, s_{t+1})$ , where the state  $s_{t+1}$  is observed after taking the action  $a_t$  in state  $s_t$  and receiving reward  $r_t$ . To improve stability, DQN uses a target network, parameterised by  $\theta^{-}$ , which is periodically copied from the regular network and kept fixed for a number of iterations.

# 3 OPTIMISTIC PESSIMISTICALLY INITIALISED  $Q$ -LEARNING

Our method Optimistic Pessimistically Initialised  $Q$ -Learning (OPIQ) ensures optimism in the  $Q$ -value estimates of unvisited, novel state-action pairs in order to drive exploration. This is achieved by augmenting the  $Q$ -value estimates in the following manner:

$$
Q ^ {+} (s, a) := Q (s, a) + \frac {C}{(N (s , a) + 1) ^ {M}},
$$

and using these  $Q^{+}$ -values during action selection and bootstrapping. In this section, we motivate OPIQ, analyse it in the tabular setting, and describe a deep RL implementation.

# 3.1 MOTIVATIONS

Optimistic initialisation does not work with neural networks. For an optimistic initialisation to benefit exploration, the  $Q$ -values must start sufficiently high. More importantly, the values for unseen state-action pairs must remain high, until they are updated. When using a deep neural network to approximate the  $Q$ -values, we can initialise the network to output optimistic values, for example, by adjusting the final bias. However, after a small amount of training, the values for novel state-action pairs may not remain high. Furthermore, due to the generalisation of neural networks we cannot know how the values for these unseen state-action pairs compare to the trained state-action pairs. Figure 2 (left), which illustrates this effect for a simple regression task, shows that different initialisations can lead to dramatically different generalisations. It is therefore prohibitively difficult to use optimistic initialisation of a deep neural network to drive exploration.

Instead, we augment our  $Q$ -value estimates with an optimistic bonus. Our motivation for the form of the bonus in equation 1,  $\frac{C}{(N(s,a) + 1)^M}$ , stems from UCB-H (Jin et al., 2018), where all tabular  $Q$ -values are initialised with  $H$  and the first update for a state-action pair completely overwrites that value because the learning rate for the update  $(\eta_1)$  is 1. One can alternatively view these  $Q$ -values as zero-initialised with the additional term  $Q(s,a) + H \cdot \mathbb{1}\{N(s,a) < 1\}$ , where  $N(s,a)$  is the visitation count for the state-action pair  $(s,a)$ . Our approach approximates the discrete indicator function  $\mathbb{1}$  as  $(N(s,a) + 1)^{-M}$  for sufficiently large  $M$ . However, since gradient descent cannot completely overwrite the  $Q$ -value estimate for a state-action pair after a single update, it is beneficial to have a smaller hyperparameter  $M$  that governs how quickly the optimism decays.

Algorithm 1 OPIQ algorithm  
Initialize  $Q_{t}(s,a)\gets 0,N(s,a,t)\gets 0,\forall (s,a,t)\in \mathcal{S}\times \mathcal{A}\times \{1,\dots,H,H + 1\}$    
for each episode  $k = 1,\ldots ,K$  do for each timestep  $t = 1,\ldots ,H$  do Take action  $a_{t}\gets \arg \max_{a}Q_{t}^{+}(s_{t},a)$  Receive  $r(s_t,a_t,t)$  and  $s_{t + 1}$  Increment  $N(s_{t},a_{t},t)$ $Q_{t}(s_{t},a_{t})\gets (1 - \eta_{N})Q_{t}(s_{t},a_{t}) + \eta_{N}(r(s_{t},a_{t},t) + b_{N}^{T} + \min \{H,\max_{a^{\prime}}Q_{t + 1}^{+}(s_{t + 1},a^{\prime})\}).$  end   
end

For a worst case analysis we assume all  $Q$ -value estimates are pessimistic. In the common scenario where all rewards are nonnegative, the lowest possible return for an episode is zero. If we then zero-initialise our  $Q$ -value estimates, as is common for neural networks, we are starting with a pessimistic initialisation. As shown in Figure 2(left), we cannot predict how a neural network will generalise, and thus we cannot predict if the  $Q$ -value estimates for unvisited state-action pairs will be optimistic or pessimistic. We thus assume they are pessimistic in order to perform a worst case analysis. However, this is not a requirement: our method works with any initialisation and rewards.

In order to then approximate an optimistic initialisation, the scaling parameter  $C$  in equation 1 can be chosen to guarantee unseen  $Q^{+}$ -values are overestimated, for example,  $C \coloneqq H$  in the undiscounted finite-horizon tabular setting and  $C \coloneqq 1 / (1 - \gamma)$  in the discounted episodic setting (assuming 1 is the maximum reward obtainable at each timestep). However, in some environments it may be beneficial to use a smaller parameter  $C$  for faster convergence. These  $Q^{+}$ -values are then used both during action selection and during bootstrapping. Note that in the finite horizon setting the counts  $N$ , and thus  $Q^{+}$ , would depend on the timestep  $t$ .

Hence, we split the optimistic  $Q^{+}$ -values into two parts: a pessimistic  $Q$ -value component and an optimistic component based solely on the counts for a state-action pair. This separates our source of optimism from the neural network function approximator, yielding  $Q^{+}$ -values that remain high for unvisited state-action pairs, assuming a suitable counting scheme. Figure 2 (right) shows the effects of adding this optimistic component to a network's outputs.

# 3.2 TABULAR REINFORCEMENT LEARNING

In order to ensure that OPIQ has a strong theoretical foundation, we must ensure it is provably efficient in the tabular domain. We restrict our analysis to the finite horizon tabular setting and only consider building upon UCB-H (Jin et al., 2018) for simplicity. Achieving a better regret bound using UCB-B (Jin et al., 2018) and extending the analysis to the infinite horizon discounted setting (Dong et al., 2019) are steps for future work.

Our algorithm removes the optimistic initialisation of UCB-H, instead using a pessimistic initialisation (all  $Q$ -values start at 0). We then use our  $Q^{+}$ -values during action selection and bootstrapping. Pseudocode is presented in Algorithm 1.

Theorem 1. For any  $p \in (0,1)$ , with probability at least  $1 - p$  the total regret of  $Q^{+}$  is at most  $\mathcal{O}(\sqrt{H^4SAT\log(SAT / p)})$  for  $M \geq 1$  and at most  $\mathcal{O}(H^{1 + M}SAT^{1 - M} + \sqrt{H^4SAT\log(SAT / p)})$  for  $0 < M < 1$ .

The proof is based on that of Theorem 1 from (Jin et al., 2018). Our  $Q^{+}$ -values are always greater than or equal to the  $Q$ -values that UCB-H would estimate, thus ensuring that our estimates are also greater than or equal to  $Q^{*}$ . Our overestimation relative to UCB-H is then governed by the quantity  $H / (N(s,a) + 1)^M$ , which when summed over all timesteps does not depend on  $T$  for  $M > 1$ . As  $M \to \infty$  we exactly recover UCB-H, and match the asymptotic performance of UCB-H for  $M \geq 1$ . Smaller values of  $M$  result in our optimism decaying more slowly, which results in more exploration. The full proof is included in Appendix I.

We also show that OPIQ without optimistic action selection or the count-based intrinsic motivation term  $b_{N}^{T}$  is not provably efficient by showing it can incur linear regret with high probability on simple MDPs (see Appendices G and H).

Our primary motivation for considering a tabular algorithm that pessimistically initialises its  $Q$ -values, is to provide a firm theoretical foundation on which to base a deep RL algorithm, which we describe in the next section.

# 3.3 DEEP REINFORCEMENT LEARNING

For deep RL, we base OPIQ on DQN (Mnih et al., 2015), which uses a deep neural network with parameters  $\theta$  as a function approximator  $Q_{\theta}$ . During action selection, we use our  $Q^{+}$ -values to determine the greedy action:

$$
a _ {t} = \underset {a} {\arg \max } \left\{Q _ {\theta} (s, a) + \frac {C _ {\text {a c t i o n}}}{(N (s , a) + 1) ^ {M}} \right\}, \tag {3}
$$

where  $C_{\text{action}}$  is a hyperparameter governing the scale of the optimistic bias during action selection. In practice, we use an  $\epsilon$ -greedy policy. After every timestep, we sample a batch of experiences from our experience replay buffer, and use  $n$ -step  $Q$ -learning (Mnih et al., 2016). We recompute the counts for each relevant state-action pair, to avoid using stale pseudo-rewards. The network is trained by gradient descent on the loss in equation 2 with the target:

$$
y _ {t} := \sum_ {i = 0} ^ {n - 1} \gamma^ {i} \left(r \left(s _ {t + i}, a _ {t + i}\right) + \frac {\beta}{\sqrt {N \left(s _ {t + i} , a _ {t + i}\right)}}\right) + \gamma^ {n} \max  _ {a ^ {\prime}} \left\{Q _ {\theta^ {-}} \left(s _ {t + n}, a ^ {\prime}\right) + \frac {C _ {\mathrm {b o o t s t r a p}}}{\left(N \left(s _ {t + n} , a ^ {\prime}\right) + 1\right) ^ {M}} \right\}. \tag {4}
$$

where  $C_{\mathrm{bootstrap}}$  is a hyperparameter that governs the scale of the optimistic bias during bootstrapping.

We use the method of static hashing (Tang et al., 2017) to obtain our pseudocounts for its generality and simplicity. More details can be found in Appendix B.

A DQN with pseudocount derived intrinsic reward (DQN + PC) (Bellemare et al., 2016) can be seen as a naive extension of UCB-H to the deep RL setting. However, it does not attempt to ensure optimism in the  $Q$ -values used during action selection and bootstrapping, which is a crucial component of UCB-H. Furthermore, even if the  $Q$ -values were initialised optimistically at the start of training they would not remain optimistic long enough to drive exploration, due to the use of neural networks. OPIQ, on the other hand, is designed with these limitations of neural networks in mind. By augmenting the neural network's  $Q$ -value estimates with optimistic bonuses of the form  $\frac{C}{(N(s,a) + 1)^M}$ , OPIQ ensures that the  $Q^{+}$ -values used during action selection and bootstrapping are optimistic. We can thus consider OPIQ as a deep version of UCB-H. Our results show that optimism during action selection and bootstrapping is extremely important for ensuring efficient exploration.

# 4 RELATED WORK

Tabular Domain: There is a wealth of literature related to provably efficient exploration in the tabular domain. Popular model-based algorithms such as R-MAX (Brafman and Tennenholtz, 2002), MBIE (and MBIE-EB) (Strehl and Littman, 2008), UCRL2 (Jaksch et al., 2010) and UCBVI (Azar et al., 2017) are all based on the principle of optimism in the face of uncertainty. Osband and Van Roy (2017) adopt a Bayesian viewpoint and argue that posterior sampling (PSRL) (Strens, 2000) is more practically efficient than approaches that are optimistic in the face of uncertainty, and prove that in Bayesian expectation PSRL matches the performance of any optimistic algorithm up to constant factors. Agrawal and Jia (2017) prove that an optimistic variant of PSRL is provably efficient under a frequentist regret bound.

The only provably efficient model-free algorithms to date are delayed  $Q$ -learning (Strehl et al., 2006) and UCB-H (and UCB-B) (Jin et al., 2018). Delayed  $Q$ -learning optimistically initialises the  $Q$ -values that are carefully controlled when they are updated. UCB-H and UCB-B also optimistically initialise the  $Q$ -values, but also utilise a count-based intrinsic motivation term and a special learning rate to achieve a favourable regret bound compared to model-based algorithms. In contrast, OPIQ pessimistically initialises the  $Q$ -values. Whilst we base our current analysis on UCB-H, the idea of augmenting pessimistically initialised  $Q$ -values can be applied to any model-free algorithm.

Deep RL Setting: A popular approach to improving exploration in deep RL is to utilise intrinsic motivation (Oudeyer and Kaplan, 2009), which computes a quantity to add to the environmental reward. Most relevant to our work is that of Bellemare et al. (2016), which takes inspiration from MBIE-EB (Strehl and Littman, 2008). Bellemare et al. (2016) utilise the number of times a state has

been visited to compute the intrinsic reward. They outline a framework for obtaining approximate counts, dubbed pseudocounts, through a learned density model over the state space. Ostrovski et al. (2017) extend the work to utilise a more expressive PixelCNN (van den Oord et al., 2016) as the density model, whereas Fu et al. (2017) train a neural network as a discriminator to also recover a density model. Machado et al. (2018) instead use the successor representation to obtain generalised counts. Choi et al. (2019) learn a feature space to count that focusses on regions of the state space the agent can control, and Pathak et al. (2017) learn a similar feature space in order to provide the error of a learned model as intrinsic reward. A simpler and more generic approach to approximate counting is static hashing which projects the state into a lower dimensional space before counting (Tang et al., 2017). None of these approaches attempt to augment or modify the  $Q$ -values used for action-selection or bootstrapping, and hence do not attempt to ensure optimistic values for novel state-action pairs.

Chen et al. (2017) build upon bootstrapped DQN (Osband et al., 2016) to obtain uncertainty estimates over the  $Q$ -values for a given state in order to act optimistically by choosing the action with the largest UCB. However, they do not utilise optimistic estimates during bootstrapping. Osband et al. (2018) also extend bootstrapped DQN to include a prior by extending RLSVI (Osband et al., 2017) to deep RL. Osband et al. (2017) show that RLSVI achieves provably efficient Bayesian expected regret, which requires a prior distribution over MDPs, whereas OPIQ achieves provably efficient worse case regret. Bootstrapped DQN with a prior is thus a model-free algorithm that has strong theoretical support in the tabular setting. Empirically, however, its performance on sparse reward tasks is worse than DQN with pseudocounts.

Machado et al. (2015) shift and scale the rewards so that a zero-initialisation is optimistic. When applied to neural networks this approach does not result in optimistic  $Q$ -values due to the generalisation of the networks. Bellemare et al. (2016) empirically show that using a pseudocount intrinsic motivation term performs much better empirically on hard exploration tasks.

Choshen et al. (2018) attempt to generalise the notion of a count to include information about the counts of future state-action pairs in a trajectory, which they use to provide bonuses during action selection. Oh and Iyengar (2018) extend delayed  $Q$ -learning to utilise these generalised counts and prove the scheme is PAC-MDP. The generalised counts are obtained through  $E$ -values which are learnt using SARSA with a constant 0 reward and  $E$ -value estimates initialised at 1. When scaling to the deep RL setting, these  $E$ -values are estimated using neural networks that cannot maintain their initialisation for unvisited state-action pairs, which is crucial for providing an incentive to explore. By contrast, OPIQ uses a separate source to generate the optimism necessary to explore the environment.

# 5 EXPERIMENTAL SETUP

We compare OPIQ against baselines and ablations on two sparse reward environments. The first is a randomized version of the Chain environment proposed by Osband et al. (2016) and used in (Shyam et al., 2019) with a chain of length 100, which we call Randomised Chain. The second is a two-dimensional maze in which the agent starts in the top left corner (white dot) and is only rewarded upon finding the goal (light grey dot). We use an image of the maze as input and randomise the actions similarly to the chain. See Appendix D for further details on the environments, baselines and hyperparameters used.

# 5.1 ABLATIONS AND BASELINES

We compare OPIQ against a variety of DQN-based approaches that use pseudocount intrinsic rewards, the DORA agent (Choshen et al., 2018) (which generates count-like optimism bonuses using a neural network), and two strong exploration baselines:

$\epsilon$ -greedy DQN: a standard DQN that uses an  $\epsilon$ -greedy policy to encourage exploration. We anneal  $\epsilon$  linearly over a fixed number of timesteps from 1 to 0.01.

$\mathbf{DQN} + \mathbf{PC}$ : we add an intrinsic reward of  $\beta / \sqrt{N(s, a)}$  to the environmental reward based on (Bellemare et al., 2016; Tang et al., 2017).

DQN R-Subtract (+PC): we subtract a constant from all environmental rewards received when training, so that a zero-initialisation is optimistic, as described for a DQN in (Bellemare et al., 2016) and based on Machado et al. (2015).

DQN Bias (+PC): we initialise the bias of the final layer of the DQN to a positive value at the start of training as a simple method for optimistic initialisation with neural networks.

$\mathbf{DQN} + \mathbf{DORA}$ : we use the generalised counts from (Choshen et al., 2018) as an intrinsic reward.

$\mathbf{DQN} + \mathbf{DORA}$  OA: we additionally use the generalised counts to provide an optimistic bonus during action selection.

$\mathbf{DQN} + \mathbf{RND}$ : we add the RND bonus from (Burda et al., 2018) as an intrinsic reward.

BSP: we use Bootstrapped DQN with randomised prior functions (Osband et al., 2018).

In order to better understand the importance of each component of our method, we also evaluate the following ablations:

Optimistic Action Selection (OPIQ w/o OB): we only use our  $Q^{+}$ -values during action selection, and use  $Q$  during bootstrapping (without Optimistic Bootstrapping). The intrinsic motivation term remains.

Optimistic Action Selection and Bootstrapping (OPIQ w/o PC): we use our  $Q^{+}$ -values during action selection and bootstrapping, but do not include an intrinsic motivation term (without Pseudo Counts).

# 6 RESULTS

# 6.1 RANDOMISED CHAIN

We first consider the visually simple domain of the randomised chain and compare the count-based methods. Figure 3 shows the performance of OPIQ compared to the baselines and ablations. OPIQ significantly outperforms the baselines, which do not have any explicit mechanism for optimism during action selection. A DQN with pseudocount derived intrinsic rewards is unable to reliably find the goal state, but setting the final layer's bias to one produces much better performance. For the DQN variant in which a constant is subtracted from all rewards, all of the configurations (including those with pseudocount derived intrinsic bonuses) were unable to find the goal on the right and thus the agents learn quickly to latch on the inferior reward of moving left.

Compared to its ablations, OPIQ is more stable in this task. OPIQ without pseudocounts performs similarly to OPIQ but is more varied across seeds, whereas the lack of optimistic bootstrapping results in worse performance and significantly more variance across seeds.

# 6.2 MAZE

We next consider the harder and more visually complex task of the Maze and compare against all baselines. Figure 4 shows that only OPIQ and ablations are able to find the goal in the sparse reward maze. This indicates that optimistic action selection can have a significant positive impact in sparse reward tasks, and shows that a naive extension of UCB-H to the deep RL setting (DQN + PC) results in insufficient exploration.

We also see in Figure 4 (right) that OPIQ explores significantly more states than the baselines and also explores slightly faster than its ablations (right), which shows the benefits of optimism during both action selection and bootstrapping. In addition, the episodic reward for the ablation without

![](images/57ae783309361017212d29cef2e9affef55c19503b48761a87a2bf6b4ed0e3ab.jpg)  
Figure 3: Results for the randomised chain environment. Median across 20 seeds is plotted and the  $25\% -75\%$  quartile is shown shaded. Left: OPIQ outperforms the baselines. Right: OPIQ is more stable than its ablations.

![](images/e1c2f4d5bcd8c98c0ace58a277f6822dff4422b6a9dabc6fb5446176de978b08.jpg)

![](images/1a92f8ff27d3b665bb219bfa7b8a46efdd8fe895ca46995ff81559504f52aeab.jpg)  
Figure 4: Results for the maze environment. Median across 8 seeds is plotted and the  $25\% -75\%$  quartile is shown shaded. Left: The episode reward. Right: Number of distinct states visited over training. The total number of states in the environment is shown as a dotted line.

![](images/63e8e319bc3c3dc05cd7a38d2e79d0bec8aebcff4cead35074666b1133c99df0.jpg)

optimistic bootstrapping is noticeably more unstable (Figure 4, left). Interestingly, OPIQ without pseudocounts performs significantly worse than the other ablations. This is surprising since the theory suggests that the count-based intrinsic motivation is only required when the reward or transitions of the MDP are stochastic (Jin et al., 2018), which is not the case here. We hypothesise that adding PC-derived intrinsic bonuses to the reward provides an easier learning problem, especially when using  $n$ -step  $Q$ -Learning, which yields the performance gap. However, our results show that the PC-derived intrinsic bonuses are not enough on their own to ensure sufficient exploration as shown by the differences between DQN + PC and OPIQ.

As expected DQN + RND performs poorly on this domain. The visual input does not vary much across the state space, resulting in the RND bonus failing to provide enough intrinsic motivation to ensure efficient exploration. Additionally it does not feature any explicit mechanism for optimism during action selection.

Both DQN+DORA and DQN+DORA OA perform poorly in this domain since their source of intrinsic motivation disappears quickly. As noted in Figure 2, neural networks do not maintain their starting initialisations after training. Thus, the intrinsic reward DORA produces goes to 0 quickly since the network producing its bonuses learns to generalise quickly. BSP is the only baseline we test that does not add an intrinsic reward to the environmental reward, and thus it performs poorly compared to the other baselines.

Figure 5 visualises the values used during action selection for a DQN + PC agent and OPIQ, showing the count-based augmentation provides optimism for relatively novel state-action pairs, driving the agent to explore more of the state-action space.

![](images/51045d0931359a6512fb13e4272cf9422da0894deb320a2a0ea095d8620ab188.jpg)  
Figure 5: Values used during action selection for each of the 4 actions. The region in blue indicates states that have already been visited. Other colours denote  $Q$ -values between 0 (black) and 10 (white). Left: The  $Q$ -values used by DQN with pseudocounts. Right:  $Q^{+}$ -values used by OPIQ with  $C_{\text{action}} = 100$ .

![](images/d644baaccaba9769c91d2277efd850ef2ee959c050f9b45db0cb0f13ace2c66d.jpg)

# 7 CONCLUSIONS AND FUTURE WORK

This paper presented OPIQ, a model-free algorithm that does not rely on an optimistic initialisation to ensure efficient exploration. Instead, OPIQ augments the  $Q$ -values estimates with a count-based optimism bonus. We showed that this is provably efficient in the tabular setting by modifying UCB-H to use a pessimistic initialisation and our augmented  $Q^{+}$ -values for action selection and bootstrapping. Since our method does not rely on a specific initialisation scheme, it easily scales to deep RL when paired with an appropriate counting scheme. Our results showed the benefits of maintaining optimism both during action selection and bootstrapping for exploration. In future work, we aim to extend OPIQ by integrating it with more expressive counting schemes.

# REFERENCES

Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In Advances in Neural Information Processing Systems, pages 1184-1194, 2017.  
Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning, pages 263-272, 2017.  
Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information Processing Systems, pages 1471-1479, 2016.  
Ronen I Brafman and Moshe Tennenholtz. R-MAX - a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research, 3(Oct):213-231, 2002.  
Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.  
Richard Y Chen, John Schulman, Pieter Abbeel, and Szymon Sidor. UCB and infogain exploration via  $Q$ -ensembles. arXiv preprint arXiv:1706.01502, 2017.  
Jongwook Choi, Yijie Guo, Marcin Moczulski, Junhyuk Oh, Neal Wu, Mohammad Norouzi, and Honglak Lee. Contingency-aware exploration in reinforcement learning. In International Conference on Learning Representations, 2019.  
Leshem Choshen, Lior Fox, and Yonatan Loewenstein. Dora the explorer: Directed outreach reinforcement action-selection. In International Conference on Learning Representations, 2018.  
Kefan Dong, Yuanhao Wang, Xiaoyu Chen, and Liwei Wang.  $Q$ -learning with UCB exploration is sample efficient for infinite-horizon MDP. arXiv preprint arXiv:1901.09311, 2019.  
Li Fan, Pei Cao, Jussara Almeida, and Andrei Z Broder. Summary cache: a scalable wide-area web cache sharing protocol. IEEE/ACM Transactions on Networking (TON), 8(3):281-293, 2000.  
Justin Fu, John Co-Reyes, and Sergey Levine. EX2: Exploration with exemplar models for deep reinforcement learning. In Advances in Neural Information Processing Systems, pages 2577-2587, 2017.  
Sudhir K Goel and Dennis M Rodriguez. A note on evaluating limits using riemann sums. Mathematics Magazine, 60(4):225-228, 1987.  
Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In Proceedings of the 36th International Conference on Machine learning, 2019.  
Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(Apr):1563-1600, 2010.  
Chi Jin, Zeyuan Allen-Zhu, Sebastian Bubeck, and Michael I Jordan. Is  $Q$ -learning provably efficient? In Advances in Neural Information Processing Systems, pages 4863-4873, 2018.  
Long-Ji Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3-4):293-321, 1992.  
Marlos C Machado, Sriram Srinivasan, and Michael H Bowling. Domain-independent optimistic initialization for reinforcement learning. In AAAI Workshop: Learning for General Competency in Video Games, 2015.

Marlos C Machado, Marc G Bellemare, and Michael Bowling. Count-based exploration with the successor representation. arXiv preprint arXiv:1807.11622, 2018.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning, pages 1928-1937, 2016.  
Min-hwan Oh and Garud Iyengar. Directed exploration in PAC model-free reinforcement learning. arXiv preprint arXiv:1808.10552, 2018.  
Ian Osband and Benjamin Van Roy. Why is posterior sampling better than optimism for reinforcement learning? In Proceedings of the 34th International Conference on Machine Learning, pages 2701-2710, 2017.  
Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped DQN. In Advances in Neural Information Processing Systems, pages 4026-4034, 2016.  
Ian Osband, Daniel Russo, Zheng Wen, and Benjamin Van Roy. Deep exploration via randomized value functions. arXiv preprint arXiv:1703.07608, 2017.  
Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep reinforcement learning. In Advances in Neural Information Processing Systems, pages 8617-8629, 2018.  
Georg Ostrovski, Marc G Bellemare, Aaron van den Oord, and Rémi Munos. Count-based exploration with neural density models. arXiv preprint arXiv:1703.01310, 2017.  
Pierre-Yves Oudeyer and Frederic Kaplan. What is intrinsic motivation? a typology of computational approaches. Frontiers in neurorobotics, 1:6, 2009.  
Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the 34th International Conference on Machine Learning, 2017.  
Pranav Shyam, Wojciech Jaśkowski, and Faustino Gomez. Model-based active exploration. In Proceedings of the 36th International Conference on Machine Learning, 2019.  
Alexander L Strehl and Michael L Littman. An analysis of model-based interval estimation for markov decision processes. Journal of Computer and System Sciences, 74(8):1309-1331, 2008.  
Alexander L Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael L Littman. PAC model-free reinforcement learning. In Proceedings of the 23rd International Conference on Machine learning, pages 881-888. ACM, 2006.  
Malcolm Strens. A bayesian framework for reinforcement learning. In Proceedings of the 17th International Conference on Machine Learning, pages 943-950, 2000.  
Adrien Ali Taïga, Aaron Courville, and Marc G Bellemare. Approximate exploration through state abstraction. arXiv preprint arXiv:1808.09819, 2018.  
Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. #Exploration: A study of count-based exploration for deep reinforcement learning. In Advances in Neural Information Processing Systems, pages 2753-2762, 2017.  
Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelCNN decoders. In Advances in Neural Information Processing Systems, pages 4790-4798, 2016.
