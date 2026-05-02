# THE REACTOR: A FAST AND SAMPLE-EFFICIENT ACTOR-CRITIC AGENT FOR REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this work we present a new agent architecture, called Reactor, which combines multiple algorithmic and architectural contributions to produce an agent with higher sample-efficiency than Prioritized Dueling DQN (Wang et al., 2016) and Categorical DQN (Bellemare et al., 2017), while giving better run-time performance than A3C (Mnih et al., 2016). Our first contribution is a new policy evaluation algorithm called Distributional Retrace, which brings multi-step off-policy updates to the distributional reinforcement learning setting. The same approach can be used to convert several classes of multi-step policy evaluation algorithms designed for expected value evaluation into distributional ones. Next, we introduce the  $\beta$ -leave-one-out policy gradient algorithm which improves the trade-off between variance and bias by using action values as a baseline. Our final algorithmic contribution is a new prioritized replay algorithm for sequences, which exploits the temporal locality of neighboring observations for more efficient replay prioritization. Using the Atari 2600 benchmarks, we show that each of these innovations contribute to both the sample efficiency and final agent performance. Finally, we demonstrate that Reactor reaches state-of-the-art performance after 200 million frames and less than a day of training.

# 1 INTRODUCTION

Model-free deep reinforcement learning has achieved several remarkable successes in domains ranging from super-human-level control in video games (Mnih et al., 2015) and the game of Go (Silver et al., 2016; 2017), to continuous motor control tasks (Lillicrap et al., 2015; Schulman et al., 2015).

Much of the recent work can be divided into two categories. First, those of which that, often building on the DQN framework, act  $\epsilon$ -greedily according to an action-value function and train using minibatches of transitions sampled from an experience replay buffer (Van Hasselt et al., 2016; Wang et al., 2015; He et al., 2017; Anschel et al., 2017). These value-function agents benefit from improved sample complexity, but tend to suffer from long runtimes (e.g. DQN requires approximately a week to train on Atari). The second category are the actor-critic agents, which includes the asynchronous advantage actor-critic (A3C) algorithm, introduced by (Mnih et al., 2016). These agents train on transitions collected by multiple actors running, and often training, in parallel (Schulman et al., 2017; Vezhnevets et al., 2017). The deep actor-critic agents train on each trajectory only once, and thus tend to have worse sample complexity. However, their distributed nature allows significantly faster training in terms of wall-clock time. Still, not all existing algorithms can be put in the above two categories and various hybrid approaches do exist (Zhao et al., 2016; O'Donoghue et al., 2017; Gu et al., 2017; Wang et al., 2017).

Data-efficiency and off-policy learning are essential for many real-world domains where interactions with the environment are expensive. Similarly, wall-clock time (time-efficiency) directly impacts an algorithm's applicability through resource costs. The focus of this work is to produce an agent that is sample- and time-efficient. To this end, we introduce a new reinforcement learning agent, called Reactor (Retrace-Actor), which takes a principled approach to combining the sample-efficiency of off-policy experience replay with the time-efficiency of asynchronous algorithms. We combine recent

advances in both categories of agents with novel contributions to produce an agent that inherits the benefits of both and reaches state-of-the-art performance over 57 Atari 2600 games.

Our primary contributions are (1) a novel policy gradient algorithm,  $\beta$ -LOO, which makes better use of action-value estimates to improve the policy gradient; (2) the first multi-step off-policy distributional reinforcement learning algorithm, distributional Retrace(λ); (3) a novel prioritized replay for off-policy sequences of transitions; and (4) an optimized network and parallel training architecture.

We begin by reviewing background material, including relevant improvements to both value-function agents and actor-critic agents. In Section 3 we introduce each of our primary contributions and present the Reactor agent. Finally, in Section 4, we present experimental results on the 57 Atari 2600 games from the Arcade Learning Environment (ALE) (Bellemare et al., 2013), as well as a series of ablation studies for the various components of Reactor.

# 2 BACKGROUND

We consider a Markov decision process (MDP) with state space  $X$  and finite action space  $A$ . A (stochastic) policy  $\pi(\cdot|x)$  is a mapping from states  $x \in X$  to a probability distribution over actions. We consider a  $\gamma$ -discounted infinite-horizon criterion, with  $\gamma \in [0,1)$  the discount factor, and define for policy  $\pi$  the action-value of a state-action pair  $(x,a)$  as

$$
Q ^ {\pi} (x, a) \stackrel {\mathrm {d e f}} {=} \mathbb {E} \Big [ \sum_ {t \geq 0} \gamma^ {t} r _ {t} | x _ {0} = x, a _ {0} = a, \pi \Big ],
$$

where  $\left(\{x_{t}\}_{t\geq 0}\right)$  is a trajectory generated by choosing  $a$  in  $x$  and following  $\pi$  thereafter, i.e.,  $a_{t} \sim \pi(\cdot|x_{t})$  (for  $t \geq 1$ ), and  $r_{t}$  is the reward signal. The objective in reinforcement learning is to find an optimal policy  $\pi^{*}$ , which maximises  $Q^{\pi}(x,a)$ . The optimal action-values are given by  $Q^{*}(x,a) = \max_{\pi} Q^{\pi}(x,a)$ .

# 2.1 VALUE-BASED ALGORITHMS

The Deep Q-Network (DQN) framework, introduced by (Mnih et al., 2015), popularised the current line of research into deep reinforcement learning by reaching human-level, and beyond, performance across 57 Atari 2600 games in the ALE. While DQN includes many specific components, the essence of the framework, much of which is shared by Neural Fitted Q-Learning (Riedmiller, 2005), is to use of a deep convolutional neural network to approximate an action-value function, training this approximate action-value function using the Q-Learning algorithm (Watkins & Dayan, 1992) and mini-batches of one-step transitions  $(x_{t},a_{t},r_{t},x_{t + 1},\gamma_{t})$  drawn randomly from an experience replay buffer (Lin, 1992). Additionally, the next-state action-values are taken from a target network, which is updated to match the current network periodically. Thus, the temporal difference (TD) error for transition  $t$  used by these algorithms is given by

$$
\delta_ {t} = r _ {t} + \gamma_ {t} \max  _ {a ^ {\prime} \in A} Q \left(x _ {t + 1}, a ^ {\prime}; \bar {\theta}\right) - Q \left(x _ {t}, a _ {t}; \theta\right), \tag {1}
$$

where  $\theta$  denotes the parameters of the network and  $\bar{\theta}$  are the parameters of the target network.

Since this seminal work, we have seen numerous extensions and improvements that all share the same underlying framework. Double DQN (Van Hasselt et al., 2016), attempts to correct for the over-estimation bias inherent in Q-Learning by changing the second term of (1) to  $Q(x_{t + 1},\arg \max_{a'\in A}Q(x_{t + 1},a';\theta);\bar{\theta})$ . The dueling architecture (Wang et al., 2015), changes the network to estimate action-values using separate network heads  $V(x;\theta)$  and  $A(x,a;\theta)$  with

$$
Q (x, a; \theta) = V (x; \theta) + A (x, a; \theta) - \frac {1}{| A |} \sum_ {a ^ {\prime}} A (x, a ^ {\prime}; \theta).
$$

Recently, (Hessel et al., 2017) introduced Rainbow, a value-based reinforcement learning agent combining many of these improvements into a single agent and demonstrating that they are largely complementary. Rainbow significantly out performs previous methods, but also inherits the poorer time-efficiency of the DQN framework. In the remainder of the section we will describe in more depth other recent improvements to DQN.

# 2.1.1 PRIORITIZED EXPERIENCE REPLAY

The experience replay buffer was first introduced by (Lin, 1992) and later used in DQN (Mnih et al., 2015). Typically, the replay buffer is essentially a first-in-first-out queue with new transitions gradually replacing older transitions. The agent would then sample a mini-batch uniformly at random from the replay buffer. Drawing inspiration from prioritized sweeping (Moore & Atkeson, 1993), prioritized experience replay replaces the uniform sampling with prioritized sampling proportional to the absolute TD error (Schaul et al., 2016).

Specifically, for a replay buffer of size  $N$ , prioritized experience replay samples transition  $t$  with probability  $P(t)$ , and applies weighted importance-sampling with  $w_{t}$  to correct for the prioritization bias, where

$$
P (t) = \frac {p _ {t} ^ {\alpha}}{\sum_ {k} p _ {k} ^ {\alpha}}, \quad w _ {t} = \left(\frac {1}{N} \cdot \frac {1}{P (t)}\right) ^ {\beta}, \quad p _ {t} = | \delta_ {t} | + \epsilon , \quad \alpha , \beta , \epsilon > 0. \tag {2}
$$

Prioritized DQN significantly increases both the sample-efficiency and final performance over DQN on the Atari 2600 benchmarks (Schaul et al., 2015).

# 2.1.2 RETRACE(λ)

Retrace  $(\lambda)$  is a convergent off-policy multi-step algorithm extending the DQN agent (Munos et al., 2016). Assume that some trajectory  $\{x_0, a_0, r_0, x_1, a_1, r_1, \ldots, x_t, a_t, r_t, \ldots,\}$  has been generated according to behaviour policy  $\mu$ , i.e.,  $a_t \sim \mu(\cdot | x_t)$ . Now we aim to evaluate the value of a different target policy  $\pi$ , i.e., we want to estimate  $Q^{\pi}$ . The Retrace algorithm will update our current estimate  $Q$  of  $Q^{\pi}$  in the direction of

$$
\Delta Q \left(x _ {t}, a _ {t}\right) \stackrel {\text {d e f}} {=} \sum_ {s \geq t} \gamma^ {s - t} \left(c _ {t + 1} \dots c _ {s}\right) \delta_ {s} ^ {\pi} Q, \tag {3}
$$

where  $\delta_s^\pi Q \stackrel{\mathrm{def}}{=} r_s + \gamma \mathbb{E}_\pi [Q(x_{s+1}, \cdot)] - Q(x_s, a_s)$  is the temporal difference at time  $s$  under  $\pi$ , and

$$
c _ {s} = \lambda \min  \left(1, \rho_ {s}\right), \quad \rho_ {s} = \frac {\pi \left(a _ {s} \mid x _ {s}\right)}{\mu \left(a _ {s} \mid x _ {s}\right)}. \tag {4}
$$

The Retrace algorithm comes with the theoretical guarantee that in finite state and action spaces, repeatedly updating our current estimate  $Q$  according to (3) produces a sequence of  $Q$  functions which converges to  $Q^{\pi}$  for a fixed  $\pi$  or to  $Q^{*}$  if we consider a sequence of policies  $\pi$  which become increasing greedy w.r.t. the  $Q$  estimates (Munos et al., 2016).

# 2.1.3 DISTRIBUTIONAL RL

Distributional reinforcement learning refers to a class of algorithms that directly estimate the distribution over returns, whose expectation gives the traditional value function (Bellemare et al., 2017). Such approaches can be made tractable with a distributional Bellman equation, and the recently proposed algorithm  $C51$  showed state-of-the-art performance in the Atari 2600 benchmarks.  $C51$  parameterizes the distribution over returns with a mixture over Dirac's centered on a uniform grid,

$$
Q (x, a; \theta) = \sum_ {i = 0} ^ {N - 1} q _ {i} (x, a; \theta) z _ {i}, \quad q _ {i} = \frac {e ^ {\theta_ {i} (x , a)}}{\sum_ {j = 0} ^ {N - 1} e ^ {\theta_ {j} (x , a)}}, \quad z _ {i} = v _ {\min } + i \frac {v _ {\max } - v _ {\min }}{N - 1}, \tag {5}
$$

with hyperparameters  $v_{\mathrm{min}}$ ,  $v_{\mathrm{max}}$  that bound the distribution support of size  $N$ .

# 2.2 ACTOR-CRITIC ALGORITHMS

In this section we review the actor-critic framework for reinforcement learning algorithms and then discuss recent advances in actor-critic algorithms along with their various trade-offs. The asynchronous advantage actor-critic (A3C) algorithm (Mnih et al., 2016), maintains a parameterized policy  $\pi(a|x;\theta)$  and value function  $V(x;\theta_v)$ , which are updated with

$$
\triangle \theta = \nabla_ {\theta} \log \pi \left(a _ {t} \mid x _ {t}; \theta\right) A \left(x _ {t}, a _ {t}; \theta , \theta_ {v}\right), \quad \triangle \theta_ {v} = A \left(x _ {t}, a _ {t}; \theta , \theta_ {v}\right) \nabla_ {\theta_ {v}} V (x _ {t}), \tag {6}
$$

$$
\text {w h e r e}, \quad A \left(x _ {t}, a _ {t}; \theta , \theta_ {v}\right) = \sum_ {k} ^ {n - 1} \gamma^ {k} r _ {t + k} + \gamma^ {n} V \left(x _ {t + n}\right) - V \left(x _ {t}\right). \tag {7}
$$

A3C uses  $M = 16$  parallel CPU workers, each acting independently in the environment and applying the above updates asynchronously to a shared set of parameters. In contrast to the previously discussed value-based methods, A3C is an on-policy algorithm, and does not use a GPU nor a replay buffer.

Proximal Policy Optimization (PPO) is a closely related actor-critic algorithm (Schulman et al., 2017), which replaces the advantage (7) with,

$$
\min  \left(\rho_ {t} A \left(x _ {t}, a _ {t}; \theta , \theta_ {v}\right), c l i p \left(\rho_ {t}, 1 - \epsilon , 1 + \epsilon\right) A \left(x _ {t}, a _ {t}; \theta , \theta_ {v}\right)\right), \epsilon > 0,
$$

where  $\rho_{t}$  is as defined in Section 2.1.2. Although both PPO and A3C run  $M$  parallel workers collecting trajectories independently in the environment, PPO collects these experiences to perform a single, synchronous, update in contrast with the asynchronous updates of A3C.

Actor-Critic Experience Replay (ACER) extends the A3C framework with an experience replay buffer, Retrace algorithm for off-policy corrections and Truncated Importance Sampling Likelihood Ratio (TISLR) algorithm used for off-policy policy optimization (Wang et al., 2017).

# 3 THE REACTOR

The Reactor is a combination of four novel contributions on top of recent improvements to both deep value-based RL and policy-gradient algorithms. Each contribution moves Reactor towards our goal of achieving both sample and time efficiency.

# 3.1  $\beta$ -LOO

The Reactor architecture represents both a policy  $\pi(a|x)$  and action-value function  $Q(x, a)$ . We use a policy gradient algorithm to train the actor  $\pi$  which makes use of our current estimate  $Q(x, a)$  of  $Q^{\pi}(x, a)$ . Let  $V^{\pi}(x_0)$  be the value function at some initial state  $x_0$ , the policy gradient theorem (Sutton et al., 2000) says that  $\nabla V^{\pi}(x_0) = \mathbb{E}\left[\sum_t \gamma^t \sum_a Q^{\pi}(x_t, a) \nabla \pi(a|x_t)\right]$ , where  $\nabla$  refers to the gradient w.r.t. policy parameters. We now consider several possible ways to estimate this gradient.

To simplify notation, we drop the dependence on the state  $x$  for now and consider the problem of estimating the quantity

$$
G = \sum_ {a} Q ^ {\pi} (a) \nabla \pi (a). \tag {8}
$$

In the off-policy case, we consider estimating  $G$  using a single action  $A$  drawn from a (possibly different from  $\pi$ ) behaviour distribution  $A \sim \mu$ . Let us assume that for the chosen action  $A$  we have access to an estimate  $R(A)$  of  $Q^{\pi}(A)$ . Then we can use likelihood ratio (LR) method combined with an importance sampling (IS) ratio (which we call ISLR) to build an unbiased estimate of  $G$ :

$$
\hat {G} _ {\mathrm {I S L R}} = \frac {\pi (A)}{\mu (A)} (R (A) - V) \nabla \log \pi (A),
$$

where  $V$  is a baseline that depend on the state but not on the chosen action. However this estimate suffers from high variance. A possible way for reducing variance is to estimate  $G$  directly from (8) by using the return  $R(A)$  for the chosen action  $A$  and our current estimate  $Q$  of  $Q^{\pi}$  for the other actions, which leads to the so-called leave-one-out (LOO) policy-gradient estimate:

$$
\hat {G} _ {\mathrm {L 0 0}} = R (A) \nabla \pi (A) + \sum_ {a \neq A} Q (a) \nabla \pi (a). \tag {9}
$$

This estimate has low variance but may be biased if the estimated  $Q$  values differ from  $Q^{\pi}$ . A better bias-variance tradeoff may be obtained by the more general  $\beta$ -LOO policy-gradient estimate:

$$
\hat {G} _ {\beta - \mathrm {L O O}} = \beta (R (A) - Q (A)) \nabla \pi (A) + \sum_ {a} Q (a) \nabla \pi (a), \tag {10}
$$

where  $\beta = \beta (\mu ,\pi ,A)$  can be a function of both policies  $\pi$  and  $\mu$  and the selected action  $A$ . Notice that when  $\beta = 1$ , (10) reduces to (9), and when  $\beta = 1 / \mu (A)$ , then (10) is

$$
\hat {G} _ {\frac {1}{\mu} \cdot \mathrm {L O O}} = \frac {\pi (A)}{\mu (A)} (R (A) - Q (A)) \nabla \log \pi (A) + \sum_ {a} Q (a) \nabla \pi (a). \tag {11}
$$

This estimate is unbiased and can be seen as a generalization of  $\hat{G}_{\mathrm{ISLR}}$  where instead of using a state-only dependent baseline, we use a state-and-action-dependent baseline (our current estimate  $Q$ ) and add the correction term  $\sum_{a} \nabla \pi(a) Q(a)$  to cancel the bias. Proposition 1 gives our analysis of the bias of  $G_{\beta-\mathrm{LOO}}$ , with a proof left to the Appendix.

![](images/0fc4e704602a3f6f81b44fa38dae0e6e5450c9ce46a182e7d2e9719f67a0c2e5.jpg)  
Figure 1: Single-step (left) and multi-step (right) distribution bootstrapping.

Proposition 1. Assume  $A \sim \mu$  and that  $\mathbb{E}[R(A)] = Q^{\pi}(A)$ . Then, the bias of  $G_{\beta -LOO}$  is  $\left|\sum_{a}(1 - \mu(a)\beta(a))\nabla \pi(a)[Q(a) - Q^{\pi}(a)]\right|$ .

Thus the bias is small when  $\beta (a)$  is close to  $1 / \mu (a)$ , or when the  $Q$ -estimates are close to the true  $Q^{\pi}$  values, and unbiased regardless of the estimates if  $\beta (a) = 1 / \mu (a)$ . The variance is low when  $\beta$  is small, therefore, in order to improve the bias-variance tradeoff we recommend using the  $\beta$ -LOO estimate with  $\beta$  defined as:  $\beta (A) = \min \left(c,\frac{1}{\mu(A)}\right)$ , for some constant  $c\geq 1$ . This truncated  $1 / \mu$  coefficient shares similarities with the truncated IS gradient estimate introduced in (Wang et al., 2017) (which we call TISLR for truncated-ISLR):

$$
\hat {G} _ {\mathrm {T I S L R}} = \min  \left(c, \frac {\pi (A)}{\mu (A)}\right) (R (A) - V) \nabla \log \pi (A) + \sum_ {a} \left(\frac {\pi (a)}{\mu (a)} - c\right) _ {+} \mu (a) (Q ^ {\pi} (a) - V) \nabla \log \pi (a).
$$

The differences are: (i) we truncate  $1 / \mu(A) = \pi(A) / \mu(A) \times 1 / \pi(A)$  instead of truncating  $\pi(A) / \mu(A)$ , which provides an additional variance reduction due to the variance of the LR  $\nabla \log \pi(A) = \frac{\nabla \pi(A)}{\pi(A)}$  (since this LR may be large when a low probability action is chosen), and (ii) we use our  $Q$ -baseline instead of a  $V$  baseline, reducing further the variance of the LR estimate.

# 3.2 DISTRIBUTIONAL RETRACE

In off-policy learning it is very difficult to produce an unbiased sample  $R(A)$  of  $Q^{\pi}(A)$  when following another policy  $\mu$ . This would require using full importance sampling correction along the trajectory. Instead, we use the off-policy corrected return computed by the Retrace algorithm, which produces a (biased) estimate of  $Q^{\pi}(A)$  but whose bias vanishes asymptotically (Munos et al., 2016).

In Reactor, we consider predicting an approximation of the return distribution function from any state-action pair  $(x, a)$  in a similar way as in Bellemare et al. (2017). The original algorithm C51 described in that paper considered single-step Bellman updates only. Here we need to extend this idea to multi-step updates and handle the off-policy correction performed by the Retrace algorithm, as defined in (3). Here is a description of those two extensions.

multi-step distributional Bellman operator: First, we extend C51 to multi-step Bellman backups. We consider return-distributions from  $(x,a)$  of the form  $\sum_{i}q_{i}(x,a)\delta_{z_{i}}$  (where  $\delta_z$  denotes a Dirac in  $z$ ) which are supported on a finite uniform grid  $\{z_i\} \in [v_{\mathrm{min}},v_{\mathrm{max}}]$ . The coefficients  $q_{i}(x,a)$  (discrete distribution) corresponds to the probabilities assigned to each atom  $z_{i}$  of the grid. From an observed  $n$ -step sequence  $\{x_t,a_t,r_t,x_{t + 1},\ldots ,x_{t + n}\}$ , generated by behavior policy (i.e.,  $a_{s}\sim \mu (\cdot |x_{s})$  for  $t\leq s < t + n$ ), we build the  $n$ -step backed-up return-distribution from  $(x_{t},a_{t})$  (for any action  $a$  selected at the last state of the sequence):

$$
\sum_ {i} q _ {i} ^ {n, a} (x _ {t}, a _ {t}) \delta_ {z _ {i} ^ {n}}, \mathrm {w i t h} z _ {i} ^ {n} = \sum_ {s = t} ^ {t + n - 1} \gamma^ {s - t} r _ {s} + \gamma^ {n} z _ {i} \mathrm {a n d} q _ {i} ^ {n, a} (x _ {t}, a _ {t}) = q _ {i} ^ {n} (x _ {t + n}, a).
$$

Since this distribution is supported on the set of atoms  $\{z_i^n\}$ , which is not necessarily aligned with the grid  $\{z_i\}$ , we do a projection step and minimize the KL-loss between the projected target and the current estimate, just as with C51 except with a different target distribution (Bellemare et al., 2017).

Distributional Retrace: Now, the Retrace algorithm defined in (3) involves an off-policy correction which is not handled by the previous  $n$ -step distributional Bellman backup. The key to extending this distributional back-up to off-policy learning is to rewrite the Retrace algorithm as a linear combination of  $n$ -step Bellman backups, weighted by some coefficients  $\alpha_{n,a}$ . Indeed, notice that (3) rewrites as

$$
\Delta Q(x_{t},a_{t}) = \sum_{n\geq 1}\sum_{a\in A}\alpha_{n,a}\Big[\underbrace{\sum_{s = t}^{t + n - 1}\gamma^{s - t}r_{s} + \gamma^{n}Q(x_{t + n},a)}_{n\text{-step Bellman backup}}\Big] - Q(x_{t},a_{t}),
$$

where  $\alpha_{n,a} = (c_{t + 1}\dots c_{t + n - 1})\big(\pi (a|x_{t + n}) - \mathbb{I}\{a = a_{t + n}\} c_{t + n}\big)$ . These coefficients depend on the degree of off-policy-ness (between  $\mu$  and  $\pi$ ) along the trajectory. We have that  $\sum_{n\geq 1}\sum_{a}\alpha_{n,a} = \sum_{n\geq 1}(c_{t + 1}\dots c_{t + n - 1})(1 - c_{t + n}) = 1$ , but notice some coefficients may be negative. However, in expectation (over the behavior policy) they are non-negative. Indeed,

$$
\begin{array}{l} \mathbb {E} _ {\mu} [ \alpha_ {n, a} ] = \mathbb {E} \Big [ (c _ {t + 1} \dots c _ {t + n - 1}) \mathbb {E} _ {a _ {t + n} \sim \mu (\cdot | x _ {t + n})} \big [ \pi (a | x _ {t + n}) - \mathbb {I} \{a = a _ {t + n} \} c _ {t + n} | x _ {t + n} \big ] \Big ] \\ { = } { \mathbb { E } \Big [ \big ( c _ { t + 1 } \ldots c _ { t + n - 1 } \big ) \Big ( \pi ( a | x _ { t + n } ) - \mu ( a | x _ { t + n } ) \min \left( 1 , \frac { \pi ( a | x _ { t + n } ) } { \mu ( a | x _ { t + n } ) } \right) \Big ) \Big ] \geq 0 , } \\ \end{array}
$$

by definition of the  $c_s$  coefficients (4). Thus in expectation (over the behavior policy), the Retrace update can be seen as a convex combination of  $n$ -step Bellman updates.

Then, the distributional Retrace algorithm can be defined as backing up a mixture of  $n$ -step distributions. More precisely, we define the Retrace target distribution as:

$$
\sum_ {i = 1} q _ {i} ^ {*} (x _ {t}, a _ {t}) \delta_ {z _ {i}}, \text {w i t h} q _ {i} ^ {*} (x _ {t}, a _ {t}) = \sum_ {n \geq 1} \sum_ {a} \alpha_ {n, a} \sum_ {j} q _ {j} ^ {n} (x _ {t}, a _ {t}) h _ {z _ {i}} (z _ {j} ^ {n}),
$$

and update the current probabilities  $q(x_{t},a_{t})$  by performing a gradient step on the KL-loss

$$
\nabla \mathrm {K L} \left(q ^ {*} \left(x _ {t}, a _ {t}\right), q \left(x _ {t}, a _ {t}\right)\right) = - \sum_ {i} q _ {i} ^ {*} \left(x _ {t}, a _ {t}\right) \nabla \log q _ {i} \left(x _ {t}, a _ {t}\right). \tag {12}
$$

Again, notice that some target "probabilities"  $q_{i}^{*}(x_{t},a_{t})$  may be negative for some sample trajectory, but in expectation they will be non-negative. Since the gradient of a KL-loss is linear w.r.t. its first argument, our update rule (12) provides an unbiased estimate of the gradient of the KL between the expected (over the behavior policy) Retrace target distribution and the current predicted distribution.

Remark: The same method can be applied to other algorithms (such as  $\mathrm{TB}(\lambda)$  (Precup et al., 2000) and importance sampling (Precup et al., 2001)) in order to derive distributional versions of other off-policy multi-step RL algorithms.

# 3.3 PRIORITIZED SEQUENCE REPLAY

Prioritized experience replay has been shown to boost both statistical efficiency and final performance of deep RL agents (Schaul et al., 2016). However, as originally defined prioritized replay does not handle sequences of transitions and weights all unsampled transitions identically. In this section we present an alternative initialization strategy, called lazy initialization, and argue that it better encodes prior information about temporal difference errors. We then briefly describe our computationally efficient prioritized sequence sampling algorithm, with full details left to the appendix.

It is widely recognized that TD errors tend to be temporally correlated, indeed the need to break this temporal correlation has been one of the primary justifications for the use of experience replay (Mnih et al., 2015). Our proposed algorithm begins with this fundamental assumption.

Assumption 1. Temporal differences are temporally correlated, with correlation decaying on average with the time-difference between two transitions.

Prioritized experience replay adds new transitions to the replay buffer with a constant priority, but given the above assumption we can devise a better method. Specifically, we propose to add experience

![](images/ec56eb4b0d85656e6387560fce033276562a4cbc09ea83082bf7d34cbedb7212.jpg)  
Figure 2: (Left) The model of parallelism of DQN, A3C and Reactor architectures. Each row represents a separate thread. In Reactor's case, each worker, consiting of a learner and an actor is run on a separate worker machine. (Right) Comparison of training times and resources for various algorithms.

<table><tr><td>Algorithm</td><td>Training Time</td><td>Type</td><td># Workers</td></tr><tr><td>DQN</td><td>8 days</td><td>GPU</td><td>1</td></tr><tr><td>Double DQN</td><td>8 days</td><td>GPU</td><td>1</td></tr><tr><td>Dueling</td><td>8 days</td><td>GPU</td><td>1</td></tr><tr><td>Prioritized DQN</td><td>8 days</td><td>GPU</td><td>1</td></tr><tr><td>Rainbow</td><td>10 days</td><td>GPU</td><td>1</td></tr><tr><td>A3C</td><td>4 days</td><td>CPU</td><td>16</td></tr><tr><td>Reactor</td><td>&lt; 2 days</td><td>CPU</td><td>10+1</td></tr><tr><td>Reactor 500m</td><td>4 days</td><td>CPU</td><td>10+1</td></tr><tr><td>Reactor*</td><td>&lt; 1 day</td><td>CPU</td><td>20+1</td></tr></table>

to the buffer with no priority, inserting a priority only after the transition has been sampled and used for training. Also, instead of sampling transitions, we assign priorities to all (overlapping) sequences of length 32. When sampling, sequences with an assigned priority are sampled proportionally to that priority. Sequences with no assigned priority are sampled proportionally to the average priority of assigned priority sequences within some local neighbourhood. Averages are weighted to compensate for sampling biases (i.e. more samples are made in areas of hight estimated priorities, and in the absence of weighting this would lead to overestimation of unassigned priorities).

The lazy initialization scheme starts with priorities  $p_t$  corresponding to the sequences  $\{x_t, a_t, \ldots, x_{t+n}\}$  for which a priority was already assigned. Then it extrapolates a priority to all other sequences in the following way. Let us define a partition  $(I_i)_i$  of the states ordered by increasing time such that each cell  $I_i$  contains exactly one state  $s_i$  with already assigned priority. We define the estimated priority  $\hat{p}_t$  to all other sequences as  $\hat{p}_t = \sum_{s_i \in J(t)} \frac{w_i}{\sum_{i' \in J(t)} w_{i'}} p(s_i)$ , where  $J(t)$  is a collection of contiguous cells  $(I_i)$  containing time  $t$ , and  $w_i = |I_i|$  is the length of the cell  $I_i$  containing  $s_i$ . (and we define  $\hat{p}_t = p_t$  for already assigned priorities). Cell sizes work as estimates of inverse local density and are used as importance weights for priority estimation. So far we have defined a class of algorithms all free to choose the partition  $(I_i)$  and the collection of cells  $I(t)$ , as long that they satisfy the above constraints.

Now with probability  $\epsilon$  we sample uniformly at random, and with probability  $1 - \epsilon$  we sample proportionally to  $\hat{p}_t$ . We implemented an algorithm satisfying the above constraints and called it Contextual Priority Tree (CPT). It is based on AVL trees (Velskii & Landis, 1976) and can execute sampling, insertion, deletion and density evaluation in  $O(n \ln(n))$  time. We describe CPT in detail in the Appendix in Section 6.3.

We treated prioritization as purely a variance reduction technique. Importance-sampling weights were evaluated as in prioritized experience replay, with fixed  $\beta = 1$  in (2). We used simple gradient magnitude estimates as priorities: 1) a mean absolute TD error along a sequence for Retrace as defined in (3) in a classical RL case and 2) total variation in the distributional Retrace case.

# 3.4 AGENT ARCHITECTURE

In order to improve CPU utilization we decoupled acting from learning. This is an important aspect of our architecture: an acting thread receives observations, submits actions to the environment, and stores transitions in memory, while a learning thread re-samples sequences of experiences from memory and trains on them (Figure 2, left). We typically execute 4-6 acting steps per each learning step. We sample sequences of length 33 in batches of 4.

We allow the agent to be distributed over multiple machines each containing action-learner pairs. Both the network and target network are stored on a shared parameter server while each machine contains its own local replay memory. Training is done by downloading a shared network, evaluating local gradients and sending them to be applied on the shared network. While the agent can also be trained on a single machine, in this work we present results of training obtained with 10 actor-learner

![](images/8bddeac90bdbb1da3bdd1c85fd6d489e9579be15d72a3e0eb52f43db31dce905.jpg)  
Figure 3: (Left) 'Reactor' includes distributional Retrace algorithm, prioritized replay and beta-LOO policy gradient with  $\beta = 1$ . The other curves show algorithms performance by removing each of the components and changing the number of workers. (Right) Performance of Reactor as a function of real time in hours compared to other state-of-art algorithms. Rainbow learning curve provided by Hessel et al. (2017) by request.

![](images/94ebf2b1acca20b4eacf20c8fc89647d914dc84d5217b5e260f5de885f1194ab.jpg)

workers and one parameter server. In Figure 2 (right) we compare resources and runtimes of Reactor with related algorithms. $^3$

# 3.4.1 NETWORK ARCHITECTURE

In some domains, such as Atari, it is useful to base decisions on a short history of past observations. Two techniques are generally used to achieve this: 1) frame stacking and 2) recurrent network architectures. We chose the latter over the former for reasons of implementation simplicity and computational efficiency. As the Retrace algorithm requires evaluating action-values over contiguous sequences of trajectories, using a recurrent architecture allowed each frame to be processed by the convolutional network only once, as opposed to  $n$  times times if  $n$  frame concatenations were used.

The Reactor architecture uses a recurrent neural network which takes an observation  $x_{t}$  as input and produces two outputs: categorical action-value distributions  $q_{i}(x_{t},a)$  ( $i$  here is a bin identifier), and policy probabilities  $\pi (a|x_{t})$ .

We use an architecture inspired by the duelling network architecture (Wang et al., 2015). We split action-value -distribution logits into state-value logits and advantage logits, which in turn are connected to the same LSTM network (Hochreiter & Schmidhuber, 1997). Final action-value logits are produced by summing state- and action-specific logits. Finally, a softmax layer on top for each action produces the distributions over discounted future returns.

The policy head uses a softmax layer mixed with a fixed uniform distribution over actions, where this mixing ratio is a hyperparameter (Wiering (1999) Section 5.1.3.). Policy and Q-networks have separate LSTMs. Both LSTMs are connected to a shared linear layer which is connected to a shared convolutional neural network (Krizhevsky et al., 2012). The precise network specification is given in Table 3 in the Appendix.

Gradients coming from the policy LSTM are blocked and only gradients originating from the Q-network LSTM are allowed to back-propagate into the convolutional neural network. We block gradients from the policy head for increased stability, as this avoids positive feedback loops between  $\pi$  and  $q_{i}$  caused by shared representations. We used the Adam optimiser (Kingma & Ba, 2014), with a learning rate of  $5\times 10^{-5}$  and zero momentum because asynchronous updates induce implicit momentum (Mitliagkas et al., 2016). Further discussion of hyperparameters and their optimization can be found in Appendix 6.1.

# 4 EXPERIMENTAL RESULTS

Figure 3 compares the performance of Reactor with the original Categorical DQN ((Bellemare et al., 2017)) algorithm and different versions of Reactor each time leaving one of the algorithmic improvements out. Reactor outperforms Categorical DQN by a wide margin. We can also see that each of the algorithmic improvements (Distributional retrace, beta-LOO and prioritized replay) contributed to the final results. While prioritization was arguably the most important component,

Beta-LOO clearly outperformed TISLR algorithm. Although distributional and non-distributional versions performed similarly in terms of median human normalized scores, distributional version of the algorithm generalized better when tested with random human starts (Table 1).

# 4.1 COMPARING TO PRIOR WORK

We evaluated Reactor with target update frequency  $T_{update} = 1000$ ,  $\lambda = 1.0$  and  $\beta$ -LOO with  $\beta = 1$  on 57 Atari games trained on 10 machines in parallel. We averaged scores over 200 episodes using 30 random human starts and noop starts (Tables 4 and 5 in the Appendix). We calculated mean and median human normalised scores across all games. We also ranked all algorithms (including random and human scores) for each game and evaluated mean rank of each algorithm across all 57 Atari games. We also evaluated mean Rank and Elo scores for each algorithm for both human and noop start settings. Please refer to Section 6.2 in the Appendix for more details.

In Table 1, we see that Reactor, with 200 million steps, exceeds the performance of all algorithms across all metrics, except for Rainbow where the story is mixed. However, the difference in time-efficiency is especially apparent when comparing Reactor and Rainbow (see Figure 3, right). Additionally, unlike Rainbow, Reactor does not use Noisy Networks (Fortunato et al., 2017), which was reported to have contributed to the performance gains.

Regarding ACER (Wang et al., 2016), another Retrace-based actor-critic architecture, both classical and distributional versions of Reactor (Figure 3) exceeded the best reported median human normalized score of 1.9 with noop starts achieved in 500 million steps<sup>4</sup>.

<table><tr><td>ALGORITHM</td><td>NORMALIZED SCORES</td><td>MEAN RANK</td><td>ELO</td></tr><tr><td>RANDOM</td><td>0.00</td><td>11.65</td><td>-563</td></tr><tr><td>HUMAN</td><td>1.00</td><td>6.82</td><td>0</td></tr><tr><td>DQN</td><td>0.69</td><td>9.05</td><td>-172</td></tr><tr><td>DDQN</td><td>1.11</td><td>7.63</td><td>-58</td></tr><tr><td>DUEL</td><td>1.17</td><td>6.35</td><td>32</td></tr><tr><td>PRIOR</td><td>1.13</td><td>6.63</td><td>13</td></tr><tr><td>PRIOR. DUEL.</td><td>1.15</td><td>6.25</td><td>40</td></tr><tr><td>A3C LSTM</td><td>1.13</td><td>6.30</td><td>37</td></tr><tr><td>RAINBOW</td><td>1.53</td><td>4.18</td><td>186</td></tr><tr><td>REACTOR ND5</td><td>1.51</td><td>4.98</td><td>126</td></tr><tr><td>REACTOR</td><td>1.65</td><td>4.58</td><td>156</td></tr><tr><td>REACTOR 500M</td><td>1.82</td><td>3.65</td><td>227</td></tr></table>

Table 1: Random human starts  

<table><tr><td>ALGORITHM</td><td>NORMALIZED SCORES</td><td>MEAN RANK</td><td>ELO</td></tr><tr><td>RANDOM</td><td>0.00</td><td>10.93</td><td>-673</td></tr><tr><td>HUMAN</td><td>1.00</td><td>6.89</td><td>0</td></tr><tr><td>DQN</td><td>0.79</td><td>8.65</td><td>-167</td></tr><tr><td>DDQN</td><td>1.18</td><td>7.28</td><td>-27</td></tr><tr><td>DUEL</td><td>1.51</td><td>5.19</td><td>143</td></tr><tr><td>PRIOR</td><td>1.24</td><td>6.11</td><td>70</td></tr><tr><td>PRIOR. DUEL.</td><td>1.72</td><td>5.44</td><td>126</td></tr><tr><td>ACER4 500M</td><td>1.9</td><td>-</td><td>-</td></tr><tr><td>RAINBOW</td><td>2.31</td><td>3.63</td><td>270</td></tr><tr><td>REACTOR ND5</td><td>1.80</td><td>4.53</td><td>195</td></tr><tr><td>REACTOR</td><td>1.87</td><td>4.46</td><td>196</td></tr><tr><td>REACTOR 500M</td><td>2.30</td><td>3.47</td><td>280</td></tr></table>

Table 2: 30 random no-op starts.

Table 1 compares two versions of our algorithm  $^{5}$ . with several other state-of-art algorithms across 57 Atari games for a fixed random seed across all games (Bellemare et al., 2013). The algorithms that we compare Reactor against are: DQN (Mnih et al., 2015), Double DQN (Van Hasselt et al., 2016), DQN with prioritised experience replay (Schaul et al., 2015) andueling architecture and prioritisedueling architecture (Wang et al., 2015). Each algorithm was exposed to 200 million frames of experience (unless stated otherwise) and the same pre-processing pipeline including 4 action repeats was used as in the original DQN paper (Mnih et al., 2015).

# 5 CONCLUSION

In this work we presented a new off-policy agent based on Retrace actor-critic architecture and show that it can achieve similar performance as the current state-of-the-art while giving significant real-time performance gains. We demonstrate the benefits of each of the suggested algorithmic improvements, including Distributional Retrace, beta-LOO policy gradient and contextual priority tree.

# REFERENCES

Oron Anschel, Nir Baram, and Nahum Shimkin. Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning. In International Conference on Machine Learning, pp. 176-185, 2017.  
Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. J. Artif. Intell. Res.(JAIR), 47:253-279, 2013.  
Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. arXiv preprint arXiv:1707.06887, 2017.  
Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, et al. Noisy networks for exploration. arXiv preprint arXiv:1706.10295, 2017.  
Shixiang Gu, Timothy Lillicrap, Zoubin Ghahramani, Richard E Turner, and Sergey Levine. Q-prop: Sample-efficient policy gradient with an off-policy critic. International Conference on Learning Representations, 2017.  
Frank S He, Yang Liu, Alexander G Schwing, and Jian Peng. Learning to play in a day: Faster deep reinforcement learning by optimality tightening. In International Conference on Learning Representations, 2017.  
Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. arXiv preprint arXiv:1710.02298, 2017.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
Long-H Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3/4):69-97, 1992.  
Ioannis Mitliagkas, Ce Zhang, Stefan Hadjis, and Christopher Ré. Asynchrony begets momentum, with an application to deep learning. In Communication, Control, and Computing (Allerton), 2016 54th Annual Allerton Conference on, pp. 997-1004. IEEE, 2016.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, 2016.  
Andrew W Moore and Christopher G Atkeson. Prioritized sweeping: Reinforcement learning with less data and less time. Machine learning, 13(1):103-130, 1993.  
Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems, pp. 1046-1054, 2016.

Brendan O'Donoghue, Remi Munos, Koray Kavukcuoglu, and Volodymyr Mnih. Combining policy gradient and q-learning. International Conference on Learning Representations, 2017.  
Doina Precup, Richard S Sutton, and Satinder Singh. Eligibility traces for off-policy policy evaluation. In Proceedings of the Seventeenth International Conference on Machine Learning, 2000.  
Doina Precup, Richard S Sutton, and Sanjoy Dasgupta. Off-policy temporal-difference learning with function approximation. In Proceedings of the 18th International Conference on Machine Learning, pp. 417-424, 2001.  
Martin Riedmiller. Neural fitted q iteration-first experiences with a data efficient neural reinforcement learning method. In ECML, volume 3720, pp. 317-328. Springer, 2005.  
Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.  
Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. In International Conference on Learning Representations, 2016.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pp. 1889-1897, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. Nature, 550(7676):354-359, 10 2017. URL http://dx.doi.org/10.1038/nature24270.  
Richard S. Sutton, David Mcallester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In *In Advances in Neural Information Processing Systems* 12, pp. 1057-1063. MIT Press, 2000.  
Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In AAAI, pp. 2094-2100, 2016.  
Adel'son G Velskii and E Landis. An algorithm for the organisation of information. Dokl. Akad. Nauk SSSR, 146:263-266, 1976.  
Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, and Koray Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. arXiv preprint arXiv:1703.01161, 2017.  
Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas. *Dueling network architectures for deep reinforcement learning*. *International Conference on Machine Learning*, pp. 1995–2003, 2015.  
Ziyu Wang, Victor Bapat, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224, 2016.  
Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with experience replay. In International Conference on Learning Representations, 2017.  
C. J. C. H. Watkins and P. Dayan. Q-learning. Machine Learning, 8(3):272-292, 1992.

Marco A Wiering. Explorations in efficient reinforcement learning. PhD thesis, University of Amsterdam, 1999.  
Dongbin Zhao, Haitao Wang, Kun Shao, and Yuanheng Zhu. Deep reinforcement learning with experience replay based on sarsa. In Computational Intelligence (SSCI), 2016 IEEE Symposium Series on, pp. 1-6. IEEE, 2016.
