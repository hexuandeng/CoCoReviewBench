# OFF-POLICY AVERAGE REWARD ACTOR-CRITIC WITH DETERMINISTIC POLICY SEARCH

Anonymous authors

Paper under double-blind review

# ABSTRACT

The average reward criterion is relatively less explored as most existing works in the Reinforcement Learning literature consider the discounted reward criterion. There are few recent works that present on-policy average reward actor-critic algorithms, but average reward off-policy actor-critic is relatively less explored. In this paper, we present both on-policy and off-policy deterministic policy gradient theorems for the average reward performance criterion. Using these theorems, we also present an Average Reward Off-Policy Deep Deterministic Policy Gradient (ARDDPG) Algorithm. We show a finite time analysis of the resulting three-timescale stochastic approximation scheme and obtain an  $\epsilon$ -optimal stationary policy with a sample complexity of  $\Omega(\epsilon^{-2.5})$ . We compare the average reward performance of our proposed algorithm and observe better empirical performance compared to state-of-the-art on-policy average reward actor-critic algorithms over MuJoCo based environments.

# 1 INTRODUCTION

The reinforcement learning (RL) paradigm has shown significant promise for finding solutions to decision making problems that rely on a reward-based feedback from the environment. Here one is mostly concerned with the long-term reward acquired by the algorithm. In the case of infinite horizon problems, the discounted reward criterion has largely been studied because of its simplicity. Major recent development in the context of RL in continuous state-action spaces has considered the discounted reward criterion (Schulman et al., 2015; 2017; Lillicrap et al., 2016; Haarnoja et al., 2018). However, there are very few works which focus on the average reward performance criterion in the continuous state-action setting (Zhang & Ross, 2021; Ma et al., 2021).

The average reward criterion has started receiving attention in recent times and there are papers that discuss the benefits of using the average reward criterion over the discounted reward (Dewanto & Gallagher, 2021; Naik et al., 2019). In the case of recurrent Markov Decision Processes (MDPs), average reward happens to be the most selective optimization criterion. Further, optimization in average reward setting is not dependent on the initial state distribution. Moreover, an obvious discrepancy between the objective function and the evaluation metric, that exists for discounted reward setting, is resolved by opting for the average reward criterion.

There are few algorithms that optimize the average reward and all of them happen to be on-policy algorithms (Zhang & Ross, 2021; Ma et al., 2021). It has been demonstrated several times that on-policy algorithms are less sample efficient than off-policy algorithms Lillicrap et al. (2016); Haarnoja et al. (2018); Fujimoto et al. (2018) for the discounted reward criterion. In this paper we try to find whether the same is true for the average reward criterion as well. We try to overcome the research gap in development of off-policy average reward algorithms for continuous state and action spaces by proposing an Average Reward Off-Policy Deep Deterministic Policy Gradient (ARO-DDPG) Algorithm.

The policy evaluation step in the case of the average reward algorithm is equivalent to finding the solution to the Poisson equation (i.e., the Bellman equation for a given policy). Poisson equation, because of its form, does not admit a unique solution but only solutions that are unique up to a constant term. Further, the policy evaluation step in this case consists of finding not just the Differential Q-value function but also the average reward. Thus, because of the required estimation of two quantities instead of one, the role of the optimizing algorithm and the target network increases here.

The following are the broad contributions of our paper:

- We provide both on-policy and off-policy deterministic policy gradient theorems for the average reward performance metric.  
- We present our Average Reward Off-Policy Deep Deterministic Policy Gradient (ARDDPG) algorithm.  
- We perform non-asymptotic convergence analysis and provide a finite time analysis of our three timescale stochastic approximation based actor-critic algorithm using a linear function approximator.  
- We show the results of implementations using our algorithm with other state-of-the-art algorithms in the literature.

The rest of the paper is structured as follows: In Section 2, we present the preliminaries on the MDP framework, the basic setting as well as the policy gradient algorithm. Section 3 presents the deterministic policy gradient theorem and our algorithm. Section 4 then presents the main theoretical results related to the finite time analysis. Section 5 presents the experimental results. In Section 6, we discuss other related work and Section 7 presents the conclusions. The detailed proofs for the finite time analysis are available in the Appendix.

# 2 PRELIMINARIES

Consider a Markov Decision Process (MDP)  $M = \{S, A, R, P, \pi\}$  where  $S \subset \mathbb{R}^n$  is the (continuous) state space,  $A \subset \mathbb{R}^m$  is the (continuous) action space,  $R: S \times A \mapsto \mathbb{R}$  denotes the reward function with  $R(s, a)$  being the reward obtained under state  $s$  and action  $a$ . Further,  $P(\cdot | s, a)$  denotes the state transition function defined as  $P: S \times A \mapsto \mu(\cdot)$ , where  $\mu: \mathcal{B}(S) \mapsto [0, 1]$  is a probability measure. The policy  $\pi$  is defined as  $\pi: S \mapsto A$ . In the above,  $\mathcal{B}(S)$  represents the Borel sigma algebra on  $S$ .

Assumption 1. The Markov process obtained under any policy  $\pi$  is ergodic.

# 2.1 DISCOUNTED REWARD MDPS

In discounted reward MDPs, discounting is controlled by  $\gamma \in (0,1)$ . The following performance metric is optimized with respect to the policy:

$$
\eta (\pi) = \mathbb {E} ^ {\pi} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} R \left(s _ {t}, a _ {t}\right) \right] = \int_ {S} \rho_ {0} (s) V ^ {\pi} (s) d s. \tag {1}
$$

Here,  $\rho_0$  is the initial state distribution and  $V^{\pi}$  is the value function.  $V_{\pi}(s)$  denotes the long term reward acquired when starting in the state  $s$ .

$$
V ^ {\pi} \left(s _ {t}\right) = \mathbb {E} ^ {\pi} \left[ R \left(s _ {t}, a _ {t}\right) + \gamma V ^ {\pi} \left(s _ {t + 1}\right) \mid s _ {t} \right]. \tag {2}
$$

# 2.2 AVERAGE REWARD MDPS

The performance metric in the case of average reward MDPs is the long-run average reward  $\rho (\pi)$  defined as follows:

$$
\rho (\pi) = \lim  _ {N \rightarrow \infty} \frac {1}{N} \mathbb {E} ^ {\pi} [ \sum_ {t = 0} ^ {N - 1} R \left(s _ {t}, a _ {t}\right) ] = \int_ {S} d ^ {\pi} (s) R ^ {\pi} (s) d s, \tag {3}
$$

where  $R^{\pi}(s) \triangleq R(s, \pi(s))$ . The limit in the first equality in equation 3 exists because of Assumption 1. The quantity  $d^{\pi}(s)$  in the second equality in equation 3 corresponds to the steady state probability of the Markov process being in state  $s \in S$  and it exists and is unique given  $\pi$  from Assumption 1 as well.

Lemma 1. There exists a unique constant  $k(= \rho(\pi))$  which satisfies the following equation:

$$
V _ {d i f f} ^ {\pi} \left(s _ {t}\right) = \mathbb {E} ^ {\pi} \left[ R \left(s _ {t}, a _ {t}\right) - k + V _ {d i f f} ^ {\pi} \left(s _ {t + 1}\right) \mid s _ {t} \right] \tag {4}
$$

Proof. See appendix for the proof.

![](images/9e3c3de519553bf8a2f02eaba2f63beb9ac70e1850ada4911f63c9fa814a81eb.jpg)

In (4),  $V_{diff}^{\pi}$  is the differential value function corresponding to the policy  $\pi$  and is defined in (5). Further, the differential Q-value or action-value function  $Q_{diff}^{\pi}$  is defined in (6).

$$
V _ {d i f f} ^ {\pi} \left(s _ {t}\right) = \mathbb {E} ^ {\pi} \left[ \sum_ {k = t} ^ {\infty} R \left(s _ {k}, a _ {k}\right) - \rho (\pi) \mid s _ {t} \right]. \tag {5}
$$

$$
Q _ {d i f f} ^ {\pi} \left(s _ {t}, a _ {t}\right) = \mathbb {E} ^ {\pi} \left[ \sum_ {k = t} ^ {\infty} R \left(s _ {k}, a _ {k}\right) - \rho (\pi) \mid s _ {t}, a _ {t} \right]. \tag {6}
$$

# 2.3 POLICY GRADIENT THEOREM

Unlike in Q-learning where we try to find the optimal Q-value function and then infer the policy from it, the policy gradient theorem (Sutton et al., 1999; Silver et al., 2014; Degris et al., 2012) allows us to directly optimize the performance metric via its gradient with respect to the policy parameters. Q-learning can be visualized to be a value iteration scheme while an algorithm based on the policy gradient theorem can be seen as mimicking policy iteration. Sutton et al. (1999) provided the policy gradient theorem for on-policy optimization of both the discounted reward and the average reward algorithms, see (7)-(8), respectively.

$$
\nabla_ {\theta} \eta (\pi) = \int_ {S} \rho^ {\pi} (s) \int_ {A} \nabla_ {\theta} \pi (a | s, \theta) Q ^ {\pi} (s, a) d a d s. \tag {7}
$$

$$
\nabla_ {\theta} \rho (\pi) = \int_ {S} d ^ {\pi} (s) \int_ {A} \nabla_ {\theta} \pi (a | s, \theta) Q _ {d i f f} ^ {\pi} (s, a) d a d s. \tag {8}
$$

In (7)  $\rho^{\pi}$  denotes the long term discounted state visitation probability density which is defined in equation 9 while  $d^{\pi}(s) = \lim_{t\to \infty}P_{t}^{\pi}(s)$  is the steady state probability density on states.  $P^{\pi}$  denotes the transition probability kernel for the Markov chain induced by policy  $\pi$  and  $P_{t}^{\pi}$  is the state distribution at instant  $t$  given by (9).

$$
\rho^ {\pi} (s) = \frac {1}{\gamma} \sum_ {t = 0} ^ {\infty} \gamma^ {t} P _ {t} ^ {\pi} (s). \tag {9}
$$

$$
P _ {t} ^ {\pi} (s) = \int_ {S \times S \dots} \rho_ {0} \left(s _ {0}\right) \prod_ {k = 0} ^ {t - 1} P ^ {\pi} \left(s _ {k + 1} \mid s _ {k}\right) d s _ {0} \dots d s _ {t - 1}. \tag {10}
$$

The policy gradient theorem in Sutton et al. (1999) is only valid for on-policy algorithms. Degris et al. (2012) came up with an approximate off-policy policy gradient theorem for stochastic policies, see (11), where  $d^{\mu}$  stands for the steady state density function corresponding to the policy  $\mu$ .

$$
\nabla_ {\theta} \eta (\pi) = \int_ {S} d ^ {\mu} (s) \int_ {A} \nabla_ {\theta} \pi (a | s, \theta) Q ^ {\pi} (s, a) d a d s. \tag {11}
$$

Silver et al. (2014) came up with the deterministic policy gradient theorem, see (12), which eventually led to the development of very successful Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2016) algorithm and Twin Delayed DDPG (TD3) algorithm (Fujimoto et al., 2018).

$$
\nabla_ {\theta} \eta (\pi) = \int_ {S} \rho^ {\pi} (s) \nabla_ {a} Q ^ {\pi} (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s. \tag {12}
$$

# 3 PROPOSED AVERAGE REWARD ALGORITHM

We now propose the deterministic policy gradient theorem for the average reward criterion. The policy gradient estimator has to be derived separately for both the on-policy and off-policy settings. Obtaining the on-policy deterministic policy gradient estimator is straightforward but dealing with the off-policy gradient estimates involves an approximate gradient (Degris et al., 2012).

# 3.1 ON-POLICY POLICY GRADIENT THEOREM

We cannot directly use the second equality of (3) to derive the policy gradient theorem because of the inability to take the derivative of steady state density function. Therefore one needs to use (4) to obtain the average reward deterministic policy gradient theorem.

Theorem 1. The gradient of  $\rho (\pi)$  with respect to policy parameter  $\theta$  is given as follows:

$$
\nabla_ {\theta} \rho (\pi) = \int_ {S} d ^ {\pi} (s) \nabla_ {a} Q _ {d i f f} ^ {\pi} (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s. \tag {13}
$$

Proof. See appendix for the proof.

![](images/47f96cf422411bae837d637a47f255580bfbe6d0a951e625901c563640f91dda.jpg)

# 3.2 COMPATIBLE FUNCTION APPROXIMATION

The result in this section is mostly inspired from Silver et al. (2014). Recall that  $Q_{diff}^{\pi}(s,a)$  is the 'true' differential  $Q$ -value of the state-action tuple  $(s,a)$  under the parameterized policy  $\pi$ . Now let  $Q_{diff}^{w}(s,a)$  denote the approximate differential  $Q$ -value of the  $(s,a)$ -tuple when function approximation with parameter  $w$  is used. Lemma 2 says that when the function approximator satisfies a compatibility condition (cf. (14,15)), then the gradient expression in (13,) is also satisfied by  $Q_{diff}^{w}$  in place of  $Q_{diff}^{\pi}$ .

Lemma 2. Assume that the differential  $Q$ -value function (6) satisfies the following:

$$
1. \nabla_ {w} \nabla_ {a} Q _ {\text {d i f f}} ^ {w} (s, a) = \nabla_ {\theta} \pi (s, \theta). \tag {14}
$$

2. Differential  $Q$ -value function parameter  $w = w_{\epsilon}^{*}$  optimizes the following error function:

$$
\zeta (\theta , w) = \frac {1}{2} \int_ {S} d ^ {\pi} (s) \| \nabla_ {a} Q _ {d i f f} ^ {\pi} (s, a) | _ {a = \pi (s)} - \nabla_ {a} Q _ {d i f f} ^ {w} (s, a) | _ {a = \pi (s)} \| ^ {2} d s. \tag {15}
$$

Then,

$$
\int_ {S} d ^ {\pi} (s) \nabla_ {a} Q _ {d i f f} ^ {\pi} (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s = \int_ {S} d ^ {\pi} (s) \nabla_ {a} Q _ {d i f f} ^ {w} (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s. \tag {16}
$$

Further, in the case when a linear function approximator is used, we obtain

$$
\nabla_ {a} Q _ {\text {d i f f}} ^ {w} (s, a) = \nabla_ {\theta} \pi (s, \theta) ^ {\intercal} w. \tag {17}
$$

Proof. See the appendix for a proof.

![](images/190810c0529f1e8c4ffd5b9239d41ca9f3bbe1cc92d32b967b124c6e4eda6067.jpg)

An important implication of lemma 2 also is that the dimension of the matrix on the left hand side and the right hand side of (14) should be the same. Hence the dimensions of the parameters  $\theta$  (used in the parameterized policy) and  $w$  (used to approximate the differential Q-value function) are the same. Lemma 2 shows that the compatible function approximation theorem has the same form in the average reward setting as the discounted reward setting.

# 3.3 OFF-POLICY POLICY GRADIENT THEOREM

In order to derive off-policy policy gradient theorem it is not possible to use the direction adopted by Degris et al. (2012) for off-policy stochastic policy gradient theorem for the discounted reward setting. We first mention our proposed approximate off-policy deterministic policy gradient theorem and then explain why some alternatives would not have worked.

Assumption 2. For the Markov chain obtained from the policy  $\pi$ , let  $K(\cdot|\cdot)$  be the transition kernel and  $S^{\pi}$  the steady state measure. Then there exists  $a > 0$  and  $\kappa \in (0,1)$  such that

$$
D _ {T V} (K ^ {t} (\cdot | s), S ^ {\pi} (\cdot)) \leq a \kappa^ {t}, \forall t, \forall s \in S.
$$

Theorem 2. The approximate gradient of the average reward  $\rho (\pi)$  with respect to the policy parameter  $\theta$  is given by the following expression:

$$
\nabla_ {\theta} \rho (\pi) \approx \int_ {S} d ^ {\mu} (s) \nabla_ {a} Q _ {d i f f} ^ {\pi} (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s = \nabla_ {\theta} \bar {\rho} (\pi). \tag {18}
$$

Further, the approximation error is  $\mathcal{E}(\pi ,\mu) = \| \nabla_{\theta}\rho (\pi) - \nabla_{\theta}\bar{\rho} (\pi)\|$ , where  $\mu$  represents the behaviour policy.  $\mathcal{E}$  satisfies

$$
\mathcal {E} (\pi , \mu) \leq Z \| \theta^ {\pi} - \theta^ {\mu} \|. \tag {19}
$$

Here,  $Z = 2^{m + 1}C(\lceil \log_{\kappa}a^{-1}\rceil +1 / \kappa)L_t$  with  $L_{t}$  being the Lipschitz constant for the transition probability density function.

Proof. See the appendix for a proof.

![](images/f25eae6c687d25950c80f89c487f61836bfbf1c410b0dcd31b79e6356c7eee41.jpg)

Theorem 2 suggests that the approximation error in the gradient increases as the difference between the target policy  $\pi$  and the behaviour policy  $\mu$  increases.

# 3.4 OFF-POLICY ALTERNATIVES

In this section we will talk about what alternatives could be thought of in place of what is suggested in section 3.3 and why those alternatives would not work.

1. One can possibly take inspiration from Degris et al. (2012) and define an objective function as in (20), which is a naive off-policy version of (3).

$$
\rho (\pi) = \int_ {S} d ^ {\mu} (s) R ^ {\pi} (s) d s. \tag {20}
$$

If, however, we take the derivative of  $\rho (\pi)$  defined above, we get the policy update rule as in (21).

$$
\nabla_ {\theta} \rho (\pi) = \int_ {S} d ^ {\mu} (s) \nabla_ {a} R (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s. \tag {21}
$$

The issue with update rule (21) is that it only considers the reward function and not the transition dynamics of the MDP. If we look at (13), the derivative of the objective function includes the differential Q-value function which encapsulates both the information of the reward function and the transition dynamics of the MDP and hence is valid.

2. A lot of work in the off-policy setting relies on importance sampling ratios. Recently a few works devised a method to estimate the steady state probability density ratio of the target and behavior policies (Zhang et al., 2020a;b; Liu et al., 2018; Nachum et al., 2019). The ratio of steady state densities could be used for deterministic policy optimization but there are certain issues which prohibit its usage, see (22).

$$
\nabla_ {\theta} \rho (\pi) = \int_ {S} d ^ {\mu} (s) \tau (s) \nabla_ {a} Q _ {d i f f} ^ {\pi} (s, a) | _ {a = \pi (s)} \nabla_ {\theta} \pi (s, \theta) d s. \tag {22}
$$

Here,  $\tau(s)$  is the steady state probability density ratio defined as  $d^{\pi}(s) / d^{\mu}(s)$ . In order to calculate  $\tau(s)$  we need information about  $(\pi(a|s), \mu(a|s)$  and  $P(s'|s, a))$ . We need the ratio  $\pi(a|s) / \mu(a|s)$  and for deterministic policies the ratio would be  $\delta(a - \pi(s) / \delta(a - \mu(s))$ , where  $\delta(\cdot)$  is the Dirac-Delta function:

$$
\frac {\delta (a - \pi (s))}{\delta (a - \mu (s))} = \left\{ \begin{array}{l l} 0 & \text {i f} a = \mu (s) \\ \infty & \text {i f} a = \pi (s) \\ \frac {0}{0} & \text {o t h e r w i s e .} \end{array} \right. \tag {23}
$$

From (23), it is clear that the ratio  $\delta(a - \pi(s) / \delta(a - \mu(s))$  will be undefined for almost all actions  $a \in A$ . Thus, we cannot use this ratio for deterministic policies. Otherwise, we need  $P(s'|s, \pi(a))$  and  $P(s'|s, \mu(a))$ . It is possible to get the information about  $P(s'|s, \mu(a))$  by sampling from the Markov process generated by the policy  $\mu$  but obtaining this information about  $P(s'|s, \pi(a))$  is impossible as in the off-policy setting data from  $\pi$  is assumed to be simply unavailable.

# 3.5 ACTOR-CRITIC UPDATE RULE

The critic and average reward parameters are estimated using the TD(0) update rule but use target estimators. In the following, (24)-(27) present the critic and average reward parameter updates. The actor is updated using (13) and the practical update is presented in (28)-(29). The target parameters are updated using Polyak averaging in (30)-(32). Let  $\{s_i, a_i, s_i'\}_{i=0}^{n-1}$  denote the batch of sampled data from the replay buffer.

$$
\begin{array}{r} \xi_ {t} ^ {j} = \frac {1}{2} \sum_ {i = 0} ^ {n - 1} \left(R \left(s _ {i}, a _ {i}\right) - \overline {{\rho_ {t}}} - Q _ {d i f f} ^ {w _ {i}} \left(s _ {i}, a _ {i}\right) + \min  \left(\bar {Q} _ {d i f f} ^ {w _ {1}}, \bar {Q} _ {d i f f} ^ {w _ {1}}\right) \left(s _ {i} ^ {\prime}, \pi \left(s _ {i} ^ {\prime}, \bar {\theta} _ {t}\right)\right)\right) ^ {2} \\ j \in \{1, 2 \} \end{array} \tag {24}
$$

$$
\xi_ {t} ^ {3} = \frac {1}{2} \sum_ {i = 0} ^ {n - 1} \left(R \left(s _ {i}, a _ {i}\right) - \rho_ {t} - \min  \left(\bar {Q} _ {\text {d i f f}} ^ {w _ {1}} \left(s _ {i}, a _ {i}\right), \bar {Q} _ {\text {d i f f}} ^ {w _ {2}} \left(s _ {i}, a _ {i}\right)\right) + \right. \tag {25}
$$

$$
\left. \min  \left(\overline {{Q}} _ {d i f f} ^ {w _ {1}}, \overline {{Q}} _ {d i f f} ^ {w _ {1}}\right) \left(s _ {i} ^ {\prime}, \pi \left(s _ {i} ^ {\prime}, \overline {{\theta_ {t}}}\right)\right)\right) ^ {2}
$$

$$
w _ {t + 1} ^ {i} = w _ {t} ^ {i} - \alpha_ {t} \nabla_ {w _ {i}} \xi_ {t} ^ {i} \quad i \in \{1, 2 \} \tag {26}
$$

$$
\rho_ {t + 1} = \rho_ {t} - \alpha_ {t} \nabla_ {p} \xi_ {t} ^ {3} \tag {27}
$$

$$
\nu_ {i} = \nabla_ {a} \min  \left(Q _ {\text {d i f f}} ^ {w _ {1}}, Q _ {\text {d i f f}} ^ {w _ {2}}\right) \left(s _ {i}, a\right) | _ {a = \pi \left(s _ {i}\right)} \nabla_ {\theta} \pi \left(s _ {i}, \theta_ {t}\right) \tag {28}
$$

$$
\theta_ {t + 1} = \theta_ {t} + \gamma_ {t} \left(\sum_ {i = 0} ^ {n - 1} \nu_ {i}\right) \tag {29}
$$

$$
\overline {{w _ {t + 1} ^ {i}}} = \overline {{w}} _ {t} ^ {i} + \beta_ {t} \left(w _ {t + 1} ^ {i} - \overline {{w}} _ {t} ^ {i}\right) \quad i \in \{1, 2 \} \tag {30}
$$

$$
\overline {{\rho_ {t + 1}}} = \overline {{\rho_ {t}}} + \beta_ {t} \left(\rho_ {t + 1} - \overline {{\rho_ {t}}}\right) \tag {31}
$$

$$
\overline {{\theta_ {t + 1}}} = \overline {{\theta_ {t}}} + \beta_ {t} \left(\theta_ {t + 1} - \overline {{\theta_ {t}}}\right) \tag {32}
$$

# 4 FINITE TIME ANALYSIS

In this section we present the finite time analysis of the on-policy and off-policy average reward actor critic algorithm with linear function approximators. First we mention the assumptions taken to perform the finite time analysis followed by the main results.

Assumption 3.  $\phi^{\pi}(s)$  denotes the feature vector of state  $s$  and satisfies  $\| \phi^{\pi}(s)\| \leq 1$ .

Assumption 4. The reward function is uniformly bounded, viz.,  $|R^{\pi}(s)| \leq C_r < \infty$ .

Algorithm 1 ARO-DDPG Practical Algorithm  
```txt
Initialize actor parameter  $\theta$  and critic parameters  $w_{1}, w_{2}$ . Initialize actor target parameter  $\theta \rightarrow \overline{\theta}$ . Initialize critic target parameters  $w_{1} \rightarrow \overline{w_{1}}, w_{2} \rightarrow \overline{w_{2}}$ . Initialize average reward parameter  $\rho$ . Initialize target average reward parameter  $\rho \rightarrow \overline{\rho}$ . Initialize Replay buffer = {}
```

1:  $t = 0$ ,  $s_0 = \text{env resett}$   
2: while  $t \leq$  total steps do  
3:  $a_t = \pi(s_t) + \epsilon$  {  $\epsilon$  denotes the noise}  
4:  $s_{t+1} \sim P(\cdot | s_t, a_t)$  and  $r_t = R(s_t, a_t)$   
5: Store  $\{s_t, a_t, s_{t+1}\}$  in the Replay Buffer  
6: if  $t \%$  eval_freq == 0 then  
7: Evaluate(agent)  
8: end if  
9: if  $t \%$  critic_update_freq == 0 then  
10: Update critic according to (24) - (27)  
11: end if  
12: if  $t \%$  actor_update_freq == 0 then  
13: Update actor according to (28) - (29)  
14: Update target estimators according to (30) - (32)  
15: end if  
16: if  $s_{t+1}$  is terminal then  
17:  $s_t = \text{env resett}$   
18: else  
19:  $s_t = s_{t+1}$   
20: end if  
21: end while

Assumption 5.  $Q_{diff}(s, a)$  is Lipschitz continuous w.r.t to  $a$ . Thus,  $\| Q_{diff}(s, a_1) - Q_{diff}(s, a_2) \| \leq L_a \| a_1 - a_2 \|$ .

Assumption 6. Parameterised policy  $\pi(s, \theta)$  is Lipschitz continuous w.r.t  $\theta$ . Thus,  $\| \pi(s, \theta_1) - \pi(s, \theta_2) \| \leq L_{\pi} \| \theta_1 - \theta_2 \|$ .

Assumption 7. The state feature mapping  $(\phi_{\pi}(s))$  defined for a policy  $\pi$  with parameter  $\theta$  is Lipschitz continuous w.r.t  $\theta$ . Thus,  $\max_s\| \phi^{\pi_1}(s) - \phi^{\pi_2}(s)\| \leq L_\phi \| \theta_1 - \theta_2\|$ .

# 4.1 ON-POLICY ANALYSIS

In this section we present the theorem for finite time analysis of the on-policy version of the algorithm with linear function approximator and target estimator for the critic and average reward.

Theorem 3. The on-policy average reward actor critic algorithm (Algorithm 2) obtains an  $\epsilon$ -accurate optimal point with sample complexity of  $\Omega(\epsilon^{-2.5})$ . We obtain

$$
\begin{array}{l} \min  _ {0 \leq t \leq T - 1} E \| \nabla_ {\theta} \rho (\theta_ {t}) \| ^ {2} = \mathcal {O} \left(\frac {1}{T ^ {0 . 4}}\right) + \mathcal {O} (1), \\ \leq \epsilon + \mathcal {O} (1). \\ \end{array}
$$

Proof. We present a proof outline here. See the appendix for complete details of the proof.

1. Obtain a bound on  $\frac{1}{T}\sum_{t=0}^{T-1} E\|\nabla_{\theta}\rho(\theta_t)\|^2$  using  $\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\|\Delta w_t\|^2$  (Lemma 3).  
2. Obtain a bound on  $\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}||\Delta w_t||^2$  using  $\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}||\Delta \rho_t||^2$  (Lemma 4).  
3. Obtain a bound on  $\frac{1}{T}\sum_{t = 0}^{T - 1}\mathbb{E}||\Delta \rho_t||^2$  (Lemma 5).  
4. The claim follows upon combining the results from the above steps.

![](images/fbe50258ac51946f6c94ec73fe1e12ae272a9f1b2030b88752dcc53d7366951a.jpg)

# 4.2 OFF-POLICY ANALYSIS

In this section we present the theorem for finite time analysis of off-policy version of the algorithm with linear function approximator and target estimator for the critic and average reward.

Theorem 4. The off-policy average reward actor critic algorithm (Algorithm 3) with behavior policy  $\mu$  obtains an  $\epsilon$ -accurate optimal point with sample complexity of  $\Omega(\epsilon^{-2.5})$ . We obtain

$$
\begin{array}{l} \min  _ {0 \leq t \leq T - 1} E \| \nabla_ {\theta} \bar {\rho} (\theta_ {t}) \| ^ {2} = \mathcal {O} \left(\frac {1}{T ^ {0 . 4}}\right) + \mathcal {O} (1) + \mathcal {O} \left(N _ {\theta} ^ {2}\right) \\ \leq \epsilon + \mathcal {O} (1) + \mathcal {O} \left(N _ {\theta} ^ {2}\right) \\ w h e r e N _ {\theta} := \max  _ {t} \| \theta_ {\mu} - \theta_ {t} \|. \\ \end{array}
$$

Proof. See the appendix for a proof.

![](images/808ee4db9638b82a4415c54f692091dda3100c9f96eff1a8e6f28f2cccc84293.jpg)

The error bound in the off-policy algorithm is the same as the on-policy algorithm except for a single term  $\mathcal{O}(N_{\theta}^{2})$ . The extra term denotes the error induced because of not using the samples from the current policy for performing updates.

# 5 EXPERIMENTAL RESULTS

We conducted experiments on six different environments using the DeepMind control suite (Tassa et al., 2018) and found the performance of ARO-DDPG to be superior than the other algorithms. All the environments selected are infinite horizon tasks. Maximum reward per time step is 1. None of the tasks have a goal reaching nature. We performed all the experiments using 10 different seeds. We show here performance comparisons with two state-of-the-art algorithms: the Average Reward TRPO (ATRPO) (Zhang & Ross, 2021) and the Average Policy Optimization (APO) (Ma et al., 2021) respectively. In general for the average reward performance, not many algorithms are available in the literature. We implemented the ATRPO algorithm using the instructions available in the original paper. We used the original hyper-parameters suggested by the author for ATRPO.

![](images/8426e9e480a789cd3209747610ec9e09625a393783a9d5feb5a49a701e5f4501.jpg)

![](images/ee62bd0159fbcbc791f7edf6fee673455fea6921890001de58c6a8cba4642f6f.jpg)

![](images/9a7a834d1e1e64d73a64f8df488f449fe6fa31b1015ca64283f05aeee7634464.jpg)

![](images/ab0779ec887fe7e61410469b148ffe8f8bfbd8f484ad0a336a78c96d67851425.jpg)

![](images/bf30478ca166e3c84f98f80a3b467072468de9ee5c7896135dcb134eba333b63.jpg)  
Environment steps (in millions)

![](images/4792297b68cdd4326e6afbe161e45098de977bc6d1c16fb220d47d021a35653c.jpg)  
Figure 1: Comparison of performance of different average reward algorithms

For our proposed algorithm we trained the agent for 1 million time steps and evaluated the agent after every 5,000 time steps in the concerned environment. The length of each episode for the training

phase was taken to be 1,000 and for the evaluation phase it was taken to be 10,000. The reason for taking longer episode length for evaluation phase was to compare the long term average reward performance of the algorithms. We also tried using episode length of 10,000 for training phase and found that to be giving poor average reward performance. We do not reset the agent if it lands in a state before completing 10,000 steps from where it is unable to escape of its own, while continuing to give a penalty for the remaining length of the episode. That way the cost of failure is very high. While training we updated the actor after performing a fixed number of environment steps. We updated the critic neural network with more frequency as compared to the actor neural network. We used target actor and critic networks along with target estimator of the average reward parameter for stability while using bootstrapping updates. We also borrowed the double Q-network trick from Fujimoto et al. (2018). Complete information regarding the set of hyper-parameters used is provided in the appendix.

# 6 RELATED WORK

Actor-Critic algorithms for average reward performance criterion is much less studied compared to discounted reward performance criterion. One of the earliest works on the average reward criterion is Mahadevan (1996). In this paper, Mahadevan compares the performance of R-learning with that of Q-learning and concludes that fine tuning is required to get better results from R-learning. R-learning is the average reward version of Q-learning. Later in 1999, Sutton et al. came up with the policy gradient theorem for both discounted and average reward criteria (Sutton et al., 1999), which formed the bedrock for development of the average reward actor-critic algorithms. The first proof of asymptotic convergence of average reward actor-critic algorithms with function approximation appeared in Konda & Tsitsiklis (2003). In 2007, Bhatnagar et al. came up with incremental natural policy gradient algorithms for the average reward setting and provided the asymptotic convergence proof of these.

Recently, Wan et al. presented a Differential Q-learning algorithm and claimed that their algorithm is able to find the exact differential value function without an offset. Further, Wan et al. came up with an extension of the options framework from the discounted setting to the average reward setting and demonstrated the performance of the algorithm in the Four-Room domain task. One of the major contributions in off-policy policy evaluation is made by Zhang et al. (2021a). Here Zhang et al. came up with a convergent off-policy evaluation scheme inspired from the gradient temporal difference learning algorithms but involving a primal-dual formulation making the policy evaluation step feasible for a neural network implementation. Zhang et al. (2021b) provided another convergent off-policy evaluation algorithm using target network and  $l_{2}$ -regularisation. In our work we use the same policy evaluation update.

Our work in this paper is actually an extension of the work of Silver et al. (2014) from the discounted to the average reward setting. In Xiong et al. (2022), a finite time analysis for deterministic policy gradient algorithm was done for the discounted reward setting. We performed the finite time analysis for the average reward deterministic policy gradient algorithm and in particular obtain the same sample complexity for our algorithm as reported by Wu et al. (2020) for stochastic policies.

# 7 CONCLUSION AND FUTURE WORK

In this paper we presented a deterministic policy gradient theorem for both on-policy and off-policy settings. We then proposed the Average Reward Off-policy Deep Deterministic Policy Gradient(ARO-DDPG) algorithm using neural network and replay buffer for high dimensional MuJoCo based environments. We observed superior performance of ARO-DDPG over existing average reward algorithms (ATRPO and PPO). At the end we provided a finite time analysis for the on-policy and off-policy algorithms obtained from the proposed policy gradient theorem and obtained a sample complexity of  $\Omega (\epsilon^{-2.5})$ . Lastly to extend the current line of work, one could try using natural gradient descent based update rule for deterministic policy. Further in the current work we tried optimizing the average reward performance (gain optimality). In the literature, optimizing the differential value function for all the states is mentioned as part of achieving Blackwell optimality. Hence actor-critic algorithms could be designed that not only optimize average reward performance but also differential value function (bias optimality).

# REFERENCES

Shalabh Bhatnagar, Mohammad Ghavamzadeh, Mark Lee, and Richard S Sutton. Incremental natural actor-critic algorithms. Advances in neural information processing systems, 20, 2007.  
Thomas Degris, Martha White, and Richard S. Sutton. Linear off-policy actor-critic. In Proceedings of the 29th International Conference on Machine Learning, ICML 2012, Edinburgh, Scotland, UK, June 26 - July 1, 2012. icml.cc / Omnipress, 2012. URL http://icml.cc/2012/papers/268.pdf.  
Vektor Dewanto and Marcus Gallagher. Examining average and discounted reward optimality criteria in reinforcement learning. CoRR, abs/2107.01348, 2021. URL https://arxiv.org/abs/2107.01348.  
Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In Jennifer G. Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 1582-1591. PMLR, 2018. URL http://proceedings.mlr.press/v80/fujimoto18a.html.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Jennifer G. Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 1856-1865. PMLR, 2018. URL http://proceedings.mlr.press/v80/haarnoja18b.html.  
Vijay R Konda and John N Tsitsiklis. Onactor-critic algorithms. SIAM journal on Control and Optimization, 42(4):1143-1166, 2003.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In Yoshua Bengio and Yann LeCun (eds.), 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016. URL http://arxiv.org/abs/1509.02971.  
Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. Breaking the curse of horizon: Infinite-horizon off-policy estimation. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pp. 5361-5371, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/dda04f9d634145a9c68d5dfe53b21272-Abstract.html.  
Xiaoteng Ma, Xiaohang Tang, Li Xia, Jun Yang, and Qianchuan Zhao. Average-reward reinforcement learning with trust region methods. In Zhi-Hua Zhou (ed.), Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pp. 2797-2803. ijcai.org, 2021. doi: 10.24963/ijcai.2021/385. URL https://doi.org/10.24963/ijcai.2021/385.  
Sridhar Mahadevan. Average reward reinforcement learning: Foundations, algorithms, and empirical results. Mach. Learn., 22(1-3):159-195, 1996. doi: 10.1023/A:1018064306595. URL https://doi.org/10.1023/A:1018064306595.  
A Yu Mitrophanov. Sensitivity and convergence of uniformly ergodic markov chains. Journal of Applied Probability, 42(4):1003-1014, 2005.  
Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. Dualiice: Behavior-agnostic estimation of discounted stationary distribution corrections. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 2315-2325, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/cf9a242b70f45317ffd281241fa66502-Abstract.html.

Abhishek Naik, Roshan Shariff, Niko Yasui, and Richard S. Sutton. Discounted reinforcement learning is not an optimization problem. CoRR, abs/1910.02140, 2019. URL http://arxiv.org/abs/1910.02140.  
John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, and Pieter Abbeel. Trust region policy optimization. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, ICML'15, pp. 1889-1897. JMLR.org, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.  
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin A. Ried-miller. Deterministic policy gradient algorithms. In Proceedings of the 31th International Conference on Machine Learning, ICML 2014, Beijing, China, 21-26 June 2014, volume 32 of JMLR Workshop and Conference Proceedings, pp. 387-395. JMLR.org, 2014. URL http://proceedings.mlr.press/v32/silver14.html.  
Richard S. Sutton, David A. McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Sara A. Solla, Todd K. Leen, and Klaus-Robert Müller (eds.), Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999], pp. 1057-1063. The MIT Press, 1999. URL http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.  
Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdelmaleki, Josh Merel, Andrew Lefrancq, Timothy P. Lillicrap, and Martin A. Riedmiller. Deepmind control suite. CoRR, abs/1801.00690, 2018. URL http://arxiv.org/abs/1801.00690.  
Yi Wan, Abhishek Naik, and Rich Sutton. Average-reward learning and planning with options. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 22758-22769. Curran Associates, Inc., 2021a. URL https://proceedings.neurips.cc/paper/2021/file/c058f544c737782deacefa532d9add4c-Paper.pdf.  
Yi Wan, Abhishek Naik, and Richard S Sutton. Learning and planning in average-reward markov decision processes. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 10653-10662. PMLR, 18-24 Jul 2021b. URL https://proceedings.mlr.press/v139/wan21a.html.  
Yue Frank Wu, Weitong ZHANG, Pan Xu, and Quanquan Gu. A finite-time analysis of two timescale actor-critic methods. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 17617-17628. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/cc9b3c69b56df284846bf2432f1cba90-Paper.pdf.  
Huaqing Xiong, Tengyu Xu, Lin Zhao, Yingbin Liang, and Wei Zhang. Deterministic policy gradient: Convergence analysis. In The 38th Conference on Uncertainty in Artificial Intelligence, 2022.  
Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. Gendice: Generalized offline estimation of stationary values. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020a. URL https://openreview.net/forum?id=Hkx1cnVFwB.  
Shangtong Zhang, Bo Liu, and Shimon Whiteson. Gradientdice: Rethinking generalized offline estimation of stationary values. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 11194-11203. PMLR, 2020b. URL http://proceedings.mlr.press/v119/zhang20r.html.

Shangtong Zhang, Yi Wan, Richard S Sutton, and Shimon Whiteson. Average-reward off-policy policy evaluation with function approximation. In International Conference on Machine Learning, pp. 12578-12588. PMLR, 2021a.

Shangtong Zhang, Hengshuai Yao, and Shimon Whiteson. Breaking the deadly triad with a target network. In International Conference on Machine Learning, pp. 12621-12631. PMLR, 2021b.

Yiming Zhang and Keith W. Ross. On-policy deep reinforcement learning for the average-reward criterion. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 12535-12545. PMLR, 2021. URL http://proceedings.mlr.press/v139/zhang21q.html.
