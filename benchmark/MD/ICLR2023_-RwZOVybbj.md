# RISK-AWARE REINFORCEMENT LEARNING WITH COHERENT RISK MEASURES AND NON-LINEAR FUNCTION APPROXIMATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the risk-aware reinforcement learning (RL) problem in the episodic finite-horizon Markov decision process with unknown transition and reward functions. In contrast to the risk-neutral RL problem, we consider minimizing the risk of having low rewards, which arise due to the intrinsic randomness of the MDPs and imperfect knowledge of the model. Our work provides a unified framework to analyze the regret of risk-aware RL policy with coherent risk measures in conjunction with non-linear function approximation, which gives the first sub-linear regret bounds in the setting. Finally, we validate our theoretical results via empirical experiments on synthetic and real-world data.

# 1 INTRODUCTION

Reinforcement learning (RL) (Sutton & Barto, 2018) is a control-theoretic problem in which an agent interacts with an unknown environment and aims to maximize its expected total reward. Due to the intrinsic randomness of the environment, even a policy with high expected total rewards may occasionally produce very low rewards. This uncertainty is problematic in many real-life applications like competitive games (Mnih et al., 2013) and healthcare (Liu et al., 2020), where the agent (or decision-maker) needs to be risk-averse. For example, the drug responses to patients are stochastic due to the patients' varying physiology or genetic profiles (McMahon & Insel, 2012), therefore, it is desirable to select a set of treatments that yield high effectiveness and minimize the possibility of adverse effects (Beutler et al., 2016; Fatemi et al., 2021). The existing RL policies that maximize the risk-neutral total reward can not lead to an optimal risk-aware RL policy for problems where the total reward has uncertainty (Yu et al., 2018). Therefore, our goal is to design an RL algorithm that learns a risk-aware RL policy to minimize the risk of having a small expected total reward.

Then, how should we learn a risk-aware RL policy? A natural approach is to directly learn a risk-aware RL policy that minimizes the risk of having a small expected total reward (Howard & Matheson, 1972). For quantifying such a risk, one can use risk measures like entropic risk (Föllmer & Knispel, 2011), value-at-risk (VaR) (Dempster, 2002), conditional value-at-risk (CVaR) (Rockafellar et al., 2000), or entropic value-at-risk (EVaR) (Ahmadi-Javid, 2012). These risk measures capture the total reward volatility and quantify the possibility of rare but catastrophic events. The entropic risk measure can be viewed as a mean-variance criterion, where the risk is expressed as the variance of total reward (Fei et al., 2021). Alternatively, VaR, CVaR, and EVaR use quantile criteria, which are often preferable for better risk management over the mean-variance criterion (Chapter 3 of Kisiala (2015)). Among these risk measures, coherent risk measures such as CVaR and EVaR are preferred as they enjoy compelling theoretical properties such as coherence (Rockafellar et al., 2000).

The risk-aware RL algorithms with CVaR as a risk measure (Bäuerle & Ott, 2011; Yu et al., 2018; Rigter et al., 2021) exist in the literature. However, apart from being customized only for CVaR, these algorithms suffer two significant shortcomings. First, most of them focus on the tabular MDP setting and need multiple complete traversals of the state space (Bäuerle & Ott, 2011; Rigter et al., 2021). These traversals are prohibitively expensive for problems with large state space and not possible in problems with continuous state space, thus limiting these algorithms' applicability in practice. Second, the existing algorithms that consider continuous or infinite state space assume that MDP is known, i.e., the probability transitions and the reward of each state are known a priori to the algorithm.

In such settings, the agent does not need to explore or generalize to unseen scenarios. Therefore, the problem considered in Yu et al. (2018) is a planning problem rather than a learning problem. This paper alleviates both shortcomings by proposing a new risk-aware RL algorithm where MDPs are unknown and uses non-linear function approximation for addressing continuous state space.

Recent works (Jin et al., 2020; Yang et al., 2020) have proposed RL algorithms with function approximation and finite-sample regret guarantees, but they only focus on the risk-neutral RL setting. Extending their results to a risk-aware RL setting is non-trivial due to two major challenges. First, the existing analyses heavily rely on the linearity of the expectation in the risk-neutral Bellman equation. This linearity property does not hold in the risk-aware RL setting when a coherent risk measure replaces the expectation in the Bellman equation. Then, how can we address this challenge? We overcome this challenge by the non-trivial application of the super-additivity property<sup>1</sup> of coherent risk measures (see Lemma 6 and its application in Appendix 1).

The risk-neutral RL algorithms only need one sample of the next state to construct an unbiased estimate of the Bellman update (Yang et al., 2020) as one can unbiasedly estimate the expectation in the risk-neutral Bellman equation with a single sample. However, this does not hold in the risk-aware RL setting. Furthermore, whether one can construct an unbiased estimate of an arbitrary risk measure using only one sample is unknown. This problem leads to the second major challenge: how can we construct an unbiased estimate of the risk-aware Bellman update? To resolve this challenge, we assume access to a weak simulator² that can sample different next states given the current state and action and use these samples to construct an unbiased estimator. Such an assumption is mild and holds in many real-world applications, e.g., a player can anticipate the opponent's next moves and hence the possible next states of the game. After resolving both challenges, we propose an algorithm that uses a risk-aware value iteration procedure based on the upper confidence bound (UCB) and has a finite-sample sub-linear regret upper bound. Specifically, our contributions are as follows:

- We first formalize the risk-aware RL setting with coherent risk measures, namely the risk-aware objective function and the risk-aware Bellman equation in Section 3. We then introduce the notion of regret for a risk-aware RL policy.  
- We propose a general risk-aware RL algorithm named Risk-Aware Upper Confidence Bound (RA-UCB) for an entire class of coherent risk measures in Section 4. RA-UCB uses UCB-based value functions with non-linear function approximation and also enjoys a finite-sample sub-linear regret upper bound guarantee.  
- We provide a unified framework to analyze regret for any coherent risk measure in Section 4.1. The novelty in our analysis is in the decomposition of risk-aware RL policy's regret by the super-additivity property of coherent risk measures (shown in the proof of Lemma 1 in Appendix E.1).  
- Our empirical experiments on synthetic and real datasets validate the different performance aspects of our proposed algorithm in Section 5.

# 2 COHERENT RISK MEASURES

Let  $Z \in L^{1}(\Omega, \mathcal{F}, \mathbb{P})^{3}$  be a real-valued random variable with a finite mean and the cumulative distribution function  $F_{Z}(z) = \mathbb{P}(Z \leq z)$ . A function  $\rho: L^{1}(\Omega, \mathcal{F}, \mathbb{P}) \to \mathbb{R} \cup \{+\infty\}$  is a coherent risk measure if it satisfies the following properties:

1. Normalized:  $\rho (0) = 0$  
2. Monotonic: If  $Z, Z' \in L^{1}(\Omega, \mathcal{F}, \mathbb{P})$  and  $\mathbb{P}(Z \leq Z') = 1$ , then  $\rho(Z) \leq \rho(Z')$ .  
3. Super-additive: If  $Z, Z' \in L^{1}(\Omega, \mathcal{F}, \mathbb{P})$ , then  $\rho(Z + Z') \geq \rho(Z) + \rho(Z')$ .  
4. Positively homogeneous: If  $Z \in L^{1}(\Omega, \mathcal{F}, \mathbb{P})$  and  $\alpha \geq 0$ , then  $\rho(\alpha Z) = \alpha \rho(Z)$ .  
5. Translation invariant: If  $Z \in L^{1}(\Omega, \mathcal{F}, \mathbb{P})$  and  $A$  is a constant variable with value  $a$ , then  $\rho(Z + A) = \rho(Z) + a$ .

Since our reward maximization setting contrasts with the cost minimization setting often considered in the literature, we aim to maximize the risk applied to the random reward, i.e., maximizing  $\rho(Z)$ . Consequently, the properties of risk measure are upended compared to those usually presented in cost minimization setting (Föllmer & Schied, 2010). For example, super-additivity in the reward maximization setting becomes sub-additivity in the cost minimization setting.

Empirical estimation of the risk. The risk of a random variable  $\rho(Z)$  is completely determined by the distribution of  $Z(F_Z)$ . In practice, we do not know the distribution  $F_Z$ ; instead, we can observe  $m$  independent and identically distributed (IID) samples  $\{Z_i\}_{i=1}^m$  from the distribution  $F_Z$ . Then we can use these samples to get an empirical estimator of  $\rho(Z)$ , which is denoted by  $\hat{\rho}(Z_1, \ldots, Z_m)$ .

# 3 PROBLEM SETTING

We consider an episodic finite-horizon Markov decision process (MDP), denoted by a tuple  $\mathcal{M} = (\mathcal{S},\mathcal{A},H,\mathbb{P},r)$ , where  $\mathcal{S}$  and  $\mathcal{A}$  are sets of possible states and actions, respectively,  $H\in \mathbb{Z}_{+}$  is the episode length,  $\mathbb{P} = \{\mathbb{P}_h\}_{h\in [H]}$  are the state transition probability measures, and  $r = \{r_h:\mathcal{S}\times \mathcal{A}\to [0,1]\}_{h\in [H]}:$  are the deterministic reward functions. We assume  $\mathcal{S}$  is a measurable space of possibly infinite cardinality, and  $\mathcal{A}$  is a finite set. For each  $h\in [H]$ ,  $\mathbb{P}_h(\cdot |x,a)$  denotes the probability transition kernel when the agent takes action  $a$  at state  $x$  in time step  $h$ .

An agent interacts with the MDP as follows. There are  $T$  episodes. In the  $t$ -th episode, the agent begins at state  $x_{1}^{t}$  chosen arbitrarily by the environment. In each step  $h \in [H]$ , the agent observes a state  $x_{h}^{t} \in S$ , selects an action  $a_{h}^{t} \in \mathcal{A}$ , and receives a reward  $r_{h}(x_{h}^{t}, a_{h}^{t})$ . The MDP then transitions to the next state following the probability transition kernel  $x_{h+1}^{t} \sim \mathbb{P}_{h}(\cdot | x_{h}^{t}, a_{h}^{t})$ . The episode terminates when the agent reaches state  $x_{H+1}$  at time step  $H + 1$ . In the last time step, the agent takes no action and receives no reward.

A policy  $\pi$  of an agent is a sequence of  $H$  functions, i.e.,  $\pi = \{\pi_h\}_{h\in [H]}$ , in which each  $\pi_h(\cdot |x)$  is a probability distribution over  $\mathcal{A}$ . Here,  $\pi_h(a|x)$  indicates the probability that the agent takes action  $a$  at state  $x$  in time step  $h$ . Any policy  $\pi$  and an initial state  $x_{1}$  determine a probability measure  $P_{x_1}^{\pi}$  and an associated stochastic process  $\{(x_h,a_h),h\in [H]\}$ . Let  $\mathbb{E}_{x_1}^{\pi}[\cdot ]$  denote the expectation operator with respect to  $P_{x_1}^{\pi}$ . The standard risk-neutral MDP objective is

$$
\max  _ {\pi} \mathbb {E} _ {x _ {1}} ^ {\pi} \left[ \sum_ {h = 1} ^ {H} r _ {h} \left(x _ {h}, a _ {h}\right) \right]. \tag {1}
$$

# 3.1 RISK-AWARE EPISODIC MDP

The risk-neutral objective defined in Eq. (1) does not account for the risk incurred due to the stochasticity in the state transitions and the agent's policy. Markov risk measures (Ruszczyński, 2010) are proposed to model and analyze such risks. The risk-aware MDP objective is defined as

$$
\max  _ {\pi} J ^ {\pi} \left(x _ {1}\right), \quad \text {w h e r e} \quad J ^ {\pi} \left(x _ {1}\right) := r _ {1} \left(x _ {1}, a _ {1}\right) + \rho \left(r _ {2} \left(x _ {2}, a _ {2}\right) + \rho \left(r _ {3} \left(x _ {3}, a _ {3}\right) + \dots\right)\right), \tag {2}
$$

where  $\rho$  is a coherent one-step conditional risk measure (Ruszczyński, 2010, Definition 6), and  $\{x_{1},a_{1},x_{2},a_{2},\ldots \}$  is a trajectory of states and actions from the MDP under policy  $\pi$ . Here,  $J^{\pi}$  is defined as a nested and multi-stage composition of  $\rho$ , rather than through a single-stage risk measure on the cumulative reward  $\rho \big(\sum_{h = 1}^{H}r_{h}(x_{h},a_{h})\big)$ . More discussions about this are in Appendix C.

# 3.2 BELLMAN EQUATION AND REGRET

The risk-aware Bellman equation is developed for the risk-aware objective defined in Eq. (2) (Ruszczyński, 2010). More specifically, let define the risk-aware state- and action-value functions with respect to the Markov risk measure  $\rho$  as

$$
\begin{array}{l} V _ {h} ^ {\pi} (x) = r _ {h} \left(x, \pi_ {h} (x)\right) + \rho \left(r _ {h + 1} \left(x _ {h + 1}, \pi_ {h + 1} \left(x _ {h + 1}\right)\right) + \rho \left(r _ {h + 2} \left(x _ {h + 2}, \pi_ {h + 2} \left(x _ {h + 2}\right)\right) + \dots\right)\right), \\ Q _ {h} ^ {\pi} (x, a) = r _ {h} (x, a) + \rho \Big (r _ {h + 1} (x _ {h + 1}, \pi_ {h + 1} (x _ {h + 1})) + \rho \big (r _ {h + 2} (x _ {h + 2}, \pi_ {h + 2} (x _ {h + 2})) + \dots \big) \Big). \\ \end{array}
$$

We also define the optimal policy  $\pi^{\star}$  to be the policy that yields the optimal value function  $V_{h}^{\star}(x) = \sup_{\pi} V_{h}^{\pi}(x)$ . The advantage of the formulation given in Eq. (2) is that one can show that the optimal policy exists, and it is Markovian (Theorem 4 of Ruszczyński (2010)). For notations convenience, for any measurable function  $V: \mathcal{S} \to [0, H]$ , we define the operator  $D_{h}^{\rho}$  as

$$
\left(D _ {h} ^ {\rho} V\right) \left(x, a\right) := \rho \left(V \left(x ^ {\prime}\right)\right), \tag {3}
$$

where the risk measure  $\rho$  is taken over the random variable  $x' \sim \mathbb{P}_h(\cdot | x, a)$ . Then, the risk-aware Bellman equation associated with a policy  $\pi$  takes the form

$$
Q _ {h} ^ {\pi} (x, a) = (r _ {h} + D _ {h} ^ {\rho} V _ {h + 1} ^ {\pi}) (x, a), \quad V _ {h} ^ {\pi} (x) = \langle Q _ {h} ^ {\pi} (x, \cdot), \pi_ {h} (\cdot | x) \rangle_ {\mathcal {A}}, \quad V _ {H + 1} ^ {\pi} (x) = 0,
$$

where  $\langle \cdot, \cdot \rangle_{\mathcal{A}}$  denote the inner product over  $\mathcal{A}$  and  $(f + g)(x) = f(x) + g(x)$  for function  $f$  and  $g$ . Similarly, the Bellman optimality equation is given by

$$
Q _ {h} ^ {\star} (x, a) = \left(r _ {h} + D _ {h} ^ {\rho} V _ {h + 1} ^ {\star}\right) (x, a), \quad V _ {h} ^ {\star} (x) = \max  _ {a \in \mathcal {A}} Q _ {h} ^ {\star} (x, a), \quad V _ {H + 1} ^ {\star} (x) = 0. \tag {4}
$$

The above equation implies that the optimal policy  $\pi^{\star}$  is the greedy policy with respect to the optimal action-value function  $\{Q_h^\star\}_{h\in [H]}$ .

In the episodic MDP setting, the agent interacts with the environment through  $T$  episodes to learn the optimal policy. At the beginning of episode  $t$ , the agent selects a policy  $\pi^t$ , and the environment chooses an initial state  $x_1^t$ . The difference in values between  $V_1^{\pi^t}(x_1^t)$  and  $V^{\star}(x_1^t)$  quantifies the sub-optimality of  $\pi^t$ , which serves as the regret of the agent at episode  $t$ . The total regret after  $T$  episodes is defined as

$$
\Re_ {T} (\rho) = \sum_ {t = 1} ^ {T} \left[ V _ {1} ^ {\star} \left(x _ {1} ^ {t}\right) - V _ {1} ^ {\pi_ {t}} \left(x _ {1} ^ {t}\right) \right]. \tag {5}
$$

Here, the policy's regret depends on the risk measure  $\rho$  via the optimal policy  $\pi^{\star}$ . A good policy should have sub-linear regret, i.e.,  $\lim_{T\to \infty}\Re_T / T = 0$ , which implies that the policy will eventually learn to select the best risk-averse actions.

Remark 1. Given two risk measures  $\rho_{1}$  and  $\rho_{2}$  with  $\Re_T(\rho_1) < \Re_T(\rho_2)$ , does not imply  $\rho_{1}$  is a better choice of risk measure for the given problem. Because the optimal policies for  $\rho_{1}$  and  $\rho_{2}$  can be different, their regrets are not directly comparable. Therefore, we can not use regret as a mechanism to compare or select the risk measure.

# 3.3 WEAK SIMULATOR ASSUMPTION

One key challenge for the risk-aware RL policy is that the empirical estimation of risk is more complex than the estimation of expectation in risk-neutral RL (Yu et al., 2018). In this paper, we assume the existence of a weak simulator that we can use to draw samples from the probability transition kernel  $P_{h}(\cdot |x,a)$  for any  $h\in [H],x\in S,a\in \mathcal{A}$ . This assumption is much weaker than the archetypal simulator assumptions often seen in the RL literature, as they also allow to query reward of a given state and action  $r_h(x,a)$ . To the best of our knowledge, all existing works in risk-aware RL with coherent risk measures require some assumptions on the transition probabilities to facilitate the risk estimation procedure. Among these assumptions, our weak simulator assumption is the weakest.

# 3.4 ESTIMATING NON-LINEAR FUNCTIONS

We use reproducing kernel Hilbert space (RKHS) as the class of non-linear functions to represent the optimal action-value function  $Q_h^*$ . For notational convenience, let us denote  $z = (x, a)$  and  $\mathcal{Z} = \mathcal{S} \times \mathcal{A}$ . Following the standard setting, we assume that  $\mathcal{Z}$  is a compact subset of  $\mathbb{R}^d$  for fixed dimension  $d$ . Let  $\mathcal{H}$  denote the RKHS defined on  $\mathcal{Z}$  with the kernel function  $k: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ . Let  $\langle \cdot, \cdot \rangle_{\mathcal{H}}$  and  $\| \cdot \|_{\mathcal{H}}$  be the inner product and the RKHS norm on  $\mathcal{H}$ , respectively. Since  $\mathcal{H}$  is a RKHS, there exists a feature map  $\phi: \mathcal{Z} \to \mathcal{H}$  such that  $\phi(z) = k(z, \cdot)$  and  $f(z) = \langle \phi(z), f \rangle_{\mathcal{H}}$  for all  $f \in \mathcal{H}$  and for all  $z \in \mathcal{Z}$ , this is known as the reproducing kernel property.

# 4 RISK-AWARE RL ALGORITHM WITH COHERENT RISK MEASURES

We now introduce our algorithm named Risk-Aware Upper Confidence Bound (RA-UCB), which is built upon the celebrated Value Iteration Algorithm (Sutton & Barto, 2018). RA-UCB first estimates the value function using kernel least-square regression. Then, it computes an optimistic bonus that gets added to the estimated value function to encourage exploration. Finally, it executes the greedy policy with respect to the estimated value function in the next episode.

RA-UCB Risk-Aware Upper Confidence Bound  
Input: Hyperparameters of coherent risk measure  $\rho$  (e.g., confidence level  $\alpha \in (0,1)$  for CVaR)  
2: for episode  $t = 1,2,\ldots,T$  do  
3: Receive the initial state  $x_1^t$  and initialize  $V_{H + 1}^t$  as the zero function  
4: for step  $h = H,\dots,1$  do  
5: Compute  $\mu_h^t$  and  $\sigma_h^t$  using Eq. (8). Then compute  $Q_h^t$  and  $V_h^t$  using Eq. (9)  
6: end for  
7: for step  $h = 1,\dots,H$  do  
8: Take action  $a_h^t\gets \underset {a\in \mathcal{A}}{\arg \max}Q_h^t (x_h^t,a)$ . Observe reward  $r_h(x_h^t,a_h^t)$  and the next state  $x_{h + 1}^t$   
9: end for  
10: end for

Recall that we defined  $z = (x,a)$  and  $\mathcal{Z} = \mathcal{S}\times \mathcal{A}$  in Section 3.4. We define the following Gram matrix  $K_h^t\in \mathbb{R}^{(t - 1)\times (t - 1)}$  and a function  $k_h^t:\mathcal{Z}\to \mathbb{R}^{t - 1}$  associated with the RKHS  $\mathcal{H}$  as

$$
K _ {h} ^ {t} = \left[ k \left(z _ {h} ^ {\tau}, z _ {h} ^ {\tau^ {\prime}}\right) \right] _ {\tau , \tau^ {\prime} \in [ t - 1 ]}, \quad k _ {h} ^ {t} (z) = \left[ k \left(z _ {h} ^ {1}, z\right), \dots , k \left(z _ {h} ^ {t - 1}, z\right) \right] ^ {\top}. \tag {6}
$$

Given the observed histories and the weak simulator, we define the response vector  $y_h^t \in \mathbb{R}^{t-1}$  as

$$
\left. \left[ y _ {h} ^ {t} \right] = \left[ r _ {h} \left(x _ {h} ^ {\tau}, a _ {h} ^ {\tau}\right) + \hat {\rho} \left(\left\{V _ {h + 1} ^ {t} \left(x _ {(i)} ^ {\prime}\right) \right\} _ {i = 1} ^ {m}\right) \right] _ {\tau = [ t - 1 ]}, \right. \tag {7}
$$

where  $\{x_{(i)}^{\prime}\}_{i = 1}^{m}$  are  $m$  next states drawn from the weak simulator  $P_{h}(\cdot |x_{h}^{\tau},a_{h}^{\tau})$ . This step contains one of the key differences between RA-UCB and its risk-neutral counterpart, with the presence of the empirical risk estimator in the definition of the response vector  $y_{h}^{t}$ . With the newly introduced notations, we define two functions  $\mu_t:\mathcal{Z}\to \mathbb{R}$  and  $\sigma_t:\mathcal{Z}\rightarrow \mathbb{R}$  as

$$
\mu_ {h} ^ {t} (z) = k _ {h} ^ {t} (z) ^ {\top} \big (K _ {h} ^ {t} + \lambda \cdot I \big) ^ {- 1} y _ {h} ^ {t}, \sigma_ {h} ^ {t} (z) = \lambda^ {- 1 / 2} \cdot \big [ k (z, z) - k _ {h} ^ {t} (z) ^ {\top} \big (K _ {h} ^ {t} + \lambda I \big) ^ {- 1} k _ {h} ^ {t} (z) \big ] ^ {1 / 2}. (8)
$$

The terms  $\mu_h^t$  and  $\sigma_h^t$  have several important connections with other literature. More specifically, it resembles the posterior mean and variance of a Gaussian process regression problem (Rasmussen, 2003), with  $y_h^t$  as its target. The second term  $\sigma_h^t$  also reduces to the UCB term used in linear bandits when the feature map  $\phi$  is finite-dimensional (Lattimore & Szepesvári, 2020). We then define our estimate of the value functions  $Q_h^t$  and  $V_h^t$  as follows:

$$
Q _ {h} ^ {t} (x, a) := \min  \left\{\mu_ {h} ^ {t} (x, a) + \beta \cdot \sigma_ {h} ^ {t} (x, a), H - h + 1 \right\}, \quad V _ {h} ^ {t} (x) := \max  _ {a \in \mathcal {A}} Q _ {h} ^ {t} (x, a), \tag {9}
$$

where  $\beta > 0$  is an exploration versus exploitation trade-off parameter.

# 4.1 MAIN THEORETICAL RESULTS

This section presents our main theoretical result, i.e., the regret upper bound guarantee of RA-UCB. We first outline the key assumption that enables the efficient approximation of the value function.

Assumption 1. Let  $R > 0$  be a fixed constant,  $\mathcal{H}$  be the RKHS, and  $\mathcal{B}(r) = \{f\in \mathcal{H}:\| f\|_{\mathcal{H}}\leq r\}$  to be the RKHS-norm ball with radius  $r$ . We assume that for any  $h\in [H]$  and any  $Q:S\times \mathcal{A}\to [0,H]$ , we have  $\mathbb{T}_h^* Q\in \mathcal{B}(RH)$ , where  $\mathbb{T}_h^*$  is the Bellman optimality operator defined in Eq. (4).

This assumption postulates that the risk-aware Bellman optimality operator maps any bounded action-value function to a function in an RKHS  $\mathcal{H}$  with a bounded norm. This assumption ensures that for all  $h\in [H]$ , the optimal action-value function  $Q_{h}^{\star}$  lies inside  $\mathcal{B}(RH)$ . Consequently, there

is no approximation error when using functions from  $\mathcal{H}$  to approximate  $Q_h^\star$ . It can be viewed as equivalent to the realizability assumption in supervised learning. Similar assumptions are made in Jin et al. (2020); Yang et al. (2020); Zanette et al. (2020). Please refer to Du et al. (2019) for a discussion on the necessity of this assumption.

Given this assumption, it is clear that the complexity of  $\mathcal{H}$  plays a central role in the regret bound of RA-UCB. Following the seminal work of Srinivas et al. (2009), we characterize the intrinsic complexity of  $\mathcal{H}$  with the notion of maximum information gain defined as

$$
\Gamma_ {k} (T, \lambda) = 1 / 2 \sup  _ {\mathcal {D} \subseteq \mathcal {Z}, | \mathcal {D} | \leq T} \left\{\log \det  \left(I + K _ {\mathcal {D}} / \lambda\right) \right\}, \tag {10}
$$

where  $k$  is the kernel function,  $\lambda > 0$  is a parameter, and  $K_{\mathcal{D}}$  is the Gram matrix. The maximum information gain depends on how fast the eigenvalues of  $\mathcal{H}$  decay to zero, and can be viewed as a proxy for the dimension of  $\mathcal{H}$  when  $\mathcal{H}$  is infinite-dimensional. Note that  $\Gamma_k(T, \lambda)$  is a problem-dependent quantity that depends on the kernel  $k$ , state space  $S$ , and action space  $\mathcal{A}$ . Furthermore, let us first define the action-value function classes  $\mathcal{Q}_{\mathrm{ucb}}(h, R, B)$  as

$$
\begin{array}{l} \mathcal {Q} _ {\mathrm {u c b}} (h, R, B) = \{Q: \\ Q (z) = \min  \left\{f (z) + \beta \cdot \lambda^ {- 1 / 2} \left[ k (z, z) - k _ {\mathcal {D}} (z) ^ {\top} \left(K _ {\mathcal {D}} + \lambda I\right) ^ {- 1} k _ {\mathcal {D}} (z) \right] ^ {1 / 2}, H - h + 1 \right\} ^ {+}, \\ f \in \mathcal {H}, \| f \| _ {\mathcal {H}} \leq R, \beta \in [ 0, B ], | \mathcal {D} | \leq T \}. \tag {11} \\ \end{array}
$$

With the appropriate choice of  $R$  and  $B$ , the set  $\mathcal{Q}_{\mathrm{ucb}}(h, R, B)$  contains every possible  $Q_h^t$  that can be constructed by RA-UCB. Therefore, the function class  $\mathcal{Q}_{\mathrm{ucb}}$  resembles the concept of hypothesis space in supervised learning. And as we will see, the complexity of  $\mathcal{Q}_{\mathrm{ucb}}$ , in particular, the covering number of  $\mathcal{Q}_{\mathrm{ucb}}$ , plays a crucial role in the regret bound of RA-UCB.

Theorem 1. Let  $\lambda = 1 + 1 / T$ ,  $\beta = B_{T}$  in RA-UCB, and let  $\Gamma_k(T,\lambda)$  be the maximal information gain defined in Eq. (10). Define a constant  $B_{T} > 0$  that satisfies  $B_{T} = \Theta \big(H(\sqrt{\Gamma_{k}(T,\lambda)} + \max_{h\in H}\sqrt{\log N_{\infty}(\epsilon,h,B_{T})})\big)$ . Suppose that the empirical risk estimate  $\hat{\rho}$  achieves the rate of  $\Xi(m,\delta)$ , i.e.,  $\mathbb{P}\big[|\rho(Z) - \hat{\rho}(\{Z_i\}_{i=1}^m)| \leq \Xi(m,\delta)\big] \geq 1 - \delta$ . Then, under Assumption 1, with a probability of at least  $1 - (T^{2}H^{2})^{-1}$ , the regret of RA-UCB is

$$
\mathfrak {R} _ {T} \leq 5 B _ {T} H \sqrt {T \Gamma_ {k} (T , \lambda)} + 2 T H \cdot \Xi (m, (8 T ^ {3} H ^ {3}) ^ {- 1}).
$$

The regret upper bound consists of two terms. The first term resembles risk-neutral regret bound (Yang et al., 2020, Theorem 4.2). Interestingly, our bound distinguishes itself from the risk-neutral setting with the presence of the second term, which quantifies how fast one can estimate the risk from observed samples. It originates from the risk-aware Bellman optimality equation, in which the one-step update requires knowledge of the risk-to-go starting from the next state (see Eq. (4) for more detail). This risk-to-go quantity is approximated by its empirical counterpart, and the discrepancies give rise to the second term in regret. Due to the weak simulator assumption, we have good control over the second term. In the following result, we derive the sufficient number of samples needed to achieve the order-optimal regret for the CVaR, which is one of the most commonly used coherent risk measures. More details about it are given in Appendix D.

Corollary 1. Let  $\rho$  be the CVaR measure defined in Eq. (15) and  $\hat{\rho}$  be the CVaR estimator defined in Eq. (16). Then, RA-UCB achieves the regret of  $\Re_T = O\big(B_T H \sqrt{T \Gamma_k(T, \lambda)}\big)$  with

$O\Big(TH\cdot \log \big(T^{5}H^{6} / B_{T}^{2}\Gamma_{k}(T,\lambda)\big)\Big)$  samples from the weak simulator.

The detailed proof of Corollary 1 is in Appendix E.4. As an example, for the commonly used squared exponential kernel, we get  $B_{T} = O\big(H \cdot \sqrt{\log(TH)} \cdot (\log T)^{d}\big)$  (Yang et al., 2020, Corollary 4) and  $\Gamma_{k}(T,\lambda) = O\big((\log T)^{d + 1}\big)$  (Srinivas et al., 2009), and thus RA-UCB incurs a regret of  $\Re_{T} = \tilde{O}\big(H^{2}\sqrt{T} (\log T)^{1.5d + 1}\big)$ . This result leads to the first sub-linear regret upper bound of the risk-aware RL policy with coherent risk measures.

# 4.2 REGRET ANALYSIS OF THEOREM 1

We first define a few notations to simplify the presentation of the proof. First, we define the temporal-difference (TD) error as

$$
\delta_ {h} ^ {t} (x, a) = \left(r _ {h} + D _ {h} ^ {\rho} V _ {h + 1} ^ {t}\right) (x, a) - Q _ {h} ^ {t} (x, a), \quad \forall (x, a) \in \mathcal {S} \times \mathcal {A}. \tag {12}
$$

For a trajectory  $\{(x_h^t, a_h^t)\}_{h \in [H]}$ , we further define the two following quantities

$$
\begin{array}{l} \zeta_ {t, h} ^ {1} = \left[ V _ {h} ^ {t} \left(x _ {h} ^ {t}\right) - V _ {h} ^ {\pi_ {t}} \left(x _ {h} ^ {t}\right) \right] - \left[ Q _ {h} ^ {t} \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) - Q _ {h} ^ {\pi_ {t}} \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) \right], \\ \zeta_ {t, h} ^ {2} = \left[ \left(D _ {h} ^ {\rho} V _ {h + 1} ^ {t}\right) \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) - \left(D _ {h} ^ {\rho} V _ {h + 1} ^ {\pi_ {t}}\right) \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) \right] - \left[ V _ {h + 1} ^ {t} \left(x _ {h + 1} ^ {t}\right) - V _ {h + 1} ^ {\pi_ {t}} \left(x _ {h + 1} ^ {t}\right) \right]. \tag {13} \\ \end{array}
$$

The random variables  $\zeta_{t,h}^{1}$  and  $\zeta_{t,h}^{2}$  capture the deviations of the value function due to two sources of randomness in the MDP - the randomness of choosing the action  $a_h^t\sim \pi_h^t (\cdot |x_h^t)$  and drawing next state  $x_{h + 1}^t\sim \mathbb{P}_h(\cdot |x_h^t,a_h^t)$ . We establish the upper bound in the following steps.

# Step 1: Decomposition of the regret.

Lemma 1. We can upper bound the regret as

$$
\mathfrak {R} (T) \leq \underbrace {- \sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} \left(\prod_ {i = 1} ^ {h - 1} \mathbb {J} _ {\pi_ {i} ^ {\star}} D _ {i} ^ {\rho}\right) \mathbb {J} _ {\pi_ {h} ^ {\star}} (- \delta_ {h} ^ {t}) \left(x _ {1} ^ {t}\right) - \sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} \delta_ {h} ^ {t} \left(x _ {h} ^ {t} , a _ {h} ^ {t}\right)} _ {\text {T e r m I}} + \underbrace {\sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} \left(\xi_ {t , h} ^ {1} + \xi_ {t , h} ^ {2}\right)} _ {\text {T e r m I I}},
$$

where  $\delta_{h}^{t},\zeta_{t,h}^{1}$  , and  $\zeta_{t,h}^{2}$  are defined above.

Proof sketch. We decompose the instantaneous regret at the  $t$ -th episode into

$$
V _ {1} ^ {\star} \left(x _ {1} ^ {t}\right) - V _ {1} ^ {\pi^ {t}} \left(x _ {1} ^ {t}\right) = \left[ V _ {1} ^ {\star} \left(x _ {1} ^ {t}\right) - V _ {1} ^ {t} \left(x _ {1} ^ {t}\right)\right) ] + \left[ V _ {1} ^ {t} \left(x _ {1} ^ {t}\right) - V _ {1} ^ {\pi^ {t}} \left(x _ {1} ^ {t}\right) \right].
$$

To upper bound the first term, we establish an inequality of the form  $V_{h}^{\star} - V_{h}^{t} \leq f(V_{h + 1}^{\star} - V_{h + 1}^{t})$  for some function  $f$ , and apply it recursively. This inequality is established using the Bellman equation, together with the super-additivity property of CVaR. Similar techniques can be applied to the upper bound of the second term. The detailed proof is given in Appendix E.1

# Step 2. Upper bounding Term I.

Lemma 2. Let  $\lambda = 1 + 1 / T$  and  $\beta = B_T$  in Algorithm RA-UCB. Then under Assumption 1, with probability at least  $1 - (2T^2 H^2)^{-1}$ , we have that for all  $t\in [T],h\in [H],x\in S$ , and  $a\in \mathcal{A}$ :

$$
- 2 \beta b _ {h} ^ {t} (x, a) \leq \delta_ {h} ^ {t} (x, a) \leq 0.
$$

The proof of Lemma 2 is in Appendix E.2. By Lemma Lemma 2,  $\delta_h^t$  is a negative function, and thus we could upper bound the first term in (I) by 0. We obtain that, with a probability of at least  $1 - (2T^2 H^2)^{-1}$ ,

$$
\text {T e r m} \mathrm {I} \leq - \sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} \delta_ {h} ^ {t} \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) \leq 2 \beta \sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} b _ {h} ^ {t} \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right),
$$

which is an upper bound of the sum of the bonus terms. Recall that we can rewrite the bonus term as

$$
b _ {h} ^ {t} \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) = \left[ \phi \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) ^ {\top} \left(\Lambda_ {h} ^ {t}\right) ^ {- 1} \phi \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) \right] ^ {1 / 2},
$$

where  $\Lambda_h^t = \sum_{\tau=1}^{t-1} \phi(x_h^\tau, a_h^\tau) \phi(x_h^\tau, a_h^\tau)^\top + \lambda \cdot I_{\mathcal{H}}$  and  $I_{\mathcal{H}}$  is the identity operator on  $\mathcal{H}$ . Then,

$$
\begin{array}{l} \text {T e r m} \mathrm {I} \leq 2 \beta \cdot \sqrt {T} \sum_ {h = 1} ^ {H} \left[ \sum_ {t = 1} ^ {T} \phi \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) ^ {\top} \left(\Lambda_ {h} ^ {t}\right) ^ {- 1} \phi \left(x _ {h} ^ {t}, a _ {h} ^ {t}\right) \right] ^ {1 / 2} \\ \leq 2 \beta \cdot \sqrt {T} \sum_ {h = 1} ^ {H} [ 2 \log \det  (I + K _ {h} ^ {T} / \lambda) ] ^ {1 / 2} \\ = 4 \beta H \cdot \sqrt {T \cdot \Gamma_ {k} (T , \lambda)}, \\ \end{array}
$$

where  $\Gamma_k(T,\lambda)$  is the maximal information gain defined in Eq. (10).

# Step 3. Upper bounding Term II.

Lemma 3. For  $\zeta_{t,h}^{1}$  and  $\zeta_{t,h}^{2}$  defined in Eq. (13). We have that, with probability at least  $1 - \delta$

$$
\sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} \left(\zeta_ {t, h} ^ {1} + \zeta_ {t, h} ^ {2}\right) \leq \sqrt {1 6 T H ^ {3} \log (2 / \delta)} + 2 T H \cdot \Xi (m, \delta / (4 T H)).
$$

Proof sketch. We show that  $\{\zeta_{t,h}^1\}_{(t,h)\in [T]\times [H]}$  is a bounded martingale difference sequence, and apply Azuma-Hoeffding concentration inequality. For  $\{\zeta_{t,h}^2\}_{(t,h)\in [T]\times [H]}$ , we use concentration inequality of the risk estimator. The complete proof is in Appendix E.3

Setting  $\delta = (2T^{2}H^{2})^{-1}$  gives us

$$
\mathrm {T e r m} \mathrm {I I} \leq \sqrt {1 6 T H ^ {3} \log (4 T ^ {2} H ^ {2})} + 2 T H \cdot \Xi (m, (8 T ^ {3} H ^ {3}) ^ {- 1})
$$

Therefore, combining these above results, with probability at least  $1 - (T^2 H^2)^{-1}$ , the regret is bounded by

$$
\begin{array}{l} \Re (T) \leq 4 \beta H \sqrt {T \Gamma_ {k} (T , \lambda)} + \sqrt {1 6 T H ^ {3} \log (4 T ^ {2} H ^ {2})} + 2 T H \cdot \Xi (m, (8 T ^ {3} H ^ {3}) ^ {- 1}) \\ \leq 5 \beta H \sqrt {T \Gamma_ {k} (T , \lambda)} + 2 T H \cdot \Xi (m, (8 T ^ {3} H ^ {3}) ^ {- 1}). \\ \end{array}
$$

Substituting  $\beta = B_{T}$  completes the proof of Theorem 1.

# 5 EXPERIMENTS

In this section, we empirically demonstrate the effectiveness of RA-UCB. We run different experiments on synthetic and real-world data with the CVaR as a risk measure, which is a commonly used coherent risk measure. We analyze the influence of the risk aversion parameter  $\alpha$  (or confidence level for CVaR) on the total reward as well as the behavior of the output policies. The code for these experiments is available in the supplementary material.

# 5.1 SYNTHETIC EXPERIMENT: ROBOT NAVIGATION

The robot navigation environment is a continuous version of the cliff walking problem considered in example 6.6 of Sutton & Barto (2018), visualized in Fig. 1. In this synthetic experiment, a robot must navigate inside a room full of obstacles to reach its goal destination. The robot navigates by choosing from 4 actions {up, down, left, right}. Since the floor is slippery, the direction of movement is perturbed by  $r \cdot \phi$ , where  $\phi \sim U(-\pi, \pi)$  and  $r \in [0,1]$  represent the angle and magnitude of the perturbation. The robot receives a positive reward of 10 for reaching the destination and a negative reward for being close to obstacles. The negative reward increases exponentially as the robot comes close to the obstacle. We set the horizon of each episode to  $H = 30$ . The robot does not know perturbation parameters ( $r = 0.3$ ) and the obstacles' positions, so it has to learn them online via interacting with the environment. We use the RBF kernel

![](images/06044f795884c8b9ce5e18b1f59a84677942f10aff5c95523fcef1f39ac27ada.jpg)  
Figure 1: Illustration of the continuous version of the cliff walking problem. The robot starts at  $(0,0)$  and must navigate to the goal area (in green). The robot gets negative rewards for being close to the obstacles and receives a reward of 10 upon reaching the goal.

and the KernelRidge regressor from Scikit-learn to approximate the state-action value function.

![](images/e9b17fae5bfe3824e7026019e50c53268935712fbcced1cec34624b6da5a8430.jpg)  
Figure 2: Estimated distribution of the cumulative reward when following the learned policy for different risk parameters. For  $\alpha = 0.9$  (leftmost plot), the policy is more risk-tolerant, which causes the average reward to be higher, but occasional small reward. As we decrease  $\alpha$ , the policy becomes more risk-averse, favoring safer paths with smaller average rewards and higher worst-case rewards.

![](images/8d338cfcf2f00b0a53a840e658f46d9e76310991b427805bd64f034ef6797a4a.jpg)

![](images/19baa2170f8460f47bb81cd5a7011cba458a5534efb123b40731a9a66cae35b7.jpg)

In Fig. 2, we show the histograms of robot's cumulative rewards that it receives in 50 episodes by following the learned policy with different values of the risk parameter  $\alpha = [0.9, 0.5, 0.1]$ . For smaller values of  $\alpha$ , the learned policy successfully mitigates the tail risk in the distribution, illustrated by the rightmost histogram having the smallest reward of at least 3.0, whereas the reward could go as low as near 0 for the remaining two policies. As we increase  $\alpha$ , the policy becomes more risk-tolerant, which leads to a higher average reward at the expense of some occasional bad rewards.

# 5.2 REAL-WORLD EXPERIMENT: TRADING

This trading setup is a generalization of the betting game environment (Bauerle & Ott, 2011; Rigter et al., 2021). This experiment considers a simplified foreign exchange trading environment based on real historical exchange rates and volumes between EUR and USD in 12 months of 2017. For simplicity, we fixed the trade volume for each hour at 10000. There are two actions in the environment: buy or sell. The state of the environment includes the current position, which is either long or short, and a vector of signal features containing the historical prices and trading volumes over a short period of time. We customize this environment based on the ForexEnv in the python package gym-anytrading.

In Fig. 3, we show a histogram of the cumulative terminal wealth achieved by the agents in 100 episodes with different risk parameters, plotted in different colors. Similar to the robot experiment, we demonstrate that for a smaller value of  $\alpha$ , the policy is risk-averse and

successfully mitigates the tail of the distribution. This can be seen that the worst-case wealth for  $\alpha = 0.1$  (in green) is higher than for  $\alpha = 0.5$  (in red) or  $\alpha = 0.9$  (in blue).

![](images/8b43cbd98065bc8a99ccd3990632a5b09efdce06cf89cf2f4fa439efd9433d0b.jpg)  
Figure 3: Estimated distribution of the normalized terminal wealth following the learned policy for different risk parameters. The vertical lines represent the average rewards. When  $\alpha = 0.9$  (the blue bar), the policy is more risk-tolerant, which causes the average reward to be higher at the expense of occasional low reward. As we decrease  $\alpha$ , the policy is more risk-averse, favoring safe paths with lower average-case rewards and higher worst-case rewards.

Computational complexity of RA-UCB: The kernel least-square regression has  $O(N^3)$  time complexity (due to the need for matrix inversion as shown in Eq. (8)) and  $O(N^2)$  memory complexity, where  $N$  is the number of samples used in the estimation. Since we need to solve  $H$  kernel ridge regression problems (i.e., one for each  $h \in [H]$ ) in each episode, RA-UCB will have  $O(Ht^3)$  time complexity and  $O(Ht^2)$  memory complexity for the  $t$ -th episode.

# 6 CONCLUSION

We proposed a risk-aware RL algorithm named RA-UCB that uses the coherent risk measures and non-linear function approximations. We then provided a finite-sample regret upper bound guarantee for RA-UCB and demonstrated its effectiveness in robot navigation and forex trading environments.

The performance of the proposed algorithm depends profoundly on the quality of the empirical risk estimator. This paper assumes access to a weak simulator that can sample the next states, thus effectively alleviating the need to estimate the risk from the observed trajectories. Therefore, a potential future direction is to relax or weaken this assumption, allowing risk-aware RL algorithms to be applicable in more practical problems. Another interesting direction is to consider the episodic MDPs, where episodes can have varying lengths horizons or even infinite horizons.

# 7 REPRODUCIBILITY STATEMENT

In this paper, we dedicate a substantial effort to improving the reproducibility and comprehensibility of both our theoretical results and empirical experiments. We formally state and discuss the necessity and implications of our assumptions (please see Section 3.3 and the paragraph below Assumption 1) before presenting our theoretical results. We also provide a 3-step proof sketch of our main theoretical result. For each step, we present the key ideas and high-level directions and refer the reader to more detailed and complete proofs in the Appendices. For the experiments, we provide details of different experimental settings in Section 5, and include our code in the supplementary materials (as a zip file).

# REFERENCES

Carlo Acerbi and Dirk Tasche. On the coherence of expected shortfall. Journal of Banking & Finance, 26(7):1487-1503, 2002.  
Amir Ahmadi-Javid. Entropic value-at-risk: A new coherent risk measure. Journal of Optimization Theory and Applications, 155(3):1105-1123, 2012.  
Philippe Artzner, Freddy Delbaen, Jean-Marc Eber, David Heath, and Hyejin Ku. Coherent multiperiod risk adjusted values and bellman's principle. Annals of Operations Research, 152(1): 5-22, 2007.  
Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, Second Series, 19(3):357-367, 1967.  
Dorian Baudry, Romain Gautron, Emilie Kaufmann, and Odalric Maillard. Optimal thompson sampling strategies for support-aware cvar bandits. In International Conference on Machine Learning, pp. 716-726. PMLR, 2021.  
Nicole Bäuerle and André Mundt. Dynamic mean-risk optimization in a binomial model. Mathematical Methods of Operations Research, 70(2):219-239, 2009.  
Nicole Bäuerle and Jonathan Ott. Markov decision processes with average-value-at-risk criteria. Mathematical Methods of Operations Research, 74(3):361-379, 2011.  
Nicole Bäuerle and Ulrich Rieder. More risk-sensitive markov decision processes. Mathematics of Operations Research, 39(1):105-120, 2014.  
Marc G. Bellemare, Will Dabney, and Mark Rowland. Distributional Reinforcement Learning. MIT Press, 2022. http://www.distributional-rl.org.  
Larry E Beutler, Kathleen Someah, Satoko Kimpara, and Kimberly Miller. Selecting the most appropriate treatment for each patient. International Journal of Clinical and Health Psychology, 16(1):99-108, 2016.  
Kang Boda and Jerzy A Filar. Time consistent dynamic risk measures. Mathematical Methods of Operations Research, 63(1):169-186, 2006.  
Vivek S Borkar. A sensitivity formula for risk-sensitive cost and the actor-critic algorithm. Systems & Control Letters, 44(5):339-346, 2001.  
Vivek S Borkar. Q-learning for risk-sensitive control. Mathematics of operations research, 27(2): 294-311, 2002.  
Yin-Lam Chow and Marco Pavone. Stochastic optimal control with dynamic, time-consistent risk constraints. In 2013 American Control Conference, pp. 390-395. IEEE, 2013.  
Yinlam Chow, Mohammad Ghavamzadeh, Lucas Janson, and Marco Pavone. Risk-constrained reinforcement learning with percentile risk criteria. The Journal of Machine Learning Research, 18(1):6070-6120, 2017.  
Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning, pp. 1096-1105. PMLR, 2018.

Michael Alan Howarth Dempster. Risk management: value at risk and beyond. Cambridge University Press, 2002.  
Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? arXiv preprint arXiv:1910.03016, 2019.  
Mehdi Fatemi, Taylor W Killian, Jayakumar Subramanian, and Marzyeh Ghassemi. Medical dead-ends and learning to identify high-risk states and treatments. Advances in Neural Information Processing Systems, 34, 2021.  
Yingjie Fei, Zhuoran Yang, Yudong Chen, Zhaoran Wang, and Qiaomin Xie. Risk-sensitive reinforcement learning: Near-optimal risk-sample tradeoff in regret. Advances in Neural Information Processing Systems, 33:22384-22395, 2020.  
Yingjie Fei, Zhuoran Yang, and Zhaoran Wang. Risk-sensitive reinforcement learning with function approximation: A debiasing approach. In International Conference on Machine Learning, pp. 3198-3207. PMLR, 2021.  
Hans Föllmer and Thomas Knispel. Entropic risk measures: Coherence vs. convexity, model ambiguity and robust large deviations. Stochastics and Dynamics, 11(02n03):333-351, 2011.  
Hans Föllmer and Alexander Schied. Convex and coherent risk measures. Encyclopedia of Quantitative Finance, pp. 355-363, 2010.  
Ronald A Howard and James E Matheson. Risk-sensitive markov decision processes. Management science, 18(7):356-369, 1972.  
Stratton C Jaquette. Markov decision processes with a new optimality criterion: Discrete time. The Annals of Statistics, 1(3):496-505, 1973.  
Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory, pp. 2137-2143. PMLR, 2020.  
Ramtin Keramati, Christoph Dann, Alex Tamkin, and Emma Brunskill. Being optimistic to be conservative: Quickly learning a cvar policy. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 4436-4443, 2020.  
Jakob Kisiala. Conditional value-at-risk: Theory and applications. arXiv preprint arXiv:1511.00140, 2015.  
Prashanth La and Mohammad Ghavamzadeh. Actor-critic algorithms for risk-sensitive mdps. Advances in neural information processing systems, 26, 2013.  
Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.  
Duan Li and Wan-Lung Ng. Optimal dynamic portfolio selection: Multiperiod mean-variance formulation. Mathematical finance, 10(3):387-406, 2000.  
Siqi Liu, Kay Choong See, Kee Yuan Ngiam, Leo Anthony Celi, Xingzhi Sun, Mengling Feng, et al. Reinforcement learning for clinical decision support in critical care: comprehensive review. Journal of medical Internet research, 22(7):e18477, 2020.  
Francis J McMahon and Thomas R Insel. Pharmacogenomics and personalized medicine in neuropsychiatry. Neuron, 74(5):773-776, 2012.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.  
Mehrdad Moharrami, Yashaswini Murthy, Argyadip Roy, and R Srikant. A policy gradient algorithm for the risk-sensitive exponential cost mdp. arXiv preprint arXiv:2202.04157, 2022.

Masahiro Ono, Marco Pavone, Yoshiaki Kuwata, and J Balaram. Chance-constrained dynamic programming with application to risk-aware robotic space exploration. Autonomous Robots, 39(4): 555-571, 2015.  
Carl Edward Rasmussen. Gaussian processes in machine learning. In Summer school on machine learning, pp. 63-71. Springer, 2003.  
Marc Rigter, Bruno Lacerda, and Nick Hawes. Risk-averse bayes-adaptive reinforcement learning. Advances in Neural Information Processing Systems, 34, 2021.  
R Tyrrell Rockafellar and Stanislav Uryasev. Conditional value-at-risk for general loss distributions. Journal of banking & finance, 26(7):1443-1471, 2002.  
R Tyrrell Rockafellar, Stanislav Uryasev, et al. Optimization of conditional value-at-risk. Journal of risk, 2:21-42, 2000.  
Mark Rowland, Marc Bellemare, Will Dabney, Rémi Munos, and Yee Whye Teh. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pp. 29-37. PMLR, 2018.  
Andrzej Ruszczyński. Risk-averse dynamic programming for markov decision processes. Mathematical programming, 125(2):235-261, 2010.  
Matthew J Sobel. The variance of discounted markov decision processes. Journal of Applied Probability, 19(4):794-802, 1982.  
Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995, 2009.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.  
Aviv Tamar, Dotan Di Castro, and Shie Mannor. Learning the variance of the reward-to-go. The Journal of Machine Learning Research, 17(1):361-396, 2016.  
Nelson Vadori, Sumitra Ganesh, Prashant Reddy, and Manuela Veloso. Risk-sensitive reinforcement learning: a martingale approach to reward uncertainty. In Proceedings of the First ACM International Conference on AI in Finance, pp. 1-9, 2020.  
Zhuoran Yang, Chi Jin, Zhaoran Wang, Mengdi Wang, and Michael I Jordan. On function approximation in reinforcement learning: Optimism in the face of large state spaces. arXiv preprint arXiv:2011.04622, 2020.  
Pengqian Yu, William B Haskell, and Huan Xu. Approximate value iteration for risk-aware markov decision processes. IEEE Transactions on Automatic Control, 63(9):3135-3142, 2018.  
Andrea Zanette, David Brandfonbrener, Emma Brunskill, Matteo Pirotta, and Alessandro Lazaric. Frequentist regret bounds for randomized least-squares value iteration. In International Conference on Artificial Intelligence and Statistics, pp. 1954-1964. PMLR, 2020.
