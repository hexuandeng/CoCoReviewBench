# SAMPLE EFFICIENT STOCHASTIC POLICY EXTRAGRADIENT ALGORITHM FOR ZERO-SUM MARKOV GAME

Anonymous authors

Paper under double-blind review

# ABSTRACT

Two-player zero-sum Markov game is a fundamental problem in reinforcement learning and game theory. Although many algorithms have been proposed for solving zero-sum Markov games in the existing literature, they generally lack the desired and important features such as model-free, provably convergent, sample efficient, symmetric and private policy updates, etc. In this paper, we develop a fully decentralized stochastic policy extragradient algorithm with all these properties for solving zero-sum Markov games. In particular, our algorithm introduces multiple stochastic estimators to accurately estimate the value functions involved in the stochastic updates, and leverages entropy regularization to accelerate the convergence. Specifically, with a proper entropy-regularization parameter, we prove that the stochastic policy extragradient algorithm has a sample complexity of the order  $\mathcal{O}\left(\frac{t_{\mathrm{mix}}A_{\mathrm{max}}}{\mu_{\mathrm{min}}\epsilon^{5.5}(1 - \gamma)^{13.5}}\right)$  for finding a solution that achieves  $\epsilon$ -Nash equilibrium duality gap. Such a sample complexity result substantially improves the state-of-the-art complexity results.

# 1 INTRODUCTION

Competitive reinforcement learning (RL) is an emerging and popular framework that has broad applications in various areas, including market pricing applications (Kononen and Oja, 2004), real-time strategy-making (Vinyals et al., 2019), board games (Silver et al., 2017; Moerland et al., 2018) and inverse RL (Zhang et al., 2019). In particular, an important and fundamental formulation of competitive RL is the two-player zero-sum Markov game, which involves two competing players that interact with a common environment and receive zero-sum rewards. Both players aim to learn the optimal policy that achieves the Nash equilibrium of accumulated rewards.

Algorithms for solving Markov games are very different from conventional single-agent RL algorithms. In particular, both players must learn to improve their policies based on feedback from the opponent and the environment, but usually the opponent will not reveal any sensitive information (e.g., actions or policy) or cooperate with each other. In the existing literature, numerous algorithms have been developed for solving zero-sum Markov games, including Q-learning (Fan et al., 2020; Zhu and Zhao, 2020), fitted Q iteration (Zhang et al., 2021), policy gradient (Daskalakis et al., 2020; Zhao et al., 2021), policy extragradient (Cen et al., 2021), model-based Monte Carlo estimation (Zhang et al., 2020), optimistic gradient descent ascent (Wei et al., 2021), etc. However, many of these algorithms require both players to access their opponent's actions (Wei et al., 2017; Sidford et al., 2020; Bai et al., 2020; Huang et al., 2021; Jafarnia-Jahromi et al., 2021), which causes privacy issues. On the other hand, some algorithms need to know about the environment transition kernel and reward mapping (Cen et al., 2021), which are usually unknown a priori in practice. Moreover, other algorithms require both players to perform asymmetric policy updates using different numbers of iterations, learning rates or exploration probabilities (Zhao et al., 2021; Daskalakis et al., 2020), which are generally hard to coordinate in advance between two competing players. Therefore, it is desired to develop an algorithm for solving Markov games that avoids all the aforementioned issues, and we want to ask the following question.

- Q1: Can we develop a fully decentralized algorithm that is model-free and takes symmetric and private policy updates for solving zero-sum Markov games?

Table 1: Summary of key properties and sample complexity of algorithms for solving discounted infinite-horizon zero-sum Markov games.  

<table><tr><td>Work</td><td>Model -free</td><td>Private update</td><td>Symmetric update</td><td>Data type</td><td>Sample complexity (duality gap ≤ ε)</td></tr><tr><td>Zou et al. (2019)</td><td>✓</td><td>×</td><td>✓</td><td>Markovian</td><td>-</td></tr><tr><td>Zhao et al. (2021)</td><td>✓</td><td>✓</td><td>×</td><td>i.i.d.</td><td>-</td></tr><tr><td>Guo et al. (2021)</td><td>✓</td><td>✓</td><td>×</td><td>i.i.d.</td><td>-</td></tr><tr><td>Cen et al. (2021)</td><td>×</td><td>✓</td><td>✓</td><td>-</td><td>-</td></tr><tr><td>Wei et al. (2021)</td><td>✓</td><td>✓</td><td>✓</td><td>Markovian</td><td>O(A3max|S|10.5/ε8(1-γ)29.5)</td></tr><tr><td>Our work</td><td>✓</td><td>✓</td><td>✓</td><td>Markovian</td><td>O(Amax/ε5.5(1-γ)13.5)</td></tr></table>

Moreover, from a theoretical perspective, the convergence and sample complexity of these existing algorithms for solving Markov games have not been comprehensively studied. Specifically, some studies established the convergence of the algorithms with i.i.d. samples (Zhao et al., 2021; Guo et al., 2021), which violates the dependent nature of samples collected from the dynamic Markov decision process. Also, other algorithms suffer from an extremely high sample complexity to achieve an approximate Nash equilibrium solution (Wei et al., 2021). Hence, we are motivated to answer the following fundamental question.

- Q2: Can we develop the algorithm mentioned in Q1 with provable convergence guarantee and an improved sample complexity for achieving a Nash equilibrium solution?

In this paper, we provide positive and comprehensive answers to both questions by developing a fully decentralized stochastic policy extragradient algorithm. In Table 1, we compare the key properties and sample complexity of our algorithm with those of the existing algorithms. We also refer to Appendix F for more explanations on Table 1. Our contributions are summarized as follows.

# 1.1 OUR CONTRIBUTIONS

We consider a standard zero-sum Markov game with discounted reward over infinite horizon. To solve such a Markov game, we propose a stochastic variant of the policy extragradient algorithm (Cen et al., 2021) that satisfies the following amenable properties.

- Our algorithm uses multiple new stochastic estimators to estimate the value functions involved in the predictive updates for solving entropy-regularized matrix games, and therefore the algorithm does not rely on any prior knowledge of the environment transition kernel (model-free). Moreover, the resulting stochastic policy updates of our algorithm for both players are symmetric and do not involve any sensitive information of the opponent (private).  
- Compared with the stochastic estimators used in (Wei et al., 2021), our estimators have much smaller variance that helps improve the estimation accuracy. Specifically, by developing new techniques (explained in the next bullet), we establish tight high-probability estimation error bounds for our stochastic estimators with Markovian samples. Then, with a proper entropy regularization parameter, we show that our stochastic policy extragradient algorithm requires a sample complexity of the order  $\mathcal{O}\left(\frac{t_{\mathrm{mix}}A_{\mathrm{max}}}{\mu_{\mathrm{min}}\epsilon^{3.5}(1 - \gamma)^{13.5}}\right)$  to achieve an  $\epsilon$ -Nash equilibrium duality gap, which substantially improves the state-of-the-art complexity result of (Wei et al., 2021).  
- We develop novel techniques to bound the estimation error of the proposed stochastic estimators, whose numerator and denominator involve sample average approximations. First, we propose a special estimation error decomposition that avoids divergence of the bound caused by possibly small numerical values of the sample average involved in the denominator of the stochastic estimators. Second, we leverage this error decomposition and the recursive structure of the stochastic estimators to derive a contraction property of the estimation errors, which eventually leads to tight bounds for the estimation error. We refer to Section 4 for more elaboration on our technical novelties.

# 1.2 OTHER RELATED WORK

Other settings of two-player zero-sum Markov games. In this paper, we focus on a standard setting of two-player zero-sum Markov game with discount and infinite horizon in the discrete time domain. There are other settings of two-player zero-sum Markov games. For example, Bai et al. (2020); Huang et al. (2021) studied a two-player zero-sum Markov game with finite horizon and without discount, whereas Daskalakis et al. (2020) considered finite random horizon without discount. Jafarnia-Jahromi et al. (2021) also considered the setting without discount, and it allows one of the players to constantly adjust its policy based on the entire history of states and actions. Ghosh et al. (2021) studied a two-player zero-sum Markov game in the continuous time domain.

Multi-agent general-sum Markov game. Some works studied multi-agent Markov games, which extend the two-player zero-sum Markov games to multiple players without the zero-sum constraint (Wang and Sandholm, 2002; Hu and Wellman, 2003; Deng et al., 2021; Leonardos et al., 2021). More specifically, Leonardos et al. (2021) defined and studied Markov potential game which has a potential function assumption. Guo et al. (2019); Elie et al. (2020); Gu et al. (2021) studied mean-field games with a large number of players.

Entropy regularization and value iteration Our algorithm leverages entropy regularization and value iteration to accelerate convergence. Entropy regularization is a popular technique that has been widely used in reinforcement learning (Neu et al., 2017; Geist et al., 2019; Mei et al., 2020; Cen et al., 2020) and Markov game (Mertikopoulos and Sandholm, 2016; Savas et al., 2019; Cen et al., 2021) to encourage environment exploration and accelerate algorithm convergence. Value iteration is also a classical method that is widely used in both single-agent reinforcement learning (Ernst et al., 2005; Tamar et al., 2016; Farahmand and Ghavamzadeh, 2021) and Markov games (Zhu and Zhao, 2020; Cen et al., 2021). With full knowledge of the environment, it exponentially converges to the fixed point of Bellman operator (Cen et al., 2021). Compared to another similar classical method called policy iteration, value iteration does not need policy evaluation which involves additional computation (Sutton and Barto, 2018).

# 2 BACKGROUND OF MARKOV GAME AND ENTROPY REGULARIZATION

# 2.1 TWO-PLAYER ZERO-SUM MARKOV GAME

In a zero-sum Markov game, two players compete with each other in a common environment. Throughout, the state space is denoted as  $S$ . The action spaces and policies of both players are denoted as  $\mathcal{A}^{(1)}, \pi^{(1)}$  and  $\mathcal{A}^{(2)}, \pi^{(2)}$ , respectively. Here,  $\pi^{(1)} \in \Delta(|\mathcal{A}^{(1)}|), \pi^{(2)} \in \Delta(|\mathcal{A}^{(2)}|)$  are random policies defined over the corresponding simplex sets. The reward function is denoted as  $R: S \times \mathcal{A}^{(1)} \times \mathcal{A}^{(2)} \to [0, 1]$ , and the discount factor is denoted as  $\gamma \in (0, 1)$ .

At any time  $t$ , both players observe state  $s_t \in S$  of the environment. Then, both players respectively select their actions following their own policies, i.e.,  $a_t^{(1)} \sim \pi^{(1)}(\cdot | s_t)$  and  $a_t^{(2)} \sim \pi^{(2)}(\cdot | s_t)$ . After that, the environment state transfers to a new state  $s_{t+1}$  following the underlying transition kernel  $\mathcal{P}(\cdot | s_t, a_t^{(1)}, a_t^{(2)})$ , and both players receive zero-sum rewards, i.e.,  $R_t^{(1)} = -R_t^{(2)} = R_t$ , where  $R_t := R(s_t, a_t^{(1)}, a_t^{(2)})$ . With this Markov decision process, we can define the following state value function associated with the players' policies  $\pi^{(1)}$  and  $\pi^{(2)}$  for any environment state  $s \in S$ .

$$
V _ {\pi^ {(1)}, \pi^ {(2)}} (s) = \mathbb {E} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} R _ {t} \mid s _ {0} = s \right]. \tag {1}
$$

The goal of both players is to compete via the following minimax game in all states  $s$ .

$$
\min  _ {\pi^ {(2)}} \max  _ {\pi^ {(1)}} V _ {\pi^ {(1)}, \pi^ {(2)}} (s). \tag {2}
$$

In particular, it has been shown in (Shapley, 1953) that there exists a Nash equilibrium policy pair  $\pi_{*}^{(1)},\pi_{*}^{(2)}$  for zero-sum Markov games, i.e.,  $V_{\pi^{(1)},\pi_{*}^{(2)}}(s)\leq V_{\pi_{*}^{(1)},\pi_{*}^{(2)}}(s)\leq V_{\pi_{*}^{(1)},\pi^{(2)}}(s)$  holds for any other policies  $\pi^{(1)},\pi^{(2)}$  and for all states  $s$ .

# 2.2 ENTROPY-REGULARIZED MARKOV GAME

Entropy regularization is a popular technique that has been widely used in reinforcement learning (Neu et al., 2017; Geist et al., 2019; Mei et al., 2020; Cen et al., 2020) and Markov game (Mertikopoulos and Sandholm, 2016; Savas et al., 2019; Cen et al., 2021) to encourage environment exploration and accelerate algorithm convergence.

Specifically, for the zero-sum Markov game, we can define an entropy-regularized state value function by adding entropy regularization to the state value function in (1) as follows (Cen et al., 2021).

$$
V _ {\pi^ {(1)}, \pi^ {(2)}} ^ {(\tau)} (s) := \mathbb {E} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} \left[ R _ {t} - \tau \ln \pi^ {(1)} \left(a _ {t} ^ {(1)} \mid s _ {t}\right) + \tau \ln \pi^ {(2)} \left(a _ {t} ^ {(2)} \mid s _ {t}\right) \right] \mid s _ {0} = s \right], \tag {3}
$$

where  $\tau > 0$  is called the regularization parameter. With the above definition, we further define the following entropy-regularized state-action value function (also called  $Q$ -function) (Cen et al., 2021).

$$
Q _ {\pi^ {(1)}, \pi^ {(2)}} ^ {(\tau)} (s, a ^ {(1)}, a ^ {(2)}) := R (s, a ^ {(1)}, a ^ {(2)}) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a ^ {(1)}, a ^ {(2)})} \left[ V _ {\pi^ {(1)}, \pi^ {(2)}} ^ {(\tau)} (s ^ {\prime}) \right]. \tag {4}
$$

In particular,  $V_{\pi^{(1)},\pi^{(2)}}^{(\tau)}$  can be obtained from  $Q_{\pi^{(1)},\pi^{(2)}}^{(\tau)}$  as follows.

$$
\begin{array}{l} V _ {\pi^ {(1)}, \pi^ {(2)}} ^ {(\tau)} (s) = \left[ \pi^ {(1)} (s) \right] ^ {\top} Q _ {\pi^ {(1)}, \pi^ {(2)}} ^ {(\tau)} (s) \pi^ {(2)} (s) + \tau \mathcal {H} \left(\pi^ {(1)} (s)\right) - \tau \mathcal {H} \left(\pi^ {(2)} (s)\right) \\ := f _ {\tau} \left(Q _ {\pi^ {(1)}, \pi^ {(2)}} ^ {(\tau)} (s), \pi^ {(1)} (s), \pi^ {(2)} (s)\right), \tag {5} \\ \end{array}
$$

where  $\mathcal{H}(\pi)$  denotes the entropy of policy  $\pi$ , and we define this mapping as  $f_{\tau}$  for convenience.

For the entropy regularized Markov game, it has an equilibrium policy pair that solves the minimax optimization problem  $\min_{\pi^{(2)}}\max_{\pi^{(1)}}V_{\pi^{(1)},\pi^{(2)}}^{\tau}(s)$ . Such a policy pair is called the quantal response equilibrium (QRE). Our goal is to find the equilibrium policy pair of the original Markov game in (2) by solving the entropy-regularized Markov game with a proper regularization parameter  $\tau$ . In particular, compared with the equilibrium policy of the Markov game, the QRE tends to have a larger entropy due to the entropy regularization, which encourages the players to explore and obtain a better understanding of the environment. Another advantage of considering the entropy-regularized Markov game is that the entropy regularization makes the minimax problem have a better optimization geometry that accelerates the convergence of the optimization process.

# 3 STOCHASTIC POLICY EXTRAGRADIENT ALGORITHM FOR ENTROPY-REGULARIZED MARKOV GAME

In this section, we develop a stochastic policy extragradient (SPE) algorithm for solving entropy-regularized Markov games. First, we recap the policy extragradient (PE) algorithm, which is introduced in (Cen et al., 2021) to solve entropy-regularized Markov games with full knowledge of the environment transition kernel and reward mapping. Then, we propose the model-free SPE algorithm that solves entropy-regularized Markov games using only stochastic samples.

# 3.1 REVIEW OF POLICY EXTRAGRADIENT ALGORITHM

Value iteration is a classical reinforcement learning algorithm that requires full knowledge of the environment and achieves an exponential convergence rate. In particular, for the entropy-regularized Markov game, the  $k$ -th value iteration update is defined as follows.

$$
Q _ {k} \left(s, a ^ {(1)}, a ^ {(2)}\right) = R \left(s, a ^ {(1)}, a ^ {(2)}\right) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a ^ {(1)}, a ^ {(2)})} \left[ V _ {k} \left(s ^ {\prime}\right) \right], \quad \forall s, a ^ {(1)}, a ^ {(2)}, \tag {6}
$$

$$
V _ {k + 1} (s) = \min  _ {\pi^ {(2)} (s)} \max  _ {\pi^ {(1)} (s)} f _ {\tau} \left(Q _ {k} (s); \pi^ {(1)} (s), \pi^ {(2)} (s)\right), \quad \forall s, \tag {7}
$$

where we define  $Q_{k}(s)\coloneqq Q_{k}(s,\cdot ,\cdot)\in \mathbb{R}^{|\mathcal{A}^{(1)}| \times |\mathcal{A}^{(2)}|}$ . This algorithm alternatively updates all the entries of the value functions  $Q_{k}$  and  $V_{k}$ . Thanks to the entropy regularization in the function  $f_{\tau}$  (see (5) for the definition), the minimax matrix game in (7) is  $\tau$ -strongly concave in  $\pi^{(1)}$  and  $\tau$ -strongly convex in  $\pi^{(2)}$ , and therefore it has a unique solution.

To solve the entropy-regularized minimax matrix game in (7), Cen et al. (2021) proposed a predictive update (PU) algorithm. Specifically, with uniform policy initialization, i.e.,  $\pi_{k,0}^{(m)}(s) = \frac{1}{|\mathcal{A}^{(m)}|},\forall m\in$ $\{1,2\} ,\forall s\in S$  , the PU algorithm performs the following policy updates: for  $t = 0,1,2,\ldots$

$$
\text {(P U)}: \left\{ \begin{array}{l} \overline {{\pi}} _ {k, t + 1} ^ {(1)} \left(a ^ {(1)} | s\right) \propto \pi_ {k, t} ^ {(1)} \left(a ^ {(1)} | s\right) ^ {1 - \eta \tau} \exp \left(\eta Q _ {k, t} ^ {(1)} \left(s, a ^ {(1)}\right)\right) \\ \overline {{\pi}} _ {k, t + 1} ^ {(2)} \left(a ^ {(2)} | s\right) \propto \pi_ {k, t} ^ {(2)} \left(a ^ {(2)} | s\right) ^ {1 - \eta \tau} \exp \left(- \eta Q _ {k, t} ^ {(2)} \left(s, a ^ {(2)}\right)\right) \\ \pi_ {k, t + 1} ^ {(1)} \left(a ^ {(1)} | s\right) \propto \pi_ {k, t} ^ {(1)} \left(a ^ {(1)} | s\right) ^ {1 - \eta \tau} \exp \left(\eta \bar {Q} _ {k, t + 1} ^ {(1)} \left(s, a ^ {(1)}\right)\right) \\ \pi_ {k, t + 1} ^ {(2)} \left(a ^ {(2)} | s\right) \propto \pi_ {k, t} ^ {(2)} \left(a ^ {(2)} | s\right) ^ {1 - \eta \tau} \exp \left(- \eta \bar {Q} _ {k, t + 1} ^ {(2)} \left(s, a ^ {(2)}\right)\right) \end{array} , \right. \tag {8}
$$

where we use the following notations (superscript  $(\backslash m)$  denotes the opponent of the  $m$ -th player.).

$$
Q _ {k, t} ^ {(m)} (s, a ^ {(m)}) := \mathbb {E} _ {a ^ {(\backslash m)} \sim \pi_ {k, t} ^ {(\backslash m)}} (s) \left[ Q _ {k} (s, a ^ {(1)}, a ^ {(2)}) \right], \quad m \in \{1, 2 \} \tag {9}
$$

$$
\bar {Q} _ {k, t + 1} ^ {(m)} (s, a ^ {(m)}) := \mathbb {E} _ {a ^ {(\backslash m)} \sim \bar {\pi} _ {k, t + 1} ^ {(\backslash m)} (s)} \left[ Q _ {k} (s, a ^ {(1)}, a ^ {(2)}) \right], \quad m \in \{1, 2 \}. \tag {10}
$$

Once we obtain the output policy pair  $(\pi_k^{(1)},\pi_k^{(2)})$  of the PU algorithm, we can obtain an approximation of  $V_{k + 1}(s)$  as  $V_{k + 1}^{\prime}(s) = f_{\tau}\big(Q_k(s);\pi_k^{(1)}(s),\pi_k^{(2)}(s)\big)$ , which will be further used in the next  $Q$ -value function update (6) to replace  $V_{k + 1}(s^{\prime})$ . The updates (6), (7) & (8) are referred to as policy extragradient (PE) algorithm.

In the PE algorithm, the PU update in (8) allows both players to take symmetric updates without revealing their private actions, and has been shown to converge to the unique solution of the entropy-regularized matrix game (7) exponentially fast (Cen et al., 2021). However, the PE algorithm has several limitations. First, in the PU update, each player  $m \in \{1, 2\}$  needs to query the quantities  $Q_{k,t}^{(m)}(s, a^{(m)})$ ,  $\overline{Q}_{k,t}^{(m)}(s, a^{(m)})$  from its opponent. To compute these quantities, the opponent needs to multiply the entire  $Q$ -table by its own policy vector. This requires both players to coordinate with each other and share a  $Q$ -table. Second, the update of the  $Q$ -table in (6) requires full knowledge of the environment transition kernel  $\mathcal{P}$  and the reward mapping  $R$ , which are unknown a priori in practice. To overcome these limitations, we develop a fully stochastic PE algorithm in the next subsection.

# 3.2 STOCHASTIC POLICY EXTRAGRADIENT ALGORITHM

The major challenge of the PE algorithm is computing the quantities  $Q_{k,t}^{(m)}$ ,  $\overline{Q}_{k,t}^{(m)}$  and  $V_{k+1}^{\prime}$ , which requires coordinating with the opponent and involves the environment information. Here, we develop a model-free and fully stochastic variant of PE that estimates these key quantities using Markovian stochastic samples. We refer to this algorithm as stochastic policy extragradient (SPE).

Specifically, we first estimate the quantity  $V_{k + 1}^{\prime}(s) = f_{\tau}\big(Q_k(s);\pi_k^{(1)}(s),\pi_k^{(2)}(s)\big)$ . By definition of  $f_{\tau}$  in (5) and the update of  $Q_{k}$  in (6) (now we use  $V_{k}^{\prime}(s^{\prime})$  instead of  $V_{k}(s^{\prime})$ ) and using some standard tricks on random variables (see Lemma 2 in Appendix B for a full proof), we can rewrite  $V_{k + 1}^{\prime}(s)$  as

$$
V _ {k + 1} ^ {\prime} (s) = \frac {\mathbb {E} \left[ \left(R \left(\widetilde {s} , \widetilde {a} ^ {(1)} , \widetilde {a} ^ {(2)}\right) + \gamma V _ {k} ^ {\prime} \left(s ^ {\prime}\right)\right) \mathbb {1} \{\widetilde {s} = s \} \right]}{\mu_ {k} (s)} + \tau \mathcal {H} \left(\pi_ {k} ^ {(1)} (s)\right) - \tau \mathcal {H} \left(\pi_ {k} ^ {(2)} (s)\right), \tag {11}
$$

where  $\mu_k(s)$  denotes the stationary state distribution associated with the policy pair  $(\pi_k^{(1)},\pi_k^{(2)})$ , and the expectation is taken over  $\widetilde{s}\sim \mu_k,\widetilde{a}^{(1)}\sim \pi_k^{(1)}(s),\widetilde{a}^{(2)}\sim \pi_k^{(2)}(s),s'\sim \mathcal{P}(\cdot |\widetilde{s},\widetilde{a}^{(1)},\widetilde{a}^{(2)})$ . To estimate this quantity, we query a set  $\mathcal{N}_{k + 1}$  (with cardinality  $N_{k + 1}$ ) of samples from the Markov decision process following the pair of policies  $(\pi_k^{(1)},\pi_k^{(2)})$ . Then, we estimate  $V_{k + 1}'(s)$  as

$$
\widehat {V} _ {k + 1} (s) = \frac {\frac {1}{N _ {k + 1}} \sum_ {i \in \mathcal {N} _ {k + 1}} \left(R _ {i} + \gamma \widehat {V} _ {k} \left(s _ {i + 1}\right)\right) \mathbb {1} \left\{s _ {i} = s \right\}}{\frac {1}{N _ {k + 1}} \sum_ {i \in \mathcal {N} _ {k + 1}} \mathbb {1} \left\{s _ {i} = s \right\}} + \tau \mathcal {H} \left(\pi_ {k} ^ {(1)} (s)\right) - \tau \mathcal {H} \left(\pi_ {k} ^ {(2)} (s)\right). \tag {12}
$$

Intuitively, we use the sample average of Markovian samples to estimate the expectation terms in (11). Thanks to the concentration phenomenon of dependent samples (Paulin, 2015), these sample averages converge to the desired expected values provided that the sample size is sufficiently large.

Next, we estimate  $Q_{k,t}^{(m)}, m \in 1,2$ . Leveraging (9) and (6), we obtain the following equivalent characterization for both players  $m \in 1,2$  (see Lemma 2 in Appendix B for the proof of equivalence).

$$
Q _ {k, t} ^ {(m)} \left(s, a ^ {(m)}\right) = \frac {\mathbb {E} \left[ \left(R \left(\widetilde {s} , \widetilde {a} ^ {(1)} , \widetilde {a} ^ {(2)}\right) + \gamma V _ {k} ^ {\prime} \left(s ^ {\prime}\right)\right) \mathbb {1} \{\widetilde {s} = s , \widetilde {a} ^ {(m)} = a ^ {(m)} \} \right]}{\mu_ {k , t} (s) \pi_ {k , t} ^ {(m)} \left(a ^ {(m)} \mid s\right)}, \tag {13}
$$

where  $\mathbb{1}\{\cdot\}$  denotes the indicator function,  $\mu_{k,t}$  denotes the stationary state distribution associated with the policy pair  $(\pi_{k,t}^{(1)},\pi_{k,t}^{(2)})$ , and the expectation is taken over  $\widetilde{s}\sim \mu_{k,t},\widetilde{a}^{(1)}\sim \pi_{k,t}^{(1)}(s),\widetilde{a}^{(2)}\sim$ $\pi_{k,t}^{(2)}(s),s^{\prime}\sim \mathcal{P}(\cdot |\widetilde{s},\widetilde{a}^{(1)},\widetilde{a}^{(2)})$ . To estimate this quantity, we query a set  $\mathcal{N}_{k,t}$  (with cardinality  $N_{k,t}$ ) of samples from the Markov decision process following a pair of smoothed policies  $\pi_{k,t}^{\prime(m)}(s) = (1 - \epsilon')\pi_{k,t}^{(m)}(s) + \frac{\epsilon'}{|A^{(m)}}\mathbf{1}$ , where  $\epsilon' \in [0,1]$  is a small smoothing constant that will be theoretically determined later. Then, we estimate  $Q_{k,t}^{(m)}(s,a^{(m)})$  as follows.

$$
\widehat {Q} _ {k, t} ^ {(m)} (s, a ^ {(m)}) = \frac {\frac {1}{N _ {k , t}} \sum_ {i \in \mathcal {N} _ {k , t}} \left(R _ {i} + \gamma \widehat {V} _ {k} \left(s _ {i + 1}\right)\right) \mathbb {1} \left\{s _ {i} = s , a _ {i} ^ {(m)} = a ^ {(m)} \right\}}{\left(\frac {1}{N _ {k , t}} \sum_ {i \in \mathcal {N} _ {k , t}} \mathbb {1} \left\{s _ {i} = s \right\}\right) \pi_ {k , t} ^ {\prime (m)} (s)}, \tag {14}
$$

where we have replaced the expectations with sample averages, and replaced  $V_{k}^{\prime}(s^{\prime})$  with  $\widehat{V}_k(s')$ . Here, the Markovian samples are queried following the  $\epsilon'$ -smoothed policies  $(\pi_{k,t}^{\prime(1)}, \pi_{k,t}^{\prime(2)})$ . On one hand,  $\epsilon'$  should not be too small so that it keeps the denominator of the above estimation away from zero. On the other hand,  $\epsilon'$  should not be too large so that it is sufficiently close to the original policy. Similarly, to estimate  $\overline{Q}_{k,t}^{(m)}$ , we query another set  $\overline{\mathcal{N}}_{k,t}$  (with cardinality  $\overline{N}_{k,t}$ ) of samples from the Markov decision process following a pair of smoothed policies  $\overline{\pi}_{k,t}^{\prime(m)}(s) = (1 - \overline{\epsilon}')\overline{\pi}_{k,t}^{(m)}(s) + \frac{\overline{\epsilon}'}{|\mathcal{A}^{(m)}|}\mathbf{1}$ , where  $\overline{\epsilon}' \in [0,1]$  will be theoretically determined later. Then, we estimate  $\overline{Q}_{k,t}^{(m)}$  as follows.

$$
\widehat {\bar {Q}} _ {k, t} ^ {(m)} (s, a ^ {(m)}) = \frac {\frac {1}{\bar {N} _ {k , t}} \sum_ {i \in \bar {\mathcal {N}} _ {k , t}} \left(R _ {i} + \gamma \widehat {V} _ {k} \left(s _ {i + 1}\right)\right) \mathbb {1} \left\{s _ {i} = s , a _ {i} ^ {(m)} = a ^ {(m)} \right\}}{\left(\frac {1}{\bar {N} _ {k , t}} \sum_ {i \in \bar {\mathcal {N}} _ {k , t}} \mathbb {1} \left\{s _ {i} = s \right\}\right) \overline {{\pi}} _ {k , t} ^ {\prime (m)} (s)}. \tag {15}
$$

Remark 1. The above estimators improve over those introduced in (Wei et al., 2021). First, to estimate the state-action probability involved in the denominator of  $Q_{k,t}^{(m)}(s,a^{(m)})$  and  $\overline{Q}_{k,t}^{(m)}(s,a^{(m)})$ , Wei et al. (2021) uses the state-action frequency estimator  $\frac{1}{N_{k,t}}\sum_{i\in \mathcal{N}_{k,t}}\mathbb{1}\{s_i = s,a_i^{(m)} = a^{(m)}\}$ , which induces a large variance due to the large state-action product space. As a comparison, we use the estimator  $\left(\frac{1}{N_{k,t}}\sum_{i\in \mathcal{N}_{k,t}}\mathbb{1}\{s_i = s\}\right)\pi_{k,t}^{\prime (m)}(s)$  that involves only the state frequency and the smoothed policy  $\pi_{k,t}^{\prime (m)}(s)$ . Therefore, our estimator has a much smaller variance. Second, to estimate  $V_{k + 1}^{\prime}$ , Wei et al. (2021) follows the smoothed policies to query the samples, while we follow the original policies  $(\pi_k^{(1)},\pi_k^{(2)})$ . This makes our estimator more accurate.

We summarize our stochastic policy extragradient (SPE) algorithm in Algorithm 1. Specifically, in SPE, we estimate the quantities  $Q_{k,t}^{(m)}$ ,  $\overline{Q}_{k,t}^{(m)}$ ,  $V_k'$  using their corresponding stochastic estimators. As a result, the SPE algorithm is model-free, and the updates for both players are symmetric and private.

# 4 FINITE-TIME CONVERGENCE ANALYSIS OF SPE

Throughout our convergence analysis, we adopt the following two standard assumptions.

Assumption 1. Denote  $T_{\pi^{(1)}, \pi^{(2)}}(s, s') := \inf \{t \geq 1 : s_t = s'|s_0 = s\}$  as the first-visit time under the policy pair  $\pi^{(1)}, \pi^{(2)}$ . We assume that

$$
\sup  _ {s, s ^ {\prime} \in \mathcal {S}} \sup  _ {\pi^ {(1)}, \pi^ {(2)}} \mathbb {E} _ {\pi^ {(1)}, \pi^ {(2)}} \left[ T _ {\pi^ {(1)}, \pi^ {(2)}} (s, s ^ {\prime}) \right] <   + \infty . \tag {16}
$$

Algorithm 1 Stochastic policy extragradient (SPE) for entropy-regularized Markov game  
Initialize:  $V_0^{\prime}(s)$  for all  $s\in S$    
for value iterations  $k = 0,1,\ldots ,K - 1$  do Initialize  $\pi_{k,0}^{(1)},\pi_{k,0}^{(2)}$  with uniform distribution.   
for PU iterations  $t = 0,1,\dots ,T_k - 1$  do Players 1,2 sample  $N_{k,t}$  Markovian samples following smoothed policies  $\pi_{k,t}^{\prime (1)},\pi_{k,t}^{\prime (2)}$  Every player  $m\in \{1,2\}$  computes  $\widehat{Q}_{k,t}^{(m)}(s,a^{(m)})$  for all  $s,a^{(m)}$  using (14). Players 1,2 sample  $\overline{N}_{k,t}$  Markovian samples following smoothed policies  $\overline{\pi}_{k,t}^{\prime (1)},\overline{\pi}_{k,t}^{\prime (2)}$  Every player  $m\in \{1,2\}$  computes  $\widehat{\overline{Q}}_{k,t}^{(m)}(s,a^{(m)})$  for all  $s,a^{(m)}$  using (15). Implement the t-th PU iteration for all  $s,a^{(1)},a^{(2)}$  using (8) with estimations (14)&(15). end Let  $\pi_k^{(m)} = \overline{\pi}_{k,T_k}^{(m)}$ $m = 1,2$  . Players sample  $N_{k}$  Markovian samples following  $\pi_k^{(1)},\pi_k^{(2)}$  Compute  $\widehat{V}_{k + 1}(s)$  for all  $s$  using (12).   
end   
Output:  $\pi_{K - 1}^{(1)},\pi_{K - 1}^{(2)}$

Assumption 1 is widely used in the reinforcement learning literature (Ortner and Auer, 2007; Ortner, 2020; Wei et al., 2021; Jafarnia-Jahromi et al., 2021). It ensures that every state will be visited at least once within a finite duration of time, thus ensuring that all the states will be visited infinitely often. This guarantees sufficient exploration. In our analysis, we use the following equivalent statement of Assumption 1 for convenience, which means that the stationary state distribution  $\mu_{\pi^{(1)},\pi^{(2)}}$  has a uniform lower bound  $\mu_{\mathrm{min}} > 0$ . Their equivalence is based on Theorem 5.5.11 of (Durrett, 2019).

$$
\mu_ {\min } := \inf  _ {s \in \mathcal {S}} \inf  _ {\pi^ {(1)}, \pi^ {(2)}} \mu_ {\pi^ {(1)}, \pi^ {(2)}} (s) = \left[ \sup  _ {s \in \mathcal {S}} \sup  _ {\pi^ {(1)}, \pi^ {(2)}} \mathbb {E} _ {\pi^ {(1)}, \pi^ {(2)}} T _ {\pi^ {(1)}, \pi^ {(2)}} (s, s) \right] ^ {- 1} > 0. \tag {17}
$$

Assumption 2. There exists a mixing time  $t_{\text{mix}} \in \mathbb{N}$  such that for any policy pair  $\pi^{(1)}, \pi^{(2)}$  and its corresponding stationary state distribution  $\mu_{\pi^{(1)}, \pi^{(2)}}$ , we have

$$
d _ {\mathrm {T V}} \left(\mathcal {P} _ {\pi^ {(1)}, \pi^ {(2)}} \left(s _ {t _ {m i x}}\right), \mu_ {\pi^ {(1)}, \pi^ {(2)}}\right) \leq \frac {1}{4}.
$$

where  $\mathcal{P}_{\pi^{(1)},\pi^{(2)}}(s_{t_{mix}})$  denotes the state distribution under the policy pair  $\pi^{(1)},\pi^{(2)}$  at time  $t_{mix}$ , and  $d_{\mathrm{TV}}$  denotes the total variation distance between two probability distributions.

Assumption 2 is also widely adopted in the existing literature (Greensmith et al., 2004; Ortner, 2020; Ciosek, 2021). It ensures that the state distribution is not too far away from its stationary distribution within a finite time  $t_{\mathrm{mix}}$ . We note that this assumption is much weaker than the popular uniformly ergodic Markov chain assumption, which assumes that  $\mathrm{d}_{TV}(\mathcal{P}_{\pi^{(1)},\pi^{(2)}}(s_t),\mu_{\pi^{(1)},\pi^{(2)}})$  decays exponentially fast as  $t\to \infty$  (Bhandari et al., 2018; Qiu et al., 2019; Xu and Liang, 2021).

In this subsection, we analyze the finite-time convergence of Algorithm 1 for solving the entropy-regularized Markov game (5). We focus on the convergence rate of the following Nash equilibrium duality gap, which is a standard optimality metric widely adopted in the existing literature (Xu et al., 2020; Jin and Sidford, 2021; Wei et al., 2021).

$$
D ^ {(\tau)} (\pi^ {(1)}, \pi^ {(2)}) := \max  _ {s} \Big (\max  _ {\pi} V _ {\pi , \pi^ {(2)}} ^ {(\tau)} (s) - \min  _ {\pi^ {\prime}} V _ {\pi^ {(1)}, \pi^ {\prime}} ^ {(\tau)} (s) \Big).
$$

In particular, when  $\tau = 0$ ,  $D^{(0)}(\pi^{(1)},\pi^{(2)})$  corresponds to the duality gap of the original Markov game. Throughout, we define  $A_{\mathrm{max}}\coloneqq \max \{|A^{(1)}|,|A^{(2)}|\}$ ,  $Q_{\mathrm{max}}\coloneqq \frac{1 + \gamma\tau\ln A_{\mathrm{max}}}{1 - \gamma}$  and  $V_{\mathrm{max}}\coloneqq \frac{1 + \tau\ln A_{\mathrm{max}}}{1 - \gamma}$ . We also require that the batch sizes of Algorithm 1 satisfy the following conditions.

$$
N _ {k, t}, \bar {N} _ {k, t} \geq \frac {6 5 0 t _ {\operatorname* {m i x}} A _ {\operatorname* {m a x}}}{\mu_ {\operatorname* {m i n}}} \ln \left(\frac {2 0 T _ {\operatorname* {s u m}} | \mathcal {S} | A _ {\operatorname* {m a x}}}{\delta \sqrt {\mu_ {\operatorname* {m i n}}}}\right), \quad \forall k, t, \tag {18}
$$

$$
N _ {k + 1} \geq \frac {6 5 0 t _ {\operatorname* {m i x}}}{\mu_ {\operatorname* {m i n}} (1 - \gamma) ^ {2}} \ln \left(\frac {4}{\delta \sqrt {\mu_ {\operatorname* {m i n}}}}\right), \quad \forall k. \tag {19}
$$

Then, we obtain the following convergence result of Algorithm 1, where  $T_{\mathrm{sum}} \coloneqq \sum_{k=0}^{K-1} T_k$ .

Theorem 1 (Finite-time convergence rate). Apply Algorithm 1 to solve the entropy-regularized Markov game with  $\tau \in (0,1]$ . Choose learning rate  $\eta = [2(\tau + Q_{\max})]^{-1}$ , initialization  $\| \widehat{V}_0 \|_\infty \leq V_{\max}$  and batch sizes  $N_{k,t}, \overline{N}_{k,t}, N_{k+1}$  that satisfy (18) & (19). Then, the Nash equilibrium duality gap converges at the following rate with probability at least  $1 - \delta$ .

$$
\begin{array}{l} D ^ {(\tau)} \left(\pi_ {K - 1} ^ {(1)}, \pi_ {K - 1} ^ {(2)}\right) \leq \mathcal {O} \left(\frac {V _ {\max } \ln A _ {\max }}{1 - \gamma} \sum_ {k = 0} ^ {K - 1} \gamma^ {K - k} (1 - \eta \tau) ^ {T _ {k} - 1} \right. \\ + \frac {V _ {\max}}{1 - \gamma} \left[ \frac {t _ {m i x} A _ {\max}}{\mu_ {\min}} \ln \left(\frac {T _ {s u m} | \mathcal {S} | A _ {\max}}{\delta \mu_ {\min}}\right) \right] ^ {2 / 3} \sum_ {k = 0} ^ {K - 1} \gamma^ {K - k - 1} \sum_ {t = 0} ^ {T _ {k} - 1} (1 - \eta \tau) ^ {T _ {k} - 2 - t} \left(\frac {1}{N _ {k , t} ^ {2 / 3}} + \frac {V _ {\max}}{\tau \overline {{N}} _ {k , t + 1} ^ {2 / 3}}\right) \\ + \frac {V _ {\max} ^ {3} \gamma^ {K}}{\tau^ {2} (1 - \gamma) ^ {2}} + \frac {t _ {m i x} V _ {\max} ^ {3}}{\tau^ {2} \mu_ {\min } (1 - \gamma) ^ {3}} \ln \left(\frac {K | \mathcal {S} |}{\delta \mu_ {\min }}\right) \sum_ {k = 0} ^ {K - 1} \frac {\gamma^ {K - k - 1}}{N _ {k + 1}}. \tag {20} \\ \end{array}
$$

Remark 2. In the proof of Theorem 1, we also prove that the convergence rate of the  $Q$ -function estimation error  $\| Q_K - Q_*^{(\tau)}\|_{\infty}$  is  $(1 - \gamma)$  times the convergence rate in (20). Here,  $Q_K$  corresponds to the  $Q$ -function associated with the policy pair  $(\pi_K^{(1)},\pi_K^{(2)})$  produced by Algorithm 1 in the  $K$ -th iteration, and  $Q_*^{(\tau)}$  corresponds to the optimal  $Q$ -function associated with the Nash equilibrium policy pair  $\pi_{*\tau}^{(1)},\pi_{*\tau}^{(2)}$  of the entropy-regularized Markov game.

Theorem 1 characterizes the convergence of duality gap of the SPE algorithm under general hyperparameter scheduling of the batch sizes  $N_{k,t}, \overline{N}_{k,t}, N_{k+1}$  and number of inner iterations  $T_k$ . Specifically, it can be seen that as the number of outer iterations  $K$  and inner iterations  $T_k$  increase, the duality gap converges to an exponentially weighted average of  $N_{k,t}^{-2/3}, \overline{N}_{k,t+1}^{-2/3}$  and  $N_{k+1}^{-1}$ , and the gap can be made arbitrarily small by choosing sufficiently large batch sizes. Moreover, we show in Theorem 2 later that the above tight convergence rate leads to a substantially improved sample complexity over the state-of-the-art result. In particular, due to the exponentially weighted average structure in (20), the sample complexity is optimized by an adaptive scheduling of the batch sizes.

We further comment on the technical proof of Theorem 1. Note that the analysis of the PE algorithm in (Cen et al., 2021) requires full knowledge of the environment and does not characterize the convergence of duality gap. As a comparison, to establish the duality gap convergence rate (20) of the model-free SPE, we need to make substantial new developments to tightly bound the estimation errors of the proposed stochastic estimators. We elaborate our technical contributions below.

- As we explained in Remark 1, our proposed stochastic estimators in (12), (14) and (15) are more accurate than those used in (Wei et al., 2021). Moreover, note that the sample averages involved in these estimators are over Markovian samples, and we need to apply the concentration inequalities developed in (Paulin, 2015) for dependent samples to establish high-probability estimation error bounds. As a comparison, Wei et al. (2021) assumes independence of these samples (although their setting is Markovian sampling).  
- We develop a much refined analysis of the state value function estimation error  $\| \widehat{V}_{k + 1} - V_{k + 1}'\|_{\infty}$ , which is the key to develop tight bounds for all the other estimation errors. Specifically, we first propose the following error decomposition for any state  $s$

$$
\left| \widehat {V} _ {k + 1} (s) - V _ {k + 1} ^ {\prime} (s) \right| = \left| \frac {\widehat {v} _ {k + 1} (s)}{\widehat {\mu} _ {k} (s)} - \frac {v _ {k + 1} (s)}{\mu_ {k} (s)} \right| \leq \frac {\left| \widehat {v} _ {k + 1} (s) - v _ {k + 1} (s) \right|}{\mu_ {k} (s)} + \left| \widehat {v} _ {k + 1} (s) \right| \left| \frac {\mu_ {k} (s) - \widehat {\mu} _ {k} (s)}{\mu_ {k} (s) \widehat {\mu} _ {k} (s)} \right|,
$$

where  $\widehat{v}_{k + 1}(s),\widehat{\mu}_{k + 1}(s)$  are sample average estimators of  $v_{k + 1}(s),\mu_{k + 1}(s)$ , respectively, and we refer to Appendix B for the definitions of these terms. The motivation is that the  $\left|\widehat{v}_{k + 1}(s)\right|$  in the second term helps cancel out the estimator  $\widehat{\mu}_k(s)$  in the denominator, and then all the denominators do not involve any sample average estimators, which may take a small numerical value that causes divergence and a loose concentration bound. By leveraging this special decomposition and the recursive structure of the stochastic estimator (12), we are able to establish the following key contraction property of the estimation error (see (84) in Appendix B).

$$
\left\| \widehat {V} _ {k + 1} - V _ {k + 1} ^ {\prime} \right\| _ {\infty} \leq \gamma \left\| \widehat {V} _ {k} - V _ {k} ^ {\prime} \right\| _ {\infty} + \mathcal {O} \left(N _ {k + 1} ^ {- 1 / 2}\right).
$$

By telescoping the above contraction bound, we obtain tight estimation error bounds for all the proposed stochastic estimators. As a comparison, Wei et al. (2021) directly applied the Azuma-Hoeffding inequality with independent samples to bound the entire estimator and obtain a loose error bound, and Liu et al. (2021) simply assumed a small upper bound for the estimation error.

- We develop a stochastic predictive update (SPU) algorithm with general inexact value function estimations and a finite-time convergence analysis of its duality gap (see Lemma 1 for the SPU algorithm and its convergence proof). This generalizes the convergence result of the PU algorithm established in (Cen et al., 2021), which uses exact value functions based on full knowledge of the environment. Finally, by incorporating our developed tight estimation error bounds into the finite-time duality gap bound of SPU, we obtain the desired convergence rate in Theorem 1.

Based on Theorem 1, we obtain the following sample complexity of SPE for achieving an  $\epsilon$ -Nash equilibrium duality gap of the original Markov game, i.e.,  $D^{(0)}(\pi_{K - 1}^{(1)},\pi_{K - 1}^{(2)})\leq \epsilon$ . Here, we adopt an adaptive batch size scheduling scheme to optimize the complexity order. The overall sample complexity is given by  $\sum_{k = 0}^{K - 1}\left[N_{k + 1} + 2\sum_{t = 0}^{T_k - 1}(N_{k,t} + \overline{N}_{k,t + 1})\right]$ .

Theorem 2 (Sample complexity). Implement Algorithm 1 with  $\eta = \mathcal{O}\big(1 - \gamma \big)$ ,  $\tau = \mathcal{O}\big(\frac{\epsilon(1 - \gamma)}{\ln A_{\max}}\big)$ ,  $K = \mathcal{O}\left[\frac{1}{1 - \gamma} \ln \left(\frac{\ln A_{\max}}{\epsilon(1 - \gamma)}\right)\right]$  and  $T_k = 1 + \frac{k \ln \gamma^{-1}}{\ln(1 - \eta \tau)^{-1}}$ . Choose the following adaptive batch sizes.

$$
N _ {k + 1} = \widetilde {\mathcal {O}} \Big (\frac {t _ {m i x} (\ln^ {2} A _ {\mathrm {m a x}}) \gamma^ {- \frac {k}{2}}}{\epsilon^ {2} \mu_ {\mathrm {m i n}} (1 - \gamma) ^ {8}} \Big), N _ {k, t} = \overline {{N}} _ {k, t} \epsilon^ {\frac {3}{2}} (1 - \gamma) ^ {3} = \widetilde {\mathcal {O}} \Big (\frac {t _ {m i x} A _ {\mathrm {m a x}} (1 - \eta \tau) ^ {\frac {- 3 (t + 1)}{5}}}{\mu_ {\mathrm {m i n}} (1 - \gamma) ^ {3}} \Big).
$$

Then, for any  $\epsilon \leq \frac{\ln A_{\mathrm{max}}}{1 - \gamma}$ , the overall sample complexity to achieve  $D^{(0)}(\pi_{K - 1}^{(1)},\pi_{K - 1}^{(2)})\leq \epsilon$  is  $\widetilde{\mathcal{O}}\Big(\frac{t_{\mathrm{mix}}A_{\mathrm{max}}}{\mu_{\mathrm{min}}\epsilon^{5.5}(1 - \gamma)^{13.5}}\Big)$ . Please refer to (117) in Appendix E for a complete expression.

The above complexity result is obtained by choosing a small  $\tau = \mathcal{O}\left(\frac{\epsilon(1 - \gamma)}{\ln A_{\mathrm{max}}}\right)$  for the convergence rate result in Theorem 1. Specifically, we show in Lemma 6 that the duality gap is Lipschitz continuous with regard to the entropy regularization parameter, i.e.,  $|D^{(\tau)}(\pi^{(1)},\pi^{(2)}) - D^{(0)}(\pi^{(1)},\pi^{(2)})|\leq$ $\frac{2\tau\ln A_{\mathrm{max}}}{1 - \gamma}$ . Therefore, by choosing a proper small  $\tau$ , convergence of the duality gap  $D^{(\tau)}$  of the entropy-regularized Markov game implies the convergence of the duality gap  $D^{(0)}$  of the original Markov game.

Our sample complexity is substantially lower than that of (Wei et al., 2021), which is of the order  $\widetilde{O}\left(\frac{A_{\mathrm{max}}^3|S|^{10.5}}{\epsilon^8\mu_{\mathrm{min}}(1 - \gamma)^{29.5}}\right)$ . This is due to multiple advantages in our algorithm design. First, our SPE algorithm uses entropy regularization to handle the simplex probability constraint, and this leads to a faster convergence than the Euclidean projected optimistic gradient update used in (Wei et al., 2021). Second, the optimistic gradient algorithm of (Wei et al., 2021) only updates the policy once per iteration, whereas our SPE performs multiple policy updates ( $T_k \gg 1$ ), which turns out to help improve the sample complexity. Moreover, our SPE allows us to use a large constant learning rate  $\eta = \mathcal{O}(1 - \gamma)$ , whereas the algorithm in (Wei et al., 2021) uses a substantially smaller learning rate  $\eta \leq \mathcal{O}(\sqrt{(1 - \gamma)^5|S|^{-1}})$  that significantly slows down the convergence.

# 5 CONCLUSION

In this paper, we developed a model-free, provably convergent, sample efficient, symmetric and private stochastic policy extra gradient algorithm for solving two-player zero-sum Markov games. Our algorithm leverages entropy regularization to facilitate the algorithm convergence and develops new stochastic estimators to accurately estimate the value functions. We proved that our SPE algorithm achieved a fast convergence rate in terms of the Nash equilibrium duality gap and moreover, achieves a substantially improved sample complexity over the state-of-the-art result. We believe our algorithm deepens the understanding of Markov games from a computation complexity perspective. In the future study, we are excited about exploring other topics along this direction. For example, it is interesting to extend SPE algorithm to the multi-agent setting for solving general-sum Markov games and competitive games that involve multiple cooperative teams.

# REFERENCES

Bai, Y., Jin, C., and Yu, T. (2020). Near-optimal reinforcement learning with self-play. Proc. Advances in Neural Information Processing Systems (Neurips), 33.  
Bhandari, J., Russo, D., and Singal, R. (2018). A finite time analysis of temporal difference learning with linear function approximation. In Proc. Conference on Learning Theory (COLT), volume 75, pages 1691-1692.  
Cen, S., Cheng, C., Chen, Y., Wei, Y., and Chi, Y. (2020). Fast global convergence of natural policy gradient methods with entropy regularization. *ArXiv:2007.06558*.  
Cen, S., Wei, Y., and Chi, Y. (2021). Fast policy extragradient methods for competitive games with entropy regularization. *ArXiv:2105.15186*.  
Ciosek, K. (2021). Imitation learning by reinforcement learning. ArXiv:2108.04763.  
Daskalakis, C., Foster, D. J., and Golowich, N. (2020). Independent policy gradient methods for competitive reinforcement learning. In Proc. Advances in Neural Information Processing Systems (Neurips), volume 33.  
Deng, X., Li, Y., Mguni, D. H., Wang, J., and Yang, Y. (2021). On the complexity of computing markov perfect equilibrium in general-sum stochastic games. *ArXiv:2109.01795*.  
Durrett, R. (2019). Probability: theory and examples, volume 49. Cambridge university press.  
Elie, R., Perolat, J., Laurière, M., Geist, M., and Pietquin, O. (2020). On the convergence of model free learning in mean field games. In Proc. the AAAI Conference on Artificial Intelligence (AAAI), volume 34, pages 7143-7150.  
Ernst, D., Geurts, P., and Wehenkel, L. (2005). Tree-based batch mode reinforcement learning. Journal of Machine Learning Research, 6:503-556.  
Fan, J., Wang, Z., Xie, Y., and Yang, Z. (2020). A theoretical analysis of deep q-learning. In Proc. Learning for Dynamics and Control (L4DC), pages 486-489.  
Farahmand, A.-M. and Ghavamzadeh, M. (2021). Pid accelerated value iteration algorithm. In Proc. International Conference on Machine Learning (ICML), pages 3143-3153.  
Geist, M., Scherrer, B., and Pietquin, O. (2019). A theory of regularized markov decision processes. In Proc. International Conference on Machine Learning (ICML), pages 2160-2169.  
Ghosh, M. K., Golui, S., Pal, C., and Pradhan, S. (2021). Zero-sum games for continuous-time markov decision processes with risk-sensitive average cost criterion. *ArXiv:2109.08837*.  
Greensmith, E., Bartlett, P. L., and Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. Journal of Machine Learning Research, 5(9).  
Gu, H., Guo, X., Wei, X., and Xu, R. (2021). Mean-field multi-agent reinforcement learning: A decentralized network approach. ArXiv:2108.02731.  
Guo, H., Fu, Z., Yang, Z., and Wang, Z. (2021). Decentralized single-timescale actor-critic on zero-sum two-player stochastic games. In Proc. International Conference on Machine Learning (ICML), pages 3899-3909.  
Guo, X., Hu, A., Xu, R., and Zhang, J. (2019). Learning mean-field games. Proc. Advances in Neural Information Processing Systems (Neurips), 32:4966-4976.  
Hu, J. and Wellman, M. P. (2003). Nash q-learning for general-sum stochastic games. Journal of machine learning research, 4(Nov):1039-1069.  
Huang, B., Lee, J. D., Wang, Z., and Yang, Z. (2021). Towards general function approximation in zero-sum markov games. ArXiv:2107.14702.  
Jafarnia-Jahromi, M., Jain, R., and Nayyar, A. (2021). Learning zero-sum stochastic games with posterior sampling. ArXiv:2109.03396.

Jin, Y. and Sidford, A. (2021). Towards tight bounds on the sample complexity of average-reward mdps. In Proc. International Conference on Machine Learning (ICML), pages 5055-5064.  
Kononen, V. and Oja, E. (2004). Asymmetric multiagent reinforcement learning in pricing applications. In Proc. IEEE International Joint Conference on Neural Networks (IEEE Cat. No. 04CH37541), volume 2, pages 1097-1102.  
Leonardos, S., Overman, W., Panageas, I., and Piliouras, G. (2021). Global convergence of multiagent policy gradient in markov potential games. ArXiv:2106.01969.  
Liu, B., Yang, Z., and Wang, Z. (2021). Policy optimization in zero-sum markov games: Fictitious self-play provably attains nash equilibria.  
Mei, J., Xiao, C., Szepesvari, C., and Schuurmans, D. (2020). On the global convergence rates of softmax policy gradient methods. In Proc. International Conference on Machine Learning (ICML), pages 6820-6829.  
Mertikopoulos, P. and Sandholm, W. H. (2016). Learning in games via reinforcement and regularization. Mathematics of Operations Research, 41(4):1297-1324.  
Moerland, T. M., Broekens, J., Plaat, A., and Jonker, C. M. (2018). A0c: Alpha zero in continuous action space. *ArXiv:1805.09613*.  
Neu, G., Jonsson, A., and Gomez, V. (2017). A unified view of entropy-regularized markov decision processes. *ArXiv:1705.07798*.  
Ortner, P. and Auer, R. (2007). Logarithmic online regret bounds for undiscounted reinforcement learning. In Proc. Advances in Neural Information Processing Systems (Neurips), volume 19, page 49.  
Ortner, R. (2020). Regret bounds for reinforcement learning via markov chain concentration. Journal of Artificial Intelligence Research, 67:115-128.  
Paulin, D. (2015). Concentration inequalities for markov chains by marton couplings and spectral methods. Electronic Journal of Probability, 20:1-32.  
Qiu, S., Yang, Z., Ye, J., and Wang, Z. (2019). On the finite-time convergence of actor-critic algorithm. In NeurIPS Optimization Foundations for Reinforcement Learning Workshop.  
Savas, Y., Ahmadi, M., Tanaka, T., and Topcu, U. (2019). Entropy-regularized stochastic games. In Proc. IEEE 58th Conference on Decision and Control (CDC), pages 5955-5962.  
Shapley, L. S. (1953). Stochastic games. Proceedings of the national academy of sciences, 39(10):1095-1100.  
Sidford, A., Wang, M., Yang, L., and Ye, Y. (2020). Solving discounted stochastic two-player games with near-optimal time and sample complexity. In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), pages 2992-3002.  
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. nature, 550(7676):354-359.  
Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.  
Tamar, A., Wu, Y., Thomas, G., Levine, S., and Abbeel, P. (2016). Value iteration networks. In Proc. Advances in neural information processing systems (Neurips), pages 2154-2162.  
Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P., et al. (2019). Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350-354.  
Wang, X. and Sandholm, T. (2002). Reinforcement learning to play an optimal nash equilibrium in team markov games. volume 15, pages 1603-1610.

Wei, C.-Y., Hong, Y.-T., and Lu, C.-J. (2017). Online reinforcement learning in stochastic games. In Proc. Advances in neural information processing systems (Neurips), pages 4994-5004.  
Wei, C.-Y., Lee, C.-W., Zhang, M., and Luo, H. (2021). Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games. In Proc. Conference on Learning Theory (COLT).  
Xu, T. and Liang, Y. (2021). Sample complexity bounds for two timescale value-based reinforcement learning algorithms. In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), pages 811-819.  
Xu, Y., Deng, Z., Wang, M., Xu, W., So, A. M.-C., and Cui, S. (2020). Voting-based multiagent reinforcement learning for intelligent IoT. IEEE Internet of Things Journal, 8(4):2681-2693.  
Zhang, K., Kakade, S., Basar, T., and Yang, L. (2020). Model-based multi-agent rl in zero-sum markov games with near-optimal sample complexity. In Proc. Advances in Neural Information Processing Systems (Neurips), volume 33.  
Zhang, K., Yang, Z., Liu, H., Zhang, T., and Basar, T. (2021). Finite-sample analysis for decentralized batch multi-agent reinforcement learning with networked agents. IEEE Transactions on Automatic Control.  
Zhang, X., Zhang, K., Miehling, E., and Basar, T. (2019). Non-cooperative inverse reinforcement learning. In Proc. Advances in Neural Information Processing Systems (Neurips), volume 32, pages 9487-9497.  
Zhao, Y., Tian, Y., Lee, J. D., and Du, S. S. (2021). Provably efficient policy gradient methods for two-player zero-sum markov games. ArXiv:2102.08903.  
Zhu, Y. and Zhao, D. (2020). Online minimax q network learning for two-player zero-sum markov games. IEEE Transactions on Neural Networks and Learning Systems.  
Zou, S., Xu, T., and Liang, Y. (2019). Finite-sample analysis for sarsa and q-learning with linear function approximation. ArXiv:1902.02234.
