# REGRET BOUNDS AND REINFORCEMENT LEARNING EXPLORATION OF EXP-BASED ALGORITHMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

EXP-based algorithms are often used for exploration in multi-armed bandit. We revisit the EXP3.P algorithm and establish both the lower and upper bounds of regret in the Gaussian multi-armed bandit setting, as well as a more general distribution option. The analyses do not require bounded rewards compared to classical regret assumptions. We also extend EXP4 from multi-armed bandit to reinforcement learning to incentivize exploration by multiple agents. The resulting algorithm has been tested on hard-to-explore games and it shows an improvement on exploration compared to state-of-the-art.

# 1 INTRODUCTION

Multi-armed bandit (MAB) is to maximize cumulative reward of a player throughout a bandit game by choosing different arms at each time step. It is also equivalent to minimizing the regret defined as the difference between the best rewards that can be achieved and the actual reward gained by the player. Formally, given time horizon  $T$ , in time step  $t \leq T$  the player choose one arm  $a_{t}$  among  $K$  arms, receives  $r_{a_t}^t$  among rewards  $r^t = (r_1^t, r_2^t, \ldots, r_K^t)$ , and maximizes the total reward  $\sum_{t=1}^{T} r_{a_t}^t$  or minimizes the regret. Computationally efficient and with abundant theoretical analyses are the EXP-type MAB algorithms. In EXP3.P, each arm has a trust coefficient (weight). The player samples each arm with probability being the sum of its normalized weights and a bias term, receives reward of the sampled arm and exponentially updates the weights based on the corresponding reward estimates. It achieves the regret of the order  $O(\sqrt{T})$  in a high probability sense. In EXP4, there are any number of experts. Each has a sample rule over actions and a weight. The player samples according to the weighted average of experts' sample rules and updates the weights respectively.

Contextual bandit is a variant of MAB by adding context or state space  $S$ . At time step  $t$ , the player has context  $s_t \in S$  with  $s_{1:T} = (s_1, s_2, \dots, s_T)$  being independent. Rewards  $r^t$  follow  $F(\mu(s_t))$  where  $F$  is any distribution and  $\mu(s_t)$  is the mean vector that depends on state  $s_t$ . Reinforcement Learning (RL) generalizes contextual bandit, where state and reward transitions follow a Markov Decision Process (MDP) represented by transition kernel  $P(s_{t+1}, r^t | a_t, s_t)$ . A key challenge in RL is the trade-off between exploration and exploitation. Exploration is to encourage the player to try new arms in MAB or new actions in RL to understand the game better. It helps to plan for the future, but with the sacrifice of potentially lowering the current reward. Exploitation aims to exploit currently known states and arms to maximize the current reward, but it potentially prevents the player to gain more information to increase local reward. To maximize the cumulative reward, the player needs to know the game by exploration, while guaranteeing current reward by exploitation.

How to incentivize exploration in RL has been a main focus in RL. Since RL is built on MAB, it is natural to extend MAB techniques to RL and UCB is such a success. UCB (Auer et al. (2002a)) motivates count-based exploration (Strehl and Littman, 2008) in RL and the subsequent Pseudo-Count exploration (Bellemare et al., 2016). New deep RL exploration algorithms have been recently proposed. Using deep neural networks to keep track of the  $Q$ -values by means of  $Q$ -networks in RL is called DQN (Mnih et al. (2013)). This combination of deep learning and RL has shown great success.  $\epsilon$ -greedy in Mnih et al. (2015) is a simple exploration technique using DQN. Besides  $\epsilon$ -greedy, intrinsic model exploration computes intrinsic rewards by focusing on experiences. Intrinsic rewards directly measure and incentivize exploration if added to extrinsic (actual) rewards of RL, e.g. DORA (Fox et al., 2018) and (Stadie et al., 2015). Random Network Distillation (RND) (Burda et al., 2018) is a more recent suggestion relying on a fixed target network. A drawback of RND is its local focus without global exploration.

In order to address weak points of these various exploration algorithms in the RL context, the notion of experts is natural and thus EXP-type MAB algorithms are appropriate. The allowance of arbitrary experts provides exploration for harder contextual bandits and hence providing exploration possibilities for RL. We develop an EXP4 exploration algorithm for RL that relies on several general experts. This is the first RL algorithm using several exploration experts enabling global exploration. Focusing on DQN, in the computational study we focus on two agents consisting of RND and  $\epsilon$ -greedy DQN.

We implement the RL EXP4 algorithm on the hard-to-explore RL game Montezuma's Revenge and compare it with the benchmark algorithm RND (Burda et al. (2018)). The numerical results show that the algorithm gains more exploration than RND and it gains the ability of global exploration by not getting stuck in local maximums of RND. Its total reward also increases with training. Overall, our algorithm improves exploration and exploitation on the benchmark game and demonstrates a learning process in RL.

Reward in RL in many cases is unbounded which relates to unbounded MAB rewards. There are three major versions of MAB: Adversarial, Stochastic, and herein introduced Gaussian. For adversarial MAB, rewards of the  $K$  arms  $r^t$  can be chosen arbitrarily by adversaries at step  $t$ . For stochastic MAB, the rewards at different steps are assumed to be i.i.d. and the rewards across arms are independent. It is assumed that  $0 \leq r_i^t \leq 1$  for any arm  $i$  and step  $t$ . For Gaussian MAB, rewards  $r^t$  follow multi-variate normal  $\mathcal{N}(\mu, \Sigma)$  with  $\mu$  being the mean vector and  $\Sigma$  the covariance matrix of the  $K$  arms. Here the rewards are neither bounded, nor independent among the arms. For this reason the introduced Gaussian MAB reflects the RL setting and is the subject of our MAB analyses of EXP3.P.

EXP-type algorithms (Auer et al. (2002b)) are optimal in the two classical MABs. Auer et al. (2002b) show lower and upper bounds on regret of the order  $O(\sqrt{T})$  for adversarial MAB and of the order  $O(\log(T))$  for stochastic MAB. All of the proofs of these regret bounds by EXP-type algorithms are based on the bounded reward assumption, which does not hold for Gaussian MAB. Therefore, the regret bounds for Gaussian MAB with unbounded rewards studied herein are significantly different from prior works.

We show both lower and upper bounds on regret of Gaussian MAB under certain assumptions. Some analyses even hold for more generally distributed MAB. Upper bounds borrow some ideas from the analysis of the EXP3.P algorithm in Auer et al. (2002b) for bounded MAB to our unbounded MAB, while lower bounds are by our brand new construction of instances. Precisely, we derive lower bounds of order  $T$  for certain fixed  $T$  and upper bounds of order  $O^{*}(\sqrt{T})$  for  $T$  being large enough. The question of bounds for any value of  $T$  remains open.

The main contributions of this work are as follows. On the analytical side we introduce Gaussian MAB with the unique aspect and challenge of unbounded rewards. We provide the very first regret lower bound in such a case by constructing a novel family of Gaussian bandits and we are able to analyze the EXP3.P algorithm for Gaussian MAB. Unbounded reward poses a non-trivial challenge in the analyses. We also provide the very first extension of EXP4 to RL exploration. We show its superior performance on two hard-to-explore RL games.

A literature review is provided in Section 2. Then in Section 3 we exhibit upper bounds for unbounded MAB of the EXP3.P algorithm and lower bounds, respectively. Section 4 discusses the EXP4 algorithm for RL exploration. Finally, in Section 5, we present numerical results related to the proposed algorithm.

# 2 LITERATURE REVIEW

The importance of exploration in RL is well understood. Count-based exploration in RL relies on UCB. Strehl and Littman (2008) develop Bellman value iteration  $V(s) = \max_{a} \hat{R}(s, a) + \gamma E[V(s')] + \beta N(s, a)^{-\frac{1}{2}}$ , where  $N(s, a)$  is the number of visits to  $(s, a)$  for state  $s$  and action  $a$ . Value  $N(s, a)^{-\frac{1}{2}}$  is positively correlated with curiosity of  $(s, a)$  and encourages exploration. This method is limited to tableau model-based MDP for small state spaces, while Bellemare et al. (2016) introduce Pseudo-Count exploration for non-tableau MDP with density models.

In conjunction with DQN,  $\epsilon$ -greedy in Mnih et al. (2015) is a simple exploration technique using DQN. Besides  $\epsilon$ -greedy, intrinsic model exploration computes intrinsic rewards by the accuracy of a model trained on experiences. Intrinsic rewards directly measure and incentivize exploration if

added to extrinsic (actual) rewards of RL, e.g. DORA in Fox et al. (2018) and Stadie et al. (2015). Intrinsic rewards in Stadie et al. (2015) are defined as  $e(s,a) = ||\sigma(s') - M_{\phi}(\sigma(s),a)||_2^2$  where  $M_{\phi}$  is a parametric model,  $s'$  is the next state and  $\sigma$  is input extraction. Intrinsic reward  $e(s,a)$  relies on stochastic transition from  $s$  to  $s'$  and brings noise to exploration. Random Network Distillation(RND) in Burda et al. (2018) addresses this by defining  $e(s,a) = ||\hat{f}(s') - f(s')||_2^2$  where  $\hat{f}$  is a parametric model and  $f$  is a randomly initialized but fixed model. Here  $e(s,a)$ , independent of the transition, only depends on state  $s'$  and drives RND to outperform other algorithms on Montezuma's Revenge. None of these algorithms use several experts which is a significant departure from our work.

In terms of MAB regret analyses focusing on EXP-type algorithms, Auer et al. (2002b) first introduce EXP3.P for bounded adversarial MAB and EXP4 for contextual bandits. Under the EXP3.P algorithm, an upper bound on regret of the order  $O(\sqrt{T})$  is achieved, which has no gap with the lower bound and hence it establishes that EXP3.P is optimal. However these regret bounds are not applicable to Gaussian MAB since rewards can be infinite. Meanwhile for unbounded MAB, Srinivas et al. (2010) demonstrate a regret bound of order  $O(\sqrt{T} \cdot \overline{\gamma_T})$  for noisy Gaussian process bandits where a reward observation contains noise. The information gain  $\gamma_T$  is not well-defined in a noiseless Gaussian setting. For noiseless Gaussian bandits, Grunewälder et al. (2010) show both the optimal lower and upper bounds on regret, but the regret definition is not consistent with the one used in Auer et al. (2002b). We establish a lower bound of the order  $O(T)$  for certain  $T$  and an upper bound of the order  $O^*(\sqrt{T})$  asymptotically on regret of unbounded noiseless Gaussian MAB following standard definitions of regret.

# 3 REGRET BOUNDS FOR GAUSSIAN MAB

For Gaussian MAB with time horizon  $T$ , at step  $0 < t \leq T$  rewards  $r^t$  follow multi-variate normal  $\mathcal{N}(\mu, \Sigma)$  where  $\mu = (\mu_1, \mu_2, \dots, \mu_K)$  is the mean vector and  $\Sigma = (a_{ij})_{i,j \in \{1, \dots, K\}}$  is the covariance matrix of the  $K$  arms. The player receives reward  $y_t = r_{a_t}^t$  by pulling arm  $a_t$ . We use  $R_T' = T \cdot \max_k \mu_k - \sum_t E[y_t]$  to denote pseudo regret called simply regret. (Note that the alternative definition of regret  $R_T = \max_i \sum_{t=1}^T r_i^t - \sum_{t=1}^T y_t$  depends on realizations of rewards.)

# 3.1 LOWER BOUNDS ON REGRET

In this section we derive a lower bound for Gaussian and general MAB under an assumption. General MAB replaces Gaussian with a general distribution. The main technique is to construct instances or sub-classes that have certain regret, no matter what strategies are deployed. We need the following assumption or setting.

Assumption 1 There are two types of arms with general  $K$  with one type being superior ( $S$  is the set of superior arms) and the other being inferior ( $I$  is the set of inferior arms). Let  $1 - q, q$  be the proportions of the superior and inferior arms, respectively which is known to the adversary and clearly  $0 \leq q \leq 1$ . The arms in  $S$  are indistinguishable and so are those in  $I$ . The first pull of the player has two steps. In the first step the player selects an inferior or superior set of arms based on  $P(S) = 1 - q$  and  $P(I) = q$  and once a set is selected, the corresponding reward of an arm from the selected set is received.

An interesting special case of Assumption 1 is the case of two arms and  $q = 1 / 2$ . In this case, the player has no prior knowledge and in the first pull chooses an arm uniformly at random.

The lower bound is defined as  $R_L(T) = \inf \sup R_T'$ , where, first, inf is taken among all the strategies and then sup is among all Gaussian MAB. All proofs are in the Appendix.

The following is the main result with respect to lower bounds and it is based on inferior arms being distributed as  $\mathcal{N}(0,1)$  and superior as  $\mathcal{N}(\mu,1)$  with  $\mu > 0$ .

Theorem 1. In Gaussian MAB under Assumption 1, for any  $q \geq 1/3$  we have  $R_L(T) \geq (q - \epsilon) \cdot \mu \cdot T$  where  $\mu$  has to satisfy  $G(q, \mu) < q$  with  $\epsilon$  and  $T$  determined by

$$
G (q, \mu) <   \epsilon <   q, \quad T \leq \frac {\epsilon - G (q , \mu)}{(1 - q) \cdot \int \left| e ^ {- \frac {x ^ {2}}{2}} - e ^ {- \frac {(x - \mu) ^ {2}}{2}} \right|} + 2
$$

and  $G(q,\mu) = \max \left\{\int \left|qe^{-\frac{x^2}{2}} - (1 - q)e^{-\frac{(x - \mu)^2}{2}}\right|dx,\int \left|(1 - q)e^{-\frac{x^2}{2}} - qe^{-\frac{(x - \mu)^2}{2}}\right|dx\right\}$ .

To prove Theorem 1, we construct a special subset of Gaussian MAB with equal variances and zero covariances. On these instances we find a unique way to explicitly represent any policy. This builds a connection between abstract policies and this concrete mathematical representation. Then we show that pseudo regret  $R_T'$  must be greater than certain values no matter what policies are deployed, which indicates a regret lower bound on these subset of instances.

The feasibility of the aforementioned conditions is established in the following theorem.

Theorem 2. In Gaussian MAB under Assumption 1, for any  $q \geq 1/3$ , there exist  $\mu$  and  $\epsilon, \epsilon < \mu$  such that  $R_L(T) \geq (q - \epsilon) \cdot \mu \cdot T$ .

The following result with two arms and equal probability in the first pull deals /with general probabilities. Even in the case of Gaussian MAB it is not a special case of Theorem 2 since it is stronger.

Theorem 3. For general MAB under Assumption 1 with  $K = 2$ ,  $q = 1/2$ , we have that  $R_L(T) \geq \frac{T \cdot \mu}{4}$  holds for any distributions  $f_0$  for the arms in  $I$  and  $f_1$  for the arms in  $S$  with  $\int |f_1 - f_0| > 0$  (possibly with unbounded support), for any  $\mu > 0$  and  $T$  satisfying  $T \leq \frac{1}{2 \cdot \int |f_0 - f_1|} + 1$ .

The theorem establishes that for any fixed  $\mu > 0$  there is a finite set of horizons  $T$  and instances of Gaussian MAB so that no algorithm can achieve regret smaller than linear in  $T$ . Table 1 provides the values of the relationship between  $\mu$  and largest  $T$  in the Gaussian case where the inferior arms are distributed based on the standard normal and the superior arms have mean  $\mu > 0$  and variance 1. For example, there is no way to attain regret lower than  $T \cdot 10^{-4} / 4$  for any  $1 \leq T \leq 2501$ . The function decreases very quickly.

Table 1: Upper bounds for  $T$  as a function of  $\mu$  

<table><tr><td>μ</td><td>10-5</td><td>10-4</td><td>10-3</td><td>10-2</td><td>10-1</td></tr><tr><td>Upper bound for T</td><td>25001</td><td>2501</td><td>251</td><td>26</td><td>3.5</td></tr></table>

The established lower bound result  $R_{L}(T) \geq O(T)$  is larger than known results of classical MAB. This is not surprising since the rewards in classical MAB are assumed to be bounded, while rewards in our setting follow an unbounded Gaussian distribution, which apparently increases regret.

Besides the known result  $O^{*}(\sqrt{T})$  of adversarial MAB and  $O^{*}(\log T)$  of stochastic MAB, for noisy Gaussian Process bandits, Srinivas et al. (2010) show  $R_{L}(T) \geq O(\sqrt{T \cdot \gamma_{T}})$ . Our lower bound for Gaussian MAB is different from this lower bound. The information gain term  $\gamma_{T}$  in noisy Gaussian bandits is not well-defined in Gaussian MAB and thus the two lower bounds are not comparable.

# 3.2 UPPER BOUNDS ON REGRET

In this section, we establish upper bounds for regret of Gaussian MAB by means of the EXP3.P algorithm (see Algorithm 1) from Auer et al. (2002b). We stress that rewards can be infinite, without the bounded assumption present in stochastic and adversarial MAB. We only consider non-degenerate Gaussian MAB where variance of each arm is strictly positive, i.e.  $\min_{i}a_{ii} > 0$ .

# Algorithm 1: EXP3.P

Initialization: Weights  $w_{i} = \exp \left(\frac{\alpha\delta}{3}\sqrt{\frac{T}{K}}\right), i \in \{1,2,\dots,K\}$  for  $\alpha > 0$  and  $\delta \in (0,1)$ ;

for  $t = 1,2,\dots ,T$  do

```matlab
for  $i = 1,2,\ldots ,K$  do  $p_i(t) = (1 - \delta)\frac{w_i(t)}{\sum_{j = 1}^Kw_j(t)} +\frac{\delta}{K}$  end Choose  $i_t$  randomly according to the distribution  $p_1(t),\dots,p_K(t)$  Receive reward  $r_{i_t}(t)$  . for  $j = 1,\dots ,K$  do  $\hat{x}_j(t) = \frac{r_j(t)}{p_j(t)}\cdot \mathbb{1}_{j = i_t},\quad w_j(t + 1) = w_j(t)\exp \frac{\delta}{3K} (\hat{x}_j(t) + \frac{\alpha}{p_j(t)\sqrt{KT}})$  end
```

Formally, we provide analyses for upper bounds on  $R_{T}$  with high probability, on  $E[R_{T}]$  and on  $R_{T}'$ . In Auer et al. (2002b) EXP3.P is studied to yield a bound on regret  $R_{T}$  with high probability in the bounded MAB setting. As part of our contributions, we show that EXP3.P regret is of the order  $O^{*}(\sqrt{T})$  in the unbounded Gaussian MAB in the case of  $R_{T}$  with high probability,  $E[R_{T}]$  and  $R_{T}'$ . The results are summarized as follows. The density of  $\mathcal{N}(\mu, \Sigma)$  is denoted by  $f$ .

Theorem 4. For Gaussian MAB, any time horizon  $T$ , for any  $0 < \eta < 1$ , EXP3.P has regret

$$
R _ {T} \leq 4 \Delta (\eta) \cdot \left(\sqrt {K T \log \left(\frac {K T}{\delta}\right)} + 4 \sqrt {\frac {5}{3} K T \log K} + 8 \log \left(\frac {K T}{\delta}\right)\right) w i t h p r o b a b i l i t y (1 - \delta) \cdot (1 - \eta) ^ {T}
$$

where  $\Delta (\eta)$  is determined by  $\int_{-\Delta}^{\Delta}\ldots \int_{-\Delta}^{\Delta}f(x_1,\ldots ,x_K)dx_1\ldots dx_K = 1 - \eta$

In the proof of Theorem 4, we first perform truncation of the rewards of Gaussian MAB by dividing the rewards to a bounded part and unbounded tail throughout the game. For the bounded part, we directly borrow the regret upper bound of EXP3.P in Auer et al. (2002b) and conclude with the regret upper bound of order  $O(\Delta (\eta)\sqrt{T})$ . Since a Gaussian distribution is a light-tailed distribution we can control the probability of tail shrinking which leads to the overall result.

The dependence of the bound on  $\Delta$  can be removed by considering large enough  $T$  as stated next.

Theorem 5. For Gaussian MAB, and any  $a > 2$ ,  $0 < \delta < 1$ , EXP3.P has regret

$$
R _ {T} \leq \log (1 / \delta) O ^ {*} (\sqrt {T}) \text {w i t h p r o b a b i l i t y} (1 - \delta) \cdot \left(1 - \frac {1}{T ^ {a}}\right) ^ {T}.
$$

The constant behind  $O^{*}$  depends on  $K, a, \mu$  and  $\Sigma$ .

The above theorems deal with  $R_{T}$  but the aforementioned lower bounds are with respect to pseudo regret. To complete the analysis of Gaussian MAB, it is desirable to have an upper bound on pseudo regret which is established next. It is easy to verify by the Jensen's inequality that  $R_{T}^{\prime} \leq E[R_{T}]$  and thus it suffices to obtain an upper bound on  $E[R_{T}]$ .

For adversarial and stochastic MAB, the upper bound for  $E[R_T]$  is of the same order as  $R_T$  which follows by a simple argument. For Gaussian MAB, establishing an upper bound on  $E[R_T]$  or  $R_T'$  based on  $R_T$  requires more work. We show an upper bound on  $E[R_T]$  by using select inequalities, limit theories, and Randemacher complexity. To this end, the main result reads as follows.

Theorem 6. The regret of EXP3.P in Gaussian MAB satisfies

$$
R _ {T} ^ {\prime} \leq E \left[ R _ {T} \right] \leq O ^ {*} (\sqrt {T}).
$$

All these three theorems also hold for sub-Gaussian MAB, which is defined by replacing Gaussian with sub-Gaussian. This generalization is straightforward and it is directly shown in the proof of Gaussian MAB in Appendix. Optimal upper bounds for adversarial MAB and noisy Gaussian Process bandits are of the same order as our upper bound. Auer et al. (2002b) derive an upper bound of the same order  $O(\sqrt{T})$  as the lower bound for adversarial MAB. For noisy Gaussian Process bandits, there is also no gap between its upper and lower bounds.

Our upper bound of the order  $O^{*}(\sqrt{T})$  is of the same order as the one for bounded MAB. In our case the upper bound result  $O^{*}(\sqrt{T})$  holds for large enough  $T$  which is hidden behind  $O^{*}$  while the linear lower bounds is valid only for small values of  $T$ . This illustrates the rationality of the lower bound of  $O(T)$  and the upper bound of order  $O^{*}(\sqrt{T})$ .

# 4 EXP4 ALGORITHM FOR RL

EXP4 has shown great success in contextual bandits. Therefore, in this section, we extend EXP4 to RL and develop EXP4-RL illustrated in Algorithm 2.

The player has experts that are represented by deep  $Q$ -networks trained by RL algorithms (there is a one to one correspondence between the experts and  $Q$ -networks). Each expert also has a trust coefficient. Trust coefficients are also updated exponentially based on the reward estimates as in EXP4. At each step of one episode, the player samples an expert ( $Q$ -network) with probability that is proportional to the weighted average of expert's trust coefficients. Then  $\epsilon$ -greedy DQN is applied on the chosen  $Q$ -network. Here different from EXP4, the player needs to store all the interaction tuples

in experience buffer since RL is a MDP. After one episode, the player trains all  $Q$ -networks with the experience buffer and uses the trained networks as experts for the next episode.

Algorithm 2: EXP4-RL  
Initialization: Trust coefficients  $w_{k} = 1$  for any  $k\in \{1,\dots ,E\}$ $E =$  number of experts   
(Q-networks),  $K =$  number of actions,  $\Delta ,\epsilon ,\eta >0$  and temperature  $z,\tau >0,n_r = -\infty$  (an   
upper bound on reward);   
while True do Initialize episode by setting  $s_0$  .   
for  $i = 1,2,\ldots ,T$  (length of episode) do Observe state  $s_i$  . Let probability of  $Q_{k}$  network be  $\rho_{k} = (1 - \eta)\frac{w_{k}}{\sum_{k = 1}^{E}w_{k}} +\frac{\eta}{E};$  Sample network  $\bar{k}$  according to  $\{\rho_k\} _k$  For  $Q_{\bar{k}}$  network, use  $\epsilon$  greedy to sample an action  $a^* = \arg \max_a Q_{\bar{k}}(s_i,a),\qquad \pi_j = (1 - \epsilon)\cdot \mathbb{1}_{j = a^*} + \frac{\epsilon}{K - 1}\cdot \mathbb{1}_{j\neq a^*}\qquad j\in \{1,2,\ldots ,K\}$  Sample action  $a_i$  based on  $\pi$  Interact with the environment to receive reward  $r_i$  and next state  $s_{i + 1}$ $n_r = \max \{r_i,n_r\}$  Update the trust coefficient  $w_{k}$  of each  $Q_{k}$  network as follows:.  $P_{k} = \epsilon$  greedy  $(Q_{k}),\hat{x}_{kj} = 1 - \frac{\mathbb{1}_{j = a}}{P_{kj} + \Delta} (n_{r} - r_{i}),j\in 1,2,\dots ,K,y_{k} = E[\hat{x}_{kj}],w_{k} = w_{k}\cdot e^{\frac{y_{k}}{z}}$  Store  $(s_i,a_i,r_i,s_{i + 1})$  in experience replay buffer B; end Update each expert's  $Q_{k}$  network from buffer B;

The basic idea is the same as EXP4 by using the experts that give advice vectors with deep  $Q$ -networks. It is a combination of deep neural networks with EXP4 updates. From a different perspective, we can also view it as an ensemble in classification (Xia et al. (2011)), by treating  $Q$ -networks as ensembles in RL, instead of classification algorithms. While  $Q$ -networks do not necessarily have to be experts, i.e., other experts can be used, these are natural in a DQN framework.

In our implementation and experiments we use two experts, thus  $E = 2$  with two  $Q$ -networks. The first one is based on RND (Burda et al. (2018)) while the second one is a simple DQN. To this end, in the algorithm before storing to the buffer, we also record  $c_r^i = ||\hat{f}(s_i) - f(s_i)||^2$ , the RND intrinsic reward as in Burda et al. (2018). This value is then added to the 4-tuple pushed to  $B$ . When updating  $Q_1$  corresponding to RND at the end of an iteration in the algorithm, by using  $r_j + c_r^j$  we modify the  $Q_1$ -network and by using  $c_r^j$  an update to  $\hat{f}$  is executed. Network  $Q_2$  pertaining to  $\epsilon$ -greedy is updated directly by using  $r_j$ .

Intuitively, Algorithm 2 circumvents this drawback with the total exploration guided by two experts with EXP4 updated trust coefficients. When the RND expert drives high exploration, its trust coefficient leads to a high total exploration. When it has low exploration, the second expert DQN should have a high one and it incentivizes the total exploration accordingly. Trust coefficients are updated by reward estimates iteratively as in EXP4, so they keep track of the long-term performance of experts and then guide the total exploration globally. These dynamics of EXP4 combined with intrinsic rewards guarantees global exploration. The experimental results exhibited in the next section verify this intuition regarding exploration behind Algorithm 2.

We point out that potentially more general RL algorithms based on  $Q$ -factors can be used, e.g., boostedapped DQN (Osband et al. (2016)), random prioritized DQN (Osband et al. (2018)) or adaptive  $\epsilon$ -greedy VDBE (Tokic (2010)) are a possibility. Furthermore, experts in EXP4 can even be policy networks trained by PPO (Schulman et al. (2017)) instead of DQN for exploration. These possibilities demonstrate the flexibility of the EXP4-RL algorithm.

# 5 COMPUTATIONAL STUDY

As a numerical demonstration of the superior performance and exploration incentive of Algorithm 2, we show the improvements on baselines on two hard-to-exlore RL games, Mountain Car and Montezuma's Revenge. More precisely, we present that the real reward on Mountain Car improves significantly by Algorithm 2 in Section 5.1. Then we implement Algorithm 2 on Montezuma's Revenge and show the growing and remarkable improvement of exploration in Section 5.2.

Intrinsic reward  $c_{r}^{i} = ||\hat{f}(s_{i}) - f(s_{i})||^{2}$  given by intrinsic model  $\hat{f}$  represents the exploration of RND in Burda et al. (2018) as introduced in Sections 2 and 4. We use the same criterion for evaluating exploration performance of our algorithm and RND herein. RND incentivizes local exploration with the single step intrinsic reward but with the absence of global exploration.

# 5.1 MOUNTAIN CAR

In this part, we summarize the experimental results of Algorithm 2 on Mountain Car, a classical control RL game. This game has very sparse positive rewards, which brings the necessity and hardness of exploration. Blog post (Rivlin (2019)) shows that RND based on DQN improves the performance of traditional DQN, since RND has intrinsic reward to incentivize exploration. We use RND on DQN from Rivlin (2019) as the baseline and show the real reward improvement of Algorithm 2, which supports the intuition and superiority of the algorithm.

The comparison between Algorithm 2 and RND is presented in Figure 1. Here the x-axis is the epoch number and the y-axis is the cumulative reward of that epoch. Figure 1a shows the raw data comparison between EXP4-RL and RND. We observe that though at first RND has several spikes exceeding those of EXP4-RL, EXP4-RL has much higher rewards than RND after 300 epochs. Overall, the relative difference of areas under the curve (AUC) is  $4.9\%$  for EXP4-RL over RND, which indicates the significant improvement of our algorithm. This improvement is better illustrated in Figure 1b with the smoothed reward values. Here there is a notable difference between EXP4-RL and RND. Note that the maximum reward hit by EXP4-RL is  $-86$  and the one by RND is  $-118$ , which additionally demonstrates our improvement on RND.

![](images/edcd13d69eb364c383c166058df7739aea898ee315fd3f527f5e5176bce6695d.jpg)  
(a) original

![](images/b42987059ce8b3d776487c5a7cec9bad0f48c5f9ae1ddfa8054f15d01d16f3fb.jpg)  
Figure 1: The performance of Algorithm 2 and RND measured by the epoch-wise reward on Mountain Car, with the left one being the original data and the right being the smoothed reward values.  
(b) smooth

We conclude that Algorithm 2 performs better than the RND baseline and that the improvement increases at the later training stage. Exploration brought by Algorithm 2 gains real reward on this hard-to-exlore Mountain Car, compared to the RND counterpart (without the DQN expert). The power of our algorithm can be enhanced by adopting more complex experts, not limited to only DQN.

# 5.2 MONTEZUMA'S REVENGE AND PURE EXPLORATION SETTING

In this section, we show the experimental details of Algorithm 2 on Montezuma's Revenge, another notoriously hard-to-exlore RL game. The benchmark on Montezuma's Revenge is RND based on DQN which achieves a reward of zero in our environment (the PPO algorithm reported in Burda et al. (2018) has reward 8,000 with many more computing resources; we ran the PPO-based RND with 10 parallel environments and 800 epochs to observe that the reward is also 0), which indicates that DQN has room for improvement regarding exploration.

To this end, we first implement the DQN-version RND (called simply RND hereafter) on Montezuma's Revenge as our benchmark by replacing the PPO with DQN. Then we implement Algorithm 2 with two experts as aforementioned. Our computing environment allows at most 10 parallel environments. In subsequent figures the x-axis always corresponds to the number of epochs. RND update probability is the proportion of experience that are used for training the intrinsic model  $\hat{f}$  (Burda et al., 2018).

A comparison between Algorithm 2 (EXP4-RL) and RND without parallel environments (the update probability is  $100\%$  since it is a single environment) is shown in Figure 2 with the emphasis on exploration by means of the intrinsic reward. We use 3 different numbers of burn-in periods (58, 68, 167 burn-in epochs) to remove the initial training steps, which is common in Gibbs sampling. Overall EXP4-RL outperforms RND with many significant spikes in the intrinsic rewards. The larger the number of burn-in periods is, the more significant is the dominance of EXP4-RL over RND. EXP4-RL has much higher exploration than RND at some epochs and stays close to RND at other epochs. At some epochs, EXP4-RL even has 6 times higher exploration. The relative difference in the areas under the curves are  $6.9\%$ ,  $17.0\%$ ,  $146.0\%$ , respectively, which quantifies the much better performance of EXP4-RL.

![](images/51a74190d6abea76b7340837e4549faecde553436f703178366d576418fbf7a4.jpg)  
(a) small

![](images/616204a5aa8b21e6a68a31d338b447c02a73ae7186e3ce9bea4708865406464e.jpg)  
(b) medium

![](images/1718357e0c1ed64b58948efe843d4abbd2e78fc3aec2dec682bdcbfdb357d547.jpg)  
(c) large

![](images/eceb7a8c28e84162b9de43b8b1664ad91e64b4445f314967a98f605b5118c144.jpg)  
(a)  $Q$  -network losses with 0.25 update  
(b) Intrinsic reward after smoothing with 0.25 update

![](images/e7e06de3d0c361d9f90c96570502b1ceba93d60615ab5462c8857d10113ae3f5.jpg)  
Figure 2: The performance of Algorithm 2 and RND measured by intrinsic reward without parallel environments with three different burn-in periods

![](images/b839872604e4f7133befa0e80d7b54b5441c3be8fa6773ca210c35995194b47b.jpg)  
Figure 3: The performance of Algorithm 2 and RND with 10 parallel environments and with RND update probability 0.25 and 0.125, measured by loss and intrinsic reward.  
(c) Intrinsic reward after smoothing with 0.125 update

We next compare EXP4-RL and RND with 10 parallel environments and different RND update probabilities in Figure 3. The experiences are generated by the 10 parallel environments.

Figure 3a shows that both experts in EXP4-RL are learning with decreasing losses of their  $Q$ -networks. The drop is steeper for the RND expert but it starts with a higher loss. With RND update probability 0.25 in Figure 3b we observe that EXP4-RL and RND are very close when RND exhibits high exploration. When RND is at its local minima, EXP4-RL outperforms it. Usually these local minima are driven by sticking to local maxima and then training the model intensively at local maxima, typical of the RND local exploration behavior. EXP4-RL improves on RND as training progresses, e.g. the improvement after 550 epochs is higher than the one between epochs 250 and 550. In terms for AUC, this is expressed by  $1.6\%$  and  $3.5\%$ , respectively. Overall, EXP4-RL improves RND local minima of exploration, keeps high exploration of RND and induces a smoother global exploration.

With the update probability of 0.125 in Figure 3c, EXP4-RL almost always outperforms RND with a notable difference. The improvement also increases with epochs and is dramatically larger at RND's local minima. These local minima appear more frequently in training of RND, so our improvement is more significant as well as crucial. The relative AUC improvement is  $49.4\%$ . The excellent performance in Figure 3c additionally shows that EXP4-RL improves RND with global exploration by improving local minima of RND or not staying at local maxima.

Overall, with either 0.25 or 0.125, EXP4-RL incentivizes global exploration on RND by not getting stuck in local exploration maxima and outperforms RND exploration aggressively. With 0.125 the improvement with respect to RND is more significant and steady. These experimental evidence verifies our intuition behind EXP4-RL and provides excellent support for it. With experts being more advanced RL exploration algorithms, e.g. DORA, EXP4-RL can bring additional possibilities.

# REFERENCES

P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3):235-256, 2002a.  
P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48-77, 2002b.  
M. F. Balcan. 8803 machine learning theory. http://cs.cmu.edu/~ninamf/ML11/lect1117.pdf, 2011.  
M. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos. Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information Processing Systems, pages 1471-1479, 2016.  
Y. Burda, H. Edwards, A. Storkey, and O. Klimov. Exploration by random network distillation. In International Conference on Learning Representations, 2018.  
S. Chatterjee. Superconcentration and related topics, volume 15. Cham: Springer, 2014.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: a large-scale hierarchical image database. In 2009 IEEE conference on Computer Vision and Pattern Recognition, pages 248-255. IEEE, 2009.  
L. Devroye, A. Mehrabian, and T. Reddad. The total variation distance between high-dimensional gaussians. arXiv preprint arXiv:1810.08693, 2018.  
J. Duchi. Probability bounds. http://ai.stanford.edu/~jduchi/projects/probability_bounds.pdf, 2009.  
L. Fox, L. Choshen, and Y. Loewenstein. Dora the explorer: directed outreach reinforcing action-selection. In International Conference on Learning Representations, 2018.  
S. Grünewäder, J. Y. Audibert, M. Opper, and J. Shawe-Taylor. Regret bounds for gaussian process bandit problems. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pages 273-280, 2010.  
A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, pages 1097-1105, 2012.  
V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.  
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Ried-miller, A. K. Fidjeland, G. Ostrovski, and S. Petersen. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
I. Osband, C. Blundell, A. Pritzel, and B. Van Roy. Deep exploration via bootstrapped dqn. In Advances in Neural Information Processing Systems, pages 4026-4034, 2016.  
I. Osband, J. Aslanides, and A. Cassirer. Randomized prior functions for deep reinforcement learning. In Advances in Neural Information Processing Systems, pages 8617-8629, 2018.  
O. Rivlin. Mountaincar_dqn_rnd. https://github.com/orrivlin/MountainCar_DQN_RND, 2019.  
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
N. Srinivas, A. Krause, S. Kakade, and M. Seeger. Gaussian process optimization in the bandit setting: no regret and experimental design. In Proceedings of the 27th International Conference on Machine Learning, 2010.  
B. C. Stadie, S. Levine, and P. Abbeel. Incentivizing exploration in reinforcement learning with deep predictive models. arXiv preprint arXiv:1507.00814, 2015.

A. L. Strehl and M. L. Littman. An analysis of model-based interval estimation for markov decision processes. Journal of Computer and System Sciences, 74(8):1309-1331, 2008.  
M. Tokic. Adaptive  $\varepsilon$ -greedy exploration in reinforcement learning based on value differences. In Annual Conference on Artificial Intelligence, pages 203-210. Springer, 2010.  
R. Xia, C. Zong, and S. Li. Ensemble of feature sets and classification algorithms for sentiment classification. Information Sciences, 181(6):1138-1152, 2011.
