# ONLINE RESTLESS BANDITS WITH UNOBSERVED STATES

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the online restless bandit problem, where each arm evolves according to a Markov chain independently, and the reward of pulling an arm depends on both the current state of the corresponding Markov chain and the action. The agent (decision maker) does not know the transition kernels and reward functions, and cannot observe the states of arms all the time. The goal is to sequentially choose which arms to pull so as to maximize the expected cumulative rewards collected. In this paper, we propose TSEETC, a learning algorithm based on Thompson Sampling with Episodic Explore-Then-Commit. The algorithm proceeds in episodes of increasing length and each episode is divided into exploration and exploitation phases. In the exploration phase in each episode, action-reward samples are collected in a round-robin way and then used to update the posterior as a mixture of Dirichlet distributions. At the beginning of the exploitation phase, TSEETC generates a sample from the posterior distribution as true parameters. It then follows the optimal policy for the sampled model for the rest of the episode. We establish the Bayesian regret bound  $\tilde{\mathcal{O}} (\sqrt{T})$  for TSEETC, where  $T$  is the time horizon. This is the first bound that is close to the lower bound of restless bandits, especially in an unobserved state setting. We show through simulations that TSEETC outperforms existing algorithms in regret.

# 1 INTRODUCTION

The restless multi-armed problem (RMAB) is a general setup to model many sequential decision making problems ranging from wireless communication (Tekin & Liu, 2011; Sheng et al., 2014), sensor/machine maintenance (Ahmad et al., 2009; Akbarzadeh & Mahajan, 2021) and healthcare (Mate et al., 2020; 2021). This problem considers one agent and  $N$  arms. Each arm  $i$  is modulated by a Markov chain  $M^i$  with state transition function  $P^i$  and reward function  $R^i$ . At each time, the agent decides which arm to pull. After the pulling, all arms undergo an action-dependent Markovian state transition. The goal is to decide which arm to pull to maximize the expected reward, i.e.,  $\mathbb{E}[\sum_{t=1}^{T} r_t]$ , where  $r_t$  is the reward at time  $t$  and  $T$  is the time horizon.

In this paper, we consider the online restless bandit problem with unknown parameters (transition functions and reward functions) and unobserved states. Many works concentrate on learning unknown parameters (Liu et al., 2010; 2011; Ortner et al., 2012; Wang et al., 2020; Xiong et al., 2022a,b) while ignoring the possibility that the states are also unknown. The unobserved states assumption is common in real-world applications, such as cache access (Paria & Sinha, 2021) and recommendation system (Peng et al., 2020). In the cache access problem, the user can only get the perceived delay but cannot know whether the requested content is stored in the cache before or after the access. Moreover, in the recommender system, we do not know the user's preference for the items. There are also some studies that consider the unobserved states. However, they often assume the parameters are known (Mate et al., 2020; Meshram et al., 2018; Akbarzadeh & Mahajan, 2021) and there is a lack of theoretical result (Peng et al., 2020; Hu et al., 2020). What is worse, the existing algorithms (Zhou et al., 2021; Jahromi et al., 2022) with theoretical guarantee do not match the lower regret bound of RMAB (Ortner et al., 2012).

One common way to handle the unknown parameters but with observed states is to use the optimism in the face of uncertainty (OFU) principle (Liu et al., 2010; 2011; Ortner et al., 2012; Wang et al., 2020; Xiong et al., 2022a). These methods sometimes do not perform close to the optimal, because

of the baseline they consider, such as pulling the fixed arms (Liu et al., 2010; 2011), which is not optimal in RMAB problem. Ortner et al. (2012) derives the lower bound  $\tilde{\mathcal{O}} (\sqrt{T})$  for RMAB problem. However, it is not clear whether there is an efficient computational method to find the optimistic model in the confidence region (Lakshmanan et al., 2015). Another way to estimate the unknown parameters is Thompson Sampling (TS) method (Jung & Tewari, 2019; Jung et al., 2019; Jahromi et al., 2022; Hong et al., 2022). TS algorithm is more computationally efficient since it only needs to solve the sampled instance, without the need to solve all instances that lie within the confident sets as OFU-based algorithms (Ouyang et al., 2017). What's more, empirical studies suggest that TS algorithms outperform OFU-based algorithms in bandit and Markov decision process (MDP) problems (Scott, 2010; Chapelle & Li, 2011; Osband & Van Roy, 2017).

Some studies assume that only the states of pulled arms are observable (Mate et al., 2020; Liu & Zhao, 2010; Wang et al., 2020; Jung & Tewari, 2019). They translate the partially observable Markov decision process (POMDP) problem into a fully observable MDP by regarding the state last observed and the time elapsed as a meta-state (Mate et al., 2020; Jung & Tewari, 2019), which is much simpler due to more observations about pulled arms. Mate et al. (2020), and Liu & Zhao (2010) derive the optimal index policy but they assume the known parameters. Restless-UCB in Wang et al. (2020) achieves with no guarantee of an  $\tilde{\mathcal{O}} (\sqrt{T})$  regret, and also restricted to a specific Markov model. There are also some works that consider that the arm's state is not visible even after pulling (Meshram et al., 2018; Akbarzadeh & Mahajan, 2021; Peng et al., 2020; Hu et al., 2020; Zhou et al., 2021) and the classic POMDP setting (Jahromi et al., 2022). However, there are still some challenges unresolved. Firstly, Meshram et al. (2018) and Akbarzadeh & Mahajan (2021) study the RMAB problem with unobserved states but with known parameters. Since the true value of the parameters are unavailable in practice, their contribution is limited. Secondly, the works study RMAB from a learning perspective, e.g., Peng et al. (2020); Hu et al. (2020) but there are no regret analysis. Thirdly, existing policies with regret bound  $\tilde{\mathcal{O}} (T^{2 / 3})$  (Zhou et al., 2021; Jahromi et al., 2022) often do not have a regret guarantee that scales as  $\tilde{\mathcal{O}} (\sqrt{T})$ , which is the lower bound in RMAB problem (Ortner et al., 2012). To the best of our knowledge, there are no provably optimal policies that perform close to the offline optimum and match the lower bound in restless bandit, especially in unobserved states setting.

In this paper, we address the above challenges for RMAB problems with unknown parameters and unobserved states. We design a learning algorithm TSEETC to estimate these unknown parameters, and benchmarked on a stronger oracle, we show that our algorithm achieves a tighter regret bound. In summary, we make the following contributions:

Problem formulation. We consider the online restless bandit problems with unobserved states and unknown parameters. Compared with Jahromi et al. (2022), our reward functions are unknown.

Algorithmic design. We propose TSEETC, a learning algorithm based on Thompson Sampling with Episodic Explore-Then-Commit. The whole learning horizon is divided into episodes of increasing length. Each episode is split into exploration and exploitation phases. In the exploration phase, to estimate the unknown parameters, we propose the first programmable and low-complexity algorithm to update the posterior distribution as a mixture of Dirichlet distributions. For the unobserved states, we use the belief state to encode the historical information. In the exploitation phases, we sample the parameters from the posterior distribution and derive an optimal policy based on the sampled parameter. What's more, we design the determined episode length in an increasing manner to control the total episode number, which is crucial to bound the regret caused by exploration.

Regret analysis. We consider a stronger oracle which solves POMDP based on our belief state. Under a Bayesian framework, we show that the expected regret of TSEETC accumulated up to time  $T$  is bounded by  $\tilde{\mathcal{O}} (\sqrt{T})$ , where  $\tilde{\mathcal{O}}$  hides logarithmic factors. This bound improves the exiting results (Zhou et al., 2021; Jahromi et al., 2022) and matches the theoretical lower bound for the restless bandit problem, especially in unobserved states setting.

Experiment results. We conduct the proof-of-concept experiments, and compare our policy with existing baseline algorithms. Our results show that outperforms existing algorithms and achieve a near-optimal regret bound.

# 2 RELATED WORK

We review the related works in two main domains: learning algorithm for unknown parameters, and methods to unknown states.

Unknown parameters. Since the system parameters are unknown in advance, it is essential to study RMAB problems from a learning perspective. Generally speaking, these works can be divided into two categories: OFU (Ortner et al., 2012; Wang et al., 2020; Xiong et al., 2022a; Zhou et al., 2021; Xiong et al., 2022b) or TS based (Jung et al., 2019; Jung & Tewari, 2019; Jahromi et al., 2022; Hong et al., 2022). The algorithms based on OFU often construct confidence sets for the system parameters at each time, find the optimistic estimator that is associated with the maximum reward, and then select an action based on the optimistic estimator. However, these methods suffer from some issues. Firstly, it may not perform close to the offline optimum because the baseline policy they consider, such as pulling only one arm, is often a heuristic policy and not optimal. In this case, the regret bound  $\mathcal{O}(\log T)$  (Liu et al., 2010) is less meaningful. Secondly, the state of the art algorithm of regret bound  $\tilde{\mathcal{O}} (\sqrt{T})$  are often computationally expensive. For example, the colored-UCRL2 algorithm (Ortner et al., 2012) suffers from high computational complexity because it should find the optimistic estimator in the confidence region. Apart from these works, posterior sampling (Jung & Tewari, 2019; Jung et al., 2019) were used to solve this problem. A TS algorithm generally samples a set of MDP parameters randomly from the posterior distribution, then actions are selected based on the sampled model. Jung & Tewari (2019) and Jung et al. (2019) provide theoretical guarantee  $\tilde{\mathcal{O}} (\sqrt{T})$  in the Bayesian setting. TS algorithms are confirmed to outperform optimistic algorithms in bandit and MDP problems (Scott, 2010; Chapelle & Li, 2011; Osband & Van Roy, 2017).

Unknown states. There are some works that consider the states of the pulled arm are observed (Mate et al., 2020; Liu & Zhao, 2010; Wang et al., 2020; Jung & Tewari, 2019). Mate et al. (2020) and Liu & Zhao (2010) assumes the unobserved states but with known parameters. Wang et al. (2020) constructs an offline instance and give the regret bound  $\tilde{\mathcal{O}}(T^{2/3})$ . Jung & Tewari (2019) considers the episodic RMAB problems and the regret bound  $\tilde{\mathcal{O}}(\sqrt{T})$  is guaranteed in the Bayesian setting. Some studies assume that the states are unobserved even after pulling. Akbarzadeh & Mahajan (2021) and Meshram et al. (2018) consider the RMAB problem with unknown states but known system parameters. And there is no regret guarantee. Peng et al. (2020) and Hu et al. (2020) consider the unknown parameters but there are also no any theoretical results. The most similar to our work is Zhou et al. (2021) and Jahromi et al. (2022). Zhou et al. (2021) considers that all arms are modulated by a common unobserved Markov Chain. They proposed the estimation method based on spectral method (Anandkumar et al., 2012) and learning algorithm based on upper confidence bound (UCB) strategy (Auer et al., 2002). They also give the regret bound  $\tilde{\mathcal{O}}(T^{2/3})$  and there is a gap between the lower bound  $\tilde{\mathcal{O}}(\sqrt{T})$  (Ortner et al., 2012). Jahromi et al. (2022) considers the POMDP setting and propose the pseudo counts to store the state-action pairs. Their learning algorithm is based on Ouyang et al. (2017) and the regret bound is also  $\tilde{\mathcal{O}}(T^{2/3})$ . And their algorithm is not programmable due to the pseudo counts is conditioned on the true counts which is uncountable.

# 3 PROBLEM SETTING

Consider a restless bandit problem with one agent and  $N$  arms. Each arm  $i \in [N] \coloneqq \{1, 2, \dots, N\}$  is associated with an independent discrete-time Markov chain  $\mathcal{M}^i = (\mathcal{S}^i, P^i)$ , where  $\mathcal{S}^i$  is the state space and  $P^i \in \mathbb{R}^{S^i \times S^i}$  the transition functions. Let  $s_t^i$  denote the state of arm  $i$  at time  $t$  and  $s_t = (s_t^1, s_t^2, \ldots, s_t^N)$  the state of all arms. Each arm  $i$  is also associated with a reward function  $R^i \in \mathbb{R}^{S^i \times \mathcal{R}}$ , where  $R^i(r \mid s)$  is the probability that the agent receives a reward  $r \in \mathcal{R}$  when he pulls arm  $i$  in state  $s$ . We assume the state spaces  $S^i$  and the reward set  $\mathcal{R}$  are finite and known to the agent. The parameters  $P^i$  and  $R^i$ ,  $i \in [N]$  are unknown, and the state  $s_t$  is also unobserved to the agent. For the sake of notational simplicity, we assume that all arms have the same state spaces with size  $S$ . Our result can be generalized in a straightforward way to allow different state spaces.

The whole game is divided into  $T$  time steps. The initial state  $s_1^i$  for each arm  $i \in [N]$  is drawn from a distribution  $h_i$  independently, which we assume to be known to the agent. At each time  $t$ , the agent chooses one arm  $a_t \in [N]$  to pull and receives a reward  $r_t \in \mathcal{R}$  with probability  $R^{a_t}(r_t \mid s_t^{a_t})$ . Note

that only the pulled arm has the reward feedback. His decision on which arm  $a_{t}$  to pull is based on the observed history  $\mathcal{H}_t = [a_1, r_1, a_2, r_2 \dots, a_{t-1}, r_{t-1}]$ . Note that the states of the arms are never observable, even after pulling. Each arm  $i$  makes a state transition independently according to the associated  $P^i$ , whether it is pulled or not. This process continues until the end of the game. The goal of the agent is to maximize the total expected reward.

We use  $\theta$  to denote the unknown  $P^i$  and  $R^i$  for  $i\in [N]$  collectively. Since the true states are unobservable, the agent maintains a belief state  $b_{t}^{i} = [b_{t}^{i}(s,\theta),s\in S^{i}]\in \Delta_{S^{i}}$  for each arm  $i$ , where

$$
b _ {t} ^ {i} (s, \theta) := \mathbb {P} \left(s _ {t} ^ {i} = s \mid \mathcal {H} _ {t}, \theta\right),
$$

and  $\Delta_{\mathcal{S}^i} := \left\{b \in \mathbb{R}_+^{\mathcal{S}^i} : \sum_{s \in \mathcal{S}^i} b(s) = 1\right\}$  is the probability simplex in  $\mathbb{R}^{\mathcal{S}^i}$ . Note that  $b_t^i(s, \theta)$  depends on the unknown model parameter  $\theta$ , which itself has to be learned by the agent. For a given  $\theta$ , the overall belief state  $b_t = (b_t^1, b_t^2, \dots, b_t^N)$  is a sufficient statistic for  $\mathcal{H}_{t-1}$  (Smallwood & Sondik, 1973), so the agent can base his decision at time  $t$  on  $b_t$  only. Let  $\Delta_b := \Delta_{\mathcal{S}^1} \times \dots \times \Delta_{\mathcal{S}^N}$ . A deterministic stationary policy  $\pi : \Delta_b \to [N]$  maps a belief state to an action. The long-term average reward of a policy  $\pi$  is defined as

$$
J ^ {\pi} (h, \theta) := \lim  _ {T \rightarrow \infty} \sup  _ {T \rightarrow \infty} \frac {1}{T} \mathbb {E} \left[ \sum_ {t = 1} ^ {T} r _ {t} \mid h, \theta \right]. \tag {1}
$$

We use  $J(h, \theta) = \sup_{\pi} J^{\pi}(h, \theta)$  to denote the optimal long-term average reward. We assume  $J(h, \theta)$  is independent with the initial distribution  $h$  as in Jahromi et al. (2022) and denoted it by  $J(\theta)$ . We make the following assumption.

Assumption 1. The smallest element  $\epsilon$  in the transition functions  $P_{i}, i \in N$  is bigger than zero.

Assumption 1 not only helps us bound the error of belief estimation (De Castro et al., 2017) but also makes the MDP weakly communicating (Bertsekas et al., 2011). For weakly communicating MDP, it is known that there exists a bounded function  $v(\cdot, \theta): \Delta_b \to \mathbb{R}$  such that for all  $b \in \Delta_b$  (Bertsekas et al., 2011),

$$
J (\theta) + v (b, \theta) = \max  _ {a} \left\{r (b, a) + \sum_ {r} P (r \mid b, a, \theta) v \left(b ^ {\prime}, \theta\right) \right\}, \tag {2}
$$

where  $v$  is the relative value function,  $r(b, a) = \sum_{s} \sum_{r} b^{a}(s, \theta) R^{a}(r \mid s)r$  is the expected reward,  $b'$  is the updated belief after obtaining the reward  $r$ , and  $P(r \mid b, a, \theta)$  is the probability of observing  $r$  in the next step, conditioned on the current belief  $b$  and action  $a$ . The corresponding optimal policy is the maximizer of the right part in equation 2. Note that if  $v(\cdot, \theta)$  satisfies equation 2, so does  $v(\cdot, \theta)$  plus any constant. Therefore, without loss of generality, and since  $v(\cdot, \theta)$  is bounded, we can assume that  $\inf_{b \in \Delta_{b}} v(b, \theta) = 0$  and define the span as  $\mathrm{sp}(\theta) := \sup_{b \in \Delta_{b}} v(b, \theta) \leq H$ .

We consider the Bayesian setting for the parameters. The parameters  $\theta^{*}$  is randomly generated from a known prior distribution  $Q$  at the beginning and then fixed but unknown to the agent. We measure the efficiency of a policy  $\pi$  by its regret, defined as the expected gap between the cumulative reward of an offline oracle and that of  $\pi$ , where the oracle is the optimal policy with the full knowledge of  $\theta^{*}$ , but unknown states. The offline oracle is similar to Zhou et al. (2021), which is stronger than those considered in Azizzadenesheli et al. (2016) and Fiez et al. (2018). We focus on the Bayesian regret of policy  $\pi$  (Ouyang et al., 2017; Jung & Tewari, 2019) as follows,

$$
R _ {T} := \mathbb {E} _ {\theta^ {*} \sim Q} \left[ \sum_ {t = 1} ^ {T} \left(J \left(\theta^ {*}\right) - r _ {t}\right) \right]. \tag {3}
$$

The above expectation is with respect to the prior distribution about  $\theta^{*}$ , the randomness in state transitions and the random reward.

# 4 THE TSEETC ALGORITHM

In section 4.1, we define the belief state and show how to update is with new observation. In section 4.2, we update the posterior distributions of unknown parameters as a mixture of Dirichlet distributions. In section 4.3, we propose our learning algorithm: Thompson Sampling with Episodic Explore-Then-Commit (TSEETC) algorithm.

# 4.1 BELIEF ENCODER FOR UNOBSERVED STATE

Here we focus on the belief update for arm  $i$  with known parameters. At time  $t$ , the belief for arm  $i$  in state  $s$  is  $b_{t}^{i}(s, \theta)$  where the  $\theta$  is the true parameters. Then after the pulling of arm  $i$ , we obtain the observation  $r_{t}$ . The belief  $b_{t}^{i}(s', \theta)$  can be update as follows:

$$
b _ {t + 1} ^ {i} \left(s ^ {\prime}, \theta\right) = \frac {\sum_ {s} b _ {t} ^ {i} (s , \theta) R ^ {i} \left(r _ {t} \mid s\right) P ^ {i} \left(s ^ {\prime} \mid s\right)}{\sum_ {s} b _ {t} ^ {i} (s , \theta) R ^ {i} \left(r _ {t} \mid s\right)}, \tag {4}
$$

where the  $P^i (s' \mid s)$  is the probability of transitioning from state  $s$  at time  $t$  to state  $s'$  for the next time  $t + 1$ , and  $R^i (r_t \mid s)$  is the probability of obtaining reward  $r_t$  under state  $s$ .

If the arm  $i$  is not pulled, we update its belief as follows:

$$
b _ {t + 1} ^ {i} \left(s ^ {\prime}, \theta\right) = \sum_ {s} b _ {t} ^ {i} (s, \theta) P ^ {i} \left(s ^ {\prime} \mid s\right). \tag {5}
$$

Then at each time, we can aggregate the belief of all arms as  $b_{t}$ . Based on equation 2, we can derive the optimal action  $a_{t}$  for current belief  $b_{t}$ .

# 4.2 MIXTURE OF DIRICHLET DISTRIBUTION

In this section, we derive the estimation method for unknown  $P^i$  and  $R^i$  based on the mixture of Dirichlet distribution. The Dirichlet distribution is parameterized by a count vector,  $\phi = (\phi_1, \ldots, \phi_k)$ , where  $\phi_i \geq 0$ , such that the density of probability distribution  $p = (p_1, \ldots, p_k)$  is defined as  $f(p \mid \phi) \propto \prod_{i=1}^{k} p_i^{\phi_i - 1}$  (Ghavamzadeh et al., 2015).

With the Dirichlet distribution, if the states are observed, the posterior distribution is just parameterized by the vector increased at each corresponding position (see Appendix C). However, we do not have access to the history  $\bar{s}_t^i$  due to the unobserved states assumption. Since the states that truly visited are unknown, all state sequences (and their corresponding Dirichlet posteriors) must be considered, with some weight proportional to the likelihood of each state sequence (Ross et al., 2011). Specially, for the unknown  $P^i$  and  $R^i$ , their prior distribution are parameterized by  $\phi^i$  and  $\psi^i$ , respectively. After observing the history  $\bar{r}_t^i$ ,  $\bar{s}_t^i$  the posterior distribution  $g_t(P^i)$  and  $g_t(R^i)$  at time  $t$  can be updated as in Lemma 1.

Lemma 1. Under the unobserved state setting and assuming that  $P^i$ ,  $R^i$  follow the Dirichlet prior distributions, the posterior distribution of transition  $P^i$  and reward function  $R^i$  at time  $t$  are as follows:

$$
g _ {t} \left(P ^ {i}\right) \propto \sum_ {\bar {s} _ {t} ^ {i} \in \mathcal {S} _ {i} ^ {t}} g _ {0} \left(P ^ {i}\right) w \left(\bar {s} _ {t} ^ {i}\right) \prod_ {s, s ^ {\prime}} \left(P ^ {i} \left(s ^ {\prime} \mid s\right)\right) ^ {N _ {s, s ^ {\prime}} ^ {i} \left(\bar {s} _ {t} ^ {i}\right) + \phi_ {s, s ^ {\prime}} ^ {i} - 1}, \tag {6}
$$

$$
g _ {t} \left(R ^ {i}\right) \propto \sum_ {\bar {s} _ {t} ^ {i} \in \mathcal {S} _ {i} ^ {t}} g _ {0} \left(R ^ {i}\right) w \left(\bar {s} _ {t} ^ {i}\right) \prod_ {s, r} \left(R ^ {i} (r \mid s)\right) ^ {N _ {s, r} ^ {i} \left(\bar {s} _ {t} ^ {i}\right) + \psi_ {s, r} ^ {i} - 1}. \tag {7}
$$

where  $g_0(P^i)$  and  $g_0(R^i)$  are the initial priors;  $w(\bar{s}_t^i)$  is the likelihood of state sequence  $\bar{s}_t^i$ .

This procedure is summarized in Algorithm 1.

Algorithm 1 Posterior Update for  $R^i (s,\cdot)$  and  $P^{i}(s,\cdot)$  
1: Input: the history length  $\tau_{1}$ , the state space  $S_{i}$ , the belief history  $\bar{b}_{\tau_1}^i$ ,  $\bar{r}_{\tau_1}^i$ , the initial parameters  $\phi_{s,s'}^i, \psi_{s,r}^i$  for  $s, s' \in S_i$ ,  $r \in \mathcal{R}$   
2: generate  $S_{i}^{\tau_{1}}$  possible state sequences  
3: calculate the weight  $w(j) = \prod_{t=1}^{\tau_{1}} b_{t}^{i}(s, \theta)$ ,  $j \in S_{i}^{\tau_{1}}$   
4: for  $j$  in  $1, \ldots, S_{i}^{\tau_{1}}$  do  
5: count the occurrence times of event  $(s, s')$  and  $(s, r)$  as  $N_{s,s'}^i, N_{s,r}^i$  in sequence  $j$   
6: update  $\phi_{s,s'}^i \gets \phi_{s,s'}^i + N_{s,s'}^i, \psi_{s,r}' \gets \psi_{s,r}^i + N_{s,r}^i$   
7: aggregate the  $\phi_{s,s'}^i$  as  $\phi(j), \psi_{s,r}^i$  as  $\psi(j)$  for all  $s, s' \in S_i$ ,  $r \in \mathcal{R}$   
8: end for  
9: update the mixture Dirichlet distribution  
 $g_{\tau_1}(P^i) \propto \sum_{j=1}^{S_i^{\tau_1}} w(j)f(P^i | \phi(j)), g_{\tau_1}(R^i) \propto \sum_{j=1}^{S_i^{\tau_1}} w(j)f(R^i | \psi(j))$

# 4.3 OUR ALGORITHM

Our algorithm, TSEETC, operates in episodes with different length. Each episode is split into exploration phase and exploitation phase. Denote the episode number is  $K_{T}$  and the first time in each episode is denoted as  $t_k$ . We use  $T_k$  to denote the length of episode  $k$  and it can be determined as:  $T_k = T_1 + k - 1$ , where  $T_1 = \left\lceil \frac{\sqrt{T} + 1}{2} \right\rceil$ . The length of exploration phase in each episode is fixed as  $\tau_1$  which satisfies  $\tau_1 K_T = \mathcal{O}(\sqrt{T})$  and  $\tau_1 \leq \frac{T_1 + K_T - 1}{2}$ . With these notations, our whole algorithm is shown below.

Algorithm 2 Thompson Sampling with Episodic Explore-Then-Commit  
1: Input: prior  $g_{0}(P), g_{0}(R)$ , initial belief  $b_{0}$ , exploration length  $\tau_{1}$ , the first episode length  $T_{1}$   
2: for episode  $k = 1, 2, \ldots, \mathbf{do}$   
3: start the first time of episode  $k$ ,  $t_{k} := t$   
4: generate  $R(t_{k}) \sim g_{t_{k}}(R)$  and  $P(t_{k}) \sim g_{t_{k}}(P)$   
5: for  $t = t_{k}, t_{k} + 1, \ldots, t_{k} + \tau_{1}$  do  
6: pull the arm  $i$  for  $\tau_{1}/N$  times in a round robin way  
7: receive the reward  $r_{t}$   
8: update the belief  $b_{t}^{i}$  using  $R(t_{k}), P(t_{k})$  based on equation 4  
9: update the belief  $b_{t}^{j}, j \in N \setminus \{i\}$  using  $P(t_{k})$  based on equation 5  
10: end for  
11: for  $i = 1, 2, \ldots, N$  do  
12: input the obtained  $\bar{r}_{\tau_{1}}, \bar{b}_{\tau_{1}}$  to Algorithm 1 to update the posterior distribution  $g_{t_{k} + \tau_{1}}(P)$ ,  $g_{t_{k} + \tau_{1}}(R)$   
13: end for  
14: generate  $R(t_{k} + \tau_{1}) \sim g_{t_{k} + \tau_{1}}(P), P(t_{k} + \tau_{1}) \sim g_{t_{k} + \tau_{1}}(R)$   
15: for  $i$  in  $0, 1, \ldots, N$  do  
16: re-update the belief  $b_{t}^{i}$  from time 0 to  $t_{k} + \tau_{1}$  based on  $R(t_{k} + \tau_{1})$  and  $P(t_{k} + \tau_{1})$   
17: end for  
18: compute  $\pi_{k}^{*}(\cdot) = \text{Oracle}(\cdot, R(t_{k} + \tau_{1}), P(t_{k} + \tau_{1}))$   
19: for  $t = t_{k} + \tau_{1} + 1, \dots, t_{k+1} - 1$  do  
20: apply action  $a_{t} = \pi_{k}^{*}(b_{t})$   
21: observe new reward  $r_{t+1}$   
22: update the belief  $b_{t}$  of all arms based equation 4, equation 5  
23: end for  
24: end for

In episode  $k$ , for the exploration phase, we first sampled the  $\theta_{t_k}$  from the distribution  $g_{t_k}(P)$  and  $g_{t_k}(R)$ . We pull each arm for  $\tau_1 / N$  times in a round robin way. For the pulled arm, we update its belief based on equation 4 using  $\theta_{t_k}$ . For the arms that are not pulled, we update its belief based on equation 5 using  $\theta_{t_k}$ . The reward and belief history of each arm are input into Algorithm 1 to update the posterior distribution after the exploration phase. Then we sample the new  $\theta_{t_k + \tau_1}$  from the posterior distribution, and re-calibrate the belief  $b_t$  based on the most recent estimated  $\theta_{t_k + \tau_1}$ . Next we enter into the exploitation phase. Firstly we derive the optimal policy  $\pi_k$  for the sampled parameter  $\theta_{t_k + \tau_1}$ . Then we use policy  $\pi_k$  for the rest of the episode  $k$ .

We control the increasing of episode length in a deterministic manner. Specially, the length for episode  $k$  is just one more than the last episode  $k$ . That's also to say, the first time  $t_{k+1}$  is determined by  $t_{k+1} = kT_1 + k$ . The intuitions behind such different episode lengths are in two folds. Firstly, as the Dirichlet counts grow larger and larger, the transition functions and reward functions defined by these counts do not change much. Then the sampled parameter  $\theta_k$  is more concentrated on true values. Therefore we should increase the length of exploitation phase to minimize the total regret. Secondly, in such a deterministic increasing manner, the episode number  $K_T$  is bounded by  $\mathcal{O}(\sqrt{T})$  as in Lemma 2. Then the regret caused by the exploration phases can be bound by  $\mathcal{O}(\sqrt{T})$ , which is an crucial part in Theorem 1.

Remark 1. We use an Oracle to derive the optimal policy for the sampled parameters in Algorithm 2. The Oracle can be the Bellman equation for POMDP as we introduced in equation 2, or the

approximation methods (Pineau et al., 2003; Silver & Veness, 2010), etc. The approximation error is discussed in section 5.

# 5 PERFORMANCE ANALYSIS

In section 5.1, we show our theoretical results and some discussions. In section 5.2, we provide a proof sketch and the detailed proof is in Appendix B.

# 5.1 REGRET BOUND AND DISCUSSIONS

Theorem 1. Suppose Assumption 1 holds and the Oracle returns the optimal policy. The Bayesian regret of TSEETC satisfies

$$
R _ {T} \leq 4 8 C _ {1} C _ {2} S \sqrt {A T \log (A T)} + (\tau_ {1} \Delta R + H + 4 C _ {1} C _ {2} S A) \sqrt {T} + C _ {1} C _ {2},
$$

where  $C_1 = L_1 + L_2MN + N + S^2$ ,  $C_2 = r_{\text{max}} + H$  are constants independent with time horizon  $T$ ,  $L_1 = 4M\left(\frac{1 - \epsilon}{\epsilon}\right)^2 / \min \left\{R_{\min}^*, 1 - R_{\max}^*\right\}$ ,  $L_2 = 4M(1 - \epsilon)^2 / \epsilon^3 + \sqrt{M}$ ,  $M = S^N$ ,  $R_{\text{max}}^*$  and  $R_{\text{min}}^*$  are the maximum and minimum element of the functions  $R^*$  respectively,  $\tau_1$  is the fixed exploration length in each episode,  $\Delta R$  is the biggest gap of the reward obtained at each two different time,  $H$  is the bounded span,  $r_{\text{max}}$  is the maximum reward obtain each time.

Remark 2. The Theorem 1 shows that the regret of TSEETC is upper bound by  $\tilde{\mathcal{O}} (\sqrt{T})$ . This is the first bound that matches the lower bound in restless bandit problem (Ortner et al., 2012) in such unobserved state setting. Although TSEETC looks similar to explore-then-commit (Lattimore & Szepesvári, 2020), a key novelty of TSEETC lies in using the approach of posterior sampling (Jung & Tewari, 2019; Russo & Van Roy, 2014) to balance exploration and exploitation in a deterministic-episode manner. Our algorithm ensures the episode length grows at a linear rate, which guarantees that the total episode number is bounded by  $\mathcal{O}(\sqrt{T})$ . Therefore the total regret caused by exploration is well controlled by  $\mathcal{O}(\sqrt{T})$  and this is better than the bound  $\mathcal{O}(T^{2 / 3})$  in Zhou et al. (2021). What's more, in the exploitation phase, our regret bound  $\tilde{\mathcal{O}} (\sqrt{T})$  is also better than  $\tilde{\mathcal{O}} (T^{2 / 3})$  (Zhou et al., 2021). This shows our posterior sampling based method is superior to UCB based solution (Osband & Van Roy, 2017). In Jahromi et al. (2022), their pseudo count of state-action pair is always smaller than the true counts with some probability at any time. However, in our algorithm, the sampled parameter is more concentrated on true values with the posterior update. Therefore, our pseudo count (defined in equation 13) based on belief approximates the true counts more closely, which helps us obtain a tighter bound.

Remark 3. (Approximation error.) If the oracle returns an  $\epsilon_{k}$ -approximate policy  $\tilde{\pi}_{k}$  in each episode instead of the optimal policy. That is to say,  $r(b,\tilde{\pi}_k(b)) + \sum_rP(r\mid b,\tilde{\pi}_k(b),\theta)v(b',\theta)\leq$ $\max_{a}\{r(b,a) + \sum_{r}P(r\mid b,a,\theta)v(b',\theta)\} -\epsilon_{k}$ . Then we should consider the extra regret  $\mathbb{E}\left[\sum_{k:t_k\leq T}(T_k - \tau_1)\epsilon_k\right]$  in exploitation phase. If we control the error as  $\epsilon_{k}\leq \frac{1}{T_{k} - \tau_{1}}$ , then we can bound the extra regret as  $\mathbb{E}\left[\sum_{k:t_k\leq T}(T_k - \tau_1)\epsilon_k\right]\leq k_T = \mathcal{O}(\sqrt{T})$  (Lemma 2). Thus the approximation error in the computation of optimal policy is only additive to the regret of our algorithm.

# 5.2 PROOF SKETCH

In our algorithm, the total regret can be decomposed as follows:

$$
R _ {T} = \underbrace {\mathbb {E} _ {\theta_ {*}} \left[ \sum_ {k = 1} ^ {k _ {T}} \sum_ {t _ {k}} ^ {t _ {k} + \tau_ {1}} J \left(\theta^ {*}\right) - r _ {t} \right]} _ {\text {R e g r e t (A)}} + \underbrace {\mathbb {E} _ {\theta_ {*}} \left[ \sum_ {k = 1} ^ {k _ {T}} \sum_ {t _ {k} + \tau_ {1} + 1} ^ {t _ {k + 1} - 1} J \left(\theta^ {*}\right) - r _ {t} \right]} _ {\text {R e g r e t (B)}}. \tag {8}
$$

Bounding Regret (A). The Regret (A) is the regret caused in the exploration phase of each episode.

This term can be simply bounded as follows:

$$
\operatorname {R e g r e t} (\mathrm {A}) \leq \mathbb {E} _ {\theta_ {*}} \left[ \sum_ {k = 1} ^ {k _ {T}} \tau_ {1} \Delta R \right] \leq \tau_ {1} \Delta R k _ {T} \tag {9}
$$

where  $\Delta R = r_{max} - r_{min}$  is the biggest gap of the reward received at each two different times. The regret in equation 9 is related with the episode number  $k_{T}$ , which can be bound in Lemma 2.

Lemma 2. (Bound the episode number) With the convention  $T_{1} = \left\lceil \frac{\sqrt{T} + 1}{2} \right\rceil$  and  $T_{k} = T_{k - 1} + 1$ , the episode number is bounded by  $K_{T} = \mathcal{O}(\sqrt{T})$ .

Bounding Regret (B). Next we bound  $\mathrm{Regret(B)}$  in the exploitation phase. Define  $\hat{b}_t$  is the belief updated with parameter  $\theta_{k}$  and  $b_{t}^{*}$  represents the belief with  $\theta^{*}$ . During episode  $k$ , based on equation 2 for the sampled parameter  $\theta_{k}$  and that  $a_{t} = \pi^{*}(\hat{b}_{t})$ , we can write:

$$
J \left(\theta_ {k}\right) + v \left(\hat {b} _ {t}, \theta_ {k}\right) = r \left(\hat {b} _ {t}, a _ {t}\right) + \sum_ {r} P \left(r \mid \hat {b} _ {t}, a _ {t}, \theta_ {k}\right) v \left(b ^ {\prime}, \theta_ {k}\right). \tag {10}
$$

With this equation, we proceed by decomposing the regret as:

$$
\operatorname {R e g r e t} (\mathrm {B}) = R _ {1} + R _ {2} + R _ {3} + R _ {4} \tag {11}
$$

where each term is defined as follows:

$$
R _ {1} = \mathbb {E} _ {\theta^ {*}} \sum_ {k = 1} ^ {k _ {T}} \left[ \left(T _ {k} - \tau_ {1} - 1\right) \left(J (\theta^ {*}) - J (\theta_ {k})\right) \right],
$$

$$
R _ {2} = \mathbb {E} _ {\theta^ {*}} \sum_ {k = 1} ^ {k _ {T}} \left[ \sum_ {t _ {k} + \tau_ {1} + 1} ^ {t _ {k + 1} - 1} \left(v (\hat {b} _ {t + 1}, \theta_ {k}) - v (\hat {b} _ {t}, \theta_ {k})\right) \right],
$$

$$
R _ {3} = \mathbb {E} _ {\theta^ {*}} \sum_ {k = 1} ^ {k _ {T}} \left[ \sum_ {t _ {k} + \tau_ {1} + 1} ^ {t _ {k + 1} - 1} \left(\sum_ {r} P \left[ r \mid \hat {b} _ {t}, a _ {t}, \theta_ {k} \right] v (b ^ {\prime}, \theta_ {k}) - v (\hat {b} _ {t + 1}, \theta_ {k})\right) \right],
$$

$$
R _ {4} = \mathbb {E} _ {\theta^ {*}} \sum_ {k = 1} ^ {k _ {T}} \left[ \sum_ {t _ {k} + \tau_ {1} + 1} ^ {t _ {k + 1} - 1} \left(r (\hat {b} _ {t}, a _ {t}) - r (b _ {t} ^ {*}, a _ {t})\right) \right].
$$

Bounding  $R_{1}$ . One key property of Posterior Sampling algorithms is that for given the history  $\mathcal{H}_{t_k}$ , the true parameter  $\theta^{*}$  and sampled  $\theta_{k}$  are identically distributed at the time  $t_k$ . Due to the length  $T_{k}$  determined and independent with  $\theta_{k}$ , then  $R_{1}$  is zero thanks to Lemma 3.

Lemma 3. (Posterior Sampling (Ouyang et al., 2017)). In TSEETC,  $t_k$  is an almost surely finite  $\sigma(\mathcal{H}_{t_k})$ -stopping time. If the prior distribution  $g_0(P), g_0(R)$  is the distribution of  $\theta^*$ , then for any measurable function  $g$ ,

$$
\mathbb {E} \left[ g \left(\theta^ {*}\right) \mid \mathcal {H} _ {t _ {k}} \right] = \mathbb {E} \left[ g \left(\theta_ {k}\right) \mid \mathcal {H} _ {t _ {k}} \right].
$$

Bounding  $R_{2}$ . The regret  $R_{2}$  is the telescopic sum of value function and can be bounded as  $R_{2} \leq HK_{T}$ . It solely depends on the episode number and the upper bound  $H$  of span function. As a result,  $R_{2}$  reduces to a finite bound over the number of episodes  $k_{T}$ , which can be bounded in Lemma 2.

Bounding  $R_{3}$  and  $R_{4}$ . The regret terms  $R_{3}$  and  $R_{4}$  is related with estimation error about  $\theta$ . Thus we should bound the parameters' error especially in our unobserved state setting. Note that when the states are observed, it is easy to count the state-action pairs and then the confidence interval can be also constructed such as in Wang et al. (2020); Xiong et al. (2022a). Recall the definition of  $\phi, \psi$ , we can define the empirical estimation of  $\hat{P}^i(s'|s)$  and  $\hat{R}^i(r|s)$  for arm  $i$  at time  $t$  as follows:

$$
\hat {P} ^ {i} \left(s ^ {\prime} \mid s\right) (t) = \frac {\phi_ {s , s ^ {\prime}} ^ {i} (t)}{\left\| \phi_ {s , r} ^ {i} (t) \right\| _ {1}}, \hat {R} ^ {i} (r \mid s) (t) = \frac {\psi_ {s , r} ^ {i} (t)}{\left\| \psi_ {s , r} ^ {i} (t) \right\| _ {1}} \tag {12}
$$

We also define the pesudo count of the state-action pair  $(s, a)$  before the episode  $k$  as

$$
N _ {t _ {k}} ^ {i} (s, a) = \left\| \psi_ {s, \cdot} ^ {i} \left(t _ {k}\right) \right\| _ {1} - \left\| \psi_ {s, \cdot} ^ {i} (0) \right\| _ {1} \tag {13}
$$

where  $\psi_{s,}^{i}(t_{k})$  represents the count of state-action  $z = (s,a)$  pair before the episode  $k$ . Then we define the confidence set for episode  $k$ , for all state-action pairs, the sampled  $P_{k}^{i}$  and  $R_{k}^{i}$  satisfy,

$$
\mathcal {M} _ {k} := \left\{P: \sum_ {s ^ {\prime} \in \mathcal {S}} \left| P \left(s ^ {\prime} \mid z\right) - \hat {P} _ {k} ^ {i} \left(s ^ {\prime} \mid z\right) \right| \leq \beta_ {k} (z), R: \sum_ {r \in \mathcal {R}} \left| R \left(r \mid z\right) - \hat {R} _ {k} ^ {i} \left(r \mid z\right) \right| \leq \beta_ {k} (z) \right\}
$$

where  $\beta_{k}(s,a)\coloneqq \sqrt{\frac{14S\log(2SAT_{k})}{\max\{1,N_{t_{k}}^{i}(s,a)\}}}$  is chosen conservatively (Auer et al., 2008) so that  $\mathcal{M}_k$  contains both  $M^{*}$  and  $M_{k}$  with high probability. Specially, for the unobserved state setting, the belief error under different parameters is upper bounded by the gap between the estimators as in Proposition 1. Then the core of the proofs lies in deriving a high-probability confidence set with our pesudo counts and show that the estimated error accumulated to  $T$  for each arm is bounded by  $\sqrt{T}$ . Then with the error bound for each arm, we can achieve the final regret bound for the MDP aggregated by all arms. With  $C_1 = L_1 + L_2MN + N + S^2$ , We show the final results here and the detailed proof in Appendix B.3,B.4.

Lemma 4.  $R_{3}$  satisfies the following bound

$$
R _ {3} \leq 4 8 C _ {1} S H \sqrt {A T \log A T} + + 4 C _ {1} S A H \sqrt {T} + C _ {1} H.
$$

Lemma 5.  $R_4$  satisfies the following bound

$$
R _ {4} \leq 4 8 C _ {1} S r _ {\max } \sqrt {A T \log (A T)} + 4 C _ {1} S A r _ {\max } \sqrt {T} + C _ {1} r _ {\max }.
$$

# 6 NUMERICAL EXPERIMENTS

In this section, we present proof-of-concept experiments. We consider two arms and there are two hidden states for each arm. We pull just one arm each time. The learning horizon  $T = 50000$ , and each algorithm runs 100 iterations. The transition functions and reward functions for all arms are the same. We initialize the algorithm with uninformed Dirichlet prior on the unknown parameters. We compare our algorithm with simple heuristics  $\epsilon$ -greedy (Lattimore & Szepesvári, 2020) ( $\epsilon = 0.01$ ), and Sliding-Window UCB (Garivier & Moulines, 2011) with specified window size, Q-learning (Hu et al., 2020) and SEEU (Zhou et al., 2021). The results are shown in Figure 1. We can find that TSEETC has the minimum regret among the five algorithms.

![](images/386e3b08b0be7c45ded3234ace89785c78d6bdf78e38e540d980c8a3ed474e3d.jpg)  
Figure 1: The cumulative regret

![](images/0efb46b6e2a250cde2e113408bec2513093ab1b86b4531d6e7045797bd64de48.jpg)  
Figure 2: The log-log regret

In Figure 2, we plot the cumulative regret versus  $T$  of the five algorithms in log-log scale. We observe that the slopes of all algorithms except for our TSEETC and SEEU are close to one, suggesting that they incur linear regrets. What is more, the slope of TSEETC is close to 0.5, which is better than SEEU. This is consistent with our theoretical result.

# 7 CONCLUSION

In this paper, we consider the restless bandit with unknown states and unknown dynamics. We propose the TSEETC algorithm to estimate these unknown parameters and derive the optimal policy. We also establish the Bayesian regret of our algorithm as  $\tilde{\mathcal{O}} (\sqrt{T})$  which is the first bound that matches the lower bound especially in restless bandit problems with unobserved states. Numerical results validate that the TSEETC algorithm outperforms other learning algorithms in regret. A related open question is whether our method can be applied to the setting where the transition functions are action dependent. We leave it for future research.

# REFERENCES

Sahand Haji Ali Ahmad, Mingyan Liu, Tara Javidi, Qing Zhao, and Bhaskar Krishnamachari. Optimality of myopic sensing in multichannel opportunistic access. IEEE Transactions on Information Theory, 55(9):4040-4050, 2009.  
Nima Akbarzadeh and Aditya Mahajan. Maintenance of a collection of machines under partial observability: Indexability and computation of whittle index. arXiv preprint arXiv:2104.05151, 2021.  
Animashree Anandkumar, Daniel Hsu, and Sham M Kakade. A method of moments for mixture models and hidden markov models. In Conference on Learning Theory, pp. 33-1. JMLR Workshop and Conference Proceedings, 2012.  
Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235-256, 2002.  
Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems, 21, 2008.  
Kamyar Azizzadenesheli, Alessandro Lazaric, and Animashree Anandkumar. Reinforcement learning of pomdpss using spectral methods. In Conference on Learning Theory, pp. 193-256. PMLR, 2016.  
Dimitri P Bertsekas et al. Dynamic programming and optimal control 3rd edition, volume ii. Belmont, MA: Athena Scientific, 2011.  
Olivier Chapelle and Lihong Li. An empirical evaluation of thompson sampling. Advances in neural information processing systems, 24, 2011.  
Yohann De Castro, Elisabeth Gassiat, and Sylvain Le Corff. Consistent estimation of the filtering and marginal smoothing distributions in nonparametric hidden markov models. IEEE Transactions on Information Theory, 63(8):4758-4777, 2017.  
Tanner Fiez, Shreyas Sekar, and Lillian J Ratliff. Multi-armed bandits for correlated markovian environments with smoothed reward feedback. arXiv preprint arXiv:1803.04008, 2018.  
Aurelien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In International Conference on Algorithmic Learning Theory, pp. 174-188. Springer, 2011.  
Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, Aviv Tamar, et al. Bayesian reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 8(5-6):359-483, 2015.  
Joey Hong, Branislav Kveton, Manzil Zaheer, Mohammad Ghavamzadeh, and Craig Boutelier. Thompson sampling with a mixture prior. In International Conference on Artificial Intelligence and Statistics, pp. 7565-7586. PMLR, 2022.  
Zhisheng Hu, Minghui Zhu, and Peng Liu. Adaptive cyber defense against multi-stage attacks using learning-based pomdp. ACM Transactions on Privacy and Security (TOPS), 24(1):1-25, 2020.  
Mehdi Jafarnia Jahromi, Rahul Jain, and Ashutosh Nayyar. Online learning for unknown partially observable mdps. In International Conference on Artificial Intelligence and Statistics, pp. 1712-1732. PMLR, 2022.  
Young Hun Jung and Ambuj Tewari. Regret bounds for thompson sampling in episodic restless bandit problems. Advances in Neural Information Processing Systems, 32, 2019.  
Young Hun Jung, Marc Abeille, and Ambuj Tewari. Thompson sampling in non-episodic restless bandits. arXiv preprint arXiv:1910.05654, 2019.  
Kailasam Lakshmanan, Ronald Ortner, and Daniil Ryabko. Improved regret bounds for undiscounted continuous reinforcement learning. In International Conference on Machine Learning, pp. 524-532. PMLR, 2015.

Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.  
Haoyang Liu, Keqin Liu, and Qing Zhao. Learning in a changing world: Non-bayesian restless multi-armed bandit. Technical report, CALIFORNIA UNIV DAVIS DEPT OF ELECTRICAL AND COMPUTER ENGINEERING, 2010.  
Haoyang Liu, Keqin Liu, and Qing Zhao. Logarithmic weak regret of non-bayesian restless multiarmed bandit. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1968-1971. IEEE, 2011.  
Keqin Liu and Qing Zhao. Indexability of restless bandit problems and optimality of whittle index for dynamic multichannel access. IEEE Transactions on Information Theory, 56(11):5547-5567, 2010.  
Aditya Mate, Jackson Killian, Haifeng Xu, Andrew Perrault, and Milind Tambe. Collapsing bandits and their application to public health intervention. Advances in Neural Information Processing Systems, 33:15639-15650, 2020.  
Aditya Mate, Andrew Perrault, and Milind Tambe. Risk-aware interventions in public health: Planning with restless multi-armed bandits. In 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS). London, UK, volume 10, 2021.  
Rahul Meshram, D Manjunath, and Aditya Gopalan. On the whittle index for restless multiarmed hidden markov bandits. IEEE Transactions on Automatic Control, 63(9):3046-3053, 2018.  
Ronald Ortner, Daniil Ryabko, Peter Auer, and Rémi Munos. Regret bounds for restless markov bandits. In International conference on algorithmic learning theory, pp. 214-228. Springer, 2012.  
Ian Osband and Benjamin Van Roy. Why is posterior sampling better than optimism for reinforcement learning? In International conference on machine learning, pp. 2701-2710. PMLR, 2017.  
Yi Ouyang, Mukul Gagrani, Ashutosh Nayyar, and Rahul Jain. Learning unknown markov decision processes: A thompson sampling approach. Advances in neural information processing systems, 30, 2017.  
Debjit Paria and Abhishek Sinha. Leadcache: Regret-optimal caching in networks. Advances in Neural Information Processing Systems, 34:4435-4447, 2021.  
Zhaoqing Peng, Junqi Jin, Lan Luo, Yaodong Yang, Rui Luo, Jun Wang, Weinan Zhang, Haiyang Xu, Miao Xu, Chuan Yu, et al. Learning to infer user hidden states for online sequential advertising. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pp. 2677-2684, 2020.  
Joelle Pineau, Geoff Gordon, Sebastian Thrun, et al. Point-based value iteration: An anytime algorithm for pomdpds. In *IJCAI*, volume 3, pp. 1025–1032. CiteSeer, 2003.  
Stéphane Ross, Joelle Pineau, Brahim Chaib-draa, and Pierre Kreitmann. A bayesian approach for learning and planning in partially observable markov decision processes. Journal of Machine Learning Research, 12(5), 2011.  
Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research, 39(4):1221-1243, 2014.  
Steven L Scott. A modern bayesian look at the multi-armed bandit. Applied Stochastic Models in Business and Industry, 26(6):639-658, 2010.  
Shang-Pin Sheng, Mingyan Liu, and Romesh Saigal. Data-driven channel modeling using spectrum measurement. IEEE Transactions on Mobile Computing, 14(9):1794-1805, 2014.  
David Silver and Joel Veness. Monte-carlo planning in large pomdpds. Advances in neural information processing systems, 23, 2010.  
Richard D Smallwood and Edward J Sondik. The optimal control of partially observable markov processes over a finite horizon. Operations research, 21(5):1071-1088, 1973.

Cem Tekin and Mingyan Liu. Online learning in opportunistic spectrum access: A restless bandit approach. In 2011 Proceedings IEEE INFOCOM, pp. 2462-2470. IEEE, 2011.  
Siwei Wang, Longbo Huang, and John Lui. Restless-ucb, an efficient and low-complexity algorithm for online restless bandits. Advances in Neural Information Processing Systems, 33:11878-11889, 2020.  
Guojun Xiong, Shufan Wang, Gang Yan, and Jian Li. Reinforcement learning for dynamic dimensioning of cloud caches: A restless bandit approach. In IEEE INFOCOM 2022-IEEE Conference on Computer Communications, pp. 2108-2117. IEEE, 2022a.  
Guojun Xiong, Shufan Wang, Gang Yan, and Jian Li. Reinforcement Learning for Dynamic Dimensioning of Cloud Caches: A Restless Bandit Approach. In Proc. of IEEE INFOCOM, 2022b.  
Xiang Zhou, Yi Xiong, Ningyuan Chen, and Xuefeng Gao. Regime switching bandits. Advances in Neural Information Processing Systems, 34, 2021.
