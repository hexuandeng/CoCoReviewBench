# WHEN CAN WE LEARN GENERAL-SUM MARKOV GAMES WITH A LARGE NUMBER OF PLAYERS SAMPLE-EFFICIENTLY?

Anonymous authors

Paper under double-blind review

# ABSTRACT

Multi-agent reinforcement learning has made substantial empirical progresses in solving games with a large number of players. However, theoretically, the best known sample complexity for finding a Nash equilibrium in general-sum games scales exponentially in the number of players due to the size of the joint action space, and there is a matching exponential lower bound. This paper investigates what learning goals admit better sample complexities in the setting of  $m$ -player general-sum Markov games with  $H$  steps,  $S$  states, and  $A_{i}$  actions per player. First, we design algorithms for learning an  $\varepsilon$ -Coarse Correlated Equilibrium (CCE) in  $\widetilde{\mathcal{O}}(H^5 S \max_{i \leq m} A_i / \varepsilon^2)$  episodes, and an  $\varepsilon$ -Correlated Equilibrium (CE) in  $\widetilde{\mathcal{O}}(H^6 S \max_{i \leq m} A_i^2 / \varepsilon^2)$  episodes. This is the first line of results for learning CCE and CE with sample complexities polynomial in  $\max_{i \leq m} A_i$ . Our algorithm for learning CE integrates an adversarial bandit subroutine which minimizes a weighted swap regret, along with several novel designs in the outer loop. Second, we consider the important special case of Markov Potential Games, and design an algorithm that learns an  $\varepsilon$ -approximate Nash equilibrium within  $\widetilde{\mathcal{O}}(S \sum_{i \leq m} A_i / \varepsilon^3)$  episodes (when only highlighting the dependence on  $S$ ,  $A_i$ , and  $\varepsilon$ ), which only depends linearly in  $\sum_{i \leq m} A_i$  and significantly improves over the best known algorithm in the  $\varepsilon$  dependence. Overall, our results shed light on what equilibria or structural assumptions on the game may enable sample-efficient learning with many players.

# 1 INTRODUCTION

Multi-agent reinforcement learning (RL) has achieved substantial recent successes in solving artificial intelligence challenges such as GO (Silver et al., 2016; 2018), multi-player games with team play such as Starcraft (Vinyals et al., 2019) and Dota2 (Berner et al., 2019), behavior learning in social interactions (Baker et al., 2019), and economic simulation (Zheng et al., 2020; Trott et al., 2021). In many applications, multi-agent RL is able to yield high quality policies for multi-player games with a large number of players (Wang et al., 2016; Yang et al., 2018).

Despite these empirical progresses, theoretical understanding of when we can sample-efficiently solve multi-player games with a large number of players remains elusive, especially in the setting of multi-player Markov games. A main bottleneck here is the exponential blow-up of the joint action space—The total number of joint actions in a generic game with simultaneous plays is equal to the product of the number of actions for each player, which scales exponentially in the number of players. Such an exponential dependence is indeed known to be unavoidable in the worst-case for certain standard problems. For example, for learning an approximate Nash equilibrium from payoff queries in an one-step multi-player general-sum game, the query complexity lower bound of (Chen et al., 2015; Rubinstein, 2016) shows that at least exponentially many queries (samples) is required, even when each player only has two possible actions and the query is noiseless. Moreover, for learning Nash equilibrium in Markov games, the best existing sample complexity upper bound also scales with the size of the joint action space (Liu et al., 2021).

Nevertheless, these exponential lower bounds do not completely rule out interesting theoretical inquiries—there may well be other notions of equilibria or additional structures within the game that allow us to learn with a better sample complexity. This motivates us to ask the following

Question: When can we solve general-sum Markov games with sample complexity milder than exponential in the number of players?

This paper makes steps towards answering the above question by considering multi-player general-sum Markov games (MGs) with  $m$  players,  $H$  steps,  $S$  states, and  $A_{i}$  actions per player. We make two lines of investigations: (1) Can we learn alternative notions of equilibria with better sample complexity than learning Nash; (2) Can the Nash equilibrium be learned with better sample complexity under additional structural assumptions on the game. This paper makes contributions on both ends, which we summarize as follows.

- We first design an algorithm that learns the  $\varepsilon$ -approximate Coarse Correlated Equilibrium (CCE) with  $\tilde{O}(H^5 S \max_{i \in [m]} A_i / \varepsilon^2)$  episodes of play (Section 3). Our algorithm CCE-V-Learning is a multi-player adaptation of the Nash V-Learning algorithm of Bai et al. (2020).  
- We design an algorithm CE-V-LEARNING which learns the stricter notion of  $\varepsilon$ -approximate Correlated Equilibrium (CE) with  $\widetilde{\mathcal{O}}(H^6 \max_{i \in [m]} A_i^2 / \varepsilon^2)$  episodes of play (Section 4). For Markov games, these are the first line of sample complexity results for learning CE and CCE that only scales polynomially with  $\max_{i \in [m]} A_i$ , and improves significantly in the  $A_i$  dependency over the current best algorithm which scales with  $\prod_{i \in [m]} A_i$ .  
- Technically, our algorithm CE-V-LEARNING makes several major modifications over CCEV-LEARNING in order to learn the CE (Section 4.2). Notably, inspired by the connection between CE and low swap-regret learning, we use a mixed-expert Follow-The-Regularized Leader algorithm within its inner loop to achieve low swap-regret for a particular adversarial bandit problem. Our analysis also contains new results for adversarial bandits on weighted swap regret and weighted regret with predicable weights, which may be of independent interest.  
- Finally, we consider learning Nash equilibrium in Markov Potential Games (MPGs), an important subclass of general-sum Markov games. By a reduction to single-agent RL, we design an algorithm NASH-CA that achieves  $\widetilde{\mathcal{O}} (\Phi_{\mathrm{max}}H^3 S\sum_{i\in [m]}A_i / \varepsilon^3)$  sample complexity, where  $\Phi_{\mathrm{max}}\leq Hm$  is the bound on the potential function (Section 5). Compared with the recent result of (Leonardos et al., 2021), we significantly improves the  $\varepsilon$  dependence from their  $1 / \varepsilon^{6}$ .

# 1.1 RELATED WORK

Learning equilibria in general-sum games The sample (query) complexity of learning Nash, CE, and CCE from samples in one-step (i.e. normal form) general-sum names with  $m$  players and  $A_{i}$  actions per player has been studied extensively in literature (Hart & Mas-Colell, 2000; Hart, 2005; Stoltz, 2005; Cesa-Bianchi & Lugosi, 2006; Blum & Mansour, 2007; Fearnley et al., 2015; Babichenko & Barman, 2015; Chen et al., 2015; Fearnley & Savani, 2016; Goldberg & Roth, 2016; Babichenko, 2016; Rubinstein, 2016; Hart & Nisan, 2018). It is known that learning Nash equilibrium requires exponential in  $m$  samples in the worst case Rubinstein (2016), whereas CE and CCE admit efficient poly  $(m, \max_{i \leq m} A_{i})$ -sample complexity algorithms by independent no-regret learning (Hart & Mas-Colell, 2000; Hart, 2005; Goldberg & Roth, 2016; Daskalakis et al., 2021). Our results for learning CE and CCE can be seen as extension of these works into Markov games. We remark that even when the game is fully known, the computational complexity for finding Nash in general-sum games is still PPAD-hard (Daskalakis, 2013).

Markov games Markov games (Shapley, 1953; Littman, 1994) is a widely used framework for game playing with sequential decision making, e.g. in multi-agent reinforcement learning. Algorithms with asymptotic convergence have been proposed in the early works of (Hu & Wellman, 2003; Littman, 2001; Hansen et al., 2013). A recent line of work studies the non-asymptotic sample complexity for learning Nash in two-player zero-sum Markov games (Bai & Jin, 2020; Xie et al., 2020; Bai et al., 2020; Zhang et al., 2020; Liu et al., 2021; Chen et al., 2021; Jin et al., 2021; Huang et al., 2021) and learning various equilibria in general-sum Markov games (Liu et al., 2021; Bai et al., 2021), building on techniques for learning single-agent Markov Decision Processes sample-efficiently (Azar et al., 2017; Jin et al., 2018). Learning the Nash equilibrium in general-sum Markov games are much harder than that in zero-sum Markov games. Liu et al. (2021) present the first line of results for learning Nash, CE, and CCE in general-sum Markov games; however their sample complexity scales with  $\prod_{i\leq m}A_i$  due to the model-based nature of their algorithm. Algorithms for learning CE

in extensive-form games has been studied in (Celli et al., 2020), though we remark Markov games and extensive-form games are different frameworks and our results do not imply each other.

Markov potential games Lastly, a recent line of works considers Markov potential games (Macua et al., 2018; Leonardos et al., 2021; Zhang et al., 2021), a subset of general-sum Markov games in which the Nash equilibrium admits more efficient algorithms. Leonardos et al. (2021) gives a sample-efficient algorithm based on the policy gradient method (Agarwal et al., 2021). The special case of Markov cooperative games is studied empirically in e.g. Lowe et al. (2017); Chao et al. (2021). For one step potential games, (Kleinberg et al., 2009; Palaiopanos et al., 2017; Cohen et al., 2017a) show the convergence to Nash equilibria of no-regret dynamics.

# 2 PRELIMINARIES

We present preliminaries for multi-player general-sum Markov games as well as the solution concept of (approximate) Nash equilibrium. Alternative solution concepts and other concrete subclasses of Markov games considered in this paper will be defined in the later sections.

Markov games A multi-player general sum Markov game (MG; (Shapley, 1953; Littman, 1994)) with  $m$  players can be described by a tuple  $\mathrm{MG}(H, S, \{\mathcal{A}_i\}_{i=1}^m, \mathbb{P}, \{r_i\}_{i=1}^m)$ , where  $H$  is the episode length,  $S$  is the state space with  $|S| = S$ ,  $\mathcal{A}_i$  is the action space for the  $i^{\text{th}}$  player with  $|\mathcal{A}_i| = A_i$ . Without loss of generality, we assume  $\mathcal{A}_i = [A_i]$ . We let  $\pmb{a} := (a_1, \dots, a_m)$  denote the vector of joint actions taken by all the players and  $\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_m$  denote the joint action space. Throughout this paper we assume that  $S$  and  $A_i$  are finite. The transition probability  $\mathbb{P} = \{\mathbb{P}_h\}_{h \in [H]}$  is the collection of transition matrices, where  $\mathbb{P}_h(\cdot | s, \pmb{a}) \in \Delta_S$  denotes the distribution of the next state when actions  $\pmb{a}$  are taken at state  $s$  at step  $h$ . The rewards  $r_i = \{r_{h,i}\}_{h \in [H], i \in [m]}$  is the collection of reward functions for the  $i^{\text{th}}$  player, where  $r_{h,i}(s, \pmb{a}) \in [0,1]$  gives the deterministic reward of  $i^{\text{th}}$  player if actions  $\pmb{a}$  are taken at state  $s$  at step  $h$ . Without loss of generality, we assume the initial state  $s_1$  is deterministic. A key feature of general-sum games is that the rewards  $r_i$  are in general different for each player  $i$ , and the goal of each player is to maximize her own cumulative reward.

Policy, value function A product policy is a collection of  $m$  policies  $\pi \coloneqq \{\pi_i\}_{i\in [m]}$  where  $\pi_{i}$  is the (potentially history-dependent) policy for the  $i$ -th player. We first focus on the case of Markov product policies, in which  $\pi_{i} = \{\pi_{h,i}:\mathcal{S}\to \Delta_{A_{i}}\}_{h\in [H]}$ , and  $\pi_{h,i}(a_i|s)$  is the probability for the  $i^{\mathrm{th}}$  player to take action  $a_{i}$  at state  $s$  at step  $h$ . For a policy  $\pi$  and  $i\in [m]$ , we use  $\pi_{-i}\coloneqq \{\pi_j\}_{j\in [m],j\neq i}$  to denote the policy of all but the  $i^{\mathrm{th}}$  player. The value function  $V_{h,i}^{\pi}(s):S\to \mathbb{R}$  is defined as the expected cumulative reward for the  $i^{\mathrm{th}}$  player when policy  $\pi$  is taken starting from state  $s$  and step  $h$ :

$$
V _ {h, i} ^ {\pi} (s) := \mathbb {E} _ {\pi} \left[ \sum_ {h ^ {\prime} = h} ^ {H} r _ {h ^ {\prime}, i} \left(s _ {h ^ {\prime}}, \boldsymbol {a} _ {h ^ {\prime}}\right) \mid s _ {h} = s \right]. \tag {1}
$$

Definition of the Q function and Bellman equations can be found in Appendix B.

Best response & Nash equilibrium For any product policy  $\pi = \{\pi_i\}_{i\in [m]}$ , the best response for the  $i^{\mathrm{th}}$  player against  $\pi_{-i}$  is defined as any policy  $\pi^{\dagger}$  such that  $V_{1,i}^{\pi^{\dagger},\pi_{-i}}(s_1) = \sup_{\pi_i'}V_{1,i}^{\pi_i',\pi_{-i}}(s_1)$ . For any Markov product policy, this best response is guaranteed to exist as the above maximization problem is equivalent to solving a Markov Decision Process (MDP) for the  $i^{\mathrm{th}}$  player. We will also use the notation  $V_{1,i}^{\dagger,\pi_{-i}}(s_1)$  to denote the above value function  $V_{1,i}^{\pi^{\dagger},\pi_{-i}}(s_1)$ .

We say  $\pi$  is a Nash equilibrium (e.g. Nash (1951); Pérolat et al. (2017)) if all players play the best response against other players, i.e., for all  $i \in [m]$ ,

$$
V _ {1, i} ^ {\pi} (s _ {1}) = V _ {1, i} ^ {\dagger , \pi_ {- i}} (s _ {1}).
$$

Note that in general-sum MGs, there may exist multiple Nash equilibrium policies with different value functions, unlike in two-player zero-sum MGs (Shapley, 1953). To measure the suboptimality

of any policy  $\pi$ , we define the NE-gap as

$$
\operatorname {N E - g a p} (\pi) := \max  _ {i \in [ m ]} \left[ \sup  _ {\mu_ {i}} V _ {1, i} ^ {\mu_ {i}, \pi_ {- i}} (s _ {1}) - V _ {1, i} ^ {\pi} (s _ {1}) \right].
$$

For any  $\varepsilon \geq 0$ , we say  $\pi$  is  $\varepsilon$ -approximate Nash equilibrium if  $\mathrm{NE - gap}(\pi) \leq \varepsilon$ .

General correlated policy A general correlated policy  $\pi$  is a set of  $H$  maps  $\pi \coloneqq \{\pi_h : \mathbb{R} \times (\mathcal{S} \times \mathcal{A} \times \mathbb{R})^{h-1} \times \mathcal{S} \to \Delta_{\mathcal{A}}\}_{h \in [H]}$ . The first argument of  $\pi_h$  is a random variable  $z \in \mathbb{R}$ , and the other arguments contain all the history information and the current state information (unlike Markov policies in which the policies only depend on the current state information). The output of  $\pi_h$  is a general distribution of actions in  $\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_m$  (unlike product policies in which the action distribution is a product distribution). The random variable  $z$  is sampled from some underlying distribution which may be shared among all steps  $h \in [H]$ .

For any correlated policy  $\pi = \{\pi_h\}_{h\in [H]}$  and any player  $i$ , we can define a marginal policy  $\pi_{-i}$  as a set of  $H$  maps  $\pi_{-i} := \{\pi_{h, - i}:\mathbb{R}\times (\mathcal{S}\times \mathcal{A}\times \mathbb{R})^{h - 1}\times \mathcal{S}\to \Delta_{\mathcal{A}_{-i}}\}_{h\in [H]}$  where  $\mathcal{A}_{-i}:= \mathcal{A}_1\times \dots \times \mathcal{A}_{i - 1}\times \mathcal{A}_{i + 1}\times \dots \times \mathcal{A}_m$ , and the output of  $\pi_{h, - i}$  is defined as the marginal distribution of the output of  $\pi_h$  restricted to the space  $\mathcal{A}_{-i}$ . For any general correlated policy  $\pi$ , we can define its initial state value function  $V_{1,i}^{\pi}(s_1)$  similar as (1). The best response value of the  $i^{\mathrm{th}}$  player against  $\pi$  is  $V_{1,i}^{\dagger,\pi^{-i}}(s_1) = \sup_{\mu_i}V_{1,i}^{\mu_i,\pi^{-i}}(s_1)$ , where  $V_{1,i}^{\mu_i,\pi^{-i}}(s_1)$  is the value function of the policy  $(\mu_i,\pi_{-i})$  (the  $i^{\mathrm{th}}$  player plays according to general policy  $\mu_i$ , and all other players play according to  $\pi_{-i}$ ), and the supremum is taken over all general policy  $\mu_i$  of the  $i^{\mathrm{th}}$  player.

Learning setting Throughout this paper we consider the interactive learning (i.e. exploration) setting where algorithms are able to play episodes within the MG and observe the realized transitions and rewards. Our focus is on the PAC sample complexity (i.e. number of episodes of play) for any learning algorithm to output an approximate equilibrium.

# 2.1 EXPONENTIAL LOWER BOUND FOR LEARNING APPROXIMATE NASH EQUILIBRIUM

The focus of this paper is the setting where the number of players  $m$  is large. Intuitively, as the joint action space has size  $|\mathcal{A}| = \prod_{i=1}^{m} A_i$  which scales exponentially in  $m$  (if each  $A_i \geq 2$ ), naive algorithms for learning Nash equilibrium may learn all  $r_i(a)$  by enumeratively querying all  $a \in \mathcal{A}$ , and this costs exponential in  $m$  samples. Unfortunately, recent work shows that such exponential in  $m$  dependence is unavoidable in the worst-case for any algorithm—there is an  $\exp(\Omega(m))$  sample complexity lower bound for learning approximate Nash, even in one-step general-sum games (Chen et al., 2015; Rubinstein, 2016):

Proposition 1 (Lower bound for learning approximate Nash (Rubinstein, 2016); Informal version of Theorem A.2). There exists some absolute constant  $\varepsilon_0 > 0$  and a family of  $m$ -player general-sum Markov games for any  $m$  with  $H = 1$ ,  $S = 1$ ,  $A_i = 2$ , and deterministic rewards, such that any algorithm that is able to find an  $\varepsilon_0$ -approximate Nash equilibrium with high probability must at least play  $2^{\Omega(m)}$  samples (episodes).

This suggests that the Nash equilibrium as a solution concept may be too hard to learn efficiently for MGs with a large number of players, and calls for alternative solution concepts or additional structural assumptions on the game in order to achieve an improved  $m$  dependence.

# 3 EFFICIENT LEARNING OF COARSE CORRELATED EQUILIBRIA (CCE)

Given the difficulty of learning Nash when the number of players  $m$  is large (Proposition 1), we consider learning other relaxed notions of equilibria for general-sum MGs. Two standard notions of equilibria for games are the Correlated Equilibrium (CE) and Coarse Correlated Equilibrium (CCE), and they satisfy  $\{\mathrm{Nash}\} \subset \{\mathrm{CE}\} \subset \{\mathrm{CCE}\}$  for general-sum MGs (Nisan et al., 2007).

We begin by considering learning CCE (most relaxed notion above) for Markov games.

Definition 2 ( $\varepsilon$ -approximate CCE for general-sum MGs). We say a (general) correlated policy  $\pi$  is an  $\varepsilon$ -approximate Coarse Correlated Equilibrium (CCE) if

$$
\max  _ {i \in [ m ]} \left(V _ {1, i} ^ {\dagger , \pi_ {- i}} (s _ {1}) - V _ {1, i} ^ {\pi} (s _ {1})\right) \leq \varepsilon .
$$

We say  $\pi$  is an (exact) CCE if the above is satisfied with  $\varepsilon = 0$ .

The following result shows that there exists an algorithm that can learn an  $\varepsilon$ -approximate CCE in general-sum Markov games within  $\tilde{O}(H^5 S \max_{i \in [m]} A_i / \varepsilon^2)$  episodes of play.

Theorem 3 (Learning  $\varepsilon$ -approximate CCE for general-sum MGs). There exists an algorithm CCE-V-LEARNING (Algorithm 4) for  $m$ -player general-sum MGs, such that running it for

$$
K \geq \mathcal {O} \bigg (\frac {H ^ {5} S \operatorname* {m a x} _ {i \in [ m ]} A _ {i} \iota}{\varepsilon^ {2}} + \frac {H ^ {4} S \iota^ {3}}{\varepsilon} \bigg),
$$

episodes ( $\iota = \log(m\max_{i\in [m]}A_iHSK / (p\varepsilon))$  is a log factor), we have with probability at least  $1 - p$  that the certified policy  $\widehat{\pi}$  defined in Algorithm 2 is an  $\varepsilon$ -approximate CCE, i.e.  $\max_{i\in [m]}\left(V_{1,i}^{\dagger,\widehat{\pi}_{-i}}(s_1) - V_{1,i}^{\widehat{\pi}}(s_1)\right) \leq \varepsilon$ .

Mild dependence on action space For small enough  $\varepsilon$ , the sample complexity featured in Theorem 3 scales as  $\widetilde{\mathcal{O}}(H^5 S \max_{i \in [m]} A_i / \varepsilon^2)$ . Most notably, this is the first algorithm that scales with  $\max_{i \in [m]} A_i$ , and exhibits a sharp difference in learning Nash and learning CCE in view of the  $\exp(\Omega(m))$  lower bound for learning Nash in Proposition 1. Indeed, existing algorithms such as Multi-Nash-VI Algorithm with CCE subroutine (Liu et al., 2021) does require  $\widetilde{\mathcal{O}}(H^3 S^2 \prod_{i=1}^{m} A_i / \varepsilon^2)$  episodes of play, which scales with  $\prod_{i \in [m]} A_i$  due to its model-based nature. We achieve significantly better dependence on  $A_i$  and also  $S$ , though slightly worse  $H$  dependence.

Overview of algorithm and techniques Our CCE-V-LEARNING algorithm (deferred to Appendix C.1 due to space limit) is a multi-player adaptation of the Nash V-Learning algorithm of Bai et al. (2020); Tian et al. (2021) for learning Nash equilibria in two-player zero-sum MGs. Similar as Bai et al. (2020), we show that this algorithm enjoys a "no-regret" like guarantee for each player at each  $(h, s)$  (Lemma C.3), which when combined with our "certified correlated policy" procedure (Algorithm 2) outputs a policy that is  $\varepsilon$ -CCE. The key feature allowing this  $\max_{i \in [m]} A_i$  dependence is that this algorithm uses uncoupled learning for each player, as opposed to the centralized model-based algorithm of Liu et al. (2021). The full proof of Theorem 3 can be found in Appendix C.

# 4 EFFICIENT LEARNING OF CORRELATED EQUILIBRIA (CE)

In this section, we move on to considering the harder problem of learning Correlated Equilibria (CE). We first present the definition of CE in Markov games.

Definition 4 (Strategy modification for  $i^{\text{th}}$  player). A strategy modification  $\phi := \{\phi_{h,s}\}_{(h,s) \in [H] \times S}$  for player  $i$  is a set of  $S \times H$  functions  $\phi_{h,s}: \mathcal{A}_i \to \mathcal{A}_i$ . A strategy modification  $\phi$  can be composed with any policy  $\pi$  to give a modified policy  $\phi \diamond \pi$  defined as follows: At any step  $h$  and state  $s$ , if  $\pi$  chooses to play  $\mathbf{a} = (a_1, \ldots, a_m)$ , the modified policy  $\phi \diamond \pi$  will play  $(a_1, \ldots, a_{i-1}, \phi_{h,s}(a_i), a_{i+1}, \ldots, a_m)$ . We use  $\Phi_i$  denote the set of all possible strategy modifications for player  $i$ .

Definition 5 ( $\varepsilon$ -approximate CE for general-sum MGs). We say a (general) correlated policy  $\pi$  is an  $\varepsilon$ -approximate CE if

$$
\max  _ {i \in [ m ]} \sup  _ {\phi \in \Phi_ {i}} \left(V _ {1, i} ^ {\phi \diamond \pi} (s _ {1}) - V _ {1, i} ^ {\pi} (s _ {1})\right) \leq \varepsilon .
$$

We say  $\pi$  is an (exact)  $CE$  if the above is satisfied with  $\varepsilon = 0$ .

Our definition of CE follows (Liu et al., 2021) and is a natural generalization of the CE for the well-studied special case of one-step (i.e. normal form) games (Nisan et al., 2007).

# 4.1 ALGORITHM DESCRIPTION

Our algorithm CE-V-LEARNING (Algorithm 1) builds further on top of CCE-V-LEARNING and Nash V-Learning, and makes several novel modifications in order to learn the CE. At a high-level, CE-V-LEARNING consists of the following steps:

- Line 6-11 (Sample action using mixed-expert FTRL): For each  $(h,s)$  we maintain  $A_{i}$  "sub-experts" indexed by  $b^{\prime} \in [A_{i}]$ . Each sub-expert  $b^{\prime}$  first computes an action distribution  $q^{b^{\prime}}(\cdot) \in \Delta_{A_{i}}$  via Follow-the-Regularized-Leader (FTRL; Line 8). Then we employ a two-step sampling procedure to obtain the action: First sample a sub-expert  $b$  from a suitable distribution  $\mu$  computed from  $\{q^{b^{\prime}}\}_{b^{\prime} \in [A_{i}]}$ , then sample the actual action  $a_{h,i}$  from  $q^{b}$ .  
- Line 13-17 (Take action and record observations): Player  $i$  takes action  $a_{h,i}$  and observes other player's actions, the reward, and the next state. Sub-expert  $b$  then computes a loss estimator and weight according to the observations, which will be used in future FTRL updates.  
- Line 19 (Optimistic value update): Updates the optimistic estimate of the value  $\overline{V}_{h,i}$  using step-size  $\alpha_{t}$  and bonus  $\overline{\beta}_{t}$ .

Finally, after executing Algorithm 1 for  $K$  episodes, we use the certified correlated policy procedure (Algorithm 2) to obtain our final output policy  $\widehat{\pi}$ . This procedure is a direct modification of the certified policy procedure of (Bai et al., 2020) and outputs a correlated policy instead of product policy. The same procedure is also used for learning CCEs earlier in Section 3.

Here we specify the hyperparameters used in Algorithm 1:

$$
\alpha_ {t} = (H + 1) / (H + t), \quad \eta_ {t} = \sqrt {\iota / A _ {i} t}, \quad \overline {{\beta}} _ {t} = c H ^ {2} A _ {i} \sqrt {\iota / t} + 2 c H ^ {2} \iota / t. \qquad (2)
$$

The constants  $\alpha_{t}^{j}$  used in Algorithm 2 is defined as

$$
\alpha_ {t} ^ {0} := \prod_ {k = 1} ^ {t} (1 - \alpha_ {k}), \quad \alpha_ {t} ^ {j} := \alpha_ {j} \prod_ {k = j + 1} ^ {t} (1 - \alpha_ {k}). \tag {3}
$$

Note that for any  $t \geq 1$ ,  $\{\alpha_t^j\}_{1 \leq j \leq t}$  sums to one and defines a distribution over  $[t]$ .

# 4.2 OVERVIEW OF TECHNIQUES

Here we briefly overview the techniques used in Algorithm 1.

Minimizing weighted swap regret via mixed-expert FTRL. The key technical advance in our Algorithm 1 over CCE-V-LEARNING and Nash V-Learning is the use of mixed-expert FTRL (Line 6-11). The purpose of this is to allow the algorithm to achieve low swap regret at each  $(h,s)$  in a suitable sense—For one-step (normal form) games, it is known that combining low-swap-regret learning for each player leads to an approximate CE (Stoltz, 2005; Cesa-Bianchi & Lugosi, 2006). To integrate this into Markov games, we utilize a celebrated reduction from low-swap-regret learning to usual low-regret learning (Blum & Mansour, 2007), which for any bandit problem with  $A_{i}$  actions maintains  $A_{i}$  sub-experts each running its own FTRL algorithm. Our particular application builds upon the two-step randomization scheme of Ito (2020) which first samples a sub-expert  $b$  and the action from this sub-expert. The distribution  $\mu (\cdot)$  for sampling the sub-expert is carefully chosen by solving a linear system (Line 10) so that  $\mu$  also coincides with the (marginal) distribution of the sampled action, from which the reduction follows.

FTRL with predictable weights. Applied naively, the above reduction does not directly work for our purpose, as our analysis requires minimizing the weighted swap regret with weights  $\alpha_{t}^{i}$ , whereas the reduction of Ito (2020) relies crucially on the vanilla (average) regret. We address this challenge by using a slightly modified FTRL algorithm for each sub-expert that takes in random but predictable weights (i.e. depending fully on prior information and "external" randomness). We present the analysis for such FTRL algorithm in Appendix G.2, and the consequent analysis for the weighted swap regret in Appendix G.1, both of which may be of independent interest.

Proposal distributions. Finally, a nuanced but important new design in CE-V-LEARNING is that all sub-experts compute a proposal action distribution to sample the sub-expert and the associated action. Then, only the sampled sub-expert takes this action, and all other proposal distributions are discarded. This is different from the original algorithms of (Blum & Mansour, 2007; Ito, 2020) in which the

Algorithm 1 CE-V-LEARNING for general-sum MGs (i-th player's version)  
1: Initialize: For any  $(s,a,h)$  set  $\overline{V}_{h,i}(s)\gets H$ $N_{h}(s)\gets 0$  . Set  $\mu_h(a|s)\gets 1 / A_i$ $q_{h}^{b^{\prime}}(a|s)\gets$ $1 / A_{i},\ell_{h,t}^{b^{\prime}}(a|s)\gets 0,N_{h}^{b^{\prime}}(s)\gets 0$  for all  $(b^{\prime},a,h,s,t)\in [A_i]\times [A_i]\times [H]\times S\times [K].$    
2: for episode  $k = 1,\dots ,K$  do   
3:Receive  $s_1$    
4:for step  $h = 1,\ldots ,H$  do   
5: // Compute proposal action distributions by FTRL   
6: Update accumulator  $t\coloneqq N_h(s_h)\leftarrow N_h(s_h) + 1.$  Set  $u_{t}\leftarrow \alpha_{t}^{t} / \alpha_{t}^{1}$    
7: Let  $t_{b^{\prime}}\gets N_{h}^{b^{\prime}}(s_{h})$  for all  $b^{\prime}\in [A_i]$  for shorthand.   
8: Compute the action distribution for all sub-experts  $b^{\prime}\in [A_i]$  ..  $q_{h}^{b^{\prime}}(a|s_{h})\propto_{a}\exp \big(-(\eta_{t_{b^{\prime}}} / u_{t})\sum_{\tau = 1}^{t_{b^{\prime}}}w_{h,\tau}(b^{\prime}|s_{h})\ell_{h,\tau}^{b^{\prime}}(a|s_{h})\big).$    
9: // Sample sub-expert  $b$  and action from  $q^{b}(\cdot)$    
10: Compute  $\mu_h(\cdot |s_h)\in \Delta_{[A_i]}$  by solving  $\mu_h(\cdot |s_h) = \sum_{b' = 1}^{A_i}\mu_h(b'|s_h)q_h^{b'}(\cdot |s_h)$    
11:Sample sub-expert  $b\sim \mu_h(\cdot |s_h)$  , and then action  $a_{h,i}\sim q_h^b (\cdot |s_h)$    
12: // Take action and feed the observations to sub-expert  $b$    
13:Take action  $a_{h,i}$  , observe the action  $a_{h, - i}$  from the other players.   
14:Observe reward  $r_{h,i} = r_{h,i}(s_h,a_{h,i},a_{h, - i})$  and next state  $s_{h + 1}$  from the environment.   
15:Update accumulator for sampled sub-expert:  $t_b\coloneqq N_h^b (s_h)\gets N_h^b (s_h) + 1.$    
16:Compute loss estimator  $\ell_{h,t_b}^b (a|s_h)\gets \frac{[H - h + 1 - (r_{h,i} + \min\{\overline{V}_{h + 1,i}(s_{h + 1}),H - h\})] / H\cdot 1\{a_{h,i} = a\}}{q_h^b(a_h|s_h) + \gamma_{t_b}}.$    
17:Set  $w_{h,t_b}(b|s_h)\gets u_t$    
18://Optimistic value update   
19: $\overline{V}_{h,i}(s_h)\gets (1 - \alpha_t)\overline{V}_{h,i}(s_h) + \alpha_t\left(r_{h,i}\left(s_h,a_{h,i},a_{h, - i}\right) + \overline{V}_{h + 1,i}\left(s_{h + 1}\right) + \overline{\beta}_t\right).$

Algorithm 2 Certified correlated policy  $\widehat{\pi}$  for general-sum MGs  
1: Sample  $k\gets$  Uniform([K]).   
2: for step  $h = 1,\dots ,H$  do   
3: Observe  $s_h$  , and set  $t\gets N_h^k (s_h)$  (the value of  $N_{h}(s_{h})$  at the beginning of the  $k^{\prime}$  th episode).   
4: Sample  $l\in [t]$  with  $\mathbb{P}(l = j) = \alpha_t^j$  (c.f. Eq. (3)).   
5:  $k\gets k_h^l (s_h)$  (the episode at the end of which the state  $s_h$  is observed exactly  $l$  times).   
6: Jointly take action  $(a_{h,1},a_{h,2},\ldots ,a_{h,m})\sim \prod_{i = 1}^{m}\mu_{h,i}^{k}(\cdot |s_{h})$  , where  $\mu_{h,i}^{k}(\cdot |s_{h})$  is the policy  $\mu_{h,i}(\cdot |s_h)$  at the beginning of  $k^{\prime}$  th episode.

FTRL update come after the sub-expert sampling and only happens for the sampled sub-expert. Our design is required here as otherwise the sub-experts are required to predict the next time when it is sampled in order to compute the weighted FTRL update, which is impossible.

# 4.3 THEORETICAL GUARANTEE

We are now ready to present the theoretical guarantee for our CE-V-LEARNING algorithm.

Theorem 6 (Learning  $\varepsilon$ -approximate CE for general-sum MGs). Suppose we run the CE-V-LEARNING algorithm (Algorithm 1) for all  $m$  players for

$$
K \geq \mathcal {O} \bigg (\frac {H ^ {6} S \max _ {i \in [ m ]} A _ {i} ^ {2} \iota}{\varepsilon^ {2}} + \frac {H ^ {4} S \max _ {i \in [ m ]} A _ {i} \iota^ {3}}{\varepsilon} \bigg),
$$

episodes ( $\iota = \log(m\max_{i\in [m]}A_iHSK / (p\varepsilon))$  is a log factor). Then with probability at least  $1 - p$ , the certified correlated policy  $\widehat{\pi}$  defined in Algorithm 2 is an  $\varepsilon$ -approximate CE i.e.  $\max_{i\in [m]}\sup_{\phi \in \Phi_i}V_{h,i}^{\phi \diamond \pi}(s_1) - V_{1,i}^{\widehat{\pi}}(s_1)\leq \varepsilon$ .

**Discussions** To the best of our knowledge, Theorem 6 presents the first result for learning CCE that scales polynomially with  $\max_{i\in [m]}A_i$ , which is significantly better than the best known existing algorithm of Multi-Nash-VI with CE subroutine (Liu et al., 2021) whose sample complexity scales with  $\prod_{i\in [m]}A_i$ . Similar as in Theorem 3, this follows as our CE-V-LEARNING performs uncoupled learning for each player, whereas Liu et al. (2021) uses model-based centralized learning.

We also observe that our sample complexity for learning CE is higher than for learning CCE by a factor of  $\widetilde{\mathcal{O}}(H\max_{i\in [m]}A_i)$ , which is expected as CE is a strictly harder notion of equilibrium and required a more sophisticated mixed-expert FTRL technique in the inner loop of the algorithm.

Finally, combining Theorem 3 & 6 with the exponential lower bound for learning Nash (Section 2.1), we obtain a full characterization of which equilibria can be learned with  $\mathrm{poly}(m)$  sample complexity in general-sum Markov games: This is possible for CCE and CE, but not Nash.

# 5 LEARNING NASH EQUILIBRIA IN MARKOV POTENTIAL GAMES

In this section, we consider learning Nash equilibria in Markov potential games (MPGs), an important subclass of general-sum MGs. Despite the curse of number of players of learning Nash in general-sum MGs, recent work shows that learning Nash in MPGs does not require sample size exponential in  $m$ , by using stochastic policy gradient based algorithms (Leonardos et al., 2021; Zhang et al., 2021). In this section, we provide an alternative algorithm NASH-CA that also achieves a mild dependence on  $m$  and an improved dependence on  $\varepsilon$  by a simple reduction to single-agent learning.

# 5.1 MARKOV POTENTIAL GAMES

We first present the definition of MPGs. Our definition is the finite-horizon variant $^2$  of the definitions of Macua et al. (2018); Leonardos et al. (2021); Zhang et al. (2021) and is slightly more general as we only require (4) on the total return. Throughout this section,  $\pi$  denotes a Markov product policy.

Definition 7. (Markov potential games) A general-sum Markov game is a Markov potential game if there exists a potential function  $\Phi : (\Delta_{\mathcal{A}_1})^H \times \dots \times (\Delta_{\mathcal{A}_m})^H \to [0, \Phi_{\max}]$ , such that for any  $i \in [m]$ , any two policies  $\pi_i, \pi_i'$  of the  $i^{th}$  player, and any policy  $\pi_{-i}$  of other players, the difference of the value functions of the  $i^{th}$  player with policies  $(\pi_i, \pi_{-i})$  and  $(\pi_i', \pi_{-i})$  is equal the difference of the potential function on the same policies, i.e.,

$$
V _ {1, i} ^ {\pi_ {i}, \pi_ {- i}} (s _ {1}) - V _ {1, i} ^ {\pi_ {i} ^ {\prime}, \pi_ {- i}} (s _ {1}) = \Phi (\pi_ {i}, \pi_ {- i}) - \Phi (\pi_ {i} ^ {\prime}, \pi_ {- i}). \tag {4}
$$

Note that the range of the potential function  $\Phi_{\mathrm{max}}$  admits a trivial upper bound  $\Phi_{\mathrm{max}} \leq mH$  (this can be seen by varying  $\pi_i$  for one  $i$  at a time). An important example of MPGs is Markov Cooperative Games (MCGs) where all players share the same reward  $r_i \equiv r$ .

# 5.2 ALGORITHM AND THEORETICAL GUARANTEE

We present a simple algorithm NASH-CA (Nash Coordinate Ascent) for learning an  $\varepsilon$ -approximate Nash equilibrium in MPGs. As its name suggests, the algorithm operates by solving single-agent Markov Decision Processes (MDPs) one player at a time, and intrinsically performing coordinate ascent on the potential function of the Markov game. Due to the potential structure of MPGs and the boundedness of the potential function, the local improvements of players across the steps can have an accumulative effect on the potential function, and the algorithm is guaranteed to stop after a bounded number of steps. We give the full description of the NASH-CA in Algorithm 3. We remark in passing that NASH-CA is additionally guaranteed to output a pure-strategy Nash equilibrium (cf. Appendix E for definition).

Theorem 8 (Sample complexity for NASH-CA). For Markov potential games, with probability at least  $1 - p$ , Algorithm 3 terminates within  $4\Phi_{max} / \varepsilon$  steps of the while loop, and outputs an  $\varepsilon$ -approximate (pure-strategy) Nash equilibrium. The total episodes of play is at most

$$
K = \mathcal {O} \left(\frac {\Phi_ {\mathrm {m a x}} H ^ {3} S \sum_ {i = 1} ^ {m} A _ {i} \iota}{\varepsilon^ {3}} + \frac {\Phi_ {m a x} H ^ {3} S ^ {2} \sum_ {i = 1} ^ {m} A _ {i} \iota^ {2}}{\varepsilon^ {2}}\right),
$$

Algorithm 3 NASH-CA for Markov potential games  
Require: Error tolerance  $\varepsilon$   
1: Initialize:  $\pi = \{\pi_i\}_{i\in [m]}$ , where  $\pi_{i} = \{\pi_{h,i}\}_{(h,i)\in [H]\times [m]}$  for some deterministic policy  $\pi_{h,i}$ .  
2: while true do  
3: Execute policy  $\pi$  for  $N = \Theta (\frac{H^2\iota}{\varepsilon^2})$  episodes and obtain  $\widehat{V}_{1,i}(\pi)$  which is the empirical average of the total return under policy  $\pi$ .  
4: for player  $i = 1,\ldots ,m$  do  
5: Fix  $\pi_{-i}$ , and let the  $i^{\mathrm{th}}$  player run UCBVI-UPLOW (Algorithm 7) for  $K_{i} = \Theta (\frac{H^{3}SA_{i}\iota}{\varepsilon^{2}} +\frac{H^{3}S^{2}A_{i}\iota^{2}}{\varepsilon})$  episodes and get a new deterministic policy  $\widehat{\pi}_i$ .  
6: Execute policy  $(\widehat{\pi}_i,\pi_{-i})$  for  $N = \Theta (\frac{H^{2}\iota}{\varepsilon^{2}})$  episodes and obtain  $\widehat{V}_{1,i}(\widehat{\pi}_i,\pi_{-i})$  which is the empirical average of the total return under policy  $(\widehat{\pi}_i,\pi_{-i})$ .  
7: Set  $\Delta_{i}\gets \widehat{V}_{1,i}(\widehat{\pi}_{i},\pi_{-i}) - \widehat{V}_{1,i}(\pi)$ .  
8: if  $\max_{i\in [m]}\Delta_i > \varepsilon /2$  then  
9: Update  $\pi_j\gets \widehat{\pi}_j$  where  $j = \arg \max_{i\in [m]}\Delta_i$ .  
10: else  
11: return  $\pi$

where  $\iota = \log \left(\frac{mHSK\max_{1\leq i\leq m}A_i}{\varepsilon p}\right)$  is a log factor.

**Discussions** For small enough  $\varepsilon$ , the sample complexity for the NASH-CA algorithm in the above Theorem is  $\widetilde{\mathcal{O}}(\Phi_{\max}H^3S\sum_{i\leq m}A_i/\varepsilon^3)$ . As  $\Phi_{\max}\leq mH$ , this at most scales with the number of players as  $m\sum_{i\leq m}A_i$ , which is much better than the exponential in  $m$  sample complexity for general-sum MGs without additional structures. Compared with recent results on learning Nash via policy gradients Leonardos et al. (2021); Zhang et al. (2021), the NASH-CA algorithm also achieves poly $(m,\max_{i\leq m}A_i)$  dependence, and significantly improves on the  $\varepsilon$  dependence from their  $\varepsilon^{-6}$  to  $\varepsilon^{-3}$ . In addition, our algorithm does require assumptions such bounded distribution mismatch coefficient as they do, due to the exploration nature of our single-agent MDP subroutine.

Also, compared with the sample complexity bound  $\widetilde{\mathcal{O}}(H^3 S \prod_{i=1}^{m} A_i / \varepsilon^2)$  of the Nash-VI algorithm (Liu et al., 2021) for general-sum MGs (not restricted to MPG), our NASH-CA algorithm doesn't suffer from the exponential dependence on  $m$  thanks to the MPG structure. We do achieve a looser in the dependence on  $\varepsilon$ , yet overall our sample complexity is still better unless  $\varepsilon < (\sum_{i=1}^{m} A_i) / (\prod_{i=1}^{m} A_i)$  is exponentially small.

A lower bound To accompany Theorem 8, we establish a sample complexity lower bound of  $\Omega(H^2\sum_{i=1}^{m} A_i / \varepsilon^2)$  for learning pure-strategy Nash in MCGs and hence MPGs (Theorem F.1 in Appendix F). This lower bound improves in the  $A_i$  dependence over the naive reduction to single-player MDPs (Domingues et al., 2021), which gives  $\Omega(H^3S \max_{i \in [m]} A_i / \varepsilon^2)$ , though is loose on the  $S, H$  dependence. The improved  $A_i$  dependence is achieved by constructing a novel class of hard instances of on one-step games (Lemma F.2), which may be of further technical interest. However, there is still a large gap between these lower bounds and the best current upper bound of either our  $\widetilde{\mathcal{O}}(\sum_{i=1}^{m} A_i / \varepsilon^3)$  or the  $\widetilde{\mathcal{O}}(\prod_{i=1}^{m} A_i / \varepsilon^2)$  of Liu et al. (2021), which we leave as future work.

# 6 CONCLUSION

This paper investigates the question of when can we solve general-sum Markov games (MGs) sample efficiently with a mild dependence on the number of players. Our results show that this is possible for learning approximate (Coarse) Correlated Equilibria in general-sum MGs, as well as learning approximate Nash equilibrium in Markov potential games. In both cases, our sample complexity bounds improve over existing results in many aspects. Our work opens up many interesting directions for future work, such as sharper algorithms for both problems, sample complexity lower bounds, or how to perform sample-efficient learning in general-sum MGs with function approximations. In addition to Markov potential games, it would also be interesting to explore alternative structural assumptions that permit sample-efficient learning.

# REFERENCES

Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22(98):1-76, 2021.  
Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning, pp. 263-272. PMLR, 2017.  
Yakov Babichenko. Query complexity of approximate nash equilibria. Journal of the ACM (JACM), 63(4):1-24, 2016.  
Yakov Babichenko and Siddharth Barman. Query complexity of correlated equilibrium. ACM Transactions on Economics and Computation (TEAC), 3(4):1-9, 2015.  
Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. In International Conference on Machine Learning, pp. 551-560. PMLR, 2020.  
Yu Bai, Chi Jin, and Tiancheng Yu. Near-optimal reinforcement learning with self-play. Advances in Neural Information Processing Systems, 33, 2020.  
Yu Bai, Chi Jin, Huan Wang, and Caiming Xiong. Sample-efficient learning of stackelberg equilibria in general-sum games. arXiv preprint arXiv:2102.11494, 2021.  
Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula. arXiv preprint arXiv:1909.07528, 2019.  
Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemyslaw Debiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.  
Avrim Blum and Yishay Mansour. From external to internal regret. Journal of Machine Learning Research, 8(6), 2007.  
Andrea Celli, Alberto Marchesi, Gabriele Farina, and Nicola Gatti. No-regret learning dynamics for extensive-form correlated equilibrium. arXiv preprint arXiv:2004.00603, 2020.  
Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge university press, 2006.  
YU Chao, A VELU, E VINITSKY, et al. The surprising effectiveness of ppo in cooperative, multiagent games. arXiv preprint arXiv:2103.01955, 2021.  
Xi Chen, Yu Cheng, and Bo Tang. Well-supported versus approximate nash equilibria: Query complexity of large games. arXiv preprint arXiv:1511.00785, 2015.  
Zixiang Chen, Dongruo Zhou, and Quanquan Gu. Almost optimal algorithms for two-player markov games with linear function approximation. arXiv preprint arXiv:2102.07404, 2021.  
Johanne Cohen, Amélie Héliou, and Panayotis Mertikopoulos. Learning with bandit feedback in potential games. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 6372-6381, 2017a.  
Michael B Cohen, Jonathan Kelner, John Peebles, Richard Peng, Anup B Rao, Aaron Sidford, and Adrian Vladu. Almost-linear-time algorithms for markov chains and new spectral primitives for directed graphs. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, pp. 410-419, 2017b.  
Christoph Dann and Emma Brunskill. Sample complexity of episodic fixed-horizon reinforcement learning. arXiv preprint arXiv:1510.08906, 2015.  
Constantinos Daskalakis. On the complexity of approximating a nash equilibrium. ACM Transactions on Algorithms (TALG), 9(3):1-35, 2013.

Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. Near-optimal no-regret learning in general games. arXiv preprint arXiv:2108.06924, 2021.  
Omar Darwiche Domingues, Pierre Ménard, Emilie Kaufmann, and Michal Valko. Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited. In Algorithmic Learning Theory, pp. 578-598. PMLR, 2021.  
John Fearnley and Rahul Savani. Finding approximate nash equilibria of bimatrix games via payoff queries. ACM Transactions on Economics and Computation (TEAC), 4(4):1-19, 2016.  
John Fearnley, Martin Gairing, Paul W Goldberg, and Rahul Savani. Learning equilibria of games via payoff queries. J. Mach. Learn. Res., 16:1305-1344, 2015.  
Paul W Goldberg and Aaron Roth. Bounds for the query complexity of approximate equilibria. ACM Transactions on Economics and Computation (TEAC), 4(4):1-25, 2016.  
Richard W Hamming. Error detecting and error correcting codes. The Bell system technical journal, 29(2):147-160, 1950.  
Thomas Dueholm Hansen, Peter Bro Miltersen, and Uri Zwick. Strategy iteration is strongly polynomial for 2-player turn-based stochastic games with a constant discount factor. Journal of the ACM (JACM), 60(1):1-16, 2013.  
Sergiu Hart. Adaptive heuristics. Econometrica, 73(5):1401-1430, 2005.  
Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127-1150, 2000.  
Sergiu Hart and Noam Nisan. The query complexity of correlated equilibria. Games and Economic Behavior, 108:401-410, 2018.  
Junling Hu and Michael P Wellman. Nash q-learning for general-sum stochastic games. Journal of machine learning research, 4(Nov):1039-1069, 2003.  
Baihe Huang, Jason D Lee, Zhaoran Wang, and Zhuoran Yang. Towards general function approximation in zero-sum markov games. arXiv preprint arXiv:2107.14702, 2021.  
Shinji Ito. A tight lower bound and efficient reduction for swap regret. Advances in Neural Information Processing Systems, 33, 2020.  
Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 4868-4878, 2018.  
Chi Jin, Qinghua Liu, and Tiancheng Yu. The power of exploiter: Provable multi-agent rl in large state spaces. arXiv preprint arXiv:2106.03352, 2021.  
Robert Kleinberg, Georgios Piliouras, and Éva Tardos. Multiplicative updates outperform generic no-regret learning in congestion games. In Proceedings of the forty-first annual ACM symposium on Theory of computing, pp. 533-542, 2009.  
Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.  
Stefanos Leonardos, Will Overman, Ioannis Panageas, and Georgios Piliouras. Global convergence of multi-agent policy gradient in markov potential games. arXiv preprint arXiv:2106.01969, 2021.  
Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In Machine learning proceedings 1994, pp. 157-163. Elsevier, 1994.  
Michael L Littman. Friend-or-foe q-learning in general-sum games. In ICML, volume 1, pp. 322-328, 2001.  
Qinghua Liu, Tiancheng Yu, Yu Bai, and Chi Jin. A sharp analysis of model-based reinforcement learning with self-play. In International Conference on Machine Learning, pp. 7001-7010. PMLR, 2021.

Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. arXiv preprint arXiv:1706.02275, 2017.  
Sergio Valcarcel Macua, Javier Zazo, and Santiago Zazo. Learning parametric closed-loop policies for markov potential games. In International Conference on Learning Representations, 2018.  
John Nash. Non-cooperative games. Annals of mathematics, pp. 286-295, 1951.  
Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. arXiv preprint arXiv:1506.03271, 2015.  
Noam Nisan, Tim Roughgarden, Eva Tardos, and Vijay V Vazirani. Algorithmic Game Theory. Cambridge University Press, 2007.  
Gerasimos Palaiopanos, Ioannis Panageas, and Georgios Piliouras. Multiplicative weights update with constant step-size in congestion games: Convergence, limit cycles and chaos. arXiv preprint arXiv:1703.01138, 2017.  
Julien Pérolat, Florian Strub, Bilal Piot, and Olivier Pietquin. Learning nash equilibrium for general-sum markov games from batch data. In Artificial Intelligence and Statistics, pp. 232-241. PMLR, 2017.  
Aviad Rubinstein. Settling the complexity of computing approximate two-player nash equilibria. In 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS), pp. 258-265. IEEE, 2016.  
Lloyd S Shapley. Stochastic games. Proceedings of the national academy of sciences, 39(10): 1095-1100, 1953.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484-489, 2016.  
David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419): 1140-1144, 2018.  
Gilles Stoltz. Incomplete information and internal regret in prediction of individual sequences. PhD thesis, Université Paris Sud-Paris XI, 2005.  
Yi Tian, Yuanhao Wang, Tiancheng Yu, and Suvrit Sra. Online learning in unknown markov games. In International Conference on Machine Learning, pp. 10279-10288. PMLR, 2021.  
Alexander Trott, Sunil Srinivasa, Douwe van der Wal, Sebastien Haneuse, and Stephan Zheng. Building a foundation for data-driven, interpretable, and robust policy design using the ai economist. arXiv preprint arXiv:2108.02904, 2021.  
Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michael Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350-354, 2019.  
Jun Wang, Weinan Zhang, and Shuai Yuan. Display advertising with real-time bidding (rtb) and behavioural targeting. arXiv preprint arXiv:1610.03013, 2016.  
Qiaomin Xie, Yudong Chen, Zhaoran Wang, and Zhuoran Yang. Learning zero-sum simultaneous-move markov games using function approximation and correlated equilibrium. In Conference on Learning Theory, pp. 3674-3682. PMLR, 2020.  
Tengyang Xie, Nan Jiang, Huan Wang, Caiming Xiong, and Yu Bai. Policy finetuning: Bridging sample-efficient offline and online reinforcement learning. arXiv preprint arXiv:2106.04895, 2021.  
Yaodong Yang, Rui Luo, Minne Li, Ming Zhou, Weinan Zhang, and Jun Wang. Mean field multiagent reinforcement learning. In International Conference on Machine Learning, pp. 5571-5580. PMLR, 2018.

Kaiqing Zhang, Sham M Kakade, Tamer Başar, and Lin F Yang. Model-based multi-agent rl in zero-sum markov games with near-optimal sample complexity. arXiv preprint arXiv:2007.07461, 2020.  
Runyu Zhang, Zhaolin Ren, and Na Li. Gradient play in multi-agent markov stochastic games: Stationary points and convergence. arXiv preprint arXiv:2106.00198, 2021.  
Stephan Zheng, Alexander Trot, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C Parkes, and Richard Socher. The ai economist: Improving equality and productivity with ai-driven tax policies. arXiv preprint arXiv:2004.13332, 2020.
