# EFFICIENT INFERENCE AND EXPLORATION FOR REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Despite an ever growing literature on reinforcement learning algorithms and applications, much less is known about their statistical inference. In this paper, we investigate the large-sample behaviors of the Q-value estimates with closed-form characterizations of the asymptotic variances. This allows us to efficiently construct confidence regions for Q-value and optimal value functions, and to develop policies to minimize their estimation errors. This also leads to a policy exploration strategy that relies on estimating the relative discrepancies among the Q estimates. Numerical experiments show superior performances of our exploration strategy than other benchmark approaches.

# 1 INTRODUCTION

We consider the classical reinforcement learning (RL) problem where the agent interacts with a random environment and aims to maximize the accumulated discounted reward over time. The environment is formulated as a Markov decision process (MDP) and the agent is uncertain about the true dynamics to start with. As the agent interacts with the environment, data about the system dynamics are collected and the agent becomes increasingly confident about her decision. With finite data, however, the potential reward from each decision is estimated with errors and the agent may be led to a suboptimal decision. Our focus in this paper is on statistically efficient methodologies to quantify these errors and uncertainties, and to demonstrate their use in obtaining better policies.

More precisely, we investigate the large-sample behaviors of estimated Q-value, optimal value function, and their associated policies. Our results are in the form of asymptotic convergence to an explicitly identified and computable Gaussian (or other) distribution, as the collected data sizes increase. The motivation of our investigation is three-fold. First, these precise asymptotic statements allow us to construct accurate confidence regions for quantities related to the optimal policy, and, like classical statistical inference, they can assess the reliability of the current estimates with respect to the data noises. Second, our results complement some finite-sample error bounds developed in the literature (Kearns & Singh, 1998; Kakade, 2003; Munos & Szepesvári, 2008), by supplementing a closed-form asymptotic variance that often shows up in the first-order terms in these bounds.

Our third and most important motivation is to design good exploration policies by directly using our tight error estimates. Motivated by recent autonomous-driving and other applications (e.g., Kalashnikov et al. (2018)), we consider the pure exploration setting where an agent is first assigned an initial period to collect as much experience as possible, and then, with the optimal policy trained offline, starts deployment to gain reward. We propose an efficient strategy to explore by optimizing the worst-case estimated relative discrepancy among the Q-values (ratio of mean squared difference to variance), which provides a proxy for the probability of selecting the best policy. Similar criteria have appeared in the so-called optimal computing budget allocation (OCBA)

procedure in simulation-based optimization (Chen & Lee, 2011) (a problem closely related to best-arm identification (Audibert & Bubeck, 2010) in online learning). In this approach, one divides computation (or observation) budget into stages in which one sequentially updates mean and variance estimates, and optimizes next-stage budget allocations according to the worst-case relative discrepancy criterion. Our proposed procedure, which we term Q-OCBA, follows this idea with a crucial use of our Q-value estimates and randomized policies to achieve the optimal allocation. We demonstrate how this idea consistently outperforms other benchmark exploration policies, both in terms of the probability in selecting the best policy and generating the tightest confidence bounds for value estimates at the end of the exploration period.

Regarding the problem of constructing tight error estimates in RL, the closest work to ours is Mannor et al. (2004; 2007), which studies the bias and variance in value function estimates with a fixed policy. Our technique resolves a main technical challenge in Mannor et al. (2004; 2007), which allows us to substantially generalize their variance results to Q-values, optimal value functions and asymptotic distributional statements. The derivation in Mannor et al. (2004; 2007) hinges on an expansion of the value function in terms of the perturbation of the transition matrix, which (as pointed out by the authors) is not easily extendable from a fixed-policy to the optimal value function. In contrast, our results utilize an implicit function theorem applied to the Bellman equation that can be verified to be sufficiently smooth. This idea turns out to allow us to obtain gradients for Q-values, translate to the optimal value function, and furthermore generalize to similar results for constrained MDP and approximate value iterations. We also relate our work to the line of studies on dynamic treatment regimes (DTR) (Laber et al., 2014) applied commonly in medical decision-making, which focuses on the statistical properties of polices on finite horizon (such as two-period). Our infinite-horizon results on the optimal value and Q-value distinguishes our developments from the DTR literature. Moreover, our result on the non-unique policy case can be demonstrated to correspond to the "non-regularity" concept in DTR, where the true parameters are very close to the decision "boundaries" that switch the optimal policy (motivated by situations of small treatment effects), thus making the obtained policy highly sensitive to estimation noises.

In the rest of this paper, we first describe our MDP setup and notations (Section 2). Then we present our results on large-sample behaviors (Section 3), demonstrate their use in exploration strategies (Section 4), and finally substantiate our findings with experimental results (Section 5). In the Appendix, we first present generalizations of our theoretical results to constrained MDP (A.1) and problems using approximate value iteration (A.2). Then we include more numerical experiments (B), followed by all the proofs (C).

# 2 PROBLEM SETUP

Consider an infinite horizon discounted reward MDP,  $\mathcal{M} = (\mathcal{S},\mathcal{A},R,P,\gamma ,\rho)$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $R(s,a)$  denotes the random reward when the agent is in state  $s\in S$  and selects action  $a\in \mathcal{A}$ ,  $P(s^{\prime}|s,a)$  is the probability of transitioning to state  $s^\prime$  in the next epoch given current state  $s$  and taken action  $a$ ,  $\gamma$  is the discount factor, and  $\rho$  is the initial state distribution. The distribution of the reward  $R$  and the transition probability  $P$  are unknown to the agent. We assume both  $\mathcal{S}$  and  $\mathcal{A}$  are finite sets. Without loss of generality, we denote  $\mathcal{S} = \{1,2,\ldots ,m_s\}$  and  $\mathcal{A} = \{1,2,\dots,m_a\}$ . Finally, we make the following stochasticity assumption: Assumption 1.  $R(s,a)$  has finite mean  $\mu_R(s,a)$  and finite variance  $\sigma_R^2 (s,a)\forall s\in S,a\in \mathcal{A}$ . For any given  $s\in S$  and  $a\in \mathcal{A}$ ,  $R(s,a)$  and  $S^{\prime}\sim P(\cdot |s,a)$  are all independent random variables.

A policy  $\pi$  is a mapping from each state  $s\in S$  to a probability measure over actions  $a\in \mathcal{A}$ . Specifically, we write  $\pi (a|s)$  as the probability of taking action  $a$  when the agent is in state  $s$  and  $\pi (\cdot |s)$  as the  $m_{a}$ -dimensional vector of action probabilities at state  $s$ . For convenience, we sometimes write  $\pi (s)$  as the

realized action given the current state is  $s$ . The value function associated with a policy  $\pi$  is defined as  $V^{\pi}(s) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^{t} R(s_{t}, \pi(s_{t})) | s_{0} = s]$  with  $s_{t+1} \sim P(|s_{t}, \pi(s_{t}))$ . The expected value function, under the initial distribution  $\rho$ , is denoted by  $\chi^{\pi} = \sum_{s} \rho(s) V^{\pi}(s)$ . A policy  $\pi^{*}$  is said to be optimal if  $V^{\pi^{*}}(s) = \max_{\pi} V^{\pi}(s)$  for all  $s \in S$ . For convenience, we denote  $V^{*} = V^{\pi^{*}}$  and  $\chi^{*} = \sum_{s} \rho(s) V^{*}(s)$ . The Q-value, denoted by  $Q(s, a)$ , is defined as  $Q(s, a) = \mu_{R}(s, a) + \gamma \mathbb{E}[V^{*}(S') | s, a]$ . Correspondingly,  $V^{*}(s) = \max_{a} Q(s, a)$  and the Bellman equation for  $Q$  takes the form

$$
Q (s, a) = \mu_ {R} (s, a) + \gamma \mathbb {E} \left[ \max  _ {a ^ {\prime}} Q \left(s ^ {\prime}, a ^ {\prime}\right) | s, a \right], \tag {1}
$$

for any  $(s,a)\in \mathcal{S}\times \mathcal{A}$ . Denoting the Bellman operator as  $\mathcal{T}_{\mu_R,P}(\cdot)$ ,  $Q$  is a fixed point associated with  $\mathcal{T}_{\mu_R,P}$ , i.e.  $Q = \mathcal{T}_{\mu_R,P}(Q)$ .

For the most part of this paper we make the following assumption about  $Q$ :

Assumption 2. For any state  $s \in S$ ,  $\arg \max_{a \in \mathcal{A}} Q(s, a)$  is unique.

Under Assumption 2, the optimal policy  $\pi^{*}$  is unique and deterministic. Let  $a^{*}(s) = \arg \max_{a\in \mathcal{A}}Q(s,a)$ . Then  $\pi^{*}(a|s) = \mathbb{1}$  ( $a = a^{*}(s)$ ), where  $\mathbb{1}(\cdot)$  denotes the indicator function.

We next introduce some statistical quantities arising from data. Suppose we have  $n$  observations (whose collection mechanism will be made precise later), which we denote as  $\{(s_t, a_t, r_t(s_t, a_t), s_t'(s_t, a_t)): 1 \leq t \leq n\}$ , where  $r_t(s_t, a_t)$  is the realized reward at time  $t$  and  $s_t'(s_t, a_t) = s_{t+1}$ . We define the sample mean  $\hat{\mu}_{R,n}$  and the sample variance  $\hat{\sigma}_{R,n}^2$  of the reward as

$$
\hat {\mu} _ {R, n} (s = i, a = j) = \frac {\sum_ {1 \leq t \leq n} r _ {t} \left(s _ {t} , a _ {t}\right) \mathbb {1} \left(s _ {t} = i , a _ {t} = j\right)}{\sum_ {1 \leq t \leq n} \mathbb {1} \left(s _ {t} = i , a _ {t} = j\right)}, \tag {2}
$$

$$
\hat {\sigma} _ {R, n} ^ {2} (s = i, a = j) = \frac {\sum_ {1 \leq t \leq n} r _ {t} \left(s _ {t} , a _ {t}\right) ^ {2} \mathbb {1} \left(s _ {t} = i , a _ {t} = j\right)}{\sum_ {1 \leq t \leq n} \mathbb {1} \left(s _ {t} = i , a _ {t} = j\right)} - \hat {\mu} _ {R, n} (i, j) ^ {2}. \tag {3}
$$

Similarly, we define the empirical transition matrix  $\tilde{P}_n$  as

$$
\hat {P} _ {n} \left(s ^ {\prime} = k \mid s = i, a = j\right) = \frac {\sum_ {1 \leq t \leq n} \mathbb {1} \left(s _ {t} = i , a _ {t} = j , s _ {t} ^ {\prime} \left(s _ {t} , a _ {t}\right) = k\right)}{\sum_ {1 \leq t \leq n} \mathbb {1} \left(s _ {t} = i , a _ {t} = j\right)} \tag {4}
$$

and its  $m_s \times m_s$  sampling covariance matrix  $\Sigma_{P_{s,a}}$  (with one sample point of  $\mathbb{1}(s_t = s, a_t = a)$ ) as

$$
\Sigma_ {P _ {s, a}} (k _ {1}, k _ {2}) = \left\{ \begin{array}{l} P (k _ {1} | s, a) (1 - P (k _ {1} | s, a)) \quad k _ {1} = k _ {2} \\ - P (k _ {1} | s, a) P (k _ {2} | s, a) \quad k _ {1} \neq k _ {2}. \end{array} \right., \text {f o r} 1 \leq k _ {1} \leq m _ {s}, 1 \leq k _ {2} \leq m _ {s}.
$$

With the data, we construct our estimate of  $Q$ , called  $\hat{Q}_n$ , which is the empirical fixed point of  $\mathcal{T}_{\hat{\mu}_{R,n},\hat{P}_n}$ , i.e.  $\hat{Q}_n = \mathcal{T}_{\hat{\mu}_{R,n},\hat{P}_n}(\hat{Q}_n)$ . Correspondingly, we also write  $\hat{V}_n^* (s) = \max_{a\in \mathcal{A}}\hat{Q}_n(s,a)$  and  $\hat{\chi}_n^* = \sum_{s\in \mathcal{S}}\rho (s)\hat{V}_n^* (s)$ .

We shall focus on the empirical errors due to noises of the collected data, and assume the MDP or Q-value evaluation can be done off-line so that the fixed point equation for  $\hat{Q}_n$  can be solved exactly.

# 3 QUANTIFYING ASYMPTOTIC ESTIMATION ERRORS

We present an array of results regarding the asymptotic behaviors of  $\hat{Q}_n$  and  $\hat{V}_n^*$ . To prepare, we first make an assumption on our exploration policy  $\pi$  to gather data. Define the extended transition probability  $\hat{P}^{\pi}$  as  $\tilde{P}^{\pi}(s', a'|s, a) = P(s'|s, a)\pi(a'|s')$ . We make the assumption:

Assumption 3. The Markov chain with transition probability  $\tilde{P}^{\pi}$  is positive recurrent.

Under Assumption 3,  $\tilde{P}^{\pi}$  has a unique stationary distribution, denoted  $w$ , equal to the long run frequency in visiting each state-action pair, i.e.  $w(s,a) = \lim_{n\to \infty}\frac{1}{n}\sum_{1\leq t\leq n}\mathbb{1}(s_t = i,a_t = j)$ , where all  $w(s,a)$ 's are positive. Note that Assumption 3 is satisfied if for any two states  $s,s'$ , there exists a sequence of actions such that  $s'$  is attainable from  $s$  under  $P$ , and, moreover, if  $\pi$  is sufficiently mixed, e.g.,  $\pi$  satisfies  $\pi (a'|s') > 0$  for all  $s',a'$ .

Our results in the sequel use the following further notations. We denote “ $\Rightarrow$ ” as “convergence in distribution”, and  $\mathcal{N}(\mu, \Sigma)$  as a multivariate Gaussian distribution with mean vector  $\mu$  and covariance matrix  $\Sigma$ . We write  $I$  as the identity matrix, and  $e_i$  as the  $i$ -th unit vector. The dimension of  $\mathcal{N}(\mu, \Sigma)$ ,  $I$  and  $e_i$  should be clear from the context. When not specified, all the vectors are column vectors. Let  $N = m_s m_a$ . In our algebraic derivations, we need to re-arrange  $\mu_R$ ,  $Q$  and  $w$  as  $N$ -dimensional vectors. We thus define the following indexing rule:  $(s = i, a = j)$  is re-indexed as  $(i - 1)m_a + j$ , e.g.  $\mu_R(i, j) = \mu_R((i - 1)m_a + j)$ . We also need to re-arrange  $\tilde{P}^\pi$  as an  $N \times N$  matrix following the same indexing rule, i.e.  $\tilde{P}^\pi (i', j'|i, j) = \tilde{P}^\pi ((i - 1)m_a + j, (i' - 1)m_a + j')$ .

# 3.1 LIMIT THEOREMS UNDER SUFFICIENT EXPLORATION

We first establish the asymptotic normality of  $\hat{Q}_n$  under exploration policy  $\pi$ :

Theorem 1. Under Assumptions 1 and 2, if the data is collected according to  $\pi$  satisfying Assumption 3, then  $\hat{Q}_n$  is a strongly consistent estimator of  $Q$ , i.e.  $\hat{Q}_n \to Q$  almost surely as  $n \to \infty$ . Moreover,

$$
\sqrt {n} (\hat {Q} _ {n} - Q) \Rightarrow \mathcal {N} (0, \Sigma) \quad a s \quad n \rightarrow \infty ,
$$

where

$$
\Sigma = (I - \gamma \tilde {P} ^ {\pi^ {*}}) ^ {- 1} W ^ {- 1} \left(D _ {R} + D _ {Q}\right) \left(\left(I - \gamma \tilde {P} ^ {\pi^ {*}}\right) ^ {- 1}\right) ^ {T}, \tag {5}
$$

$W$ ,  $D_R$  and  $D_Q$  are  $N \times N$  diagonal matrices with

$$
\begin{array}{l} W ((i - 1) m _ {a} + j, (i - 1) m _ {a} + j) = w (i, j), \quad D _ {R} ((i - 1) m _ {a} + j, (i - 1) m _ {a} + j) = \sigma_ {R} ^ {2} (i, j) \\ a n d \quad D _ {Q} ((i - 1) m _ {a} + j, (i - 1) m _ {a} + j) = (V ^ {*}) ^ {T} \Sigma_ {P _ {i, j}} V ^ {*} r e s p e c t i v e l y. \\ \end{array}
$$

In addition to the asymptotic Gaussian behavior, a key element of Theorem 1 is the explicit form of the asymptotic variance  $\Sigma$ . This is derived from the delta method (Serfling, 2009) and, intuitively, is the product of the sensitivities (i.e., gradient) of  $Q$  with respect to its parameters and the variances of the parameter estimates. Here the parameters are  $\mu_R$  and  $P$ , with corresponding gradients  $(I - \gamma \tilde{P}^{\pi^*})^{-1}$  and  $(I - \gamma \tilde{P}^{\pi^*})^{-1}V^*$ . The variances of these parameter estimates (i.e., (2) and (4)) involve  $\sigma_R^2(i,j)$  and  $\Sigma_{P_{i,j}}$ , and the sample size allocated to estimate each parameter, which is proportional to  $w(i,j)$ .

Using the relations that  $V_{n}^{*}(s) = \max_{a\in \mathcal{A}}Q(s,a)$  and  $\hat{V}_n^* (s) = \max_{a\in \mathcal{A}}\hat{Q}_n(s,a)$ , we can leverage Theorem 1 to further establish the asymptotic normality of  $\hat{V}_n^*$  and  $\hat{\chi}_n^*$ :

Corollary 1. Under Assumptions 1, 2 and 3,

$$
\sqrt {n} (\hat {V} _ {n} ^ {*} - V ^ {*}) \Rightarrow \mathcal {N} (0, \Sigma_ {V}) a n d \sqrt {n} (\hat {\chi} _ {n} ^ {*} - \chi^ {*}) \Rightarrow \mathcal {N} (0, \sigma_ {\chi} ^ {2}) a s n \to \infty
$$

where

$$
\Sigma_ {V} = \left(I - \gamma P ^ {\pi^ {*}}\right) ^ {- 1} \left(W ^ {\pi^ {*}}\right) ^ {- 1} \left[ D _ {R} ^ {\pi^ {*}} + D _ {V} ^ {\pi^ {*}} \right] \left(\left(I - \gamma P ^ {\pi^ {*}}\right) ^ {- 1}\right) ^ {T},
$$

$\sigma_{\chi}^{2} = \rho^{T}\Sigma_{V}\rho$ $P^{\pi^{*}}$  is an  $m_{s}\times m_{s}$  transition matrix with  $P^{\pi^{*}}(i,j) = P(j|s = i,a = a^{*}(s))$ $W^{\pi^{*}}$ $D_R^{\pi^*}$  and  $D_V^{\pi^*}$  are  $m_s\times m_s$  diagonal matrices with  $W^{\pi^{*}}(i,i) = w(i,a^{*}(i))$ $D_R^{\pi^*}(i,i) = \sigma_R^2 (i,a^* (i))$  and  $D_V^{\pi^*}(i,i) = (V^*)^T\Sigma_{P_{i,a^* (i)}}V^*$  respectively.

In the Appendix we also prove, using the same technique as above, a result on the large-sample behavior of the value function for a fixed policy (Corollary 2), which essentially recovers Corollary 4.1 in Mannor et al. (2007). Different from Mannor et al. (2007), we derive our results by using an implicit function theorem on the corresponding Bellman equation to obtain the gradient of  $Q$ , viewing the latter as the solution to the equation and as a function of  $\mu_R, P$ . This approach is able to generalize the results for fixed policies in Mannor et al. (2007) to the optimal value functions, and also provide distributional statements as Theorem 1 and Corollary 1 above. We also note that another potential route to obtain our results is to conduct perturbation analysis on the linear program (LP) representation of the MDP, which would also give gradient information of  $V^*$  (and hence also  $Q$ ), but using the implicit function theorem here seems sufficient.

Theorem 1 and Corollary 1 can be used immediately for statistical inference. In particular, we can construct confidence regions for subsets of the  $Q$ -value jointly, or for linear combinations of the  $Q$ -values. A quantity of interest that we will later utilize in designing good exploration policies is  $Q(s,a_{1}) - Q(s,a_{2})$ , i.e. the difference between action  $a_{1}$  and  $a_{2}$  when the agent is in state  $s$ . Define  $\sigma_{\Delta Q}^{2}$  as

$$
\sigma_ {\Delta Q} ^ {2} (s, a _ {1}, a _ {2}) = \left(e _ {(s - 1) m _ {a} + a _ {1}} - e _ {(s - 1) m _ {a} + a _ {2}}\right) ^ {T} \Sigma \left(e _ {(s - 1) m _ {a} + a _ {1}} - e _ {(s - 1) m _ {a} + a _ {2}}\right) \tag {6}
$$

and its estimator  $\hat{\sigma}_{\Delta Q,n}^2$  by replacing  $Q$ ,  $V^{*}$ ,  $\sigma_{R,n}^{2}$ ,  $w$ ,  $P$  with  $\hat{Q}_n,\hat{V}_n^*$ $\hat{\sigma}_{R,n}^2$ ,  $\hat{w}_n$ ,  $\hat{P}_n$  in  $\Sigma$ , where  $\hat{w}_n$  is the empirical frequency of visiting each state-action pair, i.e.  $\hat{w}_n(i,j) = \frac{1}{n}\sum_{1\leq t\leq n}\mathbb{1}(s_t = i,a_t = j)$ . Then the  $100(1 - \alpha)\%$  confidence interval (CI) for  $Q(s,a_1) - Q(s,a_2)$  takes the form  $\left(\hat{Q}_n(s,a_1) - \hat{Q}_n(s,a_2)\right)\pm z_\alpha \hat{\sigma}_{\Delta Q,n}^2 (s,a_1,a_2)$ , where  $z_{\alpha}$  is the  $(1 - \alpha /2)$ -quantile of  $\mathcal{N}(0,1)$ .

# 3.2 NON-UNIQUE OPTIMAL POLICY

Suppose the optimal policy for the MDP  $\mathcal{M}$  is not unique, i.e., Assumption 2 does not hold. In this situation, the estimated  $\hat{Q}_n$  and  $\hat{V}_n^*$  may "jump" around different optimal actions, leading to a more complicated large-sample behavior as described below:

Theorem 2. Suppose Assumptions 1 and 3 hold but there is no unique optimal policy. Then there exists  $K \geq 1$  distinct  $m_s \times (N m_s + N)$  matrices  $\{G_k\}_{1 \leq k \leq K}$  and a deterministic partition of  $U = \{u \in \mathcal{R}^{m_s N + m_s} : ||u|| = 1\} = \cup_{1 \leq k \leq K} U_k$  such that  $\sqrt{n} (\hat{V}_n^* - V^*) \Rightarrow \sum_{k=1}^{K} G_k \mathbb{1}(Z / ||Z|| \in U_k)$ $Z$ , where  $Z = \mathcal{N}(0, \Sigma_{R,P})$ ,  $\Sigma_{R,P} = \text{Diag}(W^{-1}D_R, D_P)$  and  $D_P = \text{Diag}(\Sigma_{P_{1,1}} / w(0m_a + 1), \ldots, \Sigma_{P_{i,j}} / w((i - 1)m_a + j), \ldots, \Sigma_{P_{m_s, m_a}} / w((m_s - 1)m_a + m_a))$ .

In the case that  $K > 1$  in Theorem 2, the limit distribution becomes non-Gaussian. This arises because the sensitivity to  $P$  or  $\mu_R$  can be very different depending on the perturbation direction, which is a consequence of solution non-uniqueness that can be formalized as a non-degeneracy in the LP representation of the MDP. We note that this phenomenon is analogous to the "non-regularity" concept in DTR that arises because the "true" parameters in these problems are very close to the decision "boundaries", which makes the obtained policy highly sensitive to estimation noises and incurs a  $1/\sqrt{n}$ -order bias behavior. Our case of non-unique optimal policy here captures precisely this same behavior, where we see in Theorem 2 that when  $K > 1$  the asymptotic limit no longer has mean zero and consequently a  $1/\sqrt{n}$ -order bias arises.

We also develop two other generalizations of large-sample results, for constrained MDP and approximate value iteration respectively (see Appendices A.1 and A.2).

# 4 EFFICIENT EXPLORATION POLICY

We utilize our results in Section 3 to design exploration policies. We focus on the setting where an agent is assigned a period to collect data by running the state transition with an exploration policy. The goal is to obtain the best policy at the end of the period in a probabilistic sense, i.e., minimize the probability of selecting a suboptimal policy for the accumulated reward.

We propose a strategy that maximizes the worst-case relative discrepancy among all Q-value estimates. More precisely, we define, for  $i \in S$ ,  $j \in \mathcal{A}$  and  $j \neq a^{*}(i)$ , the relative discrepancy as

$$
h _ {i j} = \left(Q (i, a ^ {*} (i)) - Q (i, j)\right) ^ {2} / \sigma_ {\Delta Q} ^ {2} (i, a ^ {*} (i), j),
$$

where  $\sigma_{\Delta Q}^2 (i,a^* (i),j)$  is defined in (6). Our procedure attempts to maximize the minimum of  $h_{ij}$ 's,

$$
\max  _ {w \in \mathcal {W} _ {\eta}} \min  _ {i \in \mathcal {S}} \min  _ {j \in \mathcal {A}, j \neq a ^ {*} (i)} h _ {i j}, \tag {7}
$$

where  $w$  denotes the proportions of visits on the state-action pairs, within some allocation set  $\mathcal{W}_{\eta}$  (which we will explain). Intuitively,  $h_{ij}$  captures the relative "difficulty" in obtaining the optimal policy given the estimation errors of Q's. If the Q-values are far apart, or if the estimation variance is small, then  $h_{ij}$  is large which signifies an "easy" problem, and vice versa. Criterion (7) thus aims to make the problem the "easiest". Alternatively, one can also interpret (7) from a large deviations view (Glynn & Juneja, 2004; Dong & Zhu, 2016). Suppose the Q-values for state  $i$  between two different actions  $a^{*}(i)$  and  $j$  are very close. Then, one can show that the probability of suboptimal selection between the two has roughly an exponential decay rate controlled by  $h_{ij}$ . Obviously, there can be many more comparisons to consider, but the exponential form dictates that the smallest decay rate dominates the calculation, thus leading to the inner min's in (7). Criterion like (7) is motivated from the OCBA procedure in simulation optimization (which historically has considered simple mean-value alternatives (Chen & Lee, 2011)). Here, we consider the Q-values. For convenience, we call our procedure Q-OCBA.

Implementing criterion (7) requires two additional considerations. First, solving (7) needs the model primitives  $Q$ ,  $P$  and  $\sigma_R^2$  that appear in the expression of  $h_{ij}$ . These quantities are unknown a priori, but as we collect data they can be sequentially estimated. This leads to a multi-stage optimization plus parameter update scheme. Second, since data are collected through running a Markov chain on the exploration actions, not all allocation  $w$  is admissible, i.e., realizable as the stationary distribution of the MDP. To resolve this latter issue, we will derive a convenient characterization for admissibility.

Call  $\pi (\cdot |s)$  admissible if the Markov Chain with transition probability  $\tilde{P}^{\pi}$ , defined for Assumption 3, is positive recurrent, and denote  $w_{\pi}$  as its stationary distribution. Define the set

$$
\mathcal {W} = \left\{w > 0: \sum_ {1 \leq j \leq m _ {a}} w ((i - 1) m _ {a} + j) = \sum_ {1 \leq k \leq m _ {s}} \sum_ {1 \leq l \leq m _ {a}} w ((k - 1) m _ {a} + l) P (i | k, l) \right.
$$

$\forall 1 \leq i \leq m_s, \sum_{1 \leq i \leq m_s} \sum_{1 \leq j \leq m_a} w((i - 1)m_a + j) = 1\}$ . The following provides a characterization of the set of admissible  $\pi$ :

Lemma 1. For any admission policy  $\pi$ ,  $w_{\pi} \in \mathcal{W}$ . For any  $w \in \mathcal{W}$ ,  $\pi_w$  with  $\pi_w(a = j|s = i) = w((i - 1)m_a + j) / (\sum_{k=1}^{m_a} w((i - 1)m_a + k))$  is an admissible policy.

In other words, optimizing over the set of admissible policies is equivalent to optimizing over the set of stationary distributions. The latter is much more tractable thanks to the linear structure of  $\mathcal{W}$ . In practice, we will use  $\mathcal{W}_{\eta} = \mathcal{W} \cap \{w \geq \eta\}$  for some small  $\eta > 0$  to ensure closedness of the set (our experiments use  $\eta = 10^{-6}$ ).

Algorithm 1 describes Q-OCBA. In our experiments shown next, we simply use two stages, i.e.,  $K = 2$ . Finally, we also note that criterion like (7) can be modified according to the decision goal.

For example, if one is interested in obtaining the best estimate of  $\chi^{*}$ , then it would be more beneficial to consider  $\min_{w\in \mathcal{W}_{\eta}}\sigma_{\chi}^{2}$ . We showcase this with additional experiments in the Appendix.

Input: Number of iterations  $K$ , length of each batch  $\{B_k\}_{1 < k < K}$ , initial exploration policy  $\pi_0$ ;

Initialization:  $k = 0$

while  $k <   K$  do

- Run  $\pi_k$  for  $B_k$  steps and set  $k = k + 1$ ;
- Calculate  $\hat{P}_{B_k}$ ,  $\hat{\mu}_{R,B_k}, \hat{\sigma}_{R,B_k}^2$  and  $\hat{w}_{B_k}$  based on the  $B_k$  data points collected;
- Apply value-iteration using  $\hat{P}_{B_k}$  and  $\hat{\mu}_{R,B_k}^2$  to obtain  $\hat{Q}_{B_k}$ ;
- Plug the estimates  $\hat{P}_{B_k}$ ,  $\hat{\sigma}_{R,B_k}^2$  and  $\hat{Q}_{B_k}$  into (7) to solve for the optimal  $w_k$ ;
- Set  $\pi_k(a = j|s = i) = w_k((i - 1)m_a + j) / \sum_{l=1}^{m_a} w_k((i - 1)m_a + l)$ ;

end

# Algorithm 1: Q-OCBA sequential updating rule for exploration

Note that (7) is equivalent to  $\min_w\max_{i\in S}\max_{j\in \mathcal{A},j\neq a^* (i)}\sum_{s,a}c_{ij}(s,a) / w_{s,a}$  subject to  $w\in \mathcal{W}_{\eta}$  where  $c_{ij}(s,a)$ 's are non-negative coefficients. Based on the closed-form characterization of  $\Sigma$  in Theorem 1,  $c_{ij}(s,a)$ 's can be estimated with plug-in estimators using data collected in earlier stages.

# 5 NUMERICAL EXPERIMENTS

We conduct several numerical experiments to support our large-sample results in Sections 3 and demonstrate the performance of Q-OCBA against some benchmark methods. We use the RiverSwim problem in (Osband et al., 2013) with  $m_{s}$  states and two actions at each state: swim left (0) or swim right (1) (see Figure 1). The triplet above each arc represents i) the action, 0 or 1, ii) the transition probability to the next state given the current state and action, iii) the reward under the current state and action. Note that, in this problem, rewards are given only at the left and right boundary states (where the value of  $r_{L}$  will be varied). We consider the infinite horizon setting with  $\gamma = 0.95$  and  $\rho = [1 / m_s,\dots ,1 / m_s]^T$ .

![](images/35a0ca2bd8c98c1a3881a9642640885085fb846aabf4cb0883d7fd410a159178.jpg)  
Figure 1: RiverSwim Problem

We first demonstrate the validity of our large-sample results. We use a policy that swims right with probability 0.8 at each state, i.e.  $\pi(1|s) = 0.8$ . Tables 1 and 2 show the coverage rates of the constructed  $95\%$  CIs, for a small  $m_s = 6$  (using Theorem 1 and Corollary 1) and a large  $m_s = 31$  (using Theorem 4 in the Appendix) respectively. The latter case uses a linear interpolation with  $\mathcal{S}_0 = \{1, 4, \dots, 28, 31\}$ . All coverage rates are estimated using  $10^3$  independent experimental repetitions (the bracketed numbers in the tables show the half-widths of  $95\%$  CI for the coverage estimates). For the Q-values, we report the average coverage rate over all  $(s, a)$  pairs. When the number of observations  $n$  is large enough ( $\geq 3 \times 10^4$  for exact update and  $\geq 10^5$  for interpolation), we see highly accurate CI coverages, i.e., close to  $95\%$ .

Table 1: Exact tabular update  

<table><tr><td>n</td><td>10^4</td><td>3 × 10^4</td><td>5 × 10^4</td></tr><tr><td>Q</td><td>0.77(0.03)</td><td>0.93(0.02)</td><td>0.96(0.01)</td></tr><tr><td>χπ*</td><td>0.77(0.03)</td><td>0.93(0.02)</td><td>0.96(0.01)</td></tr></table>

Table 2: Approximate value iteration  

<table><tr><td>n</td><td>10^4</td><td>10^5</td><td>10^6</td></tr><tr><td>Q</td><td>0.53(0.02)</td><td>0.95(0.01)</td><td>0.95(0.01)</td></tr><tr><td>χπ*</td><td>0.80(0.03)</td><td>0.94(0.02)</td><td>0.95(0.01)</td></tr></table>

Next we investigate the efficiency of our exploration policy. We compare Q-OCBA with  $K = 2$  to four benchmark policies: i)  $\epsilon$ -greedy with different values of  $\epsilon$ , ii) random exploration (RE) with different values of  $\pi(1|s)$ , iii) UCRL2 (a variant of UCRL) with  $\delta = 0.05$  (Jaksch et al., 2010), iv) PSRL with different posterior updating frequencies (Osband et al., 2013), i.e.,  $\mathrm{PSRL}(x)$  means PSRL is implemented with  $x$  episodes. We use  $m_s = 6$  and vary  $r_L$  from 1 to 3. To ensure fairness, we use a two-stage implementation for all policies, with  $30\%$  of iterations first dedicated to RE (with  $\pi(1|s) = 0.6$ ) as a warm start, i.e., the data are used to estimate the parameters needed for the second stage. To give enough benefit of the doubt, we notice the probabilities of correct selection for both UCRL2 and PSRL are much worse without the warm start.

Tables 3 and 4 compare the probabilities of obtaining the optimal policy (based on the estimated  $\hat{Q}_n$ 's). For  $\epsilon$ -greedy, RE, and PSRL, we report the results with the parameters that give the best performances in our numerical experiments. The probability of correct selection is estimated using  $10^{3}$  replications of the procedure. We observe that Q-OCBA substantially outperforms the other methods, both with a small data size ( $n = 10^{3}$  in Table 3) and a larger one ( $n = 10^{4}$  in Table 4). Generally, these benchmark policies perform worse for larger values of  $r_L$ . This is because for small  $r_L$ , the  $(s,a)$  pairs that need to be explored more also tend to have larger  $Q$ -values. However, as  $r_L$  increase, there is a misalignment between the  $Q$ -values and the  $(s,a)$  pairs that need more exploration.

The superiority of our Q-OCBA in these experiments come as no surprise to us. The benchmark methods like UCRL2 and PSRL are designed to minimize regret which involves balancing the exploration-exploitation trade-off. On the other hand, Q-OCBA focuses on efficient exploration only, i.e., our goal is to minimize the probability of incorrect policy selection, and this is achieved by carefully utilizing the variance information gathered from the first stage that is made possible by our derived asymptotic formulas. We provide additional numerical results in Appendix B.

Table 3: Probability of correct selection for different exploration policies,  $n = {10}^{3}$  

<table><tr><td>rL</td><td>0.2-greedy</td><td>RE(0.6)</td><td>UCRL2</td><td>PSRL(100)</td><td>Q-OCBA</td></tr><tr><td>1</td><td>0.95(0.01)</td><td>0.70(0.03)</td><td>0.44(0.03)</td><td>0.53(0.03)</td><td>0.87(0.02)</td></tr><tr><td>2</td><td>0.15(0.02)</td><td>0.29(0.03)</td><td>0.11(0.02)</td><td>0.33(0.03)</td><td>0.55(0.03)</td></tr><tr><td>3</td><td>0.00(0.00)</td><td>0.45(0.03)</td><td>0.21(0.02)</td><td>0.41(0.03)</td><td>0.84(0.02)</td></tr></table>

Table 4: Probability of correct selection for different exploration policies,  $n = 10^4$  

<table><tr><td>rL</td><td>0.2-greedy</td><td>RE(0.6)</td><td>UCRL2</td><td>PSRL(100)</td><td>Q-OCBA</td></tr><tr><td>1</td><td>1.00(0.00)</td><td>0.95(0.01)</td><td>0.82(0.02)</td><td>1.00(0.00)</td><td>1.00(0.00)</td></tr><tr><td>2</td><td>0.55(0.03)</td><td>0.80(0.03)</td><td>0.52(0.03)</td><td>0.94(0.02)</td><td>1.00(0.00)</td></tr><tr><td>3</td><td>0.21(0.03)</td><td>0.94(0.01)</td><td>0.75(0.03)</td><td>0.76(0.03)</td><td>1.00(0.00)</td></tr></table>

# REFERENCES

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 22-31. JMLR.org, 2017.  
Eitan Altman. Constrained Markov Decision Processes, volume 7. CRC Press, 1999.  
Jean-Yves Audibert and Sébastien Bubeck. Best arm identification in multi-armed bandits. In _COLT-23th Conference on learning theory-2010_, pp. 13–p, 2010.  
Craig Boutilier and Tyler Lu. Budget allocation using weakly coupled, constrained markov decision processes. In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence, pp. 52-61. AUAI Press, 2016.  
Chun-hung Chen and Loo Hay Lee. Stochastic Simulation Optimization: An Optimal Computing Budget Allocation, volume 1. World scientific, 2011.  
Yinlam Chow, Mohammad Ghavamzadeh, Lucas Janson, and Marco Pavone. Risk-constrained reinforcement learning with percentile risk criteria. The Journal of Machine Learning Research, 18(1):6070-6120, 2017.  
Jing Dong and Yi Zhu. Three asymptotic regimes for ranking and selection with general sample distributions. In Proceedings of the 2016 Winter Simulation Conference, pp. 277-288. IEEE Press, 2016.  
Eugene A Feinberg and Uriel G Rothblum. Splitting randomized stationary policies in total-reward markov decision processes. Mathematics of Operations Research, 37(1):129-153, 2012.  
Peter Glynn and Sandeep Juneja. A large deviations perspective on ordinal optimization. In Proceedings of the 36th conference on Winter Simulation Conference, pp. 577-585. Winter Simulation Conference, 2004.  
Geoffrey J Gordon. Stable function approximation in dynamic programming. In Machine Learning Proceedings 1995, pp. 261-268. Elsevier, 1995.  
T. Jaksch, R. Ortner, and P. Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(Apr):1563-1600, 2010.  
S. M. Kakade. On the sample complexity of reinforcement learning. PhD Thesis, University College London, 2003.  
Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation. arXiv preprint arXiv:1806.10293, 2018.  
M. Kearns and S. Singh. Finite-sample convergence rates for Q-learning and indirect algorithms. In Proceedings of the conference on Advances in neural information processing systems II, pp. 996-1002, 1998.  
Eric B Laber, Daniel J Lizotte, Min Qian, William E Pelham, and Susan A Murphy. Dynamic treatment regimes: Technical challenges and applications. *Electronic journal of statistics*, 8(1): 1225, 2014.

Shie Mannor, Duncan Simester, Peng Sun, and John N Tsitsiklis. Bias and variance in value function estimation. In Proceedings of the twenty-first international conference on Machine learning, pp. 72. ACM, 2004.  
Shie Mannor, Duncan Simester, Peng Sun, and John N Tsitsiklis. Bias and variance approximation in value function estimates. Management Science, 53(2):308-322, 2007.  
Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 9(May):815-857, 2008.  
Ian Osband, Daniel Russo, and Benjamin Van Roy. (More) efficient reinforcement learning via posterior sampling. In Advances in Neural Information Processing Systems, pp. 3003-3011, 2013.  
Robert J Serfling. Approximation Theorems of Mathematical Statistics, volume 162. John Wiley & Sons, 2009.
