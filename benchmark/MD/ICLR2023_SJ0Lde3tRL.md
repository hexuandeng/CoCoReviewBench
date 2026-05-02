# EXTREME Q-LEARNING: MAXENT RL WITHOUT ENTROPY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Modern Deep Reinforcement Learning (RL) algorithms require estimates of the maximal Q-value, which are difficult to compute in continuous domains with an infinite number of possible actions. In this work, we introduce a new update rule for online and offline RL which directly models the maximal value using Extreme Value Theory (EVT) inspired by Economics. By doing so, we avoid computing Q-values using out-of-distribution actions which is often a substantial source of error. Our key insight is to introduce an objective that directly estimates the optimal soft-value functions (LogSumExp) in the maximum entropy (MaxEnt) RL setting without needing to sample from a policy. Using EVT, we derive our Extreme Q-Learning framework and consequently online and, for the first time, offline MaxEnt Q-learning algorithms, that do not explicitly require access to a policy or its entropy. Finally, our method obtains strong results in the Offline D4RL benchmark outperforming prior works by 10-20 points on some tasks while offering moderate improvements over SAC and TD3 on online DM Control tasks.

# 1 INTRODUCTION

Modern Deep Reinforcement Learning (RL) algorithms have shown broad success in challenging control (Haarnoja et al., 2018; Schulman et al., 2015) and game-playing domains (Mnih et al., 2013). While tabular Q-iteration or value-iteration methods are well understood, state of the art RL algorithms often make theoretical compromises in order to deal with deep networks, high dimensional state spaces, and continuous action spaces. In particular, standard Q-learning algorithms require computing the max or soft-max over the Q-function in order to fit the Bellman equations. Yet, almost all current off-policy RL algorithms for continuous control only indirectly estimate the Q-value of the next state with separate policy networks. Consequently, these methods only estimate the Q-function of the current policy, instead of the optimal  $Q^{*}$ , and rely on policy improvement via an actor. Moreover, actor-critic approaches on their own have shown to be catastrophic in the offline settings where actions sampled from a policy are consistently out-of-distribution (Kumar et al., 2020; Fujimoto et al., 2018). As such, computing max  $Q$  for Bellman targets remains a core issue in deep RL.

One popular approach is to train Maximum Entropy (MaxEnt) policies, in hopes that they are more robust to modeling and estimation errors (Ziebart, 2010). However, the Bellman backup  $\mathcal{B}^*$  used in MaxEnt RL algorithms still requires computing the log-partition function over Q-values, which is usually intractable in high-dimensional action spaces. Instead, current methods like SAC (Haarnoja et al., 2018) rely on auxiliary policy networks, and as a result do not estimate  $\mathcal{B}^*$ , the optimal Bellman backup. Our key insight is to apply extreme value analysis used in branches of Finance and Economics to Reinforcement Learning. Ultimately, this will allow us to directly model the LogSumExp over Q-functions in the MaxEnt Framework.

Intuitively, reward or utility-seeking agents will consider the maximum of the set of possible future returns. The Extreme Value Theorem (EVT) tells us that maximal values drawn from any exponential tailed distribution follows the Generalized Extreme Value (GEV) Type-1 distribution, also referred to as the Gumbel Distribution  $\mathcal{G}$ . The Gumbel distribution is thus a prime candidate for modeling errors in Q-functions. In fact, McFadden's 2000 Nobel-prize winning work in Economics on discrete choice models (McFadden, 1972) showed that soft-optimal utility functions with logit (or softmax) choice probabilities naturally arise when utilities are assumed to have Gumbel-distributed errors. This was subsequently generalized to stochastic MDPs by Rust (1986) in 1984. Nevertheless, these

intriguing results have remained largely unknown in the RL community. By introducing a novel loss optimization framework, we bring them into the world of modern deep RL.

Empirically, we find that even modern deep RL approaches, for which errors are typically assumed to be Gaussian, exhibit errors that better approximate the Gumbel Distribution, see Figure 1. By assuming errors to be Gumbel distributed, we obtain Gumbel Regression, a consistent estimator over log-partition functions even in continuous spaces. Furthermore, making this assumption about  $Q$ -values lets us derive a new Bellman loss objective that directly solves for the optimal MaxEnt Bellman operator  $\mathcal{B}^*$ , instead of the operator under the current policy  $\mathcal{B}^\pi$ . As soft optimality gracefully emerges from our framework, we can run MaxEnt RL independently of the policy. In the online setting, we avoid using a policy network to explicitly compute entropies. In the offline setting, we completely avoid sampling from learned policy networks, minimizing the aforementioned extrapolation error. Our resulting algorithms surpass state-of-the-art (SOTA) while being practically simpler.

In this paper we outline the theoretical motivation for using Gumbel distributions in reinforcement learning, and show how it can be used to derive practical online and offline MaxEnt RL algorithms. Concretely, our contributions are as follows:

- We motivate Gumbel Regression and show it allows calculation of the log-partition function (LogSumExp) in continuous spaces. We apply it to MDPs to present a novel loss objective for RL using maximum-likelihood estimation.  
- Our formulation extends soft-Q learning to offline RL as well as continuous action spaces without the need of policy entropies. It presents a way to directly calculate the optimal soft-values  $V^{*}$  and soft-Bellman updates  $\mathcal{B}^{*}$  using SGD, which are usually intractable in continuous settings.  
- We provide the missing theoretical link between soft and conservative Q-learning, showing how these formulations can be made equivalent. We also show how Max-Ent RL emerges naturally from vanilla RL as a conservatism in our framework.  
- Finally, we empirically demonstrate state-of-art results in Offline RL, improving over prior methods by a large margin on the D4RL benchmark, and performing moderately better than SAC and TD3 in Online RL, while theoretically avoiding actor-critic formulations.

# 2 PRELIMINARIES

In this section we introduce Maximium Entropy (MaxEnt) RL and Extreme Value Theory (EVT), which we use to motivate our framework to estimate extremal values in RL.

We consider an infinite-horizon Markov decision process (MDP), defined by the tuple  $(S, \mathcal{A}, \mathcal{P}, r, \gamma)$ , where  $S, \mathcal{A}$  represent state and action spaces,  $\mathcal{P}(\mathbf{s}'|\mathbf{s}, \mathbf{a})$  represents the environment dynamics,  $r(\mathbf{s}, \mathbf{a})$  represents the reward function, and  $\gamma \in (0,1)$  represents the discount factor. In the offline RL setting, we are given a dataset  $\mathcal{D} = (\mathbf{s}, \mathbf{a}, r, \mathbf{s}')$  of tuples sampled from trajectories under a behavior policy  $\pi_{\mathcal{D}}$  without any additional environment interactions. We use  $\rho_{\pi}(\mathbf{s})$  to denote the distribution of states that a policy  $\pi(\mathbf{a}|\mathbf{s})$  generates. In the MaxEnt framework, an MDP with entropy-regularization is referred to as a soft-MDP (Bloem & Bambos, 2014) and we often use this notation.

# 2.1 MAXIMUM ENTROPY RL

Standard RL seeks to learn a policy that maximizes the expected sum of (discounted) rewards  $\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} r(\mathbf{s}_{t}, \mathbf{a}_{t})\right]$ , for  $(\mathbf{s}_{t}, \mathbf{a}_{t})$  drawn at timestep  $t$  from the trajectory distribution that  $\pi$  generates. We consider a generalized version of Maximum Entropy RL that augments the standard reward objective with the KL-divergence between the policy and a reference distribution  $\mu$ :  $\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^{t} (r(\mathbf{s}_{t}, \mathbf{a}_{t}) - \beta \log \frac{\pi(\mathbf{a}_{t} | \mathbf{s}_{t})}{\mu(\mathbf{a}_{t} | \mathbf{s}_{t})}\right]$ , where  $\beta$  is the regularization strength. When  $\mu$  is uniform  $\mathcal{U}$ , this becomes the standard MaxEnt objective used in online RL up to a constant. In the offline RL setting, we choose  $\mu$  to be the behavior policy  $\pi_{\mathcal{D}}$  that generated the fixed dataset  $\mathcal{D}$ . Consequently, this objective enforces a conservative KL-constraint on the learned policy, keeping it close to the behavior policy (Neu et al., 2017; Haarnoja et al., 2018).

In MaxEnt RL, the soft-Bellman operator  $\mathcal{B}^*:\mathbb{R}^{\mathcal{S}\times \mathcal{A}}\to \mathbb{R}^{\mathcal{S}\times \mathcal{A}}$  is defined as  $(\mathcal{B}^{*}Q)(\mathbf{s},\mathbf{a}) = r(\mathbf{s},\mathbf{a})+$ $\gamma \mathbb{E}_{\mathbf{s}'\sim \mathcal{P}(\cdot |\mathbf{s},\mathbf{a})}V^{*}(\mathbf{s}')$  where  $Q$  is the soft-Q function and  $V^{*}$  is the optimal soft-value satisfying:

$$
V ^ {*} (\mathbf {s}) = \beta \log \sum_ {\mathbf {a}} \mu (\mathbf {a} | \mathbf {s}) \exp (Q (\mathbf {s}, \mathbf {a}) / \beta) := \mathbb {L} _ {a \sim \mu (\cdot | \mathbf {s})} ^ {\beta} [ Q (\mathbf {s}, \mathbf {a}) ], \tag {1}
$$

where we denote the log-sum-exp (LSE) using an operator  $\mathbb{L}^{\beta}$  for succinctness<sup>1</sup>. The soft-Bellman operator has a unique contraction  $Q^{*}$  (Haarnoja et al., 2018) given by the soft-bellman equation:  $Q^{*} = \mathcal{B}^{*}Q^{*}$  and the optimal policy satisfies (Haarnoja et al., 2017):

$$
\pi^ {*} (\mathbf {a} | \mathbf {s}) = \mu (\mathbf {a} | \mathbf {s}) \exp \left(\left(Q ^ {*} (\mathbf {s}, \mathbf {a}) - V ^ {*} (\mathbf {s})\right) / \beta\right). \tag {2}
$$

Instead of estimating soft-values for a policy  $V^{\pi}(\mathbf{s}) = \mathbb{E}_{\mathbf{a}\sim \pi (\cdot |\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a}) - \beta \log \frac{\pi(\mathbf{a}|\mathbf{s})}{\mu(\mathbf{a}|\mathbf{s})}\right]$ , our approach will seek to directly fit the optimal soft-values  $V^{*}$ , i.e. the log-sum-exp (LSE) of Q values.

# 2.2 EXTREME VALUE THEOREM

The Fisher-Tippett or extreme value theorem tells us that the maximum of i.i.d. samples from exponentially tailed distributions will asymptotically converge to the Gumbel distribution  $\mathcal{G}(\mu, \beta)$ , which has pdf  $p(x) = \exp(-(z + e^{-z}))$  where  $z = (x - \mu) / \beta$  with location parameter  $\mu$  and scale parameter  $\beta$ .

Theorem 1 (Extreme Value Theorem (EVT) (Mood, 1950; Fisher & Tippett, 1928)). For i.i.d. random variables  $X_{1},\ldots ,X_{n}\sim f_{X}$ , with exponential tails,  $\lim_{n\to \infty}\max_i(X_i)$  follows the Gumbel (GEV-1) distribution. Furthermore,  $\mathcal{G}$  is max-stable, i.e. if  $X_{i}\sim \mathcal{G}$ , then  $\max_i(X_i)\sim \mathcal{G}$  holds.

This result is similar to the Central Limit Theorem (CLT), which states that means of i.i.d. errors approach the normal distribution. Thus, under a chain of max operations, any i.i.d. exponential tailed errors<sup>2</sup> will tend to become Gumbel distributed and stay as such. EVT will ultimately suggest us to characterize nested errors in Q-learning as following a Gumbel distribution. In particular, the Gumbel distribution  $\mathcal{G}$  exhibits unique properties we will exploit.

One intriguing consequence of the Gumbel's max-stability is its ability to convert the maximum over a discrete set into a softmax. This is known as the Gumbel-Max Trick (Papandreou & Yuille, 2010; Hazan & Jaakkola, 2012). Concretely for i.i.d.  $\epsilon_i \sim \mathcal{G}(0, \beta)$  added to a set  $\{x_1, \dots, x_n\} \in \mathbb{R}$ ,  $\max_i(x_i + \epsilon_i) \sim \mathcal{G}(\beta \log \sum_i \exp(x_i / \beta), \beta)$ , and  $\arg\max(x_i + \epsilon_i) \sim \operatorname{softmax}(x_i / \beta)$ . Furthermore, the Max-trick is unique to the Gumbel (Luce, 1977). These properties lead into the McFadden-Rust model (McFadden, 1972; Rust, 1986) of MDPs as we state below.

McFadden-Rust model: An MDP following the standard Bellman equations with stochasticity in the rewards due to unobserved state variables will satisfy the soft-Bellman equations over the observed state with actual rewards  $\bar{r} (\mathbf{s},\mathbf{a})$ , given two conditions:

1. Additive separability (AS): observed rewards have additive i.i.d. Gumbel noise, i.e.  $r(\mathbf{s},\mathbf{a}) = \bar{r} (\mathbf{s},\mathbf{a}) + \epsilon (\mathbf{s},\mathbf{a})$  , with actual rewards  $\bar{r} (\mathbf{s},\mathbf{a})$  and i.i.d. noise  $\epsilon (\mathbf{s},\mathbf{a})\sim \mathcal{G}(0,\beta)$  
2. Conditional Independence (CI): the noise  $\epsilon (\mathbf{s},\mathbf{a})$  in a given state-action pair is conditionally independent of that in any other state-action pair.

Moreover, the converse also holds: Any MDP satisfying the Bellman equations and following a softmax policy, necessarily has any i.i.d. noise in the rewards with  $AS + CI$  conditions be Gumbel distributed.

These results were first shown to hold in discrete choice theory by McFadden (1972), with the  $AS + CI$  conditions derived by Rust (1986) for discrete MDPs. We formalize these results in Appendix A and give succinct proofs using the developed properties of the Gumbel distribution. These results enable the view of a soft-MDP as an MDP with hidden i.i.d. Gumbel noise in the rewards.

Notably, this result gives a different interpretation of a soft-MDP than entropy regularization to allow us to recover the soft-Bellman equations.

# 3 EXTREME Q-LEARNING

In this section, we motivate our Extreme Q-learning framework, which directly models the soft-optimal values  $V^{*}$ , and show it naturally extends soft-Q learning. Notably, we use the Gumbel distribution to derive a new optimization framework for RL via maximum-likelihood estimation and apply it to both online and offline settings.

# 3.1 GUMBEL ERROR MODEL

Although assuming Gumbel errors in MDPs leads to intriguing properties, it is not obvious why the errors might be distributed as such. First, we empirically investigate the distribution of Bellman errors by computing them over the course of training. Specifically, we compute  $r(\mathbf{s},\mathbf{a}) - \gamma Q(\mathbf{s}',\pi (\mathbf{s}')) - Q(\mathbf{s},\mathbf{a})$  for samples  $(\mathbf{s},\mathbf{a},\mathbf{s}')$  from the replay-buffer using a single  $Q$ -function from SAC (Haarnoja et al., 2018) (See Appendix D for more details). In Figure 1, we find the errors to be skewed and better fit by a Gumbel distribution. We explain this using EVT.

Consider fitting  $Q$ -functions by learning an unbiased function approximator  $\hat{Q}$  to solve the Bellman equation. We will assume access to  $M$  such function approximators, each of which are assumed to be independent e.g.

![](images/0e5be366e59ac3b91edd86e45e8fe0b6335868ba29d2f68c864dbc3afc27f0be.jpg)  
Figure 1: Bellman errors from SAC on Cheetah-Run (Tassa et al., 2018). The Gumbel distribution better captures the skew versus the Gaussian. Plots for TD3 and more environments can be found in Appendix D.

parallel runs of a model over an experiment. We can see approximate Q-iteration as performing:

$$
\hat {Q} _ {t} (\mathbf {s}, \mathbf {a}) = \bar {Q} _ {t} (\mathbf {s}, \mathbf {a}) + \epsilon_ {t} (\mathbf {s}, \mathbf {a}), \tag {3}
$$

where  $\mathbb{E}[\hat{Q}] = \bar{Q}_t$  is the expected value of our prediction  $\hat{Q}_t$  for an intended target  $\bar{Q}_t$  over our estimators, and  $\epsilon_t$  is the (zero-centered) error in our estimate. Here, we assume the error  $\epsilon_t$  comes from the same underlying distribution for each of our estimators, and thus are i.i.d. random variables with a zero-mean. Now, consider the bootstrapped estimate using one of our M estimators chosen randomly:

$$
\hat {\mathcal {B}} ^ {*} \hat {Q} _ {t} (\mathbf {s}, \mathbf {a}) = r (\mathbf {s}, \mathbf {a}) + \gamma \max _ {\mathbf {a} ^ {\prime}} \hat {Q} _ {t} (\mathbf {s} ^ {\prime}, \mathbf {a} ^ {\prime}) = r (\mathbf {s}, \mathbf {a}) + \gamma \max _ {\mathbf {a} ^ {\prime}} (\bar {Q} _ {t} (\mathbf {s} ^ {\prime}, \mathbf {a} ^ {\prime}) + \epsilon_ {t} (\mathbf {s} ^ {\prime}, \mathbf {a} ^ {\prime})). \qquad (4)
$$

We now examine what happens after a subsequent update. At time  $t + 1$ , suppose that we fit a fresh set of  $M$  independent functional approximators  $\hat{Q}_{t + 1}$  with the target  $\hat{\mathcal{B}}^*\hat{Q}_t$ , introducing a new unbiased error  $\epsilon_{t + 1}$ . Then, for  $\bar{Q}_{t + 1} = \mathbb{E}[\hat{Q}_{t + 1}]$  it holds that

$$
\bar {Q} _ {t + 1} (\mathbf {s}, \mathbf {a}) = r (\mathbf {s}, \mathbf {a}) + \gamma \mathbb {E} _ {\mathbf {s} ^ {\prime} | \mathbf {s}, \mathbf {a}} [ \mathbb {E} _ {\epsilon_ {t}} [ \max  _ {\mathbf {a} ^ {\prime}} (\bar {Q} _ {t} (\mathbf {s} ^ {\prime}, \mathbf {a} ^ {\prime}) + \epsilon_ {t} (\mathbf {s} ^ {\prime}, \mathbf {a} ^ {\prime})) ] ]. \tag {5}
$$

As  $\bar{Q}_{t + 1}$  is an expectation over both the dynamics and the functional errors, it accounts for all uncertainty (here  $\mathbb{E}[\epsilon_{t + 1}] = 0$ ). But, the i.i.d. error  $\epsilon_t$  remains and will be propagated through the Bellman equations and its chain of max operations. Due to Theorem 1,  $\epsilon_t$  will become Gumbel distributed in the limit of  $t$ , and remain so due to the Gumbel distribution's max-stability. $^3$

This highlights a fundamental issue with approximation-based RL algorithms that minimize the Mean-Squared Error (MSE) in the Bellman Equation: they implicitly assume, via maximum likelihood estimation, that errors are Gaussian. In Appendix A, we further study the propagation of errors using the McFadden-Rust MDP model, and use it to develop a simplified Gumbel Error Model (GEM) for errors under functional approximation. In practice, the Gumbel nature of the errors may be weakened as estimators between timesteps share parameters and errors will be correlated across states and actions.

# 3.2 GUMBEL REGRESSION

The goal of our work is to directly model the log-partition function (LogSumExp) over  $Q(s, a)$  to avoid all of the aforementioned issues with taking a max in the function approximation domain.

![](images/feb84d0e1e0b6bd633a699dac664419d0f5dab024e32066ec1595892d71d7944.jpg)  
Figure 2: Left: The pdf of the Gumbel distribution with  $\mu = 0$  and different values of  $\beta$ . Center: Our Gumbel loss for different values of  $\beta$ . Right: Gumbel regression applied to a two-dimensional random variable for different values of  $\beta$ . The smaller the value of  $\beta$ , the more the regression fits the extrema.

![](images/277e81e91bbf39ede7f3cd7b6b47b0f59bcd9516da2bb7b7a05b3596e9062ad4.jpg)

![](images/5dce9da82ca4197925aa632eceace08d898bd89f2fdf5d93a2fa70c2f9a2c67a.jpg)

In this section we derive an objective function that models the LogSumExp by simply assuming errors to follow a gumbel distribution. Consider estimating a parameter  $h$  for a random variable  $X$  using samples  $x_{i}$  from a dataset  $\mathcal{D}$ , which have Gumbel distributed noise, i.e.  $x_{i} = h + \epsilon_{i}$  where  $\epsilon_{i} \sim -\mathcal{G}(0,\beta)$ . Then, the average log-likelihood of the dataset  $\mathcal{D}$  as a function of  $h$  is given as:

$$
\mathbb {E} _ {x _ {i} \sim \mathcal {D}} [ \log p (x _ {i}) ] = \mathbb {E} _ {x _ {i} \sim \mathcal {D}} \left[ - e ^ {\left((x _ {i} - h) / \beta\right)} + (x _ {i} - h) / \beta \right] \tag {6}
$$

Maximizing the log-likelihood yields the following convex minimization objective in  $h$ ,

$$
\mathcal {L} (h) = \mathbb {E} _ {x _ {i} \sim \mathcal {D}} \left[ e ^ {(x _ {i} - h) / \beta} - (x _ {i} - h) / \beta - 1 \right] \tag {7}
$$

which forms our objective function  $\mathcal{L}(\cdot)^4$  for Gumbel Regression.  $\beta$  is fixed as a hyper-parameter, and we show its affect on the loss in Figure 2. Critically, the minima of this objective under a fixed  $\beta$  is given by  $h = \beta \log \mathbb{E}_{x_i\sim \mathcal{D}}[e^{x_i / \beta}]$ , which resembles the LogSumExp with the summation replaced with an (empirical) expectation. In fact, this solution is the same as the operator  $\mathbb{L}_\mu^\beta (X)$  defined for MaxEnt in Section 2.1 with  $x_{i}$  sampled from  $\mu$ . In Figure 2, we show plots of Gumbel Regression on a simple dataset with different values of  $\beta$ . As this objective recovers  $\mathbb{L}^{\beta}(X)$ , we next use it to model soft-values in Max-Ent RL.

# 3.2.1 THEORY

Here we show that our objective is well behaved, considering the previously defined operator  $\mathbb{L}^{\beta}$  for random variables  $\mathbb{L}^{\beta}(X)\coloneqq \beta \log \mathbb{E}\left[e^{X / \beta}\right]$ . First, we show it models the extremum.

Lemma 3.1. For any  $\beta_{1} > \beta_{2}$ , we have  $\mathbb{L}^{\beta_1}(X) < \mathbb{L}^{\beta_2}(X)$ . And  $\mathbb{L}^{\infty}(X) = \mathbb{E}[X]$ ,  $\mathbb{L}^{0}(X) = \sup(X)$ . Thus, for any  $\beta \in (0,\infty)$ , the operator  $\mathbb{L}^{\beta}(X)$  is a measure that interpolates between the expectation and the max of  $X$ .

The operator  $\mathbb{L}^{\beta}(X)$  is known as the cumulant-generating function or the log-Laplace transform, and is a measure of the tail-risk closely linked to the entropic value at risk (EVaR) (Ahmadi-Javid, 2012).

Lemma 3.2. The risk measure  $\mathcal{L}$  has a unique minima at  $\beta \log \mathbb{E}\left[e^{X / \beta}\right]$ . And an empirical risk  $\hat{\mathcal{L}}$  is an unbiased estimate of the true risk. Furthermore, for  $\beta \gg 1$ ,  $\mathcal{L}(\theta) \approx \frac{1}{2\beta^2} \mathbb{E}_{x_i \sim \mathcal{D}}[(x_i - \theta)^2]$ , thus behaving as the MSE loss with errors  $\sim \mathcal{N}(0, \beta)$ .

In particular, the empirical loss  $\hat{\mathcal{L}}$  over a dataset of  $N$  samples can be minimized using stochastic gradient-descent (SGD) methods to give an unbiased estimate of the LogSumExp over the  $N$  samples.

Lemma 3.3.  $\hat{\mathbb{L}}^{\beta}(X)$  over a finite  $N$  samples is a consistent estimator of the log-partition function  $\mathbb{L}^{\beta}(X)$ . Similarly,  $\exp (\hat{\mathbb{L}}^{\beta}(X) / \beta)$  is an unbiased estimator for the partition function  $Z = \mathbb{E}\left[e^{X / \beta}\right]$

We provide concentration bounds for Lemma 3.3, and further theoretical discussion on Gumbel Regression in Appendix B.

# 3.3 MAXENT RL WITHOUT ENTROPY

Given Gumbel Regression can be used to directly model the LogSumExp, we apply it to Q-learning. First, we establish the connection of our framework with conservative Q-learning (Kumar et al., 2020).

Lemma 3.4. Consider the loss objective over  $Q$ -functions:

$$
\mathcal {L} (Q) = \mathbb {E} _ {\mathbf {s} \sim \rho_ {\mu}, \mathbf {a} \sim \mu (\cdot | \mathbf {s})} \left[ e ^ {\left(\mathcal {T} ^ {\pi} \hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - Q (\mathbf {s}, \mathbf {a})\right) / \beta} \right] - \mathbb {E} _ {\mathbf {s} \sim \rho_ {\mu}, \mathbf {a} \sim \pi (\cdot | \mathbf {s})} \left[ \left(\mathcal {T} ^ {\pi} \hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - Q (\mathbf {s}, \mathbf {a})\right) / \beta \right] - 1 \tag {8}
$$

where  $\mathcal{T}^{\pi} := r(\mathbf{s},\mathbf{a}) + \gamma \mathbb{E}_{\mathbf{s}'|\mathbf{s},\mathbf{a}}\mathbb{E}_{\mathbf{a}'\sim \pi}[Q(\mathbf{s}',\mathbf{a}')]$  is the vanilla Bellman operator under the policy  $\pi (\mathbf{a}|\mathbf{s})$ . Then minimizing  $\mathcal{L}$  gives the update rule:

$$
\forall \mathbf {s}, \mathbf {a}, k \hat {Q} ^ {k + 1} (\mathbf {s}, \mathbf {a}) = \mathcal {T} ^ {\pi} \hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - \beta \log \frac {\pi (\mathbf {a} \mid \mathbf {s})}{\mu (\mathbf {a} \mid \mathbf {s})} = \mathcal {B} ^ {\pi} \hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}).
$$

The above lemma transformers the regular bellman backup into the soft-Bellman backup without the need us entropies, letting us convert standard RL into MaxEnt RL. Here,  $\mathcal{L}(\cdot)$  does a conservative Q-update similar to CQL (Kumar et al., 2020) with the nice property that the implied conservative term is just the KL-constraint between  $\pi$  and  $\mu$ .<sup>5</sup> This enforces a entropy-regularization on our policy with respect to the behavior policy without the need of entropy. Thus, soft-Q learning naturally emerges as a conservative update on regular Q-learning under our objective. Here, Equation 8 is the dual of the KL-divergence between  $\mu$  and  $\pi$  (Garg et al., 2021), and we motivate this objective for RL and establish formal equivalence with conservative Q-learning in Appendix C.

In our framework, we use the MaxEnt Bellman operator  $\mathcal{B}^*$  which gives our ExtremeQ loss, which is the same as our Gumbel loss from the previous section:

$$
\mathcal {L} (Q) = \mathbb {E} _ {\mathbf {s}, \mathbf {a} \sim \mu} \left[ e ^ {(\hat {B} ^ {*} \hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - Q (\mathbf {s}, \mathbf {a})) / \beta} \right] - \mathbb {E} _ {\mathbf {s}, \mathbf {a} \sim \mu} [ (\hat {B} ^ {*} \hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - Q (\mathbf {s}, \mathbf {a})) / \beta ] - 1 \tag {9}
$$

This gives an update rule:  $\hat{Q}^{k + 1}(\mathbf{s},\mathbf{a}) = \mathcal{B}^*\hat{Q}^k (\mathbf{s},\mathbf{a}).\mathcal{L}(\cdot)$  here requires estimation of  $\mathcal{B}^*$  which is very hard in continuous action spaces. Under deterministic dynamics,  $\mathcal{L}$  can be obtained without  $\mathcal{B}^*$  as shown in Appendix C. However, in general we still need to estimate  $\mathcal{B}^*$ . Next, we motivate how we can solve this issue. Consider the soft-Bellman equation from Section 2.1 (Equation 1),

$$
\mathcal {B} ^ {*} Q = r (\mathbf {s}, \mathbf {a}) + \gamma \mathbb {E} _ {\mathbf {s} ^ {\prime} \sim P (\cdot | \mathbf {s}, \mathbf {a})} [ V ^ {*} (\mathbf {s} ^ {\prime}) ], \tag {10}
$$

where  $V^{*}(\mathbf{s}) = \mathbb{L}_{\mathbf{a}\sim \mu (\cdot |\mathbf{s}^{\prime})}^{\beta}[Q(\mathbf{s},\mathbf{a})]$ . Then  $V^{*}$  can be directly estimated using Gumbel regression by setting the temperature  $\beta$  to the regularization strength in the MaxEnt framework. This gives us the following ExtremeV loss objective:

$$
\mathcal {J} (V) = \mathbb {E} _ {\mathbf {s}, \mathbf {a} \sim \mu} \left[ e ^ {(\hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - V (\mathbf {s})) / \beta} \right] - \mathbb {E} _ {\mathbf {s}, \mathbf {a} \sim \mu} [ (\hat {Q} ^ {k} (\mathbf {s}, \mathbf {a}) - V (\mathbf {s})) / \beta ] - 1. \tag {11}
$$

Lemma 3.5. Minimizing  $\mathcal{J}$  over values gives the update rule:  $\hat{V}^k (\mathbf{s}) = \mathbb{L}_{\mathbf{a}\sim \mu (\cdot |\mathbf{s})}^\beta [\hat{Q}^k (\mathbf{s},\mathbf{a})]$ .

Then we can obtain  $V^{*}$  from  $Q(s, a)$  using Gumbel regression and substitute in Equation 10 to estimate the optimal bellman backup  $\mathcal{B}^{*}Q$ . Thus, Lemma 3.4 and 3.5 give us a scheme to solve the Max-Ent RL problem without the need of entropy.

# 3.4 LEARNING POLICIES

In the above section we derived a  $Q$ -learning strategy that does not require explicit use of a policy  $\pi$ . However, in continuous settings we still often want to recover a policy that can be run in the environment. Per Eq. 2 (Section 2.2), the optimal MaxEnt policy  $\pi^{*}(\mathbf{a}|\mathbf{s}) = \mu (\mathbf{a}|\mathbf{s})e^{(Q(\mathbf{s},\mathbf{a}) - V(\mathbf{s})) / \beta}$ . By minimizing the forward KL-divergence between  $\pi$  and the optimal  $\pi^{*}$  induced by  $Q$  and  $V$  we obtain the following training objective:

$$
\pi^ {*} = \underset {\pi} {\operatorname {a r g m a x}} \mathbb {E} _ {\rho_ {\mu} (\mathbf {s}, \mathbf {a})} \left[ e ^ {(Q (\mathbf {s}, \mathbf {a}) - V (\mathbf {s})) / \beta} \log \pi \right]. \tag {12}
$$

If we take  $\rho_{\mu}$  to be a dataset  $\mathcal{D}$  generated from a behavior policy  $\pi_{\mathcal{D}}$ , we exactly recover the AWR objective used by prior works in Offline RL (Peng et al., 2019; Nair et al., 2020), which can easily be computed using the offline dataset. This objective does not require sampling actions, which may

potentially take  $Q(s, a)$  out of distribution. Alternatively, if we want to sample from the policy instead of the reference distribution  $\mu$ , we can minimize the Reverse-KL divergence which gives us the SAC-like actor update:

$$
\pi^ {*} = \underset {\pi} {\operatorname {a r g m a x}} \mathbb {E} _ {\rho_ {\pi} (\mathbf {s}) \pi (\mathbf {a} | \mathbf {s})} [ Q (\mathbf {s}, \mathbf {a}) - \beta \log (\pi (\mathbf {a} | \mathbf {s}) / \mu (\mathbf {a} | \mathbf {s})) ]. \tag {13}
$$

Interestingly, we note this doesn't depend on  $V(s)$ . If  $\mu$  is chosen to be the last policy  $\pi_k$ , the second term becomes the KL-divergence between the current policy and  $\pi_k$ , performing a trust region update on  $\pi$  (Schulman et al., 2015; Vieillard et al., 2020). While estimating the log ratio  $\log(\pi(\mathbf{a}|\mathbf{s}) / \mu(\mathbf{a}|\mathbf{s}))$  can be difficult depending on choice of  $\mu$ , our Gumbel Loss  $\mathcal{I}$  removes the need for  $\mu$  during  $Q$  learning by estimating soft-  $Q$  values of the form  $Q(\mathbf{s},\mathbf{a}) - \beta \log(\pi(\mathbf{a}|\mathbf{s}) / \mu(\mathbf{a}|\mathbf{s}))$ .

# 3.5 PRACTICAL ALGORITHMS

In this section we develop a practical approach to Extreme Q-learning  $(\mathcal{X}$  -QL) for both online and offline RL. We consider parameterized functions  $V_{\theta}(\mathbf{s})$ $Q_{\phi}(\mathbf{s},\mathbf{a})$  ,and  $\pi_{\psi}(\mathbf{a}|\mathbf{s})$  and let  $\mathcal{D}$  be the training data distribution. A core issue with directly optimizing Eq. 10 is over-optimism about dynamics (Levine, 2018) when using simple-sample estimates for the Bellman backup. To overcome this issue in stochastic settings, we separate out the optimization of  $V_{\theta}$  from that of  $Q_{\phi}$  following Section 3.3. We learn  $V_{\theta}$  using Eq. 11 to directly fit the optimal soft-values  $V^{*}(\mathbf{s})$  based on Gumbel regression. Using  $V_{\theta}(\mathbf{s}^{\prime})$  we can

get single-sample estimates of  $\mathcal{B}^*$  as  $r(\mathbf{s},\mathbf{a}) + \gamma V_{\theta}(\mathbf{s}^{\prime})$ . Now we can learn an unbiased expectation over the dynamics,  $Q_{\phi}\approx \mathbb{E}_{\mathbf{s}'|\mathbf{s},\mathbf{a}}[r(\mathbf{s},\mathbf{a}) + \gamma V_{\theta}(\mathbf{s}')]$  by minimizing the Mean-squared-error (MSE) loss between the single-sample targets and  $Q_{\phi}$ :

# Algorithm 1 Extreme Q-learning  $(\mathcal{X}$  -QL) (Under Stochastic Dynamics)

1: Init  $Q_{\phi},V_{\theta}$  ,and  $\pi_{\psi}$

2: Let  $\mathcal{D} = \{(s, a, r, s')\}$  be data from  $\pi_{\mathcal{D}}$  (offline) or replay buffer (online)

3: for step  $t$  in  $\{1\ldots \mathrm{N}\}$  do

4: Train  $Q_{\phi}$  using  $\mathcal{L}(\phi)$  from Eq. 14

5: Train  $V_{\theta}$  using  $\mathcal{J}(\theta)$  from Eq. 11

(with  $\mathbf{a}\sim \mathcal{D}$  (offline) or  $\mathbf{a}\sim \pi_{\psi}$  (online))

6: Update  $\pi_{\psi}$  via Eq. 12 (offline) or Eq. 13 (online)

# 7: end for

$$
\mathcal {L} (\phi) = \mathbb {E} _ {(\mathbf {s}, \mathbf {a}, \mathbf {s} ^ {\prime}) \sim \mathcal {D}} \left[ \left(Q _ {\phi} (\mathbf {s}, \mathbf {a}) - r (\mathbf {s}, \mathbf {a}) - \gamma V _ {\theta} (\mathbf {s} ^ {\prime})) ^ {2} \right] \right. \tag {14}
$$

In deterministic dynamics, our approach is largely simplified and we directly learn a single  $Q_{\phi}$  using Eq. 9 without needing to learn  $\mathcal{B}^*$  or  $V^*$ . Similarly, we learn soft-optimal policies using Eq. 12 (offline) or Eq. 13 (online) settings.

Offline RL. In the offline setting,  $\mathcal{D}$  is specified as an offline dataset assumed to be collected with the behavior policy  $\pi_{\mathcal{D}}$ . Here, learning values with Eq. 11 has a number of practical benefits. First, we are able to fit the optimal soft-values  $V^{*}$  without sampling from a policy network, which has been shown to cause large out-of-distribution errors in the offline setting where mistakes cannot be corrected by collecting additional data. Second, we inherently enforce a KL-constraint on the optimal policy  $\pi^{*}$  and the behavior policy  $\pi_{\mathcal{D}}$ . This provides tunable conservatism via the temperature  $\beta$ . After offline training of  $Q_{\phi}$  and  $V_{\theta}$ , we can recover the policy post-training using the AWR objective (Eq. 12). Our practical implementation follows the training style of Kostrikov et al. (2021), but we train value network using our ExtremeQ loss.

Online RL. In the online setting,  $\mathcal{D}$  is usually given as a replay buffer of previously sampled states and actions. In practice, however, obtaining a good estimate of  $V^{*}(\mathbf{s}^{\prime})$  requires that we sample actions with high Q-values instead of uniform sampling from  $\mathcal{D}$ . As online learning allows agents to correct over-optimism by collecting additional data, we use a previous version of the policy network  $\pi_{\psi}$  to sample actions for the Bellman backup, amounting to the trust-region policy updates detailed at the end of Section 3.4. In practice, we modify SAC and TD3 with our formulation. To embed SAC (Haarnoja et al., 2018) with the benefits of Extreme Q-learning, we simply train  $V_{\theta}$  using Eq. 11 with  $\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \pi_{\psi_k}(\mathbf{a}|\mathbf{s})$ . This means that we do not use action probabilities when updating the value networks, unlike other MaxEnt RL approaches. The policy is learned via the objective  $\max_{\psi} \mathbb{E}[Q_{\phi}(s, \pi_{\psi}(s))]$  with added entropy regularization, as SAC does not use a fixed noise schedule. TD3 by default doesn't use a value network, and thus we use our algorithm for deterministic dynamics by changing the loss to train  $Q$  in TD3 to directly follow Eq. 9. The policy is learned in the same way as with SAC, except without entropy regularization as TD3 uses a fixed noise schedule.

# 4 EXPERIMENTS

We compare our Extreme Q-Learning  $(\mathcal{X}$  -QL) approach to state-of-the-art algorithms across a wide set of continuous control tasks in both online and offline settings. In practice, the exponential nature of the Gumbel regression poses difficult optimization challenges. We provide the details of loss implementation, offline results on Android, and hyperparameters in the Appendix D.

# 4.1 OFFLINE RL

Table 1: Averaged normalized scores on MuJoCo locomotion and Ant Maze tasks.  

<table><tr><td></td><td>Dataset</td><td>BC</td><td>10%BC</td><td>DT</td><td>AWAC</td><td>Onestep RL</td><td>TD3+BC</td><td>CQL</td><td>IQL</td><td>X-QL</td></tr><tr><td rowspan="9">Gym</td><td>halfcheetah-medium-v2</td><td>42.6</td><td>42.5</td><td>42.6</td><td>43.5</td><td>48.4</td><td>48.3</td><td>44.0</td><td>47.4</td><td>48.3</td></tr><tr><td>hopper-medium-v2</td><td>52.9</td><td>56.9</td><td>67.6</td><td>57.0</td><td>59.6</td><td>59.3</td><td>58.5</td><td>66.3</td><td>74.2</td></tr><tr><td>walker2d-medium-v2</td><td>75.3</td><td>75.0</td><td>74.0</td><td>72.4</td><td>81.8</td><td>83.7</td><td>72.5</td><td>78.3</td><td>84.2</td></tr><tr><td>halfcheetah-medium-replay-v2</td><td>36.6</td><td>40.6</td><td>36.6</td><td>40.5</td><td>38.1</td><td>44.6</td><td>45.5</td><td>44.2</td><td>45.2</td></tr><tr><td>hopper-medium-replay-v2</td><td>18.1</td><td>75.9</td><td>82.7</td><td>37.2</td><td>97.5</td><td>60.9</td><td>95.0</td><td>94.7</td><td>100.7</td></tr><tr><td>walker2d-medium-replay-v2</td><td>26.0</td><td>62.5</td><td>66.6</td><td>27.0</td><td>49.5</td><td>81.8</td><td>77.2</td><td>73.9</td><td>82.2</td></tr><tr><td>halfcheetah-medium-expert-v2</td><td>55.2</td><td>92.9</td><td>86.8</td><td>42.8</td><td>93.4</td><td>90.7</td><td>91.6</td><td>86.7</td><td>94.2</td></tr><tr><td>hopper-medium-expert-v2</td><td>52.5</td><td>110.9</td><td>107.6</td><td>55.8</td><td>103.3</td><td>98.0</td><td>105.4</td><td>91.5</td><td>111.2</td></tr><tr><td>walker2d-medium-expert-v2</td><td>107.5</td><td>109.0</td><td>108.1</td><td>74.5</td><td>113.0</td><td>110.1</td><td>108.8</td><td>109.6</td><td>112.7</td></tr><tr><td rowspan="6">AntMaze</td><td>antmaze-umaze-v0</td><td>54.6</td><td>62.8</td><td>59.2</td><td>56.7</td><td>64.3</td><td>78.6</td><td>74.0</td><td>87.5</td><td>93.8</td></tr><tr><td>antmaze-umaze-diverse-v0</td><td>45.6</td><td>50.2</td><td>53.0</td><td>49.3</td><td>60.7</td><td>71.4</td><td>84.0</td><td>62.2</td><td>82.0</td></tr><tr><td>antmaze-medium-play-v0</td><td>0.0</td><td>5.4</td><td>0.0</td><td>0.0</td><td>0.3</td><td>10.6</td><td>61.2</td><td>71.2</td><td>76.0</td></tr><tr><td>antmaze-medium-diverse-v0</td><td>0.0</td><td>9.8</td><td>0.0</td><td>0.7</td><td>0.0</td><td>3.0</td><td>53.7</td><td>70.0</td><td>73.6</td></tr><tr><td>antmaze-large-play-v0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.2</td><td>15.8</td><td>39.6</td><td>46.5</td></tr><tr><td>antmaze-large-diverse-v0</td><td>0.0</td><td>6.0</td><td>0.0</td><td>1.0</td><td>0.0</td><td>0.0</td><td>14.9</td><td>47.5</td><td>49.0</td></tr><tr><td rowspan="3">Franka</td><td>kitchen-complete-v0</td><td>65.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>43.8</td><td>62.5</td><td>82.4</td></tr><tr><td>kitchen-partial-v0</td><td>38.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>49.8</td><td>46.3</td><td>73.7</td></tr><tr><td>kitchen-mixed-v0</td><td>51.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>51.0</td><td>51.0</td><td>62.5</td></tr><tr><td></td><td>runtime</td><td>10m</td><td>10m</td><td>960m</td><td>20m</td><td>20m</td><td>20m</td><td>80m</td><td>20m</td><td>10m*</td></tr></table>

* We see very fast convergence for our method and training for 0.5M steps suffices on most tasks (others are with 1M steps).

Our offline approach strongly outperforms prior methods (Chen et al., 2021; Kumar et al., 2019; 2020; Kostrikov et al., 2021; Fujimoto & Gu, 2021) by 10-20 absolute points in many environments reaching the new state-of-the-art on the D4RL benchmark, as shown in Table 1. We particularly see large improvements on the AntMaze tasks, which require a significant amount of "stitching" between trajectories (Kostrikov et al., 2021) and get double-digit improvement on the Franka tasks. We find performance on the Gym locomotion tasks to be already largely saturated. Moreover, our method convergences significantly faster than IQL, and we show learning curves in Appendix D.  $\mathcal{X}$ -QL can be easily fine-tuned using online data to attain even higher performance as shown in Table 2, surpassing the final performance of prior works by a wide margin. On a variety of AntMaze tasks, our offline results before fine-tuning are on par-with IQL's performance after online fine-tuning. Like Kostrikov et al. (2021) we tune  $\beta$  for different environments, which corresponds to the weight on our conservative KL penalty.

# 4.2 ONLINE RL

We compare ExtremeQ variants of SAC (Haarnoja et al., 2018) and TD3 (Fujimoto et al., 2018), denoted  $\mathcal{X}$ -SAC and  $\mathcal{X}$ -TD3, to their vanilla versions on tasks in the DM Control, shown in Figure 3. Across all tasks an ExtremeQ variant matches or surpasses the performance of baselines.

Table 2: Finetuning results on the AntMaze environments  

<table><tr><td>Dataset</td><td colspan="2">CQL</td><td colspan="2">IQL</td><td colspan="2">X-QL</td></tr><tr><td>umaze-v0</td><td colspan="2">70.1 → 99.4</td><td colspan="2">86.7 → 96.0</td><td colspan="2">96.1 → 99.6</td></tr><tr><td>umaze-diverse-v0</td><td colspan="2">31.1 → 99.4</td><td colspan="2">75.0 → 84.0</td><td colspan="2">82.5 → 99.0</td></tr><tr><td>medium-play-v0</td><td colspan="2">23.0 → 0.0</td><td colspan="2">72.0 → 95.0</td><td colspan="2">80.2 → 97.0</td></tr><tr><td>medium-diverse-v0</td><td colspan="2">23.0 → 32.3</td><td colspan="2">68.3 → 92.0</td><td colspan="2">74.6 → 97.1</td></tr><tr><td>large-play-v0</td><td colspan="2">1.0 → 0.0</td><td colspan="2">25.5 → 46.0</td><td colspan="2">45.1 → 59.3</td></tr><tr><td>large-diverse-v0</td><td colspan="2">1.0 → 0.0</td><td colspan="2">42.6 → 60.7</td><td colspan="2">52.2 → 82.1</td></tr></table>

We see particularly large gains in the Hopper environment, and more significant gains in comparison to TD3 overall. Consistent with SAC (Haarnoja et al., 2018), we find the temperature  $\beta$  needs to be tuned for different environments with different reward scales and sparsity. A core component of TD3 introduced by Fujimoto et al. (2018) is Double Q-Learning, which takes the minimum of two  $Q$  functions to remove overestimate bias in the Q-target. As we assume errors to be Gumbel distributed, we expect our  $\mathcal{X}$ -variants to be more robust to such errors. In all environments except Cheetah Run, our  $\mathcal{X}$ -TD3 without the Double-Q trick performs better than standard TD3. Interestingly, we find that in the DM Control environments Double-Q learning did not boost performance across the board. While the gains from applying Extreme-Q learning are modest in the online setting, none of our methods require access to the policy distribution  $\pi(a|s)$ . In particular,  $\mathcal{X}$ -SAC only requires samples from  $\pi_{\psi}$ , unlike regular SAC which incorporates the log probabilities of  $\pi_{\psi}$  into value estimates.

![](images/2b1cddadb1b789ddff8c72ddd19b8c23475aab7c00159423da2c5a586b1af993.jpg)  
Figure 3: Results on the DM Control for SAC and TD3 based versions of ExtremeQ Learning.

![](images/75061f3f5af644e5c46fc0eb5ea7ae2f0e0414796f2231b21f7f7603f4b131a5.jpg)

![](images/0335e0901055018b6caa2f7352003f69c10ebfd7a385594ca317c4ce3f82ed27.jpg)

![](images/4e7feae44e3870fb0d3c23ce65471db8f4d56cf2e3b41002ca38b3bfd4254369.jpg)

# 5 RELATED WORK

Our approach builds on literature in online and offline RL. Here we review the most salient works. It is worth noting that inspiration for our framework comes choice theory in econometrics (Rust, 1986; McFadden, 1972), and our Gumbel loss was motivated by the work of IQ-Learn (Garg et al., 2021).

Online RL. Our work bridges the theoretical gap between RL and Max-Ent RL by introducing our Gumbel loss function. Unlike past work in MaxEnt RL (Haarnoja et al., 2018; Eysenbach & Levine, 2020), our method does not require explicit entropy estimation and instead addresses the problem of obtaining soft-value estimates (LogSumExp) in high-dimensional or continuous spaces (Vieillard et al., 2021) by directly modeling them via our proposed Gumbel loss, which to our knowledge has not previously been used in RL. Our loss objective is intrinsically linked to the KL divergence, and similar objectives have been used for mutual information estimation (Poole et al., 2019). IQ-Learn (Garg et al., 2021) - an avant-garde approach to Imitation Learning (IL) - introduced the same loss in IL to obtain an unbiased dual form for the reverse KL-divergence between an expert and policy distribution. Other works have also used forward KL-divergence to derive policy objectives (Peng et al., 2019) or for regularization (Schulman et al., 2015; Abdelmaleki et al., 2018). Prior work in RL has also examined using other types of loss functions (Bas-Serrano et al., 2021) or other formulations of the argmax in order to ease optimization (Asadi & Littman, 2017). Distinct from most off-Policy RL Methods (Lillicrap et al., 2015; Fujimoto et al., 2018; Haarnoja et al., 2018), we directly model  $\mathcal{B}^*$  like (Haarnoja et al., 2017; Heess et al., 2015) but attain significantly more stable results.

Offline RL. Prior works in offline RL can largely be categorized as relying on constrained or regularized Q-learning (Wu et al., 2019; Fujimoto & Gu, 2021; Fujimoto et al., 2019; Kumar et al., 2019; 2020; Nair et al., 2020), or extracting a greedy policy from the known behavior policy (Peng et al., 2019; Brandfonbrener et al., 2021; Chen et al., 2021). Most similar to our work, IQL (Kostrikov et al., 2021) fits expectiles of the Q-function of the behavior policy, but is not motivated to solve a particular problem or remain conservative. On the other hand, conservatism in CQL (Kumar et al., 2020) is motivated by lower-bounding the Q-function. Our method shares the best of both worlds - like IQL we do not evaluate the Q-function on out of distribution actions and like CQL we enjoy the benefits of conservatism. Compared to CQL, our approach uses a KL constraint with the behavior policy, and for the first time extends soft-Q learning to offline RL without needing a policy or explicit entropy values. Our choice of using the reverse KL divergence for policy updates follows closely from BRAC (Wu et al., 2019).

# 6 CONCLUSION

We propose Extreme Q-Learning, a new framework for MaxEnt RL that directly estimates the optimal Bellman backup  $\mathcal{B}^*$  without relying on explicit access to a policy. Theoretically, we bridge the gap between the regular, soft, and conservative Q-learning formulations. Empirically, we show that our framework can be used to develop simple SOTA RL algorithms. A number of future directions remain such as improving stability with training with the exponential Gumbel Loss function and integrating automatic tuning methods for temperature  $\beta$  like SAC (Haarnoja et al., 2018). Finally, we hope that our framework can find uses beyond RL in general Machine Learning using our Gumbel loss for estimation of log-partition functions.

# REFERENCES

Abbas Abdelmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a posteriori policy optimisation. In International Conference on Learning Representations, 2018. 9  
A. Ahmadi-Javid. Entropic value-at-risk: A new coherent risk measure. Journal of Optimization Theory and Applications, 155(3):1105-1123, 2012. URL https://EconPapers.repec.org/RePEc:spr:joptap:v:155:y:2012:i:3:d:10.1007_s10957-011-9968-2.5  
Kavosh Asadi and Michael L Littman. An alternative softmax operator for reinforcement learning. In International Conference on Machine Learning, pp. 243-252. PMLR, 2017. 9  
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 21  
Joan Bas-Serrano, Sebastian Curi, Andreas Krause, and Gergely Neu. Logistic q-learning. In International Conference on Artificial Intelligence and Statistics, pp. 3610-3618. PMLR, 2021. 9  
M. Bloem and N. Bambos. Infinite time horizon maximum causal entropy inverse reinforcement learning. 53rd IEEE Conference on Decision and Control, pp. 4911-4916, 2014. 2  
David Brandfonbrener, Will Whitney, Rajesh Ranganath, and Joan Bruna. Offline rl without off-policy evaluation. Advances in Neural Information Processing Systems, 34:4933-4946, 2021. 9  
Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34, 2021. 8, 9  
Benjamin Eysenbach and Sergey Levine. If maxent {rl} is the answer, what is the question?, 2020. URL https://openreview.net/forum?id=SkxcZCNKDS.9  
R. A. Fisher and L. H. C. Tippett. Limiting forms of the frequency distribution of the largest or smallest member of a sample. Mathematical Proceedings of the Cambridge Philosophical Society, 24(2):180-190, 1928. doi: 10.1017/S0305004100015681. 3  
Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in Neural Information Processing Systems, 34, 2021. 8, 9  
Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. ArXiv, abs/1802.09477, 2018. 1, 8, 9, 14, 19, 21  
Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, pp. 2052-2062. PMLR, 2019. 9  
Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, and Stefano Ermon. Iq-learn: Inverse soft-q learning for imitation. In Thirty-Fifth Conference on Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=Aeo-xqtb5p.6,9,18  
Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. 2017. 3, 9, 21  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861-1870. PMLR, 2018. 1, 2, 3, 4, 7, 8, 9  
Tamir Hazan and Tommi Jaakkola. On the partition function and random maximum a-posteriori perturbations. arXiv preprint arXiv:1206.6410, 2012. 3  
Nicolas Heess, Gregory Wayne, David Silver, Timothy Lillicrap, Tom Erez, and Yuval Tassa. Learning continuous control policies by stochastic value gradients. Advances in neural information processing systems, 28, 2015. 9  
Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169, 2021. 7, 8, 9, 20

Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. Advances in Neural Information Processing Systems, 32, 2019. 8, 9  
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179-1191, 2020. 1, 5, 6, 8, 9, 17, 18  
Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018. 7  
Qing Li. Continuous control benchmark of deepmind control suite and mujoco. https://github.com/LQNew/Continuous_Control_Benchmark, 2021.21  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015. 9  
R.Duncan Luce. The choice axiom after twenty years. Journal of Mathematical Psychology, 15(3):215-233, 1977. ISSN 0022-2496. doi: https://doi.org/10.1016/0022-2496(77)90032-3. URL https://www.sciencedirect.com/science/article/pii/0022249677900323.3,14  
Daniel McFadden. Conditional logit analysis of qualitative choice behavior. 1972. 1, 3, 9, 13, 14  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013. 1  
Alexander McFarlane Mood. Introduction to the theory of statistics. 1950. 3  
Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020. 6, 9  
Gergely Neu, Anders Jonsson, and V. Gómez. A unified view of entropy-regularized markov decision processes. *ArXiv*, abs/1705.07798, 2017. 2  
George Papandreou and Alan L Yuille. Gaussian sampling by local perturbations. Advances in Neural Information Processing Systems, 23, 2010. 3  
Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019. 6, 9  
Ben Poole, Sherjil Ozair, Aïron van den Oord, Alexander A. Alemi, and G. Tucker. On variational bounds of mutual information. In ICML, 2019. 9  
John Rust. Structural estimation of markov decision processes. In R. F. Engle and D. McFadden (eds.), Handbook of Econometrics, volume 4, chapter 51, pp. 3081-3143. Elsevier, 1 edition, 1986. URL https://editorialexpress.com/jrust/papers/handbook_ec_v4_rust.pdf.1,3,9  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889-1897. PMLR, 2015. 1, 7, 9  
Slavko Simić. On a new converse of jensen's inequality. *Publications De L'institut Mathematique*, 85:107-110, 01 2009. doi: 10.2298/PIM0999107S. 15  
Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdelmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.4  
Sebastian Thrun and Anton Schwartz. Issues in using function approximation for reinforcement learning. 1999. 14

Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, Rémi Munos, and Matthieu Geist. Leverage the average: an analysis of kl regularization in rl. 34th Conference on Neural Information Processing Systems, 2020. 7  
Nino Vieillard, Marcin Andrychowicz, Anton Raichuk, Olivier Pietquin, and Matthieu Geist. Implicitly regularized rl with implicit q-values. arXiv preprint arXiv:2108.07041, 2021. 9  
Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019. 9  
Denis Yarats and Ilya Kostrikov. Soft actor-critic (sac) implementation in pytorch. https://github.com/denisyarats/pytorch_sac, 2020.19, 21  
G. Alastair Young. High-dimensional statistics: A non-asymptotic viewpoint, martin j. wainwright, cambridge university press, 2019, xvii 552 pages, £57.99, hardback ISBN: 978-1-1084-9802-9. International Statistical Review, 88(1):258-261, 2020. doi: https://doi.org/10.1111/insr.12370. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12370.3  
Brian D Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy. Carnegie Mellon University, 2010. 1
