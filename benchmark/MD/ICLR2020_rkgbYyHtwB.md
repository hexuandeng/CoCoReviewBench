# DISAGREEMENT-REGULARIZED IMITATION LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a simple and effective algorithm designed to address the covariate shift problem in imitation learning. It operates by training an ensemble of policies on the expert demonstration data, and using the variance of their predictions as a cost which is minimized with RL together with a supervised behavioral cloning cost. Unlike adversarial imitation methods, it uses a fixed reward function which is easy to optimize. We prove a regret bound for the algorithm in the tabular setting which is linear in the time horizon multiplied by a coefficient which we show to be low for certain problems in which behavioral cloning fails. We evaluate our algorithm empirically across multiple pixel-based Atari environments and continuous control tasks, and show that it matches or significantly outperforms behavioral cloning and generative adversarial imitation learning.

# 1 INTRODUCTION

Training artificial agents to perform complex tasks is essential for many applications in robotics, video games and dialogue. If success on the task can be accurately described using a reward or cost function, reinforcement learning (RL) methods offer an approach to learning policies which has been shown to be successful in a wide variety of applications (Mnih et al., 2015; 2016; Lillicrap et al., 2016; Hessel et al., 2018) However, in other cases the desired behavior may only be roughly specified and it is unclear how to design a reward function to characterize it. For example, training a video game agent to adopt more human-like behavior using RL would require designing a reward function which characterizes behaviors as more or less human-like, which is difficult.

Imitation learning (IL) offers an elegant approach whereby agents are trained to mimic the demonstrations of an expert rather than optimizing a reward function. Its simplest form consists of training a policy to predict the expert's actions from states in the demonstration data using supervised learning. While appealingly simple, this approach suffers from the fact that the distribution over states observed at execution time can differ from the distribution observed during training. Minor errors which initially produce small deviations from the expert trajectories become magnified as the policy encounters states further and further from its training distribution. This phenomenon, initially noted in the early work of (Pomerleau, 1989), was formalized in the work of (Ross & Bagnell, 2010) who proved a quadratic  $\mathcal{O}(\epsilon T^2)$  bound on the regret and showed that this bound is tight. The subsequent work of (Ross et al., 2011) showed that if the policy is allowed to further interact with the environment and make queries to the expert policy, it is possible to obtain a linear bound on the regret. However, the ability to query an expert can often be a strong assumption.

In this work, we propose a new and simple algorithm called DRIL (Disagreement-Regularized Imitation Learning) to address the covariate shift problem in imitation learning, in the setting where the agent is allowed to interact with its environment. Importantly, the algorithm does not require any additional interaction with the expert. It operates by training an ensemble of policies on the demonstration data, and using the disagreement in their predictions as a cost which is optimized through RL together with a supervised behavioral cloning cost. The motivation is that the policies in the ensemble will tend to agree on the set of states covered by the expert, leading to low cost, but are more likely to disagree on states not covered by the expert, leading to high cost. The RL cost thus pushes the agent back towards the distribution of the expert, while the supervised cost ensures that it mimics the expert within the expert's distribution.

Our theoretical results show that, subject to realizability and optimization oracle assumptions, our algorithm obtains a  $\mathcal{O}(\epsilon \kappa T)$  regret bound for tabular MDPs, where  $\kappa$  is a measure which quantifies a tradeoff between the concentration of the demonstration data and the diversity of the ensemble outside the demonstration data. We evaluate DRIL empirically across multiple pixel-based Atari environments and continuous control tasks, and show that it matches or significantly outperforms behavioral cloning and generative adversarial imitation learning, often recovering expert performance with only a few trajectories.

# 2 PRELIMINARIES

Denote by  $S$  the state space,  $\mathcal{A}$  the action space, and  $\Pi$  the class of policies the learner is considering. Let  $T$  denote the task horizon and  $\pi^{\star}$  the expert policy whose behavior the learner is trying to mimic. For any policy  $\pi$ , let  $d_{\pi}$  denote the distribution over states induced by following  $\pi$ . Denote  $C(s,a)$  the expected immediate cost of performing action  $a$  in state  $s$ , which we assume is bounded in [0, 1]. In the imitation learning setting, we do not necessarily know the true costs  $C(s,a)$ , instead we observe expert demonstrations. Our goal is to find a policy  $\pi$  which minimizes an observed surrogate loss  $\ell$  between its actions and the actions of the expert under the induced distribution of states, i.e.

$$
\hat {\pi} = \arg \min  \mathbb {E} _ {s \sim d _ {\pi}} [ \ell (\pi (s), \pi^ {\star} (s)) ] \tag {1}
$$

For the following, we will assume  $\ell$  is the total variation distance (denoted by  $\| \cdot \|$ ), which is an upper bound on the  $0 - 1$  loss. Our goal is thus to minimize the following quantity, which represents the distance between the actions taken by our policy  $\pi$  and the expert policy  $\pi^{\star}$ :

$$
J _ {\exp} (\pi) = \mathbb {E} _ {s \sim d _ {\pi}} \left[ \| \pi (\cdot | s) - \pi^ {\star} (\cdot | s) \| \right] \tag {2}
$$

The following result shows that if  $\ell$  represents an upper bound on the  $0 - 1$  loss and  $C$  satisfies certain smoothness conditions, then minimizing this loss within  $\epsilon$  translates into an  $\mathcal{O}(\epsilon T)$  regret bound on the task cost  $J_{\mathrm{C}}(\pi) = \mathbb{E}_{s,a\sim \pi}[C(s,a)]$ .

Theorem 1. (Ross et al., 2011) Let  $\pi$  be such that  $J_{\mathrm{exp}}(\pi) = \epsilon$ , and  $Q_{T - t + 1}^{\pi^{\star}}(s,a) - Q_{T - t + 1}^{\pi^{\star}}(s,\pi^{\star}) \leq u$  for all  $a \in \mathcal{A}, t \in \{1,2,\dots,T\}$ ,  $d_{\pi}^{t}(s) > 0$ . Then  $J_{\mathrm{C}}(\pi) \leq J_{\mathrm{C}}(\pi^{\star}) + uT\epsilon$ .

Unfortunately, it is often not possible to optimize  $J_{\mathrm{exp}}$  directly, since it requires evaluating the expert policy on the states induced by following the current policy. The supervised behavioral cloning cost  $J_{\mathrm{BC}}$ , which is computed on states induced by the expert, is often used instead:

$$
J _ {\mathrm {B C}} (\pi) = \mathbb {E} _ {s \sim d _ {\pi^ {\star}}} [ \| \pi^ {\star} (\cdot | s) - \pi (\cdot | s) \| ] \tag {3}
$$

Minimizing this loss within  $\epsilon$  yields a quadratic regret bound on regret:

Theorem 2. (Ross & Bagnell, 2010) Let  $J_{\mathrm{BC}}(\pi) = \epsilon$ , then  $J_{\mathrm{C}}(\pi) \leq J_{\mathrm{C}}(\pi^{\star}) + T^{2}\epsilon$ .

Furthermore, this bound is tight: as we will discuss later, there exist simple problems which match the worst-case lower bound.

# 3 ALGORITHM

Our algorithm is motivated by two criteria: i) the policy should perform similarly to the expert on the expert's data distribution, and ii) the policy should move towards the expert's data distribution if it is away from it. These two criteria are addressed by combining two losses: a standard behavior cloning loss, and an additional loss which represents the variance over actions induced by sampling different policies from the posterior given the demonstration data  $\mathcal{D}$ . We call this the uncertainty cost, which is defined as:

Algorithm 1 Disagreement-Regularized Imitation Learning (DRIL)  
1: Input: Expert demonstration data  $\mathcal{D} = \{(s_i,a_i)\}_{i = 1}^N$    
2: Initialize policy  $\pi$  and policy ensemble  $E = \{\pi_e\}$    
3: for  $e = 1,E$  do   
4: Sample  $\mathcal{D}_e\sim \mathcal{D}$  with replacement, with  $|\mathcal{D}_e| = |\mathcal{D}|$    
5: Train  $\pi_{e}$  to minimize  $J_{\mathrm{BC}}(\pi_e)$  on  $\mathcal{D}_e$  to convergence.   
6: end for   
7: for  $i = 1,\ldots$  do   
8: Perform one gradient update to minimize  $J_{\mathrm{BC}}(\pi)$  using a minibatch from  $\mathcal{D}$    
9: Perform one step of policy gradient to minimize  $C_\mathrm{U}^{\mathrm{clip}}(s,a)$    
10: end for

$$
C _ {\mathrm {U}} (s, a) = \operatorname {V a r} _ {\pi \sim p (\pi | \mathcal {D})} (\pi (a | s))
$$

The motivation is that the variance over plausible policies is high outside the expert's distribution, since the data is sparse, but it is low inside the expert's distribution, since the data there is dense. Minimizing this cost encourages the policy to return to regions of dense coverage by the expert. Intuitively, this is what we would expect the expert policy  $\pi^{\star}$  to do as well. The total cost which the algorithm optimizes is given by:

$$
J _ {\mathrm {a l g}} (\pi) = \underbrace {\mathbb {E} _ {s \sim d _ {\pi^ {\star}}} \left[ \left\| \pi^ {\star} (\cdot | s) - \pi (\cdot | s) \right\| \right]} _ {J _ {\mathrm {B C}} (\pi)} + \underbrace {\mathbb {E} _ {s \sim d _ {\pi} , a \sim \pi (\cdot | s)} \left[ C _ {\mathrm {U}} (s , a) \right]} _ {J _ {\mathrm {U}} (\pi)}
$$

The first term is a behavior cloning loss and is computed over states generated by the expert policy, of which the demonstration data  $\mathcal{D}$  is a representative sample. The second term is computed over the distribution of states generated by the current policy and can be optimized using policy gradient. More precisely, we approximate the posterior  $p(\pi|\mathcal{D})$  by training an ensemble  $E = \{\pi_e\}_{e=1}^{|E|}$  of models on different bootstrap samples of the demonstration data. Note that the demonstration data is fixed, and this ensemble can be trained once offline. We then interleave the supervised behavioral cloning updates and the policy gradient updates which minimize the variance of the posterior. The full algorithm is shown in Algorithm 1.

In practice, for the supervised loss we optimize the KL divergence between the actions predicted by the policy and the expert actions, which is an upper bound on the total variation distance. We also found it helpful to use a clipped uncertainty cost:

$$
C _ {U} ^ {\mathrm {c l i p}} (s, a) = \left\{ \begin{array}{l l} + 1 & \text {i f} C _ {U} (s, a) \leq q \\ - 1 & \text {e l s e} \end{array} \right.
$$

where the threshold  $q$  is a top quantile of the raw uncertainty costs computed over the demonstration data. The threshold  $q$  defines a normal range of uncertainty based on the demonstration data, and values outside of this range incur a negative cost.

The RL cost can be optimized using any policy gradient method, in our experiments we used an advantage actor-critic algorithm (Mnih et al., 2016). We note that model-based methods could in principle be used as well if sample efficiency is a constraint.

# 4 ANALYSIS

# 4.1 COVERAGE COEFFICIENT

We now analyze DRIL for tabular MDPs. We will show that, subject to assumptions that the policy class contains an optimal policy and that we are able to optimize costs within  $\epsilon$  of their global

minimum, our algorithm obtains a regret bound which is linear  $\kappa T$ , where  $\kappa$  which is quantity specific to the environment and  $d_{\pi}^{\star}$ . Intuitively,  $\kappa$  represents a tradeoff between how concentrated the demonstration data is and how high the variance of the posterior is outside the expert distribution.

Assumption 1. (Realizability)  $\pi^{\star} \in \Pi$

Assumption 2. (Optimization Oracle) For any given cost function  $J$ , our minimization procedure returns a policy  $\hat{\pi} \in \Pi$  such that  $J(\hat{\pi}) \leq \arg \min_{\pi \in \Pi} J(\pi) + \epsilon$

The motivation behind our algorithm is that the policies in the ensemble agree inside the expert's distribution and disagree outside of it. This defines a reward function which pushes the learner back towards the expert's distribution if it strays away. However, what constitutes inside and outside the distribution, or sufficient agreement or disagreement, is ambiguous. Below we define quantities which makes these ideas precise.

Definition 1. For any set  $\mathcal{U} \subseteq S$ , define the maximum probability ratio between the state distributions induced by the expert policy and by policies in the policy class inside of  $\mathcal{U}$  as  $\alpha(\mathcal{U}) = \max_{\pi \in \Pi} \sum_{s \in \mathcal{U}} \frac{d_{\pi}(s)}{d_{\pi}^*(s)}$ .

Note that  $\alpha(\mathcal{U}) \leq \frac{1}{\min_{s \in \mathcal{U}} d_{\pi}^*(s)}$ . For a set  $\mathcal{U}$ ,  $\alpha(\mathcal{U})$  will be low if the expert distribution has high density inside of  $\mathcal{U}$ , and the states in  $\mathcal{U}$  is reachable by policies in the policy class.

Definition 2. Define the minimum variance of the posterior outside of  $\mathcal{U}$  as  $\beta(\mathcal{U}) = \min_{s \notin \mathcal{U}, a \in \mathcal{A}} \operatorname{Var}_{\pi \sim p(\pi | \mathcal{D})}[\pi(a|s)]$ .

We now define the  $\kappa$  coefficient as the minimum ratio of these two quantities over all possible subsets of  $S$ .

Definition 3. We define  $\kappa(\mathcal{U}) = \frac{\alpha(\mathcal{U})}{\beta(\mathcal{U})}$ , and  $\kappa = \min_{\mathcal{U} \subseteq \mathcal{S}} \kappa(\mathcal{U})$ .

We can view minimizing  $\kappa(\mathcal{U})$  over different  $\mathcal{U} \subseteq S$  as minimizing a tradeoff between coverage by the expert policy inside of  $\mathcal{U}$ , and variance of the posterior outside of  $\mathcal{U}$ . Note that by making  $\mathcal{U}$  very small, it may be easy to make  $\alpha(\mathcal{U})$  small, but doing so may also make  $\beta(\mathcal{U})$  small and  $\kappa(\mathcal{U})$  large. Conversely, making  $\mathcal{U}$  large may make  $\beta(\mathcal{U})$  large but may also make  $\alpha(\mathcal{U})$  large as a result.

# 4.2 REGRET BOUND

We now establish a relationship between the  $\kappa$  coefficient just defined, the cost our algorithm optimizes, and  $J_{\mathrm{exp}}$  defined in Equation (2) which we would ideally like to minimize and which translates into a regret bound. All proofs can be found in Appendix A.

Lemma 1. For any  $\pi \in \Pi$ , we have  $J_{\exp}(\pi) \leq \kappa J_{\mathrm{alg}}(\pi)$ .

This result shows that if  $\kappa$  is not too large, and we are able to make our cost function  $J_{\mathrm{alg}}(\pi)$  small, then we can ensure  $J_{\mathrm{exp}}(\pi)$  is also small. This result is only useful if our cost function can indeed achieve a small minimum. The next lemma shows that this is the case.

Lemma 2.  $\min_{\pi \in \Pi}J_{\mathrm{alg}}(\pi)\leq 2\epsilon$

Here  $\epsilon$  is the threshold specified in Assumption 2. Combining these two lemmas with the previous result of Ross et al. (2011), we get a regret bound which is linear in  $\kappa T$ .

Theorem 3. Let  $\hat{\pi}$  be the result of minimizing  $J_{\mathrm{alg}}$  using our optimization oracle, and assume that  $Q_{T - t + 1}^{\pi^*}(s,a) - Q_{T - t + 1}^{\pi^*}(s,\pi^*) \leq u$  for all  $a \in \mathcal{A}, t \in \{1,2,\dots,T\}, d_\pi^t(s) > 0$ . Then  $\hat{\pi}$  satisfies  $J_{\mathrm{C}}(\hat{\pi}) \leq J_{\mathrm{C}}(\pi^*) + 3u\kappa \epsilon T$ .

Our bound is an improvement over that of behavior cloning if  $\kappa$  is less than  $\mathcal{O}(T)$ . Note that DRIL does not require knowledge of  $\kappa$ . The quantity  $\kappa$  is problem-dependent and depends on the environment dynamics, the expert policy and the class of policies available to the learner. We next compute  $\kappa$  exactly for a problem for which behavior cloning is known to perform poorly, and show that it is independent of  $T$ .

Example 1. Consider the tabular MDP given in (Ross et al., 2011) as an example of a problem where behavioral cloning incurs quadratic regret, shown in Figure 1. There are 3 states  $(s_0, s_1, s_2)$

![](images/1f678d7e6dc276cb26f1fb9450172a4872cd1a5ac108660bd4438633665524bf.jpg)  
Figure 1: Example of a problem where behavioral cloning incurs quadratic regret.

and two actions  $(a_0, a_1)$ . The expert policy is given by  $\pi^{\star}(s_0) = a_0$ ,  $\pi^{\star}(s_1) = a_0$ ,  $\pi^{\star}(s_2) = a_1$ . Here  $d_{\pi}^{\star} = (0, \frac{1}{T}, \frac{T - 1}{T})$ . Writing out  $\kappa(\{s_1, s_2\})$  yields:

$$
\begin{array}{l} \kappa \left(\left\{s _ {1}, s _ {2} \right\}\right) = \frac {\alpha \left(\left\{s _ {1} , s _ {2} \right\}\right)}{\beta \left(\left\{s _ {1} , s _ {2} \right\}\right)} \\ = \frac {\max  _ {\pi \in \Pi} \sum_ {s \in \{s _ {1} , s _ {2} \}} \frac {d _ {\pi} (s)}{d _ {\pi} ^ {*} (s)}}{\min  _ {a \in \{a _ {0} , a _ {1} \}} \operatorname {V a r} (a | s _ {0})} \\ \end{array}
$$

For any  $\pi$ ,  $d_{\pi}(s_1) \leq \frac{1}{T}$  and  $d_{\pi}(s_2) \leq \frac{T - 1}{T}$  due to the dynamics of the MDP, so  $\frac{d_{\pi}(s)}{d_{\pi}^{*}(s)} \leq 1$  for  $s \in \{s_0, s_1\}$ . Furthermore, since  $s_0$  is never visited in the demonstration data  $\operatorname{Var}_{\pi \sim p(\pi | \mathcal{D})}(\pi(a|s_2))$  is equal to the variance of a uniform distribution with variance 1, i.e.  $\frac{1}{12}$ . Therefore:

$$
\kappa \leq \kappa (\{s _ {1}, s _ {2} \}) \leq \frac {1 + 1}{\frac {1}{1 2}} = 2 4
$$

Applying our result from Theorem 3, we see that our algorithm obtains an  $\mathcal{O}(\epsilon T)$  regret bound on this problem, in contrast to the  $\mathcal{O}(\epsilon T^{2})$  regret of behavioral cloning<sup>1</sup>.

# 5 RELATED WORK

The idea of learning through imitation dates back at least to the work of (Pomerleau, 1989), who trained a neural network to imitate the steering actions of a human driver using images as input. The problem of covariate shift was already observed, as the author notes: "when driving for itself, the network may occasionally stray from the center of the road and so must be prepared to recover by steering the vehicle back to the center of the road".

This issue was formalized in the work of (Ross & Bagnell, 2010), who on one hand proved an  $\mathcal{O}(\epsilon T^2)$  regret bound, and on the other hand provided an example showing this bound is tight. The subsequent work (Ross et al., 2011) gave an algorithm which obtains linear regret, provided the agent can both interact with the environment, and query the expert policy. Our approach also requires environment interaction, but importantly does not require the ability to query the expert.

Imitation learning has been used within the context of modern RL to help improve sample efficiency (Hester et al., 2018) or overcome exploration (Nair et al., 2017). These settings assume the reward is known and that the policies can then be fine-tuned with reinforcement learning. In this case, covariate shift is less of an issue since it can be corrected using the reinforcement signal.

![](images/0160b62a502c900d3061c408ede31c0ee589e6ea2d6920c99e8931b8d1a40082.jpg)  
Figure 2: Results on tabular MDP from (Ross & Bagnell, 2010). Shaded region represents range between  $5^{\text{th}}$  and  $95^{\text{th}}$  quantiles, computed across 500 trials. Behavior cloning exhibits poor worst-case regret, whereas DRIL has low regret across all trials.

![](images/91ea0f7230f01c3da5a76b86cce08ec98b825055fb6a691f355d7f312e530e75.jpg)

![](images/9858c30b303f6afe31181e7198764d271add659634bcb2c3364973dc954e9182.jpg)

The work of (Luo et al., 2019) also proposed a method to address the covariate shift problem when learning from demonstrations when the reward is known, by conservatively extrapolating the value function outside the training distribution using negative sampling. This addresses a different setting from ours, and requires generating plausible states which are off the manifold of training data, which may be challenging when the states are high dimensional such as images. The work of Reddy et al. (2019) proposed to treat imitation learning within the Q-learning framework, setting a positive reward for all transitions inside the demonstration data and zero reward for all other transitions in the replay buffer. However, this requires carefully decaying the reward over time as the policy produces states closer to the expert's distribution. Our approach deals with a similar setting, but uses a fixed reward function.

Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) is a state-of-the-art algorithm which addresses the same setting as ours. It operates by training a discriminator network to distinguish expert states from states generated by the current policy, and the negative output of the discriminator is used as a reward signal to train the policy. The motivation is that states which are outside the training distribution will be assigned a low reward while states which are close to it will be assigned a high reward. This encourages the policy to return to the expert distribution if it is strays away from it. However, the adversarial training procedure means that the reward function is changing over time, which can make the algorithm unstable or difficult to tune. In contrast, our approach uses a simple fixed reward function. We include comparisons to GAIL in our experiments.

Using disagreement between models in an ensemble to represent uncertainty has recently been explored in several contexts. The works of (Shyam et al., 2018; Pathak et al., 2019) used disagreement between different dynamics models to drive exploration in the context of model-based RL. Conversely, (Henaff et al., 2019) used variance across different dropout masks to prevent policies from exploiting error in dynamics models. Ensembles have also been used to represent uncertainty over Q-values in model-free RL in order to encourage exploration (Osband et al., 2016). Here, we use disagreement between different policies trained on the expert data to address covariate shift in the context of imitation learning.

# 6 EXPERIMENTS

# 6.1 TABULAR MDPS

As a first experiment, we applied DRIL to the tabular MDP of (Ross & Bagnell, 2010) shown in Figure 1. We computed the posterior over the policy parameters given the demonstration data using a Dirichlet distribution with parameters determined by visitation counts. For behavior cloning, we sampled a single policy from this posterior. For our method, we sampled 5 policies and used their negative variance to define an additional reward function. We combined this with a reward which was the probability density function of a given state-action pair under the posterior distribution, which corresponds to the supervised learning loss, and used tabular Q-learning to optimize the sum of these two reward functions. This experiment was repeated 500 times for time horizon lengths up to 500 and  $N = 1, 5, 10$  expert demonstration trajectories.

Figure 2 shows plots of the regret over the 500 different trials across different time horizons. Although the average performance of BC improves with more expert demonstrations, it exhibits poor

![](images/6c53eb927fd3babb487416aca3e10d1cf3cf85c5d045e6f57db562a70b982846.jpg)

![](images/c995594ceaaf731adb1de2bf6f42908d750c71589e05b79777ded46ad4f58a68.jpg)

![](images/08d27430dd455e0f11c42036a5b78a7936c3a0bf0d9d4ebf3f9d68bd6576d9ca.jpg)

![](images/4810d6091a1e1b12cbc8d2f5f9271341a43c679113cb14f809d181e076160f56.jpg)

![](images/02194c45b696c4c6791faef89eaa60d3f422fa0fe9d91f31b2cb3df3f3c473b8.jpg)

![](images/a1eb0e3e78ffbc850a6c30f1dcf7240ce7fcc803cefff81d3635e77ac3cef06a.jpg)

![](images/5445019c2c67190b91dcf56be7b11a670f06f9c3a3cc72b8a1d62b2283c8a7cc.jpg)

![](images/daecd8f2011c6bd4df86b5afc91871287baf5ac1f6f52c95d09f96b6a0839e37.jpg)

![](images/7acea52e0aca94300ac34b9a60ccf138b3236cc29bbd2645bb6fa647abf5125b.jpg)  
a)

![](images/25c1dcdcb60c6681ffa55d3a4738a3fec539c275fc339481d7078139ab24833a.jpg)

![](images/e6b51281f26cade14a60bf0656ab7bac95097bf1b013b635975aad0e56aa953b.jpg)

![](images/8cb20935a80196f77a125cfc8918e94959b7ff58c07901761a4e0b85e0bbb1e8.jpg)  
b)

![](images/d08b2c5c0f538a269db7740ce713bc972c263b9c9612d273c2f73f5453f58959.jpg)

![](images/060e8ab216e5cc5540cfd5307303792798842c1da3715d0a828a012a5b543bae.jpg)  
Figure 3: Results on Atari environments. a) Final policy performance for different numbers of expert trajectories. b) Evolution of policy reward and uncertainty cost during training with  $N = 5$  trajectories.

worst-case performance with some trials incurring very high regret, especially when using fewer demonstrations. Our method has low regret across all trials, which stays close to constant independently of the time horizon, even with a single demonstration. This performance is better than that suggested by our analysis, which showed a worst-case linear bound with respect to time horizon.

# 6.2 ATARI ENVIRONMENTS

We next evaluated our approach on six different Atari environments. We used pretrained PPO (Schulman et al., 2017) agents from the stable baselines repository (Hill et al., 2018) to generate  $N = \{1,3,5,10,15,20\}$  expert trajectories. We compared against two other methods: standard behavioral cloning (BC) and Generative Adversarial Imitation Learning (GAIL). Results are shown in Figure 3a. DRIL outperforms behavioral cloning across most environments and numbers of demonstrations, often by a substantial margin. In the worst case its performance matches that of behavior

![](images/570638bb4430b277302748dd06f6c5c91cc75b8f7acdb68fcd3d6447a524c6c7.jpg)

![](images/e609e9da51b9c22307ead1745cb2214e7cab562f3ad53cca3aae7b60433b037e.jpg)

![](images/c6d9c66ae0a8c6f797a49839d4435da720f46a185103dbebc9a045077f763181.jpg)

![](images/213d80fa24e3b9fe2e9bad32d96af1a17a6ae1d629ef08e803a830eb3f9e530b.jpg)  
Figure 4: Results on continuous control tasks.

![](images/aad0f2fe589e721e90a9c71e3902594c3f5029763c1359a159a110c19a25420f.jpg)

![](images/e24ad814a70cbe69d7c0444cb1bfcfdd554502d9605e724d9c63740c0e5ad6c1.jpg)

cloning. In many cases, our method is able to match the expert's performance using a small number of trajectories.

Figure 3b shows the evolution of the uncertainty cost and the policy reward throughout training. In all cases, the test reward improves while the uncertainty cost decreases. Interestingly, there is correspondence between the change in the uncertainty cost during training and the gap in performance between behavior cloning and DRIL. For example, in MsPacman there is both a small improvement in uncertainty cost over time and a small gap between behavior cloning and our method, whereas in Breakout there is a large improvement in uncertainty cost and a large gap between behavior cloning and our method. This suggests that the gains from our method comes from redirecting the policy back towards the expert manifold, which is manifested as a decrease in the uncertainty cost.

We were not able to obtain meaningful performance for GAIL on these domains, despite performing a hyperparameter search across learning rates for the policy and discriminator, and across different numbers of discriminator updates. We additionally experimented with clipping rewards in an effort to stabilize performance. These results are consistent with those of (Reddy et al., 2019), who also reported negative results when running GAIL on images. While improved performance might be possible with more sophisticated adversarial training techniques, we note that this contrasts with our method which uses a fixed reward function obtained through simple supervised learning.

# 6.3 CONTINUOUS CONTROL

We next report results of running our method on a 6 different continuous control tasks from the PyBullet $^2$  and OpenAI Gym (Brockman et al., 2016) environments. We again used pretrained agents to generate expert demonstrations. Results are shown in Figure 4. In these environments we found behavior cloning to be a much stronger baseline than for the Atari environments, and in many tasks it was able to match expert performance using as little as 3 trajectories. Our method exhibits a modest improvement on Walker2D and BipedalWalkerHardcore when a single trajectory is used, and otherwise has similar performance to behavior cloning. The fact that our method does not perform worse than behavior cloning on tasks where covariate shift is likely less of an issue provides evidence of its robustness.

# 7 CONCLUSION

Addressing covariate shift has been a long-standing challenge in imitation learning. In this work, we have proposed a new method to address this problem by penalizing the disagreement between an ensemble of different policies sampled from the posterior. Importantly, our method requires

no additional labeling by an expert. Our experimental results demonstrate that DRIL can often match expert performance while using only a small number of trajectories across a wide array of tasks, ranging from tabular MDPs to pixel-based Atari games and continuous control tasks. On the theoretical side, we have shown that our algorithm can provably obtain a low regret bound for tabular problems in which the  $\kappa$  parameter is low.

There are multiple directions for future work. On the theoretical side, extending our analysis to continuous state spaces and characterizing the  $\kappa$  parameter on a larger array of problems would help to better understand the settings where our method can expect to do well. Empirically, there are many other settings in structured prediction (Daume et al., 2009) where covariate shift is an issue and where our method could be applied. For example, in dialogue and language modeling it is common for generated text to become progressively less coherent as errors push the model off the manifold it was trained on. Our method could potentially be used to fine-tune language or translation models (Cho et al., 2014; Welleck et al., 2019) after training by applying our uncertainty-based cost function to the generated text.

# REFERENCES

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724-1734, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1179. URL https://www.aclweb.org/anthology/D14-1179.  
Hal Daumé, John Langford, and Daniel Marcu. Search-based structured prediction. CoRR, abs/0907.0786, 2009. URL http://arxiv.org/abs/0907.0786.  
Mikael Henaff, Alfredo Canziani, and Yann LeCun. Model-predictive policy learning with uncertainty regularization for driving in dense traffic. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=HygQBnOcYm.  
Matteo Hessel, Joseph Modayil, Hado P. van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Gheshlaghi Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In AAAI, 2018.  
Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John Agapiou, Joel Z. Leibo, and Audrunas Gruslys. Deep q-learning from demonstrations. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pp. 3223-3230, 2018. URL https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16976.  
Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Stable baselines. https://github.com/hill-a/stable-baselines, 2018.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems 29, pp. 4565-4573. Curran Associates, Inc., 2016.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014. URL http://arxiv.org/abs/1412.6980.cite arxiv:1412.6980Comment: Published as a conference paper at the 3rd International Conference for Learning Representations, San Diego, 2015.  
Ilya Kostrikov. Pytorch implementations of reinforcement learning algorithms. https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.  
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. CoRR, abs/1509.02971, 2016.  
Yuping Luo, Huazhe Xu, and Tengyu Ma. Learning self-correctable policies and value functions from demonstrations with negative sampling. CoRR, abs/1907.05634, 2019. URL http:// arxiv.org/abs/1907.05634.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, February 2015. ISSN 00280836. URL http://dx.doi.org/10.1038/nature14236.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 1928-1937, New York, New York, USA, 20-22 Jun 2016. PMLR. URL http://proceedings.mlr.press/v48/mniha16.html.  
Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Overcoming exploration in reinforcement learning with demonstrations. 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 6292-6299, 2017.  
Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped DQN. CoRR, abs/1602.04621, 2016. URL http://arxiv.org/abs/1602.04621.  
Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Self-supervised exploration via disagreement. In ICML, 2019.  
Dean A. Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In D. S. Touretzky (ed.), Advances in Neural Information Processing Systems 1, pp. 305-313. Morgan-Kaufmann, 1989. URL http://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network.pdf.  
Siddharth Reddy, Anca D. Dragan, and Sergey Levine. SQL: imitation learning via regularized behavioral cloning. CoRR, abs/1905.11108, 2019. URL http://arxiv.org/abs/1905.11108.  
Stephane Ross and Drew Bagnell. Efficient reductions for imitation learning. In Yee Whye Teh and Mike Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 661-668, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR. URL http://proceedings.mlr.press/v9/ross10a.html.  
Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Geoffrey Gordon, David Dunson, and Miroslav Dudk (eds.), Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pp. 627-635, Fort Lauderdale, FL, USA, 11-13 Apr 2011. PMLR. URL http://proceedings.mlr.press/v15/ross11a.html.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347.  
Pranav Shyam, Wojciech Jaskowski, and Faustino Gomez. Model-based active exploration. CoRR, abs/1810.12162, 2018.  
Sean Welleck, Kianté Brantley, Hal Daumé III, and Kyunghyun Cho. Non-monotonic sequential text generation. CoRR, abs/1902.02192, 2019. URL http://arxiv.org/abs/1902.02192.
