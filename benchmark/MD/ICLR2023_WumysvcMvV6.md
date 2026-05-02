# MIND THE GAP: OFFLINE POLICY OPTIMIZATION FOR IMPERFECT REWARDS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Reward function is essential in reinforcement learning (RL), serving as the guiding signal to incentivize an agent to solve a given task. However, reward function is notoriously difficult to design or even approximate. In many cases, only a sub-par reward function can be obtained, and it is even more challenging when zero reward signal is available, which often inflicts substantial performance loss or stringent restrictive requirements on expert demonstrations. In this study, we propose a unified offline policy optimization approach, RGM (Reward Gap Minimization), which can smartly handle diverse types of imperfect rewards. RGM is formulated as a bi-level optimization problem: the upper layer optimizes a reward correction term that performs state-action visitation distribution matching w.r.t. a small set of expert data; and the lower layer solves a pessimistic RL problem with the corrected rewards. By exploiting the duality of the lower level problem, we derive a tractable algorithm that enables sampled-based learning without any online interactions. Comprehensive experiments demonstrate that RGM achieves superior performance to existing methods under diverse settings of imperfect rewards. Further, RGM can effectively correct wrong or inconsistent rewards against expert preference, as well as retrieving useful information from biased rewards.

# 1 INTRODUCTION

Defining reward plays an imperative role in every reinforcement learning (RL) problem. It encodes the desired system behaviors, serving as a central signal to incentivize agents to learn and solve a given task (Abel et al., 2021). However, accurately describing a complex task using a numerical reward function is often impossible (Abel et al., 2021; Li et al., 2019). In rare cases, a perfect reward function can be approximated when task is extremely simple, such as rewarding large throttle to guide an autonomous vehicle to achieve a simple "accelerate task". This can be regarded as the Perfect Reward. Whereas, perfect rewards are commonly not obtainable due to task complexity or human cognitive biases (Hadfield-Menell et al., 2017). For example, when modeling drift in a car race (Cutler & How, 2016), only limited learning signals of every vehicular action are available, thus merely a roughly estimated reward can be used to represent the acceleration and steering actions of a vehicle. We call this the Partially Correct Reward. More often than not, no learning signals are available to aid in solving the given task, leading to an Incorrect Reward scenario.

In reality, such imperfect rewards (partially correct or incorrect) are common when unlimited reward tuning is not available due to costly or even dangerous online interactions (e.g., applying premature policies in medical applications). Generally, we can only access pre-collected datasets paired with imperfect rewards in real-world applications (Zhan et al., 2022), where a significant gap exists between the given reward in the offline dataset and the true reward in the intended task. Consequently, it is of great significance and practical value to devise adaptable methods that can perform robust offline policy optimization under imperfect rewards.

Reward shaping (Dorigo & Colombetti, 1994; Ng et al., 1999; Marthi, 2007; Hu et al., 2020; Hadfield-Menell et al., 2017) is the most common approach to addressing imperfect reward signals, which requires tremendous human efforts and numerous online evaluations. Another possible avenue is imitation learning (IL) (Pomerleau, 1988; Kostrikov et al., 2019; Sasaki & Yamashina, 2020; Kim et al., 2021; Ma et al., 2022; Xu et al., 2022b; Zhang et al., 2022) or inverse reinforcement learning methods (IRL) (Jarboui & Perchet, 2021), by directly imitating or deriving new reward functions

from expert behaviors. However, these methods heavily depend on the quantity and quality of expert demonstrations and offline datasets, which is often beyond reach in practice. Another key challenge is how to precisely measure the distance between the given rewards in data and the true rewards of the real task. Particularly, in the offline setting, it is impossible to evaluate the learned policy under a specific reward through environment interactions and verify if the policy produces desired behaviors, let alone revising the reward.

In this paper, we investigate this challenge of imperfect rewards under the offline RL setting, where no environment interactions are available. We first formally define the relative gap between given and perfect rewards based on state-action visitation distribution matching, and formulate the offline policy learning problem as a bi-level optimization. In the upper layer, the imperfect rewards are adjusted by a reward correction term, which is learned by minimizing the reward gap towards expert behaviors. In the lower layer, we solve a pessimistic RL problem to obtain the optimized policy under the corrected rewards. By exploiting Lagrangian duality of the lower level problem, the overall optimization procedure can be tractably solved in a fully-offline manner without any online interactions. We call this approach 'Reward Gap Minimization' (RGM). Compared to existing methods, RGM can: 1) evaluate and minimize the reward gap without any online interactions; 2) eliminate the strong dependency on human efforts and expert demonstrations; and 3) handle all three types of rewards (perfect, partially correct, incorrect) in a unified framework for reliable offline policy optimization.

In practice, we implement RGM using stochastic first-order two-timescale optimization. Through extensive experiments on D4RL datasets (Fu et al., 2020) and a discrete-space navigation task, we demonstrate that RGM achieves superior performance to existing methods. Furthermore, we show that RGM effectively corrects wrong/inconsistent rewards against expert preference and successfully retrieves useful information from biased rewards, making it an ideal tool for practical tasks where reward functions are difficult to design.

# 2 RELATED WORK

We here briefly summarize relevant methodological approaches that handle different types of rewards.

Perfect Rewards. Directly applying offline RL algorithms is a natural choice for offline policy optimization when rewards are assumed to be perfect w.r.t the given task (Fujimoto et al., 2019; Kumar et al., 2019; 2020; Xu et al., 2021; Wu et al., 2021; Niu et al., 2022; Fujimoto & Gu, 2021; Kostrikov et al., 2021a;b; Xu et al., 2022a;c; Li et al., 2022; Lee et al., 2021; Bai et al., 2021; An et al., 2021). However, specifying a reward function that aligns well with the given task requires domain knowledge and deep understanding of the task. Even given the perfect rewards, some offline RL methods still need to reshape or shift the rewards to achieve the best policy performance (Kostrikov et al., 2021a; Kumar et al., 2020), the equivalence to engineering the initialization of Q-function estimation that encourages conservative exploitation under offline learning (Sun et al., 2022).

Partially Correct Rewards. Reward shaping is the most common approach to handle partially correct rewards, by modifying the original reward function to incorporate task-specific domain knowledge (Dorigo & Colombetti, 1994; Randlov & Alstrom, 1998). Potential-based reward shaping (PBRS) is the first reward shaping approach from the perspective of the policy invariance property (Ng et al., 1999). Other reward shaping approaches include belief reward shaping (Marom & Rosman, 2018) and ethics shaping (Wu & Lin, 2018). However, these approaches follow a trial-and-error paradigm and require tremendous human efforts. Recent approaches such as population-based method (Jaderberg et al., 2019), optimal reward framework (Chentanez et al., 2004; Sorg et al., 2010; Zheng et al., 2018) and automatic reward shaping (Hu et al., 2020; Devidze et al., 2021; Marthi, 2007) can automatically shape the reward functions when online interaction is allowed. To the best knowledge of the authors, there is no reward shaping or correction mechanism existing for offline policy optimization. Therefore, when the given rewards are mostly correct but contain errors that lead to undesired agent behavior, people have to discard the given rewards and resort to other stopgaps like offline imitation learning, losing potentially useful information inside the existing rewards.

Incorrect Rewards. When rewards are believed to be totally wrong or missing, existing offline policy learning methods typically adopt offline imitation learning (IL) methods. These methods directly mimic the expert behavior from demonstration data without the presence of reward signal. Among these approaches, behavior cloning (BC) (Pomerleau, 1988; Florence et al., 2022) is the

simplest method, but is vulnerable to covariate shift and compounding errors (Rajaraman et al., 2020). Recent work alleviates this issue via distribution matching, which can be solved by learning a reward function (Jarboui & Perchet, 2021) or by tractable reformulation using Fenchel duality (Kostrikov et al., 2019; Kim et al., 2021; Ma et al., 2022). Some recent works also try to tackle this problem by leveraging a small expert dataset and a large, potentially sub-optimal dataset (Zolna et al., 2020; Xu et al., 2022b; Zhang et al., 2022). They use a discriminator to measure the optimality level of the data and further guide policy learning. The downside of these approaches is that they have strong requirements on the offline dataset to have a full coverage of expert distribution and only try to imitate the expert, rather than performing RL w.r.t the underlying reward of the task to improve beyond the policies in data.

# 3 PRELIMINARIES

Markov Decision Process under Imperfect Rewards. We consider the typical Markov Decision Process (MDP) setting (Puterman, 2014), which is defined by a tuple  $\mathcal{M} \coloneqq (S, A, r, T, \mu_0, \gamma)$ .  $S$  and  $A$  represent the state and action space,  $r: S \times A \to \mathbb{R}$  is the perfect reward function,  $T: S \times A \to \Delta(S)$  is the transition dynamics which represents the probability  $T(s_{t+1}|s_t, a_t)$  of the transition from state  $s_t$  to state  $s_{t+1}$  by executing action  $a_t$  at timestep  $t$ .  $\mu_0 \in \Delta(S)$  is the distribution of the initial state  $s_0$ , and  $\gamma \in (0,1)$  is the discount factor.

The perfect reward function  $r(s, a)$  encodes the desired behaviors of the task. But in most cases, we only have access to an imperfect human-designed reward function  $\tilde{r}(s, a)$ , which may not align well with the target task. This leads to a biased MDP  $\widetilde{\mathcal{M}} := (S, A, \tilde{r}, T, \mu_0, \gamma)$  as compared to the original MDP  $\mathcal{M}$ . To remedy the adverse effects of imperfect reward signals, existing offline policy learning studies (Zolna et al., 2020; Xu et al., 2022b; Ma et al., 2022; Kim et al., 2021; Jarboui & Perchet, 2021) introduce additional expert demonstrations  $\mathcal{D}^E = \left\{(s_0^E, a_0^E, s_1^E, \dots)^{(i)}\right\}_{i=0}^{N^E}$  to provide extra information on the desired policy behaviors. We follow a similar setup, but only consume very limited expert demonstrations. In our offline policy optimization setting, we are given a pre-collected dataset  $\mathcal{D}^O = \left\{(s_0, a_0, \tilde{r}_0, s_1, \dots)^{(i)}\right\}_{i=0}^{N^O}$  that is generated by an unknown behavior policy  $\pi^\beta$  and annotated with imperfect rewards  $\tilde{r}$ . We aim to learn an effective policy  $\pi: S \to \Delta(A)$  to capture the optimized agent behavior in  $\mathcal{M}$  rather than  $\widetilde{\mathcal{M}}$  using both  $\mathcal{D}^O$  and a very small expert dataset  $\mathcal{D}^E$ . For simplicity, we denote  $\mathcal{D} = \mathcal{D}^O \cup \mathcal{D}^E$ .

Reinforcement Learning. With a given MDP and the reward function  $r(s, a)$ , the goal of RL is to find an optimized policy  $\pi_r^*$  to maximize the expected cumulative discount reward:

$$
\pi_ {r} ^ {*} = \underset {\pi_ {r}} {\arg \max } (1 - \gamma) \mathbb {E} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r \left(s _ {t}, a _ {t}\right) | s _ {0} \sim \mu_ {0} (\cdot), a _ {t} \sim \pi_ {r} (\cdot | s _ {t}), s _ {t + 1} \sim T (\cdot | s _ {t}, a _ {t}) \right] \tag {1}
$$

This optimization objective can be equivalently written into the following succinct form (Puterman, 2014; Nachum et al., 2019b) by defining the normalized state-action visitation distribution  $d^{\pi_r}(s,a)$ :

$$
\pi_ {r} ^ {*} = \underset {\pi_ {r}} {\arg \max } \mathbb {E} _ {(s, a) \sim d ^ {\pi_ {r}}} [ r (s, a) ] \tag {2}
$$

$$
d ^ {\pi_ {r}} (s, a) = (1 - \gamma) \sum_ {t = 0} ^ {\infty} \gamma^ {t} \Pr \left[ s _ {t} = s, a _ {t} = a | s _ {0} \sim \mu_ {0} (\cdot), a _ {t} \sim \pi_ {r} (\cdot | s _ {t}), s _ {t + 1} \sim T (\cdot | s _ {t}, a _ {t}) \right] \tag {3}
$$

This RL objective is not directly applicable to offline setting, as it is no longer possible to sample from  $d^{\pi_r}$  via online interactions, thus serious distributional shift (Kumar et al., 2019) may occur without proper data-related regularization when learning from offline datasets. To tackle these problems, several recent works (Nachum et al., 2019b; Nachum & Dai, 2020; Lee et al., 2021) incorporate a regularizer into Eq. (2) to formulate a pessimistic RL framework that is usable in the offline setting:

$$
\pi_ {r} ^ {*} = \underset {\pi_ {r}} {\arg \max } \mathbb {E} _ {(s, a) \sim d ^ {\pi_ {r}}} [ r (s, a) ] - \alpha \cdot \mathrm {D} \left(d ^ {\pi_ {r}} \| d ^ {\mathcal {D}}\right) \tag {4}
$$

where  $d^{\mathcal{D}}$  represents the empirical state-action visitation distribution of dataset  $\mathcal{D}$ ,  $\mathrm{D}(\cdot \| \cdot)$  represents some statistical discrepancy measures and  $\alpha > 0$  controls the strength of the regularization.

# 4 REWARD GAP MINIMIZATION

To handle diverse reward settings (perfect and imperfect rewards) in real-world tasks under a unified framework, we have to investigate three major questions:

1) How to measure the gap between the given rewards and the underlying unknown perfect rewards?  
2) How to unify different reward settings and bridge the reward gap?  
3) How to perform offline policy optimization using an integrated framework?

Our answer to these questions is Reward Gap Minimization (RGM). We first formally define the reward gap in the perspective of state-action visitation distribution matching. A reward correction term is introduced to correct the problematic rewards that are inconsistent with expert behaviors, and to enable the preservation of useful information in partially correct rewards. Under this setup, we can model the offline policy learning problem as a bi-level optimization problem, with the upper layer minimizing the reward gap and the lower layer solving a pessimistic RL problem. To derive a tractable algorithm, we leverage Lagrangian duality to eliminate the requirement of online samples. Finally, we present a practical implementation of RGM based on stochastic first-order two-timescale optimization and extract the policy during the course of reward learning.

# 4.1 DEFINITION OF REWARD GAP

Previous approaches that tackle imperfect rewards typically resort to inverse reward design (Hadfield-Menell et al., 2017) or inverse RL (Ng et al., 2000; Abbeel & Ng, 2004), with the final goal of learning an explicit reward function that is well-aligned with the task. These algorithms, however, heavily rely on online interactions and are expensive to implement. Moreover, as observed in recent literature, there exist some tasks that cannot be captured by a numerical Markov reward function (Abel et al., 2021). Hence, learning an explicit proxy of the perfect reward function and comparing it to the given rewards is unlikely the best option to characterize the reward gap. In this study, we define the reward gap based on the outcome of the learned agent behavior, i.e., from the perspective of state-action visitation distribution matching.

Definition 1. (Reward gap) Given an arbitrary reward function  $\hat{r}(s, a)$  and the state-action visitation distribution  $d^*$  of the optimal policy induced from the perfect rewards  $r$ , the reward gap between  $\hat{r}$  and  $r$  is defined as

$$
D _ {f} \left(d ^ {\pi_ {\hat {r}} ^ {*}} \| d ^ {*}\right) \tag {5}
$$

where  $D_{f}(p\| q) = \mathbb{E}_{z\sim q}\left[f\left(\frac{p(z)}{q(z)}\right)\right]$  is the  $f$  -divergence between distributions  $p$  and  $q$  , and  $d^{\pi_r^*}$  represents the state-action visitation distribution induced by  $\pi_{\hat{r}}^{*}$  , which is derived using Eq. (4) with the reward function  $\hat{r}$

Note that  $d^{*}$  is generally unobtainable since the perfect reward function is unknown. We can alternatively use the state-action visitation distribution  $d^{E}$  in expert demonstrations  $\mathcal{D}^{E}$  to approximate  $d^{*}$ . In the next sub-section, we discuss how to adjust  $\hat{r}$  to achieve the goal of reward gap minimization.

# 4.2 BI-LEVEL OPTIMIZATION

Reward Correction. In our study, we consider  $\hat{r}(s,a) \coloneqq \tilde{r}(s,a) + \Delta r(s,a,\tilde{r})$ , where  $\Delta r(s,a,\tilde{r})$  is a learnable reward correction term that is correlated with the given imperfect rewards  $\tilde{r}$  in  $\mathcal{D}$ . The introduction of  $\Delta r(s,a,\tilde{r})$  enables us to exploit useful information within the partially correct rewards, while also correcting the wrong or inconsistent reward signals. We can further use it to construct a bi-level optimization formulation for RGM, where the upper-level problem optimizes the reward correction term to minimize the  $f$ -divergence between  $d^{\pi_{\hat{r}}^{*}}$  and  $d^{E}$ , and the lower-level problem solves  $\pi_{\hat{r}}^{*}$  as the optimal policy of a pessimistic RL problem with the corrected rewards:

$$
\Delta r ^ {*} = \underset {\Delta r} {\arg \min } \mathrm {D} _ {f} \left(d ^ {\pi_ {\hat {r}} ^ {*}} \| d ^ {E}\right) \tag {6}
$$

$$
\text {s . t .} \quad \pi_ {\hat {r}} ^ {*} = \underset {\pi} {\arg \max } \mathbb {E} _ {(s, a) \sim d ^ {\pi_ {\hat {r}}}} [ \hat {r} (s, a) ] - \alpha \mathrm {D} _ {f} \left(d ^ {\pi_ {\hat {r}}} \| d ^ {\mathcal {D}}\right) \tag {7}
$$

The above bi-level optimization formulation poses several technical difficulties, stemming from the complexity of deriving  $d^{\pi_{\hat{r}}^{*}}$  from  $\pi_{\hat{r}}^{*}$ , as well as the requirement of online samples from  $d^{\pi_{\hat{r}}^{*}}$ , which is

impossible under the offline setting. In the following, we present reformulations for both lower and upper level problems, which leads to a tractable form and an easy-to-implement algorithm.

Reformulation of Lower Level Problem. We first focus on reformulating the lower level problem by exploiting the Bellman flow constraint (Puterman, 2014) and the duality of the problem.

Definition 2. (Bellman flow constraint) Let  $\mathcal{T}_{\star}d(s) = \sum_{\bar{s},\bar{a}}T(s|\bar{s},\bar{a})d(\bar{s},\bar{a})$  denote the transpose (or adjoint) transition operator, then the Bellman flow constraint for the state-action visitation distribution  $d(s,a)$  is given as:

$$
\sum_ {a} d (s, a) = (1 - \gamma) \mu_ {0} (s) + \gamma \mathcal {T} _ {\star} d (s), \forall s \in \mathcal {S} \tag {8}
$$

If  $d(s, a) \geq 0$  satisfies the Bellman flow constraint, then  $d(s, a)$  is feasible and there is a one-to-one correspondence between  $d$  and the related policy  $\pi$ : i.e.,  $d$  is the only state-action visitation distribution for policy  $\pi(a|s) = \frac{d(s, a)}{\sum_{\bar{a}} d(s, \bar{a})}$ , while  $\pi$  is the only policy whose state-action visitation distribution is  $d$  (for detailed proof see Puterman (2014)). Leveraging the Bellman flow constraint and the one-to-one correspondence property, the lower level problem Eq. (7) can be re-written to a constraint maximization problem w.r.t.  $d$  in place of  $\pi_{\bar{r}}$ :

$$
d _ {\hat {r}} ^ {\pi_ {\hat {r}} ^ {*}} = \underset {d \geq 0} {\arg \max} \mathbb {E} _ {(s, a) \sim d} [ \hat {r} (s, a) ] - \alpha \mathbf {D} _ {f} \left(d \| d ^ {\mathcal {D}}\right)
$$

$$
\text {s . t .} \quad \sum_ {a} d (s, a) = (1 - \gamma) \mu_ {0} (s) + \gamma \mathcal {T} _ {\star} d (s), \forall s \in S \tag {9}
$$

The Lagrange dual problem of Eq. (9) is as follow:

$$
\min  _ {V (s)} \max  _ {d \geq 0} \mathbb {E} _ {(s, a) \sim d} [ \hat {r} (s, a) ] - \alpha \mathrm {D} _ {f} \left(d \| d ^ {\mathcal {P}}\right) + \sum_ {s} V (s) \left[ (1 - \gamma) \mu_ {0} (s) + \gamma \mathcal {T} _ {\star} d (s) - \sum_ {a} d (s, a) \right] \tag {10}
$$

where  $V(s)$  are Lagrange multipliers. Note that the primal problem Eq. (9) is convex w.r.t.  $d$ , and under a mild assumption (see Assumption 1 in Appendix A.2), the Slater's condition (Boyd et al., 2004) holds, which means by strong duality, we can solve the original primal problem by solving Eq. (10). After rearranging the terms, Eq. (10) can be equivalently written as the following form (see Lemma 2 in Appendix A.2 for detailed deduction):

$$
\min  _ {V (s)} \max  _ {d \geq 0} (1 - \gamma) \mathbb {E} _ {s \sim \mu_ {0}} [ V (s) ] + \mathbb {E} _ {(s, a) \sim d} [ \hat {r} (s, a) + \gamma \mathcal {T} V (s, a) - V (s) ] - \alpha \mathrm {D} _ {f} (d \| d ^ {\mathcal {D}}) \tag {11}
$$

in which  $\mathcal{T}V(s,a) = \sum_{s,a}T(s'|s,a)V(s')$  denotes the transition operator. Next, by exploiting the Fenchel conjugate, we can further transform the minimax problem Eq. (11) into a tractable single-level unconstrained minimization problem (see Proposition 1 in Appendix A.2 for detailed derivation), which eliminates the requirement of online samples:

$$
\min  _ {V (s)} (1 - \gamma) \mathbb {E} _ {s \sim \mu_ {0}} [ V (s) ] + \alpha \mathbb {E} _ {(s, a) \sim d ^ {\mathcal {D}}} \left[ f _ {\star} \left(\frac {\hat {r} (s , a) + \gamma \mathcal {T} V (s , a) - V (s)}{\alpha}\right) \right] \tag {12}
$$

where  $f_{\star}$  is the Fenchel conjugate of  $f$ . In the above formulation, the Lagrange multipliers  $V(s)$  can be equivalently perceived as some sort of state-value functions, which can be learned and optimized via a parameterized neural network, similar to the treatment used in the DICE-family of RL algorithms (Nachum et al., 2019a; Nachum & Dai, 2020).

Reformulation of Upper Level Problem. Using the property of Fenchel conjugate, we show that the optimal  $d^{*}$  and  $V^{*}$  from the lower level problem satisfy the following ideal relationship (see Proposition 2 in Appendix A.3 for details):

$$
\frac {d ^ {\pi_ {\hat {r}} ^ {*}} (s , a)}{d ^ {\mathcal {D}} (s , a)} = f _ {\star} ^ {\prime} \left(\frac {\hat {r} (s , a) + \gamma \mathcal {T} V ^ {*} (s , a) - V ^ {*} (s)}{\alpha}\right) \tag {13}
$$

Plugging the above equation into Eq. (7), we can obtain a new objective for the upper level problem:

$$
\Delta r ^ {*} = \underset {\Delta r} {\arg \min } \mathrm {D} _ {f} \left(f _ {\star} ^ {\prime} \left(\frac {\hat {r} + \gamma \mathcal {T} V ^ {*} - V ^ {*}}{\alpha}\right) d ^ {\mathcal {D}} \| d ^ {E}\right) \tag {14}
$$

![](images/70bfc80da80d40602009a7903921de0700b8fb48f2f073ed50fa32a4efaeeaab.jpg)  
Figure 1: Illustration of the reformulated bi-level optimization problem.

For simplicity, we denote  $f_{\star}^{\prime}\left(\frac{\hat{r} + \gamma\mathcal{T}V^{*} - V^{*}}{\alpha}\right)$  as  $g$ . By expanding the  $f$ -divergence, we have:

$$
\mathrm {D} _ {f} \left(d ^ {\mathcal {D}} g \| d ^ {E}\right) = \mathbb {E} _ {(s, a) \sim d ^ {E}} \left[ f \left(\frac {d ^ {\mathcal {D}} (s , a) g (s , a)}{d ^ {E} (s , a)}\right) \right] = \mathbb {E} _ {(s, a) \sim d ^ {\mathcal {D}}} \left[ \frac {d ^ {E} (s , a)}{d ^ {\mathcal {D}} (s , a)} f \left(\frac {d ^ {\mathcal {D}} (s , a)}{d ^ {E} (s , a)} g (s, a)\right) \right] \tag {15}
$$

The above objective involves computing the distribution ratio  $w(s, a) \triangleq d^{E}(s, a) / d^{\mathcal{D}}(s, a)$ . In the tabular case, we can empirically estimate  $w(s, a) = \frac{\sum_{(\bar{s}, \bar{a}) \in \mathcal{D}E} \mathbf{1}(\bar{s} = s, \bar{a} = a) / N^{E}}{\sum_{(\bar{s}, \bar{a}) \in \mathcal{D}} \mathbf{1}(\bar{s} = s, \bar{a} = a) / N}$ . But in the continuous state-action settings, estimating the distribution ratio  $w$  using only samples from  $d^{\mathcal{D}}$  and  $d^{E}$  becomes a challenge. Inspired by previous studies (Goodfellow et al., 2020; Ma et al., 2022), we instead train a discriminator  $h: S \times A \to (0, 1)$  to infer if  $(s, a)$  samples are from  $\mathcal{D}^{E}$  or not:

$$
h ^ {*} = \underset {h} {\arg \min } \mathbb {E} _ {(s, a) \sim d ^ {D}} [ \log (h (s, a)) ] + \mathbb {E} _ {(s, a) \sim d ^ {E}} [ \log (1 - h (s, a)) ] \tag {16}
$$

where the optimal discriminator is  $h^{*}(s,a) = \frac{d^{\mathcal{D}}(s,a)}{d^{\mathcal{D}}(s,a) + d^{E}(s,a)}$  (Goodfellow et al., 2020). We can optimize the above objective to obtain the optimal  $h^{*}$ , and further recover  $w(s,a) = 1 / h^{*}(s,a) - 1$ . Finally, combining all the reformulations, the final tractable form of the original bi-level optimization problem Eq. (6)-(7) is given as follows:

$$
\Delta r ^ {*} = \underset {\Delta r} {\arg \min } \mathbb {E} _ {(s, a) \sim d ^ {\mathcal {D}}} \left[ w (s, a) f \left(f _ {\star} ^ {\prime} \left(\frac {\hat {r} (s , a) + \gamma \mathcal {T} V ^ {*} (s , a) - V ^ {*} (s)}{\alpha}\right) / w (s, a)\right) \right] \tag {17}
$$

$$
\text {s . t .} V ^ {*} (s) = \underset {V (s)} {\arg \min } (1 - \gamma) \mathbb {E} _ {s \sim \mu_ {0}} [ V (s) ] + \alpha \mathbb {E} _ {(s, a) \sim d ^ {\mathcal {D}}} \left[ f _ {\star} \left(\frac {\hat {r} (s , a) + \gamma \mathcal {T} V (s , a) - V (s)}{\alpha}\right) \right]
$$

Policy Extraction. With the learned reward correction term  $\Delta r(s, a, \tilde{r})$ , we can in principle use existing offline RL algorithms to learn the policy with the corrected rewards. However, this implicates additional policy evaluation and policy improvement steps. A more elegant way is to extract the policy through weighted BC as follows, which is substantially more robust and less expensive:

$$
\pi^ {*} = \arg \min  _ {\pi} - \mathbb {E} _ {(s, a) \sim d ^ {\pi_ {\hat {r}} ^ {*}}} [ \log \pi (a | s) ] = - \mathbb {E} _ {(s, a) \sim d ^ {\mathcal {D}}} \left[ \frac {d ^ {\pi_ {\hat {r}} ^ {*}} (s , a)}{d ^ {\mathcal {D}} (s , a)} \log \pi (a | s) \right] \tag {18}
$$

where  $\frac{d^{\pi_r^*}(s,a)}{d^{\mathcal{D}}(s,a)}$  can be calculated from Eq. (13).

# 4.3 PRACTICAL IMPLEMENTATION

There exists a few classical algorithms (Colson et al., 2007; Sinha et al., 2017) to solve bi-level optimization problems. However, they typically require the problem to be mathematically well-behaved and these methods do not scale well with the problem size. In our implementation, we use the stochastic first-order two-timescale optimization technique (Borkar, 1997), which has been successfully applied in several RL algorithms (Hong et al., 2020; Cheng et al., 2022). Specifically, we make the gradient update step size of the upper level problem much smaller than the one of

Table 1: Average normalized scores of RGM compared with offline IL and RL baselines on D4RL datasets. The scores are from the final 10 evaluations with 5 seeds. (T) means policy optimization with true rewards. (B) means only imperfect rewards are available. We obtain the results via running author-provided open-source code, and some scores are reported from TD3+BC (Fujimoto & Gu, 2021) and IQL (Kostrikov et al., 2021b) papers. The top 2 scores for each dataset under imperfect rewards are marked in blue.  

<table><tr><td rowspan="2">D4RL Dataset</td><td colspan="3">Offline IL</td><td colspan="5">Offline RL</td><td rowspan="2">RGM (B)</td></tr><tr><td>BC</td><td>DWBC</td><td>SMODICE</td><td colspan="2">TD3+BC (T/B)</td><td colspan="2">IQL (T/B)</td><td>CQL (T/B)</td></tr><tr><td>hopper-r</td><td>4.9</td><td>23.9 ±2.7</td><td>5.9±4.6</td><td colspan="2">8.5±0.6 / 13.3±14.9</td><td colspan="2">7.9±0.4 / 1.3±0.2</td><td>8.3±0.2 / 1.7±0.9</td><td>21.2 ±0.4</td></tr><tr><td>halfcheetah-r</td><td>0.2</td><td>2.0±0.9</td><td>2.6 ±1.0</td><td colspan="2">11.0±1.1 / -17.1±4.0</td><td colspan="2">11.2±2.9 / 2.2 ±0.0</td><td>20.0±0.4 / -0.4±0.4</td><td>0.2±0.0</td></tr><tr><td>walker2d-r</td><td>1.7</td><td>68.3 ±13.2</td><td>-0.2±0.1</td><td colspan="2">1.6±1.7 / 0.8±0.9</td><td colspan="2">5.9±0.5 / 0.3±0.1</td><td>8.3±0.1 / 0.1±0.2</td><td>7.7 ±3.3</td></tr><tr><td>hopper-m</td><td>52.9</td><td>16.5±3.2</td><td>54.5±4.0</td><td colspan="2">59.3±4.2 / 13.7±12.9</td><td colspan="2">66.2±5.7 / 34.0±7.6</td><td>58.5±2.1 / 56.4 ±5.8</td><td>55.5 ±1.0</td></tr><tr><td>halfcheetah-m</td><td>42.6</td><td>8.2±4.1</td><td>42.9 ±0.9</td><td colspan="2">48.3±0.3 / 35.2±1.4</td><td colspan="2">47.4±0.2 / 42.0±0.5</td><td>44.0±5.4 / 43.5 ±1.0</td><td>40.7±1.4</td></tr><tr><td>walker2d-m</td><td>75.3</td><td>18.8±14.9</td><td>1.0±1.5</td><td colspan="2">83.7±2.1 / 30.1±24.2</td><td colspan="2">78.3±8.7 / 68.9±4.4</td><td>72.5±0.8 / 71.1±3.6</td><td>72.3 ±10.7</td></tr><tr><td>hopper-m-r</td><td>18.1</td><td>21.4±2.3</td><td>20.4±5.3</td><td colspan="2">60.9±18.8 / 23.5 ±10.0</td><td colspan="2">94.7±8.6 / 0.7±0.0</td><td>95.0±6.4 / 11.5±4.2</td><td>59.1 ±15.3</td></tr><tr><td>halfcheetah-m-r</td><td>36.6</td><td>9.2±3.1</td><td>37.1 ±3.3</td><td colspan="2">44.6±0.5 / 31.8±4.0</td><td colspan="2">44.2±1.2 / 18.1±13.4</td><td>45.5±0.5 / 16.5±20.0</td><td>37.8 ±2.6</td></tr><tr><td>walker2d-m-r</td><td>26.0</td><td>56.6 ±17.8</td><td>41.1±33.5</td><td colspan="2">81.8±5.5 / 7.8±2.3</td><td colspan="2">73.8±7.1 / 4.9±7.3</td><td>77.2±5.5 / 17.4±3.9</td><td>48.6 ±3.6</td></tr><tr><td>hopper-m-e</td><td>52.5</td><td>16.5±3.2</td><td>75.4 ±2.6</td><td colspan="2">98.0±9.4 / 50.8±40.8</td><td colspan="2">91.5±14.3 / 49.3±22.7</td><td>105.4±6.8 / 68.3±13.9</td><td>87.1 ±10.7</td></tr><tr><td>halfcheetah-m-e</td><td>55.2</td><td>0.0±0.8</td><td>88.2 ±4.5</td><td colspan="2">90.7±4.3 / 35.3±6.8</td><td colspan="2">86.7±5.3 / 53.4±6.8</td><td>91.6±2.8 / 64.8±4.0</td><td>81.5 ±0.9</td></tr><tr><td>walker2d-m-e</td><td>107.5</td><td>54.3±21.5</td><td>29.8±33.9</td><td colspan="2">110.1±0.5 / 44.7±35.4</td><td colspan="2">109.6±1.0 / 108.3±0.7</td><td>108.8±0.7 / 75.4±7.5</td><td>108.8 ±0.4</td></tr><tr><td>Mean Score</td><td>39.5</td><td>22.6</td><td>33.2</td><td colspan="2">58.2 / 24.6</td><td colspan="2">59.8 / 32.0</td><td>60.5 / 35.5</td><td>52.0</td></tr></table>

the lower level problem (see Figure 1 for RGM framework. Refer to Appendix B for additional implementation details of RGM with KL-divergence). In the following section, we will demonstrate through empirical experiments the efficacy of the two-timescale optimization techniques for our bi-level optimization solution.

# 5 EXPERIMENTS

In this section, we present empirical evaluations of RGM under imperfect reward settings<sup>1</sup>. We first evaluate RGM against existing methods on D4RL-v2 (Fu et al., 2020) benchmark datasets. We then provide an in-depth property analysis of the learned corrected rewards through both illustrative examples and ablations. Considering expert data collections may be costly in most real-world applications, we use only 1 expert trajectory (serving as  $\mathcal{D}^E$ ) in all our experiments for RGM.

# 5.1 COMPARISON WITH OFFLINE RL

We train RGM and SOTA offline RL methods (including TD3+BC (Fujimoto & Gu, 2021), IQL (Florence et al., 2022) and CQL (Kumar et al., 2020)) under imperfect corrupted rewards and report their performances evaluated based on the the perfect rewards $^2$  in Table 1. Since offline RL methods are not specifically designed to handle imperfect rewards thus suffer from severe performance drop (illustrated in Figure 4), we also report their performance with perfect rewards.

Table 1 shows that RGM surpasses offline RL methods under imperfect rewards by a large margin and achieves similar performance to offline RL policies that are trained on perfect rewards. This shows a remarkable advantage of RGM as it remedies the negative effects of imperfect rewards using only one expert trajectory, which can be particularly useful for a wide range of real-world scenarios. It removes the restrictive requirements on perfect rewards from offline policy optimization and eliminates severe performance degeneration when perfect rewards are unattainable.

# 5.2 COMPARISON WITH OFFLINE IL

We compare RGM with BC and SOTA offline IL methods (DWBC (Xu et al., 2022b) and SMODICE (Ma et al., 2022)) that can learn from mixed-quality data. Only offline IL methods that tackle imperfect rewards are considered as baselines, because other existing methods such as reward shaping can only be applied to online settings (see Section 2 for detailed discussions).

To align with our setting, we train offline IL baselines based on the joint dataset  $\mathcal{D} = \mathcal{D}^O \cup \mathcal{D}^{E_1}$ , where  $\mathcal{D}^O$  is the original D4RL dataset and  $\mathcal{D}^{E_1}$  contains only one expert trajectory. It is worth

![](images/640dcc1b9384bd36c84533d346610e978ae233d7adc9094649e9a808a7277675.jpg)

![](images/b990f7bd065ab3c67e7337cbf27dcce32422855ce3fda99f10f7369420b219d4.jpg)  
Figure 2: Comparison under offline IL settings, where the non-expert dataset  $\mathcal{D}^O$  already contains plenty of expert trajectories. The note "-w.e" stands for the mixed dataset that combines the original D4RL dataset with many expert trajectories (see Table 4 in Appendix E for additional results).

![](images/df216b06b27dd1480135f8d18df5b6e69e3735d5a1dfd565b871bbb69857d79b.jpg)

![](images/5b12b117d67ed338a6f9930212664e0dc2d0a80c726b6905890aa84b782c8f0d.jpg)  
(a) Expert demonstration

![](images/000e52a2fcbeefa3e0db65afa4fe9475e8916276b18130bd861e628edd12ffbb.jpg)  
(b) Results of zero  $\tilde{r}$

![](images/dffc641da23a775339af89f64ae37fe5d247f95fd858a74655357aabcdb3f362.jpg)  
Figure 3: Learned rewards  $\hat{r}$  and optimal distribution  $d^{\pi_{\hat{r}}^{*}}$  trained on two types of imperfect rewards  $\tilde{r}$ . The opacity of each square is determined by the marginal state distribution  $d^{\pi_{\hat{r}}^{*}}(s)$ . The opacity of the arrow shows the learned reward  $\hat{r}$ , where the darkest arrow points to the direction of the highest reward. The expert starts from  $\square$ , follows the path  $\square$  and arrow  $\rightarrow$  to reach the goal  $\square$ .  $\tilde{r}$  in (b) is  $+10$  at the goal and is zero at other states.  $\tilde{r}$  in (c) falsely punishes the agent on  $\square$  and correctly punishes the RL agent on a set of fires  $\uparrow$ .  
(c) Results of partially correct  $\tilde{r}$

mentioning that DWBC and SMODICE both build on the strong assumption that  $\mathcal{D}^O$  already covers a large expert dataset  $\mathcal{D}^{E_23}$ , which is a rare case in real scenarios. As a result, we can see from Table 1 that these two methods suffer from inferior performance when the restrictive requirements about full coverage of expert distribution are not satisfied. RGM, however, performs well when nearly no expert trajectories are contained in the non-expert dataset, because RGM is optimizing a RL objective that dictates relatively relaxed requirements on the quality of dataset.

To further illustrate the superiority of RGM, we compare RGM with DWBC and SMODICE under their settings by adding  $100\sim 200$  expert trajectories into  $\mathcal{D}^O$ . Results show that RGM can still outperform SOTA offline IL methods by a large margin (see Figure 2 and Table 4 in Appendix E).

# 5.3 INVESTIGATIONS ON LEARNED REWARD CORRECTION TERM

Benefits of Learned Rewards. We investigate the potential benefits of the learned rewards via demonstrative experiments in a  $8 \times 8$  grid world, and observe that the learned rewards enjoy three desirable properties: 1) encode long horizon information; 2) correct wrong rewards against expert preference; and 3) retrieve useful information from existing rewards, as shown in Figure 3.

Figure 3b shows that the learned rewards not only perceive correct learning signals on expert paths, but also generalize well on non-expert paths. We can successfully navigate to the destination at most positions by only maximizing one-step reward, meaning the learned rewards encode long horizon information. However, the learned rewards may lead the agent to collide the fire. By contrast, Figure 3c shows that the learned rewards can avoid the fire by retrieving useful information in imperfect  $\tilde{r}$ , meanwhile correcting the wrong signals against expert preference. All three desirable properties are hard to achieve in existing methods, which commonly depend on discriminator-related rewards only<sup>4</sup>.

![](images/cb49dedcb8ce62da31bb4ee6e59514e45eb4b9ce4319c8b265f8289323f42e14.jpg)  
Figure 4: Performance drop of normalized returns of SOTA offline RL methods under imperfect and RGM corrected rewards.  $H$ : Hopper;  $HC$ : HalfCheetah;  $W$ : Walker2d.  $r$ : random;  $m$ : medium;  $m-r$ : medium-replay;  $m-e$ : medium-expert. Severe performance drop shows that offline RL is largely affected given imperfect rewards. The corrected rewards learned by RGM, however, can largely overcome the negative effect of imperfect rewards.

![](images/9b64f6f3e7ae628174d80b1b90b88fe52ea64d4f59a45fa4d4ff02c3c8152177.jpg)

![](images/b98920e99e548ec99ab00378ecc75491d64a9ecdca4ebf43f8d50a38f1acfda0.jpg)

![](images/289d44e6d67a2673cf62fa14964da39dd5cbf49b181a416d6f8640c6c2854a80.jpg)  
(a) Learning curve of  $\Delta r$

![](images/c1a7ef2da7d713d35c400d9eabd247536f259c509d4606d1b990155e5ba351d0.jpg)  
(b) Effect of  $\tilde{r}$  on  $\hat{r}$

![](images/1e20b71ea896380600d51fa65e9a3246f58ee4aed99e3630daf643f5642a3751.jpg)  
Figure 5: Experiments on learned rewards in high dimensional MDPs. The superscript  $\bar{\cdot}$  means the mean value of mini-batch samples, the subscript E and O denote the value on expert and non-expert data, respectively. Large  $\bar{r}_E - \bar{r}_O$  and large  $\Delta \bar{r}_E - \Delta \bar{r}_O$  both mean a clear classification between expert and non-expert behaviors. Perfect rewards are beneficial to reward learning, but incorrect rewards are quite the opposite. However, RGM can largely remedy the negative effect of incorrect rewards.  
(c) Effect of  $\tilde{r}$  on  $\Delta r$

Offline RL with Corrected Rewards. We highlight that the learned corrected rewards  $\hat{r}$  obtained by RGM can also be utilized in other offline RL approaches. To be mentioned, the corrected rewards are optimized based on the specific  $\alpha$  in Eq. (7), hence may not be optimal to other offline RL methods. Whereas, Figure 4 shows that in D4RL experiments, the corrected rewards largely remedy most of the negative effects of the imperfect rewards and even surpass perfect rewards in some datasets.

Ablations on Learned Rewards. Additionally, we investigate the learned rewards in high-dimensional MDPs with continuous state-action spaces via monitoring the learning dynamics of both the reward correction term  $\Delta r$  and the final learned rewards  $\hat{r}$ . Figure 5a shows that the reward correction term  $\Delta r$  initially cannot distinguish expert and non-expert data well, but adapts and converges quickly. After a few training steps,  $\Delta r$  can reward expert data and punish non-expert very well. We also perform ablation on the effect of diverse types of imperfect rewards  $\tilde{r}$  on  $\Delta r$  and  $\hat{r}$ . Figure 5b shows that a perfect  $\tilde{r}$  is beneficial in obtaining a clear classification, and an incorrect  $\tilde{r}$  can be counterproductive. Whereas, RGM can correct the wrong rewards by a large margin and gives reasonable rewards. Similar effects can also be observed on  $\Delta r$ , as Figure 5c shows.

# 6 DISCUSSION AND CONCLUSION

In this paper, we propose RGM (Reward Gap Minimization), a unified offline policy optimization approach for diverse settings of imperfect rewards. RGM is formulated as a bi-level optimization problem, which achieves reward correction and simultaneous policy learning in a fully offline paradigm. Extensive experiments and illustrative examples show that RGM can perform robust policy optimization under imperfect rewards. Several desirable properties are also identified in the corrected rewards learned by RGM. One limitation of RGM is the need for a small expert dataset, which may not be easily accessible in some applications. However, RGM relaxes the strong dependencies on online reward tuning to obtain high-quality rewards, without relying on diverse expert demonstrations and tedious human efforts, which renders it a powerful tool to solve many real-world problems.

# REFERENCES

Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning, pp. 1, 2004.  
David Abel, Will Dabney, Anna Harutyunyan, Mark K Ho, Michael Littman, Doina Precup, and Satinder Singh. On the expressivity of markov reward. Advances in Neural Information Processing Systems, 34:7799-7812, 2021.  
Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline reinforcement learning with diversified q-ensemble. Advances in neural information processing systems, 34:7436-7447, 2021.  
Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, and Zhao ran Wang. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. In International Conference on Learning Representations, 2021.  
Vivek S Borkar. Stochastic approximation with two time scales. Systems & Control Letters, 29(5): 291-294, 1997.  
Stephen Boyd, Stephen P Boyd, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.  
Ching-An Cheng, Tengyang Xie, Nan Jiang, and Alekh Agarwal. Adversarily trained actor critic for offline reinforcement learning. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 3852-3878. PMLR, 2022.  
Nuttapong Chentanez, Andrew Barto, and Satinder Singh. Intrinsically motivated reinforcement learning. Advances in neural information processing systems, 17, 2004.  
Benoit Colson, Patrice Marcotte, and Gilles Savard. An overview of bilevel optimization. Annals of operations research, 153(1):235-256, 2007.  
Mark Cutler and Jonathan P How. Autonomous drifting using simulation-aided reinforcement learning. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pp. 5442-5448. IEEE, 2016.  
Bo Dai, Niao He, Yunpeng Pan, Byron Boots, and Le Song. Learning from conditional distributions via dual embeddings. In Artificial Intelligence and Statistics, pp. 1458-1467. PMLR, 2017.  
Rati Devidze, Goran Radanovic, Parameswaran Kamalaruban, and Adish Singla. Explicable reward design for reinforcement learning agents. Advances in Neural Information Processing Systems, 34: 20118-20131, 2021.  
Marco Dorigo and Marco Colombetti. Robot shaping: Developing autonomous agents through learning. Artificial intelligence, 71(2):321-370, 1994.  
Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Thompson. Implicit behavioral cloning. In Conference on Robot Learning, pp. 158-168. PMLR, 2022.  
Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.  
Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in Neural Information Processing Systems, 34, 2021.  
Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, pp. 2052-2062. PMLR, 2019.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139-144, 2020.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861-1870. PMLR, 2018.  
Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. Inverse reward design. Advances in neural information processing systems, 30, 2017.  
Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale framework for bilevel optimization: Complexity analysis and application to actor-critic. arXiv preprint arXiv:2007.05170, 2020.  
Yujing Hu, Weixun Wang, Hangtian Jia, Yixiang Wang, Yingfeng Chen, Jianye Hao, Feng Wu, and Changjie Fan. Learning to utilize shaping rewards: A new approach of reward shaping. Advances in Neural Information Processing Systems, 33:15931-15941, 2020.  
Max Jaderberg, Wojciech M Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castaneda, Charles Beattie, Neil C Rabinowitz, Ari S Morcos, Avraham Ruderman, et al. Human-level performance in 3d multiplayer games with population-based reinforcement learning. Science, 364(6443):859-865, 2019.  
Firas Jarboui and Vianney Perchet. Offline inverse reinforcement learning. arXiv preprint arXiv:2106.05068, 2021.  
Geon-Hyeong Kim, Seokin Seo, Jongmin Lee, Wonseok Jeon, HyeongJoo Hwang, Hongseok Yang, and Kee-Eung Kim. Demodice: Offline imitation learning with supplementary imperfect demonstrations. In International Conference on Learning Representations, 2021.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster), 2015.  
Ilya Kostrikov, Ofir Nachum, and Jonathan Tompson. Imitation learning via off-policy distribution matching. In International Conference on Learning Representations, 2019.  
Ilya Kostrikov, Rob Fergus, Jonathan Tompson, and Ofir Nachum. Offline reinforcement learning with fisher divergence critic regularization. In International Conference on Machine Learning, pp. 5774-5783. PMLR, 2021a.  
Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In International Conference on Learning Representations, 2021b.  
Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. Advances in Neural Information Processing Systems, 32, 2019.  
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179-1191, 2020.  
Jongmin Lee, Wonseok Jeon, Byungjun Lee, Joelle Pineau, and Kee-Eung Kim. Optidice: Offline policy optimization via stationary distribution correction estimation. In International Conference on Machine Learning, pp. 6120-6130. PMLR, 2021.  
Jianxiong Li, Xianyuan Zhan, Haoran Xu, Xiangyu Zhu, Jingjing Liu, and Ya-Qin Zhang. Distance-sensitive offline reinforcement learning. arXiv preprint arXiv:2205.11027, 2022.  
Xiao Li, Zachary Serlin, Guang Yang, and Calin Belta. A formal methods approach to interpretable reinforcement learning for robotic planning. Science Robotics, 4(37):eaay6276, 2019.  
Yecheng Ma, Andrew Shen, Dinesh Jayaraman, and Osbert Bastani. Versatile offline imitation from observations and examples via regularized state-occupancy matching. In International Conference on Machine Learning, pp. 14639–14663. PMLR, 2022.  
Ofir Marom and Benjamin Rosman. Belief reward shaping in reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Bhaskara Marthi. Automatic shaping and decomposition of reward functions. In Proceedings of the 24th International Conference on Machine learning, pp. 601-608, 2007.  
Ofir Nachum and Bo Dai. Reinforcement learning via fenchel-rockafellar duality. arXiv preprint arXiv:2001.01866, 2020.  
Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in Neural Information Processing Systems, 32, 2019a.  
Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074, 2019b.  
Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml, volume 99, pp. 278-287, 1999.  
Andrew Y Ng, Stuart Russell, et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, pp. 2, 2000.  
Haoyi Niu, Sharma Shubham, Yiwen Qiu, Ming Li, Guyue Zhou, Jianming Hu, and Xianyuan Zhan. When to trust your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning. In Advances in Neural Information Processing Systems, 2022.  
Art B. Owen. Monte Carlo theory, methods and examples. 2013.  
Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Advances in neural information processing systems, 1, 1988.  
Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.  
Nived Rajaraman, Lin Yang, Jiantao Jiao, and Kannan Ramchandran. Toward the fundamental limits of imitation learning. Advances in Neural Information Processing Systems, 33:2914-2924, 2020.  
Jette Randlov and Preben Alstrom. Learning to drive a bicycle using reinforcement learning and shaping. In ICML, volume 98, pp. 463-471. CiteSeer, 1998.  
R Tyrrell Rockafellar and Roger J-B Wets. Variational analysis, volume 317. Springer Science & Business Media, 2009.  
Fumihiro Sasaki and Ryota Yamashina. Behavioral cloning from noisy demonstrations. In International Conference on Learning Representations, 2020.  
Ankur Sinha, Pekka Malo, and Kalyanmoy Deb. A review on bilevel optimization: from classical to evolutionary approaches and applications. IEEE Transactions on Evolutionary Computation, 22 (2):276-295, 2017.  
Yang Song and Diederik P Kingma. How to train your energy-based models. arXiv preprint arXiv:2101.03288, 2021.  
Jonathan Sorg, Richard L Lewis, and Satinder Singh. Reward design via online gradient ascent. Advances in Neural Information Processing Systems, 23, 2010.  
Hao Sun, Lei Han, Rui Yang, Xiaoteng Ma, Jian Guo, and Bolei Zhou. Exploiting reward shifting in value-based deep rl. In Advances in Neural Information Processing Systems, 2022.  
Yue Wu, Shuangfei Zhai, Nitish Srivastava, Joshua M Susskind, Jian Zhang, Ruslan Salakhutdinov, and Hanlin Goh. Uncertainty weighted actor-critic for offline reinforcement learning. In International Conference on Machine Learning, pp. 11319-11328. PMLR, 2021.  
Yueh-Hua Wu and Shou-De Lin. A low-cost ethics shaping approach for designing reinforcement learning agents. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Haoran Xu, Xianyuan Zhan, Jianxiong Li, and Honglei Yin. Offline reinforcement learning with soft behavior regularization. arXiv preprint arXiv:2110.07395, 2021.  
Haoran Xu, Jiang Li, Jianxiong Li, and Xianyuan Zhan. A policy-guided imitation approach for offline reinforcement learning. In Advances in Neural Information Processing Systems, 2022a.  
Haoran Xu, Xianyuan Zhan, Honglei Yin, and Huiling Qin. Discriminator-weighted offline imitation learning from suboptimal demonstrations. In International Conference on Machine Learning, pp. 24725-24742. PMLR, 2022b.  
Haoran Xu, Xianyuan Zhan, and Xiangyu Zhu. Constraints penalized q-learning for safe offline reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022c.  
Xianyuan Zhan, Haoran Xu, Yue Zhang, Xiangyu Zhu, Honglei Yin, and Yu Zheng. Deepthermal: Combustion optimization for thermal power generating units using offline reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.  
Wenjia Zhang, Haoran Xu, Haoyi Niu, Peng Cheng, Ming Li, Heming Zhang, Guyue Zhou, and Xianyuan Zhan. Discriminator-guided model-based offline imitation learning. In Conference on Robot Learning, 2022.  
Zeyu Zheng, Junhyuk Oh, and Satinder Singh. On learning intrinsic rewards for policy gradient methods. Advances in Neural Information Processing Systems, 31, 2018.  
Konrad Zolna, Alexander Novikov, Ksenia Konyushkova, Caglar Gulcehre, Ziyu Wang, Yusuf Aytar, Misha Denil, Nando de Freitas, and Scott Reed. Offline learning from demonstrations and unlabeled experience. arXiv preprint arXiv:2011.13885, 2020.
