# SHARING KNOWLEDGE IN MULTI-TASK DEEP REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the benefit of sharing representations among tasks to enable the effective use of deep neural networks in Multi-Task Reinforcement Learning. We leverage the assumption that learning from different tasks, sharing common properties, is helpful to generalize the knowledge of them resulting in a more effective feature extraction compared to learning a single task. Intuitively, the resulting set of features offers performance benefits when used by Reinforcement Learning algorithms. We prove this by providing theoretical guarantees that highlight the conditions for which is convenient to share representations among tasks, extending the well-known finite-time bounds of Approximate Value-Iteration to the multi-task setting. In addition, we complement our analysis by proposing multi-task extensions of three Reinforcement Learning algorithms that we empirically evaluate on widely used Reinforcement Learning benchmarks showing significant improvements over the single-task counterparts in terms of sample efficiency and performance.

# 1 INTRODUCTION

Multi-Task Learning (MTL) ambitiously aims to learn multiple tasks jointly instead of learning them separately, leveraging the assumption that the considered tasks have common properties which can be exploited by Machine Learning (ML) models to generalize the learning of each of them. For instance, the features extracted in the hidden layers of a neural network trained on multiple tasks have the advantage of being a general representation of structures common to each other. This translates into an effective way of learning multiple tasks at the same time, but it can also improve the learning of each individual task compared to learning them separately (Caruana, 1997). Furthermore, the learned representation can be used to perform Transfer Learning (TL), i.e. using it as a preliminary knowledge to learn a new similar task resulting in a more effective and faster learning than learning the new task from scratch (Baxter, 2000; Thrun & Pratt, 2012).

The same benefits of extraction and exploitation of common features among the tasks achieved in MTL, can be obtained in Multi-Task Reinforcement Learning (MTRL) when training a single agent on multiple Reinforcement Learning (RL) problems with common structures (Taylor & Stone, 2009; Lazaric, 2012). In particular, in MTRL an agent can be trained on multiple tasks in the same domain, e.g. riding a bicycle or cycling while going towards a goal, or on different but similar domains, e.g. balancing a pendulum or balancing a double pendulum<sup>1</sup>. Considering recent advances in Deep Reinforcement Learning (DRL) and the resulting increase in the complexity of experimental benchmarks, the use of Deep Learning (DL) models, e.g. deep neural networks, has become a popular and effective way to extract common features among tasks in MTRL algorithms (Rusu et al., 2015; Liu et al., 2016; Higgins et al., 2017). However, despite the high representational capacity of DL models, the extraction of good features remains challenging. For instance, the performance of the learning process can degrade when unrelated tasks are used together (Caruana, 1997; Baxter, 2000); another detrimental issue may occur when the training of a single model is not balanced properly among multiple tasks (Hessel et al., 2018).

Recent developments in MTRL achieve significant results in feature extraction by means of algorithms specifically developed to address these issues. While some of these works rely on a single deep neural network to model the multi-task agent (Liu et al., 2016; Yang et al., 2017; Hessel et al., 2018), others

use multiple deep neural networks, e.g. one for each task and another for the multi-task agent (Rusu et al., 2015; Parisotto et al., 2015; Higgins et al., 2017; Teh et al., 2017). Intuitively, achieving good results in MTRL with a single deep neural network is more desirable than using many of them, since the training time is likely much less and the whole architecture is easier to implement. In this paper we study the benefits of shared representations among tasks. We theoretically motivate the intuitive effectiveness of our method, deriving theoretical guarantees that exploit the theoretical framework provided by Maurer et al. (2016), in which the authors present upper bounds on the quality of learning in MTL when extracting features for multiple tasks in a single shared representation. The significance of this result is that the cost of learning the shared representation decreases with a factor  $\mathcal{O}(1 / \sqrt{T})$ , where  $T$  is the number of tasks for many function approximator hypothesis classes. The main contribution of this work is twofold.

1. We derive upper confidence bounds for Approximate Value-Iteration (AVI) and Approximate Policy-Iteration  $(\mathrm{API})^2$  (Farahmand, 2011) in the MTRL setting, and we extend the approximation error bounds in Maurer et al. (2016) to the case of multiple tasks with different dimensionalities. Then, we show how to combine these results resulting in, to the best of our knowledge, the first proposed extension of the finite-time bounds of AVI/API to MTRL. Despite being an extension of previous works, we derive these results to justify our approach showing how the error propagation in AVI/API can theoretically benefit from learning multiple tasks jointly.  
2. We leverage these results proposing a neural network architecture, for which these bounds hold with minor assumptions, that allow us to learn multiple tasks with a single regressor extracting a common representation. We show an empirical evidence of the consequence of our bounds by means of a variant of Fitted  $Q$ -Iteration (FQI) (Ernst et al., 2005), based on our shared network and for which our bounds apply, that we call Multi Fitted  $Q$ -Iteration (MFQI). Then, we perform an empirical evaluation in challenging RL problems proposing multi-task variants of the Deep  $Q$ -Network (DQN) (Mnih et al., 2015) and Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) algorithms. These algorithms are practical implementations of the more general AVI/API framework, designed to solve complex problems. In this case, the bounds apply to these algorithms only with some assumptions, e.g. stationary sampling distribution. The outcome of the empirical analysis joins the theoretical results, showing significant performance improvements compared to the single-task version of the algorithms in various RL problems, including several MuJoCo (Todorov et al., 2012) domains.

# 2 PRELIMINARIES

Let  $B(\mathcal{X})$  be the space of bounded measurable functions w.r.t. the  $\sigma$ -algebra  $\sigma_{\mathcal{X}}$ , and similarly  $B(\mathcal{X}, L)$  be the same bounded by  $L < \infty$ .

A Markov Decision Process (MDP) is defined as a 5-tuple  $\mathcal{M} = < S, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$  is the transition distribution where  $\mathcal{P}(s'|s,a)$  is the probability of reaching state  $s'$  when performing action  $a$  in state  $s$ ,  $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$  is the reward function, and  $\gamma \in (0,1]$  is the discount factor. A deterministic policy  $\pi$  maps, for each state, the action to perform:  $\pi: \mathcal{S} \to \mathcal{A}$ . Given a policy  $\pi$ , the value of an action  $a$  in a state  $s$  represents the expected discounted cumulative reward obtained by performing  $a$  in  $s$  and following  $\pi$  thereafter:  $Q^{\pi}(s,a) \triangleq \mathbb{E}[\sum_{k=0}^{\infty} \gamma^{k} r_{i+k+1} | s_{i} = s, a_{i} = a, \pi]$ , where  $r_{i+1}$  is the reward obtained after the  $i$ -th transition. The expected discounted cumulative reward is maximized by following the optimal policy  $\pi^{*}$  which is the one that determines the optimal action values, i.e., the ones that satisfy the Bellman optimality equation (Bellman, 1954):  $Q^{*}(s,a) \triangleq \int_{\mathcal{S}} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a,s') + \gamma \max_{a'} Q^{*}(s',a')] ds'$ . The solution of the Bellman optimality equation is the fixed point of the optimal Bellman operator  $\mathcal{T}^{*}: B(\mathcal{S} \times \mathcal{A}) \to B(\mathcal{S} \times \mathcal{A})$  defined as  $(\mathcal{T}^{*}Q)(s,a) \triangleq \int_{\mathcal{S}} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a,s') + \gamma \max_{a'} Q(s',a')] ds'$ . In the MTRL setting, there are multiple MDPs  $\mathcal{M}^{(t)} = < S^{(t)}, A^{(t)}, \mathcal{P}^{(t)}, \mathcal{R}^{(t)}, \gamma^{(t)}>$  where  $t \in \{1,\dots,T\}$  and  $T$  is the number of MDPs. For each MDP  $\mathcal{M}^{(t)}$ , a deterministic policy  $\pi_t: S^{(t)} \to A^{(t)}$  induces an action-value

function  $Q_{t}^{\pi_{t}}(s^{(t)},a^{(t)}) = \mathbb{E}[\sum_{k = 0}^{\infty}\gamma^{k}r_{i + k + 1}^{(t)}|s_{i} = s^{(t)},a_{i} = a^{(t)},\pi_{t}]$ . In this setting, the goal is to maximize the sum of the expected cumulative discounted reward of each task.

In our theoretical analysis of the MTRL problem, the complexity of representation plays a central role. Given a set of input samples  $\bar{\mathbf{X}}\in \mathcal{X}^{Tn}$  and a class of functions  $\mathcal{H}$ , the Gaussian complexity of a random set  $\mathcal{H}(\bar{\mathbf{X}}) = \{(h_k(X_{ti})):h\in \mathcal{H}\} \subseteq \mathbb{R}^{KTn}$  is defined as follows:

$$
G \left(\mathcal {H} (\bar {\mathbf {X}})\right) = \mathbb {E} \left[ \sup  _ {h \in \mathcal {H}} \sum_ {t k i} \gamma_ {t k i} h _ {k} \left(X _ {t i}\right) \Bigg | X _ {t i} \right], \tag {1}
$$

where  $\gamma_{tki}$  are independent standard normal variables. We also need to define the following quantity, taken from Maurer (2016): let  $\gamma$  be a vector of  $m$  random standard normal variables, and  $f\in \mathcal{F}: Y\to \mathbb{R}^m$ , with  $Y\subseteq \mathbb{R}^n$ , we define

$$
O (\mathcal {F}) = \sup  _ {y, y ^ {\prime} \in Y, y \neq y ^ {\prime}} \mathbb {E} \left[ \sup  _ {f \in \mathcal {F}} \frac {\langle \gamma , f (y) - f \left(y ^ {\prime}\right) \rangle}{\| y - y ^ {\prime} \|} \right]. \tag {2}
$$

Equation 2 can be viewed as a Gaussian average of Lipschitz quotients, and appears in the bounds provided in this work. Finally, we define  $L(\mathcal{F})$  as the upper bound of the Lipschitz constant of all the functions  $f$  in the function class  $\mathcal{F}$ .

# 3 THEORETICAL ANALYSIS

The following theoretical study starts from the derivation of theoretical guarantees for MTRL in the AVI framework, extending the results of Farahmand (2011) in the MTRL scenario. Then, to bound the approximation error term in the AVI bound, we extend the result described in Maurer (2006) to MTRL. As we discuss, the resulting bounds described in this section clearly show the benefit of sharing representation in MTRL. To the best of our knowledge, this is the first general result for MTRL; previous works have focused on finite MDPs (Brunskill & Li, 2013) or linear models (Lazaric & Restelli, 2011).

# 3.1 MULTI-TASK REPRESENTATION LEARNING

The multi-task representation learning problem consists in learning simultaneously a set of  $T$  tasks  $\mu_t$ , modeled as probability measures over the space of the possible input-output pairs  $(x,y)$ , with  $x \in \mathcal{X}$  and  $y \in \mathbb{R}$ , being  $\mathcal{X}$  the input space. Let  $w \in \mathcal{W}: \mathcal{X} \to \mathbb{R}^J$ ,  $h \in \mathcal{H}: \mathbb{R}^J \to \mathbb{R}^K$  and  $f \in \mathcal{F}: \mathbb{R}^K \to \mathbb{R}$  be functions chosen from their respective hypothesis classes. The functions in the hypothesis classes must be Lipschitz continuous functions. Let  $\bar{\mathbf{Z}} = (\mathbf{Z}_1, \ldots, \mathbf{Z}_T)$  be the multi-sample over the set of tasks  $\boldsymbol{\mu} = (\mu_1, \dots, \mu_T)$ , where  $\mathbf{Z}_t = (Z_{t1}, \dots, Z_{tn}) \sim \mu_t^n$  and  $Z_{ti} = (X_{ti}, Y_{ti}) \sim \mu_t$ . We can formalize our regression problem as the following minimization problem:

$$
\min  \left\{\frac {1}{n T} \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {N} \ell \left(f _ {t} \left(h \left(w _ {t} \left(X _ {t i}\right)\right)\right), Y _ {t i}\right): \mathbf {f} \in \mathcal {F} ^ {T}, h \in \mathcal {H}, \mathbf {w} \in \mathcal {W} ^ {T} \right\}, \tag {3}
$$

where we use  $\mathbf{f} = (f_1, \dots, f_T)$ ,  $\mathbf{w} = (w_1, \dots, w_T)$ , and define the minimizers of Equation (3) as  $\hat{\mathbf{w}}$ ,  $\hat{h}$ , and  $\hat{\mathbf{f}}$ . We assume that the loss function  $\ell: \mathbb{R} \times \mathbb{R} \to [0,1]$  is 1-Lipschitz in the first argument for every value of the second argument. While this assumption may seem restrictive, the result obtained can be easily scaled to the general case. To use the principal result of this section, for a generic loss function  $\ell'$ , it is possible to use  $\ell(\cdot) = \ell'(\cdot) / \epsilon_{\max}$ , where  $\epsilon_{\max}$  is the maximum value of  $\ell'$ . The expected loss over the tasks, given  $\mathbf{w}$ ,  $h$  and  $\mathbf{f}$  is the task-averaged risk:

$$
\varepsilon_ {\mathrm {a v g}} (\mathbf {w}, h, \mathbf {f}) = \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {E} [ \ell (f _ {t} (h (w _ {t} (X))), Y) ] \tag {4}
$$

The minimum task-averaged risk, given the set of tasks  $\pmb{\mu}$  and the hypothesis classes  $\mathcal{W},\mathcal{H}$  and  $\mathcal{F}$  is  $\varepsilon_{\mathrm{avg}}^{*}$ , and the corresponding minimizers are  $\mathbf{w}^*$ ,  $h^*$  and  $\mathbf{f}^*$ .

# 3.2 MULTI-TASK APPROXIMATE VALUE ITERATION BOUND

We start by considering the bound for the AVI framework which applies for the single-task scenario.

Theorem 1. (Theorem 3.4 of Farahmand (2011)) Let  $K$  be a positive integer, and  $Q_{max} \leq \frac{R_{max}}{1 - \gamma}$ . Then for any sequence  $(Q_k)_{k=0}^K \subset B(\mathcal{S} \times \mathcal{A}, Q_{max})$  and the corresponding sequence  $(\varepsilon_k)_{k=0}^{K-1}$ , where  $\varepsilon_k = \|Q_{k+1} - \mathcal{T}^* Q_k\|_\nu^2$ , we have:

$$
\| Q ^ {*} - Q ^ {\pi_ {K}} \| _ {1, \rho} \leq \frac {2 \gamma}{(1 - \gamma) ^ {2}} \left[ \inf  _ {r \in [ 0, 1 ]} C _ {V I, \rho , \nu} ^ {\frac {1}{2}} (K; r) \mathcal {E} ^ {\frac {1}{2}} (\varepsilon_ {0}, \dots , \varepsilon_ {K - 1}; r) + \frac {2}{1 - \gamma} \gamma^ {K} R _ {m a x} \right], \tag {5}
$$

where

$$
\begin{array}{l} C _ {V I, \rho , \nu} (K; r) = \left(\frac {1 - \gamma}{2}\right) ^ {2} \sup  _ {\pi_ {1} ^ {\prime}, \dots , \pi_ {K} ^ {\prime}} \sum_ {k = 0} ^ {K - 1} a _ {k} ^ {2 (1 - r)} \left[ \sum_ {m \geq 0} \gamma^ {m} \left(c _ {V I _ {1}, \rho , \nu} (m, K - k; \pi_ {K} ^ {\prime}) \right. \right. \\ \left. \left. + c _ {V I _ {2}, \rho , \nu} \left(m + 1; \pi_ {k + 1} ^ {\prime}, \dots , \pi_ {K} ^ {\prime}\right)\right) \right] ^ {2}, \tag {6} \\ \end{array}
$$

with  $\mathcal{E}(\varepsilon_0,\ldots ,\varepsilon_{K - 1};r) = \sum_{k = 0}^{K - 1}\alpha_k^{2r}\varepsilon_k$ , the two coefficients  $c_{VI_1,\rho ,\nu},c_{VI_2,\rho ,\nu}$ , the distributions  $\rho$  and  $\nu$ , and the series  $\alpha_{k}$  are defined as in Farahmand (2011).

In the multi-task scenario, let the average approximation error across tasks be:

$$
\varepsilon_ {\mathrm {a v g}, k} \left(\hat {\mathbf {w}} _ {k}, \hat {h} _ {k}, \hat {\mathbf {f}} _ {k}\right) = \frac {1}{T} \sum_ {t = 1} ^ {T} \left\| Q _ {t, k + 1} - \mathcal {T} ^ {*} Q _ {t, k} \right\| _ {\nu} ^ {2}, \tag {7}
$$

where  $Q_{t,k + 1} = \hat{f}_{t,k} \circ \hat{h}_k \circ \hat{w}_{t,k}$ .

In the following, we extend the AVI bound of Theorem 1 to the multi-task scenario, by computing the average loss across tasks and pushing inside the average using Jensen's inequality.

Theorem 2. Let  $K$  be a positive integer, and  $Q_{\max} \leq \frac{R_{\max}}{1 - \gamma}$ . Then for any sequence  $(Q_k)_{k=0}^K \subset B(\mathcal{S} \times \mathcal{A}, Q_{\max})$  and the corresponding sequence  $(\varepsilon_{avg,k})_{k=0}^{K-1}$ , where  $\varepsilon_{avg,k} = \frac{1}{T} \sum_{t=1}^{T} \|Q_{t,k+1} - \mathcal{T}^* Q_{t,k}\|_\nu^2$ , we have:

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \| Q _ {t} ^ {*} - Q _ {t} ^ {\pi_ {K}} \| _ {1, \rho} \leq \frac {2 \gamma}{(1 - \gamma) ^ {2}} \left[ \inf  _ {r \in [ 0, 1 ]} C _ {V I} ^ {\frac {1}{2}} (K; r) \mathcal {E} _ {\text {a v g}} ^ {\frac {1}{2}} \left(\varepsilon_ {\text {a v g}, 0}, \dots , \varepsilon_ {\text {a v g}, K - 1}; r\right) + \frac {2 \gamma^ {K} R _ {\max , a v g}}{1 - \gamma} \right] \tag {8}
$$

$$
\begin{array}{l} w i t h \mathcal {E} _ {a v g} = \sum_ {k = 0} ^ {K - 1} \alpha_ {k} ^ {2 r} \varepsilon_ {a v g, k}, \gamma = \max  _ {t \in \{1, \dots , T \}} \gamma_ {t}, C _ {V I} ^ {\frac {1}{2}} (K; r) = \max  _ {t \in \{1, \dots , T \}} C _ {V I, \rho , \nu} ^ {\frac {1}{2}} (K; t, r), R _ {m a x, a v g} = \\ \frac {1}{T} \sum_ {t = 1} ^ {T} R _ {m a x, t} a n d \alpha_ {k} = \left\{ \begin{array}{l l} \frac {(1 - \gamma) \gamma^ {K - k - 1}}{1 - \gamma^ {K + 1}} & 0 \leq k <   K, \\ \frac {(1 - \gamma) \gamma^ {K}}{1 - \gamma^ {K + 1}} & k = K \end{array} \right.. \\ \end{array}
$$

Remarks Theorem 2 retains most of the properties of Theorem 3.4 of Farahmand (2011), except that the regression error in the bound is now task-averaged. Interestingly, the second term of the sum in Equation (8) depends on the average maximum reward for each task. In order to obtain this result we use an overly pessimistic bound on  $\gamma$  and the concentrability coefficients, however this approximation is not too loose if the MDPs are sufficiently similar.

# 3.3 MULTI-TASK APPROXIMATION ERROR BOUND

We bound the task-averaged approximation error  $\varepsilon_{\mathrm{avg}}$  at each AVI iteration  $k$  involved in (8) following a derivation similar to the one proposed by Maurer et al. (2016), obtaining:

Theorem 3. Let  $\mu, \mathcal{W}, \mathcal{H}$  and  $\mathcal{F}$  be defined as above and assume  $0 \in \mathcal{H}$  and  $f(0) = 0, \forall f \in \mathcal{F}$ . Then for  $\delta > 0$  with probability at least  $1 - \delta$  in the draw of  $\bar{\mathbf{Z}} \sim \prod_{t=1}^{T} \mu_t^n$  we have that

$$
\begin{array}{l} \varepsilon_ {a v g} (\hat {\mathbf {w}}, \hat {h}, \hat {\mathbf {f}}) \leq L (\mathcal {F}) \left(c _ {1} \frac {L (\mathcal {H}) \sup _ {l \in \{1 , . . . , T \}} G (\mathcal {W} (\mathbf {X} _ {l}))}{n} + c _ {2} \frac {\sup _ {\mathbf {w}} \| \mathbf {w} (\bar {\mathbf {X}}) \| O (\mathcal {H})}{n T}\right) \\ \left. + c _ {3} \frac {\operatorname* {m i n} _ {p \in P} G (\mathcal {H} (p))}{n T}\right) + c _ {4} \frac {\operatorname* {s u p} _ {h , \mathbf {w}} \| h (\mathbf {w} (\bar {\mathbf {X}})) \| O (\mathcal {F})}{n \sqrt {T}} + \sqrt {\frac {8 \ln \left(\frac {3}{\delta}\right)}{n T}} + \varepsilon_ {a v g} ^ {*}. \tag {9} \\ \end{array}
$$

Remarks The assumptions  $0 \in \mathcal{H}$  and  $f(0) = 0$  for all  $f \in \mathcal{F}$  are not essential for the proof and are only needed to simplify the result. For reasonable function classes, the Gaussian complexity  $G(\mathcal{W}(\mathbf{X}_l))$  is  $\mathcal{O}(\sqrt{n})$ . If  $\sup_{\mathbf{w}} \| \mathbf{w}(\bar{\mathbf{X}}) \|$  and  $\sup_{h,\mathbf{w}} \| h(\mathbf{w}(\bar{\mathbf{X}})) \|$  can be uniformly bounded, then they are  $\mathcal{O}(\sqrt{nT})$ . For some function classes, the Gaussian average of Lipschitz quotients  $O(\cdot)$  can be bounded independently from the number of samples. Given these assumptions, the first and the fourth term of the right hand side of Equation (9), which represent respectively the cost of learning the meta-state space  $\mathbf{w}$  and the task-specific  $\mathbf{f}$  mappings, are both  $\mathcal{O}(1/\sqrt{n})$ . The second term represents the cost of learning the multi-task representation  $h$  and is  $\mathcal{O}(1/\sqrt{nT})$ , thus vanishing in the multi-task limit  $T \to \infty$ . The third term can be removed if  $\forall h \in \mathcal{H}, \exists p_0 \in P : h(p) = 0$ ; even when this assumption does not hold, this term can be ignored for many classes of interest, e.g. neural networks, as it can be arbitrarily small.

The last term to be bounded in (9) is the minimum average approximation error  $\varepsilon_{\mathrm{avg}}^{*}$  at each AVI iteration  $k$ . Recalling that the task-averaged approximation error is defined as in (7), applying Theorem 5.3 by Farahmand (2011) we obtain:

Lemma 4. Let  $Q_{t,k}^{*}, \forall t \in \{1, \dots, T\}$  be the minimizers of  $\varepsilon_{avg,k}^{*}$ ,  $\check{t}_k = \arg \max_{t \in \{1, \dots, T\}} \| Q_{t,k+1}^* - \mathcal{T}^* Q_{t,k} \|_\nu^2$ , and  $b_{k,i} = \| Q_{\check{t}_k,i+1} - \mathcal{T}^* Q_{\check{t}_k,i} \|_\nu$ , then:

$$
\varepsilon_ {a v g, k} ^ {*} \leq \left(\| Q _ {\tilde {t} _ {k}, k + 1} ^ {*} - (\mathcal {T} ^ {*}) ^ {k + 1} Q _ {\tilde {t} _ {k}, 0} \| _ {\nu} + \sum_ {i = 0} ^ {k - 1} \left(\gamma_ {\tilde {t} _ {k}} C _ {A E} (\nu ; \check {t} _ {k}, P)\right) ^ {i + 1} b _ {k, k - 1 - i}\right) ^ {2}, \tag {10}
$$

with  $C_{AE}$  defined as in Farahmand (2011).

Final remarks The bound for MTRL is derived by composing the results in Theorems 2 and 3, and Lemma 4. The results above highlight the advantage of learning a shared representation. The bound in Theorem 2 shows that a small approximation error is critical to improve the convergence towards the optimal action-value function, and the bound in Theorem 3 shows that the cost of learning the shared representation at each AVI iteration is mitigated by using multiple tasks. This is particularly beneficial when the feature representation is complex, e.g. deep neural networks.

# 3.4 DISCUSSION

As stated in the remarks of Equation (9), the benefit of MTRL is evinced by the second component of the bound, i.e. the cost of learning  $h$ , which vanishes with the increase of the number of tasks. Obviously, adding more tasks require the shared representation to be large enough to include all of them, undesirably causing the term  $\sup_{h,\mathbf{w}}\| h(\mathbf{w}(\overline{\mathbf{X}}))\|$  in the fourth component of the bound to increase. This introduces a tradeoff between the number of features and number of tasks; however, for a reasonable number of tasks the number of features used in the single-task case is enough to handle them, as we show in some experiments in Section 5. Notably, since the AVI/API framework provided by Farahmand (2011) provides an easy way to include the approximation error of a generic function approximator, it is easy to show the benefit in MTRL of the bound in Equation (9). Despite being just multi-task extensions of previous works, our results are the first one to theoretically show the benefit of sharing representation in MTRL. Moreover, they serve as a significant theoretical motivation, besides to the intuitive ones, of the practical algorithms that we describe in the following sections.

# 4 SHARING REPRESENTATIONS

We want to empirically evaluate the benefit of our theoretical study in the problem of jointly learning  $T$  different tasks  $\mu_t$ , introducing a neural network architecture for which our bounds hold. Following

![](images/9b3aae8e687710f3b3df0018d5c42678e12fbc83e4fa1c585e05d5b1d7e9dd5b.jpg)  
(a) Shared network

![](images/51252fa07090dead39b057035ca5be7c7089b37b6869041897428c3647e14c71.jpg)  
(b) FQI vs MFQI

![](images/8fa921f2a2b2b4222a467c2c485665c0b605aa0a8622bd110b064e20af58d9a0.jpg)

![](images/ac4bb5c16a7984c840575b626a18fdce95603356c6291243edc3d4bce59fdc74.jpg)  
(c) #Task analysis  
Figure 1: (a) The architecture of the neural network we propose to learn  $T$  tasks simultaneously. The  $w_{t}$  block maps each input  $x_{t}$  from task  $\mu_{t}$  to a shared set of layers  $h$  which extracts a common representation of the tasks. Eventually, the shared representation is specialized in block  $f_{t}$  and the output  $y_{t}$  of the network is computed. Note that each block can be composed of arbitrarily many layers. (b) Results of FQI and MFQI averaged over 4 tasks in Car-On-Hill, showing  $\| Q^{*} - Q^{\pi K}\|$  on the left, and the discounted cumulative reward on the right. (c) Results of MFQI showing  $\| Q^{*} - Q^{\pi K}\|$  for increasing number of tasks. Both results in (b) and (c) are averaged over 100 experiments, and show the  $95\%$  confidence intervals.

our theoretical framework, the network we propose extracts representations  $w_{t}$  from inputs  $x_{t}$  for each task  $\mu_t$ , mapping them to common features in a set of shared layers  $h$ , specializing the learning of each task in respective separated layers  $f_{t}$ , and finally computing the output  $y_{t} = (f_{t} \circ h \circ w_{t})(x_{t}) = f_{t}(h(w_{t}(x_{t})))$  (Figure 1(a)). The idea behind this architecture is not new in the literature. For instance, similar ideas have already been used in DQN variants to improve exploration on the same task via bootstrapping (Osband et al., 2016) and to perform MTRL (Liu et al., 2016).

The intuitive and desirable property of this architecture is the exploitation of the regularization effect introduced by the shared representation of the jointly learned tasks. Indeed, unlike learning a single task that may end up in overfitting, forcing the model to compute a shared representation of the tasks helps the regression process to extract more general features, with a consequent reduction in the variance of the learned function. This intuitive justification for our approach, joins the theoretical benefit proven in Section 3. Note that our architecture can be used in any MTRL problem involving a regression process; indeed, it can be easily used in value-based methods as a  $Q$ -function regressor, or in policy search as a policy regressor. In both cases, the targets are learned for each task  $\mu_t$  in its respective output block  $f_t$ . Remarkably, as we show in the experimental Section 5, it is straightforward to extend RL algorithms to their multi-task variants only through the use of the proposed network architecture, without major changes to the algorithms themselves.

# 5 EXPERIMENTAL RESULTS

To empirically evince the effect described by our bounds, we propose an extension of FQI (Ernst et al., 2005; Riedmiller, 2005), that we call MFQI, for which our AVI bounds apply. Then, to empirically evaluate our approach in challenging RL problems, we introduce multi-task variants of two well-known DRL algorithms: DQN (Mnih et al., 2015) and DDPG (Lillicrap et al., 2015), which we call Multi Deep  $Q$ -Network (MDQN) and Multi Deep Deterministic Policy Gradient (MDDPG) respectively. Note that for these methodologies, our AVI and API bounds hold only with the simplifying assumption that the samples are i.i.d.; nevertheless they are useful to show the benefit of our method also in complex scenarios, e.g. MuJoCo (Todorov et al., 2012). We remark that in these experiments we are only interested in showing the benefit of learning multiple tasks with a shared representation w.r.t. learning a single task; therefore, we only compare our methods with the single task counterparts, ignoring other works on MTRL in literature. Refer to Appendix C for all the details and our motivations about the experimental settings.

# 5.1 MULTIFITTED  $Q$ -ITERATION

As a first empirical evaluation, we consider FQI, as an example of an AVI algorithm, to show the effect described by our theoretical AVI bounds in experiments. We consider the Car-On-Hill problem as described in Ernst et al. (2005), and select four different tasks from it changing the mass of the car and the value of the actions (details in Appendix C). Then, we run separate instances of FQI

![](images/33b7f7f99f04ae27d29b18812decc26826efbde19c48fa63f420429a6c590c5e.jpg)  
Figure 2: Discounted cumulative reward averaged over 100 experiments of DQN and MDQN for each task and for transfer learning in the Acrobot problem. An epoch consists of 1,000 steps, after which the greedy policy is evaluated for 2,000 steps. The  $95\%$  confidence intervals are shown.

with a single task network for each task respectively, and one of MFQI considering all the tasks simultaneously. Figure 1(b) shows the  $L_{1}$ -norm of the difference between  $Q^{*}$  and  $Q^{\pi_K}$  averaged over all the tasks. It is clear how MFQI is able to get much closer to the optimal  $Q$ -function, thus giving an empirical evidence of the AVI bounds in Theorem 2. For completeness, we also show the advantage of MFQI w.r.t. FQI in performance. Then, in Figure 1(c) we provide an empirical evidence of the benefit of increasing the number of tasks in MFQI in terms of both quality and stability.

# 5.2 MULTI DEEP  $Q$ -NETWORK

As in Liu et al. (2016), our MDQN uses separate replay memories for each task and the batch used in each training step is built picking the same number of samples from each replay memory. Furthermore, a step of the algorithm consists of exactly one step in each task. These are the only minor changes to the vanilla DQN algorithm we introduce, while all other aspects, such as the use of the target network, are not modified. Thus, the time complexity of MDQN is considerably lower than vanilla DQN thanks to the learning of  $T$  tasks with a single model, but at the cost of a higher memory complexity for the collection of samples for each task. We consider five problems with similar state spaces, sparse rewards and discrete actions: Cart-Pole, Acrobot, Mountain-Car, Car-On-Hill, and Inverted-Pendulum. The implementation of the first three problems is the one provided by the OpenAI Gym library Brockman et al. (2016), while Car-On-Hill is described in Ernst et al. (2005) and Inverted-Pendulum in Lagoudakis & Parr (2003).

Figure 2(a) shows the performance of MDQN w.r.t. to vanilla DQN that uses a single-task network structured as the multi-task one in the case with  $T = 1$ . The first three plots from the left show good performance of MDQN, which is both higher and more stable than DQN. In Car-On-Hill, MDQN is slightly slower than DQN to reach the best performance, but eventually manages to be more stable. Finally, the Inverted-Pendulum experiment is clearly too easy to solve for both approaches, but it is still useful for the shared feature extraction in MDQN. The described results provide important hints about the better quality of the features extracted by MDQN w.r.t. DQN. To further demonstrate this, we evaluate the performance of DQN on Acrobot, arguably the hardest of the five problems, using a single-task network with the shared parameters in  $h$  initialized with the weights of a multi-task network trained with MDQN on the other four problems. Arbitrarily, the pre-trained weights can be adjusted during the learning of the new task or can be kept fixed and only the remaining randomly initialized parameters in  $\mathbf{w}$  and  $\mathbf{f}$  are trained. From Figure 2(b), the advantages of initializing the weights are clear. In particular, we compare the performance of DQN without initialization w.r.t. DQN with initialization in three settings: in Unfreeze-0 the initialized weights are adjusted, in No-Unfreeze they are kept fixed, and in Unfreeze-10 they are kept fixed until epoch 10 after which they start to be optimized. Interestingly, keeping the shared weights fixed shows a significant performance improvement in the earliest epochs, but ceases to improve soon. On the other hand, the adjustment of weights from the earliest epochs shows improvements only compared to the uninitialized network in the intermediate stages of learning. The best results are achieved by starting to adjust the shared weights after epoch 10, which is approximately the point at which the improvement given by the fixed initialization starts to lessen.

![](images/89f286680178d385bbb04e6c8de6dccdaf87413c902b2d6f1a8ccee9e90de010.jpg)  
(a) Multi-task for pendulums

![](images/fdad93cb81d0ee2b6fa5a71f8e1f1a21e3c09895c71f31644211d334f60eae7d.jpg)  
(b) Transfer for pendulums

![](images/25d85bd891a794f45149d2fc28f34c6c8bc1caade2ec65b48d455e1a44e93fb1.jpg)

![](images/9ea5368b977dc09ecb71ea8da320ef11d96ca03ea9b1a04f9f4aa6cad863a622.jpg)  
(c) Multi-task for walkers

![](images/8cf4e8e641b2d267d7d7ae9af95407d6e57ee36ac51d06f4ac8f3172a68fe7b3.jpg)  
Figure 3: Discounted cumulative reward averaged over 40 experiments of DDPG and MDDPG for each task and for transfer learning in the Inverted-Double-Pendulum and Hopper problems. An epoch consists of 10,000 steps, after which the greedy policy is evaluated for 5,000 steps. The  $95\%$  confidence intervals are shown.

![](images/74eabb01e785b7ad5b077962554ee53fcb51475a946c0f1ae8aedb5158999c19.jpg)

![](images/0a7a5eca4c004f4c0a0c56f934839f803704f75fe7fa884fabb2abbf31671304.jpg)  
(d) Transfer for walkers

# 5.3 MULTI DEEP DETERMINISTIC POLICY GRADIENT

In order to show how the flexibility of our approach easily allows to perform MTRL in policy search algorithms, we propose MDDPG as a multi-task variant of DDPG. As an actor-critic method, DDPG requires an actor network and a critic network. Intuitively, to obtain MDDPG both the actor and critic networks should be built following our proposed structure. We perform separate experiments on two sets of MuJoCo Todorov et al. (2012) problems with similar continuous state and action spaces: the first set includes Inverted-Pendulum, Inverted-Double-Pendulum, and Inverted-Pendulum-Swingup as implemented in the pybullet library, whereas the second set includes Hopper-Stand, Walker-Walk, and Half-Cheetah-Run as implemented in the DeepMind Control SuiteTassa et al. (2018). Figure 3(a) shows a relevant improvement of MDDPG w.r.t. DDPG in the pendulum tasks. Indeed, while in Inverted-Pendulum, which is the easiest problem among the three, the performance of MDDPG is only slightly better than DDPG, the difference in the other two problems is significant. The advantage of MDDPG is confirmed in Figure 3(c) where it performs better than DDPG in Hopper and equally good in the other two tasks. Again, we perform a TL evaluation of DDPG in the problems where it suffers the most, by initializing the shared weights of a single-task network with the ones of a multi-task network trained with MDDPG on the other problems. Figures 3(b) and 3(d) show evident advantages of pre-training the shared weights and a significant difference between keeping them fixed or not.

# 6 CONCLUSION

We have theoretically proved the advantage in RL of using a shared representation to learn multiple tasks w.r.t. learning a single task. We have derived our results extending the AVI/API bounds (Farahmand, 2011) to MTRL, leveraging the upper bounds on the approximation error in MTL provided in Maurer et al. (2016). The results of this analysis show that the error propagation during the AVI/API iterations is reduced according to the number of tasks. Then, we proposed a practical way of exploiting this theoretical benefit which consists in an effective way of extracting shared representations of multiple tasks by means of deep neural networks. To empirically show the advantages of our method, we carried out experiments on challenging RL problems with the introduction of multi-task extensions of FQI, DQN, and DDPG based on the neural network structure we proposed. As desired, the favorable empirical results confirm the theoretical benefit we described.

# REFERENCES

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Advances in Neural Information Processing Systems, pp. 5048-5058, 2017.  
Jonathan Baxter. A model of inductive bias learning. Journal of Artificial Intelligence Research, 12: 149-198, 2000.  
Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47: 253-279, 2013.  
Richard Bellman. The theory of dynamic programming. Technical report, RAND Corp Santa Monica CA, 1954.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Emma Brunskill and Lihong Li. Sample complexity of multi-task reinforcement learning. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence, pp. 122-131. AUAI Press, 2013.  
Rich Caruana. Multitask learning. Machine learning, 28(1):41-75, 1997.  
Giovanni Cavallanti, Nicolo Cesa-Bianchi, and Claudio Gentile. Linear algorithms for online multitask classification. Journal of Machine Learning Research, 11(Oct):2901-2934, 2010.  
Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research, 6(Apr):503-556, 2005.  
Amir-massoud Farahmand. Regularization in reinforcement learning. 2011.  
Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, and Hado van Hasselt. Multi-task deep reinforcement learning with popart. arXiv preprint arXiv:1809.04474, 2018.  
Irina Higgins, Arka Pal, Andrei Rusu, Loic Matthey, Christopher Burgess, Alexander Pritzel, Matthew Botvinick, Charles Blundell, and Alexander Lerchner. Darla: Improving zero-shot transfer in reinforcement learning. In International Conference on Machine Learning, pp. 1480-1490, 2017.  
Michail G Lagoudakis and Ronald Parr. Least-squares policy iteration. Journal of machine learning research, 4(Dec):1107-1149, 2003.  
Alessandro Lazaric. Transfer in reinforcement learning: a framework and a survey. In Reinforcement Learning, pp. 143-173. Springer, 2012.  
Alessandro Lazaric and Mohammad Ghavamzadeh. Bayesian multi-task reinforcement learning. In ICML-27th International Conference on Machine Learning, pp. 599-606. Omnipress, 2010.  
Alessandro Lazaric and Marcello Restelli. Transfer from multiple mdps. In Advances in Neural Information Processing Systems, pp. 1746-1754, 2011.  
Alessandro Lazaric, Marcello Restelli, and Andrea Bonarini. Transfer of samples in batch reinforcement learning. In Proceedings of the 25th international conference on Machine learning, pp. 544-551. ACM, 2008.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
Lydia Liu, Urun Dogan, and Katja Hofmann. Decoding multitask dqn in the world of apache. In European Workshop on Reinforcement Learning, 2016.

Andreas Maurer. Bounds for linear multi-task learning. Journal of Machine Learning Research, 7 (Jan):117-139, 2006.  
Andreas Maurer. A chain rule for the expected suprema of gaussian processes. Theoretical Computer Science, 650:109-122, 2016.  
Andreas Maurer, Massimiliano Pontil, and Bernardino Romera-Paredes. The benefit of multitask representation learning. The Journal of Machine Learning Research, 17(1):2853-2884, 2016.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. In Advances in neural information processing systems, pp. 4026-4034, 2016.  
Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. arXiv preprint arXiv:1511.06342, 2015.  
Martin Riedmiller. Neural fitted q iteration-first experiences with a data efficient neural reinforcement learning method. In European Conference on Machine Learning, pp. 317-328. Springer, 2005.  
Andrei A Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy distillation. arXiv preprint arXiv:1511.06295, 2015.  
Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In International Conference on Machine Learning, pp. 1312-1320, 2015.  
Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdelmaleki, Josh Merel, Andrew Lefrancq, Timothy P. Lillicrap, and Martin A. Riedmiller. Deepmind control suite. CoRR, abs/1801.00690, 2018. URL http://arxiv.org/abs/1801.00690.  
Matthew E Taylor and Peter Stone. Transfer learning for reinforcement learning domains: A survey. Journal of Machine Learning Research, 10(Jul):1633-1685, 2009.  
Matthew E Taylor, Peter Stone, and Yaxin Liu. Transfer learning via inter-task mappings for temporal difference learning. Journal of Machine Learning Research, 8(Sep):2125-2167, 2007.  
Yee Teh, Victor Bapst, Wojciech M Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. Distral: Robust multitask reinforcement learning. In Advances in Neural Information Processing Systems, pp. 4496-4506, 2017.  
Sebastian Thrun and Lorien Pratt. Learning to learn. Springer Science & Business Media, 2012.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026-5033. IEEE, 2012.  
Aaron Wilson, Alan Fern, Soumya Ray, and Prasad Tadepalli. Multi-task reinforcement learning: a hierarchical bayesian approach. In Proceedings of the 24th international conference on Machine learning, pp. 1015-1022. ACM, 2007.  
Zhaoyang Yang, Kathryn E Merrick, Hussein A Abbass, and Lianwen Jin. Multi-task deep reinforcement learning for continuous action control. In *IJCAI*, pp. 3301-3307, 2017.
