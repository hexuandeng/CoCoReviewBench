# METRICS MATTER: A CLOSER LOOK ON SELF-PACED REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Curriculum reinforcement learning (CRL) allows to solve complex tasks by generating a tailored sequence of learning tasks, starting from easy ones and subsequently increasing their difficulty. However, the generation of such task sequences is largely governed by application assumptions, often preventing a theoretical investigation of existing approaches. Recently, Klink et al. (2021) showed how self-paced learning induces a principled interpolation between task distributions in the context of RL, resulting in high learning performance. So far, this interpolation is unfortunately limited to Gaussian distributions. Here, we show that on one side, this parametric restriction is insufficient in many learning cases but that on the other, the interpolation of self-paced RL (SPRL) can be degenerate when not restricted to this parametric form. We show that the introduction of concepts from optimal transport into SPRL prevents aforementioned issues. Experiments demonstrate that the resulting introduction of metric structure into the curriculum allows for a well-behaving non-parametric version of SPRL that leads to stable learning performance across tasks.

# 1 INTRODUCTION

Reinforcement learning (RL) (Sutton & Barto, 1998) has celebrated great successes as a framework for autonomous acquisition of desired behavior. With ever-increasing computational power, this framework and the algorithms developed under it have allowed to create learning agents capable of solving non-trivial long-horizon planning (Mnih et al., 2015; Silver et al., 2017) and control tasks (Akkaya et al., 2019). However, these successes have also highlighted the need for certain forms of regularization, such as leagues in the context of boardgames (Silver et al., 2017), a gradual diversification of simulated training environments for robotic manipulation (Akkaya et al., 2019) or a tailored training pipeline in the context of humanoid control for soccer (Liu et al., 2021). These regularizations help to overcome shortcomings of modern RL agents, such as poor exploratory behavior - a problem that is an active topic of research (Bellemare et al., 2016; Ghavamzadeh et al., 2015; Machado et al., 2020).

One can view aforementioned regularizations under the umbrella term of curriculum reinforcement learning (Narvekar et al., 2020), where the idea is to avoid shortcomings of modern (deep) RL agents such as aforementioned poor exploration by learning on a tailored sequence of tasks. Such curricula can materialize in a variety of ways and are motivated from many perspectives in the literature (Andrychowicz et al., 2017; Florensa et al., 2017; Wohlke et al., 2020). Although the resulting curricula can often be interpreted as a sequence of task distributions, these sequences typically lack a formal connection to the reinforcement learning objective of maximizing the expected reward under a given target task distribution. In a recent line of work, Klink et al. (2021) proposed the idea of self-paced reinforcement learning (SPRL), borrowing from the concept of self-paced learning that has been established in the supervised learning literature (Kumar et al., 2010; Jiang et al., 2015; Meng et al., 2017). Klink et al. showed a connection between a regularized RL objective and a sequence of task distributions that trade-off between yielding high expected reward and tasks likely under the target distribution. This interpolant has, however, so far been restricted to Gaussian distributions (Klink et al., 2020a;b; 2021). While successful in experimental evaluations, this Gaussian assumption clearly imposes a limitation on the flexibility of the curriculum and disconnects the algorithmic implementation from the established theory. This disconnect raises the question whether the observed performance of SPRL is due to the Gaussian approximation.

Contribution: The key insight presented in this paper is that the Gaussian approximation of existing SPRL implementations is indeed important for the practical performance of SPRL, as it masks the weakness of the KL divergence to express task distribution similarity in a CRL setting. In more detail, we show that

- Parametric assumptions in SPRL hinder the learning performance in task spaces in which only a non-euclidean subspace is learnable by the agent. Leaving these parametric assumptions behind allows to overcome this problem.  
In the non-parametric regime, SPRL can fail to facilitate learning on the target task distribution due to the usage of the KL divergence as a measure between task distributions.  
Wasserstein metrics - that take the metric structure of the task space into account - ensure a meaningful interpolation in aforementioned failure cases, providing higher performance.

Apart from achieving higher empirical performance in the SPRL framework, the results presented in this paper indicate that introducing a notion of metric structure on the task space may be an important next step for deriving a principled, yet practical, understanding of CRL.

# 2 RELATED WORK

The main focus of this work is on self-paced reinforcement learning (SPRL) (Klink et al., 2020a; b; 2021) that takes the concept of self-paced curriculum learning (Kumar et al., 2010) from supervised learning to reinforcement learning (RL). Opposed to supervised learning, where there is ongoing discussion about the mechanics of curricula and their effect in different situations (Weinshall & Amir, 2020; Wu et al., 2021), the mechanics seem to be more agreed upon in RL. In RL, curricula typically alleviate the problem of challenging exploration and/or try to maximize the learning speed by focusing on tasks in which the policy is still capable of improving its performance. These mechanics reduce the typically high sample complexity of modern RL algorithms, which e.g. arises from simple heuristics used for exploration such as random Gaussian noise on actions (Silver et al., 2014; Schulman et al., 2015; Haarnoja et al., 2018). Although the reason for the benefit of curricula in RL tend to be agreed upon, curriculum RL (CRL) algorithms are motivated from many different perspectives. They typically follow a particular heuristic to generate task sequences, such as the notion of learning progress (Portelas et al., 2019) or intermediate task difficulty (Florensa et al., 2018). While effective in practice, such heuristics prevent a formal analysis of the proposed algorithms. SPRL has been shown to perform an interpolation between task distributions (Klink et al., 2021), allowing to relate the effect of a curriculum to the concept of annealing in statistics (Neal, 2001). We wish to add to this formal understanding by investigating the interpolation produced by SPRL more closely, particularly in the context of curriculum reinforcement learning.

As this investigation will lead us to the problem of optimal transport, we wish to point out important literature in this field. Dating back to the work by Monge in the 18th century, optimal transport has been understood as an important fundamental concept touching upon many fields in both theory and application (Liu et al., 2019; Peyre et al., 2019; Chen et al., 2021). In probability theory, optimal transport translates to the so-called Wasserstein metric (Kantorovich, 1942) between two distributions that compares them under a given metric on the sample space. From a computational perspective, the use of entropic regularization (Cuturei, 2013) has led to tangible speed-ups in computations revolving around optimal transport and is hence widely applied (Feydy et al., 2019).

# 3 PRELIMINARIES

This section serves to introduce the necessary background on (contextual) RL, self-paced RL and optimal transport.

# 3.1 CONTEXTUAL REINFORCEMENT LEARNING

Contextual reinforcement learning can be seen as a conceptual extension to the (single task) reinforcement learning (RL) problem. (Single task) RL aims to maximize an expected reward objective by finding an optimal policy  $\pi : S \times \mathcal{A} \mapsto \mathbb{R}$  for a given MDP  $\mathcal{M} = \langle S, \mathcal{A}, p, r, p_0 \rangle$

$$
\max  _ {\pi} J (\pi) = \max  _ {\pi} \mathbb {E} _ {p _ {0} \left(\mathbf {s} _ {0}\right), p \left(\mathbf {s} _ {t + 1} \mid \mathbf {s} _ {t}, \mathbf {a} _ {t}\right), \pi \left(\mathbf {a} _ {t} \mid \mathbf {s} _ {t}\right)} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r \left(\mathbf {s} _ {t}, \mathbf {a} _ {t}\right) \right]. \tag {1}
$$

Contextual RL extends this objective to a space of MDPs  $\mathcal{M}(\mathbf{c}) = \langle S, \mathcal{A}, p_{\mathbf{c}}, r_{\mathbf{c}}, p_{0,\mathbf{c}} \rangle$  indexed by a parameter  $\mathbf{c} \in \mathcal{C}$ . The distributions  $p_0$  and  $p_{0,\mathbf{c}}$  in (1) represent distributions over initial states  $\mathbf{s}_0$  and  $p$  as well as  $p_{\mathbf{c}}$  the transition dynamics of the environment. In the contextual setting, the policy  $\pi : \mathcal{S} \times \mathcal{C} \times \mathcal{A} \mapsto \mathbb{R}$  is further conditioned on the contextual parameter  $\mathbf{c}$  and the objective is extended via an additional distribution  $\mu : \mathcal{C} \mapsto \mathbb{R}$  over contextual variables  $\mathbf{c}$

$$
\max  _ {\pi} J (\pi , \mu) = \max  _ {\pi} \mathbb {E} _ {\mu (\mathbf {c})} [ J (\pi , \mathbf {c}) ]. \tag {2}
$$

The distribution  $\mu (\mathbf{c})$  encodes the tasks  $\mathcal{M}(\mathbf{c})$  that the agent is expected to encounter. The objective  $J(\pi ,\mathbf{c})$  in Eq. (2) corresponds to the objective  $J(\pi)$  in Eq. (1) where, however, the initial state distribution  $p_0$ , the transition dynamics  $p$  as well as the reward function  $r$  of  $\mathcal{M}$  are replaced by their counterparts in  $\mathcal{M}(\mathbf{c})$ . This contextual model of optimal decision making is well-suited for learning in multiple related tasks as is the case in multi-task (Wilson et al., 2007), goal-conditioned (Schaul et al., 2015) or curriculum RL (Narvekar et al., 2020).

# 3.2 SELF-PACED REINFORCEMENT LEARNING

Self-paced reinforcement learning (SPRL) has been introduced by Klink et al. (2020a;b; 2021) as a curriculum RL algorithm that alters the context distribution  $\mu(\mathbf{c})$  in the contextual RL objective (2) to increase the learning performance of an agent and/or make it less susceptible to local optima of the objective function. SPRL computes a surrogate distribution  $p: \mathcal{C} \mapsto \mathbb{R}$  under which to train the RL agent, i.e.

$$
\max _ {\boldsymbol \pi} J (\boldsymbol \pi , p) = \max _ {\boldsymbol \pi} \mathbb {E} _ {p (\mathbf {c})} \left[ J (\boldsymbol \pi , \mathbf {c}) \right].
$$

This surrogate distribution is found by optimizing the KL divergence to the target distribution  $\mu (\mathbf{c})$  subject to two constraints (see Klink et al. (2021, Section 8))

$$
\min  _ {p} D _ {\mathrm {K L}} (p (\mathbf {c}) \| \mu (\mathbf {c})) \text {s . t .} J (\pi , p) \geq \delta \quad D _ {\mathrm {K L}} (p (\mathbf {c}) \| q (\mathbf {c})) \leq \epsilon . \tag {3}
$$

The distribution  $p(\mathbf{c})$  balances between representing similar tasks as the (target) distribution  $\mu(\mathbf{c})$  and encoding tasks in which the agent currently obtains large rewards. The KL divergence constraint w.r.t. the previous context distribution  $q(\mathbf{c})$  is employed to avoid large changes in the distribution  $p(\mathbf{c})$  during subsequent iterations. In practice, this regularization is required as the expected performance  $J(\pi, \mathbf{c})$  in context  $\mathbf{c}$  needs to be estimated from a limited set of samples e.g. via a neural network.

A particularly interesting aspect of this work is that objective (3) can be interpreted to perform a specific interpolation between the distributions  $\mu (\mathbf{c})$ ,  $q(\mathbf{c})$  and a maximum entropy distribution  $p_J(\mathbf{c})\propto \exp (\eta J(\pi ,\mathbf{c}))$  encoding high reward tasks. This interpolation is given by

$$
p _ {\alpha , \eta} (\mathbf {c}) \propto \mu (\mathbf {c}) ^ {\frac {1}{1 + \alpha}} q (\mathbf {c}) ^ {\frac {\alpha}{1 + \alpha}} \exp (\eta J (\pi , \mathbf {c})) ^ {\frac {1}{1 + \alpha}}. \tag {4}
$$

The two parameters  $\alpha$  and  $\eta$  controlling the interpolation are the Lagrangian multipliers of the two constraints in objective (3). So far, the algorithmic realizations of (3) by Klink et al. (2020a;b; 2021) restricted the distribution  $p_{\alpha,\eta}(\mathbf{c})$  to a parametric form with parameters  $\pmb{\nu}$ , in particular Gaussian distributions  $p_{\nu}(\mathbf{c}) = \mathcal{N}(\mathbf{c}|\pmb{\mu},\pmb{\Sigma})$ . In this case, optimizing (3) w.r.t.  $\pmb{\mu}$  and  $\pmb{\Sigma}$  of  $p_{\nu}$  corresponds to performing an I-projection of the analytic optimal distribution (4) to the Gaussian restriction

$$
\min _ {\pmb {\nu}} D _ {\mathrm {K L}} \left( \right.p _ {\pmb {\nu}} (\mathbf {c}) \left\| \right. \frac {1}{Z _ {\alpha , \eta}} \mu (\mathbf {c}) ^ {\frac {1}{1 + \alpha}} q (\mathbf {c}) ^ {\frac {\alpha}{1 + \alpha}} \exp (\eta J (\pi , \mathbf {c})) ^ {\frac {1}{1 + \alpha}}\left. \right).
$$

In above equation,  $Z_{\alpha,\eta}$  is the normalizer of the analytic distribution  $p_{\alpha,\eta}(\mathbf{c})$ . In this work, we are interested in investigating the distribution  $p_{\alpha,\eta}$  outside of this analytic restriction  $p_{\nu}$ , i.e. truly employing the distribution (4) instead of its I-projection to a Gaussian.

# 3.3 OPTIMAL TRANSPORT

The problem of optimally transporting mass between two distributions has been initially investigated by Monge (1781). As of today, generalizations established by Kantorovich (1942) have led to so

Algorithm 1 Self-Paced Reinforcement Learning Implementations  
Input: Context dist.  $p_0(\mathbf{c})$ , target context dist.  $\mu (\mathbf{c})$ , performance threshold  $\delta$ , distance bound  $\epsilon$  for  $k = 0$  to  $K$  do Agent Improvement: Sample contexts  $\mathbf{c}_i\sim p_k(\mathbf{c})$ ,  $i\in [1,M]$  Train policy  $\pi$  under  $\mathbf{c}_i$  and observe episodic rewards  $R_{i} = \sum_{t = 1}^{\infty}r_{\mathbf{c}_{i}}(\mathbf{s}_{t},\mathbf{a}_{t})$ ,  $i\in [1,M]$  Estimate  $J(\pi ,\mathbf{c})$  from the dataset  $\mathcal{D} = \{(c_i,R_i)|i\in [1,M]\}$  Context Distribution Update: G-SPRL Optimize (3) w.r.t.  $\mu_{k + 1}$  and  $\Sigma_{k + 1}$ , where  $p_{k + 1}(\mathbf{c}) = \mathcal{N}(\mathbf{c}|\boldsymbol {\mu}_{k + 1},\boldsymbol{\Sigma}_{k + 1})$  NP-SPRL Optimize (3) w.r.t  $\alpha$  and  $\eta$  using a discrete approximation  $\bar{\mathbf{p}}_{\alpha ,\eta}\approx p_{\alpha ,\eta}(\mathbf{c})$  (4) WB-SPRL Optimize (7) w.r.t.  $\beta$  to obtain  $p_{\beta}(\mathbf{c})$  end for

called Wasserstein distances as metrics between probability distributions defined on a metric space  $M = (d, \mathcal{C})$  with metric  $d: \mathcal{C} \times \mathcal{C} \mapsto \mathbb{R}_{\geq 0}$

$$
\begin{array}{l} \mathcal {W} _ {p} \left(p _ {1}, p _ {2}\right) = \left(\inf  _ {\gamma \in \Gamma \left(p _ {1}, p _ {2}\right)} \int_ {\mathcal {C} \times \mathcal {C}} d \left(\mathbf {c} _ {1}, \mathbf {c} _ {2}\right) ^ {p} \mathrm {d} \gamma \left(\mathbf {c} _ {1}, \mathbf {c} _ {2}\right)\right) ^ {1 / p}, \quad p \geq 1 \\ \Gamma (p _ {1}, p _ {2}) = \left\{\gamma : \mathcal {C} \times \mathcal {C} \mapsto \mathbb {R} _ {\geq 0} \middle | p _ {1} (\mathbf {c} _ {1}) = \int \gamma (\mathbf {c} _ {1}, \mathbf {c} _ {2}) \mathrm {d} \mathbf {c} _ {2}, p _ {2} (\mathbf {c} _ {2}) = \int \gamma (\mathbf {c} _ {1}, \mathbf {c} _ {2}) \mathrm {d} \mathbf {c} _ {1} \right\} \\ \end{array}
$$

The distance between  $p_1$  and  $p_2$  results from solving an optimization problem that finds a so-called plan  $\gamma$ . This plan encodes how to move density from  $p_1$  to  $p_2$  taking into account the cost of moving density between between parts of the space  $\mathcal{C}$ . This cost is encoded by the metric  $d$ . In the following, we will always assume to work with 2-Wasserstein distances under euclidean metric, i.e.  $p = 2$  and  $d(\mathbf{c}_1,\mathbf{c}_2) = \| \mathbf{c}_1 - \mathbf{c}_2\| _2$ . Particularly interesting for the investigation conducted in this paper are so called Wasserstein barycenters

$$
p _ {\mathcal {W} _ {2}} (\mathbf {c}) = \arg \min  _ {p} \sum_ {k = 1} ^ {K} w _ {k} \mathcal {W} _ {2} (p, p _ {k}), \quad \sum_ {k = 1} ^ {K} w _ {k} = 1, \tag {5}
$$

that perform a weighted interpolation between a set of distributions  $p_k$  by computing a distribution  $p$  that minimizes the weighted sum of Wasserstein distances to them.

# 4 NON-PARAMETERIC SELF-PACED REINFORCEMENT LEARNING

We now describe the two implementations of SPRL that will be evaluated in the experimental section of this paper. The first implementation – NP-SPRL– is a faithful implementation of the SPRL objective (3) that computes and uses the distribution  $p_{\alpha, \eta}(\mathbf{c})$  (4) by discretizing the context space  $\mathcal{C}$  to avoid Gaussian approximations. Furthermore, we instantiate Wasserstein barycenters in the SPRL framework under the name WB-SPRL. As shown in Algorithm 1, all implementations (including the Gaussian one from Klink et al. (2021)) only differ in the computation of the context distributions. Hence, contrasting these implementations sheds light on the importance of metrics in the context of CRL.

# 4.1 NP-SPRL

Computing the analytic distribution  $p_{\alpha, \eta}(\mathbf{c})$  and adjusting the parameters  $\alpha$  and  $\eta$  via the SPRL objective (3) will require approximations in the general case, as even sampling arbitrary continuous distributions is already an open research problem (Brooks et al., 2011; Liu & Wang, 2016; Liu et al., 2019; Wibisono, 2018). Using approximate methods resulting from this research would, however, directly interfere with our intent of evaluating the behavior of  $p_{\alpha, \eta}(\mathbf{c})$  as exactly as possible. Consequently, we discretize the continuous context spaces in our experiments in order to faithfully sample and evaluate expectations w.r.t.  $p_{\alpha, \eta}(\mathbf{c})$ . This is schematically shown in Figure 1 for one of the evaluation environments. Although such a discretization will clearly not scale gracefully to higher dimensions, it allows us to investigate the true behavior of SPRL, which, as we will show, can be sub-optimal even for low-dimensional context spaces.

With a context space  $\mathcal{C} \subseteq \mathbb{R}^d$  discretized into  $N$  cells, we can represent  $p_{\alpha,\eta}$  via a vector  $\bar{\mathbf{p}}_{\alpha,\eta} \in \mathbb{R}_{\geq 0}^{N}$ .

![](images/381da3ce57228a1046dbd178d872d73939fbffd5c74069c547a89c8f5cd68852.jpg)  
(a) Maze Environment

![](images/291bf7ca2defaddf726cb626a88a451a44a77918c1eb9900ee4fb161b1de2840.jpg)  
Figure 1: (a) The first environment for evaluation of SPRL is a maze task simulated in MuJoCo (Todorov et al., 2012), in which a point-mass needs to move in a maze of circular shape to reach desired target positions. (b) The target position is encoded via the contextual variable  $\mathbf{c} \in \mathcal{C} \subseteq \mathbb{R}^2$ . The area highlighted in red visualizes the initial position of the point mass and the walls of the environment are shown in black.  $(c + d)$  In order to faithfully evaluate expectations over functions or compute KL divergences e.g. in the SPRL objective (3), we discretize the context space  $\mathcal{C}$ .  
(b) Context Space  $\mathcal{C}$

![](images/12f577918265be1ecf3366e6cf14de692034c7764bdbdf5fbc216dcf7008c154.jpg)  
(c) Function  $f$  on  $\mathcal{C}$

![](images/77c9838c198aaa039546d7e64d2701b8e31aeac650562f7eb1b53c0f17e8878d.jpg)  
(d)  $f$  discretized on  $\mathcal{C}$

To sample a continuous context  $\mathbf{c} \in \mathcal{C}$ , we can then first sample an index of a cell from  $\bar{\mathbf{p}}_{\alpha, \eta}$  and then sample uniformly within this cell to obtain a context  $\mathbf{c}$ . Further, the KL divergences in the SPRL objective (3) are straightforward to evaluate when working with  $\bar{\mathbf{p}}_{\alpha, \eta}$ . To evaluate the expected performance, we simply evaluate the performance measure  $J(\pi, \mathbf{c})$  at the  $N$  cell centers  $\bar{\mathbf{c}}_n$  to obtain the vector  $\bar{\mathbf{J}}(\pi) \in \mathbb{R}^N$ . With that we obtain  $J(\pi, p_{\alpha, \eta}) = \bar{\mathbf{p}}_{\alpha, \eta}^T \bar{\mathbf{J}}(\pi)$ .

# 4.2 WB-SPRL

Before discussing how to incorporate Wasserstein barycenters into SPRL, we first want to answer the question why this may be useful. As mentioned in Section 3.3, the distinguishing feature of optimal transport is the ability to take metrics defined on the sample space, i.e. the context space  $\mathcal{C}$  in the case of curriculum RL, into account. Given that the contextual variables  $\mathbf{c} \in \mathcal{C}$  are representing MDPs  $\mathcal{M}(\mathbf{c})$ , we can introduce a metric on MDPs into SPRL by employing optimal transport approaches on  $\mathcal{C}$ . This metric can ensure that the curriculum only gradually changes the learning tasks, and with that stabilize learning. As we show in the experiments, this effect is indeed existent and crucial to generate meaningful interpolations between task distributions. To leverage optimal transport in an SPRL style algorithm we realize the interpolation between current distribution  $q(\mathbf{c})$ , target distribution  $\mu(\mathbf{c})$  and "value distribution"  $p_J(\mathbf{c}) \propto \exp(\eta J(\pi, \mathbf{c}))$  by Wasserstein barycenters

$$
p _ {\boldsymbol {\beta}} (\mathbf {c}) = \underset {p} {\arg \min } (1 - \beta_ {1} - \beta_ {2}) \mathcal {W} (p (\mathbf {c}), \mu (\mathbf {c})) + \beta_ {1} \mathcal {W} (p (\mathbf {c}), p _ {J} (\mathbf {c})) + \beta_ {2} \mathcal {W} (p (\mathbf {c}), q (\mathbf {c})). \tag {6}
$$

The weights  $\beta$  of the interpolation are adjusted with the goal of minimizing  $\mathcal{W}_2(p_\beta, \mu)$  while ensuring a constraint on expected performance and distance to the previous distribution  $q(\mathbf{c})$

$$
\min  _ {\boldsymbol {\beta}} \mathcal {W} _ {2} \left(p _ {\boldsymbol {\beta}} (\mathbf {c}), \mu (\mathbf {c})\right) \text {s . t} J (\pi , p _ {\boldsymbol {\beta}}) \geq \delta \quad \mathcal {W} _ {2} \left(p _ {\boldsymbol {\beta}} (\mathbf {c}), q (\mathbf {c})\right) \leq \epsilon . \tag {7}
$$

A difference w.r.t. SPRL is that the parameter  $\eta$  of the value distribution  $p_J(\mathbf{c})$  is not adjusted in optimization problem (7). Instead we adjust  $\eta$  before optimizing (7) such that  $J(\pi, p_J) \geq \delta_H$ , where  $\delta_H > \delta$  is another performance threshold. This ensures that with  $\beta_1 \to 1$ , it holds that  $J(\pi, p_{\beta}) \geq \delta$ . We resort to particle-based representations of the distributions when implementing (6) and (7). This allows us to make use of Monge maps to compute  $p_{\beta}(\mathbf{c})$  efficiently. More details on the implementation of WB-SPRL are provided in appendix B.

# 5 EXPERIMENTS

The experiments in this section serve to show the need for a realization of SPRL without parametric restrictions on  $p_{\alpha, \eta}$  but to also highlight that NP-SPRL is not necessarily well-suited for this endeavour due its ignorance w.r.t. the metric on  $\mathcal{C}$ . Experimental details can be found in appendix C<sup>1</sup>. We refer to the Gaussian realization of SPRL by Klink et al. (2021) as G-SPRL. Although our focus is on evaluating the behavior of SPRL, we also evaluate current state-of-the-art algorithms GOALGAN and ALP-GMM (Florensa et al., 2018; Portelas et al., 2019) in all environments to contrast the results with currently well-performing algorithms. With the final environment defined over a discrete space  $\mathcal{C}$ , we also investigate the algorithm by (Graves et al., 2017), naming it ACL.

![](images/5bc55fb1fd0b66728077eb2d8912ee64ffcbc8b8fea1c51ce3fc1810fbba7b50.jpg)  
(a) Learning Performance

![](images/7151fd5df5e37646b6149690dbd1773b9ef59dd9f22a68d2cd5d5919c124fb0c.jpg)

![](images/77873980cd8455c90def3dccf427537721a5516d128b4490f714efe67ab2d48a.jpg)  
Figure 2: a) Achieved success rate of different curricula and a uniform sampling baseline (Random) over iterations. Mean and standard error are computed from 10 runs. b) Parametric and non-parametric context distributions  $p_{\nu}(\mathbf{c})$  and  $p_{\alpha,\eta}(\mathbf{c})$  for a run of G-SPRL and NP-SPRL respectively. The distributions are represented by 2000 samples drawn from them.  
(b) G-SPRL context distributions  $p_{\nu}$  over iterations  
(c) NP-SPRL context distributions  $p_{\alpha ,\eta}$  over iterations

# 5.1 MAZE

The first environment that we turn to is a sparse-reward, maze-like environment depicted in Figure 1, in which an agent needs to reach a desired goal. Such environments have e.g. been investigated by Florensa et al. (2018). The contexts  $\mathbf{c} \in \mathcal{C}$  of this environment encode the goal position to be reached and hence contains unsolvable contexts (goals inside of a wall or in the inner circle of the maze). Defining  $\mu(\mathbf{c})$  to be uniform over the context space, the curriculum needs to identify and train the agent on the subspace of feasible tasks in order to achieve a good learning performance. This subspace of feasible tasks is highly non-Gaussian, making it an interesting testbed for NP- and WB-SPRL. We discretize the context space  $\mathcal{C} \subseteq \mathbb{R}^2$  of goal positions into a  $50 \times 50$  grid. Figure 2 compares the performance of the different CRL algorithms. We see that both NP- and WB-SPRL perform better than G-SPRL. This is because the performance constraint  $\mathbb{E}_{p(\mathbf{c})}[J(\pi, \mathbf{c})] \geq \delta$  in objective (3) at some point prevents the Gaussian context distribution from expanding, as otherwise too many infeasible tasks would be encoded in  $p_{\nu}(\mathbf{c})$  and hence the performance constraint violated. As shown in Figures 2 and 7, NP- and WB-SPRL can flexibly assign probability to feasible contexts, resulting in an increased learning speed compared to G-SPRL and training on  $\mu(\mathbf{c})$  directly.

# 5.2 POINT-MASS

After this promising first demonstration of NP-SPRL in the previous section, we now turn towards the point-mass environment investigated by (Klink et al., 2020a;b; 2021) to highlight its challenges. As shown in Figure 3a, a point mass needs to be steered through a narrow gate to reach a goal position on the other side of a wall. While Klink et al. only considered a narrow gate at one specific position as the target task, we will investigate a version in which the gate is located at one of two opposing positions  $\mathbf{c}_1 = [-3\ 0.5]$  and  $\mathbf{c}_2 = [3\ 0.5]$ , making  $\mu(\mathbf{c})$  a bi-modal distribution. This again

![](images/6e7c5cdb3e3254c0dbf41e21bcccd42fa185e70d041ba82f3467812cd0eec42e.jpg)  
(a) Environment

![](images/b3de09e987b241f8a6af0cfdb0b0d9b35616412b910834803fb3b22d69bfd51c.jpg)  
Figure 3: (a) The point mass environment with its two-dimensional context space. The target distributions  $\mu_{1}(\mathbf{c})$  and  $\mu_{2}(\mathbf{c})$  encode the two gates with width  $w_{g} = 0.5$ , in which the agent (black dot) is required to navigate through a narrow gate at different positions to reach the goal (red cross). (b) Discounted cumulative return over iterations obtained under different curricula for both distributions. Statistics (mean and standard error) are computed from 10 seeds.  
(b) Learning Performance

![](images/2b7b43ea4ea994a72c0d0a1253f880906d7858f3816c5e840c55e2a38a3e1e4f.jpg)  
(a)  $D_{\mathrm{KL}}(p(\mathbf{c})\parallel \mu_1(\mathbf{c}))$

![](images/cd1095ceb6c905ff71e2051b96967a5d4b4420d9f63648a74dc04930f0846e02.jpg)

![](images/078460c17949ddb2512d966e04b545008e0f3830dbc4d57a5bb96e1b8aceba2d.jpg)

![](images/57ea0805138898f9c83cd7b1d15e62c8e43f3d5b2b1d17c22f5d8b96e2453efa.jpg)

![](images/e4f5db25af66fbc3dac7d7f827205212215ad929a619b86cac47ca5589af2e74.jpg)

![](images/fe4c639dfcdfa2f0c8fad5c392bc0aed769ef19a361e972c95c3ee60849e9a18.jpg)  
(b)  $p_{\alpha ,\eta}(\mathbf{c})$  for different iterations of NP-SPRL

![](images/d9cf49318bf6bd3763fb61e818efa1878f72bb41da1ddd549b99add0cfda025b.jpg)  
Figure 4: (a) Mean KL divergences between  $\mu_1(\mathbf{c})$  and the final context distribution computed by G- and NP-SPRL as well as the minimum and maximum over 10 seeds. (b) and (c) visualize  $p_{\alpha,\eta}(\mathbf{c})$  of NP-SPRL and  $p_{\nu}(\mathbf{c})$  of G-SPRL for increasing iterations of the algorithms (left to right).

![](images/6156bc0848937008827aaff22f2a7858b6e47d2e8075ce7adeb6d2c03c20674e.jpg)  
(c)  $p_{\nu}(\mathbf{c})$  for different iterations of G-SPRL

![](images/3e22855aeec12f3734d5a3e30dafa0b4847682d81d4ca520c77333fd95d8ce3b.jpg)

challenges the Gaussian restriction of  $p_{\nu}(\mathbf{c})$  in the G-SPRL algorithm. We again discretize the context space  $\mathcal{C} \subseteq \mathbb{R}^2$ , encoding position and width of the gate, into a grid of 50 bins on each axis. We investigate two different target distributions

$$
\mu_ {1} (\mathbf {c}) = \frac {1}{2} \mathcal {N} \left(\mathbf {c} _ {1}, 1 0 ^ {- 4} \mathbf {I}\right) + \frac {1}{2} \mathcal {N} \left(\mathbf {c} _ {2}, 1 0 ^ {- 4} \mathbf {I}\right) \qquad \mu_ {2} (\mathbf {c}) = \left\{ \begin{array}{l l} \frac {1}{2} & , \text {i f} \mathbf {c} \in \{\mathbf {c} _ {1}, \mathbf {c} _ {2} \} \\ \approx 0 & , \text {e l s e .} \end{array} \right.
$$

Note that “ $\approx 0$ ” highlights that the probability is not exactly zero but has a value of  $\exp(-1000)$ , as this ensures that  $\mu(\mathbf{c})$  is absolutely continuous w.r.t.  $p_{\alpha,\eta}(\mathbf{c})$  and  $p_{\nu}(\mathbf{c})$ . Absolute continuity is required to compute the KL divergence to  $\mu(\mathbf{c})$ . Samples from the two distributions are, however, visually hard to distinguish. Nonetheless, Figure 3 shows that the performance of G- and NP-SPRL depends drastically on the choice of target distribution. Further, we see that NP-SPRL does not outperform G-SPRL in this environment, although it should be able to match the bi-modal target distributions, which G-SPRL cannot. Finally, we see that WB-SPRL outperforms both other versions of SPRL and achieves a consistent performance for both target distributions. The illustrations in Figures 4 and 5 will help to understand the underlying problem. As evident in Figure 4, NP-SPRL is indeed able to match the target distribution correctly while G-SPRL ultimately covers only one mode of the target density  $\mu_1(\mathbf{c})$ . Consequently, it is surprising that the final performance of NP-SPRL only matches the performance of G-SPRL, although G-SPRL completely ignores one of the two target contexts. Investigating Figure 4b more closely reveals that NP-SPRL only generates a proper curriculum for one of the two contexts (the left one in the images) by gradually interpolating between easy tasks and the target task. As soon as it has gained sufficient proficiency on the (left) target task, NP-SPRL simply incorporates the second target task into  $p_{\alpha,\eta}$ , however without generating an appropriate curriculum for the agent to learn this task. As shown in appendix A, this leads to the agent only solving one target task under the NP-SPRL curriculum, although both target tasks are likely under the final context distribution  $p_{\alpha,\eta}(\mathbf{c})$ . Figure 5 shows that the curricula generated by G- and NP-SPRL for  $\mu_2(\mathbf{c})$  are even more problematic. G-SPRL increases the variance of the Gaussian context distribution to match the constant (negligible) likelihood that is assigned to the non-target tasks. While NP-SPRL again encodes the two target distributions, it now

![](images/8844fb3a81a218ec41850c73eee8c10746814ac2c7f038b954c8854df0c38078.jpg)  
(a)  $D_{\mathrm{KL}}(p(\mathbf{c})\parallel \mu_2(\mathbf{c}))$

![](images/5c5c95b0041c7701443cfbf917faaf3e7785cee18e28f0710967ad7c3d03fb1c.jpg)

![](images/8ba5fd1e98a9349db558fbb80a5673b2c3296e1dc10f0f8f2dea8f72c7d60b7b.jpg)

![](images/e8ce175eff9d11662c9db76125cc4ad5972c5e0520286087169e58bcf8a23f16.jpg)

![](images/6cc9be44bf9b8efa9cf6fce1a9df6f78b2c0363b70370edf3a7cec68b41a3f92.jpg)

![](images/4b925f7a5cbac5b763843e6453e99aa0ec3b29f48522d9ce4ec53bcdce642227.jpg)  
(b)  $p_{\alpha ,\eta}(\mathbf{c})$  for different iterations of NP-SPRL

![](images/14e7c7644ca2376bf9389091b0002ba406012a1d38de0798ff243c0e4710684d.jpg)  
Figure 5: (a) Mean KL divergences between  $\mu_2(\mathbf{c})$  and the final context distribution computed by G- and NP-SPRL as well as the minimum and maximum over 10 seeds. (b) and (c) visualize  $p_{\alpha, \eta}(\mathbf{c})$  of NP-SPRL and  $p_{\nu}(\mathbf{c})$  of G-SPRL for increasing iterations of the algorithms (left to right).

![](images/f5346fc2e0750d808c1d83751688c07a5d8677c05eee37dfec1d5ebc7e11bf96.jpg)  
(c)  $p_{\nu}(\mathbf{c})$  for different iterations of G-SPRL

![](images/92fb739697fa992d303c9aab6eca63f70de200fef1dda5db47ce0eb57bd98596.jpg)

![](images/20d8694bf671a74372a8b7eb2804907d9323a597b0e7f9bf26608edf9dcd1882.jpg)

![](images/734acc3f959b8271da71b473714a26b122ec4a97fa78e3f28d756270fb91a055.jpg)

![](images/4eca41299d5ac5afe99d6a77a5422852c5da588541ddb69a56ddcda138cbb15a.jpg)

![](images/a647f594b1d4b114f2f0ef2e1c9b229077fd86d4193d2a9333c393d54d0e1c4f.jpg)

![](images/84f53600dc578069c5c9367b6052565b4e4ab504c8f61b9c4b7c0f8dc18e921e.jpg)  
(a) KL divergence interpolation: arg  $\min_p\sum_{k = 1}^K w_kD_{\mathrm{KL}}(p\parallel p_k)$  
Figure 6: Interpolations between unimodal (left) and bi-model (right) distributions  $p_1(\mathbf{c})$  and  $p_2(\mathbf{c})$  via a KL divergence-based interpolation and Wasserstein barycenters (5). In one case, all distributions are Gaussians or mixture of Gaussians (blue). In the other case, the distributions are uniform distributions (orange). The Wasserstein barycenters are computed using a particle-based approximation (Feydy et al., 2019). The visualized PDFs are then estimated using a kernel density estimation. This results in a small amount of smoothing for the uniform distributions.

![](images/5a8d6b0c78fdfb432a670445d0bccd2f927aecf7835eefe18acf0b4b1825cc28.jpg)  
(b) Wasserstein barycenter (Equation 5)

![](images/7734e7904726804c157c1e4ee6f96c5df7ea67cc6589b51c3efb36aa67d50567.jpg)

![](images/b827c2a42bb162e85c405d6468e867082bfc0a499eaa1df7001561b761b123a7.jpg)

interpolates between none of them. Consequently, neither the agent under G- nor NP-SPRL learns to solve any of the tasks. These results highlight an important shortcoming of the SPRL distribution (4) in a curriculum reinforcement learning setting. This shortcoming is visualized in Figure 6, comparing Wasserstein barycenters and an interpolation between distributions based on KL divergences. The latter one allows probability mass to directly move between distant parts of the context space, because the KL divergence between two probability distributions  $p_1: \mathcal{C} \mapsto \mathbb{R}$  and  $p_2: \mathcal{C} \mapsto \mathbb{R}$  does not incorporate any notion of a metric defined on  $\mathcal{C}$ . This lack of metric is at the heart of the observed problems, as the SPRL objective (3) basically represents such a KL divergence-based interpolation. Looking at Figure 6b and the curricula generated by WB-SPRL in Figure 7, we see that the notion of a (Euclidean) metric prevents these jumps and leads to a gradual change in the tasks encoded by the curriculum. But why do G-SPRL and (sometimes) NP-SPRL create meaningful interpolations? The reason is in the usage of Gaussian distributions, as visualized in Figure 6a. The log-likelihood of a Gaussian distribution and with that the KL divergence between two Gaussians are both defined via Mahalanobis distances, i.e. a Euclidean distance in a transformed space. Consequently, the Gaussian distributions encode a (Euclidean) metric via their log-likelihood. This further explains the performance drop of G- and NP-SPRL as soon as we switch from  $\mu_1(\mathbf{c})$  to  $\mu_2(\mathbf{c})$ . Without the implicit metric information contained in the Gaussian distribution  $\mu_1(\mathbf{c})$ , NP-SPRL completely breaks down and does not perform any reasonable interpolation. G-SPRL is challenged with matching the shape of  $\mu(\mathbf{c}_2)$  due to its Gaussian restriction.

# 5.3 PICK AND PLACE

As a final experiment, we consider the pick-and-place task in the OpenAI gym environment suite (Brockman et al., 2016) in which a robot is tasked to grasp a block on a table and move it to a desired position (see Figure 8). The sparse reward of this environment, only rewarding the robot upon completing the desired task, makes it a very challenging exploration problem, as can be seen in Figure 8 in which (default) SAC does not learn this pick and place task. One way to alleviate

![](images/1c3caf1721005cee2a9852add477da71a0d74604e3eba11eef4c8f47f2f29caf.jpg)  
(a) Maze environment

![](images/7226b83f692097229f1ab0de62f1dadf37cd94fb103ca895ce2a00c805f2a471.jpg)

![](images/6d78c12494a3cb5661d1ffaf2d8119056638644be9e0be4087bb1960ed022b48.jpg)

![](images/b4aad5867b0342660cf1a3572dc979bf13d838ff10f0ed1eb28a4ef501aca54e.jpg)

![](images/d78c47d8ded7a5079494fe3fb14ad2a45074a915f672e3d5684d74463611ad40.jpg)

![](images/32b39632a66e2f7bf2950470926e004bf35a2fbb5bbd3f696b614d71195ff1a3.jpg)  
Figure 7: Visualizations of the empirical distributions  $p_{\beta}$  generated by WB-SPRL in the maze- (a) and point-mass environments for both target distributions  $\mu_1(\mathbf{c})$  and  $\mu_2(\mathbf{c})$  ((b) + (c)).

![](images/53868f7fc05c412d1074608e5e3c908e729aa49a97542aa0d618cac9531c37f4.jpg)  
(b) Point mass environment with  $\mu_{1}(\mathbf{c})$  
(c) Point mass environment with  $\mu_{2}(\mathbf{c})$

![](images/8473bf306322158e9d15a8d6f2e2c946c530a581bf249a907d40bf7836efd286.jpg)

![](images/ba953b057864bb87eb6119e1f60d5be62fa55ad9947a95c4cd55b94f03021551.jpg)

![](images/5162448ba0ab460958f5065a0f5ad529b5adb68d67955627f51b27523b78abf4.jpg)  
(a) WB-SPRL Curriculum

![](images/56dfbbfed1f1126195d2d8292e290cb35d242805664a815256e1f718b4f44024.jpg)  
Figure 8: (a) A WB-SPRL curriculum for the pick-and-place task. The vertical bars and small images visualize the states corresponding to different time steps of the expert trajectory. (b) Success rates over iterations for different learning algorithms. Mean and standard error is computed from 10 algorithm runs.  
(b) Learning Performance

such challenging exploration problems is to learn the task via a curriculum of starting states. Such starting states can be obtained from an expert demonstration, i.e. a trajectory

$$
\boldsymbol {\tau} _ {\text {E X P E R T}}: [ 0, 1 ] \mapsto \mathcal {S}.
$$

A curriculum over this trajectory is then formally defined by choosing the context space to be the unit interval  $\mathcal{C} = [0,1] \subseteq \mathbb{R}$ . The contextual parameter  $c$  only influences the initial state distribution  $p_{0,c} = \delta_{\mathbf{s} = \tau_{\mathrm{EXPERT}}(c)}$ . We investigate the NP- and WB-SPRL algorithms in this setting by recording one execution of a hand-crafted controller that first moves the end-effector above the block to be grasped, then lowers the end-effector, grasps the block and moves to the target. We slightly randomize the position of the goal as well as the block to enforce that the agent learns a robust policy. In a realistic scenario, we only record the expert demonstration at a discrete set of states  $\{t_i | i = 1,\dots,N\}$  and hence we have a truly discrete CRL setting in this experiment very well suited for NP-SPRL. Furthermore, the finite number of contexts avoids the necessity for function approximators to approximate the expected performance  $J(\pi ,\mathbf{c})$  in a context  $\mathbf{c}$ , as we can simply estimate the performance in each (discrete) context via a sliding window. We again investigate two target context distributions, one being a narrow Gaussian target distribution with mean at  $c = 0$  and negligible variance  $(\mu_1(\mathbf{c}))$  and the other being again a Dirac-delta with a negligible amount of probability in any context to be absolutely continuous w.r.t. any distribution  $(\mu_2(\mathbf{c}))$ . The results in Figure 8 show that WB-SPRL generates a curriculum that allows the agent to learn a reliable policy by smoothly moving probability from later steps of the expert demonstration towards earlier ones. This learned policy is more reliable (25% vs. 100% success rate) and faster (69 vs. 13 steps for task completion) than the demonstration. Both default learning (always starting from the initial state) and NP-SPRL do not lead to successful policies. In appendix A, we show that the noneffectiveness of NP-SPRL is again grounded in the degenerate interpolation that puts no probability mass at intermediate time-steps regardless of the target distribution.

# 6 CONCLUSION

We investigated self-paced reinforcement learning (SPRL) outside of its current parametric restrictions. While this allows to fit complicated target distribution in the framework of SPRL, it also reveals important shortcomings of the KL divergence for measuring the similarity of (task) distributions in curriculum reinforcement learning. Stressing the need for an incorporation of metric assumptions on the context space into the curriculum, we turned towards Wasserstein distances to replace the KL divergences originally employed in SPRL. As we have shown, this alleviates the observed conceptual problems of SPRL. These findings motivate a lot of future work, such as more elaborate implementations of WB-SPRL that e.g. directly move the individual particles of the context distribution via the gradient of the performance measure  $\nabla_{\mathbf{c}}J(\pi ,\mathbf{c})$  instead of using the value distribution  $p_J(\mathbf{c})$  as a proxy for this. Another line of future work may investigate other metrics than the squared euclidean one employed in this paper. More appropriate metrics may increase the performance in settings in which the euclidean distance between two contextual variables is less well-suited to encode the similarity between the MDPs that these variables represent.

# REFERENCES

Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.  
Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Neural Information Processing Systems (NeurIPS), 2017.  
Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. In Neural Information Processing Systems (NeurIPS), 2016.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.  
Steve Brooks, Andrew Gelman, Galin Jones, and Xiao-Li Meng. Handbook of markov chain monte carlo. CRC press, 2011.  
Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrödinger bridge. SIAM Review (SIREV), 63(2):249-313, 2021.  
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Neural Information Processing Systems (NeurIPS), 2013.  
Jean Feydy, Thibault Séjourne, François-Xavier Vialard, Shun-ichi Amari, Alain Trouve, and Gabriel Peyré. Interpolating between optimal transport and mmd using sinkhorn divergences. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.  
Carlos Florensa, David Held, Markus Wulfmeier, Michael Zhang, and Pieter Abbeel. Reverse curriculum generation for reinforcement learning. In Conference on Robot Learning (CoRL), 2017.  
Carlos Florensa, David Held, Xinyang Geng, and Pieter Abbeel. Automatic goal generation for reinforcement learning agents. In International Conference on Machine Learning (ICML), 2018.  
Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, and Aviv Tamar. Bayesian reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 8(5-6):359-483, 2015.  
Alex Graves, Marc G Bellemare, Jacob Menick, Remi Munos, and Koray Kavukcuoglu. Automated curriculum learning for neural networks. In International Conference on Machine Learning (ICML), 2017.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning (ICML), 2018.  
Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Stable baselines. https://github.com/hill-a/stable-baselines, 2018.  
Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, and Alexander G Hauptmann. Self-paced curriculum learning. In AAAI Conference on Artificial Intelligence (AAAI), 2015.  
Leonid Kantorovich. On the transfer of masses (in russian). Doklady Akademii Nauk, 37(2):227-229, 1942.  
Pascal Klink, Hany Abdulsamad, Boris Belousov, and Jan Peters. Self-paced contextual reinforcement learning. In Conference on Robot Learning (CoRL), 2020a.  
Pascal Klink, Carlo D' Eramo, Jan R Peters, and Joni Pajarinen. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Neural Information Processing Systems (NeurIPS), 2020b.

Pascal Klink, Hany Abdulsamad, Boris Belousov, Carlo D'Eramo, Jan Peters, and Joni Pajarinen. A probabilistic interpretation of self-paced learning with applications to reinforcement learning. Journal of Machine Learning Research (JMLR), 22(182):1-52, 2021.  
M Pawan Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable models. In Neural Information Processing Systems (NeurIPS), 2010.  
Chang Liu, Jingwei Zhuo, Pengyu Cheng, Ruiyi Zhang, Jun Zhu, and Lawrence Carin. Understanding and accelerating particle-based variational inference. In International Conference on Machine Learning (ICML), 2019.  
Qiang Liu and Dilin Wang. Stein variational gradient descent: A general purpose bayesian inference algorithm. In Neural Information Processing Systems (NeurIPS), 2016.  
Siqi Liu, Guy Lever, Zhe Wang, Josh Merel, SM Eslami, Daniel Hennes, Wojciech M Czarnecki, Yuval Tassa, Shayegan Omidshafiei, Abbas Abdelmaleki, et al. From motor control to team play in simulated humanoid football. arXiv preprint arXiv:2105.12196, 2021.  
Marlos C Machado, Marc G Bellemare, and Michael Bowling. Count-based exploration with the successor representation. In AAAI Conference on Artificial Intelligence (AAAI), 2020.  
Deyu Meng, Qian Zhao, and Lu Jiang. A theoretical understanding of self-paced learning. Information Sciences, 414:319-328, 2017.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Gaspard Monge. Mémoire sur la théorie des déblais et des remblais. De l'Imprimerie Royale, 1781.  
Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, and Peter Stone. Curriculum learning for reinforcement learning domains: A framework and survey. Journal of Machine Learning Research (JMLR), 21(181):1-50, 2020.  
Radford M Neal. Annealed importance sampling. _Statistics and Computing_, 11(2):125-139, 2001.  
Gabriel Peyre, Marco Cuturi, et al. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355-607, 2019.  
Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In _Conference on Robot Learning (CoRL)_, 2019.  
Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In International Conference on Machine Learning (ICML), 2015.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning (ICML), 2015.  
David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International Conference on Machine Learning (ICML), 2014.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 550(7676):354, 2017.  
Richard S Sutton and Andrew G Barto. Introduction to Reinforcement Learning. MIT Press, 1998.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In International Conference on Intelligent Robots and Systems (IROS), 2012.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cirmrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261-272, 2020.  
Daphna Weinshall and Dan Amir. Theory of curriculum learning, with convex loss functions. Journal of Machine Learning Research (JMLR), 21(222):1-19, 2020.  
Andre Wibisono. Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem. In Conference on Learning Theory (COLT), 2018.  
Aaron Wilson, Alan Fern, Soumya Ray, and Prasad Tadepalli. Multi-task reinforcement learning: a hierarchical bayesian approach. In International Conference on Machine Learning (ICML), 2007.  
Jan Wöhlke, Felix Schmitt, and Herke van Hoof. A performance-based start state curriculum framework for reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pp. 1503-1511, 2020.  
Xiaoxia Wu, Ethan Dyer, and Behnam Neyshabur. When do curricula work? In International Conference on Learning Representations (ICLR), 2021.

![](images/b0049188f52cd68daf11a98c3391a82f4463b39d977e323d24fa94503ae0dabf.jpg)  
(a) Default

![](images/087e397a3ca9b072fb4a52ed76275d2ae36c2beb5ee1f940e5d62752c39b6fe5.jpg)

![](images/c3e0b7fb30b2a224a576cdbabea2b9a3ea0705e62a21e3c22ea83ca8337ab411.jpg)

![](images/f39b90a68938bb4d90800be4607e112b5f761efda862dfa950934e6348d33bcf.jpg)

![](images/3501b8102c8464f0dd2dfb5441a97ea86ca57fee30bc87239daad80ca4512da3.jpg)  
(e) WB-SPRL

![](images/f033a15b8a94a335c98b6a517167b6111d04548769ad1a40fbd36843a79b76db.jpg)  
(b) Random  
(c) G-SPRL  
(f) ALP-GMM

![](images/2f033217a387d434d8d3fd05c040cd248c9b5fc810317ea2e9b8e8ae8a293b2e.jpg)  
Figure 9: Final trajectories generated by the different investigated curricula in the point mass environment. The color encodes the context: Blue represents gates positioned at the left and red gates positioned at the right.  
(d) NP-SPRL  
(g) GOALGAN
