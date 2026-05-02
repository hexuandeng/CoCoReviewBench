# Q-PENSIEVE: BOOSTING SAMPLE EFFICIENCY OF MULTI-OBJECTIVE RL THROUGH MEMORY SHARING OF Q-SNAPSHOTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Many real-world continuous control problems are in the dilemma of weighing the pros and cons, multi-objective reinforcement learning (MORL) serves as a generic framework of learning control policies for different preferences over objectives. However, the existing MORL methods either rely on multiple passes of explicit search for finding the Pareto front and therefore are not sample-efficient, or utilizes a shared policy network for coarse knowledge sharing among policies. To boost the sample efficiency of MORL, we propose  $Q$ -Pensieve, a policy improvement scheme that stores a collection of  $Q$ -snapshots to jointly determine the policy update direction and thereby enables data sharing at the policy level. We show that  $Q$ -Pensieve can be naturally integrated with soft policy iteration with convergence guarantee. To substantiate this concept, we propose the technique of  $Q$  replay buffer, which stores the learned  $Q$ -networks from the past iterations, and arrive at a practical actor-critic implementation. Through extensive experiments and an ablation study, we demonstrate that with much fewer samples, the proposed algorithm can outperform the benchmark MORL methods on a variety of MORL benchmark tasks.

# 1 INTRODUCTION

Many real-world sequential decision-making problems involve the joint optimization of multiple objectives, while some of them may be in conflict. For example, in robot control, it is expected that the robot can run fast while consuming as little energy as possible; nevertheless, we inevitably need to use more energy to make the robot run fast, regardless of how energy-efficient the robot motion is. Moreover, various other real-world continuous control problems are also multi-objective tasks by nature, such as congestion control in communication networks (Ma et al., 2022) and diversified portfolios (Abdolmaleki et al., 2020). Moreover, the relative importance of these objectives could vary over time (Rojers and Whiteson, 2017). For example, the preference over energy and speed in robot locomotion could change with the energy budget; network service providers need to continuously switch service among various networking applications (e.g., on-demand video streaming versus real-time conferencing), each of which could have preferences over latency and throughput.

To address the above practical challenges, multi-objective reinforcement learning (MORL) serves as one classic and popular formulation for learning optimal control strategies from vector-valued reward signal and achieve favorable trade-off among the objectives. In the MORL framework, the goal is to learn a collection of policies, under which the attained return vectors recover as much of the Pareto front as possible. One popular approach to addressing MORL is to explicitly search for the Pareto front with an aim to maximize the hypervolume associated with the reward vectors, such as evolutionary search (Xu et al., 2020) and search by first-order stationarity (Kyriakis et al., 2022). While being effective, explicit search algorithms are known to be rather sample-inefficient as the data sharing among different passes of explicit search is rather limited. As a result, it is typically difficult to maintain a sufficiently diverse set of optimal policies for different preferences within a reasonable number of training samples. Another way to address MORL is to implicitly search for non-dominated policies through linear scalarization, i.e., convert the vector-valued reward signal to a single scalar with the help of a linear preference and thereafter apply a conventional single-objective RL algorithm for iteratively improving the policies (e.g., (Abels et al., 2019; Yang et al., 2019)).

To enable implicit search for diverse preferences simultaneously, a single network is typically used to express a whole collection of policies. As a result, some level of data sharing among policies of different preferences is done implicitly through the shared network parameters. However, such sharing is clearly not guaranteed to achieve policy improvement for all preferences. Therefore, there remains one critical open research question to be answered: How to boost the sample efficiency of MORL through better policy-level knowledge sharing?

To answer this question, we revisit MORL from the perspective of memory sharing among the policies learned across different training iterations and propose  $Q$ -Pensieve, where a "Pensieve", as illustrated in the novel Harry Potter, is a magical device used to store pieces of personal memories, which can later be shared with someone else. By drawing an analogy between the memory sharing among humans and the knowledge sharing among policies, we propose to construct a  $Q$ -Pensieve, which stores snapshots of the  $Q$ -functions of the policies learned in the past iterations. Upon improving the policy for a specific preference, we expect that these  $Q$ -snapshots could help jointly determine the policy update direction. In this way, we explicitly enforce knowledge sharing on the policy level and thereby enhance the sample use in learning optimal policies for various preferences. To substantiate this idea, we start by considering  $Q$ -Pensieve memory sharing in the tabular planning setting and integrate  $Q$ -Pensieve with the soft policy iteration for entropy-regularized MDPs. Inspired by (Yang et al., 2019), we leverage the envelope operation and propose the  $Q$ -Pensieve policy iteration for MORL, which we show would preserve the similar convergence guarantee as the standard single-objective soft policy iteration. Based on this result, we propose a practical implementation that consists of two major components: (i) We introduce the technique of  $Q$  replay buffer. Similar to the standard replay buffer of state transitions, a  $Q$  replay buffer is meant to achieve sample reuse and improve sample efficiency, but notably at the policy level. Through the use of  $Q$  replay buffer, we can directly obtain a large collection of  $Q$  functions, each of which corresponds to a policy in a prior training iteration, without any additional efforts or computation in forming the  $Q$ -Pensieve. (ii) We convert the  $Q$ -Pensieve policy iteration into an actor-critic off-policy MORL algorithm by adapting the soft actor critic to the multi-objective setting and using it as the base of our implementation.

The main contributions of this paper can be summarized as:

- We identify the critical sample inefficiency issue in MORL and address this issue by proposing  $Q$ -Pensieve, which is a policy improvement scheme for enhancing knowledge sharing on the policy level. We then present  $Q$ -Pensieve policy iteration and establish its convergence property.  
- We substantiate the concept of  $Q$ -Pensieve policy iteration by proposing the technique of  $Q$  replay buffer and arrive at a practical actor-critic type practical implementation.  
- We evaluate the proposed algorithm in various benchmark MORL environments, including Deep Sea Treasure and MuJoCo. Through extensive experiments and an ablation study, we demonstrate the proposed  $Q$ -Pensieve can indeed achieve significantly better empirical sample efficiency than the popular benchmark MORL algorithms, in terms of multiple common MORL performance metrics, including hypervolume and utility.

# 2 PRELIMINARIES

Multi-Objective Markov Decision Processes (MOMDPs). We consider the formulation of MOMDP defined by the tuple  $(\mathcal{S},\mathcal{A},\mathcal{P},r,\gamma ,\mathcal{D},\mathfrak{S}_{\lambda},\Lambda)$ , where  $\mathcal{S}$  denotes the state space,  $\mathcal{A}$  is the action space,  $\mathcal{P}:S\times \mathcal{A}\times \mathcal{S}\to [0,1]$  is the transition kernel of the environment,  $r:S\times \mathcal{A}\rightarrow [-r_{\mathrm{max}},r_{\mathrm{max}}]^d$  is the vector-valued reward function with  $d$  as the number of objectives,  $\gamma \in (0,1)$  is the discount factor,  $\mathcal{D}$  is the initial state distribution,  $\mathfrak{S}_{\lambda}:\mathbb{R}^{d}\to \mathbb{R}$  is the scalarization function (under some preference vector  $\lambda \in \mathbb{R}^d$ ), and  $\Lambda$  denotes the set of all preference vectors. In this paper, we focus on the linear reward scalarization setting, i.e.,  $\mathfrak{S}_{\lambda}(\boldsymbol {r}) = \boldsymbol{\lambda}^{\top}\boldsymbol {r}(s,a)$ , as commonly adopted in the MORL literature (Abels et al., 2019; Yang et al., 2019; Kyriakis et al., 2022). Without loss of generality, we let  $\Lambda$  be the unit simplex. If  $d = 1$ , an MOMDP would degenerate to a standard MDP, and we simply use  $r(s,a)$  to denote the scalar reward. At each time step  $t\in \mathbb{N}\cup \{0\}$ , the learner receives the observation  $s_t$ , takes an action  $a_{t}$ , and receives a reward vector  $r_t$ . We use  $\pi :S\to \Delta (\mathcal{A})$  to denote a stationary randomized policy, where  $\Delta (\mathcal{A})$  denotes the set of all probability distributions over the action space. Let  $\Pi$  be the set of all such policies.

Single-Objective Entropy-Regularized RL. In the standard framework of single-objective entropyregularized RL (Haarnoja et al., 2017; 2018; Geist et al., 2019), the goal is to learn an optimal policy for an entropy-regularized MDP, where an entropy regularization term is augmented to the original reward function. For a policy  $\pi \in \Pi$ , the regularized value functions  $V^{\pi}: S \to \mathbb{R}$  and  $Q^{\pi}: S \times \mathcal{A} \to \mathbb{R}$  can be characterized through the regularized Bellman equations as

$$
Q ^ {\pi} (s, a) = r (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a)} [ V ^ {\pi} (s) ], \tag {1}
$$

$$
V ^ {\pi} (s) = \mathbb {E} _ {a \sim \pi (\cdot | s)} [ Q ^ {\pi} (s, a) - \alpha \log \pi (a | s) ], \tag {2}
$$

where  $\alpha$  is a temperature parameter that specifies the relative importance of the entropy regularization term. In this setting, the goal is to learn an optimal policy  $\pi^{*}$  such that  $Q^{\pi^{*}}(s,a)\geq Q^{\pi}(s,a)$ , for all  $(s,a)$ , for all  $\pi \in \Pi$ . An optimal policy can be obtained through soft policy iteration, which alternates between soft policy evaluation and soft policy improvement: (i) Soft policy evaluation: For a policy  $\pi$ , the soft  $Q$ -function of  $\pi$  can be obtained by iteratively applying the corresponding soft Bellman backup operator  $\mathcal{T}^{\pi}$  defined as

$$
\mathcal {T} ^ {\pi} Q (s, a) = r (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a)} [ V (s ^ {\prime}) ], \tag {3}
$$

where  $V(s^{\prime}) = \mathbb{E}_{a^{\prime}\sim \pi (\cdot |s^{\prime})}[Q(s^{\prime},a^{\prime}) - \alpha \log (\pi (a^{\prime}|\bar{s}^{\prime}))]$ . (ii) Soft policy improvement: In each iteration  $k$ , the policy is updated towards an energy-based policy induced by the soft  $Q$ -function, i.e.,

$$
\pi_ {k + 1} = \arg \min  _ {\pi^ {\prime} \in \tilde {\Pi}} \mathrm {D} _ {\mathrm {K L}} \left(\pi^ {\prime} (\cdot | s) \| \frac {\exp \left(\frac {1}{\alpha} Q ^ {\pi_ {k}} (s , \cdot)\right)}{Z ^ {\pi_ {k}} (s)}\right), \tag {4}
$$

where  $\tilde{\Pi}$  is the set of parameterized policies of interest and  $Z^{\pi_k}$  is the normalization term.

Multi-Objective Entropy-Regularized RL. We extend the standard single-objective RL with entropy regularization to the multi-objective setting. For each policy  $\pi \in \Pi$ , we define the multi-objective regularized value functions via the following multi-objective version of entropy-regularized Bellman equations as follows:

$$
\boldsymbol {Q} ^ {\pi} (s, a) = \boldsymbol {r} (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a)} [ \boldsymbol {V} ^ {\pi} (s) ], \tag {5}
$$

$$
\boldsymbol {V} ^ {\pi} (s) = \mathbb {E} _ {a \sim \pi (\cdot | s)} [ \boldsymbol {Q} ^ {\pi} (s, a) - \alpha \log \pi (a | s) \mathbf {1} _ {d} ], \tag {6}
$$

where  $\mathbf{1}_d$  denotes a  $d$ -dimensional vector of all ones.

In this paper, our goal is to learn a preference-dependent policy  $\pi (\cdot |\cdot ;\pmb {\lambda})$  such that for any preference  $\pmb {\lambda}\in \Lambda ,\pmb{\lambda}^{\top}\pmb{Q}^{\pi (\cdot |\cdot ;\pmb {\lambda})}(s,a;\pmb {\lambda})\geq \pmb{\lambda}^{\top}\pmb{Q}^{\pi '}(s,a)$  , for all  $(s,a)$  , for all  $\pi^\prime \in \Pi$  . For ease of notation, we let  $V^{\pi (\cdot |\cdot ;\pmb {\lambda})}(s;\pmb {\lambda})\equiv V^{\pi}(s;\pmb {\lambda})$  and  $Q^{\pi (\cdot |\cdot ;\pmb {\lambda})}(s,a;\pmb {\lambda})\equiv Q^{\pi}(s,a;\pmb {\lambda})$  in the sequel.

# 3 ALGORITHMS

In this section, we propose our  $Q$ -Pensieve learning algorithm for boosting the sample efficiency of multi-objective RL. We first describe the idea of  $Q$ -Pensieve in the tabular planning setting by introducing  $Q$ -Pensieve soft policy iteration. We then extend the idea to develop a practical deep reinforcement learning algorithm.

# 3.1 NAIVE MULTI-OBJECTIVE SOFT POLICY ITERATION

To solve MORL in the entropy-regularized setting, one straightforward approach is to leverage the single-objective soft policy improvement with the help of linear scalarization. That is, in each iteration  $k$ , the policy can be updated by

$$
\pi_ {k + 1} (\cdot , \cdot ; \boldsymbol {\lambda}) = \arg \min  _ {\pi^ {\prime} \in \bar {\Pi}} \mathrm {D} _ {\mathrm {K L}} \left( \right.\pi^ {\prime} (\cdot | s) \left\| \right. \frac {\exp \left(\frac {1}{\alpha} \boldsymbol {\lambda} ^ {\top} \boldsymbol {Q} ^ {\pi_ {k}} (s , \cdot ; \boldsymbol {\lambda})\right)}{Z _ {\boldsymbol {\lambda}} ^ {\pi_ {k}} (s)}\left. \right). \tag {7}
$$

While (7) serves as a reasonable approach, designing a learning algorithm based on the update scheme in (7) could suffer from sample inefficiency due to the lack of policy-level knowledge sharing: In (7), the policy for each preference  $\lambda$  is updated completely separately based solely on the  $Q$ -function under  $\lambda$ . Moreover, as the update (7) relies on an accurate estimate of the  $Q$ -function, the critic

learning for the policy of each individual preference would typically require at least a moderate number of samples. These issues could be particularly critical for a large preference set in practice. While the use of a conditioned policy network (e.g., (Abels et al., 2019)), a commonly-used network architecture in the MORL literature, could somewhat mitigate this issue, it remains unclear whether the knowledge sharing induced by the conditioned network can indeed achieve policy improvement across various preferences. As a result, a systematic approach is needed for boosting the sample efficiency in MORL.

# 3.2  $Q$ -PENSIEVE SOFT POLICY ITERATION

To boost the sample efficiency of MORL, we propose to enhance the policy-level knowledge sharing by constructing a  $Q$ -Pensieve for memory sharing across iterations. Specifically, a  $Q$ -Pensieve is a collection of  $Q$ -snapshots obtained from the past iterations, and it is formed to boost the policy improvement update with respect to the  $Q$ -function in the current iteration as these  $Q$ -snapshots could offer potentially better policy improvement directions under linear scalarization. Moreover, one major computational benefit of  $Q$ -Pensieve is that these  $Q$ -snapshots are obtained without the need for any updates or additional samples from the environment (and hence are for free) as they already exist during training. We substantiate this idea by first introducing the  $Q$ -Pensieve soft policy iteration in the tabular setting (i.e.,  $|S|$  and  $|\mathcal{A}|$  are finite) as follows:

$Q$ -Pensieve Policy Improvement. In the policy improvement step of the  $k$ -th iteration, for each specific  $\lambda$ , we update the policy as

$$
\pi_ {k + 1} (\cdot |; \boldsymbol {\lambda}) = \arg \min  _ {\pi^ {\prime} \in \Pi} \mathrm {D} _ {\mathrm {K L}} \left( \right.\pi^ {\prime} (\cdot | s; \boldsymbol {\lambda}) \left\| \right. \frac {\exp \left(\sup  _ {\boldsymbol {\lambda} ^ {\prime} \in W _ {k} (\boldsymbol {\lambda}) , \boldsymbol {Q} ^ {\prime} \in \mathcal {Q} _ {k}} \frac {1}{\alpha} \boldsymbol {\lambda} ^ {\top} \boldsymbol {Q} ^ {\prime} (s , \cdot ; \boldsymbol {\lambda} ^ {\prime})\right)}{Z _ {\mathcal {Q} _ {k}} (s)}\left. \right), \tag {8}
$$

where  $Z_{\mathcal{Q}_k}$  is again the normalization term,  $W_{k}(\lambda)\subset \Lambda$  is a set of preference vectors, and  $\mathcal{Q}_k$  is a set of  $\pmb{Q}$ -snapshots. The two sets  $W_{k}(\lambda)$  and  $\mathcal{Q}_k$  are to be selected as follows:

- For  $W_{k}(\lambda)$ , the only requirement is that  $\lambda \in W_{k}(\lambda)$ , for all  $k$ . The preference sets can be different in different iterations.  
- Similarly, for  $\mathcal{Q}_k$ , the only requirement is that  $Q^{\pi_k} \in \mathcal{Q}_k$ , for all  $k$ . The set of  $\mathcal{Q}$ -snapshots can also be different in different iterations. Hence, the choice of  $\mathcal{Q}_k$  is rather flexible.

When choosing  $W_{k}(\lambda) = \{\lambda \}$  and  $\mathcal{Q}_k = \{\pmb {Q}^{\pi_k}\}$ , one would recover the update in (7).

Policy Evaluation. In the policy evaluation step, we evaluate the policy that corresponds to each preference  $\lambda$  by iteratively applying the multi-objective softmax Bellman backup operator  $\mathcal{T}_{\mathrm{MO}}^{\pi}$  as

$$
\left(\mathcal {T} _ {\mathrm {M O}} ^ {\pi} \boldsymbol {Q}\right) (s, a; \boldsymbol {\lambda}) = \boldsymbol {r} (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a), a ^ {\prime} \sim \pi (\cdot | s ^ {\prime}; \boldsymbol {\lambda})} [ \boldsymbol {Q} \left(s ^ {\prime}, a ^ {\prime}; \boldsymbol {\lambda}\right) - \alpha \log \pi \left(a ^ {\prime} \mid s ^ {\prime}; \boldsymbol {\lambda}\right) \mathbf {1} _ {d} ]. \tag {9}
$$

Remark 1 The  $Q$ -Pensieve update in (8) is inspired by the envelope Q-learning (EQL) technique (Yang et al., 2019), where in each iteration  $k$ , the Q-learning update takes into account the envelope formed by the  $Q$ -functions of the current policy  $\pi_{k}$  for different preferences. The fundamental difference between  $Q$ -Pensieve and EQL is that  $Q$ -Pensieve further achieves memory sharing across training iterations through the use of  $Q$ -snapshots from the past iterations, and EQL focuses mainly on the use of the  $Q$ -function of the current iteration.

Convergence of  $Q$ -Pensieve Soft Policy Iteration. Another nice feature of the  $Q$ -Pensieve policy improvement step is that it preserves the similar convergence result as the standard single-objective soft policy iteration, as stated below. The proof of Theorem 3.1 is provided in Appendix A.

Theorem 3.1 Under the  $Q$ -Pensieve soft policy iteration given by (8) and (9), the sequence of preference-dependent policies  $\{\pi_k\}$  converges to a policy  $\pi^*$  such that  $\lambda^\top Q^{\pi^*}(s,a;\lambda) \geq \lambda^\top Q^\pi(s,a)$  for all  $\pi \in \Pi$ , for all  $(s,a) \in S \times \mathcal{A}$  and for all  $\lambda \in \Lambda$ .

# 3.3 PRACTICAL IMPLEMENTATION OF  $Q$ -PENSIEVE

In this section, we present the implementation of proposed  $Q$ -Pensieve algorithm for learning policies with function approximation for the general state and action spaces.

$Q$  Replay Buffer. Based on (8), we know that the policy update of  $Q$ -Pensieve would involve both the current  $Q$ -function and the  $Q$ -snAPSHOT from the past iterations. To implement this, we introduce  $Q$  replay buffer, which could store multiple  $Q$ -networks in a predetermined manner (e.g., first-in first-out). Notably, unlike the conventional experience replay buffer (Mnih et al., 2013) of state transitions,  $Q$  replay buffer stores the learned  $Q$ -networks in past iterations as candidates for forming the  $Q$ -Pensieve. On the other hand, while each  $Q$ -network would require a moderate amount of memory usage, we found that in practice a rather small  $Q$  replay buffer is already effective enough for boosting the sample efficiency. We further illustrate this observation through the experimental results in Section 4.

Next, we convert the  $Q$ -Pensieve soft policy iteration into an actor-critic off-policy MORL algorithm. Specifically, we adapt the idea of soft actor critic to  $Q$ -Pensieve by minimizing the residual of the multi-objective soft  $Q$ -function: Let  $\theta$  and  $\phi$  be the parameters of the policy network and the critic network, respectively. Then, the critic network is updated by minimizing the following loss

$$
\mathcal {L} _ {\boldsymbol {Q}} (\phi ; \boldsymbol {\lambda}) = \mathbb {E} _ {(s, a) \sim \mu} \left[ \boldsymbol {\lambda} ^ {\top} \left(\boldsymbol {Q} _ {\phi} (s, a; \boldsymbol {\lambda}) - \left(\boldsymbol {r} (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim \mathcal {P} (\cdot | s, a)} \left[ \boldsymbol {V} _ {\bar {\phi}} \left(s ^ {\prime}\right) \right]\right)\right) ^ {2} \right], \tag {10}
$$

where  $\bar{\phi}$  is the parameter of the target network and  $\mu$  is the sampling distribution of the state-action pairs (e.g., a distribution induced by a replay buffer of state transitions). On the other hand, based on (8), the policy network is updated by minimizing the following objective

$$
\mathcal {L} _ {\pi} (\theta ; \boldsymbol {\lambda}) = \mathbb {E} _ {s \sim \mu} \left[ \mathbb {E} _ {a \sim \pi_ {\theta}} \left[ \sup  _ {\boldsymbol {\lambda} ^ {\prime} \in W (\boldsymbol {\lambda}), \boldsymbol {Q} ^ {\prime} \in \mathcal {Q}} \left\{\alpha \log \left(\pi_ {\theta} (a \mid s; \boldsymbol {\lambda})\right) - \boldsymbol {\lambda} ^ {\intercal} \boldsymbol {Q} ^ {\prime} (s, a; \boldsymbol {\lambda} ^ {\prime}) \right\} \right] \right]. \tag {11}
$$

The overall architecture of  $Q$ -Pensieve is provided in Figure 1. The pseudo code of the  $Q$ -Pensieve algorithm is described in Algorithm 1 in Appendix.

![](images/5b7b0642ead4b8573babf6d8f7dcfafae0b9fc7271331b4130759fa70a2c7f99.jpg)  
Figure 1: The architecture of  $Q$ -Pensieve.

# 4 EXPERIMENTS

In this section, we demonstrate the effectiveness of  $Q$ -Pensieve on various benchmark RL tasks and discuss how  $Q$ -Pensieve boosts the sample efficiency through an extensive ablation study.

# 4.1 EXPERIMENTAL CONFIGURATION

Popular Benchmark Methods. We compare the proposed algorithm against various popular benchmark methods, including the Conditioned Network with Diverse Experience Replay (CN-DER) in (Abels et al., 2019), the Prediction-Guided Multi-Objective RL (PGMORL) in (Xu et al., 2020), the Pareto Following Algorithm (PFA) in (Parisi et al., 2014), and SAC (Haarnoja et al., 2018). For

CN-DER, as the original CN-DER is built on deep Q-networks (DQN) for discrete actions, we modify the source code of Abels et al. (2019) for continuous control by implementing CN-DER on top of DDPG. Moreover, we follow the same DER technique, which uses a diverse replay buffer and gives priority according to how much the samples increase the overall diversity of the buffer. For PGMORL and PFA, we use the open-source implementation of (Xu et al., 2020) for the experiments. As these explicit search methods typically require more samples before reaching a comparable performance level, we evaluate the performance PGMORL and PFA under both 1 times and  $\beta$  times  $(\beta > 1)$  of the number of samples used by  $Q$ -Pensieve to demonstrate the sample efficiency of  $Q$ -Pensieve. For SAC, as the MORL problem reduces a single-objective one under a fixed preference, we train multiple models using single-objective SAC (one model for each fixed preference) as a performance reference for other MORL methods.

Performance Metrics. In the evaluation, we consider the following three commonly-used performance metrics for MORL:

- HyperVolume (HV): Let  $\mathcal{R}$  be a set of return vectors attained and  $\boldsymbol{r}_0 \in \mathbb{R}^d$  be a reference point. Then, we define the HyperVolume as HV :=  $\int_{H(\mathcal{R})} \mathbb{I}\{z \in H(\mathcal{R})\} dz$ , where  $H(\mathcal{R}) \coloneqq \{z \in \mathbb{R}^d : \exists r \in \mathcal{R}, r_0 \prec z \prec r\}$  and  $\mathbb{I}$  is the indicator function.  
- Utility (UT): To further evaluate the performance under linear scalarization, we define the utility metric as UT :=  $\mathbb{E}_{\boldsymbol{\lambda}}\left[\sum_{t=0}^{T}\boldsymbol{\lambda}^{\top}\boldsymbol{r}_{t}\right]$ , where the preference  $\boldsymbol{\lambda}$  is sampled uniformly from  $\Lambda$ .  
- Episodic Dominance (ED): To compare the performance of a pair of algorithms, we define Episodic Dominance as  $\mathrm{ED}_{1,2} := \mathbb{E}_{\lambda}[\mathbb{I}\{\sum_{t=0}^{T_1} \boldsymbol{\lambda}^\top \boldsymbol{r}_t^1 > \sum_{t=0}^{T_2} \boldsymbol{\lambda}^\top \boldsymbol{r}_t^2\}]$ , where  $\boldsymbol{r}_t^1, \boldsymbol{r}_t^2$  are the return vectors, and  $T_1, T_2$  are the episode lengths of algorithm 1 and 2, respectively. ED serves as a useful metric for pairwise comparison in those problems where the return vectors under different preferences can differ by a lot in magnitude (in this case, HV and UT could be dominated by the return vectors of a few preferences).

Evaluation Domains. We evaluate the algorithms in the following domains: (i) Continuous Deep Sea Treasure (DST): a two-objective continuous control task modified from the original DST environment. (ii) Multi-Objective Continuous LunarLander: a four-objective task modified from the classic control task in the OpenAI gym. (iii) Multi-Objective MuJoCo: modified benchmark locomotion tasks with either two or three objectives.

Configuration of  $Q$ -Pensieve. For  $Q$ -Pensieve, at each policy update, we set the size of the preference set  $W_{k}(\lambda)$  to be 5 (including  $\lambda$  and another four preferences drawn randomly) and set the size of the  $Q$  replay buffer to be 4, unless stated otherwise.

# 4.2 EXPERIMENTAL RESULTS

Does  $Q$ -Pensieve achieve better sample efficiency than the MORL benchmark methods? Table 1 shows the performance of  $Q$ -Pensieve and the benchmark methods in terms of the three metrics. For each algorithm, we report the mean and the standard deviation over five random seeds. We can observe that  $Q$ -Pensieve consistently enjoys higher HV, UT, and ED in almost all the domains. More importantly,  $Q$ -Pensieve indeed exhibits superior sample efficiency as it still outperforms the explicit search methods (i.e., PFA and PGMORL) even if these methods are given 10 times of the number of samples used by  $Q$ -Pensieve. Moreover, we can observe that the explicit search methods (i.e., PFA and PGMORL) often have larger HV than the implicit search method (such as CN-DER), while implicit search methods tend to have larger UT. This manifests the design principles and the characteristics of the two families of approaches, where explicit search is designed mainly for achieving large HV and implicit search typically aims for larger scalarized return.

How much improvement in sample efficiency can  $Q$ -Pensieve achieve compared to training multiple single-objective SAC models separately? To answer this question, we conduct experiments on 2-objective MuJoCo tasks and consider a whole range of 19 preference vectors ([0.05, 0.95], [0.1, 0.9], [0.15, 0.85], ..., [0.95, 0.05]). We train 19 models by using single-objective SAC, one model for each individual preference. Each model is trained for 1.5M steps (and hence the total number of steps under SAC is 28.5M steps). By contrast,  $Q$ -Pensieve only uses 1.5M steps in total in learning policies for all the preferences. Figure 2 shows the return vectors attained by  $Q$ -Pensieve and the collection of 19 SAC models.  $Q$ -Pensieve can achieve comparable or better

Table 1: Comparison of  $Q$ -Pensieve and other benchmark algorithms in terms of the three metrics across eight domains. We report the mean and standard deviation over five random seeds. The ED is calculated through comparing each algorithm to a multi-objective version of SAC (equivalent to  $Q$ -Pensieve with the size of the preference set equal to 1 and without  $Q$  replay buffer). We set  $\beta = 10$  for HalfCheetah, Ant, Ant3d, and Hopper3d, set  $\beta = 5$  for LunarLander, and set  $\beta = 3$  for DST, Hopper, and Walker2d.  

<table><tr><td>Environments</td><td>Metrics</td><td>PFA(1.5M steps)</td><td>PFA(1.5×βM steps)</td><td>PGMORL(1.5M steps)</td><td>PGMORL(1.5×βM steps)</td><td>CN-DER(1.5M steps)</td><td>Q-Pensieve(1.5M steps)</td></tr><tr><td rowspan="3">DST</td><td>HV(×102)</td><td>7.43±3.68</td><td>8.67±1.49</td><td>8.10±1.57</td><td>8.13±1.61</td><td>5.36±4.71</td><td>10.21±1.40</td></tr><tr><td>UT</td><td>-9.27±6.03</td><td>-6.86±6.06</td><td>4.90±0.44</td><td>5.02±0.35</td><td>-5.10±15.73</td><td>7.31±0.91</td></tr><tr><td>ED</td><td>0.13±0.11</td><td>0.10±0.08</td><td>0.25±0.18</td><td>0.28±0.18</td><td>0.21±0.17</td><td>0.54±0.11</td></tr><tr><td rowspan="3">LunarLander</td><td>HV(×108)</td><td>-</td><td>-</td><td>0.32±0.11</td><td>0.38±0.11</td><td>1.50±0.60</td><td>2.10±0.10</td></tr><tr><td>UT(×10)</td><td>-</td><td>-</td><td>-0.26±0.27</td><td>1.10±0.50</td><td>3.60±2.90</td><td>5.10±0.30</td></tr><tr><td>ED</td><td>-</td><td>-</td><td>0.02±0.01</td><td>0.04±0.04</td><td>0.21±0.12</td><td>0.49±0.05</td></tr><tr><td rowspan="3">HalfCheetah</td><td>HV(×107)</td><td>0.73±0.19</td><td>1.31±0.26</td><td>0.53±0.17</td><td>0.28±0.29</td><td>2.08±0.54</td><td>3.82±0.27</td></tr><tr><td>UT(×103)</td><td>0.31±0.20</td><td>1.02±0.40</td><td>-0.28±0.94</td><td>0.09±0.17</td><td>5.09±3.57</td><td>5.61±0.31</td></tr><tr><td>ED</td><td>0.08±0.10</td><td>0.10±0.06</td><td>0.01±0.00</td><td>0.11±0.05</td><td>0.02±0.01</td><td>0.54±0.08</td></tr><tr><td rowspan="3">Hopper</td><td>HV(×106)</td><td>0.49±0.46</td><td>1.01±0.62</td><td>0.63±0.48</td><td>1.31±0.48</td><td>0.56±0.16</td><td>1.33±0.20</td></tr><tr><td>UT(×102)</td><td>2.89±1.93</td><td>3.50±1.85</td><td>1.94±2.46</td><td>3.70±1.78</td><td>1.42±1.00</td><td>4.08±1.10</td></tr><tr><td>ED</td><td>0.31±0.17</td><td>0.41±0.10</td><td>0.31±0.25</td><td>0.31±0.11</td><td>0.04±0.03</td><td>0.43±0.09</td></tr><tr><td rowspan="3">Hopper3d</td><td>HV(×109)</td><td>-</td><td>-</td><td>0.29±0.37</td><td>0.91±1.39</td><td>3.70±0.81</td><td>9.56±0.60</td></tr><tr><td>UT(×103)</td><td>-</td><td>-</td><td>0.19±0.16</td><td>0.31±0.26</td><td>0.72±0.16</td><td>1.39±0.15</td></tr><tr><td>ED</td><td>-</td><td>-</td><td>0.02±0.03</td><td>0.03±0.03</td><td>0.07±0.03</td><td>0.55±0.08</td></tr><tr><td rowspan="3">Ant</td><td>HV(×106)</td><td>0.17±0.05</td><td>0.77±0.53</td><td>0.14±0.03</td><td>0.13±0.04</td><td>5.03±3.60</td><td>10.01±1.86</td></tr><tr><td>UT(×102)</td><td>-0.06±0.01</td><td>0.14±0.14</td><td>-0.21±0.15</td><td>-0.18±0.38</td><td>3.68±2.34</td><td>14.04±3.03</td></tr><tr><td>ED</td><td>0.22±0.03</td><td>0.22±0.02</td><td>0.21±0.02</td><td>0.21±0.03</td><td>0.21±0.08</td><td>0.60±0.07</td></tr><tr><td rowspan="3">Ant3d</td><td>HV(×108)</td><td>-</td><td>-</td><td>0.41±0.48</td><td>0.68±0.62</td><td>13.00±4.11</td><td>21.87±1.07</td></tr><tr><td>UT(×103)</td><td>-</td><td>-</td><td>0.18±0.05</td><td>0.25±0.05</td><td>0.49±0.23</td><td>1.14±0.22</td></tr><tr><td>ED</td><td>-</td><td>-</td><td>0.02±0.02</td><td>0.03±0.03</td><td>0.28±0.14</td><td>0.56±0.07</td></tr><tr><td rowspan="3">Walker2d</td><td>HV(×106)</td><td>0.52±0.20</td><td>1.05±0.44</td><td>0.83±0.42</td><td>1.28±0.66</td><td>0.42±0.09</td><td>1.12±0.36</td></tr><tr><td>UT(×102)</td><td>0.23±0.13</td><td>0.95±0.55</td><td>0.38±0.24</td><td>1.20±0.67</td><td>3.17±0.53</td><td>6.37±1.42</td></tr><tr><td>ED</td><td>0.32±0.06</td><td>0.37±0.09</td><td>0.30±0.10</td><td>0.34±0.12</td><td>0.21±0.11</td><td>0.48±0.10</td></tr></table>

returns than the collection of SAC models with only  $1/19$  of the samples. This further demonstrates the sample efficiency of  $\mathbf{Q}$ -Pensieve.

![](images/b8e3318be9d9c9a0347e679b0e23f1a6f77b736899c86d1c60fff607a357e75e.jpg)  
(a) Hopper

![](images/ed5e453af0ea75c74b296a6149c226dbb084d93e74779f91eb0ed71efb8d2d06.jpg)  
(b) HalfCheetah

![](images/82a9b100ef0d1d4a62a3fa0b9dcaa8c81752f7a4876692a895d546a7fa7e1547.jpg)  
Figure 2: Return vectors attained by  $Q$ -Pensieve and the collection of single-objective SAC models under 19 preferences.  
(c) DST

Why can  $Q$ -Pensieve outperform single-objective SAC in some cases? From Figures 2(a) and (c), we see that  $Q$ -Pensieve can attain some return vectors that are strictly better than those of the single-objective SAC models. The reasons behind this phenomenon are minaly two-fold: (i) Under single-objective SAC, despite that we train one model for each individual preference, it could still occur that single-objective SAC gets stuck at a sub-optimal policy under some preferences. (ii) By contrast,  $Q$ -Pensieve has a better chance of escaping from these sub-optimal policies with the help of the  $Q$ -snapshots in the  $Q$  replay buffer.

To verify the above argument, we design a hybrid SAC algorithm as follows: (a) For the first  $10^{5}$  time steps, this algorithm simply follows the single-objective SAC. (b) At time step  $10^{5}$ , it switches

to the update rule of  $Q$ -Pensieve based on the  $Q$ -snapshots stored in the  $Q$  replay buffer of another model trained under  $Q$ -Pensieve algorithm in parallel. Figure 3 shows the performance of this hybrid algorithm in DST and HalfCheetah. Clearly, the  $Q$ -Pensieve update could help the SAC model escape from the sub-optimal policies, under various preferences.

![](images/06f18d498c6c5c97d89355a3895b1ed609caf3fa60b4af3cdcf20fdca233fb79.jpg)  
(a) DST,  $\lambda = [0.9, 0.1]$

![](images/2faf2540eef678b7387fb6287539368889fa1b7f62d38a298b9840b5bd800f1e.jpg)  
(b) DST,  $\lambda = [0.8, 0.2]$

![](images/198dc7b8639d611ca899fcd072fda83d394968e91d3dff061d2415748c0e0dc0.jpg)  
Figure 3: Comparison of standard single-objective SAC and the hybrid SAC assisted by another  $Q$ -Pensieve model trained in parallel.

![](images/97b22e90551f246bc92ca1122d418cbc06a8250d4c63da67f3b9513cae4ff328.jpg)  
(c) HalfCheetah,  $\lambda = [0.5, 0.5]$  
(d) HalfCheetah,  $\lambda = [0.6, 0.4]$

An ablation study on  $Q$  reply buffer. To verify the effectiveness of the technique of  $Q$  replay buffer, we compare the performance of  $Q$ -Pensieve with buffer size equal to 4 and that without using  $Q$  replay buffer (termed "Vanilla" in Figures 4 and 5). Figure 4 and 5 show the attained return vectors and HV of both methods. We can see that  $Q$  replay buffer indeed leads to a better policy improvement behavior, in terms of both HV and the scalarized returns. However, these figures may sometimes oscillate a lot in the end period. It is because our algorithm finds solutions from another  $Q$ -vector, and their inner product of  $Q$  and preference may be quite close. We can check the points are in the same contour.

# 5 RELATED WORK

The multi-objective RL problems have been extensively studied from two major perspectives:

Explicit search. A plethora of prior works on MORL updates a policy or a set of policies by explicitly searching for the Pareto front of the reward space. To learn policies under time-varying preferences, (Natarajan and Tadepalli, 2005) presented to store a set of policies, which are to be used in searching for a proper policy for a new preference without learning from scratch. (Lizotte et al., 2012) leveraged linear value function approximation to search for optimal policies. (Van Moffaert and Nowé, 2014) proposed Pareto Q-learning, which stores the immediate rewards and the non-dominated future return vectors separately and leverage the Pareto dominance for selecting the actions in Q-learning. (Parisi et al., 2014) presented a policy gradient approach to search for non-dominated policies. (Mossalam et al., 2016) solves MORL via scalarized Q-learning along with the concept of prioritizing the

![](images/cd9b7b82f9ebb5644442678a60a13023867951a7c1720d08e9e6d8e47f26d071.jpg)  
(a) Hopper2d

![](images/6e97ecdec75f6e322b987bea740ebbbd8639ac1106d9ae853890b77e1e74079d.jpg)  
Figure 4: Return vectors attained under preference  $\lambda = [0.5, 0.5]$  at different training stages.(We also plot return vectors under others preference in Figure 7 and Figure 8 in Appendix.) A number  $x$  on the red or blue marker indicates that the model is obtained at  $100 \cdot x$  thousand steps.  
(b) Ant2d

![](images/bb321e17a70ed91c54c3a2c29690aeeb82ecd9dd4b9c45037594784e3ae6ddf8.jpg)  
(c) Walker2d

![](images/6ba82346e6251820ffa3ad9e5c401c5171a25d07938eec7325e88a17254250c8.jpg)  
(a) Ant2d

![](images/4dfc3e744a8a636e8790f9c16ea4f65cbbb8bc78a20ac363ab0d758994cd0e2b.jpg)  
Figure 5: A comparison in HV between  $Q$ -Pensieve with buffer size equal to 4 and that without using  $Q$  replay buffer at different training stages.  
(b) Walker2d

corner weights for selecting the preference of the scalarized problem. (Xu et al., 2020) proposed an evolutionary approach to search for the Pareto set of policies, with the help of a prediction model for determining the search direction. (Kyriakis et al., 2022) presented a policy gradient method by approximating the Pareto front via a first-order necessary condition. However, the above explicit search algorithms are known to be rather sample-inefficient as the knowledge sharing among different passes of search is limited.

Implicit search. Another class of algorithms are designed to improve policies for multiple preferences through implicit search. For example, (Abels et al., 2019) presents Conditioned Network, which extends the standard single-objective DQN to learning preference-dependent multi-objective  $Q$ -functions. To achieve scale-invariant MORL, (Abdolmaleki et al., 2020) proposed to first learn the  $Q$ -functions for different objectives and encode the preference through constraints. While there is implicit Recently, (Yang et al., 2019) proposes envelope Q-learning to encourage knowledge sharing among the  $Q$  functions of different the current multi-objective  $Q$ -values that any policy can benefit from other preferences' experiences, that make training more efficiently, and (Zhou et al., 2020) proposed model-based envelope value iteration base on envelope  $Q$ -learning, it provides an efficient way to get optimal multi-objective  $Q$  function. Despite that our method is inspired by (Yang et al., 2019), the main difference between our work and theirs is that we boost the sample efficiency of MORL via explicit memory sharing among policies learned during training.

# 6 CONCLUSION

This paper proposes  $Q$ -Pensieve, which significantly enhances the policy-level data sharing through in order to boost the sample efficiency of MORL problems. We substantiate the idea by presenting  $Q$ -Pensieve soft policy iteration in the tabular setting and show that it preserves the global convergence property. Then, to implement the  $Q$ -Pensieve policy improvement step, we introduce the  $Q$  replay buffer technique, which offers a simple yet effective way to maintain  $Q$ -snapshot. Our experiments demonstrate that  $Q$ -Pensieve is a promising approach in that it can outperform the state-of-the-art MORL methods with much fewer samples in a variety of MORL benchmark tasks.

# REFERENCES

Yiqing Ma, Han Tian, Xudong Liao, Junxue Zhang, Weiyan Wang, Kai Chen, and Xin Jin. Multi-objective congestion control. In Proceedings of the Seventeenth European Conference on Computer Systems, pages 218-235, 2022.  
Abbas Abdelmaleki, Sandy Huang, Leonard Hasenclever, Michael Neunert, Francis Song, Martina Zambelli, Murilo Martins, Nicolas Heess, Raia Hadsell, and Martin Riedmiller. A distributional view on multi-objective policy optimization. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 11-22. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/abdolmaleki20a.html.

Diederik M Roijers and Shimon Whiteson. Multi-objective decision making. Synthesis Lectures on Artificial Intelligence and Machine Learning, 11(1):1-129, 2017.  
Jie Xu, Yunsheng Tian, Pingchuan Ma, Daniela Rus, Shinjiro Sueda, and Wojciech Matusik. Prediction-guided multi-objective reinforcement learning for continuous robot control. In Hal Daume III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 10607-10616. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/xu20h.html.  
Panagiotis Kyriakis, Jyotirmoy Deshmukh, and Paul Bogdan. Pareto policy adaptation. In International Conference on Learning Representations, 2022.  
Axel Abels, Diederik Roijers, Tom Lenaerts, Ann Nowé, and Denis Steckelmacher. Dynamic weights in multi-objective deep reinforcement learning. In International Conference on Machine Learning, pages 11-20. PMLR, 2019.  
Runzhe Yang, Xingyuan Sun, and Karthik Narasimhan. A generalized algorithm for multi-objective reinforcement learning and policy adaptation. Advances in Neural Information Processing Systems, 32, 2019.  
Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In International Conference on Machine Learning, pages 1352-1361, 2017.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861-1870. PMLR, 2018.  
Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International Conference on Machine Learning, pages 2160-2169, 2019.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv:1312.5602, 2013.  
Simone Parisi, Matteo Pirotta, Nicola Smacchia, Luca Bascetta, and Marcello Restelli. Policy gradient approaches for multi-objective sequential decision making. In International Joint Conference on Neural Networks (IJCNN), pages 2323-2330, 2014.  
Sriram Natarajan and Prasad Tadepalli. Dynamic preferences in multi-criteria reinforcement learning. In International Conference on Machine Learning, pages 601-608, 2005.  
Daniel J Lizotte, Michael Bowling, and Susan A Murphy. Linear fitted-q iteration with multiple reward functions. Journal of Machine Learning Research, 13(1):3253-3295, 2012.  
Kristof Van Moffaert and Ann Nowé. Multi-objective reinforcement learning using sets of pareto dominating policies. Journal of Machine Learning Research, 15(1):3483-3512, 2014.  
Hossam Mossalam, Yannis M. Assael, Diederik M. Roijers, and Shimon Whiteson. Multi-objective deep reinforcement learning, 2016. URL https://arxiv.org/abs/1610.02707.  
Dongruo Zhou, Jiahao Chen, and Quanquan Gu. Provable multi-objective reinforcement learning with generative models. arXiv preprint arXiv:2011.10134, 2020.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.