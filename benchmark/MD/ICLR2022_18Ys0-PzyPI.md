# ONLINE AD HOC TEAMWORK UNDER PARTIAL OBSERVABILITY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Autonomous agents often need to work together as a team to accomplish complex cooperative tasks. Due to privacy and other realistic constraints, agents might need to collaborate with previously unknown teammates on the fly. This problem is known as ad hoc teamwork, which remains a core research challenge. Prior works usually rely heavily on strong assumptions like full observability, fixed and predefined teammates' types. This paper relaxes these assumptions with a novel reinforcement learning framework called ODITS, which allows the autonomous agent to adapt to arbitrary teammates in an online fashion. Instead of limiting teammates into a finite set of predefined types, ODITS automatically learns latent variables of teammates' behaviors to infer how to cooperate with new teammates effectively. To overcome partial observability, we introduce an information-based regularizer to derive proxy representations of the learned variables from local observations. Extensive experimental results show that ODITS significantly outperforms various baselines in widely used ad hoc teamwork tasks.

# 1 INTRODUCTION

Recently, autonomous agents including robotics and software agents are being widely deployed in different environments. In many tasks, they are progressively required to cooperate with other unknown teammates on the fly. For example, in case of search and rescue tasks in a disaster, due to privacy or lack of time, deployed robots need to interact with robots from other companies or laboratories, whose coordination protocols might not be explicitly provided in advance (Barrett and Stone, 2015). Besides, in the domain of game AI (Yannakakis, 2012), virtual agents are required to assist different agents controlled by human players. To effectively complete these tasks, autonomous agents must show high adaptation ability to collaborate with intrinsically diverse and unknown teammates. This problem is known in the literature as ad hoc teamwork (Stone et al., 2010).

Existing approaches on ad hoc teamwork usually assume that all teammates' behaviors are categorized into several predefined and fixed types, which corresponds to different coordination strategies (Barrett and Stone, 2015; Durugkar et al., 2020; Mirsky et al., 2020). Then, by reasoning over the type of interacting teammates, the agent switches its behavior to the corresponding policy. If the types are correctly recognized and the strategies are effective, the agent would accomplish the given cooperation task well. However, defining sufficiently descriptive types of teammates requires prior domain knowledge, especially in uncertain and complex environments. For example, in human-AI collaboration in Hanabi (Bard et al., 2020), there are often a wide variety of cooperative behaviors showed by human players. It is challenging for predefined types to cover all possible human players' behaviors. Further, teammates' strategies might be rapidly evolving throughout the entire teamwork. If the agent assumes that teammates' behavioral types are static and cannot adapt to current teammates' behaviors in an online fashion, teamwork would suffer from serious miscoordination (Manish Ravula and Stone, 2019; Chen et al., 2020). Rescue and search tasks are an essential class of such examples (Manish Ravula and Stone, 2019). On the other hand, existing techniques (Barrett and Stone, 2015; Stefano V. Albrecht, 2017; Chen et al., 2020; Manish Ravula and Stone, 2019) try to utilize Bayesian posteriors over teammate types to obtain optimal responses. To effectively compute posteriors, they usually assume that the agent could always know other teammates' observations and actions. However, this assumption is unrealistic in partial observable environments, in which each agent is not aware of other agents' observations.

To address the issues mentioned above, this paper introduces an adaptive reinforcement learning framework called Online aDaptation via Inferred Teamwork Situations (ODITS). Our key insight is that teamwork performance is jointly affected by the autonomous agent and other teammates' behaviors. So, the agent's optimal behavior

depends on the current teamwork situation, which indicates the influence on the environmental dynamics caused by other teammates. If the agent identifies the current teamwork situation in an online fashion, it can choose actions accordingly to ensure effective coordination. In this way, we introduce an encoder to automatically encode the core knowledge about the teamwork situations into a latent probabilistic variable. We show that without any prior knowledge, after learning from the interactive experience with given teammates, the latent variable is sufficiently descriptive to provide information about how to coordinate with new teammates' behaviors. To overcome partial observability, we propose an information-based proxy encoder to implicitly infer the learned variables from local observations. Then, the autonomous agent adapts to new teammates' behaviors dynamically and quickly by conditioning its policy on the inferred variables.

Instead of limiting teammates into several predefined and fixed types, ODITS considers a mechanism of how an agent should adapt to teammates' behaviors online. It automatically learns continuous representations of teammates' behaviors to infer how to coordinate with current teammates' actions effectively. Without domain knowledge on current environments, it enables effective ad hoc teamwork performance and fast adaptation to varying teammates, which the agent might not thoroughly observe under partial observability. In our experimental evaluation, by interactive with a small set of given teammates, the trained agents could robustly collaborate with diverse new teammates. Compared with various type-based baselines, ODITS reveals superior ad hoc teamwork performance. Moreover, our ablations show both the necessity of learning latent variables of teamwork situations and inferring the proxy representations of learned variables.

# 2 RELATED WORKS

Ad Hoc Teamwork. The core challenge of achieving cooperative ad hoc teamwork is to develop an adaptive policy robust to various unknown teammates' behaviors (Stone et al., 2010). Existing type-based approaches try to predefine types of teammates and choose policies accordingly to cooperate with unknown teammates (Chen et al., 2020; Manish Ravula and Stone, 2019; Durugkar et al., 2020; Mirsky et al., 2020; Barrett and Stone, 2015). Specifically, PLASTIC (Barrett and Stone, 2015) infers types of teammates by computing Bayesian posteriors over all types. ConvCPD (Manish Ravula and Stone, 2019) extends this work by introducing a mechanism to detect the change point of the current teammate's type. AATEAM (Chen et al., 2020) proposes an attention-based architecture to infer types in real time by extracting the temporal correlations from the state history. The drawback of these approaches is that finite and fixed types might not cover all possible situations in complex environments. One recent work avoids predefining teammates' types by leveraging graph neural networks (GNNs) to estimate the joint action value of an ad hoc team (Rahman et al., 2020). However, this work requires all teammates' observations as input, which might not always be available in the real world.

Agent Modeling. By modeling teammates' behaviors, approaches of agent modeling aims to provide auxiliary information, such as teammates' goals or future actions, for decision-making (He et al., 2016; Albrecht and Stone, 2018). Regarding the auxiliary information as the extra observation, the modeling agent can yield an adaptive policy to its teammates' behaviors. However, existing agent models require the full observations of teammates as input (Raileanu et al., 2018; Grover et al., 2018; Tacchetti et al., 2019). If the agent cannot always observe other teammates' information (e.g. observations and actions), those approaches would fail to give an accurate prediction about teammates' information.

Multi-agent Reinforcement Learning (MARL). Cooperative MARL(Foerster et al., 2017) with centralized training and decentralized execution (Oliehoek et al., 2008) (CTDE) is relevant to this work. Related approaches (Sunehag et al., 2018; Rashid et al., 2018) utilize value function factorization to overcome the limitations of both joint and independent learning paradigms simultaneously. However, these algorithms assume that the developed team is fixed and closed. The team configuration (e.g., team size, team formation, and goals) is unchanged, and agents will not meet other agents without pre-coordination. Several extended works improve the generalization ability for changed team configurations by leveraging other insights, like learning dynamic roles (Wang et al., 2020b;a) and randomized entity-wise factorization (Iqbal et al., 2020). However, intrinsically, these approaches usually focus on co-training a group of highly-coupled agents instead of a single autonomous agent that can adapt to diverse teammates' policies. Without the need to interact with other unknown agents, the agents developed by MARL approaches evolve to overfit each other. When cooperating with new teammates, they surely result in poor ad hoc teamwork performance.

# 3 BACKGROUND

Problem Formalization. Our aim is to develop a single autonomous agent, which we refer to as the ad hoc agent, that can effectively cooperate with various teammates under partial observability without pre-coordination, such as joint-training. While we focus on training a single agent in this work, similar approaches can be applied to construct an ad hoc team.

![](images/20a5d36acbb24727bc0f2c12e5dbcaad803ee649cc1fc47499e69d9f757d8ddf.jpg)  
Figure 1: Visualization of the Dec-POMDP with an additional teammate set  $\Gamma$ .

To evaluate the ad hoc agent's ability to cooperate with unknown teammates, we formally define the problem as a decentralized Partially observable Markov Decision Process (Dec-POMDP) (Oliehoek et al., 2008) with an additional assumption about the set of teammates' possible policies  $\Gamma$ .

It can be represented as a tuple  $\langle N, S, \mathcal{A}, \mathcal{O}, \mathcal{T}, P, R, O, \Gamma \rangle$ , where  $N$  denotes the number of agents required by the task,  $s \in S$  denotes the global state of the environment. The joint action  $\boldsymbol{a} \in \mathcal{A}^N$  is formed by all agent's independent actions  $a^i \in \mathcal{A}$ , where  $i$  is the index of the agent. In the environment, each agent only has access to its partial observation  $o^i \in \mathcal{O}$  drawn according to the observation function  $O(s, i)$ , and it has an observation-action history  $\tau^i \in \mathcal{T} \equiv (\mathcal{O} \times \mathcal{A})^*$ .  $P(s', |s, \boldsymbol{a})$  denotes the probability that taking joint action  $\boldsymbol{a}$  in state  $s$  results in a transition to state  $s'$ .  $R(s, \boldsymbol{a})$  is the reward function that maps a state  $s$  and a joint action  $\boldsymbol{a}$  to a team reward  $r \in \mathbb{R}$ .  $\Gamma$  represents a pool of various policies, which can be pretrained or predefined to exhibit cooperative behaviors. Without loss of generality, we denote by  $\pi_i$  the policy of the ad hoc agent and by  $\pi_{-i}$  the joint policy of all other agents.

Fig.1 shows the detailed schematics of this problem. In a Dec-POMDP with an additional teammate set  $\Gamma$ , the objective of the ad hoc agent is to maximize the expected team return when it teams up with  $N - 1$  arbitrary teammates sampled from  $\Gamma$ , though it has no prior knowledge about those teammates. Therefore, the ad hoc agent's optimal policy  $\pi_{i}^{*}$  is required to maximize the joint action value  $Q^{\pi_i}(s,a^i,\pmb{a}^{-i})$ , which indicates the expected accumulative team reward over different ad hoc teamwork:

$$
Q ^ {\pi_ {i}} \left(s, a ^ {i}, \boldsymbol {a} ^ {- i}\right) = \mathbb {E} _ {a _ {t = 1: + \infty} ^ {i} \sim \pi_ {i}, \boldsymbol {a} _ {t = 1: + \infty} ^ {- i} \sim \boldsymbol {\pi} _ {- i}, \boldsymbol {\pi} _ {- i} \sim \Gamma} \left[ \sum_ {t = 0} ^ {+ \infty} \gamma^ {t} r _ {t} \mid s _ {0} = s, \boldsymbol {a} _ {0} = \boldsymbol {a}, P \right] \tag {1}
$$

$$
Q ^ {\pi_ {i} ^ {*}} (s, a ^ {i}, \boldsymbol {a} ^ {- i}) \geq Q ^ {\pi_ {i}} (s, a ^ {i}, \boldsymbol {a} ^ {- i}), \forall \pi_ {i}, s, a _ {i}, \boldsymbol {a} ^ {- i} \tag {2}
$$

Marginal Utility is defined to measure the contribution of an ad hoc agent to the whole team utility (Genter and Stone, 2011). Given teammates' actions  $\pmb{a}^{-i}$ , there is a relationship between the marginal utility and the team utility (denoted by the joint action value) as follow:

$$
\arg \max  _ {a ^ {i}} u ^ {i} \left(s, a ^ {i}, \boldsymbol {a} ^ {- i}\right) = \arg \max  _ {a ^ {i}} Q ^ {\pi_ {i}} \left(s, a ^ {i}, \boldsymbol {a} ^ {- i}\right) \tag {3}
$$

where  $u^i (s,a^i,\pmb{a}^{-i})$  denotes the marginal utility when the ad hoc agent chooses the action  $a^i$  under the state  $s$ . Note that the marginal utility is not necessarily equal to the Q-value (Sunehag et al., 2018). The ad hoc agent chooses the action which maximizes the marginal utility to ensure the maximal team utility.

# 4 ODITS LEARNING FRAMEWORK

Our approach addresses the ad hoc teamwork problem with a novel probabilistic framework ODITS. In this section, we first introduce the overall architecture of this framework and then present a detailed description of all modules in this framework.

# 4.1 OVERVIEW

ODITS aims to estimate the ad hoc agent's marginal utility for choosing corresponding actions to maximize the team utility. To achieve the adaptive policy to unknown teammates, we model the marginal utility as a conditional function on the inferred latent variable, which implicitly represents the current teamwork situation. ODITS jointly optimizes the marginal utility function and the latent variable by two learning objectives in an end-to-end fashion. Fig.2 shows the detailed schematics of ODITS. It splits the team into two parts: teammates and the ad hoc agent.

![](images/1109bf289b047d8dba1955f1c948bad9f0ddfa67777e0e672a05740e65be2ce4.jpg)  
Figure 2: Schematics of ODITS.

First, we regard other teammates as a part of environ-

mental dynamics perceived by the ad hoc agent. Since different combinations of teammates lead to diverse and complex dynamics, we expect to learn a latent variable to describe the core information of teammates' behaviors implicitly. To do this, we introduce a teamwork situation encoder  $f$  to learn the variable. Then, a loss function (Q loss), an integrating network  $G$  and a teamwork situation decoder  $g$  are jointly proposed to regularize the information embedded in the learned variable  $c_{t}^{i}$ .

For the ad hoc agent, we expect to condition its policy on the learned variable  $c_t^i$ . However, the partial observability impedes the direct access to  $c_t^i$ . Thus, we introduce a proxy encoder  $f^*$  to infer a proxy representation  $z_t^i$  of  $c_t^i$  from local observations. We force  $z_t^i$  to be informationally consistent with  $c_t^i$  by an information-based loss function (MI loss). Then, we train a marginal utility network  $M$  to estimate the ad hoc agent's conditional marginal utility  $\hat{u}^i(\tau_t^i, a_t^i; z_t^i) \approx u^i(s_t, a_t^i, a_t^{-i})$ . For conditional behavior, a part of parameters of  $M$  are generated by the proxy decoder  $g^*$ .

Similar to the CTDE scenario (Oliehoek et al., 2008), we relax the partial observability in the training phase. ODITS is granted access to the global state  $s_t$  and other teammates' actions  $a_t^{-i}$  during training. During execution,  $G, f, g$  is removed; the ad hoc agent chooses the action which maximizes the conditional marginal utility function  $\hat{u}^i(\tau_t^i, a_t^i; z_t^i)$ .

# 4.2 LEARNING TO REPRESENT TEAMWORK SITUATIONS

For adaptive behaviors, we expect to condition the ad hoc agent's policy on other teammates. However, unknown teammates show complex behaviors. Directly conditioning the policy on them might lead to a volatile policy. To address this issue, we aim to embed the teammates' information into a compact but descriptive representation. To model the concept clearly, we formally define teamwork situation:

Definition 1 (teamwork situation) At each time step  $t$ , the ad hoc agent is in the teamwork situation  $c_t^i \in \mathcal{C}$  which is the current underlying teamwork state yielded by the environment state  $s_t$  and other teammates' actions  $\mathbf{a}_t^{-i}$ . It reflects the high-level semantics about the teammates' behaviors.

Though different teammates generate diverse state-action trajectories, we assume that they can cause similar teamwork situations at some time, and the ad hoc agent's action would affect their transitions. When the current teamwork situation is identified, the ad hoc agent can choose the action accordingly to ensure online adaptation.

Teamwork Situation Encoder  $f$ . To model the uncertainty of unknown teammates, we encode teamwork situations in a stochastic embedding space  $\mathcal{C}$ . Thus, any teamwork situation can be represented as a latent probabilistic variable  $c^i$  that is drawn from a multivariate Gaussian distribution  $\mathcal{N}(\mu_{c^i},\sigma_{c^i})$ . To enable the

dependency revealed in the definition, we use a trainable neural network  $f$  to learn the parameters of the Gaussian distribution of  $c^i$ :

$$
\left(\mu_ {c ^ {i}}, \sigma_ {c ^ {i}}\right) = f (s, \boldsymbol {a} ^ {- i}; \theta_ {f}), c ^ {i} \sim \mathcal {N} \left(\mu_ {c ^ {i}}, \sigma_ {c ^ {i}}\right) \tag {4}
$$

where  $\theta_{f}$  are parameters of  $f$

Regularizing Information Embedded in  $c^i$ . We introduce a set of modules to jointly force  $c^i$  to be sufficiently descriptive for reflecting the current teamwork situation. The key insight is that if  $c_t^i$  is able to capture the core knowledge about other teammates' current behaviors, we can predict the joint action value  $Q^{\pi_i}(s_t,a_t^i,\pmb{a}_t^{-i})$  according to  $c_t^i$  and the ad hoc agent's marginal utility  $u_{t}^{i}$ . Thus, we propose an integrating network  $G$  for generating the joint action value's estimation  $G(u_{t}^{i},c_{t}^{i})\approx Q^{\pi_{i}}(s_{t},a_{t}^{i},\pmb{a}_{t}^{-i})$ . We adopt a modified asynchronous Q-learning's loss function (Q-loss) (Mnih et al., 2016) as the optimization objective:

$$
\mathcal {L} _ {Q} = \mathbb {E} _ {\left(u _ {t} ^ {i}, c _ {t} ^ {i}, r _ {t}\right) \sim \mathcal {D}} \left[ \right.\left[ \right.\left( \right.r _ {t} + \gamma \max  _ {a _ {t + 1} ^ {i}} \bar {G} \left(u _ {t + 1} ^ {i}, c _ {t + 1} ^ {i}\right) - G \left(u _ {t} ^ {i}, c _ {t} ^ {i}\right)\left. \right] ^ {2} \left. \right] \tag {5}
$$

where  $\bar{G}$  is a periodically updated target network. The expectation is estimated with uniform samples from the replay buffer  $\mathcal{D}$ , which saves the interactive experience with training teammates.

Integrating Network  $G$ . One simple approach for integrating  $c^i$  with  $u^i$  is to formulate  $G$  as an MLP that maps their concatenation into the joint value estimation. We instead propose to map  $c^i$  into the parameters of  $G$  by a hypernetwork, which we refer to as the teamwork situation decoder  $g$ . Then,  $G$  maps the ad hoc agent's utility  $u^i$  into the value estimation. This alternative design changes the procedure of information integration. The key insight is that the decoder provides multiplicative integration to aggregate information. By contrast, the concatenation-based operation only provides additive integration, leading to a poor information integration ability (see Supplementary). We also empirically show that multiplicative integration stabilizes the training procedure and improves teamwork performance. In addition, we expect that there is a monotonicity modeling the relationship between  $G$  and the marginal utility  $u_t^i: \frac{\partial G}{\partial u_t^i} \geq 0$ . Given any  $c_t^i$ , the increase of the ad hoc agent's marginal utility results in the improved joint action value. To achieve this property, we force  $\theta_G \geq 0$ .

# 4.3 LEARNING CONDITIONAL MARGINAL UTILITY FUNCTION UNDER PARTIAL OBSERVABILITY

Apparently, the marginal utility of the ad hoc agent depends on other teammates' behaviors. Distinct behaviors result in different marginal utilities. Here, we formalize the marginal utility network  $M$  as a deep recurrent Q network (DRQN) (Hausknecht and Stone, 2015) parameterized by  $\theta_{M}$ . To enable adaptation, we force the final layers' parameters of  $M$  to condition on the learned variable  $c_{t}^{i}$ .

Proxy Encoder  $f^{*}$ . Because of partial observability, the teamwork situation encoder  $f$  is not available during execution. Thus, we introduce a proxy encoder  $f^{*}$  to estimate  $c_{t}^{i}$  from the local transition data  $b_{t}^{i} = (o_{t}^{i}, r_{t-1}, o_{t-1}^{i}, o_{t-1}^{i})$ . We assume that  $b_{t}^{i}$  can partly reflect the current teamwork situation since the transition implicitly indicates the underlying dynamics, which is primarily influenced by other teammates' behaviors. We denote the estimation of  $c_{t}^{i}$  as  $z_{t}^{i}$ . Then,  $z_{t}^{i}$  would be fed into a proxy decoder  $g^{*}(z_{t}^{i}; \theta_{g^{*}})$  parameterized by  $\theta_{g^{*}}$  to generate the parameters  $\theta_{M}$  of  $M$ , enabling the marginal utility function to condition on the proxy representation  $z_{t}^{i}$ . Similar to  $c^{i}$ , we encode  $z^{i}$  into a stochastic embedding space:

$$
\left(\mu_ {z ^ {i}}, \sigma_ {z ^ {i}}\right) = f ^ {*} \left(b ^ {i}; \theta_ {f ^ {*}}\right), z ^ {i} \sim \mathcal {N} \left(\mu_ {z ^ {i}}, \sigma_ {z ^ {i}}\right) \tag {6}
$$

where  $\theta_{f^*}$  are parameters of  $f^*$ .

Regularizing Information Embedded in  $z^i$ . To make  $z_t^i$  identifiable, we expect  $z_t^i$  to be informatively consistent with  $c_t^i$ . Thus, we introduce an information-based loss function  $\mathcal{L}_{MI}$  here to maximize the conditional mutual information  $I(z_t^i; c_t^i | b_t^i)$  between the proxy variables and the true variables. However, estimating and maximizing mutual information is often infeasible. We introduce a variational distribution  $q_{\xi}(z_t^i | c_t^i, b_t^i)$  parameterized by  $\xi$  to derive a tractable lower bound for the mutual information (Alemi et al., 2017):

![](images/2a07f894ffdcf004c8a266750c738f13c4d60e688dc197dbcce7d06605862c5a.jpg)  
Figure 3: Illustration of the modified coin game (left) and performance comparison of two algorithms (right).

![](images/bd22d553b47c5f617ce6d424500bc1c919274f9c27721070a055076868267ae6.jpg)

$$
I \left(z _ {t} ^ {i}; c _ {t} ^ {i} \mid b _ {t} ^ {i}\right) \geq \mathbb {E} _ {z _ {t} ^ {i}, c _ {t} ^ {i}, b _ {t} ^ {i}} \left[ \log \frac {q _ {\xi} \left(z _ {t} ^ {i} \mid c _ {t} ^ {i} , b _ {t} ^ {i}\right)}{p \left(z _ {t} ^ {i} \mid b _ {t} ^ {i}\right)} \right] \tag {7}
$$

where  $p(z_t^i | b_t^i)$  is the Gaussian distribution  $\mathcal{N}(\mu_{z^i}, \sigma_{z^i})$ . This lower bound can be rewritten as a loss function to be minimized:

$$
\mathcal {L} _ {M I} \left(\theta_ {f ^ {*}}, \xi\right) = = \mathbb {E} _ {\left(b _ {t} ^ {i}, s ^ {t}, \boldsymbol {a} _ {t} ^ {- i}\right) \sim \mathcal {D}} \left[ D _ {K L} \left[ p \left(z _ {t} ^ {i} \mid b _ {t} ^ {i}\right) \right] \mid q _ {\xi} \left(z _ {t} ^ {i} \mid c _ {t} ^ {i}, b _ {t} ^ {i}\right) \right] \tag {8}
$$

where  $\mathcal{D}$  is the replay buffer,  $D_{KL}[\cdot ||\cdot ]$  is the KL divergence operator. The detailed derivation can be found in Supplementary.

# 4.4 OVERALL OPTIMIZATION OBJECTIVE

To the end, the overall objective becomes:

$$
\mathcal {L} (\theta) = \mathcal {L} _ {Q} (\theta) + \lambda \mathcal {L} _ {M I} \left(\theta_ {f ^ {*}}, \xi\right) \tag {9}
$$

where  $\theta = (\theta_{f},\theta_{g},\theta_{M},\theta_{f^{*}},\theta_{p},\xi)$ ,  $\lambda$  is the scaling facor.

During the training phase, the ad hoc agent interacts with different training teammates for collecting transition data into the replay buffer  $\mathcal{D}$ . Then, samples from  $\mathcal{D}$  are fed into the framework for updating all parameters by gradients induced by the overall loss. During execution, the ad hoc agent conditions its behavior on the inferred teamwork situations by choosing actions to maximize the conditional utility function  $u^{i}(\tau_{t}^{i},a_{t}^{i};z_{t}^{i})$ . We summarize our training procedure and testing procedure in Supplementary.

# 5 EXPERIMENTS

We now empirically evaluate ODITS on various new and existing domains. All experiments in this paper are carried out 4 different random seeds, and results are shown with a  $95\%$  confidence interval over the standard deviation. In the following description, we refer to the teammates that interact with the ad hoc agent during the training phase as the training teammates, and refer to those teammates with unknown policies as testing teammates. And the "teammate types" correspond to the policy types of teammates. All experimental results illustrate the average teamwork performance when the ad hoc agent cooperates with different testing teammates. Further experiment details and hyperparameters and implementation details of all models can be found at Supplementary.

# 5.1 MODIFIED COIN GAME

To show the difference between ODITS and type-based approaches, we introduce a simple modified coin game. The game takes place on a  $7 \times 7$  map which contains 6 coins of 3 different colors (2 coins of each color). The aim of the team is to only collect any two kinds of coins (correct coins with a reward of 100) and avoid collecting the other kind of coins (false coins with a reward of -200). The policies of the teammates are predefined and illustrated in the order of colors in Fig.3 (2 training types and 2 testing types). For example, the first training type

![](images/7469fcc30e672d8f4b8119a83343226d381a6f644be91607b476158bda76d4b3.jpg)

![](images/a988ff5a7bea7cf4aba28024df30e316bb0f557165ae9e488bfb8a22d3dfe1d4.jpg)

![](images/9240833122347b098efe7b3693bfa6eb0576dfe77327ccabdd82b34afd4a9b9a.jpg)

![](images/e6e986cd07b331985fa6ca1d0fb3106aeb4d3a86622684f538425e9bc09980da.jpg)  
Figure 4: Performance comparison across various scenarios for Predator Prey (top panel) and Save the City (bottom panel).

![](images/2fd8b8e789146aaba7b20607ac9be1948782ef64ee7d2fbc143c20a069e7f320.jpg)

![](images/20a792ed045086a146dfb85390b9f9f8220e64e5b925e562b2bbdbcce96b746c.jpg)

(red  $\rightarrow$  green) indicates that the policy of this teammate is to collect red and green coins, and it will collect red coins firstly. Therefore, while the correct coins of the first training type (green  $\rightarrow$  red) and the second testing type (red  $\rightarrow$  green) are the same, they are different policies since their paths to collect coins are apparently different. Each agent has five actions: move up, down, left, right, or pass. Once an agent steps on a coin, that coin disappears from the cell. The game ends after 20 steps. To maximize the team return, the aim of the ad hoc agent is to infer and collect its current teammate's desired coins.

Here, we adopt one state-of-the-art type-based approach (AATEAM (Chen et al., 2020)) as the baseline. Fig.3 right shows the testing performance. We observe that ODITS shows superior performance and converges quickly while AATEAM shows an unstable curve. We believe this discrepancy results from the key difference between our method and type-based approaches. The baseline is hard to cooperate with new types of teammates. For example, when the baseline agent interacts with the teammate of the second testing type (green  $\rightarrow$  red) and observes that the teammate is collecting the green coins at the start stage, it would switch its own policy to the one corresponding to the second training type of teammate (green  $\rightarrow$  blue), so it would collect green coins and blue coins (false coins) simultaneously, leading to poor teamwork performance. By contrast, ODITS can be easily generalized to the testing types of teammates. During training, ODITS learns how to cooperate with the teammate according to its current behavior instead of its types. If it observes that its teammate is collecting one kind of coins, it will collect the same kind of coins, and this knowledge is automatically embedded in  $c / z$ .

# 5.2 PREDATOR PREY

Configurations. In this environment,  $m$  homogenous predators try to capture  $n$  randomly-moving preys in a  $7 \times 7$  grid world. Each predator has six actions: the moving actions in four directions, the capturing action, and waiting at a cell. Due to partial observability, each predator can only access the environmental information within two cells nearby. Besides, there are two obstacles at random locations. Episodes are 40 steps long. The predators get a team reward of 500 if two or more predators are capturing the same prey at the same time, and they are penalized for -10 if only one of them tries to capture a prey. Here, we adopt three different settings to verify the ad hoc agent's ability to cooperate with different number of teammates. They are 2 predators and 1 preys (2d1y), 4 predators and 2 preys (4d2y) and 8 predators and 4 preys (8d4y), respectively.

We compare our methods with three type-based baselines: AATEAM (Chen et al., 2020), ConvCPD (Manish Ravula and Stone, 2019), PLASTIC (Barrett and Stone, 2015). Note that these approaches assume that the ad hoc agent has full visibility on the environment. To apply them on partially observed settings, we replace the full state information used in them with partial observations of the ad hoc agent. Furthermore, we also compare two other strategies: (i) Random: The ad hoc agent chooses actions randomly. (ii) Combined: The ad hoc agent utilizes a DQN algorithm to learn a single policy using the data collected from all possible teammates. This intuitive baseline provides the view of treating the problem as a vanilla single-agent learning problem, where the agent ignores the differences between its teammates.

Before training all algorithms, we first require a teammate set that consists of various behavioral policies of teammates. Instead of crafting several teammates' cooperative policies by hand, we expect to train a set of distinct policies automatically. Therefore, we first utilize 5 different MARL algorithms (e.g. VDN (Sunehag et al., 2018)

and QMIX (Rashid et al., 2018)) to develop several teams of agents. To ensure diversity, we use different random seeds for each algorithm and save the corresponding models at 3 different checkpoints (3 million steps, 4 million steps, and 5 million steps). Then, we manually select 15 different policies showing distinct policy representations (Grover et al., 2018) from all developed models. Finally, we randomly sampled 8 policies as the training set and the other 7 policies as the testing set. During training, we define 8 teammate types that correspond to 8 policies in the training set for the type-based approaches. Then, algorithms would develop their models according to the interactive experience with training teammates. For all algorithms, agents are trained for 4.5 million time steps. The number of captured preys when the ad hoc agent cooperates with testing teammates throughout training is reported. See Supplementary for further settings.

Results. The top panel of Fig. 4 reports the results across 3 scenarios. We first observe that ODITS achieves superior results on the number of captured preys across a varying number of teammates, verifying its effectiveness. ODITS also tends to give more consistent results than other methods across different difficulties. The other 3 type-based baselines and ODITS show better results than random and combined policies, indicating that they can indeed lead to adaptive behaviors to different teammates. Furthermore, the random strategy captures some preys on  $4\mathrm{d}2\mathrm{y}$  and  $8\mathrm{d}4\mathrm{y}$ , but completely fails on  $2\mathrm{d}1\mathrm{y}$ . This indicates that without the cooperative behaviors of the ad hoc agent, other teammates can also coordinate with each other to achieve the given goal. The combined policy shows worse results than the random policy on two scenarios ( $4\mathrm{d}2\mathrm{y}$  and  $8\mathrm{d}4\mathrm{y}$ ). This might be because the combined policy shows behaviors that conflict with other teammates. With the number of teammates increasing, the increasing effects of conflicts lead to serious miscoordination.

# 5.3 SAVE THE CITY

Configurations. This is a grid world resource allocation task presented in (Iqbal et al., 2020). In this task, there are 3 distinct types of agents, and their goal is to complete the construction of all buildings on the map while preventing them from burning down. Each agent has 8 actions: stay in place, move to the next cell in one of the four cardinal directions, put out the fire, and build. We set the agents to get a team reward of 100 if they have completed a building and be penalized for -500 when one building is burned down. Agent types include firefighters (20x speedup over the base rate in reducing fires), builders (20x speedup in the building), or generalists (5x speedup in both as well 2x speedup in moving). Buildings also consist of two varieties: fast-burning and slow-burning, where the fast-burning buildings burn four times faster. In our experiments, each agent can only access the environmental information within four cells nearby. We adopt three different scenarios here to verify all methods. They are 2 agents and 1 buildings (2a2b), 4 agents and 3 buildings (4a3b), 6 agents and 4 buildings (6a4b).

Similar to training settings in Predator Prey, we select 15 distinct behavioral policies for the teammate set and randomly partition them into 8 training policies and 7 testing policies. For all algorithms, agents are trained for 4.5 million time steps. The number of completed buildings when the ad hoc agent cooperates with testings teammates throughout training is reported. See Supplementary for further settings.

Results. The bottom panel of Fig. 4 reports the results across 3 scenarios. We first observe that ODITS outperforms other baselines, verifying its effectiveness. Since the setting force all agents in the environment to be heterogeneous, the results also underpin the robustness of ODITS. Interestingly, we find that the combined policy reveals better performance than other type-based approaches. This result is not consistent with that in Predator Prey. Our intuition is that the requirement of cooperative behaviors in Save the City is less than that in Predator Prey. Actually, one agent in Save the City can complete buildings individually without the strong necessity of cooperating with other teammates' behaviors. By contrast, one predator cannot capture prey by itself. As a result, the combined policy learns a universal and effective policy by learning from more interactive experience with different teammates, while type-based approaches fail because developing distinct cooperative behaviors leads to the instability of the ad hoc agent. This hypothesis is also empirically demonstrated in our ablations.

![](images/c817cf50f12dfd746efdf5c3ed1cf0301c0ff0bb82688d78cb45968312ef2ee6.jpg)  
Figure 5: ODITS V.S. QMIX.

# 5.4 COMPARISON WITH MARL

In order to compare the performance of ODITS and MARL, we implement the commonly used algorithm QMIX (Rashid et al., 2018) as our baseline. Similar to the training procedure of ODITS, we fix one agent and train it with teammates randomly sampled from a pool consisting of 8 policies. The gradients for updating the teammates' policies are blocked but the mixing network is updating as in the original implementation of QMIX.

Figure 5 shows the comparison of ODITS and QMIX on Predator Prey  $4\mathrm{d}2\mathrm{y}$  and Save the City  $4\mathrm{a}3\mathrm{b}$ . In both environments, QMIX performs significantly worse than ODITS. This is not quite surprising because MARL algorithms usually assume that all the teammates are fixed. Therefore, although trained with multiple teammates, the agent under the QMIX framework does not learn to cooperate with an impromptu team.

# 5.5 ABLATIONS.

We perform several ablations on the Predator Prey 4d2y and Save the City 4a3b to try and determine the importance of each component of ODITS.

Adaptive Behaviors. We first consider removing the information-based loss  $\mathcal{L}_{MI}$  from the overall learning objective (denoted as w/o info.), Fig. 6 shows that without  $\mathcal{L}_{MI}$  regularizing the information embedded in  $z_{t}^{i}$ , ODITS induces worse teamwork performance. This indicates that improving the mutual information between the proxy variable and the true variable indeed results in better representations of teamwork situations. We next consider how the inferred variables of teamwork situations affect the ad hoc agent's adaptive behaviors. We remove the proxy encoder and set  $z_{t}^{i}$  as a fixed and randomly generated vector (denoted as w/o infer.). As shown in Fig. 6, conditioning on a random signal leads to a further drop in performance, indicating that irrelevant signals cannot promote the ad hoc agent to develop adaptive policies.

![](images/e26b8639f38bb8ba9a27aec87cc80d1ee6fee9a85366041d1be933e251d648b2.jpg)  
Figure 6: Ablations for different components.

Integrating Mechanism. We remove the teamwork situation encoder as well as  $\mathcal{L}_{MI}$  from the framework and feed a vector filled with 1 into the teamwork situation decoder (labeled with w/o integ.). This setting enables ODITS not to integrating the ad hoc agent's marginal utility with the information of teammates' behaviors. Compared with w/o info., it brings a larger drop in teamwork performance. One intuition is that predicting the joint-action value plays an essential role in estimating the marginal utility. Suppose the integrating network has no information on other teammates' behaviors. In that case, it cannot accurately predict the joint-action value, resulting in instability in marginal utility estimation. Despite the empirical evidence supporting this argument, however, it would be interesting to develop further theoretical insights into this training regime in future work. We finally consider the additive integration mechanism mentioned in section 4.2 (labeled with additive). We observe that despite additive integration shows an excellent performance in Save the City, it suffers from poor performance in Predator Prey, indicating that multiplicative integration provides a more stable and effective ability to integrate information from teammates and the ad hoc agent. Interestingly, we also find that most ablations get worse results in Predator Prey than those in Save the city. We believe that the different levels of cooperative requirement in two environments result in this phenomenon. The prey is captured when two nearby predators are simultaneously capturing them. By contrast, the burning building can be constructed by an individual agent. Therefore, removing mechanisms that promote the cooperative behaviors leads to the worse performance in Predator Prey.

# 6 CONCLUSIONS

This paper proposes a novel adaptive reinforcement learning algorithm called ODITS to address the challenging ad hoc teamwork problem. Without the need to predefine types of teammates, ODITS automatically learns compact but descriptive variables to infer how to coordinate with previously unknown teammates' behaviors. To overcome partial observability, we introduce an information based regularizer to estimate proxy representations of learned variables from local observations. Experimental results show that ODITS obtains superior performance compared to various baselines on several complex ad hoc teamwork benchmarks.

# REFERENCES

S. V. Albrecht and P. Stone. Autonomous agents modelling other agents: A comprehensive survey and open problems. Artificial Intelligence, 258:66-95, 2018.  
A. A. Alemi, I. Fischer, J. V. Dillon, and K. Murphy. Deep variational information bottleneck. In  $ICLR$ , 2017.  
N. Bard, J. N. Foerster, S. Chandar, N. Burch, M. Lanctot, H. F. Song, E. Parisotto, V. Dumoulin, S. Moitra, E. Hughes, I. Dunning, S. Mourad, H. Larochelle, M. G. Bellemare, and M. Bowling. The hanabi challenge: A new frontier for ai research. Artificial Intelligence, 280:103216, 2020. ISSN 0004-3702. doi: https://doi.org/10.1016/j.artint.2019.103216. URL https://www.sciencedirect.com/science/article/pii/S0004370219300116.  
S. Barrett and P. Stone. Cooperating with unknown teammates in complex domains: A robot soccer case study of ad hoc teamwork. In AAAI, January 2015.  
S. Chen, E. Andrejczuk, Z. Cao, and J. Zhang. Aateam: Achieving the ad hoc teamwork by employing the attention mechanism. In AAAI, volume 34, pages 7095-7102, 2020.  
I. Durugkar, E. Liebman, and P. Stone. Balancing individual preferences and shared objectives in multiagent reinforcement learning. In Proceedings of the 29th IJCAI), July 2020.  
J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson. Counterfactual multi-agent policy gradients. AAAI, 2017.  
N. A. Genter, Katie and P. Stone. Role-based ad hoc teamwork. In AAAI, 2011.  
A. Grover, M. Al-Shedivat, J. K. Gupta, Y. Burda, and H. Edwards. Learning policy representations in multiagent systems. In Proceedings of the 35th ICML, 2018.  
M. Hausknecht and P. Stone. Deep recurrent q-learning for partially observable mdps. In AAAI, November 2015.  
H. He, J. Boyd-Graber, K. Kwok, and H. Daumé. Opponent modeling in deep reinforcement learning. In Proceedings of the 33rd ICML, pages 1804–1813, 2016.  
S. Iqbal, C. A. S. de Witt, B. Peng, W. Böhmer, S. Whiteson, and F. Sha. Randomized entity-wise factorization for multi-agent reinforcement learning. arXiv preprint arXiv:2006.04222, 2020.  
A. Mahajan, T. Rashid, M. Samvelyan, and S. Whiteson. Maven: Multi-agent variational exploration. In NeurlPS, pages 7613-7624, 2019.  
S. A. Manish Ravula and P. Stone. Ad hoc teamwork with behavior switching agents. In IJCAI, August 2019.  
R. Mirsky, W. Macke, A. Wang, H. Yedidzion, and P. Stone. A penny for your thoughts: The value of communication in ad hoc teamwork. In Proceedings of the 29th IJCAI, July 2020.  
V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In ICML, pages 1928-1937, 2016.  
F. A. Oliehoek, M. T. Spanan, N. Vlassis, and S. Whiteson. Exploiting locality of interaction in factored dec-pomdps. In Proceedings of the 2008 AAMAS, pages 517-524, 2008.  
A. Rahman, N. Hopner, F. Christianos, and S. V. Albrecht. Open ad hoc teamwork using graph-based policy learning. In Thirty-Fifth AAAI Conference, 2020.  
R. Raileanu, E. Denton, A. Szlam, and R. Fergus. Modeling others using oneself in multi-agent reinforcement learning. In Proceedings of the 35th ICML, 2018.  
S. A. e. a. Raileanu R, Goldstein M. Fast adaptation to new environments via policy-dynamics value functions. In ICML, ICML.  
T. Rashid, M. Samvelyan, C. S. de Witt, G. Farquhar, J. Foerster, and S. Whiteson. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. In Proceedings of the 35th ICML, pages 1228-1236, 2018.

M. Samvelyan, T. Rashid, C. S. de Witt, G. Farquhar, N. Nardelli, T. G. J. Rudner, C.-M. Hung, P. H. S. Torr, J. Foerster, and S. Whiteson. The StarCraft Multi-Agent Challenge. CoRR, abs/1902.04043, 2019.  
P. S. Stefano V. Albrecht. Reasoning about hypothetical agent behaviours and their parameters. In AAMAS, 2017.  
P. Stone, G. A. Kaminka, S. Kraus, and J. S. Rosenschein. Ad hoc autonomous agent teams: Collaboration without pre-coordination. In AAAI, July 2010.  
P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. F. Zambaldi, M. Jaderberg, M. Lanctot, N. Sonnerat, J. Z. Leibo, K. Tuyls, et al. Value-decomposition networks for cooperative multi-agent learning based on team reward. In Proceedings of the 2018 AAMAS, pages 2085-2087, 2018.  
A. Tacchetti, H. F. Song, P. A. M. Mediano, V. Zambaldi, J. Kramár, N. C. Rabinowitz, T. Graepel, M. Botvinick, and P. W. Battaglia. Relational forward models for multi-agent learning. In ICLR, 2019.  
M. Tan. Multi-agent reinforcement learning independent vs. cooperative agents. In ICML, 1993.  
T. Wang, H. Dong, V. Lesser, and C. Zhang. Roma: Multi-agent reinforcement learning with emergent roles. In Proceedings of the 37th ICML, 2020a.  
T. Wang, T. Gupta, A. Mahajan, B. Peng, S. Whiteson, and C. Zhang. Rode: Learning roles to decompose multi-agent tasks. arXiv preprint arXiv:2010.01523, 2020b.  
G. N. Yannakakis. Game ai revisited. In Proceedings of the 9th conference on Computing Frontiers, pages 285-292, 2012.
