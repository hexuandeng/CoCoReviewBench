# FULLY DECENTRALIZED MODEL-BASED POLICY OPTIMIZATION WITH NETWORKED AGENTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Model-based RL is an effective approach for reducing sample complexity. However, when it comes to multi-agent setting where the number of agents is large, the model estimation can be problematic due to the exponential increased interactions. In this paper, we propose a decentralized model-based reinforcement learning algorithm for networked multi-agent systems, where agents are cooperative and communicate locally with their neighbors. We analyze our algorithm theoretically and derive an upper bound of performance discrepancy caused by model usage, and provide a sufficient condition of monotonic policy improvement. In our experiments, we compare our algorithm against other strong multi-agent baselines and demonstrate that our algorithm not only matches the asymptotic performance of model-free methods but also largely increases its sample efficiency.

# 1 INTRODUCTION

Many real world problems, such as autonomous driving, wireless communications, multi-player games can be modeled as multi-agent RL problems, where multiple autonomous agents coexist in a common environment, aiming to maximize its individual or team reward in the long term by interacting with the environment and other agents. Unlike single-agent tasks, multi-agent tasks are more challenging, due to partial observations and unstable environments when agents update their policies simultaneously. Therefore, there are hardly any one-fits-all solutions for MARL problems. Examples include networked systems control (NSC) (Chu et al., 2020), in which agents are connected via a stationary network. They perform decentralized control based on its local observations and messages from connected neighbors. Examples of networked systems include connected vehicle control (Jin & Orosz, 2014), traffic signal control (Chu et al., 2020), etc.

Despite the success of multi-agent reinforcement (RL) algorithms, their performance relies on a massive amount of model usage. Typically, a multi-agent RL algorithm needs millions of interaction with the environment to converge. On the other hand, model-based reinforcement learning (MBRL) algorithms, which utilize predictive models of the environment to help data collection, are empirically more data-efficient than model-free approaches. Although model inaccuracy performs as a bottleneck of policy quality in model-based algorithms, we can still learn a good policy with an imperfect model (Luo et al., 2019), especially combined with the trick of branched rollout (Janner et al., 2019) to limit model usage. Experimentally, MuZero (Schrittwieser et al., 2020), a model-based RL algorithm, succeeded in matching the performance of AlphaZero on Go, chess and shogi, and becomes state-of-the-art on Atari games. Model-based MARL is not fully investigated. Existing MB-MARL algorithms either limit their field of research on specific scenario, e.g. two-player zero-sum Markov game (Zhang et al., 2020) or pursuit evasion game (Bouzy & Métivier, 2007), or use tabular RL method (Bargiacchi et al., 2021). MB-MARL for multi-agent MDPs is still an open problem to be solved (Zhang et al., 2019), with profound challenges such as scalability issues caused by large state-action space and incomplete information of other agents' state or actions.

In this paper, we develop decentralized model-based algorithms on networked systems, where agents are cooperative, and able to communicate with each other. We use localized models to predict future states, and use communication to broadcast their predictions. To address the issue of model error, we adopt branched rollout (Janner et al., 2019) to limit the rollout length of model trajectories. In the policy optimization part, we use decentralized PPO (Schulman et al., 2017) with a extended value function. At last, we analyze these algorithms theoretically to bound the performance discrepancy

between our method and its model-free, centralized counterpart. At last, we run these algorithms in traffic control environments (Chu et al., 2020; Vinitsky et al., 2018) to test the performance of our algorithm. We show that our algorithm increases sample efficiency, and matches the asymptotic performance of model-free methods.

In summary, our contributions are three-fold. Firstly, we propose an algorithmic framework, which is a fully decentralized model-based reinforcement learning algorithm, which is named as Decentralized Model-based Policy Optimization (DMPO). Secondly, we analyze the theoretical performance of our algorithm. Lastly, empirical results on traffic control environments demonstrate the effectiveness of DMPO in reducing sample complexities and achieving similar asymptotic performance of model-free methods.

# 2 RELATED WORK

Model-based methods are known for their data efficiency (Kaelbling et al., 1996), especially compared with model-free algorithms. There is a vast literature on the theoretical analysis of model-based reinforcement learning. In a single-agent scenario, monotonic improvement of policy optimization has been achieved (Luo et al., 2019; Sun et al., 2018), and a later work improved the performance of model-based algorithms by limiting model usage (Janner et al., 2019). But these analysis is restricted to single-agent scenarios, whereas ours addresses multi-agent problems.

On the other hand, Networked System Control (NSC) (Chu et al., 2020) is a challenging setting for MARL algorithm to take effect. Some multi-agent algorithms falls into centralized training decentralized execution (CTDE) framework. For example, QMIX (Rashid et al., 2018) and COMA (Foerster et al., 2018) all use a centralized critic. In a large network, however, centralized training might not scale. In many scenarios, only fully decentralized algorithms can be used. Zhang et al. (2018) proposed an algorithm of NSC that can be proven to converge under linear approximation. Qu et al. (2020a) proposed truncated policy gradient, to optimize local policies with limited communication. Baking in the idea of truncated  $Q$ -learning in (Qu et al., 2020a), we generalize their algorithm to deep RL, rather than tabular RL. Factoring environmental transition into marginal transitions can be seen as factored MDP. Guestrin et al. (2001) used Dynamic Bayesian Network to predict system transition. Simao & Spaan (2019) proposed a tabular RL algorithm to ensure policy improvement at each step. However, our algorithm is a deep RL algorithm, enabling better performance in general tasks.

There are some works on applying model-based methods in MARL settings. A line of research focuses on model-based RL for two-player games. For example, Brafman & Tennenholtz (2000) solved single-controller-stochastic games, which is a certain type of two-player zero-sum game; Bouzy & Métivier (2007) performed MB-MARL in the pursuit evasion game; Zhang et al. (2020) proved that model-based method can be nearly optimally sample efficient in two-player zero-sum Markov games. Bargiacchi et al. (2021) extended the concept of prioritized sweeping into a MARL scenario. However, this is a tabular reinforcement algorithm, thus unable to deal with cases where state and action spaces are relatively large, or even continuous. In contrast to existing works, our algorithm is not only applicable to more general multi-agent problems, but is also the first fully decentralized model-based reinforcement learning algorithm.

# 3 PROBLEM SETUP

In this section, we introduce multi-agent networked MDP and model-based networked system control.

Networked MDP We consider environments with a graph structure. Specifically,  $n$  agents coexist in an underlying undirected and stationary graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$ . Agents are represented as a node in the graph, therefore  $\mathcal{V} = \{1,\dots,n\}$  is the set of agents.  $\mathcal{E}\subset \mathcal{V}\times \mathcal{V}$  comprises the edges that represent the connectivity of agents. Agents are able to communicate along the edges with their neighbors. Let  $N_{i}$  denote every neighbor of agent  $i$ , and  $\bar{N}_i = N_i\cup \{i\}$ . Furthermore, let  $N_{i}^{\kappa}$  denote the  $\kappa$ -hop neighborhood of  $i$ , i.e. the nodes whose graph distance to  $i$  is less than or equal to  $\kappa$ . For the simplicity of notation, we also define  $N_{-i}^{\kappa} = \mathcal{V}\setminus N_{i}^{\kappa}$ .

The corresponding networked MDP is defined as  $(\mathcal{G},\{\mathcal{S}_i,\mathcal{A}_i\}_{i\in \mathcal{V}},p,r)$ . Each agent  $i$  have their local state  $s_i\in S_i$ , and perform action  $a_{i}\in \mathcal{A}_{i}$ . The global state is the concatenation of all local states:  $s = (s_{1},\ldots ,s_{n})\in S\coloneqq S_{1}\times \ldots \times S_{n}$ . Similarly, the global action is  $a = (a_{1},\dots,a_{n})\in \mathcal{A}:= A_1\times \ldots \times A_n$ . For the simplicity of notation, we define  $s_{N_i}$  to be the local states of every agent in  $N_{i}$ , that is, given  $N_{i} = \{j_{1},\dots,j_{c}\}$ , then  $s_{N_i} = (s_{j_1},\dots,s_{j_c})$ .  $a_{N_i},s_{N_i^\kappa},a_{N_i^\kappa}$  are defined similarly. The transition function is defined as:  $p(s'|s,a):S\times \mathcal{A}\to S$ . Each agent possesses a localized policy  $\pi_i^{\theta_i}(a_i|s_{\bar{N}_i})$  that is parameterized by  $\theta_{i}\in \Theta_{i}$ , meaning the local policy is dependent only on states of its neighbors and itself. We use  $\theta = (\theta_{1},\dots,\theta_{n})$  to denote the tuple of localized policy parameters, and  $\pi^{\theta}(a|s) = \prod_{i = 1}^{n}\pi_{i}^{\theta_{i}}(a_{i}|s_{\bar{N}_{i}})$  denote the joint policy. We also assume that reward functions is only dependent on local state and action:  $r_i(s_i,a_i)$ , and the global reward function is defined to be the average reward  $r(s,a) = \frac{1}{n}\sum_{i = 1}^{n}r_{i}(s_{i},a_{i})$ .

The goal of reinforcement learning is to maximize the expected sum of discounted rewards, denoted by  $\eta$ :

$$
\pi^ {\theta^ {*}} = \arg \max  _ {\pi^ {\theta}} \eta [ \pi^ {\theta} ] = \arg \max  _ {\pi^ {\theta}} \mathbb {E} _ {\pi^ {\theta}} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} \cdot \frac {1}{n} \sum_ {i = 1} ^ {n} r _ {i} \left(s _ {t}, a _ {t}\right) \right], \tag {1}
$$

where  $\gamma \in (0,1)$  is the temporal discount factor. We define the stationary distribution under policy  $\pi$  to be  $d_{\pi}(s)$ .

Independent Networked System Networked system may have some extent of locality, meaning in some cases, local states and actions do not affect the states of distant agents. In such systems, environmental transitions can be factorized, and agents are able to maintain local models to predict future local states. We define Independent Networked System (INS) as follows:

Definition 1. An environment is an Independent Networked System (INS) if:

$$
p (s ^ {\prime} | s, a) = \prod_ {i = 1} ^ {n} p _ {i} (s _ {i} ^ {\prime} | s _ {\bar {N} _ {i}}, a _ {i}), \forall s ^ {\prime}, s \in \mathcal {S}, a \in \mathcal {A}.
$$

INS might be an assumption that is too strong to hold. However, for the dynamics that cannot be factorized, we can still use an INS to approximate it. Let  $D_{TV}$  denote the total variation distance between distributions, we have the following definition:

Definition 2. ( $\xi$ -dependent) Assume there exists an Independent Networked System  $\bar{p}$  such that  $\bar{p}(s'|s, a) = \prod_{i=1}^{n} p_i(s_i'|s_{\bar{N}_i}, a_i)$ . An environment is  $\xi$ -dependent, if:

$$
\sup  _ {s, a} D _ {T V} \left(p (s ^ {\prime} | s, a) \| \bar {p} (s ^ {\prime} | s, a)\right) = \sup  _ {s, a} \frac {1}{2} \sum_ {s ^ {\prime} \in \mathcal {S}} | p (s ^ {\prime} | s, a) - \bar {p} (s ^ {\prime} | s, a) | \leq \xi .
$$

To explain the intuition behind this definition, we point out that  $\xi$  is actually the lower bound of model error when we use local models  $\hat{p}(s_{\bar{N}_i}, a_i)$ . Recall that  $p(s'|s, a)$  is the real environment transition,  $\bar{p} = \prod_{i=1}^{n} p_i(s_i'|s_{\bar{N}_i}, a_i)$  is the product of marginal environment transitions, and  $\hat{p}(s, a) = \prod_{i=1}^{n} \hat{p}_i(s_i'|s_{\bar{N}_i}, a_i)$  is the product of model transitions. Then the universal model error  $D(p||\hat{p})$  can be divided into two parts: dependency bias  $D(p||\bar{p})$  and model error  $D(\bar{p}||\hat{p})$ :

$$
D (p \| \hat {p}) \leq D (p \| \bar {p}) + D (\bar {p} \| \hat {p}).
$$

Then for a  $\xi$ -dependent system, when models become very accurate, meaning  $D(\bar{p} \| \hat{p}) \approx 0$ ,  $\sup D(p \| \hat{p}) \approx \sup D(p \| \bar{p}) = \xi$ . While  $D$  can be any appropriate distance metric, we use the TV-distance hereafter for the ease of presentation. In the following, we develop theory under both INS and  $\xi$ -dependent scenarios.

# 4 DECENTRALIZED MODEL-BASED POLICY OPTIMIZATION

In this section, we formally present Decentralized Model-based Policy Optimization (DMPO), which is a fully decentralized model-based reinforcement learning algorithm. Compared with independent multi-agent PPO, DMPO is augmented in three ways: localized model, policy with one-step communication, and extended value function. We introduce the detail of localized model in 4.1. Policy and value functions are introduced in 4.2. The illustration of our algorithm is given in Figure 1. All the components mentioned above are analyzed in Section 5. We argue that under certain conditions, our algorithm ensures monotonic policy improvement.

![](images/b6040f22ad3bf482397c33d701d4f7e3c203bd03dbca01faa27ffb8abd37615f.jpg)  
(a) Neighborhood

![](images/7a47bafd824e4903a4f2c25067989ab39781c19556d6b993336059b9f79534f7.jpg)  
(b) Value function

![](images/2bc9264a1890bcbfd64b446b1cced41f1b1c624d3965934ca43eff49f5dee64e.jpg)  
Figure 1: (a) presents the concept of neighborhood. If agent  $i$  is the node in purple, then purple and orange is  $\bar{N}_i$ , and combination of purple, orange and green is  $N_i^3$ . (b) explains that extended value function takes  $s_{N_i^\kappa}$  as input, here  $\kappa = 3$ . (c) is the illustration of graph convolutional model.  
(c) Graph convolutional model

# 4.1 DECENTRALIZED PREDICTIVE MODEL

To perform decentralized model-based learning, we let each agent maintain a localized model. We allow the localized model to observe the state of 1-hop neighbor and the action of itself. The goal of a localized model is to predict the information of the next timestep, including state, reward and done. This process is denoted by  $\hat{p}_i(s_i', r_i', d_i'|s_{\bar{N}_i}, a_i)$ .

We implement a localized model with graph convolutional networks (GCN). Recall that agents are situated in a graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$ . In the first step, a node-level encoder encodes local state into node embedding,

$$
h _ {i} ^ {0} = f _ {i} ^ {\text {e n c o d e}} \left(s _ {i}\right). \tag {2}
$$

Then we perform one step of graph convolution as follows,

$$
h _ {(i, j)} = f _ {(i, j)} ^ {e d g e} (h _ {i} ^ {0}, h _ {j} ^ {0}),
$$

$$
h _ {i} ^ {1} = f _ {i} ^ {\text {n o d e}} \left(\sum_ {e = (i, j)} h _ {(i, j)}, a _ {i}\right). \tag {3}
$$

In this way,  $h_i^1$  is dependent only on  $s_{\bar{N}_i}$  and  $a_i$ . Finally, a node-level decoder generates the prediction of state, reward and done from  $h_i^1$  as follows:

$$
s _ {i} ^ {\prime} = f _ {i} ^ {s t a t e} (h _ {i} ^ {1}) + s _ {i},
$$

$$
r _ {i} ^ {\prime} = f _ {i} ^ {r e w a r d} \left(h _ {i} ^ {1}\right), \tag {4}
$$

$$
d _ {i} ^ {\prime} = f _ {i} ^ {d o n e} (h _ {i} ^ {1}).
$$

Note that we predict the next state with a skip connection, because empirically, it's more efficient to predict the change of the state rather than the state itself.

Scaling model-based methods into real tasks can result in decreased performance, even if the model is relatively accurate. One reason is the compound modeling error when long model rollouts are used, and model error compound along the rollout trajectory, making the trajectory ultimately highly inaccurate. To reduce the negative effect of model error, we adopt a branched rollout scheme proposed in (Janner et al., 2019). In branched rollout, model rollout starts not from an initial state, but from a state that was randomly selected from the most recent environmental trajectory  $\tau$ . Additionally, the model rollout length is fixed to be  $T$ . This scheme is shown to be effective in reducing the negative influence of model error both theoretically and empirically. To deal with the bias of model trajectories, at each model rollout, we allow the algorithm to fall back to the real trajectory with probability  $1 - q_{0}$ , where  $q_{0}$  is a hyperparameter. We describe the detailed framework of model usage and experiment storage in Algorithm 1.

# 4.2 PROXIMAL POLICY OPTIMIZATION WITH EXTENDED VALUE FUNCTION

To optimize the policies, we need to adopt an algorithm that can exploit network structure, whilst remaining decentralized. Independent RL algorithms that observe only local state are fully decen

Algorithm 1: Decentralized Model-based Policy Optimization (DMPO) for MARL  
Input: hyperparameters: rollout length  $T$ , truncation radius  $\kappa$   
1: Initialize the model  $p_i^{\psi_i}$ , actor  $\pi_i^{\theta_i}$  and critic  $V_i^{\phi_i}$ .  
2: Initialize replay buffers  $\mathcal{D}_i^{env}$  and  $\mathcal{D}_i^{model}$ .  
3: for  $M$  iterations do  
4: Perform environmental rollout together, and each agent  $i$  collect trajectory information  $\tau_i$ .  
5: for  $i$  in  $N$  agents do  
6:  $D_i^{env} = D_i^{env} \cup \{\tau_i\}$ .  
7: Train  $p_i^{\psi_i}$  on  $D_i^{env}$ .  
8:  $D_i^{model} = \emptyset$ .  
9: for  $B$  inner iterations: do  
10: Generate a random number  $q \sim U(0,1)$ .  
11: if  $q > q_0$  then  
12:  $\mathcal{D}_i^{model} = \tau_i$ . {Fall back to real trajectory with probability  $1 - q_0$ .}  
13: else  
14: for  $R$  rollouts,  $s \in \tau$  do  
15: Perform  $T$ -step model rollout starting from  $s$  using policy  $p_{\psi_*}$ , append to  $D_*^{model}$ .  
16: for  $G$  steps,  $i = 0, \dots, n-1$  do  
17: Take a step along the gradient to update  $\pi_i^{\theta_i}$  and critic  $V_i^{\phi_i}$  on  $D_*^{model}$

tralized, but they often fail to learn an optimal policy. Centralized algorithms that utilize centralized critics often achieve better performance than decentralized algorithms, but they might not scale to large environments where communication costs are expensive.

We propose Proximal Policy Optimization with extended value function, which is defined as  $V_{i}(s_{N_{i}^{\kappa}}) = \mathbb{E}_{s_{N_{-i}^{\kappa}}\sim d_{\pi}}[\sum_{t = 0}^{\infty}r_{i}^{t}|s_{N_{i}^{\kappa}}^{0} = s_{N_{i}^{\kappa}}]$ ,  $i\in \mathcal{V}$ . The intuition behind extended value function comes from (Qu et al., 2020a), where truncated  $Q$ -function  $Q(s_{N_i^\kappa},a_{N_i^\kappa})$  is initially proposed. In 5.3, we prove that  $V_{i}(s_{N_{i}^{\kappa}})$  is a good approximation of  $V_{i}(s)$ , with a difference decreasing exponentially with  $\kappa$ .

To generate the objective for extended value function, or return  $R_{i}$ , we use reward-to-go technique. However, because model rollout is short, standard reward-to-go returns would get a biased estimation of  $V_{i}$ . To resolve this issue, we add the value estimation of the last state to the return. In this way, with a local trajectory  $\tau_{i} = \{(s_{i}^{t},a_{i}^{t},r_{i}^{t},(s^{\prime})_{i}^{t},d_{i}^{t},\log \pi_{i}^{t}),t = 0,1,\dots,T - 1\}$ , the objective of  $V_{i}^{t}(s_{N_{i}^{\kappa}})$  is

$$
R _ {i} ^ {t} = \sum_ {l = 0} ^ {T - t - 1} \gamma^ {l} r _ {i} ^ {t + l} + V _ {i} ^ {\phi_ {i}} \left[ \left(s ^ {\prime}\right) _ {N _ {i} ^ {\kappa}} ^ {T - 1} \right], \tag {5}
$$

and the loss of value function is defined as  $\mathcal{L}_i^{value} = \frac{1}{m}\sum_{m\in \mathcal{D}_i^{model}}\left[V_i^{\phi_i}(s_{N_i^\kappa}^m) - R_i^m\right]^2$ . In policy training, extended value functions  $V_{i}$  are reduced via communication to their  $\kappa$ -hop neighbors to generate an estimation of global value function,

$$
\tilde {V} _ {i} ^ {t} = \frac {1}{n} \sum_ {j \in N _ {i} ^ {\kappa}} \tilde {V} _ {j} ^ {t}, \tag {6}
$$

and advantages  $\hat{A}_i$  are computed on  $\tilde{V}_i$  with generalized advantage estimation (GAE) (Schulman et al., 2015) for policy gradient update. The surrogate loss function of a DMPO agent is defined as

$$
\mathcal {L} _ {i} ^ {\text {p o l i c y}} = \frac {1}{m} \sum_ {m \in \mathcal {D} _ {i} ^ {\text {m o d e l}}} \min  \left(\frac {\pi_ {i} ^ {\theta_ {i}} \left(a _ {i} ^ {t} \mid s _ {\bar {N} _ {i}} ^ {t}\right)}{\pi_ {i} ^ {\theta_ {i} ^ {k}} \left(a _ {i} ^ {t} \mid s _ {\bar {N} _ {i}} ^ {t}\right)} \hat {A} _ {i} \left(s _ {V _ {i} ^ {\kappa}}\right), g \left(\epsilon , \hat {A} _ {i} \left(s _ {V _ {i} ^ {\kappa}}\right)\right)\right), \tag {7}
$$

similar to PPO-Clip loss.

The communication of  $\kappa$  step might seem costly, yet information of  $N_{i}^{\kappa}$  is only used in the training phase. We argue that in the training phase, algorithms are less sensitive with latency than execution. Furthermore, since model-based learning can effectively increase sample efficiency, we might tolerate more communication.

# 5 THEORETICAL ANALYSIS

In this section, we analyze DMPO theoretically. In 5.2, we derive a bound between the true returns and the returns under a model  $\hat{p}$  in a networked system. In 5.3, we prove that extended value function  $V_{i}(s_{N_{i}^{c}})$  is a good approximation of  $V_{i}(s)$ , and with extended value function, the true policy gradient can also be approximated.

# 5.1 BACKGROUND: MONOTONIC MODEL-BASED POLICY OPTIMIZATION

Let  $\eta[\pi]$  denote the returns of the policy in the true environment,  $\hat{\eta}[\pi]$  denote the returns of the policy under the approximated model. To analyze the difference between  $\eta[\pi]$  and  $\hat{\eta}[\pi]$ , we need to construct a bound

$$
\eta^ {p} [ \pi ] \geq \hat {\eta} ^ {\tilde {p}} [ \pi ] - C (p, \hat {p}, \pi , \pi_ {D}), \tag {8}
$$

where  $C$  is a non-negative function, and  $\pi_D$  is the data-collecting policy. According to equation 8, if every policy update ensures an improvement of  $\hat{\eta}[\pi]$  by at least  $C$ ,  $\eta[\pi]$  will improve monotonically. This inequality was first presented in single agent domain (Janner et al., 2019). In this work, we extend this to the multi-agent networked system, aiming to achieve monotonic team reward improvement.

In this work, we let  $\pi$  indicate a collective policy  $\pi = [\pi_1,\dots,\pi_n]$ , and the model  $\hat{p}$  be an INS  $\hat{p}(s'|s,a) = \prod_{i=1}^{n} \hat{p}_i(s_i'|s_{\bar{N}_i},a_i)$  that approximating the true MDP. In DMPO, each agent learns a localized model  $\hat{\pi}_i$ , policy  $\pi_i(|s_{N_k})$ , critic  $V_i(s_{N_i^k})$ , making it never a trivial extension. We give the detailed analysis in 5.2.

# 5.2 ANALYSIS OF RETURNS BOUND

In model-based learning, different rollout schemes can be chosen. The vanilla rollout assumes that models are used in an infinite horizon. The branched rollout performs a rollout from a state sampled by a state distribution of previous policy  $\pi_{D}$ , and runs  $T$  steps in  $\hat{\pi}$  according to  $\pi$ . Based on different rollout schemes, we can construct two lower bounds. Under vanilla rollout, real return and model return can be bounded by model error and policy divergence. Formal results are presented in Theorem 1. The detailed proof is deferred to Appendix C.

Theorem 1. Consider an independent networked system. Denote local model errors as  $\epsilon_{m_i} = \max_{s_{\bar{N}_i},a_i}D_{TV}[p_i(s_i'|s_{\bar{N}_i},a_i)\| \hat{p}_i(s_i'|s_{\bar{N}_i},a_i)]$ , and divergences between the data-collecting policy and evaluated policy as  $\epsilon_{\pi_i} = \max_{s_{\bar{N}_i}}D_{TV}[\pi_D(a_i|s_{\bar{N}_i})\| \pi (a_i|s_{\bar{N}_i})]$ . Assume the upper bound of rewards of all agents is  $r_{\mathrm{max}}$ . Let  $\eta^p [\pi_1,\dots,\pi_n]$  denote the real returns in the environment. Also, let  $\eta^{\hat{p}}[\pi_1,\dots,\pi_n]$  denote the returns estimated in the model trajectories, and the states and actions are collected with  $\pi_D$ . Then we have:

$$
| \eta^ {p} [ \pi_ {1}, \dots , \pi_ {n} ] - \eta^ {\hat {p}} [ \pi_ {1}, \dots , \pi_ {n} ] | \leq \frac {2 r _ {\max}}{1 - \gamma} \sum_ {i = 1} ^ {n} \left[ \frac {\epsilon_ {\pi_ {i}}}{n} + (\epsilon_ {m _ {i}} + 2 \epsilon_ {\pi_ {i}}) \cdot \sum_ {k = 0} ^ {\infty} \gamma^ {k + 1} \frac {| \bar {N _ {i}} ^ {k} |}{n} \right].
$$

Intuitively, the term  $\sum_{k=0}^{\infty} \gamma^{k+1} \frac{|\bar{N}_i^k|}{n}$  would be in the same magnitude as  $\frac{1}{1-\gamma}$ , which might be huge given the choice of  $\gamma$ , making the bound too loose to be effective. To make tighter the discrepancy bound in Theorem 1, we adopt the branched rollout scheme. The branched rollout enables a effective combination of model-based and model-free rollouts. For each rollout, we begin from a state sample from  $d_{\pi_D}$ , and run  $T$  steps in each localized  $\hat{\pi}_i$ . When branched rollout is applied in an INS, Theorem 2 gives the returns bound.

Theorem 2. Consider an independent networked system. Denote local model errors as  $\epsilon_{m_i} = \max_{s_{\bar{N}_i},a_i}D_{TV}[p_i(s_i'|s_{\bar{N}_i},a_i)\| \hat{p}_i(s_i'|s_{\bar{N}_i},a_i)]$ , and divergences between the data-collecting policy and evaluated policy as  $\epsilon_{\pi_i} = \max_{s_{\bar{N}_i}}D_{TV}[\pi_D(a_i|s_{\bar{N}_i})\| \pi (a_i|s_{\bar{N}_i})]$ . Assume the upper bound of rewards of all agents is  $r_{\mathrm{max}}$ . Let  $\eta^p [\pi_1,\dots,\pi_n]$  denote the real returns in the environment. Also, let  $\eta^{branch}[\pi_1,\dots,\pi_n]$  denote the returns estimated via  $T$ -step branched rollout scheme. Then we have:

$$
| \eta^ {p} [ \pi_ {1}, \dots , \pi_ {n} ] - \eta^ {b r a n c h} [ \pi_ {1}, \dots , \pi_ {n} ] | \leq \frac {2 r _ {\operatorname* {m a x}}}{1 - \gamma} \sum_ {i = 1} ^ {n} \left[ \epsilon_ {m _ {i}} \cdot \big (\sum_ {k = 0} ^ {T - 1} \gamma^ {k + 1} \frac {| \bar {N} _ {i} ^ {k} |}{n} \big) + \epsilon_ {\pi_ {i}} \cdot \big (\sum_ {k = T} ^ {\infty} \gamma^ {k + 1} \frac {| \bar {N} _ {i} ^ {k} |}{n} \big) \right]
$$

Comparing the results in Theorem 1 and 2, we can see that branched rollout scheme reduced the coefficient before  $\epsilon_{m_i}$  from  $\sum_{k=0}^{\infty} \gamma^{k+1} \frac{|\bar{N}_i^k|}{n} \leq \frac{\gamma}{1-\gamma}$  to  $\sum_{k=0}^{T-1} \gamma^{k+1} \frac{|\bar{N}_i^k|}{n} \leq \sum_{k=0}^{T-1} \gamma^{k+1} = \frac{\gamma(1-\gamma^T)}{1-\gamma}$ . This reduction explains that empirically, branched rollout brings better asymptotic performance. Also, if we set  $T = 0$ , this bound turns into a model-free bound. This indicates that when  $\epsilon_{m_i}$  is lower than  $\epsilon_{\pi_i}$  allowed by our algorithm, a model might increase the performance.

In reality, not every system satisfies the definition of INS. Yet we can generalize Theorem 2 into a  $\xi$ -dependent system.

Corollary 1. Consider an  $\xi$ -dependent networked system. Denote local model errors as  $\epsilon_{m_i} = \max_{s_{\bar{N}_i},a_i}D_{TV}[p_i(s_i'|s_{\bar{N}_i},a_i)\| \hat{p}_i(s_i'|s_{\bar{N}_i},a_i)]$ , and divergences between the data-collecting policy and evaluated policy as  $\epsilon_{\pi_i} = \max_{s_{\bar{N}_i}}D_{TV}[\pi_D(a_i|s_{\bar{N}_i})\| \pi (a_i|s_{\bar{N}_i})]$ . Assume the upper bound of rewards of all agents is  $r_{\mathrm{max}}$ . Let  $\eta^p [\pi_1,\dots,\pi_n]$  denote the real returns in the environment. Also, let  $\eta^{branch}[\pi_1,\dots,\pi_n]$  denote the returns estimated via  $T$ -step branched rollout scheme. Then we have:

$$
\begin{array}{l} \left| \eta^ {p} \bigl [ \pi_ {1}, \dots , \pi_ {n} \bigr ] - \eta^ {b r a n c h} \bigl [ \pi_ {1}, \dots , \pi_ {n} \bigr ] \right| \\ \leq \frac {2 r _ {\operatorname* {m a x}} \gamma}{(1 - \gamma) ^ {2}} \xi + \frac {2 r _ {\operatorname* {m a x}}}{1 - \gamma} \sum_ {i = 1} ^ {n} \left[ \epsilon_ {m _ {i}} \cdot \left(\sum_ {k = 0} ^ {T - 1} \gamma^ {k + 1} \frac {\left| \bar {N} _ {i} ^ {k} \right|}{n}\right) + \epsilon_ {\pi_ {i}} \cdot \left(\sum_ {k = T} ^ {\infty} \gamma^ {k + 1} \frac {\left| \bar {N} _ {i} ^ {k} \right|}{n}\right) \right] \\ \end{array}
$$

The proof can also be found in Appendix C. Compared to Theorem 2, Corollary 1 is more general, as it is applicable to the multi-agent systems that are not fully independent. Intuitively, if a networked system seems nearly independent, local models will be effective enough. The bound indicates that when the policy in optimized in a trust region where  $D(\pi, \pi_D) \leq \epsilon_{\pi_i}$ , the bound would also be restricted, making monotonic update more achievable.

# 5.3 EXTENDED VALUE FUNCTION

In this section, we analyze the effect of extended value function. The idea of extended value function  $V_{i}(s_{N_{i}^{\kappa}})$  comes from truncated  $Q$ -function  $Q_{i}(s_{N_{i}^{\kappa}},a_{N_{i}^{\kappa}})$  proposed in (Qu et al., 2020a). We prove that extended value function is an approximation of the real value function. The detailed proof of Theorem 3 is deferred to Appendix C.

Theorem 3. Define  $V_{i}(s_{N_{i}^{\kappa}}) = \mathbb{E}_{s_{N_{-i}^{\kappa}}\sim d_{\pi}}[\sum_{t = 0}^{\infty}r_{i}^{t}|s_{N_{i}^{\kappa}}^{0} = s_{N_{i}^{\kappa}}]$ , and  $V_{i}(s) = \mathbb{E}[\sum_{t = 0}^{\infty}r_{i}^{t}|s^{0} = s]$ , then:

$$
| V _ {i} (s) - V _ {i} (s _ {N _ {i} ^ {\kappa}}) | \leq \frac {r _ {\mathrm {m a x}}}{1 - \gamma} \gamma^ {\kappa}.
$$

From Theorem 3, it is straightforward that the global value function can be approximated with the average of all extended value functions:  $|V(s) - \frac{1}{n}\sum_{i = 1}^{n}V_{i}(s_{N_{i}^{\kappa}})|\leq \frac{r_{\max}}{1 - \gamma}\gamma^{\kappa}$ . In PPO, value functions are used for calculating advantages  $\hat{A}^{(t)} = r^{(t)} + \gamma V(s^{(t + 1)}) - V(s^{(t)})$ , and we have proven that  $V(s)$  can be estimated with the average of extended value functions  $\frac{1}{n}\sum_{i = 1}^{n}V_{i}(s_{N_{i}^{\kappa}})$ . In practice, an agent might not get the value function of distant agents. However, we can prove that  $\tilde{V}_{i} = \frac{1}{n}\sum_{j\in N_{i}^{\kappa}}V_{j}(s_{N_{j}^{\kappa}})$  is already very accurate for calculating the policy gradient for agent  $i$ . Theorem 4 justifies that the policy gradients computed based on the sum of the nearby extended value functions is a close approximation of true policy gradients.

Theorem 4. Let  $\hat{A}_t = r^{(t)} + \gamma V(s^{(t + 1)}) - V(s^{(t)})$  be the TD residual, and  $g_{i} = \mathbb{E}[\hat{A}\nabla_{\theta_i}\log \pi_i(a|s)]$  be the policy gradient. If  $\tilde{A}_t$  and  $\tilde{g}_i$  are the TD residual and policy gradient when value function  $V(s)$  is replaced by  $\tilde{V}_i(s) = \frac{1}{n}\sum_{j\in N_i^\kappa}V_j(s_{N_i^\kappa})$ , we have:

$$
\left| g _ {i} - \tilde {g} _ {i} \right| \leq \frac {\gamma^ {\kappa - 1}}{1 - \gamma} \left[ 1 - \left(1 - \gamma^ {2}\right) \frac {N _ {i} ^ {\kappa}}{n} \right] r _ {\max } g _ {\max },
$$

where  $r_{\mathrm{max}}$  and  $g_{\mathrm{max}}$  denote the upper bound of the absolute value of reward and gradient, respectively.

# 6 EXPERIMENTS

# 6.1 ENVIRONMENTS

We test our algorithm in four environments, namely Figure Eight, Ring Attenuation (Wu et al., 2017a), CACC Catchup, and CACC Slowdown (Chu et al., 2020). Detailed description and visualization of these environments is deferred to Appendix A.

Cooperative Adaptive Cruise Control The objective of CACC is to adaptively coordinate a plateau of 8 vehicles to minimize the car-following headway and speed perturbations based on real-time vehicle-to-vehicle communication. CACC consists of two scenarios: Catch-up and Slow-down. In CACC Catch-up, vehicles need to catch up to the first car. In CACC Slow-down, every vehicle is faster than the optimal speed, and they need to slow down without causing any collision. The agents receive a negative reward if the headway or the speed is not optimal. Also, whenever a collision happens, a huge negative reward of -1000 is given.

Flow environments This task consists of Figure Eight and Ring Attenuation. The objective of these environments is letting the automated vehicles achieve a target average speed inside the road network while avoiding collisions. The state of each vehicle is its velocity and position, and the action is the acceleration of itself. In Ring Attenuation, the objective is to achieve a high speed, while avoiding stop-and-go loops. Vehicles are rewarded with their speed, but also punished for their accelerations. In the perspective of a networked system, we assume that the vehicles are connected with the preceding and succeeding vehicle, thus resulting in a loop-structured graph.

# 6.2 BASELINES

We describe the following algorithms for performance comparison:

- CPPO: Centralized PPO learns a centralized critic  $V_{i}(s)$ . This baseline aims to analyze the performance when  $\kappa$  is set to be arbitrarily huge, and is used in (Vinitsky et al., 2018) as a benchmark algorithm for networked system control.  
- IC3Net (Singh et al., 2018): A communication-based multi-agent RL algorithm. The agents maintain their local hidden states with a LSTM kernel, and actively determines the communication target. Compared with DPPO, IC3Net uses hidden state and continuous communication, whereas DPPO agents directly observe the states of their neighbors.  
- DPPO: Decentralized PPO learns an independent actor and critic for each agent. We implement it by using neighbor's state for extended value estimation.  
- DMPO (our method): DMPO is a decentralized and model-based algorithm based on DPPO. On top of it, we use decentralized graph convolutional kernel as predictive model.

# 6.3 RESULTS

Figure 2 shows the episode reward v.s. number of training samples curves of the algorithms. We address that in CACC environments, DMPO uses decentralized SAC as base algorithm. Similar with DPPO, decentralized SAC uses extended  $Q$ -function  $Q_{i}(s_{N_{i}^{\kappa}},a_{N_{i}^{\kappa}})$  for its policy gradient. From the results, we conclude that our algorithm matches the asymptotic performance of model-free methods. It also learns the policy faster, resulting in increased sample efficiency.

The comparison between DMPO and DPPO can be viewed as an ablation study of model usage. In figure eight, DMPO increases sample efficiency at the beginning, but as the task becomes difficult, the sample efficiency of our method decreased. In a relatively easy task, ring attenuation, our method increased sample efficiency massively, compared with its model-free counterpart.

The comparison between the asymptotic performance of CPPO and DMPO or DPPO can be viewed as an ablation study of extended value function. From the result in four environments, we observe that the asymptotic performance of CPPO does not exceed that of the algorithms that uses extended value function. In this way, we conclude that by using extended value function, a centralized algorithm can be decomposed into decentralized algorithm, but the performance would not drop significantly.

![](images/0e5ea567df150974f8b545a72f1447292570a26cea59f185d0ce5f6498bdc984.jpg)  
(a) Figure Eight

![](images/adc1acfe188b9a5a7fc54766d129552b97d5dc38261d17d9dded394201feb5b1.jpg)  
Figure 3 shows the accuracy of our model in predicting the reward and state during training. The error is defined as the ratio of MSE loss to variance. From the figures, we conclude that neighborhood information is accurate enough for a model to predict the next state in these environments. However, in CACC Slow-down, local models might fail to learn the reward. We observe that the errors may increase as the agents explore new regions in the state space.

![](images/f9045b80e958052e7b1f8e7edf5d7a257a94a7f13bd07da33e8bc0ad272b9b83.jpg)  
(c) CACC Catch-up

![](images/ded77ec2cda8730828bc50cc97ccc50354cdb8b4bc086bbffc263a2b0cc83385.jpg)  
(b) Ring Attenuation  
(d) CACC Slow-down

![](images/1de244d14c37b9b999fecab4e9e502016dfe5ea4a10e6d3a687c3509069c641e.jpg)  
Figure 2: Training curves on multi-agent environments. Solid curves depict the mean of five trails, and shaded region correspond to standard deviation.  
(a) State Error

![](images/93a418afe215ef9f74b17d879464610c0367079dd7b35b65b58f56b2db348bbb.jpg)  
Figure 3: Figures of state and reward error. Both state error and reward error  $< 10\%$  in every environment.  
(b) Reward Error

# 7 CONCLUSIONS

In this paper, we propose algorithm DMPO, a model-based and decentralized multi-agent RL algorithm. We then give a theoretical analysis on the algorithm to analyze its performance discrepancy, compared with a model-free algorithm. By experiments in several tasks in networked systems, we show that although our algorithm is decentralized and model-based, it matches the asymptotic performance of some state-of-art multi-agent algorithms. From the results, we also conclude that using extended value function instead of centralized value function did not sacrifice performance massively, yet it makes our algorithm scalable.

# REFERENCES

Masako Bando, Katsuya Hasebe, Akihiro Nakayama, Akihiro Shibata, and Yuki Sugiyama. Dynamical model of traffic congestion and numerical simulation. Physical review E, 51(2):1035, 1995.  
Eugenio Bargiacchi, Timothy Verstraeten, and Diederik M. Roijers. Cooperative prioritized sweeping. In International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2021), pp. 160-168. IFAAMAS, 2021.  
Bruno Bouzy and Marc Métivier. Multi-agent model-based reinforcement learning experiments in the pursuit evasion game. 2007.  
Ronen I Brafman and Moshe Tennenholtz. A near-optimal polynomial time algorithm for learning in certain classes of stochastic games. Artificial Intelligence, 121(1-2):31-47, 2000.  
Tianshu Chu, Sandeep Chinchali, and Sachin Katti. Multi-agent reinforcement learning for networked system control. In International Conference on Learning Representations (ICLR), 2020. URL https://openreview.net/forum?id=Syx7A3NFvH.  
Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
Carlos Guestrin, Daphne Koller, and Ronald Parr. Multiagent planning with factored mdps. In Advances in Neural Information Processing Systems (NeurIPS), volume 1, pp. 1523-1530, 2001.  
Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2019.  
I Ge Jin and Gabor Orosz. Dynamics of connected vehicle systems with delayed acceleration feedback. Transportation Research Part C: Emerging Technologies, 46:46-64, 2014.  
Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237-285, 1996.  
Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, and Tengyu Ma. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees. In International Conference on Learning Representations (ICLR), 2019.  
Guannan Qu, Yiheng Lin, Adam Wierman, and Na Li. Scalable multi-agent reinforcement learning for networked systems with average reward. Advances in Neural Information Processing Systems (NeurIPS), 33, 2020a.  
Guannan Qu, Adam Wierman, and Na Li. Scalable reinforcement learning of localized policies for multi-agent networked systems. In Learning for Dynamics and Control (L4DC), pp. 256-266. PMLR, 2020b.  
Tabish Rashid, Mikayel Samvelyan, Christian Schroeder, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. In International Conference on Machine Learning, pp. 4295-4304. PMLR, 2018.  
Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604-609, 2020.  
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Thiago D Simao and Matthijs TJ Spaan. Safe policy improvement with baseline bootstrapping in factored environments. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), volume 33, pp. 4967-4974, 2019.  
Amanpreet Singh, Tushar Jain, and Sainbayar Sukhbaatar. Learning when to communicate at scale in multiagent cooperative and competitive tasks. arXiv preprint arXiv:1812.09755, 2018.  
Wen Sun, Geoffrey J Gordon, Byron Boots, and J Bagnell. Dual policy iteration. Advances in Neural Information Processing Systems (NeurIPS), 31:7059-7069, 2018.  
Eugene Vinitsky, Aboudy Kreidieh, Luc Le Flem, Nishant Kheterpal, Kathy Jang, Cathy Wu, Fangyu Wu, Richard Liaw, Eric Liang, and Alexandre M Bayen. Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on robot learning, pp. 399-409. PMLR, 2018.  
Cathy Wu, Aboudy Kreidieh, Kanaad Parvate, Eugene Vinitsky, and Alexandre M Bayen. Flow: Architecture and benchmarking for reinforcement learning in traffic control. arXiv preprint arXiv:1710.05465, 10, 2017a.  
Cathy Wu, Aboudy Kreidieh, Eugene Vinitsky, and Alexandre M Bayen. Emergent behaviors in mixed-autonomy traffic. In Conference on Robot Learning, pp. 398-407. PMLR, 2017b.  
Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar. Fully decentralized multi-agent reinforcement learning with networked agents. In International Conference on Machine Learning (ICML), pp. 5872-5881, 2018.  
Kaiqing Zhang, Zhuoran Yang, and Tamer Başar. Multi-agent reinforcement learning: A selective overview of theories and algorithms, 2019.  
Kaiqing Zhang, Sham Kakade, Tamer Basar, and Lin Yang. Model-based multi-agent rl in zero-sum markov games with near-optimal sample complexity. Advances in Neural Information Processing Systems (NeurIPS), 33, 2020.
