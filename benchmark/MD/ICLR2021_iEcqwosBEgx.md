# NOVEL POLICY SEEKING WITH CONSTRAINED OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We address the problem of seeking novel policies in reinforcement learning tasks. Instead of following the multi-objective framework commonly used in existing methods, we propose to rethink the problem under a novel perspective of constrained optimization. We at first introduce a new metric to evaluate the difference between policies, and then design two practical novel policy seeking methods following the new perspective, namely the Constrained Task Novel Bisector (CTNB), and the Interior Policy Differentiation (IPD), corresponding to the feasible direction method and the interior point method commonly known in the constrained optimization literature. Experimental comparisons on the MuJuCo control suite show our methods can achieve substantial improvements over previous novelty-seeking methods in terms of both the novelty of policies and their performances in the primal task.

# 1 INTRODUCTION

In the paradigm of Reinforcement Learning, an agent interacts with the environment to learn a policy that can maximize a certain form of cumulative rewards (Sutton & Barto, 1998). Modeling the policy function with a Deep Neural Network, the policy gradient method can be applied to optimize current policy (Sutton et al., 2000). However, direct optimization with respect to the reward function is prone to get stuck in sub-optimal solutions and therefore hinders the policy optimization (Liepins & Vose, 1991; Lehman & Stanley, 2011; Plappert et al., 2018). Consequently, an appropriate exploration strategy is crucial for the success of policy learning (Auer, 2002; Bellemare et al., 2016; Houthooft et al., 2016; Tang et al., 2017; Ostrovski et al., 2017; Tessler et al., 2019; Ciosek et al., 2019).

Recently many works have shown that incorporating curiosity in the policy learning leads to better exploration strategies (Pathak et al., 2017; Burda et al., 2018a;b; Liu et al., 2019). In these works, visiting a previous unseen or infrequent state is assigned with an extra curiosity bonus reward. Different from those curiosity-driven methods which focus on the discovery of new states within the learning procedure of a repeated single policy, another direction Novel Policy Seeking (Lehman & Stanley, 2011; Zhang et al., 2019; Pugh et al., 2016) focuses on learning different policies with diverse or the so-called novel behaviors to solve the primal task. In the process of novel policy seeking, policies in new iterations are usually encouraged to be different from previous policies. Therefore novel policy seeking can be viewed as an extrinsic curiosity-driven method at the level of policies, as well as an exploration strategy for a population of agents. Besides encouraging exploration (Eysenbach et al., 2018; Gangwani et al., 2018; Liu et al., 2017), novel policy seeking is also related to policy ensemble (Osband et al., 2018; 2016; Florensa et al., 2017) and evolution strategies (ES) (Salimans et al., 2017; Conti et al., 2018).

In order to generate novel policies, previous work often defines a heuristic metric for novelty estimation, e.g., differences of state distributions estimated by neural networks are used in (Zhang et al., 2019), and tries to solve the problem under the formulation of multi-objective optimization. However, most of these metrics suffer from the difficulty when dealing with episodic novelty reward, i.e., the difficulty of episodic credit assignment (Sutton et al., 1998), thus their effectiveness in learning novel policies is limited. Moreover, the difficulty of balancing different objectives impedes the agent to find a well-performing policy for the primal task, as shown by Fig. 1 which compares the policy gradients of three cases, namely the one without novel policy seeking, novelty seeking with multi-objective optimization and novelty seeking with constrained optimization methods, respectively.

![](images/660f05e75c8f2ff7d002415766af727cf7662eda5447890c25df63e1097bf69e.jpg)  
Figure 1: A comparison between the standard policy gradient method without novelty seeking (left), multi-objective optimization method (mid) and our constrained optimization approach (right) for novel policy seeking. The standard policy gradient method do not try actively to find novel solutions. The multi-objective optimization method may impede the learning procedure when the novelty gradient is being applied all the time (Zhang et al., 2019), e.g., a random initialized policy will be penalized from getting closer to previous policy due to the conflict of gradients, which limits the learning efficiency and the final performance. On the contrary, the novelty gradient of our constrained optimization approach will only be considered within a certain region to keep the policy being optimized away from highly similar solutions. Such an approach is more flexible and includes the multi-objective optimization method as its special case.

![](images/2b7ad370d5ccbfb0dc41ad8aeaefec6c10ff7a4d34a98d6401b5adde159d4172.jpg)

![](images/3eed26beed621fa143cc5217add9b9ab38b959529c4d625ab735b764f7d4c48e.jpg)

In this work, we intend to take into consideration both the novelty of learned policies as well as their performances in terms of the primal task when addressing the problem of novel policy seeking. To achieve this goal, we propose to seek novel policies with a constrained optimization formulation. Two specific algorithms under such a formulation are designed to seek novel policies while keeping their performances in the primal task, avoiding excessive novelty seeking. As a consequence, with these two algorithms, the performances of our learned novel policies can be guaranteed and even further improved.

Our contributions can be summarized in three-folds. Firstly, we introduce a new metric to compute the difference between policies with instant feedback at every timestep; Secondly, we propose a constrained optimization formulation for novel policy seeking and design two practical algorithms resembling two approaches in constrained optimization literature; Thirdly, we evaluate our proposed algorithms on the MuJoCo locomotion environments, showing the advantages of these constrained optimization novelty-seeking methods which can generate a series of diverse and well-performing policies over previous multi-objective novelty seeking methods.

# 2 RELATED WORK

Intrinsic motivation methods In previous work, different approaches are proposed to provide intrinsic motivation or intrinsic reward as a supplementary to the primal task reward for better exploration (Houthooft et al., 2016; Pathak et al., 2017; Burda et al., 2018a,b; Liu et al., 2019). All those approaches leverage the weighted sum of two rewards, the primal rewards provided by environments, and intrinsic rewards that provided by different heuristics. On the other hand, the work of DIAYN and DADS (Eysenbach et al., 2018; Sharma et al., 2019) learn diverse skills without extrinsic reward. Those approaches focus on decomposing diverse skills of a single policy, while our work focuses on learning diverse behaviors among a batch of policies for the same task.

Diverse policy seeking methods The work of Such et al. shows that different RL algorithms may converge to different policies for the same task (Such et al., 2018). On the contrary, we are interested in how to learn different policies through a single learning algorithm with the capability of avoiding local optimum. The work of Pugh et al. establishes a standard framework for understanding and comparing different approaches to search for quality diversity (QD) (Pugh et al., 2016). Conti et al. proposes a solution which avoids local optima as well as achieves higher performance by adding novelty search and QD to evolution strategies (Conti et al., 2018). The Task-Novelty Bisector (TNB) (Zhang et al., 2019) aims to solve novelty seeking problem by jointly optimize the extrinsic rewards and novelty rewards defined by an auto-encoder. In this work, one of the two proposed methods is closely related to TNB, but is adapted to the constrained optimization formulation.

Constrained Markov Decision Process The Constrained Markov Decision Process (CMDP) (Altman, 1999) considers the situation where an agent interacts with the environment under certain constraints. Formally, the CMDP can be defined as a tuple  $(S, \mathcal{A}, \gamma, r, c, C, P, s_0)$ , where  $S$  and  $\mathcal{A}$  are the state and action space;  $\gamma \in [0,1)$  is a discount factor;  $r: S \times \mathcal{A} \times S \to \mathbb{R}$  and  $c: S \times \mathcal{A} \times S \to \mathbb{R}$  denote the reward function and cost function;  $C \in \mathbb{R}^+$  is the upper bound of permitted expected cumulative cost;  $P(\cdot | s, a): S \times \mathcal{A} \to S$  denotes the transition dynamics, and  $s_0$  is the initial state. Denote the Markovian policy class as  $\Pi$ , where  $\Pi = \{\pi: S \times \mathcal{A} \to [0,1], \sum_{a} \pi(a|\pi) = 1\}$ . The learning objective of a policy for CMDP is to find a  $\pi^* \in \Pi$ , such that

$$
\pi^ {*} = \max  _ {\pi \in \Pi} \mathbb {E} _ {\tau \sim \pi , s ^ {\prime} \sim P} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r (s, a, s ^ {\prime}) \right], \quad \text {s . t .} \quad \mathbb {E} _ {\tau \sim \pi , s ^ {\prime} \sim P} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} c (s, a, s ^ {\prime}) \right] \leq C, \tag {1}
$$

where  $\tau$  indicates a trajectory  $(s_0, a_0, s_1, \ldots)$  and  $\tau \sim \pi$  represents the distribution over trajectories following policy  $\pi$ :  $a_t \sim \pi(\cdot | s_t)$ ,  $s_{t+1} \sim P(\cdot | s_t, a_t)$ ;  $t = 0, 1, 2, \ldots$ . Previous literature provides several approaches to solve CMDP (Achiam et al., 2017; Chow et al., 2018; Ray et al., 2019), and in this work we include the CPO (Achiam et al., 2017) as baseline according to Ray et al. (2019).

# 3 METHODOLOGY

In Sec.3.1, we start with defining a metric space that measures the difference between policies, which is the fundamental ingredient for the methods introduced later. In Sec.3.2, we develop a practical estimation method for this metric. Sec.3.3 describes the formulation of constrained optimization on novel policy seeking. The implementations of two practical algorithms are introduced in Sec.3.4.

We denote the policies as  $\{\pi_{\theta_i};\theta_i\in \Theta ,i = 1,2,\ldots \}$ , wherein  $\theta_{i}$  represents parameters of the  $i$ -th policy,  $\Theta$  denotes the whole parameter space. In this work, we focus on improving the behavior diversity of policies from PPO (Schulman et al., 2017), thus we use  $\Theta$  to represent  $\Theta_{\mathrm{PPO}}$  in this paper. It is worth noting that the proposed methods can be easily extended to other RL algorithms (Schulman et al., 2015; Lillicrap et al., 2015; Fujimoto et al., 2018; Haarnoja et al., 2018). To simplify the notation, we omit  $\pi$  and denote a policy  $\pi_{\theta_i}$  as  $\theta_{i}$  unless stated otherwise.

# 3.1 MEASURING THE DIFFERENCE BETWEEN POLICIES

In this work, we use the Wasserstein metric  $W_{p}$  (Ruschendorf, 1985; Villani, 2008; Arjovsky et al., 2017) to measure the distance between policies. Concretely, in this work we consider the Gaussian-parameterized policies, where the  $W_{p}$  over two policies can be written in the closed form  $W_{2}^{2}(\mathcal{N}(m_{1},\Sigma_{1}),\mathcal{N}(m_{2},\Sigma_{2})) = ||m_{1} - m_{2}||^{2} + tr[\Sigma_{1} + \Sigma_{2} - 2(\Sigma_{1}^{1 / 2}\Sigma_{2}\Sigma_{1}^{1 / 2})^{1 / 2}]$  as  $p = 2$ , where  $m_{1},\Sigma_{1},m_{2},\Sigma_{2}$  are mean and covariance metrics of the two normal distributions. In the following of this paper, we use  $D_W$  to denote the  $W_{2}$  and it is worth noting that when the covariance matrix is identical, the trace term disappears and only the term involving the means remains, i.e.,  $D_W = |m_1 - m_2|$  for Dirac delta distributions located at points  $m_{1}$  and  $m_{2}$ . This diversity metric satisfies the three properties of a metric, namely identity, symmetry as well as triangle inequality.

Proposition 1 (Metric Space  $(\Theta, \overline{D}_W^q)$ ). The expectation of  $D_W(\cdot, \cdot)$  of two policies over any state distribution  $q(s)$ :

$$
\bar {D} _ {W} ^ {q} \left(\theta_ {i}, \theta_ {j}\right) := \mathbb {E} _ {s \sim q (s)} \left[ D _ {W} \left(\theta_ {i} (a | s), \theta_ {j} (a | s)\right) \right], \tag {2}
$$

is a metric on  $\Theta$ , thus  $(\Theta, \overline{D}_W^q)$  is a metric space.

The proof of Proposition 1 is straightforward. It is worth mentioning that Jensen Shannon divergence  $D_{JS}$  or Total Variance Distance  $\bar{D}_{TV}$  (Endres & Schindelin, 2003; Fuglede & Topsoe, 2004; Schulman et al., 2015) can also be applied as alternative metric spaces, we choose  $D_W$  in our work for that the Wasserstein metric better preserves the continuity (Arjovsky et al., 2017).

On top of the metric space  $(\Theta, \overline{D}_W^q)$ , we can then compute the novelty of a policy as follows.

Definition 1 (Novelty of Policy). Given a reference policy set  $\Theta_{\text{ref}}$  such that  $\Theta_{\text{ref}} = \{\theta_i^{\text{ref}}, i = 1, 2, \ldots\}$ ,  $\Theta_{\text{ref}} \subset \Theta$ , the novelty  $U(\theta | \Theta_{\text{ref}})$  of policy  $\theta$  is the minimal difference between  $\theta$  and all policies in the reference policy set, i.e.,

$$
\mathrm {U} (\theta | \Theta_ {r e f}) := \min  _ {\theta_ {j} \in \Theta_ {r e f}} \bar {D} _ {W} ^ {q} (\theta , \theta_ {j}). \tag {3}
$$

Consequently, to encourage the discovery of novel policies discovery, typical novelty-seeking methods tend to directly maximize the novelty of a new policy, i.e.,  $\max_{\theta} \mathrm{U}(\theta | \Theta_{ref})$ , where the  $\Theta_{ref}$  includes all existing policies.

# 3.2 ESTIMATION OF  $\overline{D}_W^q (\theta_i,\theta_j)$  AND THE SELECTION OF  $q(s)$

In practice, the calculation of  $\overline{D}_W^q (\theta_i,\theta_j)$  is based on Monte Carlo estimation where we need to sample  $s$  from  $q(s)$ . Although in Eq.(2)  $q(s)$  can be selected simply as a uniform distribution over the state space, there remains two obstacles: first, in a finite state space we can get precise estimation after establishing ergodicity, but problem arises when facing continuous state spaces due to the difficulty of efficiently obtaining enough samples; second, when  $s$  is sampled from a uniform distribution  $q$ , we can only get sparse episodic reward instead of dense online reward which is more useful in learning. Therefore, we make an approximation here based on importance sampling.

Formally, we denote the domain of  $q(s)$  as  $S_q \subset S$  and assume  $q(s)$  to be a uniform distribution over  $S_q$ , without loss of generality in later analysis. Notice  $S_q$  is closely related to the algorithm being used in generating trajectories (Henderson et al., 2018). As we only care about the reachable regions of a certain algorithm (in this work, PPO), the domain  $S_q$  can be decomposed by  $S_q = \lim_{N \to \infty} \bigcup_{i=1}^{N} S_{\theta_i}$ , where  $S_{\theta_i}$  denotes all the possible states a policy  $\theta_i$  can visit given a starting state distribution.

In order to get online-reward, we estimate Eq.(2) with

$$
\overline {{D}} _ {W} ^ {q} \left(\theta_ {i}, \theta_ {j}\right) = \mathbb {E} _ {s \sim q (s)} \left[ D _ {W} \left(\theta_ {i} (a | s), \theta_ {j} (a | s)\right) \right] = \mathbb {E} _ {s \sim \rho_ {\theta_ {i}} (s)} \left[ \frac {q (s)}{\rho_ {\theta_ {i}} (s)} D _ {W} \left(\theta_ {i} (a | s), \theta_ {j} (a | s)\right) \right], \tag {4}
$$

where we use  $\rho_{\theta}(s)$  to denote the stationary state visitation frequency under policy  $\theta$ , i.e.,  $\rho_{\theta}(s) = P(s_0 = s|\theta) + P(s_1 = s|\theta) + \ldots + P(s_T = s|\theta)$  in finite horizon problems. We propose to use the averaged stationary visitation frequency as  $q(s)$ , e.g., for PPO,  $q(s) = \overline{\rho}(s) = \mathbb{E}_{\theta \sim \Theta_{\mathrm{PPO}}}[\rho_{\theta}(s)]$ . Clearly, choosing  $q(s) = \overline{\rho}(s)$  will be much better than choosing a uniform distribution as the importance weight will be closer to 1. Such an importance sampling process requires a necessary condition that  $\rho_{\theta_i}(s)$  and  $q(s)$  have the same domain, which can be guaranteed by applying a sufficient exploration noise on  $\theta$ .

Another difficulty lies in the estimation of  $\overline{\rho}(s)$ , which is always intractable given a limited number of trajectories. However, during training,  $\theta_{i}$  is a policy to be optimized and  $\theta_{j} \in \Theta_{\text{ref}}$  is a fixed reference policy. The error introduced by approximating the importance weight as 1 will get larger when  $\theta_{i}$  becomes more distinct from normal policies, at least in terms of the state visitation frequency. We may just regard increasing of the approximation error as the discovery of novel policies.

Proposition 2 (Unbiased Single Trajectory Estimation). The estimation of  $\rho_{\theta}(s)$  using a single trajectory  $\tau$  is unbiased.

The Proposition 2 follows the usual trick in RL that uses a single trajectory to estimate the stationary state visitation frequency. Given the definition of novelty and a practically unbiased sampling method, the next step is to develop an efficient learning algorithm.

# 3.3 CONSTRAINED OPTIMIZATION FORMULATION FOR NOVEL POLICY SEEKING

In the traditional RL paradigm, maximizing the expectation of cumulative rewards is commonly used as the objective. i.e.,  $\max_{\theta \in \Theta} \mathbb{E}_{\tau \sim \theta}[g]$ , where  $g = \sum_{t=0}^{\infty} \gamma^{t} r_{t}$  and  $\tau \sim \theta$  denotes a trajectory  $\tau$  sampled from the policy  $\theta$ .

To improve the diversity of different agents' behaviors, the learning objective must take both the reward from the primal task and the policy novelty into consideration. Previous approaches (Houthooft et al., 2016; Pathak et al., 2017; Burda et al., 2018a;b; Liu et al., 2019) often directly use the weighted sum of these two terms as the objective:

$$
\max  _ {\theta \in \Theta} \mathbb {E} _ {\tau \sim \theta} [ g _ {\text {t o t a l}} ] = \max  _ {\theta \in \Theta} \mathbb {E} _ {\tau \sim \theta} [ \alpha \cdot g _ {\text {t a s k}} + (1 - \alpha) \cdot g _ {\text {i n t}} ], \tag {5}
$$

where  $0 < \alpha < 1$  is a weight hyper-parameter,  $g_{\mathrm{task}}$  is the reward from the primary task, and  $g_{\mathrm{int}} = \sum_{t=0} \gamma^t r_{\mathrm{int},t}$  is the cumulative intrinsic reward of the intrinsic reward  $r_{\mathrm{int},t}$ . In our case, the intrinsic reward is the novelty reward  $r_{\mathrm{int}} = \min_{\theta_j \in \Theta_{ref}} \overline{D}_W^\overline{\rho}(\theta, \theta_j)$ . These methods can be

summarized as Weighted Sum Reward (WSR) methods (Zhang et al., 2019). Such an objective is sensitive to the selection of  $\alpha$  as well as the formulation of  $r_{\mathrm{int}}$ . For example, in our case formulating the novelty reward  $r_{\mathrm{int}}$  as  $\min_{\theta_j} \overline{D}_W^\bar{\rho}(\theta, \theta_j)$ ,  $\exp[\min_{\theta_j} \overline{D}_W^\bar{\rho}(\theta, \theta_j)]$  and  $-\exp[-\min_{\theta_j} \overline{D}_W^\bar{\rho}(\theta, \theta_j)]$  will lead to significantly different results as they determine the trade-offs in the two terms given  $\alpha$ . Besides, dilemma also arises in the selection of  $\alpha$ : while a large  $\alpha$  may undermine the contribution of intrinsic reward, a small  $\alpha$  could ignore the importance of the primal task, leading to the failure of an agent in solving the task.

To tackle such an issue, the crux is to deal with the conflict between different objectives. The work of Zhang et al. proposes the TNB, where the task reward is regarded as the dominant one while the novelty reward is regarded as subordinate Zhang et al. (2019). However, as TNB considers the novelty gradient all the time, it may hinder the learning process, e.g., Intuitively, well-performing policies should be more similar to each other than to random initialized policies. As a new random initialized policy is different enough from previous policies, considering the novelty gradient at beginning of training will result in a much slower learning process.

In order to tackle the above problems and adjust the extent of novelty in new policies, we propose to solve the novelty-seeking problem under the perspective of constrained optimization. The basic idea is as follows: while the task reward is considered as a learning objective, the novelty reward should be considered as a bonus instead of another objective, and should not impede the learning of the primal task. Fig. 1 illustrates how novelty gradients impede the learning of a policy: at the beginning of learning, a random initialized policy should in total learn to be more similar to a well-performing policy rather than be different. The seeking of novelty should not be taken into consideration all the time during learning. With such an insight, we change the multi-objective optimization problem in Eq.(5) into a constrained optimization problem as:

$$
\max  _ {\theta \in \Theta} f (\theta) = \mathbb {E} _ {\tau \sim \theta} \left[ g _ {\text {t a s k}} \right], \quad \text {s . t .} \quad g _ {t} (\theta) = \bar {r} _ {\text {i n t}, t} - r _ {0} \geq 0, \forall t = 1, 2, \dots , T, \tag {6}
$$

where  $r_0$  is a threshold indicating minimal permitted novelty, and  $\overline{r}_{\mathrm{int},t}$  denotes a moving average of  $r_{\mathrm{int},t}$ . As we need not force every single action of a new agent to be different from others. Instead, we care more about the long-term differences. Therefore, we use cumulative novelty terms as constraints. Moreover, the constraints can be flexibly applied after the first  $t_S$  timesteps (e.g.,  $t_S = 20$ ) for the consideration of similar starting sequences, so that the constraints can be written as  $g_t(\theta) \geq 0, \forall t = t_S, \dots, T$ .

# 3.4 PRACTICAL NOVEL POLICY SEEKING METHODS

We note here, WSR and TNB proposed in previous work (Zhang et al., 2019) can correspond to different approaches in constrained optimization problems, yet some important ingredients are missing. We improve TNB according to the Feasible Direction Method in constrained optimization and then propose the Interior Policy Differentiation (IPD) method according to the Interior Point Method in constrained optimization.

WSR: Penalty Method The Penalty Method considers the constraints of Eq.(6) by putting constraint  $g(\theta)$  into a penalty term, followed by solving the unconstrained problem

$$
\max  _ {\theta \in \Theta} f (\theta) + \frac {1 - \alpha}{\alpha} \min  \{g (\theta), 0 \}, \tag {7}
$$

in an iterative manner. The limit of the above unconstrained problem when  $\alpha \to 0$  then leads to the solution of the original constrained problem. As an approximation, WSR chooses a fixed weight  $\alpha$ , and uses the gradient of  $\nabla_{\theta}f + \frac{1 - \alpha}{\alpha}\nabla_{\theta}g$  instead of  $\nabla_{\theta}f + \frac{1 - \alpha}{\alpha}\nabla_{\theta}\min \{g(\theta),0\}$ , thus the final solution will intensely rely on the selection of  $\alpha$ .

TNB: Feasible Direction Method The Feasible Direction Method (FDM) (Ruszczyński, 1980; Herskovits, 1998) solves the constrained optimization problem by finding a direction  $\vec{p}$  where taking gradient upon will lead to increment of the objective function as well as constraints satisfaction, i.e.,  $\nabla_{\theta}f^{\mathrm{T}}\cdot \vec{p} >0$  , if  $g > 0$  and  $\nabla_{\theta}g^{\mathrm{T}}\cdot \vec{p} >0$  otherwise. The TNB proposes to use a revised bisector of gradients  $\nabla_{\theta}f$  and  $\nabla_{\theta}g$  as  $\vec{p}$

$$
\vec {p} = \left\{ \begin{array}{l l} \nabla_ {\theta} f + \frac {| \nabla_ {\theta} f |}{| \nabla_ {\theta} g |} \nabla_ {\theta} g \cdot \cos (\nabla_ {\theta} f, \nabla_ {\theta} g) & \text {i f} \cos (\nabla_ {\theta} f, \nabla_ {\theta} g) \leq 0 \\ \nabla_ {\theta} f + \frac {| \nabla_ {\theta} f |}{| \nabla_ {\theta} g |} \nabla_ {\theta} g & \text {i f} \cos (\nabla_ {\theta} f, \nabla_ {\theta} g) > 0 \end{array} \right. \tag {8}
$$

Clearly, Eq.(8) satisfies the constraints but it is more strict than it as the  $\nabla_{\theta}g$  term always exists during the optimization of TNB. Based on TNB, we provide a revised approach, named Constrained Task Novel Bisector (CTNB), which resembles better with FDM. Specifically, when  $g > 0$ , CTNB will not apply  $\nabla_{\theta}g$  on  $g$ . It is clear that TNB is a special case of CTNB when the novelty threshold  $r_0$  is set to infinity. We note that in both TNB and CTNB, the learning stride is fixed to be  $\frac{|\nabla_{\theta}f| + |\nabla_{\theta}g|}{2}$  and may lead to problem when  $\nabla_{\theta}f \to 0$ , where the final optimization result will rely heavily on the selection of  $g$ , i.e., the shape of  $g$  is crucial for the success of this approach.

IPD: Interior Point Method The Interior Point Method (Potra & Wright, 2000; Dantzig & Thapa, 2006) is another approach used to solve the constrained optimization problem. Thus here we solve Eq.(6) using the Interior Policy Differentiation (IPD), which can be regarded as an analogy of the Interior Point Method. In the vanilla Interior Point Method, the constrained optimization problem in Eq.(6) is solved by reforming it to an unconstrained form with an additional barrier term  $-\alpha \log g(\theta)$  in the objective as  $\max_{\theta \in \Theta} f(\theta) - \alpha \log g(\theta)$ , or more precisely in our problem with the formulation with Eq.(6) we have  $\max_{\theta \in \Theta} \mathbb{E}_{\tau \sim \theta}[g_{\mathrm{task}} - \sum_{t=0}^{T} \alpha \log (\overline{r}_{\mathrm{int},t} - r_0)]$ , where  $\alpha > 0$  is the barrier factor. Besides the log barrier term, there are other choices like  $\alpha \frac{1}{g(\theta)}$  can be used and the objective becomes  $\max_{\theta \in \Theta} f(\theta) + \alpha \frac{1}{g(\theta)}$ . As  $\alpha$  is small, the barrier term will introduce only minuscule influence on the objective. On the other hand, when  $\theta$  gets closer to the barrier, the objective will increase rapidly. The limits when  $\alpha \to 0$  then lead to the solution of Eq.(6). The convergence of such methods are provided in previous works Conn et al. (1997); Wright (2001).

However, directly applying IPM is computationally expensive and numerically unstable. In this work, we propose a simple yet novel heuristic method that resembles the idea of barrier methods: we implicitly apply such barrier terms by providing termination signals in interactions with the environments. Our method can be regarded as revising the primal task MDP into a new one in which the behaviors of agents must satisfy novelty constraints. Specifically, in the RL paradigm, the learning procedure of an agent is determined by the experiences collected during interactions with the environment and the sampling strategy used to filter experiences in the calculation of policy gradients. Since the learning process is based on sampled transitions, a more natural way can thus be used to perform the constrained optimization. We can simply bound the collected transitions in the feasible region by permitting previously trained  $M$  policies  $\theta_{i}\in \Theta_{\mathrm{ref}},i = 1,2,\ldots ,M$  sending termination signals during the training process of new agents. In other words, we implicitly bound the feasible region by terminating any new agent that steps outside it.

Consequently, during the training process, all valid samples we collected are inside the feasible region, which means these samples are less likely to appear in previously trained policies. At the end of the training, we then naturally obtain a new policy that has sufficient novelty. In this way, we no longer need to consider the trade-off between intrinsic and extrinsic rewards deliberately. The learning process of IPD is thus more robust and no longer suffers from objective inconsistency.

# 4 EXPERIMENTS

According to Proposition 2, the novelty reward  $r_{int}$  in Eq.(6) under our novelty metric can be unbiasedly approximated by  $r_{\mathrm{int}} = \min_{\theta_j \in \Theta_{ref}} \overline{D}_W^{\rho_\theta}(\theta(a|s_t), \theta_j(a_j|s_t))$ . We thus utilize this novelty metric directly throughout our experiments. We apply different novel policy seeking methods, namely WSR, TNB, CTNB, and IPD, to the backbone RL algorithm PPO (Schulman et al., 2017). The extension to other popular RL algorithms is straightforward. More implementation details are depicted in Appendix D. Experiments in the work of Henderson et al. show that one can simply change the random seeds before training to get policies that perform differently Henderson et al. (2018). Therefore, we also use PPO with varying random seeds as a baseline method for novel policy seeking. And we use the averaged differences between policies learned by this baseline as the default threshold in CTNB and IPD. Algorithm 1 and Algorithm 2 show the pseudo code of IPD and CTNB based on PPO, where the blue lines show the addition to the primal PPO algorithm.

# 4.1 THE MUJOCO ENVIRONMENT

We evaluate our proposed method on the OpenAI Gym based on the MuJoCo engine (Brockman et al., 2016; Todorov et al., 2012). Concretely, we test on three locomotion environments, the Hopper-v3

<table><tr><td>Algorithm 1 IPD</td><td>Algorithm 2 Constrained TNB</td></tr><tr><td>Input: 
(1) a behavior policy θold; 
(2) a set of previous policies {θj}, j = 1, 2, ..., M; 
(3) a novelty metric U(θ, {θj}|ρ) = U(θ, {θj}|τ) = minθj DτW(θ, θj); 
(4) a novelty threshold r0 and starting point ts 
Initialize θold; 
for iteration = 1, 2, ... do 
for t = 1, 2, ..., T do 
Step the environment by taking ac- 
tion att ~ θold and collect transi- 
tions; 
if U(θold, {θj}|τ) - r0 &lt; 0 AND 
t &gt; ts then 
| Break this episode; 
end 
end 
Update policy parameters based on 
sampled data;</td><td>Input: 
(1) to (4) same as Algo.1 
(5) a value network for cost Vc 
Initialize θold; 
for iteration = 1, 2, ... do 
for t = 1, 2, ..., T do 
| Step the environment by taking action att ~ θold 
and collect transitions; 
end 
Compute advantage of reward A_r,1, ..., A_r,T 
Compute advantage of cost A_c,1, ..., A_c,T 
Optimize surrogate loss related to reward L_rCLIP in 
PPO w.r.t. θ, with gradient gr = ∇θL_rCLIP 
Optimize surrogate loss related to cost L_cCLIP in PPO 
w.r.t. θ, with gradient gc = -∇θL_cCLIP 
if U(θold, {θj}|τ) - r0 &lt; 0 then 
| Calculate p according to Eq.(8) with gr and gc 
else 
| Calculate p with gr 
end 
Update policy parameters</td></tr><tr><td>end</td><td>end</td></tr></table>

![](images/fe025b44ed65e932a4ea9004678d5bd3edc70f8aaf2fb32f1ac7b3d4f749cee8.jpg)  
Figure 2: The performance and novelty comparison of different methods in Hopper-v3, Walker2d-v3 and HalfCheetah-v3 environments. The value of novelty is normalized to relative novelty by regarding the averaged novelty of PPO policies as the baseline. The results are from 10 policies of each method, with the points show their mean and lines show their standard deviation.

(11 observations and 3 actions), Walker2d-v3 (11 observations and 6 actions), and HalfCheetah-v3 (17 observations and 6 actions). Although relaxing the healthy termination thresholds in Hopper and Walker may permit more visible behavior diversity, all the environment parameters are set as default values in our experiments to demonstrate the generality of our method.

# 4.1.1 COMPARISON ON NOVELTY AND PERFORMANCE

We implement WSR, TNB, CTNB, and IPD using the same hyper-parameter settings per environment. And we also apply CPO Achiam et al. (2017) as a baseline as a solution of CMDP. For each method, we first train 10 policies using PPO with different random seeds. Those PPO policies are used as the primal reference policies, and then we train 10 novel policies that try to be different from previous reference policies. Concretely, in each method, the 1st novel policy is trained to be different from the previous 10 PPO policies, and the  $2nd$  should be different from the previous 11 policies, and so on. More implementation details are depicted in Appendix D.

Table 1: The Reward and Success Rate of 10 Policies. Our CTNB and IPD beat CPO, TNB and WSR in all three environments. Constrained optimization approaches outperforms multi-objective methods. Results are generated from 5 random seeds.  

<table><tr><td colspan="4">Reward</td><td colspan="3">Success Rate</td></tr><tr><td>Environment</td><td>Hopper</td><td>Walker2d</td><td>HalfCheetah</td><td>Hopper</td><td>Walker2d</td><td>HalfCheetah</td></tr><tr><td>PPO</td><td>1292 ± 650</td><td>2196 ± 200</td><td>1127 ± 308</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>WSR</td><td>1253 ± 591</td><td>1992 ± 380</td><td>1091 ± 469</td><td>0.6</td><td>0.3</td><td>0.3</td></tr><tr><td>TNB</td><td>1699 ± 573</td><td>1788 ± 214</td><td>887 ± 178</td><td>0.8</td><td>0.0</td><td>0.1</td></tr><tr><td>CPO</td><td>1681 ± 696</td><td>2082 ± 660</td><td>1194 ± 215</td><td>0.8</td><td>0.6</td><td>0.8</td></tr><tr><td>CTNB (Ours)</td><td>1721 ± 765</td><td>2405 ± 177</td><td>1251 ± 473</td><td>0.8</td><td>0.9</td><td>0.5</td></tr><tr><td>IPD (Ours)</td><td>2536 ± 557</td><td>2282 ± 206</td><td>1875 ± 533</td><td>1.0</td><td>0.6</td><td>0.9</td></tr></table>

Fig. 2 shows our experimental results in terms of novelty (the x-axis) and the performance (the y-axis). Policies close to the upper right corner are the more novel ones with higher performance. In all environments, the performance of CTNB, IPD and CPO outperforms WSR and TNB, showing the advantage of constrained optimization approaches in novel policy seeking. Specifically, the results of CTNB are all better than their multi-objective counterparts, i.e., the results from TNB, showing the superiority of seeking novel policies with constrained optimization. Moreover, the IPD method provides more novelty than CTNB and CPO, while the primal task performances are still guaranteed.

Comparisons of the task-related rewards are carried out in Table 1, where among all the four methods, IPD provides sufficient diversity with minimum loss of performance. Instead of performance decay, we find IPD is able to find better policies in the environment of Hopper and HalfCheetah. Moreover, in the Hopper environment, while the agents trained with PPO tend to fall into the same local minimum. (e.g., they all jump as far as possible and then terminate this episode. On the contrary, PPO with IPD keeps new agents away from falling into the same local minimum, because once an agent has reached some local minimum, agents learned later will try to avoid this region due to the novelty constraints. Such property shows that IPD can enhance the traditional RL schemes to tackle the local exploration challenge (Tessler et al., 2019; Ciosek et al., 2019). A similar feature brings about reward growth in the environment of HalfCheetah. Detailed analysis and discussions are developed in Appendix E.

# 4.1.2 SUCCESS RATE OF EACH METHOD

In addition to averaged reward, we also use the success rate as another metric to compare the performance of different approaches. Roughly speaking, the success rate evaluates the stability of each method in terms of generating a policy that performs as good as the policies PPO generates. In this work, we regard a policy successful when its performance achieves at least as good as the median performance of policies trained with PPO. To be specific, we use the median of the final performance of PPO as the baseline, and if a novel policy, which aims at performing differently to solve the same task, surpasses the baseline during its training process, it will be regarded as a successful policy. By definition, the success rate of PPO is 0.5 as a baseline for every environment. Table 1 shows the success rate of all the methods. The results show that all constrained novelty seeking methods (CTNB, IPD, CPO) can surpass the average baseline during training, while the multi-objective optimization approaches normally can not. Thus the performance of constrained novelty seeking methods can always be insured.

# 5 CONCLUSION

In this work, we rethink the novel policy seeking problem under the perspective of constrained optimization. We introduce a new metric to measure the distances between policies, and then we introduce the definition of policy novelty. We propose a new perspective by connecting the domain of constrained optimization to the domain of RL, and come up with two methods for seeking novel policies, namely the Constrained Task Novel Bisector (CTNB), and the Interior Policy Differentiation (IPD). Our experimental results demonstrate that the proposed method can effectively learn various well-behaved yet diverse policies, outperforming previous methods following the multi-objective formulation.

# REFERENCES

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 22-31. JMLR.org, 2017.  
Eitan Altman. Constrained Markov decision processes, volume 7. CRC Press, 1999.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 214-223, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR. URL http://proceedings.mlr.press/v70/arjovsky17a.html.  
Peter Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine Learning Research, 3(Nov):397-422, 2002.  
Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information Processing Systems, pp. 1471-1479, 2016.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell, and Alexei A Efros. Large-scale study of curiosity-driven learning. arXiv preprint arXiv:1808.04355, 2018a.  
Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018b.  
Yinlam Chow, Ofir Nachum, Edgar Duenez-Guzman, and Mohammad Ghavamzadeh. A lyapunov-based approach to safe reinforcement learning. In Advances in neural information processing systems, pp. 8092-8101, 2018.  
Kamil Ciosek, Quan Vuong, Robert Loftin, and Katja Hofmann. Better exploration with optimistic actor critic. In Advances in Neural Information Processing Systems, pp. 1785-1796, 2019.  
A Conn, Nick Gould, and Ph Toint. A globally convergent lagrangian barrier algorithm for optimization with general inequality constraints and simple bounds. Mathematics of Computation of the American Mathematical Society, 66(217):261-288, 1997.  
Edoardo Conti, Vashisht Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth Stanley, and Jeff Clune. Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents. In Advances in Neural Information Processing Systems, pp. 5027-5038, 2018.  
George B Dantzig and Mukund N Thapa. Linear programming 2: theory and extensions. Springer Science & Business Media, 2006.  
Dominik Maria Endres and Johannes E Schindelin. A new metric for probability distributions. IEEE Transactions on Information theory, 2003.  
Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.  
Carlos Florensa, Yan Duan, and Pieter Abbeel. Stochastic neural networks for hierarchical reinforcement learning. arXiv preprint arXiv:1704.03012, 2017.  
Bent Fuglede and Flemming Topsoe. Jensen-shannon divergence and hilbert space embedding. In International Symposium on Information Theory, 2004. ISIT 2004. Proceedings., pp. 31. IEEE, 2004.  
Scott Fujimoto, Herke Van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018.

Tanmay Gangwani, Qiang Liu, and Jian Peng. Learning self-imitating diverse policies. arXiv preprint arXiv:1805.10309, 2018.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.  
Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Jose Herskovits. Feasible direction interior-point technique for nonlinear optimization. Journal of optimization theory and applications, 99(1):121-146, 1998.  
Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. Variational information maximizing exploration. 2016.  
Joel Lehman and Kenneth O Stanley. Novelty search and the problem with objectives. In Genetic programming theory and practice IX, pp. 37-56. Springer, 2011.  
Gunar E Liepins and Michael D Vose. Deceptiveness and genetic algorithm dynamics. In Foundations of genetic algorithms, volume 1, pp. 36-50. Elsevier, 1991.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
Hao Liu, Alexander Trott, Richard Socher, and Caiming Xiong. Competitive experience replay. CoRR, abs/1902.00528, 2019. URL http://arxiv.org/abs/1902.00528.  
Yang Liu, Prajit Ramachandran, Qiang Liu, and Jian Peng. Stein variational policy gradient. arXiv preprint arXiv:1704.02399, 2017.  
Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped dqn. In Advances in neural information processing systems, pp. 4026-4034, 2016.  
Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep reinforcement learning. In Advances in Neural Information Processing Systems, pp. 8617-8629, 2018.  
Georg Ostrovski, Marc G Bellemare, Aäron van den Oord, and Rémi Munos. Count-based exploration with neural density models. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 2721-2730. JMLR.org, 2017.  
Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 16-17, 2017.  
Matthias Plappert, Marcin Andrychowicz, Alex Ray, Bob McGrew, Bowen Baker, Glenn Powell, Jonas Schneider, Josh Tobin, Maciek Chociej, Peter Welinder, et al. Multi-goal reinforcement learning: Challenging robotics environments and request for research. arXiv preprint arXiv:1802.09464, 2018.  
Florian A Potra and Stephen J Wright. Interior-point methods. Journal of Computational and Applied Mathematics, 124(1-2):281-302, 2000.  
Justin K Pugh, Lisa B Soros, and Kenneth O Stanley. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40, 2016.  
Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking safe exploration in deep reinforcement learning. *openai*, 2019.  
Ludger Ruschendorf. The wasserstein distance and approximation theorems. Probability Theory and Related Fields, 70(1):117-129, 1985.

Andrzej Ruszczyński. Feasible direction methods for stochastic programming problems. Mathematical Programming, 19(1):220-229, 1980.  
Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864, 2017.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889-1897, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. arXiv preprint arXiv:1907.01657, 2019.  
Felipe Petroski Such, Vashisht Madhavan, Rosanne Liu, Rui Wang, Pablo Samuel Castro, Yulun Li, Ludwig Schubert, Marc Bellemare, Jeff Clune, and Joel Lehman. An atari model zoo for analyzing, visualizing, and comparing deep reinforcement learning agents. arXiv preprint arXiv:1812.07069, 2018.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. 1998.  
Richard S Sutton, Andrew G Barto, et al. Introduction to reinforcement learning, volume 2. MIT press Cambridge, 1998.  
Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems, pp. 1057-1063, 2000.  
Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. In Advances in neural information processing systems, pp. 2753-2762, 2017.  
Chen Tessler, Guy Tennenholtz, and Shie Mannor. Distributional policy optimization: An alternative approach for continuous control. arXiv preprint arXiv:1905.09855, 2019.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In IROS, pp. 5026-5033. IEEE, 2012. ISBN 978-1-4673-1737-5. URL http://dblp.uni-trier.de/db/conf/iros/iros2012.html#TodorovET12.  
Cédric Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media, 2008.  
Stephen J Wright. On the convergence of the newton/log-barrier method. Mathematical Programming, 90(1):71-100, 2001.  
Yunbo Zhang, Wenhao Yu, and Greg Turk. Learning novel policies for tasks. CoRR, abs/1905.05252, 2019. URL http://arxiv.org/abs/1905.05252.
