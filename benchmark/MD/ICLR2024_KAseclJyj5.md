# DOI: OFFLINE DIVERSITY MAXIMIZATION UNDER IMITATION CONSTRAINTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

There has been significant recent progress in the area of unsupervised skill discovery, utilizing various information-theoretic objectives as measures of diversity. Despite these advances, challenges remain: current methods require significant online interaction, fail to leverage vast amounts of available task-agnostic data and typically lack a quantitative measure of skill utility. We address these challenges by proposing a principled offline algorithm for unsupervised skill discovery that, in addition to maximizing diversity, ensures that each learned skill imitates state-only expert demonstrations to a certain degree. Our main analytical contribution is to connect Fenchel duality, reinforcement learning, and unsupervised skill discovery to maximize a mutual information objective subject to KL-divergence state occupancy constraints. Furthermore, we demonstrate the effectiveness of our method on the standard offline benchmark D4RL and on a custom offline dataset collected from a 12-DoF quadruped robot for which the policies trained in simulation transfer well to the real robotic system.<sup>1</sup>

![](images/cdfa0df4dc09135014ebadd87691db7962193c90cd575e41d56bc1c8b4905bb7.jpg)  
Figure 1: Diverse Offline Imitation (DOI) maximizes a variational lower bound on the mutual information between latent skills  $z$  and states  $s$  visited by associated skill-conditioned policies  $\pi_z$ , subject to a KL-divergence constraint to limit the deviation of the state occupancy  $d_z(s)$  of each latent skill  $z$  from that of an expert  $d_E(s)$ .

# 1 INTRODUCTION

Recent advancements in reinforcement learning (RL) have included substantial progress in unsupervised skill discovery, aiming to empower autonomous agents with the capability to acquire a diverse set of skills directly from their environment, without relying on predefined human-engineered rewards or demonstrations. These methods have the potential to revolutionize the way RL agents learn to solve complex tasks. The growing interest in unsupervised skill discovery has led to various approaches, typically rooted in information-theoretic concepts, including empowerment (Klyubin et al., 2005; Mohamed and Jimenez Rezende, 2015; Eysenbach et al., 2019), information bottleneck (Tishby et al., 1999; Goyal et al., 2019; Kim et al., 2021a) and information gain (Houthooft

et al., 2016; Strouse et al., 2022; Park and Levine, 2023). Despite these advancements, there remains a significant challenge. Current methods demand substantial online interaction with the environment, making exploration in high-dimensional state-action spaces inefficient. Although Zahavy et al. (2022) introduced constraints to enhance skill performance and narrow the exploration space by incentivizing diverse skills to meet a certain utility measure, their approach does not eliminate the need for considerable online interaction with the environment. Meanwhile, there have been significant recent advances in large-scale data collection (Rob, 2020; Walke et al., 2023; Brohan et al., 2023) and in the development of scalable and sample-efficient offline RL algorithms that leverage diverse behaviors of pre-collected experience. However, these approaches struggle with well-known challenges, including off-policy evaluation and the out-of-distribution problem, which have been studied extensively in previous work (Levine et al., 2020; Prudencio et al., 2022).

In this work, we address the aforementioned challenges by introducing a novel problem formulation and complementing it with the first principled "offline" RL algorithm for unsupervised skill discovery that, in addition to maximizing diversity, ensures that each learned skill imitates state-only expert demonstrations to a certain degree. More specifically, we consider a problem formulation with two datasets: a large one with diverse state-action demonstrations and another much smaller one with state-only expert demonstrations. This setting is particularly valuable in robotics scenarios where expert demonstrations are limited and the domain of the expert may be different from that of the agent, such as in human demonstrations. Another potential application is to enhance the realism of computer games by creating an immersive experience of interacting with non-player characters, each behaving in a slightly different style, while all partially imitating the behavior of a human expert.

We formulate the problem as a Constrained Markov Decision Process (CMDP) (Altman, 1999; Szepesvári, 2020) that seeks to maximize diversity through a mutual information objective, subject to Kullback-Leibler (KL) divergence state occupancy constraints ensuring that each skill imitates state expert demonstrations to a certain degree. The resulting CMDP has convex objective and constraints, making the optimization problem intractable. We adopt a tractable relaxation approach consisting of an alternating scheme that maximizes a variational lower bound on mutual information, and to handle the constraints it applies Lagrange relaxation. Our method, Diverse Offline Imitation (DOI), overcomes the off-policy evaluation by leveraging the Fenchel-Rockafellar duality in RL (Nachum and Dai, 2020; Kim et al., 2022; Ma et al., 2022) to connect a dual optimal value solution (computed using offline samples) with primal optimal state-action occupancy ratios. These ratios serve as importance weights for offline training of a skill-conditioned policy, skill-discriminator, KL-divergence estimators, and Lagrange multipliers. We demonstrate the effectiveness of our method on the standard offline benchmark D4RL (Fu et al., 2020) and on a custom offline dataset collected from a 12-DoF quadruped robot Solo12 (Léziart et al., 2021). In addition, we show that DOI on simulation data transfers well to a real robot system.

# 2 RELATED WORK

In the context of skill discovery Achiam et al. (2018) and Campos et al. (2020) showed that methods like DIAYN (Eysenbach et al., 2019) can struggle to learn large numbers of skills and have a poor coverage of the state space. Strouse et al. (2022) observed that when a novel state is visited, the discriminator lacks sufficient training data to accurately classify skills, which results in a low intrinsic reward for exploration. They address this by introducing an information gain objective (involving an ensemble of discriminators) as a bonus term. Kim et al. (2021b) gave a skill discovery approach based on an information bottleneck that leads to disentangled and interpretable skill representations. Park et al. (2022; 2023) proposed a Lipschitz-constrained skill discovery method based on a distance-maximizing and controllability-aware distance function to overcome the bias toward static skills and to allow the agent to learn complex and far-reaching behaviors. Sharma et al. (2020) developed a method that simultaneously discovers predictable skills and learns their dynamics. In a follow-up work, Park and Levine (2023) addresses the problem of errors in predictive models by learning a transformed MDP, whose action space contains only easy to model and predictable actions. These works provide RL algorithms for unsupervised skill discovery that require online interaction with the environment and do not impose utility measures on the learned skills. In contrast, DOI gives a principled offline algorithm for maximizing diversity under imitation constraints.

A large body of research has focused on successor features (Dayan, 1993; Barreto et al., 2016), a powerful technique in RL for transfer of knowledge across tasks by capturing environmental dynamics, particularly promising for skill discovery when coupled with variational intrinsic motivation (Gregor et al., 2017; Barreto et al., 2018; Hansen et al., 2020) to enhance feature controllability, generalization, and task inference. In contrast to our work, these approaches do not impose performance constraints on the learned skills. Zahavy et al. (2022) cast the task of learning diverse skills, each achieving a near-optimal performance with respect to a given reward, into a constrained MDP setting with a physics-inspired diversity objective based on a minimum  $\ell_2$  distance between the successor features of different skills. However, this approach requires significant online interaction with the environment to learn the skills.

Numerous practical algorithms for offline RL have been proposed (Levine et al., 2020; Prudencio et al., 2022), including methods based on advantage-weighted behavioral cloning (Nair et al., 2020; Wang et al., 2020), conservative strategies to stay close to the original data distribution (Kumar et al., 2020; Cheng et al., 2022) and using only on-data samples (Kostrikov et al., 2022; Xu et al., 2023). While these methods excel at learning a policy that maximizes a fixed reward, they are not directly applicable in our setting, which has a non-stationary reward that depends on: i) the log-likelihood of a skill discriminator, and ii) Lagrange multipliers. In addition, these techniques cannot be used to i) train a skill discriminator and ii) estimate a KL divergence offline.

Naive importance sampling approaches for off-policy estimation are known to suffer from unbounded variance in the infinite horizon setting, a problem known in the literature as "the curse of horizon". Liu et al. (2018); Mousavi et al. (2020) addressed this challenge by providing theoretical foundations and a principled off-policy algorithm, using a backward Bellman operator, that avoids exploding variance by applying importance sampling to state-visitation distributions, and by providing practical solutions in Reproducing Kernel Hilbert Spaces. An alternative research direction in off-policy estimation, referred to as "Distribution Correction Estimation (DICE)", has introduced innovative techniques, with Nachum et al. (2019a) mitigating variance with importance sampling, Nachum et al. (2019b) enabling policy gradient from off-policy data without importance weighting, Kim et al. (2022) stabilizing offline imitation learning with imperfect demonstrations, Zhang et al. (2020) improving density ratio estimation, Dai et al. (2020) providing high-confidence off-policy evaluation. Subsequently, Xu et al. (2021) applied this approach to offline RL and demonstrated its effectiveness in continuous control tasks. Our work uses a DICE-based off-policy approach similar to OptiDICE (Lee et al., 2021; 2022) for estimating importance ratios, while considering a constrained formulation with a mutual information objective and KL-divergence imitation constraints.

# 3 PRELIMINARIES

We utilize the framework of Markov decision processes (MDPs) (Puterman, 2014), where an MDP is defined by the tuple  $(\mathcal{S},\mathcal{A},\mathcal{R},\mathcal{P},\rho_0,\gamma)$  denoting the state space, action space, reward mapping  $\mathcal{R}:S\times \mathcal{A}\mapsto \mathbb{R}$ , stochastic transition kernel  $\mathcal{P}(s'|s,a)$ , initial state distribution  $\rho_0(s)$  and discount factor  $\gamma$ . A policy  $\pi :S\mapsto \Delta (\mathcal{A})$  defines a probability distribution over the action space  $\mathcal{A}$  conditioned on the state, where  $\Delta (\cdot)$  stands for the probability simplex.

Given a policy  $\pi$ , the corresponding state-action occupancy measure  $d^{\pi}(s,a)$  is defined by  $(1 - \gamma)\sum_{t = 0}^{\infty}\gamma^{t}\mathrm{Pr}[s_{t} = s,a_{t} = a|s_{0}\sim \rho_{0},a_{t}\sim \pi (\cdot |s_{t}),s_{t + 1}\sim \mathcal{P}(\cdot |s_{t},a_{t})]$  and its associated state occupancy  $d^{\pi}(s)$  is given by marginalizing over the action space  $\sum_{a\in \mathcal{A}}d^{\pi}(s,a)$ .

In the skill discovery setting,  $z \sim p(Z)$  denotes a fixed latent skill on which we condition a policy  $\pi_z: S \times Z \mapsto \Delta(\mathcal{A})$ . We will treat  $p(Z)$  as a categorical distribution over a discrete set  $Z$  of  $|Z|$  many distinct indicator vectors in  $\mathbb{R}^{|Z|}$ . The skill-conditioned policy  $\pi_z$  induces a state occupancy denoted by  $d_z(s) \coloneqq d^{\pi_z}(s)$ , and when it is clear from the context we will refer to  $d_z(s)$  as a "skill".

We consider an offline setting with access to the following datasets: i)  $\mathcal{D}_E$  sampled from an expert state occupancy  $d_E(S)$ ; and ii)  $\mathcal{D}_O$  sampled from a state-action occupancy  $d_O(S, A)$  generated by a mixture of behaviors. Our analysis makes use of the following coverage assumption on state occupancies.

Assumption 3.1 (Expert coverage). We assume that  $d_{E}(s) > 0$  implies  $d_{O}(s) > 0$ .

# 4 METHOD

Given an expert and a coverage dataset as above, we aim to solve offline the constrained optimization problem

$$
\max  _ {\left\{d _ {z} (S) \right\} _ {z \in Z}} \quad \mathcal {I} (S; Z) \tag {1}
$$

$$
\text {s u b j e c t} \quad \mathrm {D} _ {\mathrm {K L}} (d _ {z} (S) | | d _ {E} (S)) \leq \epsilon \quad \forall z, \tag {2}
$$

where  $\mathcal{I}(S;Z)$  denotes the mutual information between states and skills. Henceforth, we shall make use of color coding to highlight the diversity signal in blue and the imitation signal in orange. The preceding problem formulation and our algorithmic framework can be easily extended to capture: i) objectives in (1) that combine conditional mutual information (c.f. DADS in (Sharma et al., 2020)) and information gain (c.f. DISDAIN in (Strouse et al., 2022)); and ii) general  $f$ -divergence constraints in (2), see Nachum and Dai (2020); Ma et al. (2022). We leave the study of these variants for future work.

Since maximizing the mutual information is generally intractable, in line with previous work (Eysenbach et al., 2019) we assume that the latent skills are sampled uniformly at random, i.e.,  $p(z) = \frac{1}{|Z|}$ , and as a trackable surrogate we consider instead the following variational lower bound

$$
\mathcal {I} (S; Z) \geq \mathbb {E} _ {p (z), d _ {z} (s)} [ \log q (z | s) ] + \mathcal {H} (p (z)) = \sum_ {z} \mathbb {E} _ {d _ {z} (s)} \left[ \frac {\log (| Z | q (z | s))}{| Z |} \right]. \tag {3}
$$

Here with  $q(z|s)$  we denote a skill-discriminator tasked with distinguishing between latent skills.

Ma et al. (2022) proposed an offline algorithm (SMODICE) that on input an expert dataset  $\mathcal{D}_E\sim$ $d_{E}(S)$  and a coverage dataset  $\mathcal{D}_O\sim d_O(S,A)$  such that  $\mathcal{D}_E\subset$  States[  $\mathcal{D}_O]$ , trains a policy  $\pi_{\widetilde{E}}$  which optimizes the problem

$$
\min  _ {\pi} \mathrm {D} _ {\mathrm {K L}} \left(d ^ {\pi} (S) | | d _ {E} (S)\right), \tag {4}
$$

and also outputs ratios  $\eta_{\widetilde{E}}(s,a) = d_{\pi_{\widetilde{E}}} (s,a) / d_O(s,a)$  for every state-action pair  $(s,a) \in \mathcal{D}_O$

An important observation is that the state constraints (2) can be reduced to state-action constraints, by training an expert policy  $\pi_{\widetilde{E}}$ , which optimizes eq. (4). More specifically, for each latent skill  $z$  we replace the state constraint (2) with the following state-action constraint

$$
\mathrm {D} _ {\mathrm {K L}} \left(d _ {z} (S, A) | | d _ {\bar {E}} (S, A)\right) \leq \epsilon , \tag {5}
$$

where  $d_{\widetilde{E}}(s,a)$  denotes the state-action occupancy  $d_{\pi_{\widetilde{E}}}(\boldsymbol {s},a)$  induced by the expert policy  $\pi_{\widetilde{E}}$

We focus on a reduction of CMDPs to MDPs using gradient-based techniques, known as Lagrangian methods (Borkar, 2005; Bhatnagar and Lakshmanan, 2012; Tessler et al., 2019). In contrast to prior work on CMDP, which has focused primarily on linear objectives and constraints, we consider the nonlinear setting with convex objectives and constraints. More specifically, we seek to maximize the right-hand side of eq. (3) subject to eq. (5). Solving this problem is equivalent to

$$
\max  _ {\substack {d _ {z} (s, a) \\ q (z | s)}} \min  _ {\lambda \geq 0} \sum_ {z} \mathbb {E} _ {d _ {z} (s)} \left[ \frac {\log (| Z | q (z | s))}{| Z |} \right] + \sum_ {z} \lambda_ {z} \left[ \epsilon - \mathrm {D} _ {\mathrm {K L}} \left(d _ {z} (S, A) \| d _ {\widetilde {E}} (S, A)\right) \right], \tag{6}
$$

where with  $\lambda_z$  we denote the Lagrange multiplier corresponding to latent skill  $z$ .

# 4.1 APPROXIMATION SCHEME

We use a popular heuristic, known in the literature as alternating optimization, to approximately compute a local optimum of Problem (6). More precisely, the method alternates between optimizing each model while holding all others fixed, and iteratively refines the solution until convergence is reached or a stopping criterion is met. Furthermore, as we can guarantee in practice that the Lagrange multipliers  $\lambda$  are always positive, we consider Problem (6) with  $\lambda > 0$ , that is

$$
\max  _ {\substack {d _ {z} (s, a) \\ q (z \mid s)}} \min  _ {\lambda > 0} \sum_ {z} \lambda_ {z} \left\{\epsilon + \mathbb {E} _ {d _ {z} (s, a)} \left[ R _ {z} ^ {\lambda} (s, a) \right] - \mathrm {D} _ {\mathrm {KL}} \left(d _ {z} (S, A) \mid \mid d _ {O} (S, A)\right) \right\}, \tag{7}
$$

![](images/1be6a32797d1e2e5dc6245b09c9c2eb06d02008a81441ecf4046b55f74f68542.jpg)  
Figure 2: Illustration of Algorithm 1. We compute expert importance ratios  $\eta_{\widetilde{E}}(s,a)$  by running SMODICE on the offline datasets  $\mathcal{D}_E$  and  $\mathcal{D}_O$ . These expert ratios are then used in the alternating scheme described in Subsec. 4.1 to obtain the importance ratios  $\eta_z(s,a)$  (with support in  $\mathcal{D}_O$ ) for each skill  $z$ . Specifically, the skill-ratios  $\eta_z(s,a)$  are computed by a DICE-like offline policy evaluation algorithm on input a reward  $R_z^\mu (s,a)$  that balances skill diversity (skill-discriminator  $q(z|s)$ ) and expert imitation (importance ratios  $\eta_{\widetilde{E}}(s,a)$ ).

where

$$
R _ {z} ^ {\lambda} (s, a) := \underbrace {\frac {1}{\lambda_ {z}}} _ {\text {C o n s t r a i n t V i o l a t i o n}} \underbrace {\frac {\log (q (z | s) | Z |)}{| Z |}} _ {\text {S k i l l D i v e r s i t y}} + \underbrace {\log \eta_ {\widetilde {E}} (s , a)} _ {\text {E x p e r t I m i t a t i o n}}. \tag {8}
$$

The reward in (8) is derived in Supp. B and relies on the following equality (see Supp. C.3)  $\mathrm{D}_{\mathrm{KL}}(d_z(S,A)||d_{\widetilde{E}}(S,A)) = \mathrm{D}_{\mathrm{KL}}(d_z(S,A)||d_O(S,A)) - \mathbb{E}_{d_z(s,a)}[\log \frac{d_{\widetilde{E}}(s,a)}{d_O(s,a)}]$  and the definition of  $\eta_{\widetilde{E}}(s,a) = d_{\widetilde{E}}(s,a) / d_O(s,a)$ .

Intuitively, the reward  $R_{z}^{\lambda}(s,a)$  balances between diversity and KL-closeness to the expert state-action occupancy. The Lagrange multiplier  $\lambda_{z}$  scales down the log-likelihood of the skill-discriminator  $q(z|s)$ , effectively reducing the diversity signal, when the state-action occupancy  $d_{z}(S,A)$  violates the KL-divergence constraint (5), and vice versa. Each term in the reward (8) involves a separate optimization procedure, which will be described in the next section.

# 4.2 APPROXIMATION PHASES

Using the alternating optimization scheme, Algorithm 1 decomposes into the following three optimization phases. In PHASE 1, we train a value function  $V_{z}^{\star}$ , ratios  $\eta_z(s,a)$  and a skill-conditioned policy  $\pi_{z}$ . In PHASE 2, we train a skill-discriminator  $q(z|s)$ . Then in PHASE 3, we compute a KL constraint estimator  $\phi_{z}$  and update accordingly the Lagrange multipliers  $\lambda_{z}$ . In addition, we perform a preprocessing phase to compute the expert ratios  $\eta_{\widetilde{E}}(s,a)$  by invoking the SMODICE algorithm.

# 4.2.1 PHASE 1

With fixed skill-discriminator  $q(z|s)$  and Lagrange multipliers  $\lambda > 0$ , Problem (7) becomes

$$
\left. \max  _ {\{d _ {z} (s, a) \} _ {z \in Z}} \sum_ {z} \lambda_ {z} \left\{\mathbb {E} _ {d _ {z} (s, a)} \left[ R _ {z} ^ {\lambda} (s, a) \right] - \mathrm {D} _ {\mathrm {K L}} \left(d _ {z} (S, A) | | d _ {O} (S, A)\right) \right\}, \right. \tag {9}
$$

or equivalently for every skill  $z$ :

$$
\max  _ {d _ {z} (s, a) \geq 0} \mathbb {E} _ {d _ {z} (s, a)} \left[ R _ {z} ^ {\lambda} (s, a) \right] - \mathrm {D} _ {\mathrm {K L}} \left(d _ {z} (S, A) | | d _ {O} (S, A)\right)
$$

$$
\text {s u b j e c t} \quad \sum_ {a} d _ {z} (s, a) = (1 - \gamma) \rho_ {0} (s) + \gamma \mathcal {T} d (s) \quad \forall s, \tag {10}
$$

where we denote with  $\mathcal{T}$  the transition operator:  $\mathcal{T}d(s^{\prime}) = \sum_{s,a}\mathcal{P}(s^{\prime}|s,a)d(s,a)$ .

Assumption 4.1 (Strict Feasibility). We assume there exists a solution such that the constraints (10) are satisfied and  $d(s, a) > 0$  for all states-action pairs  $(s, a) \in S \times \mathcal{A}$ .

Using Lagrange duality, Assum. 4.1 (which implies strong duality) and the Fenchel conjugate (see Supp. A), Nachum and Dai (2020, Sec. 6) and Ma et al. (2022, Theorem 2) showed that Problem 10 shares the same optimal value as the following optimization problem

$$
V ^ {\star} = \underset {V (s)} {\arg \min } (1 - \gamma) \mathbb {E} _ {s \sim \rho_ {0}} [ V (s) ] + \log \mathbb {E} _ {d _ {O} (s, a)} \exp \left\{R _ {z} ^ {\lambda} (s, a) + \gamma \mathcal {T} V (s, a) - V (s) \right\}, \tag {11}
$$

where  $\mathcal{T}V(s,a)\coloneqq \mathbb{E}_{\mathcal{P}(s'|s,a)}V(s')$ . Moreover, the primal optimal solution is given by

$$
\eta_ {z} (s, a) := \frac {d _ {z} ^ {\star} (s , a)}{d _ {O} (s , a)} = \operatorname {s o f t m a x} \left(R _ {z} ^ {\lambda} (s, a) + \gamma \mathcal {T} V _ {z} ^ {\star} (s, a) - V _ {z} ^ {\star} (s)\right). \tag {12}
$$

These ratios  $\eta_z(s, a)$  are then used to design an offline importance-weighted sampling procedure that, for an arbitrary function  $f$ , satisfies

$$
\mathbb {E} _ {p (z)} \mathbb {E} _ {d _ {z} ^ {*} (s, a)} [ f (s, a, z) ] = \mathbb {E} _ {p (z)} \mathbb {E} _ {d _ {O} (s, a)} [ \eta_ {z} (s, a) f (s, a, z) ]. \tag {13}
$$

Afterwards, the optimal skill-conditioned policy  $\pi_z^\star$  is trained offline using a weighted behavioral cloning, which is obtained by setting  $f(s,a,z) = \log (\pi_z(a|s))$  and maximizing the RHS of eq. (13) over all skill-conditioned policies  $\pi_z$ . In practice, gradient descent is used for optimization.

# 4.2.2 PHASE 2

We now give an offline procedure for training a skill-discriminator  $q(z|s)$ , which takes as input ratios  $\eta_z(s,a)$  of a skill-conditioned policy  $\pi_z^\star$ . The proof is presented in Supp. C.2.

Lemma 4.2. Given ratios  $\eta_z(s,a)$ , using eq. (13) applied with  $f(s,a,z) = \log (q(z|s))$ , we can compute offline an optimal skill-discriminator  $q^{\star}(z|s)$ . In particular, we optimize by gradient descent the following optimization problem  $\max_{q(z|s)}\mathbb{E}_{p(z)}\mathbb{E}_{d_O(s,a)}[\eta_z(s,a)\log (q(z|s))]$ .

The skill-conditioned policy  $\pi_z^\star$  (PHASE 1) and the skill-discriminator  $q^\star$  (PHASE 2), allow us to maximize offline the variational lower bound in eq. (3) and thus skill diversity. It remains to estimate possible constraint violations in eq. (5) and to update the Lagrange multipliers accordingly.

# 4.2.3 PHASE 3

With fixed skill-discriminator  $q^{\star}(z|s)$  and skill-conditioned policy  $\pi_z^\star (s)$ , Problem (7) reduces to  $\min_{\lambda >0}\sum_z\lambda_z\left[\epsilon -\mathrm{D}_{\mathrm{KL}}\left(d_z^\star (S,A)\| d_{\widetilde{E}}(S,A)\right)\right]$ . We will optimize the Lagrange multipliers by gradient descent. To this end, we now give an offline estimator of the KL-divergence term. The proof is presented in Supp. C.3.

Lemma 4.3. Given skill-conditioned policy ratios  $\eta_z(s,a)$  and expert ratios  $\eta_{\widetilde{E}}(s,a)$ , using eq. (13) applied with  $f(s,a,z) = \log (\eta_z(s,a) / \eta_{\widetilde{E}}(s,a))$ , we can compute offline an estimator of  $\mathrm{D}_{\mathrm{KL}}\left(d_z^\star (S,A)||d_{\widetilde{E}}(S,A)\right)$  which is given by  $\phi_z\coloneqq \mathbb{E}_{d_O(s,a)}[\eta_z(s,a)\log (\eta_z(s,a) / \eta_{\widetilde{E}}(s,a))]$ .

We note that the ratios  $\eta_z(s, a)$  and  $\eta_{\widetilde{E}}(s, a)$  are computed only on state-action pairs within the offline dataset  $\mathcal{D}_O$ . Furthermore, in practice, we ensure that these ratios are strictly positive, so that the KL estimator  $\phi_z$  is well defined and bounded.

# 5 ALGORITHM

Our optimization method consists of three phases, each of which optimizes a specific model and fixes the remaining ones. It is important to emphasize that in contrast to prior work, our problem formulation considers an optimization problem with constraints. Furthermore, the reward function in eq. (8) is non-stationary, since it depends on the bounded Lagrange multipliers that balance diversity  $(\log q(z|s))$  and expert imitation  $(\log \eta_{\widetilde{E}}(s,a))$ . This has significant algorithmic implications, as it requires solving a sequence of standard RL problems, each of which admits offline policy evaluation.

To smooth the transition of the reward signal between successive iterations, we enforce a slow change of the Lagrange multipliers. More specifically, we use the technique of bounded Lagrange multipliers (Stooke et al., 2020; Zahavy et al., 2022), which applies a Sigmoid transformation  $\lambda = \sigma(\mu)$  component-wise to unbounded variables  $\mu \in \mathbb{R}^{|Z|}$ , so that the effective reward is a convex combination of a diversity term and an expert imitation term. In practice, this transformation ensures that  $\lambda > 0$ . Hence, the reward for each latent skill  $z$  becomes

$$
R _ {z} ^ {\mu} (s, a) := \left(1 - \sigma \left(\mu_ {z}\right)\right) \frac {\log \left(q ^ {\star} (z | s) | Z |\right)}{| Z |} + \sigma \left(\mu_ {z}\right) \log \eta_ {\tilde {E}} (s, a). \tag {14}
$$

We now present the resulting multi-phase optimization procedure in Algorithm 1. For a practical implementation, we leverage the power of neural networks and deep learning techniques for accurate

function approximation. More specifically, we train an expert policy  $\pi_{\widetilde{E}}$ , a skill-conditioned policy  $\{\pi_z\}_{z \in Z}$  and a value function  $\{V_z\}_{z \in Z}$ . While practically convenient, this means that each phase of Algorithm 1 is only approximately solved. In practice, we do not solve the optimization problem to optimality in each phase, but rather perform a few gradient descent steps.

# Algorithm 1 Diverse Offline Imitation (DOI)

Input: a state-only expert dataset  $\mathcal{D}_E\sim d_E(S)$  and a state-action offline dataset  $\mathcal{D}_O\sim d_O(S,A)$

Pre-compute a state-discriminator  $c^{\star} : \mathcal{S} \to (0,1)$  via optimizing the following objective with the gradient penalty in (Gulrajani et al., 2017)  $\min_c \mathbb{E}_{d_E(s)}[\log c(s)] + \mathbb{E}_{d_O(s)}[\log (1 - c(s))]$

Apply Phase 1 with reward  $R(s, a) = \log \frac{c^{\star}(s)}{1 - c^{\star}(s)}$  to compute ratios  $\eta_{\widetilde{E}}(s, a) \coloneqq \frac{d_{\widetilde{E}}(s, a)}{d_{O}(s, a)}$  for all  $s, a \in \mathcal{D}_O$ .

# Repeat until convergence:

Phase 1. (Fixed Lagrange multipliers  $\sigma (\mu)$  and skill-discriminator values  $q^{*}(z|s))$

For each latent skill  $z$ :

compute a value function  $V_{z}^{\star}$  optimizing eq. (11) with reward  $R_{z}^{\mu}(s,a)$  in eq. (14)

compute ratios  $\eta_z(s,a)\coloneqq \frac{d_\star^*(s,a)}{d_O(s,a)} = \mathrm{softmax}\left(R_z^\mu (s,a) + \gamma \mathcal{T}V_z^\star (s,a) - V_z^* (s)\right)$  for all  $s,a\in \mathcal{D}$

train a skill-conditioned policy  $\pi_z^\star = \arg \max_{\pi_z}\mathbb{E}_{d_O(s,a)}[\eta_z(s,a)\log \pi_z(a|s)]$

Phase 2. (Fixed ratios  $\eta_z(s,a)$  and bounded Lagrange multipliers  $\sigma (\mu)$ )

Train a skill-discriminator  $q^{\star} = \arg \max_{q(\cdot |s)}\mathbb{E}_{p(z)}\mathbb{E}_{d_O(s,a)}[\eta_z(s,a)\log q(z|s)]$

Phase 3. (Fixed ratios  $\eta_{\bar{F}}(s,a)$  and  $\eta_z(s,a)$ )

Compute for each latent skill  $z$  an estimator  $\phi_z\coloneqq \mathbb{E}_{d_O(s,a)}[\eta_z(s,a)\log (\eta_z(s,a) / \eta_{\bar{E}}(s,a))]$

Optimize the loss  $\min_{\mu}\sum_{z}\sigma (\mu_{z})(\epsilon -\phi_{z})$

# 6 EXPERIMENTS

For evaluation of our method we consider 12 degree-of-freedom quadruped robot, SOLO12 (Grimminger et al., 2020), on a simple locomotion task in both simulation and the real system. We provide further evaluation on the ANT, WALKER2D, HALFCHEETAH and HOPPER environments from the D4RL benchmark (Fu et al., 2020).

For the SOLO12 evaluation we collected domain-randomized offline and expert data from simulation in the Isaac Gym (Makoviychuk et al., 2021) using saved checkpoints obtained by training the robot to track a certain velocity of the base with a version of DOMiNO (Zahavy et al., 2022). We defer the training procedure of the policies used for data collection to the Supp. E. The expert dataset was collected by using the best deterministic skill-conditioned policy from the last checkpoint of the training procedure, which was trained to track forward velocity only. In contrast, the offline dataset was acquired by employing stochastic policies gathered from various checkpoints throughout the training of the expert, featuring multiple latent skills. More than half of the offline dataset was collected by a random Gaussian policy. In line with previous approaches by Kim et al. (2022) and Ma et al. (2022), our practical implementation aims to fulfill the expert coverage Assum. 3.1. To achieve this, we create the coverage dataset  $\mathcal{D}_O$  by adding a small number of expert trajectories to the offline dataset, resulting in an (unlabeled) expert fraction of  $1/160$  in  $\mathcal{D}_O$ . To ensure that our algorithm does not have access to labeled expert actions, we discard them from the expert dataset. The resulting expert dataset  $\mathcal{D}_E$  is used to learn a state classifier, in order to compute the ratios  $\eta_{\widetilde{E}}(s,a)$ . We trained the policy for 350 steps, where each step involves the stages described in Sec. 5. In each stage, we execute 200 epochs of batched training over the data. For the computation of the skill-ratios  $\eta_z(s,a)$ , we choose a projection  $\Pi$  of the expert state (see Supp. I) that yields 3-dimensional planar and angular velocities of the robot's base in the base frame.

We have found that fitting the skill-discriminator  $q(z|s)$  is prone to collapse to the uniform distribution. To alleviate this issue, in addition to the variational lower bound objective (3), we add the DISDAIN information gain term, proposed in (Strouse et al., 2022). This bonus term is an entropy-based disagreement penalty that estimates the epistemic uncertainty of the skill-discriminator, and is implemented in practice by an ensemble of randomly initialized skill-discriminators. Due to the high initial disagreement on unvisited states, this intrinsic reward provides a strong exploration signal and leads to the discovery of more diverse behaviors. Intuitively, for states with small epistemic uncertainty, the skill-discriminator (averaged over the ensemble members) should reliably discrimi

nate between latent skills, thus making the intrinsic reward of the skill-discriminator's log-likelihood more accurate. In all figures, we denote with  $\mathrm{DOI}^{\epsilon}$  the different constraint levels. We defer further experiment details to Supp. K.

![](images/e84f4525c88e1c0e89a3e317008d85b12cd8facba2cac2eb428923f5a63d79df.jpg)  
(a)

![](images/9942044a86f1e07a9b7e6249f5f8482eb526a540f6b2fee26cd4fbcdeddaeb91.jpg)  
Figure 3: Data points separation by importance ratios  $\eta_z(s,a)$ , given different levels of  $\epsilon$  in SOLO12. (a) Distribution of importance ratios  $\eta_z(s,a)$  over the offline dataset  $\mathcal{D}_O$  for different skills with  $\mathrm{DOI}^4$  ( $\epsilon = 4$ ) (upper) and a skill-conditioned variant of SMODICE (lower). (b) Average  $\ell_1$  distance of ratios  $\eta_z$  belonging to different skills, depending on  $\epsilon$ . The higher the value of  $\epsilon$ , the greater the  $\ell_1$  distance.  
(b)

As a baseline, we consider a skill-conditioned variant of (Ma et al., 2022), denoted SMODICE†, which does not have access to the skill-discriminator  $q(z|s)$ . This is equivalent to DOI with fixed  $\sigma(\mu_z) = 1$  in the reward eq. (14). In Figure 3, we measure the state-action occupancy  $d_z(s,a)$  for each latent skill  $z$  through the proxy of importance ratios  $\eta_z(s,a)$ , for different values of  $\epsilon$ . As expected, a higher value of  $\epsilon$  increases diversity, resulting in different importance ratios per skill for individual data points. We aggregate this difference by computing an average across different skills  $\ell_1$  norm of the importance ratios  $\mathbb{E}\|\eta_{z_i} - \eta_{z_j}\|_1$  and report it in Figure 3. We note that the looser the constraint (lighter color), the easier it is to "diversify" in the sense of  $\eta_z$ . In Figure 3a, we observe diversification across the dataset assignment to skills when using DOI, whereas training an ensemble of skills with only expert imitation reward (i.e.,  $\sigma(\mu_z) = 1$ ) collapses to nearly the same importance per skill per data point. Figure 3b shows the average  $\ell_1$  distance between skill importance vectors  $\eta_z$  over the dataset for  $\epsilon \in \{0.0,1.0,2.0,4.0\}$  (lighter color indicates higher  $\epsilon$ ). Moreover, the tighter the constraint (smaller  $\epsilon$ ), the smaller the difference between the different skill importance ratios.

We have further evaluated diversity on the Monte Carlo estimates of the expected successor feature of the initial state, based on 30 policy rollouts per skill. The  $\gamma$ -discounted successor features (SFs) for state  $s$  are defined as  $\psi_z(s) = \mathbb{E}_{d_z(s)}[\phi(s)]$ , where  $d_z(s)$  is the  $\gamma$ -discounted state occupancy for a skill policy  $\pi_z$ . With slight abuse of notation, we define  $\psi_z = \mathbb{E}_{\rho_0(s)}[\psi_z(s)]$ , the expected SFs over the initial state distribution. As a diversity metric, we take the average over different skills  $\ell_2$  norm between SFs, i.e.,  $\mathbb{E}\|\psi_{z_1} - \psi_{z_2}\|_2$ . The results are presented in Figure 4 and show an alignment with the proxy diversity metric, i.e. the separation of the data indicated by the importance ratios  $\eta_z$  shows a higher distance between the expected SFs  $\psi_z$ . In terms of performance, DOI is able to achieve a forward velocity comparable to the expert (see Figure 4a) while diversifying the behavior in terms of base height  $h$  (Figure 4b). We also observed that the multipliers  $\sigma(\mu_z)$  are non-zero for all skills, indicating that the constraint is active. In addition, they stabilize at reasonable levels as training progresses, which we show in Supp. G for both the SOLO12 and ANT.

For D4RL environments, we consider the case where we have offline data generated from a random policy mixed with a small amount of expert trajectories.2. Figure 5 shows the results for both the expected average SFs distance (Figure 5a) and the average importance ratio  $\eta_z$  distance across skills (Figure 5b). We normalize the state feature  $\phi(s)$  when comparing  $\psi_z$  across environments in Figure 5a. As expected, there is a trade-off between the average skill return and the respective diversity metric across skills in most cases. Furthermore, the diversity distance that is more controllable by  $\epsilon$  corresponds to the importance ratios  $\eta_z$ . This observation is in line with expectations, since  $\eta_z$  is part of the constraint. Nonetheless, in Figure 5a we show that  $\epsilon$  retains some controllability over

![](images/170f315d179d01c24bde041821763e76174ef96e8f04012f5d2327d27cc6da40.jpg)  
(a)

![](images/9312e9364c55caabcba6822c2c43103e923a67fcc420b353e33a039616588b02.jpg)  
Figure 4: Average  $\ell_2$  distance between MC estimated successor features  $\psi_z$  of different skills (a), return  $r$  as  $\%$  of expert return and standard deviation of base height  $\mathrm{std}_z(h)$  (b), depending on  $\epsilon$  for the SOLO12.  
(b)

diversity. The WALKER2D is particularly sensitive to relaxation of the occupancy constraint with respect to performance. We hypothesize that this is due to the fact that the space of policies that achieve a stable gait is very restrictive, resulting in a significant loss of task return for even slight skill diversification. In contrast, the ANT exhibits high stability, with multiple clusters achieving close to expert performance in terms of  $r$ . These results are also consistent with SMODICE expert policies used for computing  $\eta_{\widetilde{E}}$  (see Supp. F).

![](images/2142fba448aacf40daaf6319d48f3bc9871fb890437967f1844e19d3a7044ece.jpg)  
(a)  
Figure 5: Results on D4RL environments with offline data collected from a random policy for  $\epsilon = 0.0, 0.5, 1.0, 2.0, 4.0$ . In figure (a) we observe the tradeoff between average skill return and average successor features distance over skills. In figure (b), we report the tradeoff w.r.t. average  $\ell_1$  distance of importance ratios  $\eta_z$ .

![](images/1d82b5b3dcfed96f9a10b26e5ed20946eb888f61324eb3fa1b08b6fd9d6264d3.jpg)  
(b)

# 7 CONCLUSION

We proposed DOI, a principled offline RL algorithm for unsupervised skill discovery that, in addition to maximizing diversity, ensures that each learned skill imitates state-only expert demonstrations to a certain degree. Our main analytical contribution is to connect Fenchel duality, reinforcement learning, and unsupervised skill discovery to maximize a mutual information objective subject to KL-divergence state occupancy constraints. We have shown that DOI can diversify offline policies for a 12-DoF quadruped robot (in simulation and in reality) and for several environments from the standard D4RL benchmark in terms of both  $\ell_2$  distance of expected successor features and  $\ell_1$  distance of importance ratios, which is visible from the data separation induced by  $\eta_z(s,a)$  amongst skills. The importance ratio distance, computed offline, is a robust indicator of diversity, which aligns with the online Monte Carlo diversity metric of expected successor features. The resulting skill diversity naturally entails a trade-off in task performance. We can control the amount of diversity via a KL constraint level  $\epsilon$ , which ensures that different skills remain close to the expert in terms of state-action occupancy, which also indirectly controls task performance loss. A promising direction for future research is to impose constraints on the value function of each skill to ensure near-optimal task performance.

# 8 REPRODUCIBILITY

For implementation of DOI we have used the PyTorch autograd framework. For the SOLO12 training we made use of Isaac Gym for data collection and evaluation of the learned skill policies. For the D4RL experiments we evaluated the policies using the Mujoco v2.1 rigid body simulator. The training of the skill policies with evaluation and pre-training of the SMODICE expert ratios takes about 4 hours on an NVIDIA GeForce RTX 4080 graphics card with a batch size of 512. We plan on opensourcing the code and the SOLO12 data post conference acceptance. The SOLO12 robot has been developed as part of the Open Dynamic Robot Initiative (Grimminger et al., 2020), and a full assembly kit is available at a cheap price in order to reproduce the real system experiments from Supp. H.

# REFERENCES

Robohive - a unified framework for robot learning. https://sites.google.com/view/robohive, 2020. URL https://sites.google.com/view/robohive.  
J. Achiam, H. Edwards, D. Amodei, and P. Abbeel. Variational option discovery algorithms. CoRR, abs/1807.10299, 2018. URL http://arxiv.org/abs/1807.10299.  
E. Altman. Constrained Markov decision processes, volume 7. CRC Press, 1999.  
A. Barreto, W. Dabney, R. Munos, J. J. Hunt, T. Schaul, H. Van Hasselt, and D. Silver. Successor features for transfer in reinforcement learning. arXiv preprint arXiv:1606.05312, 2016.  
A. Barreto, D. Borsa, J. Quan, T. Schaul, D. Silver, M. Hessel, D. Mankowitz, A. Zidek, and R. Munos. Transfer in deep reinforcement learning using successor features and generalised policy improvement. In International Conference on Machine Learning, pages 501-510. PMLR, 2018.  
S. Bhatnagar and K. Lakshmanan. An online actor-critic algorithm with function approximation for constrained markov decision processes. Journal of Optimization Theory and Applications, 153 (3):688-708, 2012.  
V. S. Borkar. An actor-critic algorithm for constrained markov decision processes. Systems & control letters, 54(3):207-213, 2005.  
S. P. Boyd and L. Vandenberghe. Convex optimization. Cambridge university press, 2004.  
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. S. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. RT-2: vision-language-action models transfer web knowledge to robotic control. CoRR, abs/2307.15818, 2023. doi: 10.48550/arXiv.2307.15818. URL https://doi.org/10.48550/arXiv.2307.15818.  
V. Campos, A. Trott, C. Xiong, R. Socher, X. Giro-i-Nieto, and J. Torres. Explore, discover and learn: Unsupervised discovery of state-covering skills. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 1317-1327. PMLR, 2020. URL http://proceedings.mlr.press/v119/campos20a.html.  
C. Cheng, T. Xie, N. Jiang, and A. Agarwal. Adversarily trained actor critic for offline reinforcement learning. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 3852-3878. PMLR, 2022.  
B. Dai, N. He, Y. Pan, B. Boots, and L. Song. Learning from conditional distributions via dual embeddings. In Artificial Intelligence and Statistics, pages 1458-1467. PMLR, 2017.

B. Dai, O. Nachum, Y. Chow, L. Li, C. Szepesvári, and D. Schuurmans. Coindex: Off-policy confidence interval estimation. Advances in neural information processing systems, 33:9398-9411, 2020.  
P. Dayan. Improving generalization for temporal difference learning: The successor representation. Neural Computation, 5(4):613-624, 1993. doi: 10.1162/neco.1993.5.4.613.  
B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine. Diversity is all you need: Learning skills without a reward function. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=SJx63jRqFm.  
J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.  
A. Goyal, R. Islam, D. Strouse, Z. Ahmed, H. Larochelle, M. M. Botvinick, Y. Bengio, and S. Levine. Infobot: Transfer and exploration via the information bottleneck. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=rJg8yhAqKm.  
K. Gregor, D. J. Rezende, and D. Wierstra. Variational intrinsic control. In 5th International Conference on Learning Representations, ICLR 2017, Toulouse, France, April 24-26, 2017, Workshop Track Proceedings. OpenReview.net, 2017. URL https://openreview.net/forum?id= Skc-Fo4Yg.  
F. Grimminger, A. Meduri, M. Khadiv, J. Viereck, M. Wüthrich, M. Naveau, V. Berenz, S. Heim, F. Widmaier, T. Flayols, J. Fiene, A. Badri-Spröwitz, and L. Righetti. An open torque-controlled modular robot architecture for legged locomotion research. *IEEE Robotics and Automation Letters*, 5(2):3650–3657, 2020.  
I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5767-5777, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/892c3b1c6dcccd52936e27cbd0ff683d6-Abstract.html.  
S. Hansen, W. Dabney, A. Barreto, D. Warde-Farley, T. V. de Wiele, and V. Mnih. Fast task inference with variational intrinsic successor features. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=BJeAHkrYDS.  
R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck, and P. Abbeel. Vime: Variational information maximizing exploration. In Advances in Neural Information Processing Systems, 2016.  
G.-H. Kim, S. Seo, J. Lee, W. Jeon, H. Hwang, H. Yang, and K.-E. Kim. DemoDICE: Offline imitation learning with supplementary imperfect demonstrations. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=BrPdX1bDZkQ.  
J. Kim, S. Park, and G. Kim. Unsupervised skill discovery with bottleneck option learning. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139, pages 5572-5582. PMLR, 2021a. URL http://proceedings.mlr.press/v139/kim21j.html.  
J. Kim, S. Park, and G. Kim. Unsupervised skill discovery with bottleneck option learning. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 5572-5582. PMLR, 2021b. URL http://proceedings.mlr.press/v139/kim21j.html.  
A. Klyubin, D. Polani, and C. Nehaniv. Empowerment: a universal agent-centric measure of control. In IEEE Congress on Evolutionary Computation, volume 1, pages 128-135 Vol.1, 2005. URL https://ieeexplore.ieee.org/document/1554676.

I. Kostrikov, A. Nair, and S. Levine. Offline reinforcement learning with implicit q-learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=68n2s9ZJWF8.  
A. Kumar, A. Zhou, G. Tucker, and S. Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179-1191, 2020.  
J. Lee, W. Jeon, B. Lee, J. Pineau, and K.-E. Kim. Optidice: Offline policy optimization via stationary distribution correction estimation. In International Conference on Machine Learning, pages 6120-6130. PMLR, 2021.  
J. Lee, C. Paduraru, D. J. Mankowitz, N. Heess, D. Precup, K.-E. Kim, and A. Guez. Coptidice: Offline constrained reinforcement learning via stationary distribution correction estimation. In International Conference on Learning Representations, 2022.  
S. Levine, A. Kumar, G. Tucker, and J. Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. CoRR, abs/2005.01643, 2020.  
P.-A. Léziart, T. Flayols, F. Grimminger, N. Mansard, and P. Souères. Implementation of a reactive walking controller for the new open-hardware quadruped solo-12. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 5007-5013. IEEE, 2021.  
Q. Liu, L. Li, Z. Tang, and D. Zhou. Breaking the curse of horizon: Infinite-horizon off-policy estimation. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 5361-5371, 2018. URL https://proceedings.neurips.cc/paper/2018/bitstream/dda04f9d634145a9c68d5dbe53b21272-Abstract.html.  
Y. J. Ma, A. Shen, D. Jayaraman, and O. Bastani. Versatile offline imitation from observations and examples via regularized state-occupancy matching. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 14639-14663. PMLR, 2022. URL https://proceedings.mlr.press/v162/ma22a.html.  
V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, et al. Isaac gym: High performancegpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021.  
S. Mohamed and D. Jimenez Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2015. URL https://proceedings.neurips.cc/paper/2015/bit/ e00406144c1e7e35240afed70f34166a-Abstract.html.  
A. Mousavi, L. Li, Q. Liu, and D. Zhou. Black-box off-policy estimation for infinite-horizon reinforcement learning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=S11tg1rFDS.  
O. Nachum and B. Dai. Reinforcement learning via fenchel-rockafellar duality. arXiv preprint arXiv:2001.01866, 2020.  
O. Nachum, Y. Chow, B. Dai, and L. Li. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in Neural Information Processing Systems, 32, 2019a.  
O. Nachum, B. Dai, I. Kostrikov, Y. Chow, L. Li, and D. Schuurmans. Algaedice: Policy gradient from arbitrary experience, 2019b.  
A. Nair, M. Dalal, A. Gupta, and S. Levine. Accelerating online reinforcement learning with offline datasets. CoRR, abs/2006.09359, 2020.

S. Park and S. Levine. Predictable MDP abstraction for unsupervised model-based RL. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 27246-27268. PMLR, 2023. URL https://proceedings.mlr.press/v202/park23i.html.  
S. Park, J. Choi, J. Kim, H. Lee, and G. Kim. Lipschitz-constrained unsupervised skill discovery. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=BGvt0ghNgA.  
S. Park, K. Lee, Y. Lee, and P. Abbeel. Controllability-aware unsupervised skill discovery. CoRR, abs/2302.05103, 2023. doi: 10.48550/arXiv.2302.05103. URL https://doi.org/10.48550/arXiv.2302.05103.  
R. F. Prudencio, M. R. O. A. Maximo, and E. L. Colombini. A survey on offline reinforcement learning: Taxonomy, review, and open problems. CoRR, abs/2203.01387, 2022.  
M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.  
A. Sharma, S. Gu, S. Levine, V. Kumar, and K. Hausman. Dynamics-aware unsupervised discovery of skills. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=HJgLZR4KvH.  
A. Stooke, J. Achiam, and P. Abbeel. Responsive safety in reinforcement learning by PID lagrangian methods. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 9133-9143. PMLR, 2020. URL http://proceedings.mlr.press/v119/stooke20a.html.  
D. Strouse, K. Baumli, D. Warde-Farley, V. Mnih, and S. S. Hansen. Learning more skills through optimistic exploration. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=cU8rknuhxc.  
C. Szepesvári. Constrained mdps and the reward hypothesis. Musings about machine learning and other things (blog), 2020. URL https://readingsml.blogspot.com/2020/03/constrained-mdps-and-reward-hypothesis.html.  
C. Tessler, D. J. Mankowitz, and S. Mannor. Reward constrained policy optimization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=SkfrvsA9FX.  
N. Tishby, F. C. Pereira, and W. Bialek. The information bottleneck method. In Proc. of the 37-th Annual Allerton Conference on Communication, Control and Computing, pages 368-377, 1999. URL https://arxiv.org/abs/physics/0004057.  
H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023.  
Z. Wang, A. Novikov, K. Zolna, J. S. Merel, J. T. Springenberg, S. E. Reed, B. Shahriari, N. Siegel, C. Gulcehre, N. Heess, et al. Critic regularized regression. Advances in Neural Information Processing Systems, 33:7768-7778, 2020.  
H. Xu, X. Zhan, J. Li, and H. Yin. Offline reinforcement learning with soft behavior regularization. CoRR, abs/2110.07395, 2021. URL https://arxiv.org/abs/2110.07395.  
H. Xu, L. Jiang, J. Li, Z. Yang, Z. Wang, W. K. V. Chan, and X. Zhan. Offline RL with no OOD actions: In-sample learning via implicit value regularization. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=ueYYgo2pSSU.

T. Zahavy, Y. Schroecker, F. M. P. Behbahani, K. Baumli, S. Flennerhag, S. Hou, and S. Singh. Discovering policies with domino: Diversity optimization maintaining near optimality. CoRR, abs/2205.13521, 2022. doi: 10.48550/arXiv.2205.13521. URL https://doi.org/10.48550/arXiv.2205.13521.  
S. Zhang, B. Liu, and S. Whiteson. Gradientdice: Rethinking generalized offline estimation of stationary values. In International Conference on Machine Learning, pages 11194-11203. PMLR, 2020.
