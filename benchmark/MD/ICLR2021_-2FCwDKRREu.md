# LEARNING INVARIANT REPRESENTATIONS FOR REINFORCEMENT LEARNING WITHOUT RECONSTRUCTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study how representation learning can accelerate reinforcement learning from rich observations, such as images, without relying either on domain knowledge or pixel-reconstruction. Our goal is to learn representations that both provide for effective downstream control and invariance to task-irrelevant details. Bisimulation metrics quantify behavioral similarity between states in continuous MDPs, which we propose using to learn robust latent representations which encode only the task-relevant information from observations. Our method trains encoders such that distances in latent space equal bisimulation distances in state space. We demonstrate the effectiveness of our method at disregarding task-irrelevant information using modified visual MuJoCo tasks, where the background is replaced with moving distractors and natural videos, while achieving SOTA performance. We also test a first-person highway driving task where our method learns invariance to clouds, weather, and time of day. Finally, we provide generalization results drawn from properties of bisimulation metrics, and links to causal inference.

# 1 Introduction

Learning control from images is important for many real world applications. While deep reinforcement learning (RL) has enjoyed many successes in simulated tasks, learning control from real vision is more complex, especially outdoors, where images reveal detailed scenes of a complex and unstructured world. Furthermore, while many RL algorithms can eventually learn control from real images given unlimited data, data-efficiency is often a necessity in real trials which are expensive and constrained to real-time. Prior methods for data-efficient learning of simulated visual tasks typically use representation learning. Representation learning summarizes images by encoding them into smaller vectored representations better suited for RL. For example, sequential autoencoders aim to learn lossless representations of streaming observations—sufficient to reconstruct current observations and predict future observations—from which various RL algorithms can be trained (Hafner et al., 2018; Lee et al., 2019; Yarats et al., 2019). However, such methods are task-agnostic: the models represent all dynamic elements they observe in the world,

whether they are relevant to the task or not. We argue such representations can easily "distract" RL algorithms with irrelevant information in the case of real images. The issues of distraction is less evident in popular simulation MuJoCo and Atari tasks, since any change in observation space is likely task-relevant, and thus, worth representing. By contrast, visual images that autonomous cars observe contain predominately task-irrelevant information, like cloud shapes and architectural details, illustrated in Figure 1.

Rather than learning control-agnostic representations that focus on accurate reconstruction of clouds and buildings, we would rather achieve a more compressed representation from a lossy encoder, which only retains state information relevant to our task. If we would like to learn representations that capture only task-relevant elements of the state and are invariant to task-irrelevant information, intuitively we can utilize the reward signal to determine task-relevance. As cumulative rewards are our objective, state elements are relevant not only if they influence the current reward, but also if they

![](images/53b71492f36529bfb210c7606f4efbf2798509ba10e1ade71e158d707d8abbba.jpg)  
Figure 1: Robust representations of the visual scene should be insensitive to irrelevant objects (e.g., clouds) or details (e.g., cartypes), and encode two observations equivalently if their relevant details are equal (e.g., road direction and locations of other cars).

influence state elements in the future that in turn influence future rewards. This recursive relationship can be distilled into a recursive task-aware notion of state abstraction: an ideal representation is one that is predictive of reward, and also predictive of itself in the future.

We propose learning such an invariant representation using the bisimulation metric, where the distance between two observation encodings correspond to how "behaviourally different" (Ferns & Precup, 2014) both observations are. Our main contribution is a practical representation learning method based on the bisimulation metric suitable for downstream control, which we call deep bisimulation for control (DBC). We additionally provide theoretical analysis that proves value bounds between the optimal value function of the true MDP and the optimal value function of the MDP constructed by the learned representation. Empirical evaluations demonstrate our nonreconstructive using bisimulation approach is substantially more robust to task-irrelevant distractors when compared to prior approaches that use reconstruction losses or contrastive losses. Our initial experiments insert natural videos into the background of MoJoCo control task as complex distraction. Our second setup is a high-fidelity highway driving task using CARLA (Dosovitskiy et al., 2017), showing that our representations can be trained effectively even on highly realistic images with many distractions, such as trees, clouds, buildings, and shadows. For example videos see https://sites.google.com/view/deepbisim4control.

# 2 Related Work

Our work builds on the extensive prior research on bisimulation in MDP state aggregation.

Reconstruction-based Representations. Early works on deep reinforcement learning from images (Lange & Riedmiller, 2010; Lange et al., 2012) used a two-step learning process where first an auto-encoder was trained using reconstruction loss to learn a low-dimensional representation, and subsequently a controller was learned using this representation. This allows effective leveraging of large, unlabeled datasets for learning representations for control. In practice, there is no guarantee that the learned representation will capture useful information for the control task, and significant expert knowledge and tricks are often necessary for these approaches to work. In model-based RL, one solution to this problem has been to jointly train the encoder and the dynamics model end-to-end Watter et al. (2015); Wahlström et al. (2015) – this proved effective in learning useful task-oriented representations. Hafner et al. (2018) and Lee et al. (2019) learn latent state models using a reconstruction loss, but these approaches suffer from the difficulty of learning accurate long-term predictions and often still require significant manual tuning. Gelada et al. (2019) also propose a latent dynamics model-based method and connect their approach to bisimulation metrics, using a reconstruction loss in Atari. They show that  $\ell_2$  distance in the DeepMDP representation upper bounds the bisimulation distance, whereas our objective directly learns a representation where distance in latent space is the bisimulation metric. Further, their results rely on the assumption that the learned representation is Lipschitz, whereas we show that, by directly learning a bisimilarity-based representation, we guarantee a representation that generates a Lipschitz MDP. We show experimentally that our non-reconstructive DBC method is substantially more robust to complex distractors.

Contrastive-based Representations. Contrastive losses are a self-supervised approach to learn useful representations by enforcing similarity constraints between data (van den Oord et al., 2018; Chen et al., 2020). Similarity functions can be provided as domain knowledge in the form of heuristic data augmentation, where we maximize similarity between augmentations of the same data point (Laskin et al., 2020) or nearby image patches (Hénaff et al., 2019), and minimize similarity between different data points. In the absence of this domain knowledge, contrastive representations can be trained by predicting the future (van den Oord et al., 2018). We compare to such an approach in our experiments, and show that DBC is substantially more robust. While contrastive losses do not require reconstruction, they do not inherently have a mechanism to determine downstream task relevance without manual engineering, and when trained only for prediction, they aim to capture all predictable features in the observation, which performs poorly on real images for the same reasons world models do. A better method would be to incorporate knowledge of the downstream task into the similarity function in a data-driven way, so that images that are very different pixel-wise (e.g. lighting or texture changes), can also be grouped as similar w.r.t. downstream objectives.

Bisimulation. Various forms of state abstractions have been defined in Markov decision processes (MDPs) to group states into clusters whilst preserving some property (e.g. the optimal value, or all

values, or all action values from each state) (Li et al., 2006). The strictest form, which generally preserves the most properties, is bisimulation (Larsen & Skou, 1989). Bisimulation only groups states that are indistinguishable w.r.t. reward sequences output given any action sequence tested. A related concept is bisimulation metrics (Ferns & Precup, 2014), which measure how "behaviorally similar" states are. Ferns et al. (2011) defines the bisimulation metric with respect to continuous MDPs, and propose a Monte Carlo algorithm for learning it using an exact computation of the Wasserstein distance between empirically measured transition distributions. However, this method does not scale well to large state spaces. Taylor et al. (2009) relate MDP homomorphisms to lax probabilistic bisimulation, and define a lax bisimulation metric. They then compute a value bound based on this metric for MDP homomorphisms, where approximately equivalent state-action pairs are aggregated. Most recently, Castro (2020) propose an algorithm for computing on-policy bisimulation metrics, but does so directly, without learning a representation. They focus on deterministic settings and the policy evaluation problem. We believe our work is the first to propose a gradient-based method for directly learning a representation space with the properties of bisimulation metrics and show that it works in the policy optimization setting.

# 3 Preliminaries

We start by introducing notation and outlining realistic assumptions about underlying structure in the environment. Then, we review state abstractions and metrics for state similarity.

We assume the underlying environment is a Markov decision process (MDP), described by the tuple  $\mathcal{M} = (\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)$ , where  $\mathcal{S}$  is the state space,  $\mathcal{A}$  the action space,  $\mathcal{P}(\mathbf{o}'|\mathbf{o},\mathbf{a})$  the probability of transitioning from state  $\mathbf{o} \in \mathcal{S}$  to state  $\mathbf{o}' \in \mathcal{S}$ , and  $\gamma \in [0,1)$  a discount factor. An "agent" chooses actions  $\mathbf{a} \in \mathcal{A}$  according to a policy function  $\mathbf{a} \sim \pi(\mathbf{o})$ , which updates the system state  $\mathbf{o}' \sim \mathcal{P}(\mathbf{o},\mathbf{a})$ , yielding a reward  $r = \mathcal{R}(\mathbf{o}) \in \mathbb{R}$ . The agent's goal is to maximize the expected cumulative discounted rewards by learning a good policy:  $\max_{\pi} \mathbb{E}_{\mathcal{P}}[\sum_{t=0}^{\infty} [\gamma^t \mathcal{R}(\mathbf{o}_t)]]$ .

Bisimulation is a form of state abstraction that groups states  $\mathbf{s}_i$  and  $\mathbf{s}_j$  that are "behaviorally equivalent" (Li et al., 2006). For any action sequence  $\mathbf{a}_{0:\infty}$ , the probabilistic sequence of rewards from  $\mathbf{s}_i$  and  $\mathbf{s}_j$  are identical. A more compact definition has a recursive form: two states are bisimilar if they share both the same immediate reward and equivalent distributions over the next bisimilar states (Larsen & Skou, 1989; Givan et al., 2003).

Definition 1 (Bisimulation Relations (Givan et al., 2003)). Given an MDP  $\mathcal{M}$ , an equivalence relation  $B$  between states is a bisimulation relation if, for all states  $\mathbf{o}_i, \mathbf{o}_j \in S$  that are equivalent under  $B$  (denoted  $\mathbf{o}_i \equiv_B \mathbf{o}_j$ ) the following conditions hold:

$$
\mathcal {\check {R}} (\mathbf {o} _ {i}, \mathbf {a}) = \mathcal {\check {R}} (\mathbf {o} _ {j}, \mathbf {a}) \quad \forall \mathbf {a} \in \mathcal {A}, \tag {1}
$$

$$
\mathcal {P} (G | \mathbf {o} _ {i}, \mathbf {a}) = \mathcal {P} (G | \mathbf {o} _ {j}, \mathbf {a}) \quad \forall \mathbf {a} \in \mathcal {A}, \quad \forall G \in \mathcal {S} _ {B}, \tag {2}
$$

where  $S_B$  is the partition of  $S$  under the relation  $B$  (the set of all groups  $G$  of equivalent states), and  $\mathcal{P}(G|\mathbf{o},\mathbf{a}) = \sum_{\mathbf{o}'\in G}\mathcal{P}(\mathbf{o}'|\mathbf{o},\mathbf{a})$ .

Exact partitioning with bisimulation relations is generally impractical in continuous state spaces, as the relation is highly sensitive to infinitesimal changes in the reward function or dynamics. For this reason, Bisimulation Metrics (Ferns et al., 2011; Ferns & Precup, 2014; Castro, 2020) softens the concept of state partitions, and instead defines a pseudometric space  $(S,d)$ , where a distance function  $d:S\times S\mapsto \mathbb{R}_{\geq 0}$  measures the "behavioral similarity" between two states<sup>1</sup>. Defining a distance  $d$  between states requires defining both a distance between rewards (to soften Equation (1)), and distance between state distributions (to soften Equation (2)). Prior works use the Wasserstein metric for the latter, originally used in the context of bisimulation metrics by van Breugel & Worrell (2001). The  $p^{\mathrm{th}}$  Wasserstein metric is defined between two probability distributions  $\mathcal{P}_i$  and  $\mathcal{P}_j$  as  $W_{p}(\mathcal{P}_{i},\mathcal{P}_{j};d) = (\inf_{\gamma^{\prime}\in \Gamma (\mathcal{P}_{i},\mathcal{P}_{j})}\int_{S\times S}d(\mathbf{s}_{i},\mathbf{s}_{j})^{p}\mathrm{d}\gamma^{\prime}(\mathbf{s}_{i},\mathbf{s}_{j}))^{1 / p}$ , where  $\Gamma (\mathcal{P}_i,\mathcal{P}_j)$  is the set of all couplings of  $\mathcal{P}_i$  and  $\mathcal{P}_j$ . This is known as the "earth mover" distance, denoting the cost of transporting mass from one distribution to another (Villani, 2003). Finally, the bisimulation metric is the reward difference added to the Wasserstein distance between transition distributions:

Definition 2 (Bisimulation Metric). From Theorem 2.6 in Ferns et al. (2011) with  $c \in [0,1)$ :

$$
d \left(\mathbf {o} _ {i}, \mathbf {o} _ {j}\right) = \max  _ {\mathbf {a} \in \mathcal {A}} (1 - c) \cdot \left| \mathcal {R} _ {\mathbf {o} _ {i}} ^ {\mathbf {a}} - \mathcal {R} _ {\mathbf {o} _ {j}} ^ {\mathbf {a}} \right| + c \cdot W _ {1} \left(\mathcal {P} _ {\mathbf {o} _ {i}} ^ {\mathbf {a}}, \mathcal {P} _ {\mathbf {o} _ {j}} ^ {\mathbf {a}}; d\right). \tag {3}
$$

# 4 Learning Representations for Control with Bisimulation Metrics

![](images/a697807a34004f10f899ca2c09fc3db3c498ac88f6b6e5285ea71e3fb1a52a80.jpg)  
Figure 2: Our method learns a bisimulation metric representation. Shaded in blue is the main model architecture, it is reused for both states, like a Siamese network. The loss is a weighted sum of the reward and transition distribution distances (using the Wasserstein metric  $W$ ).

<table><tr><td colspan="2">Algorithm 1 Deep Bisimulation for Control (DBC)</td></tr><tr><td>1:</td><td>for Time t = 0 to ∞ do</td></tr><tr><td>2:</td><td>Encode observation z_t = φ(ot)</td></tr><tr><td>3:</td><td>Execute action at ~ π(t)</td></tr><tr><td>4:</td><td>Record data: D ← D ∪ {ot, at, ot+1, rt+1}</td></tr><tr><td>5:</td><td>Sample batch Bi ~ D</td></tr><tr><td>6:</td><td>Permute batch randomly: Bj = permute(Bi)</td></tr><tr><td>7:</td><td>Train policy: EB_i [J(π)] ▷ Algorithm 2</td></tr><tr><td>8:</td><td>Train encoder: EB_i, B_j [J(φ)] ▷ Equation (4)</td></tr><tr><td>9:</td><td>Train dynamics model: 
J(ˆ, φ) = EB_t+1 ~ˆ(φ(ot), at) [({ˆt+1 - ŵt+1})^2]</td></tr><tr><td>10:</td><td>Train reward model: 
J(ˆR,ˆP, φ) = EB_t+1 ~ˆ(φ(ot), at) [({ˆR(ˆt+1) - rt+1})</td></tr></table>

We propose Deep Bisimulation for Control (DBC), a data-efficient approach to learn control policies from unstructured, high-dimensional observations. In contrast to prior work on bisimulation, which typically aims to learn a distance function of the form  $d: S \times S \mapsto \mathbb{R}_{\geq 0}$  between observations, our aim is instead to learn representations  $\mathcal{Z}$  under which  $\ell_1$  distances correspond to bisimulation metrics, and then use these representations to improve reinforcement learning. Our goal is to learn encoders  $\phi: S \mapsto \mathcal{Z}$  that capture representations of states that are suitable to control, while discarding any information that is irrelevant for control. Any representation that relies on reconstruction of the observation cannot do this, as these irrelevant details are still important for reconstruction. We hypothesize that bisimulation metrics can acquire this type of representation, without any reconstruction.

Bisimulation metrics are a useful form of state abstraction, but prior methods to train distance functions either do not scale to pixel observations (Ferns et al., 2011) (due to the max operator in Equation (3)), or were only designed for the (fixed) policy evaluation setting (Castro, 2020). By contrast, we learn improved representations for policy inputs, as the (non-fixed) policy improves. Our  $\pi^{*}$ -bisimulation metric is learned with gradient decent, and we prove it converges to a fixed point in Theorem 1 under some assumptions. To train our encoder  $\phi$  towards our desired relation  $d(\mathbf{o}_i,\mathbf{o}_j)\coloneqq ||\phi (\mathbf{o}_i) - \phi (\mathbf{o}_j)||_1$ , we draw batches of observations pairs, and minimise the mean square error between the on-policy bisimulation metric and Euclidean distance in the latent space:

$$
J (\phi) = \left(\left\| \mathbf {z} _ {i} - \mathbf {z} _ {j} \right\| _ {1} - \left| \hat {\mathcal {R}} (\bar {\mathbf {z}} _ {i}) - \hat {\mathcal {R}} (\bar {\mathbf {z}} _ {j}) \right| - \gamma \cdot W _ {2} \left(\hat {\mathcal {P}} (\cdot | \bar {\mathbf {z}} _ {i}, \bar {\pi} (\bar {\mathbf {z}} _ {i})) , \hat {\mathcal {P}} (\cdot | \bar {\mathbf {z}} _ {j}, \bar {\pi} (\bar {\mathbf {z}} _ {j}))\right)\right) ^ {2}, \tag {4}
$$

where  $\mathbf{z}_i = \phi(\mathbf{o}_i)$ ,  $\mathbf{z}_j = \phi(\mathbf{o}_j)$ ,  $\bar{\mathbf{z}}$  denotes  $\phi(\mathbf{o})$  with stop gradients, and  $\bar{\pi}$  is the mean policy output. Equation (4) uses both a reward model  $\hat{\mathcal{R}}$  and dynamics model  $\hat{\mathcal{P}}$ , which have their own training steps in Algorithm 1. The full model architecture and training is illustrated by Figure 2. Our reward model is a deterministic network, and our dynamics model  $\hat{\mathcal{P}}$  outputs a Gaussian distribution. For this reason, we use the 2-Wasserstein metric  $W_2$  in Equation (4), as opposed to the 1-Wasserstein in Equation (3), since the  $W_2$  metric has a convenient closed form:  $W_2(\mathcal{N}(\mu_i, \Sigma_i), \mathcal{N}(\mu_j, \Sigma_j))^2 = ||\mu_i - \mu_j||_2^2 + ||\Sigma_i^{1/2} - \Sigma_j^{1/2}||_{\mathcal{F}}^2$ , where  $||\cdot||_{\mathcal{F}}$  is the Frobenius norm. For all other distances we continue using the  $\ell_1$  norm.

Incorporating control. We combine our representation learning approach (Algorithm 1) with the soft actor-critic (SAC) algorithm (Haarnoja et al., 2018) to devise a practical reinforcement learning method. We modified SAC slightly in Algorithm 2 to allow the value function to backprop to our encoder, which

# Algorithm 2 Train Policy (changes to SAC in blue)

1: Get value:  $V = \min_{i=1,2} \hat{Q}_i(\hat{\phi}(\mathbf{o})) - \alpha \log \pi(\mathbf{a}|\phi(\mathbf{o}))$  
2: Train critics:  $J(Q_{i},\phi) = (Q_{i}(\phi (\mathbf{o})) - r - \gamma V)^{2}$  
3: Train actor:  $J(\pi) = \alpha \log p(\mathbf{a}|\phi (\mathbf{o})) - \min_{i = 1,2}Q_i(\phi (\mathbf{o}))$  
4: Train alpha:  $J(\alpha) = -\alpha \log p(\mathbf{a}|\phi (\mathbf{o}))$  
5: Update target critics:  $\hat{Q}_i\gets \tau_QQ_i + (1 - \tau_Q)\hat{Q}_i$  
6: Update target encoder:  $\phi \gets \tau_{\phi}\phi +(1 - \tau_{\phi})\phi$

can improve performance further (Yarats et al., 2019; Rakelly et al., 2019). Although, in principle, our method could be combined with any RL algorithm, including the model-free DQN (Mnih et al., 2015), or model-based PETS (Chua et al., 2018). Implementation details and hyperparameter values of DBC are summarized in the appendix, Table 2. We train DBC by iteratively updating four components in turn: a dynamics model  $\hat{\mathcal{P}}$ , reward model  $\hat{\mathcal{R}}$ , encoder  $\phi$  with Equation (4), and policy  $\pi$  (in this case, with SAC). A single loss function would be less stable, and require balancing components. The inputs of each loss function  $J(\cdot)$  in Algorithm 1 represents which components are updated. After each training step, the policy  $\pi$  is used to step in the environment, the data is collected in a replay buffer  $\mathcal{D}$ , and a batch is randomly selected to repeat training.

# 5 Generalization Bounds and Links to Causal Inference

While DBC enables representation learning without pixel reconstruction, it leaves open the question of how good the resulting representations really are. In this section, we present theoretical analysis that bounds the suboptimality of a value function trained on the representation learned via DBC. First, we show that our  $\pi^{*}$ -bisimulation metric converges to a fixed point, starting from the initialized policy  $\pi_0$  and converging to an optimal policy  $\pi^{*}$ .

Theorem 1. Let  $\mathfrak{met}$  be the space of bounded pseudometrics on  $S$  and  $\pi$  a policy that is continuously improving. Define  $\mathcal{F}:\mathfrak{met}\mapsto \mathfrak{met}$  by

$$
\mathcal {F} (d, \pi) \left(\mathbf {o} _ {i}, \mathbf {o} _ {j}\right) = (1 - c) \left| r _ {\mathbf {o} _ {i}} ^ {\pi} - r _ {\mathbf {o} _ {j}} ^ {\pi} \right| + c W (d) \left(\mathcal {P} _ {\mathbf {o} _ {i}} ^ {\pi}, \mathcal {P} _ {\mathbf {o} _ {j}} ^ {\pi}\right). \tag {5}
$$

Then  $\mathcal{F}$  has a least fixed point  $\tilde{d}$  which is a  $\pi^{*}$ -bisimulation metric.

Proof in appendix. As evidenced by Definition 2, the bisimulation metric has no direct dependence on the observation space. Pixels can change, but bisimilarity will stay the same. Instead, bisimilarity is grounded in a recursion of future transition probabilities and rewards, which is closely related to the optimal value function. In fact, the bisimulation metric gives tight bounds on the optimal value function with discount factor  $\gamma$ . We show this using the property that the optimal value function is Lipschitz with respect to the bisimulation metric, see Theorem 5 in Appendix (Ferns et al., 2004). This result also implies that the closer two states are in terms of  $\tilde{d}$ , the more likely they are to share the same optimal actions. This leads us to a generalization bound on the optimal value function of an MDP constructed from a representation space using bisimulation metrics,  $||\phi(\mathbf{o}_i) - \phi(\mathbf{o}_j)||_2 \coloneqq \tilde{d}(\mathbf{o}_i, \mathbf{o}_j)$ . We can construct a partition of this space for some  $\epsilon > 0$ , giving us  $n$  partitions where  $\frac{1}{n} < (1 - c)\epsilon$ . We denote  $\phi$  as the encoder that maps from the original state space  $S$  to each  $\epsilon$ -cluster.

Theorem 2 (Value bound based on bisimulation metrics). Given an MDP  $\tilde{\mathcal{M}}$  constructed by aggregating states in an  $\epsilon$ -neighborhood, and an encoder  $\phi$  that maps from states in the original MDP  $\mathcal{M}$  to these clusters, the optimal value functions for the two MDPs are bounded as

$$
\left| V ^ {*} (\mathbf {o}) - V ^ {*} \left(\dot {\phi} (\mathbf {o})\right) \right| \leq \frac {2 \epsilon}{(1 - \gamma) (1 - c)}. \tag {6}
$$

Proof in appendix. As  $\epsilon \to 0$  the optimal value function of the aggregated MDP converges to the original. Further, by defining a learning error for  $\phi$ ,  $\mathcal{L} := \sup_{\mathbf{o}_i, \mathbf{o}_j \in S} \left| \left| \left| \phi(\mathbf{o}_i) - \phi(\mathbf{o}_j) \right| \right|_2 - \tilde{d}(\mathbf{o}_i, \mathbf{o}_j) \right|$ , we can update the bound in Theorem 2 to incorporate  $\mathcal{L}$ :  $|V^*(\mathbf{o}) - V^*(\phi(\mathbf{o}))| \leq \frac{2\epsilon + 2\mathcal{L}}{(1 - \gamma)(1 - c)}$ .

MDP dynamics have a strong connection to causal inference and causal graphs, which are directed acyclic graphs (Jonsson & Barto, 2006; Scholkopf, 2019; Zhang et al., 2020). Specifically, the state and action at time  $t$  causally affect the next state at time  $t + 1$ . In this work, we care about the components of the state space that causally affect current and future reward. Deep bisimulation for control representations connect to causal feature sets, or the minimal feature set needed to predict a target variable (Zhang et al., 2020).

Theorem 3 (Connections to causal feature sets (Thm 1 in Zhang et al. (2020))). If we partition observations using the bisimulation metric, those clusters (a bisimulation partition) correspond to the causal feature set of the observation space with respect to current and future reward.

This connection tells us that these features are the minimal sufficient statistic of the current and future reward, and therefore consist of (and only consist of) the causal ancestors of the reward variable  $\mathcal{R}$ .

Definition 3 (Causal Ancestors). In a causal graph where nodes correspond to variables and directed edges between a parent node  $P$  and child node  $C$  are causal relationships, the causal ancestors  $AN(C)$  of a node are all nodes in the path from  $C$  to a root node.

![](images/93d67c66a6f62628fb1eddf5ca88982108e9bc0a507d889eaa124383de2fe8e7.jpg)  
Figure 3: Causal graph of two time steps. Reward depends only on  $\mathbf{o}^1$  as a causal parent, but  $\mathbf{o}^1$  causally depends on  $\mathbf{o}^2$ , so AN(R) is the set  $\{\mathbf{o}^1, \mathbf{o}^2\}$ .

If there are interventions on distractor variables, or variables that control the rendering function  $q$  and therefore the rendered observation but do not affect the reward, the causal feature set will be robust to these interventions, and correctly predict current and future reward in the linear function approximation setting (Zhang et al., 2020). As an

example, in the context of autonomous driving, an intervention can be a change in weather, or a change from day to night which affects the observation space but not the dynamics or reward. Finally, we show that a representation based on the bisimulation metric generalizes to other reward functions with the same causal ancestors, with an example causal graph in Figure 3.

Theorem 4 (Task Generalization). Given an encoder  $\phi : O \mapsto S$  that maps observations to a latent bisimulation metric representation where  $||\phi(\mathbf{o}_i) - \phi(\mathbf{o}_j)||_2 \coloneqq \tilde{d}(\mathbf{o}_i, \mathbf{o}_j)$ ,  $S$  encodes information about all the causal ancestors of the reward  $AN(R)$ .

Proof in appendix. This result shows that the learned representation will generalize to unseen reward functions, as long as the new reward function has a subset of the same causal ancestors. As an example, a representation learned for a robot to walk will likely generalize to learning to run, because the reward function depends on forward velocity and all the factors that contribute to forward velocity. However, that representation will not generalize to picking up objects, as those objects will be ignored by the learned representation, since they are not likely to be causal ancestors of a reward function designed for walking. Theorem 4 shows that the learned representation will be robust to spurious correlations, or changes in factors that are not in  $AN(R)$ . This complements Theorem 5, that the representation is a minimal sufficient statistic of the optimal value function, improving generalization over non-minimal representations. We show empirical validation of these findings in Section 6.2.

# 6 Experiments

Our central hypothesis is that our non-reconstructive bisimulation based representation learning approach should be substantially more robust to task-irrelevant distractors. To that end, we evaluate our method in a clean setting without distractors, as well as a much more difficult setting with distractors. We compare against several baselines. The first is Stochastic Latent Actor-Critic (SLAC, Lee et al. (2019)), a state-of-the-art method for pixel observations on DeepMind Control that learns a dynamics model with a reconstruction loss. The second is DeepMDP (Gelada et al., 2019), a recent method that also learns a latent representation space using a latent dynamics model, reward model, and distributional Q learning, but for which they needed a reconstruction loss to scale up to Atari. Finally, we compare against two methods using the same architecture as ours but exchange our bisimulation loss with (1) a reconstruction loss (Reconstruction) and (2) contrastive predictive coding (Oord et al., 2018) (Contrastive) to ground the dynamics model and learn a latent representation.

# 6.1 Control with Background Distraction

In this section, we benchmark DBC and the previously described baselines on the DeepMind Control (DMC) suite (Tassa et al., 2018) in two settings and nine environments (Figure 4), finger_spin, cheetah_run, and walker_walk and additional environments in the appendix.

Default Setting. Here, the pixel observations have simple backgrounds as shown in Figure 4 (top row) with training curves for our DBC and baselines. We see SLAC, a recent state-of-the-art model-based representation learning method that uses reconstruction, generally performs best.

Natural Video Setting. Next, we incorporate natural video from the Kinetics dataset (Kay et al., 2017) as background (Zhang et al., 2018), shown in Figure 4 (bottom row). The results confirm our hypothesis: although a number of prior methods can learn effectively in the absence of complex distractors, when distractors are introduced, our non-reconstructive bisimulation based method attains substantially better results.

To visualize the representation learned with our bisimulation metric loss function in Equation (4), we use a t-SNE plot (Figure 5). We see that even when the background looks drastically different, our encoder learns to ignore irrelevant information and maps observations with similar robot configurations near each other. On the far-left of Figure 5, we took 10 nearby points in the t-SNE plot and average

![](images/68bd1b38ed8e50561cda324688e65a36b228c193b6a3af47e1411d140abad208.jpg)  
Figure 4: Left observations: Pixel observations in DMC in the default setting (top row) of the finger spin (left column), cheetah (middle column), and walker (right column), and natural video distractors (bottom row). Right training curves: Results comparing out DBC method to baselines on 10 seeds with 1 standard error shaded in the default setting. The grid-location of each graph corresponds to the grid-location of each observation.

![](images/067423ebc4b1174f440b6dcfce13c68a9fe6fe0233e78cbc5b1a3ef8413cc16e.jpg)  
Figure 5: t-SNE of latent spaces learned with a bisimulation metric (left t-SNE) and VAE (right t-SNE) after training has completed, color-coded with predicted state values (higher value yellow, lower value purple). Neighboring points in the embedding space learned with a bisimulation metric have similar states and correspond to observations with the same task-related information (depicted as pairs of images with their corresponding embeddings), whereas no such structure is seen in the embedding space learned by VAE, where the same image pairs are mapped far away from each other. On the left are 3 examples of 10 neighboring points, averaged.

the observations. We see that the agent is quite crisp, which means neighboring points encode the agent in similar positions, but the backgrounds are very different, and so are blurry when averaged.

# 6.2 Generalization Experiments

We test generalization of our learned representation in two ways. First, we show that the learned representation space can generalize to different types of distractors, by training with simple distractors and testing on the natural video setting. Second, we show that our learned representation can be useful reward functions other than those it was trained for.

Generalizing over backgrounds. We first train on the simple distractors setting and evaluate on natural video. Figure 6 shows an example of the simple distractors setting and performance during training time of two experiments, blue being the zero-shot transfer to the natural video setting, and orange the baseline which trains on natural video. This result empirically validates that the representations learned by DBC are able to effectively learn to ignore the background, regardless of what the background contains or how dynamic it is.

Generalizing over reward functions. We evaluate (Figure 6) the generalization capabilities of the learned representation by training SAC with new reward functions walker_stand and walker_run using the fixed representation learned from walker.walk. This is empirical evidence that confirms Theorem 4: if the new reward functions are causally dependent on a subset of the same factors that determine the original reward function, then our representation is sufficient.

![](images/517bb2539a197dbdbde02f4c82ae91437aee8b965fefb0a0070ab13dc1299cbc.jpg)

![](images/50daafb8630ccff17d5a0cc6b597a0265ce8a9302d0535ba4a348111dc71eb0e.jpg)

![](images/3a84adf4a597ad0c79cf6905b15b85f39f6909dbe207bc9417763b4648a9d2b9.jpg)

![](images/c6044c5594b34fbdcb180df90304178b488a85715984e8f8e41b3b404005903a.jpg)

# 6.3 Comparison with other Bisimulation Encoders

Even though the purpose of bisimulation metrics by Castro (2020) is learning distances  $d$ , not representation spaces  $\mathcal{Z}$ , it nevertheless implements  $d$  with function approximation:  $d(\mathbf{o}_i, \mathbf{o}_j) = \psi(\phi(\mathbf{o}_i), \phi(\mathbf{o}_j))$  by encoding observations with  $\phi$  before computing distances with  $\psi$ , trained as:

![](images/627f7a98448db4819fac83f7eebb96707c420e046b54d31cc9a9e3e7d9752a09.jpg)  
Figure 6: Generalization of a model trained on simple distractors environment and evaluated on kinetics (left). Generalization of an encoder trained on walker.walk environment and evaluated on walker_stand (center) and walker_run (right), all in the simple distractors setting. 10 seeds, 1 standard error shaded.  
Figure 7: Bisim. results

$$
J (\phi , \psi) = \left(\psi \big (\phi (\mathbf {o} _ {i}), \phi (\mathbf {o} _ {j}) \big) - | \mathcal {R} (\mathbf {o} _ {i}) - \mathcal {R} (\mathbf {o} _ {j}) | - \gamma \hat {\psi} \Big (\hat {\phi} \big (\mathcal {P} (\mathbf {o} _ {i}, \pi (\mathbf {o} _ {i})) \big), \hat {\phi} \big (\mathcal {P} (\mathbf {o} _ {j}, \pi (\mathbf {o} _ {j})) \big) \Big)\right), \tag {7}
$$

where  $\hat{\phi}$  and  $\hat{\psi}$  are target networks. A natural question is: how does the encoder  $\phi$  above perform in control tasks? We combine  $\phi$  above with our policy in Algorithm 2 and use the same network  $\psi$  (single hidden layer 729 wide). Figure 7 shows representations from Castro (2020) can learn control, but our method learns faster. Further, our method is simpler: by comparing Equation (7) to Equation (4), our method uses the  $\ell_1$  distance between the encoding instead of introducing an addition network  $\psi$ .

# 6.4 Autonomous Driving with Visual Redundancy

Real-world control systems such as robotics and autonomous vehicles must contend with a huge variety of task-irrelevant information, such as irrelevant objects (e.g. clouds) and irrelevant details (e.g. obstacle color). To evaluate DBC on tasks with more realistic observations, we construct a highway driving scenario with photo-realistic visual observations using the CARLA simulator (Dosovitskiy et al., 2017) shown in Figure 8. The agent's goal is to drive as far as possible down CARLA's Town04's figure-8 the highway in 1000 time-steps without colliding into the 20 other moving vehicles or barriers. Our objective function rewards highway progression an penalises collisions:

![](images/6b9dedadc3932a182b08b55c0008285d01f0fe9398859ced84086d54af120ef5.jpg)  
Figure 8: Highway loop, third-person view of ego car (red), traffic during episode.

![](images/a9bbfefd301a479abb444c32a2361fe836ee6c1676ca74e129a101546cc29ddc.jpg)

![](images/65db0fad0214869d1c0a46128a16168f5e5cf5713ebb0344246602b78fb6e49b.jpg)

Table 1: Driving metrics, averaged over 100 episodes, after 100k training steps. Standard error shown. Arrow direction indicates if we desire the metric larger or smaller.

<table><tr><td></td><td colspan="2">SAC</td><td>DeepMDP</td><td>DBC (ours)</td></tr><tr><td>successes (100m) ↑</td><td colspan="2">12%</td><td>17%</td><td>24%</td></tr><tr><td>distance (m) ↑</td><td colspan="2">123.2 ± 7.43</td><td>106.7 ± 11.1</td><td>179.0 ± 11.4</td></tr><tr><td>crash intensity ↓</td><td colspan="2">4604 ± 30.7</td><td>1958 ± 15.6</td><td>2673 ± 38.5</td></tr><tr><td>average steer ↓</td><td colspan="2">16.6% ± 0.019%</td><td>10.4% ± 0.015%</td><td>7.3% ± 0.012%</td></tr><tr><td>average brake ↓</td><td colspan="2">1.3% ± 0.006%</td><td>4.3% ± 0.033%</td><td>1.6% ± 0.022%</td></tr></table>

$r_t = \mathbf{v}_{\mathrm{ego}}^\top \hat{\mathbf{u}}_{\mathrm{highway}} \cdot \Delta t - \lambda_i \cdot \mathrm{impulse} - \lambda_s \cdot |\mathrm{steer}|$ , where  $\mathbf{v}_{\mathrm{ego}}$  is the velocity vector of the ego vehicle, projected onto the highway's unit vector  $\hat{\mathbf{u}}_{\mathrm{highway}}$ , and multiplied by time discretization  $\Delta t = 0.05$  to measure highway progression in meters. Collisions result in impulses  $\in \mathbb{R}^+$ , measured in Newton-seconds. We found a steering penalty steer  $\in [-1,1]$  helped, and used weights  $\lambda_i = 10^{-4}$  and  $\lambda_s = 1$ . While more specialized objectives exist like lane-keeping, this experiment's purpose is to compare representations with observations more characteristic of real robotic tasks. We use five cameras on the vehicle's roof, each with 60 degree views. By concatenating the images together, our vehicle has a 300 degree view, observed as  $84 \times 420$  pixels. Code and install instructions in appendix.

Results in Figure 10 compare the same baselines as before, except for SLAC which is easily distracted (Figure 4). Instead we used SAC, which does not explicitly learn a representation, but performs surprisingly well from raw images. DeepMDP performs well too, perhaps given its similarly to bisimulation. But, Reconstruction and Contrastive methods again perform poorly with complex images. More intuitive metrics are in Table 1 and Figure 9 depicts the representation space as a t-SNE with corresponding observations. Each run took 12 hours on a GTX 1080 GPU.

![](images/14f3facad91e26d2863aedcc086c088b934096da735ed54a4bacbaf10922a8e0.jpg)  
Figure 9: A t-SNE diagram of encoded first-person driving observations after 10k training steps of Algorithm 1, color coded by value (V in Algorithm 2). Top: the learned representation identifies an obstacle on the right side. Whether that obstacle is a dark wall, bright car, or truck is task-irrelevant: these states are behaviourally equivalent. Left: the ego vehicle has flipped onto its left side. The different wall colors, due to a setting sun, is irrelevant: all states are equally stuck and low-value (purple t-SNE color). Right: clear highway driving. Clouds and sun position are irrelevant.

![](images/8c2e89872011ab1c9d5c96f1850566e61dd51bfacca634c235377bca0f12a433.jpg)  
Figure 10: Performance comparison with 3 seeds on the driving tasks. Our DBC method (red) performs better than DeepMDP (purple) or learning straight from pixels without a representation (SAC, green), and much better than using contrastive losses (blue). The final performance of our method is  $46.8\%$  better than the next best baseline (SAC).

# 7 Discussion

This paper presents Deep Bisimulation for Control: a new representation learning method that considers downstream control. Observations are encoded into representations that are invariant to different task-irrelevant details in the observation. We show this is important when learning control from outdoor images, or otherwise images with background "distractions". In contrast to other bisimulation methods, we show performance gains when distances in representation space match the bisimulation distance between observations. Future work: Our latent dynamics model  $\hat{\mathcal{P}}$  was only used for training our encoder in Equation (4), but could also be used for multi-step planning in latent space. An ensemble of models  $\{\hat{\mathcal{P}}_k\}_{k=1}^K$  could also help handle uncertainty better and give robustness to distributional shift between training and test observations (McAllister et al., 2019).

# References

Pablo Samuel Castro. Scalable methods for computing state similarity in deterministic Markov decision processes. In Association for the Advancement of Artificial Intelligence (AAAI), 2020.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations. arXiv:2002.05709 [cs, stat], February 2020. URL http://arxiv.org/abs/2002.05709.arXiv:2002.05709.  
Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Neural Information Processing Systems (NeurIPS), pp. 4754-4765, 2018.  
Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. arXiv preprint arXiv:1711.03938, 2017.  
Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite Markov decision processes. In Uncertainty in Artificial Intelligence (UAI), pp. 162-169, 2004. ISBN 0-9749039-0-6. URL http://dl.acm.org/citation.cfm?id=1036843.1036863.  
Norm Ferns, Prakash Panangaden, and Doina Precup. Bisimulation metrics for continuous Markov decision processes. Society for Industrial and Applied Mathematics, 40(6):1662-1714, December 2011. ISSN 0097-5397. doi: 10.1137/10080484X. URL https://doi.org/10.1137/10080484X.  
Norman Ferns and Doina Precup. Bisimulation metrics are optimal value functions. In Uncertainty in Artificial Intelligence (UAI), pp. 210-219, 2014.  
Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G. Bellemare. DeepMDP: Learning continuous latent space models for representation learning. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), International Conference on Machine Learning (ICML), volume 97, pp. 2170-2179, Jun 2019.  
Robert Givan, Thomas L. Dean, and Matthew Greig. Equivalence notions and model minimization in Markov decision processes. Artificial Intelligence, 147:163-223, 2003.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.  
Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. arXiv preprint arXiv:1811.04551, 2018.  
Olivier J Henaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, SM Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272, 2019.  
Anders Jonsson and Andrew Barto. Causal graph based decomposition of factored MDPs. J. Mach. Learn. Res., 7:2259-2301, December 2006. ISSN 1532-4435.  
Will Kay, João Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. Computing Research Repository (CoRR), 2017. URL http://arxiv.org/abs/1705.06950.  
Sascha Lange and Martin Riedmiller. Deep auto-encoder neural networks in reinforcement learning. In International Joint Conference on Neural Networks (IJCNN), pp. 1-8. IEEE, 2010.  
Sascha Lange, Martin Riedmiller, and Arne Voigtlander. Autonomous reinforcement learning on raw visual input data in a real world application. In International Joint Conference on Neural Networks (IJCNN), pp. 1-8, 2012. doi: 10.1109/IJCNN.2012.6252823.

K. G. Larsen and A. Skou. Bisimulation through probabilistic testing (preliminary report). In Symposium on Principles of Programming Languages, pp. 344-352. Association for Computing Machinery, 1989. ISBN 0897912942. doi: 10.1145/75277.75307. URL https://doi.org/10.1145/75277.75307.  
Michael Laskin, Aravind Srinivas, and Pieter Abbeel. CURL: Contrastive unsupervised representations for reinforcement learning. arXiv:2003.06417, 2020.  
Alex X Lee, Anusha Nagabandi, Pieter Abbeel, and Sergey Levine. Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model. arXiv preprint arXiv:1907.00953, 2019.  
Lihong Li, Thomas J Walsh, and Michael L Littman. Towards a unified theory of state abstraction for MDPs. In ISAIM, 2006.  
Rowan McAllister, Gregory Kahn, Jeff Clune, and Sergey Levine. Robustness to out-of-distribution inputs via task-aware generative uncertainty. In International Conference on Robotics and Automation. IEEE, 2019.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, February 2015. ISSN 00280836. URL http://dx.doi.org/10.1038/nature14236.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Kate Rakelly, Aurick Zhou, Deirdre Quillen, Chelsea Finn, and Sergey Levine. Efficient off-policy meta-reinforcement learning via probabilistic context variables. arXiv preprint arXiv:1903.08254, 2019.  
Bernhard Scholkopf. Causality for machine learning, 2019.  
Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdelmaleki, Josh Merel, Andrew Lefrancq, Timothy Lillicrap, and Martin Riedmiller. DeepMind control suite. Technical report, DeepMind, January 2018. URL https://arxiv.org/abs/1801.00690.  
Jonathan Taylor, Doina Precup, and Prakash Panagaden. Bounding performance loss in approximate MDP homomorphisms. In Neural Information Processing (NeurIPS), pp. 1649-1656, 2009. URL http://papers.nips.cc/paper/3423-bounding-performance-loss-in-approximate-mdp-homomorphisms.pdf.  
Franck van Breugel and James Worrell. Towards quantitative verification of probabilistic transition systems. In Fernando Orejas, Paul G. Spirakis, and Jan van Leeuwen (eds.), Automata, Languages and Programming, pp. 421-432. Springer, 2001. ISBN 978-3-540-48224-6. doi: 10.1007/3-540-48224-5_35.  
Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. ArXiv, abs/1807.03748, 2018.  
Cédric Villani. Topics in optimal transportation. American Mathematical Society, 01 2003.  
Niklas Wahlström, Thomas Schön, and Marc Deisenroth. From pixels to torques: Policy learning with deep dynamical models. arXiv preprint arXiv:1502.02251, 2015.  
Manuel Watter, Jost Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. In Neural Information Processing Systems (NeurIPS), pp. 2728-2736, 2015.  
Denis Yarats and Ilya Kostrikov. Soft actor-critic (SAC) implementation in PyTorch. https://github.com/denisjarats/pytorch_sac, 2020.

Denis Yarats, Amy Zhang, Ilya Kostrikov, Brandon Amos, Joelle Pineau, and Rob Fergus. Improving sample efficiency in model-free reinforcement learning from images. arXiv preprint arXiv:1910.01741, 2019.  
Amy Zhang, Yuxin Wu, and Joelle Pineau. Natural environment benchmarks for reinforcement learning. Computing Research Repository (CoRR), abs/1811.06032, 2018. URL http://arxiv.org/abs/1811.06032.  
Amy Zhang, Clare Lyle, Shagun Sodhani, Angelos Filos, Marta Kwiatkowska, Joelle Pineau, Yarin Gal, and Doina Precup. Invariant causal prediction for block mdps. In International Conference on Machine Learning (ICML), 2020.
