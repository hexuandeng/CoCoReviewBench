# COMMUNICATING HIERARCHICAL NEURAL CONTROLLER FOR LEARNING ZERO-SHOT TASK GENERALIZATION

Junhyuk Oh, Satinder Singh, Honglak Lee

University of Michigan

Ann Arbor, MI, USA

{junhyuk,baveja,honglak}@umich.edu

Pushmeet Kohli

Microsoft Research

Redmond, WA, USA

pkohli@microsoft.com

# ABSTRACT

The ability to generalize from past experience to solve previously unseen tasks is a key research challenge in reinforcement learning (RL). In this paper, we consider RL tasks defined as a sequence of high-level instructions and study two types of generalization: to unseen and longer sequences of previously seen instructions, and to sequences where the instructions themselves were previously not seen. We present a novel hierarchical deep RL architecture that consists of two interacting neural controllers: a meta controller that reads instructions and repeatedly communicates subtasks to a subtask controller that in turn learns to perform such subtasks. To generalize better to unseen instructions, we propose a regularizer that encourages to learn subtask embeddings that capture correspondences between similar subtasks. We also propose a new differentiable neural network architecture in the meta controller that learns temporal abstractions which makes learning more stable under delayed reward. Our architecture is evaluated on a non-deterministic 2D grid world where the agent should execute a list of instructions described by natural language. We demonstrate that the proposed architecture is able to generalize well over unseen instructions as well as longer lists of instructions.

# 1 INTRODUCTION

Humans can often generalize to novel tasks without additional learning by leveraging past learning experience. We would like our artificial agents to have similar "zero-shot" generalization capabilities. For example, after learning to solve tasks with instructions such as 'Go to X (or Y)' and 'Pick up Y (or Z)', our agent should be able to infer the underlying goal of new tasks with instructions like 'Go to Z', which requires disentangling the verbs ('Go to/Pick up') and the nouns/objects ('X, Y, Z'). Furthermore, we would like our agents to learn to compose policies to solve novel tasks composed of sequences of seen and unseen instructions. Developing the ability to achieve such generalizations is a key challenge in artificial intelligence and the subfield of reinforcement learning (RL).

In this paper, we study the problem of zero-shot task generalization in RL by introducing the "instruction execution" problem where the agent is required to learn through interaction with its environment how to achieve an overall task specified by a list of high-level instructions (see Figure 1). As motivation for this problem considers a human owner training its new household robot to execute complex tasks specified by natural language text that decompose the task into a sequence of instructions. Given that it is infeasible to explicitly train the robot on all possible instruction-sequences, this problem involves two types of generalizations: to unseen and longer sequences of previously seen instructions, and sequences where the some of the instructions themselves were previously not seen. Of course, the usual RL problem of learning policies through interaction to accomplish the goals of an instruction remains part of the problem as well. We assume that the agent does not receive any signal on completing or failing

![](images/036d7ff7592a18f2ee0fc817d139f3ae9c955a88bd7eb9cf65828decc481c8bc.jpg)  
Training  
Testing  
Figure 1: Example of grid-world and instructions. The agent is tasked to execute longer sequences of instructions after trained on short sequences of instructions; in addition previously unseen instructions can be given during evaluation (blue text). The agent can get more rewards if it deals with randomly appearing enemies (red outlined box) regardless of current instructions.

Visit tree

Pick up all diamond

Transform cow

Pick up candy

Transform diamond

Pick up tree

Pick up cow

Transform all meat

Visit box.

Visit duck

Visit back

ing to complete individual instructions from the environment/owner and so the informative reward signal is delayed until the end. Furthermore, there can be random events in the environment that

require the agent to interrupt whatever it is doing and deviate from the instructions to maintain some background task as described in Figure 1. Altogether this makes for a challenging zero-shot task generalization RL problem.

Brief Background: RL tasks composed of sequences of subtasks have been studied before and a number of hierarchical RL approaches designed for them. Typically these have the form of a meta controller and a set of lower-level controllers for subtasks (Sutton et al., 1999; Dietterich, 2000; Parr and Russell, 1997). The meta controller is limited to selecting one from a set of lower-level controllers to employ at any time. This makes it impossible for the low-level controller to generalize to new subtasks without training a new low-level controller separately. Much of the previous work also assumes that the overall task is fixed (e.g., Taxi domain (Dietterich, 2000; Ghavamzadeh and Mahadevan, 2003)). Transfer learning across multiple compositional tasks has typically been studied in RL formulations in which new tasks are only presented via a new reward function from the environment (Singh, 1991; 1992) and so there is no opportunity for fast model-free generalization. To the best of our knowledge, zero-shot model-free generalization to new or longer tasks as well as unseen tasks has not been well-studied in the RL setting.

Our Architecture: This paper presents a hierarchical deep RL architecture (see Figure 2a) that consists of two interacting neural controllers: a meta controller that repeatedly chooses an instruction and conditioned on the current state of the environment translates it into subtask-arguments (details on this in later sections) and communicates those to the subtask controller that in turn chooses primitive actions given the subtask. This makes the subtask controller a parameterized option (Sutton et al., 1999) module in which the parameters are the subtask-arguments mentioned above. On top of the subtask controller, the meta controller is trained to select proper subtask-arguments depending on observations from the environment, feedback from the subtask controller about termination, and the task instructions. In order to generalize over unseen instructions, we propose analogy-making regularization (discussed in Section 4.2) which encourages to learn subtask embeddings that capture correspondences between similar subtasks. In addition, we propose a new differentiable neural architecture in the meta controller that implicitly learns temporal abstractions so that it can operate at a larger time-scale and update the subtask-arguments to the subtask controller only when needed.

Our Results: We developed a 2D grid world environment where the agent can interact with many objects as illustrated in Figure 1 based on MazeBase (Sukhbaatar et al., 2015) (see Section 5.1 for details). The empirical results show that the meta-controller's ability to learn temporal abstractions and a form of analogy-making regularization were all key in allowing our hierarchical architecture to generalize in a zero-shot fashion to unseen tasks.

# 2 RELATED WORK

Hierarchical Deep Reinforcement Learning. In addition to hierarchical RL described in Section 1, there is recent work on hierarchical deep RL. Bacon and Precup (2015) proposed option-critic architecture which learns options without any domain knowledge. Kulkarni et al. (2016) proposed hierarchical Deep Q-Learning and demonstrated improved exploration in a challenging Atari game. Tessler et al. (2016) proposed a similar architecture that allows the high-level controller to choose primitive actions directly. Vezhnevets et al. (2016) proposed a deep architecture that automatically learns macro-actions. Unlike these recent works, the goal of our work is to generalize over sequences of tasks in order to scale up RL agents to large-scale tasks.

Zero-shot Task Generalization. Schaul et al. (2015) proposed universal value function approximators (UVFA) that learn a value function given a state and goal pair and showed that their framework can generalize over unseen goals. Isele et al. (2016) proposed a method for zero-shot task generalization which uses task descriptors to predict the parameter of the policy. Our subtask controller implements the idea of universal option introduced as a potential application of UVFA. Unlike the previous works, we extend the scope of task generalization to sequences of tasks and discuss how to construct neural networks to improve generalization ability.

Instruction Execution. There has been a line of work for building agents that can execute natural language instructions: Tellex et al. (2011; 2014) for robotics and MacMahon et al. (2006); Chen and Mooney (2011); Mei et al. (2015) for a simulated environment. However, these approaches focus on natural language understanding to map instructions to a sequence of actions or groundings in a supervised setting. In contrast, we focus on generalization to different sequences of instructions without any supervision for language understanding or for actions.

![](images/7a30e1252fe2c3fb8af2f98a05e817df50a8997aa658d24dde63ab2e728a63e6.jpg)  
(a) Overview

![](images/02c655751356118904f3777ec2e892242912d78add6a2b720151c97d9f292e52.jpg)  
(b) Subtask controller

![](images/a7b3ad57c08070b85052a34532dde074a50c2d9f051a6fd21ce94091e24d1a37.jpg)  
(c) Meta controller  
Figure 2: Proposed architecture. See text for details.

# 3 COMMUNICATING HIERARCHICAL NEURAL CONTROLLER

As illustrated in Figure 2a, the meta controller communicates with the subtask controller by passing subtask-arrugments. Notationally, a space of subtasks  $\mathcal{G}$  can be defined using the Cartesian product of their arguments  $\mathcal{G}^{(1)}\times \dots \times \mathcal{G}^{(n)}$ , where  $\mathcal{G}^{(i)}$  is a set of the  $i$ -th arguments (e.g.,  $\mathcal{G} = \{\mathrm{Visit},\mathrm{Pick~up}\} \times \{\mathrm{A},\mathrm{B}\}$ ). This forms a communication interface between the two controllers.

It is important to note that natural language instructions that specify tasks are not directly subtasks; indeed there is not a one-to-one correspondence between instructions and subtask-arguments. This is due to a number of important reasons. First, instructions such as 'Pick up all X' are executed by repeatedly solving a subtask [Pick up, X]. Second, the meta controller sometimes needs to interrupt ongoing subtasks and replace them with other subtasks that are not relevant to the instruction under consideration because of the background task based on the stochastic events as described in Figure 1.

The subtask controller communicates with the meta controller by giving a terminal signal for the given subtask. This communication protocol allows each controller to not only focus on their own independent roles but also cooperate with each other to learn a complex closed-loop policy.

# 3.1 SUBTASK CONTROLLER

As shown in Figure 2b, the subtask controller maps the agent's observation (or state) and subtask-arguments into a primitive action via a policy  $\pi$  and a termination function  $\beta$  which predicts whether the subtask is over or not in the current observation.

We use the parameter prediction approach (Lei Ba et al., 2015; Bertinetto et al., 2016) to condition the policy on the subtask arguments for better generalization. To develop some notation, given an input  $(\mathbf{x})$  and subtask arguments  $(\mathbf{g} = [g^{(1)},\dots,g^{(n)}])$ , the output  $(\mathbf{y})$  of a convolutional and a fully-connected layer is written as:

$$
\text {C o n v l o u t i o n :} \mathbf {y} = \varphi (\mathbf {g}) * \mathbf {x} + \mathbf {b}
$$

$$
\text {F u l l - c o n n e c t e d :} \mathbf {y} = \mathbf {W} ^ {\prime} \operatorname {d i a g} (\varphi (\mathbf {g})) \mathbf {W} \mathbf {x} + \mathbf {b}
$$

where  $\varphi$  is the embedding of the subtask learned by a multi-layer perceptron (MLP). We use matrix factorization (similar to Memisevic and Hinton (2010)) to reduce the number of parameters for the fully-connected layer. Given the observation  $\mathbf{s}_t$  and subtask  $\mathbf{g}$ , the subtask controller is defined as:

$$
\text {P o l i c y :} \pi_ {\theta_ {s}} \left(\mathbf {a} _ {t} \mid \mathbf {s} _ {t}, \mathbf {g}\right) \propto \exp \left(\varphi^ {\pi} \left(\operatorname {C N N} \left(\mathbf {s} _ {t}; \mathbf {g}\right)\right)\right)
$$

$$
\text {T e r m i n a t i o n :} \beta_ {\theta_ {s}} \left(b _ {t} | \mathbf {s} _ {t}, \mathbf {g}\right) \propto \sigma \left(\varphi^ {\beta} (\operatorname {C N N} (\mathbf {s} _ {t}; \mathbf {g}))\right)
$$

where  $\varphi^{\pi}$  and  $\varphi^{\beta}$  are MLPs,  $\sigma (\cdot)$  is the sigmoid function, and CNN  $(\mathbf{s}_t;\mathbf{g})$  denotes a convolutional neural network (CNN) with parameters predicted by the subtask.

# 3.2 META CONTROLLER

As illustrated in Figure 2c, the meta controller computes subtask arguments from the observation, the instructions  $(M)$ , and the subtask termination  $(b\sim \beta_{\theta_s}^g)$  output by the subtask controller.

Context. The meta controller uses a context vector to compute and communicate subtask-arguments to the subtask controller. Given the sentence embedding  $\mathbf{r}_{t - 1}$  retrieved at the previous time-step from the instructions (described in Section 3.2.1), the previously selected subtask  $\mathbf{g}_{t - 1} = [g_{t - 1}^{(1)},\dots,g_{t - 1}^{(n)}]$ , and the subtask termination  $b_{t}\sim \beta_{\theta_{s}}$  ( $b_{t}|\mathbf{s}_{t},\mathbf{g}_{t - 1}$ ), the meta-controller computes the context vector  $(\mathbf{h}_t)$  through a neural network with parameters predicted by  $\mathbf{r}_{t - 1},\mathbf{g}_{t - 1},b_t$  as:

$$
\mathbf {h} _ {t} = \operatorname {C N N} \left(\mathbf {s} _ {t}; \mathbf {r} _ {t - 1}, \mathbf {g} _ {t - 1}, b _ {t}\right).
$$

Intuitively,  $\mathbf{g}_{t-1}$  and  $b_{t}$  provide information about which subtask was being solved by the subtask controller and whether it has been finished or not. Note that the subtask does not necessarily match with the retrieved instruction  $(\mathbf{r}_{t-1})$ , e.g., when the agent is dealing with the background task. By combining all the information,  $\mathbf{h}_{t}$  encodes the spatio-temporal context which is used to determine the next instruction and the subtask to communicate.

# 3.2.1 SUBTASKUPDATER

The meta controller has a subtask updater that constructs a memory-structure from the list of instructions, retrieves an instruction by maintaining a pointer into the memory structure, and computes the subtask arguments.

Instruction Memory. Given instructions as a list of sentences  $M = (m_{1}, m_{2}, \dots, m_{K})$ , where each sentence consists of a list of words,  $m_{i} = (w_{1}, \dots, w_{|m_{i}|})$ , the subtask updater constructs memory blocks  $\mathbf{M} \in \mathbb{R}^{E \times K}$ , where each column is  $E$ -dimensional embedding of a sentence. The subtask module maintains a memory pointer defined over memory locations,  $\mathbf{p}_{t} \in \mathbb{R}^{K}$ , which is used for instruction retrieval. Memory construction and retrieval is formally described as:

$$
\text {M e m o r y :} \mathbf {M} = \left[ \varphi^ {w} \left(m _ {1}\right), \varphi^ {w} \left(m _ {2}\right), \dots , \varphi^ {w} \left(m _ {K}\right) \right] \quad \text {R e t r i e v a l :} \mathbf {r} _ {t} = \mathbf {M} \mathbf {p} _ {t}.
$$

Here  $\varphi^w (m_i)\in \mathbb{R}^E$  is the embedding of the  $i$ -th sentence. In this work, we used the bag-of-words (BoW) representation which computes the sum of all word embeddings in a sentence:  $\varphi^w (m_i) = \sum_{j = 1}^{|m_i|}\mathbf{W}^m w_j$  where  $\mathbf{W}^m$  is the word embedding matrix,  $\mathbf{p}_t$  is a non-negative vector which sums up to 1, and  $\mathbf{r}_t\in \mathbb{R}^E$  is the retrieved sentence embedding which is used for computing the subtask-arguments.

Location-based Memory Addressing. Since instructions should be executed sequentially, we use a location-based memory addressing mechanism (Zaremba and Sutskever, 2015; Graves et al., 2014) for the memory pointer. Specifically, the subtask updater shifts the memory pointer by  $[-1,1]$  as:

$$
\mathbf {p} _ {t} = \mathbf {l} _ {t} * \mathbf {p} _ {t - 1} \text {w h e r e} \left\{ \begin{array}{l l} \mathbf {l} _ {t} = \operatorname {S o f t m a x} \left(\varphi^ {\text {s h i f t}} \left(\mathbf {h} _ {t}\right)\right) & (\text {S o f t - s h i f t}) \\ \mathbf {l} _ {t} \sim \operatorname {S o f t m a x} \left(\varphi^ {\text {s h i f t}} \left(\mathbf {h} _ {t}\right)\right) & (\text {H a r d - s h i f t}) \end{array} \right. \tag {1}
$$

where  $\mathbf{l}_t\in \mathbb{R}^3$  is a memory-shift vector.  $\varphi^{shift}$  is an MLP. Although the hard-shift keeps the memory pointer sharp, it makes optimization difficult due to the non-differentiable operation. To ease optimization, we initialize the parameters of the hard-shift architecture to the parameters of the soft-shift architecture which is further described in Algorithm 2.

Subtask Arguments. The subtask updater takes the context  $(\mathbf{h}_t)$ , updates the memory pointer  $(\mathbf{p}_t)$ , retrieves a sentence embedding  $(\mathbf{r}_t)$ , and finally computes subtask-arguments as follows:

$$
\pi_ {\theta_ {m}} \left(\mathbf {g} _ {t} | \mathbf {h} _ {t}, \mathbf {r} _ {t}\right) = \prod_ {i} \pi_ {\theta_ {m}} \left(g _ {t} ^ {(i)} | \mathbf {h} _ {t}, \mathbf {r} _ {t}\right) \mathrm {w h e r e} \pi_ {\theta_ {m}} \left(g _ {t} ^ {(i)} | \mathbf {h} _ {t}, \mathbf {r} _ {t}\right) \propto \exp \left(\varphi_ {i} ^ {\mathrm {g o a l}} \left(\mathbf {h} _ {t}, \mathbf {r} _ {t}\right)\right)
$$

where  $\varphi_{i}^{goal}$  is an MLP for the  $i$ -th subtask argument.

# 3.2.2 DIFFERENTIABLE TEMPORAL ABSTRACTIONS

Although the subtask updater can update the memory pointer and compute correct subtask-arguments in principle, making a decision at every time-step can be inefficient because subtasks do not change very frequently. Instead, having temporally-extended actions can be useful for dealing with delayed reward by operating at a larger time-scale (Sutton et al., 1999).

To this end, we introduce an internal binary action  $c_t$  which decides whether to update the subtask updater or not. This action is defined as:  $c_t \sim \sigma(\varphi^{update}(\mathbf{h}_t))$ . If  $c_t = 1$ , the subtask updater updates the memory pointer, retrieves an instruction, and updates the subtask arguments. Otherwise, the meta controller continues commu-

# Algorithm 1 Subtask update (Hard)

<table><tr><td colspan="2">Input: ht, p t-1, rt-1, gt-1</td></tr><tr><td colspan="2">Output: pt, rt, gt</td></tr><tr><td colspan="2">ct ∼ σ (φupdate (ht))</td></tr><tr><td>if ct = 1 then</td><td>▷ Update</td></tr><tr><td colspan="2">lt ∼ Softmax (φshift (ht))</td></tr><tr><td>pt← lt * pt-1</td><td>▷ Shift</td></tr><tr><td>rt← M^T pt</td><td>▷ Retrieve</td></tr><tr><td>gt ∼ πθm (gt|ht, rt)</td><td>▷ Subtask</td></tr><tr><td>else</td><td></td></tr><tr><td colspan="2">pt← pt-1, rt← rt-1, gt← gt-1</td></tr><tr><td>end if</td><td></td></tr></table>

nicating the current subtask arguments without involving the subtask updater. The entire scheme is described in Algorithm 1.

During training of the update decision, we use L1 regularization on the probability of update to penalize frequent updates as in Vezhnevets et al. (2016). Note that the meta controller can still change the subtask-arguments without waiting for subtask termination depending on the context (e.g., when enemies appear randomly), which is analogous to interrupting options (Sutton et al., 1999).

To handle the difficulty of training with a non-differentiable update decision, we propose soft-update method shown in Algorithm 2. The key idea is to take the weighted sum of both 'update' and 'no update' scenarios. The soft-architecture (Algorithm 2) reduces to the hard-architecture (Algorithm 1) if we sample  $c_t$  and  $\mathbf{l}_t$  instead of taking the weighted sum, which justifies our initialization trick.

# Algorithm 2 Subtask update (Soft)

Input:  $\mathbf{h}_t, \mathbf{p}_{t-1}, \mathbf{r}_{t-1}, \mathbf{g}_{t-1}$

Output:  $\mathbf{p}_t, \mathbf{r}_t, \mathbf{g}_t$

$$
c _ {t} \leftarrow \sigma \left(\varphi^ {\text {u p d a t e}} (\mathbf {h} _ {t})\right)
$$

$\mathbf{l}_t \gets \text{Softmax} \left( \varphi^{shift}(\mathbf{h}_t) \right)$

$$
\tilde {\mathbf {p}} _ {t} \leftarrow \mathbf {l} _ {t} * \mathbf {p} _ {t - 1}
$$

$$
\tilde {\mathbf {r}} _ {t} \leftarrow \mathbf {M} ^ {\mathrm {T}} \tilde {\mathbf {p}} _ {t}
$$

$$
\mathbf {p} _ {t} \leftarrow c _ {t} \tilde {\mathbf {p}} _ {t} + (1 - c _ {t}) \mathbf {p} _ {t - 1}
$$

$$
\mathbf {r} _ {t} \leftarrow c _ {t} \tilde {\mathbf {r}} _ {t} + (1 - c _ {t}) \mathbf {r} _ {t - 1}
$$

$$
\begin{array}{l} g _ {t} ^ {(i)} \sim c _ {t} \pi_ {\theta_ {m}} \left(g _ {t} ^ {(i)} | \mathbf {h} _ {t}, \tilde {\mathbf {r}} _ {t}\right) \\ + \left(1 - c _ {t}\right) g _ {t - 1} ^ {(i)} \forall i \\ \end{array}
$$

# 4 LEARNING

The subtask controller is first trained on a subset of subtasks, i.e., we directly provide subtask-arguments for the chosen subset during training. Then, the meta controller is trained on a training set of task-instructions, while the subtask controller is fixed. Training methodologies for the subtask controller and the meta controller are described in Section 4.1, and more details of the objective functions are provided in the appendix D. Section 4.2 describes our analogy-making regularization applied to both controllers for generalization to unseen subtasks.

# 4.1 REINFORCEMENT LEARNING OF SUBTASK CONTROLLER AND META CONTROLLER

Subtask Controller. Inspired by the idea of improving policies through reinforcement learning after supervised learning as discussed by Levine and Koltun (2013); Silver et al. (2016), we first train the subtask controller through policy distillation (Rusu et al., 2015; Parisotto et al., 2015). The idea is to train separate policies for each subtask and use them to provide action labels to train the subtask controller. The subtask controller is further fine-tuned through the actor-critic method (Konda and Tsitsiklis, 1999) with generalized advantage estimation (GAE) (Schulman et al., 2015). During training, the subtask controller is made to predict whether the current state is terminal or not through a binary classification objective.

Meta Controller. Actor-critic method with GAE is used to update the parameters of the meta controller, while the parameters of the subtask controller are frozen.

# 4.2 ANALOGY-MAKING REGULARIZATION

When learning a non-linear subtask embedding from arguments using a neural network in both controllers,  $\varphi(g^{(1)}, g^{(2)}, \ldots, g^{(n)})$ , it is desirable for the network to learn the meaning of individual arguments as well as the relationships between them in order to infer the goal of unseen configurations of arguments. To this end, we propose a regularizer based on analogy-making (Reed et al., 2015) and manifold learning (Hadsell et al., 2006; Reed et al., 2014). The key idea is to learn correspondences between arguments such as 'Visit X': 'Visit Z': 'Pick up X': 'Pick up Z'.

More specifically, we define several constraints as follows:

$$
\left\| \varphi \left(\mathbf {g} _ {A}\right) - \varphi \left(\mathbf {g} _ {B}\right) - \varphi \left(\mathbf {g} _ {C}\right) + \varphi \left(\mathbf {g} _ {D}\right) \right\| \approx 0 \quad \text {i f} \mathbf {g} _ {A}: \mathbf {g} _ {B}::: \mathbf {g} _ {C}: \mathbf {g} _ {D} \tag {2}
$$

$$
\left\| \varphi \left(\mathbf {g} _ {A}\right) - \varphi \left(\mathbf {g} _ {B}\right) - \left(\mathbf {g} _ {C}\right) + \varphi \left(\mathbf {g} _ {D}\right) \right\| \geq \tau_ {\text {d i s}} \quad \text {i f} \mathbf {g} _ {A}: \mathbf {g} _ {B} \neq \mathbf {g} _ {C}: \mathbf {g} _ {D} \tag {3}
$$

$$
\left\| \varphi \left(\mathbf {g} _ {A}\right) - \varphi \left(\mathbf {g} _ {B}\right) \right\| \geq \tau_ {\text {d i f f}} \quad \text {i f} \mathbf {g} _ {A} \neq \mathbf {g} _ {B} \tag {4}
$$

where  $\mathbf{g}_k = \left[g_k^{(1)}, g_k^{(2)}, \ldots, g_k^{(n)}\right] \in \mathcal{G}$  are subtask arguments. Eq. (2) represents the analogy-making relationship, while Eq. (3) and Eq. (4) prevent trivial solutions. To satisfy the above constraints, we propose the following objective functions based on contrastive loss (Hadsell et al., 2006):

$$
\mathcal {L} _ {\text {s i m}} = \mathbb {E} _ {(\mathbf {g} _ {A}, \mathbf {g} _ {B}, \mathbf {g} _ {C}, \mathbf {g} _ {D}) \sim \mathcal {G} _ {\text {s i m}}} \left[ \| \varphi (\mathbf {g} _ {A}) - \varphi (\mathbf {g} _ {B}) - (\mathbf {g} _ {C}) + \varphi (\mathbf {g} _ {D}) \| ^ {2} \right] \tag {5}
$$

$$
\mathcal {L} _ {d i s} = \mathbb {E} _ {\left(\mathbf {g} _ {A}, \mathbf {g} _ {B}, \mathbf {g} _ {C}, \mathbf {g} _ {D}\right) \sim \mathcal {G} _ {d i s}} \left[ \max  \left(0, \tau_ {d i s} - \| \varphi (\mathbf {g} _ {A}) - \varphi (\mathbf {g} _ {B}) - (\mathbf {g} _ {C}) + \varphi (\mathbf {g} _ {D}) \|\right) ^ {2} \right] \tag {6}
$$

$$
\mathcal {L} _ {d i f f} = \mathbb {E} _ {\left(\mathbf {g} _ {A}, \mathbf {g} _ {B}\right) \sim \mathcal {G} _ {d i f f}} \left[ \max  \left(0, \tau_ {d i f f} - \| \varphi (\mathbf {g} _ {A}) - \varphi (\mathbf {g} _ {B}) \|\right) ^ {2} \right] \tag {7}
$$

where  $\mathcal{G}_{sim},\mathcal{G}_{dis},\mathcal{G}_{diff}$  consist of subtask arguments satisfying conditions in Eq. (2), Eq. (3) and Eq. (4) respectively.  $\tau_{dis},\tau_{diff}$  are threshold distances (hyperparameters). The final analogy-making regularizer is the weighted sum of the above three objectives.

Analogies Under Non-independence. Although we construct  $\mathcal{G}_{sim}$  assuming that all configurations of subtasks arguments are valid and independent from each other throughout the main experiment, our analogy-making regularizer can also be used to inject prior knowledge so that the agent generalizes to unseen subtasks in a specific way. For example, if some objects should be handled in a different way given the same subtask, we can still apply analogy-making regularizer so that Eq. 2 is satisfied only between the same type of objects. This is further discussed in the appendix B.

# 5 EXPERIMENTS AND RESULTS

Our experiments were designed to explore the following hypotheses: our proposed hierarchical architecture will generalize better than a non-hierarchical controller, that analogy-making regularization and learning temporal abstractions in the meta controller will both separately be beneficial for task generalization. We are also interested in understanding the qualitative properties of our agent's behavior. The demo videos are available at the following website: https://sites.google.com/a/umich.edu/junhyuk-oh/task-generalization.

# 5.1 EXPERIMENTAL SETTING

Environment. We developed a 2D grid world based on MazeBase (Sukhbaatar et al., 2015) where the agent can interact with many objects as illustrated in Figure 1. Unlike the original MazeBase, an observation is represented as a binary 3D tensor:  $\mathbf{x}_t \in \mathbb{R}^{18 \times 10 \times 10}$  where 18 is the number of object types and  $10 \times 10$  is the size of the grid world. Each channel is a binary mask indicating the presence of each object type. There are agent, blocks, water, and 15 types of objects with which the agent can interact (see Appendix E), and all of them are randomly placed for each episode.

The agent has 13 primitive actions: No-operation, Move (North/South/West/East, referred to as "NSWE"), Pick up (NSWE), and Transform (NSWE). Move actions move the agent by one cell in the specified direction. Pick up actions remove the adjacent object in the corresponding relative position, and depending on the object type Transform actions either remove it or transform it to another object.

The agent receives a time penalty  $(-0.1)$  for each time-step. Water cells act as obstacles which give  $-0.3$  when the agent visits them. The agent receives  $+1$  reward when it finishes all instructions in the correct order. Throughout the episode, an enemy randomly appears, moves, and disappears after 10 steps. Transforming an enemy gives  $+0.9$  reward. More details are described in the appendix E.

Subtasks and Instructions. The subtask space is defined as the Cartesian product of two arguments:  $\mathcal{G} = \{\text{Visit}, \text{Pick up}, \text{Transform}\} \times \{X_1, X_2, \dots, X_{15}\}$  where  $X_i$  is an object type. The agent should be on the same cell of the target object to finish 'Visit' task. For 'Pick up' and 'Transform' tasks, the agent should perform the corresponding primitive action to the target object. If there are multiple target objects in the world, the agent can perform the action to any of the target objects.

The instructions are represented as a sequence of sentences, each of which is one of the following: Visit X, Pick up X, Transform X, Pick up all X, and Transform all X where 'X' is the target object type. While the first three instructions require the agent to perform the corresponding subtask, the last two instructions require the agent to repeat the same subtask until the target objects completely disappear from the world.

Task Split. Among 45 subtasks in  $\mathcal{G}$ , only 30 subtasks are presented to the subtask controller during training. 3 subtasks from the training subtasks and 3 subtasks from the unseen subtasks were selected as the validation set to pick the best-performing subtask controller. For training the meta controller, we created four sets of sequences of instructions: training, validation, and two test sets. The training tasks consist of sequences of up to 4 instructions sampled from the set of training instructions. The validation set consists of sequences of 7 instructions with small overlaps with the training instructions and unseen instructions. The two test sets consist of 20 seen and unseen instructions respectively. More details of the task split are described in the appendix E.

Flat Controller. To understand the advantage of using the communicating hierarchical structure of our controllers, we trained a flat controller which is almost identical to the meta controller architecture except that it directly chooses primitive actions without using the subtask controller. Details

<table><tr><td rowspan="2">Agent</td><td colspan="3">Train</td><td colspan="3">Unseen</td></tr><tr><td>Reward</td><td>Success</td><td>Accuracy</td><td>Reward</td><td>Success</td><td>Accuracy</td></tr><tr><td>Concat</td><td>0.53</td><td>99.8%</td><td>99.8%</td><td>-3.71</td><td>29.2%</td><td>45.2%</td></tr><tr><td>Concat + Analogy</td><td>0.53</td><td>99.7%</td><td>99.4%</td><td>0.48</td><td>99.6%</td><td>62.9%</td></tr><tr><td>Parameter</td><td>0.56</td><td>99.9%</td><td>100.0%</td><td>-1.88</td><td>60.8%</td><td>49.6%</td></tr><tr><td>Parameter + Analogy</td><td>0.56</td><td>99.9%</td><td>100.0%</td><td>0.55</td><td>99.8%</td><td>99.6%</td></tr></table>

Table 1: Performance of subtask controller. 'Concat' and 'Parameter' correspond to the concatenation baseline and our parameter prediction architecture. 'Analogy' indicates analogy-making regularization. 'Accuracy' represents termination prediction accuracy. We assume a termination prediction is correct only if predictions are correct throughout the whole episode.

of the flat controller architecture are described in the appendix F. The flat controller is pre-trained on the training set of subtasks. To be specific, we removed the instruction memory and fed a single instruction as an additional input (i.e.,  $\mathbf{r}_t$  is fixed throughout the episode). We found that the flat controller could not learn any reasonable policy without this pre-training step which requires modification of the architecture based on domain knowledge. After pre-training, we fine-tuned the flat controller with the instruction memory on lists of instructions. Note that the flat controller is also capable of executing instructions as well as dealing with random events in principle.

# 5.2 TRAINING DETAILS

The subtask controller consists of 3 convolution layers and 2 fully-connected layers and takes the last 2 observations concatenated through channels as input. Each subtask argument  $(g^{(i)})$  is linearly transformed and multiplied with each other to compute the joint subtask embedding. This is further linearly transformed into the weight of the first convolution layer, and the weight of the first fully-connected layer. The meta controller takes the current observation as input and has 2 convolution layers and 2 fully-connected layers where the parameters of the first convolution layer and the first fully-connected layer are predicted by the joint embedding of  $\mathbf{r}_{t-1}$ ,  $\varphi(\mathbf{g}_{t-1})$ , and  $b_{t}$ .

We implemented synchronous actor-critic with 16 CPU threads based on MazeBase (Sukhbaatar et al., 2015), each of which samples a mini-batch of episodes  $(K)$  in parallel. The parameters are updated after  $16 \times K$  episodes. The details of architectures and hyperparameters are described in the appendix F.

Curriculum Learning via a Forgiving World. We conducted curriculum training by changing the size of the grid world, the density of objects, and the number of instructions according to the agent's success rate. In addition, we trained the soft-architectures on an easier forgiving environment which generates target objects whenever they do not exist. Crucially, this allows the agent to recover from past mistakes in which it removed needed target objects. The soft-architectures are fine-tuned on the original (and far more unforgiving) environment which does not regenerate target objects in the middle of the episode. Training directly in the original environment without first training in the forgiving environment leads to too much failure at executing the task and the agent does not learn successfully. Finally, the hard-architectures are initialized by the soft-architectures and further fine-tuned on the original environment.

# 5.3 EVALUATION OF SUBTASK CONTROLLER

To see how well the subtask controller performs separately from the meta controller, we evaluated it on the training set of subtasks and unseen subtasks in Table 1. 'Concat' is a baseline architecture in which the subtask embedding is concatenated to the input and the first fully-connected layer. It is shown that our parameter prediction architecture consistently outperforms the concatenation baseline. In addition, analogy-making regularization is crucial for generalization to unseen subtasks in both architectures. These results suggest that both the parameter prediction architecture and analogy-making regularization are complementary and play an important role in generalization to unseen subtasks.

In addition, the subtask controller learned a non-trivial policy by exploiting causal relationships. For example, when [Pick up, egg] is communicated as the subtask arguments, but a duck is very close to the agent, it learned to transform the duck and pick up the resulting egg because transforming the duck transforms it to an egg in our environment. More analysis of the subtask controller and the effect of analogy-making regularization is discussed in the appendix A and B.

<table><tr><td colspan="2"></td><td>Train</td><td>Test #1</td><td>Test #2</td><td>Test #3</td><td>Test #4</td></tr><tr><td colspan="2">Set of instructions</td><td>Seen</td><td>Seen</td><td>Unseen</td><td>Seen w/o all</td><td>Unseen w/o all</td></tr><tr><td colspan="2">Num of instructions</td><td>4</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td rowspan="4">Forgiving</td><td>Shortest Path</td><td>-1.56 (99.6%)</td><td colspan="2">-11.94 (99.1%)</td><td colspan="2">-9.62 (99.1%)</td></tr><tr><td>Near-Optimal</td><td>-0.96 (99.6%)</td><td colspan="2">-9.99 (99.1%)</td><td colspan="2">-8.19 (99.1%)</td></tr><tr><td>Flat</td><td>-1.64 (85.8%)</td><td>-14.53 (65.9%)</td><td>-17.25 (23.7%)</td><td>-12.38 (60.4%)</td><td>-14.18 (16.7%)</td></tr><tr><td>Hierarchical-TA-Analyogy</td><td>-1.05 (92.4%)</td><td>-11.06 (86.2%)</td><td>-13.69 (51.2%)</td><td>-8.54 (91.9%)</td><td>-9.91 (75.2%)</td></tr><tr><td rowspan="7">Original</td><td>Shortest Path</td><td>-1.62 (99.7%)</td><td colspan="2">-11.94 (99.4%)</td><td colspan="2">-8.72 (99.6%)</td></tr><tr><td>Near-Optimal</td><td>-1.34 (99.5%)</td><td colspan="2">-10.30 (99.3%)</td><td colspan="2">-7.62 (99.4%)</td></tr><tr><td>Flat</td><td>-2.38 (76.0%)</td><td>-18.83 (0.1%)</td><td>-18.92 (0.0%)</td><td>-15.09 (0.0%)</td><td>-15.17 (0.0%)</td></tr><tr><td>Hierarchical</td><td>-2.04 (72.8%)</td><td>-16.85 (16.6%)</td><td>-17.66 (6.9%)</td><td>-10.99 (49.4%)</td><td>-11.40 (47.4%)</td></tr><tr><td>Hierarchical-Analyogy</td><td>-1.74 (81.0%)</td><td>-15.89 (28.0%)</td><td>-17.23 (11.3%)</td><td>-10.11 (61.8%)</td><td>-10.66 (57.7%)</td></tr><tr><td>Hierarchical-TA</td><td>-1.38 (92.6%)</td><td>-12.96 (62.9%)</td><td>-17.19 (13.0%)</td><td>-9.11 (74.4%)</td><td>-10.37 (61.2%)</td></tr><tr><td>Hierarchical-TA-Analyogy</td><td>-1.26 (95.5%)</td><td>-11.30 (81.3%)</td><td>-14.75 (40.3%)</td><td>-8.24 (85.5%)</td><td>-9.51 (73.9%)</td></tr></table>

Table 2: Performance of meta controller. Each column corresponds to different evaluation sets of instructions, while each row corresponds to different configurations of our architecture and the flat controller. Test #3 and Test #4 do not include 'Transform/Pick up all X' instructions. 'TA' indicates the meta controller with temporal abstraction. Each entry in the table represents reward with success rate in parentheses averaged over 10-best runs among 20 independent runs. 'Shortest Path' is a hand-designed policy which executes instructions optimally based on the shortest path but ignores enemies. 'Near-Optimal' is a near-optimal policy that executes instructions based the shortest path and transforms enemies when they are close to the agent. 'Forgiving' rows show the result from the forgiving environment used for curriculum learning where target objects are regenerated whenever they do not exist in the world.

![](images/ae3a9e2be884811e6c3930ea9a0503a916339f790558760eac63895019e37bb8.jpg)  
Figure 3: Performance per number of instructions. From left to right, the plots show reward, success rate, the number of steps, and the average number of instructions completed respectively. Solid and dashed curves show the performances on seen instructions and unseen instructions respectively.

# 5.4 EVALUATION OF META CONTROLLER

We evaluated the meta controller separately from the subtask controller by providing the best-performing subtask controller during training and evaluation. The results are summarized in Table 2 and Figure 3. Note that there is a discrepancy between reward and success rate, because success rate is measured only based on the instruction execution, while reward takes into account the background task (i.e., handling randomly appearing enemy) as well as the instruction execution.

Overall performance. Table 2 shows that our hierarchical agent with temporal abstraction and analogy-making regularization, denoted Hierarchical-TA-Analogy in the table, can handle 20 seen instructions (Test #1) and 20 unseen instructions (Test #2) correctly with reasonably high success rates. In addition, that agent learned to deal with enemies whenever they appear, and thus it outperforms the 'Shortest Path' policy which is near-optimal in executing instructions while ignoring enemies. We further investigated how the number of instructions affects the performance in Figure 3. Although the performance is degraded as the number of instructions increases, our architecture finishes 18 out of 20 seen instructions and 12 out of 20 unseen instructions on average. These results show that our agent is able to generalize to longer compositions of instructions as well as unseen instructions by just learning to solve short sequences of a subset of instructions.

Flat vs. Hierarchy. All our hierarchical controllers outperform the flat controller both on the training tasks and longer/unseen instructions (see Table 2). We observed that the flat controller learned a sub-optimal policy which assumes that 'Transform/Pick up X' instructions are identical to 'Transform/Pick up all X' instructions. In other words, it always transforms or picks up all existing targets. Although this simple strategy is a reasonable sub-optimal policy because such wrong actions are not explicitly penalized in our environment other than through the accumulating penalty per-time-step, it often unnecessarily removes objects that can be potentially target objects in the future instructions. This is why the flat controller performs reasonably well on the short sequences of instructions (training) where such cases are rare and on the forgiving environment where target

![](images/4443cd34296c79fb015d6f78435a71aa9dcfbae0183923629493744aa32c4a45.jpg)  
A

![](images/af8c796767f40c00bf62e61504909b301b418b038041e1914d5b3bc7f9920ae2.jpg)  
Figure 4: Analysis of the learned policy. 'Update' shows our agent's internal update decision. 'Shift' shows our agent's memory-shift decision which is either -1, 0, or +1 from top to bottom. The bottom text shows the instruction indicated by the memory pointer, while the top text shows the subtask chosen by the meta controller. (A) the agent transforms the pig given 'Transform Pig' instruction and decides to update the subtask (Update is true) and move to the next instruction. (B) an enemy (red) appears while the agent is executing 'Pick up all meat' instruction (green boxes for meat). The agent changes the subtask to [Transform, enemy]. (C) the agent successfully transforms the enemy and sets the subtask to [Pick up, meat] to resume executing the instruction. (D) the agent picks up the last meat in the world, moves the memory pointer to the next instruction, and sets a new subtask according to the next instruction.

![](images/6453a9d415e956cf913cd4e72468bb6dba79b7d18c1eacd5b928cf6ab4d28079.jpg)  
B

![](images/96d97a543e46e7d50a3bddce63968ba605c6591ff663d21d46ee6874698cff8a.jpg)  
C

![](images/c1e7017f19f6de87a1abdf44b907f030ecdc72bb3169ed7937818ee28d8e6cb6.jpg)  
D

objects are restored whenever needed. But, it completely fails on longer instructions in the original environment because the entire task becomes unsolvable when target objects are removed in error. This implies that the flat controller struggles with detecting when a subtask is finished precisely, whereas our hierarchical controllers can easily detect when a subtask is done, because the subtask controller in our communicating architecture provides a termination signal to the meta controller.

In addition, the flat controller tends to ignore enemies, while the hierarchical controllers try to deal with enemies whenever they exist by changing the subtask-arguments communicated by the meta controller to the subtask controller, which is a better strategy to maximize the reward. The flat controller instead has to use primitive actions to deal with both instructions and enemies. This implies that our communicating hierarchical controllers have more advantages for context switching between different sources of tasks (i.e., executing instructions and dealing with enemies).

Finally, we observed that the flat controller often makes many mistakes on unseen instructions (e.g., transform X given 'Visit X' as instruction). In contrast, the hierarchical controllers do not make such mistakes as the subtask controller generalizes well to unseen instructions as discussed in Section 5.3.

Effect of Analogy-making. Table 2 shows that analogy-making significantly improves generalization performance especially on Test #2 (Hierarchical-Analyse outperforms Hierarchical, and Hierarchical-TA-Analyse outperforms Hierarchical-TA). This implies that given an unseen target object for the 'Transform/Pick up all' instruction, the meta controller without analogy-making tends to fail to check if the target object exists or not. On the other hand, there is almost no improvement by using analogy-making on Test #3 and Test #4 where there are no 'all' instruction. This is because the meta controller can simply rely on the subtask termination  $(b_{t})$  given by the subtask controller to check if the current instruction is finished for non-'all' instructions, and the subtask controller (trained with analogy-making) successfully generalizes to unseen subtasks and provides accurate termination signals to the meta controller. The empirical results showing that analogy-making consistently improves generalization performance in both non-analogy-making controllers suggests that analogy-making is crucial for generalization to unseen tasks.

Effect of Temporal Abstraction. To see the effect of temporal abstractions, we trained a baseline that updates the memory pointer and the subtask at every time-step which is shown as 'Hierarchical' and 'Hierarchical-Analogy' in Table 2. It turns out that the agent without temporal abstractions performs much worse both on the training tasks and testing tasks. We hypothesize that temporal credit assignment becomes easier with temporal abstractions because the subtask updater (described in Section 3.2.1) can operate at a larger time-scale by decoupling the update decision from the subtask selection. In particular, given 'all' instructions, the agent should repeat the same subtask while not changing the memory pointer for a long time and the reward is even more delayed. This can possibly confuse the subtask updater without temporal abstractions because it should make the

same decision for the entire time-steps of such instructions. In contrast, the subtask updater with temporal abstractions can get a direct feedback from the long-term future, since one decision made by the subtask updater results in multiple primitive actions. We conjecture that this is why the agents learn more stably with temporal abstractions under delayed reward.

Analysis of The Learned Policy. We visualized our agent's behavior on a task with a long list of instructions in Figure 4. We observed that our meta controller learned to communicate the correct subtask-arguments to the subtask controller and learned to move precisely to the next instruction by shifting the memory pointer whenever the instruction is finished. More interestingly, whenever an enemy appears, our meta controller immediately changes the subtask to [Transform, enemy] regardless of the instruction and resumes executing the instruction after dealing with the enemy. Throughout the background task and the 'all' instructions, the meta controller keeps the memory pointer unchanged as illustrated in (B-D) in the figure. In addition, the agent learned to update the memory pointer and the subtask-argument almost only when it is needed, which provides the subtask updater with temporally-extended actions. This is not only computationally efficient but also useful for learning a better policy as discussed above.

# 6 CONCLUSION

In this paper, we explored zero-shot task generalization in RL with a new problem where the agent is required to execute a sequence of instructions and to generalize over longer sequences of (unseen) instructions without additional learning. To solve the problem, we presented a hierarchical deep RL architecture in which a meta controller learns a closed-loop policy of subtask-argument communications to a subtask controller which in turn acts in the environment to accomplish the subtask communicated by the meta controller and when it has finished the subtask communicates this accomplishment back to the meta controller. Our architecture not only generalizes to unseen tasks after training but also deals with random events relevant to a background task. In addition, we proposed several techniques that led to improvements in both training and generalization performance. First, analogy-making regularization turned out to be crucial for generalization to unseen subtasks. Second, learning temporal abstractions improved the performance by making the subtask updater operate at a larger time-scale. Although our architecture's abilities are demonstrated in a relatively simple 2D grid-world, the proposed idea in principle is applicable to many domains. One interesting line of future work would be to define and solve richer task instructions such as conditional statements (i.e., IF-THEN-ELSE) and loop instructions (i.e., collect 3 target objects). Moreover, end-to-end training of the whole hierarchy and discovering the subtask decomposition would be important future work.

# REFERENCES

P.-L. Bacon and D. Precup. The option-critic architecture. In NIPS Deep Reinforcement Learning Workshop, 2015.  
L. Bertinetto, J. F. Henriques, J. Valmadre, P. H. Torr, and A. Vedaldi. Learning feed-forward one-shot learners. arXiv preprint arXiv:1606.05233, 2016.  
D. L. Chen and R. J. Mooney. Learning to interpret natural language navigation instructions from observations. In Proceedings of the 25th AAAI Conference on Artificial Intelligence (AAAI-2011), 2011.  
T. G. Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. J. Artif. Intell. Res.(JAIR), 13:227-303, 2000.  
M. Ghavamzadeh and S. Mahadevan. Hierarchical policy gradient algorithms. In ICML, pages 226-233, 2003.  
A. Graves, G. Wayne, and I. Danihelka. Neural tuning machines. arXiv preprint arXiv:1410.5401, 2014.  
R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), volume 2, pages 1735-1742. IEEE, 2006.

D. Isele, M. Rostami, and E. Eaton. Using task features for zero-shot knowledge transfer in lifelong learning. In *IJCAI*, 2016.  
V. R. Konda and J. N. Tsitsiklis. Actor-critic algorithms. In NIPS, volume 13, pages 1008-1014, 1999.  
T. D. Kulkarni, K. R. Narasimhan, A. Saeedi, and J. B. Tenenbaum. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. arXiv preprint arXiv:1604.06057, 2016.  
J. Lei Ba, K. Swersky, S. Fidler, et al. Predicting deep zero-shot convolutional neural networks using textual descriptions. In Proceedings of the IEEE International Conference on Computer Vision, pages 4247-4255, 2015.  
S. Levine and V. Koltun. Guided policy search. In Proceedings of The 30th International Conference on Machine Learning, pages 1-9, 2013.  
M. MacMahon, B. Stankiewicz, and B. Kuipers. Walk the talk: Connecting language, knowledge, and action in route instructions. In Proceedings of the 21st National Conference on Artificial Intelligence (AAAI-2006), 2006.  
H. Mei, M. Bansal, and M. R. Walter. Listen, attend, and walk: Neural mapping of navigational instructions to action sequences. arXiv preprint arXiv:1506.04089, 2015.  
R. Memisevic and G. E. Hinton. Learning to represent spatial transformations with factored higher-order boltzmann machines. Neural Computation, 22(6):1473-1492, 2010.  
E. Parisotto, J. L. Ba, and R. Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. arXiv preprint arXiv:1511.06342, 2015.  
R. Parr and S. J. Russell. Reinforcement learning with hierarchies of machines. In NIPS, 1997.  
S. Reed, K. Sohn, Y. Zhang, and H. Lee. Learning to disentangle factors of variation with manifold interaction. In Proceedings of the 31st International Conference on Machine Learning (ICML-14), pages 1431-1439, 2014.  
S. E. Reed, Y. Zhang, Y. Zhang, and H. Lee. Deep visual analogy-making. In Advances in Neural Information Processing Systems, pages 1252-1260, 2015.  
A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell. Policy distillation. arXiv preprint arXiv:1511.06295, 2015.  
T. Schaul, D. Horgan, K. Gregor, and D. Silver. Universal value function approximators. In Proceedings of The 32nd International Conference on Machine Learning, pages 1312-1320, 2015.  
J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.  
D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
S. P. Singh. The efficient learning of multiple task sequences. In NIPS, 1991.  
S. P. Singh. Transfer of learning by composing solutions of elemental sequential tasks. Machine Learning, 8(3-4):323-339, 1992.  
S. Sukhbaatar, A. Szlam, G. Synnaeve, S. Chintala, and R. Fergus. Mazebase: A sandbox for learning from games. arXiv preprint arXiv:1511.07401, 2015.  
R. S. Sutton, D. Precup, and S. Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1):181-211, 1999.

S. Tellex, T. Kollar, S. Dickerson, M. R. Walter, A. G. Banerjee, S. J. Teller, and N. Roy. Understanding natural language commands for robotic navigation and mobile manipulation. In AAAI, 2011.  
S. Tellex, R. A. Knepper, A. Li, D. Rus, and N. Roy. Asking for help using inverse semantics. In Robotics: Science and Systems, 2014.  
C. Tessler, S. Givony, T. Zahavy, D. J. Mankowitz, and S. Mannor. A deep hierarchical approach to lifelong learning in mycraf. CoRR, abs/1604.07255, 2016.  
A. S. Vezhnevets, V. Mnih, J. Agapiou, S. Osindero, A. Graves, O. Vinyals, K. Kavukcuoglu, et al. Strategic attentive writer for learning macro-actions. arXiv preprint arXiv:1606.04695, 2016.  
W. Zaremba and I. Sutskever. Reinforcement learning neural tuning machines. arXiv preprint arXiv:1505.00521, 2015.
