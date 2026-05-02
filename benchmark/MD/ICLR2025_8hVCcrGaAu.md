# EDISON: EFFICIENT DESIGN-AND-CONTROL OPTIMIZATION WITH REINFORCEMENT LEARNING AND ADAPTIVE DESIGN REUSE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Seeking good designs is a central goal of many important domains, such as robotics, integrated circuits (IC), medicine, and materials science. These design problems are expensive, time-consuming, and traditionally performed by human experts. Moreover, the barriers to domain knowledge make it challenging to propose a universal solution that generalizes to different design problems. In this paper, we propose a new method called Efficient Design and Stable Control (EDiSon) for automatic design and control in different design problems. The key ideas of our method are (1) interactive sequential modeling of the design and control process and (2) adaptive exploration and design replay. To decompose the difficulty of learning design and control as a whole, we leverage sequential modeling for both the design process and control process, with a design policy to generate step-by-step design proposals and a control policy to optimize the objective by operating the design. With deep reinforcement learning (RL), the policies learn to find good designs by maximizing a reward signal that evaluates the quality of designs. Furthermore, we propose an adaptive exploration and replay strategy based on a design memory that maintains high-quality designs generated so far. By regulating between constructing a design from scratch or replaying a design from memory to refine it, EDiSon balances the trade-off between exploration and exploitation in the design space and stabilizes the learning of the control policy. In the experiments, we evaluate our method in robotic morphology design and Tetris-based design tasks. Our results show that our method effectively learns to explore high-quality designs and outperforms previous results in terms of design score and efficiency.

# 1 INTRODUCTION

Design optimization presents a key challenge across various domains such as robotics (Gupta et al., 2021), integrated circuits (IC) (Mirhoseini et al., 2021), medicine (Coley et al., 2017), and materials science (Ghugare et al., 2023; Govindarajan et al., 2024). Traditionally, design problems are tackled by human experts through iterative manual experimentation, incurring significant costs in both time and resources. Moreover, the required specialized domain knowledge further complicates the design process and increases the need for domain expertise, hindering the generalizability of traditional approaches. Therefore, developing an efficient and general framework for different design problems with little human intervention and specialized domain knowledge is essential.

Recent advancements in reinforcement learning (RL) have made design automation a promising application (Jeong & Jo, 2021; Budak et al., 2022; Dworschak et al., 2022; Govindarajan et al., 2024). RL can rapidly discover and test potential solutions through interacting with design simulators (Sternke & Karpiak, 2023), enabling faster exploration than humans. However, the combinatorial complexity of design space often results in very few valuable designs as well as exponentially many paths to find them (Mouret & Clune, 2015; Colas et al., 2020). In addition to the difficulty of exploring valuable designs in a large and complex space, the challenge is further exacerbated when constructing the design, which is only part of the problem. This occurs when a given design also requires a control policy to achieve its task and evaluate the quality of each design (Gupta et al., 2021). For instance, constructing a robot optimized for locomotion requires both a suitable morphology design and a

![](images/8592824865eb877fe34e7f847fb5a4abeafdfdf5f3a2c57d880fadf7b26cb7df.jpg)  
Figure 1: The illustration of Efficient Design and Stable Control (EDiSon). The design policy takes steps to generate the design, which is followed by the control policy. Both the design policy and control policy learn from the return signals. Moreover, the design memory selectively stores and reuses the designs to balance the exploration-exploitation with a bandit meta-controller.

control policy that maximizes the robot's locomotion capabilities, inducing a multi-level optimization problem.

In the multi-level optimization problem, we have to address two distinct challenges: (1) Constructing the design as a Markov Decision Process (MDP) with unique transition dynamics and (2) Learning a control policy for that MDP. These problems, while both tractable with reinforcement learning (RL), have different priorities. The first problem focuses on exploring the search space for optimal designs, while the second often suffers from sample inefficiency as each new design may need a newly trained control policy. The interaction between these creates a non-stationary optimization problem requiring additional regularization for better convergence.

To address these challenges, we formulate design optimization as a multi-step MDP and propose a general framework with three key components: the design MDP for design optimization, the control MDP for control optimization, and the design buffer. The design buffer maintains a prioritized queue of high-performing designs, reducing non-stationarity and encouraging exploration-exploitation balance. We employ a bandit-based meta-controller to adjust the exploration probability dynamically, ensuring efficient and adaptive learning. This approach effectively integrates design and control optimization, leveraging past successes while continually seeking new possibilities.

Based on our general framework, we present a practical method for efficient design-and-control automation called Efficient Design and Stable Control (EDiSon), which is illustrated in Figure 1. The design policy iteratively generates designs, maximizing the reward signal from the control policy, thereby guiding optimization toward promising designs. We implement design memory through a buffer that collects high-performing and diverse designs. Our adaptive exploration and replay strategy dynamically balances between creating new designs and refining existing ones, encouraging the emergence of diverse, high-quality designs by effectively leveraging past successes while continually seeking new possibilities. The main contributions of our work are summarized as follows:

- A General and Efficient RL Framework for Design Optimization: We introduce an efficient and general framework that integrates design and control optimization into a multi-step MDP. This framework effectively addresses the dual challenges of optimizing both design and control policies, offering a more efficient and comprehensive approach to design automation.  
- Adaptive Exploration-Exploitation Trade-off in Design Optimization: We introduce a practical method, EDiSon, based on adaptive exploration and design replay. Our method leverages a bandit-based meta-controller to dynamically balance exploration and exploitation, enhancing the efficiency of design-and-control automation. By reusing successful designs from a design buffer, EDiSon ensures continuous improvement and optimal performance.

- The State-of-the-art Efficiency and Performance across Various Design Tasks: Through extensive experiments, we demonstrate that EDiSon significantly outperforms existing methods. EDiSon achieves superior results in robotic morphology design and Tetris-based design tasks, showcasing its effectiveness and efficiency.

# 2 RELATED WORK

Machine Learning for Design Autonomous design research in robotics has advanced through various approaches that have broadly focused on optimizing morphology and control. Early works proposed evolutionary algorithms to adapt the morphology of rigid body and soft body robots to solve pushing or locomotion tasks (Lipson & Pollack, 2000; Hiller & Lipson, 2012). Subsequent work extended these ideas to learning neural controllers in parallel to the morphology (Bongard & Pfeifer, 2003). Compositional Pattern-producing networks have been shown to be good for discovering new morphologies as they could adapt to the changing number of joints in a robot (Auerbach & Bongard, 2012; Jelisavcic et al., 2019). These works illustrate the progression and integration of morphology and control in autonomous design. In addition to robotics, machine learning (ML) has also been applied to many other design problems, including building design (Sun et al., 2021), as well as materials, molecular and protein design (Govindarajan et al., 2024; Ghugare et al., 2023; Watson et al., 2023) and algorithm design (Co-Reyes et al., 2021). The difference between our problem space and the above prior work is explicitly focusing on problems that include two stages of policy learning: a design stage and a synthesis/policy learning stage.

Design Optimization with RL RL has been increasingly applied to design optimization, offering efficient methods for exploring complex design spaces. Sims (1994) pioneered the use of evolutionary algorithms with RL principles to design virtual creatures with adaptable behaviors. Gupta et al. (2021) demonstrated the significant impact of optimized morphologies on learning efficiency for targeted tasks. Yuan et al. (2022) introduced an RL framework integrating transformation and control policies to streamline robot design and operation. Ha (2019) jointly optimized agent embodiment using a population-based REINFORCE algorithm. Schaff et al. (2019) applied RL to update distributions over design parameters. These advancements highlight RL's potential to automate and enhance design optimization. RL has also been applied to many other design problems, including concrete structures (Jeong & Jo, 2021), and electronic placement on microchips (Budak et al., 2022). These prior methods make inroads in using RL for design, but they lack tools to cope with the non-stationarity of the optimization to induce higher-performing solutions. In this work, we include a design buffer for adaptively managing non-stationarity and evaluating over a larger set of tasks than prior methods.

# 3 BACKGROUND

In this section, we briefly review the fundamental background used in our work and describe important aspects of settings with joint design problems and control problems.

Markov Decision Processes (MDP) Reinforcement Learning (RL) is typically formulated with the modeling of MDP, where at every time step  $t$ , the world (including the agent) exists in a state  $\mathbf{s}_t \in S$ , where the agent is able to perform actions  $\mathbf{a}_t \in \mathcal{A}$ . The action to take is determined according to a policy  $\pi(\mathbf{a}_t | \mathbf{s}_t)$  which results in a new state  $\mathbf{s}_{t+1} \in S$  and reward  $r_t = R(\mathbf{s}_t, \mathbf{a}_t)$  according to the transition probability function  $P(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{s}_t)$ . The goal of an RL agent is to optimize its policy  $\pi$  to maximize the future discounted reward  $J(\pi) = \mathbb{E}_{r_0, \dots, r_T} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$ , where  $T$  is the max time horizon, and  $\gamma$  is the discount factor.

Design-and-Control Problem In this paper, we aim to solve design problems, where we need to find a high-quality design and control it to optimize the design objective. Consider such a design problem with a design space  $\mathcal{D}$ , the purpose of this problem is to find an optimal design  $d^{\star} \in \mathcal{D}$  that maximizes an evaluation function  $F: \mathcal{D} \to \mathbb{R}$ , i.e.,  $d^{\star} = \max_{d} F(d)$ . The evaluation function  $F$  is not given a priori and is determined by a control process of design. For a design  $d$ , a control policy  $\pi$  operates with the design that leads to a control score  $f_{\pi}(d)$ , while the evaluation function  $F(d)$  is defined to be the best control score that can be achieved within a control policy space  $\Pi$ , i.e.,

![](images/f19fa4d44630d60d3b238d8ba3245338248418dc93354001aa05e90826e48ba2.jpg)  
Figure 2: The illustration of our general framework for learning design and control. The framework consists of three components: the design policy, the control policy, and the design memory, which interact with each other as described by the ordered texts.

$F(d) = \max_{\pi \in \Pi} f_{\pi}(d)$ . In real-world applications, one usually aims to find a set of designs that have good evaluation scores and are diverse at the same time.

# 4 A GENERAL FRAMEWORK FOR LEARNING DESIGN AND CONTROL

The design problems we address involve two interconnected challenges: discovering an optimal design (the design problem) and controlling that design to optimize a specific objective (the control problem). This dual challenge is prevalent in scenarios like designing a robotic morphology with a corresponding locomotion policy or creating building blocks for a geometric task. Solving these problems is complex due to the vast combinatorial design space and the intricate landscape of the design objective function. Additionally, control learning must generalize across various designs, further complicating the process. The interplay between design and control exacerbates the difficulty, as design evaluation signals are often noisy and dependent on the ongoing control learning process, while the control problem must handle a non-stationary distribution of designs generated in real time.

To handle these challenges, in this section, we propose a general framework for learning design and control. As illustrated in Figure 2, the framework consists of three components as introduced below.

Design As A Multi-Step MDP In this paper, we assume that the Markov assumption holds (see Appendix C Assumption 1) allowing us to formulate the design as a multi-step MDP. The design policy explores the design space and optimizes the design  $d \in \mathcal{D}$  regarding the design evaluation signal  $F(d)$ . We use sequential modeling for the design process, i.e., the design policy starts from an initial base design  $d_0$  and constructs it with step-by-step modifications to a final design  $d_T$ . We define a Design Markov Decision Process (Design MDP)  $M = (U, X, P, R, \gamma, \rho, E, D, g)$ , where  $\mu \in U$  is a state of the design process,  $x \in X$  is a design action,  $e \in E$  is an optional external information, and  $g: D \times X \to D$  describes the deterministic change of design affected by design action:

$$
\begin{array}{l l l} \mu_ {t} \triangleq (d _ {t}, e _ {t}) & \pi^ {\mathrm {D}} \left(x _ {t} \mid \mu_ {t}\right) \triangleq p \left(x _ {t} \mid d _ {t}, e _ {t}\right) & P \left(\mu_ {t + 1} \mid \mu_ {t}, x _ {t}\right) \triangleq \delta_ {d _ {t + 1}} p \left(e _ {t + 1} \mid d _ {t}, e _ {t}, x _ {t}\right) \\ \rho (\mu_ {0}) \triangleq p \left(d _ {0}, e _ {0}\right) & d _ {t + 1} \triangleq g \left(d _ {t}, x _ {t}\right) & R \left(\mu_ {t}, x _ {t}\right) \triangleq \left\{ \begin{array}{l l} F (d _ {T}) & \text {i f} t = T \\ 0 & \text {o t h e r w i s e} \end{array} \right. \end{array} \tag {1}
$$

where  $\delta_y$  denotes the Dirac delta distribution with a nonzero density only at  $y$ .

One key feature of the design-and-control problem is that each design  $d$  corresponds to an MDP task to solve, and the design process corresponds to a process of constructing an observation space  $\mathcal{O}_d$  and an action space  $\mathcal{A}_d$  for the control task. From a finer-grained perspective, the spaces  $\mathcal{O}_d, \mathcal{A}_d$  consist of the subspace sets  $\{O_i\}, \{A_i\}$ , each design action  $x_t$  corresponds to adding or removing a tuple of subspaces  $(O_i, A_i)$ , and the design change function  $g$  updates of the subspace sets and generates  $\mathcal{O}_d, \mathcal{A}_d$  based on the cartesian product of the subspaces chosen so far. Next, we move on to detail the control task associated with the design  $d$  and the observation and action spaces  $\mathcal{O}_d, \mathcal{A}_d$  constructed.

Control As A Multi-Step MDP The control policy manipulates a design with the purpose of best performing the control task. Essentially, given a design  $d$ , this is equivalent to learning the optimal policy in a Control Markov Decision Process (Control MDP)  $M_{d} = (\mathcal{S}_{d},\mathcal{A}_{d},\mathcal{O}_{d},\mathcal{O},P_{d},R_{d},\gamma ,\rho_{d},d)$ , where  $o\in \mathcal{O}$  is an observation of the environment and  $o^d\in \mathcal{O}^d$  is an observation of the design state (e.g., the proprioceptive state of a robot), and  $S_{d} = \mathcal{O}\times \mathcal{O}^{d}$ . Formally, the Control MDP  $M_{d}$  is defined as:

$$
s _ {t} \triangleq \left(o _ {t}, o _ {t} ^ {d}\right) \quad P _ {d} \left(s _ {t + 1} \mid s _ {t}, a _ {t}\right) \triangleq p \left(o _ {t + 1}, o _ {t + 1} ^ {d} \mid o _ {t}, o _ {t} ^ {d}, a _ {t}, d\right)
$$

$$
\rho_ {d} \left(s _ {0}\right) \triangleq p \left(o _ {0}, o _ {0} ^ {d}\right) \quad \pi^ {\mathrm {C}} \left(a _ {t} \mid s _ {t}, d\right) \triangleq p \left(a _ {t} \mid o _ {t}, o _ {t} ^ {d}, d\right) \quad R _ {d} \left(s _ {t}, a _ {t}\right) \triangleq r \left(o _ {t}, o _ {t} ^ {d}, a _ {t}, d\right)
$$

Ideally, the control policy maximizes the performance as  $\pi^{\mathrm{C}} = \arg \max_{\pi}J(\pi ,M_d)$ , which then serves as the design evaluation signal, i.e.,  $F(d) = J(\pi^{\mathrm{C}},M_d)$ .

Design Memory The design memory maintains a design buffer  $\mathcal{B} = \{d_i\}$ . The designs generated by the design policy are kept in  $\mathcal{B}$  selectively according to their evaluation (i.e., the maintenance module), e.g., with a probability  $p(d)\propto F(d)$ . Meanwhile, it provides designs for the learning of the design policy and the control policy (i.e., the replay module)

Our framework presents a unified mathematical model for design-and-control problems. Because the co-optimization of an MDP choice and a solution to the chosen MDP is non-stationary, our framework introduced a buffer to store recent high-value designs which also induces control of the non-stationarity of the designs. Specifically, the design memory keeps useful knowledge of diverse sets of best-performing designs to accelerate the learning process. In learning the design policy, the design memory enables the realization of an exploitation-exploration balance in the design space that also helps find good designs efficiently. In the learning of the control policy, the design memory stabilizes the distribution change of design MDPs and reduces the difficulty of learning over multiple designs, thus leading to better design evaluation.

# 5 EFFICIENT DESIGN AND STABLE CONTROL (EDISON)

In this section, we describe our approach to improving design optimization with RL by actively reusing designs and adaptively balancing the exploration-exploitation trade-off.

# 5.1 JOINT OPTIMIZATION OF DESIGN AND CONTROL USING REINFORCEMENT LEARNING

We leverage reinforcement learning to design the optimization by dividing the task into two distinct stages. The first stage, the design stage, identifies the optimal design for the control task. The second stage, the control stage, utilizes the generated design to complete the task, with RL agents evaluating each design based on reward feedback from the environment.

The optimization objective for the design stage can be formulated as:

$$
d ^ {*} = \arg \max  _ {d \in \mathcal {D}} F (d) \tag {2}
$$

Where  $F$  is the evaluation function for each design  $d$ . In our method, designs are evaluated during the control stage using a control policy  $\pi$ , making  $F$  dependent on  $\pi$ :  $F = J(\pi, d) = G_{d,\pi} = \mathbb{E}_{\pi, d}\left[\sum_{t=0}^{H} \gamma^t r_t\right]$ . Thus, the joint design and control optimization can be formulated as:

$$
\text {D e s i g n S t a g e :} \quad d ^ {*} = \arg \max  J (\pi , d)
$$

$$
\text {C o n t r o l S t a g e :} \quad \pi^ {*} = \underset {\pi} {\arg \max } J (\pi , d) \tag {3}
$$

As mentioned in Sec. 4, the agents typically learn two sub-policies,  $\pi^D$  and  $\pi^C$ , to address this joint optimization. The design policy  $\pi^D$  generates each design  $d_t$  from an initial design  $d_0$ , and the control policy  $\pi^C$  rolls out the control trajectory to evaluate each design.

While methods like Transform2Act (Yuan et al., 2022) have been successful, they often ignore the exploitation and reuse of previously discovered designs, starting from scratch with a less informative  $d_0$ , leading to inefficiency. In this paper, we propose a new design-and-control paradigm that actively exploits learned designs, enhancing efficiency and performance.

# 5.2 EXPLORATION AND EXPLOitation IN DESIGN SPACE

In this paper, we propose two general design methods. The first method involves designing from scratch, allowing for greater freedom to explore the entire design space. However, solely exploring the design space without exploiting current designs is often less effective. Therefore, the second method involves designing from good examples  $d_{\mathrm{good}}$ , enabling the agent to leverage useful and informative designs. This approach closely mirrors human design processes, where we often base our designs on prior work and masterpieces with exemplary performance. In practice, these good examples can be sourced from a design history or provided by humans prior to training.

For fairness, we propose not to rely on artificially given good examples. Instead, we let the agents exploit good examples they found throughout the entire learning process. To facilitate this, we implement a design buffer  $\mathcal{B}$  to store good designs encountered during training. Whenever the agent needs to design based on an example, it samples a good design  $d_{good} \sim \mathcal{P}_{\mathcal{B}}$  from this buffer, wherein  $\mathcal{P}_{\mathcal{B}} = \text{softmax}(G_d)$ . More implementation details of our design buffer can be found in App. G.

However, solely relying on existing good examples can lead to sub-optimal solutions by failing to explore the design space adequately. Ideally, the agent should first explore the entire design space and, once good designs have been identified, actively exploit these examples to inform further design efforts. To balance exploration and exploitation, we propose a hybrid approach combining two methods: (1) Exploration: designing from scratch and (2) Exploitation: designing from good examples. During each design stage in training, the agent decides to design from scratch with probability  $p$  and to design from good examples with probability  $1 - p$ . We call this probability  $p$  the design exploration rate which allows us to control exploration throughout the training process:

$$
\left\{ \begin{array}{l l} \text {E x p l o r a t i o n : D e s i g n f r o m S c r a t c h ,} & p \\ \text {E x p l o i t a t i o n : D e s i g n f r o m G o o d E x a m p l e s (D e s i g n R e u s e) ,} & 1 - p \end{array} \right. \tag {4}
$$

By adjusting the probability  $p$ , we can achieve an optimal trade-off between exploration and exploitation in the design optimization problem. Even with a fixed probability  $p$ , this method outperforms the original Transform2Act which is equivalent to the special case where  $p = 1$  and the agent constantly explores the design space from scratch. Our method offers better performance and efficiency, demonstrating the benefits of integrating both exploration and exploitation in the design process.

# 5.3 ADAPTIVE EXPLORATION IN DESIGN OPTIMIZATION

A fixed probability  $p$  helps balance exploration and exploitation but fails to let agents adaptively choose the best design method during different learning stages. Early in training, agents should explore widely using a higher  $p$ , while later stages should exploit good designs with a lower  $p$ .

To address this, we propose a meta-controller that dynamically adjusts the design exploration rate  $p$ , balancing exploration and exploitation. We use a multi-armed bandit (MAB) approach, where each bandit has two arms: arm = 0 for design from scratch and arm = 1 for design from good examples. At the start of each trajectory, the actor samples an arm  $k \in K = \{0,1\}$  using the probability distribution  $\mathcal{P}_K = \frac{e^{\mathrm{Score}_k}}{\sum_j e^{\mathrm{Score}_j}}$ . The design exploration rate  $p$  is given by  $p = \mathcal{P}_{arm=0}$ .

We use the Upper-Confidence Bound (UCB) score to manage the trade-off:

$$
\operatorname {S c o r e} _ {k} = V _ {k} + c \cdot \sqrt {\frac {\log \left(1 + \sum_ {j \neq k} ^ {K} N _ {j}\right)}{1 + N _ {k}}} \tag {5}
$$

where  $N_{k}$  is the number of visits to arm  $k$ ,  $V_{k}$  is the expected value of the returns, and the UCB term (i.e., the second term) ensures the agent doesn't repeatedly select the same arm, avoiding quick convergence to suboptimal solutions.

After sampling an arm, the agent decides whether to reuse a base design from the buffer  $\mathcal{B}$  or design from scratch. The design policy  $\pi^{\mathrm{D}}$  and control policy  $\pi^{\mathrm{C}}$  are applied to obtain a trajectory  $\tau_{i}$  and the return  $G_{i}$ , which updates the reward model  $V_{k}$  for the selected arm. To handle non-stationarity, we ensemble several MABs with different hyperparameters, allowing the agent to adapt to changing environments and maintain robust performance. More details are in the App. F.

# 5.4 EFFICIENT DESIGN AND STABLE CONTROL (EDISON) ALGORITHM

We summarize the complete process of our method in Algorithm 1, which illustrates the core steps of the Efficient Design and Stable Control (EDiSon) framework. The algorithm iterates over multiple design and control steps, dynamically adjusting between exploration and exploitation, and refining the policies over time to converge on an optimal design and control policy.

Algorithm 1 EDiSon  
Require: number of training iterations  $N$  , simple initial design  $d_{null}$  , initial design  $d_0$  , design buffer  $\mathcal{B}$  , bandit MAB, design policy  $\pi^{\mathrm{D}}$  , control policy  $\pi^C$  , length of design stage  $T$  1: Initialize design policy  $\pi^D$  and control policy  $\pi^C$  2: Initialize design buffer  $\mathcal{B}\gets (design = d_{null},value = 0)$  3: Initialize training data replay buffer  $\mathcal{M}\gets \emptyset$  4: for iteration  $i = 1$  to  $N$  do 5: while not reaching batch size do 6: for jth trajectory  $\tau_{j}$  do 7: // Design Stage 8: Sample arm  $k_{j}$  from the bandit MAB; 9: if  $k_{j} = 0$  then 10:  $d_0\gets d_{null}$ $\triangleright$  Design from scratch; 11: else 12:  $d_0\gets$  Sample from Buffer(B)  $\triangleright$  Design Reuse 13: end if 14: for iteration  $t = 1$  to  $T$  do 15: Sample design actions  $a_t^d$  using  $\pi^D$  16: Update design  $d_t$  with sampled actions  $a_t^d$  17: end for 18: // Control Stage 19: Use  $\pi^C$  to rollout control trajectory with design  $d_T$  , obtain trajectory return  $G_{j}$  20: Store trajectory  $j$  in data replay buffer  $\mathcal{M}\gets \tau_j$  21: Update design buffer  $\mathcal{B}\gets (design = d_T, value = G_j)$  22: Update bandit with  $(k_j,G_j)$  23: end for 24: end while 25: Update  $\pi^C$  and  $\pi^D$  using PPO with samples from  $\mathcal{M}$  26: end for 27: return Optimal design  $d^{*}$  , control policy  $\pi^C$  , design policy  $\pi^D$

# 6 EXPERIMENTAL RESULTS

Our experiments are designed to evaluate the effectiveness of our methods across various design optimization tasks, from robotic morphology design to microfabrication-inspired problems. Specifically, we explore Tetris-like design challenges, where a set of designed blocks is manipulated to achieve either a Tetris or target deposition pattern. We propose to address the following questions:

1. How does EDiSon perform compared to prior work in various design tasks (See Figure 3)? Can our methods find better designs (See Figure 5)?  
2. How much does adaptively balancing the exploration and exploitation in design optimization assist in finding higher-value solutions (See Figure 6)? Why not just use a fixed design exploration rate  $p$  (See Figure 6)?  
3. How much do core components of our framework, such as design reuse and adaptive exploration-exploitation trade-off, contribute to the results (See Figure 7)?

# 6.1 EXPERIMENTAL SETUP

We conduct experiments across several design-based tasks, including robotic morphology design and Tetris-based design problems. To ensure a fair comparison, we follow the same settings and network

structure for the robotic morphology design tasks as Transform2Act (Yuan et al., 2022) and adopt a 3-layer MLP for all policies and critics in the Tetris-related task. We use PPO (Schulman et al., 2017) to learn both our design policy, control policy, and critics. We utilize a separate evaluation process to continuously record scores, measuring the undiscounted episodic returns averaged over five seeds. To provide comprehensive insights, we present full learning curves for each task, addressing any issues associated with aggregated metrics. In addition to the average score, we highlight the best designs discovered by our agent during the learning process, showcasing our method's superiority in design exploration. More implementation details can be found in App. I.

**Environments.** We evaluate our algorithm on the following tasks: (1) Swimmer: A 2D agent operating in water with 0.1 viscosity, confined to the xy-plane, aiming to maximize forward speed along the x-axis. (2) 2D Locomotion: A 2D agent in the xz-plane that moves forward as quickly as possible, with rewards based on forward velocity. (3) 3D Locomotion: A 3D agent navigating along the x-axis, striving for maximum forward speed, rewarded based on velocity. (4) Gap Crosser: A 2D agent navigating across periodic gaps on the xz-plane, with rewards linked to forward speed. Additionally, we provide supplementary results for other design tasks, such as Tetris rewarded by playtime (i.e., design blocks to play Tetris longer) and Microfabrication Deposition rewarded by matching rate (i.e., design blocks to etch the deposition layers and match target pattern better) to further demonstrate our method's capabilities beyond robot design tasks (see App. L). More details about these tasks can be found in App. D.

![](images/f4ec9906317005c13f88c7ccaf57232715e698b74f3ab855ab597e37bba34ebd.jpg)

![](images/62aa62d21824ac86879330ff37693e78bb42834cc2ad4d8915f012bf0282c4e3.jpg)  
(a) 3D Locomotion  
(e) 3D Locomotion  
Figure 3: Baseline Comparison in Robotic Morphology Design Tasks. The upper panel (i.e.,  $a - d$ ) is the comparison in terms of average return, and the lower panel (i.e.,  $e - h$ ) is for the score of best design discovered (Top 1 Score). For each robot task, we plot the mean and standard deviation of total rewards against the number of simulation steps for all methods. Each curve shows a smoothed moving average over 5 points.

![](images/81c2db7e2b3b5cb84244b42d1e4287ee3383f4983a08489586e5b63303cfbecd.jpg)

![](images/4498b70eb9add224cc7697c9b8956ea376eac0d0064fd0324db4152d5be45668.jpg)  
(b) Swimmer  
(f) Swimmer

![](images/7e5591b74cec85690af498535ce836b3b21faf9e9293ed5ffb02e657a9b5f084.jpg)

![](images/59bc95be5a7bd378e276f0463379f801e4617c49dfdad374eaa681e0a51e2142.jpg)  
(c) 2D Locomotion  
(g) 2D Locomotion

![](images/797c662d43a3f2090980cce5611067e3a5f6fb5a24a0111e3aceceff0d6da466.jpg)

![](images/e4b7b1c8b80c03e51d905cd03eaba8f1652a22ea68d86aa3b12841ed3d726327.jpg)  
(d) Gap Crosser  
(h) Gap Crosser

# 6.2 SUMMARY OF RESULTS

Our experimental results in Figure 3 demonstrate the superiority of our proposed methods over the baseline, Transform2Act. The Bandit approach consistently achieves higher returns across all tasks, illustrating its effectiveness in dynamically balancing exploration and exploitation. This adaptability is crucial for optimizing performance in varied and complex environments. While the fixed design exploration parameter  $p$  also shows improvements, it remains inferior to the Bandit method, underscoring the importance of an adaptive balance in design optimization. The success of our methods can be attributed to several key factors: (1) Design Reuse: By leveraging effective designs discovered during the training process, our methods avoid the inefficiencies of always starting from scratch. Reusing successful designs enhances learning efficiency and accelerates performance improvements. (2) Adaptive Trade-off: The Bandit method enables the agent to dynamically adjust its exploration-exploitation balance during design optimization, leading to more efficient learning and higher performance. This adaptability ensures that the agent explores new designs early in training and exploits successful designs as they are discovered.

![](images/c1c017bd3add05f942dd8e2c4545a530b255a282681440029e424457397c5dff.jpg)  
(a) Tetris

![](images/95a554e9f9164ee25cbed3f00ac66add955eab10ae77773b82373d138460fe52.jpg)  
(b) Deposition

![](images/93c954e71ca8dbc84b0c405c9716beb740c46035b890b5ad5e0f32933f19fcac.jpg)  
(c) Ours

![](images/9955e759425e7a09246dea65e1bfb093d95bb34333f498610e6ce17e0a5a1c45.jpg)  
(d) Transform2Act

![](images/87b09917339043dbb65de986ea853a84eb1754945fc77a74f8866aaf73903a0c.jpg)  
Figure 4: Baseline Comparison and Best Design Discovered in Tetris-Based Tasks. (a) and (b) show the learning curve in Tetris-like Tasks. (c) and (d) show the best design in Tetris Tasks, where agents have to find 4 blocks, each represented as a  $3 \times 3$  grid with 4 squares filled (the white one).  
(a) Ours

![](images/6cdcc90565fcfffa12809a673abe807fc92f8ddcb274d1c126cb06fdef076fa4.jpg)  
(b) Transform2Act

![](images/e54e95136bfc3eab3ecf80a9359309ee464d2db0330f5b33693b42d02a56714f.jpg)  
(c) Ours

![](images/614ea3c7bd7eef48769e4000d7bc92d3ac7330ed7998b4c2da0a227e651162e7.jpg)  
(d) Transform2Act

![](images/304358ec972197b9ce592daf90a5a553bdce793d197873eb129f805394ce6c7f.jpg)  
Figure 5: Best Design Discovered in Robotic Morphology Design Tasks. (a) and (b) show the best designs found in the Gap Crosser task by our method (reward: 11572) and Transform2Act (reward: 4579). (c) and (d) illustrate the best designs found in the 2D Locomotion task by our method (reward: 15459) and Transform2Act (reward: 11416). More discovered designs can be found in App. E.  
(a) 2D Locomotion  
Figure 6: Case Study Results. For each robot task, we plot the mean and standard deviation of total rewards against the number of simulation steps for all methods.

![](images/026bd9ae623b4903fb766388840148817d46d1dcab478589ea651e29b2ee9c92.jpg)  
(b) Swimmer

![](images/da3b6817547ac9b78f6fb4b38f2c73e7ad0f1c15702fdc24cd95a56f7501245c.jpg)  
(c) 2D Locomotion

Similar results are observed in Tetris-related design tasks in Figure 4, where our method stabilizes learning curves, as detailed in Appendix L. Additionally, in the Microfabrication Deposition tasks shown in Figure 4, our method achieves better final performance than the Transform2Act method, demonstrating our effectiveness and adaptability across a range of tasks.

Further investigation into the best designs found by our methods can also help us to understand the results, which has been illustrated in Figure 5. In the Gap Crosser Task, our bipedal design (Figure 5a) offers enhanced stability and efficiency with its upright posture and elongated limbs, enabling better gap navigation than the sprawled configuration of Transform2Act's design (Figure 5b). For the 2D Locomotion Task, our design (Figure 5c) optimizes limb placement by reducing an unnecessary joint on the tail foot and adding one to the forelimb, resulting in improved speed and agility. Conversely, Transform2Act's design (Figure 5d) retains an additional hind limb, which seems less efficient. Overall, our designs are more structurally optimized for their respective tasks. For the Tetris task, our method outperforms Transform2Act by discovering four identical symmetric block structures. Our blocks simplify the learning of the control policy, facilitate continuous gameplay, and enable efficient line clearing. A more detailed analysis can be found in App. E.3.

![](images/d252dc96f49eb1e3ce5971f1fc6f010e57c7f97a706d3c63032a3a8e7d4ffb0f.jpg)  
(a) 2D Locomotion

![](images/fefc3f768b85163c37c400d1aa653471471692e6bc1231798cfcbdd48b9c8486.jpg)  
Figure 7: Ablation Study Results. The mean and standard deviation of each method over 5 random seeds are plotted. Note that Main Method means EDiSon (ours).  
(b) Swimmer

![](images/ebdae5a25d6dfb6afd3b41fc5dcd264f1753f864cf5b4cbf70abe84b6959ea4e.jpg)  
(c) Gap Crosser

# 6.3 CASE STUDY: EXPLORATION-EXPLOITATION TRADE-OFF

We divided the design exploration rate  $p$  into ten equal intervals from 0 to 1, creating methods with different exploration preferences. These methods ranged from extreme exploitation  $(p = 0)$  to extreme exploration  $(p = 1$ , corresponding to Transform2Act). The results in Figures 6a and 6b show that different tasks have distinct optimal design exploration rates. This variability underscores that achieving a balance between exploration and exploitation is non-trivial and crucial for success.

Additionally, we analyzed the design exploration rate control curve of our Bandit-based method (Figure 6c). The results demonstrate that our Bandit-based meta-controller effectively adjusts the exploration-exploitation trade-off dynamically. Our method promotes extensive exploration during early training stages, which helps discover diverse and potentially optimal designs. As training progresses, the meta-controller gradually shifts towards exploitation, utilizing the accumulated design knowledge to optimize performance. This adaptability ensures that the agent efficiently explores the design space and exploits successful designs, leading to superior performance across tasks.

# 6.4 ABLATION STUDIES

In our ablation studies, we examine two critical components: the adaptive exploration-exploitation trade-off and design reuse via the design buffer. We evaluate several variants to highlight their impact: (1) Ours w/o Bandit: Removes the adaptive mechanism. (2) Ours w/o Exploitation: Eliminates the design buffer, requiring designs from scratch. (3) Ours w/o Exploration: Sets  $p$  to 0, disabling exploration. (4) Our Main Method: Incorporates both components.

Figure 7 shows that both design reuse and adaptive exploration-exploitation are crucial. The design buffer leverages successful designs, and the adaptive mechanism balances exploration and exploitation, enhancing performance. Neither extreme exploration nor exploitation is optimal; a balanced approach, as in our main method, yields the best results, highlighting the importance of balancing these factors in design optimization tasks.

# 7 CONCLUSION AND DISCUSSION

In this paper, we presented a novel reinforcement learning framework for design optimization, demonstrating its effectiveness across tasks ranging from robotic morphology design to Tetris-based design challenges. Our Bandit-based meta-controller dynamically balances exploration and exploitation, significantly outperforming existing methods like Transform2Act. Extensive experiments highlight the importance of adaptive strategies and design reuse, revealing the limitations of a fixed exploration rate for complex design problems. Our key contributions include an adaptive exploration-exploitation mechanism, design reuse through a design buffer, and robust evaluation via comprehensive case studies. These advancements enhance performance and efficiency, paving the way for future research in design automation and impacting various domains, from robotics to material science. However, our work has limitations. The computational complexity of our meta-controller might limit its application in resource-constrained environments. Additionally, the quality and diversity of the design buffer are crucial; a lack of initial diversity could compromise performance. Future work should address these limitations to further refine and extend the applicability of our approach.

# REFERENCES

Joshua E. Auerbach and Joshua C. Bongard. On the relationship between environmental and morphological complexity in evolved robots. In Proceedings of the 14th Annual Conference on Genetic and Evolutionary Computation, GECCO '12, pp. 521-528, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450311779. doi: 10.1145/2330163.2330238.3  
Josh C. Bongard and Rolf Pfeifer. Evolving complete agents using artificial ontogeny. In Fumio Hara and Rolf Pfeifer (eds.), Morpho-functional Machines: The New Species, pp. 237-258, Tokyo, 2003. Springer Japan. ISBN 978-4-431-67869-4. 3  
Ahmet F. Budak, Zixuan Jiang, Keren Zhu, Azalia Mirhoseini, Anna Goldie, and David Z. Pan. Reinforcement learning for electronic design automation: Case studies and perspectives: (invited paper). In 2022 27th Asia and South Pacific Design Automation Conference (ASP-DAC), pp. 500-505, 2022. doi: 10.1109/ASP-DAC52403.2022.9712578. 1, 3  
John D. Co-Reyes, Yingjie Miao, Daiyi Peng, Esteban Real, Quoc V. Le, Sergey Levine, Honglak Lee, and Aleksandra Faust. Evolving reinforcement learning algorithms. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=0XXpJ4OtjW.3  
Cédric Colas, Vashisht Madhavan, Joost Huizinga, and Jeff Clune. Scaling map-elites to deep neuroevolution. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, GECCO '20, pp. 67-75, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450371285. doi: 10.1145/3377930.3390217. URL https://doi.org/10.1145/3377930.3390217. 1  
Connor W. Coley, Luke Rogers, William H. Green, and Klavs F. Jensen. Computer-assisted retrosynthesis based on molecular similarity. ACS Central Science, 3:1237 - 1245, 2017. 1  
Fabian Dworschak, Sebastian Dietze, Maximilian Wittmann, Benjamin Schleich, and Sandro Wartzack. Reinforcement learning for engineering design automation. Advanced Engineering Informatics, 52:101612, 2022. ISSN 1474-0346. doi: https://doi.org/10.1016/j.aei.2022.101612. URL https://www.sciencedirect.com/science/article/pii/S1474034622000787.1  
Aurelien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In Jyrki Kivinen, Csaba Szepesvári, Esko Ukkonen, and Thomas Zeugmann (eds.), Algorithmic Learning Theory - 22nd International Conference, ALT 2011, Espoo, Finland, October 5-7, 2011. Proceedings, volume 6925 of Lecture Notes in Computer Science, pp. 174-188. Springer, 2011. doi: 10.1007/978-3-642-24412-4\_16. URL https://doi.org/10.1007/978-3-642-24412-4_16.27  
Raj Ghugare, Santiago Miret, Adriana Hugessen, Mariano Pheiipp, and Glen Berseth. Searching for high-value molecules using reinforcement learning and transformers. arXiv preprint arXiv:2310.02902, 2023. 1, 3  
Prashant Govindarajan, Santiago Miret, Jarrid Rector-Brooks, Mariano Pielipp, Janarthanan Rajendran, and Sarath Chandar. Learning conditional policies for crystal design using offline reinforcement learning. Digital Discovery, 2024. 1, 3  
Agrim Gupta, Silvio Savarese, Surya Ganguli, and Li Fei-Fei. Embodied intelligence via learning and evolution. Nature Communications, 12(1):5721, Oct 2021. ISSN 2041-1723. doi: 10.1038/s41467-021-25874-z. URL https://doi.org/11.1038/s41467-021-25874-z.1, 3  
David Ha. Reinforcement Learning for Improving Agent Design. Artificial Life, 25(4):352-365, 11 2019. ISSN 1064-5462. doi: 10.1162/artl_a_00301. URL https://doi.org/10.1162/artl_a_00301.3  
Jonathan Hiller and Hod Lipson. Automatic design and manufacture of soft robots. IEEE Transactions on Robotics, 28(2):457-466, 2012. doi: 10.1109/TRO.2011.2172702.3

Milan Jelisavcic, Kyre Glette, Evert Haasdijk, and A. E. Eiben. Lamarckian evolution of simulated modular robots. Frontiers in Robotics and AI, 6, 2019. ISSN 2296-9144. doi: 10.3389/frobt.2019.00009. URL https://www.frontiersin.org/articles/10.3389/frobt.2019.00009.3  
Jong-Hyun Jeong and Hongki Jo. Deep reinforcement learning for automated design of reinforced concrete structures. Computer-Aided Civil and Infrastructure Engineering, 36(12):1508-1529, 2021. doi: https://doi.org/10.1111/mice.12773. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12773. 1, 3  
Hod Lipson and Jordan B. Pollack. Automatic design and manufacture of robotic lifeforms. Nature, 406(6799):974-978, Aug 2000. ISSN 1476-4687. doi: 10.1038/35023115. URL https://doi.org/10.1038/35023115.3  
Azalia Mirhoseini, Anna Goldie, Mustafa Yazgan, Joe Wenjie Jiang, Ebrahim M. Songhori, Shen Wang, Young-Joon Lee, Eric Johnson, Omkar Pathak, Azade Nazi, Jiwoo Pak, Andy Tong, Kavya Srinivasa, William Hang, Emre Tuncer, Quoc V. Le, James Laudon, Richard Ho, Roger Carpenter, and Jeff Dean. A graph placement methodology for fast chip design. Nature, 594(7862):207-212, 2021. 1  
Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. CoRR, abs/1504.04909, 2015. URL http://arxiv.org/abs/1504.04909.1  
Charles Schaff, David Yunis, Ayan Chakrabarti, and Matthew R. Walter. Jointly learning to construct and control agents using deep reinforcement learning. In 2019 International Conference on Robotics and Automation (ICRA), pp. 9798-9805, 2019. doi: 10.1109/ICRA.2019.8793537. 3  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347.8, 32, 33  
Karl Sims. Evolving virtual creatures. In Dino Schweitzer, Andrew S. Glassner, and Mike Keeler (eds.), Proceedings of the 21th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH 1994, Orlando, FL, USA, July 24-29, 1994, pp. 15-22. ACM, 1994. doi: 10.1145/192161.192167. URL https://doi.org/10.1145/192161.192167.3  
Matt Sternke and Joel Karpiak. ProteinRL: Reinforcement learning with generative protein language models for property-directed sequence design. In NeurIPS 2023 Generative AI and Biology (GenBio) Workshop, 2023. URL https://openreview.net/forum?id=sWCsSKqkXa.1  
Han Sun, Henry V. Burton, and Honglan Huang. Machine learning applications for building structural design and performance assessment: State-of-the-art review. Journal of Building Engineering, 33: 101816, 2021. ISSN 2352-7102. doi: https://doi.org/10.1016/j.jobe.2020.101816. URL https://www.sciencedirect.com/science/article/pii/S2352710220334495.3  
Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. De novo design of protein structure and function with rfdiffusion. Nature, 620(7976):1089-1100, 2023. 3  
Ye Yuan, Yuda Song, Zhengyi Luo, Wen Sun, and Kris M. Kitani. Transform2act: Learning a transform-and-control policy for efficient agent design. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=UcDUxjPYWSr.3,5,8,13,15,19,25,31, 32,33,38,40
