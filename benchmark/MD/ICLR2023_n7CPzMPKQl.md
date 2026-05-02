# INTEGRATING SYMMETRY INTO DIFFERENTIABLE PLANNING WITH STEERABLE CONVOLUTIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study how group symmetry helps improve data efficiency and generalization for end-to-end differentiable planning algorithms, when symmetry appears in decision-making tasks. Motivated by equivariant convolution networks, we treat the path planning problem as signals over grids. We show that value iteration in this case is a linear equivariant operator, which is a (steerable) convolution. This extends Value Iteration Networks (VINs) on using convolutional networks for path planning with additional rotation and reflection symmetry. Our implementation is based on VINs and uses steerable convolution networks to incorporate symmetry. The experiments are performed on four tasks: 2D navigation, visual navigation, 2 degrees of freedom (2DOFs) configuration space and workspace manipulation. Our symmetric planning algorithms improve training efficiency and generalization by large margins compared to non-equivariant counterparts, VIN and GPPN.

# 1 INTRODUCTION

Model-based planning usually struggles in complex problems, where a solution is to apply planning in more structured and reduced space (Sutton and Barto, 2018; Li et al., 2006; Ravindran and Barto, 2004; Fox and Long, 2002). When symmetry exists in a task, it could be used for planning by effectively reducing search space. However, to use symmetry, existing planning algorithms assumes perfect dynamics knowledge and requires explicitly building equivalence classes (Fox and Long, 1999; 2002; Pochter et al., 2011; Zinkevich and Balch, 2001; Narayanamurthy and Ravindran, 2008), while specific task structure can potentially alleviate these requirements.

We use the path planning problem as an example of symmetry in a task, shown in Figure 1. Given a map  $M$  (top

row), the objective is to find optimal actions  $A = \operatorname{SymPlan}(M)$  (bottom row) to a given position (red dots). If we rotated the map  $g.M$  (top right), its solution  $g.A$  (shortest path) can also be connected by a rotation with the original solution  $A$ . Specifically, we say the task has symmetry since the solutions  $\operatorname{SymPlan}(g.M) = g.\operatorname{SymPlan}(M)$  are related by a  $\odot 90^\circ$  rotation. As a more concrete example, the action in the NW corner of  $A$  is the same as the action in the SW corner of  $g.A$ , after also rotating the arrow  $\odot 90^\circ$ . This is an example of symmetry appeared in a specific task, which can be observed before solving the task or assuming other domain knowledge. If we can use the rotation (and reflection) symmetry in this task, we effectively reduce the search space by  $|C_4| = 4$  (or  $|D_4| = 8$ ) times. Instead, classic planning algorithms like  $\mathrm{A}^*$  would require searching symmetric states (NP-hard) with known dynamics (Pochter et al., 2011).

Recently, symmetry in model-free deep reinforcement learning (RL) has also been studied (van der Pol et al., 2020a; Wang et al., 2021). A core benefit of model-free RL that enables great asymptotic performance is its end-to-end differentiability. However, it can only effectively handle pixel-level "element-wise" symmetry, such as flipping or rotating state and action together. This motivates us to combine the spirit of both: is it possible to enable end-to-end differentiable planning algorithms to make use of symmetry in environments?

![](images/b2d5ae599c0a55ceb430379380dd455bea357e29d40c6d40ce2650c8374a75aa.jpg)  
Figure 1: Symmetry in path planning. Our Symmetric Planning guarantees the solutions are same up to rotations.

In this work, we propose a framework, named Symmetric Planning (SymPlan), that allows to (1) avoid explicitly building equivalence classes for symmetric states while (2) realize planning in an end-to-end differentiable manner. We are motivated by work in the equivariant network and geometric deep learning community (Bronstein et al., 2021a; Cohen et al., 2020; Kondor and Trivedi, 2018; Cohen and Welling, 2016a;b; Weiler and Cesa, 2021): view geometric data as signals over a base space. For example, an RGB image is a signal, written as mapping  $\mathbb{Z}^2\to \mathbb{R}^3$ . The theory in equivariant networks allows to inject symmetry into operations between signals by equivariant operations, such as convolutions. It satisfies our key desiderata: equivariant networks on images do not need to explicitly consider "symmetric pixels" while guarantee symmetry properties. This avoids searching symmetric states.

We use the intuition to study a straightforward but general task: path planning. We focus on 2D grid and prove that value iteration (VI) for 2D path planning is equivariant under translations, rotations, and reflections (isometries of  $\mathbb{Z}^2$ ), and further show that VI for path planning is an instance of steerable convolution network (Cohen and Welling, 2016a). In practice, we use Value Iteration Network (VIN, (Tamar et al., 2016a)) and its variants, since they only need operations between signals. We implement the equivariant steerable version of VIN, named SymVIN, and use a variant, GPPN (Lee et al., 2018), to build SymGPPN. Both SymPlan methods achieve great improvement on training efficiency and generalization performance to unseen random maps, which showcases the advantage of exploiting symmetry from environments for planning. Our contributions include:

- We propose a framework to incorporate symmetry into planning for path planning problems (on 2D grids). We also provide the derivation in detail in appendix.  
- Since the framework proves that value iteration for path planning is a steerable CNN, we implement SymVIN by replacing the 2D convolution with steerable convolution.  
- Show significant improvement in training and generalization on 2D navigation and manipulation.

# 2 RELATED WORK

Planning with symmetries (Symmetric Planning). Symmetries widely exist in various domains, and have been exploited in classic planning algorithms as well as model checking (Fox and Long, 1999; 2002; Pochter et al., 2011; Domshlak et al.; Shleyfman; Shleyfman et al., 2015; Sievers et al.; Wehrle et al.; Abdulaziz et al.; Sievers et al., 2015; Sievers; Winterer et al.; Röger et al., 2018; Sievers et al., 2019; Fiser et al., 2019). Zinkevich and Balch (2001) show the invariance of value function for an MDP with symmetry. Narayanamurthy and Ravindran (2008) prove that finding exact symmetry in MDPs is graph isomorphism complete. However, they are based on classic planning algorithms, such as  $\mathbf{A}^*$ , and have a fundamental issue with exploitation of symmetries: they explicitly construct equivalence classes of symmetric states, which explicitly represents states and introduces symmetry breaking. Therefore, they are intractable (NP-hard) in maintaining symmetries in trajectory rollout and forward search (for large state space and symmetry group) and incompatible with differentiable pipelines for representation learning, hindering it from wider applications in RL and robotics.

State abstraction for detecting symmetries. Coarsest state abstraction aggregates all symmetric states into equivalence classes, studied in MDP homomorphisms and bisimulation (Ravindran and Barto, 2004; Ferns et al., 2004; Li et al., 2006). However, they usually require perfect MDP dynamics knowledge and do not scale up well, because of the complexity in maintaining abstraction mappings (homomorphisms) and abstracted MDPs. van der Pol et al. (2020b) integrate symmetry into model-free RL based on MDP homomorphisms (Ravindran and Barto, 2004), which avoids the challenges in handling symmetry in forward search. Park et al. (2022) learn equivariant transition models, but do not consider planning. Additionally, the formulation in commonly defined symmetric MDPs (Ravindran and Barto, 2004; van der Pol et al., 2020a; Pochter et al., 2011; Zinkevich and Balch, 2001) is different from our symmetry formulation for path planning, since they study "element-wise" symmetry for every state-action pairs and require reward to be symmetric. Our reward is not symmetric and we mainly study symmetry of the underlying domain (2D grid), as further discussed in Section B.2.

Symmetries and equivariance in deep learning. Equivariant neural networks are used to incorporate symmetry in supervised learning for different domains (e.g. grid and sphere), symmetry groups (e.g. translations and rotations), and group representations (Bronstein et al., 2021b). Cohen and Welling (2016b) introduce G-CNNs, followed by Steerable CNNs (Cohen and Welling, 2016a)

which generalizes from scalar feature fields to vector fields with induced representations. Kondor and Trivedi (2018); Cohen et al. (2020) study theory on equivariant maps and convolutions. Weiler and Cesa (2021) propose to solve kernel constraints under arbitrary representations for  $E(2)$  and its subgroups by decomposing into irreducible representations, named  $E(2)$ -CNN.

Differentiable planning. Our pipeline is based on learning to plan in a neural network in a differentiable manner. Value iteration network (VIN) (Tamar et al., 2016b) is a representative work that performs value iteration using convolution on lattice grids, and has been further extended (Niu et al., 2017; Lee et al., 2018; Chaplot et al., 2021; Deac et al., 2021). Other than using convolution network, works on integrating learning and planning into differentiable networks include (Oh et al., 2017; Karkus et al., 2017; Weber et al., 2018; Srinivas et al., 2018; Schrittwieser et al., 2019; Amos and Yarats, 2019; Wang and Ba, 2019; Guez et al., 2019; Hafner et al., 2020; Pong et al., 2018; Clavera et al., 2020). In the theoretical side, Grimm et al. (2020; 2021) propose to understand the differentiable planning algorithms from value equivalence perspective.

# 3 BACKGROUND

Markov decision processes. We model the path planning problems as Markov decision processes (MDP) (Sutton and Barto, 2018). An MDP is a 5-tuple  $\mathcal{M} = \langle \mathcal{S},\mathcal{A},P,R,\gamma \rangle$  with state space  $S$ , action space  $\mathcal{A}$ , transition probability function  $P:S\times \mathcal{A}\times \mathcal{S}\to \mathbb{R}_{+}$ , reward function  $R:S\times \mathcal{A}\rightarrow \mathbb{R}$ , and discount factor  $\gamma \in [0,1]$ . Value functions  $V:S\rightarrow \mathbb{R}$  and  $Q:S\times \mathcal{A}\rightarrow \mathbb{R}$  represent expected future returns. The core component behind dynamic programming (DP) based algorithms in reinforcement learning is Bellman (optimality) equation (Sutton and Barto, 2018):  $V(s) = \max_{a}R(s,a) + \gamma \sum_{s^{\prime}}P(s^{\prime}|s,a)V(s^{\prime})$ . Value

iteration is an instance of a dynamic programming (DP) method to solve MDPs, which iteratively applies the Bellman (optimality) operator until convergence.

![](images/88aac76a6f1079f8d4dd0cc4dfe68b2d004347cf86bb2f61f39f3003a767ad6f.jpg)  
Figure 2: (Left) Construction of spatial MDPs from path planning problems, enabling  $G$ -invariant transition. (Right) A demonstration of how an action (arrow in red circle) is rotated when a map is rotated.

![](images/8eef712faa00431bd93fc076a489b4a0d0821f09c8b1dbd19866e8e03c60742a.jpg)

**Path planning.** The objective of the path planning problem is to find optimal actions for every location that navigates to the target in shortest time. However, the original path planning problem is not equivariant under translation due to obstacles, while VINs (Tamar et al., 2016a) implicitly convert it to an equivalent problem, which has equivariant transition function, thus CNNs can be used to inject translation equivariance. We visualize the construction of an equivalent "spatial MDP" in Figure 2 (Left), where the key idea is to encode obstacle information in the transition function from map (top left) into the reward function in the constructed spatial MDP (bottom right) as "trap" with  $-\infty$  reward. Further details about construction are in Section E.1 and E.3. In Figure 2 (Right), we provide a visualization of the representation  $\pi(r)$  of a rotation  $r$  of  $\odot 90^\circ$ , and how an action (arrow) is rotated  $\odot 90^\circ$  accordingly.

Value Iteration Network. Tamar et al. (2016a) proposed Value Iteration Networks (VINs) that use a convolution network to parameterize value iteration. It jointly learns in a latent MDP on 2D grid, which has the latent reward function  $\bar{R}:\mathbb{Z}^2\to \mathbb{R}^{|\mathcal{A}|}$  and value function  $\bar{V}:\mathbb{Z}^2\to \mathbb{R}$ , and applies value iteration on that MDP:

$$
\bar {Q} _ {\bar {a}, i ^ {\prime}, j ^ {\prime}} ^ {(k)} = \bar {R} _ {\bar {a}, i, j} + \sum_ {i, j} W _ {\bar {a}, i, j} ^ {V} \bar {V} _ {i ^ {\prime} - i, j ^ {\prime} - j} ^ {(k - 1)}, \quad \bar {V} _ {i, j} ^ {(k)} = \max  _ {\bar {a}} \bar {Q} _ {\bar {a}, i ^ {\prime}, j ^ {\prime}} ^ {(k)}. \tag {1}
$$

The first equation can be written as:  $\bar{Q}^{(k)} = \bar{R}^a +\mathrm{Conv2D}(\bar{V}^{(k - 1)};W_{\bar{a}}^V)$ , where the 2D convolution layer Conv2D has parameter  $W^{V}$ .

Our final goal is to use VIN to demonstrate a principled method for incorporating symmetry in differentiable planning. We intentionally omit equivariant network details and rather focus on the core idea of integrating symmetry with equivariant networks. We present the necessary group theory background in Section C and full framework and theory in Section D and E.

![](images/b2d02499ecc8c22a172c6ba4f0fa773c5972aa90f8070d98bf02a8529b6a52e0.jpg)  
Figure 3: The commutative diagram of Symmetric Value Iteration Network (SymVIN). Every row is a full computation graph of VIN. Every column is to rotate field by  $\odot 90^{\circ}$

# 4 METHOD: INTEGRATING SYMMETRY INTO PLANNING BY CONVOLUTION

In this work, we aim to exploit the inherent symmetry in a broadly existed problem: path planning. As visualized in Figure 1, the equivariance property unveils the inherent symmetry of the path planning problem on the 2D grid that we could exploit. We provide a rigorous algorithmic framework that can provably make use of symmetry in an efficient manner. To keep approachable, we first introduce how to use VIN as the foundation to build our algorithm: Symmetric VIN. In the next section, we provide the explanation on why we make this choice and introduce further theoretical guarantees on how to exploit symmetry.

How to inject symmetry? VIN uses a regular 2D convolutional network (Equation 1), which has translation equivariance (Cohen and Welling, 2016b; Kondor and Trivedi, 2018). More concretely, a VIN will output the same value function for the same map patches that up to 2D translation. We omit how to characterize translation equivariance here, since it requires a different mechanism to handle and does not decrease the search space nor reduce a path planning MDP to an easier problem.

Beyond translation, we are more interested in rotation and reflection symmetries. Intuitively, as in Figure 1, if we find the optimal solution to a map, it automatically generalizes the solution to all 8 transformed maps (4 rotations times 2 reflections, including identity transformation). This can be characterized by equivariance of a planning algorithm Plan, such as value iteration VI:  $g.\mathrm{Plan}(M) = \mathrm{Plan}(g.M)$ , where  $M$  is a maze map, and  $g$  is the symmetry group  $D_4$  under which 2D grid is invariant.

More importantly, symmetry also helps training of differentiable planning. Intuitively, symmetry in path planning poses additional constraints to its search space: if the goal is in the north, go up; if in the east, go right. In other words, the knowledge can be shared between symmetric cases, or the path planning is effectively reduced by symmetry to a smaller one. This property can also be depicted by equivariance of Bellman operators  $\mathcal{T}$ , or a step of value iteration:  $g.\mathcal{T}[V_0] = \mathcal{T}[g.V_0]$ . If we use  $\mathrm{VI}(M)$  to denote applying Bellman operators on arbitrary initialization until convergence  $\mathcal{T}^{\infty}[V_0]$ , value iteration is also equivariant:

$$
g. \vee I (M) \equiv g. \mathcal {T} ^ {\infty} [ V _ {0} ] = \mathcal {T} ^ {\infty} [ g. V _ {0} ] \equiv \vee I (g. M). \tag {2}
$$

We formally prove the equivariance in Theorem 5.1 in next section. In Theorem 5.2, we theoretical show that value iteration in path planning is a specific type of convolution: steerable convolution (Cohen and Welling, 2016a). Before that, we take the conclusion and first present the pipeline on how to use Steerable CNNs (Cohen and Welling, 2016a) to integrate symmetry.

Pipeline: SymVIN. We have shown that VI is equivariant given symmetry in path planning. We introduce our method Symmetric Value Iteration Network (SymVIN), that realizes equivariant VI by integrating equivariance into VIN w.r.t. rotation and reflection, in addition to translation. We use an instance of Steerable CNN:  $E(2)$ -Steerable CNNs (Weiler and Cesa, 2021) and their package e2cnn for implementation, which is equivariant under  $D_4$  rotation and reflection, and also  $\mathbb{Z}^2$  translation on the 2D grid  $\mathbb{Z}^2$ . In practice, to inject symmetry into VIN, we mainly need to replace the translation-equivariant Conv2D in Eq. 1 with SteerableConv:

$$
\bar {Q} _ {\bar {a}} ^ {(k)} = \bar {R} _ {\bar {a}} + \text {S t e e r a b l e C o n v} (\bar {V}; W ^ {V}), \quad \bar {V} ^ {(k)} = \max  _ {\bar {a}} \bar {Q} _ {\bar {a}} ^ {(k)}. \tag {3}
$$

We visualize the full pipeline in Figure 3. The map and goal are represented as signal  $M: \mathbb{Z}^2 \to \{0,1\}^2$ . It will be processed by another layer and output to the core value iteration loop. After some iterations, the final output will be used to predict the actions and compute cross-entropy loss.

It highlights the injected equivariance property: if we rotate the map (from  $M$  to  $g.M$ ), to guarantee the final policy function to also be equivalently rotated (from  $A$  to  $g.A$ ), we shall guarantee every transformation (e.g.,  $Q_{k} \mapsto V_{k}$  and  $V_{k} \mapsto Q_{k + 1}$ ) in value iteration to also be equivariant, for every pair of columns. We formally justify our design in the section below and provide more technical details in Section E.

Extension: Symmetric GPPN. Based on same spirit, we also implement a symmetric version of Gated path planning network (GPPN (Lee et al., 2018)). It proposes to use LSTM to alleviate the issue of unstable gradient in VINs. Although it does not strictly follow value iteration, it still follows the spirit of steerable planning. Thus, we first obtained a fully convolutional variant of GPPN from [Redacted for anonymous review], called ConvGPPN. It replaces the MLPs in the original LSTM cell with convolutional layers, and then replaces convolutions with equivariant steerable convolutions, resulting in a fully equivariant SymGPPN. See Appendix G.1 for details.

Why do we choose VIN-based planners? There are two reasons behind the choice.

1. The expected value operator in value iteration  $\sum_{s'} P(s'|s, a)V(s')$  is (1) linear in value function and (2) equivariant (shown in Theorem 5.1). Cohen et al. (2020) show that any linear equivariant operator (on homogeneous spaces 2D grid) is a (group) convolution operator.  
2. Value iteration, or Bellman (optimality) operator, consists of only maps between fields/signals over  $\mathbb{Z}$  (e.g., value map and transition function map). This enables to inject symmetry by enforcing equivariance to those maps. Take Figure 1 as example, the 4 corner states are symmetric under transformations in  $D_4$ . Equivalence enforces those 4 states to have the same value if we rotate or flip the map. This avoids the need to find if a new state is symmetric to any existing state, which is shown to be NP-hard (Narayanamurthy and Ravindran, 2008).

In summary, VIN satisfies both desiderata: (1) it uses convolution as the backbone, and (2) it operates on fields. Furthermore, we find VIN is empirically and conceptually the simplest differentiable planning algorithm that satisfies them, which leads to our decision.

# 5 THEORY: VALUE ITERATION IS STEERABLE CONVOLUTION

In the last section, we show how to exploit symmetry in path planning by equivariance from convolution via intuition. The goal of this section is to (1) connect the theoretical justification with the algorithmic design, and (2) provide intuition for the justification. Even through we focus on a specific task, we hope that the underlying guidelines on integrating symmetry into planning are useful for broader planning algorithms and problems as well. The complete version is in Section E.

Overview. There are numerous types of symmetry in various planning tasks. We study symmetry in path planning as an example, because it is a straightforward planning problem, and its solutions have been intensively studied in robotics and artificial intelligence (LaValle, 2006; Sutton and Barto, 2018). However, even for this problem, the symmetry has not been effectively exploited in its planning algorithms, such as Dijkstra's algorithm,  $\mathrm{A}^*$ , or RRT, because of NP-hard orbit finding (Narayanamurthy and Ravindran, 2008). Additionally, we focus on value iteration because it is both widely used and connects closely with convolution (Cohen and Welling, 2016a).

Symmetry from tasks. If we want to exploit inherent symmetry in a task to improve planning, there are two major steps: (1) characterize the symmetry in the task, and (2) incorporate corresponding symmetry into the planning algorithm. The theoretical results in Section E.2 mainly characterize the symmetry and direct us to a feasible planning algorithm.

The symmetry in tasks or MDPs can be specified by the equivariance property of the transition and reward function, studied in Ravindran and Barto (2004); van der Pol et al. (2020b):

$$
\bar {P} \left(s ^ {\prime} \mid s, a\right) = \bar {P} \left(g. s ^ {\prime} \mid g. s, g. a\right), \quad \forall g \in G, \forall s, a, s ^ {\prime} \tag {4}
$$

$$
\bar {R} _ {M} (s, a) = \bar {R} _ {g. M} (g. s, g. a), \quad \forall g \in G, \forall s, a \tag {5}
$$

Note that how the group  $G$  acts on states and actions is called group representation, and is decided by the space  $S$  or  $\mathcal{A}$ , which has been discussed in Equation 19 in Section E.2. We emphasize that the equivariance property of the reward function is different from prior work (Ravindran and Barto, 2004; van der Pol et al., 2020b): in our case, the reward function encodes obstacles as well, and thus depends on map input  $M$ . Intuitively, using Figure 1 as an example, if a position  $s$  is rotated  $g.s$ , to find the correct original reward  $R$  before rotation, the input map  $M$  must also be rotated  $g.M$ . More details in Section E.

Symmetry into planning. As for exploiting the symmetry in planning algorithms, we focus on value iteration and the VIN algorithm. We first prove in Theorem 5.1 that value iteration for path planning respects the equivariance property, motivating us to incorporate symmetry with equivariance.

Theorem 5.1 (informal). If transition is  $G$ -invariant, expected value operator  $\sum_{s'} P(s'|s, a)V(s')$  and value iteration are equivariant under translation, rotation, reflection on the 2D grid.

We visualize the equivariance of the central value update step  $R + \gamma P \star V_k$  in Figure 4. The upper row is a value field  $V_k$  and its rotated version  $g.V_k$  and the lower row is for  $Q$ -value fields  $Q_k$  and  $g.Q_k$  (each). The diagram shows that, if we input a rotated value  $g.V_k$ , the output  $R + \gamma P \star g.V_k$  is guaranteed to be equal to rotated  $Q$ -field  $g.Q_k$ . Additionally, rotating  $Q$ -field  $g.Q_k$  has two components: (1) spatially rotating each grid (a feature channel for an action  $Q(\cdot, a)$ ) and (2) cyclically permuting the channels (black arrows). The red dashed line points how a specific grid of a  $Q$ -value grid  $Q_k(\cdot, \text{South})$  got rotated and permuted. We discuss the theoretical guarantees in Theorem 5.1 and provide full proofs in the appendix.

However, this theorem provides intuition but is inadequate since we do not know: how to implement it like CNNs with multiple feature channels as in VINs, since the first theorem only shows for scalar-valued transition probability and value function. The

![](images/7961e1252280d3a3034dd02704655164c9111f7e8067b00d5ae23bb7ca9db784.jpg)  
Figure 4: Commutative diagram of a single step of value update, showing equivariance under rotations. Each grid in  $Q$ -value field correspond to all values of a location  $Q(\cdot, a)$ .

next result in Theorem 5.2 further proves that value iteration is a general form of convolution (steerable convolution), motivating the use of steerable CNNs by Cohen and Welling (2016a) to replace regular CNNs in VIN. Cohen et al. (2020) prove that steerable convolution is the most general linear equivariant map under some conditions, which value iteration satisfies.

Theorem 5.2 (informal). If transition is  $G$ -invariant, the expected value operator is expressible as a steerable convolution  $\star$ , which is equivariant under translation, rotation, and reflection on  $2D$  grid. The value iteration (with  $\max, +, \times$ ) then forms a deep steerable CNN (Cohen and Welling, 2016a).

We provide a complete version of the framework in Section E and the proofs in Section F. This justifies why we should use Steerable CNN (Cohen and Welling, 2016a) in implementation, since the VI itself is composed of steerable convolution and additional operations  $(\max, +, \times)$ .

Summary. We study how to inject symmetry into VIN for (2D) path planning, and expect the task-specific technical details are useful for two types of readers. (i) Using VIN. If one uses VIN for differentiable planning, the resulting algorithms SymVIN or SymGPPN can be a plug-in alternative, as a part in a larger end-to-end system. Our framework generalizes the idea behind VINs and enables us to understand its applicability and restrictions. (ii) Studying path planning. The proposed framework characterizes the symmetry in path planning, so it is possible to apply the underlying ideas to other domains. For example, it is possible to extend to even higher-dimensional continuous Euclidean spaces or spatial graphs (Weiler et al., 2018; Brandstetter et al., 2021). Additionally, we emphasize that the symmetry in spatial MDPs is different from symmetric MDPs (Zinkevich and Balch, 2001; Ravindran and Barto, 2004; van der Pol et al., 2020a), since our reward function is not  $G$ -invariant (if not conditioning on obstacles). We further discuss this in Section B.2 and E.4.

![](images/825c8997c72c37b6841802e8f461add066749b9800c902474e5ff224cb0cb6d6.jpg)  
Figure 5: (1) Visual navigation. The environment provides a set of egocentric panoramic images for each location, where a set of panoramic images in four directions is visualized. Then, a mapper layer takes them as input and predict a map, visualized in subfigure (2). The predicted map is provided to a mapper to perform path planning. (3) Workspace manipulation. The top-down view is the workspace of a 2-DOF manipulation task. It is mapped by a mapper layer to configuration space, shown in subfigure (4), and provided to planners as well.

![](images/5b04b6c1b7208936d24c085f5a2ebe58149bdb9e8b685b383c32b385da7ce8d2.jpg)

![](images/ff459f97f52a366f0e129501dae0fde8c70e4f99a8c736544ae643a4cf89282e.jpg)

![](images/76106d34202953f545e130ba65bd3564d363cd098ae17843532774dd3e0c25b1.jpg)

# 6 EXPERIMENTS

We experiment VIN, GPPN and our SymPlan methods on four path planning tasks, including using given or learned maps. The additional experiments and ablation studies are in Appendix H.

**Environments and datasets.** We demonstrate the idea in four path planning tasks: (1) 2D navigation, (2) visual navigation, (3) 2 degrees of freedom (2DOFs) configuration space manipulation, and (4) 2DoFs workspace manipulation. We focus on the 2D regular grid setting for path planning, as adopted in prior work (Tamar et al., 2016a; Lee et al., 2018; Chaplot et al., 2021). For each task, we consider using either given (2D navigation and 2-DOF configuration-space manipulation) or learned maps (visual navigation and 2-DOF workspace manipulation). In the latter case, the planner needs to jointly learn a mapper that converts egocentric panoramic images (visual navigation) or workspace states (workspace manipulation) into a map that the planners can operate on, as in (Lee et al., 2018; Chaplot et al., 2021). In both cases, we randomly generate training, validation and test data of  $10K/2K/2K$  maps for all map sizes, to demonstrate data efficiency and generalization ability of symmetric planning. Note that the test maps are unlikely to be symmetric to the training maps by any transformation from the symmetry groups  $G$ . For all environments, the planning domain is the 2D regular grid  $\mathcal{S} = \Omega = \mathbb{Z}^2$ , and the action space is to move in  $4 \odot$  directions<sup>1</sup>:  $\mathcal{A} =$  (north, west, south, east).

Methods: planner networks. We compare five planner methods, where two are our SymPlan methods. Our two equivariant methods is based on Value Iteration Networks (VIN, (Tamar et al., 2016a)) and Gated Path Planning Networks (GPPN, (Lee et al., 2018)). Our equivariant version of VIN is named SymVIN. For GPPN, we first obtained a fully convolutional version, named ConvGPPN [Redacted for anonymous review], and furthermore SymGPPN with steerable CNNs. All methods use (equivariant) convolutions with circular padding in planning in configuration spaces for the manipulation tasks, except GPPN that is not fully convolutional. Chaplot et al. (2021) propose SPT based on Transformers, while integrating symmetry to Transformers is beyond steerable convolutions, thus we do not consider it but still adopt some useful setup.

Training and evaluation. We report success rate and training curves over 3 seeds. The training process (on given maps) follows (Tamar et al., 2016a; Lee et al., 2018), where we train 30 epochs with batch size 32, and use kernel size  $F = 3$  by default. The gradient clip threshold is set to 5. The default batch size is 32, while we need to reduce for some GPPN variants, since LSTM consumes much more memory.

# 6.1 PLANNING ON GIVEN MAPS

Environmental setup. In the 2D navigation task, the map and goal are randomly generated, where the map size is  $\{15,28,50\}$ . In 2-DOF manipulation in configuration space, we adopt the setting in (Chaplot et al., 2021) and train networks to take as input of configuration space, represented by

Table 1: Averaged test success rate (%) for using  ${10}\mathrm{\;K}/2\mathrm{\;K}/2\mathrm{\;K}$  dataset for all four types of tasks.  

<table><tr><td rowspan="2">Method (10K Data)</td><td colspan="4">Navigation</td><td colspan="3">Manipulation</td></tr><tr><td>15 × 15</td><td>28 × 28</td><td>50 × 50</td><td>Visual</td><td>18 × 18</td><td>36 × 36</td><td>Workspace</td></tr><tr><td>VIN</td><td>66.97</td><td>67.57</td><td>57.92</td><td>50.83</td><td>77.82</td><td>84.32</td><td>80.44</td></tr><tr><td>SymVIN</td><td>98.99</td><td>98.14</td><td>86.20</td><td>95.50</td><td>99.98</td><td>99.36</td><td>91.10</td></tr><tr><td>GPPN</td><td>96.36</td><td>95.77</td><td>91.84</td><td>93.13</td><td>2.62</td><td>1.68</td><td>3.67</td></tr><tr><td>ConvGPPN</td><td>99.75</td><td>99.09</td><td>97.21</td><td>98.55</td><td>99.98</td><td>99.95</td><td>89.88</td></tr><tr><td>SymGPPN</td><td>99.98</td><td>99.86</td><td>99.49</td><td>99.78</td><td>100.00</td><td>99.99</td><td>90.50</td></tr></table>

two joints. We randomly generate 0 to 5 obstacles in the manipulator workspace. Then the 2 degree-of-freedom (DOF) configuration space is constructed from workspace and discretized into 2D grid with sizes  $\{18,36\}$ , corresponding to bins of  $20^{\circ}$  and  $10^{\circ}$ , respectively. All methods are trained using the same network size, where for equivariant versions, we use regular representations for all layers, which has size  $|D_4| = 8$ . We keep the same parameters for all methods, so all equivariant convolution layers with regular representations will have higher embedding sizes. Due to memory constraint, we use  $K = 30$  iterations for 2D maze navigation, and  $K = 27$  for manipulation. We use kernel sizes  $F = \{3,5,5\}$  for  $m = \{15,28,50\}$  navigation, and  $F = \{3,5\}$  for  $m = \{18,36\}$  manipulation.

Results. We show the averaged test results for both 2D navigation and C-space manipulation tasks on generalizing to unseen maps (Table 1) and the training curves for 2D navigation (Figure 6).

For VIN series, our SymVIN is much better than the vanilla VIN in terms of generalization and training performance in both environments, which learns much faster and achieves almost perfect asymptotic performance. As for GPPN, we found the fully convolutional variant ConvGPPN actually works better than the original one in (Lee et al., 2018), especially in learning speed. However, SymVIN does fluctuate in some runs,

which seems to come from initialization and label, further studied in Appendix. SymGPPN further boosts ConvGPPN and outperforms all other methods. One exception is GPPN learns poorly in C-space manipulation. For GPPN, the added circular padding in the convolution encoder leads to gradient vanishing problem.

![](images/4b60d0db44eff500d4b9123119be608f48c5024785663384efa45d8958723188.jpg)  
Figure 6: Training curves on 2D navigation with  $10\mathrm{K}$  of  $15\times 15$  maps. Faded areas indicate standard error.

Additionally, we found using regular representations (for  $D_4$  or  $C_4$ ) for state value  $V: \mathbb{Z}^2 \to \mathbb{R}^{C_V}$  (and for  $Q$ -value) works better than trivial representations. This is counterintuitive since we expect the  $V$  value to be scalar  $\mathbb{Z}^2 \to \mathbb{R}$ . One reason is that switching between regular (for  $Q$ ) and trivial (for  $V$ ) representation introduces unnecessary bottleneck. Depending on the choice of representations, we implement different max-pooling, with details in Appendix G.2. We also empirically found using FC only in the final layer  $Q_K \mapsto A$  helps stabilize the training. The ablation study on this and more are in Appendix H.

Remark. Two symmetric planners are both significantly better than their counterparts. Notably, we did not include any symmetric maps to the test data that symmetric planners would perform much better. There are several potential sources of advantages: (1) SymPlan allows parameter sharing across positions and maps and implicitly enables planning in a reduced space: every  $(s,a,s')$  seamlessly generalizes to  $(g.s,g.a,g.s')$  for any  $g\in G$ , (2) thus it uses training data more efficiently, (3) it reduces the space of hypothesis class and facilitate generalization to unseen maps.

# 6.2 PLANNING ON LEARNED MAPS: SIMULTANEOUSLY PLANNING AND MAPPING

Environmental setup. For visual navigation, we randomly generate maps using the same strategy as before, and then render four egocentric panoramic views for each location from produced 3D environments with Gym-MiniWorld (Chevalier-Boisvert, 2018), since it allows to generate 3D mazes with any layout. For  $m \times m$  maps, all egocentric views for a map is represented by  $m \times m \times 4$  RGB images. For workspace manipulation, we randomly generate 0 to 5 obstacles in workspace

as before. We use a mapper network to convert the  $96 \times 96$  workspace (image of obstacles) to the  $m \times m$  2 degree-of-freedom (DOF) configuration space (2D occupancy grid). In both environments, the setup is similar to Section 6.1, while we only use  $m = 15$  maps but longer 100 epochs for visual navigation and  $m = 18$  maps still with 30 epochs for workspace manipulation.

Methods: mapper networks and setup. For visual navigation, we implemented equivariant mapper network based on (Lee et al., 2018). The mapper network converts every image into a 256-dimensional embedding  $m \times m \times 4 \times 256$  and then predicts map layout  $m \times m \times 1$ . For workspace manipulation, we use U-net (Ronneberger et al., 2015) with residual-connection (He et al., 2015) as a mapper. For more training details, see Section H.

Results. The results are also shown in Table 1, denoted as Visual (navigation,  $15 \times 15$ ) and Workspace (manipulation,  $18 \times 18$ ). In visual navigation, the trends are similar to 2D case: two symmetric planners both train much faster. Besides vanilla VIN, all approaches finally converge to near-optimal successful rate (around  $95\%$ ), while the validation and test results show large gaps. SymGPPN has almost no generalization gap, while VIN does not generalize well to new 3D visual navigation environments. Our SymVIN improves test successful rate from less than  $50\%$  to  $90\%$  and is comparable with GPPN. Since the input is raw images and a mapper is used to learn end-to-end, it potentially causes one major source of generalization gap for some approaches. In workspace manipulation, the results are also analogous to C-space, while ours advantages over baselines are smaller. In our inspection, we found the mapper network is the bottleneck, since the mapping for obstacles from workspace to C-space is nontrivial to learn.

# 6.3 RESULTS ON GENERALIZATION TO LARGER MAPS

To demonstrate the generalization advantage of ours methods, all methods are trained in small map and tested in larger maps. All methods are trained on  $15 \times 15$  with  $K = 30$ . Then we test all methods on map size  $15 \times 15$  through  $99 \times 99$ , averaging over 3 seeds (3 model checkpoints) for each method and 1000 maps for each map size. Iterations  $K$  is set to  $\sqrt{2} \cdot M$ , where  $M$  is the testing map size (x-axis). The results are shown in Figure 7.

Results. SymVIN generalizes better than VIN, although the variance is greater. GPPN diverges for larger variable  $K$  since it is even worse than fixed  $K = 30$  in all map sizes. ConvGPPN converges, while it fluctuates for different seeds. SymGPPN shows the best generalization and has small variance. In conclusion, SymVIN and SymGPPN generalize better to different map sizes, compare to all non-equivariant baselines.

Remark. The SymPlan models demonstrate end-to-end planning and learning ability, potentially enabling further applications to other tasks as a differentiable component for planning. The additional results and ablation studies are provided in Appendix H.

![](images/47599600220cc5e674edd1fc903dab4e1e00774ab665c5ed981c8af20f7ebada.jpg)  
Figure 7: Results for testing on larger maps, when trained on size 15 map. Our methods outperform all baselines.

# 7 DISCUSSION

In this work, we study the symmetry in 2D path planning problem, and build a framework using the theory of steerable CNNs to prove that value iteration in path planning is actually a form of steerable CNN (on 2D grids). Although we focus on  $\mathbb{Z}^2$ , we can generalize to path planning on higher-dimensional or even continuous Euclidean spaces (Weiler et al., 2018; Brandstetter et al., 2021), and use equivariant operations on steerable feature fields (such as steerable convolutions, pooling, and point-wise non-linearities) from steerable CNNs. We practically show that the SymPlan algorithms exactly motivated by the theory provide great improvement. We hope the framework along with the design of practical algorithms can provide a new pathway to exploiting symmetry structure in differentiable planning.

# 8 REPRODUCIBILITY STATEMENT

We provide additional details in the appendix. We also plan to open source the codebase. We briefly outline the appendix below.

1. Additional Discussion  
2. Background: Technical background and concepts on steerable CNNs and group CNNs  
3. Method: we provide full details on how to reproduce it  
4. Theory/Framework: we provide the complete version of the theory statements  
5. Proofs: this includes all proofs  
6. Experiment / Environment / Implementation details: useful details for reproducibility  
7. Additional results

# REFERENCES

Richard S. Sutton and Andrew G. Barto. Reinforcement learning: an introduction. Adaptive computation and machine learning series. The MIT Press, Cambridge, Massachusetts, second edition, 2018. ISBN 978-0-262-03924-6.  
Lihong Li, Thomas J. Walsh, and M. Littman. Towards a Unified Theory of State Abstraction for MDPs. In AI&M, 2006.  
Balaraman Ravindran and Andrew G Barto. An algebraic approach to abstraction in reinforcement learning. PhD thesis, University of Massachusetts at Amherst, 2004.  
Maria Fox and Derek Long. Extending the exploitation of symmetries in planning. In *In Proceedings of AIPS'02*, pages 83–91, 2002.  
Maria Fox and Derek Long. The Detection and Exploitation of Symmetry in Planning Problems. In *In IJCAI*, pages 956–961. Morgan Kaufmann, 1999.  
Nir Pochter, Aviv Zohar, and Jeffrey S. Rosenschein. Exploiting Problem Symmetries in State-Based Planners. In Twenty-Fifth AAAI Conference on Artificial Intelligence, August 2011. URL https://www.aaaai.org/ocs/index.php/AAAI/AAAI11/paper/view/3732.  
Martin Zinkevich and Tucker Balch. Symmetry in Markov decision processes and its implications for single agent and multi agent learning. In *In Proceedings of the 18th International Conference on Machine Learning*, pages 632-640. Morgan Kaufmann, 2001.  
Shravan Matthur Narayanamurthy and Balaraman Ravindran. On the hardness of finding symmetries in Markov decision processes. In Proceedings of the 25th international conference on Machine learning - ICML '08, pages 688-695, Helsinki, Finland, 2008. ACM Press. ISBN 978-1-60558-205-4. doi: 10/bkswc2. URL http://portal.acm.org/citation.cfm?doid=1390156.1390243.  
Elise van der Pol, Daniel E. Worrall, Herke van Hoof, Frans A. Oliehoek, and Max Welling. MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning. arXiv:2006.16908 [cs, stat], June 2020a. URL http://arxiv.org/abs/2006.16908. arXiv:2006.16908.  
Dian Wang, Robin Walters, and Robert Platt.  $\S$ mathrm{SO}  $(2)$ $-$  -Equivariant Reinforcement Learning. September 2021. URL https://openreview.net/forum?id=7F9c0hvdfk_.  
Michael M Bronstein, Joan Bruna, Taco Cohen, and Petar Velicković. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021a.  
Taco Cohen, Mario Geiger, and Maurice Weiler. A General Theory of Equivariant CNNs on Homogeneous Spaces. arXiv:1811.02017 [cs, stat], January 2020. URL http://arxiv.org/abs/1811.02017. arXiv:1811.02017.  
Risi Kondor and Shubhendu Trivedi. On the Generalization of Equivalence and Convolution in Neural Networks to the Action of Compact Groups. arXiv:1802.03690 [cs, stat], November 2018. URL http://arxiv.org/abs/1802.03690. arXiv: 1802.03690.  
Taco S. Cohen and Max Welling. Steerable CNNs. November 2016a. URL https://openreview.net/forum?id=rJQKYt511.  
Taco S. Cohen and Max Welling. Group Equivariant Convolutional Networks. arXiv:1602.07576 [cs, stat], June 2016b. URL http://arxiv.org/abs/1602.07576. arXiv: 1602.07576.  
Maurice Weiler and Gabriele Cesa. General  $\$ 123,456$ -Equivariant Steerable CNNs. arXiv:1911.08251 [cs, eess], April 2021. URL http://arxiv.org/abs/1911.08251.arXiv:1911.08251.  
Aviv Tamar, YI WU, Garrett Thomas, Sergey Levine, and Pieter Abbeel. Value Iteration Networks. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016a. URL https://proceedings.neurips.cc/paper/2016/bit/ c21002f464c5fc5bee3b98ced83963b8-Abstract.html.

Lisa Lee, Emilio Parisotto, Devendra Singh Chaplot, Eric Xing, and Ruslan Salakhutdinov. Gated Path Planning Networks. arXiv:1806.06408 [cs, stat], June 2018. URL http://arxiv.org/abs/1806.06408. arXiv:1806.06408.  
Carmel Domshlak, Michael Katz, and Alexander Shleyfman. Enhanced Symmetry Breaking in Cost-Optimal Planning as Forward Search. page 5.  
Alexander Shleyfman. Symmetry Breaking: Satisficing Planning and Landmark Heuristics. page 5.  
Alexander Shleyfman, Michael Katz, Malte Helmert, Silvan Sievers, and Martin Wehrle. Heuristics and Symmetries in Classical Planning. Proceedings of the AAAI Conference on Artificial Intelligence, 29(1), March 2015. ISSN 2374-3468. URL https://ojs.aaaai.org/index.php/AAAI/article/view/9649. Number: 1.  
Silvan Sievers, Martin Wehrle, Malte Helmert, Alexander Shleyfman, and Michael Katz. Factored Symmetries for Merge-and-Shrink Abstractions. page 8.  
Martin Wehrle, Malte Helmert, Alexander Shleyfman, and Michael Katz. Integrating Partial Order Reduction and Symmetry Elimination for Cost-Optimal Classical Planning. page 7.  
Mohammad Abdulaziz, Michael Norrish, and Charles Gretton. Exploiting Symmetries by Planning for a Descriptive Quotient. page 8.  
Silvan Sievers, Martin Wehrle, Malte Helmert, and Michael Katz. An Empirical Case Study on Symmetry Handling in Cost-Optimal Planning as Heuristic Search. In Steffen Holldobler, Rafael Peñaloza, and Sebastian Rudolph, editors, *KI 2015: Advances in Artificial Intelligence*, volume 9324, pages 166–180. Springer International Publishing, Cham, 2015. ISBN 978-3-319-24488-4 978-3-319-24489-1. doi: 10.1007/978-3-319-24489-1_13. URL http://link.springer.com/10.1007/978-3-319-24489-1_13. Series Title: Lecture Notes in Computer Science.  
Silvan Sievers. Structural Symmetries of the Lifted Representation of Classical Planning Tasks. page 8.  
Dominik Winterer, Martin Wehrle, and Michael Katz. Structural Symmetries for Fully Observable Nondeterministic Planning. page 7.  
Gabriele Röger, Silvan Sievers, and Michael Katz. Symmetry-Based Task Reduction for Relaxed Reachability Analysis. In Twenty-Eighth International Conference on Automated Planning and Scheduling, June 2018. URL https://aaai.org/ocs/index.php/ICAPS/ICAPS18/paper/view/17772.  
Silvan Sievers, Gabriele Röger, Martin Wehrle, and Michael Katz. Theoretical Foundations for Structural Symmetries of Lifted PDDL Tasks. Proceedings of the International Conference on Automated Planning and Scheduling, 29:446-454, 2019. ISSN 2334-0843. URL https:// ojs.aaaai.org/index.php/ICAPS/article/view/3509.  
Daniel Fiser, Álvaro Torralba, and Alexander Shleyfman. Operator Mutexes and Symmetries for Simplifying Planning Tasks. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):7586-7593, July 2019. ISSN 2374-3468. doi: 10.1609/aaai.v33i01.33017586. URL https://ojs.aaai.org/index.php/AAAI/article/view/4751. Number: 01.  
N. Ferns, P. Panangaden, and Doina Precup. Metrics for Finite Markov Decision Processes. In AAAI, 2004.  
Elise van der Pol, Daniel Worrall, Herke van Hoof, Frans Oliehoek, and Max Welling. Mdp homomorphic networks: Group symmetries in reinforcement learning. Advances in Neural Information Processing Systems, 33, 2020b.  
Jung Yeon Park, Ondrej Biza, Linfeng Zhao, Jan Willem van de Meent, and Robin Walters. Learning Symmetric Embeddings for Equivariant World Models. arXiv:2204.11371 [cs], April 2022. URL http://arxiv.org/abs/2204.11371.arXiv:2204.11371.

Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Velicković. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv:2104.13478 [cs, stat], April 2021b. URL http://arxiv.org/abs/2104.13478.arXiv:2104.13478.  
Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, and Pieter Abbeel. Value iteration networks. arXiv preprint arXiv:1602.02867, 2016b.  
Sufeng Niu, Siheng Chen, Hanyu Guo, Colin Targonski, Melissa C. Smith, and Jelena Kovacevic. Generalized Value Iteration Networks: Life Beyond Lattices. arXiv:1706.02416 [cs], October 2017. URL http://arxiv.org/abs/1706.02416. arXiv:1706.02416.  
Devendra Singh Chaplot, Deepak Pathak, and Jitendra Malik. Differentiable Spatial Planning using Transformers. arXiv:2112.01010 [cs], December 2021. URL http://arxiv.org/abs/2112.01010.arXiv:2112.01010.  
Andreea Deac, Petar Velicković, Ognjen Milinković, Pierre-Luc Bacon, Jian Tang, and Mladen Nikolic. Neural Algorithmic Reasoners are Implicit Planners. October 2021. URL https://arxiv.org/abs/2110.05442v1.  
Junhyuk Oh, Satinder Singh, and Honglak Lee. Value Prediction Network. arXiv:1707.03497 [cs], November 2017. URL http://arxiv.org/abs/1707.03497. arXiv:1707.03497.  
Peter Karkus, David Hsu, and Wee Sun Lee. QMDP-Net: Deep Learning for Planning under Partial Observability. arXiv:1703.06692 [cs, stat], November 2017. URL http://arxiv.org/abs/1703.06692.arXiv:1703.06692.  
Théophane Weber, Sébastien Racanière, David P. Reichert, Lars Buesing, Arthur Guez, Danilo Jimenez Rezende, Adria Puigdomènech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, Razvan Pascanu, Peter Battaglia, Demis Hassabis, David Silver, and Daan Wierstra. Imagination-Augmented Agents for Deep Reinforcement Learning. arXiv:1707.06203 [cs, stat], February 2018. URL http://arxiv.org/abs/1707.06203. arXiv:1707.06203.  
Aravind Srinivas, Allan Jabri, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Universal Planning Networks. arXiv:1804.00645 [cs, stat], April 2018. URL http://arxiv.org/abs/1804.00645.arXiv:1804.00645.  
Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. arXiv:1911.08265 [cs, stat], November 2019. URL http://arxiv.org/abs/1911.08265.arXiv:1911.08265.  
Brandon Amos and Denis Yarats. The Differentiable Cross-Entropy Method. September 2019. doi: 10.48550/arXiv.1909.12830. URL https://arxiv.org/abs/1909.12830v4.  
Tingwu Wang and Jimmy Ba. Exploring Model-based Planning with Policy Networks. June 2019. URL https://arxiv.org/abs/1906.08649v1.  
Arthur Guez, Mehdi Mirza, Karol Gregor, Rishabh Kabra, Sébastien Racanière, Théophane Weber, David Raposo, Adam Santoro, Laurent Orseau, Tom Eccles, Greg Wayne, David Silver, and Timothy Lillicrap. An investigation of model-free planning. arXiv:1901.03559 [cs, stat], May 2019. URL http://arxiv.org/abs/1901.03559. arXiv:1901.03559.  
Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. arXiv:1912.01603 [cs], March 2020. URL http://arxiv.org/abs/1912.01603.arXiv:1912.01603.  
Vitchyr Pong, Shixiang Gu, Murtaza Dalal, and Sergey Levine. Temporal Difference Models: Model-Free Deep RL for Model-Based Control. arXiv:1802.09081 [cs], February 2018. URL http://arxiv.org/abs/1802.09081.arXiv:1802.09081.  
Ignasi Clavera, Violet Fu, and Pieter Abbeel. Model-Augmented Actor-Critic: Backpropagating through Paths. arXiv:2005.08068 [cs, stat], May 2020. URL http://arxiv.org/abs/2005.08068.arXiv:2005.08068.

Christopher Grimm, André Barreto, Satinder Singh, and David Silver. The Value Equivalence Principle for Model-Based Reinforcement Learning. arXiv:2011.03506 [cs], November 2020. URL http://arxiv.org/abs/2011.03506.arXiv:2011.03506.  
Christopher Grimm, André Barreto, Gregory Farquhar, David Silver, and Satinder Singh. Proper Value Equivalence. arXiv:2106.10316 [cs], December 2021. URL http://arxiv.org/abs/2106.10316.arXiv:2106.10316.  
Steven M. LaValle. Planning Algorithms. Cambridge University Press, May 2006. ISBN 978-1-139-45517-6.  
Maurice Weiler, M. Geiger, M. Welling, Wouter Boomsma, and Taco Cohen. 3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data. In NeurIPS, 2018.  
Johannes Brandstetter, Rob Hesselink, Elise van der Pol, Erik J. Bekkers, and Max Welling. Geometric and Physical Quantities Improve E(3) Equivariant Message Passing. arXiv:2110.02905 [cs, stat], December 2021. URL http://arxiv.org/abs/2110.02905. arXiv: 2110.02905.  
Maxime Chevalier-Boisvert. Miniworld: Minimalistic 3d environment for rl & robotics research. https://github.com/maximecb/gym-miniworld, 2018.  
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. CoRR, abs/1505.04597, 2015. URL http://arxiv.org/abs/1505.04597.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015. URL http://arxiv.org/abs/1512.03385.  
T. S. Cohen. Equivariant convolutional networks. 2021. URL https://dare.uva.nl/search?identifier=0f7014ae-ee94-430e-a5d8-37d03d8d10e6.
