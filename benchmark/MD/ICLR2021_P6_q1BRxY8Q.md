# LEARNING SAFE MULTI-AGENT CONTROL WITH DECENTRALIZED NEURAL BARRIER CERTIFICATES

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the multi-agent safe control problem where agents should avoid collisions to static obstacles and collisions with each other while reaching their goals. Our core idea is to learn the multi-agent control policy jointly with learning the control barrier functions as safety certificates. We propose a novel joint-learning framework that can be implemented in a decentralized fashion, with generalization guarantees for certain function classes. Such a decentralized framework can adapt to an arbitrarily large number of agents. Building upon this framework, we further improve the scalability by incorporating neural network architectures that are invariant to the quantity and permutation of neighboring agents. In addition, we propose a new spontaneous policy refinement method to further enforce the certificate condition during testing. We provide extensive experiments to demonstrate that our method significantly outperforms other leading multi-agent control approaches in terms of maintaining safety and completing original tasks. Our approach also shows exceptional generalization capability in that the control policy can be trained with 8 agents in one scenario, while being used on other scenarios with up to 1024 agents in complex multi-agent environments and dynamics.

# 1 INTRODUCTION

Machine learning (ML) has created unprecedented opportunities for achieving full autonomy. However, learning-based methods in autonomous systems (AS) can and do fail due to the lack of formal guarantees and limited generalization capability, which poses significant challenges for developing safety-critical AS, especially large-scale multi-agent AS, that are provably dependable.

On the other side, safety certificates (Chang et al. (2019); Jin et al. (2020); Choi et al. (2020)), which widely exist in control theory and formal methods, serve as proofs for the satisfaction of the desired properties of a system, under certain control policies. For example, once found, a Control Barrier Function (CBF) ensures that the closed-loop system always stays inside some safe set (Wieland & Allgower, 2007; Ames et al., 2014) with a CBF Quadratic Programming (QP) supervisory controller. However, it is extremely difficult to synthesize CBF by hand for complex dynamic systems, which stems a growing interest in learning-based CBF (Saveriano & Lee, 2020; Srinivasan et al., 2020; Jin et al., 2020; Boffi et al., 2020; Taylor et al., 2020; Robey et al., 2020). However, all of these studies only concern single-agent systems. How to develop learning-based approaches for safe multi-agent control that are both provably dependable and scalable remains open.

In multi-agent control, there is a constant dilemma: centralized control strategies can hardly scale to a large number of agents, while decentralized control without coordination often misses safety and performance guarantees. In this work, we propose a novel learning framework that jointly designs multi-agent control policies and safety certificate from data, which can be implemented in a decentralized fashion and scalable to an arbitrary number of agents. Specifically, we first introduce the notion of decentralized CBF as safety certificates, then propose the framework of learning decentralized CBF, with generalization error guarantees. The decentralized CBF can be seen as a contract among agents, which allows agents to learn a mutual agreement with each other on how to avoid collisions. Once such a controller is achieved through the joint-learning framework, it can be applied on an arbitrarily number of agents and in scenarios that are different from the training scenarios, which resolves the fundamental scalability issue in multi-agent control. We also propose

several effective techniques in Section 4 to make such a learning process even more scalable and practical, which are then validated extensively in Section 5.

Experimental results are indeed promising. We study both 2D and 3D safe multi-agent control problems, each with several distinct environments and complex nonholonomic dynamics. Our joint-learning framework performs exceptionally well: our control policies trained on scenarios with 8 agents can be used on up to 1024 agents while maintaining low collision rates, which has notably pushed the boundary of learning-based safe multi-agent control. Speaking of which, 1024 is not the limit of our approach but rather due to the limited computational capability of our laptop used for the experiments. We also compare our approach with both leading learning-based methods (Lowe et al., 2017; Zhang & Bastani, 2019; Liu et al., 2020) and traditional planning methods (Ma et al., 2019; Fan et al., 2020). Our approach outperforms all the other approaches in terms of both completing the tasks and maintaining safety.

Contributions. Our main contributions are three-fold: 1) We propose the first framework to jointly learning safe multi-agent control policies and CBF certificates, in a decentralized fashion. 2) We present several techniques that make the learning framework more effective and scalable for practical multi-agent systems, including the use of quantity-permutation invariant neural network architectures in learning to handle the permutation of neighbouring agents. 3) We demonstrate via extensive experiments that our method significantly outperforms other leading methods, and has exceptional generalization capability to unseen scenarios and an arbitrary number of agents, even in quite complex multi-agent environments such as ground robots and drones. The video that demonstrates the outstanding performance of our method can be found in the supplementary material.

Related Work. Learning-Based Safe Control via CBF. Barrier certificates (Prajna et al., 2007) and CBF (Wieland & Allgower, 2007) is a well-known effective tool for guaranteeing the safety of nonlinear dynamic systems. However, the existing methods for constructing CBFs either rely on specific problem structures (Chen et al., 2017b) or do not scale well (Mitchell et al., 2005). Recently, there has been an increasing interest in learning-based and data-driven safe control via CBFs, which primarily consist of two categories: learning CBFs from data (Saveriano & Lee, 2020; Srinivasan et al., 2020; Jin et al., 2020; Boffi et al., 2020), and CBF-based approach for controlling unknown systems (Wang et al., 2017; 2018; Cheng et al., 2019; Taylor et al., 2020). Our work is more pertinent to the former and is complementary to the latter, which usually assumes that the CBF is provided. None of these learning-enabled approaches, however, has addressed the multi-agent setting.

Multi-Agent Safety Certificates and Collision Avoidance. Restricted to holonomic systems, guaranteeing safety in multi-agent systems has been approached by limiting the velocities of the agents (Van den Berg et al., 2008; Alonso-Mora et al., 2013). Later, Borrmann et al. (2015) Wang et al. (2017) have proposed the framework of multi-agent CBF to generate collision-free controllers, with either perfectly known system dynamics (Borrmann et al., 2015), or with worst-case uncertainty bounds (Wang et al., 2017). Recently, Chen et al. (2020) has proposed a decentralized controller synthesized approach under this CBF framework, which is scalable to an arbitrary number of agents. However, in Chen et al. (2020) the CBF controller relies on online integration of the dynamics under the backup strategy, which can be computationally challenging for complex systems. Due to space limit, we omit other non-learning multi-agent control methods but acknowledge their importance.

Safe Multi-Agent (Reinforcement) Learning (MARL). Safety concerns have drawn increasing attention in MARL, especially with the applications to safety-critical multi-agent systems (Zhang & Bastani, 2019; Qie et al., 2019; Shalev-Shwartz et al., 2016). Under the CBF framework, Cheng et al. (2020) considered the setting with unknown system dynamics, and proposed to design robust multi-agent CBFs based on the learned dynamics. This mirrors the second category mentioned above in single-agent learning-based safe control, which is perpendicular to our focus. RL approaches have also been applied for multi-agent collision avoidance (Chen et al., 2017a; Lowe et al., 2017; Everett et al., 2018; Zhang et al., 2018). Nonetheless, no formal guarantees of safety were established in these works. One exception is Zhang & Bastani (2019), which proposed a multi-agent model predictive shielding algorithm that provably guarantees safety for any policy learned from MARL, which differs from our multi-agent CBF-based approach. More importantly, none of these MARL-based approaches scale to a massive number of, e.g., thousands of agents, as our approach does. The most scalable MARL platform, to the best of our knowledge, is Zheng et al. (2017), which may handle a comparable scale of agents as ours, but with discrete state-action spaces. This is in contrast to our continuous-space models that can model practical control systems such as robots and drones.

# 2 PRELIMINARIES

# 2.1 CONTROL BARRIER FUNCTIONS AS SAFETY CERTIFICATES

One common approach for (single-agent) safety certificate is via control barrier functions (Ames et al., 2014), which can enforce the states of dynamic systems to stay in the safe set. Specifically, let  $S \subset \mathbb{R}^n$  be the state space,  $S_d \subset S$  is the dangerous set,  $S_s = S \setminus S_d$  is the safe set, which contains the set of initial conditions  $S_0 \subset S_s$ . Also define the space of control actions as  $\mathcal{U} \subset \mathbb{R}^m$ . For a dynamic system  $\dot{s}(t) = f(s(t), u(t))$ , a control barrier function  $h: \mathbb{R}^n \mapsto \mathbb{R}$  satisfies:

$$
(\forall s \in \mathcal {S} _ {0}, h (s) \geq 0) \bigwedge (\forall s \in \mathcal {S} _ {d}, h (s) <   0) \bigwedge (\forall s \in \{s \mid h (s) \geq 0 \}, \nabla_ {s} h \cdot f (s, u) + \alpha (h) \geq 0), \tag {1}
$$

where  $\alpha (\cdot)$  is a class-  $\mathcal{K}$  function, i.e.,  $\alpha (\cdot)$  is strictly increasing and satisfies  $\alpha (0) = 0$ . For a control policy  $\pi :\mathcal{S}\to \mathcal{U}$  and CBF  $h$  it is proved in Ames et al. (2014) that if  $s(0)\in \{s\mid h(s)\geq 0\}$  and the three conditions in (1) are satisfied with  $u = \pi (x)$ , then  $s(t)\in \{s\mid h(s)\geq 0\}$  for  $\forall t\in [0,\infty)$ , which means the state would never enter the dangerous set  $S_{d}$  under  $\pi$ .

# 2.2 SAFETY OF MULTI-AGENT DYNAMIC SYSTEMS

Consider a multi-agent system with  $N$  agents, the joint state of which at time  $t$  is denoted by  $s(t) = \{s_1(t), s_2(t), \dots, s_N(t)\}$  where  $s_i(t) \in S_i \subset \mathbb{R}^n$  denotes the state of agent  $i$  at time  $t$ . The dynamics of agent  $i$  is  $\dot{s}_i(t) = f_i(s_i(t), u_i(t))$  where  $u_i(t) \in \mathcal{U}_i \subset \mathbb{R}^m$  is the control action of agent  $i$ . The overall state space and input space are denoted as  $S \doteq \bigotimes_{i=1}^{N} S_i$ ,  $\mathcal{U} \doteq \bigotimes_{i=1}^{N} \mathcal{U}_i$ . For each agent  $i$ , we define  $\mathcal{N}_i(t)$  as the set of its neighborhood agents at time  $t$ . Let  $o_i(t) \in \mathbb{R}^{n \times |\mathcal{N}_i(t)|}$  be the local observation of agent  $i$ , which is the states of  $|\mathcal{N}_i(t)|$  neighborhood agents. Notice that the dimension of  $o_i(t)$  is not fixed and depends on the quantity of neighboring agents. We assume that the safety of agent  $i$  is jointly determined by  $s_i$  and  $o_i$ . Let  $\mathcal{O}_i$  be the set of all possible observations and  $\mathcal{X}_i := S_i \times \mathcal{O}_i$  be the state-observation space that contains the safe set  $\mathcal{X}_{i,s}$ , dangerous set  $\mathcal{X}_{i,d}$  and initial conditions  $\mathcal{X}_{i,0} \subset \mathcal{X}_{i,s}$ . Let  $d: \mathcal{X}_i \to \mathbb{R}$  describe the minimum distance from agent  $i$  to other agents that it observes,  $d(s_i, o_i) < \kappa_s$  implies collision. Then  $\mathcal{X}_{i,s} = \{(s_i, o_i) | d(s_i, o_i) \geq \kappa_s\}$  and  $\mathcal{X}_{i,d} = \{(s_i, o_i) | d(s_i, o_i) < \kappa_s\}$ . Let  $\bar{d}_i: S \to \mathbb{R}$  be the lifting of  $d$  from  $\mathcal{X}_i$  to  $\mathcal{S}$ , which is well-defined since there is a surjection from  $\mathcal{S}$  to  $\mathcal{X}_i$ . Then define  $\mathcal{S}_s \doteq \{s \in S | \forall i = 1, ..., N, \bar{d}_i(s) \geq \kappa_s\}$ . The safety of a multi-agent system can be formally defined as follows:

Definition 1 (Safety of Multi-Agent Systems). If the state-observation satisfies  $d(s_i, o_i) \geq \kappa_s$  for agent  $i$  and time  $t$ , then agent  $i$  is safe at time  $t$ . If for  $\forall i$ , agent  $i$  is safe at time  $t$ , then the multi-agent system is safe at time  $t$ , and  $s \in S_s$ .

A main objective of this paper is to learn the control policy  $\pi_i(s_i(t),o_i(t))$  for  $\forall i$  such that the multi-agent system is safe. The control policy is decentralized (i.e., each agent has its own control policy and there does not exist a central controller to coordinate all the agents). In this way, our decentralized approach has the hope to scale to very a large number of agents.

# 3 LEARNING FRAMEWORK FOR MULTI-AGENT DECENTRALIZED CBF

# 3.1 DECENTRALIZED CONTROL BARRIER FUNCTIONS

For a multi-agent dynamic system, the most naive CBF would be a centralized function taking into account the cross production of all agents' states, which leads to an exponential blow-up in the state space and difficulties in modeling systems with an arbitrary number of agents. Instead, we consider a decentralized control barrier function  $h_i: \mathcal{X}_i \mapsto \mathbb{R}$ :

$$
\begin{array}{l} (\forall (s _ {i}, o _ {i}) \in \mathcal {X} _ {i, 0}, h _ {i} (s _ {i}, o _ {i}) \geq 0) \bigwedge (\forall (s _ {i}, o _ {i}) \in \mathcal {X} _ {i, d}, h _ {i} (s _ {i}, o _ {i}) <   0) \bigwedge \\ (\forall (s _ {i}, o _ {i}) \in \{(s _ {i}, o _ {i}) \mid h _ {i} (s _ {i}, o _ {i}) \geq 0 \}, \nabla_ {s _ {i}} h _ {i} \cdot f _ {i} (s _ {i}, u _ {i}) + \nabla_ {o _ {i}} h _ {i} \cdot \dot {o} _ {i} (t) + \alpha (h _ {i}) \geq 0) \tag {2} \\ \end{array}
$$

where  $\dot{o}_i(t)$  is the time derivative of the observation, which depends on the behavior of other agents. Although there is no explicit expression of this term, it can be evaluated and incorporated in the learning process. Note that the CBF  $h_i(s_i,o_i)$  is local in the sense that it only depends on the local state  $s_i$  and observation  $o_i$ . We refer to the three conditions in (2) as decentralized CBF conditions. The following proposition shows that satisfying (2) guarantees the safety of the multi-agent system.

Proposition 1 (Multi-Agent Safety Certificates with Decentralized CBF). If for  $\forall i$ , the initial state-observation  $(s_i(0), o_i(0)) \in \{(s_i, o_i) \mid h_i(s_i, o_i) \geq 0\}$  and the decentralized CBF conditions in (2) are satisfied, then  $\forall i$  and  $\forall t$ ,  $(s_i(t), o_i(t)) \in \{(s_i, o_i) \mid h_i(s_i, o_i) \geq 0\}$ , which implies the state would never enter  $\mathcal{X}_{i,d}$  for any agent  $i$ . Thus, by Definition 1, the multi-agent system is safe.

The proof of Proposition 1 is provided in the supplementary material. The key insight of Proposition 1 is that for the whole multi-agent system, the CBFs can be applied in a decentralized fashion for each agent. An agent only needs to care about its local information, and if all agents respect the same form of contract (i.e., the decentralized CBF conditions), the whole multi-agent system will be safe. This property is of great importance since it reveals that a centralized controller that coordinates all agents is not necessary to achieve safety. A centralized control policy has to deal with the dimension explosion when the number of agents grow, while a decentralized design can significantly improve the scalability to an arbitrarily large number of agents.

# 3.2 LEARNING FRAMEWORK AND GENERALIZATION GUARANTEE

From Proposition 1, we know that if we can jointly learn the control policy  $\pi_i(s_i,o_i)$  and control barrier function  $h_i(s_i,o_i)$  such that the decentralized CBF conditions in (2) are satisfied, then the multi-agent system is guaranteed to be safe. Next we formulate the optimization objective and provide a generalization bound with probabilistic guarantee. Let  $T\subset \mathbb{R}_+$  be the time interval and  $\tau_{i} = \{s_{i}(t),o_{i}(t)\}_{t\in T}$  be a trajectory of state and observation of agent  $i$ . Let  $\mathcal{T}_i$  be the set of all possible trajectories of agent  $i$ . Let  $\mathcal{H}_i$  and  $\mathcal{V}_i$  be the function classes of  $h_i$  and  $\pi_i$ . Define the function  $y_{i}:\mathcal{T}_{i}\times \mathcal{H}_{i}\times \mathcal{V}_{i}\mapsto \mathbb{R}$  as:

$$
y _ {i} \left(\tau_ {i}, h _ {i}, \pi_ {i}\right) := \min  \left\{\inf  _ {\mathcal {X} _ {i, 0} \cap \tau_ {i}} h _ {i} \left(s _ {i}, o _ {i}\right), \inf  _ {\mathcal {X} _ {i, d} \cap \tau_ {i}} - h _ {i} \left(s _ {i}, o _ {i}\right), \inf  _ {\mathcal {X} _ {i, h} \cap \tau_ {i}} \left(\dot {h} _ {i} + \alpha \left(h _ {i}\right)\right) \right\}. \tag {3}
$$

The set  $\mathcal{X}_{i,h} := \{(s_i, o_i) \mid h_i(s_i, o_i) \geq 0\}$ . Notice that the third item on the right side of Equation (3) depends on both the control policy and CBF, since  $\dot{h}_i = \nabla_{s_i} h_i \cdot f_i(s_i, u_i) + \nabla_{o_i} h_i \cdot \dot{o}_i(t), u_i = \pi_i(s_i, o_i)$ . It is clear that if we can find  $h_i$  and  $\pi_i(s_i, o_i)$  such that  $y_i(\tau_i, h_i, \pi_i) > 0$  for  $\forall \tau_i \in \mathcal{T}_i$  and  $\forall i$ , then the conditions in (2) are satisfied. For each agent  $i$ , assume that we are given  $z_i$  i.i.d trajectories  $\{\tau_i^1, \tau_i^2, \dots, \tau_i^{z_i}\}$  drawn from distribution  $\mathcal{D}_i$  during training. We solve the objective:

$$
\text {F o r a l l} i, \text {f i n d} h _ {i} \in \mathcal {H} _ {i} \text {a n d} \pi_ {i} \in \mathcal {V} _ {i}, \quad \text {s . t .} \quad y _ {i} \left(\tau_ {i} ^ {j}, h _ {i}, \pi_ {i}\right) \geq \gamma , \forall j = 1, 2, \dots z _ {i}, \tag {4}
$$

where  $\gamma > 0$  is a margin that allows us to derive probabilistic guarantees later. We denote the solution to (4) as  $\hat{h}_i$  and  $\hat{\pi}_i$ . Denote the Rademacher complexity of the function class of  $y_i$  as:

$$
\mathcal {R} _ {z _ {i}} \left(\mathcal {Y} _ {i}\right) := \sup  _ {\tau_ {i} ^ {1}, \dots \tau_ {i} ^ {z _ {i}} \sim \mathcal {D} _ {i}} \mathbb {E} _ {\xi \sim \text {U n i f} (\{\pm 1 \} ^ {z _ {i}})} \sup  _ {h _ {i} \in \mathcal {H} _ {i}, \pi_ {i} \in \mathcal {V} _ {i}} \frac {1}{z _ {i}} \left| \sum_ {j = 1} ^ {z _ {i}} \xi_ {j} y _ {i} \left(\tau_ {i} ^ {j}, h _ {i}, \pi_ {i}\right) \right|, \tag {5}
$$

where  $\xi \in \mathbb{R}^{z_i}$  is a random vector and  $\xi_j$  denotes its  $j^{th}$  element. Also we define  $\epsilon_{i}$  as the probability that the decentralized CBF conditions are violated for agent  $i$  over randomly sampled trajectories (not necessarily the samples encountered in training). Under such definition,  $\epsilon_{i}$  measures the generalization error and can be expressed as  $\epsilon_{i} = \mathbb{P}_{\tau_{i}\sim \mathcal{D}_{i}}\left[y_{i}(\tau_{i},\hat{h}_{i},\hat{u}_{i})\leq 0\right]$ . Then we have Proposition 2 that provides generalization guarantees for all the learned  $\hat{h}_i$  and  $\hat{\pi}_i$ .

Proposition 2 (Generalization Error Bound of Learning Decentralized CBF). Assume that  $|y| \leq b$  and (4) is feasible. Let  $\hat{h}_i$  and  $\hat{u}_i$  be the solutions to (4) and  $\mu$  be a universal positive constant vector. Recall that  $N$  is the number of agents. Then, for any  $\delta \in (0,1)$  the following statement holds:

$$
\mathbb {P} \left[ \bigcap_ {i = 1} ^ {N} \left(\epsilon_ {i} \leq \mu_ {i} \frac {\log^ {3} z _ {i}}{\gamma^ {2}} \mathcal {R} _ {z _ {i}} ^ {2} (\mathcal {Y} _ {i}) + \mu_ {i} \frac {\log (N \log (4 b / \gamma) / \delta)}{z _ {i}}\right) \right] \geq 1 - \delta . \tag {6}
$$

The proof is provided in the supplementary material. The left side of Equation (6) is the probability that the generalization error  $\epsilon_{i}$  is upper bounded for all the  $N$  agents. Similar to the discussions in Section 4 in Boffi et al. (2020), for specific function classes of  $\mathcal{H}_i$  and  $\nu_{i}$ , such as Lipschitz parametric function or Reproducing kernel Hilbert space function classes, the Rademacher complexity of the function classes can be further bounded, leading to vanishing generalization errors as the number of samples  $z_{i}$  increases. Such derivations are standard, and are thus omitted as they are not

![](images/c0bcb39af5dfb611c23c4b384910d1b507624718598c9c78801029fdc8ee62ab.jpg)  
Figure 1: The computational graph of the control-certificate jointly learning framework in multi-agent systems. Only the graph for agent  $i$  is shown because agents have the same graph and the computation is decentralized.

the focus of the present paper. Note that the Lipschitz function class includes some neural networks with differentiable activation functions, which will be used in our experiments in Section 4.

Although we have presented some generalization guarantee for learning decentralized CBF, there still exists a gap between the theory and practical implementation. First, the theory does not provide a concrete way of designing loss functions to realize the optimization objectives in (4). Second, in theory, there are still  $N$  pairs of functions  $(h_i, \pi_i)$  to be learned. Unfortunately, the dimension of the input  $o_i$  of the functions  $h_i$ ,  $\pi_i$  are different for each agent  $i$ , and will even change over time in practice, as the proximity of other agents is time-varying, leading to time-varying local observations. To scale to an arbitrary number of agents,  $h_i$  and  $\pi_i$  should be invariant to the quantity and permutation of neighbourhood agents. Third, the theory does not provide ways to deal with scenarios where the decentralized CBF conditions are not (strictly) satisfied, i.e., where problem (4) is not feasible, which may very likely occur when the system becomes too complex or the function classes are not rich enough. To this end, we propose effective approaches to solving these issues, facilitating the scalable learning of safe multi-agent control in practice, as to be introduced next.

# 4 SCALABLE LEARNING OF DECENTRALIZED CBF IN PRACTICE

Following the theory in Section 3, we consider the practical learning of safe multi-agent control with neural barrier certificates, i.e., using neural networks for  $\mathcal{H}$  and  $\nu$ . We will present the formulation of loss functions in Section 4.1, which corresponds to the objective in (4). Section 4.2 presents the neural network architecture of  $h_i$  and  $\pi_i$ , which are invariant to the quantity and permutation of neighboring agents. Section 4.3 demonstrates a spontaneous policy refinement method that enables the control policy to satisfy the decentralized CBF conditions as possible as it could during testing.

# 4.1 LOSS FUNCTIONS OF JOINTLY LEARNING CONTROLLER AND BARRIER CERTIFCATES

Based on Section 3.2, the main idea is to jointly learn the control policies and control barrier functions in multi-agent systems. During training, the CBFs regulate the control policies to satisfy the decentralized CBF conditions (2) so that the learned policies are safe. All agents are put into a single environment to generate experiences, which are combined to minimize the empirical loss function  $\mathcal{L}^c = \Sigma_i\mathcal{L}_i^c$ , where  $\mathcal{L}_i^c$  is the loss function for agent  $i$  formulated as:

$$
\begin{array}{l} \mathcal {L} _ {i} ^ {c} \left(\theta_ {i}, \omega_ {i}\right) = \sum_ {s _ {i} \in \mathcal {X} _ {i, 0}} \max  \left(0, \gamma - h _ {i} ^ {\theta_ {i}} \left(s _ {i}, o _ {i}\right)\right) + \sum_ {s _ {i} \in \mathcal {X} _ {i, d}} \max  \left(0, \gamma + h _ {i} ^ {\theta_ {i}} \left(s _ {i}, o _ {i}\right)\right) \\ + \sum_ {s _ {i} \in \mathcal {X} _ {i, h}} \max  \left(0, \gamma - \nabla_ {s _ {i}} h _ {i} ^ {\theta_ {i}} \cdot f _ {i} \left(s _ {i}, \pi_ {i} ^ {\omega_ {i}} \left(s _ {i}, o _ {i}\right)\right) - \nabla_ {o _ {i}} h _ {i} ^ {\theta_ {i}} \cdot \dot {o} _ {i} - \alpha \left(h _ {i} ^ {\theta_ {i}}\right)\right), \tag {7} \\ \end{array}
$$

where  $\gamma$  is the margin defined in Section 3.2.  $\theta_{i}$  and  $\omega_{i}$  are neural network parameters. On the right side of Equation (7), the three items enforce the three CBF conditions respectively. For the class- $\mathcal{K}$  function  $\alpha (\cdot)$ , we simply choose a linear function  $\alpha (h) = \lambda h$ . Note that  $\mathcal{L}^c$  mainly considers safety instead of goal reaching. To train a safe control policy  $\pi_i(s_i,o_i)$  that can drive the agent to the goal state, we also minimize the distance between  $u_{i}$  and  $u_{i}^{g}$ , where  $u_{i}^{g}$  is the reference control input computed by classical approaches (e.g., LQR and PID controllers) to reach the goal. During training, the parameters  $\theta_{i}$  and  $\omega_{i}$  are optimized via mini-batch stochastic gradient descent. We present the computational graph in Figure 1 to help understand the information flow.

# 4.2 QUANTITY-PERMUTATION INVARIANT OBSERVATION ENCODER

Recall that in Section 3.1, we define  $o_i$  as the local observation of agent  $i$ .  $o_i$  contains the states of neighboring agents and its dimension can change dynamically. In order to scale to an arbitrary

![](images/3cc2d3a8d6eb912ae0ce4f773aef12696f89e802d3ee67a5549fc57f31bc1fc0.jpg)  
Figure 2: Neural network architecture of the control policy. The blue part indicates the quantity-permutation invariant observation encoder, which maps  $o_i(t) \in \mathbb{R}^{n \times |\mathcal{N}_i(t)|}$  with time-varying dimension to a fixed length vector. The network takes the state  $s_i$  and local observation  $o_i$  as input to compute a control action  $u_i$ . The neural network of the decentralized CBF  $h_i$  has a similar architecture except that the output is a scalar.

number of agents, there are two pivotal principles of designing the neural network architectures of  $h_i(s_i,o_i)$  and  $\pi_i(s_i,o_i)$ . First, the architecture should be able to dynamically adapt to the changing quantity of observed agents that affects the dimension of  $o_i$ . Second, the architecture should be invariant to the permutation of observed agents, which should not affect the output of  $h_i$  or  $\pi_i$ . All these challenges arise from encoding the local observation  $o_i$ . Inspired by PointNet (Qi et al., 2017), we leverage the max pooling layer to build the quantity-permutation invariant observation encoder.

Let us start with a simple example with input observation  $o_i(t) \in \mathbb{R}^{n \times |\mathcal{N}_i(t)|}$ , where  $n$  is the dimension of state and  $\mathcal{N}_i(t)$  is the set of the neighboring agents at time  $t$ .  $n$  is fixed while  $\mathcal{N}_i(t)$  can change from time to time. The permutation of the columns of  $o_i$  is also dynamic. Denote the weight matrix as  $W \in \mathbb{R}^{p \times n}$  and the element-wise non-linear activation function as  $\sigma(\cdot)$ . Define the row-wise max pooling operation as  $\operatorname{RowMax}(\cdot)$ , which takes a matrix as input and outputs the maximum value of each row. Consider the following mapping  $\rho: \mathbb{R}^{n \times |\mathcal{N}_i(t)|} \mapsto \mathbb{R}^p$  formulated as

$$
\rho \left(o _ {i}\right) = \operatorname {R o w M a x} \left(\sigma \left(W o _ {i}\right)\right), \tag {8}
$$

where  $\rho$  maps a matrix  $o_i$  whose column has dynamic dimension and permutation to a fixed length feature vector  $\rho(o_i) \in \mathbb{R}^p$ . The dimension of  $\rho(o_i)$  remains the same even if the number of columns of  $o_i(t)$ , which is  $|\mathcal{N}_i(t)|$ , change over time. The network architecture of the control policy is shown in Figure 2, which uses the  $\mathrm{RowMax}(\cdot)$  operation. The network of the control barrier function is similar except that the output is a scalar instead of a vector.

# 4.3 SPONTANEOUS ONLINE POLICY REFINEMENT

We propose a spontaneous online policy refinement approach that produces even safer control policies in testing than the neural network has actually learned during training. When the model dynamics or environment settings are too complex and exceed the capability of the control policy, the decentralized CBF conditions can be violated at some points along the trajectories. Thanks to the control barrier function jointly learned with the control policy, we are able to refine the control input  $u_{i}$  online by minimizing the violation of the decentralized CBF conditions.

Given the state  $s_i$ , local observation  $o_i$ , and action  $u_i$  computed by the control policy, consider the scenario where the third CBF condition is violated, which means  $\nabla_{s_i} h_i \cdot f_i(s_i, u_i) + \nabla_{o_i} h_i \cdot \dot{o}_i + \alpha(h_i) < 0$  when  $h_i \geq 0$ . Let  $e_i \in \mathbb{R}^m$  be an increment of the action  $u_i$ . Define  $\phi(e_i): \mathbb{R}^m \mapsto \mathbb{R}$  as

$$
\phi \left(e _ {i}\right) = \max  \left(0, - \nabla_ {s _ {i}} h _ {i} \cdot f _ {i} \left(s _ {i}, u _ {i} + e _ {i}\right) - \nabla_ {o _ {i}} h _ {i} \cdot \dot {o} _ {i} - \alpha \left(h _ {i}\right)\right) + \mu | | e _ {i} | | _ {2} ^ {2}. \tag {9}
$$

For every timestep during testing, we initialize  $e_i$  to zero and check the value of  $\phi(e_i)$ .  $\phi(e_i) > 0$  indicates that the control policy is not good enough to satisfy the decentralized CBF conditions. Then we iteratively refine  $e_i$  by  $e_i = e_i - \nabla_e \phi(e_i)$  until  $\phi(e_i) - \mu ||e_i||_2^2 = 0$  or the maximum allowed iteration is exceeded. The final control input is  $u_i = u_i + e_i$ . Such a refinement can flexibly refine the control input to satisfy the decentralized CBF conditions as much as possible.

# 5 EXPERIMENTAL RESULTS

Baseline Approaches. The baseline approaches we compare with include: MAMPS (Zhang & Bastani, 2019), PIC (Liu et al., 2020) and MADDPG (Lowe et al., 2017). For the drone tasks, we also compare with model-based planning method MAFACTEST (Ma et al., 2019; Fan et al.,

![](images/3f55d7a5d8e5e194464964086c0e8aefa25c2067c22fc3b04d4f5dd434a59617.jpg)  
(a) Navigation

![](images/d010fbcd8c733055e740b6fd572c72ab30a82056c457cb9db574e30b1d11cc5b.jpg)  
(b) Predator-Prey

![](images/430971065eabe4d4ac4b9c572ab4ca6119efd7a4a76f0b473b7c84aa61f9c8b9.jpg)  
(c) Nested Rings

![](images/f0594e092d35dab99203d3de1cf837c3b9b4cfad06e24a860b653f0eb329d075.jpg)  
Figure 3: Illustrations of the 2D environments used in the experiments. The Navigation and Predator-Prey environments are adopted from the multi-agent particle environment (Lowe et al., 2017). The Nested-Rings environment is adopted from Rodríguez-Seda et al. (2014).

![](images/f2408735ca4b5f8aee0d5f4571a9d98efe657f2ac09b29562d06072a0a9be99c.jpg)

![](images/5a3ad7c237b1ac796623f07ac23377dbf70c437637f19ff1ce8262b89e59959b.jpg)

![](images/d4be621785fe09fabc1eebac74949d649d2ec0c81d8b5eb98963ff956c02773c.jpg)  
Figure 4: Safety rate and reward in the 2D tasks. Results are taken after each method converged and are averaged over 10 independent trials.

![](images/c24599545bb9b8a74b180739192841a6978450272d637aa052451eabc929e0ed.jpg)

![](images/43cbc7872df8e001a4ed21ad0129d30dcafa111d91be2c4b11287df91dc3e874.jpg)

2020). A brief description of each method is as follows. MAMPS leverages the model dynamics to iteratively switch to safe control policies when the learned policies are unsafe. PIC proposes the permutation-invariant critic to enhance the performance of multi-agent RL. We incorporate the safety reward to its reward function and denote this safe version of PIC as PIC-Safe. The safety reward is -1 when the agent enters the dangerous set. MADDPG is a pioneering work on multiagent RL, and MADDPG-Safe is obtained by adding the safety reward to the reward function that is similar to PIC-Safe. MAFACTEST is a multi-agent version of the planning method FACTEST (Fan et al., 2020) combined with priority-based search (Ma et al., 2019).

Evaluation Criteria. Since the primal focus of this paper is the safety of multi-agent systems, we use the safety rate as a criteria when evaluating the methods. The safety rate is calculated as  $\mathbb{E}_{t\in T}\left[\prod_{i = 1}^{N}\mathbb{I}((s_i(t),o_i(t))\in \mathcal{X}_s)\right]$  where  $\mathbb{I}(\cdot)$  is the indicator function that is 1 when its argument is true or 0 otherwise. In addition to the safety rate, we also calculate the average reward that considers how good the task is accomplished. The agent is given a  $+10$  reward if it reaches the goal and a -1 reward if it enters the dangerous set. Note that the agent might enter the dangerous set for many times before reaching the goal. The upper-bound of the total reward for an agent is  $+10$ , which is attained when the agent successfully reaches the goal and always stays in the safe set.

Ground Robots. We consider three tasks illustrated in Figure 3. In the Navigation task, each agent starts from a random location and aims to reach a random goal. In the Predator-Prey task, the preys aim to gather the food while avoid being caught by the predators chasing the preys. We only consider the safety of preys but not predators. In the Nested-Rings task, the agents aim to follow the reference trajectories while avoid collision. When adding more agents to an environment, we will also enlarge the area of the environment to ensure the overall density of agents remains similar. Figure 4 demonstrates that When the number of agents grows (e.g., 32 agents), our approach (MDBC) can still maintain a high safety rate and average reward, while other methods have much worse performance. We also show the generalization capability of MDBC with up to 1024 in the appendix and also visualization results in the supplementary materials.

![](images/8737bd12339119c9d1c52d21864f6e46d8bcf04bb9673d3defc79e2c45de03dc.jpg)

![](images/bee79dcfb5de2fe8c266d001e9208efcaec1aad4df0d8e68be231eb513171dfd.jpg)

![](images/7c3ec32614dfa0559feabc01ea34a43fc20d8ff6f1be640c58fabda671e81751.jpg)

![](images/25442bdfc99d402ff952a976cd9df4f8d0dad18504e8635da4795154dd841560.jpg)  
Figure 6: Environments and results of 3D tasks. In Maze and Tunnel, the initial and target locations of each drone are randomly chosen. The drones start from the initial locations and aim to reach the targets without collision. The results are taken after each method converged and are averaged over 10 independent trials.

![](images/e8cc47ff126dbbdfb0a2eb15e0b567e9b48015e4ffeff4bb7ce6ac0c304a9dda.jpg)

![](images/55322d3b64b42974e22c2497cbbf9287ab1c4a57cbbd9c545a1494c36e8f9f52.jpg)

![](images/9c366df1ab88c5909271ef6c66c0a3495c2b095ea763c70cab88391375666967.jpg)  
Figure 7: Generalization capability of MDBC in the 3D tasks. MDBC can be trained with 8 agents in one environment and generalize to 1024 agents in another environment in testing.

![](images/15ac5a64da1519dcca587389e7f11c1c6adb3fcdbcee241a754399789d7f2160.jpg)

Drones. We experiment with 3D drones whose dynamics are even more complex. Figure 6 demonstrates the environments and the results of each approach. Similar to the results of ground robots, when there are a large number of agents (e.g., 32 agents), our method can still maintain a high reward and safety rate, while other methods have worse performance. Figure 7 shows the generalization capability of our method across different environments and number of agents. For each experiment, we train 8 agents during training, but test with up to 1024 agents. The extra agents are added by copying the neural network parameters of the trained 8 agents. Results show that our method has remarkable generalization capability to diverse scenarios. For both

![](images/e48a00900b2232639ca77472a421c1d54433e94598c6140f5241d188912d1331.jpg)  
Figure 5: Illustration of the Maze environment with 1024 drones. Videos can be found in the supplementary material.

the ground robot and drone experiments, we provide video demonstrations in the supplementary material. Details regarding the model dynamics can be found in the appendix. Our implementation will be made available upon acceptance of the paper.

# 6 CONCLUSION

This paper presents a novel approach of learning safe multi-agent control via jointly learning the decentralized control barrier functions as safety certificates. We provide the theoretical generalization bound, as well as the effective techniques to realize the learning framework in practice. Experiments show that our method significantly outperforms previous methods by being able to scale to an arbitrary number of agents, and demonstrates remarkable generalization capabilities to unseen and complex multi-agent environments.

# REFERENCES

Javier Alonso-Mora, Andreas Breitenmoser, Martin Rufli, Paul Beardsley, and Roland Siegwart. Optimal reciprocal collision avoidance for multiple non-holonomic robots. In Distributed Autonomous Robotic Systems, pp. 203-216. Springer, 2013.  
Aaron D Ames, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs with application to adaptive cruise control. In Decision and Control (CDC), 2014 IEEE 53rd Annual Conference on, pp. 6271-6278. IEEE, 2014.  
Nicholas M Boffi, Stephen Tu, Nikolai Matni, Jean-Jacques E Slotine, and Vikas Sindhwani. Learning stability certificates from data. arXiv preprint arXiv:2008.05952, 2020.  
Urs Borrmann, Li Wang, Aaron D Ames, and Magnus Egerstedt. Control barrier certificates for safe swarm behavior. IFAC-Papers-OnLine, 48(27):68-73, 2015.  
Ya-Chien Chang, Nima Roohi, and Sicun Gao. Neural lyapunov control. In Advances in Neural Information Processing Systems, pp. 3245-3254, 2019.  
Yu Fan Chen, Michael Everett, Miao Liu, and Jonathan P How. Socially aware motion planning with deep reinforcement learning. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1343-1350. IEEE, 2017a.  
Yuxiao Chen, Huei Peng, and Jessy Grizzle. Obstacle avoidance for low-speed autonomous vehicles with barrier function. IEEE Transactions on Control Systems Technology, 26(1):194-206, 2017b.  
Yuxiao Chen, Andrew Singletary, and Aaron D Ames. Guaranteed obstacle avoidance for multirobot operations with limited actuation: a control barrier function approach. IEEE Control Systems Letters, 5(1):127-132, 2020.  
Richard Cheng, Gábor Orosz, Richard M Murray, and Joel W Burdick. End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks. In AAAI Conference on Artificial Intelligence, volume 33, pp. 3387-3395, 2019.  
Richard Cheng, Mohammad Javad Khojasteh, Aaron D Ames, and Joel W Burdick. Safe multi-agent interaction through robust control barrier functions with learned uncertainties. arXiv preprint arXiv:2004.05273, 2020.  
Jason Choi, Fernando Castañeda, Claire J Tomlin, and Koushil Sreenath. Reinforcement learning for safety-critical control under model uncertainty, using control lyapunov functions and control barrier functions. arXiv preprint arXiv:2004.07584, 2020.  
Michael Everett, Yu Fan Chen, and Jonathan P How. Motion planning among dynamic, decision-making agents with deep reinforcement learning. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 3052-3059. IEEE, 2018.  
Chuchu Fan, Kristina Miller, and Sayan Mitra. Fast and guaranteed safe controller synthesis for nonlinear vehicle models. In Shuvendu K. Lahiri and Chao Wang (eds.), Computer Aided Verification, pp. 629-652, Cham, 2020. Springer International Publishing.  
Wanxin Jin, Zhaoran Wang, Zhuoran Yang, and Shaoshuai Mou. Neural certificates for safe control policies. arXiv preprint arXiv:2006.08465, 2020.  
Iou-Jen Liu, Raymond A Yeh, and Alexander G Schwing. Pic: permutation invariant critic for multi-agent deep reinforcement learning. In Conference on Robot Learning, pp. 590-602, 2020.  
Ryan Lowe, Yi I Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multiagent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems, pp. 6379–6390, 2017.  
Hang Ma, Daniel Harabor, Peter. J Stuckey, Jiaoyang Li, and Sven Koenig. Searching with consistent prioritization for multi-agent path finding. AAAI 2019: Thirty-Third AAAI Conference on Artificial Intelligence, 33(1):7643-7650, 2019.

Ian M Mitchell, Alexandre M Bayen, and Claire J Tomlin. A time-dependent hamilton-jacobi formulation of reachable sets for continuous dynamic games. IEEE Transactions on automatic control, 50(7):947-957, 2005.  
Stephen Prajna, Ali Jadbabaie, and George J Pappas. A framework for worst-case and stochastic safety verification using barrier certificates. IEEE Transactions on Automatic Control, 52(8): 1415-1428, 2007.  
Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.  
Han Qie, Dianxi Shi, Tianlong Shen, Xinhai Xu, Yuan Li, and Liujing Wang. Joint optimization of multi-UAV target assignment and path planning based on multi-agent reinforcement learning. IEEE Access, 7:146264-146272, 2019.  
Alexander Robey, Haimin Hu, Lars Lindemann, Hanwen Zhang, Dimos V Dimarogonas, Stephen Tu, and Nikolai Matni. Learning control barrier functions from expert demonstrations. arXiv preprint arXiv:2004.03315, 2020.  
Erick J. Rodríguez-Seda, Chinpei Tang, Mark W. Spong, and Dušan M. Stipanović. Trajectory tracking with collision avoidance for nonholonomic vehicles with acceleration constraints and limited sensing. The International Journal of Robotics Research, 33(12):1569-1592, 2014.  
Matteo Saveriano and Dongheui Lee. Learning barrier functions for constrained motion planning with dynamical systems. arXiv preprint arXiv:2003.11500, 2020.  
Shai Shalev-Shwartz, Shaked Shammah, and Amnon Shashua. Safe, multi-agent, reinforcement learning for autonomous driving. arXiv preprint arXiv:1610.03295, 2016.  
Nathan Srebro, Karthik Sridharan, and Ambuj Tewari. Smoothness, low noise and fast rates. In Advances in Neural Information Processing Systems 23, pp. 2199-2207, 2010.  
Mohit Srinivasan, Amogh Dabholkar, Samuel Coogan, and Patricio Vela. Synthesis of control barrier functions using a supervised machine learning approach. arXiv preprint arXiv:2003.04950, 2020.  
Andrew Taylor, Andrew Singletary, Yisong Yue, and Aaron Ames. Learning for safety-critical control with control barrier functions. In *Learning for Dynamics and Control*, pp. 708-717, 2020.  
Jur Van den Berg, Ming Lin, and Dinesh Manocha. Reciprocal velocity obstacles for real-time multi-agent navigation. In IEEE International Conference on Robotics and Automation (ICRA), pp. 1928-1935. IEEE, 2008.  
Li Wang, Aaron D Ames, and Magnus Egerstedt. Safety barrier certificates for collisions-free multirobot systems. IEEE Transactions on Robotics, 33(3):661-674, 2017.  
Li Wang, Evangelos A Theodorou, and Magnus Egerstedt. Safe learning of quadrotor dynamics using barrier certificates. In IEEE International Conference on Robotics and Automation (ICRA), pp. 2460-2465. IEEE, 2018.  
Peter Wieland and Frank Allgower. Constructive safety using control barrier functions. IFAC Proceedings Volumes, 40(12):462-467, 2007.  
Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, and Tamer Basar. Fully decentralized multiagent reinforcement learning with networked agents. In International Conference on Machine Learning, pp. 5872-5881, 2018.  
Wenbo Zhang and Osbert Bastani. Mamps: Safe multi-agent reinforcement learning via model predictive shielding. arXiv preprint arXiv:1910.12639, 2019.  
Lianmin Zheng, Jiacheng Yang, Han Cai, Weinan Zhang, Jun Wang, and Yong Yu. Agent: A many-agent reinforcement learning platform for artificial collective intelligence. arXiv preprint arXiv:1712.00600, 2017.
