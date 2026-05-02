# Accelerating Quadratic Optimization with Reinforcement Learning

Anonymous Author(s)

Affiliation

Address

email

# Abstract

First-order methods for quadratic optimization such as OSQP are widely used for large-scale machine learning and embedded optimal control, where many related problems must be rapidly solved. These methods face two persistent challenges: manual hyperparameter tuning and convergence time to high-accuracy solutions. To address these, we explore how Reinforcement Learning (RL) can learn a policy to tune parameters to accelerate convergence. In experiments with well-known QP benchmarks we find that our RL policy, RLQP, significantly outperforms state-of-the-art QP solvers by up to  $3\mathrm{x}$ . RLQP generalizes surprisingly well to previously unseen problems with varying dimension and structure from different applications, including the QPLIB, Netlib LP and Maros-Meszáros problems.

# 1 Introduction

Solving quadratic programs (QPs) efficiently is critical to applications in finance, robotic control and operations research. While state-of-the-art interior-point methods scale poorly with problem dimensions, first-order methods for solving QPs typically require thousands of iterations. Moreover, real-time control applications have tight latency constraints for solvers [31]. Therefore, it is important to develop efficient heuristics to solve QPs in fewer iterations.

The Alternating Direction Method of Multipliers (ADMM) [6, 15, 18] is an efficient first-order optimization algorithm, and is the basis for the widely used and state-of-the-art Operator-Splitting QP (OSQP) solver [42]. ADMM performs a linear solve on a matrix based on the optimality conditions of the QP to generate a step direction, and then projects the step onto the constraint bounds.

While state-of-the-art, the ADMM algorithm has numerous hyperparameters that must be tuned with heuristics to regularize and control optimization. Most importantly, the step size parameter  $\rho$  has considerable impact on the convergence rate. However, is still unclear how to select  $\rho$  before attempting the QP solution. While some theoretical works compute the optimal  $\rho$  [17], they rely on solving semidefinite optimization problems which are much harder than solving the QP itself. Alternatively, some heuristics introduce "feedback" by adapting  $\rho$  throughout optimization in order to balance primal and dual residuals [42, 6, 22].

We propose RLQP (see Fig. 1), an accelerated QP solver based on OSQP that uses reinforcement learning to adapt the internal parameters of the ADMM algorithm between iterations to minimize solve times. An RL algorithm learns a policy  $\pi_{\theta} \colon S \to \mathcal{A}$ , parameterized by  $\theta$  (e.g., the weights of a neural network), that maps states in a set  $S$  to actions in a set  $\mathcal{A}$  such that the selected action maximizes an accumulated reward  $r$ . To train the policy for RLQP, we define  $S$  to be the internal state of the QP solver (e.g., the constraint bounds, the primal and dual estimates),  $\mathcal{A}$  to be the adaptation to the internal parameter  $(\rho)$  vector, and  $r$  to minimize the number of ADMM iterations taken.

![](images/2f456f083d976a6bf6cedc6b7309f8322b628559a2889bc2657072a41efd4cfa.jpg)  
Figure 1: RLQP uses deep reinforcement learning (RL) to compute a policy that adapts the internal parameters of a first-order quadratic program (QP) solver to speed up the solver's convergence rate. In a standard RL formulation, a policy computes an action based on its observation of the state of the environment, and taking the action results in a change in state and a reward. In RLQP, the policy is parameterized by a neural network, the state is the internal state of the QP solver, the action changes a parameter  $(\rho)$  of the solver, and the reward minimizes the time required to solve the QP.

RLQP's policy can be trained either jointly across general classes of QPs or with respect to a specific class. The general version of RLQP is trained once on a broad class of QPs and can be used out-of-the-box on new problems. The specialized version of RLQP is trained on a specific class of problems that the solver will repeatedly encounter. While this requires additional setup and training time, it is useful when QPs will be repeatedly solved in application (e.g., in a  $100\mathrm{Hz}$  control loop).

In experiments, we train RLQP on a set of randomized QPs, and compare convergence rates of RLQP to non-adaptive and heuristic adaptive policies. To compare generalization and specialization, we investigate RLQP's performance in the settings where 1) the train and test sets of QPs come from the same class of problems, 2) the train set contains from superset of classes contained in the test set, 3) the train set contains a subset, and 4) when the train and test sets are from distinct classes. In the results section we show that RLQP outperforms OSQP by up to  $3\mathrm{x}$ .

The contributions of this paper are:

- Two RL formulations to train policies that provide coarse (scalar) and fine (vector) grain updates to the internal parameters of a QP solver for faster convergence times  
- Policies trained jointly across QP problem classes or to specialize to specific classes  
- Experimental results showing that RLQP reduces convergence times by up to  $3\mathrm{x}$  and generalizes to different problem classes and outperform existing methods

# 2 Related Work

This work touches a number of related research areas, including convex optimization, using machine learning (ML) to speed up optimization, learning in first-order methods, and reinforcement learning.

Convex optimization Many researchers have proposed algorithms for quadratic programs, which generally fall into three classes: active set [45], interior point [35], and first-order methods. Of the active set and interior point solvers, perhaps the most well-known are Gurobi [20] and MOSEK [34]. Active-set solvers operate by iteratively adapting an active set of constraints based on the cost function gradient and dual variables [37]. Interior-point solvers iteratively introduce and vary barrier functions to represent constraints and solve unconstrained convex problems. We instead base this work on a first-order method solver, OSQP [42]. One of the advantages of OSQP over interior points solvers, is that they can readily be warm started from a near-by solution, as is common in many applications such as solving a sequential quadratic program [39] and solving QPs for model-predictive control.

ML-accelerated combinatorial optimization Accelerating combinatorial optimization problems with deep learning has been explored with wide application [4, 5], including branch-and-bound for

mixed-integer linear programming [1, 25], graph algorithms [9] and boolean satisfiability problems (SAT) [7]. Many combinatorial optimization problems have exponential search spaces and are NP-hard in a general setting. However, learning-augmented combinatorial algorithms utilize very different methods to RLQP as combinatorial problems have discrete search spaces.

Learning in first-order methods Accelerating first-order methods with machine learning has gained considerable recent interest. Li and Malik [26] demonstrate a learned optimization algorithm outperforms common first-order methods for several convex problems and a small non-convex problem. Metz et al. [32] show a learned policy outperforms first-order methods when optimizing neural networks, but finds that directly learning parameter update values can be sensitive to exploding gradient problems. We avoid this instability during optimization by learning a policy to adapt parameters of the ADMM algorithm. Wei et al. [44] recently proposed an RL agent to tune parameters for an ADMM-based inverse imaging solver.

Reinforcement Learning Overview Reinforcement learning (RL) algorithms include both on-policy algorithms, such as Proximal Policy Optimization [40], REINFORCE [43], and IMPALA [11], and off-policy algorithms, such as DQN [33] and Soft Actor Critic [21]. RLQP extends the off-policy Twin-Delayed DDPG (TD3) [13], an actor-critic framework with a exploration policy for continuous action spaces that extends Deep Deterministic Policy Gradient (DDPG) algorithm [27] while addressing approximation errors. Furthermore, in one formulation of RLQP, we train a shared policy for multiple agents following an RL approach proposed by Huang et al. [23]. With this single policy, RLQP updates multiple parameters using state associated with each constraint of a QP.

# 3 Background

In this section, we summarize QPs, the OSQP solver, and a MDP formalization.

# 3.1 Quadratic Programs

A quadratic program with  $n$  variables and  $m$  constraints takes the form:

$$
\begin{array}{l l} \text {m i n i m i z e} & (1 / 2) x ^ {T} P x + q ^ {T} x \\ \text {s u b j e c t t o} & l \leq A x \leq u, \end{array}
$$

where  $x \in \mathbb{R}^n$  is the optimization variable,  $P$  is an  $n \times n$  symmetric positive semi-definite matrix that defines the quadratic cost,  $q \in \mathbb{R}^n$  defines the linear cost,  $A$  is an  $m \times n$  matrix that defines the  $m$  linear constraints, and  $l, u \in \mathbb{R}^m$  are the constraint's lower and upper bounds. Here,  $\leq$  is an element-wise less-than-or-equal-to operator. In this form, to specify an equality constraint, the lower and upper bounds are set to the same value, and to specify a constraint unbounded from one side, a sufficiently large value (or  $\pm \infty$ ) is specified for the other side.

# 3.2 First-Order QP Solver Algorithm

The solver we speed up is OSQP, which uses a first-order ADMM method to solve QPs. We summarize OSQP here. Given a QP, OSQP first forms a  $KKT$  matrix (below), then iteratively refines a solution from a initialization point for vectors  $x^{(0)} \in \mathbb{R}^n$ ,  $y^{(0)} \in \mathbb{R}^m$ , and  $z^{(0)} \in \mathbb{R}^m$ , where the superscript in parenthesis refers to the iteration. Each iteration computes the values for the  $k + 1$  iterates by solving following linear system (e.g., with an  $LDL^T$  solver):

$$
\underbrace {\left[ \begin{array}{c c} P + \sigma I & A ^ {T} \\ A & \operatorname {d i a g} (\rho) ^ {- 1} \end{array} \right]} _ {\text {K K T m a t r i x}} \left[ \begin{array}{l} x ^ {(k + 1)} \\ v ^ {(k + 1)} \end{array} \right] = \left[ \begin{array}{c} \sigma x ^ {(k)} - q \\ z ^ {(k)} - \operatorname {d i a g} (\rho) ^ {- 1} y ^ {(k)} \end{array} \right] \tag {1}
$$

and then performing the following updates:

$$
\tilde {z} ^ {(k + 1)} \leftarrow z ^ {(k)} + \operatorname {d i a g} (\rho) ^ {- 1} (v ^ {(k + 1)} - y ^ {(k)})
$$

$$
z ^ {(k + 1)} \leftarrow \Pi \left(\tilde {z} ^ {(k + 1)} + \operatorname {d i a g} (\rho) ^ {- 1} y ^ {(k)}\right)
$$

$$
y ^ {(k + 1)} \leftarrow x ^ {(k)} + \operatorname {d i a g} (\rho) \left(\tilde {z} ^ {(k + 1)} - z ^ {(k + 1)}\right),
$$

where  $\sigma \in \mathbb{R}_+$  and  $\rho \in \mathbb{R}_+^m$  are regularization and step-size parameters, and  $\Pi : \mathbb{R}^m \to \mathbb{R}^m$  projects its argument on the constraint bounds. We use the notation  $\mathrm{diag} \colon \mathbb{R}^m \to \mathbb{S}^m$  to denote the operator that maps a vector to a diagonal matrix. When the primal and dual residual vectors are small enough in norm after  $k$  iterations,  $\mathbf{x}^{(k + 1)}$  and  $\mathbf{y}^{(k + 1)}$  are primal and dual (approximate) solutions to the QP and  $\mathbf{z}^{(k + 1)} \approx A\mathbf{x}^{(k + 1)}$  ( $z = Ax$  is true in the limit).

Internally, OSQP has a single scalar  $\bar{\rho}$  that it uses to form  $\rho$  according to the following formula:

$$
\rho_ {i} = \left\{ \begin{array}{l l} \bar {\rho} & \text {i f} l _ {i} \neq u _ {i} (\text {i n e q u a l i t y c o n s t r a i n t s}) \\ \bar {\rho} \cdot 1 0 ^ {3} & \text {i f} l _ {i} = u _ {i} (\text {e q u a l i t y c o n s t r a i n t s}), \end{array} \right. \tag {2}
$$

where the subscript  $i$  denotes the  $i$ -th coefficient of  $\rho$ , and the bounds  $l$  and  $u$ .

Periodically, between ADMM iterations, OSQP will adapt the value of  $\bar{\rho}$ . The existing hand-crafted formula for adapting  $\rho$  attempts to balance between primal and dual objectives, by setting  $\bar{\rho}^{(k + 1)}\gets \bar{\rho}^{(k)}\sqrt{\xi_{\mathrm{primal}} / \xi_{\mathrm{dual}}}$ , where  $\xi$  here are the residuals. Empirically, adapting  $\rho$  between iterations can speed up the convergence rate.

# 3.3 Multi-Agent Single-Policy MDP

In a Markov Decision Process (MDP), an agent can be in any state  $s \in S$ , take an action  $a \in \mathcal{A}$  and with the transition dynamics function,  $\mathcal{T}(\cdot | s, a)$ , transitions from state  $s$  to state  $s'$  after taking action  $a$ . The agent receives a reward  $R \colon S \times \mathcal{A} \to \mathbb{R}$  for transitioning from  $s$  to  $s'$  by taking action  $a$ . Given a tuple  $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$ , the MDP optimization objective is to find a policy  $\pi_{\theta} : \mathcal{S} \to \mathcal{A}$ , parameterized by  $\theta$ , that maximizes the expected cumulative reward  $E\left[\sum_{t=0}^{\infty} \gamma^{t} r^{t}\right]$ , where  $r^{t}$  is the reward at time  $t$  and  $\gamma \in [0,1)$  is a discount factor.

We also formulate a multi-agent single-policy MDP setting in which  $m$  agents collaborate in a shared environment in state  $s_{\mathrm{env}} \in S_{\mathrm{env}}$ . At each time step, each collaborating agent (CA)  $i$  has its own state  $s_i \in S_{\mathrm{ca}}$ , action  $a_i \in \mathcal{A}_{\mathrm{ca}}$ , and observations  $o_i \in \mathcal{O}$ , but, for computation feasibility, share a single policy  $\pi_\theta : S_{\mathrm{ca}} \to \mathcal{A}_{\mathrm{ca}}$ . State transitions for the environment and all  $m$  agents occur simultaneously according to a state transition function  $\mathcal{T} : S_{\mathrm{env}} \times S_{\mathrm{ca}}^m \times \mathcal{A}_{\mathrm{ca}}^m \to S_{\mathrm{env}} \times S_{\mathrm{ca}}^m$  and result in a single shared reward  $R : S_{\mathrm{env}} \times S_{\mathrm{ca}}^m \times \mathcal{A}_{\mathrm{ca}}^m \to \mathbb{R}$  and discount factor. The objective is to find a single shared policy  $\pi_\theta$  that maximizes the expected cumulative reward. This can be thought of as a special case of a multi-agent MDP [29] or Markov game [28], and we adapt a formulation from Huang et al. [23].

# 4 Method

The goal of RLQP is to learn a policy to adapt the  $\rho \in \mathbb{R}^m$  vector used in the ADMM update in (1) (see Fig. 1). As the dimensions of this vector vary between QPs, we propose two methods that can handle the variation in  $m$ . The first method learns a policy to adapt a scalar  $\bar{\rho}$  and then applies (2) to populate the coefficients of the  $\rho$  vector. The second method learns a policy to adapt individual coefficients of the  $\rho$  vector.

Since both the number of variables  $n$  and the number of constraints  $m$  can vary from problem to problem, and the same QP can be written in  $(n!m!)$  permutations, we propose learning policies that are problem size and permutation invariant. To do this, we provide a permutation-invariant fixed-size state of the QP solver to either policy.

# 4.1 RL Policy for Scalar Adaptation

To speed up convergence of OSQP, we hypothesize that RL can learn a scalar  $\bar{\rho}$  adaptation policy that can perform as-well-as or better than the current handcrafted policy ( $\pi_{\mathrm{hc}}$ ) of OSQP. The handcrafted policy in OSQP periodically adapts  $\rho$  by computing a single scalar  $\bar{\rho}$ , then sets the coefficients of  $\rho$  based on the value of  $\bar{\rho}$ . In both handcrafted and RL cases, the policy is a function  $\pi : S_{\bar{\rho}} \to A_{\bar{\rho}}$ , where  $S_{\bar{\rho}} \in \mathbb{R}^2$  are the primal and dual residuals stacked into a vector,  $A_{\bar{\rho}} \in \mathbb{R}$  is the value to set to  $\bar{\rho}$ . One advantage of this approach is that a simple heuristic can check that the proposed change to  $\bar{\rho}$  is sufficiently small and avoid a costly matrix factorization.

To compute this policy,  $\pi$ , we use Twin-Delayed DDPG TD3 [13], an extension of deep-deterministic policy gradients (DDPG) [27], as the action space is continuous. We summarize TD3 in Alg. 1. TD3

Algorithm 1 TD3 for  $\bar{\rho}$  (scalar)  
1: Input: exploration noise  $\sigma$ , buffer size rs  
2:  $\pi, Q \gets$  initialize policy and critic (see TD3)  
3:  $\mathcal{D} \gets$  replay buffer w/ rs  
4: env,  $s^{(0)} \gets$  new QP, its state  
5: for  $t \in \{0, \dots, T\}$  do  
6:  $\bar{\rho}^{(t)} \gets \pi(s^{(t)}) + \epsilon$ ,  $\epsilon \sim \mathcal{N}(0, \sigma)$   
7:  $s^{(t+1)}, r^{(t)}$ , done $^{(t)}$ $\leftarrow$  step(env,  $s^{(t)}, a^{(t)})$   
8: store  $(s^{(t)}, \rho^{(t)}, r^{(t)}, s^{(t+1)})$  in  $\mathcal{D}$   
9: if done $^{(t)}$  then  
10: env,  $s^{(t)} \gets$  new QP, its state  
11: update  $\pi$  and  $Q$  using data sampled from  $\mathcal{D}$

Algorithm 2 TD3 for  $\rho$  (vector)  
1: Input: exploration noise  $\sigma$ , buffer size rs  
2:  $\pi, Q \gets$  initialize policy and critic (see TD3)  
3:  $\mathcal{D} \gets$  replay buffer w/ ( $\mathbf{rs} \times (\text{avg # constraints})$ )  
4: env,  $\mathbf{s}^{(0)}, m \gets$  new QP, its state, # constraints  
5: for  $t \in \{0, \dots, T\}$  do  
6:  $\rho_i^{(t)} \gets \pi(s_i^{(t)}) + \epsilon$ ,  $\epsilon \sim \mathcal{N}(0, \sigma) \forall i \in [1 \dots m]$   
7:  $\mathbf{s}^{(t+1)}, r^{(t)}, \text{done}^{(t)} \gets \text{step}(\text{env}, \mathbf{s}^{(t)}, \boldsymbol{\rho}^{(t)})$   
8: store  $(\mathcal{D}, s_i^{(t)}, \rho_i^{(t)}, r^{(t)}, s_i^{(t+1)}) \forall i \in [1 \dots m]$   
9: if done(t) then  
10: env,  $\mathbf{s}^{(t)}, m \gets$  new QP, state, # constraints  
11: update  $\pi$  and  $Q$  using data sampled from  $\mathcal{D}$

learns the parameters  $\theta$  of a policy  $\pi_{\theta}$  network and critic  $Q$  network, where  $\pi$  determines the action to take and  $Q(s,a) = \mathbb{E}_s[r(s,a) + \gamma \mathbb{E}_{a'\sim \pi}[Q(s',a')]]$  is the expected reward for a given state-action pair following the recursive Bellman equation. TD3 updates  $Q$  by minimizing the loss on the Bellman equation, and updates the policy network using a policy gradient [43] of the objective

$$
J (\theta) = \mathbb {E} _ {s \sim \pi} [ R (s, a) ],
$$

that is,

$$
\nabla_ {\theta} J = \mathbb {E} _ {s \sim \mathcal {D}} [ \nabla_ {\theta} \pi_ {\theta} (s) \nabla_ {a} Q (s, a) | _ {a = \pi (s)} ]
$$

where  $\mathcal{D}$  is the discounted state visitation distribution [41]. For brevity, we leave out some details of TD3 in the algorithms, including:  $Q$  is composed of two networks, the minimum value of the two networks estimates the reward, exploration noise is clamped, and  $\pi$  network updates are staggered.

In RLQP, the "environment" env is an instance of a randomized QP problem, and a call to step() applies a change to  $\bar{\rho}$  (and thus via Eq. 2 to  $\rho$ ), advances a QP a fixed number of ADMM iterations, and returns the updated internal state  $s$ , a reward  $r$ , and a termination flag done. In this case, the internal state  $s$  is a vector containing the current primal and dual residuals of the QP. The reward  $r$  is  $-1$  if not done, and  $0$  if the QP is solved.

We train with randomized QPs across various problem classes (Sec. 5) that have solutions guaranteed by construction. To ensure progress, we set a step limit (not shown in the algorithm) since bad actions can cause the solver to fail to converge. During training, we also always adapt  $\rho$  in each step and ignore the heuristic adapt/no-adapt policy.

For well-scaled QPs, the residuals and  $\rho$  can reasonably range between  $10^{-6}$  and  $10^{6}$ . Since this can cause issues with training the policy networks, we train the policy network with logs of the residuals, and exponentiate the network's output to get the action to apply.

# 4.2 RL Policy for Vector Coefficient Adaptation

For some classes of QPs, the solver can further speed up convergence by adapting all coefficients of the vector  $\rho$ , instead of applying Eq. 2 to a scalar  $\bar{\rho}$ . Conceptually, this could be accomplished with a policy  $\pi_{\mathrm{vec}}: S_{\mathrm{QP}} \to A_{\mathrm{vec}}$ , where  $S_{\mathrm{QP}} \in \mathbb{R}^{O(n + m)}$  is the internal state of the solver and  $A_{\mathrm{vec}} \in \mathbb{R}_+^m$  is the new value for  $\rho$ . However, due to variation in problem size and permutation, we instead propose a simplification in which  $\pi_{\mathrm{vec}}$  is formulated as a policy  $\pi_{\rho}: S_{\rho} \to A_{\rho}$  that is applied per coefficient of  $\rho$ . Here,  $S_{\rho} \in \mathbb{R}^6$  is state corresponding to a single coefficient in  $\rho$ , and  $A_{\rho} \in \mathbb{R}$  is the value to set for that coefficient.

To define  $S_{\rho}$ , we observe that coefficients in  $\rho$  are one-to-one with coefficients in  $y, z, l, u$ , and  $Ax$ . We observe that constraint bounds are likely to have an impact on an ADMM iteration when coefficients of  $z$  are "close" to their bounds in  $l$  or  $u$ . A coefficient in  $z$  is also "close" to a solution when it is nearly equal to the corresponding coefficient in  $Ax$ . Finally, to include a permutation-invariant signal on the overall convergence, we include the primal and dual residuals of the QP solver; these are infinity norms of individual residuals, and is similar to using a max-pooling operation on

the input to a graph neural network [38, 3]. We thus define a coefficient's state as:

$$
s _ {i} = \left[ \begin{array}{c} \min  (z _ {i} - l _ {i}, u _ {i} - z _ {i}) \\ z _ {i} - (A x) _ {i} \\ y _ {i} \\ \rho_ {i} \\ \xi_ {\text {p r i m a l}} \\ \xi_ {\text {d u a l}} \end{array} \right] \in S _ {\rho}.
$$

In practice, we clamp values in each state  $s_i$  to reasonable ranges (e.g.,  $[10^{-8}, 10^{6}]$ ,  $[-10^{6}, 10^{6}]$ ,  $[-10^{6}, 10^{6}]$ ,  $[10^{-6}, 10^{6}]$ ,  $[10^{-6}, 10^{6}]$ ,  $[10^{-6}, 10^{6}]$  for the coefficients of  $s_i$ , in order). Empirically, training is more efficient if the policy operates on states with the log of the first and last 3 coefficients.

Since each step in the vector formulation applies  $m$  actions and updates  $m$  states simultaneously, we adapt the multi-agent single-policy TD3 formulation from Huang et al. [23], and show it in Alg. 2, with the main differences from Alg. 1 highlighted in blue. Before each step, step applies the policy with exploration noise to generate  $m$  actions (coefficient updates to  $\rho$ ). After each step, Alg. 2 adds the  $m$  states before the action, the  $m$  actions, and the  $m$  states after the action, along with the single reward to the replay buffer. Since each step results in  $m$  tuples added to the replay buffer, Alg. 2 allocates a replay buffer large enough to hold the average number of tuples that each QP in the training set will have.

The hypothesis of this approach is that the some coefficients, and thus policy actions for coefficients, will have more of an effect on convergence, and thus the reward, than others. When the domain for the policy function has more of an effect, the range of the actions will have lower variance. Similarly, when the policy values has less effect, the variance will be higher. This suggests that when training the policy network in this case, having a lower learning rate, and higher batch size can help. A lower learning rate will cause smaller gradient steps when training the network so that it does not overfit to some part of the high variance training data. A higher batch size will allow gradients to average out in high variance training data so that the gradient step better matches the true mean of the data.

# 5 Experiments

To train and test the proposed methods, we modify  $\mathrm{OSQP}^1$  to support direct querying and modification of its  $\rho$  vector, and integrate both  $\pi_{\bar{\rho}}$  and  $\pi_{\mathrm{vec}}$  policies for benchmarking, and a runtime flag to switch between policies. We train the network using randomly generated QPs from OSQP's benchmark suite. The form of these QPs falls into 7 classes (see below), but the specific coefficient values in the objective and constraints are generated from a random-number generator. These QPs are also guaranteed to be feasible by construction (e.g., by reverse engineering constraint values from a pre-generated solution). To separate train and test sets, we ensure that each set is generated from uniquely seeded random-number generators. Training is performed in PyTorch with a python wrapper around the modified OSQP which is written  $C / C++$ . During benchmarking, the solver performs runtime adaptation of  $\rho$  using PyTorch's  $C++$  API on the already-trained policy network. We train a small model to keep runtime network inference as fast as possible.

We evaluate all policies with 7 problem domains (referred to as the "benchmark problem") defined in Appendix A of the paper on OSQP [42]. These policies cover control, Huber fitting, support-vector machines (SVM), Lasso regression, Portfolio optimization, equality constrained, and random QP domains. Alongside RLQP, we benchmark the unmodified OSQP solver to evaluate how the RL policy improves convergence. While our focus is on improving the first-order method in OSQP with an RL policy, we include some benchmarks against the state-of-the-art commercial Gurobi solver [20] as it may be of interest to a practitioner.

We consider three evaluation configurations: (1) multi-task policy learning in which we train a single RLQP policy on a suite of random benchmark problems and test it across all problems, (2) class-specific policy learning in which we train and test the policy for a single problem domain and (3) zero-shot generalization where we test a general policy on a novel unseen problem class.

We evaluate speedups with the shifted geometric mean [19] as problems have wide variations in runtime across several orders of magnitude. This metric is the standard benchmark used by

![](images/e325f6015276f912bba02c62fb368e934dc79af21970fed023a980ec7b6e9448.jpg)  
Figure 2: Left: Comparison of general adaptation policy applied to different classes. We train an RL policy using multiple classes, and show the performance per class, along with each class. The  $y$ -axis is the shifted geometric mean across problems within each class, and the value of 1 is always assigned to the best in class. The right-most All class is the aggregate of all classes to the left of it. Right: Comparison of warm-starting performance using OSQP's warm-start benchmark.

![](images/2b6b49573ba54c5e467f1076e045d99d6a562c9b367c6e19f30e238a0b85c644.jpg)

228 optimization community. The shifted geometric mean is computed as:

$$
\exp \sum_ {i = 1} ^ {N} (1 / N) \log (\max  (1, v _ {i} + s)) - s,
$$

where  $v_{i}$  is compute time in seconds,  $s = 10$ , and  $N$  is the number of values (e.g., QPs solved).

We also evaluate on QPLIB [14], Netlib [16], and Maros and Mészáros [30], as they are well-established benchmark problems in the optimization community.

In all experiments, the policy network architecture has 3 fully-connected hidden layers of 48 with ReLU activations between the input and output layers. The input layer is normalized, and the output activation is Tanh. The critic network architectures uses the identity function as the output activation, but otherwise matches the policy. As small networks for fast CPU inferences are desirable here, we attempted to keep the network as small as possible. We performed minimal experimentation before settling on this architecture—finding that smaller networks fail to converge during training.

We trained on a system with 256 GiB RAM, two Intel Xeon E5-2650 v4 CPUs @ 2.20 GHz for a total of 24 cores (48 hyperthreads), and five NVIDIA Tesla V100s. We ran benchmarks on a system with Intel i9 8-core CPU @ 2.4 GHz and without GPU acceleration.

More extensive analysis and ablation studies can be found in the supplemental material.

# 5.1 Multi-task/General RLQP Policy

We train a general policy on a broad set of problem classes and compare solve times with different classes. During training, we sample one of seven QP domains from benchmark problem. From that sampled problem domain, we generate a random problem.

In Fig. 2, we compare the shifted geometric mean of solving 10 problems of 20 different dimension, for a total of 200 runs per class per solver. The problem dimensions for Control, Huber, SVM, Lasso are (10, 11, 12, 13, 14, 16, 17, 20, 23, 26, 31, 37, 45, 55, 68, 84, 105, 132, 166, 209); for Random and Eq are (10, 11, 12, 13, 15, 18, 23, 29, 39, 53, 73, 103, 146, 211, 304, 442, 644, 940, 1373, 2009), and for Portfolio are (5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 35, 43, 52, 65, 80, 99, 124, 154). From the results, we observe that both RLQP adaptation policies typically improve upon convergence rate from the handcrafted policy in OSQP, and in some cases, e.g. Portfolio optimization, by up to  $3x$ .

# 5.2 Problem Dimension Scaling

To test how a trained policy scales to higher dimensions, we train a policy on low dimensional problems (10 to 50), and solve problems with varying dimensions, including dimensions higher than the training set (up to 2000). For comparison, we also include a policy trained on the full dimension range (10 to 2000). From the results plotted in Fig. 3, we observe that a policy trained on a lower dimensional training set, can show improvement beyond its training range. However, as the problem size diverges more from the training set, its performance suffers and it eventually loses to

![](images/d7638b151a463a1e76517adbfd649d0b5ad131225dd16cdcb7280f1b6e588830.jpg)  
Figure 3: Solve time with increasing dimension on the Random QP problem set. We train and benchmark two vector RL adaptation policies: (dashed) on problems ranging from dimension 10 to 50, and (solid) on problems ranging from 10 to 2000. The gray box shows the range of the training data for the dashed line. When the benchmark run is in the same problem-dimension distribution as the training data, the relative performance between solvers is consistent, however, when the problem dimension is outside of the training distribution, the performance diverges.

the handcrafted policy. Both low-dimensional and full-dimension range polices, were trained using the same network architecture, we hypothesize that this behavior is a function of the training data and not a limitation of the network expressiveness. While this is a disadvantage of using smaller problems for training, in practice it may be outweighed by the advantage in training time—as each RL step requires  $O((n + m)^3)$  compute time.

# 5.3 Training a Class-Specific Policy

Many applications require QPs from the same class to be repeatedly solved. To test if training a policy specific to a QP class can outperform a policy trained on the benchmark suite, we train policies specific to the problems generated by the trust-region [8] based solver for sequential quadratic program (SQP) from a grasp-optimized motion planner (GOMP) [24] for robots. With these problems, RLQP trained on the benchmarks converges more slowly than the handcrafted policy included in OSQP. With a vector policy trained on the the QPs from the SQP, the shifted geometric mean of OSQP is 1.37. This result suggests that while a general policy may work for multiple problem classes, there are cases in which it is beneficial to train a policy specific to a problem class, particularly if the QPs from that problem class are repeatedly solved.

# 5.4 Warm Starting QPs

One benefit of first-order method such as OSQP is their ability to warm start—that is, rapidly converge from a good initial guess. We test if RLQP retains the benefit of warm start on OSQP's warm-start benchmark and show the results in Fig. 2 (right). As warm starts require fewer iterations, and thus fewer adaptations than cold starts, we expect RLQP to show a smaller improvement here. In the plot, we can see that RLQP retains the benefit of warm starting, and also gains a improvement over OSQP.

# 5.5 QPLIB

We benchmark convex continuous QP instances with constraints from QPLIB [14], and show the results in Table 1. Since there are only a few such QPLIB instances and they come from varying classes, creating a train/test split is problematic. We thus use the general policy trained on the benchmark classes. From the table, we observe that the general RLQP policy beats OSQP's heuristic policy in all but three cases. In two cases RLQP fails due to reaching an iteration or time limit, which may be resolvable by using a policy trained on problems in a similar class.

# 5.6 Netlib Linear Programming benchmark

The Netlib Linear Programming benchmark [16] contains 98 challenging real-world problems including supply-chain optimization, scheduling and control problems. As with the QPLIB benchmark, we evaluate results with a general policy trained on the benchmark classes. We solve problems

Table 1: QPLIB problems. Timing results for solving the convex continuous QPs with constraints from QPLIB [14]. The Inst. column is QPLIB's instance number. The columns  $n$  (number of variables),  $m$  (number of constraints), and non-zeros give an indication of the QP's complexity. Best results are bold. A timeout result indicates the solver terminated due to reaching an iteration or time limit (300 s). We hypothesize that the RLQP timeouts are due to the problems being out of distribution from the training data, as the policy here was trained on the benchmark classes.  

<table><tr><td>Inst.</td><td>n</td><td>m</td><td>non-zeros</td><td>OSQP</td><td>RLQP (scalar)</td><td>RLQP (vector)</td></tr><tr><td>8845</td><td>1546</td><td>777</td><td>10999</td><td>6.386</td><td>timeout</td><td>5.435</td></tr><tr><td>9002</td><td>2890</td><td>1649</td><td>12580</td><td>6.000</td><td>timeout</td><td>timeout</td></tr><tr><td>8906</td><td>5223</td><td>838</td><td>20781</td><td>1.108</td><td>1.447</td><td>0.741</td></tr><tr><td>8559</td><td>10000</td><td>5000</td><td>24998</td><td>59.648</td><td>205.372</td><td>24.083</td></tr><tr><td>8938</td><td>4001</td><td>11999</td><td>31997</td><td>timeout</td><td>timeout</td><td>0.991</td></tr><tr><td>8567</td><td>10000</td><td>7500</td><td>32497</td><td>98.511</td><td>284.112</td><td>22.222</td></tr><tr><td>8616</td><td>13870</td><td>10404</td><td>41610</td><td>0.126</td><td>0.113</td><td>0.141</td></tr><tr><td>8515</td><td>16002</td><td>8002</td><td>.56005</td><td>0.105</td><td>timeout</td><td>timeout</td></tr><tr><td>8785</td><td>10399</td><td>11362</td><td>63023</td><td>6.334</td><td>timeout</td><td>2.972</td></tr><tr><td>8495</td><td>27543</td><td>8000</td><td>73029</td><td>1.612</td><td>0.742</td><td>1.174</td></tr><tr><td>8602</td><td>34552</td><td>52983</td><td>242887</td><td>99.872</td><td>timeout</td><td>55.629</td></tr><tr><td>8547</td><td>1003001</td><td>1001000</td><td>6003001</td><td>timeout</td><td>timeout</td><td>timeout</td></tr></table>

to high-accuracy as many of these benchmarks are poorly scaled. Overall, vector formulation of RLQP is  $1.30 \times$  faster than OSQP by the scaled geomean of runtimes. We include a problem-specific breakdown in the supplementary materials.

# 5.7 Maros and Mészáros

In a manner similar to the QPLIB problems, we also benchmark on the Maros and Mészáros repository of QPs. [30]. This collection of 138 QP problems, includes many poorly scaled problems that cause OSQP to fail to converge. We compute the shifted geometric mean for problems solved by both OSQP and RLQP with the general vector policy. RLQP converges faster, with OSQP's shifted geometric mean is 1.829 times that of RLQP. Because the dataset contains 138 problems, a table of the full results is included in the Supplementary Material.

# 6 Limitations

RLQP has limitations. For QPs that converge after few iterations, and thus do not adapt  $\rho$ , having a better adaptation policy is moot. Training RLQP can take a prohibitively long time and require a large replay buffer for some applications, for example, to train the benchmark suite of QPs required several days on a high-end computer with 256 GiB—this may be mitigated to an extent by sharing learned policies between interested practitioners. The time it takes to evaluate the RL policy, especially the vector version, may reduce the performance benefit of faster convergence—this may be mitigated by learning more efficient representations, or by using dedicated neural-network processing hardware.

# 7 Conclusion

We presented RLQP, a method for using reinforcement learning (RL) to speed up the convergence rate of a first-order method quadratic program solver. RLQP uses RL to learn a policy to adapt the internal parameters of the solver to allow for fewer iterations and faster convergence. In experiments, we trained a generic policy and results suggest that a single policy can improve convergence rates for a broad class of problems. We also trained a problem-specific policy, and results suggest that problem-specific training can further speed up convergence rates.

In future work, we will explore whether additional RL policy options can speed up convergence rate further, such as training a hierarchical policy [2] in which the higher-level policy determines the interval between adaptation, performing a neural-architecture search [10], using meta-learning [12, 36] to speed up problem-specific training, and online-learning to adjust the policy at runtime to adapt to changing problems.

# References

[1] Maria-Florina Balcan, Travis Dick, Tuomas Sandholm, and Ellen Vitercik. Learning to branch. In International conference on machine learning, pages 344–353. PMLR, 2018.  
[2] Andrew G Barto and Sridhar Mahadevan. Recent advances in hierarchical reinforcement learning. Discrete event dynamic systems, 13(1):41-77, 2003.  
[3] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.  
[4] Yoshua Bengio, Andrea Lodi, and Antoine Prouvost. Machine learning for combinatorial optimization: a methodological tour d'horizon. European Journal of Operational Research, 2020.  
[5] Dimitris Bertsimas and Bartolomeo Stellato. The voice of optimization. Machine Learning, pages 1-29, 2020.  
[6] Stephen Boyd, Neal Parikh, and Eric Chu. Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc, 2011.  
[7] Xinyun Chen and Yuandong Tian. Learning to perform local rewriting for combinatorial optimization. arXiv preprint arXiv:1810.00337, 2018.  
[8] Andrew R Conn, Nicholas IM Gould, and Philippe L Toint. Trust region methods. SIAM, 2000.  
[9] Hanjun Dai, Elias B Khalil, Yuyu Zhang, Bistra Dilkina, and Le Song. Learning combinatorial optimization algorithms over graphs. arXiv preprint arXiv:1704.01665, 2017.  
[10] Thomas Elsken, Jan Hendrik Metzen, Frank Hutter, et al. Neural architecture search: A survey. J. Mach. Learn. Res., 20(55):1-21, 2019.  
[11] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures, 2018.  
[12] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pages 1126–1135. PMLR, 2017.  
[13] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning, pages 1587-1596. PMLR, 2018.  
[14] Fabio Furini et al. QPLIB: A library of quadratic programming instances. Mathematical Programming Computation, 2018. doi: 10.1007/s12532-018-0147-4. URL https://doi.org/10.1007/s12532-018-0147-4.  
[15] Daniel Gabay and Bertrand Mercier. A dual algorithm for the solution of nonlinear variational problems via finite element approximation. Computers & mathematics with applications, 2(1): 17-40, 1976.  
[16] David M Gay. Electronic mail distribution of linear programming test problems. 1985.  
[17] P. Giselsson and S. Boyd. Linear convergence and metric selection for douglas-rachford splitting and ADMM. IEEE Transactions on Automatic Control, 62(2):532-544, 2017. doi: 10.1109/TAC.2016.2564160.  
[18] Roland Glowinski and A Marroco. Sur l'approximation, par éléments finis d'ordre un, et la résolution, par pénalisation-dualité d'une classe de problèmes de dirichlet non linéaires. ESAIM: Mathematical Modelling and Numerical Analysis-Modélisation Mathématique et Analyse Numérique, 9(R2):41-76, 1975.

[19] Nicholas Gould and Jennifer Scott. A note on performance profiles for benchmarking software. ACM Transactions on Mathematical Software (TOMS), 43(2):1-5, 2016.  
[20] Gurobi Optimization, LLC. Gurobi optimizer. https://www.gurobi.com/. Accessed: 2021-01-28.  
[21] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor, 2018.  
[22] B. S. He, H. Yang, and S. L. Wang. Alternating Direction Method with Self-Adaptive Penalty Parameters for Monotone Variational Inequalities. Journal of Optimization Theory and Applications, 106(2):337–356, August 2000. ISSN 1573-2878. doi: 10.1023/A:1004603514434. URL https://doi.org/10.1023/A:1004603514434.  
[23] Wenlong Huang, Igor Mordatch, and Deepak Pathak. One policy to control them all: Shared modular policies for agent-agnostic control. In International Conference on Machine Learning, pages 4455–4464. PMLR, 2020.  
[24] Jeffrey Ichnowski, Yahav Avigal, Vishal Satish, and Ken Goldberg. Deep learning can accelerate grasp-optimized motion planning. Science Robotics, 5(48), 2020.  
[25] Elias Khalil, Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina. Learning to branch in mixed integer programming. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, 2016.  
[26] Ke Li and Jitendra Malik. Learning to optimize. arXiv preprint arXiv:1606.01885, 2016.  
[27] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
[28] Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In Machine learning proceedings 1994, pages 157-163. Elsevier, 1994.  
[29] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. arXiv preprint arXiv:1706.02275, 2017.  
[30] Istvan Maros and Csaba Mészáros. A repository of convex quadratic programming problems. Optimization Methods and Software, 11(1-4):671-681, 1999.  
[31] Jacob Mattingley and Stephen Boyd. Cvxgen: A code generator for embedded convex optimization. Optimization and Engineering, 13(1):1-27, 2012.  
[32] Luke Metz, Niru Maheswaranathan, Jeremy Nixon, Daniel Freeman, and Jascha Sohl-Dickstein. Understanding and correcting pathologies in the training of learned optimizers. In International Conference on Machine Learning, pages 4556-4565. PMLR, 2019.  
[33] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning, 2013.  
[34] MOSEK ApS. Mosek optimization toolbox. https://www.mosek.com/. Accessed: 2021-01-28.  
[35] Yurii Nesterov and Arkadii Nemirovskii. Interior-point polynomial algorithms in convex programming. SIAM, 1994.  
[36] Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.  
[37] Jorge Nocedal and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006.  
[38] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE transactions on neural networks, 20(1):61-80, 2008.

[39] John Schulman, Jonathan Ho, Alex X Lee, Ibrahim Awwal, Henry Bradlow, and Pieter Abbeel. Finding locally optimal, collision-free trajectories with sequential convex optimization. In Robotics: science and systems, volume 9, pages 1-10. Citeseer, 2013.  
[40] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017.  
[41] David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. Deterministic policy gradient algorithms. In International conference on machine learning, pages 387-395. PMLR, 2014.  
[42] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd. OSQP: an operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4):637-672, 2020. doi: 10.1007/s12532-020-00179-2. URL https://doi.org/10.1007/s12532-020-00179-2.  
[43] Richard S Sutton, David A McAllester, Satinder P Singh, Yishay Mansour, et al. Policy gradient methods for reinforcement learning with function approximation. In NIPs, volume 99, pages 1057-1063. CiteSeer, 1999.  
[44] Kaixuan Wei, Angelica Aviles-Rivero, Jingwei Liang, Ying Fu, Carola-Bibiane Schonlieb, and Hua Huang. Tuning-free plug-and-play proximal algorithm for inverse imaging problems. In International Conference on Machine Learning, pages 10158-10169. PMLR, 2020.  
[45] Philip Wolfe. The simplex method for quadratic programming. Econometrica: Journal of the Econometric Society, pages 382-398, 1959.
