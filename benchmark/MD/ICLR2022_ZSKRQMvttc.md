# ACCELERATED POLICY LEARNING WITH PARALLEL DIFFERENTIABLE SIMULATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep reinforcement learning can generate complex control policies, but requires large amounts of training data to work effectively. Recent work has attempted to address this issue by leveraging differentiable simulators. However, inherent problems such as local minima and exploding/vanishing numerical gradients prevent these methods from being generally applied to control tasks with complex contact-rich dynamics, such as humanoid locomotion in classical RL benchmarks. In this work, we present SHAC, a short-horizon actor-critic method that successfully leverages parallel differentiable simulation to accelerate policy learning. Our method alleviates problems with local minima through a smooth critic function, avoids vanishing/exploding gradients through a truncated learning window, and allows many physical environments to be run in parallel. We evaluate our method on classical RL control tasks, and show substantial improvements in sample efficiency and wall-clock time over state-of-the-art RL and differentiable simulation-based algorithms. In addition, we demonstrate the scalability of our method by applying it to the challenging high-dimensional problem of muscle-actuated locomotion with a large action space, achieving a greater than  $17 \times$  reduction in training time over the best-performing established RL algorithm. More visual results are provided at: https://sites.google.com/view/shac.

# 1 INTRODUCTION

Learning control policies is an important task in robotics and computer animation. Among various policy learning techniques, reinforcement learning (RL) has been a particularly successful tool to learn policies for systems ranging from robots (e.g., Cheetah, Shadow Hand) to complex animation characters (e.g., muscle-actuated humanoids) using only high-level reward definitions. Despite this success, RL requires large amounts of training data to approximate the policy gradient, making learning expensive and time-consuming, especially for high-dimensional problems (Figure 1, Right). The recent development of differentiable simulators opens up new possibilities for accelerating the learning and optimization of control policies. A differentiable simulator provides accurate first-order gradients of the task performance reward with respect to the control inputs. Such additional information potentially allows the use of efficient gradient-based methods to optimize policies. However, despite the availability of differentiable simulators, it has not yet been convincingly demonstrated that they can effectively accelerate policy learning in complex high-dementional and contact-rich tasks, such as some traditional RL benchmarks. There are several reasons for this:

1. Local minima may cause gradient-based optimization methods to stall.  
2. Numerical gradients may vanish/explode along the backward path for long trajectories.  
3. Discontinuous optimization landscapes can occur during policy failures/early termination.

Because of these challenges, previous work has been limited to the optimization of open-loop control policies with short horizons (Hu et al., 2018; Huang et al., 2021), or the optimization of policies for relatively simple tasks (e.g., contact-free environments) (Mora et al., 2021; Du et al., 2021). In this work, we explore the question: Can differentiable simulation accelerate policy learning in tasks with continuous closed-loop control and contact-rich dynamics?

Inspired by actor-critic RL algorithms (Konda & Tsitsiklis, 2000), we propose an approach to effectively leverage differentiable simulators for policy learning. We alleviate the problem of local minima by using a critic network that acts as a smooth surrogate to approximate the underlying

![](images/88fbc04093497549881728f5be9a0991c07dcf86a5c4566184c54f9b36b57c65.jpg)  
Figure 1: Environments: We compare to three classical physical control RL benchmarks of increasing difficulty, from left: Cartpole Swing Up + Balance, Ant, and Humanoid. In addition, we train the high-dimensional muscle-tendon driven Humanoid MTU model from Lee et al. (2019). Whereas model-free reinforcement learning (PPO, SAC) needs many samples for such high-dimensional control problems, SHAC scales efficiently through the use of analytic gradients from differentiable simulation with a parallelized implementation, both in sample complexity and wall-clock time.

![](images/541006bc4926618e83eedbfe6ddbccfb80f78fba50cd1dc16f1d6c8f3a0ea53a.jpg)

![](images/569e77177ba3248ef69ace8e782bbbe6273aa9b8f39225af3694a6cede518f30.jpg)

![](images/6e28cf65b36d8d800102432764da18a2a3b44a35334a0871a4ed423d8f4f4567.jpg)

noisy reward landscape (Figure 2). In addition, we propose a truncated learning window to shorten the backpropagation path to address problems with vanishing/exploding gradients and reduce memory requirements. Finally, the adopted terminal critic function allows us to support early termination, improving the learning efficiency when a policy failure occurs.

A further challenge with differentiable simulators is that the backward pass typically introduces some computational overhead compared to optimized forward-dynamics physics engines. To ensure meaningful comparisons, we must ensure that our learning method not only improves sample-efficiency, but also wall-clock time. GPU-based physics simulation has shown remarkable effectiveness for accelerating model-free RL algorithms (Liang et al., 2018; Allshire et al., 2021), and we design our method to effectively leverage parallel differentiable simulation using GPUs. To the best of our knowledge, this work is the first to provide a fair and comprehensive comparison between gradient-based and RL-based methods, where fairness is defined as (a) benchmarking on both RL-favored tasks and differentiable-simulation-favored tasks, (b) testing complex tasks (i.e., contact-rich tasks with long horizons), (c) comparing to the state-of-the-art implementation of RL algorithms, and (d) comparing both sample efficiency and wall-clock time.

We evaluate our method on standard RL benchmark tasks, as well as a high-dimensional character control task with over 150 actuated degrees of freedom (Figure 1). We refer to our method as Short Horizon Actor Critic (SHAC), and our experiments show that SHAC outperforms state-of-the-art policy learning methods in both sample-efficiency and wall-clock time.

# 2 RELATED WORK

Differentiable Simulation Physics-based simulation has been widely used in the robotics field (Todorov et al., 2012; Coumans & Bai, 2016). More recently, there has been interest in the construction of differentiable simulators, which directly compute the gradients of simulation outputs with respect to actions and initial conditions. These simulators may be based on auto-differentiation frameworks (Griewank & Walther, 2003; Heiden et al., 2021; Freeman et al., 2021) or analytic gradient calculation (Carpentier & Mansard, 2018; Geilinger et al., 2020; Werling et al., 2021). One challenge for differentiable simulation is the non-smoothness of contact dynamics, leading many works to focus on how to efficiently differentiate through linear complementarity (LCP) models of contact (Degrave et al., 2016; de Avila Belbute-Peres et al., 2018; Werling et al., 2021) or leverage a smooth penalty-based contact formulation (Geilinger et al., 2020; Xu et al., 2021).

Deep Reinforcement Learning Deep reinforcement learning has become a prevalent tool for learning control policies for systems ranging from robots (Hwangbo et al., 2019; OpenAI et al., 2019; Lee et al., 2020; Andrychowicz et al., 2020), to complex animation characters (Peng et al., 2018; 2021; Liu & Hodgins, 2018; Lee et al., 2019). Model-free RL algorithms treat the underlying dynamics as a black box in the policy learning process. Among them, on-policy RL approaches (Schulman et al., 2015; 2017) improve the policy from the experience generated by the current policy, while off-policy methods (Lillicrap et al., 2016; Mnih et al., 2016; Fujimoto et al., 2018; Haarnoja et al., 2018) leverage all the past experience as a learning resource to improve sample efficiency. On the other side, model-based RL methods (Kurutach et al., 2018; Janner et al., 2019; Hafner et al., 2019)

have been proposed to learn an approximated dynamics model from little experience and then fully exploit the learned dynamics model during policy learning.

Differentiable Simulation based Policy Learning The recent development of differentiable simulators opens up new possibilities to optimize control policies via the provided gradient information. Backpropagation Through Time (BPTT) (Mozer, 1995) has been widely used in previous work to showcase differentiable systems (Hu et al., 2018; 2020; Liang et al., 2019; Huang et al., 2021; Du et al., 2021). However, the noisy optimization landscape and exploding/vanishing gradients in long-horizon tasks make such straightforward first-order methods ineffective. A few works have been proposed recently to solve this issue. Qiao et al. (2021) present a sample enhancement method to increase RL sample-efficiency for the simple MuJoCo Ant environment. However, as the method follows a model-based learning framework, it is significantly slower than state-of-the-art on-policy methods such as PPO (Schulman et al., 2017). Mora et al. (2021) propose to interleave a trajectory optimization stage and an imitation learning stage to detach the policy from the computation graph so as to alleviate the exploding gradient problem. They demonstrate their methods on simple control tasks (e.g., stopping a pendulum). However, gradients flowing back through long trajectories of states can still create challenging optimization landscapes for more complex tasks. Furthermore, both methods (Mora et al., 2021; Qiao et al., 2021) require the full simulation Jacobian, which is not commonly or efficiently available in reverse-mode differentiable simulators. In contrast, our method relies only on first-order gradients. Therefore, it can naturally leverage improvements in simulators and frameworks that can provide this information.

# 3 METHOD

# 3.1 DIFFERENTIABLE DYNAMICS SIMULATION

Conceptually, we treat the simulator as an abstract function  $\mathbf{s}_{t + 1} = \mathcal{F}(\mathbf{s}_t,\mathbf{a}_t)$  that takes a state  $\mathbf{s}$  from a time  $t\to t + 1$  , where  $\mathbf{a}$  is a vector of actuation controls applied during that time-step (may represent joint torques, or muscle contraction signals depending on the problem). Given a differentiable scalar loss function  $\mathcal{L}$  , and its adjoint  $\mathcal{L}^{*} = \frac{\partial\mathcal{L}}{\partial\mathbf{s}_{t + 1}}$  , the simulator backward pass computes:

$$
\frac {\partial \mathcal {L}}{\partial \mathbf {s} _ {t}} = \left(\frac {\partial \mathcal {F}}{\partial \mathbf {s} _ {t}}\right) ^ {T} \left(\frac {\partial \mathcal {L}}{\partial \mathbf {s} _ {t + 1}}\right), \quad \frac {\partial \mathcal {L}}{\partial \mathbf {a} _ {t}} = \left(\frac {\partial \mathcal {F}}{\partial \mathbf {a} _ {t}}\right) ^ {T} \left(\frac {\partial \mathcal {L}}{\partial \mathbf {s} _ {t + 1}}\right) \tag {1}
$$

Concatenating these steps allows us to propagate gradients through an entire trajectory. Although our actor-critic method is designed to leverage efficient parallel simulation, it is compatible with any differentiable simulator that can compute these gradients.

We build our differentiable simulator on PyTorch (Paszke et al., 2019) and use a source-code transformation approach to generate forward and backward versions of our simulation kernels (Griewank & Walther, 2003; Hu et al., 2020). To compute articulation dynamics, we use the composite rigid body algorithm (CRBA), which uses dense matrix routines to build system matrices (Featherstone, 2014). We perform a dense Cholesky decomposition to obtain joint accelerations, before final semi-implicit Euler integration steps. The decomposition result may be cached so that it does not need to be repeated in the backward pass. We parallelize the simulator over environments using distributed GPU kernels for the dense matrix routines and evaluation of contact and joint forces.

Analytic articulated dynamics simulation can be non-smooth and even discontinuous when contact and joint limits are introduced, and special care must be taken to ensure smooth dynamics. To model contact, we use the frictional contact model from Geilinger et al. (2020), which approximates Coulomb friction with a linear step function. In addition, we incorporate the contact damping force formulation from Xu et al. (2021) into our dynamics model to provide better smoothness of the non-interpenetration contact dynamics. To model joint limits, a continuous penalty-based force is applied instead of enforcing limits as hard constraints. We provide more simulation details in Appendix A.1.

# 3.2 OPTIMIZATION LANDSCAPE ANALYSIS

Although smoothed physical models improve the local optimization landscape, the combination of forward dynamics and the neural network control policy renders each simulation step non-linear and non-convex. This problem is exacerbated when thousands of simulation steps are concatenated and the actions in each step are coupled by a feedback control policy. The complexity of the resulting reward landscape leads simple gradient-based methods to easily become trapped in local minima.

![](images/9ecbde3aea6cda97160420b273e315c6d4d11593502e87fd807b48138ddb4f21.jpg)  
Figure 2: Landscape comparison between BPTT and SHAC. We select one single weight from a policy and change its value by  $\Delta \theta_{k} \in [-1,1]$  to plot the task loss landscapes of BPTT and SHAC w.r.t. one policy parameter. The task horizon is  $H = 1000$  for BPTT, and the short horizon length for our method is  $h = 32$ . As we can see, longer optimization horizons lead to noisy loss landscape that are difficult to optimize, and the landscape of our method can be regarded as a smooth approximation of the real landscape.

![](images/0ae51a44ac91537830f9d77f5ab4ef67efaf69381c2d9a74f663e4e23eabd4fc.jpg)

Furthermore, to handle agent failure (e.g., a humanoid falling down) and improve sample efficiency, early termination techniques are widely used in policy learning algorithms (Brockman et al., 2016). Although these have proven effective for model-free algorithms, early termination introduces additional discontinuities to the optimization problem, which makes methods based on analytical gradients less successful.

To analyze this problem, we plot the optimization landscape in Figure 2 (Left) for a humanoid locomotion problem with a 1000-step horizon. Specifically, we take a trained policy, perturb the value of a single parameter  $\theta_{k}$  in the neural network, and re-evaluate performance for the policy variations. As shown in the figure, with long task horizons and early termination, the landscape of the humanoid problem is highly non-convex and discontinuous. In addition, the norm of the gradient  $\frac{\partial\mathcal{L}}{\partial\theta}$  computed from backpropagation is larger than  $10^{6}$ . Thus, most previous works based on differentiable simulation focus on short-horizon tasks with contact-free dynamics and no early termination, where pure gradient-based optimization (e.g., BPTT) can optimize the policy successfully.

# 3.3 SHORT-HORIZON ACTOR-CRITIC (SHAC)

To resolve the aforementioned issues of gradient-based policy learning, we propose the Short-Horizon Actor-Critic method (SHAC). Our method concurrently learns a policy network (i.e., actor)  $\pi_{\theta}$  and a value network (i.e., critic)  $V_{\phi}$  during task execution, and splits the entire task horizon into several sub-windows of smaller horizons (Figure 3). A multi-step reward in the sub-window plus a terminal value estimation from the learned critic is used to improve the policy network. The differentiable simulation is used to backpropagate the gradient through the states and actions inside the sub-windows to provide an accurate policy gradient. The trajectory rollouts are then collected and used to learn the critic network in each learning episode.

Specifically, we model each of our control problems as a finite-horizon Markov decision process (MDP) with state space  $S$ , action space  $\mathcal{A}$ , reward function  $\mathcal{R}$ , transition function  $\mathcal{F}$ , initial state distribution  $\mathcal{D}_{\mathbf{s}_0}$ , and horizon  $H$ . Each trajectory starts with an initial state sampled from the distribution  $\mathbf{s}_0 \sim \mathcal{D}_{\mathbf{s}_0}$ . At each step, an action vector  $\mathbf{a}_t$  is computed by a feedback policy  $\pi_{\theta}(\mathbf{a}_t | \mathbf{s}_t)$ . The transition function  $\mathcal{F}$  is modeled by our differentiable simulation as mentioned in Section 3.1 and defines the next state given the current state and action,  $\mathbf{s}_{t+1} \gets \mathcal{F}(\mathbf{s}_t, \mathbf{a}_t)$ . A single-step reward  $r_t = \mathcal{R}(\mathbf{s}_t, \mathbf{a}_t)$  is received at each step. The goal of the problem is then to find the policy parameters  $\theta$  that maximize the expected finite-horizon reward:

$$
J (\theta) = \underset {\pi_ {\theta}, \mathcal {D} _ {\mathbf {s} _ {0}}} {\mathbb {E}} \left[ \sum_ {t = 0} ^ {H - 1} r _ {t} \right] = \underset {\pi_ {\theta}, \mathcal {D} _ {\mathbf {s} _ {0}}} {\mathbb {E}} \left[ \sum_ {t = 0} ^ {H - 1} \mathcal {R} \left(\mathbf {s} _ {t}, \mathbf {a} _ {t}\right) \right]. \tag {2}
$$

Although our method does not constrain the policy to be deterministic or stochastic, we choose to use the stochastic policy in our experiments for the extra exploration that it provides. Specifically, the action at step  $t$  is sampled by

$$
\mathbf {a} _ {t} \sim \mathcal {N} (\mu_ {\theta} (\mathbf {s} _ {t}), \sigma_ {\theta} (\mathbf {s} _ {t})). \tag {3}
$$

![](images/5ef463843e9dc6d5afb74866bf002cb2f118119a431be14154ce2bc9a50cb252.jpg)  
Episode of horizon length  $H$

![](images/b8c91e60b1d1fc9ad2eafdcf517f60f72c9ad7ef05e88da4e22d02b3684e476b.jpg)  
Episode of short horizon length  $h$

![](images/6e1846bb87f6897fdbfd6ccb5e6a4c8d3e62143235dd1e628a99b24bcbbe4c4b.jpg)  
Figure 3: Computation graph of BPTT and SHAC. Top: BPTT propagates gradients through an entire trajectory in each episode. This leads to noisy loss landscapes, increased memory, and numerical gradient problems. Bottom: SHAC subdivides the trajectory into short optimization windows. This makes the reward function smoother and reduces memory requirements, providing a way to sample many trajectories in parallel. The environment is reset once an early termination happens. The solid arrows are gradient-preserving computations, while the dashed arrows denote that locations at which the gradients are cut off.  
Episode of short horizon length  $h$

In each learning episode, we sample  $N$  trajectories  $\{\tau_i\}$  of short horizon  $h \ll H$  in parallel from the simulation. The following policy loss is then computed:

$$
\mathcal {L} _ {\theta} = - \frac {1}{N h} \sum_ {i = 1} ^ {N} \left[ \left(\sum_ {t = t _ {0}} ^ {t _ {0} + h - 1} \gamma^ {t - t _ {0}} \mathcal {R} \left(\mathbf {s} _ {t} ^ {i}, \mathbf {a} _ {t} ^ {i}\right)\right) + \gamma^ {h} V _ {\phi} \left(\mathbf {s} _ {t _ {0} + h} ^ {i}\right) \right], \tag {4}
$$

where  $\mathbf{s}_t^i$  and  $\mathbf{a}_t^i$  are the state and actions at step  $t$  of the  $i$ -th trajectory, and  $\gamma < 1$  is a discount factor introduced to stabilize the training. Special handling such as resetting the discount ratio is conducted when task termination happens during sampling.

To compute the gradient of the policy loss  $\frac{\partial\mathcal{L}_{\theta}}{\partial\theta}$ , we treat the simulator as a differentiable layer (with backward pass shown in Eq. 1) in the PyTorch computation graph and perform regular backpropagation. For details of gradient computation, please see Appendix A.2. Our algorithm then updates the policy using one step of a gradient-based solver, e.g., Adam (Kingma & Ba, 2014). The differentiable simulator plays an important role here, as it allows us to fully utilize the underlying dynamics linking states and actions, and to optimize the policy, producing better short-horizon reward inside the trajectory, and also resulting in a more promising terminal state for the sake of long-term performance. We note that the trajectories sampled in the next episode will continue from the end state of the previous episode. However, the gradients are cut off between episodes to prevent unstable gradients during long-horizon backpropagation.

After we update the policy  $\pi_{\theta}$ , we use the trajectories collected in the current episode to train the value function  $V_{\phi}$ . The value function network is trained by the following MSE loss:

$$
\mathcal {L} _ {\phi} = \underset {\mathbf {s} \in \left\{\tau_ {i} \right\}} {\mathbb {E}} \left[ \left\| V _ {\phi} (\mathbf {s}) - \tilde {V} (\mathbf {s}) \right\| ^ {2} \right], \tag {5}
$$

where  $\tilde{V} (\mathbf{s})$  is the estimated value of state s, and is computed from the sampled short-horizon trajectories through a td-  $\lambda$  formulation (Sutton et al., 1998), which computes the estimated value by exponentially averaging different  $k$  -step returns to balance the variance and bias of the estimation:

$$
\begin{array}{l} \tilde {V} (\mathbf {s} _ {t}) = (1 - \lambda) \sum_ {t ^ {\prime} = t} ^ {h - 2} \lambda^ {t ^ {\prime} - t} \left[ \left(\sum_ {k = t} ^ {t ^ {\prime}} \gamma^ {k - t} r _ {k}\right) + \gamma^ {t ^ {\prime} - t + 1} V _ {\phi} (\mathbf {s} _ {t ^ {\prime} + 1}) \right] \\ + \lambda^ {h - t - 1} \left[ \left(\sum_ {k = t} ^ {h - 1} \gamma^ {k - t} r _ {k}\right) + \gamma^ {h - t} V _ {\phi} (\mathbf {s} _ {h}) \right]. \tag {6} \\ \end{array}
$$

Algorithm 1: Short-Horizon Actor-Critic Policy Learning  
Initialize policy  $\pi_{\theta}$  , value function  $V_{\phi}$  , and target value function  $V_{\phi^{\prime}}\gets V_{\phi}$    
for episode  $\leftarrow 1,2,\ldots ,M$  do Sample  $N$  trajectories of length  $h$  by the parallel differentiable simulation. Compute the policy loss  $\mathcal{L}_{\theta}$  defined in Eq. 4 from the sampled trajectories and  $V_{\phi^{\prime}}$  . Compute the analytical gradient  $\frac{\partial\mathcal{L}_{\theta}}{\partial\theta}$  and update the policy  $\pi_{\theta}$  one step with Adam. Compute estimated values for all the states in sampled trajectories with Eq. 6. Fit the value function  $V_{\phi}$  using the critic loss defined in Eq. 5. Update target value function:  $V_{\phi^{\prime}}\gets \alpha V_{\phi^{\prime}} + (1 - \alpha)V_{\phi}$    
end for

The estimated value  $\tilde{V} (\mathbf{s})$  is treated as constant during critic training, as in regular actor-critic RL methods. In other words, the gradient of Eq. 5 does not flow through the states and actions in Eq. 6.

We further utilize the target value function technique (Mnih et al., 2015) to stabilize the training by smoothly transitioning from the previous value function to the newly fitted one, and use the target value function  $V_{\phi'}$  to compute the policy loss (Eq. 4) and to estimate state values (Eq. 6). In addition, we apply state normalization as is common in RL algorithms. The pseudo code of our method is provided in Algorithm 1.

Our actor-critic formulation has several advantages that enable it to leverage simulation gradients effectively and efficiently. First, the terminal value function absorbs the discontinuity of long dynamics horizons and early termination into a smooth function, as shown in Figure 2 (Right). This smooth surrogate formulation helps reduce the number of local spikes and alleviates the problem of easily getting stuck in local optima. Second, the short horizon trajectories avoid numerical problems with backpropagating the gradient through deeply nested update chains. Finally, the use of short horizons allows us to update the actor more frequently, which, when combined with parallel differentiable simulation, results in a significant speed up of training time.

# 4 EXPERIMENTS

We design experiments to investigate five questions: (1) How does our method compare to the state-of-the-art RL algorithms on classical RL control tasks, in terms of both sample efficiency and wall-clock time efficiency? (2) How does our method compare to the previous differentiable simulation-based policy learning methods? (3) Does our method scale to high-dimensional problems? (4) Is the terminal critic necessary? (5) How important is the choice of short horizon length  $h$  for our method?

# 4.1 EXPERIMENT SETUP

To ensure a fair comparison for wall-clock time performance, we run all algorithms on the same GPU model (TITAN X) and CPU model (Intel Xeon(R) E5-2620). Furthermore, we conduct hyperparameter searches for all algorithms and report the performance of the best hyperparameters for each problem. In addition, we report the performance averaged from five individual runs for each algorithm on each problem. The details of the experimental setup are provided in Appendix A.3.

# 4.2 BENCHMARK CONTROL PROBLEMS

For comprehensive evaluations, we select four broad control tasks, including three classical RL tasks across different complexity levels, as well as one high-dimensional control task with a large action space. All tasks have stochastic initial states to further improve the robustness of the learned policy.

Classical Tasks: We select CartPole Swing Up, Ant and Humanoid as three representative RL tasks, as shown in Figure 1. Their difficulty spans from the simplest contact-free dynamics (CartPole Swing Up), to complex contact-rich dynamics (Humanoid). For CartPole Swing Up, we use  $H = 240$  as the task horizon, whereas the other tasks use horizons of  $H = 1000$ .

Humanoid MTU: To assess how our method scales to high-dimensional tasks, we examine the challenging problem of muscle-actuated humanoid control (Figure 1, Right). We use the lower body of the humanoid model from Lee et al. (2019), which contains 152 muscle-tendon units (MTUs). Each MTU contributes one actuated degree of freedom that controls the contractile force applied to the attachment sites on the connected bodies. The task horizon for this problem is  $H = 1000$ .

![](images/fe49628c7b76566075b272cc85966ab86acf659132a8532041553cdb62dee6e6.jpg)  
Figure 4: Learning curves comparison on four benchmark problems. Each column corresponds to a particular problem, with the top plot evaluating sample efficiency and the bottom plot evaluating wall-clock time efficiency. For better visualization, we truncate all the curves up to the maximal simulation steps/wall-clock time of our method (except for Humanoid MTU), and we provide the full plots in Appendix A.5. Each curve is averaged from five random seeds, and the shaded area shows the standard deviation. SHAC is more sample efficient than all baselines. Model-free baselines are competitive on wall-clock time on pedagogical environments such as the cartpole, but are much less effective as the problem complexity scales.

To be compatible with differentiable simulation, the reward formulations of each problem are defined as differentiable functions. The details of each task are provided in Appendix A.4.

# 4.3 RESULTS

Comparison to model-free RL. We compare SHAC with Proximal Policy Optimization (PPO) (Schulman et al., 2017) (on-policy) & Soft Actor-Critic (SAC) (Haarnoja et al., 2018) (off-policy). We select these two baseline algorithms since PPO is the state-of-the-art on-policy method with high wall-clock training time efficiency, and SAC is one of the most advanced off-policy methods with high sample efficiency. We use high-performance implementations of both methods available in RL games (Makoviichuk & Makoviychuk, 2021). To achieve state-of-the-art performance, we follow the approach proposed by Makoviychuk et al. (2021), where all simulation, reward and observation data remain on the GPU and are shared as PyTorch tensors between the RL algorithm and the parallel simulator. Both PPO and SAC implementations are parallelized and operate on vectorized states and actions. With PPO we used short episode lengths, an adaptive learning rate, and large mini-batch sizes during training to achieve the best possible performance.

As shown in the first row of Figure 4, our method shows significant improvements in sample efficiency over PPO and SAC in three classical RL problems, especially when the dimension of the problem increases (e.g., Humanoid). The analytical gradients provided by the differentiable simulation allow us to efficiently acquire the expected policy gradient through a small number of samples. In contrast, PPO and SAC have to collect many Monte-Carlo samples to estimate the policy gradient.

Model-free algorithms typically have a lower per-iteration cost than methods based on differentiable simulation; thus, it makes sense to also evaluate wall-clock time efficiency instead of sample-efficiency alone. As shown in the second row of Figure 4, the wall-clock time performance of PPO, SAC, and our method are much closer than the sample efficiency plot. Interestingly, the training speed of our method is slower than PPO at the start of training. We hypothesize that the target value network in our method is far from the real value network initially, requiring sufficient episodes to warm up. We also observe that our method consistently achieves better policies than RL methods in all problems. We hypothesize that, while RL methods are effective at exploration far from the solution, they struggle to estimate the policy gradient with sufficient accuracy near the optimum point, especially in complex problems. We also note that the backward time for our simulation is consistently around  $2 \times$  that of the forward pass. This indicates that our method still has room to improve its overall wall-clock time efficiency through the development of more optimized differentiable simulators with fast backward gradient computation. We provide a detailed timing breakdown of our method in Appendix A.5.

![](images/df3f251cbf1a5c9b246c04faae149c808ca306dd83c55506cd995719816d6678.jpg)  
Figure 5: Humanoid MTU: A sequence of frames from a learned running gait. The muscle unit color indicates the activation level at the current state.

![](images/882c63ad66fd38428ecd86587d048a4aa2d2ea341fde6bcce5605a1926890888.jpg)

![](images/f89132e754bcc0ecfac0ec1a8c40a0a8d4da986b22efa6e6693b77a2f7b2b0bc.jpg)

![](images/dc713b724278ebdd1e84db4a881f95d7eb6a60c5d049b624da8a75583790c567.jpg)

![](images/0b1cac800d776afa5190a56cc620d740742931f1b9e2dcdf72358b7011f5b818.jpg)

Comparison with previous gradient-based methods. We compare our approach to three gradient-based learning methods: (1) Backpropagation Through Time (BPTT), which has been widely used in the differentiable simulation literature (Hu et al., 2018; Du et al., 2021), (2) PODS (Mora et al., 2021), and (3) Sample Enhanced Model-based Policy Optimization (SE-MBPO) (Qiao et al., 2021).

BPTT: The original BPTT method backpropagates gradients over the entire trajectory, which results in exploding gradients as shown in Section 3.2. We modify BPTT to work on a shorter window of the tasks (64 steps for CartPole and 128 steps for other tasks), and also leverage parallel differentiable simulation to sample multiple trajectories concurrently to improve its time efficiency. As shown in Figure 4, BPTT successfully optimizes the policy for the contact-free CartPole Swing Up task, whereas it falls into local minima quickly in all other tasks involving contact. For example, the policy that BPTT learns for Ant is a stationary position leaning forward, which is a local minimum.

PODS: We compare to the first-order version of PODS, as the second-order version requires the full Jacobian of the state with respect to the whole action trajectory, which is typically unavailable in a reverse-mode differentiable simulator (including ours). Since PODS relies on a trajectory optimization step to optimize an open-loop action sequence, it is not clear how to accommodate early termination where the trajectory length can vary during optimization. Therefore, we test PODS performance only on the CartPole Swing Up problem. As shown in Figure 4, PODS quickly converges to a local optimum and is unable to improve further. This is because PODS is designed to be a method with high gradient exploitation but little exploration. Specifically, the line search applied in the trajectory optimization stage helps it converge quickly, but also prevents it from exploring more surrounding space. Furthermore, the extra simulation calls introduced by the line search and the slow imitation learning stage make it less competitive in either sample or wall-clock time efficiency.

SE-MBPO: Qiao et al. (2021) propose to improve a model-based RL method MBPO (Janner et al., 2019) by augmenting the rollout samples using data augmentation that relies on the Jacobian from the differentiable simulator. Although SE-MBPO shows high sample efficiency, the underlying model-based RL algorithm and off-policy training style cause their method to have a higher wall-clock time. As a comparison, the officially released code for SE-MBPO takes 8 hours to achieve a reasonable policy in the Ant problem used by Qiao et al. (2021), whereas our algorithm takes less than 15 minutes to acquire a policy with the same gait level in our Ant problem. Aiming for a more fair comparison, we adapt their implementation to work on our Ant problem in our simulator. However, we found that it could not successfully optimize the policy even after considerable hyperparameter tuning. Regardless, the difference in wall-clock time between two algorithms is obvious, and the training time of SE-MBPO is unlikely to be improved significantly by integrating it into our simulation environment. Furthermore, as suggested by Qiao et al. (2021), SE-MBPO does not generalize well to other tasks, whereas our method can be successfully applied to various complexity levels of tasks.

Scalability to high-dimensional problems. We test our algorithm and RL baselines on the Humanoid MTU example to compare their scalability to high-dimensional problems. With the large 152-dimensional action space, both PPO and SAC struggle to learn the policy as shown in Figure 4 (Right). Specifically, PPO and SAC learn significantly worse policies after more than 10 hours of training and with hundreds of millions of samples. This is because the amount of data required to accurately estimate the policy gradient significantly increases as the state and action spaces become large. In contrast, our method scales well due to direct access to the true gradients from differentiable simulation. To achieve the same reward level as PPO, our approach only takes around 35

minutes of training and 1.7M simulation steps. This results in over  $17\times$  and  $30\times$  wall-clock time improvement over PPO and SAC, respectively, and  $382\times$  and  $170\times$  more sample efficiency. Furthermore, after training for only 1.5 hours, our method is able to find a policy that has twice the reward of the best-performing policy from the RL methods. A learned running gait is visualized in Figure 5. Such scalability to high-dimensional control problems opens up new possibilities for applying differentiable simulation in computer animation, where complex character models are widely used to provide more natural motion.

Ablation study on the terminal critic. We introduce a terminal critic value in Eq. 4 to account for the long-term performance of the policy after the short horizon. In this experiment, we evaluate the importance of this term. By removing the terminal critic from Eq. 4, we get an algorithmic equivalent to BPTT with a short horizon window and discounted reward calculation. We apply this no-critic variation on all four problems and plot the training curve in Figure 4, denoted by "No Critic." Without a terminal critic function, the algorithm is not able to learn a reasonable policy, as it only optimizes a short-horizon reward of the policy regardless of its long-term behavior.

Ablation study on short horizon length  $h$ . The choice of horizon length  $h$  is important for the performance of our method. The horizon length cannot be too small, as it will result in worse value estimation by td-λ (Eq. 6) and underutilize the power of the differentiable simulator to predict the sensitivity of future performance to the policy weights. On the other hand, a horizon length that is too long will lead to a noisy optimization landscape and less-frequent policy updates. Empirically, we find that a short horizon length  $h = 32$  with  $N = 64$  parallel trajectories generally works well for all four tasks in our experiments. We conduct a hyperparameter study of short horizon length on the Ant task to show the influence of this hyperparameter. We run our algorithm with six variations of hyperparameters  $h = 4, 8, 16, 32, 64, 128$ . We set the corresponding number of parallel trajectories  $N = 512, 256, 128, 64, 32, 16$  for the variant, such that each one generates the same amount of samples in an episode. We run each variant for the same number of episodes  $M = 2000$  with 5 individual random seeds. In Figure 6, we report the average

![](images/936433e1e3fce0f43fc1d736e7713bad4538018140c63beac3f380298728b95f.jpg)  
Figure 6: Ablation study for short horizon length  $h$  on Ant problem.

A small  $h$  results in worse value estimation. A too large  $h$  leads to an ill-posed optimization landscape and longer training time.

reward of the best policies from 5 runs for each variant, as well as the total training time. As expected, the best reward is achieved when  $h = 16$  or 32, and the training time scales linearly as  $h$  increases.

# 5 CONCLUSION AND FUTURE WORK

In this work, we propose an approach to effectively leverage differentiable simulation for policy learning. At the core of our method is the use of a critic network that acts as a smooth surrogate to approximate the underlying noisy optimization landscape. In addition, a truncated learning window is adopted to alleviate the problem of exploding/vanishing gradients during deep backward paths. Equipped with parallel differentiable simulation, our method shows significantly higher sample efficiency and wall-clock time efficiency over state-of-the-art RL and gradient-based methods, especially when the problem complexity increases. As shown in our experiments, model-free methods demonstrate efficient learning at the start of training, but SHAC is able to achieve superior performance after a sufficient number of episodes. A compelling future direction for research is how to combine model-free methods with our gradient-based method in order to leverage the strengths of both. Furthermore, in our method, we use a fixed and predetermined short horizon length  $h$  throughout the learning process; however, future work may focus on implementing an adaptive short horizon schedule that varies with the status of the optimization landscape.

# REPRODUCIBILITY

To aid reproducibility, we provide the details of the simulation implementation in the manuscript and appendix. The pseudo code along with detailed mathematical equations of our algorithm are presented in Section 3.3. We also report all the hyperparameters used in each experiment in Appendix A.3. Furthermore, we provide the details of the reward function definition of each problem including the coefficient of each reward term in Appendix A.4. We plan to open source our code including the parallel differentiable simulation and the policy learning implementation upon the conclusion of double-blind review process.

# REFERENCES

Arthur Allshire, Mayank Mittal, Varun Lodaya, Viktor Makoviychuk, Denys Makoviichuk, Felix Widmaier, Manuel Wuthrich, Stefan Bauer, Ankur Handa, and Animesh Garg. Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger. arXiv preprint arXiv:2108.09779, 2021.  
OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chogiej, Rafal Józefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, Jonas Schneider, Szymon Sidor, Josh Tobin, Peter Welinder, Lilian Weng, and Wojciech Zaremba. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3-20, 2020. doi: 10.1177/0278364919887447. URL https://doi.org/10.1177/0278364919887447.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.  
Justin Carpentier and Nicolas Mansard. Analytical derivatives of rigid body dynamics algorithms. In Robotics: Science and systems (RSS 2018), 2018.  
Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. 2016.  
Filipe de Avila Belbute-Peres, Kevin Smith, Kelsey Allen, Josh Tenenbaum, and J. Zico Kolter. End-to-end differentiable physics for learning and control. 2018.  
Jonas Degrave, Michiel Hermans, Joni Dambre, and Francis Wyffels. A differentiable physics engine for deep learning in robotics. 2016.  
Tao Du, Kui Wu, Pingchuan Ma, Sebastien Wah, Andrew Spielberg, Daniela Rus, and Wojciech Matusik. Diffpd: Differentiable projective dynamics with contact. arXiv preprint arXiv:2101.05917, 2021.  
Roy Featherstone. Rigid body dynamics algorithms. Springer, 2014.  
C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax - a differentiable physics engine for large scale rigid body simulation, 2021.  
Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning, pp. 1587-1596. PMLR, 2018.  
Moritz Geilinger, David Hahn, Jonas Zehnder, Moritz Bächer, Bernhard Thomaszewski, and Stelian Coros. Add: Analytically differentiable dynamics for multi-body systems with frictional contact. In arXiv, 2020.  
Andreas Griewank and Andrea Walther. Introduction to automatic differentiation. PAMM, 2(1): 45-49, 2003.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861-1870. PMLR, 2018.  
Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.  
Eric Heiden, David Millard, Erwin Coumans, Yizhou Sheng, and Gaurav S Sukhatme. NeuralSim: Augmenting differentiable simulators with neural networks. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2021. URL https://github.com/google-research/tiny-differentiable-simulator.  
Yuanming Hu, Jiancheng Liu, Andrew Spielberg, Joshua B. Tenenbaum, William T. Freeman, Ji-jun Wu, Daniela Rus, and Wojciech Matusik. Chainqueen: A real-time differentiable physical simulator for soft robotics, 2018.  
Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Frédo Durand. Difftaichi: Differentiable programming for physical simulation. *ICLR*, 2020.

Zhiao Huang, Yuanming Hu, Tao Du, Siyuan Zhou, Hao Su, Joshua B. Tenenbaum, and Chuang Gan. Plasticinelab: A soft-body manipulation benchmark with differentiable physics. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=xCcdBRQEDW.  
Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics, 4(26), 2019.  
Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. Advances in Neural Information Processing Systems, 32:12519-12530, 2019.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Vijay R Konda and John N Tsitsiklis. Actor-critic algorithms. In Advances in neural information processing systems, pp. 1008-1014, 2000.  
Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble trust-region policy optimization. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=SJJinbWRZ.  
Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. Science robotics, 5(47), 2020.  
Seunghwan Lee, Moonseok Park, Kyoungmin Lee, and Jehee Lee. Scalable muscle-actuated human simulation and control. ACM Trans. Graph., 38(4), July 2019. ISSN 0730-0301. doi: 10.1145/3306346.3322972. URL https://doi.org/10.1145/3306346.3322972.  
Jacky Liang, Viktor Makoviychuk, Ankur Handa, Nuttapong Chentanez, Miles Macklin, and Dieter Fox. GPU-accelerated robotic simulation for distributed reinforcement learning, 2018.  
Junbang Liang, Ming C. Lin, and Vladlen Koltun. Differentiable cloth simulation for inverse problems. 2019.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In ICLR (Poster), 2016.  
Libin Liu and Jessica Hodgins. Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning. ACM Trans. Graph., 37(4), July 2018. ISSN 0730-0301. doi: 10.1145/3197517.3201315. URL https://doi.org/10.1145/3197517.3201315.  
Denys Makoviichuk and Viktor Makoviychuk. RL Games, 2021. URL https://github.com/ Denys88/rlGames/.  
Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning. CoRR, 2021. URL https://arxiv.org/abs/2108.10470.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529-533, 2015.  
Volodymyr Mniih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 1928-1937, New York, New York, USA, 20-22 Jun 2016. PMLR. URL https://proceedings.mlr.press/v48/mniha16.html.

Miguel Angel Zamora Mora, Momchil Peychev, Sehoon Ha, Martin Vechev, and Stelian Coros. Pods: Policy optimization via differentiable simulation. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 7805-7817. PMLR, 18-24 Jul 2021. URL http://proceedings.mlr.press/v139/mora21a.html.  
Michael Mozer. A focused backpropagation algorithm for temporal pattern recognition. Complex Systems, 3, 01 1995.  
OpenAI, Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, Jonas Schneider, Nikolas Tezak, Jerry Tworek, Peter Welinder, Lilian Weng, Qiming Yuan, Wojciech Zaremba, and Lei Zhang. Solving rubik's cube with a robot hand, 2019.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. 2019.  
Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. ACM Trans. Graph., 37 (4):143:1-143:14, July 2018. ISSN 0730-0301. doi: 10.1145/3197517.3201311. URL http://doi.acm.org/10.1145/3197517.3201311.  
Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. ACM Trans. Graph., 40 (4), July 2021. doi: 10.1145/3450626.3459670. URL http://doi.acm.org/10.1145/3450626.3459670.  
Yi-Ling Qiao, Junbang Liang, Vladlen Koltun, and Ming C. Lin. Efficient differentiable simulation of articulated bodies. In ICML, 2021.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889-1897. PMLR, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Richard S Sutton, Andrew G Barto, et al. Introduction to reinforcement learning, volume 135. MIT press Cambridge, 1998.  
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026-5033. IEEE, 2012.  
Keenon Werling, Dalton Omens, Jeongseok Lee, Ioannis Exarchos, and C. Karen Liu. Fast and Feature-Complete Differentiable Physics Engine for Articulated Rigid Bodies with Contact Constraints. In Proceedings of Robotics: Science and Systems, Virtual, July 2021. doi: 10.15607/RSS.2021.XVII.034.  
Jie Xu, Tao Chen, Lara Zlokapa, Michael Foshey, Wojciech Matusik, Shinjiro Sueda, and Pulkit Agrawal. An End-to-End Differentiable Framework for Contact-Aware Robot Design. In Proceedings of Robotics: Science and Systems, Virtual, July 2021. doi: 10.15607/RSS.2021.XVII.008.
